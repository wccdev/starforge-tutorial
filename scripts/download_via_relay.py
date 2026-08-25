#!/usr/bin/env python3
"""在【不能上外网、只能 ssh 到中继机】的推理机上直接下载模型。

场景：A 机跑推理，没有外网，也访问不了别的机器，唯独能 root 免密 ssh 到 B 机；B 机有外网。
做法：A 上跑本脚本，每个文件都通过 `ssh B curl ...` 把字节流管道回 A 直接落盘。
B 机全程不落盘、不需要有模型目录，也不用装 python，只要有 curl。

产出直接就是 HF 缓存布局，跑完 export HF_HOME=<目标目录> 就能按 repo id 加载：
    <HF_HOME>/hub/models--Qwen--Qwen3.6-27B/
        refs/main            # commit sha，结尾无换行
        snapshots/<sha>/     # 模型文件

只用标准库，可以单独 scp 到 A 上用系统 python3 跑（huggingface_hub 只在最后做可加载性
自检时用，没有也不影响下载）。

用法：
    python3 download_via_relay.py --relay root@10.0.0.2 --check      # 先验通中继链路
    python3 download_via_relay.py --relay root@10.0.0.2 --list
    python3 download_via_relay.py --relay root@10.0.0.2 --daemon
    tail -f hf_cache/relay_download.log

断点续传：未完成的文件存成 <文件>.part，重跑同一命令用 HTTP Range 从断点接着下。
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "qwen3.6-27b": {"hf": "Qwen/Qwen3.6-27B", "modelscope": "Qwen/Qwen3.6-27B"},
    "qwen3.6-35b-a3b": {"hf": "Qwen/Qwen3.6-35B-A3B", "modelscope": "Qwen/Qwen3.6-35B-A3B"},
    # 两边命名空间不同：HF 是 zai-org，魔搭是 ZhipuAI。缓存目录名一律按 HF 的来。
    "glm-4.7-flash": {"hf": "zai-org/GLM-4.7-Flash", "modelscope": "ZhipuAI/GLM-4.7-Flash"},
}

HF_ENDPOINT_DEFAULT = "https://huggingface.co"
MS_ENDPOINT = "https://www.modelscope.cn"

DEFAULT_IGNORE = ["*.gguf", "*.msgpack", "*.h5", "*.onnx", "*.onnx_data", "original/*", "consolidated*"]

MARKER = ".starforge_download_complete"

_LOG_LOCK = threading.Lock()
_LOG_FH = None


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with _LOG_LOCK:
        print(line, flush=True)
        if _LOG_FH is not None:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()


def human(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PiB"


def dir_size(path: Path) -> int:
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat().st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def is_ignored(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


class RelayError(RuntimeError):
    pass


class HttpError(RelayError):
    def __init__(self, status: str, url: str):
        self.status = status
        self.url = url
        hint = {
            "401": "仓库不存在、是私有的，或需要 --token",
            "403": "无权访问（gated 仓库需要先在网页上同意协议，再用 --token）",
            "404": "仓库或 revision 不存在",
        }.get(status, "")
        super().__init__(f"HTTP {status}{'（' + hint + '）' if hint else ''}：{url}")


class Relay:
    """通过 ssh 在中继机上执行 curl，把 HTTP 响应体作为 stdout 管道回本机。"""

    def __init__(self, ssh_argv: list[str], token: str | None, connect_timeout: int):
        self.ssh_argv = ssh_argv
        self.token = token
        self.connect_timeout = connect_timeout

    def _curl_config(self) -> str | None:
        """token 通过 curl 的 stdin 配置传，避免出现在中继机的进程列表里。"""
        if not self.token:
            return None
        return f'header = "Authorization: Bearer {self.token}"\n'

    def _auth_flag(self) -> str:
        return "-K -" if self.token else ""

    def run_text(self, remote_cmd: str, timeout: int) -> bytes:
        """执行远程命令并收集全部 stdout（用于 API 这类小响应）。"""
        proc = subprocess.run(
            [*self.ssh_argv, remote_cmd],
            input=(self._curl_config() or "").encode(),
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace").strip()
            raise RelayError(f"中继命令失败（exit {proc.returncode}）：{err[:300]}")
        return proc.stdout

    def get_json(self, url: str, timeout: int = 90) -> dict:
        """取 JSON。用 -w 把状态码附在响应体末尾，好在出错时给出人话报错。"""
        cmd = (
            f"curl -sSL -w '\\n%{{http_code}}' --max-time {timeout - 10} "
            f"{self._auth_flag()} {shlex.quote(url)}"
        )
        raw = self.run_text(cmd, timeout)
        body, _, code = raw.rpartition(b"\n")
        status = code.decode(errors="replace").strip()
        if status != "200":
            raise HttpError(status, url)
        try:
            return json.loads(body)
        except ValueError as exc:
            raise RelayError(f"中继返回的不是 JSON（HTTP {status}，前 200 字节：{body[:200]!r}）") from exc

    def http_code(self, url: str, extra: str = "", timeout: int = 60) -> str:
        cmd = (
            f"curl -sSL -o /dev/null -w '%{{http_code}}' --max-time {timeout - 10} "
            f"{self._auth_flag()} {extra} {shlex.quote(url)}"
        )
        return self.run_text(cmd, timeout).decode(errors="replace").strip()

    def supports_range(self, url: str) -> bool:
        return self.http_code(url, extra="-r 0-0") == "206"

    def stream_to(self, url: str, start: int, sink, stall_seconds: int, chunk: int = 4 << 20) -> int:
        """把 url 从 start 字节处开始的内容写进 sink，返回写入字节数。

        --speed-limit/--speed-time 让 curl 在长时间几乎不动时自己退出，
        避免链路半死不活时整个任务无限期挂着。
        """
        rng = f"-r {start}-" if start else ""
        cmd = (
            f"curl -sSL --fail {self._auth_flag()} {rng} "
            f"--speed-limit 1024 --speed-time {stall_seconds} {shlex.quote(url)}"
        )
        proc = subprocess.Popen(
            [*self.ssh_argv, cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        written = 0
        try:
            cfg = self._curl_config()
            if cfg:
                proc.stdin.write(cfg.encode())
            proc.stdin.close()
            while True:
                block = proc.stdout.read(chunk)
                if not block:
                    break
                sink.write(block)
                written += len(block)
        finally:
            proc.stdout.close()
            stderr = proc.stderr.read().decode(errors="replace").strip()
            proc.stderr.close()
            code = proc.wait()
        if code != 0:
            raise RelayError(f"传输中断（exit {code}）：{stderr[:300]}")
        return written


def build_ssh_argv(relay: str | None, override: str | None, connect_timeout: int, ssh_opts: list[str]) -> list[str]:
    if override:
        return shlex.split(override)
    if not relay:
        raise SystemExit("必须指定 --relay <user@中继机>，或用 --ssh-argv 自定义执行方式")
    argv = [
        "ssh",
        "-T",  # 不分配 tty，保证 stdout 是干净的二进制流
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
    ]
    for opt in ssh_opts:
        argv += ["-o", opt]
    argv.append(relay)
    return argv


# ---------- 各来源的元数据 ----------


def hf_meta(relay: Relay, repo_id: str, revision: str, endpoint: str, ignore: list[str]) -> tuple[str, list[tuple[str, int, str | None]]]:
    """返回 (commit sha, [(相对路径, 字节数, sha256)])。sha256 只有 LFS 大文件才有。"""
    data = relay.get_json(f"{endpoint}/api/models/{repo_id}/revision/{revision}?blobs=true")
    sha = data.get("sha")
    if not sha:
        raise RelayError(f"{repo_id} 的响应里没有 sha")
    files = []
    for sib in data.get("siblings") or []:
        name = sib.get("rfilename")
        if not name or is_ignored(name, ignore):
            continue
        lfs = sib.get("lfs") or {}
        digest = lfs.get("sha256") if isinstance(lfs, dict) else None
        files.append((name, int(sib.get("size") or 0), digest))
    return sha, files


def ms_meta(relay: Relay, repo_id: str, revision: str, ignore: list[str]) -> list[tuple[str, int, str | None]]:
    data = relay.get_json(f"{MS_ENDPOINT}/api/v1/models/{repo_id}/repo/files?Revision={revision}&Recursive=True")
    files = []
    for item in (data.get("Data") or {}).get("Files") or []:
        if item.get("Type") != "blob":
            continue
        path = item.get("Path") or item.get("Name")
        if not path or is_ignored(path, ignore):
            continue
        files.append((path, int(item.get("Size") or 0), item.get("Sha256")))
    return files


def content_sha(repo_id: str, files: list[tuple[str, int, str | None]]) -> str:
    """按文件清单算一个稳定的 40 位摘要，给 HF 上不存在的模型当 snapshots 目录名。

    内容变了摘要就变，会落到新的 snapshot 目录，不会和旧版本混在一起。
    """
    digest = hashlib.sha1()
    digest.update(repo_id.encode())
    for rel, size, file_digest in sorted(files):
        digest.update(f"\n{rel}\t{size}\t{file_digest or ''}".encode())
    return digest.hexdigest()


def resolve_hf_sha(
    relay: Relay,
    hf_repo_id: str,
    endpoint: str,
    ignore: list[str],
    files: list[tuple[str, int, str | None]],
    fake_sha: bool,
    short_name: str,
) -> str:
    try:
        sha, _ = hf_meta(relay, hf_repo_id, "main", endpoint, ignore)
        return sha
    except HttpError as exc:
        if not fake_sha:
            raise RelayError(
                f"查不到 {hf_repo_id} 在 HF 上的 commit sha（{exc}）。"
                f"若两边命名空间不同，用 --only 下载源id=HF id 显式指定；"
                f"若该模型 HF 上根本没有，加 --fake-sha 用内容摘要代替目录名，"
                f"并在推理机上设 HF_HUB_OFFLINE=1"
            ) from exc
        sha = content_sha(hf_repo_id, files)
        log(f"[{short_name}] HF 上查不到该仓库，snapshots 目录改用内容摘要 {sha[:12]}（加载时需 HF_HUB_OFFLINE=1）")
        return sha


def file_url(source: str, repo_id: str, revision: str, rel_path: str, endpoint: str) -> str:
    if source == "modelscope":
        return f"{MS_ENDPOINT}/api/v1/models/{repo_id}/repo?Revision={revision}&FilePath={rel_path}"
    return f"{endpoint}/{repo_id}/resolve/{revision}/{rel_path}"


# ---------- 下载 ----------


def fetch_one(
    relay: Relay,
    url: str,
    rel_path: str,
    expected: int,
    snapshot: Path,
    stall_seconds: int,
) -> None:
    """下单个文件。先写 .part，完整后才改名，避免半个文件冒充完整文件。"""
    final = snapshot / rel_path
    part = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)

    if expected and final.is_file() and final.stat().st_size == expected:
        return

    have = part.stat().st_size if part.is_file() else 0
    if expected and have > expected:  # 上次写坏了
        part.unlink()
        have = 0

    if have:
        if relay.supports_range(url):
            log(f"    续传 {rel_path}：从 {human(have)} 处接着下")
        else:
            # 服务端不认 Range，续传会把整文件追加到残片后面，只能从头来。
            log(f"    {rel_path}：服务端不支持 Range，丢弃 {human(have)} 残片重下")
            part.unlink()
            have = 0

    mode = "ab" if have else "wb"
    with open(part, mode) as fh:
        relay.stream_to(url, have, fh, stall_seconds)

    size = part.stat().st_size
    if expected and size != expected:
        raise RelayError(f"{rel_path} 大小不符：落盘 {size}，应为 {expected}")
    part.replace(final)


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def start_progress_monitor(target: Path, expected: int, stop: threading.Event, interval: int) -> None:
    def run() -> None:
        last, last_at = dir_size(target), time.monotonic()
        while not stop.wait(interval):
            now, now_at = dir_size(target), time.monotonic()
            rate = (now - last) / max(now_at - last_at, 1e-6)
            pct = f"{now / expected * 100:5.1f}%" if expected else "  ?  "
            eta = ""
            if expected and rate > 0:
                eta = f"，预计剩余 {max(expected - now, 0) / rate / 60:.0f} 分钟"
            log(f"    进度 {pct}  已落盘 {human(now)}  速度 {human(rate)}/s{eta}")
            last, last_at = now, now_at

    threading.Thread(target=run, daemon=True).start()


def download_model(
    relay: Relay,
    short_name: str,
    repo_id: str,
    hf_repo_id: str,
    hub_root: Path,
    *,
    source: str,
    revision: str,
    endpoint: str,
    ignore: list[str],
    workers: int,
    retries: int,
    stall_seconds: int,
    progress_interval: int,
    verify: bool,
    force: bool,
    fake_sha: bool,
) -> bool:
    repo_dir = hub_root / f"models--{hf_repo_id.replace('/', '--')}"
    marker_path = repo_dir / MARKER
    if marker_path.is_file() and not force:
        try:
            info = json.loads(marker_path.read_text(encoding="utf-8"))
            log(f"[{short_name}] 已完成（{info.get('completed_at')}，{human(info.get('bytes', 0))}），跳过")
            return True
        except ValueError:
            pass

    # snapshots 目录名必须是 HF 的 commit sha；魔搭源要单独向 HF 问一次。
    if source == "modelscope":
        files = ms_meta(relay, repo_id, revision, ignore)
        sha = resolve_hf_sha(relay, hf_repo_id, endpoint, ignore, files, fake_sha, short_name)
    else:
        sha, files = hf_meta(relay, repo_id, revision, endpoint, ignore)
    if not files:
        raise RelayError(f"{repo_id} 过滤后没有文件可下（检查 --revision，或加 --all-files）")

    total = sum(f[1] for f in files)
    snapshot = repo_dir / "snapshots" / sha
    snapshot.mkdir(parents=True, exist_ok=True)
    log(f"[{short_name}] {source}:{repo_id}@{sha[:12]} — {len(files)} 个文件，{human(total)}")

    free = shutil.disk_usage(hub_root).free
    need = max(total - dir_size(repo_dir), 0)
    if free < need * 1.05:
        log(f"[{short_name}] 磁盘不足：可用 {human(free)}，还需约 {human(need)}。中止。")
        return False

    stop = threading.Event()
    if progress_interval > 0:
        start_progress_monitor(repo_dir, total, stop, progress_interval)

    try:
        for attempt in range(1, retries + 1):
            pending = [
                f for f in files if not (snapshot / f[0]).is_file() or (snapshot / f[0]).stat().st_size != f[1]
            ]
            if not pending:
                break
            log(f"[{short_name}] 第 {attempt}/{retries} 轮：待下 {len(pending)} 个文件")
            errors: list[str] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        fetch_one,
                        relay,
                        file_url(source, repo_id, revision, rel, endpoint),
                        rel,
                        size,
                        snapshot,
                        stall_seconds,
                    ): rel
                    for rel, size, _ in pending
                }
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        errors.append(f"{futures[fut]}: {type(exc).__name__}: {exc}")
            if not errors:
                break
            log(f"[{short_name}] 本轮 {len(errors)} 个文件失败，例如 {errors[0][:200]}")
            if attempt >= retries:
                log(f"[{short_name}] 重试用尽，未完成（已下部分保留，重跑可续传）")
                return False
            backoff = min(2**attempt, 120)
            log(f"[{short_name}] {backoff}s 后重试")
            time.sleep(backoff)
    finally:
        stop.set()

    missing = [rel for rel, size, _ in files if not (snapshot / rel).is_file() or (snapshot / rel).stat().st_size != size]
    if missing:
        log(f"[{short_name}] 仍缺 {len(missing)} 个文件，例如 {missing[:3]}")
        return False

    if verify:
        checkable = [(rel, digest) for rel, _, digest in files if digest]
        log(f"[{short_name}] sha256 校验 {len(checkable)} 个文件（大文件较慢）")
        bad = []
        with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
            results = pool.map(lambda it: (it[0], sha256_of(snapshot / it[0]) == it[1]), checkable)
            for rel, ok in results:
                if not ok:
                    bad.append(rel)
        if bad:
            log(f"[{short_name}] 校验不通过：{bad[:5]}（共 {len(bad)}）。删掉这些文件后重跑。")
            return False
        log(f"[{short_name}] sha256 全部通过")

    # refs 内容必须是纯 sha，多一个换行 hub 就匹配不到 snapshots 目录。
    refs = repo_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(sha, encoding="utf-8")

    actual = dir_size(repo_dir)
    marker_path.write_text(
        json.dumps(
            {
                "source": source,
                "via": "ssh-relay",
                "repo_id": repo_id,
                "hf_repo_id": hf_repo_id,
                "sha": sha,
                "sha_is_content_digest": source == "modelscope" and fake_sha,
                "files": len(files),
                "bytes": actual,
                "completed_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"[{short_name}] 完成，{human(actual)} -> {snapshot}")
    return True


# ---------- 自检 ----------


def preflight(relay: Relay, endpoint: str, source: str, probe_repo: str) -> bool:
    ok = True
    try:
        out = relay.run_text("echo relay-ok && uname -s", timeout=40).decode(errors="replace")
        log(f"  ssh 连通：{' '.join(out.split())}")
    except Exception as exc:
        log(f"  ssh 不通：{exc}")
        return False
    try:
        out = relay.run_text("command -v curl || echo MISSING", timeout=40).decode().strip()
        if out == "MISSING":
            log("  中继机上没有 curl，装一个再来（yum install curl / apt install curl）")
            ok = False
        else:
            log(f"  中继机 curl：{out}")
    except Exception as exc:
        log(f"  检查 curl 失败：{exc}")
        return False
    base = MS_ENDPOINT if source == "modelscope" else endpoint
    api = (
        f"{base}/api/v1/models/{probe_repo}"
        if source == "modelscope"
        else f"{base}/api/models/{probe_repo}"
    )
    try:
        code = relay.http_code(api)
        log(f"  中继机访问 {base}：HTTP {code}")
        if code != "200":
            log("  中继机连不上该站点（或需要代理），换 --source / --endpoint 试试")
            ok = False
    except Exception as exc:
        log(f"  中继机访问外网失败：{exc}")
        ok = False
    return ok


def check_loadable(hf_home: Path, repo_ids: list[str]) -> None:
    code = (
        "import sys;from huggingface_hub import snapshot_download;"
        "print(snapshot_download(sys.argv[1], cache_dir=sys.argv[2], local_files_only=True))"
    )
    for repo_id in repo_ids:
        proc = subprocess.run(
            [sys.executable, "-c", code, repo_id, str(hf_home / "hub")],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            log(f"  OK   {repo_id} -> {proc.stdout.strip()}")
        else:
            tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "未知错误"
            log(f"  跳过/失败 {repo_id}：{tail}")


# ---------- CLI ----------


def spawn_background(hf_home: Path, log_path: Path) -> int:
    hf_home.mkdir(parents=True, exist_ok=True)
    argv = [a for a in sys.argv[1:] if a != "--daemon"]
    with open(log_path, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), *argv],
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=dict(os.environ, _RELAY_CHILD="1"),
            cwd=os.getcwd(),
        )
    (hf_home / "relay_download.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="经 ssh 中继机下载模型到本机 HF 缓存（无外网的推理机专用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--relay", default=os.environ.get("FORGE_RELAY"), help="中继机，如 root@10.0.0.2（可用环境变量 FORGE_RELAY）")
    p.add_argument("--hf-home", default="hf_cache", help="本机缓存根目录，模型落到 <HF_HOME>/hub/（默认 ./hf_cache）")
    p.add_argument("--source", choices=("hf", "modelscope"), default="hf", help="下载源（默认 hf）")
    p.add_argument("--endpoint", default=HF_ENDPOINT_DEFAULT, help=f"HF 端点（默认 {HF_ENDPOINT_DEFAULT}）")
    p.add_argument(
        "--only",
        action="append",
        metavar="MODEL",
        help=(
            f"指定模型，可重复。可以是内置短名（{', '.join(DEFAULT_MODELS)}）、"
            "任意 repo id（org/name），或两边命名不同时写 下载源id=HF id，"
            "例如 ZhipuAI/GLM-4.7-Flash=zai-org/GLM-4.7-Flash"
        ),
    )
    p.add_argument("--revision", default=None, help="revision（默认 hf=main / modelscope=master）")
    p.add_argument("--token", default=None, help="HF token（默认读 HF_TOKEN），经 curl stdin 传，不进中继机进程列表")
    p.add_argument("--workers", type=int, default=4, help="并发 ssh 通道数（默认 4）")
    p.add_argument("--retries", type=int, default=8, help="每个模型的整体重试轮数（默认 8）")
    p.add_argument("--stall-seconds", type=int, default=60, help="传输速度低于 1KB/s 持续多久判定卡死（默认 60）")
    p.add_argument("--connect-timeout", type=int, default=15, help="ssh 连接超时秒数（默认 15）")
    p.add_argument("--ssh-opt", action="append", default=[], metavar="OPT", help="附加 ssh -o 选项，可重复")
    p.add_argument("--ssh-argv", default=None, help="高级：直接指定执行命令的 argv（调试用，如 'bash -c'）")
    p.add_argument("--progress-interval", type=int, default=60, help="进度汇报间隔秒数，0 关闭（默认 60）")
    p.add_argument("--all-files", action="store_true", help="不过滤 gguf/onnx 等文件")
    p.add_argument("--verify", action="store_true", help="下完按 sha256 校验（HF 源只校验 LFS 大文件）")
    p.add_argument("--force", action="store_true", help="忽略完成标记重下")
    p.add_argument(
        "--fake-sha",
        action="store_true",
        help="魔搭源模型在 HF 上不存在时，用文件清单摘要当 snapshots 目录名（加载时需 HF_HUB_OFFLINE=1）",
    )
    p.add_argument("--check", action="store_true", help="只做中继链路自检")
    p.add_argument("--list", action="store_true", help="只列出待下载文件与体积")
    p.add_argument("--daemon", action="store_true", help="后台运行，日志写到 <HF_HOME>/relay_download.log")
    return p.parse_args(argv)


def resolve_targets(only: list[str] | None, source: str) -> dict[str, tuple[str, str]]:
    """返回 {短名: (下载用的 repo id, 缓存目录命名用的 HF repo id)}。

    --only 接受三种写法：
      短名                     内置模型，自带两边的命名映射
      org/repo                 两边同名，或只用一个源
      下载源id=HF id           两边命名空间不同时显式给出，例如
                               ZhipuAI/GLM-4.7-Flash=zai-org/GLM-4.7-Flash
    """
    if not only:
        return {n: (ids[source], ids["hf"]) for n, ids in DEFAULT_MODELS.items()}
    out: dict[str, tuple[str, str]] = {}
    for item in only:
        raw = item.strip()
        key = raw.lower()
        if key in DEFAULT_MODELS:
            out[key] = (DEFAULT_MODELS[key][source], DEFAULT_MODELS[key]["hf"])
            continue
        download_id, _, hf_id = raw.partition("=")
        download_id, hf_id = download_id.strip(), hf_id.strip()
        if "/" not in download_id or (hf_id and "/" not in hf_id):
            raise SystemExit(
                f"无法识别 {item!r}。可用短名：{', '.join(DEFAULT_MODELS)}；"
                "或给 repo id（org/name）；两边命名不同时用 下载源id=HF id"
            )
        out[download_id.split("/")[-1].lower()] = (download_id, hf_id or download_id)
    return out


def main(argv: list[str] | None = None) -> int:
    global _LOG_FH

    args = parse_args(argv)
    hf_home = Path(args.hf_home).expanduser().resolve()
    log_path = hf_home / "relay_download.log"

    if args.daemon and not os.environ.get("_RELAY_CHILD"):
        pid = spawn_background(hf_home, log_path)
        print(f"已后台启动，PID {pid}")
        print(f"日志：tail -f {log_path}")
        print(f"停止：kill {pid}（已下载部分保留，重跑本命令续传）")
        return 0

    source = args.source
    revision = args.revision or ("master" if source == "modelscope" else "main")
    ignore = [] if args.all_files else list(DEFAULT_IGNORE)
    token = args.token or os.environ.get("HF_TOKEN") or None

    try:
        targets = resolve_targets(args.only, source)
        ssh_argv = build_ssh_argv(args.relay, args.ssh_argv, args.connect_timeout, args.ssh_opt)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    hub_root = hf_home / "hub"
    hub_root.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("_RELAY_CHILD"):
        _LOG_FH = open(log_path, "a", encoding="utf-8")

    relay = Relay(ssh_argv, token, args.connect_timeout)

    if args.check:
        log(f"中继自检：{' '.join(ssh_argv)}")
        first = next(iter(targets.values()))[0]
        return 0 if preflight(relay, args.endpoint, source, first) else 1

    if args.list:
        grand = 0
        for short_name, (repo_id, _) in targets.items():
            try:
                if source == "modelscope":
                    files = ms_meta(relay, repo_id, revision, ignore)
                else:
                    _, files = hf_meta(relay, repo_id, revision, args.endpoint, ignore)
            except Exception as exc:
                log(f"{short_name:20s} 读取失败：{type(exc).__name__}: {exc}")
                continue
            size = sum(f[1] for f in files)
            grand += size
            log(f"{short_name:20s} {repo_id:28s} {len(files):4d} 个文件  {human(size)}")
        log(f"{'合计':20s} {'':28s} {'':4s}    {human(grand)}")
        log(f"落盘到 {hub_root}，可用空间 {human(shutil.disk_usage(hub_root).free)}")
        return 0

    log("=" * 72)
    log(f"经中继下载 {len(targets)} 个模型 -> {hub_root}")
    log(f"中继：{' '.join(ssh_argv)}   源：{source}   并发：{args.workers}")
    if not preflight(relay, args.endpoint, source, next(iter(targets.values()))[0]):
        log("自检未通过，中止。")
        return 1

    started = time.monotonic()
    done: list[str] = []
    failed: list[str] = []
    for short_name, (repo_id, hf_repo_id) in targets.items():
        try:
            ok = download_model(
                relay,
                short_name,
                repo_id,
                hf_repo_id,
                hub_root,
                source=source,
                revision=revision,
                endpoint=args.endpoint,
                ignore=ignore,
                workers=args.workers,
                retries=args.retries,
                stall_seconds=args.stall_seconds,
                progress_interval=args.progress_interval,
                verify=args.verify,
                force=args.force,
                fake_sha=args.fake_sha,
            )
        except Exception as exc:
            log(f"[{short_name}] 异常：{type(exc).__name__}: {exc}")
            ok = False
        (done if ok else failed).append(hf_repo_id if ok else short_name)

    log(f"结束，用时 {(time.monotonic() - started) / 60:.1f} 分钟；成功 {len(done)}/{len(targets)}")
    if done:
        log("离线可加载性自检：")
        check_loadable(hf_home, done)
    if failed:
        log(f"失败：{', '.join(failed)}（重跑同一命令即可续传）")
        return 1
    log(f"完成。推理时 export HF_HOME={hf_home}，按 repo id 加载即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
