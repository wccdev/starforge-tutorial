#!/usr/bin/env python3
"""在【下载机】上把模型下成 HF 缓存目录结构，整个目录拷到算力机就能直接用。

产出就是标准的 HF 缓存布局：
    <HF_HOME>/hub/models--Qwen--Qwen3.6-27B/
        refs/main            # commit sha，结尾无换行
        snapshots/<sha>/     # 模型文件
        blobs/               # 仅 --source hf 有；魔搭源直接在 snapshots 下放实体文件

算力机上 export HF_HOME=<拷过去的目录> 之后，config 里继续写
`policy.model_name: "Qwen/Qwen3.6-27B"` 即可，不用改成绝对路径。

两个下载源，用 --source 切换：
  hf（默认）    huggingface_hub 原生下载，断点信息在 blobs/*.incomplete。
                国内可加 --endpoint https://hf-mirror.com，但镜像对境外 IP 会 308 回官方站。
  modelscope    魔搭 REST + HTTP Range 续传，国内带宽通常好得多；三个目标模型的字节数
                与 HF 完全一致（已核对）。需要联网向 HF 查一次 commit sha 来命名 snapshots
                目录（只是几 KB 的元数据请求），连不上就加 --fake-sha 并在算力机设 HF_HUB_OFFLINE=1。

用法：
    uv run python scripts/download_models.py --list
    uv run python scripts/download_models.py --daemon
    uv run python scripts/download_models.py --source modelscope --daemon
    tail -f hf_cache/download.log

拷到算力机（务必 -a 保留符号链接，别用 -L 或 scp -r，那会把 blobs 展开成双份）：
    rsync -avP --exclude '*.log' --exclude '*.pid' hf_cache/ user@算力机:/data/hf_cache/
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# 目标模型：短名 -> 各源 repo id。缓存目录名一律按 HF 的 repo id 拼，
# 因为算力机上 transformers 是按 HF repo id 去找的。
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "qwen3.6-27b": {"hf": "Qwen/Qwen3.6-27B", "modelscope": "Qwen/Qwen3.6-27B"},
    "qwen3.6-35b-a3b": {"hf": "Qwen/Qwen3.6-35B-A3B", "modelscope": "Qwen/Qwen3.6-35B-A3B"},
    # 两边命名空间不同：HF 是 zai-org，魔搭是 ZhipuAI。
    "glm-4.7-flash": {"hf": "zai-org/GLM-4.7-Flash", "modelscope": "ZhipuAI/GLM-4.7-Flash"},
}

MS_API = "https://www.modelscope.cn/api/v1/models"
DEFAULT_REVISION = {"hf": "main", "modelscope": "master"}

# 微调用不到的产物（量化版、TF/ONNX 权重等），跳过可省大量带宽。--all-files 可关闭。
DEFAULT_IGNORE = [
    "*.gguf",
    "*.msgpack",
    "*.h5",
    "*.onnx",
    "*.onnx_data",
    "original/*",
    "consolidated*",
]

# 完成标记放在 models--*/ 下而不是 snapshots/ 里，免得污染模型目录。
MARKER = ".starforge_download_complete"

# 魔搭文件条目：(相对路径, 字节数, sha256)
MsFile = tuple[str, int, "str | None"]

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
    """递归统计目录字节数。符号链接按链接本身算（不重复计 blobs）。"""
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
        except (OSError, FileNotFoundError):
            continue
    return total


def is_ignored(filename: str, ignore_patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(filename, pat) for pat in ignore_patterns)


def repo_cache_dir(hub_root: Path, hf_repo_id: str) -> Path:
    return hub_root / f"models--{hf_repo_id.replace('/', '--')}"


def hf_repo_size(repo_id: str, revision: str, ignore_patterns: list[str], token: str | None) -> tuple[int, int, str | None]:
    """返回 (需下载字节数, 文件数, commit sha)。拿不到大小时返回 0，交给磁盘检查兜底。"""
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(repo_id=repo_id, revision=revision, files_metadata=True)
    total = 0
    count = 0
    for sibling in info.siblings or []:
        if is_ignored(sibling.rfilename, ignore_patterns):
            continue
        count += 1
        total += getattr(sibling, "size", None) or 0
    return total, count, getattr(info, "sha", None)


def hf_commit_sha(repo_id: str, revision: str, token: str | None) -> str | None:
    """魔搭不提供 HF 的 commit sha，但缓存目录名必须用它，所以单独查一次元数据。"""
    try:
        from huggingface_hub import HfApi

        return HfApi(token=token).model_info(repo_id=repo_id, revision=revision).sha
    except Exception as exc:
        log(f"    查 {repo_id} 的 commit sha 失败：{type(exc).__name__}: {exc}")
        return None


def ms_list_files(repo_id: str, revision: str, ignore_patterns: list[str]) -> list[MsFile]:
    """列出魔搭仓库里要下的 (相对路径, 字节数, sha256)。目录条目（Type=tree）不算。"""
    import requests

    resp = requests.get(
        f"{MS_API}/{repo_id}/repo/files",
        params={"Revision": revision, "Recursive": "True"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("Code") not in (200, None):
        raise RuntimeError(f"魔搭返回 Code={payload.get('Code')}: {payload.get('Message')}")
    files = []
    for item in payload.get("Data", {}).get("Files", []):
        if item.get("Type") != "blob":
            continue
        path = item.get("Path") or item.get("Name")
        if not path or is_ignored(path, ignore_patterns):
            continue
        files.append((path, int(item.get("Size") or 0), item.get("Sha256")))
    return files


def ms_fetch_file(repo_id: str, revision: str, rel_path: str, expected: int, snapshot: Path, chunk: int = 8 << 20) -> None:
    """下载单个魔搭文件，用 HTTP Range 从已落盘字节处续传（实测该端点返回 206）。

    先写 <file>.part，完成后才改名——否则中断留下的残片会被当成完整的 snapshot 文件。
    """
    import requests

    final = snapshot / rel_path
    part = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)

    if expected and final.exists() and final.stat().st_size == expected:
        return

    have = part.stat().st_size if part.exists() else 0
    if expected and have > expected:  # 上次写坏了，重来
        part.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(
        f"{MS_API}/{repo_id}/repo",
        params={"Revision": revision, "FilePath": rel_path},
        headers=headers,
        stream=True,
        timeout=(30, 300),
    ) as resp:
        resp.raise_for_status()
        # 服务端忽略 Range 就得从头写，否则会把整文件追加到残片后面。
        mode = "ab" if (have and resp.status_code == 206) else "wb"
        with open(part, mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if block:
                    fh.write(block)

    if expected and part.stat().st_size != expected:
        raise IOError(f"{rel_path} 大小不符：落盘 {part.stat().st_size}，应为 {expected}")
    part.replace(final)


def ms_download(repo_id: str, revision: str, snapshot: Path, files: list[MsFile], max_workers: int) -> None:
    """并发下载整个魔搭仓库；任一文件失败就抛出，交给外层重试（已下部分保留）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending = [
        (p, s) for p, s, _ in files if not (snapshot / p).exists() or (snapshot / p).stat().st_size != s
    ]
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(ms_fetch_file, repo_id, revision, p, s, snapshot): p for p, s in pending}
        try:
            for fut in as_completed(futures):
                fut.result()
        except Exception:
            for fut in futures:
                fut.cancel()
            raise


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def ms_verify(snapshot: Path, files: list[MsFile], max_workers: int) -> list[str]:
    """按魔搭给的 sha256 校验落盘文件，返回对不上的路径。大模型下很慢，仅 --verify 时调用。"""
    from concurrent.futures import ThreadPoolExecutor

    checkable = [(p, sha) for p, _, sha in files if sha and (snapshot / p).is_file()]
    bad: list[str] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, 4)) as pool:
        results = pool.map(lambda item: (item[0], sha256_of(snapshot / item[0]) == item[1]), checkable)
        for rel_path, ok in results:
            if not ok:
                bad.append(rel_path)
    return bad


def read_marker(repo_dir: Path) -> dict | None:
    marker = repo_dir / MARKER
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_refs(repo_dir: Path, revision: str, sha: str) -> None:
    """refs/<revision> 的内容必须是纯 sha，结尾多一个换行 hub 就匹配不到 snapshots 目录。"""
    refs = repo_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / revision).write_text(sha, encoding="utf-8")


def start_progress_monitor(target: Path, expected: int, stop: threading.Event, interval: int) -> threading.Thread:
    """后台线程按固定间隔汇报落盘进度——后台运行时 tqdm 进度条不可读，用这个替代。"""

    def run() -> None:
        last_size = dir_size(target)
        last_at = time.monotonic()
        while not stop.wait(interval):
            now_size = dir_size(target)
            now_at = time.monotonic()
            elapsed = max(now_at - last_at, 1e-6)
            rate = (now_size - last_size) / elapsed
            pct = f"{now_size / expected * 100:5.1f}%" if expected else "  ?  "
            eta = ""
            if expected and rate > 0:
                remain = max(expected - now_size, 0) / rate
                eta = f"，预计剩余 {remain / 60:.0f} 分钟"
            log(f"    进度 {pct}  已落盘 {human(now_size)}  速度 {human(rate)}/s{eta}")
            last_size, last_at = now_size, now_at

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def download_one(
    short_name: str,
    repo_id: str,
    hf_repo_id: str,
    hub_root: Path,
    *,
    source: str,
    revision: str,
    token: str | None,
    ignore_patterns: list[str],
    max_workers: int,
    retries: int,
    progress_interval: int,
    force: bool,
    verify: bool,
    fake_sha: bool,
) -> bool:
    repo_dir = repo_cache_dir(hub_root, hf_repo_id)
    marker = read_marker(repo_dir)
    if marker and not force:
        log(f"[{short_name}] 已完成（{marker.get('completed_at')}，{human(marker.get('bytes', 0))}），跳过")
        return True

    ms_files: list[MsFile] = []
    sha: str | None = None
    try:
        if source == "modelscope":
            ms_files = ms_list_files(repo_id, revision, ignore_patterns)
            expected_bytes, file_count = sum(f[1] for f in ms_files), len(ms_files)
        else:
            expected_bytes, file_count, sha = hf_repo_size(repo_id, revision, ignore_patterns, token)
    except Exception as exc:  # 元数据拿不到不该阻断下载，只是没法做体积预检
        log(f"[{short_name}] 读取仓库元数据失败（{type(exc).__name__}: {exc}），跳过体积预检")
        expected_bytes, file_count = 0, 0

    log(f"[{short_name}] {source}:{repo_id} — {file_count or '?'} 个文件，约 {human(expected_bytes)}")
    if expected_bytes:
        free = shutil.disk_usage(hub_root).free
        already = dir_size(repo_dir) if repo_dir.exists() else 0
        need = max(expected_bytes - already, 0)
        if free < need * 1.05:
            log(f"[{short_name}] 磁盘不足：可用 {human(free)}，还需约 {human(need)}。中止。")
            return False

    # 魔搭源要自己拼缓存目录，snapshots 的目录名必须是 HF 上的真实 commit sha。
    snapshot: Path | None = None
    if source == "modelscope":
        sha = hf_commit_sha(hf_repo_id, DEFAULT_REVISION["hf"], token)
        if not sha and fake_sha:
            sha = hashlib.sha1(hf_repo_id.encode()).hexdigest()
            log(f"[{short_name}] 用占位 sha {sha[:12]}；算力机上必须 export HF_HUB_OFFLINE=1，否则会整个重下")
        if not sha:
            log(f"[{short_name}] 连不上 huggingface.co 拿不到 commit sha。加 --fake-sha 可继续（见 --help）。")
            return False
        snapshot = repo_dir / "snapshots" / sha
        snapshot.mkdir(parents=True, exist_ok=True)

    repo_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    if progress_interval > 0:
        start_progress_monitor(repo_dir, expected_bytes, stop, progress_interval)

    try:
        for attempt in range(1, retries + 1):
            try:
                if source == "modelscope":
                    if not ms_files:
                        ms_files = ms_list_files(repo_id, revision, ignore_patterns)
                    ms_download(repo_id, revision, snapshot, ms_files, max_workers)
                    write_refs(repo_dir, DEFAULT_REVISION["hf"], sha)
                else:
                    from huggingface_hub import snapshot_download

                    # cache_dir 指向 hub 根，产出就是原生的 blobs + snapshots 符号链接布局。
                    snapshot = Path(
                        snapshot_download(
                            repo_id=repo_id,
                            revision=revision,
                            cache_dir=str(hub_root),
                            token=token,
                            ignore_patterns=ignore_patterns or None,
                            max_workers=max_workers,
                        )
                    )
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if attempt >= retries:
                    log(f"[{short_name}] 第 {attempt} 次仍失败：{type(exc).__name__}: {exc}")
                    return False
                backoff = min(2**attempt, 120)
                log(f"[{short_name}] 第 {attempt} 次失败（{type(exc).__name__}: {exc}），{backoff}s 后续传重试")
                time.sleep(backoff)
    finally:
        stop.set()

    actual = dir_size(repo_dir)
    if expected_bytes and actual < expected_bytes * 0.98:
        log(f"[{short_name}] 落盘 {human(actual)} 小于预期 {human(expected_bytes)}，判定未完成，请重跑续传")
        return False

    if verify and source == "modelscope" and ms_files and snapshot:
        log(f"[{short_name}] 开始 sha256 校验（{human(actual)}，会跑一阵）")
        bad = ms_verify(snapshot, ms_files, max_workers)
        if bad:
            log(f"[{short_name}] 校验不通过：{', '.join(bad[:5])}（共 {len(bad)} 个）。删掉这些文件后重跑。")
            return False
        log(f"[{short_name}] sha256 全部通过")

    (repo_dir / MARKER).write_text(
        json.dumps(
            {
                "source": source,
                "repo_id": repo_id,
                "hf_repo_id": hf_repo_id,
                "revision": revision,
                "sha": sha,
                "files": file_count,
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


def check_loadable(hf_home: Path, repo_ids: list[str]) -> list[str]:
    """用 local_files_only 走一遍真实解析逻辑，确认算力机离线也能按 repo id 找到模型。"""
    bad = []
    for repo_id in repo_ids:
        code = (
            "import os,sys;"
            f"os.environ['HF_HOME']={str(hf_home)!r};"
            "from huggingface_hub import snapshot_download;"
            f"print(snapshot_download({repo_id!r}, local_files_only=True))"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if proc.returncode == 0:
            log(f"  OK   {repo_id} -> {proc.stdout.strip()}")
        else:
            bad.append(repo_id)
            log(f"  失败 {repo_id}：{proc.stderr.strip().splitlines()[-1] if proc.stderr else '未知错误'}")
    return bad


def spawn_background(hf_home: Path, log_path: Path) -> int:
    """把自己重新拉起为脱离终端的后台进程，输出重定向到日志文件。"""
    hf_home.mkdir(parents=True, exist_ok=True)
    argv = [a for a in sys.argv[1:] if a != "--daemon"]
    # 必须在子进程启动前设好：huggingface_hub 在 import 时就读掉这个开关，
    # 后台模式下 tqdm 进度条会把日志刷成乱码，改由 progress monitor 汇报。
    env = dict(os.environ, _HFDL_CHILD="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    with open(log_path, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), *argv],
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
            cwd=os.getcwd(),
        )
    (hf_home / "download.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把模型下成 HF 缓存目录结构（断点续传 / 后台运行 / HF 与魔搭双源）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-home",
        default="hf_cache",
        help="缓存根目录，模型落到 <HF_HOME>/hub/（默认 ./hf_cache）。整个目录拷到算力机即可",
    )
    parser.add_argument(
        "--source",
        choices=("hf", "modelscope"),
        default=os.environ.get("FORGE_MODEL_SOURCE", "hf"),
        help="下载源：hf（默认）或 modelscope（国内快）。也可用环境变量 FORGE_MODEL_SOURCE",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME_OR_REPO",
        help=f"只下指定模型，可重复。可选短名：{', '.join(DEFAULT_MODELS)}；也可直接给 repo id",
    )
    parser.add_argument("--revision", default=None, help="指定 revision（默认 hf=main / modelscope=master）")
    parser.add_argument("--token", default=None, help="HF token（默认读环境变量 HF_TOKEN）")
    parser.add_argument("--endpoint", default=None, help="HF 镜像，如 https://hf-mirror.com（默认读 HF_ENDPOINT）")
    parser.add_argument("--max-workers", type=int, default=8, help="单个模型内并发下载线程数（默认 8）")
    parser.add_argument("--retries", type=int, default=10, help="每个模型的重试次数（默认 10，指数退避）")
    parser.add_argument("--progress-interval", type=int, default=60, help="进度汇报间隔秒数，0 关闭（默认 60）")
    parser.add_argument("--all-files", action="store_true", help="不过滤 gguf/onnx 等文件，全量下载")
    parser.add_argument("--force", action="store_true", help="忽略完成标记，重新校验下载")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="下完按 sha256 逐文件校验（仅 --source modelscope；177 GiB 大约要跑十几分钟）",
    )
    parser.add_argument(
        "--fake-sha",
        action="store_true",
        help="魔搭源在连不上 HF 时用占位 sha 命名 snapshots（算力机必须设 HF_HUB_OFFLINE=1）",
    )
    parser.add_argument("--list", action="store_true", help="只打印待下载模型与体积，不实际下载")
    parser.add_argument("--daemon", action="store_true", help="后台运行，日志写到 <HF_HOME>/download.log")
    return parser.parse_args(argv)


def resolve_targets(only: list[str] | None, source: str) -> dict[str, tuple[str, str]]:
    """短名 -> (该源的 repo id, HF repo id)。缓存目录名恒按 HF repo id 拼。"""
    if not only:
        return {name: (ids[source], ids["hf"]) for name, ids in DEFAULT_MODELS.items()}
    selected: dict[str, tuple[str, str]] = {}
    for item in only:
        key = item.strip().lower()
        if key in DEFAULT_MODELS:
            selected[key] = (DEFAULT_MODELS[key][source], DEFAULT_MODELS[key]["hf"])
        elif "/" in item:
            selected[item.split("/")[-1].lower()] = (item, item)
        else:
            raise SystemExit(f"未知模型 {item!r}；可选短名：{', '.join(DEFAULT_MODELS)}，或直接给 repo id")
    return selected


def main(argv: list[str] | None = None) -> int:
    global _LOG_FH

    args = parse_args(argv)
    hf_home = Path(args.hf_home).expanduser().resolve()
    log_path = hf_home / "download.log"

    if args.daemon and not os.environ.get("_HFDL_CHILD"):
        pid = spawn_background(hf_home, log_path)
        print(f"已后台启动，PID {pid}")
        print(f"日志：tail -f {log_path}")
        print(f"停止：kill {pid}（已下载部分保留，重跑本命令续传）")
        return 0

    source = args.source
    revision = args.revision or DEFAULT_REVISION[source]

    try:
        targets = resolve_targets(args.only, source)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    token = args.token or os.environ.get("HF_TOKEN") or None
    ignore_patterns = [] if args.all_files else list(DEFAULT_IGNORE)

    needed = "huggingface_hub" if source == "hf" else "requests"
    for module in ("huggingface_hub", needed):  # 魔搭源也要 hub 来查 commit sha
        try:
            __import__(module)
        except ImportError:
            print(
                f"缺少 {module}。本仓库已锁定该依赖，请用 uv 运行：\n  uv run python scripts/download_models.py ...",
                file=sys.stderr,
            )
            return 2

    if args.verify and source != "modelscope":
        print("提示：--verify 依赖魔搭返回的 sha256，--source hf 下不生效（hub 自带 etag 校验）", file=sys.stderr)

    hub_root = hf_home / "hub"
    hub_root.mkdir(parents=True, exist_ok=True)

    if args.list:
        grand_total = 0
        for short_name, (repo_id, _) in targets.items():
            try:
                if source == "modelscope":
                    files = ms_list_files(repo_id, revision, ignore_patterns)
                    size, count = sum(f[1] for f in files), len(files)
                else:
                    size, count, _ = hf_repo_size(repo_id, revision, ignore_patterns, token)
            except Exception as exc:
                print(f"{short_name:20s} {repo_id:28s} 读取失败：{type(exc).__name__}: {exc}")
                continue
            grand_total += size
            print(f"{short_name:20s} {repo_id:28s} {count:4d} 个文件  {human(size)}")
        print(f"{'合计':20s} {'':28s} {'':4s}    {human(grand_total)}")
        print(f"源：{source}  落盘到：{hub_root}  可用空间：{human(shutil.disk_usage(hub_root).free)}")
        return 0

    # 后台模式下 stdout 本身已重定向到该文件，再开一个句柄会让每行日志写两遍。
    if not os.environ.get("_HFDL_CHILD"):
        _LOG_FH = open(log_path, "a", encoding="utf-8")

    log("=" * 72)
    log(f"开始下载 {len(targets)} 个模型 -> {hub_root}")
    if source == "modelscope":
        log(f"源=modelscope（{MS_API}）  revision={revision}")
    else:
        log(f"源=hf  endpoint={os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}  token={'有' if token else '无'}")

    started = time.monotonic()
    failed: list[str] = []
    done_repo_ids: list[str] = []
    for short_name, (repo_id, hf_repo_id) in targets.items():
        ok = download_one(
            short_name,
            repo_id,
            hf_repo_id,
            hub_root,
            source=source,
            revision=revision,
            token=token,
            ignore_patterns=ignore_patterns,
            max_workers=args.max_workers,
            retries=args.retries,
            progress_interval=args.progress_interval,
            force=args.force,
            verify=args.verify,
            fake_sha=args.fake_sha,
        )
        if ok:
            done_repo_ids.append(hf_repo_id)
        else:
            failed.append(short_name)

    elapsed = time.monotonic() - started
    log(f"下载结束，用时 {elapsed / 60:.1f} 分钟；成功 {len(targets) - len(failed)}/{len(targets)}")

    if done_repo_ids:
        log("离线可加载性校验（local_files_only）：")
        check_loadable(hf_home, done_repo_ids)

    if failed:
        log(f"失败：{', '.join(failed)}（重跑同一命令即可续传）")
        return 1

    log("拷到算力机（-a 会保留 blobs 符号链接；别用 -L 或 scp -r，会展开成双份）：")
    log(f"  rsync -avP --exclude '*.log' --exclude '*.pid' {hf_home}/ <user>@<算力机>:/data/hf_cache/")
    log("算力机上：export HF_HOME=/data/hf_cache，config 里继续写 repo id 即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
