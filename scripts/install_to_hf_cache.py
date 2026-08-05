#!/usr/bin/env python3
"""把 download_models.py 下好的平铺目录装进 HF 缓存布局，让 config 里继续用 repo id。

为什么需要：仓库所有实验都写 `policy.model_name: "Qwen/Qwen3.6-27B"`，transformers 会去
$HF_HOME/hub 里按 repo id 找。平铺目录不在那个布局里，所以要么改成绝对路径，要么装进缓存。

目标布局（已实测确认）：
    $HF_HOME/hub/models--Qwen--Qwen3.6-27B/
        refs/main            # 内容是 commit sha，【结尾不能有换行】，否则解析不到
        snapshots/<sha>/...  # 放实体文件即可，不必做 blobs + 符号链接

关于 sha：必须是 HF 上真实的 commit sha。用假 sha 时，联网跑会当成没缓存，把整个模型
重下一遍到正确的 sha 目录；只有 HF_HUB_OFFLINE=1 时假 sha 才不会出事。

本文件不依赖仓库里的任何其他模块，可以单独 scp 到算力机上用系统 python3 直接跑
（只需要环境里有 huggingface_hub，NeMo-RL 容器自带）。

用法：
    python3 install_to_hf_cache.py --src hf_models --dry-run
    python3 install_to_hf_cache.py --src hf_models --hf-home /root/.cache/huggingface

注意 --src 要指到【装着各个模型目录的那一层】。比如模型在 hf_models/qwen3.6-27b/ 下，
就传 --src hf_models，不是 --src .

算力机连不上 huggingface.co 时（魔搭下载的 marker 里没有 sha，需要外部来源）：
  - 三个目标模型的真实 sha 已内置在 KNOWN_SHAS，什么都不用加，直接跑，全程不联网；
  - 表外的模型才会联网查，带 10s 超时（--timeout 可调）；想彻底禁掉就加 --offline；
  - 其他模型用 --sha <目录名>=<40位sha> 手动给，sha 可在能联网的机器上执行
    `curl -s https://huggingface.co/api/models/<repo_id> | python3 -c "import json,sys;print(json.load(sys.stdin)['sha'])"` 拿到；
  - 有国内镜像可用时加 --endpoint https://hf-mirror.com 让它自己去查。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

MARKER = ".download_complete"
SKIP_ENTRIES = {MARKER, ".cache", "download.log", "download.pid"}
# 扫描 --src 子目录时要忽略的噪音目录
SKIP_DIRS = {"__pycache__", "hub", "node_modules"}

# 本文件要能单独 scp 到算力机运行，所以这两张表是内联的副本，不 import download_models。
# 改这里时记得同步 download_models.py 的 DEFAULT_MODELS。
SHORT_NAME_TO_HF = {
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
    "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
    "glm-4.7-flash": "zai-org/GLM-4.7-Flash",
}
# 魔搭与 HF 的命名空间不一致，缓存目录名必须按 HF 的来，否则 transformers 找不到。
MS_TO_HF_REPO = {
    "zhipuai/glm-4.7-flash": "zai-org/GLM-4.7-Flash",
}

# 兜底用的真实 commit sha，2026-08-05 从 huggingface.co/api/models/<id> 抓的。
# 只在【连不上 HF 且 marker 里没有 sha】时启用（典型场景：魔搭下载 + 算力机无外网）。
# 比 --fake-sha 强的地方：万一算力机以后能连上 HF，hub 认这个 sha，不会把权重重下一遍。
# 如果上游仓库之后有新提交，这里就会偏旧——那时用 --sha 显式指定。
KNOWN_SHAS = {
    "Qwen/Qwen3.6-27B": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "Qwen/Qwen3.6-35B-A3B": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "zai-org/GLM-4.7-Flash": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
}


def hf_repo_id_for(short_name: str, marker: dict) -> str | None:
    """算出该用哪个 HF repo id 来拼缓存目录名。

    早期版本的 marker 里没有 hf_repo_id 字段，魔搭源存的又是魔搭的 repo id
    （如 ZhipuAI/GLM-4.7-Flash），所以要按短名和命名空间兜底换算。
    """
    if marker.get("hf_repo_id"):
        return marker["hf_repo_id"]
    if short_name.lower() in SHORT_NAME_TO_HF:
        return SHORT_NAME_TO_HF[short_name.lower()]
    repo_id = marker.get("repo_id")
    if repo_id and repo_id.lower() in MS_TO_HF_REPO:
        return MS_TO_HF_REPO[repo_id.lower()]
    return repo_id  # 非内置模型只能按原样用，必要时 --repo-id 覆盖


def fetch_sha(repo_id: str, token: str | None, timeout: float) -> str | None:
    """查 HF 上的 commit sha。

    这里用 urllib 而不是 huggingface_hub：hub 的 model_info 没有可靠的超时，
    在【包被丢弃而非拒绝】的内网机器上会一直挂着（实际卡过好几分钟）。
    """
    import urllib.error
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    req = urllib.request.Request(f"{endpoint}/api/models/{repo_id}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp).get("sha")
    except Exception as exc:
        print(f"           查 sha 失败（{type(exc).__name__}），继续用兜底方案")
        return None


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in SKIP_ENTRIES:
            continue
        yield path, rel


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def install_one(
    model_dir: Path,
    hub_root: Path,
    *,
    mode: str,
    repo_id_override: str | None,
    sha_override: str | None,
    token: str | None,
    fake_sha: bool,
    offline: bool,
    timeout: float,
    dry_run: bool,
) -> str | None:
    """装好返回该模型的 HF repo id，失败返回 None。"""
    short_name = model_dir.name
    marker_path = model_dir / MARKER
    marker = {}
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    else:
        print(f"[{short_name}] 没有 {MARKER}，下载可能没跑完；跳过（确认无误可加 --repo-id 手动装）")
        if not repo_id_override:
            return None

    repo_id = repo_id_override or hf_repo_id_for(short_name, marker)
    if not repo_id:
        print(f"[{short_name}] 无法确定 HF repo id，跳过。用 --repo-id 指定。")
        return None
    origin = marker.get("repo_id")
    if origin and origin != repo_id:
        print(f"[{short_name}] 下载源是 {marker.get('source', '?')}:{origin}，缓存目录按 HF 的 {repo_id} 命名")

    # sha 决定 snapshots 目录名。优先级：手动指定 > 下载时记下的 > 内置表 > 联网查 > 占位。
    # 内置表排在联网前面：它是免费的，而且对应的正是这批权重下载时的版本；
    # 联网只是为表外的模型兜底，且必须带超时，否则无外网的机器会卡死在这里。
    sha = sha_override or marker.get("sha")
    if not sha and repo_id in KNOWN_SHAS:
        sha = KNOWN_SHAS[repo_id]
        print(f"[{short_name}] 用内置的已知 sha {sha[:12]}（未联网，见脚本里 KNOWN_SHAS 的说明）")
    if not sha and not offline:
        print(f"[{short_name}] 表里没有 {repo_id}，联网查 sha（最多等 {timeout:.0f}s）")
        sha = fetch_sha(repo_id, token, timeout)
    if not sha and fake_sha:
        sha = hashlib.sha1(repo_id.encode()).hexdigest()
        print(f"[{short_name}] 用占位 sha {sha[:12]}；算力机上必须 export HF_HUB_OFFLINE=1，否则会整个重下")
    if not sha:
        print(
            f"[{short_name}] 拿不到 {repo_id} 的 commit sha（魔搭源不提供，且连不上 huggingface.co）。\n"
            f"           办法：--sha {short_name}=<40位sha> 手动指定，或 --fake-sha 并在算力机设 HF_HUB_OFFLINE=1。"
        )
        return None

    repo_dir = hub_root / f"models--{repo_id.replace('/', '--')}"
    snapshot_dir = repo_dir / "snapshots" / sha
    files = list(iter_files(model_dir))
    total = sum(p.stat().st_size for p, _ in files)

    print(f"[{short_name}] {repo_id}@{sha[:12]}  {len(files)} 个文件  {total / 1024**3:.1f} GiB  -> {snapshot_dir}")
    if dry_run:
        return repo_id

    for src, rel in files:
        place(src, snapshot_dir / rel, mode)

    refs = repo_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    # 结尾不能有换行：hub 直接把文件内容当 sha 去匹配 snapshots 下的目录名。
    (refs / "main").write_text(sha, encoding="utf-8")
    return repo_id


def verify(repo_ids: list[str], hf_home: Path) -> list[str]:
    """返回解析不出来的 repo id。

    这里显式传 cache_dir 而不是设 HF_HOME 环境变量：huggingface_hub 在 import 时就把
    HF_HUB_CACHE 固化了，而魔搭源那条路径会先 import 它去查 commit sha，改环境变量已经晚了。
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("装好了，但本机没有 huggingface_hub，跳过校验")
        return []
    print("\n校验（local_files_only，模拟离线加载）：")
    bad = []
    for repo_id in repo_ids:
        try:
            path = snapshot_download(repo_id, cache_dir=str(hf_home / "hub"), local_files_only=True)
            print(f"  OK   {repo_id} -> {path}")
        except Exception as exc:
            bad.append(repo_id)
            print(f"  失败 {repo_id}：{type(exc).__name__}: {exc}")
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把平铺模型目录装进 HF 缓存布局")
    parser.add_argument("--src", default="hf_models", help="download_models.py 的落盘根目录")
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME"),
        help="HF_HOME（缓存会写到 <HF_HOME>/hub）。默认读环境变量 HF_HOME",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "move", "copy"),
        default="hardlink",
        help="hardlink（默认，不占额外空间且保留原目录）/ move（跨盘时用）/ copy（占双份空间）",
    )
    parser.add_argument("--only", action="append", help="只装指定子目录名，可重复")
    parser.add_argument("--repo-id", default=None, help="手动指定 HF repo id（配合 --only 用）")
    parser.add_argument("--token", default=None, help="HF token，用于查 commit sha")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="查 sha 时走的 HF 端点，如 https://hf-mirror.com（默认读环境变量 HF_ENDPOINT）",
    )
    parser.add_argument(
        "--sha",
        action="append",
        metavar="NAME=SHA",
        help="手动指定某个模型的 commit sha，可重复。NAME 是目录名或 repo id",
    )
    parser.add_argument("--offline", action="store_true", help="完全不联网，只用 marker / 内置表 / --sha")
    parser.add_argument("--timeout", type=float, default=10.0, help="联网查 sha 的超时秒数（默认 10）")
    parser.add_argument(
        "--fake-sha",
        action="store_true",
        help="查不到真实 commit sha 时用占位值（必须配合算力机上的 HF_HUB_OFFLINE=1）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划")
    args = parser.parse_args(argv)

    if not args.hf_home:
        print("必须指定 --hf-home，或先 export HF_HOME=...", file=sys.stderr)
        return 2

    # 必须在任何 huggingface_hub import 之前设好：它在 import 时就把端点固化了。
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    sha_overrides: dict[str, str] = {}
    for item in args.sha or []:
        if "=" not in item:
            print(f"--sha 要写成 NAME=SHA，收到的是 {item!r}", file=sys.stderr)
            return 2
        name, value = item.split("=", 1)
        sha_overrides[name.strip().lower()] = value.strip()

    src_root = Path(args.src).expanduser().resolve()
    hf_home = Path(args.hf_home).expanduser().resolve()
    hub_root = hf_home / "hub"
    if not src_root.is_dir():
        print(f"源目录不存在：{src_root}", file=sys.stderr)
        return 2

    if (src_root / MARKER).is_file():
        candidates = [src_root]  # --src 直接指到了单个模型目录
    else:
        candidates = sorted(
            d for d in src_root.iterdir() if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")
        )
    if args.only:
        wanted = {name.lower() for name in args.only}
        candidates = [d for d in candidates if d.name.lower() in wanted]
    if not candidates:
        print(f"{src_root} 下没有可装的模型目录", file=sys.stderr)
        return 2

    # --src 指错一层是最常见的用法错误（比如指到 scripts/ 而模型在 scripts/hf_models/ 下）。
    if not args.repo_id and not any((d / MARKER).is_file() for d in candidates):
        hints = [d for d in candidates if any((sub / MARKER).is_file() for sub in d.iterdir() if sub.is_dir())]
        print(f"{src_root} 的子目录里没有一个带 {MARKER}，看起来 --src 指错了层。", file=sys.stderr)
        if hints:
            print(f"模型实际在这里，改用：--src {hints[0]}", file=sys.stderr)
        return 2

    if args.mode == "hardlink":
        same_fs = os.stat(src_root).st_dev == os.stat(hf_home if hf_home.exists() else hf_home.parent).st_dev
        if not same_fs:
            print("源目录与 HF_HOME 不在同一文件系统，硬链接不可用。改用 --mode move 或 --mode copy。", file=sys.stderr)
            return 2

    hub_root.mkdir(parents=True, exist_ok=True)
    token = args.token or os.environ.get("HF_TOKEN") or None

    installed: list[str] = []
    for model_dir in candidates:
        # --sha 既可以用目录名指定，也可以用 repo id 指定。
        override = sha_overrides.get(model_dir.name.lower())
        if not override:
            guess = hf_repo_id_for(model_dir.name, {})
            override = sha_overrides.get(guess.lower()) if guess else None
        repo_id = install_one(
            model_dir,
            hub_root,
            mode=args.mode,
            repo_id_override=args.repo_id,
            sha_override=override,
            token=token,
            fake_sha=args.fake_sha,
            offline=args.offline,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        if repo_id:
            installed.append(repo_id)

    if not installed:
        sys.stdout.flush()  # 否则 stderr 会抢在上面那些逐模型日志之前打印出来
        print("\n没有装成任何模型。", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n（--dry-run，未改动任何文件）")
        return 0

    bad = verify(installed, hf_home)
    if bad:
        sys.stdout.flush()
        print(f"\n{len(bad)} 个模型装完后解析不出来，别急着用：{', '.join(bad)}", file=sys.stderr)
        return 1
    print(f"\n完成。算力机上 export HF_HOME={hf_home} 后，config 里直接写 repo id 即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
