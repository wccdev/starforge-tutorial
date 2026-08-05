"""跨平台同步 NeMo-RL 官方基底配置（替代 scripts/sync_base_configs.sh）。"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# 需要更大模型基底时，把对应文件名加进来，例如 grpo_math_8B.yaml
# configs/base/grpo_megatron.yaml 是本仓库自定义 overlay，不在此同步。
SYNC_FILES = (
    "grpo_math_1B.yaml",
    "sft.yaml",
    "grpo_sliding_puzzle.yaml",
    "distillation_math.yaml",  # OPSD / on-policy 蒸馏系实验的基底
)


class SyncBaseError(Exception):
    pass


def sync_base_configs(repo_root: Path, nemo_rl_dir: str | Path) -> None:
    """从 NeMo-RL 源码 examples/configs 复制官方 yaml 到 configs/base/。"""
    src = Path(nemo_rl_dir).expanduser().resolve() / "examples" / "configs"
    if not src.is_dir():
        raise SyncBaseError(f"NeMo-RL 配置目录不存在: {src}")

    dst = repo_root / "configs" / "base"
    dst.mkdir(parents=True, exist_ok=True)

    for name in SYNC_FILES:
        src_file = src / name
        if src_file.is_file():
            shutil.copy2(src_file, dst / name)
            print(f"synced {name}")
        else:
            print(f"WARN 未找到 {src_file}，跳过")

    print("完成。请 git diff 检查变化后再提交。")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    nemo_rl = os.environ.get("NEMO_RL_DIR")
    i = 0
    while i < len(args):
        if args[i] in ("--nemo-rl", "-n") and i + 1 < len(args):
            nemo_rl = args[i + 1]
            i += 2
        else:
            i += 1

    if not nemo_rl:
        print("请设置 NEMO_RL_DIR 或使用 --nemo-rl 指向本地 NeMo-RL 源码目录", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    try:
        sync_base_configs(repo_root, nemo_rl)
    except SyncBaseError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
