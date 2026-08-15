"""命令层共享：仓库根定位、实验/profile 解析、Tab 补全回调。"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from nemo_rl_lab import cli_ui

# 包位于 <repo>/nemo_rl_lab/commands/，仓库根是上两级（editable 安装下 __file__ 指向源码）。
ROOT = Path(__file__).resolve().parent.parent.parent


class Kind(str, Enum):
    experiments = "experiments"
    projects = "projects"


def resolve_exp(name: str) -> str:
    """把实验名解析为相对仓库根的路径，接受 'experiments/x' / 'projects/x' / 'x'。"""
    cands = [name] if "/" in name else [f"experiments/{name}", f"projects/{name}"]
    for c in cands:
        if (ROOT / c).is_dir():
            return c
    cli_ui.fail(f"找不到实验「{name}」", hint="运行 lab ls 查看可用实验")


def list_exps() -> list[str]:
    out: list[str] = []
    for kind in ("experiments", "projects"):
        base = ROOT / kind
        if base.is_dir():
            out += [p.name for p in base.iterdir() if p.is_dir()]
    return sorted(set(out))


def list_profiles() -> list[str]:
    base = ROOT / "cluster"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "overrides.conf").is_file())


def resolve_profile(exp_path: str, profile: Optional[str]) -> str:
    """确定作业的硬件 profile：--profile 优先，否则读实验目录 cluster 文件；都没有直接报错。

    打包清单按 profile 收窄（只上传 cluster/<profile>/），所以 profile 必须在
    客户端就确定下来，不能留给服务端猜。
    """
    p = (profile or "").strip()
    if not p:
        cluster_file = ROOT / exp_path / "cluster"
        if not cluster_file.is_file() or not cluster_file.read_text(encoding="utf-8").strip():
            cli_ui.fail(
                "无法确定硬件 profile",
                hint=f"加 --profile <名称>，或写入实验目录：echo h100 > {exp_path}/cluster",
            )
        p = cluster_file.read_text(encoding="utf-8").strip()
    if not (ROOT / "cluster" / p / "overrides.conf").is_file():
        opts = " ".join(list_profiles())
        cli_ui.fail(f"未知硬件 profile「{p}」", hint=f"可选: {opts or '(无)'}")
    return p


# ----------------------------- 动态补全回调 -----------------------------
def complete_exp(incomplete: str) -> list[str]:
    return [e for e in list_exps() if e.startswith(incomplete)]


def complete_profile(incomplete: str) -> list[str]:
    return [p for p in list_profiles() if p.startswith(incomplete)]


def complete_method(incomplete: str) -> list[str]:
    """补全两段式方法标识；也按叶子名前缀匹配（输入 gr 可补出 nemo-rl/grpo）。"""
    from nemo_lab_sdk.recipes import recipe_names

    return [
        name for name in recipe_names()
        if name.startswith(incomplete) or name.split("/", 1)[1].startswith(incomplete)
    ]


# 共享的 profile 选项（submit/export/eval 提交时把硬件 profile 转发给服务端，决定集群 overrides）。
PROF_OPT = typer.Option(
    None, "--profile", autocompletion=complete_profile,
    help="硬件 profile（默认用实验目录下的 cluster 文件）",
)
