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
    """可用硬件 profile 名，来自服务端注册表（本仓库已无 cluster/ 目录）。

    仅用于补全与提示：拿不到（未登录/服务不可达）就静默返回空，不阻断主流程。
    """
    try:
        from nemo_rl_lab import api_client

        data = api_client.cluster_status_via_server()
        return sorted(
            str(p.get("name")) for p in (data.get("profiles") or []) if p.get("name")
        )
    except Exception:  # 网络/鉴权失败只影响补全，不阻断主流程
        return []


def resolve_profile(exp_path: str, profile: Optional[str]) -> str:
    """确定作业的硬件 profile：--profile 优先，否则读实验目录遗留的 cluster 标注（兼容旧实验）。

    profile 的 env/overrides/拓扑都在服务端注册表，名字的合法性也由服务端裁决；
    客户端只负责把选择传上去。
    """
    p = (profile or "").strip()
    if not p:
        legacy = ROOT / exp_path / "cluster"
        if legacy.is_file():
            p = legacy.read_text(encoding="utf-8").strip()
    if not p:
        opts = " ".join(list_profiles())
        cli_ui.fail(
            "无法确定硬件 profile",
            hint=f"加 --profile <名称>{f'（可选: {opts}）' if opts else ''}",
        )
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


# 共享的 profile 选项（submit/export/eval 提交时把硬件 profile 转发给服务端，决定集群 env/overrides）。
PROF_OPT = typer.Option(
    None, "--profile", autocompletion=complete_profile,
    help="硬件 profile（服务端注册表管理；`lab status` 可查看可用值）",
)
