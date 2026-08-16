"""命令层共享：项目根定位、实验/profile 解析、Tab 补全回调。"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from starforge_cli import cli_ui

#: StarForge 项目标记文件：`sf init` 生成，项目发现的唯一依据。
#: CLI 已 pip 化分发（starforge-cli），不再假设自己被 clone 在项目仓库里。
PROJECT_MARKER = "starforge.yaml"


def project_root() -> Path:
    """从 cwd 向上定位含 starforge.yaml 的项目根（类比 git 的仓库发现）。

    找不到即失败并给出可执行指引——所有需要项目上下文的命令都经此收口；
    login / status 等全局命令不触碰它。SF_PROJECT_ROOT 环境变量可显式覆盖
    （CI / 脚本场景免 cd）。
    """
    if env := os.environ.get("SF_PROJECT_ROOT"):
        p = Path(env).resolve()
        if (p / PROJECT_MARKER).is_file():
            return p
        cli_ui.fail(f"SF_PROJECT_ROOT={env} 不是 StarForge 项目（缺 {PROJECT_MARKER}）")
    cur = Path.cwd().resolve()
    for cand in (cur, *cur.parents):
        if (cand / PROJECT_MARKER).is_file():
            return cand
    cli_ui.fail(
        "当前目录不在 StarForge 项目内",
        hint="sf init <项目名> 创建新项目，或 cd 进已有项目目录（含 starforge.yaml）",
    )


def __getattr__(name: str):
    """PEP 562：`common.ROOT` 惰性解析为当前项目根。

    保持属性形态是刻意的——测试用 monkeypatch.setattr(common, "ROOT", tmp)
    注入后，真实属性优先于 __getattr__，注入语义不变。
    """
    if name == "ROOT":
        return project_root()
    raise AttributeError(name)


def _root() -> Path:
    """模块内部取项目根：优先被注入的 ROOT 属性（测试），否则实时发现。"""
    injected = globals().get("ROOT")
    return injected if isinstance(injected, Path) else project_root()


class Kind(str, Enum):
    experiments = "experiments"
    projects = "projects"


def resolve_exp(name: str) -> str:
    """把实验名解析为相对仓库根的路径，接受 'experiments/x' / 'projects/x' / 'x'。"""
    cands = [name] if "/" in name else [f"experiments/{name}", f"projects/{name}"]
    for c in cands:
        if (_root() / c).is_dir():
            return c
    cli_ui.fail(f"找不到实验「{name}」", hint="运行 sf ls 查看可用实验")


def list_exps() -> list[str]:
    out: list[str] = []
    for kind in ("experiments", "projects"):
        base = _root() / kind
        if base.is_dir():
            out += [p.name for p in base.iterdir() if p.is_dir()]
    return sorted(set(out))


def list_profiles() -> list[str]:
    """可用硬件 profile 名，来自服务端注册表（本仓库已无 cluster/ 目录）。

    仅用于补全与提示：拿不到（未登录/服务不可达）就静默返回空，不阻断主流程。
    """
    try:
        from starforge_cli import api_client

        data = api_client.cluster_status_via_server()
        return sorted(
            str(p.get("name")) for p in (data.get("profiles") or []) if p.get("name")
        )
    except Exception:  # 网络/鉴权失败只影响补全，不阻断主流程
        return []


def profile_registry() -> dict[str, dict]:
    """服务端 profile 注册表：{名称: {series, num_nodes, gpus_per_node, ...}}。

    提交时把 `--profile 名称[:总卡数]` 物化成 JobSpec 资源池要用（series 与
    默认形状的唯一来源）。拿不到就显式失败——提交本来就离不开服务端。
    """
    from starforge_cli import api_client

    try:
        data = api_client.cluster_status_via_server()
    except Exception as e:  # noqa: BLE001
        cli_ui.fail(
            "无法从服务端获取 profile 注册表",
            hint=f"提交需要在线解析 --profile 的卡型与默认形状；请先 sf login 或检查服务可达性（{e}）",
        )
    return {
        str(p.get("name")): p
        for p in (data.get("profiles") or [])
        if p.get("name")
    }


def resolve_profile(exp_path: str, profile: Optional[str]) -> str:
    """确定作业的硬件 profile：--profile 优先，否则读实验目录遗留的 cluster 标注（兼容旧实验）。

    profile 的 env/overrides/拓扑都在服务端注册表，名字的合法性也由服务端裁决；
    客户端只负责把选择传上去。
    """
    p = (profile or "").strip()
    if not p:
        legacy = _root() / exp_path / "cluster"
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
    from starforge_sdk.recipes import recipe_names

    return [
        name for name in recipe_names()
        if name.startswith(incomplete) or name.split("/", 1)[1].startswith(incomplete)
    ]


# 共享的 profile 选项（export/eval 用：资源形状由 recipe 固定，只选卡型/环境）。
PROF_OPT = typer.Option(
    None, "--profile", autocompletion=complete_profile,
    help="硬件 profile（服务端注册表管理；`sf status` 可查看可用值）",
)

# submit 用的统一资源参数：profile 即资源入口，形状用 :总卡数 修饰。
PROFILE_EXPR_OPT = typer.Option(
    [], "--profile", metavar="[ROLE=]名称[:总卡数]", autocompletion=complete_profile,
    help="目标硬件与资源，一个参数说清：h200（注册表默认形状）、h200:4（4 张卡）、"
         "h200:16（2 满节点）。可重复以按角色分池（异构扩展位）：--profile train=h200:8 --profile rollout=h100:2",
)
