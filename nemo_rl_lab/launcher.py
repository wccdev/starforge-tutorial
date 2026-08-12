"""集群侧启动器：读 JobSpec，决定怎么跑（阶段 1）。

改造前，「怎么跑」的全部决策都在 scripts/_run_experiment.sh 里，用 bash 表达：
选入口（有 run.py 用它，否则 examples/run_grpo.py）、叠 profile override、算产物目录、
把服务端下发的权威拓扑覆盖回配置文件。这些决策：
  - 服务端看不见（于是无法校验、无法展示、无法诊断）
  - 只能用环境变量与文件约定表达（于是契约隐式且易漂移）
  - 加一种后训练方法就要改这段 shell

现在这些决策由本模块基于 JobSpec 做出，shell 只保留它真正擅长的事：
source 密钥、source profile env（见 scripts/launch.sh）。

⚠ 本模块运行在 NeMo-RL 官方容器内，只能用标准库 + pyyaml（容器必有）。

用法（由 scripts/launch.sh exec 调用，cwd = 上传包根）：
    python -m nemo_rl_lab.launcher
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from nemo_rl_lab.contract import SPEC_FILE_PATH, JobSpec
from nemo_rl_lab.recipes import get_recipe

#: 服务端在上传包里写下的 spec 落点（两侧共享，定义在契约包里）。
SPEC_PATH = Path(SPEC_FILE_PATH)


class LaunchError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"[launch] {msg}", flush=True)


def load_spec(work_dir: Path) -> JobSpec:
    path = work_dir / SPEC_PATH
    if not path.is_file():
        raise LaunchError(f"未找到作业规格 {path}（服务端应在提交时写入）")
    return JobSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── 产物目录：规则与 scripts/_output_paths.sh 及 server.submit.build_output_dir 一致 ──


def train_output_dir(exp_name: str, exp_dir: Path, env: dict[str, str]) -> str:
    """训练落盘目录。

    中心化提交（有 OUTPUT_ROOT）：<OUTPUT_ROOT>/<RUN_USER>/<EXP_NAME>/<NRL_RUN_ID>
    本地直跑（无 OUTPUT_ROOT）：  <exp_dir>/outputs
    """
    root = (env.get("OUTPUT_ROOT") or "").rstrip("/")
    if not root:
        return str(exp_dir / "outputs")
    user = env.get("RUN_USER") or ""
    base = f"{root}/{user}/{exp_name}" if user else f"{root}/{exp_name}"
    run_id = env.get("NRL_RUN_ID") or ""
    return f"{base}/{run_id}" if run_id else base


# ── 入口解析 ─────────────────────────────────────────────────────────────────


def resolve_entrypoint(spec: JobSpec, work_dir: Path, env: dict[str, str]) -> Path:
    """按 recipe 声明定位训练入口脚本。

    spec.source.entrypoint 显式声明时优先（相对上传包根），否则用 recipe 的声明。
    """
    exp_dir = work_dir / spec.exp

    override = spec.spec.source.entrypoint
    if override:
        cand = work_dir / override
        if not cand.is_file():
            raise LaunchError(f"spec 指定的入口不存在: {override}")
        return cand

    recipe = get_recipe(spec.recipe_name)
    ep = recipe.entrypoint
    if ep.base == "exp":
        cand = exp_dir / (ep.path or "run.py")
        if not cand.is_file():
            raise LaunchError(
                f"方法 {recipe.name} 要求实验目录内提供 {ep.path}，但 {cand} 不存在"
            )
        return cand
    if ep.base == "workdir":
        cand = work_dir / ep.path
        if not cand.is_file():
            raise LaunchError(f"入口不存在: {cand}")
        return cand
    # base == "nemo_rl"：相对容器内 NeMo-RL 源码目录
    nemo_rl_dir = env.get("NEMO_RL_DIR") or ""
    if not nemo_rl_dir:
        raise LaunchError(f"方法 {recipe.name} 的入口在 NeMo-RL 内，但未设置 NEMO_RL_DIR")
    cand = Path(nemo_rl_dir) / ep.path
    if not cand.is_file():
        raise LaunchError(f"NeMo-RL 入口不存在: {cand}（NEMO_RL_DIR={nemo_rl_dir}）")
    return cand


# ── 配置 override ────────────────────────────────────────────────────────────


def profile_overrides(work_dir: Path, profile: str) -> list[str]:
    """读 cluster/<profile>/overrides.conf（每行一个 a.b=c，忽略注释/空行）。"""
    conf = work_dir / "cluster" / profile / "overrides.conf"
    if not conf.is_file():
        return []
    out = []
    for line in conf.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def build_overrides(spec: JobSpec, work_dir: Path, out_dir: str, env: dict[str, str]) -> list[str]:
    """装配传给训练入口的 CLI override，优先级由低到高。

    1. profile 的硬件/并行度 override（cluster/<profile>/overrides.conf）
    2. 服务端权威拓扑 —— 覆盖上面文件里的 cluster.num_nodes/gpus_per_node，
       保证「实际占卡 == 服务端记账」，用户改上传文件的卡数无法绕过配额
    3. 产物落盘目录
    4. spec 里的超参 —— 用户显式声明的，优先级最高
    """
    overrides = [
        o
        for o in profile_overrides(work_dir, env.get("CLUSTER_PROFILE") or "")
        # 有服务端权威拓扑时丢弃文件里的拓扑声明。
        if not (
            env.get("LAB_CLUSTER_NUM_NODES")
            and o.split("=", 1)[0].strip() in ("cluster.num_nodes", "cluster.gpus_per_node")
        )
    ]

    nodes, per_node = env.get("LAB_CLUSTER_NUM_NODES"), env.get("LAB_CLUSTER_GPUS_PER_NODE")
    if nodes and per_node:
        overrides += [f"cluster.num_nodes={nodes}", f"cluster.gpus_per_node={per_node}"]

    overrides += [f"checkpointing.checkpoint_dir={out_dir}", f"logger.log_dir={out_dir}/logs"]

    if not spec.is_legacy:
        recipe = get_recipe(spec.recipe_name)
        overrides += recipe.config_overrides(spec.spec.hyperparams)
    return overrides


# ── 算法插件 ─────────────────────────────────────────────────────────────────


def verify_runtime(spec: JobSpec, env: dict[str, str]) -> list[str]:
    """启动前校验 recipe 声明的运行时依赖是否已在镜像里。

    为什么值得做：Ray 集群在内网，但内网 Nexus3 能代理 PyPI/Docker，overlay 镜像
    这条路是**通的**（deploy/ray-cluster/Dockerfile 的 EXTRA_PIP + PIP_INDEX）。
    真正的限制是 Ray 的作业级 runtime_env pip 只影响 driver、不影响训练 worker，
    所以依赖必须在镜像里，而镜像是运维在构建期决定的。

    结果就是：用户提交一个需要 trl 的方法，训练要跑起来几分钟、建完 venv、
    加载完模型之后，才在某个 worker 上抛一句 ImportError。这里把它提前到启动的
    第一秒，并直接给出该跑哪条命令。

    返回未满足的依赖列表（空表示通过）。`packaging` 不可用时跳过校验并告警 ——
    校验本身是便利功能，不该成为新的失败点。
    """
    if spec.is_legacy:
        return []
    requires = list(get_recipe(spec.recipe_name).runtime.requires)
    if not requires:
        return []

    try:
        from importlib.metadata import PackageNotFoundError, version

        from packaging.requirements import Requirement
    except ImportError:  # pragma: no cover - 容器里 packaging 必有，这里只是不硬失败
        _log("warn    : 缺少 packaging，跳过运行时依赖校验")
        return []

    missing: list[str] = []
    for raw in requires:
        try:
            req = Requirement(raw)
            installed = version(req.name)
        except PackageNotFoundError:
            missing.append(f"{raw}（未安装）")
            continue
        except Exception as e:  # noqa: BLE001 — 依赖声明写错不该拖垮启动
            _log(f"warn    : 依赖声明 {raw!r} 解析失败，已跳过: {e}")
            continue
        if req.specifier and installed not in req.specifier:
            missing.append(f"{raw}（当前 {installed}）")

    if missing:
        image = env.get("LAB_IMAGE") or "(未知镜像)"
        _log(f"运行环境不满足方法 {spec.recipe_name} 的依赖要求：")
        for m in missing:
            _log(f"    - {m}")
        _log(f"当前镜像: {image}")
        _log("修复：重建 overlay 镜像并把缺失依赖加进 EXTRA_PIP，例如")
        _log(f"    EXTRA_PIP=\"{' '.join(requires)}\" bash deploy/ray-cluster/build.sh")
        _log("（内网 Nexus3 已代理 PyPI，构建期可直接拉包）")
    return missing


def install_plugins(spec: JobSpec) -> list[str]:
    """装载 spec 声明的算法补丁。

    改造前这些补丁由每个实验的 run.py 自行 import 并 install_*()，平台无从知晓
    某个作业用了哪个补丁、哪个版本，也无法复用到别的实验。现在由 recipe 声明、
    launcher 统一装载 —— monkey-patch 这个手段本身保留（对装不了新依赖的内网
    集群，零依赖是决定性优势），变的是它成了平台的一等概念。
    """
    if spec.is_legacy:
        return []
    names = list(spec.spec.recipe.plugins) or list(get_recipe(spec.recipe_name).plugins)
    if not names:
        return []

    from common.algorithms import registry  # 上传包内，非本包

    loaded = []
    for name in names:
        # DEFERRED 插件（如 opsd 需要 tokenizer）在此只做存在性校验，实际装载留给训练入口；
        # 但「补丁不在包里」这类错误会因此提前到启动期暴露，而不是训练跑起来才 ImportError。
        if registry.install(name, spec.spec.hyperparams):
            loaded.append(name)
            _log(f"plugin  : {name} 已装载")
        else:
            _log(f"plugin  : {name} 已登记（需运行时上下文，由训练入口装载）")
    return loaded


# ── 主流程 ───────────────────────────────────────────────────────────────────


def build_command(spec: JobSpec, work_dir: Path, env: dict[str, str]) -> list[str]:
    exp_dir = work_dir / spec.exp
    exp_name = exp_dir.name
    entry = resolve_entrypoint(spec, work_dir, env)
    out_dir = train_output_dir(exp_name, exp_dir, env)
    overrides = build_overrides(spec, work_dir, out_dir, env)

    cmd = [sys.executable, str(entry)]
    config = exp_dir / "config.yaml"
    if config.is_file():
        cmd += ["--config", str(config)]
    cmd += overrides

    _log(f"recipe  : {spec.recipe_name}")
    _log(f"exp     : {exp_name}")
    _log(f"entry   : {entry}")
    _log(f"out_dir : {out_dir}")
    _log(f"profile : {env.get('CLUSTER_PROFILE') or '(未设置)'}")
    _log("overrides:")
    for o in overrides:
        _log(f"    {o}")
    return cmd


def _lifecycle(env: dict[str, str], event: str) -> None:
    """向 console 打一个生命周期点（阶段 4）。

    只在中心化提交（有 NEMOLAB_* 凭据）时生效；本地直跑无声跳过。
    任何失败都吞掉 —— 打点是可观测性，绝不能让训练起不来或跑不完。
    """
    if env.get("NEMOLAB_ENABLED") != "1":
        return
    try:
        from common.observability.ingest_client import IngestClient

        IngestClient(
            endpoint=env["NEMOLAB_ENDPOINT"],
            run_id=env["NEMOLAB_RUN_ID"],
            token=env["NEMOLAB_TOKEN"],
        ).send_lifecycle(event)
    except Exception as e:  # noqa: BLE001
        _log(f"warn    : 生命周期打点 {event} 失败（不影响训练）: {e}")


def main(argv: list[str] | None = None) -> int:
    work_dir = Path(os.environ.get("LAB_WORK_DIR") or os.getcwd()).resolve()
    env = dict(os.environ)
    try:
        spec = load_spec(work_dir)
        if missing := verify_runtime(spec, env):
            raise LaunchError(
                f"方法 {spec.recipe_name} 需要的依赖不在当前镜像里: {', '.join(missing)}"
            )
        install_plugins(spec)
        cmd = build_command(spec, work_dir, env)
    except (LaunchError, Exception) as e:  # noqa: BLE001 — 启动期任何失败都要给清楚的原因
        print(f"[launch] 启动失败: {e}", file=sys.stderr, flush=True)
        return 2

    if env.get("LAB_DRY_RUN") == "1":
        print(" ".join(shlex.quote(c) for c in cmd), flush=True)
        return 0

    # started 在 exec 之前打：此刻 venv 已就绪、插件已装载、配置已解析，
    # 是「训练真正开始」最接近的时刻。Ray 报的 RUNNING 要早得多 ——
    # 建 venv、拉模型可能占好几分钟，用它当计量起点会系统性偏长。
    _lifecycle(env, "started")
    rc = subprocess.call(cmd, cwd=str(work_dir))
    _lifecycle(env, "succeeded" if rc == 0 else "failed")
    return rc


def spec_summary(spec: JobSpec) -> dict[str, Any]:
    """给日志/调试用的紧凑摘要。"""
    return {
        "recipe": spec.recipe_name,
        "version": spec.spec.recipe.version,
        "exp": spec.exp,
        "plugins": list(spec.spec.recipe.plugins),
        "gpus": spec.spec.resources.total_gpus or None,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
