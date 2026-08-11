"""JobSpec + PlatformBinding → 环境变量：两侧唯一的转换。

改造前这份映射分散在两处且互不知情：
  - 生产侧 server/services/submit.py::build_env_vars（Python）
  - 消费侧 scripts/_run_experiment.sh（Bash 注释里的约定）
改一个变量名要人肉 grep 两个仓库，且没有任何机制能发现两侧已经漂移。

现在它只有这一个实现，并由 tests/test_contract_env_golden.py 逐键锁定。
集群侧脚本从 `KNOWN_KEYS` 读取权威清单，两侧再也无法悄悄分叉。

⚠ 修改本文件 = 修改跨仓库契约。改动必须同步更新 golden 测试与集群侧消费方，
   并考虑老客户端（未携带新变量）的降级行为。
"""
from __future__ import annotations

from .binding import PlatformBinding
from .errors import SpecError
from .spec import JobSpec

# ── 契约变量清单 ──────────────────────────────────────────────────────────────
# 按注入来源分组。集群侧脚本可 import 本清单做自检（收到未知的 LAB_/NRL_ 变量即告警）。

#: 作业身份与可复现元数据。
IDENTITY_KEYS = (
    "NEMO_RL_DIR",
    "CLUSTER_PROFILE",
    "RUN_USER",
    "NRL_RUN_ID",
    "NRL_SUBMIT_USER",
    "NRL_GIT_COMMIT",
    "NRL_GIT_DIRTY",
    "NRL_CONFIG_SHA",
)

#: 服务端权威调度/拓扑。集群侧据此覆盖上传文件里的卡数。
SCHEDULING_KEYS = ("NRL_PIN_RESOURCE", "LAB_CLUSTER_NUM_NODES", "LAB_CLUSTER_GPUS_PER_NODE")

#: 产物落盘。
OUTPUT_KEYS = ("OUTPUT_ROOT",)

#: 密钥注入（二选一：路径 or 明文）。
SECRET_KEYS = ("CLUSTER_SECRETS_FILE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

#: 指标/日志上报凭据。
INGEST_KEYS = ("NEMOLAB_ENDPOINT", "NEMOLAB_RUN_ID", "NEMOLAB_TOKEN", "NEMOLAB_ENABLED")

#: 训练后闭环。
POST_KEYS = ("NRL_TRAIN_RUN_ID",)

#: 算法（recipe）身份 —— 新增。legacy 作业不注入，保证老路径逐字节不变。
RECIPE_KEYS = ("LAB_RECIPE", "LAB_RECIPE_VERSION", "LAB_RECIPE_PLUGINS", "LAB_FRAMEWORK")

KNOWN_KEYS = (
    IDENTITY_KEYS + SCHEDULING_KEYS + OUTPUT_KEYS + SECRET_KEYS + INGEST_KEYS + POST_KEYS + RECIPE_KEYS
)


def spec_to_env(spec: JobSpec, binding: PlatformBinding) -> dict[str, str]:
    """产出提交给 Ray 的 runtime_env.env_vars。

    空字符串值会被剔除（沿用改造前行为：空值等同未设置，避免集群侧 `${VAR:-default}`
    拿到空串而不是回落默认值）。

    注入顺序是契约的一部分，不能随意调整：
      1. 身份与可复现元数据
      2. 服务端权威调度（pin / 拓扑）
      3. profile 级附加 env —— **setdefault 语义**，不覆盖上面的关键变量
      4. 产物根
      5. 非密钥透传 —— 覆盖语义（运维显式配置的值优先级高于 profile 默认）
      6. 密钥
      7. 上报凭据
    """
    if spec.spec.framework.kind == "nemo-rl" and not binding.nemo_rl_dir:
        raise SpecError(
            "服务端未配置 LAB_NEMO_RL_DIR，无法代理提交 nemo-rl 框架作业。", field="binding.nemo_rl_dir"
        )

    prov = spec.provenance
    env: dict[str, str] = {
        "NEMO_RL_DIR": binding.nemo_rl_dir,
        "CLUSTER_PROFILE": binding.profile,
        "RUN_USER": binding.user,
        "NRL_RUN_ID": binding.run_id,
        "NRL_SUBMIT_USER": binding.user,
        "NRL_GIT_COMMIT": prov.git_commit or "unknown",
        "NRL_GIT_DIRTY": "1" if prov.git_dirty else "0",
        "NRL_CONFIG_SHA": prov.config_sha or "none",
    }

    if binding.pin_resource:
        env["NRL_PIN_RESOURCE"] = binding.pin_resource
    if binding.topology:
        env["LAB_CLUSTER_NUM_NODES"] = str(binding.topology[0])
        env["LAB_CLUSTER_GPUS_PER_NODE"] = str(binding.topology[1])

    # profile 级附加 env：setdefault，绝不覆盖上面已注入的权威值。
    for k, v in (binding.profile_env or {}).items():
        env.setdefault(str(k), str(v))

    if binding.output_root:
        env["OUTPUT_ROOT"] = binding.output_root

    for k, v in (binding.passthrough_env or {}).items():
        if v:
            env[str(k)] = str(v)

    # 密钥：优先容器侧文件路径（不进 dashboard）；否则注入服务端已过滤的明文。
    if binding.cluster_secrets_file:
        env["CLUSTER_SECRETS_FILE"] = binding.cluster_secrets_file
    elif binding.inline_secrets:
        env.update({str(k): str(v) for k, v in binding.inline_secrets.items()})

    if binding.hf_token:
        env["HF_TOKEN"] = binding.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = binding.hf_token

    if binding.ingest:
        env["NEMOLAB_ENDPOINT"] = binding.ingest.endpoint
        env["NEMOLAB_RUN_ID"] = binding.run_id
        env["NEMOLAB_TOKEN"] = binding.ingest.token
        env["NEMOLAB_ENABLED"] = "1"

    if binding.train_run_id:
        env["NRL_TRAIN_RUN_ID"] = binding.train_run_id

    # recipe 身份：仅非 legacy 作业注入，使老客户端路径的产出逐字节不变。
    if not spec.is_legacy:
        env["LAB_RECIPE"] = spec.spec.recipe.name
        env["LAB_FRAMEWORK"] = spec.spec.framework.kind
        if spec.spec.recipe.version:
            env["LAB_RECIPE_VERSION"] = spec.spec.recipe.version
        if spec.spec.recipe.plugins:
            env["LAB_RECIPE_PLUGINS"] = ",".join(spec.spec.recipe.plugins)

    return {k: v for k, v in env.items() if v != ""}
