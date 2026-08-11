"""跨仓库契约的 golden 测试 —— 阶段 0 的核心交付物。

这组测试锁定 `spec_to_env()` 的**逐键输出**。console 与 lab 两个仓库都跑它，
因此任何一侧单方面改动契约都会在 CI 里当场失败，而不是在集群上表现为诡异行为。

改造前的等价物是 server/services/submit.py::build_env_vars —— 下面的
GOLDEN_LEGACY_ENV 就是它在给定输入下的产出，一个键都不能变。
"""
from __future__ import annotations

import pytest

from nemo_rl_lab.contract import (
    IngestBinding,
    JobSpec,
    PlatformBinding,
    SpecError,
    legacy_spec,
    spec_to_env,
)
from nemo_rl_lab.contract.env import KNOWN_KEYS


def _binding(**over) -> PlatformBinding:
    base = dict(
        run_id="gsm8k-aiden-20260811-101500",
        user="aiden",
        profile="h200",
        nemo_rl_dir="/opt/NeMo-RL",
        output_root="/mnt/shared/outputs",
        pin_resource="accelerator_H200",
        topology=(1, 2),
        profile_env={"NCCL_DEBUG": "WARN", "CLUSTER_PROFILE": "试图覆盖权威值-应被忽略"},
        passthrough_env={"HTTP_PROXY": "http://proxy:3128", "EMPTY_IGNORED": ""},
        cluster_secrets_file="/etc/lab/secrets.env",
        hf_token="",
        ingest=IngestBinding(endpoint="https://console.internal/api/ingest", token="tok-abc"),
    )
    base.update(over)
    return PlatformBinding(**base)


def _legacy() -> JobSpec:
    return legacy_spec(
        "experiments/grpo_qwen3.5-9b_gsm8k_v1",
        user="aiden",
        client_meta={
            "git_commit": "deadbeef",
            "git_dirty": True,
            "config_sha": "sha256:1234",
            "model_name": "Qwen/Qwen3.5-9B",
        },
    )


#: 老客户端路径的权威产出。改动它 = 改动跨仓库契约，必须同步集群侧消费方。
GOLDEN_LEGACY_ENV = {
    "NEMO_RL_DIR": "/opt/NeMo-RL",
    "CLUSTER_PROFILE": "h200",
    "RUN_USER": "aiden",
    "NRL_RUN_ID": "gsm8k-aiden-20260811-101500",
    "NRL_SUBMIT_USER": "aiden",
    "NRL_GIT_COMMIT": "deadbeef",
    "NRL_GIT_DIRTY": "1",
    "NRL_CONFIG_SHA": "sha256:1234",
    "NRL_PIN_RESOURCE": "accelerator_H200",
    "LAB_CLUSTER_NUM_NODES": "1",
    "LAB_CLUSTER_GPUS_PER_NODE": "2",
    "NCCL_DEBUG": "WARN",
    "OUTPUT_ROOT": "/mnt/shared/outputs",
    "HTTP_PROXY": "http://proxy:3128",
    "CLUSTER_SECRETS_FILE": "/etc/lab/secrets.env",
    "NEMOLAB_ENDPOINT": "https://console.internal/api/ingest",
    "NEMOLAB_RUN_ID": "gsm8k-aiden-20260811-101500",
    "NEMOLAB_TOKEN": "tok-abc",
    "NEMOLAB_ENABLED": "1",
}


def test_legacy_env_is_byte_for_byte_stable():
    assert spec_to_env(_legacy(), _binding()) == GOLDEN_LEGACY_ENV


def test_legacy_job_emits_no_recipe_keys():
    """老路径不得注入 LAB_RECIPE*，否则老集群脚本会看到不认识的变量。"""
    env = spec_to_env(_legacy(), _binding())
    assert not [k for k in env if k.startswith("LAB_RECIPE")]


def test_profile_env_cannot_override_authoritative_values():
    """profile_env 是 setdefault 语义：绝不能覆盖服务端权威下发的关键变量。"""
    env = spec_to_env(_legacy(), _binding())
    assert env["CLUSTER_PROFILE"] == "h200"


def test_passthrough_overrides_profile_env():
    """透传是覆盖语义（运维显式配置 > profile 默认）。"""
    env = spec_to_env(
        _legacy(),
        _binding(profile_env={"NCCL_DEBUG": "WARN"}, passthrough_env={"NCCL_DEBUG": "INFO"}),
    )
    assert env["NCCL_DEBUG"] == "INFO"


def test_empty_values_are_dropped():
    """空串等同未设置——保证集群侧 ${VAR:-default} 能回落到默认值。"""
    env = spec_to_env(_legacy(), _binding(output_root="", pin_resource="", topology=None))
    assert "OUTPUT_ROOT" not in env
    assert "NRL_PIN_RESOURCE" not in env
    assert "LAB_CLUSTER_NUM_NODES" not in env
    assert "EMPTY_IGNORED" not in env


def test_missing_provenance_falls_back_to_sentinels():
    spec = legacy_spec("experiments/foo", user="aiden", client_meta={})
    env = spec_to_env(spec, _binding())
    assert env["NRL_GIT_COMMIT"] == "unknown"
    assert env["NRL_GIT_DIRTY"] == "0"
    assert env["NRL_CONFIG_SHA"] == "none"


def test_cluster_secrets_file_wins_over_inline_secrets():
    """配了容器侧密钥文件就不注入明文——明文会出现在 Ray dashboard 上。"""
    env = spec_to_env(
        _legacy(),
        _binding(cluster_secrets_file="/etc/lab/secrets.env", inline_secrets={"WANDB_API_KEY": "plaintext"}),
    )
    assert env["CLUSTER_SECRETS_FILE"] == "/etc/lab/secrets.env"
    assert "WANDB_API_KEY" not in env


def test_inline_secrets_used_when_no_cluster_file():
    env = spec_to_env(
        _legacy(),
        _binding(cluster_secrets_file="", inline_secrets={"WANDB_API_KEY": "plaintext"}),
    )
    assert env["WANDB_API_KEY"] == "plaintext"


def test_hf_token_sets_both_aliases():
    env = spec_to_env(_legacy(), _binding(hf_token="hf_xxx"))
    assert env["HF_TOKEN"] == "hf_xxx"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_xxx"


def test_no_ingest_binding_disables_reporting():
    env = spec_to_env(_legacy(), _binding(ingest=None))
    assert not [k for k in env if k.startswith("NEMOLAB_")]


def test_post_job_carries_train_run_id():
    env = spec_to_env(_legacy(), _binding(train_run_id="train-run-42"))
    assert env["NRL_TRAIN_RUN_ID"] == "train-run-42"


def test_nemo_rl_dir_required_for_nemo_rl_framework():
    with pytest.raises(SpecError, match="LAB_NEMO_RL_DIR"):
        spec_to_env(_legacy(), _binding(nemo_rl_dir=""))


def test_custom_framework_does_not_require_nemo_rl_dir():
    """custom 框架不经 NeMo-RL 启动，不该强制这个路径。"""
    spec = legacy_spec("experiments/foo", user="aiden", framework="custom")
    env = spec_to_env(spec, _binding(nemo_rl_dir=""))
    assert "NEMO_RL_DIR" not in env


# ── 新路径（带 recipe）────────────────────────────────────────────────────────


def _typed_spec_dict() -> dict:
    return {
        "apiVersion": "lab/v1",
        "kind": "TrainingJob",
        "metadata": {"name": "grpo-gsm8k", "project": "math-reasoning", "owner": "ignored-by-server"},
        "spec": {
            "recipe": {"name": "grpo", "version": "0.7.0", "plugins": ["opsd"]},
            "framework": {"kind": "nemo-rl"},
            "source": {"exp": "experiments/grpo_qwen3.5-9b_gsm8k_v1"},
            "model": {
                "base": "Qwen/Qwen3.5-9B",
                "peft": {"kind": "lora", "dim": 8, "alpha": 16},
                "init_from": "run/sft-aiden-20260801/checkpoint@step=2000",
            },
            "resources": {
                "pools": [
                    {"name": "train", "series": "h200", "nodes": 1, "gpus_per_node": 2},
                    {"name": "rollout", "series": "h200", "nodes": 1, "gpus_per_node": 1},
                ],
                "roles": {"actor": "train", "reference": "train", "rollout": "rollout"},
            },
            "hyperparams": {"max_num_steps": 500},
            "lifecycle": {"on_success": ["export", "eval"]},
        },
        "provenance": {"git_commit": "cafe", "config_sha": "sha256:9"},
    }


def test_typed_spec_emits_recipe_identity():
    spec = JobSpec.from_dict(_typed_spec_dict())
    env = spec_to_env(spec, _binding())
    assert env["LAB_RECIPE"] == "grpo"
    assert env["LAB_RECIPE_VERSION"] == "0.7.0"
    assert env["LAB_RECIPE_PLUGINS"] == "opsd"
    assert env["LAB_FRAMEWORK"] == "nemo-rl"


def test_every_emitted_lab_or_nrl_key_is_declared():
    """任何新增的 LAB_/NRL_/NEMOLAB_ 变量都必须登记进 KNOWN_KEYS，供集群侧自检。"""
    spec = JobSpec.from_dict(_typed_spec_dict())
    env = spec_to_env(spec, _binding(train_run_id="t-1", hf_token="hf_x"))
    platform_keys = {k for k in env if k.startswith(("LAB_", "NRL_", "NEMOLAB_"))}
    assert platform_keys <= set(KNOWN_KEYS), f"未登记的契约变量: {platform_keys - set(KNOWN_KEYS)}"


def test_spec_roundtrips_through_dict():
    original = _typed_spec_dict()
    spec = JobSpec.from_dict(original)
    assert JobSpec.from_dict(spec.to_dict()) == spec


def test_resource_pools_give_authoritative_gpu_count():
    """卡数由 pools 算出，不再从 overrides.conf 正则抠文本。"""
    spec = JobSpec.from_dict(_typed_spec_dict())
    assert spec.spec.resources.total_gpus == 3
    assert spec.spec.resources.pool_for_role("rollout").gpus == 1
    assert spec.spec.resources.pool_for_role("actor").name == "train"


def test_owner_is_server_authoritative():
    spec = JobSpec.from_dict(_typed_spec_dict()).with_owner("aiden")
    assert spec.metadata.owner == "aiden"


def test_init_from_artifact_reference_parsing():
    spec = JobSpec.from_dict(_typed_spec_dict())
    ref = spec.spec.model.init_from
    assert (ref.run_id, ref.kind, ref.step) == ("sft-aiden-20260801", "checkpoint", 2000)
    assert ref.to_text() == "run/sft-aiden-20260801/checkpoint@step=2000"
