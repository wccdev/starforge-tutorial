"""Optional verl GPU smoke entrypoints are pinned and pre-compilable."""
from pathlib import Path

import pytest
from starforge_core.frameworks import CompileRequest, compile_launch_plan
from starforge_core.recipes import get_recipe

from starforge_cli.recipe_lock import validate_recipe_lock
from starforge_cli.spec_builder import build_spec

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN = "/data/starforge/smoke/gsm8k/train.parquet"
VALIDATION = "/data/starforge/smoke/gsm8k/test.parquet"


@pytest.mark.parametrize(
    ("name", "exp_dir", "pool", "gpus", "script"),
    [
        ("verl/sft", "verl-sft", "trainer:h100:1:1", 1, "smoke_verl_sft.sh"),
        ("verl/grpo", "verl-grpo", "all:h200:1:2", 2, "smoke_verl_grpo.sh"),
    ],
)
def test_verl_smoke_contract_is_exact_and_compiles(name, exp_dir, pool, gpus, script):
    exp_rel = f"smoke/{exp_dir}"
    validate_recipe_lock(ROOT / exp_rel, name)
    spec = build_spec(
        exp_rel,
        recipe=name,
        pools=[pool],
        base_model=MODEL,
        train_data=TRAIN,
        validation_data=VALIDATION,
    )
    recipe = get_recipe(name)
    plan = compile_launch_plan(CompileRequest(
        operation="train",
        spec=spec,
        recipe=recipe,
        work_dir=ROOT,
        env={
            "STARFORGE_ENABLED": "0",
            "FORGE_CLUSTER_NUM_NODES": "1",
            "FORGE_CLUSTER_GPUS_PER_NODE": str(gpus),
        },
    ))

    smoke_script = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert spec.spec.resources.total_gpus == gpus
    runtime = recipe.runtime.resolve(recipe.runtime.default_version)
    # 关键契约是「镜像不浮动」，而钉法有两种，不能只认公开 digest 那一种：
    #   - 公开 OCI 源：按 @sha256 钉死；
    #   - deployment_artifact：平台构建的产物，没有公开 digest 可钉，
    #     身份由 runtime_id 里的确切版本给出（verl 0.9 起走这条）。
    if runtime.source is not None:
        assert "@sha256:" in runtime.source.reference
    else:
        assert runtime.deployment_artifact, "既无 OCI 源、又不是部署产物 = 镜像来源不明"
        assert recipe.runtime.default_version in runtime.runtime_id
    assert MODEL in smoke_script
    assert TRAIN in smoke_script and VALIDATION in smoke_script
    assert plan.argv


def _hydra_overrides(argv):
    return [a for a in argv if "=" in a and not a.startswith("-")]


def test_qa_tools_config_compiles_to_legal_hydra_overrides():
    """verl adapter 扁平化整份 config.yaml，非 verl 键会让 Hydra struct 模式启动即报错。

    最容易踩的是把平台数据集引用写进 config（nemo-rl 路线的正确写法）：
    `data.train.dataset` 不在 adapter 的排除表里，会原样传给
    verl.trainer.main_ppo，而 verl 的 data 配置没有 train 子节点。
    """
    exp_rel = "experiments/verl-grpo_qwen3.5-9b_qa-tools_v1"
    dataset = "aiden_lu/qa-rl-verl@v1"
    data_dir = "/data/starforge/datasets/aiden_lu/qa-rl-verl/v1"
    spec = build_spec(
        exp_rel,
        recipe="verl/grpo",
        # 本实验实际跑在 1 张 H200 上，按真实形状编译。
        pools=["all:h200:1:1"],
        base_model="Qwen/Qwen3.5-9B",
        train_data="train.parquet",
        validation_data="val.parquet",
        train_dataset=dataset,
        validation_dataset=dataset,
    )
    plan = compile_launch_plan(CompileRequest(
        operation="train",
        spec=spec,
        recipe=get_recipe("verl/grpo"),
        work_dir=ROOT,
        env={
            "STARFORGE_ENABLED": "0",
            "FORGE_CLUSTER_NUM_NODES": "1",
            "FORGE_CLUSTER_GPUS_PER_NODE": "1",
            "QA_RL_VERL_DATA_DIR": data_dir,
        },
    ))

    overrides = _hydra_overrides(plan.argv)
    leaked = [o for o in overrides if o.startswith(("data.train.", "data.validation."))]
    assert not leaked, f"数据集引用泄漏成非法 hydra 键: {leaked}"
    assert f'data.train_files="{data_dir}/train.parquet"' in overrides
    assert f'data.val_files="{data_dir}/val.parquet"' in overrides
    # Agent Loop 三件套必须真的到达训练进程。
    assert 'actor_rollout_ref.rollout.mode="async"' in overrides
    assert 'actor_rollout_ref.rollout.agent.default_agent_loop="tool_agent"' in overrides
    assert "actor_rollout_ref.rollout.multi_turn.enable=true" in overrides


def test_qa_tools_lora_lands_on_the_key_fsdp_actually_reads():
    """LoRA 必须走扁平 model.lora_rank，不能是嵌套 model.lora.rank。

    verl 0.9 的 model 配置两套 LoRA 键并存（源码自带
    `TODO: unify fsdp and megatron lora config`）：嵌套 `lora:` 字典只有 Megatron
    后端读，FSDP/FSDP2 读扁平键。本实验是 fsdp2，写成嵌套键不会报任何错，只会
    静默退回全参数微调 —— 单卡上表现为 OOM，多卡上则悄悄跑出一个与 nemo-rl
    对照侧（LoRA）不可比的结果。这两种后果都不容易归因，所以用测试钉住。
    """
    dataset = "aiden_lu/qa-rl-verl@v1"
    spec = build_spec(
        "experiments/verl-grpo_qwen3.5-9b_qa-tools_v1",
        recipe="verl/grpo",
        pools=["all:h200:1:1"],
        base_model="Qwen/Qwen3.5-9B",
        train_data="train.parquet",
        validation_data="val.parquet",
        train_dataset=dataset,
        validation_dataset=dataset,
    )
    plan = compile_launch_plan(CompileRequest(
        operation="train",
        spec=spec,
        recipe=get_recipe("verl/grpo"),
        work_dir=ROOT,
        env={
            "STARFORGE_ENABLED": "0",
            "FORGE_CLUSTER_NUM_NODES": "1",
            "FORGE_CLUSTER_GPUS_PER_NODE": "1",
            "QA_RL_VERL_DATA_DIR": "/data/starforge/datasets/aiden_lu/qa-rl-verl/v1",
        },
    ))
    overrides = _hydra_overrides(plan.argv)

    assert "actor_rollout_ref.model.lora_rank=32" in overrides
    assert "actor_rollout_ref.model.lora_alpha=64" in overrides
    nested = [o for o in overrides if o.startswith("actor_rollout_ref.model.lora.")]
    assert not nested, f"嵌套 LoRA 键只有 Megatron 读，fsdp2 下会静默失效: {nested}"
    # 单卡：vLLM 的 TP 默认是 2，不显式写 1 会去要第二张卡。
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in overrides
