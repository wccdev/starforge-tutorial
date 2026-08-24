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
    source = recipe.runtime.resolve(recipe.runtime.default_version).source
    assert source is not None
    assert source.reference.endswith(
        "@sha256:75ac03f34b82134da757e989357e8df456404a535c962cb4d0fb3dc496624648"
    )
    assert MODEL in smoke_script
    assert TRAIN in smoke_script and VALIDATION in smoke_script
    assert plan.argv


def _hydra_overrides(argv):
    return [a for a in argv if "=" in a and not a.startswith("-")]


def test_qa_tools_config_compiles_to_legal_hydra_overrides():
    """verl adapter 扁平化整份 config.yaml，非 verl 键会让 Hydra struct 模式启动即报错。

    最容易踩的是把平台数据集引用写进 config（nemo-rl 路线的正确写法）：
    `data.train.dataset` 不在 adapter 的排除表里，会原样传给
    verl.trainer.main_ppo_sync，而 verl 的 data 配置没有 train 子节点。
    """
    exp_rel = "experiments/verl-grpo_qwen3.5-9b_qa-tools_v1"
    dataset = "aiden_lu/qa-rl-verl@v1"
    data_dir = "/data/starforge/datasets/aiden_lu/qa-rl-verl/v1"
    spec = build_spec(
        exp_rel,
        recipe="verl/grpo",
        pools=["all:h200:1:8"],
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
            "FORGE_CLUSTER_GPUS_PER_NODE": "8",
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
