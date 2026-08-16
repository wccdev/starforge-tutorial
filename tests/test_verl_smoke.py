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
