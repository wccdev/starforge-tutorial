"""config 解析（defaults 继承 + _override_）与提交前校验的单元测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from starforge_cli.config_resolve import deep_merge, resolve, validate_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_deep_merge_overrides_nested():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    over = {"b": {"y": 3, "z": 4}, "c": 5}
    assert deep_merge(base, over) == {"a": 1, "b": {"x": 1, "y": 3, "z": 4}, "c": 5}


def test_deep_merge_override_marker_replaces_section():
    base = {"data": {"train": {"a": 1}, "validation": {"b": 2}}}
    over = {"data": {"_override_": True, "train": {"c": 3}}}
    # _override_ 整段替换，旧的 validation 被丢弃，标记被剥掉
    assert deep_merge(base, over) == {"data": {"train": {"c": 3}}}


def test_resolve_inherits_defaults(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("grpo:\n  num_prompts_per_step: 4\n  val_period: 10\n")
    (tmp_path / "child.yaml").write_text(
        "defaults:\n  - base.yaml\ngrpo:\n  num_generations_per_prompt: 4\n"
    )
    cfg = resolve(tmp_path / "child.yaml")
    assert cfg["grpo"]["num_prompts_per_step"] == 4  # 来自 base
    assert cfg["grpo"]["num_generations_per_prompt"] == 4  # 来自 child
    assert cfg["grpo"]["val_period"] == 10


def _grpo_cfg(npp, ngp, gbs, vbs=4, mvs=8):
    return {
        "grpo": {
            "num_prompts_per_step": npp,
            "num_generations_per_prompt": ngp,
            "val_batch_size": vbs,
            "max_val_samples": mvs,
            "max_num_steps": 100,
            "val_period": 10,
        },
        "policy": {"train_global_batch_size": gbs, "max_total_sequence_length": 1024},
    }


def test_validate_batch_product_ok():
    errors = [m for lvl, m in validate_config(_grpo_cfg(4, 4, 16)) if lvl == "error"]
    assert errors == []


def test_validate_batch_product_mismatch_is_error():
    issues = validate_config(_grpo_cfg(4, 4, 12))
    errors = [m for lvl, m in issues if lvl == "error"]
    assert len(errors) == 1
    assert "train_global_batch_size" in errors[0]


def test_validate_divisible_offpolicy_batch_is_warn_not_error():
    """rollout 批是 tgbs 整数倍（off-policy 多次更新）是 NeMo-RL 合法配置，只 warn。"""
    issues = validate_config(_grpo_cfg(32, 16, 256))  # rollout=512 = 2×256
    assert [m for lvl, m in issues if lvl == "error"] == []
    assert any("off-policy" in m for lvl, m in issues if lvl == "warn")


def test_validate_force_on_policy_requires_equality():
    cfg = _grpo_cfg(32, 16, 256)
    cfg["loss_fn"] = {"force_on_policy_ratio": True}
    errors = [m for lvl, m in validate_config(cfg) if lvl == "error"]
    assert any("force_on_policy_ratio" in m for m in errors)


def test_validate_val_batch_exceeds_samples():
    issues = validate_config(_grpo_cfg(4, 4, 16, vbs=64, mvs=8))
    errors = [m for lvl, m in issues if lvl == "error"]
    assert any("val_batch_size" in m for m in errors)


def test_validate_nonpositive_steps():
    cfg = _grpo_cfg(4, 4, 16)
    cfg["grpo"]["max_num_steps"] = 0
    errors = [m for lvl, m in validate_config(cfg) if lvl == "error"]
    assert any("max_num_steps" in m for m in errors)


def test_validate_skips_interpolated_values():
    # 带 ${...} 插值的字段无法静态取值，不应误报
    cfg = {
        "grpo": {"num_prompts_per_step": 4, "num_generations_per_prompt": 4},
        "policy": {"train_global_batch_size": "${something}"},
    }
    errors = [m for lvl, m in validate_config(cfg) if lvl == "error"]
    assert errors == []


@pytest.mark.parametrize(
    "exp",
    [
        "grpo_qwen3.5-4b_gsm8k_v1",
        "grpo_qwen3.5-9b_gsm8k_v1",
    ],
)
def test_real_experiments_validate_clean(exp):
    """仓库内现有实验应能解析且无 error（守住回归）。"""
    cfg_file = REPO_ROOT / "experiments" / exp / "config.yaml"
    if not cfg_file.is_file():
        pytest.skip(f"实验不存在: {exp}")
    cfg = resolve(cfg_file)
    errors = [m for lvl, m in validate_config(cfg, repo_root=REPO_ROOT) if lvl == "error"]
    assert errors == [], f"{exp} 校验出错: {errors}"
