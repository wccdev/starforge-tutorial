"""Recipe 注册表：目录即方法，服务端零改动即可新增后训练方法。

这组测试同时是「阶段 1 验收标准」的可执行表述。
"""
from __future__ import annotations

import pytest
import yaml

from nemo_rl_lab.contract import SpecError
from nemo_rl_lab.recipes import CATALOG_DIR, Recipe, get_recipe, recipe_names


def test_catalog_loads_and_covers_expected_methods():
    names = recipe_names()
    # dpo / distillation 是 NeMo-RL 0.7.0 早就自带、但改造前平台无从声明的方法。
    assert {"grpo", "sft", "dpo", "distillation", "opsd", "maxrl"} <= set(names)


def test_unknown_recipe_error_lists_available_ones():
    """错误信息要能自助解决问题，而不是只说「不认识」。"""
    with pytest.raises(SpecError) as e:
        get_recipe("ppo-v2")
    assert "grpo" in str(e.value)


@pytest.mark.parametrize("name", recipe_names())
def test_every_recipe_is_internally_consistent(name):
    r = get_recipe(name)
    assert r.version, f"{name} 缺少 version（作业记录要落这个字段）"
    assert r.title and r.summary, f"{name} 缺少展示文案"
    assert r.primary_metrics, f"{name} 未声明核心指标，前端与诊断无从取默认曲线"
    assert r.roles, f"{name} 未声明角色，阶段 6 的资源池映射无从校验"
    for pname, spec in r.params.items():
        assert spec.path, f"{name}.{pname} 缺少 path，无法翻译成训练框架 override"


@pytest.mark.parametrize("name", recipe_names())
def test_every_recipe_directory_name_matches_manifest(name):
    assert (CATALOG_DIR / name / "recipe.yaml").is_file()


# ── 超参校验：这是「提交前挡下错误」的那道闸 ──────────────────────────────────


def test_typo_in_hyperparam_name_is_rejected_with_suggestions():
    """改造前拼错的超参会一路跑到集群上，或者更糟：被静默忽略，用户以为自己调了参。"""
    with pytest.raises(SpecError) as e:
        get_recipe("grpo").validate_hyperparams({"max_num_step": 100})
    msg = str(e.value)
    assert "max_num_step" in msg and "max_num_steps" in msg


def test_out_of_range_value_is_rejected():
    with pytest.raises(SpecError, match="<= 1.0"):
        get_recipe("grpo").validate_hyperparams({"reference_policy_kl_penalty": 5.0})


def test_negative_step_count_is_rejected():
    with pytest.raises(SpecError, match=">= 1"):
        get_recipe("grpo").validate_hyperparams({"max_num_steps": 0})


def test_wrong_type_is_rejected():
    with pytest.raises(SpecError, match="应为 int"):
        get_recipe("grpo").validate_hyperparams({"max_num_steps": "many"})


def test_bool_is_not_accepted_as_int():
    """bool 是 int 的子类，不显式排除的话 True 会被当成合法步数。"""
    with pytest.raises(SpecError, match="bool"):
        get_recipe("grpo").validate_hyperparams({"max_num_steps": True})


def test_int_literal_accepted_for_float_param():
    """YAML 里 0 与 0.0 常混用，不该因此报错。"""
    out = get_recipe("grpo").validate_hyperparams({"reference_policy_kl_penalty": 0})
    assert out["reference_policy_kl_penalty"] == 0


def test_enum_choice_is_enforced():
    with pytest.raises(SpecError, match="不在允许集合"):
        get_recipe("opsd").validate_hyperparams({"teacher_mode": "external"})


def test_defaults_are_filled_in():
    out = get_recipe("opsd").validate_hyperparams({})
    assert out["teacher_mode"] == "self"
    assert out["per_token_kl_clip"] == 10.0


def test_non_strict_mode_passes_unknown_keys_through():
    """legacy 路径不校验超参，原样透传。"""
    out = get_recipe("grpo").validate_hyperparams({"whatever": 1}, strict=False)
    assert out["whatever"] == 1


# ── 超参 → 训练框架 override ─────────────────────────────────────────────────


def test_hyperparams_translate_to_config_overrides():
    """服务端理解作业内容的具体体现：知道超参该落到配置的哪个位置。"""
    overrides = get_recipe("grpo").config_overrides(
        {"max_num_steps": 500, "reference_policy_kl_penalty": 0.01, "val_at_start": True}
    )
    assert "grpo.max_num_steps=500" in overrides
    assert "loss_fn.reference_policy_kl_penalty=0.01" in overrides
    # bool 要渲染成训练框架 CLI 认的字面量，不能是 Python 的 True
    assert "grpo.val_at_start=true" in overrides


def test_unknown_params_are_not_translated():
    assert get_recipe("grpo").config_overrides({"nonexistent": 1}) == []


# ── lifecycle 取代硬编码白名单 ───────────────────────────────────────────────


def test_lifecycle_replaces_hardcoded_post_action_whitelist():
    assert get_recipe("grpo").supports("export")
    assert get_recipe("grpo").supports("eval")
    assert not get_recipe("grpo").supports("serve")


# ── 新增方法 = 新增目录，服务端零改动 ─────────────────────────────────────────


def test_adding_a_recipe_requires_only_a_directory(tmp_path, monkeypatch):
    """阶段 1 的验收标准，用可执行的方式表述。"""
    import nemo_rl_lab.recipes as reg

    catalog = tmp_path / "catalog"
    (catalog / "ppo").mkdir(parents=True)
    (catalog / "ppo" / "recipe.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "ppo",
                "version": "0.7.0",
                "title": "PPO",
                "summary": "带 critic 的经典策略优化",
                "entrypoint": {"base": "nemo_rl", "path": "examples/run_ppo.py"},
                "lifecycle": ["export"],
                "roles": ["actor", "critic", "reference", "rollout"],
                "metrics": {"primary": ["train/reward"]},
                "params": {"max_num_steps": {"type": "int", "min": 1, "path": "ppo.max_num_steps"}},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(reg, "CATALOG_DIR", catalog)
    reg.reset_cache()
    try:
        assert reg.recipe_names() == ["ppo"]
        r = reg.get_recipe("ppo")
        assert r.roles == ("actor", "critic", "reference", "rollout")
        assert r.config_overrides({"max_num_steps": 10}) == ["ppo.max_num_steps=10"]
    finally:
        reg.reset_cache()


def test_directory_name_must_match_manifest_name(tmp_path, monkeypatch):
    import nemo_rl_lab.recipes as reg

    catalog = tmp_path / "catalog"
    (catalog / "grpo").mkdir(parents=True)
    (catalog / "grpo" / "recipe.yaml").write_text(
        "name: not-grpo\nversion: '1'\nentrypoint: {base: nemo_rl, path: examples/x.py}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reg, "CATALOG_DIR", catalog)
    reg.reset_cache()
    try:
        with pytest.raises(SpecError, match="不一致"):
            reg.all_recipes()
    finally:
        reg.reset_cache()


def test_unknown_entrypoint_base_is_rejected():
    with pytest.raises(SpecError, match="entrypoint.base"):
        Recipe.from_dict({"name": "x", "entrypoint": {"base": "somewhere", "path": "a.py"}})


def test_unknown_param_type_is_rejected():
    with pytest.raises(SpecError, match="未知类型"):
        Recipe.from_dict(
            {
                "name": "x",
                "entrypoint": {"base": "nemo_rl", "path": "examples/x.py"},
                "params": {"p": {"type": "complex"}},
            }
        )


def test_recipe_without_entrypoint_is_rejected():
    """没有入口的 recipe 无法执行，必须在加载期就失败，而不是提交后才发现。"""
    with pytest.raises(SpecError, match="entrypoint.path"):
        Recipe.from_dict({"name": "x", "version": "1"})
