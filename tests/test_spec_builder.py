"""CLI 侧的 JobSpec 构建与本地预校验（阶段 10）。

改造前 `lab submit` 只发 exp 名 + profile，服务端对「跑的是什么方法、超参是什么」
一无所知。现在客户端构建完整 spec 并在**本地**按 recipe 声明预校验 —— 超参拼错、
取值越界在敲回车那一刻就报，不必等上传完、排完队、跑起来才发现。

本地用的是与服务端同一份 recipe 目录（都来自 nemo-lab-sdk），不会出现
「本地说没问题、服务端说不行」。
"""
from __future__ import annotations

import pytest
from nemo_lab_sdk.contract import SpecError

from nemo_rl_lab.spec_builder import build_spec, infer_recipe, parse_pool, parse_set

# ── --set 解析 ───────────────────────────────────────────────────────────────


def test_set_infers_literal_types():
    """命令行给过来的一切都是字符串，而 recipe schema 会拒绝 "500" 作为 int。"""
    got = parse_set(["max_num_steps=500", "reference_policy_kl_penalty=0.01", "val_at_start=true"])
    assert got == {"max_num_steps": 500, "reference_policy_kl_penalty": 0.01, "val_at_start": True}


@pytest.mark.parametrize("raw,expected", [
    ("k=false", False), ("k=no", False), ("k=on", True), ("k=1e-5", 1e-5), ("k=abc", "abc"),
])
def test_set_literal_edge_cases(raw, expected):
    assert parse_set([raw])["k"] == expected


@pytest.mark.parametrize("bad", ["novalue", "=v"])
def test_malformed_set_rejected(bad):
    with pytest.raises(SpecError):
        parse_set([bad])


# ── --pool 解析 ──────────────────────────────────────────────────────────────


def test_pool_parsing():
    """grpo_noncolocated（1 卡生成 / 1 卡训练）第一次能被显式表达。"""
    p = parse_pool("rollout:h100:1:2")
    assert (p.name, p.series, p.nodes, p.gpus_per_node, p.gpus) == ("rollout", "h100", 1, 2, 2)


@pytest.mark.parametrize("bad", ["train:h200:1", "train:h200:1:2:3", "train:h200:0:2", "train:h200:x:2"])
def test_malformed_pool_rejected(bad):
    with pytest.raises(SpecError):
        parse_pool(bad)


# ── 本地预校验 ───────────────────────────────────────────────────────────────


def test_typo_in_hyperparam_caught_before_any_network_call():
    with pytest.raises(SpecError) as e:
        build_spec("experiments/demo", recipe="grpo", sets=["max_num_step=100"])
    assert "max_num_steps" in str(e.value)


def test_out_of_range_caught_locally():
    with pytest.raises(SpecError):
        build_spec("experiments/demo", recipe="grpo", sets=["reference_policy_kl_penalty=99"])


def test_unknown_method_lists_available():
    with pytest.raises(SpecError) as e:
        build_spec("experiments/demo", recipe="nonexistent")
    assert "grpo" in str(e.value)


def test_unsupported_lifecycle_action_rejected():
    with pytest.raises(SpecError, match="serve"):
        build_spec("experiments/demo", recipe="grpo", on_success=["serve"])


def test_unknown_role_rejected():
    with pytest.raises(SpecError, match="critic"):
        build_spec(
            "experiments/demo", recipe="grpo",
            pools=["a:h200:1:2", "b:h100:1:1"], roles=["critic:a"],
        )


def test_role_pointing_at_undefined_pool_rejected():
    with pytest.raises(SpecError, match="未定义的资源池"):
        build_spec(
            "experiments/demo", recipe="grpo",
            pools=["train:h200:1:2", "extra:h100:1:1"], roles=["actor:nope", "rollout:extra"],
        )


def test_multiple_pools_require_explicit_roles():
    with pytest.raises(SpecError, match="--role"):
        build_spec("experiments/demo", recipe="grpo", pools=["a:h200:1:2", "b:h100:1:1"])


def test_single_pool_infers_roles():
    """只有一个池时不必手写映射：方法声明的角色全落在它上面。"""
    from nemo_lab_sdk.recipes import get_recipe

    spec = build_spec("experiments/demo", recipe="grpo", pools=["all:h200:1:8"])
    assert set(spec.spec.resources.roles) == set(get_recipe("grpo").roles)
    assert all(v == "all" for v in spec.spec.resources.roles.values())


# ── 产出形态 ─────────────────────────────────────────────────────────────────


def test_spec_carries_recipe_identity_and_defaults():
    spec = build_spec("experiments/demo", recipe="opsd")
    assert spec.recipe_name == "opsd"
    assert spec.spec.recipe.version
    assert "opsd" in spec.spec.recipe.plugins
    # 默认值在本地就补齐 —— 作业记录因此自解释
    assert spec.spec.hyperparams["teacher_mode"] == "self"


def test_init_from_becomes_artifact_ref():
    spec = build_spec(
        "experiments/dpo_v1", recipe="dpo", init_from="run/sft-1/checkpoint@step=2000",
    )
    ref = spec.spec.model.init_from
    assert (ref.run_id, ref.kind, ref.step) == ("sft-1", "checkpoint", 2000)


def test_provenance_is_recorded():
    spec = build_spec(
        "experiments/demo", recipe="grpo",
        provenance={"git_commit": "abc", "git_dirty": True, "config_sha": "sha"},
    )
    assert spec.provenance.git_commit == "abc" and spec.provenance.git_dirty


def test_owner_left_empty_for_server_to_fill():
    """客户端声明 owner 无效 —— 服务端会权威覆写。"""
    assert build_spec("experiments/demo", recipe="grpo").metadata.owner == ""


def test_validation_can_be_skipped():
    """--no-validate 时不校验超参，但仍产出结构合法的 spec。"""
    spec = build_spec("experiments/demo", recipe="grpo", sets=["whatever=1"], validate=False)
    assert spec.spec.hyperparams["whatever"] == 1


# ── 方法推断 ─────────────────────────────────────────────────────────────────


def test_method_inferred_from_experiment_dir(tmp_path):
    """与 cluster / framework 同款约定：跟着实验走，fork 时自动继承。"""
    (tmp_path / "method").write_text("opsd\n", encoding="utf-8")
    assert infer_recipe(tmp_path) == "opsd"


def test_no_method_file_returns_empty(tmp_path):
    assert infer_recipe(tmp_path) == ""
