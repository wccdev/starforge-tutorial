"""lab CLI 辅助函数的单元测试（不触发真实提交 / 网络）。

提交一律走中心化服务（server 模式），CLI 不再有本机直连 Ray / 读 submit.env 的逻辑，
故这里只覆盖与模式无关的纯本地辅助：实验解析、profile 列举、config 校验 / diff。
"""
from __future__ import annotations

import pytest
import typer

from nemo_rl_lab import cli


def test_resolve_exp_known():
    # 仓库内现有实验应可解析为 experiments/<name>
    assert cli._resolve_exp("grpo_qwen3.5-4b_gsm8k_v1") == "experiments/grpo_qwen3.5-4b_gsm8k_v1"


def test_resolve_exp_unknown_raises():
    with pytest.raises(typer.Exit):
        cli._resolve_exp("不存在的实验_xyz")


def test_list_exps_nonempty():
    exps = cli._list_exps()
    assert "grpo_qwen3.5-4b_gsm8k_v1" in exps


def test_list_profiles_has_h100():
    profiles = cli._list_profiles()
    assert "h100" in profiles


def test_method_completion_is_sdk_catalog_driven():
    assert "grpo" in cli._complete_method("g")
    assert "verl-grpo" in cli._complete_method("verl-")
    assert "trl-grpo" in cli._complete_method("trl-")
    assert "agent" not in cli._complete_method("")


def test_validate_exp_clean_on_real_experiment():
    errors, _ = cli._validate_exp("experiments/grpo_qwen3.5-4b_gsm8k_v1")
    assert errors == []


def test_custom_validation_does_not_run_nemo_config_parser(tmp_path, monkeypatch):
    import json

    from nemo_rl_lab.migrate_v2 import recipe_lock

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    exp = tmp_path / "experiments" / "custom"
    exp.mkdir(parents=True)
    (exp / "method").write_text("custom\n")
    (exp / "recipe.lock.json").write_text(json.dumps(recipe_lock("custom")))
    (exp / "train.sh").write_text("#!/usr/bin/env bash\n")
    (exp / "config.yaml").write_text("this: [is: not: nemo]\n")
    assert cli._validate_exp("experiments/custom") == ([], [])


def test_verl_validation_only_requires_mapping_config(tmp_path, monkeypatch):
    import json

    from nemo_rl_lab.migrate_v2 import recipe_lock

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    exp = tmp_path / "experiments" / "verl"
    exp.mkdir(parents=True)
    (exp / "method").write_text("verl-grpo\n")
    (exp / "recipe.lock.json").write_text(json.dumps(recipe_lock("verl-grpo")))
    (exp / "config.yaml").write_text("trainer:\n  total_epochs: 1\n")
    assert cli._validate_exp("experiments/verl") == ([], [])


def test_trl_validation_requires_entrypoint_and_mapping_config(tmp_path, monkeypatch):
    import json

    from nemo_rl_lab.migrate_v2 import recipe_lock

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    exp = tmp_path / "experiments" / "trl"
    exp.mkdir(parents=True)
    (exp / "method").write_text("trl-grpo\n")
    (exp / "recipe.lock.json").write_text(json.dumps(recipe_lock("trl-grpo")))
    (exp / "train.py").write_text("# experiment reward hook\n")
    (exp / "config.yaml").write_text("max_steps: 1\n")
    assert cli._validate_exp("experiments/trl") == ([], [])


def test_methods_summary_exposes_default_and_supported_versions(capsys):
    cli.methods(None)
    output = capsys.readouterr().out
    assert "默认 nemo-rl@0.7.0 · 支持 0.7.0" in output
    assert "默认 trl@1.10.0 · 支持 1.10.0" in output


# --------------------------- config diff（_flatten）---------------------------
def test_flatten_nested_and_lists():
    flat = cli._flatten({"a": {"b": 1, "c": [10, 20]}, "d": None})
    assert flat == {"a.b": "1", "a.c[0]": "10", "a.c[1]": "20", "d": "null"}


def test_flatten_diff_keys():
    a = cli._flatten({"x": 1, "only_a": 5, "nested": {"k": "v1"}})
    b = cli._flatten({"x": 2, "only_b": 9, "nested": {"k": "v1"}})
    changed = {k for k in a if k in b and a[k] != b[k]}
    assert changed == {"x"}
    assert (set(a) - set(b)) == {"only_a"}
    assert (set(b) - set(a)) == {"only_b"}


def test_format_user_label_basic():
    assert cli._format_user_label({"username": "alice", "role": "operator"}) == (
        "用户：alice  角色：operator"
    )


def test_format_user_label_with_email():
    line = cli._format_user_label({"username": "bob", "role": "admin", "email": "bob@corp.com"})
    assert "用户：bob" in line
    assert "角色：admin" in line
    assert "邮箱：bob@corp.com" in line
