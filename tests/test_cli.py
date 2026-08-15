"""lab CLI 命令层单测（不触发真实提交 / 网络）。

提交一律走中心化服务（server 模式），CLI 不再有本机直连集群的逻辑，
故这里只覆盖与模式无关的纯本地辅助：实验/profile 解析、config 校验、命令面组装。
"""
from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from nemo_rl_lab import cli
from nemo_rl_lab.commands import common, exp, jobs

runner = CliRunner()


# --------------------------- 实验 / profile 解析 ---------------------------
def test_resolve_exp_known():
    # 仓库内现有实验应可解析为 experiments/<name>
    assert common.resolve_exp("grpo_qwen3.5-4b_gsm8k_v1") == "experiments/grpo_qwen3.5-4b_gsm8k_v1"


def test_resolve_exp_unknown_raises():
    with pytest.raises(typer.Exit):
        common.resolve_exp("不存在的实验_xyz")


def test_list_exps_nonempty():
    assert "grpo_qwen3.5-4b_gsm8k_v1" in common.list_exps()


def test_list_profiles_has_h100():
    assert "h100" in common.list_profiles()


def test_method_completion_is_sdk_catalog_driven():
    # 叶子名前缀也能补出完整两段式标识
    assert set(common.complete_method("g")) == {"nemo-rl/grpo", "verl/grpo", "trl/grpo"}
    assert "verl/grpo" in common.complete_method("verl/")
    assert "trl/grpo" in common.complete_method("trl/")
    assert "agent" not in common.complete_method("")


def test_resolve_profile_explicit_wins():
    assert common.resolve_profile("experiments/grpo_qwen3.5-4b_gsm8k_v1", "h100") == "h100"


def test_resolve_profile_falls_back_to_cluster_file():
    # 实验目录下 cluster 文件已在 v2 迁移中显式写入。
    p = common.resolve_profile("experiments/grpo_qwen3.5-4b_gsm8k_v1", None)
    assert p in common.list_profiles()


def test_resolve_profile_unknown_fails():
    with pytest.raises(typer.Exit):
        common.resolve_profile("experiments/grpo_qwen3.5-4b_gsm8k_v1", "no-such-profile")


def test_resolve_profile_missing_everything_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    (tmp_path / "experiments" / "x").mkdir(parents=True)
    with pytest.raises(typer.Exit):
        common.resolve_profile("experiments/x", None)


# --------------------------- config 校验 ---------------------------
def test_validate_exp_clean_on_real_experiment():
    errors, _ = exp._validate_exp("experiments/grpo_qwen3.5-4b_gsm8k_v1")
    assert errors == []


def test_custom_validation_does_not_run_nemo_config_parser(tmp_path, monkeypatch):
    import json

    from nemo_rl_lab.recipe_lock import recipe_lock

    monkeypatch.setattr(common, "ROOT", tmp_path)
    e = tmp_path / "experiments" / "custom"
    e.mkdir(parents=True)
    (e / "method").write_text("custom/custom\n")
    (e / "recipe.lock.json").write_text(json.dumps(recipe_lock("custom/custom")))
    (e / "train.sh").write_text("#!/usr/bin/env bash\n")
    (e / "config.yaml").write_text("this: [is: not: nemo]\n")
    assert exp._validate_exp("experiments/custom") == ([], [])


def test_verl_validation_only_requires_mapping_config(tmp_path, monkeypatch):
    import json

    from nemo_rl_lab.recipe_lock import recipe_lock

    monkeypatch.setattr(common, "ROOT", tmp_path)
    e = tmp_path / "experiments" / "verl"
    e.mkdir(parents=True)
    (e / "method").write_text("verl/grpo\n")
    (e / "recipe.lock.json").write_text(json.dumps(recipe_lock("verl/grpo")))
    (e / "config.yaml").write_text("trainer:\n  total_epochs: 1\n")
    assert exp._validate_exp("experiments/verl") == ([], [])


def test_trl_validation_requires_entrypoint_and_mapping_config(tmp_path, monkeypatch):
    import json

    from nemo_rl_lab.recipe_lock import recipe_lock

    monkeypatch.setattr(common, "ROOT", tmp_path)
    e = tmp_path / "experiments" / "trl"
    e.mkdir(parents=True)
    (e / "method").write_text("trl/grpo\n")
    (e / "recipe.lock.json").write_text(json.dumps(recipe_lock("trl/grpo")))
    (e / "train.py").write_text("# experiment reward hook\n")
    (e / "config.yaml").write_text("max_steps: 1\n")
    assert exp._validate_exp("experiments/trl") == ([], [])


def test_methods_summary_exposes_default_and_supported_versions(capsys):
    exp.methods(None)
    output = capsys.readouterr().out
    assert "默认 nemo-rl@0.7.0 · 支持 0.7.0" in output
    assert "默认 trl@1.10.0 · 支持 1.10.0" in output


# --------------------------- status 展示 ---------------------------
def test_format_user_label_basic():
    assert jobs._format_user_label({"username": "alice", "role": "operator"}) == (
        "用户：alice  角色：operator"
    )


def test_format_user_label_with_email():
    line = jobs._format_user_label({"username": "bob", "role": "admin", "email": "bob@corp.com"})
    assert "用户：bob" in line
    assert "角色：admin" in line
    assert "邮箱：bob@corp.com" in line


# --------------------------- 命令面契约 ---------------------------
# 精简后的完整顶层命令面；新增/复活命令必须先过这条测试（防止 CLI 再度膨胀）。
EXPECTED_COMMANDS = {
    "login", "logout",
    "ls", "new", "methods", "validate",
    "submit", "export", "eval", "clean",
    "status",
}
EXPECTED_GROUPS = {"job", "dataset", "admin"}


def test_command_surface_is_closed():
    commands = {c.name or c.callback.__name__ for c in cli.app.registered_commands}
    groups = {g.name for g in cli.app.registered_groups}
    assert commands == EXPECTED_COMMANDS
    assert groups == EXPECTED_GROUPS


@pytest.mark.parametrize("removed", ["model", "completion", "sync-base", "migrate-v2", "diff",
                                     "runs", "logs", "whoami", "quota", "doctor", "prepare"])
def test_removed_commands_stay_removed(removed):
    res = runner.invoke(cli.app, [removed, "--help"])
    assert res.exit_code != 0


def test_dataset_group_includes_prepare():
    res = runner.invoke(cli.app, ["dataset", "--help"])
    assert res.exit_code == 0
    assert "prepare" in res.stdout
    assert "push" in res.stdout
    assert "visibility" in res.stdout


def test_dataset_push_exposes_public_flag():
    res = runner.invoke(cli.app, ["dataset", "push", "--help"])
    assert res.exit_code == 0
    assert "--public" in res.stdout


def test_dataset_push_forwards_namespace_and_visibility(monkeypatch, tmp_path):
    """push 把数据集名与 --public 原样交给服务端（owner 归属由服务端裁决）。"""
    from nemo_rl_lab import api_client
    from nemo_rl_lab.commands import dataset as dataset_cmd

    (tmp_path / "train.jsonl").write_text('{"q":1}\n', encoding="utf-8")
    seen: dict = {}

    def fake_push(dataset, version, root, files, visibility=None, server=None):
        seen.update(dataset=dataset, version=version, visibility=visibility, n=len(files))
        return f"alice/{dataset}"

    monkeypatch.setattr(dataset_cmd, "gate", lambda: None)
    monkeypatch.setattr(api_client, "dataset_push", fake_push)
    res = runner.invoke(cli.app, ["dataset", "push", "qa-rl", "v1", str(tmp_path), "--public"])
    assert res.exit_code == 0, res.stdout
    assert seen == {"dataset": "qa-rl", "version": "v1", "visibility": "public", "n": 1}
    assert "alice/qa-rl@v1" in res.stdout
    # 引用提示带完整 ID 与数据目录环境变量
    assert "QA_RL_DATA_DIR" in res.stdout


def test_dataset_visibility_requires_exactly_one_flag(monkeypatch):
    from nemo_rl_lab.commands import dataset as dataset_cmd

    monkeypatch.setattr(dataset_cmd, "gate", lambda: None)
    assert runner.invoke(cli.app, ["dataset", "visibility", "alice/qa-rl"]).exit_code != 0
    assert runner.invoke(
        cli.app, ["dataset", "visibility", "alice/qa-rl", "--public", "--private"]
    ).exit_code != 0


def test_job_logs_accepts_optional_id():
    res = runner.invoke(cli.app, ["job", "logs", "--help"])
    assert res.exit_code == 0
