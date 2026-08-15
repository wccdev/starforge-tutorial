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


def test_list_profiles_reads_server_registry(monkeypatch):
    """profile 注册表已服务端化：列表来自 /api/cluster/status，失败时静默为空。"""
    from nemo_rl_lab import api_client

    monkeypatch.setattr(
        api_client, "cluster_status_via_server",
        lambda *a, **k: {"profiles": [{"name": "h200"}, {"name": "h100"}]},
    )
    assert common.list_profiles() == ["h100", "h200"]

    def boom(*a, **k):
        raise RuntimeError("server unreachable")

    monkeypatch.setattr(api_client, "cluster_status_via_server", boom)
    assert common.list_profiles() == []


def test_method_completion_is_sdk_catalog_driven():
    # 叶子名前缀也能补出完整两段式标识
    assert set(common.complete_method("g")) == {"nemo-rl/grpo", "verl/grpo", "trl/grpo"}
    assert "verl/grpo" in common.complete_method("verl/")
    assert "trl/grpo" in common.complete_method("trl/")
    assert "agent" not in common.complete_method("")


def test_resolve_profile_explicit_wins():
    assert common.resolve_profile("experiments/grpo_qwen3.5-4b_gsm8k_v1", "h100") == "h100"


def test_resolve_profile_name_passes_through_without_local_validation(monkeypatch):
    """profile 名的合法性由服务端注册表裁决，客户端只透传选择。"""
    monkeypatch.setattr(common, "list_profiles", lambda: [])
    assert common.resolve_profile("experiments/grpo_qwen3.5-4b_gsm8k_v1", "some-new-profile") == "some-new-profile"


def test_resolve_profile_reads_legacy_cluster_file(tmp_path, monkeypatch):
    """旧实验遗留的 cluster 标注文件作为兼容回退。"""
    monkeypatch.setattr(common, "ROOT", tmp_path)
    e = tmp_path / "experiments" / "legacy"
    e.mkdir(parents=True)
    (e / "cluster").write_text("h200-2g\n", encoding="utf-8")
    assert common.resolve_profile("experiments/legacy", None) == "h200-2g"


def test_resolve_profile_missing_everything_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "list_profiles", lambda: [])
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
    (e / "recipe.lock.json").write_text(json.dumps(recipe_lock("verl/grpo")))
    (e / "config.yaml").write_text("trainer:\n  total_epochs: 1\n")
    assert exp._validate_exp("experiments/verl") == ([], [])


def test_trl_validation_requires_entrypoint_and_mapping_config(tmp_path, monkeypatch):
    import json

    from nemo_rl_lab.recipe_lock import recipe_lock

    monkeypatch.setattr(common, "ROOT", tmp_path)
    e = tmp_path / "experiments" / "trl"
    e.mkdir(parents=True)
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
EXPECTED_GROUPS = {"job", "dataset", "admin", "plugin", "recipe"}


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


def test_dataset_push_puts_with_server_signed_headers(monkeypatch, tmp_path):
    """PUT 的 Content-Type 必须与服务端预签名的一致，一律照抄下发的 headers。

    历史 bug：index.json 用 application/json 而签名是 octet-stream，
    对象存储回 403 SignatureDoesNotMatch。"""
    import urllib.request

    from nemo_rl_lab import api_client

    (tmp_path / "train.jsonl").write_text('{"q":1}\n', encoding="utf-8")

    signed = {"Content-Type": "application/octet-stream"}
    puts: list[dict] = []

    monkeypatch.setattr(api_client, "current_server", lambda *a, **k: "http://srv")
    monkeypatch.setattr(
        api_client, "api_post",
        lambda path, body, server=None: {
            "dataset": "alice/qa-rl", "upload_url": "http://s3/put",
            "headers": dict(signed),
        },
    )
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=0: puts.append(dict(req.headers)) or None,
    )
    api_client.dataset_push("qa-rl", "v1", tmp_path, [tmp_path / "train.jsonl"])
    assert len(puts) == 2  # train.jsonl + index.json
    for headers in puts:
        assert headers.get("Content-type") == "application/octet-stream"


def test_dataset_refs_read_from_experiment_config(monkeypatch, tmp_path):
    """数据集引用声明在实验 config（data.{train,validation}.dataset），submit 自动拾取。"""
    from nemo_rl_lab.commands import submit as submit_cmd

    e = tmp_path / "experiments" / "demo"
    e.mkdir(parents=True)
    (e / "config.yaml").write_text(
        "data:\n"
        "  train:\n"
        "    dataset: aiden_lu/gsm8k@v1\n"
        "    data_path: ${oc.env:GSM8K_DATA_DIR}/train.jsonl\n"
        "  validation:\n"
        "    dataset: aiden_lu/gsm8k@v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(submit_cmd.common, "ROOT", tmp_path)
    assert submit_cmd._dataset_refs_from_config("experiments/demo") == (
        "aiden_lu/gsm8k@v1", "aiden_lu/gsm8k@v1"
    )


def test_dataset_refs_absent_or_broken_config_is_silent(monkeypatch, tmp_path):
    """没 config、没声明、config 解析失败都返回空——报错归校验环节，这里不重复拦。"""
    from nemo_rl_lab.commands import submit as submit_cmd

    (tmp_path / "experiments" / "none").mkdir(parents=True)
    plain = tmp_path / "experiments" / "plain"
    plain.mkdir()
    (plain / "config.yaml").write_text("data:\n  train:\n    data_path: /x.jsonl\n", encoding="utf-8")
    broken = tmp_path / "experiments" / "broken"
    broken.mkdir()
    (broken / "config.yaml").write_text(":\n  - [", encoding="utf-8")

    monkeypatch.setattr(submit_cmd.common, "ROOT", tmp_path)
    assert submit_cmd._dataset_refs_from_config("experiments/none") == ("", "")
    assert submit_cmd._dataset_refs_from_config("experiments/plain") == ("", "")
    assert submit_cmd._dataset_refs_from_config("experiments/broken") == ("", "")


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
