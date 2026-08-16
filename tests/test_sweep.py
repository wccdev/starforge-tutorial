"""超参 sweep：网格展开、批量提交循环、分组标识与停止命令。"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from starforge_cli import cli
from starforge_cli.commands import sweep as sweep_cmd
from starforge_cli.commands.sweep import MAX_VARIANTS, parse_grid

runner = CliRunner()


# ── 网格展开纯逻辑 ────────────────────────────────────────────────────────────


def test_parse_grid_cartesian_product_is_stable_ordered():
    variants = parse_grid(["policy.lr=1e-5,2e-5", "grpo.kl=0.01,0.05"])
    assert variants == [
        {"policy.lr": "1e-5", "grpo.kl": "0.01"},
        {"policy.lr": "1e-5", "grpo.kl": "0.05"},
        {"policy.lr": "2e-5", "grpo.kl": "0.01"},
        {"policy.lr": "2e-5", "grpo.kl": "0.05"},
    ]


def test_parse_grid_single_axis():
    assert parse_grid(["seed=1,2,3"]) == [{"seed": "1"}, {"seed": "2"}, {"seed": "3"}]


def test_parse_grid_rejects_bad_shapes():
    with pytest.raises(ValueError, match="格式"):
        parse_grid(["no-equals-sign"])
    with pytest.raises(ValueError, match="缺少"):
        parse_grid(["key="])
    with pytest.raises(ValueError, match="重复"):
        parse_grid(["k=1,2", "k=3"])


# ── 命令行为 ─────────────────────────────────────────────────────────────────


def test_sweep_dry_run_prints_variants_without_submitting(monkeypatch, tmp_path):
    exp = tmp_path / "experiments" / "demo"
    exp.mkdir(parents=True)
    monkeypatch.setattr(sweep_cmd.common, "ROOT", tmp_path)
    res = runner.invoke(cli.app, [
        "sweep", "experiments/demo", "-g", "policy.lr=1e-5,2e-5", "--dry-run",
    ])
    assert res.exit_code == 0, res.stdout
    assert "2 个变体" in res.stdout
    assert "policy.lr=1e-5" in res.stdout and "policy.lr=2e-5" in res.stdout
    assert "未提交" in res.stdout


def test_sweep_submits_each_variant_with_shared_sweep_meta(monkeypatch, tmp_path):
    exp = tmp_path / "experiments" / "demo"
    exp.mkdir(parents=True)
    calls: list[dict] = []

    monkeypatch.setattr(sweep_cmd.common, "ROOT", tmp_path)
    monkeypatch.setattr(sweep_cmd, "gate", lambda: None)
    monkeypatch.setattr(sweep_cmd.common, "resolve_profile", lambda *a, **k: "h200")
    monkeypatch.setattr(
        sweep_cmd, "_materialize_profile_or_exit",
        lambda exprs: ("h200", ["train:h200:1:8"], []),
    )
    monkeypatch.setattr(
        sweep_cmd.packing, "git_provenance",
        lambda *a, **k: {"git_commit": "cafe", "git_dirty": False, "config_sha": "x"},
    )
    monkeypatch.setattr(sweep_cmd, "_validate_exp", lambda *a, **k: ([], []))
    monkeypatch.setattr(sweep_cmd, "_dataset_refs_from_config", lambda *a, **k: ("", ""))

    def fake_build_spec(exp_path, **kw):
        return {"sets": kw["sets"], "project": kw["project"]}

    monkeypatch.setattr(sweep_cmd, "_build_spec_or_exit", fake_build_spec)

    def fake_submit(exp_rel, profile, root, *, project=None, reporter=None,
                    spec=None, extra_meta=None):
        calls.append({"spec": spec, "project": project, "extra_meta": extra_meta})
        return {"queued": False, "job_id": f"ray-{len(calls)}", "run_id": f"r{len(calls)}"}

    monkeypatch.setattr(sweep_cmd.api_client, "submit_via_server", fake_submit)

    res = runner.invoke(cli.app, [
        "sweep", "experiments/demo",
        "-g", "policy.lr=1e-5,2e-5",
        "-s", "grpo.max_num_steps=100",
        "--project", "my-sweep",
    ])
    assert res.exit_code == 0, res.stdout
    assert len(calls) == 2
    # 固定 --set 与变体轴合并；分组名统一
    assert calls[0]["spec"]["sets"] == ["grpo.max_num_steps=100", "policy.lr=1e-5"]
    assert calls[1]["spec"]["sets"] == ["grpo.max_num_steps=100", "policy.lr=2e-5"]
    assert {c["project"] for c in calls} == {"my-sweep"}
    # sweep_id 全变体一致，sweep_params 各自不同
    sweep_ids = {c["extra_meta"]["sweep_id"] for c in calls}
    assert len(sweep_ids) == 1 and next(iter(sweep_ids)).startswith("sweep-demo-")
    assert calls[0]["extra_meta"]["sweep_params"] == {"policy.lr": "1e-5"}
    assert calls[1]["extra_meta"]["sweep_params"] == {"policy.lr": "2e-5"}
    assert "直接提交 2 个" in res.stdout


def test_sweep_variant_cap_guards_fat_fingers(monkeypatch, tmp_path):
    exp = tmp_path / "experiments" / "demo"
    exp.mkdir(parents=True)
    monkeypatch.setattr(sweep_cmd.common, "ROOT", tmp_path)
    axes = [f"k{i}=1,2,3,4,5" for i in range(3)]  # 125 > MAX_VARIANTS
    args = ["sweep", "experiments/demo"]
    for a in axes:
        args += ["-g", a]
    res = runner.invoke(cli.app, args)
    assert res.exit_code != 0
    assert str(MAX_VARIANTS) in res.stdout + res.output


def test_stop_sweep_command_calls_endpoint(monkeypatch):
    from starforge_cli.commands import jobs as jobs_cmd

    seen = {}
    monkeypatch.setattr(jobs_cmd, "gate", lambda: None)
    monkeypatch.setattr(
        jobs_cmd.api_client, "stop_sweep_via_server",
        lambda sid: seen.update(sid=sid) or {"stopped": 3, "failed": []},
    )
    res = runner.invoke(cli.app, ["job", "stop-sweep", "sweep-demo-1", "-y"])
    assert res.exit_code == 0, res.stdout
    assert seen == {"sid": "sweep-demo-1"}
    assert "已停止 3 个作业" in res.stdout


def test_extra_meta_cannot_shadow_reserved_keys(monkeypatch, tmp_path):
    from starforge_cli import api_client

    monkeypatch.setattr(api_client, "current_server", lambda *a, **k: "http://srv")
    monkeypatch.setattr(api_client, "verify_server_compatibility", lambda *a, **k: None)
    monkeypatch.setattr(
        api_client, "git_provenance",
        lambda *a, **k: {"git_commit": "c", "git_dirty": False, "config_sha": "s"},
    )

    class _Spec:
        def to_dict(self):
            return {}

    with pytest.raises(ValueError, match="保留键"):
        api_client.submit_via_server(
            "experiments/demo", "h200", tmp_path,
            spec=_Spec(), extra_meta={"exp": "evil"},
        )
