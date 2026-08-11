"""集群侧 launcher：由 JobSpec 驱动的启动决策（阶段 1）。

改造前这些决策全在 scripts/_run_experiment.sh 里用 bash 表达，服务端看不见，
加一种后训练方法就要改那段 shell。这组测试锁定搬到 Python 后的等价行为。
"""
from __future__ import annotations

import json

import pytest

from nemo_rl_lab.contract import SPEC_FILE_PATH, JobSpec
from nemo_rl_lab.launcher import (
    LaunchError,
    build_command,
    build_overrides,
    load_spec,
    resolve_entrypoint,
    train_output_dir,
)


def _spec_dict(**over):
    body = {
        "recipe": {"name": "grpo", "version": "0.7.0"},
        "source": {"exp": "experiments/demo_v1"},
        "hyperparams": {"max_num_steps": 300, "reference_policy_kl_penalty": 0.02},
    }
    body.update(over)
    return {"apiVersion": "lab/v1", "kind": "TrainingJob", "metadata": {"name": "j"}, "spec": body}


def _workdir(tmp_path, *, spec=None, profile="h200", with_config=True):
    wd = tmp_path / "work"
    exp = wd / "experiments" / "demo_v1"
    exp.mkdir(parents=True)
    if with_config:
        (exp / "config.yaml").write_text("x: 1\n")
    conf = wd / "cluster" / profile
    conf.mkdir(parents=True)
    (conf / "overrides.conf").write_text(
        "# 注释\n\ncluster.num_nodes=8\ncluster.gpus_per_node=8\npolicy.dtensor_cfg.enabled=true\n"
    )
    payload = spec if spec is not None else _spec_dict()
    p = wd / SPEC_FILE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return wd


def _env(**over):
    base = {
        "CLUSTER_PROFILE": "h200",
        "NEMO_RL_DIR": "/opt/NeMo-RL",
        "OUTPUT_ROOT": "/mnt/out",
        "RUN_USER": "alice",
        "NRL_RUN_ID": "demo_v1-alice-20260811-120000",
        "LAB_CLUSTER_NUM_NODES": "1",
        "LAB_CLUSTER_GPUS_PER_NODE": "2",
    }
    base.update(over)
    return base


# ── spec 加载 ────────────────────────────────────────────────────────────────


def test_missing_spec_file_gives_actionable_error(tmp_path):
    (tmp_path / "work").mkdir()
    with pytest.raises(LaunchError, match="作业规格"):
        load_spec(tmp_path / "work")


def test_spec_roundtrips_from_upload_package(tmp_path):
    spec = load_spec(_workdir(tmp_path))
    assert spec.recipe_name == "grpo"
    assert spec.exp == "experiments/demo_v1"


# ── 产物目录：与 scripts/_output_paths.sh 及服务端 build_output_dir 三方一致 ──


def test_centralised_output_dir_matches_shell_rule(tmp_path):
    exp_dir = tmp_path / "experiments" / "demo_v1"
    got = train_output_dir("demo_v1", exp_dir, _env())
    assert got == "/mnt/out/alice/demo_v1/demo_v1-alice-20260811-120000"


def test_local_run_without_output_root_falls_back_to_exp_dir(tmp_path):
    exp_dir = tmp_path / "experiments" / "demo_v1"
    got = train_output_dir("demo_v1", exp_dir, _env(OUTPUT_ROOT=""))
    assert got == str(exp_dir / "outputs")


def test_output_root_trailing_slash_is_normalised(tmp_path):
    got = train_output_dir("demo_v1", tmp_path, _env(OUTPUT_ROOT="/mnt/out/"))
    assert got.startswith("/mnt/out/alice/")


# ── 入口解析 ─────────────────────────────────────────────────────────────────


def test_nemo_rl_entrypoint_resolved_from_recipe(tmp_path, monkeypatch):
    wd = _workdir(tmp_path)
    nemo = tmp_path / "nemo"
    (nemo / "examples").mkdir(parents=True)
    (nemo / "examples" / "run_grpo.py").write_text("")
    entry = resolve_entrypoint(load_spec(wd), wd, _env(NEMO_RL_DIR=str(nemo)))
    assert entry == nemo / "examples" / "run_grpo.py"


def test_missing_nemo_rl_dir_is_reported_clearly(tmp_path):
    wd = _workdir(tmp_path)
    with pytest.raises(LaunchError, match="NEMO_RL_DIR"):
        resolve_entrypoint(load_spec(wd), wd, _env(NEMO_RL_DIR=""))


def test_exp_local_entrypoint_for_recipes_that_need_one(tmp_path):
    """opsd 的入口在实验目录内（它要自定义 Dataset 塞参考解 token）。"""
    wd = _workdir(tmp_path, spec=_spec_dict(recipe={"name": "opsd"}))
    (wd / "experiments" / "demo_v1" / "run.py").write_text("")
    entry = resolve_entrypoint(load_spec(wd), wd, _env())
    assert entry == wd / "experiments" / "demo_v1" / "run.py"


def test_recipe_requiring_exp_entrypoint_fails_loudly_when_absent(tmp_path):
    wd = _workdir(tmp_path, spec=_spec_dict(recipe={"name": "opsd"}))
    with pytest.raises(LaunchError, match="run.py"):
        resolve_entrypoint(load_spec(wd), wd, _env())


def test_explicit_entrypoint_override_wins(tmp_path):
    wd = _workdir(tmp_path, spec=_spec_dict(source={"exp": "experiments/demo_v1", "entrypoint": "custom/train.py"}))
    (wd / "custom").mkdir()
    (wd / "custom" / "train.py").write_text("")
    assert resolve_entrypoint(load_spec(wd), wd, _env()) == wd / "custom" / "train.py"


# ── override 装配：优先级与权威拓扑 ───────────────────────────────────────────


def test_authoritative_topology_overrides_uploaded_conf(tmp_path):
    """服务端下发的拓扑必须覆盖上传文件里的，否则用户改文件就能绕过配额记账。"""
    wd = _workdir(tmp_path)
    ov = build_overrides(load_spec(wd), wd, "/out", _env())
    assert "cluster.num_nodes=1" in ov
    assert "cluster.gpus_per_node=2" in ov
    assert "cluster.num_nodes=8" not in ov  # 上传文件里谎报的 8 节点
    assert "cluster.gpus_per_node=8" not in ov


def test_non_topology_profile_overrides_are_kept(tmp_path):
    wd = _workdir(tmp_path)
    ov = build_overrides(load_spec(wd), wd, "/out", _env())
    assert "policy.dtensor_cfg.enabled=true" in ov


def test_conf_comments_and_blank_lines_ignored(tmp_path):
    wd = _workdir(tmp_path)
    ov = build_overrides(load_spec(wd), wd, "/out", _env())
    assert not any(o.startswith("#") or not o for o in ov)


def test_local_run_keeps_uploaded_topology(tmp_path):
    """无服务端权威拓扑（本地直跑）时行为不变，仍用 overrides.conf。"""
    wd = _workdir(tmp_path)
    ov = build_overrides(load_spec(wd), wd, "/out", _env(LAB_CLUSTER_NUM_NODES="", LAB_CLUSTER_GPUS_PER_NODE=""))
    assert "cluster.num_nodes=8" in ov


def test_hyperparams_become_config_overrides(tmp_path):
    """服务端理解超参的具体体现：它知道每个超参落到配置的哪个位置。"""
    wd = _workdir(tmp_path)
    ov = build_overrides(load_spec(wd), wd, "/out", _env())
    assert "grpo.max_num_steps=300" in ov
    assert "loss_fn.reference_policy_kl_penalty=0.02" in ov


def test_output_dir_overrides_are_injected(tmp_path):
    wd = _workdir(tmp_path)
    ov = build_overrides(load_spec(wd), wd, "/out/run1", _env())
    assert "checkpointing.checkpoint_dir=/out/run1" in ov
    assert "logger.log_dir=/out/run1/logs" in ov


# ── 完整命令 ─────────────────────────────────────────────────────────────────


def test_build_command_includes_config_and_overrides(tmp_path):
    wd = _workdir(tmp_path)
    nemo = tmp_path / "nemo"
    (nemo / "examples").mkdir(parents=True)
    (nemo / "examples" / "run_grpo.py").write_text("")
    cmd = build_command(load_spec(wd), wd, _env(NEMO_RL_DIR=str(nemo)))
    assert cmd[1].endswith("run_grpo.py")
    assert "--config" in cmd
    assert any(c.startswith("grpo.max_num_steps=") for c in cmd)


def test_build_command_omits_config_flag_when_experiment_has_none(tmp_path):
    wd = _workdir(tmp_path, with_config=False)
    nemo = tmp_path / "nemo"
    (nemo / "examples").mkdir(parents=True)
    (nemo / "examples" / "run_grpo.py").write_text("")
    cmd = build_command(load_spec(wd), wd, _env(NEMO_RL_DIR=str(nemo)))
    assert "--config" not in cmd


def test_legacy_spec_gets_no_recipe_overrides(tmp_path):
    """legacy 作业没有 recipe 语义，超参不该被翻译。"""
    wd = _workdir(tmp_path, spec=_spec_dict(recipe={"name": "legacy"}, hyperparams={"max_num_steps": 5}))
    spec = JobSpec.from_dict(json.loads((wd / SPEC_FILE_PATH).read_text()))
    ov = build_overrides(spec, wd, "/out", _env())
    assert not any(o.startswith("grpo.") for o in ov)
