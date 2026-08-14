from __future__ import annotations

import json

import pytest

from nemo_rl_lab.migrate_v2 import apply_migration, check_repo


def _profile(repo, name="h100"):
    path = repo / "cluster" / name
    path.mkdir(parents=True)
    (path / "overrides.conf").write_text("cluster.num_nodes=1\ncluster.gpus_per_node=1\n")


def _exp(repo, name, *, config="", method="", cluster="h100", train=False):
    path = repo / "experiments" / name
    path.mkdir(parents=True)
    if config:
        (path / "config.yaml").write_text(config)
    if method:
        (path / "method").write_text(f"{method}\n")
    if cluster:
        (path / "cluster").write_text(f"{cluster}\n")
    if train:
        (path / "train.sh").write_text("#!/usr/bin/env bash\n")
    return path


def test_check_and_write_migrates_nemo_and_custom(tmp_path):
    repo = tmp_path
    _profile(repo)
    nemo = _exp(repo, "grpo_demo", config="defaults:\n  - ../../configs/base/grpo_math_1B.yaml\n")
    custom = _exp(repo, "external", train=True)
    items = check_repo(repo)
    assert [(item.recipe, item.needs_write) for item in items] == [("custom", True), ("grpo", True)]

    apply_migration(repo, items)
    assert (nemo / "method").read_text().strip() == "grpo"
    assert (custom / "method").read_text().strip() == "custom"
    lock = json.loads((nemo / "recipe.lock.json").read_text())
    assert lock["apiVersion"] == "lab/recipe-lock/v2"
    assert lock["recipe"]["framework_version"] == "0.7.0"


def test_migration_includes_smoke_experiments(tmp_path):
    _profile(tmp_path)
    smoke = tmp_path / "smoke" / "verl-sft"
    smoke.mkdir(parents=True)
    (smoke / "method").write_text("verl-sft\n")
    (smoke / "cluster").write_text("h100\n")

    items = check_repo(tmp_path)
    assert [(item.path, item.recipe, item.needs_write) for item in items] == [
        (smoke, "verl-sft", True)
    ]
    apply_migration(tmp_path, items)
    lock = json.loads((smoke / "recipe.lock.json").read_text())
    assert lock["recipe"]["framework"] == "verl"
    assert lock["recipe"]["framework_version"] == "0.8.0"


def test_check_reports_unclassifiable_experiment(tmp_path):
    _profile(tmp_path)
    _exp(tmp_path, "mystery", config="x: 1\n")
    assert "无法唯一判定" in check_repo(tmp_path)[0].error


def test_check_reports_unknown_method(tmp_path):
    _profile(tmp_path)
    _exp(tmp_path, "bad", config="x: 1\n", method="unknown")
    assert "未知" in check_repo(tmp_path)[0].error


def test_check_reports_missing_resource_config(tmp_path):
    path = tmp_path / "experiments" / "grpo_demo"
    path.mkdir(parents=True)
    (path / "method").write_text("grpo\n")
    assert "cluster" in check_repo(tmp_path)[0].error


def test_write_is_all_or_nothing_when_check_has_errors(tmp_path):
    _profile(tmp_path)
    good = _exp(tmp_path, "grpo_ok", config="defaults:\n  - ../../configs/base/grpo_math_1B.yaml\n")
    _exp(tmp_path, "unknown", config="x: 1\n")
    items = check_repo(tmp_path)
    with pytest.raises(ValueError, match="不写入部分结果"):
        apply_migration(tmp_path, items)
    assert not (good / "method").exists()
