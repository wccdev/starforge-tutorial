"""new_experiment 跨平台逻辑单测。"""
from __future__ import annotations

import json

import pytest

from nemo_rl_lab.new_experiment import NewExperimentError, create_experiment
from nemo_rl_lab.recipe_lock import LOCK_FILE, recipe_lock


def _lock(path, recipe):
    import json

    (path / LOCK_FILE).write_text(json.dumps(recipe_lock(recipe)))


def test_create_grpo_from_template(tmp_path):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text("defaults:\n  - ../../configs/base/grpo_math_1B.yaml\n", encoding="utf-8")
    (repo / "experiments").mkdir()

    create_experiment(repo, "experiments", "test_exp_v1", method="nemo-rl/grpo")
    dest = repo / "experiments" / "test_exp_v1"
    assert dest.is_dir()
    assert (dest / "config.yaml").is_file()
    assert not (dest / "method").exists()  # recipe 声明只在锁文件里
    assert not (dest / "cluster").exists()  # 硬件 profile 提交时用 --profile 指定
    lock = json.loads((dest / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["apiVersion"] == "lab/recipe-lock/v2"
    assert lock["recipe"]["name"] == "nemo-rl/grpo"
    assert lock["recipe"]["framework_version"] == "0.7.0"


def test_fork_patches_swanlab_and_readme(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "experiments" / "src_exp"
    src.mkdir(parents=True)
    (src / "config.yaml").write_text(
        "swanlab:\n  project: \"old\"\n  name: \"old\"\nother: 1\n",
        encoding="utf-8",
    )
    (src / "README.md").write_text("# old title\n", encoding="utf-8")
    _lock(src, "nemo-rl/grpo")
    (repo / "experiments").mkdir(exist_ok=True)

    create_experiment(repo, "experiments", "new_exp", src="src_exp")
    cfg = (repo / "experiments" / "new_exp" / "config.yaml").read_text(encoding="utf-8")
    assert 'project: "new_exp"' in cfg
    assert 'name: "new_exp"' in cfg
    assert (repo / "experiments" / "new_exp" / "README.md").read_text(encoding="utf-8").startswith("# new_exp")


def test_fork_rejects_source_without_recipe_metadata(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "experiments" / "legacy"
    src.mkdir(parents=True)
    (src / "config.yaml").write_text("defaults: []\n")
    with pytest.raises(NewExperimentError, match="recipe"):
        create_experiment(repo, "experiments", "copy", src="legacy")


def test_create_sft_method_from_recipe_template(tmp_path):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text(
        "defaults:\n  - ../../configs/base/grpo_math_1B.yaml\n\ngrpo:\n  x: 1\n\nloss_fn:\n  y: 2\n\nlogger:\n  z: 3\n",
        encoding="utf-8",
    )
    (repo / "experiments").mkdir()
    create_experiment(repo, "experiments", "sft_test", method="nemo-rl/sft")
    cfg = (repo / "experiments" / "sft_test" / "config.yaml").read_text(encoding="utf-8")
    assert "defaults:" in cfg
    lock = json.loads((repo / "experiments" / "sft_test" / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["recipe"]["name"] == "nemo-rl/sft"


def test_create_custom_uses_recipe_metadata_not_framework_marker(tmp_path):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text("defaults: []\n", encoding="utf-8")
    (repo / "experiments").mkdir()
    create_experiment(repo, "experiments", "custom_test", method="custom/custom")
    dest = repo / "experiments" / "custom_test"
    lock = json.loads((dest / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["recipe"]["name"] == "custom/custom"
    assert (dest / "train.sh").is_file()
    assert not (dest / "framework").exists()


def test_create_verl_recipe_copies_its_own_template(tmp_path):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text("wrong: nemo\n", encoding="utf-8")
    (repo / "experiments").mkdir()
    create_experiment(repo, "experiments", "verl_test", method="verl/grpo")
    dest = repo / "experiments" / "verl_test"
    assert "trainer:" in (dest / "config.yaml").read_text(encoding="utf-8")
    assert (dest / "eval.py").is_file()
    lock = json.loads((dest / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["recipe"]["name"] == "verl/grpo"


def test_create_trl_recipe_copies_framework_template_and_pins_version(tmp_path):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text("wrong: nemo\n", encoding="utf-8")
    (repo / "experiments").mkdir()
    create_experiment(
        repo,
        "experiments",
        "trl_test",
        method="trl/grpo",
        framework_version="1.10.0",
    )
    dest = repo / "experiments" / "trl_test"
    assert "num_generations:" in (dest / "config.yaml").read_text(encoding="utf-8")
    assert (dest / "train.py").is_file()
    lock = json.loads((dest / LOCK_FILE).read_text(encoding="utf-8"))
    assert lock["recipe"]["framework"] == "trl"
    assert lock["recipe"]["framework_version"] == "1.10.0"


def test_recipe_template_failure_is_atomic(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text("defaults: []\n", encoding="utf-8")
    (repo / "experiments").mkdir()

    def fail(*_args, **_kwargs):
        raise NewExperimentError("template invalid")

    monkeypatch.setattr("nemo_rl_lab.new_experiment._validate_recipe_template", fail)
    with pytest.raises(NewExperimentError, match="template invalid"):
        create_experiment(repo, "experiments", "never_visible", method="nemo-rl/grpo")
    assert not (repo / "experiments" / "never_visible").exists()
