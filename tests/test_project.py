"""项目发现（starforge.yaml）与 `sf init` 脚手架单测。"""
from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from starforge_cli import cli
from starforge_cli.commands import common
from starforge_cli.project import InitError, init_project, launch_script, scaffold_root

runner = CliRunner()


def test_scaffold_resources_are_packaged():
    """wheel / editable 安装都必须带齐脚手架与平台契约入口。"""
    root = scaffold_root()
    assert (root / "launch.sh").is_file()
    assert (root / "experiment-template" / "config.yaml").is_file()
    assert (root / "project" / "configs" / "base" / "sft.yaml").is_file()
    assert (root / "project" / "gitignore").is_file()
    assert launch_script().is_file()
    assert b"starforge_sdk.launcher" in launch_script().read_bytes()


def test_project_root_walks_up(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "starforge.yaml").write_text("name: t\n", encoding="utf-8")
    nested = root / "experiments" / "e"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("SF_PROJECT_ROOT", raising=False)
    assert common.project_root() == root.resolve()


def test_project_root_env_override(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "starforge.yaml").write_text("name: t\n", encoding="utf-8")
    monkeypatch.setenv("SF_PROJECT_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    assert common.project_root() == root.resolve()


def test_project_root_env_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("SF_PROJECT_ROOT", str(tmp_path))
    with pytest.raises(typer.Exit):
        common.project_root()


def test_project_root_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SF_PROJECT_ROOT", raising=False)
    with pytest.raises(typer.Exit):
        common.project_root()


def test_init_project_layout(tmp_path):
    dest = tmp_path / "my-lab"
    root = init_project(dest, name="demo", git=True)
    assert (root / "starforge.yaml").is_file()
    marker = (root / "starforge.yaml").read_text(encoding="utf-8")
    assert "name: demo" in marker
    assert "apiVersion: forge/project/v1" in marker
    assert (root / "experiments").is_dir()
    assert (root / "configs" / "base" / "sft.yaml").is_file()
    assert (root / "configs" / "models" / "qwen3.5-4b.yaml").is_file()
    assert (root / "common" / "README.md").is_file()
    assert (root / ".gitignore").is_file()
    assert (root / "README.md").is_file()
    assert (root / ".git").is_dir()
    # launch.sh 是平台契约，不落地到用户项目（打包时由 CLI 注入）
    assert not (root / "scripts" / "launch.sh").exists()


def test_init_refuses_existing_project(tmp_path):
    dest = tmp_path / "p"
    init_project(dest, git=False)
    with pytest.raises(InitError, match="已经是"):
        init_project(dest, git=False)


def test_init_refuses_clash_paths(tmp_path):
    dest = tmp_path / "p"
    dest.mkdir()
    (dest / "experiments").mkdir()
    with pytest.raises(InitError, match="experiments"):
        init_project(dest, git=False)


def test_init_keeps_existing_readme(tmp_path):
    dest = tmp_path / "p"
    dest.mkdir()
    (dest / "README.md").write_text("# already\n", encoding="utf-8")
    init_project(dest, name="keep", git=False)
    assert (dest / "README.md").read_text(encoding="utf-8") == "# already\n"
    assert (dest / "starforge.yaml").is_file()


def test_init_cli_yes(tmp_path):
    dest = tmp_path / "cli-lab"
    res = runner.invoke(cli.app, ["init", str(dest), "--yes", "--no-git", "--name", "x"])
    assert res.exit_code == 0, res.stdout
    assert (dest / "starforge.yaml").is_file()
    assert "name: x" in (dest / "starforge.yaml").read_text(encoding="utf-8")
    assert "StarForge 项目已创建" in res.stdout


def test_init_then_new_experiment(tmp_path):
    """pip 用户路径：sf init → sf new，不依赖 CLI 源码仓里的 experiments/。"""
    from starforge_cli.new_experiment import create_experiment

    dest = tmp_path / "lab"
    init_project(dest, name="t", git=False)
    create_experiment(dest, "experiments", "my-grpo", method="nemo-rl/grpo")
    exp = dest / "experiments" / "my-grpo"
    assert (exp / "recipe.lock.json").is_file()
    assert (exp / "config.yaml").is_file()
