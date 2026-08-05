"""`lab model` 子命令单测（不触发真实下载 / 网络）。

这组命令的价值在于「把三种网络现实收口成一个入口」，所以测的重点是**分派是否正确**：
选错通路或漏传 --relay，在内网现场表现为「卡住几分钟然后超时」，很难定位。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from nemo_rl_lab import cli

runner = CliRunner()


@pytest.fixture
def captured_cmd():
    """拦下 cli._run，只记录命令行，不真的起子进程。"""
    calls: list[list[str]] = []

    def fake_run(cmd, env=None, cwd=None):
        calls.append([str(c) for c in cmd])
        return 0

    with patch.object(cli, "_run", side_effect=fake_run):
        yield calls


def _invoke(*args):
    return runner.invoke(cli.app, ["model", *args])


# ---------------------------------------------------------------- pull 分派
def test_pull_direct_uses_download_models(captured_cmd, tmp_path):
    res = _invoke("pull", "Qwen/Qwen3.5-9B", "--hf-home", str(tmp_path))
    assert res.exit_code == 0
    cmd = captured_cmd[0]
    assert cmd[1].endswith("download_models.py")
    assert "--only" in cmd and "Qwen/Qwen3.5-9B" in cmd
    assert "--hf-home" in cmd and str(tmp_path) in cmd


def test_pull_relay_uses_relay_script(captured_cmd, tmp_path):
    res = _invoke("pull", "Qwen/Qwen3.5-9B", "--via", "relay",
                  "--relay", "root@10.0.0.2", "--hf-home", str(tmp_path))
    assert res.exit_code == 0
    cmd = captured_cmd[0]
    assert cmd[1].endswith("download_via_relay.py")
    assert cmd[cmd.index("--relay") + 1] == "root@10.0.0.2"


def test_pull_relay_reads_env_fallback(captured_cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_RELAY", "root@relay.internal")
    res = _invoke("pull", "--via", "relay", "--hf-home", str(tmp_path))
    assert res.exit_code == 0
    cmd = captured_cmd[0]
    assert cmd[cmd.index("--relay") + 1] == "root@relay.internal"


def test_pull_relay_without_target_fails(captured_cmd, monkeypatch):
    """漏传中继机必须当场报错，而不是退化成直连然后在内网里静默超时。"""
    monkeypatch.delenv("LAB_RELAY", raising=False)
    res = _invoke("pull", "--via", "relay")
    assert res.exit_code != 0
    assert captured_cmd == []


def test_pull_nexus_requires_endpoint(captured_cmd, monkeypatch):
    monkeypatch.delenv("LAB_NEXUS_HF_ENDPOINT", raising=False)
    res = _invoke("pull", "--via", "nexus")
    assert res.exit_code != 0
    assert captured_cmd == []


def test_pull_nexus_passes_endpoint(captured_cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_NEXUS_HF_ENDPOINT", "http://nexus.internal/repository/hf")
    res = _invoke("pull", "--via", "nexus", "--hf-home", str(tmp_path))
    assert res.exit_code == 0
    cmd = captured_cmd[0]
    assert cmd[cmd.index("--endpoint") + 1] == "http://nexus.internal/repository/hf"


def test_pull_unknown_via_fails(captured_cmd):
    res = _invoke("pull", "--via", "carrier-pigeon")
    assert res.exit_code != 0
    assert captured_cmd == []


def test_pull_retries_and_workers_flags(captured_cmd, tmp_path):
    res = _invoke("pull", "--hf-home", str(tmp_path), "--retries", "3", "--workers", "2")
    assert res.exit_code == 0
    cmd = captured_cmd[0]
    assert cmd[cmd.index("--retries") + 1] == "3"
    # direct 走 download_models.py，并发参数叫 --max-workers
    assert cmd[cmd.index("--max-workers") + 1] == "2"
    # 关键回归点：带值的选项不能被变长位置参数 repo 吃掉（否则会去下载一个叫「3」的模型）
    assert "--only" not in cmd


def test_pull_relay_maps_workers_flag(captured_cmd, tmp_path):
    """两个脚本对并发的命名不同：relay 是 ssh 通道数 --workers。"""
    _invoke("pull", "--via", "relay", "--relay", "u@h",
            "--hf-home", str(tmp_path), "--workers", "4")
    cmd = captured_cmd[0]
    assert cmd[cmd.index("--workers") + 1] == "4"
    assert "--max-workers" not in cmd


def test_pull_rejects_unknown_option(captured_cmd, tmp_path):
    """未声明的选项应当当场报错，而不是被静默吃成模型名。"""
    res = _invoke("pull", "--hf-home", str(tmp_path), "--no-such-flag")
    assert res.exit_code != 0
    assert captured_cmd == []


def test_pull_daemon_and_list_flags(captured_cmd, tmp_path):
    _invoke("pull", "--hf-home", str(tmp_path), "--daemon", "--list")
    cmd = captured_cmd[0]
    assert "--daemon" in cmd and "--list" in cmd


# ---------------------------------------------------------------- install
def test_install_dispatches(captured_cmd, tmp_path):
    src = tmp_path / "flat"
    src.mkdir()
    res = _invoke("install", str(src), "--hf-home", str(tmp_path / "cache"), "--dry-run")
    assert res.exit_code == 0
    cmd = captured_cmd[0]
    assert cmd[1].endswith("install_to_hf_cache.py")
    assert "--dry-run" in cmd
    assert cmd[cmd.index("--src") + 1] == str(src)


# ---------------------------------------------------------------- ls
def test_ls_empty_cache(tmp_path):
    res = _invoke("ls", "--hf-home", str(tmp_path))
    assert res.exit_code == 0
    assert "不存在" in res.stdout


def test_ls_lists_repo_ids_and_sha(tmp_path):
    repo = tmp_path / "hub" / "models--Qwen--Qwen3.5-9B"
    (repo / "refs").mkdir(parents=True)
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abcdef1234567890")
    (repo / "snapshots" / "abc" / "model.safetensors").write_bytes(b"x" * 2048)

    res = _invoke("ls", "--hf-home", str(tmp_path))
    assert res.exit_code == 0
    # 缓存目录名 models--Qwen--Qwen3.5-9B 要还原成 config 里写的 repo id
    assert "Qwen/Qwen3.5-9B" in res.stdout
    assert "abcdef123456" in res.stdout


def test_ls_ignores_non_model_dirs(tmp_path):
    hub = tmp_path / "hub"
    (hub / "datasets--foo--bar").mkdir(parents=True)
    (hub / ".locks").mkdir(parents=True)
    res = _invoke("ls", "--hf-home", str(tmp_path))
    assert res.exit_code == 0
    assert "没有模型" in res.stdout


def test_model_script_missing_fails():
    with patch.object(cli, "SCRIPTS", cli.ROOT / "no-such-dir"):
        with pytest.raises(typer.Exit):
            cli._model_script("download_models.py")
