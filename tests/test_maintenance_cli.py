"""`lab admin maintenance` 的输出契约。

这几条命令的产出会被管理员当成「能不能重启集群」的判断依据，所以最要紧的不是它调对了
接口，而是**看错了不会重启**：还有作业占卡时必须显眼地说不能重启，而不是把一行
`safe_to_restart: false` 混在 JSON 里让人看漏。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nemo_rl_lab import cli

runner = CliRunner()


def _invoke(*args, reply: dict):
    with patch("nemo_rl_lab.cli_login._admin_call", return_value=reply) as call:
        res = runner.invoke(cli.app, ["admin", "maintenance", *args])
    return res, call


# ---------------------------------------------------------------- status
def test_status_reports_safe_to_restart():
    res, call = _invoke("status", reply={
        "maintenance_mode": True, "note": "升级镜像", "active_jobs": 0,
        "blockers": [], "paused_awaiting_resume": 2, "safe_to_restart": True,
    })
    assert res.exit_code == 0
    assert call.call_args[0] == ("GET", "/api/admin/maintenance")
    assert "可以重启" in res.stdout
    assert "升级镜像" in res.stdout
    assert "待自动续训：2" in res.stdout


def test_status_warns_when_jobs_still_running():
    """★ 最重要：还有作业占卡时必须明确说「不能重启」。"""
    res, _ = _invoke("status", reply={
        "maintenance_mode": True, "note": "", "active_jobs": 2,
        "blockers": [
            {"lab_run_id": "run-a", "username": "alice", "kind": "train"},
            {"lab_run_id": "run-b", "username": "bob", "kind": "export"},
        ],
        "paused_awaiting_resume": 0, "safe_to_restart": False,
    })
    assert res.exit_code == 0
    assert "现在重启会打断" in res.stdout
    assert "可以重启" not in res.stdout
    assert "run-a" in res.stdout and "run-b" in res.stdout


def test_status_when_maintenance_off():
    res, _ = _invoke("status", reply={
        "maintenance_mode": False, "note": "", "active_jobs": 0,
        "blockers": [], "paused_awaiting_resume": 0, "safe_to_restart": False,
    })
    assert res.exit_code == 0
    assert "维护模式：关闭" in res.stdout


# ---------------------------------------------------------------- drain
def test_drain_passes_note_and_lists_paused():
    res, call = _invoke("drain", "--note", "升级至 0.7.0-20260805", reply={
        "maintenance_mode": True, "note": "升级至 0.7.0-20260805",
        "paused": ["run-1", "run-2"], "blockers": [], "failed": [],
        "remaining_active": 0, "safe_to_restart": True,
    })
    assert res.exit_code == 0
    method, path = call.call_args[0]
    assert (method, path) == ("POST", "/api/admin/maintenance/drain")
    assert call.call_args[1]["body"] == {"note": "升级至 0.7.0-20260805"}
    assert "本次暂停 2 个" in res.stdout
    assert "run-1" in res.stdout and "run-2" in res.stdout


def test_drain_surfaces_stop_failures():
    """停不掉的作业必须显眼——它意味着 Ray 上可能还在跑，绝不能重启。"""
    res, _ = _invoke("drain", reply={
        "maintenance_mode": True, "note": "", "paused": [],
        "blockers": [],
        "failed": [{"lab_run_id": "run-x", "username": "carol",
                    "reason": "Ray stop 失败，重试 drain 或手动处理"}],
        "remaining_active": 1, "safe_to_restart": False,
    })
    assert res.exit_code == 0
    assert "停止失败" in res.stdout
    assert "run-x" in res.stdout
    assert "现在重启会打断" in res.stdout


def test_drain_surfaces_non_drainable_post_jobs():
    res, _ = _invoke("drain", reply={
        "maintenance_mode": True, "note": "", "paused": ["run-1"],
        "blockers": [{"lab_run_id": "run-e", "username": "dave",
                      "reason": "训练后作业不支持 checkpoint 续跑，未强停"}],
        "failed": [], "remaining_active": 1, "safe_to_restart": False,
    })
    assert res.exit_code == 0
    assert "run-e" in res.stdout
    assert "checkpoint 续跑" in res.stdout


# ---------------------------------------------------------------- resume
def test_resume_reports_count():
    res, call = _invoke("resume", reply={"maintenance_mode": False, "resuming": 3})
    assert res.exit_code == 0
    assert call.call_args[0] == ("POST", "/api/admin/maintenance/resume")
    assert "3 个作业" in res.stdout


@pytest.mark.parametrize("sub", ["status", "drain", "resume"])
def test_subcommands_registered(sub):
    res = runner.invoke(cli.app, ["admin", "maintenance", sub, "--help"])
    assert res.exit_code == 0
