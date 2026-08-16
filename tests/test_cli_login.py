"""客户端登录/门控/SSE 单测（纯客户端，无 server 依赖）。"""
from __future__ import annotations

import base64
import hashlib
import json

import pytest
import typer

from starforge_cli import api_client, auth, packing
from starforge_cli.client_device import collect_cli_device, encode_device_param


def test_parse_sse_stream_ignores_protocol_noise():
    """SSE 解析：只取 log 事件原文，忽略 event/id/keepalive，多行 data 以 \\n 拼回。"""
    raw = (
        ': keepalive\n\n'
        'event: open\n'
        'data: {"status":"connected"}\n\n'
        'event: log\n'
        'id: 26454\n'
        'data: (VllmGenerationWorker pid=1) INFO line A\n'
        'data:     缩进的 line B\n'
        'data:\n\n'  # 结尾空行 → 还原出末尾换行
        ': keepalive\n\n'
        'event: end\n'
        'data:\n\n'
    )
    events = list(api_client.parse_sse_stream(raw.splitlines(keepends=True)))
    logs = [d for e, d in events if e == "log"]
    assert logs == ["(VllmGenerationWorker pid=1) INFO line A\n    缩进的 line B\n"]
    assert events[-1][0] == "end"
    # 不会把 open 的 JSON 当成日志输出
    assert all('"status":"connected"' not in d for e, d in events if e == "log")


def test_parse_sse_stream_roundtrips_format_sse():
    """与服务端 format_sse 配对：多行日志块编码→解析应还原原文。"""
    def format_sse(data: str, *, event=None, event_id=None) -> str:  # 镜像 server/core/sse.py
        lines = []
        if event:
            lines.append(f"event: {event}")
        if event_id:
            lines.append(f"id: {event_id}")
        if data == "":
            lines.append("data:")
        else:
            lines += [f"data: {ln}" for ln in data.split("\n")]
        lines.append("")
        return "\n".join(lines) + "\n"

    chunk = "Step 1/300\n  reward=0.5\ntrailing\n"  # 含缩进与末尾换行
    frame = format_sse(chunk, event="log", event_id="42")
    events = list(api_client.parse_sse_stream(frame.splitlines(keepends=True)))
    assert events == [("log", chunk)]  # 精确还原（含 2 空格缩进与末尾换行）


def test_pkce_pair_self_consistent():
    """CLI 生成的 verifier/challenge 自洽：challenge == base64url(sha256(verifier))，无填充。"""
    verifier, challenge = auth.pkce_pair()
    expect = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    assert challenge == expect
    assert "=" not in challenge


@pytest.fixture()
def isolated_lab(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(auth, "CRED_PATH", tmp_path / "credentials.json")
    monkeypatch.delenv("FORGE_SERVER", raising=False)
    return tmp_path


def test_server_mode_detection(isolated_lab, monkeypatch):
    assert auth.current_server() == auth.DEFAULT_FORGE_SERVER
    assert auth.is_server_mode() is True
    auth._save_server("https://lab.x.com/")
    assert auth.current_server() == "https://lab.x.com"  # 去尾斜杠
    monkeypatch.setenv("FORGE_SERVER", "https://env.x.com")
    assert auth.current_server() == "https://env.x.com"  # 环境优先
    assert auth.current_server("https://explicit.com") == "https://explicit.com"


def test_gate_fails_loud_when_not_logged_in(isolated_lab, monkeypatch):
    """fail-loud：未登录的门控直接报错退出，绝不隐式发起登录流程。"""
    monkeypatch.setattr(auth, "get_access_token", lambda *a, **kw: None)
    login_calls: list = []
    monkeypatch.setattr(auth, "_interactive_login", lambda *a, **kw: login_calls.append(a))
    with pytest.raises(typer.Exit):
        auth.gate()
    assert login_calls == []  # 没有隐式登录


def test_gate_passes_when_logged_in(isolated_lab, monkeypatch):
    monkeypatch.setattr(auth, "get_access_token", lambda *a, **kw: "token")
    auth.gate()  # 不抛即通过


def test_corrupt_credentials_fail_loud(isolated_lab):
    """fail-loud：凭据文件损坏直接报错并提示处理办法，不静默当空。"""
    (isolated_lab / "credentials.json").write_text("{corrupt")
    with pytest.raises(typer.Exit):
        auth._load_creds("https://s")


def test_creds_roundtrip(isolated_lab):
    auth._save_creds("https://lab.x.com", {"access_token": "t", "expires_at": None})
    assert auth._load_creds("https://lab.x.com")["access_token"] == "t"
    auth._clear_creds("https://lab.x.com")
    assert auth._load_creds("https://lab.x.com") is None


def test_prefer_device_flow_ssh(monkeypatch):
    monkeypatch.delenv("FORGE_DEVICE_FLOW", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    assert auth.prefer_device_flow() is False
    monkeypatch.setenv("SSH_CONNECTION", "127.0.0.1 12345 54321")
    assert auth.prefer_device_flow() is True
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    assert auth.prefer_device_flow(force=True) is True


def test_device_login_polls_until_token(isolated_lab, monkeypatch):
    calls = {"n": 0}

    def fake_http(server, method, path, *, body=None, timeout=10.0):
        if path == "/api/cli/device/code":
            return 200, {
                "device_code": "dc",
                "user_code": "ABCD-1234",
                "verification_uri": "http://lab/cli/device",
                "verification_uri_complete": "http://lab/cli/device?user_code=ABCD-1234",
                "expires_in": 60,
                "interval": 0,
            }
        calls["n"] += 1
        if calls["n"] < 2:
            return 400, {"detail": "authorization_pending"}
        return 200, {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
            "user": {"username": "alice"},
        }

    monkeypatch.setattr(auth, "_http_json", fake_http)
    monkeypatch.setattr(auth.time, "sleep", lambda _: None)
    monkeypatch.setattr(auth.webbrowser, "open", lambda *_: None)
    creds = auth._device_login("https://lab.x.com")
    assert creds["access_token"] == "at"
    assert calls["n"] == 2


def test_get_access_token_valid_and_expired(isolated_lab):
    import time

    auth._save_creds("https://s", {"access_token": "valid", "expires_at": time.time() + 100})
    assert auth.get_access_token("https://s") == "valid"
    auth._save_creds("https://s", {"access_token": "old", "expires_at": time.time() - 10,
                                   "refresh_token": None})
    assert auth.get_access_token("https://s") is None


# ----------------------------- 打包 / 可追溯 -----------------------------
def _git_repo(tmp_path):
    import subprocess

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t.com")
    git("config", "user.name", "t")
    return git


def test_git_provenance(tmp_path):
    git = _git_repo(tmp_path)
    e = tmp_path / "experiments" / "demo_v1"
    e.mkdir(parents=True)
    (e / "config.yaml").write_text("a: 1\n")
    git("add", ".")
    git("commit", "-qm", "init")

    prov = packing.git_provenance(tmp_path, "experiments/demo_v1")
    assert prov["git_commit"]
    assert prov["git_dirty"] is False
    assert len(prov["config_sha"]) == 12

    (e / "config.yaml").write_text("a: 2\n")  # 改动 → dirty
    assert packing.git_provenance(tmp_path, "experiments/demo_v1")["git_dirty"] is True


def test_git_provenance_outside_repo_fails_loud(tmp_path):
    """fail-loud：不在 git 仓库内直接报错，不再返回 'unknown'。"""
    with pytest.raises(typer.Exit):
        packing.git_provenance(tmp_path, "experiments/x")


def test_collect_cli_device():
    info = collect_cli_device()
    assert info["source"] == "lab-cli"
    assert info["hostname"]
    assert info["os"]
    assert info.get("device_id")  # 16 hex chars
    assert len(info["device_id"]) == 16


def test_encode_device_roundtrip():
    raw = encode_device_param(collect_cli_device())
    assert raw
    assert "=" not in raw
    pad = "=" * (-len(raw) % 4)
    data = json.loads(base64.urlsafe_b64decode(raw + pad))
    assert data["source"] == "lab-cli"
