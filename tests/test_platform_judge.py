"""平台 judge 端点接入与环境指标上报（best-effort 契约）。"""
from __future__ import annotations

import importlib
import io
import json

import common.telemetry as telemetry


def _reload_reward(monkeypatch, **env):
    for key in ("STARFORGE_JUDGE_ENDPOINT", "STARFORGE_JUDGE_TOKEN",
                "JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import common.rewards.qa_judge_reward as mod

    return importlib.reload(mod)


def test_platform_judge_endpoint_takes_priority(monkeypatch):
    mod = _reload_reward(
        monkeypatch,
        STARFORGE_JUDGE_ENDPOINT="https://console.internal/api/judge",
        STARFORGE_JUDGE_TOKEN="tok-judge",
        JUDGE_BASE_URL="http://should-not-win:8001/v1",
    )
    assert mod.JUDGE_BASE_URL == "https://console.internal/api/judge/v1"
    assert mod.JUDGE_API_KEY == "tok-judge"


def test_local_judge_env_still_works_without_platform(monkeypatch):
    mod = _reload_reward(monkeypatch, JUDGE_BASE_URL="http://127.0.0.1:9001/v1", JUDGE_API_KEY="k")
    assert mod.JUDGE_BASE_URL == "http://127.0.0.1:9001/v1"
    assert mod.JUDGE_API_KEY == "k"
    # 复原为默认，避免影响其它测试的模块级状态
    _reload_reward(monkeypatch)


def test_report_metrics_posts_ingest_payload(monkeypatch):
    monkeypatch.setenv("STARFORGE_ENDPOINT", "https://console.internal/api/ingest")
    monkeypatch.setenv("STARFORGE_RUN_ID", "run-1")
    monkeypatch.setenv("STARFORGE_TOKEN", "tok")
    captured = {}

    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data)
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(telemetry.urllib.request, "urlopen", fake_urlopen)
    ok = telemetry.report_metrics({"env/search_rate": 0.5, "env/answers": 8}, step=3)
    assert ok is True
    assert captured["url"] == "https://console.internal/api/ingest/metrics"
    assert captured["auth"] == "Bearer tok"
    assert captured["body"]["run_id"] == "run-1"
    assert {p["key"]: p["value"] for p in captured["body"]["points"]} == {
        "env/search_rate": 0.5, "env/answers": 8.0,
    }
    assert all(p["step"] == 3 for p in captured["body"]["points"])


def test_report_metrics_never_raises(monkeypatch):
    monkeypatch.setenv("STARFORGE_ENDPOINT", "https://console.internal/api/ingest")
    monkeypatch.setenv("STARFORGE_RUN_ID", "run-1")
    monkeypatch.setenv("STARFORGE_TOKEN", "tok")

    def broken(*a, **k):
        raise ConnectionResetError("boom")

    monkeypatch.setattr(telemetry.urllib.request, "urlopen", broken)
    assert telemetry.report_metrics({"env/x": 1.0}) is False


def test_report_metrics_noop_without_credentials(monkeypatch):
    for key in ("STARFORGE_ENDPOINT", "STARFORGE_RUN_ID", "STARFORGE_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    assert telemetry.report_metrics({"env/x": 1.0}) is False
