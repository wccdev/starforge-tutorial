"""common.observability：采集库单测（util / IngestClient / NeMoLabLogger / patch）。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.observability.ingest_client import IngestClient
from common.observability.util import flatten_dict, scalarize_metric


def test_flatten_dict_nested():
    out = flatten_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3})
    assert out == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_scalarize_metric():
    assert scalarize_metric(1) == 1.0
    assert scalarize_metric(True) == 1.0
    assert scalarize_metric([2, 4]) == 3.0
    assert scalarize_metric("x") is None
    assert scalarize_metric(None) is None


def _resp_ok():
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = lambda: None
    return m


def test_ingest_client_batches_and_posts():
    calls = []

    def _post(url, json=None, headers=None, timeout=None):
        calls.append((url, json))
        return _resp_ok()

    client = IngestClient("http://host/api/ingest", "run-1", "tok", flush_interval=999)
    client.enqueue_metrics([{"key": "train/reward", "step": 1, "value": 0.5}])
    with patch("requests.post", _post):
        client.flush()

    assert len(calls) == 1
    url, payload = calls[0]
    assert url == "http://host/api/ingest/metrics"
    assert payload["run_id"] == "run-1"
    assert payload["points"][0]["value"] == 0.5
    assert calls[0][0].startswith("http://host/api/ingest")


def test_ingest_client_requeues_on_failure():
    def _boom(url, json=None, headers=None, timeout=None):
        raise RuntimeError("network down")

    client = IngestClient("http://host/api/ingest", "run-1", "tok", flush_interval=999)
    client.enqueue_metrics([{"key": "k", "step": 1, "value": 1.0}])
    with patch("requests.post", _boom):
        client.flush()
    # 失败后点位回灌队列，下次仍可重试
    assert client._metric_q.qsize() == 1


def test_send_environment_nodes_reports_delivery():
    calls = []

    def _post(url, json=None, headers=None, timeout=None):
        calls.append((url, json))
        return _resp_ok()

    client = IngestClient("http://host/api/ingest", "run-1", "tok", flush_interval=999)
    nodes = [{"node_id": "n1", "hostname": "spark-gb10-1", "cpu": {"cores": 20}}]
    with patch("requests.post", _post):
        assert client.send_environment_nodes(nodes) is True

    assert calls[0][0] == "http://host/api/ingest/environment/nodes"
    assert calls[0][1] == {"run_id": "run-1", "nodes": nodes}


def test_send_environment_nodes_reports_failure():
    """返回 False 而不是抛异常：调用方靠它决定下一拍要不要重发。"""

    def _boom(url, json=None, headers=None, timeout=None):
        raise RuntimeError("network down")

    client = IngestClient("http://host/api/ingest", "run-1", "tok", flush_interval=999)
    with patch("requests.post", _boom):
        assert client.send_environment_nodes([{"node_id": "n1"}]) is False


def test_collect_node_hardware_is_self_contained():
    """远端节点上跑的探针只带硬件 + 主机名：overview/packages 是 driver 的上下文。"""
    from common.observability.env_probe import collect_node_hardware

    snap = collect_node_hardware()
    assert snap["hostname"]
    assert set(snap) <= {"hostname", "cpu", "gpu"}


def test_logger_enqueues_metrics(monkeypatch):
    monkeypatch.setenv("NEMOLAB_ENDPOINT", "http://host/api/ingest")
    monkeypatch.setenv("NEMOLAB_RUN_ID", "run-1")
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    monkeypatch.setenv("NEMOLAB_MONITOR_HARDWARE", "0")

    posted = []

    def _post(url, json=None, headers=None, timeout=None):
        posted.append((url, json))
        return _resp_ok()

    with patch("requests.post", _post):
        from common.observability.logger import NeMoLabLogger
        from common.observability.session import start_observability, stop_observability

        start_observability()
        try:
            nl = NeMoLabLogger({})
            nl.log_metrics({"reward": 0.6, "loss": 0.2}, step=1, prefix="train")
            nl.log_metrics({"accuracy": 0.35}, step=2, prefix="validation")
            nl._ingest.flush()
            nl.finish()
        finally:
            stop_observability()

    keys = {p["key"] for _, body in posted for p in body.get("points", [])}
    assert "train/reward" in keys
    assert "validation/accuracy" in keys


def test_terminal_proxy_posts_logs(monkeypatch):
    monkeypatch.setenv("NEMOLAB_ENDPOINT", "http://host/api/ingest")
    monkeypatch.setenv("NEMOLAB_RUN_ID", "run-1")
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")

    posted = []

    def _post(url, json=None, headers=None, timeout=None):
        posted.append((url, json))
        return _resp_ok()

    with patch("requests.post", _post):
        from common.observability.session import start_observability, stop_observability

        start_observability()
        import sys

        print("train-start", file=sys.stdout)
        stop_observability()

    log_posts = [b for url, b in posted if url.endswith("/logs")]
    assert log_posts
    assert any("train-start" in c for p in log_posts for c in p.get("chunks", []))


def test_logger_requires_credentials(monkeypatch):
    for var in ("NEMOLAB_ENDPOINT", "NEMOLAB_RUN_ID", "NEMOLAB_TOKEN", "NRL_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    from common.observability.logger import NeMoLabLogger

    with pytest.raises(ValueError):
        NeMoLabLogger({})


def test_patch_is_noop_without_token(monkeypatch):
    monkeypatch.delenv("NEMOLAB_TOKEN", raising=False)
    import common.observability.patch as patch_mod

    patch_mod._PATCHED = False
    # 无 token、无 nemo_rl 也不应抛错（直接返回）
    patch_mod.apply_patch()
    assert patch_mod._PATCHED is False
