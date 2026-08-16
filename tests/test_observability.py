"""common.observability：采集库单测（util / IngestClient / StarForgeLogger / patch）。"""
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
    nodes = [{"node_id": "n1", "hostname": "train-node-1", "cpu": {"cores": 20}}]
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
    monkeypatch.setenv("STARFORGE_ENDPOINT", "http://host/api/ingest")
    monkeypatch.setenv("STARFORGE_RUN_ID", "run-1")
    monkeypatch.setenv("STARFORGE_TOKEN", "tok")
    monkeypatch.setenv("STARFORGE_MONITOR_HARDWARE", "0")

    posted = []

    def _post(url, json=None, headers=None, timeout=None):
        posted.append((url, json))
        return _resp_ok()

    with patch("requests.post", _post):
        from common.observability.logger import StarForgeLogger
        from common.observability.session import start_observability, stop_observability

        start_observability()
        try:
            nl = StarForgeLogger({})
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
    monkeypatch.setenv("STARFORGE_ENDPOINT", "http://host/api/ingest")
    monkeypatch.setenv("STARFORGE_RUN_ID", "run-1")
    monkeypatch.setenv("STARFORGE_TOKEN", "tok")

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
    for var in ("STARFORGE_ENDPOINT", "STARFORGE_RUN_ID", "STARFORGE_TOKEN", "NRL_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    from common.observability.logger import StarForgeLogger

    with pytest.raises(ValueError):
        StarForgeLogger({})


def test_patch_is_noop_without_token(monkeypatch):
    monkeypatch.delenv("STARFORGE_TOKEN", raising=False)
    import common.observability.patch as patch_mod

    patch_mod._PATCHED = False
    # 无 token、无 nemo_rl 也不应抛错（直接返回）
    patch_mod.apply_patch()
    assert patch_mod._PATCHED is False


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class _FakeIngest:
    def __init__(self):
        self.run_id = "r1"
        self.payloads = []

    def enqueue_validation(self, payload):
        self.payloads.append(payload)
        return True


def _dpo_datum(user_ids, chosen_ids, rejected_ids):
    return {
        "message_log_chosen": [
            {"role": "user", "token_ids": user_ids},
            {"role": "assistant", "token_ids": chosen_ids},
        ],
        "message_log_rejected": [
            {"role": "user", "token_ids": user_ids},
            {"role": "assistant", "token_ids": rejected_ids},
        ],
    }


def test_dpo_samples_extracted_from_dataset(monkeypatch):
    from common.observability.patch import _upload_dpo_samples

    data = [
        _dpo_datum([ord("q"), ord("?")], [ord("4")], [ord("5")]),
        _dpo_datum([ord("p")], [ord("y")], [ord("n")]),
    ]

    class _Loader:
        dataset = data

    ingest = _FakeIngest()
    _upload_dpo_samples(ingest, 12, {"val": _Loader()}, _FakeTokenizer())

    assert len(ingest.payloads) == 1
    payload = ingest.payloads[0]
    assert payload["step"] == 12
    assert payload["total_samples"] == 2
    s = payload["samples"][0]
    assert s["user"] == "q?"
    assert s["assistant"] == "4"
    assert s["reward"] is None
    assert s["extra"] == {"chosen": "4", "rejected": "5"}


def test_dpo_samples_accept_tensor_like_token_ids():
    from common.observability.patch import _upload_dpo_samples

    class TensorLike(list):
        def __bool__(self):
            raise RuntimeError("Boolean value of Tensor is ambiguous")

    data = [_dpo_datum(TensorLike([113]), TensorLike([52]), TensorLike([53]))]

    class _Loader:
        dataset = data

    ingest = _FakeIngest()
    _upload_dpo_samples(ingest, 2, {"val": _Loader()}, _FakeTokenizer())
    assert ingest.payloads[0]["samples"][0]["extra"] == {"chosen": "4", "rejected": "5"}


def test_dpo_samples_respect_upload_limit(monkeypatch):
    from common.observability.patch import _upload_dpo_samples

    monkeypatch.setenv("STARFORGE_VAL_UPLOAD_SAMPLES", "1")
    data = [_dpo_datum([97], [98], [99])] * 5

    class _Loader:
        dataset = data

    ingest = _FakeIngest()
    _upload_dpo_samples(ingest, 1, {"val": _Loader()}, _FakeTokenizer())
    assert ingest.payloads[0]["total_samples"] == 1


def test_zero_sample_limit_disables_dpo_upload(monkeypatch):
    from common.observability.patch import _upload_dpo_samples

    monkeypatch.setenv("STARFORGE_VAL_UPLOAD_SAMPLES", "0")
    data = [_dpo_datum([113], [52], [53])]

    class _Loader:
        dataset = data

    ingest = _FakeIngest()
    _upload_dpo_samples(ingest, 1, {"val": _Loader()}, _FakeTokenizer())
    assert ingest.payloads == []


def test_zero_sample_limit_disables_conversation_upload(monkeypatch):
    from common.observability.patch import _upload_validation_samples

    monkeypatch.setenv("STARFORGE_VAL_UPLOAD_SAMPLES", "0")
    ingest = _FakeIngest()
    logs = [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]]
    _upload_validation_samples(ingest, 1, logs, [1.0])
    assert ingest.payloads == []


def test_dpo_samples_no_dataloader_is_noop():
    from common.observability.patch import _upload_dpo_samples

    ingest = _FakeIngest()
    _upload_dpo_samples(ingest, 1, {}, _FakeTokenizer())
    assert ingest.payloads == []


def _sft_datum():
    return {
        "message_log": [
            {"role": "system", "content": "brief", "token_ids": [1]},
            {"role": "user", "content": "hello", "token_ids": [2, 3]},
            {"role": "assistant", "content": "world", "token_ids": [4]},
        ]
    }


def test_sft_skeletons_do_not_advance_loader():
    from common.observability.patch import _sft_sample_skeletons

    class Loader:
        dataset = [_sft_datum()]

        def __iter__(self):
            raise AssertionError("observer must not consume stateful loader")

    samples = _sft_sample_skeletons(Loader(), _FakeTokenizer(), 8, 12000)
    assert samples[0]["user"] == "system: brief\nuser: hello"
    assert samples[0]["extra"] == {"reference": "world"}
    assert samples[0]["_prompt_token_ids"] == [1, 2, 3]


def test_sft_generation_failure_restores_training_state(monkeypatch):
    import sys
    from types import SimpleNamespace

    from common.observability.patch import _materialize_sft_completions

    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.distributed.batched_data_dict",
        SimpleNamespace(BatchedDataDict=lambda value: value),
    )
    class _Tensor:
        def __init__(self, values):
            self.values = values

        def max(self):
            return _Tensor(max(self.values))

        def item(self):
            return self.values

        def __getitem__(self, index):
            return _Tensor(self.values[index])

    class _Matrix:
        def __setitem__(self, key, value):
            return None

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            long=object(),
            tensor=lambda values, dtype=None: _Tensor(values),
            full=lambda shape, fill, dtype=None: _Matrix(),
        ),
    )

    class Policy:
        cfg = {"max_total_sequence_length": 256}

        def __init__(self):
            self.events = []

        def prepare_for_generation(self):
            self.events.append("prepare_generation")

        def generate(self, data, greedy):
            assert greedy is True
            self.events.append("generate")
            raise RuntimeError("generation failed")

        def finish_generation(self):
            self.events.append("finish_generation")

        def prepare_for_training(self):
            self.events.append("prepare_training")

    sample = {
        "user": "q",
        "assistant": "",
        "extra": {"reference": "a"},
        "_prompt_token_ids": [1, 2],
    }
    policy = Policy()
    with pytest.raises(RuntimeError, match="generation failed"):
        _materialize_sft_completions(policy, _FakeTokenizer(), [sample])
    assert policy.events == [
        "prepare_generation",
        "generate",
        "finish_generation",
        "prepare_training",
    ]


def test_sft_upload_falls_back_to_reference_only(monkeypatch):
    import common.observability.patch as patch_mod

    class Loader:
        dataset = [_sft_datum()]

    def fail_generation(*args, **kwargs):
        raise RuntimeError("oom")

    monkeypatch.setattr(patch_mod, "_materialize_sft_completions", fail_generation)
    ingest = _FakeIngest()
    patch_mod._upload_sft_samples(
        ingest, 7, object(), Loader(), _FakeTokenizer()
    )
    sample = ingest.payloads[0]["samples"][0]
    assert sample["assistant"] == ""
    assert sample["extra"]["reference"] == "world"
    assert "_prompt_token_ids" not in sample
