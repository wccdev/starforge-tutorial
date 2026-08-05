"""common.observability.report：框架无关上报 SDK 单测。

这个 SDK 的核心契约有两条，都会在生产里被真实触发，所以必须锁死：
  1. 没有 NEMOLAB_TOKEN（本地直跑）时**全程 no-op**，不建连、不起线程；
  2. 任何内部异常都不能冒泡——上报是旁路，采集挂了不能带崩训练。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.observability import report


@pytest.fixture(autouse=True)
def _reset_state():
    """每个用例前后都把模块级状态清干净（init 是幂等的，会记住上次结果）。"""
    report._state.update({"inited": False, "hw": None, "step": 0})
    yield
    report._state.update({"inited": False, "hw": None, "step": 0})


@pytest.fixture
def fake_ingest():
    ingest = MagicMock()
    ingest.run_id = "run-1"
    with patch("common.observability.session.get_ingest", return_value=ingest), patch(
        "common.observability.session.start_observability", return_value=object()
    ):
        yield ingest


# ---------------------------------------------------------------- no-op 契约
def test_disabled_without_token(monkeypatch):
    monkeypatch.delenv("NEMOLAB_TOKEN", raising=False)
    assert report.enabled() is False
    assert report.init() is False
    # 未启用时这些调用必须是纯 no-op（不抛、不做任何事）
    report.log({"loss": 1.0}, step=1)
    report.log_hparams({"lr": 1e-5})
    report.finish()


def test_log_before_init_is_noop(monkeypatch, fake_ingest):
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    report.log({"loss": 1.0}, step=1)
    fake_ingest.enqueue_metrics.assert_not_called()


# ---------------------------------------------------------------- 正常路径
def test_init_and_log_metrics(monkeypatch, fake_ingest):
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    assert report.init(hparams={"lr": 1e-5}, monitor_hardware=False) is True

    fake_ingest.enqueue_hparams.assert_called_once_with({"lr": 1e-5})

    report.log({"loss": 0.5, "acc": 0.9}, step=7)
    points = fake_ingest.enqueue_metrics.call_args[0][0]
    assert {p["key"] for p in points} == {"loss", "acc"}
    assert all(p["step"] == 7 for p in points)
    assert {p["value"] for p in points} == {0.5, 0.9}


def test_log_flattens_and_drops_unscalarizable(monkeypatch, fake_ingest):
    """训练脚本常常把整个 metrics dict 直接丢进来，里面混着字符串是常态——丢弃而不是报错。"""
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    report.init(monitor_hardware=False)

    report.log({"train": {"loss": 1.5}, "note": "hello", "n": 3}, step=1)
    points = fake_ingest.enqueue_metrics.call_args[0][0]
    keys = {p["key"] for p in points}
    assert keys == {"train.loss", "n"}  # 嵌套摊平，字符串丢弃


def test_log_auto_increments_step(monkeypatch, fake_ingest):
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    report.init(monitor_hardware=False)

    report.log({"loss": 1.0})
    report.log({"loss": 0.9})
    steps = [c[0][0][0]["step"] for c in fake_ingest.enqueue_metrics.call_args_list]
    assert steps == [1, 2]


def test_prefix_applied_once(monkeypatch, fake_ingest):
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    report.init(monitor_hardware=False)

    report.log({"accuracy": 0.7, "validation/loss": 0.2}, step=3, prefix="validation")
    keys = {p["key"] for p in fake_ingest.enqueue_metrics.call_args[0][0]}
    assert keys == {"validation/accuracy", "validation/loss"}  # 已带前缀的不再重复加


def test_init_is_idempotent(monkeypatch, fake_ingest):
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    with patch("common.observability.session.start_observability") as start:
        report.init(monitor_hardware=False)
        report.init(monitor_hardware=False)
        assert start.call_count == 1


def test_finish_stops_hardware_monitor(monkeypatch, fake_ingest):
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    hw = MagicMock()
    with patch("common.observability.hardware_monitor.HardwareMonitor", return_value=hw):
        report.init(monitor_hardware=True)
    report.finish()
    hw.stop.assert_called_once()
    fake_ingest.flush.assert_called_once()
    report.finish()  # 幂等：第二次不应再 stop
    assert hw.stop.call_count == 1


# ---------------------------------------------------------------- 异常吞噬
def test_log_never_raises(monkeypatch, fake_ingest):
    """采集挂了不能带崩训练——这是整个 SDK 最重要的一条。"""
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    report.init(monitor_hardware=False)
    fake_ingest.enqueue_metrics.side_effect = RuntimeError("transport exploded")
    report.log({"loss": 1.0}, step=1)  # 不抛即通过


def test_init_never_raises(monkeypatch):
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    with patch(
        "common.observability.session.start_observability",
        side_effect=RuntimeError("boom"),
    ):
        assert report.init() is False


# ---------------------------------------------------------------- HF 回调
def test_hf_callback_lifecycle(monkeypatch, fake_ingest):
    """鸭子类型适配 HF Trainer：只按方法名调用，不做 isinstance 检查。"""
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    cb = report.NeMoLabCallback(monitor_hardware=False)

    args = MagicMock()
    args.to_sanitized_dict.return_value = {"learning_rate": 1e-5}
    state = MagicMock()
    state.global_step = 42

    cb.on_train_begin(args=args, state=state, control=None)
    assert report._state["inited"] is True
    fake_ingest.enqueue_hparams.assert_called_once_with({"learning_rate": 1e-5})

    cb.on_log(args=args, state=state, control=None, logs={"loss": 0.3})
    points = fake_ingest.enqueue_metrics.call_args[0][0]
    assert points[0] == {**points[0], "key": "loss", "step": 42, "value": 0.3}

    cb.on_evaluate(args=args, state=state, control=None, metrics={"accuracy": 0.8})
    keys = {p["key"] for p in fake_ingest.enqueue_metrics.call_args[0][0]}
    assert keys == {"validation/accuracy"}

    cb.on_train_end(args=args, state=state, control=None)
    assert report._state["inited"] is False


def test_hf_callback_survives_bad_args(monkeypatch, fake_ingest):
    """HF 各版本 TrainingArguments 接口不一致，取不到超参也不能挂。"""
    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    cb = report.NeMoLabCallback(monitor_hardware=False)
    args = MagicMock()
    args.to_sanitized_dict.side_effect = RuntimeError("not supported")
    cb.on_train_begin(args=args, state=None, control=None)
    assert report._state["inited"] is True
