
def test_extract_message_log_samples():
    from common.observability.validation_extract import extract_message_log_samples

    logs = [
        [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}],
        [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}],
    ]
    rewards = [1.0, 0.0]
    samples, dist, avg = extract_message_log_samples(logs, rewards, num_samples=2)
    assert len(samples) == 2
    assert samples[0]["user"] == "Q1"
    assert samples[0]["assistant"] == "A1"
    assert avg == 0.5
    assert dist == [{"reward": 0.0, "count": 1}, {"reward": 1.0, "count": 1}]


def test_extract_all_uses_original_index():
    """num_samples=None：全量上报，idx=原始位置(1-based)，dist/avg 覆盖全量。"""
    from common.observability.validation_extract import extract_message_log_samples

    logs = [[{"role": "user", "content": f"Q{i}"}] for i in range(5)]
    rewards = [0.0, 1.0, 0.0, 1.0, 0.0]
    samples, dist, avg = extract_message_log_samples(logs, rewards, num_samples=None)
    assert [s["idx"] for s in samples] == [1, 2, 3, 4, 5]
    assert [s["user"] for s in samples] == [f"Q{i}" for i in range(5)]
    assert avg == 0.4
    assert dist == [{"reward": 0.0, "count": 3}, {"reward": 1.0, "count": 2}]


def test_extract_cap_preserves_original_idx():
    """截断上报时按 reward 两端取样，但 idx 仍是原始位置、原始顺序，便于跨轮对齐。"""
    from common.observability.validation_extract import extract_message_log_samples

    logs = [[{"role": "user", "content": f"Q{i}"}] for i in range(4)]
    rewards = [0.1, 0.9, 0.2, 0.8]  # 最高=idx1(0.9)，最低=idx0(0.1)
    samples, _dist, _avg = extract_message_log_samples(logs, rewards, num_samples=2)
    assert [s["idx"] for s in samples] == [1, 2]  # 原始位置 0、1 → 1-based
    assert {round(s["reward"], 1) for s in samples} == {0.1, 0.9}


def test_extract_clips_long_fields():
    from common.observability.validation_extract import extract_message_log_samples

    long = "x" * 5000
    logs = [[{"role": "user", "content": long}, {"role": "assistant", "content": long}]]
    samples, _d, _a = extract_message_log_samples(logs, [1.0], max_field_chars=200)
    assert len(samples[0]["user"]) <= 200
    assert "[truncated" in samples[0]["user"]
    assert len(samples[0]["assistant"]) <= 200


def test_validation_ctx_detects_step():
    from common.observability.validation_ctx import (
        active_validation_step,
        clear_validation_step,
        feed_log_text,
    )

    clear_validation_step()
    feed_log_text("noise\nStarting validation at step 42\n")
    assert active_validation_step() == 42
    clear_validation_step()
    assert active_validation_step() is None


def test_upload_runs_even_when_print_fails(monkeypatch):
    """回归：rich 打印抛错时，结构化样本仍应上报（旧实现先 print 再 upload 会整轮丢样）。"""
    from unittest.mock import MagicMock

    import common.observability.patch as patch_mod
    from common.observability.validation_ctx import clear_validation_step, feed_log_text

    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    monkeypatch.setenv("NEMOLAB_VAL_CHUNK", "4")

    uploaded: list[dict] = []

    class FakeIngest:
        run_id = "run-1"

        def enqueue_validation(self, payload):
            uploaded.append(payload)
            return True

    monkeypatch.setattr(patch_mod, "_PATCHED", False)

    # 伪造 nemo_rl.utils.logger 模块，避免依赖完整安装
    import sys
    import types

    logger_mod = types.ModuleType("nemo_rl.utils.logger")

    class Logger:
        def __init__(self, cfg):
            self.base_log_dir = "/tmp"
            self.loggers = []

    def boom_print(*_a, **_k):
        raise RuntimeError("rich explode")

    logger_mod.Logger = Logger
    logger_mod.print_message_log_samples = boom_print
    sys.modules["nemo_rl"] = types.ModuleType("nemo_rl")
    sys.modules["nemo_rl.utils"] = types.ModuleType("nemo_rl.utils")
    sys.modules["nemo_rl.utils.logger"] = logger_mod

    monkeypatch.setattr(
        "common.observability.logger.NeMoLabLogger",
        MagicMock,
    )
    monkeypatch.setattr(
        "common.observability.session.get_ingest",
        lambda: FakeIngest(),
    )

    clear_validation_step()
    feed_log_text("▶ Starting validation at step 30...\n")
    patch_mod.apply_patch()

    logs = [[{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": "A"}] for i in range(3)]
    rewards = [0.0, 1.0, 0.5]
    # 打印会炸，但不应阻止上报
    logger_mod.print_message_log_samples(logs, rewards, num_samples=2, step=30)

    assert uploaded, "print 失败时仍应上报验证样本"
    assert uploaded[0]["step"] == 30
    assert uploaded[0]["total_samples"] == 3
    assert len(uploaded[0]["samples"]) == 3


def test_upload_falls_back_to_step_when_ctx_missing(monkeypatch):
    """日志解析没跟上（active_step=None）时，仍用调用方 step 上报。"""
    from unittest.mock import MagicMock

    import common.observability.patch as patch_mod
    from common.observability.validation_ctx import clear_validation_step

    monkeypatch.setenv("NEMOLAB_TOKEN", "tok")
    uploaded: list[dict] = []

    class FakeIngest:
        run_id = "run-1"

        def enqueue_validation(self, payload):
            uploaded.append(payload)
            return True

    monkeypatch.setattr(patch_mod, "_PATCHED", False)

    import sys
    import types

    logger_mod = types.ModuleType("nemo_rl.utils.logger")

    class Logger:
        def __init__(self, cfg):
            self.base_log_dir = "/tmp"
            self.loggers = []

    def ok_print(*_a, **_k):
        return None

    logger_mod.Logger = Logger
    logger_mod.print_message_log_samples = ok_print
    sys.modules["nemo_rl"] = types.ModuleType("nemo_rl")
    sys.modules["nemo_rl.utils"] = types.ModuleType("nemo_rl.utils")
    sys.modules["nemo_rl.utils.logger"] = logger_mod

    monkeypatch.setattr("common.observability.logger.NeMoLabLogger", MagicMock)
    monkeypatch.setattr("common.observability.session.get_ingest", lambda: FakeIngest())

    clear_validation_step()  # 模拟日志解析丢失
    patch_mod.apply_patch()

    logs = [[{"role": "user", "content": "Q"}]]
    logger_mod.print_message_log_samples(logs, [1.0], num_samples=1, step=90)

    assert uploaded and uploaded[0]["step"] == 90


def test_enqueue_validation_returns_false_on_error(monkeypatch):
    from common.observability.ingest_client import IngestClient

    client = IngestClient("http://host/api/ingest", "run-1", "tok")

    def boom(*_a, **_k):
        raise RuntimeError("timeout")

    monkeypatch.setattr(client, "_post", boom)
    ok = client.enqueue_validation(
        {"run_id": "run-1", "step": 0, "chunk_index": 0, "samples": [{"idx": 1}]}
    )
    assert ok is False
