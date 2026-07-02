
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
