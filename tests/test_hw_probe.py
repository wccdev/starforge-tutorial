"""hw_probe：GPU 进程归属与 UUID。"""
from __future__ import annotations

from types import SimpleNamespace

from common.observability import hw_probe


class _Mem:
    def __init__(self, used: int, total: int = 80 << 30):
        self.used = used
        self.total = total


def test_gpu_belongs_to_job_by_pid():
    handle = object()
    assert hw_probe._gpu_belongs_to_job(handle, frozenset({1234}), 2048.0) is False

    def _pids(_h):
        return {1234, 5678}

    orig = hw_probe._gpu_compute_pids
    hw_probe._gpu_compute_pids = _pids
    try:
        assert hw_probe._gpu_belongs_to_job(handle, frozenset({1234}), 2048.0) is True
        assert hw_probe._gpu_belongs_to_job(handle, frozenset({9999}), 2048.0) is False
        assert hw_probe._gpu_belongs_to_job(handle, frozenset(), 2048.0) is False
    finally:
        hw_probe._gpu_compute_pids = orig


def test_gpu_belongs_to_job_mem_fallback_when_no_pids(monkeypatch):
    handle = object()

    def _mem(_h):
        return _Mem(used=int(3000 * (1024**2)))

    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda _h: set())
    monkeypatch.setattr(
        "pynvml.nvmlDeviceGetMemoryInfo",
        _mem,
        raising=False,
    )
    import pynvml

    monkeypatch.setattr(pynvml, "nvmlDeviceGetMemoryInfo", _mem, raising=False)
    assert hw_probe._gpu_belongs_to_job(handle, None, 2048.0) is True
    assert hw_probe._gpu_belongs_to_job(handle, None, 4096.0) is False


def test_collect_local_hw_empty_when_job_pids_empty(monkeypatch):
    class _FakePynvml:
        NVML_TEMPERATURE_GPU = 0

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(_i):
            return object()

    monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakePynvml())
    out = hw_probe.collect_local_hw(job_pids=frozenset())
    assert not any(k.startswith("gpu.") for k in out["metrics"])
    assert out["gpu_uuids"] == {}


def test_collect_local_hw_reports_uuid_for_job_gpus(monkeypatch):
    seen: list[int] = []

    def _belongs(handle, job_pids, min_mem):
        seen.append(1)
        return job_pids is not None and 42 in job_pids

    monkeypatch.setattr(hw_probe, "_gpu_belongs_to_job", _belongs)

    class _FakePynvml:
        NVML_TEMPERATURE_GPU = 0

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 3

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):
            return i

        @staticmethod
        def nvmlDeviceGetUtilizationRates(_h):
            return SimpleNamespace(gpu=90, memory=20)

        @staticmethod
        def nvmlDeviceGetMemoryInfo(_h):
            return _Mem(used=int(50_000 * (1024**2)))

        @staticmethod
        def nvmlDeviceGetUUID(h):
            return f"GPU-physical-{h}"

    monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakePynvml())
    monkeypatch.setattr(
        hw_probe,
        "_gpu_belongs_to_job",
        lambda h, pids, _m: int(h) == 1 and pids == frozenset({42}),
    )
    out = hw_probe.collect_local_hw(job_pids=frozenset({42}))
    assert "gpu.0.mem.value" in out["metrics"]
    assert "gpu.1.mem.value" not in out["metrics"]
    assert out["gpu_uuids"] == {0: "GPU-physical-1"}
