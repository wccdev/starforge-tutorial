"""hw_probe：GPU 进程归属与 UUID。"""
from __future__ import annotations

from types import SimpleNamespace

from common.observability import hw_probe


class _Mem:
    def __init__(self, used: int, total: int = 80 << 30):
        self.used = used
        self.total = total


def test_gpu_belongs_to_job_by_pid(monkeypatch):
    handle = object()
    # 不带任何 compute-process 接口的 pynvml：_gpu_compute_pids 应给出空集。
    monkeypatch.setitem(__import__("sys").modules, "pynvml", SimpleNamespace())
    assert hw_probe._gpu_belongs_to_job(handle, frozenset({1234}), 2048.0) is False

    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda _h: {1234, 5678})
    assert hw_probe._gpu_belongs_to_job(handle, frozenset({1234}), 2048.0) is True
    assert hw_probe._gpu_belongs_to_job(handle, frozenset({9999}), 2048.0) is False
    assert hw_probe._gpu_belongs_to_job(handle, frozenset(), 2048.0) is False


def test_gpu_belongs_to_job_mem_fallback_when_no_pids(monkeypatch):
    handle = object()

    def _mem(_h, version=None):
        return _Mem(used=int(3000 * (1024**2)))

    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda _h: set())
    monkeypatch.setitem(
        __import__("sys").modules,
        "pynvml",
        SimpleNamespace(nvmlDeviceGetMemoryInfo=_mem),
    )
    assert hw_probe._gpu_belongs_to_job(handle, None, 2048.0) is True
    assert hw_probe._gpu_belongs_to_job(handle, None, 4096.0) is False


class _NotSupported(Exception):
    """替身：对应 pynvml.NVMLError_NotSupported。"""


def test_nvml_memory_prefers_v2(monkeypatch):
    """GB10 上 v1 不可用，v2 有时能返回统一内存池的数字。"""
    calls: list[object] = []

    class _FakePynvml:
        nvmlMemory_v2 = 0x02000028

        @staticmethod
        def nvmlDeviceGetMemoryInfo(_h, version=None):
            calls.append(version)
            if version is None:
                raise _NotSupported("Not Supported")
            return _Mem(used=1, total=119 << 30)

    monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakePynvml())
    assert hw_probe.nvml_memory(object()).total == 119 << 30
    assert calls == [0x02000028]


def test_nvml_memory_none_when_unsupported(monkeypatch):
    class _FakePynvml:
        nvmlMemory_v2 = 0x02000028

        @staticmethod
        def nvmlDeviceGetMemoryInfo(_h, version=None):
            raise _NotSupported("Not Supported")

    monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakePynvml())
    assert hw_probe.nvml_memory(object()) is None


def test_unified_memory_gpu_still_reports_util_and_power(monkeypatch):
    """统一内存设备查不到显存，但利用率/温度/功耗都正常，不能因此整机丢指标。

    回归：GB10 上 nvmlDeviceGetMemoryInfo 抛 NotSupported，异常一路冒到最外层的
    `except Exception: pass`，整台机器的 GPU 曲线全部消失。
    """

    class _FakePynvml:
        NVML_TEMPERATURE_GPU = 0
        nvmlMemory_v2 = 0x02000028

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 1

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):
            return i

        @staticmethod
        def nvmlDeviceGetUtilizationRates(_h):
            return SimpleNamespace(gpu=42, memory=7)

        @staticmethod
        def nvmlDeviceGetMemoryInfo(_h, version=None):
            raise _NotSupported("Not Supported")

        @staticmethod
        def nvmlDeviceGetTemperature(_h, _kind):
            return 30

        @staticmethod
        def nvmlDeviceGetPowerUsage(_h):
            return 4000

        @staticmethod
        def nvmlDeviceGetUUID(_h):
            return "GPU-151b37b4"

    monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakePynvml())
    monkeypatch.setattr(hw_probe, "_gpu_belongs_to_job", lambda *_a: True)

    metrics = hw_probe.collect_local_hw(job_pids=frozenset({42}))["metrics"]

    assert metrics["gpu.0.pct"] == 42.0
    assert metrics["gpu.0.temp"] == 30.0
    assert metrics["gpu.0.power"] == 4.0
    assert metrics["gpu.0.mem.time"] == 7.0
    # 显存查不到就别编：统一内存池的占用不等于这张卡的显存占用。
    assert "gpu.0.mem.pct" not in metrics
    assert "gpu.0.mem.value" not in metrics


def test_gpu_kept_when_memory_unqueryable_and_no_pids(monkeypatch):
    """无 PID 集合时靠显存阈值筛闲卡；查不到显存就无从判断，宁可多报也别整机消失。"""
    monkeypatch.setattr(hw_probe, "nvml_memory", lambda _h: None)
    assert hw_probe._gpu_belongs_to_job(object(), None, 2048.0) is True


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


class _TwoGpus:
    """两张卡：0 号闲着，1 号显存吃满。默认不提供 compute-process 接口。"""

    NVML_TEMPERATURE_GPU = 0
    used_by_index = {0: 10, 1: 50_000}

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
    def nvmlDeviceGetHandleByIndex(i):
        return i

    @staticmethod
    def nvmlDeviceGetUtilizationRates(_h):
        return SimpleNamespace(gpu=90, memory=20)

    @classmethod
    def nvmlDeviceGetMemoryInfo(cls, h, version=None):
        return _Mem(used=int(cls.used_by_index[int(h)] * (1024**2)))

    @staticmethod
    def nvmlDeviceGetUUID(h):
        return f"GPU-{h}"


def test_no_gpu_metrics_when_pid_attribution_empty_and_no_fallback(monkeypatch):
    """默认行为不变：认不出归属就不报，避免把别人的卡算到自己头上。"""
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _TwoGpus())
    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda _h: set())

    out = hw_probe.collect_local_hw(job_pids=frozenset())

    assert not [k for k in out["metrics"] if k.startswith("gpu.")]


def test_gpu_fallback_recovers_metrics_when_pids_unavailable(monkeypatch):
    """节点归属已由 placement group 证明时，PID 查空应退回显存启发式，而不是整机不报。

    GB10 节点的 dashboard agent 挂了，list_actors 查不到那边的 actor，job_pids 恒为空。
    """
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _TwoGpus())
    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda _h: set())

    out = hw_probe.collect_local_hw(job_pids=frozenset(), gpu_fallback=True)

    # 只认出忙着的那张（物理 1 号），且重新编号为 gpu.0
    assert out["metrics"]["gpu.0.pct"] == 90.0
    assert out["gpu_uuids"] == {0: "GPU-1"}
    assert "gpu.1.pct" not in out["metrics"]


def test_gpu_fallback_not_used_when_pid_attribution_works(monkeypatch):
    """PID 能认出卡时不该顺带把邻居的卡也收进来。"""
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _TwoGpus())
    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda h: {42} if int(h) == 0 else set())

    out = hw_probe.collect_local_hw(job_pids=frozenset({42}), gpu_fallback=True)

    assert out["gpu_uuids"] == {0: "GPU-0"}
