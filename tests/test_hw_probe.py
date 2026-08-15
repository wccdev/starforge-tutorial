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


def test_nvml_memory_raises_when_unsupported(monkeypatch):
    """平台只支持独立显存 GPU：显存查询失败就该抛错，不做任何兜底。"""

    class _FakePynvml:
        @staticmethod
        def nvmlDeviceGetMemoryInfo(_h):
            raise _NotSupported("Not Supported")

    monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakePynvml())
    import pytest

    with pytest.raises(_NotSupported):
        hw_probe.nvml_memory(object())


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
    # 序号是物理 NVML 序号：认出的是 1 号卡，就报 gpu.1，不重排成 gpu.0
    assert "gpu.1.mem.value" in out["metrics"]
    assert "gpu.0.mem.value" not in out["metrics"]
    assert out["gpu_uuids"] == {1: "GPU-physical-1"}
    assert out["gpu_attribution"] == "pid"


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

    训练节点的 dashboard agent 挂掉时，list_actors 查不到那边的 actor，job_pids 恒为空。
    """
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _TwoGpus())
    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda _h: set())

    out = hw_probe.collect_local_hw(job_pids=frozenset(), gpu_fallback=True)

    # 只认出忙着的那张（物理 1 号），序号保持物理序号
    assert out["metrics"]["gpu.1.pct"] == 90.0
    assert out["gpu_uuids"] == {1: "GPU-1"}
    assert "gpu.0.pct" not in out["metrics"]
    assert out["gpu_attribution"] == "mem"


def test_gpu_fallback_not_used_when_pid_attribution_works(monkeypatch):
    """PID 能认出卡时不该顺带把邻居的卡也收进来。"""
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _TwoGpus())
    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda h: {42} if int(h) == 0 else set())

    out = hw_probe.collect_local_hw(job_pids=frozenset({42}), gpu_fallback=True)

    assert out["gpu_uuids"] == {0: "GPU-0"}


def test_pid_attribution_matches_actor_child_process(monkeypatch):
    """占卡的是 actor 的子进程（vLLM EngineCore worker）时也要认出来。

    回归：只比对 actor PID 会漏认 → 整拍落进显存启发式 → 邻居作业的卡被画进面板。
    """
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _TwoGpus())
    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda h: {777} if int(h) == 1 else set())

    class _Proc:
        """777 是 555 的孙子进程，555 才是 Ray actor。"""

        _parents = {777: 666, 666: 555, 555: 1}

        def __init__(self, pid):
            self.pid = int(pid)

        def parent(self):
            nxt = self._parents.get(self.pid)
            return None if nxt is None else _Proc(nxt)

    monkeypatch.setitem(
        __import__("sys").modules, "psutil", SimpleNamespace(Process=_Proc)
    )

    out = hw_probe.collect_local_hw(job_pids=frozenset({555}), gpu_fallback=True)

    assert out["gpu_attribution"] == "pid"
    assert out["gpu_uuids"] == {1: "GPU-1"}


def test_sticky_selection_keeps_previously_confirmed_card(monkeypatch):
    """PID 这拍查空时沿用上次确认过的卡，而不是退到显存启发式。

    回归（作业 grpo_…_lora32_64）：colocated 训练在生成阶段只剩 vLLM 子进程占卡，
    PID 归属间歇失灵 → 退到显存启发式 → 同机另一个作业的卡被当成本作业的第二张卡，
    面板上凭空多出一条曲线，看门狗也把那张卡记到了本作业头上。
    """
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _TwoGpus())
    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda _h: set())
    # 两张卡都忙（邻居作业占着 0 号）：没有 sticky 就会两张一起报
    monkeypatch.setattr(_TwoGpus, "used_by_index", {0: 50_000, 1: 50_000})

    out = hw_probe.collect_local_hw(
        job_pids=frozenset({42}), gpu_fallback=True, known_gpu_uuids={"GPU-1"}
    )

    assert out["gpu_attribution"] == "sticky"
    assert out["gpu_uuids"] == {1: "GPU-1"}
    assert "gpu.0.pct" not in out["metrics"]


def test_mem_fallback_capped_by_bundle_gpu_count(monkeypatch):
    """从没认出过卡时只能按显存挑，但张数要被 PG 分到的卡数封顶（取占用最高的）。"""
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _TwoGpus())
    monkeypatch.setattr(hw_probe, "_gpu_compute_pids", lambda _h: set())
    monkeypatch.setattr(_TwoGpus, "used_by_index", {0: 30_000, 1: 70_000})

    out = hw_probe.collect_local_hw(
        job_pids=frozenset({42}), gpu_fallback=True, max_gpus=1
    )

    assert out["gpu_attribution"] == "mem"
    assert out["gpu_uuids"] == {1: "GPU-1"}
