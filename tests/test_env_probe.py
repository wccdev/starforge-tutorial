"""env_probe：静态硬件快照的容错。

回归背景：GB10 / DGX Spark 用统一内存，没有独立显存，NVML 的 nvmlDeviceGetMemoryInfo
直接抛 NVMLError_NotSupported。异常从 `_collect_nvidia_gpu` 冒到 Ray remote task 外面，
整份节点快照采不回来，训练日志里每一拍刷一条 traceback。
"""
from __future__ import annotations

import sys

import pytest

from common.observability import env_probe


class _NotSupported(Exception):
    """替身：对应 pynvml.NVMLError_NotSupported。"""


class _Mem:
    def __init__(self, total: int):
        self.total = total
        self.used = 0


class _BasePynvml:
    """够用的 pynvml 替身；子类改写想测的那一项。"""

    nvmlMemory_v2 = 0x02000028
    count = 1

    @classmethod
    def nvmlInit(cls):
        return None

    @classmethod
    def nvmlShutdown(cls):
        return None

    @classmethod
    def nvmlSystemGetDriverVersion(cls):
        return b"595.71.05"

    @classmethod
    def nvmlDeviceGetCount(cls):
        return cls.count

    @classmethod
    def nvmlDeviceGetHandleByIndex(cls, i):
        return i

    @classmethod
    def nvmlDeviceGetName(cls, h):
        return b"NVIDIA GB10"

    @classmethod
    def nvmlDeviceGetMemoryInfo(cls, h, version=None):
        return _Mem(total=140 << 30)


@pytest.fixture(autouse=True)
def _no_cuda_probe(monkeypatch):
    """别在单测里真去 shell 出 nvcc / nvidia-smi。"""
    monkeypatch.setattr(env_probe, "_cuda_version", lambda: "13.2")


def _install(monkeypatch, cls):
    monkeypatch.setitem(sys.modules, "pynvml", cls)


def test_unified_memory_falls_back_to_system_ram(monkeypatch):
    """GB10 查不到显存时用系统内存顶上——那本来就是这块 GPU 能用的全部容量。"""

    class _GB10(_BasePynvml):
        @classmethod
        def nvmlDeviceGetMemoryInfo(cls, h, version=None):
            raise _NotSupported("Not Supported")

    _install(monkeypatch, _GB10)
    monkeypatch.setattr(env_probe, "_system_memory_gb", lambda: 119)

    gpu = env_probe._collect_nvidia_gpu()

    assert gpu["count"] == 1
    assert gpu["driver_version"] == "595.71.05"
    assert gpu["devices"] == [{"index": 0, "name": "NVIDIA GB10", "memory_gb": 119}]


def test_discrete_gpu_still_uses_nvml_memory(monkeypatch):
    _install(monkeypatch, _BasePynvml)
    monkeypatch.setattr(env_probe, "_system_memory_gb", lambda: 2015)

    gpu = env_probe._collect_nvidia_gpu()

    assert gpu["devices"][0]["memory_gb"] == 140


def test_one_bad_device_does_not_sink_the_snapshot(monkeypatch):
    class _Flaky(_BasePynvml):
        count = 3

        @classmethod
        def nvmlDeviceGetHandleByIndex(cls, i):
            if i == 1:
                raise _NotSupported("Not Supported")
            return i

    _install(monkeypatch, _Flaky)

    gpu = env_probe._collect_nvidia_gpu()

    assert [d["index"] for d in gpu["devices"]] == [0, 2]


def test_unreadable_name_does_not_raise(monkeypatch):
    class _NoName(_BasePynvml):
        @classmethod
        def nvmlDeviceGetName(cls, h):
            raise _NotSupported("Not Supported")

    _install(monkeypatch, _NoName)

    assert env_probe._collect_nvidia_gpu()["devices"][0]["name"] == "GPU 0"


def test_collect_node_hardware_survives_unsupported_nvml(monkeypatch):
    """远端探针的最终契约：GB10 上照样返回快照，绝不抛异常。"""

    class _GB10(_BasePynvml):
        @classmethod
        def nvmlDeviceGetMemoryInfo(cls, h, version=None):
            raise _NotSupported("Not Supported")

    _install(monkeypatch, _GB10)

    snap = env_probe.collect_node_hardware()

    assert snap["hostname"]
    assert snap["gpu"]["devices"][0]["name"] == "NVIDIA GB10"


def test_no_nvidia_gpu_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert env_probe._collect_nvidia_gpu() is None


def test_job_scope_caps_devices_to_bundle_gpu_count(monkeypatch):
    """作业只用 2 卡时，环境快照不能把整机 8 卡都列出来。"""

    class _Eight(_BasePynvml):
        count = 8

        @classmethod
        def nvmlDeviceGetName(cls, h):
            return b"NVIDIA H200"

        @classmethod
        def nvmlDeviceGetMemoryInfo(cls, h, version=None):
            # 每张都「忙」：没有 max_gpus 封顶会全报
            return _Mem(total=140 << 30)

        @classmethod
        def nvmlDeviceGetUUID(cls, h):
            return f"GPU-{h}".encode()

    _install(monkeypatch, _Eight)
    monkeypatch.setattr(
        "common.observability.hw_probe._gpu_compute_pids", lambda _h: set()
    )

    gpu = env_probe._collect_nvidia_gpu(
        job_pids=[],
        gpu_fallback=True,
        max_gpus=2,
        min_mem_mib=0.0,
    )

    assert gpu["count"] == 2
    assert [d["index"] for d in gpu["devices"]] == [0, 1]
    assert all(d["name"] == "NVIDIA H200" for d in gpu["devices"])


def test_unfiltered_collect_still_lists_all_gpus(monkeypatch):
    """driver 侧 collect_environment() 不传过滤参数，仍报整机清单作兜底。"""

    class _Eight(_BasePynvml):
        count = 8

        @classmethod
        def nvmlDeviceGetName(cls, h):
            return b"NVIDIA H200"

    _install(monkeypatch, _Eight)

    gpu = env_probe._collect_nvidia_gpu()

    assert gpu["count"] == 8
    assert len(gpu["devices"]) == 8


def test_capacity_fallback_when_attribution_empty(monkeypatch):
    """PID/显存都认不出时，仍按 PG 卡数截断，避免把 8 卡写进作业环境。"""

    class _Eight(_BasePynvml):
        count = 8

        @classmethod
        def nvmlDeviceGetName(cls, h):
            return b"NVIDIA H200"

        @classmethod
        def nvmlDeviceGetMemoryInfo(cls, h, version=None):
            return _Mem(total=140 << 30)  # used=0 → 显存启发式认不出

        @classmethod
        def nvmlDeviceGetUUID(cls, h):
            return f"GPU-{h}".encode()

    _install(monkeypatch, _Eight)
    monkeypatch.setattr(
        "common.observability.hw_probe._gpu_compute_pids", lambda _h: set()
    )

    gpu = env_probe._collect_nvidia_gpu(
        job_pids=[42],
        gpu_fallback=True,
        max_gpus=2,
        min_mem_mib=2048.0,
    )

    assert gpu["count"] == 2
    assert [d["index"] for d in gpu["devices"]] == [0, 1]
