"""本地硬件探测（pynvml + psutil，指标 key 对齐 SwanLab）。

GPU 采集原则（供看门狗单作业用卡归因）：
- 遍历**物理节点**上的全部 GPU（不依赖 driver 进程的 CUDA_VISIBLE_DEVICES）。
- scope=job 时传入本 Ray 作业的 actor PID 集合，仅上报这些进程占用的卡。
- 每张卡附带物理 UUID（gpu_uuid），服务端按 UUID 去重计数，避免多作业逻辑 idx 撞车。
- 无 job_pids 时不报 GPU（等 actor 就绪；看门狗有启动宽限期）。
"""
from __future__ import annotations

import os
import socket
from typing import Any

# 与 console watchdog_gpu_min_mem_mib 默认对齐；仅在没有 job_pids 的 local/cluster 模式作兜底。
DEFAULT_MIN_MEM_MIB = 2048.0


def collect_local_hw(
    *,
    job_pids: frozenset[int] | None = None,
    min_mem_mib: float = DEFAULT_MIN_MEM_MIB,
    include_process: bool = True,
) -> dict[str, Any]:
    """本机硬件指标。

    Args:
        include_process: 是否带上进程级指标（cpu.thds / mem.proc / mem.proc.pct）。
            这些指标读的是 `psutil.Process()`，也就是**当前进程**。在 driver 上跑就是
            训练主进程，有意义；被派到远端节点当 Ray task 跑，量到的却是探针任务自己
            （实测 113 MiB，而 vLLM worker 是几十个 GB），挂在「进程内存」标签下纯属误导。
            所以远端采集要关掉，只保留机器级指标。
    """
    metrics: dict[str, Any] = {}
    gpu_uuids: dict[int, str] = {}
    try:
        import psutil

        metrics["cpu.pct"] = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        metrics["mem.pct"] = float(vm.percent)
        metrics["mem.proc.avail"] = float(vm.available) / (1024**2)
        if include_process:
            proc = psutil.Process()
            metrics["cpu.thds"] = float(proc.num_threads())
            metrics["mem.proc"] = float(proc.memory_info().rss) / (1024**2)
            metrics["mem.proc.pct"] = float(proc.memory_percent())
    except Exception:
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            n = pynvml.nvmlDeviceGetCount()
            out_idx = 0
            for physical in range(n):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
                except Exception:
                    continue
                if not _gpu_belongs_to_job(handle, job_pids, min_mem_mib):
                    continue
                uuid = _gpu_uuid(handle)
                if uuid:
                    gpu_uuids[out_idx] = uuid
                # 逐项容错：统一内存设备（GB10）上显存查询不可用，但利用率/温度/功耗都正常，
                # 一处失败就把整机 GPU 指标全丢掉是不划算的。
                util = None
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f"gpu.{out_idx}.pct"] = float(util.gpu)
                except Exception:
                    pass
                mem = nvml_memory(handle)
                if mem is not None and getattr(mem, "total", 0):
                    metrics[f"gpu.{out_idx}.mem.pct"] = float(100.0 * mem.used / mem.total)
                    metrics[f"gpu.{out_idx}.mem.value"] = float(mem.used >> 20)
                try:
                    metrics[f"gpu.{out_idx}.temp"] = float(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except Exception:
                    pass
                try:
                    metrics[f"gpu.{out_idx}.power"] = float(
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    )
                except Exception:
                    pass
                if util is not None:
                    try:
                        metrics[f"gpu.{out_idx}.mem.time"] = float(util.memory)
                    except Exception:
                        pass
                out_idx += 1
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    return {"metrics": metrics, "gpu_uuids": gpu_uuids}


def collect_hw_snapshot(
    *,
    job_pids: frozenset[int] | list[int] | None = None,
    min_mem_mib: float | None = None,
    include_process: bool = True,
) -> dict[str, Any]:
    pids: frozenset[int] | None
    if job_pids is None:
        pids = None
    elif isinstance(job_pids, frozenset):
        pids = job_pids
    else:
        pids = frozenset(int(x) for x in job_pids)
    mem_mib = DEFAULT_MIN_MEM_MIB if min_mem_mib is None else float(min_mem_mib)
    hw = collect_local_hw(
        job_pids=pids, min_mem_mib=mem_mib, include_process=include_process
    )
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "metrics": hw.get("metrics") or {},
        "gpu_uuids": hw.get("gpu_uuids") or {},
    }


def _gpu_belongs_to_job(
    handle: Any,
    job_pids: frozenset[int] | None,
    min_mem_mib: float,
) -> bool:
    """判定物理 GPU 是否归属本作业。"""
    if job_pids is not None:
        if not job_pids:
            return False
        for pid in _gpu_compute_pids(handle):
            if pid in job_pids:
                return True
        return False
    # local / cluster 调试模式：无 PID 集合时按显存阈值过滤空闲卡。
    mem = nvml_memory(handle)
    if mem is None:
        # 统一内存设备查不到显存占用，此时无从判断闲忙——宁可多报一张卡，
        # 也好过让这台机器在面板上整个消失。
        return True
    return float(mem.used) >= min_mem_mib * (1024**2)


def _gpu_compute_pids(handle: Any) -> set[int]:
    import pynvml

    pids: set[int] = set()
    for getter in (
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
        getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
    ):
        if getter is None:
            continue
        try:
            for proc in getter(handle):
                pid = getattr(proc, "pid", None)
                if pid:
                    pids.add(int(pid))
            if pids:
                return pids
        except Exception:
            continue
    return pids


def nvml_memory(handle: Any) -> Any | None:
    """NVML 显存信息；查不到返回 None。

    GB10 / DGX Spark 这类统一内存设备没有独立显存，v1 的 nvmlDeviceGetMemoryInfo 会直接
    抛 NVMLError_NotSupported（nvidia-smi 上显示为 Memory-Usage: Not Supported）。
    先试 v2，它在部分驱动上能返回统一内存池的数字；都不行就交给调用方决定怎么兜底。
    """
    try:
        import pynvml
    except Exception:
        return None

    v2 = getattr(pynvml, "nvmlMemory_v2", None)
    if v2 is not None:
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(handle, version=v2)
        except Exception:
            pass
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        return None


def _gpu_uuid(handle: Any) -> str | None:
    try:
        import pynvml

        raw = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        text = str(raw).strip()
        return text or None
    except Exception:
        return None
