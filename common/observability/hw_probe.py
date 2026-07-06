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
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    gpu_uuids: dict[int, str] = {}
    try:
        import psutil

        metrics["cpu.pct"] = float(psutil.cpu_percent(interval=None))
        metrics["cpu.thds"] = float(psutil.Process().num_threads())
        vm = psutil.virtual_memory()
        proc = psutil.Process()
        metrics["mem.pct"] = float(vm.percent)
        metrics["mem.proc"] = float(proc.memory_info().rss) / (1024**2)
        metrics["mem.proc.pct"] = float(proc.memory_percent())
        metrics["mem.proc.avail"] = float(vm.available) / (1024**2)
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
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"gpu.{out_idx}.pct"] = float(util.gpu)
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
) -> dict[str, Any]:
    pids: frozenset[int] | None
    if job_pids is None:
        pids = None
    elif isinstance(job_pids, frozenset):
        pids = job_pids
    else:
        pids = frozenset(int(x) for x in job_pids)
    mem_mib = DEFAULT_MIN_MEM_MIB if min_mem_mib is None else float(min_mem_mib)
    hw = collect_local_hw(job_pids=pids, min_mem_mib=mem_mib)
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
    try:
        import pynvml

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(mem.used) >= min_mem_mib * (1024**2)
    except Exception:
        return False


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
