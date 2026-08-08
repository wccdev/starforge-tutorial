"""本地硬件探测（pynvml + psutil，指标 key 对齐 SwanLab）。

GPU 采集原则（供看门狗单作业用卡归因）：
- 遍历**物理节点**上的全部 GPU（不依赖 driver 进程的 CUDA_VISIBLE_DEVICES）。
- scope=job 时传入本 Ray 作业的 actor PID 集合，仅上报这些进程占用的卡。
- 每张卡附带物理 UUID（gpu_uuid），服务端按 UUID 去重计数，避免多作业逻辑 idx 撞车。
- 指标 key 里的序号是**物理 NVML 序号**，不是「第几张被选中的卡」（见 _select_gpus）。
- 无 job_pids 时不报 GPU（等 actor 就绪；看门狗有启动宽限期）。
"""
from __future__ import annotations

import os
import socket
from typing import Any

# 与 console watchdog_gpu_min_mem_mib 默认对齐；仅在没有 job_pids 的 local/cluster 模式作兜底。
DEFAULT_MIN_MEM_MIB = 2048.0
# NVML 只报直接占卡的那个进程；vLLM 的 EngineCore worker 是 actor 的子进程，往上找几层。
PID_ANCESTOR_MAX_DEPTH = 6


def collect_local_hw(
    *,
    job_pids: frozenset[int] | None = None,
    min_mem_mib: float = DEFAULT_MIN_MEM_MIB,
    include_process: bool = True,
    gpu_fallback: bool = False,
    known_gpu_uuids: frozenset[str] | set[str] | list[str] | None = None,
    max_gpus: int | None = None,
) -> dict[str, Any]:
    """本机硬件指标。

    Args:
        gpu_fallback: PID 归属认不出任何 GPU 时，是否退回显存启发式认卡。仅在调用方
            已确知本节点属于该作业时开启（见 _select_gpus 的说明）。
        known_gpu_uuids: 本节点先前**已被 PID 认定**属于本作业的卡。PID 这拍查空时优先
            沿用它，别退到显存启发式（那会把邻居作业的卡收进来）。
        max_gpus: 本作业在这台机器上分到的卡数（来自 GPU placement group bundle）。
            只在显存启发式那一档生效，给挑出来的张数封顶。
        include_process: 是否带上进程级指标（cpu.thds / mem.proc / mem.proc.pct）。
            这些指标读的是 `psutil.Process()`，也就是**当前进程**。在 driver 上跑就是
            训练主进程，有意义；被派到远端节点当 Ray task 跑，量到的却是探针任务自己
            （实测 113 MiB，而 vLLM worker 是几十个 GB），挂在「进程内存」标签下纯属误导。
            所以远端采集要关掉，只保留机器级指标。

    Returns:
        metrics / gpu_uuids（按物理序号）/ gpu_attribution（这批卡是怎么认出来的：
        pid | sticky | mem | none，调用方据此决定要不要把这批 UUID 记成「已确认」）。
    """
    metrics: dict[str, Any] = {}
    gpu_uuids: dict[int, str] = {}
    attribution = "none"
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
            devices: list[tuple[int, Any]] = []
            for physical in range(n):
                try:
                    devices.append((physical, pynvml.nvmlDeviceGetHandleByIndex(physical)))
                except Exception:
                    continue
            selected, attribution = _select_gpus(
                devices,
                job_pids=job_pids,
                min_mem_mib=min_mem_mib,
                gpu_fallback=gpu_fallback,
                known_gpu_uuids=frozenset(known_gpu_uuids or ()),
                max_gpus=max_gpus,
            )
            for phys_idx, handle in selected:
                uuid = _gpu_uuid(handle)
                if uuid:
                    gpu_uuids[phys_idx] = uuid
                # 逐项容错：统一内存设备（GB10）上显存查询不可用，但利用率/温度/功耗都正常，
                # 一处失败就把整机 GPU 指标全丢掉是不划算的。
                util = None
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics[f"gpu.{phys_idx}.pct"] = float(util.gpu)
                except Exception:
                    pass
                mem = nvml_memory(handle)
                if mem is not None and getattr(mem, "total", 0):
                    metrics[f"gpu.{phys_idx}.mem.pct"] = float(100.0 * mem.used / mem.total)
                    metrics[f"gpu.{phys_idx}.mem.value"] = float(mem.used >> 20)
                try:
                    metrics[f"gpu.{phys_idx}.temp"] = float(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except Exception:
                    pass
                try:
                    metrics[f"gpu.{phys_idx}.power"] = float(
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    )
                except Exception:
                    pass
                if util is not None:
                    try:
                        metrics[f"gpu.{phys_idx}.mem.time"] = float(util.memory)
                    except Exception:
                        pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    return {"metrics": metrics, "gpu_uuids": gpu_uuids, "gpu_attribution": attribution}


def collect_hw_snapshot(
    *,
    job_pids: frozenset[int] | list[int] | None = None,
    min_mem_mib: float | None = None,
    include_process: bool = True,
    gpu_fallback: bool = False,
    known_gpu_uuids: frozenset[str] | set[str] | list[str] | None = None,
    max_gpus: int | None = None,
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
        job_pids=pids,
        min_mem_mib=mem_mib,
        include_process=include_process,
        gpu_fallback=gpu_fallback,
        known_gpu_uuids=known_gpu_uuids,
        max_gpus=max_gpus,
    )
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "metrics": hw.get("metrics") or {},
        "gpu_uuids": hw.get("gpu_uuids") or {},
        "gpu_attribution": hw.get("gpu_attribution") or "none",
    }


def _select_gpus(
    devices: list[tuple[int, Any]],
    *,
    job_pids: frozenset[int] | None,
    min_mem_mib: float,
    gpu_fallback: bool,
    known_gpu_uuids: frozenset[str],
    max_gpus: int | None,
) -> tuple[list[tuple[int, Any]], str]:
    """从整机的卡里挑出属于本作业的，并说明是靠什么认出来的。

    序号用**物理 NVML 序号**，不再按选中顺序重排。重排看着整齐（单卡作业永远是 gpu.0），
    代价是选中集合一变，同一张卡的序号就漂移：实测某次作业前 1 分钟只认出物理 1 号卡、
    报成 gpu.0，之后多认了一张，这张卡就变成 gpu.1——同一张卡的曲线在面板上断成两截，
    UUID 一样却出现在两个 key 下。物理序号是这台机器上的稳定标识，不随选择结果变化。

    三档归属，按可信度从高到低：

    pid    NVML 进程列表里有本作业的 actor（或其子进程）。唯一的直接证据。
    sticky 这拍 PID 查空，但本节点先前已经被 pid 认过卡，沿用那批 UUID。colocated 训练
           在生成阶段会把策略卸载、只剩 vLLM 子进程占卡，PID 归属就是在这种时候间歇失灵的。
    mem    从没认出过，只能按显存挑忙卡。这一档会把同机邻居作业的卡也算进来，所以要
           用 max_gpus（PG bundle 给的卡数）封顶，取显存占用最高的那几张。
    """
    if job_pids is not None:
        by_pid = [
            (idx, h) for idx, h in devices if _gpu_belongs_to_job(h, job_pids, min_mem_mib)
        ]
        if by_pid:
            return by_pid, "pid"

        if known_gpu_uuids:
            sticky = [
                (idx, h) for idx, h in devices if (_gpu_uuid(h) or "") in known_gpu_uuids
            ]
            if sticky:
                return sticky, "sticky"

        if not gpu_fallback:
            return [], "none"

    # local / cluster 调试模式（job_pids is None）本来就走显存启发式；
    # scope=job 走到这里则是 PID + sticky 都没结果、且调用方已用 PG 证明本机在跑本作业。
    busy = [(idx, h) for idx, h in devices if _gpu_belongs_to_job(h, None, min_mem_mib)]
    if max_gpus is not None and max_gpus > 0 and len(busy) > max_gpus:
        busy = sorted(busy, key=lambda d: _gpu_mem_used(d[1]), reverse=True)[:max_gpus]
        busy.sort(key=lambda d: d[0])
    return busy, "mem"


def _gpu_mem_used(handle: Any) -> float:
    mem = nvml_memory(handle)
    if mem is None:
        return 0.0
    try:
        return float(getattr(mem, "used", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _gpu_belongs_to_job(
    handle: Any,
    job_pids: frozenset[int] | None,
    min_mem_mib: float,
) -> bool:
    """判定物理 GPU 是否归属本作业。"""
    if job_pids is not None:
        if not job_pids:
            return False
        gpu_pids = _gpu_compute_pids(handle)
        if gpu_pids & job_pids:
            return True
        # NVML 报的是直接占卡的进程，未必是 actor 本身：vLLM 的 EngineCore worker 是
        # actor fork 出来的独立 PID，只比对 actor PID 会漏认，进而把整拍推进显存启发式。
        return any(_has_job_ancestor(pid, job_pids) for pid in gpu_pids)
    # local / cluster 调试模式：无 PID 集合时按显存阈值过滤空闲卡。
    mem = nvml_memory(handle)
    if mem is None:
        # 统一内存设备查不到显存占用，此时无从判断闲忙——宁可多报一张卡，
        # 也好过让这台机器在面板上整个消失。
        return True
    return float(mem.used) >= min_mem_mib * (1024**2)


def _has_job_ancestor(pid: int, job_pids: frozenset[int]) -> bool:
    """pid 的祖先里有没有本作业的 actor（最多往上找 PID_ANCESTOR_MAX_DEPTH 层）。"""
    try:
        import psutil

        proc = psutil.Process(int(pid))
    except Exception:
        return False
    for _ in range(PID_ANCESTOR_MAX_DEPTH):
        try:
            proc = proc.parent()
        except Exception:
            return False
        if proc is None or proc.pid <= 1:
            return False
        if proc.pid in job_pids:
            return True
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
