"""运行环境快照采集（对齐 SwanLab ProbePython 的 metadata / requirements 思路）。"""
from __future__ import annotations

import multiprocessing
import os
import platform
import re
import socket
import subprocess
import sys
from typing import Any


def collect_environment() -> dict[str, Any]:
    return {
        "overview": _collect_overview(),
        "hardware": _collect_hardware(),
        "packages": _collect_packages(),
    }


def collect_node_hardware() -> dict[str, Any]:
    """本机静态硬件 + 主机名。设计为可被 `ray.remote` 派到任意节点上就地执行。

    与 `collect_environment` 的区别是不带 overview/packages：那些是 driver 进程的
    上下文（命令行、cwd、pip 列表），远端节点上采了也没有意义。
    """
    return {"hostname": socket.gethostname(), **_collect_hardware()}


def _collect_overview() -> dict[str, Any]:
    os_pretty = None
    freedesktop = getattr(platform, "freedesktop_os_release", None)
    if freedesktop is not None:
        try:
            os_pretty = freedesktop().get("PRETTY_NAME")
        except Exception:
            pass
    return {
        "os": platform.platform(),
        "os_pretty": os_pretty,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "command": " ".join(sys.argv),
    }


def _collect_hardware() -> dict[str, Any]:
    cpu = _collect_cpu()
    gpu = _collect_nvidia_gpu()
    out: dict[str, Any] = {}
    if cpu:
        out["cpu"] = cpu
    if gpu:
        out["gpu"] = gpu
    return out


def _collect_cpu() -> dict[str, Any] | None:
    brand = _cpu_brand()
    cores = multiprocessing.cpu_count()
    memory_gb = _system_memory_gb()
    if not brand and not cores and memory_gb is None:
        return None
    return {
        "brand": brand,
        "cores": cores,
        "memory_gb": memory_gb,
    }


def _cpu_brand() -> str | None:
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    brand = platform.processor()
    return brand or None


def _collect_nvidia_gpu() -> dict[str, Any] | None:
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        count = pynvml.nvmlDeviceGetCount()
        return {
            "vendor": "nvidia",
            "driver_version": _nvml_text(pynvml.nvmlSystemGetDriverVersion),
            "cuda_version": _cuda_version(),
            "count": count,
            "devices": [d for d in (_describe_gpu(pynvml, i) for i in range(count)) if d],
        }
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _describe_gpu(pynvml: Any, index: int) -> dict[str, Any] | None:
    """单张卡的静态描述。任何一项查不到都不应拖垮整份快照。

    尤其是显存：GB10 / DGX Spark 用统一内存，没有独立显存，NVML 会直接返回 NotSupported。
    """
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:
        return None
    name = _nvml_text(pynvml.nvmlDeviceGetName, handle)
    return {
        "index": index,
        "name": name or f"GPU {index}",
        "memory_gb": _gpu_memory_gb(handle),
    }


def _gpu_memory_gb(handle: Any) -> float | None:
    """单卡显存容量（GB）。

    统一内存架构下 NVML 查不到显存，此时用系统内存顶上：那本来就是这块 GPU 能用的
    全部容量（DGX Spark 标称的 128 GB 即由此而来），比留空更贴近事实。
    """
    from common.observability.hw_probe import nvml_memory

    mem = nvml_memory(handle)
    total = getattr(mem, "total", None) if mem is not None else None
    if total:
        return round(int(total) / (1024**3))
    return _system_memory_gb()


def _system_memory_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3))
    except Exception:
        return None


def _nvml_text(fn: Any, *args: Any) -> str | None:
    try:
        raw = fn(*args)
    except Exception:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip() or None


def _cuda_version() -> str | None:
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, timeout=5)
        for line in out.splitlines():
            if "release" in line.lower():
                m = re.search(r"release\s+([\d.]+)", line, re.I)
                if m:
                    return m.group(1)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, timeout=5)
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _collect_packages() -> str:
    for cmd in (
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        ["uv", "pip", "list", "--format=freeze"],
    ):
        try:
            out = subprocess.check_output(cmd, text=True, timeout=20)
            if out.strip():
                return out
        except Exception:
            continue
    return ""
