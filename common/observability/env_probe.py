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
    memory_gb = None
    try:
        import psutil

        memory_gb = round(psutil.virtual_memory().total / (1024**3))
    except Exception:
        pass
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
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8")
        count = pynvml.nvmlDeviceGetCount()
        devices = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                {
                    "index": i,
                    "name": name,
                    "memory_gb": round(int(mem.total) / (1024**3)),
                }
            )
        return {
            "vendor": "nvidia",
            "driver_version": driver,
            "cuda_version": _cuda_version(),
            "count": count,
            "devices": devices,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


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
