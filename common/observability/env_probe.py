"""运行环境快照采集：Python / 依赖 / Git / 系统 metadata。"""
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


def collect_node_hardware(
    *,
    job_pids: list[int] | frozenset[int] | None = None,
    min_mem_mib: float | None = None,
    gpu_fallback: bool = False,
    known_gpu_uuids: list[str] | frozenset[str] | None = None,
    max_gpus: int | None = None,
) -> dict[str, Any]:
    """本机静态硬件 + 主机名。设计为可被 `ray.remote` 派到任意节点上就地执行。

    与 `collect_environment` 的区别是不带 overview/packages：那些是 driver 进程的
    上下文（命令行、cwd、pip 列表），远端节点上采了也没有意义。

    GPU 过滤参数与 `collect_hw_snapshot` 对齐：作业环境页应只展示本作业占用的卡，
    不能把同机邻居 / 整机库存（例如 8×H200 里只用 2 张）全列出来。
    """
    return {
        "hostname": socket.gethostname(),
        **_collect_hardware(
            job_pids=job_pids,
            min_mem_mib=min_mem_mib,
            gpu_fallback=gpu_fallback,
            known_gpu_uuids=known_gpu_uuids,
            max_gpus=max_gpus,
        ),
    }


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
        "image": collect_image_id(),
    }


# 平台镜像在构建期写入的指纹文件（见 console 的 deploy/ray-cluster/Dockerfile）。
IMAGE_FINGERPRINT_FILE = "/etc/starforge-image"


def collect_image_id() -> str:
    """本作业跑在哪个容器镜像里。

    代码版本（git commit）和配置版本（config_sha）本来就有记录，唯独环境没有——
    而依赖一升级，老作业就永远说不清结果差异到底来自代码还是环境。这里把它补上。

    优先级：统一 launcher 已导出的 FORGE_IMAGE（集群侧唯一事实来源）>
    指纹文件 > 官方 NeMo-RL 镜像自带的 build id > unknown。
    """
    if env_image := os.environ.get("FORGE_IMAGE", "").strip():
        return env_image

    fields = _read_fingerprint_file(IMAGE_FINGERPRINT_FILE)
    if tag := fields.get("FORGE_IMAGE_TAG"):
        return f"{tag}@{fields.get('FORGE_IMAGE_BUILD_ID', 'unknown')}"

    # 还没换成平台镜像时，官方镜像自带这两个环境变量，聊胜于无。
    if build_id := os.environ.get("NVIDIA_BUILD_ID", "").strip():
        return f"nemo-rl-official@{build_id}"
    if commit := os.environ.get("NEMO_RL_COMMIT", "").strip():
        return f"nemo-rl@{commit}"
    return "unknown"


def _read_fingerprint_file(path: str) -> dict[str, str]:
    """读 KEY=VALUE 指纹文件；读不到就当作没有（采集是旁路，不能因此报错）。"""
    out: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def _collect_hardware(
    *,
    job_pids: list[int] | frozenset[int] | None = None,
    min_mem_mib: float | None = None,
    gpu_fallback: bool = False,
    known_gpu_uuids: list[str] | frozenset[str] | None = None,
    max_gpus: int | None = None,
) -> dict[str, Any]:
    cpu = _collect_cpu()
    gpu = _collect_nvidia_gpu(
        job_pids=job_pids,
        min_mem_mib=min_mem_mib,
        gpu_fallback=gpu_fallback,
        known_gpu_uuids=known_gpu_uuids,
        max_gpus=max_gpus,
    )
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


def _collect_nvidia_gpu(
    *,
    job_pids: list[int] | frozenset[int] | None = None,
    min_mem_mib: float | None = None,
    gpu_fallback: bool = False,
    known_gpu_uuids: list[str] | frozenset[str] | None = None,
    max_gpus: int | None = None,
) -> dict[str, Any] | None:
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        from common.observability.hw_probe import DEFAULT_MIN_MEM_MIB, _select_gpus

        n = pynvml.nvmlDeviceGetCount()
        devices: list[tuple[int, Any]] = []
        for physical in range(n):
            try:
                devices.append((physical, pynvml.nvmlDeviceGetHandleByIndex(physical)))
            except Exception:
                continue

        # driver 侧 collect_environment() 不传过滤参数：保留整机清单作兜底。
        # 作业节点探针会带上 job_pids / max_gpus，只报本作业的卡。
        filter_requested = (
            job_pids is not None
            or max_gpus is not None
            or bool(known_gpu_uuids)
            or gpu_fallback
        )
        if filter_requested:
            pids: frozenset[int] | None
            if job_pids is None:
                pids = None
            elif isinstance(job_pids, frozenset):
                pids = job_pids
            else:
                pids = frozenset(int(x) for x in job_pids)
            selected, _attribution = _select_gpus(
                devices,
                job_pids=pids,
                min_mem_mib=(
                    DEFAULT_MIN_MEM_MIB if min_mem_mib is None else float(min_mem_mib)
                ),
                gpu_fallback=gpu_fallback,
                known_gpu_uuids=frozenset(known_gpu_uuids or ()),
                max_gpus=max_gpus,
            )
            # PG / FORGE_CLUSTER_GPUS_PER_NODE 已告诉我们本节点卡数，但 PID/显存这拍
            # 还认不出时：按物理序号截到 max_gpus。静态页同型号卡型号/显存一样，
            # 张数正确比物理序号完美更重要；没有 max_gpus 时宁可不报卡，也别把
            # head 整机库存写进「运行节点」（异构集群常见误报）。
            if (
                not selected
                and max_gpus is not None
                and max_gpus > 0
                and devices
            ):
                selected = devices[: int(max_gpus)]
            elif not selected and filter_requested:
                selected = []
            indices = [idx for idx, _ in selected]
        else:
            # 仅 driver 侧 collect_environment()：整机清单作「调度端」兜底，UI 会标注来源。
            indices = [idx for idx, _ in devices]

        described = [d for d in (_describe_gpu(pynvml, i) for i in indices) if d]
        return {
            "vendor": "nvidia",
            "driver_version": _nvml_text(pynvml.nvmlSystemGetDriverVersion),
            "cuda_version": _cuda_version(),
            "count": len(described),
            "devices": described,
        }
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _describe_gpu(pynvml: Any, index: int) -> dict[str, Any] | None:
    """单张卡的静态描述。"""
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


def _gpu_memory_gb(handle: Any) -> float:
    """单卡显存容量（GB）；查询失败直接抛错。"""
    from common.observability.hw_probe import nvml_memory

    return round(int(nvml_memory(handle).total) / (1024**3))


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
