"""GB10 / DGX Spark：绕过 NVML GetMemoryInfo NotSupported。

GB10 是 CPU/GPU 统一内存，没有独立 framebuffer；
`pynvml.nvmlDeviceGetMemoryInfo` 会抛 `NVMLError_NotSupported`。

会炸的调用点包括：
  - `torch.cuda.device_memory_used` → Megatron `log_gpu_memory`
  - `nemo_rl.utils.nvml.get_free_memory_bytes` → colocated vLLM refit 算 buffer

本模块在 **pynvml 底层** 打兜底（一处修好所有调用方），并经 Ray
`worker_process_setup_hook` 注入到 MegatronPolicyWorker / VllmGenerationWorker。

启用（任一即可）：
  - `NRL_PATCH_NVML_MEMORY=1`（`cluster/gb10-spark/env.sh` 默认打开）
  - `NRL_PIN_RESOURCE=acc_gb10`（服务端按 GB10 profile 注入时自动开）
关闭：`NRL_PATCH_NVML_MEMORY=0`（显式优先于 pin 自动启用）。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

# Ray 内部约定：default_worker 启动时读这个 env，load_class 后调用。
# 见 ray/_private/ray_constants.py::WORKER_PROCESS_SETUP_HOOK_ENV_VAR
_RAY_HOOK_ENV = "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR"
_HOOK_PATH = "common.gb10_nvml_patch.worker_setup"

_PYNVML_PATCHED = False
_TORCH_PATCHED = False
_RAY_HOOK_INSTALLED = False


def enabled() -> bool:
    """显式 0 关闭；显式 1 或 pin 到 acc_gb10 时打开。"""
    raw = os.environ.get("NRL_PATCH_NVML_MEMORY", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    pin = os.environ.get("NRL_PIN_RESOURCE", "").strip()
    return pin == "acc_gb10"


def _uma_memory_info() -> SimpleNamespace:
    """UMA 回退：优先 CUDA runtime，再退到 /proc/meminfo。"""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = max(int(total) - int(free), 0)
            return SimpleNamespace(total=int(total), free=int(free), used=used)
    except Exception:
        pass

    total = free = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            vals: dict[str, int] = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    vals[parts[0].rstrip(":")] = int(parts[1]) * 1024
        total = vals.get("MemTotal", 0)
        free = vals.get("MemAvailable", vals.get("MemFree", 0))
    except Exception:
        pass
    used = max(total - free, 0)
    return SimpleNamespace(total=total, free=free, used=used)


def apply_pynvml_patch() -> bool:
    """给 pynvml.nvmlDeviceGetMemoryInfo 打兜底。幂等。

    这是主补丁：NeMo-RL `get_free_memory_bytes`、PyTorch `device_memory_used`
    都走这条 API。
    """
    global _PYNVML_PATCHED
    if _PYNVML_PATCHED:
        return True
    try:
        import pynvml
    except Exception:
        return False
    if getattr(pynvml, "_nemolab_nvml_mem_patched", False):
        _PYNVML_PATCHED = True
        return True

    orig = pynvml.nvmlDeviceGetMemoryInfo

    def _safe_get_memory_info(handle, *args, **kwargs):
        try:
            return orig(handle, *args, **kwargs)
        except Exception:
            return _uma_memory_info()

    pynvml.nvmlDeviceGetMemoryInfo = _safe_get_memory_info
    pynvml._nemolab_nvml_mem_patched = True
    _PYNVML_PATCHED = True
    print(
        "[nemolab] GB10 NVML patch: pynvml.nvmlDeviceGetMemoryInfo "
        "→ CUDA/meminfo fallback on NotSupported"
    )
    return True


def apply_device_memory_patch() -> bool:
    """给 torch.cuda.device_memory_used 打兜底（pynvml 补丁的双保险）。幂等。"""
    global _TORCH_PATCHED
    if _TORCH_PATCHED:
        return True
    try:
        import torch
    except Exception:
        return False

    orig = getattr(torch.cuda, "device_memory_used", None)
    if orig is None:
        return False
    if getattr(torch.cuda, "_nemolab_device_memory_patched", False):
        _TORCH_PATCHED = True
        return True

    def _safe_device_memory_used(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except Exception:
            try:
                return int(torch.cuda.memory_allocated())
            except Exception:
                return 0

    torch.cuda.device_memory_used = _safe_device_memory_used
    torch.cuda._nemolab_device_memory_patched = True
    _TORCH_PATCHED = True
    print(
        "[nemolab] GB10 NVML patch: torch.cuda.device_memory_used "
        "→ fallback to memory_allocated on NotSupported"
    )
    return True


def apply_nemo_rl_nvml_patch() -> bool:
    """若 nemo_rl.utils.nvml 已可导入，再包一层 get_free_memory_bytes。"""
    try:
        import nemo_rl.utils.nvml as nrl_nvml
    except Exception:
        return False
    if getattr(nrl_nvml, "_nemolab_free_mem_patched", False):
        return True

    orig = nrl_nvml.get_free_memory_bytes

    def _safe_get_free_memory_bytes(device_idx: int) -> float:
        try:
            return float(orig(device_idx))
        except Exception:
            info = _uma_memory_info()
            print(
                f"[nemolab] GB10 NVML patch: get_free_memory_bytes({device_idx}) "
                f"→ fallback free={info.free / (1024**3):.2f}GiB"
            )
            return float(info.free)

    nrl_nvml.get_free_memory_bytes = _safe_get_free_memory_bytes
    nrl_nvml._nemolab_free_mem_patched = True
    print("[nemolab] GB10 NVML patch: nemo_rl.utils.nvml.get_free_memory_bytes wrapped")
    return True


def worker_setup() -> None:
    """Ray `worker_process_setup_hook` 入口（每个 worker/actor 进程启动时调用）。"""
    apply_pynvml_patch()
    apply_device_memory_patch()
    apply_nemo_rl_nvml_patch()


def install_ray_worker_hook() -> None:
    """让后续 init_ray() 把 hook 路径塞进所有 Ray worker 的 env_vars。

    NeMo-RL 的 init_ray 会 `runtime_env={"env_vars": dict(os.environ)}`，
    所以在这里 setdefault 内部 hook env 即可——不必再 monkeypatch ray.init。
    """
    global _RAY_HOOK_INSTALLED
    if _RAY_HOOK_INSTALLED or not enabled():
        return
    existing = os.environ.get(_RAY_HOOK_ENV, "").strip()
    if existing and existing != _HOOK_PATH:
        print(
            f"[nemolab] GB10 NVML patch: Ray hook env 已被占用（{existing}），跳过注入"
        )
        _RAY_HOOK_INSTALLED = True
        return
    os.environ[_RAY_HOOK_ENV] = _HOOK_PATH
    _RAY_HOOK_INSTALLED = True
    print(f"[nemolab] GB10 NVML patch: Ray worker hook → {_HOOK_PATH}")


def apply_patch() -> None:
    """boot 入口：driver 本地补丁 + 注册 Ray worker hook。"""
    if not enabled():
        return
    install_ray_worker_hook()
    apply_pynvml_patch()
    apply_device_memory_patch()
    apply_nemo_rl_nvml_patch()
