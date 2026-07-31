"""GB10 / DGX Spark：绕过 NVML GetMemoryInfo NotSupported。

GB10 是 CPU/GPU 统一内存，没有独立 framebuffer；
`pynvml.nvmlDeviceGetMemoryInfo` / `torch.cuda.device_memory_used` 会抛
`NVMLError_NotSupported`。Megatron 的 `log_gpu_memory` → `device_memory_summary`
在 `prepare_for_generation` 里无条件调用它，会把已加载完模型的作业直接打挂。

本模块做两件事：
1. 给 `torch.cuda.device_memory_used` 加 try/except，失败时回退到
   `torch.cuda.memory_allocated()`（离散卡路径不变）。
2. 通过 Ray 的 `worker_process_setup_hook` 机制，让 **MegatronPolicyWorker 等
   Ray actor** 在启动时也打上同一补丁（driver 侧 patch 进不了隔离 venv worker）。

启用（任一即可）：
  - `NRL_PATCH_NVML_MEMORY=1`（`cluster/gb10-spark/env.sh` 默认打开）
  - `NRL_PIN_RESOURCE=acc_gb10`（服务端按 GB10 profile 注入时自动开；
    `gb10-spark-single` 无独立 env.sh 也能盖住）
关闭：`NRL_PATCH_NVML_MEMORY=0`（显式优先于 pin 自动启用）。
"""
from __future__ import annotations

import os

# Ray 内部约定：default_worker 启动时读这个 env，load_class 后调用。
# 见 ray/_private/ray_constants.py::WORKER_PROCESS_SETUP_HOOK_ENV_VAR
_RAY_HOOK_ENV = "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR"
_HOOK_PATH = "common.gb10_nvml_patch.worker_setup"

_PATCHED = False
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


def apply_device_memory_patch() -> bool:
    """给当前进程的 torch.cuda.device_memory_used 打兜底。幂等。"""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import torch
    except Exception:
        return False

    orig = getattr(torch.cuda, "device_memory_used", None)
    if orig is None:
        return False
    if getattr(torch.cuda, "_nemolab_device_memory_patched", False):
        _PATCHED = True
        return True

    def _safe_device_memory_used(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except Exception:
            # UMA / GB10：NVML 不支持；用当前进程 CUDA 分配量顶上，够日志用。
            try:
                return int(torch.cuda.memory_allocated())
            except Exception:
                return 0

    torch.cuda.device_memory_used = _safe_device_memory_used
    torch.cuda._nemolab_device_memory_patched = True
    _PATCHED = True
    print(
        "[nemolab] GB10 NVML patch: torch.cuda.device_memory_used "
        "→ fallback to memory_allocated on NotSupported"
    )
    return True


def worker_setup() -> None:
    """Ray `worker_process_setup_hook` 入口（每个 worker/actor 进程启动时调用）。"""
    apply_device_memory_patch()


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
    # driver 侧也打上（多数路径用不到，但无害且便于本地复现）。
    apply_device_memory_patch()
