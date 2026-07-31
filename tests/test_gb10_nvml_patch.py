"""GB10 NVML NotSupported → pynvml / torch / nemo_rl.utils.nvml 兜底。"""
from __future__ import annotations

import sys
import types

import pytest

import common.gb10_nvml_patch as patch


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    patch._PYNVML_PATCHED = False
    patch._TORCH_PATCHED = False
    patch._RAY_HOOK_INSTALLED = False
    monkeypatch.delenv(patch._RAY_HOOK_ENV, raising=False)
    monkeypatch.delenv("NRL_PATCH_NVML_MEMORY", raising=False)
    monkeypatch.delenv("NRL_PIN_RESOURCE", raising=False)


def _fake_torch(*, raise_nvml: bool, allocated: int = 42, mem_get_info=None):
    cuda = types.SimpleNamespace()

    def device_memory_used():
        if raise_nvml:
            raise RuntimeError("Not Supported")
        return 99

    cuda.device_memory_used = device_memory_used
    cuda.memory_allocated = lambda: allocated
    cuda.is_available = lambda: True
    if mem_get_info is not None:
        cuda.mem_get_info = lambda: mem_get_info
    return types.SimpleNamespace(cuda=cuda)


def _fake_pynvml(*, raise_on_mem: bool = True):
    class _Err(Exception):
        pass

    def get_mem(handle, *a, **k):
        if raise_on_mem:
            raise _Err("Not Supported")
        return types.SimpleNamespace(total=100, free=40, used=60)

    mod = types.ModuleType("pynvml")
    mod.NVMLError = _Err
    mod.nvmlDeviceGetMemoryInfo = get_mem
    return mod


def test_disabled_by_default(monkeypatch):
    assert patch.enabled() is False
    patch.apply_patch()
    assert patch._RAY_HOOK_ENV not in __import__("os").environ


def test_enabled_by_acc_gb10_pin(monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    assert patch.enabled() is True


def test_explicit_off_beats_pin(monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    monkeypatch.setenv("NRL_PATCH_NVML_MEMORY", "0")
    assert patch.enabled() is False


def test_installs_ray_hook_env_when_enabled(monkeypatch):
    monkeypatch.setenv("NRL_PATCH_NVML_MEMORY", "1")
    patch.apply_patch()
    assert __import__("os").environ[patch._RAY_HOOK_ENV] == patch._HOOK_PATH


def test_installs_ray_hook_via_pin(monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    patch.apply_patch()
    assert __import__("os").environ[patch._RAY_HOOK_ENV] == patch._HOOK_PATH


def test_does_not_overwrite_existing_ray_hook(monkeypatch):
    monkeypatch.setenv("NRL_PATCH_NVML_MEMORY", "1")
    monkeypatch.setenv(patch._RAY_HOOK_ENV, "other.module.setup")
    patch.install_ray_worker_hook()
    assert __import__("os").environ[patch._RAY_HOOK_ENV] == "other.module.setup"


def test_pynvml_fallback_uses_cuda_mem_get_info(monkeypatch):
    fake_nvml = _fake_pynvml(raise_on_mem=True)
    fake_torch = _fake_torch(raise_nvml=True, mem_get_info=(7 * 1024**3, 20 * 1024**3))
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(patch, "_read_meminfo", lambda: (0, 0))

    assert patch.apply_pynvml_patch() is True
    info = fake_nvml.nvmlDeviceGetMemoryInfo(handle=0)
    assert info.free == 7 * 1024**3
    assert info.total == 20 * 1024**3
    assert info.used == 13 * 1024**3


def test_uma_prefers_memavailable_over_tiny_cuda_free(monkeypatch):
    """复现 refit buffer 过小：cuda free~1GiB，但系统还有几十 GiB MemAvailable。"""
    monkeypatch.setattr(
        patch, "_read_meminfo", lambda: (128 * 1024**3, 80 * 1024**3)
    )
    fake_torch = _fake_torch(raise_nvml=True, mem_get_info=(1 * 1024**3, 20 * 1024**3))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info = patch._uma_memory_info()
    assert info.free == 80 * 1024**3
    assert info.total == 128 * 1024**3


def test_pynvml_passthrough_when_supported(monkeypatch):
    fake_nvml = _fake_pynvml(raise_on_mem=False)
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    assert patch.apply_pynvml_patch() is True
    info = fake_nvml.nvmlDeviceGetMemoryInfo(handle=0)
    assert info.free == 40


def test_torch_fallback_to_memory_allocated(monkeypatch):
    fake = _fake_torch(raise_nvml=True, allocated=1234)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert patch.apply_device_memory_patch() is True
    assert fake.cuda.device_memory_used() == 1234


def test_nemo_rl_get_free_memory_bytes_fallback(monkeypatch):
    """复现本次崩溃：refit → get_free_memory_bytes → NVML NotSupported。"""
    nrl_nvml = types.ModuleType("nemo_rl.utils.nvml")

    def boom(device_idx: int) -> float:
        raise RuntimeError(f"Failed to get free memory for device {device_idx}: Not Supported")

    nrl_nvml.get_free_memory_bytes = boom
    pkg = types.ModuleType("nemo_rl.utils")
    pkg.nvml = nrl_nvml
    root = types.ModuleType("nemo_rl")
    root.utils = pkg
    for name, m in [
        ("nemo_rl", root),
        ("nemo_rl.utils", pkg),
        ("nemo_rl.utils.nvml", nrl_nvml),
    ]:
        monkeypatch.setitem(sys.modules, name, m)

    fake_torch = _fake_torch(raise_nvml=True, mem_get_info=(5 * 1024**3, 16 * 1024**3))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(patch, "_read_meminfo", lambda: (0, 0))

    assert patch.apply_nemo_rl_nvml_patch() is True
    assert nrl_nvml.get_free_memory_bytes(0) == float(5 * 1024**3)


def test_worker_setup_is_idempotent(monkeypatch):
    fake_nvml = _fake_pynvml(raise_on_mem=True)
    fake_torch = _fake_torch(raise_nvml=True, allocated=7, mem_get_info=(1, 2))
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(patch, "_read_meminfo", lambda: (0, 0))

    patch.worker_setup()
    patch.worker_setup()
    assert fake_nvml.nvmlDeviceGetMemoryInfo(0).free == 1
    assert fake_torch.cuda.device_memory_used() == 7
