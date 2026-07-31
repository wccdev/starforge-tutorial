"""GB10 NVML NotSupported → torch.cuda.device_memory_used 兜底。"""
from __future__ import annotations

import sys
import types

import pytest

import common.gb10_nvml_patch as patch


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    patch._PATCHED = False
    patch._RAY_HOOK_INSTALLED = False
    monkeypatch.delenv(patch._RAY_HOOK_ENV, raising=False)
    monkeypatch.delenv("NRL_PATCH_NVML_MEMORY", raising=False)


def _fake_torch(*, raise_nvml: bool, allocated: int = 42):
    """最小 torch 替身：只提供 cuda.device_memory_used / memory_allocated。"""
    cuda = types.SimpleNamespace()

    def device_memory_used():
        if raise_nvml:
            raise RuntimeError("Not Supported")
        return 99

    cuda.device_memory_used = device_memory_used
    cuda.memory_allocated = lambda: allocated
    return types.SimpleNamespace(cuda=cuda)


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


def test_fallback_to_memory_allocated(monkeypatch):
    fake = _fake_torch(raise_nvml=True, allocated=1234)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert patch.apply_device_memory_patch() is True
    assert fake.cuda.device_memory_used() == 1234


def test_passthrough_when_nvml_works(monkeypatch):
    fake = _fake_torch(raise_nvml=False)
    monkeypatch.setitem(sys.modules, "torch", fake)

    assert patch.apply_device_memory_patch() is True
    assert fake.cuda.device_memory_used() == 99


def test_worker_setup_is_idempotent(monkeypatch):
    fake = _fake_torch(raise_nvml=True, allocated=7)
    monkeypatch.setitem(sys.modules, "torch", fake)

    patch.worker_setup()
    patch.worker_setup()
    assert fake.cuda.device_memory_used() == 7
    assert getattr(fake.cuda, "_nemolab_device_memory_patched") is True
