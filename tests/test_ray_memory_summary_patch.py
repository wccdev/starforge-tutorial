"""Ray memory_summary 超时兜底：GB10 Step1 MemoryTracker 不应再打死作业。"""
from __future__ import annotations

import types

import pytest

from common import ray_memory_summary_patch as patch_mod


@pytest.fixture(autouse=True)
def _reset_patch_state(monkeypatch):
    monkeypatch.setattr(patch_mod, "_PATCHED", False)
    monkeypatch.setenv("NRL_PATCH_RAY_MEMORY_SUMMARY", "1")
    monkeypatch.delenv("NRL_PIN_RESOURCE", raising=False)


def test_enabled_by_env_and_gb10_pin(monkeypatch):
    monkeypatch.setenv("NRL_PATCH_RAY_MEMORY_SUMMARY", "0")
    assert patch_mod.enabled() is False
    monkeypatch.setenv("NRL_PATCH_RAY_MEMORY_SUMMARY", "1")
    assert patch_mod.enabled() is True
    monkeypatch.delenv("NRL_PATCH_RAY_MEMORY_SUMMARY")
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    assert patch_mod.enabled() is True


def test_memory_summary_soft_fails_on_deadline(monkeypatch):
    class InactiveRpcError(RuntimeError):
        pass

    fake_api = types.ModuleType("ray._private.internal_api")

    def boom(*_a, **_k):
        raise InactiveRpcError(
            'status = StatusCode.DEADLINE_EXCEEDED details = "Deadline Exceeded"'
        )

    fake_api.memory_summary = boom
    monkeypatch.setitem(
        __import__("sys").modules, "ray._private.internal_api", fake_api
    )
    # 保证 import ray._private.internal_api 能拿到我们的 fake（简化：直接塞父包）
    ray_mod = types.ModuleType("ray")
    private = types.ModuleType("ray._private")
    monkeypatch.setitem(__import__("sys").modules, "ray", ray_mod)
    monkeypatch.setitem(__import__("sys").modules, "ray._private", private)
    monkeypatch.setitem(__import__("sys").modules, "ray._private.internal_api", fake_api)

    assert patch_mod.apply_patch() is True
    out = fake_api.memory_summary(stats_only=True, num_entries=5)
    assert out.startswith("[nemolab] ray memory_summary skipped")
    assert "DEADLINE" in out or "InactiveRpcError" in out


def test_non_rpc_errors_still_raise(monkeypatch):
    fake_api = types.ModuleType("ray._private.internal_api")

    def boom(*_a, **_k):
        raise ValueError("real bug")

    fake_api.memory_summary = boom
    monkeypatch.setitem(__import__("sys").modules, "ray", types.ModuleType("ray"))
    monkeypatch.setitem(
        __import__("sys").modules, "ray._private", types.ModuleType("ray._private")
    )
    monkeypatch.setitem(__import__("sys").modules, "ray._private.internal_api", fake_api)

    assert patch_mod.apply_patch() is True
    with pytest.raises(ValueError, match="real bug"):
        fake_api.memory_summary()
