"""卡型 pin：把 NRL_PIN_RESOURCE 合并进 NeMo-RL 建 placement group 时的 bundle 规格。

这些用例盯的是 2026-07-30 那次事故：上一版补丁往 `RayVirtualCluster.node_resource_constraints`
赋值，而现场那版 NeMo-RL 根本没有这个字段，于是 pin 静默失效、gb10 的作业跑到 H200 上
撞进别人的显存里 OOM。所以除了「注入对不对」，这里更要守住「做不到时必须炸」。
"""
from __future__ import annotations

import sys
import types

import pytest

import common.ray_pin as ray_pin
from common.ray_pin import PinError

_HEAD = {"Alive": True, "Resources": {"GPU": 8.0, "CPU": 128.0, "accelerator_type:H200": 1.0}}
_SPARK = {"Alive": True, "Resources": {"GPU": 1.0, "CPU": 20.0, "acc_gb10": 1.0}}


def _fake_ray(nodes, initialized=True):
    mod = types.ModuleType("ray")
    mod.is_initialized = lambda: initialized
    mod.nodes = lambda: list(nodes)
    return mod


@pytest.fixture()
def vc(monkeypatch):
    """注入 fake nemo_rl.distributed.virtual_cluster + ray，并复位补丁状态。

    这里 fake 的 `placement_group` 是 Ray 的公开 API（NeMo-RL 只是转发调用），
    不是 NeMo-RL 的内部字段——上一版栽跟头正是因为假设了后者。
    """
    ray_pin._PATCHED = False
    calls = []

    def _placement_group(bundles, strategy="PACK", name="", **kw):
        calls.append({"bundles": bundles, "strategy": strategy, "name": name})
        return f"pg:{name}"

    mod = types.ModuleType("nemo_rl.distributed.virtual_cluster")
    mod.placement_group = _placement_group
    pkg = types.ModuleType("nemo_rl.distributed")
    pkg.virtual_cluster = mod
    root = types.ModuleType("nemo_rl")
    root.distributed = pkg
    for name, m in [
        ("nemo_rl", root),
        ("nemo_rl.distributed", pkg),
        ("nemo_rl.distributed.virtual_cluster", mod),
        ("ray", _fake_ray([_HEAD, _SPARK])),
    ]:
        monkeypatch.setitem(sys.modules, name, m)
    mod.calls = calls
    yield mod
    ray_pin._PATCHED = False


def test_no_pin_env_is_noop(vc, monkeypatch):
    monkeypatch.delenv("NRL_PIN_RESOURCE", raising=False)
    before = vc.placement_group
    ray_pin.apply_pin_patch()
    assert vc.placement_group is before  # 完全没碰上游


def test_pin_injected_into_gpu_bundles(vc, monkeypatch):
    """复现事故那一刻：grpo_policy_cluster-node0 的 bundle 当时只有 {GPU, CPU}。

    NeMo-RL 是以关键字传 bundles 的，这里照它的调用形态来。
    """
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    ray_pin.apply_pin_patch()

    vc.placement_group(
        bundles=[{"CPU": 2.0, "GPU": 1.0}], strategy="PACK", name="grpo_policy_cluster-node0",
    )

    assert vc.calls[0]["bundles"] == [{"CPU": 2.0, "GPU": 1.0, "acc_gb10": 0.001}]
    # 转发的其余参数原样保留
    assert vc.calls[0]["strategy"] == "PACK"
    assert vc.calls[0]["name"] == "grpo_policy_cluster-node0"


def test_pin_injected_into_every_gpu_bundle(vc, monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    ray_pin.apply_pin_patch()

    vc.placement_group([{"CPU": 1.0, "GPU": 1.0}] * 2)

    assert all(b["acc_gb10"] == 0.001 for b in vc.calls[0]["bundles"])


def test_cpu_only_bundles_are_left_alone(vc, monkeypatch):
    """建 venv 的 STRICT_SPREAD 组要按节点铺开，pin 上去就无处可去了。"""
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    ray_pin.apply_pin_patch()

    vc.placement_group([{"CPU": 1.0}] * 3, strategy="STRICT_SPREAD")

    assert vc.calls[0]["bundles"] == [{"CPU": 1.0}] * 3


def test_existing_constraints_are_preserved(vc, monkeypatch):
    """已有约束（如 NVLink 域 pin）保持不动，只补自己那一项。"""
    monkeypatch.setenv("NRL_PIN_RESOURCE", "accelerator_type:H200")
    ray_pin.apply_pin_patch()

    vc.placement_group([{"CPU": 2.0, "GPU": 1.0, "nvlink_domain_abc": 0.001}])

    assert vc.calls[0]["bundles"][0] == {
        "CPU": 2.0,
        "GPU": 1.0,
        "nvlink_domain_abc": 0.001,
        "accelerator_type:H200": 0.001,
    }


def test_caller_bundles_are_not_mutated(vc, monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    ray_pin.apply_pin_patch()

    original = [{"CPU": 2.0, "GPU": 1.0}]
    vc.placement_group(original)

    assert original == [{"CPU": 2.0, "GPU": 1.0}]


def test_patch_is_idempotent(vc, monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    ray_pin.apply_pin_patch()
    once = vc.placement_group
    ray_pin.apply_pin_patch()
    assert vc.placement_group is once  # 没有套第二层


def test_missing_upstream_symbol_raises(vc, monkeypatch):
    """事故守门员：上游改了 API，必须当场炸，而不是静默放行。"""
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    del vc.placement_group

    with pytest.raises(PinError, match="上游 API 已变"):
        ray_pin.apply_pin_patch()


def test_pin_resource_absent_from_cluster_raises(vc, monkeypatch):
    """卡型没上线时当场说清楚，别让 placement group 干等到超时。"""
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    monkeypatch.setitem(sys.modules, "ray", _fake_ray([_HEAD]))  # 只剩 H200
    ray_pin.apply_pin_patch()

    with pytest.raises(PinError) as e:
        vc.placement_group([{"CPU": 2.0, "GPU": 1.0}])

    assert "acc_gb10" in str(e.value)
    assert "accelerator_type:H200" in str(e.value)  # 提示集群里实际有什么
    assert not vc.calls  # 没有把没 pin 的 bundle 放出去


def test_skips_cluster_check_before_ray_init(vc, monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    monkeypatch.setitem(sys.modules, "ray", _fake_ray([], initialized=False))
    ray_pin.apply_pin_patch()

    vc.placement_group([{"CPU": 2.0, "GPU": 1.0}])

    assert vc.calls[0]["bundles"][0]["acc_gb10"] == 0.001
