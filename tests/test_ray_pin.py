"""卡型 pin 猴子补丁：把 NRL_PIN_RESOURCE 注入 RayVirtualCluster.node_resource_constraints。"""
from __future__ import annotations

import sys
import types

import pytest

import common.ray_pin as ray_pin


class _FakeCluster:
    """模拟 NeMo-RL RayVirtualCluster 的相关属性与 __init__ 语义。"""

    def __init__(self, bundle_ct_per_node_list, *, use_gpus=True,
                 node_resource_constraints=None):
        if node_resource_constraints is not None:
            assert len(node_resource_constraints) == len(bundle_ct_per_node_list)
        self._bundle_ct_per_node_list = bundle_ct_per_node_list
        self.use_gpus = use_gpus
        self.node_resource_constraints = node_resource_constraints


@pytest.fixture()
def fake_vc(monkeypatch):
    """注入 fake nemo_rl.distributed.virtual_cluster，并复位补丁状态。"""
    ray_pin._PATCHED = False
    mod = types.ModuleType("nemo_rl.distributed.virtual_cluster")
    mod.RayVirtualCluster = _FakeCluster
    pkg = types.ModuleType("nemo_rl.distributed")
    pkg.virtual_cluster = mod
    root = types.ModuleType("nemo_rl")
    root.distributed = pkg
    monkeypatch.setitem(sys.modules, "nemo_rl", root)
    monkeypatch.setitem(sys.modules, "nemo_rl.distributed", pkg)
    monkeypatch.setitem(sys.modules, "nemo_rl.distributed.virtual_cluster", mod)
    yield mod
    ray_pin._PATCHED = False


def test_no_pin_env_is_noop(fake_vc, monkeypatch):
    monkeypatch.delenv("NRL_PIN_RESOURCE", raising=False)
    ray_pin.apply_pin_patch()
    c = fake_vc.RayVirtualCluster([8])
    assert c.node_resource_constraints is None


def test_pin_injected_when_none(fake_vc, monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "accelerator_type:H200")
    ray_pin.apply_pin_patch()
    c = fake_vc.RayVirtualCluster([8])
    assert c.node_resource_constraints == [{"accelerator_type:H200": 0.001}]


def test_pin_multi_node_matches_length(fake_vc, monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "acc_gb10")
    ray_pin.apply_pin_patch()
    c = fake_vc.RayVirtualCluster([1, 1])
    assert c.node_resource_constraints == [{"acc_gb10": 0.001}, {"acc_gb10": 0.001}]


def test_pin_merges_into_existing_constraints(fake_vc, monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "accelerator_type:H200")
    ray_pin.apply_pin_patch()
    existing = [{"nvlink_domain_abc": 0.001}]
    c = fake_vc.RayVirtualCluster([8], node_resource_constraints=existing)
    assert c.node_resource_constraints == [
        {"nvlink_domain_abc": 0.001, "accelerator_type:H200": 0.001}
    ]


def test_pin_skipped_for_cpu_cluster(fake_vc, monkeypatch):
    monkeypatch.setenv("NRL_PIN_RESOURCE", "accelerator_type:H200")
    ray_pin.apply_pin_patch()
    c = fake_vc.RayVirtualCluster([4], use_gpus=False)
    assert c.node_resource_constraints is None
