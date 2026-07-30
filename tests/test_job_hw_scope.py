"""scope=job 的采集范围：哪台机器该出现在硬件面板上，哪些指标该归给谁。

回归背景：异构集群里 driver 常驻 head、训练在 GB10。把 head 也采进来，单机单卡作业的
每张图都会多出一条与训练无关的线；而远端探针采到的「进程内存」量的是探针任务自己
（实测 113 MiB，vLLM worker 是几十个 GB），挂在那个标签下纯属误导。
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from common.observability import hardware_monitor as hm
from common.observability.hardware_monitor import HardwareMonitor

HEAD = "node-head"
GB10 = "node-gb10"


class _Ingest:
    def enqueue_hardware(self, points):
        pass


@pytest.fixture
def monitor(monkeypatch):
    mon = HardwareMonitor(_Ingest(), scope="job")
    monkeypatch.setattr(hm, "current_ray_node_id", lambda: HEAD)
    monkeypatch.setattr(hm, "discover_job_pids", lambda: set())
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(is_initialized=lambda: True))
    return mon


def _fake_local(monkeypatch):
    """替掉本机探针，返回可区分的机器级 + 进程级指标。"""

    def _collect(*, job_pids=None, min_mem_mib=0.0, include_process=True, gpu_fallback=False):
        metrics = {"cpu.pct": 5.0, "mem.pct": 5.2, "mem.proc.avail": 1957322.0}
        if include_process:
            metrics |= {"cpu.thds": 63.0, "mem.proc": 1285.0, "mem.proc.pct": 0.06}
        return {"hostname": "localhost", "pid": 1, "metrics": metrics, "gpu_uuids": {}}

    monkeypatch.setattr(hm, "collect_hw_snapshot", _collect)


def _fake_remote(monkeypatch, seen: list[dict] | None = None):
    """替掉远端 fan-out，记录调用参数并返回 spark-2 的机器级指标。"""

    def _collect_nodes(node_ids, *, job_pids, gpu_fallback=False):
        if seen is not None:
            seen.append(
                {
                    "node_ids": list(node_ids),
                    "job_pids": job_pids,
                    "gpu_fallback": gpu_fallback,
                }
            )
        return [
            {
                "key": key,
                "value": value,
                "node_id": node_ids[0],
                "worker_id": "spark-2",
                "gpu_idx": None,
                "ts": "2026-07-30T08:00:00Z",
            }
            for key, value in (("cpu.pct", 11.2), ("mem.pct", 48.8))
        ]

    monkeypatch.setattr(HardwareMonitor, "_collect_nodes_hw", staticmethod(_collect_nodes))


def _hosts(points: list[dict]) -> set[str]:
    return {p["worker_id"] for p in points}


def _keys_of(points: list[dict], host: str) -> set[str]:
    return {p["key"] for p in points if p["worker_id"] == host}


def test_head_machine_metrics_stay_out_when_training_is_elsewhere(monkeypatch, monitor):
    """head 只跑 driver 时，它的整机 CPU/内存不该出现在面板上。"""
    _fake_local(monkeypatch)
    _fake_remote(monkeypatch)
    monkeypatch.setattr(hm, "discover_gpu_node_ids", lambda: {GB10})
    monkeypatch.setattr(hm, "discover_job_node_ids", lambda **kw: {HEAD})

    points = monitor._collect()

    assert _keys_of(points, "spark-2") == {"cpu.pct", "mem.pct"}
    # localhost 只剩进程级指标，机器级的一个都不该有
    assert _keys_of(points, "localhost") == {"cpu.thds", "mem.proc", "mem.proc.pct"}


def test_remote_probe_does_not_report_its_own_process(monkeypatch, monitor):
    """远端只采机器级指标：那边的 psutil.Process() 是探针自己，不是训练进程。"""
    _fake_local(monkeypatch)
    captured: dict = {}

    def _collect(*, job_pids=None, min_mem_mib=0.0, include_process=True, gpu_fallback=False):
        captured["include_process"] = include_process
        return {"hostname": "spark-2", "pid": 9, "metrics": {"cpu.pct": 11.2}, "gpu_uuids": {}}

    monkeypatch.setattr(hm, "collect_hw_snapshot", _collect)
    monkeypatch.setattr(hm, "discover_gpu_node_ids", lambda: {GB10})
    monkeypatch.setattr(hm, "discover_job_node_ids", lambda **kw: set())

    class _Remote:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            captured.update(kwargs)
            return "ref"

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: True,
            remote=lambda **kw: (lambda fn: _Remote()),
            get=lambda refs, timeout=None: [
                {"hostname": "spark-2", "metrics": {"cpu.pct": 11.2}, "gpu_uuids": {}}
            ],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        SimpleNamespace(
            NodeAffinitySchedulingStrategy=lambda *, node_id, soft: SimpleNamespace(node_id=node_id)
        ),
    )

    monitor._collect()

    assert captured["include_process"] is False


def test_driver_on_training_node_reports_everything(monkeypatch, monitor):
    """单机作业里 driver 和训练在同一台，机器级 + 进程级都归它，不该被拆开。"""
    _fake_local(monkeypatch)
    _fake_remote(monkeypatch)
    monkeypatch.setattr(hm, "discover_gpu_node_ids", lambda: {HEAD})
    monkeypatch.setattr(hm, "discover_job_node_ids", lambda **kw: {HEAD})

    points = monitor._collect()

    assert _hosts(points) == {"localhost"}
    assert _keys_of(points, "localhost") == {
        "cpu.pct",
        "mem.pct",
        "mem.proc.avail",
        "cpu.thds",
        "mem.proc",
        "mem.proc.pct",
    }


def test_gpu_placement_decides_which_machines_are_charted(monkeypatch, monitor):
    seen: list[dict] = []
    _fake_local(monkeypatch)
    _fake_remote(monkeypatch, seen)
    monkeypatch.setattr(hm, "discover_gpu_node_ids", lambda: {GB10})
    monkeypatch.setattr(hm, "discover_job_node_ids", lambda **kw: {HEAD, "node-other"})

    monitor._collect()

    assert seen[0]["node_ids"] == [GB10]
    # PG 认定的训练节点，允许在 PID 归属查空时退回显存启发式认卡
    assert seen[0]["gpu_fallback"] is True
