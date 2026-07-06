"""硬件监控：仅采集与当前 Ray 作业相关的节点资源。

默认 scope=job：本 job alive actors 所在节点；GPU 按 actor PID 归属到物理卡（非整机枚举）。
多节点作业自动 fan-out 到这些节点；不扫整个 Ray 集群无关机器。

scope=local  — 仅本进程所在机器（纯 SwanLab 行为，按显存阈值过滤空闲卡）
scope=cluster — 全集群 alive 节点（调试用，NEMOLAB_MONITOR_CLUSTER=1 等价）
"""
from __future__ import annotations

import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Literal

from common.observability.hw_probe import DEFAULT_MIN_MEM_MIB, collect_hw_snapshot
from common.observability.job_nodes import (
    current_ray_node_id,
    discover_job_node_ids,
    discover_job_pids,
)
from common.observability.sampling import swanlab_monitor_interval
from common.observability.util import scalarize_metric

MonitorScope = Literal["local", "job", "cluster"]
NODE_DISCOVERY_TTL = 60.0


class HardwareMonitor:
    def __init__(
        self,
        ingest,
        *,
        collection_interval: float = 10.0,
        dynamic_interval: bool = True,
        scope: MonitorScope = "job",
    ):
        self.ingest = ingest
        self.base_interval = max(5.0, float(collection_interval))
        self.dynamic_interval = dynamic_interval
        self.scope: MonitorScope = scope
        self.min_mem_mib = float(
            os.environ.get("NEMOLAB_GPU_MIN_MEM_MIB", str(DEFAULT_MIN_MEM_MIB))
        )
        self._samples_collected = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._node_cache: tuple[float, set[str]] | None = None
        self._pid_cache: tuple[float, frozenset[int]] | None = None

    def start(self) -> None:
        if self.scope in ("job", "cluster"):
            try:
                import ray  # noqa: F401
            except ImportError:
                if self.scope == "cluster":
                    print("NeMoLab hardware monitor skipped: ray not available")
                    return
                self.scope = "local"
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="NeMoLab·Monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=120)

    def _sleep_interval(self) -> float:
        return swanlab_monitor_interval(
            self._samples_collected,
            base_interval=self.base_interval,
            dynamic=self.dynamic_interval,
        )

    def _loop(self) -> None:
        while self._running:
            try:
                points = self._collect()
                if points:
                    self.ingest.enqueue_hardware(points)
                    self._samples_collected += 1
            except Exception as e:
                print(f"NeMoLab hardware monitor error: {e}")
            time.sleep(self._sleep_interval())

    def _collect(self) -> list[dict]:
        if self.scope == "local":
            return self._collect_local_hw(
                node_id=current_ray_node_id(),
                job_pids=None,
            )
        if self.scope == "cluster":
            return self._collect_cluster_hw()
        return self._collect_job_hw()

    def _job_node_ids(self) -> set[str]:
        now = time.time()
        if self._node_cache and now - self._node_cache[0] < NODE_DISCOVERY_TTL:
            return self._node_cache[1]
        ids = discover_job_node_ids()
        self._node_cache = (now, ids)
        return ids

    def _job_pids(self) -> frozenset[int]:
        now = time.time()
        if self._pid_cache and now - self._pid_cache[0] < NODE_DISCOVERY_TTL:
            return self._pid_cache[1]
        pids = frozenset(discover_job_pids())
        self._pid_cache = (now, pids)
        return pids

    def _collect_job_hw(self) -> list[dict]:
        import ray

        if not ray.is_initialized():
            return self._collect_local_hw(job_pids=None)
        node_ids = self._job_node_ids()
        job_pids = self._job_pids()
        if not node_ids:
            return self._collect_local_hw(job_pids=job_pids)
        current = current_ray_node_id()
        points: list[dict] = []
        if current and current in node_ids:
            points.extend(
                self._collect_local_hw(node_id=current, job_pids=job_pids)
            )
        remote = sorted(nid for nid in node_ids if nid != current)
        if remote:
            points.extend(self._collect_nodes_hw(remote, job_pids=job_pids))
        return points

    def _collect_local_hw(
        self,
        *,
        node_id: str | None = None,
        job_pids: frozenset[int] | None = None,
    ) -> list[dict]:
        snap = collect_hw_snapshot(
            job_pids=job_pids,
            min_mem_mib=self.min_mem_mib,
        )
        ts = datetime.now(timezone.utc).isoformat()
        worker_id = snap.get("hostname") or socket.gethostname()
        return _snap_to_points(
            snap, ts=ts, node_id=node_id, worker_id=worker_id
        )

    def _collect_nodes_hw(
        self,
        node_ids: list[str],
        *,
        job_pids: frozenset[int] | None,
    ) -> list[dict]:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        if not node_ids:
            return []
        remote_collect = ray.remote(num_cpus=0)(collect_hw_snapshot)
        ts = datetime.now(timezone.utc).isoformat()
        points: list[dict] = []
        futures = []
        pid_arg = list(job_pids) if job_pids is not None else None
        for node_id in node_ids:
            futures.append(
                remote_collect.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(job_pids=pid_arg, min_mem_mib=self.min_mem_mib)
            )
        snapshots = ray.get(futures)
        for node_id, snap in zip(node_ids, snapshots, strict=False):
            worker_id = snap.get("hostname") or node_id
            points.extend(
                _snap_to_points(
                    snap, ts=ts, node_id=node_id, worker_id=worker_id
                )
            )
        return points

    def _collect_cluster_hw(self) -> list[dict]:
        import ray

        if not ray.is_initialized():
            return []
        node_ids = [
            str(n.get("NodeID"))
            for n in ray.nodes()
            if n.get("Alive") and n.get("NodeID")
        ]
        return self._collect_nodes_hw(node_ids, job_pids=None)


def _snap_to_points(
    snap: dict,
    *,
    ts: str,
    node_id: str | None,
    worker_id: str,
) -> list[dict]:
    gpu_uuids: dict[int, str] = snap.get("gpu_uuids") or {}
    points: list[dict] = []
    for key, value in (snap.get("metrics") or {}).items():
        scalar = scalarize_metric(value)
        if scalar is None:
            continue
        idx = _gpu_index(key)
        point: dict = {
            "key": key,
            "value": scalar,
            "node_id": node_id,
            "worker_id": worker_id,
            "gpu_idx": idx,
            "ts": ts,
        }
        if idx is not None and idx in gpu_uuids:
            point["gpu_uuid"] = gpu_uuids[idx]
        points.append(point)
    return points


def _gpu_index(key: str) -> int | None:
    if not key.startswith("gpu."):
        return None
    parts = key.split(".")
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return None
