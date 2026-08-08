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

from common.observability.env_probe import collect_node_hardware
from common.observability.hw_probe import DEFAULT_MIN_MEM_MIB, collect_hw_snapshot
from common.observability.job_nodes import (
    current_ray_node_id,
    discover_gpu_bundle_counts,
    discover_gpu_node_ids,
    discover_job_node_ids,
    discover_job_pids,
)
from common.observability.sampling import swanlab_monitor_interval
from common.observability.util import scalarize_metric

MonitorScope = Literal["local", "job", "cluster"]
NODE_DISCOVERY_TTL = 60.0
ENV_PROBE_TIMEOUT = 30.0
ENV_PROBE_MAX_ATTEMPTS = 5
# 描述「某个进程」而非「某台机器」的指标；只有在 driver 上采才说得通。
PROCESS_METRIC_KEYS = frozenset({"cpu.thds", "mem.proc", "mem.proc.pct"})


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
        self._capacity_cache: tuple[float, dict[str, int]] | None = None
        # node_id → 这台机器上「PID 归属确认过」的卡 UUID。PID 查空那几拍靠它兜底，
        # 免得退到显存启发式、把同机邻居作业的卡也画进本作业的面板。
        self._confirmed_gpus: dict[str, frozenset[str]] = {}
        self._env_nodes_reported: list[str] = []
        # 上次上报的「每节点 GPU 张数」。异构下先报错（整机库存）再按 PG 纠正时，
        # 节点集合可能不变，要靠这个签名触发重报。
        self._env_nodes_gpu_counts: dict[str, int] = {}
        self._env_nodes_failures = 0
        self._env_nodes_disabled = False

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
            try:
                self._sync_env_nodes()
            except Exception as e:
                self._env_nodes_failures += 1
                if self._env_nodes_failures >= ENV_PROBE_MAX_ATTEMPTS:
                    self._env_nodes_disabled = True  # 放弃，别把训练日志刷满
                    print(f"NeMoLab environment nodes probe gave up after {self._env_nodes_failures} tries: {e}")
                else:
                    print(f"NeMoLab environment nodes probe error: {e}")
            time.sleep(self._sleep_interval())

    def _sync_env_nodes(self) -> None:
        """把作业运行节点的静态硬件采回来上报；节点集合或卡数不对就重报。

        启动时 session 采的那份环境快照来自 driver 进程，异构集群里 driver 常驻 head，
        于是「系统硬件」永远显示 head 那台机器——作业 pin 到 GB10 却显示 8 卡 H200 就是
        这么来的。这里把快照重新采到真正跑训练的机器上（actor / GPU PG 节点），
        绝不把 driver 所在机当成运行节点。

        为什么要持续同步而不是采一次就锁定：监控线程比 GPU worker 早起来（实测早 6 秒），
        第一拍看到的往往只是 driver 侧的辅助 actor。锁定第一次的结果，就等于把启动过程中
        某个瞬间的残缺视图当成最终答案钉死——上一版正是这么把 head 写死的。
        """
        if self._env_nodes_disabled or self.scope != "job":
            return
        if not hasattr(self.ingest, "send_environment_nodes"):
            self._env_nodes_disabled = True  # 老版本 IngestClient，别每拍都白试一遍
            return

        import ray

        if not ray.is_initialized():
            return
        # 环境页专用节点集合：比时序监控更严，GPU 作业在 PG 未就绪前宁可空着。
        node_ids = sorted(self._env_training_node_ids())
        if not node_ids:
            return
        if (
            node_ids == self._env_nodes_reported
            and self._env_gpu_counts_match_capacity(node_ids)
        ):
            return

        nodes = self._collect_nodes_env(node_ids)
        if nodes and self.ingest.send_environment_nodes(nodes):
            self._env_nodes_reported = node_ids
            self._env_nodes_gpu_counts = {
                str(n.get("node_id") or ""): int(((n.get("gpu") or {}).get("count") or 0))
                for n in nodes
            }

    def _training_node_ids(self) -> set[str]:
        """本作业真正跑训练的节点（时序监控用）。"""
        gpu_nodes = discover_gpu_node_ids()
        if gpu_nodes:
            return gpu_nodes
        # 没有 GPU placement group（纯 CPU 作业，或 PG 还没建起来）时退回 actor 落点。
        # 不允许回退到 driver：宁可这拍不报，也别把 head 的硬件当成训练节点。
        return discover_job_node_ids(driver_fallback=False)

    def _env_training_node_ids(self) -> set[str]:
        """环境页「运行节点」：只认 GPU PG；异构下绝不能用 head 上的辅助 actor 冒充。

        时序监控可以短暂回退 actor 落点（曲线晚几秒出现无所谓），但环境快照会长期挂在
        页面上。GPU 作业在 PG 未就绪时返回空集，等下一拍——driver 快照仍作标注过的兜底。
        """
        gpu_nodes = discover_gpu_node_ids()
        if gpu_nodes:
            return gpu_nodes
        if self._expected_gpus_per_node() is not None:
            return set()
        return discover_job_node_ids(driver_fallback=False)

    @staticmethod
    def _expected_gpus_per_node() -> int | None:
        """提交侧注入的每节点卡数（LAB_CLUSTER_GPUS_PER_NODE）；GPU 作业的权威拓扑。"""
        raw = os.environ.get("LAB_CLUSTER_GPUS_PER_NODE", "").strip()
        if not raw:
            return None
        try:
            n = int(raw)
        except ValueError:
            return None
        return n if n > 0 else None

    def _node_max_gpus(self, node_id: str) -> int | None:
        """本作业在该节点分到的卡数：PG bundle 优先，否则用提交侧每节点卡数。"""
        cap = self._gpu_capacity().get(node_id)
        if cap is not None:
            return cap
        return self._expected_gpus_per_node()

    def _env_gpu_counts_match_capacity(self, node_ids: list[str]) -> bool:
        """已上报的每节点 GPU 张数是否与本作业配额一致；多了说明曾把整机库存写进去。"""
        if not self._env_nodes_gpu_counts:
            return False
        for node_id in node_ids:
            want = self._node_max_gpus(node_id)
            if want is None:
                continue
            got = self._env_nodes_gpu_counts.get(node_id)
            if got is None or got > want:
                return False
        return True

    def _collect_nodes_env(self, node_ids: list[str]) -> list[dict]:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        remote_collect = ray.remote(num_cpus=0)(collect_node_hardware)
        # 派到 actor/GPU 节点就地采；过滤参数与时序监控一致。
        # job_pids 是全集群 actor PID：只在本机 NVML 进程表里命中的才会认到卡，
        # 不会把 head 上的 PID 误套到 worker 的 GPU 上。
        pid_arg = list(self._job_pids())
        futures = []
        for node_id in node_ids:
            known = self._confirmed_gpus.get(node_id)
            futures.append(
                remote_collect.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(
                    job_pids=pid_arg,
                    min_mem_mib=self.min_mem_mib,
                    gpu_fallback=True,
                    known_gpu_uuids=sorted(known) if known else None,
                    max_gpus=self._node_max_gpus(node_id),
                )
            )
        # 带超时：这是挂在监控线程里的一次性任务，某个节点卡住不能连带把硬件时序也停掉。
        snapshots = ray.get(futures, timeout=ENV_PROBE_TIMEOUT)
        return [
            {"node_id": node_id, **snap}
            for node_id, snap in zip(node_ids, snapshots, strict=False)
        ]

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
        ids = self._training_node_ids()
        self._node_cache = (now, ids)
        return ids

    def _job_pids(self) -> frozenset[int]:
        now = time.time()
        if self._pid_cache and now - self._pid_cache[0] < NODE_DISCOVERY_TTL:
            return self._pid_cache[1]
        pids = frozenset(discover_job_pids())
        self._pid_cache = (now, pids)
        return pids

    def _gpu_capacity(self) -> dict[str, int]:
        """本作业在各节点分到的卡数（PG bundle 口径），给显存启发式封顶用。"""
        now = time.time()
        if self._capacity_cache and now - self._capacity_cache[0] < NODE_DISCOVERY_TTL:
            return self._capacity_cache[1]
        try:
            counts = discover_gpu_bundle_counts()
        except Exception:
            counts = {}
        self._capacity_cache = (now, counts)
        return counts

    def _remember_gpu_uuids(self, node_id: str | None, snap: dict) -> None:
        """只记 PID 认出来的那批卡；sticky/显存启发式的结果不能当证据自我强化。"""
        if snap.get("gpu_attribution") != "pid":
            return
        uuids = frozenset(str(u) for u in (snap.get("gpu_uuids") or {}).values() if u)
        if uuids:
            self._confirmed_gpus[node_id or ""] = uuids

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
                self._collect_local_hw(
                    node_id=current, job_pids=job_pids, gpu_fallback=True
                )
            )
        else:
            # driver 不在训练节点上（异构集群常态：driver 在 head，训练在 GB10）。
            # 这台机器的 CPU/内存跟训练无关，不该画进面板——单机单卡作业却出现两条线
            # 就是这么来的。但 driver 自己的进程指标是真实且唯一的：远端探针量到的
            # 「进程内存」其实是探针任务自己，只能在这里采。
            points.extend(self._collect_driver_process_hw())
        remote = sorted(nid for nid in node_ids if nid != current)
        if remote:
            # 这批节点是发现逻辑认定的训练节点，本身就是「这台机器在跑本作业」的证据，
            # 所以允许它们在 PID 归属查空时退回显存启发式认卡。
            points.extend(
                self._collect_nodes_hw(remote, job_pids=job_pids, gpu_fallback=True)
            )
        return points

    def _collect_driver_process_hw(self) -> list[dict]:
        """仅 driver 进程自身的指标，不含所在机器的整机指标。"""
        snap = collect_hw_snapshot(job_pids=frozenset(), min_mem_mib=self.min_mem_mib)
        metrics = {
            key: value
            for key, value in (snap.get("metrics") or {}).items()
            if key in PROCESS_METRIC_KEYS
        }
        if not metrics:
            return []
        return _snap_to_points(
            {"metrics": metrics},
            ts=datetime.now(timezone.utc).isoformat(),
            node_id=current_ray_node_id(),
            worker_id=snap.get("hostname") or socket.gethostname(),
        )

    def _collect_local_hw(
        self,
        *,
        node_id: str | None = None,
        job_pids: frozenset[int] | None = None,
        gpu_fallback: bool = False,
    ) -> list[dict]:
        snap = collect_hw_snapshot(
            job_pids=job_pids,
            min_mem_mib=self.min_mem_mib,
            gpu_fallback=gpu_fallback,
            known_gpu_uuids=self._confirmed_gpus.get(node_id or ""),
            max_gpus=self._gpu_capacity().get(node_id or "") if node_id else None,
        )
        self._remember_gpu_uuids(node_id, snap)
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
        gpu_fallback: bool = False,
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
        capacity = self._gpu_capacity()
        for node_id in node_ids:
            known = self._confirmed_gpus.get(node_id)
            futures.append(
                remote_collect.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote(
                    job_pids=pid_arg,
                    min_mem_mib=self.min_mem_mib,
                    include_process=False,
                    gpu_fallback=gpu_fallback,
                    known_gpu_uuids=sorted(known) if known else None,
                    max_gpus=capacity.get(node_id),
                )
            )
        snapshots = ray.get(futures)
        for node_id, snap in zip(node_ids, snapshots, strict=False):
            self._remember_gpu_uuids(node_id, snap)
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
