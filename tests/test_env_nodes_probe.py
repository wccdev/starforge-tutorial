"""作业运行节点的静态硬件快照（「环境 → 系统硬件」的数据来源）。

回归背景：启动时 session 采的环境快照来自 driver 进程，异构集群里 driver 常驻 head，
于是作业跑在训练节点、页面却显示 head 那台 8 卡 H200。这里的 fan-out 负责把快照重采到
真正跑 actor 的机器上。
"""
import sys
from types import SimpleNamespace

import pytest

from common.observability import hardware_monitor as hm
from common.observability.hardware_monitor import HardwareMonitor


class _Ingest:
    def __init__(self, *, ok: bool = True):
        self.ok = ok
        self.sent: list[list[dict]] = []

    def enqueue_hardware(self, points):
        pass

    def send_environment_nodes(self, nodes):
        self.sent.append(nodes)
        return self.ok


class _LegacyIngest:
    """老版本 IngestClient：没有 send_environment_nodes。"""

    def enqueue_hardware(self, points):
        pass


def _install_fake_ray(monkeypatch, *, snapshots: dict[str, dict], initialized: bool = True):
    """最小可用的 ray 替身：remote 任务在本进程直接按节点取预设快照。"""
    calls: list[dict] = []

    class _Remote:
        def __init__(self):
            self.node_id: str | None = None

        def options(self, *, scheduling_strategy):
            out = _Remote()
            out.node_id = scheduling_strategy.node_id
            return out

        def remote(self, *args, **kwargs):
            calls.append({"node_id": self.node_id or "", "kwargs": kwargs})
            return self.node_id

    fake_ray = SimpleNamespace(
        is_initialized=lambda: initialized,
        remote=lambda **kwargs: (lambda fn: _Remote()),
        get=lambda refs, timeout=None: [snapshots[r] for r in refs],
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(
        sys.modules,
        "ray.util",
        SimpleNamespace(),
    )
    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        SimpleNamespace(
            NodeAffinitySchedulingStrategy=lambda *, node_id, soft: SimpleNamespace(
                node_id=node_id, soft=soft
            )
        ),
    )
    return calls


TRAIN_NODE = {
    "hostname": "train-node-1",
    "cpu": {"brand": "NVIDIA H100", "cores": 20, "memory_gb": 119},
    "gpu": {"vendor": "nvidia", "count": 1, "devices": [{"index": 0, "name": "NVIDIA H100"}]},
}
TRAIN_NODE_2 = {**TRAIN_NODE, "hostname": "train-node-2"}


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.delenv("LAB_CLUSTER_GPUS_PER_NODE", raising=False)
    monkeypatch.delenv("LAB_CLUSTER_NUM_NODES", raising=False)
    return HardwareMonitor(_Ingest(), scope="job")


def _gpu_nodes(monkeypatch, *node_ids: str):
    monkeypatch.setattr(hm, "discover_gpu_node_ids", lambda: set(node_ids))


def _actor_nodes(monkeypatch, *node_ids: str):
    monkeypatch.setattr(hm, "discover_job_node_ids", lambda **kw: set(node_ids))


def test_reports_hardware_of_gpu_nodes(monkeypatch, monitor):
    _install_fake_ray(monkeypatch, snapshots={"node-train": TRAIN_NODE})
    _gpu_nodes(monkeypatch, "node-train")

    monitor._sync_env_nodes()

    assert monitor.ingest.sent == [[{"node_id": "node-train", **TRAIN_NODE}]]


def test_gpu_placement_beats_actor_placement(monkeypatch, monitor):
    """head 上常驻辅助 actor 是正常的，但它不是训练节点。

    回归：作业跑在训练节点，页面却显示 head 那台 8 卡 H200——就是因为按 actor 落点判定。
    """
    _install_fake_ray(monkeypatch, snapshots={"node-train": TRAIN_NODE})
    _gpu_nodes(monkeypatch, "node-train")
    _actor_nodes(monkeypatch, "node-head")

    monitor._sync_env_nodes()

    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-train"]


def test_falls_back_to_actor_nodes_without_gpu_placement_group(monkeypatch, monitor):
    """纯 CPU 作业没有 GPU placement group，此时 actor 落点是唯一线索。"""
    monkeypatch.delenv("LAB_CLUSTER_GPUS_PER_NODE", raising=False)
    _install_fake_ray(monkeypatch, snapshots={"node-a": TRAIN_NODE})
    _gpu_nodes(monkeypatch)
    _actor_nodes(monkeypatch, "node-a")

    monitor._sync_env_nodes()

    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-a"]


def test_gpu_job_waits_for_placement_group_not_head_actors(monkeypatch, monitor):
    """异构 GPU 作业：PG 未就绪前不要把 head 辅助 actor 写成运行节点。"""
    monkeypatch.setenv("LAB_CLUSTER_GPUS_PER_NODE", "2")
    _install_fake_ray(monkeypatch, snapshots={"node-head": TRAIN_NODE, "node-train": TRAIN_NODE_2})
    _gpu_nodes(monkeypatch)
    _actor_nodes(monkeypatch, "node-head")

    monitor._sync_env_nodes()
    assert monitor.ingest.sent == []

    _gpu_nodes(monkeypatch, "node-train")
    monitor._sync_env_nodes()
    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-train"]


def test_actor_fallback_refuses_driver(monkeypatch, monitor):
    """回退到 actor 落点时也不许兜底到 driver，否则又会把 head 报上去。"""
    monkeypatch.delenv("LAB_CLUSTER_GPUS_PER_NODE", raising=False)
    seen: list[dict] = []
    _install_fake_ray(monkeypatch, snapshots={})
    _gpu_nodes(monkeypatch)
    monkeypatch.setattr(
        hm, "discover_job_node_ids", lambda **kw: (seen.append(kw), set())[1]
    )

    monitor._sync_env_nodes()

    assert seen == [{"driver_fallback": False}]


def test_covers_every_gpu_node(monkeypatch, monitor):
    _install_fake_ray(monkeypatch, snapshots={"node-a": TRAIN_NODE, "node-b": TRAIN_NODE_2})
    _gpu_nodes(monkeypatch, "node-b", "node-a")

    monitor._sync_env_nodes()

    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-a", "node-b"]
    assert [n["hostname"] for n in monitor.ingest.sent[0]] == ["train-node-1", "train-node-2"]


def test_does_not_resend_unchanged_node_set(monkeypatch, monitor):
    _install_fake_ray(monkeypatch, snapshots={"node-train": TRAIN_NODE})
    _gpu_nodes(monkeypatch, "node-train")

    monitor._sync_env_nodes()
    monitor._sync_env_nodes()

    assert len(monitor.ingest.sent) == 1


def test_corrects_itself_when_node_set_grows(monkeypatch, monitor):
    """CPU 作业：监控比 worker 早起来时，节点集合长大后要纠正，不能锁死第一拍。"""
    monkeypatch.delenv("LAB_CLUSTER_GPUS_PER_NODE", raising=False)
    _install_fake_ray(monkeypatch, snapshots={"node-head": TRAIN_NODE, "node-train": TRAIN_NODE_2})
    _gpu_nodes(monkeypatch)
    _actor_nodes(monkeypatch, "node-head")

    monitor._sync_env_nodes()
    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-head"]

    _gpu_nodes(monkeypatch, "node-train")  # GPU placement group 建好了
    monitor._sync_env_nodes()

    assert [n["node_id"] for n in monitor.ingest.sent[1]] == ["node-train"]


def test_resends_when_reported_gpu_count_exceeds_quota(monkeypatch, monitor):
    """同节点集合下若先前误报了整机卡数，配额就绪后要重报纠正。"""
    monkeypatch.setenv("LAB_CLUSTER_GPUS_PER_NODE", "2")
    fat = {
        "hostname": "h200-host",
        "cpu": {"brand": "Xeon", "cores": 192, "memory_gb": 2015},
        "gpu": {
            "vendor": "nvidia",
            "count": 8,
            "devices": [{"index": i, "name": "NVIDIA H200"} for i in range(8)],
        },
    }
    slim = {
        **fat,
        "gpu": {
            "vendor": "nvidia",
            "count": 2,
            "devices": [{"index": i, "name": "NVIDIA H200"} for i in range(2)],
        },
    }
    snaps = {"node-h200": fat}
    _install_fake_ray(monkeypatch, snapshots=snaps)
    _gpu_nodes(monkeypatch, "node-h200")
    monkeypatch.setattr(hm, "discover_gpu_bundle_counts", lambda: {})

    monitor._sync_env_nodes()
    assert monitor.ingest.sent[0][0]["gpu"]["count"] == 8

    snaps["node-h200"] = slim
    monkeypatch.setattr(hm, "discover_gpu_bundle_counts", lambda: {"node-h200": 2})
    monitor._capacity_cache = None

    monitor._sync_env_nodes()
    assert len(monitor.ingest.sent) == 2
    assert monitor.ingest.sent[1][0]["gpu"]["count"] == 2


def test_retries_next_tick_when_upload_fails(monkeypatch):
    monitor = HardwareMonitor(_Ingest(ok=False), scope="job")
    _install_fake_ray(monkeypatch, snapshots={"node-train": TRAIN_NODE})
    _gpu_nodes(monkeypatch, "node-train")

    monitor._sync_env_nodes()
    monitor._sync_env_nodes()

    assert len(monitor.ingest.sent) == 2


def test_waits_for_nodes_before_reporting(monkeypatch, monitor):
    """节点还没起来时这拍跳过，下一拍再试——不能拿 driver 的硬件顶上。"""
    ready = {"yes": False}
    _install_fake_ray(monkeypatch, snapshots={"node-train": TRAIN_NODE})
    _actor_nodes(monkeypatch)
    monkeypatch.setattr(
        hm, "discover_gpu_node_ids", lambda: {"node-train"} if ready["yes"] else set()
    )

    monitor._sync_env_nodes()
    assert monitor.ingest.sent == []

    ready["yes"] = True
    monitor._sync_env_nodes()
    assert len(monitor.ingest.sent) == 1


def test_skips_before_ray_init(monkeypatch, monitor):
    _install_fake_ray(monkeypatch, snapshots={}, initialized=False)
    _gpu_nodes(monkeypatch, "node-train")

    monitor._sync_env_nodes()

    assert monitor.ingest.sent == []
    assert monitor._env_nodes_reported == []


def test_skips_non_job_scope(monkeypatch):
    """scope=local/cluster 时节点集合与本作业无关，报上去只会误导。"""
    monitor = HardwareMonitor(_Ingest(), scope="local")
    monkeypatch.setattr(hm, "discover_gpu_node_ids", lambda: pytest.fail("不该发现节点"))

    monitor._sync_env_nodes()

    assert monitor.ingest.sent == []


def test_tolerates_legacy_ingest_without_endpoint(monkeypatch):
    monitor = HardwareMonitor(_LegacyIngest(), scope="job")
    monkeypatch.setattr(hm, "discover_gpu_node_ids", lambda: pytest.fail("不该发现节点"))

    monitor._sync_env_nodes()

    assert monitor._env_nodes_disabled is True  # 别每拍都白试一遍


def test_passes_job_gpu_scope_into_node_hardware_probe(monkeypatch, monitor):
    """环境节点探针必须带上本作业的 PID / PG 卡数，否则会把整机 GPU 库存报上去。"""
    calls = _install_fake_ray(monkeypatch, snapshots={"node-h200": TRAIN_NODE})
    _gpu_nodes(monkeypatch, "node-h200")
    monkeypatch.setattr(hm, "discover_job_pids", lambda: {101, 102})
    monkeypatch.setattr(hm, "discover_gpu_bundle_counts", lambda: {"node-h200": 2})
    monitor._confirmed_gpus["node-h200"] = frozenset({"GPU-a", "GPU-b"})

    monitor._sync_env_nodes()

    assert calls[0]["node_id"] == "node-h200"
    kw = calls[0]["kwargs"]
    assert sorted(kw["job_pids"]) == [101, 102]
    assert kw["max_gpus"] == 2
    assert kw["gpu_fallback"] is True
    assert kw["known_gpu_uuids"] == ["GPU-a", "GPU-b"]


def test_max_gpus_falls_back_to_cluster_env_topology(monkeypatch, monitor):
    """PG bundle 计数暂缺时，用 LAB_CLUSTER_GPUS_PER_NODE 封顶（异构提交侧权威拓扑）。"""
    monkeypatch.setenv("LAB_CLUSTER_GPUS_PER_NODE", "2")
    calls = _install_fake_ray(monkeypatch, snapshots={"node-h200": TRAIN_NODE})
    _gpu_nodes(monkeypatch, "node-h200")
    monkeypatch.setattr(hm, "discover_gpu_bundle_counts", lambda: {})

    monitor._sync_env_nodes()

    assert calls[0]["kwargs"]["max_gpus"] == 2
