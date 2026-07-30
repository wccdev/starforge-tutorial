"""作业运行节点的静态硬件快照（「环境 → 系统硬件」的数据来源）。

回归背景：启动时 session 采的环境快照来自 driver 进程，异构集群里 driver 常驻 head，
于是作业 pin 在 GB10、页面却显示 head 那台 8 卡 H200。这里的 fan-out 负责把快照重采到
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
    calls: list[str] = []

    class _Remote:
        def __init__(self):
            self.node_id: str | None = None

        def options(self, *, scheduling_strategy):
            out = _Remote()
            out.node_id = scheduling_strategy.node_id
            return out

        def remote(self, *args, **kwargs):
            calls.append(self.node_id or "")
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


GB10 = {
    "hostname": "spark-gb10-1",
    "cpu": {"brand": "NVIDIA GB10", "cores": 20, "memory_gb": 119},
    "gpu": {"vendor": "nvidia", "count": 1, "devices": [{"index": 0, "name": "NVIDIA GB10"}]},
}
GB10_2 = {**GB10, "hostname": "spark-gb10-2"}


@pytest.fixture
def monitor():
    return HardwareMonitor(_Ingest(), scope="job")


def _gpu_nodes(monkeypatch, *node_ids: str):
    monkeypatch.setattr(hm, "discover_gpu_node_ids", lambda: set(node_ids))


def _actor_nodes(monkeypatch, *node_ids: str):
    monkeypatch.setattr(hm, "discover_job_node_ids", lambda **kw: set(node_ids))


def test_reports_hardware_of_gpu_nodes(monkeypatch, monitor):
    _install_fake_ray(monkeypatch, snapshots={"node-gb10": GB10})
    _gpu_nodes(monkeypatch, "node-gb10")

    monitor._sync_env_nodes()

    assert monitor.ingest.sent == [[{"node_id": "node-gb10", **GB10}]]


def test_gpu_placement_beats_actor_placement(monkeypatch, monitor):
    """head 上常驻辅助 actor 是正常的，但它不是训练节点。

    回归：作业 pin 在 GB10，页面却显示 head 那台 8 卡 H200——就是因为按 actor 落点判定。
    """
    _install_fake_ray(monkeypatch, snapshots={"node-gb10": GB10})
    _gpu_nodes(monkeypatch, "node-gb10")
    _actor_nodes(monkeypatch, "node-head")

    monitor._sync_env_nodes()

    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-gb10"]


def test_falls_back_to_actor_nodes_without_gpu_placement_group(monkeypatch, monitor):
    """纯 CPU 作业没有 GPU placement group，此时 actor 落点是唯一线索。"""
    _install_fake_ray(monkeypatch, snapshots={"node-a": GB10})
    _gpu_nodes(monkeypatch)
    _actor_nodes(monkeypatch, "node-a")

    monitor._sync_env_nodes()

    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-a"]


def test_actor_fallback_refuses_driver(monkeypatch, monitor):
    """回退到 actor 落点时也不许兜底到 driver，否则又会把 head 报上去。"""
    seen: list[dict] = []
    _install_fake_ray(monkeypatch, snapshots={})
    _gpu_nodes(monkeypatch)
    monkeypatch.setattr(
        hm, "discover_job_node_ids", lambda **kw: (seen.append(kw), set())[1]
    )

    monitor._sync_env_nodes()

    assert seen == [{"driver_fallback": False}]


def test_covers_every_gpu_node(monkeypatch, monitor):
    _install_fake_ray(monkeypatch, snapshots={"node-a": GB10, "node-b": GB10_2})
    _gpu_nodes(monkeypatch, "node-b", "node-a")

    monitor._sync_env_nodes()

    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-a", "node-b"]
    assert [n["hostname"] for n in monitor.ingest.sent[0]] == ["spark-gb10-1", "spark-gb10-2"]


def test_does_not_resend_unchanged_node_set(monkeypatch, monitor):
    _install_fake_ray(monkeypatch, snapshots={"node-gb10": GB10})
    _gpu_nodes(monkeypatch, "node-gb10")

    monitor._sync_env_nodes()
    monitor._sync_env_nodes()

    assert len(monitor.ingest.sent) == 1


def test_corrects_itself_when_node_set_grows(monkeypatch, monitor):
    """监控线程比 GPU worker 早起来，第一拍拿到的视图往往是残缺的。

    回归：上一版采到一次就永久锁定，把启动瞬间看到的 head 当成了最终答案。
    """
    _install_fake_ray(monkeypatch, snapshots={"node-head": GB10, "node-gb10": GB10_2})
    _gpu_nodes(monkeypatch)
    _actor_nodes(monkeypatch, "node-head")

    monitor._sync_env_nodes()
    assert [n["node_id"] for n in monitor.ingest.sent[0]] == ["node-head"]

    _gpu_nodes(monkeypatch, "node-gb10")  # GPU placement group 建好了
    monitor._sync_env_nodes()

    assert [n["node_id"] for n in monitor.ingest.sent[1]] == ["node-gb10"]


def test_retries_next_tick_when_upload_fails(monkeypatch):
    monitor = HardwareMonitor(_Ingest(ok=False), scope="job")
    _install_fake_ray(monkeypatch, snapshots={"node-gb10": GB10})
    _gpu_nodes(monkeypatch, "node-gb10")

    monitor._sync_env_nodes()
    monitor._sync_env_nodes()

    assert len(monitor.ingest.sent) == 2


def test_waits_for_nodes_before_reporting(monkeypatch, monitor):
    """节点还没起来时这拍跳过，下一拍再试——不能拿 driver 的硬件顶上。"""
    ready = {"yes": False}
    _install_fake_ray(monkeypatch, snapshots={"node-gb10": GB10})
    _actor_nodes(monkeypatch)
    monkeypatch.setattr(
        hm, "discover_gpu_node_ids", lambda: {"node-gb10"} if ready["yes"] else set()
    )

    monitor._sync_env_nodes()
    assert monitor.ingest.sent == []

    ready["yes"] = True
    monitor._sync_env_nodes()
    assert len(monitor.ingest.sent) == 1


def test_skips_before_ray_init(monkeypatch, monitor):
    _install_fake_ray(monkeypatch, snapshots={}, initialized=False)
    _gpu_nodes(monkeypatch, "node-gb10")

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
