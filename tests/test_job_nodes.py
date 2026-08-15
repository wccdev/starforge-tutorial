"""job_nodes：发现本 Ray 作业占用的节点。"""
from types import SimpleNamespace

from common.observability import job_nodes
from common.observability.job_nodes import discover_job_node_ids, runtime_ray_job_id


class _Actor:
    def __init__(self, node_id: str, state: str = "ALIVE"):
        self.node_id = node_id
        self.state = state


def test_discover_job_node_ids_from_actors():
    def _list(**kwargs):
        assert kwargs["filters"] == [("job_id", "=", "job-abc")]
        return [_Actor("node-a"), _Actor("node-b"), _Actor("node-a")]

    nodes = discover_job_node_ids(
        list_actors=_list,
        job_id="job-abc",
    )
    assert nodes == {"node-a", "node-b"}


def test_discover_job_node_ids_skips_dead():
    def _list(**kwargs):
        return [_Actor("node-a", "ALIVE"), _Actor("node-b", "DEAD")]

    nodes = discover_job_node_ids(list_actors=_list, job_id="j1")
    assert nodes == {"node-a"}


def test_discover_excludes_pure_driver_node(monkeypatch):
    """driver 节点不跑本 job 的 actor 时，不应被计入（单机单卡两条线根因）。"""
    monkeypatch.setattr(job_nodes, "current_ray_node_id", lambda: "driver-node")

    def _list(**kwargs):
        return [_Actor("worker-node")]

    nodes = discover_job_node_ids(list_actors=_list, job_id="j1")
    assert nodes == {"worker-node"}


def test_discover_falls_back_to_driver_when_no_actors(monkeypatch):
    """查不到任何 actor 时回退 driver 节点兜底，避免面板全空。"""
    monkeypatch.setattr(job_nodes, "current_ray_node_id", lambda: "driver-node")

    def _list(**kwargs):
        return []

    nodes = discover_job_node_ids(list_actors=_list, job_id="j1")
    assert nodes == {"driver-node"}


def test_discover_without_driver_fallback_returns_empty(monkeypatch):
    """静态硬件快照用严格模式：查不到 actor 就返回空集，让调用方下次再试。

    那份数据会长期留在作业详情页上，把 driver（异构集群里通常是 head）的硬件写进去，
    一次就错到底——宁可晚几秒。
    """
    monkeypatch.setattr(job_nodes, "current_ray_node_id", lambda: "driver-node")

    def _list(**kwargs):
        return []

    assert discover_job_node_ids(list_actors=_list, job_id="j1", driver_fallback=False) == set()


def test_discover_without_driver_fallback_still_returns_actor_nodes(monkeypatch):
    monkeypatch.setattr(job_nodes, "current_ray_node_id", lambda: "driver-node")

    def _list(**kwargs):
        return [_Actor("worker-node")]

    nodes = discover_job_node_ids(list_actors=_list, job_id="j1", driver_fallback=False)
    assert nodes == {"worker-node"}


def test_discover_includes_driver_when_it_runs_actor(monkeypatch):
    """driver 节点同时承载本 job 的 actor（单机作业）时仍应计入。"""
    monkeypatch.setattr(job_nodes, "current_ray_node_id", lambda: "node-a")

    def _list(**kwargs):
        return [_Actor("node-a"), _Actor("node-b")]

    nodes = discover_job_node_ids(list_actors=_list, job_id="j1")
    assert nodes == {"node-a", "node-b"}


def _pg(job_id: str, state: str, bundles: list[dict], name: str = ""):
    """字段对齐 `ray list placement-groups --detail` 的真实输出。"""
    return SimpleNamespace(
        placement_group_id="pg-1", name=name, creator_job_id=job_id, state=state, bundles=bundles
    )


def _bundle(node_id: str, **resources):
    return {"unit_resources": resources, "node_id": node_id}


def test_gpu_nodes_from_placement_groups():
    def _list(**kwargs):
        return [
            _pg("j1", "CREATED", [_bundle("node-train", GPU=1.0, CPU=2.0)], "grpo_policy_cluster-node0")
        ]

    nodes = job_nodes.discover_gpu_node_ids(list_placement_groups=_list, job_id="j1")
    assert nodes == {"node-train"}


def test_gpu_nodes_ignore_cpu_only_bundles():
    """NeMo-RL 建 venv 时按节点铺开的 STRICT_SPREAD 组是纯 CPU 的，不算训练节点。

    回归：把这组算进去，head 就会被当成训练机，页面显示成 8 卡 H200。
    """

    def _list(**kwargs):
        return [
            _pg("j1", "CREATED", [_bundle("node-train", GPU=1.0, CPU=2.0)]),
            _pg(
                "j1",
                "CREATED",
                [_bundle("node-head", CPU=1.0), _bundle("node-train", CPU=1.0), _bundle("node-b", CPU=1.0)],
            ),
        ]

    nodes = job_nodes.discover_gpu_node_ids(list_placement_groups=_list, job_id="j1")
    assert nodes == {"node-train"}


def test_gpu_nodes_ignore_other_jobs_and_removed_groups():
    def _list(**kwargs):
        return [
            _pg("other", "CREATED", [_bundle("node-head", GPU=8.0)]),
            _pg("j1", "REMOVED", [_bundle("node-stale", GPU=1.0)]),
            _pg("j1", "CREATED", [_bundle("node-train", GPU=1.0)]),
        ]

    nodes = job_nodes.discover_gpu_node_ids(list_placement_groups=_list, job_id="j1")
    assert nodes == {"node-train"}


def test_gpu_nodes_accept_dict_shaped_state_api():
    """state API 时而给 dataclass 时而给 dict，两种都得能取。"""

    def _list(**kwargs):
        return [
            {
                "creator_job_id": "j1",
                "state": "CREATED",
                "bundles": [{"unit_resources": {"GPU": 1.0}, "node_id": "node-train"}],
            }
        ]

    nodes = job_nodes.discover_gpu_node_ids(list_placement_groups=_list, job_id="j1")
    assert nodes == {"node-train"}


def test_gpu_nodes_empty_when_pending_placement_has_no_node_yet():
    def _list(**kwargs):
        return [_pg("j1", "PENDING", [{"unit_resources": {"GPU": 1.0}, "node_id": None}])]

    assert job_nodes.discover_gpu_node_ids(list_placement_groups=_list, job_id="j1") == set()


def test_gpu_nodes_empty_when_state_api_unavailable():
    def _list(**kwargs):
        raise RuntimeError("state API down")

    assert job_nodes.discover_gpu_node_ids(list_placement_groups=_list, job_id="j1") == set()


def test_runtime_ray_job_id_env_fallback(monkeypatch):
    monkeypatch.setenv("RAY_JOB_ID", "env-job")
    fake_ray = SimpleNamespace(
        get_runtime_context=lambda: SimpleNamespace(get_job_id=lambda: None)
    )
    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)
    assert runtime_ray_job_id() == "env-job"


def test_discover_job_pids_from_actors():
    class _Actor:
        def __init__(self, pid: int, state: str = "ALIVE"):
            self.pid = pid
            self.state = state

    def _list(**kwargs):
        assert kwargs["filters"] == [("job_id", "=", "job-abc")]
        return [_Actor(1001), _Actor(1002), _Actor(999, "DEAD")]

    pids = job_nodes.discover_job_pids(list_actors=_list, job_id="job-abc")
    assert pids == {1001, 1002}
