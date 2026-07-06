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


def test_discover_includes_driver_when_it_runs_actor(monkeypatch):
    """driver 节点同时承载本 job 的 actor（单机作业）时仍应计入。"""
    monkeypatch.setattr(job_nodes, "current_ray_node_id", lambda: "node-a")

    def _list(**kwargs):
        return [_Actor("node-a"), _Actor("node-b")]

    nodes = discover_job_node_ids(list_actors=_list, job_id="j1")
    assert nodes == {"node-a", "node-b"}


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
