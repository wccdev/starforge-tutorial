"""发现当前 Ray 作业占用的节点（driver + 本 job 的 alive actors）。"""
from __future__ import annotations

import os
from typing import Callable


def runtime_ray_job_id() -> str | None:
    try:
        import ray

        jid = ray.get_runtime_context().get_job_id()
        if jid:
            return str(jid)
    except Exception:
        pass
    for env in ("RAY_JOB_ID", "JOB_ID"):
        val = os.environ.get(env)
        if val:
            return val
    return None


def current_ray_node_id() -> str | None:
    try:
        import ray

        if not ray.is_initialized():
            return None
        return str(ray.get_runtime_context().get_node_id())
    except Exception:
        return None


def discover_job_node_ids(
    *,
    list_actors: Callable | None = None,
    job_id: str | None = None,
) -> set[str]:
    """返回本作业【实际运行 actor】的 Ray node_id 集合。

    只统计承载本 job alive actor 的节点（= 真正跑训练/生成的 GPU worker）。
    纯 driver/head 节点（不跑本 job 的 actor，其 GPU 与本次训练无关）不计入，
    否则监控面板会把那台无关机器的 GPU 也画成一条线（单机单卡作业却出现两条线的根因）。
    仅当查不到任何 actor 节点时（作业刚启动、或 State API 暂不可用）才回退到
    driver 所在节点兜底，保证面板不至于全空。
    """
    cur = current_ray_node_id()

    jid = job_id or runtime_ray_job_id()
    if not jid:
        return {cur} if cur else set()

    if list_actors is None:
        try:
            from ray.util.state import list_actors as _list_actors

            list_actors = _list_actors
        except Exception:
            return {cur} if cur else set()

    try:
        actors = list_actors(
            filters=[("job_id", "=", jid)],
            limit=500,
            detail=True,
            timeout=5,
        )
    except Exception:
        return {cur} if cur else set()

    nodes: set[str] = set()
    for actor in actors or []:
        state = getattr(actor, "state", None) or ""
        if str(state).upper() in ("DEAD", "RESTARTING"):
            continue
        node_id = getattr(actor, "node_id", None)
        if node_id:
            nodes.add(str(node_id))

    # 查到了 actor 节点就严格只采这些节点；一个都没查到才回退 driver 兜底。
    if not nodes and cur:
        nodes.add(cur)
    return nodes


def discover_job_pids(
    *,
    list_actors: Callable | None = None,
    job_id: str | None = None,
) -> set[int]:
    """返回本 Ray 作业 alive actor 的 PID 集合（用于 GPU 进程级归属）。

    探针在 driver / num_cpus=0 的远端 task 上运行，看不到 CUDA_VISIBLE_DEVICES 限制；
    必须靠 NVML 进程列表 + 本集合，才能把「整机 8 卡」收窄到「本作业实际占用的卡」。
    """
    jid = job_id or runtime_ray_job_id()
    if not jid:
        return set()

    if list_actors is None:
        try:
            from ray.util.state import list_actors as _list_actors

            list_actors = _list_actors
        except Exception:
            return set()

    try:
        actors = list_actors(
            filters=[("job_id", "=", jid)],
            limit=500,
            detail=True,
            timeout=5,
        )
    except Exception:
        return set()

    pids: set[int] = set()
    for actor in actors or []:
        state = getattr(actor, "state", None) or ""
        if str(state).upper() in ("DEAD", "RESTARTING"):
            continue
        pid = getattr(actor, "pid", None)
        if pid:
            try:
                pids.add(int(pid))
            except (TypeError, ValueError):
                continue
    return pids
