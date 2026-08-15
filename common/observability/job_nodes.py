"""发现当前 Ray 作业占用的节点（driver + 本 job 的 alive actors）。"""
from __future__ import annotations

import math
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
    driver_fallback: bool = True,
) -> set[str]:
    """返回本作业【实际运行 actor】的 Ray node_id 集合。

    只统计承载本 job alive actor 的节点（= 真正跑训练/生成的 GPU worker）。
    纯 driver/head 节点（不跑本 job 的 actor，其 GPU 与本次训练无关）不计入，
    否则监控面板会把那台无关机器的 GPU 也画成一条线（单机单卡作业却出现两条线的根因）。

    Args:
        driver_fallback: 查不到任何 actor 节点时（作业刚启动、或 State API 暂不可用）
            是否回退到 driver 所在节点。时序监控要开着，宁可先画 driver 也别让面板全空；
            静态硬件快照要关掉，因为那份数据会长期留在页面上，写错一次就一直错下去，
            此时返回空集、留给调用方下次再试才是对的。
    """
    cur = current_ray_node_id() if driver_fallback else None

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


def discover_gpu_node_ids(
    *,
    list_placement_groups: Callable | None = None,
    job_id: str | None = None,
) -> set[str]:
    """返回本作业的 GPU placement group 落在哪些节点，即真正跑训练的机器。

    比 actor 列表更贴近「训练在哪台机器上」这个问题，两个层面都更靠得住：

    语义上，GPU bundle 的落点就是训练节点。而 actor 是不分闲忙的——NeMo-RL 建 venv 时
    会用 STRICT_SPREAD 在每个节点铺一组 CPU-only actor，driver 侧也常驻辅助 actor，
    这些落在 head 完全正常，却会让 head 被当成训练节点（作业跑在 H100、页面显示
    8 卡 H200 的直接原因）。

    可靠性上，PG 的 bundle 落点由 GCS 记账，不依赖各节点的 dashboard agent；agent 挂掉
    的节点上，actor 明细是查不全的。
    """
    return set(
        discover_gpu_bundle_counts(
            list_placement_groups=list_placement_groups, job_id=job_id
        )
    )


def discover_gpu_bundle_counts(
    *,
    list_placement_groups: Callable | None = None,
    job_id: str | None = None,
) -> dict[str, int]:
    """本作业在每个节点上拿到了几张卡（GPU bundle 的 GPU 数之和）。

    这是「本作业在这台机器上最多能用几张卡」的权威答案——由 GCS 记账，不依赖 NVML
    的显存猜测。探针在 PID 归属查空、只能按显存挑忙卡时，用它给挑出来的张数封顶，
    邻居作业的卡就不会被算进来（同机跑两个实验时曾把对方的卡画成第二条线）。
    """
    jid = job_id or runtime_ray_job_id()
    if not jid:
        return {}

    if list_placement_groups is None:
        try:
            from ray.util.state import list_placement_groups as _list_pgs

            list_placement_groups = _list_pgs
        except Exception:
            return {}

    try:
        groups = list_placement_groups(limit=500, detail=True, timeout=5)
    except Exception:
        return {}

    counts: dict[str, int] = {}
    for pg in groups or []:
        if str(_field(pg, "creator_job_id") or "") != jid:
            continue
        if str(_field(pg, "state") or "").upper() == "REMOVED":
            continue
        for bundle in _field(pg, "bundles") or []:
            resources = _field(bundle, "unit_resources") or {}
            try:
                gpus = float(resources.get("GPU") or 0)
            except (TypeError, ValueError):
                gpus = 0.0
            node_id = _field(bundle, "node_id")
            if gpus > 0 and node_id:
                # 分数卡（GPU=0.5）向上取整：占了半张也是占了这张卡。
                counts[str(node_id)] = counts.get(str(node_id), 0) + math.ceil(gpus)
    return counts


def _field(obj: object, name: str):
    """state API 时而给 dataclass 时而给 dict，两种都得能取。"""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


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
