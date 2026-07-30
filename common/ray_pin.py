"""卡型 pin：把作业约束到目标卡型的节点上。

异构 Ray 集群（H200 / H100 / GB10 混布）下，NeMo-RL 默认按「任意空闲 GPU」调度，
无法保证作业落在目标卡型。本模块用猴子补丁，在 NeMo-RL 真正创建 placement group 时，
把环境变量 `NRL_PIN_RESOURCE` 指定的 Ray 自定义资源合并进每个 GPU bundle——
Ray 便只会把作业调度到带该资源的节点。

pin 资源来源（由 console 提交时按 profile→series 注入 `NRL_PIN_RESOURCE`）：
  - 优先复用 Ray 自动探测的 `accelerator_type:H200` / `accelerator_type:H100`；
  - Ray 不识别的卡（如 GB10）用运维在 `ray start --resources` 手工注册的自定义资源。

无 `NRL_PIN_RESOURCE`（本地直跑 / 单一卡型集群）时为 no-op。

注入点为什么选在 `placement_group()` 调用上
--------------------------------------------
上一版是在 `RayVirtualCluster.__init__` 之后给实例设 `node_resource_constraints`
属性，指望上游建 bundle 时会读它。2026-07-30 事故表明这个假设不成立：现场
`/opt/nemo-rl` 那版压根没有这个字段，赋值石沉大海，而补丁照样打印「已注入」。
结果 gb10 的作业被 Ray 随手扔到 H200 上，撞进别人的显存里 OOM。

`placement_group(bundles=...)` 是 Ray 的公开 API，也是 bundle 规格通往调度器的唯一
出口。在这里注入，无论上游怎么重构 bundle 的构造逻辑都拦得住，且能当场核对结果。

失败必须炸，不能静默降级
------------------------
pin 失效的后果是作业跑错卡型、绕过时段闸门、挤爆别人的显存，比作业直接失败严重得多。
所以本模块只在「没要求 pin」时保持沉默，一旦要求了 pin 又做不到，一律抛
`PinError` 让作业当场失败。
"""
from __future__ import annotations

import os

# 每个 bundle 对 pin 资源的需求量：取极小值，仅作「亲和标记」，不消耗节点资源额度。
# 与 NeMo-RL NVLink 域 pin 的取值（0.001）一致。
_PIN_AMOUNT = 0.001

_PATCHED = False


class PinError(RuntimeError):
    """卡型 pin 无法生效。调用方不应捕获——带着错误的卡型跑下去比失败更糟。"""


def pin_resource() -> str:
    """本次作业要求的 pin 资源名；未要求时为空串。"""
    return os.environ.get("NRL_PIN_RESOURCE", "").strip()


def apply_pin_patch() -> None:
    """幂等地给 NeMo-RL 的 placement group 创建路径打上卡型 pin 补丁。

    Raises:
        PinError: 要求了 pin 但补丁挂不上去（nemo_rl 不可导入、或上游 API 变了）。
    """
    global _PATCHED
    pin = pin_resource()
    if not pin or _PATCHED:
        return

    try:
        import nemo_rl.distributed.virtual_cluster as vc
    except ImportError as e:
        raise PinError(f"要求 pin 到 {pin}，但 nemo_rl 不可导入：{e}") from e

    orig = getattr(vc, "placement_group", None)
    if not callable(orig):
        raise PinError(
            f"要求 pin 到 {pin}，但 nemo_rl.distributed.virtual_cluster 里找不到 "
            f"placement_group——上游 API 已变，pin 无法生效，拒绝带着错误的卡型继续。"
        )

    def _patched_placement_group(bundles, *args, **kwargs):
        return orig(_pin_bundles(bundles, pin), *args, **kwargs)

    vc.placement_group = _patched_placement_group
    _PATCHED = True
    print(f"[nemolab] 卡型 pin 补丁已应用：{pin}")


def _pin_bundles(bundles, pin: str):
    """把 pin 资源合并进每个 GPU bundle，并把结果打出来备查。

    只动申请了 GPU 的 bundle：CPU-only 的 bundle（如 NeMo-RL 建 venv 时按节点铺开的
    STRICT_SPREAD 组）pin 上去只会让它无处可去。
    """
    if not isinstance(bundles, list):
        raise PinError(f"要求 pin 到 {pin}，但 bundles 不是 list（{type(bundles).__name__}），无法注入。")
    for b in bundles:
        if not isinstance(b, dict):
            raise PinError(f"要求 pin 到 {pin}，但 bundle 不是 dict（{type(b).__name__}），无法注入。")

    def _wants_gpu(b: dict) -> bool:
        return float(b.get("GPU") or 0) > 0

    if not any(_wants_gpu(b) for b in bundles):
        return bundles  # 纯 CPU 的辅助组，没什么可 pin 的

    _assert_cluster_has(pin)
    pinned = [{**b, pin: b.get(pin, _PIN_AMOUNT)} if _wants_gpu(b) else dict(b) for b in bundles]
    print(f"[nemolab] 卡型 pin 已注入 bundle：{pinned}")
    return pinned


def _assert_cluster_has(pin: str) -> None:
    """集群里必须有节点带这个资源，否则 placement group 只会干等到超时。

    与其让运维对着一句「Timed out waiting for placement groups」猜半天，
    不如当场说清楚是哪张卡型没上线。
    """
    import ray

    if not ray.is_initialized():
        return  # 还没连上集群就查不了；留给下一次（真正建 PG 时）再查

    total = 0.0
    for node in ray.nodes():
        if node.get("Alive"):
            total += float(node.get("Resources", {}).get(pin, 0) or 0)
    if total > 0:
        return

    online = sorted(
        {
            key
            for node in ray.nodes()
            if node.get("Alive")
            for key in node.get("Resources", {})
            if key.startswith("accelerator_type:") or key.startswith("acc_")
        }
    )
    raise PinError(
        f"要求 pin 到 {pin}，但集群里没有任何在线节点带这个资源。"
        f"当前在线卡型资源：{online or '（无）'}。"
        f"请确认目标卡型的节点已加入集群，或 ray start 时用 --resources 注册了该资源。"
    )
