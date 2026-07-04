"""卡型 pin：给 NeMo-RL 的 RayVirtualCluster 注入 node_resource_constraints。

异构 Ray 集群（H200 / H100 / GB10 混布）下，NeMo-RL main 默认按「任意空闲 GPU」
调度，无法保证作业落在目标卡型。本模块用猴子补丁在 RayVirtualCluster 构造后，
把环境变量 `NRL_PIN_RESOURCE` 指定的 Ray 自定义资源注入到每个 bundle 的
node_resource_constraints——Ray 便只会把作业调度到带该资源的节点（即目标卡型）。

pin 资源来源（由 console 提交时按 profile→series 注入 NRL_PIN_RESOURCE）：
  - 优先复用 Ray 自动探测的 `accelerator_type:H200` / `accelerator_type:H100`；
  - Ray 不识别的卡（如 GB10）用运维在 `ray start --resources` 手工注册的自定义资源。

无 `NRL_PIN_RESOURCE`（本地直跑 / 单一卡型集群）时为 no-op，行为与未打补丁一致。
任何异常都不影响训练（pin 是调度优化，失败则回落 NeMo-RL 默认调度）。
"""
from __future__ import annotations

import os

# 每个 bundle 对 pin 资源的需求量：取极小值，仅作「亲和标记」，不消耗节点资源额度。
# 与 NeMo-RL NVLink 域 pin 的取值（0.001）一致。
_PIN_AMOUNT = 0.001

_PATCHED = False


def apply_pin_patch() -> None:
    """幂等地给 RayVirtualCluster.__init__ 打上卡型 pin 补丁。"""
    global _PATCHED
    if _PATCHED:
        return
    if not os.environ.get("NRL_PIN_RESOURCE", "").strip():
        return  # 未指定 pin：no-op
    try:
        import nemo_rl.distributed.virtual_cluster as vc
    except ImportError:
        print("[nemolab] pin patch skipped: nemo_rl not importable")
        return

    cls = vc.RayVirtualCluster
    orig_init = cls.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            _inject_pin(self)
        except Exception as e:  # noqa: BLE001 — pin 是旁路优化，绝不影响训练
            print(f"[nemolab] pin injection failed (training continues): {e}")

    cls.__init__ = _patched_init
    _PATCHED = True
    print("[nemolab] RayVirtualCluster 卡型 pin 补丁已应用")


def _inject_pin(cluster) -> None:
    """把 pin 资源合并进每个逻辑节点的 node_resource_constraints。"""
    pin = os.environ.get("NRL_PIN_RESOURCE", "").strip()
    if not pin:
        return
    # CPU-only 集群不做卡型 pin（无 accelerator_type 资源，注入会导致无法调度）。
    if not getattr(cluster, "use_gpus", True):
        return
    bundle_list = getattr(cluster, "_bundle_ct_per_node_list", None)
    if not bundle_list:
        return
    n = len(bundle_list)
    existing = getattr(cluster, "node_resource_constraints", None)
    if existing is None:
        cluster.node_resource_constraints = [{pin: _PIN_AMOUNT} for _ in range(n)]
    else:
        # 已有约束（如 NVLink 域 pin）：合并 pin，不覆盖既有 key。
        if len(existing) != n:
            print(
                f"[nemolab] pin skipped: node_resource_constraints 长度 {len(existing)} "
                f"≠ 节点数 {n}"
            )
            return
        for d in existing:
            if isinstance(d, dict):
                d.setdefault(pin, _PIN_AMOUNT)
    print(f"[nemolab] 卡型 pin 已注入 {n} 个节点 bundle：{{{pin}: {_PIN_AMOUNT}}}")
