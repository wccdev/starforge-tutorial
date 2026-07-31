"""Ray memory_summary 超时兜底：避免 NeMo-RL MemoryTracker 把作业打死。

GB10 colocated（Megatron + vLLM）在 Step1 Generation 前常把节点打到重负载，
NeMo-RL `memory_tracker.snapshot_start_of_stage` 会调

    ray._private.internal_api.memory_summary → FormatGlobalMemoryInfo(timeout=60)

node manager 60s 无响应就抛 `grpc DEADLINE_EXCEEDED`，且**未被捕获** → 作业 FAILED。
这与训练逻辑无关，只是旁路诊断打印。

本补丁让 `memory_summary` 在 RPC 超时/不可用时返回短占位串，训练继续。

启用（任一即可）：
  - `NRL_PATCH_RAY_MEMORY_SUMMARY=1`
  - `NRL_PIN_RESOURCE=acc_gb10`（GB10 profile 默认）
关闭：`NRL_PATCH_RAY_MEMORY_SUMMARY=0`
"""
from __future__ import annotations

import os

_PATCHED = False


def enabled() -> bool:
    raw = os.environ.get("NRL_PATCH_RAY_MEMORY_SUMMARY", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return os.environ.get("NRL_PIN_RESOURCE", "").strip() == "acc_gb10"


def _is_soft_rpc_failure(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc)
    needles = (
        "DEADLINE_EXCEEDED",
        "Deadline Exceeded",
        "InactiveRpcError",
        "RpcError",
        "unavailable",
        "UNAVAILABLE",
    )
    return any(n in name or n in msg for n in needles)


def apply_patch() -> bool:
    """包装 ray memory_summary；失败或未启用时返回 False。"""
    global _PATCHED
    if not enabled():
        return False
    if _PATCHED:
        return True
    try:
        import ray._private.internal_api as api
    except Exception as e:
        print(f"[nemolab] Ray memory_summary patch skipped (import): {e}")
        return False

    if getattr(api, "_nemolab_memory_summary_patched", False):
        _PATCHED = True
        return True

    orig = api.memory_summary

    def safe_memory_summary(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except Exception as e:
            if _is_soft_rpc_failure(e):
                return (
                    "[nemolab] ray memory_summary skipped "
                    f"({type(e).__name__}: {str(e)[:180]})"
                )
            raise

    api.memory_summary = safe_memory_summary  # type: ignore[assignment]
    api._nemolab_memory_summary_patched = True
    _PATCHED = True
    print("[nemolab] Ray memory_summary patch: soft-fail on FormatGlobalMemoryInfo timeout")
    return True
