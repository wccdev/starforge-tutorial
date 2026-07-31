"""训练入口包装器：先给 NeMo-RL Logger 挂上 NeMoLabLogger 后端，再运行原始入口。

由 scripts/_run_experiment.sh 调用：
    uv run python scripts/nemolab_boot.py <ENTRY> [args...]
等价于 `python <ENTRY> [args...]`，唯一区别是运行前 apply_patch()。
无 NEMOLAB_TOKEN（本地直跑）时 patch 为 no-op，行为与直接 `python <ENTRY>` 完全一致。
"""
from __future__ import annotations

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/nemolab_boot.py <entry.py> [args...]", file=sys.stderr)
        return 2

    # 卡型 pin：独立于可观测性（本地直跑也可能需要 pin；由 NRL_PIN_RESOURCE 控制，
    # 未设则 no-op）。放在 import 训练入口前，确保补丁先于 placement group 创建生效。
    #
    # 这里刻意不吞异常：pin 失效意味着作业会跑到错误的卡型上、绕过时段闸门、挤爆
    # 别人的显存，比作业当场失败严重得多。
    if os.environ.get("NRL_PIN_RESOURCE", "").strip():
        from common.ray_pin import apply_pin_patch

        apply_pin_patch()

    # GB10 NVML 兜底：Megatron log_gpu_memory → torch.cuda.device_memory_used 在
    # 统一内存上会抛 NotSupported。必须在 init_ray 之前装好 Ray worker hook，
    # 否则补丁只落在 driver、进不了 MegatronPolicyWorker。失败不阻断训练。
    try:
        from common.gb10_nvml_patch import apply_patch as apply_nvml_patch

        apply_nvml_patch()
    except Exception as e:
        print(f"[nemolab] GB10 NVML patch skipped: {e}")

    try:
        from common.observability.session import start_observability

        start_observability()
        from common.observability.patch import apply_patch

        apply_patch()
    except Exception as e:  # 采集是旁路，任何异常都不应影响训练
        print(f"[nemolab] patch skipped: {e}")

    entry = sys.argv[1]
    sys.argv = [entry, *sys.argv[2:]]
    try:
        runpy.run_path(entry, run_name="__main__")
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    finally:
        try:
            from common.observability.session import stop_observability

            stop_observability()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
