"""StarForgeLogger：实现 NeMo-RL LoggerInterface 兼容接口，主动上报 console。

指标上报只是**兜底**：平台侧 starforge_core 的框架桥装上后会设
STARFORGE_METRICS_BRIDGE=1，本类据此让路 —— 两个生产端同时上报同一批指标就是双份
数据，而且两端的嵌套键展平分隔符不同（平台用 "/"，这里用 "."），双写会把同一个
指标裂成两条曲线。只有平台桥缺席的路径（离线跑 run.py、自定义镜像入口）才由这里报。

硬件采集与超参上报没有第二个生产者，不受这个开关影响。
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from common.observability.hardware_monitor import HardwareMonitor
from common.observability.session import get_ingest
from common.observability.util import flatten_dict, scalarize_metric

#: 由 starforge_core.frameworks.observability.METRICS_BRIDGE_ENV 设置。
METRICS_BRIDGE_ENV = "STARFORGE_METRICS_BRIDGE"


def _platform_bridge_active() -> bool:
    return os.environ.get(METRICS_BRIDGE_ENV) == "1"


class StarForgeLogger:
    """Drop-in backend，经 common.observability.patch 挂到 nemo_rl.utils.logger.Logger。"""

    def __init__(self, cfg: dict | None = None, log_dir: str | None = None):
        del log_dir
        cfg = cfg or {}
        ingest = get_ingest()
        if ingest is None:
            raise ValueError(
                "StarForgeLogger requires active observability session "
                "(start_observability before NeMo-RL Logger init)"
            )
        self._ingest = ingest
        monitor_interval = float(
            cfg.get("monitor_interval")
            or os.environ.get("STARFORGE_MONITOR_INTERVAL", "10")
        )
        monitor_dynamic = str(
            cfg.get(
                "monitor_dynamic_interval",
                os.environ.get("STARFORGE_MONITOR_DYNAMIC", "1"),
            )
        ).lower() not in ("0", "false", "no")
        monitor_hardware = str(
            cfg.get("monitor_hardware", os.environ.get("STARFORGE_MONITOR_HARDWARE", "1"))
        ).lower() not in ("0", "false", "no")
        scope_raw = str(
            cfg.get("monitor_scope")
            or os.environ.get("STARFORGE_MONITOR_SCOPE")
            or (
                "cluster"
                if str(
                    cfg.get(
                        "monitor_cluster",
                        os.environ.get("STARFORGE_MONITOR_CLUSTER", "0"),
                    )
                ).lower()
                in ("1", "true", "yes")
                else "job"
            )
        ).lower()
        monitor_scope = scope_raw if scope_raw in ("local", "job", "cluster") else "job"
        self._hw_monitor: HardwareMonitor | None = None
        if monitor_hardware:
            self._hw_monitor = HardwareMonitor(
                self._ingest,
                collection_interval=monitor_interval,
                dynamic_interval=monitor_dynamic,
                scope=monitor_scope,  # type: ignore[arg-type]
            )
            self._hw_monitor.start()

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: Optional[str] = "",
        step_metric: Optional[str] = None,
        step_finished: bool = False,
    ) -> None:
        del step_metric, step_finished
        if _platform_bridge_active():
            return
        flat = flatten_dict(metrics)
        if prefix:
            flat = {
                f"{prefix}/{k}" if not k.startswith(f"{prefix}/") else k: v
                for k, v in flat.items()
            }
        ts = datetime.now(timezone.utc).isoformat()
        points = []
        for key, value in flat.items():
            scalar = scalarize_metric(value)
            if scalar is None:
                continue
            points.append({"key": key, "step": int(step), "value": scalar, "ts": ts})
        if points:
            self._ingest.enqueue_metrics(points)

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self._ingest.enqueue_hparams(flatten_dict(params))

    def log_plot(self, figure, step: int, name: str) -> None:
        del figure, step, name

    def log_histogram(self, histogram: list[Any], step: int, name: str) -> None:
        scalar = scalarize_metric(histogram)
        if scalar is not None:
            self.log_metrics({name: scalar}, step)

    def finish(self) -> None:
        if self._hw_monitor:
            self._hw_monitor.stop()

    def __del__(self) -> None:
        try:
            self.finish()
        except Exception:
            pass
