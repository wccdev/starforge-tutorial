"""硬件监控采样策略：随已采集轮次拉长间隔，控制长训练的上报量。"""

from __future__ import annotations

# 0~10 点 10s，10~50 点 30s，50+ 点 60s
MONITOR_TIER1 = 10
MONITOR_TIER2 = 50
MONITOR_INTERVAL_SHORT = 10.0
MONITOR_INTERVAL_MID = 30.0
MONITOR_INTERVAL_LONG = 60.0
MONITOR_MIN_INTERVAL = 5.0


def monitor_interval(
    samples_collected: int,
    *,
    base_interval: float = MONITOR_INTERVAL_SHORT,
    dynamic: bool = True,
) -> float:
    """根据已采集轮次返回下一次 sleep 间隔（秒）。"""
    base = max(MONITOR_MIN_INTERVAL, float(base_interval))
    if not dynamic:
        return base
    n = max(0, int(samples_collected))
    if n < MONITOR_TIER1:
        return base
    if n < MONITOR_TIER2:
        return max(base, MONITOR_INTERVAL_MID)
    return max(base, MONITOR_INTERVAL_LONG)
