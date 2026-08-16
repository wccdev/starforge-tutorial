"""环境/奖励层的轻量指标上报（best-effort，绝不影响训练）。

训练容器里由平台注入 STARFORGE_ENDPOINT / STARFORGE_RUN_ID / STARFORGE_TOKEN；
环境代码（reward hacking 监控指标：judge 降级率、search_rate、长度漂移等）
用本模块把自定义指标打到 console 的 /api/ingest/metrics —— 与训练主指标
同库同看板，诊断规则可以直接消费。

设计约束：
  - **绝不抛异常**：上报失败静默丢弃（本地 print 仍是第一落点）。
  - 无第三方依赖（环境代码运行在训练容器 venv，依赖面必须为零）。
"""
from __future__ import annotations

import json
import os
import urllib.request


def ingest_ready() -> bool:
    return bool(
        os.environ.get("STARFORGE_ENDPOINT")
        and os.environ.get("STARFORGE_RUN_ID")
        and os.environ.get("STARFORGE_TOKEN")
    )


def report_metrics(points: dict[str, float], *, step: int = 0, timeout: float = 5.0) -> bool:
    """上报一批自定义指标（key → value）。成功返回 True；任何失败返回 False。"""
    if not ingest_ready() or not points:
        return False
    endpoint = os.environ["STARFORGE_ENDPOINT"].rstrip("/")
    run_id = os.environ["STARFORGE_RUN_ID"]
    body = json.dumps({
        "run_id": run_id,
        "points": [
            {"key": str(k), "step": int(step), "value": float(v)}
            for k, v in points.items()
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/metrics",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['STARFORGE_TOKEN']}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001 — best-effort：上报通道问题不能影响训练
        return False
