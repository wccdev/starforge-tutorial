"""框架无关的 NeMoLab 上报 SDK —— 让**任何**训练脚本都能把曲线打进 console。

背景：本仓库原有的上报链路是 `patch.py` 猴子补丁 `nemo_rl.utils.logger.Logger`，
只对 NeMo-RL 生效。一旦实验换框架（TRL / verl / 纯 HF Trainer / 自己写的循环），
作业在 console 上就只剩一堆日志、没有任何指标曲线。本模块把底下那套传输设施
（IngestClient + HardwareMonitor，本来就与框架无关）暴露成 5 行就能接的公开 API。

用法一 —— 手写训练循环：

    from common.observability import report

    report.init(hparams={"lr": 1e-5})          # 无 NEMOLAB_TOKEN（本地直跑）时全程 no-op
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        report.log({"loss": loss}, step=step)
    report.finish()

用法二 —— HF Trainer / TRL（SFTTrainer、GRPOTrainer、GOLDTrainer…）：

    from common.observability.report import NeMoLabCallback

    trainer = SFTTrainer(..., callbacks=[NeMoLabCallback()])

设计约束（都是被真实事故推着定下来的）：
  * **绝不抛异常**。上报是旁路，采集挂了也不能带崩训练——所以每个公开函数都吞异常。
  * **无 token 即 no-op**。本地直跑 / 单测环境下不该有任何网络行为。
  * **不 import nemo_rl**。这正是本模块存在的意义。
"""

from __future__ import annotations

import atexit
import os
import threading
from datetime import datetime, timezone
from typing import Any, Mapping

from common.observability.util import flatten_dict, scalarize_metric

_lock = threading.Lock()
_state: dict[str, Any] = {"inited": False, "hw": None, "step": 0}


def enabled() -> bool:
    """console 是否注入了上报凭据（本地直跑为 False）。"""
    return bool(os.environ.get("NEMOLAB_TOKEN"))


def init(
    *,
    hparams: Mapping[str, Any] | None = None,
    monitor_hardware: bool = True,
    monitor_interval: float | None = None,
) -> bool:
    """启动上报会话。可重复调用（幂等）。返回是否真的启用了上报。

    与 `nemolab_boot.py` 共享同一个 `ObservabilitySession`：若入口已经过 boot 包装
    （集中提交的默认路径），这里只是复用现成会话，不会重复建连或重复装终端捕获。

    Args:
        hparams:          超参，落到 console 的「配置」面板。
        monitor_hardware: 是否起后台线程采每卡 GPU/显存/功耗。自定义框架建议开——
                          否则 console 上这个作业的硬件面板会是空的。
        monitor_interval: 采样间隔秒；None 则用 NEMOLAB_MONITOR_INTERVAL，默认 10。
    """
    if not enabled():
        return False
    with _lock:
        if _state["inited"]:
            if hparams:
                log_hparams(hparams)
            return True
        try:
            from common.observability.session import get_ingest, start_observability

            start_observability()
            ingest = get_ingest()
            if ingest is None:
                return False

            if monitor_hardware:
                from common.observability.hardware_monitor import HardwareMonitor

                interval = float(
                    monitor_interval
                    if monitor_interval is not None
                    else os.environ.get("NEMOLAB_MONITOR_INTERVAL", "10")
                )
                hw = HardwareMonitor(ingest, collection_interval=interval)
                hw.start()
                _state["hw"] = hw

            _state["inited"] = True
            atexit.register(finish)
            print(f"[nemolab] report SDK 已启用（run={ingest.run_id}）", flush=True)
        except Exception as e:  # 采集是旁路，任何异常都不应影响训练
            print(f"[nemolab] report init 跳过: {e}", flush=True)
            return False

    if hparams:
        log_hparams(hparams)
    return True


def log(metrics: Mapping[str, Any], step: int | None = None, prefix: str = "") -> None:
    """上报一组标量指标。

    非标量（tensor / ndarray / list）会被 `scalarize_metric` 折算成均值；折算不出来的直接丢弃，
    不会报错——训练脚本往往顺手把整个 metrics dict 丢进来，里面混着字符串和张量是常态。

    step 省略时自动递增，方便「每个 epoch 调一次」这种粗粒度用法。
    """
    if not _state["inited"]:
        return
    try:
        from common.observability.session import get_ingest

        ingest = get_ingest()
        if ingest is None:
            return

        if step is None:
            _state["step"] += 1
            step = _state["step"]
        else:
            _state["step"] = int(step)

        flat = flatten_dict(dict(metrics))
        ts = datetime.now(timezone.utc).isoformat()
        points = []
        for key, value in flat.items():
            scalar = scalarize_metric(value)
            if scalar is None:
                continue
            name = f"{prefix}/{key}" if prefix and not key.startswith(f"{prefix}/") else key
            points.append({"key": name, "step": int(step), "value": scalar, "ts": ts})
        if points:
            ingest.enqueue_metrics(points)
    except Exception:
        pass


def log_hparams(params: Mapping[str, Any]) -> None:
    """上报超参（console「配置」面板）。"""
    if not _state["inited"]:
        return
    try:
        from common.observability.session import get_ingest

        if (ingest := get_ingest()) is not None:
            ingest.enqueue_hparams(flatten_dict(dict(params)))
    except Exception:
        pass


def finish() -> None:
    """停止硬件采集并把缓冲区刷干净。幂等；atexit 也会兜底调用。"""
    with _lock:
        if not _state["inited"]:
            return
        _state["inited"] = False
        hw = _state.pop("hw", None)
    try:
        if hw is not None:
            hw.stop()
        from common.observability.session import get_ingest

        if (ingest := get_ingest()) is not None:
            ingest.flush()
    except Exception:
        pass


class NeMoLabCallback:
    """HuggingFace `TrainerCallback` 适配器 —— TRL 各类 Trainer 同样适用。

    不继承 `transformers.TrainerCallback`：那样会把 transformers 变成本模块的 import 期依赖，
    而本文件必须在没有 transformers 的客户端 venv 里也能 import（单测就在那里跑）。
    HF 的回调机制是鸭子类型的——只按方法名调用，不做 isinstance 检查——所以直接实现同名方法即可。
    """

    def __init__(self, *, monitor_hardware: bool = True, prefix: str = "") -> None:
        self.monitor_hardware = monitor_hardware
        self.prefix = prefix

    def on_train_begin(self, args=None, state=None, control=None, **kwargs):
        hparams: dict[str, Any] = {}
        try:
            if args is not None and hasattr(args, "to_sanitized_dict"):
                hparams = args.to_sanitized_dict()
        except Exception:
            hparams = {}
        init(hparams=hparams or None, monitor_hardware=self.monitor_hardware)
        return control

    def on_log(self, args=None, state=None, control=None, logs=None, **kwargs):
        if logs:
            step = int(getattr(state, "global_step", 0) or 0)
            log(logs, step=step, prefix=self.prefix)
        return control

    def on_evaluate(self, args=None, state=None, control=None, metrics=None, **kwargs):
        if metrics:
            step = int(getattr(state, "global_step", 0) or 0)
            log(metrics, step=step, prefix="validation")
        return control

    def on_train_end(self, args=None, state=None, control=None, **kwargs):
        finish()
        return control
