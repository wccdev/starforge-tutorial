"""Monkey-patch nemo_rl.utils.logger：NeMoLabLogger + 验证样本结构化上报。"""
from __future__ import annotations

import os

_PATCHED = False


def _val_upload_config() -> tuple[int | None, int]:
    """验证样本上报配置（环境变量，默认上报全量）。

    - NEMOLAB_VAL_UPLOAD_SAMPLES：整轮上报的样本数上限；0/未设/非法 → None（全量）。
    - NEMOLAB_VAL_CHUNK：分片大小，单片样本条数（默认 64），避免大 payload 触发代理体积限制。
    """
    raw = os.environ.get("NEMOLAB_VAL_UPLOAD_SAMPLES", "").strip()
    upload_n: int | None = None
    if raw:
        try:
            v = int(raw)
            upload_n = v if v > 0 else None
        except ValueError:
            upload_n = None
    try:
        chunk = int(os.environ.get("NEMOLAB_VAL_CHUNK", "64"))
    except ValueError:
        chunk = 64
    if chunk <= 0:
        chunk = 64
    return upload_n, chunk


def apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    if not os.environ.get("NEMOLAB_TOKEN"):
        return
    try:
        import nemo_rl.utils.logger as logger_mod
    except ImportError:
        print("NeMoLab patch skipped: nemo_rl not importable")
        return

    from common.observability.logger import NeMoLabLogger
    from common.observability.session import get_ingest
    from common.observability.validation_ctx import active_validation_step, clear_validation_step
    from common.observability.validation_extract import extract_message_log_samples

    _orig_init = logger_mod.Logger.__init__
    _orig_del = getattr(logger_mod.Logger, "__del__", None)
    _orig_print_samples = logger_mod.print_message_log_samples

    def _patched_init(self, cfg):
        _orig_init(self, cfg)
        nemolab_log_dir = os.path.join(self.base_log_dir, "nemolab")
        os.makedirs(nemolab_log_dir, exist_ok=True)
        try:
            self.nemolab_logger = NeMoLabLogger({}, log_dir=nemolab_log_dir)
            self.loggers.append(self.nemolab_logger)
        except Exception as e:
            print(f"NeMoLab logger init failed (training continues): {e}")
            self.nemolab_logger = None

    def _patched_del(self):
        nl = getattr(self, "nemolab_logger", None)
        if nl is not None:
            nl.finish()
        if _orig_del is not None:
            _orig_del(self)

    def _patched_print_message_log_samples(
        message_logs, rewards, num_samples=5, step=0
    ):
        # 日志仍只打印 num_samples 条；上报数量与打印数量解耦（见 _val_upload_config）。
        _orig_print_samples(message_logs, rewards, num_samples=num_samples, step=step)
        ingest = get_ingest()
        if ingest is None:
            return
        val_step = active_validation_step()
        if val_step is None or val_step != step:
            return
        try:
            upload_n, chunk_size = _val_upload_config()
            samples, dist, avg_reward = extract_message_log_samples(
                message_logs, rewards, num_samples=upload_n
            )
            if not samples:
                return
            total = len(samples)
            chunks = [samples[i : i + chunk_size] for i in range(0, total, chunk_size)]
            for ci, part in enumerate(chunks):
                payload = {
                    "run_id": ingest.run_id,
                    "step": val_step,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                    "total_samples": total,
                    "samples": part,
                }
                if ci == 0:  # 元数据只随首片上报
                    payload["avg_reward"] = avg_reward
                    payload["dist"] = dist
                ingest.enqueue_validation(payload)
        except Exception as e:
            print(f"NeMoLab validation upload failed (training continues): {e}")
        finally:
            clear_validation_step()

    logger_mod.Logger.__init__ = _patched_init
    logger_mod.Logger.__del__ = _patched_del
    logger_mod.print_message_log_samples = _patched_print_message_log_samples
    _PATCHED = True
    print("NeMoLab logger patch applied")
