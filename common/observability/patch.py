"""Monkey-patch nemo_rl.utils.logger：NeMoLabLogger + 验证样本结构化上报。"""
from __future__ import annotations

import os

_PATCHED = False

# agent 轨迹含检索 env，单条可达数十 KB；默认小片 + 字段截断，避免整轮 POST 超时被静默丢掉。
_DEFAULT_CHUNK = 8
_DEFAULT_FIELD_CHARS = 12_000


def _val_upload_config() -> tuple[int | None, int, int]:
    """验证样本上报配置（环境变量）。

    - NEMOLAB_VAL_UPLOAD_SAMPLES：整轮上报条数上限；0/未设/非法 → None（全量）。
    - NEMOLAB_VAL_CHUNK：分片大小（默认 8），避免大 payload 触发代理体积限制 / 超时。
    - NEMOLAB_VAL_FIELD_CHARS：单字段（user/assistant/env）字符上限（默认 12000）。
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
        chunk = int(os.environ.get("NEMOLAB_VAL_CHUNK", str(_DEFAULT_CHUNK)))
    except ValueError:
        chunk = _DEFAULT_CHUNK
    if chunk <= 0:
        chunk = _DEFAULT_CHUNK
    try:
        field_chars = int(os.environ.get("NEMOLAB_VAL_FIELD_CHARS", str(_DEFAULT_FIELD_CHARS)))
    except ValueError:
        field_chars = _DEFAULT_FIELD_CHARS
    if field_chars <= 0:
        field_chars = _DEFAULT_FIELD_CHARS
    return upload_n, chunk, field_chars


def _enqueue_samples(ingest, step: int, samples, chunk_size: int, avg_reward=None, dist=None) -> None:
    total = len(samples)
    chunks = [samples[i : i + chunk_size] for i in range(0, total, chunk_size)]
    ok = 0
    for ci, part in enumerate(chunks):
        payload = {
            "run_id": ingest.run_id,
            "step": step,
            "chunk_index": ci,
            "total_chunks": len(chunks),
            "total_samples": total,
            "samples": part,
        }
        if ci == 0:  # 元数据只随首片上报
            payload["avg_reward"] = avg_reward
            payload["dist"] = dist
        if ingest.enqueue_validation(payload):
            ok += 1
    if ok < len(chunks):
        print(
            f"NeMoLab validation upload partial: step={step} "
            f"chunks_ok={ok}/{len(chunks)} samples={total}",
            flush=True,
        )
    else:
        print(
            f"NeMoLab validation upload ok: step={step} samples={total} chunks={len(chunks)}",
            flush=True,
        )


def _upload_validation_samples(ingest, step: int, message_logs, rewards) -> None:
    from common.observability.validation_extract import extract_message_log_samples

    upload_n, chunk_size, field_chars = _val_upload_config()
    samples, dist, avg_reward = extract_message_log_samples(
        message_logs,
        rewards,
        num_samples=upload_n,
        max_field_chars=field_chars,
    )
    if samples:
        _enqueue_samples(ingest, step, samples, chunk_size, avg_reward=avg_reward, dist=dist)


def _dpo_message_log_parts(message_log, tokenizer) -> list[str]:
    """message_log 各消息转纯文本（decode token_ids 去特殊 token，失败回退 content）。"""
    parts = []
    for m in message_log:
        text = ""
        ids = m.get("token_ids")
        if ids:
            try:
                text = tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                text = ""
        if not text:
            text = str(m.get("content") or "")
        text = text.strip()
        if text:
            parts.append(text)
    return parts


def _upload_dpo_samples(ingest, step: int, val_dataloader, tokenizer) -> None:
    """DPO 验证结束：从 val_dataloader 底层数据集抽 prompt/chosen/rejected 偏好样本。

    nemo-rl 的 DPO validate 只算聚合 loss，不打印样本；逐样本 margin 也未暴露
    （loss 里 rewards_chosen/rejected 只输出 batch 均值），故样本 reward 置 None，
    margin 趋势走 validation/* 指标曲线。样本的价值在文本对比。
    """
    if not val_dataloader:
        return
    loader = next(iter(val_dataloader.values()))
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return
    upload_n, chunk_size, field_chars = _val_upload_config()
    total = len(dataset)
    limit = total if upload_n is None else min(upload_n, total)
    samples = []
    for i in range(limit):
        datum = dataset[i]
        chosen_log = datum.get("message_log_chosen") or []
        rejected_log = datum.get("message_log_rejected") or []
        chosen_parts = _dpo_message_log_parts(chosen_log, tokenizer)
        if not chosen_parts:
            continue
        rejected_parts = _dpo_message_log_parts(rejected_log, tokenizer)
        # message_log_chosen = context + completion，最后一条即 chosen，其余为 prompt
        prompt, chosen = "\n".join(chosen_parts[:-1]), chosen_parts[-1]
        rejected = rejected_parts[-1] if rejected_parts else ""
        samples.append(
            {
                "idx": i,
                "user": prompt[:field_chars],
                "assistant": chosen[:field_chars],
                "reward": None,
                "env_info": "",
                "extra": {
                    "chosen": chosen[:field_chars],
                    "rejected": rejected[:field_chars],
                },
            }
        )
    if samples:
        _enqueue_samples(ingest, step, samples, chunk_size)


def _patch_dpo_validate() -> None:
    """包装 nemo_rl.algorithms.dpo.validate：验证结束后上报偏好样本。"""
    try:
        import nemo_rl.algorithms.dpo as dpo_mod
    except ImportError:
        return
    if getattr(dpo_mod.validate, "__nemolab_dpo_patched__", False):
        return
    _orig_validate = dpo_mod.validate

    def _wrapped_validate(*args, **kwargs):
        result = _orig_validate(*args, **kwargs)
        # 样本上报默认开，NEMOLAB_VAL_DPO_SAMPLES=0 关闭
        if os.environ.get("NEMOLAB_VAL_DPO_SAMPLES", "1").strip().lower() not in ("0", "false"):
            try:
                from common.observability.session import get_ingest

                ingest = get_ingest()
                if ingest is not None:
                    # validate(policy, val_dataloader, tokenizer, ..., step, ...)：位置/关键字混合防御
                    val_dataloader = kwargs.get("val_dataloader") or (args[1] if len(args) > 1 else None)
                    tokenizer = kwargs.get("tokenizer") or (args[2] if len(args) > 2 else None)
                    step = kwargs.get("step") if "step" in kwargs else (args[4] if len(args) > 4 else 0)
                    if val_dataloader is not None and tokenizer is not None:
                        _upload_dpo_samples(ingest, int(step), val_dataloader, tokenizer)
            except Exception as e:
                print(f"NeMoLab DPO validation upload failed (training continues): {e}", flush=True)
        return result

    _wrapped_validate.__nemolab_dpo_patched__ = True
    dpo_mod.validate = _wrapped_validate


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
        # GRPO/PPO 仅在验证结束调用本函数。上报必须独立于 rich 打印：
        # agent 长轨迹打印失败/超时会抛异常，旧实现先 print 再 upload → 整轮样本丢失。
        ingest = get_ingest()
        val_step = active_validation_step()
        # 日志解析到的 step 优先；解析失败时回退到调用方传入的 step（验证场景可靠）。
        if val_step is not None and val_step != step:
            # 防御：上下文与调用 step 不一致时不误报
            upload_step = None
        else:
            upload_step = val_step if val_step is not None else step

        if ingest is not None and upload_step is not None:
            try:
                _upload_validation_samples(ingest, upload_step, message_logs, rewards)
            except Exception as e:
                print(f"NeMoLab validation upload failed (training continues): {e}", flush=True)
            finally:
                clear_validation_step()
        elif val_step is not None:
            clear_validation_step()

        try:
            _orig_print_samples(message_logs, rewards, num_samples=num_samples, step=step)
        except Exception as e:
            print(f"NeMoLab sample print failed (upload may still have succeeded): {e}", flush=True)

    logger_mod.Logger.__init__ = _patched_init
    logger_mod.Logger.__del__ = _patched_del
    logger_mod.print_message_log_samples = _patched_print_message_log_samples
    _patch_dpo_validate()
    _PATCHED = True
    print("NeMoLab logger patch applied")
