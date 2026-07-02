"""日志上报前格式化：去 ANSI / Ray worker 前缀，无时间戳行补全本地时间（SwanLab 风格）。"""
from __future__ import annotations

import re
from datetime import datetime, timezone

ANSI_ESC = re.compile(r"\x1b\[[0-9;]*m", re.IGNORECASE)
ANSI_BARE = re.compile(r"\[(?:\d{1,3})m")
RAY_WORKER_PREFIX = re.compile(r"\([A-Za-z][\w.]*\s+pid=\d+,\s*ip=[\d.]+\)\s*")
ISO_TS_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+")
VLLM_LOG_TS = re.compile(r"^(INFO|WARNING|ERROR|DEBUG)\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+")


def _strip_ansi(text: str) -> str:
    return ANSI_BARE.sub("", ANSI_ESC.sub("", text))


def _strip_ray_prefix(text: str) -> str:
    return RAY_WORKER_PREFIX.sub("", text, count=1).lstrip()


def _vllm_ts_to_iso(partial: str, *, now: datetime) -> str:
    md, _, time_part = partial.partition(" ")
    if not md or not time_part:
        return ""
    mo, _, da = md.partition("-")
    if not mo or not da:
        return ""
    return f"{now.year}-{mo.zfill(2)}-{da.zfill(2)} {time_part}"


def format_log_line(line: str, *, now: datetime | None = None) -> str:
    """单行：清洗前缀/ANSI；已有 ISO 时间戳则保留；否则补当前时间。"""
    now = now or datetime.now(timezone.utc).astimezone()
    raw = line.rstrip("\r\n")
    nl = line[len(raw) :] if len(line) > len(raw) else ""

    cleaned = _strip_ray_prefix(_strip_ansi(raw))
    cleaned = _strip_ray_prefix(cleaned)

    if not cleaned.strip():
        return line

    if ISO_TS_PREFIX.match(cleaned):
        return f"{cleaned}{nl}"

    m = VLLM_LOG_TS.match(cleaned)
    if m:
        level, partial = m.group(1), m.group(2)
        ts = _vllm_ts_to_iso(partial, now=now)
        rest = cleaned[m.end() :].lstrip()
        body = f"{level} {rest}".strip() if rest else level
        return f"{ts} {body}{nl}" if ts else f"{cleaned}{nl}"

    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"{ts} {cleaned.lstrip()}{nl}"


def format_log_chunk_for_ingest(text: str) -> str:
    """按行格式化后拼接（保留原有换行结构）。"""
    if not text:
        return text
    now = datetime.now(timezone.utc).astimezone()
    # splitlines(keepends=True) 保留空行与末尾无换行片段
    parts = text.splitlines(keepends=True)
    if not parts:
        return format_log_line(text, now=now)
    out: list[str] = []
    for part in parts:
        if part.endswith("\n"):
            out.append(format_log_line(part[:-1], now=now) + "\n")
        elif part.endswith("\r"):
            out.append(format_log_line(part[:-1], now=now) + "\r")
        else:
            out.append(format_log_line(part, now=now))
    return "".join(out)
