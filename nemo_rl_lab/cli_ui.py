"""CLI 用户可见的错误 / 提示格式化（stderr，简洁可读）。

避免把 API / 业务错误当作 typer.BadParameter 抛出——那会显示误导性的
「Invalid value:」前缀。风格参考现代 CLI：短标题 + 要点 + 可执行提示。
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import typer


@dataclass(frozen=True)
class ParsedMessage:
    title: str
    items: tuple[str, ...] = ()
    body: str = ""
    hint: str = ""


# 常见服务端文案 → 更短的标题与固定提示
_KNOWN_TITLES: tuple[tuple[str, str], ...] = (
    ("提交前 HuggingFace 资源预检未通过", "HuggingFace 资源预检未通过"),
    ("HuggingFace 资源预检未通过", "HuggingFace 资源预检未通过"),
)

_HINT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("未绑定 HF", "绑定后重试"), "在 Web 控制台绑定 HuggingFace：集成 → HuggingFace"),
    (("gated", "访问条款"), "到 HuggingFace 接受该资源的访问条款后再试"),
    (("请先运行 lab login", "登录"), "运行 lab login 登录"),
    (("登录令牌无效", "登录失败"), "运行 lab login 重新登录"),
)

# 含这些片段时不附加 CLI 侧「→ 提示」（服务端文案已足够或会误导）
_HINT_SUPPRESS = ("不是有效的 HuggingFace repo id", "继承了未 override", "org/name")


def http_error_detail(e: urllib.error.HTTPError, *, fallback: str) -> str:
    """从 HTTP 响应提取可读错误信息（不暴露状态码等实现细节）。"""
    raw = e.read().decode(errors="ignore")
    try:
        payload = json.loads(raw)
        detail = payload.get("detail", payload)
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict) and first.get("msg"):
                return str(first["msg"])
            return str(first)
    except json.JSONDecodeError:
        pass
    text = raw.strip()
    return text[:240] if text else fallback


def parse_message(text: str) -> ParsedMessage:
    """把服务端长文本拆成标题 / 要点 / 提示。"""
    raw = (text or "").strip()
    if not raw:
        return ParsedMessage(title="操作失败")

    title = raw
    items: list[str] = []
    body = ""

    if "\n" in raw:
        first, rest = raw.split("\n", 1)
        title = first.rstrip("：").strip()
        for line in rest.splitlines():
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].strip())
            elif line:
                body = f"{body}\n{line}".strip() if body else line
    elif raw.startswith("- "):
        items.append(raw[2:].strip())
        title = "操作失败"

    for prefix, short in _KNOWN_TITLES:
        if title.startswith(prefix):
            title = short
            break
    if title.endswith("："):
        title = title[:-1]

    hint = _guess_hint(" ".join(items) or raw)
    if not items and not body:
        body = raw if title != raw else ""

    return ParsedMessage(title=title, items=tuple(items), body=body, hint=hint)


def _guess_hint(text: str) -> str:
    if any(s in text for s in _HINT_SUPPRESS):
        return ""
    for keys, hint in _HINT_RULES:
        if any(k in text for k in keys):
            return hint
    return ""


def _shorten_bullet(item: str) -> tuple[str, str]:
    """把「主句；补充说明」拆成两行，便于扫读。"""
    extra = ""
    core = item.strip()
    if "；" in core:
        core, extra = core.split("；", 1)
        core, extra = core.strip(), extra.strip()
    # 常见 HF 预检：「无权访问 dataset X（私有/未授权，401）」
    m = re.match(r"^(无权访问|无法访问|找不到)\s+(model|dataset)\s+(\S+)", core)
    if m:
        verb, kind, name = m.groups()
        name = name.split("（", 1)[0].split("(", 1)[0]
        kind_label = "模型" if kind == "model" else "数据集"
        rest = core[m.end():].strip()
        tail_parts = [p.strip("（）() ") for p in (rest, verb, extra) if p and p.strip("（）() ")]
        return f"{kind_label} {name}", " · ".join(tail_parts)
    if extra:
        return core, extra
    return core, ""


def emit_error(
    title: str,
    *,
    items: Optional[list[str]] = None,
    body: str = "",
    hint: str = "",
) -> None:
    """向 stderr 输出一块结构化错误（不退出）。"""
    typer.echo("", err=True)
    typer.secho(f"  ✗  {title}", fg=typer.colors.RED, bold=True, err=True)
    if items:
        for item in items:
            head, tail = _shorten_bullet(item)
            typer.secho(f"     • {head}", fg=typer.colors.RED, err=True)
            if tail:
                typer.secho(f"       {tail}", err=True)
    elif body:
        for line in body.splitlines():
            typer.secho(f"     {line}", fg=typer.colors.RED, err=True)
    if hint:
        typer.echo("", err=True)
        typer.secho(f"  → {hint}", fg=typer.colors.YELLOW, err=True)
    typer.echo("", err=True)


def emit_warning(title: str, *, body: str = "", hint: str = "") -> None:
    typer.secho(f"  !  {title}", fg=typer.colors.YELLOW, bold=True, err=True)
    if body:
        for line in body.splitlines():
            typer.secho(f"     {line}", err=True)
    if hint:
        typer.secho(f"  → {hint}", fg=typer.colors.YELLOW, err=True)


def fail(
    message: str,
    *,
    title: str = "",
    items: Optional[list[str]] = None,
    hint: str = "",
    code: int = 1,
) -> None:
    """打印错误并退出（替代 typer.BadParameter 用于非参数校验场景）。"""
    if title or items or hint:
        emit_error(title or "操作失败", items=items, body="" if items else message, hint=hint)
    else:
        parsed = parse_message(message)
        emit_error(
            parsed.title,
            items=list(parsed.items) or None,
            body=parsed.body if not parsed.items else "",
            hint=parsed.hint or hint,
        )
    raise typer.Exit(code)


def fail_http(e: urllib.error.HTTPError, *, fallback: str, title: str = "") -> None:
    """HTTP 4xx/5xx：解析 detail 后友好展示并退出。"""
    detail = http_error_detail(e, fallback=fallback)
    parsed = parse_message(detail)
    emit_error(
        title or parsed.title,
        items=list(parsed.items) or None,
        body=parsed.body if not parsed.items else detail,
        hint=parsed.hint,
    )
    raise typer.Exit(1) from e


# ----------------------------- 提交进度条（打包 → 上传 → 受理）-----------------------------
def human_bytes(n: float) -> str:
    """字节数 → 人类可读（1023 B / 5.8 MB）。"""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def format_elapsed(seconds: float) -> str:
    """阶段耗时 → 紧凑可读（3.2s / 1m 05s）。"""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _stage_elapsed(stage: "_StageState") -> float:
    if stage.started is None:
        return 0.0
    end = stage.finished if stage.finished is not None else time.monotonic()
    return max(0.0, end - stage.started)


@dataclass
class _StageState:
    key: str
    label: str
    status: str = "pending"  # pending | active | done
    detail: str = ""
    started: float | None = None
    finished: float | None = None


class _PlainReporter:
    """无 rich / 非 TTY 时的降级上报：分阶段输出 + 完成耗时。"""

    def __init__(self) -> None:
        self._phase_started = time.monotonic()

    def _mark_phase_done(self, label: str) -> None:
        elapsed = format_elapsed(time.monotonic() - self._phase_started)
        typer.secho(f"  ✓  {label}  {elapsed}", fg=typer.colors.GREEN, dim=True, err=True)
        self._phase_started = time.monotonic()

    def start_pack(self, total: int) -> None:
        self._phase_started = time.monotonic()
        typer.echo(f"  ·  打包工作目录（{total} 个文件）…", err=True)

    def pack_tick(self, n: int = 1) -> None:  # noqa: D401
        pass

    def start_upload(self, total_bytes: int) -> None:
        self._mark_phase_done("打包完成")
        typer.echo(f"  ·  上传到 Lab（{human_bytes(total_bytes)}）…", err=True)

    def upload_tick(self, n: int) -> None:
        pass

    def awaiting_server(self) -> None:
        self._mark_phase_done("上传完成")
        typer.echo("  ·  服务端受理（预检 / 配额 / Ray 提交）…", err=True)

    def finish(self) -> None:
        self._mark_phase_done("已受理")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _PipelineReporter:
    """Claude Code 风格垂直步骤条：已完成 ✓ + 当前 spinner + 右对齐耗时。"""

    _ORDER = ("pack", "upload", "server")

    def __init__(self, console) -> None:
        self._console = console
        self._live = None
        self._pack_total = 0
        self._pack_done = 0
        self._upload_total = 0
        self._upload_done = 0
        self._stages = {
            "pack": _StageState("pack", "打包工作目录"),
            "upload": _StageState("upload", "上传到 Lab"),
            "server": _StageState("server", "服务端受理"),
        }

    def _ensure_live(self) -> None:
        if self._live is not None:
            return
        from rich.live import Live

        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.__enter__()

    def _activate(self, key: str, *, detail: str = "") -> None:
        stage = self._stages[key]
        stage.status = "active"
        stage.started = time.monotonic()
        stage.finished = None
        stage.detail = detail
        self._refresh()

    def _complete(self, key: str, *, detail: str = "") -> None:
        stage = self._stages[key]
        stage.status = "done"
        if stage.started is None:
            stage.started = time.monotonic()
        stage.finished = time.monotonic()
        if detail:
            stage.detail = detail
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self):
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        table = Table(show_header=False, box=None, padding=(0, 1), expand=False, pad_edge=False)
        table.add_column(width=2, no_wrap=True)
        table.add_column(min_width=14, no_wrap=True)
        table.add_column(min_width=28)
        table.add_column(justify="right", min_width=8, no_wrap=True)

        table.add_row("", Text("lab submit", style="bold"), "", "")

        for key in self._ORDER:
            stage = self._stages[key]
            if stage.status == "pending":
                continue
            elapsed = format_elapsed(_stage_elapsed(stage))
            if stage.status == "done":
                table.add_row(
                    Text("✓", style="green"),
                    Text(stage.label, style="dim"),
                    Text(stage.detail, style="dim"),
                    Text(elapsed, style="dim"),
                )
            else:
                table.add_row(
                    Spinner("dots", style="cyan", speed=0.85),
                    Text(stage.label, style="bold cyan"),
                    Text(stage.detail),
                    Text(elapsed, style="bold cyan"),
                )
        return table

    def start_pack(self, total: int) -> None:
        self._ensure_live()
        self._pack_total = total
        self._pack_done = 0
        self._activate("pack", detail=f"0/{total} 文件")

    def pack_tick(self, n: int = 1) -> None:
        if self._stages["pack"].status != "active":
            return
        self._pack_done += n
        self._stages["pack"].detail = f"{self._pack_done}/{self._pack_total} 文件"
        self._refresh()

    def start_upload(self, total_bytes: int) -> None:
        self._complete("pack", detail=f"{self._pack_total}/{self._pack_total} 文件")
        self._upload_total = total_bytes
        self._upload_done = 0
        self._activate("upload", detail=f"0 B / {human_bytes(total_bytes)}")

    def upload_tick(self, n: int) -> None:
        if self._stages["upload"].status != "active":
            return
        self._upload_done += n
        elapsed = _stage_elapsed(self._stages["upload"])
        detail = f"{human_bytes(self._upload_done)} / {human_bytes(self._upload_total)}"
        if elapsed >= 0.2 and self._upload_done > 0:
            detail += f"  ·  {human_bytes(self._upload_done / elapsed)}/s"
        self._stages["upload"].detail = detail
        self._refresh()

    def awaiting_server(self) -> None:
        upload = self._stages["upload"]
        upload_detail = human_bytes(self._upload_total)
        elapsed = _stage_elapsed(upload)
        if elapsed >= 0.2 and self._upload_total > 0:
            upload_detail += f"  ·  {human_bytes(self._upload_total / elapsed)}/s"
        self._complete("upload", detail=upload_detail)
        self._activate("server", detail="预检 · 配额 · 提交")

    def finish(self) -> None:
        self._complete("server", detail="完成")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            return self._live.__exit__(*exc)
        return False


@contextmanager
def submit_progress():
    """提交进度条上下文：`with submit_progress() as reporter: submit_via_server(..., reporter=reporter)`。

    有 rich 且输出是 TTY 时用垂直步骤条；否则（CI / 管道 / 无 rich）降级为分阶段状态行。
    """
    reporter = _make_reporter()
    with reporter as r:
        yield r


def _make_reporter():
    if not sys.stderr.isatty():
        return _PlainReporter()
    try:
        from rich.console import Console
    except Exception:  # noqa: BLE001
        return _PlainReporter()
    console = Console(stderr=True, highlight=False, soft_wrap=False)
    return _PipelineReporter(console)
