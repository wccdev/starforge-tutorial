"""终端代理：拦截 stdout/stderr，批量经 IngestClient 上报（参照 SwanLab TerminalProxy）。"""
from __future__ import annotations

import os
import queue
import threading
from typing import Literal


class TerminalProxy:
    """捕获训练进程 stdout/stderr，异步批量 POST 到 console ingest /logs。"""

    def __init__(
        self,
        ingest,
        *,
        proxy_type: Literal["all", "stdout", "stderr", "none"] = "all",
        max_chunk_chars: int = 8192,
    ) -> None:
        self._ingest = ingest
        self._proxy_type = proxy_type
        self._max_chunk_chars = max_chunk_chars
        self._init_pid = os.getpid()
        self._stdout_q: queue.Queue[str] = queue.Queue(maxsize=50_000)
        self._stderr_q: queue.Queue[str] = queue.Queue(maxsize=50_000)
        self._stopped = threading.Event()
        self._installed = False
        self._stdout_capture = None
        self._stderr_capture = None
        self._worker: threading.Thread | None = None

    def install(self) -> None:
        if self._installed or self._proxy_type == "none":
            return
        from common.observability.terminal_capture import StreamCapture

        if self._proxy_type in ("all", "stdout"):
            self._stdout_capture = StreamCapture(
                "stdout", self._stdout_q.put_nowait, self._init_pid
            )
            self._stdout_capture.install()
        if self._proxy_type in ("all", "stderr"):
            self._stderr_capture = StreamCapture(
                "stderr", self._stderr_q.put_nowait, self._init_pid
            )
            self._stderr_capture.install()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="NeMoLab·Terminal"
        )
        self._worker.start()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._stdout_capture:
            self._stdout_capture.uninstall()
            self._stdout_capture = None
        if self._stderr_capture:
            self._stderr_capture.uninstall()
            self._stderr_capture = None
        self._stopped.set()
        if self._worker:
            self._worker.join(timeout=5)
            self._worker = None
        self._drain_and_flush(final=True)
        self._installed = False

    def _worker_loop(self) -> None:
        while not self._stopped.is_set() or not self._stdout_q.empty() or not self._stderr_q.empty():
            self._drain_and_flush(final=False)
            if self._stopped.is_set() and self._stdout_q.empty() and self._stderr_q.empty():
                break
            self._stopped.wait(0.25)

    def _drain_and_flush(self, *, final: bool) -> None:
        from common.observability.log_format import format_log_chunk_for_ingest

        buf: list[str] = []
        for q in (self._stdout_q, self._stderr_q):
            while True:
                try:
                    buf.append(q.get_nowait())
                except queue.Empty:
                    break
        if not buf:
            if final:
                self._ingest.enqueue_log_eof()
            return
        text = format_log_chunk_for_ingest("".join(buf))
        try:
            from common.observability.validation_ctx import feed_log_text

            feed_log_text(text)
        except Exception:
            pass
        if len(text) > self._max_chunk_chars:
            for i in range(0, len(text), self._max_chunk_chars):
                self._ingest.enqueue_log(text[i : i + self._max_chunk_chars])
        else:
            self._ingest.enqueue_log(text)
        if final:
            self._ingest.enqueue_log_eof()
