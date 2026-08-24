"""终端代理：拦截 stdout/stderr，供验证样本解析；必要时兜底上报日志。

两个用途要分清
──────────────────────────────────────────────────────────────────────────────
1. 验证样本要从日志里认出「Starting validation at step N」—— 始终要做；
2. 日志上报只是**兜底**：单机路径由平台侧（starforge_core 的 LogForwarder）
   转发，它设 STARFORGE_LOG_FORWARD=1，本类据此让路 —— 两个生产端同时上报
   就是双份日志。只有平台没有转发的路径（当前是多机）才由这里上报。

日志的**文本格式不在这里管**：规范化由平台侧 ingest 统一做（平台契约不该由
作业包决定，否则换个框架、或实验删掉本目录就会静默改变平台行为）。
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Literal

#: 由 starforge_core.log_forward.LOG_FORWARD_ENV 设置，取值 "1" 表示平台已转发。
LOG_FORWARD_ENV = "STARFORGE_LOG_FORWARD"


class TerminalProxy:
    """捕获训练进程 stdout/stderr；平台未转发日志时兜底 POST 到 ingest /logs。"""

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
        self._upload_logs = os.environ.get(LOG_FORWARD_ENV, "").strip() != "1"
        # 上次没凑够一整行的尾巴，留到下次一起发（见 _drain_and_flush）。
        self._tail = ""
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
            target=self._worker_loop, daemon=True, name="StarForge·Terminal"
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
            self._drain_and_flush()
            if self._stopped.is_set() and self._stdout_q.empty() and self._stderr_q.empty():
                break
            self._stopped.wait(0.25)

    def _drain_and_flush(self, *, final: bool = False) -> None:
        buf: list[str] = []
        for q in (self._stdout_q, self._stderr_q):
            while True:
                try:
                    buf.append(q.get_nowait())
                except queue.Empty:
                    break
        text = "".join(buf)
        if text:
            try:
                from common.observability.validation_ctx import feed_log_text

                feed_log_text(text)
            except Exception:
                pass
        # eof 不由这里发：容器被 SIGKILL（OOM 等）时本进程没机会发，平台在作业
        # 进入终态时统一补，那才是可靠的归属。
        if not self._upload_logs:
            return
        payload, self._tail = self._split_on_line_boundary(self._tail + text, final=final)
        if not payload:
            return
        if len(payload) > self._max_chunk_chars:
            for i in range(0, len(payload), self._max_chunk_chars):
                self._ingest.enqueue_log(payload[i : i + self._max_chunk_chars])
        else:
            self._ingest.enqueue_log(payload)

    def _split_on_line_boundary(self, text: str, *, final: bool) -> tuple[str, str]:
        """切到最后一个行结束符，返回 (要上报的, 留着的尾巴)。

        跨请求切在半行上会让平台把时间戳补到一行中间（平台按行规范化，它无法
        知道下一个请求是上一行的续写）。进度条只写 `\\r`，所以它也算行结束符。
        收尾时（final）与尾巴长到不像一行时不再等，直接发出去。
        """
        if final or len(text) >= self._max_chunk_chars:
            return text, ""
        cut = max(text.rfind("\n"), text.rfind("\r")) + 1
        return text[:cut], text[cut:]
