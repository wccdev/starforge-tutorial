"""stdout/stderr write 拦截。

Passthrough-first：先写原始流，再入队上报；重入保护 + fork 安全。
"""
from __future__ import annotations

import contextvars
import os
import sys
from typing import Callable, Literal, cast

_in_callback: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_starforge_console_in_callback",
    default=False,
)


class StreamCapture:
    def __init__(
        self,
        stream_name: Literal["stdout", "stderr"],
        on_write: Callable[[str], None],
        init_pid: int,
    ) -> None:
        self._stream_name = stream_name
        self._on_write = on_write
        self._init_pid = init_pid
        self._original_write: Callable | None = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        stream = getattr(sys, self._stream_name)
        self._original_write = stream.write
        stream.write = self._make_wrapper()  # type: ignore[method-assign]
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        stream = getattr(sys, self._stream_name)
        stream.write = self._original_write  # type: ignore[method-assign]
        self._original_write = None
        self._installed = False

    def _make_wrapper(self):
        original_write = cast(Callable[[str], int], self._original_write)
        on_write = self._on_write
        init_pid = self._init_pid

        def write_wrapper(data) -> int:
            n = original_write(data)
            if os.getpid() != init_pid:
                return n
            if _in_callback.get():
                return n
            _in_callback.set(True)
            try:
                if isinstance(data, bytes):
                    text = data[:n].decode("utf-8", errors="replace")
                else:
                    text = data[:n]
                on_write(text)
            except Exception:
                pass
            finally:
                _in_callback.set(False)
            return n

        return write_wrapper
