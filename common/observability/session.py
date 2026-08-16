"""StarForge 采集会话：终端日志 + 指标/硬件共用一个 IngestClient。"""
from __future__ import annotations

import atexit
import os

_session: "ObservabilitySession | None" = None


class ObservabilitySession:
    def __init__(self, ingest, terminal) -> None:
        self.ingest = ingest
        self.terminal = terminal


def start_observability() -> ObservabilitySession | None:
    """有 STARFORGE_TOKEN 时启动终端捕获 + 传输；本地直跑为 no-op。"""
    global _session
    if _session is not None:
        return _session
    if not os.environ.get("STARFORGE_TOKEN"):
        return None

    endpoint = os.environ.get("STARFORGE_ENDPOINT", "")
    run_id = os.environ.get("STARFORGE_RUN_ID") or os.environ.get("NRL_RUN_ID", "")
    token = os.environ.get("STARFORGE_TOKEN", "")
    if not endpoint or not run_id or not token:
        return None

    from common.observability.ingest_client import IngestClient
    from common.observability.terminal_proxy import TerminalProxy

    flush_interval = float(os.environ.get("STARFORGE_FLUSH_INTERVAL", "1.5"))
    ingest = IngestClient(endpoint, run_id, token, flush_interval=flush_interval)
    ingest.start()
    try:
        from common.observability.env_probe import collect_environment

        ingest.enqueue_environment(collect_environment())
    except Exception as e:
        print(f"StarForge environment probe skipped: {e}")
    terminal = TerminalProxy(ingest)
    terminal.install()
    _session = ObservabilitySession(ingest, terminal)
    atexit.register(stop_observability)
    print(f"StarForge observability started for run {run_id}")
    return _session


def stop_observability() -> None:
    global _session
    if _session is None:
        return
    try:
        if _session.terminal:
            _session.terminal.uninstall()
        _session.ingest.stop()
    except Exception:
        pass
    _session = None


def get_ingest():
    return _session.ingest if _session else None
