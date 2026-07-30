"""仿 SwanLab Transport：批量 HTTP 上报到 console ingest API。"""
from __future__ import annotations

import json
import os
import threading
import time
from queue import Empty, Queue


class IngestClient:
    def __init__(
        self,
        endpoint: str,
        run_id: str,
        token: str,
        *,
        flush_interval: float = 1.5,
        batch_size: int = 256,
        fallback_path: str | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.run_id = run_id
        self.token = token
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.fallback_path = fallback_path
        self._metric_q: Queue[dict] = Queue(maxsize=100_000)
        self._hardware_q: Queue[dict] = Queue(maxsize=100_000)
        self._log_q: Queue[str] = Queue(maxsize=100_000)
        self._log_eof_pending = False
        self._hparams_pending: dict | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="NeMoLab·Transport"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.flush_interval * 3)
        self.flush()

    def enqueue_metrics(self, points: list[dict]) -> None:
        for p in points:
            try:
                self._metric_q.put_nowait(p)
            except Exception:
                pass

    def enqueue_hardware(self, points: list[dict]) -> None:
        for p in points:
            try:
                self._hardware_q.put_nowait(p)
            except Exception:
                pass

    def enqueue_hparams(self, params: dict) -> None:
        with self._lock:
            self._hparams_pending = dict(params)

    def enqueue_environment(self, payload: dict) -> None:
        try:
            self._post("environment", {"run_id": self.run_id, **payload})
        except Exception:
            pass

    def send_environment_nodes(self, nodes: list[dict]) -> bool:
        """上报作业各节点的静态硬件。返回是否送达，失败由调用方决定是否重试。

        这里不走队列：它一个作业只发一次，排队反而要多维护一条重投路径。
        """
        try:
            self._post("environment/nodes", {"run_id": self.run_id, "nodes": nodes})
            return True
        except Exception as e:
            print(f"NeMoLab environment nodes upload failed: {e}")
            return False

    def enqueue_log(self, chunk: str) -> None:
        if not chunk:
            return
        try:
            self._log_q.put_nowait(chunk)
        except Exception:
            pass

    def enqueue_log_eof(self) -> None:
        with self._lock:
            self._log_eof_pending = True

    def enqueue_validation(self, payload: dict) -> None:
        """整轮验证样本 + 元数据，单次 POST。"""
        try:
            self._post("validation", payload)
        except Exception:
            pass

    def flush(self) -> None:
        self._flush_metrics()
        self._flush_hardware()
        self._flush_hparams()
        self._flush_logs()

    def _loop(self) -> None:
        while self._running:
            try:
                self.flush()
            except Exception as e:
                print(f"NeMoLab IngestClient flush error: {e}")
            time.sleep(self.flush_interval)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict) -> None:
        import requests

        url = f"{self.endpoint}/{path.lstrip('/')}"
        try:
            resp = requests.post(
                url, json=payload, headers=self._headers(), timeout=15
            )
            resp.raise_for_status()
        except Exception as e:
            if self.fallback_path:
                os.makedirs(os.path.dirname(self.fallback_path), exist_ok=True)
                with open(self.fallback_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"path": path, "payload": payload, "error": str(e)}
                        )
                        + "\n"
                    )
            raise

    def _drain(self, q: Queue, limit: int) -> list[dict]:
        out: list[dict] = []
        for _ in range(limit):
            try:
                out.append(q.get_nowait())
            except Empty:
                break
        return out

    def _flush_metrics(self) -> None:
        points = self._drain(self._metric_q, self.batch_size)
        if not points:
            return
        try:
            self._post("metrics", {"run_id": self.run_id, "points": points})
        except Exception:
            for p in points:
                try:
                    self._metric_q.put_nowait(p)
                except Exception:
                    pass

    def _flush_hardware(self) -> None:
        points = self._drain(self._hardware_q, self.batch_size)
        if not points:
            return
        try:
            self._post("hardware", {"run_id": self.run_id, "points": points})
        except Exception:
            for p in points:
                try:
                    self._hardware_q.put_nowait(p)
                except Exception:
                    pass

    def _flush_hparams(self) -> None:
        with self._lock:
            params = self._hparams_pending
            self._hparams_pending = None
        if not params:
            return
        try:
            self._post("hparams", {"run_id": self.run_id, "params": params})
        except Exception:
            with self._lock:
                self._hparams_pending = params

    def _drain_logs(self, limit: int) -> list[str]:
        out: list[str] = []
        for _ in range(limit):
            try:
                out.append(self._log_q.get_nowait())
            except Empty:
                break
        return out

    def _flush_logs(self) -> None:
        chunks = self._drain_logs(self.batch_size)
        with self._lock:
            eof = self._log_eof_pending
        if not chunks and not eof:
            return
        payload: dict = {"run_id": self.run_id, "chunks": chunks}
        if eof:
            payload["eof"] = True
        try:
            self._post("logs", payload)
            if eof:
                with self._lock:
                    self._log_eof_pending = False
        except Exception:
            for c in chunks:
                try:
                    self._log_q.put_nowait(c)
                except Exception:
                    pass
            if eof:
                with self._lock:
                    self._log_eof_pending = True
