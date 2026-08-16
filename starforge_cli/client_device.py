"""CLI 登录设备信息：终端采集 + 稳定设备 ID（哈希，不上传原始 UUID）。

业界 CLI/OAuth 场景常见做法（GitHub CLI、gcloud、AWS SSO 等）：
  - 客户端自报 hostname / OS / 用户（可读标签）
  - 读取 OS 级稳定标识（machine-id / Platform UUID / MachineGuid）后 **只上传 SHA256 短哈希**
  - 不采集浏览器式 canvas/WebGL 指纹；不做跨站追踪

与服务端 server/core/client_device.py 的字段契约保持一致。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Optional


def collect_cli_device() -> dict[str, str]:
    """从当前终端环境采集设备信息（sf login 打开浏览器前调用）。"""
    hostname = (socket.gethostname() or platform.node() or "").strip()[:64]
    shell_path = (os.environ.get("SHELL") or "").strip()
    shell = Path(shell_path).name[:32] if shell_path else ""

    info: dict[str, str] = {
        "hostname": hostname or "unknown",
        "os": (platform.system() or "")[:32],
        "os_release": (platform.release() or "")[:32],
        "machine": (platform.machine() or "")[:32],
        "platform": (platform.platform(terse=True) or "")[:120],
        "user": (os.environ.get("USER") or os.environ.get("USERNAME") or "")[:64],
        "terminal": (os.environ.get("TERM") or "")[:32],
        "shell": shell,
        "lab_version": _lab_version()[:32],
        "source": "lab-cli",
    }
    device_id = _device_id_hash(_stable_machine_id(), fallback=hostname)
    if device_id:
        info["device_id"] = device_id
    return info


def encode_device_param(info: dict[str, Any]) -> str:
    raw = json.dumps(info, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _lab_version() -> str:
    try:
        from importlib.metadata import version

        return version("starforge")
    except Exception:  # noqa: BLE001
        return "unknown"


def _stable_machine_id() -> Optional[str]:
    """读取 OS 原生稳定标识（原始值仅本地使用，不上传）。"""
    system = platform.system()
    try:
        if system == "Darwin":
            proc = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            for line in proc.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    parts = line.split('"')
                    if len(parts) >= 2:
                        val = parts[-2].strip()
                        return val or None
        elif system == "Linux":
            for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
                if path.is_file():
                    val = path.read_text(encoding="utf-8", errors="ignore").strip()
                    if val:
                        return val
        elif system == "Windows":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as key:
                val, _ = winreg.QueryValueEx(key, "MachineGuid")
                if val:
                    return str(val).strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def _device_id_hash(stable_id: Optional[str], *, fallback: str) -> Optional[str]:
    raw = (stable_id or "").strip() or (fallback or "").strip()
    if not raw:
        return None
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
