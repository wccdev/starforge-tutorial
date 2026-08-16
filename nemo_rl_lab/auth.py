"""登录 / 凭据 / 命令门控（客户端核心，不含任何业务 API）。

仅用标准库（http.server / urllib / webbrowser）+ typer，不依赖 web extra，
保证未装 fastapi 的纯客户端也能 `lab login`。

本地状态：
  ~/.lab/config.json       {"server": "https://nemolab.gcoreinc.com"}
  ~/.lab/credentials.json  {"<server>": {access_token, refresh_token, expires_at, user}}
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import typer

from nemo_rl_lab import cli_ui

# 官方中心化 Lab 服务（lab login 默认；未配置时 CLI 亦指向此地址）
DEFAULT_LAB_SERVER = "https://nemolab.gcoreinc.com"

MSG_NOT_LOGGED_IN = "请先运行 lab login"

LAB_DIR = Path(os.environ.get("LAB_HOME") or (Path.home() / ".lab"))
CONFIG_PATH = LAB_DIR / "config.json"
CRED_PATH = LAB_DIR / "credentials.json"


# ----------------------------- PKCE（stdlib）-----------------------------
def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


# ----------------------------- 本地配置/凭据 -----------------------------
def _read_json(path: Path) -> dict:
    """读本地状态文件；文件损坏直接报错，不静默当空处理。"""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        cli_ui.fail(
            f"本地状态文件损坏（非法 JSON）: {path}",
            hint=f"删除该文件后重新 lab login：rm {path}",
        )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # 凭据文件必须在创建瞬间就是 0600：先 write_text 再 chmod 会留下一个
    # 按默认 umask（常为 0644，世界可读）写入完整 token 的窗口。
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
    try:
        path.chmod(0o600)  # 收紧历史版本以 0644 创建的旧文件
    except OSError as e:
        print(f"[lab] 警告：无法收紧本地状态文件权限 {path}: {e}", file=sys.stderr)


def current_server(explicit: Optional[str] = None) -> Optional[str]:
    """server 地址优先级：显式 > 环境 LAB_SERVER > config.json > 官方默认。"""
    s = (
        explicit
        or os.environ.get("LAB_SERVER")
        or _read_json(CONFIG_PATH).get("server")
        or DEFAULT_LAB_SERVER
    )
    return s.rstrip("/") if s else None


def is_server_mode() -> bool:
    return current_server() is not None


def _save_server(server: str) -> None:
    cfg = _read_json(CONFIG_PATH)
    cfg["server"] = server
    _write_json(CONFIG_PATH, cfg)


def _save_creds(server: str, creds: dict) -> None:
    all_creds = _read_json(CRED_PATH)
    all_creds[server] = creds
    _write_json(CRED_PATH, all_creds)


def _load_creds(server: str) -> Optional[dict]:
    return _read_json(CRED_PATH).get(server)


def _clear_creds(server: str) -> None:
    all_creds = _read_json(CRED_PATH)
    if server in all_creds:
        del all_creds[server]
        _write_json(CRED_PATH, all_creds)


# ----------------------------- HTTP（stdlib）-----------------------------
def _api(server: str, method: str, path: str, *, token: Optional[str] = None, body: Optional[dict] = None,
         timeout: float = 10.0) -> dict:
    url = f"{server}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _http_json(server: str, method: str, path: str, *, body: Optional[dict] = None, timeout: float = 10.0) -> tuple[int, dict]:
    """HTTP 请求并返回 (status, json)；不抛 HTTPError，便于轮询 pending。"""
    url = f"{server}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw.decode(errors="ignore") or e.reason}
        return e.code, payload


# ----------------------------- token 生命周期 -----------------------------
def _refresh(server: str, creds: dict) -> Optional[dict]:
    rt = creds.get("refresh_token")
    if not rt:
        return None
    # 限流（429）视为瞬时错误，短退避重试，避免误判为凭据失效而强制重登。
    for attempt in range(3):
        try:
            resp = _api(server, "POST", "/api/auth/refresh", body={"refresh_token": rt})
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        creds["access_token"] = resp["access_token"]
        # 服务端轮转 refresh 时回传新 token，必须持久化，否则下次续期会用已吊销的旧 token 失败。
        if resp.get("refresh_token"):
            creds["refresh_token"] = resp["refresh_token"]
        creds["expires_at"] = time.time() + resp.get("expires_in", 3600) - 60
        _save_creds(server, creds)
        return creds
    return None


def get_access_token(server: str, *, auto_refresh: bool = True) -> Optional[str]:
    """返回有效 access token；过期则用 refresh 续期；都不行返回 None。"""
    creds = _load_creds(server)
    if not creds:
        return None
    exp = creds.get("expires_at")
    if exp is None or time.time() < exp:
        return creds.get("access_token")
    if auto_refresh:
        refreshed = _refresh(server, creds)
        if refreshed:
            return refreshed["access_token"]
    return None


# ----------------------------- 命令门控 -----------------------------
def gate() -> None:
    """集群类命令执行前的登录门槛：未登录直接报错，不隐式发起登录流程。"""
    server = current_server()
    if not get_access_token(server):
        cli_ui.fail(MSG_NOT_LOGGED_IN, hint="运行 lab login 登录")


# ----------------------------- 环境检测 / 设备码登录 -----------------------------
def prefer_device_flow(*, force: bool = False, no_browser: bool = False) -> bool:
    """SSH / 无图形环境优先走 RFC 8628 设备码流程。"""
    if force or no_browser:
        return True
    if os.environ.get("LAB_DEVICE_FLOW", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return True
    return False


def _device_login(server: str, timeout: float = 900.0) -> dict:
    from nemo_rl_lab.client_device import collect_cli_device, encode_device_param

    device = encode_device_param(collect_cli_device())
    status, resp = _http_json(server, "POST", "/api/cli/device/code", body={"device": device})
    if status != 200:
        detail = resp.get("detail", resp)
        cli_ui.fail(f"无法启动登录：{detail}")

    device_code = resp["device_code"]
    user_code = resp["user_code"]
    verification_uri = resp.get("verification_uri_complete") or resp.get("verification_uri", f"{server}/cli/device")
    interval = int(resp.get("interval", 5))
    expires_at = time.time() + float(resp.get("expires_in", timeout))

    typer.echo("")
    typer.secho("请用浏览器完成登录：", fg=typer.colors.YELLOW)
    typer.echo(f"  打开 {verification_uri}")
    typer.secho(f"  验证码：{user_code}", fg=typer.colors.CYAN, bold=True)
    typer.echo("")

    if not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")):
        try:
            webbrowser.open(verification_uri)
        except Exception:
            pass

    while time.time() < expires_at:
        time.sleep(interval)
        status, tok = _http_json(server, "POST", "/api/cli/device/token", body={"device_code": device_code})
        if status == 200:
            return {
                "access_token": tok["access_token"],
                "refresh_token": tok.get("refresh_token"),
                "expires_at": time.time() + tok.get("expires_in", 3600) - 60,
                "user": tok.get("user"),
            }
        detail = tok.get("detail", "")
        if detail == "authorization_pending":
            continue
        if detail == "slow_down" or status == 429:  # 限流：放慢轮询而非中止授权
            interval = min(interval + 5, 60)
            continue
        cli_ui.fail(f"登录失败：{detail or status}")

    cli_ui.fail("登录超时，请重试。")


def _interactive_login(server: str, *, device_flow: bool = False, no_browser: bool = False) -> dict:
    if prefer_device_flow(force=device_flow, no_browser=no_browser):
        return _device_login(server)
    return _browser_login(server)


# ----------------------------- 回环登录流 -----------------------------
class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict = {}
    success_redirect: str = ""

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        qs = urllib.parse.parse_qs(parsed.query)
        type(self).result = {
            "code": (qs.get("code") or [None])[0],
            "state": (qs.get("state") or [None])[0],
            "error": (qs.get("error") or [None])[0],
        }
        redirect = type(self).success_redirect
        if redirect and not type(self).result.get("error"):
            self.send_response(302)
            self.send_header("Location", redirect)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            "<html><body style='font-family:sans-serif;text-align:center;margin-top:80px'>"
            "<h2>授权失败</h2><p>请关闭此页面并在终端重试 lab login。</p>"
            "</body></html>".encode()
        )

    def log_message(self, *a):  # 静默
        pass


def _browser_login(server: str, timeout: float = 180.0) -> dict:
    from nemo_rl_lab.client_device import collect_cli_device, encode_device_param

    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    httpd = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
    port = httpd.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    _CallbackHandler.result = {}
    _CallbackHandler.success_redirect = f"{server.rstrip('/')}/cli/success"

    device = encode_device_param(collect_cli_device())
    q = urllib.parse.urlencode(
        {"redirect_uri": redirect_uri, "state": state, "challenge": challenge, "device": device},
    )
    auth_url = f"{server}/cli/authorize?{q}"
    typer.echo("正在打开浏览器…")
    webbrowser.open(auth_url)

    deadline = time.time() + timeout

    def _serve():
        while not _CallbackHandler.result and time.time() < deadline:
            httpd.handle_request()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    t.join(timeout)
    httpd.server_close()

    res = _CallbackHandler.result
    if not res:
        cli_ui.fail("登录超时，请重试。")
    if res.get("error"):
        cli_ui.fail(f"登录失败：{res['error']}")
    if res.get("state") != state:
        cli_ui.fail("登录校验失败，请重试。")

    resp = _api(
        server, "POST", "/api/cli/token",
        body={
            "code": res["code"],
            "verifier": verifier,
            "redirect_uri": redirect_uri,
            "device": device,
        },
    )
    return {
        "access_token": resp["access_token"],
        "refresh_token": resp.get("refresh_token"),
        "expires_at": time.time() + resp.get("expires_in", 3600) - 60,
        "user": resp.get("user"),
    }
