"""中心化 Lab Server 的 CLI 侧：登录 / 凭据管理 / 命令门控。

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
import random
import secrets
import subprocess
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

# 需要登录才能用的集群类命令（server 模式下门控）；纯本地命令不在此列。
GATED_COMMANDS = {
    "submit", "export", "eval", "status", "logs", "clean",
    "job-list", "job-logs", "job-samples", "job-status",
    "job-stop", "job-delete", "job-clean", "job-cancel-all",
}


# ----------------------------- PKCE（stdlib）-----------------------------
def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


# ----------------------------- 本地配置/凭据 -----------------------------
def _read_json(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    try:
        path.chmod(0o600)
    except OSError:
        pass


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


# ----------------------------- 代理提交（Phase B，客户端侧）-----------------------------
def _git_out(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_provenance(repo_root: Path, exp_rel: str) -> dict:
    """提交可追溯：git commit / dirty / config 指纹。"""
    commit = (_git_out(["rev-parse", "--short", "HEAD"], repo_root) or "unknown").strip()
    dirty = bool(_git_out(["status", "--porcelain"], repo_root).strip())
    cfg = repo_root / exp_rel / "config.yaml"
    if cfg.is_file():
        config_sha = hashlib.sha256(cfg.read_bytes()).hexdigest()[:12]
    else:
        config_sha = "none"
    return {"git_commit": commit, "git_dirty": dirty, "config_sha": config_sha}


def pack_working_dir(repo_root: Path) -> bytes:
    """打包 working-dir 为 tar.gz：用 git 跟踪 + 未忽略文件（精确遵循 .gitignore）。"""
    import io
    import tarfile

    listing = _git_out(["ls-files", "--cached", "--others", "--exclude-standard"], repo_root)
    files = [f for f in listing.splitlines() if f.strip()]
    if not files:
        cli_ui.fail("请在 git 仓库目录内运行，且存在可上传的文件。")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in files:
            p = repo_root / rel
            if p.is_file():  # 跳过已删除/软链异常
                tar.add(p, arcname=rel)
    return buf.getvalue()


def _bearer_request(server: str, method: str, path: str, *, data: Optional[bytes] = None,
                    headers: Optional[dict] = None, timeout: float = 60.0):
    """带 token 的请求；返回 urlopen 的响应对象（调用方负责读取/关闭）。"""
    token = get_access_token(server)
    if not token:
        cli_ui.fail(MSG_NOT_LOGGED_IN, hint="运行 lab login 登录")
    h = {"Authorization": f"Bearer {token}"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(f"{server}{path}", data=data, headers=h, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def submit_via_server(exp_rel: str, profile: Optional[str], repo_root: Path,
                      server: Optional[str] = None, project: Optional[str] = None) -> dict:
    """server 模式提交：打包上传 + 服务端注入密钥后代理提交，返回 {job_id, run_id, ...}。"""
    srv = current_server(server)
    meta = {"exp": exp_rel, "profile": profile or "", **git_provenance(repo_root, exp_rel)}
    if project:
        meta["project"] = project
    blob = pack_working_dir(repo_root)
    headers = {"Content-Type": "application/gzip", "X-Lab-Meta": json.dumps(meta, ensure_ascii=False)}
    try:
        with _bearer_request(srv, "POST", "/api/jobs", data=blob, headers=headers, timeout=300.0) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="提交失败，请稍后重试。")


def submit_post_via_server(action: str, exp_rel: str, profile: Optional[str], flags: list[str],
                           repo_root: Path, server: Optional[str] = None) -> dict:
    """server 模式训练后闭环（export/eval）：打包上传 → 服务端代理提交 post_train.sh。"""
    srv = current_server(server)
    meta = {"action": action, "exp": exp_rel, "profile": profile or "",
            "flags": flags, **git_provenance(repo_root, exp_rel)}
    blob = pack_working_dir(repo_root)
    headers = {"Content-Type": "application/gzip", "X-Lab-Meta": json.dumps(meta, ensure_ascii=False)}
    try:
        with _bearer_request(srv, "POST", "/api/post", data=blob, headers=headers, timeout=300.0) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        label = "导出" if action == "export" else "评测"
        cli_ui.fail_http(e, fallback=f"{label}提交失败，请稍后重试。")


def clean_via_server(exp_rel: str, server: Optional[str] = None) -> dict:
    """server 模式：清理本实验在集群上的产物目录（checkpoint/日志），经服务端在集群侧删除。"""
    srv = current_server(server)
    path = f"/api/clean?exp={urllib.parse.quote(exp_rel)}"
    try:
        with _bearer_request(srv, "POST", path) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="清理失败，请稍后重试。")


def usage_via_server(server: Optional[str] = None) -> dict:
    """取本人配额 + 实时用量。"""
    srv = current_server(server)
    with _bearer_request(srv, "GET", "/api/usage/mine") as r:
        return json.loads(r.read() or b"{}")


def whoami_via_server(server: Optional[str] = None) -> dict:
    """取当前登录身份 + 配额。"""
    srv = current_server(server)
    token = get_access_token(srv)
    if not token:
        cli_ui.fail(MSG_NOT_LOGGED_IN, hint="运行 lab login 登录")
    return _api(srv, "GET", "/api/whoami", token=token)


def list_my_jobs(server: Optional[str] = None, limit: int = 50) -> list[dict]:
    """获取作业列表。"""
    srv = current_server(server)
    with _bearer_request(srv, "GET", f"/api/jobs/mine?limit={limit}") as r:
        return (json.loads(r.read() or b"{}")).get("jobs", [])


def job_control_via_server(action: str, job_id: str, server: Optional[str] = None) -> dict:
    """停止 / 删除作业。"""
    srv = current_server(server)
    path = f"/api/job/{action}?id={urllib.parse.quote(job_id)}"
    labels = {"stop": "停止作业", "delete": "删除记录"}
    try:
        with _bearer_request(srv, "POST", path) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback=f"{labels.get(action, '操作')}失败。")


def latest_job_via_server(server: Optional[str] = None) -> Optional[str]:
    srv = current_server(server)
    if not srv:
        return None
    try:
        with _bearer_request(srv, "GET", "/api/jobs/mine?limit=1") as r:
            jobs = (json.loads(r.read() or b"{}")).get("jobs", [])
    except urllib.error.HTTPError:
        return None
    return jobs[0].get("ray_submission_id") if jobs else None


def iter_sse_events(lines):
    """把 SSE 字节/字符行流解析为 (event, event_id, data) 事件序列。

    服务端按 SSE 协议发帧：`event:` / `id:` / `data:`，多行内容拆成多条 `data:` 行，
    `: keepalive` 为注释心跳。规范要求：空行分发一个事件、data 多行以 \n 拼回、
    冒号后仅去掉一个前导空格（保留日志缩进）。无 event 字段默认 "message"。

    event_id 遵循 WHATWG EventSource 的“粘性 last event id”语义：仅在出现新的
    `id:` 字段时更新，并跨事件保留——用于断线续传（Last-Event-ID / ?from=）。
    """
    event = "message"
    event_id: Optional[str] = None
    data_lines: list[str] = []
    for raw in lines:
        line = raw.decode(errors="ignore") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.rstrip("\r\n")
        if line == "":  # 空行 = 事件结束
            if data_lines:
                yield event, event_id, "\n".join(data_lines)
            event, data_lines = "message", []  # event_id 不重置（粘性）
            continue
        if line.startswith(":"):  # 注释（keepalive），忽略
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):  # 仅去一个前导空格
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
        elif field == "id":
            event_id = value
    if data_lines:  # 末尾无空行兜底
        yield event, event_id, "\n".join(data_lines)


def parse_sse_stream(lines):
    """向后兼容包装：仅产出 (event, data)，丢弃 id（历史调用方/测试契约）。"""
    for event, _event_id, data in iter_sse_events(lines):
        yield event, data


_STREAM_BACKOFF_MAX = 30.0


def stream_logs_via_server(job_id: str, server: Optional[str] = None,
                           tail: Optional[int] = None) -> None:
    """经服务端 SSE 接口跟随作业日志（客户端不直连 Ray）。

    只把 log 事件原文还原后打到 stdout，不暴露 event:/id:/data:/keepalive 等协议噪音。
    tail 给定时只回放最后 N 行历史日志再跟随（默认 2000；0 或 None=全量）。

    健壮性：服务端为多副本 Redis Streams 推送，长连接可能被反代/实例切换回收。
    本函数在连接非正常结束（未收到 end 事件）时按指数退避自动重连，并携带
    Last-Event-ID 头 + ?from=<id> 从断点续传，避免日志丢失或重复回放历史。
    作业到达终态时服务端发 end 事件，收到后干净退出（不再重连）。
    """
    srv = current_server(server)

    last_id: Optional[str] = None
    backoff = 1.0
    ended = False
    try:
        while not ended:
            q = {"id": job_id}
            if last_id is not None:  # 续传：从断点之后继续，不重复回放 tail
                q["from"] = last_id
            elif tail is not None:
                q["tail"] = str(tail)
            path = f"/api/job/logs/stream?{urllib.parse.urlencode(q)}"
            headers = {"Last-Event-ID": last_id} if last_id is not None else None
            try:
                with _bearer_request(srv, "GET", path, headers=headers, timeout=None) as r:
                    backoff = 1.0  # 连上即重置退避
                    for event, eid, data in iter_sse_events(r):  # urllib 响应按行迭代
                        if eid is not None:
                            last_id = eid  # 记录断点续传位置
                        if event == "log":
                            sys.stdout.write(data)  # data 已按 \n 还原，含原始换行
                            sys.stdout.flush()
                        elif event == "error":
                            cli_ui.emit_error("日志流异常", body=data)
                        elif event == "end":
                            ended = True
                            break
                        # open / 其它事件：静默忽略
            except urllib.error.HTTPError as e:
                # 4xx（限流 429 除外）通常不可恢复：作业不存在 / 无权限 / 鉴权失效。
                if e.code != 429 and 400 <= e.code < 500:
                    cli_ui.fail_http(e, fallback="无法读取日志。")
                # 429 限流 / 5xx：退避后重连。
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                # 网络中断 / 连接被回收：退避后用 from=last_id 续传。
                pass

            if ended:
                break
            # 抖动退避，避免实例重启时的重连风暴。
            time.sleep(backoff + random.uniform(0, backoff * 0.25))
            backoff = min(backoff * 2, _STREAM_BACKOFF_MAX)
    except KeyboardInterrupt:
        typer.echo("\n已停止跟随。")


def job_overview_via_server(job_id: str, server: Optional[str] = None) -> dict:
    """取作业概览（含 validations 列表）。"""
    srv = current_server(server)
    path = f"/api/job?id={urllib.parse.quote(job_id)}"
    try:
        with _bearer_request(srv, "GET", path) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="无法获取作业信息。")


def samples_via_server(job_id: str, vidx: int, offset: int = 0, limit: int = 6,
                       server: Optional[str] = None) -> dict:
    """取某次验证的多轮对话样本（分页）。"""
    srv = current_server(server)
    q = urllib.parse.urlencode({"id": job_id, "vidx": vidx, "offset": offset, "limit": limit})
    try:
        with _bearer_request(srv, "GET", f"/api/samples?{q}") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="无法获取验证样本。")


def cluster_status_via_server(server: Optional[str] = None) -> Optional[dict]:
    """取集群 GPU 概览 + 活跃作业；Ray/服务不可达时返回 None（status 预检用）。"""
    srv = current_server(server)
    if not srv:
        return None
    try:
        with _bearer_request(srv, "GET", "/api/cluster/status") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError:
        return None


def batch_via_server(action: str, server: Optional[str] = None) -> dict:
    """批量作业控制：cancel-all / clean。"""
    srv = current_server(server)
    try:
        with _bearer_request(srv, "POST", f"/api/jobs/{action}") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="操作失败，请稍后重试。")


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


# ----------------------------- 命令门控 -----------------------------
def gate(command: str) -> None:
    """集群类命令执行前：未登录则自动发起登录。"""
    server = current_server()
    if command not in GATED_COMMANDS:
        return
    token = get_access_token(server)
    if token:
        return
    if prefer_device_flow():
        typer.secho("正在登录…", fg=typer.colors.YELLOW)
    else:
        typer.secho("正在打开浏览器登录…", fg=typer.colors.YELLOW)
    creds = _interactive_login(server)
    _save_creds(server, creds)
    u = (creds.get("user") or {}).get("username", "?")
    typer.secho(f"✓ 已登录：{u}", fg=typer.colors.GREEN)


# ----------------------------- Typer 命令 -----------------------------
def login(
    server: Optional[str] = typer.Option(
        None, "--server", "-s",
        help=f"Lab 服务地址（默认 {DEFAULT_LAB_SERVER}）",
    ),
    token: Optional[str] = typer.Option(None, "--token", help="非交互登录：直接用服务令牌（CI 用）"),
    device_flow: bool = typer.Option(False, "--device-flow", help="强制使用设备码登录（SSH / 无浏览器）"),
    no_browser: bool = typer.Option(False, "--no-browser", help="不打开浏览器（等同 --device-flow）"),
) -> None:
    """登录 Lab（本机默认浏览器；SSH 环境走验证码）。"""
    srv = current_server(server)
    _save_server(srv)
    if token:
        creds = {"access_token": token, "refresh_token": None, "expires_at": None, "user": None}
        try:
            who = _api(srv, "GET", "/api/whoami", token=token)
            creds["user"] = who.get("user")
        except urllib.error.HTTPError:
            cli_ui.fail("登录令牌无效，请重新登录。", hint="运行 lab login 重新登录")
        _save_creds(srv, creds)
    else:
        creds = _interactive_login(srv, device_flow=device_flow, no_browser=no_browser)
        _save_creds(srv, creds)
    u = (creds.get("user") or {}).get("username", "?")
    typer.secho(f"✓ 已登录：{u}", fg=typer.colors.GREEN)


def logout(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="指定 Lab 地址（默认当前）"),
) -> None:
    """登出当前账号。"""
    srv = current_server(server)
    creds = _load_creds(srv)
    if not creds:
        typer.echo("当前未登录。")
        return
    if creds.get("refresh_token"):
        try:
            _api(srv, "POST", "/api/auth/logout", body={"refresh_token": creds["refresh_token"]})
        except urllib.error.URLError:
            pass
    _clear_creds(srv)
    typer.secho("✓ 已登出", fg=typer.colors.GREEN)


def whoami(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="指定 Lab 地址（默认当前）"),
) -> None:
    """显示当前账号与配额。"""
    srv = current_server(server)
    try:
        who = whoami_via_server(srv)
    except typer.Exit:
        raise
    except urllib.error.HTTPError:
        typer.secho("无法获取账号信息，请稍后重试。", fg=typer.colors.RED)
        raise typer.Exit(1) from None
    user = who.get("user") or {}
    quota = who.get("quota") or {}
    typer.echo(f"用户：{user.get('username') or '?'}  角色：{user.get('role') or '?'}")
    cap = quota.get("max_concurrent_gpus")
    jobs = quota.get("max_concurrent_jobs")
    daily = quota.get("daily_gpu_hours")
    typer.echo(
        f"配额：GPU {'不限' if cap is None else cap}"
        f" · 作业 {jobs if jobs is not None else '不限'}"
        f" · 日 GPU 时 {daily if daily else '不限'}"
    )


def quota(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="指定 Lab 地址（默认当前）"),
) -> None:
    """查看配额与实时用量。"""
    srv = current_server(server)
    token = get_access_token(srv)
    if not token:
        typer.secho(MSG_NOT_LOGGED_IN, fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    data = _api(srv, "GET", "/api/quota", token=token)
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


# ----------------------------- lab admin（管理员）-----------------------------
def _admin_call(method: str, path: str, *, body: Optional[dict] = None) -> dict:
    srv = current_server()
    token = get_access_token(srv)
    if not token:
        cli_ui.fail(MSG_NOT_LOGGED_IN, hint="运行 lab login 登录")
    try:
        return _api(srv, method, path, token=token, body=body)
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="请求失败。")


admin_app = typer.Typer(
    no_args_is_help=True,
    help="管理员：用户与配额管理（需 admin 权限）。",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@admin_app.command("users", help="列出所有用户")
def admin_users() -> None:
    data = _admin_call("GET", "/api/admin/users")
    for u in data.get("users", []):
        flag = " [disabled]" if u.get("disabled") else ""
        typer.echo(f"{u['id']:>3}  {u['username']:<20} {u['role']:<10} {u.get('auth_source','')}{flag}")


@admin_app.command("user-add", help="新建本地账号")
def admin_user_add(
    username: str = typer.Argument(...),
    password: str = typer.Option(..., "--password", "-p", prompt=True, hide_input=True),
    role: str = typer.Option("operator", "--role", help="admin | operator | viewer"),
    email: Optional[str] = typer.Option(None, "--email"),
) -> None:
    u = _admin_call("POST", "/api/admin/users", body={"username": username, "password": password, "role": role, "email": email})
    typer.secho(f"✓ 已创建 {u['username']}（{u['role']}）", fg=typer.colors.GREEN)


@admin_app.command("set-role", help="修改用户角色")
def admin_set_role(username: str = typer.Argument(...), role: str = typer.Argument(...)) -> None:
    u = _admin_call("PATCH", f"/api/admin/users/{username}/role?role={urllib.parse.quote(role)}")
    typer.secho(f"✓ {u['username']} → {u['role']}", fg=typer.colors.GREEN)


@admin_app.command("disable", help="停用/启用用户（--on 停用，--off 启用）")
def admin_disable(
    username: str = typer.Argument(...),
    disabled: bool = typer.Option(True, "--on/--off", help="--on 停用，--off 启用"),
) -> None:
    u = _admin_call("PATCH", f"/api/admin/users/{username}/disabled?disabled={str(disabled).lower()}")
    typer.secho(f"✓ {u['username']} disabled={u['disabled']}", fg=typer.colors.GREEN)


@admin_app.command("set-quota", help="设置用户算力配额")
def admin_set_quota(
    username: str = typer.Argument(...),
    gpus: int = typer.Option(8, "--gpus", help="并发 GPU 上限"),
    jobs: int = typer.Option(4, "--jobs", help="并发作业上限"),
    daily_gpu_hours: int = typer.Option(0, "--daily-gpu-hours", help="每日 GPU-时（0=不限）"),
    profiles: Optional[str] = typer.Option(None, "--profiles", help="允许的 profile（逗号分隔，空=全部）"),
    priority: int = typer.Option(0, "--priority", help="排队优先级"),
) -> None:
    allowed = [p.strip() for p in profiles.split(",") if p.strip()] if profiles else []
    q = _admin_call("POST", "/api/admin/quotas", body={
        "username": username, "max_concurrent_gpus": gpus, "max_concurrent_jobs": jobs,
        "daily_gpu_hours": daily_gpu_hours, "allowed_profiles": allowed, "priority": priority,
    })
    typer.secho(f"✓ 已设置 {username} 配额：{json.dumps(q, ensure_ascii=False)}", fg=typer.colors.GREEN)
