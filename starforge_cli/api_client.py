"""Console API 客户端：所有经中心化服务的网络调用。

原则：HTTP 失败一律显式报错（cli_ui.fail_http），不返回 None 静默降级。
打包/上传只发生在 catalog 握手通过之后——契约不符时一个字节都不传。
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from starforge_cli import cli_ui
from starforge_cli.auth import MSG_NOT_LOGGED_IN, _api, current_server, get_access_token
from starforge_cli.catalog import CatalogCompatibilityError, verify_catalog_compatibility
from starforge_cli.packing import git_provenance, list_working_files, pack_working_dir


# ----------------------------- 基础请求 -----------------------------
def _bearer_request(server: str, method: str, path: str, *, data=None,
                    headers: Optional[dict] = None, timeout: Optional[float] = 60.0):
    """带 token 的请求；返回 urlopen 的响应对象（调用方负责读取/关闭）。"""
    token = get_access_token(server)
    if not token:
        cli_ui.fail(MSG_NOT_LOGGED_IN, hint="运行 forge login 登录")
    h = {"Authorization": f"Bearer {token}"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(f"{server}{path}", data=data, headers=h, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def api_get(path: str, server: Optional[str] = None) -> dict:
    """带 token 的 GET，返回 JSON。"""
    srv = current_server(server)
    try:
        with _bearer_request(srv, "GET", path) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="请求失败，请稍后重试。")


def api_post(path: str, payload: dict, server: Optional[str] = None) -> dict:
    srv = current_server(server)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        with _bearer_request(
            srv, "POST", path, data=body, headers={"Content-Type": "application/json"}
        ) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="请求失败，请稍后重试。")


def api_patch(path: str, payload: dict, server: Optional[str] = None) -> dict:
    srv = current_server(server)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        with _bearer_request(
            srv, "PATCH", path, data=body, headers={"Content-Type": "application/json"}
        ) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="请求失败，请稍后重试。")


def api_post_bytes(
    path: str, blob: bytes, *, content_type: str = "application/gzip",
    server: Optional[str] = None,
) -> dict:
    """带 token 的二进制 POST（插件包发布等小体积上传），返回 JSON。"""
    srv = current_server(server)
    try:
        with _bearer_request(
            srv, "POST", path, data=blob, headers={"Content-Type": content_type},
            timeout=120.0,
        ) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="上传失败，请稍后重试。")


def api_get_bytes(path: str, server: Optional[str] = None) -> tuple[bytes, dict]:
    """带 token 的 GET，返回 (原始字节, 响应头)。"""
    srv = current_server(server)
    try:
        with _bearer_request(srv, "GET", path, timeout=120.0) as r:
            return r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="下载失败，请稍后重试。")


def _admin_call(method: str, path: str, *, body: Optional[dict] = None) -> dict:
    srv = current_server()
    token = get_access_token(srv)
    if not token:
        cli_ui.fail(MSG_NOT_LOGGED_IN, hint="运行 forge login 登录")
    try:
        return _api(srv, method, path, token=token, body=body)
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="请求失败。")


# ----------------------------- catalog 握手 -----------------------------
def verify_server_compatibility(spec, server: Optional[str] = None) -> None:
    """联网握手；任何不匹配都在打包前终止。"""
    srv = current_server(server)
    payload = api_get("/api/recipes", server=srv)
    try:
        verify_catalog_compatibility(spec, payload)
    except CatalogCompatibilityError as exc:
        cli_ui.fail(str(exc), hint="升级 CLI/SDK 或让平台发布完全一致的 recipe 版本")


# ----------------------------- 流式上传 -----------------------------
class _ProgressReader:
    """把字节流包成「边读边回报」的类文件对象，供 urllib 流式上传时驱动进度条。

    urllib/http.client 会分块调用 read() 直到读空；读空时触发 on_done（= 上传完毕、
    开始等待服务端受理）。
    """

    def __init__(self, data: bytes, on_read=None, on_done=None):
        self._buf = io.BytesIO(data)
        self._total = len(data)
        self._on_read = on_read
        self._on_done = on_done
        self._done_fired = False

    def read(self, size: int = -1) -> bytes:
        chunk = self._buf.read(size)
        if chunk:
            if self._on_read:
                self._on_read(len(chunk))
        elif not self._done_fired:
            self._done_fired = True
            if self._on_done:
                self._on_done()
        return chunk

    def __len__(self) -> int:
        return self._total


def _upload_and_submit(srv: str, path: str, meta: dict, repo_root: Path, *,
                       exp_rel: str, profile: str, reporter, fail_msg: str) -> dict:
    """通用：清单式打包 → 流式上传 → 解析响应，全程可选驱动进度条。"""
    files, skipped = list_working_files(
        repo_root, exp_rel=exp_rel, profile=profile, with_stats=True
    )
    if reporter:
        reporter.start_pack(len(files))
    blob = pack_working_dir(
        repo_root, files, on_add=(reporter.pack_tick if reporter else None)
    )
    headers = {
        "Content-Type": "application/gzip",
        "X-Lab-Meta": json.dumps(meta, ensure_ascii=False),
        "Content-Length": str(len(blob)),
    }
    if reporter:
        reporter.start_upload(len(blob))
        data = _ProgressReader(blob, on_read=reporter.upload_tick, on_done=reporter.awaiting_server)
    else:
        data = blob
    try:
        with _bearer_request(srv, "POST", path, data=data, headers=headers, timeout=300.0) as r:
            result = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback=fail_msg)
    if reporter:
        reporter.finish()
    if isinstance(result, dict):
        result.setdefault("upload_files", len(files))
        result.setdefault("upload_skipped", skipped)
        result.setdefault("upload_bytes", len(blob))
    return result


def submit_via_server(exp_rel: str, profile: str, repo_root: Path,
                      server: Optional[str] = None, project: Optional[str] = None,
                      reporter=None, spec=None, extra_meta: Optional[dict] = None) -> dict:
    """server 模式提交：清单式打包上传 + 服务端注入密钥后代理提交。

    spec：必填的 lab/v2 JobSpec；profile：必填（--profile 或旧实验遗留 cluster 标注解析而来）。
    extra_meta：附加的 client_meta 键（如 sweep_id/sweep_params），不得覆盖平台保留键。
    上传前先与 Console 做精确 catalog 握手，握手不过一个字节都不上传。
    返回值附带 upload_files / upload_skipped / upload_bytes 便于 CLI 展示。
    """
    if spec is None:
        raise ValueError("提交必须携带 lab/v2 JobSpec")
    if not profile:
        raise ValueError("提交必须携带显式硬件 profile")
    srv = current_server(server)
    verify_server_compatibility(spec, server=srv)
    meta = {"exp": exp_rel, "profile": profile, **git_provenance(repo_root, exp_rel)}
    if project:
        meta["project"] = project
    if extra_meta:
        reserved = set(meta) | {"spec"}
        clash = reserved & set(extra_meta)
        if clash:
            raise ValueError(f"extra_meta 不得覆盖平台保留键: {sorted(clash)}")
        meta.update(extra_meta)
    meta["spec"] = spec.to_dict()
    return _upload_and_submit(
        srv, "/api/jobs", meta, repo_root,
        exp_rel=exp_rel, profile=profile, reporter=reporter,
        fail_msg="提交失败，请稍后重试。",
    )


def submit_post_via_server(action: str, exp_rel: str, profile: str, flags: list[str],
                           repo_root: Path, server: Optional[str] = None, reporter=None,
                           spec=None) -> dict:
    """server 模式训练后闭环：与训练共用 lab/v2 launcher 与 catalog 握手。"""
    if spec is None:
        raise ValueError("训练后作业必须携带 lab/v2 JobSpec")
    if not profile:
        raise ValueError("训练后作业必须携带显式硬件 profile")
    srv = current_server(server)
    verify_server_compatibility(spec, server=srv)
    meta = {"action": action, "exp": exp_rel, "profile": profile,
            "flags": flags, "spec": spec.to_dict(), **git_provenance(repo_root, exp_rel)}
    label = "导出" if action == "export" else "评测"
    return _upload_and_submit(
        srv, "/api/post", meta, repo_root,
        exp_rel=exp_rel, profile=profile, reporter=reporter,
        fail_msg=f"{label}提交失败，请稍后重试。",
    )


# ----------------------------- 作业查询 / 控制 -----------------------------
def clean_via_server(exp_rel: str, server: Optional[str] = None) -> dict:
    """清理本实验在集群上的产物目录（checkpoint/日志），经服务端在集群侧删除。"""
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
    try:
        with _bearer_request(srv, "GET", "/api/usage/mine") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="无法获取用量信息。")


def whoami_via_server(server: Optional[str] = None) -> dict:
    """取当前登录身份 + 配额。"""
    srv = current_server(server)
    token = get_access_token(srv)
    if not token:
        cli_ui.fail(MSG_NOT_LOGGED_IN, hint="运行 forge login 登录")
    try:
        return _api(srv, "GET", "/api/whoami", token=token)
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="无法获取账号信息。")


def list_my_jobs(server: Optional[str] = None, limit: int = 50) -> list[dict]:
    """获取作业列表。"""
    srv = current_server(server)
    try:
        with _bearer_request(srv, "GET", f"/api/jobs/mine?limit={limit}") as r:
            return (json.loads(r.read() or b"{}")).get("jobs", [])
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="无法获取作业列表。")


def job_control_via_server(action: str, job_id: str, server: Optional[str] = None) -> dict:
    """停止 / 删除 / 暂停 / 继续作业。"""
    srv = current_server(server)
    path = f"/api/job/{action}?id={urllib.parse.quote(job_id)}"
    labels = {"stop": "停止作业", "delete": "删除记录", "pause": "暂停作业", "resume": "继续作业"}
    try:
        with _bearer_request(srv, "POST", path) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback=f"{labels.get(action, '操作')}失败。")


def latest_job_via_server(server: Optional[str] = None) -> Optional[str]:
    """最近一个作业的 ID；没有任何作业时返回 None，HTTP 失败显式报错。"""
    srv = current_server(server)
    try:
        with _bearer_request(srv, "GET", "/api/jobs/mine?limit=1") as r:
            jobs = (json.loads(r.read() or b"{}")).get("jobs", [])
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="无法获取作业列表。")
    return jobs[0].get("job_ref") if jobs else None


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


def cluster_status_via_server(server: Optional[str] = None) -> dict:
    """取集群 GPU 概览 + 活跃作业；服务不可达时显式报错。"""
    srv = current_server(server)
    try:
        with _bearer_request(srv, "GET", "/api/cluster/status") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="无法获取集群状态。")


def batch_via_server(action: str, server: Optional[str] = None) -> dict:
    """批量作业控制：cancel-all / clean。"""
    srv = current_server(server)
    try:
        with _bearer_request(srv, "POST", f"/api/jobs/{action}") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="操作失败，请稍后重试。")


def stop_sweep_via_server(sweep_id: str, server: Optional[str] = None) -> dict:
    """一键停止一个超参 sweep 的全部活跃作业。"""
    srv = current_server(server)
    path = f"/api/sweeps/{urllib.parse.quote(sweep_id, safe='')}/stop"
    try:
        with _bearer_request(srv, "POST", path) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="停止 sweep 失败，请稍后重试。")


# ----------------------------- 数据集上传 -----------------------------
def _sha256_file(path: Path) -> tuple[str, int]:
    """分块计算文件 sha256 与大小 —— 不把整个文件读进内存（parquet 动辄几 GB）。"""
    import hashlib

    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _push_state_paths():
    import os

    from starforge_cli.auth import FORGE_DIR, _read_json, _write_json

    # 调用时读 FORGE_HOME（而不是沿用 import 时固化的 FORGE_DIR）：测试与多环境切换友好。
    base = Path(os.environ.get("FORGE_HOME") or FORGE_DIR)
    return base / "dataset-push-state.json", _read_json, _write_json


def _put_streaming(url: str, payload, size: int, headers: dict, timeout: float) -> None:
    """预签名 PUT。文件走流式（http.client 分块发送文件对象），不整份进内存。

    必须显式带 Content-Length：S3 预签名 PUT 不接受 chunked 传输编码，
    urllib 对无长度的 body 会自动切到 chunked。
    """
    h = {**headers, "Content-Length": str(size)}
    if isinstance(payload, (bytes, bytearray)):
        req = urllib.request.Request(url, data=bytes(payload), method="PUT", headers=h)
        urllib.request.urlopen(req, timeout=timeout)
        return
    with open(payload, "rb") as fh:
        req = urllib.request.Request(url, data=fh, method="PUT", headers=h)
        urllib.request.urlopen(req, timeout=timeout)


def dataset_push(
    dataset: str,
    version: str,
    root: Path,
    files: list,
    visibility: Optional[str] = None,
    server: Optional[str] = None,
) -> str:
    """上传一个数据集版本：逐文件走预签名 PUT，最后写 index.json。返回完整数据集 ID。

    dataset 可以是 `<owner>/<name>`，也可以是裸 `<name>`（服务端归到当前用户命名空间）。
    visibility 只在数据集首次创建时生效（public|private，默认 private）。

    直连对象存储，不经过 console —— 数据集动辄几 GB，穿过 API 进程内存是没道理的；
    同理文件本体也不整份读进 CLI 内存（流式 PUT + 分块 sha256）。
    客户端全程不持有对象存储凭据，只拿一次性预签名 URL。

    断点续传：每个文件 PUT 成功后把 (rel → sha256) 记进本地状态文件；重跑同一
    dataset@version 时内容未变的文件直接跳过。键含 sha256，本地文件改过就会重传
    —— 不会把旧内容的记录当成已上传（否则 index.json 的校验和会与对象不符，
    作业侧下载校验必炸）。版本完整上传（index.json 落位）后清除状态。
    """
    srv = current_server(server)

    def _upload_url(filename: str) -> dict:
        body: dict = {"dataset": dataset, "version": version, "filename": filename}
        if visibility:
            body["visibility"] = visibility
        return api_post("/api/datasets/upload-url", body, server=srv)

    def _put_with_retry(rel: str, payload, size: int, *, timeout: float, attempts: int = 3) -> dict:
        """取预签名 URL → PUT，瞬时失败退避重试（每轮重新取 URL：403 常为过期）。"""
        for attempt in range(attempts):
            up = _upload_url(rel)
            headers = dict(up.get("headers") or {"Content-Type": "application/octet-stream"})
            try:
                _put_streaming(up["upload_url"], payload, size, headers, timeout)
                return up
            except urllib.error.HTTPError as e:
                # 403 多为预签名过期（重取 URL 后重试）；5xx 视为瞬时；其余 4xx 是
                # 确定性错误（409 版本已封存等），重试只会重复同一个失败。
                if not (e.code == 403 or e.code >= 500) or attempt == attempts - 1:
                    cli_ui.fail(f"上传 {rel} 失败: HTTP {e.code}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt == attempts - 1:
                    cli_ui.fail(
                        f"上传 {rel} 失败: {e}",
                        hint="网络恢复后直接重跑同一命令，已上传的文件会自动跳过",
                    )
            time.sleep(1.5 * (attempt + 1))
        raise AssertionError("unreachable")

    state_path, read_state, write_state = _push_state_paths()
    state_key = f"{srv}|{dataset}@{version}"
    state = read_state(state_path)
    done: dict = dict(state.get(state_key) or {})

    ds_id = dataset
    index_files = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        sha, size = _sha256_file(f)
        index_files.append({"name": rel, "size": size, "sha256": sha})
        if done.get(rel) == sha:
            print(f"  = {rel}  上次已传完，跳过")
            continue
        up = _put_with_retry(rel, f, size, timeout=600)
        ds_id = up.get("dataset") or ds_id
        done[rel] = sha
        state[state_key] = done
        write_state(state_path, state)
        print(f"  ↑ {rel}  {cli_ui.human_bytes(size)}")

    # index.json 最后写：它的存在即「这个版本已完整上传」的标记（此后版本不可变）。
    body = json.dumps({"files": index_files}, ensure_ascii=False).encode("utf-8")
    up = _put_with_retry("index.json", body, len(body), timeout=60)
    ds_id = up.get("dataset") or ds_id
    # 版本已封存：清掉断点状态，避免状态文件随历史版本无限增长。
    if state.pop(state_key, None) is not None:
        write_state(state_path, state)
    return ds_id


# ----------------------------- SSE 日志流 -----------------------------
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
    """经服务端 SSE 接口跟随作业日志（客户端不直连集群）。

    只把 log 事件原文还原后打到 stdout，不暴露 event:/id:/data:/keepalive 等协议噪音。
    tail 给定时只回放最后 N 行历史日志再跟随（默认 2000；0 或 None=全量）。

    健壮性：服务端为多副本 Redis Streams 推送，长连接可能被反代/实例切换回收。
    本函数在连接非正常结束（未收到 end 事件）时按指数退避自动重连，并携带
    Last-Event-ID 头 + ?from=<id> 从断点续传，避免日志丢失或重复回放历史。
    作业到达终态时服务端发 end 事件，收到后干净退出（不再重连）。
    """
    import random

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
