"""Playground 推理服务：训练产物一键起 vLLM 服务并试用。

服务生命周期完全由 console 管理（GPU 台账 + 闲置 TTL 自动回收）；
CLI 只是端点的瘦封装。对话试用建议用 web「Playground」页（流式 UI）。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
from typing import Optional

import typer

from nemo_rl_lab import api_client, cli_ui
from nemo_rl_lab.auth import gate

serve_app = typer.Typer(no_args_is_help=True, help="推理服务（Playground）")


def _call(method: str, path: str, body: dict | None = None) -> dict:
    srv = api_client.current_server(None)
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else None
    try:
        with api_client._bearer_request(srv, method, path, data=data, headers=headers) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback="Playground 操作失败")


@serve_app.command("start", help="启动推理服务（模型：HF id / 绝对路径 / run:<run_id>）")
def serve_start(
    model: str = typer.Argument(..., help="HF id / 共享盘绝对路径 / run:<run_id>"),
    gpus: int = typer.Option(1, "--gpus", "-g", help="张数（张量并行度）"),
    ttl_hours: Optional[float] = typer.Option(None, "--ttl-hours", help="闲置 TTL（小时），默认服务端配置"),
) -> None:
    gate()
    body: dict = {"model": model, "gpus": gpus}
    if ttl_hours:
        body["ttl_s"] = ttl_hours * 3600
    res = _call("POST", "/api/playground/start", body)
    typer.secho(f"✓ 推理服务已启动  {res.get('lab_run_id')}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  模型: {res.get('model')}  端口: {res.get('port')}  到期: {res.get('expires_at')}")
    typer.echo("  对话试用：web「Playground」页；一键停止：lab serve stop <run_id>")


@serve_app.command("ls", help="列出我的推理服务")
def serve_ls() -> None:
    gate()
    res = _call("GET", "/api/playground")
    rows = res.get("servings") or []
    if not rows:
        typer.echo("没有推理服务。启动：lab serve start <模型>")
        return
    for r in rows:
        flag = "●" if r.get("active") else "○"
        typer.echo(
            f"{flag} {r.get('lab_run_id')}  {r.get('status'):<9} "
            f"{r.get('model') or '-'}  {r.get('gpus')}GPU  到期 {r.get('expires_at') or '-'}"
        )


@serve_app.command("stop", help="停止推理服务")
def serve_stop(run_id: str = typer.Argument(...)) -> None:
    gate()
    _call("POST", f"/api/playground/{urllib.parse.quote(run_id, safe='')}/stop")
    typer.secho("✓ 已停止", fg=typer.colors.GREEN)


@serve_app.command("extend", help="续期推理服务 TTL")
def serve_extend(
    run_id: str = typer.Argument(...),
    hours: float = typer.Option(1.0, "--hours", help="续期时长（小时）"),
) -> None:
    gate()
    res = _call("POST", f"/api/playground/{urllib.parse.quote(run_id, safe='')}/extend",
                {"extra_s": hours * 3600})
    typer.secho(f"✓ 已续期，到期 {res.get('expires_at')}", fg=typer.colors.GREEN)
