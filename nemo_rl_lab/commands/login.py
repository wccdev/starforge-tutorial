"""身份命令：login / logout（凭据管理核心见 nemo_rl_lab.auth）。"""
from __future__ import annotations

import urllib.error
from typing import Optional

import typer

from nemo_rl_lab import auth, cli_ui


def login(
    server: Optional[str] = typer.Option(
        None, "--server", "-s",
        help=f"Lab 服务地址（默认 {auth.DEFAULT_LAB_SERVER}）",
    ),
    token: Optional[str] = typer.Option(None, "--token", help="非交互登录：直接用服务令牌（CI 用）"),
    device_flow: bool = typer.Option(False, "--device-flow", help="强制使用设备码登录（SSH / 无浏览器）"),
    no_browser: bool = typer.Option(False, "--no-browser", help="不打开浏览器（等同 --device-flow）"),
) -> None:
    """登录 Lab（本机默认浏览器；SSH 环境走验证码）。"""
    srv = auth.current_server(server)
    auth._save_server(srv)
    if token:
        creds = {"access_token": token, "refresh_token": None, "expires_at": None, "user": None}
        try:
            who = auth._api(srv, "GET", "/api/whoami", token=token)
            creds["user"] = who.get("user")
        except urllib.error.HTTPError:
            cli_ui.fail("登录令牌无效，请重新登录。", hint="运行 lab login 重新登录")
        auth._save_creds(srv, creds)
    else:
        creds = auth._interactive_login(srv, device_flow=device_flow, no_browser=no_browser)
        auth._save_creds(srv, creds)
    u = (creds.get("user") or {}).get("username", "?")
    typer.secho(f"✓ 已登录：{u}", fg=typer.colors.GREEN)


def logout(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="指定 Lab 地址（默认当前）"),
) -> None:
    """登出当前账号。"""
    srv = auth.current_server(server)
    creds = auth._load_creds(srv)
    if not creds:
        typer.echo("当前未登录。")
        return
    if creds.get("refresh_token"):
        try:
            auth._api(srv, "POST", "/api/auth/logout", body={"refresh_token": creds["refresh_token"]})
        except urllib.error.URLError:
            pass
    auth._clear_creds(srv)
    typer.secho("✓ 已登出", fg=typer.colors.GREEN)
