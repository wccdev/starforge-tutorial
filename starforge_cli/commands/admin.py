"""管理员命令：用户 / 配额 / 维护模式（需 admin 权限）。"""
from __future__ import annotations

import json
import urllib.parse
from typing import Optional

import typer

from starforge_cli import api_client

admin_app = typer.Typer(
    no_args_is_help=True,
    help="管理员：用户与配额管理（需 admin 权限）。",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@admin_app.command("users", help="列出所有用户")
def admin_users() -> None:
    data = api_client._admin_call("GET", "/api/admin/users")
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
    u = api_client._admin_call(
        "POST", "/api/admin/users",
        body={"username": username, "password": password, "role": role, "email": email},
    )
    typer.secho(f"✓ 已创建 {u['username']}（{u['role']}）", fg=typer.colors.GREEN)


@admin_app.command("set-role", help="修改用户角色")
def admin_set_role(username: str = typer.Argument(...), role: str = typer.Argument(...)) -> None:
    u = api_client._admin_call("PATCH", f"/api/admin/users/{username}/role?role={urllib.parse.quote(role)}")
    typer.secho(f"✓ {u['username']} → {u['role']}", fg=typer.colors.GREEN)


@admin_app.command("disable", help="停用/启用用户（--on 停用，--off 启用）")
def admin_disable(
    username: str = typer.Argument(...),
    disabled: bool = typer.Option(True, "--on/--off", help="--on 停用，--off 启用"),
) -> None:
    u = api_client._admin_call("PATCH", f"/api/admin/users/{username}/disabled?disabled={str(disabled).lower()}")
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
    q = api_client._admin_call("POST", "/api/admin/quotas", body={
        "username": username, "max_concurrent_gpus": gpus, "max_concurrent_jobs": jobs,
        "daily_gpu_hours": daily_gpu_hours, "allowed_profiles": allowed, "priority": priority,
    })
    typer.secho(f"✓ 已设置 {username} 配额：{json.dumps(q, ensure_ascii=False)}", fg=typer.colors.GREEN)


# ----------------------------- 维护模式（滚动升级集群镜像）-----------------------------
# 完整流程见 console 仓库 deploy/ray-cluster/README.md「升级」一节：
#   forge admin maintenance drain --note "升级镜像至 0.7.0-20260805"
#   forge admin maintenance status            # 等到「可以重启」
#   （各节点 docker compose pull && up -d）
#   forge admin maintenance resume            # 作业自动从 checkpoint 续训
maintenance_app = typer.Typer(
    no_args_is_help=True,
    help="维护模式：排空集群 → 升级 → 恢复（作业从 checkpoint 续训，不丢进度）",
    context_settings={"help_option_names": ["-h", "--help"]},
)
admin_app.add_typer(maintenance_app, name="maintenance")


def _print_maintenance(data: dict) -> None:
    on = data.get("maintenance_mode")
    typer.secho(
        f"维护模式：{'开启' if on else '关闭'}" + (f"（{data['note']}）" if data.get("note") else ""),
        fg=typer.colors.YELLOW if on else typer.colors.GREEN,
    )
    if paused := data.get("paused"):
        typer.echo(f"本次暂停 {len(paused)} 个训练作业：")
        for rid in paused:
            typer.echo(f"  · {rid}")
    if pending := data.get("paused_awaiting_resume"):
        typer.echo(f"待自动续训：{pending} 个")

    # blockers / failed 是决定「能不能动集群」的关键，务必显眼
    for key, label, color in (
        ("blockers", "仍占卡（不支持 checkpoint 续跑，需等它跑完或手动停）", typer.colors.YELLOW),
        ("failed", "停止失败（重跑 drain 或手动处理）", typer.colors.RED),
    ):
        for item in data.get(key) or []:
            typer.secho(
                f"  ⚠ [{label}] {item.get('lab_run_id')} ({item.get('username')})"
                + (f" — {item['reason']}" if item.get("reason") else ""),
                fg=color,
            )

    if "safe_to_restart" in data:
        if data["safe_to_restart"]:
            typer.secho("\n✓ 集群已排空，可以重启节点了", fg=typer.colors.GREEN, bold=True)
            typer.echo("  各节点：docker compose --profile <head|worker> pull && up -d")
            typer.echo("  升级完成后：forge admin maintenance resume")
        else:
            n = data.get("remaining_active", data.get("active_jobs", "?"))
            typer.secho(f"\n✗ 还有 {n} 个作业占着卡，现在重启会打断它们", fg=typer.colors.RED, bold=True)


@maintenance_app.command("status", help="查看维护状态与是否可以安全重启集群")
def maintenance_status() -> None:
    _print_maintenance(api_client._admin_call("GET", "/api/admin/maintenance"))


@maintenance_app.command("drain", help="进入维护模式并排空集群（幂等，没排干净就再跑一次）")
def maintenance_drain(
    note: str = typer.Option("", "--note", help="维护说明，会回显给被拦下的提交者"),
) -> None:
    _print_maintenance(api_client._admin_call("POST", "/api/admin/maintenance/drain", body={"note": note}))


@maintenance_app.command("resume", help="退出维护模式，被暂停的作业自动从 checkpoint 续训")
def maintenance_resume() -> None:
    data = api_client._admin_call("POST", "/api/admin/maintenance/resume")
    typer.secho(
        f"✓ 已退出维护模式，{data.get('resuming', 0)} 个作业将由队列按原优先级自动续训",
        fg=typer.colors.GREEN,
    )
