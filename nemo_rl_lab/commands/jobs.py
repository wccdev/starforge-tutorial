"""作业观测与控制：status + job 子命令组（全部经 Console）。"""
from __future__ import annotations

from typing import Optional

import typer

from nemo_rl_lab import api_client, cli_ui
from nemo_rl_lab.auth import gate
from nemo_rl_lab.commands import common

job_app = typer.Typer(
    no_args_is_help=True,
    help="作业管理",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _format_user_label(user: dict) -> str:
    """把 /api/whoami 的 user 格式化为单行展示。"""
    username = user.get("username") or "?"
    role = user.get("role") or "?"
    parts = [f"用户：{username}", f"角色：{role}"]
    if user.get("email"):
        parts.append(f"邮箱：{user['email']}")
    return "  ".join(parts)


def status() -> None:
    """账号、配额、用量与活跃作业。"""
    gate()
    who = api_client.whoami_via_server()
    user = who.get("user") or {}
    typer.echo(_format_user_label(user))
    typer.echo("")

    data = api_client.usage_via_server()
    q, u = data.get("quota") or {}, data.get("usage") or {}
    cap = q.get("max_concurrent_gpus")
    typer.echo("我的用量")
    typer.echo(f"  并发 GPU : {u.get('active_gpus', 0)} / {'不限' if cap is None else cap}")
    typer.echo(f"  并发作业 : {u.get('active_jobs', 0)} / {q.get('max_concurrent_jobs') or '不限'}")
    typer.echo(f"  今日/累计 GPU-hours : {u.get('gpu_hours_today', 0):.1f} / {u.get('gpu_hours_total', 0):.1f}")
    running = u.get("running") or []
    typer.echo("\n我的活跃作业")
    if not running:
        typer.echo("  （无）")
    else:
        for r in running:
            jid = (r.get("job_ref") or r.get("lab_run_id") or "-")[:26]
            typer.echo(f"  {jid:<26} {r.get('status','-'):<10} GPU={r.get('gpus') or '-'}  {r.get('exp','-')}")

    cluster = api_client.cluster_status_via_server()
    gpu = (cluster or {}).get("gpu") or {}
    if gpu:
        accel = "/".join(gpu.get("accel") or []) or "GPU"
        typer.echo("\n集群 GPU")
        typer.echo(
            f"  {accel} : 空闲 {gpu.get('gpu_free', 0):g} / 共 {gpu.get('gpu_total', 0):g}"
            f"（占用 {gpu.get('gpu_used', 0):g}）"
        )
        typer.echo(f"  活跃作业 : {cluster.get('active_count', 0)}")
    typer.echo("\n查看日志：lab job logs [作业 ID]")


def _server_jobs_table(jobs: list[dict]) -> None:
    if not jobs:
        typer.echo("（无作业）")
        return
    typer.echo(f"{'TIME':<20} {'JOB ID':<26} {'状态':<10} {'GPU':>4}  实验")
    for j in jobs:
        jid = (j.get("job_ref") or j.get("lab_run_id") or "-")[:26]
        typer.echo(
            f"{str(j.get('submitted_at', '-'))[:19]:<20} {jid:<26} "
            f"{str(j.get('status','-')):<10} {str(j.get('requested_gpus') or '-'):>4}  "
            f"{j.get('exp','-')}"
        )


@job_app.command("ls", help="作业列表（含提交历史）")
def job_ls(
    all_jobs: bool = typer.Option(False, "--all", help="显示全部（默认最近 15 条）"),
    exp: Optional[str] = typer.Option(
        None, "--exp", autocompletion=common.complete_exp, help="只看某实验（接受全名或末段名）"
    ),
    limit: int = typer.Option(15, "-n", "--limit", help="显示条数（--all 时忽略）"),
) -> None:
    gate()
    jobs = api_client.list_my_jobs(limit=200 if all_jobs else limit)
    if exp:
        jobs = [j for j in jobs if exp in (j.get("exp") or "")]
    _server_jobs_table(jobs)


@job_app.command("logs", help="跟随作业日志（省略作业 ID 则跟最近一个）")
def job_logs(
    job_id: Optional[str] = typer.Argument(None, help="作业 ID（见 lab job ls）；省略=最近一个"),
    tail: Optional[int] = typer.Option(
        2000, "-n", "--tail", help="只回放最后 N 行历史日志再跟随（默认 2000；-n 0 看全量）"
    ),
) -> None:
    gate()
    jid = job_id or api_client.latest_job_via_server()
    if not jid:
        cli_ui.emit_warning("还没有作业", hint="运行 lab submit 提交训练")
        raise typer.Exit(1)
    api_client.stream_logs_via_server(jid, tail=tail)


@job_app.command("status", help="查看作业状态")
def job_status(
    job_id: str = typer.Argument(..., help="作业 ID"),
) -> None:
    gate()
    match = [j for j in api_client.list_my_jobs(limit=200)
             if job_id in (j.get("job_ref") or "", j.get("lab_run_id") or "")]
    if not match:
        cli_ui.fail(f"未找到作业 {job_id}")
    _server_jobs_table(match)


@job_app.command("samples", help="查看某次验证的多轮对话轨迹（默认最近一次验证）")
def job_samples(
    job_id: str = typer.Argument(..., help="作业 ID（见 lab job ls）"),
    vidx: int = typer.Option(-1, "--vidx", help="验证轮次下标（默认 -1=最近一次）"),
    n: int = typer.Option(6, "-n", "--limit", help="显示样本条数"),
) -> None:
    gate()
    overview = api_client.job_overview_via_server(job_id)
    vals = overview.get("validations") or []
    if not vals:
        typer.secho("该作业暂无验证样本。", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    idx = vidx if vidx >= 0 else len(vals) + vidx
    if idx < 0 or idx >= len(vals):
        typer.secho(f"验证下标越界：vidx={vidx}，共 {len(vals)} 轮。", fg=typer.colors.RED)
        raise typer.Exit(1)
    page = api_client.samples_via_server(job_id, idx, 0, n)
    samples = page.get("samples") or []
    typer.echo(
        f"验证 step={page.get('step', '?')}（第 {idx + 1}/{len(vals)} 轮）  "
        f"样本 {len(samples)}/{page.get('total', len(samples))}"
    )
    for s in samples:
        typer.echo("")
        typer.secho(f"── Sample {s.get('idx', '?')} | reward={s.get('reward', '?')} ──", fg=typer.colors.CYAN)
        if s.get("user"):
            typer.secho("USER:", fg=typer.colors.GREEN)
            typer.echo(s["user"])
        if s.get("assistant"):
            typer.secho("ASSISTANT:", fg=typer.colors.BLUE)
            typer.echo(s["assistant"])
        if s.get("env"):
            typer.secho("ENVIRONMENT:", fg=typer.colors.MAGENTA)
            typer.echo(s["env"])


@job_app.command("stop", help="停止作业（运行中 → 终止）")
def job_stop(
    job_id: str = typer.Argument(..., help="作业 ID"),
) -> None:
    gate()
    api_client.job_control_via_server("stop", job_id)
    typer.secho("✓ 已停止作业", fg=typer.colors.GREEN)


@job_app.command("pause", help="暂停作业（保留 checkpoint，可继续）")
def job_pause(
    job_id: str = typer.Argument(..., help="作业 ID"),
) -> None:
    gate()
    api_client.job_control_via_server("pause", job_id)
    typer.secho("✓ 已暂停作业（恢复后从最近 checkpoint 继续，最多丢最近 save_period 步）", fg=typer.colors.GREEN)


@job_app.command("resume", help="继续已暂停的作业（自动从最近 checkpoint 续训）")
def job_resume(
    job_id: str = typer.Argument(..., help="作业 ID"),
) -> None:
    gate()
    res = api_client.job_control_via_server("resume", job_id)
    typer.secho(f"✓ {res.get('message') or '已加入恢复队列'}", fg=typer.colors.GREEN)


@job_app.command("delete", help="删除某个已结束的作业记录（运行中需先 stop）")
def job_delete(
    job_id: str = typer.Argument(..., help="作业 ID"),
) -> None:
    gate()
    api_client.job_control_via_server("delete", job_id)
    typer.secho("✓ 已删除记录", fg=typer.colors.GREEN)


@job_app.command("cancel-all", help="停止我所有运行中 / 排队中的作业")
def job_cancel_all(
    yes: bool = typer.Option(False, "-y", "--yes", help="跳过确认"),
) -> None:
    gate()
    if not yes:
        typer.confirm("将停止你【全部】运行中/排队中的作业，确认？", abort=True)
    res = api_client.batch_via_server("cancel-all")
    typer.secho(f"✓ 已停止 {res.get('stopped', 0)} 个作业", fg=typer.colors.GREEN)


@job_app.command("clean", help="清理已结束作业的显示记录")
def job_clean() -> None:
    gate()
    res = api_client.batch_via_server("clean")
    typer.secho(f"✓ 已清理 {res.get('deleted', 0)} 个终态作业记录", fg=typer.colors.GREEN)
