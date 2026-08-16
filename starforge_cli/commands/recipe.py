"""recipe 锁状态与显式升级。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from starforge_cli import cli_ui
from starforge_cli.commands import common
from starforge_cli.recipe_lock import (
    LOCK_FILE,
    LockInspection,
    RecipeLockBatchError,
    RecipeLockError,
    RecipeLockManager,
    iter_lock_dirs,
)

recipe_app = typer.Typer(
    no_args_is_help=True,
    help="recipe 锁管理",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _rel(path: Path) -> str:
    try:
        return str(path.parent.relative_to(common.ROOT))
    except ValueError:
        return str(path.parent)


def _print_inspection(item: LockInspection) -> None:
    typer.echo(f"{_rel(item.path)}")
    typer.echo(f"  state: {item.state.value}")
    if item.recipe_name:
        fw = f"  framework: {item.framework_version}" if item.framework_version else ""
        typer.echo(f"  recipe: {item.recipe_name}{fw}")
    for diff in item.diffs:
        typer.echo(f"  - {diff.field}: {diff.locked or '∅'} → {diff.current or '∅'}")
    if not item.is_current:
        typer.echo(f"  hint: sf recipe upgrade {_rel(item.path)}")


def _config_errors(exp_dir: Path, recipe) -> list[str]:
    from starforge_cli.commands.exp import validate_exp_config

    return validate_exp_config(exp_dir, recipe, repo_root=common.ROOT)


@recipe_app.command("status", help="查看实验 recipe 锁与当前 catalog 的差异")
def recipe_status(
    exp: Optional[str] = typer.Argument(
        None, autocompletion=common.complete_exp, help="实验名或路径；省略则需 --all"
    ),
    all_exps: bool = typer.Option(False, "--all", help="扫描 experiments/projects/smoke"),
    server: bool = typer.Option(False, "--server", help="再与 Console catalog 握手对照"),
) -> None:
    manager = RecipeLockManager()
    if all_exps:
        items = [manager.inspect(path) for path in iter_lock_dirs(common.ROOT)]
    elif exp:
        items = [manager.inspect(common.ROOT / common.resolve_exp(exp))]
    else:
        cli_ui.fail("指定实验名，或加 --all")

    stale = 0
    for item in items:
        _print_inspection(item)
        if not item.is_current:
            stale += 1
        if server and item.expected is not None:
            _echo_server_drift(item)
    if stale:
        raise typer.Exit(1)


def _echo_server_drift(item: LockInspection) -> None:
    from starforge_cli import api_client

    if item.expected is None:
        return
    try:
        payload = api_client.api_get("/api/recipes")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"  server: {exc}")
        return
    selected = next(
        (
            recipe
            for recipe in (payload.get("recipes") or [])
            if recipe.get("name") == item.recipe_name
        ),
        None,
    )
    if selected is None:
        typer.echo(f"  server: Console 未启用 {item.recipe_name}")
        return
    wanted = item.expected["recipe"]
    drift = [
        f"{field} server={selected.get(field)!r} local={wanted.get(field)!r}"
        for field in ("version", "digest")
        if selected.get(field) != wanted.get(field)
    ]
    if drift:
        typer.echo("  server: " + "; ".join(drift))
        return
    typer.echo("  server: current")


@recipe_app.command("upgrade", help="把实验锁升级到当前 catalog（默认不改 framework version）")
def recipe_upgrade(
    exp: Optional[str] = typer.Argument(
        None, autocompletion=common.complete_exp, help="实验名或路径；与 --all 互斥"
    ),
    all_exps: bool = typer.Option(False, "--all", help="先全量校验，再统一写入"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只显示 diff，不写文件"),
    framework_version: Optional[str] = typer.Option(
        None, "--framework-version", help="升级时改到指定精确框架版本"
    ),
    accept_runtime_change: bool = typer.Option(
        False,
        "--accept-runtime-change",
        help="锁定的框架版本已从 catalog 移除时，改用当前默认版本",
    ),
) -> None:
    manager = RecipeLockManager()
    selected = (framework_version or "").strip()
    try:
        if all_exps:
            items = manager.upgrade_all(
                common.ROOT,
                framework_version=selected,
                accept_runtime_change=accept_runtime_change,
                dry_run=dry_run,
                validate_config=_config_errors,
            )
        elif exp:
            items = [
                manager.upgrade(
                    common.ROOT / common.resolve_exp(exp),
                    framework_version=selected,
                    accept_runtime_change=accept_runtime_change,
                    dry_run=dry_run,
                    validate_config=_config_errors,
                )
            ]
        else:
            cli_ui.fail("指定实验名，或加 --all")
    except RecipeLockBatchError as exc:
        cli_ui.emit_error(str(exc), items=[item.message for item in exc.failures])
        raise typer.Exit(1) from exc
    except RecipeLockError as exc:
        cli_ui.emit_error(exc.inspection.message)
        raise typer.Exit(1) from exc

    for item in items:
        _print_inspection(item)
    typer.secho(
        f"{'将' if dry_run else '已'}处理 {len(items)} 个 {LOCK_FILE}",
        fg=typer.colors.GREEN,
    )
