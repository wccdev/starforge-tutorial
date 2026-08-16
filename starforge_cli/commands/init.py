"""`sf init`：交互式创建 StarForge 微调项目。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from starforge_cli import cli_ui
from starforge_cli.project import InitError, init_project


def init(
    directory: Optional[str] = typer.Argument(
        None, help="项目目录（省略则交互询问；. 表示当前目录）"
    ),
    name: Optional[str] = typer.Option(None, "--name", help="项目名（默认目录名）"),
    no_git: bool = typer.Option(False, "--no-git", help="不初始化 git 仓库"),
    yes: bool = typer.Option(False, "--yes", "-y", help="非交互：全部取默认值"),
) -> None:
    """创建 StarForge 项目：experiments/ + configs/（官方基底）+ common/ 骨架。

    项目是独立的 git 仓库——实验配置与共享代码归你和团队所有，
    与 CLI 工具的版本解耦（升级 CLI 只需 pip install -U starforge-cli）。
    """
    target = (directory or "").strip()
    if not target and not yes:
        target = typer.prompt("项目目录", default="my-starforge-lab")
    if not target:
        target = "my-starforge-lab"
    dest = Path(target).expanduser()

    project_name = (name or "").strip() or (dest.resolve().name if target != "." else Path.cwd().name)
    if not yes and not name:
        project_name = typer.prompt("项目名", default=project_name)

    use_git = not no_git
    if not yes and not no_git:
        use_git = typer.confirm("初始化 git 仓库？", default=True)

    try:
        root = init_project(dest if target != "." else Path.cwd(), name=project_name, git=use_git)
    except InitError as e:
        cli_ui.fail(str(e))

    typer.secho(f"✓ StarForge 项目已创建：{root}", fg=typer.colors.GREEN)
    typer.echo("下一步：")
    if target not in (".",):
        typer.echo(f"  cd {target}")
    typer.echo("  sf login --server https://<你的 StarForge 域名>   # 首次")
    typer.echo("  sf new my-exp --method nemo-rl/grpo")
    typer.echo("  sf submit my-exp --profile <卡型:卡数>")
