"""实验资产命令：ls / new / validate / methods（纯本地 + SDK catalog，不联网）。"""
from __future__ import annotations

from typing import Optional

import typer

from nemo_rl_lab import cli_ui
from nemo_rl_lab.commands import common
from nemo_rl_lab.new_experiment import NewExperimentError, create_experiment


def ls() -> None:
    """列出实验 / 项目。"""
    for kind in ("experiments", "projects"):
        base = common.ROOT / kind
        if not base.is_dir():
            continue
        exps = sorted(p.name for p in base.iterdir() if p.is_dir())
        typer.echo(f"\n[{kind}] ({len(exps)})")
        for e in exps:
            typer.echo(f"  - {e}")


def new(
    name: str = typer.Argument(..., help="实验名"),
    from_exp: Optional[str] = typer.Option(
        None, "--from", autocompletion=common.complete_exp,
        help="从已有实验 fork",
    ),
    method: str = typer.Option(
        "nemo-rl/grpo", "--method", "-m", autocompletion=common.complete_method,
        help="方法标识 <framework>/<method>；默认 nemo-rl/grpo，`lab methods` 查看全部",
    ),
    framework_version: Optional[str] = typer.Option(
        None,
        "--framework-version",
        help="精确框架版本；必须是 recipe catalog 已发布版本",
    ),
    cluster: Optional[str] = typer.Option(
        None, "--cluster", autocompletion=common.complete_profile,
        help="目标集群 profile",
    ),
    kind: common.Kind = typer.Option(
        common.Kind.experiments, "--kind", help="experiments 或 projects"
    ),
) -> None:
    """新建实验（--from fork 现成实验；--method 来自 SDK recipe catalog）。"""
    if from_exp and method != "nemo-rl/grpo":
        typer.secho("fork 会继承来源实验配置，--method 已忽略。", fg=typer.colors.YELLOW)
    src = ""
    if from_exp:
        from pathlib import Path

        src = Path(common.resolve_exp(from_exp)).name
    try:
        create_experiment(
            common.ROOT,
            kind.value,
            name,
            src=src,
            cluster=cluster or "",
            method=method,
            framework_version=framework_version or "",
        )
    except NewExperimentError as e:
        cli_ui.fail(str(e))


def _validate_exp(exp_path: str, recipe_override: str = "") -> tuple[list[str], list[str]]:
    """只运行 recipe 所属框架的 validator，不跨框架猜测。"""
    import yaml
    from nemo_lab_sdk.contract import SpecError
    from nemo_lab_sdk.recipes import get_recipe

    from nemo_rl_lab.config_resolve import resolve, validate_framework_config
    from nemo_rl_lab.recipe_lock import validate_recipe_lock
    from nemo_rl_lab.spec_builder import infer_recipe

    exp_dir = common.ROOT / exp_path
    recipe_name = recipe_override.strip() or infer_recipe(exp_dir)
    if not recipe_name:
        return [f"实验缺少 method recipe 声明: {exp_path}"], []
    try:
        recipe = get_recipe(recipe_name)
        validate_recipe_lock(exp_dir, recipe_name)
    except (SpecError, ValueError) as exc:
        return [str(exc)], []

    if recipe.entrypoint.kind == "experiment":
        entry = (exp_dir / recipe.entrypoint.value).resolve()
        if not entry.is_relative_to(exp_dir.resolve()) or not entry.is_file():
            return [f"{recipe.framework} 实验缺少入口: {recipe.entrypoint.value}"], []
    if recipe.framework == "custom":
        return [], []

    cfg_file = exp_dir / "config.yaml"
    if not cfg_file.is_file():
        return [f"{recipe.framework} 实验缺少 config.yaml: {exp_path}"], []
    try:
        if recipe.framework == "nemo-rl":
            cfg = resolve(cfg_file)   # defaults 继承合并
        else:
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"解析 {recipe.framework} config 失败: {exc}"], []
    if not isinstance(cfg, dict):
        return [f"{recipe.framework} config 根节点必须是对象"], []

    # 与服务端提交入口同一份 SDK validator：本地过了，服务端就不会因同类问题 422。
    try:
        issues = validate_framework_config(
            recipe.framework, cfg, repo_root=common.ROOT, recipe=recipe
        )
    except ValueError as exc:
        return [str(exc)], []
    return (
        [message for level, message in issues if level == "error"],
        [message for level, message in issues if level == "warn"],
    )


def validate(
    exp: str = typer.Argument(..., autocompletion=common.complete_exp, help="实验名或路径"),
) -> None:
    """校验实验 config（提交前本地检查）。"""
    exp_path = common.resolve_exp(exp)
    errors, warns = _validate_exp(exp_path)
    if errors:
        cli_ui.emit_error(
            f"{exp_path}：{len(errors)} 处错误" + (f"，{len(warns)} 处告警" if warns else ""),
            items=errors,
        )
        raise typer.Exit(1)
    if warns:
        cli_ui.emit_warning(f"{exp_path}：{len(warns)} 处告警", body="\n".join(f"• {w}" for w in warns))
    suffix = f"（{len(warns)} 个告警）" if warns else ""
    typer.secho(f"✓ {exp_path}：通过{suffix}", fg=typer.colors.GREEN)


def methods(
    name: Optional[str] = typer.Argument(None, help="方法名；不传则列出全部"),
) -> None:
    """列出可用的后训练方法与它们的超参。

    方法目录来自 nemo-lab-sdk，与服务端是同一份 —— 这里看到的就是提交时会被校验的。
    """
    from nemo_lab_sdk.contract import SpecError
    from nemo_lab_sdk.recipes import get_recipe, recipe_names

    if not name:
        for n in recipe_names():
            r = get_recipe(n)
            typer.echo(f"{n:22s} {r.title}")
            typer.echo(
                f"{'':22s} 默认 {r.framework}@{r.runtime.default_version}"
                f" · 支持 {', '.join(r.runtime.supported_versions)}"
            )
            typer.echo(f"{'':22s} {r.summary.strip()}")
        typer.echo("\n用 `lab methods <方法名>` 看它的可调超参。")
        return

    try:
        r = get_recipe(name)
    except SpecError as e:
        cli_ui.emit_error(str(e))
        raise typer.Exit(1) from e

    typer.echo(f"{r.id} v{r.version} —— {r.title}")
    typer.echo(f"  {r.summary.strip()}\n")
    typer.echo(f"  框架      : {r.framework}@{r.runtime.default_version}（默认）")
    typer.echo(f"  支持版本  : {', '.join(r.runtime.supported_versions)}")
    typer.echo(f"  角色      : {', '.join(r.roles)}")
    typer.echo(f"  训练后动作: {', '.join(r.lifecycle) or '（无）'}")
    if r.plugins:
        typer.echo(f"  算法插件  : {', '.join(r.plugins)}")
    if r.runtime.requires:
        typer.echo(f"  默认依赖  : {', '.join(r.runtime.requires)}")
    typer.echo(f"  核心指标  : {', '.join(r.primary_metrics)}\n")
    typer.echo("  可调超参（--set KEY=VALUE）:")
    for p in r.params.values():
        rng = []
        if p.minimum is not None:
            rng.append(f"≥{p.minimum}" if not p.exclusive_minimum else f">{p.minimum}")
        if p.maximum is not None:
            rng.append(f"≤{p.maximum}")
        if p.choices:
            rng.append("|".join(str(c) for c in p.choices))
        meta = f"{p.type}{' ' + ','.join(rng) if rng else ''}"
        default = f" (默认 {p.default})" if p.default is not None else ""
        typer.echo(f"    {p.name:32s} {meta}{default}")
        if p.doc:
            typer.echo(f"    {'':32s} {p.doc.strip()}")
