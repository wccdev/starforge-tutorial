"""超参 sweep 批量提交：网格展开 → N 个变体逐个走标准提交链路。

设计取舍：
  - **不造第二条提交路径**：每个变体就是一次普通 `forge submit`（同一份 spec 构建、
    校验、打包、配额准入、容量排队），服务端零特殊逻辑。sweep 只是「展开 + 循环 +
    分组标识」。
  - 分组复用现成机制：统一 --project（web 实验分组页天然聚合）+
    client_meta.sweep_id / sweep_params（服务端落 jobs.sweep_id 索引列，
    一键停止 / 徽标展示用）。
  - 配额不足的变体自动入队（persist_and_enqueue），不会失败——批量提交天然
    被配额与容量闸门限流。
"""
from __future__ import annotations

import itertools
import time
from typing import Optional

import typer

from starforge_cli import api_client, cli_ui, packing
from starforge_cli.auth import gate
from starforge_cli.commands import common
from starforge_cli.commands.exp import _validate_exp
from starforge_cli.commands.submit import (
    _build_spec_or_exit,
    _dataset_refs_from_config,
    _echo_submit_result,
    _materialize_profile_or_exit,
    _require_clean_tree,
)

#: 变体数上限：防止一个手滑的网格把整个集群队列塞满。
MAX_VARIANTS = 64


def parse_grid(exprs: list[str]) -> list[dict[str, str]]:
    """把 --set-grid "key=v1,v2" 列表展开为变体字典列表（笛卡尔积，顺序稳定）。"""
    axes: list[tuple[str, list[str]]] = []
    for raw in exprs:
        expr = (raw or "").strip()
        if "=" not in expr:
            raise ValueError(f"--set-grid 格式应为 key=v1,v2：{raw!r}")
        key, _, values_raw = expr.partition("=")
        key = key.strip()
        values = [v.strip() for v in values_raw.split(",") if v.strip()]
        if not key or not values:
            raise ValueError(f"--set-grid 缺少键或值：{raw!r}")
        if key in (k for k, _ in axes):
            raise ValueError(f"--set-grid 键重复：{key}")
        axes.append((key, values))
    if not axes:
        return []
    combos = itertools.product(*(values for _, values in axes))
    return [
        {key: value for (key, _), value in zip(axes, combo, strict=True)}
        for combo in combos
    ]


def sweep(
    exp: str = typer.Argument(..., autocompletion=common.complete_exp, help="实验名或路径"),
    set_grid: list[str] = typer.Option(
        ..., "--set-grid", "-g", metavar="KEY=V1,V2",
        help="网格轴，可重复；多轴取笛卡尔积。如 -g policy.lr=1e-5,2e-5 -g grpo.kl=0.01,0.05",
    ),
    set_: list[str] = typer.Option(
        [], "--set", "-s", metavar="KEY=VALUE", help="所有变体共用的固定覆盖，可重复",
    ),
    profile: list[str] = common.PROFILE_EXPR_OPT,
    project: Optional[str] = typer.Option(
        None, "--project", "-p",
        help="分组名（web 实验分组页按它聚合）；默认 sweep-<实验名>-<时间戳>",
    ),
    method: Optional[str] = typer.Option(
        None, "--method", "-m", autocompletion=common.complete_method,
        help="方法标识；不传则读实验 recipe.lock.json",
    ),
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="允许工作区有未提交改动"),
    no_validate: bool = typer.Option(False, "--no-validate", help="跳过提交前校验"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印变体列表，不提交"),
) -> None:
    """网格展开超参并批量提交（每个变体 = 一次标准提交，配额/排队照常生效）。"""
    try:
        variants = parse_grid(set_grid)
    except ValueError as e:
        cli_ui.fail(str(e))
    if not variants:
        cli_ui.fail("--set-grid 没有产生任何变体")
    if len(variants) > MAX_VARIANTS:
        cli_ui.fail(
            f"网格展开出 {len(variants)} 个变体，超过上限 {MAX_VARIANTS}",
            hint="缩小网格，或分多个 sweep 提交",
        )

    exp_path = common.resolve_exp(exp)
    exp_short = exp_path.rstrip("/").split("/")[-1]
    sweep_id = f"sweep-{exp_short}-{time.strftime('%Y%m%d-%H%M%S')}"
    group = (project or "").strip() or sweep_id

    typer.secho(f"sweep {sweep_id}：{len(variants)} 个变体", fg=typer.colors.CYAN, bold=True)
    for i, variant in enumerate(variants):
        typer.echo(f"  [{i + 1:>2}] " + "  ".join(f"{k}={v}" for k, v in variant.items()))
    if dry_run:
        typer.echo("（--dry-run：未提交）")
        return

    gate()
    exprs = [e for e in (profile or []) if e.strip()]
    if not exprs:
        exprs = [common.resolve_profile(exp_path, None)]
    resolved_profile, pools, roles = _materialize_profile_or_exit(exprs)
    prov = packing.git_provenance(common.ROOT, exp_path)
    _require_clean_tree(allow_dirty, prov)
    if not no_validate:
        errors, _ = _validate_exp(exp_path, method or "")
        if errors:
            cli_ui.emit_error(
                f"config 校验未通过（{len(errors)} 处）", items=errors,
                hint="修复后重试，或加 --no-validate 跳过",
            )
            raise typer.Exit(1)
    cfg_train_ds, cfg_val_ds = _dataset_refs_from_config(exp_path)

    fixed = list(set_ or [])
    submitted = 0
    queued = 0
    for i, variant in enumerate(variants):
        variant_sets = fixed + [f"{k}={v}" for k, v in variant.items()]
        typer.secho(
            f"\n[{i + 1}/{len(variants)}] " + "  ".join(f"{k}={v}" for k, v in variant.items()),
            fg=typer.colors.CYAN,
        )
        spec = _build_spec_or_exit(
            exp_path, method=method, project=group, sets=variant_sets,
            pools=pools, roles=roles, init_from=None, then=[],
            train_dataset=cfg_train_ds, validation_dataset=cfg_val_ds,
            provenance=prov, validate=not no_validate,
        )
        with cli_ui.submit_progress() as reporter:
            res = api_client.submit_via_server(
                exp_path, resolved_profile, common.ROOT,
                project=group, reporter=reporter, spec=spec,
                extra_meta={"sweep_id": sweep_id, "sweep_params": variant},
            )
        _echo_submit_result(res)
        if res.get("queued"):
            queued += 1
        else:
            submitted += 1

    typer.echo("")
    typer.secho(
        f"sweep {sweep_id} 完成：直接提交 {submitted} 个，入队 {queued} 个。",
        fg=typer.colors.GREEN, bold=True,
    )
    typer.echo(f"  查看进度：forge job ls --all（分组名 {group}）")
    typer.echo(f"  一键停止：forge job stop-sweep {sweep_id}")
