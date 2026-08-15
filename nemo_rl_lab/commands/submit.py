"""作业提交命令：submit / export / eval / clean（构建 JobSpec → 打包 → 经 Console 提交）。"""
from __future__ import annotations

from typing import Optional

import typer

from nemo_rl_lab import api_client, cli_ui, packing
from nemo_rl_lab.auth import gate
from nemo_rl_lab.commands import common
from nemo_rl_lab.commands.exp import _validate_exp


def _require_clean_tree(allow_dirty: bool, prov: dict) -> None:
    """工作区有未提交改动时必须显式 --allow-dirty，保证提交可追溯到确切 commit。"""
    if prov["git_dirty"] and not allow_dirty:
        cli_ui.fail(
            "工作区有未提交改动，提交内容将无法追溯到确切 commit。",
            hint="git commit 后重试；确要带脏改动提交，加 --allow-dirty 显式确认",
        )


def submit(
    exp: str = typer.Argument(..., autocompletion=common.complete_exp, help="实验名或路径"),
    profile: Optional[str] = common.PROF_OPT,
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="实验名称（用于分组展示），不传则默认使用目录名"
    ),
    method: Optional[str] = typer.Option(
        None, "--method", "-m", autocompletion=common.complete_method,
        help="方法标识 <framework>/<method>（如 nemo-rl/grpo、verl/grpo）；"
             "不传则读实验目录 recipe.lock.json 的声明",
    ),
    set_: list[str] = typer.Option(
        [], "--set", "-s", metavar="KEY=VALUE",
        help="覆盖超参，可重复。本地按方法声明校验类型与区间，拼错立刻报错",
    ),
    pool: list[str] = typer.Option(
        [], "--pool", metavar="NAME:SERIES:NODES:GPUS",
        help="资源池，可重复。例：--pool train:h200:1:4 --pool rollout:h100:1:2",
    ),
    role: list[str] = typer.Option(
        [], "--role", metavar="ROLE:POOL", help="角色到资源池的映射，可重复。例：--role rollout:rollout",
    ),
    init_from: Optional[str] = typer.Option(
        None, "--init-from", metavar="run/<RUN_ID>/checkpoint[@step=N]",
        help="从上一阶段的产物起训（SFT → DPO → GRPO 流水线）",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="基座模型路径或 Hub id；verl/TRL 必填"
    ),
    train_data: Optional[str] = typer.Option(
        None, "--train-data", help="训练数据路径；verl/TRL 必填"
    ),
    validation_data: Optional[str] = typer.Option(
        None, "--validation-data", help="验证数据路径；verl/TRL 必填"
    ),
    train_dataset: Optional[str] = typer.Option(
        None, "--train-dataset", metavar="<owner>/<name>[@version]",
        help="平台数据集引用：作业启动时自动拉到共享缓存并注入 <NAME>_DATA_DIR",
    ),
    validation_dataset: Optional[str] = typer.Option(
        None, "--validation-dataset", metavar="<owner>/<name>[@version]",
        help="验证集的平台数据集引用；用法同 --train-dataset",
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        help="仅 custom recipe 使用；一等框架镜像由版本 catalog 精确固定",
    ),
    then: list[str] = typer.Option(
        [], "--then", metavar="ACTION", help="训练成功后自动执行（export/eval），可重复",
    ),
    observability_url: Optional[str] = typer.Option(
        None,
        "--observability-url",
        help="仅 external observability recipe 使用；platform recipe 禁止设置",
    ),
    framework_version: Optional[str] = typer.Option(
        None,
        "--framework-version",
        help="精确框架版本；显式指定会更新 recipe.lock.json",
    ),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="允许工作区有未提交改动（默认拒绝，保证可追溯）"
    ),
    no_validate: bool = typer.Option(False, "--no-validate", help="跳过提交前校验"),
) -> None:
    """提交训练作业（提交前自动校验 config 与超参）。"""
    gate()
    exp_path = common.resolve_exp(exp)
    resolved_profile = common.resolve_profile(exp_path, profile)
    prov = packing.git_provenance(common.ROOT, exp_path)
    _require_clean_tree(allow_dirty, prov)
    if not no_validate:
        errors, _ = _validate_exp(exp_path, method or "")
        if errors:
            cli_ui.emit_error(
                f"config 校验未通过（{len(errors)} 处）",
                items=errors,
                hint="修复后重试，或加 --no-validate 跳过",
            )
            raise typer.Exit(1)

    spec = _build_spec_or_exit(
        exp_path, method=method, project=project, sets=set_, pools=pool, roles=role,
        init_from=init_from, then=then, observability_url=observability_url,
        model=model, train_data=train_data, validation_data=validation_data,
        train_dataset=train_dataset, validation_dataset=validation_dataset,
        framework_version=framework_version,
        image=image,
        provenance=prov,
        validate=not no_validate,
    )
    # 清单式打包 → 上传到中心化服务 → 服务端注入密钥/路径后代理提交（密钥/地址不外泄）。
    with cli_ui.submit_progress() as reporter:
        res = api_client.submit_via_server(
            exp_path, resolved_profile, common.ROOT,
            project=project, reporter=reporter, spec=spec,
        )
    _echo_submit_result(res)


def _build_spec_or_exit(exp_path: str, *, method, project, sets, pools, roles,
                        init_from, then, observability_url=None, model=None,
                        train_data=None, validation_data=None,
                        train_dataset=None, validation_dataset=None,
                        framework_version=None, image=None, provenance=None,
                        validate: bool):
    """构建强 JobSpec；缺少显式 recipe 时立即退出。"""
    from nemo_lab_sdk.contract import SpecError

    from nemo_rl_lab import spec_builder

    recipe = (method or "").strip() or spec_builder.infer_recipe(common.ROOT / exp_path)
    if not recipe:
        cli_ui.emit_error(
            "实验没有声明 recipe",
            hint="加 --method <framework>/<method>，或用 `lab new` 生成 recipe.lock.json；`lab methods` 查看可用值",
        )
        raise typer.Exit(1)
    try:
        from nemo_rl_lab.plugins_lock import read_plugin_lock
        from nemo_rl_lab.recipe_lock import validate_recipe_lock, write_recipe_lock

        locked_version = validate_recipe_lock(common.ROOT / exp_path, recipe)
        selected_version = (framework_version or "").strip() or locked_version
        if selected_version != locked_version:
            # 显式的 --framework-version 是固定锁操作，不是一次性覆盖。
            write_recipe_lock(common.ROOT / exp_path, recipe, selected_version)
        # 实验锁定的平台插件引用随 spec 提交；服务端校验存在性/启停/digest 并注入包体。
        plugin_uses = read_plugin_lock(common.ROOT / exp_path)
        return spec_builder.build_spec(
            exp_path,
            recipe=recipe,
            framework_version=selected_version,
            project=project or "",
            sets=list(sets or []),
            pools=list(pools or []),
            roles=list(roles or []),
            init_from=init_from or "",
            on_success=list(then or []),
            observability_url=observability_url or "",
            base_model=model or "",
            train_data=train_data or "",
            validation_data=validation_data or "",
            train_dataset=train_dataset or "",
            validation_dataset=validation_dataset or "",
            image=image or "",
            provenance=provenance or packing.git_provenance(common.ROOT, exp_path),
            validate=validate,
            plugin_uses=plugin_uses,
        )
    except (SpecError, ValueError) as e:
        cli_ui.emit_error("作业规格校验未通过", items=[str(e)],
                          hint="用 `lab methods` 查看可用方法与超参")
        raise typer.Exit(1) from e


def _echo_submit_result(res: dict, label: str = "") -> None:
    """统一展示提交结果：排队（202，含卡型时段/容量原因）与直接提交两种形态。"""
    gpus = res.get("requested_gpus")
    if res.get("queued"):
        msg = f"⏳ 已排队{label}  run {res.get('run_id')}"
        if gpus is not None:
            msg += f"  ·  {gpus} GPU"
        typer.secho(msg, fg=typer.colors.YELLOW)
        if res.get("message"):
            typer.secho(f"  {res['message']}", fg=typer.colors.BRIGHT_BLACK)
        _echo_upload_summary(res)
        typer.echo("  满足条件后自动提交；查看状态：lab job ls")
        _echo_submit_warnings(res)
        return
    msg = f"✓ 已提交{label}  作业 {res.get('job_id')}"
    if gpus is not None:
        msg += f"  ·  {gpus} GPU"
    if res.get("dry_run"):
        msg += "  ·  预演"
    typer.secho(msg, fg=typer.colors.GREEN)
    _echo_upload_summary(res)
    typer.echo(f"  查看日志：lab job logs {res.get('job_id')}")
    _echo_submit_warnings(res)


def _echo_submit_warnings(res: dict) -> None:
    """服务端下发的 profile 告警：提交受理了，但目标卡型/拓扑与集群实际情况对不上。

    典型是集群里根本没有该卡型（作业会一直 PENDING）。走 stderr，便于在管道里也醒目。
    """
    for w in res.get("warnings") or []:
        cli_ui.emit_warning(str(w))


def _echo_upload_summary(res: dict) -> None:
    """一行灰字汇报本次上传的文件数 / 体积 / 已略过的非负载文件。"""
    files = res.get("upload_files")
    if files is None:
        return
    parts = [f"上传 {files} 个文件", cli_ui.human_bytes(res.get("upload_bytes") or 0)]
    skipped = res.get("upload_skipped") or 0
    if skipped:
        parts.append(f"清单外略过 {skipped} 个文件")
    typer.secho("  " + "  ·  ".join(parts), fg=typer.colors.BRIGHT_BLACK)


# ----------------------------- 训练后闭环（export / eval）-----------------------------
def _submit_post(action: str, exp_path: str, profile: Optional[str], flags: list[str],
                 dry_run: bool, allow_dirty: bool) -> int:
    """构建并预编译训练后强契约，再通过统一 launcher 提交。"""
    from nemo_lab_sdk.frameworks import CompileRequest, compile_launch_plan
    from nemo_lab_sdk.recipes import get_recipe

    from nemo_rl_lab import spec_builder
    from nemo_rl_lab.recipe_lock import validate_recipe_lock

    recipe_name = spec_builder.infer_recipe(common.ROOT / exp_path)
    if not recipe_name:
        cli_ui.emit_error(
            "实验没有声明 recipe",
            hint="实验缺少 recipe.lock.json；用 `lab new` 重建或提交时补 --framework-version 生成",
        )
        return 1
    recipe = get_recipe(recipe_name)
    try:
        validate_recipe_lock(common.ROOT / exp_path, recipe.name)
    except ValueError as exc:
        cli_ui.emit_error("实验 recipe 锁校验失败", items=[str(exc)])
        return 1
    if not recipe.supports(action):
        cli_ui.emit_error(
            f"recipe {recipe.name} 不支持 {action}",
            hint=f"支持：{', '.join(recipe.lifecycle) or '（无）'}",
        )
        return 1
    series = common.resolve_profile(exp_path, profile)
    prov = packing.git_provenance(common.ROOT, exp_path)
    if not dry_run:
        _require_clean_tree(allow_dirty, prov)
    try:
        spec = spec_builder.build_spec(
            exp_path,
            recipe=recipe.name,
            pools=[f"lifecycle:{series}:1:{recipe.lifecycle_resources[action]}"],
            provenance=prov,
            operation=action,
        )
        plan = compile_launch_plan(CompileRequest(
            operation=action,
            spec=spec,
            recipe=recipe,
            work_dir=common.ROOT,
            env={"NEMOLAB_ENABLED": "0", "OUTPUT_ROOT": "/tmp/nemo-lab-dry-run"},
            action_args=tuple(flags),
        ))
    except (ValueError, OSError) as exc:
        cli_ui.emit_error("训练后作业规格校验未通过", items=[str(exc)])
        return 1
    if dry_run:
        typer.echo(" ".join(plan.argv))
        return 0
    gate()
    with cli_ui.submit_progress() as reporter:
        res = api_client.submit_post_via_server(
            action, exp_path, series, flags, common.ROOT, reporter=reporter, spec=spec
        )
    _echo_submit_result(res, label="导出" if action == "export" else "评测")
    return 0


def export_ckpt(
    exp: str = typer.Argument(..., autocompletion=common.complete_exp, help="实验名或路径"),
    checkpoint: str = typer.Option(..., "--checkpoint", help="artifact registry 中记录的 checkpoint 路径"),
    checkpoint_format: str = typer.Option(
        ..., "--checkpoint-format",
        help="nemo-dcp | nemo-megatron | verl-fsdp | verl-megatron | huggingface",
    ),
    push_repo: Optional[str] = typer.Option(None, "--push-repo", help="转换后上传到 HF Hub repo（user/name，需 HF_TOKEN）"),
    profile: Optional[str] = common.PROF_OPT,
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="允许工作区有未提交改动"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将提交的命令，不实际提交"),
) -> None:
    """将 checkpoint 转为 HuggingFace 格式（可推 Hub）。"""
    flags = ["--checkpoint", checkpoint, "--checkpoint-format", checkpoint_format]
    if push_repo:
        flags += ["--push-repo", push_repo]
    raise typer.Exit(_submit_post("export", common.resolve_exp(exp), profile, flags, dry_run, allow_dirty))


def eval_ckpt(
    ctx: typer.Context,
    exp: str = typer.Argument(..., autocompletion=common.complete_exp, help="实验名或路径"),
    model: Optional[str] = typer.Option(None, "--model", help="NeMo-RL：HF 模型路径/Hub id"),
    eval_config: Optional[str] = typer.Option(None, "--eval-config", help="NeMo-RL：显式评测配置路径"),
    data: Optional[str] = typer.Option(None, "--data", help="verl：显式评测数据路径"),
    profile: Optional[str] = common.PROF_OPT,
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="允许工作区有未提交改动"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将提交的命令，不实际提交"),
) -> None:
    """按 recipe 的原生评测入口执行；NeMo-RL 用 --model/--eval-config，verl 用 --data。"""
    flags: list[str] = []
    if model:
        flags += ["--model", model]
    if eval_config:
        flags += ["--eval-config", eval_config]
    if data:
        flags += ["--data", data]
    extra = list(ctx.args)  # `--` 之后透传给 run_eval.py 的覆盖项
    if extra:
        flags += ["--", *extra]
    raise typer.Exit(_submit_post("eval", common.resolve_exp(exp), profile, flags, dry_run, allow_dirty))


def clean(
    exp: str = typer.Argument(..., autocompletion=common.complete_exp, help="实验名或路径"),
    yes: bool = typer.Option(False, "-y", "--yes", help="跳过确认"),
) -> None:
    """清理实验在集群上的 checkpoint 与日志（不可恢复）。"""
    gate()
    exp_path = common.resolve_exp(exp)
    if not yes:
        typer.confirm(
            f"将删除 {exp_path} 在集群上的训练产物，不可恢复。继续？",
            abort=True,
        )
    res = api_client.clean_via_server(exp_path)
    typer.secho(f"✓ 已提交清理  作业 {res.get('job_id')}", fg=typer.colors.GREEN)
    typer.echo(f"  查看进度：lab job logs {res.get('job_id')}")
