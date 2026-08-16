"""starforge 统一 CLI：app 组装（命令实现见 starforge_cli/commands/）。

命令面即架构：CLI 是 Console 的瘦客户端，只做四件事——
  身份        login / logout
  实验资产    ls / new / methods / validate（纯本地 + SDK catalog）
  作业契约    submit / export / eval / clean（JobSpec + 清单式打包，经 Console）
  观测控制    status / job *（经 Console；集群细节客户端不可见）
外加 dataset *（数据生命周期）、plugin *（插件中心）与 admin *（管理员）三个子组。
"""
from __future__ import annotations

import typer

from starforge_cli.commands import (
    admin,
    bench,
    dataset,
    exp,
    jobs,
    login,
    plugin,
    recipe,
    serve,
    submit,
    sweep,
)

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="StarForge（星锻）· 大模型后训练平台 CLI",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# ----------------------------- 身份 -----------------------------
app.command(help="登录 Lab")(login.login)
app.command(help="登出")(login.logout)

# ----------------------------- 实验资产（纯本地）-----------------------------
app.command(help="列出实验 / 项目")(exp.ls)
app.command(help="新建实验（--from fork 现成实验；--method 来自 SDK recipe catalog）")(exp.new)
app.command(name="methods", help="列出可用的后训练方法与它们的超参")(exp.methods)
app.command(help="校验实验 config（提交前本地检查）")(exp.validate)

# ----------------------------- 作业契约（经 Console）-----------------------------
app.command(help="提交训练作业（提交前自动校验 config 与超参）")(submit.submit)
app.command(help="超参 sweep：网格展开批量提交（每变体一次标准提交，配额/排队照常生效）")(sweep.sweep)
app.command(help="标准基准评测（lm-eval / evalscope），分数入库平台看板")(bench.bench)
app.command(name="export", help="将 checkpoint 转为 HuggingFace 格式（可推 Hub）")(submit.export_ckpt)
app.command(
    name="eval",
    help="按 recipe 的原生评测入口执行；NeMo-RL 用 --model/--eval-config，verl 用 --data",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(submit.eval_ckpt)
app.command(help="清理实验在集群上的 checkpoint 与日志（不可恢复）")(submit.clean)

# ----------------------------- 观测控制（经 Console）-----------------------------
app.command(help="账号、配额、用量与活跃作业")(jobs.status)
app.add_typer(jobs.job_app, name="job")

# ----------------------------- 数据集 / 插件 / 管理员 -----------------------------
app.add_typer(recipe.recipe_app, name="recipe")
app.add_typer(serve.serve_app, name="serve")
app.add_typer(dataset.dataset_app, name="dataset")
app.add_typer(plugin.plugin_app, name="plugin")
app.add_typer(admin.admin_app, name="admin")


if __name__ == "__main__":
    app()
