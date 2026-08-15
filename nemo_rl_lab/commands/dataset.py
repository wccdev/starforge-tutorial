"""数据集生命周期：prepare（本地预处理）→ push（上传对象存储）→ ls（查看版本）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from nemo_rl_lab import api_client, cli_ui
from nemo_rl_lab.auth import gate
from nemo_rl_lab.commands import common

dataset_app = typer.Typer(
    no_args_is_help=True,
    help="数据集：本地预处理、上传到对象存储、按版本分发给作业",
    context_settings={"help_option_names": ["-h", "--help"]},
)

DATA_PREP = {
    "gsm8k": "prepare_gsm8k.py",
    "alpaca": "prepare_alpaca.py",
    "qa_rl": "prepare_qa_rl.py",
    "opsd_math": "prepare_opsd_math.py",
}


def _complete_dataset(incomplete: str) -> list[str]:
    return [d for d in sorted(DATA_PREP) if d.startswith(incomplete)]


@dataset_app.command(
    "prepare",
    help="本地预处理数据集（gsm8k / alpaca / qa_rl / opsd_math）",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def dataset_prepare(
    ctx: typer.Context,
    dataset: str = typer.Argument(..., autocompletion=_complete_dataset, help="数据集名"),
) -> None:
    import os
    import subprocess

    script_name = DATA_PREP.get(dataset)
    if not script_name:
        cli_ui.fail(f"未知数据集「{dataset}」", hint=f"可选：{', '.join(DATA_PREP)}")
    script = common.ROOT / "common" / "data" / script_name
    if not script.is_file():
        cli_ui.fail(f"缺少脚本 {script}", hint="仓库可能不完整，检查 common/data/ 目录")
    cmd = [sys.executable, str(script), *ctx.args]
    typer.echo("› " + " ".join(cmd))
    # 用当前解释器（项目 uv 环境，含 datasets）跑数据脚本。
    raise typer.Exit(
        subprocess.run(cmd, env=os.environ.copy(), cwd=str(common.ROOT)).returncode
    )


@dataset_app.command("ls", help="列出可见的数据集（公开的 + 自己的）")
def dataset_ls(
    dataset: Optional[str] = typer.Argument(
        None, help="数据集 ID（<owner>/<name>）；不传则列全部可见的"
    ),
    version: str = typer.Option("", "--version", "-v", help="看某个版本的文件清单；默认最新"),
) -> None:
    gate()
    if dataset:
        if "/" not in dataset:
            cli_ui.fail(
                f"数据集 ID 必须是 <owner>/<name>，得到 {dataset!r}",
                hint="用 `lab dataset ls` 查看完整 ID",
            )
        owner, _, name = dataset.partition("/")
        q = f"?version={version}" if version else ""
        d = api_client.api_get(f"/api/datasets/{owner}/{name}{q}")
        vis = "公开" if d["visibility"] == "public" else "私有"
        typer.echo(
            f"{d['id']}@{d['version']}  [{vis}]  "
            f"{d['total_bytes'] / 1e6:.1f} MB  {len(d['files'])} 个文件"
        )
        for f in d["files"]:
            typer.echo(f"  {f['name']:40s} {f['size'] / 1e6:8.2f} MB  {(f.get('sha256') or '')[:12]}")
        return
    rows = api_client.api_get("/api/datasets")["datasets"]
    if not rows:
        typer.echo("（还没有可见的数据集；`lab dataset push` 上传一个）")
        return
    for d in rows:
        vis = "公开" if d["visibility"] == "public" else "私有"
        typer.echo(f"{d['id']:32s} [{vis}] {', '.join(d['versions']) or '（无版本）'}")


@dataset_app.command("push", help="上传一个数据集版本（默认私有，--public 公开）")
def dataset_push(
    dataset: str = typer.Argument(
        ..., help="数据集名（归到自己命名空间），或完整 <owner>/<name>（admin 可跨命名空间）"
    ),
    version: str = typer.Argument(..., help="版本，如 v1 / 20260812"),
    path: str = typer.Argument(..., help="本地目录"),
    public: bool = typer.Option(
        False, "--public", help="首次创建时设为公开（所有人可在训练中引用）"
    ),
) -> None:
    """上传目录下的全部文件，并生成带 sha256 的 index.json。

    校验和不是可选项：作业侧靠它判断下载是否被截断。静默接受一个截断的
    train.jsonl，就是拿脏数据训练几小时后才发现结果不对。

    版本不可变：已完整上传的版本不能覆写，要更新请换新版本号。
    """
    gate()
    root = Path(path)
    if not root.is_dir():
        cli_ui.fail(f"不是目录: {path}")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        cli_ui.fail(f"目录为空: {path}")
    ds_id = api_client.dataset_push(
        dataset, version, root, files, visibility="public" if public else None,
    )
    short = ds_id.rpartition("/")[2]
    typer.echo(f"已上传 {ds_id}@{version}（{len(files)} 个文件）")
    typer.echo(
        f"训练里引用它：实验 config 里 data.train.dataset: {ds_id}@{version}"
        f"，数据目录环境变量 ${{oc.env:{short.upper().replace('-', '_')}_DATA_DIR}}"
    )


@dataset_app.command("visibility", help="改数据集可见性（owner 或 admin）")
def dataset_visibility(
    dataset: str = typer.Argument(..., help="数据集 ID（<owner>/<name>）"),
    public: bool = typer.Option(False, "--public", help="设为公开"),
    private: bool = typer.Option(False, "--private", help="设为私有"),
) -> None:
    gate()
    if public == private:
        cli_ui.fail("必须且只能指定 --public 或 --private 之一")
    if "/" not in dataset:
        cli_ui.fail(
            f"数据集 ID 必须是 <owner>/<name>，得到 {dataset!r}",
            hint="用 `lab dataset ls` 查看完整 ID",
        )
    owner, _, name = dataset.partition("/")
    vis = "public" if public else "private"
    r = api_client.api_patch(f"/api/datasets/{owner}/{name}", {"visibility": vis})
    typer.echo(f"{r['id']} 现在是{'公开' if r['visibility'] == 'public' else '私有'}数据集")
