"""标准基准评测提交：`forge bench <exp> --model … --suites …`。

评测作业就是一次普通提交（evalkit/benchmark recipe，同一条配额/排队/产物链路）；
本命令只做三件事：解析模型引用（`run:<run_id>` → 该 run 的 hf_export 产物路径）、
把评测参数翻成超参、走标准提交。分数自动入库平台 benchmark 看板。
"""
from __future__ import annotations

import json
import urllib.error
from typing import Optional

import typer

from starforge_cli import api_client, cli_ui, packing
from starforge_cli.auth import gate
from starforge_cli.commands import common
from starforge_cli.commands.submit import (
    _build_spec_or_exit,
    _echo_submit_result,
    _materialize_profile_or_exit,
    _require_clean_tree,
)


def _resolve_model_ref(model: str) -> str:
    """`run:<run_id>` → 该 run 已登记的 hf_export 产物路径；其余原样返回。"""
    ref = (model or "").strip()
    if not ref.startswith("run:"):
        return ref
    run_id = ref[len("run:"):].strip()
    if not run_id:
        cli_ui.fail("run: 引用缺少 run_id（形如 run:grpo-alice-20260101-120000）")
    try:
        with api_client._bearer_request(
            api_client.current_server(None), "GET",
            f"/api/jobs/{run_id}/artifacts",
        ) as r:
            data = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        cli_ui.fail_http(e, fallback=f"查询 run {run_id} 产物失败")
    exports = [a for a in data.get("artifacts") or [] if a.get("kind") == "hf_export"]
    if not exports:
        cli_ui.fail(
            f"run {run_id} 没有 hf_export 产物",
            hint=f"先导出：forge export {run_id}",
        )
    return str(exports[-1]["path"])


def bench(
    exp: str = typer.Argument(..., autocompletion=common.complete_exp,
                              help="评测实验（forge new <名字> --method evalkit/benchmark 创建）"),
    model: str = typer.Option(..., "--model", "-m",
                              help="HF 模型 id / 共享盘绝对路径 / run:<run_id>（取该 run 的 hf_export）"),
    suites: str = typer.Option(..., "--suites",
                               help="逗号分隔的基准名（如 gsm8k,mmlu / ceval）"),
    runner: str = typer.Option("lm-eval", "--runner", help="评测后端：lm-eval | evalscope"),
    limit: Optional[int] = typer.Option(None, "--limit", help="每基准样本上限（冒烟用）"),
    extra_args: str = typer.Option("", "--extra-args", help="透传给 runner 的附加参数"),
    profile: list[str] = common.PROFILE_EXPR_OPT,
    project: Optional[str] = typer.Option(None, "--project", "-p", help="分组名（默认实验名）"),
    allow_dirty: bool = typer.Option(False, "--allow-dirty", help="允许工作区有未提交改动"),
) -> None:
    """提交基准评测作业（lm-eval / evalscope），分数自动入库平台看板。"""
    gate()
    exp_path = common.resolve_exp(exp)
    resolved_model = _resolve_model_ref(model)

    exprs = [e for e in (profile or []) if e.strip()]
    if not exprs:
        exprs = [common.resolve_profile(exp_path, None)]
    resolved_profile, pools, roles = _materialize_profile_or_exit(exprs)
    prov = packing.git_provenance(common.ROOT, exp_path)
    _require_clean_tree(allow_dirty, prov)

    sets = [f"runner={runner}", f"suites={suites}"]
    if limit:
        sets.append(f"limit={limit}")
    if extra_args:
        sets.append(f"extra_args={extra_args}")
    # evalkit 实验没有 config 树，跳过 config 校验；超参仍由 recipe schema 严格校验。
    spec = _build_spec_or_exit(
        exp_path, method="evalkit/benchmark", project=project, sets=sets,
        pools=pools, roles=roles, init_from=None, then=[],
        model=resolved_model, provenance=prov, validate=False,
    )
    with cli_ui.submit_progress() as reporter:
        res = api_client.submit_via_server(
            exp_path, resolved_profile, common.ROOT,
            project=project, reporter=reporter, spec=spec,
        )
    _echo_submit_result(res, label="（基准评测）")
    typer.echo("  评完看分数：web「Benchmarks」页，或 GET /api/benchmarks")
