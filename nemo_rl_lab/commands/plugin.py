"""插件命令：ls / info / publish / install / remove。

插件是平台托管、digest 锁定的扩展包（algorithm 算法补丁 / data-prep 数据脚本）。
代码只在用户自己的作业容器（或本机）执行，控制平面只存储与分发。

    lab plugin publish ./my-plugin          # 发布（名字版本以 plugin.yaml 为准）
    lab plugin install alice/myalgo --exp x # 下载到本地 + 写实验锁文件
    lab submit x ...                        # 锁文件里的引用随 JobSpec 提交
"""
from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Optional

import typer

from nemo_rl_lab import api_client, cli_ui
from nemo_rl_lab.auth import gate
from nemo_rl_lab.commands import common

plugin_app = typer.Typer(
    no_args_is_help=True,
    help="插件中心：发布 / 安装 / 管理平台托管的扩展包（算法补丁、数据脚本）",
    context_settings={"help_option_names": ["-h", "--help"]},
)

#: 本地安装目录（仓库根下）。data-prep 插件的 prepare_*.py 从这里被发现。
LOCAL_PLUGINS_DIR = "lab_plugins"


def _split_ref(ref: str) -> tuple[str, str, str]:
    """解析 <owner>/<name>[@version] → (owner, name, version)。"""
    base, _, version = ref.strip().partition("@")
    owner, _, name = base.partition("/")
    if not owner or not name or "/" in name:
        cli_ui.fail(
            f"插件 ID 必须是 <owner>/<name>[@version]，收到 {ref!r}",
            hint="用 `lab plugin ls` 查看完整 ID",
        )
    return owner, name, version


@plugin_app.command("ls", help="列出平台上的插件")
def plugin_ls() -> None:
    gate()
    rows = api_client.api_get("/api/plugins")["plugins"]
    if not rows:
        typer.echo("（还没有插件；`lab plugin publish <目录>` 发布一个）")
        return
    for p in rows:
        state = "" if p["enabled"] else "  [已禁用]"
        typer.echo(
            f"{p['id']:32s} {p['kind']:10s} @{p['latest_version']:<10s}{state} {p['summary']}"
        )


@plugin_app.command("info", help="查看插件详情（manifest、版本历史、digest）")
def plugin_info(
    ref: str = typer.Argument(..., help="插件 ID：<owner>/<name>[@version]"),
) -> None:
    gate()
    owner, name, version = _split_ref(ref)
    q = f"?version={version}" if version else ""
    p = api_client.api_get(f"/api/plugins/{owner}/{name}{q}")
    typer.echo(f"{p['id']}@{p['version']}  [{p['kind']}]  {'可用' if p['enabled'] else '已禁用'}")
    if p.get("summary"):
        typer.echo(f"  {p['summary']}")
    typer.echo(f"  digest : {p['digest']}")
    typer.echo(f"  发布者 : {p['created_by']}  {p['created_at']}")
    manifest = p.get("manifest") or {}
    if manifest.get("entrypoint"):
        typer.echo(f"  入口   : {manifest['entrypoint']}  ({manifest.get('load') or 'eager'})")
    if (manifest.get("requires") or {}).get("sdk"):
        typer.echo(f"  要求   : SDK {manifest['requires']['sdk']}")
    typer.echo("  版本   :")
    for v in p.get("versions") or []:
        state = "" if v["enabled"] else "  [已禁用]"
        typer.echo(f"    {v['version']:12s} {v['digest'][:19]}…  {v['created_at']}{state}")


@plugin_app.command("publish", help="发布一个插件版本（目录内须有 plugin.yaml；版本不可变）")
def plugin_publish(
    path: str = typer.Argument(..., help="插件目录"),
    owner: Optional[str] = typer.Option(
        None, "--owner", help="目标命名空间（仅 admin 可跨；默认自己）"
    ),
) -> None:
    gate()
    from nemo_lab_sdk.plugins import PluginError, digest_files, directory_digest, load_manifest

    src = Path(path)
    if not src.is_dir():
        cli_ui.fail(f"不是目录: {path}")
    try:
        manifest = load_manifest(src)
    except PluginError as e:
        cli_ui.fail(f"插件包非法：{e}", hint="检查 plugin.yaml（schema: lab-plugin/v1）")
    digest = directory_digest(src)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in digest_files(src):
            tar.add(f, arcname=f.relative_to(src).as_posix())
    blob = buf.getvalue()

    q = f"?owner={owner}" if owner else ""
    row = api_client.api_post_bytes(f"/api/plugins/publish{q}", blob)
    if row["digest"] != digest:
        # 理论上不可能：两侧跑的是 SDK 同一份 directory_digest。真出现说明打包/解包链路有毛病。
        cli_ui.fail(
            f"服务端摘要 {row['digest']} 与本地 {digest} 不一致，发布结果不可信",
            hint="检查 CLI 与服务端 SDK 版本是否一致",
        )
    typer.secho(
        f"✓ 已发布 {row['id']}@{row['version']}  [{row['kind']}]  "
        f"{cli_ui.human_bytes(row['size_bytes'])}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  digest: {row['digest']}")
    typer.echo(f"  实验里使用：lab plugin install {row['id']}@{row['version']} --exp <实验>")


@plugin_app.command(
    "install",
    help="安装插件：下载到本地 lab_plugins/，并（可选）锁定到实验的 plugins.lock.json",
)
def plugin_install(
    ref: str = typer.Argument(..., help="插件 ID：<owner>/<name>[@version]，缺省最新版"),
    exp: Optional[str] = typer.Option(
        None, "--exp", "-e", autocompletion=common.complete_exp,
        help="锁定到该实验：提交时随 JobSpec 引用，由平台注入作业包",
    ),
) -> None:
    gate()
    from nemo_lab_sdk.plugins import directory_digest

    owner, name, version = _split_ref(ref)
    q = f"?version={version}" if version else ""
    meta = api_client.api_get(f"/api/plugins/{owner}/{name}{q}")
    if not meta["enabled"]:
        cli_ui.fail(f"插件 {meta['id']}@{meta['version']} 已被管理员禁用")
    blob, _headers = api_client.api_get_bytes(f"/api/plugins/{owner}/{name}/package{q}")

    dest = common.ROOT / LOCAL_PLUGINS_DIR / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                cli_ui.fail(f"插件包含非法归档成员路径: {m.name}")
        tar.extractall(dest, filter="data")
    local_digest = directory_digest(dest)
    if local_digest != meta["digest"]:
        shutil.rmtree(dest, ignore_errors=True)
        cli_ui.fail(
            f"下载内容摘要 {local_digest} 与平台记录 {meta['digest']} 不一致，已删除",
            hint="重试；若持续失败请联系管理员",
        )
    typer.secho(
        f"✓ 已安装 {meta['id']}@{meta['version']} → {LOCAL_PLUGINS_DIR}/{name}/",
        fg=typer.colors.GREEN,
    )

    if meta["kind"] == "data-prep":
        typer.echo("  数据脚本已可用：lab dataset prepare 会自动发现其中的 prepare_*.py")
    if not exp:
        if meta["kind"] == "algorithm":
            typer.echo(f"  提交训练时使用：lab plugin install {meta['id']}@{meta['version']} --exp <实验>")
        return

    from nemo_rl_lab.plugins_lock import LOCK_FILE, upsert_plugin_lock

    exp_path = common.resolve_exp(exp)
    try:
        plugins = upsert_plugin_lock(
            common.ROOT / exp_path,
            {"id": meta["id"], "version": meta["version"], "digest": meta["digest"]},
        )
    except ValueError as e:
        cli_ui.fail(str(e))
    typer.echo(f"  已锁定到 {exp_path}/{LOCK_FILE}（{len(plugins)} 个插件），提交时自动生效")


@plugin_app.command("remove", help="从实验的 plugins.lock.json 移除一个插件引用")
def plugin_remove(
    ref: str = typer.Argument(..., help="插件 ID：<owner>/<name>"),
    exp: str = typer.Option(
        ..., "--exp", "-e", autocompletion=common.complete_exp, help="实验名或路径"
    ),
) -> None:
    from nemo_rl_lab.plugins_lock import remove_plugin_lock

    owner, name, _ = _split_ref(ref)
    exp_path = common.resolve_exp(exp)
    try:
        removed = remove_plugin_lock(common.ROOT / exp_path, f"{owner}/{name}")
    except ValueError as e:
        cli_ui.fail(str(e))
    if not removed:
        cli_ui.fail(f"实验 {exp_path} 未引用插件 {owner}/{name}")
    typer.secho(f"✓ 已从 {exp_path} 移除 {owner}/{name}", fg=typer.colors.GREEN)


@plugin_app.command("enable", help="启用插件（admin）")
def plugin_enable(
    ref: str = typer.Argument(..., help="插件 ID：<owner>/<name>[@version]"),
) -> None:
    _set_enabled(ref, True)


@plugin_app.command("disable", help="禁用插件：新作业不可再引用，历史记录保留（admin）")
def plugin_disable(
    ref: str = typer.Argument(..., help="插件 ID：<owner>/<name>[@version]"),
) -> None:
    _set_enabled(ref, False)


def _set_enabled(ref: str, enabled: bool) -> None:
    gate()
    owner, name, version = _split_ref(ref)
    body: dict = {"enabled": enabled}
    if version:
        body["version"] = version
    r = api_client.api_patch(f"/api/plugins/{owner}/{name}", body)
    verb = "启用" if enabled else "禁用"
    typer.secho(f"✓ 已{verb} {r['id']}（{r['versions_affected']} 个版本）", fg=typer.colors.GREEN)
