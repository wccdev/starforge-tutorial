"""作业负载打包：清单式文件列表 + tar.gz + git 溯源。

打包是白名单（清单式）而非黑名单：一次提交只上传运行时真正需要的五类路径——
实验目录本身、共享代码 common/、配置继承根 configs/、所选硬件 profile、
集群侧入口 scripts/launch.sh。其余（他人实验、文档、工具脚本）一律不上传。
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import subprocess
from pathlib import Path

from starforge_cli import cli_ui


def _git_out(args: list[str], cwd: Path) -> str:
    """git 命令输出；失败直接报错——提交/打包必须在完好的 git 仓库内进行。"""
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError:
        cli_ui.fail("找不到 git 命令", hint="安装 git 后重试")
    except subprocess.CalledProcessError as e:
        cli_ui.fail(
            f"git {' '.join(args)} 失败: {e.stderr.strip() or e}",
            hint="请在 git 仓库目录内运行",
        )


def git_provenance(repo_root: Path, exp_rel: str) -> dict:
    """提交可追溯：git commit / dirty / config 指纹。git 不可用时直接报错。"""
    commit = _git_out(["rev-parse", "--short", "HEAD"], repo_root).strip()
    if not commit:
        cli_ui.fail("无法确定 git commit", hint="请在有提交历史的 git 仓库内运行")
    dirty = bool(_git_out(["status", "--porcelain"], repo_root).strip())
    cfg = repo_root / exp_rel / "config.yaml"
    if cfg.is_file():
        config_sha = hashlib.sha256(cfg.read_bytes()).hexdigest()[:12]
    else:
        config_sha = "none"
    return {"git_commit": commit, "git_dirty": dirty, "config_sha": config_sha}


# 清单内仍要剔除的非运行时产物（实验目录里偶尔混入的文档/报告）。
_UPLOAD_EXCLUDE_SUFFIXES = (".pdf",)


def _is_upload_excluded(rel: str) -> bool:
    """该相对路径是否属于「不随作业上传」的非运行时产物。"""
    return rel.replace("\\", "/").lower().endswith(_UPLOAD_EXCLUDE_SUFFIXES)


# 疑似密钥/凭据文件：命中即拒绝打包（fail-closed）。
# 清单用 `git ls-files --others` 收集未跟踪文件，.gitignore 覆盖不到的命名
# （hf_token.txt / credentials.json / *.pem …）会随 --allow-dirty 静默出网。
# 注意别用过宽的 "token*"：会误伤 tokenizer_config.json 这类训练必需文件。
_SENSITIVE_EXACT = frozenset({
    ".env", ".envrc", ".netrc",
    "credentials.json", "token.txt", "api_key.txt",
})
_SENSITIVE_GLOBS = (
    ".env.*", "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*",
    "secrets.*", "*.secret", "hf_token*", "*_credentials.json",
)


def _is_sensitive(rel: str) -> bool:
    base = rel.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if base in _SENSITIVE_EXACT:
        return True
    return any(fnmatch.fnmatch(base, pat) for pat in _SENSITIVE_GLOBS)


# 集群 Linux 侧会 source/读取；Windows 工作区可能是 CRLF，上传前须规范为 LF。
_UNIX_LF_SUFFIXES = (".sh", ".conf")
_UNIX_LF_BASENAMES = frozenset({"sf", "lab"})


def _needs_unix_lf(rel: str) -> bool:
    r = rel.replace("\\", "/")
    base = r.rsplit("/", 1)[-1]
    return base in _UNIX_LF_BASENAMES or r.endswith(_UNIX_LF_SUFFIXES)


def _normalize_unix_lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def upload_manifest(exp_rel: str, profile: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """作业包白名单：(目录前缀, 精确文件)。

    集群侧运行时的全部依赖：
      experiments/<exp>/   实验本体（config / recipe.lock / run.py …）
      common/              实验 run.py 通过 REPO_ROOT sys.path 引用的共享代码
      configs/             实验 config.yaml 用 ../../configs/ 相对路径继承的基底
      scripts/launch.sh    集群侧唯一入口

    profile 的 env/overrides 已服务端化（Console 硬件注册表经环境变量注入），
    不再随包上传 cluster/ 目录；profile 参数保留用于入参完整性校验。
    """
    if not exp_rel or not profile:
        raise ValueError("打包清单需要显式的实验路径与硬件 profile")
    prefixes = (
        exp_rel.rstrip("/") + "/",
        "common/",
        "configs/",
    )
    exact = ("scripts/launch.sh",)
    return prefixes, exact


def list_working_files(repo_root: Path, *, exp_rel: str, profile: str,
                       with_stats: bool = False):
    """作业负载文件清单：git 跟踪 + 未忽略（遵循 .gitignore），再按白名单收窄。

    with_stats=True 时返回 (files, skipped_count)，skipped 为 git 工作树中
    不属于本次作业负载而被略过的文件数。
    """
    prefixes, exact = upload_manifest(exp_rel, profile)
    listing = _git_out(["ls-files", "--cached", "--others", "--exclude-standard"], repo_root)
    raw = [f.strip() for f in listing.splitlines() if f.strip()]
    files = [
        f for f in raw
        if (f.startswith(prefixes) or f in exact) and not _is_upload_excluded(f)
    ]
    if sensitive := [f for f in files if _is_sensitive(f)]:
        cli_ui.fail(
            "作业包中发现疑似密钥/凭据文件，拒绝打包上传",
            title="作业包中发现疑似密钥/凭据文件，拒绝打包上传",
            items=sensitive,
            hint="移出上传目录或加入 .gitignore；确属训练必需请改名并确认其中不含密钥",
        )
    if not files:
        cli_ui.fail(
            "作业包为空：清单内没有可上传的文件。",
            hint=f"确认实验目录 {exp_rel}/ 存在且已入 git",
        )
    # 未跟踪文件只会在 --allow-dirty 时走到这里（clean-tree 检查把 dirty 拦在前面）。
    # 它们不受 commit 溯源约束，必须让用户看见都上传了什么。
    tracked = {f.strip() for f in _git_out(["ls-files"], repo_root).splitlines() if f.strip()}
    if untracked := [f for f in files if f not in tracked]:
        cli_ui.emit_warning(
            f"{len(untracked)} 个未跟踪文件将随作业上传（不受 commit 溯源约束）",
            body="\n".join(untracked[:20]) + ("\n…" if len(untracked) > 20 else ""),
        )
    return (files, len(raw) - len(files)) if with_stats else files


def pack_working_dir(repo_root: Path, files: list[str], on_add=None) -> bytes:
    """把清单文件打成 tar.gz；清单里的文件缺失直接报错，不静默跳过。

    on_add(n)：每加入一个文件回调一次，用于驱动进度条。
    """
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in files:
            p = repo_root / rel
            if not p.is_file():
                cli_ui.fail(
                    f"打包失败：清单文件不存在或不是普通文件: {rel}",
                    hint="文件可能已删除但仍在 git 索引里；git add -A 后重试",
                )
            arcname = rel.replace("\\", "/")
            if _needs_unix_lf(rel):
                data = _normalize_unix_lf(p.read_bytes())
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                info.mtime = int(p.stat().st_mtime)
                info.mode = p.stat().st_mode
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.add(p, arcname=arcname)
            if on_add:
                on_add(1)
        _inject_launch_script(tar, files)
    return buf.getvalue()


def _inject_launch_script(tar, files: list[str]) -> None:
    """把平台契约入口 scripts/launch.sh 从 CLI 包资源注入作业包。

    launch.sh 是「集群侧怎么启动」的平台契约，不属于用户项目内容——由 CLI
    打包时注入（单一事实来源），pip install -U starforge-cli 即自动携带新契约。
    项目里若显式存在同名文件（清单已含）则尊重项目版本，不重复注入。
    """
    import io as _io
    import tarfile as _tarfile

    if "scripts/launch.sh" in files:
        return
    from starforge_cli.project import launch_script

    src = launch_script()
    if not src.is_file():
        cli_ui.fail(f"CLI 包资源缺少 launch.sh：{src}（安装损坏，请重装 starforge-cli）")
    data = _normalize_unix_lf(src.read_bytes())
    info = _tarfile.TarInfo(name="scripts/launch.sh")
    info.size = len(data)
    info.mode = 0o755
    tar.addfile(info, _io.BytesIO(data))
