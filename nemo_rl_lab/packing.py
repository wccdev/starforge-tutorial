"""作业负载打包：清单式文件列表 + tar.gz + git 溯源。

打包是白名单（清单式）而非黑名单：一次提交只上传运行时真正需要的五类路径——
实验目录本身、共享代码 common/、配置继承根 configs/、所选硬件 profile、
集群侧入口 scripts/launch.sh。其余（他人实验、文档、工具脚本）一律不上传。
"""
from __future__ import annotations

import hashlib
import io
import subprocess
from pathlib import Path

from nemo_rl_lab import cli_ui


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


# 集群 Linux 侧会 source/读取；Windows 工作区可能是 CRLF，上传前须规范为 LF。
_UNIX_LF_SUFFIXES = (".sh", ".conf")
_UNIX_LF_BASENAMES = frozenset({"lab"})


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
    if not files:
        cli_ui.fail(
            "作业包为空：清单内没有可上传的文件。",
            hint=f"确认实验目录 {exp_rel}/ 存在且已入 git",
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
    return buf.getvalue()
