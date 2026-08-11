"""路径安全的名字校验 —— 契约的一部分，两侧共用一份。

为什么放在契约包：这些名字（实验名、run_id、项目名）会被拼进集群产物路径、
Ray metadata 和 shell 命令行。校验规则一旦两侧不一致，就会出现「服务端放行、
集群侧拼出越界路径」这类问题。规则必须只有一处定义。

字符集：字母数字开头，其后仅允许 字母/数字/. _ -
杜绝 `..`、`/`、空格与 shell 元字符。
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from .errors import SpecError

#: 单段名字（实验名、run_id）：首字符必须是字母数字，避免 `-foo` 被当成命令行选项。
SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: 项目名：会被编进实验分组键 proj/<user>/<project>，需能安全进 URL 与路径。
#: 比 SEGMENT_RE 宽松一位（允许首字符是 . _ -），与既有 _PROJECT_NAME_RE 保持一致。
PROJECT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def safe_segment(value: str, *, field: str = "name") -> str:
    """校验单段名字；返回原值，非法则抛 SpecError。"""
    v = (value or "").strip()
    if not v or v in (".", "..") or not SEGMENT_RE.match(v):
        raise SpecError(f"非法名字 {value!r}（需字母数字开头，仅含字母/数字/. _ -）", field=field)
    return v


def safe_run_id(run_id: str) -> str:
    """校验 run_id 可安全用作路径段。"""
    return safe_segment(run_id, field="run_id")


def safe_project(project: str) -> str:
    """校验项目名（允许空 → 返回空串，表示未指定）。"""
    v = (project or "").strip()
    if not v:
        return ""
    if v in (".", "..") or not PROJECT_RE.match(v):
        raise SpecError(f"非法项目名 {project!r}（仅允许字母数字与 . _ -，且不能是 . 或 ..）", field="metadata.project")
    return v


def safe_exp_rel(exp_rel: str) -> str:
    """校验实验相对路径：允许多段（experiments/foo），禁止 `..`/绝对路径/危险字符。

    返回规范化后的相对路径（去掉尾部斜杠）。

    注：改造前的实现是 `.strip("/")`，会把 `/abs/path` **静默改写**成 `abs/path`。
    那不构成越权（结果仍被限制在上传包目录内），但把用户给的路径悄悄换成另一个是错的——
    这里改为显式拒绝绝对路径。实践中客户端 `lab submit` 产出的恒为 `experiments/<name>`，
    不受影响。
    """
    raw = (exp_rel or "").strip()
    if raw.startswith("/"):
        raise SpecError(f"实验路径须为相对路径，收到绝对路径 {exp_rel!r}", field="spec.source.exp")
    raw = raw.rstrip("/")
    segs = raw.split("/") if raw else []
    if not segs:
        raise SpecError(f"非法实验路径 {exp_rel!r}", field="spec.source.exp")
    for seg in segs:
        safe_segment(seg, field="spec.source.exp")
    return "/".join(segs)


def exp_basename(exp_rel: str) -> str:
    """取实验路径末段并校验（与集群侧 EXP_NAME 的取法一致）。"""
    name = PurePosixPath((exp_rel or "").strip().rstrip("/")).name
    return safe_segment(name, field="spec.source.exp")
