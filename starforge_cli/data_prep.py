"""数据预处理脚本注册表：按约定发现，不再维护硬编码名单。

加数据集 = 往 common/data/ 放一个 prepare_<name>.py —— 与「加方法 = 往
catalog 加 recipe 目录」同一哲学。CLI（补全、错误提示、执行）都从这里取，
后续插件系统（kind=data-prep）把已安装插件的脚本目录并入扫描即可，命令层
零改动。

脚本首行 docstring 会作为该数据集的一句话说明展示在 CLI 里，用 ast 读取而
不 import —— 数据脚本普遍拉重依赖（datasets/pandas），列个清单不该付这个价。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_PREFIX = "prepare_"


@dataclass(frozen=True)
class DataPrep:
    name: str
    script: Path
    summary: str


def _summary(script: Path) -> str:
    try:
        doc = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8")))
    except (OSError, SyntaxError):
        return ""
    return doc.strip().splitlines()[0].strip() if doc else ""


def default_dirs() -> list[Path]:
    """扫描目录：仓库内置的 common/data + 已安装插件（forge_plugins/<name>/）。

    内置目录在前：同名脚本内置优先，插件不能悄悄替换内置数据集的语义。
    """
    from starforge_cli.commands.common import ROOT

    dirs = [ROOT / "common" / "data"]
    plugins_root = ROOT / "forge_plugins"
    if plugins_root.is_dir():
        dirs += sorted(p for p in plugins_root.iterdir() if p.is_dir())
    return dirs


def discover(dirs: Iterable[Path] | None = None) -> dict[str, DataPrep]:
    """扫描目录下的 prepare_<name>.py，返回 name → DataPrep（同名先到先得）。"""
    found: dict[str, DataPrep] = {}
    for d in dirs if dirs is not None else default_dirs():
        if not d.is_dir():
            continue
        for script in sorted(d.glob(f"{_PREFIX}*.py")):
            name = script.stem[len(_PREFIX):]
            if name and name not in found:
                found[name] = DataPrep(name=name, script=script, summary=_summary(script))
    return found
