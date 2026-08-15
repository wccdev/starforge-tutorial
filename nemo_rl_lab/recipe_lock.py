"""实验 recipe 锁文件（recipe.lock.json）的生成与校验。

锁文件把「这个实验用哪个 recipe 的哪个精确框架版本」固定下来，
与 SDK recipe catalog 严格比对——不一致就拒绝提交，绝不猜。
"""
from __future__ import annotations

import json
from pathlib import Path

from nemo_lab_sdk import __version__ as SDK_VERSION
from nemo_lab_sdk.recipes import get_recipe

LOCK_FILE = "recipe.lock.json"
LOCK_VERSION = "lab/recipe-lock/v2"


def recipe_lock(recipe_name: str, framework_version: str = "") -> dict:
    recipe = get_recipe(recipe_name)
    selected = framework_version.strip() or recipe.runtime.default_version
    recipe.runtime.resolve(selected)
    return {
        "apiVersion": LOCK_VERSION,
        "sdk_version": SDK_VERSION,
        "recipe": {
            "name": recipe.id,
            "version": recipe.version,
            "digest": recipe.digest,
            "framework": recipe.framework,
            "framework_version": selected,
        },
    }


def write_recipe_lock(exp_dir: Path, recipe_name: str, framework_version: str = "") -> None:
    (exp_dir / LOCK_FILE).write_text(
        json.dumps(
            recipe_lock(recipe_name, framework_version),
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def validate_recipe_lock(exp_dir: Path, recipe_name: str) -> str:
    """校验锁文件与当前 SDK catalog 完全一致，返回锁定的框架版本。"""
    path = exp_dir / LOCK_FILE
    if not path.is_file():
        raise ValueError(
            f"实验缺少 {LOCK_FILE}；用 `lab new` 重建实验，"
            "或提交时加 --framework-version 显式生成"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} 非法: {exc}") from exc
    locked_version = str((payload.get("recipe") or {}).get("framework_version") or "")
    if not locked_version:
        raise ValueError(
            f"{path} 缺少 framework_version；提交时加 --framework-version 显式重写锁文件"
        )
    expected = recipe_lock(recipe_name, locked_version)
    if payload != expected:
        raise ValueError(
            f"{path} 与当前 SDK recipe 不一致；提交时加 --framework-version 显式重写锁文件"
        )
    return locked_version
