"""Recipe 注册表 —— 后训练方法的目录。

借鉴 verl 的 `recipe/` 布局：每种算法一个自包含的目录，共享同一套核心执行链路。
verl 的 dapo / prime / spin / sppo 各自一个包，加新算法不动核心；这里同理。

**加一种后训练方法 = 在 catalog/ 下新增一个目录，服务端零改动。**
这是阶段 1 的验收标准。

为什么 catalog 放在 lab 包里而不是 console
──────────────────────────────────────────────────────────────────────────────
一个 recipe 是不可分割的整体：元数据（服务端提交前要用来校验）与入口（集群侧要执行）
必须同版本。若把元数据放 console、入口放 lab，就又制造了一处跨仓库隐式契约 ——
正是本次改造要消灭的东西。

依赖方向 console → lab 已存在（config_resolve 复用），故 console 直接读本注册表即可，
不成环；集群侧 launcher 读的是同一份。
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

from nemo_rl_lab.contract.errors import SpecError

from .model import ENTRYPOINT_BASES, Entrypoint, ParamSpec, Recipe

#: recipe 目录根。每个子目录含一个 recipe.yaml。
CATALOG_DIR = Path(__file__).parent / "catalog"

__all__ = [
    "CATALOG_DIR",
    "ENTRYPOINT_BASES",
    "Entrypoint",
    "ParamSpec",
    "Recipe",
    "all_recipes",
    "get_recipe",
    "recipe_names",
]


@functools.lru_cache(maxsize=1)
def _load_catalog() -> dict[str, Recipe]:
    out: dict[str, Recipe] = {}
    if not CATALOG_DIR.is_dir():
        return out
    for entry in sorted(CATALOG_DIR.iterdir()):
        manifest = entry / "recipe.yaml"
        if not manifest.is_file():
            continue
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        recipe = Recipe.from_dict(data)
        if recipe.name != entry.name:
            raise SpecError(
                f"recipe 目录名 {entry.name!r} 与 recipe.yaml 里的 name {recipe.name!r} 不一致"
            )
        out[recipe.name] = recipe
    return out


def all_recipes() -> dict[str, Recipe]:
    """全部已注册 recipe（name → Recipe）。"""
    return dict(_load_catalog())


def recipe_names() -> list[str]:
    return sorted(_load_catalog())


def get_recipe(name: str) -> Recipe:
    """按名取 recipe；不存在则抛 SpecError（服务端据此回 400，并列出可用值）。"""
    catalog = _load_catalog()
    key = (name or "").strip()
    if key not in catalog:
        raise SpecError(
            f"未知的后训练方法 {name!r}。可用: {', '.join(sorted(catalog)) or '（catalog 为空）'}",
            field="spec.recipe.name",
        )
    return catalog[key]


def reset_cache() -> None:
    """测试用：清空 catalog 缓存。"""
    _load_catalog.cache_clear()
