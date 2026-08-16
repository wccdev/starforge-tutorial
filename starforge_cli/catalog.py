"""与 Console recipe catalog 的契约校验（纯函数，不联网）。

v2 握手：SDK 用兼容范围，recipe bundle / framework / runtime_id 仍精确匹配。
v1 握手保留给尚未升级的 Console，仍要求 SDK 字符串全等。
"""
from __future__ import annotations

import hashlib
import json

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


class CatalogCompatibilityError(ValueError):
    """CLI、Console 与 recipe catalog 不是同一份精确契约。"""


_CATALOG_VERSIONS = ("forge/recipe-catalog/v1", "forge/recipe-catalog/v2")


def verify_catalog_compatibility(spec, payload: dict) -> None:
    """在上传前验证 Console 公布的 SDK/recipe/adapter 契约。"""
    from starforge_sdk import __version__ as sdk_version
    from starforge_sdk.contract import API_VERSION

    api_version = payload.get("apiVersion")
    if api_version not in _CATALOG_VERSIONS:
        raise CatalogCompatibilityError("Console recipe catalog apiVersion 不兼容")
    versions = (payload.get("contract") or {}).get("versions")
    if versions != [API_VERSION]:
        raise CatalogCompatibilityError(
            f"Console JobSpec contract 不兼容：server={versions!r}, cli={[API_VERSION]!r}"
        )
    _verify_sdk_handshake(payload.get("sdk") or {}, sdk_version, api_version=api_version)
    recipes = payload.get("recipes")
    if not isinstance(recipes, list):
        raise CatalogCompatibilityError("Console recipe catalog 缺少 recipes 数组")
    canonical = []
    for item in recipes:
        if not isinstance(item, dict):
            raise CatalogCompatibilityError("Console recipe catalog 含非法 recipe 项")
        canonical.append({
            "name": item.get("name"),
            "version": item.get("version"),
            "digest": item.get("digest"),
        })
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("catalog_digest") != f"sha256:{digest}":
        raise CatalogCompatibilityError("Console recipe catalog digest 校验失败")

    selected = next((item for item in recipes if item.get("name") == spec.recipe_name), None)
    if selected is None:
        raise CatalogCompatibilityError(f"Console 未启用 recipe {spec.recipe_name!r}")
    expected = {
        "version": spec.spec.recipe.version,
        "digest": spec.spec.recipe.digest,
        "framework": spec.spec.framework.kind,
        "adapter": spec.spec.framework.kind,
    }
    drift = {
        key: {"server": selected.get(key), "cli": value}
        for key, value in expected.items()
        if selected.get(key) != value
    }
    if drift:
        raise CatalogCompatibilityError(
            f"recipe {spec.recipe_name!r} 精确契约不一致: {json.dumps(drift, ensure_ascii=False)}"
        )
    framework_version = spec.spec.framework.version
    supported = selected.get("supported_framework_versions")
    if not isinstance(supported, list) or framework_version not in supported:
        raise CatalogCompatibilityError(
            f"Console 未发布 {spec.spec.framework.kind}@{framework_version}；server={supported!r}"
        )
    runtime = selected.get("runtime") or {}
    variants = runtime.get("versions") if isinstance(runtime, dict) else None
    variant = variants.get(framework_version) if isinstance(variants, dict) else None
    if not isinstance(variant, dict):
        raise CatalogCompatibilityError(
            f"Console catalog 缺少 {spec.spec.framework.kind}@{framework_version} 的 runtime 变体"
        )
    # runtime_id 为空是合法状态（custom/user-managed 没有部署执行工件）；
    # 序列化会省略空字段，两侧都归一化成 "" 再比，避免 None != "" 假阳性。
    server_runtime_id = str(variant.get("runtime_id") or "").strip()
    if server_runtime_id != (spec.spec.framework.runtime_id or "").strip():
        raise CatalogCompatibilityError(
            f"Console runtime_id 不兼容：server={server_runtime_id!r}, "
            f"cli={spec.spec.framework.runtime_id!r}"
        )
    recipe_requires = str(selected.get("sdk_requires") or "").strip()
    if recipe_requires:
        _require_sdk_in_range(sdk_version, recipe_requires, where="recipe.sdk_requires")


def _verify_sdk_handshake(server_sdk: dict, cli_version: str, *, api_version: str) -> None:
    server_version = str(server_sdk.get("version") or "").strip()
    requirement = str(server_sdk.get("requirement") or "").strip()
    if api_version == "forge/recipe-catalog/v1":
        if server_version != cli_version or requirement != f"=={cli_version}":
            raise CatalogCompatibilityError(
                f"SDK 版本不兼容：server={server_sdk!r}, cli=={cli_version}"
            )
        return
    if not server_version or not requirement:
        raise CatalogCompatibilityError(f"SDK 握手缺少 version/requirement：{server_sdk!r}")
    _require_sdk_in_range(cli_version, requirement, where="catalog.sdk.requirement")
    _require_sdk_in_range(server_version, requirement, where="catalog.sdk.requirement")


def _require_sdk_in_range(version: str, requirement: str, *, where: str) -> None:
    try:
        parsed = Version(version)
        spec = SpecifierSet(requirement, prereleases=True)
    except (InvalidVersion, InvalidSpecifier) as exc:
        raise CatalogCompatibilityError(
            f"非法 SDK 兼容声明 {where}={requirement!r} version={version!r}"
        ) from exc
    if parsed not in spec:
        raise CatalogCompatibilityError(
            f"SDK 版本不兼容：{where}={requirement}，实际 {version}"
        )
