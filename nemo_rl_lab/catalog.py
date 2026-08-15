"""与 Console recipe catalog 的精确契约校验（纯函数，不联网）。"""
from __future__ import annotations

import hashlib
import json


class CatalogCompatibilityError(ValueError):
    """CLI、Console 与 recipe catalog 不是同一份精确契约。"""


def verify_catalog_compatibility(spec, payload: dict) -> None:
    """在上传前验证 Console 公布的精确 SDK/recipe/adapter 契约。"""
    from nemo_lab_sdk import __version__ as sdk_version
    from nemo_lab_sdk.contract import API_VERSION

    if payload.get("apiVersion") != "lab/recipe-catalog/v1":
        raise CatalogCompatibilityError("Console recipe catalog apiVersion 不兼容")
    versions = (payload.get("contract") or {}).get("versions")
    if versions != [API_VERSION]:
        raise CatalogCompatibilityError(
            f"Console JobSpec contract 不兼容：server={versions!r}, cli={[API_VERSION]!r}"
        )
    server_sdk = payload.get("sdk") or {}
    if server_sdk.get("version") != sdk_version or server_sdk.get("requirement") != f"=={sdk_version}":
        raise CatalogCompatibilityError(
            f"SDK 版本不兼容：server={server_sdk!r}, cli=={sdk_version}"
        )
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
    server_runtime_id = variant.get("runtime_id") if isinstance(variant, dict) else None
    if server_runtime_id != spec.spec.framework.runtime_id:
        raise CatalogCompatibilityError(
            f"Console runtime_id 不兼容：server={server_runtime_id!r}, "
            f"cli={spec.spec.framework.runtime_id!r}"
        )
