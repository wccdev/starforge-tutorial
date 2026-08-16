"""实验 recipe 锁文件（recipe.lock.json）的生成、检查与显式升级。

对外只暴露三个稳定动作：inspect / upgrade / require_current。
锁住的是当前 catalog 的 recipe bundle 与精确 runtime，不是创建它的 SDK 版本。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from starforge_sdk import __version__ as SDK_VERSION
from starforge_sdk.contract import SpecError
from starforge_sdk.recipes import Recipe, get_recipe

LOCK_FILE = "recipe.lock.json"
LOCK_VERSION = "lab/recipe-lock/v3"
_LEGACY_LOCK_VERSION = "lab/recipe-lock/v2"
_SUPPORTED_LOCK_VERSIONS = (LOCK_VERSION, _LEGACY_LOCK_VERSION)
_EXPERIMENT_KINDS = ("experiments", "projects", "smoke")


class LockState(str, Enum):
    CURRENT = "current"
    SDK_INCOMPATIBLE = "sdk_incompatible"
    RECIPE_STALE = "recipe_stale"
    RUNTIME_REMOVED = "runtime_removed"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class LockDiff:
    field: str
    locked: Any
    current: Any


@dataclass(frozen=True)
class LockInspection:
    state: LockState
    path: Path
    recipe_name: str
    framework_version: str
    payload: dict[str, Any] | None
    expected: dict[str, Any] | None
    diffs: tuple[LockDiff, ...]
    message: str

    @property
    def is_current(self) -> bool:
        return self.state is LockState.CURRENT


class RecipeLockError(ValueError):
    def __init__(self, inspection: LockInspection):
        super().__init__(inspection.message)
        self.inspection = inspection


class RecipeLockManager:
    """锁文件的唯一入口：解析、分类、升级、提交前校验。"""

    def inspect(self, exp_dir: Path, recipe_name: str = "") -> LockInspection:
        path = Path(exp_dir) / LOCK_FILE
        if not path.is_file():
            return _malformed(
                path,
                recipe_name,
                f"实验缺少 {LOCK_FILE}；用 `forge new` 重建，或 `forge recipe upgrade` 生成",
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _malformed(path, recipe_name, f"{path} 非法: {exc}")
        if not isinstance(payload, dict):
            return _malformed(path, recipe_name, f"{path} 根节点必须是对象")

        normalized = _normalize_lock(payload)
        if normalized is None:
            return _malformed(
                path,
                recipe_name,
                f"{path} 不是支持的 recipe lock（{_LEGACY_LOCK_VERSION} / {LOCK_VERSION}）",
            )
        locked_name = recipe_name.strip() or normalized["recipe_name"]
        if not locked_name:
            return _malformed(path, "", f"{path} 缺少 recipe.name")
        if recipe_name.strip() and normalized["recipe_name"] not in {"", locked_name}:
            return _malformed(
                path,
                locked_name,
                f"{path} 锁定的是 {normalized['recipe_name']!r}，与请求的 {locked_name!r} 不一致",
            )
        try:
            recipe = get_recipe(locked_name)
        except SpecError as exc:
            return _malformed(path, locked_name, str(exc))

        sdk_reason = recipe.sdk_incompatibility()
        if sdk_reason:
            return LockInspection(
                state=LockState.SDK_INCOMPATIBLE,
                path=path,
                recipe_name=locked_name,
                framework_version=normalized["framework_version"],
                payload=payload,
                expected=None,
                diffs=(),
                message=f"{path}: {sdk_reason}",
            )

        locked_runtime = normalized["framework_version"]
        if locked_runtime and locked_runtime not in recipe.runtime.supported_versions:
            expected = recipe_lock(locked_name, recipe.runtime.default_version)
            return LockInspection(
                state=LockState.RUNTIME_REMOVED,
                path=path,
                recipe_name=locked_name,
                framework_version=locked_runtime,
                payload=payload,
                expected=expected,
                diffs=_diffs(normalized, expected),
                message=(
                    f"{path} 锁定的 {recipe.framework}@{locked_runtime} 已从 catalog 移除；"
                    f"可用: {', '.join(recipe.runtime.supported_versions)}。"
                    f"执行 `forge recipe upgrade {exp_dir.name} --accept-runtime-change`"
                    " 或指定 --framework-version"
                ),
            )

        expected = recipe_lock(locked_name, locked_runtime)
        diffs = _diffs(normalized, expected)
        if diffs:
            return LockInspection(
                state=LockState.RECIPE_STALE,
                path=path,
                recipe_name=locked_name,
                framework_version=locked_runtime or expected["framework"]["version"],
                payload=payload,
                expected=expected,
                diffs=diffs,
                message=(
                    f"{path} 与当前 SDK recipe bundle 不一致。"
                    f"执行 `forge recipe upgrade {exp_dir.name}` 后重新提交"
                ),
            )
        return LockInspection(
            state=LockState.CURRENT,
            path=path,
            recipe_name=locked_name,
            framework_version=expected["framework"]["version"],
            payload=payload,
            expected=expected,
            diffs=(),
            message=f"{path} 已锁定当前 recipe bundle",
        )

    def require_current(self, exp_dir: Path, recipe_name: str = "") -> str:
        inspection = self.inspect(exp_dir, recipe_name)
        if not inspection.is_current:
            raise RecipeLockError(inspection)
        return inspection.framework_version

    def write_current(
        self,
        exp_dir: Path,
        recipe_name: str,
        framework_version: str = "",
    ) -> dict[str, Any]:
        payload = recipe_lock(recipe_name, framework_version)
        _atomic_write(Path(exp_dir) / LOCK_FILE, payload)
        return payload

    def upgrade(
        self,
        exp_dir: Path,
        *,
        recipe_name: str = "",
        framework_version: str = "",
        accept_runtime_change: bool = False,
        dry_run: bool = False,
        validate_config: Callable[[Path, Recipe], list[str]] | None = None,
    ) -> LockInspection:
        inspection = self.inspect(exp_dir, recipe_name)
        if inspection.state is LockState.MALFORMED:
            raise RecipeLockError(inspection)
        if inspection.state is LockState.SDK_INCOMPATIBLE:
            raise RecipeLockError(inspection)

        selected = framework_version.strip()
        if inspection.state is LockState.RUNTIME_REMOVED:
            if not selected and not accept_runtime_change:
                raise RecipeLockError(inspection)
            selected = selected or get_recipe(inspection.recipe_name).runtime.default_version
        elif not selected:
            selected = inspection.framework_version

        recipe = get_recipe(inspection.recipe_name)
        try:
            recipe.runtime.resolve(selected)
        except SpecError as exc:
            raise RecipeLockError(
                LockInspection(
                    state=LockState.RUNTIME_REMOVED,
                    path=inspection.path,
                    recipe_name=inspection.recipe_name,
                    framework_version=selected,
                    payload=inspection.payload,
                    expected=recipe_lock(inspection.recipe_name, recipe.runtime.default_version),
                    diffs=inspection.diffs,
                    message=str(exc),
                )
            ) from exc

        if validate_config is not None:
            errors = validate_config(Path(exp_dir), recipe)
            if errors:
                raise RecipeLockError(
                    LockInspection(
                        state=LockState.RECIPE_STALE,
                        path=inspection.path,
                        recipe_name=inspection.recipe_name,
                        framework_version=selected,
                        payload=inspection.payload,
                        expected=recipe_lock(inspection.recipe_name, selected),
                        diffs=inspection.diffs,
                        message=(
                            f"{inspection.path} 升级后 config 不兼容: "
                            + "；".join(errors)
                        ),
                    )
                )

        expected = recipe_lock(inspection.recipe_name, selected)
        if not dry_run:
            _atomic_write(inspection.path, expected)
        return LockInspection(
            state=LockState.CURRENT,
            path=inspection.path,
            recipe_name=inspection.recipe_name,
            framework_version=selected,
            payload=expected if not dry_run else inspection.payload,
            expected=expected,
            diffs=_diffs(_normalize_lock(inspection.payload or {}) or {}, expected),
            message=(
                f"{inspection.path} 将升级到当前 recipe bundle"
                if dry_run
                else f"{inspection.path} 已升级到当前 recipe bundle"
            ),
        )

    def upgrade_all(
        self,
        repo_root: Path,
        *,
        framework_version: str = "",
        accept_runtime_change: bool = False,
        dry_run: bool = False,
        validate_config: Callable[[Path, Recipe], list[str]] | None = None,
    ) -> list[LockInspection]:
        targets = list(iter_lock_dirs(repo_root))
        planned: list[tuple[Path, LockInspection]] = []
        failures: list[LockInspection] = []
        for exp_dir in targets:
            inspection = self.inspect(exp_dir)
            if inspection.is_current:
                planned.append((exp_dir, inspection))
                continue
            try:
                planned.append((
                    exp_dir,
                    self.upgrade(
                        exp_dir,
                        framework_version=framework_version,
                        accept_runtime_change=accept_runtime_change,
                        dry_run=True,
                        validate_config=validate_config,
                    ),
                ))
            except RecipeLockError as exc:
                failures.append(exc.inspection)
        if failures:
            raise RecipeLockBatchError(failures)

        if dry_run:
            return [item for _, item in planned]

        backups: list[tuple[Path, bytes | None]] = []
        written: list[LockInspection] = []
        try:
            for exp_dir, inspection in planned:
                if inspection.expected is None or inspection.is_current and inspection.payload == inspection.expected:
                    written.append(inspection)
                    continue
                path = exp_dir / LOCK_FILE
                previous = path.read_bytes() if path.is_file() else None
                _atomic_write(path, inspection.expected)
                backups.append((path, previous))
                written.append(
                    LockInspection(
                        state=LockState.CURRENT,
                        path=path,
                        recipe_name=inspection.recipe_name,
                        framework_version=inspection.expected["framework"]["version"],
                        payload=inspection.expected,
                        expected=inspection.expected,
                        diffs=inspection.diffs,
                        message=f"{path} 已升级到当前 recipe bundle",
                    )
                )
        except Exception:
            _rollback(backups)
            raise
        return written


class RecipeLockBatchError(ValueError):
    def __init__(self, failures: list[LockInspection]):
        lines = [item.message for item in failures]
        super().__init__("批量升级中止，未写入任何锁文件:\n" + "\n".join(f"- {line}" for line in lines))
        self.failures = failures


def recipe_lock(recipe_name: str, framework_version: str = "") -> dict[str, Any]:
    recipe = get_recipe(recipe_name)
    selected = framework_version.strip() or recipe.runtime.default_version
    variant = recipe.runtime.resolve(selected)
    return {
        "apiVersion": LOCK_VERSION,
        "recipe": {
            "name": recipe.id,
            "version": recipe.version,
            "digest": recipe.digest,
            "manifest_digest": recipe.manifest_digest,
            "template_digest": recipe.template_digest,
        },
        "framework": {
            "kind": recipe.framework,
            "version": selected,
            "runtime_id": variant.runtime_id,
        },
        "created_by": {"sdk_version": SDK_VERSION},
        "requires": {"sdk": recipe.sdk_requires},
    }


def write_recipe_lock(exp_dir: Path, recipe_name: str, framework_version: str = "") -> None:
    RecipeLockManager().write_current(exp_dir, recipe_name, framework_version)


def validate_recipe_lock(exp_dir: Path, recipe_name: str) -> str:
    """提交/校验入口：锁必须已经是当前 bundle，绝不静默改写。"""
    return RecipeLockManager().require_current(exp_dir, recipe_name)


def iter_lock_dirs(repo_root: Path) -> Iterable[Path]:
    root = Path(repo_root)
    for kind in _EXPERIMENT_KINDS:
        base = root / kind
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if path.is_dir() and (path / LOCK_FILE).is_file():
                yield path


def _normalize_lock(payload: dict[str, Any]) -> dict[str, Any] | None:
    api = str(payload.get("apiVersion") or "").strip()
    recipe = payload.get("recipe") if isinstance(payload.get("recipe"), dict) else {}
    if api == LOCK_VERSION:
        framework = payload.get("framework") if isinstance(payload.get("framework"), dict) else {}
        requires = payload.get("requires") if isinstance(payload.get("requires"), dict) else {}
        return {
            "apiVersion": api,
            "recipe_name": str(recipe.get("name") or "").strip(),
            "recipe_version": str(recipe.get("version") or "").strip(),
            "digest": str(recipe.get("digest") or "").strip(),
            "manifest_digest": str(recipe.get("manifest_digest") or "").strip(),
            "template_digest": str(recipe.get("template_digest") or "").strip(),
            "framework_kind": str(framework.get("kind") or "").strip(),
            "framework_version": str(framework.get("version") or "").strip(),
            "runtime_id": str(framework.get("runtime_id") or "").strip(),
            "sdk_requires": str(requires.get("sdk") or "").strip(),
        }
    if api == _LEGACY_LOCK_VERSION:
        return {
            "apiVersion": api,
            "recipe_name": str(recipe.get("name") or "").strip(),
            "recipe_version": str(recipe.get("version") or "").strip(),
            "digest": str(recipe.get("digest") or "").strip(),
            "manifest_digest": "",
            "template_digest": "",
            "framework_kind": str(recipe.get("framework") or "").strip(),
            "framework_version": str(recipe.get("framework_version") or "").strip(),
            "runtime_id": "",
            "sdk_requires": "",
        }
    return None


def _diffs(normalized: dict[str, Any], expected: dict[str, Any]) -> tuple[LockDiff, ...]:
    current = _normalize_lock(expected) or {}
    fields = (
        ("apiVersion", "apiVersion"),
        ("recipe.name", "recipe_name"),
        ("recipe.version", "recipe_version"),
        ("recipe.digest", "digest"),
        ("recipe.manifest_digest", "manifest_digest"),
        ("recipe.template_digest", "template_digest"),
        ("framework.kind", "framework_kind"),
        ("framework.version", "framework_version"),
        ("framework.runtime_id", "runtime_id"),
        ("requires.sdk", "sdk_requires"),
    )
    out: list[LockDiff] = []
    for label, key in fields:
        locked = normalized.get(key, "")
        wanted = current.get(key, "")
        if locked != wanted:
            out.append(LockDiff(field=label, locked=locked, current=wanted))
    return tuple(out)


def _malformed(path: Path, recipe_name: str, message: str) -> LockInspection:
    return LockInspection(
        state=LockState.MALFORMED,
        path=path,
        recipe_name=recipe_name,
        framework_version="",
        payload=None,
        expected=None,
        diffs=(),
        message=message,
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _rollback(backups: list[tuple[Path, bytes | None]]) -> None:
    for path, previous in reversed(backups):
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(previous)
