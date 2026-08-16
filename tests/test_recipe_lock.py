"""recipe.lock v3：范围兼容、显式升级、原子回滚。"""
from __future__ import annotations

import json

import pytest
from starforge_sdk.recipes import get_recipe

from starforge_cli.recipe_lock import (
    LOCK_FILE,
    LOCK_VERSION,
    LockState,
    RecipeLockBatchError,
    RecipeLockError,
    RecipeLockManager,
    recipe_lock,
    validate_recipe_lock,
    write_recipe_lock,
)


def _v2_lock(recipe_name: str, framework_version: str = "") -> dict:
    recipe = get_recipe(recipe_name)
    selected = framework_version or recipe.runtime.default_version
    return {
        "apiVersion": "lab/recipe-lock/v2",
        "sdk_version": "2.1.0",
        "recipe": {
            "name": recipe.id,
            "version": recipe.version,
            "digest": "sha256:" + "a" * 64,
            "framework": recipe.framework,
            "framework_version": selected,
        },
    }


def test_current_v3_lock_is_accepted(tmp_path):
    write_recipe_lock(tmp_path, "nemo-rl/grpo")
    assert validate_recipe_lock(tmp_path, "nemo-rl/grpo") == "0.7.0"
    payload = json.loads((tmp_path / LOCK_FILE).read_text(encoding="utf-8"))
    assert payload["apiVersion"] == LOCK_VERSION
    assert "sdk_version" not in payload
    assert payload["created_by"]["sdk_version"] == "2.1.0"


def test_v2_lock_is_stale_and_can_upgrade(tmp_path):
    (tmp_path / LOCK_FILE).write_text(json.dumps(_v2_lock("nemo-rl/grpo")), encoding="utf-8")
    manager = RecipeLockManager()
    inspection = manager.inspect(tmp_path, "nemo-rl/grpo")
    assert inspection.state is LockState.RECIPE_STALE
    with pytest.raises(RecipeLockError, match="forge recipe upgrade"):
        manager.require_current(tmp_path, "nemo-rl/grpo")
    upgraded = manager.upgrade(tmp_path, recipe_name="nemo-rl/grpo")
    assert upgraded.is_current
    assert validate_recipe_lock(tmp_path, "nemo-rl/grpo") == "0.7.0"


def test_created_by_sdk_version_is_audit_only(tmp_path):
    payload = recipe_lock("nemo-rl/grpo")
    payload["created_by"]["sdk_version"] = "9.9.9"
    (tmp_path / LOCK_FILE).write_text(json.dumps(payload), encoding="utf-8")
    assert RecipeLockManager().inspect(tmp_path, "nemo-rl/grpo").is_current


def test_removed_runtime_requires_explicit_accept(tmp_path):
    payload = recipe_lock("nemo-rl/grpo")
    payload["framework"]["version"] = "9.9.9"
    payload["framework"]["runtime_id"] = "gone"
    (tmp_path / LOCK_FILE).write_text(json.dumps(payload), encoding="utf-8")
    manager = RecipeLockManager()
    inspection = manager.inspect(tmp_path, "nemo-rl/grpo")
    assert inspection.state is LockState.RUNTIME_REMOVED
    with pytest.raises(RecipeLockError, match="--accept-runtime-change"):
        manager.upgrade(tmp_path, recipe_name="nemo-rl/grpo")
    upgraded = manager.upgrade(
        tmp_path, recipe_name="nemo-rl/grpo", accept_runtime_change=True
    )
    assert upgraded.framework_version == "0.7.0"


def test_upgrade_all_is_atomic_on_failure(tmp_path):
    first = tmp_path / "experiments" / "ok"
    second = tmp_path / "experiments" / "bad"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / LOCK_FILE).write_text(json.dumps(_v2_lock("nemo-rl/grpo")), encoding="utf-8")
    (second / LOCK_FILE).write_text("{", encoding="utf-8")
    before = (first / LOCK_FILE).read_text(encoding="utf-8")
    with pytest.raises(RecipeLockBatchError):
        RecipeLockManager().upgrade_all(tmp_path)
    assert (first / LOCK_FILE).read_text(encoding="utf-8") == before


def test_upgrade_all_dry_run_does_not_write(tmp_path):
    exp = tmp_path / "experiments" / "stale"
    exp.mkdir(parents=True)
    (exp / LOCK_FILE).write_text(json.dumps(_v2_lock("nemo-rl/grpo")), encoding="utf-8")
    before = (exp / LOCK_FILE).read_text(encoding="utf-8")
    items = RecipeLockManager().upgrade_all(tmp_path, dry_run=True)
    assert items[0].is_current
    assert (exp / LOCK_FILE).read_text(encoding="utf-8") == before
    RecipeLockManager().upgrade_all(tmp_path)
    assert json.loads((exp / LOCK_FILE).read_text(encoding="utf-8"))["apiVersion"] == LOCK_VERSION
