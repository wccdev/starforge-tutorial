"""一次性把实验目录迁移到显式 recipe + lab/v2 锁文件。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nemo_lab_sdk import __version__ as SDK_VERSION
from nemo_lab_sdk.contract import SpecError
from nemo_lab_sdk.recipes import get_recipe

LOCK_FILE = "recipe.lock.json"
LOCK_VERSION = "lab/recipe-lock/v1"


@dataclass(frozen=True)
class MigrationItem:
    path: Path
    recipe: str = ""
    needs_write: bool = False
    error: str = ""


def recipe_lock(recipe_name: str) -> dict:
    recipe = get_recipe(recipe_name)
    return {
        "apiVersion": LOCK_VERSION,
        "sdk_version": SDK_VERSION,
        "recipe": {
            "name": recipe.name,
            "version": recipe.version,
            "digest": recipe.digest,
            "framework": recipe.framework,
        },
    }


def validate_recipe_lock(exp_dir: Path, recipe_name: str) -> None:
    path = exp_dir / LOCK_FILE
    if not path.is_file():
        raise ValueError(f"实验缺少 {LOCK_FILE}；运行 `lab migrate-v2 --write`")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} 非法: {exc}") from exc
    expected = recipe_lock(recipe_name)
    if payload != expected:
        raise ValueError(
            f"{path} 与当前 SDK recipe 不一致；运行 `lab migrate-v2 --write` 显式升级"
        )


def _infer_legacy_recipe(exp_dir: Path) -> str:
    """仅供一次性迁移；正常运行路径绝不调用此推断。"""
    if (exp_dir / "train.sh").is_file():
        return "custom"
    config = exp_dir / "config.yaml"
    if not config.is_file():
        raise ValueError("无法判定 recipe：既没有 method/train.sh，也没有 config.yaml")
    text = config.read_text(encoding="utf-8").lower()
    run_text = (exp_dir / "run.py").read_text(encoding="utf-8").lower() if (exp_dir / "run.py").is_file() else ""
    candidates: list[str] = []
    if "install_opsd" in run_text or exp_dir.name.startswith("opsd_"):
        candidates.append("opsd")
    elif "install_maxrl" in run_text or exp_dir.name.startswith("maxrl_"):
        candidates.append("maxrl")
    elif "base/sft.yaml" in text or exp_dir.name.startswith("sft_"):
        candidates.append("sft")
    elif "base/distillation" in text or exp_dir.name.startswith("distillation_"):
        candidates.append("distillation")
    elif "base/grpo" in text or exp_dir.name.startswith(("grpo_", "agent-grpo_")):
        candidates.append("grpo")
    if len(candidates) != 1:
        raise ValueError("无法唯一判定 recipe；请先手工写入 method")
    return candidates[0]


def _experiment_dirs(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for kind in ("experiments", "projects"):
        base = repo_root / kind
        if base.is_dir():
            out.extend(path for path in base.iterdir() if path.is_dir())
    return sorted(out)


def check_repo(repo_root: Path) -> list[MigrationItem]:
    items: list[MigrationItem] = []
    for exp_dir in _experiment_dirs(repo_root):
        method_file = exp_dir / "method"
        recipe_name = method_file.read_text(encoding="utf-8").strip() if method_file.is_file() else ""
        try:
            if not recipe_name:
                recipe_name = _infer_legacy_recipe(exp_dir)
            get_recipe(recipe_name)
            cluster_file = exp_dir / "cluster"
            if not cluster_file.is_file() or not cluster_file.read_text(encoding="utf-8").strip():
                raise ValueError("缺少显式 cluster 资源配置")
            profile = cluster_file.read_text(encoding="utf-8").strip()
            if not (repo_root / "cluster" / profile / "overrides.conf").is_file():
                raise ValueError(f"未知 cluster profile: {profile}")
            expected = recipe_lock(recipe_name)
            lock_path = exp_dir / LOCK_FILE
            current = None
            if lock_path.is_file():
                try:
                    current = json.loads(lock_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    current = None
            items.append(MigrationItem(
                path=exp_dir,
                recipe=recipe_name,
                needs_write=(not method_file.is_file() or current != expected),
            ))
        except (ValueError, SpecError) as exc:
            items.append(MigrationItem(path=exp_dir, recipe=recipe_name, error=str(exc)))
    return items


def apply_migration(repo_root: Path, items: list[MigrationItem]) -> None:
    errors = [item for item in items if item.error]
    if errors:
        raise ValueError("迁移检查存在错误；修复后重新执行，不写入部分结果")
    for item in items:
        if not item.needs_write:
            continue
        (item.path / "method").write_text(f"{item.recipe}\n", encoding="utf-8")
        (item.path / LOCK_FILE).write_text(
            json.dumps(recipe_lock(item.recipe), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
