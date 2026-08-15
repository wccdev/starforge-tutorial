"""跨平台新建 / fork 实验（唯一入口：lab new，macOS / Linux / Windows 共用）。"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path


class NewExperimentError(Exception):
    pass


def _resolve_src_dir(repo_root: Path, src: str) -> Path:
    for c in (src, f"experiments/{src}", f"projects/{src}"):
        p = repo_root / c
        if p.is_dir():
            return p
    raise NewExperimentError(
        f"找不到来源实验: {src}（试过 {src} / experiments/{src} / projects/{src}）"
    )


def _patch_fork_metadata(dest: Path, name: str) -> None:
    """fork 后改 swanlab project/name 与 README 标题（保留注释）。"""
    cfg = dest / "config.yaml"
    if cfg.is_file():
        lines = cfg.read_text(encoding="utf-8").splitlines()
        in_sw, sw_indent = False, 0
        for i, ln in enumerate(lines):
            s, indent = ln.strip(), len(ln) - len(ln.lstrip())
            if s == "swanlab:":
                in_sw, sw_indent = True, indent
                continue
            if in_sw:
                if s and indent <= sw_indent:
                    in_sw = False
                else:
                    m = re.match(r"^(\s*)(project|name):\s*.*$", ln)
                    if m:
                        lines[i] = f'{m.group(1)}{m.group(2)}: "{name}"'
        cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = dest / "README.md"
    if readme.is_file():
        rl = readme.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(rl):
            if ln.startswith("# "):
                rl[i] = f"# {name}"
                break
        readme.write_text("\n".join(rl) + "\n", encoding="utf-8")


def _validate_recipe_template(dest: Path, recipe) -> None:
    """只校验 recipe 声明的结构，不跨框架执行别的 validator。"""
    if recipe.framework in {"nemo-rl", "verl"} and not (dest / "config.yaml").is_file():
        raise NewExperimentError(f"recipe {recipe.id} 模板缺少 config.yaml")
    if recipe.entrypoint.kind == "experiment":
        entry = (dest / recipe.entrypoint.value).resolve()
        root = dest.resolve()
        if not entry.is_relative_to(root) or not entry.is_file():
            raise NewExperimentError(
                f"recipe {recipe.id} 模板缺少实验入口 {recipe.entrypoint.value}"
            )


def _copy_recipe_template(dest: Path, recipe) -> None:
    from nemo_lab_sdk.recipes import recipe_directory

    template = recipe_directory(recipe.id) / recipe.template
    if not template.is_dir():
        raise NewExperimentError(f"recipe {recipe.id} 缺少模板目录: {template}")
    shutil.copytree(template, dest, dirs_exist_ok=True)


def _fork_experiment(repo_root: Path, kind: str, name: str, src: str) -> None:
    dest = repo_root / kind / name
    if dest.exists():
        raise NewExperimentError(f"已存在: {dest}")

    src_dir = _resolve_src_dir(repo_root, src)
    from nemo_rl_lab.commands.exp import validate_exp_config
    from nemo_rl_lab.recipe_lock import RecipeLockError, RecipeLockManager
    from nemo_rl_lab.spec_builder import infer_recipe

    recipe_name = infer_recipe(src_dir)
    if not recipe_name:
        raise NewExperimentError(f"来源实验缺少 recipe.lock.json（recipe 声明）: {src_dir}")
    shutil.copytree(src_dir, dest)
    outputs = dest / "outputs"
    if outputs.exists():
        shutil.rmtree(outputs)

    _patch_fork_metadata(dest, name)
    try:
        RecipeLockManager().upgrade(
            dest,
            recipe_name=recipe_name,
            validate_config=lambda path, recipe: validate_exp_config(
                path, recipe, repo_root=repo_root
            ),
        )
    except RecipeLockError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise NewExperimentError(
            f"fork 后无法升级到当前 recipe：{exc}。"
            f"先对来源执行 `lab recipe upgrade {src_dir.name}`"
        ) from exc

    print(f"已 fork 实验: {dest}（来源: {src}）")
    print(f"  · config.yaml 的 swanlab project/name 与 README 标题已改为: {name}")
    print(f"下一步: 改 {dest}/config.yaml 顶部【① 调参区】试你的超参，"
          f"然后 lab submit {name} --profile <目标集群>")


def _create_from_template(
    repo_root: Path,
    kind: str,
    name: str,
    method: str,
    framework_version: str = "",
) -> None:
    dest = repo_root / kind / name
    if dest.exists():
        raise NewExperimentError(f"已存在: {dest}")

    from nemo_lab_sdk.contract import SpecError
    from nemo_lab_sdk.recipes import get_recipe

    try:
        recipe = get_recipe(method)
        selected_framework_version = framework_version.strip() or recipe.runtime.default_version
        recipe.runtime.resolve(selected_framework_version)
    except SpecError as exc:
        # get_recipe 的报错已列出可用值，并对无前缀裸名给出两段式提示。
        raise NewExperimentError(f"--method 非法: {exc}") from exc
    template = repo_root / "templates" / "experiment-template"
    if not template.is_dir():
        raise NewExperimentError(f"缺少模板目录: {template}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.staging-", dir=dest.parent))
    try:
        shutil.copytree(template, staging, dirs_exist_ok=True)
        _copy_recipe_template(staging, recipe)
        gitkeep = staging / ".gitkeep"
        if gitkeep.is_file():
            gitkeep.unlink()
        # recipe 声明只存在于 recipe.lock.json（infer_recipe 从锁读），不再写 method 标注文件；
        # 硬件 profile 在提交时用 --profile 指定（env/overrides/拓扑均由服务端注册表下发）。
        from nemo_rl_lab.recipe_lock import write_recipe_lock

        write_recipe_lock(staging, recipe.id, selected_framework_version)
        _validate_recipe_template(staging, recipe)
        staging.replace(dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"已创建实验: {dest}（method={recipe.id}, "
        f"framework={recipe.framework}@{selected_framework_version}）"
    )
    print("下一步:")
    print(f"  1. 编辑 {dest}/README.md（目标 / 模型 / 数据 / 监控）")
    print(f"  2. 按 {recipe.id} recipe 编辑模板文件")
    print(f"  3. 用 lab submit {name} --profile <目标集群>[:总卡数] 提交（如 h200 / h200:4）")


def create_experiment(
    repo_root: Path,
    kind: str,
    name: str,
    *,
    src: str = "",
    method: str = "nemo-rl/grpo",
    framework_version: str = "",
) -> None:
    """新建或 fork 实验；失败时抛 NewExperimentError。"""
    if kind not in ("experiments", "projects"):
        raise NewExperimentError("第一个参数必须是 experiments 或 projects")

    if src:
        _fork_experiment(repo_root, kind, name, src)
    else:
        _create_from_template(
            repo_root,
            kind,
            name,
            method,
            framework_version=framework_version,
        )
