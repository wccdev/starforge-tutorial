"""StarForge 项目脚手架：`sf init` 的实现。

CLI 经 pip 分发（starforge-cli），用户不 clone 工具仓库；`sf init` 生成
自包含的项目目录（自己的 git 仓库），布局与作业包白名单严格对应：

    my-lab/
    ├── starforge.yaml     项目标记（项目发现的唯一依据）
    ├── experiments/       实验目录（sf new 在此创建）
    ├── configs/           官方基底 + 模型片段（脚手架落地，用户可调可 pin）
    ├── common/            项目共享代码（数据脚本 / 环境 / 奖励）
    └── .gitignore

`scripts/launch.sh`（平台契约入口）不落地到项目——提交打包时由 CLI 从包资源
注入，升级 CLI 自动携带新契约（单一事实来源）。
"""
from __future__ import annotations

import shutil
import subprocess
from importlib.metadata import version as _pkg_version
from importlib.resources import files
from pathlib import Path

PROJECT_MARKER = "starforge.yaml"


def scaffold_root() -> Path:
    """包内脚手架资源根（editable 与 wheel 安装均为真实目录）。"""
    return Path(str(files("starforge_cli") / "scaffold"))


def launch_script() -> Path:
    """平台契约入口 scripts/launch.sh 的包内真身（打包时注入作业包）。"""
    return scaffold_root() / "launch.sh"


def experiment_template() -> Path:
    """实验脚手架模板（sf new 用）。"""
    return scaffold_root() / "experiment-template"


class InitError(RuntimeError):
    pass


def init_project(dest: Path, *, name: str = "", git: bool = True) -> Path:
    """在 dest 生成 StarForge 项目；目录已是项目或非空冲突时报错。"""
    dest = dest.resolve()
    if (dest / PROJECT_MARKER).is_file():
        raise InitError(f"{dest} 已经是 StarForge 项目")
    if dest.exists() and any(dest.iterdir()):
        # 允许在非空目录初始化（如已有 README 的空仓库），但拒绝覆盖关键路径
        for clash in ("experiments", "configs", "common", PROJECT_MARKER):
            if (dest / clash).exists():
                raise InitError(f"{dest} 下已存在 {clash}，拒绝覆盖；换个目录或清理后重试")
    dest.mkdir(parents=True, exist_ok=True)

    src = scaffold_root() / "project"
    shutil.copytree(src / "configs", dest / "configs")
    shutil.copytree(src / "common", dest / "common")
    if not (dest / ".gitignore").exists():
        shutil.copyfile(src / "gitignore", dest / ".gitignore")
    (dest / "experiments").mkdir(exist_ok=True)
    keep = dest / "experiments" / ".gitkeep"
    if not keep.exists():
        keep.write_text("")

    project_name = (name or dest.name).strip()
    try:
        cli_version = _pkg_version("starforge-cli")
    except Exception:  # noqa: BLE001 — 源码运行（未安装分发元数据）
        cli_version = "dev"
    (dest / PROJECT_MARKER).write_text(
        "# StarForge 项目标记：`sf` 命令以此发现项目根，勿删除。\n"
        f'apiVersion: forge/project/v1\n'
        f'name: {project_name}\n'
        f'created_by: starforge-cli {cli_version}\n',
        encoding="utf-8",
    )
    readme = dest / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {project_name}\n\n"
            "StarForge（星锻）微调项目。常用命令：\n\n"
            "```bash\n"
            "sf new my-exp --method nemo-rl/grpo   # 新建实验\n"
            "sf validate my-exp                    # 本地校验\n"
            "sf submit my-exp --profile h200:8     # 提交训练\n"
            "sf job logs                           # 跟随日志\n"
            "```\n\n"
            "目录说明：`experiments/` 实验本体；`configs/` 官方基底与模型片段；\n"
            "`common/` 项目共享代码（数据脚本 / Agent 环境 / 奖励函数）。\n",
            encoding="utf-8",
        )

    if git and shutil.which("git") and not (dest / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=dest, check=False)
        # 本地身份：CI / 干净机器没有 user.name 时也能完成首提交（submit 需要 git 溯源）
        subprocess.run(["git", "config", "user.email", "starforge@localhost"], cwd=dest, check=False)
        subprocess.run(["git", "config", "user.name", "sf init"], cwd=dest, check=False)
        subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=dest, check=False)
        subprocess.run(["git", "add", "-A"], cwd=dest, check=False)
        subprocess.run(
            ["git", "commit", "-q", "-m", "sf init: StarForge project scaffold"],
            cwd=dest, check=False,
        )
    return dest
