"""跨平台新建 / fork 实验（替代 scripts/new_experiment.sh，macOS / Linux / Windows 共用）。"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path


class NewExperimentError(Exception):
    pass


def _list_profiles(repo_root: Path) -> list[str]:
    base = repo_root / "cluster"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "overrides.conf").is_file())


def _validate_cluster(repo_root: Path, cluster: str) -> None:
    if not cluster:
        return
    conf = repo_root / "cluster" / cluster / "overrides.conf"
    if not conf.is_file():
        opts = " ".join(_list_profiles(repo_root))
        raise NewExperimentError(
            f"未知集群 profile: {cluster}（cluster/{cluster}/overrides.conf 不存在）\n"
            f"可选: {opts or '(无)'}"
        )


def _resolve_src_dir(repo_root: Path, src: str) -> Path:
    for c in (src, f"experiments/{src}", f"projects/{src}"):
        p = repo_root / c
        if p.is_dir():
            return p
    raise NewExperimentError(
        f"找不到来源实验: {src}（试过 {src} / experiments/{src} / projects/{src}）"
    )


def _read_cluster_file(dest: Path) -> str:
    cluster_file = dest / "cluster"
    if not cluster_file.is_file():
        return "未设置"
    return cluster_file.read_text(encoding="utf-8").strip() or "未设置"


def _write_cluster_file(dest: Path, cluster: str) -> None:
    (dest / "cluster").write_text(f"{cluster}\n", encoding="utf-8")


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


def _drop_yaml_block(text: str, key: str) -> str:
    lines, out, i = text.splitlines(keepends=True), [], 0
    while i < len(lines):
        if re.match(rf"^{key}:\s*$", lines[i]):
            i += 1
            while i < len(lines) and not re.match(r"^[A-Za-z_]+:", lines[i]):
                i += 1
        else:
            out.append(lines[i])
            i += 1
    return "".join(out)


def _apply_sft_method(dest: Path) -> None:
    cfg = dest / "config.yaml"
    t = cfg.read_text(encoding="utf-8").replace(
        "../../configs/base/grpo_math_1B.yaml", "../../configs/base/sft.yaml"
    )
    t = _drop_yaml_block(_drop_yaml_block(t, "grpo"), "loss_fn")
    sft_block = (
        "sft:\n"
        "  max_num_epochs: 1\n"
        "  val_period: 50\n"
        "  val_batches: 8\n\n"
        "# 数据集：SFT 读指令数据（见 common/data/README.md 与官方 examples/run_sft.py）\n"
        "# data:\n"
        "#   train:\n"
        "#     data_path: /abs/path/train.jsonl\n\n"
    )
    t = t.replace("logger:", sft_block + "logger:", 1)
    cfg.write_text(t, encoding="utf-8")

    run_sh = dest / "run.sh"
    t = run_sh.read_text(encoding="utf-8")
    t = re.sub(
        r'^#\s*export ENTRY="\$\{ENTRY:-examples/run_sft\.py\}"',
        'export ENTRY="${ENTRY:-examples/run_sft.py}"',
        t,
        flags=re.M,
    )
    run_sh.write_text(t, encoding="utf-8")


def _apply_agent_method(dest: Path, repo_root: Path) -> None:
    cfg = dest / "config.yaml"
    t = cfg.read_text(encoding="utf-8").replace(
        "../../configs/base/grpo_math_1B.yaml", "../../configs/base/grpo_sliding_puzzle.yaml"
    )
    t = re.sub(
        r"^(\s*max_rollout_turns:\s*)1\b.*$",
        r"\g<1>6            # 多轮 Agent：工具调用 + 答题轮数上限",
        t,
        flags=re.M,
    )
    cfg.write_text(t, encoding="utf-8")
    shutil.copy2(repo_root / "templates" / "agent-run.py.tmpl", dest / "run.py")


def _apply_custom_method(dest: Path, repo_root: Path) -> None:
    """自定义框架骨架：实验自带 train.sh，不经 NeMo-RL 启动。

    只放 train.sh + framework 标记；config.yaml 留着但由 train.sh 自己解释（各框架配置格式不同，
    强行套 NeMo-RL 的 defaults 继承体系反而添乱）。
    """
    tmpl = repo_root / "templates" / "custom-framework" / "train.sh"
    if not tmpl.is_file():
        raise NewExperimentError(f"缺少模板: {tmpl}")
    shutil.copy2(tmpl, dest / "train.sh")
    (dest / "framework").write_text("custom\n", encoding="utf-8")
    # NeMo-RL 的 defaults 继承对自定义框架没意义，清成一个空壳让用户自己填。
    (dest / "config.yaml").write_text(
        "# 本实验用自定义框架（FRAMEWORK=custom），本文件由 train.sh 自行解释。\n"
        "# NeMo-RL 的 defaults 继承体系在这里不生效——想用什么格式都行（yaml/json/argparse）。\n",
        encoding="utf-8",
    )


def _fork_experiment(
    repo_root: Path, kind: str, name: str, src: str, cluster: str
) -> None:
    dest = repo_root / kind / name
    if dest.exists():
        raise NewExperimentError(f"已存在: {dest}")

    src_dir = _resolve_src_dir(repo_root, src)
    shutil.copytree(src_dir, dest)
    outputs = dest / "outputs"
    if outputs.exists():
        shutil.rmtree(outputs)

    _patch_fork_metadata(dest, name)
    if cluster:
        _write_cluster_file(dest, cluster)

    print(f"已 fork 实验: {dest}（来源: {src}）")
    print(f"  · config.yaml 的 swanlab project/name 与 README 标题已改为: {name}")
    print(f"  · 目标集群(cluster): {_read_cluster_file(dest)}")
    print(f"下一步: 改 {dest}/config.yaml 顶部【① 调参区】试你的超参，然后 lab submit {name}")


def _create_from_template(
    repo_root: Path, kind: str, name: str, cluster: str, method: str
) -> None:
    dest = repo_root / kind / name
    if dest.exists():
        raise NewExperimentError(f"已存在: {dest}")

    template = repo_root / "templates" / "experiment-template"
    if not template.is_dir():
        raise NewExperimentError(f"缺少模板目录: {template}")

    shutil.copytree(template, dest)
    gitkeep = dest / ".gitkeep"
    if gitkeep.is_file():
        gitkeep.unlink()

    if cluster:
        _write_cluster_file(dest, cluster)

    if method == "grpo":
        pass
    elif method == "sft":
        _apply_sft_method(dest)
    elif method == "agent":
        _apply_agent_method(dest, repo_root)
    elif method == "custom":
        _apply_custom_method(dest, repo_root)
    else:
        shutil.rmtree(dest)
        raise NewExperimentError(f"未知 --method: {method}（可选 grpo | sft | agent | custom）")

    print(f"已创建实验: {dest}（method={method}）")
    print(f"  · 目标集群(cluster): {_read_cluster_file(dest)}（按需改：echo h100 > {dest}/cluster）")
    print("下一步:")
    print(f"  1. 编辑 {dest}/README.md（目标 / 模型 / 数据 / SwanLab）")
    print(f"  2. 编辑 {dest}/config.yaml（基底已设为 {method}；写本实验差异）")
    if method == "sft":
        print(f"  3. SFT 入口已设好（run.sh 的 ENTRY=examples/run_sft.py）；填好数据后 lab submit {name}")
    elif method == "agent":
        print(
            f"  3. 编辑 {dest}/run.py（已放骨架：实现你的环境 + 数据，见文件内 TODO 与 multitool 范例）"
        )
    elif method == "custom":
        print(f"  3. 编辑 {dest}/train.sh（填你的启动命令；文件头写清了可用的 LAB_* 契约）")
        print("     ⚠️ 新依赖装不进作业，需要 overlay 镜像——先看 cluster/README.md")
    else:
        print(f"  3. 自定义多轮环境：写 {dest}/run.py（自动选用）。见 configs/README.md")


def create_experiment(
    repo_root: Path,
    kind: str,
    name: str,
    *,
    src: str = "",
    cluster: str = "",
    method: str = "grpo",
) -> None:
    """新建或 fork 实验；失败时抛 NewExperimentError。"""
    if kind not in ("experiments", "projects"):
        raise NewExperimentError("第一个参数必须是 experiments 或 projects")
    _validate_cluster(repo_root, cluster)

    if src:
        _fork_experiment(repo_root, kind, name, src, cluster)
    else:
        _create_from_template(repo_root, kind, name, cluster, method)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：new_experiment.sh <kind> <name> [src] [cluster]（LAB_METHOD 环境变量）。"""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print(
            "用法: python -m nemo_rl_lab.new_experiment "
            "<experiments|projects> <实验名> [来源实验] [集群profile]",
            file=sys.stderr,
        )
        return 1

    kind, name = args[0], args[1]
    src = args[2] if len(args) > 2 else ""
    cluster = args[3] if len(args) > 3 else ""
    method = os.environ.get("LAB_METHOD", "grpo")

    repo_root = Path(__file__).resolve().parent.parent
    try:
        create_experiment(repo_root, kind, name, src=src, cluster=cluster, method=method)
    except NewExperimentError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
