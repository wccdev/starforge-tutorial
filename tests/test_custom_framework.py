"""FRAMEWORK=custom 分支的集群侧行为（干跑，不真的起训练）。

这条路径是「平台能不能跑 NeMo-RL 以外的代码」的落点。它最容易出的问题不是跑不起来，
而是**跑起来了但契约没兑现**——产物写到临时目录里、拓扑变量没传下去，
到作业结束才发现 checkpoint 没了。所以这里逐条锁契约。
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_EXP = REPO_ROOT / "scripts" / "_run_experiment.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")


def _mk_exp(tmp_path: Path, *, framework: str | None, with_train_sh: bool) -> Path:
    """在临时目录里造一个仿真仓库：<root>/experiments/<name>/ + <root>/cluster/<profile>/。"""
    root = tmp_path / "repo"
    exp = root / "experiments" / "demo_exp"
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text("policy: {}\n")
    (exp / "cluster").write_text("h100\n")
    if framework:
        (exp / "framework").write_text(f"{framework}\n")
    if with_train_sh:
        # 把契约变量原样打印出来，测试据此断言
        (exp / "train.sh").write_text(
            "#!/usr/bin/env bash\n"
            'echo "OUT=${LAB_OUT_DIR}"\n'
            'echo "EXP=${LAB_EXP_DIR}"\n'
            'echo "ROOT=${LAB_REPO_ROOT}"\n'
            'echo "NAME=${LAB_EXP_NAME}"\n'
            'echo "PROFILE=${LAB_CLUSTER_PROFILE}"\n'
            'echo "NODES=${LAB_CLUSTER_NUM_NODES:-unset}"\n'
            'echo "GPUS=${LAB_CLUSTER_GPUS_PER_NODE:-unset}"\n'
            'echo "PYPATH=${PYTHONPATH}"\n'
            'echo "SECRET=${MY_SECRET:-unset}"\n'
        )

    prof = root / "cluster" / "h100"
    prof.mkdir(parents=True)
    (prof / "overrides.conf").write_text("cluster.num_nodes=1\ncluster.gpus_per_node=1\n")
    (prof / "env.sh").write_text("export PROFILE_ENV_LOADED=1\n")

    # _run_experiment.sh 会 source 同目录的 _output_paths.sh，故一并复制脚本
    scripts = root / "scripts"
    scripts.mkdir()
    for name in ("_run_experiment.sh", "_output_paths.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / name, scripts / name)
    return exp


def _run(exp: Path, **env_extra: str) -> subprocess.CompletedProcess:
    root = exp.parent.parent
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(root),
        "NEMO_RL_DIR": "/opt/nemo-rl",
        "LAB_DRY_RUN": "1",
        **env_extra,
    }
    return subprocess.run(
        ["bash", str(root / "scripts" / "_run_experiment.sh"), str(exp)],
        capture_output=True, text=True, env=env,
    )


# ---------------------------------------------------------------- 框架选择
def test_defaults_to_nemo_rl(tmp_path):
    exp = _mk_exp(tmp_path, framework=None, with_train_sh=False)
    out = _run(exp)
    assert out.returncode == 0, out.stderr
    assert "[run] frame   : nemo-rl" in out.stdout
    assert "nemolab_boot.py" in out.stdout  # 仍走原有 NeMo-RL 入口


def test_framework_file_selects_custom(tmp_path):
    """与 cluster 文件同款约定：一行文本跟着实验走，fork 时自动继承。"""
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=True)
    out = _run(exp)
    assert out.returncode == 0, out.stderr
    assert "[run] frame   : custom" in out.stdout
    assert "nemolab_boot.py" not in out.stdout


def test_env_overrides_framework_file(tmp_path):
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=True)
    out = _run(exp, FRAMEWORK="nemo-rl")
    assert out.returncode == 0, out.stderr
    assert "[run] frame   : nemo-rl" in out.stdout


def test_unknown_framework_fails_fast(tmp_path):
    exp = _mk_exp(tmp_path, framework=None, with_train_sh=False)
    out = _run(exp, FRAMEWORK="verl-but-typo")
    assert out.returncode != 0
    assert "未知 FRAMEWORK" in out.stdout + out.stderr


def test_custom_without_train_sh_fails_with_hint(tmp_path):
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=False)
    out = _run(exp)
    assert out.returncode != 0
    assert "train.sh" in out.stderr


def test_custom_does_not_require_nemo_rl_dir(tmp_path):
    """custom 不经 NeMo-RL 启动，就不该因为缺 NEMO_RL_DIR 而拒绝启动。"""
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=True)
    root = exp.parent.parent
    out = subprocess.run(
        ["bash", str(root / "scripts" / "_run_experiment.sh"), str(exp)],
        capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root),
             "LAB_DRY_RUN": "1"},
    )
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------- 契约
def _contract(exp: Path, **env_extra: str) -> dict[str, str]:
    """真跑 train.sh（不 dry-run），把它打印的契约变量解析成 dict。"""
    root = exp.parent.parent
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(root),
        **env_extra,
    }
    out = subprocess.run(
        ["bash", str(root / "scripts" / "_run_experiment.sh"), str(exp)],
        capture_output=True, text=True, env=env,
    )
    assert out.returncode == 0, out.stderr
    return dict(
        line.split("=", 1)
        for line in out.stdout.splitlines()
        if "=" in line and not line.startswith("[")
    )


def test_custom_exports_contract_vars(tmp_path):
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=True)
    got = _contract(exp)
    assert got["EXP"] == str(exp)
    assert got["ROOT"] == str(exp.parent.parent)
    assert got["NAME"] == "demo_exp"
    assert got["PROFILE"] == "h100"
    # PYTHONPATH 必须含仓库根，train.sh 里才能 import common.observability.report
    assert str(exp.parent.parent) in got["PYPATH"]


def test_custom_out_dir_is_created_and_isolated(tmp_path):
    """产物目录必须已建好且按 <用户>/<实验>/<run_id> 隔离——否则多人同跑会互相覆盖。"""
    shared = tmp_path / "shared"
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=True)
    got = _contract(
        exp,
        OUTPUT_ROOT=str(shared),
        RUN_USER="alice",
        NRL_RUN_ID="demo_exp-alice-20260805-120000",
    )
    out_dir = Path(got["OUT"])
    assert out_dir == shared / "alice" / "demo_exp" / "demo_exp-alice-20260805-120000"
    assert out_dir.is_dir(), "train.sh 拿到手时产物目录就应该已经建好"


def test_custom_receives_authoritative_topology(tmp_path):
    """服务端权威拓扑必须传到 train.sh——自定义框架靠它决定起几个进程。"""
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=True)
    got = _contract(exp, LAB_CLUSTER_NUM_NODES="2", LAB_CLUSTER_GPUS_PER_NODE="8")
    assert got["NODES"] == "2"
    assert got["GPUS"] == "8"


def test_custom_sources_cluster_secrets(tmp_path):
    """密钥仍由集群侧密钥文件注入，自定义框架不该自己去碰密钥。"""
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=True)
    secrets = tmp_path / "secrets.env"
    secrets.write_text("MY_SECRET=s3cr3t\n")
    got = _contract(exp, CLUSTER_SECRETS_FILE=str(secrets))
    assert got["SECRET"] == "s3cr3t"


# ---------------------------------------------------------------- 模板
def test_template_train_sh_fails_loudly_before_edit(tmp_path):
    """未填训练命令的模板必须报错退出，而不是「成功」跑完一个空作业。"""
    exp = _mk_exp(tmp_path, framework="custom", with_train_sh=False)
    shutil.copy2(REPO_ROOT / "templates" / "custom-framework" / "train.sh", exp / "train.sh")
    root = exp.parent.parent
    out = subprocess.run(
        ["bash", str(root / "scripts" / "_run_experiment.sh"), str(exp)],
        capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root)},
    )
    assert out.returncode != 0
    assert "还没填训练命令" in out.stderr
