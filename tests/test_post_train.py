"""训练后闭环测试：集群侧 post_train.sh 的 step 发现 / 后端检测逻辑（干跑）。

runtime_env 的组装与密钥分流已上移到中心化服务端，不再由本机 CLI 处理，故此处只覆盖
集群侧脚本本身的行为。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POST_TRAIN = REPO_ROOT / "scripts" / "post_train.sh"


# --------------------------- post_train.sh 干跑（step 发现 + 后端检测）---------------------------
def _run_post(tmp_ckpt: Path, *args: str) -> str:
    env = {"LAB_DRY_RUN": "1", "NEMO_RL_DIR": "/opt/nemo-rl", "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        ["bash", str(POST_TRAIN), *args, "--ckpt-dir", str(tmp_ckpt)],
        capture_output=True, text=True, env=env,
    )
    return proc.stdout


def _mk_step(root: Path, n: int, megatron: bool):
    step = root / f"step_{n}"
    (step / "policy" / "tokenizer").mkdir(parents=True)
    (step / "config.yaml").write_text("policy: {}\n")
    if megatron:
        (step / "policy" / "weights" / f"iter_{n:07d}").mkdir(parents=True)
    else:
        w = step / "policy" / "weights"
        w.mkdir(parents=True)
        (w / ".metadata").write_text("")


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_export_with_run_id_and_output_root(tmp_path: Path):
    run_id = "grpo_demo_v1-alice-20250630-120000"
    ckpt = tmp_path / "shared" / "alice" / "foo" / run_id
    _mk_step(ckpt, 10, megatron=False)
    env = {
        "LAB_DRY_RUN": "1",
        "NEMO_RL_DIR": "/opt/nemo-rl",
        "OUTPUT_ROOT": str(tmp_path / "shared"),
        "RUN_USER": "alice",
        "PATH": "/usr/bin:/bin",
    }
    proc = subprocess.run(
        ["bash", str(POST_TRAIN), "export", "experiments/foo", "--run-id", run_id],
        capture_output=True, text=True, env=env,
    )
    out = proc.stdout
    assert "ckpt_root:" in out and run_id in out
    assert "step=10" in out and "backend=dcp" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_export_picks_latest_step_and_megatron(tmp_path: Path):
    _mk_step(tmp_path, 5, megatron=True)
    _mk_step(tmp_path, 12, megatron=True)
    out = _run_post(tmp_path, "export", "experiments/foo")
    assert "step=12" in out and "backend=megatron" in out
    assert "convert_megatron_to_hf.py" in out
    assert "iter_0000012" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_export_explicit_step(tmp_path: Path):
    _mk_step(tmp_path, 5, megatron=True)
    _mk_step(tmp_path, 12, megatron=True)
    out = _run_post(tmp_path, "export", "experiments/foo", "--step", "5")
    assert "step=5" in out and "iter_0000005" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_export_dcp_backend(tmp_path: Path):
    _mk_step(tmp_path, 10, megatron=False)
    out = _run_post(tmp_path, "export", "experiments/foo")
    assert "backend=dcp" in out
    assert "convert_dcp_to_hf.py" in out
    assert "policy/weights --hf-ckpt-path" in out  # dcp 用 weights 目录本身


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_eval_without_model_exports_first(tmp_path: Path):
    _mk_step(tmp_path, 8, megatron=False)
    out = _run_post(tmp_path, "eval", "experiments/foo")
    assert "convert_dcp_to_hf.py" in out  # 先导出
    assert "run_eval.py" in out  # 再评测


# --------------------------- 实验自带评测脚本（eval.py）---------------------------
# 约定与训练入口 run.py 一致：实验目录里有 eval.py 就用之，否则回落官方 run_eval.py。
# 这条分支的价值是「官方协议对不上时能换掉」，所以要锁住：选对了脚本、给对了模型、
# 且透传了覆盖项——选错了的表现是评测跑完但口径不对，数字看着正常却没法和论文比。
def _mk_exp_with_eval(tmp_path: Path, *, with_eval_py: bool) -> tuple[Path, Path]:
    """造一个仿真仓库，返回 (repo_root, ckpt_root)。"""
    root = tmp_path / "repo"
    exp = root / "experiments" / "demo_exp"
    exp.mkdir(parents=True)
    if with_eval_py:
        (exp / "eval.py").write_text("# custom evaluator\n")
    scripts = root / "scripts"
    scripts.mkdir()
    for name in ("post_train.sh", "_output_paths.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / name, scripts / name)

    ckpt = tmp_path / "ckpt"
    _mk_step(ckpt, 10, megatron=False)
    return root, ckpt


def _run_eval(root: Path, ckpt: Path, *extra: str) -> str:
    proc = subprocess.run(
        ["bash", str(root / "scripts" / "post_train.sh"), "eval", "experiments/demo_exp",
         "--ckpt-dir", str(ckpt), "--model", "/models/hf-export", *extra],
        capture_output=True, text=True,
        env={"LAB_DRY_RUN": "1", "NEMO_RL_DIR": "/opt/nemo-rl", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_eval_prefers_experiment_eval_py(tmp_path: Path):
    root, ckpt = _mk_exp_with_eval(tmp_path, with_eval_py=True)
    out = _run_eval(root, ckpt)
    assert "实验自带" in out
    assert "experiments/demo_exp/eval.py" in out
    assert "--model /models/hf-export" in out
    # 不该再走官方入口
    assert "examples/run_eval.py" not in out
    # 必须带 --no-sync（用容器预建 venv，作业里不解析依赖）与 PYTHONPATH（能 import common.*）
    assert "--no-sync" in out
    assert "PYTHONPATH=" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_eval_falls_back_to_official_runner(tmp_path: Path):
    root, ckpt = _mk_exp_with_eval(tmp_path, with_eval_py=False)
    out = _run_eval(root, ckpt)
    assert "examples/run_eval.py" in out
    assert "实验自带" not in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_eval_forwards_overrides_to_custom_script(tmp_path: Path):
    """`--` 之后的参数要原样透传，否则改不了每题采样数这类协议参数。"""
    root, ckpt = _mk_exp_with_eval(tmp_path, with_eval_py=True)
    out = _run_eval(root, ckpt, "--", "--n", "4", "--datasets", "aime24")
    assert "--n 4" in out
    assert "--datasets aime24" in out
