"""镜像指纹：作业跑在哪个环境里。

代码版本(git)和配置版本(config_sha)本来就有记录，唯独环境没有。补上之后，
「为什么同样的代码这次结果不一样」才有可能查下去——所以这条链路的两端都要锁：
集群侧 shell 取到的值，和上报给 console 的值，必须是同一个。
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from common.observability import env_probe

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- Python 侧
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LAB_IMAGE", "NVIDIA_BUILD_ID", "NEMO_RL_COMMIT"):
        monkeypatch.delenv(var, raising=False)
    # 默认指向一个不存在的路径，避免读到开发机上真实的 /etc/nemo-lab-image
    monkeypatch.setattr(env_probe, "IMAGE_FINGERPRINT_FILE", "/nonexistent/nemo-lab-image")


def test_prefers_lab_image_env(monkeypatch):
    """集群侧 _run_experiment.sh 已经算好并导出了，Python 侧不该再自己算一遍。"""
    monkeypatch.setenv("LAB_IMAGE", "registry/nemo-rl-lab:0.7.0-20260805@abc123")
    assert env_probe.collect_image_id() == "registry/nemo-rl-lab:0.7.0-20260805@abc123"


def test_reads_fingerprint_file(monkeypatch, tmp_path):
    fp = tmp_path / "nemo-lab-image"
    fp.write_text(
        "LAB_IMAGE_TAG=nexus/nemo-rl-lab:0.7.0-20260805\n"
        "LAB_IMAGE_BUILD_ID=20260805-a1b2c3d4\n"
        "LAB_IMAGE_BUILD_DATE=2026-08-05T10:00:00Z\n"
    )
    monkeypatch.setattr(env_probe, "IMAGE_FINGERPRINT_FILE", str(fp))
    assert env_probe.collect_image_id() == "nexus/nemo-rl-lab:0.7.0-20260805@20260805-a1b2c3d4"


def test_falls_back_to_official_build_id(monkeypatch):
    """还没换成平台镜像时，官方镜像自带 NVIDIA_BUILD_ID，聊胜于无。"""
    monkeypatch.setenv("NVIDIA_BUILD_ID", "12345678")
    assert env_probe.collect_image_id() == "nemo-rl-official@12345678"


def test_falls_back_to_nemo_rl_commit(monkeypatch):
    monkeypatch.setenv("NEMO_RL_COMMIT", "deadbeef")
    assert env_probe.collect_image_id() == "nemo-rl@deadbeef"


def test_unknown_when_nothing_available():
    assert env_probe.collect_image_id() == "unknown"


def test_never_raises_on_unreadable_file(monkeypatch, tmp_path):
    """采集是旁路：指纹文件是个目录 / 没权限，都不能让训练挂掉。"""
    monkeypatch.setattr(env_probe, "IMAGE_FINGERPRINT_FILE", str(tmp_path))  # 目录不是文件
    assert env_probe.collect_image_id() == "unknown"


def test_image_lands_in_environment_payload(monkeypatch):
    """指纹要真的出现在上报 payload 里，否则 console 面板上还是看不到。"""
    monkeypatch.setenv("LAB_IMAGE", "img@build1")
    assert env_probe.collect_environment()["overview"]["image"] == "img@build1"


# ---------------------------------------------------------------- 集群侧 shell
@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_run_experiment_exports_and_prints_image(tmp_path):
    """_run_experiment.sh 必须把 LAB_IMAGE 打进作业日志——那是事后追查的第一现场。"""
    root = tmp_path / "repo"
    exp = root / "experiments" / "demo"
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text("policy: {}\n")
    (exp / "cluster").write_text("h100\n")
    (exp / "framework").write_text("custom\n")
    (exp / "train.sh").write_text('#!/usr/bin/env bash\necho "SEEN=${LAB_IMAGE}"\n')

    prof = root / "cluster" / "h100"
    prof.mkdir(parents=True)
    (prof / "overrides.conf").write_text("cluster.num_nodes=1\ncluster.gpus_per_node=1\n")

    scripts = root / "scripts"
    scripts.mkdir()
    for name in ("_run_experiment.sh", "_output_paths.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / name, scripts / name)

    out = subprocess.run(
        ["bash", str(scripts / "_run_experiment.sh"), str(exp)],
        capture_output=True, text=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(root),
            # 模拟官方镜像的回退路径（开发机上没有 /etc/nemo-lab-image）
            "NVIDIA_BUILD_ID": "99887766",
        },
    )
    assert out.returncode == 0, out.stderr
    assert "[run] image   : nemo-rl-official@99887766" in out.stdout
    # 且必须传给 train.sh —— 自定义框架同样要能上报自己跑在哪个镜像里
    assert "SEEN=nemo-rl-official@99887766" in out.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_run_experiment_image_unknown_without_any_marker(tmp_path):
    root = tmp_path / "repo"
    exp = root / "experiments" / "demo"
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text("policy: {}\n")
    (exp / "cluster").write_text("h100\n")
    prof = root / "cluster" / "h100"
    prof.mkdir(parents=True)
    (prof / "overrides.conf").write_text("cluster.num_nodes=1\n")
    scripts = root / "scripts"
    scripts.mkdir()
    for name in ("_run_experiment.sh", "_output_paths.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / name, scripts / name)

    out = subprocess.run(
        ["bash", str(scripts / "_run_experiment.sh"), str(exp)],
        capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root),
             "NEMO_RL_DIR": "/opt/nemo-rl", "LAB_DRY_RUN": "1"},
    )
    assert out.returncode == 0, out.stderr
    assert "[run] image   : unknown" in out.stdout
