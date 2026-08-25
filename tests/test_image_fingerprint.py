"""镜像指纹：作业跑在哪个环境里。

代码版本(git)和配置版本(config_sha)本来就有记录，唯独环境没有。补上之后，
「为什么同样的代码这次结果不一样」才有可能查下去——所以这条链路的两端都要锁：
集群侧 shell 取到的值，和上报给 console 的值，必须是同一个。
"""
from __future__ import annotations

import pytest

from common.observability import env_probe


# ---------------------------------------------------------------- Python 侧
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("FORGE_IMAGE", "NVIDIA_BUILD_ID", "NEMO_RL_COMMIT"):
        monkeypatch.delenv(var, raising=False)
    # 默认指向一个不存在的路径，避免读到开发机上真实的 /etc/starforge-image
    monkeypatch.setattr(env_probe, "IMAGE_FINGERPRINT_FILE", "/nonexistent/starforge-image")


def test_prefers_forge_image_env(monkeypatch):
    """统一 launcher 已经算好并导出了，Python 侧不该再自己算一遍。"""
    monkeypatch.setenv("FORGE_IMAGE", "registry/starforge:0.7.0-20260805@abc123")
    assert env_probe.collect_image_id() == "registry/starforge:0.7.0-20260805@abc123"


def test_reads_fingerprint_file(monkeypatch, tmp_path):
    fp = tmp_path / "starforge-image"
    fp.write_text(
        "FORGE_IMAGE_TAG=nexus/starforge:0.7.0-20260805\n"
        "FORGE_IMAGE_BUILD_ID=20260805-a1b2c3d4\n"
        "FORGE_IMAGE_BUILD_DATE=2026-08-05T10:00:00Z\n"
    )
    monkeypatch.setattr(env_probe, "IMAGE_FINGERPRINT_FILE", str(fp))
    assert env_probe.collect_image_id() == "nexus/starforge:0.7.0-20260805@20260805-a1b2c3d4"


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
    monkeypatch.setenv("FORGE_IMAGE", "img@build1")
    assert env_probe.collect_environment()["overview"]["image"] == "img@build1"
