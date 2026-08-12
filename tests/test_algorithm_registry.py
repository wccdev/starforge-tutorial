"""算法插件注册表（阶段 1）。

改造前每个实验的 run.py 自己 import 并 install_*()，平台无从知晓哪个作业用了
哪个补丁。monkey-patch 这个手段保留（对装不了新依赖的内网集群是决定性优势），
变的是它成了平台可声明、可发现、可记账的一等概念。
"""
from __future__ import annotations

import pytest

from common.algorithms import registry
from common.algorithms.registry import DEFERRED, EAGER, Plugin, PluginError


def test_builtin_plugins_are_registered():
    assert set(registry.all_plugins()) >= {"maxrl", "opsd"}


def test_unknown_plugin_error_lists_available():
    with pytest.raises(PluginError) as e:
        registry.get("nonexistent")
    assert "maxrl" in str(e.value)


def test_opsd_is_deferred_because_it_needs_tokenizer():
    """install_opsd 需要 pad_token_id / max_seq_len，launcher 阶段拿不到。"""
    assert registry.get("opsd").kind == DEFERRED


def test_maxrl_is_eager():
    assert registry.get("maxrl").kind == EAGER


def test_launcher_install_skips_deferred_but_validates_existence():
    """launcher 装不了 DEFERRED，但「补丁不在包里」这类错误要提前到启动期暴露。"""
    assert registry.install("opsd", {}) is False


def test_deferred_plugin_rejects_missing_runtime_context():
    with pytest.raises(PluginError, match="pad_token_id"):
        registry.install_deferred("opsd", {"teacher_mode": "self"})


def test_eager_plugin_cannot_be_installed_as_deferred():
    with pytest.raises(PluginError, match="eager"):
        registry.install_deferred("maxrl", {})


def test_custom_plugin_can_be_registered():
    calls = []
    registry.register(
        Plugin(
            name="_probe",
            kind=EAGER,
            version="0",
            summary="test",
            install=lambda params=None, **_: calls.append(params),
        )
    )
    try:
        assert registry.install("_probe", {"a": 1}) is True
        assert calls == [{"a": 1}]
    finally:
        registry.all_plugins().pop("_probe", None)
        registry._REGISTRY.pop("_probe", None)


def test_every_plugin_declares_version_and_summary():
    """作业记录要落「用了哪个补丁的哪个版本」，缺一不可。"""
    for name, p in registry.all_plugins().items():
        assert p.version, f"{name} 缺 version"
        assert p.summary, f"{name} 缺 summary"
        assert p.kind in (EAGER, DEFERRED), f"{name} kind 非法"


def test_recipe_declared_plugins_all_exist():
    """recipe 声明的 plugin 必须在注册表里，否则作业到集群才炸。"""
    from nemo_lab_sdk.recipes import all_recipes

    known = set(registry.all_plugins())
    for name, r in all_recipes().items():
        missing = set(r.plugins) - known
        assert not missing, f"recipe {name} 声明了未注册的插件: {missing}"
