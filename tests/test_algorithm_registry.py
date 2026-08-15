"""算法插件注册表。

补丁通过注册表声明与装载，平台可发现、可记账哪个作业用了哪个补丁；
monkey-patch 这个手段本身保留（对装不了新依赖的内网集群是决定性优势）。
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


def test_install_deferred_prefers_packaged_plugin():
    """launcher 若已登记同名插件包（deferred），训练入口的调用透明切换过去。

    这是 opsd 等官方插件迁移到插件包链路的兼容桥：实验入口不改一行。
    """
    from nemo_lab_sdk import plugins as sdk_plugins

    calls = []
    sdk_plugins.register_deferred(
        "opsd", lambda params, **ctx: calls.append((params, ctx)), {"teacher_mode": "ref"}
    )
    try:
        # 本地表里的 opsd 缺 pad_token_id 会炸；走到插件包版本则不会。
        registry.install_deferred("opsd", {"teacher_mode": "self"}, pad_token_id=0)
        assert calls == [({"teacher_mode": "ref"}, {"pad_token_id": 0})]
    finally:
        sdk_plugins.reset_deferred()


def test_official_plugin_packages_are_valid():
    """plugins/ 下的官方插件包必须过 manifest 校验，且与本地注册表同名同 kind 语义。"""
    from pathlib import Path

    from nemo_lab_sdk.plugins import LOAD_DEFERRED, LOAD_EAGER, load_manifest

    root = Path(__file__).resolve().parent.parent / "plugins"
    expected_load = {"maxrl": LOAD_EAGER, "opsd": LOAD_DEFERRED}
    for name, load in expected_load.items():
        m = load_manifest(root / name)
        assert m.name == name
        assert m.kind == "algorithm"
        assert m.load == load
        assert m.requires_sdk


def test_all_plugin_packages_in_repo_pass_validation():
    """plugins/ 里的每个包（含 examples/ 教程样板）都必须能直接发布。

    教程样板是用户照抄的起点，样板本身不合法等于教程在撒谎。
    """
    from pathlib import Path

    from nemo_lab_sdk.plugins import directory_digest, load_manifest

    root = Path(__file__).resolve().parent.parent / "plugins"
    manifests = sorted(root.rglob("plugin.yaml"))
    assert len(manifests) >= 4, f"应至少有官方 2 个 + 示例 2 个插件包，实际 {len(manifests)}"
    for path in manifests:
        m = load_manifest(path.parent)
        assert m.name == path.parent.name, f"{path.parent} 目录名应与 manifest name 一致"
        assert m.summary, f"{m.name} 缺 summary"
        assert directory_digest(path.parent).startswith("sha256:")


def test_recipe_declared_plugins_all_exist():
    """recipe 声明的 plugin 必须在注册表里，否则作业到集群才炸。"""
    from nemo_lab_sdk.recipes import all_recipes

    known = set(registry.all_plugins())
    for name, r in all_recipes().items():
        missing = set(r.plugins) - known
        assert not missing, f"recipe {name} 声明了未注册的插件: {missing}"
