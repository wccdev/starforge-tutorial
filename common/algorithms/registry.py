"""算法插件注册表。

补丁由 recipe 的 `plugins:` 声明，平台记账、launcher 统一装载，而不是各实验
入口自行 import 并调用 install_*()：这样平台能知道哪个作业用了哪个补丁的哪个
版本，能复用到别的实验，也能在提交时校验「这个方法要求的补丁在不在包里」。
monkey-patch 这个手段本身保留 —— 对装不了新依赖的内网集群，零依赖补丁是
决定性优势。

两类插件（这个区分是被现实逼出来的，不是设计洁癖）
──────────────────────────────────────────────────────────────────────────────
  EAGER    不需要运行时上下文，launcher 在 exec 训练入口前就能装（如 maxrl：
           只是给优势估计器加一个分支）。
  DEFERRED 需要 tokenizer / 已解析的配置才能装（如 opsd：install_opsd 要
           pad_token_id 与 max_seq_len）。这类由训练入口在拿到上下文后调用
           `install_deferred()`，但仍走注册表 —— 入口不再硬编码 import 哪个模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

EAGER = "eager"
DEFERRED = "deferred"


@dataclass(frozen=True)
class Plugin:
    name: str
    kind: str
    version: str
    summary: str
    install: Callable[..., None]


class PluginError(RuntimeError):
    pass


# ── 各插件的装载适配器 ────────────────────────────────────────────────────────


def _install_maxrl(_params: Mapping[str, Any] | None = None, **_ctx: Any) -> None:
    from common.algorithms.maxrl import install_maxrl_estimator

    install_maxrl_estimator()


def _install_opsd(params: Mapping[str, Any] | None = None, **ctx: Any) -> None:
    """OPSD 需要 tokenizer 才有的信息，故只能在训练入口内装。

    ctx 必填：pad_token_id、max_seq_len；可选 make_divisible_by。
    teacher_mode 从 recipe 超参取（recipe 已校验过取值范围）。
    """
    # 先校验上下文再 import：opsd 会拉起 torch，而「少传了参数」这类配置错误
    # 不该要求先加载几百 MB 的深度学习栈才能报出来。
    missing = [k for k in ("pad_token_id", "max_seq_len") if ctx.get(k) is None]
    if missing:
        raise PluginError(
            f"opsd 插件需要运行时上下文 {', '.join(missing)}；"
            f"请在训练入口取到 tokenizer 后调用 registry.install_deferred('opsd', ...)"
        )

    from common.algorithms.opsd import install_opsd

    install_opsd(
        teacher_mode=str((params or {}).get("teacher_mode") or "self"),
        pad_token_id=ctx["pad_token_id"],
        max_seq_len=ctx["max_seq_len"],
        make_divisible_by=ctx.get("make_divisible_by", 1),
    )


_REGISTRY: dict[str, Plugin] = {
    "maxrl": Plugin(
        name="maxrl",
        kind=EAGER,
        version="0.1.0",
        summary="优势估计改按组内平均奖励（通过率）归一化，学习信号集中到低通过率难题",
        install=_install_maxrl,
    ),
    "opsd": Plugin(
        name="opsd",
        kind=DEFERRED,
        version="0.1.0",
        summary="同模型自蒸馏：老师=学生权重但额外看到参考解，对同一条轨迹做 teacher-forcing",
        install=_install_opsd,
    ),
}


def all_plugins() -> dict[str, Plugin]:
    return dict(_REGISTRY)


def get(name: str) -> Plugin:
    key = (name or "").strip()
    if key not in _REGISTRY:
        raise PluginError(
            f"未知算法插件 {name!r}。可用: {', '.join(sorted(_REGISTRY)) or '（无）'}"
        )
    return _REGISTRY[key]


def install(name: str, params: Mapping[str, Any] | None = None) -> bool:
    """launcher 调用：装载 EAGER 插件。

    DEFERRED 插件在此只做存在性校验（把「补丁不在包里」这类错误提前到启动期，
    而不是训练跑起来才 ImportError），实际装载留给训练入口。
    返回是否真的装载了。
    """
    plugin = get(name)
    if plugin.kind == DEFERRED:
        return False
    plugin.install(params)
    return True


def install_deferred(name: str, params: Mapping[str, Any] | None = None, **ctx: Any) -> None:
    """训练入口调用：在拿到 tokenizer / 配置后装载 DEFERRED 插件。

    插件包优先：launcher 若已把同名 deferred 插件（平台注入、digest 锁定的
    lab_plugins 包）登记到 SDK 表，这里透明切换过去 —— 训练入口零改动，
    params 采用 launcher 登记时的 spec 超参（与 eager 插件同一口径）。
    """
    try:
        from starforge_sdk import plugins as sdk_plugins
    except ImportError:
        sdk_plugins = None
    if sdk_plugins is not None and name in sdk_plugins.deferred_names():
        sdk_plugins.install_deferred(name, **ctx)
        return

    plugin = get(name)
    if plugin.kind != DEFERRED:
        raise PluginError(f"插件 {name} 是 {plugin.kind} 类，应由 launcher 装载，不是训练入口")
    plugin.install(params, **ctx)


def register(plugin: Plugin) -> None:
    """注册自定义插件（测试与第三方扩展用）。"""
    _REGISTRY[plugin.name] = plugin
