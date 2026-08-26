"""opsd 官方插件包入口（deferred：需要 tokenizer 才有的上下文）。

算法实现的单一事实来源仍是 common/algorithms/opsd.py（随作业包上传）；
本包是插件链路上的适配层。launcher 在启动期把本入口登记到
starforge.plugins 的 deferred 表，训练入口拿到 tokenizer 后统一经
common.algorithms.registry.install_deferred("opsd", ...) 装载（registry
会优先取插件包版本）。发布：

    sf plugin publish plugins/opsd --owner <你的账号>
"""
from __future__ import annotations

from typing import Any, Mapping


def install(params: Mapping[str, Any] | None = None, **ctx: Any) -> None:
    # 先校验上下文再 import：opsd 会拉起 torch，「少传了参数」这类配置错误
    # 不该要求先加载几百 MB 的深度学习栈才能报出来。
    missing = [k for k in ("pad_token_id", "max_seq_len") if ctx.get(k) is None]
    if missing:
        raise RuntimeError(
            f"opsd 插件需要运行时上下文 {', '.join(missing)}；"
            "请在训练入口取到 tokenizer 后经 install_deferred('opsd', ...) 装载"
        )

    from common.algorithms.opsd import install_opsd

    install_opsd(
        teacher_mode=str((params or {}).get("teacher_mode") or "self"),
        pad_token_id=ctx["pad_token_id"],
        max_seq_len=ctx["max_seq_len"],
        make_divisible_by=ctx.get("make_divisible_by", 1),
    )
