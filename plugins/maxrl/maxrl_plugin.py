"""maxrl 官方插件包入口。

算法实现的单一事实来源仍是 common/algorithms/maxrl.py（随作业包上传）；
本包是它在插件链路上的适配层 —— 让 maxrl 可以按 `<owner>/maxrl@<version>`
被发布、锁定、注入，而不是只能靠 recipe 内置名。发布：

    sf plugin publish plugins/maxrl --owner <你的账号>
"""
from __future__ import annotations

from typing import Any, Mapping


def install(_params: Mapping[str, Any] | None = None, **_ctx: Any) -> None:
    from common.algorithms.maxrl import install_maxrl_estimator

    install_maxrl_estimator()
