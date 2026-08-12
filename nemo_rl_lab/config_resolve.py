"""实验 config.yaml 解析与静态校验 —— 实现已迁至 nemo-lab-sdk。

为什么迁走：`lab validate` 在本地校验、console 在提交时校验、集群侧 launcher 在
启动时解析，三方用的必须是同一套 defaults 继承与 `_override_` 语义。放在客户端
包里意味着 console 要依赖客户端才能校验配置 —— 方向是反的。

此处保留同名转发，既有 `from nemo_rl_lab.config_resolve import ...` 无需改动。
新代码请直接用 `nemo_lab_sdk.config_resolve`。
"""
from __future__ import annotations

from nemo_lab_sdk.config_resolve import *  # noqa: F401,F403
from nemo_lab_sdk.config_resolve import (  # noqa: F401
    deep_merge,
    load_yaml,
    resolve,
)
