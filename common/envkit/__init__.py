"""envkit：RL Agent 环境的纯逻辑内核（与传输层无关）。

按 NeMo Gym 的架构精髓抽取（不硬包一层皮，抽内核做两条腿）：

  内核（本包）      标签解析、工具执行、判分 —— 纯函数，与 NeMo-RL 的
                    EnvironmentInterface 和 HTTP 都无关
  腿一：训练主线    common/environments/* 的 step() 是内核的薄壳
                    （批量映射 + Ray Actor 包装），训练行为不变
  腿二：Gym server  envkit.gym_server 把同一内核暴露成 HTTP 协议
                    （/seed_session + /verify + 类型化工具路由），供
                    NeMo Gym / 任意 rollout 框架接入

工具中的代码执行统一走 starforge_core.sandbox.SandboxProvider 分派
（本地子进程 / 外部 E2B 兼容沙箱，由平台注入的环境变量决定）。
"""
from __future__ import annotations

from common.envkit.tags import extract_tag
from common.envkit.tools import TOOLS, safe_eval, tool_calc, tool_python, tool_search

__all__ = [
    "TOOLS",
    "extract_tag",
    "safe_eval",
    "tool_calc",
    "tool_python",
    "tool_search",
]
