"""verl @function_tool 硬件工具：让 agent 拿着编译器与 lint 的报错改自己的 Verilog。

官方机制（verl docs/sglang_multiturn/multiturn，v0.8 起）：
  - @function_tool 把普通 Python 函数注册为工具，schema 由
    transformers.utils.get_json_schema() 从类型注解 + Google 风格 docstring 推断，
    所以【每个参数必须有类型注解，且 docstring 必须有 Args 段】，缺了注册时直接报错。
  - config 的 rollout.multi_turn.function_tool_path 指向本文件（相对作业包根）。
  - 函数工具是无状态的（每次调用就是 fn(**parameters)）。

为什么这里**不需要** BaseTool 的有状态沙箱
──────────────────────────────────────────────────────────────────────────────
直觉上「agent 改代码」要有个持久工作区。但这两个工具每次都收整份模块源码、返回
诊断信息 —— 状态本来就在对话历史里，模型下一轮自己带着上一版代码。给一道单模块
题（VerilogEval / RTLLM 的形态）建目录、管生命周期、清理，只是把状态从一个天然
持有它的地方搬到一个需要维护的地方。真要做多文件工程题（CVDP 的 repo 级任务）时
再换 BaseTool + tool_config_path。

为什么**没有** run_testbench 工具
──────────────────────────────────────────────────────────────────────────────
两个原因，都是硬的：

  1. 判卷标准不能交到 agent 手上。奖励用的是这道题的隐藏 testbench；给它一个能跑
     testbench 的工具，等于让它一边试一边逼近判据 —— 学到的是「怎么试出答案」，
     不是「怎么设计电路」。VerilogEval 的公开/隐藏划分就是这个道理。
  2. 这两个工具都**不执行**任何东西：iverilog 只编译、verilator --lint-only 只分析。
     `vvp` 才会真的跑，而 Verilog 有 `$system` —— 跑模型写的 testbench 等价于跑模型
     写的 shell。执行只发生在奖励那一侧（common/rewards/rtl_reward.py），
     在作业容器里、临时目录里、不继承宿主环境、带超时。

lint 是给综合段留的抓手：奖励的第三段扣「推断出锁存器 / 组合环」，而那些东西在
testbench 里看不出来。不给 lint，模型就只能靠运气拿那 0.2。
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from verl.tools.function_tool import function_tool

#: 回灌进上下文的诊断长度上限。整份编译日志会挤掉模型自己的代码，
#: 而 iverilog 的第一条错误几乎总是根因，后面全是它的余波。
_MAX_CHARS = int(os.environ.get("RTL_TOOL_MAX_CHARS", "1200"))
_TIMEOUT = int(os.environ.get("RTL_TOOL_TIMEOUT", "60"))

_FENCED = re.compile(
    r"```(?:systemverilog|verilog|sv|v)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def _clean(code: str) -> str:
    """模型经常把代码包在 ``` 里传进来。原样丢给 iverilog 必然编不过，
    而那不是它设计能力的问题 —— 剥掉围栏，别让格式噪声吃掉一次工具调用。"""
    blocks = _FENCED.findall(code or "")
    return (blocks[-1] if blocks else (code or "")).strip()


def _run(argv: list[str], work: Path) -> tuple[int, str]:
    """跑一个只做静态分析的工具。env 只留 PATH：作业容器里有平台注入的端点与
    凭据，没有理由出现在模型生成的代码面前。"""
    try:
        proc = subprocess.run(
            argv, cwd=str(work), timeout=_TIMEOUT,
            capture_output=True, text=True, errors="replace",
            env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")},
        )
    except subprocess.TimeoutExpired:
        return -1, f"超时（{_TIMEOUT}s）"
    except OSError as exc:
        return -1, f"工具不可用：{exc}（镜像里没装？见实验 README「镜像要装什么」）"
    return proc.returncode, f"{proc.stdout}\n{proc.stderr}".strip()


@function_tool("compile_rtl")
def compile_rtl(code: str) -> str:
    """用 Icarus Verilog 编译一段 SystemVerilog 模块，返回编译是否通过与报错信息。

    Args:
        code: 完整的模块源码，从 module 到 endmodule。不要只给片段或 diff。
    """
    source = _clean(code)
    if "module" not in source:
        return "[编译] 没有收到模块源码：请把完整的 module … endmodule 传进来。"
    with tempfile.TemporaryDirectory(prefix="rtl-tool-") as tmp:
        work = Path(tmp)
        (work / "design.sv").write_text(source, encoding="utf-8")
        code_, output = _run(
            ["iverilog", "-g2012", "-o", "design.out", "design.sv"], work
        )
    if code_ == 0:
        return "[编译] 通过，无语法错误。"
    return f"[编译] 失败：\n{output[:_MAX_CHARS]}"


@function_tool("lint_rtl")
def lint_rtl(code: str) -> str:
    """用 Verilator 静态检查一段 SystemVerilog，报告推断出的锁存器、位宽不匹配等综合期问题。

    Args:
        code: 完整的模块源码，从 module 到 endmodule。
    """
    source = _clean(code)
    if "module" not in source:
        return "[lint] 没有收到模块源码：请把完整的 module … endmodule 传进来。"
    with tempfile.TemporaryDirectory(prefix="rtl-lint-") as tmp:
        work = Path(tmp)
        (work / "design.sv").write_text(source, encoding="utf-8")
        # --lint-only：只分析不生成 C++，也不执行任何东西。
        # -Wall 打开 LATCH / WIDTH 这类默认关掉的告警 —— 那正是奖励综合段要扣的东西。
        code_, output = _run(["verilator", "--lint-only", "-Wall", "design.sv"], work)
    if code_ == 0 and not output:
        return "[lint] 干净，没有发现锁存器 / 位宽 / 组合环问题。"
    return f"[lint] 发现问题：\n{output[:_MAX_CHARS]}"
