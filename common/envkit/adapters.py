"""Gym 协议的参考适配器：与训练主线共用同一内核。

  ToolEnvAdapter    example_tool_env 的双协议参考（calc/search/python + 数值判分）
  QADocsAdapter     qa_docs_agent_env 的双协议参考（docs_search + 关键词/裁判判分）
  RtlEnvAdapter     RTL/FPGA agent 环境骨架（compile/simulate 经沙箱执行，
                    verify 留待接真实测试台 —— 见类 docstring 的接入路径）
"""
from __future__ import annotations

from typing import Any, Callable

from common.envkit.tools import TOOLS, safe_eval


class ToolEnvAdapter:
    """多工具数值题环境（example_tool_env 语义）。

    payload: {"question": str, "target": float, "kb": {...}, "answer_tolerance": float}
    verify:  response 是最终答案文本（数值表达式），按容差判对错。
    """

    name = "tool-env"

    def __init__(self) -> None:
        self.tools: dict[str, Callable[[str, dict], str]] = dict(TOOLS)

    def seed_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "target" not in payload:
            raise ValueError("payload 缺少 target（正确答案数值）")
        return {
            "question": str(payload.get("question") or ""),
            "target": float(payload["target"]),
            "answer_tolerance": float(payload.get("answer_tolerance", 1e-6)),
            "kb": dict(payload.get("kb") or {}),
            "code_timeout": float(payload.get("code_timeout", 5)),
        }

    def verify(self, ctx: dict[str, Any], response: str) -> dict[str, Any]:
        try:
            pred = safe_eval(response.strip())
            correct = abs(pred - ctx["target"]) <= ctx["answer_tolerance"]
        except Exception:  # noqa: BLE001 — 答案不可解析 = 答错，不是服务错误
            return {"reward": 0.0, "correct": False, "reason": "答案不可解析"}
        return {"reward": 1.0 if correct else 0.0, "correct": correct}


class QADocsAdapter:
    """检索问答环境（qa_docs_agent_env 语义的 Gym 腿）。

    payload: {"question": str, "expected": str}
    tools:   docs_search —— 由构造方注入检索函数（BM25 索引 / 平台文档服务），
             与训练主线的检索实现共用。
    verify:  复用 qa_reward 判分（有裁判凭据时可换 qa_judge_reward）。
    """

    name = "qa-docs"

    def __init__(self, search_fn: Callable[[str], str]) -> None:
        self._search = search_fn
        self.tools: dict[str, Callable[[str, dict], str]] = {
            "docs_search": self._docs_search,
        }

    def _docs_search(self, arg: str, ctx: dict[str, Any]) -> str:
        query = arg.strip()
        if not query:
            return "search 错误: 查询为空"
        ctx["search_count"] = int(ctx.get("search_count", 0)) + 1
        return self._search(query)

    def seed_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        expected = str(payload.get("expected") or "").strip()
        if not expected:
            raise ValueError("payload 缺少 expected（参考答案）")
        return {
            "question": str(payload.get("question") or ""),
            "expected": expected,
            "search_count": 0,
        }

    def verify(self, ctx: dict[str, Any], response: str) -> dict[str, Any]:
        from common.rewards import qa_reward

        groups = qa_reward._load_synonyms()
        reward = float(qa_reward._grade_one(ctx["expected"], response, groups))
        return {"reward": reward, "search_count": ctx.get("search_count", 0)}


class RtlEnvAdapter:
    """RTL/FPGA agent 环境骨架。

    接入路径（真实化时按序替换 stub）：
      1. compile 工具：把 iverilog/verilator 编译命令写进沙箱代码（HttpSandboxProvider
         指向装有 EDA 工具链的沙箱镜像），返回编译诊断。
      2. simulate 工具：跑 testbench，返回波形摘要/断言结果。
      3. verify：按 testbench 通过率给 reward（部分分），可叠加 lint 惩罚。
    当前 stub 仅回显协议形状，够 rollout 框架联调；不产生有效训练信号。
    """

    name = "rtl-env"

    def __init__(self) -> None:
        self.tools: dict[str, Callable[[str, dict], str]] = {
            "compile": self._compile,
            "simulate": self._simulate,
        }

    def _compile(self, arg: str, ctx: dict[str, Any]) -> str:
        ctx["last_source"] = arg
        # stub：真实实现把编译命令交给沙箱执行（见类 docstring）。
        if "module" not in arg:
            return "compile 错误: 未发现 module 定义"
        return "compile ok（stub：接入 EDA 沙箱后返回真实诊断）"

    def _simulate(self, arg: str, ctx: dict[str, Any]) -> str:
        if not ctx.get("last_source"):
            return "simulate 错误: 先用 compile 提交 RTL 源码"
        return "simulate ok（stub：接入 testbench 后返回断言结果）"

    def seed_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "spec": str(payload.get("spec") or ""),
            "testbench": str(payload.get("testbench") or ""),
        }

    def verify(self, ctx: dict[str, Any], response: str) -> dict[str, Any]:
        del response
        # stub：真实实现按 testbench 通过率给部分分。
        return {"reward": 0.0, "reason": "RTL 环境骨架：verify 未接真实测试台"}
