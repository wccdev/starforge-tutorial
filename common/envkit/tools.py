"""环境工具的内核实现（纯函数：`(arg, ctx) -> str`）。

代码执行统一经 SandboxProvider 分派：平台注入 STARFORGE_SANDBOX_ENDPOINT 时
走外部沙箱（E2B 兼容），否则容器内子进程 —— 环境代码不感知差异。
"""
from __future__ import annotations

import ast
import operator
from typing import Any

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """安全地计算一个纯算术表达式（只允许数字与 + - * / // % ** 和括号）。"""

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_ev(node.operand))
        raise ValueError("不支持的表达式")

    return _ev(ast.parse(expr, mode="eval"))


def tool_calc(arg: str, ctx: dict[str, Any]) -> str:
    try:
        return f"{safe_eval(arg):g}"
    except Exception as e:  # noqa: BLE001
        return f"calc 错误: {e}"


def tool_search(arg: str, ctx: dict[str, Any]) -> str:
    """在本题知识库（ctx['kb']: dict[str, str]）里按相关度检索，返回最相关条目。"""
    kb: dict[str, str] = ctx.get("kb", {}) or {}
    query = arg.strip().lower()
    if not query:
        return "search 错误: 查询为空"
    terms = [t for t in query.replace("，", " ").split() if t]
    scored: list[tuple[int, str]] = []
    for k, v in kb.items():
        text = f"{k} {v}".lower()
        score = sum(1 for t in terms if t in text)
        if query in k.lower():
            score += 2
        if score > 0:
            scored.append((score, v))
    if not scored:
        return "未检索到相关信息"
    top = max(s for s, _ in scored)
    hits = [v for s, v in scored if s == top]
    return " | ".join(dict.fromkeys(hits))[:500]


def tool_python(arg: str, ctx: dict[str, Any]) -> str:
    """执行一段 Python 代码，返回 stdout（经 SandboxProvider 分派，带超时）。"""
    timeout = float(ctx.get("code_timeout", 5))
    try:
        from starforge_sdk.sandbox import get_sandbox_provider

        result = get_sandbox_provider().run_code(arg, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — 沙箱层失败给模型可读反馈，不炸训练
        return f"python 错误: {e}"
    if result.error:
        return f"python 执行失败: {result.error}"
    out = result.stdout.strip()
    if out:
        return out[:500]
    err = result.stderr.strip()
    return f"(无 stdout) 报错: {err[:300]}" if err else "(无输出)"


#: 工具注册表：要加工具就往这里加 name -> callable(arg:str, ctx:dict)->str
TOOLS = {"calc": tool_calc, "search": tool_search, "python": tool_python}
