"""verl @function_tool 检索工具：在集群容器内 grep 本地 markdown 资料库。

官方机制（verl docs/sglang_multiturn/multiturn，v0.8 起）：
  - @function_tool 把普通 Python 函数注册为工具，schema 由
    transformers.utils.get_json_schema() 从类型注解 + Google 风格 docstring 推断，
    所以【每个参数必须有类型注解，且 docstring 必须有 Args 段】，缺了注册时直接报错。
  - config 的 rollout.multi_turn.function_tool_path 指向本文件（相对作业包根）。
  - 函数工具是无状态的（每次调用就是 fn(**parameters)）；需要按轨迹建/销毁状态
    （沙箱、临时目录）时改用 BaseTool + tool_config_path。

与 nemo-rl 版 QADocsAgentEnv 的 <search> 语义对齐：DOCS_DIR（默认 /data/docs）下
逐文件命中关键词，按命中行返回带出处的片段；目录不存在时返回占位提示，流程可跑通。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from verl.tools.function_tool import function_tool

_MAX_CHARS = int(os.environ.get("DOCS_MAX_CHARS", "900"))
_MAX_FILES = 3
_CONTEXT_LINES = 2


def _score_line(line: str, terms: list[str]) -> int:
    lowered = line.lower()
    return sum(1 for t in terms if t in lowered)


@function_tool("search_docs")
def search_docs(query: str) -> str:
    """在公司技术资料库（本地 markdown 文档）中检索关键词，返回带出处的相关片段。

    Args:
        query: 检索关键词，只保留核心术语，如 "CMP 铜 去除 步骤"。
    """
    docs_dir = Path(os.environ.get("DOCS_DIR", "/data/docs"))
    if not docs_dir.is_dir():
        return f"[检索结果] 资料目录 {docs_dir} 不存在，请基于已有知识作答。"

    terms = [t.lower() for t in re.split(r"[\s,，、]+", query.strip()) if t]
    if not terms:
        return "[检索结果] 检索词为空，请提供核心术语。"

    hits: list[tuple[int, str, int, list[str]]] = []
    for md in sorted(docs_dir.rglob("*.md")):
        lines = md.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            score = _score_line(line, terms)
            if score > 0:
                lo, hi = max(0, i - _CONTEXT_LINES), min(len(lines), i + _CONTEXT_LINES + 1)
                snippet = [f"L{j + 1}: {lines[j]}" for j in range(lo, hi) if lines[j].strip()]
                hits.append((score, md.name, i, snippet))

    if not hits:
        return f"[检索结果] 未找到与「{query}」相关的资料，可换关键词再试一次或直接作答。"

    hits.sort(key=lambda h: -h[0])
    out, seen_files = [], set()
    for _, name, _, snippet in hits:
        if name in seen_files:
            continue
        seen_files.add(name)
        out.append(f"【{name}】\n" + "\n".join(snippet))
        if len(seen_files) >= _MAX_FILES:
            break
    text = "[检索结果]\n" + "\n\n".join(out)
    return text[:_MAX_CHARS]
