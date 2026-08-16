"""模型输出的标签协议解析（<tool>…</tool> / <answer>…</answer> / <search>…</search>）。

这是全部环境共用的唯一实现 —— 语义细节（闭标签缺失仍取全文）曾在两个环境里
各写一份并漂移出 bug，收口到这里。
"""
from __future__ import annotations

from typing import Optional


def extract_tag(text: str, tag: str) -> Optional[str]:
    """取最后一个 <tag>...</tag> 的内容；没有开标签则 None。

    闭标签缺失时仍取开标签后全文：vLLM/NeMo-RL 用 stop_strings=["</tag>"]
    截断时默认不把停止串写进生成文本，若这里要求成对标签，每一轮都会被误判
    「格式不对」，整个实验拿不到有效奖励信号。
    """
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    s = text.rfind(open_t)
    if s == -1:
        return None
    e = text.find(close_t, s + len(open_t))
    body = text[s + len(open_t) : (e if e != -1 else None)].strip()
    return body if body else ("" if e != -1 else None)
