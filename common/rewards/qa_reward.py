#!/usr/bin/env python3
r"""题库 GRPO 规则奖励函数（配 build_qa_dataset.py 产出的数据集）。

为什么不用 math_verify：单选/多选/判断的答案是字母集合，填空是有序短语，
简答是要点覆盖，math_verify 面向数学表达式判不准。这里按 expected_answer 的
[type] 前缀分派规则判分，全程客观、无需调用模型。

接口对齐 llm_train.sh 里 math_rule_reward_fn(queries, completions, expected_answers)：
返回与输入等长的 float 列表。

判分规则（模型答案须写进 \boxed{...}，否则 FORMAT_PENALTY）：
    single/bool : 字母完全相等 → 1.0 否则 0.0
    multiple    : 见 MULTI_MODE（可用环境变量 QA_MULTI_MODE 覆盖）
                  - "exact"          字母集合完全相等才 1.0（最稳，防刷）
                  - "partial_penalty"(默认) (选对 − w·选错)/应选数，截断[0,1]
                                     w=MULTI_WRONG_WEIGHT（环境变量 QA_MULTI_WRONG_WEIGHT，默认 0.5）。
                                     w=1.0 是旧行为（错一个抵一个、全选易归零）；w=0.5 更平滑、
                                     对"多选对但带一两个错"给部分分，梯度更密、仍惩罚乱选/防全选刷分。
                  - "f1"             预测集合与正确集合的 F1（对漏选/多选都平滑，但全选有保底分易被刷）
    fill        : 逐空匹配，reward = 答对空数 / 总空数。每空接受“/”拆分的多种写法
                  + synonyms.json 同义词扩展
    short       : 关键词覆盖率 = 命中要点数 / 总要点数。要点用子串匹配（含/同义词扩展）

同义词表：与本文件同目录的 synonyms.json（可选）。格式见文件示例：
    {"groups": [["对","正确","yes"], ["错","错误","no"]], "version": 1}
同组词互为等价；填空和简答匹配时自动扩展。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

FORMAT_PENALTY = -0.5
# 多选判分模式与错选惩罚权重均可用环境变量覆盖，便于实验快速切换而不改代码。
MULTI_MODE = os.environ.get("QA_MULTI_MODE", "partial_penalty")  # "exact" | "partial_penalty" | "f1"
MULTI_WRONG_WEIGHT = float(os.environ.get("QA_MULTI_WRONG_WEIGHT", "0.5"))  # partial_penalty 里每个错选的扣分权重
SHORT_MIN_KW_LEN = 1            # 过短要点（<该长度，归一化后）不计入分母，避免噪声
# simple 题「关键词覆盖率」在哪段文本上统计：
#   "completion"（默认，单轮 baseline 的历史行为）：boxed + 整段回答都算覆盖。
#   "boxed"（多轮检索 Agent 必须用这个）：只认 \boxed{} 里的内容。
# ⚠️ 为什么 Agent 场景必须切 "boxed"：检索环境会把资料原文回灌给模型，模型只要在回答里
#    大段复述检索片段，就能把 gold 要点全部「覆盖」到——这是单轮无工具 baseline 不存在的
#    刷分通道，而 make_eval_cfg() 只归零 reward shaping、管不到这里，验证分会一起被抬高，
#    A/B 对比失真。切成 boxed 后模型必须自己把要点提炼进答案框，才算答对。
SHORT_SCOPE = os.environ.get("QA_SHORT_SCOPE", "completion")  # "completion" | "boxed"
_SYN_PATH = Path(__file__).parent / "synonyms.json"

_BOXED = re.compile(r"\\boxed\s*\{")
_PUNCT = re.compile(r"[，,。．\.、；;：:！!？\?\"'`（）()【】\[\]/／\\\s]+")


# ---------- 同义词表 ----------
def _load_synonyms() -> list[set[str]]:
    if not _SYN_PATH.exists():
        return []
    raw = json.loads(_SYN_PATH.read_text(encoding="utf-8"))
    return [{_norm(w) for w in g if _norm(w)} for g in raw.get("groups", [])]


def _expand_syn(alts: set[str], groups: list[set[str]]) -> set[str]:
    out = set(alts)
    for g in groups:
        if out & g:
            out |= g
    return out


# ---------- 文本归一化 ----------
def _norm(s: str) -> str:
    """去空白/标点/斜杠、转小写。用于宽松匹配。"""
    return _PUNCT.sub("", str(s).strip().lower().replace("\u3000", " "))


def _alts_for_blank(raw_blank: str, groups: list[set[str]]) -> set[str]:
    """一个空的可接受写法：整体 + 以 / 拆分的各部分（只增不减），再做同义扩展。"""
    parts = re.split(r"[/／]", raw_blank)
    alts = {_norm(raw_blank)} | {_norm(p) for p in parts}
    alts.discard("")
    return _expand_syn(alts, groups)


# ---------- \boxed 提取 ----------
def extract_boxed(text: str) -> str | None:
    """提取最后一个 \boxed{...}，正确处理嵌套花括号。"""
    last = None
    for m in _BOXED.finditer(text):
        i, depth, buf = m.end(), 1, []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(ch)
            i += 1
        if depth == 0:
            last = "".join(buf)
    return last


def _letters(s: str) -> set[str]:
    return set(re.findall(r"[A-Z]", s.upper()))


def _grade_multiple(pred: set[str], gold: set[str]) -> float:
    if MULTI_MODE == "exact":
        return 1.0 if pred == gold else 0.0
    if MULTI_MODE == "f1":
        if not pred and not gold:
            return 1.0
        tp = len(pred & gold)
        if tp == 0:
            return 0.0
        prec, rec = tp / len(pred), tp / len(gold)
        return 2 * prec * rec / (prec + rec)
    # partial_penalty（默认）：选对得分，选错按 MULTI_WRONG_WEIGHT 扣分；截断到 [0,1]。
    # w<1 比旧的 w=1 更平滑——"全对但多带一两个错选"仍能拿到大部分分，给 RL 更密的梯度，
    # 同时保留对乱选/全选的惩罚（gold 远小于选项数时，全选仍会被扣到 0），不至于被刷分。
    if not gold:
        return 0.0
    correct = len(pred & gold)
    wrong = len(pred - gold)
    return max(0.0, (correct - MULTI_WRONG_WEIGHT * wrong) / len(gold))


def _grade_one(expected: str, completion: str, groups: list[set[str]]) -> float:
    boxed = extract_boxed(completion)
    if boxed is None:
        return FORMAT_PENALTY

    m = re.match(r"\s*\[(\w+)\]\s*(.*)", expected, flags=re.DOTALL)
    if not m:
        return 1.0 if _norm(boxed) == _norm(expected) else 0.0
    qtype, gold = m.group(1), m.group(2).strip()

    if qtype in ("single", "bool"):
        return 1.0 if _letters(boxed) == _letters(gold) else 0.0

    if qtype == "multiple":
        return _grade_multiple(_letters(boxed), _letters(gold))

    if qtype == "fill":
        gold_blanks = [b for b in gold.split("|||")]
        pred_parts = [p for p in re.split(r"[;；\n]", boxed) if p.strip()]
        if not gold_blanks:
            return 0.0
        hit = 0
        for k, gb in enumerate(gold_blanks):
            if k >= len(pred_parts):
                break
            if _norm(pred_parts[k]) in _alts_for_blank(gb, groups):
                hit += 1
        return hit / len(gold_blanks)

    if qtype == "short":
        kws = [k for k in gold.split("|||") if k.strip()]
        kws = [k for k in kws if len(_norm(k)) >= SHORT_MIN_KW_LEN]
        if not kws:
            return 0.0
        # SHORT_SCOPE="boxed" 时只认答案框，防止「复述检索原文」刷覆盖率（见文件头注释）。
        ans_norm = (
            _norm(boxed)
            if SHORT_SCOPE == "boxed"
            else _norm(boxed) + "|" + _norm(completion)
        )
        hit = 0
        for k in kws:
            alts = _alts_for_blank(k, groups)
            if any(a and a in ans_norm for a in alts):
                hit += 1
        return hit / len(kws)

    return 0.0


def qa_rule_reward_fn(queries, completions, expected_answers, **kwargs):
    """NeMo-RL 规则奖励入口。三个等长列表，返回 float 列表。"""
    groups = _load_synonyms()
    return [_grade_one(exp, comp, groups)
            for comp, exp in zip(completions, expected_answers, strict=False)]


if __name__ == "__main__":
    g = _load_synonyms()
    cases = [
        ("[single] B", r"答案 \boxed{B}", 1.0),
        ("[single] B", r"\boxed{C}", 0.0),
        ("[single] B", r"我觉得是 B", FORMAT_PENALTY),
        ("[multiple] A,C,D", r"\boxed{D, A, C}", 1.0),
        ("[multiple] A,C,D", r"\boxed{A,C}", 2 / 3),         # 漏选: (2-0)/3
        ("[multiple] A,C,D", r"\boxed{A,B,C,D}", 2.5 / 3),   # 全选(多1错): (3-0.5·1)/3，w=0.5
        ("[bool] A", r"\boxed{A}", 1.0),
        ("[fill] 拒收/reject ||| 特采/waive ||| 放行/Release",
         r"\boxed{reject; 特采; Release}", 1.0),            # /拆分备选生效
        ("[fill] 正向 ||| 3V ||| 0V ||| 0.7V",
         r"\boxed{正向; 3V; 1V; 0.7V}", 0.75),
        ("[short] 低温/掺杂 ||| 纯度高 ||| 横向扩散小",
         r"离子注入是低温工艺，纯度高。\boxed{低温; 纯度高}", 2 / 3),
    ]
    ok = True
    for exp, comp, want in cases:
        got = _grade_one(exp, comp, g)
        flag = "OK " if abs(got - want) < 1e-9 else "FAIL"
        ok = ok and flag == "OK "
        print(f"{flag} want={want:.4f} got={got:.4f}  {exp[:40]!r}")
    print(f"MULTI_MODE={MULTI_MODE}", "| ALL PASS" if ok else "| SELFTEST FAILED")
