"""题库 GRPO 规则奖励的单元测试（守住 boxed 解析 / 各题型判分）。"""
from __future__ import annotations

import pytest

from common.rewards import qa_reward
from common.rewards.qa_reward import (
    FORMAT_PENALTY,
    extract_boxed,
    qa_rule_reward_fn,
)


def grade(expected: str, completion: str) -> float:
    return qa_rule_reward_fn([""], [completion], [expected])[0]


def test_extract_boxed_simple():
    assert extract_boxed(r"答案是 \boxed{B}") == "B"


def test_extract_boxed_nested_braces():
    assert extract_boxed(r"\boxed{a_{1} + b}") == "a_{1} + b"


def test_extract_boxed_takes_last():
    assert extract_boxed(r"\boxed{A} 再想想 \boxed{C}") == "C"


def test_extract_boxed_none():
    assert extract_boxed("没有框") is None


@pytest.mark.parametrize(
    "expected,completion,want",
    [
        ("[single] B", r"答案 \boxed{B}", 1.0),
        ("[single] B", r"\boxed{C}", 0.0),
        ("[single] B", r"我觉得是 B", FORMAT_PENALTY),  # 没写 boxed → 格式罚分
        ("[bool] A", r"\boxed{A}", 1.0),
        ("[multiple] A,C,D", r"\boxed{D, A, C}", 1.0),
        ("[multiple] A,C,D", r"\boxed{A,C}", 2 / 3),         # 漏选: (2-0)/3
        ("[multiple] A,C,D", r"\boxed{A,B,C,D}", 2.5 / 3),   # 全选(多1错): (3-0.5·1)/3，w=0.5
        (
            "[fill] 拒收/reject ||| 特采/waive ||| 放行/Release",
            r"\boxed{reject; 特采; Release}",
            1.0,
        ),
        ("[fill] 正向 ||| 3V ||| 0V ||| 0.7V", r"\boxed{正向; 3V; 1V; 0.7V}", 0.75),
        (
            "[short] 低温/掺杂 ||| 纯度高 ||| 横向扩散小",
            r"离子注入是低温工艺，纯度高。\boxed{低温; 纯度高}",
            2 / 3,
        ),
    ],
)
def test_grade_cases(expected, completion, want):
    assert grade(expected, completion) == pytest.approx(want)


def test_reward_fn_returns_same_length():
    comps = [r"\boxed{B}", r"\boxed{C}", "无框"]
    exps = ["[single] B", "[single] B", "[single] B"]
    out = qa_rule_reward_fn(["", "", ""], comps, exps)
    assert len(out) == 3
    assert out[0] == 1.0 and out[1] == 0.0 and out[2] == FORMAT_PENALTY


# ─────────── short 题的覆盖率统计范围（QA_SHORT_SCOPE / short_answer_scope）───────────
# 多轮检索环境会把资料原文回灌给模型。若整段回答都算关键词覆盖（"completion"），
# 模型只要复述检索片段就能刷满 short 分——单轮 baseline 没有这条通道，
# 且 make_eval_cfg() 只归零 reward shaping、管不到判分口径，验证分会一起虚高。

_SHORT_GOLD = "[short] 低温/掺杂 ||| 纯度高 ||| 横向扩散小"
# 模型把检索到的资料整段抄进正文，答案框里只写了一个要点
_PARROTED = r"资料原文：离子注入是低温工艺，纯度高，横向扩散小。\boxed{低温}"


def test_short_scope_completion_is_gamed_by_parroting(monkeypatch):
    """回归：completion 口径下，复述检索原文能拿满分——这正是要堵的刷分通道。"""
    monkeypatch.setattr(qa_reward, "SHORT_SCOPE", "completion")
    assert grade(_SHORT_GOLD, _PARROTED) == pytest.approx(1.0)


def test_short_scope_boxed_requires_answer_in_box(monkeypatch):
    """boxed 口径下只认答案框：抄原文不再得分，必须自己把要点提炼进 \\boxed{}。"""
    monkeypatch.setattr(qa_reward, "SHORT_SCOPE", "boxed")
    assert grade(_SHORT_GOLD, _PARROTED) == pytest.approx(1 / 3)
    # 真正把要点写进答案框的，仍然照常得分
    assert grade(_SHORT_GOLD, r"\boxed{低温; 纯度高; 横向扩散小}") == pytest.approx(1.0)


def test_short_scope_does_not_affect_other_question_types(monkeypatch):
    """开关只作用于 short：其余题型本来就只看 \\boxed{}，不能被牵连。"""
    monkeypatch.setattr(qa_reward, "SHORT_SCOPE", "boxed")
    assert grade("[single] B", r"分析一堆 C D \boxed{B}") == pytest.approx(1.0)
    assert grade("[fill] 正向 ||| 3V", r"正文里写了 3V \boxed{正向; 3V}") == pytest.approx(1.0)
