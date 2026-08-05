"""按题分组的验证指标：avg@N / pass@N / majority@N。

口径要和论文 `eval/evaluate_math.py` 对得上，否则复现出的数字没法逐项对照。
这里把三个指标的定义、以及 majority 的「归一化只用于分组、判分仍用真实 reward」
这条关键设计，全部锁成用例。
"""
from __future__ import annotations

import pytest

from common.algorithms.opsd_eval import (
    ValSample,
    extract_final_answer,
    grouped_metrics,
    normalize_answer,
)


def _s(pid, reward, answer=None):
    return ValSample(problem_id=pid, reward=reward, answer=answer)


# ---------------------------------------------------------------- 三个指标
def test_empty_returns_empty():
    """没样本时返回空 dict，让调用方跳过上报——别往曲线里写个 0。"""
    assert grouped_metrics([]) == {}


def test_avg_pass_majority_on_a_worked_example():
    # 题 A：3 条里 2 条对，且对的答案 "5" 占多数 → pass ✓ majority ✓
    # 题 B：3 条里 1 条对，但错答案 "9" 占多数     → pass ✓ majority ✗
    # 题 C：3 条全错                               → pass ✗ majority ✗
    samples = [
        _s("A", 1.0, "5"), _s("A", 1.0, "5"), _s("A", 0.0, "7"),
        _s("B", 1.0, "4"), _s("B", 0.0, "9"), _s("B", 0.0, "9"),
        _s("C", 0.0, "1"), _s("C", 0.0, "2"), _s("C", 0.0, "3"),
    ]
    m = grouped_metrics(samples)
    assert m["avg_at_n"] == pytest.approx(3 / 9)      # 总正确/总生成
    assert m["pass_at_n"] == pytest.approx(2 / 3)     # A、B 至少一条对
    assert m["majority_at_n"] == pytest.approx(1 / 3) # 只有 A 的众数是对的
    assert m["num_problems"] == 3
    assert m["samples_per_problem"] == pytest.approx(3.0)


def test_pass_at_n_is_upper_bound_of_avg():
    """pass@N ≥ avg@N 恒成立——这是定义决定的，也是「会做但不稳」的判据。"""
    samples = [_s("A", 1.0, "1")] + [_s("A", 0.0, "2")] * 9
    m = grouped_metrics(samples)
    assert m["avg_at_n"] == pytest.approx(0.1)
    assert m["pass_at_n"] == 1.0


def test_samples_per_problem_exposes_missing_repeat():
    """repeat 没配上时这个值会是 1.0 —— 一眼看出验证退化成 avg@1。"""
    m = grouped_metrics([_s("A", 1.0, "1"), _s("B", 0.0, "2")])
    assert m["samples_per_problem"] == pytest.approx(1.0)


def test_correct_threshold_respected():
    samples = [_s("A", 0.4, "1"), _s("A", 0.6, "1")]
    assert grouped_metrics(samples)["avg_at_n"] == pytest.approx(0.5)
    assert grouped_metrics(samples, correct_threshold=0.7)["avg_at_n"] == 0.0


# ---------------------------------------------------------------- majority 的判分来源
def test_majority_correctness_comes_from_reward_not_string_match():
    """★ 关键设计：归一化只用于分组，正确性取该组样本的真实 reward。

    否则我这个轻量归一化就变成了判分器，`\\frac{1}{2}` vs `0.5` 之类的分歧
    会直接产出错误的 majority 数字。
    """
    # 众数是 "\frac{1}{2}"（2 票），判分器认为它是对的（reward=1）
    samples = [
        _s("A", 1.0, r"\frac{1}{2}"),
        _s("A", 1.0, r"\dfrac{1}{2}"),  # 归一化后与上一条同组
        _s("A", 0.0, "0.5"),            # 判分器可能也认对，但这里 reward=0，独立成组
    ]
    m = grouped_metrics(samples)
    assert m["majority_at_n"] == 1.0


def test_majority_false_when_winning_group_is_wrong():
    samples = [_s("A", 0.0, "9"), _s("A", 0.0, "9"), _s("A", 1.0, "4")]
    assert grouped_metrics(samples)["majority_at_n"] == 0.0


def test_majority_ignores_unparsed_answers():
    """没写出 \\boxed{} 的样本不参与投票，但仍计入 avg@N（它确实答错了）。"""
    samples = [_s("A", 0.0, None), _s("A", 0.0, None), _s("A", 1.0, "7")]
    m = grouped_metrics(samples)
    assert m["majority_at_n"] == 1.0          # 唯一有效票是 "7"，且它是对的
    assert m["avg_at_n"] == pytest.approx(1 / 3)
    assert m["unparsed_answer_rate"] == pytest.approx(2 / 3)


def test_majority_false_when_no_parsable_answer():
    assert grouped_metrics([_s("A", 0.0, None)] * 3)["majority_at_n"] == 0.0


def test_unparsed_rate_flags_format_collapse():
    """这个比例偏高说明准确率低是「没按格式作答」而非「不会做」，两者处方完全不同。"""
    m = grouped_metrics([_s("A", 0.0, None)] * 9 + [_s("A", 1.0, "3")])
    assert m["unparsed_answer_rate"] == pytest.approx(0.9)


# ---------------------------------------------------------------- 归一化
@pytest.mark.parametrize(
    "a,b",
    [
        ("5", " 5 "),
        ("5", "5."),
        ("$5$", "5"),
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"\tfrac{1}{2}", r"\frac{1}{2}"),
        (r"\left(1,2\right)", "(1,2)"),
        ("A B", "AB"),
        ("Foo", "foo"),
    ],
)
def test_normalize_treats_these_as_same(a, b):
    assert normalize_answer(a) == normalize_answer(b)


@pytest.mark.parametrize("a,b", [("1/2", "0.5"), ("5", "50"), ("(1,2)", "(2,1)")])
def test_normalize_keeps_these_distinct(a, b):
    """刻意保守：不做数值求值。宁可低估 majority，也不要把不同答案错误合并。"""
    assert normalize_answer(a) != normalize_answer(b)


def test_normalize_handles_none_and_empty():
    assert normalize_answer(None) == ""
    assert normalize_answer("") == ""


# ---------------------------------------------------------------- 答案抽取
def test_extract_final_answer_takes_last_assistant_turn():
    log = [
        {"role": "user", "content": "题目 \\boxed{999}"},           # 题面里的不算
        {"role": "assistant", "content": "先试试 \\boxed{3}"},
        {"role": "assistant", "content": "重新算，答案是 \\boxed{7}"},
    ]
    assert extract_final_answer(log) == "7"


def test_extract_final_answer_handles_nested_braces():
    log = [{"role": "assistant", "content": r"答案 \boxed{\frac{1}{2}}"}]
    assert extract_final_answer(log) == r"\frac{1}{2}"


def test_extract_final_answer_none_when_missing():
    assert extract_final_answer([{"role": "assistant", "content": "忘了写框"}]) is None
    assert extract_final_answer([]) is None
    assert extract_final_answer(None) is None
