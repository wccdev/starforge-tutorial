"""离线评测的装配逻辑（不碰 vLLM / Ray）。

这里测的是「把生成结果和判分结果拼回题目」这一步。它出错的方式很阴险：
reward 配到别人的答案上，指标看着完全正常，数字却全是错的。所以顺序契约必须锁死。
"""
from __future__ import annotations

import json

import pytest

from common.eval.math_eval import (
    DEFAULT_PROMPT_TEMPLATE,
    EvalSpec,
    assemble_samples,
    build_prompts,
    discover_eval_files,
    format_report,
    read_jsonl,
)


class _FakeTokenizer:
    """只实现 apply_chat_template，够 build_prompts 用。"""

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True,
                            add_special_tokens=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in chat)


# ---------------------------------------------------------------- prompt
def test_build_prompts_uses_template_and_chat_role():
    rows = [{"problem": "1+1=?"}, {"problem": "2+2=?"}]
    out = build_prompts(rows, EvalSpec(), _FakeTokenizer())
    assert len(out) == 2
    assert out[0].startswith("user:")
    assert "1+1=?" in out[0]
    assert "\\boxed" in out[0]  # 与训练时同构的作答格式要求


def test_build_prompts_includes_system_prompt():
    out = build_prompts(
        [{"problem": "x"}], EvalSpec(system_prompt="be brief"), _FakeTokenizer()
    )
    assert out[0].startswith("system:be brief|user:")


def test_default_template_matches_training_shape():
    """评测题面必须和训练时同构，否则测的是「适应新格式的能力」而非解题能力。"""
    assert "Please reason step by step" in DEFAULT_PROMPT_TEMPLATE
    assert "\\boxed" in DEFAULT_PROMPT_TEMPLATE


# ---------------------------------------------------------------- 装配
def test_assemble_samples_keeps_alignment():
    rows = [{"answer": "1"}, {"answer": "2"}]
    completions = [["a", "b"], ["c", "d", "e"]]   # 题 0 采 2 条，题 1 采 3 条
    scores = [1.0, 0.0, 0.0, 1.0, 0.0]
    extracted = ["1", "9", "7", "2", "8"]

    samples = assemble_samples(rows, completions, scores, extracted)

    assert [s.problem_id for s in samples] == ["0", "0", "1", "1", "1"]
    assert [s.reward for s in samples] == scores
    assert [s.answer for s in samples] == extracted


def test_assemble_samples_rejects_length_mismatch():
    """★ 长度对不上必须报错——静默错位会产出「看着正常但全错」的指标。"""
    rows = [{"answer": "1"}]
    with pytest.raises(ValueError, match="判分结果与生成数量不一致"):
        assemble_samples(rows, [["a", "b"]], [1.0], ["1"])


def test_assemble_samples_rejects_group_count_mismatch():
    with pytest.raises(ValueError, match="生成分组数"):
        assemble_samples([{"answer": "1"}], [["a"], ["b"]], [1.0, 0.0], ["1", "2"])


def test_assemble_then_group_gives_expected_metrics():
    """串起来验一次：2 题，题 0 全对、题 1 半对。"""
    from common.algorithms.opsd_eval import grouped_metrics

    rows = [{"answer": "1"}, {"answer": "2"}]
    samples = assemble_samples(
        rows,
        [["a", "b"], ["c", "d"]],
        [1.0, 1.0, 1.0, 0.0],
        ["1", "1", "2", "9"],
    )
    m = grouped_metrics(samples)
    assert m["avg_at_n"] == pytest.approx(0.75)
    assert m["pass_at_n"] == 1.0
    assert m["samples_per_problem"] == pytest.approx(2.0)


def test_assemble_handles_none_extracted():
    """判分器抛异常时 extracted 为 None——不能因此崩，应计入未抽出答案。"""
    from common.algorithms.opsd_eval import grouped_metrics

    samples = assemble_samples([{"answer": "1"}], [["a", "b"]], [0.0, 0.0], [None, None])
    assert grouped_metrics(samples)["unparsed_answer_rate"] == 1.0


# ---------------------------------------------------------------- 协议默认值
def test_spec_defaults_match_paper():
    """默认协议对齐论文 eval/run_eval.sh；改了要有意识地改，别悄悄漂移。"""
    spec = EvalSpec()
    assert spec.n == 12                 # --val_n 12
    assert spec.temperature == 1.0      # --temperature 1.0
    assert spec.max_tokens == 38912     # evaluate_math.py 默认


# ---------------------------------------------------------------- 文件发现 / 报告
def test_discover_eval_files(tmp_path):
    for name in ("eval_aime24.jsonl", "eval_hmmt25.jsonl", "val.jsonl", "eval_x.txt"):
        (tmp_path / name).write_text("")
    found = discover_eval_files(str(tmp_path))
    assert list(found) == ["aime24", "hmmt25"]  # 排序稳定，且只认 eval_*.jsonl


def test_discover_eval_files_missing_dir():
    assert discover_eval_files("/nonexistent/dir") == {}


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n")
    assert read_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


def test_format_report_is_single_line_and_has_all_three():
    line = format_report("aime24", {
        "avg_at_n": 0.25, "pass_at_n": 0.6, "majority_at_n": 0.4,
        "num_problems": 30.0, "samples_per_problem": 12.0,
        "unparsed_answer_rate": 0.05,
    })
    assert "\n" not in line
    assert "avg@N=0.250" in line and "pass@N=0.600" in line and "maj@N=0.400" in line
    assert "30 题 × 12" in line


def test_format_report_handles_empty_metrics():
    assert "无样本" in format_report("aime24", {})
