"""题库自检脚本：在花 GPU 之前先确认 testbench 真的能判分。

最关键的一条是变异检查：**把参考实现改坏，分数必须掉下来**。判不出错的 testbench
不会让训练报错，只会让那道题全组同分 → GRPO 优势恒为 0 → 曲线平着，
看起来像「模型学不会」，其实是这道题根本没有信号。这类题查不出来就只能烧 GPU。

本机不一定装了 iverilog，所以按仓库惯例注入假执行器：假的 vvp 模拟一个**会判**的
testbench（看设计对不对给不同的 mismatch 数）和一个**瞎判**的 testbench（永远 0 错）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from common.rewards.rtl_reward import SIMULATION, SYNTAX, RtlReward, ToolRun

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "verl-grpo_qwen3.5-9b_rtl-agent_v1"

REFERENCE = "module adder2(input [1:0] a, input [1:0] b, output [2:0] s); assign s = a + b; endmodule"
GOOD_TB = (
    "module tb; integer e,n; adder2 dut(a,b,s);"
    ' initial begin $display("Mismatches: %0d in %0d samples", e, n); $finish; end endmodule'
)
SPEC = "实现 module adder2(input [1:0] a, input [1:0] b, output [2:0] s)，s = a + b。"


def _load():
    spec = importlib.util.spec_from_file_location("rtl_check_data", EXP / "check_data.py")
    module = importlib.util.module_from_spec(spec)
    # 先进 sys.modules 再 exec：check_data 里有 @dataclass，而 dataclasses 会去
    # sys.modules[cls.__module__] 找注解命名空间，不登记就 AttributeError。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_reward(*, blind: bool):
    """blind=True 模拟一个判不出错的 testbench：不管设计对不对都报 0 错。"""

    def runner(argv, cwd, timeout):
        if argv[0] != "vvp":
            return ToolRun(code=0)
        design = (Path(cwd) / "design.sv").read_text(encoding="utf-8")
        wrong = 0 if (blind or "a + b" in design) else 4
        return ToolRun(code=0, output=f"Mismatches: {wrong} in 4 samples")

    def build(top: str) -> RtlReward:
        return RtlReward(
            weights={SYNTAX: 0.1, SIMULATION: 0.9}, top=top,
            runner=runner, which=lambda _n: "/bin/fake",
        )

    return build


@pytest.fixture()
def checker(monkeypatch):
    module = _load()

    def _with(blind: bool = False):
        monkeypatch.setattr(module, "_reward", _fake_reward(blind=blind))
        return module

    return _with


def _row(**over) -> dict:
    return {"id": "adder2", "top": "adder2", "spec": SPEC,
            "reference": REFERENCE, "testbench": GOOD_TB, **over}


# ── 好样本 ──────────────────────────────────────────────────────────────────


def test_a_sound_problem_passes(checker):
    module = checker()
    report = module.check_row(0, _row())
    assert report.ok, report.errors
    assert report.reference_score == pytest.approx(1.0)
    assert report.mutant_scores, "应该真的生成了变异体"


# ── 最关键的一条：判不出错的 testbench ──────────────────────────────────────


def test_a_blind_testbench_is_caught(checker):
    """把参考实现改坏、分数却不掉 —— 这道题在训练里是纯噪声。"""
    module = checker(blind=True)
    report = module.check_row(0, _row())
    assert not report.ok
    assert any("判不出错" in e for e in report.errors)
    # 报告里要带上每个变异体的分数，否则人没法判断是 testbench 弱还是变异太温和。
    assert report.mutant_scores


def test_mutations_stay_syntactically_valid_operators():
    """变异要改行为、不能改成语法错误：那样掉分只证明编译器 work，
    不证明 testbench 会判。"""
    module = _load()
    mutated = dict(module._mutate(REFERENCE))
    assert mutated, "参考实现里应该有可改的算子"
    for name, source in mutated.items():
        assert source != REFERENCE, name
        assert "endmodule" in source, name


# ── testbench 本身的硬要求 ──────────────────────────────────────────────────


def test_a_testbench_without_a_mismatch_count_is_rejected(checker):
    """没有计数行就退化成二值奖励，早期 rollout 全 0，训不动。"""
    module = checker()
    report = module.check_row(0, _row(testbench="module tb; initial $finish; endmodule"))
    assert not report.ok
    assert any("Mismatches" in e for e in report.errors)


def test_a_missing_testbench_is_rejected(checker):
    module = checker()
    report = module.check_row(0, _row(testbench=""))
    assert not report.ok
    assert any("缺 testbench" in e for e in report.errors)


def test_a_testbench_without_finish_is_flagged(checker):
    """不 $finish 的 testbench 会挂住 vvp，超时判 0 —— 看起来像「设计错」。"""
    module = checker()
    tb = 'module tb; initial $display("Mismatches: 0 in 1 samples"); endmodule'
    report = module.check_row(0, _row(testbench=tb))
    assert any("$finish" in w for w in report.warnings)


def test_a_reference_that_does_not_score_full_marks_is_rejected(checker):
    """testbench 判错了自己的正确答案 —— 这道题永远拿不到满分。"""
    module = checker()
    report = module.check_row(0, _row(reference="module adder2(input a, output s); assign s = a; endmodule"))
    assert not report.ok
    assert any("不是满分" in e for e in report.errors)


# ── 缺字段的降级 ────────────────────────────────────────────────────────────


def test_without_a_reference_the_deep_checks_are_skipped_with_a_warning(checker):
    """没有参考实现就查不了最贵的那条，要说出来而不是默默放过。"""
    module = checker()
    report = module.check_row(0, _row(reference=""))
    assert report.ok
    assert any("reference" in w for w in report.warnings)
    assert report.mutant_scores == {}


def test_a_spec_that_omits_the_module_name_is_flagged(checker):
    """spec 没钉死接口 → 模型逻辑再对也连不上 testbench → 0.1 封顶。"""
    module = checker()
    report = module.check_row(0, _row(spec="做一个两位加法器。"))
    assert any("模块名" in w for w in report.warnings)


def test_a_missing_spec_is_an_error(checker):
    module = checker()
    report = module.check_row(0, _row(spec=""))
    assert not report.ok
