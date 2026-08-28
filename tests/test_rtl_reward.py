"""RTL 三段式奖励：编译 → 仿真 → 综合。

盯的是「这条奖励能不能真的训得动」，而不只是「函数返回了一个数」：

  部分分      仿真给 1 - 错/总。二值化会让早期 rollout 全是 0，组内奖励恒等，
              GRPO 优势全为 0 —— 训练照跑但什么都学不到
  分段可见    总分一个数分不出「一直卡在编译」和「一直卡在综合」
  失败要响    缺工具、没 testbench 必须报错，不能静默降级
  不吃宿主环境 会执行模型生成的代码，env 里不该有平台注入的端点与凭据
"""
from __future__ import annotations

import pytest

from common.rewards.rtl_reward import (
    DEFAULT_WEIGHTS,
    FORMAT_PENALTY,
    SIMULATION,
    SYNTAX,
    SYNTHESIS,
    RewardConfigError,
    RtlReward,
    ToolRun,
    extract_verilog,
    read_simulation,
    rtl_reward_fn,
    run_tool,
    synthesis_problems,
)

DESIGN = "module adder(input a, output b); assign b = a; endmodule"
TB = "module tb; initial $finish; endmodule"


class FakeTools:
    """按 argv[0] 给结果的假执行器。"""

    def __init__(self, **outcomes: ToolRun):
        self.outcomes = outcomes
        self.calls: list[list[str]] = []

    def __call__(self, argv, cwd, timeout):
        self.calls.append(list(argv))
        return self.outcomes.get(argv[0], ToolRun(code=0))


def _reward(tools, **over) -> RtlReward:
    return RtlReward(runner=tools, which=lambda _n: "/usr/bin/fake", **over)


# ── 一条能往上爬的坡 ────────────────────────────────────────────────────────


def test_full_pass_scores_one():
    out = _reward(FakeTools(vvp=ToolRun(0, "Mismatches: 0 in 100 samples"))).score(DESIGN, TB)
    assert out.total == pytest.approx(1.0)


def test_compile_failure_scores_zero_and_names_the_stage():
    out = _reward(FakeTools(iverilog=ToolRun(1, "syntax error"))).score(DESIGN, TB)
    assert out.total == 0.0
    assert out.stage(SIMULATION).skipped == "syntax 未通过"
    assert out.stage(SYNTHESIS).skipped == "syntax 未通过"


def test_compiles_but_simulation_fails_keeps_the_entry_ticket():
    """编过了就该拿到 0.1 —— 这一步的分是模型学会「先写出合法 Verilog」的抓手。"""
    out = _reward(FakeTools(vvp=ToolRun(0, "Mismatches: 100 in 100 samples"))).score(DESIGN, TB)
    assert out.total == pytest.approx(DEFAULT_WEIGHTS[SYNTAX])


def test_simulation_gives_partial_credit():
    """整套奖励最重要的一行：差 3 个和全错不是同一个数。"""
    out = _reward(FakeTools(vvp=ToolRun(0, "Mismatches: 3 in 100 samples"))).score(DESIGN, TB)
    assert out.stage(SIMULATION).score == pytest.approx(0.97)
    assert out.total == pytest.approx(0.1 + 0.7 * 0.97)


def test_partial_credit_is_monotonic():
    """错得越少分越高。不满足这条，GRPO 就是在爬一座乱序的山。"""
    totals = [
        _reward(FakeTools(vvp=ToolRun(0, f"Mismatches: {w} in 100 samples"))).score(DESIGN, TB).total
        for w in (90, 50, 10, 0)
    ]
    assert totals == sorted(totals)


def test_synthesis_catches_an_inferred_latch_that_simulation_cannot():
    """推断出的锁存器在 testbench 里看不出来，在芯片上是个 bug。

    yosys 把它报成 warning 后照样退出 0 —— 只看退出码等于放它过去。
    """
    tools = FakeTools(
        vvp=ToolRun(0, "Mismatches: 0 in 100 samples"),
        yosys=ToolRun(0, "Warning: inferring latch for signal `q'"),
    )
    out = _reward(tools).score(DESIGN, TB)
    assert out.stage(SIMULATION).ok is True
    assert out.stage(SYNTHESIS).ok is False
    assert out.total == pytest.approx(0.8)


def test_testbench_compile_failure_is_distinguished():
    """设计单独编得过、加上 testbench 编不过 —— 通常是端口名或位宽对不上。
    这是最常见的失败，要单独说清楚模型才有得改。"""
    calls = {"n": 0}

    def runner(argv, cwd, timeout):
        if argv[0] == "iverilog":
            calls["n"] += 1
            return ToolRun(0) if calls["n"] == 1 else ToolRun(1, "port mismatch")
        return ToolRun(0)

    out = RtlReward(runner=runner, which=lambda _n: "/bin/x").score(DESIGN, TB)
    assert out.stage(SYNTAX).ok is True
    assert "端口名/位宽" in out.stage(SIMULATION).detail


# ── 分段可见 ────────────────────────────────────────────────────────────────


def test_breakdown_always_carries_every_stage():
    out = _reward(FakeTools(iverilog=ToolRun(1))).score(DESIGN, TB)
    assert [s.name for s in out.stages] == [SYNTAX, SIMULATION, SYNTHESIS]


def test_to_dict_carries_contribution_not_just_score():
    body = _reward(FakeTools(vvp=ToolRun(0, "Mismatches: 50 in 100 samples"))).score(
        DESIGN, TB
    ).to_dict()
    assert body["stages"][SIMULATION]["score"] == pytest.approx(0.5)
    assert body["stages"][SIMULATION]["contribution"] == pytest.approx(0.35)


# ── 失败要响，不能静默降级 ──────────────────────────────────────────────────


def test_missing_tool_fails_at_construction():
    """静默少一段会让整轮训练在更小的尺度上跑，而那从 reward 曲线上看不出来。"""
    with pytest.raises(RewardConfigError) as exc:
        RtlReward(runner=FakeTools(), which=lambda n: None if n == "yosys" else "/bin/x")
    assert "yosys" in str(exc.value)


def test_disabling_a_stage_by_weight_needs_no_tool():
    reward = RtlReward(
        weights={SYNTAX: 0.2, SIMULATION: 0.8, SYNTHESIS: 0.0},
        runner=FakeTools(vvp=ToolRun(0, "Mismatches: 0 in 4 samples")),
        which=lambda n: None if n == "yosys" else "/bin/x",
    )
    out = reward.score(DESIGN, TB)
    assert out.stage(SYNTHESIS).skipped == "disabled"
    assert out.total == pytest.approx(1.0)


def test_all_zero_weights_is_rejected():
    with pytest.raises(RewardConfigError):
        RtlReward(weights=dict.fromkeys(DEFAULT_WEIGHTS, 0.0),
                  runner=FakeTools(), which=lambda _n: "/bin/x")


def test_unknown_stage_is_rejected():
    with pytest.raises(RewardConfigError):
        RtlReward(weights={"vibes": 1.0}, runner=FakeTools(), which=lambda _n: "/bin/x")


def test_missing_testbench_is_an_error_not_a_free_07():
    """没有 testbench 时 simulation 段会退化成「编译得过就给 0.7」。"""
    with pytest.raises(RewardConfigError) as exc:
        _reward(FakeTools()).score(DESIGN, "")
    assert "testbench" in str(exc.value)


# ── 从回复里抠代码 ──────────────────────────────────────────────────────────


def test_fenced_block_wins_over_surrounding_prose():
    """模型常在代码后继续解释；把解释一起喂给 iverilog 必然编不过。"""
    text = f"思路如下。\n```verilog\n{DESIGN}\n```\n这样就实现了进位。"
    assert extract_verilog(text) == DESIGN


def test_the_last_module_block_wins():
    """改错时模型常先复述旧代码再给新版本 —— 要的是新的那份。"""
    text = "旧版：\n```verilog\nmodule old; endmodule\n```\n新版：\n```verilog\nmodule new; endmodule\n```"
    assert "new" in extract_verilog(text)


def test_a_bare_module_without_fences_is_still_found():
    assert extract_verilog(f"这是实现：\n{DESIGN}\n完成。") == DESIGN


def test_no_code_at_all_returns_empty():
    assert extract_verilog("我认为应该用一个计数器。") == ""


def test_batch_fn_gives_format_penalty_when_no_code():
    """给负分而不是 0：0 和「写了但编不过」同分，模型学不到要按格式输出。"""
    scores = rtl_reward_fn(["没有代码"], [TB])
    assert scores == [FORMAT_PENALTY]


def test_batch_fn_rejects_ragged_inputs():
    with pytest.raises(ValueError):
        rtl_reward_fn(["a", "b"], [TB])


# ── 读 testbench 输出 ───────────────────────────────────────────────────────


def test_failure_marker_beats_a_zero_exit_code():
    """testbench 常常在失败时仍然退出 0；只看退出码会把全部失败判成通过。"""
    assert read_simulation(ToolRun(0, "ERROR: mismatch at t=40"))[:2] == (0.0, False)


def test_mismatch_count_beats_both_markers():
    score, ok, note = read_simulation(ToolRun(0, "Mismatches: 2 in 8 samples\nERROR"))
    assert score == pytest.approx(0.75)
    assert ok is False
    assert "2/8" in note


def test_timeout_scores_zero_and_says_why():
    """超时几乎总是组合环或少了 $finish —— 那是设计错误，不是「差一点」。"""
    score, ok, note = read_simulation(ToolRun(-1, timed_out=True))
    assert (score, ok) == (0.0, False)
    assert "组合环" in note


def test_zero_total_samples_does_not_divide_by_zero():
    assert 0.0 <= read_simulation(ToolRun(0, "Mismatches: 0 in 0 samples"))[0] <= 1.0


@pytest.mark.parametrize("line,label", [
    ("Warning: inferring latch for signal `q'", "推断出锁存器"),
    ("found combinational loop in module top", "组合环"),
    ("Warning: multiple driver for wire x", "多驱动"),
])
def test_synthesis_problems_are_named(line, label):
    assert synthesis_problems(line) == [label]


def test_clean_synthesis_output_has_no_problems():
    assert synthesis_problems("Executing SYNTH pass.\nChip area: 42") == []


# ── 沙箱 ────────────────────────────────────────────────────────────────────


def test_run_tool_does_not_inherit_the_host_environment(tmp_path, monkeypatch):
    """会执行模型生成的代码：作业容器里平台注入的端点与凭据不该出现在它面前。"""
    import sys

    monkeypatch.setenv("FORGE_INGEST_TOKEN", "leak-me")
    script = tmp_path / "probe.py"
    script.write_text("import os;print(os.environ.get('FORGE_INGEST_TOKEN',''))", encoding="utf-8")
    run = run_tool([sys.executable, str(script)], tmp_path, 30)
    assert run.code == 0
    assert "leak-me" not in run.output


def test_run_tool_reports_a_timeout_instead_of_hanging(tmp_path):
    import sys

    script = tmp_path / "hang.py"
    script.write_text("import time;time.sleep(30)", encoding="utf-8")
    assert run_tool([sys.executable, str(script)], tmp_path, 1).timed_out is True


def test_run_tool_reports_a_missing_binary(tmp_path):
    run = run_tool(["definitely-not-a-real-tool-xyz"], tmp_path, 5)
    assert run.code != 0
    assert "无法执行" in run.output
