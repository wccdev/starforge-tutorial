#!/usr/bin/env python
"""题库自检：在花 GPU 之前，先确认每道题的 testbench 真的能判分。

用法：
  python experiments/verl-grpo_qwen3.5-9b_rtl-agent_v1/check_data.py \
      datasets/rtl_rl/train.jsonl
  # 只看坏的：--quiet；坏样本另存一份：--write-bad bad.jsonl

需要本机装 iverilog（brew install icarus-verilog / apt install iverilog）。
yosys、verilator 不是必需 —— 本脚本只查判分链路，不查综合。

为什么要有这一步
──────────────────────────────────────────────────────────────────────────────
奖励的 0.7 全压在 testbench 上。testbench 有毛病时训练**不会报错**，只会：

  · 判不出错（写错的实现也满分）→ 全组同分 → GRPO 优势恒为 0 → 曲线平着，
    看起来像"模型学不会"，其实是这道题根本没有信号
  · 不 $finish            → vvp 挂住 → 超时判 0 → 看起来像"设计错"
  · 不打印计数            → 退化成二值 → 早期 rollout 全 0，训不动
  · 接口和 spec 对不上    → 逻辑再对也编不过 → 0.1 封顶

这四件事都要跑一次才看得出来，而跑一次比训练便宜六个数量级。

五道检查
──────────────────────────────────────────────────────────────────────────────
  1. 字段齐全           spec / testbench / reference 在不在，top 和模块名对不对
  2. 参考实现编得过      连参考实现都编不过，题目本身就是坏的
  3. 参考实现拿满分      ★ 拿不到 1.0 说明 testbench 判错了自己的正确答案
  4. 变异实现掉分        ★★ 最关键：把参考实现改坏，分数必须掉下来。
                        不掉 = testbench 判不出错 = 这道题在训练里是纯噪声
  5. 能在超时内跑完      不 $finish 的 testbench 会拖死 rollout worker

第 3、4 两条要求样本里有 `reference`（参考实现）。没有 reference 的样本只跑
1、2、5，并给出警告 —— 那样最贵的第 4 条就查不了，强烈建议补上。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.rewards.rtl_reward import (  # noqa: E402
    SIMULATION,
    SYNTAX,
    RewardConfigError,
    RtlReward,
)

#: 变异算子：把参考实现改坏的几种最小改动。只要有**任意一个**能让分数掉下来，
#: 就说明 testbench 至少能判出一类错误。全都不掉分 = 它什么都判不出来。
#:
#: 挑的都是"语法仍然合法、行为一定变"的改动 —— 改成语法错误是作弊：
#: 那样掉分只能说明编译器work，不能说明 testbench 会判。
MUTATIONS: tuple[tuple[str, str, str], ...] = (
    ("反转所有加法", r"(?<![\w+])\+(?![\w+=])", "-"),
    ("反转相等判断", r"==", "!="),
    ("反转逻辑与或", r"&&", "||"),
    ("常量置零", r"(?<![\w'])1'b1", "1'b1 & 1'b0"),
    ("移位反向", r"<<", ">>"),
)

#: 变异后仍然算"通过"的分数线。参考实现应该是 1.0；变异体掉到这条线以下才算
#: testbench 判出来了。留一点余量：有的变异只影响少数用例，部分分会落在 0.9 上下。
MUTANT_MAX = 0.95

#: 单道题的检查预算。真训练时 rollout 会并发几十条，这里串行，给紧一点。
TIMEOUT = 60


@dataclass
class RowReport:
    index: int
    row_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reference_score: float | None = None
    mutant_scores: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _reward(top: str) -> RtlReward:
    # 只查判分链路，不查综合：yosys 不一定装了，而综合段的问题
    # （锁存器/组合环）是**设计**的问题，不是 testbench 的问题。
    return RtlReward(
        weights={SYNTAX: 0.1, SIMULATION: 0.9},
        top=top, compile_timeout=TIMEOUT, simulate_timeout=TIMEOUT,
    )


def _mutate(source: str) -> list[tuple[str, str]]:
    """生成变异体。只保留真的改动了源码的那些。"""
    out: list[tuple[str, str]] = []
    for name, pattern, replacement in MUTATIONS:
        mutated, count = re.subn(pattern, replacement, source, count=1)
        if count and mutated != source:
            out.append((name, mutated))
    return out


def check_row(index: int, item: dict) -> RowReport:
    row_id = str(item.get("id") or item.get("top") or f"#{index + 1}")
    report = RowReport(index=index, row_id=row_id)

    # 1. 字段
    spec = str(item.get("spec") or "").strip()
    testbench = str(item.get("testbench") or "").strip()
    reference = str(item.get("reference") or "").strip()
    top = str(item.get("top") or "").strip()
    if not spec:
        report.errors.append("缺 spec")
    if not testbench:
        report.errors.append("缺 testbench —— simulation 段（0.7）无从算起")
        return report
    if not re.search(r"\$finish\b", testbench):
        # 不是硬错误：有的 testbench 靠 $stop 或跑完自然结束。但绝大多数挂住的
        # testbench 都缺它，所以单独提出来。
        report.warnings.append("testbench 里没有 $finish，确认它一定会结束")
    if not re.search(r"[Mm]ismatches\s*:", testbench):
        report.errors.append(
            "testbench 不打印 'Mismatches: N in M samples' —— 奖励会退化成二值，"
            "早期 rollout 全 0，训不动"
        )
    if top and not re.search(rf"\b{re.escape(top)}\b", spec):
        report.warnings.append(f"spec 里没出现模块名 {top!r}，模型可能写出对不上的接口")
    if not reference:
        report.warnings.append("没有 reference（参考实现），跳过判分能力检查（第 3/4 条）")
        return report

    reward = _reward(top)

    # 2 + 3. 参考实现编得过、且拿满分
    breakdown = reward.score(reference, testbench)
    report.reference_score = breakdown.total
    syntax = breakdown.stage(SYNTAX)
    if not syntax.ok:
        report.errors.append(f"参考实现编不过：{syntax.detail[:200]}")
        return report
    sim = breakdown.stage(SIMULATION)
    if sim.detail.startswith("仿真超时"):
        report.errors.append("参考实现跑不完（组合环？testbench 缺 $finish？）")
        return report
    if breakdown.total < 1.0:
        report.errors.append(
            f"参考实现只拿到 {breakdown.total:.3f}，不是满分 —— "
            f"testbench 判错了自己的正确答案：{sim.detail[:200]}"
        )
        return report

    # 4. 变异实现必须掉分
    mutants = _mutate(reference)
    if not mutants:
        report.warnings.append("生成不出变异体（源码里没有可改的算子），判分能力未验证")
        return report
    for name, mutated in mutants:
        report.mutant_scores[name] = reward.score(mutated, testbench).total
    caught = [n for n, s in report.mutant_scores.items() if s <= MUTANT_MAX]
    if not caught:
        report.errors.append(
            "★ 把参考实现改坏之后分数没掉（"
            + "、".join(f"{n} {s:.3f}" for n, s in report.mutant_scores.items())
            + "）—— testbench 判不出错，这道题在训练里是纯噪声：全组同分、优势恒为 0"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", type=Path, help="题库 jsonl（每行一道题）")
    parser.add_argument("--quiet", action="store_true", help="只打印有问题的样本")
    parser.add_argument("--write-bad", type=Path, default=None, help="把坏样本另存一份")
    parser.add_argument("--limit", type=int, default=0, help="只查前 N 行（先摸底）")
    args = parser.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(f"找不到 {args.jsonl}")
    # 缺工具是一次性的环境问题，不是某一道题的问题。在这里一次说清楚，
    # 而不是让每道题都报一遍同一句话。
    try:
        _reward("")
    except RewardConfigError as exc:
        raise SystemExit(
            f"{exc}\n本脚本只需要 iverilog：\n"
            f"  macOS  brew install icarus-verilog\n"
            f"  Debian apt install iverilog"
        ) from exc

    rows: list[dict] = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    reports = [check_row(i, row) for i, row in enumerate(rows)]
    bad = [r for r in reports if not r.ok]
    warned = [r for r in reports if r.ok and r.warnings]

    for report in reports:
        if args.quiet and report.ok and not report.warnings:
            continue
        mark = "✗" if not report.ok else ("!" if report.warnings else "✓")
        score = "" if report.reference_score is None else f"  参考={report.reference_score:.3f}"
        print(f"{mark} {report.row_id}{score}")
        for message in report.errors:
            print(f"    错误：{message}")
        for message in report.warnings:
            print(f"    提醒：{message}")

    print(
        f"\n共 {len(reports)} 道：{len(reports) - len(bad)} 可用"
        f"（其中 {len(warned)} 条有提醒），{len(bad)} 道有问题。"
    )
    if bad and args.write_bad:
        args.write_bad.write_text(
            "\n".join(json.dumps(rows[r.index], ensure_ascii=False) for r in bad),
            encoding="utf-8",
        )
        print(f"坏样本已写到 {args.write_bad}")
    if bad:
        print("\n先修 testbench 再训练：这些题在训练里不产生任何信号，只消耗 GPU。")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
