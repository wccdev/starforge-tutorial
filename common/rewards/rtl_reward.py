"""RTL 三段式奖励：编译 → 仿真 → 综合。

为什么是三段，而不是一个通过率
──────────────────────────────────────────────────────────────────────────────
只判「testbench 过没过」的奖励对 GRPO 几乎没用：早期 rollout 基本全是 0，组内奖励
恒等 → 优势全为 0 → 训练照跑但学不到任何东西（和 README 坑 4 说的是同一个病，
只是病因不同）。三段给的是一条能往上爬的坡：

    syntax      0.1   编译得过 —— 入场券，先让它是合法的 Verilog
    simulation  0.7   testbench 过多少 —— 主信号，且给部分分
    synthesis   0.2   综合得出来且没推断出锁存器 / 组合环

**仿真给部分分是这套奖励的核心。** VerilogEval / RTLLM 的 testbench 打印
「Mismatches: 3 in 100 samples」，那是 0.97 而不是 0。二值化会把「差一点」和
「完全不对」抹成同一个数，而它们之间的距离正是模型要学的东西。

综合段单独存在的理由：yosys 把「推断出锁存器」报成 warning 后照样退出 0，而推断
出的锁存器在 testbench 里看不出来、在芯片上是个 bug。只看退出码等于放它过去。

后一段以前一段通过为前提 —— 编不过的没法仿真，仿不对的综合出来也没意义。失败的
段记 0 而不是「未尝试」：GRPO 要的是一个数。

安全：它会编译并**执行**模型生成的代码
──────────────────────────────────────────────────────────────────────────────
Verilog 有 `$system`，跑一个模型写的 testbench 等价于跑一段模型写的 shell。
所以：工作目录是每次新建的临时目录、子进程 env 只留 PATH（作业容器里有平台注入的
端点与凭据，没有理由出现在它面前）、每段都有超时（一个组合环会让 vvp 永远不返回）。
它跑在作业容器里，容器本身就是隔离边界 —— **不要在本地开发机上直接喂不可信的样本**。

与评测侧的关系
──────────────────────────────────────────────────────────────────────────────
平台评测走的是 VerilogEval / RTLLM 自带的 harness（`sf bench run --suites
verilogeval-v2`），判据是它们自己的 testbench 与 pass@k。这里是**训练期的塑形奖励**，
口径刻意更细（部分分 + 综合段）—— 两者不是同一个数，也不该是：训练要坡度，
评测要可比。报告最终效果时以评测侧为准。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

SYNTAX, SIMULATION, SYNTHESIS = "syntax", "simulation", "synthesis"
STAGES = (SYNTAX, SIMULATION, SYNTHESIS)

DEFAULT_WEIGHTS: dict[str, float] = {SYNTAX: 0.1, SIMULATION: 0.7, SYNTHESIS: 0.2}
STAGE_TOOLS = {SYNTAX: "iverilog", SIMULATION: "iverilog", SYNTHESIS: "yosys"}

#: 没有从回复里抠出任何 Verilog。给负分而不是 0：0 和「写了但编不过」同分，
#: 模型就学不到「至少要按格式输出一段代码」。与 qa_reward 的 FORMAT_PENALTY 同一个用意。
FORMAT_PENALTY = -0.5

#: 工具输出留多少给诊断。整份日志乘以每步 64 条轨迹会淹掉训练日志本身。
MAX_DETAIL = 800

DESIGN_NAME = "design.sv"
TB_NAME = "testbench.sv"

#: 从回复里抠 Verilog。优先带语言标记的围栏块 —— 模型经常在代码后面继续解释，
#: 把解释一起喂给 iverilog 必然编不过，而那不是它设计能力的问题。
_FENCED = re.compile(
    r"```(?:systemverilog|verilog|sv|v)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_BARE_MODULE = re.compile(r"(\bmodule\b.*?\bendmodule\b)", re.DOTALL)

#: testbench 输出里的 mismatch 计数：唯一能给部分分的信号。
_MISMATCH = re.compile(r"[Mm]ismatches\s*:\s*(\d+)\s+in\s+(\d+)\s+samples")
_PASS = re.compile(r"(?i)\b(all tests passed|test passed|PASSED|SUCCESS)\b")
#: testbench 常常在失败时仍然退出 0，光看退出码会把全部失败判成通过。
_FAIL = re.compile(r"(?i)(\$fatal|\bFAILED\b|\bERROR\b|\bMismatch\b)")

#: yosys 会把这些报成 warning 后继续跑完并退出 0 —— 而它们正是「仿真过了、硬件
#: 却不对」的经典原因。
_SYNTH_PROBLEMS = (
    (re.compile(r"(?i)inferring latch|found latch"), "推断出锁存器"),
    (re.compile(r"(?i)combinational loop|found logic loop"), "组合环"),
    (re.compile(r"(?i)multiple driver|conflicting drivers"), "多驱动"),
    (re.compile(r"(?i)\berror\b"), "yosys 报错"),
)


class RewardConfigError(ValueError):
    """奖励配置不成立（缺工具、权重非法、没有 testbench）。

    与「跑出来分数低」是两回事：那是结果，这是这套奖励根本算不出来。
    在 verl 里它会让作业**启动即失败**，这是故意的 —— 静默降级的奖励会让整轮训练
    在一个比声明更小的尺度上跑，而那件事从 reward 曲线上看不出来。
    """


@dataclass(frozen=True)
class ToolRun:
    code: int
    output: str = ""
    timed_out: bool = False


ToolRunner = Callable[[Sequence[str], Path, int], ToolRun]


def run_tool(argv: Sequence[str], cwd: Path, timeout: int) -> ToolRun:
    """跑一个外部工具。env 只留 PATH —— 见模块注释的安全一节。"""
    try:
        proc = subprocess.run(
            list(argv), cwd=str(cwd), timeout=timeout,
            capture_output=True, text=True, errors="replace",
            env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")},
        )
    except subprocess.TimeoutExpired:
        return ToolRun(code=-1, output=f"超时（{timeout}s）", timed_out=True)
    except OSError as exc:
        return ToolRun(code=-1, output=f"无法执行 {argv[0]!r}: {exc}")
    return ToolRun(code=proc.returncode, output=f"{proc.stdout}\n{proc.stderr}".strip())


def extract_verilog(text: str) -> str:
    """从模型回复里抠出 Verilog 源码。抠不到返回空串。"""
    blocks = _FENCED.findall(text or "")
    if blocks:
        # 取最后一个围栏块：模型改错时常常先复述旧代码再给新版本。
        for block in reversed(blocks):
            if "module" in block:
                return block.strip()
        return blocks[-1].strip()
    bare = _BARE_MODULE.search(text or "")
    return bare.group(1).strip() if bare else ""


@dataclass(frozen=True)
class StageResult:
    name: str
    weight: float
    score: float
    ok: bool
    detail: str = ""
    skipped: str = ""

    @property
    def contribution(self) -> float:
        return self.weight * self.score


@dataclass(frozen=True)
class Breakdown:
    """总分 + 每段的分。

    永远带着分段返回，不只返回一个数：训练时看总分，调试时要知道它卡在哪一段。
    「奖励一直是 0.1」和「一直是 0.8」是两个完全不同的问题，总分本身分不出来。
    """

    total: float
    stages: tuple[StageResult, ...] = ()

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 6),
            "stages": {
                s.name: {
                    "score": round(s.score, 6),
                    "contribution": round(s.contribution, 6),
                    "ok": s.ok, "skipped": s.skipped, "detail": s.detail,
                }
                for s in self.stages
            },
        }


def _clip(value: float) -> float:
    return 0.0 if value < 0 else 1.0 if value > 1 else float(value)


def read_simulation(run: ToolRun) -> tuple[float, bool, str]:
    """从 testbench 输出读分数。优先级：计数 > 失败标记 > 通过标记 > 退出码。"""
    if run.timed_out:
        # 超时几乎总是组合环或少了 $finish：那是设计错误，不是「差一点」。
        return 0.0, False, "仿真超时（组合环？testbench 缺 $finish？）"
    if match := _MISMATCH.search(run.output):
        wrong, total = int(match.group(1)), int(match.group(2))
        if total > 0:
            score = _clip(1.0 - wrong / total)
            return score, wrong == 0, f"mismatch {wrong}/{total}"
    if _FAIL.search(run.output):
        return 0.0, False, "命中失败标记"
    if _PASS.search(run.output):
        return 1.0, True, "命中通过标记"
    if run.code != 0:
        return 0.0, False, f"vvp 退出码 {run.code}"
    return 1.0, True, "无失败标记且正常退出"


def synthesis_problems(output: str) -> list[str]:
    return [label for pattern, label in _SYNTH_PROBLEMS if pattern.search(output)]


@dataclass
class RtlReward:
    """三段式 RTL 奖励。

    权重为 0 的段不跑；权重 > 0 但工具不在 PATH 上，**构造时**就报错 ——
    见 RewardConfigError 的注释。
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    compile_timeout: int = 60
    simulate_timeout: int = 120
    synth_timeout: int = 120
    top: str = ""
    #: 内部 seam：测试注入假执行器，训练时用 run_tool。
    runner: ToolRunner = run_tool
    which: Callable[[str], str | None] = shutil.which

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(STAGES)
        if unknown:
            raise RewardConfigError(f"未知阶段 {sorted(unknown)}（可选: {', '.join(STAGES)}）")
        if any(w < 0 for w in self.weights.values()):
            raise RewardConfigError(f"权重不能为负：{self.weights}")
        if sum(self.weights.values()) <= 0:
            raise RewardConfigError("权重全为 0 —— 这样的奖励恒等于 0，训练学不到任何东西")
        missing = sorted({
            STAGE_TOOLS[n] for n, w in self.weights.items()
            if w > 0 and not self.which(STAGE_TOOLS[n])
        })
        if missing:
            raise RewardConfigError(
                f"缺少工具 {'、'.join(missing)}：声明了权重却装不上，奖励会静默地少一段而"
                f"曲线上看不出来。在训练镜像里装上它（见实验 README「镜像要装什么」），"
                f"或把对应阶段的权重设为 0。"
            )

    # ── 段 ──────────────────────────────────────────────────────────────

    def _syntax(self, work: Path) -> StageResult:
        run = self.runner(
            ["iverilog", "-g2012", "-o", "design.out", DESIGN_NAME], work, self.compile_timeout
        )
        return StageResult(
            SYNTAX, self.weights.get(SYNTAX, 0.0), 1.0 if run.code == 0 else 0.0,
            run.code == 0, detail=run.output[:MAX_DETAIL],
        )

    def _simulate(self, work: Path) -> StageResult:
        weight = self.weights.get(SIMULATION, 0.0)
        compiled = self.runner(
            ["iverilog", "-g2012", "-o", "sim.out", DESIGN_NAME, TB_NAME],
            work, self.compile_timeout,
        )
        if compiled.code != 0:
            # 设计单独编过了、加上 testbench 编不过 —— 通常是端口名或位宽对不上。
            # 这是最常见的失败，单独说清楚，模型才有得改。
            return StageResult(
                SIMULATION, weight, 0.0, False,
                detail=f"带 testbench 编译失败（端口名/位宽？）：{compiled.output[:MAX_DETAIL]}",
            )
        run = self.runner(["vvp", "sim.out"], work, self.simulate_timeout)
        score, ok, note = read_simulation(run)
        return StageResult(SIMULATION, weight, score, ok,
                           detail=f"{note} | {run.output}"[:MAX_DETAIL])

    def _synthesize(self, work: Path) -> StageResult:
        weight = self.weights.get(SYNTHESIS, 0.0)
        top = f"synth -top {self.top}" if self.top else "synth -auto-top"
        run = self.runner(
            ["yosys", "-p", f"read_verilog -sv {DESIGN_NAME}; {top}"], work, self.synth_timeout
        )
        if run.code != 0:
            return StageResult(SYNTHESIS, weight, 0.0, False, detail=run.output[:MAX_DETAIL])
        problems = synthesis_problems(run.output)
        return StageResult(
            SYNTHESIS, weight, 0.0 if problems else 1.0, not problems,
            detail=("；".join(problems) or "干净")[:MAX_DETAIL],
        )

    # ── 入口 ────────────────────────────────────────────────────────────

    def score(self, design: str, testbench: str) -> Breakdown:
        """给一份设计打分。design 抠不到时由调用方给 FORMAT_PENALTY，不进这里。"""
        if self.weights.get(SIMULATION, 0.0) > 0 and not testbench.strip():
            raise RewardConfigError(
                "simulation 段权重 > 0 但这道题没有 testbench —— 那样这条奖励会退化成"
                "「编译得过就给 0.7」。检查 prepare_data.py 是否把 testbench 写进了 ground_truth。"
            )
        with tempfile.TemporaryDirectory(prefix="rtl-reward-") as tmp:
            work = Path(tmp)
            (work / DESIGN_NAME).write_text(design, encoding="utf-8")
            (work / TB_NAME).write_text(testbench, encoding="utf-8")

            stages: list[StageResult] = []
            gated = ""
            for name, run_stage in (
                (SYNTAX, self._syntax), (SIMULATION, self._simulate), (SYNTHESIS, self._synthesize),
            ):
                weight = self.weights.get(name, 0.0)
                if weight <= 0:
                    stages.append(StageResult(name, 0.0, 0.0, False, skipped="disabled"))
                elif gated:
                    stages.append(StageResult(name, weight, 0.0, False, skipped=gated))
                else:
                    result = run_stage(work)
                    stages.append(result)
                    if not result.ok:
                        gated = f"{name} 未通过"
        return Breakdown(_clip(sum(s.contribution for s in stages)), tuple(stages))


def rtl_reward_fn(
    completions: Sequence[str], testbenches: Sequence[str],
    tops: Sequence[str] | None = None, **kwargs: Any,
) -> list[float]:
    """批量接口，与 common/rewards 里其它奖励一致：等长列表进，等长 float 出。"""
    tops = list(tops or [""] * len(completions))
    out: list[float] = []
    # strict=True：长度不齐是调用方的 bug，静默按最短截断会让一部分样本
    # 悄悄拿不到奖励，而 GRPO 里那表现为「有些组优势异常」，极难归因。
    for completion, testbench, top in zip(completions, testbenches, tops, strict=True):
        design = extract_verilog(completion)
        if not design:
            out.append(FORMAT_PENALTY)
            continue
        out.append(RtlReward(top=top, **kwargs).score(design, testbench).total)
    return out
