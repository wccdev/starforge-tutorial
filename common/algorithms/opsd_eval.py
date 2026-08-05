"""按题分组的验证指标：avg@N / pass@N / majority@N。

## 为什么需要

AIME 只有 30 题。每题只采 1 条时，准确率的分辨率就是 1/30 = 3.3%，相邻两次验证光靠
采样噪声就能差 6~10 个百分点——曲线剧烈抖动，分不清是方法起效还是在掷骰子。
论文的 `eval/run_eval.sh` 用 `--val_n 12`，NeMo-RL 官方 `distillation_math.yaml` 用
`AIME2024 + repeat: 16`，都是同一个意思：**每题采 N 条再聚合**。

NeMo-RL 的 `validate()` 只会把所有样本的 reward 求个平均（= avg@N），拿不到按题分组的
pass@N 与 majority@N。本模块补上这两个，口径对齐论文 `eval/evaluate_math.py`：

    avg@N       总正确数 / 总生成数            —— 与 NeMo-RL 的 validation/accuracy 同义
    pass@N      至少有一条正确的题 / 总题数     —— 上界，反映"模型能不能做出来"
    majority@N  多数投票答案正确的题 / 总题数   —— 实际部署时最接近的口径

三者一起看才有信息量：pass@N 高而 avg@N 低 = 会做但不稳，正是蒸馏该改善的地方；
两者一起涨才是真的学会了。

## majority@N 的正确性判定

多数投票要把答案字符串**互相比较**来找众数，这一步只能靠字符串归一化，而归一化永远不可能
和数学判分器完全一致（`1/2` vs `0.5` vs `\frac{1}{2}`）。

所以这里只用字符串归一化来**分组**，不用它判对错：选出票数最多的那组之后，
该组的正确性直接取**该组样本的真实 reward**（数学判分器给的）。这样归一化再粗糙，
最多是把本该合并的两组拆开（低估 majority），而不会把错答案判成对的。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# 数学答案的轻量归一化：只用于**分组**，不参与判分（判分用真实 reward，见模块文档）。
_STRIP_WRAPPERS = re.compile(r"\\(?:left|right|!|,|;|\s)")
_TRAILING = " \t\n.$"


def normalize_answer(text: str) -> str:
    r"""把答案字符串归一化到可比较的形式（仅用于多数投票分组）。

    刻意保守：只做「显然等价」的处理（去空白/`$`/`\left`/`\right`/末尾句点、
    去掉 `\dfrac`→`\frac` 这类同义命令）。不做数值求值——`\frac{1}{2}` 与 `0.5`
    仍会被判为不同组，宁可低估 majority 也不要把两个不同答案错误合并。
    """
    s = (text or "").strip().strip(_TRAILING)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = _STRIP_WRAPPERS.sub("", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


@dataclass(frozen=True)
class ValSample:
    """一条验证样本：属于哪道题、判分器给的 reward、模型写出的答案。"""

    problem_id: str
    reward: float
    answer: Optional[str] = None  # 从 \boxed{} 抽出；抽不到为 None（格式不合规）


def grouped_metrics(
    samples: list[ValSample], *, correct_threshold: float = 0.5
) -> dict[str, float]:
    """按题聚合出 avg@N / pass@N / majority@N 及若干诊断量。

    Args:
        samples:           所有验证样本（同一题的 N 条 repeat 混在一起，用 problem_id 分组）。
        correct_threshold: reward 超过它算答对。二元奖励下 0.5 即可；
                           若环境带 reward shaping，需按实际口径调整。

    Returns:
        指标 dict。样本为空时返回空 dict（调用方据此跳过上报，避免污染曲线）。
    """
    if not samples:
        return {}

    by_problem: dict[str, list[ValSample]] = {}
    for s in samples:
        by_problem.setdefault(s.problem_id, []).append(s)

    n_problems = len(by_problem)
    n_correct = sum(1 for s in samples if s.reward > correct_threshold)

    passed = 0
    majority_correct = 0
    unparsed = 0
    for group in by_problem.values():
        if any(s.reward > correct_threshold for s in group):
            passed += 1
        if _majority_is_correct(group, correct_threshold):
            majority_correct += 1
        unparsed += sum(1 for s in group if not s.answer)

    # 每题实际采样数（repeat 生效了没有）——配错时这里会是 1，一眼看得出来
    samples_per_problem = len(samples) / n_problems

    return {
        "avg_at_n": n_correct / len(samples),
        "pass_at_n": passed / n_problems,
        "majority_at_n": majority_correct / n_problems,
        "num_problems": float(n_problems),
        "samples_per_problem": samples_per_problem,
        # 没写出 \boxed{} 的比例。偏高说明模型没遵守格式，此时准确率低是格式问题不是能力问题
        "unparsed_answer_rate": unparsed / len(samples),
    }


def _majority_is_correct(group: list[ValSample], threshold: float) -> bool:
    """该题的多数投票答案是否正确。

    分组用归一化字符串，**判分用该组样本的真实 reward**（见模块文档）。
    平票时取「先出现的」——与 `Counter.most_common` 的稳定顺序一致，避免结果随机抖动。
    """
    votes = Counter(normalize_answer(s.answer) for s in group if s.answer)
    if not votes:
        return False  # 整题没有一条写出合规答案
    winner, _ = votes.most_common(1)[0]
    return any(
        normalize_answer(s.answer) == winner and s.reward > threshold
        for s in group
        if s.answer
    )


def extract_final_answer(message_log: list[dict]) -> Optional[str]:
    """从一条对话里取模型最终写出的 \\boxed{} 答案；没有则 None。

    只看最后一条 assistant 消息：推理过程中可能出现中间结论的 boxed，
    评测口径应当以最终作答为准（与 `eval/evaluate_math.py` 一致）。
    """
    from common.rewards.qa_reward import extract_boxed

    for msg in reversed(message_log or []):
        if msg.get("role") == "assistant":
            return extract_boxed(str(msg.get("content") or ""))
    return None


# 数据集在 extra_env_info 里放的题目标识（同一题的 N 条 repeat 共用同一个值）。
PROBLEM_ID_KEY = "problem_id"


def install_grouped_val_metrics(*, correct_threshold: float = 0.5) -> None:
    """让 NeMo-RL 的验证除了 accuracy 之外，再吐出 pass@N / majority@N。

    做法：`validate()` 期间旁听每一批 rollout，记下 (题号, reward, 抽出的答案)，
    验证结束后按题聚合，把指标并进 `validate()` 的返回值——于是它们会随
    `logger.log_metrics(val_metrics, prefix="validation")` 一起上报，
    在 console / SwanLab 上和 accuracy 并列出现，无需改 NeMo-RL 一行代码。

    必须在 `distillation_train()` 之前调用；依赖 `install_opsd()` 已装好的 rollout 补丁
    （两者共用同一个旁听机制，调用先后无所谓）。
    """
    from nemo_rl.algorithms import distillation

    from common.algorithms import opsd

    if getattr(distillation, "_opsd_val_metrics_installed", False):
        return

    state: dict = {"collecting": False, "samples": []}

    def _on_rollout(batch) -> None:
        # 训练期的 rollout 同样会走到这里，用开关区分：只在验证窗口内收集。
        if not state["collecting"]:
            return
        rewards = batch["total_reward"]
        infos = batch["extra_env_info"]
        logs = batch["message_log"]
        for i in range(len(infos)):
            info = infos[i] or {}
            pid = str(info.get(PROBLEM_ID_KEY, i))
            state["samples"].append(
                ValSample(
                    problem_id=pid,
                    reward=float(rewards[i]),
                    answer=extract_final_answer(logs[i]),
                )
            )

    opsd.add_rollout_listener(_on_rollout)

    _orig_validate = distillation.validate

    def _patched_validate(*args, **kwargs):
        state["collecting"] = True
        state["samples"] = []
        try:
            val_metrics, timings = _orig_validate(*args, **kwargs)
        finally:
            state["collecting"] = False

        try:
            extra = grouped_metrics(state["samples"], correct_threshold=correct_threshold)
        except Exception as e:  # noqa: BLE001  指标是旁路，算不出来也不能打断训练
            print(f"[OPSD] 分组验证指标计算失败（已跳过）: {e}", flush=True)
            extra = {}

        if extra:
            val_metrics.update(extra)
            print(
                f"[OPSD] 验证 {int(extra['num_problems'])} 题 × "
                f"{extra['samples_per_problem']:.1f} 条："
                f"avg@N={extra['avg_at_n']:.3f} pass@N={extra['pass_at_n']:.3f} "
                f"majority@N={extra['majority_at_n']:.3f}",
                flush=True,
            )
            if extra["samples_per_problem"] < 2:
                print(
                    "[OPSD] ⚠️ 每题只采到 1 条：pass@N / majority@N 退化成 accuracy，"
                    "且 30 题的 avg@1 噪声极大（分辨率 3.3%）。请设 data.val_repeat。",
                    flush=True,
                )
        return val_metrics, timings

    distillation.validate = _patched_validate
    distillation._opsd_val_metrics_installed = True
    print("[OPSD] 已安装分组验证指标（avg@N / pass@N / majority@N）", flush=True)
