"""verl 自定义奖励（官方 custom_reward_function 机制）：题库最终判分。

签名与 verl 内置 reward_score 一致（docs/preparation/reward_function）：
  compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

config 的 custom_reward_function.path 指向本文件即可，官方入口动态加载，无需自定义 main。
判分口径与 nemo-rl 对照实验（grpo_qwen3.5-9b_qa-rl-agent_v3）保持一致：
只看 \boxed{} 内的最终答案，工具调用本身不给分——保证 A/B 可比。
"""
from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
FORMAT_PENALTY = -0.5  # 写不出 \boxed{} 的重罚，与 nemo-rl 版 qa_reward 同值


def _normalize(text: str) -> str:
    return re.sub(r"[\s,，、]+", "", text.strip().lower())


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """答案在 \\boxed{} 内则按精确匹配判分；多选题按选项集合比对。"""
    matches = _BOXED.findall(solution_str or "")
    if not matches:
        return FORMAT_PENALTY
    answer = _normalize(matches[-1])
    truth = _normalize(str(ground_truth))
    if not truth:
        return 0.0
    # 多选题（如 "A,C"）：集合相等满分，部分命中给部分分、有错选扣成 0。
    if re.fullmatch(r"[a-h](?:[a-h])+", truth):
        got, want = set(answer), set(truth)
        if not got:
            return 0.0
        if got == want:
            return 1.0
        if got <= want:
            return len(got) / len(want) * 0.5
        return 0.0
    return 1.0 if answer == truth else 0.0
