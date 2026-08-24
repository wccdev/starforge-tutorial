"""verl 自定义奖励（官方 custom_reward_function 机制）：题库最终判分。

签名与 verl 内置 reward_score 一致（docs/preparation/reward_function）：
  compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

config 的 custom_reward_function.path 指向本文件即可，官方入口动态加载，无需自定义 main。

判分口径与 nemo-rl 对照实验（grpo_qwen3.5-9b_qa-rl-agent_v3）严格一致：直接复用
common/rewards 的规则判分（common/ 随作业包上传，见 sf 打包白名单）。
★ 不要在这里重写一份简化判分：题库的 ground_truth 带 [type] 前缀
  （"[single] A" / "[multiple] A,B" / "[fill] a ||| b" / "[short] kw1 ||| kw2"），
  自己写的精确匹配会把前缀当答案比对 → 全样本判 0，GRPO 组内奖励恒等、优势全为 0，
  训练照跑但学不到任何东西。前缀分派与填空/简答口径都在 qa_reward 里。

SHORT_SCOPE 必须是 boxed：检索工具会把资料原文回灌进上下文，若简答题按整段回答统计
关键词覆盖率，模型只要复述检索片段就能刷满分——这是单轮无工具 baseline 不存在的通道，
会同时抬高验证分、让 A/B 失真。nemo-rl 对照实验的 short_answer_scope 同为 boxed。
"""
from __future__ import annotations

from common.rewards import qa_reward

qa_reward.SHORT_SCOPE = "boxed"

FORMAT_PENALTY = qa_reward.FORMAT_PENALTY


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """按 ground_truth 的 [type] 前缀分派规则判分；无 \\boxed{} 则 FORMAT_PENALTY。"""
    scores = qa_reward.qa_rule_reward_fn([""], [solution_str or ""], [str(ground_truth)])
    return scores[0]
