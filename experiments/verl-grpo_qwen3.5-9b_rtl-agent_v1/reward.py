"""verl 自定义奖励（官方 custom_reward_function 机制）：RTL 三段式判分。

签名与 verl 内置 reward_score 一致（docs/preparation/reward_function）：
  compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

config 的 reward.custom_reward_function.path（以及顶层旧键 custom_reward_function.path）
指向本文件即可，官方入口动态加载，无需自定义 main。verl 0.9 V1 只读前者。

判分逻辑在 common/rewards/rtl_reward.py（common/ 随作业包上传）。这里只做三件事：
把 testbench 从 ground_truth 里取出来、把顶层模块名从 extra_info 里取出来、
抠不到代码时给 FORMAT_PENALTY。

★ 不要在这里重写一份简化判分。三段式的部分分（1 - 错/总）是这套奖励能不能训得动
  的关键：只判「过没过」的话早期 rollout 基本全是 0，组内奖励恒等 → GRPO 优势
  全为 0 → 训练照跑但什么都学不到（与 qa-tools 实验 README 坑 4 是同一种失败，
  只是病因不同）。

★ 这个数与平台评测的 pass@1 **不是同一个口径**，也不该是。评测走 VerilogEval /
  RTLLM 自带的 harness（sf bench run --suites verilogeval-v2），要的是可比；
  这里要的是坡度，所以给部分分、还额外扣综合期问题。报告效果时以评测侧为准。
"""
from __future__ import annotations

from common.rewards import rtl_reward

FORMAT_PENALTY = rtl_reward.FORMAT_PENALTY


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """ground_truth 是这道题的隐藏 testbench 源码；extra_info.top 是顶层模块名。"""
    design = rtl_reward.extract_verilog(solution_str or "")
    if not design:
        # 给负分而不是 0：0 和「写了但编不过」同分，模型就学不到「至少要按格式
        # 输出一整个 module」。与 common/rewards/qa_reward 的 FORMAT_PENALTY 同一个用意。
        return FORMAT_PENALTY
    top = (extra_info or {}).get("top", "") if isinstance(extra_info, dict) else ""
    reward = rtl_reward.RtlReward(top=str(top or ""))
    return reward.score(design, str(ground_truth or "")).total
