"""单轮题库 QA 奖励环境（NeMo-RL 0.7.0）。

把 common/rewards 里的判分逻辑包成一个 GRPO 环境：每个 prompt 是一道题，模型答一次
（单轮，max_rollout_turns=1），环境据 `expected_answer` 算 reward 并立即结束。

与多轮 Agent 环境（example_tool_env.py）的区别：这里只判分、不调用工具、不续轮。

判分来源由 cfg.use_judge 决定：
  - use_judge=true ：common.rewards.qa_judge_reward.qa_judge_reward_fn
                     （简答走裁判 LLM，端点连不上自动回退关键词覆盖率）
  - use_judge=false：common.rewards.qa_reward.qa_rule_reward_fn（纯规则，零成本）

环境从每条样本的 extra_env_info 读取：
  - expected_answer：带 [type] 前缀的金标准（见 common/rewards/README.md）
  - query         ：题面（裁判 LLM 构造评分 prompt 时用）
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional, TypedDict

import ray
import torch
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

# 确保 Ray actor 进程里能 import 到本仓库的 common 包
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class QAMetadata(TypedDict, total=False):
    expected_answer: str  # 带 [type] 前缀的金标准
    query: str            # 题面（裁判 LLM 用）


def _last_assistant_text(message_log: LLMMessageLogType) -> str:
    """取最后一条 assistant 消息的文本内容（即本轮模型作答）。"""
    for msg in reversed(message_log):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


@ray.remote  # pragma: no cover
class QARewardEnv(EnvironmentInterface[QAMetadata]):
    """单轮题库判分环境（Ray Actor）。"""

    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.use_judge = bool(self.cfg.get("use_judge", True))
        # 延迟导入：reward 模块依赖同义词表/可选裁判端点，放到 actor 内导入更稳。
        if self.use_judge:
            from common.rewards.qa_judge_reward import qa_judge_reward_fn

            self._reward_fn = qa_judge_reward_fn
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            self._reward_fn = qa_rule_reward_fn

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[QAMetadata],
    ) -> EnvironmentReturn[QAMetadata]:
        completions = [_last_assistant_text(log) for log in message_log_batch]
        queries = [str(m.get("query", "")) for m in metadata]
        expected = [str(m.get("expected_answer", "")) for m in metadata]

        rewards = self._reward_fn(queries, completions, expected)

        n = len(completions)
        observations = [
            {"role": "environment", "content": f"得分: {float(r):.3f}"}
            for r in rewards
        ]
        return EnvironmentReturn(
            observations=observations,
            metadata=[None] * n,            # 单轮：无后续状态
            next_stop_strings=[None] * n,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor([True] * n, dtype=torch.bool),  # 单轮：一律结束
            answers=expected,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        ).float()
        if len(rewards) == 0:
            return batch, {}
        metrics = {
            "qa_mean_reward": rewards.mean().item(),
            "qa_perfect_rate": (rewards >= 1.0).float().mean().item(),
            "qa_format_penalty_rate": (rewards < 0).float().mean().item(),
        }
        return batch, metrics
