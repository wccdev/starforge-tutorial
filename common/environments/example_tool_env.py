"""可运行的多轮「多工具」Agent Environment（NeMo-RL 0.6.0）。

照官方 `nemo_rl/environments/games/sliding_puzzle.py` 的结构写成，可直接用于 GRPO 多轮
Agent 训练。内置三个工具，贴近真实 Agent 场景：

  - calc   ：算术计算（安全求值）
  - search ：在该题的知识库（KB）里检索事实
  - python ：执行一段 Python 代码并返回 stdout（子进程 + 超时）

协议（写在数据集 prompt 里，见实验 run.py）：
  - 调用工具： <tool>calc: 2+3*4</tool> / <tool>search: 苹果 单价</tool> / <tool>python: print(sum(i*i for i in range(1,6)))</tool>
  - 给出答案： <answer>14</answer>
env 解析模型上一轮输出 → 分发工具 / 校验答案 / 纠正格式 / 超轮判负。

⚠️ 安全提示：`python` 工具会执行**模型生成的代码**。请只在隔离的 NeMo-RL 训练容器内运行，
   生产环境应换成更强的沙箱（容器 / seccomp / 资源限制）。
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

import ray
import torch
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

# ============================ 工具实现（envkit 内核） ============================
# 工具与算术求值收口到 common/envkit（与 Gym server 腿共用同一实现）；
# python 工具经 starforge.sandbox.SandboxProvider 分派（本地子进程 / 外部沙箱）。
# 保留原模块级名字：实验 run.py 与测试按旧名引用。
from common.envkit.tools import TOOLS as TOOLS  # noqa: E402
from common.envkit.tools import safe_eval as safe_eval  # noqa: E402
from common.envkit.tools import tool_calc as _tool_calc  # noqa: E402, F401
from common.envkit.tools import tool_python as _tool_python  # noqa: E402, F401
from common.envkit.tools import tool_search as _tool_search  # noqa: E402, F401


# ============================ 环境状态 / 元数据 ============================
class ToolAgentMetadata(TypedDict, total=False):
    target: float          # 正确答案
    num_turns: int         # 已交互轮数
    max_turns: int         # 最大轮数
    question: str          # 题面
    answer_tolerance: float
    kb: dict[str, str]     # 本题知识库（供 search 使用）
    code_timeout: float    # python 工具超时


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """标签解析收口到 envkit（闭标签缺失仍取全文的语义，全环境唯一实现）。"""
    from common.envkit.tags import extract_tag

    return extract_tag(text, tag)


class ToolAgentRunner:
    """单条轨迹一轮的处理逻辑（与官方 SlidingPuzzleRunner 对应）。"""

    NEXT_STOP_STRINGS = ["</tool>", "</answer>"]

    def process_turn(
        self,
        message_log: LLMMessageLogType,
        metadata: ToolAgentMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[ToolAgentMetadata],
        Optional[list[str]],
    ]:
        num_turns = int(metadata["num_turns"])
        max_turns = int(metadata["max_turns"])
        tol = float(metadata.get("answer_tolerance", 1e-6))

        # 超过最大轮数：判负并结束
        if num_turns >= max_turns:
            return (
                {"role": "environment", "content": f"已达最大轮数 {max_turns}，结束。"},
                0.0,
                True,
                None,
                None,
                None,
            )

        # 取最后一条 assistant 内容
        content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            content = str(message_log[-1]["content"]).strip()

        next_meta: ToolAgentMetadata = dict(metadata)  # type: ignore[assignment]
        next_meta["num_turns"] = num_turns + 1

        # 1) 最终答案
        answer_str = _extract_tag(content, "answer")
        if answer_str is not None:
            try:
                pred = safe_eval(answer_str)
                correct = abs(pred - float(metadata["target"])) <= tol
            except Exception:  # noqa: BLE001
                correct = False
            obs = "回答正确！" if correct else f"回答错误。正确答案是 {float(metadata['target']):g}。"
            return (
                {"role": "environment", "content": obs},
                1.0 if correct else 0.0,
                True,
                None,
                None,
                [answer_str],
            )

        # 2) 工具调用
        tool_call = _extract_tag(content, "tool")
        if tool_call is not None:
            name, _, arg = tool_call.partition(":")
            name = name.strip().lower()
            arg = arg.strip()
            if name in TOOLS:
                result = TOOLS[name](arg, metadata)
                obs = f"[{name}] {result}"
            else:
                obs = f"未知工具 '{name}'，可用工具：{', '.join(TOOLS)}"
            return (
                {"role": "environment", "content": obs},
                0.0,
                False,
                self.NEXT_STOP_STRINGS,
                next_meta,
                None,
            )

        # 3) 格式非法：提示并让其重试（不结束，计一轮）
        obs = (
            "格式不对。调用工具用 <tool>工具名: 参数</tool>（工具：calc/search/python），"
            "给答案用 <answer>数值</answer>。"
        )
        return (
            {"role": "environment", "content": obs},
            0.0,
            False,
            self.NEXT_STOP_STRINGS,
            next_meta,
            None,
        )


@ray.remote  # pragma: no cover
class ToolAgentEnv(EnvironmentInterface[ToolAgentMetadata]):
    """多轮多工具 Agent 环境（Ray Actor）。"""

    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.runner = ToolAgentRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ToolAgentMetadata],
    ) -> EnvironmentReturn[ToolAgentMetadata]:
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata, strict=True)
        ]

        observations, rewards, terminateds = [], [], []
        all_stop_strings, all_next_metadata, all_answers = [], [], []
        for obs, rew, term, stops, meta, answ in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            all_stop_strings.append(stops)
            all_next_metadata.append(meta)
            all_answers.append(answ)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=all_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        success_rate = (
            (final_rewards == 1.0).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )
        return batch, {"tool_agent_success_rate": success_rate}
