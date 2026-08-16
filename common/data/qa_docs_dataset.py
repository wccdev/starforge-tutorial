"""题库「多轮 + 本地文档检索」Agent 的共享数据集（v3 GRPO 与 MaxRL v2 共用）。

两个实验的数据形态完全一致，唯一实验变量在别处（RL 目标 / 检索说明文案），
所以检索说明（preamble）由调用方注入，这里不写死。
"""
from __future__ import annotations

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from torch.utils.data import Dataset

from common.bootstrap import read_jsonl

TASK_NAME = "qa_docs"
# 生成到 </search> 即停，让环境返回检索结果；直接作答则生成到 EOS。
STOP_STRINGS = ["</search>"]


class QADocsJsonlDataset(Dataset):
    """读题库 jsonl，转成多轮本地文档检索 Agent 的 DatumSpec（query 前加检索说明）。"""

    def __init__(self, path: str, tokenizer, input_key: str, output_key: str,
                 max_turns: int, system_prompt: str | None = None, *, preamble: str):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.max_turns = int(max_turns)
        self.system_prompt = system_prompt
        self.preamble = preamble

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        query = str(row[self.input_key])
        expected = str(row[self.output_key])

        chat: list[dict[str, str]] = []
        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.append({"role": "user", "content": self.preamble + query})

        prompt_text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=False
        ).strip()
        token_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        message_log: LLMMessageLogType = [
            {"role": "user", "content": prompt_text, "token_ids": token_ids}
        ]
        return {
            "message_log": message_log,
            "length": len(token_ids),
            "extra_env_info": {
                "expected_answer": expected,
                "query": query,           # 用原题面（不含检索说明）给裁判判分
                "num_turns": 0,
                "max_turns": self.max_turns,
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
            "stop_strings": STOP_STRINGS,
        }


def build_qa_docs_datasets(config, tokenizer, *, preamble: str, data_dir_hint: str):
    """按 config.data 构建 train/val 数据集与 max_turns（v2/v3 共用的装配逻辑）。"""
    import os

    from common.bootstrap import resolve_data_dir

    data_cfg = config.data
    data_dir = resolve_data_dir(data_cfg, "QA_RL_DATA_DIR", data_dir_hint)
    # 提示：本地资料检索目录由环境变量 DOCS_DIR 控制（默认 /data/docs），须从集群容器内可达。
    input_key = data_cfg.get("input_key", "query")
    output_key = data_cfg.get("output_key", "expected_answer")
    system_prompt = data_cfg.get("system_prompt") or None

    env_cfg = dict(config.env[TASK_NAME]["cfg"])
    max_rollout_turns = int(config.grpo["max_rollout_turns"])
    # 默认必须留出「最后一轮只能作答」的余量：max_turns == max_rollout_turns 时
    # 环境的超轮判负分支**永远不可达**（NeMo-RL rollout 循环跑满即退出、不再调 step），
    # 「搜满 N 轮不作答」净收益为 +N×search_step_reward 的正数，成为零风险刷分策略
    # —— 详见 qa_docs_agent_env 超轮分支处的可达性注释。曾经的默认值恰好复现该坏配置。
    max_turns = int(env_cfg.get("max_turns", max(1, max_rollout_turns - 1)))
    if max_turns >= max_rollout_turns:
        raise ValueError(
            f"env.{TASK_NAME}.cfg.max_turns({max_turns}) 必须 ≤ grpo.max_rollout_turns"
            f"({max_rollout_turns}) - 1，否则超轮判负永远不会触发，"
            "「只检索不作答」成为零风险刷分策略。"
        )

    train_dataset = QADocsJsonlDataset(
        os.path.join(data_dir, "train.jsonl"), tokenizer, input_key, output_key,
        max_turns, system_prompt, preamble=preamble,
    )
    val_dataset = QADocsJsonlDataset(
        os.path.join(data_dir, "val.jsonl"), tokenizer, input_key, output_key,
        max_turns, system_prompt, preamble=preamble,
    )
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条（每条可多轮检索，max_turns={max_turns}）")
    return train_dataset, val_dataset, env_cfg
