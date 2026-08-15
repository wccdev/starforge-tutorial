#!/usr/bin/env python
# 题库单轮 GRPO 训练脚本（NeMo-RL 0.7.0）。
# 数据：datasets/qa_rl 的 train/val jsonl（每行 {"query", "expected_answer": "[type] ..."}）。
# 奖励：common/environments/qa_env.py 的 QARewardEnv，内部调用 common/rewards 的判分逻辑
#       （简答可走 LLM 裁判，端点连不上自动回退关键词覆盖率）。
# 由 grpo recipe 的显式入口通过 NeMoRLAdapter 调用；样板流程见 common/bootstrap.py。
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torch.utils.data import Dataset

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType

from common import bootstrap
from common.environments.qa_env import QARewardEnv

TASK_NAME = "qa"


class QAJsonlDataset(Dataset):
    """读题库 jsonl，按需把每条转成 DatumSpec（单轮）。"""

    def __init__(self, path: str, tokenizer, input_key: str, output_key: str,
                 system_prompt: str | None = None):
        self.rows = bootstrap.read_jsonl(path)
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        query = str(row[self.input_key])
        expected = str(row[self.output_key])

        chat: list[dict[str, str]] = []
        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.append({"role": "user", "content": query})

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
            "extra_env_info": {"expected_answer": expected, "query": query},
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
        }


def main():
    config, _ = bootstrap.load_experiment_config(
        THIS_DIR, MasterConfig, description="题库单轮 GRPO 训练"
    )
    tokenizer = bootstrap.init_runtime(config, "grpo")

    data_cfg = config.data
    data_dir = bootstrap.resolve_data_dir(
        data_cfg, "QA_RL_DATA_DIR",
        "请先 `lab dataset prepare qa_rl`，再 `export QA_RL_DATA_DIR=<repo>/datasets/qa_rl`。",
    )
    input_key = data_cfg.get("input_key", "query")
    output_key = data_cfg.get("output_key", "expected_answer")
    system_prompt = data_cfg.get("system_prompt") or None

    train_dataset = QAJsonlDataset(
        os.path.join(data_dir, "train.jsonl"), tokenizer, input_key, output_key, system_prompt
    )
    val_dataset = QAJsonlDataset(
        os.path.join(data_dir, "val.jsonl"), tokenizer, input_key, output_key, system_prompt
    )
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")

    env_cfg = config.env[TASK_NAME]["cfg"]
    env = QARewardEnv.options(num_gpus=0).remote(cfg=dict(env_cfg))

    bootstrap.run_grpo(config, tokenizer, train_dataset, val_dataset, {TASK_NAME: env})


if __name__ == "__main__":
    main()
