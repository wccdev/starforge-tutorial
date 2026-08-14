#!/usr/bin/env python
# 题库单轮 GRPO 训练脚本（NeMo-RL 0.6.0）。
# 数据：datasets/qa_rl 的 train/val jsonl（每行 {"query", "expected_answer": "[type] ..."}）。
# 奖励：common/environments/qa_env.py 的 QARewardEnv，内部调用 common/rewards 的判分逻辑
#       （简答可走 LLM 裁判，端点连不上自动回退关键词覆盖率）。
# 由 grpo recipe 的显式入口通过 NeMoRLAdapter 调用。
import argparse
import json
import os
import pprint
import sys
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.environments.qa_env import QARewardEnv

TASK_NAME = "qa"


def parse_args():
    parser = argparse.ArgumentParser(description="题库单轮 GRPO 训练")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class QAJsonlDataset(Dataset):
    """读题库 jsonl，按需把每条转成 DatumSpec（单轮）。"""

    def __init__(self, path: str, tokenizer, input_key: str, output_key: str,
                 system_prompt: str | None = None):
        self.rows = _read_jsonl(path)
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
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    print(f"已加载配置: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    # 新版 NeMo-RL 的 MasterConfig 是 pydantic BaseModel：顶层字段用属性访问（config.policy），
    # 嵌套仍是普通 dict（config.policy["generation"]）。setup() 要求传入 MasterConfig 实例。
    config = OmegaConf.to_container(config, resolve=True)
    config: MasterConfig = MasterConfig(**config)
    print("最终配置：")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 日志目录: {config.logger['log_dir']}")

    init_ray()
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    # 数据路径：优先 QA_RL_DATA_DIR，其次 config.data.data_dir
    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit(
            "未指定数据目录。请先 `lab prepare qa_rl`，再 `export QA_RL_DATA_DIR=<repo>/datasets/qa_rl`。"
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
    task_to_env = {TASK_NAME: env}

    # NeMo-RL main：setup() 返回 11 个值（新增第 3 位 nemo_gym actor，cluster 变为
    # (train_cluster, inference_cluster) 元组）。未使用的用 _ 前缀占位。
    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
