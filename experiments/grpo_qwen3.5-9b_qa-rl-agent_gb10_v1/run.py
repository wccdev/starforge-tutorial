#!/usr/bin/env python
# 题库「多轮 + 本地文档检索」Agent GRPO 训练脚本（NeMo-RL 0.6.0）。
#
# 这是与单轮 baseline（grpo_qwen3.5-9b_qa-rl_v1）做 A/B 对比的【对照组 / treatment】：
#   同一份题库数据 / 模型 / LoRA / batch / seq / 裁判奖励，唯一差异是——
#   模型回答前可以**多轮调用 <search> 在集群容器内 grep 本地资料**（/data/docs 下的 markdown）再作答
#   （见 common/environments/qa_docs_agent_env.py）。
#
# 数据：datasets/qa_rl 的 train/val jsonl（每行 {"query", "expected_answer": "[type] ..."}），与 baseline 完全一致；
#       本脚本只在 query 前**加一段"可检索本地资料"的说明**，答案格式仍沿用题目里的 \boxed{}。
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

from common.environments.qa_docs_agent_env import QADocsAgentEnv, make_eval_cfg

TASK_NAME = "qa_docs"
STOP_STRINGS = ["</search>"]  # 生成到 </search> 即停，让环境返回检索结果；直接作答则生成到 EOS

# 在原题面前加这段说明，告诉模型「可多轮检索本地资料」；答案格式仍由题目自带的 \boxed{} 指令决定。
# ⚠️ Base 模型（非 Instruct）零样本难以稳定遵循「<search>…</search> + \boxed{}」多轮协议，
#    故内置一个 1-shot 示例把完整轮次（检索→收到[检索结果]→作答）演示一遍，显著提升格式遵循率。
#    示例只演示协议形态，不泄露真实题目答案；baseline 也用同款 Base，加示例不改变「检索」这个唯一变量。
DOCS_PREAMBLE = (
    "你可以在回答前多轮检索公司技术资料库来获取依据：\n"
    "若题目涉及不确定的设备、工艺、规范或数值，先输出 <search>关键词</search>（系统会在资料库查并把片段以「[检索结果]」回灌给你）。\n"
    "检索词只保留核心术语；通常一次定向检索后就应作答，仅当首次没有相关资料时才换关键词再查一次。\n"
    "拿到资料后，按题目要求作答，并把答案放入 \\boxed{...}（单选/多选题填选项字母，如 \\boxed{B} 或 \\boxed{A,C}）。资料不足或无需检索也可直接作答。\n\n"
    "下面是一个完整示例（仅演示交互格式）：\n"
    "问题（单选）：CMP 制程中用于去除晶圆表面大部分铜层的步骤是？ A. 精抛  B. 主抛  C. 后清洗  D. 退火\n"
    "<search>CMP 铜 去除 步骤</search>\n"
    "[检索结果]\n【cmp_process.md】\nL12: 铜 CMP 分为主抛(bulk removal)与精抛两步，主抛负责去除大部分铜层……\n"
    "根据资料，去除大部分铜层的是主抛，对应 B。\\boxed{B}\n\n"
    "现在请按同样方式回答下面的问题：\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="题库多轮+本地文档检索 GRPO 训练")
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


class QADocsJsonlDataset(Dataset):
    """读题库 jsonl，转成多轮本地文档检索 Agent 的 DatumSpec（query 前加检索说明）。"""

    def __init__(self, path: str, tokenizer, input_key: str, output_key: str,
                 max_turns: int, system_prompt: str | None = None):
        self.rows = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.max_turns = int(max_turns)
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
        chat.append({"role": "user", "content": DOCS_PREAMBLE + query})

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
    # MasterConfig 是 pydantic BaseModel：顶层属性访问，嵌套仍是 dict。
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
            "未指定数据目录。请先把题库放到集群并 `export QA_RL_DATA_DIR=<cluster>/datasets/qa_rl`。"
        )
    # 提示：本地资料检索目录由环境变量 DOCS_DIR 控制（默认 /data/docs），须从集群容器内可达。
    input_key = data_cfg.get("input_key", "query")
    output_key = data_cfg.get("output_key", "expected_answer")
    system_prompt = data_cfg.get("system_prompt") or None

    env_cfg = dict(config.env[TASK_NAME]["cfg"])
    max_turns = int(env_cfg.get("max_turns", config.grpo["max_rollout_turns"]))

    train_dataset = QADocsJsonlDataset(
        os.path.join(data_dir, "train.jsonl"), tokenizer, input_key, output_key,
        max_turns, system_prompt,
    )
    val_dataset = QADocsJsonlDataset(
        os.path.join(data_dir, "val.jsonl"), tokenizer, input_key, output_key,
        max_turns, system_prompt,
    )
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条（每条可多轮检索，max_turns={max_turns}）")

    # 训练环境与验证环境用【两个实例】：检索后端、判分方式完全相同，只有 reward shaping 不同。
    #   训练：带 shaping（检索即时奖励 / 检索后答对加成 / 不作答惩罚）——引导模型真的去查资料。
    #   验证：make_eval_cfg() 把 shaping 归零，只留最终判分。
    # 必须分开：NeMo-RL 的 validation/accuracy 就是 mean(total_reward)，而 total_reward 是逐轮奖励累加，
    # 若验证也带 shaping，则「用了工具」本身白送分（本配置最多 +0.2 绝对值），验证分虚高，
    # 且与单轮无工具 baseline（qa-rl_v1，纯最终判分）不再同尺度，A/B 结论会被工具加分污染。
    train_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=env_cfg)
    val_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=make_eval_cfg(env_cfg))
    task_to_env = {TASK_NAME: train_env}
    val_task_to_env = {TASK_NAME: val_env}
    print("训练环境带检索 reward shaping；验证环境已归零 → validation/accuracy = 纯答题得分")

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
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
