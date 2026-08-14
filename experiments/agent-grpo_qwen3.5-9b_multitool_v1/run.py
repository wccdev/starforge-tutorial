#!/usr/bin/env python
# 多轮多工具 Agent（检索 + 计算 + 代码执行）GRPO 训练脚本（NeMo-RL 0.6.0）。
# 改编自官方 examples/run_grpo_sliding_puzzle.py：环境换成自定义多工具环境，
# 数据集随机生成三类需要工具的任务。由 grpo recipe 的显式入口调用。
import argparse
import itertools
import os
import pprint
import random
import sys
from typing import Any, Iterator

from omegaconf import OmegaConf
from torch.utils.data import IterableDataset

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

from common.environments.example_tool_env import ToolAgentEnv, safe_eval

TASK_NAME = "tool_agent"
STOP_STRINGS = ["</tool>", "</answer>"]
ITEMS = ["苹果", "香蕉", "橙子", "牛奶", "面包", "鸡蛋", "咖啡", "茶叶"]

PROMPT_HEADER = (
    "你是一个会使用工具的智能体。请通过调用工具解决问题，再给出最终数值答案。\n"
    "可用工具（每次单独一行调用）：\n"
    "- calc：计算算术表达式，如 <tool>calc: 2+3*4</tool>\n"
    "- search：检索题目提供的资料，如 <tool>search: 苹果 单价</tool>\n"
    "- python：执行 Python 代码并打印结果，如 <tool>python: print(sum(i*i for i in range(1,6)))</tool>\n"
    "得到结果后用如下格式给出最终答案：<answer>数值</answer>\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="多轮多工具 GRPO 训练")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _make_arith(max_number: int, num_operands: int) -> tuple[str, float]:
    ops = ["+", "-", "*"]
    nums = [str(random.randint(1, max_number)) for _ in range(num_operands)]
    parts = [nums[0]]
    for n in nums[1:]:
        parts += [random.choice(ops), n]
    expr = " ".join(parts)
    return expr, safe_eval(expr)


def _make_task(env_cfg: dict[str, Any]) -> tuple[str, float, dict[str, str]]:
    """随机生成一个需要工具的任务，返回 (题面, 正确答案, 知识库)。"""
    max_number = int(env_cfg.get("max_number", 50))
    kind = random.choice(["calc", "search_calc", "code"])

    if kind == "calc":
        expr, target = _make_arith(max_number, int(env_cfg.get("num_operands", 3)))
        return (f"计算 {expr} 的结果（用 calc 工具）。", target, {})

    if kind == "search_calc":
        item = random.choice(ITEMS)
        price = random.randint(2, max_number)
        qty = random.randint(2, 9)
        # 知识库：目标事实 + 干扰项
        kb = {f"{item}单价": f"{item}的单价是 {price} 元"}
        for other in random.sample([x for x in ITEMS if x != item], k=3):
            kb[f"{other}单价"] = f"{other}的单价是 {random.randint(2, max_number)} 元"
        q = f"买 {qty} 个{item}一共多少钱？先用 search 查{item}单价，再用 calc 计算。"
        return (q, float(price * qty), kb)

    # code：用 python 求 1..N 的平方和
    n = random.randint(5, 20)
    target = sum(i * i for i in range(1, n + 1))
    return (f"求 1 到 {n} 的所有整数的平方和（用 python 工具计算）。", float(target), {})


def generate_datum(tokenizer, env_cfg: dict[str, Any], idx: int) -> DatumSpec:
    question, target, kb = _make_task(env_cfg)
    prompt = PROMPT_HEADER + f"问题：{question}"
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    token_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    message_log: LLMMessageLogType = [
        {"role": "user", "content": prompt_text, "token_ids": token_ids}
    ]
    metadata = {
        "target": float(target),
        "num_turns": 0,
        "max_turns": int(env_cfg.get("max_turns", 6)),
        "question": question,
        "answer_tolerance": float(env_cfg.get("answer_tolerance", 1e-6)),
        "kb": kb,
        "code_timeout": float(env_cfg.get("code_timeout", 5)),
    }
    return {
        "message_log": message_log,
        "length": len(token_ids),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": TASK_NAME,
        "stop_strings": STOP_STRINGS,
    }


class IterableToolDataset(IterableDataset):
    def __init__(self, tokenizer, env_cfg, length):
        super().__init__()
        self.tokenizer = tokenizer
        self.env_cfg = env_cfg
        self.length = length

    def __iter__(self) -> Iterator[DatumSpec]:
        for i in itertools.count():
            yield generate_datum(self.tokenizer, self.env_cfg, i)

    def __len__(self):
        return self.length


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

    env_cfg = config.env[TASK_NAME]["cfg"]
    env = ToolAgentEnv.options(num_gpus=0).remote(cfg=dict(env_cfg))
    task_to_env = {TASK_NAME: env}

    ds_length = (
        config.grpo["num_prompts_per_step"]
        * config.grpo["num_generations_per_prompt"]
        * config.grpo["max_num_steps"]
    )
    dataset = IterableToolDataset(tokenizer, env_cfg, ds_length)
    val_dataset = IterableToolDataset(tokenizer, env_cfg, config.grpo["max_val_samples"])

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
    ) = setup(config, tokenizer, dataset, val_dataset)

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
