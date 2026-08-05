#!/usr/bin/env python
# OPSD（On-Policy Self-Distillation，arXiv:2601.18734）复现实验入口。
#
# 一句话：同一个模型既当学生又当老师——学生只看题目、on-policy 采样出一条解答；
#         老师看到「题目 + 参考解」，对**学生刚采样出的那条解答**做 teacher-forcing 前向。
#         老师因为偷看了答案，分布更尖锐更正确，用它逐 token 监督学生。
#         算法实现见 common/algorithms/opsd.py（含索引对齐推导）。
#
# 与官方实现（siyan-zhao/OPSD，基于 TRL 的 GOLD trainer）的关系：
#   本脚本不引入 TRL / flash-attn 等任何新依赖，而是复用 NeMo-RL 0.7.0 自带的 on-policy
#   distillation 主循环，只把「老师吃什么输入」这一处换掉。对只能访问内网的 Ray 集群来说，
#   这意味着**零镜像变更**即可开跑。
#
# 数据：本地 jsonl（每行 {"problem", "solution", "answer"}），不走 HF datasets——
#       集群无外网，任何 dataset_name 形式的在线拉取都会失败。
#       solution = 参考解全文（喂给老师）；answer = 最终答案（验证时判分用）。

import argparse
import json
import os
import pprint
import sys
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.distillation import MasterConfig, distillation_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.algorithms.opsd import HINT_KEY, install_opsd, make_clipped_loss_fn

TASK_NAME = "math"

# 学生看到的：只有题目。
STUDENT_TEMPLATE = "{problem}\n\n请一步步推理，并把最终答案写进 \\boxed{{}}。"

# 老师看到的：题目 + 参考解。老师并不是要「复述」这段参考解，而是在读过它之后，
# 对学生已经写下的那条解答重新给出逐 token 分布——这就是 OPSD 的监督信号来源。
TEACHER_TEMPLATE = (
    "{problem}\n\n"
    "（以下参考解仅供你内部参考，不要在回答里提及它的存在）\n"
    "参考解：{solution}\n\n"
    "请一步步推理，并把最终答案写进 \\boxed{{}}。"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="OPSD 自蒸馏训练")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line := line.strip():
                rows.append(json.loads(line))
    return rows


class OPSDMathDataset(Dataset):
    """读本地 jsonl，产出带「参考解条件化 prompt」的 DatumSpec。

    每条样本准备两份 prompt：
      message_log            学生用（只有题目）——它决定 on-policy 采样出什么。
      extra_env_info[HINT_KEY]  老师用（题目+参考解）——只在算 top-k 分布时用，不参与采样。

    两者都在这里就 tokenize 好，避免把 tokenizer 传到 OPSDTeacher 里（老师侧只做张量搬运）。
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        *,
        problem_key: str = "problem",
        solution_key: str = "solution",
        answer_key: str = "answer",
        system_prompt: str | None = None,
    ):
        self.rows = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.problem_key = problem_key
        self.solution_key = solution_key
        self.answer_key = answer_key
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, text: str) -> tuple[torch.Tensor, str]:
        chat: list[dict[str, str]] = []
        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.append({"role": "user", "content": text})
        rendered = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=False
        ).strip()
        ids = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        return ids["input_ids"][0], rendered

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        problem = str(row[self.problem_key])
        solution = str(row.get(self.solution_key, ""))
        answer = str(row.get(self.answer_key, "")) or solution

        student_ids, student_text = self._encode(
            STUDENT_TEMPLATE.format(problem=problem)
        )
        hint_ids, _ = self._encode(
            TEACHER_TEMPLATE.format(problem=problem, solution=solution)
        )

        message_log: LLMMessageLogType = [
            {"role": "user", "content": student_text, "token_ids": student_ids}
        ]
        return {
            "message_log": message_log,
            "length": len(student_ids),
            "extra_env_info": {
                # MathEnvironment 用它在验证时判分（accuracy）。
                "ground_truth": answer,
                # OPSDTeacher 用它换掉老师侧的 prompt 段。
                HINT_KEY: hint_ids,
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
        }


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    print(f"已加载配置: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)

    # opsd 段是本仓库自定义的（MasterConfig 是 extra="allow"，但先取出来更清晰）。
    opsd_cfg: dict[str, Any] = dict(config.pop("opsd", {}) or {})
    config: MasterConfig = MasterConfig(**config)
    print("最终配置：")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 日志目录: {config.logger['log_dir']}")

    init_ray()
    set_seed(config.distillation["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    # ── 数据 ────────────────────────────────────────────────────────────────
    data_cfg: dict[str, Any] = config.data
    data_dir = os.environ.get("OPSD_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit(
            "未指定数据目录。请设置 OPSD_DATA_DIR 或在 config.yaml 的 data.data_dir 写死"
            "（目录下需有 train.jsonl / val.jsonl，字段 problem / solution / answer）。"
        )

    ds_kwargs = dict(
        problem_key=data_cfg.get("problem_key", "problem"),
        solution_key=data_cfg.get("solution_key", "solution"),
        answer_key=data_cfg.get("answer_key", "answer"),
        system_prompt=data_cfg.get("system_prompt") or None,
    )
    train_dataset = OPSDMathDataset(
        os.path.join(data_dir, "train.jsonl"), tokenizer, **ds_kwargs
    )
    val_dataset = OPSDMathDataset(
        os.path.join(data_dir, "val.jsonl"), tokenizer, **ds_kwargs
    )
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(val_dataset)} 条")

    env = MathEnvironment.options(num_gpus=0).remote(cfg=config.env[TASK_NAME])
    task_to_env = {TASK_NAME: env}

    # ── 安装 OPSD ───────────────────────────────────────────────────────────
    # 必须在 setup() 之前：setup 里会先建老师、再建学生，install_opsd 正是劫持这两次构造，
    # 好让 teacher_mode="self" 时老师直接复用学生权重（不加载第二份模型）。
    install_opsd(
        teacher_mode=opsd_cfg.get("teacher_mode", "self"),
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=config.policy["max_total_sequence_length"],
        make_divisible_by=config.policy.get("make_sequence_length_divisible_by", 1),
    )

    (
        student_policy,
        teacher_policy,
        student_generation,
        _nemo_gym,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    # per-token KL 截断（论文的稳定化手段）；opsd.per_token_kl_clip = null 则退化成原版损失。
    loss_fn = make_clipped_loss_fn(config.loss_fn, opsd_cfg.get("per_token_kl_clip"))

    distillation_train(
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        distillation_state,
        master_config,
    )


if __name__ == "__main__":
    main()
