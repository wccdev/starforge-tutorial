#!/usr/bin/env python
# 题库「多轮 + 本地文档检索」Agent GRPO 训练脚本（NeMo-RL 0.7.0）。
#
# 这是与单轮 baseline（grpo_qwen3.5-9b_qa-rl_v1）做 A/B 对比的【对照组 / treatment】：
#   同一份题库数据 / 模型 / LoRA / batch / seq / 裁判奖励，唯一差异是——
#   模型回答前可以**多轮调用 <search> 在集群容器内 grep 本地资料**（/data/docs 下的 markdown）再作答
#   （见 common/environments/qa_docs_agent_env.py）。
#
# 数据：datasets/qa_rl 的 train/val jsonl（每行 {"query", "expected_answer": "[type] ..."}），与 baseline 完全一致；
#       共享数据集实现在 common/data/qa_docs_dataset.py，本脚本只注入检索说明文案。
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig

from common import bootstrap
from common.data.qa_docs_dataset import TASK_NAME, build_qa_docs_datasets
from common.environments.qa_docs_agent_env import QADocsAgentEnv, make_eval_cfg

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


def main():
    config, _ = bootstrap.load_experiment_config(
        THIS_DIR, MasterConfig, description="题库多轮+本地文档检索 GRPO 训练"
    )
    tokenizer = bootstrap.init_runtime(config, "grpo")

    train_dataset, val_dataset, env_cfg = build_qa_docs_datasets(
        config, tokenizer, preamble=DOCS_PREAMBLE,
        data_dir_hint="config 需声明 data.train.dataset（平台引用，提交时注入 QA_RL_DATA_DIR）；"
                      "本地跑请 `export QA_RL_DATA_DIR=<repo>/datasets/qa_rl`。",
    )

    # 训练环境与验证环境用【两个实例】：检索后端、判分方式完全相同，只有 reward shaping 不同。
    #   训练：带 shaping（检索即时奖励 / 检索后答对加成 / 不作答惩罚）——引导模型真的去查资料。
    #   验证：make_eval_cfg() 把 shaping 归零，只留最终判分。
    # 必须分开：NeMo-RL 的 validation/accuracy 就是 mean(total_reward)，而 total_reward 是逐轮奖励累加，
    # 若验证也带 shaping，则「用了工具」本身白送分（本配置最多 +0.2 绝对值），验证分虚高，
    # 且与单轮无工具 baseline（qa-rl_v1，纯最终判分）不再同尺度，A/B 结论会被工具加分污染。
    train_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=env_cfg)
    val_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=make_eval_cfg(env_cfg))
    print("训练环境带检索 reward shaping；验证环境已归零 → validation/accuracy = 纯答题得分")

    bootstrap.run_grpo(
        config, tokenizer, train_dataset, val_dataset,
        {TASK_NAME: train_env}, {TASK_NAME: val_env},
    )


if __name__ == "__main__":
    main()
