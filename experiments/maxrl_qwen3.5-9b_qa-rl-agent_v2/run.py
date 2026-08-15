#!/usr/bin/env python
# 题库「多轮 + 本地文档检索」Agent GRPO 训练脚本（NeMo-RL 0.7.0）——【MaxRL 版 / v2】。
#
# 与 v1（grpo_qwen3.5-9b_qa-rl-agent_v1）的唯一差异：把 GRPO 的优势归一化从「除以组内标准差 σ」
#   改成论文《Maximum Likelihood Reinforcement Learning》(MaxRL, arXiv:2602.02710) 的「除以组内平均奖励 μ」：
#       GRPO : Â = (r-μ)/σ        MaxRL : Â = (r-μ)/μ   （μ=0 整组置 0）
#   实现见 common/algorithms/maxrl.py；通过 run_grpo 的 before_train 钩子安装 install_maxrl_estimator()，
#   并由 config.yaml 的 grpo.adv_estimator.name="maxrl" 触发。数据/模型/LoRA/batch/seq/裁判/多轮检索
#   与 v1 完全一致，唯一变量是 RL 目标（GRPO → MaxRL），便于直接对比两条曲线。
#
# 数据：datasets/qa_rl 的 train/val jsonl，与 baseline 完全一致；
#       共享数据集实现在 common/data/qa_docs_dataset.py，本脚本只注入检索说明文案。
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig

from common import bootstrap
from common.algorithms.maxrl import install_maxrl_estimator
from common.data.qa_docs_dataset import TASK_NAME, build_qa_docs_datasets
from common.environments.qa_docs_agent_env import QADocsAgentEnv, make_eval_cfg

# 在原题面前加这段说明，告诉模型「可多轮检索本地资料」；答案格式仍由题目自带的 \boxed{} 指令决定。
# ⚠️ Base 模型（非 Instruct）零样本难以稳定遵循「<search>…</search> + \boxed{}」多轮协议，
#    故内置一个 1-shot 示例把完整轮次（检索→收到[检索结果]→作答）演示一遍，显著提升格式遵循率。
#    示例只演示协议形态，不泄露真实题目答案；baseline 也用同款 Base，加示例不改变「检索」这个唯一变量。
DOCS_PREAMBLE = (
    "你可以在回答前多轮检索公司技术资料库来获取依据：\n"
    "需要检索时，输出 <search>关键词</search>（系统会用 grep 在资料库里查并把相关片段以「[检索结果]」回灌给你）；可多次换关键词检索。\n"
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
        THIS_DIR, MasterConfig, description="题库多轮+本地文档检索 GRPO 训练（MaxRL）"
    )
    tokenizer = bootstrap.init_runtime(config, "grpo")

    train_dataset, val_dataset, env_cfg = build_qa_docs_datasets(
        config, tokenizer, preamble=DOCS_PREAMBLE,
        data_dir_hint="config 需声明 data.train.dataset（平台引用，提交时注入 QA_RL_DATA_DIR）；"
                      "本地跑请 `export QA_RL_DATA_DIR=<repo>/datasets/qa_rl`。",
    )

    # 训练/验证用两个实例：检索与判分完全相同，只有 reward shaping 不同（训练带、验证归零）。
    # NeMo-RL 的 validation/accuracy = mean(total_reward)（逐轮奖励累加），验证若带 shaping 则
    # 「用了工具」本身白送分 → 验证分虚高，且与 GRPO v3 / 无工具 baseline 不再同尺度。
    train_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=env_cfg)
    val_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=make_eval_cfg(env_cfg))
    print("训练环境带检索 reward shaping；验证环境已归零 → validation/accuracy = 纯答题得分")

    # MaxRL 优势估计器必须在 grpo_train() 之前注册：估计器是在 grpo_train 内部
    # _create_advantage_estimator() 创建的（grpo.adv_estimator.name=="maxrl" 时生效）。
    bootstrap.run_grpo(
        config, tokenizer, train_dataset, val_dataset,
        {TASK_NAME: train_env}, {TASK_NAME: val_env},
        before_train=install_maxrl_estimator,
    )


if __name__ == "__main__":
    main()
