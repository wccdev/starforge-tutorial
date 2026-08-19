# starforge-tutorial

StarForge（星锻）**官方示例项目**。布局与 `sf init` 生成的用户项目相同：这是给人读、给人 fork 的完整微调仓库，不是 CLI 工具源码。

```bash
pip install starforge-cli
sf login --server https://<你的 StarForge 域名>
sf ls
sf validate grpo_qwen3.5-4b_gsm8k_v1
# 日常请另开自己的项目：
#   sf init my-lab --yes
```

CLI 与控制平面源码在 [starforge](https://github.com/wccdev/starforge) 仓的 `cli/` 与 `server/`。

## 示例实验

| 实验 | 方法 |
| --- | --- |
| `experiments/sft_qwen3.5-4b_alpaca_v1` | SFT |
| `experiments/grpo_qwen3.5-4b_gsm8k_v1` | GRPO |
| `experiments/grpo_qwen3.5-9b_gsm8k_v1` | GRPO |
| `experiments/agent-grpo_qwen3.5-9b_multitool_v1` | 多轮 Agent |
| `experiments/verl-grpo_qwen3.5-9b_qa-tools_v1` | verl GRPO |
| `experiments/trl-grpo_qwen3.5-9b_qa-tools_v1` | TRL GRPO |

`common/` 是随作业包上传的共享代码（数据脚本 / 环境 / 奖励）。`plugins/` 是示例算法插件。

## 目录

```
starforge-tutorial/
├── starforge.yaml      # 仓库标记，并声明项目名
├── experiments/        # 示例实验
├── configs/            # 官方基底 + 模型片段
├── common/             # 共享代码
├── plugins/            # 示例插件
└── smoke/              # 最小 GPU smoke
```
