# 命名规范

统一命名能让几十上百个微调实验保持可检索、可对比。**新建任何实验前先读本文件。**

## 实验目录命名

```
<method>_<model>_<dataset>[_<tag>]
```

字段之间用下划线 `_`，字段内部用连字符 `-`。

### method（训练方法）

| 取值 | 含义 |
| --- | --- |
| `sft` | 监督微调 |
| `grpo` | GRPO 强化学习 |
| `dpo` | DPO |
| `ppo` | PPO |
| `rm` | 奖励模型训练 |
| `agent-grpo` | 多轮 Agent / 工具调用的 GRPO 训练 |

### model（基础模型）

`<家族><版本>-<规模>`，全小写。例如：

- `qwen3.5-4b`
- `qwen3.5-9b`
- `llama3.1-8b`

规模统一小写 `b`（billion）。带后缀的模型如 `qwen3.5-4b-instruct`。

### dataset（数据集）

数据集短名，全小写：`gsm8k`、`alpaca`、`math`、`toolbench`、`hh-rlhf` 等。混合数据集用 `+` 连接，如 `gsm8k+math`。

### tag（可选）

用来区分同一组合的多次迭代：`v1`、`v2`，或日期 `20260602`，或关键差异 `lr2e6`、`8k-ctx`。

### 示例

```
sft_qwen3.5-4b_alpaca_v1
sft_qwen3.5-9b_alpaca+sharegpt_v2
grpo_qwen3.5-4b_gsm8k_v1
grpo_qwen3.5-9b_math_20260602
rm_qwen3.5-4b_hh-rlhf_v1
agent-grpo_qwen3.5-9b_toolbench_v1
```

## 实验目录一律放 experiments/

练习、调参、复现，以及正式交付、需长期维护的方向，都放 `experiments/`。
（早期分出的 `projects/` 已并入 `experiments/`，不再区分两套布局。）

## 实验目录 ≠ 项目

`experiments/<名字>/` 是一次可提交的**实验**；**项目**是 `starforge.yaml` 的
`name`，控制台按 `@<用户名>/<项目名>` 把提交聚合成项目。别把两者混称。

## SwanLab 命名对齐

为了让代码仓库和 SwanLab 看板能对上：

- **SwanLab project** = 实验目录名（如 `grpo_qwen3.5-9b_gsm8k_v2`），或按模型聚合（如 `qwen3.5-9b`）。
- **SwanLab experiment（run）名** = 关键超参组合，如 `lr1e6-bs64-kl0.001`。

在实验的 `README.md` 里务必贴上 SwanLab 链接。详见 [`swanlab.md`](swanlab.md)。

## checkpoint / 导出命名

```
<实验名>/outputs/step_<N>/          # 训练中间 checkpoint（.gitignore）
<实验名>/outputs/final/             # 最终 checkpoint
<实验名>/hf_export/                 # 转 HuggingFace 格式（.gitignore）
```
