# grpo_qwen3.5-4b_gsm8k_v1

单轮 GRPO：`qwen3.5-4b` 在 GSM8K 上做数学推理。来自低显存机型实测的
`grpo_math_4B_megatron` 配方（LoRA dim16 / lr 2e-4），并改为**非 colocated 生成**。

## 调什么（调参面）

打开 `config.yaml`，顶部有「调参速查」表，下面【① 调参区】就是你要动的几行：

| 旋钮 | 作用 | 典型范围 / 调过头 |
| --- | --- | --- |
| `lr` | 学习率 | LoRA 1e-4~2e-4；太高 reward 崩，太低学不动 |
| `dim` / `alpha` | LoRA 容量 | dim 8/16/32/64，alpha≈2×dim；大=能力强但易过拟合 |
| `reference_policy_kl_penalty` | 贴原模型力度 | 0~0.05；大=稳但慢，小=快但易跑偏 |
| `num_generations_per_prompt` | 每题采样数（组内基线） | 4/8/16；多=梯度稳但更慢更吃显存 |
| `num_prompts_per_step` | 每步题目数 | × 上一行 = `train_global_batch_size`（须相等） |
| `max_total_sequence_length` | 上下文长度 | 数学 1024 够用；大=更吃显存 |
| `max_num_steps` / `val_period` | 训多久 / 多久验证 | 看收敛情况调 |

> 硬件/分布式（卡数、并行度、NCCL）不在这里调，由服务端 profile 注册表下发（FORGE_PROFILE_OVERRIDES）。
> 奖励逻辑在 `env.math`（数学判分用 NeMo-RL 内置 `math_verify`），自定义奖励才动 `common/rewards/`。

## 与 9B 版的区别

| 项 | 4B（本实验） | 9B（`grpo_qwen3.5-9b_gsm8k_v1`） |
| --- | --- | --- |
| LoRA | dim16 / alpha32 / **lr 2e-4** | dim8 / alpha16 / lr 1e-4 |
| batch | num_prompts=4 / gen=4 / global=16 | num_prompts=4 / gen=8 / global=32 |
| 序列 | 1024 | 1250 |
| 生成 | **非 colocated**（1 卡生成 / 1 卡训练） | 非 colocated |

> **非 colocated + 2 卡 ⇒ PP=1**：2 张卡里 1 张专跑生成、1 张训练，训练侧只剩 1 卡，
> 无法 PP=2（PP=2 需要 2 张训练卡，那要回 colocated）。并行度由服务端 profile 注册表下发。

## 数据准备（与 9B 共用）

```bash
lab prepare gsm8k                              # 写到 datasets/gsm8k/{train,val}.jsonl
```

数据随作业上传，统一 launcher 会校验并准备 dataset manifest，提交时无需手动 export。

## 组成

- `config.yaml` — 继承 `grpo_math_1B` + `qwen3.5-4b` + `grpo_megatron`(Megatron+低显存调优) + `grpo_lora`(覆盖成 4B 配方) + `grpo_noncolocated`。
- `recipe.lock.json` — 固定 grpo recipe、官方入口与 digest。

## 运行

```bash
# 提交到集群（经中心化服务）
sf submit grpo_qwen3.5-4b_gsm8k_v1
lab logs                          # 跟随最近一个作业日志
```

## SwanLab

- project：`grpo_qwen3.5-4b_gsm8k_v1`，run：`lora-lr2e4-dim16-noncolo`，链接：<回填>

## 结果与结论

- 关键指标：val:accuracy / reward
- 结论 / 下一步：
