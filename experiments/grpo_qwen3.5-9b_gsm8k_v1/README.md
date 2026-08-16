# grpo_qwen3.5-9b_gsm8k_v1

单轮 GRPO 实验：`qwen3.5-9b` 在 GSM8K 上做数学推理。作为多轮 Agent 实验的**对照**
（同模型、同 GRPO，但单轮、无工具）。

## 目标

用标准 GRPO（`max_rollout_turns=1`）在 GSM8K 上训练，观察答对率随训练提升，与
`agent-grpo_qwen3.5-9b_multitool_v1` 的多轮工具范式对比。

## 数据准备（先做）

GSM8K 的答案是「推理 + #### 数字」，需抽取干净金标准答案：

```bash
# 在仓库根目录
lab prepare gsm8k                              # 写到 datasets/gsm8k/{train,val}.jsonl
```

- **`forge submit`（经服务端到集群）**：`datasets/gsm8k/` 随作业上传，统一 launcher 校验并准备 dataset manifest。

## 组成

- `config.yaml` — 继承 `grpo_math_1B` + `qwen3.5-9b` + `grpo_megatron`(Megatron+低显存调优) + `grpo_lora`(LoRA)，
  `_override_` 替换 `data` 为 `ResponseDataset` 指向上面的 jsonl，处理器 `math_hf_data_processor`、环境 `math`。
- `recipe.lock.json` — 固定 grpo recipe、官方入口与 digest。

## 关键超参（低显存实测起点）

- 后端：Megatron-Core + **LoRA**（dim8/alpha16，lr 1e-4，wd 0，cosine）。回全参数：删 `defaults` 里 `grpo_lora.yaml`。
- batch：`num_prompts_per_step=4`、`num_generations_per_prompt=8`、`train_global_batch_size=32`、`micro=1`、`seq=1250`。
- 显存紧：降 `max_total_sequence_length`、`gpu_memory_utilization`，或减 `num_generations_per_prompt`（注意 global 要整除 prompts×gen）。

## SwanLab

- project：`grpo_qwen3.5-9b_gsm8k_v1`，run：`lora-lr1e4-g8-kl0.01`，链接：<回填>

## 运行

```bash
forge submit grpo_qwen3.5-9b_gsm8k_v1
```

产物落到本目录 `outputs/`（已 .gitignore）。

## 结果与结论

- 关键指标：val:accuracy / reward
- 结论 / 下一步：
