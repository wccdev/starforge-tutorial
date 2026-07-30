# configs/ — 配置继承体系

NeMo-RL 0.7.0 **原生支持配置继承**（`nemo_rl/utils/config.py` 的 `load_config`）：配置里写
`defaults: parent.yaml` 即可继承，支持多继承、嵌套继承、`${}` 插值、`_override_` 整段覆盖，
再叠加命令行 Hydra override。官方自己就这么用（`grpo_sliding_puzzle.yaml` 继承 `grpo_math_1B.yaml`）。

所以每个模型 / 每个实验都有自己的配置、不断调参，但**只写差异**，公共部分继承。

## 三层结构

```
configs/
├── base/      祖父：官方 v0.7.0 example 原样副本（version-locked，由 sync_base_configs.sh 同步，勿手改）
│   ├── grpo_math_1B.yaml          # GRPO 基底
│   ├── sft.yaml                   # SFT 基底
│   ├── grpo_sliding_puzzle.yaml   # 多轮 Agent 基底（已继承 grpo_math_1B）
│   ├── grpo_megatron.yaml         # ★自定义 overlay：Megatron 后端 + GB10 实测显存/性能调优
│   ├── grpo_lora.yaml             # ★自定义 overlay：LoRA/PEFT（GB10 上让 9B 能跑起来）
│   └── grpo_noncolocated.yaml     # ★自定义 overlay：非 colocated 生成（1 卡生成 / 1 卡训练）
└── models/    父：各基础模型的公共片段（model_name / tokenizer / 显存策略…）
    ├── qwen3.5-4b.yaml
    └── qwen3.5-9b.yaml
```

## 训练后端 + LoRA（两个 overlay）

本仓库 GRPO 实验默认 **Megatron-Core + LoRA**，靠两个 overlay 叠加（都来自在 2× GB10 上跑通的配置）：

```yaml
defaults:
  - ../../configs/base/grpo_math_1B.yaml
  - ../../configs/models/qwen3.5-9b.yaml
  - ../../configs/base/grpo_megatron.yaml   # ① Megatron 后端 + GB10 显存/性能调优
  - ../../configs/base/grpo_lora.yaml       # ② LoRA（lr 1e-4 / wd 0 / cosine）
policy:
  max_total_sequence_length: 1250
  train_global_batch_size: 32               # = num_prompts_per_step * num_generations_per_prompt
grpo:
  num_prompts_per_step: 4
  num_generations_per_prompt: 8
```

- `grpo_megatron.yaml`：关 DTensor、开 Megatron，置空 FSDP 的 `policy.optimizer/scheduler`，并带上 GB10 实测显存项（`activation_checkpointing` / `empty_unused_memory_level=2` / `apply_rope_fusion=false` / `defer_fp32_logits` / `enforce_eager` / 关 sequence packing 等）。
- `grpo_lora.yaml`：开 `megatron_cfg.peft`，LoRA 学习率写在 `megatron_cfg.optimizer.lr`（1e-4，比全参数高 2 个量级）。
- `grpo_noncolocated.yaml`（可选第三层）：生成与训练各占独立 GPU（实测 9B 用法）。2× GB10 上 = 1 卡生成 / 1 卡训练 → 训练侧 PP=1。删此行即回 colocated（共用 GPU）。

切回方式：
- **回全参数**：删 `grpo_lora.yaml` 一行，并把 lr 写到 `policy.megatron_cfg.optimizer.lr`（如 1e-6）。
- **回 DTensor/FSDP**：删 `grpo_megatron.yaml` 一行，并把 lr 写回 `policy.optimizer.kwargs.lr`。

> 并行度（TP/PP/CP）放 `cluster/<profile>/overrides.conf`；9B 实测 PP=1，4B 实测 PP=2。
> `converter_type` 无需按模型改：Megatron 用 AutoBridge 按 HF 架构自动识别。
> 显存类调优放在 overlay（merge）而非 `overrides.conf`：CLI override 是 struct 模式且对 SFT 也生效，而 SFT 没有 `policy.generation` 等键。

实验目录里的 `config.yaml` 是**子**：

```yaml
defaults:
  - ../../configs/base/grpo_math_1B.yaml   # 方法基底
  - ../../configs/models/qwen3.5-9b.yaml   # 模型片段
# 下面只写本实验差异：数据集 / lr / kl / swanlab / 步数 ...
grpo:
  num_generations_per_prompt: 16
loss_fn:
  reference_policy_kl_penalty: 0.01
logger:
  swanlab_enabled: true
  swanlab: { project: "grpo_qwen3.5-9b_gsm8k_v1", name: "lr1e6-g16-kl0.01" }
```

> 合并规则：多继承中**后面覆盖前面**，实验自身再覆盖基底/模型。需要整段替换（不合并）时，
> 在该段加 `_override_: true`。

## 方法 → 入口 → 基底

| 方法 | `ENTRY`（run.sh） | 基底（defaults 第一项） |
| --- | --- | --- |
| SFT | `examples/run_sft.py` | `../../configs/base/sft.yaml` |
| GRPO | `examples/run_grpo.py` | `../../configs/base/grpo_math_1B.yaml` |
| 多轮 Agent | `examples/run_grpo.py` | `../../configs/base/grpo_sliding_puzzle.yaml` |

> 更大模型可加 `grpo_math_8B.yaml` 等官方基底：把文件名加进 `scripts/sync_base_configs.sh` 同步。

## 集群 / 硬件

不进 `config.yaml`，由 `cluster/<profile>/overrides.conf` 在运行时以 CLI override 叠加（见 `cluster/README.md`），
这样切硬件（GB10 ↔ H200）不动实验配置。

## 常用字段（0.7.0）

| 作用 | key |
| --- | --- |
| 基础模型 | `policy.model_name` |
| 序列长度 | `policy.max_total_sequence_length` |
| 学习率（Megatron/LoRA，默认） | `policy.megatron_cfg.optimizer.lr`（LoRA 用 1e-4） |
| 学习率（DTensor 后端） | `policy.optimizer.kwargs.lr` |
| LoRA 开关 / 秩 | `policy.megatron_cfg.peft.enabled` / `.dim` / `.alpha` |
| 全局 batch（须整除 prompts×gen） | `policy.train_global_batch_size` |
| KL 惩罚 | `loss_fn.reference_policy_kl_penalty` |
| 每步 prompt 数 / 每 prompt 采样数 | `grpo.num_prompts_per_step` / `grpo.num_generations_per_prompt` |
| 多轮轮数 | `grpo.max_rollout_turns` |
| 节点 / 卡数 | `cluster.num_nodes` / `cluster.gpus_per_node` |
| 启用 SwanLab | `logger.swanlab_enabled` + `logger.swanlab.{project,name}` |
