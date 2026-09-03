# DeepSeek-V4-Flash-0731 · GRPO + LoRA（8×H200）

从 `verl-grpo_qwen3.5-9b_qa-tools_v1` 分叉：同一套题库、Agent Loop、检索工具和奖励，换
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
（284B MoE / 13B 激活）。训练后端按 veRL 0.9.0 / [#6473](https://github.com/verl-project/verl/pull/6473)
走 **Megatron-Bridge actor/ref + vLLM rollout**，用 LoRA 把全参 64 卡压到单机 8 张 H200。

## 为什么不能照抄源实验的 FSDP2

源实验是 Qwen3.5-9B + `fsdp2` + 扁平 `model.lora_rank`。DSv4 带 MLA、DSA hybrid attention、
MXFP4 routed experts 和 hash-router，官方只验了 Megatron。扁平 LoRA 键给 FSDP 读，
Megatron 只认嵌套 `model.lora.*`——写错会静默退回全参，8 卡立刻 OOM。

| | 官方 `run_deepseek_v4_flash_megatron.sh` | 本实验 |
|---|---|---|
| 硬件 | 8 节点 × 8 卡（64 GPU） | 1 节点 8×H200（`--profile h200`） |
| 后端 | Megatron-Bridge + vLLM | 同左 |
| 参数 | 全参，lr `1e-6` | LoRA rank 32 / alpha 64，lr `1e-4` |
| 并行 | TP=1, **PP=8**, EP=8, DP=8 | TP=1, **PP=1**, EP=8, DP=8 |
| 数据 | DAPO-Math / AIME | 同一份 `qa-rl-verl` + `--corpus` |
| 序列 | prompt 2048 + response **10240** | 2048 + 2048（题库口径） |
| 奖励 | DAPO overlong buffer | `reward.py` → `common/rewards` |

PP=8 且 EP=8 需要 `DP ≥ 8`，world size 至少 64。8 卡上只能 PP=1、EP=8。
不要抄官方的 `pipeline_model_parallel_layout`。

## 文件

- `config.yaml` — Megatron LoRA / 并行 / vLLM FP8 / Agent Loop
- `tools.py` — `@function_tool` 检索（读 `DOCS_DIR`）
- `reward.py` — `compute_score`，复用 `common/rewards`
- `prepare_data.py` — jsonl → parquet（已有 `aiden_lu/qa-rl-verl@v1` 可跳过）
- `eval.py` — recipe lifecycle 钩子
- `recipe.lock.json` — 锁 `verl/grpo` 0.9.0

## 提交

```bash
sf validate verl-grpo_deepseek-v4-flash_qa-tools_v1
sf submit verl-grpo_deepseek-v4-flash_qa-tools_v1 --profile h200 \
    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
    --train-dataset aiden_lu/qa-rl-verl@v1 --train-data train.parquet \
    --validation-dataset aiden_lu/qa-rl-verl@v1 --validation-data val.parquet \
    --corpus aiden_lu/qa-docs@v1
```

`--profile h200` 是整节点 8 张。写成 `h200:1` 会按单卡起，EP=8 直接起不来。

漏写 `--corpus` 不会报错，检索永远空手，reward 上不去。

## 关键旋钮

| 键 | 值 | 原因 |
|---|---|---|
| `model_engine` | `megatron` | Hydra 用来组装 `megatron_actor` / `megatron_ref` |
| `model.lora.rank/alpha` | 32 / 64 | Megatron 嵌套键；对齐源实验 |
| `model.lora.merge` | true | vLLM 0.24 的 DSv4 还不支持 LoRA；折进基座再同步 |
| `lora.target_modules` | `linear_q_down/up` + `linear_kv_proj` + `linear_proj` + `linear_fc1/fc2` + CSA indexer | hybrid 没有 `linear_kv_down/up` / `linear_q_proj` / `linear_wk` |
| `actor.optim.lr` | `1e-4` | LoRA；不要用官方全参的 `1e-6` |
| `megatron.EP` / rollout `EP=DP` | 8 | 8 卡 MoE；vLLM 要求 `EP = TP × DP` |
| `param/grad/optimizer_offload` | true | 284B 基座冻结也要占显存 |
| `router_replay.mode` | `R3` | 与 rollout 路由对齐 |
| `rollout.enforce_eager` | true | 量化权重重构会打坏 CUDA graph |
| `rollout.load_format` | `safetensors` | vLLM 先加载基座；merge 后再同步折好的权重 |
| `gpu_memory_utilization` | 0.5 | 官方口径；0.3 会在 KV cache 分配时报没内存 |
| `kv_cache_dtype` | `fp8` | 官方 DSv4 脚本 |

纯显存旋钮（改了不影响梯度）：`ppo_micro_batch_size_per_gpu`、`gpu_memory_utilization`、`max_num_seqs`。

## DSML 工具格式

DSv4-0731 的工具调用是 DSML（`<｜DSML｜invoke>`），不是 hermes 的 `<tool_call>`。
veRL 0.9 的 `multi_turn.format` 没有 `dsml`。本实验开了
`use_inference_chat_template`，让模型自带 encoding 注入工具 schema。

若 rollout 日志里始终没有 `search_docs`：

1. 先确认 `--corpus` 已挂上、`DOCS_DIR` 有文件。
2. 看生成文本里是 DSML 还是 `<tool_call>`。若是前者，hermes 解析器吃不到，工具不会跑，
   训练仍会走——reward 会偏低，看起来像「没学到」。
3. 要先验证 Megatron+LoRA 链路，可把 `multi_turn.enable` 设 false、数据集里的
   `agent_name` 改成 `single_turn_agent`，单轮 GRPO 能先跑通。

## 常见坑（本实验新增）

源实验 README 的平台坑（数据集/模型不要写进 YAML、必须 parquet、logger 用 console）
和奖励坑（`[type]` 前缀、两套 reward 键）这里都适用，不重复。

1. **`model_engine` 没切到 megatron** → 仍组装 FSDP actor，`megatron.*` 无效或启动即报错。
2. **LoRA 写成扁平 `lora_rank`** → Megatron 不读，退回全参，8 卡 OOM。
3. **`--profile h200:1`** → EP=8 要 8 张卡。
4. **照抄官方 PP=8 + pipeline layout** → 8 卡 DP=1，EP=8 不整除。
5. **照抄官方 lr `1e-6`** → LoRA 几乎不动。
6. **`override_transformer_config` 里 DSv4 专用键被 Hydra struct 拒** → 启动报
   `Key 'apply_dsa_kernel_fusion' is not in struct`。这些键已写成 config 末尾的
   `+actor_rollout_ref...` 顶层项，不要再嵌进 `override_transformer_config:` 字典。

7. **vLLM `DeepseekV4ForCausalLM does not support LoRA yet`** → 必须 `model.lora.merge=true`。
   `merge=false` 会给 vLLM 传 `enable_lora`，0.24 直接炸。

8. **RewardLoop `KeyError: deepseek_v4`** → 那些 worker 不读 `external_lib`。
   镜像里给 `verl.utils.tokenizer.hf_tokenizer` 打了 `dsv4_hf_register`；
   Ray 的 `worker_process_setup_hook` 字符串不会执行，单靠它不够。

9. **镜像** 必须是平台 `verl-0.9.0`（含 Megatron-Bridge 与 DSv4 的 vLLM FP8/MXFP4 路径）。
   旧镜像会在建模型或权重同步时报错。先**不要**为了 chat template 去重编镜像：
   DSv4 官方仓库就没有 jinja，`tokenizer.apply_chat_template` 会把整份题库滤成 0 条。
   `common.hf_compat` 会在启动时挂上一份最小 DSv4 模板；rollout 仍走
   `tokenizer_mode=deepseek_v4` 的官方 Python encoding。
