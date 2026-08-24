# verl Agent Loop 工具调用示例：题库 GRPO + 本地文档检索

与 nemo-rl 的 `grpo_qwen3.5-9b_qa-rl-agent_v3` 是**同一场景的 verl 官方实现对照**：
模型答题前可多轮调用检索工具查本地资料（`DOCS_DIR`，默认 `/data/docs`）再作答。
展示 verl 做「自定义环境 + Agent 多轮」的全部三个官方机制，**不需要自定义入口**。

## 两条路线的对照

| | nemo-rl 路线（qa-rl-agent_v3） | verl 路线（本实验） |
|---|---|---|
| 多轮循环 | 自定义 Environment 类 + `max_rollout_turns` | 官方 Agent Loop（`tool_agent`），config 声明 |
| 环境交互 | `QADocsAgentEnv.step()` 解析 `<search>` | `tools.py` 里 `@function_tool` 的 `search_docs` |
| 判分 | Environment 内调 `common/rewards` | `reward.py` 的 `compute_score`（`custom_reward_function`） |
| 入口 | 实验自带 `run.py` | 官方 `verl.trainer.main_ppo`，无自定义入口 |

## 硬件与 A/B 契约

**跑在 1 张 H200 上**，提交用 `--profile h200:1`（单写 `h200` 是整节点 8 张）。

nemo-rl 侧注释写明「batch/模型/LoRA/裁判须严格一致，核心变量只有多轮+检索」，
所以两侧这些值必须成对改。左列是 nemo-rl 侧的权威出处：

| nemo-rl（qa-rl-agent_v3） | verl（本实验 `config.yaml`） |
|---|---|
| `megatron_cfg.peft` dim 32 / alpha 64 | `actor_rollout_ref.model.{lora_rank,lora_alpha}`（扁平键，见坑 17） |
| `dropout: 0` | FSDP 路径无此键，PEFT 默认即 0 |
| `target_modules` 四个 Megatron 融合层名 | 换算成 7 个 HF 名（`q/k/v/o/gate/up/down_proj`） |
| lr 1e-4、min_lr 1e-5、wd 0、cosine、warmup 50/1000 | `actor.optim.{lr,min_lr_ratio,weight_decay,lr_scheduler_type,lr_warmup_steps_ratio}` |
| `num_prompts_per_step: 8` | `data.train_batch_size: 8` |
| `num_generations_per_prompt: 8` | `actor_rollout_ref.rollout.n: 8` |
| `train_global_batch_size: 64` | `actor.ppo_mini_batch_size: 8`（verl 是题数口径，内部 ×n） |
| `max_total_sequence_length: 4096` | `max_prompt_length 2048 + max_response_length 2048` |
| `max_rollout_turns: 3` | `multi_turn.max_assistant_turns: 3` |

不属于契约的纯显存/速度旋钮（改了不影响梯度数学，可按余量调）：
`ppo_micro_batch_size_per_gpu`、`log_prob_micro_batch_size_per_gpu`、`gpu_memory_utilization`。

## 文件说明

- `config.yaml` — Agent Loop / 多轮 / 工具 / 奖励全部在此声明（详见文件内注释）
- `tools.py` — `@function_tool` 检索工具；schema 由类型注解 + Google docstring 自动推断
- `reward.py` — `compute_score(data_source, solution_str, ground_truth, extra_info)` 最终判分
- `prepare_data.py` — 题库 jsonl → parquet，**每行注入 `agent_name: "tool_agent"`（关键）**
- `eval.py` — recipe lifecycle eval 钩子（模板标配）

## 跑起来

verl 要 parquet，而共享题库 `aiden_lu/qa-rl` 是 nemo-rl 口径的 jsonl，所以 verl 路线
另有一份 parquet 数据集 `aiden_lu/qa-rl-verl`（同一份题库转的，不引入数据差异）。

```bash
# 1. 数据（本地一次性）：jsonl 题库 → verl parquet，再推成平台数据集版本
python experiments/verl-grpo_qwen3.5-9b_qa-tools_v1/prepare_data.py \
    --data-dir datasets/qa_rl --out-dir /tmp/qa_rl_verl
sf dataset push aiden_lu/qa-rl-verl v1 /tmp/qa_rl_verl --public

# 2. 校验 + 提交（模型须用支持工具调用 chat template 的 Instruct 版）
sf validate verl-grpo_qwen3.5-9b_qa-tools_v1
sf submit verl-grpo_qwen3.5-9b_qa-tools_v1 --profile h200:1 \
    --model Qwen/Qwen3.5-9B \
    --train-dataset aiden_lu/qa-rl-verl@v1 --train-data train.parquet \
    --validation-dataset aiden_lu/qa-rl-verl@v1 --validation-data val.parquet
```

## 常见坑

### 平台侧（verl adapter 的 `config_mode=hydra-overrides`）

1. **数据集/模型引用写进 `config.yaml`** → adapter 把整份 YAML 扁平化成 `key=value`
   传给 `verl.trainer.main_ppo`，只排除模型/数据/topology 那几个权威绑定键。
   `data.train.dataset` 不在排除表里，会原样变成 `data.train.dataset=...`，而 verl 的
   `data` 没有 `train` 子节点 → **Hydra struct 模式启动即报错**。
   verl 路线只能用 `--train-dataset` / `--validation-dataset` 提交时绑定。
   （nemo-rl 路线写在 config 里是对的，那边整份 YAML 就是训练配置——两条路线在这点上相反。）
2. **数据集里没有 parquet** → verl `RLHFDataset` 读 parquet，`--train-data train.parquet`
   指向不存在的文件会在建数据集时失败。`sf dataset ls` 只列名字，用
   `GET /api/datasets/<owner>/<name>` 看文件清单。
3. **`trainer.logger: [console]` 不用改**：平台 patch 了 `verl.utils.tracking.Tracking.log`
   取指标，console 就够，不需要也不该配 wandb/swanlab。

### verl 侧（官方 issue / 文档）

4. **`ground_truth` 的 `[type]` 前缀被当答案比对** → 题库答案是
   `"[single] A"` / `"[multiple] A,B"` / `"[fill] a ||| b"` / `"[short] kw1 ||| kw2"`，
   自己写精确匹配会让**正确答案也判 0**，组内奖励恒等 → GRPO 优势全为 0 →
   训练照跑、reward 曲线平得像直线但什么都没学到。`reward.py` 直接复用
   `common/rewards`（与 nemo-rl 对照实验同源，A/B 才可比），不要重写简化版。
5. **数据集行缺 `agent_name: "tool_agent"`** → 异步模式静默回落单轮、工具永不触发
   （issue #2986）。`prepare_data.py` 已注入；`config.yaml` 也设了 `default_agent_loop` 兜底。
6. **`rollout.mode` 不是 `async`** → Agent Loop 不生效。
7. **Base 模型 + hermes 格式** → chat template 不支持工具调用，全程不出 `<tool_call>`；
   用 Instruct 模型，或换 `format` 为模型模板支持的格式。
8. **`max_prompt_length` 按题面裸长度估** → hermes 会把工具 JSON schema 注入 system，
   本题库最长题面 1359 字符，1024 装不下；verl 默认 `truncation=error` 会直接抛异常。
   本实验给 2048 并开 `filter_overlong_prompts` 兜住离群题。
9. **`max_response_length` 按单轮估** → 多轮轨迹（生成 + 工具回灌）会被截断。
    本实验给 2048，与 `max_prompt_length` 2048 合成 4096，对齐 nemo-rl 侧的整条轨迹总预算。
10. 旧 `verl/interactions` 机制自 v0.8 起废弃，网上老教程的 `interaction_config_path` 不要用。
11. **只给 actor 的 micro batch** → `validate_config` 在 `use_dynamic_bsz=false` 时要求
    `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` 必填，且
    `use_kl_loss: true`（启用 reference policy）时 `actor_rollout_ref.ref` 也必填；
    少一个就在 hydra 解析后、训练开始前抛 `Please set at least one of …`。
    不带 `_per_gpu` 的旧键已废弃，两者同时给也会报错。
    这两处只做前向不反向，但 logits 很吃显存（batch × 4096 token × 151k 词表 × 2B，
    单卡下 8 就要 ~10G），别照抄「取 actor 的 2 倍」，单卡给 2。
12. **锁的框架版本必须是平台 catalog 支持的那个**（本实验锁 0.9.0）。0.9 移除了
    `main_ppo_sync`、统一入口 `main_ppo`（V1 trainer 默认、`trainer.v1.trainer_mode` 默认
    sync），入口由 recipe 的版本矩阵按 `framework.version` 自动选，不需要动 config。
    平台 catalog 升级后 recipe 摘要会变，`sf recipe status` 报 `recipe_stale` 时
    跑 `sf recipe upgrade` 刷新，否则提交会被完整性闸拒绝。

### 单卡（本实验 1×H200）专属

13. **`rollout.tensor_model_parallel_size` 默认是 2，不是 1** → 单卡提交时 vLLM 会去要
    第二张卡而起不来。必须在 config 里显式写 1。多卡场景反而不用管默认值。
14. **9B 全参数在一张 H200 上装不下** → 参数 bf16 18G + 梯度 bf16 18G + Adam fp32 72G +
    FSDP fp32 主权重 36G ≈ 144G > 141G，还没算 reference policy 与 vLLM。
    必须走 LoRA（`model.lora.rank > 0`；verl 默认 0 = 全参数）或 CPU offload
    （`actor.fsdp_config.{param_offload,optimizer_offload}`，默认都 false）。
    本实验用 LoRA——既为装得下，也因为 nemo-rl 对照侧就是 LoRA。
15. **配了 LoRA 却沿用全参数的 lr** → LoRA 需要高 1~2 个量级的 lr。
    本实验 1e-4；留着全参数口径的 1e-6 会 reward 曲线基本不动，看起来像「没学到」
    但其实是学习率问题，很难自查。
16. **`gpu_memory_utilization` 照抄多卡值** → 单卡上 base 权重 18G + ref policy 18G 常驻，
    0.7×141≈99G 会把卡挤爆。本实验给 0.5。
17. ★ **LoRA 写成嵌套 `model.lora.rank` 而不是扁平 `model.lora_rank`** → 用 fsdp/fsdp2 时
    **静默失效**、退回全参数微调。verl 0.9 的 `model` 配置里两套 LoRA 键并存
    （`workers/config/model.py` 自己标着 `TODO: unify fsdp and megatron lora config`）：

    | 后端 | 键 | 命名体系 |
    |---|---|---|
    | Megatron | 嵌套 `model.lora.{rank,alpha,dropout,target_modules}` | `linear_qkv` 等 |
    | **FSDP（本实验）** | 扁平 `model.{lora_rank,lora_alpha,target_modules}` | `q_proj` 等 HF 名 |

    `ppo_trainer.yaml` 的 `model` 段两套都列着、都不报错，所以写错不会有任何提示：
    单卡上表现为 OOM，多卡上则是安静地跑出一个全参数结果、和 LoRA 的 A/B 不可比。
    vLLM 侧（`vllm_async_server.py`）先读嵌套再回落扁平，所以只设扁平键就能贯通 rollout。
