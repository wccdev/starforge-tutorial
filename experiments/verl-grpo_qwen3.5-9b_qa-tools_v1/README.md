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
| 入口 | 实验自带 `run.py` | 官方 `verl.trainer.main_ppo_sync`，无自定义入口 |

## 文件说明

- `config.yaml` — Agent Loop / 多轮 / 工具 / 奖励全部在此声明（详见文件内注释）
- `tools.py` — `@function_tool` 检索工具；schema 由类型注解 + Google docstring 自动推断
- `reward.py` — `compute_score(data_source, solution_str, ground_truth, extra_info)` 最终判分
- `prepare_data.py` — 题库 jsonl → parquet，**每行注入 `agent_name: "tool_agent"`（关键）**
- `eval.py` — recipe lifecycle eval 钩子（模板标配）

## 跑起来

```bash
# 1. 数据（本地一次性）：与 nemo-rl 对照实验共用同一份题库 jsonl
python experiments/verl-grpo_qwen3.5-9b_qa-tools_v1/prepare_data.py \
    --data-dir /path/to/datasets/qa_rl --out-dir /path/to/datasets/qa_rl_parquet

# 2. 校验 + 提交（模型须用支持工具调用 chat template 的 Instruct 版）
forge validate verl-grpo_qwen3.5-9b_qa-tools_v1
forge submit verl-grpo_qwen3.5-9b_qa-tools_v1 \
    --model Qwen/Qwen3.5-9B \
    --train-data <集群路径>/qa_rl_parquet/train.parquet \
    --validation-data <集群路径>/qa_rl_parquet/val.parquet
```

## 常见坑（均来自 verl 官方 issue / 文档）

1. **数据集行缺 `agent_name: "tool_agent"`** → 异步模式静默回落单轮、工具永不触发
   （issue #2986）。`prepare_data.py` 已注入；`config.yaml` 也设了 `default_agent_loop` 兜底。
2. **`rollout.mode` 不是 `async`** → Agent Loop 不生效。
3. **Base 模型 + hermes 格式** → chat template 不支持工具调用，全程不出 `<tool_call>`；
   用 Instruct 模型，或换 `format` 为模型模板支持的格式。
4. **`max_response_length` 按单轮估** → 多轮轨迹（生成 + 工具回灌）会被截断；本实验给 3072。
5. v0.8 已废弃旧 `verl/interactions` 机制，网上老教程的 `interaction_config_path` 不要用。
