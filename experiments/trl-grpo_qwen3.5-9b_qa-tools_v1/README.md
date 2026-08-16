# TRL GRPO 工具调用示例：题库 + 本地文档检索

与 nemo-rl 的 `grpo_qwen3.5-9b_qa-rl-agent_v3`、verl 的 `verl-grpo_qwen3.5-9b_qa-tools_v1`
是**同一场景的第三种框架实现**：模型答题前可多轮调用检索工具查本地资料
（`DOCS_DIR`，默认 `/data/docs`）再作答，判分只看 `\boxed{}` 最终答案。

## 三条路线对照

| | nemo-rl | verl | TRL（本实验） |
|---|---|---|---|
| 多轮循环 | 自定义 Environment + `max_rollout_turns` | Agent Loop（`tool_agent`），config 声明 | `GRPOTrainer(tools=[...])` 内置工具循环 |
| 环境交互 | `QADocsAgentEnv.step()` 解析 `<search>` | `@function_tool` 文件 + `function_tool_path` | 普通 Python 函数直接传 `tools` |
| 判分 | Environment 内调 `common/rewards` | `custom_reward_function.path` 独立文件 | `reward_funcs` 可调用对象 |
| 配置 | config.yaml（defaults 继承树） | config.yaml（hydra 覆盖树） | config.yaml（平铺 GRPOConfig 键值） |
| 入口 | 实验 `run.py`（recipe experiment_override） | 官方 module，无自定义入口 | 实验 `train.py`（TRL recipe 的固定契约） |

TRL 是三者中最「代码优先」的：工具、奖励、数据构造都在 `train.py` 里，
config 只承载 GRPOConfig 超参。这正是 TRL 官方形态（框架没有配置化注入机制）。

## 文件说明

- `train.py` — 全部逻辑：`search_docs` 工具（类型注解 + Google docstring，schema 由此推断）、
  `qa_boxed_reward`（conversational completion 取最后一条 assistant 消息判分）、
  题库 jsonl/parquet → conversational prompt 的数据构造
- `config.yaml` — 平铺 GRPOConfig 键值（trl-yaml 直传构造器，`--set` 超参覆盖其上）
- `eval.py` — recipe lifecycle eval 钩子

## 跑起来

```bash
# 数据无需预处理：直接用与 nemo-rl 对照实验同源的题库 jsonl
forge validate trl-grpo_qwen3.5-9b_qa-tools_v1
forge submit trl-grpo_qwen3.5-9b_qa-tools_v1 \
    --model Qwen/Qwen3.5-9B \
    --train-data <集群路径>/datasets/qa_rl/train.jsonl \
    --validation-data <集群路径>/datasets/qa_rl/val.jsonl
```

## 常见坑（来自 TRL 官方文档）

1. **工具循环要求 chat template「前缀保持」**（追加 tool 消息不能改变前文渲染）。
   Qwen3 / DeepSeek-V3 等已知家族 TRL 会自动换补丁模板；模型用 **Instruct** 版，
   Base 版不会产出工具调用。
2. **数据集必须是 conversational 格式**（prompt 列 = 消息列表），tool 消息才能回灌；
   `train.py` 的 `_dataset()` 已做转换。
3. **奖励函数签名用 `**kwargs` 兜底**；数据集额外列（本实验的 `answer`）按列名传入。
4. **多轮预算**：所有轮生成 + 工具回灌都占 `max_completion_length`，按单轮估会截断
   （本实验给 3072）。
5. 需要按轨迹状态（沙箱/临时目录）或环境自产任务时，改用 `environment_factory`
   （环境公开方法自动成为工具，`reset()` 产任务、`get_reward()` 拥有奖励；
   需 transformers>=5.2）；数据集驱动的题库场景用 `tools` 即可。
