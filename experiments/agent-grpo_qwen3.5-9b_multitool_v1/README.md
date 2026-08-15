# agent-grpo_qwen3.5-9b_multitool_v1

多轮多工具 Agent GRPO 练习实验。改编自官方 `run_grpo_sliding_puzzle.py`。

## 目标

让 `qwen3.5-9b` 学会**多轮调用多个工具**完成需要外部能力的任务：检索资料（search）、
做算术（calc）、跑代码（python），最后给出正确数值答案。跑通 NeMo-RL 0.6.0 的多轮
Agent / 自定义环境 / 多工具链路。

## 任务类型（run.py 随机生成，均为数值可验证）

| 类型 | 示例问题 | 需要的工具 |
| --- | --- | --- |
| calc | 计算 12 + 7 * 3 | `calc` |
| search_calc | 买 5 个苹果一共多少钱（KB 含单价 + 干扰项） | `search` → `calc` |
| code | 求 1..15 的平方和 | `python` |

## 工具协议

| 动作 | 模型输出 | 环境响应 |
| --- | --- | --- |
| 计算 | `<tool>calc: 2+3*4</tool>` | `[calc] 14` |
| 检索 | `<tool>search: 苹果 单价</tool>` | `[search] 苹果的单价是 8 元` |
| 代码 | `<tool>python: print(sum(i*i for i in range(1,6)))</tool>` | `[python] 55` |
| 答题 | `<answer>14</answer>` | 校验对错，对则奖励 1.0 并结束 |

停止串 `</tool>` / `</answer>` 让模型每轮在动作处停下，交给环境推进。

## 组成

- `config.yaml` — 继承 `grpo_sliding_puzzle`（多轮基底）+ `qwen3.5-9b` + `grpo_megatron`(Megatron+低显存调优) + `grpo_lora`(LoRA)，`_override_` 替换 env。
- `run.py` — 自定义训练脚本：生成三类任务 + KB，实例化 `ToolAgentEnv`，调 `grpo_train`。
- `method` / `recipe.lock.json` — 显式固定 recipe 与入口，不按文件存在性探测框架。

## 关键超参（低显存实测起点）

- 模型：`Qwen/Qwen3.5-9B-Base`（与你实测跑通的 9B 数学同款，HF_HOME 已缓存，免再下大文件）。
- 后端：Megatron-Core + **LoRA**（lr 1e-4/dim8/cosine）。回全参数：删 `defaults` 里 `grpo_lora.yaml`。
- 生成：**非 colocated**（1 卡生成 / 1 卡训练），需占满 2 卡。
- batch：`num_prompts_per_step=4` / `num_generations_per_prompt=8` / `train_global_batch_size=32` / `micro=1`。
- **多轮上下文比单轮长**：`seq=2048` 起（单轮 9B 实测 1250）。OOM 就降到 1536/1024，或减小 `num_generations_per_prompt` / `max_rollout_turns`。

## SwanLab

- project：`agent-grpo_qwen3.5-9b_multitool_v1`，run：`lora-turns6-g8-lr1e4`，链接：<回填>
- 答对率：`validation/accuracy`（验证集）、`train/reward`（训练 rollout 平均奖励）。
  注：环境奖励是 答对=1.0/答错=0.0，所以"奖励均值"=答对率。环境里 `global_post_process_and_metrics`
  返回的 `tool_agent_success_rate` 在当前 NeMo-RL 的 GRPO 流程里不会被记录，别去 SwanLab 找它。
- 走偏诊断：`train/natural_termination_rate`（正常给 `<answer>` 收尾的比例，越高越好）、
  `train/truncation_rate`（超长截断比例，越低越好）、`train/avg_turns_per_sample`（平均轮数）、
  `train/max_turns_reached_rate`（用满 6 轮没答出的比例）、`train/baseline_reward/pct_1`（整组全对比例）。

## 运行

```bash
lab submit agent-grpo_qwen3.5-9b_multitool_v1
```

> ⚠️ `python` 工具会执行模型生成的代码，请只在隔离的 NeMo-RL 训练容器内运行。

## 扩展成你的真实工具

1. 在 `common/environments/example_tool_env.py` 的 `TOOLS` 加工具（如真实检索 API、SQL、HTTP）。
2. 改 `run.py` 的 `_make_task` / `PROMPT_HEADER` 换成你的任务、数据与知识库。
3. 调 `config.yaml` 的 `max_rollout_turns`、`env.tool_agent.cfg`、超参。
