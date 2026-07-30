# common/environments — 自定义环境（奖励来源）

NeMo-RL 里 GRPO 的奖励由 **Environment** 产生（而非独立 reward 函数）。把跨实验复用的
自定义环境放这里：

- 数学/通用单轮任务：通常用内置环境（配置 `data.default.env_name=math` 等），无需自写。
- 多轮 Agent / 工具调用：实现自定义 Environment + 自定义 run 脚本。

## `example_tool_env.py` — 可运行的工具调用环境

一个**计算器工具调用**多轮环境，照官方 `nemo_rl/environments/games/sliding_puzzle.py`
结构写成，可直接训练。导出：

- `ToolAgentEnv`：`@ray.remote` 的 `EnvironmentInterface`，实现 `step()` 返回 6 字段
  `EnvironmentReturn(observations, metadata, next_stop_strings, rewards, terminateds, answers)`。
- `TOOLS`：工具注册表（`name -> callable(arg)->str`），加工具就往这里加。
- `safe_eval`：安全算术求值（给计算器工具与答案校验用）。

配套用法见 `experiments/agent-grpo_qwen3.5-9b_calc-tool_v1/`（含自定义 `run.py`：生成
`DatumSpec` 任务 + 构建 `task_to_env` + `grpo_train`）。

> 实现要点：环境是 Ray actor；多轮训练需要**自定义 run 脚本**来喂数据和环境（纯改配置不够）；
> `step()` 的返回结构、`DatumSpec` 字段以你装的 0.6.0 源码为准（参考 `nemo_rl/environments/` 内置环境）。

## reward shaping 只属于训练——验证要单独建环境

带工具的环境常给「调用工具」本身发奖励（引导模型别闭卷瞎猜）。这类 shaping **不能进验证**：
NeMo-RL 的 `validation/accuracy` 就是 `mean(total_reward)`，而 `total_reward` 是**逐轮奖励的累加**
（`nemo_rl/experience/rollouts.py`），所以验证一旦带 shaping，「用了工具」本身就白送分 → 验证分虚高，
且与不带工具的 baseline 不再同尺度，A/B 结论失真。

做法：`grpo_train()` 的第 7、8 个参数是 `task_to_env` / `val_task_to_env`，**给验证传另一个实例**，
其 cfg 把 shaping 归零（`qa_docs_agent_env.make_eval_cfg()` 就是干这个的），检索与判分保持一致：

```python
train_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=env_cfg)
val_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=make_eval_cfg(env_cfg))
grpo_train(..., {TASK: train_env}, {TASK: val_env}, ...)
```
