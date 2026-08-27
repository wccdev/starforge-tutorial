# common/environments

GRPO rewards come from a NeMo-RL **Environment**, not a standalone reward function. Shared custom envs live here.

- Single-turn math: usually a builtin (`data.default.env_name=math`). No file needed.
- Multi-turn / tools: custom Environment + a custom `run.py`.

## `example_tool_env.py`

Calculator tool-calling env, same shape as `nemo_rl/environments/games/sliding_puzzle.py`. Exports `ToolAgentEnv` (Ray actor, `EnvironmentReturn` with six fields), a `TOOLS` registry, and `safe_eval`.

There is no `calc-tool` experiment in this repo. For a submitable multi-turn job see `experiments/agent-grpo_qwen3.5-9b_sliding-puzzle_v1` (upstream puzzle) or `experiments/grpo_qwen3.5-9b_qa-rl-agent_v3` (search + QA).

`step()` / `DatumSpec` fields follow the NeMo-RL version in the training image (catalog is 0.7.0).

## Reward shaping is train-only

A tool env often pays for "called a tool". That must not go into validation: `validation/accuracy` is `mean(total_reward)` and `total_reward` sums per-turn rewards. Shaping on val inflates the score and breaks A/B vs a no-tool baseline.

`grpo_train()` takes `task_to_env` and `val_task_to_env`. Pass a second instance whose cfg zeros shaping (`qa_docs_agent_env.make_eval_cfg()`):

```python
train_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=env_cfg)
val_env = QADocsAgentEnv.options(num_gpus=0).remote(cfg=make_eval_cfg(env_cfg))
grpo_train(..., {TASK: train_env}, {TASK: val_env}, ...)
```
