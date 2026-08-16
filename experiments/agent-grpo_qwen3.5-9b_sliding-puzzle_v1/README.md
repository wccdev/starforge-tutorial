# agent-grpo_qwen3.5-9b_sliding-puzzle_v1

单机 1×H100 上的**最简多轮 Agent GRPO 示例**。用 NeMo-RL 自带的「滑块拼图」环境，把多轮 RL 链路先跑通。

## 为什么用滑块拼图

多轮 Agent 示例里它最省事，最适合"先跑通"：

- **多轮**：模型每轮输出一步移动，环境返回新棋面，循环到拼好或用尽步数。
- **奖励自动判定**：拼好=1，不用接裁判 LLM。
- **零外部依赖**：不接 RAGFlow / 工具服务器 / 裁判端点（对比同目录 `qa-rl-agent`、`multitool`）。
- **环境与入口都在 NeMo-RL 里**：直接用自带 `examples/run_grpo_sliding_puzzle.py`，本目录无需写 `run.py`。

## 目标集群

`cluster` 文件 = `h100`（单机 1×H100 80GB）。`forge submit` 默认用它；`--profile` 可临时换卡（换卡通常要重调显存类超参）。

## 组成

- `config.yaml` — 继承 `grpo_sliding_puzzle`（多轮拼图基底）+ `qwen3.5-9b` + `grpo_megatron`（Megatron 后端，单卡 **colocated**）+ `grpo_lora`（**LoRA**，单卡跑 9B 的关键），只写差异。
- `recipe.lock.json` — 固定 recipe、入口和 digest；profile 与产物由统一 launcher 处理。

## 调什么（调参面）

打开 `config.yaml` 顶部「调参速查」，下面【① 调参区】就是要动的几行：

| 旋钮 | 作用 | 典型范围 / 调过头 |
| --- | --- | --- |
| `gpu_memory_utilization` | 单卡 colocated 下 vLLM 占显存比例 | 0.3~0.5；OOM 降、富余升 |
| `max_total_sequence_length` | 上下文长度（多轮会累积） | 1024；OOM 先降这个到 768 |
| `num_generations_per_prompt` | 每题采样数（组内基线） | 8；多→梯度稳但更慢更吃显存 |
| `max_rollout_turns` | 多轮上限（拼图越大需越多轮） | 12（2x2）；3x3 要加大 |
| `game_config.size` / `shuffle_moves` | 拼图难度 | size 2→3 变难；shuffle 越大越难 |
| `lr` / `dim` / `alpha` | LoRA 步长与容量（来自 `grpo_lora` 基底） | lr1e-4 / dim8 / alpha16 |
| `reference_policy_kl_penalty` | 贴原模型力度 | 0~0.05 |
| `max_num_steps` / `val_period` | 训多久 / 多久验证 | 先 50 步跑通，再调大 |

> 硬件/分布式（节点数、并行度）由服务端 h100 profile 注册表下发，不在本 config。

## 单卡关键约束

- **必须 colocated**（vLLM 与训练分时复用这一张卡）：本实验未引入 `grpo_noncolocated`，即为 colocated（别加那行，单卡没有第二张卡给生成）。
- **vLLM TP=1**：服务端 h100 profile 的 overrides 已设。
- **用 LoRA**：全参数 9B 在单张 80GB 上 colocated 几乎必 OOM。

## 运行

```bash
# 提交到集群（经中心化服务，全程在本机；先 forge login 接入服务）
uv run forge submit agent-grpo_qwen3.5-9b_sliding-puzzle_v1
uv run forge logs <job_id>               # 实时日志（不给 job_id 跟随最近一个）
```

> 前提：H100 容器里已装 NeMo-RL 0.6.0，且 `Qwen/Qwen3.5-9B-Base` 已缓存到 `HF_HOME`（或集群能直连 huggingface.co）。

## SwanLab

- project：`agent-grpo_qwen3.5-9b_sliding-puzzle_v1`，run：`h100-lora-2x2-g8`。
- 看 `train/reward`（rollout 平均奖励，因奖励=拼好/没拼好，约等于成功率）、`validation/accuracy`。
- 走偏诊断：`train/natural_termination_rate`、`train/truncation_rate`、`train/avg_turns_per_sample`。
