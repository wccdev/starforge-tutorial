# configs/base — 供继承的基底配置

这里是 **NeMo-RL v0.7.0 官方 example 配置的原样副本**（version-locked），作为所有实验
`defaults` 继承的「祖父配置」。实验只写差异，调参不动这里。

| 文件 | 来源（v0.7.0） | 用途 |
| --- | --- | --- |
| `grpo_math_1B.yaml` | `examples/configs/grpo_math_1B.yaml` | GRPO 基底（最常用的祖父） |
| `sft.yaml` | `examples/configs/sft.yaml` | SFT 基底 |
| `grpo_sliding_puzzle.yaml` | `examples/configs/grpo_sliding_puzzle.yaml` | 多轮 Agent 基底（本身 `defaults: grpo_math_1B.yaml`） |

> 官方就是这么用继承的：`grpo_sliding_puzzle.yaml` 第一行即 `defaults: "grpo_math_1B.yaml"`，
> 只覆盖 `grpo.max_rollout_turns`、`env` 等差异。

## 更新基底（升级 NeMo-RL 版本时）

```bash
NEMO_RL_DIR=/path/to/NeMo-RL bash scripts/sync_base_configs.sh
```

会从你本地的 NeMo-RL 源码把 example 配置同步过来。**不要手改这些文件**——要调参请在
实验的 `config.yaml` 里覆盖。
