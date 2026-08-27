# sft_qwen3.5-4b_alpaca_v1

监督微调（SFT）示例：`qwen3.5-4b` 在 Alpaca 指令数据上做 SFT。和本仓的 GRPO / Agent 实验一起看即可。

## 目标

用标准 SFT 让模型学会遵循指令（instruction → response）。最常见的自定义微调范式，
演示「本地 jsonl + ResponseDataset + sft_processor」这条真实数据接入链路。

## 数据准备（先做）

```bash
# 在仓库根目录
python common/data/prepare_alpaca.py             # 写到 datasets/alpaca/{train,val}.jsonl
export ALPACA_DATA_DIR="$(pwd)/datasets/alpaca"   # 供 config.yaml 的 ${oc.env:ALPACA_DATA_DIR} 解析
```

每条样本格式：`{"input": 指令(可含上下文), "output": 目标回复}`。

## 组成

- `config.yaml` — 继承 `configs/base/sft.yaml` + `qwen3.5-4b`，`_override_` 替换 `data` 为
  `ResponseDataset` 指向上面的 jsonl，处理器 `sft_processor`；并把基底里 squad 专用的
  `chat_template` 置 null，改用模型默认对话模板。
- `recipe.lock.json` — 固定 sft recipe、入口与 digest。

## SFT vs GRPO

| | 本实验（SFT） | `grpo_qwen3.5-4b_gsm8k_v1`（GRPO） |
| --- | --- | --- |
| 学习信号 | 监督：模仿标注回复 | 强化：按奖励/正确性优化 |
| 是否需要环境 | 否 | 是（math 环境验证） |
| 入口 | `examples/run_sft.py` | `examples/run_grpo.py` |

## SwanLab

- project：`sft_qwen3.5-4b_alpaca_v1`，run：`lr5e6-bs32-ep3`，链接：<回填>

## 运行

```bash
sf submit sft_qwen3.5-4b_alpaca_v1
```

产物落到本目录 `outputs/`（已 .gitignore）。

## 结果与结论

- 关键指标：val:val_loss
- 结论 / 下一步：
