# configs/models — 各基础模型的公共片段

每个基础模型一个 partial 配置（只含该模型通用字段，如 `policy.model_name`、tokenizer、
该模型适配的序列长度 / 并行 / 显存策略）。**不含 `defaults`**，作为实验「多继承」的一项使用。

实验 `config.yaml` 里：

```yaml
defaults:
  - ../../configs/base/grpo_math_1B.yaml   # 方法基底（官方 v0.6.0）
  - ../../configs/models/qwen3.5-9b.yaml   # 模型片段（覆盖基底里的模型字段）
# 下面再写本实验差异（数据集 / lr / kl / swanlab ...）
```

多继承中**后面的覆盖前面的**，实验自身的顶层键再覆盖两者。新增模型就加一个
`<model>.yaml`，命名与 `docs/naming-convention.md` 的 model 字段一致。
