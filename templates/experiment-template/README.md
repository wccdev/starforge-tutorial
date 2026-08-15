# <method>_<model>_<dataset>_<tag>

> 复制本模板新建实验：`lab new <新实验名> --method <framework>/<method>`
> 实验名遵循 `docs/naming-convention.md`。

## 目标

一句话说明这个实验要验证 / 达成什么。

## 配置（NeMo-RL 0.6.0，配置继承）

- `config.yaml` 通过 `defaults` 继承基底（`configs/base/`）+ 模型片段（`configs/models/`），
  **只写本实验差异**；不断调参就改 `config.yaml` 的「本实验差异」部分。
- 训练入口、指标与产物契约由 SDK 中的版本化 recipe 声明；实验目录不写 `framework`
  标记，也不按 `run.py`/`train.sh` 是否存在猜测框架。
- 硬件 profile：`h100 | h200 | h200-2g`（提交时 `lab submit --profile` 指定；env/overrides 由 Console 服务端注册表下发）。

## 监控

- project：`<实验名>`
- run：`<超参组合>`
- 链接：<贴上监控面板链接>

## 运行

```bash
uv run lab submit <实验名> --pool train:h100:1:1
```

产物（checkpoint / 日志）落到本目录 `outputs/`（已 .gitignore）。

## 结果与结论

- 关键指标：
- 结论 / 下一步：
