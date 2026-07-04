# <method>_<model>_<dataset>_<tag>

> 复制本模板新建实验：`bash scripts/new_experiment.sh experiments <新实验名>`
> 实验名遵循 `docs/naming-convention.md`。

## 目标

一句话说明这个实验要验证 / 达成什么。

## 配置（NeMo-RL 0.6.0，配置继承）

- `config.yaml` 通过 `defaults` 继承基底（`configs/base/`）+ 模型片段（`configs/models/`），
  **只写本实验差异**；不断调参就改 `config.yaml` 的「本实验差异」部分。
- 通用启动逻辑都在 `scripts/_run_experiment.sh`（单一事实来源）；本实验 `run.sh` 只声明差异。
  入口：GRPO 默认无需改；SFT 取消 `run.sh` 里 `ENTRY` 那行注释；自定义环境写本目录 `run.py`（自动选用）。见 `configs/README.md` 方法对照表。
- 硬件 profile：`h100 | gb10-spark | h200`（实验目录 `cluster` 文件绑定；`cluster/<profile>/overrides.conf` 运行时叠加）。

## SwanLab

- project：`<实验名>`
- run：`<超参组合>`
- 链接：<贴上 SwanLab 链接>

## 运行

```bash
NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
```

产物（checkpoint / 日志）落到本目录 `outputs/`（已 .gitignore）。

## 结果与结论

- 关键指标：
- 结论 / 下一步：
