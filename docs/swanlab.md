# SwanLab 接入（NeMo-RL 0.6.0）

NeMo-RL 0.6.0 **原生支持 SwanLab** logger（与 WandB / TensorBoard / MLflow 并列），无需自己写日志代码。所有训练日志统一上传云端 SwanLab，便于跨实验、跨硬件对比。

## 1. 登录

```bash
pip install swanlab
swanlab login        # 粘贴 API Key
# 或用环境变量
export SWANFORGE_API_KEY=xxxxxxxx
```

> 在本仓库工作流里，`SWANFORGE_API_KEY`（以及 `HF_TOKEN` / `HF_ENDPOINT` 国内镜像等）由**中心化 Lab 服务**
> 持有，并在集群侧注入到作业进程，本机不入库、也无需任何 `submit.env`。`sf submit` 时服务端自动转发。
> 上面的 `swanlab login` / `export SWANFORGE_API_KEY` 仅用于本机临时调试。

## 2. 在配置 / override 里启用

配置中的 logger 段（官方默认 `swanlab_enabled: false`）：

```yaml
logger:
  log_dir: "logs"
  wandb_enabled: false
  tensorboard_enabled: false
  swanlab_enabled: true          # 打开 SwanLab
  monitor_gpus: true             # GPU 利用率也上报到 SwanLab
  swanlab:
    project: "grpo_qwen3.5-9b_gsm8k_v2"   # 对齐实验目录名
    name: "lr1e6-g16-kl0.01"              # 对齐关键超参
```

用 CLI override 等价写法（由 NeMoRLAdapter 编译为独立 argv）：

```bash
uv run python examples/run_grpo.py --config <base.yaml> \
  logger.swanlab_enabled=true \
  logger.swanlab.project=grpo_qwen3.5-9b_gsm8k_v2 \
  logger.swanlab.name=lr1e6-g16-kl0.01 \
  logger.monitor_gpus=true
```

## 3. 命名对齐（重要）

| SwanLab 概念 | NeMo-RL key | 取值 |
| --- | --- | --- |
| project | `logger.swanlab.project` | 实验目录名（或按模型聚合，如 `qwen3.5-9b`） |
| run | `logger.swanlab.name` | 关键超参组合，如 `lr1e6-g16-kl0.01-h200` |

不同硬件跑同一实验时，在 `name` 后缀 `-h100` / `-h200`，便于对比吞吐与收敛。

## 4. 回填链接

每个实验的 `README.md` 必须贴上对应 SwanLab 链接，做到「代码 ↔ 看板」一一对应。
