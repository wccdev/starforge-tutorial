#!/usr/bin/env bash
# 实验启动脚本（NeMo-RL 0.6.0）。通用逻辑都在 scripts/_run_experiment.sh（单一事实来源）；
# 本文件只声明【本实验差异】。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 训练入口（按方法选其一）：
#   - GRPO（默认）：什么都不写——本目录有 run.py 自动用之，否则 examples/run_grpo.py
#   - SFT         ：取消下一行注释
# export ENTRY="${ENTRY:-examples/run_sft.py}"
#   - 自定义环境（多轮 Agent 等）：在本目录写 run.py，会被自动选用（无需设 ENTRY）

exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
