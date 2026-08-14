#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# The cluster must provide these two tiny parquet fixtures explicitly.
uv run lab submit smoke/verl-sft \
  --profile h100 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-data /data/nemo-lab/smoke/gsm8k/train.parquet \
  --validation-data /data/nemo-lab/smoke/gsm8k/test.parquet \
  --pool trainer:h100:1:1 \
  --set max_num_epochs=1 \
  --set train_batch_size=2 \
  --set max_length=512
