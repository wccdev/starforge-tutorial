#!/usr/bin/env bash
# 在【集群容器内】预下载 HuggingFace 模型到 HF_HOME，避免训练时在线拉取失败。
#
# 典型原因：设了 HF_ENDPOINT=hf-mirror.com，但容器连不上 mirror；
# 或集群无外网。解决：unset 镜像，在容器里先跑本脚本，再 forge submit。
#
# 用法（在 NeMo-RL 容器内，先导出所需环境变量）：
#   export HF_TOKEN=...   HF_HOME=/path/to/hf_cache
#   bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B
#   bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-9B
set -euo pipefail

MODEL_ID="${1:?用法: prefetch_hf_model.sh <HF_REPO_ID>，如 Qwen/Qwen3.5-4B}"

HF_HOME="${HF_HOME:-/home/aidenlu/nemo-rl-work/hf_cache}"
export HF_HOME
mkdir -p "${HF_HOME}"

# 集群下载不要用镜像（连不上 + 大文件 308 跳官方站）
unset HF_ENDPOINT

echo "[prefetch] 模型   : ${MODEL_ID}"
echo "[prefetch] HF_HOME: ${HF_HOME}"
echo "[prefetch] HF_ENDPOINT: ${HF_ENDPOINT:-<未设置，走 huggingface.co>}"

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" --token "${HF_TOKEN:-}"
elif [[ -n "${NEMO_RL_DIR:-}" && -d "${NEMO_RL_DIR}" ]]; then
  cd "${NEMO_RL_DIR}"
  uv run python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    '${MODEL_ID}',
    cache_dir=os.environ['HF_HOME'],
    token=os.environ.get('HF_TOKEN') or None,
)
print('完成:', '${MODEL_ID}')
"
else
  python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    '${MODEL_ID}',
    cache_dir=os.environ['HF_HOME'],
    token=os.environ.get('HF_TOKEN') or None,
)
print('完成:', '${MODEL_ID}')
"
fi

echo "[prefetch] 完成。确认缓存："
ls -la "${HF_HOME}/hub" 2>/dev/null | tail -5 || ls -la "${HF_HOME}" | tail -5
