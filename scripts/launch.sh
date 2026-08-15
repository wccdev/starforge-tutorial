#!/usr/bin/env bash
# JobSpec 作业的集群侧入口。
#
# 本脚本只做 shell 真正擅长的事——把密钥与 profile 环境变量 source 进进程环境。
# launcher.py 经 FrameworkAdapter 做全部决策：选入口、叠 override、算产物目录。
#
# 旧实现曾把这些决策放在 shell；现在服务端和 SDK 共享同一份显式契约，
# 加一种后训练方法就要改这段 shell。现在加方法只需加一个 recipe 目录。
#
set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORK_DIR}"
export LAB_WORK_DIR="${WORK_DIR}"

# 集群侧预置密钥文件（服务端只转发路径，密钥不进 Ray dashboard）。
if [[ -n "${CLUSTER_SECRETS_FILE:-}" && -f "${CLUSTER_SECRETS_FILE}" ]]; then
  set -a; source "${CLUSTER_SECRETS_FILE}"; set +a
  echo "[launch] secrets : sourced ${CLUSTER_SECRETS_FILE}"
elif [[ -n "${CLUSTER_SECRETS_FILE:-}" ]]; then
  echo "[launch] configured CLUSTER_SECRETS_FILE does not exist: ${CLUSTER_SECRETS_FILE}" >&2
  exit 2
fi

# 硬件/网络 env（NCCL、Ray 内存、PyTorch 分配）；多节点须与 ray start 用同一份。
PROFILE_ENV="${WORK_DIR}/cluster/${CLUSTER_PROFILE:-}/env.sh"
if [[ -n "${CLUSTER_PROFILE:-}" && -f "${PROFILE_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${PROFILE_ENV}"
  echo "[launch] env     : sourced ${PROFILE_ENV}"
fi

# 入口由 nemo-lab-sdk 提供（镜像里已装），不再依赖上传包内的 Python 包。
exec python -m nemo_lab_sdk.launcher "$@"
