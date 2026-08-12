#!/usr/bin/env bash
# JobSpec 作业的集群侧入口（阶段 1）。
#
# 与 _run_experiment.sh 的分工：
#   本脚本      只做 shell 真正擅长的事——把密钥与 profile 环境变量 source 进进程环境。
#   launcher.py 做全部**决策**：选入口、叠 override、算产物目录、装载算法插件。
#
# 改造前这两件事都挤在 _run_experiment.sh 里用 bash 表达，服务端看不见任何决策，
# 加一种后训练方法就要改这段 shell。现在加方法只需加一个 recipe 目录。
#
# 老客户端（无 JobSpec）继续走 _run_experiment.sh，两条路并存。
set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORK_DIR}"
export LAB_WORK_DIR="${WORK_DIR}"

# 集群侧预置密钥文件（服务端只转发路径，密钥不进 Ray dashboard）。
if [[ -n "${CLUSTER_SECRETS_FILE:-}" && -f "${CLUSTER_SECRETS_FILE}" ]]; then
  set -a; source "${CLUSTER_SECRETS_FILE}"; set +a
  echo "[launch] secrets : sourced ${CLUSTER_SECRETS_FILE}"
fi

# 硬件/网络 env（NCCL、Ray 内存、PyTorch 分配）；多节点须与 ray start 用同一份。
PROFILE_ENV="${WORK_DIR}/cluster/${CLUSTER_PROFILE:-}/env.sh"
if [[ -n "${CLUSTER_PROFILE:-}" && -f "${PROFILE_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${PROFILE_ENV}"
  echo "[launch] env     : sourced ${PROFILE_ENV}"
fi

# 数据目录：与 _run_experiment.sh 同约定——未显式设置时指向仓库内 datasets/<name>。
for _d in "${WORK_DIR}"/datasets/*/; do
  [[ -d "${_d}" ]] || continue
  _name="$(basename "${_d}")"
  _var="$(echo "${_name}" | tr '[:lower:]-' '[:upper:]_')_DATA_DIR"
  if [[ -z "${!_var:-}" && -f "${_d}train.jsonl" ]]; then
    export "${_var}=${_d%/}"
  fi
done

# 入口由 nemo-lab-sdk 提供（镜像里已装），不再依赖上传包内的 Python 包。
exec python -m nemo_lab_sdk.launcher "$@"
