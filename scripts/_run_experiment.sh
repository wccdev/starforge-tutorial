#!/usr/bin/env bash
# 实验启动·通用逻辑（NeMo-RL 0.6.0）——所有实验的 run.sh 都把通用部分收口到这里，
# 单一事实来源：改一次，所有实验生效。各实验 run.sh 只声明自己的差异（主要是 ENTRY），
# 然后 `exec bash scripts/_run_experiment.sh "${EXP_DIR}"`。
#
# 入参：$1 = 实验目录绝对路径（EXP_DIR）。
# 约定的可选环境变量（由各实验 run.sh / 中心化服务在集群侧注入）：
#   ENTRY            训练入口（不设则：本目录有 run.py 用之，否则 examples/run_grpo.py）
#   NEMO_RL_DIR      容器内 NeMo-RL 0.6.0 源码目录（必填）
#   CLUSTER_PROFILE  硬件 profile（不设则读实验自带 cluster 文件，再兜底 gb10-spark）
#   OUTPUT_ROOT      产物根目录（不设则落到 EXP_DIR/outputs）；RUN_USER 再做多人隔离
#   NRL_RUN_ID       单次训练 run id（中心化提交时注入）；产物落到 .../<实验名>/<run_id>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_output_paths.sh
source "${SCRIPT_DIR}/_output_paths.sh"

EXP_DIR="${1:?用法: _run_experiment.sh <实验目录绝对路径>（由各实验 run.sh 传入）}"
[[ -d "${EXP_DIR}" ]] || { echo "实验目录不存在: ${EXP_DIR}"; exit 1; }
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
EXP_NAME="$(basename "${EXP_DIR}")"

# 本地 NeMo-RL 0.6.0 源码目录（必填）
NEMO_RL_DIR="${NEMO_RL_DIR:?请设置 NEMO_RL_DIR 指向 NeMo-RL 0.6.0 源码目录}"

# 硬件 profile：默认读本实验绑定的集群（同目录 cluster 文件，可选 cluster/ 下 h100 | gb10-spark | h200）。
# 本实验超参（batch/seq/LoRA/并行度/显存）都是按该集群的卡调出来的，换卡通常要重调。
# 优先级：环境 CLUSTER_PROFILE（服务端注入 / --profile）> 自带 cluster 文件 > gb10-spark 兜底。
if [[ -z "${CLUSTER_PROFILE:-}" && -f "${EXP_DIR}/cluster" ]]; then
  CLUSTER_PROFILE="$(tr -d '[:space:]' < "${EXP_DIR}/cluster")"
fi
CLUSTER_PROFILE="${CLUSTER_PROFILE:-gb10-spark}"
CONFIG="${EXP_DIR}/config.yaml"                        # 继承基底 + 本实验差异
PROFILE_CONF="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/overrides.conf"
PROFILE_ENV="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/env.sh"

# 训练入口：实验 run.sh 显式 export ENTRY（SFT / 自定义示例）优先；否则本目录有 run.py 用它，
# 再否则用 GRPO 官方入口。
if [[ -z "${ENTRY:-}" ]]; then
  if [[ -f "${EXP_DIR}/run.py" ]]; then ENTRY="${EXP_DIR}/run.py"; else ENTRY="examples/run_grpo.py"; fi
fi

read_conf() { [[ -f "$1" ]] && grep -vE '^[[:space:]]*(#|$)' "$1" || true; }

# 集群/硬件 override（CLI，运行时按 profile 叠加）+ 产物落盘
OVERRIDES=()
while IFS= read -r l; do [[ -n "$l" ]] && OVERRIDES+=("$l"); done < <(read_conf "${PROFILE_CONF}")

# 权威拓扑：中心化服务按 profile 下发 LAB_CLUSTER_NUM_NODES/GPUS_PER_NODE（配额计量以此为准）。
# 有注入则剔除 overrides.conf 里的 cluster.num_nodes/gpus_per_node，改用服务端值——
# 保证「实际占卡 == 服务端记账」，用户改上传文件的卡数无法绕过配额。
# 本地直跑（无该环境变量）时行为不变，仍用 overrides.conf。
if [[ -n "${LAB_CLUSTER_NUM_NODES:-}" && -n "${LAB_CLUSTER_GPUS_PER_NODE:-}" ]]; then
  _kept=()
  for _o in ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do
    case "$_o" in
      cluster.num_nodes=*|cluster.gpus_per_node=*) ;;  # 丢弃文件里的拓扑
      *) _kept+=("$_o") ;;
    esac
  done
  OVERRIDES=(${_kept[@]+"${_kept[@]}"} \
    "cluster.num_nodes=${LAB_CLUSTER_NUM_NODES}" \
    "cluster.gpus_per_node=${LAB_CLUSTER_GPUS_PER_NODE}")
  echo "[run] topology(服务端权威): num_nodes=${LAB_CLUSTER_NUM_NODES} gpus_per_node=${LAB_CLUSTER_GPUS_PER_NODE}"
fi
# 产物（checkpoint + 每步样本 jsonl + 日志）落盘位置。
# 经服务端提交时 EXP_DIR 在 Ray 上传的临时包目录里（训练结束被清理、不回传本机），
# 故由服务端注入 OUTPUT_ROOT（集群持久路径/共享盘）后产物落到
#   OUTPUT_ROOT[/<用户>]/<实验名>/<NRL_RUN_ID>
# 多人共用平台时设 RUN_USER，同一实验多次提交按 run_id 隔离，互不覆盖。
OUT_DIR="$(_lab_train_output_dir "${EXP_NAME}" "${EXP_DIR}")"
OVERRIDES+=("checkpointing.checkpoint_dir=${OUT_DIR}")
OVERRIDES+=("logger.log_dir=${OUT_DIR}/logs")

echo "[run] exp     : ${EXP_NAME}"
echo "[run] out_dir : ${OUT_DIR}"
echo "[run] profile : ${CLUSTER_PROFILE}"
echo "[run] entry   : ${ENTRY}"
echo "[run] config  : ${CONFIG}"
# 可复现元数据（由 lab submit 注入；容器内直跑时为空）。落到作业日志，便于事后回查代码/配置版本。
echo "[run] version : run_id=${NRL_RUN_ID:-(直跑)} git=${NRL_GIT_COMMIT:-?}$([[ "${NRL_GIT_DIRTY:-0}" == 1 ]] && echo '+dirty') config=${NRL_CONFIG_SHA:-?}"
echo "[run] cluster/产物 overrides:"; printf '          %s\n' "${OVERRIDES[@]}"

# 集群侧预置密钥文件（容器内路径，由中心化服务注入 CLUSTER_SECRETS_FILE 并随作业转发其路径）。
# 配了它就不必把密钥明文塞进 runtime_env（不会暴露在 Ray dashboard）；密钥在此处 source 进作业进程。
if [[ -n "${CLUSTER_SECRETS_FILE:-}" && -f "${CLUSTER_SECRETS_FILE}" ]]; then
  set -a; source "${CLUSTER_SECRETS_FILE}"; set +a
  echo "[run] secrets : sourced ${CLUSTER_SECRETS_FILE}"
fi

# 硬件/网络 env（NCCL、Ray 内存、PyTorch 分配）；多节点须与 ray start 用同一份
[[ -f "${PROFILE_ENV}" ]] && source "${PROFILE_ENV}"

# 数据目录：未显式设置 *_DATA_DIR 时，默认指向本仓库 datasets/<name>。
# 经服务端提交时该目录随作业上传（仅排除 raw/data 缓存），
# 故 config 里的 ${oc.env:GSM8K_DATA_DIR} 等无需手填即可解析；
# 想用集群上已有的大数据，则由服务端注入同名变量覆盖（或在 config.yaml 写死 data_dir）。
for _ds in gsm8k:GSM8K_DATA_DIR alpaca:ALPACA_DATA_DIR qa_rl:QA_RL_DATA_DIR; do
  _name="${_ds%%:*}"; _var="${_ds##*:}"
  if [[ -z "${!_var:-}" && -d "${REPO_ROOT}/datasets/${_name}" ]]; then
    export "${_var}=${REPO_ROOT}/datasets/${_name}"
    echo "[run] ${_var}=${REPO_ROOT}/datasets/${_name} (默认指向仓库内数据)"
  fi
done

# 经 ray job submit 时，作业自带 runtime_env（working_dir + 转发的 env_vars）；NeMo-RL 的
# init_ray 还会再 ray.init(runtime_env=...) 传一份 env_vars，键重叠会被 Ray 判为冲突报错。
# 置 1 让 Ray 合并 Job 与 Driver 的 runtime_env（冲突以 Driver 为准，值相同无副作用；直跑无害）。
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1

# 训练入口经 nemolab_boot.py 包装：运行前给 NeMo-RL Logger 挂上 NeMoLabLogger 后端，
# 把训练指标 + 每卡硬件主动上报中心化 console（落库供前端展示，不依赖反向爬 Ray 日志）。
# 仅当 console 注入 NEMOLAB_TOKEN 时生效；本地直跑无该变量，boot 为透明 no-op，行为不变。
# boot 脚本在上传的 working_dir(REPO_ROOT) 内，按路径调用即可，无需在 NeMo-RL venv 里装包。
BOOT="${REPO_ROOT}/scripts/nemolab_boot.py"
cd "${NEMO_RL_DIR}"
exec uv run --no-sync python "${BOOT}" "${ENTRY}" --config "${CONFIG}" "${OVERRIDES[@]}"
