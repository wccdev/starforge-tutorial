#!/usr/bin/env bash
# 实验启动·通用逻辑（NeMo-RL 0.7.0）——所有实验的 run.sh 都把通用部分收口到这里，
# 单一事实来源：改一次，所有实验生效。各实验 run.sh 只声明自己的差异（主要是 ENTRY），
# 然后 `exec bash scripts/_run_experiment.sh "${EXP_DIR}"`。
#
# 入参：$1 = 实验目录绝对路径（EXP_DIR）。
# 约定的可选环境变量（由各实验 run.sh / 中心化服务在集群侧注入）：
#   ENTRY            训练入口（不设则：本目录有 run.py 用之，否则 examples/run_grpo.py）
#   NEMO_RL_DIR      容器内 NeMo-RL 0.7.0 源码目录（FRAMEWORK=nemo-rl 时必填）
#   CLUSTER_PROFILE  硬件 profile（不设则读实验自带 cluster 文件，再兜底 gb10-spark）
#   OUTPUT_ROOT      产物根目录（不设则落到 EXP_DIR/outputs）；RUN_USER 再做多人隔离
#   NRL_RUN_ID       单次训练 run id（中心化提交时注入）；产物落到 .../<实验名>/<run_id>
#   FRAMEWORK        nemo-rl（默认）| custom —— 见下方「自定义框架」一节
#   LAB_DRY_RUN      置 1 只打印最终命令、不真正执行（供单测与排错用）
#
# ── 自定义框架（FRAMEWORK=custom）──────────────────────────────────────────────
# 用于跑 NeMo-RL 以外的训练代码（TRL / verl / OpenRLHF / 纯 HF Trainer / 自己写的循环）。
# 实验目录里放一个可执行的 train.sh，本脚本把「集群侧那些脏活」都替它办好之后再 exec 它：
#   已 source 好 密钥文件 + profile env.sh；已算好产物目录并 export LAB_OUT_DIR；
#   已 export LAB_EXP_DIR / LAB_REPO_ROOT / LAB_CLUSTER_NUM_NODES / LAB_CLUSTER_GPUS_PER_NODE
#   / NEMOLAB_*（指标上报凭据）/ 各数据集目录变量。
# train.sh 里请：① 把 checkpoint 写进 $LAB_OUT_DIR；② 用 common/observability/report.py
# 把指标打进 console（否则 console 上这个作业只有日志没有曲线）；③ 遵守 LAB_CLUSTER_* 拓扑。
# ⚠️ 配额不靠自觉：服务端按 profile 注册表记账，watchdog 还会做集群级 Ray 用卡对账，
#    实际占卡超出记账会被告警/停止。custom 只是放开「跑什么代码」，不是放开「占多少卡」。
# ⚠️ 新依赖装不进来：Ray 跑在 NeMo-RL 官方容器里，作业级 `uv run --with` 只影响 driver 进程，
#    训练 worker 用的仍是容器 venv。要加包只能做 overlay 镜像，见 cluster/README.md。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_output_paths.sh
source "${SCRIPT_DIR}/_output_paths.sh"

EXP_DIR="${1:?用法: _run_experiment.sh <实验目录绝对路径>（由各实验 run.sh 传入）}"
[[ -d "${EXP_DIR}" ]] || { echo "实验目录不存在: ${EXP_DIR}"; exit 1; }
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
EXP_NAME="$(basename "${EXP_DIR}")"

# 训练框架：环境 FRAMEWORK（服务端/run.sh 注入）> 实验自带 framework 文件 > nemo-rl 兜底。
# 与 cluster 文件同款约定：一行文本，跟着实验走，fork 实验时自动继承。
if [[ -z "${FRAMEWORK:-}" && -f "${EXP_DIR}/framework" ]]; then
  FRAMEWORK="$(tr -d '[:space:]' < "${EXP_DIR}/framework")"
fi
FRAMEWORK="${FRAMEWORK:-nemo-rl}"
case "${FRAMEWORK}" in
  nemo-rl|custom) ;;
  *) echo "未知 FRAMEWORK=${FRAMEWORK}（可选：nemo-rl | custom）"; exit 1 ;;
esac

# 容器内 NeMo-RL 0.7.0 源码目录。custom 框架不经 NeMo-RL 启动，故不强制。
if [[ "${FRAMEWORK}" == "nemo-rl" ]]; then
  NEMO_RL_DIR="${NEMO_RL_DIR:?请设置 NEMO_RL_DIR 指向 NeMo-RL 0.7.0 源码目录}"
fi

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
echo "[run] frame   : ${FRAMEWORK}"
if [[ "${FRAMEWORK}" == "nemo-rl" ]]; then
  echo "[run] entry   : ${ENTRY}"
  echo "[run] config  : ${CONFIG}"
fi
# 可复现元数据（由 lab submit 注入；容器内直跑时为空）。落到作业日志，便于事后回查代码/配置版本。
echo "[run] version : run_id=${NRL_RUN_ID:-(直跑)} git=${NRL_GIT_COMMIT:-?}$([[ "${NRL_GIT_DIRTY:-0}" == 1 ]] && echo '+dirty') config=${NRL_CONFIG_SHA:-?}"
if [[ "${FRAMEWORK}" == "nemo-rl" ]]; then
  echo "[run] cluster/产物 overrides:"; printf '          %s\n' "${OVERRIDES[@]}"
fi

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
for _ds in gsm8k:GSM8K_DATA_DIR alpaca:ALPACA_DATA_DIR qa_rl:QA_RL_DATA_DIR opsd_math:OPSD_DATA_DIR; do
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

# ── 自定义框架分支 ──────────────────────────────────────────────────────────────
# 到这里为止，集群侧的脏活（密钥、profile env、产物目录、数据目录）已经全部办完，
# 且它们与训练框架无关。所以 custom 只需要把这些结果 export 出去，然后把控制权交给
# 实验自己的 train.sh —— 不 cd 进 NEMO_RL_DIR，也不拼 NeMo-RL 的 --config/override。
if [[ "${FRAMEWORK}" == "custom" ]]; then
  TRAIN_SH="${EXP_DIR}/train.sh"
  if [[ ! -f "${TRAIN_SH}" ]]; then
    echo "FRAMEWORK=custom 需要实验目录下有 train.sh：${TRAIN_SH}" >&2
    echo "（参考 templates/custom-framework/train.sh）" >&2
    exit 1
  fi
  # 交给 train.sh 的契约（写进 README，别改名）
  export LAB_OUT_DIR="${OUT_DIR}"
  export LAB_EXP_DIR="${EXP_DIR}"
  export LAB_EXP_NAME="${EXP_NAME}"
  export LAB_REPO_ROOT="${REPO_ROOT}"
  export LAB_CLUSTER_PROFILE="${CLUSTER_PROFILE}"
  # PYTHONPATH 指到仓库根，train.sh 里才能 `from common.observability import report`
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  mkdir -p "${OUT_DIR}"
  echo "[run] custom  : ${TRAIN_SH}"
  echo "[run] 契约    : LAB_OUT_DIR / LAB_EXP_DIR / LAB_REPO_ROOT / LAB_CLUSTER_{NUM_NODES,GPUS_PER_NODE}"
  if [[ "${LAB_DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] exec bash ${TRAIN_SH}"
    exit 0
  fi
  exec bash "${TRAIN_SH}"
fi

# ── NeMo-RL 分支（默认）────────────────────────────────────────────────────────
# 训练入口经 nemolab_boot.py 包装：运行前给 NeMo-RL Logger 挂上 NeMoLabLogger 后端，
# 把训练指标 + 每卡硬件主动上报中心化 console（落库供前端展示，不依赖反向爬 Ray 日志）。
# 仅当 console 注入 NEMOLAB_TOKEN 时生效；本地直跑无该变量，boot 为透明 no-op，行为不变。
# boot 脚本在上传的 working_dir(REPO_ROOT) 内，按路径调用即可，无需在 NeMo-RL venv 里装包。
BOOT="${REPO_ROOT}/scripts/nemolab_boot.py"
# 干跑先于 cd：排错时容器外也能看命令，不必先有一个真的 NEMO_RL_DIR。
if [[ "${LAB_DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] cd ${NEMO_RL_DIR} && uv run --no-sync python ${BOOT} ${ENTRY} --config ${CONFIG} ${OVERRIDES[*]}"
  exit 0
fi
cd "${NEMO_RL_DIR}"
exec uv run --no-sync python "${BOOT}" "${ENTRY}" --config "${CONFIG}" "${OVERRIDES[@]}"
