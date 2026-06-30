#!/usr/bin/env bash
# 训练后闭环（在【集群容器内】执行）：把 NeMo-RL 0.6.0 的 checkpoint 转成 HF 格式（export）、
# 或对 checkpoint 跑独立评测（eval）。由 `lab export` / `lab eval` 经 ray job submit 调起，
# 也可在 head 容器里直接 `bash scripts/post_train.sh ...` 跑。薄封装官方脚本（单一事实来源）：
#   - 转换： examples/converters/convert_dcp_to_hf.py（DTensor）/ convert_megatron_to_hf.py（Megatron, --extra mcore）
#   - 评测： examples/run_eval.py（仅吃 HF 格式模型）
#
# checkpoint 目录约定（与 scripts/_run_experiment.sh 落盘一致）：
#   <CKPT_ROOT>/step_<N>/{config.yaml, policy/weights[/iter_*], policy/tokenizer}
#   CKPT_ROOT 默认 = OUTPUT_ROOT[/<RUN_USER>]/<实验名>/<train_run_id>；
#   未设 OUTPUT_ROOT 时回退到 <仓库>/<exp>/outputs。
#
# 用法：
#   bash scripts/post_train.sh export <exp_rel> [--run-id RUN] [--step N] [--out DIR] [--push-repo user/name]
#   bash scripts/post_train.sh eval   <exp_rel> [--run-id RUN] [--step N] [--model HF_PATH] [--eval-config CFG] [-- 覆盖项...]
# 环境变量：NEMO_RL_DIR(必填)、OUTPUT_ROOT/RUN_USER/NRL_TRAIN_RUN_ID(定位 checkpoint)、HF_TOKEN、LAB_DRY_RUN=1。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_output_paths.sh
source "${SCRIPT_DIR}/_output_paths.sh"

ACTION="${1:?用法: post_train.sh <export|eval> <exp_rel> [flags]}"
EXP_REL="${2:?缺少实验相对路径，如 experiments/grpo_qwen3.5-9b_gsm8k_v1}"
shift 2
EXP_REL="${EXP_REL%/}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_DIR="${NEMO_RL_DIR:?请设置 NEMO_RL_DIR 指向容器内 NeMo-RL 0.6.0 源码目录}"
EXP_NAME="$(basename "${EXP_REL}")"

# ---------- 解析 flags ----------
STEP=""            # 空 = 自动取最新 step_<N>
OUT=""             # export 输出目录；空 = CKPT_ROOT/hf_export/step_<N>
PUSH_REPO=""       # 非空 = 转换后 huggingface-cli upload 到该 repo
MODEL=""           # eval 用：直接评测此 HF 模型路径/Hub id（给了就跳过 export）
EVAL_CONFIG="examples/configs/evals/eval.yaml"
CKPT_ROOT_OVERRIDE=""
TRAIN_RUN_ID="${NRL_TRAIN_RUN_ID:-}"
EVAL_OVERRIDES=()  # `--` 之后透传给 run_eval.py 的 NeMo-RL 覆盖项

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)        STEP="${2:?--step 需要数值}"; shift 2 ;;
    --run-id)      TRAIN_RUN_ID="${2:?--run-id 需要 lab_run_id}"; shift 2 ;;
    --out)         OUT="${2:?--out 需要目录}"; shift 2 ;;
    --push-repo)   PUSH_REPO="${2:?--push-repo 需要 user/name}"; shift 2 ;;
    --model)       MODEL="${2:?--model 需要路径或 Hub id}"; shift 2 ;;
    --eval-config) EVAL_CONFIG="${2:?--eval-config 需要配置路径}"; shift 2 ;;
    --ckpt-dir)    CKPT_ROOT_OVERRIDE="${2:?--ckpt-dir 需要目录}"; shift 2 ;;
    --)            shift; EVAL_OVERRIDES=("$@"); break ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

if [[ -n "${CKPT_ROOT_OVERRIDE}" ]]; then
  CKPT_ROOT="${CKPT_ROOT_OVERRIDE}"
else
  CKPT_ROOT="$(_lab_resolve_ckpt_root "${EXP_NAME}" "${REPO_ROOT}/${EXP_REL}" "${TRAIN_RUN_ID}")"
fi

# 干跑：打印命令而非执行（本地/集群均可用于检查路径与命令是否正确）。
DRY="${LAB_DRY_RUN:-0}"
run() {
  echo "› $*"
  [[ "${DRY}" == "1" ]] || "$@"
}

# ---------- 发现 step ----------
# 取 CKPT_ROOT 下的 step_<N>；--step 指定则用之，否则取 N 最大者（最新）。
resolve_step_dir() {
  if [[ -n "${STEP}" ]]; then
    echo "${CKPT_ROOT}/step_${STEP}"
    return
  fi
  local latest="" maxn=-1 d n
  for d in "${CKPT_ROOT}"/step_*; do
    [[ -d "$d" ]] || continue
    n="${d##*/step_}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if (( n > maxn )); then maxn="$n"; latest="$d"; fi
  done
  echo "${latest}"
}

STEP_DIR="$(resolve_step_dir)"
if [[ -z "${STEP_DIR}" || ! -d "${STEP_DIR}" ]]; then
  echo "找不到 checkpoint：${CKPT_ROOT}/step_<N>（CKPT_ROOT=${CKPT_ROOT}）"
  echo "  · 确认 OUTPUT_ROOT/RUN_USER/--run-id 与训练时一致；或用 --ckpt-dir / --step 指定。"
  [[ "${DRY}" == "1" ]] || exit 1
  STEP_DIR="${CKPT_ROOT}/step_<N>"   # 干跑下给个占位继续打印
fi
STEP_NUM="${STEP_DIR##*/step_}"
STEP_CONFIG="${STEP_DIR}/config.yaml"

# ---------- 后端检测（按文件系统，零依赖且权威）----------
# Megatron 把权重存成 policy/weights/iter_<NNNN...>；DTensor(DCP) 直接存在 policy/weights 下。
WEIGHTS_DIR="${STEP_DIR}/policy/weights"
BACKEND="dcp"
MCORE_WEIGHTS=""
if compgen -G "${WEIGHTS_DIR}/iter_*" > /dev/null 2>&1; then
  BACKEND="megatron"
  MCORE_WEIGHTS="$(ls -d "${WEIGHTS_DIR}"/iter_* 2>/dev/null | sort | tail -1)"
fi

echo "[post] action  : ${ACTION}"
echo "[post] exp     : ${EXP_REL}"
echo "[post] ckpt_root: ${CKPT_ROOT}"
echo "[post] ckpt    : ${STEP_DIR}  (step=${STEP_NUM}, backend=${BACKEND})"

# ---------- export：DCP/Megatron → HF ----------
do_export() {
  local out="${OUT:-${CKPT_ROOT}/hf_export/step_${STEP_NUM}}"
  echo "[post] hf_out  : ${out}"
  run mkdir -p "${out}"
  run cd "${NEMO_RL_DIR}"
  if [[ "${BACKEND}" == "megatron" ]]; then
    # Megatron 转换需要 mcore extra；权重路径指向 iter_* 目录。
    run uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
      --config "${STEP_CONFIG}" \
      --megatron-ckpt-path "${MCORE_WEIGHTS:-${WEIGHTS_DIR}/iter_0000000}" \
      --hf-ckpt-path "${out}"
  else
    run uv run python examples/converters/convert_dcp_to_hf.py \
      --config "${STEP_CONFIG}" \
      --dcp-ckpt-path "${WEIGHTS_DIR}" \
      --hf-ckpt-path "${out}"
  fi
  # 带上 tokenizer（HF 模型权重+tokenizer 放一起，便于直接 from_pretrained / 溯源）。
  if [[ -d "${STEP_DIR}/policy/tokenizer" ]]; then
    run rsync -a "${STEP_DIR}/policy/tokenizer/" "${out}/"
  else
    echo "[post] 警告: 未找到 ${STEP_DIR}/policy/tokenizer，转换后请自行补 tokenizer 文件。"
  fi
  # 可选：推送到 HuggingFace Hub（需 HF_TOKEN）。新版 CLI 用 `hf upload`（huggingface-cli 已废弃）。
  if [[ -n "${PUSH_REPO}" ]]; then
    [[ "${DRY}" == "1" ]] || : "${HF_TOKEN:?推送 Hub 需要 HF_TOKEN（由中心化服务在集群侧注入）}"
    run uv run hf upload "${PUSH_REPO}" "${out}" --repo-type model
    echo "[post] 已推送到 https://huggingface.co/${PUSH_REPO}"
  fi
  echo "[post] export 完成 → ${out}"
  EXPORTED_DIR="${out}"
}

# ---------- eval：对 HF 模型跑 run_eval.py ----------
do_eval() {
  local model="${MODEL}"
  if [[ -z "${model}" ]]; then
    # 未指定模型：先把当前 step 导成 HF，再评测它。
    echo "[post] eval 未指定 --model，先导出当前 checkpoint 为 HF 再评测。"
    do_export
    model="${EXPORTED_DIR}"
  fi
  echo "[post] eval model : ${model}"
  echo "[post] eval config: ${EVAL_CONFIG}"
  run cd "${NEMO_RL_DIR}"
  # `${arr[@]+"${arr[@]}"}`：兼容 set -u 下的空数组（含 macOS 自带 bash 3.2）。
  run uv run python examples/run_eval.py \
    --config "${EVAL_CONFIG}" \
    "generation.model_name=${model}" \
    ${EVAL_OVERRIDES[@]+"${EVAL_OVERRIDES[@]}"}
  echo "[post] eval 完成"
}

case "${ACTION}" in
  export) do_export ;;
  eval)   do_eval ;;
  *) echo "未知 action: ${ACTION}（支持 export | eval）"; exit 2 ;;
esac
