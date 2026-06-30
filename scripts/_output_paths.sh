#!/usr/bin/env bash
# 产物路径规则（_run_experiment.sh / post_train.sh 共用，与 server.services.submit.build_output_dir 一致）
#
# 中心化提交（设 OUTPUT_ROOT）：
#   训练：  <OUTPUT_ROOT>/<RUN_USER>/<EXP_NAME>/<NRL_RUN_ID>
#   本地直跑（无 OUTPUT_ROOT）：<exp_dir>/outputs（不拆 run 子目录）
#
# post_train 用 NRL_TRAIN_RUN_ID（或 --run-id）定位训练产物；未指定时兼容旧版扁平目录。

_lab_exp_output_base() {
  local exp_name="$1"
  if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    echo "${OUTPUT_ROOT%/}${RUN_USER:+/${RUN_USER}}/${exp_name}"
  fi
}

# 训练落盘目录（_run_experiment.sh）
_lab_train_output_dir() {
  local exp_name="$1"
  local exp_dir="$2"
  if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    local base
    base="$(_lab_exp_output_base "${exp_name}")"
    if [[ -n "${NRL_RUN_ID:-}" ]]; then
      echo "${base}/${NRL_RUN_ID}"
    else
      echo "${base}"
    fi
  else
    echo "${exp_dir}/outputs"
  fi
}

# export/eval 定位 checkpoint 根目录（post_train.sh）
_lab_resolve_ckpt_root() {
  local exp_name="$1"
  local exp_dir="$2"
  local train_run_id="${3:-${NRL_TRAIN_RUN_ID:-}}"

  if [[ -z "${OUTPUT_ROOT:-}" ]]; then
    echo "${exp_dir}/outputs"
    return
  fi

  local base
  base="$(_lab_exp_output_base "${exp_name}")"

  if [[ -n "${train_run_id}" ]]; then
    echo "${base}/${train_run_id}"
    return
  fi

  # 旧版：checkpoint 直接在 <base>/step_*（run 级隔离升级前）
  if compgen -G "${base}/step_*" > /dev/null 2>&1; then
    echo "${base}"
    return
  fi

  # 自动：取含 step_* 的最新 run 子目录（按 mtime）
  local best="" best_mtime=0 d mt
  for d in "${base}"/*/; do
    [[ -d "${d}" ]] || continue
    if ! compgen -G "${d}step_*" > /dev/null 2>&1; then
      continue
    fi
    if mt=$(stat -c %Y "${d}" 2>/dev/null); then
      :
    else
      mt=$(stat -f %m "${d}" 2>/dev/null || echo 0)
    fi
    if (( mt > best_mtime )); then
      best_mtime=$mt
      best="${d%/}"
    fi
  done
  if [[ -n "${best}" ]]; then
    echo "${best}"
    return
  fi

  echo "${base}"
}
