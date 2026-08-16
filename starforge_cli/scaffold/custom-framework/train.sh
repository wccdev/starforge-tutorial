#!/usr/bin/env bash
# 显式 custom recipe 的训练脚本。
#
# 由 starforge-sdk 的 CustomAdapter 直接执行。框架来自版本化 recipe，不读取
# FRAMEWORK 环境变量或 framework 文件，也不会由其他 adapter 失败后回退至此。
#
# ── 已经给你准备好的环境变量（契约，别改名）─────────────────────────────────────
#   FORGE_OUT_DIR                 产物目录（已 mkdir）。checkpoint / 日志请写这里，
#                               它已按 <用户>/<实验>/<run_id> 隔离好，不会和别人互相覆盖
#   FORGE_EXP_DIR                 本实验目录（就是这个文件所在目录）
#   FORGE_WORK_DIR                上传包根目录
#   FORGE_FRAMEWORK / FORGE_RECIPE  固定为 custom / custom
#   FORGE_CLUSTER_NUM_NODES       ★服务端权威拓扑：你实际能用几个节点
#   FORGE_CLUSTER_GPUS_PER_NODE   ★服务端权威拓扑：每节点几张卡
#   STARFORGE_ENDPOINT/RUN_ID/TOKEN   指标上报凭据（无则为本地直跑，上报自动 no-op）
#   HF_HOME / HF_TOKEN / 各 *_DATA_DIR   由服务端按需注入
#
# ── 三件必须做的事 ─────────────────────────────────────────────────────────────
#  ① checkpoint 写进 $FORGE_OUT_DIR —— 写别处的话作业结束就随临时目录一起没了
#  ② 用 common/observability/report.py 上报指标 —— 否则 console 上只有日志、没有曲线：
#        from common.observability.report import StarForgeCallback
#        trainer = SFTTrainer(..., callbacks=[StarForgeCallback()])
#     手写循环则用 report.init() / report.log({...}, step=i) / report.finish()
#  ③ 遵守 FORGE_CLUSTER_* 拓扑 —— 配额按它记账，watchdog 会做集群级 Ray 用卡对账，
#     实际占卡超出记账会被告警甚至停止作业
#
# ⚠️ 装不了新包：Ray 跑在 NeMo-RL 官方容器里，作业级 `uv run --with` 只影响 driver 进程，
#    训练 worker 用的仍是容器 venv。需要 TRL / flash-attn 这类新依赖，就得做 overlay 镜像
#    并让管理员用它重起 Ray 集群，见 cluster/README.md「overlay 镜像」一节。
set -euo pipefail

: "${FORGE_WORK_DIR:?FORGE_WORK_DIR is required}"
: "${FORGE_EXP_DIR:?FORGE_EXP_DIR is required}"
: "${FORGE_OUT_DIR:?FORGE_OUT_DIR is required}"
: "${FORGE_FRAMEWORK:?FORGE_FRAMEWORK is required}"
: "${FORGE_RECIPE:?FORGE_RECIPE is required}"
: "${FORGE_CLUSTER_NUM_NODES:?FORGE_CLUSTER_NUM_NODES is required}"
: "${FORGE_CLUSTER_GPUS_PER_NODE:?FORGE_CLUSTER_GPUS_PER_NODE is required}"

echo "[train] out_dir  : ${FORGE_OUT_DIR}"
echo "[train] topology : ${FORGE_CLUSTER_NUM_NODES:-?} 节点 × ${FORGE_CLUSTER_GPUS_PER_NODE:-?} 卡"

# ── TODO: 换成你自己的启动命令 ────────────────────────────────────────────────
# 示例（HF accelerate，单节点多卡）：
#
#   exec accelerate launch \
#       --num_processes "${FORGE_CLUSTER_GPUS_PER_NODE:-1}" \
#       "${FORGE_EXP_DIR}/train.py" \
#       --output_dir "${FORGE_OUT_DIR}" \
#       --config "${FORGE_EXP_DIR}/config.yaml"
#
# 示例（直接跑一个 python 脚本）：
#
#   exec python "${FORGE_EXP_DIR}/train.py" --output-dir "${FORGE_OUT_DIR}"

echo "[train] 还没填训练命令：请编辑 ${FORGE_EXP_DIR}/train.sh" >&2
exit 1
