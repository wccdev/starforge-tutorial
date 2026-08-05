#!/usr/bin/env bash
# 自定义框架训练脚本（FRAMEWORK=custom）—— 跑 NeMo-RL 以外的训练代码。
#
# 由 scripts/_run_experiment.sh 在集群侧 exec 本文件。到这里为止，那些「每个人都要重写一遍」
# 的脏活已经办完：密钥已 source、profile env.sh 已 source、产物目录已建好、数据目录变量已导出。
# 你只需要关心「怎么起你的训练」。
#
# ── 已经给你准备好的环境变量（契约，别改名）─────────────────────────────────────
#   LAB_OUT_DIR                 产物目录（已 mkdir）。checkpoint / 日志请写这里，
#                               它已按 <用户>/<实验>/<run_id> 隔离好，不会和别人互相覆盖
#   LAB_EXP_DIR                 本实验目录（就是这个文件所在目录）
#   LAB_REPO_ROOT               仓库根（已加进 PYTHONPATH）
#   LAB_EXP_NAME / LAB_CLUSTER_PROFILE
#   LAB_CLUSTER_NUM_NODES       ★服务端权威拓扑：你实际能用几个节点
#   LAB_CLUSTER_GPUS_PER_NODE   ★服务端权威拓扑：每节点几张卡
#   NEMOLAB_ENDPOINT/RUN_ID/TOKEN   指标上报凭据（无则为本地直跑，上报自动 no-op）
#   HF_HOME / HF_TOKEN / 各 *_DATA_DIR   由服务端按需注入
#
# ── 三件必须做的事 ─────────────────────────────────────────────────────────────
#  ① checkpoint 写进 $LAB_OUT_DIR —— 写别处的话作业结束就随临时目录一起没了
#  ② 用 common/observability/report.py 上报指标 —— 否则 console 上只有日志、没有曲线：
#        from common.observability.report import NeMoLabCallback
#        trainer = SFTTrainer(..., callbacks=[NeMoLabCallback()])
#     手写循环则用 report.init() / report.log({...}, step=i) / report.finish()
#  ③ 遵守 LAB_CLUSTER_* 拓扑 —— 配额按它记账，watchdog 会做集群级 Ray 用卡对账，
#     实际占卡超出记账会被告警甚至停止作业
#
# ⚠️ 装不了新包：Ray 跑在 NeMo-RL 官方容器里，作业级 `uv run --with` 只影响 driver 进程，
#    训练 worker 用的仍是容器 venv。需要 TRL / flash-attn 这类新依赖，就得做 overlay 镜像
#    并让管理员用它重起 Ray 集群，见 cluster/README.md「overlay 镜像」一节。
set -euo pipefail

echo "[train] out_dir  : ${LAB_OUT_DIR}"
echo "[train] topology : ${LAB_CLUSTER_NUM_NODES:-?} 节点 × ${LAB_CLUSTER_GPUS_PER_NODE:-?} 卡"

# ── TODO: 换成你自己的启动命令 ────────────────────────────────────────────────
# 示例（HF accelerate，单节点多卡）：
#
#   exec accelerate launch \
#       --num_processes "${LAB_CLUSTER_GPUS_PER_NODE:-1}" \
#       "${LAB_EXP_DIR}/train.py" \
#       --output_dir "${LAB_OUT_DIR}" \
#       --config "${LAB_EXP_DIR}/config.yaml"
#
# 示例（直接跑一个 python 脚本）：
#
#   exec python "${LAB_EXP_DIR}/train.py" --output-dir "${LAB_OUT_DIR}"

echo "[train] 还没填训练命令：请编辑 ${LAB_EXP_DIR}/train.sh" >&2
exit 1
