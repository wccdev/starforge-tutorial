# 单机 8× H200 (141GB) 环境。被实验 run.sh 在集群侧统一 source。
# 单节点 8 卡走 NVLink/NVSwitch，节点内通信无需 RoCE/IB；不要设 NCCL_IB_* / 网卡名，
# 否则会误绑不存在的接口（那是 gb10-spark 多节点才需要的）。

# --- PyTorch 显存分配（缓解碎片；须与 vLLM 内存池兼容，勿用 expandable_segments）---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

# --- Megatron：固定单条 CUDA 流连接，保证 kernel 顺序与数值可复现 ---
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Ray 本地实例内存监控（host RAM 足够时放宽，避免训练进程被 OOM killer 误杀）---
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=2000

# --- NCCL：节点内 NVLink 通信，只留日志级别 ---
export NCCL_DEBUG=WARN
