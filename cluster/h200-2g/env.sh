# 单机切 2× H200 环境。由统一集群 launcher 加载。
# 与 h200 同节点内 NVLink，不设 NCCL_IB_* / 网卡名。

# --- PyTorch 显存分配（缓解碎片；须与 vLLM 内存池兼容，勿用 expandable_segments）---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

# --- Megatron：固定单条 CUDA 流连接，保证 kernel 顺序与数值可复现 ---
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Ray 本地实例内存监控（host RAM 足够时放宽，避免训练进程被 OOM killer 误杀）---
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=2000

# --- NCCL：节点内 NVLink 通信，只留日志级别 ---
export NCCL_DEBUG=WARN
