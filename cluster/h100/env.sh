# 单机 1× H100 (80GB) 环境。由统一集群 launcher 加载。
# 单机单卡没有跨节点通信，故不配 RoCE/IB/多网卡那一套——
# 那是 gb10-spark/env.sh（2 节点）才需要的；在单机上指定网卡名只会误绑不存在的接口。

# --- PyTorch 显存分配（缓解碎片；须与 vLLM 兼容）---
# GRPO 默认用 vLLM 做 rollout（VllmGenerationWorker / CuMem memory pool）。
# expandable_segments:True 与 vLLM 内存池互斥，会直接 AssertionError（见 vllm/device_allocator/cumem.py）。
# 故 H100 与 gb10-spark 对齐：max_split_size + GC 阈值，不用 expandable_segments。
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

# --- Megatron 推荐：固定单条 CUDA 流连接，保证 kernel 顺序与数值可复现 ---
# 单卡 TP=1 时无 TP 通信，留着无害；NeMo/Megatron 容器默认即此值。
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Ray 本地实例内存监控（单机 host RAM 足够时放宽，避免训练进程被 OOM killer 误杀）---
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=2000

# --- NCCL：单卡通信平凡，只留日志级别 ---
# 不要设 NCCL_SOCKET_IFNAME / NCCL_IB_* —— 单机会因网卡名不匹配而初始化失败。
export NCCL_DEBUG=WARN
