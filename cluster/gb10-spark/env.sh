# GB10 / DGX Spark 集群环境（实测跑通的配置）。
# 被各实验 run.sh 在集群侧统一 source（Ray 集群由管理员/服务端起时也应 source 同一份）。
# 一处修改，训练启动即生效。网卡名 / HCA 按你机器实际填。

# --- 网络 / NCCL（跨节点通信，RoCE） ---
# export NCCL_SOCKET_IFNAME=enp1s0f0np0   # 有固定 IP 的管理/数据网卡（NCCL socket 用）
# export GLOO_SOCKET_IFNAME=enP7s7        # gloo 用的网卡
# export NCCL_IB_HCA=rocep1s0f0           # RoCE HCA
# export NCCL_IB_GID_INDEX=3              # RoCE v2 必备
# export NCCL_IB_DISABLE=0                # 开启 IB/RoCE（=1 则纯 socket）
# export NCCL_NET_GDR_LEVEL=PHB           # GPUDirect RDMA 级别
# export NCCL_P2P_DISABLE=1               # 绕开跨机 P2P / CUDA IPC
# export CUDA_DISABLE_P2P=1

# --- 调试 / 并发 ---
# export NCCL_DEBUG=WARN
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Ray 内存监控（OOM killer 阈值，在 ray start 处生效） ---
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=2000

# --- PyTorch 显存分配（缓解碎片，训练时生效） ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
