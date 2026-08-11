"""PlatformBinding —— 服务端在提交时刻注入的那一半。

职责边界（这是整个契约包的关键设计）
──────────────────────────────────────────────────────────────────────────────
  JobSpec          用户**想跑什么**：算法、模型、数据、资源诉求、超参
  PlatformBinding  平台**决定怎么跑**：集群路径、产物根、密钥、卡型 pin、权威拓扑、
                   上报凭据、以及平台分配的 run_id

两者都不包含「怎么把它们变成环境变量」——那是 env.spec_to_env() 唯一的职责。

这样切分之后，`server/services/submit.py` 只负责「从服务端配置里取值填 binding」，
不再同时承担映射规则；映射规则被 golden 测试锁死，两个仓库共用。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IngestBinding:
    """训练侧回传指标/日志到 console 的凭据（NeMoLabLogger 用）。"""

    endpoint: str
    token: str


@dataclass(frozen=True)
class PlatformBinding:
    """服务端注入侧。所有字段都应是**已解析完毕的最终值**，不含任何待求值逻辑。

    尤其注意 `inline_secrets`：调用方（服务端）负责读密钥文件并剔除按策略不该下发的项
    （例如启用 HF 集成后要剔除共享 HF_TOKEN，改用用户个人 token）。契约层不读文件、
    不做策略判断，只做映射——否则密钥策略会分裂成两处。
    """

    run_id: str
    user: str

    #: 生效的硬件 profile（客户端指定 > 服务端默认）。
    profile: str = ""

    #: 容器内 NeMo-RL 源码目录。framework=nemo-rl 时必填。
    nemo_rl_dir: str = ""

    #: 集群持久化产物根。为空则产物落到实验目录下（本地直跑行为）。
    output_root: str = ""

    #: 卡型 pin 用的 Ray 自定义资源 key（异构集群精确调度）。
    pin_resource: str = ""

    #: 服务端权威拓扑 (num_nodes, gpus_per_node)。集群侧用它覆盖上传文件里的拓扑，
    #: 保证「实际占卡 == 服务端记账」。
    topology: tuple[int, int] | None = None

    #: profile 级附加环境变量（NCCL/内存等）。以 setdefault 语义注入，不覆盖关键变量。
    profile_env: dict[str, str] = field(default_factory=dict)

    #: 非密钥透传项。
    passthrough_env: dict[str, str] = field(default_factory=dict)

    #: 集群侧预置密钥文件路径。设了它就不必把密钥明文塞进 runtime_env（不进 Ray dashboard）。
    cluster_secrets_file: str = ""

    #: 明文密钥（仅在未配置 cluster_secrets_file 时使用）。调用方须已按策略过滤。
    inline_secrets: dict[str, str] = field(default_factory=dict)

    #: 用户个人 HuggingFace token。
    hf_token: str = ""

    #: 指标上报凭据。None 表示不启用上报。
    ingest: IngestBinding | None = None

    #: 训练后作业（export/eval）关联的训练 run_id。
    train_run_id: str = ""
