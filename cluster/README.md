# cluster/ — 硬件 / 分布式 profile

NeMo-RL 0.6.0 的集群设置（`cluster.num_nodes`、`cluster.gpus_per_node`）和并行度都在训练配置里，通过 **CLI override** 调整。本目录把「不同硬件的 override」抽出来复用，训练配置与硬件解耦。

每个 profile 一个子目录，核心是 `overrides.conf`（每行一个 `key=value`，`#` 为注释）：

- `h100/` — 单机 1× H100 80GB（单节点单卡，远程微调平台主力）
- `gb10-spark/` — 2× DGX Spark GB10（Ray 2 节点）
- `h200/` — 单机 8× H200 141GB（异构集群新增卡型）

## 用法

每个实验**自带目标集群**（实验目录下一行 `cluster` 文件，记录 profile 名）。`run.sh` 默认读它选 profile，自动把对应 `overrides.conf` 追加到训练命令；`CLUSTER_PROFILE` 环境变量 / `lab submit --profile` 可临时覆盖：

```bash
bash run.sh                              # 用实验自带 cluster（软绑定）
CLUSTER_PROFILE=h100 bash run.sh         # 临时换单机单卡
```

> **profile 优先级**：`--profile`（显式）> 实验自带 `cluster` 文件 > `gb10-spark` 兜底。
> 新建实验时用 `lab new <名字> --cluster h100` 写好绑定；`lab new <名字> --from <实验>` 会继承来源实验的绑定。

提交一律经**中心化 Lab 服务**：本机不读任何 `submit.env`，Ray 地址 / 密钥 / 产物根目录（`OUTPUT_ROOT`）/
数据目录 / 裁判 LLM 等都由服务端在集群侧注入。`--profile` 只决定用哪份 `overrides.conf` + `env.sh`，
随作业一起上传，在集群侧由 `scripts/_run_experiment.sh` 叠加。

等价于（集群侧实际执行）：

```bash
uv run python examples/run_grpo.py --config <base.yaml> \
    cluster.num_nodes=2 cluster.gpus_per_node=1 ...   # 来自 profile 的 overrides.conf
```

## 各 profile 包含

- `overrides.conf` — 节点数 / 每节点 GPU / 并行度等 NeMo-RL 覆盖项（CLI override）
- `env.sh` — 集群 env（NCCL/RoCE 网络 + Ray 内存监控 + PyTorch 显存分配），被实验 `run.sh` 在集群侧 source

> `overrides.conf` 走 CLI override（进 NeMo-RL 配置）；`env.sh` 走进程环境变量（NCCL/Ray/PyTorch 这类不属于训练配置的开关）。两者互补。

> **⚠️ 拓扑以服务端为权威（集中提交时）**：`cluster.num_nodes` / `cluster.gpus_per_node` 决定占几张卡、也决定配额计量。经中心化服务提交时，这两项由**服务端 profile 注册表**权威下发（`LAB_CLUSTER_NUM_NODES/GPUS_PER_NODE`），并在集群侧 `_run_experiment.sh` 里**覆盖** `overrides.conf` 的对应行——保证「实际占卡 == 服务端记账」，改本地文件的卡数不会影响集中提交的占卡与配额。要调整集中提交的拓扑，请让管理员改服务端注册表（`LAB_CLUSTER_PROFILES`）。`overrides.conf` 里的这两行只在**本地直跑**（无服务端注入）时生效。experiment 级的并行/调参（TP/PP、colocated 等）仍归研究员、放实验 config。

## 集群与 Ray

集群的搭建与 Ray 起停由**集群管理员 / 中心化服务**负责，提交者无需关心：`lab submit` 把作业交给服务端，
服务端连到已就绪的 Ray 集群代理执行。多节点的网卡 / HCA / IB 等网络参数维护在各 profile 的 `env.sh`
（当前 `gb10-spark/env.sh` 是两台 Spark 的实测值）。

## 依赖与环境

本仓库以 **NVIDIA NeMo-RL 0.7.0** 为训练框架；NeMo-RL 用 **`uv`** 管理依赖与运行（`uv run python ...`）。依赖按**架构**分别维护，强烈建议用 **NeMo-RL 官方容器镜像**跑训练，避免手装 CUDA / wheel 踩坑。

| 硬件 profile | 架构 | 安装方式 |
| --- | --- | --- |
| `gb10-spark` | aarch64 + Blackwell | aarch64 容器 / wheel |
| `h100` / `h200` | x86_64 + Hopper | x86_64 容器 / wheel |

### 集群容器内：NeMo-RL 0.7.0

训练框架**预装在容器镜像里**，不随作业上传。容器内的 NeMo-RL 路径（`NEMO_RL_DIR`，必须是**容器内**绝对路径）由中心化服务在集群侧注入。

若需自行克隆 / 升级 NeMo-RL（容器内执行）：

```bash
git clone --branch v0.7.0 https://github.com/NVIDIA-NeMo/RL.git NeMo-RL
cd NeMo-RL
uv sync
uv run python examples/run_grpo.py --help   # 验证
```

> 使用 `nvcr.io/nvidia/nemo-rl:v0.7.0`（CUDA 13）；各 backend（DTensor / Megatron）的额外依赖以 v0.7.0 官方文档为准。
> 源码与容器 fingerprint 不一致时，见 NeMo-RL 文档的 `NRL_FORCE_REBUILD_VENVS` / 重建镜像说明。

### overlay 镜像：想加新依赖时（内网集群必读）

**Ray 本身就跑在 NeMo-RL 官方容器里**（`uv run --no-sync ray start`），所以「容器 = 集群的环境」。
由此推出三条硬约束，先认清再动手，能省掉一整天的试错：

1. **作业级装包基本无效。** `uv run --with xxx` 只影响 driver 进程；训练真正跑在 Ray actor 里，
   用的是容器 venv，看不到 driver 的 overlay 环境。Ray `runtime_env.pip` 又会和 NeMo-RL 自己设的
   worker runtime_env 打架。**新依赖只能进镜像。**
2. **没有 per-job 镜像。** Ray 集群是长驻的，换镜像 = 重起 Ray 集群。所以 overlay 镜像是
   「集群级决策」，不是「实验级决策」——需要管理员配合，别指望自助。
3. **NeMo-RL 会在作业运行时现建 worker venv**（`nemo_rl/utils/venvs.py`：按 worker 类型在
   `$NEMO_RL_DIR/venvs/` 下 `uv sync`）。这一步**要访问 PyPI 索引**。内网集群若没配 Nexus 代理，
   第一次跑新 backend（vLLM / Megatron）就会卡在这里。两个解法：镜像里预建好这些 venv，
   或者配好内网索引（见下）。

#### 内网索引

在容器里（或服务端 `LAB_PASSTHROUGH_ENV`）配好，`uv sync` / `pip install` 才能在内网工作：

```bash
export UV_DEFAULT_INDEX=http://nexus.<内网域名>/repository/pypi/simple
export PIP_INDEX_URL=http://nexus.<内网域名>/repository/pypi/simple
export HF_ENDPOINT=http://nexus.<内网域名>/repository/hf-proxy   # 模型走内网
```

> 服务端已有 `passthrough_env` 透传机制，把上面几条写进 `LAB_PASSTHROUGH_ENV` 即可对所有作业生效，
> 无需改一行代码。

#### overlay Dockerfile 骨架

```dockerfile
FROM nvcr.io/nvidia/nemo-rl:v0.7.0

ARG NEXUS=http://nexus.<内网域名>/repository/pypi/simple
ENV UV_DEFAULT_INDEX=${NEXUS} PIP_INDEX_URL=${NEXUS}

# ★ 必须装进 NeMo-RL 那个 venv —— 装到系统 python 里，训练进程根本看不到
ENV NEMO_RL_DIR=/opt/nemo-rl
RUN uv pip install --python ${NEMO_RL_DIR}/.venv/bin/python \
        "trl==0.2x.x" "flash-attn==2.8.3" --no-build-isolation

# ★ 强烈建议：把 worker venv 预建进镜像，作业运行时就不必联网 uv sync
RUN cd ${NEMO_RL_DIR} && NRL_FORCE_REBUILD_VENVS=true uv run python -c "print('warm')"
```

**架构必须对上**：`h100` / `h200` 是 x86_64，`gb10-spark` 是 aarch64 + Blackwell，两套镜像分开构建。
`flash-attn` 这类要编译的包在 aarch64 上基本是坑，规划 L3 实验时优先放 H100/H200。

#### 跑非 NeMo-RL 框架

镜像备好后，实验侧用 `FRAMEWORK=custom`：实验目录放一个 `train.sh`，
`scripts/_run_experiment.sh` 会把密钥 / profile env / 产物目录 / 拓扑都准备好再 exec 它。
骨架与完整契约见 `templates/custom-framework/train.sh`，或直接 `lab new <名字> --method custom`。

> 配额不靠自觉：服务端按 profile 注册表记账，watchdog 还会做集群级 Ray 用卡对账。
> `custom` 放开的是「跑什么代码」，不是「占多少卡」。

### 本机：开发期依赖

本仓库根目录 **`pyproject.toml` + `uv`** 只管理 lab CLI 与数据预处理（`typer`、`datasets`、`pyyaml`），不含 NeMo-RL / vLLM / CUDA：

```bash
uv sync
uv run lab login --server https://lab.company.com   # 接入中心化服务
uv run lab ls
```

三种调用方式见根目录 README「统一 CLI」一节。

### 密钥与可追溯

`SWANLAB_API_KEY`、`HF_TOKEN`、`JUDGE_API_KEY`、Ray 地址、数据目录等都由**中心化服务**持有并在集群侧注入，
本仓库不入库任何密钥、也不再有 `submit.env`。多人共用集群时，服务端按账号隔离配额与产物目录。

`lab submit` / `lab export` / `lab eval` 每次由服务端记录 git commit / dirty / config 指纹与 `run_id`，
落到作业日志（`[run] version: ...` 行）。随时 `uv run lab runs`（`--all` / `--exp <名>`）查看「我的提交历史」并对上作业状态；
`uv run lab status` 在提交前看自己的配额、用量与活跃作业。
