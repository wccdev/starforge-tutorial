# cluster/ — 硬件 / 分布式 profile

NeMo-RL 0.6.0 的集群设置（`cluster.num_nodes`、`cluster.gpus_per_node`）和并行度都在训练配置里，通过 **CLI override** 调整。本目录把「不同硬件的 override」抽出来复用，训练配置与硬件解耦。

每个 profile 一个子目录，核心是 `overrides.conf`（每行一个 `key=value`，`#` 为注释）：

- `h100/` — 单机 1× H100 80GB（单节点单卡，远程微调平台主力）
- `gb10-spark/` — 2× DGX Spark GB10（Ray 2 节点）
- `h200/` — 单机 8× H200 141GB（异构集群新增卡型）
- `h200-2g/` — 单机切 2× H200（console 注册 `gpus_per_node=2`；TP=1×CP=1，DP=2）

## 用法

每个实验**自带目标集群**（实验目录下一行 `cluster` 文件，记录 profile 名）。`lab submit`
读取它；`--profile` 可以显式覆盖：

```bash
lab submit grpo_qwen3.5-4b_gsm8k_v1
lab submit grpo_qwen3.5-4b_gsm8k_v1 --profile h100
```

> **profile 优先级**：`--profile`（显式）> 实验自带 `cluster` 文件。两者都没有时直接失败，
> 不选择默认集群。
> 新建实验时用 `lab new <名字> --cluster h100` 写好绑定；`lab new <名字> --from <实验>` 会继承来源实验的绑定。

提交一律经**中心化 Lab 服务**：本机不读任何 `submit.env`，Ray 地址 / 密钥 / 产物根目录（`OUTPUT_ROOT`）/
数据目录 / 裁判 LLM 等都由服务端在集群侧注入。`--profile` 选择受控 profile；集群侧
`scripts/launch.sh` 只加载受信环境，SDK adapter 负责编译配置和拓扑。

等价于（集群侧实际执行）：

```bash
uv run python examples/run_grpo.py --config <base.yaml> \
    cluster.num_nodes=2 cluster.gpus_per_node=1 ...   # 来自 profile 的 overrides.conf
```

## 各 profile 包含

- `overrides.conf` — 节点数 / 每节点 GPU / 并行度等 NeMo-RL 覆盖项（CLI override）
- `env.sh` — 集群 env（NCCL/RoCE 网络 + Ray 内存监控 + PyTorch 显存分配），由统一 launcher 加载

> `overrides.conf` 走 CLI override（进 NeMo-RL 配置）；`env.sh` 走进程环境变量（NCCL/Ray/PyTorch 这类不属于训练配置的开关）。两者互补。

> **⚠️ 拓扑以 JobSpec 与服务端注册表为权威**：`resources.pools` 决定占卡与配额计量；
> Console 校验 profile、池和 recipe role，adapter 再把同一拓扑写入框架原生命令。平台不解析
> `overrides.conf` 猜 GPU，也不会在配置冲突时悄悄改用另一组数量。

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
3. **actor venv 可能在作业运行时现建**（`nemo_rl/utils/venvs.py`：按 actor 类型在
   `NEMO_RL_VENV_DIR`=`/opt/ray_venvs` 下 `uv sync`）。官方镜像**已经预建了主 venv**
   （`/opt/nemo_rl_venv`），但 `/opt/ray_venvs` 是空的。这一步要够索引，内网下可能卡住。
   平台镜像在构建期把它一并预热了，见下。

#### 构建时要拿到四类包（内网 PyPI 镜像不够）

NeMo-RL 的依赖来源，用 `uv.lock` 实际枚举出来是这样：

| 来源 | 内容 | 官方 PyPI 镜像覆盖 |
| --- | --- | --- |
| `files.pythonhosted.org` | 1221 个 wheel | ✅ |
| `download*.pytorch.org` | torch / vision / audio / triton | ❌ |
| `flashinfer.ai/whl/cu130` | flashinfer 系列 | ❌ |
| `github.com` | release 资产 + `nemo_run` git 依赖 | ❌ |

而且这些 URL **同时钉死在 `uv.lock` 里**（`index = "https://download.pytorch.org/..."`、
直连的 `url = "https://github.com/..."`）——所以**改 `pyproject.toml` 的 index 没有用**，
必须连 uv.lock 一起改，每次升级 NeMo-RL 都要重来。别走这条路。

两个可行解：**在能出网的机器上构建**（推荐），或**全内网构建 + 带白名单的 HTTPS 正向代理**
（`HTTPS_PROXY`，一次覆盖四类且不动任何文件）。

> 顺带：**不需要配 NVIDIA 的 PyPI**。NeMo-RL 没有引用 `pypi.nvidia.com`；
> `nvidia-cudnn-cu13` 之类是 torch 的普通 PyPI 依赖。

#### 运行时索引

运行时使用 recipe 固定的镜像和依赖，不在作业启动时从索引补装框架。
索引只用于镜像构建或排错环境。
服务端已有 `passthrough_env` 透传机制，写进 `LAB_PASSTHROUGH_ENV` 即对所有作业生效，零代码：

```bash
UV_DEFAULT_INDEX=https://nexus.<内网域名>/repository/pypi-group/simple
PIP_INDEX_URL=https://nexus.<内网域名>/repository/pypi-group/simple
HF_ENDPOINT=https://nexus.<内网域名>/repository/hf-proxy
```

#### 镜像与集群部署

Dockerfile / compose / 构建脚本都在 **console 仓库的 `deploy/ray-cluster/`**（集群运维属于控制面）：

```bash
bash deploy/ray-cluster/build.sh                        # 构建 + 记台账
EXTRA_PIP="trl==0.2x.x" bash deploy/ray-cluster/build.sh  # 叠加额外依赖
docker compose --profile head up -d                     # 节点起停
```

**架构必须对上**：`h100` / `h200` 是 x86_64，`gb10-spark` 是 aarch64 + Blackwell，两套镜像分开构建。
`flash-attn` 这类要编译的包在 aarch64 上基本是坑，规划 L3 实验时优先放 H100/H200。

**⚠️ 装额外依赖前先想版本冲突**：TRL 和 NeMo-RL 都吃 transformers，装错版本会让**所有**
NeMo-RL 训练一起挂。这是「单胖镜像」路线的天花板，撞到了就该上多 Ray 集群方案。

#### 跑 verl 或 custom

verl 是一等 adapter：用 `lab new <名字> --method verl-sft|verl-grpo` 创建固定模板，并在提交时
显式传 `--model`、`--train-data`、`--validation-data`。自定义框架必须选择 `custom` recipe：
`lab new <名字> --method custom`。平台不读取 `FRAMEWORK` 或 `framework` 文件，也不会在其他
adapter 失败后执行 `train.sh`。

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
