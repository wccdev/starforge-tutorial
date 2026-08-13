# nemo-rl-lab

基于 **NVIDIA NeMo-RL** 的大模型微调实验室。涵盖：

- **SFT**（监督微调）
- **GRPO / 强化学习**（RL）
- **多轮 Agent 训练**（NeMo-RL 较新特性，工具调用 / 多轮对话）

横跨多个基础模型（如 `qwen3.5-4b`、`qwen3.5-9b` 等）× 多个数据集。所有训练日志统一上传到云端 **SwanLab**。

> **本项目的初衷**：拿到仓库、配好远程机器，就能直接开跑微调——环境/分布式/提交这些脏活都内化掉，
> 你只需要关注**调参本身**（学习率、KL、采样数、数据、奖励）。

## 最快上手（3 步开跑）

> 训练跑在远程 H100 容器里，你只在自己机器上提交、看结果，全程不进容器、本机无需 GPU。
> 提交统一经**中心化 Lab 服务**：服务端持有 Ray 地址 / 密钥 / 数据目录，本机不直连 Ray、无需任何 `submit.env`。

```bash
# 1) 装本机 CLI（只是提交客户端，无需 GPU）+ 接入中心化服务（登录一次）
uv sync
lab login                                       # 登录官方 Lab 服务（默认 https://nemolab.gcoreinc.com）

# 2) 选实验、按需调参：打开 experiments/<exp>/config.yaml 顶部「调参速查」改几行
lab ls                                          # 看现成实验
lab new my_run --from grpo_qwen3.5-4b_gsm8k_v1  # 或 fork 一个来调参（自动改 SwanLab 名、继承目标集群）

# 3) 准备数据 → 提交 → 看结果
lab prepare gsm8k
lab submit grpo_qwen3.5-4b_gsm8k_v1             # 用实验自带的目标集群；--profile 可临时换
lab logs                                        # 跟随最近一个作业的实时日志
```

每个实验「调什么 / 数据 / 奖励 / 怎么跑」见其目录下 `README.md`。

## 硬件

| Profile | 说明 | 配置目录 |
| --- | --- | --- |
| `h200` | 单机 8× NVIDIA H200 141GB（**当前主力**） | `cluster/h200/` |
| `h200-2g` | 同机器，只要 2 张卡（小实验 / 调试，别占满集群） | `cluster/h200-2g/` |
| `h100` | 单机 1× NVIDIA H100 80GB | `cluster/h100/` |
| `gb10-spark` | 2× NVIDIA DGX Spark（GB10 Grace-Blackwell），Ray 组成 2 节点集群 | `cluster/gb10-spark/` |

训练配置与硬件解耦：NeMo-RL 通过 CLI override 调集群（`cluster.num_nodes` / `cluster.gpus_per_node`）；硬件相关 override 抽到 `cluster/<profile>/overrides.conf`。每个实验**自带目标集群**（实验目录下 `cluster` 文件，`lab new --cluster` 写入）——因为 batch/seq/LoRA/显存等超参都是按某张卡的显存调出来的；`lab submit --profile` 可临时换卡跑。

## 它和控制平面是什么关系

本仓是**客户端**：管实验、改超参、提交、看结果。真正的鉴权 / 配额 / 排队 / 调度
在控制平面 `nemo-rl-console` 里，本机不直连 Ray。

两边靠一份独立的契约包 `nemo-lab-sdk` 对接（源码在 console 仓的 `sdk/`）：

```
              nemo-lab-sdk          JobSpec、后训练方法目录、集群侧 launcher
             ╱      │      ╲
       lab CLI   console   训练镜像
```

**这个方向很重要**：以前是 console `pip install nemo-rl-lab`，控制平面依赖客户端 ——
本仓发个版就可能让服务端跑不起来。现在三方都只依赖契约包，彼此不再互相依赖。

「一个作业长什么样」也从 32 个环境变量的隐式约定，变成了一份带版本的 `JobSpec`：

```yaml
apiVersion: lab/v1
kind: TrainingJob
spec:
  recipe:    {name: grpo, version: 0.7.0}
  resources: {pools: [{name: train, series: h200, nodes: 1, gpus_per_node: 8}]}
```

`lab submit` 生成它，服务端据此装配作业。两边各有一份 golden test 钉住这份映射，
契约漂移会在 CI 当场失败，而不是在某次训练跑到一半时。

**加一种后训练方法不需要改本仓代码** —— 方法定义在 SDK 的 recipe 目录里，
`lab methods` 列的就是它。

> 本地联调 SDK：`uv pip install -e ../nemo-rl-console/sdk`
> （默认从内网 Git 拉，见 `pyproject.toml` 的 `[tool.uv.sources]`）

## 目录结构

```
nemo-rl-lab/
├── lab                       # CLI 薄 shim（= uv run lab）
├── nemo_rl_lab/              # 统一 CLI 实现（Typer；cli.py 为入口）
├── pyproject.toml            # uv 项目：依赖 + lab 命令入口（[project.scripts]）
├── uv.lock                   # 锁定版本（uv sync 用）
├── README.md                 # 本文件：总览
├── .gitignore
├── docs/                     # 文档
│   ├── naming-convention.md  # 命名规范（务必先读）
│   └── swanlab.md            # SwanLab 接入说明
├── cluster/                  # 硬件 / 分布式 profile + 依赖与环境说明（见 cluster/README.md）
│   ├── h100/                 # 单机 1× H100（远程微调平台主力）
│   ├── gb10-spark/           # 2× DGX Spark GB10
│   └── h200/                 # 单机 8× H200（异构集群新增卡型）
├── configs/                  # 配置继承体系（NeMo-RL 原生 defaults）
│   ├── base/                 # 祖父：官方 v0.6.0 example 原样副本（勿手改）
│   └── models/               # 父：各基础模型公共片段（qwen3.5-4b / 9b ...）
├── common/                   # 跨实验复用代码
│   ├── data/                 # 数据处理 / data processor
│   ├── environments/         # 自定义 Environment（GRPO 奖励来源 / 多轮 Agent）
│   └── utils/
├── datasets/                 # 数据集「元数据」（不放大文件，见下方约定）
├── templates/                # 新实验脚手架模板
│   └── experiment-template/
├── experiments/              # 练习 / 探索性实验
└── projects/                 # 正式 / 交付级项目
```

> 配置工作流：每个实验有自己的 `config.yaml`，通过 `defaults` **继承基底 + 模型片段，只写差异**
> （NeMo-RL 0.6.0 原生支持，官方亦如此）。`run.sh` 以该 `config.yaml` 为 `--config`，运行时再叠加
> `cluster/<profile>/overrides.conf` 的硬件 override。详见 `configs/README.md`。

## experiments vs projects

- **`experiments/`**：练习、调参、试错、复现。允许快糙猛，但每个目录必须有 `README.md` 记录目标、结论、SwanLab 链接。
- **`projects/`**：正式项目，要求可复现：固定依赖、固定数据版本、完整 eval、产出 checkpoint 导出流程。

两者内部目录布局一致（见 `templates/experiment-template/`），区别只是成熟度要求。

## 命名规范（核心）

每个实验目录统一命名为：

```
<method>_<model>_<dataset>[_<tag>]
```

- `method`：`sft` | `grpo` | `dpo` | `ppo` | `rm`（奖励模型）| `agent-grpo`（多轮 Agent）
- `model`：`qwen3.5-4b` | `qwen3.5-9b` | ...
- `dataset`：`gsm8k` | `alpaca` | `toolbench` | ...
- `tag`：可选，`v1` / `v2` 或日期 `20260602`

示例：

```
sft_qwen3.5-4b_alpaca_v1
grpo_qwen3.5-9b_gsm8k_v2
agent-grpo_qwen3.5-9b_toolbench_v1
```

字段间用 `_` 分隔，字段内（如模型名 `qwen3.5-4b`）用 `-`，避免歧义。完整规则见 [`docs/naming-convention.md`](docs/naming-convention.md)。

## 统一 CLI（`lab`）

所有操作都通过 `lab` 入口（[Typer](https://typer.tiangolo.com) 实现，纯 Python，**macOS / Linux / Windows 完全兼容**）：

```bash
uv run lab login                               # 接入官方 Lab 服务（默认 https://nemolab.gcoreinc.com）
uv run lab ls                                # 列出实验 / 项目
uv run lab methods                           # 有哪些后训练方法、各自能调什么超参（--method 的取值来源）
uv run lab new grpo_qwen3.5-4b_gsm8k_v1 --method grpo --cluster h200   # 从骨架新建实验
uv run lab diff grpo_qwen3.5-4b_gsm8k_v1 grpo_qwen3.5-9b_gsm8k_v1      # 对比两实验有效 config 差异（fork 调参常用）
uv run lab prepare gsm8k                     # 预处理数据集（gsm8k / alpaca / qa_rl）
uv run lab doctor                            # 体检：是否已登录 / 服务可达 / 当前配额
uv run lab status                            # 我的配额 / 用量 / 活跃作业（submit 前预检，别撞满卡）
uv run lab validate grpo_qwen3.5-4b_gsm8k_v1 # 提交前静态校验 config（本地秒级，省得跑到集群才报错）
uv run lab submit agent-grpo_qwen3.5-9b_multitool_v1   # 经服务端提交作业到集群（提交前自动校验）
uv run lab logs                              # 跟随最近一个作业日志（= lab job logs 便捷版）
uv run lab export grpo_qwen3.5-9b_gsm8k_v1   # 训练后：把 checkpoint 转 HF（自适应 dcp/megatron），可 --push-repo 推 Hub
uv run lab eval grpo_qwen3.5-9b_gsm8k_v1     # 训练后：对 checkpoint 跑独立评测（未给 --model 时先自动导出）
uv run lab runs                              # 我的提交历史（服务端台账：run_id / 状态 / GPU）
uv run lab job stop <job_id>                 # 停止运行中的作业
uv run lab sync-base --nemo-rl /opt/NeMo-RL  # 升级版本时同步官方基底配置
```

> 首次使用：`uv run lab login` 接入官方 Lab 服务，再 `uv run lab doctor` 确认已登录、服务可达，然后 `lab submit`。
> 提交一律经服务端代理：Ray 地址 / 密钥 / 数据目录都在服务端，本机不直连 Ray、无需任何 `submit.env`。
> 每次 `lab submit` 会自动：① 校验 config（batch 三者相等等，不过不放行，可 `--no-validate` 跳过）；
> ② 由服务端记录 git commit / dirty / config 指纹与 `run_id`。
> 事后 `lab runs` 看「我的提交历史」对上作业状态（RUNNING/SUCCEEDED/FAILED…）；
> `lab status` 则在提交前看自己的配额、用量与活跃作业，避免撞满卡。

三种等价调用方式：

| 方式 | 说明 |
| --- | --- |
| `uv run lab ...` | 推荐；uv 自动同步项目环境再运行，**macOS / Linux / Windows 均可用** |
| `./lab ...`（macOS / Linux）或 `lab.cmd ...`（Windows） | 仓库根的薄 shim，内部就是 `uv run lab` |
| `lab ...` | `uv sync` 后 `.venv/bin/lab`（Unix）或 `.venv\Scripts\lab.exe`（Windows）已生成；激活 venv 即可直接用 |

`uv run lab <子命令> --help` 看每个命令的参数。CLI 封装 `nemo_rl_lab/` 下的 Python 实现（`lab new` / `lab sync-base` 等不再依赖 bash）。
实现见 `nemo_rl_lab/cli.py`。

### 终端补全（Tab）

子命令、实验名、数据集、profile 都支持 Tab 补全。**推荐**显式指定 shell（不依赖自动检测，CI / IDE 终端也能用）：

```bash
# 已激活 venv 或 PATH 里有 lab 时
lab completion install zsh          # macOS 默认 shell
lab completion install bash
lab completion install fish
lab completion install powershell   # Windows PowerShell 5.x
lab completion install pwsh         # PowerShell Core

# 只打印脚本、手动粘贴到配置里
lab completion show zsh
```

仓库根用 **`./lab` shim**（未把 `.venv/bin` 加入 PATH）时，安装 bash 包装：

```bash
./lab completion install bash --wrapper
# 或：lab completion show bash --wrapper >> ~/.bashrc
```

之后在**仓库根目录** `./lab sub<Tab>`、`./lab submit <Tab>` 即可补全。

Typer 自带的（需当前终端能自动识别 shell，非 TTY 时常失败）：

```bash
lab --install-completion
lab --show-completion
```

| 场景 | 建议 |
|------|------|
| macOS / Linux，venv 里 `lab` | `lab completion install zsh`（或 bash/fish） |
| 仓库根 `./lab` | `lab completion install bash --wrapper` |
| Windows PowerShell | `lab completion install powershell` |
| Windows cmd.exe | 不支持；请用 PowerShell、Git Bash 或 WSL |
| 自动检测失败 `Shell not supported` | 改用 `lab completion install <shell>` |

说明：

- 补全注册在命令名 **`lab`** 或 **`./lab`（wrapper）** 上；`./lab` 的 wrapper 补全仅在仓库根生效。
- 实验名列表来自安装包旁的 `experiments/` 目录；editable 安装（`uv sync`）下与仓库同步。
- 安装后需**重开终端**或 `source ~/.zshrc` / `source ~/.bashrc`。

## 新建一个实验（细节）

```bash
# 方式一：从空白模板新建，并绑定目标集群（写入实验自带 cluster 文件）
uv run lab new grpo_qwen3.5-4b_gsm8k_v1 --cluster h100   # 或 bash scripts/new_experiment.sh experiments <name> "" h100

# 方式二（推荐调参）：fork 一个现成实验，只改超参试不同配置
uv run lab new grpo_qwen3.5-4b_gsm8k_lr1e4 --from grpo_qwen3.5-4b_gsm8k_v1
#   自动 copy 目录、把 config.yaml 的 swanlab project/name 改成新名（避免日志撞车）、并继承来源实验的目标集群
#   想换到别的集群再加 --cluster <profile>

cd experiments/<新实验名>
# 1. 改 config.yaml 顶部「调参区」：lr / kl / 采样数 / 数据集 / seq（这些数值按目标集群的卡调）
# 2. 目标集群写在同目录 cluster 文件（lab new 已写好；想改：echo gb10-spark > cluster）
# 3. （新建空白时）改 README.md 与 defaults；若是 SFT/Agent，run.sh 顶部改 ENTRY（见 configs/README.md）
# 4. 提交（用实验自带集群；--profile 可临时换）：
uv run lab submit <新实验名>
```

## 示例实验（覆盖三种方法）

| 实验 | 方法 | 说明 |
| --- | --- | --- |
| [`experiments/sft_qwen3.5-4b_alpaca_v1`](experiments/sft_qwen3.5-4b_alpaca_v1) | SFT | Alpaca 指令监督微调（本地 jsonl + ResponseDataset） |
| [`experiments/grpo_qwen3.5-4b_gsm8k_v1`](experiments/grpo_qwen3.5-4b_gsm8k_v1) | GRPO（单轮） | GSM8K 数学推理（4B + LoRA dim16/lr2e-4，非 colocated） |
| [`experiments/grpo_qwen3.5-9b_gsm8k_v1`](experiments/grpo_qwen3.5-9b_gsm8k_v1) | GRPO（单轮） | GSM8K 数学推理，math 环境验证 |
| [`experiments/grpo_qwen3.5-9b_qa-rl_v1`](experiments/grpo_qwen3.5-9b_qa-rl_v1) | GRPO（单轮，自定义判分） | 自有技术培训考题；客观题规则判分 + 简答 LLM 裁判 |
| [`experiments/agent-grpo_qwen3.5-9b_multitool_v1`](experiments/agent-grpo_qwen3.5-9b_multitool_v1) | GRPO（多轮 Agent） | 多工具（检索/计算/代码）调用，自定义环境 |

数据预处理脚本见 `common/data/`（gsm8k / alpaca / qa_rl）。自定义环境见 `common/environments/`，判分逻辑见 `common/rewards/`。

## 训练工作流（本机 → 中心化服务 → 集群）

**在本机写代码 + 提交，训练跑在集群容器里**，日常提交不进容器、不需要 GPU、代码随作业自动上传。
提交统一经中心化 Lab 服务：本机把工作目录打包上传，服务端注入 Ray 地址 / 密钥 / 数据目录后代理提交到集群。

```bash
# A. 一次性：装本机 CLI + 接入中心化服务
uv sync
uv run lab login

# B. 每次：提交、看/停作业（全程经服务端，本机不直连 Ray）
uv run lab submit grpo_qwen3.5-4b_gsm8k_v1
uv run lab job ls                   # 我的作业列表
uv run lab logs <job_id>            # 实时日志（不给 job_id 跟随最近一个）
uv run lab job stop <job_id>        # 停止作业
```

## 训练后闭环（导出 / 评测）

训练产物（checkpoint）落在集群 `OUTPUT_ROOT[/<RUN_USER>]/<实验名>/step_<N>/`。两条命令把它变成「可交付资产」，
执行同样在集群（薄封装 NeMo-RL 0.6.0 官方脚本，经服务端提交、不进容器）：

```bash
# 导出：DCP/Megatron checkpoint → HuggingFace 格式（按后端自适应选转换器，自动带上 tokenizer）
uv run lab export grpo_qwen3.5-9b_gsm8k_v1                 # 默认最新 step；产物落 <ckpt>/hf_export/step_<N>
uv run lab export grpo_qwen3.5-9b_gsm8k_v1 --step 170 --push-repo myorg/qwen-gsm8k   # 指定步数并推到 HF Hub
uv run lab export grpo_qwen3.5-9b_gsm8k_v1 --dry-run       # 只打印将执行的转换命令，不提交

# 评测：对 checkpoint 跑 run_eval.py（仅吃 HF 格式；未给 --model 时先自动导出再评测）
uv run lab eval grpo_qwen3.5-9b_gsm8k_v1                                  # 默认 eval 配置
uv run lab eval grpo_qwen3.5-9b_gsm8k_v1 --eval-config examples/configs/evals/math_eval.yaml \
    -- generation.temperature=0.6 generation.top_p=0.95                  # `--` 之后透传给 run_eval.py
uv run lab eval grpo_qwen3.5-9b_gsm8k_v1 --model myorg/qwen-gsm8k         # 直接评测某 HF 模型/Hub id
```

- **后端自适应**：GRPO（Megatron 后端）走 `convert_megatron_to_hf.py`（`--extra mcore`），SFT（DTensor）走 `convert_dcp_to_hf.py`。
  脚本按 checkpoint 里是否存在 `policy/weights/iter_*` 自动判别，无需手选。
- **step 选择**：默认取最新 `step_<N>`；`--step N` 指定。
- **导出/评测也记台账**：与 `submit` 一样由服务端记录 action / run_id / commit，可追溯（`lab runs` 查看）。
- 集群侧细节见 [`scripts/post_train.sh`](scripts/post_train.sh)（支持 `LAB_DRY_RUN=1`）。

## 快速开始

1. **本机 CLI + 经服务端提交**：上方「最快上手」
2. **集群内 NeMo-RL / 依赖 / 架构差异**：[`cluster/README.md`](cluster/README.md)（§依赖与环境）
3. 配置 SwanLab：[`docs/swanlab.md`](docs/swanlab.md)
4. 集群 / 硬件 profile：[`cluster/README.md`](cluster/README.md)
5. 命名规范：[`docs/naming-convention.md`](docs/naming-convention.md)
