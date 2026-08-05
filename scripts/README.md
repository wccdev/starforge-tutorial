# scripts/ — 通用脚本

作业提交统一走中心化服务：`lab login` 接入后 `lab submit <exp>`，由服务端打包上传并在集群代理执行。
本目录下的脚本只负责「在集群侧执行」或「本地工具」，不再有从本机直连 Ray 的提交脚本。

- `new_experiment.sh` — 从模板快速新建实验（薄封装 → `python -m nemo_rl_lab.new_experiment`，跨平台）
  ```bash
  bash scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_v1
  # 等价：uv run lab new grpo_qwen3.5-4b_gsm8k_v1 --method grpo --cluster h100
  ```
- `_run_experiment.sh` — **实验启动通用逻辑**（集群侧执行）：各实验 `run.sh` 收口于此，叠加
  `cluster/<profile>/overrides.conf` + `env.sh`，落盘到服务端注入的 `OUTPUT_ROOT`。
  支持两种框架（`FRAMEWORK` 环境变量 / 实验目录下的 `framework` 文件）：
  - `nemo-rl`（默认）— 经 `nemolab_boot.py` 起 NeMo-RL 入口
  - `custom` — 跑实验自带的 `train.sh`（TRL / verl / 纯 HF Trainer…）。契约见
    `templates/custom-framework/train.sh`；`lab new <名字> --method custom` 直接生成骨架。
    ⚠️ 新依赖装不进作业，需 overlay 镜像，见 `cluster/README.md`。

  排错：置 `LAB_DRY_RUN=1` 只打印最终命令、不真正执行。

> 下面三个模型下载脚本已固化成一等公民命令，日常用 `lab model` 即可，不必直接调：
> `lab model pull <repo> --via direct|relay|nexus` · `lab model install <平铺目录>` · `lab model ls`。
> 脚本本身保留，供需要冷门开关时直接调用。
- `prefetch_hf_model.sh` — **在集群容器内**预下载 HF 模型到 `HF_HOME`（避免训练时连不上 hf-mirror.com）
  ```bash
  # 在容器里先导出所需环境变量
  export HF_TOKEN=... HF_HOME=/data/hf_cache
  bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B
  ```
- `download_models.py` — **在能连外网的下载机上**把模型直接下成 HF 缓存目录结构，
  整个目录拷到算力机、`export HF_HOME=<该目录>` 就能用，config 里继续写 repo id 不用改路径。
  支持断点续传（中断后重跑同一命令）、后台运行、完成标记跳过、磁盘体积预检、HF / 魔搭双源。
  ```bash
  uv run python scripts/download_models.py --list                       # 先看体积（三个模型合计约 177 GiB）
  uv run python scripts/download_models.py --daemon                     # 走 HF，后台下载到 ./hf_cache
  uv run python scripts/download_models.py --source modelscope --daemon # 国内走魔搭
  tail -f hf_cache/download.log
  ```
  同步时**必须用 `rsync -a` 保留符号链接**——HF 源产出的 `snapshots/` 里是指向 `blobs/` 的相对软链，
  用 `rsync -aL` 或 `scp -r` 会把它们展开成实体文件，体积正好翻倍（实测 3.7M → 7.4M）。
  ```bash
  rsync -avP --exclude '*.log' --exclude '*.pid' hf_cache/ user@算力机:/data/hf_cache/
  ```
  **国内加速**：`--endpoint https://hf-mirror.com` 走 HF 镜像，但该站对境外 IP 会 308 回官方站
  （`prefetch_hf_model.sh` 里那条注释说的就是这个）；国内机器更推荐 `--source modelscope`，
  三个目标模型在魔搭上的字节数与 HF 完全一致，且魔搭 API 带 sha256，可用 `--verify` 校验完整性。

  布局要点（都是实测确认的坑）：`refs/main` 里的 sha **结尾不能有换行**，多一个 `\n` 就解析不到；
  snapshots 目录名必须是 **HF 上真实的 commit sha**，否则联网时 hub 会认为没缓存、整个重下一遍
  （魔搭源不提供该 sha，脚本会联网向 HF 查一次几 KB 的元数据；连不上就用 `--fake-sha` 并在算力机
  设 `HF_HUB_OFFLINE=1`）。魔搭源在 `snapshots/<sha>/` 下直接放实体文件、不造 `blobs/`，
  实测离线和联网都能正常解析且不会触发重下。
- `download_via_relay.py` — **推理机不能上外网、只能 ssh 到某台中继机**时用这个。在推理机上直接跑，
  每个文件都通过 `ssh <中继机> curl` 把字节流管道回本机落盘，中继机全程不落盘、不需要有模型目录、
  也不用装 python（只要有 curl）。省掉「先在中继机下完再 scp 过来」这一步和那份中间磁盘占用。
  只用标准库，可以单独 scp 到推理机上用系统 `python3` 跑。
  ```bash
  python3 download_via_relay.py --relay root@10.0.0.2 --check    # 先验链路：ssh 通不通、有没有 curl、能不能出网
  python3 download_via_relay.py --relay root@10.0.0.2 --list
  python3 download_via_relay.py --relay root@10.0.0.2 --hf-home /data/hf_cache --daemon
  tail -f /data/hf_cache/relay_download.log
  ```
  `--only` 不限于内置的三个模型，可以是任意 repo id；两边命名空间不同时用 `下载源id=HF id`
  显式给出（缓存目录一律按右边的 HF 名字建，这样 config 里写 HF repo id 就能加载）：
  ```bash
  --only Qwen/Qwen3.5-4B                                    # HF 源，任意仓库
  --only ZhipuAI/GLM-4.7-Flash=zai-org/GLM-4.7-Flash        # 魔搭下载，按 HF 名字建缓存目录
  --only iic/xxx --fake-sha                                 # 魔搭独有、HF 上没有的模型
  ```
  `--fake-sha` 用文件清单摘要当 `snapshots/` 目录名（内容变了摘要才变，重跑稳定不会分裂出多个
  快照目录），这种模型加载时必须设 `HF_HUB_OFFLINE=1`，否则 hub 联网对不上 sha 会整个重下。
  断点续传靠 HTTP Range：未完成的存成 `<文件>.part`，重跑同一命令从断点接着下。实测从 900 MB 残片
  续传 953 MB 的模型只花 10 秒（完整下载要 2 分钟）且 sha256 通过。续传前会先发一个 1 字节的
  Range 探测，服务端若不认 Range 就丢弃残片重下，不会把整文件追加到残片后面搞出个坏文件。
  产出与 `download_models.py` 完全一致的缓存布局，跑完 `export HF_HOME=<目录>` 直接按 repo id 加载。
  `--verify` 走 sha256（HF 源只校验 LFS 大文件，魔搭源全量校验）。`--token` 经 curl 的 stdin 传，
  不会出现在中继机的进程列表里。`--workers N` 是并发 ssh 通道数，链路不稳就调小。
- `install_to_hf_cache.py` — 把**旧版平铺布局**（早期 `download_models.py` 下到 `hf_models/` 的产物）
  转成上面的缓存布局，默认硬链接、不占额外空间。新下载的不需要这一步。
  ```bash
  export HF_HOME=/data/hf_cache
  uv run python scripts/install_to_hf_cache.py --src hf_models --dry-run
  ```
- `sync_base_configs.sh` — 升级 NeMo-RL 版本时同步官方基底配置到 `configs/base/`（薄封装 → `python -m nemo_rl_lab.sync_base`）
- `post_train.sh` — **训练后闭环**（集群侧执行，由 `lab export` / `lab eval` 经服务端代理调起）：
  把 checkpoint 转 HF（按后端自适应 `convert_dcp_to_hf.py` / `convert_megatron_to_hf.py`，可推 HF Hub），
  或对 checkpoint 跑评测。带 `LAB_DRY_RUN=1` 只打印命令不执行。

  评测入口按「实验自带优先」选：**实验目录下有 `eval.py` 就用它**（与训练入口 `run.py` 同款约定），
  否则回落官方 `examples/run_eval.py`。`--` 之后的参数原样透传给所选入口。
  自带 `eval.py` 用于官方协议对不上的场景——比如要按论文口径每题采 N 条、
  算 pass@N / majority@N，或一次评多个数据集（见 `experiments/opsd_qwen3.5-9b_math_v1/eval.py`）。
  ```bash
  # 通常用 CLI（经服务端提交，执行在集群）：
  uv run lab export grpo_qwen3.5-9b_gsm8k_v1 [--step N] [--push-repo user/name]
  uv run lab eval   grpo_qwen3.5-9b_gsm8k_v1 [--step N] [-- generation.temperature=0.6]
  ```
