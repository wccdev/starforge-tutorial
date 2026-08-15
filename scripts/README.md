# scripts/ — 通用脚本

作业提交统一走中心化服务：`lab login` 接入后 `lab submit <exp>`，由服务端打包上传并在集群代理执行。
本目录下的脚本只负责「在集群侧执行」或「本地工具」，不再有从本机直连 Ray 的提交脚本。

- `launch.sh` — 集群侧薄入口：只加载受信密钥/profile 环境，然后进入 SDK launcher。
  训练命令由版本化 recipe 选择 `nemo-rl`、`verl` 或显式 `custom` adapter 编译；
  不读取 `FRAMEWORK` 环境变量或实验目录 `framework` 文件，也不存在 adapter 回退。
  **它是唯一会随作业上传到集群的脚本**（`lab submit` 清单式打包）。

> 新建实验用 `lab new`；模型权重入内网属于运维操作，直接调下面的下载脚本
> （不再提供 `lab model` 包装——CLI 只保留作业契约相关的命令面）。
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
- `smoke_verl_sft.sh` — 可选真实 GPU smoke：固定 verl SFT recipe、Qwen2.5 0.5B、
  最小数据路径与 1×H100，失败直接退出。
- `smoke_verl_grpo.sh` — 可选真实 GPU smoke：固定 verl GRPO recipe、Qwen2.5 0.5B、
  最小数据路径、单步训练与 2×H200，失败直接退出。

训练后 `export/eval` 不再经过 shell 分支；它们和训练一样进入 `nemo-lab-launch`，由 recipe
选定 adapter。checkpoint 格式、评测数据和不支持的动作都会在提交前严格校验。
