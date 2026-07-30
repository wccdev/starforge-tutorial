# grpo_qwen3.5-9b_qa-rl-agent_gb10_v1（单台 DGX Spark GB10 保守运行档）

用 **GRPO** 在自有**技术培训考题题库**上强化训练 **Qwen 3.5 9B**。**多轮**：模型回答前可多次调用 `<search>`，由环境在**集群容器内**对本地资料目录做 **BM25** 检索 **markdown** 文件，拿到资料再作答。

> **这是 A/B 对比的处理组**：多轮 + **本地文档检索工具（默认 BM25，可切 grep）**。
> 基线组（实验一 / baseline）= [`grpo_qwen3.5-9b_qa-rl_v1`](../grpo_qwen3.5-9b_qa-rl_v1)：单轮、无工具。
> 两个实验共用同一数据集 / 模型 / LoRA / batch / 裁判奖励，**唯一变量**是「能否多轮检索本地资料」。
> 对比目标：让模型边查公司技术资料边答题**能否提升**它在公司技术考题上的作答准确率。

> ✅ **检索方式：本地 BM25（默认）。** `<search>` 在训练进程所在容器里对 `DOCS_DIR` 下 markdown 做
> **纯 Python 自实现的 BM25** 相关度检索（带排序、抗 OCR 噪声），Top-K 片段（带文件名+行号）回灌给模型；
> 不依赖任何外部服务/向量库。设 `DOCS_RETRIEVER=grep` 可切回旧的 grep 后端。
> `DOCS_DIR`（默认 `/data/docs`）目录不存在时 `<search>` 返回占位提示，流水线仍可跑通但拿不到资料。

## 与基线的唯一差异

| 维度 | 实验一 baseline | 实验二 treatment（本实验） |
| --- | --- | --- |
| 轮数 | 单轮（答一次即结束） | 多轮（`max_rollout_turns=3`） |
| 工具 | 无 | `<search>关键词</search>` → 容器内 BM25 检索本地 markdown（可切 grep） |
| 环境 | `common/environments/qa_env.py` `QARewardEnv` | `common/environments/qa_docs_agent_env.py` `QADocsAgentEnv` |
| 奖励 | qa 规则 + 简答裁判 | **同源**（最终答案复用同一套 qa 奖励）；训练另加检索 shaping，**验证不加**（见下节） |
| 数据 / 模型 / LoRA / batch | —— 完全一致 —— | 同一数据 / 模型；Megatron LoRA，有效 batch 16 |
| seq | 1536 | 2048（GB10 统一内存保守档） |

模型作答协议：检索用 `<search>关键词</search>`；作答把要点放入 `\boxed{...}`（与基线同一答案格式，保证判分一致）。

## 奖励：训练带检索加分，验证不带

奖励分两层：

- **最终判分**（训练/验证共用）：`\boxed{}` 里的答案交给同一套 qa 奖励（客观题规则 / 简答裁判），与 baseline 同源。
- **检索 reward shaping（仅训练）**：不再对“查到任意资料”给分；仅在检索后答对时给很小的 `+answer_search_bonus`（0.05），超轮不作答 `−no_answer_penalty`（0.2），无标签输出或空 `<search>` 各扣 `0.02`。这避免模型为刷即时 reward 无效重复检索。

验证环境由 `run.py` 用 `make_eval_cfg()` 另建一个实例，把所有 shaping 项全部归零（检索后端与判分方式不变）。
必须这样分开：NeMo-RL 的 `validation/accuracy` 就是 `mean(total_reward)`，而 `total_reward` 是**逐轮奖励的累加**——
验证若照抄训练 cfg，「用了工具」本身也会改变奖励尺度，
既看不出真实答题水平，也没法和无工具 baseline 同尺度比。所以：

- `validation/accuracy` = 纯答题得分（1.0 答对 / 0.0 答错或超轮不答），可直接与 baseline 对比；
- `train/reward` 含 shaping，会 >1.0，**不要**拿它当准确率读。

## 本地文档检索接入 · BM25 / grep（`common/environments/qa_docs_agent_env.py` 的 `docs_search()`）

`docs_search()` 按 `DOCS_RETRIEVER` 分派检索后端，默认 **BM25**：

- **BM25（默认，推荐）**：纯 Python 自实现、零外部依赖。首次检索时在 actor 进程内**懒构建一次倒排索引并缓存**（训练期资料不变）——遍历 `DOCS_DIR` 下 markdown，按空行分段、超长段再按 `DOCS_CHUNK_LINES` 行切窗成 chunk，分词建倒排与 IDF；查询时按 BM25 给每个 chunk 打**相关度分**、取 Top-K 回灌（带文件名+行号）。相比 grep「命中即返回、无排序」，BM25 召回与排序都更稳、**抗 OCR 噪声**。分词复用零依赖分词器（英文/型号正则 + 中文 2-gram，**不引 jieba**）。
- **grep（`DOCS_RETRIEVER=grep` 切回）**：`subprocess` 调 `grep -rinI -F` 递归检索，两段式——先整句精确匹配，落空再分词 OR 召回（多个 `-e`）。

通过环境变量配置（由中心化服务在集群侧注入到作业）：

```bash
DOCS_RETRIEVER=bm25          # 检索后端：bm25（默认）| grep
DOCS_DIR=/data/docs          # 资料根目录（含子目录），只搜其中 markdown。须是【容器内】真实存在的路径
DOCS_GLOB=*.md               # 只搜哪些文件，默认只搜 markdown
DOCS_TOP_K=3                 # 最多回灌几个命中片段（grep 按文件聚合 / bm25 按 chunk）
DOCS_MAX_CHARS=500           # 全局默认；本实验会以 cfg 中 retrieval_max_chars=400 覆盖该 actor
# —— BM25 专用 ——
DOCS_CHUNK_LINES=12          # 检索单元(chunk)大小：超长段落按多少行切窗
BM25_K1=1.5                  # 词频饱和系数
BM25_B=0.75                  # 文档长度归一化强度
# —— grep 专用 ——
DOCS_CONTEXT_LINES=2         # 每个命中带几行上下文（grep -C）
DOCS_MAX_PER_FILE=3          # 单文件最多取几处命中（grep -m）
DOCS_TIMEOUT=15              # 单次 grep 子进程超时（秒）
DOCS_OR_FALLBACK=1           # 整句查不到时是否再做「分词 OR 召回」。1 开 / 0 关
DOCS_MAX_TERMS=12            # OR 召回时最多用几个关键词（防碎词把所有行都召回）
```

> ⚠️ 检索发生在【集群训练进程】所在容器里 → `DOCS_DIR` 必须是**集群容器内**真实存在、含资料的路径（Mac 本机没有也没关系）。
> 模型可多轮换关键词逐步逼近答案（这正是 agentic 的部分）。

**正式跑前，在集群容器里自测资料已挂载（务必）：**

```bash
ls /data/docs                                                   # 确认资料目录已挂载、有子目录
grep -rinI --include="*.md" -C2 "随便挑一道题里的关键词" /data/docs | head   # 期望能打印出命中片段
```

> 换别的检索方式（向量检索 / 全文索引），只在 `docs_search()` 分派里加一个后端即可，环境其余逻辑不变。

## 跑起来

前置与基线相同（题库在集群 + `QA_RL_DATA_DIR` + 简答裁判 `JUDGE_*`），详见基线 README。本实验额外需要资料目录 `DOCS_*`（均由中心化服务在集群侧注入）：

```bash
# 1) 确保题库在集群、资料 markdown 已放到容器内 DOCS_DIR；服务端已注入 QA_RL_DATA_DIR / JUDGE_* / DOCS_*
# 2) 提交
lab submit grpo_qwen3.5-9b_qa-rl-agent_v1
```

## 单 GB10 防 OOM 配置

- **Megatron-Core + LoRA**：仅训练 rank-8 adapter，避免全参 Adam 状态占用 GB10 的统一 128GB 内存。
- **小 rollout**：`2 prompts × 8 generations = 16`，关闭动态采样，避免候选补采样导致瞬时内存翻倍。
- **Megatron generation**：不启动 vLLM，避开 GB10 上 vLLM colocated sleep/refit 的 CUDA 崩溃；`generation_batch_size=4`、4 GiB 中间 buffer、关闭 generation CUDA graphs。
- **短多轮轨迹**：`seq=2048`、每轮最多 384 token、检索回灌 400 字、最多一次检索后作答。
- **首跑跳过 reference policy**：`KL=0` 与 `skip_reference_policy_logprobs_calculation=true`，减少一份 9B logprob 的峰值；稳定后可用 CLI override `loss_fn.reference_policy_kl_penalty=0.01` 进行质量复跑。
- **NeMo-RL v0.7 保护**：开启 `overlong_filtering`，并受益于 selected-token logprob 内存降低；不启用 PPO、CISPO、MOPD 或动态采样，它们会增加单 GB10 的模型/rollout 峰值。
- **Qwen3.5 稳定性优先**：此档不再使用 vLLM；若需恢复 vLLM generation，必须先解决其 sleep/refit CUDA 错误。

首跑只训练 100 步、每 25 步验证。若仍 OOM，按顺序降低：`max_total_sequence_length=1792` → `num_generations_per_prompt=4` 且 `train_global_batch_size=8` → `retrieval_max_chars=300`。

> 运行镜像必须为 `nvcr.io/nvidia/nemo-rl:v0.7.0`（CUDA 13 / vLLM 0.20）；不要在 v0.6 容器内单独升级 Python 包。

## 看多轮检索轨迹

验证时每次会把若干条完整多轮对话（含 `<search>` 与 grep 检索结果）打印到作业日志，直接看日志即可：

```bash
uv run lab logs <JOB_ID>      # 不给 JOB_ID 则跟随最近一个作业
```

## 结论 / 记录

（训练后补：最佳 step、val 准确率、与 baseline 的对比、本地检索是否带来提升、SwanLab 链接、踩坑。）
