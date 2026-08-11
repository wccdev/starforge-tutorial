# grpo_qwen3.5-9b_qa-rl-agent_v3（H200 优化版 · 实验二 / treatment）

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
| 数据 / 模型 / LoRA / batch | —— 完全一致 —— | 同一数据 / 模型 / LoRA；为多轮轨迹使用有效 batch 64 |
| seq | 1536 | 4096（H200 141GB：容纳检索回灌，避免答案被截断） |

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
DOCS_MAX_CHARS=500           # 全局默认；本实验会以 cfg 中 retrieval_max_chars=900 覆盖该 actor
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

## H200 优化：稳定在线检索 RL

旧版 GRPO 曾在 reward 全 0 / 全 1 的组内因标准差归一化而产生极端 advantage，导致熵塌缩和复读。本版使用：

- **Reinforce++**：按 prompt 均值去基线，再在有效 token 上全局归一化；避免组标准差接近 0 时的数值爆炸。R1-Searcher 的检索 Agent 也采用这一估计器。
- **DAPO dynamic sampling**：每步先生成 `8×16` 个候选，只保留奖励有成败差异的 4 个 prompt 组训练；没有梯度的全对/全错组不消耗更新。
- **H200 序列预算**：`seq=4096`、每轮 `max_new_tokens=512`、单次检索最多回灌 900 字，优先保证“检索一次后能完整作答”，而不是长文本生成。
- **参考策略锚定**：`KL=0.02` 加入 Reinforce++ 的 token reward，压制策略漂移和重复格式。
- **NeMo-RL v0.7 长轨迹保护**：`overlong_filtering=true` 丢弃撞到生成上限的未完成轨迹；v0.7 的 GRPO selected-token logprob 内存优化可降低长序列训练压力。
  ⚠️ 该开关**必须**与 `advantage_clip_low/high=∓5.0` + `reward_shaping`（soft overlong punishment）同时使用，三者缺一会复现下述事故。
- **Qwen3.5 vLLM 稳定性优先**：v0.7 / vLLM 0.20 有 CUDA-graph/Ray 间歇 hang 的上游已知问题，因此 H200 profile 强制 `enforce_eager=true`。吞吐会下降；确认上游修复前不要改回 false。

运行时关注 `train/approx_entropy`、`train/advantages/max`、`dynamic_sampling_num_gen_batches` 和纯答题 `validation/accuracy`。若动态采样频繁达到 8 个补采样批次，说明当前题目过易或过难，应先改善题目难度混合，而不是提高学习率。

### 事故复盘：20260810 step103 NaN 崩溃（raysubmit_6UEmn8hcQtHrRKCv）

该 run 训到 step 103 报 `RuntimeError: iteration 103: found NaN in local grad norm for bucket #0`。
**不是显存问题、也不是学习率过高**（LoRA lr=1e-4 有 warmup+cosine，且 `clip_grad=1.0` 全程生效；
NaN 产生在 DP 通信之前即梯度裁剪之前，裁剪救不了）。真正的链条是 `overlong_filtering` 单独使用
造成的**幸存者偏差**：

| 环节 | 指标证据 |
|---|---|
| ① 生成长度发散 | `avg_turns_per_sample` 1.66(s35)→3.00(s103)；`mean_gen_tokens` 446→1326 |
| ② 截断率飙升 | `truncation_rate` 0.09(s35)→0.98(s102)→**1.0(s103)** |
| ③ 有效样本塌缩 | `num_valid_samples` 58(s32)→6(s80)→**1**(s101)；`global_valid_toks` 27929→209 |
| ④ 优势爆炸 | `advantages/mean` 常态 -0.2~-2 → **-66.8**(s102)，min -99.3 |
| ⑤ 梯度崩 | `grad_norm` 0.3~0.75(s1-80)→**10.6**(s101)；s103 有效 token 归零 → NaN |

关键点：`qa_reward.py` 的 `FORMAT_PENALTY=-0.5`（写不出 `\boxed{}` 的重罚）本身完全够用，
但被 `overlong_filtering` 拦在梯度之外，**一次都没传到模型**——模型只从"没被截断"的样本里学习，
于是"多搜一轮、多写一点"看起来永远只有收益没有代价，长度单调发散直到 100% 截断。
同期 `validation/accuracy` 从 0.2686(s50) 跌到 0.1045(s100)、`approx_entropy` 0.725→0.125（复读塌缩）。

修复（见 config.yaml 对应注释）：`advantage_clip_low/high=∓5.0` 防崩 + `reward_shaping` 的
soft overlong punishment 治本（在**还没被截断**的缓冲区内按超长程度扣分，这些样本仍在梯度里）
+ `val_period` 50→25 让劣化能早一半发现。姊妹实验 `gb10_v1` 曾踩过同一个坑，取同一组 clip 值。

> 运行镜像必须为 `nvcr.io/nvidia/nemo-rl:v0.7.0`（CUDA 13 / vLLM 0.20）；不要在 v0.6 容器内单独升级 Python 包。

## 看多轮检索轨迹

验证时每次会把若干条完整多轮对话（含 `<search>` 与 grep 检索结果）打印到作业日志，直接看日志即可：

```bash
uv run lab logs <JOB_ID>      # 不给 JOB_ID 则跟随最近一个作业
```

## 结论 / 记录

（训练后补：最佳 step、val 准确率、与 baseline 的对比、本地检索是否带来提升、SwanLab 链接、踩坑。）
