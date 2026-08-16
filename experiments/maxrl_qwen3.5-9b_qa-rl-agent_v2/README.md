# maxrl_qwen3.5-9b_qa-rl-agent_v2（MaxRL 改版）

用 **MaxRL**（GRPO 的最大似然改版）在自有**技术培训考题题库**上强化训练 **Qwen 3.5 9B**。**多轮**：模型回答前可多次调用 `<search>`，由环境在**集群容器内**对本地资料目录做 **BM25** 检索 **markdown** 文件，拿到资料再作答。

> **这是 [`grpo_qwen3.5-9b_qa-rl-agent_v1`](../grpo_qwen3.5-9b_qa-rl-agent_v1) 的 MaxRL 改版**：
> 数据 / 模型 / LoRA / batch / seq / 裁判奖励 / 多轮检索**全部一致**，**唯一变量是 RL 优化目标**——
> 把 GRPO 的优势归一化从「除以组内标准差 σ」换成 MaxRL 的「除以组内平均奖励 μ」。
> 对比目标：在同样的多轮检索题库任务上，**MaxRL 能否比 GRPO 更准 / 更抗过拟合（pass@k 不退化）**。

## MaxRL：相对 GRPO 改了什么（核心）

论文《Maximum Likelihood Reinforcement Learning》（MaxRL, arXiv:2602.02710）指出：传统 RL 最大化「期望通过率 pass@1」，只是最大似然目标 `J_ML=log p` 的**一阶近似**；按 rollout 数 `N` 估计时，MaxRL 的无偏梯度对应把 `J_ML` 截断到 `T=N` 阶，**增大 rollout 不只是降方差，而是直接逼近更高保真的最大似然目标**，从而把学习信号集中到低通过率难题、缓解 pass@k 退化（分布锐化）。

落到代码就是**优势计算改一行**（见 `common/algorithms/maxrl.py`）：

| 方法 | 优势 Â_i | 对难题(μ 小)的加权 |
| --- | --- | --- |
| REINFORCE | `(r_i − μ)` | 无额外加权 |
| GRPO（v1） | `(r_i − μ) / σ` | ~ `1/√μ` |
| **MaxRL（本实验）** | `(r_i − μ) / μ` | ~ `1/μ`（更强，趋近最大似然） |

其中 `μ` = 该 prompt 一组 rollout 的**平均奖励 = 经验通过率**（含样本自身，非 leave-one-out），`σ` 为组内标准差。`μ=0`（整组无一答对）或 `μ=1`（全对）时该组优势全为 0，与 GRPO「无变化不回传梯度」一致。二元奖励下，难题答对得 `(1−μ)/μ≈1/μ` 的大正优势、失败得 `−1`。KL 惩罚 / ratio clip / token-level loss 等其余部分与 GRPO 完全一致。

**接入方式**：`config.yaml` 里 `grpo.adv_estimator.name: "maxrl"` 触发；`run.py` 在 `grpo_train()` 前调用 `install_maxrl_estimator()` 给 NeMo-RL 的 `_create_advantage_estimator` 打补丁。删掉 config 里那两行 `adv_estimator` 即原样回退成 GRPO（= v1）。

> ✅ **检索方式：本地 BM25（默认）。** `<search>` 在训练进程所在容器里对 `DOCS_DIR` 下 markdown 做
> **纯 Python 自实现的 BM25** 相关度检索（带排序、抗 OCR 噪声），Top-K 片段（带文件名+行号）回灌给模型；
> 不依赖任何外部服务/向量库。设 `DOCS_RETRIEVER=grep` 可切回旧的 grep 后端。
> `DOCS_DIR`（默认 `/data/docs`）目录不存在时 `<search>` 返回占位提示，流水线仍可跑通但拿不到资料。

## 与 v1（GRPO 版）的唯一差异

| 维度 | v1（`grpo_...-agent_v1`，GRPO） | v2 本实验（MaxRL） |
| --- | --- | --- |
| **RL 优化目标** | GRPO，优势 `Â=(r−μ)/σ` | **MaxRL，优势 `Â=(r−μ)/μ`** |
| 数据 / 模型 / LoRA / batch / seq | —— 完全一致 —— | —— 完全一致 —— |
| 多轮 + 本地检索 | 有（`max_rollout_turns=3`，BM25） | —— 完全一致 —— |
| 环境 / 奖励 / 裁判 | `QADocsAgentEnv` + qa 奖励 | —— 完全一致 —— |

模型作答协议：检索用 `<search>关键词</search>`；作答把要点放入 `\boxed{...}`（与 v1 同一答案格式，保证判分一致）。

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
DOCS_MAX_CHARS=500           # 单次检索回灌进上下文的总字符上限（短 seq 多轮防 OOM）
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
sf submit maxrl_qwen3.5-9b_qa-rl-agent_v2
```

> 想和 GRPO 版同图对比：v1 与本实验共用 SwanLab project `qa-rl-compare`（曲线名 `grpo-*` vs `maxrl-multiturn-kb`），各自提交即可叠在一张图上看 GRPO vs MaxRL。

## 与 v1 严格对齐的超参（勿动，动了破坏 GRPO vs MaxRL 对比）

本实验目标集群见同目录 `cluster` 文件（`h100`）。除 RL 目标（MaxRL）外，所有可调项都与 v1 逐一对齐：

- `num_prompts_per_step=4` / `num_generations_per_prompt=8` / `train_global_batch_size=32`（与 v1 严格一致）
- `max_total_sequence_length=3072`（与 v1 一致；多轮检索轨迹需要更大预算，H100 80GB 宽裕）
- `max_rollout_turns=3` + `env.qa_docs.cfg.max_turns=3`
- `DOCS_MAX_CHARS` 等检索项与 v1 一致（见上 `DOCS_*` 一节）

> ⚠️ MaxRL 的优势分母是组均值 μ（通过率），μ 很小时优势量级 ~`1/μ` 会比 GRPO 大不少。若早期出现梯度过大/不稳，
> 可适当调高 `loss_fn.reference_policy_kl_penalty`（贴原模型）或依赖已有的 `ratio_clip`/`max_grad_norm` 兜底；
> 论文报告 MaxRL 初期涨得比 GRPO 慢、但后期更高且 pass@k 更健康，验证时多看几个 val 点再下结论。

任务 ~100-200 步即收敛，`val_period=50` 在 50/100/150/200/250 都有验证点。
**仍 OOM 再按序降**：`max_turns→2` → `DOCS_MAX_CHARS→400` → `seq→1280`。**batch 任何时候都不要动。**

## 看多轮检索轨迹

验证时每次会把若干条完整多轮对话（含 `<search>` 与 grep 检索结果）打印到作业日志，直接看日志即可：

```bash
uv run sf logs <JOB_ID>      # 不给 JOB_ID 则跟随最近一个作业
```

## 结论 / 记录

（训练后补：最佳 step、val 准确率、与 baseline 的对比、本地检索是否带来提升、SwanLab 链接、踩坑。）
