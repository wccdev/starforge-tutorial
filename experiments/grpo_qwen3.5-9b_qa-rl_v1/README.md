# grpo_qwen3.5-9b_qa-rl_v1（对比实验 · 实验一 / baseline）

用 **GRPO** 在自有**技术培训考题题库**上强化训练 **Qwen 3.5 9B**。单轮：模型答一道题，环境判分即结束。

> **这是 A/B 对比的基线组**：单轮、**无工具**。
> 对照组（实验二 / treatment）= [`grpo_qwen3.5-9b_qa-rl-agent_v1`](../grpo_qwen3.5-9b_qa-rl-agent_v1)：**多轮 + 本地文档检索工具**（容器内 grep `/data/docs` 的 markdown）。
> 两个实验共用同一数据集 / 模型 / LoRA / batch / seq / 裁判奖励，**唯一变量**是「能否多轮检索本地资料回答」。
> 先跑本实验拿到 baseline 曲线，再跑实验二对比。

## 目标

- 提升模型在公司技术培训考题（单选/多选/判断/填空/简答）上的作答准确率。
- 客观题用规则判分；**简答题用 LLM-as-judge**（裁判 LLM 打 0~1 分，端点连不上自动回退关键词覆盖率）。

## 组成

| 部分 | 位置 |
| --- | --- |
| 判分逻辑 | `common/rewards/`（`qa_reward.py` 规则 + `qa_judge_reward.py` 裁判，`synonyms.json` 同义词） |
| 奖励环境 | `common/environments/qa_env.py` 的 `QARewardEnv`（单轮，包装上面的判分） |
| 数据 | `datasets/qa_rl/`（由 `common/data/prepare_qa_rl.py` 从 `raw/` 生成） |
| 启动 | 本目录 `run.py`（自建数据集 + 实例化环境 + grpo_train）；`config.yaml` 写差异 |

数据答案格式（`expected_answer` 带 `[type]` 前缀）见 `common/rewards/README.md`。

## 跑起来（`forge submit` 经中心化服务到集群）

**前置 1 · 题库数据要在集群上。** `datasets/qa_rl/` 被 `.gitignore`（公司题库），作业上传 working_dir 尊重 .gitignore → **不会自动上传**。先把题库放到集群共享盘，并在 `config.yaml` 的 `data.data_dir` 指过去（当前已写死 `/data/datasets/qa_rl`；也可由服务端注入 `QA_RL_DATA_DIR` 覆盖）：

```bash
# 在本机：先本地预处理（若 datasets/qa_rl 还没 train/val.jsonl）
lab prepare qa_rl
# 再把 *.jsonl 放到集群共享盘的 /data/datasets/qa_rl（与 config 的 data_dir 一致）
```

**前置 2 · 简答裁判 LLM。** 裁判端点（`JUDGE_BASE_URL` / `JUDGE_MODEL` / `JUDGE_API_KEY`）由中心化服务在集群侧注入到作业，本仓库不入库这些值。

> ⚠️ `JUDGE_MODEL` 要填对：在**集群容器**里 `curl -H "Authorization: Bearer <key>" <JUDGE_BASE_URL>/models` 看 `data[].id`。
> 填错或端点连不上不会让训练崩——简答会自动**回退到关键词覆盖率**（只是判分变糙）。
> 不想用裁判：把 `config.yaml` 的 `env.qa.cfg.use_judge` 设 `false`，全部走规则判分、零成本。

**提交：**

```bash
forge submit grpo_qwen3.5-9b_qa-rl_v1
```

> 注意：环境（Ray actor）需要能 `import common.*`，本仓库根目录会随作业上传到集群作业工作目录（服务端打包 working-dir）。

## 关键超参 / 调参入口（`config.yaml`）

- 后端：Megatron-Core + **LoRA**（继承 `grpo_megatron` + `grpo_lora`；低显存实测起点 lr 1e-4/dim8）。回全参数：删 `defaults` 里 `grpo_lora.yaml`。
- batch：`num_prompts_per_step=4` / `num_generations_per_prompt=8` / `train_global_batch_size=32`（须整除 prompts×gen）/ `micro=1` / `seq=1536`。
- `loss_fn.reference_policy_kl_penalty`：KL 约束强度。
- `policy.max_total_sequence_length`：多选题带选项较长，按显存调；OOM 就降。
- `env.qa.cfg.use_judge`：简答是否走裁判 LLM。
- `logger.swanlab.*`：云端日志项目/run 名。

## 想纯规则、零成本？

把 `env.qa.cfg.use_judge` 设为 `false`，全部走规则判分（简答=关键词覆盖率），不需要裁判端点。

## 结论 / 记录

（训练后补：最佳 step、val 准确率、SwanLab 链接、踩坑。）
