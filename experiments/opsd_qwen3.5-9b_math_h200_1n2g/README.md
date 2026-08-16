# opsd_qwen3.5-9b_math_h200_1n2g — On-Policy Self-Distillation 复现

复现论文 **《Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models》**
（arXiv:2601.18734，官方实现 <https://github.com/siyan-zhao/OPSD>）。

## 目标

验证「不需要更强的外部教师，模型可以给自己当老师」这个论断在自有 9B base 模型 + 自有题库上是否成立，
以及相对 GRPO 的样本效率差距。

## 方法

```
学生 π(· | x)        只看题目 x，on-policy 采样出解答 y ~ π(·|x)
老师 π(· | x, y*)    额外看到参考解 y*，对同一条 y 做 teacher-forcing 前向
损失  L = Σ_t KL( π(·|x,y*)_t ‖ π(·|x)_t )        t 只取 response token
```

老师和学生是**同一份权重**（`opsd.teacher_mode: self`），差别只在上下文里有没有参考解。
相比 GRPO 只有一个标量奖励，这里每个 token 都拿到一个完整分布，信号密度高几个数量级；
相比普通 SFT/蒸馏，被监督的 y 是学生自己采样出来的，不存在 exposure bias。

## 实现路径（为什么不用官方那套）

官方基于 TRL 的实验性 GOLD trainer + flash-attn 2.8.3。我们的 Ray 集群跑在 NeMo-RL 官方容器里、
只有内网，装新包要动镜像。而 NeMo-RL 0.7.0 **自带 on-policy distillation 主循环**
（`nemo_rl/algorithms/distillation.py`），它与 OPSD 的唯一差别是「老师吃的输入不一样」。

所以本实验走的是 `common/algorithms/opsd.py`：不 fork 主循环，只做三处外科手术
（老师输入重建与位置重对齐 / rollout batch 暂存 / 自蒸馏时跳过第二份模型加载）。
**零新增依赖、零镜像变更。** 索引对齐的推导写在 `realign_topk` 的 docstring 里，
配套单测 `tests/test_opsd.py`（11 条，CPU 即可跑）。

## 数据

用**论文作者发布的官方训练集** `siyanzhao/Openthoughts_math_30k_opsd`（29434 条），
就是官方 `opsd_train.py` 加载的那个；其 `data_collator.py` 读的也正是 `problem` / `solution`
两个字段，与本实验 schema 一致。

集群共享盘 `/data/datasets/opsd_math/`（`config.yaml` 的 `data.data_dir`）：

| 文件 | 用途 |
| --- | --- |
| `train.jsonl` | 训练 |
| `val.jsonl` | 训练中 in-loop 验证（= AIME24） |
| `eval_aime24/25.jsonl`、`eval_hmmt25.jsonl` | 训练后离线评测，对照论文三列 |

每行 `{"problem": ..., "solution": <参考解全文>, "answer": <最终答案>}`。

**不走 HF `dataset_name`** —— 集群无外网，在线拉取必然失败。生成与投放见
`datasets/opsd_math/README.md`（`lab prepare opsd_math` + rsync）。

## 关键参数

| 参数 | 作用 | 备注 |
| --- | --- | --- |
| `opsd.teacher_mode` | `self` / `fixed` | `self` 是论文主设定，省一整份模型显存 |
| `opsd.per_token_kl_clip` | 单 token KL 上限 | 论文的稳定器；看 `opsd_kl_clip_frac` 调 |
| `loss_fn.kl_type` | `reverse` | mode-seeking，推理任务通常正确路径少 |
| `distillation.topk_logits_k` | 老师传给学生的分布宽度 | 64 起步 |
| `policy.max_total_sequence_length` | **要装下题目+参考解+解答** | 老师序列比学生长 |

⚠️ `distillation.max_rollout_turns` 必须为 1。OPSD 的老师要把 prompt 整段换成「题目+参考解」，
多轮轨迹里 response 被工具结果打断，这个操作无定义 —— `opsd.py` 会直接报错而不是猜一个对齐。

## 跑

```bash
forge submit opsd_qwen3.5-9b_math_h200_1n2g          # 用实验自带的 h200-2g
```

## 训练后评测（对照论文表格）

```bash
forge eval opsd_qwen3.5-9b_math_h200_1n2g --run-id <训练 run_id>
# 只评一个集 / 改协议参数（-- 之后原样透传给 eval.py）：
forge eval opsd_qwen3.5-9b_math_h200_1n2g --run-id <run_id> -- --datasets aime24 --n 4
```

本目录的 `eval.py` 由 OPSD recipe 的显式 eval 生命周期入口调用。
它按论文协议（每题 12 条 / temp 1.0 / max_tokens 38912）在三个评测集上打分，
输出可与论文表格逐项对照的 `avg@N / pass@N / majority@N`。

**判分和训练中的 validation 是同一套实现**（`common/eval/math_eval.py` →
`common/algorithms/opsd_eval.py`，判分器都用 NeMo-RL 的 `HFVerifyWorker`），
所以训练曲线上的数和最终报告里的数可比。分开写两套评测最容易出的事故不是算错，
而是两边算得都对但口径不同，最后说不清差异从哪来。

| | in-loop validation | `forge eval` |
| --- | --- | --- |
| 时机 | 训练中每 20 步 | 训练结束后 |
| 评测集 | 只 AIME24（省时间） | AIME24 + AIME25 + HMMT25 |
| 生成长度 | 受训练 `max_total_sequence_length` 约束 | 38912（论文口径） |
| 用途 | 看趋势 | 出最终数字 |

## 需要盯的指标

验证按官方 `eval/run_eval.sh` 的口径做：AIME24 每题采 12 条（`data.val_repeat: 12`），
按题聚合出三个指标。AIME 只有 30 题，每题采 1 条时分辨率是 3.3%，
相邻两次验证光靠噪声就能差 6~10 个点 —— 所以 repeat 不是可选项。

| 指标 | 含义 | 怎么读 |
| --- | --- | --- |
| `validation/avg_at_n` | 总正确/总生成 | 主结果，等价于 `accuracy` |
| `validation/pass_at_n` | 至少一条对的题占比 | 上界。它高而 avg 低 = **会做但不稳**，正是蒸馏该改善的 |
| `validation/majority_at_n` | 多数投票正确的题占比 | 最接近实际部署口径 |
| `validation/unparsed_answer_rate` | 没写出 `\boxed{}` 的比例 | 偏高说明分数低是**格式问题**不是能力问题 |
| `validation/samples_per_problem` | 每题实际采样数 | 应为 12。是 1 说明 `val_repeat` 没生效 |
| `opsd_kl_clip_frac` | 被截断的 token 占比 | 长期 >5% → clip 太狠；≈0 → 稳定器没起作用 |
| `opsd_hint_len_mean` | 老师 prompt 平均长度 | 贴着 `max_seq_len - response` → 参考解被大量左截断，该调大 seq |
| `opsd_response_len_mean` | 学生解答平均长度 | 贴着 `max_new_tokens` → 大量截断，KL 信号残缺 |

`pass@N` 与 `avg@N` 一起涨，才是真的学会了；只有 `avg@N` 涨而 `pass@N` 不动，
说明只是把已经会做的题做得更稳，能力边界没推开。

> `majority@N` 的正确性判定不靠字符串匹配：归一化只用来**分组**找众数，
> 该组是否正确取的是数学判分器给的真实 reward。所以归一化再粗糙也只会低估，
> 不会把错答案判成对的。实现见 `common/algorithms/opsd_eval.py`。

## 结论

（跑完填：SwanLab 链接、与 GRPO baseline 的对比曲线、是否复现论文结论）
