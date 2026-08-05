# opsd_qwen3.5-9b_math_v1 — On-Policy Self-Distillation 复现

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

集群共享盘 `/data/datasets/opsd_math/{train,val}.jsonl`（或由服务端注入 `OPSD_DATA_DIR`），每行：

```json
{"problem": "题目", "solution": "参考解全文（喂给老师）", "answer": "最终答案（验证判分用）"}
```

**不走 HF `dataset_name`** —— 集群无外网，在线拉取必然失败。数据入内网见根目录
`scripts/download_models.py` / `download_via_relay.py`。

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
lab submit opsd_qwen3.5-9b_math_v1          # 用实验自带的 h200
```

## 需要盯的指标

| 指标 | 含义 | 异常信号 |
| --- | --- | --- |
| `validation/accuracy` | 主结果 | 应显著快于同预算 GRPO |
| `opsd_kl_clip_frac` | 被截断的 token 占比 | 长期 >5% → clip 太狠；≈0 → 稳定器没起作用 |
| `opsd_hint_len_mean` | 老师 prompt 平均长度 | 贴着 `max_seq_len - response` → 参考解被大量左截断，该调大 seq |
| `opsd_response_len_mean` | 学生解答平均长度 | 贴着 `max_new_tokens` → 大量截断，KL 信号残缺 |

## 结论

（跑完填：SwanLab 链接、与 GRPO baseline 的对比曲线、是否复现论文结论）
