# opsd_math — OPSD 自蒸馏训练数据

供 `experiments/opsd_qwen3.5-9b_math_h200_1n2g` 使用。

## 来源

| | 数据集 | 条数 | 产出文件 |
| --- | --- | --- | --- |
| 训练 | `siyanzhao/Openthoughts_math_30k_opsd` | 29434 | `train.jsonl` |
| 评测 | `HuggingFaceH4/aime_2024` | 30 | `eval_aime24.jsonl` |
| 评测 | `yentinglin/aime_2025` | 30 | `eval_aime25.jsonl` |
| 评测 | `MathArena/hmmt_feb_2025` | 30 | `eval_hmmt25.jsonl` |

三个评测集就是论文 `eval/evaluate_math.py` 加载的那三个，用于训练后离线评测、
逐项对照论文数字。其中 AIME24 另存一份为 `val.jsonl`，作训练中的 in-loop 验证
（`--val-short` 可换）。AIME25/HMMT25 不进 in-loop：HMMT25 没有 `solution` 字段，
而且每多一个评测集，训练中的验证就多花一份时间。

训练集就是官方 `opsd_train.py` 里 `load_dataset(...)` 加载的那个；其 `data_collator.py`
读的也正是 `problem` / `solution` 两个字段，与本仓库 `run.py` 的 schema 一致。

> 换成 `agentica-org/DeepScaleR-Preview-Dataset`（同为 problem/solution/answer）也能跑，
> 但那就不是在复现论文了，注意在实验 README 里写清楚。

## 字段

```json
{"problem": "题目", "solution": "参考解全文（喂给老师）", "answer": "最终答案（验证判分用）"}
```

**`solution` 是这份数据的核心。** OPSD 的老师 = 同一个模型 + 额外看到参考解；
只有最终答案的数据集用不了——老师看不到推理过程就退化成和学生一样只看题目，
KL 恒等于 0，训练白跑。`prepare_opsd_math.py` 因此把「有无参考解」当硬过滤条件。

`answer` 官方训练集里没有独立列，由脚本从 `solution` 的最后一个 `\boxed{...}` 抽取
（正确处理 `\boxed{\frac{1}{2}}` 这类嵌套花括号）；抽不到的样本丢弃——判不了分的样本
留着只会污染 `validation/accuracy`。

## 生成

在**能出网的机器上**（中继机即可）：

```bash
lab prepare opsd_math                      # 默认取 8000 条训练样本
lab prepare opsd_math -- --max-train 0     # 全量 29k
```

## 放到集群

产物只有几十 MB，直接 rsync 即可，不必走模型那套 relay：

```bash
rsync -avP datasets/opsd_math/ <用户>@<h200>:/data/datasets/opsd_math/
```

路径已写死在实验 `config.yaml` 的 `data.data_dir`；要放别处就改那个字段为集群绝对路径。

## License

两个上游数据集的 license 以其 HuggingFace 数据卡为准，内部使用前请自行确认。
实际数据文件不入库（见 `datasets/README.md` 的约定）。
