#!/usr/bin/env python
"""把 OPSD 官方数学数据集处理成本仓库实验可读的 jsonl。

OPSD（arXiv:2601.18734）的老师 = 同一个模型 + **额外看到参考解**。所以这份数据与
普通 RL/SFT 数学数据的关键差别只有一条：**每条样本必须带一份完整的参考解**。
只有最终答案是不够的——老师看不到推理过程就退化成和学生一样只看题目，
KL 恒等于 0，整个训练白跑。本脚本因此把「有没有参考解」当成硬过滤条件。

## 数据源

默认用**论文作者自己发布的训练集**（复现首选）：

    siyanzhao/Openthoughts_math_30k_opsd    29434 条

这是官方 `opsd_train.py` 里 `load_dataset(...)` 加载的同一个数据集，其
`data_collator.py` 也正是读 `feature["problem"]` / `feature["solution"]` 两个字段，
与本仓库 `experiments/opsd_qwen3.5-9b_math_h200_1n2g/run.py` 的 schema 完全对齐。

验证集用 AIME24（论文的评测集之一，NeMo-RL 官方 recipe 亦用它）：

    HuggingFaceH4/aime_2024                 30 条，problem/solution/answer

`--train-name` 可换成别的源（如 `agentica-org/DeepScaleR-Preview-Dataset`，
同样是 problem/solution/answer 三字段），但那样就不是在复现论文了，注意区分。

## 关于 answer 字段

官方训练集**没有独立的最终答案列**（有 `Answer`，但按数据卡是另一路 COT 数据的字段，
覆盖不完整）。本脚本从 `solution` 里抽 `\\boxed{...}` 作为 answer，供验证判分用；
抽不到的样本会被丢弃——判不了分的样本留着只会污染 validation/accuracy。

## 输出

    <out>/train.jsonl   {"problem": ..., "solution": <参考解全文>, "answer": <最终答案>}
    <out>/val.jsonl     同上

用法（建议经 CLI：`sf dataset prepare opsd_math`）：
    python common/data/prepare_opsd_math.py                  # 默认取 8000 条
    python common/data/prepare_opsd_math.py --max-train 0     # 全量 29k

⚠️ 要联网拉 HuggingFace，**在能出网的机器上跑**（中继机就行）。
   产物只有几十 MB，rsync 到 H200 即可，不必走模型那套 relay。
"""
import json
import os
import random
import sys

import typer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 抽 \boxed{} 的实现只留一份（含嵌套花括号配对），避免两处实现日后漂移。
from common.rewards.qa_reward import extract_boxed  # noqa: E402

# 参考解短于此基本是「答案本身」或残缺内容，喂给老师没有信息量。
MIN_SOLUTION_CHARS = 80

# 论文评测集（eval/evaluate_math.py 加载的就是这三个）。仅供训练后离线评测，
# 不参与训练中的 in-loop validation——AIME25/HMMT25 没有参考解，也评不了老师。
EVAL_SETS = {
    "aime24": ("HuggingFaceH4/aime_2024", "train"),
    "aime25": ("yentinglin/aime_2025", "train"),
    "hmmt25": ("MathArena/hmmt_feb_2025", "train"),
}


def _clean(row: dict) -> dict | None:
    """规整一条样本；缺参考解 / 参考解过短 / 抽不到答案则返回 None。"""
    problem = (row.get("problem") or "").strip()
    solution = (row.get("solution") or "").strip()
    if not problem or len(solution) < MIN_SOLUTION_CHARS:
        return None
    # 与文件头文档一致：优先从本条 solution 抽 \boxed{}（与参考解必然一致），
    # 列值只作回退 —— 官方训练集的 `Answer` 按数据卡属另一路 COT 数据，
    # 可能与本条 solution 不一致，优先取列会引入错误金标准、验证判分静默失真。
    answer = (extract_boxed(solution) or "").strip() or (row.get("answer") or row.get("Answer") or "").strip()
    if not answer:
        return None
    return {"problem": problem, "solution": solution, "answer": answer}


def _read_jsonl_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(
    out: str = typer.Option(
        os.path.join(REPO_ROOT, "datasets", "opsd_math"),
        help="输出目录（默认 <repo>/datasets/opsd_math）",
    ),
    train_name: str = typer.Option(
        "siyanzhao/Openthoughts_math_30k_opsd",
        help="训练集 HF 名（默认 = OPSD 论文官方训练集）",
    ),
    val_short: str = typer.Option(
        "aime24", help=f"训练中 in-loop 验证用哪个评测集（可选 {'/'.join(EVAL_SETS)}）"
    ),
    max_train: int = typer.Option(
        8000,
        help="训练样本上限（0=全量 29k）。首轮复现不必上全量："
        "OPSD 每个 token 都有监督信号，密度远高于 RL，几千条就能看出趋势",
    ),
    seed: int = typer.Option(42, help="取样随机种子（复现用）"),
) -> None:
    """预处理 OPSD 数学自蒸馏数据 -> problem/solution/answer jsonl。"""
    from datasets import load_dataset

    os.makedirs(out, exist_ok=True)

    # ---- 训练集 ----
    ds = load_dataset(train_name, split="train")
    kept, dropped = [], 0
    for ex in ds:
        rec = _clean(ex)
        if rec is None:
            dropped += 1
        else:
            kept.append(rec)

    print(f"训练集 {train_name}：原始 {len(ds)} 条")
    print(f"  丢弃 {dropped} 条（无参考解 / 参考解 <{MIN_SOLUTION_CHARS} 字符 / 抽不到 \\boxed 答案）")
    if not kept:
        raise SystemExit(
            "过滤后一条都不剩——请确认数据集是否含 problem / solution 字段。"
        )

    if max_train and len(kept) > max_train:
        random.Random(seed).shuffle(kept)
        kept = kept[:max_train]
        print(f"  取样 {max_train} 条（seed={seed}）")

    train_path = os.path.join(out, "train.jsonl")
    _write(train_path, kept)
    print(f"  写入 {len(kept)} 条 -> {train_path}")

    # ---- 评测集（AIME24 / AIME25 / HMMT25）----
    # 只做「学生采样 → 判分」，用不到参考解，故不按 solution 过滤：
    # 否则会因个别题缺解而凭空缩小本就只有 30 题的评测集。
    written: dict[str, int] = {}
    for short, (hf_name, split) in EVAL_SETS.items():
        try:
            eds = load_dataset(hf_name, split=split)
        except Exception as e:  # noqa: BLE001
            # 某个评测集拉不到不该毁掉整次准备——训练用的 val.jsonl 才是必需品。
            print(f"⚠️ 评测集 {short}（{hf_name}）拉取失败，已跳过：{e}")
            continue
        rows = []
        for ex in eds:
            # 字段名各家不同：AIME24/25 用 problem，HMMT25 也是 problem，
            # opencompass 系列用 question——都兜住。
            problem = (ex.get("problem") or ex.get("question") or "").strip()
            answer = str(ex.get("answer") or "").strip()
            if problem and answer:
                rows.append({
                    "problem": problem,
                    "solution": (ex.get("solution") or "").strip(),
                    "answer": answer,
                })
        _write(os.path.join(out, f"eval_{short}.jsonl"), rows)
        written[short] = len(rows)
        print(f"评测集 {short}（{hf_name}）：{len(rows)} 条 -> eval_{short}.jsonl")

    # 训练中的 in-loop validation 用 AIME24（论文评测集之一，且它带 solution）。
    val_src = os.path.join(out, f"eval_{val_short}.jsonl")
    if not os.path.isfile(val_src):
        raise SystemExit(
            f"训练需要 val.jsonl，但评测集 {val_short} 没拉到。"
            "请检查网络后重跑，或用 --val-short 换一个已成功的评测集。"
        )
    val_path = os.path.join(out, "val.jsonl")
    _write(val_path, _read_jsonl_rows(val_src))
    print(f"in-loop 验证集 = {val_short}（{written.get(val_short, 0)} 题）-> val.jsonl")

    total_mb = sum(
        os.path.getsize(os.path.join(out, n))
        for n in os.listdir(out)
        if n.endswith(".jsonl")
    ) / 1e6
    avg_sol = sum(len(r["solution"]) for r in kept) / len(kept)

    print(f"\n完成。共 {total_mb:.1f} MB，参考解平均 {avg_sol:.0f} 字符。")
    print("\n下一步 —— 拷到 H200（几十 MB，直接 rsync，不必走模型那套 relay）：")
    print(f"  rsync -avP {out}/ <用户>@<h200>:/data/datasets/opsd_math/")
    print("\n实验 config 已写死 data_dir=/data/datasets/opsd_math；")
    print("要放别处就改 config.yaml 的 data.data_dir（集群绝对路径）。")
    print("\neval_*.jsonl 供训练后离线评测（对照论文的 AIME24/AIME25/HMMT25 三列），")
    print("不参与训练中的 in-loop validation，不会拖慢训练。")


if __name__ == "__main__":
    typer.run(main)
