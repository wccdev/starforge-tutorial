#!/usr/bin/env python
r"""把题库 RL 数据整理成单轮 GRPO（QARewardEnv）可用的干净 jsonl。

输入（默认 datasets/qa_rl/raw/，即从旧项目复制过来的原始文件）：
    train.jsonl / val.jsonl  —— 已是 {"query", "expected_answer": "[type] ...", "meta"} 格式
    shortanswer.jsonl        —— 另一种格式 {"query", "reference_keywords": [...], ...}

处理：
    - train/val：原样校验通过即透传（必须有 query + expected_answer）。
    - shortanswer：转成 {"query": <加简答指令>, "expected_answer": "[short] kw1 ||| kw2 ..."}，
      关键词列表用 " ||| " 连接（同一要点的多种写法在原数据里已用 "/" 写好，reward 会自动拆）。
      无关键词的样本跳过（没法判分）。

输出（默认 datasets/qa_rl/）：
    train.jsonl  —— 客观题 train + 转换后的简答题（默认合并，让训练覆盖 LLM 裁判路径）
    val.jsonl    —— 客观题 val（不含简答，验证更快、无裁判开销）
    short.jsonl  —— 单独的简答题（备查 / 离线评估用）

用法（建议经 CLI：`lab prepare qa_rl`，在项目 uv 环境里跑）：
    python common/data/prepare_qa_rl.py                 # 用默认路径
    python common/data/prepare_qa_rl.py --no-merge-short # 简答不并入 train（只写 short.jsonl）
之后上传为平台数据集版本（实验 config 用 data.train.dataset 引用，提交时自动分发）：
    sf dataset push qa-rl <新版本> <输出目录>
本地干跑才需要 export QA_RL_DATA_DIR=<repo>/datasets/qa_rl。
"""
import json
import os

import typer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SHORT_PROMPT = (
    "下面是一道简答题，请先简要作答，再把你认为的关键要点放入 \\boxed{{}}，"
    "多个要点用 “;” 分隔（如 \\boxed{{要点1; 要点2}}）。\n\n题目：{q}"
)


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _passthrough(rows: list[dict], src_name: str) -> list[dict]:
    """校验客观题数据：必须有 query 与 expected_answer。"""
    out = []
    for i, r in enumerate(rows):
        if not r.get("query") or not r.get("expected_answer"):
            raise ValueError(f"{src_name} 第 {i} 行缺少 query/expected_answer: {r!r:.120}")
        out.append(r)
    return out


def _convert_short(rows: list[dict]) -> list[dict]:
    """把 shortanswer 记录转成 [short] 格式；无关键词的跳过。"""
    out = []
    skipped = 0
    for r in rows:
        query = (r.get("query") or "").strip()
        kws = [k.strip() for k in (r.get("reference_keywords") or []) if k and k.strip()]
        if not query or not kws:
            skipped += 1
            continue
        out.append(
            {
                "query": SHORT_PROMPT.format(q=query),
                "expected_answer": "[short] " + " ||| ".join(kws),
                "meta": {"type": "short", "source": "shortanswer", "num_keywords": len(kws)},
            }
        )
    if skipped:
        print(f"  简答题跳过 {skipped} 条（无 query 或无关键词）")
    return out


def main(
    src: str = typer.Option(
        os.path.join(REPO_ROOT, "datasets", "qa_rl", "raw"),
        help="原始数据目录（含 train/val/shortanswer.jsonl）",
    ),
    out: str = typer.Option(
        os.path.join(REPO_ROOT, "datasets", "qa_rl"), help="输出目录"
    ),
    merge_short: bool = typer.Option(
        True,
        "--merge-short/--no-merge-short",
        help="简答题是否并入 train（--no-merge-short 则只单独写 short.jsonl）",
    ),
) -> None:
    """预处理题库 RL 数据 -> 单轮 GRPO jsonl。"""
    os.makedirs(out, exist_ok=True)

    train = _passthrough(_read_jsonl(os.path.join(src, "train.jsonl")), "train.jsonl")
    val = _passthrough(_read_jsonl(os.path.join(src, "val.jsonl")), "val.jsonl")

    short_path = os.path.join(src, "shortanswer.jsonl")
    short = _convert_short(_read_jsonl(short_path)) if os.path.exists(short_path) else []

    if short and merge_short:
        train = train + short

    _write_jsonl(os.path.join(out, "train.jsonl"), train)
    _write_jsonl(os.path.join(out, "val.jsonl"), val)
    if short:
        _write_jsonl(os.path.join(out, "short.jsonl"), short)

    print(f"train.jsonl : {len(train)} 条" + ("（含简答）" if short and merge_short else ""))
    print(f"val.jsonl   : {len(val)} 条（客观题）")
    if short:
        print(f"short.jsonl : {len(short)} 条（简答题，备查/评估）")
    print("\n完成。上传为平台数据集新版本（版本不可变，更新数据请递增版本号）：")
    print(f"  uv run sf dataset push qa-rl <新版本> {out}")
    print("然后把实验 config 的 data.train.dataset 指到新版本；本地干跑才需要：")
    print(f"  export QA_RL_DATA_DIR={out}")


if __name__ == "__main__":
    typer.run(main)
