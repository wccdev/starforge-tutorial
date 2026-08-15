"""TRL GRPO 工具调用训练入口：题库 + search_docs 检索本地文档后作答。

TRL 官方形态（GRPOTrainer 文档「Agent Training」/「Using a custom reward function」）：
  - 工具 = 普通 Python 函数（类型注解 + Google docstring 必须齐全，schema 由此推断），
    经 `tools=[...]` 传给 Trainer，模型在生成中多轮调用、结果以 tool 消息回灌。
  - 奖励 = 可调用对象列表 `reward_funcs`，签名用 **kwargs 兜底，返回 list[float]；
    数据集除 prompt 外的列（这里是 answer）会按列名作为关键字参数传入。
  - 有按轨迹状态/环境自产任务的场景才用 environment_factory（本实验数据集驱动，用 tools 即可）。

平台契约：recipe 入口即本文件，adapter 传入
  --config --model --train-data --validation-data --output-dir --overrides-json
判分口径与 nemo-rl / verl 两个对照实验一致：只看 \\boxed{} 最终答案，检索不给分。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# ───────────────────────────── 检索工具（与 verl 版 tools.py 同语义）─────────────────────────────

_MAX_CHARS = int(os.environ.get("DOCS_MAX_CHARS", "900"))
_MAX_FILES = 3
_CONTEXT_LINES = 2


def search_docs(query: str) -> str:
    """在公司技术资料库（本地 markdown 文档）中检索关键词，返回带出处的相关片段。

    Args:
        query: 检索关键词，只保留核心术语，如 "CMP 铜 去除 步骤"。

    Returns:
        带文件名出处的资料片段；未命中或资料目录不存在时返回提示文本。
    """
    docs_dir = Path(os.environ.get("DOCS_DIR", "/data/docs"))
    if not docs_dir.is_dir():
        return f"[检索结果] 资料目录 {docs_dir} 不存在，请基于已有知识作答。"
    terms = [t.lower() for t in re.split(r"[\s,，、]+", query.strip()) if t]
    if not terms:
        return "[检索结果] 检索词为空，请提供核心术语。"

    hits: list[tuple[int, str, list[str]]] = []
    for md in sorted(docs_dir.rglob("*.md")):
        lines = md.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            score = sum(1 for t in terms if t in line.lower())
            if score > 0:
                lo, hi = max(0, i - _CONTEXT_LINES), min(len(lines), i + _CONTEXT_LINES + 1)
                snippet = [f"L{j + 1}: {lines[j]}" for j in range(lo, hi) if lines[j].strip()]
                hits.append((score, md.name, snippet))
    if not hits:
        return f"[检索结果] 未找到与「{query}」相关的资料，可换关键词再试一次或直接作答。"

    hits.sort(key=lambda h: -h[0])
    out, seen = [], set()
    for _, name, snippet in hits:
        if name in seen:
            continue
        seen.add(name)
        out.append(f"【{name}】\n" + "\n".join(snippet))
        if len(seen) >= _MAX_FILES:
            break
    return ("[检索结果]\n" + "\n\n".join(out))[:_MAX_CHARS]


# ───────────────────────────── 判分（与 nemo-rl / verl 对照实验同口径）─────────────────────────────

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
FORMAT_PENALTY = -0.5


def _normalize(text: str) -> str:
    return re.sub(r"[\s,，、]+", "", text.strip().lower())


def _last_assistant_text(completion) -> str:
    """conversational 格式的 completion 是消息列表；取模型最后一条 assistant 文本。"""
    if isinstance(completion, list):
        texts = [m.get("content") or "" for m in completion if m.get("role") == "assistant"]
        return texts[-1] if texts else ""
    return str(completion)


def score_one(text: str, truth: str) -> float:
    matches = _BOXED.findall(text or "")
    if not matches:
        return FORMAT_PENALTY
    answer, want = _normalize(matches[-1]), _normalize(truth)
    if not want:
        return 0.0
    if re.fullmatch(r"[a-h](?:[a-h])+", want):
        got = set(answer)
        if not got:
            return 0.0
        if got == set(want):
            return 1.0
        if got <= set(want):
            return len(got) / len(want) * 0.5
        return 0.0
    return 1.0 if answer == want else 0.0


def qa_boxed_reward(completions, answer, **kwargs):
    """只看 \\boxed{} 最终答案；工具调用本身不给分，保证与另两条路线 A/B 可比。"""
    return [
        score_one(_last_assistant_text(completion), str(target))
        for completion, target in zip(completions, answer, strict=True)
    ]


# ───────────────────────────── 数据与入口 ─────────────────────────────

SYSTEM_PROMPT = (
    "回答前可调用 search_docs 工具在公司技术资料库检索依据；"
    "检索词只保留核心术语，通常一次定向检索后就应作答。"
    "最终答案必须放入 \\boxed{...}（单选/多选题填选项字母，如 \\boxed{B} 或 \\boxed{A,C}）。"
)


def _dataset(path: str):
    """题库 jsonl/parquet（{"query", "expected_answer"}）→ conversational prompt + answer 列。

    工具循环要求 conversational 格式（消息列表），tool 消息才能按 chat template 回灌。
    """
    kind = "parquet" if Path(path).suffix == ".parquet" else "json"
    dataset = load_dataset(kind, data_files=path, split="train")
    if not {"query", "expected_answer"} <= set(dataset.column_names):
        raise ValueError("题库需要 query / expected_answer 两列（与 nemo-rl 对照实验同源）")
    return dataset.map(
        lambda item: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["query"]},
            ],
            "answer": item["expected_answer"],
        },
        remove_columns=[c for c in dataset.column_names if c not in ("prompt", "answer")],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overrides-json", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    config.update(json.loads(args.overrides_json))
    config["output_dir"] = args.output_dir
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=qa_boxed_reward,
        tools=[search_docs],
        args=GRPOConfig(**config),
        train_dataset=_dataset(args.train_data),
        eval_dataset=_dataset(args.validation_data),
    )
    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final_model"))


if __name__ == "__main__":
    main()
