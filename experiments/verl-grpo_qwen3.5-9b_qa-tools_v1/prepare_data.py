#!/usr/bin/env python
"""把题库 jsonl 转成 verl RLHFDataset 需要的 parquet（本地一次性运行）。

用法：
  python experiments/verl-grpo_qwen3.5-9b_qa-tools_v1/prepare_data.py \
      --data-dir <datasets/qa_rl 路径> --out-dir <输出目录>
  然后提交时 --train-data <out>/train.parquet --validation-data <out>/val.parquet。

关键 schema（对照 verl examples/data_preprocess/gsm8k_tool_agent_loop.py）：
  - prompt        chat 消息列表（config 已开 data.return_raw_chat）
  - agent_name    ★必须 = "tool_agent"：异步模式按它路由 Agent Loop，
                  缺了会静默回落 single_turn_agent、工具永远不触发（issue #2986）
  - reward_model.ground_truth  ★带 [type] 前缀原样透传（"[single] A" / "[multiple] A,B"
                  / "[fill] a ||| b" / "[short] kw1 ||| kw2"）。reward.py 复用
                  common/rewards 按前缀分派判分，前缀【不能】在这里剥掉
  - data_source   奖励路由用的数据集名
输入 jsonl 每行 {"query": ..., "expected_answer": "[type] ..."}，与 nemo-rl 对照实验同源。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# 只讲工具用法。答案格式不在这里说——题面（query）已按题型自带 \boxed{} 指令
# （单选填字母、填空按空用 ; 分隔、简答列要点），system 里再写一套会互相矛盾。
SYSTEM_PROMPT = (
    "回答前可调用 search_docs 工具在公司技术资料库检索依据；"
    "检索词只保留核心术语，通常一次定向检索后就应作答。"
    "最终答案必须按题目要求的格式放入 \\boxed{...}。"
)


def _rows(path: Path, split: str):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        yield {
            "data_source": "qa_rl",
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["query"]},
            ],
            "reward_model": {"style": "rule", "ground_truth": item["expected_answer"]},
            "extra_info": {"split": split, "index": i},
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="含 train.jsonl / val.jsonl 的题库目录")
    parser.add_argument("--out-dir", required=True, help="parquet 输出目录")
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        src = data_dir / f"{split}.jsonl"
        if not src.is_file():
            raise SystemExit(f"缺少 {src}")
        df = pd.DataFrame(list(_rows(src, split)))
        df.to_parquet(out_dir / f"{split}.parquet")
        print(f"{split}: {len(df)} 行 -> {out_dir / f'{split}.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
