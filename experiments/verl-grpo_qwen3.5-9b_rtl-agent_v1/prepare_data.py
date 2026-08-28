#!/usr/bin/env python
"""把 RTL 题库 jsonl 转成 verl RLHFDataset 需要的 parquet（本地一次性运行）。

用法：
  python experiments/verl-grpo_qwen3.5-9b_rtl-agent_v1/prepare_data.py \
      --data-dir <datasets/rtl_rl 路径> --out-dir <输出目录>
  然后提交时 --train-data <out>/train.parquet --validation-data <out>/val.parquet。

关键 schema（对照 verl examples/data_preprocess/gsm8k_tool_agent_loop.py）：
  - prompt        chat 消息列表（config 已开 data.return_raw_chat）
  - agent_name    ★必须 = "tool_agent"：异步模式按它路由 Agent Loop，
                  缺了会静默回落 single_turn_agent、工具永远不触发（issue #2986）
  - reward_model.ground_truth  ★这道题的**隐藏 testbench 源码**，整段字符串透传。
                  它不进 prompt，agent 全程看不到 —— 见 tools.py 里为什么没有
                  run_testbench 工具。
  - extra_info.top             顶层模块名，yosys 综合要用（留空则 -auto-top）
  - data_source   奖励路由用的数据集名

输入 jsonl 每行：
  {"spec": "设计一个 4 位同步计数器……", "testbench": "module tb; … endmodule", "top": "counter"}

题库从哪来
──────────────────────────────────────────────────────────────────────────────
不要把 VerilogEval / RTLLM 的题目复制进这个仓库当训练集：它们是**评测集**，
训练用了就等于泄题，之后 sf bench 的分数不再说明任何事情（平台的数据集互查会
把这类重叠标出来）。自建题库、或用它们的**训练划分**，评测留给 harness。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# 只讲工具用法与输出格式。设计要求在题面（spec）里，system 里再写一套会互相矛盾。
SYSTEM_PROMPT = (
    "你是数字电路设计工程师。用 SystemVerilog 实现题目要求的模块。\n"
    "可以调用 compile_rtl 检查语法、调用 lint_rtl 检查锁存器与位宽问题；"
    "带着报错改一版再试，通常两三轮内就该收敛。\n"
    "最终回复里必须给出完整的模块源码，放在 ```verilog 代码块里，"
    "从 module 写到 endmodule，不要只给片段或 diff。"
)


def _rows(path: Path, split: str):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("testbench", "").strip():
            # 没有 testbench 的题，奖励的 simulation 段（0.7）无从算起，
            # 整条奖励会退化成「编译得过就给 0.1」。宁可在这里炸，也别让它混进训练集。
            raise SystemExit(f"{path}:{i + 1} 缺少 testbench")
        yield {
            "data_source": "rtl_rl",
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["spec"]},
            ],
            "reward_model": {"style": "rule", "ground_truth": item["testbench"]},
            "extra_info": {"split": split, "index": i, "top": item.get("top", "")},
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
