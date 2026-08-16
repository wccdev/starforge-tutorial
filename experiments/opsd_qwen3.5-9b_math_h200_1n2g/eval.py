#!/usr/bin/env python
# OPSD 训练后离线评测：按论文协议在 AIME24 / AIME25 / HMMT25 上给 checkpoint 打分。
#
# 由 OPSD recipe 的 eval 生命周期动作通过统一 launcher 调起：
#     forge eval opsd_qwen3.5-9b_math_h200_1n2g --run-id <训练 run_id>
#
# 与训练中的 in-loop validation 的分工：
#   in-loop  只跑 AIME24、每题 12 条、受训练 max_seq 约束 —— 用来在训练途中看趋势
#   本脚本   三个评测集、可放到论文的 38912 token —— 用来出最终结果、和论文表格逐项对照
# 两者共用同一套判分与指标实现（common/eval/math_eval.py → common/algorithms/opsd_eval.py），
# 所以训练曲线上的数和最终报告里的数是可比的。

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.eval.math_eval import (  # noqa: E402
    EvalSpec,
    discover_eval_files,
    evaluate_dataset,
    format_report,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPSD 离线评测（AIME24 / AIME25 / HMMT25）")
    p.add_argument("--model", required=True, help="HF 格式模型路径（artifact manifest 已登记）")
    p.add_argument(
        "--data-dir",
        default=os.environ.get("OPSD_DATA_DIR") or "/data/datasets/opsd_math",
        help="评测集目录（内含 eval_<名字>.jsonl）",
    )
    p.add_argument("--datasets", default="", help="只评这些（逗号分隔，如 aime24,aime25）；空=全部")
    p.add_argument("--n", type=int, default=12, help="每题采样数（论文 --val_n 12）")
    p.add_argument("--temperature", type=float, default=1.0, help="论文 1.0；thinking 模式 0.6")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--max-tokens", type=int, default=38912,
        help="论文默认。竞赛题思考链很长，砍小会系统性低估分数",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tp", type=int, default=int(os.environ.get("FORGE_CLUSTER_GPUS_PER_NODE") or 1),
        help="vLLM tensor_parallel_size（默认取服务端下发的每节点卡数）",
    )
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    files = discover_eval_files(args.data_dir)
    if args.datasets:
        want = [s.strip() for s in args.datasets.split(",") if s.strip()]
        missing = [w for w in want if w not in files]
        if missing:
            raise SystemExit(
                f"这些评测集不在 {args.data_dir}：{', '.join(missing)}；"
                f"可用的有：{', '.join(files) or '（无）'}。"
                "请先在能出网的机器上 `lab prepare opsd_math` 再 rsync 过来。"
            )
        files = {k: files[k] for k in want}
    if not files:
        raise SystemExit(
            f"{args.data_dir} 下没有 eval_*.jsonl。"
            "请先 `lab prepare opsd_math` 生成，再 rsync 到集群。"
        )

    spec = EvalSpec(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(f"[eval] model    : {args.model}")
    print(f"[eval] datasets : {', '.join(files)}")
    print(
        f"[eval] protocol : n={spec.n} temp={spec.temperature} top_p={spec.top_p} "
        f"top_k={spec.top_k} max_tokens={spec.max_tokens} tp={args.tp}"
    )

    # 上报到 console：跑不通也不该让评测失败（本地直跑时本就没有 STARFORGE_* 凭据）。
    from common.observability import report

    report.init(
        hparams={
            "eval_model": args.model, "eval_n": spec.n,
            "eval_temperature": spec.temperature, "eval_max_tokens": spec.max_tokens,
        },
        monitor_hardware=False,  # 评测是短任务，硬件曲线意义不大
    )

    import ray
    from transformers import AutoTokenizer
    from vllm import LLM

    ray.init(ignore_reinit_error=True)  # 判分 worker 要用
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 一个 9B 模型加载要几十秒到几分钟，三个评测集共用同一个实例。
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=spec.max_tokens + 4096,  # 留出题面 + chat template 的长度
        enforce_eager=True,  # 与训练侧一致：规避 Qwen3.5 + vLLM CUDA-graph 的间歇 hang
    )

    results: dict[str, dict] = {}
    for name, path in files.items():
        rows = read_jsonl(path)
        print(f"\n[eval] ▶ {name}：{len(rows)} 题 × {spec.n} 条 ...", flush=True)
        metrics = evaluate_dataset(llm, tokenizer, rows, spec)
        results[name] = metrics
        print("[eval] " + format_report(name, metrics), flush=True)
        # 每个评测集一个 step，指标在 console 上按数据集分开看
        report.log(metrics, step=len(results), prefix=f"eval/{name}")

    print("\n" + "=" * 72)
    print("最终结果（可与论文表格逐项对照）")
    print("=" * 72)
    for name, metrics in results.items():
        print(format_report(name, metrics))
    print("=" * 72)

    report.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
