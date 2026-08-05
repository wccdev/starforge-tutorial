"""数学推理离线评测：每题采 N 条 → 判分 → avg@N / pass@N / majority@N。

与训练中的 in-loop validation 是**同一套口径**，差别只在协议参数：

|            | in-loop validation      | 本模块（离线）              |
| ---------- | ----------------------- | --------------------------- |
| 时机       | 训练中每 val_period 步  | 训练后对 checkpoint 跑      |
| 每题采样   | data.val_repeat（12）   | --n（论文 eval 用 12）      |
| 生成长度   | 受训练 max_seq 约束     | 可放到论文的 38912          |
| 评测集     | 只有 AIME24（省时间）   | AIME24 + AIME25 + HMMT25    |
| 指标       | 同一组 grouped_metrics  | 同一组 grouped_metrics      |

两处刻意复用同一份实现，是为了让「训练曲线上的那个数」和「最终报告里的那个数」可比。
分开写两套评测最容易出的事故不是算错，而是算得都对但口径不同，最后没人说得清差异从哪来。

## 判分为什么走 Ray actor

判分用 NeMo-RL 自己的 `HFVerifyWorker`（`nemo_rl.environments.math_environment`），
而不是本地重写一份 math-verify 调用。原因同上：训练时的 reward 就是它给的，
离线评测换一个实现，两边数字就不再可比。

顺带还有个好处：它的 `return_extracted_answer=True` 会返回**判分器自己抽取的答案**，
拿来做多数投票的分组，比我们再正则抠一次 `\\boxed{}` 更贴合判分逻辑。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from common.algorithms.opsd_eval import ValSample, grouped_metrics

# 与实验 run.py 的 STUDENT_TEMPLATE 保持一致：评测时给模型看的题面必须和训练时同构，
# 否则测的是「模型能不能适应新格式」而不是「模型会不会做题」。
DEFAULT_PROMPT_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


@dataclass
class EvalSpec:
    """一次评测的协议参数。默认值对齐论文 eval/run_eval.sh + evaluate_math.py。"""

    n: int = 12                    # 每题采样数（论文 --val_n 12）
    temperature: float = 1.0       # 论文 --temperature 1.0（thinking 模式用 0.6）
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 38912        # 论文默认；竞赛题的思考链很长，砍小会系统性低估
    seed: int = 42
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    system_prompt: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompts(rows: list[dict], spec: EvalSpec, tokenizer) -> list[str]:
    """把题目渲染成模型输入（套 chat template），每题一条；采样数由 SamplingParams.n 控制。"""
    out: list[str] = []
    for row in rows:
        chat: list[dict[str, str]] = []
        if spec.system_prompt:
            chat.append({"role": "system", "content": spec.system_prompt})
        chat.append(
            {"role": "user", "content": spec.prompt_template.format(problem=row["problem"])}
        )
        out.append(
            tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, add_special_tokens=False
            )
        )
    return out


def assemble_samples(
    rows: list[dict],
    completions: list[list[str]],
    scores: list[float],
    extracted: list[Optional[str]],
) -> list[ValSample]:
    """把「题目 × N 条生成 × 判分结果」拍平成 grouped_metrics 要的样本列表。

    三个输入的展平顺序必须一致（题 0 的 N 条、题 1 的 N 条…）；不一致的话
    reward 会配到别人的答案上，指标看着正常却全错——所以这里显式校验长度。
    """
    flat_expected = sum(len(c) for c in completions)
    if not (len(scores) == len(extracted) == flat_expected):
        raise ValueError(
            f"判分结果与生成数量不一致：生成 {flat_expected} 条，"
            f"score {len(scores)} 条，extracted {len(extracted)} 条"
        )
    if len(completions) != len(rows):
        raise ValueError(f"生成分组数 {len(completions)} != 题目数 {len(rows)}")

    samples: list[ValSample] = []
    k = 0
    for pid, comp in enumerate(completions):
        for _ in comp:
            samples.append(
                ValSample(problem_id=str(pid), reward=float(scores[k]), answer=extracted[k])
            )
            k += 1
    return samples


def verify_batch(
    responses: list[str], ground_truths: list[str], *, num_workers: int = 8
) -> tuple[list[float], list[Optional[str]]]:
    """用 NeMo-RL 的 HFVerifyWorker 判分，返回 (分数, 判分器抽取的答案)。

    切成 num_workers 份并行：单条 math-verify 有时会在复杂表达式上卡几秒，
    30 题 × 12 条串行判分能拖到分钟级。
    """
    import ray
    from nemo_rl.environments.math_environment import HFVerifyWorker

    if not responses:
        return [], []

    workers = [HFVerifyWorker.remote() for _ in range(max(1, num_workers))]
    chunk = (len(responses) + len(workers) - 1) // len(workers)
    futures = []
    for i, w in enumerate(workers):
        lo, hi = i * chunk, min((i + 1) * chunk, len(responses))
        if lo >= hi:
            continue
        futures.append(
            w.verify.remote(
                responses[lo:hi], ground_truths[lo:hi], return_extracted_answer=True
            )
        )

    scores: list[float] = []
    extracted: list[Optional[str]] = []
    for s, e in ray.get(futures):
        scores.extend(s)
        extracted.extend(e)
    return scores, extracted


def evaluate_dataset(
    llm,
    tokenizer,
    rows: list[dict],
    spec: EvalSpec,
    *,
    answer_key: str = "answer",
) -> dict[str, float]:
    """对一个评测集跑完整流程，返回 avg@N / pass@N / majority@N 等指标。

    `llm` 是一个 vLLM `LLM` 实例；由调用方创建并在多个评测集之间复用——
    一个 9B 模型的加载要几十秒到几分钟，三个评测集各加载一次纯属浪费。
    """
    from vllm import SamplingParams

    prompts = build_prompts(rows, spec, tokenizer)
    sampling = SamplingParams(
        n=spec.n,
        temperature=spec.temperature,
        top_p=spec.top_p,
        top_k=spec.top_k,
        max_tokens=spec.max_tokens,
        seed=spec.seed,
    )
    outputs = llm.generate(prompts, sampling)

    completions = [[o.text for o in out.outputs] for out in outputs]
    flat_responses: list[str] = []
    flat_truths: list[str] = []
    for row, comps in zip(rows, completions, strict=True):
        truth = str(row.get(answer_key, ""))
        for text in comps:
            flat_responses.append(text)
            flat_truths.append(truth)

    scores, extracted = verify_batch(flat_responses, flat_truths)
    samples = assemble_samples(rows, completions, scores, extracted)

    metrics = grouped_metrics(samples)
    # 生成长度：贴着 max_tokens 说明被截断，低分是「没写完」而非「不会做」。
    lens = [len(t) for c in completions for t in c]
    if lens:
        metrics["mean_completion_chars"] = sum(lens) / len(lens)
    return metrics


def format_report(name: str, metrics: dict[str, float]) -> str:
    """一行式结果，便于在作业日志里直接和论文表格对照。"""
    if not metrics:
        return f"{name:<10} （无样本）"
    return (
        f"{name:<10} "
        f"avg@N={metrics['avg_at_n']:.3f}  "
        f"pass@N={metrics['pass_at_n']:.3f}  "
        f"maj@N={metrics['majority_at_n']:.3f}  "
        f"({int(metrics['num_problems'])} 题 × {metrics['samples_per_problem']:.0f}，"
        f"未抽出答案 {metrics['unparsed_answer_rate']:.1%})"
    )


def discover_eval_files(data_dir: str) -> dict[str, str]:
    """找出 data_dir 下的 eval_<名字>.jsonl，返回 {名字: 路径}（按名字排序）。"""
    out: dict[str, str] = {}
    if not os.path.isdir(data_dir):
        return out
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("eval_") and fn.endswith(".jsonl"):
            out[fn[len("eval_") : -len(".jsonl")]] = os.path.join(data_dir, fn)
    return out
