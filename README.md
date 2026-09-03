# starforge-tutorial

Official example project for [StarForge](https://github.com/wccdev/starforge). Same layout as `sf init`: experiments, configs, shared code. This is not the CLI source.

```bash
pip install starforge-core
sf login --server https://<your console>
sf ls
sf validate grpo_qwen3.5-4b_gsm8k_v1
```

Day-to-day work belongs in your own repo:

```bash
sf init my-lab --yes
```

`starforge-core` is the client (`sf`). The control plane is a separate package, `starforge-server`.

## Experiments

| Directory | Method | Notes |
| --- | --- | --- |
| `sft_qwen3.5-4b_alpaca_v1` | `nemo-rl/sft` | Local Alpaca jsonl |
| `grpo_qwen3.5-4b_gsm8k_v1` | `nemo-rl/grpo` | GSM8K, Megatron + LoRA |
| `agent-grpo_qwen3.5-9b_sliding-puzzle_v1` | `nemo-rl/grpo` | Built-in sliding-puzzle env, 1×H100 |
| `grpo_qwen3.5-9b_qa-rl_v1` | `nemo-rl/grpo` | Internal QA, single turn, no tools |
| `grpo_qwen3.5-9b_qa-rl-agent_v3` | `nemo-rl/grpo` | Same QA, multi-turn BM25 search |
| `maxrl_qwen3.5-9b_qa-rl-agent_v2` | `nemo-rl/maxrl` | Same agent setup, MaxRL advantage |
| `opsd_qwen3.5-9b_math_h200_1n2g` | `nemo-rl/opsd` | On-policy self-distillation |
| `verl-grpo_qwen3.5-9b_qa-tools_v1` | `verl/grpo` | Same QA+tools scene, verl Agent Loop |
| `verl-grpo_deepseek-v4-flash_qa-tools_v1` | `verl/grpo` | DSv4-Flash-0731 GRPO + Megatron LoRA, 8×H200 |
| `trl-grpo_qwen3.5-9b_qa-tools_v1` | `trl/grpo` | Same scene, TRL `tools=` |
| `verl-grpo_qwen3.5-9b_rtl-agent_v1` | `verl/grpo` | RTL design agent: compiler-in-the-loop, three-stage reward ([data guide](docs/rtl-dataset.md)) |

`common/` ships with the job package (data scripts, environments, rewards). `plugins/` has example algorithm / data-prep plugins. `smoke/` is a tiny-GPU verl check.

## Layout

```
starforge.yaml      repo marker + project name
experiments/        one directory per submitable job
configs/            NeMo-RL bases + model fragments
common/             shared code uploaded with the job
plugins/            example plugins
datasets/           metadata / prepare scripts (large files gitignored)
scripts/            HF cache download helpers, verl smoke
smoke/              minimal verl SFT / GRPO
```

New experiment:

```bash
sf new my-grpo --method nemo-rl/grpo
```

Hardware is `--profile` at submit time, not a `cluster/` tree in this repo.
