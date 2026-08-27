# Naming

```
<method>_<model>_<dataset>[_<tag>]
```

Underscores between fields, hyphens inside a field.

| method | Meaning |
| --- | --- |
| `sft` | supervised fine-tune |
| `grpo` | GRPO (also used for sliding-puzzle / QA agent in this repo) |
| `dpo` / `ppo` / `rm` | as named |
| `maxrl` / `opsd` | catalog methods with their own recipe |
| `agent-grpo` | multi-turn GRPO (directory prefix only; lockfile is still `nemo-rl/grpo`) |
| `verl-grpo` / `trl-grpo` | same idea on verl / TRL |

Model: `<family><version>-<size>`, lowercase `b`: `qwen3.5-4b`, `qwen3.5-9b-instruct`.

Dataset short name: `gsm8k`, `alpaca`, `qa-rl`. Mix with `+` (`gsm8k+math`).

Tag: `v1`, a date, or a knob (`lr2e6`, `8k-ctx`).

Everything submitable lives under `experiments/`. The project name is `starforge.yaml` `name`, not the experiment directory.

Optional SwanLab: project = experiment dir (or model), run name = knobs (`lr1e6-bs64-kl0.001`). Put the URL in the experiment README. See `docs/swanlab.md`.

```
<experiment>/outputs/step_<N>/
<experiment>/outputs/final/
<experiment>/hf_export/
```

Those output dirs are gitignored.
