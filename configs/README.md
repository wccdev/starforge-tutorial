# configs/

NeMo-RL 0.7.0 config inheritance (`load_config` in `nemo_rl/utils/config.py`): `defaults:` plus later keys win. Experiments write diffs only.

```
configs/
├── base/      grandfather: v0.7.0 example copies (do not hand-edit) + a few overlays
│   ├── grpo_math_1B.yaml
│   ├── sft.yaml
│   ├── grpo_sliding_puzzle.yaml
│   ├── grpo_megatron.yaml      # Megatron + low-memory knobs
│   ├── grpo_lora.yaml          # PEFT; lr lives on megatron optimizer
│   └── grpo_noncolocated.yaml  # generate GPU ≠ train GPU
└── models/    model fragments (name, tokenizer, memory)
    ├── qwen3.5-4b.yaml
    └── qwen3.5-9b.yaml
```

Default GRPO examples in this repo stack Megatron + LoRA:

```yaml
defaults:
  - ../../configs/base/grpo_math_1B.yaml
  - ../../configs/models/qwen3.5-9b.yaml
  - ../../configs/base/grpo_megatron.yaml
  - ../../configs/base/grpo_lora.yaml
```

Drop `grpo_lora.yaml` to go full-parameter (put lr on `policy.megatron_cfg.optimizer.lr`, e.g. 1e-6). Drop `grpo_megatron.yaml` to go DTensor/FSDP (lr back on `policy.optimizer.kwargs.lr`). `grpo_noncolocated.yaml` is optional; on 2 GPUs that is 1 generate / 1 train, so PP=1.

Node count, TP/PP/CP, NCCL come from the server hardware registry via `--profile`. There is no `cluster/` directory in this repo.

Experiment `config.yaml` is the child:

```yaml
defaults:
  - ../../configs/base/grpo_math_1B.yaml
  - ../../configs/models/qwen3.5-9b.yaml
grpo:
  num_generations_per_prompt: 16
loss_fn:
  reference_policy_kl_penalty: 0.01
```

Later `defaults` entries override earlier ones. `_override_: true` replaces a whole block instead of merging.

| Method | Typical base |
| --- | --- |
| SFT | `configs/base/sft.yaml` |
| GRPO | `configs/base/grpo_math_1B.yaml` |
| Multi-turn agent | `configs/base/grpo_sliding_puzzle.yaml` |

| Knob | Key |
| --- | --- |
| Base model | `policy.model_name` |
| Seq length | `policy.max_total_sequence_length` |
| LR (Megatron / LoRA) | `policy.megatron_cfg.optimizer.lr` (LoRA often 1e-4) |
| LR (DTensor) | `policy.optimizer.kwargs.lr` |
| LoRA | `policy.megatron_cfg.peft.enabled` / `.dim` / `.alpha` |
| Global batch | `policy.train_global_batch_size` (must equal prompts × gens) |
| KL | `loss_fn.reference_policy_kl_penalty` |
| Prompts / gens | `grpo.num_prompts_per_step` / `grpo.num_generations_per_prompt` |
| Multi-turn cap | `grpo.max_rollout_turns` |
