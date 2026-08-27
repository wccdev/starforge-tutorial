# configs/base

Grandfather configs. Official NeMo-RL v0.7.0 examples, copied as-is. Tune in the experiment `config.yaml`, not here.

| File | Upstream | Use |
| --- | --- | --- |
| `grpo_math_1B.yaml` | `examples/configs/grpo_math_1B.yaml` | GRPO |
| `sft.yaml` | `examples/configs/sft.yaml` | SFT |
| `grpo_sliding_puzzle.yaml` | `examples/configs/grpo_sliding_puzzle.yaml` | Multi-turn puzzle (`defaults: grpo_math_1B.yaml`) |

Overlays in this directory (`grpo_megatron.yaml`, `grpo_lora.yaml`, `grpo_noncolocated.yaml`) are ours. They are not upstream copies.

On a NeMo-RL bump:

```bash
cp /path/to/NeMo-RL/examples/configs/grpo_math_1B.yaml configs/base/
cp /path/to/NeMo-RL/examples/configs/sft.yaml configs/base/
cp /path/to/NeMo-RL/examples/configs/grpo_sliding_puzzle.yaml configs/base/
```
