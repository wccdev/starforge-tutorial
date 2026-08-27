# configs/models

One fragment per base model: `policy.model_name`, tokenizer, seq / memory knobs. No `defaults` of its own.

```yaml
defaults:
  - ../../configs/base/grpo_math_1B.yaml
  - ../../configs/models/qwen3.5-9b.yaml
```

Later entries win. Filename matches the `model` field in `docs/naming-convention.md`.
