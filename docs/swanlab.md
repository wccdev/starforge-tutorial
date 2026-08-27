# SwanLab (optional)

Console curves come from the platform runner / `starforge.report`. Enable SwanLab only if you also want its cloud.

```bash
pip install swanlab
swanlab login
```

On a cluster job, `SWANFORGE_API_KEY` (and `HF_TOKEN`) are injected by the server. You do not commit them. For a laptop debug session you can `export SWANFORGE_API_KEY=…` yourself.

```yaml
logger:
  swanlab_enabled: true
  monitor_gpus: true
  swanlab:
    project: "grpo_qwen3.5-9b_gsm8k_v2"   # experiment dir, or a model name
    name: "lr1e6-g16-kl0.01"              # knobs
```

Same hardware, different boxes: suffix the run name `-h100` / `-h200`. Paste the URL into the experiment README.
