# experiments/

One subdirectory = one submitable job. Names: `docs/naming-convention.md`.

```bash
sf new grpo_qwen3.5-4b_gsm8k_v1 --method nemo-rl/grpo
```

That writes `config.yaml`, a template README, and `recipe.lock.json`. Copying directories by hand skips the lockfile.

Every experiment needs a `README.md`: goal, knobs you actually changed, result, optional SwanLab link.

Long-lived work also wants:

- pinned data and recipe versions
- `sf eval <name>`
- `sf export <name>` (optional `--push-repo`)
- a submit you can find later with `sf job ls` (commit / config fingerprint / run id live on the server)

An experiment directory is not a project. The project is `name` in `starforge.yaml`. The console groups this repo's runs as `@<user>/<project>`.
