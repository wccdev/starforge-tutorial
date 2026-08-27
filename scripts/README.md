# scripts/

Jobs go through the console: `sf login`, then `sf submit <exp>`. Nothing in this directory submits to Ray from your laptop.

| Script | Where it runs | What |
| --- | --- | --- |
| `prefetch_hf_model.sh` | Inside a training container | `HF_HOME` prefetch so training does not hit hf-mirror.com |
| `download_models.py` | A machine with internet | Writes an HF cache tree you can rsync to the GPU box |
| `download_via_relay.py` | GPU box that can only SSH to a jump host | `ssh … curl` pipe; jump host never stores the files |
| `install_to_hf_cache.py` | Once | Old flat `hf_models/` layout → HF cache (hardlinks) |
| `smoke_verl_sft.sh` / `smoke_verl_grpo.sh` | GPU | Tiny Qwen2.5 0.5B verl jobs |

Prefetch in a container:

```bash
export HF_TOKEN=... HF_HOME=/data/hf_cache
bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B
```

Download on a connected machine:

```bash
uv run python scripts/download_models.py --list
uv run python scripts/download_models.py --daemon
uv run python scripts/download_models.py --source modelscope --daemon   # China
```

Copy with `rsync -a` so `snapshots/` stays as relative symlinks into `blobs/`. `rsync -aL` or `scp -r` doubles the size.

`refs/main` must not have a trailing newline. Snapshot dirs must be the real HF commit sha (or hub will re-download). ModelScope does not give that sha; the script asks HF for a few KB of metadata, or you pass `--fake-sha` and set `HF_HUB_OFFLINE=1` on the GPU box.

Relay (GPU box, jump host has curl only):

```bash
python3 scripts/download_via_relay.py --relay root@10.0.0.2 --check
python3 scripts/download_via_relay.py --relay root@10.0.0.2 --hf-home /data/hf_cache --daemon
```

`--only Qwen/Qwen3.5-4B` or `modelscope_id=hf_id` when namespaces differ. Incomplete files are `*.part`; a 1-byte Range probe decides whether resume is safe.
