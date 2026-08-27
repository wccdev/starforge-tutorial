# Plugins

A plugin is a directory with `plugin.yaml`. After you publish it, anyone can install it, lock it into an experiment, and have it injected into the job.

| Path | Kind | Role |
| --- | --- | --- |
| `examples/rloo/` | algorithm (eager) | Copy this to write an advantage estimator |
| `examples/tiny-qa-prep/` | data-prep | Copy this to write a local prepare script |
| `maxrl/` | algorithm (eager) | Adapter over `common/algorithms/maxrl` |
| `opsd/` | algorithm (deferred) | Needs tokenizer context; see `opsd/` |

```
my-plugin/
├── plugin.yaml
└── my_module.py          # algorithm only
```

```yaml
schema: forge/plugin/v1
name: my-plugin           # leaf name; public id is <owner>/<name>
version: 0.1.0            # immutable; same version + different bytes → 409
kind: algorithm           # algorithm | data-prep
load: eager               # algorithm only: eager | deferred
summary: one line
entrypoint: my_module:install
requires:
  core: ">=0.3,<1"        # PEP 440; launcher refuses the job if unmet
```

## algorithm

The platform unpacks the plugin under `forge_plugins/<name>/` and the launcher checks the digest, then:

```python
def install(params, **ctx) -> None: ...
```

`params` is the job hyperparam snapshot. `ctx` is empty for eager; deferred gets runtime bits (`pad_token_id`, …).

Eager: `install` runs before training. Typical pattern: wrap a factory, add a branch, stay idempotent. See `examples/rloo/`.

Deferred: launcher only registers. The training entry calls `common.algorithms.registry.install_deferred(name, **ctx)` once it has a tokenizer. See `opsd/`.

Imports see the job package plus the training image. The plugin should be self-contained. New Python deps have to already be in the image.

## data-prep

No entrypoint. After install, CLI finds `prepare_<dataset>.py`:

```bash
sf dataset prepare              # list (builtins + plugins)
sf dataset prepare tiny_qa      # runs forge_plugins/*/prepare_tiny_qa.py
```

The first-line docstring is the one-liner in that list. A builtin in `common/data/` wins on name clash. See `examples/tiny-qa-prep/`.

## Publish → install → submit

```bash
sf plugin publish plugins/examples/rloo
sf plugin ls
sf plugin info <owner>/rloo
sf plugin install <owner>/rloo --exp experiments/my-exp
sf submit experiments/my-exp
```

Digest is sha256 of the directory (skip `__pycache__` / `.pyc`). Publish, submit, and load each check it.

```python
from pathlib import Path
from starforge.plugins import load_manifest, directory_digest
p = Path("plugins/examples/rloo")
print(load_manifest(p), directory_digest(p))
```

Cap 32 MB (code, not weights). Versions are immutable. Admins can disable a plugin for new submits. You publish to your own namespace (`--owner` is admin). Unmet `requires.core` fails at launch, not after a queue wait.
