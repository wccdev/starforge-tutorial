# common/

Code that ships with the job package. Put *your* data scripts, environments, rewards, and algorithm patches here. Do not copy platform metrics collection.

| Path | What |
| --- | --- |
| `data/` | download / clean / convert |
| `environments/` | custom NeMo-RL Environment (GRPO reward source, multi-turn) |
| `rewards/` | rule rewards / LLM judge |
| `algorithms/` | OPSD / MaxRL patches |
| `envkit/` | tool-calling / gym helpers |
| `eval/` | shared eval bits |
| `bootstrap.py` | NeMo-RL `run.py` boilerplate |

Curves:

```python
from starforge.report import init, log, finish
```

Catalog methods (NeMo-RL / verl / TRL / OpenRLHF) already wire the framework logger. Custom jobs with `observability: platform` get `starforge` on `PYTHONPATH`.
