"""tutorial 测试：若同级有 starforge 仓，把 cli/ 加进 path（smoke 契约测需要）。"""
from __future__ import annotations

import sys
from pathlib import Path

_CLI = Path(__file__).resolve().parents[2] / "nemo-rl-console" / "cli"
_CLI_ALT = Path(__file__).resolve().parents[2] / "starforge" / "cli"
for cand in (_CLI, _CLI_ALT):
    if cand.is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
        break
