"""tutorial 测试：若同级有平台仓，把 core/ 加进 path（import starforge）。

CLI 与内核已合并为同一只 ``starforge`` 包。仓库不在同级时退回 venv 里已安装的版本。
"""
from __future__ import annotations

import sys
from pathlib import Path

_SIBLINGS = Path(__file__).resolve().parents[2]
for repo in (_SIBLINGS / "nemo-rl-console", _SIBLINGS / "starforge"):
    core = repo / "core"
    if not core.is_dir():
        continue
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    break
