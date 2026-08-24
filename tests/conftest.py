"""tutorial 测试：若同级有 starforge 仓，把它的 cli/ 与 core/ 一起加进 path。

smoke 契约测要拿 CLI 去校验 recipe.lock.json 里的 core recipe bundle 摘要，
所以 cli 与 core 必须来自【同一个 checkout】：仓库源码的 cli 配上 PyPI 装的
旧 core，摘要必然对不上，测试会以 RecipeLockError 失败，且报错让人以为是
lock 文件该 upgrade（其实 lock 没问题，是测试环境凑了两个版本）。
仓库不在同级时两者都不插入，退回 venv 里成套的版本，同样是自洽的。
"""
from __future__ import annotations

import sys
from pathlib import Path

_SIBLINGS = Path(__file__).resolve().parents[2]
for repo in (_SIBLINGS / "nemo-rl-console", _SIBLINGS / "starforge"):
    parts = [repo / "cli", repo / "core"]
    if not all(p.is_dir() for p in parts):
        continue
    for part in parts:
        if str(part) not in sys.path:
            sys.path.insert(0, str(part))
    break
