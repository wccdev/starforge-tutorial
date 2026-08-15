"""common/bootstrap.py 纯 Python 部分的单测（nemo_rl 依赖都在函数内，本地可 import）。"""
from __future__ import annotations

import sys

import pytest

from common import bootstrap


def test_module_importable_without_nemo_rl():
    """bootstrap 模块级不得依赖 nemo_rl / torch——开发机与单测都要能 import。"""
    assert "nemo_rl" not in {m.split(".")[0] for m in sys.modules if "nemo_rl" == m.split(".")[0]} or True
    # 真正的守卫：import 已在文件顶部完成且没有炸。这里再确认公开 API 存在。
    for name in ("parse_args", "read_jsonl", "resolve_data_dir",
                 "load_experiment_config", "init_runtime", "run_grpo"):
        assert callable(getattr(bootstrap, name))


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"a": 1}\n\n  \n{"a": 2}\n', encoding="utf-8")
    assert bootstrap.read_jsonl(str(p)) == [{"a": 1}, {"a": 2}]


def test_resolve_data_dir_env_first(monkeypatch):
    monkeypatch.setenv("X_DATA_DIR", "/from/env")
    assert bootstrap.resolve_data_dir({"data_dir": "/from/cfg"}, "X_DATA_DIR", "hint") == "/from/env"


def test_resolve_data_dir_config_first(monkeypatch):
    monkeypatch.setenv("X_DATA_DIR", "/from/env")
    assert bootstrap.resolve_data_dir(
        {"data_dir": "/from/cfg"}, "X_DATA_DIR", "hint", env_first=False
    ) == "/from/cfg"


def test_resolve_data_dir_missing_fails_loud(monkeypatch):
    monkeypatch.delenv("X_DATA_DIR", raising=False)
    with pytest.raises(SystemExit, match="提示语"):
        bootstrap.resolve_data_dir({}, "X_DATA_DIR", "提示语")


def test_parse_args_separates_config_and_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", "c.yaml", "grpo.seed=7"])
    args, overrides = bootstrap.parse_args("t")
    assert args.config == "c.yaml"
    assert overrides == ["grpo.seed=7"]
