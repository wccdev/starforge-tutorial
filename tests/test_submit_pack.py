"""上传打包的过滤 / 流式进度逻辑单测（不触发真实 git / 网络）。

覆盖：
- 非运行时产物（.cursor / docs / reports / *.pdf）不进作业包，但正常代码/配置保留；
- list_working_files 的 with_stats 计数；
- _ProgressReader 边读边回报字节、读空触发 on_done；
- cli_ui.human_bytes 展示。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nemo_rl_lab import cli_login, cli_ui


# --------------------------- 上传排除规则 ---------------------------
@pytest.mark.parametrize(
    "rel",
    [
        ".cursor/skills/ui-ux-pro-max/SKILL.md",
        ".cursor/skills/ui-ux-pro-max/data/colors.csv",
        "docs/MaxRL.pdf",
        "docs/naming-convention.md",
        "reports/finetuning-comparison-report.html",
        "experiments/x/whitepaper.pdf",
    ],
)
def test_upload_excluded_paths(rel):
    assert cli_login._is_upload_excluded(rel) is True


@pytest.mark.parametrize(
    "rel",
    [
        "nemo_rl_lab/cli.py",
        "common/rewards/qa_reward.py",
        "experiments/grpo_x/config.yaml",
        "configs/base/grpo_math_1B.yaml",
        "datasets/gsm8k/train.jsonl",
        "README.md",
        "pyproject.toml",
    ],
)
def test_upload_kept_paths(rel):
    assert cli_login._is_upload_excluded(rel) is False


def test_list_working_files_filters_and_counts(monkeypatch):
    listing = "\n".join([
        "nemo_rl_lab/cli.py",
        "common/rewards/qa_reward.py",
        ".cursor/skills/x/SKILL.md",
        "docs/MaxRL.pdf",
        "reports/r.html",
        "experiments/e/config.yaml",
        "",  # 空行应被忽略
    ])
    monkeypatch.setattr(cli_login, "_git_out", lambda *a, **k: listing)
    files, skipped = cli_login.list_working_files(Path("/repo"), with_stats=True)
    assert files == [
        "nemo_rl_lab/cli.py",
        "common/rewards/qa_reward.py",
        "experiments/e/config.yaml",
    ]
    assert skipped == 3  # .cursor + docs/*.pdf + reports


def test_list_working_files_empty_fails(monkeypatch):
    # 全被排除 → 视为无可上传文件，走 cli_ui.fail（抛 typer.Exit）。
    import typer

    monkeypatch.setattr(cli_login, "_git_out", lambda *a, **k: "docs/a.pdf\n.cursor/b.csv")
    with pytest.raises(typer.Exit):
        cli_login.list_working_files(Path("/repo"))


# --------------------------- 流式上传进度 ---------------------------
def test_progress_reader_reports_and_done():
    ticks: list[int] = []
    done: list[bool] = []
    reader = cli_login._ProgressReader(
        b"0123456789", on_read=ticks.append, on_done=lambda: done.append(True)
    )
    assert len(reader) == 10
    assert reader.read(4) == b"0123"
    assert reader.read(4) == b"4567"
    assert reader.read(4) == b"89"
    assert ticks == [4, 4, 2]
    assert done == []  # 尚未读空
    assert reader.read(4) == b""  # 读空
    assert done == [True]
    # 再次读空不重复触发 on_done
    assert reader.read() == b""
    assert done == [True]


# --------------------------- 人类可读体积 ---------------------------
@pytest.mark.parametrize(
    "n,expected",
    [(0, "0 B"), (512, "512 B"), (1024, "1.0 KB"), (1536, "1.5 KB"),
     (5 * 1024 * 1024, "5.0 MB"), (3 * 1024 ** 3, "3.0 GB")],
)
def test_human_bytes(n, expected):
    assert cli_ui.human_bytes(n) == expected


# --------------------------- 降级 reporter 不炸 ---------------------------
def test_plain_reporter_is_noop_safe():
    r = cli_ui._PlainReporter()
    with r:
        r.start_pack(3)
        r.pack_tick()
        r.start_upload(100)
        r.upload_tick(50)
        r.awaiting_server()
        r.finish()
