"""清单式打包 / 流式进度逻辑单测（不触发真实网络）。

覆盖：
- 打包白名单：只有 experiments/<exp>/ + common/ + configs/ + cluster/<profile>/ + scripts/launch.sh 入包；
- 清单文件缺失 / 作业包为空 → 显式报错（fail-loud，不静默跳过）；
- CRLF 规范化（集群 Linux 侧 source 的 .sh/.conf/lab）；
- _ProgressReader 边读边回报字节、读空触发 on_done；
- cli_ui 的人类可读展示。
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import typer

from nemo_rl_lab import api_client, cli_ui, packing

# --------------------------- 打包白名单 ---------------------------
GIT_TREE = "\n".join([
    "experiments/e/config.yaml",
    "experiments/e/method",
    "experiments/e/whitepaper.pdf",       # 清单内但属非运行时产物 → 剔除
    "experiments/other/config.yaml",      # 他人实验 → 不上传
    "common/rewards/qa_reward.py",
    "configs/base/grpo_math_1B.yaml",
    "cluster/h100/env.sh",
    "cluster/h100/overrides.conf",
    "cluster/h200/env.sh",                # 未选中的 profile → 不上传
    "scripts/launch.sh",
    "scripts/download_models.py",         # 工具脚本 → 不上传
    "nemo_rl_lab/cli.py",                 # CLI 源码 → 不上传
    "README.md",
    "docs/MaxRL.pdf",
    "datasets/gsm8k/train.jsonl",
    "",  # 空行应被忽略
])


def test_manifest_keeps_only_runtime_payload(monkeypatch):
    monkeypatch.setattr(packing, "_git_out", lambda *a, **k: GIT_TREE)
    files, skipped = packing.list_working_files(
        Path("/repo"), exp_rel="experiments/e", profile="h100", with_stats=True
    )
    assert files == [
        "experiments/e/config.yaml",
        "experiments/e/method",
        "common/rewards/qa_reward.py",
        "configs/base/grpo_math_1B.yaml",
        "cluster/h100/env.sh",
        "cluster/h100/overrides.conf",
        "scripts/launch.sh",
    ]
    assert skipped == len([ln for ln in GIT_TREE.splitlines() if ln]) - len(files)


def test_manifest_requires_exp_and_profile():
    with pytest.raises(ValueError):
        packing.upload_manifest("", "h100")
    with pytest.raises(ValueError):
        packing.upload_manifest("experiments/e", "")


def test_empty_payload_fails_loud(monkeypatch):
    monkeypatch.setattr(packing, "_git_out", lambda *a, **k: "docs/a.pdf\nREADME.md")
    with pytest.raises(typer.Exit):
        packing.list_working_files(Path("/repo"), exp_rel="experiments/e", profile="h100")


def test_pack_missing_file_fails_loud(tmp_path):
    """fail-loud：清单里的文件磁盘上不存在 → 报错，不静默跳过。"""
    with pytest.raises(typer.Exit):
        packing.pack_working_dir(tmp_path, ["experiments/e/config.yaml"])


def test_pack_respects_gitignore(tmp_path):
    import subprocess

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("experiments/e/outputs/\n")
    exp = tmp_path / "experiments" / "e"
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text("a: 1\n")
    (exp / "untracked_new.txt").write_text("new but not ignored\n")
    (exp / "outputs").mkdir()
    (exp / "outputs" / "big.bin").write_text("x" * 100)
    prof = tmp_path / "cluster" / "h100"
    prof.mkdir(parents=True)
    (prof / "overrides.conf").write_text("cluster.num_nodes=1\n")
    git("add", ".gitignore", "experiments/e/config.yaml", "cluster")
    git("commit", "-qm", "init")

    files = packing.list_working_files(tmp_path, exp_rel="experiments/e", profile="h100")
    blob = packing.pack_working_dir(tmp_path, files)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert "experiments/e/config.yaml" in names
    assert "experiments/e/untracked_new.txt" in names  # 未提交但未被忽略 → 应上传
    assert "experiments/e/outputs/big.bin" not in names  # gitignore → 不上传


# --------------------------- CRLF 规范化 ---------------------------
@pytest.mark.parametrize(
    "rel,expected",
    [
        ("scripts/launch.sh", True),
        ("cluster/h100/overrides.conf", True),
        ("lab", True),
        ("nemo_rl_lab/cli.py", False),
        ("lab.cmd", False),
    ],
)
def test_needs_unix_lf(rel, expected):
    assert packing._needs_unix_lf(rel) is expected


def test_normalize_unix_lf():
    assert packing._normalize_unix_lf(b"a\r\nb\rc") == b"a\nb\nc"


def test_pack_working_dir_normalizes_crlf_for_cluster_scripts(tmp_path):
    launch_sh = tmp_path / "scripts" / "launch.sh"
    launch_sh.parent.mkdir(parents=True)
    launch_sh.write_bytes(b"#!/bin/bash\r\nset -euo pipefail\r\n")
    conf = tmp_path / "cluster" / "h100" / "overrides.conf"
    conf.parent.mkdir(parents=True)
    conf.write_bytes(b"cluster.num_nodes=1\r\n")
    (tmp_path / "keep.py").write_text("print(1)\n", encoding="utf-8")

    blob = packing.pack_working_dir(
        tmp_path,
        [
            "scripts/launch.sh",
            "cluster/h100/overrides.conf",
            "keep.py",
        ],
    )
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        run_data = tar.extractfile("scripts/launch.sh").read()
        conf_data = tar.extractfile("cluster/h100/overrides.conf").read()
        py_data = tar.extractfile("keep.py").read()
    assert run_data == b"#!/bin/bash\nset -euo pipefail\n"
    assert conf_data == b"cluster.num_nodes=1\n"
    assert py_data == b"print(1)\n"


def test_repo_shell_scripts_use_lf_only():
    """门禁：仓库内 .sh / lab 须 LF，配合 .gitattributes 防 Windows checkout 污染。"""
    repo = Path(__file__).resolve().parents[1]
    paths = sorted(repo.rglob("*.sh"))
    lab = repo / "lab"
    if lab.is_file():
        paths.append(lab)
    assert paths, "expected at least one shell entry in repo"
    bad = [p.relative_to(repo) for p in paths if b"\r" in p.read_bytes()]
    assert not bad, f"CRLF in Unix scripts (use LF): {', '.join(str(p) for p in bad)}"


# --------------------------- 流式上传进度 ---------------------------
def test_progress_reader_reports_and_done():
    ticks: list[int] = []
    done: list[bool] = []
    reader = api_client._ProgressReader(
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


# --------------------------- 耗时格式 ---------------------------
@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0.0s"), (3.2, "3.2s"), (59.9, "59.9s"), (60, "1m 00s"), (125, "2m 05s"), (3661, "1h 01m")],
)
def test_format_elapsed(seconds, expected):
    assert cli_ui.format_elapsed(seconds) == expected


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


def test_pipeline_reporter_stages_and_timing():
    """垂直步骤条：已完成 ✓、当前 spinner、服务端独立阶段。"""
    from rich.console import Console

    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    reporter = cli_ui._PipelineReporter(console)
    with reporter:
        reporter.start_pack(2)
        assert reporter._stages["pack"].status == "active"
        reporter.pack_tick(2)
        reporter.start_upload(2048)
        assert reporter._stages["pack"].status == "done"
        assert reporter._stages["upload"].status == "active"
        reporter.upload_tick(2048)
        reporter.awaiting_server()
        assert reporter._stages["upload"].status == "done"
        assert reporter._stages["server"].status == "active"
        assert reporter._stages["server"].started is not None
        assert reporter._stages["server"].started >= reporter._stages["upload"].finished
        reporter.finish()
        assert reporter._stages["server"].status == "done"
    panel = reporter._render()
    with console.capture() as capture:
        console.print(panel)
    rendered = capture.get()
    assert "lab submit" in rendered
    assert "服务端受理" in rendered


def test_emit_error_stops_progress_first():
    """出错文案必须在步骤条停掉之后才输出，否则会被 Live 的下一帧擦掉（提交失败无提示）。"""
    from rich.console import Console

    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    reporter = cli_ui._PipelineReporter(console)
    with reporter:
        reporter.start_pack(1)
        reporter.awaiting_server()
        assert reporter._live is not None
        assert cli_ui._active_progress is reporter
        cli_ui.emit_error("提交失败", body="配额不足")
        assert reporter._live is None  # Live 已停，后续输出不会被覆盖
        assert cli_ui._active_progress is None
    assert cli_ui._active_progress is None
