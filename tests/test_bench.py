"""forge bench：模型引用解析与评测提交参数翻译。"""
from __future__ import annotations

import io
import json

from typer.testing import CliRunner

from starforge_cli import cli
from starforge_cli.commands import bench as bench_cmd

runner = CliRunner()


def test_run_ref_resolves_hf_export(monkeypatch):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_bearer(srv, method, path):
        assert path == "/api/jobs/run-42/artifacts"
        return _Resp(json.dumps({"artifacts": [
            {"kind": "checkpoint", "path": "/out/step_100"},
            {"kind": "hf_export", "path": "/out/hf_export"},
        ]}).encode())

    monkeypatch.setattr(bench_cmd.api_client, "current_server", lambda *a: "http://srv")
    monkeypatch.setattr(bench_cmd.api_client, "_bearer_request", fake_bearer)
    assert bench_cmd._resolve_model_ref("run:run-42") == "/out/hf_export"
    # 非 run: 引用原样返回
    assert bench_cmd._resolve_model_ref("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"


def test_bench_submits_evalkit_spec(monkeypatch, tmp_path):
    (tmp_path / "experiments" / "bench-demo").mkdir(parents=True)
    calls = {}

    monkeypatch.setattr(bench_cmd.common, "ROOT", tmp_path)
    monkeypatch.setattr(bench_cmd, "gate", lambda: None)
    monkeypatch.setattr(bench_cmd.common, "resolve_profile", lambda *a, **k: "h200")
    monkeypatch.setattr(
        bench_cmd, "_materialize_profile_or_exit",
        lambda exprs: ("h200", ["eval:h200:1:1"], []),
    )
    monkeypatch.setattr(
        bench_cmd.packing, "git_provenance",
        lambda *a, **k: {"git_commit": "cafe", "git_dirty": False, "config_sha": "x"},
    )

    def fake_build_spec(exp_path, **kw):
        calls["method"] = kw["method"]
        calls["sets"] = kw["sets"]
        calls["model"] = kw["model"]
        calls["validate"] = kw["validate"]
        return {"spec": True}

    monkeypatch.setattr(bench_cmd, "_build_spec_or_exit", fake_build_spec)
    monkeypatch.setattr(
        bench_cmd.api_client, "submit_via_server",
        lambda *a, **k: {"queued": False, "job_id": "ray-1", "run_id": "r1"},
    )

    res = runner.invoke(cli.app, [
        "bench", "experiments/bench-demo",
        "--model", "Qwen/Qwen3-0.6B",
        "--suites", "gsm8k,mmlu",
        "--runner", "evalscope",
        "--limit", "20",
    ])
    assert res.exit_code == 0, res.stdout
    assert calls["method"] == "evalkit/benchmark"
    assert calls["model"] == "Qwen/Qwen3-0.6B"
    assert calls["sets"] == ["runner=evalscope", "suites=gsm8k,mmlu", "limit=20"]
    assert calls["validate"] is False
