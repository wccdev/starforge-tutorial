"""TRL 工具调用示例实验（qa-tools）的本地守护测试。

本地环境没有 trl / datasets 包，用 stub 加载 train.py——
只验证我们自己的工具/判分/数据转换逻辑与 config 契约，不验证 TRL 本身。
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "trl-grpo_qwen3.5-9b_qa-tools_v1"


def _load_train():
    stubs = {
        "trl": types.ModuleType("trl"),
        "datasets": types.ModuleType("datasets"),
    }
    stubs["trl"].GRPOConfig = object
    stubs["trl"].GRPOTrainer = object
    stubs["datasets"].load_dataset = lambda *a, **k: None
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("trl_qa_tools_train", EXP / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_search_docs_hits_and_placeholder(tmp_path, monkeypatch):
    train = _load_train()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cmp_process.md").write_text(
        "# CMP 制程\n铜 CMP 分为主抛与精抛两步，主抛负责去除大部分铜层。\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCS_DIR", str(docs))
    hit = train.search_docs("CMP 主抛 铜")
    assert hit.startswith("[检索结果]") and "cmp_process.md" in hit

    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "missing"))
    assert "不存在" in train.search_docs("任何词")


def test_tool_docstring_supports_schema_inference():
    """TRL 由类型注解 + Google docstring 推断工具 schema；缺 Args 段注册会失败。"""
    train = _load_train()
    doc = train.search_docs.__doc__ or ""
    assert "Args:" in doc and "Returns:" in doc
    # train.py 开了 from __future__ import annotations，注解为字符串形式。
    assert train.search_docs.__annotations__.get("query") in (str, "str")


def test_qa_boxed_reward_reads_last_assistant_message():
    train = _load_train()
    conversational = [
        [
            {"role": "assistant", "content": '{"name": "search_docs", "arguments": {"query": "CMP"}}'},
            {"role": "tool", "content": "[检索结果]【cmp_process.md】主抛负责去除大部分铜层"},
            {"role": "assistant", "content": "根据资料，答案是 \\boxed{B}"},
        ],
        [{"role": "assistant", "content": "没写盒子"}],
        "纯文本 completion \\boxed{A,C}",
    ]
    rewards = train.qa_boxed_reward(conversational, answer=["B", "B", "A,C"])
    assert rewards == [1.0, train.FORMAT_PENALTY, 1.0]


def test_config_is_flat_grpo_config_and_omits_platform_keys():
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert "output_dir" not in cfg, "output_dir 由平台权威覆写，写了会被静默替换"
    assert not cfg.get("report_to"), "observability 由平台接管"
    assert all(not isinstance(v, dict) for v in cfg.values()), "TRL config 是平铺键值"
    assert cfg["num_generations"] >= 2, "GRPO 组相对优势需要每题多条采样"
    assert cfg["max_completion_length"] > 1024, "多轮预算须大于单轮值"
