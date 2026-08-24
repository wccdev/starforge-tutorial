"""verl Agent Loop 示例实验（qa-tools）的本地守护测试。

本地环境没有 verl 包，用 stub 替代 @function_tool 装饰器加载 tools.py——
只验证我们自己的检索/判分逻辑与 config 契约，不验证 verl 本身。
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "verl-grpo_qwen3.5-9b_qa-tools_v1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tools():
    stub = types.ModuleType("verl.tools.function_tool")
    stub.function_tool = lambda arg=None: (arg if callable(arg) else (lambda fn: fn))
    saved = {k: sys.modules.get(k) for k in ("verl", "verl.tools", "verl.tools.function_tool")}
    sys.modules["verl"] = types.ModuleType("verl")
    sys.modules["verl.tools"] = types.ModuleType("verl.tools")
    sys.modules["verl.tools.function_tool"] = stub
    try:
        return _load("qa_tools_example", EXP / "tools.py")
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def test_search_docs_hits_and_placeholder(tmp_path, monkeypatch):
    tools = _load_tools()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cmp_process.md").write_text(
        "# CMP 制程\n铜 CMP 分为主抛与精抛两步，主抛负责去除大部分铜层。\n精抛用于表面平整。\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCS_DIR", str(docs))
    hit = tools.search_docs("CMP 主抛 铜")
    assert hit.startswith("[检索结果]") and "cmp_process.md" in hit and "主抛" in hit

    miss = tools.search_docs("量子退火")
    assert "未找到" in miss

    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "no-such-dir"))
    assert "不存在" in tools.search_docs("任何词")


def test_reward_scoring_matches_qa_contract():
    """ground_truth 必须按题库真实格式（带 [type] 前缀）判分。

    前缀是必须的：题库 expected_answer 长这样，剥掉前缀或拿整串去精确匹配，
    正确答案也会判 0 —— 组内奖励恒等 → GRPO 优势全 0 → 训练照跑但学不到东西。
    """
    reward = _load("qa_tools_reward", EXP / "reward.py")
    assert reward.compute_score("qa_rl", "答案 \\boxed{B}", "[single] B") == 1.0
    assert reward.compute_score("qa_rl", "\\boxed{C}", "[single] B") == 0.0
    assert reward.compute_score("qa_rl", "\\boxed{A}", "[bool] A") == 1.0
    assert reward.compute_score("qa_rl", "\\boxed{a,c}", "[multiple] A,C") == 1.0
    assert reward.compute_score("qa_rl", "\\boxed{Check}", "[fill] Check") == 1.0
    assert reward.compute_score("qa_rl", "没有盒子", "[single] B") == reward.FORMAT_PENALTY


def test_reward_delegates_to_shared_qa_contract():
    """判分口径必须与 nemo-rl 对照实验同源，否则 A/B 不可比。"""
    from common.rewards import qa_reward

    reward = _load("qa_tools_reward", EXP / "reward.py")
    assert reward.FORMAT_PENALTY == qa_reward.FORMAT_PENALTY
    # 检索会把资料原文回灌进上下文，简答题若按整段回答统计覆盖率就能靠复述刷分。
    assert qa_reward.SHORT_SCOPE == "boxed"
    gt = "[short] 低温/掺杂 ||| 纯度高 ||| 横向扩散小"
    parroted = "资料提到低温、纯度高、横向扩散小。\\boxed{不知道}"
    assert reward.compute_score("qa_rl", parroted, gt) == 0.0


def test_config_declares_official_agent_loop_contract():
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    rollout = cfg["actor_rollout_ref"]["rollout"]
    assert rollout["mode"] == "async", "Agent Loop 依赖异步 rollout"
    assert rollout["multi_turn"]["enable"] is True
    assert rollout["agent"]["default_agent_loop"] == "tool_agent"
    assert cfg["data"]["return_raw_chat"] is True
    for key in ("function_tool_path",):
        rel = rollout["multi_turn"][key]
        assert (ROOT / rel).is_file(), f"{key} 指向的文件必须存在于仓库内: {rel}"
    reward_rel = cfg["custom_reward_function"]["path"]
    assert (ROOT / reward_rel).is_file()


def test_config_keeps_dataset_refs_out_of_hydra_overrides():
    """verl adapter 把整份 config 扁平化成 hydra override，data.train 会变非法键。

    config_mode=hydra-overrides 只排除模型/数据/topology 那几个权威绑定键；
    `data.train.dataset` 会原样传成 `data.train.dataset=...`，而 verl 的 data
    配置没有 train 子节点 → Hydra struct 模式启动即报错。数据集引用只能走
    提交时的 --train-dataset/--validation-dataset。
    """
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    data = cfg["data"]
    assert "train" not in data and "validation" not in data
    # 模型同理由平台权威键绑定，不能写在 config 里。
    assert "path" not in cfg["actor_rollout_ref"].get("model", {})


def test_config_prompt_budget_fits_question_bank():
    """max_prompt_length 要容得下最长题面；verl 默认 truncation=error 会直接中断。"""
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["data"]["max_prompt_length"] >= 2048
    assert cfg["data"]["filter_overlong_prompts"] is True


def test_prepare_data_injects_agent_name(tmp_path):
    prepare = _load("qa_tools_prepare", EXP / "prepare_data.py")
    src = tmp_path / "train.jsonl"
    src.write_text('{"query": "问题？", "expected_answer": "[single] B"}\n', encoding="utf-8")
    row = next(prepare._rows(src, "train"))
    assert row["agent_name"] == "tool_agent", "缺 agent_name 会静默回落单轮（issue #2986）"
    # [type] 前缀原样透传给 reward.py 分派判分，不能在这里剥掉。
    assert row["reward_model"]["ground_truth"] == "[single] B"
    assert row["prompt"][-1]["content"] == "问题？"
