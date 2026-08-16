"""envkit 内核：标签/工具纯函数、Gym server 协议与参考适配器。"""
from __future__ import annotations

import pytest

from common.envkit.adapters import QADocsAdapter, RtlEnvAdapter, ToolEnvAdapter
from common.envkit.tags import extract_tag
from common.envkit.tools import TOOLS, safe_eval, tool_calc, tool_python, tool_search

# ── 标签解析（全环境唯一实现的语义锁定） ────────────────────────────────────────


def test_extract_tag_takes_last_and_tolerates_missing_close():
    assert extract_tag("a <tool>x</tool> b <tool>y</tool>", "tool") == "y"
    # stop_strings 截断：闭标签缺失仍取全文
    assert extract_tag("推理…<answer>42", "answer") == "42"
    assert extract_tag("没有标签", "tool") is None
    # 成对但为空 → 空串（区别于 None：模型给了空标签）
    assert extract_tag("<answer></answer>", "answer") == ""
    # 开标签后无内容且无闭标签 → None
    assert extract_tag("<answer>", "answer") is None


def _import_env_module(name: str):
    """环境模块 import 期要 ray/torch/nemo_rl（只在集群装）；按仓库既有做法打桩。"""
    import importlib
    import sys
    import types
    from typing import Generic, TypeVar

    ray_mod = types.ModuleType("ray")
    ray_mod.remote = lambda cls: cls
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = "float32"
    torch_mod.bool = "bool"
    torch_mod.tensor = lambda data, dtype=None: list(data)
    data_interfaces = types.ModuleType("nemo_rl.data.interfaces")
    data_interfaces.LLMMessageLogType = list
    batched = types.ModuleType("nemo_rl.distributed.batched_data_dict")
    batched.BatchedDataDict = dict
    env_interfaces = types.ModuleType("nemo_rl.environments.interfaces")
    _T = TypeVar("_T")

    class EnvironmentInterface(Generic[_T]):
        pass

    class EnvironmentReturn:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    env_interfaces.EnvironmentInterface = EnvironmentInterface
    env_interfaces.EnvironmentReturn = EnvironmentReturn
    stubs = {
        "ray": ray_mod,
        "torch": torch_mod,
        "nemo_rl": types.ModuleType("nemo_rl"),
        "nemo_rl.data": types.ModuleType("nemo_rl.data"),
        "nemo_rl.data.interfaces": data_interfaces,
        "nemo_rl.distributed": types.ModuleType("nemo_rl.distributed"),
        "nemo_rl.distributed.batched_data_dict": batched,
        "nemo_rl.environments": types.ModuleType("nemo_rl.environments"),
        "nemo_rl.environments.interfaces": env_interfaces,
    }
    for key, mod in stubs.items():
        sys.modules.setdefault(key, mod)
    return importlib.import_module(f"common.environments.{name}")


def test_envs_delegate_to_envkit_tag_parser():
    """两个训练环境的 _extract_tag 必须是 envkit 的薄壳（防再度漂移）。"""
    example_tool_env = _import_env_module("example_tool_env")
    qa_docs_agent_env = _import_env_module("qa_docs_agent_env")

    for fn in (example_tool_env._extract_tag, qa_docs_agent_env._extract_tag):
        assert fn("<tool>calc: 1+1", "tool") == "calc: 1+1"
        assert fn("x<answer>9</answer>", "answer") == "9"


# ── 工具内核 ─────────────────────────────────────────────────────────────────


def test_safe_eval_arithmetic_only():
    assert safe_eval("2+3*4") == 14
    with pytest.raises(Exception):
        safe_eval("__import__('os')")


def test_tool_calc_and_search():
    assert tool_calc("2**5", {}) == "32"
    kb = {"苹果 单价": "苹果每斤 5 元", "香蕉 单价": "香蕉每斤 3 元"}
    assert "苹果" in tool_search("苹果 单价", {"kb": kb})
    assert tool_search("", {"kb": kb}).startswith("search 错误")


def test_tool_python_routes_through_sandbox_provider(monkeypatch):
    """python 工具经 SandboxProvider 分派：注入假 provider 验证。"""
    import starforge_sdk.sandbox as sandbox_mod

    class _FakeProvider:
        name = "fake"

        def run_code(self, code, *, timeout=10.0, language="python"):
            from starforge_sdk.sandbox import SandboxResult

            assert code == "print('hi')"
            assert timeout == 5.0
            return SandboxResult(stdout="hi\n")

    monkeypatch.setattr(sandbox_mod, "get_sandbox_provider", lambda: _FakeProvider())
    assert tool_python("print('hi')", {}) == "hi"


def test_tool_python_sandbox_failure_is_model_readable(monkeypatch):
    import starforge_sdk.sandbox as sandbox_mod

    class _Broken:
        def run_code(self, *a, **k):
            from starforge_sdk.sandbox import SandboxResult

            return SandboxResult(error="沙箱服务不可达: boom")

    monkeypatch.setattr(sandbox_mod, "get_sandbox_provider", lambda: _Broken())
    out = tool_python("print(1)", {})
    assert out.startswith("python 执行失败")


def test_tool_registry_is_shared_with_training_env():
    example_tool_env = _import_env_module("example_tool_env")
    assert example_tool_env.TOOLS is TOOLS


# ── Gym server 协议 ──────────────────────────────────────────────────────────


@pytest.fixture()
def gym_client():
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from common.envkit.gym_server import create_gym_app

    app = create_gym_app(ToolEnvAdapter())
    return TestClient(app)


def test_gym_seed_tool_verify_roundtrip(gym_client):
    seed = gym_client.post("/seed_session", json={"payload": {
        "question": "2+3*4=?", "target": 14, "kb": {"提示": "先乘后加"},
    }})
    assert seed.status_code == 200, seed.text
    sid = seed.json()["session_id"]
    assert set(seed.json()["tools"]) == {"calc", "python", "search"}

    tool = gym_client.post("/tools/calc", json={"session_id": sid, "arg": "2+3*4"})
    assert tool.json()["observation"] == "14"

    verify = gym_client.post("/verify", json={"session_id": sid, "response": "14"})
    body = verify.json()
    assert body["reward"] == 1.0 and body["correct"] is True
    assert body["tool_calls"] == 1
    # verify 即终局：会话销毁
    again = gym_client.post("/verify", json={"session_id": sid, "response": "14"})
    assert again.status_code == 404


def test_gym_sessions_are_isolated(gym_client):
    a = gym_client.post("/seed_session", json={"payload": {"target": 1, "kb": {"k": "甲"}}}).json()["session_id"]
    b = gym_client.post("/seed_session", json={"payload": {"target": 2, "kb": {"k": "乙"}}}).json()["session_id"]
    obs_a = gym_client.post("/tools/search", json={"session_id": a, "arg": "k"}).json()["observation"]
    obs_b = gym_client.post("/tools/search", json={"session_id": b, "arg": "k"}).json()["observation"]
    assert obs_a == "甲" and obs_b == "乙"
    assert gym_client.post("/verify", json={"session_id": a, "response": "1"}).json()["reward"] == 1.0
    assert gym_client.post("/verify", json={"session_id": b, "response": "1"}).json()["reward"] == 0.0


def test_gym_rejects_unknown_tool_and_bad_payload(gym_client):
    sid = gym_client.post("/seed_session", json={"payload": {"target": 1}}).json()["session_id"]
    assert gym_client.post("/tools/nope", json={"session_id": sid, "arg": ""}).status_code == 404
    assert gym_client.post("/tools/calc", json={"session_id": "ghost", "arg": "1"}).status_code == 404
    # payload 校验失败 → 422
    assert gym_client.post("/seed_session", json={"payload": {}}).status_code == 422


# ── 参考适配器 ────────────────────────────────────────────────────────────────


def test_qa_docs_adapter_verify_uses_qa_reward(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from common.envkit.gym_server import create_gym_app

    hits = []

    def fake_search(query: str) -> str:
        hits.append(query)
        return f"资料：{query} 的答案在文档 3"

    client = TestClient(create_gym_app(QADocsAdapter(fake_search)))
    sid = client.post("/seed_session", json={"payload": {
        "question": "TCP 三次握手的第一步是什么", "expected": "[short] SYN",
    }}).json()["session_id"]
    obs = client.post("/tools/docs_search", json={"session_id": sid, "arg": "三次握手"}).json()
    assert "资料" in obs["observation"] and hits == ["三次握手"]
    res = client.post(
        "/verify",
        json={"session_id": sid, "response": "第一步是发送 SYN 包。\\boxed{SYN}"},
    ).json()
    assert res["reward"] > 0
    assert res["search_count"] == 1


def test_rtl_skeleton_exposes_compile_simulate_protocol():
    adapter = RtlEnvAdapter()
    ctx = adapter.seed_session({"spec": "计数器", "testbench": "tb.v"})
    assert adapter.tools["simulate"]("", ctx).startswith("simulate 错误")
    assert "compile ok" in adapter.tools["compile"]("module counter; endmodule", ctx)
    assert "simulate ok" in adapter.tools["simulate"]("", ctx)
    verdict = adapter.verify(ctx, "module counter; endmodule")
    assert verdict["reward"] == 0.0 and "骨架" in verdict["reason"]
