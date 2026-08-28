"""verl Agent Loop RTL 示例实验的本地守护测试。

本地环境没有 verl 包，用 stub 替代 @function_tool 装饰器加载 tools.py——
只验证我们自己的工具/判分逻辑与 config 契约，不验证 verl 本身。
（与 test_verl_qa_tools_example.py 同一套做法。）
"""
from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "verl-grpo_qwen3.5-9b_rtl-agent_v1"

DESIGN = "module adder(input a, output b); assign b = a; endmodule"


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
        return _load("rtl_agent_tools", EXP / "tools.py")
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


# ── 工具 ────────────────────────────────────────────────────────────────────


def test_tools_reject_a_fragment_with_a_usable_message():
    """模型第一轮常只给一段 always 块。回一句能照做的话，别让它空转一轮。"""
    tools = _load_tools()
    for out in (tools.compile_rtl("assign b = a;"), tools.lint_rtl("assign b = a;")):
        assert "module" in out


def test_tools_strip_markdown_fences():
    """模型经常把代码包在 ``` 里传进来；原样丢给 iverilog 必然编不过，
    而那不是它设计能力的问题。"""
    tools = _load_tools()
    fenced = f"```verilog\n{DESIGN}\n```"
    assert tools.compile_rtl(fenced) == tools.compile_rtl(DESIGN)


def test_a_missing_binary_says_where_to_fix_it():
    """iverilog/verilator 都不在 verl 官方镜像里。报错要把人指到镜像那一步，
    而不是留下一句 FileNotFoundError。"""
    tools = _load_tools()
    if shutil.which("iverilog"):
        pytest.skip("本机装了 iverilog，这条只在没装时有意义")
    assert "镜像" in tools.compile_rtl(DESIGN)


@pytest.mark.skipif(not shutil.which("iverilog"), reason="需要 iverilog")
def test_compile_rtl_really_compiles():
    tools = _load_tools()
    assert "通过" in tools.compile_rtl(DESIGN)
    assert "失败" in tools.compile_rtl("module broken(); assign = ; endmodule")


def test_the_agent_only_gets_static_analysis_tools():
    """Verilog 有 $system —— 跑模型写的 testbench 等价于跑模型写的 shell。

    两层防线，都在这里守住：
      1. 注册的工具只有编译与 lint，没有能跑 testbench 的。给了它，agent 学到的
         就是「怎么试出判据」而不是「怎么设计电路」（VerilogEval 的公开/隐藏划分
         就是这个道理）。
      2. 调起来的可执行文件只有 iverilog 与 verilator，两者都只做静态分析。
         真正执行的 vvp 只出现在奖励那一侧，在临时目录里、不继承宿主环境、带超时。
    """
    tree = ast.parse((EXP / "tools.py").read_text(encoding="utf-8"))
    registered = {
        dec.args[0].value
        for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        for dec in node.decorator_list
        if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "function_tool"
    }
    assert registered == {"compile_rtl", "lint_rtl"}

    # argv 列表的第一项就是可执行文件名。
    executables = {
        node.elts[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.List) and node.elts
        and isinstance(node.elts[0], ast.Constant) and isinstance(node.elts[0].value, str)
    }
    assert executables <= {"iverilog", "verilator"}, f"意外的可执行文件: {executables}"


# ── 判分 ────────────────────────────────────────────────────────────────────


def test_reward_delegates_to_the_shared_three_stage_contract():
    from common.rewards import rtl_reward

    reward = _load("rtl_agent_reward", EXP / "reward.py")
    assert reward.FORMAT_PENALTY == rtl_reward.FORMAT_PENALTY


def test_reward_gives_format_penalty_when_there_is_no_code():
    """0 和「写了但编不过」同分的话，模型学不到「至少要输出一整个 module」。"""
    reward = _load("rtl_agent_reward", EXP / "reward.py")
    assert reward.compute_score("rtl_rl", "我觉得该用计数器。", "module tb; endmodule") == \
        reward.FORMAT_PENALTY


def test_reward_reads_the_top_module_from_extra_info():
    """yosys 综合要顶层名；prepare_data 把它放在 extra_info.top。"""
    reward = _load("rtl_agent_reward", EXP / "reward.py")
    from common.rewards import rtl_reward

    seen = {}
    original = rtl_reward.RtlReward

    class Spy(original):
        def __init__(self, **kw):
            seen.update(kw)
            super().__init__(**kw, runner=lambda *a: rtl_reward.ToolRun(0),
                             which=lambda _n: "/bin/x")

    rtl_reward.RtlReward = Spy
    try:
        reward.compute_score("rtl_rl", f"```verilog\n{DESIGN}\n```", "module tb; endmodule",
                             {"top": "adder"})
    finally:
        rtl_reward.RtlReward = original
    assert seen["top"] == "adder"


# ── config 契约 ─────────────────────────────────────────────────────────────


def test_config_declares_official_agent_loop_contract():
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    rollout = cfg["actor_rollout_ref"]["rollout"]
    assert rollout["mode"] == "async", "Agent Loop 依赖异步 rollout"
    assert rollout["multi_turn"]["enable"] is True
    assert rollout["agent"]["default_agent_loop"] == "tool_agent"
    assert cfg["data"]["return_raw_chat"] is True
    rel = rollout["multi_turn"]["function_tool_path"]
    assert (ROOT / rel).is_file(), f"function_tool_path 必须指向仓库内文件: {rel}"

    reward_rel = cfg["custom_reward_function"]["path"]
    assert (ROOT / reward_rel).is_file()
    # verl 0.9 V1 只读 reward.custom_reward_function；顶层旧键不够。
    v1 = cfg["reward"]["custom_reward_function"]
    assert v1["path"] == reward_rel
    assert v1["name"] == cfg["custom_reward_function"]["name"] == "compute_score"


def test_config_keeps_dataset_refs_out_of_hydra_overrides():
    """verl adapter 把整份 config 扁平化成 hydra override，data.train 会变非法键。"""
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert "train" not in cfg["data"] and "validation" not in cfg["data"]
    assert "path" not in cfg["actor_rollout_ref"].get("model", {})


def test_response_budget_leaves_room_for_a_whole_module():
    """轨迹被截断 → 拿不到完整的 ```verilog 块 → 判 FORMAT_PENALTY，
    看起来像模型不听话，其实是预算不够。Verilog 比 QA 的答案长得多。"""
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["data"]["max_response_length"] >= 3072


def test_lora_uses_the_flat_keys_fsdp_actually_reads():
    """嵌套 lora.* 只有 Megatron 后端读。写成 lora.rank 会让 LoRA 静默失效
    退回全参数，然后在单卡上 OOM。"""
    model = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))[
        "actor_rollout_ref"]["model"]
    assert model["lora_rank"] > 0 and "lora" not in model


# ── 数据准备 ────────────────────────────────────────────────────────────────


def _prepare(tmp_path, rows):
    src = tmp_path / "src"
    src.mkdir()
    for split in ("train", "val"):
        (src / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
    out = tmp_path / "out"
    return subprocess.run(
        [sys.executable, str(EXP / "prepare_data.py"),
         "--data-dir", str(src), "--out-dir", str(out)],
        capture_output=True, text=True, cwd=str(ROOT),
    ), out


def test_prepare_data_emits_the_agent_name_verl_routes_on(tmp_path):
    """缺 agent_name 时异步模式静默回落单轮、工具永不触发（verl issue #2986）。"""
    import pandas as pd

    proc, out = _prepare(tmp_path, [
        {"spec": "写一个反相器", "testbench": "module tb; initial $finish; endmodule", "top": "inv"},
    ])
    assert proc.returncode == 0, proc.stderr
    df = pd.read_parquet(out / "train.parquet")
    assert set(df["agent_name"]) == {"tool_agent"}
    assert df.iloc[0]["reward_model"]["ground_truth"].startswith("module tb")
    assert df.iloc[0]["extra_info"]["top"] == "inv"


def test_prepare_data_refuses_a_row_without_a_testbench(tmp_path):
    """没有 testbench 的题会让奖励退化成「编译得过就给 0.1」。
    宁可在建数据集时炸，也别让它混进训练集之后只表现为 reward 上不去。"""
    proc, _ = _prepare(tmp_path, [{"spec": "写一个反相器", "testbench": "", "top": "inv"}])
    assert proc.returncode != 0
    assert "testbench" in proc.stderr


def test_the_prompt_never_carries_the_hidden_testbench(tmp_path):
    """testbench 是判卷标准。进了 prompt，agent 就是在对着答案写代码。"""
    import pandas as pd

    tb = "module tb; initial $display(\"SECRET\"); endmodule"
    proc, out = _prepare(tmp_path, [{"spec": "写一个反相器", "testbench": tb, "top": "inv"}])
    assert proc.returncode == 0, proc.stderr
    prompt = json.dumps(list(pd.read_parquet(out / "train.parquet").iloc[0]["prompt"]),
                        ensure_ascii=False)
    assert "SECRET" not in prompt
