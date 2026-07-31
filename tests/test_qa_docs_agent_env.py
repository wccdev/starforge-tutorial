"""多轮检索环境 QADocsAgentEnv 的奖励用例：守住「检索加分只属于训练，验证是纯判分」。

盯的问题：验证曾与训练共用同一个环境实例，于是 `search_step_reward` / `answer_search_bonus` /
`no_answer_penalty` 也进了验证分。NeMo-RL 的 `validation/accuracy` 就是 `mean(total_reward)`，
而 `total_reward` 是逐轮奖励累加 —— 结果「用了工具」本身白送分，验证分虚高，且与无工具 baseline
不再同尺度。修法是验证单独建一个 `make_eval_cfg()` 派生的环境实例。

环境模块 import 期要 ray / torch / nemo_rl，这三者只在集群装；本地按仓库既有做法打桩，
被测的判分与 shaping 逻辑都是纯 Python，不受影响。
"""
from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from typing import Generic, TypeVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# 用 QADocsAgentEnv 的实验：它们的 grpo_train 必须收到两个不同的环境映射
AGENT_RUNS = [
    "experiments/grpo_qwen3.5-9b_qa-rl-agent_v3/run.py",
    "experiments/grpo_qwen3.5-9b_qa-rl-agent_gb10_v1/run.py",
    "experiments/maxrl_qwen3.5-9b_qa-rl-agent_v2/run.py",
]

TRAIN_CFG = {
    "use_judge": False,  # 走规则判分：离线可跑、结果确定
    "max_turns": 3,
    "search_step_reward": 0.05,
    "answer_search_bonus": 0.1,
    "search_bonus_min_score": 1.0,
    "no_answer_penalty": 0.2,
    "format_error_penalty": 0.02,
    "invalid_search_penalty": 0.02,
}


def _stub_modules() -> dict[str, types.ModuleType]:
    """ray / torch / nemo_rl 的最小替身：只覆盖环境模块 import 与 step() 真正用到的那几个名字。"""
    ray_mod = types.ModuleType("ray")
    ray_mod.remote = lambda cls: cls  # 本测试直接实例化普通类，不起 actor

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

    return {
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


@pytest.fixture()
def env_mod(monkeypatch):
    """装好替身后重新 import 环境模块（`common.environments` 的 __init__ 也会连带 import 同批依赖）。"""
    for name, mod in _stub_modules().items():
        monkeypatch.setitem(sys.modules, name, mod)
    for name in list(sys.modules):
        if name == "common.environments" or name.startswith("common.environments."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("common.environments.qa_docs_agent_env")


def _answer_turn(env_mod, cfg, *, completion: str, did_search: bool) -> float:
    """跑一轮「模型给出最终答案」，返回该样本这一步的奖励。"""
    env = env_mod.QADocsAgentEnv(cfg=cfg)
    ret = env.step(
        [[{"role": "assistant", "content": completion}]],
        [{
            "expected_answer": "[single] B",
            "query": "题面",
            "num_turns": 1,
            "max_turns": int(cfg["max_turns"]),
            "did_search": did_search,
        }],
    )
    return ret.rewards[0]


def test_make_eval_cfg_zeroes_shaping_only(env_mod):
    eval_cfg = env_mod.make_eval_cfg(TRAIN_CFG)
    assert eval_cfg["search_step_reward"] == 0.0
    assert eval_cfg["answer_search_bonus"] == 0.0
    assert eval_cfg["no_answer_penalty"] == 0.0
    assert eval_cfg["format_error_penalty"] == 0.0
    assert eval_cfg["invalid_search_penalty"] == 0.0
    # 检索后端与判分方式必须保持一致，验证才只差「工具不加分」这一件事
    assert eval_cfg["use_judge"] is False
    assert eval_cfg["max_turns"] == 3
    assert eval_cfg["search_bonus_min_score"] == 1.0
    # 不能就地改训练 cfg（同一份 dict 还要拿去建训练环境）
    assert TRAIN_CFG["answer_search_bonus"] == 0.1


def test_train_gives_search_bonus_but_eval_does_not(env_mod):
    """检索过且答对：训练 1.0+0.1，验证必须就是 1.0（否则 validation/accuracy 会 >1）。"""
    assert _answer_turn(
        env_mod, TRAIN_CFG, completion=r"根据资料 \boxed{B}", did_search=True
    ) == pytest.approx(1.1)
    assert _answer_turn(
        env_mod, env_mod.make_eval_cfg(TRAIN_CFG), completion=r"根据资料 \boxed{B}", did_search=True
    ) == pytest.approx(1.0)


def test_wrong_answer_scores_zero_in_both(env_mod):
    """答错就是 0：检索加成只在 base 分达到 search_bonus_min_score 时才给，不会救错答案。"""
    for cfg in (TRAIN_CFG, env_mod.make_eval_cfg(TRAIN_CFG)):
        assert _answer_turn(
            env_mod, cfg, completion=r"\boxed{C}", did_search=True
        ) == pytest.approx(0.0)


def test_no_answer_penalty_is_train_only(env_mod):
    """超轮仍不作答：训练扣分防刷分；验证里这只该算"没答对"= 0，不该把准确率压成负数。"""
    def run(cfg):
        env = env_mod.QADocsAgentEnv(cfg=cfg)
        ret = env.step(
            [[{"role": "assistant", "content": "还在想"}]],
            [{
                "expected_answer": "[single] B",
                "query": "题面",
                "num_turns": 3,
                "max_turns": 3,
                "did_search": True,
            }],
        )
        assert ret.terminateds[0] is True
        return ret.rewards[0]

    assert run(TRAIN_CFG) == pytest.approx(-0.2)
    assert run(env_mod.make_eval_cfg(TRAIN_CFG)) == pytest.approx(0.0)


def test_retrieval_step_reward_is_train_only(env_mod, monkeypatch):
    """有效检索的即时奖励同理：训练 +0.05，验证 0。"""
    monkeypatch.setattr(env_mod, "docs_search", lambda q: "【cmp.md】\nL12: 主抛去除大部分铜层")

    def run(cfg):
        env = env_mod.QADocsAgentEnv(cfg=cfg)
        ret = env.step(
            [[{"role": "assistant", "content": "<search>CMP 铜</search>"}]],
            [{
                "expected_answer": "[single] B",
                "query": "题面",
                "num_turns": 0,
                "max_turns": 3,
                "did_search": False,
            }],
        )
        assert ret.terminateds[0] is False
        assert ret.metadata[0]["did_search"] is True  # 取回了资料 → 记上，供后续答对加成判断
        return ret.rewards[0]

    assert run(TRAIN_CFG) == pytest.approx(0.05)
    assert run(env_mod.make_eval_cfg(TRAIN_CFG)) == pytest.approx(0.0)


def test_search_without_close_tag_still_retrieves(env_mod, monkeypatch):
    """vLLM stop_strings=['</search>'] 常不把闭标签写入生成文本；不能因此误判格式不对。"""
    seen: list[str] = []

    def fake_search(q: str) -> str:
        seen.append(q)
        return "【cmp.md】\nL12: 主抛去除大部分铜层"

    monkeypatch.setattr(env_mod, "docs_search", fake_search)
    env = env_mod.QADocsAgentEnv(cfg=env_mod.make_eval_cfg(TRAIN_CFG))
    ret = env.step(
        [[{"role": "assistant", "content": "先查一下\n<search>CMP 铜 去除"}]],
        [{
            "expected_answer": "[single] B",
            "query": "题面",
            "num_turns": 0,
            "max_turns": 2,
            "did_search": False,
        }],
    )
    assert seen == ["CMP 铜 去除"]
    assert ret.terminateds[0] is False
    assert ret.metadata[0]["did_search"] is True
    assert "[检索结果]" in ret.observations[0]["content"]
    assert "格式不对" not in ret.observations[0]["content"]


def test_bad_tool_format_penalty_is_train_only(env_mod):
    """无标签输出和空 search 只在训练期扣分；验证必须仍是纯正确率。"""
    metadata = [{
        "expected_answer": "[single] B",
        "query": "题面",
        "num_turns": 0,
        "max_turns": 3,
        "did_search": False,
    }]

    for completion, expected in (("没有工具标签", -0.02), ("<search></search>", -0.02)):
        train = env_mod.QADocsAgentEnv(cfg=TRAIN_CFG).step(
            [[{"role": "assistant", "content": completion}]], metadata
        )
        evaluate = env_mod.QADocsAgentEnv(cfg=env_mod.make_eval_cfg(TRAIN_CFG)).step(
            [[{"role": "assistant", "content": completion}]], metadata
        )
        assert train.rewards[0] == pytest.approx(expected)
        assert evaluate.rewards[0] == pytest.approx(0.0)


@pytest.mark.parametrize("rel_path", AGENT_RUNS)
def test_run_script_passes_a_separate_val_env(rel_path):
    """守住接线：grpo_train 的第 7/8 个参数是 task_to_env / val_task_to_env，不能是同一个。

    环境侧再对，只要 run.py 把训练环境传两遍（历史写法），验证分照样带 shaping。
    """
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "grpo_train"
    ]
    assert len(calls) == 1, f"{rel_path} 里应恰好有一处 grpo_train 调用"
    args = calls[0].args
    assert len(args) >= 8, f"{rel_path} 的 grpo_train 位置参数不足 8 个"
    train_env, val_env = ast.unparse(args[6]), ast.unparse(args[7])
    assert train_env != val_env, (
        f"{rel_path} 把 {train_env} 同时当训练与验证环境；"
        "验证要用 make_eval_cfg() 另建实例，否则检索加分会算进 validation/accuracy"
    )
    assert "make_eval_cfg" in source, f"{rel_path} 的验证环境应由 make_eval_cfg() 派生 cfg"


@pytest.mark.parametrize(
    "rel_path",
    [
        "experiments/grpo_qwen3.5-9b_qa-rl-agent_v3/run.py",
        "experiments/grpo_qwen3.5-9b_qa-rl-agent_gb10_v1/run.py",
    ],
)
def test_v07_setup_result_is_fully_unpacked(rel_path):
    """v0.7 setup() 返回 13 项；漏掉 MOPD teacher 字段会在昂贵的 worker 初始化后才失败。"""
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    setup_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "setup"
    ]
    assert len(setup_assignments) == 1
    target = setup_assignments[0].targets[0]
    assert isinstance(target, ast.Tuple)
    assert len(target.elts) == 13, (
        f"{rel_path} 必须完整解包 NeMo-RL v0.7 setup() 的 13 项返回值"
    )
