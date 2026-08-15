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
    "experiments/maxrl_qwen3.5-9b_qa-rl-agent_v2/run.py",
]

TRAIN_CFG = {
    "use_judge": False,  # 走规则判分：离线可跑、结果确定
    "max_turns": 3,
    "search_step_reward": 0.05,
    "answer_search_bonus": 0.1,
    "search_bonus_min_score": 1.0,
    "no_answer_penalty": 0.2,
    "no_search_answer_penalty": 0.1,
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
    assert eval_cfg["no_search_answer_penalty"] == 0.0
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


def test_no_search_answer_penalty_is_train_only(env_mod):
    """未检索就作答：训练扣分压过闭卷捷径；验证仍是纯判分（答对=1.0）。"""
    assert _answer_turn(
        env_mod, TRAIN_CFG, completion=r"\boxed{B}", did_search=False
    ) == pytest.approx(0.9)
    assert _answer_turn(
        env_mod, env_mod.make_eval_cfg(TRAIN_CFG), completion=r"\boxed{B}", did_search=False
    ) == pytest.approx(1.0)
    # 检索后答对净收益应高于闭卷答对，避免策略退化为不用工具
    searched = _answer_turn(
        env_mod, TRAIN_CFG, completion=r"\boxed{B}", did_search=True
    )
    closed = _answer_turn(
        env_mod, TRAIN_CFG, completion=r"\boxed{B}", did_search=False
    )
    assert searched > closed


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


# ─────────── 超轮惩罚的「可达性」：env.max_turns 必须 < grpo.max_rollout_turns ───────────
# NeMo-RL 的 rollout 是 `for turn in range(max_rollout_turns)`，跑满即退出、不会再调一次 step()，
# 只把样本标成 max_turns_reached（既不判分也不扣分，overlong_filtering 也只管 truncated）。
# num_turns 每轮 +1 ⇒ 第 k 轮进 step() 时 num_turns == k-1。
# 所以 env.max_turns == max_rollout_turns 时，「超轮不作答」分支永远打不到，
# 「一直检索不作答」的净收益是 +N×search_step_reward 的正数 —— 一条零风险刷分路径。


def _simulate_rollout(env_mod, cfg, *, max_rollout_turns: int) -> float:
    """模拟 NeMo-RL 的多轮循环：模型每轮都只检索、从不作答。返回整条轨迹累计奖励。"""
    env = env_mod.QADocsAgentEnv(cfg=cfg)
    meta = {
        "expected_answer": "[single] B",
        "query": "题面",
        "num_turns": 0,
        "max_turns": int(cfg["max_turns"]),
        "did_search": False,
    }
    total = 0.0
    for _ in range(max_rollout_turns):  # ← 对应 rollouts.py 的 for turn in range(...)
        ret = env.step([[{"role": "assistant", "content": "<search>关键词</search>"}]], [meta])
        total += ret.rewards[0]
        if ret.terminateds[0]:
            break
        meta = ret.metadata[0]
    return total


def test_no_answer_penalty_unreachable_when_max_turns_equals_rollout_turns(env_mod, monkeypatch):
    """回归：max_turns == max_rollout_turns 时，「只检索不作答」净收益为【正】——这正是要避免的配置。"""
    monkeypatch.setattr(env_mod, "docs_search", lambda q: "【cmp.md】\nL12: 有资料")
    bad_cfg = {**TRAIN_CFG, "max_turns": 3}
    total = _simulate_rollout(env_mod, bad_cfg, max_rollout_turns=3)
    assert total == pytest.approx(0.15)  # 3×0.05，惩罚分支从未触发
    assert total > 0, "配置错误时刷分路径成立——这是本用例要固定住的反例"


def test_no_answer_penalty_reachable_when_max_turns_is_one_less(env_mod, monkeypatch):
    """修法：max_turns = max_rollout_turns - 1，第 3 轮触发 -0.2，「只检索不作答」净收益转负。"""
    monkeypatch.setattr(env_mod, "docs_search", lambda q: "【cmp.md】\nL12: 有资料")
    good_cfg = {**TRAIN_CFG, "max_turns": 2}
    total = _simulate_rollout(env_mod, good_cfg, max_rollout_turns=3)
    assert total == pytest.approx(0.05 + 0.05 - 0.2)
    assert total < 0, "只检索不作答必须净亏，否则会被 RL 当成零风险刷分策略"


def test_answering_on_last_turn_is_still_graded(env_mod):
    """max_turns=2 不能误伤：最后一轮正常作答仍走判分（step() 里 boxed 分支排在超轮分支之前）。"""
    env = env_mod.QADocsAgentEnv(cfg={**TRAIN_CFG, "max_turns": 2})
    ret = env.step(
        [[{"role": "assistant", "content": r"根据资料 \boxed{B}"}]],
        [{
            "expected_answer": "[single] B",
            "query": "题面",
            "num_turns": 2,  # 已达上限，但这一轮给出了答案
            "max_turns": 2,
            "did_search": True,
        }],
    )
    assert ret.terminateds[0] is True
    assert ret.rewards[0] > 0.9  # 判分为 1.0（+检索加成），不是 -no_answer_penalty


# ─────────── 检索加成：阈值与按分数缩放 ───────────


def test_search_bonus_covers_partial_credit_question_types(env_mod):
    """min_score=1.0 会让 fill/short/multiple（最依赖检索的题型）永远拿不到检索加成。"""
    cfg_strict = {**TRAIN_CFG, "search_bonus_min_score": 1.0, "search_bonus_scaled": False}
    cfg_fixed = {**TRAIN_CFG, "search_bonus_min_score": 0.5, "search_bonus_scaled": True}

    def multi_2of3(cfg):  # gold A,C,D 答出 A,C → base 分 2/3
        env = env_mod.QADocsAgentEnv(cfg=cfg)
        ret = env.step(
            [[{"role": "assistant", "content": r"\boxed{A,C}"}]],
            [{
                "expected_answer": "[multiple] A,C,D",
                "query": "题面",
                "num_turns": 1,
                "max_turns": int(cfg["max_turns"]),
                "did_search": True,
            }],
        )
        return ret.rewards[0]

    assert multi_2of3(cfg_strict) == pytest.approx(2 / 3)          # 检索白用了，零加成
    assert multi_2of3(cfg_fixed) == pytest.approx(2 / 3 + 0.1 * (2 / 3))  # 按得分比例给


def test_search_bonus_scaled_is_monotonic_in_score(env_mod):
    """按比例缩放不能反转顺序：答得越好，总分越高。"""
    cfg = {**TRAIN_CFG, "search_bonus_min_score": 0.5, "search_bonus_scaled": True}

    def score(boxed, gold):
        env = env_mod.QADocsAgentEnv(cfg=cfg)
        return env.step(
            [[{"role": "assistant", "content": rf"\boxed{{{boxed}}}"}]],
            [{
                "expected_answer": gold,
                "query": "题面",
                "num_turns": 1,
                "max_turns": int(cfg["max_turns"]),
                "did_search": True,
            }],
        ).rewards[0]

    assert score("A,C,D", "[multiple] A,C,D") > score("A,C", "[multiple] A,C,D")


# ─────────── BM25 相关度下限 ───────────


def test_bm25_rejects_low_relevance_instead_of_returning_noise(env_mod, tmp_path, monkeypatch):
    """查不到相关资料时必须返回「未检索到」，而不是硬塞 Top-K 噪声。

    这是「模型学到检索没用」的一大来源：拿回一堆不相关片段，既误导作答，又照拿检索奖励。
    """
    (tmp_path / "cmp.md").write_text(
        "# CMP 制程\n\n铜 CMP 分为主抛与精抛两步，主抛负责去除大部分铜层。\n",
        encoding="utf-8",
    )
    (tmp_path / "safety.md").write_text(
        "# 高处作业\n\n登高作业务必佩戴安全帽和安全带，脚手架需经验收。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(env_mod, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(env_mod, "_BM25_CACHE", {})

    hit = env_mod.docs_search("CMP 铜 主抛")
    assert "cmp.md" in hit and env_mod._is_useful_retrieval(hit)

    miss = env_mod.docs_search("OFD 电子发票 源文件 报销流程")
    assert "未检索到相关资料" in miss
    assert not env_mod._is_useful_retrieval(miss), "低相关检索不能算「有效检索」，否则照发检索奖励"


def test_bm25_relative_cutoff_drops_weak_chunks(env_mod, tmp_path, monkeypatch):
    """相对截断：只保留与 Top1 同量级的片段，避免弱相关块挤占回灌预算。"""
    (tmp_path / "a.md").write_text(
        "# 铜 CMP\n\n铜 CMP 主抛 精抛 铜 CMP 主抛 铜 CMP 主抛 去除铜层。\n",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text("# 杂项\n\n设备维护记录里提到过一次铜。\n", encoding="utf-8")
    monkeypatch.setattr(env_mod, "DOCS_DIR", str(tmp_path))
    monkeypatch.setattr(env_mod, "_BM25_CACHE", {})
    monkeypatch.setattr(env_mod, "BM25_REL_CUTOFF", 0.5)

    out = env_mod.docs_search("铜 CMP 主抛")
    assert "a.md" in out and "b.md" not in out


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
    """守住接线：bootstrap.run_grpo 的第 5/6 个位置参数是 task_to_env / val_task_to_env，不能是同一个。

    环境侧再对，只要 run.py 把训练环境传两遍（历史写法），验证分照样带 shaping。
    """
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_grpo"
    ]
    assert len(calls) == 1, f"{rel_path} 里应恰好有一处 bootstrap.run_grpo 调用"
    args = calls[0].args
    assert len(args) >= 6, (
        f"{rel_path} 的 run_grpo 必须显式给出第 6 个位置参数 val_task_to_env"
    )
    train_env, val_env = ast.unparse(args[4]), ast.unparse(args[5])
    assert train_env != val_env, (
        f"{rel_path} 把 {train_env} 同时当训练与验证环境；"
        "验证要用 make_eval_cfg() 另建实例，否则检索加分会算进 validation/accuracy"
    )
    assert "make_eval_cfg" in source, f"{rel_path} 的验证环境应由 make_eval_cfg() 派生 cfg"


def test_v07_setup_result_is_fully_unpacked_in_bootstrap():
    """v0.7 setup() 返回 13 项；漏掉 MOPD teacher 字段会在昂贵的 worker 初始化后才失败。

    解包已收敛到 common/bootstrap.py 的 run_grpo（各实验 run.py 不再自行解包），
    所以只需要守住这一处：显式长度守卫 + 13 元组解包。
    """
    source = (REPO_ROOT / "common" / "bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tuple_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Tuple)
        and isinstance(node.value, ast.Name)
        and node.value.id == "values"
    ]
    assert len(tuple_assignments) == 1, "run_grpo 应恰好有一处对 setup() 结果的元组解包"
    assert len(tuple_assignments[0].targets[0].elts) == 13, (
        "common/bootstrap.py 必须完整解包 NeMo-RL v0.7 setup() 的 13 项返回值"
    )
    assert "len(values) != 13" in source, "run_grpo 缺少 13 元组的显式长度守卫"


@pytest.mark.parametrize(
    "rel_path",
    AGENT_RUNS + [
        "experiments/grpo_qwen3.5-9b_qa-rl_v1/run.py",
        "experiments/agent-grpo_qwen3.5-9b_multitool_v1/run.py",
    ],
)
def test_grpo_run_scripts_delegate_to_bootstrap(rel_path):
    """所有 GRPO 实验入口不得自行调用 setup()/grpo_train()——样板只允许存在于 bootstrap。"""
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert "bootstrap.run_grpo(" in source, f"{rel_path} 应通过 bootstrap.run_grpo 进入训练"
    direct_calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    for banned in ("grpo_train", "setup", "register_omegaconf_resolvers"):
        assert banned not in direct_calls, (
            f"{rel_path} 不应直接调用 {banned}——样板已收敛到 common/bootstrap.py"
        )
