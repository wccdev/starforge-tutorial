"""OPSD 对齐逻辑单测（纯 torch，不需要 GPU / NeMo-RL）。

OPSD 里唯一「写错了也不会报错、只会静静地学歪」的地方就是师生序列的位置对齐：
老师的 prompt（题目+参考解）比学生的 prompt（只有题目）长，两边的 response 段虽然
token 完全一样，但绝对下标差了 (Lq - Lp)。所以这里把对齐当成契约来测。
"""

import pytest

# 本机 lab CLI venv 刻意不装 torch（客户端无需 GPU 依赖）；这些用例在训练容器 / CI 里跑。
torch = pytest.importorskip("torch", reason="需要 torch，在训练容器内运行")

from types import SimpleNamespace  # noqa: E402

from common.algorithms.opsd import (  # noqa: E402
    build_hint_inputs,
    config_vocab_size,
    realign_topk,
    split_prompt_response,
)


# ---------------------------------------------------------------- split
def test_split_prompt_response_basic():
    # 两条样本：prompt 长 3 / 2，response 长 2 / 3。
    token_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
        ]
    )
    lengths = torch.tensor([5, 5])
    prompt_lens, response_lens = split_prompt_response(token_mask, lengths)
    assert prompt_lens.tolist() == [3, 2]
    assert response_lens.tolist() == [2, 3]


def test_split_prompt_response_rejects_multi_turn():
    """response 段被工具结果打断 → 无法定义「把 prompt 换成参考解」，必须报错而不是猜。"""
    token_mask = torch.tensor([[0, 1, 1, 0, 1, 0]])
    with pytest.raises(ValueError, match="不连续"):
        split_prompt_response(token_mask, torch.tensor([6]))


def test_split_prompt_response_rejects_empty_response():
    with pytest.raises(ValueError, match="没有任何 response token"):
        split_prompt_response(torch.zeros(1, 4, dtype=torch.long), torch.tensor([4]))


# ---------------------------------------------------------------- hint 输入
def test_build_hint_inputs_preserves_response_tokens():
    """核心契约：换头之后 response 段必须逐 token 与学生一致。"""
    input_ids = torch.tensor(
        [
            [10, 11, 12, 90, 91, 0],
            [20, 21, 80, 81, 82, 0],
        ]
    )
    prompt_lens = torch.tensor([3, 2])
    response_lens = torch.tensor([2, 3])
    hints = [torch.tensor([7, 7, 7, 7]), torch.tensor([8, 8])]

    hint_ids, hint_lengths, hint_prompt_lens = build_hint_inputs(
        input_ids,
        prompt_lens,
        response_lens,
        hints,
        pad_token_id=0,
        max_seq_len=32,
    )

    assert hint_prompt_lens.tolist() == [4, 2]
    assert hint_lengths.tolist() == [6, 5]
    # 样本 0：[7,7,7,7] + [90,91]
    assert hint_ids[0, :6].tolist() == [7, 7, 7, 7, 90, 91]
    # 样本 1：[8,8] + [80,81,82]，尾部 padding
    assert hint_ids[1, :5].tolist() == [8, 8, 80, 81, 82]
    assert hint_ids[1, 5].item() == 0


def test_build_hint_inputs_left_truncates_hint_not_response():
    """预算不够时只能砍参考解的开头；砍掉 response 会让师生 token 对不上，KL 失去意义。"""
    input_ids = torch.tensor([[1, 2, 50, 51, 52]])
    hints = [torch.arange(100, 120)]  # 20 个 token，远超预算

    hint_ids, hint_lengths, hint_prompt_lens = build_hint_inputs(
        input_ids,
        torch.tensor([2]),
        torch.tensor([3]),
        hints,
        pad_token_id=0,
        max_seq_len=8,
    )

    assert int(hint_prompt_lens[0]) == 5  # 8 - 3(response)
    assert int(hint_lengths[0]) == 8
    # 保留的是参考解的【尾部】
    assert hint_ids[0, :5].tolist() == [115, 116, 117, 118, 119]
    # response 一个 token 都不能少
    assert hint_ids[0, 5:8].tolist() == [50, 51, 52]


def test_build_hint_inputs_pads_to_divisible_length():
    input_ids = torch.tensor([[1, 2, 30, 31]])
    hint_ids, _, _ = build_hint_inputs(
        input_ids,
        torch.tensor([2]),
        torch.tensor([2]),
        [torch.tensor([9, 9, 9])],
        pad_token_id=0,
        max_seq_len=64,
        make_divisible_by=8,
    )
    assert hint_ids.shape[1] == 8


def test_build_hint_inputs_rejects_response_filling_budget():
    with pytest.raises(ValueError, match="max_total_sequence_length"):
        build_hint_inputs(
            torch.tensor([[1, 40, 41, 42]]),
            torch.tensor([1]),
            torch.tensor([3]),
            [torch.tensor([9, 9])],
            pad_token_id=0,
            max_seq_len=3,
        )


# ---------------------------------------------------------------- 重对齐
def _positional_teacher(batch: int, seq: int, topk: int) -> torch.Tensor:
    """构造一个「值 = 下标」的张量，便于断言搬运是否落在正确位置。"""
    base = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    return base.expand(batch, seq, topk).clone()


def test_realign_topk_places_response_at_student_positions():
    """out[Lp-1 + j] 必须等于 teacher[Lq-1 + j]，j 遍历整个 response。"""
    student_seq, teacher_seq, topk = 10, 14, 3
    logits = _positional_teacher(1, teacher_seq, topk)
    indices = logits.to(torch.long)

    student_prompt_lens = torch.tensor([4])
    hint_prompt_lens = torch.tensor([9])
    response_lens = torch.tensor([5])

    out_logits, out_indices = realign_topk(
        logits,
        indices,
        student_prompt_lens=student_prompt_lens,
        hint_prompt_lens=hint_prompt_lens,
        response_lens=response_lens,
        student_seq_len=student_seq,
    )

    assert out_logits.shape == (1, student_seq, topk)
    for j in range(5):
        assert out_logits[0, 4 - 1 + j, 0].item() == pytest.approx(9 - 1 + j)
        assert out_indices[0, 4 - 1 + j, 0].item() == 9 - 1 + j
    # response 之外全为 0（会被 token_mask 屏蔽）
    assert out_logits[0, :3].abs().sum().item() == 0
    assert out_logits[0, 8:].abs().sum().item() == 0


def test_realign_topk_is_identity_when_prompts_match():
    """退化情形：hint prompt 与学生 prompt 等长时，重对齐必须是恒等搬运。

    这条是防回归的锚——如果哪天索引推导被改错，等长情形通常仍能跑通，
    但会整体错位一格，这里能立刻抓到。
    """
    seq, topk = 12, 4
    logits = _positional_teacher(2, seq, topk)
    indices = logits.to(torch.long)
    prompt_lens = torch.tensor([5, 3])
    response_lens = torch.tensor([6, 8])

    out_logits, _ = realign_topk(
        logits,
        indices,
        student_prompt_lens=prompt_lens,
        hint_prompt_lens=prompt_lens,
        response_lens=response_lens,
        student_seq_len=seq,
    )

    pairs = zip(prompt_lens.tolist(), response_lens.tolist(), strict=True)
    for i, (p, r) in enumerate(pairs):
        lo, hi = p - 1, p - 1 + r
        assert torch.equal(out_logits[i, lo:hi], logits[i, lo:hi])


def test_realign_topk_rejects_out_of_range_response():
    logits = _positional_teacher(1, 6, 2)
    with pytest.raises(ValueError, match="越出老师序列"):
        realign_topk(
            logits,
            logits.to(torch.long),
            student_prompt_lens=torch.tensor([2]),
            hint_prompt_lens=torch.tensor([5]),
            response_lens=torch.tensor([4]),
            student_seq_len=10,
        )


def test_realign_topk_roundtrip_with_build_hint_inputs():
    """把 build_hint_inputs 与 realign_topk 串起来验一次端到端的位置一致性。

    用一个「假模型」：位置 t 的输出直接取输入序列里第 t+1 个 token 的 id（即完美预测下一个
    token）。若对齐正确，则重排后学生位置 Lp-1+j 处的值，应当恰好是学生自己 response 的
    第 j+1 个 token（最后一个位置除外，它预测的是序列外的下一个 token）。
    """
    input_ids = torch.tensor([[1, 2, 3, 71, 72, 73, 74]])
    prompt_lens, response_lens = torch.tensor([3]), torch.tensor([4])
    hints = [torch.tensor([5, 5, 5, 5, 5, 5])]

    hint_ids, _, hint_prompt_lens = build_hint_inputs(
        input_ids, prompt_lens, response_lens, hints, pad_token_id=0, max_seq_len=32
    )

    # 假模型：位置 t 的 top-1 = 输入序列第 t+1 个 token。
    nxt = torch.cat([hint_ids[:, 1:], torch.zeros_like(hint_ids[:, :1])], dim=1)
    teacher_logits = nxt.unsqueeze(-1).to(torch.float32)
    teacher_indices = nxt.unsqueeze(-1)

    out_logits, _ = realign_topk(
        teacher_logits,
        teacher_indices,
        student_prompt_lens=prompt_lens,
        hint_prompt_lens=hint_prompt_lens,
        response_lens=response_lens,
        student_seq_len=input_ids.shape[1],
    )

    # 学生 response 是 [71,72,73,74]，起点 Lp=3。位置 Lp-1+j 预测第 j 个 response token。
    predicted = [int(out_logits[0, 3 - 1 + j, 0]) for j in range(4)]
    assert predicted == [71, 72, 73, 74]


# ---------------------------------------------------------------- vocab_size（Qwen3.5 嵌套 config）
def test_config_vocab_size_top_level():
    assert config_vocab_size(SimpleNamespace(vocab_size=32000)) == 32000


def test_config_vocab_size_nested_text_config():
    """Qwen3.5：顶层无 vocab_size，在 text_config 里。"""
    cfg = SimpleNamespace(text_config=SimpleNamespace(vocab_size=248320))
    assert config_vocab_size(cfg) == 248320


def test_config_vocab_size_get_text_config():
    text = SimpleNamespace(vocab_size=248320)
    cfg = SimpleNamespace(get_text_config=lambda: text)
    assert config_vocab_size(cfg) == 248320


def test_config_vocab_size_missing_raises():
    with pytest.raises(AttributeError, match="没有 vocab_size"):
        config_vocab_size(SimpleNamespace(text_config=SimpleNamespace()))
