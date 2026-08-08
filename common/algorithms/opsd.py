"""OPSD —— On-Policy Self-Distillation（论文《Self-Distilled Reasoner》arXiv:2601.18734）。

核心论点：不需要一个更强的外部教师。**同一个模型**既当学生又当老师，差别只在「看到什么」：

    学生 π(· | x)          只看题目 x，on-policy 采样出一条解答 y ~ π(·|x)
    老师 π(· | x, y*)      额外看到参考解 y*，对**同一条 y** 做 teacher-forcing 前向

老师因为「偷看了答案」，在同样的 token 位置上分布更尖锐、更正确；用它来监督学生，就得到了一个
不需要外部大模型、且始终 on-policy（监督信号落在学生自己采样的轨迹上，无分布漂移）的蒸馏信号：

    L = Σ_t  KL( π(·|x,y*)_t  ‖  π(·|x)_t )     （t 只取 response token）

与 GRPO 这类 RL 的区别：RL 只有一个标量奖励，OPSD 拿到的是**每个 token 上的完整分布**，
信号密度高出几个数量级；与普通 SFT/蒸馏的区别：y 是学生自己采样的（on-policy），
不是数据集里的 y*，因此不存在「训练时看的是自己写不出来的句子」这个经典 exposure bias。

──────────────────────────────────────────────────────────────────────────────
落到 NeMo-RL 0.7.0 上的实现路径（本文件做的全部事情）
──────────────────────────────────────────────────────────────────────────────
NeMo-RL 自带 on-policy distillation 主循环（`nemo_rl/algorithms/distillation.py`），
它已经把「学生采样 → 老师算 top-k logits → KL 训练」这条链路跑通了。原版第 870 行附近是：

    teacher_topk = teacher_policy.get_topk_logits(train_data, k=...)   # 师生吃同一份 train_data

OPSD 相对它**只差一件事：老师吃的输入不一样**（多了参考解）。所以本文件不 fork 主循环，
只做三处外科手术（与 common/algorithms/maxrl.py 的 install_* 范式一致）：

  1. `OPSDTeacher`      —— 包在老师 policy 外面。收到学生的 train_data 后，就地重建一份
                           「参考解条件化」的输入喂给模型，再把 top-k 结果**按 response token
                           重新对齐**回学生的位置坐标系（师生 prompt 长度不同，这是全篇唯一
                           容易写错的地方，见 `realign_topk`）。
  2. `install_opsd()`   —— ① 劫持 rollout 函数，把当前 step 的 batch 暂存下来（老师需要从
                           `extra_env_info` 里取参考解 token）；② 劫持 `Policy` 构造，
                           teacher_mode="self" 时**不再创建第二份模型**，老师直接复用学生权重
                           （省一整份显存，也才是论文说的「单模型自蒸馏」）；③ 修补
                           `check_vocab_equality`，兼容 Qwen3.5 把 vocab_size 放在 text_config。
  3. `ClippedDistillationLossFn` —— 论文的 per-token KL clipping：单个 token 的 KL 超过阈值就
                           截断，避免少数「老师极度自信、学生完全没概率」的 token 主导梯度。

代价：**零新增依赖**。这对只能访问内网的 Ray 集群是决定性的——不用改镜像、不用装包。

──────────────────────────────────────────────────────────────────────────────
数据侧的约定（由实验的 run.py 负责满足）
──────────────────────────────────────────────────────────────────────────────
每条样本的 `extra_env_info` 里必须带一个 key（名字见 `HINT_KEY`）：

    extra_env_info["opsd_hint_token_ids"] = <参考解条件化 prompt 的 token id 序列>

即把 `[题目 x + 参考解 y*]` 套上 chat template、加好 generation prompt 之后的 token 序列。
它替换学生 prompt 那一段；response 段（学生自己采样出来的 y）两边完全一致。
参见 experiments/opsd_qwen3.5-9b_math_h200_1n2g/run.py 的 `OPSDMathDataset`。
"""

from __future__ import annotations

from typing import Any, Optional

import torch

# extra_env_info 里携带「参考解条件化 prompt」的 key。
HINT_KEY = "opsd_hint_token_ids"

# 当前 step 的 rollout batch（由 install_opsd 劫持 rollout 函数写入，OPSDTeacher 读取后清空）。
# 之所以要这个旁路：NeMo-RL 主循环传给老师的 train_data 只有 input_ids/mask，
# 不含参考解；而 batch 里的 extra_env_info 有。两者顺序严格一致（train_data 就是按
# repeated_batch["message_log"] 顺序摊平出来的），OPSDTeacher 里有断言兜底。
_PENDING_BATCH: Any = None

# rollout 完成时的旁听者。老师取参考解、验证指标按题聚合，都需要拿到「这一轮 rollout 的
# 完整 batch」；与其各自去猴子补丁同一个函数（互相覆盖、顺序敏感），不如只补一次、多方旁听。
_ROLLOUT_LISTENERS: list = []


def add_rollout_listener(fn) -> None:
    """注册 rollout 旁听者：每次 rollout 结束时以完成后的 batch 调用一次。

    旁听者内部异常一律吞掉——它们都是旁路（指标/采集），不能把训练带崩。
    """
    if fn not in _ROLLOUT_LISTENERS:
        _ROLLOUT_LISTENERS.append(fn)


# =============================================================================
#  1. 序列切分：从 train_data 还原出「prompt 多长 / response 在哪」
# =============================================================================
def split_prompt_response(
    token_mask: torch.Tensor, input_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 token_mask 推出每条样本的 prompt 长度与 response 长度。

    NeMo-RL 的 `token_mask`（由 message 的 token_loss_mask 摊平而来）在 assistant token 上为 1、
    其余为 0。OPSD 是单轮任务（题目 → 一段解答），因此 1 必须是**连续一段**；多轮轨迹
    （assistant 被工具结果打断成多段）无法定义「把 prompt 换成参考解」这个操作，直接报错，
    而不是悄悄算出一个错的对齐。

    Args:
        token_mask:    [B, S]，response token 处为 1。
        input_lengths: [B]，padding 前的真实长度。

    Returns:
        (prompt_lens, response_lens)，都是 [B] 的 int64 张量。
    """
    mask = (token_mask > 0).to(torch.int64)
    lengths = input_lengths.to(torch.int64)
    batch = mask.shape[0]

    response_lens = mask.sum(dim=1)
    if (response_lens == 0).any():
        bad = int((response_lens == 0).nonzero()[0])
        raise ValueError(
            f"OPSD: 第 {bad} 条样本没有任何 response token（token_mask 全 0）。"
            "通常意味着生成被立即截断，请检查 policy.generation.max_new_tokens。"
        )

    # 首个 1 的位置 = prompt 长度（argmax 对 0/1 张量取到第一个最大值）。
    prompt_lens = mask.argmax(dim=1)

    # 连续性校验：[prompt_len, prompt_len + response_len) 区间内必须全是 1。
    idx = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand(batch, -1)
    lo = prompt_lens.unsqueeze(1)
    hi = (prompt_lens + response_lens).unsqueeze(1)
    expected = ((idx >= lo) & (idx < hi)).to(torch.int64)
    if not torch.equal(expected, mask):
        raise ValueError(
            "OPSD: token_mask 中的 response 段不连续（疑似多轮轨迹）。"
            "OPSD 的老师需要把 prompt 整段换成『题目+参考解』，多轮无法定义该操作；"
            "请把 distillation.max_rollout_turns 设为 1。"
        )
    if (prompt_lens + response_lens > lengths).any():
        raise ValueError("OPSD: prompt_len + response_len 超过 input_lengths，数据摊平异常。")

    return prompt_lens, response_lens


# =============================================================================
#  2. 构造老师输入：[参考解条件化 prompt] + [学生采样出的同一条 response]
# =============================================================================
def build_hint_inputs(
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    hint_prompts: list[torch.Tensor],
    *,
    pad_token_id: int,
    max_seq_len: int,
    make_divisible_by: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把学生的 [prompt|response] 换头成老师的 [hint_prompt|response]。

    response 段逐 token 原样搬运——这是 OPSD 的关键：师生必须在**同一条轨迹**上比较分布，
    老师只是多看到了参考解，不是去生成它自己的解答。

    hint prompt 过长时（题目+参考解本来就比题目长）做**左截断**：砍掉参考解前面的内容而不是
    砍 response。response 一旦被砍，师生 token 就对不上了，KL 完全失去意义。

    Returns:
        (hint_ids [B, S_h], hint_lengths [B], hint_prompt_lens [B])
    """
    batch = input_ids.shape[0]
    if len(hint_prompts) != batch:
        raise ValueError(f"OPSD: hint prompt 数量 {len(hint_prompts)} != batch {batch}")

    device = input_ids.device
    hint_prompt_lens = torch.empty(batch, dtype=torch.int64)
    rows: list[torch.Tensor] = []

    for i in range(batch):
        r_len = int(response_lens[i])
        p_len = int(prompt_lens[i])
        response = input_ids[i, p_len : p_len + r_len]

        hint = hint_prompts[i].to(device=device, dtype=input_ids.dtype).flatten()
        # 左截断：给 response 留够位置，砍掉 hint prompt 的开头。
        budget = max_seq_len - r_len
        if budget <= 0:
            raise ValueError(
                f"OPSD: 第 {i} 条 response 长度 {r_len} 已占满 max_total_sequence_length "
                f"{max_seq_len}，老师无处安放题目与参考解。请调大 policy.max_total_sequence_length。"
            )
        if hint.numel() > budget:
            hint = hint[hint.numel() - budget :]

        hint_prompt_lens[i] = hint.numel()
        rows.append(torch.cat([hint, response]))

    hint_lengths = torch.tensor([r.numel() for r in rows], dtype=torch.int32)
    seq_len = int(hint_lengths.max())
    if make_divisible_by > 1:
        seq_len = ((seq_len + make_divisible_by - 1) // make_divisible_by) * make_divisible_by

    hint_ids = torch.full(
        (batch, seq_len), pad_token_id, dtype=input_ids.dtype, device=device
    )
    for i, row in enumerate(rows):
        hint_ids[i, : row.numel()] = row

    return hint_ids, hint_lengths, hint_prompt_lens


# =============================================================================
#  3. 重对齐：把老师的 top-k 结果搬回学生的位置坐标系
# =============================================================================
def realign_topk(
    teacher_logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    student_prompt_lens: torch.Tensor,
    hint_prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    student_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 [B, S_h, k] 的老师输出重排成 [B, S, k]，与学生位置一一对应。

    ★ 全文件最容易写错的地方，把索引推导写清楚：

    `get_topk_logits` 返回**全长** [B, S, k]（见 dtensor_policy_worker_v2.get_topk_logits 的
    docstring），「预测下一个 token」的移位由下游 `model_utils.py` 的
    `teacher_topk_logprobs[:, :-1, :]` 统一处理。所以在本函数里沿用原始坐标系：
    下标 t 处的分布，预测的是**第 t+1 个 token**。设某条样本：

        学生序列  = [ p (长 Lp) | r (长 Lr) ]      第 j 个 response token 的绝对位置 = Lp + j
        老师序列  = [ q (长 Lq) | r (长 Lr) ]      同一个 token 的绝对位置          = Lq + j

    因此预测该 token 的下标分别是 `Lp + j - 1` 与 `Lq + j - 1`（j 从 0 开始；j=0 时下标是
    「最后一个 prompt token」，正是它预测出第一个 response token，合法）。整段搬运即：

        out[i, Lp-1 : Lp-1+Lr] = teacher[i, Lq-1 : Lq-1+Lr]

    response 以外的位置填 0——损失那边会被 `token_mask[:, 1:]` 屏蔽掉，填什么都不影响，
    但填 0 能让「万一 mask 没盖住」立刻暴露成异常大的 KL，而不是悄悄污染梯度。
    """
    batch, _, topk = teacher_logits.shape
    out_len = student_seq_len

    logits = torch.zeros(
        (batch, out_len, topk), dtype=teacher_logits.dtype, device=teacher_logits.device
    )
    indices = torch.zeros(
        (batch, out_len, topk), dtype=teacher_indices.dtype, device=teacher_indices.device
    )

    for i in range(batch):
        r_len = int(response_lens[i])
        s_start = int(student_prompt_lens[i]) - 1
        t_start = int(hint_prompt_lens[i]) - 1
        if s_start < 0 or t_start < 0:
            raise ValueError(f"OPSD: 第 {i} 条样本 prompt 为空，无法定位 response 起点。")
        if s_start + r_len > out_len:
            raise ValueError(
                f"OPSD: 第 {i} 条 response 越出学生序列（{s_start + r_len} > {out_len}）。"
            )
        if t_start + r_len > teacher_logits.shape[1]:
            raise ValueError(
                f"OPSD: 第 {i} 条 response 越出老师序列"
                f"（{t_start + r_len} > {teacher_logits.shape[1]}）。"
            )
        logits[i, s_start : s_start + r_len] = teacher_logits[i, t_start : t_start + r_len]
        indices[i, s_start : s_start + r_len] = teacher_indices[i, t_start : t_start + r_len]

    return logits, indices


# =============================================================================
#  4. 老师包装器
# =============================================================================
class OPSDTeacher:
    """包在老师 policy 外面：接学生的 train_data，返回「看过参考解」的 top-k 分布。

    对 NeMo-RL 主循环而言它就是个普通 teacher policy（只用到 `prepare_for_lp_inference`
    / `get_topk_logits` / `offload_after_refit` 三个方法），因此不需要改主循环一行代码。

    teacher_mode:
        "self"  —— 老师就是学生本人（论文主设定）。`inner` 在学生 policy 创建后由
                   `install_opsd` 回填，不额外占显存。
        "fixed" —— 老师是一份独立加载的固定权重（论文附录的对照设定，通常是同一个 base
                   模型的冻结快照）。显存翻倍，但老师不随训练漂移。
    """

    def __init__(
        self,
        inner: Any = None,
        *,
        pad_token_id: int,
        max_seq_len: int,
        make_divisible_by: int = 1,
        teacher_mode: str = "self",
    ):
        self.inner = inner
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.make_divisible_by = make_divisible_by
        self.teacher_mode = teacher_mode
        self.last_metrics: dict[str, float] = {}

    def bind(self, inner: Any) -> None:
        """self 模式下把学生 policy 回填进来当老师。"""
        self.inner = inner

    # ---- 主循环会调用的生命周期方法 ----------------------------------------
    def prepare_for_lp_inference(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner().prepare_for_lp_inference(*args, **kwargs)

    def offload_after_refit(self, *args: Any, **kwargs: Any) -> None:
        # self 模式：老师和学生是同一份权重，主循环紧接着就会 `student.prepare_for_training()`
        # 把它搬回 GPU。这里真的 offload 只会白白多一次 CPU↔GPU 往返，故 no-op。
        if self.teacher_mode == "self":
            return None
        return self._inner().offload_after_refit(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # 其余方法（save_checkpoint / shutdown 等）透传给被包装对象。
        # 注意用 __dict__.get 而不是 self.inner：后者在 inner 尚未赋值时会无限递归回本方法。
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(
                f"OPSD: 老师尚未绑定内层 policy，无法访问 {name!r}。"
                "请确认 install_opsd() 在 distillation.setup() 之前调用。"
            )
        return getattr(inner, name)

    # ---- 核心 -------------------------------------------------------------
    def get_topk_logits(self, data: Any, k: int, timer: Any = None, **kwargs: Any) -> Any:
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        hint_prompts = take_pending_hints(data["input_ids"].shape[0])

        prompt_lens, response_lens = split_prompt_response(
            data["token_mask"], data["input_lengths"]
        )
        hint_ids, hint_lengths, hint_prompt_lens = build_hint_inputs(
            data["input_ids"],
            prompt_lens,
            response_lens,
            hint_prompts,
            pad_token_id=self.pad_token_id,
            max_seq_len=self.max_seq_len,
            make_divisible_by=self.make_divisible_by,
        )

        hint_data: Any = BatchedDataDict(
            {
                "input_ids": hint_ids,
                "input_lengths": hint_lengths,
                # 老师这边只做前向，mask 不参与计算；给全 1 以免下游按 mask 做剪裁。
                "token_mask": torch.ones_like(hint_ids),
                "sample_mask": data["sample_mask"],
            }
        )

        out = self._inner().get_topk_logits(hint_data, k=k, timer=timer, **kwargs)

        logits, indices = realign_topk(
            out["topk_logits"],
            out["topk_indices"],
            student_prompt_lens=prompt_lens,
            hint_prompt_lens=hint_prompt_lens,
            response_lens=response_lens,
            student_seq_len=data["input_ids"].shape[1],
        )
        self.last_metrics = {
            "opsd_hint_len_mean": float(hint_prompt_lens.float().mean()),
            "opsd_response_len_mean": float(response_lens.float().mean()),
            "opsd_teacher_seq_len": float(hint_ids.shape[1]),
        }
        return BatchedDataDict({"topk_logits": logits, "topk_indices": indices})

    def _inner(self) -> Any:
        if self.inner is None:
            raise RuntimeError(
                "OPSD: teacher_mode='self' 但学生 policy 还没绑定到老师上。"
                "请确认 install_opsd() 在 distillation.setup() 之前调用。"
            )
        return self.inner


# =============================================================================
#  5. per-token KL clipping 损失
# =============================================================================
def make_clipped_loss_fn(loss_config: Any, clip: Optional[float]) -> Any:
    """在 NeMo-RL 的 DistillationLossFn 上加 per-token KL 截断。

    论文的稳定化手段：极少数 token 上老师极其自信、学生几乎零概率，KL 会爆到几十上百，
    单个 token 就能主导整个 batch 的梯度。按 token 截断到 `clip` 之后，这些位置仍然提供
    「往这个方向走」的信号，但不再垄断梯度。clip=None 则退化成原版损失。
    """
    from nemo_rl.algorithms.loss.loss_functions import DistillationLossFn

    if clip is None:
        return DistillationLossFn(loss_config)

    if clip <= 0:
        raise ValueError(f"OPSD: per_token_kl_clip 必须为正数或 null，收到 {clip}")

    class ClippedDistillationLossFn(DistillationLossFn):
        """DistillationLossFn + 逐 token KL 截断。"""

        def __init__(self, cfg: Any, clip_value: float):
            super().__init__(cfg)
            self.clip_value = clip_value

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            with _clip_per_token_kl(self.clip_value) as stats:
                loss, metrics = super().__call__(*args, **kwargs)
            metrics.update(stats)
            return loss, metrics

    return ClippedDistillationLossFn(loss_config, clip)


class _clip_per_token_kl:
    """上下文管理器：在 DistillationLossFn 内部把 `masked_mean` 的输入先做截断。

    为什么用这种方式而不是复制一遍父类 __call__：父类的 top-k 求和、zero_outside_topk 修正项、
    mixed KL 权重这些逻辑会随 NeMo-RL 升级而变，复制过来就等于分叉。这里只拦最后一步归约，
    截断作用在「每个 token 的 KL 标量」上——正是论文定义 clipping 的那个量。
    """

    def __init__(self, clip: float):
        self.clip = clip
        self.stats: dict[str, float] = {}

    def __enter__(self) -> dict[str, float]:
        from nemo_rl.algorithms.loss import loss_functions

        self._orig = loss_functions.masked_mean

        def patched(values: torch.Tensor, mask: torch.Tensor, *a: Any, **kw: Any):
            clipped = values.clamp(min=-self.clip, max=self.clip)
            denom = mask.sum()
            if denom > 0:
                over = ((values.abs() > self.clip) * mask).sum() / denom
                self.stats["opsd_kl_clip_frac"] = float(over)
                self.stats["opsd_kl_pre_clip_max"] = float((values * mask).abs().max())
            return self._orig(clipped, mask, *a, **kw)

        loss_functions.masked_mean = patched
        return self.stats

    def __exit__(self, *exc: Any) -> None:
        from nemo_rl.algorithms.loss import loss_functions

        loss_functions.masked_mean = self._orig


# =============================================================================
#  6. 安装：劫持 rollout 暂存 batch + 劫持 Policy 构造实现自蒸馏
# =============================================================================
def take_pending_hints(expected_batch: int) -> list[torch.Tensor]:
    """取出当前 step 的参考解 token（并清空暂存），顺序与 train_data 严格一致。"""
    global _PENDING_BATCH
    batch = _PENDING_BATCH
    _PENDING_BATCH = None
    if batch is None:
        raise RuntimeError(
            "OPSD: 没有暂存到本 step 的 rollout batch。"
            "请确认 install_opsd() 在 distillation_train() 之前调用，且未绕过 rollout 函数。"
        )

    infos = batch["extra_env_info"]
    if len(infos) != expected_batch:
        raise RuntimeError(
            f"OPSD: rollout batch 大小 {len(infos)} 与训练 batch {expected_batch} 不一致，"
            "无法保证参考解与样本一一对应。"
        )

    hints: list[torch.Tensor] = []
    for i, info in enumerate(infos):
        raw = (info or {}).get(HINT_KEY)
        if raw is None:
            raise RuntimeError(
                f"OPSD: 第 {i} 条样本的 extra_env_info 缺少 '{HINT_KEY}'。"
                "数据集必须提供『题目+参考解』条件化 prompt 的 token ids，见本文件模块级文档。"
            )
        hints.append(raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw))
    return hints


def config_vocab_size(config: Any) -> int:
    """读 HF AutoConfig 的 vocab_size。

    Qwen3.5 等复合配置把词表大小放在 `text_config`（或 `get_text_config()`）里，
    顶层没有 `vocab_size`；NeMo-RL 的 `check_vocab_equality` 直接读顶层会
    `AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'`。
    """
    text = None
    get_text = getattr(config, "get_text_config", None)
    if callable(get_text):
        try:
            text = get_text()
        except Exception:  # noqa: BLE001
            text = None
    if text is None:
        text = getattr(config, "text_config", None)
    for obj in (text, config):
        if obj is None:
            continue
        vs = getattr(obj, "vocab_size", None)
        if vs is not None:
            return int(vs)
    raise AttributeError(
        f"{type(config).__name__} 没有 vocab_size（text_config 里也没有）"
    )


def install_opsd(
    *,
    teacher_mode: str = "self",
    pad_token_id: int,
    max_seq_len: int,
    make_divisible_by: int = 1,
) -> None:
    """把 OPSD 装进 NeMo-RL 的 distillation 主循环。必须在 `distillation.setup()` 之前调用。

    做三件事：
      ① 劫持 rollout 函数 —— 每个 step 的 batch 暂存下来，老师从里面取参考解。
      ② 劫持 `Policy` 构造 —— 主循环写死了「先建老师、再建学生」，这里在建老师时返回
         一个 `OPSDTeacher` 壳；teacher_mode="self" 时壳内为空，等学生建好后回填
         （于是全程只有一份模型权重）；"fixed" 时壳内包着真正加载出来的老师。
      ③ 修补 `check_vocab_equality` —— 兼容 Qwen3.5 等把 vocab_size 放在 text_config 的模型。
    """
    from nemo_rl.algorithms import distillation

    if teacher_mode not in ("self", "fixed"):
        raise ValueError(f"OPSD: teacher_mode 只能是 'self' / 'fixed'，收到 {teacher_mode!r}")

    if getattr(distillation, "_opsd_installed", False):
        return

    # ---- ① rollout 暂存 ----------------------------------------------------
    def _stash(fn):
        def wrapper(*args: Any, **kwargs: Any):
            global _PENDING_BATCH
            result = fn(*args, **kwargs)
            batch = result[0] if isinstance(result, tuple) else result.final_batch
            _PENDING_BATCH = batch
            for listener in _ROLLOUT_LISTENERS:
                try:
                    listener(batch)
                except Exception as e:  # noqa: BLE001  旁路失败不能影响训练
                    print(f"[OPSD] rollout 旁听者异常（已忽略）: {e}", flush=True)
            return result

        return wrapper

    for name in ("run_multi_turn_rollout", "run_async_multi_turn_rollout"):
        if hasattr(distillation, name):
            setattr(distillation, name, _stash(getattr(distillation, name)))

    # ---- ② Policy 构造劫持 -------------------------------------------------
    real_policy_cls = distillation.Policy
    pending: dict[str, OPSDTeacher] = {}

    def policy_factory(*args: Any, **kwargs: Any):
        # NeMo-RL 的 setup() 用 name_prefix 区分两个 policy，且【先建老师、后建学生】
        # （distillation.py 的 Policy(name_prefix="teacher") 在 Policy(name_prefix="student") 之前）。
        # 这里严格按名字分派，而不是「第一次调用/第二次调用」——顺序一旦在上游变了，
        # 按次序猜会静默绑错对象，按名字则会走到下面的兜底分支、老老实实报错。
        prefix = kwargs.get("name_prefix")
        if prefix == "teacher":
            teacher = OPSDTeacher(
                inner=None if teacher_mode == "self" else real_policy_cls(*args, **kwargs),
                pad_token_id=pad_token_id,
                max_seq_len=max_seq_len,
                make_divisible_by=make_divisible_by,
                teacher_mode=teacher_mode,
            )
            if teacher_mode == "self":
                pending["teacher"] = teacher
                print("[OPSD] teacher_mode=self：跳过第二份模型加载，老师复用学生权重。", flush=True)
            return teacher

        policy = real_policy_cls(*args, **kwargs)
        if prefix == "student" and (teacher := pending.pop("teacher", None)) is not None:
            teacher.bind(policy)
            print("[OPSD] 学生 policy 已绑定为老师（同一份权重）。", flush=True)
        return policy

    distillation.Policy = policy_factory

    # ---- ③ Qwen3.5 嵌套 vocab_size -----------------------------------------
    # NeMo-RL 原版 `student_config.vocab_size`；复合配置顶层没有该字段会直接炸。
    from transformers import AutoConfig, AutoTokenizer

    def check_vocab_equality(tokenizer, student_model_name, teacher_model_name):
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        skip_hint = "Set NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true to skip this check."
        assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab(), (
            f"Token->ID mapping differs between student and teacher. {skip_hint}"
        )
        assert len(tokenizer) == len(teacher_tokenizer), (
            f"Effective vocab sizes differ between student and teacher. {skip_hint}"
        )
        student_vs = config_vocab_size(AutoConfig.from_pretrained(student_model_name))
        teacher_vs = config_vocab_size(AutoConfig.from_pretrained(teacher_model_name))
        assert student_vs == teacher_vs, (
            f"Model config vocab sizes differ between student and teacher. {skip_hint}"
        )

    distillation.check_vocab_equality = check_vocab_equality
    distillation._opsd_installed = True
    print(f"[OPSD] 已安装（teacher_mode={teacher_mode}, max_seq_len={max_seq_len}）", flush=True)
