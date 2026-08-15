"""教程示例：RLOO 优势估计器（自包含 algorithm 插件的完整写法）。

与官方 maxrl/opsd 插件包不同 —— 那两个只是 common/algorithms/ 的薄适配层，
本示例把算法实现**整个装进插件包**：不依赖上传包里的任何用户代码，发布后
任何人 `lab plugin install` 即可在自己的实验里使用。

RLOO（REINFORCE Leave-One-Out）：
    Â_i = r_i − mean(r_j, j≠i)
基线是同组其余样本的平均奖励（不含自身），无偏且不需要 value model。

启用方式（安装并锁定本插件后，在实验 config.yaml 里）：
    grpo:
      adv_estimator:
        name: rloo
"""
from __future__ import annotations

from typing import Any, Mapping

import torch


def rloo_advantages(
    prompt_ids: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """[b] 奖励 → [b, seq_len] token 级优势。组内只有 1 个样本时优势为 0。"""
    rewards = rewards.float()
    adv = torch.zeros_like(rewards)
    ids = prompt_ids if prompt_ids.dim() == 1 else None
    if ids is not None:
        groups = [prompt_ids == u for u in torch.unique(ids)]
    else:
        uniques = torch.unique(prompt_ids, dim=0)
        groups = [(prompt_ids == uniques[i]).all(dim=1) for i in range(len(uniques))]
    for m in groups:
        n = int(m.sum())
        if n < 2:
            continue
        total = rewards[m].sum()
        # leave-one-out 均值：(Σr − r_i) / (n−1)
        adv[m] = rewards[m] - (total - rewards[m]) / (n - 1)
    return adv.unsqueeze(-1).expand(mask.shape)


class RLOOAdvantageEstimator:
    """与 NeMo-RL 的 GRPOAdvantageEstimator 同接口。"""

    def __init__(self, estimator_config: Any, loss_config: Any):
        pass

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return rloo_advantages(prompt_ids, rewards, mask)


def install(_params: Mapping[str, Any] | None = None, **_ctx: Any) -> None:
    """插件入口（manifest 的 entrypoint 指到这里）。

    eager 插件由 launcher 在训练进程启动前调用，签名固定为
    install(params, **ctx)：params 是作业的 hyperparams 快照，ctx 对 eager
    恒为空。做法与 common/algorithms/maxrl.py 相同：给 NeMo-RL 的
    _create_advantage_estimator 包一层，新增 name=="rloo" 分支。幂等。
    """
    import nemo_rl.algorithms.grpo as grpo_mod

    if getattr(grpo_mod, "_rloo_installed", False):
        return
    original = grpo_mod._create_advantage_estimator

    def _create_with_rloo(master_config):
        grpo_cfg = getattr(master_config, "grpo", None) or master_config["grpo"]
        adv_cfg = (grpo_cfg.get("adv_estimator", {}) or {}) if hasattr(grpo_cfg, "get") else (
            getattr(grpo_cfg, "adv_estimator", {}) or {}
        )
        name = adv_cfg.get("name") if hasattr(adv_cfg, "get") else getattr(adv_cfg, "name", None)
        if name == "rloo":
            print("  ✓ Using RLOO advantage estimator (leave-one-out 基线，插件 rloo)", flush=True)
            return RLOOAdvantageEstimator(adv_cfg, None)
        return original(master_config)

    grpo_mod._create_advantage_estimator = _create_with_rloo
    grpo_mod._rloo_installed = True
