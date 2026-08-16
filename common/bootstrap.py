"""实验 run.py 的公共 bootstrap（在集群容器内运行，依赖 nemo_rl）。

每个实验的 run.py 只保留三样东西：数据集类（怎么把样本变成 DatumSpec）、
环境构造（用哪个 Env、reward shaping 怎么配）、算法插件（如 MaxRL/OPSD 的 install_*）。
配置加载、CLI overrides、Ray 初始化、tokenizer、setup/train 的样板全部收敛到这里。

nemo_rl 的 import 全部放在函数内：本模块可以在无 nemo_rl 的开发机上被 import
（供单测检查结构），真正执行只发生在集群容器里。

版本契约：本平台 recipe catalog 只发布 NeMo-RL 0.7.0，setup() 的返回值按 0.7 的
13 元组解包；长度不符说明镜像与 recipe 锁不一致，直接报错，不做多版本兼容。
"""
from __future__ import annotations

import argparse
import json
import os
import pprint
from typing import Any, Callable


def parse_args(description: str) -> tuple[argparse.Namespace, list[str]]:
    """标准实验入口参数：--config + 任意 hydra overrides。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args, overrides = parser.parse_known_args()
    return args, overrides


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line := line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_data_dir(data_cfg: dict[str, Any], env_var: str, hint: str, *,
                     env_first: bool = True) -> str:
    """数据目录：环境变量与 config.data.data_dir 二选一，都没有直接报错。

    env_first=False 时 config 优先（如 OPSD：数据在集群共享盘，config 写绝对路径，
    环境变量只作可选覆盖）。
    """
    from_env = os.environ.get(env_var)
    from_cfg = data_cfg.get("data_dir")
    data_dir = (from_env or from_cfg) if env_first else (from_cfg or from_env)
    if not data_dir:
        raise SystemExit(f"未指定数据目录。{hint}")
    return data_dir


def load_experiment_config(this_dir: str, master_config_cls, *, description: str,
                           pop_sections: tuple[str, ...] = ()):
    """加载实验配置：--config（默认 <实验目录>/config.yaml）+ hydra overrides →
    MasterConfig 实例（pydantic BaseModel：顶层属性访问，嵌套仍是 dict），并递增 log_dir。

    pop_sections：本仓库自定义、不属于 MasterConfig schema 的段（如 opsd），
    先取出再构造。返回 (config, popped)。
    """
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )
    from nemo_rl.utils.logger import get_next_experiment_dir
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    args, overrides = parse_args(description)
    config_path = args.config or os.path.join(this_dir, "config.yaml")

    config = load_config(config_path)
    print(f"已加载配置: {config_path}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    raw = OmegaConf.to_container(config, resolve=True)
    popped = {name: dict(raw.pop(name, {}) or {}) for name in pop_sections}
    config = master_config_cls(**raw)
    print("最终配置：")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 日志目录: {config.logger['log_dir']}")
    return (config, popped) if pop_sections else (config, {})


def init_runtime(config, algo_section: str):
    """init_ray + set_seed + tokenizer + generation config；返回 tokenizer。

    algo_section：seed 所在的顶层段名（grpo / distillation）。
    """
    from nemo_rl.algorithms.utils import get_tokenizer, set_seed
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.models.generation import configure_generation_config

    init_ray()
    set_seed(getattr(config, algo_section)["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )
    return tokenizer


def run_grpo(config, tokenizer, train_dataset, val_dataset, task_to_env: dict,
             val_task_to_env: dict | None = None,
             before_train: Callable[[], None] | None = None) -> None:
    """GRPO 标准流程：setup（按 NeMo-RL 0.7 的 13 元组契约解包）→ before_train 钩子 → grpo_train。

    before_train：算法插件安装点（如 install_maxrl_estimator），在 setup 之后、
    grpo_train 之前执行——优势估计器等是在 grpo_train 内部创建的。
    """
    from nemo_rl.algorithms.grpo import grpo_train, setup

    values = setup(config, tokenizer, train_dataset, val_dataset)
    # NeMo-RL 0.7：13 元组（末尾两个 MOPD teacher 字段在同步 GRPO 中未使用，但必须解包）。
    if len(values) != 13:
        raise RuntimeError(
            f"NeMo-RL setup() 返回 {len(values)} 个值，与 0.7.0 的 13 元组契约不符；"
            "训练镜像与 recipe 锁定版本不一致，检查 recipe.lock.json 与运行时镜像"
        )
    (
        policy,
        policy_generation,
        _nemo_gym,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        _teacher_worker_groups,
        _alias_to_group_alias,
    ) = values

    if before_train is not None:
        before_train()

    if val_task_to_env is None:
        # 显著警告而不是静默回退：验证若复用带 reward shaping 的训练环境，
        # validation/accuracy = mean(total_reward) 会把检索奖励/加成算进去，
        # 验证分虚高且无法与无工具 baseline 同尺度对比（带 shaping 的环境应
        # 用 make_eval_cfg 派生全零 shaping 的验证实例，见 qa_docs_agent_env）。
        print(
            "[bootstrap] ⚠ 未提供 val_task_to_env，验证将复用训练环境。"
            "若训练环境带 reward shaping（检索奖励/加成/罚分），validation/accuracy 将失真；"
            "请为验证传入 shaping 全零的环境实例。",
            flush=True,
        )

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env if val_task_to_env is not None else task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )
