"""nemo_rl_lab.contract —— 平台与集群之间的作业契约。

这是 nemo-rl-console（控制面）与 nemo-rl-lab（客户端 + 集群侧执行）之间**唯一**的
共享契约。两个仓库都从这里 import，因此契约无法再悄悄分叉。

依赖方向：console → lab（单向，与既有的 config_resolve 复用一致），不成环。
本包只用标准库，可在 NeMo-RL 官方容器内直接 import。

    from nemo_rl_lab.contract import JobSpec, PlatformBinding, spec_to_env

    env = spec_to_env(spec, binding)
"""
from __future__ import annotations

from .binding import IngestBinding, PlatformBinding
from .env import KNOWN_KEYS, spec_to_env
from .errors import SpecError
from .legacy import legacy_spec
from .names import exp_basename, safe_exp_rel, safe_project, safe_run_id, safe_segment
from .spec import (
    API_VERSION,
    FRAMEWORKS,
    KIND_TRAINING,
    KNOWN_LIFECYCLE_ACTIONS,
    RECIPE_LEGACY,
    ArtifactRef,
    DataRef,
    DataSpec,
    FrameworkRef,
    JobSpec,
    JobSpecBody,
    LifecycleSpec,
    Metadata,
    ModelSpec,
    PeftSpec,
    Provenance,
    RecipeRef,
    ResourcePool,
    ResourceSpec,
    SourceSpec,
)

__all__ = [
    "API_VERSION",
    "ArtifactRef",
    "DataRef",
    "DataSpec",
    "FRAMEWORKS",
    "FrameworkRef",
    "IngestBinding",
    "JobSpec",
    "JobSpecBody",
    "KIND_TRAINING",
    "KNOWN_KEYS",
    "KNOWN_LIFECYCLE_ACTIONS",
    "LifecycleSpec",
    "Metadata",
    "ModelSpec",
    "PeftSpec",
    "PlatformBinding",
    "Provenance",
    "RECIPE_LEGACY",
    "RecipeRef",
    "ResourcePool",
    "ResourceSpec",
    "SourceSpec",
    "SpecError",
    "exp_basename",
    "legacy_spec",
    "safe_exp_rel",
    "safe_project",
    "safe_run_id",
    "safe_segment",
    "spec_to_env",
]
