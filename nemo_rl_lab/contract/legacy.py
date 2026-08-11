"""老客户端兼容：把「exp + client_meta」合成一份 recipe=legacy 的 JobSpec。

老客户端（`lab submit` 的现有版本）只上传实验目录与一份 X-Lab-Meta，不带 JobSpec。
服务端为它合成本模块产出的 spec，使新旧两条提交路径在下游**完全同构**——
调度、计量、审计、集群侧 launcher 都只认 JobSpec，不必到处写 `if legacy:`。

合成出的 spec 满足 `JobSpec.is_legacy is True`，`spec_to_env()` 据此不注入
LAB_RECIPE* 变量，因此老路径的环境变量产出逐字节不变。
"""
from __future__ import annotations

from typing import Any, Mapping

from .names import exp_basename
from .spec import (
    RECIPE_LEGACY,
    FrameworkRef,
    JobSpec,
    JobSpecBody,
    Metadata,
    ModelSpec,
    Provenance,
    RecipeRef,
    SourceSpec,
)


def legacy_spec(
    exp_rel: str,
    *,
    user: str = "",
    client_meta: Mapping[str, Any] | None = None,
    framework: str = "nemo-rl",
) -> JobSpec:
    """从老客户端的 exp + client_meta 合成 JobSpec。

    client_meta 中被消费的键（其余忽略，仍由调用方按需保存）：
        git_commit / git_dirty / config_sha / model_name / project / display_name
    """
    meta = dict(client_meta or {})
    return JobSpec(
        metadata=Metadata(
            name=exp_basename(exp_rel),
            project=str(meta.get("project") or "").strip(),
            owner=user,
            display_name=str(meta.get("display_name") or "").strip(),
        ),
        spec=JobSpecBody(
            recipe=RecipeRef(name=RECIPE_LEGACY),
            source=SourceSpec(exp=exp_rel),
            framework=FrameworkRef(kind=framework),
            model=ModelSpec(base=str(meta.get("model_name") or "").strip()),
        ),
        provenance=Provenance(
            git_commit=str(meta.get("git_commit") or "").strip(),
            git_dirty=bool(meta.get("git_dirty")),
            config_sha=str(meta.get("config_sha") or "").strip(),
            model_name=str(meta.get("model_name") or "").strip(),
        ),
    )
