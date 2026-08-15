"""从 CLI 参数构建 JobSpec。

客户端构建完整 JobSpec，并在**本地**按 recipe 声明预校验 —— 超参拼错、
取值越界在敲回车那一刻就能看到，不必等上传完、排完队、跑起来才发现。

本地校验用的是与服务端**同一份** recipe 目录（都来自 nemo-lab-sdk），所以不会
出现「本地说没问题、服务端说不行」。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from nemo_lab_sdk import __version__ as SDK_VERSION
from nemo_lab_sdk.contract import (
    ArtifactRef,
    DataRef,
    DataSpec,
    FrameworkRef,
    JobSpec,
    JobSpecBody,
    LifecycleSpec,
    Metadata,
    ModelSpec,
    Provenance,
    RecipeRef,
    ResourcePool,
    ResourceSpec,
    SourceSpec,
    SpecError,
    exp_basename,
)
from nemo_lab_sdk.recipes import get_recipe


def parse_set(items: list[str]) -> dict[str, Any]:
    """把 `--set k=v` 解析成超参字典，并按字面量推断类型。

    类型推断是必要的：recipe schema 会拒绝 "500"（字符串）作为 max_num_steps，
    而命令行给过来的一切都是字符串。推断规则刻意保守 —— 只认 bool/int/float，
    其余保持字符串，避免把 "1e-5" 之外的东西误判。
    """
    out: dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise SpecError(f"--set 需要 key=value 形式，收到 {raw!r}")
        key, _, val = raw.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            raise SpecError(f"--set 的 key 不能为空: {raw!r}")
        out[key] = _literal(val)
    return out


def _literal(v: str) -> Any:
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_pool(raw: str) -> ResourcePool:
    """解析 `--pool name:series:nodes:gpus_per_node`。

    例：--pool train:h200:1:4 --pool rollout:h100:1:2
    这正是 grpo_noncolocated（1 卡生成 / 1 卡训练）第一次能被显式表达出来。
    """
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 4:
        raise SpecError(
            f"--pool 需要 name:series:nodes:gpus_per_node 四段，收到 {raw!r}"
            "（例：train:h200:1:4）"
        )
    name, series, nodes, per_node = parts
    for label, v in (("nodes", nodes), ("gpus_per_node", per_node)):
        if not v.isdigit() or int(v) <= 0:
            raise SpecError(f"--pool 的 {label} 应为正整数，收到 {v!r}")
    return ResourcePool(name=name, series=series, nodes=int(nodes), gpus_per_node=int(per_node))


def build_spec(
    exp_rel: str,
    *,
    recipe: str,
    framework_version: str = "",
    user: str = "",
    project: str = "",
    display_name: str = "",
    sets: Optional[list[str]] = None,
    pools: Optional[list[str]] = None,
    roles: Optional[list[str]] = None,
    init_from: str = "",
    on_success: Optional[list[str]] = None,
    image: str = "",
    observability_url: str = "",
    base_model: str = "",
    train_data: str = "",
    validation_data: str = "",
    train_dataset: str = "",
    validation_dataset: str = "",
    operation: str = "train",
    provenance: Optional[dict] = None,
    validate: bool = True,
) -> JobSpec:
    """构建并（默认）本地校验 JobSpec。

    校验失败抛 SpecError —— CLI 捕获后打印可读错误，不发起任何网络请求。
    """
    # get_recipe 的报错已列出可用值，并对无前缀裸名给出两段式提示，直接透传。
    r = get_recipe(recipe)
    selected_framework_version = framework_version.strip() or r.runtime.default_version
    selected_runtime = r.runtime.resolve(selected_framework_version)
    requested_image = image.strip()
    if r.framework == "custom":
        selected_image = requested_image
    else:
        if requested_image:
            raise SpecError(
                f"{r.framework}@{selected_framework_version} 的执行工件由 Console deployment "
                "runtime registry 按 runtime_id 精确解析；一等框架不允许 --image"
            )
        selected_image = ""

    observability = r.adapter_options.get("observability")
    if observability == "external" and not observability_url.strip():
        raise SpecError(
            f"方法 {r.id} 使用 external observability，必须显式提供 --observability-url"
        )
    if observability == "platform" and observability_url.strip():
        raise SpecError(f"方法 {r.id} 使用 platform observability，不允许 --observability-url")
    if r.framework == "custom" and not selected_image:
        raise SpecError("custom recipe 要求显式提供 --image")
    if r.framework in {"verl", "trl"} and operation == "train":
        missing = [
            flag
            for flag, value in (
                ("--model", base_model),
                ("--train-data", train_data),
                ("--validation-data", validation_data),
            )
            if not value.strip()
        ]
        if missing:
            raise SpecError(f"{r.framework} recipe 要求显式提供：{' '.join(missing)}")

    # 平台数据集引用（<owner>/<name>[@version]）：格式在本地就校验，
    # 存在性与可见性由服务端在提交准入时裁决。
    for flag, ref in (("--train-dataset", train_dataset), ("--validation-dataset", validation_dataset)):
        ds_id = ref.strip().partition("@")[0]
        if ref.strip() and ("/" not in ds_id or len([s for s in ds_id.split("/") if s]) != 2):
            raise SpecError(
                f"{flag} 必须是 <owner>/<name>[@version]（如 alice/gsm8k@v1），收到 {ref!r}；"
                "用 `lab dataset ls` 查看完整 ID"
            )

    hyperparams = parse_set(sets or [])
    if validate:
        # 与服务端同一份 recipe 声明：不会出现「本地说没问题、服务端说不行」。
        hyperparams = r.validate_hyperparams(hyperparams, strict=True)

    pool_objs = tuple(parse_pool(p) for p in (pools or []))
    if not pool_objs:
        raise SpecError("lab/v2 要求显式资源池；请至少传一个 --pool name:series:nodes:gpus_per_node")
    role_map: dict[str, str] = {}
    for raw in roles or []:
        if ":" not in raw:
            raise SpecError(f"--role 需要 role:pool 形式，收到 {raw!r}")
        role, _, pool = raw.partition(":")
        role_map[role.strip()] = pool.strip()

    if pool_objs and not role_map:
        # 只有一个池时可以合理推断：recipe 声明的角色全落在它上面。
        if len(pool_objs) == 1:
            role_map = {role: pool_objs[0].name for role in r.roles}
        else:
            raise SpecError(
                f"声明了 {len(pool_objs)} 个资源池但没有 --role 映射。"
                f"方法 {r.id} 的角色：{', '.join(r.roles)}"
            )
    if validate:
        known = {p.name for p in pool_objs}
        for role, pool in role_map.items():
            if role not in r.roles:
                raise SpecError(f"方法 {r.id} 没有角色 {role!r}；可用：{', '.join(r.roles)}")
            if pool not in known:
                raise SpecError(f"角色 {role!r} 指向未定义的资源池 {pool!r}")

    actions = tuple(a.strip() for a in (on_success or []) if a.strip())
    if validate:
        for a in actions:
            if not r.supports(a):
                raise SpecError(
                    f"方法 {r.id} 不支持训练后动作 {a!r}；支持：{', '.join(r.lifecycle) or '（无）'}"
                )

    prov = provenance or {}
    return JobSpec(
        metadata=Metadata(
            name=exp_basename(exp_rel),
            project=project or "",
            owner=user or "",          # 服务端会权威覆写
            display_name=display_name or "",
        ),
        spec=JobSpecBody(
            recipe=RecipeRef(name=r.id, version=r.version, digest=r.digest, plugins=r.plugins),
            source=SourceSpec(exp=exp_rel),
            framework=FrameworkRef(
                kind=r.framework,
                version=selected_framework_version,
                runtime_id=selected_runtime.runtime_id,
                image=selected_image,
                observability_url=observability_url.strip(),
            ),
            model=ModelSpec(
                base=base_model.strip(),
                init_from=ArtifactRef.parse(init_from) if init_from else None,
            ),
            # dataset 与 path 是两个正交维度：dataset 触发分发（拉到共享缓存、
            # 注入 <NAME>_DATA_DIR 环境变量），path 是训练框架实际读的位置
            # （nemo-rl 通常在 config 里用 ${oc.env:<NAME>_DATA_DIR} 引用，不需要 path）。
            data=DataSpec(
                train=(
                    DataRef(dataset=train_dataset.strip(), path=train_data.strip())
                    if (train_data.strip() or train_dataset.strip()) else None
                ),
                validation=(
                    DataRef(dataset=validation_dataset.strip(), path=validation_data.strip())
                    if (validation_data.strip() or validation_dataset.strip()) else None
                ),
            ),
            resources=ResourceSpec(pools=pool_objs, roles=role_map),
            hyperparams=hyperparams,
            lifecycle=LifecycleSpec(on_success=actions),
        ),
        provenance=Provenance(
            sdk_version=SDK_VERSION,
            git_commit=str(prov.get("git_commit") or ""),
            git_dirty=bool(prov.get("git_dirty")),
            config_sha=str(prov.get("config_sha") or ""),
        ),
    )


def infer_recipe(exp_path: Path) -> str:
    """从实验目录读取显式 recipe：同目录的 `method` 文件。

    它跟着实验走，fork 时自动继承。读不到返回空串，由调用方明确报错；
    不读取 framework 文件，也不按其他文件的存在性推断。
    """
    f = exp_path / "method"
    if f.is_file():
        return f.read_text(encoding="utf-8").strip()
    return ""
