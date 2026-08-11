"""JobSpec —— 平台与集群之间的作业规格，两侧唯一的事实来源。

设计约束（重要）
──────────────────────────────────────────────────────────────────────────────
**只用标准库。** 本模块会被集群侧 launcher 在 NeMo-RL 官方容器里 import，而该容器
无法安装新依赖（见 scripts/_run_experiment.sh 顶部说明）。因此这里用 dataclass +
手写校验，不引 pydantic。服务端如需 FastAPI 模型，在自己的 API 边界上包一层即可。

它取代了什么
──────────────────────────────────────────────────────────────────────────────
改造前：客户端上传整个仓库 → 服务端拼 `bash <exp>/run.sh` → 集群侧 shell 靠约 32 个
约定俗成的环境变量 + 正则解析 conf 文本决定一切。这份契约不存在于任何单一位置，
一半写在 Python 里、一半写在 Bash 注释里。

改造后：客户端提交一份 JobSpec；`env.spec_to_env()` 是它到环境变量的**唯一**转换；
golden 测试锁定该转换的逐键输出，两个仓库共用同一组测试。

版本化
──────────────────────────────────────────────────────────────────────────────
`apiVersion` 允许将来演进而不破坏老客户端：服务端可同时接受 lab/v1 与后续版本，
并把老版本升级成新版本的内存表示。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import SpecError
from .names import safe_exp_rel, safe_project, safe_segment

#: 当前契约版本。集群侧 launcher 用它拒绝跑不认识的 spec。
API_VERSION = "lab/v1"

#: 已知的 kind。目前只有训练作业；export/eval 不是独立 kind，而是训练作业的 lifecycle。
KIND_TRAINING = "TrainingJob"

#: 已知框架。custom 表示用户自带 train.sh，平台只负责准备环境与记账。
FRAMEWORKS = ("nemo-rl", "custom")

#: legacy 兼容用的 recipe 名。服务端为「只上传了 run.sh、没带 spec」的老客户端
#: 合成一份 recipe=legacy 的 JobSpec，使新旧两条路径在下游完全同构。
RECIPE_LEGACY = "legacy"


def _require_str(data: Mapping[str, Any], key: str, *, field_path: str, default: str | None = None) -> str:
    v = data.get(key, default)
    if v is None:
        raise SpecError("缺少必填字段", field=field_path)
    if not isinstance(v, str):
        raise SpecError(f"应为字符串，实际是 {type(v).__name__}", field=field_path)
    return v


def _opt_str(data: Mapping[str, Any], key: str, *, field_path: str) -> str | None:
    v = data.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise SpecError(f"应为字符串，实际是 {type(v).__name__}", field=field_path)
    return v.strip() or None


def _require_mapping(value: Any, *, field_path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SpecError(f"应为对象，实际是 {type(value).__name__}", field=field_path)
    return dict(value)


def _positive_int(value: Any, *, field_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"应为正整数，实际是 {value!r}", field=field_path)
    if value <= 0:
        raise SpecError(f"应为正整数，实际是 {value}", field=field_path)
    return value


# ─────────────────────────────── metadata ───────────────────────────────


@dataclass(frozen=True)
class Metadata:
    """作业身份。`owner` 由服务端权威覆写，客户端声明无效。"""

    name: str
    project: str = ""
    owner: str = ""
    display_name: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Metadata:
        d = _require_mapping(data, field_path="metadata")
        labels = _require_mapping(d.get("labels"), field_path="metadata.labels")
        return cls(
            name=safe_segment(_require_str(d, "name", field_path="metadata.name"), field="metadata.name"),
            project=safe_project(d.get("project") or ""),
            owner=(d.get("owner") or "").strip(),
            display_name=(d.get("display_name") or "").strip(),
            labels={str(k): str(v) for k, v in labels.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.project:
            out["project"] = self.project
        if self.owner:
            out["owner"] = self.owner
        if self.display_name:
            out["display_name"] = self.display_name
        if self.labels:
            out["labels"] = dict(self.labels)
        return out


# ─────────────────────────────── provenance ───────────────────────────────


@dataclass(frozen=True)
class Provenance:
    """提交时记录的既成事实（不是用户意图），用于事后复现与追责。

    与 spec 分开放在顶层：spec 描述「想跑什么」，provenance 描述「实际提交的是哪份代码」。
    """

    git_commit: str = ""
    git_dirty: bool = False
    config_sha: str = ""
    model_name: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Provenance:
        d = _require_mapping(data, field_path="provenance")
        return cls(
            git_commit=str(d.get("git_commit") or "").strip(),
            git_dirty=bool(d.get("git_dirty")),
            config_sha=str(d.get("config_sha") or "").strip(),
            model_name=str(d.get("model_name") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.git_commit:
            out["git_commit"] = self.git_commit
        if self.git_dirty:
            out["git_dirty"] = True
        if self.config_sha:
            out["config_sha"] = self.config_sha
        if self.model_name:
            out["model_name"] = self.model_name
        return out


# ─────────────────────────────── recipe ───────────────────────────────


@dataclass(frozen=True)
class RecipeRef:
    """算法（后训练方法）引用 —— 平台侧的一等公民。

    `plugins` 列出该作业要装载的算法补丁（如 opsd / maxrl）。改造前这些补丁由每个实验的
    run.py 自行 import 并 install_*()，平台无从知晓；现在由 launcher 按此字段统一装载，
    平台因而能记录「哪个作业用了哪个补丁的哪个版本」。
    """

    name: str
    version: str = ""
    plugins: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> RecipeRef:
        d = _require_mapping(data, field_path="spec.recipe")
        if not d:
            raise SpecError("缺少必填字段", field="spec.recipe")
        name = safe_segment(_require_str(d, "name", field_path="spec.recipe.name"), field="spec.recipe.name")
        raw_plugins = d.get("plugins") or ()
        if isinstance(raw_plugins, (str, bytes)) or not isinstance(raw_plugins, Sequence):
            raise SpecError("应为字符串数组", field="spec.recipe.plugins")
        plugins = tuple(
            safe_segment(str(p), field="spec.recipe.plugins[]") for p in raw_plugins
        )
        return cls(name=name, version=str(d.get("version") or "").strip(), plugins=plugins)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.version:
            out["version"] = self.version
        if self.plugins:
            out["plugins"] = list(self.plugins)
        return out


@dataclass(frozen=True)
class FrameworkRef:
    kind: str = "nemo-rl"
    image: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> FrameworkRef:
        d = _require_mapping(data, field_path="spec.framework")
        kind = (d.get("kind") or "nemo-rl").strip()
        if kind not in FRAMEWORKS:
            raise SpecError(f"未知框架 {kind!r}（可选：{', '.join(FRAMEWORKS)}）", field="spec.framework.kind")
        return cls(kind=kind, image=str(d.get("image") or "").strip())

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.image:
            out["image"] = self.image
        return out


# ─────────────────────────────── source ───────────────────────────────


@dataclass(frozen=True)
class SourceSpec:
    """代码来源：上传包内的实验相对路径 + 可选的显式入口。

    `entrypoint` 为空时由 recipe 决定入口（recipe=legacy 时回落到集群侧的老规则：
    实验目录有 run.py 用它，否则 examples/run_grpo.py）。
    """

    exp: str
    entrypoint: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> SourceSpec:
        d = _require_mapping(data, field_path="spec.source")
        if not d:
            raise SpecError("缺少必填字段", field="spec.source")
        return cls(
            exp=safe_exp_rel(_require_str(d, "exp", field_path="spec.source.exp")),
            entrypoint=str(d.get("entrypoint") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"exp": self.exp}
        if self.entrypoint:
            out["entrypoint"] = self.entrypoint
        return out


# ─────────────────────────────── model ───────────────────────────────


@dataclass(frozen=True)
class ArtifactRef:
    """指向某次 run 产出的产物，用于多阶段后训练串联（SFT → DPO → GRPO）。

    文本形式：``run/<run_id>/<kind>`` 或 ``run/<run_id>/<kind>@step=<N>``
    服务端负责校验引用存在性与访问权限（阶段 5 的产物注册表）。
    """

    run_id: str
    kind: str = "checkpoint"
    step: int | None = None

    @classmethod
    def parse(cls, text: str, *, field_path: str = "spec.model.init_from") -> ArtifactRef:
        raw = (text or "").strip()
        if not raw:
            raise SpecError("产物引用不能为空", field=field_path)
        body, _, step_part = raw.partition("@")
        step: int | None = None
        if step_part:
            key, _, val = step_part.partition("=")
            if key.strip() != "step" or not val.strip().isdigit():
                raise SpecError(f"非法产物限定符 {step_part!r}（仅支持 @step=<整数>）", field=field_path)
            step = int(val)
        segs = [s for s in body.split("/") if s]
        if len(segs) != 3 or segs[0] != "run":
            raise SpecError(
                f"非法产物引用 {text!r}（应为 run/<run_id>/<kind>[@step=N]）", field=field_path
            )
        return cls(
            run_id=safe_segment(segs[1], field=field_path),
            kind=safe_segment(segs[2], field=field_path),
            step=step,
        )

    def to_text(self) -> str:
        base = f"run/{self.run_id}/{self.kind}"
        return f"{base}@step={self.step}" if self.step is not None else base


@dataclass(frozen=True)
class PeftSpec:
    kind: str = ""
    dim: int | None = None
    alpha: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PeftSpec | None:
        d = _require_mapping(data, field_path="spec.model.peft")
        if not d:
            return None
        return cls(
            kind=str(d.get("kind") or "").strip(),
            dim=d.get("dim"),
            alpha=d.get("alpha"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.kind:
            out["kind"] = self.kind
        if self.dim is not None:
            out["dim"] = self.dim
        if self.alpha is not None:
            out["alpha"] = self.alpha
        return out


@dataclass(frozen=True)
class ModelSpec:
    base: str = ""
    peft: PeftSpec | None = None
    init_from: ArtifactRef | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ModelSpec:
        d = _require_mapping(data, field_path="spec.model")
        init_raw = _opt_str(d, "init_from", field_path="spec.model.init_from")
        return cls(
            base=str(d.get("base") or "").strip(),
            peft=PeftSpec.from_dict(d.get("peft")),
            init_from=ArtifactRef.parse(init_raw) if init_raw else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.base:
            out["base"] = self.base
        if self.peft:
            peft = self.peft.to_dict()
            if peft:
                out["peft"] = peft
        if self.init_from:
            out["init_from"] = self.init_from.to_text()
        return out


# ─────────────────────────────── data ───────────────────────────────


@dataclass(frozen=True)
class DataRef:
    dataset: str = ""
    split: str = ""
    path: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, field_path: str) -> DataRef | None:
        d = _require_mapping(data, field_path=field_path)
        if not d:
            return None
        return cls(
            dataset=str(d.get("dataset") or "").strip(),
            split=str(d.get("split") or "").strip(),
            path=str(d.get("path") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in (("dataset", self.dataset), ("split", self.split), ("path", self.path)) if v}


@dataclass(frozen=True)
class DataSpec:
    train: DataRef | None = None
    validation: DataRef | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> DataSpec:
        d = _require_mapping(data, field_path="spec.data")
        return cls(
            train=DataRef.from_dict(d.get("train"), field_path="spec.data.train"),
            validation=DataRef.from_dict(d.get("validation"), field_path="spec.data.validation"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.train:
            out["train"] = self.train.to_dict()
        if self.validation:
            out["validation"] = self.validation.to_dict()
        return out


# ─────────────────────────────── resources ───────────────────────────────


@dataclass(frozen=True)
class ResourcePool:
    """一组同构 GPU。借鉴 verl 的 ResourcePool：角色可共享一个池，也可分池。

    卡数由 nodes × gpus_per_node 得出，**不再从 overrides.conf 正则抠文本**。
    """

    name: str
    series: str = ""
    nodes: int = 1
    gpus_per_node: int = 1

    @property
    def gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, field_path: str) -> ResourcePool:
        d = _require_mapping(data, field_path=field_path)
        return cls(
            name=safe_segment(_require_str(d, "name", field_path=f"{field_path}.name"), field=f"{field_path}.name"),
            series=str(d.get("series") or "").strip(),
            nodes=_positive_int(d.get("nodes", 1), field_path=f"{field_path}.nodes"),
            gpus_per_node=_positive_int(d.get("gpus_per_node", 1), field_path=f"{field_path}.gpus_per_node"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "nodes": self.nodes, "gpus_per_node": self.gpus_per_node}
        if self.series:
            out["series"] = self.series
        return out


@dataclass(frozen=True)
class ResourceSpec:
    """资源诉求。空 pools 表示「由 profile 决定」（legacy 路径）。"""

    pools: tuple[ResourcePool, ...] = ()
    roles: dict[str, str] = field(default_factory=dict)

    @property
    def total_gpus(self) -> int:
        """全部池的卡数之和 —— 配额准入与计量的口径。"""
        return sum(p.gpus for p in self.pools)

    def pool(self, name: str) -> ResourcePool | None:
        return next((p for p in self.pools if p.name == name), None)

    def pool_for_role(self, role: str) -> ResourcePool | None:
        target = self.roles.get(role)
        return self.pool(target) if target else None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ResourceSpec:
        d = _require_mapping(data, field_path="spec.resources")
        raw_pools = d.get("pools") or ()
        if isinstance(raw_pools, (str, bytes)) or not isinstance(raw_pools, Sequence):
            raise SpecError("应为数组", field="spec.resources.pools")
        pools = tuple(
            ResourcePool.from_dict(p, field_path=f"spec.resources.pools[{i}]") for i, p in enumerate(raw_pools)
        )
        seen: set[str] = set()
        for p in pools:
            if p.name in seen:
                raise SpecError(f"资源池名重复: {p.name}", field="spec.resources.pools")
            seen.add(p.name)
        roles_raw = _require_mapping(d.get("roles"), field_path="spec.resources.roles")
        roles = {str(k): str(v) for k, v in roles_raw.items()}
        for role, pool_name in roles.items():
            if pool_name not in seen:
                raise SpecError(
                    f"角色 {role!r} 指向未定义的资源池 {pool_name!r}", field="spec.resources.roles"
                )
        return cls(pools=pools, roles=roles)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.pools:
            out["pools"] = [p.to_dict() for p in self.pools]
        if self.roles:
            out["roles"] = dict(self.roles)
        return out


# ─────────────────────────────── lifecycle ───────────────────────────────

#: 训练成功后可自动触发的动作。改造前这是服务端硬编码的 POST_ACTIONS 白名单；
#: 现在由各 recipe 在自己的 recipe.yaml 里声明支持哪些，服务端只做交集校验。
KNOWN_LIFECYCLE_ACTIONS = ("export", "eval")


@dataclass(frozen=True)
class LifecycleSpec:
    on_success: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> LifecycleSpec:
        d = _require_mapping(data, field_path="spec.lifecycle")
        raw = d.get("on_success") or ()
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise SpecError("应为字符串数组", field="spec.lifecycle.on_success")
        actions = tuple(str(a).strip() for a in raw if str(a).strip())
        for a in actions:
            safe_segment(a, field="spec.lifecycle.on_success[]")
        return cls(on_success=actions)

    def to_dict(self) -> dict[str, Any]:
        return {"on_success": list(self.on_success)} if self.on_success else {}


# ─────────────────────────────── 顶层 ───────────────────────────────


@dataclass(frozen=True)
class JobSpecBody:
    recipe: RecipeRef
    source: SourceSpec
    framework: FrameworkRef = field(default_factory=FrameworkRef)
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    hyperparams: dict[str, Any] = field(default_factory=dict)
    lifecycle: LifecycleSpec = field(default_factory=LifecycleSpec)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> JobSpecBody:
        d = _require_mapping(data, field_path="spec")
        if not d:
            raise SpecError("缺少必填字段", field="spec")
        return cls(
            recipe=RecipeRef.from_dict(d.get("recipe")),
            source=SourceSpec.from_dict(d.get("source")),
            framework=FrameworkRef.from_dict(d.get("framework")),
            model=ModelSpec.from_dict(d.get("model")),
            data=DataSpec.from_dict(d.get("data")),
            resources=ResourceSpec.from_dict(d.get("resources")),
            hyperparams=_require_mapping(d.get("hyperparams"), field_path="spec.hyperparams"),
            lifecycle=LifecycleSpec.from_dict(d.get("lifecycle")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "recipe": self.recipe.to_dict(),
            "source": self.source.to_dict(),
            "framework": self.framework.to_dict(),
        }
        for key, value in (
            ("model", self.model.to_dict()),
            ("data", self.data.to_dict()),
            ("resources", self.resources.to_dict()),
            ("hyperparams", dict(self.hyperparams)),
            ("lifecycle", self.lifecycle.to_dict()),
        ):
            if value:
                out[key] = value
        return out


@dataclass(frozen=True)
class JobSpec:
    """一份完整的作业规格。"""

    metadata: Metadata
    spec: JobSpecBody
    provenance: Provenance = field(default_factory=Provenance)
    api_version: str = API_VERSION
    kind: str = KIND_TRAINING

    # 便捷读取（下游大量使用，避免到处写 spec.spec.source.exp）
    @property
    def exp(self) -> str:
        return self.spec.source.exp

    @property
    def recipe_name(self) -> str:
        return self.spec.recipe.name

    @property
    def is_legacy(self) -> bool:
        return self.spec.recipe.name == RECIPE_LEGACY

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobSpec:
        d = _require_mapping(data, field_path="<root>")
        api_version = _require_str(d, "apiVersion", field_path="apiVersion", default=API_VERSION)
        if api_version != API_VERSION:
            raise SpecError(
                f"不支持的 apiVersion {api_version!r}（本端支持 {API_VERSION}）", field="apiVersion"
            )
        kind = _require_str(d, "kind", field_path="kind", default=KIND_TRAINING)
        if kind != KIND_TRAINING:
            raise SpecError(f"不支持的 kind {kind!r}（本端支持 {KIND_TRAINING}）", field="kind")
        return cls(
            api_version=api_version,
            kind=kind,
            metadata=Metadata.from_dict(d.get("metadata")),
            spec=JobSpecBody.from_dict(d.get("spec")),
            provenance=Provenance.from_dict(d.get("provenance")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": self.metadata.to_dict(),
            "spec": self.spec.to_dict(),
        }
        prov = self.provenance.to_dict()
        if prov:
            out["provenance"] = prov
        return out

    def with_owner(self, owner: str) -> JobSpec:
        """服务端权威覆写提交者。客户端声明的 owner 一律无效。"""
        from dataclasses import replace

        return replace(self, metadata=replace(self.metadata, owner=owner))
