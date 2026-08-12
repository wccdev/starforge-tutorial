"""Recipe 数据模型与超参校验器。

为什么是自定义的迷你 schema 而不是 JSON Schema
──────────────────────────────────────────────────────────────────────────────
本模块要能在 NeMo-RL 官方容器里 import（集群侧 launcher 用它解析 recipe），
而**运行时**装不了新包 —— 注意这不是网络问题：内网 Nexus3 能代理 PyPI，
overlay 镜像里装什么都行；装不进来的是**作业级**依赖，因为 Ray 的
runtime_env pip 只作用于 driver 进程，训练 worker 用的仍是容器 venv。

所以这里的取舍不是「没得选」，而是权衡：需要覆盖的场景很窄（类型、必填、
区间、枚举、默认值），一个约 80 行的校验器就够，且不必为它重建镜像。
真需要完整 JSON Schema（嵌套对象、oneOf）时，把 jsonschema 加进 overlay
镜像的 EXTRA_PIP 即可，这条路是通的。

pyyaml 是 nemo-rl-lab 的既有依赖，且 NeMo-RL 本身用 YAML 配置，容器内必有。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from nemo_rl_lab.contract.errors import SpecError

#: 入口脚本的基准目录。
#:   nemo_rl —— 相对容器内 NEMO_RL_DIR（官方 examples/*.py）
#:   workdir —— 相对上传包根（仓库自带脚本）
#:   exp     —— 相对实验目录（实验自带 run.py）
ENTRYPOINT_BASES = ("nemo_rl", "workdir", "exp")

_TYPES: dict[str, tuple[type, ...]] = {
    "int": (int,),
    "float": (int, float),  # 允许整数字面量喂给浮点参数（YAML 里 0 与 0.0 常混用）
    "bool": (bool,),
    "str": (str,),
}


@dataclass(frozen=True)
class ParamSpec:
    """一个可调超参的声明。

    `path` 是它在训练框架配置里的落点（点分路径）。有了它，平台就能把 JobSpec 的
    扁平超参翻译成 NeMo-RL 的 CLI override —— 这是「服务端理解作业内容」的具体体现，
    改造前服务端对超参一无所知。
    """

    name: str
    type: str = "str"
    path: str = ""
    doc: str = ""
    required: bool = False
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    choices: tuple[Any, ...] = ()

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> ParamSpec:
        t = str(data.get("type") or "str")
        if t not in _TYPES and t != "enum":
            raise SpecError(f"recipe 参数 {name} 声明了未知类型 {t!r}", field=f"params.{name}.type")
        raw_choices = data.get("choices") or ()
        if isinstance(raw_choices, (str, bytes)) or not isinstance(raw_choices, Sequence):
            raise SpecError(f"recipe 参数 {name} 的 choices 应为数组", field=f"params.{name}.choices")
        return cls(
            name=name,
            type=t,
            path=str(data.get("path") or ""),
            doc=str(data.get("doc") or ""),
            required=bool(data.get("required")),
            default=data.get("default"),
            minimum=data.get("min"),
            maximum=data.get("max"),
            exclusive_minimum=bool(data.get("exclusive_min")),
            choices=tuple(raw_choices),
        )

    def check(self, value: Any) -> Any:
        """校验单个取值；返回规范化后的值，非法则抛 SpecError。"""
        where = f"spec.hyperparams.{self.name}"

        if self.type == "enum":
            if value not in self.choices:
                raise SpecError(
                    f"取值 {value!r} 不在允许集合 {list(self.choices)} 内", field=where
                )
            return value

        expected = _TYPES[self.type]
        # bool 是 int 的子类：不显式排除的话 True 会被当成合法的 int。
        if self.type != "bool" and isinstance(value, bool):
            raise SpecError(f"应为 {self.type}，实际是 bool", field=where)
        if not isinstance(value, expected):
            raise SpecError(f"应为 {self.type}，实际是 {type(value).__name__}", field=where)

        if self.type in ("int", "float"):
            if self.minimum is not None:
                too_small = value <= self.minimum if self.exclusive_minimum else value < self.minimum
                if too_small:
                    op = ">" if self.exclusive_minimum else ">="
                    raise SpecError(f"应满足 {op} {self.minimum}，实际是 {value}", field=where)
            if self.maximum is not None and value > self.maximum:
                raise SpecError(f"应满足 <= {self.maximum}，实际是 {value}", field=where)
        if self.choices and value not in self.choices:
            raise SpecError(f"取值 {value!r} 不在允许集合 {list(self.choices)} 内", field=where)
        return value


@dataclass(frozen=True)
class Runtime:
    """该方法对运行环境的要求。

    背景：Ray 集群在内网，但内网 Nexus3 可以代理任意上游（PyPI / Docker），
    因此 overlay 镜像是**通的**（见 deploy/ray-cluster/Dockerfile 的 EXTRA_PIP
    与 PIP_INDEX）。真正装不进来的是**作业级**依赖 —— Ray 的 runtime_env pip
    只作用于 driver 进程，训练 worker 用的仍是容器 venv。这两个约束常被混为一谈。

    于是正确的做法不是「不许有依赖」，而是**声明依赖并在启动时校验**：
      - 缺包/版本不符 → 启动即报错，并给出该跑哪条构建命令
      - 而不是训练跑起来几分钟后一句 ImportError

    `requires` 用 PEP 508 语法（"trl>=0.9"）。

    ⚠️ 单胖镜像有天花板：TRL 与 NeMo-RL 都吃 transformers，版本互斥时装谁都会
       让另一边挂掉。声明式依赖让这种冲突在**加 recipe 时**就能被发现，
       而不是在某次镜像重建后炸掉所有训练。
    """

    requires: tuple[str, ...] = ()
    #: 该方法要求的 overlay 镜像标签；留空表示用部署默认镜像。
    image: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, recipe: str) -> Runtime:
        d = dict(data or {})
        raw = d.get("requires") or ()
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise SpecError(f"recipe {recipe} 的 runtime.requires 应为字符串数组")
        return cls(requires=tuple(str(r).strip() for r in raw if str(r).strip()),
                   image=str(d.get("image") or "").strip())

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.requires:
            out["requires"] = list(self.requires)
        if self.image:
            out["image"] = self.image
        return out


@dataclass(frozen=True)
class Entrypoint:
    base: str
    path: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, recipe: str) -> Entrypoint:
        d = dict(data or {})
        base = str(d.get("base") or "nemo_rl")
        if base not in ENTRYPOINT_BASES:
            raise SpecError(
                f"recipe {recipe} 的 entrypoint.base={base!r} 未知（可选 {', '.join(ENTRYPOINT_BASES)}）"
            )
        path = str(d.get("path") or "").strip()
        if base != "exp" and not path:
            raise SpecError(f"recipe {recipe} 缺少 entrypoint.path")
        return cls(base=base, path=path)


@dataclass(frozen=True)
class Recipe:
    """一种后训练方法。加一个方法 = 加一个目录，服务端零改动。"""

    name: str
    version: str
    title: str = ""
    summary: str = ""
    framework: str = "nemo-rl"
    entrypoint: Entrypoint = field(default_factory=lambda: Entrypoint("nemo_rl", ""))
    #: 超参落进训练配置的哪个段（用于生成 override 前缀提示）。
    config_section: str = ""
    #: 该方法支持的训练后动作。取代服务端硬编码的 POST_ACTIONS 白名单。
    lifecycle: tuple[str, ...] = ()
    #: 默认装载的算法补丁（common/algorithms/ 下的注册名）。
    plugins: tuple[str, ...] = ()
    #: 该方法涉及的角色，供资源池映射校验（阶段 6）。
    roles: tuple[str, ...] = ()
    #: 核心指标契约：前端默认曲线与诊断阈值都读它。
    primary_metrics: tuple[str, ...] = ()
    #: 运行环境要求（overlay 镜像 / 额外依赖）。
    runtime: Runtime = field(default_factory=Runtime)
    params: dict[str, ParamSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Recipe:
        name = str(data.get("name") or "").strip()
        if not name:
            raise SpecError("recipe.yaml 缺少 name")
        raw_params = data.get("params") or {}
        if not isinstance(raw_params, Mapping):
            raise SpecError(f"recipe {name} 的 params 应为对象")
        metrics = data.get("metrics") or {}
        return cls(
            name=name,
            version=str(data.get("version") or "").strip(),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            framework=str(data.get("framework") or "nemo-rl").strip(),
            entrypoint=Entrypoint.from_dict(data.get("entrypoint"), recipe=name),
            config_section=str(data.get("config_section") or "").strip(),
            lifecycle=tuple(str(x) for x in (data.get("lifecycle") or ())),
            plugins=tuple(str(x) for x in (data.get("plugins") or ())),
            roles=tuple(str(x) for x in (data.get("roles") or ())),
            primary_metrics=tuple(str(x) for x in ((metrics or {}).get("primary") or ())),
            runtime=Runtime.from_dict(data.get("runtime"), recipe=name),
            params={k: ParamSpec.from_dict(k, v or {}) for k, v in raw_params.items()},
        )

    # ── 超参校验 ────────────────────────────────────────────────────────────

    def validate_hyperparams(self, params: Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
        """校验并补齐默认值。

        strict=True 时拒绝未声明的键 —— 这正是「提交前挡下拼错的超参名」的那道闸。
        改造前一个拼错的超参会一路跑到集群上，等训练起来才失败（或者更糟：被静默忽略，
        用户以为自己调了参，实际跑的是默认值）。
        """
        given = dict(params or {})
        if strict:
            unknown = sorted(set(given) - set(self.params))
            if unknown:
                known = ", ".join(sorted(self.params)) or "（该 recipe 未声明可调超参）"
                raise SpecError(
                    f"recipe {self.name} 不认识这些超参: {', '.join(unknown)}。可用: {known}",
                    field="spec.hyperparams",
                )

        out: dict[str, Any] = {}
        for key, spec in self.params.items():
            if key in given:
                out[key] = spec.check(given[key])
            elif spec.required:
                raise SpecError(f"recipe {self.name} 缺少必填超参 {key}", field="spec.hyperparams")
            elif spec.default is not None:
                out[key] = spec.default
        # 非 strict 时保留未声明的键（legacy 路径不校验超参）
        for key, value in given.items():
            out.setdefault(key, value)
        return out

    def config_overrides(self, params: Mapping[str, Any]) -> list[str]:
        """把超参翻译成训练框架的 CLI override（`a.b.c=value`）。

        只翻译声明了 `path` 的参数；没有 path 的参数由 recipe 入口自行消费。
        """
        out: list[str] = []
        for key, value in (params or {}).items():
            spec = self.params.get(key)
            if not spec or not spec.path:
                continue
            out.append(f"{spec.path}={_render(value)}")
        return out

    def supports(self, action: str) -> bool:
        return action in self.lifecycle


def _render(value: Any) -> str:
    """按训练框架 CLI override 的字面量约定渲染取值。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
