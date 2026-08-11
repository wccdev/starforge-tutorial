"""JobSpec 结构校验：非法输入必须在提交前被挡下，而不是跑到集群上才炸。

这些名字最终会被拼进集群产物路径、Ray metadata 与 shell 命令行，
校验规则只在 contract.names 定义一处，两侧共用。
"""
from __future__ import annotations

import pytest

from nemo_rl_lab.contract import ArtifactRef, JobSpec, ResourceSpec, SpecError, safe_exp_rel, safe_run_id


def _minimal() -> dict:
    return {
        "apiVersion": "lab/v1",
        "kind": "TrainingJob",
        "metadata": {"name": "job1"},
        "spec": {"recipe": {"name": "grpo"}, "source": {"exp": "experiments/foo"}},
    }


def test_minimal_spec_is_valid():
    spec = JobSpec.from_dict(_minimal())
    assert spec.recipe_name == "grpo"
    assert spec.exp == "experiments/foo"
    assert not spec.is_legacy


@pytest.mark.parametrize("bad", ["../etc/passwd", "/abs/path", "experiments/../..", "", "exp/../../x"])
def test_path_traversal_in_exp_is_rejected(bad):
    with pytest.raises(SpecError):
        safe_exp_rel(bad)


@pytest.mark.parametrize("bad", ["-rf", "a b", "a;rm -rf /", "..", "", "$(id)"])
def test_shell_metacharacters_in_run_id_are_rejected(bad):
    with pytest.raises(SpecError):
        safe_run_id(bad)


def test_unknown_api_version_is_rejected():
    d = _minimal() | {"apiVersion": "lab/v99"}
    with pytest.raises(SpecError, match="apiVersion"):
        JobSpec.from_dict(d)


def test_unknown_kind_is_rejected():
    d = _minimal() | {"kind": "Deployment"}
    with pytest.raises(SpecError, match="kind"):
        JobSpec.from_dict(d)


def test_missing_recipe_is_rejected():
    d = _minimal()
    del d["spec"]["recipe"]
    with pytest.raises(SpecError, match="spec.recipe"):
        JobSpec.from_dict(d)


def test_missing_source_is_rejected():
    d = _minimal()
    del d["spec"]["source"]
    with pytest.raises(SpecError, match="spec.source"):
        JobSpec.from_dict(d)


def test_unknown_framework_is_rejected():
    d = _minimal()
    d["spec"]["framework"] = {"kind": "megatron-lm"}
    with pytest.raises(SpecError, match="framework"):
        JobSpec.from_dict(d)


def test_role_pointing_at_undefined_pool_is_rejected():
    with pytest.raises(SpecError, match="未定义的资源池"):
        ResourceSpec.from_dict({"pools": [{"name": "train"}], "roles": {"actor": "nonexistent"}})


def test_duplicate_pool_names_are_rejected():
    with pytest.raises(SpecError, match="重复"):
        ResourceSpec.from_dict({"pools": [{"name": "train"}, {"name": "train"}]})


@pytest.mark.parametrize("bad", [0, -1, "2", 1.5, True])
def test_non_positive_gpu_counts_are_rejected(bad):
    with pytest.raises(SpecError):
        ResourceSpec.from_dict({"pools": [{"name": "train", "nodes": bad}]})


def test_empty_resources_means_profile_decides():
    """legacy 路径不带 pools，卡数仍由服务端 profile 注册表决定。"""
    rs = ResourceSpec.from_dict({})
    assert rs.pools == () and rs.total_gpus == 0


@pytest.mark.parametrize(
    "bad",
    [
        "checkpoint",                       # 缺 run/ 前缀
        "run/only-two",                     # 段数不足
        "run/a/b/c",                        # 段数过多
        "run/../x/checkpoint",              # 路径穿越
        "run/a/checkpoint@epoch=3",         # 不支持的限定符
        "run/a/checkpoint@step=abc",        # 非整数 step
    ],
)
def test_malformed_artifact_references_are_rejected(bad):
    with pytest.raises(SpecError):
        ArtifactRef.parse(bad)


def test_artifact_reference_without_step():
    ref = ArtifactRef.parse("run/sft-001/hf_export")
    assert (ref.run_id, ref.kind, ref.step) == ("sft-001", "hf_export", None)
    assert ref.to_text() == "run/sft-001/hf_export"


def test_illegal_project_name_is_rejected():
    d = _minimal()
    d["metadata"]["project"] = "a/b"
    with pytest.raises(SpecError, match="项目名"):
        JobSpec.from_dict(d)


def test_spec_error_is_a_value_error():
    """服务端提交路径已有 except ValueError -> HTTP 400，契约异常须落进这道网。"""
    assert issubclass(SpecError, ValueError)
