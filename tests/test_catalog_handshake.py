from __future__ import annotations

import hashlib
import json

import pytest
from nemo_lab_sdk.recipes import get_recipe

from nemo_rl_lab.catalog import CatalogCompatibilityError, verify_catalog_compatibility
from nemo_rl_lab.spec_builder import build_spec


def _payload(spec, **recipe_overrides):
    recipe = get_recipe(spec.recipe_name)
    item = {
        "name": recipe.id,
        "version": recipe.version,
        "digest": recipe.digest,
        "framework": recipe.framework,
        "adapter": recipe.framework,
        "framework_version": recipe.runtime.default_version,
        "supported_framework_versions": list(recipe.runtime.supported_versions),
        "runtime": recipe.runtime.to_dict(),
    }
    item.update(recipe_overrides)
    canonical = [{"name": item["name"], "version": item["version"], "digest": item["digest"]}]
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "apiVersion": "lab/recipe-catalog/v2",
        "contract": {"versions": ["lab/v2"]},
        "sdk": {"version": "2.1.0", "requirement": ">=2.1,<3"},
        "catalog_digest": f"sha256:{digest}",
        "recipes": [item],
    }


def _spec():
    return build_spec("experiments/demo", recipe="nemo-rl/grpo", pools=["all:h100:1:1"])


def test_exact_catalog_handshake_accepts_matching_recipe():
    verify_catalog_compatibility(_spec(), _payload(_spec()))


@pytest.mark.parametrize(
    "field,value",
    [("version", "0.0.0"), ("digest", "sha256:wrong"), ("framework", "verl"), ("adapter", "verl")],
)
def test_catalog_handshake_rejects_recipe_drift(field, value):
    spec = _spec()
    with pytest.raises(CatalogCompatibilityError):
        verify_catalog_compatibility(spec, _payload(spec, **{field: value}))


def test_catalog_handshake_rejects_sdk_drift():
    spec = _spec()
    payload = _payload(spec)
    payload["sdk"] = {"version": "9.0.0", "requirement": ">=9,<10"}
    with pytest.raises(CatalogCompatibilityError, match="SDK"):
        verify_catalog_compatibility(spec, payload)


def test_catalog_handshake_accepts_compatible_sdk_range():
    spec = _spec()
    payload = _payload(spec)
    payload["sdk"] = {"version": "2.2.0", "requirement": ">=2.1,<3"}
    verify_catalog_compatibility(spec, payload)


def test_catalog_handshake_still_accepts_legacy_v1_exact_match():
    spec = _spec()
    payload = _payload(spec)
    payload["apiVersion"] = "lab/recipe-catalog/v1"
    payload["sdk"] = {"version": "2.1.0", "requirement": "==2.1.0"}
    verify_catalog_compatibility(spec, payload)


def test_catalog_handshake_rejects_tampered_catalog_digest():
    spec = _spec()
    payload = _payload(spec)
    payload["catalog_digest"] = "sha256:tampered"
    with pytest.raises(CatalogCompatibilityError, match="catalog digest"):
        verify_catalog_compatibility(spec, payload)


def test_catalog_handshake_rejects_framework_version_or_runtime_id_drift():
    spec = _spec()
    payload = _payload(spec, supported_framework_versions=["9.9.9"])
    with pytest.raises(CatalogCompatibilityError, match="未发布"):
        verify_catalog_compatibility(spec, payload)

    payload = _payload(spec)
    payload["recipes"][0]["runtime"]["versions"]["0.7.0"]["runtime_id"] = "wrong-runtime"
    with pytest.raises(CatalogCompatibilityError, match="runtime_id"):
        verify_catalog_compatibility(spec, payload)


def test_submit_does_not_package_when_handshake_fails(tmp_path, monkeypatch):
    from nemo_rl_lab import api_client

    spec = _spec()
    packed = []

    monkeypatch.setattr(api_client, "current_server", lambda _server=None: "https://lab.example")
    monkeypatch.setattr(
        api_client,
        "verify_server_compatibility",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CatalogCompatibilityError("drift")),
    )
    monkeypatch.setattr(
        api_client,
        "pack_working_dir",
        lambda *_args, **_kwargs: packed.append(True),
    )

    with pytest.raises(CatalogCompatibilityError, match="drift"):
        api_client.submit_via_server(
            "experiments/demo", "h100", tmp_path, spec=spec
        )
    assert packed == []


def test_submit_requires_explicit_profile(tmp_path):
    from nemo_rl_lab import api_client

    with pytest.raises(ValueError, match="profile"):
        api_client.submit_via_server("experiments/demo", "", tmp_path, spec=_spec())
