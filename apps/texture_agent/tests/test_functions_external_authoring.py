# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from texture_agent.functions.external_authoring import (
    ExternalAuthoringCapabilityReceipt,
    ExternalAuthoringSpec,
    _file_uri_path,
    evaluate_external_authoring_feasibility,
    external_authoring_spec_digest,
    external_authoring_spec_payload,
    unavailable_external_authoring_feasibility,
    validate_external_authoring_result,
)


def _spec(**overrides: object) -> ExternalAuthoringSpec:
    config: dict[str, object] = {
        "adapter_id": "approved-painter-adapter",
        "workflow": "paint",
        "tool_name": "Approved Painter",
        "tool_version": "1.2.3",
        "required_map_channels": ["albedo", "normal", "orm"],
        "required_auxiliary_artifacts": ["project"],
        "parameters": {"preset_name": "painted-metal-v1"},
    }
    config.update(overrides)
    return ExternalAuthoringSpec.from_config(config)


def _receipt_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "texture-agent-external-authoring-capabilities.v1",
        "ready": True,
        "adapter_id": "approved-painter-adapter",
        "adapter_version": "4.5.6",
        "tool_name": "Approved Painter",
        "tool_version": "1.2.3",
        "headless": True,
        "license_status": "valid",
        "deployment_mode": "remote_headless",
        "environment_digest": "sha256:" + "a" * 64,
        "supported_workflows": ["paint"],
        "supported_map_channels": ["albedo", "normal", "orm"],
        "supported_auxiliary_artifacts": ["project"],
        "seed_control": True,
        "deterministic_parameters": True,
        "normalized_output": "texture_variation_maps",
        "diagnostics": [],
    }
    payload.update(overrides)
    return payload


def _receipt(**overrides: object) -> ExternalAuthoringCapabilityReceipt:
    return ExternalAuthoringCapabilityReceipt.from_payload(
        _receipt_payload(**overrides)
    )


def _result_metadata(
    spec: ExternalAuthoringSpec,
    receipt: ExternalAuthoringCapabilityReceipt,
    source: Path,
    output_sha256: dict[str, str],
    *,
    seed: int | None = 1046,
) -> dict[str, object]:
    return {
        "schema_version": "texture-agent-external-authoring-result.v1",
        "spec_digest": external_authoring_spec_digest(spec),
        "adapter_id": spec.adapter_id,
        "adapter_version": receipt.adapter_version,
        "workflow": spec.workflow,
        "tool_name": receipt.tool_name,
        "tool_version": receipt.tool_version,
        "headless": True,
        "license_status": "valid",
        "deployment_mode": receipt.deployment_mode,
        "environment_digest": receipt.environment_digest,
        "seed": seed,
        "source_asset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": output_sha256,
    }


def test_external_authoring_spec_digest_is_stable_and_in_request() -> None:
    spec = _spec()

    first = external_authoring_spec_digest(spec)
    second = external_authoring_spec_digest(_spec())
    payload = external_authoring_spec_payload(spec)

    assert first == second
    assert len(first) == 64
    assert payload["spec_digest"] == first
    assert payload["required_map_channels"] == ["albedo", "normal", "orm"]


def test_external_authoring_spec_accepts_optional_lists_and_safe_uris() -> None:
    spec = _spec(
        required_auxiliary_artifacts=None,
        preset_uri="https://assets.example/presets/painted-metal-v1.spp",
    )

    assert spec.required_auxiliary_artifacts == ()
    assert spec.preset_uri == "https://assets.example/presets/painted-metal-v1.spp"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"headless_required": False}, "interactive DCC execution"),
        ({"required_map_channels": ["normal"]}, "must include albedo"),
        ({"normalized_output": "replacement_usd"}, "Texture Agent owns USD/PBR"),
        ({"parameters": {"api_key": "not-allowed"}}, "must not contain credentials"),
        (
            {"preset_uri": "https://example/preset?token=not-allowed"},
            "must not embed credentials",
        ),
        (
            {"preset_uri": "https://user:password@example/preset"},
            "must not embed credentials",
        ),
        (
            {"preset_uri": "https://example/preset?X-Amz-Signature=not-allowed"},
            "must not embed credentials",
        ),
        (
            {"parameters": {"layers": [{"secret_token": "not-allowed"}]}},
            "must not contain credentials",
        ),
        ({"tool_name": "   "}, "tool_name must be non-empty when provided"),
        ({"tool_version": "   "}, "tool_version must be non-empty when provided"),
    ],
)
def test_external_authoring_spec_rejects_unsafe_contracts(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(**override)


def test_external_authoring_feasibility_go_and_no_go() -> None:
    spec = _spec()

    assert evaluate_external_authoring_feasibility(spec, _receipt())["verdict"] == "go"

    no_go = evaluate_external_authoring_feasibility(
        spec,
        _receipt(
            ready=False,
            license_status="unavailable",
            headless=False,
            seed_control=False,
        ),
    )
    assert no_go["verdict"] == "no_go"
    assert {item["code"] for item in no_go["diagnostics"]} >= {
        "EXTERNAL_AUTHORING_NOT_READY",
        "EXTERNAL_AUTHORING_LICENSE_UNAVAILABLE",
        "EXTERNAL_AUTHORING_HEADLESS_UNAVAILABLE",
        "EXTERNAL_AUTHORING_REPRODUCIBILITY_UNSUPPORTED",
    }


def test_external_authoring_feasibility_rejects_contract_mismatches() -> None:
    no_go = evaluate_external_authoring_feasibility(
        _spec(),
        _receipt(
            adapter_id="different-adapter",
            adapter_version="",
            tool_name="",
            tool_version="",
            deployment_mode="interactive",
            environment_digest="",
            supported_workflows=["designer_graph"],
            supported_map_channels=["albedo"],
            supported_auxiliary_artifacts=[],
            normalized_output="replacement_usd",
        ),
    )

    assert no_go["verdict"] == "no_go"
    assert {item["code"] for item in no_go["diagnostics"]} >= {
        "EXTERNAL_AUTHORING_ADAPTER_MISMATCH",
        "EXTERNAL_AUTHORING_PROVENANCE_INCOMPLETE",
        "EXTERNAL_AUTHORING_TOOL_MISMATCH",
        "EXTERNAL_AUTHORING_VERSION_MISMATCH",
        "EXTERNAL_AUTHORING_DEPLOYMENT_UNSUPPORTED",
        "EXTERNAL_AUTHORING_WORKFLOW_UNSUPPORTED",
        "EXTERNAL_AUTHORING_OUTPUT_UNSUPPORTED",
        "EXTERNAL_AUTHORING_NORMALIZATION_UNSUPPORTED",
    }


def test_unavailable_external_authoring_feasibility_is_no_go() -> None:
    feasibility = unavailable_external_authoring_feasibility(_spec(), "HTTP 404")

    assert feasibility["verdict"] == "no_go"
    assert feasibility["diagnostics"][0]["code"] == (
        "EXTERNAL_AUTHORING_PREFLIGHT_UNAVAILABLE"
    )


def test_external_authoring_result_binds_source_maps_and_preflight(
    tmp_path: Path,
) -> None:
    spec = _spec()
    receipt = _receipt()
    source = tmp_path / "prepared.usd"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    maps = {}
    output_sha256 = {}
    for channel in spec.required_map_channels:
        path = tmp_path / f"{channel}.png"
        path.write_bytes(f"{channel}-bytes".encode())
        maps[channel] = str(path)
        output_sha256[channel] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "external_authoring": _result_metadata(
            spec,
            receipt,
            source,
            output_sha256,
        )
    }
    auxiliary = {"external_authoring": {"project": [{"uri": "file:///project"}]}}

    assert (
        validate_external_authoring_result(
            spec=spec,
            receipt=receipt,
            metadata=metadata,
            auxiliary_artifacts=auxiliary,
            source_asset_uri=source.as_uri(),
            seed=1046,
            map_paths=maps,
        )
        == []
    )

    metadata["external_authoring"]["environment_digest"] = "sha256:changed"
    output_sha256["albedo"] = "0" * 64
    diagnostics = validate_external_authoring_result(
        spec=spec,
        receipt=receipt,
        metadata=metadata,
        auxiliary_artifacts={},
        source_asset_uri=source.as_uri(),
        seed=1046,
        map_paths=maps,
    )
    assert {item["code"] for item in diagnostics} >= {
        "EXTERNAL_AUTHORING_PROVENANCE_MISMATCH",
        "EXTERNAL_AUTHORING_OUTPUT_DIGEST_MISMATCH",
        "EXTERNAL_AUTHORING_AUXILIARY_MISSING",
    }


def test_external_authoring_result_rejects_incomplete_provenance_and_outputs(
    tmp_path: Path,
) -> None:
    spec = _spec()
    receipt = _receipt()
    source = tmp_path / "prepared.usd"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    albedo = tmp_path / "albedo.png"
    albedo.write_bytes(b"albedo-bytes")
    maps = {"albedo": str(albedo), "normal": "https://example/normal.png"}

    missing = validate_external_authoring_result(
        spec=spec,
        receipt=receipt,
        metadata={},
        auxiliary_artifacts={},
        source_asset_uri=source.as_uri(),
        seed=1046,
        map_paths=maps,
    )
    assert [item["code"] for item in missing] == [
        "EXTERNAL_AUTHORING_PROVENANCE_MISSING"
    ]

    provenance = _result_metadata(spec, receipt, source, {}, seed=None)
    provenance["schema_version"] = "unsupported"
    provenance["source_asset_sha256"] = "0" * 64
    provenance["output_sha256"] = None
    diagnostics = validate_external_authoring_result(
        spec=spec,
        receipt=receipt,
        metadata={"external_authoring": provenance},
        auxiliary_artifacts={},
        source_asset_uri=str(source),
        seed=None,
        map_paths=maps,
    )
    assert {item["code"] for item in diagnostics} >= {
        "EXTERNAL_AUTHORING_PROVENANCE_INVALID",
        "EXTERNAL_AUTHORING_REPRODUCIBILITY_UNSUPPORTED",
        "EXTERNAL_AUTHORING_SOURCE_DIGEST_MISMATCH",
        "EXTERNAL_AUTHORING_OUTPUT_DIGEST_MISSING",
        "EXTERNAL_AUTHORING_OUTPUT_MISSING",
    }

    remote_source = validate_external_authoring_result(
        spec=spec,
        receipt=receipt,
        metadata={
            "external_authoring": _result_metadata(
                spec,
                receipt,
                source,
                {"albedo": hashlib.sha256(albedo.read_bytes()).hexdigest()},
            )
        },
        auxiliary_artifacts={},
        source_asset_uri="s3://approved-bucket/prepared.usd",
        seed=1046,
        map_paths=maps,
    )
    assert any(
        item["code"] == "EXTERNAL_AUTHORING_SOURCE_DIGEST_UNVERIFIABLE"
        for item in remote_source
    )

    assert _file_uri_path("file://remote-host/prepared.usd") is None
    assert _file_uri_path("relative/prepared.usd") == Path("relative/prepared.usd")
    assert _file_uri_path("s3://approved-bucket/prepared.usd") is None
