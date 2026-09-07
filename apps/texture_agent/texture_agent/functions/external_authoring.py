# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contract helpers for external texture-authoring tools.

External DCC products stay behind the Texture Variation service boundary.  This
module describes the caller-owned authoring intent, the backend capability
receipt required before launch, and the provenance that a completed job must
return before Texture Agent accepts its maps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, unquote, urlparse

EXTERNAL_AUTHORING_SPEC_SCHEMA_VERSION = "texture-agent-external-authoring.v1"
EXTERNAL_AUTHORING_CAPABILITIES_SCHEMA_VERSION = (
    "texture-agent-external-authoring-capabilities.v1"
)
EXTERNAL_AUTHORING_RESULT_SCHEMA_VERSION = "texture-agent-external-authoring-result.v1"
EXTERNAL_AUTHORING_FEASIBILITY_SCHEMA_VERSION = (
    "texture-agent-external-authoring-feasibility.v1"
)

_WORKFLOWS = frozenset({"paint", "designer_graph", "hybrid"})
_MAP_CHANNELS = frozenset({"albedo", "normal", "orm"})
_AUXILIARY_ARTIFACTS = frozenset({"project", "graph", "preset", "log"})
_DEPLOYMENT_MODES = frozenset({"local_headless", "remote_headless", "cloud_headless"})
_NORMALIZED_OUTPUT = "texture_variation_maps"
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "password",
        "secret",
        "signature",
        "token",
    }
)


def _diagnostic(
    code: str,
    *,
    message: str,
    recommended_action: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": code,
        "severity": "error",
        "stage": "generate_textures",
        "prim_path": None,
        "material_name": None,
        "message": message,
        "recommended_action": recommended_action,
        "details": details or {},
    }


def _nonempty(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"external_authoring.{field_name} must be non-empty")
    return text


def _optional_nonempty(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(
            f"external_authoring.{field_name} must be non-empty when provided"
        )
    return text


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"external_authoring.{field_name} must be a list")
    values = tuple(str(item).strip() for item in value if str(item).strip())
    if len(values) != len(value):
        raise ValueError(
            f"external_authoring.{field_name} must not contain empty values"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"external_authoring.{field_name} must not contain duplicates")
    return values


def _sensitive_parameter_paths(value: Any, path: str = "parameters") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                paths.append(f"{path}.{key_text}")
            paths.extend(_sensitive_parameter_paths(item, f"{path}.{key_text}"))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            paths.extend(_sensitive_parameter_paths(item, f"{path}[{index}]"))
    return paths


def _validate_nonsecret_uri(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    uri = value.strip()
    if not uri:
        raise ValueError(f"external_authoring.{field_name} must be non-empty")
    parsed = urlparse(uri)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            f"external_authoring.{field_name} must not embed credentials in its URI"
        )
    query_keys = {
        key.lower().replace("-", "_")
        for component in (parsed.query, parsed.fragment)
        for key, _value in parse_qsl(component, keep_blank_values=True)
    }
    if any(
        any(part in query_key for part in _SENSITIVE_KEY_PARTS)
        for query_key in query_keys
    ):
        raise ValueError(
            f"external_authoring.{field_name} must not embed credentials in its URI"
        )
    return uri


@dataclass(frozen=True)
class ExternalAuthoringSpec:
    """Backend-neutral authoring intent supplied with every DCC-backed job."""

    adapter_id: str
    workflow: Literal["paint", "designer_graph", "hybrid"]
    schema_version: str = EXTERNAL_AUTHORING_SPEC_SCHEMA_VERSION
    headless_required: bool = True
    tool_name: str | None = None
    tool_version: str | None = None
    preset_uri: str | None = None
    template_uri: str | None = None
    required_map_channels: tuple[str, ...] = ("albedo",)
    required_auxiliary_artifacts: tuple[str, ...] = ()
    normalized_output: str = _NORMALIZED_OUTPUT
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> ExternalAuthoringSpec:
        """Parse and validate ``texture.external_authoring`` config."""
        if not isinstance(value, dict):
            raise ValueError("texture.external_authoring must be an object")
        schema_version = str(
            value.get("schema_version") or EXTERNAL_AUTHORING_SPEC_SCHEMA_VERSION
        )
        if schema_version != EXTERNAL_AUTHORING_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "texture.external_authoring.schema_version must be "
                f"{EXTERNAL_AUTHORING_SPEC_SCHEMA_VERSION!r}"
            )
        adapter_id = _nonempty(value.get("adapter_id"), "adapter_id")
        workflow = _nonempty(value.get("workflow"), "workflow")
        if workflow not in _WORKFLOWS:
            raise ValueError(
                "external_authoring.workflow must be one of "
                + ", ".join(sorted(_WORKFLOWS))
            )
        headless_required = value.get("headless_required", True)
        if headless_required is not True:
            raise ValueError(
                "external_authoring.headless_required must be true; interactive "
                "DCC execution is not supported"
            )
        required_maps = _string_list(
            value.get("required_map_channels", ["albedo"]),
            "required_map_channels",
        )
        if not required_maps or "albedo" not in required_maps:
            raise ValueError(
                "external_authoring.required_map_channels must include albedo"
            )
        unknown_maps = sorted(set(required_maps) - _MAP_CHANNELS)
        if unknown_maps:
            raise ValueError(
                "external_authoring.required_map_channels contains unsupported "
                f"channels: {', '.join(unknown_maps)}"
            )
        required_auxiliary = _string_list(
            value.get("required_auxiliary_artifacts", []),
            "required_auxiliary_artifacts",
        )
        unknown_auxiliary = sorted(set(required_auxiliary) - _AUXILIARY_ARTIFACTS)
        if unknown_auxiliary:
            raise ValueError(
                "external_authoring.required_auxiliary_artifacts contains "
                f"unsupported roles: {', '.join(unknown_auxiliary)}"
            )
        normalized_output = str(value.get("normalized_output") or _NORMALIZED_OUTPUT)
        if normalized_output != _NORMALIZED_OUTPUT:
            raise ValueError(
                "external_authoring.normalized_output must be "
                f"{_NORMALIZED_OUTPUT!r}; Texture Agent owns USD/PBR application"
            )
        parameters = value.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("external_authoring.parameters must be an object")
        sensitive_paths = _sensitive_parameter_paths(parameters)
        if sensitive_paths:
            raise ValueError(
                "external_authoring.parameters must not contain credentials: "
                + ", ".join(sensitive_paths)
            )

        tool_name = _optional_nonempty(value.get("tool_name"), "tool_name")
        tool_version = _optional_nonempty(value.get("tool_version"), "tool_version")
        return cls(
            adapter_id=adapter_id,
            workflow=cast(Literal["paint", "designer_graph", "hybrid"], workflow),
            schema_version=schema_version,
            headless_required=True,
            tool_name=tool_name,
            tool_version=tool_version,
            preset_uri=_validate_nonsecret_uri(value.get("preset_uri"), "preset_uri"),
            template_uri=_validate_nonsecret_uri(
                value.get("template_uri"), "template_uri"
            ),
            required_map_channels=required_maps,
            required_auxiliary_artifacts=required_auxiliary,
            normalized_output=normalized_output,
            parameters=parameters,
        )


@dataclass(frozen=True)
class ExternalAuthoringCapabilityReceipt:
    """Sanitized backend receipt returned by the mandatory preflight endpoint."""

    ready: bool
    adapter_id: str
    adapter_version: str
    tool_name: str
    tool_version: str
    headless: bool
    license_status: str
    deployment_mode: str
    environment_digest: str
    supported_workflows: tuple[str, ...]
    supported_map_channels: tuple[str, ...]
    supported_auxiliary_artifacts: tuple[str, ...]
    seed_control: bool
    deterministic_parameters: bool
    normalized_output: str
    schema_version: str = EXTERNAL_AUTHORING_CAPABILITIES_SCHEMA_VERSION
    diagnostics: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any]
    ) -> ExternalAuthoringCapabilityReceipt:
        if not isinstance(payload, dict):
            raise ValueError("External authoring preflight response must be an object")
        schema_version = str(payload.get("schema_version") or "")
        if schema_version != EXTERNAL_AUTHORING_CAPABILITIES_SCHEMA_VERSION:
            raise ValueError(
                "External authoring preflight schema must be "
                f"{EXTERNAL_AUTHORING_CAPABILITIES_SCHEMA_VERSION!r}"
            )
        diagnostics = payload.get("diagnostics") or []
        if not isinstance(diagnostics, list) or not all(
            isinstance(item, dict) for item in diagnostics
        ):
            raise ValueError("External authoring preflight diagnostics must be a list")
        return cls(
            ready=payload.get("ready") is True,
            adapter_id=str(payload.get("adapter_id") or "").strip(),
            adapter_version=str(payload.get("adapter_version") or "").strip(),
            tool_name=str(payload.get("tool_name") or "").strip(),
            tool_version=str(payload.get("tool_version") or "").strip(),
            headless=payload.get("headless") is True,
            license_status=str(payload.get("license_status") or "").strip(),
            deployment_mode=str(payload.get("deployment_mode") or "").strip(),
            environment_digest=str(payload.get("environment_digest") or "").strip(),
            supported_workflows=_string_list(
                payload.get("supported_workflows", []), "supported_workflows"
            ),
            supported_map_channels=_string_list(
                payload.get("supported_map_channels", []),
                "supported_map_channels",
            ),
            supported_auxiliary_artifacts=_string_list(
                payload.get("supported_auxiliary_artifacts", []),
                "supported_auxiliary_artifacts",
            ),
            seed_control=payload.get("seed_control") is True,
            deterministic_parameters=payload.get("deterministic_parameters") is True,
            normalized_output=str(payload.get("normalized_output") or "").strip(),
            schema_version=schema_version,
            diagnostics=tuple(diagnostics),
        )


def external_authoring_spec_payload(spec: ExternalAuthoringSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["required_map_channels"] = list(spec.required_map_channels)
    payload["required_auxiliary_artifacts"] = list(spec.required_auxiliary_artifacts)
    payload["spec_digest"] = external_authoring_spec_digest(spec)
    return payload


def external_authoring_spec_digest(spec: ExternalAuthoringSpec) -> str:
    payload = asdict(spec)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capability_receipt_payload(
    receipt: ExternalAuthoringCapabilityReceipt,
) -> dict[str, Any]:
    payload = asdict(receipt)
    for key in (
        "supported_workflows",
        "supported_map_channels",
        "supported_auxiliary_artifacts",
        "diagnostics",
    ):
        payload[key] = list(payload[key])
    return payload


def evaluate_external_authoring_feasibility(
    spec: ExternalAuthoringSpec,
    receipt: ExternalAuthoringCapabilityReceipt,
) -> dict[str, Any]:
    """Return a durable go/no-go receipt for one adapter capability response."""
    diagnostics: list[dict[str, Any]] = [dict(item) for item in receipt.diagnostics]

    def reject(code: str, message: str, action: str, **details: Any) -> None:
        diagnostics.append(
            _diagnostic(
                code,
                message=message,
                recommended_action=action,
                details=details,
            )
        )

    if not receipt.ready:
        reject(
            "EXTERNAL_AUTHORING_NOT_READY",
            "External texture-authoring backend reported that it is not ready.",
            "Provision an approved headless tool runtime and rerun preflight.",
        )
    if receipt.adapter_id != spec.adapter_id:
        reject(
            "EXTERNAL_AUTHORING_ADAPTER_MISMATCH",
            "External authoring backend resolved a different adapter.",
            "Route to the requested adapter or update the explicit adapter_id.",
            requested=spec.adapter_id,
            actual=receipt.adapter_id,
        )
    if not receipt.adapter_version or not receipt.tool_name or not receipt.tool_version:
        reject(
            "EXTERNAL_AUTHORING_PROVENANCE_INCOMPLETE",
            "External authoring preflight omitted exact adapter or tool versions.",
            "Return exact adapter_version, tool_name, and tool_version values.",
        )
    if spec.tool_name and receipt.tool_name != spec.tool_name:
        reject(
            "EXTERNAL_AUTHORING_TOOL_MISMATCH",
            "External authoring backend resolved a different tool.",
            "Provision the requested tool or update the pinned tool_name.",
            requested=spec.tool_name,
            actual=receipt.tool_name,
        )
    if spec.tool_version and receipt.tool_version != spec.tool_version:
        reject(
            "EXTERNAL_AUTHORING_VERSION_MISMATCH",
            "External authoring backend resolved a different tool version.",
            "Provision the pinned tool version before authoring.",
            requested=spec.tool_version,
            actual=receipt.tool_version,
        )
    if not receipt.headless:
        reject(
            "EXTERNAL_AUTHORING_HEADLESS_UNAVAILABLE",
            "External authoring backend cannot run in approved headless mode.",
            "Provision an approved batch/headless API; interactive automation is unsupported.",
        )
    if receipt.license_status != "valid":
        reject(
            "EXTERNAL_AUTHORING_LICENSE_UNAVAILABLE",
            "External authoring backend did not report a valid license entitlement.",
            "Provision licensing outside the request and return only sanitized status.",
            license_status=receipt.license_status or "missing",
        )
    if receipt.deployment_mode not in _DEPLOYMENT_MODES:
        reject(
            "EXTERNAL_AUTHORING_DEPLOYMENT_UNSUPPORTED",
            "External authoring backend reported an unsupported deployment mode.",
            "Use an approved local, remote, or cloud headless deployment.",
            deployment_mode=receipt.deployment_mode or "missing",
        )
    if not receipt.environment_digest:
        reject(
            "EXTERNAL_AUTHORING_PROVENANCE_INCOMPLETE",
            "External authoring preflight omitted the environment digest.",
            "Return a stable digest for the tool, adapter, plugins, and export profile.",
        )
    if spec.workflow not in receipt.supported_workflows:
        reject(
            "EXTERNAL_AUTHORING_WORKFLOW_UNSUPPORTED",
            "External authoring backend does not support the requested workflow.",
            "Select a supported workflow or adapter.",
            workflow=spec.workflow,
            supported=list(receipt.supported_workflows),
        )
    missing_maps = sorted(
        set(spec.required_map_channels) - set(receipt.supported_map_channels)
    )
    missing_auxiliary = sorted(
        set(spec.required_auxiliary_artifacts)
        - set(receipt.supported_auxiliary_artifacts)
    )
    if missing_maps or missing_auxiliary:
        reject(
            "EXTERNAL_AUTHORING_OUTPUT_UNSUPPORTED",
            "External authoring backend cannot produce every required artifact.",
            "Change the explicit output requirements or select a capable adapter.",
            missing_map_channels=missing_maps,
            missing_auxiliary_artifacts=missing_auxiliary,
        )
    if receipt.normalized_output != spec.normalized_output:
        reject(
            "EXTERNAL_AUTHORING_NORMALIZATION_UNSUPPORTED",
            "External authoring backend cannot return the normalized map contract.",
            "Export Texture Variation map artifacts and let Texture Agent author USD/PBR output.",
            requested=spec.normalized_output,
            actual=receipt.normalized_output,
        )
    if not receipt.seed_control or not receipt.deterministic_parameters:
        reject(
            "EXTERNAL_AUTHORING_REPRODUCIBILITY_UNSUPPORTED",
            "External authoring backend lacks required deterministic controls.",
            "Expose seed control and deterministic parameter resolution before launch.",
            seed_control=receipt.seed_control,
            deterministic_parameters=receipt.deterministic_parameters,
        )

    has_errors = any(item.get("severity") == "error" for item in diagnostics)
    return {
        "schema_version": EXTERNAL_AUTHORING_FEASIBILITY_SCHEMA_VERSION,
        "verdict": "no_go" if has_errors else "go",
        "spec": external_authoring_spec_payload(spec),
        "capabilities": capability_receipt_payload(receipt),
        "diagnostics": diagnostics,
    }


def unavailable_external_authoring_feasibility(
    spec: ExternalAuthoringSpec,
    error: str,
) -> dict[str, Any]:
    """Return a no-go receipt when the mandatory preflight cannot be obtained."""
    return {
        "schema_version": EXTERNAL_AUTHORING_FEASIBILITY_SCHEMA_VERSION,
        "verdict": "no_go",
        "spec": external_authoring_spec_payload(spec),
        "capabilities": None,
        "diagnostics": [
            _diagnostic(
                "EXTERNAL_AUTHORING_PREFLIGHT_UNAVAILABLE",
                message="External authoring capability preflight was unavailable.",
                recommended_action=(
                    "Expose POST /v1/texture-authoring/preflight on an approved "
                    "headless adapter service and retry."
                ),
                details={"error": error[:500]},
            )
        ],
    }


def _file_uri_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            return None
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_external_authoring_result(
    *,
    spec: ExternalAuthoringSpec,
    receipt: ExternalAuthoringCapabilityReceipt,
    metadata: dict[str, Any],
    auxiliary_artifacts: dict[str, Any],
    source_asset_uri: str,
    seed: int | None,
    map_paths: dict[str, str],
) -> list[dict[str, Any]]:
    """Validate provenance and artifact digests for one completed DCC job."""
    diagnostics: list[dict[str, Any]] = []

    def reject(code: str, message: str, action: str, **details: Any) -> None:
        diagnostics.append(
            _diagnostic(
                code,
                message=message,
                recommended_action=action,
                details=details,
            )
        )

    provenance = metadata.get("external_authoring")
    if not isinstance(provenance, dict):
        reject(
            "EXTERNAL_AUTHORING_PROVENANCE_MISSING",
            "External authoring result omitted its provenance receipt.",
            "Return metadata.external_authoring using the v1 result schema.",
        )
        return diagnostics
    if provenance.get("schema_version") != EXTERNAL_AUTHORING_RESULT_SCHEMA_VERSION:
        reject(
            "EXTERNAL_AUTHORING_PROVENANCE_INVALID",
            "External authoring result used an unsupported provenance schema.",
            f"Return {EXTERNAL_AUTHORING_RESULT_SCHEMA_VERSION} metadata.",
            actual=provenance.get("schema_version"),
        )

    expected = {
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
    }
    if seed is None:
        reject(
            "EXTERNAL_AUTHORING_REPRODUCIBILITY_UNSUPPORTED",
            "External authoring request did not resolve an explicit seed.",
            "Set texture.seed or use a texture plan with a deterministic unit seed.",
        )
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            reject(
                "EXTERNAL_AUTHORING_PROVENANCE_MISMATCH",
                "External authoring result provenance does not match preflight/request.",
                "Reject the result and rerun against the exact preflight environment.",
                field=key,
                expected=expected_value,
                actual=provenance.get(key),
            )

    source_path = _file_uri_path(source_asset_uri)
    if source_path is None or not source_path.is_file():
        reject(
            "EXTERNAL_AUTHORING_SOURCE_DIGEST_UNVERIFIABLE",
            "Texture Agent could not verify the authoring source asset digest.",
            "Use an accessible prepared file URI for external authoring.",
        )
    else:
        source_digest = _sha256_file(source_path)
        if provenance.get("source_asset_sha256") != source_digest:
            reject(
                "EXTERNAL_AUTHORING_SOURCE_DIGEST_MISMATCH",
                "External authoring result was not bound to the prepared source asset.",
                "Reject the result and rerun with the exact prepared USD bytes.",
                expected=source_digest,
                actual=provenance.get("source_asset_sha256"),
            )

    output_digests = provenance.get("output_sha256")
    if not isinstance(output_digests, dict):
        reject(
            "EXTERNAL_AUTHORING_OUTPUT_DIGEST_MISSING",
            "External authoring result omitted output map digests.",
            "Return SHA-256 values for every required map channel.",
        )
        output_digests = {}
    for channel in spec.required_map_channels:
        path_text = map_paths.get(channel) or ""
        path = Path(path_text) if path_text and "://" not in path_text else None
        if path is None or not path.is_file():
            reject(
                "EXTERNAL_AUTHORING_OUTPUT_MISSING",
                "External authoring result omitted a required normalized map.",
                "Export every required map channel through the normalized contract.",
                channel=channel,
            )
            continue
        actual_digest = _sha256_file(path)
        if output_digests.get(channel) != actual_digest:
            reject(
                "EXTERNAL_AUTHORING_OUTPUT_DIGEST_MISMATCH",
                "External authoring map bytes do not match backend provenance.",
                "Reject the result and investigate export or transfer corruption.",
                channel=channel,
                expected=output_digests.get(channel),
                actual=actual_digest,
            )

    external_auxiliary = auxiliary_artifacts.get("external_authoring")
    if not isinstance(external_auxiliary, dict):
        external_auxiliary = {}
    missing_auxiliary = [
        role
        for role in spec.required_auxiliary_artifacts
        if not external_auxiliary.get(role)
    ]
    if missing_auxiliary:
        reject(
            "EXTERNAL_AUTHORING_AUXILIARY_MISSING",
            "External authoring result omitted required project/graph evidence.",
            "Return the requested authoring artifacts with relative or service URIs.",
            missing_auxiliary_artifacts=missing_auxiliary,
        )
    return diagnostics
