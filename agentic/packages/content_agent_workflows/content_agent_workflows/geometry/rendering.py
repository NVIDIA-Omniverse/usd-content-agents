# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-render adapter for final Geometry workflow evidence."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from world_understanding.functions.graphics.render_validation import (
    validate_image_artifact,
    validate_render_response,
)
from world_understanding.utils.nvcf_utils import (
    NVCF_INVOCATION_HOST,
    create_nvcf_headers,
    get_base_url,
    get_nvcf_api_key,
)
from world_understanding.validation.usd_rendering import (
    render_usd_visual_evidence,
)

from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256

GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION = (
    "content-agent-workflows.geometry-render-evidence.v3"
)
GEOMETRY_REMOTE_OVRTX_IDENTITY_SCHEMA_VERSION = (
    "content-agent-workflows.geometry-remote-ovrtx-identity.v2"
)
GEOMETRY_RENDER_CANDIDATE_BINDING_SCHEMA_VERSION = (
    "content-agent-workflows.geometry-render-candidate-binding.v1"
)
_REMOTE_OVRTX_HEALTH_CONNECT_TIMEOUT_SECONDS = 10.0
_REMOTE_OVRTX_HEALTH_READ_TIMEOUT_SECONDS = 5.0
_REMOTE_OVRTX_HEALTH_TOTAL_TIMEOUT_SECONDS = 30.0
_REMOTE_OVRTX_HEALTH_MAX_BYTES = 64 * 1024
_REMOTE_OVRTX_HEALTH_CHUNK_BYTES = 4096
GeometryRenderPreset = Literal[
    "hero", "4view", "six_view", "vertical4", "material_review", "turntable"
]
OvRTXRenderMode = Literal["rt1", "rt2", "pt"]
CameraProjection = Literal["perspective", "orthographic"]
GeometryRemoteOvrtxTrustMode = Literal[
    "nvcf_bearer_token",
    "endpoint_bearer_token",
    "loopback",
    "operator_trusted_unauthenticated",
]

_PRESET_VIEWS: dict[str, list[str]] = {
    "hero": ["hero"],
    "4view": ["front", "back", "left", "right"],
    "six_view": ["review_6"],
    "vertical4": ["front", "right", "top", "corner"],
    "material_review": ["hero", "front", "right"],
}
_PRESET_LABELS: dict[str, list[str]] = {
    "hero": ["hero"],
    "4view": ["front", "back", "left", "right"],
    "six_view": ["right", "left", "front", "back", "top", "iso"],
    "vertical4": ["front", "right", "top", "corner"],
    "material_review": ["iso", "front", "right"],
}
_SIGNED_AXIS_VECTORS: dict[str, tuple[int, int, int]] = {
    "+X": (1, 0, 0),
    "-X": (-1, 0, 0),
    "+Y": (0, 1, 0),
    "-Y": (0, -1, 0),
    "+Z": (0, 0, 1),
    "-Z": (0, 0, -1),
}
_FAILURE_SEVERITIES = {"error", "fail", "failure", "fatal"}
_EDGE_ON_EXEMPT_CODES = {"render.low_contrast"}
_RENDERER_FAILURE_STATUSES = {
    "blank_render",
    "cancelled",
    "canceled",
    "empty_response",
    "error",
    "exception",
    "fail",
    "failed",
    "failure",
    "load_error",
    "timed_out",
    "timeout",
    "unavailable",
}
_RENDERER_STATUS_FIELDS = ("status", "render_status", "renderer_status")


def _cross_axis(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _negate_axis(value: tuple[int, int, int]) -> tuple[int, int, int]:
    return (-value[0], -value[1], -value[2])


def _render_direction(*vectors: tuple[int, int, int]) -> str:
    combined = tuple(sum(vector[index] for vector in vectors) for index in range(3))
    if any(component not in {-1, 0, 1} for component in combined):
        raise ValueError(
            f"render direction is not an orthogonal axis combination: {combined}"
        )
    tokens = [
        f"{'+' if component > 0 else '-'}{axis.lower()}"
        for axis, component in zip("XYZ", combined, strict=True)
        if component
    ]
    if not tokens:
        raise ValueError("render direction cannot be zero")
    return "".join(tokens)


def _stage_coordinate_system(
    usd_path: Path,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Read axis semantics retained on the exact USD being rendered."""

    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadNone)
        if stage is None:
            raise RuntimeError("OpenUSD could not open the render source")
        custom_data = dict(stage.GetRootLayer().customLayerData)
        key = next(
            (
                candidate
                for candidate in (
                    "geometryCanonicalCoordinateSystem",
                    "geometrySourceCoordinateSystem",
                )
                if isinstance(custom_data.get(candidate), Mapping)
            ),
            None,
        )
        if key is None:
            return None, None, None
        raw = dict(custom_data[key])
        coordinate_system = {
            "up_axis": str(raw.get("up_axis") or "").upper(),
            "forward_axis": str(raw.get("forward_axis") or "").upper(),
            "handedness": str(raw.get("handedness") or "").lower(),
        }
        up_axis = coordinate_system["up_axis"]
        forward_axis = coordinate_system["forward_axis"]
        handedness = coordinate_system["handedness"]
        if (
            up_axis not in {"X", "Y", "Z"}
            or forward_axis not in _SIGNED_AXIS_VECTORS
            or forward_axis[-1] == up_axis
            or handedness not in {"left", "right"}
        ):
            raise ValueError(f"invalid retained coordinate system: {coordinate_system}")
        observed_up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
        if observed_up_axis != up_axis:
            raise ValueError(
                f"retained up axis {up_axis} does not match stage up axis "
                f"{observed_up_axis}"
            )
        return coordinate_system, key, None
    except Exception as exc:
        return (
            None,
            None,
            {
                "code": "render.coordinate_system_unresolved",
                "severity": "error",
                "message": (
                    "Geometry could not resolve the retained coordinate system, so "
                    "semantic front/back/left/right render labels are not trustworthy."
                ),
                "subject": str(usd_path),
                "details": {
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
            },
        )


def _semantic_view_plan(
    usd_path: Path,
    preset: GeometryRenderPreset,
) -> tuple[list[str], list[str], dict[str, Any] | None, dict[str, Any] | None]:
    labels = list(_PRESET_LABELS[preset])
    coordinate_system, coordinate_source, issue = _stage_coordinate_system(usd_path)
    if coordinate_system is None:
        return list(_PRESET_VIEWS[preset]), labels, None, issue

    up = _SIGNED_AXIS_VECTORS[f"+{coordinate_system['up_axis']}"]
    forward = _SIGNED_AXIS_VECTORS[coordinate_system["forward_axis"]]
    right = (
        _cross_axis(forward, up)
        if coordinate_system["handedness"] == "right"
        else _cross_axis(up, forward)
    )
    semantic_directions = {
        "front": _render_direction(forward),
        "back": _render_direction(_negate_axis(forward)),
        "right": _render_direction(right),
        "left": _render_direction(_negate_axis(right)),
        "top": _render_direction(up),
        "iso": _render_direction(right, forward, up),
        "corner": _render_direction(right, forward, up),
        "hero": _render_direction(right, forward, up),
    }
    directions = [semantic_directions[label] for label in labels]
    return (
        directions,
        labels,
        {
            "status": "resolved",
            "coordinate_source": coordinate_source,
            "coordinate_system": coordinate_system,
            "labels": labels,
            "directions": directions,
        },
        None,
    )


_RENDERER_FAILURE_FIELDS = ("error", "errors", "failure", "failures")


class GeometryRemoteOvrtxHealth(BaseModel):
    """Identity and readiness fields returned by the OVRTX health protocol."""

    model_config = ConfigDict(extra="ignore", strict=True)

    service: Literal["ovrtx-rendering-api"]
    renderer: Literal["ovrtx"]
    status: Literal["healthy"]
    gpu_initialized: Literal[True]


class GeometryRemoteOvrtxIdentityEvidence(BaseModel):
    """Observed OVRTX health response bound to the remote render endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[
        "content-agent-workflows.geometry-remote-ovrtx-identity.v2"
    ] = GEOMETRY_REMOTE_OVRTX_IDENTITY_SCHEMA_VERSION
    endpoint: str
    health_url: str
    observed_at: str
    http_status: Literal[200]
    trust_mode: GeometryRemoteOvrtxTrustMode
    health: GeometryRemoteOvrtxHealth

    @field_validator("endpoint", "health_url")
    @classmethod
    def _require_url(cls, value: str) -> str:
        url = value.strip()
        if not url:
            raise ValueError("URL fields must not be empty")
        return url

    @field_validator("observed_at")
    @classmethod
    def _require_utc_observation_time(cls, value: str) -> str:
        timestamp = value.strip()
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("observed_at must be an ISO 8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("observed_at must include a UTC offset")
        return timestamp


class GeometryRenderImageBinding(BaseModel):
    """One rendered image bound to its bytes and exact source USD bytes."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["view", "presentation"]
    view: str = Field(min_length=1)
    source_usd_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ovrtx_render_mode: OvRTXRenderMode
    ovrtx_num_sensor_updates: int = Field(ge=1)
    active_aov: str | None = None


class GeometryRenderEvidence(BaseModel):
    """Accepted shared render response plus generic image-health results."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["content-agent-workflows.geometry-render-evidence.v3"] = (
        GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION
    )
    status: Literal["pass", "fail", "unavailable"]
    backend: str
    requested_backend: Literal["ovrtx", "remote"] = "ovrtx"
    renderer: Literal["ovrtx"] | None = None
    renderer_endpoint: str | None = None
    renderer_identity_verified: bool = False
    renderer_identity_evidence: dict[str, Any] = Field(default_factory=dict)
    render_redirects_allowed: bool | None = None
    preset: GeometryRenderPreset
    source_usd_path: str = Field(min_length=1)
    source_usd_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_usd_sha256_after_render: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    ovrtx_render_mode: OvRTXRenderMode
    ovrtx_num_sensor_updates: int = Field(ge=1)
    active_aov: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    image_bindings: list[GeometryRenderImageBinding] = Field(default_factory=list)
    presentation_image_path: str | None = None
    shared_render_status: str = "not_evaluated"
    shared_render_issues: list[dict[str, Any]] = Field(default_factory=list)
    response_validation: dict[str, Any] = Field(default_factory=dict)
    image_validation: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_requested_backend(cls, value: Any) -> Any:
        """Keep reports created before ``requested_backend`` backward compatible."""

        if not isinstance(value, Mapping) or "requested_backend" in value:
            return value
        backend = str(value.get("backend") or "").strip().lower()
        if backend not in {"ovrtx", "remote"}:
            backend = "ovrtx"
        return {**value, "requested_backend": backend}

    @model_validator(mode="after")
    def validate_digest_binding_contract(self) -> GeometryRenderEvidence:
        if self.source_usd_sha256 is None:
            issue_codes = {
                str(issue.get("code") or "") for issue in self.shared_render_issues
            }
            if not (
                self.status == "fail"
                and "render.source_usd_unavailable_before_render" in issue_codes
            ):
                raise ValueError(
                    "source_usd_sha256 may be null only for failed pre-render source evidence"
                )
            return self

        if any(
            binding.source_usd_sha256 != self.source_usd_sha256
            for binding in self.image_bindings
        ):
            raise ValueError(
                "image binding source digest does not match render evidence"
            )
        if any(
            binding.ovrtx_render_mode != self.ovrtx_render_mode
            or binding.ovrtx_num_sensor_updates != self.ovrtx_num_sensor_updates
            or binding.active_aov != self.active_aov
            for binding in self.image_bindings
        ):
            raise ValueError(
                "image binding OVRTX settings do not match render evidence"
            )
        if self.status != "pass":
            return self
        transport_backend = self.backend.strip().lower()
        if transport_backend not in {"ovrtx", "remote"}:
            raise ValueError(
                "passing Geometry render evidence requires an OVRTX transport"
            )
        if transport_backend == "remote" and not (
            self.renderer == "ovrtx" and self.renderer_identity_verified
        ):
            raise ValueError(
                "passing remote Geometry render evidence requires verified "
                "OVRTX renderer identity"
            )
        executed = self.metadata.get("executed_ovrtx_settings")
        if (
            not isinstance(executed, dict)
            or executed.get("ovrtx_render_mode") != self.ovrtx_render_mode
            or executed.get("ovrtx_num_sensor_updates") != self.ovrtx_num_sensor_updates
            or not isinstance(executed.get("active_aov"), str)
            or not executed["active_aov"].strip()
            or executed.get("active_aov") != self.active_aov
            or self.metadata.get("executed_ovrtx_settings_verified") is not True
        ):
            raise ValueError(
                "passing Geometry render evidence requires complete executed "
                "OVRTX settings"
            )
        if self.source_usd_sha256_after_render != self.source_usd_sha256:
            raise ValueError(
                "passing render evidence requires a stable source USD digest"
            )
        expected_paths = {*self.image_paths}
        if self.presentation_image_path:
            expected_paths.add(self.presentation_image_path)
        binding_paths = {binding.path for binding in self.image_bindings}
        if (
            not expected_paths
            or len(binding_paths) != len(self.image_bindings)
            or binding_paths != expected_paths
        ):
            raise ValueError(
                "passing render evidence requires one digest binding for every retained image"
            )
        return self


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _identity_issue(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "fail",
        "message": message,
        "details": details,
    }


def _executed_ovrtx_settings(
    render_response: object,
    *,
    transport_backend: Literal["ovrtx", "remote"],
    expected_mode: OvRTXRenderMode,
    expected_sensor_updates: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return consistent executed settings or a fail-closed evidence issue."""

    scope = "remote" if transport_backend == "remote" else "local"
    scope_label = scope.capitalize()
    results = (
        render_response.get("results") if isinstance(render_response, Mapping) else None
    )
    if not isinstance(results, list) or not results:
        return None, _identity_issue(
            code=f"render.{scope}_ovrtx_settings_unverified",
            message=(
                f"{scope_label} OVRTX evidence omitted per-view executed renderer settings."
            ),
            details={"expected_result_count_minimum": 1},
        )

    observed: set[tuple[str, int, str]] = set()
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            return None, _identity_issue(
                code=f"render.{scope}_ovrtx_settings_unverified",
                message=(
                    f"{scope_label} OVRTX evidence contains a result without executed "
                    "renderer settings."
                ),
                details={"result_index": index},
            )
        mode = result.get("ovrtx_render_mode")
        sensor_updates = result.get("ovrtx_num_sensor_updates")
        active_aov = result.get("active_aov")
        if (
            mode not in {"rt1", "rt2", "pt"}
            or not isinstance(sensor_updates, int)
            or isinstance(sensor_updates, bool)
            or sensor_updates < 1
            or not isinstance(active_aov, str)
            or not active_aov.strip()
        ):
            return None, _identity_issue(
                code=f"render.{scope}_ovrtx_settings_unverified",
                message=(
                    f"{scope_label} OVRTX evidence contains incomplete executed renderer "
                    "settings."
                ),
                details={"result_index": index},
            )
        observed.add((mode, sensor_updates, active_aov.strip()))

    if len(observed) != 1:
        return None, _identity_issue(
            code=f"render.{scope}_ovrtx_settings_inconsistent",
            message=(
                f"{scope_label} OVRTX views disagree about their executed renderer settings."
            ),
            details={"distinct_setting_count": len(observed)},
        )
    mode, sensor_updates, active_aov = observed.pop()
    settings = {
        "ovrtx_render_mode": mode,
        "ovrtx_num_sensor_updates": sensor_updates,
        "active_aov": active_aov,
    }
    if mode != expected_mode or sensor_updates != expected_sensor_updates:
        return settings, _identity_issue(
            code=f"render.{scope}_ovrtx_settings_mismatch",
            message=(
                f"{scope_label} OVRTX executed settings do not match the requested "
                "acceptance-render settings."
            ),
            details={
                "expected_mode": expected_mode,
                "observed_mode": mode,
                "expected_sensor_updates": expected_sensor_updates,
                "observed_sensor_updates": sensor_updates,
            },
        )
    return settings, None


def _remote_endpoint_rejection_reasons(endpoint: str) -> list[str]:
    """Return non-secret reason codes for an unsafe remote renderer URL."""

    reasons: list[str] = []
    if endpoint != endpoint.strip():
        reasons.append("surrounding_whitespace")
    try:
        parsed = urlsplit(endpoint)
        if parsed.scheme.lower() not in {"http", "https"}:
            reasons.append("unsupported_scheme")
        if not parsed.hostname:
            reasons.append("missing_hostname")
        if parsed.username is not None or parsed.password is not None:
            reasons.append("userinfo")
        if "?" in endpoint:
            reasons.append("query")
        if "#" in endpoint:
            reasons.append("fragment")
        try:
            parsed.port
        except ValueError:
            reasons.append("invalid_port")
    except ValueError:
        reasons.append("invalid_url")
    return sorted(set(reasons))


def _unsafe_remote_endpoint_issue(reasons: list[str]) -> dict[str, Any]:
    return _identity_issue(
        code="render.ovrtx_endpoint_rejected",
        message=(
            "Remote OVRTX evidence requires an HTTP(S) base URL with a "
            "hostname and without userinfo, query parameters, or fragments."
        ),
        details={
            "transport_backend": "remote",
            "rejection_reasons": reasons,
        },
    )


def _raw_remote_endpoint_value(remote_base_url: str | None) -> str | None:
    """Mirror remote endpoint precedence without normalizing unsafe URLs."""

    if remote_base_url is not None:
        return remote_base_url
    return os.environ.get("RENDER_ENDPOINT") or os.environ.get(
        "NVCF_RENDER_FUNCTION_ID"
    )


def _raw_remote_endpoint_rejection_reasons(endpoint: str) -> list[str]:
    """Reject unsafe explicit URLs before generic endpoint normalization."""

    if "://" in endpoint:
        return _remote_endpoint_rejection_reasons(endpoint)
    reasons: list[str] = []
    if "?" in endpoint:
        reasons.append("query")
    if "#" in endpoint:
        reasons.append("fragment")
    return reasons


def _is_canonical_nvcf_endpoint(endpoint: str) -> bool:
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return False
    hostname = (parsed.hostname or "").lower().rstrip(".")
    suffix = NVCF_INVOCATION_HOST.lower()
    return (
        parsed.scheme.lower() == "https"
        and hostname.endswith(suffix)
        and len(hostname) > len(suffix)
    )


def _select_remote_bearer_credential(
    *,
    remote_endpoint: str,
    endpoint_api_key: str | None,
) -> tuple[
    str,
    Literal["nvcf_bearer_token", "endpoint_bearer_token"] | None,
    dict[str, Any] | None,
]:
    explicit_api_key = (endpoint_api_key or "").strip()
    trust_mode: Literal["nvcf_bearer_token", "endpoint_bearer_token"] | None
    if explicit_api_key:
        api_key = explicit_api_key
        trust_mode = "endpoint_bearer_token"
    elif _is_canonical_nvcf_endpoint(remote_endpoint):
        api_key = get_nvcf_api_key().strip()
        trust_mode = "nvcf_bearer_token" if api_key else None
    else:
        api_key = ""
        trust_mode = None

    if api_key and urlsplit(remote_endpoint).scheme.lower() != "https":
        return (
            "",
            None,
            _identity_issue(
                code="render.ovrtx_bearer_requires_https",
                message=(
                    "Remote OVRTX bearer credentials may be sent only to an "
                    "HTTPS endpoint."
                ),
                details={
                    "transport_backend": "remote",
                    "credential_scope": (
                        "endpoint" if endpoint_api_key else "nvcf_hosted"
                    ),
                },
            ),
        )
    return api_key, trust_mode, None


def _is_loopback_endpoint(endpoint: str) -> bool:
    """Return whether an HTTP endpoint is bound to a loopback host."""

    try:
        hostname = urlsplit(endpoint).hostname
    except ValueError:
        return False
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped and mapped.is_loopback)


def _observe_remote_ovrtx_identity(
    remote_endpoint: str | None,
    *,
    api_key: str,
    bearer_trust_mode: Literal[
        "nvcf_bearer_token",
        "endpoint_bearer_token",
    ]
    | None,
    allow_unauthenticated_identity: bool = False,
) -> tuple[
    Literal["ovrtx"] | None,
    bool,
    dict[str, Any],
    dict[str, Any] | None,
]:
    if remote_endpoint is None:
        return (
            None,
            False,
            {},
            _identity_issue(
                code="render.ovrtx_identity_unverified",
                message=(
                    "Remote rendering has no resolved endpoint for an OVRTX "
                    "identity/readiness observation."
                ),
                details={
                    "transport_backend": "remote",
                    "required_service": "ovrtx-rendering-api",
                    "required_renderer": "ovrtx",
                },
            ),
        )

    endpoint_rejection_reasons = _remote_endpoint_rejection_reasons(remote_endpoint)
    if endpoint_rejection_reasons:
        return (
            None,
            False,
            {},
            _unsafe_remote_endpoint_issue(endpoint_rejection_reasons),
        )

    health_url = f"{remote_endpoint.rstrip('/')}/health"
    observation: dict[str, Any] = {
        "schema_version": GEOMETRY_REMOTE_OVRTX_IDENTITY_SCHEMA_VERSION,
        "endpoint": remote_endpoint,
        "health_url": health_url,
        "observed_at": _utc_now(),
    }
    if bool(api_key) != (bearer_trust_mode is not None):
        return (
            None,
            False,
            observation,
            _identity_issue(
                code="render.ovrtx_credential_scope_invalid",
                message=(
                    "Remote OVRTX bearer credential scope was not resolved "
                    "consistently."
                ),
                details={"transport_backend": "remote"},
            ),
        )
    trust_mode: GeometryRemoteOvrtxTrustMode
    if api_key and bearer_trust_mode is not None:
        trust_mode = bearer_trust_mode
    elif _is_loopback_endpoint(remote_endpoint):
        trust_mode = "loopback"
    elif allow_unauthenticated_identity:
        trust_mode = "operator_trusted_unauthenticated"
    else:
        return (
            None,
            False,
            observation,
            _identity_issue(
                code="render.ovrtx_health_auth_required",
                message=(
                    "Remote OVRTX identity cannot be accepted from a non-loopback "
                    "endpoint without a bearer credential or explicit operator "
                    "trust for an unauthenticated sidecar."
                ),
                details={
                    "transport_backend": "remote",
                    "health_url": health_url,
                    "loopback": False,
                    "operator_trust_enabled": False,
                },
            ),
        )
    observation["trust_mode"] = trust_mode
    response: requests.Response | None = None
    started_at = time.monotonic()
    try:
        headers = create_nvcf_headers(
            api_key,
            int(_REMOTE_OVRTX_HEALTH_TOTAL_TIMEOUT_SECONDS),
        )
        response = requests.get(
            health_url,
            headers=headers,
            timeout=(
                _REMOTE_OVRTX_HEALTH_CONNECT_TIMEOUT_SECONDS,
                _REMOTE_OVRTX_HEALTH_READ_TIMEOUT_SECONDS,
            ),
            allow_redirects=False,
            stream=True,
        )
        observation["http_status"] = response.status_code
        if response.status_code != 200:
            return (
                None,
                False,
                observation,
                _identity_issue(
                    code="render.ovrtx_health_status_invalid",
                    message=(
                        "The remote renderer health observation did not return "
                        f"HTTP 200 (received {response.status_code})."
                    ),
                    details={
                        "transport_backend": "remote",
                        "health_url": health_url,
                        "http_status": response.status_code,
                    },
                ),
            )

        payload_bytes = bytearray()
        for chunk in response.iter_content(chunk_size=_REMOTE_OVRTX_HEALTH_CHUNK_BYTES):
            if (
                time.monotonic() - started_at
                > _REMOTE_OVRTX_HEALTH_TOTAL_TIMEOUT_SECONDS
            ):
                raise TimeoutError(
                    "health response exceeded the total observation deadline"
                )
            if not chunk:
                continue
            payload_bytes.extend(chunk)
            if len(payload_bytes) > _REMOTE_OVRTX_HEALTH_MAX_BYTES:
                raise ValueError(
                    f"health response exceeds {_REMOTE_OVRTX_HEALTH_MAX_BYTES} bytes"
                )
        payload = json.loads(payload_bytes)
        if not isinstance(payload, dict):
            raise ValueError("health response must be a JSON object")
        identity = GeometryRemoteOvrtxIdentityEvidence.model_validate(
            {**observation, "health": payload}
        )
    except requests.RequestException as exc:
        observation["error_type"] = type(exc).__name__
        return (
            None,
            False,
            observation,
            _identity_issue(
                code="render.ovrtx_health_probe_failed",
                message=(
                    "The bounded remote OVRTX identity/readiness health "
                    f"observation failed: {exc}"
                ),
                details={
                    "transport_backend": "remote",
                    "health_url": health_url,
                    "error_type": type(exc).__name__,
                },
            ),
        )
    except TimeoutError as exc:
        observation["error_type"] = type(exc).__name__
        return (
            None,
            False,
            observation,
            _identity_issue(
                code="render.ovrtx_health_probe_timed_out",
                message=(
                    "The remote OVRTX identity/readiness health observation "
                    f"exceeded its total deadline: {exc}"
                ),
                details={
                    "transport_backend": "remote",
                    "health_url": health_url,
                    "error_type": type(exc).__name__,
                    "total_timeout_seconds": (
                        _REMOTE_OVRTX_HEALTH_TOTAL_TIMEOUT_SECONDS
                    ),
                },
            ),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        observation["payload_error_type"] = type(exc).__name__
        return (
            None,
            False,
            observation,
            _identity_issue(
                code="render.ovrtx_health_payload_invalid",
                message=(
                    "The remote renderer health response was not a bounded, "
                    f"valid OVRTX health payload: {exc}"
                ),
                details={
                    "transport_backend": "remote",
                    "health_url": health_url,
                    "http_status": observation.get("http_status"),
                },
            ),
        )
    finally:
        if response is not None:
            response.close()
    proof = identity.model_dump(mode="json")
    return "ovrtx", True, proof, None


def _renderer_identity(
    *,
    backend: Literal["ovrtx", "remote"],
    remote_endpoint: str | None,
    remote_api_key: str,
    remote_bearer_trust_mode: Literal[
        "nvcf_bearer_token",
        "endpoint_bearer_token",
    ]
    | None,
    allow_unauthenticated_remote_identity: bool,
) -> tuple[
    Literal["ovrtx"] | None,
    bool,
    dict[str, Any],
    dict[str, Any] | None,
]:
    if backend == "ovrtx":
        return (
            "ovrtx",
            True,
            {
                "source": "explicit_local_backend",
                "renderer": "ovrtx",
                "ready": True,
            },
            None,
        )

    return _observe_remote_ovrtx_identity(
        remote_endpoint,
        api_key=remote_api_key,
        bearer_trust_mode=remote_bearer_trust_mode,
        allow_unauthenticated_identity=allow_unauthenticated_remote_identity,
    )


class GeometryRenderCandidateBinding(BaseModel):
    """Fail-closed admission receipt for evidence tied to one candidate USD."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "content-agent-workflows.geometry-render-candidate-binding.v1"
    ] = GEOMETRY_RENDER_CANDIDATE_BINDING_SCHEMA_VERSION
    status: Literal["pass", "fail"]
    candidate_usd_path: str = Field(min_length=1)
    candidate_sha256_before_render: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    candidate_sha256_after_render: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    candidate_sha256_after_verification: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evidence_source_usd_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evidence_source_usd_sha256_after_render: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    render_metadata_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evidence_report_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    verified_image_bindings: list[GeometryRenderImageBinding] = Field(
        default_factory=list
    )
    failures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_admission_contract(self) -> GeometryRenderCandidateBinding:
        if self.status == "fail":
            if not self.failures:
                raise ValueError("failed candidate binding requires diagnostics")
            return self
        expected_sha256 = self.candidate_sha256_before_render
        if self.failures:
            raise ValueError("passing candidate binding cannot retain failures")
        if expected_sha256 is None or any(
            digest != expected_sha256
            for digest in (
                self.candidate_sha256_after_render,
                self.candidate_sha256_after_verification,
                self.evidence_source_usd_sha256,
                self.evidence_source_usd_sha256_after_render,
            )
        ):
            raise ValueError("passing candidate binding requires one stable USD digest")
        if self.render_metadata_sha256 is None or self.evidence_report_sha256 is None:
            raise ValueError(
                "passing candidate binding requires retained metadata and report digests"
            )
        if not self.verified_image_bindings or any(
            binding.source_usd_sha256 != expected_sha256
            for binding in self.verified_image_bindings
        ):
            raise ValueError(
                "passing candidate binding requires images bound to the candidate digest"
            )
        return self


def verify_geometry_render_candidate_evidence(
    evidence: GeometryRenderEvidence,
    *,
    candidate_usd_path: Path,
    candidate_sha256_before_render: str | None,
    candidate_sha256_after_render: str | None,
) -> GeometryRenderCandidateBinding:
    """Bind an accepted OVRTX report and its images to the current candidate."""

    candidate_path = candidate_usd_path.expanduser().resolve()
    failures: list[str] = []
    verified_bindings: list[GeometryRenderImageBinding] = []

    if evidence.status != "pass":
        failures.append("required OVRTX render evidence did not pass")
    if evidence.shared_render_status != "completed":
        failures.append("required OVRTX renderer did not report completed status")
    if str(evidence.backend).lower() != "ovrtx":
        failures.append("required changed-geometry evidence was not produced by OVRTX")

    try:
        evidence_source_path = Path(evidence.source_usd_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        evidence_source_path = None
    if evidence_source_path != candidate_path:
        failures.append(
            "render evidence source path does not match the required candidate USD"
        )

    expected_sha256 = candidate_sha256_before_render
    if expected_sha256 is None:
        failures.append("candidate USD was unavailable for the pre-render digest check")
    else:
        if candidate_sha256_after_render != expected_sha256:
            failures.append("candidate USD changed or disappeared during rendering")
        if evidence.source_usd_sha256 != expected_sha256:
            failures.append(
                "render evidence source digest does not match the required candidate USD"
            )
        if evidence.source_usd_sha256_after_render != expected_sha256:
            failures.append(
                "render evidence did not retain the candidate digest after rendering"
            )

    metadata = evidence.metadata
    render_metadata_sha256: str | None = None
    try:
        metadata_bytes = json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        render_metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    except (TypeError, ValueError):
        failures.append("OVRTX render metadata is not canonical JSON")
    if not metadata:
        failures.append("required OVRTX render metadata is missing")
    elif str(metadata.get("backend") or "").lower() != "ovrtx":
        failures.append("render metadata does not identify the OVRTX backend")

    stage_preparation = metadata.get("stage_preparation")
    matching_stage_entries: list[dict[str, Any]] = []
    if isinstance(stage_preparation, list):
        for item in stage_preparation:
            if not isinstance(item, dict):
                continue
            try:
                item_path = Path(str(item.get("usd_path") or "")).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if item_path == candidate_path:
                matching_stage_entries.append(item)
    if len(matching_stage_entries) != 1:
        failures.append(
            "render metadata does not contain exactly one preparation record for the candidate USD"
        )
    elif matching_stage_entries[0].get("usd_sha256") != expected_sha256:
        failures.append(
            "render metadata preparation digest does not match the candidate USD"
        )

    view_bindings = [
        binding for binding in evidence.image_bindings if binding.role == "view"
    ]
    response_cameras = metadata.get("response_cameras")
    if not isinstance(response_cameras, list) or [
        str(camera) for camera in response_cameras
    ] != [binding.view for binding in view_bindings]:
        failures.append(
            "render metadata camera records do not match the retained view bindings"
        )
    image_count = metadata.get("image_count")
    if type(image_count) is not int or image_count != len(evidence.image_paths):
        failures.append(
            "render metadata image count does not match the retained render views"
        )

    expected_image_paths = set(evidence.image_paths)
    if evidence.presentation_image_path:
        expected_image_paths.add(evidence.presentation_image_path)
    bound_image_paths = {binding.path for binding in evidence.image_bindings}
    if not expected_image_paths or bound_image_paths != expected_image_paths:
        failures.append(
            "required render evidence is missing an image binding or contains a stale binding"
        )
    for binding in evidence.image_bindings:
        binding_failures = False
        if binding.source_usd_sha256 != expected_sha256:
            failures.append(
                f"render image {binding.path!r} is bound to different geometry"
            )
            binding_failures = True
        try:
            current_image_sha256 = file_sha256(Path(binding.path))
        except OSError:
            failures.append(
                f"render image {binding.path!r} is unavailable during candidate admission"
            )
            binding_failures = True
        else:
            if current_image_sha256 != binding.sha256:
                failures.append(
                    f"render image {binding.path!r} changed after evidence collection"
                )
                binding_failures = True
        if not binding_failures:
            verified_bindings.append(binding)

    evidence_report_sha256: str | None = None
    if evidence.report_path is None:
        failures.append("required OVRTX evidence report path is missing")
    else:
        try:
            report_bytes = Path(evidence.report_path).read_bytes()
            evidence_report_sha256 = hashlib.sha256(report_bytes).hexdigest()
            retained_evidence = GeometryRenderEvidence.model_validate_json(report_bytes)
        except (OSError, ValueError):
            failures.append("required OVRTX evidence report is unavailable or invalid")
        else:
            if retained_evidence.model_dump(mode="json") != evidence.model_dump(
                mode="json"
            ):
                failures.append(
                    "retained OVRTX evidence report does not match the admitted evidence"
                )

    try:
        candidate_sha256_after_verification = file_sha256(candidate_path)
    except OSError:
        candidate_sha256_after_verification = None
    if candidate_sha256_after_verification != expected_sha256:
        failures.append(
            "candidate USD changed or disappeared during evidence admission"
        )

    return GeometryRenderCandidateBinding(
        status="fail" if failures else "pass",
        candidate_usd_path=str(candidate_path),
        candidate_sha256_before_render=candidate_sha256_before_render,
        candidate_sha256_after_render=candidate_sha256_after_render,
        candidate_sha256_after_verification=candidate_sha256_after_verification,
        evidence_source_usd_sha256=evidence.source_usd_sha256,
        evidence_source_usd_sha256_after_render=(
            evidence.source_usd_sha256_after_render
        ),
        render_metadata_sha256=render_metadata_sha256,
        evidence_report_sha256=evidence_report_sha256,
        verified_image_bindings=verified_bindings,
        failures=failures,
    )


def _has_failure_issue(issues: list[dict[str, Any]]) -> bool:
    return any(
        str(issue.get("severity") or "error").lower() in _FAILURE_SEVERITIES
        for issue in issues
    )


def _renderer_failure_metadata_findings(
    *,
    render_response: Any = None,
    renderer_metadata: Any = None,
    response_validation: Any = None,
    legacy_metadata: Any = None,
) -> list[dict[str, str]]:
    """Return explicit renderer failure markers retained in response metadata.

    A renderer can return a decodable image while also reporting a terminal
    status or error. Image health alone must not turn that contradictory result
    into accepted Geometry evidence.
    """

    findings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(field: str, value: Any) -> None:
        rendered_value = str(value).strip()
        key = (field, rendered_value)
        if key in seen:
            return
        seen.add(key)
        findings.append({"field": field, "value": rendered_value[:500]})

    def has_failure_value(value: Any) -> bool:
        if value is None or value is False:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, Mapping | Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            return bool(value)
        return bool(value)

    def inspect(payload: Any, source: str) -> None:
        if not isinstance(payload, Mapping):
            return
        for field in _RENDERER_STATUS_FIELDS:
            raw_status = payload.get(field)
            normalized = str(raw_status or "").strip().lower().replace("-", "_")
            if normalized in _RENDERER_FAILURE_STATUSES:
                add(f"{source}.{field}", raw_status)
        for field in _RENDERER_FAILURE_FIELDS:
            value = payload.get(field)
            if not has_failure_value(value):
                continue
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if has_failure_value(item):
                        add(f"{source}.{field}.{key}", item)
            elif isinstance(value, Sequence) and not isinstance(
                value, str | bytes | bytearray
            ):
                for index, item in enumerate(value):
                    if has_failure_value(item):
                        add(f"{source}.{field}[{index}]", item)
            else:
                add(f"{source}.{field}", value)

    inspect(renderer_metadata, "metadata")
    inspect(legacy_metadata, "report")
    inspect(render_response, "render_response")
    if isinstance(render_response, Mapping):
        raw_results = render_response.get("results")
        if isinstance(raw_results, Sequence) and not isinstance(
            raw_results, str | bytes | bytearray
        ):
            for index, entry in enumerate(raw_results):
                inspect(entry, f"render_response.results[{index}]")
    if isinstance(response_validation, Mapping):
        inspect(response_validation.get("metadata"), "response_validation.metadata")
    return findings


def _validate_image_paths(
    image_paths: list[Path], *, backend: str
) -> list[dict[str, Any]]:
    return [
        validate_image_artifact(
            path,
            backend=backend,
            min_width=256,
            min_height=256,
            detect_error_material_artifacts=True,
        ).to_dict()
        for path in image_paths
    ]


def _collect_image_digests(
    image_paths: list[Path],
    *,
    phase: str,
    issues: list[dict[str, Any]],
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in image_paths:
        try:
            digests[str(path)] = file_sha256(path)
        except OSError as exc:
            issues.append(
                {
                    "code": "render.image_unavailable_for_digest",
                    "severity": "error",
                    "message": f"Rendered image became unavailable during {phase}: {exc}",
                    "subject": str(path),
                    "details": {
                        "error_type": type(exc).__name__,
                        "phase": phase,
                    },
                }
            )
    return digests


def _record_image_digest_changes(
    before: dict[str, str],
    after: dict[str, str],
    *,
    phase: str,
    issues: list[dict[str, Any]],
) -> set[str]:
    changed: set[str] = set()
    for path in sorted(before.keys() & after.keys()):
        if before[path] == after[path]:
            continue
        changed.add(path)
        issues.append(
            {
                "code": "render.image_changed",
                "severity": "error",
                "message": (
                    "Rendered image bytes changed while evidence was being "
                    f"collected during {phase}."
                ),
                "subject": path,
                "details": {
                    "phase": phase,
                    "sha256_before": before[path],
                    "sha256_after": after[path],
                },
            }
        )
    return changed


def _is_edge_on_thin_geometry(path: Path, validation: dict[str, Any]) -> bool:
    """Recognize auditable high-contrast thin geometry without accepting tiny blobs."""

    issues = validation.get("issues") or []
    codes = {str(issue.get("code") or "") for issue in issues}
    if not codes or not codes.issubset(_EDGE_ON_EXEMPT_CODES):
        return False
    metrics = validation.get("metrics") or {}
    occupancy = float(metrics.get("nonblack_pixel_ratio") or 0.0)
    dynamic_range = float(metrics.get("luma_dynamic_range") or 0.0)
    if not 0.001 <= occupancy <= 0.02 or dynamic_range < 20.0:
        return False
    try:
        with Image.open(path) as opened:
            grayscale = opened.convert("L")
            mask = grayscale.point(lambda value: 255 if value > 4 else 0)
            bbox = mask.getbbox()
            if bbox is None:
                return False
            span_x = bbox[2] - bbox[0]
            span_y = bbox[3] - bbox[1]
            major_span = max(span_x / opened.width, span_y / opened.height)
            minor_span = min(span_x, span_y)
    except Exception:
        return False
    # A paired default/variant stage can place two long thin members diagonally.
    # Each remains legible while their combined foreground spans only about a
    # quarter of the image. Smaller marks are still rejected as tiny blobs.
    return major_span >= 0.25 and minor_span >= 1


def _compose_grid(
    image_paths: list[Path],
    labels: list[str],
    output_path: Path,
    *,
    columns: int,
) -> Path:
    images: list[Image.Image] = []
    try:
        for path in image_paths:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        rows = (len(images) + columns - 1) // columns
        canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), "black")
        draw = ImageDraw.Draw(canvas)
        for index, (image, label) in enumerate(zip(images, labels, strict=True)):
            column = index % columns
            row = index // columns
            x = column * cell_width + (cell_width - image.width) // 2
            y = row * cell_height + (cell_height - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text(
                (column * cell_width + 12, row * cell_height + 10),
                label,
                fill="white",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    finally:
        for image in images:
            image.close()
    return output_path


def render_geometry_evidence(
    *,
    usd_path: Path,
    output_dir: Path,
    preset: GeometryRenderPreset,
    backend: Literal["ovrtx", "remote"] = "ovrtx",
    remote_base_url: str | None = None,
    remote_api_key: str | None = None,
    remote_allow_unauthenticated_identity: bool = False,
    image_width: int = 1024,
    image_height: int = 1024,
    focus_prim_path: str | None = None,
    isolate_prim_paths: Sequence[str] = (),
    frame_isolated_context: bool = True,
    camera_margin: float = 1.2,
    camera_projection: CameraProjection = "perspective",
    view_directions: Sequence[str] | None = None,
    studio_lighting: bool | None = None,
    studio_dome: bool = True,
    studio_dome_intensity: float | None = None,
    studio_reflection_safe: bool | None = None,
    studio_hdri_intensity: float = 600.0,
    studio_reflection_intensity: float = 100.0,
    ovrtx_mode: OvRTXRenderMode | None = None,
    ovrtx_num_sensor_updates: int | None = None,
) -> GeometryRenderEvidence:
    """Render accepted evidence through shared infrastructure."""

    source_path = usd_path.expanduser().resolve()
    render_dir = output_dir.resolve()
    render_dir.mkdir(parents=True, exist_ok=True)
    report_path = render_dir / f"geometry_render_evidence_{preset}.json"
    effective_ovrtx_mode: OvRTXRenderMode = (
        ovrtx_mode if ovrtx_mode is not None else "pt"
    )
    effective_ovrtx_num_sensor_updates = (
        ovrtx_num_sensor_updates if ovrtx_num_sensor_updates is not None else 64
    )
    if effective_ovrtx_num_sensor_updates < 1:
        raise ValueError("ovrtx_num_sensor_updates must be positive")
    try:
        source_usd_sha256 = file_sha256(source_path)
    except OSError as exc:
        result = GeometryRenderEvidence(
            status="fail",
            backend=backend,
            requested_backend=backend,
            preset=preset,
            source_usd_path=str(source_path),
            source_usd_sha256=None,
            source_usd_sha256_after_render=None,
            ovrtx_render_mode=effective_ovrtx_mode,
            ovrtx_num_sensor_updates=effective_ovrtx_num_sensor_updates,
            shared_render_status="not_evaluated",
            shared_render_issues=[
                {
                    "code": "render.source_usd_unavailable_before_render",
                    "severity": "error",
                    "message": f"The source USD is unavailable for rendering: {exc}",
                    "subject": str(source_path),
                    "details": {"error_type": type(exc).__name__},
                }
            ],
            report_path=str(report_path),
        )
        atomic_write_json(report_path, result)
        return result
    resolved_remote_endpoint: str | None = None
    resolved_remote_api_key = ""
    remote_bearer_trust_mode: (
        Literal[
            "nvcf_bearer_token",
            "endpoint_bearer_token",
        ]
        | None
    ) = None
    remote_endpoint_issue: dict[str, Any] | None = None
    if backend == "remote":
        raw_remote_endpoint = _raw_remote_endpoint_value(remote_base_url)
        raw_rejection_reasons = (
            _raw_remote_endpoint_rejection_reasons(raw_remote_endpoint)
            if raw_remote_endpoint is not None
            else []
        )
        if raw_rejection_reasons:
            remote_endpoint_issue = _unsafe_remote_endpoint_issue(raw_rejection_reasons)
        else:
            try:
                candidate_remote_endpoint = get_base_url(
                    raw_remote_endpoint,
                    "RENDER_ENDPOINT",
                    "NVCF_RENDER_FUNCTION_ID",
                ).rstrip("/")
                endpoint_rejection_reasons = _remote_endpoint_rejection_reasons(
                    candidate_remote_endpoint
                )
                if endpoint_rejection_reasons:
                    remote_endpoint_issue = _unsafe_remote_endpoint_issue(
                        endpoint_rejection_reasons
                    )
                else:
                    resolved_remote_endpoint = candidate_remote_endpoint
            except ValueError:
                # The shared renderer reports the unavailable endpoint with its
                # normal diagnostics below.
                pass
        if resolved_remote_endpoint is not None:
            (
                resolved_remote_api_key,
                remote_bearer_trust_mode,
                credential_issue,
            ) = _select_remote_bearer_credential(
                remote_endpoint=resolved_remote_endpoint,
                endpoint_api_key=remote_api_key,
            )
            if credential_issue is not None:
                remote_endpoint_issue = credential_issue
    (
        renderer,
        renderer_identity_verified,
        renderer_identity_evidence,
        renderer_identity_issue,
    ) = (
        (None, False, {}, remote_endpoint_issue)
        if remote_endpoint_issue is not None
        else _renderer_identity(
            backend=backend,
            remote_endpoint=resolved_remote_endpoint,
            remote_api_key=resolved_remote_api_key,
            remote_bearer_trust_mode=remote_bearer_trust_mode,
            allow_unauthenticated_remote_identity=(
                remote_allow_unauthenticated_identity
            ),
        )
    )
    render_redirects_allowed = False if backend == "remote" else None
    if preset == "turntable":
        turntable_issues = [
            {
                "code": "render.turntable_requires_scene_session",
                "severity": "warning",
                "message": (
                    "Batch validation rendering does not provide accepted "
                    "turntable evidence; use usd-cli render-frames."
                ),
            }
        ]
        if renderer_identity_issue is not None:
            turntable_issues.append(renderer_identity_issue)
        result = GeometryRenderEvidence(
            status="unavailable",
            backend=backend,
            requested_backend=backend,
            renderer=renderer,
            renderer_endpoint=resolved_remote_endpoint,
            renderer_identity_verified=renderer_identity_verified,
            renderer_identity_evidence=renderer_identity_evidence,
            render_redirects_allowed=render_redirects_allowed,
            preset=preset,
            source_usd_path=str(source_path),
            source_usd_sha256=source_usd_sha256,
            source_usd_sha256_after_render=source_usd_sha256,
            ovrtx_render_mode=effective_ovrtx_mode,
            ovrtx_num_sensor_updates=effective_ovrtx_num_sensor_updates,
            shared_render_status="unavailable",
            shared_render_issues=turntable_issues,
            metadata={
                "requested_backend": backend,
                "transport_backend": backend,
                "renderer": renderer,
                "renderer_identity_verified": renderer_identity_verified,
            },
            report_path=str(report_path),
        )
        atomic_write_json(report_path, result)
        return result

    semantic_view_plan: dict[str, Any] | None = None
    coordinate_system_issue: dict[str, Any] | None = None
    if view_directions:
        requested_views = list(view_directions)
        labels = list(_PRESET_LABELS[preset])
    else:
        (
            requested_views,
            labels,
            semantic_view_plan,
            coordinate_system_issue,
        ) = _semantic_view_plan(source_path, preset)
    use_studio_lighting = (
        preset == "material_review" if studio_lighting is None else studio_lighting
    )
    use_reflection_safe = (
        backend == "ovrtx" if studio_reflection_safe is None else studio_reflection_safe
    )
    resolved_dome_intensity = (
        (20.0 if use_reflection_safe else 350.0)
        if studio_dome_intensity is None
        else studio_dome_intensity
    )
    policy: dict[str, Any] = {
        "render_backend": backend,
        "runtime_render_views": requested_views,
        "render_image_width": image_width,
        "render_image_height": image_height,
        "render_studio_lighting": use_studio_lighting,
        "render_camera_projection": camera_projection,
    }
    if use_studio_lighting:
        policy["render_studio_dome"] = studio_dome
        policy["render_studio_dome_intensity"] = resolved_dome_intensity
        policy["render_studio_reflection_safe"] = use_reflection_safe
        if use_reflection_safe:
            policy["render_studio_hdri_intensity"] = studio_hdri_intensity
            policy["render_studio_reflection_intensity"] = studio_reflection_intensity
    if focus_prim_path:
        policy["render_focus_prim_path"] = focus_prim_path
    if isolate_prim_paths:
        policy["render_isolate_prim_paths"] = list(isolate_prim_paths)
        policy["render_frame_isolated_context"] = frame_isolated_context
    if focus_prim_path or isolate_prim_paths:
        policy["render_camera_margin"] = camera_margin
    policy["render_ovrtx_mode"] = effective_ovrtx_mode
    policy["render_ovrtx_num_sensor_updates"] = effective_ovrtx_num_sensor_updates
    if resolved_remote_endpoint is not None:
        # Resolve once and pass the same endpoint into rendering so the
        # retained identity proof cannot be checked against one endpoint while
        # the shared helper renders through another.
        policy["render_base_url"] = resolved_remote_endpoint
    if backend == "remote":
        # Pass an explicit empty value for trusted unauthenticated sidecars so
        # the shared backend cannot fall back to NGC_API_KEY.
        policy["render_api_key"] = resolved_remote_api_key
        policy["render_allow_redirects"] = False
    shared = (
        {
            "status": "failed",
            "backend": "remote",
            "image_paths": [],
            "issues": [],
            "metadata": {"endpoint_rejected": True},
        }
        if remote_endpoint_issue is not None
        else render_usd_visual_evidence(
            usd_paths=[source_path],
            working_dir=render_dir,
            policy=policy,
        )
    )
    shared_status = str(shared.get("status") or "failed").strip().lower()
    actual_backend = str(shared.get("backend") or "").strip().lower()
    backend_matches_request = actual_backend == backend
    if not backend_matches_request:
        renderer = None
        renderer_identity_verified = False
    image_paths = [Path(path).resolve() for path in shared.get("image_paths") or []]
    validation_backend = actual_backend or "unknown"
    shared_issues = [
        dict(issue) for issue in shared.get("issues") or [] if isinstance(issue, dict)
    ]
    if coordinate_system_issue is not None:
        shared_issues.append(coordinate_system_issue)
    image_digests_at_render_return = _collect_image_digests(
        image_paths,
        phase="render return",
        issues=shared_issues,
    )
    if shared_status.lower() == "completed" and not image_paths:
        shared_issues.append(
            _identity_issue(
                code="render.acceptance_images_missing",
                message=(
                    "The shared renderer reported completion without any "
                    "rendered image evidence."
                ),
                details={
                    "requested_backend": backend,
                    "actual_backend": actual_backend or None,
                },
            )
        )
    if renderer_identity_issue is not None:
        shared_issues.append(renderer_identity_issue)
    if not backend_matches_request:
        shared_issues.append(
            _identity_issue(
                code="render.backend_provenance_mismatch",
                message=(
                    "The shared renderer's actual transport backend did not "
                    "match the requested backend."
                ),
                details={
                    "requested_backend": backend,
                    "actual_backend": actual_backend or None,
                },
            )
        )
    if renderer != "ovrtx" or not renderer_identity_verified:
        shared_issues.append(
            _identity_issue(
                code="render.ovrtx_provenance_unverified",
                message=(
                    "Geometry final evidence requires verified OVRTX renderer identity."
                ),
                details={
                    "requested_backend": backend,
                    "actual_backend": actual_backend or None,
                    "renderer": renderer,
                    "renderer_identity_verified": renderer_identity_verified,
                },
            )
        )
    response_validation: dict[str, Any] = {}
    render_response = shared.get("render_response")
    executed_ovrtx_settings: dict[str, Any] | None = None
    executed_ovrtx_settings_verified = False
    evidence_ovrtx_mode = effective_ovrtx_mode
    evidence_ovrtx_num_sensor_updates = effective_ovrtx_num_sensor_updates
    evidence_active_aov: str | None = None
    if shared_status == "completed":
        executed_ovrtx_settings, settings_issue = _executed_ovrtx_settings(
            render_response,
            transport_backend=backend,
            expected_mode=effective_ovrtx_mode,
            expected_sensor_updates=effective_ovrtx_num_sensor_updates,
        )
        if executed_ovrtx_settings is not None:
            evidence_ovrtx_mode = executed_ovrtx_settings["ovrtx_render_mode"]
            evidence_ovrtx_num_sensor_updates = executed_ovrtx_settings[
                "ovrtx_num_sensor_updates"
            ]
            evidence_active_aov = executed_ovrtx_settings["active_aov"]
        if settings_issue is not None:
            shared_issues.append(settings_issue)
        else:
            executed_ovrtx_settings_verified = True
    response_cameras = list(
        (shared.get("metadata") or {}).get("response_cameras") or []
    )
    evidence_labels = (
        response_cameras
        if len(response_cameras) == len(image_paths)
        else labels
        if len(labels) == len(image_paths)
        else [f"view_{index + 1}" for index in range(len(image_paths))]
    )
    if shared_status != "unavailable" and len(image_paths) != len(labels):
        shared_issues.append(
            {
                "code": "render.view_count_mismatch",
                "severity": "error",
                "message": (
                    f"Expected {len(labels)} render views for {preset!r}, received "
                    f"{len(image_paths)}. Individual artifacts are retained for diagnosis."
                ),
                "subject": str(render_dir),
                "details": {
                    "expected_count": len(labels),
                    "observed_count": len(image_paths),
                },
            }
        )
    if render_response is not None:
        expected_cameras = list(
            (shared.get("metadata") or {}).get("response_cameras") or labels
        )
        response_result = validate_render_response(
            render_response,
            expected_cameras=expected_cameras,
            backend=validation_backend,
        )
        response_validation = response_result.to_dict()
        shared_issues.extend(issue.to_dict() for issue in response_result.issues)
    renderer_failure_findings = _renderer_failure_metadata_findings(
        render_response=render_response,
        renderer_metadata=shared.get("metadata"),
        response_validation=response_validation,
    )
    if renderer_failure_findings:
        shared_issues.append(
            {
                "code": "render.renderer_reported_failure",
                "severity": "fail",
                "message": (
                    "The renderer returned image evidence while also retaining "
                    "a terminal status, error, or failure marker."
                ),
                "details": {"findings": renderer_failure_findings},
            }
        )
    presentation_path: Path | None = None
    presentation_sha256_after_composition: str | None = None
    if len(image_paths) > 1 and len(image_paths) == len(labels):
        columns = 3 if preset in {"six_view", "material_review"} else 2
        try:
            composed_path = _compose_grid(
                image_paths,
                labels,
                render_dir / f"geometry_{preset}.png",
                columns=columns,
            )
            presentation_sha256_after_composition = file_sha256(composed_path)
            presentation_path = composed_path
        except Exception as exc:
            shared_issues.append(
                {
                    "code": "render.presentation_grid_failed",
                    "severity": "warn",
                    "message": (
                        "Presentation-grid composition failed; individually "
                        f"validated render views remain available: {exc}"
                    ),
                    "subject": str(render_dir),
                    "details": {"error_type": type(exc).__name__},
                }
            )
    image_digests_before_validation = _collect_image_digests(
        image_paths,
        phase="pre-validation",
        issues=shared_issues,
    )
    _record_image_digest_changes(
        image_digests_at_render_return,
        image_digests_before_validation,
        phase="presentation composition",
        issues=shared_issues,
    )
    image_results = _validate_image_paths(image_paths, backend=validation_backend)
    image_digests_after_validation = _collect_image_digests(
        image_paths,
        phase="post-validation",
        issues=shared_issues,
    )
    images_changed_during_validation = _record_image_digest_changes(
        image_digests_before_validation,
        image_digests_after_validation,
        phase="image validation",
        issues=shared_issues,
    )
    edge_on_paths: set[str] = set()
    for path, label, validation in zip(
        image_paths, evidence_labels, image_results, strict=True
    ):
        if _is_edge_on_thin_geometry(path, validation):
            edge_on_paths.add(str(path))
            shared_issues.append(
                {
                    "code": "render.edge_on_thin_geometry",
                    "severity": "warning",
                    "message": (
                        f"Accepted high-contrast edge-on evidence for {label!r}; the original "
                        "per-image validator findings remain preserved."
                    ),
                    "subject": str(path),
                    "details": {
                        "view": label,
                        "original_issue_codes": [
                            issue.get("code")
                            for issue in validation.get("issues") or []
                        ],
                    },
                }
            )
    image_issues = [
        issue
        for validation in image_results
        for issue in validation.get("issues") or []
        if not (
            str(issue.get("subject") or "") in edge_on_paths
            and str(issue.get("code") or "") in _EDGE_ON_EXEMPT_CODES
        )
    ]
    image_bindings: list[GeometryRenderImageBinding] = []
    for path, view in zip(image_paths, evidence_labels, strict=True):
        try:
            if str(path) in images_changed_during_validation:
                continue
            current_sha256 = file_sha256(path)
            validated_sha256 = image_digests_after_validation.get(str(path))
            if validated_sha256 is None or current_sha256 != validated_sha256:
                shared_issues.append(
                    {
                        "code": "render.image_changed_after_validation",
                        "severity": "error",
                        "message": (
                            "Rendered image bytes changed after validation; no "
                            "accepted binding was created."
                        ),
                        "subject": str(path),
                        "details": {
                            "view": view,
                            "validated_sha256": validated_sha256,
                            "current_sha256": current_sha256,
                        },
                    }
                )
                continue
            image_bindings.append(
                GeometryRenderImageBinding(
                    path=str(path),
                    sha256=validated_sha256,
                    role="view",
                    view=view,
                    source_usd_sha256=source_usd_sha256,
                    ovrtx_render_mode=evidence_ovrtx_mode,
                    ovrtx_num_sensor_updates=evidence_ovrtx_num_sensor_updates,
                    active_aov=evidence_active_aov,
                )
            )
        except OSError as exc:
            shared_issues.append(
                {
                    "code": "render.image_unavailable_for_binding",
                    "severity": "error",
                    "message": f"Rendered image could not be digest-bound: {exc}",
                    "subject": str(path),
                    "details": {"error_type": type(exc).__name__, "view": view},
                }
            )
    if presentation_path is not None:
        try:
            current_sha256 = file_sha256(presentation_path)
            if current_sha256 != presentation_sha256_after_composition:
                shared_issues.append(
                    {
                        "code": "render.presentation_changed_after_composition",
                        "severity": "error",
                        "message": (
                            "Presentation image bytes changed after composition; "
                            "no accepted binding was created."
                        ),
                        "subject": str(presentation_path),
                        "details": {
                            "composed_sha256": presentation_sha256_after_composition,
                            "current_sha256": current_sha256,
                        },
                    }
                )
            else:
                image_bindings.append(
                    GeometryRenderImageBinding(
                        path=str(presentation_path),
                        sha256=current_sha256,
                        role="presentation",
                        view=preset,
                        source_usd_sha256=source_usd_sha256,
                        ovrtx_render_mode=evidence_ovrtx_mode,
                        ovrtx_num_sensor_updates=evidence_ovrtx_num_sensor_updates,
                        active_aov=evidence_active_aov,
                    )
                )
        except OSError as exc:
            shared_issues.append(
                {
                    "code": "render.presentation_unavailable_for_binding",
                    "severity": "error",
                    "message": (f"Presentation image could not be digest-bound: {exc}"),
                    "subject": str(presentation_path),
                    "details": {"error_type": type(exc).__name__},
                }
            )
    try:
        source_usd_sha256_after_render = file_sha256(source_path)
    except OSError as exc:
        source_usd_sha256_after_render = None
        shared_issues.append(
            {
                "code": "render.source_usd_unavailable_after_render",
                "severity": "error",
                "message": (
                    "The source USD became unavailable while render evidence was "
                    f"being collected: {exc}"
                ),
                "subject": str(source_path),
                "details": {"error_type": type(exc).__name__},
            }
        )
    if (
        source_usd_sha256_after_render is not None
        and source_usd_sha256_after_render != source_usd_sha256
    ):
        shared_issues.append(
            {
                "code": "render.source_usd_changed",
                "severity": "error",
                "message": (
                    "The source USD changed while render evidence was being "
                    "collected; the resulting images cannot be accepted."
                ),
                "subject": str(source_path),
                "details": {
                    "source_usd_sha256_before_render": source_usd_sha256,
                    "source_usd_sha256_after_render": source_usd_sha256_after_render,
                },
            }
        )
    status: Literal["pass", "fail", "unavailable"]
    if _has_failure_issue(shared_issues) or _has_failure_issue(image_issues):
        status = "fail"
    elif shared_status == "unavailable":
        status = "unavailable"
    elif shared_status != "completed":
        status = "fail"
    else:
        status = "pass"
    result = GeometryRenderEvidence(
        status=status,
        backend=actual_backend or "unknown",
        requested_backend=backend,
        renderer=renderer,
        renderer_endpoint=resolved_remote_endpoint,
        renderer_identity_verified=renderer_identity_verified,
        renderer_identity_evidence=renderer_identity_evidence,
        render_redirects_allowed=render_redirects_allowed,
        preset=preset,
        source_usd_path=str(source_path),
        source_usd_sha256=source_usd_sha256,
        source_usd_sha256_after_render=source_usd_sha256_after_render,
        ovrtx_render_mode=evidence_ovrtx_mode,
        ovrtx_num_sensor_updates=evidence_ovrtx_num_sensor_updates,
        active_aov=evidence_active_aov,
        image_paths=[str(path) for path in image_paths],
        image_bindings=image_bindings,
        presentation_image_path=str(presentation_path) if presentation_path else None,
        shared_render_status=shared_status,
        shared_render_issues=shared_issues,
        response_validation=response_validation,
        image_validation=image_results,
        metadata={
            **dict(shared.get("metadata") or {}),
            "requested_backend": backend,
            "transport_backend": actual_backend or "unknown",
            "renderer": renderer,
            "renderer_identity_verified": renderer_identity_verified,
            "edge_on_thin_views": sorted(edge_on_paths),
            "requested_ovrtx_render_mode": effective_ovrtx_mode,
            "requested_ovrtx_num_sensor_updates": (effective_ovrtx_num_sensor_updates),
            "executed_ovrtx_settings": executed_ovrtx_settings,
            "executed_ovrtx_settings_verified": executed_ovrtx_settings_verified,
            "ovrtx_render_mode": evidence_ovrtx_mode,
            "ovrtx_num_sensor_updates": evidence_ovrtx_num_sensor_updates,
            "active_aov": evidence_active_aov,
            "semantic_view_plan": semantic_view_plan,
        },
        report_path=str(report_path),
    )
    atomic_write_json(report_path, result)
    return result
