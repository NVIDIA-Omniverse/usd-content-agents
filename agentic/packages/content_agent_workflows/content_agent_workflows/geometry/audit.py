# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic audit checks for provider-neutral source bundles and geometry."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from world_understanding.functions.graphics.render_validation import (
    RENDER_LOW_CONTRAST,
    validate_image_artifact,
)
from world_understanding.utils.nvcf_utils import resolve_endpoint_or_function_id

from .rendering import (
    GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION,
    GeometryRemoteOvrtxIdentityEvidence,
    _renderer_failure_metadata_findings,
)

GEOMETRY_ASSET_AUDIT_SCHEMA_VERSION = "content-agent-workflows.geometry-audit.v1"

AuditSeverity = Literal["info", "warning", "error"]
AuditSource = Literal["authoring", "render", "workflow"]

_DEBUG_GEOMETRY_TERMS = (
    "debug_marker",
    "grasp_clearance_marker",
    "clearance_marker",
)
_SIX_VIEW_LABEL_REGIONS = (
    (0, 0, 112, 56),
    (0, 0, 112, 56),
    (0, 0, 112, 56),
    (0, 0, 112, 56),
    (0, 0, 112, 56),
    (0, 0, 112, 56),
)


class GeometryAuditSignal(BaseModel):
    """One machine-readable geometry audit finding."""

    model_config = ConfigDict(extra="forbid")

    severity: AuditSeverity
    code: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source: AuditSource
    blocking: bool = False
    repair_hint: str | None = None
    line: int | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuthoringContract(BaseModel):
    """Small provider-neutral source summary for downstream agents."""

    model_config = ConfigDict(extra="forbid")

    artifact_path: str | None = None
    artifact_format: str | None = None
    source_manifest_path: str | None = None
    source_bundle_id: str | None = None
    source_revision: str | None = None
    source_provider_id: str | None = None
    source_representation_id: str | None = None
    fidelity: str | None = None
    prompt_supplied: bool = False
    parameter_count: int = 0
    verification_assertion_count: int = 0
    semantic_part_count: int = 0
    declared_authoring_quality_checks: list[str] = Field(default_factory=list)
    variant_parameters: list[str] = Field(default_factory=list)
    variant_id: str | None = None
    param_overrides: dict[str, Any] = Field(default_factory=dict)


class GeometryAssetAuditReport(BaseModel):
    """Audit report emitted beside geometry workflow artifacts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = GEOMETRY_ASSET_AUDIT_SCHEMA_VERSION
    passed: bool
    authoring_contract: AuthoringContract
    signals: list[GeometryAuditSignal] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def audit_geometry_asset(
    *,
    source_path: Path | None = None,
    source_text: str | None = None,
    prompt: str | None = None,
    context: dict[str, Any] | None = None,
    render_image_path: Path | None = None,
    render_presentation_path: Path | None = None,
    render_metadata_path: Path | None = None,
    render_preset: str = "six_view",
    require_ovrtx_render: bool = False,
    trusted_artifact_roots: tuple[Path, ...] = (),
) -> GeometryAssetAuditReport:
    """Run deterministic source and render checks for one geometry asset."""

    del source_text, trusted_artifact_roots
    context = context or {}
    prepared_source_metadata = cast(
        dict[str, Any], context.get("prepared_source_metadata") or {}
    )
    source_bundle = _source_bundle_metadata(prepared_source_metadata)
    contract = _authoring_contract(
        source_path=source_path,
        source_bundle=source_bundle,
        prompt=prompt,
        context=context,
    )
    signals: list[GeometryAuditSignal] = []
    signals.extend(
        _audit_source_authoring(
            source_bundle=source_bundle,
            prompt=prompt,
            context=context,
        )
    )
    signals.extend(
        _audit_render_artifacts(
            render_image_path=render_image_path,
            render_presentation_path=render_presentation_path,
            render_metadata_path=render_metadata_path,
            render_preset=render_preset,
            require_ovrtx_render=require_ovrtx_render,
        )
    )
    blocking = [signal for signal in signals if signal.blocking]
    return GeometryAssetAuditReport(
        passed=not blocking,
        authoring_contract=contract,
        signals=signals,
        metadata={
            "blocking_signal_count": len(blocking),
            "signal_count": len(signals),
        },
    )


def write_audit_artifacts(
    *,
    output_dir: Path,
    report: GeometryAssetAuditReport,
) -> dict[str, str]:
    """Write audit report, contract, render report, and summary JSON artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    authoring_contract_path = output_dir / "authoring_contract.json"
    authoring_contract_path.write_text(
        json.dumps(
            report.authoring_contract.model_dump(mode="json"), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    render_signals = [
        signal.model_dump(mode="json")
        for signal in report.signals
        if signal.source == "render"
    ]
    render_audit_path = output_dir / "render_audit_report.json"
    render_audit_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "passed": not any(signal["blocking"] for signal in render_signals),
                "signals": render_signals,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "geometry_audit_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_path = output_dir / "asset_audit_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "passed": report.passed,
                "blocking_signal_count": report.metadata["blocking_signal_count"],
                "signal_count": report.metadata["signal_count"],
                "codes": sorted({signal.code for signal in report.signals}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "authoring_contract_path": str(authoring_contract_path),
        "render_audit_path": str(render_audit_path),
        "geometry_audit_path": str(report_path),
        "asset_audit_summary_path": str(summary_path),
    }


def _authoring_contract(
    *,
    source_path: Path | None,
    source_bundle: dict[str, Any] | None,
    prompt: str | None,
    context: dict[str, Any],
) -> AuthoringContract:
    metadata = cast(dict[str, Any], context.get("prepared_source_metadata") or {})
    parameter_ranges = metadata.get("parameter_ranges") or {}
    semantic_parts = metadata.get("semantic_parts") or []
    parameters = _bundle_records(source_bundle, "parameters")
    assertions = _bundle_records(source_bundle, "verification_assertions")
    parts = _bundle_records(source_bundle, "parts")
    selected = _bundle_record(source_bundle, "selected_representation")
    producer = _bundle_record(source_bundle, "producer")
    variant_parameters = [
        str(item)
        for item in (
            context.get("variant_parameters")
            or metadata.get("variant_parameters")
            or list(parameter_ranges)
        )
    ]
    return AuthoringContract(
        artifact_path=str(source_path) if source_path else None,
        artifact_format=_source_format(source_path),
        source_manifest_path=(
            str(source_bundle.get("manifest_path"))
            if source_bundle and source_bundle.get("manifest_path")
            else None
        ),
        source_bundle_id=(
            str(source_bundle.get("bundle_id"))
            if source_bundle and source_bundle.get("bundle_id")
            else None
        ),
        source_revision=(
            str(source_bundle.get("source_revision"))
            if source_bundle and source_bundle.get("source_revision")
            else None
        ),
        source_provider_id=(
            str(producer.get("provider_id")) if producer.get("provider_id") else None
        ),
        source_representation_id=(
            str(selected.get("representation_id"))
            if selected.get("representation_id")
            else None
        ),
        fidelity=(
            "provider_source_bundle"
            if source_bundle is not None
            else str(metadata.get("source_fidelity_tier") or "") or None
        ),
        prompt_supplied=prompt is not None,
        parameter_count=max(
            len(parameters),
            len(parameter_ranges) if isinstance(parameter_ranges, dict) else 0,
        ),
        verification_assertion_count=max(
            len(assertions),
            len(metadata.get("verification_assertions") or [])
            if isinstance(metadata.get("verification_assertions"), list)
            else 0,
        ),
        semantic_part_count=max(
            len(parts), len(semantic_parts) if isinstance(semantic_parts, list) else 0
        ),
        declared_authoring_quality_checks=[
            str(check) for check in context.get("authoring_quality_checks") or []
        ],
        variant_parameters=variant_parameters,
        variant_id=str(context["variant_id"]) if context.get("variant_id") else None,
        param_overrides=dict(context.get("param_overrides") or {}),
    )


def _source_format(source_path: Path | None) -> str | None:
    if source_path is None:
        return None
    return source_path.suffix.lower().lstrip(".")


def _source_bundle_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    value = metadata.get("source_bundle")
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "geometry.source.v1"
    ):
        return None
    return dict(value)


def _bundle_records(
    source_bundle: dict[str, Any] | None,
    key: str,
) -> list[dict[str, Any]]:
    if source_bundle is None:
        return []
    value = source_bundle.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _bundle_record(
    source_bundle: dict[str, Any] | None,
    key: str,
) -> dict[str, Any]:
    if source_bundle is None:
        return {}
    value = source_bundle.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _normalized_parameter_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _audit_source_authoring(
    *,
    source_bundle: dict[str, Any] | None,
    prompt: str | None,
    context: dict[str, Any],
) -> list[GeometryAuditSignal]:
    """Evaluate source semantics without parsing or executing provider-native data."""

    checks = {str(check) for check in context.get("authoring_quality_checks") or []}
    metadata = cast(dict[str, Any], context.get("prepared_source_metadata") or {})
    parameters = _bundle_records(source_bundle, "parameters")
    parts = _bundle_records(source_bundle, "parts")
    provider_assertions = _bundle_records(source_bundle, "verification_assertions")
    semantic_parts = metadata.get("semantic_parts")
    parameter_ranges = metadata.get("parameter_ranges")
    workflow_assertions = metadata.get("verification_assertions")
    semantic_text = json.dumps(
        {
            "parameters": parameters,
            "parts": parts,
            "provider_assertions": provider_assertions,
            "semantic_parts": semantic_parts
            if isinstance(semantic_parts, list)
            else [],
            "workflow_assertions": (
                workflow_assertions if isinstance(workflow_assertions, list) else []
            ),
        },
        sort_keys=True,
    ).lower()
    prompt_text = str(prompt or "").lower()
    signals: list[GeometryAuditSignal] = []
    known_checks = {
        "contact_surface_semantics",
        "fin_port_semantics",
        "handle_grasp_clearance",
        "mate_clearance_semantics",
        "no_debug_markers",
        "parameter_ranges",
        "real_cavity_boolean",
        "semantic_parts",
        "semantic_variant_parameters",
        "thread_proxy_semantics",
        "prompt_variant_parameters",
        "variant_parameters",
        "verify_probes",
    }
    for check in sorted(checks - known_checks):
        signals.append(
            _signal(
                "authoring.unknown_quality_check",
                f"Unknown authoring quality check {check!r}.",
                blocking=True,
                metadata={"check": check},
            )
        )

    for assertion in provider_assertions:
        assertion_status = str(assertion.get("status") or "")
        if assertion_status not in {"failed", "warning"}:
            continue
        signals.append(
            _signal(
                f"authoring.provider_assertion_{assertion_status}",
                str(
                    assertion.get("summary") or "Authoring provider reported an issue."
                ),
                blocking=assertion_status == "failed",
                metadata={
                    "assertion_id": assertion.get("assertion_id"),
                    "provider_assertion": True,
                },
            )
        )

    if not checks:
        return signals

    declared_parameters = parameters
    if not declared_parameters and isinstance(parameter_ranges, dict):
        declared_parameters = [
            {"name": name, **(value if isinstance(value, dict) else {})}
            for name, value in parameter_ranges.items()
        ]
    parameter_names = {
        _normalized_parameter_name(item.get("name") or item.get("id"))
        for item in declared_parameters
        if item.get("name") or item.get("id")
    }
    if "parameter_ranges" in checks:
        numeric_parameters = [
            item
            for item in declared_parameters
            if str(item.get("value_type") or item.get("type") or "number")
            in {"number", "integer", "int", "float"}
        ]
        unbounded = [
            str(item.get("name") or item.get("id"))
            for item in numeric_parameters
            if (item.get("minimum", item.get("min")) is None)
            or (item.get("maximum", item.get("max")) is None)
        ]
        if not declared_parameters or unbounded:
            signals.append(
                _signal(
                    "authoring.parameter_ranges_missing",
                    "Semantic CAD parameters are missing complete numeric bounds.",
                    blocking=True,
                    repair_hint="Declare task-relevant ParameterNode minimum and maximum values.",
                    metadata={"unbounded": unbounded},
                )
            )

    validation_count = len(provider_assertions)
    if not validation_count and isinstance(workflow_assertions, list):
        validation_count = len(workflow_assertions)
    if "verify_probes" in checks and validation_count < 1:
        signals.append(
            _signal(
                "authoring.validation_checks_missing",
                "Authoring source does not carry provider assertions or workflow checks.",
                blocking=True,
                repair_hint="Add typed assertions, then run Geometry-owned validation.",
            )
        )

    semantic_part_count = max(
        len(parts),
        len(semantic_parts) if isinstance(semantic_parts, list) else 0,
    )
    if "semantic_parts" in checks and semantic_part_count < 1:
        signals.append(
            _signal(
                "authoring.semantic_parts_missing",
                "Authoring source does not expose named semantic parts.",
                blocking=True,
                repair_hint="Declare named parts and bind them to source representations.",
            )
        )

    if "real_cavity_boolean" in checks:
        has_negative_space = any(
            term in semantic_text
            for term in ("cavity", "hollow", "interior", "opening")
        )
        has_subtraction_evidence = any(
            term in semantic_text
            for term in ("subtract", "difference", "boolean", "no_geometry_in_volume")
        )
        if not has_negative_space or not has_subtraction_evidence:
            signals.append(
                _signal(
                    "authoring.real_cavity_boolean_missing",
                    "Container affordance lacks semantic negative-space and subtraction evidence.",
                    blocking=True,
                    repair_hint="Declare the cavity operation and a binding opening/intrusion invariant.",
                )
            )

    if "handle_grasp_clearance" in checks and not all(
        term in semantic_text for term in ("handle", "grasp", "clearance")
    ):
        signals.append(
            _signal(
                "authoring.handle_grasp_clearance_missing",
                "Handle affordance lacks semantic handle, grasp, and clearance evidence.",
                blocking=True,
            )
        )

    variant_parameters = [
        str(value) for value in context.get("variant_parameters") or []
    ]
    normalized_variants = {
        _normalized_parameter_name(value) for value in variant_parameters
    }
    missing_variants = sorted(normalized_variants - parameter_names)
    proof_names = {
        _normalized_parameter_name(value)
        for value in _variant_proof_param_names(context)
    }
    missing_proof = sorted(proof_names - parameter_names)
    if "variant_parameters" in checks:
        if not variant_parameters and len(parameter_names) < 2:
            signals.append(
                _signal(
                    "authoring.variant_parameters_missing",
                    "Scenario claims variants but exposes fewer than two semantic parameters.",
                    blocking=True,
                )
            )
        if missing_variants or missing_proof:
            signals.append(
                _signal(
                    "authoring.variant_parameters_not_declared",
                    "Variant controls are not declared by source-bundle parameters.",
                    blocking=True,
                    metadata={
                        "missing_variant_parameters": missing_variants,
                        "missing_proof_parameters": missing_proof,
                    },
                )
            )
    if "semantic_variant_parameters" in checks:
        generic = {"scale", "size", "width", "height", "length"}
        semantic_variants = normalized_variants - generic
        if len(semantic_variants) < 2:
            signals.append(
                _signal(
                    "authoring.semantic_variant_parameters_missing",
                    "Scenario exposes fewer than two object-specific semantic controls.",
                    blocking=True,
                    metadata={"variant_parameters": variant_parameters},
                )
            )
    if "prompt_variant_parameters" in checks:
        missing_prompt = [
            value for value in variant_parameters if value.lower() not in prompt_text
        ]
        if missing_prompt:
            signals.append(
                _signal(
                    "authoring.prompt_variant_parameters_missing",
                    "Prompt does not name every declared scenario variant parameter.",
                    blocking=True,
                    metadata={"missing": missing_prompt},
                )
            )

    required_term_groups = {
        "thread_proxy_semantics": (("male_thread",), ("female_thread",)),
        "mate_clearance_semantics": (("lug",), ("socket",), ("clearance",)),
        "contact_surface_semantics": (
            ("contact", "workpiece", "hard_stop", "stop", "seat"),
            ("keepout", "mount", "datum", "pad", "fixture"),
        ),
        "fin_port_semantics": (("fin",), ("port",)),
    }
    for check, groups in required_term_groups.items():
        if check not in checks:
            continue
        missing_groups = [
            group
            for group in groups
            if not any(term in semantic_text for term in group)
        ]
        if missing_groups:
            signals.append(
                _signal(
                    f"authoring.{check}_missing",
                    f"Source-bundle semantic records do not satisfy {check!r}.",
                    blocking=True,
                    metadata={"missing_term_groups": missing_groups},
                )
            )

    if "no_debug_markers" in checks:
        offenders = [term for term in _DEBUG_GEOMETRY_TERMS if term in semantic_text]
        if offenders:
            signals.append(
                _signal(
                    "authoring.debug_geometry_present",
                    "Authoring source publishes debug or clearance marker parts.",
                    blocking=True,
                    metadata={"terms": offenders},
                )
            )
    return signals


def _variant_proof_param_names(context: dict[str, Any]) -> set[str]:
    proof = context.get("variant_proof") or {}
    names: set[str] = set()
    if not isinstance(proof, dict):
        return names
    for row in proof.get("rows") or []:
        if not isinstance(row, dict):
            continue
        params = row.get("params") or {}
        if isinstance(params, dict):
            names.update(str(name) for name in params)
    return names


def _audit_render_artifacts(
    *,
    render_image_path: Path | None,
    render_presentation_path: Path | None = None,
    render_metadata_path: Path | None,
    render_preset: str,
    require_ovrtx_render: bool,
) -> list[GeometryAuditSignal]:
    signals: list[GeometryAuditSignal] = []
    if require_ovrtx_render and render_metadata_path is None:
        signals.append(
            GeometryAuditSignal(
                severity="error",
                code="render.required_ovrtx_missing",
                summary="Acceptance render is required but no OVRTX metadata artifact was supplied.",
                source="render",
                blocking=True,
                repair_hint="Render with OVRTX and pass the generated metadata path into the audit.",
            )
        )
    if require_ovrtx_render and render_image_path is None:
        signals.append(
            GeometryAuditSignal(
                severity="error",
                code="render.required_ovrtx_image_missing",
                summary=(
                    "Acceptance render is required but no rendered image "
                    "artifact was supplied."
                ),
                source="render",
                blocking=True,
                repair_hint=(
                    "Provide an individually validated image from the "
                    "successful OVRTX render."
                ),
            )
        )
    if render_metadata_path is not None:
        signals.extend(
            _audit_ovrtx_metadata(
                render_metadata_path,
                require_ovrtx_render,
                render_image_path=render_image_path,
            )
        )
    if render_image_path is not None:
        # Acceptance remains bound to an individual OVRTX image. Grid framing
        # is evaluated only against the derived presentation image below.
        signals.extend(
            _audit_image_health(render_image_path, render_preset="individual")
        )
    if render_presentation_path is not None:
        signals.extend(
            _audit_image_health(
                render_presentation_path,
                render_preset=render_preset,
            )
        )
    return signals


def _audit_ovrtx_metadata(
    render_metadata_path: Path,
    require_ovrtx_render: bool,
    *,
    render_image_path: Path | None = None,
) -> list[GeometryAuditSignal]:
    if not render_metadata_path.exists():
        return [
            GeometryAuditSignal(
                severity="error",
                code="render.metadata_missing",
                summary=f"Render metadata does not exist: {render_metadata_path}",
                source="render",
                blocking=require_ovrtx_render,
            )
        ]
    try:
        metadata = json.loads(render_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            GeometryAuditSignal(
                severity="error",
                code="render.metadata_invalid_json",
                summary=f"Render metadata is not valid JSON: {exc}",
                source="render",
                blocking=require_ovrtx_render,
                artifact_paths=[str(render_metadata_path)],
            )
        ]
    if not isinstance(metadata, dict):
        return [
            GeometryAuditSignal(
                severity="error",
                code="render.metadata_invalid_shape",
                summary="Render metadata must be a JSON object.",
                source="render",
                blocking=require_ovrtx_render,
                artifact_paths=[str(render_metadata_path)],
            )
        ]
    renderer = str(metadata.get("renderer") or "").lower()
    backend = str(metadata.get("backend") or renderer).lower()
    requested_backend = str(metadata.get("requested_backend") or "").lower()
    identity_verified = metadata.get("renderer_identity_verified") is True
    identity_evidence = metadata.get("renderer_identity_evidence")
    image_paths = metadata.get("image_paths")
    shared_render_issues = metadata.get("shared_render_issues")
    if (
        require_ovrtx_render
        and metadata.get("schema_version") != GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION
    ):
        return [
            GeometryAuditSignal(
                severity="error",
                code="render.metadata_schema_unsupported",
                summary=(
                    "OVRTX acceptance requires the current Geometry render "
                    "evidence schema."
                ),
                source="render",
                blocking=True,
                artifact_paths=[str(render_metadata_path)],
                metadata={
                    "expected_schema_version": (
                        GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION
                    ),
                    "actual_schema_version": metadata.get("schema_version"),
                },
            )
        ]
    renderer_failure_findings = _renderer_failure_metadata_findings(
        render_response=metadata.get("render_response"),
        renderer_metadata=metadata.get("metadata"),
        response_validation=metadata.get("response_validation"),
        legacy_metadata={
            field: metadata.get(field)
            for field in ("error", "errors", "failure", "failures")
        },
    )
    if renderer_failure_findings:
        return [
            GeometryAuditSignal(
                severity="error",
                code="render.renderer_reported_failure",
                summary=(
                    "Render metadata retains a terminal renderer status, error, "
                    "or failure marker."
                ),
                source="render",
                blocking=True,
                artifact_paths=[str(render_metadata_path)],
                repair_hint=(
                    "Resolve the renderer-reported failure and regenerate the "
                    "acceptance evidence."
                ),
                metadata={"findings": renderer_failure_findings},
            )
        ]
    shared_render_failed = isinstance(shared_render_issues, list) and any(
        isinstance(issue, dict)
        and str(issue.get("severity") or "error").lower()
        in {"error", "fail", "failure", "fatal"}
        for issue in shared_render_issues
    )
    successful_render = (
        metadata.get("status") == "pass"
        and metadata.get("shared_render_status") == "completed"
        and isinstance(image_paths, list)
        and any(isinstance(path, str) and path.strip() for path in image_paths)
        and not shared_render_failed
    )
    if require_ovrtx_render and not successful_render:
        return [
            GeometryAuditSignal(
                severity="error",
                code="render.required_ovrtx_unsuccessful",
                summary=(
                    "OVRTX acceptance requires a passing, completed render "
                    "report with at least one validated image."
                ),
                source="render",
                blocking=True,
                artifact_paths=[str(render_metadata_path)],
                repair_hint=(
                    "Resolve render failures and rerun OVRTX until the report "
                    "passes with an accepted image."
                ),
                metadata={
                    "status": metadata.get("status"),
                    "shared_render_status": metadata.get("shared_render_status"),
                    "image_count": len(image_paths)
                    if isinstance(image_paths, list)
                    else 0,
                },
            )
        ]
    if require_ovrtx_render and render_image_path is not None:
        report_image_paths: set[Path] = set()
        try:
            for value in image_paths:
                if not isinstance(value, str) or not value.strip():
                    continue
                candidate = Path(value.strip())
                if not candidate.is_absolute():
                    candidate = render_metadata_path.parent / candidate
                report_image_paths.add(candidate.resolve())
            supplied_image_path = render_image_path.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            return [
                GeometryAuditSignal(
                    severity="error",
                    code="render.required_ovrtx_image_path_invalid",
                    summary=(
                        "An OVRTX report image path could not be safely "
                        f"resolved: {exc}"
                    ),
                    source="render",
                    blocking=True,
                    artifact_paths=[str(render_metadata_path)],
                    repair_hint=(
                        "Regenerate the render report with ordinary, resolvable "
                        "artifact paths."
                    ),
                    metadata={"error_type": type(exc).__name__},
                )
            ]
        if supplied_image_path not in report_image_paths:
            return [
                GeometryAuditSignal(
                    severity="error",
                    code="render.required_ovrtx_image_unbound",
                    summary=(
                        "The supplied acceptance image is not one of the "
                        "individual images recorded by the OVRTX render report."
                    ),
                    source="render",
                    blocking=True,
                    artifact_paths=[
                        str(render_metadata_path),
                        str(supplied_image_path),
                    ],
                    repair_hint=(
                        "Supply an accepted individual render view whose "
                        "resolved path appears in the report image_paths."
                    ),
                )
            ]
    if require_ovrtx_render and backend in {"ovrtx", "remote"}:
        retained_metadata = metadata.get("metadata")
        executed = (
            retained_metadata.get("executed_ovrtx_settings")
            if isinstance(retained_metadata, dict)
            else None
        )
        image_bindings = metadata.get("image_bindings")
        settings_valid = (
            isinstance(retained_metadata, dict)
            and retained_metadata.get("executed_ovrtx_settings_verified") is True
            and isinstance(executed, dict)
            and executed.get("ovrtx_render_mode")
            == metadata.get("ovrtx_render_mode")
            == retained_metadata.get("requested_ovrtx_render_mode")
            and executed.get("ovrtx_num_sensor_updates")
            == metadata.get("ovrtx_num_sensor_updates")
            == retained_metadata.get("requested_ovrtx_num_sensor_updates")
            and isinstance(executed.get("active_aov"), str)
            and bool(executed["active_aov"].strip())
            and executed.get("active_aov")
            == metadata.get("active_aov")
            == retained_metadata.get("active_aov")
            and isinstance(image_bindings, list)
            and bool(image_bindings)
            and all(
                isinstance(binding, dict)
                and binding.get("active_aov") == executed.get("active_aov")
                for binding in image_bindings
            )
        )
        if not settings_valid:
            scope = "remote" if backend == "remote" else "local"
            return [
                GeometryAuditSignal(
                    severity="error",
                    code=f"render.{scope}_ovrtx_settings_unverified",
                    summary=(
                        f"{scope.capitalize()} OVRTX acceptance evidence does not bind complete "
                        "executed settings matching the request."
                    ),
                    source="render",
                    blocking=True,
                    artifact_paths=[str(render_metadata_path)],
                    repair_hint=(
                        "Use an OVRTX renderer that reports its executed mode, "
                        "sensor-update count, and active AOV for every view, then "
                        "regenerate evidence."
                    ),
                )
            ]
    remote_identity_valid = False
    if backend == "remote":
        try:
            identity = GeometryRemoteOvrtxIdentityEvidence.model_validate(
                identity_evidence
            )
            reported_endpoint = str(metadata.get("renderer_endpoint") or "")
            normalized_endpoint = resolve_endpoint_or_function_id(
                identity.endpoint
            ).rstrip("/")
            remote_identity_valid = (
                identity_verified
                and renderer == "ovrtx"
                and requested_backend == "remote"
                and metadata.get("render_redirects_allowed") is False
                and bool(reported_endpoint)
                and normalized_endpoint
                == resolve_endpoint_or_function_id(reported_endpoint).rstrip("/")
                and identity.health_url == f"{normalized_endpoint}/health"
            )
        except (TypeError, ValueError):
            remote_identity_valid = False
    local_ovrtx = (
        backend == "ovrtx"
        and requested_backend == "ovrtx"
        and renderer == "ovrtx"
        and identity_verified
        and isinstance(identity_evidence, dict)
        and identity_evidence.get("source") == "explicit_local_backend"
    )
    if require_ovrtx_render and not (local_ovrtx or remote_identity_valid):
        return [
            GeometryAuditSignal(
                severity="error",
                code="render.required_ovrtx",
                summary=(
                    "Acceptance render metadata does not prove a local or "
                    "identity-verified remote OVRTX renderer."
                ),
                source="render",
                blocking=True,
                artifact_paths=[str(render_metadata_path)],
                repair_hint=(
                    "Use explicit local OVRTX or retain the remote OVRTX "
                    "identity/readiness health response and no-redirect render "
                    "policy bound to the same endpoint."
                ),
                metadata={
                    "backend": backend,
                    "renderer": renderer,
                    "renderer_identity_verified": metadata.get(
                        "renderer_identity_verified"
                    ),
                },
            )
        ]
    return []


def _audit_image_health(
    render_image_path: Path,
    *,
    render_preset: str,
) -> list[GeometryAuditSignal]:
    result = validate_image_artifact(
        render_image_path,
        backend="ovrtx",
        min_width=256,
        min_height=256,
        detect_error_material_artifacts=True,
    )
    signals = [
        GeometryAuditSignal(
            severity="warning" if issue.code == RENDER_LOW_CONTRAST else "error",
            code=issue.code,
            summary=issue.message,
            source="render",
            blocking=issue.code != RENDER_LOW_CONTRAST,
            artifact_paths=[str(render_image_path)],
            metadata=issue.details,
        )
        for issue in result.issues
    ]
    if not result.readable or result.width is None or result.height is None:
        return signals
    if render_preset != "six_view":
        return signals
    return [
        *signals,
        *_audit_image_framing(
            render_image_path,
            width=result.width,
            height=result.height,
        ),
    ]


def _audit_image_framing(
    render_image_path: Path,
    *,
    width: int,
    height: int,
) -> list[GeometryAuditSignal]:
    if width < 900 or height < 500:
        return []
    # This check divides a composed six-view presentation into a 3x2 grid.
    # The workflow deliberately passes an accepted individual render into the
    # asset audit, so do not misinterpret that square image as six tiny tiles.
    if abs((width / height) - 1.5) > 0.05:
        return []
    try:
        from PIL import Image
    except Exception:
        return []

    try:
        image = Image.open(render_image_path).convert("RGB")
    except Exception:
        return []
    tile_w = max(1, width // 3)
    tile_h = max(1, height // 2)
    if tile_w < 160 or tile_h < 160:
        return []

    blocking_tiles: list[dict[str, Any]] = []
    warning_tiles: list[dict[str, Any]] = []
    labels = ["front", "back", "left", "right", "top", "iso"]
    for index, label in enumerate(labels):
        col = index % 3
        row = index // 3
        left = col * tile_w
        top = row * tile_h
        right = (col + 1) * tile_w if col < 2 else width
        bottom = (row + 1) * tile_h if row < 1 else height
        bbox = _foreground_bbox(
            image.crop((left, top, right, bottom)),
            ignore_region=_SIX_VIEW_LABEL_REGIONS[index],
        )
        if bbox is None:
            continue
        min_x, min_y, max_x, max_y = bbox
        local_w = right - left
        local_h = bottom - top
        margins = {
            "left": min_x,
            "top": min_y,
            "right": local_w - 1 - max_x,
            "bottom": local_h - 1 - max_y,
        }
        min_margin = min(margins.values())
        tile_info = {"view": label, "margins": margins, "minimum_margin_px": min_margin}
        if min_margin <= 2:
            blocking_tiles.append(tile_info)
        elif min_margin < max(14, int(min(local_w, local_h) * 0.035)):
            warning_tiles.append(tile_info)

    signals: list[GeometryAuditSignal] = []
    if blocking_tiles:
        signals.append(
            GeometryAuditSignal(
                severity="error",
                code="render.image_content_clipped",
                summary="Rendered object content touches a six-view tile edge.",
                source="render",
                blocking=True,
                artifact_paths=[str(render_image_path)],
                repair_hint="Increase OVRTX camera fit padding or adjust the camera target for this asset.",
                metadata={"tiles": blocking_tiles},
            )
        )
    if warning_tiles:
        signals.append(
            GeometryAuditSignal(
                severity="warning",
                code="render.image_tight_framing",
                summary="Rendered object content is very close to a six-view tile edge.",
                source="render",
                blocking=False,
                artifact_paths=[str(render_image_path)],
                repair_hint="Use a looser camera fit so visual review does not look cropped.",
                metadata={"tiles": warning_tiles},
            )
        )
    return signals


def _foreground_bbox(
    image: Any,
    *,
    ignore_region: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    pixels = image.load()
    width, height = image.size
    background = pixels[0, 0]
    xs: list[int] = []
    ys: list[int] = []
    ix0, iy0, ix1, iy1 = ignore_region or (-1, -1, -1, -1)
    for y in range(height):
        for x in range(width):
            if ix0 <= x < ix1 and iy0 <= y < iy1:
                continue
            r, g, b = pixels[x, y]
            if (
                abs(r - background[0]) + abs(g - background[1]) + abs(b - background[2])
                <= 36
            ):
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _signal(
    code: str,
    summary: str,
    *,
    blocking: bool,
    repair_hint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GeometryAuditSignal:
    return GeometryAuditSignal(
        severity="error" if blocking else "warning",
        code=code,
        summary=summary,
        source="authoring",
        blocking=blocking,
        repair_hint=repair_hint,
        metadata=metadata or {},
    )
