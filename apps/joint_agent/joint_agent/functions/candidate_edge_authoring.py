# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author exact USD joint edges from ready Stage 2 candidates.

This module is intentionally narrower than a simulation-readiness pass. It
translates already-resolved Stage 2 topology into USD physics joint prims and
does not add rigid bodies, mass, collision, articulation-root, joint-state,
drive, mimic, contact, or geometry-derived limit opinions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from world_understanding.functions.physics.joint_rigger.source_binding import (
    BoundInputDirectory,
    _validate_bound_projection_dependencies,  # noqa: F401 - compatibility alias
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    materialize_bound_input as _materialize_bound_input,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    remove_bound_input_directory as _remove_bound_input_directory,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    restore_bound_projection_paths as _restore_bound_projection_paths,
)

from joint_agent.functions.articulation_candidates import (
    STAGE2_SCHEMA_VERSION,
    Stage2ArticulationCandidate,
)
from joint_agent.functions.artifact_transaction import (
    StagedArtifact as _StagedArtifact,
)
from joint_agent.functions.artifact_transaction import (
    promote_staged_artifacts as _promote_staged_artifacts,
)
from joint_agent.functions.artifact_transaction import (
    remove_artifact as _remove_artifact,
)

ADAPTER_NAME = "stage2_candidate_edges"
DIAGNOSTICS_SCHEMA_VERSION = "joint-agent-rigger-diagnostics-v0"
VALIDATION_SCHEMA_VERSION = "joint-agent-rigger-validation-v0"
AUTHORING_SCHEMA_VERSION = "joint-agent-stage2-candidate-edge-authoring-v0"
_READY_STATUS = "ready_for_rigger_input"
_SUPPORTED_JOINT_TYPES = frozenset({"revolute", "prismatic", "spherical"})
_AXIS_VECTORS: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
_AXIS_TOKENS = {"x": "X", "y": "Y", "z": "Z"}
_DISALLOWED_FALLBACK_SOURCES = frozenset(
    {"geometry_inferred", "structural_fallback", "template_default", "unknown"}
)
_SOURCE_BACKED_LIMIT_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
        "template_default",
    }
)
_BODY0_BODY1_EDGE_ROLE = "body0_body1_edge"
_BODY1_OWNERSHIP_ROLE = "body1_ownership"
_ENDPOINT_CANONICALIZATION_ROLE = "endpoint_canonicalization"
_RAW_USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc"})
_MAX_STAGE2_DOCUMENT_BYTES = 64 * 1024 * 1024
_CANDIDATE_READINESS_SHA256_FIELD = "articulation_candidates_sha256"


class _Stage2Summary(BaseModel):
    """Core summary fields that must agree with the full candidate list."""

    model_config = ConfigDict(extra="allow", strict=True)

    candidate_count: int = Field(ge=0)
    ready_candidate_count: int = Field(ge=0)
    review_required_candidate_count: int = Field(ge=0)


class _Stage2Document(BaseModel):
    """Exact top-level v0 Stage 2 document consumed by this adapter."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["joint-agent-stage2-v0"]
    summary: _Stage2Summary
    candidates: list[Stage2ArticulationCandidate]


@dataclass(frozen=True)
class _JointPlan:
    candidate: Stage2ArticulationCandidate
    joint_path: str
    body0: str
    body1: str
    joint_type: str
    axis_token: str
    motion_axis_world: tuple[float, float, float]
    local_pos0: tuple[float, float, float]
    local_pos1: tuple[float, float, float]
    local_rot0: tuple[float, tuple[float, float, float]]
    local_rot1: tuple[float, tuple[float, float, float]]
    anchor_world: tuple[float, float, float]
    lower_limit: float | None
    upper_limit: float | None
    authored_limit_unit: str | None


def author_stage2_candidate_edges(
    *,
    input_usd_path: str | Path,
    articulation_candidates_path: str | Path,
    output_usd_path: str | Path,
    diagnostics_path: str | Path,
    validation_path: str | Path,
    predictions_path: str | Path | None = None,
    candidate_readiness: Mapping[str, Any] | None = None,
    _skip_direct_artifact_clear: bool = False,
    _bound_input_descriptor: int | None = None,
    _bound_input_sha256: str | None = None,
    _bound_input_dependencies: tuple[tuple[str, int, str, str, bool], ...] = (),
    _logical_output_parent: str | Path | None = None,
) -> dict[str, Any]:
    """Author ready, exact Stage 2 edges without adding body physics schemas.

    The candidate document, source stage, endpoint paths, deterministic joint
    paths, transforms, axes, and limits are all preflighted before any output
    artifact is created. The candidate document must be a regular UTF-8 JSON
    file no larger than 64 MiB. Predictions are accepted only for diagnostic
    path reporting and are never opened or interpreted by this adapter. The
    private clear-skip is reserved for bridge-owned, cryptographically absent
    targets; it does not bypass path validation, output validation, or staged
    promotion.
    """
    input_path = Path(input_usd_path)
    candidates_path = Path(articulation_candidates_path)
    output_path = Path(output_usd_path)
    diagnostics = Path(diagnostics_path)
    validation = Path(validation_path)
    predictions = Path(predictions_path) if predictions_path is not None else None
    if (_bound_input_descriptor is None) != (_bound_input_sha256 is None):
        raise ValueError(
            "_bound_input_descriptor and _bound_input_sha256 must be paired"
        )
    if _bound_input_descriptor is not None and not _skip_direct_artifact_clear:
        raise ValueError("bound input descriptors are reserved for bridge authoring")
    if _bound_input_dependencies and _bound_input_descriptor is None:
        raise ValueError("bound input dependencies require a bound root descriptor")
    logical_output_parent = (
        Path(_logical_output_parent) if _logical_output_parent is not None else None
    )
    if logical_output_parent is not None and not _skip_direct_artifact_clear:
        raise ValueError("logical output parents are reserved for bridge authoring")

    _validate_direct_artifact_paths(
        input_path=input_path,
        articulation_candidates_path=candidates_path,
        predictions_path=predictions,
        output_path=output_path,
        diagnostics_path=diagnostics,
        validation_path=validation,
    )
    expected_candidates_sha256 = _candidate_readiness_sha256(candidate_readiness)
    bound_candidate_payload: str | None = None
    if expected_candidates_sha256 is not None:
        bound_candidate_payload = _read_stable_stage2_document(candidates_path)
        if (
            hashlib.sha256(bound_candidate_payload.encode("utf-8")).hexdigest()
            != expected_candidates_sha256
        ):
            raise RuntimeError(
                "Stage 2 candidate document no longer matches candidate readiness"
            )
    private_target_guard: Callable[[], None] | None = None
    if _skip_direct_artifact_clear:
        private_targets = _private_authoring_targets(
            input_path=input_path,
            output_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
        )
        _require_private_targets_absent(private_targets)

        def require_private_targets_absent() -> None:
            _require_private_targets_absent(private_targets)

        private_target_guard = require_private_targets_absent
    if not _skip_direct_artifact_clear:
        _clear_previous_direct_artifacts(
            input_path=input_path,
            articulation_candidates_path=candidates_path,
            predictions_path=predictions,
            output_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
        )
    _validate_output_extension(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    authoring_input_path = input_path
    bound_input_dir: BoundInputDirectory | None = None
    bound_projection_restore_paths: Mapping[Path, Path] = {}
    try:
        if _bound_input_descriptor is not None:
            assert _bound_input_sha256 is not None
            (
                authoring_input_path,
                bound_input_dir,
                bound_projection_restore_paths,
            ) = _materialize_bound_input(
                descriptor=_bound_input_descriptor,
                expected_sha256=_bound_input_sha256,
                logical_input_path=input_path,
                dependencies=_bound_input_dependencies,
            )
        document = (
            _load_and_validate_document(candidates_path)
            if bound_candidate_payload is None
            else _parse_and_validate_document(
                bound_candidate_payload,
                path=candidates_path,
            )
        )
        stage, plans, input_snapshot = _preflight_stage_and_edges(
            authoring_input_path,
            document,
        )
    except BaseException as preflight_error:
        if bound_input_dir is not None:
            try:
                _remove_bound_input_directory(bound_input_dir)
            except Exception as cleanup_error:
                preflight_error.add_note(
                    "Bound input cleanup also failed: " + str(cleanup_error)
                )
        raise
    # Do not retain a live source-stage handle while an output package is
    # copied/extracted and replaced.
    del stage

    if candidate_readiness and candidate_readiness.get("status") == "blocked":
        # Unlike the compatibility adapters, direct edge authoring performs its
        # exact document/stage preflight before honoring the optional policy
        # block. Malformed input therefore never produces blocked-result
        # artifacts that could be mistaken for a valid preflight.
        try:
            return _write_readiness_blocked_result(
                input_path=input_path,
                predictions_path=predictions,
                candidates_path=candidates_path,
                configured_output_path=output_path,
                diagnostics_path=diagnostics,
                validation_path=validation,
                candidate_readiness=candidate_readiness,
                prebackup_validator=private_target_guard,
                replace_existing=not _skip_direct_artifact_clear,
            )
        finally:
            if bound_input_dir is not None:
                _remove_bound_input_directory(bound_input_dir)

    warnings = [
        str(warning) for warning in (candidate_readiness or {}).get("warnings", [])
    ]
    ignored_count = len(document.candidates) - len(plans)
    if ignored_count:
        warnings.append(
            f"ignored {ignored_count} candidate(s) whose review_status is not "
            f"{_READY_STATUS}"
        )
    if not plans:
        warnings.append(
            "stage2_candidate_edges found no ready candidate edges; no generated "
            "USD was written"
        )
        try:
            return _write_no_joints_authored_result(
                input_path=input_path,
                predictions_path=predictions,
                candidates_path=candidates_path,
                configured_output_path=output_path,
                diagnostics_path=diagnostics,
                validation_path=validation,
                candidate_count=len(document.candidates),
                ignored_candidate_count=ignored_count,
                candidate_readiness=candidate_readiness,
                warnings=warnings,
                prebackup_validator=private_target_guard,
                replace_existing=not _skip_direct_artifact_clear,
            )
        finally:
            if bound_input_dir is not None:
                _remove_bound_input_directory(bound_input_dir)

    if (
        bound_input_dir is not None
        and input_path.suffix.lower() in _RAW_USD_EXTENSIONS
        and output_path.suffix.lower() in _RAW_USD_EXTENSIONS
        and logical_output_parent is None
    ):
        _remove_bound_input_directory(bound_input_dir)
        raise RuntimeError("Bound raw output requires a logical output parent")

    temp_output, output_staging_dir = _staged_output_path(input_path, output_path)
    temp_output_access_path = temp_output.resolve(strict=False)
    preparation_dir: Path | None = None
    staged_evidence: list[_StagedArtifact] = []
    primary_error: BaseException | None = None
    try:
        editable_path, preparation_dir = _prepare_editable_output(
            authoring_input_path,
            temp_output,
            output_anchor_path=temp_output_access_path,
        )
        editable_access_path = (
            temp_output_access_path if editable_path == temp_output else editable_path
        )
        output_stage = _open_stage(
            editable_access_path,
            label="editable output USD",
        )
        authored_edges = _author_plans(output_stage, plans)
        if not output_stage.GetRootLayer().Save():
            raise RuntimeError(f"Could not save authored USD layer: {editable_path}")
        del output_stage

        if output_path.suffix.lower() == ".usdz":
            _package_usdz(editable_access_path, temp_output_access_path)

        validation_checks = _validate_authored_output(
            temp_output_access_path,
            plans,
            input_snapshot=input_snapshot,
        )
        if (
            bound_input_dir is not None
            and input_path.suffix.lower() in _RAW_USD_EXTENSIONS
            and output_path.suffix.lower() in _RAW_USD_EXTENSIONS
        ):
            assert logical_output_parent is not None
            _restore_bound_projection_paths(
                temp_output_access_path,
                projection_root=bound_input_dir.path / "filesystem",
                logical_output_parent=logical_output_parent,
                restore_paths=bound_projection_restore_paths,
            )
        diagnostics_payload = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "authoring_schema_version": AUTHORING_SCHEMA_VERSION,
            "adapter": ADAPTER_NAME,
            "status": "authored",
            "input_usd_path": str(input_path),
            "predictions_path": (str(predictions) if predictions is not None else None),
            "predictions_consumed": False,
            "articulation_candidates_path": str(candidates_path),
            "source_candidate_schema_version": STAGE2_SCHEMA_VERSION,
            "output_usd_path": str(output_path),
            "candidate_count": len(document.candidates),
            "ready_candidate_count": len(plans),
            "ignored_candidate_count": ignored_count,
            "authored_joint_count": len(plans),
            "authored_edges": authored_edges,
            "candidate_readiness": dict(candidate_readiness or {}),
            "warnings": warnings,
            "errors": [],
        }
        validation_payload = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "authoring_schema_version": AUTHORING_SCHEMA_VERSION,
            "adapter": ADAPTER_NAME,
            "status": "passed",
            "output_usd_path": str(output_path),
            "validation_skipped": False,
            "authored_joint_count": len(plans),
            "checks": validation_checks,
            "candidate_readiness": dict(candidate_readiness or {}),
            "warnings": warnings,
            "errors": [],
        }
        output_artifacts = _output_promotion_artifacts(
            input_path=input_path,
            staged_output=temp_output,
            output_path=output_path,
            output_staging_dir=output_staging_dir,
            replace_existing=not _skip_direct_artifact_clear,
        )
        if _skip_direct_artifact_clear:
            staged_evidence.append(
                _stage_json_artifact(
                    diagnostics,
                    diagnostics_payload,
                    label="diagnostics",
                    replace_existing=False,
                )
            )
            staged_evidence.append(
                _stage_json_artifact(
                    validation,
                    validation_payload,
                    label="validation",
                    replace_existing=False,
                )
            )
        else:
            staged_evidence.append(
                _stage_json_artifact(
                    diagnostics,
                    diagnostics_payload,
                    label="diagnostics",
                )
            )
            staged_evidence.append(
                _stage_json_artifact(
                    validation,
                    validation_payload,
                    label="validation",
                )
            )
        # Evidence becomes durable first. For USDZ-to-raw output, the sidecar
        # follows immediately before the generated root, which is the final
        # commit point consumers use to recognize a complete artifact set.
        _promote_staged_artifacts(
            [*staged_evidence, *output_artifacts],
            prebackup_validator=private_target_guard,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _remove_artifact(temp_output)
        for artifact in staged_evidence:
            _remove_artifact(artifact.staged_path)
        if preparation_dir is not None:
            shutil.rmtree(preparation_dir, ignore_errors=True)
        if output_staging_dir is not None:
            shutil.rmtree(output_staging_dir, ignore_errors=True)
        if bound_input_dir is not None:
            try:
                _remove_bound_input_directory(bound_input_dir)
            except Exception as cleanup_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "Bound input cleanup also failed: " + str(cleanup_error)
                    )
                else:
                    raise

    return {
        "joint_rigger_status": "authored",
        "rigged_usd_path": str(output_path),
        "joint_rigger_diagnostics_path": str(diagnostics),
        "joint_rigger_validation_path": str(validation),
        "joint_rigger_warnings": warnings,
        "joint_rigger_errors": [],
        "joint_rigger_candidate_readiness": dict(candidate_readiness or {}),
        "authored_joint_count": len(plans),
        "apply_joint_rigger_skipped": False,
    }


def _validate_direct_artifact_paths(
    *,
    input_path: Path,
    articulation_candidates_path: Path,
    predictions_path: Path | None,
    output_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
) -> None:
    """Reject destructive aliases before any direct-adapter artifact cleanup."""
    paths = [
        ("input_usd_path", input_path),
        ("articulation_candidates_path", articulation_candidates_path),
    ]
    if predictions_path is not None:
        paths.append(("predictions_path", predictions_path))
    paths.extend(
        [
            ("output_usd_path", output_path),
            ("diagnostics_path", diagnostics_path),
            ("validation_path", validation_path),
        ]
    )
    seen: dict[Path, str] = {}
    for label, path in paths:
        resolved = path.expanduser().resolve(strict=False)
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(
                f"{label} must not reference the same path as {previous}: {path}"
            )
        seen[resolved] = label


def _clear_previous_direct_artifacts(
    *,
    input_path: Path,
    articulation_candidates_path: Path,
    predictions_path: Path | None,
    output_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
) -> None:
    """Invalidate every prior artifact before a direct-adapter rerun."""
    from joint_agent.functions.joint_rigger_adapter import (
        _clear_stale_apply_artifacts,
    )

    stable_sidecar = _stable_raw_sidecar(input_path, output_path)
    if stable_sidecar is not None:
        read_inputs = [
            ("input_usd_path", input_path),
            ("articulation_candidates_path", articulation_candidates_path),
        ]
        if predictions_path is not None:
            read_inputs.append(("predictions_path", predictions_path))
        _ensure_read_inputs_outside_sidecar(read_inputs, stable_sidecar)

    _clear_stale_apply_artifacts(
        output_usd_path=output_path,
        diagnostics_path=diagnostics_path,
        validation_path=validation_path,
    )
    if stable_sidecar is not None:
        _clear_stable_sidecar(stable_sidecar)


def _write_readiness_blocked_result(
    *,
    input_path: Path,
    predictions_path: Path | None,
    candidates_path: Path,
    configured_output_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    candidate_readiness: Mapping[str, Any],
    prebackup_validator: Callable[[], None] | None = None,
    replace_existing: bool = True,
) -> dict[str, Any]:
    readiness = dict(candidate_readiness)
    warnings = [str(warning) for warning in readiness.get("warnings", [])]
    errors = [
        "articulation candidates are not ready for Joint Rigger input",
        *[str(error) for error in readiness.get("errors", [])],
    ]
    diagnostics_payload = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "adapter": ADAPTER_NAME,
        "status": "blocked_unready_candidates",
        "input_usd_path": str(input_path),
        "predictions_path": (
            str(predictions_path) if predictions_path is not None else None
        ),
        "articulation_candidates_path": str(candidates_path),
        "output_usd_path": str(configured_output_path),
        "mock_noop": False,
        "authored_joint_count": 0,
        "candidate_readiness": readiness,
        "warnings": warnings,
        "errors": errors,
    }
    validation_payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "adapter": ADAPTER_NAME,
        "status": "blocked_unready_candidates",
        "output_usd_path": None,
        "validation_skipped": True,
        "candidate_readiness": readiness,
        "warnings": warnings,
        "errors": errors,
    }
    artifacts: list[tuple[Path, Mapping[str, Any], str]] = [
        (diagnostics_path, diagnostics_payload, "diagnostics"),
        (validation_path, validation_payload, "validation"),
    ]
    if prebackup_validator is None and replace_existing:
        _write_json_artifacts_transactionally(artifacts)
    else:
        _write_json_artifacts_transactionally(
            artifacts,
            prebackup_validator=prebackup_validator,
            replace_existing=replace_existing,
        )
    return {
        "joint_rigger_status": "blocked_unready_candidates",
        "rigged_usd_path": None,
        "joint_rigger_diagnostics_path": str(diagnostics_path),
        "joint_rigger_validation_path": str(validation_path),
        "joint_rigger_warnings": warnings,
        "joint_rigger_errors": errors,
        "joint_rigger_candidate_readiness": readiness,
        "authored_joint_count": 0,
        "apply_joint_rigger_skipped": True,
    }


def _write_no_joints_authored_result(
    *,
    input_path: Path,
    predictions_path: Path | None,
    candidates_path: Path,
    configured_output_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    candidate_count: int,
    ignored_candidate_count: int,
    candidate_readiness: Mapping[str, Any] | None,
    warnings: list[str],
    prebackup_validator: Callable[[], None] | None = None,
    replace_existing: bool = True,
) -> dict[str, Any]:
    readiness = dict(candidate_readiness or {})
    diagnostics_payload = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "authoring_schema_version": AUTHORING_SCHEMA_VERSION,
        "adapter": ADAPTER_NAME,
        "status": "no_joints_authored",
        "input_usd_path": str(input_path),
        "predictions_path": (
            str(predictions_path) if predictions_path is not None else None
        ),
        "predictions_consumed": False,
        "articulation_candidates_path": str(candidates_path),
        "source_candidate_schema_version": STAGE2_SCHEMA_VERSION,
        "output_usd_path": None,
        "configured_output_usd_path": str(configured_output_path),
        "candidate_count": candidate_count,
        "ready_candidate_count": 0,
        "ignored_candidate_count": ignored_candidate_count,
        "authored_joint_count": 0,
        "authored_edges": [],
        "candidate_readiness": readiness,
        "warnings": warnings,
        "errors": [],
    }
    validation_payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "authoring_schema_version": AUTHORING_SCHEMA_VERSION,
        "adapter": ADAPTER_NAME,
        "status": "no_joints_authored",
        "output_usd_path": None,
        "configured_output_usd_path": str(configured_output_path),
        "validation_skipped": True,
        "validation_skip_reason": "no ready Stage 2 candidate edges were authored",
        "authored_joint_count": 0,
        "candidate_readiness": readiness,
        "warnings": warnings,
        "errors": [],
    }
    artifacts: list[tuple[Path, Mapping[str, Any], str]] = [
        (diagnostics_path, diagnostics_payload, "diagnostics"),
        (validation_path, validation_payload, "validation"),
    ]
    if prebackup_validator is None and replace_existing:
        _write_json_artifacts_transactionally(artifacts)
    else:
        _write_json_artifacts_transactionally(
            artifacts,
            prebackup_validator=prebackup_validator,
            replace_existing=replace_existing,
        )
    return {
        "joint_rigger_status": "no_joints_authored",
        "rigged_usd_path": None,
        "joint_rigger_diagnostics_path": str(diagnostics_path),
        "joint_rigger_validation_path": str(validation_path),
        "joint_rigger_warnings": warnings,
        "joint_rigger_errors": [],
        "joint_rigger_candidate_readiness": readiness,
        "authored_joint_count": 0,
        "apply_joint_rigger_skipped": False,
    }


def _load_and_validate_document(path: Path) -> _Stage2Document:
    try:
        payload = _read_stable_stage2_document(path)
    except (UnicodeError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "msg", str(exc))
        raise ValueError(f"Invalid Stage 2 JSON at {path}: {detail}") from exc
    return _parse_and_validate_document(payload, path=path)


def _candidate_readiness_sha256(
    candidate_readiness: Mapping[str, Any] | None,
) -> str | None:
    if candidate_readiness is None:
        return None
    value = candidate_readiness.get(_CANDIDATE_READINESS_SHA256_FIELD)
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "candidate_readiness.articulation_candidates_sha256 must be a "
            "lowercase SHA-256"
        )
    return value


def _parse_and_validate_document(
    payload: str | bytes,
    *,
    path: Path,
) -> _Stage2Document:
    """Validate exact Stage 2 bytes already bound by a trusted caller."""

    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "msg", str(exc))
        raise ValueError(f"Invalid Stage 2 JSON at {path}: {detail}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Stage 2 candidate document must be a JSON object")
    if raw.get("schema_version") != STAGE2_SCHEMA_VERSION:
        raise ValueError(
            "Stage 2 candidate document schema_version must be exactly "
            f"{STAGE2_SCHEMA_VERSION}"
        )
    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("Stage 2 candidate document candidates must be a list")
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Stage 2 candidate {index} must be a JSON object")
        if candidate.get("schema_version") != STAGE2_SCHEMA_VERSION:
            raise ValueError(
                f"Stage 2 candidate {index} schema_version must be exactly "
                f"{STAGE2_SCHEMA_VERSION}"
            )
    try:
        document = _Stage2Document.model_validate(raw, strict=True)
    except ValidationError as exc:
        raise ValueError(f"Invalid full Stage 2 candidate document: {exc}") from exc

    candidate_ids = [candidate.candidate_id for candidate in document.candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        duplicate_ids = sorted(
            candidate_id
            for candidate_id, count in Counter(candidate_ids).items()
            if count > 1
        )
        raise ValueError(
            "Stage 2 candidate_id values must be unique; duplicates: "
            + ", ".join(duplicate_ids)
        )

    ready_count = sum(
        candidate.review_status == _READY_STATUS for candidate in document.candidates
    )
    review_count = len(document.candidates) - ready_count
    expected_counts = {
        "candidate_count": len(document.candidates),
        "ready_candidate_count": ready_count,
        "review_required_candidate_count": review_count,
    }
    for field, expected in expected_counts.items():
        actual = getattr(document.summary, field)
        if actual != expected:
            raise ValueError(
                f"Stage 2 summary.{field}={actual} does not match candidates "
                f"({expected})"
            )
    return document


def _read_stable_stage2_document(path: Path) -> str:
    """Read one unchanged regular candidate file without following or blocking."""

    def state(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    try:
        expected = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Articulation candidates file not found: {path}"
        ) from exc
    if not stat.S_ISREG(expected.st_mode):
        raise FileNotFoundError(f"Articulation candidates file not found: {path}")
    if expected.st_size > _MAX_STAGE2_DOCUMENT_BYTES:
        raise ValueError(
            "Stage 2 candidate document exceeds the "
            f"{_MAX_STAGE2_DOCUMENT_BYTES}-byte limit: {path}"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Articulation candidates file not found: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or state(opened) != state(expected):
            raise RuntimeError(
                f"Stage 2 candidate document changed before it was opened: {path}"
            )
        payload = bytearray()
        offset = 0
        while offset < opened.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, opened.st_size - offset),
                offset,
            )
            if not chunk:
                raise RuntimeError(
                    f"Stage 2 candidate document changed while read: {path}"
                )
            payload.extend(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, offset):
            raise RuntimeError(f"Stage 2 candidate document grew while read: {path}")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if state(after) != state(opened) or state(current) != state(opened):
            raise RuntimeError(f"Stage 2 candidate document changed while read: {path}")
    finally:
        os.close(descriptor)
    return bytes(payload).decode("utf-8")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in Stage 2 document: {key}")
        result[key] = value
    return result


def _preflight_stage_and_edges(
    input_path: Path,
    document: _Stage2Document,
) -> tuple[Any, list[_JointPlan], dict[str, Any]]:
    from pxr import Sdf, UsdGeom

    stage = _open_stage(input_path, label="input USD")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ValueError("Input USD must define a valid defaultPrim")
    default_path = default_prim.GetPath()
    if not default_path.IsAbsoluteRootOrPrimPath() or default_path.IsAbsoluteRootPath():
        raise ValueError(f"Input USD defaultPrim path is invalid: {default_path}")
    if default_prim.IsInstance() or default_prim.IsInstanceProxy():
        raise ValueError(
            "Input USD defaultPrim cannot be an instance for no-reshape joint authoring"
        )

    joints_scope_path = default_path.AppendChild("Joints")
    existing_scope = stage.GetPrimAtPath(joints_scope_path)
    if (
        existing_scope
        and existing_scope.IsValid()
        and (existing_scope.IsInstance() or existing_scope.IsInstanceProxy())
    ):
        raise ValueError(
            f"Cannot author joints below instance path: {joints_scope_path}"
        )

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError(
            f"Input USD metersPerUnit must be positive; got {meters_per_unit}"
        )

    plans: list[_JointPlan] = []
    planned_paths: set[str] = set()
    for candidate in document.candidates:
        if candidate.review_status != _READY_STATUS:
            continue
        plan = _preflight_ready_candidate(
            stage=stage,
            candidate=candidate,
            joints_scope_path=joints_scope_path,
            meters_per_unit=meters_per_unit,
        )
        if plan.joint_path in planned_paths:
            raise ValueError(
                f"Deterministic Stage 2 joint path collision: {plan.joint_path}"
            )
        planned_paths.add(plan.joint_path)
        if stage.GetPrimAtPath(Sdf.Path(plan.joint_path)).IsValid():
            raise ValueError(
                f"Refusing to overwrite existing prim at joint path: {plan.joint_path}"
            )
        plans.append(plan)

    input_snapshot = {
        "default_prim_path": str(default_path),
        "prim_paths": sorted(str(prim.GetPath()) for prim in stage.Traverse()),
        "applied_schemas": {
            str(prim.GetPath()): list(prim.GetAppliedSchemas())
            for prim in stage.Traverse()
        },
        "instanceable_flags": {
            str(prim.GetPath()): bool(prim.IsInstanceable())
            for prim in stage.Traverse()
        },
    }
    return stage, plans, input_snapshot


def _preflight_ready_candidate(
    *,
    stage: Any,
    candidate: Stage2ArticulationCandidate,
    joints_scope_path: Any,
    meters_per_unit: float,
) -> _JointPlan:
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom

    prefix = f"Stage 2 candidate {candidate.candidate_id}"
    if candidate.unresolved_reason_codes:
        raise ValueError(f"{prefix} is ready but has unresolved_reason_codes")
    if candidate.unresolved_questions:
        raise ValueError(f"{prefix} is ready but has unresolved_questions")
    if candidate.motion_type not in _SUPPORTED_JOINT_TYPES:
        raise ValueError(
            f"{prefix} has unsupported motion_type: {candidate.motion_type}"
        )
    if candidate.joint_type_hint != candidate.motion_type:
        raise ValueError(f"{prefix} motion_type and joint_type_hint must match exactly")
    if len(candidate.moving_part_prims) != 1:
        raise ValueError(f"{prefix} must resolve exactly one moving body/body1")
    body0 = candidate.fixed_parent_prim
    body1 = candidate.moving_part_prims[0]
    if body0 is None:
        raise ValueError(f"{prefix} is missing exact fixed_parent_prim/body0")
    _validate_absolute_prim_path(body0, label=f"{prefix} body0", sdf=Sdf)
    _validate_absolute_prim_path(body1, label=f"{prefix} body1", sdf=Sdf)
    if body0 == body1:
        raise ValueError(f"{prefix} body0 and body1 must be different paths")
    if candidate.parent_resolution_source in {"structural_fallback", "unresolved"}:
        raise ValueError(
            f"{prefix} cannot use parent fallback source "
            f"{candidate.parent_resolution_source}"
        )

    required_sources = (
        "motion_type",
        "axis_hint",
        "motion_axis_world",
        "fixed_parent_prim",
    )
    for field in required_sources:
        source = candidate.field_sources.get(field)
        if source is None:
            raise ValueError(f"{prefix} is missing field_sources.{field}")
        if source in _DISALLOWED_FALLBACK_SOURCES:
            raise ValueError(
                f"{prefix} field_sources.{field} cannot use fallback source {source}"
            )

    expected_world_axis = _AXIS_VECTORS.get(candidate.axis_hint)
    if expected_world_axis is None:
        raise ValueError(f"{prefix} requires an explicit coordinate-axis axis_hint")
    actual_world_axis = candidate.motion_axis_world
    if actual_world_axis is None or tuple(actual_world_axis) != expected_world_axis:
        raise ValueError(
            f"{prefix} motion_axis_world must exactly match axis_hint "
            f"{candidate.axis_hint}"
        )

    _preflight_topology_evidence(
        candidate,
        body0=body0,
        body1=body1,
        prefix=prefix,
        sdf=Sdf,
    )

    body0_prim = stage.GetPrimAtPath(body0)
    body1_prim = stage.GetPrimAtPath(body1)
    for label, prim, path in (
        ("body0", body0_prim, body0),
        ("body1", body1_prim, body1),
    ):
        if not prim or not prim.IsValid() or not prim.IsActive():
            raise ValueError(f"{prefix} {label} path does not resolve: {path}")
        if prim.IsInstanceProxy():
            raise ValueError(
                f"{prefix} {label} is an instance proxy; no-reshape authoring "
                f"cannot target it: {path}"
            )
        if not UsdGeom.Xformable(prim):
            raise ValueError(f"{prefix} {label} is not transformable: {path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    body0_xform = xform_cache.GetLocalToWorldTransform(body0_prim)
    body1_xform = xform_cache.GetLocalToWorldTransform(body1_prim)
    _validate_invertible_transform(body0_xform, label=f"{prefix} body0")
    _validate_invertible_transform(body1_xform, label=f"{prefix} body1")
    anchor_world_vec = body1_xform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    local_pos0 = body0_xform.GetInverse().Transform(anchor_world_vec)
    local_pos1 = body1_xform.GetInverse().Transform(anchor_world_vec)

    axis_token = _AXIS_TOKENS[candidate.axis_hint[-1]]
    base_axis = _AXIS_VECTORS[candidate.axis_hint[-1]]
    local_axis0 = _normalized_direction(
        body0_xform.GetInverse().TransformDir(Gf.Vec3d(*expected_world_axis)),
        label=f"{prefix} body0 local axis",
    )
    local_axis1 = _normalized_direction(
        body1_xform.GetInverse().TransformDir(Gf.Vec3d(*expected_world_axis)),
        label=f"{prefix} body1 local axis",
    )
    local_rot0 = _rotation_tuple(Gf.Rotation(Gf.Vec3d(*base_axis), local_axis0))
    local_rot1 = _rotation_tuple(Gf.Rotation(Gf.Vec3d(*base_axis), local_axis1))

    lower_limit, upper_limit, authored_limit_unit = _preflight_limits(
        candidate,
        meters_per_unit=meters_per_unit,
    )
    hash_payload = json.dumps(
        {
            "candidate_id": candidate.candidate_id,
            "body0": body0,
            "body1": body1,
            "joint_type": candidate.motion_type,
            "motion_axis_world": expected_world_axis,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()[:12]
    valid_candidate_id = Tf.MakeValidIdentifier(candidate.candidate_id) or "candidate"
    joint_name = f"{valid_candidate_id}_{digest}"
    joint_path = str(joints_scope_path.AppendChild(joint_name))

    return _JointPlan(
        candidate=candidate,
        joint_path=joint_path,
        body0=body0,
        body1=body1,
        joint_type=candidate.motion_type,
        axis_token=axis_token,
        motion_axis_world=expected_world_axis,
        local_pos0=_vec3_tuple(local_pos0),
        local_pos1=_vec3_tuple(local_pos1),
        local_rot0=local_rot0,
        local_rot1=local_rot1,
        anchor_world=_vec3_tuple(anchor_world_vec),
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        authored_limit_unit=authored_limit_unit,
    )


def _preflight_topology_evidence(
    candidate: Stage2ArticulationCandidate,
    *,
    body0: str,
    body1: str,
    prefix: str,
    sdf: Any,
) -> None:
    """Require structured evidence to support each accepted topology field."""
    if not candidate.axis_evidence:
        raise ValueError(f"{prefix} requires structured axis_evidence")
    axis_sources = {item.source for item in candidate.axis_evidence}
    disallowed_axis_sources = sorted(axis_sources & _DISALLOWED_FALLBACK_SOURCES)
    if disallowed_axis_sources:
        raise ValueError(
            f"{prefix} axis_evidence cannot use fallback sources: "
            + ", ".join(disallowed_axis_sources)
        )
    required_axis_sources = {
        candidate.field_sources["axis_hint"],
        candidate.field_sources["motion_axis_world"],
    }
    unsupported_axis_sources = []
    for source in required_axis_sources:
        source_items = [
            item for item in candidate.axis_evidence if item.source == source
        ]
        if not source_items or any(
            not _axis_evidence_value_supports_hint(item.value, candidate.axis_hint)
            for item in source_items
        ):
            unsupported_axis_sources.append(source)
    unsupported_axis_sources.sort()
    if unsupported_axis_sources:
        raise ValueError(
            f"{prefix} axis_evidence does not support field_sources and axis_hint "
            f"{candidate.axis_hint}: " + ", ".join(unsupported_axis_sources)
        )

    if not candidate.connectivity_evidence:
        raise ValueError(f"{prefix} requires structured connectivity_evidence")
    connectivity_sources = {item.source for item in candidate.connectivity_evidence}
    disallowed_connectivity_sources = sorted(
        connectivity_sources & _DISALLOWED_FALLBACK_SOURCES
    )
    if disallowed_connectivity_sources:
        raise ValueError(
            f"{prefix} connectivity_evidence cannot use fallback sources: "
            + ", ".join(disallowed_connectivity_sources)
        )
    typed_items_valid = all(
        _typed_connectivity_item_is_valid(
            item,
            body0=body0,
            body1=body1,
            sdf=sdf,
        )
        for item in candidate.connectivity_evidence
    )
    parent_source = candidate.field_sources["fixed_parent_prim"]
    parent_source_items = [
        item for item in candidate.connectivity_evidence if item.source == parent_source
    ]
    if not typed_items_valid or not _required_source_connectivity_is_valid(
        parent_source_items,
        body0=body0,
        body1=body1,
        sdf=sdf,
    ):
        raise ValueError(
            f"{prefix} connectivity_evidence does not support exact body0/body1 "
            f"with field_sources.fixed_parent_prim={parent_source}"
        )


def _required_source_connectivity_is_valid(
    items: list[Any],
    *,
    body0: str,
    body1: str,
    sdf: Any,
) -> bool:
    """Require one edge and reject ambiguous role-less v0 source evidence."""
    has_edge = False
    for item in items:
        if item.connectivity_role == _BODY0_BODY1_EDGE_ROLE:
            has_edge = True
            continue
        if item.connectivity_role is not None:
            continue

        legacy_role = _roleless_v0_connectivity_role(
            item,
            body0=body0,
            body1=body1,
            sdf=sdf,
        )
        if legacy_role is None:
            return False
        if legacy_role == _BODY0_BODY1_EDGE_ROLE:
            has_edge = True
    return has_edge


def _roleless_v0_connectivity_role(
    item: Any,
    *,
    body0: str,
    body1: str,
    sdf: Any,
) -> str | None:
    """Classify only the two bounded connectivity shapes emitted before roles."""
    raw_paths = item.prim_paths
    if not raw_paths or len(raw_paths) != len(set(raw_paths)):
        return None
    paths = [sdf.Path(path) for path in raw_paths]
    if any(not _is_absolute_non_root_prim_path(path) for path in paths):
        return None

    path_values = set(raw_paths)
    endpoints = {body0, body1}
    if item.value == f"{body0}->{body1}":
        if len(raw_paths) == 2 and path_values == endpoints:
            return _BODY0_BODY1_EDGE_ROLE
        return None

    if item.value == body0:
        if len(raw_paths) == 2 and path_values == endpoints:
            return _BODY0_BODY1_EDGE_ROLE
        if len(raw_paths) == 3 and endpoints < path_values:
            [alias] = path_values - endpoints
            if _is_strict_ancestor_path(alias, body0, sdf=sdf):
                return _BODY0_BODY1_EDGE_ROLE
        return None

    if item.value == body1:
        if raw_paths == [body1]:
            return _BODY1_OWNERSHIP_ROLE
        if len(raw_paths) == 2 and body1 in path_values:
            [alias] = path_values - {body1}
            if _paths_are_strictly_related(alias, body1, sdf=sdf):
                return _BODY1_OWNERSHIP_ROLE
    return None


def _is_absolute_non_root_prim_path(path: Any) -> bool:
    """Return whether ``path`` names an absolute, non-root USD prim."""
    return path.IsAbsolutePath() and path.IsPrimPath() and not path.IsAbsoluteRootPath()


def _is_strict_ancestor_path(candidate: str, anchor: str, *, sdf: Any) -> bool:
    """Return whether ``candidate`` is a strict ancestor of ``anchor``."""
    candidate_path = sdf.Path(candidate)
    anchor_path = sdf.Path(anchor)
    return bool(candidate_path != anchor_path and anchor_path.HasPrefix(candidate_path))


def _paths_are_strictly_related(left: str, right: str, *, sdf: Any) -> bool:
    """Return whether either prim path is a strict ancestor of the other."""
    return _is_strict_ancestor_path(left, right, sdf=sdf) or _is_strict_ancestor_path(
        right,
        left,
        sdf=sdf,
    )


def _typed_connectivity_item_is_valid(
    item: Any,
    *,
    body0: str,
    body1: str,
    sdf: Any,
) -> bool:
    """Validate explicit roles while leaving legacy role-less logic unchanged."""
    if item.connectivity_role is None:
        return True
    if item.connectivity_role == _BODY0_BODY1_EDGE_ROLE:
        return bool(item.value == body0 and item.prim_paths == [body0, body1])
    if item.connectivity_role == _BODY1_OWNERSHIP_ROLE:
        return _connectivity_item_confirms_body1(item, body1=body1, sdf=sdf)
    return _connectivity_item_confirms_endpoint_canonicalization(
        item,
        body0=body0,
        body1=body1,
        sdf=sdf,
    )


def _connectivity_item_confirms_body1(
    item: Any,
    *,
    body1: str,
    sdf: Any,
) -> bool:
    """Distinguish endpoint ownership evidence from a body0/body1 edge claim."""
    if item.value != body1 or body1 not in item.prim_paths:
        return False
    body1_path = sdf.Path(body1)
    return all(
        (prim_path := sdf.Path(path)).IsAbsoluteRootOrPrimPath()
        and not prim_path.IsAbsoluteRootPath()
        and prim_path.HasPrefix(body1_path)
        for path in item.prim_paths
    )


def _connectivity_item_confirms_endpoint_canonicalization(
    item: Any,
    *,
    body0: str,
    body1: str,
    sdf: Any,
) -> bool:
    """Validate one ordered, absolute wrapper-to-selected-endpoint lineage."""
    if len(item.prim_paths) != 2:
        return False
    wrapper, leaf = item.prim_paths
    if item.value != leaf or leaf not in {body0, body1} or wrapper == leaf:
        return False
    wrapper_path = sdf.Path(wrapper)
    leaf_path = sdf.Path(leaf)
    if any(
        not path.IsAbsolutePath() or not path.IsPrimPath()
        for path in (wrapper_path, leaf_path)
    ):
        return False
    return bool(leaf_path.HasPrefix(wrapper_path))


def _axis_evidence_value_supports_hint(value: str | None, axis_hint: str) -> bool:
    normalized_value = value or ""
    return (
        normalized_value == axis_hint
        or normalized_value.rsplit("->", maxsplit=1)[-1] == axis_hint
    )


def _validate_absolute_prim_path(value: str, *, label: str, sdf: Any) -> None:
    path = sdf.Path(value)
    if not path.IsAbsoluteRootOrPrimPath() or path.IsAbsoluteRootPath():
        raise ValueError(f"{label} must be an exact absolute USD prim path: {value}")


def _validate_invertible_transform(matrix: Any, *, label: str) -> None:
    determinant = float(matrix.GetDeterminant())
    if not math.isfinite(determinant) or math.isclose(determinant, 0.0, abs_tol=1e-12):
        raise ValueError(f"{label} transform is not invertible")


def _normalized_direction(vector: Any, *, label: str) -> Any:
    length = float(vector.GetLength())
    if not math.isfinite(length) or math.isclose(length, 0.0, abs_tol=1e-12):
        raise ValueError(f"{label} cannot be normalized")
    return vector / length


def _rotation_tuple(rotation: Any) -> tuple[float, tuple[float, float, float]]:
    quaternion = rotation.GetQuat()
    imaginary = quaternion.GetImaginary()
    return (
        float(quaternion.GetReal()),
        (float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
    )


def _vec3_tuple(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _preflight_limits(
    candidate: Stage2ArticulationCandidate,
    *,
    meters_per_unit: float,
) -> tuple[float | None, float | None, str | None]:
    prefix = f"Stage 2 candidate {candidate.candidate_id}"
    has_limit = candidate.lower_limit is not None or candidate.upper_limit is not None
    for label, value in (
        ("lower_limit", candidate.lower_limit),
        ("upper_limit", candidate.upper_limit),
    ):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{prefix} {label} must be finite")
    if (
        candidate.lower_limit is not None
        and candidate.upper_limit is not None
        and candidate.lower_limit > candidate.upper_limit
    ):
        raise ValueError(f"{prefix} lower_limit must not exceed upper_limit")
    if candidate.limit_readiness == "source_backed":
        if not has_limit:
            raise ValueError(f"{prefix} source_backed limits have no numeric value")
        if candidate.limit_source not in _SOURCE_BACKED_LIMIT_SOURCES:
            raise ValueError(
                f"{prefix} limit_source is not accepted source-backed evidence: "
                f"{candidate.limit_source}"
            )
        if not candidate.limit_evidence:
            raise ValueError(f"{prefix} source_backed limits require limit_evidence")
        mismatched_sources = sorted(
            {
                item.source
                for item in candidate.limit_evidence
                if item.source != candidate.limit_source
            }
        )
        if mismatched_sources:
            raise ValueError(
                f"{prefix} limit_evidence sources must match limit_source "
                f"{candidate.limit_source}; got {', '.join(mismatched_sources)}"
            )
        if candidate.motion_type == "spherical":
            raise ValueError(
                f"{prefix} has scalar spherical limits; multi-axis cone limits "
                "are outside the Stage 2 candidate-edge contract"
            )
        expected_unit = "degrees" if candidate.motion_type == "revolute" else "meters"
        if candidate.limit_unit != expected_unit:
            raise ValueError(
                f"{prefix} {candidate.motion_type} limits must use {expected_unit}"
            )
        if candidate.motion_type == "prismatic":
            return (
                _optional_divide(candidate.lower_limit, meters_per_unit),
                _optional_divide(candidate.upper_limit, meters_per_unit),
                "stage_units",
            )
        return candidate.lower_limit, candidate.upper_limit, "degrees"

    if has_limit:
        raise ValueError(
            f"{prefix} has numeric limits without limit_readiness=source_backed"
        )
    return None, None, None


def _optional_divide(value: float | None, divisor: float) -> float | None:
    return None if value is None else float(value) / divisor


def _temporary_output_path(output_path: Path) -> Path:
    _validate_output_extension(output_path)
    suffix = output_path.suffix.lower()
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=suffix,
        delete=False,
    ) as stream:
        temp_path = Path(stream.name)
    temp_path.unlink()
    return temp_path


def _validate_output_extension(output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix not in _RAW_USD_EXTENSIONS and suffix != ".usdz":
        raise ValueError(
            "stage2_candidate_edges output must use .usd, .usda, .usdc, or .usdz; "
            f"got {output_path}"
        )


def _staged_output_path(
    input_path: Path,
    output_path: Path,
) -> tuple[Path, Path | None]:
    if not _is_usdz_to_raw(input_path, output_path):
        return _temporary_output_path(output_path), None
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.stage-",
        )
    ).resolve()
    return staging_dir / output_path.name, staging_dir


def _output_promotion_artifacts(
    *,
    input_path: Path,
    staged_output: Path,
    output_path: Path,
    output_staging_dir: Path | None,
    replace_existing: bool = True,
) -> list[_StagedArtifact]:
    if not _is_usdz_to_raw(input_path, output_path):
        return [
            _StagedArtifact(
                staged_path=staged_output,
                target_path=output_path,
                label="generated USD",
                replace_existing=replace_existing,
            )
        ]
    if output_staging_dir is None:
        raise RuntimeError("USDZ-to-raw output staging directory is missing")

    staged_sidecar = output_staging_dir / f"{output_path.stem}_assets"
    final_sidecar = output_path.parent / staged_sidecar.name
    if not staged_sidecar.is_dir() or staged_sidecar.is_symlink():
        raise RuntimeError(
            f"USDZ-to-raw staged sidecar is missing or invalid: {staged_sidecar}"
        )
    return [
        _StagedArtifact(
            staged_path=staged_sidecar,
            target_path=final_sidecar,
            label="USDZ raw-output sidecar",
            replace_existing=replace_existing,
        ),
        _StagedArtifact(
            staged_path=staged_output,
            target_path=output_path,
            label="generated USD",
            replace_existing=replace_existing,
        ),
    ]


def _private_authoring_targets(
    *,
    input_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
) -> tuple[Path, ...]:
    """Return every bridge-private path that must remain absent until commit."""

    paths = [diagnostics_path, validation_path]
    stable_sidecar = _stable_raw_sidecar(input_path, output_path)
    if stable_sidecar is not None:
        paths.append(stable_sidecar)
    paths.append(output_path)
    return tuple(paths)


def _require_private_targets_absent(paths: tuple[Path, ...]) -> None:
    """Reject any entry raced into a bridge-reserved private target name."""

    for path in paths:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise FileExistsError(
            f"Bridge-private authoring target was unexpectedly created: {path}"
        )


def _is_usdz_to_raw(input_path: Path, output_path: Path) -> bool:
    return (
        input_path.suffix.lower() == ".usdz"
        and output_path.suffix.lower() in _RAW_USD_EXTENSIONS
    )


def _stable_raw_sidecar(input_path: Path, output_path: Path) -> Path | None:
    if not _is_usdz_to_raw(input_path, output_path):
        return None
    return output_path.parent / f"{output_path.stem}_assets"


def _ensure_read_inputs_outside_sidecar(
    read_inputs: list[tuple[str, Path]],
    sidecar_path: Path,
) -> None:
    sidecar_resolved = sidecar_path.resolve(strict=False)
    for label, path in read_inputs:
        try:
            path.resolve(strict=False).relative_to(sidecar_resolved)
        except ValueError:
            continue
        raise RuntimeError(
            "Refusing to replace USDZ sidecar because configured read input "
            f"{label} is inside the sidecar directory: {path}"
        )


def _clear_stable_sidecar(sidecar_path: Path) -> None:
    if not sidecar_path.exists() and not sidecar_path.is_symlink():
        return
    if sidecar_path.is_symlink() or not sidecar_path.is_dir():
        raise RuntimeError(
            f"Refusing to replace invalid USDZ sidecar artifact: {sidecar_path}"
        )
    shutil.rmtree(sidecar_path)


def _prepare_editable_output(
    input_path: Path,
    temp_output: Path,
    *,
    output_anchor_path: Path | None = None,
) -> tuple[Path, Path | None]:
    # Reuse the adapter boundary's composition-aware copy/extract machinery.
    # This import is intentionally local so the adapter can import this module
    # without a circular module-initialization dependency.
    from joint_agent.functions.joint_rigger_adapter import _prepare_usd_for_handoff

    return cast(
        tuple[Path, Path | None],
        _prepare_usd_for_handoff(
            input_path,
            temp_output,
            output_anchor_path=output_anchor_path,
            prune_sidecar_members=True,
        ),
    )


def _package_usdz(editable_path: Path, temp_output: Path) -> None:
    from joint_agent.functions.joint_rigger_adapter import _package_usdz_for_handoff

    _package_usdz_for_handoff(editable_path, temp_output)


def _open_stage(path: Path, *, label: str) -> Any:
    from pxr import Usd

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open {label}: {path}")
    return stage


def _author_plans(stage: Any, plans: list[_JointPlan]) -> list[dict[str, Any]]:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    default_path = stage.GetDefaultPrim().GetPath()
    joints_scope_path = default_path.AppendChild("Joints")
    if not stage.GetPrimAtPath(joints_scope_path).IsValid():
        UsdGeom.Scope.Define(stage, joints_scope_path)

    schemas = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
    }
    authored_edges: list[dict[str, Any]] = []
    for plan in plans:
        joint = schemas[plan.joint_type].Define(stage, Sdf.Path(plan.joint_path))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(plan.body0)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(plan.body1)])
        joint.CreateAxisAttr(plan.axis_token)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*plan.local_pos0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*plan.local_pos1))
        joint.CreateLocalRot0Attr(_quatf(plan.local_rot0, gf=Gf))
        joint.CreateLocalRot1Attr(_quatf(plan.local_rot1, gf=Gf))
        if plan.lower_limit is not None:
            joint.CreateLowerLimitAttr(float(plan.lower_limit))
        if plan.upper_limit is not None:
            joint.CreateUpperLimitAttr(float(plan.upper_limit))

        edge_record = _authored_edge_record(plan)
        prim = joint.GetPrim()
        prim.SetCustomDataByKey("jointAgent:candidateId", plan.candidate.candidate_id)
        prim.SetCustomDataByKey(
            "jointAgent:sourceSchemaVersion",
            STAGE2_SCHEMA_VERSION,
        )
        prim.SetCustomDataByKey(
            "jointAgent:fieldProvenance",
            json.dumps(edge_record["field_provenance"], sort_keys=True),
        )
        prim.SetCustomDataByKey(
            "jointAgent:copiedEvidence",
            json.dumps(edge_record["copied_evidence"], sort_keys=True),
        )
        authored_edges.append(edge_record)
    return authored_edges


def _quatf(
    value: tuple[float, tuple[float, float, float]],
    *,
    gf: Any,
) -> Any:
    real, imaginary = value
    return gf.Quatf(float(real), gf.Vec3f(*imaginary))


def _authored_edge_record(plan: _JointPlan) -> dict[str, Any]:
    candidate = plan.candidate
    connectivity_sources = sorted(
        {item.source for item in candidate.connectivity_evidence}
    )
    field_provenance: dict[str, Any] = {
        "joint_type": {
            "candidate_field": "motion_type",
            "source": candidate.field_sources["motion_type"],
        },
        "body0": {
            "candidate_field": "fixed_parent_prim",
            "source": candidate.field_sources["fixed_parent_prim"],
        },
        "body1": {
            "candidate_field": "moving_part_prims[0]",
            "evidence_sources": connectivity_sources,
            "source_prediction_ids": list(candidate.source_prediction_ids),
        },
        "axis": {
            "candidate_fields": ["axis_hint", "motion_axis_world"],
            "axis_hint_source": candidate.field_sources["axis_hint"],
            "motion_axis_world_source": candidate.field_sources["motion_axis_world"],
        },
        "local_frames": {
            "source": candidate.field_sources["motion_axis_world"],
            "derivation": "world_axis_transformed_into_each_body_local_frame",
        },
        "anchor": {
            "source": "inferred_body1_world_origin",
            "derivation": "shared_anchor_at_body1_world_origin",
        },
        "limits": {
            "source": candidate.limit_source,
            "readiness": candidate.limit_readiness,
            "source_unit": candidate.limit_unit,
            "authored_unit": plan.authored_limit_unit,
        },
    }
    copied_evidence = {
        "evidence": candidate.evidence,
        "source_prediction_ids": list(candidate.source_prediction_ids),
        "axis_evidence": [
            item.model_dump(mode="json") for item in candidate.axis_evidence
        ],
        "connectivity_evidence": [
            item.model_dump(mode="json") for item in candidate.connectivity_evidence
        ],
        "limit_evidence": [
            item.model_dump(mode="json") for item in candidate.limit_evidence
        ],
    }
    return {
        "candidate_id": candidate.candidate_id,
        "joint_path": plan.joint_path,
        "joint_type": plan.joint_type,
        "body0": plan.body0,
        "body1": plan.body1,
        "axis_token": plan.axis_token,
        "motion_axis_world": list(plan.motion_axis_world),
        "local_pos0": list(plan.local_pos0),
        "local_pos1": list(plan.local_pos1),
        "local_rot0": [plan.local_rot0[0], *plan.local_rot0[1]],
        "local_rot1": [plan.local_rot1[0], *plan.local_rot1[1]],
        "anchor_world": list(plan.anchor_world),
        "lower_limit": plan.lower_limit,
        "upper_limit": plan.upper_limit,
        "field_provenance": field_provenance,
        "copied_evidence": copied_evidence,
    }


def _validate_authored_output(
    path: Path,
    plans: list[_JointPlan],
    *,
    input_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = _open_stage(path, label="authored output USD")
    if str(stage.GetDefaultPrim().GetPath()) != input_snapshot["default_prim_path"]:
        raise RuntimeError("Authored output changed the source defaultPrim path")

    per_joint: list[dict[str, Any]] = []
    schema_by_type = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "spherical": UsdPhysics.SphericalJoint,
    }
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for plan in plans:
        prim = stage.GetPrimAtPath(plan.joint_path)
        if (
            not prim
            or not prim.IsValid()
            or not prim.IsA(schema_by_type[plan.joint_type])
        ):
            raise RuntimeError(
                f"Authored output is missing expected {plan.joint_type} joint: "
                f"{plan.joint_path}"
            )
        joint = schema_by_type[plan.joint_type](prim)
        _require_single_target(joint.GetBody0Rel(), plan.body0, field="body0")
        _require_single_target(joint.GetBody1Rel(), plan.body1, field="body1")
        if joint.GetAxisAttr().Get() != plan.axis_token:
            raise RuntimeError(f"Authored joint axis token mismatch: {plan.joint_path}")

        body0 = stage.GetPrimAtPath(plan.body0)
        body1 = stage.GetPrimAtPath(plan.body1)
        body0_xform = xform_cache.GetLocalToWorldTransform(body0)
        body1_xform = xform_cache.GetLocalToWorldTransform(body1)
        anchor0_world = body0_xform.Transform(
            Gf.Vec3d(*_vec3_tuple(joint.GetLocalPos0Attr().Get()))
        )
        anchor1_world = body1_xform.Transform(
            Gf.Vec3d(*_vec3_tuple(joint.GetLocalPos1Attr().Get()))
        )
        _require_close_vector(anchor0_world, plan.anchor_world, label="body0 anchor")
        _require_close_vector(anchor1_world, plan.anchor_world, label="body1 anchor")

        base_axis = Gf.Vec3d(*_AXIS_VECTORS[plan.axis_token.lower()])
        local_axis0 = Gf.Rotation(joint.GetLocalRot0Attr().Get()).TransformDir(
            base_axis
        )
        local_axis1 = Gf.Rotation(joint.GetLocalRot1Attr().Get()).TransformDir(
            base_axis
        )
        world_axis0 = body0_xform.TransformDir(local_axis0)
        world_axis1 = body1_xform.TransformDir(local_axis1)
        _require_close_vector(world_axis0, plan.motion_axis_world, label="body0 axis")
        _require_close_vector(world_axis1, plan.motion_axis_world, label="body1 axis")

        if plan.joint_type == "spherical":
            _require_authored_limit(
                joint.GetConeAngle0LimitAttr(),
                None,
                "cone angle 0",
            )
            _require_authored_limit(
                joint.GetConeAngle1LimitAttr(),
                None,
                "cone angle 1",
            )
        else:
            _require_authored_limit(
                joint.GetLowerLimitAttr(),
                plan.lower_limit,
                "lower",
            )
            _require_authored_limit(
                joint.GetUpperLimitAttr(),
                plan.upper_limit,
                "upper",
            )
        per_joint.append(
            {
                "candidate_id": plan.candidate.candidate_id,
                "joint_path": plan.joint_path,
                "status": "passed",
                "checks": {
                    "joint_type": "pass",
                    "body0": "pass",
                    "body1": "pass",
                    "signed_world_axis": "pass",
                    "shared_anchor": "pass",
                    "source_backed_limits": "pass",
                },
            }
        )

    output_applied_schemas = {
        str(prim.GetPath()): list(prim.GetAppliedSchemas()) for prim in stage.Traverse()
    }
    for prim_path, source_schemas in input_snapshot["applied_schemas"].items():
        if output_applied_schemas.get(prim_path) != source_schemas:
            raise RuntimeError(
                "stage2_candidate_edges changed applied physics schemas on "
                f"source prim: {prim_path}"
            )
    for plan in plans:
        if output_applied_schemas.get(plan.joint_path):
            raise RuntimeError(
                f"stage2_candidate_edges added an applied API schema to {plan.joint_path}"
            )

    for prim_path, source_instanceable in input_snapshot["instanceable_flags"].items():
        output_prim = stage.GetPrimAtPath(prim_path)
        if not output_prim or bool(output_prim.IsInstanceable()) != source_instanceable:
            raise RuntimeError(
                "stage2_candidate_edges changed the instanceable flag on source "
                f"prim: {prim_path}"
            )

    allowed_new_paths = {plan.joint_path for plan in plans}
    if plans:
        allowed_new_paths.add(f"{input_snapshot['default_prim_path']}/Joints")
    output_paths = {str(prim.GetPath()) for prim in stage.Traverse()}
    unexpected_new_paths = sorted(
        output_paths - set(input_snapshot["prim_paths"]) - allowed_new_paths
    )
    if unexpected_new_paths:
        raise RuntimeError(
            "stage2_candidate_edges authored unexpected prim paths: "
            + ", ".join(unexpected_new_paths)
        )

    return {
        "joint_count": len(plans),
        "joint_graph_fidelity": "pass",
        "source_prim_paths_preserved": True,
        "applied_physics_schemas_unchanged": True,
        "instanceable_flags_preserved": True,
        "forbidden_schema_authoring": False,
        "per_joint": per_joint,
    }


def _require_single_target(relationship: Any, expected: str, *, field: str) -> None:
    targets = [str(target) for target in relationship.GetTargets()]
    if targets != [expected]:
        raise RuntimeError(
            f"Authored joint {field} targets do not match: {targets} != [{expected}]"
        )


def _require_close_vector(actual: Any, expected: Any, *, label: str) -> None:
    actual_values = _vec3_tuple(actual)
    expected_values = _vec3_tuple(expected)
    actual_length = math.sqrt(sum(value * value for value in actual_values))
    expected_length = math.sqrt(sum(value * value for value in expected_values))
    if actual_length > 0 and expected_length > 0 and "axis" in label:
        actual_values = (
            actual_values[0] / actual_length,
            actual_values[1] / actual_length,
            actual_values[2] / actual_length,
        )
        expected_values = (
            expected_values[0] / expected_length,
            expected_values[1] / expected_length,
            expected_values[2] / expected_length,
        )
    if any(
        not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-5)
        for left, right in zip(actual_values, expected_values, strict=True)
    ):
        raise RuntimeError(
            f"Authored {label} mismatch: {actual_values} != {expected_values}"
        )


def _require_authored_limit(attribute: Any, expected: float | None, label: str) -> None:
    authored = attribute.HasAuthoredValueOpinion()
    if expected is None:
        if authored:
            raise RuntimeError(f"Authored unexpected {label} joint limit")
        return
    if not authored or not math.isclose(
        float(attribute.Get()),
        expected,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError(f"Authored {label} joint limit does not match source")


def _stage_json_artifact(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
    replace_existing: bool = True,
) -> _StagedArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    completed = False
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        completed = True
        return _StagedArtifact(
            staged_path=temporary,
            target_path=path,
            label=label,
            replace_existing=replace_existing,
        )
    finally:
        if temporary is not None and temporary.exists() and not completed:
            temporary.unlink()


def _write_json_artifacts_transactionally(
    artifacts: list[tuple[Path, Mapping[str, Any], str]],
    *,
    prebackup_validator: Callable[[], None] | None = None,
    replace_existing: bool = True,
) -> None:
    staged: list[_StagedArtifact] = []
    try:
        for path, payload, label in artifacts:
            staged.append(
                _stage_json_artifact(
                    path,
                    payload,
                    label=label,
                    replace_existing=replace_existing,
                )
            )
        _promote_staged_artifacts(
            staged,
            prebackup_validator=prebackup_validator,
        )
    finally:
        for artifact in staged:
            _remove_artifact(artifact.staged_path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json_artifacts_transactionally([(path, payload, path.name)])
