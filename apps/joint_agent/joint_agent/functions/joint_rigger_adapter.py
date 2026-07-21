# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disabled-by-default Joint Rigger adapter boundary."""

from __future__ import annotations

import copy
import errno
import importlib
import inspect
import json
import ntpath
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from world_understanding.functions.physics.joint_rigger import (
    JointRiggerArtifactError,
    JointRiggerArtifactTargets,
    JointRiggerContractError,
    JointRiggerResultV1,
)
from world_understanding.utils.usd.package import (
    UsdzPackageError,
    extract_usdz_package_for_edit,
)

from joint_agent.functions.artifact_transaction import (
    StagedArtifact,
    promote_staged_artifacts,
    remove_artifact,
)
from joint_agent.joint_rigger_options import (
    CANDIDATE_REQUIRED_JOINT_RIGGER_ADAPTERS,
    DEFAULT_CANDIDATE_READINESS_POLICY,
    DEFAULT_JOINT_RIGGER_ADAPTER,
    DEFAULT_MISSING_DEPENDENCY_POLICY,
    DEFAULT_USD_JOINT_RIGGER_APPLY_COLLISION,
    DEFAULT_USD_JOINT_RIGGER_APPLY_MASSES,
    DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
    SUPPORTED_CANDIDATE_READINESS_POLICIES,
    SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS,
    SUPPORTED_MISSING_DEPENDENCY_POLICIES,
    CandidateReadinessPolicy,
    InternalJointRiggerAdapterName,
    MissingDependencyPolicy,
    format_allowed_values,
)

DIAGNOSTICS_SCHEMA_VERSION = "joint-agent-rigger-diagnostics-v0"
VALIDATION_SCHEMA_VERSION = "joint-agent-rigger-validation-v0"
READINESS_SCHEMA_VERSION = "joint-agent-rigger-readiness-v0"
_READY_REVIEW_STATUS = "ready_for_rigger_input"
_MOTION_TYPES_REQUIRING_AXIS = {"revolute", "prismatic"}
_RIGGABLE_MOTION_TYPES = {"revolute", "prismatic", "spherical"}
_REAL_ADAPTER_NON_SUCCESS_STATUSES = {
    "artifact_error",
    "blocked",
    "blocked_unready_candidates",
    "error",
    "failed",
    "failure",
    "no_joints_authored",
    "skipped",
}
_REAL_ADAPTER_SKIP_STATUSES = {"blocked", "blocked_unready_candidates", "skipped"}
_OWNED_CORE_RESULT_STATUSES = {
    "no_joints_authored": "no_joints_authored",
    "unavailable": "skipped",
    "incompatible": "error",
    "rejected": "blocked",
    "failed": "failed",
}
_USD_EXTENSIONS = {".usd", ".usda", ".usdc"}
_SELF_CONTAINED_PACKAGE_ASSET_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".dds",
    ".exr",
    ".gif",
    ".hdr",
    ".jpeg",
    ".jpg",
    ".ktx",
    ".ktx2",
    ".png",
    ".tga",
    ".tif",
    ".tiff",
    ".tx",
    ".webp",
}
_PRUNABLE_PACKAGE_DEPENDENCY_EXTENSIONS = (
    _USD_EXTENSIONS | _SELF_CONTAINED_PACKAGE_ASSET_EXTENSIONS
)
# Match the concrete range expanded by the pinned OpenUSD resolver. Values
# outside this window remain unresolved and must never trigger preservation.
_UDIM_TILE_MIN = 1001
_UDIM_TILE_MAX = 1100


def apply_joint_rigger(
    *,
    input_usd_path: str | Path,
    predictions_path: str | Path | None = None,
    output_usd_path: str | Path,
    diagnostics_path: str | Path,
    validation_path: str | Path,
    articulation_candidates_path: str | Path | None = None,
    adapter: InternalJointRiggerAdapterName = DEFAULT_JOINT_RIGGER_ADAPTER,
    on_missing_dependency: MissingDependencyPolicy = DEFAULT_MISSING_DEPENDENCY_POLICY,
    on_unready_candidates: CandidateReadinessPolicy = (
        DEFAULT_CANDIDATE_READINESS_POLICY
    ),
    joint_rigger_template: str = DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
    apply_masses: bool | None = None,
    apply_collision: bool | None = None,
) -> dict[str, Any]:
    """Apply a Joint Rigger adapter boundary.

    The mock adapter remains report-only, the external adapter is lazy-imported,
    and the owned adapters author topology without applying body physics schemas.
    ``owned_core`` consumes predictions when supplied so exact Stage 1 rigid-link
    membership and Stage 2 topology enter the shared first-class V1/V2 contract
    path.
    """
    input_path = Path(input_usd_path)
    predictions = Path(predictions_path) if predictions_path is not None else None
    output_path = Path(output_usd_path)
    diagnostics = Path(diagnostics_path)
    validation = Path(validation_path)
    candidates = (
        Path(articulation_candidates_path) if articulation_candidates_path else None
    )
    if apply_masses is None:
        apply_masses = (
            False if adapter == "owned_core" else DEFAULT_USD_JOINT_RIGGER_APPLY_MASSES
        )
    if apply_collision is None:
        apply_collision = (
            False
            if adapter == "owned_core"
            else DEFAULT_USD_JOINT_RIGGER_APPLY_COLLISION
        )

    if adapter not in SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS:
        raise ValueError(f"Unsupported Joint Rigger adapter: {adapter}")
    if on_missing_dependency not in SUPPORTED_MISSING_DEPENDENCY_POLICIES:
        raise ValueError(
            "on_missing_dependency must be one of: "
            f"{format_allowed_values(SUPPORTED_MISSING_DEPENDENCY_POLICIES)}; "
            f"got {on_missing_dependency}"
        )
    if on_unready_candidates not in SUPPORTED_CANDIDATE_READINESS_POLICIES:
        raise ValueError(
            "on_unready_candidates must be one of: "
            f"{format_allowed_values(SUPPORTED_CANDIDATE_READINESS_POLICIES)}; "
            f"got {on_unready_candidates}"
        )
    if adapter == "owned_core" and (apply_masses or apply_collision):
        raise ValueError(
            "owned_core is topology-only; apply_masses and apply_collision "
            "must both be false"
        )
    if adapter in CANDIDATE_REQUIRED_JOINT_RIGGER_ADAPTERS and candidates is None:
        raise ValueError(f"{adapter} requires articulation_candidates_path")
    _validate_distinct_artifact_paths(
        input_usd_path=input_path,
        predictions_path=predictions,
        output_usd_path=output_path,
        diagnostics_path=diagnostics,
        validation_path=validation,
        articulation_candidates_path=candidates,
    )

    readiness = _evaluate_candidate_readiness(
        articulation_candidates_path=candidates,
        policy=on_unready_candidates,
    )
    if readiness["status"] == "blocked" and adapter != "stage2_candidate_edges":
        _require_candidate_readiness_matches_path(
            articulation_candidates_path=candidates,
            readiness=readiness,
        )
        return _write_readiness_blocked_result(
            adapter=adapter,
            input_usd_path=input_path,
            predictions_path=predictions,
            output_usd_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
            articulation_candidates_path=candidates,
            readiness=readiness,
        )

    if adapter in CANDIDATE_REQUIRED_JOINT_RIGGER_ADAPTERS:
        _require_candidate_readiness_identity(readiness)

    if adapter == "stage2_candidate_edges":
        assert candidates is not None
        from joint_agent.functions.candidate_edge_authoring import (
            author_stage2_candidate_edges,
        )

        return cast(
            dict[str, Any],
            author_stage2_candidate_edges(
                input_usd_path=input_path,
                predictions_path=predictions,
                articulation_candidates_path=candidates,
                output_usd_path=output_path,
                diagnostics_path=diagnostics,
                validation_path=validation,
                candidate_readiness=readiness,
            ),
        )

    if adapter == "owned_core":
        assert candidates is not None
        from joint_agent.functions.joint_rigger_core_bridge import (
            InitialNoReadyJointCandidatesError,
            author_stage2_articulation_contract_via_core,
            author_stage2_candidate_edges_via_core,
        )

        try:
            targets = JointRiggerArtifactTargets(
                output_path=output_path,
                diagnostics_path=diagnostics,
                result_path=validation,
            )
            # Preserve the established zero-ready result through the exact
            # Stage 2 preflight; malformed all-unready documents still fail.
            has_mixed_readiness = (
                readiness["ready_candidate_count"] > 0
                and readiness["unready_candidate_count"] > 0
            )
            if predictions is None:
                core_result = author_stage2_candidate_edges_via_core(
                    input_usd_path=input_path,
                    articulation_candidates_path=candidates,
                    artifact_targets=targets,
                    candidate_readiness=readiness,
                )
            else:
                # Keep prediction-backed V1/V2 body membership for mixed
                # readiness while explicitly projecting only the ready subset.
                core_result = author_stage2_articulation_contract_via_core(
                    input_usd_path=input_path,
                    articulation_candidates_path=candidates,
                    predictions_path=predictions,
                    artifact_targets=targets,
                    candidate_readiness=readiness,
                    allow_ready_subset=has_mixed_readiness,
                )
        except InitialNoReadyJointCandidatesError:
            if readiness["ready_candidate_count"] != 0:
                raise
            return _write_owned_core_no_joints_result(
                input_usd_path=input_path,
                predictions_path=predictions,
                output_usd_path=output_path,
                diagnostics_path=diagnostics,
                validation_path=validation,
                articulation_candidates_path=candidates,
                readiness=readiness,
            )
        except JointRiggerContractError as exc:
            if not has_mixed_readiness or exc.code not in {
                "stage2_parent_link_ambiguous",
                "stage2_parent_link_requires_review",
            }:
                raise
            return _owned_core_unsafe_subset_result(
                readiness=readiness,
                error=exc,
            )
        return _owned_core_result(
            result=core_result,
            output_usd_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
            readiness=readiness,
        )

    if adapter == "mock":
        if predictions is None:
            raise ValueError("mock adapter requires predictions_path")
        return _run_mock_adapter(
            input_usd_path=input_path,
            predictions_path=predictions,
            output_usd_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
            articulation_candidates_path=candidates,
            readiness=readiness,
        )

    if predictions is None:
        raise ValueError("usd_joint_rigger adapter requires predictions_path")

    try:
        module = importlib.import_module("usd_joint_rigger")
    except ImportError as exc:
        if getattr(exc, "name", None) != "usd_joint_rigger":
            raise RuntimeError(
                "usd_joint_rigger adapter package was found, but one of its "
                "dependencies could not be imported"
            ) from exc
        if on_missing_dependency == "block":
            raise RuntimeError(
                "usd_joint_rigger adapter requested but package or one of its "
                "dependencies could not be imported"
            ) from exc
        return _write_skip_result(
            input_usd_path=input_path,
            predictions_path=predictions,
            output_usd_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
            articulation_candidates_path=candidates,
            readiness=readiness,
            warning=(
                "usd_joint_rigger package could not be imported; apply step skipped"
            ),
        )
    except OSError as exc:
        if on_missing_dependency == "block":
            raise RuntimeError(
                "usd_joint_rigger adapter requested but package or one of its "
                "dependencies could not be imported"
            ) from exc
        return _write_skip_result(
            input_usd_path=input_path,
            predictions_path=predictions,
            output_usd_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
            articulation_candidates_path=candidates,
            readiness=readiness,
            warning=(
                "usd_joint_rigger package or one of its dependencies could not "
                "be imported; apply step skipped"
            ),
        )

    _clear_stale_apply_artifacts(
        output_usd_path=output_path,
        diagnostics_path=diagnostics,
        validation_path=validation,
    )

    runner = getattr(module, "apply_joint_rigger", None)
    if callable(runner):
        result = _run_native_usd_joint_rigger(
            runner=runner,
            input_usd_path=input_path,
            predictions_path=predictions,
            output_usd_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
            articulation_candidates_path=candidates,
            joint_rigger_template=joint_rigger_template,
            apply_masses=apply_masses,
            apply_collision=apply_collision,
        )
    else:
        result = _run_usd_joint_rigger_handoff_adapter(
            module=module,
            input_usd_path=input_path,
            predictions_path=predictions,
            output_usd_path=output_path,
            diagnostics_path=diagnostics,
            validation_path=validation,
            articulation_candidates_path=candidates,
            template_name=joint_rigger_template,
            apply_masses=apply_masses,
            apply_collision=apply_collision,
        )
    if not isinstance(result, dict):
        raise RuntimeError(
            "usd_joint_rigger.apply_joint_rigger returned non-dict result"
        )
    output_warnings = _normalize_real_adapter_output_artifact(
        result=result,
        expected_output_path=output_path,
    )
    augmentation_warnings = _augment_real_adapter_artifacts(
        diagnostics_path=diagnostics,
        validation_path=validation,
        readiness=readiness,
    )
    real_adapter_warnings = [
        *output_warnings,
        *augmentation_warnings,
        *readiness["warnings"],
    ]
    if real_adapter_warnings:
        result["joint_rigger_warnings"] = _merge_warnings(
            result.get("joint_rigger_warnings", []),
            real_adapter_warnings,
        )
    if "joint_rigger_diagnostics_path" not in result and diagnostics.exists():
        result["joint_rigger_diagnostics_path"] = str(diagnostics)
    if "joint_rigger_validation_path" not in result and validation.exists():
        result["joint_rigger_validation_path"] = str(validation)
    result["joint_rigger_candidate_readiness"] = readiness
    return result


def _owned_core_result(
    *,
    result: JointRiggerResultV1,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Map the shared v1 result onto the established apply-step result keys."""

    succeeded = result.status == "succeeded"
    status = (
        "authored"
        if succeeded
        else _OWNED_CORE_RESULT_STATUSES.get(result.status, "failed")
    )
    return {
        "joint_rigger_status": status,
        "rigged_usd_path": str(output_usd_path) if succeeded else None,
        "joint_rigger_diagnostics_path": str(diagnostics_path),
        "joint_rigger_validation_path": str(validation_path),
        "joint_rigger_warnings": _merge_warnings(
            list(result.diagnostics.warnings),
            list(readiness.get("warnings", [])),
        ),
        "joint_rigger_errors": list(result.diagnostics.errors),
        "joint_rigger_candidate_readiness": dict(readiness),
        "authored_joint_count": (
            len(result.diagnostics.joint_diagnostics) if succeeded else 0
        ),
        "apply_joint_rigger_skipped": status in {"blocked", "skipped"},
    }


def _write_owned_core_no_joints_result(
    *,
    input_usd_path: Path,
    predictions_path: Path | None,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    articulation_candidates_path: Path,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Complete a valid zero-ready owned-core request without invoking the facade."""

    warning = "owned_core found no ready candidate edges; no generated USD was written"
    warnings = _merge_warnings(readiness.get("warnings", []), [warning])
    _clear_stale_apply_artifact(output_usd_path, "generated USD")

    diagnostics = _base_diagnostics(
        adapter="owned_core",
        status="no_joints_authored",
        input_usd_path=input_usd_path,
        predictions_path=predictions_path,
        output_usd_path=output_usd_path,
        articulation_candidates_path=articulation_candidates_path,
        warnings=warnings,
        errors=[],
    )
    diagnostics.update(
        {
            "output_usd_path": None,
            "configured_output_usd_path": str(output_usd_path),
            "candidate_readiness": dict(readiness),
            "authored_joint_count": 0,
        }
    )
    validation = _base_validation(
        adapter="owned_core",
        status="no_joints_authored",
        output_usd_path=None,
        warnings=warnings,
        errors=[],
    )
    validation.update(
        {
            "configured_output_usd_path": str(output_usd_path),
            "validation_skipped": True,
            "validation_skip_reason": "no ready Stage 2 candidate edges were authored",
            "candidate_readiness": dict(readiness),
            "authored_joint_count": 0,
        }
    )
    _write_json_atomic(diagnostics_path, diagnostics)
    _write_json_atomic(validation_path, validation)
    return {
        "joint_rigger_status": "no_joints_authored",
        "rigged_usd_path": None,
        "joint_rigger_diagnostics_path": str(diagnostics_path),
        "joint_rigger_validation_path": str(validation_path),
        "joint_rigger_warnings": warnings,
        "joint_rigger_errors": [],
        "joint_rigger_candidate_readiness": dict(readiness),
        "authored_joint_count": 0,
        "apply_joint_rigger_skipped": False,
    }


def _owned_core_unsafe_subset_result(
    *,
    readiness: Mapping[str, Any],
    error: JointRiggerContractError,
) -> dict[str, Any]:
    """Report unsafe mixed-subset dependencies without publishing artifacts.

    The core facade releases its captured-target reservation before propagating
    a backend contract error. Any path mutation here could therefore race a
    concurrent successful publisher. Keep this failure result in memory and
    leave the existing artifact bundle unchanged.
    """

    detail = f"{error.code}: {error.detail}"
    warning = (
        "owned_core did not author the mixed ready subset because its parent-link "
        "dependencies require review"
    )
    warnings = _merge_warnings(readiness.get("warnings", []), [warning])
    errors = [detail]
    dependency_safety = {
        "status": "blocked",
        "code": error.code,
        "detail": error.detail,
    }
    reported_readiness = dict(readiness)
    reported_readiness["dependency_safety"] = dependency_safety
    return {
        "joint_rigger_status": "blocked",
        "rigged_usd_path": None,
        "joint_rigger_diagnostics_path": None,
        "joint_rigger_validation_path": None,
        "joint_rigger_warnings": warnings,
        "joint_rigger_errors": errors,
        "joint_rigger_candidate_readiness": reported_readiness,
        "authored_joint_count": 0,
        "apply_joint_rigger_skipped": True,
    }


def _normalize_real_adapter_output_artifact(
    *,
    result: dict[str, Any],
    expected_output_path: Path,
) -> list[str]:
    returned_path_value = result.get("rigged_usd_path")
    expected_output = expected_output_path.expanduser().resolve()
    status = _normalized_real_adapter_status(result.get("joint_rigger_status"))
    warnings: list[str] = []

    if expected_output_path.is_file():
        if status in _REAL_ADAPTER_NON_SUCCESS_STATUSES:
            result["rigged_usd_path"] = None
            result["apply_joint_rigger_skipped"] = (
                bool(result.get("apply_joint_rigger_skipped"))
                or status in _REAL_ADAPTER_SKIP_STATUSES
            )
            return warnings
        if returned_path_value is not None:
            returned_path_text = str(returned_path_value).strip()
            if not returned_path_text:
                warnings.append(
                    "usd_joint_rigger returned an empty rigged_usd_path; "
                    "using the configured output artifact"
                )
            else:
                returned_path = Path(returned_path_text).expanduser().resolve()
                if returned_path != expected_output:
                    warnings.append(
                        "usd_joint_rigger returned rigged_usd_path "
                        f"{returned_path_value}, but the configured "
                        f"output_usd_path is {expected_output_path}; using the "
                        "configured output artifact"
                    )
        result["rigged_usd_path"] = str(expected_output_path)
        result["apply_joint_rigger_skipped"] = False
        return warnings

    result["rigged_usd_path"] = None
    result["apply_joint_rigger_skipped"] = (
        bool(result.get("apply_joint_rigger_skipped"))
        or status in _REAL_ADAPTER_SKIP_STATUSES
    )
    missing_output_error = (
        f"usd_joint_rigger did not write generated USD artifact: {expected_output_path}"
    )
    if status == "skipped":
        warnings.append(missing_output_error)
        return warnings

    result["joint_rigger_errors"] = _merge_errors(
        result.get("joint_rigger_errors", []),
        [missing_output_error],
    )
    if _real_adapter_status_needs_artifact_error(result.get("joint_rigger_status")):
        result["joint_rigger_status"] = "artifact_error"
    return warnings


def _real_adapter_status_needs_artifact_error(status: Any) -> bool:
    normalized_status = _normalized_real_adapter_status(status)
    if not normalized_status:
        return True
    return normalized_status not in _REAL_ADAPTER_NON_SUCCESS_STATUSES


def _normalized_real_adapter_status(status: Any) -> str:
    return str(status).strip().lower() if status else ""


def _run_native_usd_joint_rigger(
    *,
    runner: Any,
    input_usd_path: Path,
    predictions_path: Path,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    articulation_candidates_path: Path | None,
    joint_rigger_template: str,
    apply_masses: bool,
    apply_collision: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "input_usd_path": str(input_usd_path),
        "predictions_path": str(predictions_path),
        "output_usd_path": str(output_usd_path),
        "diagnostics_path": str(diagnostics_path),
        "validation_path": str(validation_path),
        "articulation_candidates_path": (
            str(articulation_candidates_path) if articulation_candidates_path else None
        ),
    }
    optional_kwargs = {
        "joint_rigger_template": joint_rigger_template,
        "apply_masses": apply_masses,
        "apply_collision": apply_collision,
    }
    if _callable_accepts_kwargs(runner):
        kwargs.update(optional_kwargs)
    else:
        accepted = set(_callable_parameter_names(runner))
        kwargs.update(
            {key: value for key, value in optional_kwargs.items() if key in accepted}
        )
    return cast(dict[str, Any], runner(**kwargs))


def _callable_accepts_kwargs(callable_obj: Any) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _callable_parameter_names(callable_obj: Any) -> tuple[str, ...]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return ()
    return tuple(signature.parameters)


def _clear_stale_apply_artifacts(
    *,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
) -> None:
    _clear_stale_apply_artifact(output_usd_path, "generated USD")
    _clear_stale_apply_artifact(diagnostics_path, "diagnostics")
    _clear_stale_apply_artifact(validation_path, "validation")


def _clear_stale_apply_artifact(path: Path, artifact_label: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(
            f"Refusing to replace existing {artifact_label} artifact directory: {path}"
        )
    path.unlink()


def _run_usd_joint_rigger_handoff_adapter(
    *,
    module: Any,
    input_usd_path: Path,
    predictions_path: Path,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    articulation_candidates_path: Path | None,
    template_name: str,
    apply_masses: bool,
    apply_collision: bool,
) -> dict[str, Any]:
    create_joints = getattr(module, "create_joints", None)
    if not callable(create_joints):
        raise RuntimeError(
            "usd_joint_rigger package does not expose apply_joint_rigger(...) "
            "or create_joints(...); cannot run the handoff adapter"
        )
    if not input_usd_path.exists():
        raise FileNotFoundError(f"Input USD not found: {input_usd_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions, prediction_metadata = _load_handoff_predictions(
        module=module,
        predictions_path=predictions_path,
    )
    temp_dir: Path | None = None
    try:
        editable_usd_path, temp_dir = _prepare_usd_for_handoff(
            input_usd_path,
            output_usd_path,
        )
        from pxr import Usd

        stage = Usd.Stage.Open(str(editable_usd_path))
        if not stage:
            raise RuntimeError(
                f"Could not open generated USD stage: {editable_usd_path}"
            )
        deinstanced_prim_count = _deinstance_stage_for_handoff(stage)

        warnings: list[str] = list(prediction_metadata["warnings"])
        if deinstanced_prim_count:
            warnings.append(
                "usd_joint_rigger provisional handoff de-instanced "
                f"{deinstanced_prim_count} prim(s) before authoring because USD "
                "instance proxies are read-only"
            )
        initial_joint_paths = _collect_usd_joint_prim_paths(stage)
        joint_errors: list[str] = []
        try:
            joint_errors.extend(
                _coerce_messages(
                    create_joints(stage, predictions, template_name=template_name)
                )
            )
        except Exception as exc:
            joint_errors.append(_handoff_call_error("create_joints", exc))

        if not joint_errors:
            if apply_masses:
                apply_link_masses = getattr(module, "apply_link_masses", None)
                if callable(apply_link_masses):
                    try:
                        warnings.extend(
                            _coerce_messages(
                                apply_link_masses(
                                    stage,
                                    predictions,
                                    template_name=template_name,
                                )
                            )
                        )
                    except Exception as exc:
                        joint_errors.append(
                            _handoff_call_error("apply_link_masses", exc)
                        )
                else:
                    warnings.append(
                        "usd_joint_rigger.apply_link_masses is unavailable; "
                        "mass authoring was skipped"
                    )
            if apply_collision:
                apply_collision_fn = getattr(module, "apply_collision", None)
                if callable(apply_collision_fn):
                    try:
                        warnings.extend(
                            _coerce_messages(
                                apply_collision_fn(
                                    stage,
                                    predictions,
                                    template_name=template_name,
                                )
                            )
                        )
                    except Exception as exc:
                        joint_errors.append(_handoff_call_error("apply_collision", exc))
                else:
                    warnings.append(
                        "usd_joint_rigger.apply_collision is unavailable; "
                        "collision authoring was skipped"
                    )

        if not stage.GetRootLayer().Save():
            raise RuntimeError(
                f"Could not save generated USD stage: {editable_usd_path}"
            )

        authored_joint_paths = sorted(
            _collect_usd_joint_prim_paths(stage) - initial_joint_paths
        )
        authored_joint_count = len(authored_joint_paths)
        if editable_usd_path != output_usd_path:
            try:
                _package_usdz_for_handoff(editable_usd_path, output_usd_path)
            except Exception as exc:
                joint_errors.append(_handoff_call_error("package_usdz", exc))
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
    if joint_errors:
        status = "failed"
        if not output_usd_path.exists():
            joint_errors = _merge_errors(
                joint_errors,
                [
                    "usd_joint_rigger did not write generated USD artifact: "
                    f"{output_usd_path}"
                ],
            )
    elif authored_joint_count == 0:
        status = "no_joints_authored"
        warnings.append(
            "usd_joint_rigger.create_joints completed without errors but authored "
            "no new USD physics joint prims"
        )
    else:
        status = "authored"
    diagnostics = _base_diagnostics(
        adapter="usd_joint_rigger",
        status=status,
        input_usd_path=input_usd_path,
        predictions_path=predictions_path,
        output_usd_path=output_usd_path,
        articulation_candidates_path=articulation_candidates_path,
        warnings=warnings,
        errors=joint_errors,
    )
    diagnostics.update(
        {
            "runner": "usd_joint_rigger.create_joints",
            "provisional_handoff": True,
            "mock_noop": False,
            "template_name": template_name,
            "apply_masses": apply_masses,
            "apply_collision": apply_collision,
            "deinstanced_prim_count": deinstanced_prim_count,
            "authored_joint_count": authored_joint_count,
            "authored_joint_paths": authored_joint_paths,
            "prediction_format": prediction_metadata["format"],
            "prediction_count": prediction_metadata["prediction_count"],
            "role_normalized_prediction_count": prediction_metadata[
                "role_normalized_prediction_count"
            ],
        }
    )
    validation = _base_validation(
        adapter="usd_joint_rigger",
        status=status,
        output_usd_path=output_usd_path,
        warnings=warnings,
        errors=joint_errors,
    )
    validation.update(
        {
            "validation_skipped": True,
            "validation_skip_reason": (
                "handoff adapter smoke-validates generated USD artifact only; "
                "reference or simulator validation is not part of this step"
            ),
            "authored_joint_count": authored_joint_count,
            "authored_joint_paths": authored_joint_paths,
        }
    )
    _write_json_atomic(diagnostics_path, diagnostics)
    _write_json_atomic(validation_path, validation)

    return {
        "joint_rigger_status": status,
        "rigged_usd_path": (
            str(output_usd_path)
            if status == "authored" and output_usd_path.exists()
            else None
        ),
        "joint_rigger_diagnostics_path": str(diagnostics_path),
        "joint_rigger_validation_path": str(validation_path),
        "joint_rigger_warnings": warnings,
        "joint_rigger_errors": joint_errors,
        "authored_joint_count": authored_joint_count,
        "apply_joint_rigger_skipped": False,
    }


def _handoff_call_error(call_name: str, exc: Exception) -> str:
    return f"usd_joint_rigger.{call_name} failed: {type(exc).__name__}: {exc}"


def _deinstance_stage_for_handoff(stage: Any) -> int:
    deinstanced = 0
    processed_paths: set[str] = set()
    while True:
        changed = False
        for prim in list(stage.Traverse()):
            if prim.IsInstanceProxy() or not prim.IsInstanceable():
                continue
            path = _prim_path_key(prim)
            if path in processed_paths:
                continue
            prim.SetInstanceable(False)
            processed_paths.add(path)
            deinstanced += 1
            changed = True
        if not changed:
            return deinstanced


def _prim_path_key(prim: Any) -> str:
    get_path = getattr(prim, "GetPath", None)
    if callable(get_path):
        return str(get_path())
    return f"<anonymous-prim:{id(prim)}>"


def _prepare_usd_for_handoff(
    input_usd_path: Path,
    output_usd_path: Path,
    *,
    output_anchor_path: Path | None = None,
    prune_sidecar_members: bool = False,
) -> tuple[Path, Path | None]:
    """Prepare an editable handoff while preserving legacy package contents."""

    output_usd_path.parent.mkdir(parents=True, exist_ok=True)
    src_ext = input_usd_path.suffix.lower()
    out_ext = output_usd_path.suffix.lower()

    if out_ext not in _USD_EXTENSIONS and out_ext != ".usdz":
        raise RuntimeError(
            "usd_joint_rigger handoff output must use .usd, .usda, .usdc, or .usdz; "
            f"got {output_usd_path}"
        )

    if out_ext == ".usdz":
        temp_dir = Path(tempfile.mkdtemp(prefix="joint-rigger-usdz-"))
        try:
            if src_ext == ".usdz":
                editable_usd_path = _extract_usdz_for_handoff(
                    input_usd_path,
                    temp_dir,
                )
            else:
                editable_usd_path = temp_dir / f"{output_usd_path.stem}.usda"
                _export_usd_for_handoff(input_usd_path, editable_usd_path)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return editable_usd_path, temp_dir

    if src_ext == ".usdz":
        _export_extracted_usdz_for_raw_output(
            input_usd_path,
            output_usd_path,
            prune_sidecar_members=prune_sidecar_members,
        )
        return output_usd_path, None

    if (
        src_ext == out_ext
        and src_ext in _USD_EXTENSIONS
        and input_usd_path.parent.resolve() == output_usd_path.parent.resolve()
    ):
        shutil.copy2(input_usd_path, output_usd_path)
        return output_usd_path, None

    _export_usd_for_handoff(
        input_usd_path,
        output_usd_path,
        output_anchor_path=output_anchor_path,
    )
    return output_usd_path, None


def _export_extracted_usdz_for_raw_output(
    input_usdz_path: Path,
    output_usd_path: Path,
    *,
    prune_sidecar_members: bool = False,
) -> None:
    """Export raw USD, pruning package extras only for explicit core callers."""

    sidecar_dir = output_usd_path.parent / f"{output_usd_path.stem}_assets"
    _ensure_input_not_inside_sidecar(input_usdz_path, sidecar_dir)
    if sidecar_dir.exists() or sidecar_dir.is_symlink():
        if not sidecar_dir.is_dir() or sidecar_dir.is_symlink():
            raise RuntimeError(
                f"Refusing to replace existing USDZ sidecar artifact: {sidecar_dir}"
            )

    staging_dir = Path(
        tempfile.mkdtemp(
            dir=output_usd_path.parent,
            prefix=f".{output_usd_path.stem}.handoff-",
        )
    )
    staged_output = staging_dir / output_usd_path.name
    staged_sidecar = staging_dir / sidecar_dir.name
    staged_sidecar.mkdir()
    try:
        extracted_root = _extract_usdz_for_handoff(
            input_usdz_path,
            staged_sidecar,
        )
        _export_usd_for_handoff(extracted_root, staged_output)
        if prune_sidecar_members:
            _prune_unreferenced_sidecar_members(
                output_usd_path=staged_output,
                sidecar_root=staged_sidecar,
            )
        # Keep the root last: it is the commit point that tells consumers the
        # sidecar/output pair was published completely.
        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged_path=staged_sidecar,
                    target_path=sidecar_dir,
                    label="USDZ raw-output sidecar",
                ),
                StagedArtifact(
                    staged_path=staged_output,
                    target_path=output_usd_path,
                    label="generated USD",
                ),
            ]
        )
    finally:
        remove_artifact(staged_output)
        shutil.rmtree(staging_dir, ignore_errors=True)


def _prune_unreferenced_sidecar_members(
    *,
    output_usd_path: Path,
    sidecar_root: Path,
) -> None:
    """Prune only when every package dependency has a complete USD inventory.

    USD layers and self-contained raster payloads cannot hide additional
    relative files. Shader/source documents and unknown formats can, so their
    already-bounded validated package is preserved intact rather than risk a
    dangling include or runtime resource.
    """

    from world_understanding.functions.physics.joint_rigger.models import (
        JointRiggerContractError,
    )
    from world_understanding.functions.physics.joint_rigger.reference import (
        local_usd_dependency_paths,
    )

    members = sorted(
        sidecar_root.rglob("*"),
        key=lambda path: (len(path.parts), path.as_posix()),
        reverse=True,
    )
    for member in members:
        if member.is_symlink():
            raise RuntimeError(
                f"Refusing to prune a symlinked USDZ sidecar member: {member}"
            )
        if not (member.is_file() or member.is_dir()):
            raise RuntimeError(f"Unsupported USDZ sidecar member type: {member}")

    try:
        dependencies = {
            path.expanduser().resolve(strict=True)
            for path in local_usd_dependency_paths(output_usd_path)
        }
    except JointRiggerContractError as exc:
        if exc.code != "unresolved_artifact_dependency" or not (
            _unresolved_udim_dependencies_have_concrete_sidecar_members(
                unresolved_locators=exc.unresolved_dependency_paths,
                sidecar_root=sidecar_root,
            )
        ):
            raise
        # OpenUSD can report re-anchored absolute UDIM locators as unresolved
        # even when their concrete tiles are present. The helper permits only
        # that verified in-package case; every other incomplete inventory fails.
        return
    sidecar_resolved = sidecar_root.resolve(strict=True)
    preserve_full_package = any(
        path.suffix.lower() not in _PRUNABLE_PACKAGE_DEPENDENCY_EXTENSIONS
        and _is_relative_to(path, sidecar_resolved)
        for path in dependencies
    )
    if preserve_full_package:
        return
    for member in members:
        if member.is_symlink():
            raise RuntimeError(
                f"Refusing to prune a symlinked USDZ sidecar member: {member}"
            )
        if member.is_file():
            if member.resolve(strict=True) not in dependencies:
                member.unlink()
            continue
        if member.is_dir():
            try:
                member.rmdir()
            except OSError as exc:
                if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                    raise
            continue
        raise RuntimeError(f"Unsupported USDZ sidecar member type: {member}")


def _unresolved_udim_dependencies_have_concrete_sidecar_members(
    *,
    unresolved_locators: tuple[str, ...],
    sidecar_root: Path,
) -> bool:
    """Recognize only unresolved, in-package UDIM locators with real tiles."""

    locators = tuple(sorted(set(unresolved_locators)))
    if not locators:
        return False
    sidecar_resolved = sidecar_root.resolve(strict=True)
    for locator in locators:
        candidate = Path(locator)
        if (
            "://" in locator
            or not candidate.is_absolute()
            or candidate.name.count("<UDIM>") != 1
            or "<UDIM>" in candidate.parent.as_posix()
        ):
            return False
        members = _concrete_udim_members(candidate)
        try:
            parent = candidate.parent.resolve(strict=True)
        except OSError:
            return False
        if (
            not members
            or not _is_relative_to(parent, sidecar_resolved)
            or any(not _is_relative_to(member, sidecar_resolved) for member in members)
        ):
            return False
    return True


def _concrete_udim_members(pattern_path: Path) -> tuple[Path, ...]:
    """Return safe concrete tiles for one literal ``<UDIM>`` filesystem path."""

    if (
        pattern_path.name.count("<UDIM>") != 1
        or pattern_path.suffix.lower() not in _SELF_CONTAINED_PACKAGE_ASSET_EXTENSIONS
    ):
        return ()
    try:
        parent = pattern_path.parent.resolve(strict=True)
    except OSError:
        return ()
    prefix, suffix = pattern_path.name.split("<UDIM>", maxsplit=1)
    tile_name = re.compile(
        rf"{re.escape(prefix)}(?P<udim>[0-9]{{4}}){re.escape(suffix)}"
    )
    matches: list[Path] = []
    for member in parent.iterdir():
        match = tile_name.fullmatch(member.name)
        if match is None or not (
            _UDIM_TILE_MIN <= int(match.group("udim")) <= _UDIM_TILE_MAX
        ):
            continue
        if member.is_symlink() or not member.is_file():
            return ()
        matches.append(member.resolve(strict=True))
    return tuple(sorted(matches, key=lambda path: path.as_posix()))


def _ensure_input_not_inside_sidecar(input_usdz_path: Path, sidecar_dir: Path) -> None:
    input_resolved = input_usdz_path.resolve()
    sidecar_resolved = sidecar_dir.resolve(strict=False)
    if _is_relative_to(input_resolved, sidecar_resolved):
        raise RuntimeError(
            "Refusing to replace USDZ sidecar because the input package is inside "
            f"the sidecar directory: {input_usdz_path}"
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _extract_usdz_for_handoff(input_usdz_path: Path, temp_dir: Path) -> Path:
    from pxr import Usd

    extract_dir = temp_dir / "input"
    try:
        editable_usd_path = extract_usdz_package_for_edit(
            input_usdz_path,
            extract_dir,
        )
    except UsdzPackageError as exc:
        raise RuntimeError(str(exc)) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not extract USDZ package: {input_usdz_path}"
        ) from exc
    if not editable_usd_path.exists():
        raise RuntimeError(
            f"Extracted USDZ root layer was not found: {editable_usd_path}"
        )
    if not Usd.Stage.Open(str(editable_usd_path)):
        raise RuntimeError(
            f"Could not open extracted USDZ root layer: {input_usdz_path}"
        )
    return editable_usd_path


def _export_usd_for_handoff(
    input_usd_path: Path,
    output_usd_path: Path,
    *,
    output_anchor_path: Path | None = None,
) -> None:
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(input_usd_path))
    if not stage:
        raise RuntimeError(f"Could not open source USD stage: {input_usd_path}")
    source_layer = stage.GetRootLayer()
    if not source_layer.Export(str(output_usd_path)):
        raise RuntimeError(f"Could not export {input_usd_path} to {output_usd_path}")
    output_layer = Sdf.Layer.FindOrOpen(str(output_usd_path))
    if not output_layer:
        raise RuntimeError(f"Could not open exported USD layer: {output_usd_path}")
    # One exhaustive pass covers composition arcs, clips, metadata, variants,
    # and scalar/array asset values. Splitting composition from attribute
    # rewrites can resolve the second pass against already-moved locators.
    _rewrite_moved_layer_asset_paths(
        source_layer=source_layer,
        output_layer=output_layer,
        output_anchor_path=output_anchor_path,
    )
    if not output_layer.Save():
        raise RuntimeError(f"Could not save exported USD layer: {output_usd_path}")
    if not Usd.Stage.Open(str(output_anchor_path or output_usd_path)):
        raise RuntimeError(f"Could not open exported USD stage: {output_usd_path}")


def _rewrite_moved_layer_asset_paths(
    *,
    source_layer: Any,
    output_layer: Any,
    output_anchor_path: Path | None = None,
) -> int:
    from pxr import UsdUtils

    source_path = Path(source_layer.realPath or source_layer.identifier).resolve()
    output_path = (
        output_anchor_path.resolve(strict=False)
        if output_anchor_path is not None
        else Path(output_layer.realPath or output_layer.identifier).resolve()
    )
    if source_path.parent == output_path.parent:
        return 0

    rewritten = 0

    def remap_asset_path(asset_path: str) -> str:
        if (
            not asset_path
            or "://" in asset_path
            or _is_absolute_local_asset_path(asset_path)
        ):
            return asset_path
        resolved = source_layer.ComputeAbsolutePath(asset_path)
        if not resolved:
            return asset_path
        resolved_path = Path(resolved)
        if not resolved_path.exists() and not _concrete_udim_members(resolved_path):
            return asset_path
        output_target = Path(os.path.abspath(output_path.parent / asset_path))
        if output_target == Path(os.path.abspath(resolved_path)):
            return asset_path
        return _relpath_or_absolute(resolved_path, output_path.parent)

    def rewrite_asset_path(asset_path: str) -> str:
        nonlocal rewritten
        new_path = remap_asset_path(asset_path)
        if new_path != asset_path:
            rewritten += 1
        return new_path

    UsdUtils.ModifyAssetPaths(
        output_layer,
        rewrite_asset_path,
        keepEmptyPathsInArrays=True,
    )
    return rewritten


def _is_absolute_local_asset_path(asset_path: str) -> bool:
    return os.path.isabs(asset_path) or ntpath.isabs(asset_path)


def _relpath_or_absolute(resolved_path: Path, output_parent: Path) -> str:
    try:
        return os.path.relpath(resolved_path, output_parent).replace("\\", "/")
    except ValueError:
        return str(resolved_path).replace("\\", "/")


def _package_usdz_for_handoff(input_usd_path: Path, output_usdz_path: Path) -> None:
    from pxr import UsdUtils

    if not UsdUtils.CreateNewUsdzPackage(str(input_usd_path), str(output_usdz_path)):
        raise RuntimeError(f"Could not package {input_usd_path} to {output_usdz_path}")


def _load_handoff_predictions(
    *,
    module: Any,
    predictions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_text = predictions_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        if predictions_path.suffix.lower() == ".json":
            raise ValueError(
                f"Invalid JSON prediction artifact at {predictions_path}: {exc.msg}"
            ) from exc
        parsed = None

    if isinstance(parsed, dict) and "id" in parsed:
        predictions = _predictions_from_raw_list(module, [parsed])
        prediction_format = "single_record"
        raw_count = 1
    elif isinstance(parsed, dict):
        predictions = copy.deepcopy(parsed)
        prediction_format = "dict"
        raw_count = len(predictions)
    elif isinstance(parsed, list):
        predictions = _predictions_from_raw_list(module, parsed)
        prediction_format = "list"
        raw_count = len(parsed)
    else:
        raw_list = _read_jsonl_predictions(raw_text, predictions_path)
        predictions = _predictions_from_raw_list(module, raw_list)
        prediction_format = "jsonl"
        raw_count = len(raw_list)

    if not isinstance(predictions, dict):
        raise RuntimeError("usd_joint_rigger prediction conversion returned non-dict")

    role_normalized_count = _normalize_prediction_roles_for_handoff(predictions)
    return predictions, {
        "format": prediction_format,
        "prediction_count": raw_count,
        "role_normalized_prediction_count": role_normalized_count,
        "warnings": [],
    }


def _predictions_from_raw_list(module: Any, raw_list: list[Any]) -> dict[str, Any]:
    converter = getattr(module, "predictions_from_raw_json", None)
    if not callable(converter):
        raise RuntimeError(
            "usd_joint_rigger.predictions_from_raw_json is required for "
            "JSON-list or JSONL prediction artifacts"
        )
    predictions = converter(raw_list)
    return copy.deepcopy(predictions)


def _read_jsonl_predictions(raw_text: str, predictions_path: Path) -> list[Any]:
    raw_list: list[Any] = []
    for lineno, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw_list.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL prediction at {predictions_path}:{lineno}: {exc.msg}"
            ) from exc
    return raw_list


def _normalize_prediction_roles_for_handoff(predictions: dict[str, Any]) -> int:
    normalized_count = 0
    for entry in predictions.values():
        if not isinstance(entry, dict):
            continue
        classification = entry.get("classification")
        if not isinstance(classification, dict):
            continue
        role = _non_empty_string(
            classification.get("role"),
        ) or _non_empty_string(entry.get("role"))
        if not role or role.casefold() == "unknown":
            continue
        component_name = _non_empty_string(classification.get("component_name"))
        if component_name and component_name != role:
            classification.setdefault("semantic_component_name", component_name)
        if classification.get("component_name") != role:
            classification["component_name"] = role
            normalized_count += 1
    return normalized_count


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_messages(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _count_usd_joint_prims(stage: Any) -> int:
    return len(_collect_usd_joint_prim_paths(stage))


def _collect_usd_joint_prim_paths(stage: Any) -> set[str]:
    from pxr import UsdPhysics

    joint_paths: set[str] = set()
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint_paths.add(str(prim.GetPath()))
    return joint_paths


def _merge_errors(existing: Any, additions: list[str]) -> list[str]:
    return _merge_messages(existing, additions)


def _merge_warnings(existing: Any, additions: list[str]) -> list[str]:
    return _merge_messages(existing, additions)


def _merge_messages(existing: Any, additions: list[str]) -> list[str]:
    messages = [str(value) for value in existing] if isinstance(existing, list) else []
    if existing and not isinstance(existing, list):
        messages.append(str(existing))
    for message in additions:
        if message not in messages:
            messages.append(message)
    return messages


def _augment_real_adapter_artifacts(
    *,
    diagnostics_path: Path,
    validation_path: Path,
    readiness: dict[str, Any],
) -> list[str]:
    return [
        warning
        for warning in (
            _augment_json_artifact_with_readiness(
                diagnostics_path,
                "diagnostics",
                readiness,
            ),
            _augment_json_artifact_with_readiness(
                validation_path,
                "validation",
                readiness,
            ),
        )
        if warning
    ]


def _augment_json_artifact_with_readiness(
    path: Path,
    artifact_label: str,
    readiness: dict[str, Any],
) -> str | None:
    if not path.exists():
        return (
            f"usd_joint_rigger did not write {artifact_label} artifact: {path}; "
            "candidate_readiness was not persisted"
        )
    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return (
            f"usd_joint_rigger {artifact_label} artifact could not be updated "
            f"with candidate_readiness: {path}: {exc}"
        )
    if not isinstance(payload, dict):
        return (
            f"usd_joint_rigger {artifact_label} artifact is not a JSON object: "
            f"{path}; candidate_readiness was not persisted"
        )

    payload["candidate_readiness"] = readiness
    if readiness["warnings"]:
        payload["warnings"] = _merge_warnings(
            payload.get("warnings", []),
            readiness["warnings"],
        )
    try:
        _write_json_atomic(path, payload)
    except OSError as exc:
        return (
            f"usd_joint_rigger {artifact_label} artifact could not be updated "
            f"with candidate_readiness: {path}: {exc}"
        )
    return None


def _validate_distinct_artifact_paths(
    *,
    input_usd_path: Path,
    predictions_path: Path | None,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    articulation_candidates_path: Path | None,
) -> None:
    paths = [("input_usd_path", input_usd_path)]
    if predictions_path is not None:
        paths.append(("predictions_path", predictions_path))
    paths.extend(
        [
            ("output_usd_path", output_usd_path),
            ("diagnostics_path", diagnostics_path),
            ("validation_path", validation_path),
        ]
    )
    if articulation_candidates_path is not None:
        paths.append(("articulation_candidates_path", articulation_candidates_path))

    seen: dict[Path, str] = {}
    for label, path in paths:
        normalized = path.expanduser().resolve()
        previous = seen.get(normalized)
        if previous:
            raise ValueError(
                f"{label} must not reference the same path as {previous}: {path}"
            )
        seen[normalized] = label


def _run_mock_adapter(
    *,
    input_usd_path: Path,
    predictions_path: Path,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    articulation_candidates_path: Path | None,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    if not input_usd_path.exists():
        raise FileNotFoundError(f"Input USD not found: {input_usd_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    output_usd_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_usd_path, output_usd_path)

    warnings = [
        "mock adapter did not author USD physics joints",
        "real Joint Rigger package and acceptance assets are not configured",
        *readiness["warnings"],
    ]

    diagnostics = _base_diagnostics(
        adapter="mock",
        status="mock_noop",
        input_usd_path=input_usd_path,
        predictions_path=predictions_path,
        output_usd_path=output_usd_path,
        articulation_candidates_path=articulation_candidates_path,
        warnings=warnings,
        errors=[],
    )
    diagnostics["mock_noop"] = True
    diagnostics["authored_joint_count"] = 0
    diagnostics["candidate_readiness"] = readiness
    validation = _base_validation(
        adapter="mock",
        status="mock_noop",
        output_usd_path=output_usd_path,
        warnings=warnings,
        errors=[],
    )
    validation["validation_skipped"] = True
    validation["candidate_readiness"] = readiness

    _write_json_atomic(diagnostics_path, diagnostics)
    _write_json_atomic(validation_path, validation)

    return {
        "joint_rigger_status": "mock_noop",
        "rigged_usd_path": str(output_usd_path),
        "joint_rigger_diagnostics_path": str(diagnostics_path),
        "joint_rigger_validation_path": str(validation_path),
        "joint_rigger_warnings": warnings,
        "joint_rigger_errors": [],
        "joint_rigger_candidate_readiness": readiness,
        "authored_joint_count": 0,
        "apply_joint_rigger_skipped": False,
    }


def _write_skip_result(
    *,
    input_usd_path: Path,
    predictions_path: Path,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    articulation_candidates_path: Path | None,
    readiness: dict[str, Any],
    warning: str,
) -> dict[str, Any]:
    _clear_stale_apply_artifacts(
        output_usd_path=output_usd_path,
        diagnostics_path=diagnostics_path,
        validation_path=validation_path,
    )
    warnings = [warning, *readiness["warnings"]]
    diagnostics = _base_diagnostics(
        adapter="usd_joint_rigger",
        status="skipped",
        input_usd_path=input_usd_path,
        predictions_path=predictions_path,
        output_usd_path=output_usd_path,
        articulation_candidates_path=articulation_candidates_path,
        warnings=warnings,
        errors=[],
    )
    diagnostics["mock_noop"] = False
    diagnostics["authored_joint_count"] = 0
    diagnostics["candidate_readiness"] = readiness
    validation = _base_validation(
        adapter="usd_joint_rigger",
        status="skipped",
        output_usd_path=None,
        warnings=warnings,
        errors=[],
    )
    validation["validation_skipped"] = True
    validation["candidate_readiness"] = readiness

    _write_json_atomic(diagnostics_path, diagnostics)
    _write_json_atomic(validation_path, validation)

    return {
        "joint_rigger_status": "skipped",
        "rigged_usd_path": None,
        "joint_rigger_diagnostics_path": str(diagnostics_path),
        "joint_rigger_validation_path": str(validation_path),
        "joint_rigger_warnings": warnings,
        "joint_rigger_errors": [],
        "joint_rigger_candidate_readiness": readiness,
        "authored_joint_count": 0,
        "apply_joint_rigger_skipped": True,
    }


def _write_readiness_blocked_result(
    *,
    adapter: str,
    input_usd_path: Path,
    predictions_path: Path | None,
    output_usd_path: Path,
    diagnostics_path: Path,
    validation_path: Path,
    articulation_candidates_path: Path | None,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    _clear_stale_apply_artifacts(
        output_usd_path=output_usd_path,
        diagnostics_path=diagnostics_path,
        validation_path=validation_path,
    )
    errors = [
        "articulation candidates are not ready for Joint Rigger input",
        *readiness["errors"],
    ]
    warnings = list(readiness["warnings"])
    diagnostics = _base_diagnostics(
        adapter=adapter,
        status="blocked_unready_candidates",
        input_usd_path=input_usd_path,
        predictions_path=predictions_path,
        output_usd_path=output_usd_path,
        articulation_candidates_path=articulation_candidates_path,
        warnings=warnings,
        errors=errors,
    )
    diagnostics["mock_noop"] = False
    diagnostics["authored_joint_count"] = 0
    diagnostics["candidate_readiness"] = readiness
    validation = _base_validation(
        adapter=adapter,
        status="blocked_unready_candidates",
        output_usd_path=None,
        warnings=warnings,
        errors=errors,
    )
    validation["validation_skipped"] = True
    validation["candidate_readiness"] = readiness

    _write_json_atomic(diagnostics_path, diagnostics)
    _write_json_atomic(validation_path, validation)

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


def _evaluate_candidate_readiness(
    *,
    articulation_candidates_path: Path | None,
    policy: CandidateReadinessPolicy,
) -> dict[str, Any]:
    readiness: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "policy": policy,
        "status": "skipped",
        "candidate_count": 0,
        "ready_candidate_count": 0,
        "unready_candidate_count": 0,
        "unready_candidates": [],
        "warnings": [],
        "errors": [],
    }
    if articulation_candidates_path is None:
        return readiness

    readiness["articulation_candidates_path"] = str(articulation_candidates_path)
    if not articulation_candidates_path.exists():
        warning = (
            f"articulation candidates file not found: {articulation_candidates_path}"
        )
        readiness["warnings"].append(warning)
        if policy == "block":
            readiness["status"] = "blocked"
            readiness["errors"].append(warning)
        else:
            readiness["status"] = "warning"
        return readiness

    from joint_agent.functions.joint_rigger_core_bridge import (
        _read_stable_regular_file,
    )

    try:
        candidate_sha256, candidate_bytes = _read_stable_regular_file(
            articulation_candidates_path,
            label="articulation candidates file",
            capture_payload=True,
            max_bytes=64 * 1024 * 1024,
        )
        assert candidate_bytes is not None
        readiness["articulation_candidates_sha256"] = candidate_sha256
        candidate_document = json.loads(candidate_bytes)
    except (OSError, UnicodeError, JointRiggerArtifactError) as exc:
        error = (
            f"articulation candidates file could not be read: "
            f"{articulation_candidates_path}: {exc}"
        )
        return _candidate_readiness_error(readiness, policy, error)
    except json.JSONDecodeError as exc:
        error = (
            f"articulation candidates file is invalid JSON: "
            f"{articulation_candidates_path}: {exc.msg}"
        )
        return _candidate_readiness_error(readiness, policy, error)

    if not isinstance(candidate_document, dict):
        return _candidate_readiness_error(
            readiness,
            policy,
            "articulation candidates document must be a JSON object",
        )
    if "candidates" not in candidate_document:
        return _candidate_readiness_error(
            readiness,
            policy,
            "articulation candidates document must contain a list field: candidates",
        )
    candidates = candidate_document["candidates"]
    if not isinstance(candidates, list):
        return _candidate_readiness_error(
            readiness,
            policy,
            "articulation candidates document must contain a list field: candidates",
        )

    readiness["candidate_count"] = len(candidates)
    unready_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = _candidate_id(candidate, index)
        reasons = _candidate_unready_reasons(candidate)
        if reasons:
            unready_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "reasons": reasons,
                    "unresolved_reason_codes": _unresolved_reason_codes(candidate),
                }
            )

    readiness["unready_candidates"] = unready_candidates
    readiness["unready_candidate_count"] = len(unready_candidates)
    readiness["ready_candidate_count"] = len(candidates) - len(unready_candidates)
    if unready_candidates:
        warning = (
            "articulation candidates readiness check found "
            f"{len(unready_candidates)} unready candidate(s)"
        )
        readiness["warnings"].append(warning)
        if policy == "block":
            readiness["status"] = "blocked"
            readiness["errors"].append(warning)
        else:
            readiness["status"] = "warning"
    else:
        readiness["status"] = "ready"
    return readiness


def _candidate_readiness_error(
    readiness: dict[str, Any],
    policy: CandidateReadinessPolicy,
    error: str,
) -> dict[str, Any]:
    readiness["warnings"].append(error)
    if policy == "block":
        readiness["status"] = "blocked"
        readiness["errors"].append(error)
    else:
        readiness["status"] = "warning"
    return readiness


def _require_candidate_readiness_matches_path(
    *,
    articulation_candidates_path: Path | None,
    readiness: Mapping[str, Any],
) -> None:
    """Reject stale compatibility evidence before a blocked result is written."""

    if articulation_candidates_path is None:
        return
    from joint_agent.functions.joint_rigger_core_bridge import (
        _candidate_file_sha256,
        _candidate_readiness_sha256,
    )

    expected_sha256 = _candidate_readiness_sha256(readiness)
    if expected_sha256 is None:
        return
    if (
        _candidate_file_sha256(
            articulation_candidates_path,
            label="articulation candidates file",
        )
        != expected_sha256
    ):
        raise JointRiggerArtifactError(
            "Articulation candidates no longer match candidate readiness"
        )


def _require_candidate_readiness_identity(
    readiness: Mapping[str, Any],
) -> str:
    """Require candidate-consuming adapters to use a bound readiness snapshot."""

    from joint_agent.functions.joint_rigger_core_bridge import (
        _candidate_readiness_sha256,
    )

    expected_sha256 = _candidate_readiness_sha256(readiness)
    if expected_sha256 is None:
        raise JointRiggerArtifactError(
            "Candidate readiness did not bind an exact articulation candidate SHA-256"
        )
    return expected_sha256


def _candidate_id(candidate: Any, index: int) -> str:
    if isinstance(candidate, dict):
        candidate_id = candidate.get("candidate_id") or candidate.get("id")
        if candidate_id:
            return str(candidate_id)
    return f"candidate_{index:04d}"


def _candidate_unready_reasons(candidate: Any) -> list[str]:
    if not isinstance(candidate, dict):
        return ["candidate entry must be a JSON object"]

    reasons: list[str] = []
    review_status = candidate.get("review_status")
    if review_status != _READY_REVIEW_STATUS:
        reasons.append(f"review_status is not {_READY_REVIEW_STATUS}")

    motion_type = str(candidate.get("motion_type") or "").strip().lower()
    if motion_type not in _RIGGABLE_MOTION_TYPES:
        reasons.append("motion_type is not riggable")

    moving_part_prims = candidate.get("moving_part_prims")
    if (
        not isinstance(moving_part_prims, list)
        or not moving_part_prims
        or not all(isinstance(value, str) and value for value in moving_part_prims)
    ):
        reasons.append("moving_part_prims must be a non-empty list of prim paths")

    fixed_parent_prim = candidate.get("fixed_parent_prim")
    if not isinstance(fixed_parent_prim, str) or not fixed_parent_prim.strip():
        reasons.append("fixed_parent_prim is missing")

    if motion_type in _MOTION_TYPES_REQUIRING_AXIS and not _is_vector3(
        candidate.get("motion_axis_world")
    ):
        reasons.append("motion_axis_world must be a 3-vector")

    if "unresolved_reason_codes" not in candidate:
        reasons.append("unresolved_reason_codes must be present")
    elif not isinstance(candidate["unresolved_reason_codes"], list):
        reasons.append("unresolved_reason_codes must be a list")
    elif candidate["unresolved_reason_codes"]:
        reasons.append("unresolved_reason_codes must be empty")

    return reasons


def _unresolved_reason_codes(candidate: Any) -> list[str]:
    if not isinstance(candidate, dict):
        return []
    if "unresolved_reason_codes" not in candidate:
        return []
    reason_codes = candidate["unresolved_reason_codes"]
    if not isinstance(reason_codes, list):
        return [str(reason_codes)]
    return [str(value) for value in reason_codes]


def _is_vector3(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    return all(
        isinstance(component, int | float) and not isinstance(component, bool)
        for component in value
    )


def _base_diagnostics(
    *,
    adapter: str,
    status: str,
    input_usd_path: Path,
    predictions_path: Path | None,
    output_usd_path: Path,
    articulation_candidates_path: Path | None,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "adapter": adapter,
        "status": status,
        "input_usd_path": str(input_usd_path),
        "predictions_path": (
            str(predictions_path) if predictions_path is not None else None
        ),
        "articulation_candidates_path": (
            str(articulation_candidates_path) if articulation_candidates_path else None
        ),
        "output_usd_path": str(output_usd_path),
        "authored_joint_count": 0,
        "warnings": warnings,
        "errors": errors,
    }


def _base_validation(
    *,
    adapter: str,
    status: str,
    output_usd_path: Path | None,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "adapter": adapter,
        "status": status,
        "output_usd_path": str(output_usd_path) if output_usd_path else None,
        "validation_skipped": False,
        "warnings": warnings,
        "errors": errors,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
