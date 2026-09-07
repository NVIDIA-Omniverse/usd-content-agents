# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-phase Material executor for an external asset reasoning loop."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from content_agent_workflows.asset_composition import (
    ArtifactBinding,
    bind_usd_dependency_closure,
    verify_usd_dependency_closure,
)
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.scene_correspondence import SceneOptimizerPathMap
from content_agent_workflows.common.usd_cli_session import (
    WorkflowUsdCliSession,
    stage_up_axis_is_y,
)
from content_agent_workflows.common.validation_evidence import (
    EvidenceArtifact,
    material_assignment_validation_evidence,
)
from content_agent_workflows.material_assignment import (
    MaterialCandidatePolicy,
    MaterialDecisionPolicyError,
    MaterialFinalizationPolicy,
    build_material_authoring_evidence,
    finalize_material_policy,
    load_material_manifest,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .runner import (
    OPTIMIZER_SELECTION_FIXED,
    MaterialAssignConfig,
    _build_request,
    _operation_counts,
    _prepare_run_dir,
    _prepare_usd_cli_material_inputs,
    _prepare_usd_cli_optimized_inspection,
    _publish_materialized_output,
    _reject_unsafe_run_links,
    _validate_config,
    _validate_materialized_usd_output,
    find_repo_root,
)
from .trace import TraceWriter, build_trace

MATERIAL_COORDINATOR_REQUEST_SCHEMA_VERSION: Literal[
    "content-agents.material-coordinator-request.v1"
] = "content-agents.material-coordinator-request.v1"
MATERIAL_COORDINATOR_RESULT_SCHEMA_VERSION: Literal[
    "content-agents.material-coordinator-result.v1"
] = "content-agents.material-coordinator-result.v1"
MATERIAL_COORDINATOR_APPLICATION_SCHEMA_VERSION: Literal[
    "content-agents.material-application-receipt.v1"
] = "content-agents.material-application-receipt.v1"
MATERIAL_POST_APPLY_REVIEW_SCHEMA_VERSION: Literal[
    "content-agents.material-post-apply-review.v1"
] = "content-agents.material-post-apply-review.v1"
MATERIAL_OVRTX_RENDER_TIMEOUT_SECONDS = 300.0


def _material_ovrtx_render_timeout(request: MaterialCoordinatorRequest) -> float:
    """Keep the legacy Material evidence budget as workflow-owned policy."""

    return max(
        MATERIAL_OVRTX_RENDER_TIMEOUT_SECONDS, request.scene_tool_timeout_seconds
    )


class BoundFile(BaseModel):
    """Frozen identity for one coordinator Material input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class MaterialCoordinatorRequest(BaseModel):
    """Frozen preparation/finalization contract for one Material attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agents.material-coordinator-request.v1"] = (
        MATERIAL_COORDINATOR_REQUEST_SCHEMA_VERSION
    )
    run_dir: str
    repository_root: str
    source: BoundFile
    materials_yaml: BoundFile
    materials_usd: BoundFile
    materials_usd_dependencies: list[BoundFile] = Field(default_factory=list)
    reference_images: list[BoundFile] = Field(default_factory=list)
    reference_files: list[BoundFile] = Field(default_factory=list)
    output_usd_path: str
    scene_tool_timeout_seconds: float = Field(gt=0)
    optimize: bool
    root_prim_path: str | None = None
    material_candidate_space: str
    skip_instances: bool
    skip_prototypes: bool
    skip_invisible: bool
    flatten_prototypes: bool | None = None
    enable_deinstance: bool | None = None
    enable_split: bool | None = None
    enable_deduplicate: bool | None = False
    respect_existing_material_bindings: bool
    material_restore_timeout_seconds: float = Field(gt=0)


class MaterialCoordinatorPreparation(BaseModel):
    """Evidence returned before the coordinator authors a decision patch."""

    model_config = ConfigDict(extra="forbid")

    request_path: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_path: str
    packet_binding: BoundFile
    session_id: str = Field(min_length=1)
    scene_tool: Literal["usd-cli"] = "usd-cli"
    visible_candidates_path: str
    visible_candidates_binding: BoundFile
    palette_path: str
    palette_binding: BoundFile
    authoring_context_path: str
    authoring_context_binding: BoundFile
    assignment_seed_path: str
    assignment_seed_binding: BoundFile
    candidate_table_path: str
    candidate_table_binding: BoundFile
    initial_render_paths: list[str] = Field(default_factory=list)
    initial_render_bindings: list[BoundFile] = Field(default_factory=list)
    initial_render_records_binding: BoundFile
    receipt_checkpoint_binding: BoundFile


class MaterialCoordinatorResult(BaseModel):
    """Typed executor/finalizer result reviewed by the asset coordinator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["content-agents.material-coordinator-result.v1"] = (
        MATERIAL_COORDINATOR_RESULT_SCHEMA_VERSION
    )
    status: Literal["pass", "conditional"]
    output_usd_path: str
    output_usd_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: BoundFile
    decision_patch: BoundFile
    evidence: list[BoundFile] = Field(min_length=1)
    unresolved_issues: list[str] = Field(default_factory=list)


class MaterialCoordinatorReviewRequired(BaseModel):
    """Digest-bound application result awaiting an actual post-apply VQA turn."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["content-agents.material-application-receipt.v1"] = (
        MATERIAL_COORDINATOR_APPLICATION_SCHEMA_VERSION
    )
    status: Literal["review_required"] = "review_required"
    request: BoundFile
    decision_patch: BoundFile
    policy: BoundFile
    materialized_usd: BoundFile
    receipt_checkpoint_binding: BoundFile
    final_render_bindings: list[BoundFile] = Field(min_length=1)
    evidence: list[BoundFile] = Field(min_length=1)
    applied_source_prim_paths: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(min_length=1)


class MaterialCoordinatorRelease(BaseModel):
    """Observable cleanup result for one prepared usd-cli session."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["released", "failed"]
    scene_tool: Literal["usd-cli"] = "usd-cli"
    session_id: str = Field(min_length=1)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _MaterialApplicationArtifacts:
    """Exact usd-cli outputs produced while applying one workflow decision."""

    local_output: Path
    applied_paths: tuple[str, ...]
    final_records: tuple[dict[str, Any], ...]
    turntable_path: Path
    receipt_checkpoint_path: Path
    receipt_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _PublishedMaterialResult:
    evidence_paths: tuple[Path, ...]
    status: Literal["pass", "conditional"]
    unresolved_issues: tuple[str, ...]


def _bound(path: Path) -> BoundFile:
    source = path.expanduser()
    if source.is_symlink():
        raise ValueError(f"Material coordinator input is not a regular file: {source}")
    resolved = source.resolve()
    if not resolved.is_file():
        raise ValueError(
            f"Material coordinator input is not a regular file: {resolved}"
        )
    return BoundFile(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _run_artifact(run_dir: Path, path: str | Path) -> Path:
    """Resolve a regular, non-symlink artifact confined to the Material run."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError(f"Material evidence is not a regular run file: {source}")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(run_dir.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(
            f"Material evidence escapes the run directory: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"Material evidence is not a regular run file: {resolved}")
    return resolved


def _bound_dependency_closure(path: Path) -> list[BoundFile]:
    return [
        BoundFile(
            path=binding.path,
            sha256=binding.sha256,
            size_bytes=binding.size_bytes,
        )
        for binding in bind_usd_dependency_closure(path)
    ]


def _request_from_config(
    config: MaterialAssignConfig,
    run_dir: Path,
) -> MaterialCoordinatorRequest:
    if config.output_usd_path is None:
        raise ValueError("Coordinator Material preparation requires output_usd_path")
    return MaterialCoordinatorRequest(
        run_dir=str(run_dir),
        repository_root=str(config.repo_root.resolve()),
        source=_bound(config.usd_path),
        materials_yaml=_bound(config.materials_yaml),
        materials_usd=_bound(config.materials_usd),
        materials_usd_dependencies=_bound_dependency_closure(config.materials_usd),
        reference_images=[_bound(path) for path in config.reference_images],
        reference_files=[_bound(path) for path in config.reference_files or []],
        output_usd_path=str(config.output_usd_path.resolve()),
        scene_tool_timeout_seconds=config.scene_tool_timeout_seconds,
        optimize=config.optimize,
        root_prim_path=config.root_prim_path,
        material_candidate_space=config.material_candidate_space,
        skip_instances=config.skip_instances,
        skip_prototypes=config.skip_prototypes,
        skip_invisible=config.skip_invisible,
        flatten_prototypes=config.flatten_prototypes,
        enable_deinstance=config.enable_deinstance,
        enable_split=config.enable_split,
        enable_deduplicate=config.enable_deduplicate,
        respect_existing_material_bindings=config.respect_existing_material_bindings,
        material_restore_timeout_seconds=config.material_restore_timeout_seconds,
    )


def _staged_usd_input(
    run_dir: Path,
    *,
    label: str,
    expected_sha256: str,
) -> Path:
    """Return one digest-bound staged USD input confined to this run."""

    root = run_dir.resolve(strict=True)
    manifest_path = root / "raw" / f"staged_input_{label}.json"
    try:
        manifest = load_json(manifest_path)
        raw_path = (
            manifest.get("staged_usd_path") if isinstance(manifest, dict) else None
        )
        staged = Path(str(raw_path)).resolve(strict=True)
        staged.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Material coordinator requires the staged {label!r} USD input"
        ) from exc
    if not staged.is_file() or file_sha256(staged) != expected_sha256:
        raise ValueError(
            f"Material coordinator staged {label!r} USD does not match its "
            "frozen input identity"
        )
    return staged


def _staged_scene_inputs(
    run_dir: Path,
    request: MaterialCoordinatorRequest,
) -> tuple[Path, Path]:
    return (
        _staged_usd_input(
            run_dir,
            label="source",
            expected_sha256=request.source.sha256,
        ),
        _staged_usd_input(
            run_dir,
            label="material_library",
            expected_sha256=request.materials_usd.sha256,
        ),
    )


def _session(
    run_dir: Path,
    request: MaterialCoordinatorRequest,
    *,
    receipt_checkpoint_sha256: str | None = None,
) -> WorkflowUsdCliSession:
    staged_source, staged_library = _staged_scene_inputs(run_dir, request)
    return WorkflowUsdCliSession.create(
        owner_root=run_dir,
        project_dir=run_dir,
        identity=str(run_dir),
        workflow="asset-material",
        input_roots=(staged_source, staged_library),
        receipt_checkpoint_sha256=receipt_checkpoint_sha256,
    )


def _render_path(payload: dict[str, Any]) -> Path:
    for artifact in payload.get("artifacts", []):
        if isinstance(artifact, dict) and str(artifact.get("label", "")).startswith(
            "rgb:"
        ):
            return Path(str(artifact["path"])).resolve(strict=True)
    raise RuntimeError("usd-cli render response omitted its OVRTX image")


def _artifact_path(payload: dict[str, Any], label: str) -> Path:
    for artifact in payload.get("artifacts", []):
        if isinstance(artifact, dict) and artifact.get("label") == label:
            return Path(str(artifact["path"])).resolve(strict=True)
    raise RuntimeError(f"usd-cli render response omitted {label!r} evidence")


def _ovrtx_transport_from_probe(payload: object) -> Literal["ovrtx", "remote"]:
    """Accept local OVRTX or an identity-attested remote OVRTX transport."""

    if not isinstance(payload, dict):
        raise ValueError("usd-cli OVRTX probe response is not an object")
    renderer = payload.get("resolved_renderer")
    transport = payload.get("transport")
    if (
        payload.get("schema_version") != "usd-cli.render-probe.v1"
        or payload.get("ready") is not True
        or payload.get("engine") != "ovrtx"
        or renderer not in {"ovrtx", "remote"}
        or transport != ("remote" if renderer == "remote" else "local")
    ):
        raise ValueError("usd-cli probe did not attest the required OVRTX engine")
    if renderer == "remote":
        backends = payload.get("backends")
        if (
            not isinstance(backends, list)
            or not backends
            or any(
                not isinstance(backend, dict)
                or backend.get("engine") != "ovrtx"
                or backend.get("status") not in {"alive", "ready", "healthy"}
                for backend in backends
            )
        ):
            raise ValueError(
                "usd-cli remote transport lacks OVRTX identity attestation"
            )
    return renderer


def _complete_material_preparation(
    *,
    run_dir: Path,
    request: MaterialCoordinatorRequest,
    request_path: Path,
    session: WorkflowUsdCliSession,
    source_usd: Path,
    inspection_usd: Path,
    correspondence: SceneOptimizerPathMap | None,
    optimization: Any,
) -> MaterialCoordinatorPreparation:
    """Complete preparation while the caller owns failure-only cleanup."""

    render_timeout_seconds = _material_ovrtx_render_timeout(request)
    session.open(inspection_usd)
    up_axis_y = stage_up_axis_is_y(inspection_usd)
    ovrtx_probe = session.require_ovrtx(run_dir / "ovrtx_probe")
    _ovrtx_transport_from_probe(ovrtx_probe)
    atomic_write_json(run_dir / "raw" / "ovrtx_render_probe.json", ovrtx_probe)
    evidence = build_material_authoring_evidence(
        run_dir=run_dir,
        session_id=session.session_id,
        source_usd=source_usd,
        inspection_usd=inspection_usd,
        correspondence=correspondence,
        materials_yaml=Path(request.materials_yaml.path),
        materials_usd=Path(request.materials_usd.path),
        policy=MaterialCandidatePolicy(
            material_candidate_space=request.material_candidate_space,
            root_prim_path=request.root_prim_path,
            skip_instances=request.skip_instances,
            skip_prototypes=request.skip_prototypes,
            skip_invisible=request.skip_invisible,
        ),
        respect_existing_material_bindings=request.respect_existing_material_bindings,
    )
    candidates_path = Path(evidence["visible_candidates"])
    table_path = Path(evidence["visible_candidate_table"])
    seed_path = Path(evidence["material_assignment_seed"])
    palette_path = Path(evidence["material_palette"])
    context_path = Path(evidence["material_authoring_context_md"])
    initial_dir = run_dir / "evidence_renders"
    initial_response_dir = run_dir / "raw" / "initial_render_responses"
    initial_render_records = []
    for name, direction in (
        ("initial_top", "+z"),
        ("initial_bottom", "-z"),
        ("initial_oblique", "+x-y+z"),
    ):
        record = session.render_view(
            output_dir=initial_dir,
            name=name,
            direction=direction,
            width=640,
            height=480,
            timeout_seconds=render_timeout_seconds,
            up_axis_y=up_axis_y,
        )
        segmentation = session.run_json(
            [
                "render",
                "--seg",
                "--res",
                "640x480",
                "--output",
                str(initial_dir / f"{name}_segmentation.png"),
            ],
            timeout_seconds=render_timeout_seconds,
        )
        segmentation_response_path = initial_response_dir / f"{name}_segmentation.json"
        atomic_write_json(segmentation_response_path, segmentation)
        record["segmentation_path"] = str(_artifact_path(segmentation, "segmentation"))
        record["segmentation_legend_path"] = str(
            _artifact_path(segmentation, "segmentation legend")
        )
        record["segmentation_response_path"] = str(segmentation_response_path)
        record["segmentation_response"] = segmentation
        initial_render_records.append(record)
    initial_renders = [Path(record["image_path"]) for record in initial_render_records]
    initial_render_records_path = run_dir / "raw" / "initial_render_records.json"
    atomic_write_text(
        initial_render_records_path,
        json.dumps(initial_render_records, indent=2, sort_keys=True) + "\n",
    )
    initial_evidence_paths = [
        _run_artifact(run_dir, str(record[field]))
        for record in initial_render_records
        for field in (
            "image_path",
            "response_path",
            "camera_json_path",
            "segmentation_path",
            "segmentation_legend_path",
            "segmentation_response_path",
        )
    ]
    packet_path = run_dir / "raw" / "material_run_packet.json"
    atomic_write_json(
        packet_path,
        {
            "schema_version": "content-agents.material-run-packet.v2",
            "scene_tool": "usd-cli",
            "session_id": session.session_id,
            "source_usd": request.source.path,
            "inspection_usd": str(inspection_usd),
            "scene_optimizer": (
                optimization.prompt_metadata() if optimization is not None else None
            ),
            "materials_yaml": request.materials_yaml.path,
            "materials_usd": request.materials_usd.path,
            "visible_candidates": str(candidates_path),
            "material_palette": str(palette_path),
            "initial_evidence_renders": initial_render_records,
        },
    )
    result = MaterialCoordinatorPreparation(
        request_path=str(request_path),
        request_sha256=file_sha256(request_path),
        packet_path=str(packet_path),
        packet_binding=_bound(packet_path),
        session_id=session.session_id,
        visible_candidates_path=str(candidates_path),
        visible_candidates_binding=_bound(candidates_path),
        palette_path=str(palette_path),
        palette_binding=_bound(palette_path),
        authoring_context_path=str(context_path),
        authoring_context_binding=_bound(context_path),
        assignment_seed_path=str(seed_path),
        assignment_seed_binding=_bound(seed_path),
        candidate_table_path=str(table_path),
        candidate_table_binding=_bound(table_path),
        initial_render_paths=[str(path) for path in initial_renders],
        initial_render_bindings=[
            _bound(path) for path in dict.fromkeys(initial_evidence_paths)
        ],
        initial_render_records_binding=_bound(initial_render_records_path),
        receipt_checkpoint_binding=_bound(session.receipt_checkpoint_file),
    )
    atomic_write_json(run_dir / "coordinator_preparation.json", result)
    return result


def prepare_material_for_coordinator(
    config: MaterialAssignConfig,
) -> MaterialCoordinatorPreparation:
    """Prepare deterministic candidate, palette, and OVRTX evidence."""

    _validate_config(config)
    if config.optimizer_selection != OPTIMIZER_SELECTION_FIXED:
        raise ValueError(
            "Coordinator must supply explicit optimizer settings before Material prepare"
        )
    run_dir = _prepare_run_dir(config)
    if (run_dir / "coordinator_request.json").exists():
        raise ValueError(
            f"Coordinator Material preparation requires a fresh run: {run_dir}"
        )
    generic_request = _build_request(config, run_dir)
    atomic_write_json(run_dir / "request.json", generic_request)
    request = _request_from_config(config, run_dir)
    request_path = run_dir / "coordinator_request.json"
    atomic_write_json(request_path, request)
    _prepare_usd_cli_material_inputs(
        config=config,
        run_dir=run_dir,
        trusted_source_baseline=None,
    )
    source_usd = Path(request.source.path)
    staged_source, _staged_library = _staged_scene_inputs(run_dir, request)
    inspection_usd = staged_source
    correspondence: SceneOptimizerPathMap | None = None
    optimization = None
    if request.optimize:
        optimization = _prepare_usd_cli_optimized_inspection(
            config=config,
            run_dir=run_dir,
            trace_writer=TraceWriter(run_dir),
        )
        inspection_usd = optimization.inspection_usd_path
        correspondence_payload = load_json(optimization.correspondence_path)
        if not isinstance(correspondence_payload, dict):
            raise ValueError("Scene Optimizer correspondence artifact is not an object")
        source_to_inspection = correspondence_payload.get("source_to_inspection")
        inspection_to_source = correspondence_payload.get("inspection_to_source")
        if not isinstance(source_to_inspection, dict) or not isinstance(
            inspection_to_source, dict
        ):
            raise ValueError("Scene Optimizer correspondence artifact is incomplete")
        correspondence = SceneOptimizerPathMap(
            source_to_inspection_map={
                str(path): [str(value) for value in values]
                for path, values in source_to_inspection.items()
                if isinstance(path, str) and isinstance(values, list)
            },
            inspection_to_source_map={
                str(path): [str(value) for value in values]
                for path, values in inspection_to_source.items()
                if isinstance(path, str) and isinstance(values, list)
            },
        )
    elif request.material_candidate_space == "inspection":
        # The staged source is byte-identical but has a different file identity.
        # An empty map explicitly describes the identity prim-path transform.
        correspondence = SceneOptimizerPathMap()
    session = _session(run_dir, request)
    try:
        return _complete_material_preparation(
            run_dir=run_dir,
            request=request,
            request_path=request_path,
            session=session,
            source_usd=source_usd,
            inspection_usd=inspection_usd,
            correspondence=correspondence,
            optimization=optimization,
        )
    except Exception:  # noqa: BLE001 - cleanup must preserve the original failure
        try:
            session.close()
        except Exception:  # noqa: BLE001 - best-effort failure cleanup
            pass
        raise


def _load_request(run_dir: Path) -> MaterialCoordinatorRequest:
    try:
        request = MaterialCoordinatorRequest.model_validate(
            load_json(run_dir / "coordinator_request.json")
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(
            f"Invalid prepared Material coordinator request: {exc}"
        ) from exc
    if Path(request.run_dir) != run_dir:
        raise ValueError("Prepared Material request belongs to another run directory")
    return request


def _load_preparation(run_dir: Path) -> MaterialCoordinatorPreparation:
    try:
        return MaterialCoordinatorPreparation.model_validate(
            load_json(run_dir / "coordinator_preparation.json")
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(
            f"Invalid prepared Material coordinator evidence: {exc}"
        ) from exc


def _verify_binding(binding: BoundFile) -> None:
    if _bound(Path(binding.path)) != binding:
        raise ValueError(f"Prepared Material evidence identity changed: {binding.path}")


def _verify_parent_carried_manifest_digest(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"{label} requires a valid parent-carried SHA-256 digest")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{label} identity changed since the parent phase boundary")


def _verify_prepared_evidence(
    run_dir: Path,
    preparation: MaterialCoordinatorPreparation,
    *,
    verify_receipt_checkpoint: bool = True,
) -> None:
    request_path = run_dir / "coordinator_request.json"
    if Path(preparation.request_path).resolve() != request_path:
        raise ValueError("Material preparation binds another coordinator request")
    if file_sha256(request_path) != preparation.request_sha256:
        raise ValueError("Prepared Material coordinator request identity changed")
    bindings = [
        preparation.packet_binding,
        preparation.visible_candidates_binding,
        preparation.palette_binding,
        preparation.authoring_context_binding,
        preparation.assignment_seed_binding,
        preparation.candidate_table_binding,
        preparation.initial_render_records_binding,
        *preparation.initial_render_bindings,
    ]
    if verify_receipt_checkpoint:
        bindings.append(preparation.receipt_checkpoint_binding)
    for binding in bindings:
        _verify_binding(binding)


def _release_material_session(
    run_dir: Path,
    request: MaterialCoordinatorRequest,
    session_id: str,
    *,
    receipt_checkpoint_sha256: str,
) -> MaterialCoordinatorRelease:
    error: str | None = None
    try:
        session = _session(
            run_dir,
            request,
            receipt_checkpoint_sha256=receipt_checkpoint_sha256,
        )
        if session.session_id != session_id:
            raise ValueError("Prepared usd-cli session identity changed")
        session.close()
        status: Literal["released", "failed"] = "released"
    except Exception as exc:  # noqa: BLE001 - cleanup result is observable
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    result = MaterialCoordinatorRelease(
        status=status,
        session_id=session_id,
        error=error,
    )
    atomic_write_json(run_dir / "raw" / "material_session_release.json", result)
    return result


def release_material_for_coordinator(
    run_dir: str | Path,
    *,
    preparation_sha256: str,
    application_receipt_sha256: str | None = None,
) -> MaterialCoordinatorRelease:
    """Release the one prepared usd-cli sidecar after an early stop."""

    root = Path(run_dir).expanduser().resolve()
    _reject_unsafe_run_links(root)
    _verify_parent_carried_manifest_digest(
        root / "coordinator_preparation.json",
        preparation_sha256,
        label="Material coordinator preparation",
    )
    preparation = _load_preparation(root)
    request = _load_request(root)
    application_receipt_path = root / "raw" / "material_application_receipt.json"
    checkpoint = preparation.receipt_checkpoint_binding
    if application_receipt_path.is_file():
        if application_receipt_sha256 is None:
            raise ValueError(
                "Material application release requires its parent-carried receipt digest"
            )
        _verify_parent_carried_manifest_digest(
            application_receipt_path,
            application_receipt_sha256,
            label="Material application receipt",
        )
        checkpoint = _load_application_receipt(root).receipt_checkpoint_binding
    elif application_receipt_sha256 is not None:
        raise ValueError("Material application receipt does not exist for release")
    return _release_material_session(
        root,
        request,
        preparation.session_id,
        receipt_checkpoint_sha256=checkpoint.sha256,
    )


def _verify_request_inputs(request: MaterialCoordinatorRequest) -> None:
    for binding in (
        request.source,
        request.materials_yaml,
        request.materials_usd,
        *request.reference_images,
        *request.reference_files,
    ):
        _verify_binding(binding)
    verify_usd_dependency_closure(
        request.materials_usd.path,
        [
            ArtifactBinding(
                path=binding.path,
                sha256=binding.sha256,
                size_bytes=binding.size_bytes,
            )
            for binding in request.materials_usd_dependencies
        ],
    )


def _validate_coordinator_decision_patch(
    payload: dict[str, object],
    *,
    run_dir: Path,
    preparation: MaterialCoordinatorPreparation,
    request: MaterialCoordinatorRequest,
) -> None:
    if payload.get("schema_version") != "content-agents.material-decision-patch.v1":
        raise ValueError("Material decision patch has the wrong schema version")
    for key in ("material_assignments", "reviewed_no_override"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Material decision patch requires {key}")
    assessment = payload.get("visual_quality_assessment")
    if not isinstance(assessment, dict):
        raise ValueError("Material decision patch requires visual_quality_assessment")
    status = assessment.get("status")
    if status not in {"pass", "fixed", "unresolved_issues"}:
        raise ValueError("Material visual quality assessment has an invalid status")
    unresolved = assessment.get("unresolved_issues")
    if not isinstance(unresolved, list) or any(
        not isinstance(item, str) or not item.strip() for item in unresolved
    ):
        raise ValueError(
            "Material visual quality assessment has invalid unresolved issues"
        )
    if (status == "unresolved_issues") != bool(unresolved):
        raise ValueError("Material visual quality assessment status is inconsistent")
    checked = assessment.get("checked_views")
    if not isinstance(checked, list) or not checked:
        raise ValueError("Material visual quality assessment requires checked views")
    expected_bindings = (
        *preparation.initial_render_bindings,
        *request.reference_images,
        *request.reference_files,
    )
    initial_paths = {
        Path(binding.path).resolve() for binding in preparation.initial_render_bindings
    }
    expected_paths = {Path(binding.path).resolve() for binding in expected_bindings}
    checked_paths = [Path(str(value)).expanduser().resolve() for value in checked]
    if (
        len(checked_paths) != len(expected_paths)
        or set(checked_paths) != expected_paths
    ):
        raise ValueError(
            "Material visual quality assessment must check the exact sealed initial "
            "OVRTX and frozen reference evidence"
        )
    for binding in expected_bindings:
        _verify_binding(binding)
        path = Path(binding.path).resolve()
        if path in initial_paths:
            _run_artifact(run_dir, path)


def _assignment_paths(group: dict[str, Any]) -> list[str]:
    for key in ("source_prim_paths", "prim_paths", "runtime_prim_paths"):
        values = group.get(key)
        if isinstance(values, list) and values:
            return [str(value) for value in values]
    return []


def _response_data(response: object, *, operation: str) -> dict[str, Any]:
    """Return the typed payload from a usd-cli command response.

    The workflow evaluates this evidence; usd-cli merely reports the scene fact.
    Keeping the check here prevents the scene tool from acquiring workflow policy.
    """

    if not isinstance(response, dict) or response.get("ok") is not True:
        raise ValueError(f"usd-cli {operation} did not succeed")
    data = response.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"usd-cli {operation} returned no structured audit data")
    return data


def _require_clean_slate_appearance_audit(response: object) -> dict[str, Any]:
    """Fail closed unless the post-clear usd-cli audit confirms a blank appearance."""

    data = _response_data(response, operation="appearance audit")
    counts = data.get("counts")
    if data.get("clear") is not True or not isinstance(counts, dict):
        raise ValueError(
            "usd-cli appearance audit did not confirm a clean-slate overlay"
        )
    required_zero_counts = (
        "effective_material_bindings",
        "effective_shader_appearances",
        "direct_shader_outputs",
        "display_values",
    )
    if any(counts.get(key) != 0 for key in required_zero_counts):
        raise ValueError("usd-cli appearance audit found residual source appearance")
    return data


def _require_material_audit_aggregate(
    response: object,
) -> dict[str, Any]:
    """Require the exact aggregate audit while allowing bounded list truncation."""

    data = _response_data(response, operation="material audit")
    counts = data.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("usd-cli material audit omitted aggregate binding evidence")
    if counts.get("invalid_binding_targets") != 0:
        raise ValueError("usd-cli material audit found invalid material bindings")
    return data


def _verify_selected_material_bindings(
    *,
    session: WorkflowUsdCliSession,
    decision: dict[str, Any],
    imported_material_paths: dict[str, str],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Verify every selected prim or GeomSubset through a neutral point query."""

    records: list[dict[str, Any]] = []
    assignments = decision.get("material_assignments")
    assert isinstance(assignments, list)
    for group in assignments:
        assert isinstance(group, dict)
        expected_name = str(group.get("material_name") or "").strip()
        expected_source_path = str(group.get("material_path") or "").strip()
        # The manifest binding is authoritative. material_name is only a display
        # label for workflow reasoning/evidence; the exact local identity comes
        # from usd-cli's import response and may encode the full source path.
        from pxr import Sdf

        source_path = Sdf.Path(expected_source_path)
        if (
            not source_path.IsAbsolutePath()
            or not source_path.IsPrimPath()
            or source_path == Sdf.Path.absoluteRootPath
        ):
            raise ValueError(
                f"Material assignment has an invalid manifest path: {expected_source_path!r}"
            )
        expected_local_path = imported_material_paths.get(expected_source_path)
        if expected_local_path is None:
            raise ValueError(
                "usd-cli material import omitted the local identity for manifest "
                f"path {expected_source_path!r}"
            )
        for path in _assignment_paths(group):
            response = session.run_json(
                ["material-binding", path],
                timeout_seconds=timeout_seconds,
            )
            data = _response_data(response, operation=f"material binding query {path}")
            material = data.get("bound_material_path")
            if material != expected_local_path:
                raise ValueError(
                    "usd-cli material binding query found a target without its expected "
                    "chosen material: "
                    f"{path} -> {material!r}, expected {expected_local_path!r} "
                    f"for {expected_name!r}"
                )
            records.append(
                {
                    "prim_path": path,
                    "material_name": expected_name,
                    "material_source_path": expected_source_path,
                    "bound_material_path": material,
                    "status": "pass",
                    "query_response": response,
                }
            )
    return records


def _apply_material_decision(
    *,
    run_dir: Path,
    request: MaterialCoordinatorRequest,
    preparation: MaterialCoordinatorPreparation,
    decision: dict[str, Any],
) -> _MaterialApplicationArtifacts:
    render_timeout_seconds = _material_ovrtx_render_timeout(request)
    session = _session(
        run_dir,
        request,
        receipt_checkpoint_sha256=preparation.receipt_checkpoint_binding.sha256,
    )
    if session.session_id != preparation.session_id:
        raise ValueError("Prepared usd-cli session identity changed")
    staged_source, staged_library = _staged_scene_inputs(run_dir, request)
    # Preparation's camera commands intentionally leave unsaved view state in the
    # parent-owned session. Finalization is a new digest-bound phase whose source
    # of truth is the unchanged staged file, so explicitly discard only that view
    # state before authoring the accepted Material decision.
    session.open(staged_source, force_reload=True)
    up_axis_y = stage_up_axis_is_y(staged_source)
    applied_paths: list[str] = []
    imported_material_paths: dict[str, str] = {}
    operation_receipts: list[dict[str, Any]] = []
    assignments = decision.get("material_assignments")
    assert isinstance(assignments, list)
    selected_paths = {
        path
        for group in assignments
        if isinstance(group, dict)
        for path in _assignment_paths(group)
    }
    candidate_payload = load_json(Path(preparation.visible_candidates_binding.path))
    candidate_rows = (
        candidate_payload.get("candidates")
        if isinstance(candidate_payload, dict)
        else None
    )
    if not isinstance(candidate_rows, list):
        raise ValueError("Material candidate evidence omitted candidate rows")
    deinstance_roots = sorted(
        {
            str(root)
            for candidate in candidate_rows
            if isinstance(candidate, dict)
            and str(candidate.get("source_path") or "") in selected_paths
            for root in candidate.get("deinstance_root_paths", [])
            if isinstance(root, str) and root.startswith("/")
        }
    )
    for root_path in deinstance_roots:
        response = session.run_json(
            ["set", root_path, "instanceable", "false"],
            timeout_seconds=request.scene_tool_timeout_seconds,
        )
        operation_receipts.append(
            {
                "operation": "deinstance",
                "source_prim_path": root_path,
                "response": response,
            }
        )
    if not request.respect_existing_material_bindings:
        clear_response = session.run_json(["appearance", "clear"])
        appearance_audit = session.run_json(["appearance", "audit"])
        _require_clean_slate_appearance_audit(appearance_audit)
        cleared_working_usd = run_dir / "raw" / "appearance_cleared.usda"
        clear_save_response = session.run_json(
            ["save", str(cleared_working_usd), "--flatten"],
            timeout_seconds=request.material_restore_timeout_seconds,
        )
        # appearance clear authors its masking opinions in an anonymous session
        # layer. Reopen the flattened derivative so subsequent library references
        # are authored relative to a file-backed root layer.
        session.open(cleared_working_usd)
        atomic_write_json(
            run_dir / "raw" / "appearance_clear_report.json",
            {
                "schema_version": "content-agents.appearance-clear-report.v1",
                "scene_tool": "usd-cli",
                "clear_response": clear_response,
                "audit_response": appearance_audit,
                "save_response": clear_save_response,
                "cleared_usd": _bound(cleared_working_usd).model_dump(mode="json"),
                "status": "pass",
            },
        )
    for raw_group in assignments:
        if not isinstance(raw_group, dict):
            raise ValueError("Material assignment group must be an object")
        name = str(raw_group.get("material_name") or "").strip()
        material_path = str(raw_group.get("material_path") or "").strip()
        if not name or not material_path.startswith("/"):
            raise ValueError("Material assignment requires manifest name and path")
        paths = _assignment_paths(raw_group)
        if not paths:
            raise ValueError("Material assignment group has no target prim paths")
        for prim_path in paths:
            response = session.run_json(
                [
                    "material",
                    prim_path,
                    "--library",
                    str(staged_library),
                    "--library-prim",
                    material_path,
                ],
                timeout_seconds=request.scene_tool_timeout_seconds,
            )
            response_data = _response_data(
                response,
                operation=f"material import {material_path} for {prim_path}",
            )
            imported_path = response_data.get("material_path")
            if not isinstance(imported_path, str) or not imported_path.startswith("/"):
                raise ValueError(
                    "usd-cli exact material import omitted its local material path"
                )
            previous_import = imported_material_paths.setdefault(
                material_path,
                imported_path,
            )
            if previous_import != imported_path:
                raise ValueError(
                    "usd-cli exact material import changed local identity within one "
                    f"workflow session: {material_path} -> {previous_import}, {imported_path}"
                )
            operation_receipts.append(
                {
                    "operation": "material",
                    "source_prim_path": prim_path,
                    "material_name": name,
                    "material_source_path": material_path,
                    "bound_material_path": imported_path,
                    "response": response,
                }
            )
            applied_paths.append(prim_path)
    material_audit = session.run_json(
        ["material", "audit", "--effective", "--include-subsets"],
        timeout_seconds=request.scene_tool_timeout_seconds,
    )
    aggregate_audit = _require_material_audit_aggregate(material_audit)
    binding_records = _verify_selected_material_bindings(
        session=session,
        decision=decision,
        imported_material_paths=imported_material_paths,
        timeout_seconds=request.scene_tool_timeout_seconds,
    )
    atomic_write_json(
        run_dir / "raw" / "material_binding_audit.json",
        {
            "schema_version": "content-agents.material-binding-audit.v1",
            "scene_tool": "usd-cli",
            "status": "pass",
            "expected_target_count": len(binding_records),
            "verified_target_count": sum(
                record["status"] == "pass" for record in binding_records
            ),
            "records": binding_records,
            "errors": [],
            "aggregate_audit_response": material_audit,
            "aggregate_audit_truncated": aggregate_audit.get("truncated") is True,
            "aggregate_omitted_counts": {
                key: value
                for key, value in aggregate_audit.items()
                if key.endswith("_omitted")
            },
        },
    )
    local_output = run_dir / "output" / "materialized.usda"
    save_response = session.run_json(
        ["save", str(local_output), "--flatten"],
        timeout_seconds=request.material_restore_timeout_seconds,
    )
    # Final evidence must be rendered from the exact on-disk derivative whose
    # digest is attached to every render record, not an equivalent live stage.
    session.open(local_output)
    up_axis_y = stage_up_axis_is_y(local_output)
    rendered_usd = _bound(local_output)
    final_dir = run_dir / "final_renders"
    final_records: list[dict[str, Any]] = []
    final_response_dir = run_dir / "raw" / "final_render_responses"
    for name, direction in (
        ("final_top", "+z"),
        ("final_oblique", "+x-y+z"),
        ("final_side_px", "+x"),
        ("final_front_py", "+y"),
    ):
        record = session.render_view(
            output_dir=final_dir,
            name=name,
            direction=direction,
            width=640,
            height=480,
            timeout_seconds=render_timeout_seconds,
            up_axis_y=up_axis_y,
        )
        if record.get("renderer") not in {"ovrtx", "remote"}:
            raise ValueError(
                f"Material final render {name!r} did not use the required OVRTX backend"
            )
        segmentation = session.run_json(
            [
                "render",
                "--seg",
                "--res",
                "640x480",
                "--output",
                str(final_dir / f"{name}_segmentation.png"),
            ],
            timeout_seconds=render_timeout_seconds,
        )
        if segmentation.get("ok") is not True:
            raise ValueError(f"usd-cli final segmentation {name!r} did not succeed")
        segmentation_response_path = final_response_dir / f"{name}_segmentation.json"
        atomic_write_json(segmentation_response_path, segmentation)
        record["segmentation_path"] = str(_artifact_path(segmentation, "segmentation"))
        record["segmentation_legend_path"] = str(
            _artifact_path(segmentation, "segmentation legend")
        )
        record["segmentation_response_path"] = str(segmentation_response_path)
        record["segmentation_response"] = segmentation
        record["kind"] = "verification_view"
        record["rendered_usd"] = rendered_usd.model_dump(mode="json")
        final_records.append(record)
    turntable_frames: list[Path] = []
    for index in range(24):
        angle = 2.0 * math.pi * index / 24
        direction = f"{math.cos(angle):+.8f}x{math.sin(angle):+.8f}y+0.35000000z"
        name = f"final_turntable_{index:03d}"
        record = session.render_view(
            output_dir=final_dir,
            name=name,
            direction=direction,
            width=640,
            height=480,
            timeout_seconds=render_timeout_seconds,
            up_axis_y=up_axis_y,
        )
        if record.get("renderer") not in {"ovrtx", "remote"}:
            raise ValueError(
                f"Material turntable frame {index} did not use the required OVRTX engine"
            )
        record["kind"] = "turntable_frame"
        record["rendered_usd"] = rendered_usd.model_dump(mode="json")
        final_records.append(record)
        turntable_frames.append(Path(str(record["image_path"])))
    turntable_path = final_dir / "final_turntable.gif"
    from PIL import Image

    frames = []
    for path in turntable_frames:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    frames[0].save(
        turntable_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    operation_receipts_path = run_dir / "raw" / "material_operation_receipts.json"
    atomic_write_json(
        operation_receipts_path,
        {
            "schema_version": "content-agents.material-operation-receipts.v1",
            "scene_tool": "usd-cli",
            "operations": operation_receipts,
            "save_response": save_response,
        },
    )
    final_records_path = run_dir / "raw" / "final_render_records.json"
    atomic_write_json(
        final_records_path,
        {
            "schema_version": "content-agents.material-final-renders.v1",
            "scene_tool": "usd-cli",
            "render_engine": "ovrtx",
            "transports": sorted({str(record["renderer"]) for record in final_records}),
            "turntable": {
                "frame_count": len(turntable_frames),
                "gif_path": str(turntable_path),
            },
            "renders": final_records,
        },
    )
    return _MaterialApplicationArtifacts(
        local_output=local_output,
        applied_paths=tuple(sorted(set(applied_paths))),
        final_records=tuple(final_records),
        turntable_path=turntable_path,
        receipt_checkpoint_path=session.receipt_checkpoint_file,
        receipt_paths=(operation_receipts_path, final_records_path),
    )


def _deduplicated_strings(values: list[object]) -> list[str]:
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def _final_render_evidence(
    *,
    run_dir: Path,
    records: tuple[dict[str, Any], ...],
) -> tuple[list[Path], list[Path], list[EvidenceArtifact]]:
    """Validate and enumerate every final OVRTX artifact and exact response."""

    images: list[Path] = []
    paths: list[Path] = []
    evidence: list[EvidenceArtifact] = []
    probe = load_json(run_dir / "raw" / "ovrtx_render_probe.json")
    attested_transport = _ovrtx_transport_from_probe(probe)
    base_fields = (
        ("image_path", "render", "Final OVRTX RGB render."),
        ("response_path", "render_response", "Exact usd-cli render response."),
        ("camera_json_path", "camera", "Camera parameters for the final render."),
    )
    segmentation_fields = (
        ("segmentation_path", "segmentation", "Final OVRTX segmentation render."),
        (
            "segmentation_legend_path",
            "segmentation_legend",
            "Legend for the final segmentation render.",
        ),
        (
            "segmentation_response_path",
            "segmentation_response",
            "Exact usd-cli segmentation response.",
        ),
    )
    for record in records:
        name = str(record.get("name") or "unnamed")
        if record.get("renderer") != attested_transport:
            raise ValueError(
                f"Final Material render {name!r} does not match its attested "
                "OVRTX transport"
            )
        render_response_path = _run_artifact(run_dir, str(record["response_path"]))
        render_response = load_json(render_response_path)
        summary = (
            render_response.get("summary")
            if isinstance(render_response, dict)
            else None
        )
        if (
            not isinstance(summary, dict)
            or summary.get("backend") != attested_transport
        ):
            raise ValueError(
                f"Final Material render {name!r} lacks an exact OVRTX backend response"
            )
        rendered_usd_payload = record.get("rendered_usd")
        if not isinstance(rendered_usd_payload, dict):
            raise ValueError(
                f"Final Material render {name!r} omitted its rendered USD identity"
            )
        rendered_usd_path = _run_artifact(
            run_dir,
            str(rendered_usd_payload.get("path") or ""),
        )
        rendered_usd = _bound(rendered_usd_path)
        if rendered_usd_payload != rendered_usd.model_dump(mode="json"):
            raise ValueError(
                f"Final Material render {name!r} has a changed rendered USD identity"
            )
        fields = (
            base_fields
            if record.get("kind") == "turntable_frame"
            else (*base_fields, *segmentation_fields)
        )
        for field, kind, description in fields:
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Final Material render {name!r} omitted {field}")
            path = _run_artifact(run_dir, value)
            binding = _bound(path)
            paths.append(path)
            if field == "image_path":
                images.append(path)
            evidence.append(
                EvidenceArtifact(
                    kind=kind,
                    path=str(path),
                    description=description,
                    metadata={
                        "view": name,
                        "render_engine": "ovrtx",
                        "transport": attested_transport,
                        "sha256": binding.sha256,
                        "size_bytes": binding.size_bytes,
                        "rendered_usd_path": rendered_usd.path,
                        "rendered_usd_sha256": rendered_usd.sha256,
                        "rendered_usd_size_bytes": rendered_usd.size_bytes,
                    },
                )
            )
    return images, paths, evidence


def _application_evidence_paths(
    *,
    run_dir: Path,
    application: _MaterialApplicationArtifacts,
) -> tuple[Path, ...]:
    _images, final_artifacts, _validation = _final_render_evidence(
        run_dir=run_dir,
        records=application.final_records,
    )
    preparation = _load_preparation(run_dir)
    paths = [
        run_dir / "coordinator_request.json",
        run_dir / "coordinator_preparation.json",
        run_dir / "raw" / "material_applied_decision_patch.json",
        run_dir / "raw" / "material_finalization_policy.json",
        run_dir / "raw" / "material_run_packet.json",
        run_dir / "raw" / "ovrtx_render_probe.json",
        run_dir / "raw" / "visible_candidate_prims.json",
        run_dir / "raw" / "material_palette.json",
        run_dir / "raw" / "material_authoring_context.md",
        run_dir / "raw" / "material_assignment_seed.json",
        run_dir / "raw" / "visible_candidate_table.tsv",
        run_dir / "raw" / "material_binding_audit.json",
        application.local_output,
        application.turntable_path,
        application.receipt_checkpoint_path,
        *application.receipt_paths,
        *[Path(binding.path) for binding in preparation.initial_render_bindings],
        Path(preparation.initial_render_records_binding.path),
        *final_artifacts,
    ]
    clear_report = run_dir / "raw" / "appearance_clear_report.json"
    if clear_report.is_file():
        paths.append(clear_report)
    cleared_usd = run_dir / "raw" / "appearance_cleared.usda"
    if cleared_usd.is_file():
        paths.append(cleared_usd)
    command_receipts = run_dir / "raw" / "usd_cli_command_receipts.jsonl"
    if command_receipts.is_file():
        paths.append(command_receipts)
    return tuple(dict.fromkeys(_run_artifact(run_dir, path) for path in paths))


def _write_application_receipt(
    *,
    run_dir: Path,
    application: _MaterialApplicationArtifacts,
) -> MaterialCoordinatorReviewRequired:
    images, _final_artifacts, _validation = _final_render_evidence(
        run_dir=run_dir,
        records=application.final_records,
    )
    receipt = MaterialCoordinatorReviewRequired(
        request=_bound(run_dir / "coordinator_request.json"),
        decision_patch=_bound(run_dir / "raw" / "material_applied_decision_patch.json"),
        policy=_bound(run_dir / "raw" / "material_finalization_policy.json"),
        materialized_usd=_bound(application.local_output),
        receipt_checkpoint_binding=_bound(application.receipt_checkpoint_path),
        final_render_bindings=[
            *[_bound(path) for path in images],
            _bound(application.turntable_path),
        ],
        evidence=[
            _bound(path)
            for path in _application_evidence_paths(
                run_dir=run_dir,
                application=application,
            )
        ],
        applied_source_prim_paths=list(application.applied_paths),
        unresolved_issues=[
            "Post-apply OVRTX renders require an evidence-bound visual review."
        ],
    )
    atomic_write_json(run_dir / "raw" / "material_application_receipt.json", receipt)
    return receipt


def _load_application_receipt(run_dir: Path) -> MaterialCoordinatorReviewRequired:
    try:
        receipt = MaterialCoordinatorReviewRequired.model_validate(
            load_json(run_dir / "raw" / "material_application_receipt.json")
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(f"Invalid Material application receipt: {exc}") from exc
    for binding in (
        receipt.request,
        receipt.decision_patch,
        receipt.policy,
        receipt.materialized_usd,
        receipt.receipt_checkpoint_binding,
        *receipt.final_render_bindings,
        *receipt.evidence,
    ):
        _verify_binding(binding)
        _run_artifact(run_dir, binding.path)
    return receipt


def _validate_post_apply_review(
    payload: dict[str, Any],
    *,
    run_dir: Path,
    receipt: MaterialCoordinatorReviewRequired,
) -> dict[Path, BoundFile]:
    if payload.get("schema_version") != MATERIAL_POST_APPLY_REVIEW_SCHEMA_VERSION:
        raise ValueError("Material post-apply review has the wrong schema version")
    status = payload.get("status")
    if status not in {"pass", "fixed", "unresolved_issues"}:
        raise ValueError("Material post-apply review has an invalid status")
    unresolved = payload.get("unresolved_issues")
    if not isinstance(unresolved, list) or any(
        not isinstance(item, str) or not item.strip() for item in unresolved
    ):
        raise ValueError("Material post-apply review has invalid unresolved issues")
    if (status == "unresolved_issues") != bool(unresolved):
        raise ValueError("Material post-apply review status is inconsistent")
    checked = payload.get("checked_views")
    if not isinstance(checked, list) or any(
        not isinstance(item, str) or not item.strip() for item in checked
    ):
        raise ValueError("Material post-apply review requires checked views")
    expected = {
        Path(binding.path): binding for binding in receipt.final_render_bindings
    }
    checked_paths = {Path(value).expanduser().resolve() for value in checked}
    if checked_paths != set(expected):
        raise ValueError(
            "Material post-apply review must check every exact final OVRTX render"
        )
    raw_bindings = payload.get("checked_view_bindings")
    if not isinstance(raw_bindings, list):
        raise ValueError("Material post-apply review requires checked view bindings")
    try:
        supplied = [BoundFile.model_validate(value) for value in raw_bindings]
    except ValidationError as exc:
        raise ValueError(f"Invalid Material checked view binding: {exc}") from exc
    supplied_by_path = {Path(binding.path): binding for binding in supplied}
    if supplied_by_path != expected or len(supplied) != len(expected):
        raise ValueError(
            "Material post-apply review bindings differ from the applied render receipt"
        )
    for path, binding in supplied_by_path.items():
        _run_artifact(run_dir, path)
        _verify_binding(binding)
    return supplied_by_path


def _application_from_receipt(
    run_dir: Path,
    receipt: MaterialCoordinatorReviewRequired,
) -> _MaterialApplicationArtifacts:
    payload = load_json(run_dir / "raw" / "final_render_records.json")
    records = payload.get("renders") if isinstance(payload, dict) else None
    turntable = payload.get("turntable") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("Material final render receipt is invalid")
    if not isinstance(turntable, dict) or not isinstance(
        turntable.get("gif_path"), str
    ):
        raise ValueError("Material turntable receipt is invalid")
    return _MaterialApplicationArtifacts(
        local_output=Path(receipt.materialized_usd.path),
        applied_paths=tuple(receipt.applied_source_prim_paths),
        final_records=tuple(records),
        turntable_path=_run_artifact(run_dir, str(turntable["gif_path"])),
        receipt_checkpoint_path=Path(receipt.receipt_checkpoint_binding.path),
        receipt_paths=(
            run_dir / "raw" / "material_operation_receipts.json",
            run_dir / "raw" / "final_render_records.json",
        ),
    )


def _policy_from_receipt(
    run_dir: Path,
    receipt: MaterialCoordinatorReviewRequired,
) -> MaterialFinalizationPolicy:
    decision = load_json(Path(receipt.decision_patch.path))
    policy = load_json(Path(receipt.policy.path))
    if not isinstance(decision, dict) or not isinstance(policy, dict):
        raise ValueError("Material application policy receipt is invalid")
    assignments = decision.get("material_assignments")
    reviewed = decision.get("reviewed_no_override")
    rejected = policy.get("rejected_groups")
    coverage = policy.get("coverage")
    if (
        not isinstance(assignments, list)
        or not isinstance(reviewed, list)
        or not isinstance(rejected, list)
        or not isinstance(coverage, dict)
    ):
        raise ValueError("Material application policy receipt is incomplete")
    return MaterialFinalizationPolicy(
        material_assignments=tuple(assignments),
        reviewed_no_override=tuple(reviewed),
        rejected_groups=tuple(rejected),
        coverage={str(key): int(value) for key, value in coverage.items()},
    )


def _publish_materialized_usd(source: Path, output: Path) -> None:
    """Publish a materialized layer using the requested USD encoding."""

    output_suffix = output.suffix.lower()
    if output_suffix not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError(
            f"Material output must use a .usd, .usda, .usdc, or .usdz suffix: {output}"
        )

    if output_suffix == ".usdz":
        from pxr import Sdf, UsdUtils

        with tempfile.TemporaryDirectory(
            prefix="content-workflow-material-package-"
        ) as scratch_directory:
            package = Path(scratch_directory) / "materialized.usdz"
            if not UsdUtils.CreateNewUsdzPackage(
                Sdf.AssetPath(str(source)),
                str(package),
            ):
                raise RuntimeError(f"Could not package materialized USD: {output}")
            _validate_materialized_usd_output(package)
            _publish_materialized_output(package, output)
    else:
        _publish_materialized_output(source, output)

    _validate_materialized_usd_output(output)


def _publish_result_artifacts(
    *,
    run_dir: Path,
    request: MaterialCoordinatorRequest,
    decision: dict[str, Any],
    review_assessment: dict[str, Any],
    policy: MaterialFinalizationPolicy,
    application: _MaterialApplicationArtifacts,
    supplied_checked_views: dict[Path, BoundFile],
) -> _PublishedMaterialResult:
    output = Path(request.output_usd_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish_materialized_usd(application.local_output, output)

    final_images, final_artifact_paths, validation_artifacts = _final_render_evidence(
        run_dir=run_dir,
        records=application.final_records,
    )
    final_reviewed = bool(final_images) and all(
        supplied_checked_views.get(path) == _bound(path) for path in final_images
    )
    supplied_assessment = dict(review_assessment)
    supplied_unresolved = supplied_assessment.get("unresolved_issues")
    assert isinstance(supplied_unresolved, list)
    unresolved = _deduplicated_strings(list(supplied_unresolved))
    if not final_reviewed:
        unresolved.append(
            "Post-apply OVRTX renders require an evidence-bound visual review in a "
            "separate coordinator turn."
        )

    coverage = dict(policy.coverage)
    rejected_target_count = sum(
        max(1, len(_assignment_paths(group))) for group in policy.rejected_groups
    )
    coverage.update(
        {
            "material_decision_prim_count": coverage["claimed_candidate_prim_count"],
            "missing_assignment_prim_count": coverage["unassigned_visible_prim_count"],
            "rejected_assignment_prim_count": rejected_target_count,
        }
    )
    expected_applied = {
        path
        for group in policy.material_assignments
        for path in _assignment_paths(group)
    }
    unbound_paths = sorted(expected_applied - set(application.applied_paths))
    if coverage["unassigned_visible_prim_count"]:
        unresolved.append(
            f"{coverage['unassigned_visible_prim_count']} visible material candidates "
            "remain unassigned."
        )
    if rejected_target_count:
        unresolved.append(
            f"Workflow policy rejected {rejected_target_count} material targets."
        )
    if unbound_paths:
        unresolved.append(
            f"usd-cli did not materialize {len(unbound_paths)} selected material targets."
        )
    unresolved = _deduplicated_strings(unresolved)

    supplied_status = str(supplied_assessment["status"])
    clean_final_review = final_reviewed and not unresolved
    status: Literal["pass", "conditional"] = (
        "pass"
        if clean_final_review and supplied_status in {"pass", "fixed"}
        else "conditional"
    )
    supplied_issues = supplied_assessment.get("issues_found", [])
    if not isinstance(supplied_issues, list) or not all(
        isinstance(item, dict) for item in supplied_issues
    ):
        supplied_issues = []
    supplied_fixes = supplied_assessment.get("issues_fixed", [])
    if not isinstance(supplied_fixes, list):
        supplied_fixes = []
    assessment = {
        "schema_version": "content-agents.visual-quality-assessment.v1",
        "status": supplied_status if status == "pass" else "unresolved_issues",
        # Preserve the views the coordinator actually reviewed. Never relabel newly
        # generated final renders as checked evidence.
        "checked_views": list(supplied_assessment["checked_views"]),
        "reference_images": [binding.path for binding in request.reference_images],
        "reference_files": [binding.path for binding in request.reference_files],
        "issues_found": supplied_issues,
        "issues_fixed": _deduplicated_strings(supplied_fixes),
        "unresolved_issues": unresolved,
        "assessment_notes": str(
            supplied_assessment.get("assessment_notes")
            or (
                "The supplied review exactly binds every final OVRTX render."
                if final_reviewed
                else "The supplied review covers pre-apply evidence; post-apply final "
                "OVRTX renders are generated but not yet visually reviewed."
            )
        ),
        "review_scope": "final" if final_reviewed else "pre_apply",
        "final_render_review": {
            "status": "reviewed" if final_reviewed else "not_evaluated",
            "required_views": [str(path) for path in final_images],
        },
        "checked_view_bindings": [
            binding.model_dump(mode="json")
            for binding in supplied_checked_views.values()
        ],
    }
    atomic_write_json(run_dir / "visual_quality_assessment.json", assessment)

    materialized_status = (
        "succeeded"
        if not unbound_paths and not policy.rejected_groups
        else "conditional"
    )
    assignments = {
        "schema_version": "content-agents.assignments.v1",
        "source_usd": request.source.path,
        "material_assignments": list(policy.material_assignments),
        "reviewed_no_override": list(policy.reviewed_no_override),
        "rejected_assignment_groups": list(policy.rejected_groups),
        "coverage": coverage,
        "materialized_usd": {
            "status": materialized_status,
            "requested_output_path": str(output),
            "output_path": str(output),
            "unbound_source_prim_paths": unbound_paths,
            "uncovered_assignment_groups": list(policy.rejected_groups),
            "unresolved_mappings": unbound_paths,
        },
        "pre_apply_visual_quality_assessment": decision["visual_quality_assessment"],
        "visual_quality_assessment": assessment,
    }
    atomic_write_json(run_dir / "assignments.json", assignments)
    operation_counts = _operation_counts(
        run_dir,
        [],
        list(policy.material_assignments),
        coverage=coverage,
        final_review=assessment,
        visual_quality=assessment,
    )
    atomic_write_json(run_dir / "api_operation_counts.json", operation_counts)
    restore_response_path = run_dir / "raw" / "material_restore_response.json"
    atomic_write_json(
        restore_response_path,
        {
            "scene_tool": "usd-cli",
            "status": materialized_status,
            "output_usd_path": str(output),
            "restored_source_prim_paths": list(application.applied_paths),
            "restored_edit_count": len(application.applied_paths),
            "unresolved_mappings": unbound_paths,
            "unbound_source_prim_paths": unbound_paths,
        },
    )
    validation = material_assignment_validation_evidence(
        asset=request.source.path,
        target_runtime="usd-cli/ovrtx",
        visual_materials_status="pass" if status == "pass" else "warning",
        evidence_artifacts=validation_artifacts,
        warnings=[] if status == "pass" else unresolved,
        unresolved_issues=unresolved,
    )
    atomic_write_json(run_dir / "validation_evidence.json", validation)
    summary = (
        "The workflow-selected material decision was applied through usd-cli, and "
        "the supplied VQA exactly binds every final OVRTX render.\n"
        if status == "pass"
        else "The workflow-selected material decision was applied through usd-cli. "
        "The final OVRTX render and segmentation artifacts are sealed, but a "
        "separate coordinator review is still required before acceptance.\n"
    )
    atomic_write_text(
        run_dir / "final_summary.md",
        "# Material coordinator result\n\n" + summary,
    )
    build_trace(run_dir)
    trace_paths = sorted(
        (path for path in (run_dir / "trace").rglob("*") if path.is_file()),
        key=str,
    )
    preparation = _load_preparation(run_dir)
    base_evidence = [
        run_dir / "coordinator_request.json",
        run_dir / "coordinator_preparation.json",
        run_dir / "assignments.json",
        run_dir / "api_operation_counts.json",
        run_dir / "visual_quality_assessment.json",
        run_dir / "validation_evidence.json",
        run_dir / "final_summary.md",
        restore_response_path,
        run_dir / "raw" / "material_application_receipt.json",
        run_dir / "raw" / "material_post_apply_review.json",
        run_dir / "raw" / "material_binding_audit.json",
        run_dir / "raw" / "material_applied_decision_patch.json",
        run_dir / "raw" / "material_finalization_policy.json",
        run_dir / "raw" / "material_run_packet.json",
        run_dir / "raw" / "ovrtx_render_probe.json",
        run_dir / "raw" / "visible_candidate_prims.json",
        run_dir / "raw" / "material_palette.json",
        run_dir / "raw" / "material_authoring_context.md",
        run_dir / "raw" / "material_assignment_seed.json",
        run_dir / "raw" / "visible_candidate_table.tsv",
        application.local_output,
        application.receipt_checkpoint_path,
        *application.receipt_paths,
        *[Path(binding.path) for binding in preparation.initial_render_bindings],
        Path(preparation.initial_render_records_binding.path),
        *final_artifact_paths,
        *trace_paths,
    ]
    clear_report = run_dir / "raw" / "appearance_clear_report.json"
    if clear_report.is_file():
        base_evidence.append(clear_report)
    evidence_paths = tuple(
        dict.fromkeys(_run_artifact(run_dir, path) for path in base_evidence)
    )
    return _PublishedMaterialResult(
        evidence_paths=evidence_paths,
        status=status,
        unresolved_issues=tuple(unresolved),
    )


def finalize_material_for_coordinator(
    run_dir: str | Path,
    *,
    decision_patch_path: str | Path,
    preparation_sha256: str,
) -> MaterialCoordinatorReviewRequired:
    """Apply and render, then stop for a real post-apply coordinator review."""

    root = Path(run_dir).expanduser().resolve()
    _reject_unsafe_run_links(root)
    _verify_parent_carried_manifest_digest(
        root / "coordinator_preparation.json",
        preparation_sha256,
        label="Material coordinator preparation",
    )
    if (root / "raw" / "material_application_receipt.json").exists():
        raise ValueError(
            "Material decision was already applied; submit the post-apply review "
            "instead of reapplying it"
        )
    preparation = _load_preparation(root)
    request = _load_request(root)
    _verify_prepared_evidence(root, preparation)
    _verify_request_inputs(request)
    decision_path = Path(decision_patch_path).expanduser().resolve()
    canonical = root / "raw" / "material_decision_patch.json"
    if decision_path != canonical or not decision_path.is_file():
        raise ValueError(f"Coordinator decision patch must use {canonical}")
    raw_decision = load_json(decision_path)
    if not isinstance(raw_decision, dict):
        raise ValueError("Material decision patch must be a JSON object")
    decision = dict(raw_decision)
    _validate_coordinator_decision_patch(
        decision,
        run_dir=root,
        preparation=preparation,
        request=request,
    )
    candidates = load_json(root / "raw" / "visible_candidate_prims.json")
    if not isinstance(candidates, dict):
        raise ValueError("Material candidate evidence is not a JSON object")
    manifest = load_material_manifest(
        Path(request.materials_yaml.path),
        library_override=Path(request.materials_usd.path),
    )
    try:
        policy = finalize_material_policy(
            decision,
            candidates=candidates,
            manifest=manifest,
            respect_existing_material_bindings=request.respect_existing_material_bindings,
        )
    except MaterialDecisionPolicyError as exc:
        atomic_write_json(
            root / "raw" / "rejected_material_assignments.json", exc.as_dict()
        )
        raise
    atomic_write_json(
        root / "raw" / "material_finalization_policy.json",
        {
            "schema_version": "content-agents.material-finalization-policy.v1",
            "coverage": policy.coverage,
            "rejected_groups": list(policy.rejected_groups),
        },
    )
    # Only the workflow-normalized canonical decisions reach the scene adapter.
    # Runtime aliases are evidence, never raw authoring targets.
    applied_decision = {
        **decision,
        "material_assignments": list(policy.material_assignments),
        "reviewed_no_override": list(policy.reviewed_no_override),
    }
    applied_decision_path = root / "raw" / "material_applied_decision_patch.json"
    atomic_write_json(applied_decision_path, applied_decision)
    application = _apply_material_decision(
        run_dir=root,
        request=request,
        preparation=preparation,
        decision=applied_decision,
    )
    return _write_application_receipt(
        run_dir=root,
        application=application,
    )


def review_material_for_coordinator(
    run_dir: str | Path,
    *,
    review_patch_path: str | Path,
    preparation_sha256: str,
    application_receipt_sha256: str,
) -> MaterialCoordinatorResult:
    """Publish only after a separate VQA binds the exact post-apply renders."""

    root = Path(run_dir).expanduser().resolve()
    _reject_unsafe_run_links(root)
    _verify_parent_carried_manifest_digest(
        root / "coordinator_preparation.json",
        preparation_sha256,
        label="Material coordinator preparation",
    )
    _verify_parent_carried_manifest_digest(
        root / "raw" / "material_application_receipt.json",
        application_receipt_sha256,
        label="Material application receipt",
    )
    preparation = _load_preparation(root)
    request = _load_request(root)
    _verify_prepared_evidence(
        root,
        preparation,
        verify_receipt_checkpoint=False,
    )
    _verify_request_inputs(request)
    receipt = _load_application_receipt(root)
    canonical = root / "raw" / "material_post_apply_review.json"
    review_path = Path(review_patch_path).expanduser().resolve()
    if review_path != canonical or not review_path.is_file():
        raise ValueError(f"Coordinator post-apply review must use {canonical}")
    raw_review = load_json(review_path)
    if not isinstance(raw_review, dict):
        raise ValueError("Material post-apply review must be a JSON object")
    review = dict(raw_review)
    checked_bindings = _validate_post_apply_review(
        review,
        run_dir=root,
        receipt=receipt,
    )
    decision = load_json(Path(receipt.decision_patch.path))
    if not isinstance(decision, dict):
        raise ValueError("Material applied decision receipt is invalid")
    application = _application_from_receipt(root, receipt)
    policy = _policy_from_receipt(root, receipt)
    published = _publish_result_artifacts(
        run_dir=root,
        request=request,
        decision=decision,
        review_assessment=review,
        policy=policy,
        application=application,
        supplied_checked_views=checked_bindings,
    )
    output = Path(request.output_usd_path)
    result = MaterialCoordinatorResult(
        status=published.status,
        output_usd_path=str(output),
        output_usd_sha256=file_sha256(output),
        request=_bound(root / "coordinator_request.json"),
        decision_patch=receipt.decision_patch,
        evidence=[_bound(path) for path in published.evidence_paths],
        unresolved_issues=list(published.unresolved_issues),
    )
    atomic_write_json(root / "coordinator_result.json", result)
    release = _release_material_session(
        root,
        request,
        preparation.session_id,
        receipt_checkpoint_sha256=receipt.receipt_checkpoint_binding.sha256,
    )
    if release.status != "released":
        raise RuntimeError("usd-cli session release failed; retry post-apply review")
    return result


def _config_from_request(request: MaterialCoordinatorRequest) -> MaterialAssignConfig:
    return MaterialAssignConfig(
        repo_root=Path(request.repository_root),
        usd_path=Path(request.source.path),
        reference_images=[Path(binding.path) for binding in request.reference_images],
        reference_files=[Path(binding.path) for binding in request.reference_files],
        materials_yaml=Path(request.materials_yaml.path),
        materials_usd=Path(request.materials_usd.path),
        output_dir=Path(request.run_dir),
        output_usd_path=Path(request.output_usd_path),
        optimize=request.optimize,
        optimizer_selection=OPTIMIZER_SELECTION_FIXED,
        root_prim_path=request.root_prim_path,
        material_candidate_space=request.material_candidate_space,
        skip_instances=request.skip_instances,
        skip_prototypes=request.skip_prototypes,
        skip_invisible=request.skip_invisible,
        flatten_prototypes=request.flatten_prototypes,
        enable_deinstance=request.enable_deinstance,
        enable_split=request.enable_split,
        enable_deduplicate=request.enable_deduplicate,
        respect_existing_material_bindings=request.respect_existing_material_bindings,
        preflight=True,
        scene_tool_timeout_seconds=request.scene_tool_timeout_seconds,
        material_restore_timeout_seconds=request.material_restore_timeout_seconds,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or finalize Material work for a parent coordinator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--usd", type=Path, required=True)
    prepare.add_argument("--materials-yaml", type=Path, required=True)
    prepare.add_argument("--materials-usd", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--output-usd", type=Path, required=True)
    prepare.add_argument("--scene-tool-timeout", type=float, default=300.0)
    prepare.add_argument("--reference-image", type=Path, action="append", default=[])
    prepare.add_argument("--reference-file", type=Path, action="append", default=[])
    prepare.add_argument("--repo-root", type=Path)
    prepare.add_argument(
        "--optimize", action=argparse.BooleanOptionalAction, default=False
    )
    prepare.add_argument("--root-prim-path")
    prepare.add_argument("--material-candidate-space", default="source")
    prepare.add_argument(
        "--skip-instances", action=argparse.BooleanOptionalAction, default=True
    )
    prepare.add_argument(
        "--skip-prototypes", action=argparse.BooleanOptionalAction, default=False
    )
    prepare.add_argument(
        "--skip-invisible", action=argparse.BooleanOptionalAction, default=False
    )
    prepare.add_argument("--flatten-prototypes", action=argparse.BooleanOptionalAction)
    prepare.add_argument("--enable-deinstance", action=argparse.BooleanOptionalAction)
    prepare.add_argument("--enable-split", action=argparse.BooleanOptionalAction)
    prepare.add_argument(
        "--enable-deduplicate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    prepare.add_argument("--respect-existing-material-bindings", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--decision-patch", type=Path, required=True)
    finalize.add_argument("--preparation-sha256", required=True)
    review = subparsers.add_parser("review")
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--review-patch", type=Path, required=True)
    review.add_argument("--preparation-sha256", required=True)
    review.add_argument("--application-receipt-sha256", required=True)
    release = subparsers.add_parser("release")
    release.add_argument("--run-dir", type=Path, required=True)
    release.add_argument("--preparation-sha256", required=True)
    release.add_argument("--application-receipt-sha256")
    return parser


def _cli_output_payload(output: BaseModel) -> dict[str, Any]:
    """Add parent-carried phase seals to stdout, never to the sealed manifests."""

    payload = output.model_dump(mode="json")
    if isinstance(output, MaterialCoordinatorPreparation):
        payload["preparation_sha256"] = file_sha256(
            Path(output.request_path).parent / "coordinator_preparation.json"
        )
    elif isinstance(output, MaterialCoordinatorReviewRequired):
        payload["application_receipt_sha256"] = file_sha256(
            Path(output.request.path).parent
            / "raw"
            / "material_application_receipt.json"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            repo_root = (
                args.repo_root or find_repo_root(args.usd.expanduser().resolve().parent)
            ).resolve()
            output = prepare_material_for_coordinator(
                MaterialAssignConfig(
                    repo_root=repo_root,
                    usd_path=args.usd.expanduser().resolve(),
                    reference_images=[path.resolve() for path in args.reference_image],
                    reference_files=[path.resolve() for path in args.reference_file],
                    materials_yaml=args.materials_yaml.expanduser().resolve(),
                    materials_usd=args.materials_usd.expanduser().resolve(),
                    output_dir=args.output_dir.expanduser().resolve(),
                    output_usd_path=args.output_usd.expanduser().resolve(),
                    optimize=args.optimize,
                    optimizer_selection=OPTIMIZER_SELECTION_FIXED,
                    root_prim_path=args.root_prim_path,
                    material_candidate_space=args.material_candidate_space,
                    skip_instances=args.skip_instances,
                    skip_prototypes=args.skip_prototypes,
                    skip_invisible=args.skip_invisible,
                    flatten_prototypes=args.flatten_prototypes,
                    enable_deinstance=args.enable_deinstance,
                    enable_split=args.enable_split,
                    enable_deduplicate=args.enable_deduplicate,
                    respect_existing_material_bindings=(
                        args.respect_existing_material_bindings
                    ),
                    preflight=True,
                    scene_tool_timeout_seconds=args.scene_tool_timeout,
                )
            )
        elif args.command == "finalize":
            output = finalize_material_for_coordinator(
                args.run_dir,
                decision_patch_path=args.decision_patch,
                preparation_sha256=args.preparation_sha256,
            )
        elif args.command == "review":
            output = review_material_for_coordinator(
                args.run_dir,
                review_patch_path=args.review_patch,
                preparation_sha256=args.preparation_sha256,
                application_receipt_sha256=args.application_receipt_sha256,
            )
        else:
            output = release_material_for_coordinator(
                args.run_dir,
                preparation_sha256=args.preparation_sha256,
                application_receipt_sha256=args.application_receipt_sha256,
            )
    except Exception as exc:  # noqa: BLE001 - stable CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_cli_output_payload(output), indent=2, sort_keys=True))
    if isinstance(output, MaterialCoordinatorRelease) and output.status == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
