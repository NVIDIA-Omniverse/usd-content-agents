# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Material-task survey and usd-cli execution for Workflow 2 work items."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from content_agent_workflows.common.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
    read_contained_artifact,
    resolve_artifact_path,
)
from content_agent_workflows.common.usd_cli import validate_ovrtx_probe
from content_agent_workflows.common.usd_cli_session import WorkflowUsdCliSession
from content_agent_workflows.material_assignment.manifest import (
    ResolvedMaterialManifest,
    load_material_manifest,
)

from .contracts import (
    AgentPlanPointer,
    AssetTaskResult,
    DecisionLedgerEntry,
)
from .material_appearance import (
    build_material_appearance_index,
    rank_display_color_candidates,
    render_display_color_targets,
)
from .runtime import (
    AssetTaskRuntimeError,
    ProcessingPaths,
    begin_work_item,
    commit_work_item,
    fail_work_item,
    get_work_item,
)

MATERIAL_SURVEY_SCHEMA_VERSION = "content-agent-workflows.material-survey.v1"
MATERIAL_DECISION_SCHEMA_VERSION = "content-agent-workflows.material-decision.v1"
MATERIAL_BATCH_PLAN_SCHEMA_VERSION = "content-agent-workflows.material-batch-plan.v1"
MATERIAL_TASK_REQUEST_SCHEMA_VERSION = (
    "content-agent-workflows.material-task-request.v2"
)
LEGACY_MATERIAL_TASK_REQUEST_SCHEMA_VERSION = (
    "content-agent-workflows.material-task-request.v1"
)
MATERIAL_VALIDATION_SCHEMA_VERSION = (
    "content-agent-workflows.material-task-validation.v1"
)
MAX_MATERIAL_STAGE_BYTES = 512 * 1024 * 1024
# OpenUSD's TfSafeOutputFile appends two PID/token-bearing temporary suffixes
# before replacing the requested layer.  The current Windows build still hits
# the legacy MAX_PATH boundary for that intermediate name even when Python can
# address the requested path itself.  Reserve a conservative suffix budget and
# reject an overlong run root before starting a daemon-backed work item.
WINDOWS_OPENUSD_REPLACE_SUFFIX_CODE_UNITS = 72
WINDOWS_LEGACY_MAX_PATH_CODE_UNITS = 260
APPEARANCE_EVIDENCE_POLICY_SCHEMA_VERSION = (
    "content-agent-workflows.appearance-evidence-policy.v1"
)
AppearanceEvidenceSource = Literal["material_binding", "display_color"]
MaterialSceneBackend = Literal["usd-cli"]


class AppearanceEvidenceScope(BaseModel):
    """Scoped permission to expose source-authored appearance evidence."""

    model_config = ConfigDict(extra="forbid")

    root: str = Field(min_length=1)
    sources: list[AppearanceEvidenceSource] = Field(default_factory=list)
    mode: Literal["hint_only", "seed_coverage"] = "hint_only"
    reason: str | None = None

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: str) -> str:
        normalized = value.rstrip("/") or "/"
        if not normalized.startswith("/"):
            raise ValueError("appearance-evidence scope roots must be absolute paths")
        return normalized

    @field_validator("sources")
    @classmethod
    def validate_sources(
        cls, value: list[AppearanceEvidenceSource]
    ) -> list[AppearanceEvidenceSource]:
        if not value:
            raise ValueError("appearance-evidence scopes require at least one source")
        if len(value) != len(set(value)):
            raise ValueError("appearance-evidence scope sources must be unique")
        return value

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        return value.strip() or None


class AppearanceEvidencePolicy(BaseModel):
    """Controls whether old CAD appearance is exposed as task evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[APPEARANCE_EVIDENCE_POLICY_SCHEMA_VERSION] = (
        APPEARANCE_EVIDENCE_POLICY_SCHEMA_VERSION
    )
    default: Literal["ignore", "expose_all"] = "ignore"
    global_sources: list[AppearanceEvidenceSource] = Field(default_factory=list)
    scopes: list[AppearanceEvidenceScope] = Field(default_factory=list)

    @field_validator("global_sources")
    @classmethod
    def validate_global_sources(
        cls, value: list[AppearanceEvidenceSource]
    ) -> list[AppearanceEvidenceSource]:
        if len(value) != len(set(value)):
            raise ValueError("appearance-evidence global_sources must be unique")
        return value


class MaterialTaskRequest(BaseModel):
    """Scene-level material intent shared by every work item in one task."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        MATERIAL_TASK_REQUEST_SCHEMA_VERSION,
        LEGACY_MATERIAL_TASK_REQUEST_SCHEMA_VERSION,
    ] = MATERIAL_TASK_REQUEST_SCHEMA_VERSION
    domain: Literal["material"] = "material"
    reference_images: list[str] = Field(default_factory=list)
    reference_files: list[str] = Field(default_factory=list)
    material_library_yaml: str | None = None
    material_library_path: str | None = None
    candidate_space: Literal["source"] = "source"
    respect_existing_material_bindings: bool = False
    appearance_evidence_policy: AppearanceEvidencePolicy = Field(
        default_factory=AppearanceEvidencePolicy
    )
    processing_policy: dict[str, Any] = Field(default_factory=dict)
    additional_instructions: str | None = None

    @field_validator("additional_instructions", mode="before")
    @classmethod
    def normalize_additional_instructions(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None


class MaterialCandidateEvidence(BaseModel):
    """One source-space surface candidate and its existing appearance evidence."""

    model_config = ConfigDict(extra="forbid")

    prim_path: str
    prim_type: str
    mesh_path: str
    face_count: int = Field(ge=0)
    bound_material_path: str | None = None
    bound_material_name: str | None = None
    diffuse_color: list[float] | None = None
    display_color: list[float] | None = None
    display_color_interpolation: str | None = None
    display_color_value_count: int = Field(default=0, ge=0)
    metallic: float | None = None
    roughness: float | None = None
    opacity: float | None = None


class MaterialSurvey(BaseModel):
    """Deterministic material candidate survey for one work item."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MATERIAL_SURVEY_SCHEMA_VERSION] = (
        MATERIAL_SURVEY_SCHEMA_VERSION
    )
    work_item_id: str
    asset_label: str
    source_usd: str
    original_root_path: str
    candidates: list[MaterialCandidateEvidence]
    visibility_policy: Literal["visible_only", "all"] = "all"
    skipped_invisible_mesh_count: int = Field(default=0, ge=0)
    appearance_evidence_policy: AppearanceEvidencePolicy = Field(
        default_factory=AppearanceEvidencePolicy
    )
    evidence_paths: list[str] = Field(default_factory=list)


class MaterialAssignmentDecision(BaseModel):
    """Agent-authored library assignment for one surveyed source candidate."""

    model_config = ConfigDict(extra="forbid")

    target_prim_path: str
    coverage_mode: Literal["explicit", "descendants"] = "explicit"
    covered_candidate_paths: list[str] = Field(default_factory=list)
    material_name: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    informed_by_candidate: str | None = None

    @model_validator(mode="after")
    def validate_coverage(self) -> MaterialAssignmentDecision:
        if self.coverage_mode == "explicit" and not self.covered_candidate_paths:
            raise ValueError("explicit coverage requires covered_candidate_paths")
        if self.coverage_mode == "descendants" and self.covered_candidate_paths:
            raise ValueError(
                "descendants coverage computes candidates and must not list them"
            )
        return self


class MaterialDecisionPatch(BaseModel):
    """Complete agent decision for one material work item."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MATERIAL_DECISION_SCHEMA_VERSION] = (
        MATERIAL_DECISION_SCHEMA_VERSION
    )
    work_item_id: str
    source_usd: str
    material_library_yaml: str
    material_library_path: str
    task_request_digest: str | None = None
    assignments: list[MaterialAssignmentDecision]
    evidence_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    informed_by_results: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_targets(self) -> MaterialDecisionPatch:
        targets = [assignment.target_prim_path for assignment in self.assignments]
        if len(targets) != len(set(targets)):
            raise ValueError("material assignment targets must be unique")
        explicitly_covered = [
            candidate
            for assignment in self.assignments
            for candidate in assignment.covered_candidate_paths
        ]
        if len(explicitly_covered) != len(set(explicitly_covered)):
            raise ValueError("each surveyed candidate may be covered only once")
        return self


class MaterialBatchItem(BaseModel):
    """One agent-ordered material execution request."""

    model_config = ConfigDict(extra="forbid")

    work_item_id: str
    decision_path: str
    render: bool = False


class MaterialBatchPlan(BaseModel):
    """Agent-selected execution order over already-authored decisions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MATERIAL_BATCH_PLAN_SCHEMA_VERSION] = (
        MATERIAL_BATCH_PLAN_SCHEMA_VERSION
    )
    items: list[MaterialBatchItem]
    stop_on_error: bool = False

    @model_validator(mode="after")
    def validate_unique_items(self) -> MaterialBatchPlan:
        identities = [item.work_item_id for item in self.items]
        if len(identities) != len(set(identities)):
            raise ValueError("material batch work_item_id values must be unique")
        return self


@dataclass(frozen=True, slots=True)
class _MaterialSceneExecution:
    """Backend-neutral facts returned by one low-level scene executor."""

    preview_layer: Path
    output_stage: Any
    assignments_response: dict[str, object]
    apply_response: dict[str, object]
    command_records: list[dict[str, object]]
    command_artifact_path: Path
    render_paths: list[str]
    render_validation_path: Path | None
    render_validation_errors: list[str]
    backend_artifact_paths: tuple[Path, ...] = ()


def _open_captured_material_stage(data: bytes, *, suffix: str) -> Any:
    """Open one parent-private USD snapshot from already captured bytes."""

    from pxr import Usd

    if suffix.lower() not in {".usd", ".usda", ".usdc"}:
        suffix = ".usda"
    with tempfile.TemporaryDirectory(prefix="material-task-usd-audit-") as value:
        root = Path(value).resolve(strict=True)
        root.chmod(0o700)
        snapshot = root / f"captured{suffix}"
        atomic_write_bytes(snapshot, data, within=root)
        stage = Usd.Stage.Open(str(snapshot))
    return stage


def _material_scene_backend(request: MaterialTaskRequest) -> MaterialSceneBackend:
    """Require the canonical low-level scene backend."""

    configured = request.processing_policy.get("scene_backend")
    if configured not in (None, "usd-cli"):
        raise AssetTaskRuntimeError(
            "Material requests using the retired scene backend cannot be "
            "replayed; regenerate the request with scene_backend='usd-cli'."
        )
    session_scope = request.processing_policy.get("scene_session_scope")
    if session_scope not in (None, "per_asset"):
        raise AssetTaskRuntimeError(
            "usd-cli material tasks require "
            "processing_policy.scene_session_scope='per_asset'"
        )
    return "usd-cli"


def _json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return str(value)


def _material_evidence(prim: Any) -> dict[str, object]:
    from pxr import Usd, UsdShade

    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return {}
    evidence: dict[str, object] = {
        "bound_material_path": str(material.GetPath()),
        "bound_material_name": material.GetPrim().GetName(),
    }
    shader_prim = None
    for descendant in Usd.PrimRange(material.GetPrim()):
        if descendant.IsA(UsdShade.Shader):
            shader_prim = descendant
            break
    if shader_prim is None:
        return evidence
    shader = UsdShade.Shader(shader_prim)
    for field, names in {
        "diffuse_color": ("diffuseColor", "base_color", "baseColor"),
        "metallic": ("metallic", "metalness"),
        "roughness": ("roughness",),
        "opacity": ("opacity",),
    }.items():
        for name in names:
            shader_input = shader.GetInput(name)
            if not shader_input:
                continue
            value = shader_input.Get()
            if value is not None:
                evidence[field] = _json_value(value)
                break
    return evidence


def _display_color_evidence(prim: Any) -> dict[str, object]:
    """Return compact authored display-color evidence for a source Gprim."""

    from pxr import UsdGeom

    if not prim.IsA(UsdGeom.Gprim):
        return {}
    primvar = UsdGeom.Gprim(prim).GetDisplayColorPrimvar()
    if not primvar or not primvar.HasAuthoredValue():
        return {}
    values = primvar.ComputeFlattened()
    if not values:
        return {}
    value_count = len(values)
    return {
        "display_color": [
            sum(float(value[channel]) for value in values) / value_count
            for channel in range(3)
        ],
        "display_color_interpolation": str(primvar.GetInterpolation()),
        "display_color_value_count": value_count,
    }


def _path_is_within_scope(path: str, scope_root: str) -> bool:
    normalized_path = path.rstrip("/") or "/"
    normalized_scope = scope_root.rstrip("/") or "/"
    if normalized_scope == "/":
        return normalized_path.startswith("/")
    return normalized_path == normalized_scope or normalized_path.startswith(
        normalized_scope + "/"
    )


def _coerce_appearance_evidence_policy(
    value: AppearanceEvidencePolicy | dict[str, object] | None,
) -> AppearanceEvidencePolicy:
    if isinstance(value, AppearanceEvidencePolicy):
        return value
    if value is None:
        return AppearanceEvidencePolicy()
    return AppearanceEvidencePolicy.model_validate(value)


def _appearance_sources_for_path(
    policy: AppearanceEvidencePolicy,
    path: str,
) -> set[AppearanceEvidenceSource]:
    sources: set[AppearanceEvidenceSource] = set(policy.global_sources)
    if policy.default == "expose_all":
        sources.update(("material_binding", "display_color"))
    for scope in policy.scopes:
        if _path_is_within_scope(path, scope.root):
            sources.update(scope.sources)
    return sources


def _appearance_policy_allows_scope(
    policy: AppearanceEvidencePolicy,
    source: AppearanceEvidenceSource,
    scope_path: str,
) -> bool:
    if policy.default == "expose_all" or source in policy.global_sources:
        return True
    return any(
        source in scope.sources and _path_is_within_scope(scope_path, scope.root)
        for scope in policy.scopes
    )


def _effective_appearance_evidence_policy(
    request: MaterialTaskRequest,
) -> AppearanceEvidencePolicy:
    """Return the explicit policy for a material task request.

    `respect_existing_material_bindings` controls preservation semantics in the
    task workflow. It does not implicitly expose authored CAD colors as hint
    evidence; use `appearance_evidence_policy` for that.
    """

    return request.appearance_evidence_policy


def _candidate_evidence(
    *,
    policy: AppearanceEvidencePolicy,
    material_prim: Any,
    display_prim: Any,
) -> dict[str, object]:
    evidence: dict[str, object] = {}
    material_sources = _appearance_sources_for_path(
        policy, str(material_prim.GetPath())
    )
    display_sources = _appearance_sources_for_path(policy, str(display_prim.GetPath()))
    if "display_color" in display_sources:
        evidence.update(_display_color_evidence(display_prim))
    if "material_binding" in material_sources:
        evidence.update(_material_evidence(material_prim))
    return evidence


def survey_usd_material_candidates(
    *,
    work_item_id: str,
    asset_label: str,
    usd_path: str | Path,
    original_root_path: str,
    evidence_paths: list[str] | None = None,
    skip_invisible: bool = True,
    appearance_evidence_policy: AppearanceEvidencePolicy
    | dict[str, object]
    | None = None,
) -> MaterialSurvey:
    """Survey source Mesh/GeomSubset surfaces without changing the stage."""

    from pxr import Usd, UsdGeom

    policy = _coerce_appearance_evidence_policy(appearance_evidence_policy)
    source = Path(usd_path).expanduser().resolve()
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise AssetTaskRuntimeError(f"Could not open working USD: {source}")
    candidates: list[MaterialCandidateEvidence] = []
    skipped_invisible_mesh_count = 0
    for prim in stage.Traverse():
        if prim.IsInstanceProxy():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if (
            skip_invisible
            and UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
        ):
            skipped_invisible_mesh_count += 1
            continue
        mesh = UsdGeom.Mesh(prim)
        face_count = len(mesh.GetFaceVertexCountsAttr().Get() or [])
        subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim))
        covered_faces: set[int] = set()
        material_subsets = []
        for subset in subsets:
            subset_prim = subset.GetPrim()
            if subset_prim.IsInstanceProxy():
                continue
            indices = subset.GetIndicesAttr().Get() or []
            if indices:
                material_subsets.append((subset_prim, len(indices)))
                covered_faces.update(int(index) for index in indices)
        for subset_prim, subset_face_count in material_subsets:
            candidates.append(
                MaterialCandidateEvidence(
                    prim_path=str(subset_prim.GetPath()),
                    prim_type=subset_prim.GetTypeName(),
                    mesh_path=str(prim.GetPath()),
                    face_count=subset_face_count,
                    **_candidate_evidence(
                        policy=policy,
                        material_prim=subset_prim,
                        display_prim=prim,
                    ),
                )
            )
        if not material_subsets or len(covered_faces) < face_count:
            candidates.append(
                MaterialCandidateEvidence(
                    prim_path=str(prim.GetPath()),
                    prim_type=prim.GetTypeName(),
                    mesh_path=str(prim.GetPath()),
                    face_count=max(face_count - len(covered_faces), 0),
                    **_candidate_evidence(
                        policy=policy,
                        material_prim=prim,
                        display_prim=prim,
                    ),
                )
            )
    if not candidates:
        qualifier = "visible " if skip_invisible else ""
        message = f"No {qualifier}material candidates found in {source}"
        if skip_invisible and skipped_invisible_mesh_count:
            message += (
                "; material-processing decomposition must exclude assets that "
                "contain only invisible meshes"
            )
        raise AssetTaskRuntimeError(message)
    return MaterialSurvey(
        work_item_id=work_item_id,
        asset_label=asset_label,
        source_usd=str(source),
        original_root_path=original_root_path,
        candidates=candidates,
        visibility_policy="visible_only" if skip_invisible else "all",
        skipped_invisible_mesh_count=skipped_invisible_mesh_count,
        appearance_evidence_policy=policy,
        evidence_paths=evidence_paths or [],
    )


def _safe_path_component(value: str, label: str) -> str:
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise AssetTaskRuntimeError(f"Unsafe {label} path component: {value!r}")
    return value


def _item_dir(processing_dir: Path, work_item_id: str) -> Path:
    item, _state = get_work_item(processing_dir, work_item_id)
    return (
        processing_dir
        / "assets"
        / _safe_path_component(item.manifest_id, "manifest_id")
        / _safe_path_component(item.asset_id, "asset_id")
        / "tasks"
        / _safe_path_component(item.task_id, "task_id")
    )


def load_material_task_request(
    processing_dir: str | Path,
    task_id: str = "material",
) -> tuple[MaterialTaskRequest, Path, str]:
    """Load and identify the frozen scene-level request for a material task."""

    from .runtime import _load_run

    root = Path(processing_dir).expanduser().resolve()
    _paths, _inventory, state, task_catalog, _manifests = _load_run(root)
    task = next(
        (candidate for candidate in task_catalog.tasks if candidate.task_id == task_id),
        None,
    )
    if task is None or task.domain != "material":
        raise AssetTaskRuntimeError(f"Unknown material task: {task_id}")
    task_catalog_path = Path(state.task_catalog_path).expanduser().resolve()
    request_path = resolve_artifact_path(
        task.request_path, base_dir=task_catalog_path.parent
    )
    try:
        request = MaterialTaskRequest.model_validate(load_json(request_path))
    except (OSError, ValueError, ValidationError) as exc:
        raise AssetTaskRuntimeError(
            f"Invalid material task request at {request_path}: {exc}"
        ) from exc
    request_digest = file_sha256(request_path)
    expected_digest = state.task_request_digests.get(task_id)
    if expected_digest and request_digest != expected_digest:
        raise AssetTaskRuntimeError(
            f"Material task request changed after preparation: {request_path}"
        )
    return request, request_path, request_digest


def survey_work_item(
    processing_dir: str | Path,
    work_item_id: str,
    *,
    evidence_paths: list[str] | None = None,
) -> tuple[MaterialSurvey, Path]:
    """Survey and persist one material work item."""

    root = Path(processing_dir).expanduser().resolve()
    item, _state = get_work_item(root, work_item_id)
    if not item.working_usd_path:
        raise AssetTaskRuntimeError(f"Work item has no working USD: {work_item_id}")
    request, _request_path, _request_digest = load_material_task_request(
        root, item.task_id
    )
    survey = survey_usd_material_candidates(
        work_item_id=work_item_id,
        asset_label=item.asset_label or Path(item.working_usd_path).stem,
        usd_path=item.working_usd_path,
        original_root_path=item.original_root_path,
        evidence_paths=evidence_paths,
        appearance_evidence_policy=_effective_appearance_evidence_policy(request),
    )
    output_path = _item_dir(root, work_item_id) / "material_survey.json"
    atomic_write_json(output_path, survey)
    return survey, output_path


def survey_material_inventory(
    processing_dir: str | Path,
    *,
    task_id: str = "material",
    render_index_path: str | Path | None = None,
) -> dict[str, object]:
    """Persist candidate surveys for every work item in one material task."""

    from .runtime import _load_run

    root = Path(processing_dir).expanduser().resolve()
    _paths, inventory, _state, _tasks, _manifests = _load_run(root)
    task_request, task_request_path, task_request_digest = load_material_task_request(
        root, task_id
    )
    render_by_asset: dict[str, str] = {}
    if render_index_path is not None:
        render_index_file = Path(render_index_path).expanduser().resolve()
        render_index = load_json(render_index_file)
        records = render_index.get("renders", render_index.get("records", []))
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                asset_id = record.get("asset_id") or record.get("name")
                image_path = record.get("image_path")
                if isinstance(asset_id, str) and isinstance(image_path, str):
                    imported_image = render_index_file.parent / f"{asset_id}.png"
                    render_by_asset[asset_id] = str(
                        imported_image.resolve()
                        if imported_image.is_file()
                        else resolve_artifact_path(
                            image_path,
                            base_dir=render_index_file.parent,
                        )
                    )

    entries: list[dict[str, object]] = []
    for item in inventory.work_items:
        if item.task_id != task_id:
            continue
        evidence_key = item.asset_label or (
            Path(item.working_usd_path).stem if item.working_usd_path else item.asset_id
        )
        evidence = (
            [render_by_asset[evidence_key]] if evidence_key in render_by_asset else []
        )
        survey, path = survey_work_item(
            root, item.work_item_id, evidence_paths=evidence
        )
        entries.append(
            {
                "work_item_id": item.work_item_id,
                "asset_id": item.asset_id,
                "asset_label": evidence_key,
                "survey_path": str(path),
                "candidate_count": len(survey.candidates),
                "evidence_paths": survey.evidence_paths,
            }
        )
    index_path = root / "shared_evidence" / "material_surveys_index.json"
    atomic_write_json(
        index_path,
        {
            "schema_version": "content-agent-workflows.material-surveys-index.v1",
            "task_id": task_id,
            "task_request": {
                "path": str(task_request_path),
                "sha256": task_request_digest,
                "additional_instructions": task_request.additional_instructions,
                "appearance_evidence_policy": _effective_appearance_evidence_policy(
                    task_request
                ).model_dump(mode="json"),
            },
            "entries": entries,
        },
    )
    return {
        "survey_count": len(entries),
        "candidate_count": sum(int(entry["candidate_count"]) for entry in entries),
        "task_request_path": str(task_request_path),
        "task_request_digest": task_request_digest,
        "index_path": str(index_path),
    }


def match_work_item_display_colors(
    processing_dir: str | Path,
    work_item_id: str,
    *,
    scope_paths: list[str],
    top_k: int = 5,
    appearance_cache_dir: str | Path | None = None,
    swatch_template_path: str | Path | None = None,
) -> dict[str, object]:
    """Render and rank prompt-scoped library candidates for one work item."""

    root = Path(processing_dir).expanduser().resolve()
    item, _state = get_work_item(root, work_item_id)
    normalized_root = item.original_root_path.rstrip("/") or "/"
    normalized_scopes = [scope.rstrip("/") or "/" for scope in scope_paths]
    invalid_scopes = [
        scope
        for scope in normalized_scopes
        if normalized_root != "/"
        and scope != normalized_root
        and not scope.startswith(normalized_root + "/")
    ]
    if invalid_scopes:
        raise AssetTaskRuntimeError(
            "Display-color scopes must be within the work-item root; "
            f"root={normalized_root}, invalid={invalid_scopes}"
        )
    request, request_path, request_digest = load_material_task_request(
        root, item.task_id
    )
    appearance_policy = _effective_appearance_evidence_policy(request)
    unauthorized_scopes = [
        scope
        for scope in normalized_scopes
        if not _appearance_policy_allows_scope(
            appearance_policy, "display_color", scope
        )
    ]
    if unauthorized_scopes:
        raise AssetTaskRuntimeError(
            "Display-color matching requires appearance_evidence_policy scopes "
            f"that include display_color; unauthorized={unauthorized_scopes}"
        )
    survey_path = _item_dir(root, work_item_id) / "material_survey.json"
    if survey_path.is_file():
        survey = MaterialSurvey.model_validate(load_json(survey_path))
    else:
        survey, survey_path = survey_work_item(root, work_item_id)
    if not request.material_library_yaml or not request.material_library_path:
        raise AssetTaskRuntimeError(
            "Display-color matching requires material_library_yaml and "
            "material_library_path in the task request"
        )
    yaml_path = resolve_artifact_path(
        request.material_library_yaml, base_dir=request_path.parent
    )
    library_path = resolve_artifact_path(
        request.material_library_path, base_dir=request_path.parent
    )
    if swatch_template_path is None:
        try:
            import material_agent
        except ImportError as exc:  # pragma: no cover - package dependency
            raise AssetTaskRuntimeError(
                "Cannot locate the material-agent swatch template"
            ) from exc
        template_path = (
            Path(material_agent.__file__).resolve().parent.parent
            / "data"
            / "templates"
            / "thumbnail_template.usd"
        )
    else:
        template_path = Path(swatch_template_path).expanduser().resolve()
    cache_dir = (
        Path(appearance_cache_dir).expanduser().resolve()
        if appearance_cache_dir is not None
        else root / "shared_evidence" / "material_appearance"
    )
    scoped_candidates = [
        candidate
        for candidate in survey.candidates
        if any(
            scope == "/"
            or candidate.prim_path == scope
            or candidate.prim_path.startswith(scope + "/")
            for scope in normalized_scopes
        )
    ]
    if not scoped_candidates:
        raise AssetTaskRuntimeError(
            "Display-color scopes contain no surveyed material candidates"
        )
    scoped_colors = [
        candidate.display_color
        for candidate in scoped_candidates
        if candidate.display_color is not None
    ]
    if not scoped_colors:
        raise AssetTaskRuntimeError(
            "Display-color scopes contain no candidates with authored display color"
        )
    try:
        appearance_index, appearance_index_path = build_material_appearance_index(
            material_library_yaml=yaml_path,
            material_library_path=library_path,
            swatch_template_path=template_path,
            cache_dir=cache_dir,
        )
        target_appearances = render_display_color_targets(
            colors=scoped_colors,
            swatch_template_path=template_path,
            output_dir=_item_dir(root, work_item_id) / "display_color_target_swatches",
        )
        matches = rank_display_color_candidates(
            work_item_id=work_item_id,
            task_request_path=request_path,
            task_request_digest=request_digest,
            survey_path=survey_path,
            survey=survey.model_dump(mode="json"),
            appearance_index_path=appearance_index_path,
            appearance_index=appearance_index,
            target_appearances=target_appearances,
            scope_paths=normalized_scopes,
            top_k=top_k,
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        raise AssetTaskRuntimeError(
            f"Display-color material matching failed for {work_item_id}: {exc}"
        ) from exc
    output_path = atomic_write_json(
        _item_dir(root, work_item_id) / "display_color_matches.json", matches
    )
    return {
        "work_item_id": work_item_id,
        "task_request_digest": request_digest,
        "scope_paths": normalized_scopes,
        "matched_candidate_count": len(matches.matches),
        "candidate_without_display_color_count": len(
            matches.candidates_without_display_color
        ),
        "material_count": len(appearance_index.materials),
        "appearance_index_path": str(appearance_index_path),
        "matches_path": str(output_path),
    }


def _resolve_material_library(
    yaml_path: Path,
    expected_library_path: Path,
) -> ResolvedMaterialManifest:
    try:
        manifest = load_material_manifest(yaml_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AssetTaskRuntimeError(
            f"Invalid workflow material manifest {yaml_path}: {exc}"
        ) from exc
    if manifest.library_path != expected_library_path:
        raise AssetTaskRuntimeError(
            "Decision material_library_path does not match the workflow "
            "material manifest"
        )
    return manifest


def _validate_decision(
    decision: MaterialDecisionPatch,
    survey: MaterialSurvey,
    *,
    task_request_digest: str | None = None,
    task_request: MaterialTaskRequest | None = None,
    task_request_path: Path | None = None,
) -> ResolvedMaterialManifest:
    if decision.work_item_id != survey.work_item_id:
        raise AssetTaskRuntimeError("Material decision and survey identities differ")
    if (
        Path(decision.source_usd).expanduser().resolve()
        != Path(survey.source_usd).expanduser().resolve()
    ):
        raise AssetTaskRuntimeError("Material decision source_usd differs from survey")
    if task_request_digest and decision.task_request_digest != task_request_digest:
        raise AssetTaskRuntimeError(
            "Material decision does not cite the frozen task request digest"
        )
    expected = {candidate.prim_path for candidate in survey.candidates}
    coverage_by_target: dict[str, set[str]] = {}
    for assignment in decision.assignments:
        if assignment.coverage_mode == "descendants":
            target_prefix = assignment.target_prim_path.rstrip("/") + "/"
            covered = {
                candidate
                for candidate in expected
                if candidate == assignment.target_prim_path
                or candidate.startswith(target_prefix)
            }
        else:
            covered = set(assignment.covered_candidate_paths)
        if not covered:
            raise AssetTaskRuntimeError(
                f"Material assignment covers no candidates: {assignment.target_prim_path}"
            )
        coverage_by_target[assignment.target_prim_path] = covered
    coverage_counts: dict[str, int] = {}
    for covered in coverage_by_target.values():
        for candidate in covered:
            coverage_counts[candidate] = coverage_counts.get(candidate, 0) + 1
    multiply_covered = sorted(
        candidate for candidate, count in coverage_counts.items() if count > 1
    )
    if multiply_covered:
        raise AssetTaskRuntimeError(
            f"Material candidates are covered more than once: {multiply_covered}"
        )
    actual = set(coverage_counts)
    if expected != actual:
        raise AssetTaskRuntimeError(
            "Material decisions do not exactly cover surveyed candidates; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(survey.source_usd)
    if stage is None:
        raise AssetTaskRuntimeError(
            f"Could not reopen surveyed USD: {survey.source_usd}"
        )
    for assignment in decision.assignments:
        target = Sdf.Path(assignment.target_prim_path)
        if not target.IsAbsolutePath() or not stage.GetPrimAtPath(target):
            raise AssetTaskRuntimeError(
                f"Material assignment target does not exist: {target}"
            )
        for candidate_reference in coverage_by_target[assignment.target_prim_path]:
            candidate = Sdf.Path(candidate_reference)
            if candidate != target and not candidate.HasPrefix(target):
                raise AssetTaskRuntimeError(
                    f"Assignment target {target} is not an ancestor of {candidate}"
                )
    yaml_path = Path(decision.material_library_yaml).expanduser().resolve()
    library_path = Path(decision.material_library_path).expanduser().resolve()
    if task_request is not None and task_request_path is not None:
        if task_request.material_library_yaml and task_request.material_library_path:
            expected_yaml_path = resolve_artifact_path(
                task_request.material_library_yaml,
                base_dir=task_request_path.parent,
            )
            expected_library_path = resolve_artifact_path(
                task_request.material_library_path,
                base_dir=task_request_path.parent,
            )
            if yaml_path != expected_yaml_path or library_path != expected_library_path:
                raise AssetTaskRuntimeError(
                    "Material decision library does not match the frozen material task request"
                )
    manifest = _resolve_material_library(yaml_path, library_path)
    unknown = sorted(
        {
            assignment.material_name
            for assignment in decision.assignments
            if assignment.material_name not in manifest.by_name
        }
    )
    if unknown:
        raise AssetTaskRuntimeError(f"Unknown material names: {unknown}")
    return manifest


def _load_material_decision(path: Path) -> MaterialDecisionPatch:
    try:
        return MaterialDecisionPatch.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise AssetTaskRuntimeError(
            f"Invalid material decision at {path}: {exc}"
        ) from exc


def _validate_usd_cli_library_lookups(
    decision: MaterialDecisionPatch,
    manifest: ResolvedMaterialManifest,
) -> None:
    """Fail before mutation when usd-cli cannot address the exact manifest prim."""

    from pxr import Tf, Usd, UsdShade

    stage = Usd.Stage.Open(str(manifest.library_path))
    if stage is None:
        raise AssetTaskRuntimeError(
            f"Could not reopen material USD library: {manifest.library_path}"
        )
    requested_names = {assignment.material_name for assignment in decision.assignments}
    for name in sorted(requested_names):
        entry = manifest.by_name[name]
        lookup_name = Tf.MakeValidIdentifier(entry.name).lower()
        matches = sorted(
            str(prim.GetPath())
            for prim in stage.Traverse()
            if UsdShade.Material(prim) and prim.GetName().lower() == lookup_name
        )
        if matches != [entry.binding_path]:
            raise AssetTaskRuntimeError(
                "usd-cli cannot unambiguously address the selected workflow "
                f"material {entry.name!r}; expected {entry.binding_path!r}, "
                f"lookup matches={matches}"
            )


def _usd_cli_local_material_path(stage: Any, material_name: str) -> str:
    """Return the exact local path used by usd-cli for a library material."""

    from pxr import Sdf, Tf

    default_prim = stage.GetDefaultPrim()
    default_prim_path = (
        default_prim.GetPath()
        if default_prim
        and default_prim.IsValid()
        and default_prim.GetPath().pathString != "/"
        else Sdf.Path.absoluteRootPath
    )
    return (
        default_prim_path.AppendChild("Looks")
        .AppendChild(Tf.MakeValidIdentifier(material_name))
        .pathString
    )


def _validate_usd_cli_import_destinations(
    *,
    stage_path: str | Path,
    decision: MaterialDecisionPatch,
) -> None:
    """Fail closed when a library import could compose with existing opinions."""

    from pxr import Usd

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise AssetTaskRuntimeError(
            f"Could not reopen staged USD before material import: {stage_path}"
        )

    names_by_destination: dict[str, set[str]] = {}
    for assignment in decision.assignments:
        destination = _usd_cli_local_material_path(stage, assignment.material_name)
        names_by_destination.setdefault(destination, set()).add(
            assignment.material_name
        )

    ambiguous = {
        path: sorted(names)
        for path, names in names_by_destination.items()
        if len(names) > 1
    }
    if ambiguous:
        raise AssetTaskRuntimeError(
            "Selected material names resolve to the same usd-cli import "
            f"destination: {ambiguous}"
        )

    occupied = sorted(
        path for path in names_by_destination if stage.GetPrimAtPath(path).IsValid()
    )
    if occupied:
        raise AssetTaskRuntimeError(
            "usd-cli library import destination already exists in the staged "
            "scene; importing by reference could leave stronger source material "
            f"opinions active instead of the selected library material: {occupied}"
        )


def _validate_saved_material_bindings(
    *,
    output_stage: Any,
    decision: MaterialDecisionPatch,
    manifest: ResolvedMaterialManifest,
    scene_backend: MaterialSceneBackend,
) -> list[str]:
    """Reopen the derivative and verify every accepted target's effective binding."""

    from pxr import UsdShade

    errors: list[str] = []
    for assignment in decision.assignments:
        prim = output_stage.GetPrimAtPath(assignment.target_prim_path)
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        if not material:
            errors.append(
                f"Saved {scene_backend} derivative has no effective material at "
                f"{assignment.target_prim_path}."
            )
            continue
        expected_path = manifest.by_name[assignment.material_name].binding_path
        actual_path = str(material.GetPrim().GetPath())
        # ``usd-cli material --library`` imports the selected library material
        # beneath the opened stage's Looks scope.  Composition flattening then
        # removes the reference while preserving that canonical local path, so a
        # library material declared at /World/Looks/<Name> can legitimately be
        # bound from <defaultPrim>/Looks/<Name> in the saved derivative.  The
        # A flattened scene-tool derivative localizes the same way: apply-materials
        # remaps every library material path beneath the input stage's default
        # prim, so a default prim named unlike the library root also binds from
        # <defaultPrim>/Looks/<Name>.
        #
        # Derive the one allowed localized path from the stage default prim and
        # the exact identifier the backends author.  Do not accept a
        # basename-only match: /Counterfeit/.../<Name> must continue to fail
        # validation.
        localized_path = _usd_cli_local_material_path(
            output_stage, assignment.material_name
        )
        if scene_backend == "usd-cli":
            saved_expected_paths = {localized_path}
        else:
            saved_expected_paths = {expected_path, localized_path}
        if actual_path not in saved_expected_paths:
            errors.append(
                f"Saved {scene_backend} derivative bound the wrong material at "
                f"{assignment.target_prim_path}: expected manifest path "
                f"{expected_path!r}, got {actual_path!r}."
            )
    return errors


def _render_validation(
    *,
    image_reference: str | None,
    backend_label: str,
    image_bytes: bytes | None = None,
) -> tuple[dict[str, object] | None, list[str]]:
    errors: list[str] = []
    blankness: dict[str, object] | None = None
    if not image_reference:
        errors.append(
            f"{backend_label} verification render did not return an image path."
        )
        return blankness, errors
    image_path = Path(image_reference).expanduser().resolve()
    if image_bytes is None and not image_path.is_file():
        errors.append(f"{backend_label} verification render is missing: {image_path}.")
        return blankness, errors
    from world_understanding.utils.image_blankness import analyze_image_blankness

    stats = analyze_image_blankness(
        image_bytes if image_bytes is not None else image_path
    )
    blankness = stats.to_dict()
    if stats.blank:
        errors.append(
            f"{backend_label} verification render is blank: "
            f"{image_path} ({stats.reason})."
        )
    return blankness, errors


def _validate_usd_cli_output_scope(
    *,
    processing_root: Path,
    output_dir: Path,
) -> None:
    """Reject an output path that escapes through a child-controlled symlink."""

    root = processing_root.resolve(strict=True)
    try:
        relative_output = output_dir.relative_to(root)
    except ValueError as exc:
        raise AssetTaskRuntimeError(
            f"usd-cli per-asset output escapes the processing run: {output_dir}"
        ) from exc
    current = root
    for component in relative_output.parts:
        current = current / component
        if current.is_symlink():
            raise AssetTaskRuntimeError(
                "usd-cli per-asset output must not contain symlink components: "
                f"{current}"
            )


def _usd_cli_session_scope(
    *,
    processing_root: Path,
    output_dir: Path,
    work_item_id: str,
    attempt_count: int,
    input_roots: tuple[Path, ...],
) -> WorkflowUsdCliSession:
    """Create one trusted daemon project and named session per item attempt."""

    _validate_usd_cli_output_scope(
        processing_root=processing_root,
        output_dir=output_dir,
    )
    if attempt_count < 1:
        raise AssetTaskRuntimeError("usd-cli material attempt_count must be positive")
    if os.name == "nt":
        # Keep OpenUSD-authored layers out of the descriptive per-asset artifact
        # tree.  The terse, digest-bound scope is still run-confined, unique per
        # work item and attempt, and leaves enough room for TfSafeOutputFile's
        # private temporary names on native Windows.
        item_digest = hashlib.sha256(work_item_id.encode("utf-8")).hexdigest()[:20]
        project_dir = processing_root / f".m-{item_digest}-a{attempt_count:04d}"
        clean_slate_target = project_dir / "clean_slate_layer.usda"
        target_code_units = len(os.fspath(clean_slate_target).encode("utf-16-le")) // 2
        if (
            target_code_units + WINDOWS_OPENUSD_REPLACE_SUFFIX_CODE_UNITS
            >= WINDOWS_LEGACY_MAX_PATH_CODE_UNITS
        ):
            raise AssetTaskRuntimeError(
                "Windows run directory is too long for OpenUSD atomic layer "
                "publication; choose a shorter --output-dir "
                f"(target={target_code_units} UTF-16 code units, reserved "
                f"suffix={WINDOWS_OPENUSD_REPLACE_SUFFIX_CODE_UNITS}, limit="
                f"{WINDOWS_LEGACY_MAX_PATH_CODE_UNITS - 1}): "
                f"{clean_slate_target}"
            )
    else:
        project_dir = output_dir / "usd_cli_sessions" / f"attempt-{attempt_count:04d}"
    try:
        return WorkflowUsdCliSession.create(
            owner_root=processing_root,
            project_dir=project_dir,
            identity=f"{work_item_id}:attempt:{attempt_count}",
            workflow="material-task",
            input_roots=input_roots,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise AssetTaskRuntimeError(
            "Could not create the usd-cli per-asset session scope: "
            f"{project_dir}: {exc}"
        ) from exc


def _require_ovrtx_probe(
    payload: dict[str, object],
    *,
    probe_dir: Path,
    require_clear_appearance: bool,
) -> Path:
    """Require strict OVRTX identity plus one contained, decodable 64px render."""

    required_capabilities = ("appearance.clear.v1",) if require_clear_appearance else ()
    try:
        validate_ovrtx_probe(
            payload,
            required_capabilities=required_capabilities,
        )
    except RuntimeError as exc:
        raise AssetTaskRuntimeError(str(exc)) from exc

    render = payload.get("render")
    render_path = render.get("path") if isinstance(render, dict) else None
    if not isinstance(render_path, str) or not render_path:
        raise AssetTaskRuntimeError("OVRTX probe did not return a render artifact path")
    try:
        render_artifact = read_contained_artifact(
            probe_dir,
            render_path,
            max_bytes=16 * 1024 * 1024,
            image=True,
            capture_bytes=True,
        )
    except ValueError as exc:
        raise AssetTaskRuntimeError(
            f"OVRTX probe render evidence is unsafe: {exc}"
        ) from exc

    from PIL import Image

    assert render_artifact.data is not None
    with Image.open(io.BytesIO(render_artifact.data)) as image:
        dimensions = image.size
    if dimensions != (64, 64):
        raise AssetTaskRuntimeError(
            f"OVRTX probe render artifact has unexpected dimensions: {dimensions!r}"
        )
    reported_size = render.get("size_bytes")
    if reported_size != render_artifact.size_bytes:
        raise AssetTaskRuntimeError(
            "OVRTX probe render artifact size does not match its evidence record"
        )
    return render_artifact.path


def _require_clear_appearance_audit(payload: dict[str, object]) -> None:
    data = payload.get("data")
    if (
        not isinstance(data, dict)
        or data.get("clear") is not True
        or data.get("overlay_active") is not True
    ):
        raise AssetTaskRuntimeError(
            "usd-cli appearance audit did not prove a clean-slate overlay"
        )
    counts = data.get("counts")
    required_counts = {
        "binding_relationships_with_targets",
        "effective_material_bindings",
        "effective_shader_appearances",
        "display_values",
        "instance_proxies",
    }
    if (
        not isinstance(counts, dict)
        or not required_counts.issubset(counts)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value != 0
            for key, value in counts.items()
            if key in required_counts
        )
    ):
        raise AssetTaskRuntimeError(
            "usd-cli appearance audit reported remaining appearance evidence"
        )


def _execute_usd_cli_material(
    *,
    processing_root: Path,
    output_dir: Path,
    work_item_id: str,
    attempt_count: int,
    working_usd_path: str,
    task_request: MaterialTaskRequest,
    decision: MaterialDecisionPatch,
    manifest: ResolvedMaterialManifest,
    render: bool,
) -> _MaterialSceneExecution:
    """Use usd-cli only for validated, low-level per-asset scene operations."""

    _validate_usd_cli_import_destinations(
        stage_path=working_usd_path,
        decision=decision,
    )
    session = _usd_cli_session_scope(
        processing_root=processing_root,
        output_dir=output_dir,
        work_item_id=work_item_id,
        attempt_count=attempt_count,
        input_roots=(
            Path(working_usd_path).resolve(),
            manifest.library_path.resolve(),
        ),
    )
    project_dir = session.project_dir
    session_id = session.session_id
    source_digest = file_sha256(working_usd_path)
    command_records: list[dict[str, object]] = []
    backend_artifacts: list[Path] = []
    primary_error = False
    try:
        probe_dir = project_dir / "ovrtx_probe"
        atomic_write_text(
            probe_dir / ".workflow-owned",
            "content-agent-workflows.material-task-probe\n",
            within=processing_root,
        )
        probe_response = session.require_ovrtx(probe_dir)
        probe_execution_source = probe_response.get("execution_source")
        if probe_execution_source not in {
            "parent-readiness-reuse",
            "render-probe",
        }:
            raise AssetTaskRuntimeError(
                "usd-cli OVRTX readiness omitted its execution source"
            )
        probe_render_path = _require_ovrtx_probe(
            probe_response,
            probe_dir=probe_dir,
            require_clear_appearance=(
                not task_request.respect_existing_material_bindings
            ),
        )
        probe_path = atomic_write_json(
            output_dir / "ovrtx_probe.json",
            probe_response,
            within=processing_root,
        )
        backend_artifacts.extend([probe_path, probe_render_path])
        command_records.append(
            {
                "command": probe_execution_source,
                "status": "passed",
                "engine": "ovrtx",
            }
        )

        session.open(Path(working_usd_path))
        command_records.append({"command": "open", "status": "applied"})
        session.run_json(["checkpoint", "save", "workflow-open", "--full"])
        command_records.append(
            {
                "command": "checkpoint save",
                "checkpoint": "workflow-open",
                "status": "applied",
            }
        )
        if not task_request.respect_existing_material_bindings:
            clear_response = session.run_json(["appearance", "clear"])
            clear_audit = session.run_json(["appearance", "audit"])
            _require_clear_appearance_audit(clear_audit)
            clear_report_path = atomic_write_json(
                output_dir / "appearance_clear_report.json",
                {
                    "schema_version": (
                        "content-agent-workflows.appearance-clear-report.v1"
                    ),
                    "scene_backend": "usd-cli",
                    "session_id": session_id,
                    "source_usd": str(Path(working_usd_path).resolve()),
                    "clear_response": clear_response,
                    "audit_response": clear_audit,
                },
                within=processing_root,
            )
            backend_artifacts.append(clear_report_path)
            command_records.extend(
                [
                    {"command": "appearance clear", "status": "applied"},
                    {"command": "appearance audit", "status": "passed"},
                ]
            )

            # ``appearance clear`` deliberately authors its masks in the anonymous
            # session layer.  The clean usd-cli package revision resolves library
            # references relative to the file-backed root layer, so material imports
            # must not be attempted while that session layer remains the edit target.
            # Persist the audited clean composition first, then reopen that immutable
            # derivative as the file-backed working stage for all accepted bindings.
            clean_slate_layer = project_dir / "clean_slate_layer.usda"
            clean_slate_save_response = session.run_json(
                ["save", str(clean_slate_layer), "--flatten"]
            )
            try:
                clean_slate_artifact = read_contained_artifact(
                    processing_root,
                    clean_slate_layer,
                    max_bytes=MAX_MATERIAL_STAGE_BYTES,
                )
            except ValueError as exc:
                raise AssetTaskRuntimeError(
                    f"usd-cli did not create a safe clean-slate derivative: {exc}"
                ) from exc
            if clean_slate_artifact.size_bytes <= 0:
                raise AssetTaskRuntimeError(
                    "usd-cli created an empty clean-slate derivative"
                )
            clean_slate_layer = clean_slate_artifact.path
            backend_artifacts.append(clean_slate_layer)
            command_records.append(
                {
                    "command": "save",
                    "output": str(clean_slate_layer),
                    "flatten": True,
                    "purpose": "clean-slate",
                    "status": "applied",
                    "response": clean_slate_save_response,
                }
            )
            session.run_json(
                [
                    "open",
                    str(clean_slate_layer),
                    "--force-reload",
                ],
            )
            command_records.append(
                {
                    "command": "open",
                    "input": str(clean_slate_layer),
                    "force_reload": True,
                    "purpose": "clean-slate",
                    "status": "applied",
                }
            )
            session.run_json(
                [
                    "checkpoint",
                    "save",
                    "workflow-clean-slate",
                    "--full",
                ],
            )
            command_records.append(
                {
                    "command": "checkpoint save",
                    "checkpoint": "workflow-clean-slate",
                    "status": "applied",
                }
            )
            _validate_usd_cli_import_destinations(
                stage_path=clean_slate_layer,
                decision=decision,
            )

        normalized_assignments: list[dict[str, object]] = []
        for assignment in decision.assignments:
            material = manifest.by_name[assignment.material_name]
            response = session.run_json(
                [
                    "material",
                    assignment.target_prim_path,
                    "--library",
                    str(manifest.library_path),
                    "--name",
                    material.name,
                ],
            )
            record = {
                "target_prim_path": assignment.target_prim_path,
                "material_name": material.name,
                "material_path": material.binding_path,
                "status": "applied",
            }
            command_records.append({"command": "material", **record})
            normalized_assignments.append({**record, "response": response})

        audit_response = session.run_json(
            ["material", "audit", "--effective", "--include-subsets"]
        )
        command_records.append({"command": "material audit", "status": "passed"})
        session.run_json(["checkpoint", "save", "workflow-applied", "--full"])
        command_records.append(
            {
                "command": "checkpoint save",
                "checkpoint": "workflow-applied",
                "status": "applied",
            }
        )

        preview_layer = project_dir / "preview_layer.usda"
        save_response = session.run_json(["save", str(preview_layer), "--flatten"])
        command_records.append(
            {
                "command": "save",
                "output": str(preview_layer),
                "flatten": True,
                "status": "applied",
            }
        )
        try:
            preview_artifact = read_contained_artifact(
                processing_root,
                preview_layer,
                max_bytes=MAX_MATERIAL_STAGE_BYTES,
                capture_bytes=True,
            )
        except ValueError as exc:
            raise AssetTaskRuntimeError(
                f"usd-cli did not create a safe material derivative: {exc}"
            ) from exc
        if preview_artifact.size_bytes <= 0:
            raise AssetTaskRuntimeError("usd-cli created an empty material derivative")
        assert preview_artifact.data is not None
        preview_layer = preview_artifact.path
        output_stage = _open_captured_material_stage(
            preview_artifact.data,
            suffix=preview_layer.suffix,
        )

        render_paths: list[str] = []
        render_validation_path: Path | None = None
        render_validation_errors: list[str] = []
        if render:
            render_dir = project_dir / "final_renders"
            atomic_write_text(
                render_dir / ".workflow-owned",
                "content-agent-workflows.material-task-render\n",
                within=processing_root,
            )
            image_path = render_dir / "final_oblique.png"
            render_response = session.run_json(
                [
                    "render",
                    "--res",
                    "512x512",
                    "-o",
                    str(image_path),
                ],
            )
            try:
                image_artifact = read_contained_artifact(
                    processing_root,
                    image_path,
                    max_bytes=128 * 1024 * 1024,
                    image=True,
                    capture_bytes=True,
                )
            except ValueError as exc:
                raise AssetTaskRuntimeError(
                    f"usd-cli produced unsafe final render evidence: {exc}"
                ) from exc
            assert image_artifact.data is not None
            image_path = image_artifact.path
            blankness, render_validation_errors = _render_validation(
                image_reference=str(image_path),
                backend_label="usd-cli OVRTX",
                image_bytes=image_artifact.data,
            )
            render_validation_path = atomic_write_json(
                render_dir / "final_oblique_validation.json",
                {
                    "schema_version": (
                        "content-agent-workflows.material-render-validation.v1"
                    ),
                    "passed": not render_validation_errors,
                    "image_path": str(image_path),
                    "blankness": blankness,
                    "response": render_response,
                    "errors": render_validation_errors,
                },
                within=processing_root,
            )
            render_paths.extend([str(image_path), str(render_validation_path)])
            command_records.append({"command": "render", "status": "applied"})

        source_digest_after = file_sha256(working_usd_path)
        if source_digest_after != source_digest:
            raise AssetTaskRuntimeError(
                "usd-cli material execution changed the immutable source USD"
            )
        command_artifact_path = atomic_write_json(
            output_dir / "usd_cli_commands.json",
            {
                "schema_version": (
                    "content-agent-workflows.usd-cli-command-records.v1"
                ),
                "project_dir": str(project_dir),
                "session_id": session_id,
                "commands": command_records,
            },
            within=processing_root,
        )
        assignments_response: dict[str, object] = {
            "schema_version": ("content-agent-workflows.material-task-assignments.v1"),
            "scene_backend": "usd-cli",
            "assignments": normalized_assignments,
            "binding_audit": audit_response,
        }
        apply_response: dict[str, object] = {
            "schema_version": (
                "content-agent-workflows.material-task-apply-response.v1"
            ),
            "scene_backend": "usd-cli",
            "status": "applied",
            "applied_assignment_count": len(normalized_assignments),
            "save_response": save_response,
            "source_sha256_before": source_digest,
            "source_sha256_after": source_digest_after,
            "session_id": session_id,
            "project_dir": str(project_dir),
        }
        return _MaterialSceneExecution(
            preview_layer=preview_layer,
            output_stage=output_stage,
            assignments_response=assignments_response,
            apply_response=apply_response,
            command_records=command_records,
            command_artifact_path=command_artifact_path,
            render_paths=render_paths,
            render_validation_path=render_validation_path,
            render_validation_errors=render_validation_errors,
            backend_artifact_paths=tuple(backend_artifacts),
        )
    except Exception:
        primary_error = True
        raise
    finally:
        try:
            session.close()
        except RuntimeError as exc:
            if not primary_error:
                raise AssetTaskRuntimeError(
                    f"Could not close the per-asset usd-cli daemon: {exc}"
                ) from exc


def run_material_work_item(
    processing_dir: str | Path,
    work_item_id: str,
    *,
    decision_path: str | Path,
    render: bool = False,
    actor: str = "agent",
) -> AssetTaskResult:
    """Apply and validate one material decision through its frozen scene backend."""

    root = Path(processing_dir).expanduser().resolve()
    decision_file = Path(decision_path).expanduser().resolve()
    item, item_state = get_work_item(root, work_item_id)
    if item_state.status == "completed" and item_state.result_path:
        return AssetTaskResult.model_validate(load_json(item_state.result_path))
    if not item.working_usd_path:
        raise AssetTaskRuntimeError(f"Work item has no working USD: {work_item_id}")
    output_dir = _item_dir(root, work_item_id)
    survey_path = output_dir / "material_survey.json"
    if survey_path.is_file():
        survey = MaterialSurvey.model_validate(load_json(survey_path))
    else:
        survey, survey_path = survey_work_item(root, work_item_id)
    task_request, task_request_path, task_request_digest = load_material_task_request(
        root, item.task_id
    )
    decision = _load_material_decision(decision_file)
    scene_backend = _material_scene_backend(task_request)
    manifest = _validate_decision(
        decision,
        survey,
        task_request_digest=task_request_digest,
        task_request=task_request,
        task_request_path=task_request_path,
    )
    _validate_usd_cli_library_lookups(decision, manifest)
    _validate_usd_cli_import_destinations(
        stage_path=item.working_usd_path,
        decision=decision,
    )
    _validate_usd_cli_output_scope(
        processing_root=root,
        output_dir=output_dir,
    )
    palette_path = atomic_write_json(
        output_dir / "material_palette.json", manifest.as_palette()
    )

    begin_work_item(root, work_item_id, actor=actor)
    try:
        _running_item, running_state = get_work_item(root, work_item_id)
        execution = _execute_usd_cli_material(
            processing_root=root,
            output_dir=output_dir,
            work_item_id=work_item_id,
            attempt_count=running_state.attempt_count,
            working_usd_path=item.working_usd_path,
            task_request=task_request,
            decision=decision,
            manifest=manifest,
            render=render,
        )
        preview_layer = execution.preview_layer
        assignments_response = execution.assignments_response
        apply_response = execution.apply_response
        render_paths = execution.render_paths
        render_validation_path = execution.render_validation_path
        render_validation_errors = execution.render_validation_errors
        output_stage = execution.output_stage
        assignments_path = atomic_write_json(
            output_dir / "assignments.json", assignments_response
        )
        apply_response_path = atomic_write_json(
            output_dir / "material_apply_response.json", apply_response
        )

        applied_count = apply_response.get("applied_assignment_count")
        if not isinstance(applied_count, int):
            assignments = (
                assignments_response.get("assignments", [])
                if isinstance(assignments_response, Mapping)
                else []
            )
            applied_count = len(assignments) if isinstance(assignments, list) else 0
        validation_errors = list(render_validation_errors)
        if output_stage is None:
            validation_errors.append("Authored preview layer could not be opened.")
        else:
            validation_errors.extend(
                _validate_saved_material_bindings(
                    output_stage=output_stage,
                    decision=decision,
                    manifest=manifest,
                    scene_backend=scene_backend,
                )
            )
        if applied_count != len(decision.assignments):
            validation_errors.append(
                f"Expected {len(decision.assignments)} assignments, got {applied_count}."
            )
        validation_path = atomic_write_json(
            output_dir / "validation.json",
            {
                "schema_version": MATERIAL_VALIDATION_SCHEMA_VERSION,
                "passed": not validation_errors,
                "work_item_id": work_item_id,
                "scene_backend": scene_backend,
                "candidate_count": len(survey.candidates),
                "decision_count": len(decision.assignments),
                "applied_assignment_count": applied_count,
                "output_stage_opened": output_stage is not None,
                "render_validation_path": (
                    str(render_validation_path) if render_validation_path else None
                ),
                "errors": validation_errors,
            },
        )
        if validation_errors:
            raise AssetTaskRuntimeError("; ".join(validation_errors))

        pointer = AgentPlanPointer.model_validate(
            load_json(ProcessingPaths.from_output_dir(root).plan_pointer)
        )
        domain_outputs = {
            "task_request_path": str(task_request_path),
            "survey_path": str(survey_path),
            "decision_path": str(decision_file),
            "material_palette_path": str(palette_path),
            "assignments_path": str(assignments_path),
            "preview_layer_path": str(preview_layer),
            "apply_response_path": str(apply_response_path),
            "validation_path": str(validation_path),
            "scene_command_artifact_path": str(execution.command_artifact_path),
        }
        for index, backend_path in enumerate(execution.backend_artifact_paths):
            domain_outputs[f"backend_artifact_{index}"] = str(backend_path)
        for index, render_path in enumerate(render_paths):
            domain_outputs[f"render_artifact_{index}"] = render_path
        result = AssetTaskResult(
            task_id=item.task_id,
            domain="material",
            manifest_id=item.manifest_id,
            asset_id=item.asset_id,
            original_root_path=item.original_root_path,
            working_usd_path=item.working_usd_path,
            domain_outputs=domain_outputs,
            provenance={
                "agent_plan_revision": pointer.current_revision,
                "task_request_digest": task_request_digest,
                "informed_by_results": decision.informed_by_results,
            },
            warnings=decision.warnings,
        )
        result_path = atomic_write_json(output_dir / "result.json", result)
        ledger_entry = DecisionLedgerEntry(
            work_item_id=work_item_id,
            domain="material",
            task_id=item.task_id,
            evidence_summary=decision.evidence_summary,
            artifact_paths=[
                *decision.evidence_paths,
                str(survey_path),
                str(decision_file),
                str(task_request_path),
                str(palette_path),
                str(assignments_path),
                str(preview_layer),
                str(execution.command_artifact_path),
                *(str(path) for path in execution.backend_artifact_paths),
                *render_paths,
            ],
            confidence=decision.confidence,
            rationale="; ".join(
                assignment.rationale for assignment in decision.assignments
            ),
            validation_status="passed",
            agent_plan_revision=pointer.current_revision,
            task_request_digest=task_request_digest,
            informed_by_results=decision.informed_by_results,
        )
        ledger_entry_path = atomic_write_json(
            output_dir / "ledger_entry.json", ledger_entry
        )
        commit_work_item(
            root,
            work_item_id,
            result_path=result_path,
            validation_path=validation_path,
            ledger_entry_path=ledger_entry_path,
            actor=actor,
        )
        return result
    except Exception as exc:
        try:
            _item, current_state = get_work_item(root, work_item_id)
            if current_state.status == "running":
                fail_work_item(root, work_item_id, reason=str(exc), actor=actor)
        except Exception:
            pass
        if isinstance(exc, AssetTaskRuntimeError):
            raise
        raise AssetTaskRuntimeError(
            f"Material task failed for {work_item_id}: {exc}"
        ) from exc


def run_material_batch(
    processing_dir: str | Path,
    batch_plan_path: str | Path,
    *,
    actor: str = "agent",
    fail_fast: bool = False,
) -> dict[str, object]:
    """Execute an agent-authored material plan sequentially with resume."""

    plan_file = Path(batch_plan_path).expanduser().resolve()
    try:
        plan = MaterialBatchPlan.model_validate(load_json(plan_file))
    except (OSError, ValueError, ValidationError) as exc:
        raise AssetTaskRuntimeError(
            f"Invalid material batch plan {plan_file}: {exc}"
        ) from exc
    completed: list[str] = []
    failed: dict[str, str] = {}
    for batch_item in plan.items:
        decision_path = resolve_artifact_path(
            batch_item.decision_path, base_dir=plan_file.parent
        )
        try:
            run_material_work_item(
                processing_dir,
                batch_item.work_item_id,
                decision_path=decision_path,
                render=batch_item.render,
                actor=actor,
            )
        except AssetTaskRuntimeError as exc:
            failed[batch_item.work_item_id] = str(exc)
            if plan.stop_on_error or fail_fast:
                break
        else:
            completed.append(batch_item.work_item_id)
    return {
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed_work_item_ids": completed,
        "failures": failed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Survey or execute Workflow 2 material work items."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    survey = subparsers.add_parser("survey")
    survey.add_argument("--processing-dir", type=Path, required=True)
    survey.add_argument("--task-id", default="material")
    survey.add_argument("--render-index", type=Path)

    match_display_color = subparsers.add_parser("match-display-color")
    match_display_color.add_argument("--processing-dir", type=Path, required=True)
    match_display_color.add_argument("--work-item-id", required=True)
    match_display_color.add_argument("--scope", action="append", required=True)
    match_display_color.add_argument("--top-k", type=int, default=5)
    match_display_color.add_argument("--appearance-cache-dir", type=Path)
    match_display_color.add_argument("--swatch-template", type=Path)

    run_item = subparsers.add_parser("run-item")
    run_item.add_argument("--processing-dir", type=Path, required=True)
    run_item.add_argument("--work-item-id", required=True)
    run_item.add_argument("--decision", type=Path, required=True)
    run_item.add_argument("--render", action="store_true")
    run_item.add_argument("--actor", default="agent")

    run_batch = subparsers.add_parser("run-batch")
    run_batch.add_argument("--processing-dir", type=Path, required=True)
    run_batch.add_argument("--batch-plan", type=Path, required=True)
    run_batch.add_argument("--actor", default="agent")
    run_batch.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop after the first failed work item and return a nonzero exit "
            "status so supervised workflows fail at the originating command."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "survey":
            output = survey_material_inventory(
                args.processing_dir,
                task_id=args.task_id,
                render_index_path=args.render_index,
            )
        elif args.command == "match-display-color":
            output = match_work_item_display_colors(
                args.processing_dir,
                args.work_item_id,
                scope_paths=args.scope,
                top_k=args.top_k,
                appearance_cache_dir=args.appearance_cache_dir,
                swatch_template_path=args.swatch_template,
            )
        elif args.command == "run-item":
            output = run_material_work_item(
                args.processing_dir,
                args.work_item_id,
                decision_path=args.decision,
                render=args.render,
                actor=args.actor,
            ).model_dump(mode="json")
        elif args.command == "run-batch":
            output = run_material_batch(
                args.processing_dir,
                args.batch_plan,
                actor=args.actor,
                fail_fast=args.fail_fast,
            )
        else:  # pragma: no cover
            raise AssertionError(f"Unhandled command: {args.command}")
    except AssetTaskRuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    if (
        args.command == "run-batch"
        and args.fail_fast
        and isinstance(output.get("failed_count"), int)
        and output["failed_count"] > 0
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
