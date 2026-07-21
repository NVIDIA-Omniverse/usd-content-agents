# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Material-task survey and Workbench execution for Workflow 2 work items."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from content_workbench_agent_client.client import (
    apply_command,
    close_session,
    create_session,
    get_material_assignments,
    post_json,
    render_view,
    session_url,
    wait_until_healthy,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
    resolve_artifact_path,
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
APPEARANCE_EVIDENCE_POLICY_SCHEMA_VERSION = (
    "content-agent-workflows.appearance-evidence-policy.v1"
)
DEFAULT_WORKBENCH_URL = os.environ.get("CONTENT_WORKBENCH_URL", "http://127.0.0.1:8088")

AppearanceEvidenceSource = Literal["material_binding", "display_color"]


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
    workbench_url: str,
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
            workbench_url=workbench_url,
        )
        target_appearances = render_display_color_targets(
            colors=scoped_colors,
            swatch_template_path=template_path,
            output_dir=_item_dir(root, work_item_id) / "display_color_target_swatches",
            workbench_url=workbench_url,
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


def _load_material_library(
    yaml_path: Path,
    expected_library_path: Path,
) -> dict[str, str]:
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AssetTaskRuntimeError(
            f"Cannot read material library {yaml_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise AssetTaskRuntimeError(f"Invalid material library metadata: {yaml_path}")
    configured_library = payload.get("library_path")
    if isinstance(configured_library, str):
        configured_path = resolve_artifact_path(
            configured_library, base_dir=yaml_path.parent
        )
        if configured_path != expected_library_path:
            raise AssetTaskRuntimeError(
                "Decision material_library_path does not match material YAML"
            )
    materials: dict[str, str] = {}
    for entry in payload["entries"]:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        binding = entry.get("binding")
        if isinstance(name, str) and isinstance(binding, str):
            materials[name] = binding
    return materials


def _validate_decision(
    decision: MaterialDecisionPatch,
    survey: MaterialSurvey,
    *,
    task_request_digest: str | None = None,
    task_request: MaterialTaskRequest | None = None,
    task_request_path: Path | None = None,
) -> dict[str, str]:
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
    if not library_path.is_file():
        raise AssetTaskRuntimeError(
            f"Material USD library does not exist: {library_path}"
        )
    materials = _load_material_library(yaml_path, library_path)
    unknown = sorted(
        {
            assignment.material_name
            for assignment in decision.assignments
            if assignment.material_name not in materials
        }
    )
    if unknown:
        raise AssetTaskRuntimeError(f"Unknown material names: {unknown}")
    return materials


def _load_material_decision(path: Path) -> MaterialDecisionPatch:
    try:
        return MaterialDecisionPatch.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise AssetTaskRuntimeError(
            f"Invalid material decision at {path}: {exc}"
        ) from exc


def run_material_work_item(
    processing_dir: str | Path,
    work_item_id: str,
    *,
    decision_path: str | Path,
    workbench_url: str,
    render: bool = False,
    actor: str = "agent",
) -> AssetTaskResult:
    """Apply and validate one material decision through Content Workbench."""

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
    materials = _validate_decision(
        decision,
        survey,
        task_request_digest=task_request_digest,
        task_request=task_request,
        task_request_path=task_request_path,
    )

    begin_work_item(root, work_item_id, actor=actor)
    session_id = ""
    try:
        wait_until_healthy(workbench_url, timeout_seconds=30.0)
        session = create_session(
            workbench_url,
            {
                "scene_path": item.working_usd_path,
                "optimize": False,
                "clear_materials": not task_request.respect_existing_material_bindings,
                "width": 512,
                "height": 512,
            },
        )
        session_id_value = session.get("session_id")
        if not isinstance(session_id_value, str) or not session_id_value:
            raise AssetTaskRuntimeError("Workbench did not return a session_id")
        session_id = session_id_value
        command_records: list[dict[str, object]] = []
        library_path = str(Path(decision.material_library_path).expanduser().resolve())
        for assignment in decision.assignments:
            response = apply_command(
                workbench_url,
                session_id,
                "material_override",
                {
                    "prim_path": assignment.target_prim_path,
                    "space": "source",
                    "unbind_existing": not task_request.respect_existing_material_bindings,
                    "material": {
                        "source": "material_library",
                        "library_path": library_path,
                        "material_name": assignment.material_name,
                        "material_path": materials[assignment.material_name],
                    },
                },
            )
            command_records.append(
                {
                    "target_prim_path": assignment.target_prim_path,
                    "material_name": assignment.material_name,
                    "status": response.get("status", "applied"),
                }
            )

        assignments_response = get_material_assignments(workbench_url, session_id)
        preview_layer = output_dir / "preview_layer.usda"
        apply_response = post_json(
            session_url(
                workbench_url,
                session_id,
                "/authoring/material-assignments:apply",
            ),
            {
                "output_usd_path": str(preview_layer),
                "output_mode": "layer",
                "material_profile": "preview_surface",
                "overwrite": True,
            },
        )
        atomic_write_json(
            output_dir / "workbench_commands.json", {"commands": command_records}
        )
        assignments_path = atomic_write_json(
            output_dir / "assignments.json", assignments_response
        )
        apply_response_path = atomic_write_json(
            output_dir / "material_apply_response.json", apply_response
        )

        render_paths: list[str] = []
        render_validation_path: Path | None = None
        render_validation_errors: list[str] = []
        if render:
            (output_dir / "final_renders").mkdir(parents=True, exist_ok=True)
            record = render_view(
                workbench_url=workbench_url,
                session_id=session_id,
                output_dir=output_dir / "final_renders",
                name="final_oblique",
                direction="oblique",
                width=512,
                height=512,
                render_quality="inspection",
            )
            render_paths.extend(
                str(path)
                for key in ("image_path", "camera_json_path", "response_path")
                if (path := record.get(key))
            )
            image_reference = record.get("image_path")
            blankness: dict[str, object] | None = None
            if not isinstance(image_reference, str) or not image_reference:
                render_validation_errors.append(
                    "Workbench verification render did not return an image path."
                )
            else:
                image_path = Path(image_reference).expanduser().resolve()
                if not image_path.is_file():
                    render_validation_errors.append(
                        f"Workbench verification render is missing: {image_path}."
                    )
                else:
                    from world_understanding.utils.image_blankness import (
                        analyze_image_blankness,
                    )

                    stats = analyze_image_blankness(image_path)
                    blankness = stats.to_dict()
                    if stats.blank:
                        render_validation_errors.append(
                            "Workbench verification render is blank: "
                            f"{image_path} ({stats.reason})."
                        )
            render_validation_path = atomic_write_json(
                output_dir / "final_renders" / "final_oblique_validation.json",
                {
                    "schema_version": (
                        "content-agent-workflows.material-render-validation.v1"
                    ),
                    "passed": not render_validation_errors,
                    "image_path": image_reference,
                    "blankness": blankness,
                    "errors": render_validation_errors,
                },
            )
            render_paths.append(str(render_validation_path))

        from pxr import Usd

        output_stage = (
            Usd.Stage.Open(str(preview_layer)) if preview_layer.is_file() else None
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
            "assignments_path": str(assignments_path),
            "preview_layer_path": str(preview_layer),
            "apply_response_path": str(apply_response_path),
            "validation_path": str(validation_path),
        }
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
                str(assignments_path),
                str(preview_layer),
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
    finally:
        if session_id:
            try:
                close_session(workbench_url, session_id)
            except Exception:
                pass


def run_material_batch(
    processing_dir: str | Path,
    batch_plan_path: str | Path,
    *,
    workbench_url: str,
    actor: str = "agent",
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
                workbench_url=workbench_url,
                render=batch_item.render,
                actor=actor,
            )
        except AssetTaskRuntimeError as exc:
            failed[batch_item.work_item_id] = str(exc)
            if plan.stop_on_error:
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
    match_display_color.add_argument("--workbench-url", default=DEFAULT_WORKBENCH_URL)

    run_item = subparsers.add_parser("run-item")
    run_item.add_argument("--processing-dir", type=Path, required=True)
    run_item.add_argument("--work-item-id", required=True)
    run_item.add_argument("--decision", type=Path, required=True)
    run_item.add_argument("--workbench-url", default=DEFAULT_WORKBENCH_URL)
    run_item.add_argument("--render", action="store_true")
    run_item.add_argument("--actor", default="agent")

    run_batch = subparsers.add_parser("run-batch")
    run_batch.add_argument("--processing-dir", type=Path, required=True)
    run_batch.add_argument("--batch-plan", type=Path, required=True)
    run_batch.add_argument("--workbench-url", default=DEFAULT_WORKBENCH_URL)
    run_batch.add_argument("--actor", default="agent")
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
                workbench_url=args.workbench_url,
                top_k=args.top_k,
                appearance_cache_dir=args.appearance_cache_dir,
                swatch_template_path=args.swatch_template,
            )
        elif args.command == "run-item":
            output = run_material_work_item(
                args.processing_dir,
                args.work_item_id,
                decision_path=args.decision,
                workbench_url=args.workbench_url,
                render=args.render,
                actor=args.actor,
            ).model_dump(mode="json")
        elif args.command == "run-batch":
            output = run_material_batch(
                args.processing_dir,
                args.batch_plan,
                workbench_url=args.workbench_url,
                actor=args.actor,
            )
        else:  # pragma: no cover
            raise AssertionError(f"Unhandled command: {args.command}")
    except AssetTaskRuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
