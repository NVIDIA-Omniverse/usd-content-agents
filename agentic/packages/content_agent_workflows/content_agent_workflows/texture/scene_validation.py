# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""usd-cli render/VQA adapters for bounded Texture workflows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pxr import Sdf, Usd, UsdGeom, UsdShade
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from texture_agent.functions.detail_policy import apply_detail_policy_to_prompt
from world_understanding.config.s3 import WU_S3_PROFILE, WU_S3_REGION
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    open_confined_directory_at,
)
from world_understanding.utils.s3_utils import download_file_from_s3

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.usd_cli_session import (
    WorkflowUsdCliPackageRoutePin,
    WorkflowUsdCliSession,
    direction_angles,
    stage_up_axis_is_y,
    validated_ovrtx_render_metadata,
)

from .models import (
    TEXTURE_CODING_AGENT_COMPANION_BACKEND,
    TextureGeneratorInputs,
    TextureInspectionResult,
    TextureInspectionUnit,
    TexturePlanDocument,
    TextureUnitArtifact,
    TextureValidationFinding,
    TextureValidationResult,
    TextureValidationStatus,
    TextureWorkflowRequest,
)
from .runtime import (
    TextureArtifactDigest,
    TextureWorkflowRuntimeError,
    collect_artifact_digests,
    collect_validation_evidence_digests,
    texture_plan_digest,
    texture_request_digest,
    texture_source_identity_digest,
)
from .scope_validation import (
    TextureScopeInvariantReport,
    validate_texture_scope_invariants,
)

DEFAULT_MAX_TEXTURE_SOURCE_ASSET_BYTES = 500 * 1024 * 1024
_MAX_CACHED_TEXTURE_ASSESSMENT_BYTES = 1024 * 1024
_TEXTURE_VALIDATION_RESULT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-validation-result.v1"
] = "content-agent-workflows.texture-validation-result.v1"
_TEXTURE_VALIDATION_POLICY_DIGEST_SCHEMA = (
    "content-agent-workflows.texture-validation-policy.v1"
)
_VALIDATION_RESULT_MANIFEST_NAME = "validation_result.json"


def _ensure_private_texture_directory(path: str | Path) -> Path:
    """Create or verify one workflow-owned, symlink-free mode-0700 directory."""

    absolute = Path(os.path.abspath(Path(path).expanduser()))
    if absolute.name in {"", ".", ".."}:
        raise TextureWorkflowRuntimeError(
            f"Texture workflow directory path is unsafe: {absolute}"
        )
    parent = absolute.parent
    if not parent.exists():
        _ensure_private_texture_directory(parent)

    def validate_descriptor(descriptor: int, *, created: bool) -> None:
        # Windows confinement is handle- and ACL-based; POSIX mode bits there
        # are neither authoritative nor guaranteed to round-trip through chmod.
        if os.name != "posix":  # pragma: win32 cover
            return
        metadata = os.fstat(descriptor)
        if created and stat.S_ISDIR(metadata.st_mode):
            # Normalize umask and inherited setgid only on the inode this call
            # created. A permissive pre-existing directory remains fail-closed.
            os.fchmod(descriptor, 0o700)
            metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_nlink < 2
        ):
            raise TextureWorkflowRuntimeError(
                f"Texture workflow directory is not owner-controlled 0700: {absolute}"
            )

    try:
        with open_confined_directory(parent) as parent_descriptor:
            try:
                with open_confined_directory_at(
                    parent_descriptor,
                    absolute.name,
                    create=True,
                    mode=0o700,
                    exclusive_create=True,
                ) as target_descriptor:
                    validate_descriptor(target_descriptor, created=True)
            except FileExistsError:
                with open_confined_directory_at(
                    parent_descriptor,
                    absolute.name,
                ) as target_descriptor:
                    validate_descriptor(target_descriptor, created=False)
    except (ArtifactPathError, OSError, ValueError) as exc:
        raise TextureWorkflowRuntimeError(
            f"Texture workflow directory is unsafe: {absolute}"
        ) from exc
    return absolute


def _proposed_generator_inputs(
    request: TextureWorkflowRequest,
    plan: TexturePlanDocument,
    unit_context: Mapping[str, Any],
) -> TextureGeneratorInputs:
    """Normalize exact provider proposal inputs for outer acceptance."""

    plan_payload = plan.model_dump(mode="json")
    execution = plan_payload.get("execution")
    if not isinstance(execution, dict):
        execution = {}
    material_paths = tuple(
        str(path) for path in unit_context.get("material_prim_paths") or ()
    )
    material_textures = request.metadata.get("material_textures")
    texture_requests: list[Mapping[str, Any]] = []
    if isinstance(material_textures, Mapping):
        for material_path in material_paths:
            candidate = material_textures.get(material_path)
            if isinstance(candidate, Mapping):
                texture_requests.append(candidate)
    if texture_requests and any(
        candidate != texture_requests[0] for candidate in texture_requests[1:]
    ):
        raise ValueError(
            "Texture request assigns incompatible generator inputs to one unit"
        )
    texture_request = texture_requests[0] if texture_requests else None
    prompt = str((texture_request or {}).get("prompt") or request.intent).strip()
    raw_detail_policy = str(
        (texture_request or {}).get("detail_policy")
        or request.metadata.get("detail_policy")
        or "surface_only"
    )
    if raw_detail_policy not in {"default", "surface_only"}:
        raise ValueError("Texture proposal has an unsupported detail policy")
    detail_policy: Literal["default", "surface_only"] = (
        "default" if raw_detail_policy == "default" else "surface_only"
    )
    prompt = apply_detail_policy_to_prompt(prompt, detail_policy)
    explicit_backend = request.metadata.get("texture_backend")
    proposed_backend = explicit_backend or execution.get("backend")
    if proposed_backend in {None, "not_requested", "provider-default"}:
        proposed_backend = TEXTURE_CODING_AGENT_COMPANION_BACKEND
    return TextureGeneratorInputs(
        backend=str(proposed_backend),
        engine=(
            str(request.metadata["backend_engine"])
            if request.metadata.get("backend_engine") is not None
            else None
        ),
        prompt=prompt,
        detail_policy=detail_policy,
        seed=(
            int(request.metadata["seed"])
            if request.metadata.get("seed") is not None
            else None
        ),
        texture_size=int(execution.get("texture_size") or 1024),
        reference_artifacts=tuple(
            item.artifact for item in request.reference_artifacts
        ),
    )


def _bound_material_path(prim: Usd.Prim) -> str | None:
    """Return the effective material binding for one inspected source prim."""

    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return None
    material_prim = material.GetPrim()
    if not material_prim or not material_prim.IsValid():
        return None
    return str(material.GetPath())


def _source_bound_surfaces(stage: Usd.Stage) -> dict[str, str]:
    """Map every bound renderable or material subset to its source material."""

    bound: dict[str, str] = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset):
            material_path = _bound_material_path(prim)
            if material_path is not None:
                bound[str(prim.GetPath())] = material_path
    return bound


def _inspect_uv_primvar(prim: Usd.Prim) -> dict[str, Any]:
    """Return factual ``primvars:st`` state for a renderable or subset target."""

    uv_prim = prim.GetParent() if prim.IsA(UsdGeom.Subset) else prim
    primvar = UsdGeom.PrimvarsAPI(uv_prim).FindPrimvarWithInheritance("st")
    if not primvar or not primvar.GetAttr().IsValid():
        return {
            "prim_path": str(prim.GetPath()),
            "uv_prim_path": str(uv_prim.GetPath()),
            "status": "missing",
            "primvar": "primvars:st",
        }
    value = primvar.Get()
    value_count = len(value) if value is not None and hasattr(value, "__len__") else 0
    indices = primvar.GetIndices()
    indices_count = len(indices) if indices is not None else 0
    interpolation = str(primvar.GetInterpolation())
    supported_interpolations = {
        "constant",
        "uniform",
        "vertex",
        "varying",
        "faceVarying",
    }
    point_count: int | None = None
    face_count: int | None = None
    face_vertex_count: int | None = None
    expected_element_count: int | None = None
    if interpolation == "constant":
        expected_element_count = 1
    elif uv_prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(uv_prim)
        points = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        point_count = len(points) if points is not None else 0
        face_count = len(face_vertex_counts) if face_vertex_counts is not None else 0
        face_vertex_count = (
            sum(int(count) for count in face_vertex_counts)
            if face_vertex_counts is not None
            else 0
        )
        expected_element_count = {
            "uniform": face_count,
            "vertex": point_count,
            "varying": point_count,
            "faceVarying": face_vertex_count,
        }.get(interpolation)

    finite_values = True
    if value is not None:
        for item in value:
            try:
                components = tuple(item)
            except TypeError:
                components = (item,)
            try:
                if not all(math.isfinite(float(component)) for component in components):
                    finite_values = False
                    break
            except (TypeError, ValueError):
                finite_values = False
                break
    indexed = bool(primvar.IsIndexed())
    indices_in_range = not indexed or (
        value_count > 0
        and indices is not None
        and all(0 <= int(index) < value_count for index in indices)
    )
    reasons: list[str] = []
    if interpolation not in supported_interpolations:
        status = "unsupported"
        reasons.append(f"unsupported interpolation {interpolation or '<empty>'}")
    elif interpolation == "constant" and value_count == 0:
        status = "missing"
        reasons.append("primvars:st has no values")
    elif interpolation == "constant":
        status = "repair_required"
        reasons.append("constant primvars:st cannot vary across the target surface")
    elif expected_element_count is None or expected_element_count <= 0:
        status = "unsupported"
        reasons.append("target topology has no supported UV element count")
    elif value_count == 0:
        status = "missing"
        reasons.append("primvars:st has no values")
    elif not finite_values:
        status = "repair_required"
        reasons.append("primvars:st contains non-finite values")
    elif indexed and indices_count != expected_element_count:
        status = "repair_required"
        reasons.append("primvars:st index count differs from topology")
    elif indexed and not indices_in_range:
        status = "repair_required"
        reasons.append("primvars:st indices are outside the value array")
    elif not indexed and value_count != expected_element_count:
        status = "repair_required"
        reasons.append("primvars:st value count differs from topology")
    else:
        status = "ready"
    return {
        "prim_path": str(prim.GetPath()),
        "uv_prim_path": str(uv_prim.GetPath()),
        "status": status,
        "primvar": str(primvar.GetPrimvarName()),
        "interpolation": interpolation,
        "value_count": value_count,
        "indexed": indexed,
        "indices_count": indices_count,
        "indices_in_range": indices_in_range,
        "finite_values": finite_values,
        "expected_element_count": expected_element_count,
        "point_count": point_count,
        "face_count": face_count,
        "face_vertex_count": face_vertex_count,
        "reasons": reasons,
    }


def _inspect_source_unit(
    stage: Usd.Stage,
    *,
    unit_id: str,
    unit_context: Mapping[str, Any],
    bound_surfaces: Mapping[str, str],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    Literal["ready", "missing", "repair_required", "unsupported"],
    dict[str, Any],
]:
    """Validate provider target claims against source materials and UV primvars."""

    material_paths = tuple(
        str(path) for path in unit_context.get("material_prim_paths") or ()
    )
    member_paths = tuple(
        str(path) for path in unit_context.get("member_prim_paths") or ()
    )
    subset_paths = tuple(
        str(path) for path in unit_context.get("member_subset_paths") or ()
    )
    if not material_paths or not (member_paths or subset_paths):
        raise TextureWorkflowRuntimeError(
            "Texture source inspection requires material and surface paths for "
            f"{unit_id}"
        )
    for material_path in material_paths:
        prim = stage.GetPrimAtPath(material_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdShade.Material):
            raise TextureWorkflowRuntimeError(
                f"Texture proposal material is absent from the source: {material_path}"
            )
    uv_records: list[dict[str, Any]] = []
    material_records: list[dict[str, Any]] = []
    for member_path in (*member_paths, *subset_paths):
        prim = stage.GetPrimAtPath(member_path)
        if not prim or not prim.IsValid():
            raise TextureWorkflowRuntimeError(
                f"Texture proposal member is absent from the source: {member_path}"
            )
        bound_material = bound_surfaces.get(member_path)
        if bound_material not in material_paths:
            raise TextureWorkflowRuntimeError(
                "Texture proposal member binding differs from its claimed material: "
                f"{member_path} -> {bound_material or '<unbound>'}"
            )
        material_records.append(
            {"prim_path": member_path, "material_prim_path": bound_material}
        )
        uv_records.append(_inspect_uv_primvar(prim))
    uv_states = {record["status"] for record in uv_records}
    if uv_states == {"ready"}:
        uv_status: Literal["ready", "missing", "repair_required", "unsupported"] = (
            "ready"
        )
    elif "unsupported" in uv_states:
        uv_status = "unsupported"
    elif "repair_required" in uv_states:
        uv_status = "repair_required"
    else:
        uv_status = "missing"
    return (
        material_paths,
        member_paths,
        subset_paths,
        uv_status,
        {
            "uv_scope": "target_prims",
            "material_bindings": material_records,
            "primvars": uv_records,
        },
    )


def _explicit_source_scope(
    request: TextureWorkflowRequest,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the required explicit request scope before runtime admission."""

    explicit_materials = tuple(
        str(path) for path in request.metadata.get("explicit_material_paths") or ()
    )
    explicit_prims = tuple(
        str(path) for path in request.metadata.get("explicit_prim_paths") or ()
    )
    if not explicit_materials and not explicit_prims:
        raise TextureWorkflowRuntimeError(
            "Texture source inspection requires explicit material or prim scope "
            "in request metadata"
        )
    return explicit_materials, explicit_prims


def _require_exact_source_scope(
    request: TextureWorkflowRequest,
    *,
    planned_material_paths: tuple[str, ...],
    material_aliases: Mapping[str, str],
    planned_surface_paths: tuple[str, ...],
    bound_surfaces: Mapping[str, str],
) -> None:
    """Reject provider omissions as well as expansions against the source stage."""

    explicit_materials, explicit_prims = _explicit_source_scope(request)
    if explicit_materials:
        canonical_materials = {
            material_aliases.get(path, path) for path in explicit_materials
        }
        expected_surfaces = {
            path
            for path, material in bound_surfaces.items()
            if material_aliases.get(material, material) in canonical_materials
        }
        expected_materials = canonical_materials
    else:
        expected_surfaces = {
            path
            for path in bound_surfaces
            if any(
                path == root or path.startswith(f"{root}/") for root in explicit_prims
            )
        }
        expected_materials = {
            material_aliases.get(bound_surfaces[path], bound_surfaces[path])
            for path in expected_surfaces
        }
    if not expected_surfaces:
        raise TextureWorkflowRuntimeError(
            "Texture source inspection found no bound surfaces in the requested scope"
        )
    if set(planned_surface_paths) != expected_surfaces:
        missing = sorted(expected_surfaces - set(planned_surface_paths))
        unexpected = sorted(set(planned_surface_paths) - expected_surfaces)
        raise TextureWorkflowRuntimeError(
            "Texture proposal does not exactly cover source-bound surfaces; missing: "
            + ", ".join(missing or ("<none>",))
            + "; unexpected: "
            + ", ".join(unexpected or ("<none>",))
        )
    if set(planned_material_paths) != expected_materials:
        raise TextureWorkflowRuntimeError(
            "Texture proposal material targets differ from source-bound request scope"
        )


class _TextureValidationIdentity(BaseModel):
    """Exact inputs and candidate bytes covered by one validation result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_asset_path: str = Field(min_length=1)
    output_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_artifact_digests: dict[str, TextureArtifactDigest] = Field(min_length=1)
    unit_ids: tuple[str, ...] = Field(min_length=1)
    iteration: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_unit_artifact_scope(self) -> Self:
        if set(self.unit_artifact_digests) != set(self.unit_ids):
            raise ValueError(
                "validation artifact digests must exactly cover evaluated unit IDs"
            )
        for unit_id, digest in self.unit_artifact_digests.items():
            if digest.unit_id != unit_id:
                raise ValueError(
                    "validation artifact digest unit ID does not match its map key"
                )
        return self


class _TextureValidationResultManifest(BaseModel):
    """Atomic, typed result journal for crash-safe usd-cli validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.texture-validation-result.v1"] = (
        _TEXTURE_VALIDATION_RESULT_SCHEMA_VERSION
    )
    identity: _TextureValidationIdentity
    result: TextureValidationResult
    evidence_sha256_by_path: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_result_claims(self) -> Self:
        if self.result.iteration != self.identity.iteration:
            raise ValueError("validation result iteration does not match its identity")
        if self.result.evaluated_unit_ids != self.identity.unit_ids:
            raise ValueError("validation result unit IDs do not match its identity")
        if (
            Path(self.result.output_asset_path).expanduser().resolve()
            != Path(self.identity.output_asset_path).expanduser().resolve()
        ):
            raise ValueError(
                "validation result output path does not match its identity"
            )
        evidence_paths = {
            path
            for finding in self.result.findings
            for path in finding.evidence_artifact_paths
        }
        if set(self.evidence_sha256_by_path) != evidence_paths:
            raise ValueError(
                "validation evidence digests must exactly cover result evidence"
            )
        for digest in self.evidence_sha256_by_path.values():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "validation evidence digests must be lowercase SHA-256"
                )
        return self


class TextureSceneValidator(Protocol):
    """Scene render/VQA boundary used by both workflow launch modes."""

    def inspect(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_dir: Path,
    ) -> TextureInspectionResult:
        """Return source-bound material, UV, render, and capability evidence."""

    def validate(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        iteration: int,
        output_dir: Path,
    ) -> TextureValidationResult:
        """Validate exactly ``unit_ids`` and identify per-unit failures."""


class TextureVisualAssessment(BaseModel):
    """Typed visual judgment produced from live usd-cli renders."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TextureValidationStatus
    summary: str = Field(min_length=1)


class _TextureValidationAssessmentEvidence(BaseModel):
    """Typed, per-unit evidence record that backs one validation finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    assessment: TextureVisualAssessment
    focus_prim_paths: tuple[str, ...] = Field(min_length=1)
    member_prim_path_count: int = Field(ge=0)
    source_image_paths: tuple[str, ...] = Field(min_length=1)
    output_image_paths: tuple[str, ...] = Field(min_length=1)
    view_pair_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_view_pairs(self) -> Self:
        if len(self.source_image_paths) != len(self.output_image_paths):
            raise ValueError("assessment evidence requires paired image paths")
        if self.view_pair_count != len(self.source_image_paths):
            raise ValueError(
                "assessment evidence view count does not match image paths"
            )
        return self


class TextureVisualAssessor(Protocol):
    """Semantic VQA boundary over paired source/output usd-cli renders."""

    def assess(
        self,
        *,
        intent: str,
        unit_id: str,
        unit_context: Mapping[str, Any],
        source_image_paths: tuple[str, ...],
        output_image_paths: tuple[str, ...],
    ) -> TextureVisualAssessment:
        """Judge one selected unit against the requested appearance."""


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        index = 0
        while index < len(text):
            start = text.find("{", index)
            if start < 0:
                break
            try:
                candidate, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                index = start + 1
                continue
            if isinstance(candidate, dict):
                candidates.append(candidate)
            index = max(end, start + 1)
        if len(candidates) != 1:
            raise ValueError(
                "Texture VQA response must contain exactly one JSON object"
            )
        payload = candidates[0]
    if not isinstance(payload, dict):
        raise ValueError("Texture VQA response must be a JSON object")
    return payload


class VlmTextureVisualAssessor:
    """Strict JSON VQA adapter for an initialized vision-language model."""

    def __init__(self, vlm: Any) -> None:
        self._vlm = vlm

    def assess(
        self,
        *,
        intent: str,
        unit_id: str,
        unit_context: Mapping[str, Any],
        source_image_paths: tuple[str, ...],
        output_image_paths: tuple[str, ...],
    ) -> TextureVisualAssessment:
        if len(source_image_paths) != len(output_image_paths):
            raise ValueError("Texture VQA requires paired source/output views")
        labels = [
            f"Images {index * 2 + 1} and {index * 2 + 2} are source/output "
            f"view pair {index + 1}."
            for index in range(len(source_image_paths))
        ]
        prompt = "\n".join(
            [
                f"Texture request: {intent}",
                f"Selected unit: {unit_id}",
                "Selected-unit context: "
                + json.dumps(dict(unit_context), sort_keys=True),
                *labels,
                (
                    "Decide whether the output visibly satisfies the requested "
                    "appearance for this selected unit. Fail visible artifacts, "
                    "wrong appearance, missing change, or changes outside the "
                    "described target. Return only JSON with status ('pass' or "
                    "'fail') and a concise summary."
                ),
            ]
        )
        images = [
            path
            for pair in zip(
                source_image_paths,
                output_image_paths,
                strict=True,
            )
            for path in pair
        ]
        response = self._vlm.generate(
            prompt=prompt,
            images=images,
            system_prompt=(
                "You are a strict visual quality assessor for textured 3D assets. "
                "Use only the paired evidence and return valid JSON."
            ),
            temperature=0.0,
            max_tokens=512,
        )
        return TextureVisualAssessment.model_validate(_extract_json_object(response))


class LiveUsdCliTextureValidator:
    """Render paired evidence in one workflow-owned usd-cli session."""

    def __init__(
        self,
        *,
        assessor: TextureVisualAssessor | None,
        validation_policy_id: str,
        directions: Sequence[str] = ("+x-y+z", "+z", "-z"),
        width: int = 1024,
        height: int = 768,
        render_quality: str = "inspection",
        max_snapshot_prims: int = 4096,
        max_focus_paths_per_unit: int = 4,
        max_view_pairs_per_unit: int = 12,
        max_source_asset_bytes: int | None = DEFAULT_MAX_TEXTURE_SOURCE_ASSET_BYTES,
    ) -> None:
        if (
            not isinstance(validation_policy_id, str)
            or not validation_policy_id.strip()
            or validation_policy_id != validation_policy_id.strip()
        ):
            raise ValueError(
                "validation_policy_id must be a non-empty stable identifier "
                "without surrounding whitespace"
            )
        if not directions:
            raise ValueError("Live usd-cli validation requires at least one view")
        if max_snapshot_prims < 1:
            raise ValueError("max_snapshot_prims must be positive")
        if max_focus_paths_per_unit < 1:
            raise ValueError(
                "Live usd-cli validation requires at least one focus path per unit"
            )
        if max_view_pairs_per_unit < len(directions):
            raise ValueError(
                "Live usd-cli validation requires room for every direction "
                "within max_view_pairs_per_unit"
            )
        if max_source_asset_bytes is not None and max_source_asset_bytes < 1:
            raise ValueError(
                "max_source_asset_bytes must be a positive integer or None"
            )
        self.assessor = assessor
        self.validation_policy_id = validation_policy_id
        self.directions = tuple(directions)
        self.width = width
        self.height = height
        self.render_quality = render_quality
        self.max_snapshot_prims = max_snapshot_prims
        self.max_focus_paths_per_unit = max_focus_paths_per_unit
        self.max_view_pairs_per_unit = max_view_pairs_per_unit
        self.max_source_asset_bytes = max_source_asset_bytes
        self._package_route_pin: WorkflowUsdCliPackageRoutePin | None = None

    def preflight_package_route(self) -> WorkflowUsdCliPackageRoutePin:
        """Pin one trusted usd-cli route before Texture preparation mutates state."""

        if self._package_route_pin is None:
            self._package_route_pin = WorkflowUsdCliSession.preflight_package_route()
        else:
            WorkflowUsdCliSession.verify_package_route(self._package_route_pin)
        return self._package_route_pin

    def collect_candidate_evidence(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        output_dir: Path,
        reference_artifacts: tuple[Any, ...],
        provided_images: tuple[Any, ...] = (),
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
        """Collect deterministic static and paired OVRTX evidence without VQA."""

        from content_agent_workflows.texture.capabilities import (
            TextureUnitRenderEvidence,
        )
        from content_agent_workflows.texture.embedded_decision import bind_file

        if set(unit_ids) - set(unit_artifacts):
            raise ValueError("Texture evidence requires an artifact for every unit")
        evidence_dir = output_dir / "candidate_evidence"
        source_asset_path = self._local_source_asset(
            request,
            evidence_dir / "source_asset",
        )
        invariant_report = validate_texture_scope_invariants(
            source_asset_path=source_asset_path,
            output_asset_path=output_asset_path,
            plan=plan,
        )
        invariant_path = atomic_write_json(
            evidence_dir / "scope_invariants.json",
            invariant_report,
        )
        self._reject_partial_scope_invariant_failure(
            invariant_report,
            plan=plan,
            unit_ids=unit_ids,
        )
        unit_focus_paths = {
            unit_id: self._unit_focus_paths(plan, unit_id) for unit_id in unit_ids
        }
        session = self._create_capture_session(
            owner_root=evidence_dir,
            scene_paths=(source_asset_path, output_asset_path),
        )
        try:
            (
                source_images_by_unit,
                source_shared_artifacts,
                source_artifacts_by_unit,
            ) = self._capture_scene(
                scene_path=source_asset_path,
                evidence_dir=evidence_dir / "source",
                unit_focus_paths=unit_focus_paths,
                session=session,
            )
            (
                candidate_images_by_unit,
                candidate_shared_artifacts,
                candidate_artifacts_by_unit,
            ) = self._capture_scene(
                scene_path=output_asset_path,
                evidence_dir=evidence_dir / "candidate",
                unit_focus_paths=unit_focus_paths,
                session=session,
            )
        finally:
            session.close()
        manifest_path = atomic_write_json(
            evidence_dir / "candidate_evidence_manifest.json",
            {
                "schema_version": (
                    "content-agent-workflows.texture-candidate-render-evidence.v1"
                ),
                "source_asset": bind_file(source_asset_path).model_dump(mode="json"),
                "candidate_asset": bind_file(output_asset_path).model_dump(mode="json"),
                "unit_ids": list(unit_ids),
                "unit_artifacts": {
                    unit_id: unit_artifacts[unit_id].model_dump(mode="json")
                    for unit_id in unit_ids
                },
                "reference_artifacts": [
                    item.model_dump(mode="json") for item in reference_artifacts
                ],
                "provided_images": [
                    item.model_dump(mode="json") for item in provided_images
                ],
                "renderer": {
                    "provider": "usd-cli-ovrtx",
                    "directions": list(self.directions),
                    "width": self.width,
                    "height": self.height,
                    "render_quality": self.render_quality,
                },
            },
        )
        unit_evidence = tuple(
            TextureUnitRenderEvidence(
                unit_id=unit_id,
                source_images=tuple(
                    bind_file(path) for path in source_images_by_unit[unit_id]
                ),
                candidate_images=tuple(
                    bind_file(path) for path in candidate_images_by_unit[unit_id]
                ),
            )
            for unit_id in unit_ids
        )
        static_paths = tuple(
            dict.fromkeys(
                (
                    str(invariant_path),
                    str(manifest_path),
                    *source_shared_artifacts,
                    *candidate_shared_artifacts,
                    *(
                        path
                        for unit_id in unit_ids
                        for path in source_artifacts_by_unit[unit_id]
                        if path not in source_images_by_unit[unit_id]
                    ),
                    *(
                        path
                        for unit_id in unit_ids
                        for path in candidate_artifacts_by_unit[unit_id]
                        if path not in candidate_images_by_unit[unit_id]
                    ),
                )
            )
        )
        return (
            unit_evidence,
            tuple(bind_file(path) for path in static_paths),
            {
                "provider": "usd-cli-ovrtx",
                "renderer": "ovrtx",
                "current_run": True,
                "directions": list(self.directions),
                "width": self.width,
                "height": self.height,
                "render_quality": self.render_quality,
                "semantic_assessment": "not_evaluated",
                "per_image_provenance": "bound-response-camera-v1",
            },
        )

    def _render_focused_view(
        self,
        *,
        session: WorkflowUsdCliSession,
        output_dir: Path,
        name: str,
        direction: str,
        focus_path: str,
        scene_path: Path,
        up_axis_y: bool,
    ) -> dict[str, Any]:
        azimuth, elevation = direction_angles(
            direction,
            up_axis_y=up_axis_y,
        )
        session.run_json(
            [
                "camera",
                "orbit",
                focus_path,
                "--az",
                str(azimuth),
                "--el",
                str(elevation),
            ]
        )
        image_path = output_dir / f"{name}.png"
        response_path = output_dir / f"{name}_response.json"
        camera_path = output_dir / f"{name}_camera.json"
        response = session.run_json(
            [
                "render",
                "--res",
                f"{self.width}x{self.height}",
                "--mode",
                "quality" if self.render_quality == "final" else "fast",
                "--output",
                str(image_path.resolve()),
            ]
        )
        summary = response.get("summary")
        if not isinstance(summary, dict) or summary.get("backend") not in {
            "ovrtx",
            "remote",
        }:
            raise RuntimeError("usd-cli Texture render did not confirm OVRTX")
        if not image_path.is_file():
            raise RuntimeError(
                f"usd-cli did not produce Texture evidence: {image_path}"
            )
        normalized_response, camera = self._scoreable_render_provenance(
            response=response,
            summary=summary,
            image_path=image_path,
            scene_path=scene_path,
            focus_path=focus_path,
            direction=direction,
        )
        atomic_write_json(response_path, normalized_response)
        atomic_write_json(camera_path, camera)
        return {
            "name": name,
            "direction": direction,
            "focus_path": focus_path,
            "width": self.width,
            "height": self.height,
            "render_quality": self.render_quality,
            "image_path": str(image_path.resolve()),
            "response_path": str(response_path.resolve()),
            "camera_json_path": str(camera_path.resolve()),
        }

    def _scoreable_render_provenance(
        self,
        *,
        response: Mapping[str, Any],
        summary: Mapping[str, Any],
        image_path: Path,
        scene_path: str | Path,
        focus_path: str,
        direction: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Project a raw usd-cli render envelope into benchmark camera evidence."""

        data = response.get("data")
        results = data.get("results") if isinstance(data, Mapping) else None
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError(
                "usd-cli Texture render did not report exactly one render result"
            )
        result = results[0]
        if not isinstance(result, Mapping):
            raise RuntimeError("usd-cli Texture render result is malformed")
        result_path = result.get("path")
        if not isinstance(result_path, str) or not result_path:
            raise RuntimeError(
                "usd-cli Texture render result is missing its image path"
            )
        if Path(result_path).expanduser().resolve() != image_path.resolve():
            raise RuntimeError(
                "usd-cli Texture render result does not bind the requested evidence image"
            )

        camera_path = summary.get("camera")
        result_camera = result.get("camera")
        if (
            not isinstance(camera_path, str)
            or not camera_path.startswith("/")
            or result_camera != camera_path
        ):
            raise RuntimeError(
                "usd-cli Texture render did not report a matching absolute camera path"
            )
        transform = summary.get("camera_world_transform")
        if not self._is_finite_camera_transform(transform):
            raise RuntimeError(
                "usd-cli Texture render did not report a finite camera world transform"
            )
        try:
            render_metadata = validated_ovrtx_render_metadata(response)
        except RuntimeError as exc:
            raise RuntimeError(
                "usd-cli Texture render did not report complete OVRTX provenance"
            ) from exc
        render_mode = render_metadata["ovrtx_render_mode"]
        sensor_updates = render_metadata["ovrtx_num_sensor_updates"]
        active_aov = render_metadata["active_aov"]
        if active_aov != "LdrColor":
            raise RuntimeError("usd-cli Texture render did not report the beauty AOV")

        raw_camera_state = summary.get("camera_state")
        if raw_camera_state is not None and not isinstance(raw_camera_state, Mapping):
            raise RuntimeError("usd-cli Texture render camera state is malformed")
        camera_state = dict(raw_camera_state or {})
        reported_focus = camera_state.get("last_framed_prim_path")
        if reported_focus is not None and reported_focus != focus_path:
            raise RuntimeError(
                "usd-cli Texture render camera state does not match the focused prim"
            )
        camera_state["last_framed_prim_path"] = focus_path

        resolved_scene = Path(scene_path).expanduser().resolve()
        if resolved_scene.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
            raise RuntimeError("usd-cli Texture render scene is not a USD asset")
        normalized_response = {
            "renderer": "ovrtx",
            "backend": render_metadata["backend"],
            "transport": render_metadata["transport"],
            "status": "success",
            "preview_scene_path": str(resolved_scene),
            "ovrtx_render_mode": render_mode,
            "ovrtx_num_sensor_updates": sensor_updates,
            "active_aov": active_aov,
            "render_quality": self.render_quality,
        }
        renderer_identity = render_metadata["renderer_identity"]
        if renderer_identity is not None:
            normalized_response["renderer_identity"] = renderer_identity
        camera = {
            "camera_path": camera_path,
            "camera_state": camera_state,
            "camera_world_transform": transform,
            "image_width": self.width,
            "image_height": self.height,
            "direction": direction,
            "ovrtx_render_mode": render_mode,
            "ovrtx_num_sensor_updates": sensor_updates,
            "active_aov": active_aov,
            "render_quality": self.render_quality,
        }
        return normalized_response, camera

    @staticmethod
    def _is_finite_camera_transform(value: object) -> bool:
        return bool(
            isinstance(value, list)
            and len(value) == 4
            and all(
                isinstance(row, list)
                and len(row) == 4
                and all(
                    isinstance(component, int | float)
                    and not isinstance(component, bool)
                    and math.isfinite(component)
                    for component in row
                )
                for row in value
            )
        )

    def _capture_scene(
        self,
        *,
        scene_path: str,
        evidence_dir: Path,
        unit_focus_paths: Mapping[str, tuple[str, ...]],
        session: WorkflowUsdCliSession | None = None,
    ) -> tuple[
        dict[str, tuple[str, ...]],
        tuple[str, ...],
        dict[str, tuple[str, ...]],
    ]:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        resolved_scene = Path(scene_path).expanduser().resolve(strict=True)
        owns_session = session is None
        if session is None:
            session = self._create_capture_session(
                owner_root=evidence_dir,
                scene_paths=(resolved_scene,),
            )
        try:
            session.require_ovrtx(evidence_dir / "ovrtx_probe")
            # Camera and render commands author scratch state in the live layer.
            # A preserve disposition intentionally renders the same source path
            # twice, so the second open must discard that session-only state.
            session.open(resolved_scene, force_reload=True)
            up_axis_y = stage_up_axis_is_y(resolved_scene)
            stage_info = session.run_json(["info"])
            snapshot_prim_count = self._snapshot_prim_count(stage_info)
            if snapshot_prim_count > self.max_snapshot_prims:
                raise RuntimeError(
                    "usd-cli Texture scene exceeds max_snapshot_prims "
                    f"({snapshot_prim_count} > {self.max_snapshot_prims})"
                )
            snapshot = session.run_json(
                ["snapshot", "--materials", "--bounds", "--properties"]
            )
            snapshot_path = atomic_write_json(
                evidence_dir / "scene_snapshot.json",
                snapshot,
            )
            image_paths_by_unit: dict[str, tuple[str, ...]] = {}
            artifact_paths_by_unit: dict[str, tuple[str, ...]] = {}
            for unit_id, focus_paths in unit_focus_paths.items():
                unit_dir = evidence_dir / unit_id
                unit_dir.mkdir(parents=True, exist_ok=True)
                image_paths: list[str] = []
                artifact_paths: list[str] = []
                for focus_index, focus_path in enumerate(focus_paths):
                    for direction_index, direction in enumerate(self.directions):
                        record = self._render_focused_view(
                            session=session,
                            output_dir=unit_dir,
                            name=(
                                f"member-{focus_index:03d}-view-{direction_index:02d}"
                            ),
                            direction=direction,
                            focus_path=focus_path,
                            scene_path=resolved_scene,
                            up_axis_y=up_axis_y,
                        )
                        image_paths.append(str(Path(record["image_path"]).resolve()))
                        artifact_paths.extend(
                            str(Path(path).resolve())
                            for path in (
                                record["image_path"],
                                record["response_path"],
                                record.get("camera_json_path"),
                            )
                            if path
                        )
                image_paths_by_unit[unit_id] = tuple(image_paths)
                artifact_paths_by_unit[unit_id] = tuple(dict.fromkeys(artifact_paths))
            return (
                image_paths_by_unit,
                (str(snapshot_path),),
                artifact_paths_by_unit,
            )
        finally:
            if owns_session:
                session.close()

    @staticmethod
    def _snapshot_prim_count(response: Mapping[str, Any]) -> int:
        """Read the bounded scene-size fact from either supported usd-cli summary."""

        summary = response.get("summary")
        if not isinstance(summary, Mapping):
            raise RuntimeError("usd-cli Texture scene info is missing its summary")
        count = summary.get("prims", summary.get("prim_count"))
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RuntimeError("usd-cli Texture scene info is missing its prim count")
        return count

    def _create_capture_session(
        self,
        *,
        owner_root: Path,
        scene_paths: tuple[str | Path, ...],
    ) -> WorkflowUsdCliSession:
        """Create the one USD sidecar session owned by this evidence capture run."""

        resolved_owner_root = owner_root.resolve()
        resolved_scenes = tuple(
            Path(scene_path).expanduser().resolve(strict=True)
            for scene_path in scene_paths
        )
        return WorkflowUsdCliSession.create(
            owner_root=resolved_owner_root,
            project_dir=resolved_owner_root,
            identity=(
                f"{resolved_owner_root}:"
                + ":".join(str(scene_path) for scene_path in resolved_scenes)
            ),
            workflow="texture-validation",
            input_roots=resolved_scenes,
            package_route=self.preflight_package_route(),
        )

    @staticmethod
    def _unit_context(
        plan: TexturePlanDocument,
        unit_id: str,
    ) -> dict[str, Any]:
        unit = next(
            (item for item in plan.selected_units if item.unit_id == unit_id),
            None,
        )
        if unit is None:
            raise ValueError(f"Texture VQA unit is outside the plan: {unit_id}")
        return unit.model_dump(mode="json")

    @classmethod
    def _unit_member_paths(
        cls,
        plan: TexturePlanDocument,
        unit_id: str,
    ) -> tuple[str, ...]:
        context = cls._unit_context(plan, unit_id)
        paths: list[str] = []
        for field_name in ("member_prim_paths", "member_subset_paths"):
            raw_paths = context.get(field_name)
            if isinstance(raw_paths, list | tuple):
                paths.extend(path for item in raw_paths if (path := str(item).strip()))
        member_paths = tuple(dict.fromkeys(paths))
        if not member_paths:
            raise ValueError(
                "Texture VQA requires member prim paths for focused evidence: "
                f"{unit_id}"
            )
        return member_paths

    @staticmethod
    def _require_absolute_non_root_prim_path(
        raw_path: Any,
        *,
        field_name: str,
        unit_id: str,
    ) -> str:
        path_text = str(raw_path).strip()
        path = Sdf.Path(path_text)
        if (
            not path.IsAbsolutePath()
            or not path.IsPrimPath()
            or path == Sdf.Path.absoluteRootPath
        ):
            raise ValueError(
                f"Texture VQA {field_name} must resolve to an absolute non-root "
                f"prim path for {unit_id}: {path_text!r}"
            )
        return str(path)

    def _unit_focus_paths(
        self,
        plan: TexturePlanDocument,
        unit_id: str,
    ) -> tuple[str, ...]:
        context = self._unit_context(plan, unit_id)
        focus_paths: list[str] = []
        raw_prim_paths = context.get("member_prim_paths")
        if isinstance(raw_prim_paths, list | tuple):
            focus_paths.extend(
                self._require_absolute_non_root_prim_path(
                    path,
                    field_name="member_prim_paths",
                    unit_id=unit_id,
                )
                for path in raw_prim_paths
            )
        raw_subset_paths = context.get("member_subset_paths")
        if isinstance(raw_subset_paths, list | tuple):
            focus_paths.extend(
                self._require_absolute_non_root_prim_path(
                    Sdf.Path(str(path).strip()).GetParentPath(),
                    field_name="member_subset_paths parent",
                    unit_id=unit_id,
                )
                for path in raw_subset_paths
            )
        focus_paths = list(dict.fromkeys(path for path in focus_paths if path))
        if not focus_paths:
            raise ValueError(
                "Texture VQA requires member prim or subset paths for focused "
                f"evidence: {unit_id}"
            )
        path_cap = min(
            self.max_focus_paths_per_unit,
            self.max_view_pairs_per_unit // len(self.directions),
        )
        return tuple(focus_paths[:path_cap])

    def _local_source_asset(
        self,
        request: TextureWorkflowRequest,
        staging_dir: Path,
    ) -> str:
        source_asset = request.source_asset
        if not source_asset.startswith("s3://"):
            return source_asset

        expected_sha256 = texture_source_identity_digest(request)
        bucket_and_key = source_asset.removeprefix("s3://")
        _bucket, separator, key = bucket_and_key.partition("/")
        file_name = Path(key).name if separator else ""
        if not file_name or file_name in {".", ".."}:
            raise ValueError(
                "Texture usd-cli validation requires an S3 object key with a file name"
            )
        staging_dir = _ensure_private_texture_directory(staging_dir)
        staged_path = staging_dir / file_name
        if staged_path.is_symlink():
            raise TextureWorkflowRuntimeError(
                f"Texture S3 source staging target is unsafe: {staged_path}"
            )
        if staged_path.is_file():
            if file_sha256(staged_path) != expected_sha256:
                raise TextureWorkflowRuntimeError(
                    "Existing staged S3 source bytes differ from the frozen digest"
                )
            metadata = staged_path.lstat()
            effective_uid = getattr(os, "geteuid", lambda: metadata.st_uid)()
            if metadata.st_uid != effective_uid:
                raise TextureWorkflowRuntimeError(
                    f"Texture S3 source staging target is not owner-controlled: {staged_path}"
                )
            staged_path.chmod(0o600)
            return str(staged_path.resolve())
        if staged_path.exists():
            raise TextureWorkflowRuntimeError(
                f"Texture S3 source staging target is unsafe: {staged_path}"
            )
        downloaded_path = Path(
            download_file_from_s3(
                source_asset,
                staged_path,
                profile_name=WU_S3_PROFILE or None,
                region_name=WU_S3_REGION or None,
                max_bytes=self.max_source_asset_bytes,
            )
        )
        if not downloaded_path.is_file():
            raise RuntimeError(
                "Texture usd-cli validation did not materialize the S3 "
                f"source asset: {source_asset}"
            )
        if file_sha256(downloaded_path) != expected_sha256:
            raise TextureWorkflowRuntimeError(
                "Downloaded S3 source bytes do not match metadata.source_asset_sha256"
            )
        downloaded_metadata = downloaded_path.lstat()
        effective_uid = getattr(
            os,
            "geteuid",
            lambda: downloaded_metadata.st_uid,
        )()
        if (
            stat.S_ISLNK(downloaded_metadata.st_mode)
            or not stat.S_ISREG(downloaded_metadata.st_mode)
            or downloaded_metadata.st_uid != effective_uid
        ):
            raise TextureWorkflowRuntimeError(
                f"Downloaded S3 source is not an owner-controlled regular file: {downloaded_path}"
            )
        downloaded_path.chmod(0o600)
        return str(downloaded_path.resolve())

    def _validation_policy_digest(self) -> str:
        policy = {
            "schema_version": _TEXTURE_VALIDATION_POLICY_DIGEST_SCHEMA,
            "validation_policy_id": self.validation_policy_id,
            "directions": self.directions,
            "width": self.width,
            "height": self.height,
            "render_quality": self.render_quality,
            "max_snapshot_prims": self.max_snapshot_prims,
            "max_focus_paths_per_unit": self.max_focus_paths_per_unit,
            "max_view_pairs_per_unit": self.max_view_pairs_per_unit,
        }
        encoded = json.dumps(
            policy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validation_identity(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        iteration: int,
    ) -> _TextureValidationIdentity:
        resolved_output_path = Path(output_asset_path).expanduser().resolve()
        selected_artifacts: dict[str, TextureUnitArtifact] = {}
        for unit_id in unit_ids:
            artifact = unit_artifacts[unit_id]
            if artifact.unit_id != unit_id:
                raise ValueError(
                    "usd-cli validation artifact unit ID does not match "
                    f"mapping key {unit_id}"
                )
            selected_artifacts[unit_id] = artifact
        return _TextureValidationIdentity(
            request_digest=texture_request_digest(request),
            plan_digest=texture_plan_digest(plan),
            source_asset_sha256=texture_source_identity_digest(request, plan=plan),
            output_asset_path=str(resolved_output_path),
            output_asset_sha256=file_sha256(resolved_output_path),
            validation_policy_digest=self._validation_policy_digest(),
            unit_artifact_digests=collect_artifact_digests(selected_artifacts),
            unit_ids=unit_ids,
            iteration=iteration,
        )

    def inspect(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_dir: Path,
    ) -> TextureInspectionResult:
        """Capture provider-neutral source evidence before outer planning."""

        from content_agent_workflows.common.domain_execution import (
            ExecutionArtifactBinding,
        )

        _explicit_source_scope(request)
        # Admission is deliberately side-effect free and precedes source
        # localization or inspection-directory creation. Session creation repeats
        # the check immediately before use to fail closed on route drift.
        self.preflight_package_route()
        inspection_dir = _ensure_private_texture_directory(
            output_dir / "usd_cli_inspection"
        )
        source_asset_path = (
            Path(self._local_source_asset(request, inspection_dir / "source_asset"))
            .expanduser()
            .resolve()
        )
        source = ExecutionArtifactBinding(
            path=str(source_asset_path),
            sha256=file_sha256(source_asset_path),
            size_bytes=source_asset_path.stat().st_size,
        )
        context = request.execution_context
        if (
            context is not None
            and context.embedded_stage is not None
            and source != context.embedded_stage.input_asset
        ):
            raise TextureWorkflowRuntimeError(
                "Texture inspection source differs from the embedded stage input"
            )
        stage = Usd.Stage.Open(str(source_asset_path))
        if stage is None:
            raise TextureWorkflowRuntimeError(
                "Texture inspection could not open the exact source USD"
            )
        bound_surfaces = _source_bound_surfaces(stage)
        inspected_units: dict[
            str,
            tuple[
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                Literal["ready", "missing", "repair_required", "unsupported"],
                dict[str, Any],
            ],
        ] = {}
        for unit_id in plan.selected_unit_ids:
            inspected_units[unit_id] = _inspect_source_unit(
                stage,
                unit_id=unit_id,
                unit_context=self._unit_context(plan, unit_id),
                bound_surfaces=bound_surfaces,
            )
        _require_exact_source_scope(
            request,
            planned_material_paths=tuple(
                dict.fromkeys(
                    material_path
                    for values in inspected_units.values()
                    for material_path in values[0]
                )
            ),
            material_aliases={
                str(alias): str(material_path)
                for unit_id in plan.selected_unit_ids
                for material_path in self._unit_context(plan, unit_id).get(
                    "material_prim_paths", ()
                )
                for alias in (
                    material_path,
                    *self._unit_context(plan, unit_id).get("material_alias_paths", ()),
                )
            },
            planned_surface_paths=tuple(
                dict.fromkeys(
                    surface_path
                    for values in inspected_units.values()
                    for surface_path in (*values[1], *values[2])
                )
            ),
            bound_surfaces=bound_surfaces,
        )
        unit_focus_paths = {
            unit_id: self._unit_focus_paths(plan, unit_id)
            for unit_id in plan.selected_unit_ids
        }
        inspection_source_dir = _ensure_private_texture_directory(
            inspection_dir / "source"
        )
        for unit_id in plan.selected_unit_ids:
            _ensure_private_texture_directory(inspection_source_dir / unit_id)
        (
            source_images_by_unit,
            shared_artifacts,
            source_artifacts_by_unit,
        ) = self._capture_scene(
            scene_path=str(source_asset_path),
            evidence_dir=inspection_source_dir,
            unit_focus_paths=unit_focus_paths,
        )
        units: list[TextureInspectionUnit] = []
        for unit_id in plan.selected_unit_ids:
            unit_context = self._unit_context(plan, unit_id)
            (
                material_prim_paths,
                member_prim_paths,
                member_subset_paths,
                uv_status,
                uv_facts,
            ) = inspected_units[unit_id]
            uv_facts = {
                **uv_facts,
                "detail_policy": unit_context.get("detail_policy", "surface_only"),
                "focus_prim_paths": list(unit_focus_paths[unit_id]),
            }
            units.append(
                TextureInspectionUnit(
                    unit_id=unit_id,
                    material_prim_paths=material_prim_paths,
                    material_alias_paths=tuple(
                        str(path)
                        for path in unit_context.get("material_alias_paths") or ()
                    ),
                    member_prim_paths=member_prim_paths,
                    member_subset_paths=member_subset_paths,
                    uv_status=uv_status,
                    uv_facts=uv_facts,
                    proposed_generator_inputs=_proposed_generator_inputs(
                        request,
                        plan,
                        unit_context,
                    ),
                )
            )
        facts_path = atomic_write_json(
            inspection_dir / "inspection_facts.json",
            {
                "schema_version": "content-agent-workflows.texture-inspection-facts.v1",
                "source": source.model_dump(mode="json"),
                "units": [unit.model_dump(mode="json") for unit in units],
                "capability_constraints": [
                    "surface texturing only; label and legend authoring excluded",
                    "target scope is immutable after outer acceptance",
                    "visual assessment is critique and not semantic authority",
                ],
            },
        )

        def binding(raw_path: str | Path) -> ExecutionArtifactBinding:
            path = Path(raw_path).expanduser().resolve()
            return ExecutionArtifactBinding(
                path=str(path),
                sha256=file_sha256(path),
                size_bytes=path.stat().st_size,
            )

        before_render_paths = tuple(
            dict.fromkeys(
                path
                for unit_id in plan.selected_unit_ids
                for path in source_artifacts_by_unit[unit_id]
            )
        )
        if not before_render_paths or any(
            not source_images_by_unit[unit_id] for unit_id in plan.selected_unit_ids
        ):
            raise TextureWorkflowRuntimeError(
                "Texture inspection requires a fresh render for every target"
            )
        return TextureInspectionResult(
            source=source,
            proposal_plan_digest=texture_plan_digest(plan),
            units=tuple(units),
            before_render_artifacts=tuple(
                binding(path) for path in before_render_paths
            ),
            inspection_artifacts=(
                binding(facts_path),
                *(binding(path) for path in shared_artifacts),
            ),
            reference_artifacts=request.reference_artifacts,
            capability_constraints=(
                "surface texturing only; label and legend authoring excluded",
                "target scope is immutable after outer acceptance",
                "visual assessment is critique and not semantic authority",
            ),
            renderer_metadata={
                "interface": "provider-neutral-texture-renderer.v1",
                "directions": list(self.directions),
                "width": self.width,
                "height": self.height,
                "render_quality": self.render_quality,
            },
            tool_metadata={
                "inspection_adapter": "live-usd-cli-texture-validator",
                "validation_policy_id": self.validation_policy_id,
            },
        )

    @staticmethod
    def _load_validation_manifest(
        path: Path,
    ) -> _TextureValidationResultManifest:
        try:
            return _TextureValidationResultManifest.model_validate(load_json(path))
        except (OSError, ValueError, ValidationError) as exc:
            raise TextureWorkflowRuntimeError(
                f"Invalid Texture validation result manifest at {path}: {exc}"
            ) from exc

    @staticmethod
    def _load_scope_invariant_report(path: Path) -> TextureScopeInvariantReport:
        try:
            return TextureScopeInvariantReport.model_validate(load_json(path))
        except (OSError, ValueError, ValidationError) as exc:
            raise TextureWorkflowRuntimeError(
                f"Invalid Texture scope invariant report at {path}: {exc}"
            ) from exc

    @staticmethod
    def _reject_partial_scope_invariant_failure(
        report: TextureScopeInvariantReport,
        *,
        plan: TexturePlanDocument,
        unit_ids: tuple[str, ...],
    ) -> None:
        if report.passed or set(unit_ids) == set(plan.selected_unit_ids):
            return
        raise TextureWorkflowRuntimeError(
            "Stage-wide Texture scope invariants failed while usd-cli was "
            "validating only a partial selected-unit scope. The violation cannot "
            "be safely attributed to the pending units, so the workflow cannot "
            "continue or finalize this output."
        )

    @staticmethod
    def _verify_manifest_evidence(
        manifest: _TextureValidationResultManifest,
        *,
        path: Path,
    ) -> None:
        iteration_dir = path.parent.resolve()
        assessment_paths = {
            (iteration_dir / f"{finding.unit_id}_assessment.json").resolve()
            for finding in manifest.result.findings
        }
        resolved_evidence_paths: dict[Path, str] = {}
        for raw_path, expected_digest in manifest.evidence_sha256_by_path.items():
            evidence_path = Path(raw_path).expanduser().resolve()
            if evidence_path in resolved_evidence_paths:
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation evidence aliases the same path for "
                    f"{path}: {raw_path!r} and "
                    f"{resolved_evidence_paths[evidence_path]!r}"
                )
            resolved_evidence_paths[evidence_path] = raw_path
            if not evidence_path.is_file():
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation evidence is missing for "
                    f"{path}: {evidence_path}"
                )
            if evidence_path in assessment_paths:
                try:
                    assessment_size = evidence_path.stat().st_size
                except OSError as exc:
                    raise TextureWorkflowRuntimeError(
                        "Could not inspect cached Texture validation assessment "
                        f"evidence at {evidence_path}: {exc}"
                    ) from exc
                if assessment_size > _MAX_CACHED_TEXTURE_ASSESSMENT_BYTES:
                    raise TextureWorkflowRuntimeError(
                        "Cached Texture validation assessment evidence exceeds "
                        f"the {_MAX_CACHED_TEXTURE_ASSESSMENT_BYTES}-byte limit "
                        f"at {evidence_path}"
                    )
            if file_sha256(evidence_path) != expected_digest:
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation evidence bytes changed for "
                    f"{path}: {evidence_path}"
                )

        for finding in manifest.result.findings:
            assessment_path = (
                iteration_dir / f"{finding.unit_id}_assessment.json"
            ).resolve()
            finding_evidence_paths = tuple(
                Path(raw_path).expanduser().resolve()
                for raw_path in finding.evidence_artifact_paths
            )
            if finding_evidence_paths.count(assessment_path) != 1:
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation finding does not reference exactly "
                    "one canonical assessment evidence file for "
                    f"{finding.unit_id} in {path}"
                )
            raw_assessment_path = resolved_evidence_paths.get(assessment_path)
            if raw_assessment_path is None:
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation assessment is not covered by the "
                    f"evidence digest manifest for {finding.unit_id} in {path}"
                )
            try:
                with assessment_path.open("rb") as assessment_stream:
                    assessment_bytes = assessment_stream.read(
                        _MAX_CACHED_TEXTURE_ASSESSMENT_BYTES + 1
                    )
            except OSError as exc:
                raise TextureWorkflowRuntimeError(
                    "Could not read cached Texture validation assessment evidence "
                    f"at {assessment_path}: {exc}"
                ) from exc
            if len(assessment_bytes) > _MAX_CACHED_TEXTURE_ASSESSMENT_BYTES:
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation assessment evidence exceeds "
                    f"the {_MAX_CACHED_TEXTURE_ASSESSMENT_BYTES}-byte limit "
                    f"at {assessment_path}"
                )
            expected_assessment_digest = manifest.evidence_sha256_by_path[
                raw_assessment_path
            ]
            if (
                hashlib.sha256(assessment_bytes).hexdigest()
                != expected_assessment_digest
            ):
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation evidence bytes changed for "
                    f"{path}: {assessment_path}"
                )
            try:
                assessment_evidence = (
                    _TextureValidationAssessmentEvidence.model_validate_json(
                        assessment_bytes
                    )
                )
            except (ValueError, ValidationError) as exc:
                raise TextureWorkflowRuntimeError(
                    "Invalid cached Texture validation assessment evidence at "
                    f"{assessment_path}: {exc}"
                ) from exc
            if assessment_evidence.unit_id != finding.unit_id:
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation assessment unit ID does not match "
                    f"finding {finding.unit_id} in {path}"
                )
            if (
                assessment_evidence.assessment.status != finding.status
                or assessment_evidence.assessment.summary != finding.summary
            ):
                raise TextureWorkflowRuntimeError(
                    "Cached Texture validation finding does not match hashed "
                    f"per-unit assessment evidence for {finding.unit_id} in {path}"
                )

    def validate(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        iteration: int,
        output_dir: Path,
    ) -> TextureValidationResult:
        if set(unit_ids) - set(unit_artifacts):
            raise ValueError("usd-cli validation requires an artifact for every unit")
        iteration_dir = output_dir / "usd_cli_validation" / f"iteration-{iteration}"
        manifest_path = iteration_dir / _VALIDATION_RESULT_MANIFEST_NAME
        validation_identity = self._validation_identity(
            request=request,
            plan=plan,
            output_asset_path=output_asset_path,
            unit_artifacts=unit_artifacts,
            unit_ids=unit_ids,
            iteration=iteration,
        )
        if manifest_path.is_file():
            manifest = self._load_validation_manifest(manifest_path)
            if manifest.identity == validation_identity:
                self._verify_manifest_evidence(manifest, path=manifest_path)
                self._reject_partial_scope_invariant_failure(
                    self._load_scope_invariant_report(
                        iteration_dir / "scope_invariants.json"
                    ),
                    plan=plan,
                    unit_ids=unit_ids,
                )
                return manifest.result

        source_asset_path = self._local_source_asset(
            request,
            iteration_dir / "source_asset",
        )
        unit_focus_paths = {
            unit_id: self._unit_focus_paths(plan, unit_id) for unit_id in unit_ids
        }
        unit_member_path_counts = {
            unit_id: len(self._unit_member_paths(plan, unit_id)) for unit_id in unit_ids
        }
        invariant_report = validate_texture_scope_invariants(
            source_asset_path=source_asset_path,
            output_asset_path=output_asset_path,
            plan=plan,
        )
        invariant_path = atomic_write_json(
            iteration_dir / "scope_invariants.json",
            invariant_report,
        )
        self._reject_partial_scope_invariant_failure(
            invariant_report,
            plan=plan,
            unit_ids=unit_ids,
        )
        session = self._create_capture_session(
            owner_root=iteration_dir,
            scene_paths=(source_asset_path, output_asset_path),
        )
        try:
            (
                source_images_by_unit,
                source_shared_artifacts,
                source_artifacts_by_unit,
            ) = self._capture_scene(
                scene_path=source_asset_path,
                evidence_dir=iteration_dir / "source",
                unit_focus_paths=unit_focus_paths,
                session=session,
            )
            (
                output_images_by_unit,
                output_shared_artifacts,
                output_artifacts_by_unit,
            ) = self._capture_scene(
                scene_path=output_asset_path,
                evidence_dir=iteration_dir / "output",
                unit_focus_paths=unit_focus_paths,
                session=session,
            )
        finally:
            session.close()
        shared_artifacts = (
            str(invariant_path),
            *source_shared_artifacts,
            *output_shared_artifacts,
        )

        findings: list[TextureValidationFinding] = []
        for unit_id in unit_ids:
            source_images = source_images_by_unit[unit_id]
            output_images = output_images_by_unit[unit_id]
            if invariant_report.passed:
                if self.assessor is None:
                    raise RuntimeError(
                        "Texture semantic validation requires an explicitly selected "
                        "visual assessor; use collect_candidate_evidence for "
                        "provider-free evidence"
                    )
                assessment = self.assessor.assess(
                    intent=request.intent,
                    unit_id=unit_id,
                    unit_context=self._unit_context(plan, unit_id),
                    source_image_paths=source_images,
                    output_image_paths=output_images,
                )
            else:
                assessment = TextureVisualAssessment(
                    status="fail",
                    summary=(
                        "Deterministic scope validation rejected changes outside "
                        "the selected material or unchanged-geometry contract."
                    ),
                )
            assessment_path = atomic_write_json(
                iteration_dir / f"{unit_id}_assessment.json",
                {
                    "unit_id": unit_id,
                    "intent": request.intent,
                    "assessment": assessment.model_dump(mode="json"),
                    "focus_prim_paths": unit_focus_paths[unit_id],
                    "member_prim_path_count": unit_member_path_counts[unit_id],
                    "source_image_paths": source_images,
                    "output_image_paths": output_images,
                    "view_pair_count": len(source_images),
                },
            )
            findings.append(
                TextureValidationFinding(
                    unit_id=unit_id,
                    status=assessment.status,
                    summary=assessment.summary,
                    evidence_artifact_paths=(
                        *shared_artifacts,
                        *source_artifacts_by_unit[unit_id],
                        *output_artifacts_by_unit[unit_id],
                        str(assessment_path),
                    ),
                )
            )

        result = TextureValidationResult(
            iteration=iteration,
            evaluated_unit_ids=unit_ids,
            findings=tuple(findings),
            output_asset_path=output_asset_path,
        )
        if file_sha256(source_asset_path) != validation_identity.source_asset_sha256:
            raise TextureWorkflowRuntimeError(
                "Texture validation source bytes changed before result checkpointing"
            )
        if (
            self._validation_identity(
                request=request,
                plan=plan,
                output_asset_path=output_asset_path,
                unit_artifacts=unit_artifacts,
                unit_ids=unit_ids,
                iteration=iteration,
            )
            != validation_identity
        ):
            raise TextureWorkflowRuntimeError(
                "Texture validation identity changed before result checkpointing"
            )
        evidence_sha256_by_path = collect_validation_evidence_digests((result,))
        atomic_write_json(
            manifest_path,
            _TextureValidationResultManifest(
                identity=validation_identity,
                result=result,
                evidence_sha256_by_path=evidence_sha256_by_path,
            ),
        )
        return result


class MockTextureValidationCall(BaseModel):
    """Recorded validation request for workflow assertions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration: int
    unit_ids: tuple[str, ...]


class MockTextureSceneValidator:
    """Mock usd-cli render/VQA adapter with a per-pass failure schedule."""

    def __init__(
        self,
        failure_schedule: Sequence[Sequence[str]] = (),
    ) -> None:
        self._failure_schedule = tuple(
            tuple(failed_ids) for failed_ids in failure_schedule
        )
        self.calls: list[MockTextureValidationCall] = []

    def inspect(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_dir: Path,
    ) -> TextureInspectionResult:
        """Write deterministic provider-neutral inspection fixtures."""

        from content_agent_workflows.common.domain_execution import (
            ExecutionArtifactBinding,
        )

        source_path = Path(request.source_asset).expanduser().resolve()

        def binding(path: Path) -> ExecutionArtifactBinding:
            resolved = path.resolve()
            return ExecutionArtifactBinding(
                path=str(resolved),
                sha256=file_sha256(resolved),
                size_bytes=resolved.stat().st_size,
            )

        source = binding(source_path)
        context = request.execution_context
        if (
            context is not None
            and context.embedded_stage is not None
            and source != context.embedded_stage.input_asset
        ):
            raise TextureWorkflowRuntimeError(
                "Texture inspection source differs from the embedded stage input"
            )
        inspection_dir = output_dir / "usd_cli_inspection"
        units: list[TextureInspectionUnit] = []
        render_bindings: list[ExecutionArtifactBinding] = []
        for unit in plan.selected_units:
            context_payload = unit.model_dump(mode="json")
            evidence_path = atomic_write_json(
                inspection_dir / f"{unit.unit_id}-before-render.json",
                {
                    "mock": True,
                    "source_sha256": source.sha256,
                    "unit_id": unit.unit_id,
                    "renderer": "provider-neutral-texture-renderer.v1",
                },
            )
            render_bindings.append(binding(evidence_path))
            units.append(
                TextureInspectionUnit(
                    unit_id=unit.unit_id,
                    material_prim_paths=tuple(
                        str(path)
                        for path in context_payload.get("material_prim_paths") or ()
                    ),
                    material_alias_paths=tuple(
                        str(path)
                        for path in context_payload.get("material_alias_paths") or ()
                    ),
                    member_prim_paths=tuple(
                        str(path)
                        for path in context_payload.get("member_prim_paths") or ()
                    ),
                    member_subset_paths=tuple(
                        str(path)
                        for path in context_payload.get("member_subset_paths") or ()
                    ),
                    uv_status="ready",
                    uv_facts={"uv_scope": "target_prims", "mock": True},
                    proposed_generator_inputs=_proposed_generator_inputs(
                        request,
                        plan,
                        context_payload,
                    ),
                )
            )
        facts_path = atomic_write_json(
            inspection_dir / "inspection_facts.json",
            {
                "schema_version": "content-agent-workflows.texture-inspection-facts.v1",
                "source": source.model_dump(mode="json"),
                "units": [unit.model_dump(mode="json") for unit in units],
            },
        )
        return TextureInspectionResult(
            source=source,
            proposal_plan_digest=texture_plan_digest(plan),
            units=tuple(units),
            before_render_artifacts=tuple(render_bindings),
            inspection_artifacts=(binding(facts_path),),
            capability_constraints=(
                "surface texturing only; label and legend authoring excluded",
                "target scope is immutable after outer acceptance",
                "visual assessment is critique and not semantic authority",
            ),
            renderer_metadata={"interface": "provider-neutral-texture-renderer.v1"},
            tool_metadata={"inspection_adapter": "mock-scene-validator"},
        )

    def validate(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        iteration: int,
        output_dir: Path,
    ) -> TextureValidationResult:
        del request, plan
        if set(unit_ids) - set(unit_artifacts):
            raise ValueError("usd-cli validation requires an artifact for every unit")
        call_index = len(self.calls)
        scheduled_failures = (
            self._failure_schedule[call_index]
            if call_index < len(self._failure_schedule)
            else ()
        )
        unknown_failures = set(scheduled_failures) - set(unit_ids)
        if unknown_failures:
            raise ValueError(
                "mock usd-cli failures must be within the evaluated unit IDs: "
                f"{unknown_failures}"
            )
        self.calls.append(
            MockTextureValidationCall(iteration=iteration, unit_ids=unit_ids)
        )

        findings: list[TextureValidationFinding] = []
        for unit_id in unit_ids:
            evidence_path = (
                output_dir
                / "usd_cli_validation"
                / f"iteration-{iteration}"
                / f"{unit_id}.json"
            )
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            status: TextureValidationStatus = (
                "fail" if unit_id in scheduled_failures else "pass"
            )
            atomic_write_json(
                evidence_path,
                {
                    "mock": True,
                    "output_asset_path": output_asset_path,
                    "status": status,
                    "unit_id": unit_id,
                },
            )
            findings.append(
                TextureValidationFinding(
                    unit_id=unit_id,
                    status=status,
                    summary=(
                        "Mock usd-cli VQA identified a unit-specific defect."
                        if status == "fail"
                        else "Mock usd-cli VQA accepted the unit artifact."
                    ),
                    evidence_artifact_paths=(str(evidence_path.resolve()),),
                )
            )

        return TextureValidationResult(
            iteration=iteration,
            evaluated_unit_ids=unit_ids,
            findings=tuple(findings),
            output_asset_path=output_asset_path,
        )


__all__ = [
    "LiveUsdCliTextureValidator",
    "MockTextureSceneValidator",
    "MockTextureValidationCall",
    "TextureVisualAssessment",
    "TextureVisualAssessor",
    "TextureSceneValidator",
    "VlmTextureVisualAssessor",
]
