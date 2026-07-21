# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-bound authoring of Gate 3A stage metadata derivatives.

The lane is deliberately plan-only: it never guesses a stage unit and never
discovers an edit to make. A signed-off JSON plan names the identity default
root, every descendant unit-bake scale, linear physics value, and nested kind
opinion. The implementation validates that complete plan against a private
source snapshot, authors a deterministic USDZ, reads it back, and publishes it
atomically with a machine receipt.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from world_understanding.utils.usd.package import safe_usdz_member_parts

from .conform_profile import _validate_isa001_dependency_closure
from .gate3a_hygiene import inspect_gate3a_physics_inventory

GATE3A_STAGE_METADATA_PLAN_SCHEMA_VERSION = (
    "content-agent-workflows.simready-gate3a-stage-metadata-plan.v2"
)
GATE3A_STAGE_METADATA_RECEIPT_SCHEMA_VERSION = (
    "content-agent-workflows.simready-gate3a-stage-metadata-receipt.v2"
)
GATE3A_STAGE_METADATA_REQUIREMENT = "G3A.METADATA.001"

_Sha256 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
_Text = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]


class _Blocked(ValueError):
    """Expected unsafe or stale input."""


class StageMetadataEvidence(BaseModel):
    """Exact machine evidence approved by the asset owner."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    artifact_path: _Text
    artifact_sha256: _Sha256

    @field_validator("artifact_path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("evidence artifact_path must be absolute")
        return value


class StageMetadataProvenance(BaseModel):
    """Owner approval and immutable evidence set."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    approved_by: _Text
    approval_reference: _Text
    evidence: Annotated[list[StageMetadataEvidence], Field(min_length=1)]


class StageMetadataTransformScale(BaseModel):
    """One explicit descendant/reset frontier and its compensating scale."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    prim_path: _Text
    scale: Annotated[float, Field(strict=True, gt=0.0)]


class StageMetadataLinearQuantity(BaseModel):
    """One affected linear limit, drive, or state value."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    prim_path: _Text
    attribute_name: _Text
    source_value: float | list[float]
    target_value: float | list[float]

    @model_validator(mode="after")
    def finite_and_same_shape(self) -> StageMetadataLinearQuantity:
        source = (
            self.source_value
            if isinstance(self.source_value, list)
            else [self.source_value]
        )
        target = (
            self.target_value
            if isinstance(self.target_value, list)
            else [self.target_value]
        )
        if len(source) != len(target) or len(source) not in {1, 3}:
            raise ValueError(
                "linear quantity values must have the same scalar/vec3 shape"
            )
        if not all(math.isfinite(value) for value in (*source, *target)):
            raise ValueError("linear quantity values must be finite")
        return self


class StageMetadataKindOpinion(BaseModel):
    """One invalid nested kind opinion explicitly approved for clearing."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    prim_path: _Text
    source_kind: _Text


class Gate3AStageMetadataPlan(BaseModel):
    """Strict complete stage-metadata edit plan."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[
        "content-agent-workflows.simready-gate3a-stage-metadata-plan.v2"
    ]
    source_asset_path: _Text
    source_asset_sha256: _Sha256
    provenance: StageMetadataProvenance
    source_units_mode: Literal[
        "authored_stage_metadata", "owner_approved_missing_metadata"
    ]
    source_meters_per_unit: Annotated[float, Field(strict=True, gt=0.0)]
    target_meters_per_unit: Annotated[float, Field(strict=True)] = 1.0
    default_root_path: _Text
    transform_scales: Annotated[list[StageMetadataTransformScale], Field(min_length=1)]
    linear_quantities: list[StageMetadataLinearQuantity]
    clear_nested_kinds: list[StageMetadataKindOpinion]

    @field_validator("source_asset_path")
    @classmethod
    def source_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("source_asset_path must be absolute")
        return value

    @field_validator("default_root_path")
    @classmethod
    def default_root_is_absolute_prim_path(cls, value: str) -> str:
        if not value.startswith("/") or value == "/":
            raise ValueError("default_root_path must be an absolute prim path")
        return value

    @model_validator(mode="after")
    def unique_plan_entries(self) -> Gate3AStageMetadataPlan:
        groups = (
            [item.prim_path for item in self.transform_scales],
            [(item.prim_path, item.attribute_name) for item in self.linear_quantities],
            [item.prim_path for item in self.clear_nested_kinds],
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("stage-metadata plan contains duplicate entries")
        if not math.isfinite(self.source_meters_per_unit):
            raise ValueError("source_meters_per_unit must be finite and positive")
        if self.target_meters_per_unit != 1.0:
            raise ValueError("target_meters_per_unit must be exactly 1.0")
        return self


@dataclass(frozen=True)
class Gate3AStageMetadataResult:
    """Outcome of one fail-closed authoring attempt."""

    status: str
    passed: bool
    reason: str
    output_path: Path
    receipt_path: Path | None
    report: dict[str, Any]


@dataclass(frozen=True)
class _SourceBinding:
    """One descriptor-pinned source retained through publication."""

    descriptor: int
    state: tuple[int, ...]
    sha256: str


def author_gate3a_stage_metadata_derivative(
    *, asset_path: Path, plan_path: Path, output_dir: Path
) -> Gate3AStageMetadataResult:
    """Validate, author, prove, and atomically publish one deterministic USDZ."""

    asset_path = asset_path.expanduser().absolute()
    plan_path = plan_path.expanduser().absolute()
    output_dir = output_dir.expanduser().absolute()
    report: dict[str, Any] = {
        "schema_version": GATE3A_STAGE_METADATA_RECEIPT_SCHEMA_VERSION,
        "requirement": GATE3A_STAGE_METADATA_REQUIREMENT,
        "asset_path": str(asset_path),
        "plan_path": str(plan_path),
        "changes": [],
    }
    workspace: Path | None = None
    source_binding: _SourceBinding | None = None
    openusd_error: Any = RuntimeError
    try:
        from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdUtils

        openusd_error = Tf.ErrorException

        _reject_symlink_components(asset_path, "source asset")
        _reject_symlink_components(plan_path, "stage-metadata plan")
        _regular_file(plan_path, "stage-metadata plan")
        plan_bytes = _stable_file_bytes(plan_path, "stage-metadata plan")
        plan = Gate3AStageMetadataPlan.model_validate(_strict_json(plan_bytes))
        plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
        if str(asset_path) != plan.source_asset_path:
            raise _Blocked("plan source_asset_path does not match the requested source")
        evidence = _verify_evidence(plan.provenance.evidence)
        if _is_within(output_dir, asset_path.parent):
            raise _Blocked("output directory cannot be inside the source directory")

        workspace = Path(tempfile.mkdtemp(prefix=".gate3a-stage-metadata-"))
        os.chmod(workspace, 0o700)
        snapshot = workspace / "source.usdz"
        source_binding = _capture_source_snapshot(
            asset_path, expected_sha256=plan.source_asset_sha256, snapshot=snapshot
        )
        source_sha256 = source_binding.sha256
        source_stage = Usd.Stage.Open(str(snapshot), load=Usd.Stage.LoadAll)
        if source_stage is None:
            raise _Blocked("unable to open private source snapshot")
        source_state = _inspect_and_validate(
            stage=source_stage,
            plan=plan,
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
        )
        source_inventory = inspect_gate3a_physics_inventory(snapshot)
        source_root_member, source_members = _usdz_member_inventory(snapshot)
        extracted = workspace / "package"
        root_relative = _extract_usdz(snapshot, extracted)
        if root_relative.as_posix() != source_root_member:
            raise _Blocked("USDZ root member changed during private extraction")
        source_stage = None
        gc.collect()

        root_path = extracted / root_relative
        build_stage = Usd.Stage.Open(str(root_path), load=Usd.Stage.LoadAll)
        if build_stage is None:
            raise _Blocked("unable to open private package root")
        _validate_isa001_dependency_closure(
            stage=build_stage,
            source_root=root_path,
            source_tree=extracted,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        _apply_plan(build_stage, plan, Gf=Gf, UsdGeom=UsdGeom)
        if not build_stage.GetRootLayer().Save():
            raise OSError("could not save authored root layer")
        build_stage = None
        gc.collect()

        candidate = workspace / "candidate.usdz"
        _write_deterministic_usdz(extracted, candidate, root_relative=root_relative)
        output_stage = Usd.Stage.Open(str(candidate), load=Usd.Stage.LoadAll)
        if output_stage is None:
            raise _Blocked("unable to read back deterministic USDZ")
        output_state = _inspect_output(
            output_stage, plan, Gf=Gf, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics
        )
        output_stage = None
        gc.collect()
        output_extracted = workspace / "output-package"
        output_root_relative = _extract_usdz(candidate, output_extracted)
        output_extracted_stage = Usd.Stage.Open(
            str(output_extracted / output_root_relative), load=Usd.Stage.LoadAll
        )
        if output_extracted_stage is None:
            raise _Blocked("unable to open extracted output package root")
        _validate_isa001_dependency_closure(
            stage=output_extracted_stage,
            source_root=output_extracted / output_root_relative,
            source_tree=output_extracted,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        output_extracted_stage = None
        gc.collect()
        output_root_member, output_members = _usdz_member_inventory(candidate)
        output_inventory = inspect_gate3a_physics_inventory(candidate)
        if source_inventory.sha256 != output_inventory.sha256:
            raise _Blocked("physics inventory, joint graph, or filtered pairs changed")
        if output_root_member != source_root_member:
            raise _Blocked("USDZ root member changed")
        source_dependencies = {
            name: digest
            for name, digest in source_members.items()
            if name != source_root_member
        }
        output_dependencies = {
            name: digest
            for name, digest in output_members.items()
            if name != output_root_member
        }
        if source_dependencies != output_dependencies:
            raise _Blocked("package dependency bytes changed")
        if source_state["physical_signature"] != output_state["physical_signature"]:
            raise _Blocked("physical world transforms or geometry changed")
        if (
            source_state["joint_anchor_signature"]
            != output_state["joint_anchor_signature"]
        ):
            raise _Blocked("physical joint anchors changed")
        _verify_immutable_inputs(
            asset_path,
            source_binding,
            plan_path,
            plan_sha256,
            evidence,
        )

        output_sha256 = _file_sha256(candidate)
        changes = _changes(plan)
        receipt = {
            "schema_version": GATE3A_STAGE_METADATA_RECEIPT_SCHEMA_VERSION,
            "requirement": GATE3A_STAGE_METADATA_REQUIREMENT,
            "status": "AUTHORED",
            "passed": True,
            "reason": "Published an exact owner-approved Gate 3A stage-metadata derivative.",
            "plan_schema_version": plan.schema_version,
            "plan_sha256": plan_sha256,
            "source_asset_path": str(asset_path),
            "source_asset_sha256": source_sha256,
            "output_asset_sha256": output_sha256,
            "provenance": plan.provenance.model_dump(mode="json"),
            "changes": changes,
            "source_units_mode": plan.source_units_mode,
            "source_meters_per_unit": plan.source_meters_per_unit,
            "source_meters_per_unit_explicit": (
                plan.source_units_mode == "authored_stage_metadata"
            ),
            "default_root_path": plan.default_root_path,
            "default_root_identity_preserved": True,
            "unit_bake_transform_target_count": len(plan.transform_scales),
            "physical_world_transforms_preserved": True,
            "physical_geometry_preserved": True,
            "physical_joint_anchors_preserved": True,
            "source_physical_state_sha256": source_state["physical_signature"],
            "output_physical_state_sha256": output_state["physical_signature"],
            "source_world_transform_sha256": source_state["world_transform_signature"],
            "output_world_transform_sha256": output_state["world_transform_signature"],
            "source_geometry_sha256": source_state["geometry_signature"],
            "output_geometry_sha256": output_state["geometry_signature"],
            "source_joint_anchor_sha256": source_state["joint_anchor_signature"],
            "output_joint_anchor_sha256": output_state["joint_anchor_signature"],
            "physics_inventory_preserved": True,
            "joint_graph_preserved": True,
            "filtered_pairs_preserved": True,
            "dependencies_preserved": True,
            "source_bytes_preserved": True,
            "source_physics_inventory_sha256": source_inventory.sha256,
            "output_physics_inventory_sha256": output_inventory.sha256,
            "readback_verified": True,
        }
        final_path, receipt_path, publication = _publish_bundle(
            candidate=candidate,
            output_dir=output_dir,
            asset_stem=asset_path.stem,
            output_sha256=output_sha256,
            plan_sha256=plan_sha256,
            receipt=receipt,
        )
        try:
            _verify_immutable_inputs(
                asset_path,
                source_binding,
                plan_path,
                plan_sha256,
                evidence,
            )
        except BaseException:
            if publication == "published":
                _remove_failed_publication(final_path, receipt_path)
            raise
        report = {
            **receipt,
            "asset_path": str(asset_path),
            "plan_path": str(plan_path),
            "output_path": str(final_path),
            "receipt_path": str(receipt_path),
            "publication_outcome": publication,
        }
        return Gate3AStageMetadataResult(
            "AUTHORED", True, receipt["reason"], final_path, receipt_path, report
        )
    except (
        openusd_error,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        report.update(status="BLOCKED", passed=False, reason=str(exc), failure=str(exc))
        return Gate3AStageMetadataResult(
            "BLOCKED", False, str(exc), asset_path, None, report
        )
    finally:
        if source_binding is not None:
            os.close(source_binding.descriptor)
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)


def _inspect_and_validate(
    *,
    stage: Any,
    plan: Gate3AStageMetadataPlan,
    Gf: Any,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> dict[str, Any]:
    root_layer = stage.GetRootLayer()
    used_layers = [
        layer
        for layer in stage.GetUsedLayers()
        if layer.identifier != stage.GetSessionLayer().identifier
    ]
    if used_layers != [root_layer]:
        raise _Blocked("composition ambiguity: stage must contain exactly one layer")
    if root_layer.subLayerPaths:
        raise _Blocked("composition ambiguity: sublayers are not supported")
    authored_mpu = stage.HasAuthoredMetadata("metersPerUnit")
    if plan.source_units_mode == "authored_stage_metadata":
        if not authored_mpu:
            raise _Blocked("source units mode requires authored metersPerUnit")
        if float(stage.GetMetadata("metersPerUnit")) != plan.source_meters_per_unit:
            raise _Blocked("source metersPerUnit does not match the plan")
    elif authored_mpu:
        raise _Blocked(
            "owner-approved missing-metadata mode requires metersPerUnit to be absent"
        )

    default_root = stage.GetDefaultPrim()
    if (
        not default_root
        or default_root.GetParent() != stage.GetPseudoRoot()
        or not default_root.IsA(UsdGeom.Xformable)
    ):
        raise _Blocked("stage must have one top-level Xformable default root")
    if str(default_root.GetPath()) != plan.default_root_path:
        raise _Blocked("plan default_root_path does not match the stage default prim")
    _validate_default_root_transform(default_root, Gf=Gf, UsdGeom=UsdGeom)

    scale_targets = _unit_bake_scale_targets(stage, default_root, UsdGeom)
    planned_scales = {item.prim_path: item for item in plan.transform_scales}
    expected_scales = {str(prim.GetPath()) for prim in scale_targets}
    if set(planned_scales) != expected_scales:
        missing = sorted(expected_scales - set(planned_scales))
        extra = sorted(set(planned_scales) - expected_scales)
        raise _Blocked(
            "transform_scales must enumerate every unit-bake frontier exactly; "
            f"missing={missing}, extra={extra}"
        )
    for prim in stage.TraverseAll():
        if (
            prim.IsInstance()
            or prim.IsInstanceProxy()
            or prim.IsInstanceable()
            or bool(prim.GetVariantSets().GetNames())
        ):
            raise _Blocked(f"instances or variants are unsafe at {prim.GetPath()}")
        if prim.GetPrimStack() and any(
            spec.hasReferences or spec.hasPayloads for spec in prim.GetPrimStack()
        ):
            raise _Blocked(f"composition arcs are unsafe at {prim.GetPath()}")
        for attr in prim.GetAttributes():
            if attr.GetNumTimeSamples():
                raise _Blocked(f"time samples are unsafe at {attr.GetPath()}")
    for prim in scale_targets:
        _validate_root_transform(prim, Gf=Gf, UsdGeom=UsdGeom)
        if planned_scales[str(prim.GetPath())].scale != plan.source_meters_per_unit:
            raise _Blocked(
                "transform scale is not the exact metersPerUnit compensation: "
                f"{prim.GetPath()}"
            )
    nested_kinds = {
        str(prim.GetPath()): str(prim.GetMetadata("kind"))
        for prim in stage.TraverseAll()
        if prim.GetPath().pathElementCount > 1 and prim.HasAuthoredMetadata("kind")
    }
    planned_kinds = {
        item.prim_path: item.source_kind for item in plan.clear_nested_kinds
    }
    if nested_kinds != planned_kinds:
        raise _Blocked(
            "clear_nested_kinds is incomplete or contains an unproven opinion"
        )
    for path in planned_kinds:
        spec = root_layer.GetPrimAtPath(Sdf.Path(path))
        if spec is None or not spec.HasInfo("kind"):
            raise _Blocked(f"nested kind is not owned by the root layer: {path}")
    affected = _affected_linear_quantities(stage, UsdPhysics)
    planned = {
        (item.prim_path, item.attribute_name): item for item in plan.linear_quantities
    }
    if set(affected) != set(planned):
        raise _Blocked(
            "linear_quantities is incomplete or contains an unaffected property"
        )
    factor = plan.source_meters_per_unit
    for key, value in affected.items():
        item = planned[key]
        if not _same_value(value, item.source_value):
            raise _Blocked(f"linear source value is stale: {key}")
        expected = _scaled_value(value, factor)
        if not _same_value(expected, item.target_value):
            raise _Blocked(
                f"linear target is not exact metersPerUnit compensation: {key}"
            )
    _validate_unit_bake_coverage(
        stage,
        default_root=default_root,
        scale_targets=scale_targets,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
    )
    return _physical_state(
        stage,
        plan.source_meters_per_unit,
        Gf,
        UsdGeom,
        UsdPhysics,
        default_root_path=plan.default_root_path,
    )


def _unit_bake_scale_targets(stage: Any, default_root: Any, UsdGeom: Any) -> list[Any]:
    """Return every transform frontier that must receive unit compensation."""

    targets: dict[str, Any] = {}
    top_roots = list(stage.GetPseudoRoot().GetChildren())
    if not top_roots:
        raise _Blocked("stage has no top-level roots")
    for prim in top_roots:
        if not prim.IsA(UsdGeom.Xformable):
            raise _Blocked(f"top-level root is not Xformable: {prim.GetPath()}")
        if prim != default_root:
            targets[str(prim.GetPath())] = prim

    def add_first_xformable_descendants(parent: Any) -> None:
        for child in parent.GetChildren():
            if child.IsA(UsdGeom.Xformable):
                targets[str(child.GetPath())] = child
            else:
                add_first_xformable_descendants(child)

    add_first_xformable_descendants(default_root)
    for prim in stage.TraverseAll():
        if prim == default_root or not prim.IsA(UsdGeom.Xformable):
            continue
        if UsdGeom.Xformable(prim).GetResetXformStack():
            targets[str(prim.GetPath())] = prim
    if not targets:
        raise _Blocked("default root has no safe descendant unit-bake frontier")
    return [targets[path] for path in sorted(targets)]


def _validate_default_root_transform(prim: Any, *, Gf: Any, UsdGeom: Any) -> None:
    """Require the asset default root to satisfy the pinned origin rule."""

    if prim.GetAttribute("xformOp:scale:gate3aStageMetadata"):
        raise _Blocked(f"reserved transform scale already exists: {prim.GetPath()}")
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    if not Gf.IsClose(matrix, Gf.Matrix4d(1.0), 1e-9):
        raise _Blocked(
            "default root local transform must be identity before unit baking: "
            f"{prim.GetPath()}"
        )


def _validate_root_transform(prim: Any, *, Gf: Any, UsdGeom: Any) -> None:
    """Reject transform roots whose exact outer scaling is ambiguous."""

    xformable = UsdGeom.Xformable(prim)
    if prim.GetAttribute("xformOp:scale:gate3aStageMetadata"):
        raise _Blocked(f"reserved root scale already exists: {prim.GetPath()}")
    matrix = xformable.GetLocalTransformation()
    if any(
        not math.isclose(float(matrix[row][3]), 0.0, abs_tol=1e-12) for row in range(3)
    ) or not math.isclose(float(matrix[3][3]), 1.0, abs_tol=1e-12):
        raise _Blocked(f"non-affine root transform is unsafe: {prim.GetPath()}")
    rows = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    lengths = [math.sqrt(sum(value * value for value in row)) for row in rows]
    if any(length <= 1e-12 for length in lengths):
        raise _Blocked(f"singular root transform is unsafe: {prim.GetPath()}")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if determinant <= 0.0:
        raise _Blocked(f"reflected root transform is unsafe: {prim.GetPath()}")
    for first in range(3):
        for second in range(first + 1, 3):
            cosine = sum(
                rows[first][index] * rows[second][index] for index in range(3)
            ) / (lengths[first] * lengths[second])
            if not math.isclose(cosine, 0.0, abs_tol=1e-9):
                raise _Blocked(f"sheared root transform is unsafe: {prim.GetPath()}")


def _affected_linear_quantities(
    stage: Any, UsdPhysics: Any
) -> dict[tuple[str, str], Any]:
    result: dict[tuple[str, str], Any] = {}
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
        is_distance = prim.IsA(UsdPhysics.DistanceJoint)
        for attr in prim.GetAttributes():
            name = attr.GetName()
            # Local joint anchors are body-local geometry coordinates.  The
            # planned top-root scale already compensates them; scaling those
            # attributes again would move the physical anchor twice.
            affected = is_prismatic and (
                name in {"physics:lowerLimit", "physics:upperLimit"}
                or name.startswith("drive:linear:physics:target")
                or name == "drive:linear:physics:maxForce"
                or name.startswith("state:linear:physics:")
            )
            affected = affected or (
                is_distance and name in {"physics:minDistance", "physics:maxDistance"}
            )
            if not affected or not attr.HasAuthoredValueOpinion():
                continue
            value = attr.Get()
            scalar = (
                float(value)
                if isinstance(value, int | float)
                else [float(v) for v in value]
            )
            values = scalar if isinstance(scalar, list) else [scalar]
            if not all(math.isfinite(v) for v in values):
                raise _Blocked(f"non-finite authored linear quantity: {attr.GetPath()}")
            result[(str(prim.GetPath()), name)] = scalar
    return result


def _validate_unit_bake_coverage(
    stage: Any,
    *,
    default_root: Any,
    scale_targets: list[Any],
    UsdGeom: Any,
    UsdPhysics: Any,
) -> None:
    """Prove every spatial or joint endpoint prim receives one unit bake."""

    target_paths = [target.GetPath() for target in scale_targets]

    def require_covered(path: Any, label: str) -> None:
        candidates = [target for target in target_paths if path.HasPrefix(target)]
        if not candidates:
            raise _Blocked(f"{label} is outside every unit-bake frontier: {path}")
        deepest_depth = max(target.pathElementCount for target in candidates)
        if sum(target.pathElementCount == deepest_depth for target in candidates) != 1:
            raise _Blocked(f"{label} has ambiguous unit-bake ownership: {path}")

    if (
        default_root.IsA(UsdGeom.Boundable)
        or default_root.IsA(UsdPhysics.Joint)
        or default_root.HasAPI(UsdPhysics.RigidBodyAPI)
        or default_root.HasAPI(UsdPhysics.CollisionAPI)
    ):
        raise _Blocked(
            "default root directly owns spatial physics or geometry that cannot be "
            "safely baked below an identity root"
        )
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Boundable):
            require_covered(prim.GetPath(), "geometry prim")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            require_covered(prim.GetPath(), "rigid-body prim")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            require_covered(prim.GetPath(), "collision prim")
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for label, relationship in (
            ("body0", joint.GetBody0Rel()),
            ("body1", joint.GetBody1Rel()),
        ):
            targets = relationship.GetTargets()
            if len(targets) != 1 or not targets[0].IsPrimPath():
                raise _Blocked(
                    f"joint {label} endpoint is not one singular prim at {prim.GetPath()}"
                )
            if not stage.GetPrimAtPath(targets[0]):
                raise _Blocked(f"joint {label} endpoint is missing at {prim.GetPath()}")
            require_covered(targets[0], f"joint {label} endpoint")


def _physical_state(
    stage: Any,
    mpu: float,
    Gf: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    *,
    default_root_path: str,
) -> dict[str, Any]:
    cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
    transforms: list[Any] = []
    geometry: list[Any] = []
    anchors: list[Any] = []
    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Xformable):
            matrix = cache.GetLocalToWorldTransform(prim)
            translation = matrix.ExtractTranslation()
            linear_factor = 1.0 if prim_path == default_root_path else mpu
            transforms.append(
                (
                    prim_path,
                    [
                        [
                            float(matrix[row][column]) * linear_factor
                            for column in range(3)
                        ]
                        for row in range(3)
                    ],
                    [float(value) * mpu for value in translation],
                )
            )
        if prim.IsA(UsdGeom.Boundable):
            bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            geometry.append(
                (
                    prim_path,
                    [float(v) * mpu for v in bound.GetMin()],
                    [float(v) * mpu for v in bound.GetMax()],
                )
            )
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            for side, rel, attr in (
                (0, joint.GetBody0Rel(), joint.GetLocalPos0Attr()),
                (1, joint.GetBody1Rel(), joint.GetLocalPos1Attr()),
            ):
                targets = rel.GetTargets()
                if len(targets) != 1:
                    raise _Blocked(
                        f"joint endpoint is not singular at {prim.GetPath()}"
                    )
                body = stage.GetPrimAtPath(targets[0])
                point = cache.GetLocalToWorldTransform(body).Transform(
                    Gf.Vec3d(attr.Get())
                )
                anchors.append((prim_path, side, [float(v) * mpu for v in point]))
    transform_signature = _json_hash(transforms)
    geometry_signature = _json_hash(geometry)
    return {
        "physical_signature": _json_hash([transforms, geometry]),
        "world_transform_signature": transform_signature,
        "geometry_signature": geometry_signature,
        "joint_anchor_signature": _json_hash(anchors),
    }


def _apply_plan(
    stage: Any, plan: Gate3AStageMetadataPlan, *, Gf: Any, UsdGeom: Any
) -> None:
    stage.SetMetadata("metersPerUnit", 1.0)
    for scale_item in plan.transform_scales:
        prim = stage.GetPrimAtPath(scale_item.prim_path)
        xformable = UsdGeom.Xformable(prim)
        previous_order = list(xformable.GetXformOpOrderAttr().Get() or [])
        op = xformable.AddScaleOp(
            UsdGeom.XformOp.PrecisionDouble, "gate3aStageMetadata"
        )
        op.Set(Gf.Vec3d(scale_item.scale))
        scale_token = op.GetOpName()
        authored_order = list(xformable.GetXformOpOrderAttr().Get() or [])
        if authored_order.count(scale_token) != 1:
            raise _Blocked(
                f"could not author one transform scale op: {scale_item.prim_path}"
            )
        reset_tokens = [
            token for token in previous_order if str(token) == "!resetXformStack!"
        ]
        if len(reset_tokens) > 1 or (
            reset_tokens and previous_order[0] != reset_tokens[0]
        ):
            raise _Blocked(f"ambiguous reset xform order: {scale_item.prim_path}")
        ordered = [*reset_tokens, scale_token]
        ordered.extend(
            token for token in previous_order if str(token) != "!resetXformStack!"
        )
        if not xformable.GetXformOpOrderAttr().Set(ordered):
            raise OSError(
                f"could not set outer transform scale order: {scale_item.prim_path}"
            )
    for quantity in plan.linear_quantities:
        attr = stage.GetPrimAtPath(quantity.prim_path).GetAttribute(
            quantity.attribute_name
        )
        target = quantity.target_value
        if isinstance(target, list):
            target = (
                Gf.Vec3f(*target)
                if attr.GetTypeName().type.typeName == "GfVec3f"
                else Gf.Vec3d(*target)
            )
        attr.Set(target)
    for kind_opinion in plan.clear_nested_kinds:
        stage.GetPrimAtPath(kind_opinion.prim_path).ClearMetadata("kind")


def _inspect_output(
    stage: Any,
    plan: Gate3AStageMetadataPlan,
    *,
    Gf: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> dict[str, Any]:
    if (
        not stage.HasAuthoredMetadata("metersPerUnit")
        or float(stage.GetMetadata("metersPerUnit")) != 1.0
    ):
        raise _Blocked("output metersPerUnit readback failed")
    default_root = stage.GetDefaultPrim()
    if not default_root or str(default_root.GetPath()) != plan.default_root_path:
        raise _Blocked("output default root readback failed")
    _validate_default_root_transform(default_root, Gf=Gf, UsdGeom=UsdGeom)
    scale_targets = _unit_bake_scale_targets(stage, default_root, UsdGeom)
    if {str(prim.GetPath()) for prim in scale_targets} != {
        item.prim_path for item in plan.transform_scales
    }:
        raise _Blocked("output unit-bake frontier readback failed")
    _validate_unit_bake_coverage(
        stage,
        default_root=default_root,
        scale_targets=scale_targets,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
    )
    for scale_item in plan.transform_scales:
        prim = stage.GetPrimAtPath(scale_item.prim_path)
        attr = prim.GetAttribute("xformOp:scale:gate3aStageMetadata")
        if not attr or not _same_value(
            [float(v) for v in attr.Get()], [scale_item.scale] * 3
        ):
            raise _Blocked(f"transform scale readback failed: {scale_item.prim_path}")
        order = list(UsdGeom.Xformable(prim).GetXformOpOrderAttr().Get() or [])
        scale_index = order.index(attr.GetName())
        expected_index = 1 if order and str(order[0]) == "!resetXformStack!" else 0
        if scale_index != expected_index:
            raise _Blocked(f"transform scale is not outermost: {scale_item.prim_path}")
    for quantity in plan.linear_quantities:
        value = (
            stage.GetPrimAtPath(quantity.prim_path)
            .GetAttribute(quantity.attribute_name)
            .Get()
        )
        value = (
            float(value)
            if isinstance(value, int | float)
            else [float(v) for v in value]
        )
        if not _same_value(value, quantity.target_value):
            raise _Blocked(
                "linear quantity readback failed: "
                f"{quantity.prim_path}.{quantity.attribute_name}"
            )
    if any(
        stage.GetPrimAtPath(item.prim_path).HasAuthoredMetadata("kind")
        for item in plan.clear_nested_kinds
    ):
        raise _Blocked("planned nested kind survived readback")
    return _physical_state(
        stage,
        1.0,
        Gf,
        UsdGeom,
        UsdPhysics,
        default_root_path=plan.default_root_path,
    )


def _extract_usdz(source: Path, destination: Path) -> Path:
    destination.mkdir(mode=0o700)
    normalized_names: set[str] = set()
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if not infos:
            raise _Blocked("USDZ is empty")
        for info in infos:
            parts = safe_usdz_member_parts(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                parts is None
                or info.is_dir()
                or stat.S_ISLNK(mode)
                or info.flag_bits & 0x1
                or info.compress_type != zipfile.ZIP_STORED
            ):
                raise _Blocked(f"unsafe USDZ member: {info.filename}")
            normalized = "/".join(parts)
            if normalized in normalized_names:
                raise _Blocked(f"duplicate USDZ member: {normalized}")
            normalized_names.add(normalized)
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                target,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
            )
            with archive.open(info) as stream, os.fdopen(descriptor, "wb") as output:
                shutil.copyfileobj(stream, output)
    root_parts = safe_usdz_member_parts(infos[0].filename)
    if root_parts is None or Path(root_parts[-1]).suffix.lower() not in {
        ".usd",
        ".usda",
        ".usdc",
    }:
        raise _Blocked("first USDZ member is not a USD root layer")
    return Path(*root_parts)


def _write_deterministic_usdz(
    root: Path, destination: Path, *, root_relative: Path
) -> None:
    files = _private_tree_files(root)
    root_path = root / root_relative
    if root_path not in files:
        raise _Blocked("private package root is missing")
    ordered_files = [root_path, *(path for path in files if path != root_path)]
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path in ordered_files:
            name = path.relative_to(root).as_posix()
            info = _aligned_usdz_member_info(archive, name, path.stat().st_size)
            with (
                path.open("rb") as source_stream,
                archive.open(
                    info,
                    "w",
                    force_zip64=path.stat().st_size > zipfile.ZIP64_LIMIT,
                ) as output_stream,
            ):
                shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
    _validate_usdz_alignment(destination)


def _private_tree_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            child = current_path / directory
            if child.is_symlink():
                raise _Blocked(f"private package contains a symlink: {child}")
        for name in names:
            child = current_path / name
            metadata = child.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise _Blocked(f"private package contains a non-file: {child}")
            files.append(child)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _aligned_usdz_member_info(
    archive: zipfile.ZipFile, name: str, source_size: int
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.file_size = source_size
    info.external_attr = 0o100644 << 16
    if archive.fp is None:  # pragma: no cover - ZipFile internal invariant
        raise OSError("USDZ output stream is unavailable")
    zip64_extra_size = 20 if source_size > zipfile.ZIP64_LIMIT else 0
    data_start = archive.fp.tell() + 30 + len(name.encode("utf-8")) + zip64_extra_size
    padding_size = (-data_start) % 64
    if 0 < padding_size < 4:
        padding_size += 64
    if padding_size:
        payload_size = padding_size - 4
        info.extra = struct.pack("<HH", 0x1986, payload_size) + (b"\0" * payload_size)
    return info


def _validate_usdz_alignment(path: Path) -> None:
    with path.open("rb") as raw, zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            raw.seek(info.header_offset)
            header = raw.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise _Blocked(f"invalid USDZ local header: {info.filename}")
            name_length, extra_length = struct.unpack_from("<HH", header, 26)
            data_offset = info.header_offset + 30 + name_length + extra_length
            if data_offset % 64:
                raise _Blocked(f"USDZ member is not 64-byte aligned: {info.filename}")


def _usdz_member_inventory(path: Path) -> tuple[str, dict[str, str]]:
    inventory: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos:
            raise _Blocked("USDZ is empty")
        for info in infos:
            parts = safe_usdz_member_parts(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                parts is None
                or info.is_dir()
                or stat.S_ISLNK(mode)
                or info.flag_bits & 0x1
                or info.compress_type != zipfile.ZIP_STORED
            ):
                raise _Blocked(f"unsafe USDZ member: {info.filename}")
            name = "/".join(parts)
            if name in inventory:
                raise _Blocked(f"duplicate USDZ member: {name}")
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            inventory[name] = digest.hexdigest()
        root_parts = safe_usdz_member_parts(infos[0].filename)
        if root_parts is None:
            raise _Blocked("USDZ root member is unsafe")
        root_name = "/".join(root_parts)
        if Path(root_name).suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise _Blocked("first USDZ member is not a USD root layer")
        return root_name, inventory


def _publish_bundle(
    *,
    candidate: Path,
    output_dir: Path,
    asset_stem: str,
    output_sha256: str,
    plan_sha256: str,
    receipt: dict[str, Any],
) -> tuple[Path, Path, str]:
    """Atomically publish the USDZ and receipt as one directory."""

    _mkdir_regular_chain(output_dir)
    publish_root = output_dir / "stage-metadata"
    publish_root.mkdir(mode=0o755, exist_ok=True)
    if publish_root.is_symlink() or not publish_root.is_dir():
        raise _Blocked(f"publication root is not a regular directory: {publish_root}")
    bundle_name = f"{output_sha256}-{plan_sha256}"
    final_dir = publish_root / bundle_name
    asset_name = f"{asset_stem}.gate3a-stage-metadata.usdz"
    receipt_name = "receipt.json"
    final_path = final_dir / asset_name
    receipt_path = final_dir / receipt_name
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    def verify_existing(outcome: str) -> tuple[Path, Path, str]:
        if final_dir.is_symlink() or not final_dir.is_dir():
            raise _Blocked("content-addressed publication is not a directory")
        _regular_file(final_path, "published stage-metadata asset")
        _regular_file(receipt_path, "published stage-metadata receipt")
        if _file_sha256(final_path) != output_sha256:
            raise _Blocked("content-addressed output contains conflicting bytes")
        if receipt_path.read_bytes() != receipt_bytes:
            raise _Blocked("content-addressed receipt contains conflicting bytes")
        return final_path, receipt_path, outcome

    if final_dir.exists() or final_dir.is_symlink():
        return verify_existing("reused")

    build_dir = Path(
        tempfile.mkdtemp(prefix=".stage-metadata-bundle-", dir=publish_root)
    )
    os.chmod(build_dir, 0o700)
    renamed_bundle_identity: tuple[int, int] | None = None
    try:
        staged_asset = build_dir / asset_name
        _copy_regular_file(candidate, staged_asset)
        os.chmod(staged_asset, 0o644)
        staged_receipt = build_dir / receipt_name
        descriptor = os.open(
            staged_receipt,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o644,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(receipt_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(build_dir)
        try:
            os.rename(build_dir, final_dir)
        except FileExistsError:
            return verify_existing("reused-race")
        published_metadata = final_dir.stat(follow_symlinks=False)
        renamed_bundle_identity = (
            published_metadata.st_dev,
            published_metadata.st_ino,
        )
        try:
            _fsync_directory(publish_root)
            return verify_existing("published")
        except BaseException as publication_error:
            try:
                _rollback_renamed_bundle(
                    final_dir,
                    publish_root=publish_root,
                    expected_identity=renamed_bundle_identity,
                )
            except BaseException as rollback_error:
                raise OSError(
                    "publication failed and rollback could not be durably "
                    f"confirmed: {rollback_error}"
                ) from publication_error
            raise
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)


def _copy_regular_file(source: Path, target: Path) -> None:
    descriptor = os.open(
        target,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with (
            source.open("rb") as input_stream,
            os.fdopen(descriptor, "wb", closefd=False) as output_stream,
        ):
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    finally:
        os.close(descriptor)


def _remove_failed_publication(asset_path: Path, receipt_path: Path) -> None:
    """Remove a bundle published by this call when the final trust check fails."""

    bundle = asset_path.parent
    if receipt_path.parent != bundle:
        raise _Blocked("failed publication paths do not share one atomic bundle")
    if bundle.is_symlink() or not bundle.is_dir():
        raise _Blocked("failed publication bundle changed before cleanup")
    shutil.rmtree(bundle)
    _fsync_directory(bundle.parent)


def _rollback_renamed_bundle(
    bundle: Path,
    *,
    publish_root: Path,
    expected_identity: tuple[int, int],
) -> None:
    """Remove only the bundle this call renamed before publication failed."""

    metadata = bundle.stat(follow_symlinks=False)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise _Blocked("renamed publication changed before rollback")
    shutil.rmtree(bundle)
    _fsync_directory(publish_root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_evidence(
    items: list[StageMetadataEvidence],
) -> tuple[tuple[Path, str], ...]:
    identities = tuple(
        (Path(item.artifact_path), item.artifact_sha256) for item in items
    )
    for path, digest in identities:
        _reject_symlink_components(path, "evidence")
        if _stable_file_sha256(path, "evidence") != digest:
            raise _Blocked(f"evidence SHA-256 is stale: {path}")
    return identities


def _verify_immutable_inputs(
    asset: Path,
    source_binding: _SourceBinding,
    plan: Path,
    plan_sha: str,
    evidence: tuple[tuple[Path, str], ...],
) -> None:
    _verify_source_binding(asset, source_binding)
    if _stable_file_sha256(plan, "stage-metadata plan") != plan_sha:
        raise _Blocked("plan mutated during authoring")
    for path, digest in evidence:
        if _stable_file_sha256(path, "evidence") != digest:
            raise _Blocked("evidence mutated during authoring")


def _changes(plan: Gate3AStageMetadataPlan) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    changes.append(
        {
            "kind": "set_meters_per_unit",
            "source": plan.source_meters_per_unit,
            "source_units_mode": plan.source_units_mode,
            "target": 1.0,
        }
    )
    changes.append(
        {
            "kind": "preserve_identity_default_root",
            "prim_path": plan.default_root_path,
        }
    )
    changes.extend(
        {"kind": "add_unit_bake_transform_scale", **item.model_dump()}
        for item in plan.transform_scales
    )
    changes.extend(
        {"kind": "scale_linear_quantity", **item.model_dump()}
        for item in plan.linear_quantities
    )
    changes.extend(
        {"kind": "clear_nested_kind", **item.model_dump()}
        for item in plan.clear_nested_kinds
    )
    return changes


def _strict_json(payload: bytes) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise _Blocked(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        payload.decode(),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            _Blocked(f"non-finite JSON number: {value}")
        ),
    )


def _scaled_value(value: Any, factor: float) -> Any:
    return (
        [item * factor for item in value] if isinstance(value, list) else value * factor
    )


def _same_value(left: Any, right: Any) -> bool:
    lhs = left if isinstance(left, list) else [left]
    rhs = right if isinstance(right, list) else [right]
    return len(lhs) == len(rhs) and all(
        math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-9)
        for a, b in zip(lhs, rhs, strict=True)
    )


def _json_hash(value: Any) -> str:
    def canonical(item: Any) -> Any:
        if isinstance(item, float):
            # Source assets commonly store transforms and bounds in float32.
            # Compare physical metres at 10-micrometre resolution so a unit
            # rewrite does not fail solely on the float32 -> float64 path.
            return round(item, 5)
        if isinstance(item, list):
            return [canonical(child) for child in item]
        if isinstance(item, tuple):
            return [canonical(child) for child in item]
        return item

    return hashlib.sha256(
        json.dumps(
            canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise _Blocked(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise _Blocked(f"{label} must be a regular non-symlink file: {path}")


def _capture_source_snapshot(
    path: Path, *, expected_sha256: str, snapshot: Path
) -> _SourceBinding:
    """Pin source bytes by descriptor and materialize one private snapshot."""

    _reject_symlink_components(path, "source asset")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (named.st_dev, named.st_ino):
            raise _Blocked("source asset changed while it was opened")
        output_descriptor = os.open(
            snapshot,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        digest = hashlib.sha256()
        try:
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor, min(1024 * 1024, before.st_size - offset), offset
                )
                if not chunk:
                    raise _Blocked("source asset changed while it was snapshotted")
                digest.update(chunk)
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(output_descriptor, remaining)
                    if written <= 0:  # pragma: no cover - regular-file invariant
                        raise OSError("short write while creating source snapshot")
                    remaining = remaining[written:]
                offset += len(chunk)
            if os.pread(descriptor, 1, offset):
                raise _Blocked("source asset grew while it was snapshotted")
            os.fsync(output_descriptor)
        finally:
            os.close(output_descriptor)
        after = os.fstat(descriptor)
        if _descriptor_state(before) != _descriptor_state(after):
            raise _Blocked("source asset changed while it was snapshotted")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise _Blocked("plan source_asset_sha256 is stale")
        os.chmod(snapshot, 0o400)
        if _stable_file_sha256(snapshot, "private source snapshot") != actual_sha256:
            raise _Blocked("private source snapshot changed source bytes")
        return _SourceBinding(
            descriptor=descriptor,
            state=_descriptor_state(after),
            sha256=actual_sha256,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_source_binding(path: Path, binding: _SourceBinding) -> None:
    _reject_symlink_components(path, "source asset")
    observed = os.fstat(binding.descriptor)
    if _descriptor_state(observed) != binding.state:
        raise _Blocked("source asset mutated during authoring")
    if _stable_descriptor_sha256(binding.descriptor) != binding.sha256:
        raise _Blocked("source asset bytes mutated during authoring")
    named = os.stat(path, follow_symlinks=False)
    if _descriptor_state(named) != binding.state:
        raise _Blocked("source asset path changed during authoring")
    if _stable_file_sha256(path, "source asset") != binding.sha256:
        raise _Blocked("source asset path bytes changed during authoring")


def _stable_descriptor_sha256(descriptor: int) -> str:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise _Blocked("source descriptor no longer identifies a regular file")
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise _Blocked("source descriptor changed while it was hashed")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, offset):
        raise _Blocked("source descriptor grew while it was hashed")
    if _descriptor_state(os.fstat(descriptor)) != _descriptor_state(before):
        raise _Blocked("source descriptor changed while it was hashed")
    return digest.hexdigest()


def _stable_file_bytes(path: Path, label: str) -> bytes:
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _Blocked(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not chunk:
                raise _Blocked(f"{label} changed while it was read")
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, offset):
            raise _Blocked(f"{label} grew while it was read")
        if _descriptor_state(os.fstat(descriptor)) != _descriptor_state(before):
            raise _Blocked(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stable_file_sha256(path: Path, label: str) -> str:
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        return _stable_descriptor_sha256(descriptor)
    finally:
        os.close(descriptor)


def _descriptor_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise _Blocked(f"{label} path component is missing: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise _Blocked(f"{label} path contains a symlink: {current}")


def _mkdir_regular_chain(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, "output directory")
    if not path.is_dir():
        raise _Blocked(f"output directory is not a directory: {path}")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
