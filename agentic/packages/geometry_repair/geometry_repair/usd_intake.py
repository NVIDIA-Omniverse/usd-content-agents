# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-mutating OpenUSD dependency and representation-role intake."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .dependency_localization import (
    DependencyApprovalPolicy,
    DependencyRemapInput,
    DependencyResolutionMethod,
)

USD_INTAKE_SCHEMA_VERSION = "geometry-repair.usd-intake.v1"
_USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
_HASH_CHUNK_BYTES = 1024 * 1024

DependencyKind = Literal["sublayer", "reference", "payload", "asset"]
DependencyStatus = Literal[
    "resolved_local",
    "resolved_package",
    "unresolved_local",
    "remote_not_fetched",
    "outside_allowed_roots",
    "skipped_size_limit",
    "dynamic_pattern",
]
PrimRole = Literal[
    "render",
    "source_collision",
    "disabled_visual_collision_api",
    "helper",
    "guide",
    "unknown",
]

_GEOMETRY_TIME_VARYING_ATTRIBUTES = {
    "axis",
    "curveVertexCounts",
    "extent",
    "faceVertexCounts",
    "faceVertexIndices",
    "height",
    "holeIndices",
    "lengths",
    "points",
    "radius",
    "size",
    "widths",
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UsdDependencyRecord(_FrozenModel):
    """One authored dependency occurrence and its safe local-resolution evidence."""

    source_layer: str
    kind: DependencyKind
    authored_path: str
    status: DependencyStatus
    local_path: str | None = None
    package_member: str | None = None
    resolution_method: DependencyResolutionMethod | None = None
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    size_bytes: int | None = Field(default=None, ge=0)
    remote_scheme: str | None = None
    reason_code: str | None = None
    reason: str | None = None


class UsdRoleEvidence(_FrozenModel):
    """One authored or schema-derived signal supporting a prim role."""

    signal: str
    value: str
    supports: PrimRole | None = None
    decisive: bool = False


class UsdPrimvarFact(_FrozenModel):
    name: str
    type_name: str
    interpolation: str | None = None
    element_size: int = Field(default=1, ge=1)
    value_count: int = Field(default=0, ge=0)
    index_count: int = Field(default=0, ge=0)
    authored: bool = False


class UsdMaterialSubsetFact(_FrozenModel):
    path: str
    family_name: str | None = None
    element_type: str | None = None
    index_count: int = Field(default=0, ge=0)
    material_binding_targets: list[str] = Field(default_factory=list)


class UsdMaterialFact(_FrozenModel):
    binding_relationships: dict[str, list[str]] = Field(default_factory=dict)
    computed_material_path: str | None = None
    computed_binding_status: Literal["bound", "unbound", "not_evaluated"] = "unbound"
    subsets: list[UsdMaterialSubsetFact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class UsdCollisionFact(_FrozenModel):
    api_applied: bool = False
    enabled: bool | None = None
    enabled_authored: bool = False
    approximation: str | None = None
    approximation_authored: bool = False
    physics_owner_path: str | None = None


class UsdNativePrimitiveFact(_FrozenModel):
    schema_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class UsdPrimRecord(_FrozenModel):
    """Composed source facts for one prim, without triangulation or mutation."""

    path: str
    name: str
    type_name: str
    parent_path: str | None
    child_count: int = Field(ge=0)
    active: bool
    defined: bool
    abstract: bool
    instance: bool
    instance_proxy: bool
    local_transform: list[list[float]] | None = None
    world_transform: list[list[float]] | None = None
    resets_xform_stack: bool | None = None
    transform_time_varying: bool = False
    purpose: str | None = None
    visibility: str | None = None
    kind: str | None = None
    variant_selections: dict[str, str] = Field(default_factory=dict)
    material: UsdMaterialFact
    primvars: list[UsdPrimvarFact] = Field(default_factory=list)
    collision: UsdCollisionFact
    native_primitive: UsdNativePrimitiveFact | None = None
    role: PrimRole
    role_evidence: list[UsdRoleEvidence] = Field(min_length=1)
    inspection_warnings: list[str] = Field(default_factory=list)


class UsdIntakeReport(_FrozenModel):
    """Complete, source-preserving dependency and prim-role inventory."""

    schema_version: Literal["geometry-repair.usd-intake.v1"] = USD_INTAKE_SCHEMA_VERSION
    source_path: str
    source_sha256: str = Field(min_length=64, max_length=64)
    source_size_bytes: int = Field(ge=0)
    source_unchanged: bool
    layer_readable: bool
    stage_readable: bool
    composition_complete: bool
    stage_error: str | None = None
    default_prim_path: str | None = None
    meters_per_unit: float | None = Field(default=None, gt=0.0)
    up_axis: str | None = None
    dependencies: list[UsdDependencyRecord] = Field(default_factory=list)
    prims: list[UsdPrimRecord] = Field(default_factory=list)
    joint_prim_paths: list[str] = Field(default_factory=list)
    rigid_body_prim_paths: list[str] = Field(default_factory=list)
    time_varying_prim_paths: list[str] = Field(default_factory=list)
    geometry_time_varying_prim_paths: list[str] = Field(default_factory=list)
    skeleton_prim_paths: list[str] = Field(default_factory=list)
    animation_prim_paths: list[str] = Field(default_factory=list)
    brep_prim_paths: list[str] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class UsdIntakeUnavailableError(RuntimeError):
    """Raised when OpenUSD is unavailable or the source layer cannot be read."""


class _DependencyOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_layer: str
    kind: DependencyKind
    authored_path: str


def _stable_file_hash(path: Path, *, max_bytes: int) -> tuple[str | None, int, str | None]:
    before = path.stat()
    if before.st_size > max_bytes:
        return None, before.st_size, f"file size exceeds hash limit {max_bytes} bytes"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        return None, after.st_size, "file changed while it was hashed"
    return digest.hexdigest(), after.st_size, None


def _asset_paths(value: Any) -> Iterable[str]:
    try:
        from pxr import Sdf
    except ImportError:
        return
    if isinstance(value, Sdf.AssetPath):
        if value.path:
            yield str(value.path)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _asset_paths(item)
        return
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        for item in value:
            yield from _asset_paths(item)


def _layer_dependencies(
    layer: Any,
    *,
    warnings: list[str] | None = None,
) -> list[_DependencyOccurrence]:
    from pxr import Sdf

    source_layer = str(layer.realPath or layer.identifier)
    discovered: set[tuple[str, str]] = set()

    def add(kind: DependencyKind, authored_path: str) -> None:
        if authored_path:
            discovered.add((kind, str(authored_path)))

    for sublayer in layer.subLayerPaths:
        add("sublayer", str(sublayer))

    def visit(path: Any) -> None:
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.PrimSpec):
            for reference in spec.referenceList.GetAppliedItems():
                add("reference", str(reference.assetPath))
            for payload in spec.payloadList.GetAppliedItems():
                add("payload", str(payload.assetPath))
        if spec is not None and hasattr(spec, "ListInfoKeys"):
            for key in spec.ListInfoKeys():
                try:
                    value = spec.GetInfo(key)
                except Exception as exc:
                    if warnings is not None:
                        warnings.append(
                            "dependency metadata inspection failed for "
                            f"{source_layer}:{path} key {key!r}: {type(exc).__name__}: {exc}"
                        )
                    continue
                for asset_path in _asset_paths(value):
                    add("asset", asset_path)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    known_paths = {path for _, path in discovered}
    for external in layer.GetExternalReferences():
        external_path = str(external)
        if external_path and external_path not in known_paths:
            add("reference", external_path)
    for external in layer.GetExternalAssetDependencies():
        external_path = str(external)
        if external_path and external_path not in known_paths:
            add("asset", external_path)
    return [
        _DependencyOccurrence(
            source_layer=source_layer,
            kind=kind,
            authored_path=authored_path,
        )
        for kind, authored_path in sorted(discovered)
    ]


def _resolve_dependency(
    occurrence: _DependencyOccurrence,
    *,
    approval_policy: DependencyApprovalPolicy,
    max_hash_bytes: int,
) -> UsdDependencyRecord:
    decision = approval_policy.resolve(
        source_layer=occurrence.source_layer,
        authored_path=occurrence.authored_path,
    )
    if not decision.resolved or decision.local_path is None:
        if decision.reason_code == "dynamic_pattern":
            status: DependencyStatus = "dynamic_pattern"
        elif decision.reason_code == "remote_not_fetched":
            status = "remote_not_fetched"
        elif decision.reason_code in {
            "not_approved",
            "symlink_escape",
            "traversal_escape",
            "unsafe_package_member",
        }:
            status = "outside_allowed_roots"
        else:
            status = "unresolved_local"
        return UsdDependencyRecord(
            **occurrence.model_dump(),
            status=status,
            local_path=decision.local_path,
            package_member=decision.package_member,
            resolution_method=decision.resolution_method,
            remote_scheme=decision.remote_scheme,
            reason_code=decision.reason_code,
            reason=decision.reason,
        )

    candidate = Path(decision.local_path)
    digest, size_bytes, error = _stable_file_hash(candidate, max_bytes=max_hash_bytes)
    if error is not None:
        status: DependencyStatus = (
            "skipped_size_limit" if "size exceeds hash limit" in error else "unresolved_local"
        )
        return UsdDependencyRecord(
            **occurrence.model_dump(),
            status=status,
            local_path=str(candidate),
            package_member=decision.package_member,
            resolution_method=decision.resolution_method,
            size_bytes=size_bytes,
            reason_code="hash_failed",
            reason=error,
        )
    if decision.expected_sha256 is not None and digest != decision.expected_sha256:
        return UsdDependencyRecord(
            **occurrence.model_dump(),
            status="unresolved_local",
            local_path=str(candidate),
            package_member=decision.package_member,
            resolution_method=decision.resolution_method,
            sha256=digest,
            size_bytes=size_bytes,
            reason_code="remap_hash_mismatch",
            reason="exact remap target does not match its approved SHA-256 digest",
        )
    return UsdDependencyRecord(
        **occurrence.model_dump(),
        status="resolved_package" if decision.package_member else "resolved_local",
        local_path=str(candidate),
        package_member=decision.package_member,
        resolution_method=decision.resolution_method,
        sha256=digest,
        size_bytes=size_bytes,
    )


def _discover_dependencies(
    root_layer: Any,
    *,
    approval_policy: DependencyApprovalPolicy,
    max_dependencies: int,
    max_hash_bytes: int,
) -> tuple[list[UsdDependencyRecord], list[str]]:
    from pxr import Ar, Sdf

    queue = [root_layer]
    visited_layers: set[str] = set()
    records: list[UsdDependencyRecord] = []
    warnings: list[str] = []
    while queue:
        layer = queue.pop(0)
        layer_identity = str(layer.realPath or layer.identifier)
        if layer_identity in visited_layers:
            continue
        visited_layers.add(layer_identity)
        for occurrence in _layer_dependencies(layer, warnings=warnings):
            if len(records) >= max_dependencies:
                warnings.append(
                    f"dependency inventory stopped at configured limit {max_dependencies}"
                )
                return records, warnings
            record = _resolve_dependency(
                occurrence,
                approval_policy=approval_policy,
                max_hash_bytes=max_hash_bytes,
            )
            records.append(record)
            if (
                occurrence.kind in {"sublayer", "reference", "payload"}
                and record.status in {"resolved_local", "resolved_package"}
                and record.local_path is not None
                and Path(record.local_path).suffix.lower() in _USD_SUFFIXES
            ):
                dependency_identifier = record.local_path
                if record.package_member:
                    dependency_identifier = str(
                        Ar.JoinPackageRelativePath(record.local_path, record.package_member)
                    )
                try:
                    dependency_layer = Sdf.Layer.FindOrOpen(dependency_identifier)
                except Exception as exc:
                    warnings.append(
                        f"could not inspect dependency layer {dependency_identifier}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    if dependency_layer is None:
                        warnings.append(
                            f"could not inspect dependency layer {dependency_identifier}: "
                            "Sdf.Layer.FindOrOpen returned None"
                        )
                    else:
                        queue.append(dependency_layer)
    return records, warnings


def _matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _value_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str | bytes):
        return 1
    try:
        return len(value)
    except TypeError:
        return 1


def _material_fact(prim: Any) -> UsdMaterialFact:
    from pxr import UsdGeom, UsdShade

    binding_relationships = {
        str(relationship.GetName()): sorted(str(path) for path in relationship.GetTargets())
        for relationship in prim.GetRelationships()
        if str(relationship.GetName()).startswith("material:binding")
    }
    computed_material_path = None
    computed_binding_status: Literal["bound", "unbound", "not_evaluated"] = "unbound"
    warnings: list[str] = []
    try:
        material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material and material.GetPrim().IsValid():
            computed_material_path = str(material.GetPath())
            computed_binding_status = "bound"
    except Exception as exc:
        computed_binding_status = "not_evaluated"
        warnings.append(f"computed material binding inspection failed: {type(exc).__name__}: {exc}")

    subsets: list[UsdMaterialSubsetFact] = []
    if prim.IsA(UsdGeom.Gprim):
        try:
            geometry = UsdGeom.Gprim(prim)
            raw_subsets = UsdGeom.Subset.GetAllGeomSubsets(geometry)
        except Exception as exc:
            raw_subsets = []
            warnings.append(f"material subset inspection failed: {type(exc).__name__}: {exc}")
        for subset in sorted(raw_subsets, key=lambda item: str(item.GetPath())):
            subset_prim = subset.GetPrim()
            subset_bindings = sorted(
                {
                    str(target)
                    for relationship in subset_prim.GetRelationships()
                    if str(relationship.GetName()).startswith("material:binding")
                    for target in relationship.GetTargets()
                }
            )
            subsets.append(
                UsdMaterialSubsetFact(
                    path=str(subset.GetPath()),
                    family_name=str(subset.GetFamilyNameAttr().Get() or "") or None,
                    element_type=str(subset.GetElementTypeAttr().Get() or "") or None,
                    index_count=_value_count(subset.GetIndicesAttr().Get()),
                    material_binding_targets=subset_bindings,
                )
            )
    return UsdMaterialFact(
        binding_relationships=binding_relationships,
        computed_material_path=computed_material_path,
        computed_binding_status=computed_binding_status,
        subsets=subsets,
        warnings=warnings,
    )


def _primvars(prim: Any) -> tuple[list[UsdPrimvarFact], list[str]]:
    from pxr import UsdGeom

    facts: list[UsdPrimvarFact] = []
    try:
        primvars = UsdGeom.PrimvarsAPI(prim).GetPrimvars()
    except Exception as exc:
        primvars = []
        warnings = [f"primvar inspection failed: {type(exc).__name__}: {exc}"]
    else:
        warnings = []
    for primvar in sorted(primvars, key=lambda item: str(item.GetPrimvarName())):
        attribute = primvar.GetAttr()
        facts.append(
            UsdPrimvarFact(
                name=str(primvar.GetPrimvarName()),
                type_name=str(attribute.GetTypeName()),
                interpolation=str(primvar.GetInterpolation() or "") or None,
                element_size=max(1, int(primvar.GetElementSize() or 1)),
                value_count=_value_count(primvar.Get()),
                index_count=_value_count(primvar.GetIndices()),
                authored=bool(attribute.HasAuthoredValueOpinion()),
            )
        )
    return facts, warnings


def _collision_fact(prim: Any) -> UsdCollisionFact:
    from pxr import UsdPhysics

    api_applied = bool(prim.HasAPI(UsdPhysics.CollisionAPI))
    enabled = None
    enabled_authored = False
    if api_applied:
        attribute = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        value = attribute.Get()
        enabled = True if value is None else bool(value)
        enabled_authored = bool(attribute.HasAuthoredValueOpinion())

    approximation = None
    approximation_authored = False
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        attribute = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        value = attribute.Get()
        approximation = str(value) if value is not None else None
        approximation_authored = bool(attribute.HasAuthoredValueOpinion())
    elif prim.HasAttribute("physics:approximation"):
        attribute = prim.GetAttribute("physics:approximation")
        value = attribute.Get()
        approximation = str(value) if value is not None else None
        approximation_authored = bool(attribute.HasAuthoredValueOpinion())

    owner_path = None
    ancestor = prim
    while ancestor and ancestor.IsValid() and not ancestor.IsPseudoRoot():
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI) or ancestor.HasAPI(
            UsdPhysics.ArticulationRootAPI
        ):
            owner_path = str(ancestor.GetPath())
            break
        ancestor = ancestor.GetParent()
    return UsdCollisionFact(
        api_applied=api_applied,
        enabled=enabled,
        enabled_authored=enabled_authored,
        approximation=approximation,
        approximation_authored=approximation_authored,
        physics_owner_path=owner_path,
    )


def _native_primitive(prim: Any) -> UsdNativePrimitiveFact | None:
    from pxr import UsdGeom

    definitions: list[tuple[type[Any], tuple[str, ...]]] = [
        (UsdGeom.Cube, ("size",)),
        (UsdGeom.Sphere, ("radius",)),
        (UsdGeom.Cylinder, ("radius", "height", "axis")),
        (UsdGeom.Capsule, ("radius", "height", "axis")),
        (UsdGeom.Cone, ("radius", "height", "axis")),
    ]
    if hasattr(UsdGeom, "Plane"):
        definitions.append((UsdGeom.Plane, ("width", "length", "axis")))
    for schema_type, names in definitions:
        if not prim.IsA(schema_type):
            continue
        schema = schema_type(prim)
        parameters: dict[str, Any] = {}
        for name in names:
            getter = getattr(schema, f"Get{name.title()}Attr")
            value = getter().Get()
            parameters[name] = str(value) if name == "axis" and value is not None else value
        return UsdNativePrimitiveFact(
            schema_type=str(prim.GetTypeName()),
            parameters=parameters,
        )
    return None


def _role(
    prim: Any,
    *,
    purpose: str | None,
    visibility: str | None,
    kind: str | None,
    material: UsdMaterialFact,
    collision: UsdCollisionFact,
) -> tuple[PrimRole, list[UsdRoleEvidence]]:
    from pxr import UsdGeom

    evidence: list[UsdRoleEvidence] = []
    if collision.api_applied:
        evidence.append(
            UsdRoleEvidence(
                signal="UsdPhysics.CollisionAPI",
                value=f"enabled={collision.enabled}",
                supports=(
                    "source_collision" if collision.enabled else "disabled_visual_collision_api"
                ),
                decisive=True,
            )
        )
        return (
            "source_collision" if collision.enabled else "disabled_visual_collision_api",
            evidence,
        )

    lowered = f"{prim.GetName()} {kind or ''}".lower()
    if purpose:
        evidence.append(UsdRoleEvidence(signal="purpose", value=purpose))
    if visibility:
        evidence.append(
            UsdRoleEvidence(
                signal="visibility",
                value=visibility,
                supports=None,
                decisive=False,
            )
        )
    if purpose == str(UsdGeom.Tokens.guide):
        evidence.append(
            UsdRoleEvidence(
                signal="purpose",
                value=purpose,
                supports="guide",
                decisive=True,
            )
        )
        return "guide", evidence
    helper_markers = ("helper", "locator", "gizmo", "debug", "guide")
    if any(marker in lowered for marker in helper_markers):
        evidence.append(
            UsdRoleEvidence(
                signal="name_or_kind",
                value=lowered.strip(),
                supports="helper",
                decisive=True,
            )
        )
        return "helper", evidence
    if prim.IsA(UsdGeom.Gprim) and purpose != str(UsdGeom.Tokens.proxy):
        material_signal = bool(
            material.computed_material_path or material.binding_relationships or material.subsets
        )
        evidence.append(
            UsdRoleEvidence(
                signal="geometry_schema",
                value=str(prim.GetTypeName()),
                supports="render",
                decisive=True,
            )
        )
        if material_signal:
            evidence.append(
                UsdRoleEvidence(
                    signal="material_evidence",
                    value="authored material binding or subset",
                    supports="render",
                )
            )
        return "render", evidence
    if prim.IsA(UsdGeom.Gprim):
        evidence.append(
            UsdRoleEvidence(
                signal="ambiguous_proxy_geometry",
                value=str(purpose),
                supports="unknown",
                decisive=True,
            )
        )
    else:
        evidence.append(
            UsdRoleEvidence(
                signal="non_geometry_schema",
                value=str(prim.GetTypeName() or "untyped"),
                supports="unknown",
                decisive=True,
            )
        )
    return "unknown", evidence


def _prim_record(prim: Any, transform_cache: Any) -> UsdPrimRecord:
    from pxr import Usd, UsdGeom

    imageable = UsdGeom.Imageable(prim) if prim.IsA(UsdGeom.Imageable) else None
    purpose = str(imageable.ComputePurpose()) if imageable else None
    visibility = str(imageable.ComputeVisibility()) if imageable else None
    kind_value = Usd.ModelAPI(prim).GetKind()
    kind = str(kind_value) if kind_value else None

    local_transform = None
    world_transform = None
    resets_xform_stack = None
    transform_time_varying = False
    if prim.IsA(UsdGeom.Xformable):
        xformable = UsdGeom.Xformable(prim)
        local_result = xformable.GetLocalTransformation(Usd.TimeCode.Default())
        if isinstance(local_result, tuple):
            local_matrix, resets_xform_stack = local_result
        else:
            local_matrix = local_result
            resets_xform_stack = bool(xformable.GetResetXformStack())
        local_transform = _matrix_rows(local_matrix)
        world_transform = _matrix_rows(transform_cache.GetLocalToWorldTransform(prim))
        transform_time_varying = bool(xformable.TransformMightBeTimeVarying())

    variant_sets = prim.GetVariantSets()
    selections = {
        name: str(variant_sets.GetVariantSelection(name))
        for name in sorted(variant_sets.GetNames())
        if variant_sets.GetVariantSelection(name)
    }
    material = _material_fact(prim)
    primvars, inspection_warnings = _primvars(prim)
    collision = _collision_fact(prim)
    role, role_evidence = _role(
        prim,
        purpose=purpose,
        visibility=visibility,
        kind=kind,
        material=material,
        collision=collision,
    )
    parent = prim.GetParent()
    return UsdPrimRecord(
        path=str(prim.GetPath()),
        name=str(prim.GetName()),
        type_name=str(prim.GetTypeName()),
        parent_path=(str(parent.GetPath()) if parent and not parent.IsPseudoRoot() else None),
        child_count=len(prim.GetChildren()),
        active=bool(prim.IsActive()),
        defined=bool(prim.IsDefined()),
        abstract=bool(prim.IsAbstract()),
        instance=bool(prim.IsInstance()),
        instance_proxy=bool(prim.IsInstanceProxy()),
        local_transform=local_transform,
        world_transform=world_transform,
        resets_xform_stack=resets_xform_stack,
        transform_time_varying=transform_time_varying,
        purpose=purpose,
        visibility=visibility,
        kind=kind,
        variant_selections=selections,
        material=material,
        primvars=primvars,
        collision=collision,
        native_primitive=_native_primitive(prim),
        role=role,
        role_evidence=role_evidence,
        inspection_warnings=inspection_warnings,
    )


def _has_geometry_time_samples(prim: Any, record: UsdPrimRecord) -> bool:
    if record.transform_time_varying:
        return True
    return any(
        str(attribute.GetName()) in _GEOMETRY_TIME_VARYING_ATTRIBUTES
        and attribute.ValueMightBeTimeVarying()
        for attribute in prim.GetAttributes()
    )


def _root_only_stage(root_layer: Any) -> Any:
    """Build an in-memory root-only view after removing composition arcs."""

    from pxr import Sdf, Usd

    isolated = Sdf.Layer.CreateAnonymous("geometry_repair_intake.usda")
    isolated.TransferContent(root_layer)
    isolated.subLayerPaths = []

    def clear_arcs(path: Any) -> None:
        spec = isolated.GetObjectAtPath(path)
        if isinstance(spec, Sdf.PrimSpec):
            spec.referenceList.ClearEdits()
            spec.payloadList.ClearEdits()

    isolated.Traverse(Sdf.Path.absoluteRootPath, clear_arcs)
    return Usd.Stage.Open(isolated, load=Usd.Stage.LoadNone)


def inventory_usd_stage(
    source_path: str | Path,
    *,
    allowed_dependency_roots: Sequence[str | Path] | None = None,
    dependency_remap_manifest: DependencyRemapInput = None,
    require_explicit_dependency_approval: bool = False,
    max_dependencies: int = 20_000,
    max_hash_bytes: int = 8 * 1024 * 1024 * 1024,
    include_instance_proxies: bool = True,
) -> UsdIntakeReport:
    """Inventory a USD source without changing or flattening it.

    Dependency hashing is restricted to the source directory unless callers add
    explicit roots. Remote identifiers are recorded and never passed to a resolver.
    Unsafe composition arcs cause prim inventory to use an isolated root-layer view;
    the report then marks composition as incomplete instead of silently omitting it.
    """

    if max_dependencies < 1:
        raise ValueError("max_dependencies must be positive")
    if max_hash_bytes < 1:
        raise ValueError("max_hash_bytes must be positive")
    source = Path(source_path).expanduser().resolve()
    if source.suffix.lower() not in _USD_SUFFIXES:
        raise ValueError(f"USD intake requires a USD source, got {source.suffix or '<none>'}")
    if not source.is_file():
        raise FileNotFoundError(f"USD intake source is not a file: {source}")
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel
    except ImportError as exc:
        raise UsdIntakeUnavailableError(f"OpenUSD Python bindings are unavailable: {exc}") from exc

    source_digest, source_size, source_hash_error = _stable_file_hash(
        source,
        max_bytes=max(max_hash_bytes, source.stat().st_size),
    )
    if source_hash_error or source_digest is None:
        raise RuntimeError(f"Could not hash USD source safely: {source_hash_error}")
    try:
        root_layer = Sdf.Layer.FindOrOpen(str(source))
    except Exception as exc:
        raise UsdIntakeUnavailableError(
            f"Could not read USD source layer {source}: {type(exc).__name__}: {exc}"
        ) from exc
    if root_layer is None:
        raise UsdIntakeUnavailableError(f"Could not read USD source layer {source}")

    approval_policy = DependencyApprovalPolicy(
        approved_roots=allowed_dependency_roots or (),
        remap_manifest=dependency_remap_manifest,
        source_directory_root=(None if require_explicit_dependency_approval else source.parent),
    )
    dependencies, warnings = _discover_dependencies(
        root_layer,
        approval_policy=approval_policy,
        max_dependencies=max_dependencies,
        max_hash_bytes=max_hash_bytes,
    )
    unsafe_composition = [
        dependency
        for dependency in dependencies
        if dependency.kind in {"sublayer", "reference", "payload"}
        and dependency.status not in {"resolved_local", "resolved_package"}
    ]
    remapped_composition = [
        dependency
        for dependency in dependencies
        if dependency.kind in {"sublayer", "reference", "payload"}
        and dependency.resolution_method == "exact_remap"
    ]
    composition_complete = not unsafe_composition and not remapped_composition
    remote_dependencies = [
        dependency for dependency in dependencies if dependency.status == "remote_not_fetched"
    ]
    stage = None
    stage_error = None
    try:
        stage = (
            Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
            if composition_complete
            else _root_only_stage(root_layer)
        )
    except Exception as exc:
        stage_error = f"{type(exc).__name__}: {exc}"
    if stage is None and stage_error is None:
        stage_error = "Usd.Stage.Open returned None"

    prims: list[UsdPrimRecord] = []
    joint_prim_paths: list[str] = []
    rigid_body_prim_paths: list[str] = []
    time_varying_prim_paths: list[str] = []
    geometry_time_varying_prim_paths: list[str] = []
    skeleton_prim_paths: list[str] = []
    animation_prim_paths: list[str] = []
    brep_prim_paths: list[str] = []
    default_prim_path = None
    meters_per_unit = None
    up_axis = None
    if stage is not None:
        default_prim = stage.GetDefaultPrim()
        default_prim_path = str(default_prim.GetPath()) if default_prim else None
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        up_axis = str(UsdGeom.GetStageUpAxis(stage))
        transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        traversal = Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies()
            if include_instance_proxies
            else Usd.PrimAllPrimsPredicate,
        )
        for prim in traversal:
            if not prim.IsValid():
                continue
            record = _prim_record(prim, transform_cache)
            prims.append(record)
            path = str(prim.GetPath())
            if prim.IsA(UsdPhysics.Joint):
                joint_prim_paths.append(path)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_prim_paths.append(path)
            if any(attribute.ValueMightBeTimeVarying() for attribute in prim.GetAttributes()):
                time_varying_prim_paths.append(path)
            if _has_geometry_time_samples(prim, record):
                geometry_time_varying_prim_paths.append(path)
            # OpenUSD distributions do not all expose the historical SkelRoot
            # Python schema class, but the authored type name remains stable.
            if prim.GetTypeName() == "SkelRoot" or prim.IsA(UsdSkel.Skeleton):
                skeleton_prim_paths.append(path)
            if prim.IsA(UsdSkel.Animation):
                animation_prim_paths.append(path)
            if prim.GetTypeName() == "BrepArray" or prim.HasAttribute("brep:regionCount"):
                brep_prim_paths.append(path)
        prims.sort(key=lambda item: item.path)
    if unsafe_composition:
        warnings.append(
            "prim inventory used an isolated root-layer view because unsafe or unresolved "
            "composition dependencies were not opened"
        )
    if remapped_composition:
        warnings.append(
            "prim inventory used an isolated root-layer view because exact remaps require "
            "a rewritten localized package before composition"
        )
    if remote_dependencies and not unsafe_composition:
        warnings.append("remote asset identifiers were recorded and never fetched")

    final_digest, _final_size, final_hash_error = _stable_file_hash(
        source,
        max_bytes=max(max_hash_bytes, source_size),
    )
    source_unchanged = final_hash_error is None and final_digest == source_digest
    if not source_unchanged:
        warnings.append("source digest changed during non-mutating intake")
    unresolved = sorted(
        {
            dependency.authored_path
            for dependency in dependencies
            if dependency.status not in {"resolved_local", "resolved_package"}
        }
    )
    return UsdIntakeReport(
        source_path=str(source),
        source_sha256=source_digest,
        source_size_bytes=source_size,
        source_unchanged=source_unchanged,
        layer_readable=True,
        stage_readable=stage is not None,
        composition_complete=composition_complete,
        stage_error=stage_error,
        default_prim_path=default_prim_path,
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
        dependencies=dependencies,
        prims=prims,
        joint_prim_paths=sorted(set(joint_prim_paths)),
        rigid_body_prim_paths=sorted(set(rigid_body_prim_paths)),
        time_varying_prim_paths=sorted(set(time_varying_prim_paths)),
        geometry_time_varying_prim_paths=sorted(set(geometry_time_varying_prim_paths)),
        skeleton_prim_paths=sorted(set(skeleton_prim_paths)),
        animation_prim_paths=sorted(set(animation_prim_paths)),
        brep_prim_paths=sorted(set(brep_prim_paths)),
        unresolved_dependencies=unresolved,
        warnings=sorted(set(warnings)),
    )


__all__ = [
    "USD_INTAKE_SCHEMA_VERSION",
    "UsdCollisionFact",
    "UsdDependencyRecord",
    "UsdIntakeReport",
    "UsdIntakeUnavailableError",
    "UsdMaterialFact",
    "UsdMaterialSubsetFact",
    "UsdNativePrimitiveFact",
    "UsdPrimRecord",
    "UsdPrimvarFact",
    "UsdRoleEvidence",
    "inventory_usd_stage",
]
