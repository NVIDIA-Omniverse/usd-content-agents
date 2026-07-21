# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, semantics-preserving hygiene for frozen Gate 3A rules."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import stat
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conform_profile import (
    GATE3A_HYGIENE_OUTPUT_DIR,
    GATE3A_HYGIENE_RECEIPT_SCHEMA_VERSION,
    GATE3A_HYGIENE_REQUIREMENT,
    _absolute_path,
    _file_sha256,
    _isa001_source_package,
    _isa001_tree_sha256,
    _private_mkdtemp,
    _publish_isa001_tree,
    _relative_to,
    _save_stage_root_layer,
    _stage_open_path,
    _validate_isa001_dependency_closure,
)

_CAMERA_SIGNATURE = {
    "clippingRange": "float2",
    "horizontalAperture": "float",
    "projection": "token",
    "verticalAperture": "float",
}


@dataclass(frozen=True)
class Gate3AHygieneResult:
    """Result returned by the standalone Gate 3A hygiene derivative."""

    status: str
    passed: bool
    reason: str
    output_path: Path
    report: dict[str, Any]
    package_root: Path | None = None


@dataclass(frozen=True)
class Gate3APhysicsInventory:
    """Canonical Joint Agent physics topology bound to a hygiene proof."""

    payload: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class _PrimvarRepair:
    prim_path: str
    attribute_name: str
    type_name: Any
    interpolation: Any
    element_size: int
    value_metadata: dict[str, Any]
    indices_metadata: dict[str, Any] | None
    compact_values: Any
    compact_indices: tuple[int, ...]
    flattened_values: Any
    original_value_count: int
    original_index_count: int


@dataclass(frozen=True)
class _ShaderIdRepair:
    prim_path: str
    layer_paths: tuple[str, ...]
    mdl_source_asset: str
    mdl_sub_identifier: str


@dataclass(frozen=True)
class _HygienePlan:
    orphan_paths: tuple[str, ...]
    primvars: tuple[_PrimvarRepair, ...]
    shader_ids: tuple[_ShaderIdRepair, ...]

    @property
    def changed(self) -> bool:
        return bool(self.orphan_paths or self.primvars or self.shader_ids)


def inspect_gate3a_physics_inventory(asset_path: Path) -> Gate3APhysicsInventory:
    """Read and hash the exact physics topology of a local USD asset."""

    try:
        from pxr import Usd, UsdPhysics
    except ImportError as exc:
        raise ValueError(f"OpenUSD physics APIs are unavailable: {exc}") from exc
    selected_root, path_error = _stage_open_path(_absolute_path(asset_path))
    if selected_root is None:
        raise ValueError(path_error or f"Unable to select a USD root: {asset_path}")
    stage = Usd.Stage.Open(str(selected_root), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"Unable to open physics inventory USD: {selected_root}")
    inventory = _physics_inventory(stage=stage, UsdPhysics=UsdPhysics)
    stage = None
    gc.collect()
    return inventory


def repair_gate3a_hygiene(
    *,
    asset_path: Path,
    package_root: Path | None = None,
    output_dir: Path,
    expected_physics_inventory_sha256: str | None,
) -> Gate3AHygieneResult:
    """Publish a deterministic derivative for three frozen Gate 3A findings."""

    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt
    except ImportError as exc:
        return _blocked_result(
            asset_path=asset_path,
            reason=f"OpenUSD Python APIs are unavailable: {exc}",
        )

    asset_path = _absolute_path(asset_path)
    output_dir = _absolute_path(output_dir)
    source_tree: Path | None = None
    source_root: Path | None = None
    extraction_dir: Path | None = None
    build_dir: Path | None = None
    report: dict[str, Any] = {
        "schema_version": GATE3A_HYGIENE_RECEIPT_SCHEMA_VERSION,
        "requirement": GATE3A_HYGIENE_REQUIREMENT,
        "asset_path": str(asset_path),
        "changes": [],
    }
    try:
        if expected_physics_inventory_sha256 is None:
            raise ValueError(
                "G3A.HYG.001 requires expected_physics_inventory_sha256 so a "
                "source or physics-stripped artifact cannot become proof."
            )
        source_tree, source_root, extraction_dir = _hygiene_source_package(
            asset_path=asset_path,
            package_root=package_root,
            output_dir=output_dir,
        )
        _require_hygiene_destination_outside_source(
            source_tree=source_tree,
            destination=output_dir,
            label="output directory",
        )
        source_asset_sha256 = _file_sha256(asset_path) if asset_path.is_file() else None
        source_root_sha256 = _file_sha256(source_root)
        source_tree_sha256 = _isa001_tree_sha256(source_tree)
        source_inventory = _tree_file_sha256(source_tree)
        source_stage = Usd.Stage.Open(str(source_root), load=Usd.Stage.LoadAll)
        if source_stage is None:
            raise ValueError(
                f"Unable to open Gate 3A hygiene source USD: {source_root}"
            )
        source_physics = _physics_inventory(
            stage=source_stage,
            UsdPhysics=UsdPhysics,
        )
        _validate_joint_agent_physics_inventory(source_physics)
        if source_physics.sha256 != expected_physics_inventory_sha256:
            raise ValueError(
                "Gate 3A hygiene input physics inventory does not match the "
                "expected SHA-256: expected "
                f"{expected_physics_inventory_sha256}, received "
                f"{source_physics.sha256}."
            )
        _validate_isa001_dependency_closure(
            stage=source_stage,
            source_root=source_root,
            source_tree=source_tree,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        plan = _build_hygiene_plan(
            stage=source_stage,
            package_root=source_tree,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
            UsdUtils=UsdUtils,
            Vt=Vt,
        )
        source_relative_root = source_root.relative_to(source_tree)
        report.update(
            {
                "source_root": str(source_root),
                "source_was_usdz": extraction_dir is not None,
                "source_asset_sha256": source_asset_sha256,
                "source_root_sha256": source_root_sha256,
                "source_tree_sha256": source_tree_sha256,
                "planned_change_count": (
                    len(plan.orphan_paths) + len(plan.primvars) + len(plan.shader_ids)
                ),
                "source_physics_inventory": source_physics.payload,
                "source_physics_inventory_sha256": source_physics.sha256,
                "expected_physics_inventory_sha256": (
                    expected_physics_inventory_sha256
                ),
            }
        )
        source_stage = None
        gc.collect()

        if not plan.changed:
            unchanged_output = asset_path if extraction_dir is not None else source_root
            report.update(
                {
                    "output_root": str(unchanged_output),
                    "output_root_sha256": _file_sha256(unchanged_output),
                    "output_tree_sha256": source_tree_sha256,
                    "remaining_findings": [],
                    "output_physics_inventory": source_physics.payload,
                    "output_physics_inventory_sha256": source_physics.sha256,
                    "physics_inventory_preserved": True,
                    "reused_output": True,
                }
            )
            return Gate3AHygieneResult(
                status="REPAIRED",
                passed=True,
                reason="The staged asset has no deterministic Gate 3A hygiene repairs.",
                output_path=unchanged_output,
                package_root=package_root,
                report=report,
            )

        publish_root = output_dir / GATE3A_HYGIENE_OUTPUT_DIR
        _require_hygiene_destination_outside_source(
            source_tree=source_tree,
            destination=publish_root,
            label="publish tree",
        )
        publish_root.mkdir(parents=True, exist_ok=True)
        build_dir = _private_mkdtemp(
            prefix=".gate3a-hygiene-build-", directory=publish_root
        )
        _require_hygiene_destination_outside_source(
            source_tree=source_tree,
            destination=build_dir,
            label="build tree",
        )
        try:
            shutil.copytree(source_tree, build_dir, dirs_exist_ok=True)
        finally:
            _normalize_private_build_tree(build_dir)
        if _isa001_tree_sha256(build_dir) != source_tree_sha256:
            raise ValueError("Gate 3A hygiene package copy changed source identity.")
        build_root = build_dir / source_relative_root
        build_stage = Usd.Stage.Open(str(build_root), load=Usd.Stage.LoadAll)
        if build_stage is None:
            raise ValueError(f"Unable to open Gate 3A hygiene build USD: {build_root}")
        changes = _apply_hygiene_plan(
            stage=build_stage,
            build_tree=build_dir,
            plan=plan,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            Vt=Vt,
        )
        save_error = _save_stage_root_layer(build_stage)
        if save_error:
            raise OSError(save_error)
        build_stage = None
        gc.collect()

        output_stage = Usd.Stage.Open(str(build_root), load=Usd.Stage.LoadAll)
        if output_stage is None:
            raise ValueError(f"Unable to open Gate 3A hygiene output USD: {build_root}")
        output_physics = _physics_inventory(
            stage=output_stage,
            UsdPhysics=UsdPhysics,
        )
        if output_physics != source_physics:
            raise ValueError(
                "Gate 3A hygiene changed the expected physics inventory: "
                f"source {source_physics.sha256}, output {output_physics.sha256}."
            )
        _validate_isa001_dependency_closure(
            stage=output_stage,
            source_root=build_root,
            source_tree=build_dir,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        _verify_hygiene_readback(
            stage=output_stage,
            plan=plan,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
            Vt=Vt,
        )
        remaining = _build_hygiene_plan(
            stage=output_stage,
            package_root=build_dir,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
            UsdUtils=UsdUtils,
            Vt=Vt,
        )
        if remaining.changed:
            raise ValueError("Gate 3A hygiene output still has repairable findings.")
        output_stage = None
        gc.collect()

        output_inventory = _tree_file_sha256(build_dir)
        output_root_sha256 = _file_sha256(build_root)
        authored_layers = {
            source_relative_root.as_posix(),
            *(layer for repair in plan.shader_ids for layer in repair.layer_paths),
        }
        changed_unowned_files = sorted(
            relative
            for relative, digest in source_inventory.items()
            if relative not in authored_layers
            and output_inventory.get(relative) != digest
        )
        if changed_unowned_files:
            raise ValueError(
                "Gate 3A hygiene changed bytes outside planned authored layers: "
                + ", ".join(changed_unowned_files[:5])
            )
        if set(output_inventory) != set(source_inventory):
            raise ValueError("Gate 3A hygiene changed the package file inventory.")
        if _isa001_tree_sha256(source_tree) != source_tree_sha256:
            raise ValueError(
                "Staged USD package changed while Gate 3A hygiene was being built."
            )
        if source_asset_sha256 is not None:
            if _file_sha256(asset_path) != source_asset_sha256:
                raise ValueError(
                    "Source asset changed while Gate 3A hygiene was being built."
                )

        output_tree_sha256 = _isa001_tree_sha256(build_dir)
        final_tree, reused_output = _publish_isa001_tree(
            build_dir=build_dir,
            publish_root=publish_root,
            tree_sha256=output_tree_sha256,
        )
        build_dir = None
        final_root = final_tree / source_relative_root
        report.update(
            {
                "changes": changes,
                "output_root": str(final_root),
                "output_root_sha256": output_root_sha256,
                "output_tree_sha256": output_tree_sha256,
                "remaining_findings": [],
                "source_identity_verified": True,
                "dependencies_preserved": True,
                "readback_verified": True,
                "output_physics_inventory": output_physics.payload,
                "output_physics_inventory_sha256": output_physics.sha256,
                "physics_inventory_preserved": True,
                "reused_output": reused_output,
            }
        )
        return Gate3AHygieneResult(
            status="REPAIRED",
            passed=True,
            reason="Published an atomic deterministic Gate 3A hygiene derivative.",
            output_path=final_root,
            package_root=final_tree,
            report=report,
        )
    except (
        IndexError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        report.update({"changes": [], "failure": str(exc)})
        return _blocked_result(asset_path=asset_path, reason=str(exc), report=report)
    finally:
        if build_dir is not None:
            shutil.rmtree(build_dir, ignore_errors=True)
        if extraction_dir is not None:
            shutil.rmtree(extraction_dir, ignore_errors=True)


def _hygiene_source_package(
    *, asset_path: Path, package_root: Path | None, output_dir: Path
) -> tuple[Path, Path, Path | None]:
    if package_root is not None:
        return _isa001_source_package(
            asset_path=asset_path,
            package_root=package_root,
            output_dir=output_dir,
        )
    try:
        return _isa001_source_package(
            asset_path=asset_path,
            package_root=output_dir / "staged",
            output_dir=output_dir,
        )
    except ValueError as exc:
        selected_root, path_error = _stage_open_path(asset_path)
        if selected_root is None:
            raise ValueError(path_error or str(exc)) from exc
        root_path = _absolute_path(selected_root)
        hygiene_root = _absolute_path(output_dir / GATE3A_HYGIENE_OUTPUT_DIR)
        relative = _relative_to(root_path, hygiene_root)
        if relative is None or len(relative.parts) < 2:
            raise
        source_tree = hygiene_root / relative.parts[0]
        if source_tree.is_symlink() or not source_tree.is_dir():
            raise ValueError(
                f"Gate 3A hygiene source tree is not a regular directory: {source_tree}"
            ) from exc
        return source_tree, root_path, None


def _physics_inventory(*, stage: Any, UsdPhysics: Any) -> Gate3APhysicsInventory:
    rigid_bodies: list[str] = []
    colliders: list[str] = []
    joints: list[dict[str, Any]] = []
    articulation_roots: list[str] = []
    filtered_pairs: list[dict[str, Any]] = []
    canonical_pairs: set[tuple[str, str]] = set()
    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim_path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(prim_path)
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append(
                {
                    "path": prim_path,
                    "type_name": prim.GetTypeName(),
                    "body0_targets": sorted(
                        str(path) for path in joint.GetBody0Rel().GetTargets()
                    ),
                    "body1_targets": sorted(
                        str(path) for path in joint.GetBody1Rel().GetTargets()
                    ),
                }
            )
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim_path)
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            targets = sorted(
                str(path)
                for path in UsdPhysics.FilteredPairsAPI(prim)
                .GetFilteredPairsRel()
                .GetTargets()
            )
            filtered_pairs.append({"body_path": prim_path, "targets": targets})
            canonical_pairs.update(
                (min(prim_path, target), max(prim_path, target))
                for target in targets
                if target != prim_path
            )
    payload: dict[str, Any] = {
        "schema_version": "content-agent-workflows.gate3a-physics-inventory.v1",
        "rigid_bodies": sorted(rigid_bodies),
        "colliders": sorted(colliders),
        "joints": sorted(joints, key=lambda item: item["path"]),
        "articulation_roots": sorted(articulation_roots),
        "filtered_pairs": sorted(
            filtered_pairs,
            key=lambda item: item["body_path"],
        ),
        "canonical_filtered_pair_relationships": [
            list(pair) for pair in sorted(canonical_pairs)
        ],
    }
    payload["counts"] = {
        "rigid_bodies": len(payload["rigid_bodies"]),
        "colliders": len(payload["colliders"]),
        "joints": len(payload["joints"]),
        "articulation_roots": len(payload["articulation_roots"]),
        "filtered_pair_bodies": len(payload["filtered_pairs"]),
        "filtered_pair_directed_targets": sum(
            len(item["targets"]) for item in payload["filtered_pairs"]
        ),
        "filtered_pair_relationships": len(
            payload["canonical_filtered_pair_relationships"]
        ),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return Gate3APhysicsInventory(
        payload=payload,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _validate_joint_agent_physics_inventory(
    inventory: Gate3APhysicsInventory,
) -> None:
    counts = inventory.payload["counts"]
    required = ("rigid_bodies", "colliders", "joints", "articulation_roots")
    missing = [name for name in required if counts[name] < 1]
    if missing:
        raise ValueError(
            "Gate 3A hygiene input is not a generated Joint Agent physics asset; "
            "the following inventory counts are zero: " + ", ".join(missing)
        )


def _build_hygiene_plan(
    *,
    stage: Any,
    package_root: Path,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdShade: Any,
    UsdUtils: Any,
    Vt: Any,
) -> _HygienePlan:
    return _HygienePlan(
        orphan_paths=tuple(
            _orphan_helper_paths(stage=stage, Sdf=Sdf, UsdUtils=UsdUtils)
        ),
        primvars=tuple(
            _primvar_repairs(
                stage=stage,
                Sdf=Sdf,
                Usd=Usd,
                UsdGeom=UsdGeom,
                Vt=Vt,
            )
        ),
        shader_ids=tuple(
            _shader_id_repairs(
                stage=stage,
                package_root=package_root,
                Sdf=Sdf,
                UsdShade=UsdShade,
            )
        ),
    )


def _orphan_helper_paths(*, stage: Any, Sdf: Any, UsdUtils: Any) -> list[str]:
    root_layer = stage.GetRootLayer()
    default_prim = stage.GetDefaultPrim()
    incoming = _incoming_prim_paths(stage=stage, Sdf=Sdf, UsdUtils=UsdUtils)
    paths: list[str] = []
    for prim in stage.GetPseudoRoot().GetAllChildren():
        if not default_prim or prim == default_prim:
            continue
        stack = prim.GetPrimStack()
        if (
            prim.HasDefiningSpecifier()
            or prim.GetTypeName()
            or len(stack) != 1
            or stack[0].layer != root_layer
            or stack[0].specifier != Sdf.SpecifierOver
            or stack[0].typeName
            or prim.GetAllChildren()
            or str(prim.GetPath()) in incoming
            or _has_composition_arcs(prim)
            or _has_outgoing_targets(prim)
            or not _has_hidden_kit_metadata(prim)
            or not _has_camera_signature(prim, Sdf=Sdf)
        ):
            continue
        paths.append(str(prim.GetPath()))
    return sorted(paths)


def _incoming_prim_paths(*, stage: Any, Sdf: Any, UsdUtils: Any) -> set[str]:
    paths: set[str] = set()
    for prim in stage.TraverseAll():
        for relationship in prim.GetRelationships():
            paths.update(
                str(target.GetPrimPath()) for target in relationship.GetTargets()
            )
        for attribute in prim.GetAttributes():
            paths.update(
                str(connection.GetPrimPath())
                for connection in attribute.GetConnections()
            )

    root_layer = stage.GetRootLayer()
    root_identifier = str(getattr(root_layer, "realPath", "") or "")
    if not root_identifier:
        raise ValueError("Gate 3A hygiene root layer has no stable local path.")
    try:
        dependency_layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(
            root_identifier
        )
    except RuntimeError as exc:
        raise ValueError(
            f"Could not inspect authored incoming helper targets: {exc}"
        ) from exc
    if unresolved:
        raise ValueError(
            "Could not inspect authored incoming helper targets because the USD "
            "dependency closure is unresolved."
        )

    layers: dict[str, Any] = {}
    for layer in [*stage.GetLayerStack(False), *dependency_layers]:
        identifier = str(
            getattr(layer, "realPath", "") or getattr(layer, "identifier", "") or ""
        )
        layers.setdefault(identifier, layer)

    def authored_targets(layer: Any) -> None:
        def inspect(path: Any) -> None:
            spec = layer.GetObjectAtPath(path)
            if isinstance(spec, Sdf.RelationshipSpec):
                values = spec.targetPathList.GetAppliedItems()
            elif isinstance(spec, Sdf.AttributeSpec):
                values = spec.connectionPathList.GetAppliedItems()
            else:
                return
            anchor = spec.path.GetPrimPath().StripAllVariantSelections()
            for value in values:
                target = value
                if not target.IsAbsolutePath():
                    target = target.MakeAbsolutePath(anchor)
                target = target.GetPrimPath().StripAllVariantSelections()
                if not target.isEmpty and target.IsAbsolutePath():
                    paths.add(str(target))

        layer.Traverse(Sdf.Path.absoluteRootPath, inspect)

    for identifier in sorted(layers):
        authored_targets(layers[identifier])
    return paths


def _has_outgoing_targets(prim: Any) -> bool:
    if any(relationship.GetTargets() for relationship in prim.GetRelationships()):
        return True
    return any(attribute.GetConnections() for attribute in prim.GetAttributes())


def _has_composition_arcs(prim: Any) -> bool:
    prim_index = prim.GetPrimIndex()
    return bool(prim_index.IsValid() and list(prim_index.rootNode.children))


def _has_hidden_kit_metadata(prim: Any) -> bool:
    return (
        prim.GetCustomDataByKey("omni:kit:hide_in_stage_window") is True
        and prim.GetCustomDataByKey("omni:kit:no_delete") is True
    )


def _has_camera_signature(prim: Any, *, Sdf: Any) -> bool:
    for name, expected_type in _CAMERA_SIGNATURE.items():
        attribute = prim.GetAttribute(name)
        if (
            not attribute
            or not attribute.HasAuthoredValue()
            or attribute.GetTypeName()
            != getattr(Sdf.ValueTypeNames, expected_type.title())
        ):
            return False
    center = prim.GetAttribute("omni:kit:centerOfInterest")
    return bool(
        center
        and center.HasAuthoredValue()
        and center.GetTypeName()
        in {Sdf.ValueTypeNames.Double3, Sdf.ValueTypeNames.Vector3d}
    )


def _primvar_repairs(
    *, stage: Any, Sdf: Any, Usd: Any, UsdGeom: Any, Vt: Any
) -> list[_PrimvarRepair]:
    skipped_types = {
        Sdf.ValueTypeNames.BoolArray,
        Sdf.ValueTypeNames.UCharArray,
        Sdf.ValueTypeNames.IntArray,
        Sdf.ValueTypeNames.UIntArray,
        Sdf.ValueTypeNames.Int64Array,
        Sdf.ValueTypeNames.UInt64Array,
    }
    repairs: list[_PrimvarRepair] = []
    for prim in stage.TraverseAll():
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        if not primvars_api:
            continue
        primvars = sorted(
            primvars_api.GetPrimvarsWithAuthoredValues(),
            key=lambda item: item.GetName(),
        )
        for primvar in primvars:
            type_name = primvar.GetTypeName()
            if primvar.GetInterpolation() == UsdGeom.Tokens.constant:
                continue
            if not type_name.isArray:
                raise ValueError(
                    "Refusing malformed non-array primvar: "
                    f"{primvar.GetAttr().GetPath()}"
                )
            if (
                type_name in skipped_types
                or primvar.GetNamespace() == "primvars:skel"
                or primvar.GetElementSize() != 1
            ):
                continue
            value_attr = primvar.GetAttr()
            indices_attr = primvar.GetIndicesAttr()
            sample_times = sorted(
                {
                    *value_attr.GetTimeSamples(),
                    *(indices_attr.GetTimeSamples() if indices_attr else []),
                }
            )
            time_codes = [Usd.TimeCode.Default()]
            time_codes.extend(Usd.TimeCode(value) for value in sample_times)
            repair_samples: list[
                tuple[Any, Any, list[Any], tuple[int, ...], Any, tuple[int, ...]]
            ] = []
            inspected = 0
            for time_code in time_codes:
                values = value_attr.Get(time_code)
                if values is None:
                    if time_code.IsDefault() and sample_times:
                        continue
                    raise ValueError(
                        f"Primvar has no readable values: {primvar.GetAttr().GetPath()}"
                    )
                inspected += 1
                if not values:
                    raise ValueError(
                        f"Primvar values are empty: {primvar.GetAttr().GetPath()}"
                    )
                unique_values, old_to_new = _compact_values(values)
                indices = _validated_primvar_indices(
                    primvar=primvar,
                    values=values,
                    time_code=time_code,
                )
                flattened = primvar.ComputeFlattened(time_code)
                if not flattened:
                    raise ValueError(
                        f"Primvar indices are invalid: {primvar.GetAttr().GetPath()}"
                    )
                validator_warns = _validator_reports_indexable(
                    primvar=primvar,
                    flattened=flattened,
                    indices=indices,
                )
                if not validator_warns:
                    continue
                if len(unique_values) == len(values):
                    raise ValueError(
                        "Validator-indexable primvar has no compactable value table: "
                        f"{primvar.GetAttr().GetPath()}"
                    )
                repair_samples.append(
                    (
                        time_code,
                        values,
                        unique_values,
                        old_to_new,
                        flattened,
                        indices,
                    )
                )
            if inspected == 0:
                raise ValueError(
                    f"Primvar has no readable values: {primvar.GetAttr().GetPath()}"
                )
            if not repair_samples:
                continue
            if (
                sample_times
                or value_attr.ValueMightBeTimeVarying()
                or (indices_attr and indices_attr.ValueMightBeTimeVarying())
            ):
                raise ValueError(
                    "Refusing time-varying primvar repair after inspecting the "
                    "default and every authored value/index sample: "
                    f"{primvar.GetAttr().GetPath()}"
                )
            (
                time_code,
                values,
                unique_values,
                old_to_new,
                flattened,
                indices,
            ) = repair_samples[0]
            if not time_code.IsDefault() or len(repair_samples) != 1:
                raise ValueError(
                    f"Refusing non-default primvar repair: {primvar.GetAttr().GetPath()}"
                )
            if prim.IsInstanceProxy() or prim.IsInPrototype():
                raise ValueError(
                    f"Refusing to author an instanced primvar: {primvar.GetAttr().GetPath()}"
                )
            variant_spec_path = _variant_scoped_primvar_spec(
                value_attr=value_attr,
                indices_attr=indices_attr,
            )
            if variant_spec_path is not None:
                raise ValueError(
                    "Refusing variant-scoped primvar repair without a "
                    f"variant-qualified edit target: {variant_spec_path}"
                )
            compact_indices = tuple(old_to_new[index] for index in indices)
            compact_values = type(values)(unique_values)
            projected = type(flattened)(
                [compact_values[index] for index in compact_indices]
            )
            if not _exact_array_equal(flattened, projected):
                raise ValueError(
                    "Primvar compaction cannot prove exact equality: "
                    f"{primvar.GetAttr().GetPath()}"
                )
            repairs.append(
                _PrimvarRepair(
                    prim_path=str(prim.GetPath()),
                    attribute_name=primvar.GetName(),
                    type_name=type_name,
                    interpolation=primvar.GetInterpolation(),
                    element_size=primvar.GetElementSize(),
                    value_metadata=dict(value_attr.GetAllMetadata()),
                    indices_metadata=(
                        dict(indices_attr.GetAllMetadata()) if indices_attr else None
                    ),
                    compact_values=compact_values,
                    compact_indices=compact_indices,
                    flattened_values=flattened,
                    original_value_count=len(values),
                    original_index_count=len(indices),
                )
            )
    return repairs


def _variant_scoped_primvar_spec(*, value_attr: Any, indices_attr: Any) -> str | None:
    for attribute in (value_attr, indices_attr):
        if not attribute:
            continue
        for spec in attribute.GetPropertyStack():
            spec_path = getattr(spec, "path", None)
            if spec_path is not None and spec_path.ContainsPrimVariantSelection():
                return str(spec_path)
    return None


def _validated_primvar_indices(
    *, primvar: Any, values: Any, time_code: Any
) -> tuple[int, ...]:
    if not primvar.IsIndexed():
        return tuple(range(len(values)))
    indices = primvar.GetIndices(time_code)
    if indices is None:
        raise ValueError(
            f"Primvar indices are unreadable: {primvar.GetAttr().GetPath()}"
        )
    validated: list[int] = []
    for index in indices:
        value = int(index)
        if value < 0 or value >= len(values):
            raise ValueError(
                f"Primvar indices are out of bounds: {primvar.GetAttr().GetPath()}"
            )
        validated.append(value)
    if not validated:
        raise ValueError(f"Primvar indices are empty: {primvar.GetAttr().GetPath()}")
    return tuple(validated)


def _validator_reports_indexable(
    *, primvar: Any, flattened: Any, indices: tuple[int, ...]
) -> bool:
    repeated_positions = _repeated_value_positions(flattened)
    if not repeated_positions:
        return False
    if not primvar.IsIndexed():
        return True
    counts = Counter(indices)
    return sum(count for count in counts.values() if count > 1) < len(
        repeated_positions
    )


def _repeated_value_positions(values: Any) -> set[int]:
    _unique, value_ids = _compact_values(values)
    first_positions: dict[int, int] = {}
    repeated: set[int] = set()
    for position, value_id in enumerate(value_ids):
        first = first_positions.setdefault(value_id, position)
        if first != position:
            repeated.update((first, position))
    return repeated


def _compact_values(values: Any) -> tuple[list[Any], tuple[int, ...]]:
    unique: list[Any] = []
    old_to_new: list[int] = []
    hashed: dict[Any, int] = {}
    for value in values:
        try:
            index = hashed.get(value)
        except TypeError:
            index = next(
                (
                    candidate
                    for candidate, existing in enumerate(unique)
                    if _exact_value_equal(value, existing)
                ),
                None,
            )
        if index is None:
            index = len(unique)
            unique.append(value)
            try:
                hashed[value] = index
            except TypeError:
                pass
        old_to_new.append(index)
    return unique, tuple(old_to_new)


def _exact_value_equal(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _exact_array_equal(left: Any, right: Any) -> bool:
    return len(left) == len(right) and all(
        _exact_value_equal(a, b) for a, b in zip(left, right, strict=True)
    )


def _shader_id_repairs(
    *, stage: Any, package_root: Path, Sdf: Any, UsdShade: Any
) -> list[_ShaderIdRepair]:
    repairs: list[_ShaderIdRepair] = []
    for prim in stage.TraverseAll():
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        implementation = shader.GetImplementationSourceAttr()
        identifier = shader.GetIdAttr()
        source_asset = prim.GetAttribute("info:mdl:sourceAsset")
        sub_identifier = prim.GetAttribute("info:mdl:sourceAsset:subIdentifier")
        if (
            not implementation.HasAuthoredValue()
            or implementation.Get() != UsdShade.Tokens.sourceAsset
            or not identifier.HasAuthoredValue()
            or not source_asset.HasAuthoredValue()
            or source_asset.GetTypeName() != Sdf.ValueTypeNames.Asset
            or not sub_identifier.HasAuthoredValue()
            or sub_identifier.GetTypeName() != Sdf.ValueTypeNames.Token
        ):
            continue
        asset_value = source_asset.Get()
        sub_identifier_value = sub_identifier.Get()
        if (
            not isinstance(asset_value, Sdf.AssetPath)
            or not asset_value.path
            or not str(asset_value.path).lower().endswith(".mdl")
            or not str(sub_identifier_value).strip()
        ):
            continue
        identifier_stack = identifier.GetPropertyStack()
        if not identifier_stack:
            raise ValueError(
                f"Stale shader ID has no authored spec: {identifier.GetPath()}"
            )
        layer_paths = tuple(
            sorted(
                {
                    _package_layer_path(spec.layer, package_root=package_root)
                    for spec in identifier_stack
                }
            )
        )
        repairs.append(
            _ShaderIdRepair(
                prim_path=str(prim.GetPath()),
                layer_paths=layer_paths,
                mdl_source_asset=str(asset_value.path),
                mdl_sub_identifier=str(sub_identifier_value),
            )
        )
    return repairs


def _apply_hygiene_plan(
    *,
    stage: Any,
    build_tree: Path,
    plan: _HygienePlan,
    Sdf: Any,
    UsdGeom: Any,
    Vt: Any,
) -> list[dict[str, Any]]:
    stage.SetEditTarget(stage.GetRootLayer())
    changes: list[dict[str, Any]] = []
    for path in plan.orphan_paths:
        if not stage.RemovePrim(path):
            raise ValueError(f"Could not remove proven Kit helper over: {path}")
        changes.append({"kind": "remove_kit_helper_over", "prim_path": path})
    for primvar_repair in plan.primvars:
        prim = stage.GetPrimAtPath(primvar_repair.prim_path)
        primvar = UsdGeom.Primvar(prim.GetAttribute(primvar_repair.attribute_name))
        if not primvar or not primvar.Set(primvar_repair.compact_values):
            raise ValueError(
                f"Could not author compact primvar values: {primvar_repair.prim_path}."
                f"{primvar_repair.attribute_name}"
            )
        if not primvar.SetIndices(Vt.IntArray(primvar_repair.compact_indices)):
            raise ValueError(
                f"Could not author compact primvar indices: {primvar_repair.prim_path}."
                f"{primvar_repair.attribute_name}"
            )
        changes.append(
            {
                "kind": "compact_primvar",
                "prim_path": primvar_repair.prim_path,
                "attribute": primvar_repair.attribute_name,
                "original_value_count": primvar_repair.original_value_count,
                "compacted_value_count": len(primvar_repair.compact_values),
                "index_count": primvar_repair.original_index_count,
            }
        )
    for shader_repair in plan.shader_ids:
        property_path = Sdf.Path(shader_repair.prim_path).AppendProperty("info:id")
        for relative_path in shader_repair.layer_paths:
            layer_path = build_tree / relative_path
            layer = Sdf.Layer.FindOrOpen(str(layer_path))
            property_spec = layer.GetPropertyAtPath(property_path) if layer else None
            if layer is None or property_spec is None:
                raise ValueError(
                    "Could not find stale shader info:id spec: "
                    f"{relative_path}:{property_path}"
                )
            prim_spec = layer.GetPrimAtPath(Sdf.Path(shader_repair.prim_path))
            if prim_spec is None:
                raise ValueError(
                    f"Could not find shader prim spec: {relative_path}:"
                    f"{shader_repair.prim_path}"
                )
            prim_spec.RemoveProperty(property_spec)
            if not layer.Save():
                raise OSError(f"Could not save Gate 3A hygiene layer: {layer_path}")
        changes.append(
            {
                "kind": "remove_stale_shader_id",
                "prim_path": shader_repair.prim_path,
                "layers": list(shader_repair.layer_paths),
            }
        )
    return changes


def _verify_hygiene_readback(
    *, stage: Any, plan: _HygienePlan, Usd: Any, UsdGeom: Any, UsdShade: Any, Vt: Any
) -> None:
    for path in plan.orphan_paths:
        if stage.GetPrimAtPath(path):
            raise ValueError(f"Removed Kit helper over remains composed: {path}")
    for primvar_repair in plan.primvars:
        prim = stage.GetPrimAtPath(primvar_repair.prim_path)
        primvar = UsdGeom.Primvar(prim.GetAttribute(primvar_repair.attribute_name))
        if not primvar:
            raise ValueError(
                "Compacted primvar is missing: "
                f"{primvar_repair.prim_path}.{primvar_repair.attribute_name}"
            )
        value_attr = primvar.GetAttr()
        indices_attr = primvar.GetIndicesAttr()
        if (
            primvar.GetTypeName() != primvar_repair.type_name
            or primvar.GetInterpolation() != primvar_repair.interpolation
            or primvar.GetElementSize() != primvar_repair.element_size
            or dict(value_attr.GetAllMetadata()) != primvar_repair.value_metadata
            or (
                primvar_repair.indices_metadata is not None
                and dict(indices_attr.GetAllMetadata())
                != primvar_repair.indices_metadata
            )
        ):
            raise ValueError(
                "Compacted primvar metadata changed: "
                f"{primvar_repair.prim_path}.{primvar_repair.attribute_name}"
            )
        if not _exact_array_equal(value_attr.Get(), primvar_repair.compact_values):
            raise ValueError(
                "Compacted primvar value table mismatch: "
                f"{primvar_repair.prim_path}.{primvar_repair.attribute_name}"
            )
        if (
            tuple(int(item) for item in primvar.GetIndices())
            != primvar_repair.compact_indices
        ):
            raise ValueError(
                "Compacted primvar index mismatch: "
                f"{primvar_repair.prim_path}.{primvar_repair.attribute_name}"
            )
        flattened = primvar.ComputeFlattened(Usd.TimeCode.EarliestTime())
        if not _exact_array_equal(flattened, primvar_repair.flattened_values):
            raise ValueError(
                "Compacted primvar changed flattened values: "
                f"{primvar_repair.prim_path}.{primvar_repair.attribute_name}"
            )
    for shader_repair in plan.shader_ids:
        shader = UsdShade.Shader(stage.GetPrimAtPath(shader_repair.prim_path))
        if not shader or shader.GetIdAttr().HasAuthoredValue():
            raise ValueError(f"Stale shader info:id remains: {shader_repair.prim_path}")
        if (
            shader.GetImplementationSource() != UsdShade.Tokens.sourceAsset
            or str(shader.GetSourceAsset("mdl").path) != shader_repair.mdl_source_asset
            or str(shader.GetSourceAssetSubIdentifier("mdl"))
            != shader_repair.mdl_sub_identifier
        ):
            raise ValueError(
                "MDL source changed while removing shader ID: "
                f"{shader_repair.prim_path}"
            )


def _package_layer_path(layer: Any, *, package_root: Path) -> str:
    identifier = str(getattr(layer, "realPath", "") or "")
    if not identifier or identifier.startswith("anon:"):
        raise ValueError(
            "Stale shader ID layer has no stable local path: "
            f"{getattr(layer, 'identifier', identifier)}"
        )
    layer_path = _absolute_path(Path(identifier))
    relative = _relative_to(layer_path, _absolute_path(package_root))
    if relative is None or layer_path.is_symlink() or not layer_path.is_file():
        raise ValueError(
            f"Stale shader ID layer is outside the copied package: {layer_path}"
        )
    return relative.as_posix()


def _tree_file_sha256(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _canonical_filesystem_location(path: Path) -> Path:
    return Path(os.path.realpath(_absolute_path(path)))


def _require_hygiene_destination_outside_source(
    *, source_tree: Path, destination: Path, label: str
) -> None:
    canonical_source = _canonical_filesystem_location(source_tree)
    canonical_destination = _canonical_filesystem_location(destination)
    if _relative_to(canonical_destination, canonical_source) is not None:
        raise ValueError(
            "Gate 3A hygiene "
            f"{label} must not equal or be nested under its source tree: "
            f"{canonical_destination}"
        )


def _normalize_private_build_tree(root: Path) -> None:
    def make_private_writable(path: Path) -> None:
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Gate 3A hygiene build tree contains a symlink: {path}")
        if stat.S_ISDIR(mode):
            owner_mode = stat.S_IRWXU
        elif stat.S_ISREG(mode):
            owner_mode = stat.S_IRUSR | stat.S_IWUSR | (mode & stat.S_IXUSR)
        else:
            raise ValueError(f"Gate 3A hygiene build tree contains a non-file: {path}")
        os.chmod(path, owner_mode, follow_symlinks=False)

    make_private_writable(root)
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        for name in directory_names:
            make_private_writable(directory_path / name)
        for name in file_names:
            make_private_writable(directory_path / name)


def _blocked_result(
    *, asset_path: Path, reason: str, report: dict[str, Any] | None = None
) -> Gate3AHygieneResult:
    payload = dict(report or {})
    payload.update(
        {
            "schema_version": GATE3A_HYGIENE_RECEIPT_SCHEMA_VERSION,
            "requirement": GATE3A_HYGIENE_REQUIREMENT,
            "asset_path": str(asset_path),
            "status": "BLOCKED",
            "passed": False,
            "reason": reason,
        }
    )
    return Gate3AHygieneResult(
        status="BLOCKED",
        passed=False,
        reason=f"Could not safely author Gate 3A hygiene: {reason}",
        output_path=asset_path,
        report=payload,
    )
