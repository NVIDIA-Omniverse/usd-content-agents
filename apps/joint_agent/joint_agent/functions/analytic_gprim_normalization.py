# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic normalization of analytic Cube render/collider ownership.

The v2 contract recognizes every structurally eligible ``UsdGeom.Cube`` that
owns ``PhysicsCollisionAPI``, including combined rigid-body/collider Cubes and
standalone collider-only Cubes. It keeps the source path, moves the analytic
collider to a deterministic child, and adds an exact canonical render mesh as
its sibling. It also removes ``PhysicsRigidBodyAPI`` from strictly proven inert
wrapper Xforms whose direct children own the real rigid bodies. Unsupported
ownership or composition fails before publication.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from joint_agent.functions.artifact_transaction import (
    StagedArtifact,
    promote_staged_artifacts,
    remove_artifact,
)

ANALYTIC_GPRIM_NORMALIZATION_VERSION = "joint-agent-analytic-gprim-normalization-v2"
ANALYTIC_GPRIM_RECEIPT_SCHEMA_VERSION = (
    "joint-agent-analytic-gprim-normalization-receipt-v2"
)
ANALYTIC_GPRIM_RENDER_CHILD_NAME = "RenderMesh"
ANALYTIC_GPRIM_COLLIDER_CHILD_NAME = "AnalyticCollider"

_RAW_USD_SUFFIXES = {".usd", ".usda", ".usdc"}
_USD_SUFFIXES = _RAW_USD_SUFFIXES | {".usdz"}
_FIXED_PACKAGE_MTIME = 315532800  # 1980-01-01T00:00:00Z, the ZIP epoch.
_ZIP_LOCAL_HEADER_SIZE = 30
_ZIP_ALIGNMENT = 64

_CANONICAL_POINTS = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
)
_CANONICAL_FACE_VERTEX_COUNTS = (4, 4, 4, 4, 4, 4)
_CANONICAL_FACE_VERTEX_INDICES = (
    0,
    3,
    2,
    1,
    4,
    5,
    6,
    7,
    0,
    1,
    5,
    4,
    1,
    2,
    6,
    5,
    2,
    3,
    7,
    6,
    3,
    0,
    4,
    7,
)
_CANONICAL_NORMALS = (
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
)

_BODY_API_SCHEMAS = {
    "PhysicsArticulationRootAPI",
    "PhysicsFilteredPairsAPI",
    "PhysicsMassAPI",
    "PhysicsRigidBodyAPI",
    "PhysxRigidBodyAPI",
}
_NEUTRAL_API_SCHEMAS = {"MaterialBindingAPI"}
_INERT_WRAPPER_ALLOWED_API_SCHEMAS = {
    "CollectionAPI",
    "IsaacLinkAPI",
    *_NEUTRAL_API_SCHEMAS,
}
_COLLIDER_API_SCHEMAS = {
    "MjcCollisionAPI",
    "NewtonCollisionAPI",
    "PhysicsCollisionAPI",
    "PhysxCollisionAPI",
}
_UNSUPPORTED_ANALYTIC_API_SCHEMAS = {
    "PhysicsMeshCollisionAPI",
    "PhysxSDFMeshCollisionAPI",
}

_BODY_PHYSICS_PROPERTIES = {
    "physics:angularVelocity",
    "physics:centerOfMass",
    "physics:density",
    "physics:diagonalInertia",
    "physics:kinematicEnabled",
    "physics:mass",
    "physics:principalAxes",
    "physics:rigidBodyEnabled",
    "physics:startsAsleep",
    "physics:velocity",
}
_MASS_PROPERTIES = {
    "physics:centerOfMass",
    "physics:density",
    "physics:diagonalInertia",
    "physics:mass",
    "physics:principalAxes",
}
_RIGID_BODY_STATE_PROPERTIES = {
    "physics:angularVelocity",
    "physics:kinematicEnabled",
    "physics:rigidBodyEnabled",
    "physics:startsAsleep",
    "physics:velocity",
}
_COLLIDER_PHYSICS_PROPERTIES = {"physics:collisionEnabled"}
_PASSIVE_WRAPPER_INVENTORY_RELATIONSHIPS = {"isaac:physics:robotLinks"}
_GEOMETRY_PROPERTIES = {"doubleSided", "extent", "orientation", "size"}
_CANONICAL_MESH_PROPERTIES = {
    "faceVertexCounts",
    "faceVertexIndices",
    "normals",
    "points",
    "subdivisionScheme",
}


class AnalyticGprimNormalizationError(ValueError):
    """Fail-closed normalization error with a stable machine reason code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class AnalyticGprimNormalizationResult:
    """Published normalization artifact and its canonical receipt."""

    output_asset_path: Path
    receipt_path: Path
    source_asset_sha256: str
    output_asset_sha256: str
    normalized_body_paths: tuple[str, ...]
    demoted_wrapper_paths: tuple[str, ...]


@dataclass(frozen=True)
class _SourceCapture:
    package_root: Path
    root_layer_path: Path
    root_entry: str
    source_container: str
    source_manifest: dict[str, Any]


@dataclass(frozen=True)
class _CubePlan:
    body_path: str
    source_kind: str
    render_mesh_path: str
    collider_path: str
    size: float
    parent_api_schemas: tuple[str, ...]
    collider_api_schemas: tuple[str, ...]
    render_property_names: tuple[str, ...]
    collider_property_names: tuple[str, ...]
    retained_body_snapshot: dict[str, Any]
    render_snapshot: dict[str, Any]
    collider_snapshot: dict[str, Any]
    source_world_corners: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class _WrapperPlan:
    path: str
    retained_api_schemas: tuple[str, ...]
    source_snapshot: dict[str, Any]


@dataclass(frozen=True)
class _NormalizationProof:
    default_prim_path: str
    stage_metadata: dict[str, Any]
    prim_inventory: tuple[tuple[str, str], ...]
    joint_graph: dict[str, Any]
    filtered_pairs: dict[str, Any]
    plans: tuple[_CubePlan, ...]
    wrapper_plans: tuple[_WrapperPlan, ...]


def normalize_analytic_cube_gprims(
    *,
    input_asset_path: str | Path,
    output_asset_path: str | Path,
    receipt_path: str | Path,
) -> AnalyticGprimNormalizationResult:
    """Publish a deterministic render/collider split for all eligible Cubes.

    Eligibility is structural and exhaustive: every active, defined Cube with
    ``PhysicsCollisionAPI`` is normalized in one transaction unless it is the
    exact analytic child of an existing canonical normalization output. Proven
    inert rigid wrapper Xforms are demoted in the same transaction. The caller
    cannot pass an asset or path allowlist.
    """

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

    source = _regular_input_asset(Path(input_asset_path))
    output = _absolute_output_path(Path(output_asset_path), label="output asset")
    receipt = _absolute_output_path(Path(receipt_path), label="receipt")
    _validate_artifact_paths(source=source, output=output, receipt=receipt)

    source_stat = source.stat()
    source_sha256 = _file_sha256(source)
    workspace_parent = output.parent
    workspace_parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(
            prefix=".analytic-gprim-normalization-",
            dir=workspace_parent,
        )
    )
    staged_output = workspace / f"normalized{output.suffix.lower()}"
    staged_receipt = workspace / "receipt.json"
    stage: Any | None = None
    try:
        capture = _capture_source(
            source=source,
            source_sha256=source_sha256,
            workspace=workspace,
        )
        stage = Usd.Stage.Open(str(capture.root_layer_path), load=Usd.Stage.LoadAll)
        if stage is None:
            _fail("invalid_source_stage", f"could not open {source}")
        _validate_stage_composition(stage)
        _validate_dependency_closure(
            root_layer_path=capture.root_layer_path,
            package_root=capture.package_root,
            expected_files=set(capture.source_manifest["entry_paths"]),
            UsdUtils=UsdUtils,
        )
        proof = _preflight_normalization(
            stage,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )

        _apply_normalization(
            stage,
            proof.plans,
            proof.wrapper_plans,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            Vt=Vt,
        )
        _validate_normalized_stage(
            stage,
            proof,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        if not stage.GetRootLayer().Save():
            _fail("output_write_failed", "OpenUSD did not save the normalized root")
        stage = None
        gc.collect()

        _write_output_asset(
            root_layer_path=capture.root_layer_path,
            package_root=capture.package_root,
            staged_output=staged_output,
            output_suffix=output.suffix.lower(),
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        output_manifest = _asset_manifest(staged_output)
        output_sha256 = _file_sha256(staged_output)

        readback = Usd.Stage.Open(str(staged_output), load=Usd.Stage.LoadAll)
        if readback is None:
            _fail("output_readback_failed", f"could not reopen {staged_output}")
        _validate_normalized_stage(
            readback,
            proof,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        if output.suffix.lower() in _RAW_USD_SUFFIXES:
            _validate_dependency_closure(
                root_layer_path=staged_output,
                package_root=staged_output.parent,
                expected_files={staged_output.name},
                UsdUtils=UsdUtils,
            )
        readback = None
        gc.collect()

        _require_source_unchanged(
            source,
            expected_sha256=source_sha256,
            expected_stat=source_stat,
        )
        receipt_payload = _receipt_payload(
            source_sha256=source_sha256,
            source_capture=capture,
            output_sha256=output_sha256,
            output_manifest=output_manifest,
            proof=proof,
        )
        _write_canonical_json(staged_receipt, receipt_payload)

        def validate_source_before_commit() -> None:
            _require_source_unchanged(
                source,
                expected_sha256=source_sha256,
                expected_stat=source_stat,
            )
            if _file_sha256(staged_output) != output_sha256:
                _fail("staged_output_changed", "staged output bytes changed")

        promote_staged_artifacts(
            [
                StagedArtifact(
                    staged_path=staged_receipt,
                    target_path=receipt,
                    label="analytic Gprim normalization receipt",
                    replace_existing=False,
                ),
                StagedArtifact(
                    staged_path=staged_output,
                    target_path=output,
                    label="analytic Gprim normalized asset",
                    replace_existing=False,
                ),
            ],
            prebackup_validator=validate_source_before_commit,
            precommit_validator=validate_source_before_commit,
        )
        return AnalyticGprimNormalizationResult(
            output_asset_path=output,
            receipt_path=receipt,
            source_asset_sha256=source_sha256,
            output_asset_sha256=output_sha256,
            normalized_body_paths=tuple(plan.body_path for plan in proof.plans),
            demoted_wrapper_paths=tuple(plan.path for plan in proof.wrapper_plans),
        )
    except AnalyticGprimNormalizationError:
        raise
    except Exception as exc:
        raise AnalyticGprimNormalizationError(
            "normalization_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    finally:
        stage = None
        gc.collect()
        remove_artifact(staged_output)
        remove_artifact(staged_receipt)
        shutil.rmtree(workspace, ignore_errors=True)


def _regular_input_asset(path: Path) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        _fail("source_missing", f"source asset does not exist: {absolute}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("source_not_regular", f"source must be a regular non-symlink: {absolute}")
    if absolute.suffix.lower() not in _USD_SUFFIXES:
        _fail(
            "unsupported_asset_format", f"unsupported source suffix: {absolute.suffix}"
        )
    return absolute


def _absolute_output_path(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    if label == "output asset" and absolute.suffix.lower() not in _USD_SUFFIXES:
        _fail(
            "unsupported_asset_format", f"unsupported output suffix: {absolute.suffix}"
        )
    if label == "receipt" and absolute.suffix.lower() != ".json":
        _fail("invalid_receipt_path", "receipt path must use the .json suffix")
    return absolute


def _validate_artifact_paths(*, source: Path, output: Path, receipt: Path) -> None:
    identities = {
        source.resolve(strict=True),
        output.resolve(strict=False),
        receipt.resolve(strict=False),
    }
    if len(identities) != 3:
        _fail("artifact_path_collision", "source, output, and receipt must be distinct")
    for target, label in ((output, "output asset"), (receipt, "receipt")):
        if target.exists() or target.is_symlink():
            _fail("output_exists", f"refusing to replace existing {label}: {target}")


def _capture_source(
    *, source: Path, source_sha256: str, workspace: Path
) -> _SourceCapture:
    package_root = workspace / "package"
    package_root.mkdir()
    if source.suffix.lower() == ".usdz":
        captured_package = workspace / "source.usdz"
        shutil.copyfile(source, captured_package)
        if _file_sha256(captured_package) != source_sha256:
            _fail("source_capture_mismatch", "captured USDZ bytes differ from source")
        manifest = _extract_usdz_without_size_cap(captured_package, package_root)
        root_entry = manifest["root_entry"]
        return _SourceCapture(
            package_root=package_root,
            root_layer_path=package_root.joinpath(*PurePosixPath(root_entry).parts),
            root_entry=root_entry,
            source_container="usdz",
            source_manifest=manifest,
        )

    captured_root = package_root / source.name
    shutil.copyfile(source, captured_root)
    if _file_sha256(captured_root) != source_sha256:
        _fail("source_capture_mismatch", "captured USD bytes differ from source")
    entry = {
        "path": captured_root.name,
        "size": captured_root.stat().st_size,
        "sha256": _file_sha256(captured_root),
    }
    return _SourceCapture(
        package_root=package_root,
        root_layer_path=captured_root,
        root_entry=captured_root.name,
        source_container="raw_usd",
        source_manifest={
            "container": "raw_usd",
            "root_entry": captured_root.name,
            "entry_paths": [captured_root.name],
            "entries": [entry],
            "dependency_bundle_sha256": _canonical_json_sha256([entry]),
        },
    )


def _extract_usdz_without_size_cap(
    package_path: Path, destination: Path
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    normalized_names: set[str] = set()
    file_names: set[str] = set()
    try:
        with zipfile.ZipFile(package_path) as archive:
            infos = archive.infolist()
            if not infos:
                _fail("invalid_usdz", "USDZ package is empty")
            root_info = infos[0]
            root_parts = _validate_usdz_info(root_info, require_file=True)
            root_entry = "/".join(root_parts)
            if Path(root_entry).suffix.lower() not in _RAW_USD_SUFFIXES:
                _fail("invalid_usdz", "USDZ first entry must be a USD root layer")
            for info in infos:
                parts = _validate_usdz_info(info, require_file=False)
                normalized = "/".join(parts)
                if normalized in normalized_names:
                    _fail("invalid_usdz", f"duplicate normalized entry: {normalized}")
                normalized_names.add(normalized)
                for depth in range(1, len(parts)):
                    ancestor = "/".join(parts[:depth])
                    if ancestor in file_names:
                        _fail(
                            "invalid_usdz",
                            f"file/member ancestor collision: {ancestor}",
                        )
                target = destination.joinpath(*parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                file_names.add(normalized)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with (
                    archive.open(info) as source_stream,
                    target.open("xb") as output_stream,
                ):
                    for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                        output_stream.write(chunk)
                        digest.update(chunk)
                entries.append(
                    {
                        "path": normalized,
                        "size": target.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise AnalyticGprimNormalizationError("invalid_usdz", str(exc)) from exc
    ordered_entries = sorted(entries, key=lambda item: item["path"])
    return {
        "container": "usdz",
        "root_entry": root_entry,
        "entry_paths": [entry["path"] for entry in ordered_entries],
        "entries": ordered_entries,
        "dependency_bundle_sha256": _canonical_json_sha256(ordered_entries),
    }


def _validate_usdz_info(
    info: zipfile.ZipInfo,
    *,
    require_file: bool,
) -> tuple[str, ...]:
    name = info.filename
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail("invalid_usdz", f"unsafe USDZ entry path: {name!r}")
    if require_file and info.is_dir():
        _fail("invalid_usdz", "USDZ first entry cannot be a directory")
    if info.flag_bits & 0x1:
        _fail("invalid_usdz", f"encrypted USDZ entry: {name}")
    mode = (info.external_attr >> 16) & 0o170000
    if mode == stat.S_IFLNK:
        _fail("invalid_usdz", f"symbolic-link USDZ entry: {name}")
    if not info.is_dir():
        if info.compress_type != zipfile.ZIP_STORED:
            _fail("invalid_usdz", f"compressed USDZ entry: {name}")
        data_offset = (
            info.header_offset
            + _ZIP_LOCAL_HEADER_SIZE
            + len(name.encode("utf-8"))
            + len(info.extra)
        )
        if data_offset % _ZIP_ALIGNMENT:
            _fail("invalid_usdz", f"unaligned USDZ entry: {name}")
    return tuple(path.parts)


def _validate_stage_composition(stage: Any) -> None:
    root_layer = stage.GetRootLayer()
    if root_layer.subLayerPaths:
        _fail("unsupported_composition", "sublayers are outside the v1 contract")
    if root_layer.HasRelocates():
        _fail("unsupported_composition", "relocates are outside the v1 contract")
    used_layers = [layer for layer in stage.GetUsedLayers() if not layer.anonymous]
    if len(used_layers) != 1 or used_layers[0] != root_layer:
        _fail("unsupported_composition", "the source must compose from one root layer")
    for prim in _traverse_all(stage):
        prim_stack = prim.GetPrimStack()
        if any(spec.layer != root_layer for spec in prim_stack):
            _fail("unsupported_composition", f"multi-layer prim at {prim.GetPath()}")


def _reject_candidate_composition(prim: Any) -> None:
    """Reject composition only where it can influence an edited body prim."""

    root_layer = prim.GetStage().GetRootLayer()
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if (
            current.IsInstance()
            or current.IsInstanceProxy()
            or current.IsInstanceable()
        ):
            _fail(
                "unsupported_instance",
                f"candidate composition contains an instance at {current.GetPath()}",
            )
        if current.HasVariantSets() or current.GetVariantSets().GetNames():
            _fail(
                "unsupported_variants",
                f"candidate composition contains variants at {current.GetPath()}",
            )
        if (
            current.HasAuthoredReferences()
            or current.HasAuthoredPayloads()
            or current.HasAuthoredInherits()
            or current.HasAuthoredSpecializes()
        ):
            _fail("unsupported_composition", f"composition arc at {current.GetPath()}")
        if any(
            str(key).startswith("clips") for key in current.GetAllAuthoredMetadata()
        ):
            _fail("unsupported_composition", f"value clips at {current.GetPath()}")
        prim_stack = current.GetPrimStack()
        if any(spec.layer != root_layer for spec in prim_stack):
            _fail("unsupported_composition", f"multi-layer prim at {current.GetPath()}")
        current = current.GetParent()


def _validate_dependency_closure(
    *,
    root_layer_path: Path,
    package_root: Path,
    expected_files: set[str],
    UsdUtils: Any,
) -> None:
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            str(root_layer_path)
        )
    except Exception as exc:
        raise AnalyticGprimNormalizationError(
            "dependency_inventory_failed",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if unresolved:
        _fail(
            "unresolved_dependency",
            f"unresolved dependencies: {sorted(map(str, unresolved))}",
        )
    package_root_resolved = package_root.resolve(strict=True)
    resolved_files: set[str] = set()
    for item in (*layers, *assets):
        candidate = _resolved_dependency_path(item, base=root_layer_path.parent)
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(package_root_resolved)
        except (FileNotFoundError, ValueError) as exc:
            raise AnalyticGprimNormalizationError(
                "outside_dependency",
                f"dependency is not a regular in-package file: {candidate}",
            ) from exc
        metadata = resolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail("outside_dependency", f"dependency is not a regular file: {resolved}")
        resolved_files.add(relative.as_posix())
    if resolved_files != expected_files:
        _fail(
            "unsupported_package_contents",
            "package files must exactly equal the resolved dependency closure: "
            f"resolved={sorted(resolved_files)}, package={sorted(expected_files)}",
        )


def _resolved_dependency_path(item: Any, *, base: Path) -> Path:
    candidate_text = (
        str(getattr(item, "realPath", ""))
        or str(getattr(item, "resolvedPath", ""))
        or str(getattr(item, "path", ""))
        or str(getattr(item, "identifier", item))
    )
    candidate = Path(candidate_text)
    return candidate if candidate.is_absolute() else base / candidate


def _preflight_normalization(
    stage: Any,
    *,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> _NormalizationProof:
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        _fail("invalid_default_prim", "source stage must have a valid default prim")
    prims = _traverse_all(stage)
    candidates = [
        prim
        for prim in prims
        if prim.IsActive()
        and prim.IsDefined()
        and prim.IsA(UsdGeom.Cube)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and not _is_existing_normalized_collider(
            prim,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
        )
    ]
    candidate_paths = {str(prim.GetPath()) for prim in candidates}
    body_candidate_paths = {
        str(prim.GetPath())
        for prim in candidates
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    _reject_ambiguous_external_targets(
        stage,
        candidate_paths,
        body_candidate_paths=body_candidate_paths,
        UsdPhysics=UsdPhysics,
    )
    plans = tuple(
        _preflight_cube(
            stage,
            prim,
            candidate_paths=candidate_paths,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        for prim in sorted(candidates, key=lambda item: str(item.GetPath()))
    )
    joint_endpoint_paths = _joint_endpoint_paths(stage, UsdPhysics=UsdPhysics)
    externally_targeted_paths = _externally_targeted_prim_paths(stage)
    wrapper_plans = tuple(
        _preflight_wrapper(prim, Usd=Usd, UsdGeom=UsdGeom)
        for prim in sorted(prims, key=lambda item: str(item.GetPath()))
        if _is_provably_inert_wrapper(
            prim,
            default_prim_path=str(default_prim.GetPath()),
            joint_endpoint_paths=joint_endpoint_paths,
            externally_targeted_paths=externally_targeted_paths,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
        )
    )
    if not plans and not wrapper_plans:
        _fail(
            "no_eligible_cubes",
            "stage has no eligible analytic Cubes or inert rigid wrappers",
        )
    return _NormalizationProof(
        default_prim_path=str(default_prim.GetPath()),
        stage_metadata=_stage_metadata_snapshot(stage),
        prim_inventory=_prim_inventory(stage),
        joint_graph=_joint_graph_snapshot(stage, UsdPhysics=UsdPhysics),
        filtered_pairs=_filtered_pairs_snapshot(stage),
        plans=plans,
        wrapper_plans=wrapper_plans,
    )


def _is_existing_normalized_collider(
    prim: Any,
    *,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> bool:
    if str(prim.GetName()) != ANALYTIC_GPRIM_COLLIDER_CHILD_NAME:
        return False
    parent = prim.GetParent()
    render = parent.GetChild(ANALYTIC_GPRIM_RENDER_CHILD_NAME)
    if not render or not render.IsValid():
        return False

    size = _cube_size(prim, UsdGeom=UsdGeom)
    mesh = UsdGeom.Mesh(render)
    operations = UsdGeom.Xformable(render).GetOrderedXformOps() if mesh else []
    expected_scale = (size / 2.0, size / 2.0, size / 2.0)
    collider_api_bases = {
        schema.split(":", maxsplit=1)[0] for schema in _authored_api_schema_tokens(prim)
    }
    canonical = bool(
        parent.IsA(UsdGeom.Xform)
        and not parent.IsA(UsdGeom.Gprim)
        and mesh
        and _usd_value(mesh.GetPointsAttr().Get())
        == [list(value) for value in _CANONICAL_POINTS]
        and _usd_value(mesh.GetFaceVertexCountsAttr().Get())
        == list(_CANONICAL_FACE_VERTEX_COUNTS)
        and _usd_value(mesh.GetFaceVertexIndicesAttr().Get())
        == list(_CANONICAL_FACE_VERTEX_INDICES)
        and _usd_value(mesh.GetNormalsAttr().Get())
        == [list(value) for value in _CANONICAL_NORMALS]
        and str(mesh.GetNormalsInterpolation()) == str(UsdGeom.Tokens.uniform)
        and str(mesh.GetOrientationAttr().Get()) == str(UsdGeom.Tokens.rightHanded)
        and _usd_value(mesh.GetExtentAttr().Get())
        == [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
        and str(mesh.GetSubdivisionSchemeAttr().Get()) == str(UsdGeom.Tokens.none)
        and len(operations) == 1
        and operations[0].GetOpType() == UsdGeom.XformOp.TypeScale
        and tuple(float(value) for value in operations[0].Get()) == expected_scale
        and not UsdGeom.Xformable(render).GetResetXformStack()
        and str(UsdGeom.Imageable(prim).ComputePurpose()) == str(UsdGeom.Tokens.guide)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and collider_api_bases.issubset({*_COLLIDER_API_SCHEMAS, "PhysicsMassAPI"})
        and not _authored_api_schema_tokens(render)
        and not render.HasAPI(UsdPhysics.CollisionAPI)
        and not parent.HasAPI(UsdPhysics.CollisionAPI)
    )
    if not canonical:
        _fail(
            "malformed_existing_normalization",
            f"reserved normalization children are not canonical at {parent.GetPath()}",
        )
    return True


def _is_provably_inert_wrapper(
    prim: Any,
    *,
    default_prim_path: str,
    joint_endpoint_paths: set[str],
    externally_targeted_paths: set[str],
    UsdGeom: Any,
    UsdPhysics: Any,
) -> bool:
    if (
        not prim.IsActive()
        or not prim.IsDefined()
        or not prim.IsA(UsdGeom.Xform)
        or prim.IsA(UsdGeom.Gprim)
        or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        return False
    path = str(prim.GetPath())
    if (
        path == default_prim_path
        or path in joint_endpoint_paths
        or path in externally_targeted_paths
        or prim.IsInstance()
        or prim.IsInstanceProxy()
    ):
        return False

    schemas = _authored_api_schema_tokens(prim)
    if "PhysicsRigidBodyAPI" not in schemas:
        return False
    for schema in schemas:
        base = schema.split(":", maxsplit=1)[0]
        if base == "PhysicsRigidBodyAPI":
            continue
        if base not in _INERT_WRAPPER_ALLOWED_API_SCHEMAS:
            return False
    if any(
        str(prop.GetName()).startswith(("physics:", "physx"))
        for prop in prim.GetAuthoredProperties()
    ):
        return False

    children = list(prim.GetAllChildren())
    return bool(children) and all(
        child.IsActive()
        and child.IsDefined()
        and not child.IsInstance()
        and not child.IsInstanceProxy()
        and child.HasAPI(UsdPhysics.RigidBodyAPI)
        for child in children
    )


def _preflight_wrapper(prim: Any, *, Usd: Any, UsdGeom: Any) -> _WrapperPlan:
    _reject_candidate_composition(prim)
    schemas = _authored_api_schema_tokens(prim)
    retained_schemas = tuple(
        schema for schema in schemas if schema != "PhysicsRigidBodyAPI"
    )
    return _WrapperPlan(
        path=str(prim.GetPath()),
        retained_api_schemas=retained_schemas,
        source_snapshot=_wrapper_semantics_snapshot(prim, Usd=Usd, UsdGeom=UsdGeom),
    )


def _preflight_cube(
    stage: Any,
    prim: Any,
    *,
    candidate_paths: set[str],
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> _CubePlan:
    path = str(prim.GetPath())
    _reject_candidate_composition(prim)
    render_path = f"{path}/{ANALYTIC_GPRIM_RENDER_CHILD_NAME}"
    collider_path = f"{path}/{ANALYTIC_GPRIM_COLLIDER_CHILD_NAME}"
    for reserved in (render_path, collider_path):
        if stage.GetPrimAtPath(reserved):
            _fail(
                "reserved_path_collision", f"reserved child already exists: {reserved}"
            )
    for descendant in _descendants(prim):
        if descendant.IsA(UsdGeom.Gprim):
            _fail(
                "nested_gprim",
                f"nested Gprim under eligible Cube: {descendant.GetPath()}",
            )
    if any(other != path and other.startswith(f"{path}/") for other in candidate_paths):
        _fail("nested_gprim", f"eligible Cube contains another eligible Cube: {path}")

    _reject_time_varying_or_connected_candidate(prim, UsdGeom=UsdGeom)
    size = _cube_size(prim, UsdGeom=UsdGeom)
    orientation = str(UsdGeom.Gprim(prim).GetOrientationAttr().Get())
    if orientation != str(UsdGeom.Tokens.rightHanded):
        _fail(
            "unsupported_orientation", f"Cube orientation must be rightHanded: {path}"
        )
    imageable = UsdGeom.Imageable(prim)
    if str(imageable.ComputeEffectiveVisibility()) == str(UsdGeom.Tokens.invisible):
        _fail("nonrenderable_source", f"eligible Cube is effectively invisible: {path}")
    if str(imageable.ComputePurpose()) == str(UsdGeom.Tokens.guide):
        _fail("nonrenderable_source", f"eligible Cube has guide purpose: {path}")

    source_schemas = _authored_api_schema_tokens(prim)
    source_kind = (
        "combined_body_collider"
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        else "collider_only"
    )
    if source_kind == "collider_only" and not _has_rigid_body_ancestor(
        prim,
        UsdPhysics=UsdPhysics,
    ):
        _fail(
            "ambiguous_api_ownership",
            f"collider-only Cube has no rigid-body ancestor: {path}",
        )
    parent_schemas, collider_schemas = _classify_api_schemas(
        path,
        source_schemas,
        source_kind=source_kind,
    )
    render_properties: list[str] = []
    collider_properties: list[str] = []
    retained_properties: list[str] = []
    for prop in prim.GetAuthoredProperties():
        name = str(prop.GetName())
        if name.startswith("primvars:"):
            render_properties.append(name)
        elif name in _GEOMETRY_PROPERTIES:
            continue
        elif source_kind == "collider_only" and (
            name in _RIGID_BODY_STATE_PROPERTIES or name.startswith("physxRigidBody:")
        ):
            _fail(
                "ambiguous_api_ownership",
                f"collider-only Cube owns rigid-body state at {path}: {name}",
            )
        elif source_kind == "collider_only" and name in _MASS_PROPERTIES:
            collider_properties.append(name)
        elif name in _COLLIDER_PHYSICS_PROPERTIES or name.startswith(
            ("mjc:", "newton:", "physxCollision:")
        ):
            collider_properties.append(name)
        elif (
            name.startswith("physics:")
            and name not in _BODY_PHYSICS_PROPERTIES
            and name != "physics:filteredPairs"
        ):
            _fail(
                "ambiguous_api_ownership",
                f"unclassified physics property at {path}: {name}",
            )
        elif name.startswith("physx") and not name.startswith("physxRigidBody:"):
            _fail(
                "ambiguous_api_ownership",
                f"unclassified PhysX property at {path}: {name}",
            )
        elif name in _CANONICAL_MESH_PROPERTIES:
            _fail(
                "ambiguous_geometry", f"unexpected mesh property on Cube {path}: {name}"
            )
        else:
            retained_properties.append(name)

    _validate_effective_primvars(prim, UsdGeom=UsdGeom)
    source_world_corners = _cube_world_corners(
        prim,
        size=size,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )
    return _CubePlan(
        body_path=path,
        source_kind=source_kind,
        render_mesh_path=render_path,
        collider_path=collider_path,
        size=size,
        parent_api_schemas=parent_schemas,
        collider_api_schemas=collider_schemas,
        render_property_names=tuple(sorted(render_properties)),
        collider_property_names=tuple(sorted(collider_properties)),
        retained_body_snapshot=_retained_body_snapshot(
            prim,
            retained_property_names=retained_properties,
            retained_api_schemas=parent_schemas,
            Usd=Usd,
            UsdGeom=UsdGeom,
        ),
        render_snapshot=_render_semantics_snapshot(
            prim,
            UsdGeom=UsdGeom,
            UsdShade=UsdShade,
        ),
        collider_snapshot=_collider_semantics_snapshot(
            prim,
            size=size,
            property_names=collider_properties,
            api_schemas=collider_schemas,
            UsdShade=UsdShade,
        ),
        source_world_corners=source_world_corners,
    )


def _classify_api_schemas(
    path: str,
    schemas: Sequence[str],
    *,
    source_kind: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parent: list[str] = []
    collider: list[str] = []
    for schema in schemas:
        base = schema.split(":", maxsplit=1)[0]
        if base in _UNSUPPORTED_ANALYTIC_API_SCHEMAS:
            _fail(
                "analytic_collision_required",
                f"unsupported mesh collision API at {path}: {schema}",
            )
        if base in _COLLIDER_API_SCHEMAS or (
            source_kind == "collider_only" and base == "PhysicsMassAPI"
        ):
            collider.append(schema)
        elif (
            base in _BODY_API_SCHEMAS
            or base in _NEUTRAL_API_SCHEMAS
            or base == "CollectionAPI"
        ):
            parent.append(schema)
        else:
            _fail(
                "ambiguous_api_ownership",
                f"unsupported API ownership at {path}: {schema}",
            )
    has_rigid_body = "PhysicsRigidBodyAPI" in parent
    if (
        "PhysicsCollisionAPI" not in collider
        or (source_kind == "combined_body_collider" and not has_rigid_body)
        or (source_kind == "collider_only" and has_rigid_body)
    ):
        _fail(
            "ambiguous_api_ownership",
            f"eligible Cube APIs changed during preflight: {path}",
        )
    if source_kind == "collider_only" and any(
        schema.split(":", maxsplit=1)[0]
        in {
            "PhysicsArticulationRootAPI",
            "PhysicsFilteredPairsAPI",
            "PhysxRigidBodyAPI",
        }
        for schema in parent
    ):
        _fail(
            "ambiguous_api_ownership",
            f"collider-only Cube owns rigid-body state at {path}",
        )
    return tuple(parent), tuple(collider)


def _has_rigid_body_ancestor(prim: Any, *, UsdPhysics: Any) -> bool:
    current = prim.GetParent()
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        current = current.GetParent()
    return False


def _reject_time_varying_or_connected_candidate(prim: Any, *, UsdGeom: Any) -> None:
    path = prim.GetPath()
    for attr in prim.GetAttributes():
        if attr.GetTimeSamples():
            _fail(
                "time_varying_input", f"time-sampled input at {path}.{attr.GetName()}"
            )
        if attr.HasAuthoredConnections():
            _fail("connected_input", f"connected input at {path}.{attr.GetName()}")
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        xformable = UsdGeom.Xformable(current)
        if xformable:
            attrs = [
                xformable.GetXformOpOrderAttr(),
                *(operation.GetAttr() for operation in xformable.GetOrderedXformOps()),
            ]
            for attr in attrs:
                if attr.GetTimeSamples():
                    _fail(
                        "time_varying_input",
                        f"time-sampled transform at {current.GetPath()}",
                    )
                if attr.HasAuthoredConnections():
                    _fail(
                        "connected_input", f"connected transform at {current.GetPath()}"
                    )
        visibility = UsdGeom.Imageable(current).GetVisibilityAttr()
        if visibility and visibility.GetTimeSamples():
            _fail(
                "time_varying_input", f"time-sampled visibility at {current.GetPath()}"
            )
        current = current.GetParent()


def _cube_size(prim: Any, *, UsdGeom: Any) -> float:
    attr = UsdGeom.Cube(prim).GetSizeAttr()
    if str(attr.GetTypeName()) != "double":
        _fail("malformed_size", f"Cube size has the wrong type at {prim.GetPath()}")
    value = attr.Get()
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail("malformed_size", f"Cube size is not numeric at {prim.GetPath()}")
    size = float(value)
    if not math.isfinite(size) or size <= 0.0:
        _fail(
            "malformed_size",
            f"Cube size must be positive and finite at {prim.GetPath()}",
        )
    return size


def _validate_effective_primvars(prim: Any, *, UsdGeom: Any) -> None:
    for primvar in UsdGeom.PrimvarsAPI(prim).FindPrimvarsWithInheritance():
        attr = primvar.GetAttr()
        if not attr.HasAuthoredValueOpinion():
            continue
        if attr.GetTimeSamples() or attr.HasAuthoredConnections():
            _fail(
                "unsupported_primvar",
                f"dynamic primvar at {prim.GetPath()}: {primvar.GetName()}",
            )
        if str(primvar.GetInterpolation()) != str(UsdGeom.Tokens.constant):
            _fail(
                "unsupported_primvar",
                f"only constant primvars are supported at {prim.GetPath()}: {primvar.GetName()}",
            )
        indices = primvar.GetIndicesAttr()
        if indices and (indices.GetTimeSamples() or indices.HasAuthoredConnections()):
            _fail(
                "unsupported_primvar",
                f"dynamic primvar indices at {prim.GetPath()}: {primvar.GetName()}",
            )


def _joint_endpoint_paths(stage: Any, *, UsdPhysics: Any) -> set[str]:
    endpoints: set[str] = set()
    for prim in _traverse_all(stage):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        for name in ("physics:body0", "physics:body1"):
            relationship = prim.GetRelationship(name)
            if not relationship:
                continue
            endpoints.update(
                str(target.GetPrimPath()) for target in relationship.GetTargets()
            )
    return endpoints


def _externally_targeted_prim_paths(stage: Any) -> set[str]:
    targets: set[str] = set()
    for prim in _traverse_all(stage):
        for relationship in prim.GetRelationships():
            if str(relationship.GetName()) in _PASSIVE_WRAPPER_INVENTORY_RELATIONSHIPS:
                continue
            targets.update(
                str(target.GetPrimPath()) for target in relationship.GetTargets()
            )
        for attribute in prim.GetAttributes():
            targets.update(
                str(connection.GetPrimPath())
                for connection in attribute.GetConnections()
            )
    return targets


def _reject_ambiguous_external_targets(
    stage: Any,
    candidate_paths: set[str],
    *,
    body_candidate_paths: set[str],
    UsdPhysics: Any,
) -> None:
    for prim in _traverse_all(stage):
        is_joint = prim.IsA(UsdPhysics.Joint)
        has_filtered_pairs = "PhysicsFilteredPairsAPI" in {
            str(token).split(":", maxsplit=1)[0] for token in prim.GetAppliedSchemas()
        }
        for relationship in prim.GetRelationships():
            name = str(relationship.GetName())
            for target in relationship.GetTargets():
                target_prim = str(target.GetPrimPath())
                if target_prim not in candidate_paths:
                    continue
                if (
                    target_prim in body_candidate_paths
                    and is_joint
                    and name in {"physics:body0", "physics:body1"}
                ):
                    continue
                if (
                    target_prim in body_candidate_paths
                    and has_filtered_pairs
                    and name == "physics:filteredPairs"
                ):
                    continue
                if str(prim.GetPath()) == target_prim:
                    continue
                _fail(
                    "ambiguous_api_ownership",
                    f"relationship {prim.GetPath()}.{name} targets analytic Cube {target}",
                )
        for attribute in prim.GetAttributes():
            for connection in attribute.GetConnections():
                if str(connection.GetPrimPath()) in candidate_paths:
                    _fail(
                        "connected_input",
                        f"connection {prim.GetPath()}.{attribute.GetName()} targets {connection}",
                    )


def _apply_normalization(
    stage: Any,
    plans: Sequence[_CubePlan],
    wrapper_plans: Sequence[_WrapperPlan],
    *,
    Sdf: Any,
    UsdGeom: Any,
    Vt: Any,
) -> None:
    for plan in plans:
        prim = stage.GetPrimAtPath(plan.body_path)
        copied_render_properties = {
            name: _copyable_property_snapshot(prim.GetProperty(name))
            for name in plan.render_property_names
        }
        copied_collider_properties = {
            name: _copyable_property_snapshot(prim.GetProperty(name))
            for name in plan.collider_property_names
        }
        source_double_sided = bool(UsdGeom.Gprim(prim).GetDoubleSidedAttr().Get())

        if not prim.SetTypeName("Xform"):
            _fail("authoring_failed", f"could not retype {plan.body_path} to Xform")
        _set_applied_schemas(prim, plan.parent_api_schemas, Sdf=Sdf)
        for name in (
            *plan.render_property_names,
            *plan.collider_property_names,
            *sorted(_GEOMETRY_PROPERTIES),
        ):
            if prim.HasProperty(name) and not prim.RemoveProperty(name):
                _fail("authoring_failed", f"could not remove {plan.body_path}.{name}")

        render_mesh = UsdGeom.Mesh.Define(stage, plan.render_mesh_path)
        mesh = render_mesh.GetPrim()
        render_mesh.CreatePointsAttr(Vt.Vec3fArray(_CANONICAL_POINTS))
        render_mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray(_CANONICAL_FACE_VERTEX_COUNTS)
        )
        render_mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray(_CANONICAL_FACE_VERTEX_INDICES)
        )
        render_mesh.CreateNormalsAttr(Vt.Vec3fArray(_CANONICAL_NORMALS))
        render_mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
        render_mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        render_mesh.CreateExtentAttr(
            Vt.Vec3fArray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
        )
        render_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        render_mesh.CreateDoubleSidedAttr(source_double_sided)
        render_mesh.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            (plan.size / 2.0, plan.size / 2.0, plan.size / 2.0)
        )
        for name, snapshot in copied_render_properties.items():
            _restore_property(mesh, name, snapshot)

        collider = UsdGeom.Cube.Define(stage, plan.collider_path)
        collider.CreateSizeAttr(plan.size)
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        _set_applied_schemas(collider.GetPrim(), plan.collider_api_schemas, Sdf=Sdf)
        for name, snapshot in copied_collider_properties.items():
            _restore_property(collider.GetPrim(), name, snapshot)

    for wrapper_plan in wrapper_plans:
        wrapper = stage.GetPrimAtPath(wrapper_plan.path)
        _set_applied_schemas(
            wrapper,
            wrapper_plan.retained_api_schemas,
            Sdf=Sdf,
        )


def _set_applied_schemas(prim: Any, schemas: Sequence[str], *, Sdf: Any) -> None:
    if not prim.SetMetadata(
        "apiSchemas", Sdf.TokenListOp.CreateExplicit(list(schemas))
    ):
        _fail("authoring_failed", f"could not set API schemas at {prim.GetPath()}")


def _copyable_property_snapshot(prop: Any) -> dict[str, Any]:
    if not prop or not prop.IsValid():
        _fail("authoring_failed", "planned property disappeared before authoring")
    if hasattr(prop, "GetTypeName"):
        attr = prop
        return {
            "kind": "attribute",
            "type_name": attr.GetTypeName(),
            "custom": bool(attr.IsCustom()),
            "variability": attr.GetVariability(),
            "has_authored_default": bool(attr.HasAuthoredMetadata("default")),
            "authored_default": attr.GetMetadata("default"),
            "metadata": dict(attr.GetAllAuthoredMetadata()),
        }
    relationship = prop
    return {
        "kind": "relationship",
        "custom": bool(relationship.IsCustom()),
        "targets": tuple(relationship.GetTargets()),
        "metadata": dict(relationship.GetAllAuthoredMetadata()),
    }


def _restore_property(prim: Any, name: str, snapshot: Mapping[str, Any]) -> None:
    metadata = dict(snapshot["metadata"])
    for key in ("default", "timeSamples", "connectionPaths", "targetPaths"):
        metadata.pop(key, None)
    if snapshot["kind"] == "attribute":
        attr = prim.CreateAttribute(
            name,
            snapshot["type_name"],
            custom=bool(snapshot["custom"]),
            variability=snapshot["variability"],
        )
        if snapshot["has_authored_default"] and not attr.Set(
            snapshot["authored_default"]
        ):
            _fail("authoring_failed", f"could not restore {prim.GetPath()}.{name}")
        for key, value in sorted(metadata.items()):
            if not attr.SetMetadata(key, value):
                _fail(
                    "authoring_failed",
                    f"could not restore metadata {prim.GetPath()}.{name}.{key}",
                )
        return
    relationship = prim.CreateRelationship(name, custom=bool(snapshot["custom"]))
    if not relationship.SetTargets(snapshot["targets"]):
        _fail("authoring_failed", f"could not restore targets {prim.GetPath()}.{name}")
    for key, value in sorted(metadata.items()):
        if not relationship.SetMetadata(key, value):
            _fail(
                "authoring_failed",
                f"could not restore metadata {prim.GetPath()}.{name}.{key}",
            )


def _validate_normalized_stage(
    stage: Any,
    proof: _NormalizationProof,
    *,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    if str(stage.GetDefaultPrim().GetPath()) != proof.default_prim_path:
        _fail("invariant_failed", "default prim changed")
    if _stage_metadata_snapshot(stage) != proof.stage_metadata:
        _fail("invariant_failed", "stage metadata changed")
    if _joint_graph_snapshot(stage, UsdPhysics=UsdPhysics) != proof.joint_graph:
        _fail("invariant_failed", "joint graph changed")
    if _filtered_pairs_snapshot(stage) != proof.filtered_pairs:
        _fail("invariant_failed", "filtered-pair semantics changed")
    _validate_prim_inventory(stage, proof)

    for plan in proof.plans:
        body = stage.GetPrimAtPath(plan.body_path)
        render = stage.GetPrimAtPath(plan.render_mesh_path)
        collider = stage.GetPrimAtPath(plan.collider_path)
        if not body.IsA(UsdGeom.Xform) or body.IsA(UsdGeom.Gprim):
            _fail(
                "invariant_failed", f"body was not retyped to Xform: {plan.body_path}"
            )
        if _authored_api_schema_tokens(body) != plan.parent_api_schemas:
            _fail("invariant_failed", f"body API ownership changed: {plan.body_path}")
        if body.HasAPI(UsdPhysics.CollisionAPI):
            _fail("invariant_failed", f"body still owns CollisionAPI: {plan.body_path}")
        if (
            _retained_body_snapshot(
                body,
                retained_property_names=tuple(
                    prop["name"] for prop in plan.retained_body_snapshot["properties"]
                ),
                retained_api_schemas=plan.parent_api_schemas,
                Usd=Usd,
                UsdGeom=UsdGeom,
            )
            != plan.retained_body_snapshot
        ):
            _fail("invariant_failed", f"body state changed: {plan.body_path}")

        if not render.IsA(UsdGeom.Mesh):
            _fail(
                "invariant_failed",
                f"render child is not a Mesh: {plan.render_mesh_path}",
            )
        _validate_canonical_mesh(render, plan, UsdGeom=UsdGeom)
        if render.HasAPI(UsdPhysics.CollisionAPI):
            _fail(
                "invariant_failed",
                f"render Mesh owns CollisionAPI: {plan.render_mesh_path}",
            )
        if (
            _render_semantics_snapshot(
                render,
                UsdGeom=UsdGeom,
                UsdShade=UsdShade,
            )
            != plan.render_snapshot
        ):
            _fail(
                "invariant_failed", f"render semantics changed: {plan.render_mesh_path}"
            )

        if not collider.IsA(UsdGeom.Cube):
            _fail(
                "invariant_failed",
                f"collider child is not a Cube: {plan.collider_path}",
            )
        if _authored_api_schema_tokens(collider) != plan.collider_api_schemas:
            _fail(
                "invariant_failed",
                f"collider API ownership changed: {plan.collider_path}",
            )
        if collider.HasAPI(UsdPhysics.RigidBodyAPI):
            _fail(
                "invariant_failed",
                f"collider child owns RigidBodyAPI: {plan.collider_path}",
            )
        if str(UsdGeom.Imageable(collider).ComputePurpose()) != str(
            UsdGeom.Tokens.guide
        ):
            _fail(
                "invariant_failed",
                f"collider child is not guide purpose: {plan.collider_path}",
            )
        if (
            _collider_semantics_snapshot(
                collider,
                size=plan.size,
                property_names=plan.collider_property_names,
                api_schemas=plan.collider_api_schemas,
                UsdShade=UsdShade,
            )
            != plan.collider_snapshot
        ):
            _fail(
                "invariant_failed", f"collider semantics changed: {plan.collider_path}"
            )
        if (
            _cube_world_corners(
                collider,
                size=plan.size,
                Usd=Usd,
                UsdGeom=UsdGeom,
            )
            != plan.source_world_corners
        ):
            _fail(
                "invariant_failed",
                f"analytic collider world geometry changed: {plan.collider_path}",
            )

    for wrapper_plan in proof.wrapper_plans:
        wrapper = stage.GetPrimAtPath(wrapper_plan.path)
        if not wrapper.IsA(UsdGeom.Xform) or wrapper.IsA(UsdGeom.Gprim):
            _fail(
                "invariant_failed",
                f"rigid wrapper type changed: {wrapper_plan.path}",
            )
        if wrapper.HasAPI(UsdPhysics.RigidBodyAPI):
            _fail(
                "invariant_failed",
                f"rigid wrapper was not demoted: {wrapper_plan.path}",
            )
        if _authored_api_schema_tokens(wrapper) != wrapper_plan.retained_api_schemas:
            _fail(
                "invariant_failed",
                f"rigid wrapper APIs changed: {wrapper_plan.path}",
            )
        if (
            _wrapper_semantics_snapshot(wrapper, Usd=Usd, UsdGeom=UsdGeom)
            != wrapper_plan.source_snapshot
        ):
            _fail(
                "invariant_failed",
                f"rigid wrapper state changed: {wrapper_plan.path}",
            )


def _validate_canonical_mesh(prim: Any, plan: _CubePlan, *, UsdGeom: Any) -> None:
    mesh = UsdGeom.Mesh(prim)
    expected = {
        "points": [list(value) for value in _CANONICAL_POINTS],
        "counts": list(_CANONICAL_FACE_VERTEX_COUNTS),
        "indices": list(_CANONICAL_FACE_VERTEX_INDICES),
        "normals": [list(value) for value in _CANONICAL_NORMALS],
        "orientation": str(UsdGeom.Tokens.rightHanded),
        "extent": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "subdivision": str(UsdGeom.Tokens.none),
        "normal_interpolation": str(UsdGeom.Tokens.uniform),
    }
    actual = {
        "points": _usd_value(mesh.GetPointsAttr().Get()),
        "counts": _usd_value(mesh.GetFaceVertexCountsAttr().Get()),
        "indices": _usd_value(mesh.GetFaceVertexIndicesAttr().Get()),
        "normals": _usd_value(mesh.GetNormalsAttr().Get()),
        "orientation": str(mesh.GetOrientationAttr().Get()),
        "extent": _usd_value(mesh.GetExtentAttr().Get()),
        "subdivision": str(mesh.GetSubdivisionSchemeAttr().Get()),
        "normal_interpolation": str(mesh.GetNormalsInterpolation()),
    }
    if actual != expected:
        _fail("invariant_failed", f"canonical mesh payload differs at {prim.GetPath()}")
    xformable = UsdGeom.Xformable(prim)
    operations = xformable.GetOrderedXformOps()
    expected_scale = (plan.size / 2.0, plan.size / 2.0, plan.size / 2.0)
    if (
        len(operations) != 1
        or operations[0].GetOpType() != UsdGeom.XformOp.TypeScale
        or tuple(float(value) for value in operations[0].Get()) != expected_scale
        or xformable.GetResetXformStack()
    ):
        _fail("invariant_failed", f"canonical mesh scale differs at {prim.GetPath()}")


def _validate_prim_inventory(stage: Any, proof: _NormalizationProof) -> None:
    expected = dict(proof.prim_inventory)
    for plan in proof.plans:
        expected[plan.body_path] = "Xform"
        expected[plan.render_mesh_path] = "Mesh"
        expected[plan.collider_path] = "Cube"
    actual = dict(_prim_inventory(stage))
    if actual != expected:
        _fail(
            "invariant_failed", "normalization changed unexpected prim paths or types"
        )


def _write_output_asset(
    *,
    root_layer_path: Path,
    package_root: Path,
    staged_output: Path,
    output_suffix: str,
    Sdf: Any,
    UsdUtils: Any,
) -> None:
    if output_suffix == ".usdz":
        for member in sorted(package_root.rglob("*")):
            if member.is_file():
                os.utime(member, (_FIXED_PACKAGE_MTIME, _FIXED_PACKAGE_MTIME))
        if not UsdUtils.CreateNewUsdzPackage(str(root_layer_path), str(staged_output)):
            _fail("output_write_failed", "OpenUSD could not create the USDZ package")
        _asset_manifest(staged_output)
        return
    layer = Sdf.Layer.FindOrOpen(str(root_layer_path))
    if layer is None or not layer.Export(str(staged_output)):
        _fail(
            "output_write_failed",
            f"could not export normalized layer to {output_suffix}",
        )


def _asset_manifest(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".usdz":
        entry = {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        return {
            "container": "raw_usd",
            "root_entry": path.name,
            "entry_paths": [path.name],
            "entries": [entry],
            "dependency_bundle_sha256": _canonical_json_sha256([entry]),
        }
    entries: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos:
                _fail("output_readback_failed", "output USDZ is empty")
            root_parts = _validate_usdz_info(infos[0], require_file=True)
            root_entry = "/".join(root_parts)
            if Path(root_entry).suffix.lower() not in _RAW_USD_SUFFIXES:
                _fail("output_readback_failed", "output USDZ first entry is not USD")
            seen: set[str] = set()
            for info in infos:
                parts = _validate_usdz_info(info, require_file=False)
                name = "/".join(parts)
                if name in seen:
                    _fail("output_readback_failed", f"duplicate output entry: {name}")
                seen.add(name)
                if info.is_dir():
                    continue
                digest = hashlib.sha256()
                with archive.open(info) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                entries.append(
                    {"path": name, "size": info.file_size, "sha256": digest.hexdigest()}
                )
    except zipfile.BadZipFile as exc:
        raise AnalyticGprimNormalizationError(
            "output_readback_failed", str(exc)
        ) from exc
    ordered = sorted(entries, key=lambda item: item["path"])
    return {
        "container": "usdz",
        "root_entry": root_entry,
        "entry_paths": [entry["path"] for entry in ordered],
        "entries": ordered,
        "dependency_bundle_sha256": _canonical_json_sha256(ordered),
    }


def _receipt_payload(
    *,
    source_sha256: str,
    source_capture: _SourceCapture,
    output_sha256: str,
    output_manifest: Mapping[str, Any],
    proof: _NormalizationProof,
) -> dict[str, Any]:
    path_mapping = [
        {
            "source_kind": plan.source_kind,
            "source_body_path": plan.body_path,
            "output_body_path": plan.body_path,
            "source_collider_path": plan.body_path,
            "output_collider_path": plan.collider_path,
            "output_render_mesh_path": plan.render_mesh_path,
            "mass_api_migrated_to_collider": (
                "PhysicsMassAPI" in plan.collider_api_schemas
            ),
        }
        for plan in proof.plans
    ]
    return {
        "schema_version": ANALYTIC_GPRIM_RECEIPT_SCHEMA_VERSION,
        "normalization_version": ANALYTIC_GPRIM_NORMALIZATION_VERSION,
        "source_identity": {
            "container": source_capture.source_container,
            "root_entry": source_capture.root_entry,
            "asset_sha256": source_sha256,
            "dependency_bundle_sha256": source_capture.source_manifest[
                "dependency_bundle_sha256"
            ],
        },
        "output_identity": {
            "container": output_manifest["container"],
            "root_entry": output_manifest["root_entry"],
            "asset_sha256": output_sha256,
            "dependency_bundle_sha256": output_manifest["dependency_bundle_sha256"],
        },
        "path_mapping": path_mapping,
        "collider_path_migrations": [
            {
                "source_collider_path": item["source_collider_path"],
                "output_collider_path": item["output_collider_path"],
            }
            for item in path_mapping
        ],
        "demoted_rigid_wrapper_paths": [plan.path for plan in proof.wrapper_plans],
        "canonical_mesh": {
            "points": [list(value) for value in _CANONICAL_POINTS],
            "face_vertex_counts": list(_CANONICAL_FACE_VERTEX_COUNTS),
            "face_vertex_indices": list(_CANONICAL_FACE_VERTEX_INDICES),
            "normals": [list(value) for value in _CANONICAL_NORMALS],
            "orientation": "rightHanded",
            "extent": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            "subdivision_scheme": "none",
            "scale_rule": "source_cube_size_divided_by_2",
        },
        "invariants": {
            "source_bytes_preserved": True,
            "body_paths_preserved": True,
            "body_world_transforms_preserved": True,
            "joint_graph_preserved": True,
            "analytic_collider_world_geometry_preserved": True,
            "rigid_wrapper_state_preserved": True,
            "joint_graph_sha256": _canonical_json_sha256(proof.joint_graph),
            "filtered_pairs_sha256": _canonical_json_sha256(proof.filtered_pairs),
            "normalized_body_state_sha256": _canonical_json_sha256(
                [plan.retained_body_snapshot for plan in proof.plans]
            ),
            "demoted_wrapper_state_sha256": _canonical_json_sha256(
                [plan.source_snapshot for plan in proof.wrapper_plans]
            ),
        },
    }


def _retained_body_snapshot(
    prim: Any,
    *,
    retained_property_names: Iterable[str],
    retained_api_schemas: Sequence[str],
    Usd: Any,
    UsdGeom: Any,
) -> dict[str, Any]:
    return {
        "path": str(prim.GetPath()),
        "metadata": _prim_metadata_without_type_and_apis(prim),
        "api_schemas": list(retained_api_schemas),
        "properties": [
            _property_semantics_snapshot(prim.GetProperty(name))
            for name in sorted(retained_property_names)
        ],
        "world_transform": _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ),
    }


def _wrapper_semantics_snapshot(prim: Any, *, Usd: Any, UsdGeom: Any) -> dict[str, Any]:
    properties: list[dict[str, Any]] = []
    for prop in sorted(
        prim.GetAuthoredProperties(),
        key=lambda item: str(item.GetName()),
    ):
        snapshot = _property_semantics_snapshot(prop)
        if hasattr(prop, "GetTimeSamples"):
            snapshot["time_samples"] = [
                {
                    "time": float(time),
                    "value": _usd_value(prop.Get(Usd.TimeCode(time))),
                }
                for time in prop.GetTimeSamples()
            ]
        properties.append(snapshot)
    return {
        "path": str(prim.GetPath()),
        "type": str(prim.GetTypeName()),
        "metadata": _prim_metadata_without_type_and_apis(prim),
        "properties": properties,
        "children": [str(child.GetPath()) for child in prim.GetAllChildren()],
        "world_transform": _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ),
    }


def _render_semantics_snapshot(
    prim: Any, *, UsdGeom: Any, UsdShade: Any
) -> dict[str, Any]:
    imageable = UsdGeom.Imageable(prim)
    gprim = UsdGeom.Gprim(prim)
    return {
        "effective_visibility": str(imageable.ComputeEffectiveVisibility()),
        "purpose": str(imageable.ComputePurpose()),
        "double_sided": bool(gprim.GetDoubleSidedAttr().Get()),
        "visual_material": _bound_material_path(prim, UsdShade=UsdShade, purpose=""),
        "primvars": _effective_primvar_snapshot(prim, UsdGeom=UsdGeom),
    }


def _collider_semantics_snapshot(
    prim: Any,
    *,
    size: float,
    property_names: Iterable[str],
    api_schemas: Sequence[str],
    UsdShade: Any,
) -> dict[str, Any]:
    return {
        "size": size,
        "api_schemas": list(api_schemas),
        "properties": [
            _property_semantics_snapshot(prim.GetProperty(name))
            for name in sorted(property_names)
        ],
        "physics_material": _bound_material_path(
            prim,
            UsdShade=UsdShade,
            purpose="physics",
        ),
    }


def _bound_material_path(prim: Any, *, UsdShade: Any, purpose: str) -> str | None:
    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        materialPurpose=purpose
    )
    if not material or not material.GetPrim().IsValid():
        return None
    return str(material.GetPath())


def _effective_primvar_snapshot(prim: Any, *, UsdGeom: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for primvar in UsdGeom.PrimvarsAPI(prim).FindPrimvarsWithInheritance():
        attr = primvar.GetAttr()
        if not attr.HasAuthoredValueOpinion():
            continue
        indices = primvar.GetIndicesAttr()
        records.append(
            {
                "name": str(primvar.GetPrimvarName()),
                "type": str(attr.GetTypeName()),
                "interpolation": str(primvar.GetInterpolation()),
                "element_size": int(primvar.GetElementSize()),
                "value": _usd_value(attr.Get()),
                "indices": _usd_value(indices.Get()) if indices else None,
                "unauthored_values_index": int(primvar.GetUnauthoredValuesIndex()),
            }
        )
    return sorted(records, key=lambda item: item["name"])


def _property_semantics_snapshot(prop: Any) -> dict[str, Any]:
    if not prop or not prop.IsValid():
        _fail("invariant_failed", "expected property is missing")
    if hasattr(prop, "GetTypeName"):
        return {
            "name": str(prop.GetName()),
            "kind": "attribute",
            "type": str(prop.GetTypeName()),
            "custom": bool(prop.IsCustom()),
            "variability": str(prop.GetVariability()),
            "metadata": _usd_value(prop.GetAllAuthoredMetadata()),
            "value": _usd_value(prop.Get()),
            "connections": [str(path) for path in prop.GetConnections()],
        }
    return {
        "name": str(prop.GetName()),
        "kind": "relationship",
        "custom": bool(prop.IsCustom()),
        "metadata": _usd_value(prop.GetAllAuthoredMetadata()),
        "targets": [str(path) for path in prop.GetTargets()],
    }


def _joint_graph_snapshot(stage: Any, *, UsdPhysics: Any) -> dict[str, Any]:
    records = []
    for prim in _traverse_all(stage):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        records.append(
            {
                "path": str(prim.GetPath()),
                "type": str(prim.GetTypeName()),
                "metadata": _prim_metadata_without_type_and_apis(prim),
                "api_schemas": [str(token) for token in prim.GetAppliedSchemas()],
                "properties": [
                    _property_semantics_snapshot(prop)
                    for prop in sorted(
                        prim.GetAuthoredProperties(),
                        key=lambda item: str(item.GetName()),
                    )
                ],
            }
        )
    return {"joints": records}


def _filtered_pairs_snapshot(stage: Any) -> dict[str, Any]:
    records = []
    for prim in _traverse_all(stage):
        schemas = [
            token
            for token in _authored_api_schema_tokens(prim)
            if token.split(":", maxsplit=1)[0] == "PhysicsFilteredPairsAPI"
        ]
        if not schemas:
            continue
        relationship = prim.GetRelationship("physics:filteredPairs")
        records.append(
            {
                "path": str(prim.GetPath()),
                "api_schemas": schemas,
                "targets": [str(path) for path in relationship.GetTargets()],
                "metadata": _usd_value(relationship.GetAllAuthoredMetadata()),
            }
        )
    return {"owners": records}


def _authored_api_schema_tokens(prim: Any) -> tuple[str, ...]:
    value = prim.GetMetadata("apiSchemas")
    if value is None:
        return ()
    return tuple(str(token) for token in value.GetAppliedItems())


def _stage_metadata_snapshot(stage: Any) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _usd_value(stage.GetPseudoRoot().GetAllAuthoredMetadata()),
    )


def _prim_metadata_without_type_and_apis(prim: Any) -> dict[str, Any]:
    metadata = dict(prim.GetAllAuthoredMetadata())
    metadata.pop("typeName", None)
    metadata.pop("apiSchemas", None)
    return cast(dict[str, Any], _usd_value(metadata))


def _prim_inventory(stage: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (str(prim.GetPath()), str(prim.GetTypeName()))
            for prim in _traverse_all(stage)
        )
    )


def _cube_world_corners(
    prim: Any,
    *,
    size: float,
    Usd: Any,
    UsdGeom: Any,
) -> tuple[tuple[float, float, float], ...]:
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    half = size / 2.0
    corners: list[tuple[float, float, float]] = []
    for point in _CANONICAL_POINTS:
        transformed = transform.Transform(
            (point[0] * half, point[1] * half, point[2] * half)
        )
        corners.append(
            (
                float(transformed[0]),
                float(transformed[1]),
                float(transformed[2]),
            )
        )
    return tuple(corners)


def _matrix_values(matrix: Any) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _traverse_all(stage: Any) -> list[Any]:
    from pxr import Usd

    return list(Usd.PrimRange.Stage(stage, Usd.PrimAllPrimsPredicate))


def _descendants(prim: Any) -> list[Any]:
    from pxr import Usd

    result = list(Usd.PrimRange.AllPrims(prim))
    return result[1:]


def _require_source_unchanged(
    source: Path,
    *,
    expected_sha256: str,
    expected_stat: os.stat_result,
) -> None:
    current = source.lstat()
    expected_identity = (
        expected_stat.st_dev,
        expected_stat.st_ino,
        expected_stat.st_size,
        expected_stat.st_mtime_ns,
    )
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if current_identity != expected_identity or _file_sha256(source) != expected_sha256:
        _fail(
            "source_changed",
            f"source bytes or identity changed during normalization: {source}",
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _usd_value(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _usd_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return [float(value.GetReal()), *[float(item) for item in imaginary]]
    try:
        return [_usd_value(item) for item in value]
    except TypeError:
        pass
    return str(value)


def _fail(code: str, detail: str) -> None:
    raise AnalyticGprimNormalizationError(code, detail)


__all__ = [
    "ANALYTIC_GPRIM_COLLIDER_CHILD_NAME",
    "ANALYTIC_GPRIM_NORMALIZATION_VERSION",
    "ANALYTIC_GPRIM_RECEIPT_SCHEMA_VERSION",
    "ANALYTIC_GPRIM_RENDER_CHILD_NAME",
    "AnalyticGprimNormalizationError",
    "AnalyticGprimNormalizationResult",
    "normalize_analytic_cube_gprims",
]
