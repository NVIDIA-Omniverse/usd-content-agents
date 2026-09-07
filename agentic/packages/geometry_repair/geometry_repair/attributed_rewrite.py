# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact USD face/corner rewrites for bounded topology-preserving edits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh_io import copy_usd_stage


@dataclass(frozen=True)
class ExactTopologyRewrite:
    """One output triangle array with exact source face/corner provenance."""

    vertices: np.ndarray
    triangles: np.ndarray
    source_face_ids: np.ndarray
    source_corner_ids: np.ndarray

    def validate(self, *, source_vertices: np.ndarray, source_triangles: np.ndarray) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int64)
        source_faces = np.asarray(self.source_face_ids, dtype=np.int64)
        source_corners = np.asarray(self.source_corner_ids, dtype=np.int64)
        if vertices.shape != np.asarray(source_vertices).shape or not np.array_equal(
            vertices,
            np.asarray(source_vertices, dtype=np.float64),
        ):
            raise ValueError("exact topology rewrite must retain every source vertex in order")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            raise ValueError("exact topology rewrite requires triangular output faces")
        if source_faces.shape != (len(triangles),):
            raise ValueError("source_face_ids must identify every output face")
        if source_corners.shape != (len(triangles), 3):
            raise ValueError("source_corner_ids must identify every output face-corner")
        if np.any(source_faces < 0) or np.any(source_faces >= len(source_triangles)):
            raise ValueError("source_face_ids contain an out-of-range face")
        if np.any(source_corners < 0) or np.any(source_corners > 2):
            raise ValueError("source_corner_ids must be in the triangular range [0, 2]")
        if len(set(source_faces.tolist())) != len(source_faces):
            raise ValueError("exact topology rewrite cannot duplicate one source face")
        source = np.asarray(source_triangles, dtype=np.int64)
        expected = source[source_faces[:, None], source_corners]
        if not np.array_equal(triangles, expected):
            raise ValueError("output triangles do not match their exact source face-corners")


@dataclass(frozen=True)
class GeneratedPatchTopologyRewrite:
    """Source triangles followed by generated faces using only source vertices."""

    vertices: np.ndarray
    triangles: np.ndarray
    generated_face_sources: dict[int, tuple[int, ...]]

    def validate(self, *, source_vertices: np.ndarray, source_triangles: np.ndarray) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int64)
        source_points = np.asarray(source_vertices, dtype=np.float64)
        source_faces = np.asarray(source_triangles, dtype=np.int64).reshape((-1, 3))
        if vertices.shape != source_points.shape or not np.array_equal(vertices, source_points):
            raise ValueError("generated patch rewrite must retain every source vertex in order")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,):
            raise ValueError("generated patch rewrite requires triangular output faces")
        if len(triangles) <= len(source_faces):
            raise ValueError("generated patch rewrite must append at least one face")
        if not np.array_equal(triangles[: len(source_faces)], source_faces):
            raise ValueError("generated patch rewrite must retain every source face in order")
        expected_generated = set(range(len(source_faces), len(triangles)))
        if set(self.generated_face_sources) != expected_generated:
            raise ValueError("every appended face requires generated-face source evidence")
        for output_face, sources in self.generated_face_sources.items():
            if not sources:
                raise ValueError(f"generated face {output_face} has no boundary source faces")
            if len(set(sources)) != len(sources) or any(
                source < 0 or source >= len(source_faces) for source in sources
            ):
                raise ValueError(f"generated face {output_face} has invalid source-face evidence")
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError("generated patch contains an out-of-range vertex")


def exact_topology_rewrite(
    *,
    vertices: np.ndarray,
    source_triangles: np.ndarray,
    output_triangles: np.ndarray,
    source_face_ids: np.ndarray,
) -> ExactTopologyRewrite:
    """Derive exact corner provenance for retained and optionally reoriented faces."""

    source = np.asarray(source_triangles, dtype=np.int64).reshape((-1, 3))
    output = np.asarray(output_triangles, dtype=np.int64).reshape((-1, 3))
    face_ids = np.asarray(source_face_ids, dtype=np.int64).reshape((-1,))
    if len(output) != len(face_ids):
        raise ValueError("one source face ID is required for every output triangle")
    corner_ids = np.empty((len(output), 3), dtype=np.int64)
    for output_face, (triangle, source_face_id) in enumerate(zip(output, face_ids, strict=True)):
        if source_face_id < 0 or source_face_id >= len(source):
            raise ValueError(f"source face {source_face_id} is out of range")
        source_face = source[source_face_id]
        for output_corner, vertex_id in enumerate(triangle):
            matches = np.flatnonzero(source_face == vertex_id)
            if len(matches) != 1:
                raise ValueError(
                    f"output face {output_face} corner {output_corner} does not map to "
                    f"one source corner on face {source_face_id}"
                )
            corner_ids[output_face, output_corner] = int(matches[0])
    rewrite = ExactTopologyRewrite(
        vertices=np.asarray(vertices, dtype=np.float64),
        triangles=output,
        source_face_ids=face_ids,
        source_corner_ids=corner_ids,
    )
    rewrite.validate(source_vertices=np.asarray(vertices), source_triangles=source)
    return rewrite


def _selected_array(values, element_ids: list[int], *, element_size: int = 1):
    if element_size < 1:
        raise ValueError("primvar element size must be positive")
    raw = list(values)
    selected = []
    for element_id in element_ids:
        start = element_id * element_size
        stop = start + element_size
        if start < 0 or stop > len(raw):
            raise ValueError(f"attribute element {element_id} exceeds the authored value count")
        selected.extend(raw[start:stop])
    return type(values)(selected)


def _require_default_only(attribute, *, label: str) -> None:
    if attribute and attribute.GetNumTimeSamples() > 0:
        raise ValueError(
            f"{label} has time samples; exact animated topology remapping is unsupported"
        )


def _remap_primvar(primvar, *, face_ids: list[int], corner_ids: list[int]) -> None:
    from pxr import UsdGeom, Vt

    interpolation = str(primvar.GetInterpolation())
    if interpolation not in {str(UsdGeom.Tokens.uniform), str(UsdGeom.Tokens.faceVarying)}:
        return
    _require_default_only(primvar.GetAttr(), label=f"primvar {primvar.GetPrimvarName()}")
    element_ids = face_ids if interpolation == str(UsdGeom.Tokens.uniform) else corner_ids
    values = primvar.Get()
    if values is None:
        raise ValueError(f"primvar {primvar.GetPrimvarName()} has no authored values")
    if primvar.IsIndexed():
        indices_attr = primvar.GetIndicesAttr()
        _require_default_only(indices_attr, label=f"primvar indices {primvar.GetPrimvarName()}")
        indices = list(primvar.GetIndices())
        if any(element_id < 0 or element_id >= len(indices) for element_id in element_ids):
            raise ValueError(f"primvar {primvar.GetPrimvarName()} indices do not cover topology")
        primvar.SetIndices(Vt.IntArray([int(indices[element_id]) for element_id in element_ids]))
        return
    primvar.Set(
        _selected_array(
            values,
            element_ids,
            element_size=max(int(primvar.GetElementSize()), 1),
        )
    )


def _logical_elements(values, *, element_size: int) -> list[tuple[object, ...]]:
    raw = list(values)
    if element_size < 1 or len(raw) % element_size:
        raise ValueError("authored array does not contain complete primvar elements")
    return [tuple(raw[index : index + element_size]) for index in range(0, len(raw), element_size)]


def _resolved_domain_elements(primvar) -> tuple[list[tuple[object, ...]], list[int] | None]:
    values = primvar.Get()
    if values is None:
        raise ValueError(f"primvar {primvar.GetPrimvarName()} has no authored values")
    logical = _logical_elements(values, element_size=max(int(primvar.GetElementSize()), 1))
    if not primvar.IsIndexed():
        return logical, None
    indices = [int(value) for value in primvar.GetIndices()]
    if any(index < 0 or index >= len(logical) for index in indices):
        raise ValueError(f"primvar {primvar.GetPrimvarName()} has invalid authored indices")
    return logical, indices


def _agreeing_element(
    logical: list[tuple[object, ...]],
    authored_indices: list[int] | None,
    domain_ids: list[int],
    *,
    label: str,
) -> tuple[tuple[object, ...], int | None]:
    if not domain_ids:
        raise ValueError(f"{label} has no contributing source elements")
    value_ids = [
        authored_indices[domain_id] if authored_indices is not None else domain_id
        for domain_id in domain_ids
    ]
    if any(value_id < 0 or value_id >= len(logical) for value_id in value_ids):
        raise ValueError(f"{label} does not cover its contributing source topology")
    values = [logical[value_id] for value_id in value_ids]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{label} has conflicting values across the generated patch boundary")
    return first, value_ids[0] if authored_indices is not None else None


def _generated_corner_source_ids(
    source_triangles: np.ndarray,
    output_triangle: np.ndarray,
    source_faces: tuple[int, ...],
) -> list[list[int]]:
    corner_sources: list[list[int]] = []
    for vertex_id in output_triangle:
        matches = [
            source_face * 3 + source_corner
            for source_face in source_faces
            for source_corner, source_vertex in enumerate(source_triangles[source_face])
            if int(source_vertex) == int(vertex_id)
        ]
        if not matches:
            raise ValueError(
                f"generated patch vertex {int(vertex_id)} is not on its declared boundary"
            )
        corner_sources.append(sorted(set(matches)))
    return corner_sources


def _append_generated_primvar_values(
    primvar,
    *,
    source_face_count: int,
    source_triangles: np.ndarray,
    rewrite: GeneratedPatchTopologyRewrite,
) -> None:
    from pxr import UsdGeom, Vt

    interpolation = str(primvar.GetInterpolation())
    if interpolation not in {str(UsdGeom.Tokens.uniform), str(UsdGeom.Tokens.faceVarying)}:
        return
    _require_default_only(primvar.GetAttr(), label=f"primvar {primvar.GetPrimvarName()}")
    logical, authored_indices = _resolved_domain_elements(primvar)
    base_domain_count = (
        source_face_count if interpolation == str(UsdGeom.Tokens.uniform) else source_face_count * 3
    )
    if authored_indices is None and len(logical) != base_domain_count:
        raise ValueError(f"primvar {primvar.GetPrimvarName()} does not cover source topology")
    if authored_indices is not None and len(authored_indices) != base_domain_count:
        raise ValueError(f"primvar {primvar.GetPrimvarName()} indices do not cover source topology")

    appended_indices: list[int] = []
    appended_values: list[tuple[object, ...]] = []
    for output_face in sorted(rewrite.generated_face_sources):
        sources = rewrite.generated_face_sources[output_face]
        if interpolation == str(UsdGeom.Tokens.uniform):
            contributors = [int(value) for value in sources]
            value, value_id = _agreeing_element(
                logical,
                authored_indices,
                contributors,
                label=f"primvar {primvar.GetPrimvarName()}",
            )
            if value_id is None:
                appended_values.append(value)
            else:
                appended_indices.append(value_id)
            continue
        corner_sources = _generated_corner_source_ids(
            source_triangles,
            rewrite.triangles[output_face],
            sources,
        )
        for contributors in corner_sources:
            value, value_id = _agreeing_element(
                logical,
                authored_indices,
                contributors,
                label=f"primvar {primvar.GetPrimvarName()}",
            )
            if value_id is None:
                appended_values.append(value)
            else:
                appended_indices.append(value_id)

    if authored_indices is not None:
        primvar.SetIndices(Vt.IntArray([*authored_indices, *appended_indices]))
        return
    values = primvar.Get()
    flattened = [item for element in [*logical, *appended_values] for item in element]
    primvar.Set(type(values)(flattened))


def _remap_mesh_attributes(mesh, rewrite: ExactTopologyRewrite) -> None:
    from pxr import UsdGeom, Vt

    face_ids = [int(value) for value in rewrite.source_face_ids]
    corner_ids = [
        source_face * 3 + int(source_corner)
        for source_face, corners in zip(
            rewrite.source_face_ids,
            rewrite.source_corner_ids,
            strict=True,
        )
        for source_corner in corners
    ]
    normals = mesh.GetNormalsAttr()
    if normals and normals.HasAuthoredValueOpinion():
        _require_default_only(normals, label="mesh normals")
        interpolation = str(mesh.GetNormalsInterpolation())
        values = normals.Get()
        if interpolation == str(UsdGeom.Tokens.uniform):
            normals.Set(_selected_array(values, face_ids))
        elif interpolation == str(UsdGeom.Tokens.faceVarying):
            normals.Set(_selected_array(values, corner_ids))
        elif interpolation not in {
            str(UsdGeom.Tokens.constant),
            str(UsdGeom.Tokens.vertex),
            str(UsdGeom.Tokens.varying),
        }:
            raise ValueError(f"unsupported normals interpolation {interpolation!r}")

    for primvar in UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvars():
        _remap_primvar(primvar, face_ids=face_ids, corner_ids=corner_ids)

    source_to_output = {
        source_face: output_face for output_face, source_face in enumerate(face_ids)
    }
    for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
        indices_attr = subset.GetIndicesAttr()
        _require_default_only(indices_attr, label=f"subset {subset.GetPath()} indices")
        source_subset = [int(value) for value in (indices_attr.Get() or [])]
        indices_attr.Set(
            Vt.IntArray(
                sorted(
                    source_to_output[source_face]
                    for source_face in source_subset
                    if source_face in source_to_output
                )
            )
        )

    holes = mesh.GetHoleIndicesAttr()
    if holes and holes.HasAuthoredValueOpinion():
        _require_default_only(holes, label="mesh hole indices")
        source_holes = [int(value) for value in (holes.Get() or [])]
        holes.Set(
            Vt.IntArray(
                sorted(
                    source_to_output[source_face]
                    for source_face in source_holes
                    if source_face in source_to_output
                )
            )
        )


def _append_generated_normals(mesh, rewrite: GeneratedPatchTopologyRewrite) -> bool:
    from pxr import Gf, UsdGeom

    normals = mesh.GetNormalsAttr()
    if not normals or not normals.HasAuthoredValueOpinion():
        return False
    _require_default_only(normals, label="mesh normals")
    interpolation = str(mesh.GetNormalsInterpolation())
    if interpolation in {
        str(UsdGeom.Tokens.constant),
        str(UsdGeom.Tokens.vertex),
        str(UsdGeom.Tokens.varying),
    }:
        return False
    if interpolation not in {str(UsdGeom.Tokens.uniform), str(UsdGeom.Tokens.faceVarying)}:
        raise ValueError(f"unsupported normals interpolation {interpolation!r}")
    values = list(normals.Get() or [])
    expected = len(rewrite.triangles) - len(rewrite.generated_face_sources)
    expected *= 1 if interpolation == str(UsdGeom.Tokens.uniform) else 3
    if len(values) != expected:
        raise ValueError("authored normals do not cover source topology")
    generated = []
    for output_face in sorted(rewrite.generated_face_sources):
        triangle = np.asarray(rewrite.triangles[output_face], dtype=np.int64)
        points = np.asarray(rewrite.vertices, dtype=np.float64)[triangle]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        magnitude = float(np.linalg.norm(normal))
        if not np.isfinite(magnitude) or magnitude <= 1e-15:
            raise ValueError(f"generated face {output_face} has no finite normal")
        normal /= magnitude
        value = Gf.Vec3f(*(float(item) for item in normal))
        generated.extend([value] if interpolation == str(UsdGeom.Tokens.uniform) else [value] * 3)
    normals.Set(type(normals.Get())([*values, *generated]))
    return True


def _append_generated_subsets(mesh, rewrite: GeneratedPatchTopologyRewrite) -> None:
    from pxr import UsdGeom, Vt

    subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
    memberships = {
        str(subset.GetPath()): {int(value) for value in (subset.GetIndicesAttr().Get() or [])}
        for subset in subsets
    }
    appended: dict[str, list[int]] = {path: [] for path in memberships}
    for output_face, source_faces in sorted(rewrite.generated_face_sources.items()):
        signatures = [
            tuple(path for path, faces in sorted(memberships.items()) if source_face in faces)
            for source_face in source_faces
        ]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(
                f"generated face {output_face} crosses conflicting geometry-subset boundaries"
            )
        for path in signatures[0]:
            appended[path].append(output_face)
    for subset in subsets:
        path = str(subset.GetPath())
        existing = [int(value) for value in (subset.GetIndicesAttr().Get() or [])]
        subset.GetIndicesAttr().Set(Vt.IntArray([*existing, *appended[path]]))


def rewrite_usd_triangle_meshes_exact(
    source: str | Path,
    target: str | Path,
    rewrites: dict[str, ExactTopologyRewrite],
) -> Path:
    """Apply exact face deletion/reordering while retaining attributed USD topology."""

    from pxr import Usd, UsdGeom, Vt

    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    copy_usd_stage(source_path, target_path)
    stage = Usd.Stage.Open(str(target_path))
    if stage is None:
        raise RuntimeError(f"Could not open repair working copy {target_path}")
    for prim_path, rewrite in sorted(rewrites.items()):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Repair target mesh no longer exists: {prim_path}")
        if prim.IsInstanceProxy():
            raise RuntimeError(f"Repair target is a read-only instance proxy: {prim_path}")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64).reshape((-1, 3))
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if len(counts) == 0 or np.any(counts != 3) or len(indices) != len(counts) * 3:
            raise ValueError(f"{prim_path}: exact rewrite requires source triangles")
        source_triangles = indices.reshape((-1, 3))
        rewrite.validate(source_vertices=points, source_triangles=source_triangles)
        _remap_mesh_attributes(mesh, rewrite)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray.FromNumpy(np.full(len(rewrite.triangles), 3, dtype=np.int32))
        )
        mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(np.asarray(rewrite.triangles, dtype=np.int32).reshape((-1,)))
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        prim.SetCustomDataByKey("geometryRepairExactSourceFaceCount", len(rewrite.source_face_ids))
    stage.GetRootLayer().Save()
    return target_path


def rewrite_usd_triangle_meshes_with_generated_patches(
    source: str | Path,
    target: str | Path,
    rewrites: dict[str, GeneratedPatchTopologyRewrite],
) -> Path:
    """Append classified patches while preserving or explicitly reauthoring attributes."""

    from pxr import Usd, UsdGeom, Vt

    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    copy_usd_stage(source_path, target_path)
    stage = Usd.Stage.Open(str(target_path))
    if stage is None:
        raise RuntimeError(f"Could not open repair working copy {target_path}")
    for prim_path, rewrite in sorted(rewrites.items()):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Repair target mesh no longer exists: {prim_path}")
        if prim.IsInstanceProxy():
            raise RuntimeError(f"Repair target is a read-only instance proxy: {prim_path}")
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64).reshape((-1, 3))
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if len(counts) == 0 or np.any(counts != 3) or len(indices) != len(counts) * 3:
            raise ValueError(f"{prim_path}: generated patch rewrite requires source triangles")
        source_triangles = indices.reshape((-1, 3))
        rewrite.validate(source_vertices=points, source_triangles=source_triangles)
        normals_reauthored = _append_generated_normals(mesh, rewrite)
        for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            _append_generated_primvar_values(
                primvar,
                source_face_count=len(source_triangles),
                source_triangles=source_triangles,
                rewrite=rewrite,
            )
        _append_generated_subsets(mesh, rewrite)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.GetFaceVertexCountsAttr().Set(
            Vt.IntArray.FromNumpy(np.full(len(rewrite.triangles), 3, dtype=np.int32))
        )
        mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(np.asarray(rewrite.triangles, dtype=np.int32).reshape((-1,)))
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        prim.SetCustomDataByKey(
            "geometryRepairGeneratedPatchFaceCount",
            len(rewrite.generated_face_sources),
        )
        prim.SetCustomDataByKey("geometryRepairGeneratedPatchNormalsReauthored", normals_reauthored)
        records = {}
        for output_face, source_faces in sorted(rewrite.generated_face_sources.items()):
            source_corners = _generated_corner_source_ids(
                source_triangles,
                rewrite.triangles[output_face],
                source_faces,
            )
            records[str(output_face)] = {
                "source_faces": list(source_faces),
                "source_corners": [
                    [int(corner // 3), int(corner % 3)] for corner in map(min, source_corners)
                ],
            }
        prim.SetCustomDataByKey(
            "geometryRepairGeneratedPatchMap",
            json.dumps(
                {
                    "schema_version": "geometry-repair.generated-patch-map.v1",
                    "attribute_policy": "exact_boundary_agreement_or_refuse",
                    "normals_reauthored": normals_reauthored,
                    "faces": records,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    stage.GetRootLayer().Save()
    return target_path
