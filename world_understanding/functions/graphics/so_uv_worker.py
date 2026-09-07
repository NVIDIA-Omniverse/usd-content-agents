# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene Optimizer UV generation subprocess worker.

Runs in an isolated subprocess with packman USD + SO bindings (same
ABI isolation as ``so_worker.py``).  Executes ``generateAtlasUVs`` or
``generateProjectionUVs`` on a USD stage and exports the result.

Must NOT be imported by the main process.
"""

import json
import math
import os
import sys
import time
import traceback

if __package__:
    from .so_export import _normalize_dependency_roots, export_stage_portably
else:  # copied beside ``so_export.py`` in the ABI-isolated subprocess
    from so_export import _normalize_dependency_roots, export_stage_portably


def _walk_prim_specs(prim_spec):
    """Yield one root-layer prim spec tree without composing the stage."""
    yield prim_spec
    for child in prim_spec.nameChildren:
        yield from _walk_prim_specs(child)


def _materialize_hidden_mesh_ancestors(stage, Sdf, requested_paths):
    """Temporarily define flattened prototype roots hidden behind ``over``.

    ``UsdStage::Flatten`` preserves instances by writing their backing geometry
    below an undefined top-level ``over`` (normally
    ``/Flattened_Prototype_*``).  That spec is still the mutable backing path,
    but neither Scene Optimizer nor a default ``UsdPrimRange`` sees its mesh
    descendants.  Promote only the ancestors needed by the requested paths, or
    mesh-bearing hidden over chains for a full-stage request.  The caller restores
    every changed spec before export, so the synthetic roots never become part
    of the published composed scene.
    """
    root_layer = stage.GetRootLayer()
    if requested_paths:
        candidate_paths = []
        for raw_path in requested_paths:
            path = Sdf.Path(str(raw_path))
            if (
                not path.IsAbsolutePath()
                or path.IsAbsoluteRootPath()
                or not path.IsPrimPath()
            ):
                raise ValueError(
                    "Scene Optimizer paths must be absolute, non-root prim paths: "
                    f"{raw_path!r}"
                )
            candidate_paths.extend(path.GetPrefixes())
    else:
        candidate_paths = []
        for root_spec in root_layer.rootPrims:
            for mesh_spec in _walk_prim_specs(root_spec):
                if mesh_spec.typeName != "Mesh":
                    continue
                mesh_prim = stage.GetPrimAtPath(mesh_spec.path)
                if mesh_prim and mesh_prim.IsDefined():
                    continue
                for prefix in mesh_spec.path.GetPrefixes():
                    ancestor_spec = root_layer.GetPrimAtPath(prefix)
                    if (
                        ancestor_spec is not None
                        and ancestor_spec.specifier == Sdf.SpecifierOver
                    ):
                        candidate_paths.append(prefix)

    changed = []
    seen = set()
    for path in candidate_paths:
        path_text = str(path)
        if path_text in seen:
            continue
        seen.add(path_text)
        spec = root_layer.GetPrimAtPath(path)
        if spec is None or spec.specifier == Sdf.SpecifierDef:
            continue
        changed.append((spec, spec.specifier, spec.typeName))
        spec.specifier = Sdf.SpecifierDef
        if not spec.typeName:
            spec.typeName = "Xform"
    return changed


def _restore_materialized_specs(changed_specs):
    """Restore exact specifier/type state after temporary SO traversal."""
    for spec, specifier, type_name in reversed(changed_specs):
        spec.specifier = specifier
        spec.typeName = type_name


def _scoped_meshes(stage, Usd, UsdGeom, requested_paths):
    """Return distinct meshes in the exact operation scope.

    Prefer ordinary composed meshes, including temporarily materialized
    flattened-prototype backings.  Then add instance-proxy meshes that are not
    already represented by one of those backing paths.  The second pass keeps
    instance-root and instance-only stages visible without processing both a
    writable backing mesh and every proxy that resolves to it.
    """
    roots = []
    if requested_paths:
        for path in requested_paths:
            prim = stage.GetPrimAtPath(str(path))
            if prim and prim.IsValid():
                roots.append(prim)
    else:
        roots.append(stage.GetPseudoRoot())

    meshes = []
    seen = set()
    for root in roots:
        for prim in Usd.PrimRange(root):
            if prim.IsPseudoRoot() or not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if path in seen:
                continue
            seen.add(path)
            meshes.append(prim)

    backing_paths = set(seen)
    for root in roots:
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.IsPseudoRoot() or not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if path in seen:
                continue
            if prim.IsInstanceProxy() and any(
                str(spec.path) in backing_paths for spec in prim.GetPrimStack()
            ):
                continue
            seen.add(path)
            meshes.append(prim)
    return meshes


def _resolve_requested_meshes(stage, Usd, UsdGeom, requested_paths, operation):
    """Resolve every requested path independently or fail the whole operation."""
    resolutions = []
    unresolved = []
    for requested_path in requested_paths:
        mesh_paths = [
            str(prim.GetPath())
            for prim in _scoped_meshes(stage, Usd, UsdGeom, [requested_path])
        ]
        resolutions.append(
            {
                "requested_path": str(requested_path),
                "mesh_count": len(mesh_paths),
                "mesh_paths": mesh_paths,
            }
        )
        if not mesh_paths:
            unresolved.append(str(requested_path))
    if unresolved:
        raise RuntimeError(
            f"{operation} requested paths resolved no mesh prims: {unresolved}"
        )
    return resolutions


def _strip_scoped_uvs(meshes):
    """Remove authored scoped UVs so overwrite proves fresh SO authorship.

    Instance proxies are read-only.  A proxy with no existing UV opinion does
    not need clearing and remains a valid SO scope.  If it already exposes an
    authored UV, fail closed when that opinion cannot be removed rather than
    claiming that a later unchanged value was freshly authored.
    """
    for prim in meshes:
        for property_name in ("primvars:st:indices", "primvars:st"):
            attr = prim.GetAttribute(property_name)
            if not attr or not attr.HasAuthoredValue():
                continue
            try:
                prim.RemoveProperty(property_name)
            except Exception as exc:  # noqa: BLE001 - USD raises Tf errors here
                raise RuntimeError(
                    "Could not clear pre-existing UV property "
                    f"{property_name} on mesh {prim.GetPath()}; fresh Scene "
                    "Optimizer authorship cannot be proven"
                ) from exc
            remaining = prim.GetAttribute(property_name)
            if remaining and remaining.HasAuthoredValue():
                proxy_note = " instance proxy" if prim.IsInstanceProxy() else ""
                raise RuntimeError(
                    "Could not clear pre-existing UV property "
                    f"{property_name} on{proxy_note} mesh {prim.GetPath()}; fresh "
                    "Scene Optimizer authorship cannot be proven"
                )


def _mesh_topology_snapshot(prim):
    """Return immutable validated mesh topology for post-SO comparison."""
    path = str(prim.GetPath())
    points_attr = prim.GetAttribute("points")
    face_counts_attr = prim.GetAttribute("faceVertexCounts")
    face_indices_attr = prim.GetAttribute("faceVertexIndices")
    points = points_attr.Get() if points_attr else None
    face_counts = face_counts_attr.Get() if face_counts_attr else None
    face_indices = face_indices_attr.Get() if face_indices_attr else None
    if points is None or face_counts is None or face_indices is None:
        raise RuntimeError(f"Mesh {path} is missing required topology attributes")

    point_values = []
    for index, point in enumerate(points):
        try:
            coordinates = tuple(float(point[axis]) for axis in range(3))
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Mesh {path} point {index} is not a 3D coordinate"
            ) from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise RuntimeError(f"Mesh {path} point {index} is non-finite")
        point_values.append(coordinates)
    if not point_values:
        raise RuntimeError(f"Mesh {path} has no points")

    counts = tuple(int(count) for count in face_counts)
    indices = tuple(int(index) for index in face_indices)
    if not counts or any(count < 3 for count in counts):
        raise RuntimeError(f"Mesh {path} has invalid faceVertexCounts")
    if sum(counts) != len(indices):
        raise RuntimeError(
            f"Mesh {path} faceVertexCounts sum {sum(counts)} does not match "
            f"faceVertexIndices count {len(indices)}"
        )
    if any(index < 0 or index >= len(point_values) for index in indices):
        raise RuntimeError(f"Mesh {path} has out-of-range faceVertexIndices")
    return (tuple(point_values), counts, indices)


def _snapshot_scoped_topologies(meshes):
    """Snapshot exact topology for each resolved mesh path."""
    return {str(prim.GetPath()): _mesh_topology_snapshot(prim) for prim in meshes}


def _uv_topology_error(prim, topology_snapshot):
    """Return why one exact mesh lacks a usable locally authored ``st``."""
    st_attr = prim.GetAttribute("primvars:st")
    if not st_attr or not st_attr.HasAuthoredValue():
        return "missing authored primvars:st"
    values = st_attr.Get()
    if values is None or len(values) == 0:
        return "primvars:st is empty"
    for index, value in enumerate(values):
        try:
            coordinates = tuple(float(value[axis]) for axis in range(2))
        except (IndexError, TypeError, ValueError):
            return f"primvars:st value {index} is not a 2D coordinate"
        if not all(math.isfinite(component) for component in coordinates):
            return f"primvars:st value {index} is non-finite"

    points, face_counts, face_indices = topology_snapshot
    interpolation = str(st_attr.GetMetadata("interpolation") or "constant")
    if interpolation == "constant":
        expected_count = 1
    elif interpolation == "uniform":
        expected_count = len(face_counts)
    elif interpolation in {"vertex", "varying"}:
        expected_count = len(points)
    elif interpolation == "faceVarying":
        expected_count = len(face_indices)
    else:
        return f"unsupported interpolation {interpolation!r}"
    if expected_count <= 0:
        return f"mesh topology has no {interpolation} elements"

    indices_attr = prim.GetAttribute("primvars:st:indices")
    indices = (
        indices_attr.Get() if indices_attr and indices_attr.HasAuthoredValue() else None
    )
    authored_count = len(indices) if indices is not None else len(values)
    if authored_count != expected_count:
        return (
            f"{interpolation} UV count {authored_count} does not match topology "
            f"count {expected_count}"
        )
    if indices is not None and any(
        int(index) < 0 or int(index) >= len(values) for index in indices
    ):
        return "primvars:st:indices references values outside primvars:st"
    return None


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: so_uv_worker.py '<json_params>'\n")
        sys.exit(1)

    try:
        params = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Error: Invalid JSON in arguments: {exc}\n")
        sys.exit(1)

    if not isinstance(params, dict):
        sys.stderr.write("Error: JSON arguments must be an object\n")
        sys.exit(1)
    manifest_path = params.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        sys.stderr.write(
            "Error: JSON parameter manifest_path must be a non-empty string\n"
        )
        sys.exit(1)

    operation = params.get("operation", "unknown")
    op_params = params.get("op_params", {})
    total_start = time.time()
    failure_phase = "request_validation"

    try:
        missing = [
            key
            for key in (
                "input_usd_path",
                "output_usd_path",
                "approved_dependency_roots",
            )
            if key not in params
        ]
        if missing:
            raise KeyError(f"Missing required JSON parameter(s): {', '.join(missing)}")

        input_usd_path = params["input_usd_path"]
        output_usd_path = params["output_usd_path"]
        approved_dependency_roots = [
            str(root)
            for root in _normalize_dependency_roots(params["approved_dependency_roots"])
        ]

        failure_phase = "runtime_import"
        from omni.scene.optimizer.core import ExecutionContext, SceneOptimizerCore
        from pxr import Sdf, Usd, UsdGeom

        failure_phase = "operation"
        stage = Usd.Stage.Open(input_usd_path)
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {input_usd_path}")

        requested_paths = op_params.get("paths") or []
        if not isinstance(requested_paths, list):
            raise TypeError("Scene Optimizer paths must be a list of prim paths")
        materialized_specs = _materialize_hidden_mesh_ancestors(
            stage,
            Sdf,
            requested_paths,
        )

        ctx = None
        try:
            try:
                requested_path_meshes = _resolve_requested_meshes(
                    stage,
                    Usd,
                    UsdGeom,
                    requested_paths,
                    operation,
                )
                scoped_meshes = _scoped_meshes(stage, Usd, UsdGeom, requested_paths)
                scoped_mesh_paths = tuple(str(prim.GetPath()) for prim in scoped_meshes)
                mesh_count = len(scoped_mesh_paths)
                if mesh_count == 0:
                    scope = (
                        ", ".join(str(path) for path in requested_paths) or "<stage>"
                    )
                    raise RuntimeError(
                        f"{operation} resolved zero mesh prims in scope: {scope}"
                    )

                original_topologies = _snapshot_scoped_topologies(scoped_meshes)
                pre_uv_errors = {
                    path: _uv_topology_error(prim, original_topologies[path])
                    for path, prim in zip(
                        scoped_mesh_paths,
                        scoped_meshes,
                        strict=True,
                    )
                }
                overwrite_existing = bool(op_params.get("overwriteExisting", False))
                authorship_required_paths = tuple(
                    path
                    for path in scoped_mesh_paths
                    if overwrite_existing or pre_uv_errors[path] is not None
                )
                validated_preexisting_uv_paths = tuple(
                    path
                    for path in scoped_mesh_paths
                    if not overwrite_existing and pre_uv_errors[path] is None
                )
                if overwrite_existing:
                    # The worker input is disposable. Removing the old opinions
                    # makes the postcondition prove SO authored every requested
                    # mesh instead of merely recounting pre-existing UVs.
                    _strip_scoped_uvs(scoped_meshes)
                    uncleared_paths = [
                        str(prim.GetPath())
                        for prim in scoped_meshes
                        if (
                            (st_attr := prim.GetAttribute("primvars:st"))
                            and st_attr.HasAuthoredValue()
                        )
                    ]
                    if uncleared_paths:
                        raise RuntimeError(
                            "Could not clear pre-existing UVs from disposable Scene "
                            f"Optimizer scope: {uncleared_paths}"
                        )

                ctx = ExecutionContext()
                ctx.set_stage(stage)
                so = SceneOptimizerCore.getInstance()
                op_start = time.time()
                so.executeOperation(operation, ctx, op_params)
                op_time = time.time() - op_start

                post_meshes = _scoped_meshes(stage, Usd, UsdGeom, requested_paths)
                post_mesh_by_path = {str(prim.GetPath()): prim for prim in post_meshes}
                if set(post_mesh_by_path) != set(scoped_mesh_paths):
                    missing_paths = sorted(
                        set(scoped_mesh_paths) - set(post_mesh_by_path)
                    )
                    added_paths = sorted(
                        set(post_mesh_by_path) - set(scoped_mesh_paths)
                    )
                    raise RuntimeError(
                        f"{operation} changed its resolved mesh scope; "
                        f"missing={missing_paths}, added={added_paths}"
                    )

                post_topologies = _snapshot_scoped_topologies(post_meshes)
                changed_topology_paths = [
                    path
                    for path in scoped_mesh_paths
                    if post_topologies[path] != original_topologies[path]
                ]
                if changed_topology_paths:
                    raise RuntimeError(
                        f"{operation} changed mesh topology outside the UV contract: "
                        f"{changed_topology_paths}"
                    )

                uv_errors = {
                    path: error
                    for path in scoped_mesh_paths
                    if (
                        error := _uv_topology_error(
                            post_mesh_by_path[path],
                            original_topologies[path],
                        )
                    )
                    is not None
                }
                if uv_errors:
                    details = "; ".join(
                        f"{path}: {error}" for path, error in uv_errors.items()
                    )
                    raise RuntimeError(
                        f"{operation} did not author topology-valid UVs on all "
                        f"{mesh_count} scoped meshes: {details}"
                    )
                uv_authored_mesh_paths = authorship_required_paths
                uv_validated_mesh_paths = tuple(scoped_mesh_paths)
                meshes_with_uvs = len(uv_validated_mesh_paths)
            finally:
                _restore_materialized_specs(materialized_specs)

            failure_phase = "export"
            if not export_stage_portably(
                stage,
                output_usd_path,
                approved_dependency_roots=approved_dependency_roots,
            ):
                raise RuntimeError(f"Failed to export USD stage: {output_usd_path}")

            output_size = (
                os.path.getsize(output_usd_path)
                if os.path.exists(output_usd_path)
                else 0
            )
            failure_phase = "cleanup"
        finally:
            if ctx is not None:
                ctx.remove_stage()

        manifest = {
            "status": "success",
            "operation": operation,
            "operation_time": op_time,
            "total_time": time.time() - total_start,
            "stage_size_bytes": output_size,
            "mesh_count": mesh_count,
            "meshes_with_uvs": meshes_with_uvs,
            "requested_path_count": len(requested_paths),
            "requested_paths": [str(path) for path in requested_paths],
            "requested_path_meshes": requested_path_meshes,
            "scoped_mesh_paths": list(scoped_mesh_paths),
            "uv_authored_mesh_paths": list(uv_authored_mesh_paths),
            "uv_validated_mesh_paths": list(uv_validated_mesh_paths),
            "validated_preexisting_uv_paths": list(validated_preexisting_uv_paths),
            "authorship_required_mesh_paths": list(authorship_required_paths),
            "overwrite_existing": overwrite_existing,
        }

    except Exception as exc:  # noqa: BLE001 — subprocess must always write manifest
        manifest = {
            "status": "error",
            "operation": operation,
            "total_time": time.time() - total_start,
            "failure_phase": failure_phase,
            "error_type": type(exc).__name__,
            "error": traceback.format_exc()[-2000:],
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


if __name__ == "__main__":
    main()
