# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene Optimizer subprocess worker — runs with packman USD bindings.

This script runs in an isolated subprocess with:
- PYTHONPATH pointing to the SO package's ``python/`` and packman USD's
  ``lib/python/`` (providing ``pxr`` with matching ABI)
- LD_LIBRARY_PATH pointing to packman USD ``lib/``, SO ``lib/``, and
  SO ``extraLibs/``

It must NOT be imported by the main process (ABI conflict with the app's
pxr/OpenUSD bindings). It is executed via ``subprocess.run`` from
``scene_optimizer_local.py``.
"""

import json
import os
import sys
import time
import traceback
from typing import Any

if __package__:
    from .so_export import _normalize_dependency_roots, export_stage_portably
    from .so_export import export_layer_for as _export_layer_for
else:  # copied beside ``so_export.py`` in the ABI-isolated subprocess
    from so_export import _normalize_dependency_roots, export_stage_portably
    from so_export import export_layer_for as _export_layer_for

# ---------------------------------------------------------------------------
# Correspondence map helpers (pxr-only, no omni.usd / carb)
#
# Mirrors the traversal conventions in world_understanding.utils.usd.prim
# (which cannot be imported here due to ABI isolation).
# ---------------------------------------------------------------------------


def _natural_sort_key(text):
    """Generate a natural sorting key for strings containing numbers.

    ``Mesh_part_10`` sorts after ``Mesh_part_2`` (not before, as with
    lexicographic sort).  Matches the Kit service's ``_natural_sort_key``.
    """
    import re

    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


def capture_mesh_paths(stage, include_instance_proxies=False):
    """Return sorted list of all Mesh prim paths on *stage*.

    Mirrors ``world_understanding.utils.usd.prim.get_all_mesh_prim_paths``
    traversal: starts from pseudo-root, skips pseudo-root prim. By default,
    uses ``Usd.PrimDefaultPredicate``; set ``include_instance_proxies=True``
    to traverse into instance proxies.

    Args:
        stage: USD stage to traverse.
        include_instance_proxies: If True, include instance proxy meshes
            (needed before dedup to capture the full initial state).
    """
    from pxr import Usd, UsdGeom

    predicate = (
        Usd.TraverseInstanceProxies()
        if include_instance_proxies
        else Usd.PrimDefaultPredicate
    )
    paths = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
        if prim.IsPseudoRoot():
            continue
        if prim.IsA(UsdGeom.Mesh):
            paths.append(str(prim.GetPath()))
    return sorted(paths, key=_natural_sort_key)


def track_split_meshes(meshes_before, meshes_after):
    """Detect meshes that were split by ``splitMeshes``.

    Split meshes are removed and replaced by new meshes under the same
    parent with a ``_part`` / ``_part_N`` suffix.  Parts are returned in
    natural sort order (``_part_2`` before ``_part_10``).

    Returns:
        dict mapping each removed original path to its list of split-part
        paths (naturally ordered).
    """
    before_set = set(meshes_before)
    after_set = set(meshes_after)
    removed = before_set - after_set
    added = sorted(after_set - before_set, key=_natural_sort_key)

    split_mapping = {}
    for orig in sorted(removed, key=_natural_sort_key):
        parent = orig.rsplit("/", 1)[0] if "/" in orig else ""
        name = orig.rsplit("/", 1)[-1]
        parts = [p for p in added if p.startswith(parent + "/" + name + "_part")]
        if parts:
            split_mapping[orig] = parts
    return split_mapping


def _merge_split_mappings(existing, new):
    """Chain split mappings across repeated ``splitMeshes`` operations."""
    if not existing:
        return new
    if not new:
        return existing

    existing_targets = {target for targets in existing.values() for target in targets}
    merged = {}
    for original, targets in existing.items():
        chained_targets = []
        for target in targets:
            chained_targets.extend(new.get(target, [target]))
        merged[original] = chained_targets

    for original, targets in new.items():
        if original not in merged and original not in existing_targets:
            merged[original] = targets

    return merged


def track_deduplicate_geometry(stage):
    """Detect deduplication: instanceable Xforms referencing prototypes.

    After ``deduplicateGeometry``, duplicate meshes are replaced by
    instanceable Xform prims whose internal reference points to a
    prototype.  The mesh child (usually ``Geometry``) under the
    instanceable Xform is the instance; it maps to the corresponding
    mesh under the prototype.

    Returns:
        dict mapping instance mesh path -> prototype mesh path.
    """
    from pxr import Usd, UsdGeom

    instance_to_prototype = {}
    # Must traverse into instances to find proxy meshes
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        parent = prim.GetParent()
        if not parent or not parent.IsInstance():
            continue
        # Find the internal reference on the parent Xform
        prim_stack = parent.GetPrimStack()
        for spec in prim_stack:
            if not spec.hasReferences:
                continue
            ref_list = spec.referenceList
            # Check all reference list types: explicit, prepended, appended
            items = (
                list(ref_list.explicitItems)
                + list(ref_list.prependedItems)
                + list(ref_list.appendedItems)
            )
            for ref in items:
                # Internal reference: empty asset path, prim path set
                if not ref.assetPath and ref.primPath:
                    prototype_parent = str(ref.primPath)
                    mesh_name = prim.GetName()
                    prototype_mesh = prototype_parent + "/" + mesh_name
                    instance_path = str(prim.GetPath())
                    if instance_path != prototype_mesh:
                        instance_to_prototype[instance_path] = prototype_mesh
    return instance_to_prototype


def build_correspondence_map(
    initial_meshes,
    split_mapping,
    instance_to_prototype,
    ran_split,
    ran_dedup,
):
    """Build the full correspondence map matching the NVCF format.

    The ``original_to_prototype`` mapping chains through:
      original -> split parts (if split) -> dedup prototype (if deduped)

    For non-split, non-deduped meshes the mapping is identity.
    """
    original_to_prototype = {}

    # Pre-compute set of prototype targets for O(1) lookup in _resolve_dedup
    prototype_targets = set(instance_to_prototype.values()) if ran_dedup else set()

    def _resolve_dedup(path):
        """Resolve a path through dedup, handling Mesh->Xform/Geometry restructuring."""
        # Direct match: path itself is an instance
        if path in instance_to_prototype:
            return instance_to_prototype[path]
        # After dedup, Mesh prims become Xform with /Geometry child
        geom_path = path + "/Geometry"
        if geom_path in instance_to_prototype:
            return instance_to_prototype[geom_path]
        # Prototype itself was restructured (Mesh -> Xform/Geometry)
        if geom_path in prototype_targets:
            return geom_path
        return path

    for orig in initial_meshes:
        if orig in split_mapping:
            parts = split_mapping[orig]
            original_to_prototype[orig] = [_resolve_dedup(part) for part in parts]
        else:
            original_to_prototype[orig] = [_resolve_dedup(orig)]

    summary = {
        "note": "Tracks deinstancing, split, and deduplication for material mapping",
        "operations_run": {
            "deinstance": False,
            "split": ran_split,
            "deduplicate": ran_dedup,
        },
        "total_original_prims": len(initial_meshes),
        "meshes_before_deinstance": len(initial_meshes),
        "meshes_after_deinstance": len(initial_meshes),
        "meshes_deinstanced": 0,
        "meshes_split": len(split_mapping),
        "instances_tracked": len(instance_to_prototype),
    }

    return {
        "summary": summary,
        "deinstance_mapping": {},
        "split_mapping": split_mapping,
        "deduplication_mapping": {
            "instance_to_prototype": instance_to_prototype,
        },
        "full_mapping": {
            "original_to_prototype": original_to_prototype,
        },
    }


# ---------------------------------------------------------------------------
# Main worker entry point
# ---------------------------------------------------------------------------


def export_layer_for(stage: Any) -> Any:
    """Compatibility facade for the shared worker-safe export decision."""
    return _export_layer_for(stage)


def _parse_operation_result(result: Any) -> tuple[bool, str | None, Any]:
    """Normalize the supported Scene Optimizer operation return values."""
    if isinstance(result, tuple):
        if len(result) != 3:
            return (
                False,
                "Scene Optimizer executeOperation returned an invalid tuple "
                f"of length {len(result)}; expected 3",
                None,
            )

        success, error, output = result
        if not isinstance(success, bool):
            return (
                False,
                "Scene Optimizer executeOperation returned a non-boolean "
                f"success value: {success!r}",
                output,
            )
        if error is not None and not isinstance(error, str):
            return (
                False,
                "Scene Optimizer executeOperation returned a non-string "
                f"error value: {error!r}",
                output,
            )
        if not success and not error:
            error = "Scene Optimizer operation reported failure"
        return success, error, output

    # Some synchronous pybind releases expose executeOperation as void. An
    # exception remains authoritative failure; None preserves compatibility
    # with that contract and downstream USD validation still gates admission.
    if result is None:
        return True, None, None

    # Older bindings may return a bare bool. Only True is an unambiguous
    # legacy success; False and all other shapes must fail closed.
    if result is True:
        return True, None, None
    if result is False:
        return False, "Scene Optimizer operation reported failure", None
    return (
        False,
        "Scene Optimizer executeOperation returned an unsupported result "
        f"of type {type(result).__name__}",
        None,
    )


def _json_safe_operation_output(value: Any) -> Any:
    """Retain native diagnostics without making manifest publication fragile."""
    if value is None:
        return None
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return {
            "serializable": False,
            "type": type(value).__name__,
        }
    return value


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: so_worker.py '<json_params>'\n")
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
    if not manifest_path:
        sys.stderr.write("Error: Missing required JSON parameter: manifest_path\n")
        sys.exit(1)

    results = []
    total_start = time.time()

    try:
        missing = [
            key
            for key in (
                "input_usd_path",
                "output_usd_path",
                "approved_dependency_roots",
                "operations",
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
        operations = params["operations"]

        from omni.scene.optimizer.core import ExecutionContext, SceneOptimizerCore
        from pxr import Usd

        stage = Usd.Stage.Open(input_usd_path)
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {input_usd_path}")

        # Capture initial mesh paths for correspondence tracking
        initial_meshes = capture_mesh_paths(stage)
        split_mapping = {}
        instance_to_prototype = {}
        ran_split = False
        ran_dedup = False
        meshes_before_op = list(initial_meshes)

        ctx = ExecutionContext()
        ctx.set_stage(stage)
        so = SceneOptimizerCore.getInstance()

        for op_name, op_params in operations:
            op_start = time.time()
            try:
                native_result = so.executeOperation(op_name, ctx, op_params)
                op_success, op_error, op_output = _parse_operation_result(native_result)
                operation_metadata = {
                    "name": op_name,
                    "success": op_success,
                    "result_contract": (
                        "legacy_void"
                        if native_result is None
                        else "structured_or_boolean"
                    ),
                }
                if op_error:
                    operation_metadata["error"] = op_error
                if op_output is not None:
                    operation_metadata["output"] = _json_safe_operation_output(
                        op_output
                    )

                if not op_success:
                    operation_metadata["time"] = time.time() - op_start
                    results.append(operation_metadata)
                    break

                # Track correspondence after each relevant operation
                if op_name == "splitMeshes":
                    ran_split = True
                    meshes_after_op = capture_mesh_paths(stage)
                    split_mapping = _merge_split_mappings(
                        split_mapping,
                        track_split_meshes(
                            meshes_before_op,
                            meshes_after_op,
                        ),
                    )
                    meshes_before_op = meshes_after_op
                elif op_name == "deduplicateGeometry":
                    ran_dedup = True
                    meshes_after_op = capture_mesh_paths(stage)
                    instance_to_prototype = track_deduplicate_geometry(stage)
                    meshes_before_op = meshes_after_op
                else:
                    meshes_before_op = capture_mesh_paths(stage)

                operation_metadata["time"] = time.time() - op_start
                results.append(operation_metadata)
            except Exception:  # noqa: BLE001 — must catch all to write manifest
                results.append(
                    {
                        "name": op_name,
                        "success": False,
                        "time": time.time() - op_start,
                        "error": traceback.format_exc()[-500:],
                    }
                )
                break  # Stage is in unknown state — stop processing

        any_failed = any(not r.get("success", True) for r in results)

        # Only export if all operations succeeded — a failed operation
        # leaves the stage in an unknown state that could produce a
        # corrupted USD file.
        if not any_failed:
            if not export_stage_portably(
                stage,
                output_usd_path,
                approved_dependency_roots=approved_dependency_roots,
            ):
                raise RuntimeError(f"Failed to export USD stage: {output_usd_path}")

        ctx.remove_stage()

        if any_failed:
            failed_ops = [r["name"] for r in results if not r.get("success", True)]
            manifest = {
                "status": "error",
                "optimization_time": time.time() - total_start,
                "operations_executed": results,
                "stage_size_bytes": 0,
                "error": f"Operation(s) failed: {', '.join(failed_ops)}",
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            return

        output_size = (
            os.path.getsize(output_usd_path) if os.path.exists(output_usd_path) else 0
        )

        correspondence_map = build_correspondence_map(
            initial_meshes,
            split_mapping,
            instance_to_prototype,
            ran_split,
            ran_dedup,
        )

        manifest = {
            "status": "success",
            "optimization_time": time.time() - total_start,
            "operations_executed": results,
            "stage_size_bytes": output_size,
            "correspondence_map": correspondence_map,
        }

    except Exception:  # noqa: BLE001 — subprocess must always write manifest
        manifest = {
            "status": "error",
            "optimization_time": time.time() - total_start,
            "operations_executed": results,
            "stage_size_bytes": 0,
            "error": traceback.format_exc()[-2000:],
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


if __name__ == "__main__":
    main()
