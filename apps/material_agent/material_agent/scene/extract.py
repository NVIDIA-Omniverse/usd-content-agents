# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sub-asset extraction — extract individual sub-assets from a large USD scene.

Uses Usd.Stage.OpenMasked + Flatten to create standalone USD files
for each detected sub-asset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pxr import Sdf, Usd

from .manifest import SceneManifest

logger = logging.getLogger(__name__)


def extract_sub_asset(
    scene_usd_path: Path,
    prim_path: str,
    output_path: Path,
    flatten: bool = True,
    skip_instance_subtrees: bool = False,
) -> Path:
    """Extract one sub-asset from a USD scene using OpenMasked + Flatten.

    The extracted USD preserves the full prim hierarchy from the scene root
    so that material layer "over" prims map 1:1 back to the original scene.

    Args:
        scene_usd_path: Path to the source USD scene.
        prim_path: Prim path of the sub-asset root.
        output_path: Where to write the extracted USD.
        flatten: Whether to flatten the extracted stage.
        skip_instance_subtrees: If True, remove children of instance root
            prims from the flattened layer. Instance root prims are kept as
            empty Xforms so they still exist, but their children (which will
            be handled by per-payload processing) are stripped to avoid
            wasted VLM calls.

    Returns:
        Path to the extracted USD file.
    """
    from pxr import Usd

    logger.info(f"Extracting sub-asset: {prim_path} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Force-release any cached USD state from previous extractions to avoid
    # clipCache assertion failures when extracting multiple assets sequentially.
    import gc

    gc.collect()

    # Walk the Sdf layer to find all instanceable prims in the subtree.
    # Include their local-reference targets (prototype sources) in the
    # population mask so Flatten() can inline their geometry.
    from pxr import Sdf

    mask_paths = [prim_path]
    instance_prim_paths: list[str] = []
    root_layer = Sdf.Layer.FindOrOpen(str(scene_usd_path))
    if root_layer:
        _collect_instanceable_prims(
            root_layer, prim_path, instance_prim_paths, mask_paths
        )

    mask = Usd.StagePopulationMask(mask_paths)
    stage = Usd.Stage.OpenMasked(str(scene_usd_path), mask)
    if not stage:
        raise RuntimeError(
            f"Failed to open masked stage for {prim_path} in {scene_usd_path}"
        )

    if flatten:
        # Clear instanceable on all instance prims so Flatten() inlines
        # their prototype geometry.  Modify the session layer directly via
        # Sdf to avoid USD stage-level cache conflicts that cause clipCache
        # assertion failures during sequential extractions.
        if instance_prim_paths:
            session = stage.GetSessionLayer()
            for ip in instance_prim_paths:
                over = Sdf.CreatePrimInLayer(session, ip)
                if over:
                    over.SetInfo("instanceable", False)

        # Flatten resolves all composition arcs into a single layer
        flat_layer = stage.Flatten()

        # Remove instance subtree children if requested
        if skip_instance_subtrees:
            _strip_instance_children(flat_layer, stage, prim_path)

        # Export the flattened layer
        flat_layer.Export(str(output_path))
    else:
        # Export the root layer directly (keeps composition references)
        stage.GetRootLayer().Export(str(output_path))

    # Explicitly release the stage AND evict the session layer from the Sdf
    # layer cache.  SetInstanceable(False) modifies the session layer; if it
    # stays cached, the next OpenMasked call against the same scene file can
    # trigger a USD clipCache assertion failure.
    session_layer = stage.GetSessionLayer()
    del stage
    if session_layer:
        session_layer.Clear()
    gc.collect()

    logger.info(f"Extracted sub-asset to: {output_path}")
    return output_path


def _collect_instanceable_prims(
    layer: Sdf.Layer,
    root_path: str,
    instance_paths: list[str],
    mask_paths: list[str],
) -> None:
    """Recursively find instanceable prims and their prototype sources.

    Walks the Sdf layer (no stage needed) starting at *root_path*.
    For each prim with ``instanceable = true``, appends the prim path to
    *instance_paths* and appends the local-reference target to *mask_paths*
    so that ``Usd.Stage.OpenMasked`` includes the prototype geometry.

    Does NOT recurse into instanceable prims — their children live in the
    prototype source, which is included via *mask_paths*.
    """
    spec = layer.GetPrimAtPath(root_path)
    if not spec:
        return

    if spec.HasInfo("instanceable") and spec.GetInfo("instanceable"):
        instance_paths.append(root_path)
        refs = spec.referenceList.prependedItems
        if refs and refs[0].primPath:
            proto_path = str(refs[0].primPath)
            if proto_path not in mask_paths:
                mask_paths.append(proto_path)
        # Don't recurse into instanceable prims — their children
        # are in the prototype source, not under this path.
        return

    for child_name in spec.nameChildren.keys():
        _collect_instanceable_prims(
            layer, root_path + "/" + child_name, instance_paths, mask_paths
        )


def _strip_instance_children(
    flat_layer: Sdf.Layer,
    original_stage: Usd.Stage,
    prim_path: str,
) -> None:
    """Remove children of instance root prims from a flattened layer.

    Instance root prims are kept as empty Xforms so they still appear in
    the hierarchy, but their children are removed because those subtrees
    will be processed independently via per-payload pipeline runs.

    Args:
        flat_layer: The flattened Sdf.Layer to modify.
        original_stage: The original (masked) stage with instance info.
        prim_path: Root prim path of the sub-asset.
    """
    from pxr import Sdf

    # Find instance root prims in the original stage under prim_path
    instance_roots: list[str] = []
    root_prim = original_stage.GetPrimAtPath(prim_path)
    if not root_prim:
        return

    for prim in root_prim.GetAllChildren():
        _find_instances_recursive(prim, instance_roots)

    if not instance_roots:
        return

    logger.info(
        f"Stripping children of {len(instance_roots)} instance prims under {prim_path}"
    )

    for inst_path in instance_roots:
        spec = flat_layer.GetPrimAtPath(inst_path)
        if not spec:
            continue

        # Remove all children of the instance root
        children_to_remove = list(spec.nameChildren.keys())
        for child_name in children_to_remove:
            flat_layer.RemovePrimIfInert(Sdf.Path(inst_path).AppendChild(child_name))
            # RemovePrimIfInert only removes if truly inert; use
            # RemoveSpec for non-inert children
            child_path = Sdf.Path(inst_path).AppendChild(child_name)
            if flat_layer.GetPrimAtPath(child_path):
                _remove_prim_recursive(flat_layer, child_path)

        logger.debug(
            f"Stripped {len(children_to_remove)} children from instance {inst_path}"
        )


def _find_instances_recursive(prim: Usd.Prim, result: list[str]) -> None:
    """Recursively find instance root prims."""
    if prim.IsInstance():
        result.append(str(prim.GetPath()))
        return  # Don't recurse into instance children
    for child in prim.GetAllChildren():
        _find_instances_recursive(child, result)


def _remove_prim_recursive(layer: Sdf.Layer, path: Sdf.Path) -> None:
    """Recursively remove a prim and all its descendants from a layer."""
    spec = layer.GetPrimAtPath(path)
    if not spec:
        return
    # Remove children first (depth-first)
    for child_name in list(spec.nameChildren.keys()):
        _remove_prim_recursive(layer, path.AppendChild(child_name))
    # Remove the prim itself
    parent_path = path.GetParentPath()
    parent_spec = layer.GetPrimAtPath(parent_path)
    if parent_spec:
        parent_spec.RemoveNameChild(spec)
    else:
        layer.pseudoRoot.RemoveNameChild(spec)


def extract_all(
    scene_usd_path: Path,
    manifest: SceneManifest,
    output_dir: Path,
    names_filter: list[str] | None = None,
    flatten: bool = True,
    max_workers: int = 1,
) -> SceneManifest:
    """Extract all processable assets from a scene.

    Updates the manifest with extraction paths and status.
    When payload groups exist, instance subtrees are stripped from
    extracted USDs to avoid wasted VLM calls (those subtrees are
    processed independently via per-payload pipeline runs).

    Args:
        scene_usd_path: Path to the source USD scene.
        manifest: Scene manifest with detected sub-assets.
        output_dir: Base directory for extracted USDs.
        names_filter: Optional name/path filter for assets.
        flatten: Whether to flatten extracted stages.
        max_workers: Number of parallel extraction workers (default 1 = serial).

    Returns:
        Updated SceneManifest.
    """
    import concurrent.futures
    import threading

    assets = manifest.get_processable_assets(names_filter)

    # Build a set of instance paths that have payload groups covering them.
    # Only strip instance subtrees for these — other instances (e.g. single-mesh
    # prototypes with no payload group) need their geometry preserved.
    payload_instance_paths: set[str] = set()
    for pg in manifest.payload_groups:
        if pg.status != "skipped":
            for ip in pg.instance_paths:
                payload_instance_paths.add(ip)

    logger.info(
        f"Extracting {len(assets)} sub-assets from {scene_usd_path} "
        f"(workers={max_workers})"
    )

    # Build unique safe names: append ID suffix when names collide
    safe_names = _unique_safe_names(assets)

    counter_lock = threading.Lock()
    completed = [0]

    def _extract_one(sa):
        safe_name = safe_names[sa.id]
        asset_dir = output_dir / safe_name
        extracted_path = asset_dir / f"{safe_name}.usd"

        # Only strip instance subtrees if this sub-asset contains instances
        # that are covered by payload groups.  Sub-assets whose instances
        # have no payload group (e.g. single-mesh prototypes) keep their
        # geometry so the per-asset pipeline can process them.
        skip_instances = bool(payload_instance_paths) and any(
            ip == sa.prim_path or ip.startswith(sa.prim_path + "/")
            for ip in payload_instance_paths
        )

        with counter_lock:
            completed[0] += 1
            idx = completed[0]
        logger.info(f"[{idx}/{len(assets)}] Extracting '{sa.name}' ({sa.prim_path})")
        try:
            extract_sub_asset(
                scene_usd_path=scene_usd_path,
                prim_path=sa.prim_path,
                output_path=extracted_path,
                flatten=flatten,
                skip_instance_subtrees=skip_instances,
            )
            sa.extracted_usd = str(extracted_path)
            sa.status = "extracted"
        except Exception:
            logger.exception(f"Failed to extract '{sa.name}'")
            sa.status = "failed"

    if max_workers <= 1:
        for sa in assets:
            _extract_one(sa)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_extract_one, assets))

    return manifest


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a directory/file name."""
    import re

    # Replace non-alphanumeric chars (except underscore/hyphen) with underscore
    safe = re.sub(r"[^\w\-]", "_", name)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe.lower() if safe else "unnamed"


def _unique_safe_names(assets: list) -> dict[str, str]:
    """Build a mapping of asset ID → unique safe name.

    When multiple assets share the same sanitized name, appends the asset ID
    as a suffix (e.g. ``default_obj_230``) to disambiguate.
    """
    from collections import Counter

    name_counts = Counter(_sanitize_name(sa.name) for sa in assets)
    result: dict[str, str] = {}
    for sa in assets:
        safe = _sanitize_name(sa.name)
        if name_counts[safe] > 1:
            # Append ID to disambiguate
            suffix = _sanitize_name(sa.id)
            safe = f"{safe}_{suffix}"
        result[sa.id] = safe
    return result
