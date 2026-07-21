# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene analysis — detect sub-assets in a large USD scene.

Wraps the existing detect_objects() pipeline from world_understanding
to produce a SceneManifest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .manifest import InstanceGroup, PayloadGroup, SceneManifest, SubAsset

logger = logging.getLogger(__name__)


def analyze_scene(
    scene_usd_path: Path,
    skip_geometry: bool = False,
    building_block_min_reuse: int = 20,
    filters: dict[str, Any] | None = None,
    llm_config: dict[str, Any] | None = None,
    token_tracker: Any | None = None,
    working_dir: Path | None = None,
    detect_payload_groups: bool = True,
    detect_native_prototypes: bool = True,
    extract_large_payload_representatives: bool = True,
) -> SceneManifest:
    """Analyze a USD scene and detect sub-assets.

    Args:
        scene_usd_path: Path to the USD scene file.
        skip_geometry: Skip vertex/face counting (faster but less info).
        building_block_min_reuse: Minimum reuse count for building-block detection.
        filters: Optional dict with keys:
            - include_paths: list[str] — only include prims under these paths
            - exclude_paths: list[str] — exclude prims under these paths
            - min_mesh_count: int — skip objects with fewer meshes
            - exclude_invisible_assets: bool — skip objects with no visible
              descendant meshes. Legacy alias: skip_invisible.
        llm_config: Optional LLM config for split refinement. Dict with keys
            ``backend``, ``model``, and optionally ``temperature``,
            ``max_tokens``. If None, LLM refinement is skipped.
        token_tracker: Optional TokenTracker for LLM refinement usage.
        working_dir: Optional directory for analysis side artifacts such as
            extracted prototype or representative USDs. Defaults to the legacy
            sibling ``.{scene}_working`` directory.
        detect_payload_groups: Detect payload-backed instance groups.
        detect_native_prototypes: Detect USD native prototype groups.
        extract_large_payload_representatives: Extract smaller representative
            files for large payloads with internal instancing.

    Returns:
        SceneManifest with detected sub-assets and instance groups.
    """
    from pxr import Usd, UsdGeom
    from world_understanding.functions.graphics.usd_scene_analysis import (
        detect_objects,
    )
    from world_understanding.utils.usd.composition import collect_composition_arcs
    from world_understanding.utils.usd.prim import collect_mesh_geometry_stats

    filters = filters or {}
    include_paths: list[str] = filters.get("include_paths", [])
    exclude_paths: list[str] = filters.get("exclude_paths", [])
    min_mesh_count: int = filters.get("min_mesh_count", 0)
    exclude_invisible_assets: bool = bool(
        filters.get(
            "exclude_invisible_assets",
            filters.get("skip_invisible", False),
        )
    )

    logger.info(f"Opening USD stage: {scene_usd_path}")
    stage = Usd.Stage.Open(str(scene_usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {scene_usd_path}")

    logger.info("Collecting composition arcs...")
    composition_data = collect_composition_arcs(stage)

    logger.info("Collecting mesh geometry stats...")
    geometry_stats = collect_mesh_geometry_stats(stage, skip_geometry=skip_geometry)

    logger.info("Detecting objects...")
    objects, instance_groups_raw = detect_objects(
        stage,
        composition_data,
        geometry_stats,
        skip_geometry=skip_geometry,
        building_block_min_reuse=building_block_min_reuse,
    )

    logger.info(
        f"Detected {len(objects)} objects, {len(instance_groups_raw)} instance groups"
    )

    # LLM refinement: ask LLM whether composite objects should be split
    if llm_config:
        from .llm_refine import refine_objects_with_llm

        logger.info("Running LLM refinement...")
        objects, instance_groups_raw = refine_objects_with_llm(
            stage=stage,
            objects=objects,
            instance_groups=instance_groups_raw,
            llm_config=llm_config,
            max_split_depth=llm_config.get("max_split_depth", 5),
            min_mesh_for_review=llm_config.get("min_mesh_for_review", 100),
            max_workers=llm_config.get("max_workers", 16),
            auto_split_threshold=llm_config.get("auto_split_threshold", 20),
            token_tracker=token_tracker,
        )

    def _has_visible_descendant_mesh(prim_path: str) -> bool:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            return False

        root_imageable = UsdGeom.Imageable(prim)
        if root_imageable and (
            root_imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
        ):
            return False

        prim_range = iter(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        for descendant in prim_range:
            imageable = UsdGeom.Imageable(descendant)
            if (
                imageable
                and imageable.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
            ):
                prim_range.PruneChildren()
                continue
            if not descendant.IsA(UsdGeom.Mesh):
                continue
            return True
        return False

    # Convert to SubAsset dataclasses with filtering
    sub_assets: list[SubAsset] = []
    skipped_invisible = 0
    for obj in objects:
        prim_path = obj.get("path", "")

        # Apply include filter
        if include_paths:
            if not any(
                prim_path == p or prim_path.startswith(p + "/") for p in include_paths
            ):
                continue

        # Apply exclude filter
        if exclude_paths:
            if any(
                prim_path == p or prim_path.startswith(p + "/") for p in exclude_paths
            ):
                continue

        # Apply min mesh count filter
        mesh_count = obj.get("mesh_count", 0)
        if mesh_count < min_mesh_count:
            continue

        if exclude_invisible_assets and not _has_visible_descendant_mesh(prim_path):
            skipped_invisible += 1
            continue

        sub_assets.append(
            SubAsset(
                id=obj.get("id", ""),
                name=obj.get("name", ""),
                prim_path=prim_path,
                parent_group=obj.get("parent_group"),
                source_classification=obj.get("source_classification"),
                mesh_count=mesh_count,
                vertex_count=obj.get("vertex_count", 0),
                instance_group=obj.get("instance_group"),
                split_context=obj.get("split_context"),
            )
        )

    if skipped_invisible:
        logger.info(
            "Skipped %d sub-assets with no visible descendant meshes "
            "(exclude_invisible_assets=True)",
            skipped_invisible,
        )

    # Convert instance groups
    instance_groups: list[InstanceGroup] = []
    for ig_raw in instance_groups_raw:
        members = ig_raw.get("member_paths", [])
        group_name = ig_raw.get("group_name", "")

        # Find representative: first member that is also in sub_assets,
        # or the first sub-asset whose prim_path is a descendant of a member
        # (happens when LLM refinement split the member into children).
        representative_id: str | None = None
        members_set = set(members)
        for sa in sub_assets:
            if sa.prim_path in members_set:
                representative_id = sa.id
                break
        if representative_id is None:
            for sa in sub_assets:
                for mp in members:
                    if sa.prim_path.startswith(mp + "/"):
                        representative_id = sa.id
                        break
                if representative_id is not None:
                    break

        source_file = ig_raw.get("source_file")

        instance_groups.append(
            InstanceGroup(
                group_name=group_name,
                source_file=source_file,
                instance_count=ig_raw.get("instance_count", len(members)),
                member_paths=members,
                representative_id=representative_id,
            )
        )

    # Detect structural duplicates (optional — groups flattened assets with
    # identical mesh hierarchy so only the representative is processed).
    detect_dupes = filters.get("detect_structural_duplicates", False)
    if detect_dupes:
        sub_assets, dupe_groups = _detect_structural_duplicates(stage, sub_assets)
        instance_groups.extend(dupe_groups)

        # Ensure instance group representatives are never tagged as structural
        # duplicate members — they must be processed so their predictions can
        # propagate to instance group members.
        ig_rep_ids = {
            ig.representative_id for ig in instance_groups if ig.representative_id
        }
        untagged = 0
        for sa in sub_assets:
            if sa.instance_group and sa.id in ig_rep_ids:
                logger.info(
                    f"Untagging '{sa.name}' from structural group "
                    f"'{sa.instance_group}' — needed as instance group representative"
                )
                sa.instance_group = None
                untagged += 1

        logger.info(
            f"Structural duplicate detection: {len(dupe_groups)} groups, "
            f"{sum(ig.instance_count for ig in dupe_groups)} duplicates "
            f"(will process {len([sa for sa in sub_assets if not sa.instance_group])} unique)"
        )
        if untagged:
            logger.info(
                f"Untagged {untagged} structural dup members needed as "
                f"instance group representatives"
            )

    payload_groups: list[PayloadGroup] = []
    if detect_payload_groups:
        # Detect payload groups (unique payload files referenced by instance prims)
        payload_groups = _detect_payload_groups(stage, scene_usd_path)
        logger.info(f"Detected {len(payload_groups)} unique payload groups")

        # Extract representative files for large payloads with internal instancing.
        # These payloads are too large for NVCF but contain repeated geometry — only
        # the non-instance prototype source prims need processing.
        if extract_large_payload_representatives:
            if working_dir is None:
                _extract_large_payload_representatives(
                    payload_groups,
                    scene_usd_path,
                )
            else:
                _extract_large_payload_representatives(
                    payload_groups,
                    scene_usd_path,
                    working_dir=working_dir,
                )

    if detect_native_prototypes:
        # Detect prototype groups (USD native instances sharing same prototype)
        if working_dir is None:
            prototype_groups = _detect_prototype_groups(stage, scene_usd_path)
        else:
            prototype_groups = _detect_prototype_groups(
                stage,
                scene_usd_path,
                working_dir=working_dir,
            )
        if prototype_groups:
            logger.info(f"Detected {len(prototype_groups)} prototype payload groups")
            payload_groups.extend(prototype_groups)

    # Build analysis summary
    analysis_summary = {
        "total_prims": geometry_stats.get("total_prims", 0),
        "total_meshes": geometry_stats.get("total_meshes", 0),
        "total_vertices": geometry_stats.get("total_vertices", 0),
        "total_objects_detected": len(objects),
        "total_objects_after_filter": len(sub_assets),
        "total_instance_groups": len(instance_groups),
        "total_payload_groups": len(payload_groups),
        "total_payload_instances": sum(pg.instance_count for pg in payload_groups),
        "composition": {
            "sublayer_count": composition_data.get("sublayer_count", 0),
            "reference_count": composition_data.get("reference_count", 0),
            "unique_sub_usd_count": composition_data.get("unique_sub_usd_count", 0),
        },
    }

    manifest = SceneManifest(
        scene_usd_path=str(scene_usd_path.resolve()),
        generated_at=SceneManifest.timestamp(),
        analysis=analysis_summary,
        sub_assets=sub_assets,
        instance_groups=instance_groups,
        payload_groups=payload_groups,
    )

    logger.info(
        f"Scene analysis complete: {len(sub_assets)} sub-assets, "
        f"{len(instance_groups)} instance groups, "
        f"{len(payload_groups)} payload groups"
    )
    return manifest


def _detect_structural_duplicates(
    stage: Any,
    sub_assets: list[SubAsset],
) -> tuple[list[SubAsset], list[InstanceGroup]]:
    """Detect sub-assets with identical mesh hierarchy and group them.

    For flattened USD scenes without native instancing, repeated geometry
    (e.g. safety fences, duplicated robots) produces many identical sub-assets.
    This function fingerprints each sub-asset by its sorted relative mesh paths
    and groups identical ones.  The first member becomes the representative;
    the rest are tagged with ``instance_group`` so the pipeline can skip them
    and copy results from the representative.

    Args:
        stage: The opened USD stage.
        sub_assets: List of SubAsset objects to check.

    Returns:
        Tuple of (updated sub_assets, new InstanceGroup list).
    """
    import hashlib

    from pxr import Usd, UsdGeom

    # Fingerprint each sub-asset by relative mesh path hierarchy.
    # Skip assets already assigned to a native instance group — they are
    # already handled and must not be re-tagged as structural duplicates.
    fingerprints: dict[str, list[SubAsset]] = {}
    for sa in sub_assets:
        if sa.instance_group:
            continue
        prim = stage.GetPrimAtPath(sa.prim_path)
        if not prim or not prim.IsValid():
            fingerprints.setdefault("__invalid__" + sa.id, []).append(sa)
            continue
        mesh_paths = sorted(
            str(p.GetPath()).replace(sa.prim_path + "/", "")
            for p in Usd.PrimRange(prim)
            if p.IsA(UsdGeom.Mesh)
        )
        fp = hashlib.blake2s("|".join(mesh_paths).encode(), digest_size=16).hexdigest()
        fingerprints.setdefault(fp, []).append(sa)

    # Build instance groups for duplicates
    new_groups: list[InstanceGroup] = []
    for _fp, members in fingerprints.items():
        if len(members) < 2:
            continue

        representative = members[0]
        group_name = f"structural_{representative.name}"

        for member in members[1:]:
            member.instance_group = group_name

        new_groups.append(
            InstanceGroup(
                group_name=group_name,
                source_file=None,
                instance_count=len(members) - 1,
                member_paths=[m.prim_path for m in members[1:]],
                representative_id=representative.id,
            )
        )
        logger.debug(
            f"Structural group '{group_name}': {len(members)}x "
            f"({representative.mesh_count} meshes)"
        )

    return sub_assets, new_groups


def _detect_payload_groups(
    stage: Any,
    scene_usd_path: Path,
) -> list[PayloadGroup]:
    """Detect unique payload files referenced by instance prims.

    Walks the stage, finds prims with ``IsInstance() == True``, extracts
    their payload arcs, and deduplicates by resolved payload file path.
    Returns one PayloadGroup per unique payload file.

    Args:
        stage: The opened USD stage.
        scene_usd_path: Path to the scene USD (for resolving relative paths).

    Returns:
        List of PayloadGroup objects.
    """
    import re

    scene_dir = scene_usd_path.resolve().parent

    # Map resolved payload path → (group_name, list of instance prim paths)
    payload_map: dict[str, list[str]] = {}

    for prim in stage.Traverse():
        if not prim.IsInstance():
            continue

        prim_path = str(prim.GetPath())

        # Get payloads from the prim's composition arcs
        prim_index = prim.GetPrimIndex()
        if not prim_index or not prim_index.rootNode:
            continue

        # Walk the prim index tree to find payload arcs
        payload_paths = _collect_payload_paths_from_node(prim_index.rootNode, scene_dir)

        for resolved_path in payload_paths:
            payload_map.setdefault(resolved_path, []).append(prim_path)

    # Create PayloadGroup objects, auto-skipping empty payloads
    groups: list[PayloadGroup] = []
    skipped = 0
    for payload_file, instance_paths in sorted(payload_map.items()):
        # Generate group name from the payload filename
        payload_name = Path(payload_file).stem
        safe_name = re.sub(r"[^\w\-]", "_", payload_name)
        safe_name = re.sub(r"_+", "_", safe_name).strip("_").lower()

        # Check mesh count — skip payloads with 0 meshes (lights, empties)
        mesh_count = _count_payload_meshes(payload_file)

        group = PayloadGroup(
            id=f"payload_{safe_name}",
            group_name=safe_name,
            payload_file=payload_file,
            instance_count=len(instance_paths),
            instance_paths=sorted(instance_paths),
        )

        if mesh_count == 0:
            group.status = "skipped"
            skipped += 1
            logger.info(
                f"Auto-skipping payload '{safe_name}': 0 meshes ({payload_file})"
            )
        else:
            logger.debug(
                f"Payload group '{safe_name}': {payload_file} "
                f"({len(instance_paths)} instances, {mesh_count} meshes)"
            )

        groups.append(group)

    if skipped:
        logger.info(f"Auto-skipped {skipped}/{len(groups)} empty payload groups")

    # Build the dependency DAG (discover nested payloads, compute depth)
    groups = _build_payload_dag(groups)

    return groups


def _build_payload_dag(groups: list[PayloadGroup]) -> list[PayloadGroup]:
    """Discover nested payload dependencies and compute DAG structure.

    Uses BFS to open each payload file as an Sdf.Layer, collect its
    payload/reference arc targets, and discover nested payloads. Computes
    depth (leaf=0, parent=1+max(child depth)) and fills child/parent edges.

    Payloads discovered during BFS that were not in the initial list (nested
    payloads not directly referenced by scene-level instance prims) are added
    as new PayloadGroup entries with ``instance_count=0``.

    Args:
        groups: Initial list of PayloadGroup objects (top-level, from scene).

    Returns:
        Extended list with nested payloads added, depth/edges populated.
    """
    import re

    from material_agent.scene.payload_dag_utils import build_dag, compute_depths

    # Build a lookup from resolved payload file path -> PayloadGroup
    file_to_group: dict[str, PayloadGroup] = {}
    for pg in groups:
        resolved = str(Path(pg.payload_file).resolve())
        file_to_group[resolved] = pg

    # BFS to build the full DAG
    root_files = set(file_to_group.keys())
    adj = build_dag(root_files)

    # Create PayloadGroup entries for any newly discovered nested payloads
    for node in adj:
        if node not in file_to_group:
            payload_name = Path(node).stem
            safe_name = re.sub(r"[^\w\-]", "_", payload_name)
            safe_name = re.sub(r"_+", "_", safe_name).strip("_").lower()

            # Check mesh count for auto-skip
            mesh_count = _count_payload_meshes(node)

            new_pg = PayloadGroup(
                id=f"payload_{safe_name}",
                group_name=safe_name,
                payload_file=node,
                instance_count=0,
                instance_paths=[],
            )
            if mesh_count == 0:
                new_pg.status = "skipped"
                logger.info(f"Auto-skipping nested payload '{safe_name}': 0 meshes")

            groups.append(new_pg)
            file_to_group[node] = new_pg
            logger.debug(f"Discovered nested payload: {safe_name} ({node})")

    # Compute depths and fill edges
    depths = compute_depths(adj)

    for node, children in adj.items():
        pg = file_to_group.get(node)
        if not pg:  # pragma: no cover - defensive after DAG node materialization
            continue
        pg.depth = depths.get(node, 0)
        pg.child_payload_files = sorted(children)
        # Build parent edges (reverse)
        for child in children:
            child_pg = file_to_group.get(child)
            if child_pg and node not in child_pg.parent_payload_files:
                child_pg.parent_payload_files.append(node)

    # Sort parent lists
    for pg in groups:
        pg.parent_payload_files.sort()

    leaves = sum(1 for pg in groups if pg.depth == 0 and pg.status != "skipped")
    parents = sum(1 for pg in groups if pg.depth > 0 and pg.status != "skipped")
    max_depth = max((pg.depth for pg in groups), default=0)
    logger.info(
        f"Payload DAG: {len(groups)} total, {leaves} leaves, "
        f"{parents} parents, max depth {max_depth}"
    )

    return groups


def _count_payload_meshes(payload_file: str) -> int:
    """Count meshes in a payload USD file.

    Opens the payload as a standalone stage and counts Mesh prims.
    Returns 0 for files that contain only lights, Xforms, or are empty.

    Args:
        payload_file: Absolute path to the payload USD file.

    Returns:
        Number of Mesh prims in the payload.
    """
    from pxr import Usd, UsdGeom

    try:
        stage = Usd.Stage.Open(payload_file)
        if not stage:
            return 0
        count = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                count += 1
        return count
    except Exception:
        logger.warning(f"Failed to count meshes in payload: {payload_file}")
        return 0


def _detect_prototype_groups(
    stage: Any,
    scene_usd_path: Path,
    working_dir: Path | None = None,
) -> list[PayloadGroup]:
    """Detect USD native prototype groups and return them as PayloadGroups.

    For scenes using USD native instancing (no payload arcs), prototypes
    contain the shared geometry. Each prototype is extracted to a standalone
    USD file so it can flow through the same pipeline as payload groups.

    Args:
        stage: The opened USD stage.
        scene_usd_path: Path to the scene USD (for output directory).

    Returns:
        List of PayloadGroup objects for prototypes with meshes.
    """
    import re

    from pxr import Usd, UsdGeom

    prototypes = stage.GetPrototypes()
    if not prototypes:
        return []

    # Build prototype -> instance paths mapping
    proto_to_instances: dict[str, list[str]] = {}
    for prim in stage.Traverse():
        if prim.IsInstance():
            proto = prim.GetPrototype()
            if proto and proto.IsValid():
                proto_path = str(proto.GetPath())
                proto_to_instances.setdefault(proto_path, []).append(
                    str(prim.GetPath())
                )

    # Working directory for extracted prototypes
    analysis_dir = (
        working_dir
        if working_dir is not None
        else scene_usd_path.resolve().parent / f".{scene_usd_path.stem}_working"
    )
    proto_dir = analysis_dir / "prototypes"

    groups: list[PayloadGroup] = []
    skipped = 0

    for proto in prototypes:
        proto_path = str(proto.GetPath())
        instance_paths = proto_to_instances.get(proto_path, [])

        # Count meshes in prototype — skip if none
        mesh_count = 0
        for prim in Usd.PrimRange(proto, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1

        # Generate safe name from prototype path
        # Prototype paths look like /__Prototype_1, /__Prototype_23, etc.
        proto_name = proto.GetName()
        # Use the first instance's name for a more descriptive group name
        if instance_paths:
            first_instance = stage.GetPrimAtPath(sorted(instance_paths)[0])
            if first_instance and first_instance.IsValid():
                proto_name = first_instance.GetName()

        safe_name = re.sub(r"[^\w\-]", "_", proto_name)
        safe_name = re.sub(r"_+", "_", safe_name).strip("_").lower()

        group = PayloadGroup(
            id=f"proto_{safe_name}",
            group_name=safe_name,
            payload_file="",  # Will be set after extraction
            instance_count=len(instance_paths),
            instance_paths=sorted(instance_paths),
            depth=0,  # Prototypes are always leaves (no DAG)
        )

        # Skip prototypes with 0 meshes (empty geometry containers).
        # Single-mesh prototypes are kept — they may be instance group
        # representatives whose members need material propagation.
        if mesh_count == 0:
            group.status = "skipped"
            skipped += 1
            logger.info(
                f"Auto-skipping prototype '{safe_name}': 0 meshes ({proto_path})"
            )
            groups.append(group)
            continue

        # Extract prototype to standalone USD
        # Pick the first instance as representative and flatten its subtree
        if not instance_paths:
            group.status = "skipped"
            skipped += 1
            groups.append(group)
            continue

        representative_path = sorted(instance_paths)[0]
        out_dir = proto_dir / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{safe_name}.usd"

        try:
            _extract_prototype(stage, representative_path, str(out_file))
            group.payload_file = str(out_file.resolve())
            logger.debug(
                f"Prototype group '{safe_name}': {proto_path} "
                f"({len(instance_paths)} instances, {mesh_count} meshes) "
                f"-> {out_file}"
            )
        except Exception:
            logger.warning(
                f"Failed to extract prototype '{safe_name}' from {representative_path}",
                exc_info=True,
            )
            group.status = "failed"

        groups.append(group)

    active = [g for g in groups if g.status not in ("skipped", "failed")]
    logger.info(
        f"Prototype groups: {len(prototypes)} total, {len(active)} active, "
        f"{skipped} skipped"
    )
    # Only return active groups — no point bloating the manifest with
    # thousands of skipped single-mesh entries (e.g. accom has 7664).
    return active


def _extract_prototype(
    stage: Any,
    representative_path: str,
    output_path: str,
) -> None:
    """Extract a prototype's geometry via a representative instance prim.

    For USD native instances, the geometry lives inside a shared prototype
    (``/__Prototype_N``).  A simple population-masked flatten of the instance
    prim produces an empty Xform shell because the prototype content is not
    included in the mask.

    The fix: include the prototype path in the population mask and clear
    ``instanceable`` on the representative prim *in the masked extraction
    stage only* so that ``Flatten()`` inlines the prototype geometry under
    the instance path.  This does NOT de-instance the original scene — the
    masked stage is a throwaway used solely for extraction.

    Args:
        stage: The opened USD stage.
        representative_path: Prim path of one instance to extract.
        output_path: Where to write the extracted USD.
    """
    from pxr import Usd

    root_layer = stage.GetRootLayer()

    # Check if the representative prim is a USD native instance so we can
    # include its prototype in the population mask.
    prim = stage.GetPrimAtPath(representative_path)
    mask_paths = [representative_path]
    if prim and prim.IsInstance():
        proto = prim.GetPrototype()
        if proto:
            mask_paths.append(str(proto.GetPath()))

    mask = Usd.StagePopulationMask(mask_paths)
    masked_stage = Usd.Stage.OpenMasked(
        root_layer,
        mask,
        Usd.Stage.LoadAll,
    )

    if not masked_stage:
        raise RuntimeError(f"Failed to open masked stage for {representative_path}")

    # Clear instanceable in the *extraction stage only* so Flatten() inlines
    # the prototype geometry under the instance's path.  The original scene
    # is never modified.
    masked_prim = masked_stage.GetPrimAtPath(representative_path)
    if masked_prim and masked_prim.IsInstance():
        masked_prim.SetInstanceable(False)

    flat_layer = masked_stage.Flatten()
    flat_layer.Export(output_path)
    logger.debug(f"Extracted prototype to {output_path}")


def _collect_payload_paths_from_node(
    node: Any,
    scene_dir: Path,
) -> list[str]:
    """Recursively collect resolved payload file paths from a PcpNodeRef tree.

    Args:
        node: A PcpNodeRef from the prim index.
        scene_dir: Directory of the scene USD for resolving relative paths.

    Returns:
        List of resolved absolute payload file paths.
    """
    from pxr import Pcp

    results: list[str] = []

    # Check if this node is a payload arc
    if node.arcType == Pcp.ArcTypePayload:
        layer = node.layerStack.layers[0] if node.layerStack.layers else None
        if layer:
            layer_path = layer.realPath
            if layer_path:
                resolved = Path(layer_path).resolve()
                if resolved.exists():
                    results.append(str(resolved))

    # Recurse into children
    for child in node.children:
        results.extend(_collect_payload_paths_from_node(child, scene_dir))

    return results


# ---------------------------------------------------------------------------
# Large payload representative extraction
# ---------------------------------------------------------------------------

# Payloads above this size (bytes) are checked for internal instancing.
_LARGE_PAYLOAD_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB


def _extract_large_payload_representatives(
    payload_groups: list[PayloadGroup],
    scene_usd_path: Path,
    working_dir: Path | None = None,
) -> None:
    """Extract representative files for large payloads with internal instancing.

    For payload files above ``_LARGE_PAYLOAD_THRESHOLD_BYTES`` that use USD
    native instancing internally, the full file is too large for NVCF
    (SO + renderer).  These payloads typically contain a small number of
    unique prototype source prims replicated via instancing.

    This function:
    1. Opens each large payload and checks for internal prototypes.
    2. Extracts only the non-instance prototype source prims to a smaller
       USD file preserving the original prim path hierarchy.
    3. Sets ``representative_path`` on the PayloadGroup so config_gen
       can use the smaller file for SO/render/predict.

    The original ``payload_file`` is unchanged — the apply step's output.usd
    will still sublayer it.

    Args:
        payload_groups: List of PayloadGroup objects to check.
        scene_usd_path: Scene USD path (for determining working directory).
    """
    import os

    for pg in payload_groups:
        if pg.status == "skipped" or not pg.payload_file:
            continue

        try:
            file_size = os.path.getsize(pg.payload_file)
        except OSError:
            continue

        if file_size < _LARGE_PAYLOAD_THRESHOLD_BYTES:
            continue

        size_mb = file_size / (1024 * 1024)
        logger.info(
            f"Large payload '{pg.group_name}' ({size_mb:.0f} MB) — "
            f"checking for internal instancing"
        )

        if working_dir is None:
            rep_path = _extract_prototype_sources(pg.payload_file, scene_usd_path)
        else:
            rep_path = _extract_prototype_sources(
                pg.payload_file,
                scene_usd_path,
                working_dir=working_dir,
            )
        if rep_path:
            pg.representative_path = str(rep_path)
            rep_size_mb = os.path.getsize(rep_path) / (1024 * 1024)
            logger.info(
                f"  Extracted representative: {rep_path.name} "
                f"({rep_size_mb:.1f} MB, {size_mb / rep_size_mb:.0f}x smaller)"
            )
        else:
            logger.info("  No internal instancing found, will process as-is")


def _extract_prototype_sources(
    payload_file: str,
    scene_usd_path: Path,
    working_dir: Path | None = None,
) -> Path | None:
    """Extract non-instance prototype source prims from a payload file.

    Opens the payload as a stage, finds all prims that are prototype sources
    (non-instance siblings among groups of instance prims), and creates a
    population-masked extraction containing only those prims.

    The extracted file preserves the full prim path hierarchy so that
    predictions map back to the same paths in the original payload.

    Args:
        payload_file: Absolute path to the payload USD file.
        scene_usd_path: Scene USD path (for determining default output directory).
        working_dir: Optional analysis working directory for representative USDs.

    Returns:
        Path to the extracted representative file, or None if the payload
        does not use internal instancing.
    """
    from pxr import Usd, UsdGeom

    try:
        stage = Usd.Stage.Open(payload_file)
    except Exception:
        logger.warning(
            f"Failed to open payload for prototype extraction: {payload_file}"
        )
        return None

    if not stage:
        return None

    prototypes = stage.GetPrototypes()
    if not prototypes:
        return None

    # Find the non-instance prototype source prims.
    # For each parent that has a mix of instance and non-instance children,
    # the non-instance child is the prototype source.
    source_paths: list[str] = []

    def _find_sources(prim: Usd.Prim) -> None:
        children = prim.GetChildren()
        if not children:
            return

        has_instance = any(c.IsInstance() for c in children)
        if has_instance:
            for c in children:
                if not c.IsInstance():
                    # Check it has geometry
                    mesh_count = sum(1 for p in Usd.PrimRange(c) if p.IsA(UsdGeom.Mesh))
                    if mesh_count > 0:
                        source_paths.append(str(c.GetPath()))
        # Recurse into non-instance children to find nested groups
        for c in children:
            if not c.IsInstance():
                _find_sources(c)

    for root_prim in stage.GetPseudoRoot().GetChildren():
        _find_sources(root_prim)

    if not source_paths:
        return None

    logger.debug(
        f"  Found {len(source_paths)} prototype source prims: "
        f"{[p.rsplit('/', 1)[-1] for p in source_paths]}"
    )

    # Create a population-masked stage with only the source prims
    root_layer = stage.GetRootLayer()
    mask = Usd.StagePopulationMask(source_paths)
    masked_stage = Usd.Stage.OpenMasked(root_layer, mask, Usd.Stage.LoadAll)
    if not masked_stage:
        logger.warning("Failed to open masked stage for prototype extraction")
        return None

    # Flatten and export
    analysis_dir = (
        working_dir
        if working_dir is not None
        else scene_usd_path.resolve().parent / f".{scene_usd_path.stem}_working"
    )
    rep_dir = analysis_dir / "representatives"
    rep_dir.mkdir(parents=True, exist_ok=True)

    payload_stem = Path(payload_file).stem
    out_file = rep_dir / f"{payload_stem}_representative.usd"

    flat_layer = masked_stage.Flatten()
    flat_layer.Export(str(out_file))
    logger.debug(f"  Exported representative to {out_file}")

    return out_file
