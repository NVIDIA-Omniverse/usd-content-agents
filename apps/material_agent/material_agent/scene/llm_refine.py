# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-based refinement for scene object detection.

After the deterministic analysis detects objects, this module asks an LLM
whether large composite objects should be split into their children.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert at understanding 3D scene hierarchies. You will be given \
a detected object from a USD scene and its direct children with mesh counts. \
Decide whether this object is a single coherent asset or a container grouping \
multiple distinct assets that should be processed independently.

Rules:
- SPLIT if children are functionally distinct objects (e.g. a conveyor, a \
scanner, and a laser grouped under a "system" node).
- SPLIT if the total mesh count is very large (roughly >500 meshes) AND \
children represent distinct modular sections that could be processed \
independently. Large assets built from repeated modular sections (e.g. \
ceiling tiles, wall panels, conveyor segments) should be split even if \
children are structurally related — the sheer size makes monolithic \
processing impractical.
- KEEP if children are parts of one coherent object (e.g. wheels, chassis, \
and body of a single vehicle).
- KEEP if the total mesh count is manageable (<500) and children are \
tightly coupled parts of one thing.
- When in doubt and the asset is small, KEEP. When in doubt and the asset \
is large (>500 meshes), SPLIT.

Return ONLY a JSON object:
{"action": "split" | "keep", "reason": "brief explanation"}\
"""

_USER_PROMPT_TEMPLATE = """\
Detected object: {name}
Prim path: {prim_path}
Total meshes: {mesh_count}
Total vertices: {vertex_count}

Direct children with geometry:
{children_list}\
"""


def _build_children_list(stage: Any, prim_path: str) -> list[dict[str, Any]]:
    """Collect direct children with mesh stats for a prim."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return []

    children = []
    for child in prim.GetChildren():
        mesh_count = sum(1 for p in Usd.PrimRange(child) if p.IsA(UsdGeom.Mesh))
        if mesh_count == 0:
            continue
        vertex_count = 0
        for p in Usd.PrimRange(child):
            if p.IsA(UsdGeom.Mesh):
                pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
                if pts:
                    vertex_count += len(pts)
        children.append(
            {
                "name": child.GetName(),
                "path": str(child.GetPath()),
                "mesh_count": mesh_count,
                "vertex_count": vertex_count,
            }
        )
    return children


def _format_children_list(children: list[dict[str, Any]]) -> str:
    """Format children for the LLM prompt."""
    lines = []
    for c in children:
        lines.append(
            f"  - {c['name']}: {c['mesh_count']} meshes, {c['vertex_count']:,} vertices"
        )
    return "\n".join(lines)


def _build_split_context(
    parent_obj: dict[str, Any],
    child_name: str,
    sibling_names: list[str],
) -> dict[str, Any]:
    """Build split context for a child created by splitting a parent.

    Captures the parent name and sibling names so downstream VLM prediction
    can understand the broader context of where this asset fits.

    If the parent itself has split_context (nested split), the ancestor
    chain is extended rather than replaced.
    """
    parent_split = parent_obj.get("split_context")
    ancestors: list[str] = []
    if parent_split:
        ancestors = list(parent_split.get("ancestors", []))
    ancestors.append(parent_obj["name"])

    return {
        "parent_name": parent_obj["name"],
        "sibling_names": sibling_names,
        "ancestors": ancestors,
    }


def refine_objects_with_llm(
    stage: Any,
    objects: list[dict[str, Any]],
    instance_groups: list[dict[str, Any]],
    llm_config: dict[str, Any],
    min_children_for_review: int = 2,
    max_split_depth: int = 5,
    min_mesh_for_review: int = 100,
    max_workers: int = 16,
    auto_split_threshold: int = 20,
    token_tracker: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Refine detected objects by recursively asking an LLM whether to split.

    Uses level-by-level BFS so all LLM calls within a depth level are fired
    in parallel via ``ThreadPoolExecutor``.  Auto-split and auto-descend cases
    are resolved without LLM calls.  Objects below ``min_mesh_for_review``
    meshes are auto-kept.

    Args:
        stage: The open USD stage.
        objects: Detected objects from ``detect_objects()``.
        instance_groups: Instance groups from ``detect_objects()``.
        llm_config: Dict with keys ``backend``, ``model``, and optionally
            ``temperature``, ``max_tokens``, ``api_key``, ``max_workers``.
        min_children_for_review: Minimum number of geometry children to
            trigger LLM review (default 2).
        max_split_depth: Maximum recursion depth for splitting (default 5).
        min_mesh_for_review: Skip LLM review for objects with fewer meshes
            than this threshold (default 100).
        max_workers: Maximum parallel LLM calls per depth level (default 16).
        auto_split_threshold: Objects with this many or more non-leaf children
            are auto-split without LLM review (default 20).
        token_tracker: Optional TokenTracker for LLM refinement usage.

    Returns:
        Tuple of ``(refined_objects, instance_groups)`` — objects may have
        new entries where parents were split.
    """
    from concurrent.futures import ThreadPoolExecutor

    from langchain_core.messages import HumanMessage, SystemMessage
    from world_understanding.functions.models.chat_models import (
        create_chat_model_from_config,
    )
    from world_understanding.utils.llm_parsing import extract_json_from_llm_response
    from world_understanding.utils.usd.prim import get_subtree_geometry_stats

    max_workers = llm_config.get("max_workers", max_workers)

    llm = create_chat_model_from_config(
        llm_config,
        defaults={"model": "gcp/google/gemini-3.1-pro-preview", "max_tokens": 256},
    )
    if llm is None:
        logger.warning("No API key for LLM refinement — skipping")
        return objects, instance_groups

    # Track highest object ID for generating new IDs
    max_id = 0
    for obj in objects:
        obj_id = obj.get("id", "")
        if obj_id.startswith("obj_"):
            try:
                max_id = max(max_id, int(obj_id[4:]))
            except ValueError:
                pass

    _AUTO_SPLIT_CHILDREN_THRESHOLD = auto_split_threshold

    final_objects: list[dict[str, Any]] = []
    split_paths: set[str] = set()
    llm_calls = 0
    auto_descends = 0
    splits = 0

    # pending: list of (obj_dict, depth) to process — BFS level-by-level
    pending: list[tuple[dict[str, Any], int]] = []
    for obj in objects:
        children_check = _build_children_list(stage, obj["path"])
        if children_check:
            pending.append((obj, 0))
        else:
            final_objects.append(obj)

    if not pending:
        logger.info("LLM refinement: no candidates to review")
        return objects, instance_groups

    logger.info(
        f"LLM refinement: {len(pending)} initial candidates "
        f"(max_depth={max_split_depth}, min_mesh={min_mesh_for_review}, "
        f"max_workers={max_workers})"
    )

    def _invoke_llm(prompt: str) -> Any | None:
        """Call LLM for one item; returns response or None on error."""
        try:
            response = llm.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
        except Exception:
            logger.exception("LLM call failed")
            return None

        try:
            from .stats import record_model_response_usage

            record_model_response_usage(
                token_tracker,
                response,
                llm_config.get("model") or None,
                "scene_analyze_llm",
            )
        except Exception:
            logger.warning("Failed to record LLM usage", exc_info=True)
        return response

    def _make_child_obj(
        obj: dict[str, Any],
        child: dict[str, Any],
        child_id: int,
        sibling_names: list[str],
        inherit_split_context: bool = False,
    ) -> dict[str, Any]:
        child_stats = get_subtree_geometry_stats(
            stage, child["path"], skip_geometry=False
        )
        return {
            "id": f"obj_{child_id:03d}",
            "name": child["name"],
            "path": child["path"],
            "parent_group": obj["name"]
            if not inherit_split_context
            else (obj.get("parent_group") or obj["name"]),
            "source_classification": None,
            "source_files": [],
            "mesh_count": child_stats["mesh_count"],
            "vertex_count": child_stats["vertex_count"],
            "face_count": child_stats["face_count"],
            "prim_type_breakdown": child_stats["prim_type_breakdown"],
            "bounding_box": None,
            "instance_group": None,
            "llm_classification": None,
            "llm_description": None,
            "split_context": (
                obj.get("split_context")
                if inherit_split_context
                else _build_split_context(obj, child["name"], sibling_names)
            ),
        }

    while pending:
        # Collect all items at the current depth level (BFS ordering guarantees
        # items are grouped by depth since children are appended to the end).
        current_depth = pending[0][1]
        current_batch: list[tuple[dict[str, Any], int]] = []
        next_pending: list[tuple[dict[str, Any], int]] = []
        for item in pending:
            (current_batch if item[1] == current_depth else next_pending).append(item)
        pending = next_pending
        depth_prefix = "  " * current_depth

        # --- Phase 1: USD reads (sequential) — classify each item ---
        # Each prepared item gets an "action" key and optionally a "prompt".
        prepared: list[dict[str, Any]] = []
        for obj, depth in current_batch:
            children = _build_children_list(stage, obj["path"])
            item: dict[str, Any] = {"obj": obj, "depth": depth, "children": children}

            if len(children) == 0:
                item["action"] = "keep"

            elif obj.get("mesh_count", 0) < min_mesh_for_review:
                item["action"] = "keep"
                logger.debug(
                    f"{depth_prefix}[d{depth}] {obj['name']}: KEEP "
                    f"({obj.get('mesh_count', 0)} meshes < {min_mesh_for_review})"
                )

            else:
                leaf_children = sum(1 for c in children if c["mesh_count"] <= 1)
                leaf_ratio = leaf_children / len(children)
                # Check if any child is large enough to warrant splitting
                # even when the overall leaf ratio is high.  E.g. a compute
                # tray with 2 large sub-assemblies + 110 screws should still
                # be split because the sub-assemblies are independently
                # complex, unlike a flat mesh-bag where every child is a
                # single mesh (like a humanoid body or a machine housing).
                has_large_child = any(
                    c["mesh_count"] >= min_mesh_for_review for c in children
                )

                if (
                    len(children) >= _AUTO_SPLIT_CHILDREN_THRESHOLD
                    and leaf_ratio >= 0.5
                    and not has_large_child
                ):
                    item["action"] = "keep"
                    logger.info(
                        f"{depth_prefix}[d{depth}] {obj['name']}: KEEP "
                        f"({len(children)} children, {leaf_ratio:.0%} are single-mesh "
                        f"leaves — coherent object, not a container)"
                    )

                elif (
                    len(children) >= _AUTO_SPLIT_CHILDREN_THRESHOLD
                    and leaf_ratio >= 0.5
                    and has_large_child
                ):
                    item["action"] = "auto_split"
                    large_names = [
                        c["name"]
                        for c in children
                        if c["mesh_count"] >= min_mesh_for_review
                    ]
                    logger.info(
                        f"{depth_prefix}[d{depth}] {obj['name']}: AUTO-SPLIT "
                        f"(high leaf ratio but {len(large_names)} large "
                        f"child(ren): {', '.join(large_names[:5])})"
                    )

                elif len(children) >= _AUTO_SPLIT_CHILDREN_THRESHOLD:
                    item["action"] = "auto_split"
                    logger.info(
                        f"{depth_prefix}[d{depth}] {obj['name']}: AUTO-SPLIT "
                        f"({len(children)} children >= {_AUTO_SPLIT_CHILDREN_THRESHOLD})"
                    )

                elif depth >= max_split_depth:
                    item["action"] = "keep"
                    logger.debug(
                        f"{depth_prefix}[d{depth}] {obj['name']}: KEEP (depth limit)"
                    )

                elif len(children) == 1:
                    item["action"] = "auto_descend"
                    logger.info(
                        f"{depth_prefix}[d{depth}] {obj['name']}: AUTO-DESCEND "
                        f"(1 geo child: {children[0]['name']}, "
                        f"{children[0]['mesh_count']} meshes)"
                    )

                else:
                    # 2+ children: needs LLM decision
                    item["action"] = "llm"
                    item["prompt"] = _USER_PROMPT_TEMPLATE.format(
                        name=obj["name"],
                        prim_path=obj["path"],
                        mesh_count=obj["mesh_count"],
                        vertex_count=obj["vertex_count"],
                        children_list=_format_children_list(children),
                    )

            prepared.append(item)

        # --- Phase 2: LLM calls in parallel ---
        llm_items = [p for p in prepared if p["action"] == "llm"]
        if llm_items:
            n_workers = min(max_workers, len(llm_items))
            logger.info(
                f"[d{current_depth}] Firing {len(llm_items)} LLM calls "
                f"in parallel (workers={n_workers})"
            )
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                responses = list(
                    executor.map(_invoke_llm, [p["prompt"] for p in llm_items])
                )
            for p, resp in zip(llm_items, responses, strict=False):
                p["llm_response"] = resp
            llm_calls += len(llm_items)

        # --- Phase 3: Process results, assign IDs, queue children ---
        for p in prepared:
            action = p["action"]
            obj = p["obj"]
            depth = p["depth"]
            children = p["children"]

            if action == "keep":
                final_objects.append(obj)

            elif action == "auto_split":
                splits += 1
                split_paths.add(obj["path"])
                sibling_names = [c["name"] for c in children]
                for child in children:
                    max_id += 1
                    child_obj = _make_child_obj(obj, child, max_id, sibling_names)
                    if child_obj["mesh_count"] < min_mesh_for_review:
                        final_objects.append(child_obj)
                    else:
                        pending.append((child_obj, depth + 1))

            elif action == "auto_descend":
                auto_descends += 1
                split_paths.add(obj["path"])
                child = children[0]
                max_id += 1
                child_obj = _make_child_obj(
                    obj, child, max_id, [], inherit_split_context=True
                )
                pending.append((child_obj, depth + 1))

            elif action == "llm":
                resp = p.get("llm_response")
                if resp is None:
                    final_objects.append(obj)
                    continue

                result = extract_json_from_llm_response(
                    resp.content, expected_keys=["action"]
                )
                if not result:
                    logger.warning(
                        f"Failed to parse LLM response for {obj['name']}, keeping"
                    )
                    final_objects.append(obj)
                    continue

                llm_action = result.get("action", "keep").lower()
                reason = result.get("reason", "")
                logger.info(
                    f"{depth_prefix}[d{depth}] {obj['name']}: {llm_action.upper()} "
                    f"({len(children)} children, {obj.get('mesh_count', 0)} meshes) "
                    f"— {reason}"
                )

                if llm_action == "split":
                    splits += 1
                    split_paths.add(obj["path"])
                    sibling_names = [c["name"] for c in children]
                    for child in children:
                        max_id += 1
                        child_obj = _make_child_obj(obj, child, max_id, sibling_names)
                        pending.append((child_obj, depth + 1))
                else:
                    final_objects.append(obj)

    # Build refined list: original non-split objects + final objects from queue.
    # Deduplicate by path.
    original_kept = [obj for obj in objects if obj["path"] not in split_paths]
    seen_paths: set[str] = set()
    refined: list[dict[str, Any]] = []
    for obj in original_kept:
        if obj["path"] not in seen_paths:
            seen_paths.add(obj["path"])
            refined.append(obj)
    for obj in final_objects:
        if obj["path"] not in seen_paths:
            seen_paths.add(obj["path"])
            refined.append(obj)

    logger.info(
        f"LLM refinement complete: {splits} split (incl. auto), "
        f"{auto_descends} auto-descended, {llm_calls} LLM calls "
        f"({len(objects)} → {len(refined)} objects)"
    )
    return refined, instance_groups
