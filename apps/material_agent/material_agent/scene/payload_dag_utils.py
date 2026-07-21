# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable utilities for building a payload dependency DAG from USD files.

Provides:
- ``collect_arcs_from_layer()`` — collect payload/reference arc targets from a layer
- ``collect_arcs_from_file()`` — same but opens a file, including its sublayers
- ``build_dag()`` — BFS from a set of root files to build the full adjacency list
- ``compute_depths()`` — depth(leaf)=0, depth(parent)=1+max(child depth)
- ``topological_sort_leaves_first()`` — Kahn's algorithm, leaves first
- ``rewrite_arcs_in_layer()`` — rewrite payload/reference arcs in a layer

Used by:
- ``material_agent/scene/analyze.py`` — payload DAG construction in the pipeline
- ``material_agent/scene/run.py`` — creating modified parent copies
- ``material_agent/scene/collect.py`` — rewriting scene sublayer arcs
- ``analyze_payload_dag.py`` — standalone DAG analysis script
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from pxr import Sdf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arc collection from Sdf layers
# ---------------------------------------------------------------------------


def collect_arcs_from_layer(layer: Sdf.Layer) -> set[str]:
    """Collect all payload and reference asset-path targets from a layer.

    Walks every prim spec recursively. Returns resolved, absolute file paths
    for targets that exist on disk. Ignores empty asset paths (internal refs).
    """
    results: set[str] = set()

    def _walk(spec: Sdf.PrimSpec) -> None:
        # Payloads
        if spec.payloadList:
            for item_list in (
                spec.payloadList.prependedItems,
                spec.payloadList.appendedItems,
                spec.payloadList.explicitItems,
            ):
                for item in item_list:
                    if item.assetPath:
                        resolved = layer.ComputeAbsolutePath(item.assetPath)
                        if resolved and os.path.isfile(resolved):
                            results.add(str(Path(resolved).resolve()))

        # References
        if spec.referenceList:
            for item_list in (
                spec.referenceList.prependedItems,
                spec.referenceList.appendedItems,
                spec.referenceList.explicitItems,
            ):
                for item in item_list:
                    if item.assetPath:
                        resolved = layer.ComputeAbsolutePath(item.assetPath)
                        if resolved and os.path.isfile(resolved):
                            results.add(str(Path(resolved).resolve()))

        for child in spec.nameChildren:
            _walk(child)

    for root_prim in layer.rootPrims:
        _walk(root_prim)

    return results


def collect_arcs_from_file(file_path: str) -> set[str]:
    """Open a USD file as Sdf.Layer, collect arc targets including sublayers.

    Also recurses into the file's own sublayers (but NOT into arc targets —
    that is handled by the BFS in ``build_dag``).
    """
    from pxr import Sdf

    all_children: set[str] = set()

    if not os.path.isfile(file_path):
        return all_children

    try:
        layer = Sdf.Layer.FindOrOpen(file_path)
    except Exception:
        return all_children

    if not layer:
        return all_children

    # Direct arcs in this layer
    all_children.update(collect_arcs_from_layer(layer))

    # Sublayers (part of the same layer stack, not separate dependencies,
    # but they may contain payload arcs that we need to discover)
    for sl_path in layer.subLayerPaths:
        resolved = layer.ComputeAbsolutePath(sl_path)
        if resolved and os.path.isfile(resolved):
            resolved = str(Path(resolved).resolve())
            try:
                sl_layer = Sdf.Layer.FindOrOpen(resolved)
            except Exception:
                continue
            if sl_layer:
                all_children.update(collect_arcs_from_layer(sl_layer))

    return all_children


# ---------------------------------------------------------------------------
# DAG construction via BFS
# ---------------------------------------------------------------------------


def build_dag(root_files: set[str]) -> dict[str, set[str]]:
    """BFS from root files, discovering nested children at each level.

    Returns adjacency list: parent -> {children}.
    Every discovered node appears as a key (leaves map to empty sets).
    """
    adj: dict[str, set[str]] = {}
    queue: deque[str] = deque(sorted(root_files))
    visited: set[str] = set(root_files)

    while queue:
        current = queue.popleft()
        children = collect_arcs_from_file(current)
        children.discard(current)  # no self-loops
        adj[current] = children

        for child in children:
            if child not in visited:
                visited.add(child)
                queue.append(child)

    # Ensure every child that appeared also has a key
    all_nodes: set[str] = set(adj.keys())
    for children in list(adj.values()):
        all_nodes.update(children)
    for node in all_nodes:
        adj.setdefault(node, set())

    return adj


# ---------------------------------------------------------------------------
# Depth computation
# ---------------------------------------------------------------------------


def compute_depths(adj: dict[str, set[str]]) -> dict[str, int]:
    """Compute depth for each node: leaf=0, parent=1+max(child depth)."""
    cache: dict[str, int] = {}

    def _depth(node: str, stack: set[str]) -> int:
        if node in cache:
            return cache[node]
        if node in stack:
            return 0  # cycle guard
        stack.add(node)
        children = adj.get(node, set())
        if not children:
            cache[node] = 0
        else:
            cache[node] = 1 + max(_depth(c, stack) for c in children)
        stack.discard(node)
        return cache[node]

    for node in adj:
        _depth(node, set())
    return cache


# ---------------------------------------------------------------------------
# Topological sort (leaves first)
# ---------------------------------------------------------------------------


def topological_sort_leaves_first(adj: dict[str, set[str]]) -> list[str]:
    """Kahn's algorithm emitting leaves first (nodes with out-degree 0 first).

    This is the correct processing order: leaf payloads with no
    sub-dependencies first, then parents of leaves, etc.
    """
    out_degree: dict[str, int] = {n: len(ch) for n, ch in adj.items()}
    reverse_adj: dict[str, set[str]] = defaultdict(set)
    for parent, children in adj.items():
        for child in children:
            reverse_adj[child].add(parent)

    queue: deque[str] = deque(sorted(n for n, d in out_degree.items() if d == 0))
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for parent in sorted(reverse_adj.get(node, set())):
            out_degree[parent] -= 1
            if out_degree[parent] == 0:
                queue.append(parent)

    if len(order) != len(adj):
        remaining = set(adj.keys()) - set(order)
        logger.warning(f"Cycle detected involving {len(remaining)} payload nodes")
        order.extend(sorted(remaining))

    return order


# ---------------------------------------------------------------------------
# Arc rewriting
# ---------------------------------------------------------------------------


def rewrite_arcs_in_layer(
    layer: Sdf.Layer,
    child_map: dict[str, str],
    resolve_from: str | Path | None = None,
) -> int:
    """Rewrite payload/reference arc asset paths in a layer.

    Walks all prim specs. For payload and reference arcs whose resolved
    paths match entries in ``child_map``, rewrites them to point to the
    updated file. Computes relative paths from the layer's location.

    Args:
        layer: The Sdf.Layer to modify (in-place).
        child_map: Dict mapping original absolute file path -> updated absolute path.
        resolve_from: Optional path (file or directory) to resolve relative
            asset paths from.  Use this when the layer has been copied to a
            different directory and its relative arcs would otherwise resolve
            incorrectly.  If a file path is given, its parent directory is
            used.  If None, arcs are resolved from the layer's own location.

    Returns:
        Number of arcs rewritten.
    """
    from pxr import Sdf

    layer_dir = Path(layer.realPath).resolve().parent

    # For resolving arcs, use the original location if provided
    if resolve_from is not None:
        resolve_dir = Path(resolve_from).resolve()
        if resolve_dir.is_file():
            resolve_dir = resolve_dir.parent
        # Open the original layer (read-only) just for arc resolution
        # If we can't, fall back to the layer itself
        resolve_layer = Sdf.Layer.FindOrOpen(str(resolve_from))
        if not resolve_layer:
            resolve_layer = layer
    else:
        resolve_layer = layer

    rewritten = 0

    # When resolve_from is set, the layer has been copied to a different
    # dir.  ALL relative arcs need fixing, not just the ones in child_map.
    fix_all_paths = resolve_from is not None and resolve_layer is not layer

    def _rewrite_list(spec_list: Any, make_item):
        nonlocal rewritten
        for list_attr in ("prependedItems", "appendedItems", "explicitItems"):
            items = list(getattr(spec_list, list_attr, []))
            new_items = []
            changed = False
            for item in items:
                if item.assetPath:
                    # Resolve from the original location, not the copy's
                    resolved = resolve_layer.ComputeAbsolutePath(item.assetPath)
                    resolved = str(Path(resolved).resolve()) if resolved else ""
                    if resolved in child_map:
                        # Arc matches update map — rewrite to updated target
                        new_rel = os.path.relpath(
                            child_map[resolved], layer_dir
                        ).replace("\\", "/")
                        new_items.append(make_item(new_rel, item))
                        changed = True
                        rewritten += 1
                        continue
                    if fix_all_paths and resolved and os.path.isfile(resolved):
                        # Arc doesn't match update map but the layer moved,
                        # so rewrite it to resolve correctly from new location
                        new_rel = os.path.relpath(resolved, layer_dir).replace(
                            "\\", "/"
                        )
                        if new_rel != item.assetPath:
                            new_items.append(make_item(new_rel, item))
                            changed = True
                            continue
                new_items.append(item)
            if changed:
                setattr(spec_list, list_attr, new_items)

    def _walk(spec: Sdf.PrimSpec) -> None:
        if spec.payloadList:
            _rewrite_list(
                spec.payloadList,
                lambda path, old: Sdf.Payload(path, old.primPath, old.layerOffset),
            )
        if spec.referenceList:
            _rewrite_list(
                spec.referenceList,
                lambda path, old: Sdf.Reference(
                    path, old.primPath, old.layerOffset, old.customData
                ),
            )
        for child in spec.nameChildren:
            _walk(child)

    for root_prim in layer.rootPrims:
        _walk(root_prim)

    return rewritten
