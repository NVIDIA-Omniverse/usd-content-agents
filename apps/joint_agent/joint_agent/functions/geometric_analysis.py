# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Geometric contact graph analysis for articulated bodies.

Fallback strategy when the USD hierarchy is flat or uninformative. Builds a
contact graph from actual mesh vertex proximity, finds the kinematic chain
backbone, and orders prims along it for VLM-assisted segment assignment.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CONTACT_GRAPH_THRESHOLD_M = 0.002
MAX_VERTICES_SAMPLED_PER_MESH = 500
GEOMETRIC_RNG_SEED = 42

GEOMETRIC_HEURISTIC_PATHS = [
    "contact_graph_threshold",
    "graph_diameter_backbone",
    "lowest_z_orientation",
]


def load_mesh_vertices(usd_path: str, mesh_paths: list[str]) -> dict[str, dict]:
    """Load world-space vertex data for specified mesh prims.

    Args:
        usd_path: Path to the USD file.
        mesh_paths: List of prim paths to load.

    Returns:
        Dict mapping prim_path -> {"points", "center", "vol", "size"}
    """
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    xform_cache = UsdGeom.XformCache()
    mesh_data: dict[str, dict] = {}

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        path = str(prim.GetPath())
        if path not in mesh_paths:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        wx = xform_cache.GetLocalToWorldTransform(prim)
        wpts = np.array([list(wx.Transform(Gf.Vec3d(pt))) for pt in points])
        size = wpts.max(axis=0) - wpts.min(axis=0)
        mesh_data[path] = {
            "points": wpts,
            "center": wpts.mean(axis=0),
            "vol": float(np.prod(size)),
            "size": size,
        }

    return mesh_data


def build_contact_graph(
    mesh_data: dict[str, dict],
    threshold: float = CONTACT_GRAPH_THRESHOLD_M,
    max_sample: int = MAX_VERTICES_SAMPLED_PER_MESH,
    rng_seed: int = GEOMETRIC_RNG_SEED,
) -> dict[str, set[str]]:
    """Build a contact graph from mesh vertex proximity.

    Two meshes are "in contact" if their closest vertices are within threshold.
    Uses a single combined KD-tree for O(N) queries instead of O(N^2) pairwise.

    Args:
        mesh_data: Per-mesh data with "points" arrays.
        threshold: Contact distance threshold in meters.
        max_sample: Max vertices to sample per mesh for KD-tree.
        rng_seed: Random seed for deterministic vertex subsampling.

    Returns:
        Adjacency dict mapping prim_path -> set of neighbor prim_paths.
    """
    from scipy.spatial import cKDTree

    rng = np.random.RandomState(rng_seed)
    paths = sorted(mesh_data.keys())

    # Subsample and collect all vertices into a single array, tracking
    # which mesh each vertex belongs to via a parallel label array.
    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    mesh_idx = {p: i for i, p in enumerate(paths)}

    for p in paths:
        pts = mesh_data[p]["points"]
        if len(pts) > max_sample:
            idx = rng.choice(len(pts), max_sample, replace=False)
            pts = pts[idx]
        all_points.append(pts)
        all_labels.append(np.full(len(pts), mesh_idx[p], dtype=np.int32))

    combined_points = np.vstack(all_points)
    combined_labels = np.concatenate(all_labels)

    # Build one KD-tree over all vertices
    tree = cKDTree(combined_points)

    # For each vertex, find all neighbors within threshold. If any neighbor
    # belongs to a different mesh, those two meshes are in contact.
    adjacency: dict[str, set[str]] = defaultdict(set)
    pairs = tree.query_pairs(r=threshold)
    for i, j in pairs:
        li, lj = combined_labels[i], combined_labels[j]
        if li != lj:
            pa, pb = paths[li], paths[lj]
            adjacency[pa].add(pb)
            adjacency[pb].add(pa)

    return dict(adjacency)


def find_chain_backbone(
    adjacency: dict[str, set[str]],
    paths: list[str],
) -> list[str]:
    """Find the longest path in the contact graph (tree diameter).

    This is the kinematic chain backbone for a serial chain robot.

    Args:
        adjacency: Contact graph adjacency dict.
        paths: List of all prim paths.

    Returns:
        Ordered list of prim paths forming the backbone.
    """

    def _bfs_farthest(start: str) -> str:
        visited = {start: 0}
        queue = [start]
        farthest = start
        max_d = 0
        while queue:
            node = queue.pop(0)
            for nb in adjacency.get(node, []):
                if nb not in visited:
                    visited[nb] = visited[node] + 1
                    queue.append(nb)
                    if visited[nb] > max_d:
                        max_d = visited[nb]
                        farthest = nb
        return farthest

    def _find_path(start: str, end: str) -> list[str]:
        visited: dict[str, str | None] = {start: None}
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node == end:
                break
            for nb in adjacency.get(node, []):
                if nb not in visited:
                    visited[nb] = node
                    queue.append(nb)
        path_list: list[str] = []
        n: str | None = end
        while n is not None:
            path_list.append(n)
            n = visited.get(n)
        return list(reversed(path_list))

    end1 = _bfs_farthest(paths[0])
    end2 = _bfs_farthest(end1)
    return _find_path(end1, end2)


def order_prims_along_chain(
    paths: list[str],
    adjacency: dict[str, set[str]],
    backbone: list[str],
    mesh_data: dict[str, dict],
) -> list[str]:
    """Order all prims along the kinematic chain backbone.

    Side-branch prims are inserted at their backbone parent's position.

    Args:
        paths: All prim paths.
        adjacency: Contact graph.
        backbone: Backbone path.
        mesh_data: Per-mesh geometric data.

    Returns:
        Ordered list of all prim paths from base to end-effector.
    """
    bb_set = set(backbone)
    bb_idx = {p: i for i, p in enumerate(backbone)}

    # Map side branches to their backbone neighbor
    for p in paths:
        if p not in bb_idx:
            for nb in adjacency.get(p, []):
                if nb in bb_set:
                    bb_idx[p] = bb_idx[nb]
                    break

    ordered = sorted(paths, key=lambda p: bb_idx.get(p, 0))

    # Orient: base should be first (lowest Z typically)
    if (
        len(backbone) >= 2
        and mesh_data[backbone[0]]["center"][2] > mesh_data[backbone[-1]]["center"][2]
    ):
        ordered = list(reversed(ordered))

    return ordered


def build_chain_table(
    ordered_paths: list[str],
    mesh_data: dict[str, dict],
    adjacency: dict[str, set[str]],
    bb_idx: dict[str, int],
) -> str:
    """Build a markdown table of prims ordered along the chain.

    Args:
        ordered_paths: Prims ordered from base to end-effector.
        mesh_data: Per-mesh geometric data.
        adjacency: Contact graph.
        bb_idx: Backbone index for each prim.

    Returns:
        Formatted markdown table string.
    """
    lines = [
        "| # | Prim | Volume | Aspect | Chain Pos | Neighbors |",
        "|---|------|--------|--------|-----------|-----------|",
    ]
    for i, p in enumerate(ordered_paths):
        short = p.split("/")[-1]
        vol = mesh_data[p]["vol"]
        size = mesh_data[p]["size"]
        aspect = float(max(size) / max(min(size), 1e-6))
        pos = bb_idx.get(p, -1)
        neighbors = [n.split("/")[-1] for n in adjacency.get(p, [])]
        lines.append(
            f"| {i + 1} | {short} | {vol:.6f} | {aspect:.1f} | "
            f"{pos} | {', '.join(sorted(neighbors))} |"
        )
    return "\n".join(lines)


def analyze_geometry(
    usd_path: str,
    mesh_paths: list[str],
    segment_names: list[str],
    vlm_generate_fn: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Analyze articulated body structure from mesh geometry.

    Builds a contact graph, finds the chain backbone, orders prims, then
    sends the ordered table to a VLM for segment assignment.

    Args:
        usd_path: Path to the USD file.
        mesh_paths: List of mesh prim paths.
        segment_names: Ordered segment names.
        vlm_generate_fn: Callable(system_prompt, user_prompt) -> str.

    Returns:
        (assignments, metadata)
    """
    import re

    logger.info("Loading mesh vertices...")
    mesh_data = load_mesh_vertices(usd_path, mesh_paths)
    if not mesh_data:
        return {}, {
            "strategy": "geometric",
            "reason": "no mesh data",
            "prompt_library_used": False,
            "heuristic_paths_used": [],
        }

    paths = sorted(mesh_data.keys())
    logger.info("Building contact graph (%d meshes)...", len(paths))
    adjacency = build_contact_graph(mesh_data)
    num_edges = sum(len(v) for v in adjacency.values()) // 2
    logger.info("Contact graph: %d nodes, %d edges", len(paths), num_edges)

    logger.info("Finding chain backbone...")
    backbone = find_chain_backbone(adjacency, paths)
    logger.info("Backbone: %d nodes", len(backbone))

    # Build backbone index
    bb_set = set(backbone)
    bb_idx = {p: i for i, p in enumerate(backbone)}
    for p in paths:
        if p not in bb_idx:
            for nb in adjacency.get(p, []):
                if nb in bb_set:
                    bb_idx[p] = bb_idx[nb]
                    break

    ordered = order_prims_along_chain(paths, adjacency, backbone, mesh_data)
    table_str = build_chain_table(ordered, mesh_data, adjacency, bb_idx)

    # VLM prompt
    names_str = ", ".join(segment_names)
    user_prompt = f"""You are classifying components of an articulated body into \
{len(segment_names)} segments.
The components are ordered along the kinematic chain (base to end-effector),
derived from a mesh contact graph analysis.

Valid segment names: {names_str}

{table_str}

Assign each component to a segment. Components that touch each other and have
similar volume likely belong to the same segment. High-volume, high-aspect-ratio
components are structural links. Small components adjacent to them are
caps/covers belonging to the same segment.

Return a JSON mapping each prim name to its segment:
<answer>
{{"prim_name": "segment_name", ...}}
</answer>"""

    system_prompt = "You are a robotics expert classifying articulated body components."
    response = vlm_generate_fn(system_prompt, user_prompt)
    logger.info("VLM response received (%d chars)", len(response))

    # Parse
    assignments: dict[str, str] = {}
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    json_str = answer_match.group(1).strip() if answer_match else response
    json_obj = re.search(r"\{[\s\S]*\}", json_str)
    if json_obj:
        try:
            import json

            data = json.loads(json_obj.group())
            short_to_full = {p.split("/")[-1]: p for p in mesh_paths}
            for key, value in data.items():
                full = short_to_full.get(key.strip("/").split("/")[-1])
                if full and isinstance(value, str):
                    assignments[full] = value
        except json.JSONDecodeError:
            logger.warning("Failed to parse geometric VLM response")

    metadata = {
        "strategy": "geometric",
        "backbone_length": len(backbone),
        "num_edges": num_edges,
        "num_assigned": len(assignments),
        "prompt_library_used": False,
        "heuristic_paths_used": list(GEOMETRIC_HEURISTIC_PATHS),
        "heuristic_parameters": {
            "contact_threshold_m": CONTACT_GRAPH_THRESHOLD_M,
            "max_vertices_sampled_per_mesh": MAX_VERTICES_SAMPLED_PER_MESH,
            "rng_seed": GEOMETRIC_RNG_SEED,
        },
    }

    return assignments, metadata
