# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchy-based structural analysis for articulated bodies.

Extracts the USD scene tree with per-mesh geometric stats, sends it to an LLM
to classify each node's structural role (joint, link, cap, root) and hierarchy
pattern (joint_centric vs link_centric), then applies a deterministic boundary
convention to produce segment assignments.

No assumptions are made about prim naming conventions — the LLM reasons from
tree structure and geometry alone.
"""

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Default retry settings for LLM calls
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0
MAX_PHYSICS_CHAIN_CANDIDATES_PER_SEGMENT = 64
MAX_PHYSICS_CHAIN_EDGES = 4096
MAX_PHYSICS_CHAIN_TOTAL_CANDIDATES = 512
MAX_PHYSICS_CHAIN_TRANSITIONS = 65536
PHYSICS_CHAIN_ROLES = {"joint", "link"}


@dataclass(frozen=True)
class _PhysicsJointChainValidation:
    """Validated physics-chain roles and the mesh coverage that proved them."""

    node_roles: dict[str, dict[str, str]]
    mesh_assignments: dict[str, str]
    covered_mesh_paths: frozenset[str]


@dataclass(frozen=True)
class _ParsedAnalysisResponse:
    """Parsed LLM response with both sanitized roles and raw-shape diagnostics."""

    is_informative: bool
    hierarchy_pattern: str
    node_roles: dict[str, dict[str, str]]
    malformed_node_roles: tuple[str, ...]


def _fibonacci_delay(attempt: int, base_delay: float = DEFAULT_BASE_DELAY) -> float:
    """Fibonacci backoff delay (matches predict step convention)."""
    if attempt <= 1:
        return base_delay
    fib = [1, 1]
    for _ in range(2, attempt + 1):
        fib.append(fib[-1] + fib[-2])
    return fib[attempt] * base_delay


def _call_with_retry(
    fn: Callable[..., str],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    label: str = "LLM call",
    **kwargs: Any,
) -> str:
    """Call a function with Fibonacci backoff retry on failure.

    Args:
        fn: Callable to invoke.
        *args: Positional arguments for fn.
        max_retries: Maximum number of attempts.
        label: Label for log messages.
        **kwargs: Keyword arguments for fn.

    Returns:
        The string result from fn, or "" if every attempt returned an
        empty/blank response.  Callers must handle both "" returns
        (transient empty responses) and exceptions (hard failures).

    Raises:
        The last exception if all retries are exhausted due to errors.
    """
    for attempt in range(max_retries):
        try:
            result = fn(*args, **kwargs)
            if result and result.strip():
                if attempt > 0:
                    logger.info(
                        "%s succeeded on attempt %d/%d",
                        label,
                        attempt + 1,
                        max_retries,
                    )
                return result
            # Empty response — retry
            if attempt < max_retries - 1:
                delay = _fibonacci_delay(attempt)
                logger.warning(
                    "Empty %s response on attempt %d/%d, retrying in %.1fs...",
                    label,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("Empty %s response after %d attempts", label, max_retries)
                return ""
        except Exception as e:
            if attempt < max_retries - 1:
                delay = _fibonacci_delay(attempt)
                logger.warning(
                    "%s error on attempt %d/%d: %s. Retrying in %.1fs...",
                    label,
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("%s failed after %d attempts: %s", label, max_retries, e)
                raise
    return ""


# ---------------------------------------------------------------------------
# Scene tree extraction
# ---------------------------------------------------------------------------


def extract_scene_tree(usd_path: str) -> tuple[str, list[str]]:
    """Extract the USD hierarchy as an annotated text tree.

    Each node includes geometric stats (vertex count, bounding box volume,
    aspect ratio) so the LLM can reason about structure from geometry alone.

    Args:
        usd_path: Path to the USD file.

    Returns:
        (tree_text, mesh_paths) where tree_text is the formatted tree string
        and mesh_paths is a list of full prim paths for all Mesh prims.
    """
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    xform_cache = UsdGeom.XformCache()
    lines: list[str] = []
    mesh_paths: list[str] = []

    def _get_mesh_stats(prim: Any) -> dict[str, Any] | None:
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            return None
        world_xform = xform_cache.GetLocalToWorldTransform(prim)
        wpts = np.array([list(world_xform.Transform(Gf.Vec3d(pt))) for pt in pts])
        size = wpts.max(axis=0) - wpts.min(axis=0)
        return {
            "verts": len(pts),
            "vol": float(np.prod(size)),
            "aspect": float(max(size) / max(min(size), 1e-6)),
            "size": [float(s) for s in size],
        }

    def _walk(prim: Any, indent: int = 0) -> None:
        name = prim.GetName()
        prim_type = prim.GetTypeName()

        # Skip render/settings/material prims
        if prim_type in (
            "Material",
            "Shader",
            "RenderSettings",
            "RenderProduct",
            "RenderVar",
        ):
            return
        if name in ("Render", "OmniverseKit", "Looks", "materials") and not prim_type:
            return

        children = list(prim.GetChildren())
        prefix = "  " * indent

        stats_str = ""
        if prim_type == "Mesh":
            stats = _get_mesh_stats(prim)
            if stats:
                stats_str = (
                    f" [verts={stats['verts']}, vol={stats['vol']:.6f}, "
                    f"aspect={stats['aspect']:.1f}, "
                    f"size={stats['size'][0]:.3f}x{stats['size'][1]:.3f}"
                    f"x{stats['size'][2]:.3f}]"
                )
            mesh_paths.append(str(prim.GetPath()))

        child_count = ""
        if children and prim_type != "Mesh":
            child_count = f" [{len(children)} children]"

        lines.append(f"{prefix}{name} ({prim_type}){stats_str}{child_count}")
        for child in children:
            _walk(child, indent + 1)

    root = stage.GetPseudoRoot()
    for child in root.GetChildren():
        child_name = child.GetName()
        child_type = child.GetTypeName()
        if child_name in ("Render", "OmniverseKit") or child_type in (
            "",
            "RenderSettings",
        ):
            continue
        _walk(child)

    return "\n".join(lines), mesh_paths


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are an expert robotics engineer analyzing USD scene hierarchies of
articulated bodies (robot arms, excavators, humanoid limbs, etc.).

Your task: examine the scene tree — including the geometric stats for each
mesh (vertex count, bounding box volume, aspect ratio) — and determine
whether the hierarchy encodes a kinematic chain.

**Do NOT rely on prim names.** Names may be generic (Xform_001, Mesh_003)
or descriptive (shoulder_link, Joint_002). Use the hierarchy STRUCTURE and
GEOMETRY to reason:
- Nesting depth that increases along a chain
- Alternating pattern of small compact nodes (joints) and large elongated
  nodes (links)
- Side branches with small meshes (caps, covers, fasteners)
- Bounding box volume and aspect ratio distinguish joints from links

**Node roles** (classify each Xform/group node):
- root: The top-level container of the articulated body
- joint: A rotational axis / joint housing (typically compact, moderate volume)
- link: A rigid structural body connecting two joints (typically elongated,
  high volume, high aspect ratio)
- cap: A cover, end-cap, or cosmetic panel at a joint (small volume, often
  a side branch)
"""


def build_analysis_prompt(
    tree_text: str,
    segment_names: list[str],
) -> str:
    """Build the prompt asking the LLM to analyze the hierarchy.

    Args:
        tree_text: Formatted scene tree with geometric stats.
        segment_names: Ordered segment names from base to end-effector.

    Returns:
        The user prompt string.
    """
    names_str = ", ".join(segment_names)
    n = len(segment_names)
    return f"""Analyze this USD scene tree (with per-mesh geometric stats):

```
{tree_text}
```

The asset is an articulated body with {n} named segments: {names_str}
(ordered from base to end-effector).

**Instructions:**

1. Is this hierarchy informative (does the nesting encode a kinematic chain)?
   Answer YES or NO.

2. If YES: identify the main kinematic chain nodes in order from base to
   end-effector. For EACH node in the chain, classify its role:
   - "root": top-level container
   - "joint": a revolute/prismatic joint assembly
   - "link": a rigid structural body between joints
   - "cap": a cover or end-cap at a joint

3. Map each chain node to one of the {n} segment names.
   Use the geometric stats to guide your mapping — large elongated meshes
   (high volume, high aspect ratio) are structural links; small meshes at
   branches are caps/covers.

4. For nodes classified as "cap": assign them to the SAME segment as the
   nearest structural link or joint body they are attached to (their parent
   or sibling in the hierarchy).

5. Classify the hierarchy pattern as one of:
   - "joint_centric": nesting follows joints (each level = a joint axis,
     meshes at a joint are the joint housing). E.g., Joint_001/Joint_002/...
   - "link_centric": nesting follows links (each level = a named rigid body,
     the link IS the segment). E.g., base/shoulder_link/upper_arm_link/...
   This determines whether boundary adjustment is needed for joint/cap nodes.

**Response format:**
<analysis>Your structural reasoning (reference geometry, not names)</analysis>
<informative>YES or NO</informative>
<hierarchy_pattern>joint_centric or link_centric</hierarchy_pattern>
<node_roles>
{{
  "node_name": {{"role": "joint|link|cap|root", "segment": "segment_name"}},
  ...
}}
</node_roles>"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _sanitize_node_roles(
    raw_node_roles: Any,
) -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
    """Return well-formed node roles and malformed raw entry names."""
    if not isinstance(raw_node_roles, dict):
        logger.warning("Ignoring malformed node_roles payload from LLM response")
        return {}, ("<node_roles>",)

    node_roles: dict[str, dict[str, str]] = {}
    invalid_nodes: list[str] = []
    for node_name, info in raw_node_roles.items():
        if not isinstance(node_name, str) or not isinstance(info, dict):
            invalid_nodes.append(str(node_name))
            continue
        role = info.get("role")
        segment = info.get("segment")
        if not isinstance(role, str) or not isinstance(segment, str):
            invalid_nodes.append(node_name)
            continue
        node_roles[node_name] = {"role": role, "segment": segment}

    if invalid_nodes:
        logger.warning(
            "Ignoring malformed node_roles entries from LLM response: %s",
            ", ".join(sorted(invalid_nodes)),
        )
    return node_roles, tuple(sorted(invalid_nodes))


def _parse_analysis_response_details(
    response: str,
) -> _ParsedAnalysisResponse:
    """Parse the LLM's analysis response with raw-shape diagnostics.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed response details where node_roles maps
        node_name -> {"role": ..., "segment": ...}.
    """
    # Is informative?
    inform_match = re.search(
        r"<informative>\s*(YES|NO)\s*</informative>", response, re.IGNORECASE
    )
    is_informative = bool(inform_match and inform_match.group(1).upper() == "YES")

    # Hierarchy pattern
    pattern_match = re.search(
        r"<hierarchy_pattern>\s*(joint_centric|link_centric)\s*</hierarchy_pattern>",
        response,
        re.IGNORECASE,
    )
    hierarchy_pattern = "joint_centric"
    if pattern_match:
        hierarchy_pattern = pattern_match.group(1).lower()

    # Node roles
    node_roles: dict[str, dict[str, str]] = {}
    malformed_node_roles: tuple[str, ...] = ()
    roles_match = re.search(r"<node_roles>(.*?)</node_roles>", response, re.DOTALL)
    if roles_match:
        json_match = re.search(r"\{[\s\S]*\}", roles_match.group(1))
        if json_match:
            try:
                node_roles, malformed_node_roles = _sanitize_node_roles(
                    json.loads(json_match.group())
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse node_roles JSON from LLM response")
                malformed_node_roles = ("<node_roles>",)

    return _ParsedAnalysisResponse(
        is_informative=is_informative,
        hierarchy_pattern=hierarchy_pattern,
        node_roles=node_roles,
        malformed_node_roles=malformed_node_roles,
    )


def parse_analysis_response(
    response: str,
) -> tuple[bool, str, dict[str, dict[str, str]]]:
    """Parse the LLM's analysis response."""
    parsed = _parse_analysis_response_details(response)
    return parsed.is_informative, parsed.hierarchy_pattern, parsed.node_roles


# ---------------------------------------------------------------------------
# Segment assignment
# ---------------------------------------------------------------------------


def assign_segments_from_roles(
    mesh_paths: list[str],
    node_roles: dict[str, dict[str, str]],
    segment_names: list[str],
    hierarchy_pattern: str = "joint_centric",
) -> dict[str, str]:
    """Assign segment names to meshes using LLM-classified node roles.

    For joint-centric hierarchies, applies a boundary convention: from the 3rd
    segment onward, joints and caps belong to the preceding segment. For
    link-centric hierarchies, uses the LLM's direct mapping.

    Args:
        mesh_paths: List of full prim paths for all meshes.
        node_roles: LLM's classification of each node.
        segment_names: Ordered segment names.
        hierarchy_pattern: "joint_centric" or "link_centric".

    Returns:
        Dict mapping each mesh prim path to its segment name.
    """
    seg_to_idx = {s: i for i, s in enumerate(segment_names)}

    # Build node_name -> (segment, role) mapping
    node_to_seg: dict[str, str] = {}
    node_to_role: dict[str, str] = {}
    for node_name, info in node_roles.items():
        seg = info.get("segment", "")
        role = info.get("role", "")
        if seg and seg in segment_names:
            node_to_seg[node_name] = seg
            node_to_role[node_name] = role

    # Apply boundary convention for joint-centric hierarchies only.
    # In joint-centric nesting, components AT a joint belong to the link that
    # ENDS at that joint (the preceding segment). This applies from the 3rd
    # segment onward (the elbow region in a robot arm).
    if hierarchy_pattern == "joint_centric":
        logger.info("Applying boundary convention (joint-centric hierarchy)")
        for node_name, role in node_to_role.items():
            if role in ("cap", "joint"):
                seg = node_to_seg[node_name]
                idx = seg_to_idx.get(seg, 0)
                if idx >= 3:
                    node_to_seg[node_name] = segment_names[idx - 1]
    else:
        logger.info("No boundary convention needed (link-centric hierarchy)")

    logger.info("Node-to-segment mapping:")
    for node, seg in node_to_seg.items():
        role = node_to_role.get(node, "?")
        logger.info("  %s (%s) -> %s", node, role, seg)

    leaf_counts = Counter(path.strip("/").split("/")[-1] for path in mesh_paths)
    ambiguous_leaf_names = {name for name, count in leaf_counts.items() if count > 1}

    # Assign each mesh: walk ancestors deepest-first, find first match. Duplicate
    # mesh leaf names are ambiguous in node_roles, so prefer their unique parent
    # links when present.
    assignments: dict[str, str] = {}
    for mesh_path in mesh_paths:
        parts = mesh_path.strip("/").split("/")
        assigned = None
        for idx, part in enumerate(reversed(parts)):
            if idx == 0 and part in ambiguous_leaf_names:
                continue
            if part in node_to_seg:
                assigned = node_to_seg[part]
                break
        assignments[mesh_path] = assigned or segment_names[0]

    return assignments


def _classified_ancestor_matches(
    mesh_paths: list[str],
    node_roles: dict[str, dict[str, str]],
    segment_names: list[str],
) -> dict[str, bool]:
    """Return whether each mesh path matched a classified ancestor node."""
    allowed = set(segment_names)
    classified_nodes = {
        node_name
        for node_name, info in node_roles.items()
        if info.get("segment") in allowed
    }

    leaf_counts = Counter(path.strip("/").split("/")[-1] for path in mesh_paths)
    ambiguous_leaf_names = {name for name, count in leaf_counts.items() if count > 1}

    matches: dict[str, bool] = {}
    for mesh_path in mesh_paths:
        parts = mesh_path.strip("/").split("/")
        matches[mesh_path] = any(
            part in classified_nodes
            for idx, part in enumerate(reversed(parts))
            if not (idx == 0 and part in ambiguous_leaf_names)
        )
    return matches


def _validate_prompt_library_assignments(
    assignments: dict[str, str],
    mesh_paths: list[str],
    node_roles: dict[str, dict[str, str]],
    segment_names: list[str],
    matched_mesh_paths: frozenset[str] | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return label and mesh-coverage gaps for prompt-library constrained output."""
    unknown, missing_raw = _prompt_library_role_label_gaps(node_roles, segment_names)

    assigned = set(assignments.values())
    missing_assigned = [name for name in segment_names if name not in assigned]

    if matched_mesh_paths is None:
        matched = _classified_ancestor_matches(mesh_paths, node_roles, segment_names)
        unmatched_meshes = [path for path in mesh_paths if not matched[path]]
    else:
        unmatched_meshes = [
            path for path in mesh_paths if path not in matched_mesh_paths
        ]

    missing = sorted(set(missing_raw + missing_assigned), key=segment_names.index)
    return unknown, missing, missing_assigned, unmatched_meshes


def _prompt_library_role_label_gaps(
    node_roles: dict[str, dict[str, str]],
    segment_names: list[str],
) -> tuple[list[str], list[str]]:
    """Return unknown and missing labels from the raw role taxonomy."""
    allowed = set(segment_names)
    unknown = sorted(
        {
            info.get("segment", "")
            for info in node_roles.values()
            if info.get("segment") and info.get("segment") not in allowed
        }
    )
    present = {
        info.get("segment", "")
        for info in node_roles.values()
        if info.get("segment") in allowed
    }
    missing = [name for name in segment_names if name not in present]
    return unknown, missing


def _physics_joint_edges(usd_path: str) -> set[tuple[str, str]]:
    """Return directed articulated body edges from explicit USD physics joints."""
    from pxr import Usd

    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception:
        return set()
    if stage is None:
        return set()

    edges: set[tuple[str, str]] = set()
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()
        if prim_type not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            continue
        body0 = prim.GetRelationship("physics:body0").GetTargets()
        body1 = prim.GetRelationship("physics:body1").GetTargets()
        if body0 and body1:
            a = str(body0[0])
            b = str(body1[0])
            edges.add((a, b))
    return edges


def _physics_fixed_joint_body_paths(usd_path: str) -> set[str]:
    """Return child body paths attached by fixed joints."""
    adjacency = _physics_fixed_joint_body_adjacency(usd_path)
    body_paths: set[str] = set()
    for a, neighbors in adjacency.items():
        for b in neighbors:
            if _is_strict_descendant_path(b, a):
                body_paths.add(b)
            elif _is_strict_descendant_path(a, b):
                body_paths.add(a)
            else:
                body_paths.update((a, b))
    return body_paths


def _physics_fixed_joint_body_adjacency(usd_path: str) -> dict[str, set[str]]:
    """Return undirected body adjacency from explicit USD fixed joints."""
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(usd_path)
    except Exception:
        return {}
    if stage is None:
        return {}

    adjacency: dict[str, set[str]] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue
        body0 = prim.GetRelationship("physics:body0").GetTargets()
        body1 = prim.GetRelationship("physics:body1").GetTargets()
        if not (body0 and body1):
            continue
        a = str(body0[0])
        b = str(body1[0])
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def _unique_role_prim_paths(
    usd_path: str,
    node_roles: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Resolve classified node names to unique USD prim paths."""
    from collections import defaultdict

    from pxr import Usd

    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception:
        return {}
    if stage is None:
        return {}

    names = set(node_roles)
    matches: dict[str, list[str]] = defaultdict(list)
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in names:
            matches[name].append(str(prim.GetPath()))

    return {name: paths[0] for name, paths in matches.items() if len(paths) == 1}


def _is_strict_descendant_path(path: str, ancestor: str) -> bool:
    """Return whether path is below ancestor in USD namespace."""
    normalized_ancestor = ancestor.rstrip("/")
    return path != normalized_ancestor and path.startswith(f"{normalized_ancestor}/")


def _parent_prim_path(path: str) -> str:
    """Return parent USD prim path text."""
    stripped = path.rstrip("/")
    parent, _sep, _leaf = stripped.rpartition("/")
    return parent or "/"


def _normalize_identity_text(value: str) -> str:
    """Normalize node, path, or segment text for conservative identity checks."""
    with_camel_boundaries = re.sub(r"([a-z])([A-Z0-9])", r"\1_\2", value)
    normalized = re.sub(r"[^a-z0-9]+", "_", with_camel_boundaries.lower())
    return normalized.strip("_")


def _segment_identity_aliases(segment: str) -> set[str]:
    """Return conservative text aliases that identify a prompt-library segment."""
    normalized = _normalize_identity_text(segment)
    aliases = {normalized}
    for suffix in ("_base", "_flange"):
        if normalized.endswith(suffix):
            aliases.add(normalized[: -len(suffix)])
    return {alias for alias in aliases if alias}


def _segment_identity_supported(node_name: str, path: str, segment: str) -> bool:
    """Return whether node/path text corroborates the assigned segment.

    This is intentionally narrow and only used when direct physics connectivity
    is otherwise ambiguous. It rejects salvage for flat chains whose labels are
    connected but semantically reversed.
    """
    normalized_text = _normalize_identity_text(f"{node_name} {path}")
    bounded_text = f"_{normalized_text}_"
    compact_text = normalized_text.replace("_", "")
    for alias in _segment_identity_aliases(segment):
        if f"_{alias}_" in bounded_text:
            return True
        compact_alias = alias.replace("_", "")
        if not any(char.isdigit() for char in compact_alias) and compact_alias:
            if compact_alias in compact_text:
                return True
    return False


def _mesh_paths_covered_by_role_path(
    mesh_paths: list[str] | None,
    role_path: str,
) -> frozenset[str]:
    """Return mesh paths covered by a validated role prim path."""
    if not mesh_paths:
        return frozenset()
    return frozenset(
        mesh_path
        for mesh_path in mesh_paths
        if mesh_path == role_path or _is_strict_descendant_path(mesh_path, role_path)
    )


def _mesh_paths_structurally_owned_by_role_path(
    mesh_paths: list[str] | None,
    role_path: str,
    segment: str,
    descendant_candidate_paths: set[str],
    fixed_joint_body_paths: set[str],
) -> frozenset[str]:
    """Return meshes owned by a broad chain candidate.

    For nested root-anchored chains, an ancestor link's subtree also contains all
    downstream link bodies. Ownership is structural: a candidate owns direct mesh
    children and fixed-joint rigid subassemblies, but not descendant chain
    candidates. Segment-name identity remains a fallback for authored helper
    groups that are not represented in physics.
    """
    if not descendant_candidate_paths:
        return _mesh_paths_covered_by_role_path(mesh_paths, role_path)
    if not mesh_paths:
        return frozenset()

    excluded_paths = {
        path
        for path in descendant_candidate_paths
        if _is_strict_descendant_path(path, role_path)
    }
    owned_fixed_paths = {
        path
        for path in fixed_joint_body_paths
        if _is_strict_descendant_path(path, role_path)
        and not any(
            path == excluded or _is_strict_descendant_path(path, excluded)
            for excluded in excluded_paths
        )
    }

    covered = []
    for mesh_path in mesh_paths:
        if mesh_path != role_path and not _is_strict_descendant_path(
            mesh_path, role_path
        ):
            continue
        if any(
            mesh_path == excluded or _is_strict_descendant_path(mesh_path, excluded)
            for excluded in excluded_paths
        ):
            continue

        relative_path = mesh_path.removeprefix(role_path).strip("/")
        if _parent_prim_path(mesh_path) == role_path:
            covered.append(mesh_path)
        elif any(
            mesh_path == fixed_path or _is_strict_descendant_path(mesh_path, fixed_path)
            for fixed_path in owned_fixed_paths
        ):
            covered.append(mesh_path)
        elif relative_path and _segment_identity_supported("", relative_path, segment):
            covered.append(mesh_path)
    return frozenset(covered)


def _physics_body_path_for_role_path(
    role_path: str,
    joint_body_paths: set[str],
) -> str | None:
    """Return the nearest physics joint body at or above a classified role path."""
    path = role_path.rstrip("/") or "/"
    while True:
        if path in joint_body_paths:
            return path
        if path == "/":
            return None
        path = _parent_prim_path(path)


def _supports_first_physics_chain_candidate(
    candidate: tuple[str, str],
    segment: str,
    body_path: str,
    candidate_body_paths: set[str],
) -> bool:
    """Return whether a first-segment candidate is specific enough to salvage.

    Root-anchored USDs often use an articulation/root prim as a joint endpoint.
    That root is not a safe base-link substitute: by descendant coverage it can
    make unrelated branches look assigned. Allow the first segment only when its
    own text identifies the segment, or when it is a physical endpoint that is
    not just an ancestor container for the rest of the candidate chain.
    """
    node_name, path = candidate
    if _segment_identity_supported(node_name, path, segment):
        return True
    return not any(
        other_path != body_path and _is_strict_descendant_path(other_path, body_path)
        for other_path in candidate_body_paths
    )


def _fixed_joint_owner_paths(
    body_path: str,
    candidate_body_paths: set[str],
    fixed_joint_body_adjacency: dict[str, set[str]],
) -> set[str]:
    """Return fixed-connected non-chain bodies owned by a chain body."""
    owned_paths: set[str] = set()
    visited = {body_path}
    stack = list(fixed_joint_body_adjacency.get(body_path, set()))
    while stack:
        path = stack.pop()
        if path in visited:
            continue
        visited.add(path)
        if path in candidate_body_paths:
            continue
        if any(
            candidate_path != body_path
            and _is_strict_descendant_path(candidate_path, path)
            for candidate_path in candidate_body_paths
        ):
            continue
        owned_paths.add(path)
        stack.extend(fixed_joint_body_adjacency.get(path, set()) - visited)
    return owned_paths


def _candidate_coverage(
    mesh_paths: list[str] | None,
    candidate: tuple[str, str],
    segment: str,
    body_path: str,
    candidate_body_paths: set[str],
    fixed_joint_body_paths: set[str],
    fixed_joint_body_adjacency: dict[str, set[str]],
    include_fixed_ancestors: bool = False,
) -> frozenset[str]:
    """Return mesh coverage for a chain candidate.

    If a selected candidate is an ancestor of later chain candidates, it must
    not blanket-cover every descendant mesh. Use subtree subtraction instead of
    requiring segment names in mesh leaves, since converted USDs often use
    generic mesh names such as ``ring_big`` or ``arm``.
    """
    owner_paths = {candidate[1]}
    body_path_is_broad_anchor = (
        body_path != candidate[1]
        and _is_strict_descendant_path(candidate[1], body_path)
        and any(
            candidate_path != body_path
            and _is_strict_descendant_path(candidate_path, body_path)
            for candidate_path in candidate_body_paths
        )
    )
    if not body_path_is_broad_anchor:
        owner_paths.add(body_path)
    owner_paths.update(
        _fixed_joint_owner_paths(
            body_path,
            candidate_body_paths,
            fixed_joint_body_adjacency,
        )
    )
    if include_fixed_ancestors:
        owner_paths.update(
            path
            for path in fixed_joint_body_paths
            if _is_strict_descendant_path(candidate[1], path)
        )

    coverage: frozenset[str] = frozenset()
    for owner_path in owner_paths:
        descendant_candidate_paths = {
            path
            for path in candidate_body_paths
            if path != body_path and _is_strict_descendant_path(path, owner_path)
        }
        coverage |= _mesh_paths_structurally_owned_by_role_path(
            mesh_paths,
            owner_path,
            segment,
            descendant_candidate_paths,
            fixed_joint_body_paths,
        )
    return coverage


def _physics_chain_state_sort_key(
    state: tuple[tuple[str, str], str | None, bool, bool],
) -> tuple[str, str, str, bool, bool]:
    """Return a stable ordering key for physics-chain DP states."""
    candidate, anchor, requires_identity, identity_valid = state
    return (
        candidate[0],
        candidate[1],
        anchor or "",
        requires_identity,
        identity_valid,
    )


def _immediate_joint_body_descendants_by_path(
    paths: set[str],
    joint_body_paths: set[str],
) -> dict[str, list[str]]:
    """Return immediate joint-body descendants for each path.

    This precomputes namespace ancestry once for salvage validation rather than
    scanning all joint endpoints inside every candidate transition.
    """
    descendants_by_path: dict[str, list[str]] = {path: [] for path in paths}
    for path in sorted(paths):
        prefix = f"{path.rstrip('/')}/"
        for body_path in sorted(joint_body_paths):
            if not body_path.startswith(prefix):
                continue

            parent = _parent_prim_path(body_path)
            has_joint_body_between = False
            while parent != path and parent.startswith(prefix):
                if parent in joint_body_paths:
                    has_joint_body_between = True
                    break
                parent = _parent_prim_path(parent)

            if not has_joint_body_between:
                descendants_by_path[path].append(body_path)
    return descendants_by_path


def _supports_physics_joint_chain_transition(
    previous_path: str,
    candidate_path: str,
    undirected_edges: set[tuple[str, str]],
    immediate_descendants: dict[str, list[str]],
    anchors_by_body: dict[str, set[str]],
) -> tuple[bool, str | None, bool]:
    """Return whether USD physics supports previous_path -> candidate_path.

    Some USDs author direct body0 -> body1 joints between adjacent links. Others
    keep body0 at the articulation root and put each nested link on body1; for
    those, hierarchy ancestry supplies the chain order while the joint endpoint
    proves that the candidate is an articulated body. Fallback transitions
    return the common root anchor that must stay stable along the chain.
    """
    if (previous_path, candidate_path) in undirected_edges:
        return True, None, not _is_strict_descendant_path(candidate_path, previous_path)

    if immediate_descendants.get(previous_path) != [candidate_path]:
        return False, None, False

    fallback_anchors = {
        anchor
        for anchor in anchors_by_body.get(candidate_path, set())
        if _is_strict_descendant_path(previous_path, anchor)
    }
    if len(fallback_anchors) == 1:
        return True, next(iter(fallback_anchors)), False

    return False, None, False


def _validated_physics_joint_chain_validation(
    usd_path: str,
    node_roles: dict[str, dict[str, str]],
    segment_names: list[str],
    mesh_paths: list[str] | None = None,
) -> _PhysicsJointChainValidation | None:
    """Return validated roles and mesh coverage for an explicit USD joint chain.

    This is intentionally used only as a salvage check when the LLM marks a
    prompt-library hierarchy as non-informative. Completeness proves taxonomy
    coverage; this proves that the classified link roles also align with an
    independent kinematic signal authored in the USD.
    """
    role_paths = _unique_role_prim_paths(usd_path, node_roles)
    if not role_paths:
        return None

    edges = _physics_joint_edges(usd_path)
    if not edges:
        return None
    if len(edges) > MAX_PHYSICS_CHAIN_EDGES:
        logger.info(
            "Skipping physics joint chain validation: %d edges exceeds cap %d",
            len(edges),
            MAX_PHYSICS_CHAIN_EDGES,
        )
        return None

    paths_by_segment: dict[str, list[tuple[str, str]]] = {
        name: [] for name in segment_names
    }
    for node_name, info in node_roles.items():
        segment = info.get("segment")
        role = info.get("role", "").strip().lower()
        path = role_paths.get(node_name)
        if segment in paths_by_segment and role in PHYSICS_CHAIN_ROLES and path:
            paths_by_segment[segment].append((node_name, path))

    if any(not paths_by_segment[name] for name in segment_names):
        return None
    total_candidates = sum(len(candidates) for candidates in paths_by_segment.values())
    if total_candidates > MAX_PHYSICS_CHAIN_TOTAL_CANDIDATES:
        logger.info(
            "Skipping physics joint chain validation: %d total candidates "
            "exceeds cap %d",
            total_candidates,
            MAX_PHYSICS_CHAIN_TOTAL_CANDIDATES,
        )
        return None
    for segment, candidates in paths_by_segment.items():
        if len(candidates) > MAX_PHYSICS_CHAIN_CANDIDATES_PER_SEGMENT:
            logger.info(
                "Skipping physics joint chain validation: segment %s has %d "
                "candidates, exceeding cap %d",
                segment,
                len(candidates),
                MAX_PHYSICS_CHAIN_CANDIDATES_PER_SEGMENT,
            )
            return None

    joint_body_paths = {path for edge in edges for path in edge}
    candidate_body_paths = {
        candidate: body_path
        for candidates in paths_by_segment.values()
        for candidate in candidates
        for body_path in [
            _physics_body_path_for_role_path(candidate[1], joint_body_paths)
        ]
        if body_path is not None
    }
    for segment in segment_names:
        paths_by_segment[segment] = [
            candidate
            for candidate in paths_by_segment[segment]
            if candidate in candidate_body_paths
        ]
    if any(not paths_by_segment[name] for name in segment_names):
        return None

    candidate_body_path_set = set(candidate_body_paths.values())
    fixed_joint_body_paths = _physics_fixed_joint_body_paths(usd_path)
    fixed_joint_body_adjacency = _physics_fixed_joint_body_adjacency(usd_path)
    first_segment = segment_names[0]
    paths_by_segment[first_segment] = [
        candidate
        for candidate in paths_by_segment[first_segment]
        if _supports_first_physics_chain_candidate(
            candidate,
            first_segment,
            candidate_body_paths[candidate],
            candidate_body_path_set,
        )
    ]
    if not paths_by_segment[first_segment]:
        return None

    immediate_descendants = _immediate_joint_body_descendants_by_path(
        candidate_body_path_set,
        joint_body_paths,
    )
    undirected_edges = edges | {(dst, src) for src, dst in edges}
    anchors_by_body: dict[str, set[str]] = {}
    for src, dst in edges:
        anchors_by_body.setdefault(src, set()).add(dst)
        anchors_by_body.setdefault(dst, set()).add(src)

    expected_mesh_paths = frozenset(mesh_paths or [])
    previous: dict[
        tuple[tuple[str, str], str | None, bool, bool],
        tuple[tuple[tuple[str, str], str | None, bool, bool] | None, frozenset[str]],
    ] = {
        (
            candidate,
            None,
            False,
            _segment_identity_supported(candidate[0], candidate[1], first_segment),
        ): (
            None,
            _candidate_coverage(
                mesh_paths,
                candidate,
                first_segment,
                candidate_body_paths[candidate],
                candidate_body_path_set,
                fixed_joint_body_paths,
                fixed_joint_body_adjacency,
                include_fixed_ancestors=True,
            ),
        )
        for candidate in sorted(paths_by_segment[first_segment])
    }
    predecessor_layers = [previous]
    transition_count = 0
    for segment in segment_names[1:]:
        next_candidates = sorted(paths_by_segment[segment])
        current: dict[
            tuple[tuple[str, str], str | None, bool, bool],
            tuple[
                tuple[tuple[str, str], str | None, bool, bool] | None,
                frozenset[str],
            ],
        ] = {}
        for candidate in next_candidates:
            for prior in sorted(previous, key=_physics_chain_state_sort_key):
                transition_count += 1
                if transition_count > MAX_PHYSICS_CHAIN_TRANSITIONS:
                    logger.info(
                        "Skipping physics joint chain validation: transition "
                        "count exceeds cap %d",
                        MAX_PHYSICS_CHAIN_TRANSITIONS,
                    )
                    return None
                (
                    prior_candidate,
                    prior_anchor,
                    prior_requires_identity,
                    prior_identity_valid,
                ) = prior
                _prior_predecessor, prior_coverage = previous[prior]
                is_supported, transition_anchor, transition_requires_identity = (
                    _supports_physics_joint_chain_transition(
                        candidate_body_paths[prior_candidate],
                        candidate_body_paths[candidate],
                        undirected_edges,
                        immediate_descendants,
                        anchors_by_body,
                    )
                )
                if not is_supported:
                    continue
                requires_identity = (
                    prior_requires_identity or transition_requires_identity
                )
                identity_valid = prior_identity_valid and _segment_identity_supported(
                    candidate[0],
                    candidate[1],
                    segment,
                )
                if requires_identity and not identity_valid:
                    continue
                if (
                    transition_anchor is not None
                    and prior_anchor is not None
                    and transition_anchor != prior_anchor
                ):
                    continue
                state = (
                    candidate,
                    transition_anchor or prior_anchor,
                    requires_identity,
                    identity_valid,
                )
                coverage = prior_coverage | _candidate_coverage(
                    mesh_paths,
                    candidate,
                    segment,
                    candidate_body_paths[candidate],
                    candidate_body_path_set,
                    fixed_joint_body_paths,
                    fixed_joint_body_adjacency,
                )
                existing = current.get(state)
                if existing is not None and len(coverage) <= len(existing[1]):
                    continue
                current[state] = (
                    prior,
                    coverage,
                )
        if not current:
            return None
        predecessor_layers.append(current)
        previous = current

    selected = max(
        previous,
        key=lambda state: (
            len(previous[state][1]),
            _physics_chain_state_sort_key(state),
        ),
    )
    if expected_mesh_paths and previous[selected][1] != expected_mesh_paths:
        return None

    chain: list[tuple[str, str]] = []
    for layer in reversed(predecessor_layers):
        chain.append(selected[0])
        predecessor, _coverage = layer[selected]
        if predecessor is None:
            break
        selected = predecessor

    ordered_chain = list(reversed(chain))
    mesh_assignments: dict[str, str] = {}
    covered_mesh_paths: frozenset[str] = frozenset()
    for idx, (segment, candidate) in enumerate(
        zip(segment_names, ordered_chain, strict=True)
    ):
        segment_coverage = _candidate_coverage(
            mesh_paths,
            candidate,
            segment,
            candidate_body_paths[candidate],
            candidate_body_path_set,
            fixed_joint_body_paths,
            fixed_joint_body_adjacency,
            include_fixed_ancestors=idx == 0,
        )
        covered_mesh_paths |= segment_coverage
        for mesh_path in sorted(segment_coverage):
            mesh_assignments[mesh_path] = segment

    validated_names = {name for name, _path in ordered_chain}
    return _PhysicsJointChainValidation(
        node_roles={
            name: dict(info)
            for name, info in node_roles.items()
            if name in validated_names
        },
        mesh_assignments=mesh_assignments,
        covered_mesh_paths=covered_mesh_paths,
    )


def _validated_physics_joint_chain_roles(
    usd_path: str,
    node_roles: dict[str, dict[str, str]],
    segment_names: list[str],
    mesh_paths: list[str] | None = None,
) -> dict[str, dict[str, str]] | None:
    """Return the validated role subset for an explicit USD joint chain."""
    validation = _validated_physics_joint_chain_validation(
        usd_path,
        node_roles,
        segment_names,
        mesh_paths,
    )
    if validation is None:
        return None
    return validation.node_roles


def _validates_physics_joint_chain(
    usd_path: str,
    node_roles: dict[str, dict[str, str]],
    segment_names: list[str],
) -> bool:
    """Return whether classified roles match an explicit USD joint chain."""
    return (
        _validated_physics_joint_chain_roles(usd_path, node_roles, segment_names)
        is not None
    )


# ---------------------------------------------------------------------------
# Segment name inference
# ---------------------------------------------------------------------------

INFER_SEGMENTS_SYSTEM_PROMPT = """\
You are a robotics expert. Given a USD scene tree of an articulated body,
identify the robot and determine its segment names.

Do NOT rely on prim names — reason from the tree structure (nesting depth,
number of chain nodes, branching pattern) and any geometric stats provided."""


def build_infer_segments_prompt(
    tree_text: str,
    asset_type: str | None = None,
    asset_subtype: str | None = None,
) -> str:
    """Build a prompt to infer segment names from the scene tree."""
    context = ""
    if asset_type:
        context = f"\n\nThe asset has been identified as: {asset_type}"
        if asset_subtype:
            context += f" ({asset_subtype})"
        context += "."

    return f"""Analyze this USD scene tree and determine the segment names
for this articulated body:{context}

```
{tree_text}
```

Determine:
1. What type of articulated body is this? (robot arm, humanoid, quadruped, etc.)
2. How many degrees of freedom / segments does it have?
3. What are the standard segment names, ordered from base to end-effector?

Use conventional robotics naming for the identified robot type. For example:
- 6-axis robot arm: base, shoulder, upper_arm, forearm, wrist_1, wrist_2, wrist_3
- 7-DOF arm: link0 through link7 (or base, shoulder, upper_arm, elbow, forearm, wrist, hand, flange)

Respond with JSON only:
<answer>
{{
  "robot_type": "...",
  "dof": N,
  "segment_names": ["base", "shoulder", ...]
}}
</answer>"""


def infer_segment_names(
    usd_path: str,
    llm_generate_fn: Any,
    asset_type: str | None = None,
    asset_subtype: str | None = None,
    vlm_generate_with_images_fn: Any | None = None,
    preview_images: list[str] | None = None,
    *,
    asset_confidence: Any | None = None,
    use_prompt_library: bool = False,
    robot_id: str | None = None,
) -> list[str]:
    """Infer segment names from the scene tree and optional preview images.

    When preview images are available and a VLM generate function is provided,
    the VLM is used for identification — this is more reliable for generic
    hierarchies where the tree alone doesn't reveal the robot model.

    Args:
        usd_path: Path to the USD file.
        llm_generate_fn: Callable(system_prompt, user_prompt) -> str.
        asset_type: Optional asset type from identify_asset step.
        asset_subtype: Optional asset subtype.
        asset_confidence: Optional confidence from identify_asset step.
        vlm_generate_with_images_fn: Optional callable(system_prompt, user_prompt,
            image_paths) -> str. Used when preview images are available.
        preview_images: Optional list of preview image paths from identify_asset.

    Returns:
        List of segment names ordered from base to end-effector,
        or an empty list if inference fails.
    """
    # Prompt-library shortcut: only explicit opt-in may return canonical robot
    # segment names directly. Generic/default runs must not silently use known
    # benchmark or filename-specific prompt data.
    matched_entry, prompt_library_metadata = _resolve_prompt_library_entry(
        use_prompt_library=use_prompt_library,
        robot_id=robot_id,
        asset_type=asset_type,
        asset_subtype=asset_subtype,
        asset_confidence=asset_confidence,
        usd_path=usd_path,
    )
    if prompt_library_metadata.get("prompt_library_robot_id"):
        if matched_entry is None:
            logger.warning(
                "Prompt library entry %r is unavailable; falling back to "
                "model segment inference",
                prompt_library_metadata["prompt_library_robot_id"],
            )
        else:
            logger.info(
                "Prompt library hit: robot_id=%s, source=%s, %d segments",
                matched_entry.robot_id,
                prompt_library_metadata.get("prompt_library_match_source"),
                len(matched_entry.component_names),
            )
            return list(matched_entry.component_names)

    tree_text, _ = extract_scene_tree(usd_path)
    user_prompt = build_infer_segments_prompt(tree_text, asset_type, asset_subtype)

    # If we have preview images and a VLM, use vision for identification
    # This is critical for generic hierarchies (Joint_001, Joint_002...)
    # where the LLM alone can't determine the robot model
    if preview_images and vlm_generate_with_images_fn:
        logger.info(
            "Using VLM with %d preview images for robot identification",
            len(preview_images),
        )
        response = _call_with_retry(
            vlm_generate_with_images_fn,
            INFER_SEGMENTS_SYSTEM_PROMPT,
            user_prompt,
            preview_images,
            label="segment inference (VLM)",
        )
    else:
        response = _call_with_retry(
            llm_generate_fn,
            INFER_SEGMENTS_SYSTEM_PROMPT,
            user_prompt,
            label="segment inference (LLM)",
        )

    logger.info("Segment inference response (%d chars)", len(response))

    # Parse
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    json_str = answer_match.group(1).strip() if answer_match else response
    json_match = re.search(r"\{[\s\S]*\}", json_str)
    if json_match:
        try:
            data = json.loads(json_match.group())
            names = data.get("segment_names", []) if isinstance(data, dict) else []
            if (
                isinstance(names, list)
                and len(names) >= 2
                and all(isinstance(name, str) for name in names)
            ):
                logger.info("Inferred %d segments: %s", len(names), ", ".join(names))
                return names
        except json.JSONDecodeError:
            pass

    logger.warning(
        "Failed to infer segment names; no hardcoded default taxonomy is used"
    )
    return []


def _has_meaningful_asset_subtype(asset_subtype: str | None) -> bool:
    """Return whether subtype is a real identity signal, not a placeholder."""
    if not asset_subtype:
        return False
    subtype_token = re.sub(r"[^a-z0-9]+", "", asset_subtype.lower())
    if subtype_token in {"", "unknown", "none", "null", "na", "notapplicable"}:
        return False
    return True


def _has_robot_arm_asset_type(asset_type: str | None) -> bool:
    """Return whether asset_type is a strong robot-arm identity signal."""
    if not asset_type:
        return False
    asset_type_token = re.sub(r"[^a-z0-9]+", "", asset_type.lower())
    if asset_type_token in {
        "robot",
        "robotarm",
        "roboticarm",
        "industrialrobot",
        "industrialrobotarm",
        "collaborativerobot",
        "collaborativerobotarm",
        "articulatedrobot",
        "articulatedrobotarm",
        "robotmanipulator",
        "manipulator",
    }:
        return True
    return (
        "robotarm" in asset_type_token
        or "roboticarm" in asset_type_token
        or "manipulator" in asset_type_token
    )


def _has_robot_arm_asset_subtype(asset_subtype: str | None) -> bool:
    """Return whether asset_subtype itself identifies a robot arm class."""
    if asset_subtype is None or not _has_meaningful_asset_subtype(asset_subtype):
        return False
    subtype_token = re.sub(r"[^a-z0-9]+", "", asset_subtype.lower())
    if subtype_token in {
        "robotarm",
        "roboticarm",
        "industrialrobotarm",
        "collaborativerobotarm",
        "articulatedrobotarm",
        "robotmanipulator",
        "roboticmanipulator",
        "manipulator",
    }:
        return True
    return (
        "robotarm" in subtype_token
        or "roboticarm" in subtype_token
        or "manipulator" in subtype_token
    )


def _has_robot_arm_compatible_asset_type(asset_type: str | None) -> bool:
    """Return whether a broad asset_type can safely pair with robot-arm subtype."""
    if not asset_type:
        return True
    if _has_robot_arm_asset_type(asset_type):
        return True
    asset_type_token = re.sub(r"[^a-z0-9]+", "", asset_type.lower())
    return asset_type_token in {
        "equipment",
        "industrialequipment",
        "machine",
        "industrialmachine",
        "machinery",
        "industrialmachinery",
    }


def _has_trusted_asset_confidence(asset_confidence: Any | None) -> bool:
    """Return whether identify_asset confidence is strong enough to shortcut."""
    if isinstance(asset_confidence, str):
        return asset_confidence.strip().lower() == "high"
    if isinstance(asset_confidence, bool):
        return False
    if isinstance(asset_confidence, int | float):
        return float(asset_confidence) >= 0.8
    return False


def _has_prompt_library_shortcut_signal(
    asset_type: str | None,
    asset_subtype: str | None,
    asset_confidence: Any | None = None,
) -> bool:
    """Return whether metadata is strong enough to trust a library shortcut."""
    if not _has_trusted_asset_confidence(asset_confidence):
        return False
    if _has_robot_arm_asset_type(asset_type) and _has_meaningful_asset_subtype(
        asset_subtype
    ):
        return True
    if _has_robot_arm_asset_subtype(
        asset_subtype
    ) and _has_robot_arm_compatible_asset_type(asset_type):
        return True
    return False


def _resolve_prompt_library_robot_id(
    *,
    lookup_robot_id: Callable[..., str | None],
    use_prompt_library: bool,
    robot_id: str | None,
    asset_type: str | None,
    asset_subtype: str | None,
    asset_confidence: Any | None,
    usd_path: str | None,
) -> tuple[str | None, str | None]:
    """Resolve an opt-in prompt-library robot ID and its source."""
    if not use_prompt_library:
        return None, None

    explicit_robot_id = robot_id.strip() if isinstance(robot_id, str) else ""
    if explicit_robot_id:
        return explicit_robot_id, "explicit_robot_id"

    if _has_prompt_library_shortcut_signal(
        asset_type,
        asset_subtype,
        asset_confidence,
    ):
        matched_robot_id = lookup_robot_id(asset_type, asset_subtype, None)
        if matched_robot_id:
            return matched_robot_id, "asset_metadata"
        matched_robot_id = lookup_robot_id(asset_type, asset_subtype, usd_path)
        if matched_robot_id:
            return matched_robot_id, "usd_filename"

    return None, None


def _load_prompt_entry_or_error(
    load_prompt: Callable[[str], Any],
    robot_id: str,
) -> tuple[Any | None, str | None]:
    """Load a prompt entry, returning an error string for bad explicit IDs."""
    try:
        return load_prompt(robot_id), None
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        logger.warning("Prompt library entry %r is unavailable: %s", robot_id, exc)
        return None, str(exc)


def _resolve_prompt_library_entry(
    *,
    use_prompt_library: bool,
    robot_id: str | None,
    asset_type: str | None,
    asset_subtype: str | None,
    asset_confidence: Any | None,
    usd_path: str | None,
) -> tuple[Any | None, dict[str, Any]]:
    """Resolve an opt-in prompt-library entry and provenance metadata."""
    from joint_agent.prompts import load_prompt, lookup_robot_id

    matched_robot_id, match_source = _resolve_prompt_library_robot_id(
        lookup_robot_id=lookup_robot_id,
        use_prompt_library=use_prompt_library,
        robot_id=robot_id,
        asset_type=asset_type,
        asset_subtype=asset_subtype,
        asset_confidence=asset_confidence,
        usd_path=usd_path,
    )
    if not matched_robot_id:
        return None, {}

    matched_entry, prompt_library_error = _load_prompt_entry_or_error(
        load_prompt,
        matched_robot_id,
    )
    metadata: dict[str, Any] = {
        "prompt_library_robot_id": (
            matched_entry.robot_id if matched_entry else matched_robot_id
        ),
        "prompt_library_match_source": match_source,
    }
    if prompt_library_error:
        metadata["prompt_library_error"] = prompt_library_error
    return matched_entry, metadata


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def analyze_hierarchy(
    usd_path: str,
    segment_names: list[str] | None,
    llm_generate_fn: Any,
    asset_type: str | None = None,
    asset_subtype: str | None = None,
    vlm_generate_with_images_fn: Any | None = None,
    preview_images: list[str] | None = None,
    *,
    asset_confidence: Any | None = None,
    use_prompt_library: bool = False,
    robot_id: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Analyze a USD file's hierarchy and produce segment assignments.

    This is the main entry point for hierarchy-based analysis. It extracts
    the scene tree, sends it to an LLM, and returns assignments.

    Args:
        usd_path: Path to the USD file.
        segment_names: Ordered segment names from base to end-effector.
            If None, inferred automatically from the scene tree via LLM.
        llm_generate_fn: Callable(system_prompt, user_prompt) -> str.
            Abstracted so callers can provide any LLM backend.
        asset_type: Optional asset type from identify_asset step.
        asset_subtype: Optional asset subtype.
        asset_confidence: Optional confidence from identify_asset step.

    Returns:
        (assignments, metadata) where assignments maps prim_path -> segment_name
        and metadata contains strategy details.
    """

    def _base_metadata() -> dict[str, Any]:
        return {
            "prompt_library_used": False,
            "heuristic_paths_used": [],
        }

    # Extract scene tree
    tree_text, mesh_paths = extract_scene_tree(usd_path)
    logger.info("Extracted scene tree: %d meshes", len(mesh_paths))

    if not mesh_paths:
        logger.warning("No meshes found in USD file")
        return {}, {"strategy": "none", "reason": "no meshes", **_base_metadata()}

    # If the robot is in the prompt library, reuse that match for both segment
    # inference and analysis-prompt augmentation.
    from joint_agent.prompts import render_analysis_system_prompt

    matched_entry, prompt_library_metadata = _resolve_prompt_library_entry(
        use_prompt_library=use_prompt_library,
        robot_id=robot_id,
        asset_type=asset_type,
        asset_subtype=asset_subtype,
        asset_confidence=asset_confidence,
        usd_path=usd_path,
    )
    trusted_prompt_library = bool(matched_entry)
    prompt_library_active = False

    def _with_common_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        metadata.setdefault("heuristic_paths_used", [])
        metadata["prompt_library_used"] = prompt_library_active
        metadata.update(prompt_library_metadata)
        return metadata

    # Infer segment names if not provided
    segments_inferred = False
    if not segment_names:
        logger.info("No segment names provided — inferring from scene tree...")
        if matched_entry:
            logger.info(
                "Prompt library hit: robot_id=%s, %d segments",
                matched_entry.robot_id,
                len(matched_entry.component_names),
            )
            segment_names = list(matched_entry.component_names)
            prompt_library_active = True
        else:
            segment_names = infer_segment_names(
                usd_path,
                llm_generate_fn,
                asset_type,
                asset_subtype,
                vlm_generate_with_images_fn=vlm_generate_with_images_fn,
                preview_images=preview_images,
                asset_confidence=asset_confidence,
                use_prompt_library=False,
                robot_id=None,
            )
        segments_inferred = True
        if not segment_names:
            logger.info("Segment names unresolved; hierarchy analysis is disabled")
            return {}, _with_common_metadata(
                {
                    "strategy": "none",
                    "reason": "segment_names_unresolved",
                    "segments_inferred": segments_inferred,
                    "num_meshes": len(mesh_paths),
                    "num_assigned": 0,
                }
            )

    prompt_library_validation_required = bool(
        trusted_prompt_library
        and matched_entry
        and matched_entry.completeness_required
        and list(matched_entry.component_names) == list(segment_names)
    )

    # Build prompt and call LLM. If the robot is in the prompt library,
    # augment the system prompt with per-component visual cues + tiebreakers.
    user_prompt = build_analysis_prompt(tree_text, segment_names)
    system_prompt = ANALYSIS_SYSTEM_PROMPT
    if matched_entry and list(matched_entry.component_names) == list(segment_names):
        prompt_library_active = True
        system_prompt = render_analysis_system_prompt(
            ANALYSIS_SYSTEM_PROMPT, matched_entry
        )
        logger.info(
            "Augmenting analysis system prompt from library entry %s",
            matched_entry.robot_id,
        )

    response = _call_with_retry(
        llm_generate_fn,
        system_prompt,
        user_prompt,
        label="hierarchy analysis (LLM)",
    )
    logger.info("LLM response received (%d chars)", len(response))

    # Parse response
    parsed_response = _parse_analysis_response_details(response)
    is_informative = parsed_response.is_informative
    hierarchy_pattern = parsed_response.hierarchy_pattern
    node_roles = parsed_response.node_roles
    malformed_node_roles = list(parsed_response.malformed_node_roles)
    logger.info(
        "Hierarchy: informative=%s, pattern=%s, %d nodes classified",
        is_informative,
        hierarchy_pattern,
        len(node_roles),
    )

    def _non_informative_metadata() -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "strategy": "none",
            "reason": "hierarchy_not_informative",
            "hierarchy_pattern": hierarchy_pattern,
        }
        if segments_inferred:
            metadata["segment_names"] = segment_names
            metadata["segments_inferred"] = True
        return metadata

    if malformed_node_roles:
        raw_unknown_labels, raw_missing_required_segments = (
            _prompt_library_role_label_gaps(node_roles, segment_names)
        )
        if prompt_library_validation_required and matched_entry:
            logger.info(
                "Prompt-library analysis returned malformed raw role entries "
                "for %s: %s",
                matched_entry.robot_id,
                malformed_node_roles,
            )
            metadata = {
                "strategy": "none",
                "reason": "prompt_library_incomplete",
                "hierarchy_pattern": hierarchy_pattern,
                "node_roles": node_roles,
                "segment_names": segment_names,
                "segments_inferred": segments_inferred,
                "prompt_library_robot_id": matched_entry.robot_id,
                "unknown_labels": raw_unknown_labels,
                "malformed_node_roles": malformed_node_roles,
                "missing_required_segments": raw_missing_required_segments,
                "missing_assigned_segments": segment_names,
                "unmatched_mesh_paths": mesh_paths,
                "num_meshes": len(mesh_paths),
                "num_assigned": 0,
            }
            if not is_informative:
                metadata["non_informative_validation"] = (
                    "malformed_prompt_library_roles"
                )
            return {}, _with_common_metadata(metadata)

        logger.info(
            "Hierarchy analysis returned malformed raw role entries: %s",
            malformed_node_roles,
        )
        return {}, _with_common_metadata(
            {
                "strategy": "none",
                "reason": "malformed_node_roles",
                "hierarchy_pattern": hierarchy_pattern,
                "node_roles": node_roles,
                "malformed_node_roles": malformed_node_roles,
                "segment_names": segment_names,
                "segments_inferred": segments_inferred,
                "num_meshes": len(mesh_paths),
                "num_assigned": 0,
            }
        )

    if not node_roles:
        if (
            trusted_prompt_library
            and matched_entry
            and matched_entry.completeness_required
            and list(matched_entry.component_names) == list(segment_names)
        ):
            logger.info(
                "Prompt-library analysis was not informative for %s",
                matched_entry.robot_id,
            )
            return {}, _with_common_metadata(
                {
                    "strategy": "none",
                    "reason": "prompt_library_incomplete",
                    "hierarchy_pattern": hierarchy_pattern,
                    "node_roles": node_roles,
                    "segment_names": segment_names,
                    "segments_inferred": segments_inferred,
                    "unknown_labels": [],
                    "malformed_node_roles": malformed_node_roles,
                    "missing_required_segments": segment_names,
                    "missing_assigned_segments": segment_names,
                    "unmatched_mesh_paths": mesh_paths,
                    "non_informative_validation": (
                        "malformed_prompt_library_roles"
                        if malformed_node_roles
                        else "missing_prompt_library_roles"
                    ),
                    "num_meshes": len(mesh_paths),
                    "num_assigned": 0,
                }
            )
        logger.info("Hierarchy not informative — no assignments produced")
        return {}, _with_common_metadata(_non_informative_metadata())
    assignment_node_roles = node_roles
    assignment_hierarchy_pattern = hierarchy_pattern
    assignment_mesh_assignments: dict[str, str] | None = None
    assignment_matched_mesh_paths: frozenset[str] | None = None
    non_informative_validation: str | None = None
    if not is_informative:
        if not (
            trusted_prompt_library
            and matched_entry
            and matched_entry.completeness_required
            and list(matched_entry.component_names) == list(segment_names)
        ):
            logger.info("Hierarchy not informative — no assignments produced")
            return {}, _with_common_metadata(_non_informative_metadata())
        raw_unknown_labels, raw_missing_required_segments = (
            _prompt_library_role_label_gaps(node_roles, segment_names)
        )
        if raw_unknown_labels:
            logger.info(
                "Prompt-library analysis marked non-informative for %s but "
                "returned unknown labels: %s",
                matched_entry.robot_id,
                raw_unknown_labels,
            )
            return {}, _with_common_metadata(
                {
                    "strategy": "none",
                    "reason": "prompt_library_incomplete",
                    "hierarchy_pattern": hierarchy_pattern,
                    "node_roles": node_roles,
                    "segment_names": segment_names,
                    "segments_inferred": segments_inferred,
                    "unknown_labels": raw_unknown_labels,
                    "malformed_node_roles": malformed_node_roles,
                    "missing_required_segments": raw_missing_required_segments,
                    "missing_assigned_segments": segment_names,
                    "unmatched_mesh_paths": mesh_paths,
                    "non_informative_validation": "unknown_prompt_library_labels",
                    "num_meshes": len(mesh_paths),
                    "num_assigned": 0,
                }
            )
        physics_validation = _validated_physics_joint_chain_validation(
            usd_path,
            node_roles,
            segment_names,
            mesh_paths,
        )
        if physics_validation is None:
            logger.info(
                "Prompt-library analysis marked non-informative for %s and "
                "did not validate against an explicit USD physics joint chain",
                matched_entry.robot_id,
            )
            return {}, _with_common_metadata(
                {
                    "strategy": "none",
                    "reason": "prompt_library_incomplete",
                    "hierarchy_pattern": hierarchy_pattern,
                    "node_roles": node_roles,
                    "segment_names": segment_names,
                    "segments_inferred": segments_inferred,
                    "unknown_labels": raw_unknown_labels,
                    "malformed_node_roles": malformed_node_roles,
                    "missing_required_segments": raw_missing_required_segments,
                    "missing_assigned_segments": segment_names,
                    "unmatched_mesh_paths": mesh_paths,
                    "non_informative_validation": "missing_physics_joint_chain",
                    "num_meshes": len(mesh_paths),
                    "num_assigned": 0,
                }
            )
        assignment_node_roles = physics_validation.node_roles
        assignment_hierarchy_pattern = "link_centric"
        assignment_mesh_assignments = physics_validation.mesh_assignments
        assignment_matched_mesh_paths = physics_validation.covered_mesh_paths
        non_informative_validation = "physics_joint_chain"
        logger.info(
            "Prompt-library analysis marked non-informative for %s; "
            "classified node roles validated against USD physics joint chain",
            matched_entry.robot_id,
        )

    # Assign segments
    if assignment_mesh_assignments is None:
        assignments = assign_segments_from_roles(
            mesh_paths,
            assignment_node_roles,
            segment_names,
            assignment_hierarchy_pattern,
        )
    else:
        assignments = dict(assignment_mesh_assignments)

    if (
        trusted_prompt_library
        and matched_entry
        and matched_entry.completeness_required
        and list(matched_entry.component_names) == list(segment_names)
    ):
        (
            unknown_labels,
            missing_labels,
            missing_assigned_labels,
            unmatched_meshes,
        ) = _validate_prompt_library_assignments(
            assignments,
            mesh_paths,
            assignment_node_roles,
            segment_names,
            assignment_matched_mesh_paths,
        )
        if malformed_node_roles or unknown_labels or missing_labels or unmatched_meshes:
            logger.info(
                "Prompt-library completeness check failed for %s: "
                "malformed=%s unknown=%s missing=%s unmatched_meshes=%d",
                matched_entry.robot_id,
                malformed_node_roles,
                unknown_labels,
                missing_labels,
                len(unmatched_meshes),
            )
            return {}, _with_common_metadata(
                {
                    "strategy": "none",
                    "reason": "prompt_library_incomplete",
                    "hierarchy_pattern": hierarchy_pattern,
                    "node_roles": node_roles,
                    "segment_names": segment_names,
                    "segments_inferred": segments_inferred,
                    "unknown_labels": unknown_labels,
                    "malformed_node_roles": malformed_node_roles,
                    "missing_required_segments": missing_labels,
                    "missing_assigned_segments": missing_assigned_labels,
                    "unmatched_mesh_paths": unmatched_meshes,
                    "num_meshes": len(mesh_paths),
                    "num_assigned": len(assignments),
                }
            )

    metadata = {
        "strategy": "hierarchy",
        "hierarchy_pattern": assignment_hierarchy_pattern,
        "node_roles": assignment_node_roles,
        "segment_names": segment_names,
        "segments_inferred": segments_inferred,
        "num_meshes": len(mesh_paths),
        "num_assigned": len(assignments),
    }
    if non_informative_validation:
        metadata["non_informative_validation"] = non_informative_validation

    return assignments, _with_common_metadata(metadata)
