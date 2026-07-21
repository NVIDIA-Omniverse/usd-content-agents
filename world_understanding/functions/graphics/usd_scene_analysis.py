# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD scene analysis functions.

Higher-level analysis functions for detecting and classifying objects
in USD scenes using a feature-scoring algorithm.
"""

from __future__ import annotations

import hashlib
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from pxr import Usd

logger = logging.getLogger(__name__)

_SKIP_TYPES = frozenset({"Material", "Shader"})
_MATERIAL_SCOPE_NAMES = frozenset({"Looks", "Materials", "materials", "looks"})
_GENERIC_INSTANCE_BASE_NAMES = frozenset(
    {
        "body",
        "component",
        "geom",
        "geometry",
        "mesh",
        "node",
        "object",
        "obj",
        "part",
        "shape",
        "solid",
    }
)
_GENERIC_CONTEXT_NAMES = frozenset(
    {
        "assets",
        "geometry",
        "meshes",
        "model",
        "models",
        "root",
        "scene",
        "world",
    }
)


def _canonical_instance_context_name(name: str) -> str:
    """Return a stable semantic context name for numbered instance containers."""
    # Siemens/NX-style product names often keep a part code and append a short
    # instance suffix, e.g. FORKLIFT__MFE_000007669_5.  Preserve the part code
    # while removing only the authored instance suffix.
    match = re.match(r"^(.+)_\d{1,3}$", name)
    if match and "__" in match.group(1):
        return match.group(1)
    return name


def _semantic_instance_context(path: str) -> str | None:
    """Find the nearest useful semantic ancestor for generic instance names."""
    parts = [part for part in path.split("/") if part]
    for part in reversed(parts[:-1]):
        canonical = _canonical_instance_context_name(part)
        if not canonical:
            continue  # pragma: no cover - path parts are filtered to non-empty strings
        lower = canonical.lower()
        if lower in _GENERIC_CONTEXT_NAMES:
            continue
        if re.fullmatch(r"(?:arrangement|group|node|xform)_?\d*", lower):
            continue
        return canonical
    return None


def _name_pattern_group_key(path: str, base_name: str) -> tuple[str, str | None]:
    """Return a conservative grouping key for numbered-name instances."""
    canonical_base = _canonical_instance_context_name(base_name)
    if canonical_base.lower() in _GENERIC_INSTANCE_BASE_NAMES:
        return canonical_base, _semantic_instance_context(path)
    return canonical_base, None


def _instance_base_name(name: str) -> str | None:
    digit_start = len(name)
    while digit_start > 0 and name[digit_start - 1].isdigit():
        digit_start -= 1
    if digit_start == len(name):
        return None

    base = name[:digit_start]
    if base.endswith("__I"):
        base = base[:-3]
    elif base.endswith("__"):
        base = base[:-2]
    elif base.endswith("_"):
        base = base[:-1]
    return base or None


def _topology_group_key(obj: dict[str, Any]) -> tuple[int, int, int]:
    """Return a conservative geometry key for duplicate grouping."""
    return (
        int(obj.get("mesh_count", 0) or 0),
        int(obj.get("vertex_count", 0) or 0),
        int(obj.get("face_count", 0) or 0),
    )


def _bound_material_identity(material: Any, root_path: str | None = None) -> str | None:
    """Return a stable material identity for conservative grouping.

    Materials scoped under ``root_path`` (per-asset ``Looks`` copies emitted by
    CAD exporters) are identified by their root-relative path plus a fingerprint
    of their directly authored shader inputs, so structurally identical assets
    with private material copies compare equal only when their appearance
    actually matches too -- a shared relative path alone (e.g. two unrelated
    "Looks/Copper" prims with different authored colors) is not enough to
    collapse two otherwise-distinct assets into one representative.
    """
    if not material:
        return None
    prim = material.GetPrim()
    if not prim or not prim.IsValid():
        return None
    identity = str(prim.GetPath())
    if root_path and identity.startswith(root_path + "/"):
        relative = identity[len(root_path) + 1 :]
        fingerprint = _material_appearance_fingerprint(material)
        if fingerprint is None:
            # The shader network could not be fully fingerprinted (e.g. a
            # connection into a node graph or missing prim we can't resolve
            # a value from). Collapsing by relative path alone risks a
            # false-positive match between two private materials that only
            # *look* identical because we failed to inspect them, so fall
            # back to the material's absolute identity instead.
            return identity
        return f"{relative}#{fingerprint}" if fingerprint else relative
    return identity


def _material_appearance_fingerprint(material: Any) -> str | None:
    """Return a short hash of a material's authored surface shader network.

    Returns ``None`` when the network cannot be fully resolved (e.g. a
    connection leads into something other than a ``UsdShade.Shader``, such as
    a node graph) rather than an empty string, so callers can distinguish
    "genuinely no authored values" (safe to match on path alone) from
    "could not inspect this" (unsafe to match at all).
    """
    from pxr import UsdShade

    usd_material = UsdShade.Material(material.GetPrim())
    if not usd_material:
        return None
    surface_output = usd_material.GetSurfaceOutput()
    source = surface_output.GetConnectedSource() if surface_output else None
    if not source:
        return None
    shader = UsdShade.Shader(source[0].GetPrim())
    if not shader:
        return None
    values = _shader_network_fingerprint_values(shader, visited=set())
    if values is None:
        return None
    if not values:
        return ""
    payload = repr(sorted(values)).encode("utf-8")
    return hashlib.blake2s(payload, digest_size=4).hexdigest()


def _shader_network_fingerprint_values(
    shader: Any, *, visited: set[str]
) -> list[tuple[str, str]] | None:
    """Recursively collect (name, value) pairs across a connected shader network.

    Directly authored (unconnected) input values are hashed as-is. Connected
    inputs (e.g. a texture reader feeding ``diffuseColor``) are resolved by
    recursing into the upstream shader so its authored values (such as a
    texture's ``file`` asset path) are incorporated too, instead of being
    silently dropped because the downstream input itself has no direct
    value. Returns ``None`` if any connection cannot be resolved to a
    ``UsdShade.Shader`` (e.g. it points into a node graph), since that
    means part of the network's appearance can't be inspected.
    """
    from pxr import UsdShade

    shader_path = str(shader.GetPath())
    if shader_path in visited:
        return []
    visited.add(shader_path)

    shader_id = shader.GetShaderId() or shader.GetPrim().GetTypeName()
    values: list[tuple[str, str]] = [("__shader__", str(shader_id))]
    for shader_input in shader.GetInputs():
        connection = shader_input.GetConnectedSource()
        if connection is not None:
            connected_shader = UsdShade.Shader(connection[0].GetPrim())
            if not connected_shader:
                return None
            nested = _shader_network_fingerprint_values(
                connected_shader, visited=visited
            )
            if nested is None:
                return None
            # Nest the upstream shader's fingerprint as a single value under
            # this specific input rather than flattening it into the parent
            # list. Flattening would let two networks that wire the same
            # upstream shaders to *different* inputs (e.g. swapping which
            # texture feeds diffuseColor vs emissiveColor) collapse to the
            # same fingerprint once the combined value set is sorted, even
            # though they look nothing alike. The upstream shader's absolute
            # path is deliberately excluded (see the identity docstring
            # above): only its own authored/nested values matter here.
            # The source output name and attribute type are included too,
            # since connecting the same upstream shader through a different
            # output port (e.g. a different color channel) can produce a
            # different appearance despite an identical nested fingerprint.
            source_output_name, source_attr_type = connection[1], connection[2]
            values.append(
                (
                    shader_input.GetBaseName(),
                    repr(
                        (str(source_output_name), str(source_attr_type), sorted(nested))
                    ),
                )
            )
            continue
        value = shader_input.Get()
        if value is not None:
            values.append((shader_input.GetBaseName(), repr(value)))
    return values


def _surface_identity_group_key(
    stage: Usd.Stage,
    root_path: str,
) -> tuple[tuple[str, int, str | None], ...]:
    """Return surface partitions used only to split ambiguous duplicate groups."""
    from pxr import Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return ()

    surfaces: list[tuple[str, int, str | None]] = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_count_known = bool(face_counts)
        face_count = len(face_counts or [])
        covered_faces: set[int] = set()
        material_subsets: list[tuple[Any, int]] = []
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)):
            indices = subset.GetIndicesAttr().Get() or []
            if indices:
                material_subsets.append((subset.GetPrim(), len(indices)))
                covered_faces.update(int(index) for index in indices)
        for subset_prim, subset_face_count in material_subsets:
            material, _relationship = UsdShade.MaterialBindingAPI(
                subset_prim
            ).ComputeBoundMaterial()
            surfaces.append(
                (
                    subset_prim.GetTypeName(),
                    subset_face_count,
                    _bound_material_identity(material, root_path)
                    or subset_prim.GetName(),
                )
            )
        if (
            not material_subsets
            or not face_count_known
            or len(covered_faces) < face_count
        ):
            material, _relationship = UsdShade.MaterialBindingAPI(
                prim
            ).ComputeBoundMaterial()
            surfaces.append(
                (
                    prim.GetTypeName(),
                    max(face_count - len(covered_faces), 0) if face_count_known else 0,
                    _bound_material_identity(material, root_path),
                )
            )
    return tuple(surfaces)


def _surface_identity_subgroup_suffix(
    subgroup_key: tuple[
        tuple[int, int, int],
        tuple[tuple[str, int, str | None], ...],
    ],
) -> str:
    """Return an insertion-order-independent suffix for ambiguous groups."""
    topology, surfaces = subgroup_key
    normalized = (
        topology,
        tuple((kind, count, material or "") for kind, count, material in surfaces),
    )
    payload = repr(normalized).encode("utf-8")
    return hashlib.blake2s(payload, digest_size=4).hexdigest()


def _subtree_has_mesh(prim: Usd.Prim) -> bool:
    """Return True if any descendant (or the prim itself) is a Mesh."""
    from pxr import Usd, UsdGeom

    return any(
        p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
    )


def _find_content_root(prim: Usd.Prim, max_depth: int = 5) -> Usd.Prim:
    """Descend through thin hierarchy nodes to find the actual content root.

    Many USD scenes wrap their content in a chain of thin containers
    (e.g. ``/Root/Root/HumanoidsDemo``). This helper walks down the
    hierarchy as long as the current node has very few children and the
    content is concentrated in a single child (not spread across siblings).

    Args:
        prim: Starting prim to descend from.
        max_depth: Maximum number of levels to descend.

    Returns:
        The deepest prim that looks like the content root.
    """
    from pxr import Usd

    if max_depth <= 0:
        return prim
    children = list(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    # Exclude material scopes (Looks, Materials) from content root search
    children = [c for c in children if c.GetName() not in _MATERIAL_SCOPE_NAMES]
    if len(children) > 5:
        return prim
    if not children:
        return prim

    # Count grandchildren for each child to gauge content distribution
    gc_counts = [
        (c, len(list(c.GetFilteredChildren(Usd.TraverseInstanceProxies()))))
        for c in children
    ]
    gc_counts.sort(key=lambda x: x[1], reverse=True)
    best_child, best_gc_count = gc_counts[0]

    # Stop if multiple children have significant content — the content is
    # spread across siblings rather than concentrated in one subtree.
    #
    # Returning the prim here can hand back the stage pseudo-root itself
    # (path "/") when there is no valid default prim. That used to be
    # unsafe: `detect_objects` built its path-prefix filter as
    # `scene_root_path + "/"`, which becomes "//" for the pseudo-root and
    # matches no real prim path, silently dropping every object in the
    # scene. `_content_root_prefix()` (used by `detect_objects` and
    # `_build_subtree_refs_cache`) now treats "/" as its own prefix, so
    # returning the pseudo-root here is safe -- and is exactly the right
    # answer when the pseudo-root has multiple significant top-level
    # assemblies, since descending into only the busiest one would drop
    # every object under the others instead.
    if len(gc_counts) > 1:
        _, second_gc_count = gc_counts[1]
        if second_gc_count >= 2 and second_gc_count >= best_gc_count * 0.3:
            return prim
        # Grandchild counts alone can miss thin sibling assemblies (a chain
        # of single-child wrappers still holding real geometry). Descending
        # past a sibling with mesh content would drop it from candidacy
        # entirely, so stop while meshes exist in more than one child.
        if sum(1 for c in children if _subtree_has_mesh(c)) > 1:
            return prim

    if best_gc_count >= len(children):
        return _find_content_root(best_child, max_depth - 1)
    return prim


# ------------------------------------------------------------------
# Phase 0 helpers: pre-computation caches
# ------------------------------------------------------------------


def _build_mesh_ancestry_cache(root_prim: Usd.Prim) -> set[str]:
    """Build a set of all prim paths that have at least one Mesh descendant.

    Walks every prim under *root_prim*; for each Mesh found, marks all
    ancestor paths up to (and including) the root.
    """
    from pxr import Usd

    paths_with_meshes: set[str] = set()
    root_path = str(root_prim.GetPath())
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if str(prim.GetTypeName()) == "Mesh":
            # Walk upward from the mesh's parent to root_path
            cur = prim.GetParent()
            while cur and cur.IsValid():
                cp = str(cur.GetPath())
                if cp in paths_with_meshes:
                    break  # ancestors already cached
                paths_with_meshes.add(cp)
                if cp == root_path:
                    break
                cur = cur.GetParent()
    return paths_with_meshes


def _content_root_prefix(root_path: str) -> str:
    """Return the prefix that marks a path as under *root_path*.

    A real content root like ``/World`` is matched via ``/World/``. The
    stage pseudo-root's own path *is* ``/``, so naively appending another
    ``/`` would produce ``//``, which prefix-matches no real prim path
    (every real path has exactly one leading ``/``) -- silently excluding
    every prim in the scene. `_find_content_root` can still return the
    pseudo-root through several of its return paths (e.g. a wide or
    balanced top-level layout with no default prim), so callers that turn
    a content root into a path-prefix filter must handle ``/`` itself.
    """
    return root_path if root_path == "/" else root_path + "/"


def _build_subtree_refs_cache(
    prim_refs: dict[str, list[str]], root_path: str
) -> dict[str, set[str]]:
    """Build a mapping of prim_path -> set of all sub-USD asset paths in subtree.

    Only considers prim paths that are under *root_path*.  The result is
    built bottom-up by sorting paths longest-first and propagating each
    prim's own refs to all its ancestors.
    """
    cache: dict[str, set[str]] = {}

    # Collect only refs under the content root
    relevant: list[tuple[str, list[str]]] = []
    prefix = _content_root_prefix(root_path)
    for p, refs in prim_refs.items():
        if p == root_path or p.startswith(prefix):
            relevant.append((p, refs))

    # Sort deepest first so children are processed before parents
    relevant.sort(key=lambda x: x[0].count("/"), reverse=True)

    for p, refs in relevant:
        own = set(refs)
        cache.setdefault(p, set()).update(own)
        # Propagate upward to each ancestor down to root_path
        parts = p.split("/")
        # Build ancestor paths from parent up to root. Joining an empty
        # leading part list (only possible when root_path is the
        # pseudo-root "/") yields "", which is not a valid path -- normalize
        # it back to "/" so the root_path sentinel below is actually reached.
        for i in range(len(parts) - 1, 0, -1):
            ancestor = "/".join(parts[:i]) or "/"
            if len(ancestor) < len(root_path):
                break  # pragma: no cover - relevant paths are constrained under root_path
            cache.setdefault(ancestor, set()).update(cache[p])
            if ancestor == root_path:
                break

    return cache


def _compute_sibling_homogeneity_map(
    parent_prim: Usd.Prim, prim_refs: dict[str, list[str]]
) -> dict[str, float]:
    """Compute sibling_homogeneity for all children of *parent_prim*.

    Groups siblings by their frozen direct-ref sets, then for each child
    returns the fraction of siblings sharing the same ref signature.
    """
    from pxr import Usd

    children = list(parent_prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    if not children:
        return {}

    # Map each child path to its frozen ref set
    ref_sigs: dict[str, frozenset[str]] = {}
    for child in children:
        cp = str(child.GetPath())
        refs = prim_refs.get(cp, [])
        ref_sigs[cp] = frozenset(refs)

    # Count how many siblings share each signature
    from collections import Counter

    sig_counts = Counter(ref_sigs.values())

    n_siblings = len(children)
    result: dict[str, float] = {}
    for child in children:
        cp = str(child.GetPath())
        sig = ref_sigs[cp]
        result[cp] = sig_counts[sig] / n_siblings
    return result


# ------------------------------------------------------------------
# Phase 2: CandidateFeatures dataclass
# ------------------------------------------------------------------


@dataclass
class CandidateFeatures:
    """Feature vector for a candidate prim."""

    path: str
    subtree_ref_diversity: int = 0
    max_subtree_reuse: int = 0
    direct_ref_reuse: int = 0
    sibling_homogeneity: float = 0.0
    child_count: int = 0
    child_type_diversity: int = 0
    has_skel_root: bool = False
    rel_depth: int = 0
    has_mesh_descendants: bool = False
    classification: str = "component"
    subtree_refs: set[str] = field(default_factory=set)


# ------------------------------------------------------------------
# Phase 3: Classification rules
# ------------------------------------------------------------------

_CLASSIFICATION_OBJECT_ROOT = "object_root"
_CLASSIFICATION_CATEGORY = "category"
_CLASSIFICATION_BUILDING_BLOCK = "building_block"
_CLASSIFICATION_COMPONENT = "component"


def _classify_candidate(feat: CandidateFeatures, threshold: int) -> str:
    """Apply priority-ordered classification rules to a candidate."""
    # Rule 1: building block (only prims that directly reference a
    # highly-reused asset — containers of building blocks are NOT excluded)
    if feat.subtree_ref_diversity <= 1 and feat.direct_ref_reuse >= threshold:
        return _CLASSIFICATION_BUILDING_BLOCK

    # Rule 2: depth-1 container (no direct refs → organizational grouping)
    if feat.rel_depth == 1 and feat.direct_ref_reuse == 0:
        return _CLASSIFICATION_CATEGORY

    # Rule 3: multi-asset assembly
    if feat.subtree_ref_diversity >= 2:
        return _CLASSIFICATION_OBJECT_ROOT

    # Rule 4: single-asset, low-reuse
    if feat.direct_ref_reuse > 0 and feat.direct_ref_reuse < threshold:
        return _CLASSIFICATION_OBJECT_ROOT

    # Rule 5: SkelRoot presence
    if feat.has_skel_root:
        return _CLASSIFICATION_OBJECT_ROOT

    # Rule 6: inline geometry (no sub-USD refs but has mesh descendants)
    if feat.subtree_ref_diversity == 0 and feat.has_mesh_descendants:
        return _CLASSIFICATION_OBJECT_ROOT

    # Rule 7: category node (many children, shallow depth, no direct refs)
    if feat.child_count >= 3 and feat.rel_depth <= 2 and feat.direct_ref_reuse == 0:
        return _CLASSIFICATION_CATEGORY

    # Default
    return _CLASSIFICATION_COMPONENT


# ------------------------------------------------------------------
# Phase 4: Non-overlap resolution
# ------------------------------------------------------------------


def _resolve_overlaps(
    classifications: dict[str, str],
) -> dict[str, str]:
    """Resolve overlapping object_root claims.

    Greedy shallowest-first claiming: sorts object_roots by depth then
    alphabetically.  An object_root is demoted to component if any
    ancestor is already claimed (by another object_root) or if any
    ancestor is a building_block.
    """
    resolved = dict(classifications)

    # Building blocks block their entire subtree
    blocked: set[str] = {
        p for p, c in resolved.items() if c == _CLASSIFICATION_BUILDING_BLOCK
    }

    object_roots = sorted(
        [p for p, c in resolved.items() if c == _CLASSIFICATION_OBJECT_ROOT],
        key=lambda p: (p.count("/"), p),
    )

    claimed: set[str] = set()
    for path in object_roots:
        inside_blocked = any(path.startswith(b + "/") for b in blocked)
        if inside_blocked:
            resolved[path] = _CLASSIFICATION_COMPONENT
            continue
        ancestor_claimed = any(path.startswith(c + "/") for c in claimed)
        if ancestor_claimed:
            resolved[path] = _CLASSIFICATION_COMPONENT
            continue
        claimed.add(path)

    return resolved


# ------------------------------------------------------------------
# Main detection function
# ------------------------------------------------------------------


def detect_objects(
    stage: Usd.Stage,
    composition_data: dict[str, Any],
    geometry_stats: dict[str, Any],
    skip_geometry: bool = False,
    building_block_min_reuse: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect objects in a USD scene using a feature-scoring algorithm.

    Implements a 6-phase detection strategy:
      0. Pre-computation -- build lookup caches
      1. Candidate selection -- walk hierarchy for viable prims
      2. Feature extraction -- compute feature vector per candidate
      3. Classification -- apply priority-ordered rules
      4. Non-overlap resolution -- resolve ancestor/descendant conflicts
      5. Instance grouping & source classification

    Args:
        stage: The USD stage to analyze.
        composition_data: Output of
            :func:`~world_understanding.utils.usd.composition.collect_composition_arcs`.
        geometry_stats: Output of
            :func:`~world_understanding.utils.usd.prim.collect_mesh_geometry_stats`.
        skip_geometry: If True, skip vertex/face counting in subtree stats.
        building_block_min_reuse: Minimum reuse count for building block
            classification.  The actual threshold is
            ``max(building_block_min_reuse, median_ref_count * 10)``.

    Returns:
        Tuple of ``(objects, instance_groups)`` where each object is a dict
        with keys: id, name, path, parent_group, source_classification,
        source_files, mesh_count, vertex_count, face_count,
        prim_type_breakdown, bounding_box, instance_group,
        llm_classification, llm_description.
    """
    from pxr import Pcp, Usd

    from world_understanding.utils.usd.prim import (
        get_bbox_from_prim,
        get_subtree_geometry_stats,
    )

    # ==================================================================
    # Setup
    # ==================================================================
    default_prim = stage.GetDefaultPrim()
    scene_root_prim = (
        default_prim
        if (default_prim and default_prim.IsValid())
        else stage.GetPseudoRoot()
    )
    scene_root_prim = _find_content_root(scene_root_prim)
    scene_root_path = str(scene_root_prim.GetPath())
    logger.info(f"Scene root detected: {scene_root_path}")

    # ==================================================================
    # Phase 0: Pre-computation
    # ==================================================================

    # prim_path -> [asset_paths]
    prim_refs: dict[str, list[str]] = {}
    for sub_usd in composition_data.get("sub_usd_files", []):
        for ref_prim in sub_usd.get("referencing_prims", []):
            prim_refs.setdefault(ref_prim, []).append(sub_usd["asset_path"])

    # asset_path -> global reference count
    asset_reuse_count: dict[str, int] = {}
    for sub_usd in composition_data.get("sub_usd_files", []):
        asset_reuse_count[sub_usd["asset_path"]] = sub_usd.get(
            "reference_count", len(sub_usd.get("referencing_prims", []))
        )

    # Building block threshold
    ref_counts = sorted(asset_reuse_count.values()) if asset_reuse_count else [0]
    median_ref = statistics.median(ref_counts) if ref_counts else 0
    bb_threshold = max(building_block_min_reuse, int(median_ref * 10))
    logger.info(
        f"Building block threshold: {bb_threshold} "
        f"(median_ref={median_ref}, min_reuse={building_block_min_reuse})"
    )

    # Mesh ancestry cache
    paths_with_meshes = _build_mesh_ancestry_cache(scene_root_prim)

    # Subtree refs cache
    subtree_refs_cache = _build_subtree_refs_cache(prim_refs, scene_root_path)

    # ==================================================================
    # Phase 1: Candidate Selection
    # ==================================================================
    candidates: dict[str, CandidateFeatures] = {}
    prefix = _content_root_prefix(scene_root_path)

    for prim in Usd.PrimRange(scene_root_prim, Usd.TraverseInstanceProxies()):
        prim_path = str(prim.GetPath())
        # Skip the content root itself
        if prim_path == scene_root_path:
            continue
        # Must be under content root
        if not prim_path.startswith(prefix):
            continue  # pragma: no cover - PrimRange(scene_root_prim) stays under root
        # Skip instance proxy descendants — instance roots (IsInstance)
        # stay as candidates but their proxy children don't expand into
        # separate candidates.  Prototypes are handled via Signal 0.
        if prim.IsInstanceProxy():
            continue
        # Must have at least one child (not a leaf)
        children = prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
        child_list = list(children)
        if not child_list:
            continue
        # Skip Material/Shader types
        type_name = str(prim.GetTypeName())
        if type_name in _SKIP_TYPES:
            continue
        # Must have mesh descendants OR sub-USD references in subtree
        has_meshes = prim_path in paths_with_meshes
        has_subtree_refs = bool(subtree_refs_cache.get(prim_path))
        if not has_meshes and not has_subtree_refs:
            continue

        candidates[prim_path] = CandidateFeatures(path=prim_path)

    logger.info(f"Phase 1: {len(candidates)} candidates selected")

    # ==================================================================
    # Phase 2: Feature Extraction
    # ==================================================================
    # Pre-compute sibling homogeneity per parent
    sibling_homo_cache: dict[str, float] = {}
    parents_computed: set[str] = set()

    for path in candidates:
        prim = stage.GetPrimAtPath(path)
        parent = prim.GetParent()
        if not parent or not parent.IsValid():
            continue  # pragma: no cover - selected candidates always have valid parents
        parent_path = str(parent.GetPath())
        if parent_path not in parents_computed:
            parents_computed.add(parent_path)
            homo_map = _compute_sibling_homogeneity_map(parent, prim_refs)
            sibling_homo_cache.update(homo_map)

    for path, feat in candidates.items():
        prim = stage.GetPrimAtPath(path)

        # Relative depth
        # Sliced by `len(prefix)` rather than `len(scene_root_path) + 1` so
        # this stays correct when scene_root_path is the pseudo-root "/"
        # (prefix "/" itself, not scene_root_path + "/").
        rel_part = path[len(prefix) :]
        feat.rel_depth = rel_part.count("/") + 1

        # Subtree refs
        feat.subtree_refs = subtree_refs_cache.get(path, set())
        feat.subtree_ref_diversity = len(feat.subtree_refs)

        # Max subtree reuse
        if feat.subtree_refs:
            feat.max_subtree_reuse = max(
                asset_reuse_count.get(a, 0) for a in feat.subtree_refs
            )

        # Direct ref reuse
        direct_refs = prim_refs.get(path, [])
        if direct_refs:
            feat.direct_ref_reuse = max(
                asset_reuse_count.get(a, 0) for a in direct_refs
            )

        # Sibling homogeneity
        feat.sibling_homogeneity = sibling_homo_cache.get(path, 0.0)

        # Child count and type diversity
        child_list = list(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        feat.child_count = len(child_list)
        child_types = {str(c.GetTypeName()) for c in child_list}
        feat.child_type_diversity = len(child_types)

        # SkelRoot among direct children
        feat.has_skel_root = any(str(c.GetTypeName()) == "SkelRoot" for c in child_list)

        # Mesh descendants
        feat.has_mesh_descendants = path in paths_with_meshes

    logger.info("Phase 2: features extracted")

    # ==================================================================
    # Phase 3: Classification
    # ==================================================================
    for _path, feat in candidates.items():
        feat.classification = _classify_candidate(feat, bb_threshold)

    classifications = {p: f.classification for p, f in candidates.items()}

    obj_count = sum(
        1 for c in classifications.values() if c == _CLASSIFICATION_OBJECT_ROOT
    )
    cat_count = sum(
        1 for c in classifications.values() if c == _CLASSIFICATION_CATEGORY
    )
    bb_count = sum(
        1 for c in classifications.values() if c == _CLASSIFICATION_BUILDING_BLOCK
    )
    logger.info(
        f"Phase 3: {obj_count} object_roots, {cat_count} categories, "
        f"{bb_count} building_blocks (pre-overlap)"
    )

    # ==================================================================
    # Phase 4: Non-overlap Resolution
    # ==================================================================
    resolved = _resolve_overlaps(classifications)

    # Leaf-category promotion (post-overlap): if a category has no
    # remaining descendant object_roots, promote it to object_root so it
    # appears as an object for material assignment.
    for path in list(resolved):
        if resolved[path] != _CLASSIFICATION_CATEGORY:
            continue
        cat_prefix = path + "/"
        has_descendant_obj = any(
            c == _CLASSIFICATION_OBJECT_ROOT
            for p, c in resolved.items()
            if p.startswith(cat_prefix)
        )
        if not has_descendant_obj:
            resolved[path] = _CLASSIFICATION_OBJECT_ROOT

    obj_count = sum(1 for c in resolved.values() if c == _CLASSIFICATION_OBJECT_ROOT)
    cat_count = sum(1 for c in resolved.values() if c == _CLASSIFICATION_CATEGORY)
    logger.info(
        f"Phase 4: {obj_count} object_roots, {cat_count} categories (post-overlap)"
    )

    # ==================================================================
    # Phase 5 (part 1): Object Assembly
    # ==================================================================
    # Only compute expensive geometry stats for final object_roots
    final_roots = sorted(
        [p for p, c in resolved.items() if c == _CLASSIFICATION_OBJECT_ROOT],
        key=lambda p: (p.count("/"), p),
    )

    obj_counter = 0

    def _next_id() -> str:
        nonlocal obj_counter
        obj_counter += 1
        return f"obj_{obj_counter:03d}"

    def _get_parent_group(path: str) -> str | None:
        if scene_root_path and path.startswith(prefix):
            rel = path[len(prefix) :]
        else:
            rel = path  # pragma: no cover - final roots are selected under scene_root_path
        parts = [p for p in rel.split("/") if p]
        if len(parts) >= 2:
            return parts[0]
        return None

    def _compute_bbox_dict(path: str) -> dict[str, Any] | None:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return (
                None  # pragma: no cover - final roots come from valid traversed prims
            )
        try:
            bbox = get_bbox_from_prim(prim)
            rng = bbox.ComputeAlignedRange()
            mn = rng.GetMin()
            mx = rng.GetMax()
            return {"min": [mn[0], mn[1], mn[2]], "max": [mx[0], mx[1], mx[2]]}
        except Exception:
            return None

    def _collect_subtree_refs_sorted(root_path: str) -> list[str]:
        return sorted(subtree_refs_cache.get(root_path, set()))

    objects: list[dict[str, Any]] = []
    for path in final_roots:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue  # pragma: no cover - final roots come from valid traversed prims
        stats = get_subtree_geometry_stats(stage, path, skip_geometry=skip_geometry)
        source_files = _collect_subtree_refs_sorted(path)
        bbox = _compute_bbox_dict(path)
        parent_group = _get_parent_group(path)

        obj: dict[str, Any] = {
            "id": _next_id(),
            "name": prim.GetName(),
            "path": path,
            "parent_group": parent_group,
            "source_classification": None,
            "source_files": source_files,
            "mesh_count": stats["mesh_count"],
            "vertex_count": stats["vertex_count"],
            "face_count": stats["face_count"],
            "prim_type_breakdown": stats["prim_type_breakdown"],
            "bounding_box": bbox,
            "instance_group": None,
            "llm_classification": None,
            "llm_description": None,
        }
        objects.append(obj)

    logger.info(f"Phase 5: {len(objects)} objects assembled")

    # ==================================================================
    # Phase 5 (part 2): Instance Grouping & Source Classification
    # ==================================================================
    instance_groups: list[dict[str, Any]] = []
    assigned: set[str] = set()  # object paths already in a group

    # Signal 0: USD native prototype grouping (strongest signal)
    # Instances sharing the same prototype have identical geometry and
    # need the same materials — group them before any other signal.
    proto_groups: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        prim = stage.GetPrimAtPath(obj["path"])
        if prim and prim.IsValid() and prim.IsInstance():
            proto = prim.GetPrototype()
            if proto and proto.IsValid():
                proto_path = str(proto.GetPath())
                proto_groups.setdefault(proto_path, []).append(obj)

    for proto_path, members in sorted(proto_groups.items()):
        members = sorted(members, key=lambda member: member["path"])
        if len(members) < 2:
            continue
        group_name = members[0]["name"]
        ig: dict[str, Any] = {
            "group_name": group_name,
            "source_file": proto_path,
            "instance_count": len(members),
            "member_paths": [m["path"] for m in members],
        }
        instance_groups.append(ig)
        for m in members:
            m["instance_group"] = group_name
            assigned.add(m["path"])

    # Signal 1: Same direct sub-USD (strongest)
    source_groups: dict[str, dict[tuple[int, int, int], list[dict[str, Any]]]] = {}
    for obj in objects:
        if len(obj["source_files"]) == 1:
            sf = obj["source_files"][0]
            topology = _topology_group_key(obj)
            source_groups.setdefault(sf, {}).setdefault(topology, []).append(obj)
    for sf, topology_groups in sorted(source_groups.items()):
        for topology, members in sorted(topology_groups.items()):
            members = sorted(members, key=lambda member: member["path"])
            if len(members) < 2:
                continue
            group_name = Path(sf).stem
            if len(topology_groups) > 1:
                mc, vc, fc = topology
                group_name = f"{group_name}_{mc}m_{vc}v_{fc}f"
            ig = {
                "group_name": group_name,
                "source_file": sf,
                "instance_count": len(members),
                "member_paths": [m["path"] for m in members],
            }
            instance_groups.append(ig)
            for m in members:
                m["instance_group"] = group_name
                assigned.add(m["path"])

    # Signal 1b: Reference-source duplicate detection via PcpPrimIndex
    # Catches prims that reference the same USD file but whose source_files
    # lists diverged (e.g. collected at different layer levels).
    def _get_reference_source(prim: Usd.Prim) -> str | None:
        """Extract the primary reference source file from a prim's PcpPrimIndex."""
        try:
            prim_index = prim.GetPrimIndex()
            root_node = prim_index.rootNode
            for child in root_node.children:
                if child.arcType == Pcp.ArcTypeReference:
                    layer = (
                        child.layerStack.layers[0] if child.layerStack.layers else None
                    )
                    if layer:
                        identifier = layer.identifier
                        if identifier and not identifier.startswith("anon:"):
                            return identifier
        except Exception:  # pragma: no cover - defensive Pcp API guard
            pass
        return None

    ref_source_groups: dict[tuple[str, tuple[int, int, int]], list[dict[str, Any]]] = {}
    for obj in objects:
        if obj["path"] in assigned:
            continue
        prim = stage.GetPrimAtPath(obj["path"])
        if not prim or not prim.IsValid():
            continue  # pragma: no cover - objects are assembled from valid prims
        ref_src = _get_reference_source(prim)
        if ref_src:
            key = (ref_src, _topology_group_key(obj))
            ref_source_groups.setdefault(key, []).append(obj)

    for (ref_src, _topology), members in ref_source_groups.items():
        if len(members) < 2:
            continue
        group_name = Path(ref_src).stem
        ig = {
            "group_name": group_name,
            "source_file": ref_src,
            "instance_count": len(members),
            "member_paths": [m["path"] for m in members],
        }
        instance_groups.append(ig)
        for m in members:
            m["instance_group"] = group_name
            assigned.add(m["path"])

    # Signal 1b fallback: same name + exact topology match.
    # Catches objects referencing different source files that contain
    # identical geometry (e.g. fixture.usd vs fixture_loaded.usd).
    name_topo_groups: dict[tuple[str, tuple[int, int, int]], list[dict[str, Any]]] = {}
    for obj in objects:
        if obj["path"] in assigned:
            continue
        if obj["mesh_count"] > 0:
            key = (obj["name"], _topology_group_key(obj))
            name_topo_groups.setdefault(key, []).append(obj)

    for (name, _topology), members in name_topo_groups.items():
        if len(members) < 2:
            continue
        source = members[0]["source_files"]
        ig = {
            "group_name": name,
            "source_file": source[0]
            if len(source) == 1
            else (source if source else None),
            "instance_count": len(members),
            "member_paths": [m["path"] for m in members],
        }
        instance_groups.append(ig)
        for m in members:
            m["instance_group"] = name
            assigned.add(m["path"])

    # Signal 2: Name pattern
    name_groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for obj in objects:
        if obj["path"] in assigned:
            continue
        base = _instance_base_name(obj["name"])
        if base is not None:
            name_groups.setdefault(
                _name_pattern_group_key(obj["path"], base), []
            ).append(obj)

    for (base_name, context_name), members in sorted(name_groups.items()):
        members = sorted(members, key=lambda member: member["path"])
        if len(members) < 2:
            continue
        # Sub-group by mesh, vertex, and face counts to avoid grouping prims with
        # different topology. This is especially important for generic names
        # such as node1480/node1534, where the semantic parent disambiguates
        # product type but geometry still needs to match.
        mc_subgroups: dict[
            tuple[tuple[int, int, int], tuple[tuple[str, int, str | None], ...]],
            list[dict[str, Any]],
        ] = {}
        for m in members:
            key = (
                _topology_group_key(m),
                _surface_identity_group_key(stage, m["path"]),
            )
            mc_subgroups.setdefault(key, []).append(m)
        for subgroup_key, sub_members in sorted(
            mc_subgroups.items(),
            key=lambda item: _surface_identity_subgroup_suffix(item[0]),
        ):
            (mc, vc, fc), _surface_key = subgroup_key
            if len(sub_members) < 2:
                continue
            group_base = f"{context_name}_{base_name}" if context_name else base_name
            group_name = (
                group_base
                if len(mc_subgroups) == 1
                else (
                    f"{group_base}_{mc}m_{vc}v_{fc}f_"
                    f"{_surface_identity_subgroup_suffix(subgroup_key)}"
                )
            )
            sub_members = sorted(sub_members, key=lambda member: member["path"])
            source = sub_members[0]["source_files"]
            ig = {
                "group_name": group_name,
                "source_file": source[0]
                if len(source) == 1
                else (source if source else None),
                "instance_count": len(sub_members),
                "member_paths": [m["path"] for m in sub_members],
            }
            instance_groups.append(ig)
            for m in sub_members:
                m["instance_group"] = group_name
                assigned.add(m["path"])

    # Signal 3: Subtree reference fingerprint
    fingerprint_groups: dict[frozenset[str], list[dict[str, Any]]] = {}
    for obj in objects:
        if obj["path"] in assigned:
            continue
        sf = frozenset(obj["source_files"])
        if sf:  # skip objects with no source files
            fingerprint_groups.setdefault(sf, []).append(obj)

    for sf_set, members in sorted(
        fingerprint_groups.items(),
        key=lambda item: tuple(sorted(item[0])),
    ):
        members = sorted(members, key=lambda member: member["path"])
        if len(members) < 2:
            continue
        # Sub-group by topology to avoid grouping prims with different geometry.
        mc_subgroups_fp: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for m in members:
            mc_subgroups_fp.setdefault(_topology_group_key(m), []).append(m)
        for (mc, _vc, _fc), sub_members in sorted(mc_subgroups_fp.items()):
            sub_members = sorted(sub_members, key=lambda member: member["path"])
            if len(sub_members) < 2:
                continue
            # Use shortest common stem as group name
            stems = [Path(f).stem for f in sorted(sf_set)]
            group_name = stems[0] if stems else "unnamed_group"
            if len(mc_subgroups_fp) > 1:
                group_name = f"{group_name}_{mc}m"
            ig = {
                "group_name": group_name,
                "source_file": sorted(sf_set)[0]
                if len(sf_set) == 1
                else sorted(sf_set),
                "instance_count": len(sub_members),
                "member_paths": [m["path"] for m in sub_members],
            }
            instance_groups.append(ig)
            for m in sub_members:
                m["instance_group"] = group_name
                assigned.add(m["path"])

    # Source classification
    for obj in objects:
        n_source_files = len(obj["source_files"])
        if n_source_files == 0:
            obj["source_classification"] = "INLINE"
        elif n_source_files == 1:
            obj["source_classification"] = "FILE"
        else:
            obj["source_classification"] = "MIXED"

    return objects, instance_groups
