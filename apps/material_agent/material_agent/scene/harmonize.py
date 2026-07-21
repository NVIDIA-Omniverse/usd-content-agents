# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harmonize material predictions via multi-signal grouping.

Groups prims that likely represent the same physical part using three signals:
1. **Geometry fingerprint** – (vertex_count, face_count, bbox_dimensions) from USD
2. **Name template** – prim path with trailing digits stripped (Column2 → Column{})
3. **Part-number signature** – rare, CAD-like path segments (existing logic)

When grouped prims received different material predictions, ALL reasonings are
sent to an LLM which decides whether to **unify** (same material for all) or
**keep** (leave different materials as-is).

Works at both **asset level** (within a single sub-asset) and **scene level**
(across all sub-assets after their pipelines complete).
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DIR_FD_ATOMIC_WRITE_SUPPORTED = os.name != "nt"

# ---------------------------------------------------------------------------
# Signature helpers (legacy, kept as one of three grouping signals)
# ---------------------------------------------------------------------------

_INSTANCE_SUFFIX_RE = re.compile(r"_U20__U28_\d+_U29_$")
# Plain trailing instance index: _1, _2, ..., _999 at end of segment.
_PLAIN_INSTANCE_SUFFIX_RE = re.compile(r"_(\d{1,3})$")
_LONG_DIGITS_RE = re.compile(r"\d{4,}")
_BLOCK_SEP_RE = re.compile(r"__+|_U2D_")


def normalize_segment(segment: str) -> str:
    """Strip encoded instance-index suffix (``_U20__U28_N_U29_``)."""
    return _INSTANCE_SUFFIX_RE.sub("", segment)


def _normalize_segment_for_signature(
    segment: str,
    known_bases: set[str] | None = None,
) -> str:
    """Normalize a segment for signature computation.

    Strips both encoded and plain instance-index suffixes.  Plain suffixes
    (``_1``, ``_7``, ``_12``) are only stripped when the resulting base is
    present in *known_bases* — i.e., the segment without ``_N`` also appears
    as a prim-path segment elsewhere in the scene.  This prevents stripping
    digits that are part of a CAD part number (e.g.,
    ``T348XX_GRILL_REV00_0708_3`` keeps ``_3`` because
    ``T348XX_GRILL_REV00_0708`` never appears as a standalone segment).

    Examples::

        GB300_DGX_FULL_ASM_0412_1  →  GB300_DGX_FULL_ASM_0412  (base exists)
        GB300_DGX_PSU_ASM_1_3      →  GB300_DGX_PSU_ASM_1      (base exists)
        T348XX_GRILL_REV00_0708_3  →  T348XX_GRILL_REV00_0708_3 (base absent)
    """
    result = _INSTANCE_SUFFIX_RE.sub("", segment)
    if known_bases is not None:
        m = _PLAIN_INSTANCE_SUFFIX_RE.search(result)
        if m:
            base = result[: m.start()]
            if base in known_bases:
                result = base
    return result


def is_part_number(segment: str) -> bool:
    """Heuristic: does *segment* look like a CAD part number?"""
    if len(segment) < 10:
        return False
    if _LONG_DIGITS_RE.search(segment):
        return True
    blocks = _BLOCK_SEP_RE.split(segment)
    digit_blocks = sum(1 for b in blocks if re.search(r"\d", b))
    alpha_blocks = sum(1 for b in blocks if re.search(r"[A-Za-z]", b))
    return digit_blocks >= 2 and alpha_blocks >= 1


# ---------------------------------------------------------------------------
# Signal 1: Signature-based grouping (existing logic)
# ---------------------------------------------------------------------------


def _collect_known_bases(prim_paths: list[str]) -> set[str]:
    """Build the set of all raw segment names across all paths.

    Used by :func:`_normalize_segment_for_signature` to decide whether a
    trailing ``_N`` is an instance index (base exists without ``_N``) or
    part of a CAD part number (base never appears alone).
    """
    bases: set[str] = set()
    for path in prim_paths:
        for seg in path.strip("/").split("/"):
            # Store both raw and encoded-instance-stripped forms
            bases.add(seg)
            bases.add(normalize_segment(seg))
    return bases


def build_segment_frequency(
    prim_paths: list[str],
    known_bases: set[str] | None = None,
) -> Counter[str]:
    """Count normalised segment frequency across all prim paths.

    When *known_bases* is provided, plain instance suffixes (``_1``,
    ``_7``) are also stripped during normalisation — see
    :func:`_normalize_segment_for_signature`.
    """
    counter: Counter[str] = Counter()
    for path in prim_paths:
        segments = path.strip("/").split("/")
        counter.update(
            {_normalize_segment_for_signature(s, known_bases) for s in segments}
        )
    return counter


def compute_signature(
    prim_path: str,
    freq: Counter[str],
    total_prims: int,
    max_freq_ratio: float = 0.05,
    known_bases: set[str] | None = None,
) -> tuple[str, ...]:
    """Ordered tuple of rare, part-number-like segments + leaf mesh index.

    When *known_bases* is provided, plain instance suffixes are normalised
    so that ``GB300_DGX_FULL_ASM_0412_1`` and ``GB300_DGX_FULL_ASM_0412``
    produce the same signature component.
    """
    segments = prim_path.strip("/").split("/")
    rare_parts: list[str] = []
    leaf_mesh_idx: str | None = None

    for seg in segments:
        norm = _normalize_segment_for_signature(seg, known_bases)
        if norm.startswith("mesh_U5B_"):
            leaf_mesh_idx = norm
            continue
        if norm == "mesh" or norm.startswith("mesh_I"):
            continue
        ratio = freq.get(norm, total_prims) / total_prims
        if ratio < max_freq_ratio and is_part_number(norm):
            rare_parts.append(norm)

    if leaf_mesh_idx:
        rare_parts.append(leaf_mesh_idx)
    return tuple(rare_parts)


def _group_by_signature(
    predictions: list[dict[str, Any]],
    min_signature_len: int = 2,
    max_freq_ratio: float = 0.05,
) -> dict[str, list[int]]:
    """Group prediction indices by part-number signature.

    Returns dict of group_key → list of prediction indices.
    """
    prim_paths = [p["id"] for p in predictions]
    known_bases = _collect_known_bases(prim_paths)
    freq = build_segment_frequency(prim_paths, known_bases)
    total = len(predictions)

    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, pred in enumerate(predictions):
        sig = compute_signature(pred["id"], freq, total, max_freq_ratio, known_bases)
        if len(sig) >= min_signature_len:
            groups[sig].append(idx)

    # Return only multi-member groups, keyed by string for merging
    return {f"sig:{'/'.join(k)}": v for k, v in groups.items() if len(v) > 1}


# ---------------------------------------------------------------------------
# Signal 2: Name template grouping
# ---------------------------------------------------------------------------


def _name_template(prim_path: str) -> str:
    """Strip trailing digits from each path segment to build a template.

    Example: /Root/CraneBase/Column2/shape/mesh → /Root/CraneBase/Column{}/shape/mesh
    """
    segments = prim_path.strip("/").split("/")
    result = []
    for seg in segments:
        # Replace trailing digits with {}
        digit_start = len(seg)
        while digit_start > 0 and seg[digit_start - 1].isdigit():
            digit_start -= 1
        stripped = seg[:digit_start] + "{}" if digit_start < len(seg) else seg
        result.append(stripped)
    return "/".join(result)


def _group_by_name_template(
    predictions: list[dict[str, Any]],
) -> dict[str, list[int]]:
    """Group prediction indices by name template.

    Only returns groups where the template differs from the original
    (i.e., at least one segment had trailing digits stripped) AND
    the group has 2+ members.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, pred in enumerate(predictions):
        prim_path = pred["id"]
        template = _name_template(prim_path)
        # Only group if template is different from original (digits were stripped)
        if template != prim_path.strip("/"):
            groups[f"name:{template}"].append(idx)

    return {k: v for k, v in groups.items() if len(v) > 1}


# ---------------------------------------------------------------------------
# Signal 3: Geometry fingerprint grouping
# ---------------------------------------------------------------------------


def _compute_geometry_fingerprints(
    predictions: list[dict[str, Any]],
    optimized_usd_path: str | None,
) -> dict[str, list[int]]:
    """Group prediction indices by geometry fingerprint from USD stage.

    Fingerprint = (vertex_count, face_count, rounded_bbox_dimensions).
    Returns dict of group_key → list of prediction indices.
    """
    prim_fingerprints = _compute_geometry_fingerprint_map(
        predictions, optimized_usd_path
    )

    # Group by fingerprint
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, pred in enumerate(predictions):
        fp = prim_fingerprints.get(pred["id"])
        if fp:
            groups[f"geo:{fp}"].append(idx)

    return {k: v for k, v in groups.items() if len(v) > 1}


def _compute_geometry_fingerprint_map(
    predictions: list[dict[str, Any]],
    optimized_usd_path: str | None,
) -> dict[str, str]:
    """Return prediction prim path → geometry fingerprint from a USD stage."""
    if not optimized_usd_path:
        return {}

    try:
        from pxr import Usd
    except ImportError:
        logger.debug("pxr not available — skipping geometry fingerprinting")
        return {}

    usd_path = Path(optimized_usd_path)
    if not usd_path.exists():
        logger.warning("Optimized USD not found: %s", usd_path)
        return {}

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        logger.warning("Failed to open USD stage: %s", usd_path)
        return {}

    prim_fingerprints: dict[str, str] = {}
    for pred in predictions:
        prim_path = pred["id"]
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            # Try parent (prediction might be on a mesh child)
            parent_path = "/".join(prim_path.rsplit("/", 1)[:-1])
            prim = stage.GetPrimAtPath(parent_path) if parent_path else None

        if not prim or not prim.IsValid():
            continue

        fp = _fingerprint_prim(prim)
        if fp:
            prim_fingerprints[prim_path] = fp
    return prim_fingerprints


def _constrain_name_groups_by_geometry(
    name_groups: dict[str, list[int]],
    predictions: list[dict[str, Any]],
    geometry_by_path: dict[str, str],
) -> dict[str, list[int]]:
    """Split name-template groups by geometry when fingerprints are available.

    Generic CAD node names such as ``node123`` and ``mesh456`` can otherwise
    group unlike siblings, for example a tabletop with workbench legs.  If a
    geometry fingerprint is available for any group member, keep only
    same-fingerprint subgroups from that name-template group.  When no geometry
    data is available, retain the original behavior.
    """
    if not geometry_by_path:
        return name_groups

    constrained: dict[str, list[int]] = {}
    for group_name, members in name_groups.items():
        fp_groups: dict[str, list[int]] = defaultdict(list)
        missing: list[int] = []
        for idx in members:
            fp = geometry_by_path.get(predictions[idx]["id"])
            if fp:
                fp_groups[fp].append(idx)
            else:
                missing.append(idx)

        if len(missing) == len(members):
            constrained[group_name] = members
            continue

        if missing:
            logger.warning(
                "Name-template group %s has %d member(s) without geometry "
                "fingerprints; keeping missing-fingerprint members in a "
                "separate subgroup.",
                group_name,
                len(missing),
            )
            constrained[f"{group_name}|geo:missing"] = missing

        for fp, fp_members in fp_groups.items():
            if len(fp_members) > 1:
                constrained[f"{group_name}|geo:{fp}"] = fp_members

    return constrained


def _fingerprint_prim(prim: Any) -> str | None:
    """Compute a geometry fingerprint for a prim.

    Looks at the prim and its mesh children to build a fingerprint
    from vertex count, face count, and bounding box dimensions.
    """
    from pxr import UsdGeom

    # Collect mesh stats from this prim or its children
    total_verts = 0
    total_faces = 0
    meshes_found = 0

    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return None

    # Check if this prim itself is a mesh
    mesh = UsdGeom.Mesh(prim)
    if mesh:
        verts = mesh.GetPointsAttr().Get()
        faces = mesh.GetFaceVertexCountsAttr().Get()
        if verts:
            total_verts += len(verts)
        if faces:
            total_faces += len(faces)
        meshes_found += 1
    else:
        # Check immediate children for meshes
        for child in prim.GetChildren():
            mesh = UsdGeom.Mesh(child)
            if mesh:
                verts = mesh.GetPointsAttr().Get()
                faces = mesh.GetFaceVertexCountsAttr().Get()
                if verts:
                    total_verts += len(verts)
                if faces:
                    total_faces += len(faces)
                meshes_found += 1

    if meshes_found == 0:
        return None

    # Get bounding box dimensions (rounded to avoid floating point noise)
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.ComputeAlignedRange()
    if bbox_range.IsEmpty():
        dims = (0, 0, 0)
    else:
        size = bbox_range.GetSize()
        # Round to 2 decimals to group nearly-identical geometry
        dims = (round(size[0], 2), round(size[1], 2), round(size[2], 2))

    return f"{total_verts}_{total_faces}_{dims[0]}x{dims[1]}x{dims[2]}"


# ---------------------------------------------------------------------------
# Unified grouping: merge all three signals via union-find
# ---------------------------------------------------------------------------


class _UnionFind:
    """Simple union-find for merging overlapping groups."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def _merge_groups(
    all_signal_groups: list[dict[str, list[int]]],
    total_predictions: int,
) -> dict[int, list[int]]:
    """Merge groups from multiple signals using union-find.

    If prediction i and j appear in the same group from ANY signal,
    they end up in the same merged group.

    Returns dict of representative_index → list of member indices.
    """
    uf = _UnionFind(total_predictions)

    for signal_groups in all_signal_groups:
        for members in signal_groups.values():
            if len(members) < 2:
                continue
            first = members[0]
            for other in members[1:]:
                uf.union(first, other)

    # Collect merged groups
    merged: dict[int, list[int]] = defaultdict(list)
    for i in range(total_predictions):
        root = uf.find(i)
        merged[root].append(i)

    # Only return groups with 2+ members
    return {k: v for k, v in merged.items() if len(v) > 1}


# ---------------------------------------------------------------------------
# Conflict detection & LLM resolution
# ---------------------------------------------------------------------------


def _find_conflicts(
    groups: dict[int, list[int]],
    predictions: list[dict[str, Any]],
) -> dict[int, list[int]]:
    """Return only groups where members disagree on the material."""
    conflicts = {}
    for rep, members in groups.items():
        materials = set()
        for i in members:
            mat = predictions[i].get("materials", {}).get("material")
            if mat is not None:
                materials.add(mat)
        if len(materials) > 1:
            conflicts[rep] = members
    return conflicts


def _build_harmonize_prompt(
    members: list[dict[str, Any]],
    group_signals: list[str],
) -> str:
    """Build the LLM prompt for a conflict group.

    The LLM reads ALL reasonings and decides whether to unify or keep.
    """
    # Collect all unique materials with their reasonings
    mat_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in members:
        mat = m["materials"]["material"]
        mat_entries[mat].append(m)

    options_text = []
    for mat, entries in mat_entries.items():
        # Gather all reasonings for this material
        reasonings = []
        for entry in entries:
            reasoning = entry["materials"].get("original_response", "")
            r_match = re.search(r"<reasoning>(.*?)</reasoning>", reasoning, re.DOTALL)
            r_text = r_match.group(1).strip() if r_match else reasoning[:500]
            if r_text:
                reasonings.append(r_text)

        prim_paths = [e["id"] for e in entries]
        options_text.append(
            f'Material: "{mat}" ({len(entries)} prims)\n'
            f"Prim paths: {prim_paths[:5]}{'...' if len(prim_paths) > 5 else ''}\n"
            f"Reasonings:\n"
            + "\n---\n".join(reasonings[:3])  # cap at 3 to avoid token explosion
        )

    signals_text = ", ".join(group_signals)

    return (
        "You are reviewing material predictions for a group of 3D mesh prims "
        "that were identified as likely representing the same physical part. "
        f"They were grouped by these signals: {signals_text}.\n\n"
        "Each material option below includes the prim paths and the VLM's "
        "reasoning for that prediction. Review ALL reasonings carefully.\n\n"
        "DECIDE:\n"
        "- If these prims truly represent the same physical part and should "
        "share ONE material, respond with: "
        '{"action": "unify", "material": "<best material name>", '
        '"reason": "<brief justification>"}\n'
        "- If the prims represent different parts that legitimately need "
        "different materials, respond with: "
        '{"action": "keep", "reason": "<brief justification>"}\n\n'
        "When choosing to unify, pick the material with the most specific "
        "and concrete visual evidence in its reasoning.\n\n"
        + "\n\n===\n\n".join(options_text)
        + "\n\nRespond with JSON only."
    )


def _resolve_single_group(
    member_indices: list[int],
    predictions: list[dict[str, Any]],
    group_signals: list[str],
    llm: Any,
    model_name: str | None = None,
    token_tracker: Any | None = None,
) -> dict[str, str]:
    """Resolve a single conflict group via LLM. Thread-safe.

    Returns prim_path → material remap (only for prims that need changing).
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from world_understanding.utils.llm_parsing import (
        extract_json_from_llm_response,
    )

    members = [predictions[i] for i in member_indices]
    prompt = _build_harmonize_prompt(members, group_signals)
    remap: dict[str, str] = {}

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an expert at industrial material identification. "
                        "You review grouped mesh predictions and decide whether "
                        "they should be unified or kept as-is."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        from .stats import record_model_response_usage

        record_model_response_usage(
            token_tracker,
            response,
            model_name,
            "scene_harmonize_llm",
        )
        result = extract_json_from_llm_response(response.content)
        if not result or not isinstance(result, dict):
            logger.warning("LLM parse failed for group: %s", response.content[:200])
            return remap

        action = result.get("action", "").lower()
        reason = result.get("reason", "")

        if action == "unify":
            best_mat = result.get("material", "")
            if best_mat:
                for m in members:
                    if m["materials"]["material"] != best_mat:
                        remap[m["id"]] = best_mat
                logger.info(
                    "LLM UNIFY (%d prims): → %s (%s)",
                    len(members),
                    best_mat,
                    reason[:80],
                )
            else:
                logger.warning("LLM chose unify but returned empty material")
        elif action == "keep":
            logger.info(
                "LLM KEEP (%d prims, %d materials): %s",
                len(members),
                len({m["materials"]["material"] for m in members}),
                reason[:80],
            )
        else:
            logger.warning("LLM returned unknown action: %s", action)

    except Exception:
        logger.exception("LLM resolution failed for group")

    return remap


def _resolve_conflicts(
    conflicts: dict[int, list[int]],
    predictions: list[dict[str, Any]],
    group_signals_map: dict[int, list[str]],
    llm_config: dict[str, Any] | None = None,
    mode: str = "full",
    token_tracker: Any | None = None,
) -> dict[str, str]:
    """Resolve all conflict groups, optionally using LLM.

    Args:
        conflicts: Map of representative index → member indices.
        predictions: Full prediction list.
        group_signals_map: Map of representative → signal names.
        llm_config: LLM configuration dict (only used in ``"full"`` mode).
        mode: ``"full"`` (LLM resolution with majority-vote fallback) or
            ``"simple"`` (majority vote only, no LLM call).
    """
    if not conflicts:
        return {}

    if mode == "simple":
        logger.info(
            "Resolving %d conflict groups via majority vote (simple mode)",
            len(conflicts),
        )
        return _majority_vote_fallback(conflicts, predictions)

    if not llm_config:
        logger.warning(
            "%d conflict groups need LLM but no llm_config — falling back to majority vote",
            len(conflicts),
        )
        return _majority_vote_fallback(conflicts, predictions)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from world_understanding.functions.models.chat_models import (
        create_chat_model_from_config,
    )

    llm = create_chat_model_from_config(llm_config, defaults={"max_tokens": 2048})
    if llm is None:
        logger.warning(
            "%d conflict groups need LLM but no API key — falling back to majority vote",
            len(conflicts),
        )
        return _majority_vote_fallback(conflicts, predictions)

    if len(conflicts) == 1:
        rep, members = next(iter(conflicts.items()))
        signals = group_signals_map.get(rep, [])
        return _resolve_single_group(
            members,
            predictions,
            signals,
            llm,
            llm_config.get("model"),
            token_tracker,
        )

    configured_max_workers = llm_config.get("max_workers", 16)
    try:
        configured_max_workers_int = int(configured_max_workers)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid harmonize llm.max_workers=%r; using default 16",
            configured_max_workers,
        )
        configured_max_workers_int = 16

    remap: dict[str, str] = {}
    max_workers = min(len(conflicts), max(1, configured_max_workers_int))
    logger.info(
        "Resolving %d conflict groups in parallel (%d workers)",
        len(conflicts),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _resolve_single_group,
                members,
                predictions,
                group_signals_map.get(rep, []),
                llm,
                llm_config.get("model"),
                token_tracker,
            ): rep
            for rep, members in conflicts.items()
        }
        for future in as_completed(futures):
            rep = futures[future]
            try:
                group_remap = future.result()
                remap.update(group_remap)
            except Exception:
                logger.exception("Thread failed for group rep=%d", rep)

    return remap


def _majority_vote_fallback(
    conflicts: dict[int, list[int]],
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    """Fallback when no LLM is available: use majority vote."""
    remap: dict[str, str] = {}
    for _rep, member_indices in conflicts.items():
        members = [predictions[i] for i in member_indices]
        mat_counts = Counter(m["materials"]["material"] for m in members)
        top_mat, top_count = mat_counts.most_common(1)[0]
        if top_count > len(members) / 2:
            for m in members:
                if m["materials"]["material"] != top_mat:
                    remap[m["id"]] = top_mat
    return remap


# ---------------------------------------------------------------------------
# Apply remap to prediction files
# ---------------------------------------------------------------------------


def _resolve_trusted_root(trusted_root: Path) -> Path:
    """Resolve a trusted output root used for predictions writes."""
    root = trusted_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Trusted predictions root is not a directory: {root}")
    return root


def _require_path_under_root(path: Path, trusted_root: Path) -> None:
    try:
        path.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(
            f"Predictions path must stay under trusted root {trusted_root}: {path}"
        ) from exc


def _atomic_write_text_under_root_portable(
    path: Path,
    trusted_root: Path,
    text: str,
) -> None:
    """Portable atomic text write for platforms without directory fd APIs."""
    resolved_path = path.resolve(strict=False)
    _require_path_under_root(resolved_path, trusted_root)
    if resolved_path.exists() and not resolved_path.is_file():
        raise ValueError(f"Predictions path is not a regular file: {resolved_path}")

    tmp_path = resolved_path.with_name(
        f".{resolved_path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    target_mode = (
        stat.S_IMODE(resolved_path.stat().st_mode) if resolved_path.exists() else None
    )
    replaced = False

    try:
        with tmp_path.open("x", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        if target_mode is not None:
            os.chmod(tmp_path, target_mode)
        os.replace(tmp_path, resolved_path)
        replaced = True
    finally:
        if not replaced:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _atomic_write_text_under_root(
    path: Path,
    trusted_root: Path,
    text: str,
) -> None:
    """Atomically write text by resolving path components from trusted_root."""
    relative_path = path.relative_to(trusted_root)
    if not relative_path.parts or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        raise ValueError(f"Invalid trusted-root-relative path: {relative_path}")
    if not _DIR_FD_ATOMIC_WRITE_SUPPORTED:
        _atomic_write_text_under_root_portable(path, trusted_root, text)
        return

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
    open_dir_fds: list[int] = []
    file_fd = -1
    tmp_name: str | None = None
    target_mode: int | None = None

    try:
        parent_fd = os.open(trusted_root, directory_flags)
        open_dir_fds.append(parent_fd)
        for part in relative_path.parts[:-1]:
            parent_fd = os.open(part, directory_flags | nofollow, dir_fd=parent_fd)
            open_dir_fds.append(parent_fd)

        try:
            target_fd = os.open(
                relative_path.name,
                os.O_RDONLY | nofollow,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            pass
        else:
            try:
                target_mode = stat.S_IMODE(os.fstat(target_fd).st_mode)
            finally:
                os.close(target_fd)

        tmp_name = f".{relative_path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        create_mode = target_mode if target_mode is not None else 0o666
        file_fd = os.open(tmp_name, file_flags, create_mode, dir_fd=parent_fd)
        if target_mode is not None:
            os.fchmod(file_fd, target_mode)
        with os.fdopen(file_fd, "w", encoding="utf-8") as file:
            file_fd = -1
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            tmp_name,
            relative_path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        tmp_name = None
        os.fsync(parent_fd)
    finally:
        if file_fd != -1:
            os.close(file_fd)
        if tmp_name is not None and open_dir_fds:
            try:
                os.unlink(tmp_name, dir_fd=open_dir_fds[-1])
            except FileNotFoundError:
                pass
        for directory_fd in reversed(open_dir_fds):
            os.close(directory_fd)


def _resolve_predictions_jsonl(
    pred_file: Path,
    trusted_root: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve and validate an existing predictions JSONL file."""
    pred_file = pred_file.resolve(strict=True)
    root = _resolve_trusted_root(trusted_root or pred_file.parent)
    _require_path_under_root(pred_file, root)

    if not pred_file.is_file() or pred_file.suffix != ".jsonl":
        raise ValueError(f"Not a predictions JSONL file: {pred_file}")
    return pred_file, root


def apply_prim_remap(
    pred_file: Path,
    remap: dict[str, str],
    *,
    trusted_root: Path,
) -> int:
    """Apply per-prim material remap to a predictions JSONL file.

    The target file must resolve under ``trusted_root``.

    Returns number of predictions updated.
    """
    pred_file, root = _resolve_predictions_jsonl(pred_file, trusted_root)

    lines = pred_file.read_text(encoding="utf-8").strip().split("\n")
    updated = 0
    new_lines = []

    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            entry = json.loads(line)
            prim_id = entry.get("id", "")
            if prim_id in remap:
                new_mat = remap[prim_id]
                mats = entry.get("materials", {})
                old_mat = mats.get("material", "")
                if old_mat != new_mat:
                    mats["material"] = new_mat
                    mats["harmonized_from"] = old_mat
                    entry["materials"] = mats
                    updated += 1
            new_lines.append(json.dumps(entry))
        except json.JSONDecodeError:
            new_lines.append(line)

    _atomic_write_text_under_root(pred_file, root, "\n".join(new_lines) + "\n")
    return updated


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _write_harmonize_report(
    predictions_path: Path,
    total_predictions: int,
    groups: dict[int, list[int]],
    conflicts: dict[int, list[int]],
    remap: dict[str, str],
    predictions: list[dict[str, Any]],
    group_signals_map: dict[int, list[str]],
    *,
    trusted_root: Path,
) -> Path:
    """Write a JSON report summarising the harmonize step."""
    predictions_path, root = _resolve_predictions_jsonl(predictions_path, trusted_root)
    report: dict[str, Any] = {
        "total_predictions": total_predictions,
        "groups_count": len(groups),
        "conflict_groups_count": len(conflicts),
        "remapped_prims": len(remap),
        "ungrouped_count": total_predictions
        - len({i for members in groups.values() for i in members}),
        "groups": [],
    }

    for rep, member_indices in groups.items():
        members = [predictions[i] for i in member_indices]
        mat_counts: dict[str, int] = {}
        for m in members:
            mat = m["materials"]["material"]
            mat_counts[mat] = mat_counts.get(mat, 0) + 1
        is_conflict = rep in conflicts
        signals = group_signals_map.get(rep, [])

        group_entry: dict[str, Any] = {
            "signals": signals,
            "member_count": len(members),
            "materials": mat_counts,
            "conflict": is_conflict,
            "prim_paths": [m["id"] for m in members],
        }
        if is_conflict:
            group_entry["remaps"] = {
                m["id"]: remap[m["id"]] for m in members if m["id"] in remap
            }
        report["groups"].append(group_entry)

    report_path = (predictions_path.parent / "harmonize_report.json").resolve()
    _require_path_under_root(report_path, root)
    _atomic_write_text_under_root(
        report_path,
        root,
        json.dumps(report, indent=2, ensure_ascii=False),
    )
    logger.info("Harmonize report written to %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------


def harmonize_asset_predictions(
    predictions_path: Path,
    llm_config: dict[str, Any] | None = None,
    optimized_usd_path: str | None = None,
    trusted_root: Path | None = None,
    token_tracker: Any | None = None,
) -> tuple[Path, dict[str, str]]:
    """Harmonize predictions within a single asset using multi-signal grouping.

    Three grouping signals are used and merged via union-find:
    1. Part-number signature (rare CAD-like path segments)
    2. Name template (strip trailing digits: Column2 → Column{})
    3. Geometry fingerprint (vertex/face count + bbox from USD)

    For conflict groups, ALL reasonings are sent to an LLM which decides
    whether to unify or keep different materials.

    Returns:
        Tuple of (predictions_path, remap dict).
    """
    predictions_path, root = _resolve_predictions_jsonl(predictions_path, trusted_root)
    predictions = _load_predictions(predictions_path)
    if len(predictions) < 2:
        logger.info("Too few predictions to harmonize (%d)", len(predictions))
        _write_harmonize_report(
            predictions_path,
            len(predictions),
            {},
            {},
            {},
            predictions,
            {},
            trusted_root=root,
        )
        return predictions_path, {}

    # Compute all three grouping signals.  Name-template grouping is constrained
    # by geometry where possible to avoid grouping unlike generic node siblings.
    sig_groups = _group_by_signature(predictions)
    raw_name_groups = _group_by_name_template(predictions)
    geometry_by_path = _compute_geometry_fingerprint_map(
        predictions, optimized_usd_path
    )
    name_groups = _constrain_name_groups_by_geometry(
        raw_name_groups, predictions, geometry_by_path
    )
    geo_groups = _compute_geometry_fingerprints(predictions, optimized_usd_path)

    logger.info(
        "Grouping signals: %d signature, %d name-template, %d geometry groups",
        len(sig_groups),
        len(name_groups),
        len(geo_groups),
    )

    # Merge all signals via union-find
    all_signal_groups = [sig_groups, name_groups, geo_groups]
    merged_groups = _merge_groups(all_signal_groups, len(predictions))

    if not merged_groups:
        logger.info("No multi-member groups found after merging signals")
        _write_harmonize_report(
            predictions_path,
            len(predictions),
            {},
            {},
            {},
            predictions,
            {},
            trusted_root=root,
        )
        return predictions_path, {}

    # Build a signal map: for each merged group representative, record which
    # signals contributed members to that group
    group_signals_map = _build_group_signals_map(
        merged_groups, sig_groups, name_groups, geo_groups
    )

    # Find conflicts
    conflicts = _find_conflicts(merged_groups, predictions)

    logger.info(
        "Found %d merged groups (%d with conflicts)",
        len(merged_groups),
        len(conflicts),
    )

    if not conflicts:
        _write_harmonize_report(
            predictions_path,
            len(predictions),
            merged_groups,
            {},
            {},
            predictions,
            group_signals_map,
            trusted_root=root,
        )
        return predictions_path, {}

    # Resolve via LLM
    remap = _resolve_conflicts(
        conflicts,
        predictions,
        group_signals_map,
        llm_config,
        token_tracker=token_tracker,
    )

    if remap:
        updated = apply_prim_remap(predictions_path, remap, trusted_root=root)
        logger.info("Harmonized %d predictions in %s", updated, predictions_path)

    _write_harmonize_report(
        predictions_path,
        len(predictions),
        merged_groups,
        conflicts,
        remap,
        predictions,
        group_signals_map,
        trusted_root=root,
    )

    return predictions_path, remap


def _build_group_signals_map(
    merged_groups: dict[int, list[int]],
    sig_groups: dict[str, list[int]],
    name_groups: dict[str, list[int]],
    geo_groups: dict[str, list[int]],
) -> dict[int, list[str]]:
    """For each merged group, identify which signals contributed."""
    # Build reverse map: prediction_index → set of signal keys
    index_to_signals: dict[int, set[str]] = defaultdict(set)
    for _key, indices in sig_groups.items():
        for i in indices:
            index_to_signals[i].add("signature")
    for _key, indices in name_groups.items():
        for i in indices:
            index_to_signals[i].add("name_template")
    for _key, indices in geo_groups.items():
        for i in indices:
            index_to_signals[i].add("geometry")

    result: dict[int, list[str]] = {}
    for rep, members in merged_groups.items():
        signals: set[str] = set()
        for i in members:
            signals |= index_to_signals.get(i, set())
        result[rep] = sorted(signals)

    return result


def harmonize_scene_predictions(
    manifest: Any,
    llm_config: dict[str, Any] | None = None,
    mode: str = "full",
    token_tracker: Any | None = None,
) -> dict[str, str]:
    """Harmonize predictions across all sub-assets in a scene.

    Gathers restored (or raw) predictions from every completed sub-asset,
    builds signatures globally, resolves conflicts, and writes fixes back.

    Args:
        manifest: Scene manifest with completed sub-assets.
        llm_config: LLM configuration dict (only used in ``"full"`` mode).
        mode: ``"full"`` uses multi-signal grouping + LLM resolution with
            majority-vote fallback.  ``"simple"`` uses the same multi-signal
            grouping but resolves all conflicts with majority vote only
            (no LLM call).  Both modes write the ``harmonized_from`` audit
            field.
        token_tracker: Optional TokenTracker for scene harmonization usage.
    """
    # Gather all predictions with source file tracking
    all_predictions: list[dict[str, Any]] = []
    pred_files: dict[str, Path] = {}  # prim_path → source file
    pred_roots: dict[Path, Path] = {}

    for sa in manifest.sub_assets:
        if sa.status != "completed" or not sa.working_dir:
            continue
        working_dir = Path(sa.working_dir)
        pred_file = _find_best_predictions(working_dir)
        if not pred_file:
            continue
        pred_file, root = _resolve_predictions_jsonl(pred_file, working_dir)
        pred_roots[pred_file] = root
        for entry in _load_predictions(pred_file):
            prim_id = entry["id"]
            all_predictions.append(entry)
            pred_files[prim_id] = pred_file

    if len(all_predictions) < 2:
        logger.info("Too few scene predictions to harmonize")
        return {}

    logger.info("Harmonizing %d predictions across scene", len(all_predictions))

    # Use signature grouping for cross-asset harmonization
    sig_groups = _group_by_signature(all_predictions)
    name_groups = _group_by_name_template(all_predictions)
    merged_groups = _merge_groups([sig_groups, name_groups], len(all_predictions))

    conflicts = _find_conflicts(merged_groups, all_predictions)

    if not conflicts:
        logger.info("No cross-asset conflicts (%d groups checked)", len(merged_groups))
        return {}

    logger.info(
        "Found %d cross-asset conflict groups (%d merged groups)",
        len(conflicts),
        len(merged_groups),
    )

    group_signals_map = _build_group_signals_map(
        merged_groups,
        sig_groups,
        name_groups,
        {},
    )
    remap = _resolve_conflicts(
        conflicts,
        all_predictions,
        group_signals_map,
        llm_config,
        mode=mode,
        token_tracker=token_tracker,
    )

    if remap:
        # Group remap entries by their source prediction file
        file_remaps: dict[Path, dict[str, str]] = defaultdict(dict)
        for prim_path, new_mat in remap.items():
            if prim_path in pred_files:
                file_remaps[pred_files[prim_path]][prim_path] = new_mat

        total_updated = 0
        for pred_file, file_remap in file_remaps.items():
            updated = apply_prim_remap(
                pred_file,
                file_remap,
                trusted_root=pred_roots[pred_file],
            )
            total_updated += updated

        logger.info(
            "Harmonized %d predictions across %d files",
            total_updated,
            len(file_remaps),
        )

    return remap


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load predictions from a JSONL file."""
    preds = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            preds.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return preds


def _find_best_predictions(working_dir: Path) -> Path | None:
    """Find the best predictions file in a working directory.

    Prefers restored predictions (original scene paths) over raw.
    """
    restored = working_dir / "restored" / "restored_predictions.jsonl"
    if restored.exists():
        return restored
    raw = working_dir / "predictions" / "predictions.jsonl"
    if raw.exists():
        return raw
    return None
