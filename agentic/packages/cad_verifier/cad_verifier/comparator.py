# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral three-axis comparator evaluation.

Instead of relying on single-axis silhouette IoU, score geometry, topology,
and semantic-part placement independently:

1. **Geometric (bbox-IoU)** — the built asset's overall bounding box vs the
   comparator's. Catches gross sizing mistakes.
2. **Topological (kind-count parity)** — does the built asset have the right
   number of each ``asset_gen:kind`` (e.g., 4 wheels, 2 sensors)? Robust to
   spatial drift; catches missing or duplicated semantic parts.
3. **Semantic (kind-matching by spatial proximity)** — for each kind in the
   built asset, find the nearest comparator prim of the same kind and score
   by their bbox-volume ratio + center-distance / asset-diagonal.

Comparator USDs are loaded read-only via the OpenUSD Python bindings. Every
report stamps a ``rubric_version`` so cross-run comparisons detect rubric drift.

Public API:

    evaluate_against_comparator(built_usd, comparator_usd) → ComparatorReport
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RUBRIC_VERSION = "v1"


@dataclass
class KindCountRow:
    kind: str
    built: int
    comparator: int
    score: float  # 1.0 = exact match; falls off as |built - comparator| grows


@dataclass
class KindMatchRow:
    kind: str
    bbox_volume_ratio: float  # built / comparator (clamped 0..1 by min/max)
    center_distance_norm: float  # 0 = same place, 1 = full asset diagonal away
    score: float


@dataclass
class ComparatorReport:
    rubric_version: str = RUBRIC_VERSION
    built_usd: str = ""
    comparator_usd: str = ""
    geometric: float = 0.0  # bbox IoU 0..1
    topological: float = 0.0  # kind-count parity 0..1
    semantic: float = 0.0  # kind-position match 0..1
    overall: float = 0.0  # weighted blend
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "geometric": 0.3,
            "topological": 0.4,
            "semantic": 0.3,
        }
    )
    kind_counts: list[KindCountRow] = field(default_factory=list)
    kind_matches: list[KindMatchRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_version": self.rubric_version,
            "built_usd": self.built_usd,
            "comparator_usd": self.comparator_usd,
            "geometric": self.geometric,
            "topological": self.topological,
            "semantic": self.semantic,
            "overall": self.overall,
            "weights": dict(self.weights),
            "kind_counts": [asdict(r) for r in self.kind_counts],
            "kind_matches": [asdict(r) for r in self.kind_matches],
            "notes": list(self.notes),
        }


# ---------- USD I/O -------------------------------------------------------


def _load_stage(usd_path: str | Path):
    try:
        from pxr import Usd
    except ImportError as e:
        raise RuntimeError(
            "OpenUSD Python bindings are required; install `usd-exchange>=2.3,<3`"
        ) from e
    p = Path(usd_path)
    if not p.exists():
        raise FileNotFoundError(f"USD not found: {p}")
    return Usd.Stage.Open(str(p))


def _walk_meshes(stage) -> list:
    """Return every UsdGeom.Mesh prim in the stage."""
    from pxr import UsdGeom

    out = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            out.append(prim)
    return out


def _bbox_of_mesh(prim) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Compute world-space bbox of a single mesh prim from its Points attr."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    pts_attr = mesh.GetPointsAttr()
    if not pts_attr:
        return None
    pts = pts_attr.Get()
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _stage_bbox(stage) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    meshes = _walk_meshes(stage)
    if not meshes:
        return ((0, 0, 0), (0, 0, 0))
    mn_x = mn_y = mn_z = float("inf")
    mx_x = mx_y = mx_z = float("-inf")
    for m in meshes:
        bb = _bbox_of_mesh(m)
        if bb is None:
            continue
        mn_x = min(mn_x, bb[0][0])
        mn_y = min(mn_y, bb[0][1])
        mn_z = min(mn_z, bb[0][2])
        mx_x = max(mx_x, bb[1][0])
        mx_y = max(mx_y, bb[1][1])
        mx_z = max(mx_z, bb[1][2])
    if mn_x == float("inf"):
        return ((0, 0, 0), (0, 0, 0))
    return ((mn_x, mn_y, mn_z), (mx_x, mx_y, mx_z))


def _kind_of_prim(prim) -> str | None:
    """Read the asset_gen:kind custom attribute (REQ-MAT-5)."""
    attr = prim.GetAttribute("asset_gen:kind")
    if attr and attr.HasValue():
        return str(attr.Get())
    # Some external evaluators stamp an aggregate `asset_gen:kinds` TokenArray
    # on the root prim. This per-prim reader intentionally does not decode it.
    return None


def _meshes_by_kind(stage) -> dict[str, list]:
    out: dict[str, list] = {}
    for m in _walk_meshes(stage):
        k = _kind_of_prim(m)
        if k:
            out.setdefault(k, []).append(m)
    return out


def _bbox_iou(a, b) -> float:
    (a_mn, a_mx), (b_mn, b_mx) = a, b
    inter_mn = (max(a_mn[0], b_mn[0]), max(a_mn[1], b_mn[1]), max(a_mn[2], b_mn[2]))
    inter_mx = (min(a_mx[0], b_mx[0]), min(a_mx[1], b_mx[1]), min(a_mx[2], b_mx[2]))
    inter_extent = [max(0.0, inter_mx[i] - inter_mn[i]) for i in range(3)]
    inter_vol = inter_extent[0] * inter_extent[1] * inter_extent[2]
    a_extent = [max(0.0, a_mx[i] - a_mn[i]) for i in range(3)]
    b_extent = [max(0.0, b_mx[i] - b_mn[i]) for i in range(3)]
    a_vol = a_extent[0] * a_extent[1] * a_extent[2]
    b_vol = b_extent[0] * b_extent[1] * b_extent[2]
    union_vol = a_vol + b_vol - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol


def _bbox_center(bb) -> tuple[float, float, float]:
    return (
        (bb[0][0] + bb[1][0]) / 2,
        (bb[0][1] + bb[1][1]) / 2,
        (bb[0][2] + bb[1][2]) / 2,
    )


def _bbox_volume(bb) -> float:
    return (
        max(0.0, (bb[1][0] - bb[0][0]))
        * max(0.0, (bb[1][1] - bb[0][1]))
        * max(0.0, (bb[1][2] - bb[0][2]))
    )


# ---------- Axes -----------------------------------------------------------


def _topological_score(
    built_kinds: dict[str, int], comp_kinds: dict[str, int]
) -> tuple[float, list[KindCountRow]]:
    all_kinds = sorted(set(built_kinds) | set(comp_kinds))
    rows = []
    if not all_kinds:
        return 0.0, rows
    total = 0.0
    for k in all_kinds:
        b = built_kinds.get(k, 0)
        c = comp_kinds.get(k, 0)
        if b == c:
            score = 1.0
        elif max(b, c) == 0:
            score = 1.0
        else:
            # Symmetric ratio penalty: 1 - |b - c| / max(b, c)
            score = max(0.0, 1.0 - abs(b - c) / max(b, c))
        rows.append(KindCountRow(kind=k, built=b, comparator=c, score=score))
        total += score
    return total / len(all_kinds), rows


def _semantic_score(
    built_by_kind: dict[str, list], comp_by_kind: dict[str, list], asset_diag: float
) -> tuple[float, list[KindMatchRow]]:
    rows: list[KindMatchRow] = []
    if not comp_by_kind:
        return 0.0, rows
    total = 0.0
    n = 0
    for kind, comp_meshes in comp_by_kind.items():
        comp_bboxes = [_bbox_of_mesh(p) for p in comp_meshes]
        comp_bboxes = [b for b in comp_bboxes if b is not None]
        if not comp_bboxes:
            continue
        built_meshes = built_by_kind.get(kind, [])
        if not built_meshes:
            rows.append(
                KindMatchRow(kind=kind, bbox_volume_ratio=0.0, center_distance_norm=1.0, score=0.0)
            )
            n += 1
            continue
        # Greedy nearest-center match per built→comparator.
        built_bboxes = [_bbox_of_mesh(p) for p in built_meshes]
        built_bboxes = [b for b in built_bboxes if b is not None]
        for bb in built_bboxes:
            bc = _bbox_center(bb)
            best_score = 0.0
            best_vol_ratio = 0.0
            best_dist = 1.0
            for cb in comp_bboxes:
                cc = _bbox_center(cb)
                dist = math.sqrt(sum((bc[i] - cc[i]) ** 2 for i in range(3)))
                dist_norm = min(1.0, dist / max(asset_diag, 1.0))
                vb = _bbox_volume(bb)
                vc = _bbox_volume(cb)
                vol_ratio = (min(vb, vc) / max(vb, vc)) if max(vb, vc) > 0 else 0.0
                score = (1.0 - dist_norm) * 0.5 + vol_ratio * 0.5
                if score > best_score:
                    best_score = score
                    best_vol_ratio = vol_ratio
                    best_dist = dist_norm
            rows.append(
                KindMatchRow(
                    kind=kind,
                    bbox_volume_ratio=best_vol_ratio,
                    center_distance_norm=best_dist,
                    score=best_score,
                )
            )
            total += best_score
            n += 1
    return (total / n) if n > 0 else 0.0, rows


# ---------- Top-level ------------------------------------------------------


def evaluate_against_comparator(
    built_usd: str | Path,
    comparator_usd: str | Path,
    weights: dict[str, float] | None = None,
) -> ComparatorReport:
    """Run the 3-axis comparator rubric.

    Args:
        built_usd: path to the asset USD CAD Agent just produced.
        comparator_usd: path to the OEM / reference USD (with same
            ``asset_gen:kind`` attribute on each mesh).
        weights: dict overriding the default 0.3 / 0.4 / 0.3 axis weights.
    """
    rep = ComparatorReport(
        built_usd=str(built_usd),
        comparator_usd=str(comparator_usd),
    )
    if weights:
        rep.weights = {**rep.weights, **weights}

    built_stage = _load_stage(built_usd)
    comp_stage = _load_stage(comparator_usd)

    # 1) Geometric — overall bbox IoU.
    a_bb = _stage_bbox(built_stage)
    b_bb = _stage_bbox(comp_stage)
    rep.geometric = _bbox_iou(a_bb, b_bb)

    # Asset diagonal — used to normalize semantic distance.
    if b_bb != ((0, 0, 0), (0, 0, 0)):
        diag = math.sqrt(sum((b_bb[1][i] - b_bb[0][i]) ** 2 for i in range(3)))
    else:
        diag = 1.0
    diag = max(diag, 1.0)

    # 2) Topological — kind counts.
    built_by_kind = _meshes_by_kind(built_stage)
    comp_by_kind = _meshes_by_kind(comp_stage)
    if not comp_by_kind:
        rep.notes.append(
            "comparator USD has no asset_gen:kind tags — topological/semantic axes set to 0"
        )
    built_counts = {k: len(v) for k, v in built_by_kind.items()}
    comp_counts = {k: len(v) for k, v in comp_by_kind.items()}
    rep.topological, rep.kind_counts = _topological_score(built_counts, comp_counts)

    # 3) Semantic — kind-matching by spatial proximity.
    rep.semantic, rep.kind_matches = _semantic_score(built_by_kind, comp_by_kind, diag)

    # Overall — weighted blend.
    w = rep.weights
    rep.overall = (
        w["geometric"] * rep.geometric
        + w["topological"] * rep.topological
        + w["semantic"] * rep.semantic
    )
    return rep
