# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction analyzer for symmetry and consistency checking.

Analyzes VLM material predictions to detect:
- Symmetry violations: symmetric prim pairs with different materials
- Consistency violations: structurally similar parts with inconsistent materials

Used by JudgeTask to provide per-prim feedback for iterative refinement.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _split_trailing_number(value: str) -> tuple[str, int] | None:
    start = len(value)
    while start > 0 and value[start - 1].isdigit():
        start -= 1
    if start == len(value):
        return None
    return value[:start], int(value[start:])


@dataclass
class SymmetryViolation:
    """A pair of symmetric prims with mismatched materials."""

    prim_a: str
    prim_b: str
    material_a: str
    material_b: str
    suggested: str
    detection_method: str  # "spatial", "path", or "both"


@dataclass
class ConsistencyViolation:
    """A group of structurally similar prims with inconsistent materials."""

    group_name: str
    prims: list[str]
    materials: dict[str, list[str]]  # material -> list of prim IDs
    suggested: str


@dataclass
class AnalysisResult:
    """Complete analysis result with violations and feedback."""

    symmetry_pairs: list[tuple[str, str]]
    symmetry_violations: list[SymmetryViolation]
    consistency_violations: list[ConsistencyViolation]
    score: float
    critique: str
    prim_feedback: dict[str, str]  # prim_id -> feedback text
    resolved_assignments: dict[str, str]  # prim_id -> material to assign directly


class PredictionAnalyzer:
    """Analyze predictions for symmetry and consistency violations.

    Uses two complementary methods for symmetry detection:
    1. **Prim path analysis**: Detect naming patterns (e.g., consecutive IDs,
       left/right suffixes) that indicate symmetric pairs
    2. **Spatial analysis**: Mirror prim bounding box centers across the
       symmetry axis and find matching counterparts

    Args:
        predictions: List of prediction dicts with 'id' and 'materials.material'
        prims_metadata: List of prim metadata dicts with bounding box info.
            If None, only prim path analysis is used.
        symmetry_tolerance: Max distance between a prim center and its mirror
            match for spatial detection (in scene units)
        consistency_threshold: Minimum fraction of prims in a group that must
            agree on a material for it to be considered consistent
        detect_numbered_path_symmetry: Whether consecutive numeric suffixes
            should be treated as path-based symmetry evidence. This is kept
            optional because many authored assets use numbered suffixes only
            as opaque IDs.
    """

    def __init__(
        self,
        predictions: list[dict[str, Any]],
        prims_metadata: list[dict[str, Any]] | None = None,
        symmetry_tolerance: float = 5.0,
        consistency_threshold: float = 0.6,
        resolve_symmetry_directly: bool = True,
        resolve_consistency_directly: bool = True,
        detect_numbered_path_symmetry: bool = True,
    ):
        self.predictions = predictions
        self.prims_metadata = prims_metadata or []
        self.symmetry_tolerance = symmetry_tolerance
        self.consistency_threshold = consistency_threshold
        self.resolve_symmetry_directly = resolve_symmetry_directly
        self.resolve_consistency_directly = resolve_consistency_directly
        self.detect_numbered_path_symmetry = detect_numbered_path_symmetry

        # Build lookup maps
        self._pred_by_id: dict[str, dict[str, Any]] = {}
        self._material_by_id: dict[str, str] = {}
        for pred in predictions:
            pred_id = pred.get("id", "")
            self._pred_by_id[pred_id] = pred
            materials = pred.get("materials", {})
            if isinstance(materials, dict):
                self._material_by_id[pred_id] = materials.get("material", "")
            else:
                self._material_by_id[pred_id] = str(materials)

        # Build metadata lookup by prim path
        self._meta_by_path: dict[str, dict[str, Any]] = {}
        for meta in self.prims_metadata:
            path = meta.get("prim_path", "")
            self._meta_by_path[path] = meta

    def analyze(self) -> AnalysisResult:
        """Run full analysis and return results.

        Returns:
            AnalysisResult with violations, score, critique, and per-prim feedback
        """
        # Detect symmetric pairs
        pairs = self.detect_symmetry_pairs()
        logger.info("Detected %d symmetric pairs", len(pairs))

        # Check for violations
        sym_violations = self._check_symmetry_violations(pairs)
        logger.info("Found %d symmetry violations", len(sym_violations))

        consistency_violations = self._check_consistency_violations()
        logger.info("Found %d consistency violations", len(consistency_violations))

        # Compute score
        score = self._compute_score(pairs, sym_violations, consistency_violations)

        # Generate critique, per-prim feedback, and resolved assignments
        critique = self._generate_critique(sym_violations, consistency_violations)
        prim_feedback = self._generate_prim_feedback(
            sym_violations, consistency_violations
        )
        resolved = self._generate_resolved_assignments(
            sym_violations, consistency_violations
        )

        return AnalysisResult(
            symmetry_pairs=pairs,
            symmetry_violations=sym_violations,
            consistency_violations=consistency_violations,
            score=score,
            critique=critique,
            prim_feedback=prim_feedback,
            resolved_assignments=resolved,
        )

    # ========================================================================
    # Symmetry Detection
    # ========================================================================

    def detect_symmetry_pairs(self) -> list[tuple[str, str]]:
        """Detect symmetric prim pairs using path analysis and spatial analysis.

        Returns:
            List of (prim_a, prim_b) tuples representing symmetric pairs.
            Convention: prim_a < prim_b lexicographically.
        """
        path_pairs = self._detect_pairs_from_paths()
        spatial_pairs = self._detect_pairs_from_bounding_boxes()

        # Merge: union of both detection methods, deduplicate
        all_pairs_set: set[tuple[str, str]] = set()
        for a, b in path_pairs + spatial_pairs:
            pair = (min(a, b), max(a, b))
            all_pairs_set.add(pair)

        pairs = sorted(all_pairs_set)
        logger.debug(
            "Symmetry detection: %d path pairs, %d spatial pairs, %d unique",
            len(path_pairs),
            len(spatial_pairs),
            len(pairs),
        )
        return pairs

    def _detect_pairs_from_paths(self) -> list[tuple[str, str]]:
        """Detect symmetric pairs from prim path naming conventions.

        Handles patterns like:
        - Consecutive numbered parts: tesla34/tesla35, tesla36/tesla37
        - Left/right tokens: left_arm/right_arm, part_left/part_right
        - Numbered suffixes differing by pattern: part_01/part_02
        """
        pred_ids = list(self._material_by_id.keys())
        pred_id_set = set(pred_ids)
        pairs: list[tuple[str, str]] = []
        used: set[str] = set()

        # Strategy 0: exact path mirroring by swapping left/right tokens in every
        # path segment. This catches assets like Unitree G1 where the useful
        # side token may be several segments above a generic leaf like
        # /mesh/Geometry.
        for pid_a in pred_ids:
            if pid_a in used:
                continue
            for pid_b in self._mirrored_path_candidates(pid_a):
                if pid_b in pred_id_set and pid_b not in used and pid_b != pid_a:
                    pairs.append((pid_a, pid_b))
                    used.add(pid_a)
                    used.add(pid_b)
                    break

        if self.detect_numbered_path_symmetry:
            # Strategy 1: Consecutive numbered pairs. Treat this as optional
            # weak evidence because many assets use numeric suffixes as opaque
            # authoring IDs rather than left/right counterpart names.
            numbered: dict[str, tuple[str, int]] = {}
            for pid in pred_ids:
                if pid in used:
                    continue
                # Get the short name (parent segment of the path)
                short = self._get_short_name(pid)
                numbered_suffix = _split_trailing_number(short)
                if numbered_suffix:
                    prefix, num = numbered_suffix
                    numbered[pid] = (prefix, num)

            # Group by prefix and find consecutive pairs
            by_prefix: dict[str, list[tuple[str, int]]] = {}
            for pid, (prefix, num) in numbered.items():
                by_prefix.setdefault(prefix, []).append((pid, num))

            for _prefix, items in by_prefix.items():
                items.sort(key=lambda x: x[1])
                for i in range(len(items) - 1):
                    pid_a, num_a = items[i]
                    pid_b, num_b = items[i + 1]
                    # Consecutive numbers suggest a symmetric pair
                    if num_b - num_a == 1 and pid_a not in used and pid_b not in used:
                        pairs.append((pid_a, pid_b))
                        used.add(pid_a)
                        used.add(pid_b)

        # Strategy 2: Left/right naming patterns
        lr_patterns = [
            (r"^left_", r"right_"),
            (r"_left_", r"_right_"),
            (r"_left$", r"_right$"),
            (r"^l_", r"r_"),
            (r"_l_", r"_r_"),
            (r"_l$", r"_r$"),
            (r"^Left", r"Right"),
            (r"Left", r"Right"),
            (r"^L_", r"R_"),
            (r"_L_", r"_R_"),
        ]
        for pid_a in pred_ids:
            if pid_a in used:
                continue
            short_a = self._get_short_name(pid_a)
            for left_pat, right_pat in lr_patterns:
                if re.search(left_pat, short_a):
                    expected_right = re.sub(
                        left_pat, right_pat.replace(r"$", ""), short_a
                    )
                    # Find matching prim
                    for pid_b in pred_ids:
                        if pid_b in used or pid_b == pid_a:
                            continue
                        short_b = self._get_short_name(pid_b)
                        if short_b == expected_right:
                            pairs.append((pid_a, pid_b))
                            used.add(pid_a)
                            used.add(pid_b)
                            break

        return pairs

    def _mirrored_path_candidates(self, prim_id: str) -> list[str]:
        """Return exact left/right mirrored path candidates for a prim path."""
        candidates: list[str] = []
        seen: set[str] = set()
        side_pairs = [
            ("left", "right"),
            ("right", "left"),
            ("Left", "Right"),
            ("Right", "Left"),
            ("LEFT", "RIGHT"),
            ("RIGHT", "LEFT"),
            ("l", "r"),
            ("r", "l"),
            ("L", "R"),
            ("R", "L"),
        ]
        for source, target in side_pairs:
            candidate = self._replace_side_tokens(prim_id, source, target)
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
        return candidates

    @staticmethod
    def _replace_side_tokens(path: str, source: str, target: str) -> str | None:
        """Replace side tokens in each path segment.

        Tokens must be delimited by the segment start/end or underscores. This
        avoids matching unrelated words while covering names such as
        left_shoulder, shoulder_left, and upper_left_panel.
        """
        pattern = re.compile(rf"(^|_){re.escape(source)}(?=_|$)")

        def repl(match: re.Match[str]) -> str:
            return f"{match.group(1)}{target}"

        parts = path.split("/")
        changed = False
        mirrored_parts: list[str] = []
        for part in parts:
            mirrored = pattern.sub(repl, part)
            if mirrored != part:
                changed = True
            mirrored_parts.append(mirrored)

        if not changed:
            return None
        return "/".join(mirrored_parts)

    def _detect_pairs_from_bounding_boxes(self) -> list[tuple[str, str]]:
        """Detect symmetric pairs by mirroring bounding box centers.

        Algorithm:
        1. Compute center of each prim's bounding box
        2. Find the model's symmetry axis (try Y, X, Z — pick best)
        3. For each prim, mirror its center across the symmetry plane
        4. Find the nearest prim to the mirrored position
        5. If within tolerance, they're a symmetric pair
        """
        if not self.prims_metadata:
            return []

        # Compute centers for all prims that have predictions
        centers: dict[str, tuple[float, float, float]] = {}
        for meta in self.prims_metadata:
            prim_path = meta.get("prim_path", "")
            if prim_path not in self._material_by_id:
                continue

            extent_str = meta.get("metadata", {}).get("extent", "")
            center = self._parse_extent_center(extent_str)
            if center is not None:
                centers[prim_path] = center

        if len(centers) < 2:
            return []

        # Auto-detect symmetry axis by trying each axis
        best_axis = self._find_best_symmetry_axis(centers)
        if best_axis is None:
            logger.debug("Could not detect a symmetry axis from bounding boxes")
            return []

        logger.debug("Detected symmetry axis: %s", ["X", "Y", "Z"][best_axis])

        # Find the model center along the symmetry axis
        axis_values = [c[best_axis] for c in centers.values()]
        model_center = (min(axis_values) + max(axis_values)) / 2.0

        # Find symmetric pairs
        pairs: list[tuple[str, str]] = []
        used: set[str] = set()
        prim_ids = list(centers.keys())

        for pid_a in prim_ids:
            if pid_a in used:
                continue

            center_a = centers[pid_a]
            # Mirror across symmetry plane
            mirrored = list(center_a)
            mirrored[best_axis] = 2 * model_center - center_a[best_axis]

            # Check if this prim is near the center (torso/center part, no mirror)
            dist_from_center = abs(center_a[best_axis] - model_center)
            if dist_from_center < self.symmetry_tolerance:
                continue  # Center part, skip

            # Find nearest prim to mirrored position
            best_match = None
            best_dist = float("inf")
            for pid_b in prim_ids:
                if pid_b in used or pid_b == pid_a:
                    continue
                center_b = centers[pid_b]
                dist = sum((mirrored[k] - center_b[k]) ** 2 for k in range(3)) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_match = pid_b

            if best_match and best_dist <= self.symmetry_tolerance:
                pairs.append((pid_a, best_match))
                used.add(pid_a)
                used.add(best_match)

        return pairs

    def _find_best_symmetry_axis(
        self, centers: dict[str, tuple[float, float, float]]
    ) -> int | None:
        """Find the axis that produces the most valid symmetric pairs.

        Tries each axis (X=0, Y=1, Z=2) and returns the one with the most pairs.
        """
        best_axis = None
        best_count = 0

        for axis in range(3):
            count = self._count_mirror_pairs(centers, axis)
            if count > best_count:
                best_count = count
                best_axis = axis

        # Need at least 1 pair for it to be a valid symmetry axis
        return best_axis if best_count >= 1 else None

    def _count_mirror_pairs(
        self, centers: dict[str, tuple[float, float, float]], axis: int
    ) -> int:
        """Count how many mirror pairs exist for a given axis."""
        axis_values = [c[axis] for c in centers.values()]
        model_center = (min(axis_values) + max(axis_values)) / 2.0

        count = 0
        used: set[str] = set()
        prim_ids = list(centers.keys())

        for pid_a in prim_ids:
            if pid_a in used:
                continue

            center_a = centers[pid_a]
            dist_from_center = abs(center_a[axis] - model_center)
            if dist_from_center < self.symmetry_tolerance:
                continue

            mirrored = list(center_a)
            mirrored[axis] = 2 * model_center - center_a[axis]

            for pid_b in prim_ids:
                if pid_b in used or pid_b == pid_a:
                    continue
                center_b = centers[pid_b]
                dist = sum((mirrored[k] - center_b[k]) ** 2 for k in range(3)) ** 0.5
                if dist <= self.symmetry_tolerance:
                    count += 1
                    used.add(pid_a)
                    used.add(pid_b)
                    break

        return count

    # ========================================================================
    # Violation Checking
    # ========================================================================

    def _check_symmetry_violations(
        self, pairs: list[tuple[str, str]]
    ) -> list[SymmetryViolation]:
        """Check which symmetric pairs have mismatched materials."""
        # Compute detection sets ONCE before the loop
        path_pairs = {
            (min(a, b), max(a, b)) for a, b in self._detect_pairs_from_paths()
        }
        spatial_pairs = {
            (min(a, b), max(a, b)) for a, b in self._detect_pairs_from_bounding_boxes()
        }

        violations = []
        for prim_a, prim_b in pairs:
            mat_a = self._material_by_id.get(prim_a, "")
            mat_b = self._material_by_id.get(prim_b, "")

            if mat_a and mat_b and mat_a != mat_b:
                # Choose suggested material: prefer the one that appears more
                # frequently across all predictions (global popularity)
                suggested = self._pick_dominant_material(mat_a, mat_b)

                # Determine detection method
                key = (min(prim_a, prim_b), max(prim_a, prim_b))
                in_path = key in path_pairs
                in_spatial = key in spatial_pairs
                if in_path and in_spatial:
                    method = "both"
                elif in_path:
                    method = "path"
                else:
                    method = "spatial"

                violations.append(
                    SymmetryViolation(
                        prim_a=prim_a,
                        prim_b=prim_b,
                        material_a=mat_a,
                        material_b=mat_b,
                        suggested=suggested,
                        detection_method=method,
                    )
                )

        return violations

    def _check_consistency_violations(self) -> list[ConsistencyViolation]:
        """Check for inconsistent material usage across similar part groups.

        Groups prims by their structural category (extracted from prim path)
        and checks if a dominant material exists.
        """
        # Group by structural category from prim path
        groups: dict[str, list[str]] = {}
        for pid in self._material_by_id:
            group = self._extract_group_name(pid)
            if group:
                groups.setdefault(group, []).append(pid)

        violations = []
        for group_name, prim_ids in groups.items():
            if len(prim_ids) < 2:
                continue

            # Count materials in this group
            mat_counts: Counter[str] = Counter()
            mat_prims: dict[str, list[str]] = {}
            for pid in prim_ids:
                mat = self._material_by_id.get(pid, "")
                if mat:
                    mat_counts[mat] += 1
                    mat_prims.setdefault(mat, []).append(pid)

            if len(mat_counts) <= 1:
                continue  # All same material, no issue

            # Check if dominant material meets threshold
            total = sum(mat_counts.values())
            dominant_mat, dominant_count = mat_counts.most_common(1)[0]
            dominance_ratio = dominant_count / total

            if dominance_ratio < self.consistency_threshold:
                # No clear dominant material — flag as violation
                violations.append(
                    ConsistencyViolation(
                        group_name=group_name,
                        prims=prim_ids,
                        materials=mat_prims,
                        suggested=dominant_mat,
                    )
                )

        return violations

    # ========================================================================
    # Scoring
    # ========================================================================

    def _compute_score(
        self,
        pairs: list[tuple[str, str]],
        sym_violations: list[SymmetryViolation],
        consistency_violations: list[ConsistencyViolation],
    ) -> float:
        """Compute prediction consistency score (0-1).

        Weights:
        - Symmetry: each violation deducts proportionally from the symmetry score
        - Consistency: each violation deducts proportionally from the consistency score
        - Final score = 0.7 * symmetry_score + 0.3 * consistency_score
        """
        # Guard: if no predictions at all, return a low score
        if not self._material_by_id:
            logger.warning("No predictions to analyze - returning score 0.0")
            return 0.0

        # Symmetry score
        if pairs:
            sym_score = 1.0 - (len(sym_violations) / len(pairs))
        else:
            sym_score = 1.0  # No pairs to check

        # Consistency score
        total_groups = len(
            {
                self._extract_group_name(pid)
                for pid in self._material_by_id
                if self._extract_group_name(pid)
            }
        )
        if total_groups > 0:
            con_score = 1.0 - (len(consistency_violations) / total_groups)
        else:
            con_score = 1.0

        score = 0.7 * max(0.0, sym_score) + 0.3 * max(0.0, con_score)
        return round(max(0.0, min(1.0, score)), 3)

    # ========================================================================
    # Critique & Feedback Generation
    # ========================================================================

    def _generate_critique(
        self,
        sym_violations: list[SymmetryViolation],
        consistency_violations: list[ConsistencyViolation],
    ) -> str:
        """Generate human-readable critique text for the VLM feedback loop."""
        parts: list[str] = []

        if not sym_violations and not consistency_violations:
            return "All predictions are symmetric and consistent. No issues found."

        if sym_violations:
            parts.append("**SYMMETRY ISSUES:**")
            for v in sym_violations:
                short_a = self._get_short_name(v.prim_a)
                short_b = self._get_short_name(v.prim_b)
                if self.resolve_symmetry_directly:
                    parts.append(
                        f"- {short_a} ('{v.material_a}') should match "
                        f"{short_b} ('{v.material_b}'). "
                        f"Recommended: Both should use '{v.suggested}'."
                    )
                else:
                    parts.append(
                        f"- {short_a} ('{v.material_a}') should match "
                        f"{short_b} ('{v.material_b}'). Re-evaluate both "
                        f"symmetric parts against the reference image and "
                        f"rendered views, then choose the shared material that "
                        f"best matches the target appearance."
                    )

        if consistency_violations:
            parts.append("")
            parts.append("**CONSISTENCY ISSUES:**")
            for v in consistency_violations:
                mat_summary = ", ".join(
                    f"'{mat}' ({len(pids)} parts)"
                    for mat, pids in sorted(
                        v.materials.items(), key=lambda x: -len(x[1])
                    )
                )
                if self.resolve_consistency_directly:
                    parts.append(
                        f"- Group '{v.group_name}' has inconsistent materials: "
                        f"{mat_summary}. "
                        f"Recommended: Use '{v.suggested}' for all."
                    )
                else:
                    parts.append(
                        f"- Group '{v.group_name}' has inconsistent materials: "
                        f"{mat_summary}. Re-evaluate the affected parts against "
                        f"the reference image and rendered views instead of "
                        f"choosing a material only from group frequency."
                    )

        return "\n".join(parts)

    def _generate_prim_feedback(
        self,
        sym_violations: list[SymmetryViolation],
        consistency_violations: list[ConsistencyViolation],
    ) -> dict[str, str]:
        """Generate per-prim feedback dict keyed by prim ID.

        Each prim with a violation gets specific text explaining what to change.
        """
        feedback: dict[str, str] = {}

        for v in sym_violations:
            short_a = self._get_short_name(v.prim_a)
            short_b = self._get_short_name(v.prim_b)

            if not self.resolve_symmetry_directly:
                feedback[v.prim_a] = (
                    f"SYMMETRY MISMATCH: This part was assigned '{v.material_a}', "
                    f"but its symmetric counterpart ({short_b}) was assigned "
                    f"'{v.material_b}'. Re-evaluate this part against the "
                    f"reference image and rendered views, and choose the material "
                    f"that should be shared by both symmetric parts."
                )
                feedback[v.prim_b] = (
                    f"SYMMETRY MISMATCH: This part was assigned '{v.material_b}', "
                    f"but its symmetric counterpart ({short_a}) was assigned "
                    f"'{v.material_a}'. Re-evaluate this part against the "
                    f"reference image and rendered views, and choose the material "
                    f"that should be shared by both symmetric parts."
                )
                continue

            if v.material_a != v.suggested:
                feedback[v.prim_a] = (
                    f"SYMMETRY MISMATCH: You assigned '{v.material_a}' but the "
                    f"symmetric counterpart ({short_b}) was assigned "
                    f"'{v.material_b}'. These symmetric parts must have the "
                    f"same material. Please assign '{v.suggested}' to this part."
                )
            if v.material_b != v.suggested:
                feedback[v.prim_b] = (
                    f"SYMMETRY MISMATCH: You assigned '{v.material_b}' but the "
                    f"symmetric counterpart ({short_a}) was assigned "
                    f"'{v.material_a}'. These symmetric parts must have the "
                    f"same material. Please assign '{v.suggested}' to this part."
                )

        for v in consistency_violations:
            for mat, pids in v.materials.items():
                if not self.resolve_consistency_directly:
                    for pid in pids:
                        if pid not in feedback:  # Don't overwrite symmetry feedback
                            short = self._get_short_name(pid)
                            feedback[pid] = (
                                f"CONSISTENCY REVIEW: {short} is in the "
                                f"'{v.group_name}' group, which has mixed "
                                f"materials. Re-evaluate this part against the "
                                f"reference image and rendered views instead of "
                                f"choosing a material only from group frequency."
                            )
                elif mat != v.suggested:
                    for pid in pids:
                        if pid not in feedback:  # Don't overwrite symmetry feedback
                            short = self._get_short_name(pid)
                            feedback[pid] = (
                                f"CONSISTENCY ISSUE: You assigned '{mat}' to "
                                f"{short}, but most similar parts in the "
                                f"'{v.group_name}' group use '{v.suggested}'. "
                                f"Please assign '{v.suggested}' for consistency."
                            )

        return feedback

    def _generate_resolved_assignments(
        self,
        sym_violations: list[SymmetryViolation],
        consistency_violations: list[ConsistencyViolation],
    ) -> dict[str, str]:
        """Generate resolved material assignments for prims with violations.

        Instead of asking the VLM to re-predict, the analyzer determines the
        correct material directly based on symmetry/consistency rules and
        outputs concrete assignments.

        Returns:
            Dict mapping prim_id -> material name to assign directly.
        """
        resolved: dict[str, str] = {}

        # Symmetry: both prims in a pair get the suggested material only when
        # direct resolution is enabled. Otherwise the feedback path sends both
        # prims back through the VLM, where reference images are still visible.
        if self.resolve_symmetry_directly:
            for v in sym_violations:
                resolved[v.prim_a] = v.suggested
                resolved[v.prim_b] = v.suggested

        # Consistency: outlier prims get the dominant material
        if self.resolve_consistency_directly:
            for v in consistency_violations:
                for mat, pids in v.materials.items():
                    if mat != v.suggested:
                        for pid in pids:
                            if pid not in resolved:  # Don't override symmetry fixes
                                resolved[pid] = v.suggested

        return resolved

    # ========================================================================
    # Helpers
    # ========================================================================

    def _get_short_name(self, prim_id: str) -> str:
        """Extract a readable short name from a prim path."""
        parts = prim_id.rstrip("/").split("/")
        generic = {
            "",
            "Geometry",
            "Mesh",
            "mesh",
            "visuals",
            "collisions",
            "colliders",
        }
        # Walk upward until we find a semantic segment. Some assets end in
        # /mesh/Geometry, where the old parent-segment rule returned "mesh".
        name = next(
            (part for part in reversed(parts) if part not in generic), parts[-1]
        )
        # Remove common prefixes for readability
        name = re.sub(r"^humanoid__unitree__g1__", "", name)
        return name

    def _extract_group_name(self, prim_id: str) -> str:
        """Extract structural group name from prim path.

        Examples:
        - '.../g1__legs__tesla34/...' -> 'legs'
        - '.../g1__hands__tesla36/...' -> 'hands'
        - '.../g1__torso__tesla38/...' -> 'torso'

        Falls back to the parent-of-parent path segment.
        """
        # Try to extract group from common naming patterns
        # Pattern: __<group>__<identifier>
        match = re.search(r"__(\w+?)__(?:tesla\d+|left|right|[lr]_)", prim_id, re.I)
        if match:
            return match.group(1)

        # Fallback: use the grandparent path segment
        parts = prim_id.rstrip("/").split("/")
        if len(parts) >= 3:
            return parts[-3].split("__")[-1] if "__" in parts[-3] else parts[-3]

        return ""

    def _pick_dominant_material(self, mat_a: str, mat_b: str) -> str:
        """Pick the more globally popular material from a pair.

        Counts how often each material appears across ALL predictions.
        """
        all_materials = list(self._material_by_id.values())
        count_a = all_materials.count(mat_a)
        count_b = all_materials.count(mat_b)
        return mat_a if count_a >= count_b else mat_b

    @staticmethod
    def _parse_extent_center(
        extent_str: str,
    ) -> tuple[float, float, float] | None:
        """Parse extent string and return center point.

        Extent format: '[(-7.08, 8.12, -58.02), (13.42, 15.40, -51.52)]'
        """
        if not extent_str:
            return None

        try:
            # Extract numbers from the extent string
            numbers = re.findall(r"-?\d+\.?\d*", extent_str)
            if len(numbers) < 6:
                return None

            min_x, min_y, min_z = (
                float(numbers[0]),
                float(numbers[1]),
                float(numbers[2]),
            )
            max_x, max_y, max_z = (
                float(numbers[3]),
                float(numbers[4]),
                float(numbers[5]),
            )

            return (
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                (min_z + max_z) / 2.0,
            )
        except (ValueError, IndexError):
            logger.debug("Failed to parse extent: %s", extent_str)
            return None


def load_predictions(predictions_path: str | Path) -> list[dict[str, Any]]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def load_prims_metadata(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load prim metadata from a prims.jsonl file.

    Looks for prims.jsonl in the dataset's usd/ subdirectory.
    Falls back to the dataset directory itself.
    """
    dataset_path = Path(dataset_path)

    # Try usd/prims.jsonl relative to dataset
    candidates = [
        dataset_path.parent / "usd" / "prims.jsonl",
        dataset_path.parent / "prims.jsonl",
    ]

    for path in candidates:
        if path.exists():
            metadata = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metadata.append(json.loads(line))
            logger.info("Loaded %d prim metadata entries from %s", len(metadata), path)
            return metadata

    logger.warning(
        "No prims.jsonl found near %s. Spatial symmetry detection will be skipped.",
        dataset_path,
    )
    return []
