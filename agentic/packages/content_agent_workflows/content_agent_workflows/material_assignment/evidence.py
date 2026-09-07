# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-owned Material authoring evidence.

This module deliberately owns candidate selection, grouping, and the historical
Material evidence artifacts.  It reads the source USD directly; scene tools only
open, render, and author the already-decided USD operations.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
)
from content_agent_workflows.common.scene_correspondence import SceneOptimizerPathMap

from .decisions import VISIBLE_CANDIDATE_PRIMS_SCHEMA_VERSION
from .manifest import load_material_manifest

MATERIAL_CANDIDATE_SPACE_SOURCE = "source"
MATERIAL_CANDIDATE_SPACE_INSPECTION = "inspection"
SOURCE_CANDIDATE_SELECTION_RULE = (
    "Canonical source-space material coverage universe generated from visible "
    "source Mesh and GeomSubset candidates. Runtime paths are retained as "
    "evidence; source paths are the authorable targets."
)
INSPECTION_CANDIDATE_SELECTION_RULE = (
    "Inspection-space material candidates generated from visible Mesh and "
    "GeomSubset surfaces. Each candidate is restored to exactly one source "
    "target through workflow-owned optimizer correspondence."
)


@dataclass(frozen=True, slots=True)
class MaterialCandidatePolicy:
    """Explicit Material candidate/coverage policy for one workflow run."""

    material_candidate_space: str = MATERIAL_CANDIDATE_SPACE_SOURCE
    root_prim_path: str | None = None
    skip_instances: bool = True
    skip_prototypes: bool = False
    skip_invisible: bool = False

    def normalized(self) -> MaterialCandidatePolicy:
        root = (
            self.root_prim_path.strip() if isinstance(self.root_prim_path, str) else ""
        )
        return MaterialCandidatePolicy(
            material_candidate_space=(
                MATERIAL_CANDIDATE_SPACE_INSPECTION
                if self.material_candidate_space == MATERIAL_CANDIDATE_SPACE_INSPECTION
                else MATERIAL_CANDIDATE_SPACE_SOURCE
            ),
            root_prim_path=root or None,
            skip_instances=bool(self.skip_instances),
            skip_prototypes=bool(self.skip_prototypes),
            skip_invisible=bool(self.skip_invisible),
        )

    def as_dict(self) -> dict[str, Any]:
        policy = self.normalized()
        return {
            "material_candidate_space": policy.material_candidate_space,
            "root_prim_path": policy.root_prim_path,
            "skip_instances": policy.skip_instances,
            "skip_prototypes": policy.skip_prototypes,
            "skip_invisible": policy.skip_invisible,
        }


def build_material_authoring_evidence(
    *,
    run_dir: Path,
    session_id: str,
    source_usd: Path,
    inspection_usd: Path | None = None,
    correspondence: SceneOptimizerPathMap | None = None,
    materials_yaml: Path | None,
    materials_usd: Path | None,
    policy: MaterialCandidatePolicy,
    respect_existing_material_bindings: bool,
) -> dict[str, str]:
    """Write stable Material candidate/context artifacts from the selected stage.

    The artifact names and schemas are intentionally retained for workflow and
    benchmark consumers.  ``session_id`` identifies the usd-cli workflow session,
    not a policy-bearing scene service.
    """

    normalized = policy.normalized()
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_usd = source_usd.expanduser().resolve()
    inspection_usd = (inspection_usd or source_usd).expanduser().resolve()
    selected_usd = (
        inspection_usd
        if normalized.material_candidate_space == MATERIAL_CANDIDATE_SPACE_INSPECTION
        else source_usd
    )
    if (
        normalized.material_candidate_space == MATERIAL_CANDIDATE_SPACE_INSPECTION
        and selected_usd != source_usd
        and correspondence is None
    ):
        raise ValueError(
            "Inspection-space material candidates require optimizer correspondence "
            "to restore source targets."
        )
    survey_root = normalized.root_prim_path
    if (
        survey_root
        and normalized.material_candidate_space == MATERIAL_CANDIDATE_SPACE_INSPECTION
        and selected_usd != source_usd
    ):
        survey_root = _inspection_root_for_source_root(
            selected_usd,
            source_root=survey_root,
            correspondence=correspondence,
        )
    survey = _survey_material_candidates(
        selected_usd,
        root_prim_path=survey_root,
        skip_instances=normalized.skip_instances,
        skip_prototypes=normalized.skip_prototypes,
        skip_invisible=normalized.skip_invisible,
    )
    unsupported = survey["unsupported_materializable_gprims"]
    if unsupported:
        examples = ", ".join(
            f"{item['path']} ({item['type_name']})" for item in unsupported[:10]
        )
        raise ValueError(
            "Material candidate survey does not support materializable non-Mesh "
            f"Gprims: {examples}"
        )
    candidates = [
        _candidate_from_survey(
            item,
            path_space=normalized.material_candidate_space,
            correspondence=correspondence,
        )
        for item in survey["candidates"]
    ]
    if normalized.skip_instances:
        source_instance_occurrences = _source_instance_occurrence_paths(source_usd)
        for candidate in candidates:
            original_source_paths: list[str] = []
            for source_path in candidate["source_paths"]:
                for original_path in source_instance_occurrences.get(
                    source_path, [source_path]
                ):
                    if original_path not in original_source_paths:
                        original_source_paths.append(original_path)
            candidate["original_source_paths"] = original_source_paths
    if (
        normalized.material_candidate_space == MATERIAL_CANDIDATE_SPACE_SOURCE
        and normalized.skip_instances
    ):
        candidates = _coalesce_source_candidates(candidates)
    if normalized.root_prim_path:
        scoped_candidates: list[dict[str, Any]] = []
        for item in candidates:
            if _is_within(item["source_path"], normalized.root_prim_path):
                scoped_candidates.append(item)
                continue
            scoped_original_paths = [
                path
                for path in item["original_source_paths"]
                if _is_within(path, normalized.root_prim_path)
            ]
            if scoped_original_paths:
                item["original_source_paths"] = scoped_original_paths
                scoped_candidates.append(item)
        candidates = scoped_candidates
    if normalized.skip_prototypes:
        candidates = [
            item for item in candidates if "/__Prototype_" not in item["source_path"]
        ]
    palette = _load_palette(materials_yaml, materials_usd)
    groups = _candidate_groups(candidates, respect_existing_material_bindings)
    snapshot = _snapshot(
        session_id=session_id,
        source_usd=source_usd,
        inspection_usd=inspection_usd,
        candidates=candidates,
        survey=survey,
    )
    visible = {
        "schema_version": VISIBLE_CANDIDATE_PRIMS_SCHEMA_VERSION,
        "session_id": session_id,
        "source_usd": str(source_usd),
        "inspection_usd": str(inspection_usd),
        "path_space": normalized.material_candidate_space,
        "material_candidate_policy": normalized.as_dict(),
        "candidate_visible_prim_count": len(candidates),
        "candidate_selection_rule": (
            INSPECTION_CANDIDATE_SELECTION_RULE
            if normalized.material_candidate_space
            == MATERIAL_CANDIDATE_SPACE_INSPECTION
            else SOURCE_CANDIDATE_SELECTION_RULE
        ),
        "excluded_non_candidates": survey["excluded_non_candidates"],
        "traversal": survey["traversal"],
        "candidates": candidates,
    }
    context = {
        "schema_version": "content-agent-workflows.material-authoring-context.v1",
        "session_id": session_id,
        "root_prim_path": normalized.root_prim_path,
        "source_scene_path": str(source_usd),
        "inspection_scene_path": str(inspection_usd),
        "path_space": normalized.material_candidate_space,
        "material_candidate_policy": normalized.as_dict(),
        "traversal": survey["traversal"],
        "material_binding_policy": {
            "respect_existing_material_bindings": respect_existing_material_bindings,
            "description": (
                "Existing material bindings seed preserved coverage groups."
                if respect_existing_material_bindings
                else "Existing material bindings and display colors are diagnostic "
                "evidence only and do not imply an authoring decision."
            ),
        },
        "summary": {
            "prim_count": len(snapshot["paths"]),
            "candidate_count": len(candidates),
            "preliminary_candidate_count": len(candidates),
            "candidate_group_count": len(groups),
            "material_palette_count": palette["material_count"],
            "truncated": False,
        },
        "usage_guidance": [
            "Use visible_candidate_prims.json as the canonical coverage universe.",
            "Source prim paths are authorable targets; runtime paths are evidence.",
            "Use material_assignment_seed.json as a starting point and make a "
            "decision only from reference and rendered evidence.",
            "Candidate rows are coverage evidence, not scene commands.",
        ],
        "candidate_groups": groups,
        "candidates": candidates,
        "material_palette": palette,
    }
    seed = _assignment_seed(context)
    artifacts = {
        "scene_snapshot": raw_dir / "scene_snapshot.json",
        "tree_paths": raw_dir / "tree_paths.json",
        "properties": raw_dir / "properties_batch_all.json",
        "material_bindings": raw_dir / "material_binding_batch_all.json",
        "path_translations": raw_dir / "path_translation_batch_all.json",
        "visible_candidates_preliminary": raw_dir
        / "visible_candidate_prims_preliminary.json",
        "visible_candidates": raw_dir / "visible_candidate_prims.json",
        "material_authoring_context": raw_dir / "material_authoring_context.json",
        "material_authoring_context_md": raw_dir / "material_authoring_context.md",
        "visible_candidate_table": raw_dir / "visible_candidate_table.tsv",
        "material_palette": raw_dir / "material_palette.json",
        "material_assignment_seed": raw_dir / "material_assignment_seed.json",
    }
    atomic_write_json(artifacts["scene_snapshot"], snapshot)
    atomic_write_json(
        artifacts["tree_paths"],
        {
            "session_id": session_id,
            "root_prim_path": normalized.root_prim_path,
            "paths": snapshot["paths"],
            "nodes": snapshot["nodes"],
        },
    )
    atomic_write_json(
        artifacts["properties"],
        {"session_id": session_id, "results": snapshot["properties"]},
    )
    atomic_write_json(
        artifacts["material_bindings"],
        {"session_id": session_id, "results": snapshot["material_bindings"]},
    )
    atomic_write_json(
        artifacts["path_translations"],
        {"session_id": session_id, "results": snapshot["path_translations"]},
    )
    atomic_write_json(
        artifacts["visible_candidates_preliminary"],
        {
            "session_id": session_id,
            "source_usd": str(source_usd),
            "inspection_usd": str(inspection_usd),
            "candidate_visible_prim_count": len(candidates),
            "candidates": candidates,
            "excluded_non_candidates": survey["excluded_non_candidates"],
            "traversal": survey["traversal"],
            "summary": context["summary"],
        },
    )
    atomic_write_json(artifacts["visible_candidates"], visible)
    atomic_write_json(artifacts["material_authoring_context"], context)
    atomic_write_json(artifacts["material_palette"], palette)
    atomic_write_json(artifacts["material_assignment_seed"], seed)
    atomic_write_text(
        artifacts["material_authoring_context_md"], _context_markdown(context)
    )
    _write_table(artifacts["visible_candidate_table"], candidates)
    return {name: str(path) for name, path in artifacts.items()}


def _survey_material_candidates(
    usd_path: Path,
    *,
    root_prim_path: str | None,
    skip_instances: bool,
    skip_prototypes: bool,
    skip_invisible: bool,
) -> dict[str, Any]:
    """Collect workflow-owned candidate facts from one explicitly selected stage."""

    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Could not open material candidate USD: {usd_path}")
    if root_prim_path and not stage.GetPrimAtPath(root_prim_path).IsValid():
        raise ValueError(
            f"Material candidate root prim does not exist: {root_prim_path}"
        )
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    traversal = {
        "stage": str(usd_path),
        "root_prim_path": root_prim_path or "/",
        "skip_instances": skip_instances,
        "skip_prototypes": skip_prototypes,
        "skip_invisible": skip_invisible,
        # Proxies must always be traversed: with --skip-instances they are
        # collapsed to an authorable instance target, rather than discarded.
        "instance_proxy_traversal": True,
    }
    excluded = Counter()
    candidates: list[dict[str, Any]] = []
    unsupported_materializable_gprims: list[dict[str, str]] = []
    local_reference_source_roots = (
        _local_reference_source_roots(stage) if skip_prototypes else set()
    )
    prims = list(stage.Traverse(Usd.TraverseInstanceProxies()))
    traversal["stage_gprim_count"] = sum(prim.IsA(UsdGeom.Gprim) for prim in prims)
    for prim in prims:
        path = str(prim.GetPath())
        if root_prim_path and not _is_within(path, root_prim_path):
            excluded["outside_root"] += 1
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if skip_prototypes and _is_under_any(path, local_reference_source_roots):
            excluded["prototype_source"] += 1
            continue
        # Runtime /__Prototype_* prims are composed implementation details and
        # can never receive authored opinions. ``skip_prototypes`` controls
        # authorable source-side prototype roots, not this runtime universe.
        if prim.IsInPrototype() or prim.IsPrototype():
            excluded["runtime_prototype"] += 1
            continue
        imageable = UsdGeom.Imageable(prim)
        if skip_invisible and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            excluded["invisible"] += 1
            continue
        if prim.IsA(UsdGeom.BasisCurves):
            excluded["ignored_basis_curves"] += 1
            continue
        if not prim.IsA(UsdGeom.Mesh):
            if bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange().IsEmpty():
                excluded["non_renderable_gprim"] += 1
            else:
                unsupported_materializable_gprims.append(
                    {"path": path, "type_name": prim.GetTypeName()}
                )
            continue
        mesh = UsdGeom.Mesh(prim)
        face_count = len(mesh.GetFaceVertexCountsAttr().Get() or [])
        subsets = UsdGeom.Subset.GetAllGeomSubsets(imageable)
        covered_faces: set[int] = set()
        subset_rows: list[tuple[Any, int]] = []
        for subset in subsets:
            subset_prim = subset.GetPrim()
            family_name = subset.GetFamilyNameAttr().Get()
            if family_name != UsdShade.Tokens.materialBind:
                # GeomSubsets also encode semantic, physics, and other partitions.
                # Only the explicit materialBind family is safe to treat as a
                # material target; rewriting an unclassified subset would also
                # destroy workflow-external meaning.
                excluded["non_material_subset_family"] += 1
                continue
            indices = subset.GetIndicesAttr().Get() or []
            if indices:
                subset_rows.append((subset_prim, len(indices)))
                covered_faces.update(int(index) for index in indices)
        for subset_prim, subset_face_count in subset_rows:
            target = _authorable_instance_target(stage, subset_prim)
            if skip_instances and subset_prim.IsInstanceProxy() and target is None:
                excluded["unresolvable_instance_proxy"] += 1
                continue
            candidates.append(
                _survey_row(
                    material_prim=subset_prim,
                    display_prim=prim,
                    mesh_path=path,
                    face_count=subset_face_count,
                    bbox_cache=bbox_cache,
                    material_api=UsdShade.MaterialBindingAPI,
                    candidate_path=(
                        target.path
                        if skip_instances and target is not None
                        else str(subset_prim.GetPath())
                    ),
                    runtime_path=str(subset_prim.GetPath()),
                    instance_collapsed=bool(
                        skip_instances and target is not None and target.collapsed
                    ),
                    deinstance_root_path=(
                        target.deinstance_root_path if target is not None else None
                    ),
                )
            )
        if not subset_rows or len(covered_faces) < face_count:
            target = _authorable_instance_target(stage, prim)
            if skip_instances and prim.IsInstanceProxy() and target is None:
                excluded["unresolvable_instance_proxy"] += 1
                continue
            candidates.append(
                _survey_row(
                    material_prim=prim,
                    display_prim=prim,
                    mesh_path=path,
                    face_count=max(face_count - len(covered_faces), 0),
                    bbox_cache=bbox_cache,
                    material_api=UsdShade.MaterialBindingAPI,
                    candidate_path=(
                        target.path if skip_instances and target is not None else path
                    ),
                    runtime_path=path,
                    instance_collapsed=bool(
                        skip_instances and target is not None and target.collapsed
                    ),
                    deinstance_root_path=(
                        target.deinstance_root_path if target is not None else None
                    ),
                )
            )
    return {
        "source_usd": str(usd_path),
        "original_root_path": root_prim_path or "/",
        "candidates": candidates,
        "unsupported_materializable_gprims": unsupported_materializable_gprims,
        "traversal": traversal,
        "excluded_non_candidates": [
            {"reason": reason, "count": count}
            for reason, count in sorted(excluded.items())
        ],
    }


def _survey_row(
    *,
    material_prim: Any,
    display_prim: Any,
    mesh_path: str,
    face_count: int,
    bbox_cache: Any,
    material_api: Any,
    candidate_path: str,
    runtime_path: str,
    instance_collapsed: bool,
    deinstance_root_path: str | None,
) -> dict[str, Any]:
    material, relationship = material_api(material_prim).ComputeBoundMaterial()
    binding_source = None
    if relationship and relationship.GetPrim() and relationship.GetPrim().IsValid():
        binding_source = str(relationship.GetPrim().GetPath())
    material_path = str(material.GetPath()) if material else None
    direct = binding_source == str(material_prim.GetPath())
    display_color = _display_color(display_prim)
    center, size = _bounds(bbox_cache, display_prim)
    return {
        "prim_path": candidate_path,
        "runtime_path": runtime_path,
        "prim_type": material_prim.GetTypeName(),
        "mesh_path": mesh_path,
        "face_count": face_count,
        "bound_material_path": material_path,
        "bound_material_name": material.GetPrim().GetName() if material else None,
        "binding_source_path": binding_source,
        "material_binding_type": "direct"
        if direct
        else "inherited"
        if material
        else "none",
        "display_color": display_color,
        "bounds_center": center,
        "bounds_size": size,
        "size_hint": _size_hint(size),
        "shape_hint": _shape_hint(size),
        "instance_collapsed": instance_collapsed,
        "deinstance_root_path": deinstance_root_path,
    }


@dataclass(frozen=True, slots=True)
class _InstanceAuthoringTarget:
    path: str
    collapsed: bool
    deinstance_root_path: str | None = None


def _authorable_instance_target(
    stage: Any, prim: Any
) -> _InstanceAuthoringTarget | None:
    """Restore one proxy part to an exact authorable source-space target.

    Same-layer references map to their referenced source prim plus the proxy's
    child suffix, preserving the historical Workbench candidate semantics.
    External reference layers are not edited.  Their exact composed child path
    is retained and the enclosing instance root is recorded for an explicit,
    workflow-requested deinstance operation before authoring.
    """

    if not prim.IsInstanceProxy():
        return _InstanceAuthoringTarget(path=str(prim.GetPath()), collapsed=False)

    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.IsInstance() and not current.IsInstanceProxy():
            root = current
            break
        current = current.GetParent()
    else:
        return None

    from pxr import Usd

    relative = prim.GetPath().MakeRelativePath(root.GetPath())
    root_layer = stage.GetRootLayer()
    query = Usd.PrimCompositionQuery.GetDirectReferences(root)
    for arc in query.GetCompositionArcs():
        target = arc.GetTargetNode()
        if target.layerStack.identifier.rootLayer != root_layer:
            continue
        source_path = target.path if prim == root else target.path.AppendPath(relative)
        source_prim = stage.GetPrimAtPath(source_path)
        if source_prim and source_prim.IsValid() and not source_prim.IsInstanceProxy():
            return _InstanceAuthoringTarget(
                path=source_path.pathString,
                collapsed=True,
            )
    return _InstanceAuthoringTarget(
        path=str(prim.GetPath()),
        collapsed=False,
        deinstance_root_path=str(root.GetPath()),
    )


def _source_instance_occurrence_paths(source_usd: Path) -> dict[str, list[str]]:
    """Map canonical local-reference targets to composed source occurrences."""

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise ValueError(
            f"Could not open source USD for instance aliases: {source_usd}"
        )
    result: dict[str, list[str]] = {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsInstanceProxy() or not (
            prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset)
        ):
            continue
        target = _authorable_instance_target(stage, prim)
        if target is None or not target.collapsed:
            continue
        occurrence_path = str(prim.GetPath())
        occurrences = result.setdefault(target.path, [])
        if occurrence_path not in occurrences:
            occurrences.append(occurrence_path)
    return result


def _local_reference_source_roots(stage: Any) -> set[str]:
    """Return same-layer reference roots used as authored instance templates."""

    from pxr import Usd

    result: set[str] = set()
    root_layer = stage.GetRootLayer()
    for prim in stage.Traverse():
        if not prim.IsInstance():
            continue
        query = Usd.PrimCompositionQuery.GetDirectReferences(prim)
        for arc in query.GetCompositionArcs():
            target = arc.GetTargetNode()
            if target.layerStack.identifier.rootLayer == root_layer:
                result.add(target.path.pathString)
    return result


def _is_under_any(path: str, roots: set[str]) -> bool:
    return any(_is_within(path, root) for root in roots)


def _inspection_root_for_source_root(
    inspection_usd: Path,
    *,
    source_root: str,
    correspondence: SceneOptimizerPathMap | None,
) -> str:
    """Translate an author-supplied source root before inspecting optimized USD."""

    if correspondence is None:
        raise ValueError("Inspection root selection requires optimizer correspondence")
    translation = correspondence.translate_source_to_inspection(source_root)
    roots = translation.inspection_paths
    if translation.ambiguous or len(roots) != 1:
        raise ValueError(
            "Ambiguous optimizer correspondence for source root "
            f"{source_root}: {roots}. Refusing to choose an inspection root."
        )
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(inspection_usd))
    if stage is None or not stage.GetPrimAtPath(Sdf.Path(roots[0])).IsValid():
        raise ValueError(
            "Optimizer correspondence produced no inspection root for source root "
            f"{source_root}: {roots[0]}"
        )
    return roots[0]


def _display_color(prim: Any) -> list[float] | None:
    value = prim.GetAttribute("primvars:displayColor").Get()
    if not value:
        return None
    try:
        first = value[0]
        color = first if len(first) >= 3 else value
    except (IndexError, TypeError):
        color = value
    try:
        return [float(color[index]) for index in range(3)]
    except (IndexError, TypeError):
        return None


def _bounds(cache: Any, prim: Any) -> tuple[list[float], list[float]]:
    try:
        value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if value.IsEmpty():
            return [], []
        minimum, maximum = value.GetMin(), value.GetMax()
        center = [(float(minimum[i]) + float(maximum[i])) * 0.5 for i in range(3)]
        size = [float(maximum[i]) - float(minimum[i]) for i in range(3)]
        return center, size
    except Exception:  # noqa: BLE001 - a candidate remains useful without bounds
        return [], []


def _size_hint(size: list[float]) -> str:
    if not size:
        return "unknown"
    extent = max(size)
    if extent <= 0:
        return "unknown"
    return "small" if extent < 0.1 else "large" if extent > 1.0 else "medium"


def _shape_hint(size: list[float]) -> str:
    if len(size) != 3:
        return "unknown"
    ordered = sorted(abs(value) for value in size)
    if ordered[2] <= 0:
        return "unknown"
    if ordered[1] <= 0:
        return "thin_or_degenerate"
    long_to_mid = ordered[2] / ordered[1]
    mid_to_short = ordered[1] / max(ordered[0], 1e-9)
    if long_to_mid >= 5.0 and mid_to_short <= 2.5:
        return "slender_bar"
    if mid_to_short >= 5.0 and long_to_mid <= 2.5:
        return "thin_panel"
    if long_to_mid <= 2.0 and mid_to_short <= 2.5:
        return "blocky"
    return "irregular"


def _candidate_from_survey(
    item: dict[str, Any],
    *,
    path_space: str,
    correspondence: SceneOptimizerPathMap | None,
) -> dict[str, Any]:
    path = str(item["prim_path"])
    runtime_path = str(item.get("runtime_path") or path)
    if path_space == MATERIAL_CANDIDATE_SPACE_INSPECTION:
        # A collapsed candidate's durable target is its instance root, but the
        # optimizer maps the observed proxy mesh. Translate that mesh path to
        # recover the canonical source/prototype authoring target.
        correspondence_path = runtime_path if item.get("instance_collapsed") else path
        translation = (
            correspondence.translate_inspection_to_source(correspondence_path)
            if correspondence is not None
            else None
        )
        source_paths = translation.source_paths if translation is not None else [path]
        if translation is not None and translation.ambiguous:
            raise ValueError(
                "Ambiguous optimizer correspondence for inspection candidate "
                f"{path}: {source_paths}. Refusing to guess a source authoring target."
            )
        runtime_space = MATERIAL_CANDIDATE_SPACE_INSPECTION
        runtime_paths = [runtime_path]
        inspection_paths = [runtime_path]
    else:
        source_paths = [path]
        translation = (
            correspondence.translate_source_to_inspection(path)
            if correspondence is not None
            else None
        )
        runtime_space = MATERIAL_CANDIDATE_SPACE_SOURCE
        runtime_paths = [runtime_path]
        inspection_paths = (
            translation.inspection_paths if translation is not None else [path]
        )
    display = item.get("display_color")
    material_path = item.get("bound_material_path")
    deinstance_root = item.get("deinstance_root_path")
    return {
        "runtime_space": runtime_space,
        "runtime_path": runtime_paths[0],
        "runtime_paths": runtime_paths,
        "source_path": source_paths[0],
        "source_paths": source_paths,
        "original_source_paths": source_paths,
        "inspection_paths": inspection_paths,
        "inspection_path": inspection_paths[0],
        "name": _path_name(path),
        "parent": _parent_path(path),
        "path_tokens": _path_tokens(path),
        "semantic_hint": _semantic_hint(path),
        "type_name": str(item.get("prim_type") or "Mesh"),
        "candidate_reason": "visible_source_mesh_or_material_subset",
        "candidate_reasons": ["visible_source_mesh_or_material_subset"],
        "material_binding_type": item.get("material_binding_type", "none"),
        "material_name": item.get("bound_material_name")
        or _path_name(str(material_path or ""))
        or "unbound/default",
        "material_path": material_path,
        "direct_material_paths": [material_path] if material_path else [],
        "binding_source_path": item.get("binding_source_path"),
        "current_appearance_source": "material_binding"
        if material_path
        else "display_color"
        if display
        else "unbound_default",
        "display_color": display,
        "display_colors": [display] if isinstance(display, list) else [],
        "display_color_label": _display_color_label(display),
        "has_material_override": False,
        "ambiguous_translation": bool(translation and translation.ambiguous),
        "source_instance_count": 1,
        "instance_collapsed": bool(item.get("instance_collapsed")),
        "deinstance_root_paths": (
            [str(deinstance_root)] if isinstance(deinstance_root, str) else []
        ),
        "bounds_center": item.get("bounds_center", []),
        "bounds_size": item.get("bounds_size", []),
        "size_hint": item.get("size_hint", "unknown"),
        "shape_hint": item.get("shape_hint", "unknown"),
        "recommended_initial_status": "ambiguous_unassigned",
        "requires_material_assignment": False,
        "mesh_path": item.get("mesh_path"),
        "face_count": item.get("face_count", 0),
    }


def _coalesce_source_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse repeated instance observations onto each exact authoring target."""

    by_path: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        path = str(candidate["source_path"])
        current = by_path.get(path)
        if current is None:
            by_path[path] = candidate
            continue
        for key in (
            "runtime_paths",
            "inspection_paths",
            "original_source_paths",
            "deinstance_root_paths",
        ):
            values = current[key]
            for value in candidate[key]:
                if value not in values:
                    values.append(value)
        current["runtime_path"] = current["runtime_paths"][0]
        current["inspection_path"] = current["inspection_paths"][0]
        current["source_instance_count"] = len(current["runtime_paths"])
        current["instance_collapsed"] = bool(
            current["instance_collapsed"] or candidate["instance_collapsed"]
        )
    return [by_path[path] for path in sorted(by_path)]


def _snapshot(
    *,
    session_id: str,
    source_usd: Path,
    inspection_usd: Path,
    candidates: list[dict[str, Any]],
    survey: dict[str, Any],
) -> dict[str, Any]:
    paths = sorted({path for item in candidates for path in item["source_paths"]})
    properties = []
    bindings = []
    for item in candidates:
        if item["display_color"]:
            properties.append(
                {
                    "prim_path": item["source_path"],
                    "properties": {
                        "attributes": {
                            "primvars:displayColor": {"value": item["display_color"]}
                        }
                    },
                }
            )
        bindings.append(
            {
                "prim_path": item["source_path"],
                "material_path": item["material_path"],
                "binding_source_path": item["binding_source_path"],
            }
        )
    return {
        "session_id": session_id,
        "source_scene_path": str(source_usd),
        "inspection_scene_path": str(inspection_usd),
        "root_prim_path": survey.get("original_root_path", "/"),
        "paths": paths,
        "nodes": [
            {
                "path": path,
                "name": _path_name(path),
                "type_name": next(
                    item["type_name"]
                    for item in candidates
                    if item["source_path"] == path
                ),
                "active": True,
                "loaded": True,
                "children": False,
                "child_paths": [],
            }
            for path in paths
        ],
        "properties": properties,
        "material_bindings": bindings,
        "path_translations": [
            {"runtime_path": path, "source_paths": [path], "ambiguous": False}
            for path in paths
        ],
        "candidates": candidates,
        "excluded_non_candidates": [],
        "summary": {
            "prim_count": len(paths),
            "candidate_count": len(candidates),
            "ambiguous_translation_count": 0,
            "truncated": False,
        },
    }


def _candidate_groups(
    candidates: list[dict[str, Any]], respect_existing: bool
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        key = (
            str(item["material_path"] or item["material_name"])
            if respect_existing
            else _authoring_key(item)
        )
        grouped[key].append(item)
    result = []
    for index, (key, items) in enumerate(
        sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])), 1
    ):
        runtime_paths = sorted(
            {
                path
                for item in items
                for path in (
                    list(item.get("runtime_paths") or [])
                    or [str(item.get("runtime_path") or "")]
                )
                if path
            }
        )
        source_paths = sorted({path for item in items for path in item["source_paths"]})
        inspection_paths = sorted(
            {
                path
                for item in items
                for path in list(item.get("inspection_paths") or [])
                if path
            }
        )
        material_path = items[0]["material_path"]
        result.append(
            {
                "group_id": f"existing_material_{index:02d}"
                if respect_existing
                else f"authoring_family_{index:02d}",
                "grouping_basis": "existing_material"
                if respect_existing
                else "authoring_family",
                "authoring_family": str(
                    items[0]["material_name"]
                    if respect_existing
                    else key.replace(":", " ").replace("_", " ")
                ),
                "material_name": items[0]["material_name"]
                if respect_existing
                else None,
                "material_path": material_path if respect_existing else None,
                "existing_material_names": dict(
                    sorted(
                        Counter(str(item["material_name"]) for item in items).items()
                    )
                ),
                "existing_material_paths": sorted(
                    {
                        str(item["material_path"])
                        for item in items
                        if item["material_path"]
                    }
                ),
                "current_appearance_sources": dict(
                    sorted(
                        Counter(
                            str(item["current_appearance_source"]) for item in items
                        ).items()
                    )
                ),
                "display_color_counts": dict(
                    sorted(
                        Counter(
                            str(item["display_color_label"] or "none") for item in items
                        ).items()
                    )
                ),
                "recommended_coverage_status": "preserved_existing"
                if respect_existing and material_path
                else "ambiguous_unassigned",
                "requires_material_assignment": False,
                "candidate_count": len(items),
                "runtime_evidence_count": sum(
                    len(item.get("runtime_paths") or []) for item in items
                ),
                "collapsed_instance_candidate_count": sum(
                    1 for item in items if item.get("instance_collapsed")
                ),
                "runtime_space": str(items[0].get("runtime_space") or "source"),
                "runtime_paths": runtime_paths,
                "example_runtime_paths": runtime_paths[:8],
                "source_paths": source_paths,
                "example_source_paths": source_paths[:8],
                "inspection_paths": inspection_paths,
                "type_counts": dict(
                    sorted(Counter(str(item["type_name"]) for item in items).items())
                ),
                "reason_counts": dict(
                    sorted(
                        Counter(str(item["candidate_reason"]) for item in items).items()
                    )
                ),
                "size_hints": dict(
                    sorted(Counter(str(item["size_hint"]) for item in items).items())
                ),
                "shape_hints": dict(
                    sorted(Counter(str(item["shape_hint"]) for item in items).items())
                ),
                "semantic_hints": dict(
                    sorted(
                        Counter(str(item["semantic_hint"]) for item in items).items()
                    )
                ),
                "ambiguous_translation_count": sum(
                    1 for item in items if item.get("ambiguous_translation")
                ),
            }
        )
    return result


def _assignment_seed(context: dict[str, Any]) -> dict[str, Any]:
    groups = context["candidate_groups"]
    assignments = []
    for group in groups:
        status = group["recommended_coverage_status"]
        runtime_space = str(group.get("runtime_space") or context["path_space"])
        prim_paths = (
            group["runtime_paths"]
            if runtime_space == MATERIAL_CANDIDATE_SPACE_INSPECTION
            else group["source_paths"]
        )
        assignments.append(
            {
                "family": f"Seed: {group['authoring_family']}",
                "group_id": group["group_id"],
                "coverage_status": status,
                "material_name": group["material_name"]
                if status != "ambiguous_unassigned"
                else None,
                "material_path": group["material_path"],
                "grouping_basis": group["grouping_basis"],
                "authoring_family": group["authoring_family"],
                "semantic_hints": group["semantic_hints"],
                "shape_hints": group["shape_hints"],
                "existing_material_names": group["existing_material_names"],
                "existing_material_paths": group["existing_material_paths"],
                "current_appearance_sources": group["current_appearance_sources"],
                "display_color_counts": group["display_color_counts"],
                "runtime_space": runtime_space,
                "runtime_prim_paths": group["runtime_paths"],
                "source_prim_paths": group["source_paths"],
                "prim_paths": prim_paths,
                "rationale": "Seeded from existing material evidence."
                if status == "preserved_existing"
                else "Seeded as an unresolved authoring family; use reference and rendered evidence.",
            }
        )
    return {
        "schema_version": "content-agents.material-assignment-seed.v1",
        "session_id": context["session_id"],
        "source_usd": context["source_scene_path"],
        "inspection_usd": context["inspection_scene_path"],
        "path_space": context["path_space"],
        "material_candidate_policy": context["material_candidate_policy"],
        "library_path": context["material_palette"].get("materials_usd"),
        "per_prim_material_assignment_count": 0,
        "current_appearance_sources": dict(
            sorted(
                Counter(
                    str(item["current_appearance_source"])
                    for item in context["candidates"]
                ).items()
            )
        ),
        "display_color_counts": dict(
            sorted(
                Counter(
                    str(item["display_color_label"] or "none")
                    for item in context["candidates"]
                ).items()
            )
        ),
        "coverage": {
            "candidate_visible_prim_count": len(context["candidates"]),
            "material_decision_prim_count": len(context["candidates"]),
            "material_assignment_prim_count": 0,
            "preserved_existing_prim_count": sum(
                group["candidate_count"]
                for group in groups
                if group["recommended_coverage_status"] == "preserved_existing"
            ),
            "ambiguous_unassigned_prim_count": sum(
                group["candidate_count"]
                for group in groups
                if group["recommended_coverage_status"] == "ambiguous_unassigned"
            ),
            "coverage_notes": "Seed coverage generated from raw/visible_candidate_prims.json.",
        },
        "assignments": assignments,
        "final_review": {
            "issues_found": [],
            "issues_fixed": [],
            "unresolved_issues": [],
            "review_notes": "Seed placeholder; replace after final review.",
        },
        "visual_quality_assessment": {
            "status": "unreviewed_seed",
            "issues_found": [],
            "issues_fixed": [],
            "unresolved_issues": [],
            "checked_views": [],
            "assessment_notes": "Seed placeholder; replace after final renders.",
        },
    }


def _load_palette(
    materials_yaml: Path | None, materials_usd: Path | None
) -> dict[str, Any]:
    if materials_yaml is None:
        return {
            "schema_version": "content-agents.material-palette.v1",
            "materials_yaml": None,
            "materials_usd": str(materials_usd) if materials_usd else None,
            "material_count": 0,
            "materials": [],
            "tags": {},
        }
    return load_material_manifest(
        materials_yaml,
        library_override=materials_usd,
        validate_material_prims=False,
        allow_empty=True,
    ).as_palette()


def _write_table(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "runtime_path",
        "runtime_paths",
        "runtime_space",
        "source_path",
        "source_paths",
        "type_name",
        "material_name",
        "material_path",
        "current_appearance_source",
        "display_color",
        "display_color_label",
        "candidate_reason",
        "source_instance_count",
        "instance_collapsed",
        "recommended_initial_status",
        "requires_material_assignment",
        "size_hint",
        "shape_hint",
        "semantic_hint",
        "path_tokens",
        "ambiguous_translation",
    ]
    # csv is used only for its stable escaping; the finished table remains a legacy TSV artifact.
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for candidate in candidates:
        writer.writerow(
            {
                key: json.dumps(candidate[key], sort_keys=True)
                if isinstance(candidate.get(key), dict | list)
                else candidate.get(key)
                for key in fields
            }
        )
    atomic_write_text(path, buffer.getvalue())


def _context_markdown(context: dict[str, Any]) -> str:
    summary = context["summary"]
    lines = [
        "# Material Authoring Context",
        "",
        "## Candidate Policy",
        "",
        f"- Path space: {context['path_space']}",
        f"- Candidates: {summary['candidate_count']}",
        f"- Candidate groups: {summary['candidate_group_count']}",
        f"- Material palette entries: {summary['material_palette_count']}",
        "",
        "## Candidate Groups",
        "",
    ]
    for group in context["candidate_groups"]:
        lines.append(
            f"- **{group['group_id']}**: {group['candidate_count']} candidates; {group['authoring_family']}"
        )
    lines.extend(
        ["", "Candidate rows are workflow coverage evidence, not scene operations.", ""]
    )
    return "\n".join(lines)


def _display_color_label(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        red, green, blue = (float(item) for item in value[:3])
    except (TypeError, ValueError):
        return None
    if max(red, green, blue) - min(red, green, blue) <= 0.08:
        return (
            "dark_gray_display_color"
            if max(red, green, blue) < 0.28
            else "light_gray_display_color"
            if max(red, green, blue) > 0.68
            else "gray_display_color"
        )
    if green >= red + 0.18 and green >= blue + 0.18:
        return "green_display_color"
    if red >= green + 0.16 and red >= blue + 0.08:
        return "red_display_color"
    if blue >= red + 0.16 and blue >= green + 0.08:
        return "blue_display_color"
    return "mixed_display_color"


def _authoring_key(item: dict[str, Any]) -> str:
    semantic = str(item["semantic_hint"])
    shape = str(item.get("shape_hint") or "unknown")
    size = str(item.get("size_hint") or "unknown")
    appearance = str(item["display_color_label"] or "")
    if semantic != "generic_geometry":
        return (
            f"semantic:{semantic}:appearance:{appearance}"
            if appearance
            else f"semantic:{semantic}"
        )
    appearance_suffix = f":appearance:{appearance}" if appearance else ""
    parent = _path_name(str(item["parent"]))
    parent_tokens = [token for token in _path_tokens(parent) if token != "mesh"]
    if parent_tokens:
        return f"path:{'_'.join(parent_tokens[:3])}:{shape}:{size}{appearance_suffix}"
    return f"shape:{shape}:{size}{appearance_suffix}"


def _semantic_hint(path: str) -> str:
    tokens = set(_path_tokens(path))
    for hint, words in (
        ("wheel_roller", {"wheel", "roller", "caster", "tire"}),
        ("rail_bar", {"rail", "bar", "rod", "shaft", "pin"}),
        ("panel_frame", {"panel", "frame", "cover", "shell", "plate", "lid"}),
        ("fastener", {"screw", "bolt", "nut", "washer", "rivet"}),
    ):
        if tokens & words:
            return hint
    return "generic_geometry"


def _path_tokens(path: str) -> list[str]:
    return list(
        dict.fromkeys(
            token
            for token in re.split(r"[^a-z0-9]+", path.lower())
            if token
            and token
            not in {
                "mesh",
                "geometry",
                "visual",
                "collision",
                "left",
                "right",
                "l",
                "r",
            }
            and not token.isdigit()
        )
    )


def _path_name(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1] if path else ""


def _parent_path(path: str) -> str:
    stripped = path.rstrip("/")
    return stripped.rsplit("/", 1)[0] if "/" in stripped[1:] else ""


def _is_within(path: str, root: str) -> bool:
    normalized = root.rstrip("/") or "/"
    if normalized == "/":
        return path.startswith("/")
    return path == normalized or path.startswith(normalized + "/")
