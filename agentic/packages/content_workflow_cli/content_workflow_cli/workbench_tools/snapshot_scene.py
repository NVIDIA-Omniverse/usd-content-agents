# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call the Content Workbench scene snapshot API and write workflow artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from content_workbench_agent_client import snapshot_scene as _client_snapshot_scene

DEFAULT_WORKBENCH_URL = "http://127.0.0.1:8088"
CANDIDATE_SELECTION_RULE = (
    "Workbench-generated visible/renderable material candidate hints: active, "
    "loaded, effectively visible prims with non-empty bounds that are renderable "
    "geometry or have material bindings. Agents may refine this checklist before "
    "final coverage."
)
RUNTIME_CANDIDATE_SELECTION_RULE = (
    "Canonical optimized-session runtime material coverage universe generated "
    "from renderable inspection-space Mesh candidates. Source prim paths are "
    "recorded as export expansions and are not live material_override command targets."
)
SOURCE_CANDIDATE_SELECTION_RULE = (
    "Canonical source-space material coverage universe generated from visible "
    "runtime Mesh evidence. Instance-proxy candidates are collapsed to their "
    "authorable source/prototype prim paths when skip_instances is enabled; "
    "runtime paths are retained as evidence and preview fan-out."
)
MATERIAL_CANDIDATE_SPACE_SOURCE = "source"
MATERIAL_CANDIDATE_SPACE_INSPECTION = "inspection"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaterialCandidatePolicy:
    """Policy for converting Workbench visible hints into prediction candidates."""

    material_candidate_space: str = MATERIAL_CANDIDATE_SPACE_SOURCE
    root_prim_path: str | None = None
    skip_instances: bool = True
    skip_prototypes: bool = False
    skip_invisible: bool = False

    def normalized(self) -> MaterialCandidatePolicy:
        space = (
            MATERIAL_CANDIDATE_SPACE_INSPECTION
            if self.material_candidate_space == MATERIAL_CANDIDATE_SPACE_INSPECTION
            else MATERIAL_CANDIDATE_SPACE_SOURCE
        )
        root = (
            self.root_prim_path.strip()
            if isinstance(self.root_prim_path, str)
            else None
        )
        return MaterialCandidatePolicy(
            material_candidate_space=space,
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        snapshot = fetch_snapshot(
            workbench_url=args.workbench_url,
            session_id=args.session_id,
            root_prim_path=args.root_prim_path,
            include_properties=args.include_properties,
            include_material_bindings=args.include_material_bindings,
            include_path_translations=args.include_path_translations,
            include_candidate_hints=args.include_candidate_hints,
            max_prims=args.max_prims,
            timeout=args.timeout,
        )
        artifacts = write_snapshot_artifacts(
            snapshot,
            args.run_dir,
            materials_yaml=args.materials_yaml,
            materials_usd=args.materials_usd,
            append_trace=not args.no_trace,
            respect_existing_material_bindings=args.respect_existing_material_bindings,
            candidate_policy=MaterialCandidatePolicy(
                material_candidate_space=args.material_candidate_space,
                root_prim_path=args.root_prim_path,
                skip_instances=args.skip_instances,
                skip_prototypes=args.skip_prototypes,
                skip_invisible=args.skip_invisible,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - command boundary
        print(f"content-workbench-snapshot-scene: error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(compact_summary(snapshot, artifacts), sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="content-workbench-snapshot-scene",
        description=(
            "Call Content Workbench /scene/snapshot, write standard run artifacts, "
            "and print only compact counts."
        ),
    )
    parser.add_argument(
        "--workbench-url",
        default=os.getenv("CONTENT_WORKBENCH_URL", DEFAULT_WORKBENCH_URL),
        help="Content Workbench endpoint.",
    )
    parser.add_argument("--session-id", required=True, help="Workbench session ID.")
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="content-workflow-cli run directory.",
    )
    parser.add_argument(
        "--materials-yaml",
        type=Path,
        default=None,
        help=(
            "Optional material manifest. When provided, write an agent-ready "
            "material palette summary."
        ),
    )
    parser.add_argument(
        "--materials-usd",
        type=Path,
        default=None,
        help="Optional material library USD path to record in compact context files.",
    )
    parser.add_argument(
        "--root-prim-path",
        default=None,
        help="Optional snapshot root prim path. Defaults to the session root.",
    )
    parser.add_argument(
        "--max-prims",
        type=int,
        default=4096,
        help="Maximum prim paths to include in the snapshot.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--no-properties",
        dest="include_properties",
        action="store_false",
        help="Do not write per-prim properties.",
    )
    parser.add_argument(
        "--no-material-bindings",
        dest="include_material_bindings",
        action="store_false",
        help="Do not write per-prim material bindings.",
    )
    parser.add_argument(
        "--no-path-translations",
        dest="include_path_translations",
        action="store_false",
        help="Do not write inspection-to-source path translations.",
    )
    parser.add_argument(
        "--no-candidate-hints",
        dest="include_candidate_hints",
        action="store_false",
        help="Do not write Workbench visible candidate hints.",
    )
    parser.add_argument(
        "--material-candidate-space",
        choices=[MATERIAL_CANDIDATE_SPACE_SOURCE, MATERIAL_CANDIDATE_SPACE_INSPECTION],
        default=MATERIAL_CANDIDATE_SPACE_SOURCE,
        help=(
            "Canonical material prediction/coverage path space. The default "
            "`source` matches material-agent by predicting authorable source "
            "targets and retaining runtime paths as evidence."
        ),
    )
    parser.add_argument(
        "--skip-instances",
        dest="skip_instances",
        action="store_true",
        default=True,
        help=(
            "Collapse instance-proxy/runtime candidates to authorable source "
            "targets. This matches material-agent's default."
        ),
    )
    parser.add_argument(
        "--include-instances",
        dest="skip_instances",
        action="store_false",
        help="Keep per-instance candidates instead of collapsing them.",
    )
    parser.add_argument(
        "--skip-prototypes",
        dest="skip_prototypes",
        action="store_true",
        default=False,
        help="Skip candidates whose authoring target is a local prototype source.",
    )
    parser.add_argument(
        "--include-prototypes",
        dest="skip_prototypes",
        action="store_false",
        help="Keep prototype/source candidates. This is the default.",
    )
    parser.add_argument(
        "--skip-invisible",
        dest="skip_invisible",
        action="store_true",
        default=False,
        help="Skip invisible candidates. Workbench visible hints already apply effective visibility.",
    )
    parser.add_argument(
        "--include-invisible",
        dest="skip_invisible",
        action="store_false",
        help="Do not add extra invisible filtering beyond Workbench visible hints.",
    )
    parser.add_argument(
        "--respect-existing-material-bindings",
        dest="respect_existing_material_bindings",
        action="store_true",
        default=False,
        help=(
            "Seed coverage from existing material bindings. By default existing "
            "authored appearance is ignored for authoring decisions and kept "
            "only as diagnostic evidence."
        ),
    )
    parser.add_argument(
        "--ignore-existing-material-bindings",
        dest="respect_existing_material_bindings",
        action="store_false",
        help=(
            "Ignore existing authored appearance when building material "
            "authoring seed groups, including material bindings and display "
            "colors. This is the default."
        ),
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not append a trace event.",
    )
    parser.set_defaults(
        include_properties=True,
        include_material_bindings=True,
        include_path_translations=True,
        include_candidate_hints=True,
    )
    return parser


def fetch_snapshot(
    *,
    workbench_url: str,
    session_id: str,
    root_prim_path: str | None = None,
    include_properties: bool = True,
    include_material_bindings: bool = True,
    include_path_translations: bool = True,
    include_candidate_hints: bool = True,
    max_prims: int = 4096,
    timeout: float = 300.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "include_properties": include_properties,
        "include_material_bindings": include_material_bindings,
        "include_path_translations": include_path_translations,
        "include_candidate_hints": include_candidate_hints,
        "max_prims": max_prims,
    }
    if root_prim_path:
        payload["root_prim_path"] = root_prim_path
    try:
        return _client_snapshot_scene(
            workbench_url,
            session_id,
            payload,
            timeout=timeout,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Workbench snapshot request failed: {exc}") from exc


def _normalize_candidate_policy(
    value: MaterialCandidatePolicy | dict[str, Any] | None,
) -> MaterialCandidatePolicy:
    if isinstance(value, MaterialCandidatePolicy):
        return value.normalized()
    if isinstance(value, dict):
        return MaterialCandidatePolicy(
            material_candidate_space=str(
                value.get("material_candidate_space")
                or value.get("candidate_space")
                or MATERIAL_CANDIDATE_SPACE_SOURCE
            ),
            root_prim_path=_optional_string(
                value.get("root_prim_path") or value.get("root_prim")
            ),
            skip_instances=bool(value.get("skip_instances", True)),
            skip_prototypes=bool(value.get("skip_prototypes", False)),
            skip_invisible=bool(value.get("skip_invisible", False)),
        ).normalized()
    return MaterialCandidatePolicy().normalized()


def write_snapshot_artifacts(
    snapshot: dict[str, Any],
    run_dir: Path,
    *,
    materials_yaml: Path | None = None,
    materials_usd: Path | None = None,
    append_trace: bool = True,
    respect_existing_material_bindings: bool = False,
    candidate_policy: MaterialCandidatePolicy | dict[str, Any] | None = None,
) -> dict[str, str]:
    policy = _normalize_candidate_policy(candidate_policy)
    raw_dir = run_dir / "raw"
    trace_dir = run_dir / "trace"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

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
    _write_json(artifacts["scene_snapshot"], snapshot)
    _write_json(
        artifacts["tree_paths"],
        {
            "session_id": snapshot["session_id"],
            "root_prim_path": snapshot.get("root_prim_path"),
            "paths": snapshot.get("paths", []),
            "nodes": [_legacy_tree_node(node) for node in snapshot.get("nodes", [])],
        },
    )
    _write_json(
        artifacts["properties"],
        {
            "session_id": snapshot["session_id"],
            "results": snapshot.get("properties", []),
        },
    )
    _write_json(
        artifacts["material_bindings"],
        {
            "session_id": snapshot["session_id"],
            "results": snapshot.get("material_bindings", []),
        },
    )
    _write_json(
        artifacts["path_translations"],
        {
            "session_id": snapshot["session_id"],
            "results": snapshot.get("path_translations", []),
        },
    )
    candidates = snapshot.get("candidates", [])
    _write_json(
        artifacts["visible_candidates_preliminary"],
        {
            "session_id": snapshot["session_id"],
            "source_usd": snapshot.get("source_scene_path"),
            "inspection_usd": snapshot.get("inspection_scene_path"),
            "candidate_visible_prim_count": len(candidates),
            "candidate_selection_rule": CANDIDATE_SELECTION_RULE,
            "candidates": candidates,
            "excluded_non_candidates": snapshot.get("excluded_non_candidates", []),
            "summary": snapshot.get("summary", {}),
        },
    )
    coverage_candidates = _coverage_candidates(snapshot, policy)
    _write_json(
        artifacts["visible_candidates"],
        _visible_candidate_artifact(snapshot, coverage_candidates, policy),
    )
    context = _material_authoring_context(
        snapshot,
        coverage_candidates=coverage_candidates,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        respect_existing_material_bindings=respect_existing_material_bindings,
        candidate_policy=policy,
    )
    _write_json(artifacts["material_authoring_context"], context)
    _write_json(artifacts["material_palette"], context["material_palette"])
    _write_candidate_table(artifacts["visible_candidate_table"], context["candidates"])
    _write_json(
        artifacts["material_assignment_seed"],
        _material_assignment_seed(context),
    )
    artifacts["material_authoring_context_md"].write_text(
        _material_authoring_context_markdown(context),
        encoding="utf-8",
    )
    if append_trace:
        _append_trace_event(snapshot, artifacts, trace_dir / "events.jsonl")
    return {name: str(path) for name, path in artifacts.items()}


def compact_summary(
    snapshot: dict[str, Any], artifacts: dict[str, str]
) -> dict[str, Any]:
    summary = dict(snapshot.get("summary") or {})
    return {
        "status": "ok",
        "session_id": snapshot.get("session_id"),
        "root_prim_path": snapshot.get("root_prim_path"),
        "prim_count": summary.get("prim_count", len(snapshot.get("paths", []))),
        "candidate_count": summary.get(
            "candidate_count",
            len(snapshot.get("candidates", [])),
        ),
        "coverage_candidate_count": _artifact_count(
            artifacts.get("visible_candidates")
        ),
        "agent_context": artifacts.get("material_authoring_context"),
        "assignment_seed": artifacts.get("material_assignment_seed"),
        "candidate_table": artifacts.get("visible_candidate_table"),
        "material_palette": artifacts.get("material_palette"),
        "ambiguous_translation_count": summary.get("ambiguous_translation_count", 0),
        "truncated": bool(summary.get("truncated", False)),
        "artifacts": artifacts,
    }


def _material_authoring_context(
    snapshot: dict[str, Any],
    *,
    coverage_candidates: list[dict[str, Any]],
    materials_yaml: Path | None,
    materials_usd: Path | None,
    respect_existing_material_bindings: bool,
    candidate_policy: MaterialCandidatePolicy,
) -> dict[str, Any]:
    display_color_by_path = _display_color_by_path(snapshot)
    candidates = [
        _compact_candidate(
            candidate,
            respect_existing_material_bindings=respect_existing_material_bindings,
            display_color_by_path=display_color_by_path,
        )
        for candidate in coverage_candidates
    ]
    material_palette = _load_material_palette(materials_yaml, materials_usd)
    groups = _candidate_groups(
        candidates,
        respect_existing_material_bindings=respect_existing_material_bindings,
    )
    return {
        "schema_version": "content-workbench.material-authoring-context.v1",
        "session_id": snapshot.get("session_id"),
        "root_prim_path": snapshot.get("root_prim_path"),
        "source_scene_path": snapshot.get("source_scene_path"),
        "inspection_scene_path": snapshot.get("inspection_scene_path"),
        "path_space": _artifact_path_space(snapshot, candidate_policy),
        "material_candidate_policy": candidate_policy.as_dict(),
        "material_binding_policy": {
            "respect_existing_material_bindings": respect_existing_material_bindings,
            "description": (
                "Existing material bindings seed preserved coverage groups."
                if respect_existing_material_bindings
                else "Existing material bindings and authored display colors are "
                "cleared in the Workbench inspection session and must not be used "
                "as authoring evidence unless explicitly requested."
            ),
        },
        "summary": {
            **dict(snapshot.get("summary") or {}),
            "candidate_count": len(candidates),
            "preliminary_candidate_count": len(snapshot.get("candidates", [])),
            "candidate_group_count": len(groups),
            "material_palette_count": material_palette["material_count"],
        },
        "usage_guidance": [
            "Use this compact context before inspecting raw scene_snapshot.json.",
            "Use visible_candidate_prims.json as the canonical material coverage universe.",
            "When path_space is source, prim_paths/source_prim_paths are authorable source targets and runtime_prim_paths are visual evidence or preview fan-out.",
            "When path_space is inspection, use runtime_prim_paths for Workbench material_override commands and source_prim_paths only for export/debug context.",
            "Use material_assignment_seed.json as a starting point, then edit only decisions changed by visual evidence.",
            "When existing material bindings are cleared, use authoring families, shape hints, references, and clean-slate renders rather than existing binding or display-color groups as preservation decisions.",
            "Use visible_candidate_table.tsv for path lookup and raw scene_snapshot.json only for targeted follow-up.",
            "Do not convert candidate rows into material assignment commands.",
            "Groups with recommended_coverage_status=preserved_existing are covered without a Workbench command only when existing bindings are explicitly respected.",
            "Material assignment commands are only for visible material families whose current material is wrong or missing.",
        ],
        "candidate_groups": groups,
        "candidates": candidates,
        "material_palette": material_palette,
    }


def _compact_candidate(
    candidate: dict[str, Any],
    *,
    respect_existing_material_bindings: bool,
    display_color_by_path: dict[str, list[float]],
) -> dict[str, Any]:
    inspection_paths = _string_list(candidate.get("inspection_paths"))
    inspection_path = (
        inspection_paths[0]
        if inspection_paths
        else str(candidate.get("inspection_path") or "")
    )
    source_paths = _string_list(candidate.get("source_paths"))
    source_path = (
        _optional_string(candidate.get("source_path"))
        or _first_string(source_paths)
        or inspection_path
    )
    direct_targets = _string_list(candidate.get("direct_targets")) or _string_list(
        candidate.get("preliminary_material_targets")
    )
    material_override = candidate.get("material_override")
    override_material = _override_material(material_override)
    material_path = (
        override_material.get("material_path")
        or _first_string(direct_targets)
        or _optional_string(candidate.get("bound_material_path"))
    )
    material_name = (
        override_material.get("material_name")
        or _material_name_from_path(material_path)
        or "unbound/default"
    )
    bounds_size = _float_list(candidate.get("bounds_size"))
    bounds_center = _float_list(candidate.get("bounds_center"))
    if not bounds_size or not bounds_center:
        samples = candidate.get("bounds_samples")
        if isinstance(samples, list) and samples:
            sample = samples[0]
            if isinstance(sample, dict):
                bounds_size = bounds_size or _float_list(sample.get("size"))
                bounds_center = bounds_center or _float_list(sample.get("center"))
    semantic_hint = _semantic_hint(source_path)
    display_colors = _display_colors_for_paths(
        [*source_paths, source_path, inspection_path],
        display_color_by_path,
    )
    display_color = display_colors[0] if display_colors else None
    display_color_label = _display_color_label(display_color)
    current_appearance_source = (
        "material_binding"
        if material_path
        else "display_color"
        if display_color is not None
        else "unbound_default"
    )
    runtime_path = str(candidate.get("runtime_path") or inspection_path)
    runtime_paths = _string_list(candidate.get("runtime_paths")) or [runtime_path]
    return {
        "runtime_space": str(candidate.get("runtime_space") or "inspection"),
        "runtime_path": runtime_path,
        "runtime_paths": runtime_paths,
        "source_path": source_path,
        "source_paths": source_paths or [source_path],
        "original_source_paths": _string_list(candidate.get("original_source_paths")),
        "inspection_paths": inspection_paths or [inspection_path],
        "inspection_path": inspection_path,
        "name": _path_name(source_path),
        "parent": _parent_path(source_path),
        "path_tokens": _path_tokens(source_path),
        "semantic_hint": semantic_hint,
        "type_name": str(candidate.get("type_name") or ""),
        "candidate_reason": str(candidate.get("candidate_reason") or ""),
        "candidate_reasons": _string_list(candidate.get("candidate_reasons")),
        "material_binding_type": str(candidate.get("material_binding_type") or "none"),
        "material_name": material_name,
        "material_path": material_path,
        "direct_material_paths": direct_targets,
        "binding_source_path": _optional_string(candidate.get("binding_source_path")),
        "current_appearance_source": current_appearance_source,
        "display_color": display_color,
        "display_colors": display_colors,
        "display_color_label": display_color_label,
        "has_material_override": bool(material_override),
        "ambiguous_translation": bool(
            candidate.get("ambiguous_translation", False)
            or candidate.get("translation_ambiguous", False)
        ),
        "source_instance_count": candidate.get("source_instance_count"),
        "instance_collapsed": bool(candidate.get("instance_collapsed", False)),
        "bounds_center": bounds_center,
        "bounds_size": bounds_size,
        "size_hint": _size_hint(bounds_size),
        "shape_hint": _shape_hint(bounds_size),
        "recommended_initial_status": (
            "preserved_existing"
            if respect_existing_material_bindings and material_path
            else "ambiguous_unassigned"
        ),
        "requires_material_assignment": False,
    }


def _candidate_groups(
    candidates: list[dict[str, Any]],
    *,
    respect_existing_material_bindings: bool,
) -> list[dict[str, Any]]:
    by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        key = (
            str(candidate.get("material_path") or candidate.get("material_name"))
            if respect_existing_material_bindings
            else _authoring_group_key(candidate)
        )
        by_material[key].append(candidate)

    groups: list[dict[str, Any]] = []
    for index, (material_key, items) in enumerate(
        sorted(by_material.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    ):
        runtime_paths = sorted(
            {
                runtime_path
                for item in items
                for runtime_path in (
                    _string_list(item.get("runtime_paths"))
                    or [str(item.get("runtime_path") or "")]
                )
                if runtime_path
            }
        )
        source_paths = sorted(
            {
                source_path
                for item in items
                for source_path in _string_list(item.get("source_paths"))
            }
        )
        material_path = _optional_string(items[0].get("material_path"))
        recommended_status = (
            "preserved_existing"
            if respect_existing_material_bindings and material_path
            else "ambiguous_unassigned"
        )
        type_counts = Counter(str(item.get("type_name") or "") for item in items)
        reason_counts = Counter(
            str(item.get("candidate_reason") or "") for item in items
        )
        material_counts = Counter(
            str(item.get("material_name") or "") for item in items
        )
        appearance_source_counts = Counter(
            str(item.get("current_appearance_source") or "") for item in items
        )
        display_color_counts = Counter(
            str(item.get("display_color_label") or "none") for item in items
        )
        semantic_counts = Counter(
            str(item.get("semantic_hint") or "") for item in items
        )
        collapsed_instance_count = sum(
            1 for item in items if item.get("instance_collapsed")
        )
        runtime_evidence_count = sum(
            len(_string_list(item.get("runtime_paths"))) for item in items
        )
        authoring_family = (
            str(items[0].get("material_name") or material_key)
            if respect_existing_material_bindings
            else _authoring_family_label(material_key)
        )
        groups.append(
            {
                "group_id": (
                    f"existing_material_{index + 1:02d}"
                    if respect_existing_material_bindings
                    else f"authoring_family_{index + 1:02d}"
                ),
                "grouping_basis": (
                    "existing_material"
                    if respect_existing_material_bindings
                    else "authoring_family"
                ),
                "authoring_family": authoring_family,
                "material_name": (
                    str(items[0].get("material_name") or material_key)
                    if respect_existing_material_bindings
                    else None
                ),
                "material_path": (
                    material_path if respect_existing_material_bindings else None
                ),
                "existing_material_names": dict(sorted(material_counts.items())),
                "current_appearance_sources": dict(
                    sorted(appearance_source_counts.items())
                ),
                "display_color_counts": dict(sorted(display_color_counts.items())),
                "existing_material_paths": sorted(
                    {
                        str(item.get("material_path"))
                        for item in items
                        if item.get("material_path")
                    }
                ),
                "recommended_coverage_status": recommended_status,
                "requires_material_assignment": False,
                "candidate_count": len(items),
                "runtime_evidence_count": runtime_evidence_count,
                "collapsed_instance_candidate_count": collapsed_instance_count,
                "runtime_space": str(items[0].get("runtime_space") or "inspection"),
                "runtime_paths": runtime_paths,
                "example_runtime_paths": runtime_paths[:8],
                "source_paths": source_paths,
                "example_source_paths": source_paths[:8],
                "inspection_paths": sorted(
                    {
                        inspection_path
                        for item in items
                        for inspection_path in _string_list(
                            item.get("inspection_paths")
                        )
                    }
                ),
                "type_counts": dict(sorted(type_counts.items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "size_hints": dict(
                    sorted(
                        Counter(str(item.get("size_hint")) for item in items).items()
                    )
                ),
                "shape_hints": dict(
                    sorted(
                        Counter(str(item.get("shape_hint")) for item in items).items()
                    )
                ),
                "semantic_hints": dict(sorted(semantic_counts.items())),
                "ambiguous_translation_count": sum(
                    1 for item in items if item.get("ambiguous_translation")
                ),
            }
        )
    return groups


def _display_color_by_path(snapshot: dict[str, Any]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for record in snapshot.get("properties", []):
        if not isinstance(record, dict):
            continue
        prim_path = _optional_string(record.get("prim_path"))
        if not prim_path:
            continue
        properties = record.get("properties")
        if not isinstance(properties, dict):
            continue
        attributes = properties.get("attributes")
        if not isinstance(attributes, dict):
            continue
        for key in ("primvars:displayColor", "displayColor"):
            color = _display_color_from_attribute(attributes.get(key))
            if color is not None:
                result[prim_path] = color
                break
    return result


def _display_colors_for_paths(
    paths: list[str],
    display_color_by_path: dict[str, list[float]],
) -> list[list[float]]:
    colors: list[list[float]] = []
    seen = set()
    for path in paths:
        color = display_color_by_path.get(path)
        if color is None:
            continue
        key = tuple(color)
        if key in seen:
            continue
        seen.add(key)
        colors.append(color)
    return colors


def _display_color_from_attribute(attribute: Any) -> list[float] | None:
    value = attribute.get("value") if isinstance(attribute, dict) else attribute
    if isinstance(value, list) and value:
        return _display_color_triplet(value[0])
    return _display_color_triplet(value)


def _display_color_triplet(value: Any) -> list[float] | None:
    if isinstance(value, list | tuple) and len(value) >= 3:
        try:
            return [round(float(component), 4) for component in value[:3]]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", value, flags=re.I)
        if len(numbers) >= 3:
            try:
                return [round(float(component), 4) for component in numbers[:3]]
            except ValueError:
                return None
    return None


def _display_color_label(color: list[float] | None) -> str | None:
    if color is None or len(color) < 3:
        return None
    red, green, blue = color[:3]
    if max(color[:3]) - min(color[:3]) <= 0.08:
        if max(color[:3]) < 0.28:
            return "dark_gray_display_color"
        if max(color[:3]) > 0.68:
            return "light_gray_display_color"
        return "gray_display_color"
    if green >= red + 0.18 and green >= blue + 0.18:
        if red > 0.55 and blue < 0.2:
            return "yellow_green_display_color"
        return "green_display_color"
    if red >= green + 0.16 and green >= blue + 0.08:
        return "orange_brown_display_color"
    if red >= blue + 0.16 and red >= green + 0.08:
        return "red_display_color"
    if blue >= red + 0.16 and blue >= green + 0.08:
        return "blue_display_color"
    return "mixed_display_color"


def _coverage_candidates(
    snapshot: dict[str, Any],
    policy: MaterialCandidatePolicy,
) -> list[dict[str, Any]]:
    if _artifact_path_space(snapshot, policy) == MATERIAL_CANDIDATE_SPACE_INSPECTION:
        return _runtime_coverage_candidates(snapshot, policy)
    return _source_coverage_candidates(snapshot, policy)


def _source_coverage_candidates(
    snapshot: dict[str, Any],
    policy: MaterialCandidatePolicy,
) -> list[dict[str, Any]]:
    container_material_targets = _container_material_targets(snapshot)
    instance_ref_map = (
        _local_instance_reference_map(
            _optional_string(snapshot.get("source_scene_path"))
        )
        if policy.skip_instances or policy.skip_prototypes
        else {}
    )
    prototype_roots = {
        ref_path for ref_path in instance_ref_map.values() if isinstance(ref_path, str)
    }
    records: dict[str, dict[str, Any]] = {}
    for candidate in snapshot.get("candidates", []):
        if policy.skip_invisible and candidate.get("effective_visible") is False:
            continue
        if str(candidate.get("type_name") or "") != "Mesh":
            continue
        targets = _source_material_targets(candidate)
        if not targets:
            continue
        inspection_path = str(candidate.get("inspection_path") or "")
        candidate_reason = str(candidate.get("candidate_reason") or "")
        direct_targets = _string_list(candidate.get("direct_targets"))
        bound_material_path = _optional_string(candidate.get("bound_material_path"))
        if bound_material_path:
            _append_unique(direct_targets, bound_material_path)
        bounds_center = _float_list(candidate.get("bounds_center"))
        bounds_size = _float_list(candidate.get("bounds_size"))
        ambiguous = bool(candidate.get("ambiguous_translation", False))
        for target in targets:
            original_target = target
            remapped_from_instance = False
            if policy.skip_instances:
                target, remapped_from_instance, skip = _remap_instance_source_target(
                    target,
                    instance_ref_map,
                )
                if skip:
                    continue
            if policy.skip_prototypes and _is_under_any_source_root(
                target,
                prototype_roots,
            ):
                continue
            material_targets = list(direct_targets)
            if not material_targets:
                material_targets = list(
                    container_material_targets.get(target)
                    or container_material_targets.get(original_target)
                    or []
                )
            record = records.setdefault(
                target,
                {
                    "runtime_space": (
                        "inspection" if _session_optimized(snapshot) else "source"
                    ),
                    "runtime_path": inspection_path,
                    "runtime_paths": [],
                    "source_path": target,
                    "source_paths": [target],
                    "original_source_paths": [],
                    "inspection_paths": [],
                    "type_name": "Mesh",
                    "candidate_reasons": [],
                    "preliminary_material_targets": [],
                    "bounds_samples": [],
                    "translation_ambiguous": False,
                    "source_instance_count": 0,
                    "instance_collapsed": False,
                },
            )
            _append_unique(record["runtime_paths"], inspection_path)
            _append_unique(record["inspection_paths"], inspection_path)
            _append_unique(record["original_source_paths"], original_target)
            _append_unique(record["candidate_reasons"], candidate_reason)
            for material_target in material_targets:
                _append_unique(
                    record["preliminary_material_targets"],
                    material_target,
                )
            if bounds_center or bounds_size:
                record["bounds_samples"].append(
                    {
                        "center": bounds_center,
                        "size": bounds_size,
                    }
                )
            record["translation_ambiguous"] = (
                bool(record["translation_ambiguous"]) or ambiguous
            )
            record["source_instance_count"] = len(record["runtime_paths"])
            record["instance_collapsed"] = (
                bool(record["instance_collapsed"]) or remapped_from_instance
            )

    return [records[path] for path in sorted(records)]


def _runtime_coverage_candidates(
    snapshot: dict[str, Any],
    policy: MaterialCandidatePolicy | None = None,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    candidate_entries: list[tuple[dict[str, Any], str, list[str]]] = []
    deduplicated_source_paths: set[str] = set()
    for candidate in snapshot.get("candidates", []):
        if (
            policy is not None
            and policy.skip_invisible
            and candidate.get("effective_visible") is False
        ):
            continue
        if str(candidate.get("type_name") or "") != "Mesh":
            continue
        inspection_path = str(candidate.get("inspection_path") or "")
        if not inspection_path:
            continue
        source_paths = _source_material_targets(candidate) or [inspection_path]
        candidate_entries.append((candidate, inspection_path, source_paths))
        if len(source_paths) > 1:
            deduplicated_source_paths.update(source_paths)

    for candidate, inspection_path, source_paths in candidate_entries:
        if len(source_paths) == 1 and source_paths[0] in deduplicated_source_paths:
            continue
        direct_targets = _string_list(candidate.get("direct_targets"))
        bound_material_path = _optional_string(candidate.get("bound_material_path"))
        if bound_material_path:
            _append_unique(direct_targets, bound_material_path)
        bounds_center = _float_list(candidate.get("bounds_center"))
        bounds_size = _float_list(candidate.get("bounds_size"))
        record = records.setdefault(
            inspection_path,
            {
                "runtime_space": "inspection",
                "runtime_path": inspection_path,
                "runtime_paths": [inspection_path],
                "source_path": source_paths[0],
                "source_paths": [],
                "inspection_path": inspection_path,
                "inspection_paths": [inspection_path],
                "type_name": "Mesh",
                "candidate_reasons": [],
                "preliminary_material_targets": [],
                "bounds_samples": [],
                "translation_ambiguous": False,
                "source_instance_count": 0,
                "deduplicated": False,
            },
        )
        for source_path in source_paths:
            _append_unique(record["source_paths"], source_path)
        record["source_instance_count"] = len(record["source_paths"])
        record["deduplicated"] = len(record["source_paths"]) > 1
        _append_unique(
            record["candidate_reasons"],
            str(candidate.get("candidate_reason") or ""),
        )
        for material_target in direct_targets:
            _append_unique(record["preliminary_material_targets"], material_target)
        if bounds_center or bounds_size:
            record["bounds_samples"].append(
                {
                    "center": bounds_center,
                    "size": bounds_size,
                }
            )
        record["translation_ambiguous"] = bool(
            record["translation_ambiguous"]
            or candidate.get("ambiguous_translation", False)
            or len(record["source_paths"]) > 1
        )

    return [records[path] for path in sorted(records)]


def _container_material_targets(snapshot: dict[str, Any]) -> dict[str, list[str]]:
    by_path: dict[str, list[str]] = defaultdict(list)
    for candidate in snapshot.get("candidates", []):
        if str(candidate.get("type_name") or "") == "Mesh":
            continue
        material_targets = _string_list(candidate.get("direct_targets"))
        bound_material_path = _optional_string(candidate.get("bound_material_path"))
        if bound_material_path:
            _append_unique(material_targets, bound_material_path)
        if not material_targets:
            continue
        paths = _string_list(candidate.get("source_paths")) or [
            str(candidate.get("inspection_path") or "")
        ]
        for path in paths:
            for material_target in material_targets:
                _append_unique(by_path[path], material_target)
    return by_path


def _source_material_targets(candidate: dict[str, Any]) -> list[str]:
    inspection_path = str(candidate.get("inspection_path") or "")
    source_paths = _string_list(candidate.get("source_paths"))
    if inspection_path.endswith("/Geometry"):
        non_geometry_paths = [
            path for path in source_paths if not path.rstrip("/").endswith("/Geometry")
        ]
        if non_geometry_paths:
            return _unique_strings(non_geometry_paths)
        fallback_paths = source_paths or [
            _optional_string(candidate.get("binding_source_path")) or inspection_path
        ]
        return _unique_strings(
            _parent_path(path) if path.rstrip("/").endswith("/Geometry") else path
            for path in fallback_paths
        )
    return _unique_strings(source_paths or [inspection_path])


def _local_instance_reference_map(scene_path: str | None) -> dict[str, str | None]:
    if not scene_path or not Path(scene_path).is_file():
        return {}
    try:
        from pxr import Usd  # type: ignore
    except Exception as exc:
        LOGGER.debug("USD Python bindings unavailable for instance remapping: %s", exc)
        return {}
    try:
        stage = Usd.Stage.Open(scene_path)
    except Exception as exc:
        LOGGER.warning(
            "Could not inspect instance references in %s: %s", scene_path, exc
        )
        return {}
    if not stage:
        return {}
    result: dict[str, str | None] = {}
    for prim in stage.Traverse():
        if not prim.IsInstance():
            continue
        ref_path: str | None = None
        for spec in prim.GetPrimStack():
            added = spec.referenceList.GetAddedOrExplicitItems()
            if added:
                ref = added[0]
                if not ref.assetPath and ref.primPath:
                    ref_path = str(ref.primPath)
                break
        result[str(prim.GetPath())] = (
            _valid_sibling_prototype_root(stage, str(prim.GetPath())) or ref_path
        )
    return result


def _valid_sibling_prototype_root(stage: Any, instance_root: str) -> str | None:
    parts = instance_root.strip("/").split("/")
    if len(parts) < 3:
        return None
    candidate = "/" + "/".join([parts[0], "Prototypes", parts[-1]])
    try:
        prim = stage.GetPrimAtPath(candidate)
    except Exception:
        return None
    if prim and prim.IsValid():
        return candidate
    return None


def _remap_instance_source_target(
    source_path: str,
    instance_root_to_ref_prim: dict[str, str | None],
) -> tuple[str, bool, bool]:
    for instance_root in sorted(instance_root_to_ref_prim, key=len, reverse=True):
        if source_path != instance_root and not source_path.startswith(
            instance_root + "/"
        ):
            continue
        ref_prim = instance_root_to_ref_prim[instance_root]
        if not ref_prim:
            return source_path, False, False
        suffix = source_path[len(instance_root) :]
        return ref_prim + suffix, True, False
    return source_path, False, False


def _is_under_any_source_root(path: str, roots: set[str]) -> bool:
    if path.startswith("/__Prototype_"):
        return True
    for root in roots:
        if path == root or path.startswith(root.rstrip("/") + "/"):
            return True
    return False


def _visible_candidate_artifact(
    snapshot: dict[str, Any],
    candidates: list[dict[str, Any]],
    policy: MaterialCandidatePolicy,
) -> dict[str, Any]:
    path_space = _artifact_path_space(snapshot, policy)
    excluded = set(_string_list(snapshot.get("excluded_non_candidates")))
    for candidate in snapshot.get("candidates", []):
        if str(candidate.get("type_name") or "") == "Mesh":
            continue
        source_paths = _string_list(candidate.get("source_paths"))
        excluded.update(source_paths or [str(candidate.get("inspection_path") or "")])
    if path_space == "inspection":
        excluded.difference_update(
            str(candidate["runtime_path"]) for candidate in candidates
        )
    else:
        excluded.difference_update(
            str(candidate["source_path"]) for candidate in candidates
        )
    excluded.discard("")
    return {
        "session_id": snapshot["session_id"],
        "source_usd": snapshot.get("source_scene_path"),
        "inspection_usd": snapshot.get("inspection_scene_path"),
        "path_space": path_space,
        "material_candidate_policy": policy.as_dict(),
        "candidate_visible_prim_count": len(candidates),
        "candidate_selection_rule": (
            RUNTIME_CANDIDATE_SELECTION_RULE
            if path_space == MATERIAL_CANDIDATE_SPACE_INSPECTION
            else SOURCE_CANDIDATE_SELECTION_RULE
        ),
        "excluded_non_candidates": sorted(excluded),
        "candidates": candidates,
    }


def _artifact_path_space(
    snapshot: dict[str, Any],
    policy: MaterialCandidatePolicy | None = None,
) -> str:
    if policy is not None:
        return policy.normalized().material_candidate_space
    return MATERIAL_CANDIDATE_SPACE_SOURCE


def _session_optimized(snapshot: dict[str, Any]) -> bool:
    optimization = snapshot.get("optimization")
    if isinstance(optimization, dict) and optimization.get("enabled"):
        return True
    return False


def _material_assignment_seed(context: dict[str, Any]) -> dict[str, Any]:
    groups = context["candidate_groups"]
    path_space = str(context.get("path_space") or "source")
    candidate_count = len(context["candidates"])
    preserved_count = sum(
        group["candidate_count"]
        for group in groups
        if group.get("recommended_coverage_status") == "preserved_existing"
    )
    ambiguous_count = sum(
        group["candidate_count"]
        for group in groups
        if group.get("recommended_coverage_status") == "ambiguous_unassigned"
    )
    assignments = []
    appearance_source_counts: Counter[str] = Counter()
    display_color_counts: Counter[str] = Counter()
    for group in groups:
        status = str(group.get("recommended_coverage_status") or "ambiguous_unassigned")
        appearance_source_counts.update(
            {
                str(name): int(count)
                for name, count in (
                    group.get("current_appearance_sources") or {}
                ).items()
            }
        )
        display_color_counts.update(
            {
                str(name): int(count)
                for name, count in (group.get("display_color_counts") or {}).items()
            }
        )
        material_name = str(
            group.get("material_name")
            or group.get("authoring_family")
            or "Existing/default material"
        )
        material_path = group.get("material_path")
        assignments.append(
            {
                "family": f"Seed: {group.get('authoring_family') or material_name}",
                "group_id": group.get("group_id"),
                "coverage_status": status,
                "material_name": material_name
                if status != "ambiguous_unassigned"
                else None,
                "material_path": material_path,
                "grouping_basis": group.get("grouping_basis"),
                "authoring_family": group.get("authoring_family"),
                "semantic_hints": group.get("semantic_hints", {}),
                "shape_hints": group.get("shape_hints", {}),
                "existing_material_names": group.get("existing_material_names", {}),
                "existing_material_paths": group.get("existing_material_paths", []),
                "current_appearance_sources": group.get(
                    "current_appearance_sources",
                    {},
                ),
                "display_color_counts": group.get("display_color_counts", {}),
                "runtime_space": group.get("runtime_space") or "inspection",
                "runtime_prim_paths": group.get("runtime_paths", []),
                "source_prim_paths": group.get("source_paths", []),
                "prim_paths": (
                    group.get("runtime_paths", [])
                    if path_space == "inspection"
                    else group.get("source_paths", [])
                ),
                "rationale": (
                    "Seeded from Workbench's existing material binding evidence. "
                    "Keep as-is when visual review confirms the existing material; "
                    "replace with a material_assignment decision only for changed "
                    "visible material families."
                    if status == "preserved_existing"
                    else "Seeded as an authoring family with existing authored "
                    "appearance ignored. Choose material-library overrides from "
                    "reference/render evidence or leave unresolved when the visible "
                    "family cannot be safely assigned."
                ),
            }
        )
    return {
        "schema_version": "content-agents.material-assignment-seed.v1",
        "session_id": context.get("session_id"),
        "source_usd": context.get("source_scene_path"),
        "inspection_usd": context.get("inspection_scene_path"),
        "path_space": path_space,
        "material_candidate_policy": context.get("material_candidate_policy", {}),
        "library_path": context["material_palette"].get("materials_usd"),
        "per_prim_material_assignment_count": 0,
        "current_appearance_sources": dict(sorted(appearance_source_counts.items())),
        "display_color_counts": dict(sorted(display_color_counts.items())),
        "coverage": {
            "candidate_visible_prim_count": candidate_count,
            "material_decision_prim_count": candidate_count,
            "material_assignment_prim_count": 0,
            "preserved_existing_prim_count": preserved_count,
            "ambiguous_unassigned_prim_count": ambiguous_count,
            "coverage_notes": (
                "Seed coverage generated from raw/visible_candidate_prims.json. "
                "Edit assignments after visual inspection; do not treat preserved "
                "seed rows as Workbench material assignment commands."
            ),
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


def _load_material_palette(
    materials_yaml: Path | None,
    materials_usd: Path | None,
) -> dict[str, Any]:
    if materials_yaml is None:
        return {
            "materials_yaml": None,
            "materials_usd": str(materials_usd) if materials_usd else None,
            "material_count": 0,
            "materials": [],
            "tags": {},
        }

    manifest = yaml.safe_load(materials_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Material manifest must be a YAML mapping: {materials_yaml}")

    entries = manifest.get("entries") or []
    if not isinstance(entries, list):
        raise ValueError(f"Material manifest entries must be a list: {materials_yaml}")

    library_path = materials_usd
    if library_path is None:
        raw_library_path = manifest.get("library_path")
        if isinstance(raw_library_path, str) and raw_library_path:
            library_path = (materials_yaml.parent / raw_library_path).resolve()

    materials = []
    by_tag: dict[str, list[str]] = defaultdict(list)
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        name = str(raw_entry.get("name") or "")
        binding = str(raw_entry.get("binding") or "")
        description = str(raw_entry.get("description") or "")
        tags = _material_tags(name, description)
        semantics = _material_manifest_semantics(name, description)
        material = {
            "name": name,
            "material_path": binding,
            "description": description,
            "tags": tags,
            "manifest_semantics": semantics,
        }
        materials.append(material)
        for tag in tags:
            by_tag[tag].append(name)

    return {
        "materials_yaml": str(materials_yaml),
        "materials_usd": str(library_path) if library_path else None,
        "material_count": len(materials),
        "materials": materials,
        "tags": {tag: sorted(names) for tag, names in sorted(by_tag.items())},
    }


def _material_tags(name: str, description: str) -> list[str]:
    text = f"{name} {description}".lower()
    tags = []
    for tag, needles in {
        "black": ("black", "dark"),
        "blue": ("blue", "cyan"),
        "red": ("red", "ruby"),
        "orange": ("orange",),
        "yellow": ("yellow", "gold"),
        "white": ("white", "ivory"),
        "gray": ("gray", "grey", "silver", "gunmetal"),
        "metal": (
            "metal",
            "steel",
            "aluminum",
            "brass",
            "bronze",
            "copper",
            "iron",
            "gold",
            "silver",
        ),
        "plastic": ("plastic",),
        "rubber": ("rubber", "silicone"),
        "glass": ("glass", "transparent", "clear", "translucent"),
        "paint": ("paint", "automotive"),
        "matte": ("matte", "dull", "rough"),
        "glossy": ("gloss", "polished", "reflective", "mirror"),
        "brushed": ("brushed", "grain", "streak"),
    }.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    return tags


def _material_manifest_semantics(name: str, description: str) -> dict[str, list[str]]:
    text = f"{name} {description}".lower()
    return {
        "colors": _matched_terms(
            text,
            {
                "black": ("black", "charcoal", "jet-black"),
                "blue": ("blue", "navy", "cyan", "turquoise"),
                "brown": ("brown", "russet"),
                "clear": ("clear", "transparent", "colorless"),
                "gold": ("gold", "yellow-gold"),
                "gray": ("gray", "grey", "silver", "gunmetal"),
                "green": ("green",),
                "orange": ("orange",),
                "red": ("red", "ruby"),
                "white": ("white", "ivory"),
                "yellow": ("yellow",),
            },
        ),
        "substances": _matched_terms(
            text,
            {
                "automotive_paint": ("automotive paint", "car paint"),
                "glass": ("glass",),
                "metal": ("metal", "metallic"),
                "plastic": ("plastic",),
                "rubber": ("rubber",),
                "silicone": ("silicone",),
                "steel": ("steel",),
            },
        ),
        "finishes": _matched_terms(
            text,
            {
                "brushed": ("brushed", "grain", "streak"),
                "glossy": ("gloss", "reflective", "polished"),
                "matte": ("matte", "dull", "non-glossy", "rough"),
                "painted": ("paint", "paint-coated", "automotive"),
                "polished": ("polished", "mirror"),
            },
        ),
    }


def _matched_terms(text: str, term_needles: dict[str, tuple[str, ...]]) -> list[str]:
    return [
        term
        for term, needles in term_needles.items()
        if any(needle in text for needle in needles)
    ]


def _write_candidate_table(path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "runtime_path",
        "runtime_paths",
        "runtime_space",
        "source_path",
        "source_paths",
        "original_source_paths",
        "inspection_path",
        "inspection_paths",
        "type_name",
        "material_name",
        "material_path",
        "current_appearance_source",
        "display_color",
        "display_colors",
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
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {field: _tsv_value(candidate.get(field)) for field in fieldnames}
            )


def _material_authoring_context_markdown(context: dict[str, Any]) -> str:
    summary = context["summary"]
    palette = context["material_palette"]
    candidate_policy = context.get("material_candidate_policy") or {}
    lines = [
        "# Material Authoring Context",
        "",
        "Use this compact file before reading raw scene snapshots.",
        "",
        "## Candidate Policy",
        "",
        f"- Path space: {context.get('path_space')}",
        f"- Skip instances: {candidate_policy.get('skip_instances')}",
        f"- Skip prototypes: {candidate_policy.get('skip_prototypes')}",
        f"- Skip invisible: {candidate_policy.get('skip_invisible')}",
        f"- Root prim: {candidate_policy.get('root_prim_path') or 'session root'}",
        "",
        "## Material Binding Policy",
        "",
        str(
            context.get("material_binding_policy", {}).get(
                "description", "Existing material binding policy was not recorded."
            )
        ),
        "",
        "## Counts",
        "",
        f"- Candidates: {summary.get('candidate_count', 0)}",
        f"- Preliminary Workbench hints: {summary.get('preliminary_candidate_count', 0)}",
        f"- Candidate groups: {summary.get('candidate_group_count', 0)}",
        f"- Material palette entries: {summary.get('material_palette_count', 0)}",
        f"- Truncated: {summary.get('truncated', False)}",
        "",
        "## Candidate Groups",
        "",
        "| Group | Basis | Source Candidates | Runtime Evidence | Initial Coverage | Authoring Family | Current Appearance | Existing Materials | Example Runtime Paths | Example Source Paths |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for group in context["candidate_groups"][:40]:
        runtime_examples = "<br>".join(group.get("example_runtime_paths", [])[:4])
        examples = "<br>".join(group["example_source_paths"][:4])
        appearance = ", ".join(
            f"{name} x{count}"
            for name, count in sorted((group.get("display_color_counts") or {}).items())
            if name != "none"
        )
        if not appearance:
            appearance = ", ".join(
                f"{name} x{count}"
                for name, count in sorted(
                    (group.get("current_appearance_sources") or {}).items()
                )
                if name
            )
        existing = ", ".join(
            f"{name or 'unbound'} x{count}"
            for name, count in (group.get("existing_material_names") or {}).items()
        )
        lines.append(
            f"| {group['group_id']} | {group.get('grouping_basis')} | "
            f"{group['candidate_count']} | "
            f"{group.get('runtime_evidence_count', group['candidate_count'])} | "
            f"{group['recommended_coverage_status']} | "
            f"{_md_escape(str(group.get('authoring_family') or group.get('material_name') or ''))} | "
            f"{_md_escape(appearance)} | {_md_escape(existing)} | "
            f"{_md_escape(runtime_examples)} | "
            f"{_md_escape(examples)} |"
        )

    lines.extend(
        [
            "",
            "Rows/groups above are coverage evidence, not a material assignment plan. "
            "When existing bindings are ignored, authoring families are unresolved "
            "until a material decision is made from reference/render evidence.",
            "",
            "## Material Palette",
            "",
            "| Material | Binding | Manifest Semantics | Description |",
            "|---|---|---|---|",
        ]
    )
    for material in palette["materials"]:
        description = str(material.get("description") or "")
        if len(description) > 140:
            description = description[:137] + "..."
        semantics = material.get("manifest_semantics")
        if not isinstance(semantics, dict):
            semantics = {}
        semantic_parts = []
        for label in ("colors", "substances", "finishes"):
            values = semantics.get(label)
            if isinstance(values, list) and values:
                semantic_parts.append(
                    f"{label}={','.join(str(value) for value in values)}"
                )
        semantic_text = "; ".join(semantic_parts)
        lines.append(
            f"| {_md_escape(str(material.get('name') or ''))} | "
            f"{_md_escape(str(material.get('material_path') or ''))} | "
            f"{_md_escape(semantic_text)} | "
            f"{_md_escape(description)} |"
        )

    lines.extend(
        [
            "",
            "## Companion Files",
            "",
            "- `raw/visible_candidate_table.tsv`: one row per Workbench candidate.",
            "- `raw/visible_candidate_prims.json`: canonical material coverage candidates.",
            "- `raw/material_assignment_seed.json`: grouped starting assignments to edit after visual inspection.",
            "- `raw/material_authoring_context.json`: full compact context.",
            "- `raw/material_palette.json`: parsed material library manifest.",
            "- `raw/scene_snapshot.json`: raw source of truth for targeted fallback only.",
            "",
        ]
    )
    return "\n".join(lines)


def _override_material(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    material = value.get("material")
    if not isinstance(material, dict):
        return {}
    name = _optional_string(material.get("name"))
    path = (
        _optional_string(material.get("material_path"))
        or _optional_string(material.get("path"))
        or _optional_string(material.get("binding"))
    )
    return {
        key: material_value
        for key, material_value in {
            "material_name": name,
            "material_path": path,
        }.items()
        if material_value
    }


def _material_name_from_path(path: str | None) -> str | None:
    if not path:
        return None
    return _path_name(path).replace("_", " ")


def _path_name(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1] if path else ""


def _parent_path(path: str) -> str:
    stripped = path.rstrip("/")
    if "/" not in stripped[1:]:
        return ""
    parent = stripped.rsplit("/", 1)[0]
    return parent or "/"


def _size_hint(bounds_size: list[float]) -> str:
    if not bounds_size:
        return "unknown"
    max_extent = max(abs(value) for value in bounds_size)
    if max_extent >= 10.0:
        return "large"
    if max_extent >= 1.0:
        return "medium"
    if max_extent > 0.0:
        return "small"
    return "flat_or_tiny"


def _shape_hint(bounds_size: list[float]) -> str:
    if len(bounds_size) != 3:
        return "unknown"
    dims = sorted(abs(value) for value in bounds_size)
    if dims[2] <= 0.0:
        return "unknown"
    if dims[1] <= 0.0:
        return "thin_or_degenerate"
    long_to_mid = dims[2] / dims[1]
    mid_to_short = dims[1] / max(dims[0], 1e-9)
    if long_to_mid >= 5.0 and mid_to_short <= 2.5:
        return "slender_bar"
    if mid_to_short >= 5.0 and long_to_mid <= 2.5:
        return "thin_panel"
    if long_to_mid <= 2.0 and mid_to_short <= 2.5:
        return "blocky"
    return "irregular"


def _path_tokens(path: str) -> list[str]:
    path_segments = [segment for segment in path.lower().split("/") if segment]
    if len(path_segments) > 1:
        path = "/".join(path_segments[1:])
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", path.lower())
        if token
        and token
        not in {
            "mesh",
            "visual",
            "visuals",
            "collision",
            "collisions",
            "link",
            "default",
            "geometry",
            "rev",
            "left",
            "right",
            "l",
            "r",
        }
        and not token.isdigit()
    ]
    return _unique_strings(tokens)


def _semantic_hint(path: str) -> str:
    tokens = set(_path_tokens(path))
    text = " ".join(tokens)
    rules = [
        ("logo_marking", {"logo", "decal", "label", "badge", "emblem", "marking"}),
        ("head_shell", {"head", "face", "helmet", "camera"}),
        ("hand_gripper", {"hand", "finger", "thumb", "palm", "gripper"}),
        ("foot_ankle", {"foot", "feet", "sole", "toe", "ankle"}),
        ("waist_pelvis", {"waist", "pelvis"}),
        ("hip_joint", {"hip"}),
        ("knee_joint", {"knee"}),
        ("shoulder_joint", {"shoulder"}),
        ("elbow_joint", {"elbow"}),
        ("wrist_joint", {"wrist"}),
        ("torso_shell", {"torso", "chest", "body"}),
        ("wheel_roller", {"wheel", "roller", "caster", "tire"}),
        ("rail_bar", {"rail", "bar", "rod", "shaft", "pin"}),
        ("panel_frame", {"panel", "frame", "cover", "shell", "plate", "lid"}),
        ("fastener", {"screw", "bolt", "nut", "washer", "rivet"}),
    ]
    for hint, needles in rules:
        if tokens & needles:
            return hint
    if "arm" in text or "forearm" in tokens:
        return "arm_shell"
    if tokens & {"leg", "thigh", "shin", "calf"}:
        return "leg_shell"
    return "generic_geometry"


def _authoring_group_key(candidate: dict[str, Any]) -> str:
    semantic_hint = str(candidate.get("semantic_hint") or "generic_geometry")
    shape_hint = str(candidate.get("shape_hint") or "unknown")
    size_hint = str(candidate.get("size_hint") or "unknown")
    display_color_label = str(candidate.get("display_color_label") or "")
    appearance_suffix = (
        f":appearance:{display_color_label}" if display_color_label else ""
    )
    if semantic_hint != "generic_geometry":
        return f"semantic:{semantic_hint}{appearance_suffix}"
    parent = _path_name(str(candidate.get("parent") or ""))
    parent_tokens = [token for token in _path_tokens(parent) if token != "mesh"]
    if parent_tokens:
        return (
            f"path:{'_'.join(parent_tokens[:3])}:{shape_hint}:{size_hint}"
            f"{appearance_suffix}"
        )
    return f"shape:{shape_hint}:{size_hint}"


def _authoring_family_label(group_key: str) -> str:
    _, _, value = group_key.partition(":")
    label = value or group_key
    label = label.replace(":", " ")
    return label.replace("_", " ").strip() or "generic geometry"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _unique_strings(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value:
            _append_unique(result, value)
    return result


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(round(float(item), 6))
        except (TypeError, ValueError):
            return []
    return result


def _first_string(value: list[str]) -> str | None:
    return value[0] if value else None


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _tsv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _artifact_count(path: str | None) -> int | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(value, dict):
        count = value.get("candidate_visible_prim_count")
        if isinstance(count, int):
            return count
        candidates = value.get("candidates")
        if isinstance(candidates, list):
            return len(candidates)
    return None


def _legacy_tree_node(node: dict[str, Any]) -> dict[str, Any]:
    child_paths = list(node.get("child_paths") or [])
    return {
        "path": node.get("path"),
        "name": node.get("name"),
        "type_name": node.get("type_name", ""),
        "active": bool(node.get("active", True)),
        "loaded": bool(node.get("loaded", True)),
        "children": bool(node.get("children", child_paths)),
        "children_count": len(child_paths),
        "child_paths": child_paths,
    }


def _append_trace_event(
    snapshot: dict[str, Any],
    artifacts: dict[str, Path],
    trace_path: Path,
) -> None:
    summary = snapshot.get("summary") or {}
    event = {
        "schema_version": "content-agents.trace.v1",
        "event_type": "api",
        "phase": "scene_snapshot",
        "summary": (
            "Fetched one-call Workbench scene snapshot with hierarchy, properties, "
            "material bindings, path translations, and candidate hints."
        ),
        "artifacts": [str(path) for path in artifacts.values()],
        "data": {
            "api_calls": ["POST /sessions/{session_id}/scene/snapshot"],
            "prim_count": summary.get("prim_count"),
            "candidate_count": summary.get("candidate_count"),
            "ambiguous_translation_count": summary.get("ambiguous_translation_count"),
            "truncated": summary.get("truncated", False),
        },
    }
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
