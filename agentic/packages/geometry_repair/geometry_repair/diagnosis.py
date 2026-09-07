# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic format, topology, B-rep, and profile diagnosis."""

from __future__ import annotations

import re
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .brep import BREP_SUFFIXES, inspect_brep, ocp_available
from .mesh_io import MESH_SUFFIXES, USD_SUFFIXES, audit_asset_intersections, measure_asset
from .models import (
    Diagnosis,
    DiagnosisIssue,
    GeometryMetrics,
    GeometryRole,
    RepairProfile,
)
from .roles import build_diagnosis_role_validation


def _default_issue_roles(
    issue_id: str,
    candidate_repairs: list[str],
) -> list[GeometryRole]:
    if issue_id.startswith("brep:"):
        return ["brep_source"]
    if issue_id == "asset:unresolved_material_dependencies":
        return ["render"]
    if issue_id == "asset:source_collision_geometry_separated" or any(
        "collision" in candidate for candidate in candidate_repairs
    ):
        return ["collision"]
    if issue_id.startswith("semantic:"):
        return ["render", "collision", "helper"]
    if issue_id.startswith("format:") or issue_id in {
        "asset:default_prim_missing",
        "asset:noncanonical_stage_units",
        "asset:noncanonical_up_axis",
        "asset:units_unknown",
        "asset:unresolved_dependencies",
    }:
        return ["render", "collision", "helper"]
    if issue_id.startswith("mesh:") or issue_id == "asset:no_mesh_geometry":
        return ["render", "collision"]
    return ["render"]


def _affected_prim_paths(evidence: dict) -> list[str]:
    paths: set[str] = set()
    for key, value in evidence.items():
        if key.endswith("_prim_paths") and isinstance(value, list):
            paths.update(str(item) for item in value if isinstance(item, str) and item)
        if key == "meshes" and isinstance(value, list):
            paths.update(
                str(item["path"])
                for item in value
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            )
    return sorted(paths)


def _issue(
    issue_id: str,
    *,
    category: str,
    scope: str,
    severity: str,
    confidence: float = 1.0,
    summary: str,
    evidence: dict | None = None,
    candidate_repairs: list[str] | None = None,
    blocking_profiles: list[RepairProfile] | None = None,
    affected_roles: list[GeometryRole] | None = None,
    affected_prim_paths: list[str] | None = None,
) -> DiagnosisIssue:
    issue_evidence = evidence or {}
    repairs = candidate_repairs or []
    return DiagnosisIssue.model_validate(
        {
            "issue_id": issue_id,
            "category": category,
            "scope": scope,
            "severity": severity,
            "confidence": confidence,
            "fact_kind": "measured_fact",
            "summary": summary,
            "evidence": issue_evidence,
            "candidate_repairs": repairs,
            "blocking_profiles": blocking_profiles or [],
            "affected_roles": affected_roles or _default_issue_roles(issue_id, repairs),
            "affected_prim_paths": affected_prim_paths
            if affected_prim_paths is not None
            else _affected_prim_paths(issue_evidence),
        }
    )


def _mesh_issues(metrics: GeometryMetrics, profile: RepairProfile) -> list[DiagnosisIssue]:
    measured = metrics.mesh
    issues: list[DiagnosisIssue] = []
    all_profiles: list[RepairProfile] = [
        "visual_only",
        "static_environment",
        "rigid_pick_place",
        "articulated_rigid",
        "contact_rich",
        "deformable_or_cae",
    ]
    dynamic_profiles: list[RepairProfile] = [
        "rigid_pick_place",
        "articulated_rigid",
        "contact_rich",
        "deformable_or_cae",
    ]
    element_quality_profiles: list[RepairProfile] = ["deformable_or_cae"]
    simulation_profiles: list[RepairProfile] = ["static_environment", *dynamic_profiles]
    if measured.mesh_count == 0:
        issues.append(
            _issue(
                "asset:no_mesh_geometry",
                category="format",
                scope="asset",
                severity="error",
                summary="No renderable mesh geometry was found.",
                blocking_profiles=all_profiles,
            )
        )
    if metrics.instance_proxy_mesh_count:
        issues.append(
            _issue(
                "asset:instance_proxy_geometry",
                category="semantic",
                scope="asset",
                severity="info",
                summary=(
                    "Renderable geometry includes read-only USD instance proxies; "
                    "deinstance only when a later repair must mutate those meshes."
                ),
                evidence={"mesh_count": metrics.instance_proxy_mesh_count},
                candidate_repairs=["scene_optimizer_deinstance"],
            )
        )
    if metrics.collision_source_mesh_count:
        issues.append(
            _issue(
                "asset:source_collision_geometry_separated",
                category="semantic",
                scope="asset",
                severity="info",
                summary=(
                    "Existing collision/proxy meshes were separated from render-geometry "
                    "diagnosis and retained in the immutable source package."
                ),
                evidence={
                    "mesh_count": metrics.collision_source_mesh_count,
                    "paths": metrics.source_collision_paths,
                },
            )
        )
    zero_thickness_meshes = [
        {
            "path": record.path,
            "geometric_dimension": record.metrics.geometric_dimension,
            "minimum_bbox_extent_m": record.metrics.minimum_bbox_extent_m,
            "enclosed_volume_m3": record.metrics.enclosed_volume_m3,
            "watertight": record.metrics.watertight,
        }
        for record in metrics.meshes
        if record.metrics.zero_thickness_status == "fail"
    ]
    if zero_thickness_meshes:
        issues.append(
            _issue(
                "mesh:zero_thickness_geometry",
                category="topology",
                scope="asset",
                severity="error",
                summary=(
                    "One or more closed meshes collapse spatially or enclose no measurable "
                    "volume; they cannot be treated as simulation solids."
                ),
                evidence={"meshes": zero_thickness_meshes},
                candidate_repairs=[
                    "confirm_sheet_or_shell_intent",
                    "reconstruct_or_thicken_from_authoritative_dimensions",
                ],
                blocking_profiles=simulation_profiles,
            )
        )
    unconfirmed_sheet_meshes = [
        {
            "path": record.path,
            "geometric_dimension": record.metrics.geometric_dimension,
            "minimum_bbox_extent_m": record.metrics.minimum_bbox_extent_m,
            "boundary_edge_count": record.metrics.boundary_edge_count,
            "watertight": record.metrics.watertight,
        }
        for record in metrics.meshes
        if record.metrics.zero_thickness_status == "not_evaluated"
        and record.metrics.geometric_dimension is not None
        and record.metrics.geometric_dimension < 3
    ]
    if unconfirmed_sheet_meshes:
        issues.append(
            _issue(
                "mesh:sheet_intent_unconfirmed",
                category="semantic",
                scope="asset",
                severity="warning",
                summary=(
                    "One or more open meshes are geometrically lower-dimensional. Their "
                    "sheet, shell, membrane, or missing-thickness intent must be confirmed."
                ),
                evidence={"meshes": unconfirmed_sheet_meshes},
                candidate_repairs=[
                    "confirm_sheet_or_shell_intent",
                    "supply_authoritative_thickness_or_boundary_conditions",
                ],
                blocking_profiles=[
                    "rigid_pick_place",
                    "articulated_rigid",
                    "contact_rich",
                ],
            )
        )
    for field, issue_id, summary in (
        (
            "invalid_index_count",
            "mesh:invalid_indices",
            "Mesh indices are malformed or out of range.",
        ),
        (
            "non_finite_vertex_count",
            "mesh:non_finite_vertices",
            "Mesh vertices contain NaN or infinity.",
        ),
        (
            "non_finite_normal_count",
            "mesh:non_finite_normals",
            "Authored mesh normals contain NaN or infinity.",
        ),
        (
            "non_finite_uv_count",
            "mesh:non_finite_uvs",
            "Authored mesh texture coordinates contain NaN or infinity.",
        ),
        (
            "non_finite_transform_count",
            "mesh:non_finite_transforms",
            "Mesh transforms contain NaN or infinity.",
        ),
        (
            "singular_transform_count",
            "mesh:singular_transforms",
            "Mesh transforms collapse one or more spatial dimensions.",
        ),
    ):
        count = int(getattr(measured, field))
        if count:
            issues.append(
                _issue(
                    issue_id,
                    category="numeric",
                    scope="asset",
                    severity="error",
                    summary=summary,
                    evidence={"count": count},
                    blocking_profiles=all_profiles,
                )
            )
    for field, issue_id, summary in (
        (
            "degenerate_face_count",
            "mesh:degenerate_faces",
            "Zero-area or needle triangles were detected.",
        ),
        ("duplicate_face_count", "mesh:duplicate_faces", "Duplicate faces were detected."),
        (
            "inconsistent_orientation_edge_count",
            "mesh:inconsistent_orientation",
            "Adjacent faces use inconsistent edge orientation.",
        ),
    ):
        count = int(getattr(measured, field))
        if count:
            issues.append(
                _issue(
                    issue_id,
                    category="topology",
                    scope="asset",
                    severity="error",
                    summary=summary,
                    evidence={"count": count},
                    candidate_repairs=["trimesh_conservative_cleanup"],
                    blocking_profiles=all_profiles,
                )
            )
    if measured.over_connected_edge_count:
        issues.append(
            _issue(
                "mesh:over_connected_edges",
                category="topology",
                scope="asset",
                severity="error",
                summary="Edges incident to more than two faces were detected.",
                evidence={"count": measured.over_connected_edge_count},
                candidate_repairs=["exact_local_mesh_repair"],
                blocking_profiles=all_profiles,
            )
        )
    if measured.non_manifold_vertex_count:
        issues.append(
            _issue(
                "mesh:non_manifold_vertices",
                category="topology",
                scope="asset",
                severity="error",
                summary="Vertices with disconnected incident face fans were detected.",
                evidence={"count": measured.non_manifold_vertex_count},
                candidate_repairs=["exact_local_mesh_repair"],
                blocking_profiles=all_profiles,
            )
        )
    if measured.boundary_edge_count:
        issues.append(
            _issue(
                "mesh:boundary_edges",
                category="topology",
                scope="asset",
                severity="warning",
                summary="Open boundary edges were detected; they may be intentional sheets or holes.",
                evidence={"count": measured.boundary_edge_count},
                candidate_repairs=[
                    "classify_sheet_or_protected_opening",
                    "exact_local_hole_repair",
                ],
                blocking_profiles=dynamic_profiles,
            )
        )
    if measured.duplicate_vertex_count:
        issues.append(
            _issue(
                "mesh:coincident_vertices",
                category="topology",
                scope="asset",
                severity="info",
                summary=(
                    "Coincident source vertices were retained; positional identity was used only "
                    "for non-mutating seam topology analysis."
                ),
                evidence={
                    "count": measured.duplicate_vertex_count,
                    "indexed_boundary_edges": measured.indexed_boundary_edge_count,
                    "geometric_boundary_edges": measured.boundary_edge_count,
                },
                candidate_repairs=["review_bounded_vertex_weld"],
            )
        )
    if measured.near_duplicate_vertex_count:
        issues.append(
            _issue(
                "mesh:near_duplicate_vertices",
                category="topology",
                scope="asset",
                severity="info",
                summary="Near-coincident vertices were measured but were not welded automatically.",
                evidence={"count": measured.near_duplicate_vertex_count},
                candidate_repairs=["review_bounded_vertex_weld"],
            )
        )
    if measured.unused_vertex_count:
        issues.append(
            _issue(
                "mesh:unused_vertices",
                category="topology",
                scope="asset",
                severity="info",
                summary="Vertices not referenced by any valid face were detected.",
                evidence={"count": measured.unused_vertex_count},
                candidate_repairs=["trimesh_conservative_cleanup"],
            )
        )
    if measured.needle_triangle_count:
        issues.append(
            _issue(
                "mesh:needle_triangles",
                category="topology",
                scope="asset",
                severity="warning",
                summary="Extremely thin or high-aspect-ratio triangles were detected.",
                evidence={
                    "count": measured.needle_triangle_count,
                    "aspect_ratio_p95": measured.triangle_aspect_ratio_p95,
                    "aspect_ratio_max": measured.triangle_aspect_ratio_max,
                },
                candidate_repairs=["geogram_local_repair"],
                blocking_profiles=element_quality_profiles,
            )
        )
    if measured.tiny_component_count:
        issues.append(
            _issue(
                "mesh:tiny_components",
                category="semantic",
                scope="asset",
                severity="warning",
                confidence=0.8,
                summary=(
                    "Tiny disconnected components were found; they may be debris or intentional "
                    "functional parts and are never deleted without explicit intent."
                ),
                evidence={
                    "count": measured.tiny_component_count,
                    "smallest_face_count": measured.smallest_component_face_count,
                    "smallest_area_ratio": measured.smallest_component_area_ratio,
                },
                candidate_repairs=["classify_component_intent"],
            )
        )
    if measured.inverted_shell_count:
        issues.append(
            _issue(
                "mesh:inverted_shells",
                category="topology",
                scope="asset",
                severity="error",
                summary="Closed shells with inward-facing orientation were detected.",
                evidence={"count": measured.inverted_shell_count},
                candidate_repairs=["trimesh_conservative_cleanup"],
                blocking_profiles=all_profiles,
            )
        )
    elif measured.inverted_shell_status == "not_evaluated":
        issues.append(
            _issue(
                "mesh:inverted_shells_not_evaluated",
                category="topology",
                scope="asset",
                severity="warning",
                summary="Shell-orientation checking exceeded its bounded component scope.",
                candidate_repairs=["approved_exact_intersection_worker"],
                blocking_profiles=dynamic_profiles,
            )
        )
    if measured.nested_shell_status == "fail":
        issues.append(
            _issue(
                "mesh:nested_shells",
                category="topology",
                scope="asset",
                severity="warning",
                confidence=0.9,
                summary="One or more closed shells are nested inside another shell.",
                evidence={"count": measured.nested_shell_count},
                candidate_repairs=["classify_internal_shell_intent"],
                blocking_profiles=dynamic_profiles,
            )
        )
    elif measured.nested_shell_status == "not_evaluated" and measured.connected_component_count > 1:
        issues.append(
            _issue(
                "mesh:nested_shells_not_evaluated",
                category="topology",
                scope="asset",
                severity="warning",
                summary="Nested-shell classification could not be completed within the local kernel.",
                candidate_repairs=["approved_exact_containment_worker"],
                blocking_profiles=dynamic_profiles,
            )
        )
    if measured.coplanar_overlap_status == "fail":
        issues.append(
            _issue(
                "mesh:coplanar_overlaps",
                category="topology",
                scope="asset",
                severity="error",
                summary="Non-adjacent coplanar triangles overlap with positive area.",
                evidence={"count": measured.coplanar_overlap_count},
                candidate_repairs=["geogram_local_repair"],
                blocking_profiles=all_profiles,
            )
        )
    elif measured.coplanar_overlap_status == "not_evaluated":
        issues.append(
            _issue(
                "mesh:coplanar_overlaps_not_evaluated",
                category="topology",
                scope="asset",
                severity="warning",
                summary="Coplanar-overlap checking exceeded its bounded evaluation scope.",
                candidate_repairs=["approved_exact_intersection_worker"],
                blocking_profiles=dynamic_profiles,
            )
        )
    if measured.non_uniform_transform_count or measured.sheared_transform_count:
        issues.append(
            _issue(
                "mesh:physics_incompatible_transforms",
                category="semantic",
                scope="asset",
                severity="warning",
                summary=(
                    "Non-uniform or sheared transforms require explicit baking for portable "
                    "collision geometry."
                ),
                evidence={
                    "non_uniform_count": measured.non_uniform_transform_count,
                    "sheared_count": measured.sheared_transform_count,
                },
                candidate_repairs=["bake_collision_working_copy_transforms"],
            )
        )
    if measured.reflected_transform_count:
        issues.append(
            _issue(
                "mesh:reflected_transforms",
                category="semantic",
                scope="asset",
                severity="warning",
                summary="Negative-determinant transforms may invert collision winding.",
                evidence={"count": measured.reflected_transform_count},
                candidate_repairs=["bake_collision_working_copy_transforms"],
            )
        )
    if measured.non_triangular_face_count:
        issues.append(
            _issue(
                "mesh:non_triangular_faces",
                category="topology",
                scope="asset",
                severity="info",
                summary="Non-triangular faces are present and retained as source geometry.",
                evidence={"count": measured.non_triangular_face_count},
            )
        )
    if metrics.unresolved_geometry_dependency_count:
        issues.append(
            _issue(
                "asset:unresolved_dependencies",
                category="format",
                scope="asset",
                severity="error",
                summary="One or more referenced source dependencies could not be resolved.",
                evidence={"count": metrics.unresolved_geometry_dependency_count},
                blocking_profiles=all_profiles,
            )
        )
    if metrics.unresolved_material_dependency_count:
        issues.append(
            _issue(
                "asset:unresolved_material_dependencies",
                category="format",
                scope="asset",
                severity="warning",
                summary=(
                    "One or more material or texture dependencies are unresolved; geometry can "
                    "continue, but visual-material readiness remains downstream-owned."
                ),
                evidence={"count": metrics.unresolved_material_dependency_count},
            )
        )
    if metrics.default_prim_path is None and metrics.source_format in {
        "usd",
        "usda",
        "usdc",
        "usdz",
    }:
        ambiguous = metrics.root_count != 1
        issues.append(
            _issue(
                "asset:default_prim_missing",
                category="format",
                scope="asset",
                severity="error" if ambiguous else "warning",
                summary=(
                    "USD default prim is missing and the root is ambiguous."
                    if ambiguous
                    else "USD default prim is missing but can be assigned to the sole root."
                ),
                evidence={"root_count": metrics.root_count},
                candidate_repairs=["usd_structure_repair"] if not ambiguous else [],
                blocking_profiles=all_profiles if ambiguous else simulation_profiles,
            )
        )
    if metrics.meters_per_unit is None:
        issues.append(
            _issue(
                "asset:units_unknown",
                category="semantic",
                scope="asset",
                severity="info" if profile == "visual_only" else "warning",
                summary="Physical units are not explicitly established.",
                candidate_repairs=["confirm_and_normalize_stage_metrics"],
                blocking_profiles=simulation_profiles,
            )
        )
    if metrics.up_axis and metrics.up_axis.upper() != "Z":
        issues.append(
            _issue(
                "asset:noncanonical_up_axis",
                category="semantic",
                scope="asset",
                severity="info" if profile == "visual_only" else "warning",
                summary=f"Stage up axis is {metrics.up_axis}; canonical repair output is Z-up.",
                evidence={"up_axis": metrics.up_axis},
                candidate_repairs=["usd_structure_repair"],
                blocking_profiles=simulation_profiles,
            )
        )
    if metrics.meters_per_unit is not None and abs(metrics.meters_per_unit - 1.0) > 1e-12:
        issues.append(
            _issue(
                "asset:noncanonical_stage_units",
                category="semantic",
                scope="asset",
                severity="info" if profile == "visual_only" else "warning",
                summary=(
                    f"Stage metersPerUnit is {metrics.meters_per_unit}; canonical repair output uses 1.0."
                ),
                evidence={"meters_per_unit": metrics.meters_per_unit},
                candidate_repairs=["usd_structure_repair"],
                blocking_profiles=simulation_profiles,
            )
        )
    if measured.self_intersection_status == "fail":
        issues.append(
            _issue(
                "mesh:self_intersections",
                category="topology",
                scope="asset",
                severity="error",
                summary="Non-adjacent triangles intersect within a source mesh.",
                evidence={
                    "count": measured.self_intersection_count,
                    "broad_phase_pairs": measured.self_intersection_broad_phase_pairs,
                    "candidate_pairs": measured.self_intersection_candidate_pairs,
                },
                candidate_repairs=["approved_exact_local_mesh_repair"],
                blocking_profiles=all_profiles,
            )
        )
    elif measured.self_intersection_status == "not_evaluated":
        issues.append(
            _issue(
                "mesh:self_intersections_not_evaluated",
                category="topology",
                scope="asset",
                severity="warning",
                summary="Exact self-intersection checking exceeded its bounded evaluation scope.",
                evidence={"reason": measured.self_intersection_reason},
                candidate_repairs=["approved_exact_intersection_worker"],
                blocking_profiles=dynamic_profiles,
            )
        )
    if metrics.inter_part_intersection_status == "fail":
        issues.append(
            _issue(
                "mesh:inter_part_intersections",
                category="semantic",
                scope="asset",
                severity="warning",
                confidence=0.75,
                summary=(
                    "Triangles from distinct source parts intersect; intent is ambiguous and "
                    "must not be repaired by silently fusing or deleting parts."
                ),
                evidence={"count": metrics.inter_part_intersection_count},
                candidate_repairs=["classify_inter_part_contact_intent"],
            )
        )
    elif metrics.inter_part_intersection_status == "not_evaluated":
        issues.append(
            _issue(
                "mesh:inter_part_intersections_not_evaluated",
                category="semantic",
                scope="asset",
                severity="warning",
                summary="Inter-part intersection checking exceeded its bounded evaluation scope.",
                evidence={"reason": metrics.inter_part_intersection_reason},
            )
        )
    return issues


def _brep_diagnosis(
    source: Path, profile: RepairProfile
) -> tuple[GeometryMetrics, list[DiagnosisIssue]]:
    available, unavailable_reason = ocp_available()
    if not available:
        metrics = GeometryMetrics(source_format=source.suffix.lower().lstrip("."))
        return metrics, [
            _issue(
                "brep:backend_unavailable",
                category="brep",
                scope="asset",
                severity="error",
                summary=(
                    "Native B-rep inspection and repair are unavailable in this runtime. "
                    "Supply a validated USD or mesh representation from the authoring provider."
                ),
                evidence={"reason": unavailable_reason or "OCP runtime is unavailable"},
                blocking_profiles=[profile],
            )
        ]
    _, brep = inspect_brep(source)
    metrics = GeometryMetrics(source_format=source.suffix.lower().lstrip("."), brep=brep)
    if brep.error:
        return metrics, [
            _issue(
                "brep:read_failed",
                category="brep",
                scope="asset",
                severity="error",
                summary="OpenCascade could not read the native CAD source.",
                evidence={"error": brep.error},
                blocking_profiles=[profile],
            )
        ]
    detail_issues: list[DiagnosisIssue] = []
    for status, count in sorted(brep.validity_status_counts.items()):
        if status == "NoError" or count <= 0:
            continue
        slug = re.sub(r"(?<!^)(?=[A-Z])", "_", status).lower().replace(":", "_")
        affected_subshapes = {
            label: counts[status]
            for label, counts in brep.subshape_status_counts.items()
            if counts.get(status, 0) > 0
        }
        detail_issues.append(
            _issue(
                f"brep:status:{slug}",
                category="brep",
                scope="asset",
                severity="warning",
                summary=f"OpenCascade reported {count} {status} B-rep validity finding(s).",
                evidence={
                    "status": status,
                    "count": count,
                    "affected_subshape_types": affected_subshapes,
                },
                candidate_repairs=["ocp_shape_heal"],
            )
        )
    if brep.tiny_edge_count:
        detail_issues.append(
            _issue(
                "brep:tiny_edges",
                category="brep",
                scope="asset",
                severity="warning",
                summary=f"Native CAD contains {brep.tiny_edge_count} tolerance-scale edge(s).",
                evidence={
                    "minimum_edge_length_source_units": (brep.minimum_edge_length_source_units),
                    "threshold_source_units": brep.tiny_feature_threshold_source_units,
                },
                candidate_repairs=["confirm_protected_features_before_defeaturing"],
            )
        )
    if brep.tiny_face_count:
        detail_issues.append(
            _issue(
                "brep:tiny_faces",
                category="brep",
                scope="asset",
                severity="warning",
                summary=f"Native CAD contains {brep.tiny_face_count} tolerance-scale face(s).",
                evidence={
                    "minimum_face_area_source_units2": brep.minimum_face_area_source_units2,
                    "threshold_source_units": brep.tiny_feature_threshold_source_units,
                },
                candidate_repairs=["confirm_protected_features_before_defeaturing"],
            )
        )
    if brep.valid:
        return metrics, detail_issues
    return metrics, [
        _issue(
            "brep:invalid_shape",
            category="brep",
            scope="asset",
            severity="error",
            summary="OpenCascade BRepCheck reports an invalid native CAD shape.",
            evidence=brep.model_dump(mode="json"),
            candidate_repairs=["ocp_shape_heal"],
            blocking_profiles=[profile],
        ),
        *detail_issues,
    ]


def _usd_semantic_issues(
    facts: dict[str, list[str]],
    metrics: GeometryMetrics,
    profile: RepairProfile,
) -> list[DiagnosisIssue]:
    joints = facts.get("joint_prim_paths", [])
    rigid_bodies = facts.get("rigid_body_prim_paths", [])
    time_varying = facts.get("time_varying_prim_paths", [])
    geometry_time_varying = facts.get(
        "geometry_time_varying_prim_paths",
        time_varying,
    )
    skeletons = facts.get("skeleton_prim_paths", [])
    animations = facts.get("animation_prim_paths", [])
    brep_prim_paths = facts.get("brep_prim_paths", [])
    issues: list[DiagnosisIssue] = []
    dynamic_evidence = bool(joints or geometry_time_varying or skeletons or animations)
    if profile in {"rigid_pick_place", "static_environment"} and dynamic_evidence:
        issues.append(
            _issue(
                "semantic:dynamic_source_conflicts_with_rigid_profile",
                category="semantic",
                scope="asset",
                severity="error",
                summary="Authored joints, animation, skeletons, or time samples conflict with the selected rigid profile.",
                evidence={
                    "joint_prim_paths": joints,
                    "time_varying_prim_paths": time_varying,
                    "geometry_time_varying_prim_paths": geometry_time_varying,
                    "skeleton_prim_paths": skeletons,
                    "animation_prim_paths": animations,
                },
                blocking_profiles=[profile],
            )
        )
    if profile == "articulated_rigid":
        render_part_count = len(metrics.source_part_paths)
        if joints and (len(rigid_bodies) < 2 or render_part_count < 2):
            issues.append(
                _issue(
                    "semantic:fused_or_incomplete_articulated_links",
                    category="semantic",
                    scope="asset",
                    severity="error",
                    summary="The articulated source has joint evidence but fewer than two independently represented rigid/render parts.",
                    evidence={
                        "joint_prim_paths": joints,
                        "rigid_body_prim_paths": rigid_bodies,
                        "render_part_count": render_part_count,
                    },
                    blocking_profiles=["articulated_rigid"],
                )
            )
        elif not joints:
            issues.append(
                _issue(
                    "semantic:articulation_source_authority_missing",
                    category="semantic",
                    scope="asset",
                    severity="warning",
                    summary="The articulated profile lacks authored USD joint evidence and requires an authoritative external link/joint mapping.",
                    evidence={
                        "rigid_body_prim_paths": rigid_bodies,
                        "render_part_count": render_part_count,
                    },
                )
            )
    if brep_prim_paths:
        issues.append(
            _issue(
                "brep:validation_not_evaluated",
                category="brep",
                scope="asset",
                severity="error",
                summary=(
                    "Authored USD B-rep geometry requires an available exact "
                    "validator before this asset can be certified."
                ),
                evidence={"brep_prim_paths": brep_prim_paths},
                blocking_profiles=[profile],
            )
        )
    return issues


def diagnose_asset(
    source_path: str | Path,
    *,
    profile: RepairProfile,
    output_path: str | Path | None = None,
    unresolved_dependency_count: int = 0,
    unresolved_geometry_dependency_count: int = 0,
    unresolved_material_dependency_count: int = 0,
    scalable_audit_budget: object | None = None,
    run_scalable_audit: bool = True,
    usd_semantic_facts: dict[str, list[str]] | None = None,
) -> Diagnosis:
    """Diagnose one source without mutating it."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Geometry diagnosis source is not a file: {source}")
    suffix = source.suffix.lower()
    scalable_audit_path: Path | None = None
    if suffix in USD_SUFFIXES | MESH_SUFFIXES:
        metrics = measure_asset(
            source,
            unresolved_dependency_count=unresolved_dependency_count,
            unresolved_geometry_dependency_count=unresolved_geometry_dependency_count,
            unresolved_material_dependency_count=unresolved_material_dependency_count,
        )
        if run_scalable_audit:
            scalable_audit_path = (
                Path(output_path).with_name(f"{Path(output_path).stem}.scalable-audit.json")
                if output_path is not None
                else None
            )
            scalable, _payload = audit_asset_intersections(
                source,
                output_path=scalable_audit_path,
                budget=scalable_audit_budget,
            )
            complete = scalable.self_intersection.complete
            within_self = scalable.self_intersection.within_part_count
            within_coplanar = scalable.coplanar_overlap.within_part_count
            cross_self = scalable.self_intersection.cross_part_count
            audit_reason = (
                scalable.self_intersection.reason or scalable.coplanar_overlap.reason or None
            )
            metrics.mesh.self_intersection_status = (
                "fail" if within_self else "pass" if complete else "not_evaluated"
            )
            metrics.mesh.self_intersection_count = within_self
            metrics.mesh.self_intersection_broad_phase_pairs = (
                scalable.resources.broad_pairs_examined
            )
            metrics.mesh.self_intersection_candidate_pairs = scalable.resources.exact_pair_tests
            metrics.mesh.self_intersection_reason = audit_reason if not complete else None
            metrics.mesh.coplanar_overlap_status = (
                "fail" if within_coplanar else "pass" if complete else "not_evaluated"
            )
            metrics.mesh.coplanar_overlap_count = within_coplanar
            metrics.self_intersection_status = metrics.mesh.self_intersection_status
            metrics.inter_part_intersection_status = (
                "fail" if cross_self else "pass" if complete else "not_evaluated"
            )
            metrics.inter_part_intersection_count = cross_self
            metrics.inter_part_intersection_reason = audit_reason if not complete else None
        issues = _mesh_issues(metrics, profile)
        if suffix in USD_SUFFIXES and usd_semantic_facts is not None:
            issues.extend(_usd_semantic_issues(usd_semantic_facts, metrics, profile))
    elif suffix in BREP_SUFFIXES:
        metrics, issues = _brep_diagnosis(source, profile)
    else:
        metrics = GeometryMetrics(source_format=suffix.lstrip(".") or "unknown")
        issues = [
            _issue(
                "format:unsupported",
                category="format",
                scope="asset",
                severity="error",
                summary=f"Geometry repair does not support {suffix or 'this source format'} directly.",
                evidence={"suffix": suffix},
                blocking_profiles=[profile],
            )
        ]
    blocking = [item.issue_id for item in issues if profile in item.blocking_profiles]
    repairable = [item.issue_id for item in issues if item.candidate_repairs]
    status = (
        "fail"
        if blocking
        else "conditional"
        if any(item.severity == "warning" for item in issues)
        else "pass"
    )
    diagnosis_path = str(Path(output_path).resolve()) if output_path else None
    diagnosis = Diagnosis(
        source_path=str(source),
        source_sha256=file_sha256(source),
        profile=profile,
        metrics=metrics,
        issues=issues,
        blocking_issue_ids=blocking,
        repairable_issue_ids=repairable,
        role_validation=build_diagnosis_role_validation(
            profile=profile,
            metrics=metrics,
            issues=issues,
            diagnosis_path=diagnosis_path,
        ),
        status=status,
        report_path=diagnosis_path,
        scalable_audit_path=str(scalable_audit_path.resolve()) if scalable_audit_path else None,
    )
    if output_path is not None:
        atomic_write_json(output_path, diagnosis)
    return diagnosis
