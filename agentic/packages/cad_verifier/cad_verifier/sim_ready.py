# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulation-readiness checks for CAD Agent outputs.

These checks are intentionally static and fast. They do not replace Isaac Sim
or Newton runtime validation; they catch upstream CAD issues that make assets
expensive to use in robotics workflows.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from .checks import all_checks

SIM_READY_PROFILE_ID = "geometry-agent.isaac-openusd.v1"
STATIC_VISUAL_PROFILE_ID = "geometry-agent.static-visual-asset.v1"
RIGID_DYNAMIC_PROFILE_ID = "geometry-agent.rigid-dynamic-asset.v1"
ARTICULATED_PROFILE_ID = "geometry-agent.articulated-asset.v1"
ROBOTICS_MANIPULATION_PROFILE_ID = "geometry-agent.robotics-manipulation-asset.v1"
INSERTION_FIXTURE_PROFILE_ID = "geometry-agent.insertion-or-fixture-asset.v1"

OFFICIAL_SIMREADY_DEFAULT_PROFILE = "Prop-Robotics-Neutral"
OFFICIAL_SIMREADY_DEFAULT_VERSION = "1.0.0"
OFFICIAL_SIMREADY_PROFILE_ALIASES = {
    SIM_READY_PROFILE_ID: OFFICIAL_SIMREADY_DEFAULT_PROFILE,
    STATIC_VISUAL_PROFILE_ID: OFFICIAL_SIMREADY_DEFAULT_PROFILE,
    RIGID_DYNAMIC_PROFILE_ID: "Prop-Robotics-Physx",
    INSERTION_FIXTURE_PROFILE_ID: "Prop-Robotics-Physx",
    ARTICULATED_PROFILE_ID: "Robot-Body-Runnable",
    ROBOTICS_MANIPULATION_PROFILE_ID: "Robot-Body-Runnable",
    "static_visual_asset": OFFICIAL_SIMREADY_DEFAULT_PROFILE,
    "rigid_dynamic_asset": "Prop-Robotics-Physx",
    "insertion_or_fixture_asset": "Prop-Robotics-Physx",
    "articulated_asset": "Robot-Body-Runnable",
    "robotics_manipulation_asset": "Robot-Body-Runnable",
}


def resolve_official_simready_profile(
    *,
    local_profile_id: str | None = None,
    asset_type: str | None = None,
    official_profile: str | None = None,
) -> str:
    """Resolve a CAD intent alias without replacing Foundation profile IDs."""

    if official_profile:
        return official_profile
    for key in (local_profile_id, asset_type):
        if key and str(key) in OFFICIAL_SIMREADY_PROFILE_ALIASES:
            return OFFICIAL_SIMREADY_PROFILE_ALIASES[str(key)]
    return OFFICIAL_SIMREADY_DEFAULT_PROFILE


@dataclass(frozen=True)
class SimReadyProfile:
    """One validation policy for a family of simulation assets."""

    profile_id: str
    asset_type: str
    required_levels: tuple[str, ...]
    requires_physics: bool = False
    requires_physics_materials: bool = False
    requires_articulation: bool = False
    requires_task_semantics: bool = False
    requires_runtime_smoke: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "asset_type": self.asset_type,
            "required_levels": list(self.required_levels),
            "requires_physics": self.requires_physics,
            "requires_physics_materials": self.requires_physics_materials,
            "requires_articulation": self.requires_articulation,
            "requires_task_semantics": self.requires_task_semantics,
            "requires_runtime_smoke": self.requires_runtime_smoke,
            "description": self.description,
        }


SIM_READY_PROFILES: dict[str, SimReadyProfile] = {
    STATIC_VISUAL_PROFILE_ID: SimReadyProfile(
        STATIC_VISUAL_PROFILE_ID,
        "static_visual_asset",
        ("cad_preflight", "mesh_topology", "usd_core"),
        description="Valid OpenUSD visual asset with mesh health, materials, and semantics.",
    ),
    RIGID_DYNAMIC_PROFILE_ID: SimReadyProfile(
        RIGID_DYNAMIC_PROFILE_ID,
        "rigid_dynamic_asset",
        (
            "cad_preflight",
            "mesh_topology",
            "usd_core",
            "physics_authoring",
            "physics_materials",
        ),
        requires_physics=True,
        requires_physics_materials=True,
        description="Rigid-body simulation asset with colliders, mass, and physics materials.",
    ),
    ARTICULATED_PROFILE_ID: SimReadyProfile(
        ARTICULATED_PROFILE_ID,
        "articulated_asset",
        (
            "cad_preflight",
            "mesh_topology",
            "usd_core",
            "physics_authoring",
            "physics_materials",
            "articulation",
        ),
        requires_physics=True,
        requires_physics_materials=True,
        requires_articulation=True,
        description="Multibody asset with articulation root, joint bindings, and limits.",
    ),
    ROBOTICS_MANIPULATION_PROFILE_ID: SimReadyProfile(
        ROBOTICS_MANIPULATION_PROFILE_ID,
        "robotics_manipulation_asset",
        (
            "cad_preflight",
            "mesh_topology",
            "usd_core",
            "physics_authoring",
            "physics_materials",
            "articulation",
            "task_semantics",
            "runtime_smoke",
        ),
        requires_physics=True,
        requires_physics_materials=True,
        requires_articulation=True,
        requires_task_semantics=True,
        requires_runtime_smoke=True,
        description="Robotics/contact asset with task semantics and runtime smoke evidence.",
    ),
    INSERTION_FIXTURE_PROFILE_ID: SimReadyProfile(
        INSERTION_FIXTURE_PROFILE_ID,
        "insertion_or_fixture_asset",
        (
            "cad_preflight",
            "mesh_topology",
            "usd_core",
            "physics_authoring",
            "physics_materials",
            "task_semantics",
            "runtime_smoke",
        ),
        requires_physics=True,
        requires_physics_materials=True,
        requires_task_semantics=True,
        requires_runtime_smoke=True,
        description="Insertion/fixture asset with clearance, mating, and contact metadata.",
    ),
}

_ASSET_TYPE_PROFILE_IDS = {
    profile.asset_type: profile_id for profile_id, profile in SIM_READY_PROFILES.items()
}
_ASSET_TYPE_ALIASES = {
    "static": "static_visual_asset",
    "visual": "static_visual_asset",
    "static_visual": "static_visual_asset",
    "rigid": "rigid_dynamic_asset",
    "dynamic": "rigid_dynamic_asset",
    "rigid_body": "rigid_dynamic_asset",
    "articulated": "articulated_asset",
    "articulation": "articulated_asset",
    "robotics": "robotics_manipulation_asset",
    "robot": "robotics_manipulation_asset",
    "manipulation": "robotics_manipulation_asset",
    "grasp": "robotics_manipulation_asset",
    "dex": "robotics_manipulation_asset",
    "insertion": "insertion_or_fixture_asset",
    "fixture": "insertion_or_fixture_asset",
}

_TASK_SEMANTIC_ISSUE_NAMES = {
    "contact_interfaces_declared",
    "contact_task_multiple_parts",
    "insertion_axes_declared",
    "mating_surfaces_declared",
    "clearance_checks_declared",
    "contact_tolerance_metadata",
    "grasp_affordances_declared",
    "mass_inertia_enforced",
    "joint_limits_declared",
    "gdt_tolerance_checks",
}


@dataclass
class SimReadyIssue:
    """One simulation-readiness finding."""

    name: str
    passed: bool
    severity: str = "error"
    message: str = ""
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class SimReadyReport:
    """Structured static report for downstream simulation consumers."""

    passed: bool
    issues: list[SimReadyIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sim_budget: dict[str, Any] = field(default_factory=dict)
    contact_interfaces: list[dict[str, Any]] = field(default_factory=list)
    recommended_collision_proxies: list[dict[str, Any]] = field(default_factory=list)
    insertion_axes: list[dict[str, Any]] = field(default_factory=list)
    mating_surfaces: list[dict[str, Any]] = field(default_factory=list)
    clearance_checks: list[dict[str, Any]] = field(default_factory=list)
    grasp_affordances: list[dict[str, Any]] = field(default_factory=list)
    joint_limits: list[dict[str, Any]] = field(default_factory=list)
    mass_properties: list[dict[str, Any]] = field(default_factory=list)
    gdt_checks: list[dict[str, Any]] = field(default_factory=list)
    mesh_topology: dict[str, Any] = field(default_factory=dict)
    geometry_handoff: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking_issues(self) -> list[SimReadyIssue]:
        return [i for i in self.issues if not i.passed and i.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "blocking_issues": [i.to_dict() for i in self.blocking_issues],
            "warnings": list(self.warnings),
            "sim_budget": dict(self.sim_budget),
            "contact_interfaces": list(self.contact_interfaces),
            "recommended_collision_proxies": list(self.recommended_collision_proxies),
            "insertion_axes": list(self.insertion_axes),
            "mating_surfaces": list(self.mating_surfaces),
            "clearance_checks": list(self.clearance_checks),
            "grasp_affordances": list(self.grasp_affordances),
            "joint_limits": list(self.joint_limits),
            "mass_properties": list(self.mass_properties),
            "gdt_checks": list(self.gdt_checks),
            "mesh_topology": dict(self.mesh_topology),
            "geometry_handoff": dict(self.geometry_handoff),
            "profile": dict(self.profile),
        }


@dataclass
class SimReadyLevel:
    """One profile level inside a SimReady certificate."""

    name: str
    status: str
    mandatory: bool = True
    message: str = ""
    issues: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status not in {"fail", "error", "incomplete"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "mandatory": self.mandatory,
            "message": self.message,
            "issues": list(self.issues),
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass
class SimReadyCertificate:
    """Versioned profile result for an Isaac/OpenUSD simulation handoff."""

    profile_id: str
    status: str
    levels: list[SimReadyLevel]
    artifacts: dict[str, Any] = field(default_factory=dict)
    fix_strategies: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    asset_type: str | None = None
    requested_profile_id: str | None = None
    resolved_profile_id: str | None = None
    validation_mode: str = "official_or_local"

    @property
    def passed(self) -> bool:
        return self.status not in {"fail", "error", "incomplete"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "requested_profile_id": self.requested_profile_id,
            "resolved_profile_id": self.resolved_profile_id or self.profile_id,
            "asset_type": self.asset_type,
            "validation_mode": self.validation_mode,
            "status": self.status,
            "passed": self.passed,
            "levels": [level.to_dict() for level in self.levels],
            "artifacts": dict(self.artifacts),
            "fix_strategies": list(self.fix_strategies),
            "assumptions": list(self.assumptions),
        }


def _shape_entries(result: Any) -> list[tuple[str, Any]]:
    """Return named shape entries from a cad_kernel Result-like object."""

    return [(record["name"], record["shape"]) for record in _shape_entry_records(result)]


def _shape_entry_records(result: Any) -> list[dict[str, Any]]:
    """Return named shape records with scene/assembly metadata preserved."""

    if getattr(result, "shape", None) is not None:
        return [{"name": "Part", "shape": result.shape, "metadata": {}}]
    if getattr(result, "scene", None):
        records: list[dict[str, Any]] = []
        for i, node in enumerate(result.scene):
            shape = node["shape"] if isinstance(node, dict) else getattr(node, "shape", None)
            if shape is None:
                continue
            name = (
                str(node.get("name", f"part_{i}"))
                if isinstance(node, dict)
                else str(getattr(node, "name", f"part_{i}"))
            )
            records.append(
                {
                    "name": name,
                    "shape": shape,
                    "metadata": _scene_entry_metadata(node),
                }
            )
        return records
    if getattr(result, "assembly", None) is not None:
        return [
            {
                "name": p.name,
                "shape": p.shape,
                "metadata": {
                    **dict(getattr(p, "metadata", {}) or {}),
                    **(
                        {"connectors": getattr(p, "connectors", {})}
                        if getattr(p, "connectors", None)
                        else {}
                    ),
                },
            }
            for p in result.assembly.parts.values()
            if p.shape is not None
        ]
    return []


def _scene_entry_metadata(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return {
            str(key): value
            for key, value in entry.items()
            if key not in {"name", "shape", "color", "kind", "material"}
        }
    metadata = getattr(entry, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    metadata = getattr(entry, "meta", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _shape_kind(shape: Any) -> str | None:
    try:
        return getattr(shape, "kind_value", None)
    except Exception:
        return None


def _static_check_severity(check_name: str) -> str:
    # Thin-wall detection is a DFM/manufacturing warning. For simulation
    # handoff it is still useful evidence, but it should not block a scene
    # that already exports bounded collision proxies and valid mass properties.
    if check_name == "thin_wall_check":
        return "warning"
    return "error"


def _mesh_triangle_count(shape: Any, tolerance: float = 0.5) -> int | None:
    """Best-effort triangle count through a backend-neutral shape adapter."""

    try:
        tessellate = getattr(shape, "tessellate", None)
        if not callable(tessellate):
            return None
        out = tessellate(tolerance)
        if hasattr(out, "tri_verts"):
            return int(len(out.tri_verts))
        _verts, tris = out
        if not tris:
            return 0
        if isinstance(tris[0], tuple):
            return len(tris)
        return len(tris) // 3
    except Exception:
        return None


def resolve_sim_ready_profile(
    result: Any | None = None,
    *,
    task_type: str | None = None,
    asset_type: str | None = None,
    profile_id: str | None = None,
) -> SimReadyProfile:
    """Resolve the strictest applicable profile from explicit inputs and result shape."""

    requested = (profile_id or "").strip()
    if requested in SIM_READY_PROFILES:
        return SIM_READY_PROFILES[requested]

    type_key = (asset_type or "").strip().lower().replace("-", "_")
    type_key = _ASSET_TYPE_ALIASES.get(type_key, type_key)
    if type_key in _ASSET_TYPE_PROFILE_IDS:
        return SIM_READY_PROFILES[_ASSET_TYPE_PROFILE_IDS[type_key]]

    task_l = (task_type or "").lower()
    if _is_insertion_like(task_l):
        return SIM_READY_PROFILES[INSERTION_FIXTURE_PROFILE_ID]
    if _is_contact_rich(task_l) or _has_token(task_l, _GRASP_TASK_TOKENS):
        return SIM_READY_PROFILES[ROBOTICS_MANIPULATION_PROFILE_ID]

    assembly = getattr(result, "assembly", None) if result is not None else None
    joints = list(getattr(assembly, "joints", {}).values()) if assembly is not None else []
    moving = [
        j
        for j in joints
        if getattr(j, "kind", None) in {"revolute", "prismatic", "continuous", "cylindrical"}
    ]
    if moving:
        return SIM_READY_PROFILES[ARTICULATED_PROFILE_ID]
    entries = _shape_entries(result) if result is not None else []
    if assembly is not None or len(entries) > 1:
        return SIM_READY_PROFILES[RIGID_DYNAMIC_PROFILE_ID]
    return SIM_READY_PROFILES[STATIC_VISUAL_PROFILE_ID]


def _vec_sub(a: Any, b: Any) -> tuple[float, float, float]:
    return (
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
        float(a[2]) - float(b[2]),
    )


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _tri_area(a: Any, b: Any, c: Any) -> float:
    x, y, z = _cross(_vec_sub(b, a), _vec_sub(c, a))
    return 0.5 * math.sqrt(x * x + y * y + z * z)


def _quantized_point(point: Any, eps: float = 1e-9) -> tuple[int, int, int]:
    return tuple(int(round(float(coord) / eps)) for coord in point)  # type: ignore[return-value]


def audit_usd_mesh_topology(
    usd_path: str | Path | None,
    *,
    max_triangles: int = 250_000,
) -> dict[str, Any]:
    """Inspect exported USD meshes for simulation-blocking topology defects."""

    if not usd_path:
        return {
            "status": "unavailable",
            "message": "No USD artifact was provided for mesh topology audit.",
            "issues": [
                _certificate_issue(
                    "mesh_topology_usd_present",
                    passed=False,
                    severity="warning",
                    message="No USD artifact was available.",
                    suggestion="Export USD before mesh topology validation.",
                    source="mesh_topology",
                )
            ],
            "metrics": {},
        }
    path = Path(usd_path)
    if not path.exists():
        return {
            "status": "fail",
            "message": f"USD artifact does not exist: {path}",
            "issues": [
                _certificate_issue(
                    "mesh_topology_usd_present",
                    passed=False,
                    severity="error",
                    message=f"USD artifact does not exist: {path}",
                    suggestion="Export a valid USD artifact before validation.",
                    source="mesh_topology",
                )
            ],
            "metrics": {},
        }

    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None")

        metrics: dict[str, Any] = {
            "mesh_count": 0,
            "face_count": 0,
            "triangle_equivalent_count": 0,
            "empty_mesh_count": 0,
            "nan_inf_vertex_count": 0,
            "degenerate_face_count": 0,
            "duplicate_face_count": 0,
            "boundary_edge_count": 0,
            "over_connected_edge_count": 0,
            "non_manifold_edge_count": 0,
            "non_triangular_face_count": 0,
            "per_mesh": [],
        }
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            purpose = UsdGeom.Imageable(prim).ComputePurpose()
            if purpose == UsdGeom.Tokens.guide:
                continue
            metrics["mesh_count"] += 1
            mesh = UsdGeom.Mesh(prim)
            points = list(mesh.GetPointsAttr().Get() or [])
            counts = [int(v) for v in list(mesh.GetFaceVertexCountsAttr().Get() or [])]
            indices = [int(v) for v in list(mesh.GetFaceVertexIndicesAttr().Get() or [])]
            face_count = len(counts)
            metrics["face_count"] += face_count
            mesh_record = {
                "path": str(prim.GetPath()),
                "points": len(points),
                "faces": face_count,
                "triangle_equivalent_count": 0,
                "nan_inf_vertex_count": 0,
                "degenerate_face_count": 0,
                "duplicate_face_count": 0,
                "boundary_edge_count": 0,
                "over_connected_edge_count": 0,
                "non_manifold_edge_count": 0,
                "non_triangular_face_count": 0,
                "passed": True,
                "reasons": [],
            }
            if not points or not counts or not indices:
                metrics["empty_mesh_count"] += 1
                mesh_record["passed"] = False
                mesh_record["reasons"].append("mesh has no points/faces/indices")
            invalid_vertices = 0
            for point in points:
                if not _finite_vector(point, length=3):
                    invalid_vertices += 1
            mesh_record["nan_inf_vertex_count"] = invalid_vertices
            metrics["nan_inf_vertex_count"] += invalid_vertices
            if invalid_vertices:
                mesh_record["passed"] = False
                mesh_record["reasons"].append("mesh contains NaN/Inf vertices")

            offset = 0
            edge_counts: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = {}
            seen_faces: set[tuple[tuple[int, int, int], ...]] = set()
            for count in counts:
                face_indices = indices[offset : offset + count]
                offset += count
                if count < 3 or any(i < 0 or i >= len(points) for i in face_indices):
                    mesh_record["degenerate_face_count"] += 1
                    continue
                if count != 3:
                    mesh_record["non_triangular_face_count"] += 1
                qface = tuple(_quantized_point(points[i]) for i in face_indices)
                sorted_face = tuple(sorted(qface))
                if sorted_face in seen_faces:
                    mesh_record["duplicate_face_count"] += 1
                seen_faces.add(sorted_face)
                for a, b in zip(qface, qface[1:] + qface[:1], strict=False):
                    edge = tuple(sorted((a, b)))  # type: ignore[arg-type]
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
                tri_equiv = max(1, count - 2)
                mesh_record["triangle_equivalent_count"] += tri_equiv
                for tri_index in range(1, count - 1):
                    if (
                        _tri_area(
                            points[face_indices[0]],
                            points[face_indices[tri_index]],
                            points[face_indices[tri_index + 1]],
                        )
                        <= 1e-16
                    ):
                        mesh_record["degenerate_face_count"] += 1
                        break
            mesh_record["boundary_edge_count"] = sum(
                1 for edge_count in edge_counts.values() if edge_count == 1
            )
            mesh_record["over_connected_edge_count"] = sum(
                1 for edge_count in edge_counts.values() if edge_count > 2
            )
            # Preserve the established aggregate metric while exposing the two
            # failure modes separately for repair routing.
            mesh_record["non_manifold_edge_count"] = (
                mesh_record["boundary_edge_count"] + mesh_record["over_connected_edge_count"]
            )
            for key in (
                "triangle_equivalent_count",
                "degenerate_face_count",
                "duplicate_face_count",
                "boundary_edge_count",
                "over_connected_edge_count",
                "non_manifold_edge_count",
                "non_triangular_face_count",
            ):
                metrics[key] += mesh_record[key]
            if (
                mesh_record["degenerate_face_count"]
                or mesh_record["duplicate_face_count"]
                or mesh_record["non_manifold_edge_count"]
                or mesh_record["non_triangular_face_count"]
            ):
                mesh_record["passed"] = False
                if mesh_record["degenerate_face_count"]:
                    mesh_record["reasons"].append("mesh has zero-area or malformed faces")
                if mesh_record["duplicate_face_count"]:
                    mesh_record["reasons"].append("mesh has duplicate faces")
                if mesh_record["boundary_edge_count"]:
                    mesh_record["reasons"].append("mesh has boundary edges")
                if mesh_record["over_connected_edge_count"]:
                    mesh_record["reasons"].append("mesh has over-connected non-manifold edges")
                if mesh_record["non_triangular_face_count"]:
                    mesh_record["reasons"].append("mesh has non-triangular faces")
            metrics["per_mesh"].append(mesh_record)

        checks = [
            (
                "mesh_topology_meshes_present",
                metrics["mesh_count"] > 0,
                f"{metrics['mesh_count']} mesh prim(s)",
                "Export at least one renderable mesh.",
            ),
            (
                "mesh_topology_no_empty_meshes",
                metrics["empty_mesh_count"] == 0,
                f"{metrics['empty_mesh_count']} empty mesh(es)",
                "Remove or repair empty mesh prims before simulation.",
            ),
            (
                "mesh_topology_vertices_finite",
                metrics["nan_inf_vertex_count"] == 0,
                f"{metrics['nan_inf_vertex_count']} invalid vertex value(s)",
                "Remove NaN/Inf vertex coordinates.",
            ),
            (
                "mesh_topology_no_degenerate_faces",
                metrics["degenerate_face_count"] == 0,
                f"{metrics['degenerate_face_count']} degenerate face(s)",
                "Remove zero-area or malformed faces.",
            ),
            (
                "mesh_topology_no_duplicate_faces",
                metrics["duplicate_face_count"] == 0,
                f"{metrics['duplicate_face_count']} duplicate face(s)",
                "Deduplicate mesh faces.",
            ),
            (
                "mesh_topology_watertight_edges",
                metrics["non_manifold_edge_count"] == 0,
                (
                    f"{metrics['boundary_edge_count']} boundary edge(s), "
                    f"{metrics['over_connected_edge_count']} over-connected edge(s)"
                ),
                "Use watertight collision/render meshes for simulation handoff.",
            ),
            (
                "mesh_topology_triangulated",
                metrics["non_triangular_face_count"] == 0,
                f"{metrics['non_triangular_face_count']} non-triangular face(s)",
                "Triangulate exported mesh faces.",
            ),
            (
                "mesh_topology_triangle_budget",
                metrics["triangle_equivalent_count"] <= max_triangles,
                f"{metrics['triangle_equivalent_count']} triangle-equivalent face(s)",
                "Simplify render meshes or split collision geometry.",
            ),
        ]
        issues = [
            _certificate_issue(
                name,
                passed=passed,
                severity="info" if passed else "error",
                message=message,
                suggestion=suggestion,
                source="mesh_topology",
            )
            for name, passed, message, suggestion in checks
        ]
        failed = [issue for issue in issues if not issue["passed"] and issue["severity"] == "error"]
        return {
            "status": "fail" if failed else "pass",
            "message": "USD mesh topology is simulation-ready"
            if not failed
            else f"{len(failed)} mesh topology issue(s)",
            "issues": issues,
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"Could not inspect USD mesh topology: {exc}",
            "issues": [
                _certificate_issue(
                    "mesh_topology_parse",
                    passed=False,
                    severity="error",
                    message=f"Could not inspect USD mesh topology: {exc}",
                    suggestion="Verify usd-exchange is installed and the USD file is valid.",
                    source="mesh_topology",
                )
            ],
            "metrics": {},
        }


def audit_usd_geometry_handoff(
    usd_path: str | Path | None,
    *,
    max_collision_triangles: int = 256,
    max_collision_to_visual_ratio: float = 0.25,
    require_separate_collisions: bool = True,
    require_external_textures: bool = False,
) -> dict[str, Any]:
    """Audit the visual/decal/collider representation contract in OpenUSD.

    This complements topology checks. It verifies that render meshes, image
    decals, and low-cost collision proxies are independently authored and that
    packaged texture references resolve from the exported asset directory.
    """

    if not usd_path:
        return {
            "status": "unavailable",
            "message": "No USD artifact was provided for geometry handoff audit.",
            "issues": [
                _certificate_issue(
                    "geometry_handoff_usd_present",
                    passed=False,
                    severity="warning",
                    message="No USD artifact was available.",
                    suggestion="Export USD before geometry handoff validation.",
                    source="geometry_handoff",
                )
            ],
            "metrics": {},
        }
    path = Path(usd_path)
    if not path.exists():
        return {
            "status": "fail",
            "message": f"USD artifact does not exist: {path}",
            "issues": [
                _certificate_issue(
                    "geometry_handoff_usd_present",
                    passed=False,
                    severity="error",
                    message=f"USD artifact does not exist: {path}",
                    suggestion="Export a valid USD artifact before validation.",
                    source="geometry_handoff",
                )
            ],
            "metrics": {},
        }
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None")

        metrics: dict[str, Any] = {
            "mesh_count": 0,
            "visual_mesh_count": 0,
            "collision_mesh_count": 0,
            "decal_mesh_count": 0,
            "visual_triangles": 0,
            "collision_triangles": 0,
            "max_collision_mesh_triangles": 0,
            "collision_to_visual_ratio": None,
            "shared_visual_collision_count": 0,
            "decal_collision_count": 0,
            "textured_decal_count": 0,
            "untextured_decal_count": 0,
            "texture_count": 0,
            "missing_texture_count": 0,
            "absolute_local_texture_count": 0,
            "physical_material_provenance_count": 0,
            "per_mesh": [],
            "textures": [],
        }

        def _bound_texture_count(prim: Any) -> int:
            shader_paths: set[str] = set()
            material_paths: set[Any] = set()
            for relationship in prim.GetRelationships():
                if relationship.GetName().startswith("material:binding"):
                    material_paths.update(relationship.GetTargets())
            for material_path in material_paths:
                material_prim = stage.GetPrimAtPath(material_path)
                if not material_prim:
                    continue
                for descendant in Usd.PrimRange(material_prim):
                    if descendant.GetTypeName() != "Shader":
                        continue
                    shader = UsdShade.Shader(descendant)
                    if shader.GetIdAttr().Get() == "UsdUVTexture":
                        shader_paths.add(str(descendant.GetPath()))
            return len(shader_paths)

        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            counts = [int(value) for value in list(mesh.GetFaceVertexCountsAttr().Get() or [])]
            triangles = sum(max(0, count - 2) for count in counts)
            collision = _has_api(prim, UsdPhysics.CollisionAPI)
            purpose = str(UsdGeom.Imageable(prim).ComputePurpose())
            visibility = str(UsdGeom.Imageable(prim).ComputeVisibility())
            appearance_role = str(
                prim.GetCustomDataByKey("cad_kernel:appearance_role") or ""
            ).lower()
            decal = appearance_role == "decal"
            bound_texture_count = _bound_texture_count(prim) if decal else 0
            authored_budget = prim.GetCustomDataByKey("cad_kernel:collision_max_triangles")
            try:
                authored_budget = int(authored_budget) if authored_budget is not None else None
            except (TypeError, ValueError):
                authored_budget = None
            record = {
                "path": str(prim.GetPath()),
                "role": "collision" if collision else "decal" if decal else "visual",
                "triangles": triangles,
                "purpose": purpose,
                "visibility": visibility,
                "authored_collision_budget": authored_budget,
                "bound_texture_count": bound_texture_count,
            }
            metrics["mesh_count"] += 1
            if collision:
                metrics["collision_mesh_count"] += 1
                metrics["collision_triangles"] += triangles
                metrics["max_collision_mesh_triangles"] = max(
                    metrics["max_collision_mesh_triangles"], triangles
                )
                if purpose != str(UsdGeom.Tokens.proxy) or visibility != str(
                    UsdGeom.Tokens.invisible
                ):
                    metrics["shared_visual_collision_count"] += 1
                if prim.GetCustomDataByKey("cad_kernel:physical_material_provenance"):
                    metrics["physical_material_provenance_count"] += 1
            else:
                metrics["visual_mesh_count"] += 1
                metrics["visual_triangles"] += triangles
            if decal:
                metrics["decal_mesh_count"] += 1
                if collision:
                    metrics["decal_collision_count"] += 1
                if bound_texture_count > 0:
                    metrics["textured_decal_count"] += 1
                else:
                    metrics["untextured_decal_count"] += 1
            metrics["per_mesh"].append(record)

        if metrics["visual_triangles"] > 0:
            metrics["collision_to_visual_ratio"] = (
                metrics["collision_triangles"] / metrics["visual_triangles"]
            )

        for prim in stage.Traverse():
            if prim.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(prim)
            if shader.GetIdAttr().Get() != "UsdUVTexture":
                continue
            asset = shader.GetInput("file").Get()
            authored = str(getattr(asset, "path", asset) or "")
            resolved = str(getattr(asset, "resolvedPath", "") or "")
            is_remote = "://" in authored and not authored.startswith("file://")
            local_path = None
            if not is_remote:
                raw = authored[7:] if authored.startswith("file://") else authored
                local_path = Path(raw)
                if not local_path.is_absolute():
                    local_path = path.parent / local_path
                local_path = local_path.resolve()
            exists = bool(is_remote or resolved or (local_path and local_path.is_file()))
            raw_path = authored[7:] if authored.startswith("file://") else authored
            if local_path is not None and Path(raw_path).is_absolute():
                metrics["absolute_local_texture_count"] += 1
            if not exists:
                metrics["missing_texture_count"] += 1
            metrics["texture_count"] += 1
            metrics["textures"].append(
                {
                    "shader": str(prim.GetPath()),
                    "authored_uri": authored,
                    "resolved_uri": resolved or (str(local_path) if local_path else authored),
                    "exists": exists,
                    "portable": is_remote or not Path(raw_path).is_absolute(),
                }
            )

        collision_budget_ok = all(
            record["triangles"]
            <= min(
                max_collision_triangles,
                record["authored_collision_budget"]
                if record["authored_collision_budget"] is not None
                else max_collision_triangles,
            )
            for record in metrics["per_mesh"]
            if record["role"] == "collision"
        )
        ratio = metrics["collision_to_visual_ratio"]
        ratio_ok = ratio is not None and ratio <= max_collision_to_visual_ratio
        checks = [
            (
                "geometry_handoff_visuals_present",
                metrics["visual_mesh_count"] > 0,
                f"{metrics['visual_mesh_count']} visual mesh(es)",
                "Author at least one render-purpose visual mesh.",
            ),
            (
                "geometry_handoff_separate_colliders",
                (not require_separate_collisions)
                or (
                    metrics["collision_mesh_count"] > 0
                    and metrics["shared_visual_collision_count"] == 0
                ),
                (
                    f"{metrics['collision_mesh_count']} proxy collider(s), "
                    f"{metrics['shared_visual_collision_count']} render/collision reuse(s)"
                ),
                "Author invisible proxy-purpose colliders independently from visual meshes.",
            ),
            (
                "geometry_handoff_collision_budget",
                (not require_separate_collisions) or collision_budget_ok,
                (
                    f"maximum {metrics['max_collision_mesh_triangles']} triangles per collider; "
                    f"limit {max_collision_triangles}"
                ),
                "Use primitive or convex collision proxies and honor each authored triangle budget.",
            ),
            (
                "geometry_handoff_collision_visual_ratio",
                (not require_separate_collisions) or ratio_ok,
                f"collision/visual triangle ratio={ratio}",
                "Reduce collider triangles below the configured fraction of visual triangles.",
            ),
            (
                "geometry_handoff_decals_visual_only",
                metrics["decal_collision_count"] == 0,
                f"{metrics['decal_collision_count']} decal collider(s)",
                "Keep decal carriers visual-only; never include labels in collision or inertia.",
            ),
            (
                "geometry_handoff_textures_resolve",
                metrics["missing_texture_count"] == 0,
                f"{metrics['missing_texture_count']} unresolved texture(s)",
                "Package local textures beside the USD and author relative asset paths.",
            ),
            (
                "geometry_handoff_textures_portable",
                metrics["absolute_local_texture_count"] == 0,
                f"{metrics['absolute_local_texture_count']} absolute local texture path(s)",
                "Use relative packaged texture paths instead of machine-local paths.",
            ),
            (
                "geometry_handoff_external_textures_present",
                (not require_external_textures)
                or (
                    metrics["texture_count"] > 0
                    and metrics["decal_mesh_count"] > 0
                    and metrics["untextured_decal_count"] == 0
                ),
                (
                    f"{metrics['texture_count']} external texture shader(s), "
                    f"{metrics['textured_decal_count']}/{metrics['decal_mesh_count']} "
                    "textured decal mesh(es)"
                ),
                "Bind every declared decal to a replaceable external image texture.",
            ),
            (
                "geometry_handoff_physical_material_provenance",
                (not require_separate_collisions)
                or metrics["physical_material_provenance_count"] == metrics["collision_mesh_count"],
                (
                    f"{metrics['physical_material_provenance_count']}/"
                    f"{metrics['collision_mesh_count']} collider material provenance record(s)"
                ),
                "Bind an explicit or provenance-marked physical material to every collider.",
            ),
        ]
        issues = [
            _certificate_issue(
                name,
                passed=passed,
                severity="info" if passed else "error",
                message=message,
                suggestion=suggestion,
                source="geometry_handoff",
            )
            for name, passed, message, suggestion in checks
        ]
        failed = [item for item in issues if not item["passed"] and item["severity"] == "error"]
        return {
            "status": "fail" if failed else "pass",
            "message": "USD geometry handoff contract passed"
            if not failed
            else f"{len(failed)} geometry handoff issue(s)",
            "issues": issues,
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"Could not inspect USD geometry handoff: {exc}",
            "issues": [
                _certificate_issue(
                    "geometry_handoff_parse",
                    passed=False,
                    severity="error",
                    message=f"Could not inspect USD geometry handoff: {exc}",
                    suggestion="Verify the USD and texture package before handoff.",
                    source="geometry_handoff",
                )
            ],
            "metrics": {},
        }


def _usd_stage_checks(usd_path: str | Path | None) -> list[SimReadyIssue]:
    """Inspect USD stage-level simulation conventions when a USD is present."""

    if not usd_path:
        return [
            SimReadyIssue(
                "usd_present",
                False,
                "warning",
                "No USD path was provided; stage units/up-axis were not checked.",
                "Export USD before downstream Isaac Sim handoff.",
            )
        ]
    path = Path(usd_path)
    if not path.exists():
        return [
            SimReadyIssue(
                "usd_present",
                False,
                "error",
                f"USD path does not exist: {path}",
                "Export or provide a valid USD artifact.",
            )
        ]
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path))
        up_axis = UsdGeom.GetStageUpAxis(stage)
        meters = UsdGeom.GetStageMetersPerUnit(stage)
        return [
            SimReadyIssue(
                "usd_up_axis_z",
                str(up_axis).lower() == "z",
                "error",
                f"upAxis={up_axis}",
                "Export Isaac-facing USD as Z-up.",
            ),
            SimReadyIssue(
                "usd_meters_per_unit_1",
                abs(float(meters) - 1.0) < 1e-9,
                "error",
                f"metersPerUnit={meters}",
                "Export Isaac-facing USD with metersPerUnit=1.0.",
            ),
        ]
    except Exception as exc:
        return [
            SimReadyIssue(
                "usd_parse",
                False,
                "error",
                f"Could not inspect USD: {exc}",
                "Verify usd-exchange is installed and the USD file is valid.",
            )
        ]


def _fix_strategy_for_issue(name: str) -> str:
    """Return a stable repair strategy name for a SimReady issue."""

    if name in {"semantic_kinds"} or name.endswith(":semantic_kinds"):
        return "add_semantic_kind_tags"
    if "usd_" in name or name.startswith("usd_"):
        return "reexport_isaac_openusd"
    if "collision" in name or "proxy" in name:
        return "add_or_simplify_collision_proxies"
    if "mass" in name or "inertia" in name:
        return "bind_density_or_explicit_inertia"
    if "physics_material" in name or "friction" in name or "restitution" in name:
        return "author_and_bind_physics_materials"
    if "joint" in name:
        return "declare_joint_axis_limits_and_bindings"
    if "contact" in name or "mating" in name or "clearance" in name:
        return "add_contact_axis_clearance_and_tolerance_metadata"
    if "grasp" in name:
        return "tag_grasp_affordance_regions"
    if "gdt" in name or "tolerance" in name:
        return "add_positive_tolerance_and_datum_metadata"
    if "mesh" in name or "triangle" in name:
        return "lower_render_mesh_or_split_collision_geometry"
    if "assertion" in name:
        return "fix_failing_verify_probes"
    return "inspect_and_repair_cad_source"


def _certificate_issue(
    name: str,
    *,
    passed: bool,
    severity: str,
    message: str,
    suggestion: str | None = None,
    source: str = "sim_ready",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "message": message,
        "suggestion": suggestion,
        "source": source,
        "fix_strategy": _fix_strategy_for_issue(name),
        "metadata": dict(metadata or {}),
    }


def _legacy_issue_dict(issue: SimReadyIssue) -> dict[str, Any]:
    return _certificate_issue(
        issue.name,
        passed=issue.passed,
        severity=issue.severity,
        message=issue.message,
        suggestion=issue.suggestion,
        source="static_cad",
    )


def _has_api(prim: Any, api_cls: Any) -> bool:
    try:
        return bool(prim.HasAPI(api_cls))
    except Exception:
        try:
            return bool(api_cls(prim))
        except Exception:
            return False


def audit_usd_physics_authoring(usd_path: str | Path | None) -> dict[str, Any]:
    """Inspect USD Physics authoring that Isaac/PhysX consumers need.

    This is deliberately structural: it proves the USD stage carries physics
    schemas and relationships, not that a dynamic rollout behaves correctly.
    """

    if not usd_path:
        return {
            "status": "unavailable",
            "message": "No USD artifact was provided for physics authoring audit.",
            "issues": [
                _certificate_issue(
                    "physics_usd_present",
                    passed=False,
                    severity="warning",
                    message="No USD artifact was available.",
                    suggestion="Export USD before Isaac/OpenUSD handoff.",
                    source="physics_authored",
                )
            ],
            "metrics": {},
        }

    path = Path(usd_path)
    if not path.exists():
        return {
            "status": "fail",
            "message": f"USD artifact does not exist: {path}",
            "issues": [
                _certificate_issue(
                    "physics_usd_present",
                    passed=False,
                    severity="error",
                    message=f"USD artifact does not exist: {path}",
                    suggestion="Export a valid USD artifact before validation.",
                    source="physics_authored",
                )
            ],
            "metrics": {},
        }

    try:
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None")

        metrics = {
            "mesh_count": 0,
            "rigid_body_count": 0,
            "collision_count": 0,
            "mass_count": 0,
            "articulation_root_count": 0,
            "joint_count": 0,
            "moving_joint_count": 0,
            "limited_moving_joint_count": 0,
            "joint_binding_count": 0,
            "default_prim": str(stage.GetDefaultPrim().GetPath())
            if stage.GetDefaultPrim()
            else None,
        }

        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                metrics["mesh_count"] += 1
            if _has_api(prim, UsdPhysics.RigidBodyAPI):
                metrics["rigid_body_count"] += 1
            if _has_api(prim, UsdPhysics.CollisionAPI):
                metrics["collision_count"] += 1
            if _has_api(prim, UsdPhysics.MassAPI):
                metrics["mass_count"] += 1
            if _has_api(prim, UsdPhysics.ArticulationRootAPI):
                metrics["articulation_root_count"] += 1
            type_name = str(prim.GetTypeName() or "")
            if type_name.startswith("Physics") and type_name.endswith("Joint"):
                metrics["joint_count"] += 1
                joint = UsdPhysics.Joint(prim)
                body0 = list(joint.GetBody0Rel().GetTargets() or [])
                body1 = list(joint.GetBody1Rel().GetTargets() or [])
                pinned_fixed_joint = type_name == "PhysicsFixedJoint" and bool(body0) != bool(body1)
                if (body0 and body1) or pinned_fixed_joint:
                    metrics["joint_binding_count"] += 1
                if type_name in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
                    metrics["moving_joint_count"] += 1
                    lower = prim.GetAttribute("physics:lowerLimit")
                    upper = prim.GetAttribute("physics:upperLimit")
                    if lower.HasAuthoredValueOpinion() and upper.HasAuthoredValueOpinion():
                        metrics["limited_moving_joint_count"] += 1

        checks = [
            (
                "usd_default_prim",
                bool(metrics["default_prim"]),
                "USD stage has a default prim"
                if metrics["default_prim"]
                else "USD stage is missing a default prim",
                "Set a default prim on exported USD stages.",
            ),
            (
                "usd_meshes_present",
                metrics["mesh_count"] > 0,
                f"{metrics['mesh_count']} mesh prim(s)",
                "Export at least one mesh prim.",
            ),
            (
                "usd_rigid_bodies_present",
                metrics["rigid_body_count"] > 0,
                f"{metrics['rigid_body_count']} rigid body prim(s)",
                "Apply UsdPhysics.RigidBodyAPI to simulation parts.",
            ),
            (
                "usd_colliders_present",
                metrics["collision_count"] > 0,
                f"{metrics['collision_count']} collision prim(s)",
                "Apply UsdPhysics.CollisionAPI to collision geometry.",
            ),
            (
                "usd_mass_properties_present",
                metrics["mass_count"] > 0,
                f"{metrics['mass_count']} mass API prim(s)",
                "Apply UsdPhysics.MassAPI with mass, center of mass, and inertia.",
            ),
        ]
        if metrics["joint_count"]:
            checks.append(
                (
                    "usd_joint_bindings_present",
                    metrics["joint_binding_count"] == metrics["joint_count"],
                    (
                        f"{metrics['joint_binding_count']}/{metrics['joint_count']} "
                        "joint(s) bind body0/body1 or pin a fixed joint to world"
                    ),
                    "Bind every physics joint to body relationships; fixed root joints may pin one side to world.",
                )
            )

        issues = [
            _certificate_issue(
                name,
                passed=passed,
                severity="info" if passed else "error",
                message=message,
                suggestion=suggestion,
                source="physics_authored",
            )
            for name, passed, message, suggestion in checks
        ]
        failed = [issue for issue in issues if not issue["passed"] and issue["severity"] == "error"]
        return {
            "status": "fail" if failed else "pass",
            "message": "USD physics schemas are structurally authored"
            if not failed
            else f"{len(failed)} USD physics authoring issue(s)",
            "issues": issues,
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"Could not inspect USD physics authoring: {exc}",
            "issues": [
                _certificate_issue(
                    "usd_physics_parse",
                    passed=False,
                    severity="error",
                    message=f"Could not inspect USD physics authoring: {exc}",
                    suggestion="Verify usd-exchange is installed and the USD file is valid.",
                    source="physics_authored",
                )
            ],
            "metrics": {},
        }


def audit_usd_physics_materials(usd_path: str | Path | None) -> dict[str, Any]:
    """Inspect physics material authoring and collider bindings in a USD stage."""

    if not usd_path:
        return {
            "status": "unavailable",
            "message": "No USD artifact was provided for physics material audit.",
            "issues": [
                _certificate_issue(
                    "physics_material_usd_present",
                    passed=False,
                    severity="warning",
                    message="No USD artifact was available.",
                    suggestion="Export USD before physics material validation.",
                    source="physics_materials",
                )
            ],
            "metrics": {},
        }
    path = Path(usd_path)
    if not path.exists():
        return {
            "status": "fail",
            "message": f"USD artifact does not exist: {path}",
            "issues": [
                _certificate_issue(
                    "physics_material_usd_present",
                    passed=False,
                    severity="error",
                    message=f"USD artifact does not exist: {path}",
                    suggestion="Export a valid USD artifact before validation.",
                    source="physics_materials",
                )
            ],
            "metrics": {},
        }

    try:
        from pxr import Usd, UsdPhysics, UsdShade

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None")

        metrics = {
            "physics_material_count": 0,
            "collision_count": 0,
            "bound_collision_material_count": 0,
            "materials_with_static_friction": 0,
            "materials_with_dynamic_friction": 0,
            "materials_with_restitution": 0,
            "materials_with_density": 0,
        }
        physics_material_paths: set[str] = set()
        for prim in stage.Traverse():
            if _has_api(prim, UsdPhysics.MaterialAPI):
                metrics["physics_material_count"] += 1
                physics_material_paths.add(str(prim.GetPath()))
                api = UsdPhysics.MaterialAPI(prim)
                if api.GetStaticFrictionAttr().HasAuthoredValueOpinion():
                    metrics["materials_with_static_friction"] += 1
                if api.GetDynamicFrictionAttr().HasAuthoredValueOpinion():
                    metrics["materials_with_dynamic_friction"] += 1
                if api.GetRestitutionAttr().HasAuthoredValueOpinion():
                    metrics["materials_with_restitution"] += 1
                if api.GetDensityAttr().HasAuthoredValueOpinion():
                    metrics["materials_with_density"] += 1
        for prim in stage.Traverse():
            if not _has_api(prim, UsdPhysics.CollisionAPI):
                continue
            metrics["collision_count"] += 1
            try:
                physics_rel = prim.GetRelationship("material:binding:physics")
                physics_targets = physics_rel.GetTargets() if physics_rel else []
                if any(str(target) in physics_material_paths for target in physics_targets):
                    metrics["bound_collision_material_count"] += 1
                    continue
                material, _binding_rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if material and str(material.GetPrim().GetPath()) in physics_material_paths:
                    metrics["bound_collision_material_count"] += 1
            except Exception:
                pass

        checks = [
            (
                "physics_materials_present",
                metrics["physics_material_count"] > 0,
                f"{metrics['physics_material_count']} physics material prim(s)",
                "Apply UsdPhysics.MaterialAPI to a bound material.",
            ),
            (
                "physics_materials_friction_authored",
                metrics["materials_with_static_friction"] > 0
                and metrics["materials_with_dynamic_friction"] > 0,
                (
                    f"static={metrics['materials_with_static_friction']} "
                    f"dynamic={metrics['materials_with_dynamic_friction']}"
                ),
                "Author static and dynamic friction values for simulated contact.",
            ),
            (
                "physics_materials_restitution_authored",
                metrics["materials_with_restitution"] > 0,
                f"{metrics['materials_with_restitution']} restitution attr(s)",
                "Author restitution for simulated contact materials.",
            ),
            (
                "physics_materials_density_authored",
                metrics["materials_with_density"] > 0,
                f"{metrics['materials_with_density']} density attr(s)",
                "Author density or explicit mass properties for simulated bodies.",
            ),
            (
                "physics_materials_bound_to_colliders",
                metrics["collision_count"] == 0
                or metrics["bound_collision_material_count"] == metrics["collision_count"],
                (
                    f"{metrics['bound_collision_material_count']}/"
                    f"{metrics['collision_count']} collider(s) bound"
                ),
                "Bind physics materials to collision prims.",
            ),
        ]
        issues = [
            _certificate_issue(
                name,
                passed=passed,
                severity="info" if passed else "error",
                message=message,
                suggestion=suggestion,
                source="physics_materials",
            )
            for name, passed, message, suggestion in checks
        ]
        failed = [issue for issue in issues if not issue["passed"] and issue["severity"] == "error"]
        return {
            "status": "fail" if failed else "pass",
            "message": "USD physics materials are authored and bound"
            if not failed
            else f"{len(failed)} physics material issue(s)",
            "issues": issues,
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"Could not inspect USD physics materials: {exc}",
            "issues": [
                _certificate_issue(
                    "physics_materials_parse",
                    passed=False,
                    severity="error",
                    message=f"Could not inspect USD physics materials: {exc}",
                    suggestion="Verify usd-exchange is installed and the USD file is valid.",
                    source="physics_materials",
                )
            ],
            "metrics": {},
        }


def author_placeholder_simready_demo(
    usd_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Author legacy placeholder physics for explicit demo fixtures only.

    This helper fabricates physical and grasp properties and must not be used as
    production repair or as evidence that an arbitrary imported asset is
    SimReady. Use the profile-driven ``geometry_repair`` package instead.
    """

    source = Path(usd_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return {
            "status": "fail",
            "message": f"USD source does not exist: {source}",
            "artifacts": {},
            "metrics": {"fixes_applied": 0},
            "issues": [
                _certificate_issue(
                    "repair_usd_source_present",
                    passed=False,
                    severity="error",
                    message=f"USD source does not exist: {source}",
                    suggestion="Export a USD before repair.",
                    source="repair_result",
                )
            ],
        }

    for suffix in (".usd", ".usda", ".usdc"):
        stale = target.with_suffix(suffix)
        if stale != target and stale.exists():
            stale.unlink()
    edit_target = (
        target
        if target.suffix.lower() == ".usda"
        else target.with_name(f".{target.stem}.repair.usda")
    )
    if edit_target != target and edit_target.exists():
        edit_target.unlink()
    shutil.copyfile(source, edit_target)
    fixes: list[str] = []
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

        stage = Usd.Stage.Open(str(edit_target))
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        fixes.extend(["set_up_axis_z", "set_meters_per_unit_1"])
        if not stage.GetDefaultPrim():
            roots = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
            if roots:
                stage.SetDefaultPrim(roots[0])
                fixes.append("set_default_prim")

        default_prim = stage.GetDefaultPrim()
        metadata = {
            "identifier": target.stem,
            "version": "1.0.0",
            "description": f"CAD Agent SimReady repaired asset {target.stem}",
            "source_asset": source.name,
            "generator": "cad_verifier.author_placeholder_simready_demo",
        }
        sidecar = target.with_suffix(".json")
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        layer_data = dict(stage.GetRootLayer().customLayerData or {})
        layer_data["SimReady_Metadata"] = json.dumps(metadata, sort_keys=True)
        stage.GetRootLayer().customLayerData = layer_data
        fixes.extend(["write_simready_sidecar_metadata", "write_simready_layer_metadata"])

        scope_parent = (
            str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else ""
        )
        looks_scope_path = f"{scope_parent}/Looks" if scope_parent else "/Looks"
        physics_scope_path = (
            f"{scope_parent}/PhysicsMaterials" if scope_parent else "/PhysicsMaterials"
        )

        UsdGeom.Scope.Define(stage, looks_scope_path)
        visual_material = UsdShade.Material.Define(
            stage, f"{looks_scope_path}/default_simready_visual"
        )
        shader = UsdShade.Shader.Define(
            stage, f"{looks_scope_path}/default_simready_visual/PreviewSurface"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.55, 0.58, 0.62)
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.58)
        visual_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        fixes.append("author_default_visual_material")

        UsdGeom.Scope.Define(stage, physics_scope_path)
        material = UsdShade.Material.Define(
            stage, f"{physics_scope_path}/default_simready_material"
        )
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(0.9)
        physics_material.CreateDynamicFrictionAttr(0.7)
        physics_material.CreateRestitutionAttr(0.05)
        physics_material.CreateDensityAttr(1000.0)
        fixes.append("author_default_physics_material")

        for prim in stage.Traverse():
            if prim.GetName() == "Joints" and not prim.GetTypeName():
                UsdGeom.Scope.Define(stage, prim.GetPath())
                fixes.append(f"type_joints_scope:{prim.GetPath()}")
            if _has_api(prim, UsdPhysics.CollisionAPI):
                try:
                    binding = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding.Bind(visual_material)
                except Exception:
                    pass
                try:
                    prim.CreateRelationship("material:binding:full").SetTargets(
                        [visual_material.GetPath()]
                    )
                    prim.CreateRelationship("material:binding:physics").SetTargets(
                        [material.GetPath()]
                    )
                    fixes.append(f"bind_collision_material:{prim.GetPath()}")
                except Exception:
                    pass
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                try:
                    binding = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding.Bind(visual_material)
                except Exception:
                    pass
                try:
                    prim.CreateRelationship("material:binding:full").SetTargets(
                        [visual_material.GetPath()]
                    )
                    prim.CreateRelationship("material:binding:physics").SetTargets(
                        [material.GetPath()]
                    )
                    fixes.append(f"bind_collision_material:{prim.GetPath()}")
                except Exception:
                    pass
                parent = prim.GetParent()
                if parent and parent.IsValid() and str(parent.GetName()) == "Collisions":
                    owner = parent.GetParent()
                    if owner and owner.IsValid():
                        parent = owner
                body_prim = parent if parent and parent.IsValid() else prim
                UsdPhysics.RigidBodyAPI.Apply(body_prim)
                mass_api = UsdPhysics.MassAPI.Apply(body_prim)
                if not mass_api.GetMassAttr().HasAuthoredValueOpinion():
                    mass_api.CreateMassAttr(0.1)
                if not mass_api.GetCenterOfMassAttr().HasAuthoredValueOpinion():
                    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
                if (
                    hasattr(mass_api, "CreateDiagonalInertiaAttr")
                    and not mass_api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion()
                ):
                    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))
                fixes.append(f"ensure_rigid_mass_collision:{body_prim.GetPath()}")
        if default_prim and default_prim.IsValid():
            grasp_path = f"{default_prim.GetPath()}/grasp_identifier_01"
            curve = UsdGeom.BasisCurves.Define(stage, grasp_path)
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            curve.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
            curve.CreatePointsAttr(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(-0.05, 0.0, 0.0),
                        Gf.Vec3f(0.05, 0.0, 0.0),
                    ]
                )
            )
            curve.CreateWidthsAttr(Vt.FloatArray([0.004]))
            curve.CreateExtentAttr(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(-0.052, -0.002, -0.002),
                        Gf.Vec3f(0.052, 0.002, 0.002),
                    ]
                )
            )
            try:
                UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(visual_material)
            except Exception:
                pass
            curve.GetPrim().CreateRelationship("material:binding:full").SetTargets(
                [visual_material.GetPath()]
            )
            fixes.append(f"author_grasp_vector:{grasp_path}")
        root_layer = stage.GetRootLayer()
        root_layer.Save()
        if edit_target != target:
            root_layer.Export(str(target))
            fixes.append("export_crate_usd")
            try:
                edit_target.unlink()
            except OSError:
                pass
        return {
            "status": "pass",
            "message": f"Applied {len(fixes)} USD repair action(s).",
            "artifacts": {"repaired_usd_path": str(target)},
            "metrics": {"fixes_applied": len(fixes), "fixes": fixes},
            "issues": [
                _certificate_issue(
                    "repair_usd_applied",
                    passed=True,
                    severity="info",
                    message=f"Applied {len(fixes)} USD repair action(s).",
                    suggestion=None,
                    source="repair_result",
                    metadata={"fixes": fixes},
                )
            ],
        }
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"USD repair failed: {exc}",
            "artifacts": {"repaired_usd_path": str(target)},
            "metrics": {"fixes_applied": len(fixes), "fixes": fixes},
            "issues": [
                _certificate_issue(
                    "repair_usd_failed",
                    passed=False,
                    severity="error",
                    message=f"USD repair failed: {exc}",
                    suggestion="Inspect the USD and repair source/export authoring.",
                    source="repair_result",
                    metadata={"fixes": fixes},
                )
            ],
        }


def repair_usd_for_simready(
    usd_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Deprecated compatibility alias for explicit legacy demo authoring."""

    import warnings

    warnings.warn(
        "repair_usd_for_simready fabricates placeholder physics and is deprecated; "
        "use geometry_repair.run_geometry_repair for imported assets",
        DeprecationWarning,
        stacklevel=2,
    )
    return author_placeholder_simready_demo(usd_path, output_path)


def local_runtime_smoke(
    usd_path: str | Path | None, physics_audit: dict[str, Any]
) -> dict[str, Any]:
    """Run the local/offline runtime-smoke substitute.

    A real Isaac/PhysX rollout is environment-dependent, so the default smoke
    proves that the stage loads and that physics authoring is structurally
    present. Server code may replace this with an external Isaac command.
    """

    if not usd_path:
        return {
            "status": "unavailable",
            "message": "No USD artifact was available for runtime smoke.",
            "issues": [],
            "metrics": {},
        }
    if physics_audit.get("status") == "fail":
        return {
            "status": "fail",
            "message": "Runtime smoke blocked by physics authoring failures.",
            "issues": [
                _certificate_issue(
                    "runtime_smoke_physics_authoring",
                    passed=False,
                    severity="error",
                    message="Runtime smoke blocked by physics authoring failures.",
                    suggestion="Fix USD physics authoring before dynamic simulation.",
                    source="runtime_smoke",
                )
            ],
            "metrics": {},
        }
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None")
        return {
            "status": "pass",
            "message": "USD stage loads and physics schemas are present; dynamic Isaac smoke not run.",
            "issues": [
                _certificate_issue(
                    "runtime_smoke_dynamic_isaac",
                    passed=True,
                    severity="warning",
                    message="Dynamic Isaac/PhysX rollout was not run in this environment.",
                    suggestion=(
                        "Set GEOMETRY_AGENT_ISAAC_SMOKE_COMMAND to run a "
                        "simulator-backed smoke test."
                    ),
                    source="runtime_smoke",
                    metadata={"mode": "local_structural"},
                )
            ],
            "metrics": {"mode": "local_structural"},
        }
    except Exception as exc:
        return {
            "status": "fail",
            "message": f"USD stage failed local runtime smoke: {exc}",
            "issues": [
                _certificate_issue(
                    "runtime_smoke_usd_load",
                    passed=False,
                    severity="error",
                    message=f"USD stage failed local runtime smoke: {exc}",
                    suggestion="Fix USD parsing/loading errors before simulator handoff.",
                    source="runtime_smoke",
                )
            ],
            "metrics": {},
        }


_AXIS_NAMES = ("x", "y", "z")
_CONTACT_TASK_TOKENS = (
    "insert",
    "plug",
    "socket",
    "cable",
    "ram",
    "slot",
    "drawer",
    "thread",
    "cap",
    "press",
    "fit",
    "mate",
    "grasp",
    "dex",
    "manipulat",
)
_INSERTION_TASK_TOKENS = (
    "insert",
    "plug",
    "socket",
    "ram",
    "slot",
    "drawer",
    "thread",
    "cap",
    "press",
    "fit",
)
_GRASP_TASK_TOKENS = ("grasp", "dex", "pick", "gripper", "finger", "hand", "manipulat")
_GRASP_AFFORDANCE_TOKENS = (
    "grasp",
    "grip",
    "handle",
    "finger",
    "pad",
    "knurl",
    "jaw",
    "suction",
)
_CLEARANCE_TOKENS = ("clearance", "tolerance", "fit", "gap", "allowance", "gd&t", "gdt")
_INSERTION_AXIS_TOKENS = (
    "insertion axis",
    "slide axis",
    "prismatic",
    "travel",
    "stroke",
)
_DATUM_REQUIRED_GDT = {
    "parallelism",
    "perpendicularity",
    "angularity",
    "position",
    "concentricity",
    "circular_runout",
    "total_runout",
}


def _has_token(text: str, tokens: tuple[str, ...]) -> bool:
    normalised = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    words = set(normalised.split())
    for token in tokens:
        needle = "".join(ch if ch.isalnum() else " " for ch in token.lower()).strip()
        if not needle:
            continue
        if " " in needle:
            if needle in normalised:
                return True
        elif len(needle) <= 4:
            if needle in words:
                return True
        elif needle in normalised:
            return True
    return False


def _is_contact_rich(task_l: str) -> bool:
    # ``plug-in`` is commonly a product adjective (for example, "plug-in
    # transformer pick-and-place"), not an instruction to insert a plug.
    # Preserve the unhyphenated command phrase "plug in" and explicit
    # insertion task names.
    return _has_token(task_l.replace("plug-in", ""), _CONTACT_TASK_TOKENS)


def _is_insertion_like(task_l: str) -> bool:
    return _has_token(task_l.replace("plug-in", ""), _INSERTION_TASK_TOKENS)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _finite_vector(value: Any, *, length: int | None = None) -> bool:
    try:
        seq = list(value)
    except Exception:
        return False
    if length is not None and len(seq) != length:
        return False
    return all(_finite_number(v) for v in seq)


def _normalised_axis(axis: Any) -> tuple[float, float, float] | None:
    if not _finite_vector(axis, length=3):
        return None
    x, y, z = (float(v) for v in axis)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        return None
    return (x / norm, y / norm, z / norm)


def _shape_bbox_record(name: str, shape: Any, kind: str | None) -> dict[str, Any] | None:
    try:
        bb = shape.bounding_box()
        mn = [float(v) for v in bb["min"]]
        mx = [float(v) for v in bb["max"]]
    except Exception:
        return None
    if len(mn) != 3 or len(mx) != 3:
        return None
    extents = [max(0.0, mx[i] - mn[i]) for i in range(3)]
    center = [(mn[i] + mx[i]) * 0.5 for i in range(3)]
    diag = math.sqrt(sum(e * e for e in extents))
    return {
        "part": name,
        "kind": kind,
        "min_mm": mn,
        "max_mm": mx,
        "extents_mm": extents,
        "center_mm": center,
        "diagonal_mm": diag,
    }


def _axis_overlap_and_gap(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[list[float], list[float], list[int]]:
    gaps: list[float] = []
    overlaps: list[float] = []
    directions: list[int] = []
    for axis in range(3):
        amin = float(a["min_mm"][axis])
        amax = float(a["max_mm"][axis])
        bmin = float(b["min_mm"][axis])
        bmax = float(b["max_mm"][axis])
        if amax < bmin:
            gaps.append(bmin - amax)
            overlaps.append(0.0)
            directions.append(1)
        elif bmax < amin:
            gaps.append(amin - bmax)
            overlaps.append(0.0)
            directions.append(-1)
        else:
            overlap = max(0.0, min(amax, bmax) - max(amin, bmin))
            gaps.append(-overlap)
            overlaps.append(overlap)
            directions.append(1 if b["center_mm"][axis] >= a["center_mm"][axis] else -1)
    return gaps, overlaps, directions


def _pair_contact_records(
    bbox_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    contact_interfaces: list[dict[str, Any]] = []
    mating_surfaces: list[dict[str, Any]] = []
    clearance_checks: list[dict[str, Any]] = []

    for a, b in combinations(bbox_records, 2):
        gaps, overlaps, directions = _axis_overlap_and_gap(a, b)
        overlap_axes = [idx for idx, gap in enumerate(gaps) if gap <= 0.0]
        if len(overlap_axes) < 2:
            continue

        separated = [idx for idx, gap in enumerate(gaps) if gap > 0.0]
        if separated:
            axis = min(separated, key=lambda idx: gaps[idx])
        else:
            axis = min(range(3), key=lambda idx: abs(gaps[idx]))
        face_axes = [idx for idx in range(3) if idx != axis]
        overlap_a = max(0.0, overlaps[face_axes[0]])
        overlap_b = max(0.0, overlaps[face_axes[1]])
        overlap_area = overlap_a * overlap_b
        if overlap_area <= 0.0:
            continue

        face_span = max(1e-9, min(overlap_a, overlap_b))
        clearance = max(0.0, gaps[axis])
        # Scale the proximity allowance to the actual mating face instead of
        # using a fixed part size; meter-scale scenes and tiny fixtures both
        # get the same relative treatment.
        if gaps[axis] > face_span * 0.5:
            continue

        normal = [0.0, 0.0, 0.0]
        normal[axis] = float(directions[axis])
        record = {
            "part_a": a["part"],
            "part_b": b["part"],
            "kind_a": a["kind"],
            "kind_b": b["kind"],
            "axis": _AXIS_NAMES[axis],
            "normal_from_a_to_b": normal,
            "clearance_mm": clearance,
            "overlap_area_mm2": overlap_area,
            "source": "bbox_pair",
            "passed": math.isfinite(clearance) and overlap_area > 0.0,
        }
        mating_surfaces.append(record)
        contact_interfaces.append(
            {
                "part_a": a["part"],
                "part_b": b["part"],
                "type": "bbox_pair_mating",
                "axis": record["axis"],
                "clearance_mm": clearance,
                "overlap_area_mm2": overlap_area,
                "source": "bbox_pair",
            }
        )
        clearance_checks.append(
            {
                "part_a": a["part"],
                "part_b": b["part"],
                "axis": record["axis"],
                "clearance_mm": clearance,
                "source": "bbox_pair",
                "passed": math.isfinite(clearance),
            }
        )

    return contact_interfaces, mating_surfaces, clearance_checks


def _result_text_blobs(result: Any) -> list[str]:
    blobs: list[str] = []
    for record in _shape_entry_records(result):
        blobs.append(str(record.get("name") or ""))
        kind = _shape_kind(record.get("shape"))
        if kind:
            blobs.append(str(kind))
        metadata = record.get("metadata") or {}
        if metadata:
            blobs.append(_metadata_text(metadata))
    for dim in getattr(result, "dimensions", []) or []:
        label = getattr(dim, "label", None)
        if label:
            blobs.append(str(label))
    for assertion in getattr(result, "assertions", []) or []:
        for attr in ("label", "message"):
            value = getattr(assertion, attr, None)
            if value:
                blobs.append(str(value))
    for ann in getattr(result, "gdt_annotations", []) or []:
        for attr in ("kind", "label", "notes", "modifier"):
            value = getattr(ann, attr, None)
            if value:
                blobs.append(str(value))
        refs = getattr(ann, "datum_refs", None)
        if refs:
            blobs.append(" ".join(str(r) for r in refs))
    for name, spec in (getattr(result, "params", {}) or {}).items():
        blobs.append(str(name))
        unit = getattr(spec, "unit", None)
        if unit:
            blobs.append(str(unit))
    return blobs


def _metadata_text(value: Any) -> str:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _metadata_text(to_dict())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_metadata_text(item)}" for key, item in value.items())
    if isinstance(value, list | tuple | set):
        return " ".join(_metadata_text(item) for item in value)
    return str(value)


def _has_result_evidence(result: Any, tokens: tuple[str, ...]) -> bool:
    return _has_token(" ".join(_result_text_blobs(result)).lower(), tokens)


def _gdt_records(result: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ann in getattr(result, "gdt_annotations", []) or []:
        kind = str(getattr(ann, "kind", "") or "")
        label = str(getattr(ann, "label", "") or "")
        tolerance = getattr(ann, "tolerance", None)
        datum_refs = list(getattr(ann, "datum_refs", []) or [])
        reasons: list[str] = []
        if not kind:
            reasons.append("missing GD&T kind")
        if kind == "datum":
            if not label:
                reasons.append("datum is missing a label")
        elif kind == "surface_roughness":
            if not _finite_number(tolerance) or float(tolerance) <= 0.0:
                reasons.append("surface roughness must be positive")
        else:
            if not _finite_number(tolerance) or float(tolerance) <= 0.0:
                reasons.append("tolerance must be positive")
            if kind in _DATUM_REQUIRED_GDT and not datum_refs:
                reasons.append(f"{kind} requires at least one datum reference")
        records.append(
            {
                "kind": kind,
                "label": label,
                "tolerance": None if tolerance is None else float(tolerance),
                "datum_refs": datum_refs,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
    return records


def _mass_property_record(name: str, props: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    mass_g = props.get("mass_g")
    volume_mm3 = props.get("volume_mm3")
    com = props.get("com")
    inertia = props.get("inertia")

    if not _finite_number(mass_g) or float(mass_g) <= 0.0:
        reasons.append("mass_g must be positive")
    if not _finite_number(volume_mm3) or float(volume_mm3) <= 0.0:
        reasons.append("volume_mm3 must be positive")
    if not _finite_vector(com, length=3):
        reasons.append("center of mass must be a finite 3-vector")

    inertia_diag: list[float] = []
    inertia_ok = True
    try:
        rows = [list(row) for row in inertia]
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            inertia_ok = False
        elif not all(_finite_number(v) for row in rows for v in row):
            inertia_ok = False
        else:
            inertia_diag = [float(rows[i][i]) for i in range(3)]
            trace = sum(inertia_diag)
            tolerance = max(1e-9, abs(trace) * 1e-9)
            if any(v < -tolerance for v in inertia_diag) or trace <= tolerance:
                inertia_ok = False
    except Exception:
        inertia_ok = False
    if not inertia_ok:
        reasons.append("inertia tensor must be finite with a positive diagonal trace")

    return {
        "part": name,
        "mass_g": None if not _finite_number(mass_g) else float(mass_g),
        "volume_mm3": None if not _finite_number(volume_mm3) else float(volume_mm3),
        "com_mm": list(com) if _finite_vector(com, length=3) else None,
        "inertia_diag_g_mm2": inertia_diag,
        "passed": not reasons,
        "reasons": reasons,
    }


def _joint_limit_records(assembly: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if assembly is None:
        return records
    for joint in getattr(assembly, "joints", {}).values():
        kind = str(getattr(joint, "kind", "") or "")
        axis = _normalised_axis(getattr(joint, "axis", None))
        min_val = getattr(joint, "min_val", None)
        max_val = getattr(joint, "max_val", None)
        default_val = getattr(joint, "default_val", 0.0)
        reasons: list[str] = []
        if kind in {"prismatic", "revolute"}:
            if axis is None:
                reasons.append("moving joint axis must be finite and non-zero")
            if min_val is None or max_val is None:
                reasons.append("moving joint must declare min/max limits")
            elif not (_finite_number(min_val) and _finite_number(max_val)):
                reasons.append("joint limits must be finite")
            elif float(max_val) <= float(min_val):
                reasons.append("joint max limit must be greater than min limit")
            elif not _finite_number(default_val) or not (
                float(min_val) <= float(default_val) <= float(max_val)
            ):
                reasons.append("default joint value must be inside limits")
        elif kind != "fixed":
            reasons.append(f"unsupported joint kind {kind!r}")
        records.append(
            {
                "name": getattr(joint, "name", ""),
                "kind": kind,
                "parent": getattr(joint, "parent", ""),
                "child": getattr(joint, "child", ""),
                "axis": axis,
                "min": min_val,
                "max": max_val,
                "default": default_val,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
    return records


def _insertion_axes_from_assembly(assembly: Any) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    if assembly is None:
        return axes
    for joint in getattr(assembly, "joints", {}).values():
        if getattr(joint, "kind", None) != "prismatic":
            continue
        axis = _normalised_axis(getattr(joint, "axis", None))
        axes.append(
            {
                "name": getattr(joint, "name", ""),
                "source": "joint_prismatic",
                "parent": getattr(joint, "parent", ""),
                "child": getattr(joint, "child", ""),
                "axis": axis,
                "min_mm": getattr(joint, "min_val", None),
                "max_mm": getattr(joint, "max_val", None),
                "default_mm": getattr(joint, "default_val", None),
                "passed": axis is not None,
            }
        )
    return axes


def _semantic_insertion_axes(
    mating_surfaces: list[dict[str, Any]],
    result: Any,
) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    if not _has_result_evidence(result, _INSERTION_AXIS_TOKENS):
        return axes
    for surface in mating_surfaces:
        normal = surface.get("normal_from_a_to_b")
        if not _finite_vector(normal, length=3):
            continue
        axes.append(
            {
                "name": f"{surface['part_a']}_{surface['part_b']}_{surface['axis']}_annotation_axis",
                "source": "annotation_plus_mating_surface",
                "part_a": surface["part_a"],
                "part_b": surface["part_b"],
                "axis": [float(v) for v in normal],
                "passed": True,
            }
        )
    return axes


def _coerce_metadata_dict(value: Any) -> dict[str, Any] | None:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
        except Exception:
            return None
    if isinstance(value, dict):
        return dict(value)
    return None


def _walk_metadata_dicts(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    data = _coerce_metadata_dict(value)
    if data is not None:
        out.append(data)
        for item in data.values():
            out.extend(_walk_metadata_dicts(item))
        return out
    if isinstance(value, list | tuple | set):
        for item in value:
            out.extend(_walk_metadata_dicts(item))
    return out


def _is_connector_like(data: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in data}
    text = _metadata_text(data).lower()
    has_axis = any(key in keys for key in ("axis", "insertion_axis", "normal_axis", "normal"))
    has_connector_word = "connector" in text or "mating" in text or "receiver" in text
    has_gender = any(
        str(data.get(key, "")).lower() in {"male", "female"} for key in ("gender", "role")
    )
    has_contact = any(
        key in keys for key in ("clearance", "clearance_mm", "positive_clearance_mm", "mating_face")
    )
    return has_axis and (has_connector_word or has_gender or has_contact)


def _axis_from_metadata(data: dict[str, Any]) -> list[float] | None:
    for key in ("insertion_axis", "axis", "normal_axis", "normal"):
        axis = _normalised_axis(data.get(key))
        if axis is not None:
            return axis
    return None


def _metadata_clearance_mm(data: dict[str, Any]) -> float | None:
    for key in (
        "clearance_mm",
        "positive_clearance_mm",
        "clearance",
        "gap_mm",
        "tolerance_mm",
    ):
        value = data.get(key)
        if _finite_number(value):
            return float(value)
    measurements = data.get("measurements")
    if isinstance(measurements, dict):
        nested = _metadata_clearance_mm(measurements)
        if nested is not None:
            return nested
    return None


def _metadata_gender(data: dict[str, Any], part_text: str) -> str | None:
    for key in ("gender", "role"):
        value = str(data.get(key, "") or "").lower()
        if value in {"male", "female"}:
            return value
    text = f"{part_text} {_metadata_text(data)}".lower()
    tokens = _semantic_tokens(text)
    if {"female", "receiver", "socket"} & tokens:
        return "female"
    if {"male", "plug", "insert"} & tokens:
        return "male"
    return None


def _semantic_tokens(text: str) -> set[str]:
    normalised = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    return {token for token in normalised.split("_") if token}


def _scene_connector_records(
    entry_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    connectors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in entry_records:
        part = str(record.get("name") or "")
        kind = _shape_kind(record.get("shape"))
        part_text = f"{part} {kind or ''}"
        metadata = record.get("metadata") or {}
        for data in _walk_metadata_dicts(metadata):
            if not _is_connector_like(data):
                continue
            axis = _axis_from_metadata(data)
            if axis is None:
                continue
            name = str(data.get("name") or data.get("connectorName") or f"{part}_connector")
            gender = _metadata_gender(data, part_text)
            key = (part, name, gender or "")
            if key in seen:
                continue
            seen.add(key)
            connectors.append(
                {
                    "name": name,
                    "part": part,
                    "kind": kind,
                    "gender": gender,
                    "axis": axis,
                    "clearance_mm": _metadata_clearance_mm(data),
                    "source_metadata": data,
                }
            )
    return connectors


def _scene_metadata_task_records(
    entry_records: list[dict[str, Any]],
    bbox_records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    connectors = _scene_connector_records(entry_records)
    insertion_axes: list[dict[str, Any]] = []
    contact_interfaces: list[dict[str, Any]] = []
    mating_surfaces: list[dict[str, Any]] = []
    clearance_checks: list[dict[str, Any]] = []

    for connector in connectors:
        insertion_axes.append(
            {
                "name": connector["name"],
                "source": "scene_connector",
                "part": connector["part"],
                "gender": connector.get("gender"),
                "axis": connector["axis"],
                "clearance_mm": connector.get("clearance_mm"),
                "passed": True,
            }
        )

    males = [conn for conn in connectors if conn.get("gender") == "male"]
    females = [conn for conn in connectors if conn.get("gender") == "female"]
    for male in males[:8]:
        for female in females[:8]:
            if male["part"] == female["part"]:
                continue
            axis = male.get("axis") or female.get("axis") or [0.0, 0.0, -1.0]
            clearance = male.get("clearance_mm")
            if clearance is None:
                clearance = female.get("clearance_mm")
            surface = {
                "part_a": male["part"],
                "part_b": female["part"],
                "kind_a": male.get("kind"),
                "kind_b": female.get("kind"),
                "axis": "semantic",
                "normal_from_a_to_b": axis,
                "clearance_mm": 0.0 if clearance is None else float(clearance),
                "overlap_area_mm2": None,
                "source": "scene_connector_pair",
                "passed": True,
            }
            mating_surfaces.append(surface)
            contact_interfaces.append(
                {
                    "part_a": male["part"],
                    "part_b": female["part"],
                    "type": "scene_connector_mating",
                    "axis": axis,
                    "clearance_mm": surface["clearance_mm"],
                    "source": "scene_connector_pair",
                }
            )
            if clearance is not None:
                clearance_checks.append(
                    {
                        "part_a": male["part"],
                        "part_b": female["part"],
                        "axis": axis,
                        "clearance_mm": float(clearance),
                        "source": "scene_connector_pair",
                        "passed": True,
                    }
                )

    if not mating_surfaces:
        semantic = _semantic_male_female_pair_records(bbox_records)
        if semantic is not None:
            contact, surface, clearance = semantic
            contact_interfaces.append(contact)
            mating_surfaces.append(surface)
            clearance_checks.append(clearance)
            insertion_axes.append(
                {
                    "name": f"{surface['part_a']}_{surface['part_b']}_semantic_insertion_axis",
                    "source": "semantic_male_female_pair",
                    "part_a": surface["part_a"],
                    "part_b": surface["part_b"],
                    "axis": surface["normal_from_a_to_b"],
                    "passed": True,
                }
            )
    return contact_interfaces, mating_surfaces, clearance_checks, insertion_axes


def _semantic_male_female_pair_records(
    bbox_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    males = [
        rec
        for rec in bbox_records
        if "male" in _semantic_tokens(f"{rec.get('part') or ''} {rec.get('kind') or ''}")
    ]
    females = [
        rec
        for rec in bbox_records
        if {"female", "receiver", "socket"}
        & _semantic_tokens(f"{rec.get('part') or ''} {rec.get('kind') or ''}")
    ]
    if not males or not females:
        return None
    male = max(males, key=lambda rec: rec.get("diagonal_mm") or 0.0)
    female = max(females, key=lambda rec: rec.get("diagonal_mm") or 0.0)
    if male["part"] == female["part"]:
        return None
    axis = [0.0, 0.0, -1.0]
    surface = {
        "part_a": male["part"],
        "part_b": female["part"],
        "kind_a": male.get("kind"),
        "kind_b": female.get("kind"),
        "axis": "semantic",
        "normal_from_a_to_b": axis,
        "clearance_mm": 0.0,
        "overlap_area_mm2": None,
        "source": "semantic_male_female_pair",
        "passed": True,
    }
    contact = {
        "part_a": male["part"],
        "part_b": female["part"],
        "type": "semantic_male_female_mating",
        "axis": axis,
        "clearance_mm": 0.0,
        "source": "semantic_male_female_pair",
    }
    clearance = {
        "part_a": male["part"],
        "part_b": female["part"],
        "axis": axis,
        "clearance_mm": 0.0,
        "source": "semantic_male_female_pair",
        "passed": True,
    }
    return contact, surface, clearance


def _grasp_affordance_records(
    bbox_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rec in bbox_records:
        text = f"{rec.get('part') or ''} {rec.get('kind') or ''}".lower()
        hits = [token for token in _GRASP_AFFORDANCE_TOKENS if token in text]
        if not hits:
            continue
        records.append(
            {
                "part": rec["part"],
                "kind": rec["kind"],
                "source": "semantic_kind_or_part_name",
                "tokens": hits,
                "center_mm": rec["center_mm"],
                "extents_mm": rec["extents_mm"],
                "passed": True,
            }
        )
    return records


def _semantic_grasp_affordance_records(
    bbox_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    graspable_tokens = (
        "male",
        "plug",
        "insert",
        "shaft",
        "cap",
        "sleeve",
        "collar",
        "knob",
        "drawer",
        "tray",
        "jaw",
        "finger",
        "handle",
        "pad",
        "grip",
    )
    for rec in bbox_records:
        text = f"{rec.get('part') or ''} {rec.get('kind') or ''}".lower()
        semantic = _semantic_tokens(text)
        hits = [
            token
            for token in graspable_tokens
            if token in semantic or (token not in {"male"} and token in text)
        ]
        if not hits:
            continue
        records.append(
            {
                "part": rec["part"],
                "kind": rec["kind"],
                "source": "semantic_graspable_part_name",
                "tokens": hits,
                "center_mm": rec["center_mm"],
                "extents_mm": rec["extents_mm"],
                "passed": True,
            }
        )
    return records


def _dedupe_named_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (
            str(record.get("source") or ""),
            str(record.get("name") or record.get("part") or record.get("part_a") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def validate_result_for_sim(
    result: Any,
    *,
    usd_path: str | Path | None = None,
    max_triangles: int = 250_000,
    require_kinds: bool = True,
    task_type: str | None = None,
    asset_type: str | None = None,
    profile_id: str | None = None,
    require_separate_collision_geometry: bool = False,
    require_external_textures: bool = False,
    max_collision_triangles: int = 256,
    max_collision_to_visual_ratio: float = 0.25,
) -> SimReadyReport:
    """Run static simulation-readiness checks against a cad_kernel Result."""

    profile = resolve_sim_ready_profile(
        result,
        task_type=task_type,
        asset_type=asset_type,
        profile_id=profile_id,
    )
    issues: list[SimReadyIssue] = []
    warnings: list[str] = []
    sim_budget: dict[str, Any] = {
        "max_triangles": max_triangles,
        "part_count": 0,
        "triangle_count": 0,
        "mass_g_total": None,
    }
    mesh_topology = (
        audit_usd_mesh_topology(usd_path, max_triangles=max_triangles) if usd_path else {}
    )
    geometry_handoff = (
        audit_usd_geometry_handoff(
            usd_path,
            max_collision_triangles=max_collision_triangles,
            max_collision_to_visual_ratio=max_collision_to_visual_ratio,
            require_separate_collisions=require_separate_collision_geometry,
            require_external_textures=require_external_textures,
        )
        if usd_path
        else {}
    )
    contact_interfaces: list[dict[str, Any]] = []
    collision_proxies: list[dict[str, Any]] = []
    bbox_records: list[dict[str, Any]] = []
    mass_properties: list[dict[str, Any]] = []
    gdt_checks = _gdt_records(result)
    task_l = (task_type or "").lower()
    contact_rich = _is_contact_rich(task_l) or profile.requires_task_semantics
    insertion_like = (
        _is_insertion_like(task_l) or profile.asset_type == "insertion_or_fixture_asset"
    )
    grasp_required = (
        contact_rich
        or _has_token(task_l, _GRASP_TASK_TOKENS)
        or profile.asset_type == "robotics_manipulation_asset"
    )

    if not getattr(result, "success", False):
        issues.append(
            SimReadyIssue(
                "cad_kernel_build",
                False,
                "error",
                "; ".join(getattr(result, "errors", []) or ["cad_kernel execution failed"]),
                "Fix CAD source errors before simulation validation.",
            )
        )
        return SimReadyReport(
            False,
            issues=issues,
            warnings=warnings,
            sim_budget=sim_budget,
            mesh_topology=mesh_topology,
            geometry_handoff=geometry_handoff,
            profile=profile.to_dict(),
        )

    assertion_failures = [a for a in getattr(result, "assertions", []) if not a.passed]
    issues.append(
        SimReadyIssue(
            "inline_assertions",
            not assertion_failures,
            "error",
            "all assertions passed"
            if not assertion_failures
            else f"{len(assertion_failures)} assertion(s) failed",
            "Fix failing verify.* probes before simulation handoff.",
        )
    )

    entry_records = _shape_entry_records(result)
    entries = [(record["name"], record["shape"]) for record in entry_records]
    sim_budget["part_count"] = len(entries)
    issues.append(
        SimReadyIssue(
            "has_sim_parts",
            bool(entries),
            "error",
            f"{len(entries)} shape part(s) available",
            "Return a Shape, Assembly, or multi-part scene.",
        )
    )

    missing_kinds: list[str] = []
    total_tris = 0
    mass_total = 0.0
    mass_available = False
    for name, shape in entries:
        try:
            report = all_checks(shape)
            for check in report.checks:
                severity = "info" if check.passed else _static_check_severity(check.name)
                issues.append(
                    SimReadyIssue(
                        f"{name}:{check.name}",
                        check.passed,
                        severity,
                        check.message,
                    )
                )
        except Exception as exc:
            issues.append(
                SimReadyIssue(
                    f"{name}:static_checks",
                    False,
                    "error",
                    f"Static checks failed: {exc}",
                )
            )

        kind = _shape_kind(shape)
        if not kind:
            missing_kinds.append(name)

        tris = _mesh_triangle_count(shape)
        if tris is not None:
            total_tris += tris
            if tris > max_triangles:
                issues.append(
                    SimReadyIssue(
                        f"{name}:mesh_budget",
                        False,
                        "warning",
                        f"{tris} triangles exceeds per-asset budget {max_triangles}",
                        "Generate a lower-resolution render mesh or a simpler collision proxy.",
                    )
                )

        bbox_record = _shape_bbox_record(name, shape, kind)
        if bbox_record is not None:
            bbox_records.append(bbox_record)
            extents = list(bbox_record["extents_mm"])
            collision_proxies.append(
                {
                    "part": name,
                    "kind": kind,
                    "proxy": "bbox",
                    "extents_mm": extents,
                    "source": "shape_bounding_box",
                }
            )
            # Large flat lower faces are likely support/contact zones; record
            # the bbox floor as a conservative contact hint for downstream tools.
            contact_interfaces.append(
                {
                    "part": name,
                    "kind": kind,
                    "type": "bbox_floor",
                    "role": "support",
                    "z_mm": float(bbox_record["min_mm"][2]),
                    "footprint_mm": [extents[0], extents[1]],
                    "source": "shape_bounding_box",
                }
            )
        else:
            issues.append(
                SimReadyIssue(
                    f"{name}:collision_proxy_bbox",
                    False,
                    "error" if contact_rich else "warning",
                    "could not compute a bounding-box collision proxy",
                    "Ensure geometry is valid enough to expose a bounding box.",
                )
            )

        try:
            props = shape.mass_props()
            mass_record = _mass_property_record(name, props)
            mass_properties.append(mass_record)
            mass_g = mass_record["mass_g"]
            if mass_g is not None:
                mass_total += float(mass_g)
                mass_available = True
            issues.append(
                SimReadyIssue(
                    f"{name}:mass_properties",
                    mass_record["passed"],
                    "error" if not mass_record["passed"] else "info",
                    "mass/inertia properties are finite and positive"
                    if mass_record["passed"]
                    else "; ".join(mass_record["reasons"]),
                    "Bind material density or simplify geometry before physics handoff.",
                )
            )
        except Exception as exc:
            mass_properties.append(
                {
                    "part": name,
                    "mass_g": None,
                    "volume_mm3": None,
                    "com_mm": None,
                    "inertia_diag_g_mm2": [],
                    "passed": False,
                    "reasons": [f"mass/inertia estimate unavailable: {exc}"],
                }
            )
            issues.append(
                SimReadyIssue(
                    f"{name}:mass_properties",
                    False,
                    "error",
                    f"mass/inertia estimate unavailable: {exc}",
                    "Check tessellation/material density before Isaac Lab handoff.",
                )
            )

    sim_budget["triangle_count"] = total_tris
    if mass_available:
        sim_budget["mass_g_total"] = mass_total
    issues.append(
        SimReadyIssue(
            "mesh_triangle_budget",
            total_tris <= max_triangles,
            "error" if contact_rich else "warning",
            f"{total_tris} total triangles against budget {max_triangles}",
            "Use simplified render meshes and explicit collision proxies for simulation assets.",
        )
    )
    proxy_budget = max(1, len(entries) * 3)
    sim_budget["collision_proxy_budget"] = {
        "proxy_count": len(collision_proxies),
        "max_proxy_count": proxy_budget,
    }
    proxy_budget_ok = (
        len(collision_proxies) >= len(entries) and len(collision_proxies) <= proxy_budget
    )
    issues.append(
        SimReadyIssue(
            "collision_proxy_budget",
            proxy_budget_ok,
            "error" if contact_rich else "warning",
            f"{len(collision_proxies)} proxy recommendation(s) for {len(entries)} part(s)",
            "Keep collision proxies explicit and bounded; prefer primitive proxies over render meshes.",
        )
    )
    if require_kinds:
        issues.append(
            SimReadyIssue(
                "semantic_kinds",
                not missing_kinds,
                "error",
                "all parts have asset_gen:kind labels"
                if not missing_kinds
                else f"missing kind labels: {missing_kinds}",
                "Call .kind('<semantic_label>') on every meaningful part.",
            )
        )
    elif missing_kinds:
        warnings.append(f"missing semantic kind labels: {missing_kinds}")

    issues.extend(_usd_stage_checks(usd_path))
    for item in geometry_handoff.get("issues", []):
        issues.append(
            SimReadyIssue(
                name=str(item.get("name") or "geometry_handoff"),
                passed=bool(item.get("passed", False)),
                severity=str(item.get("severity") or "error"),
                message=str(item.get("message") or ""),
                suggestion=item.get("suggestion"),
            )
        )

    assembly = getattr(result, "assembly", None)
    joint_limits = _joint_limit_records(assembly)
    insertion_axes = _insertion_axes_from_assembly(assembly)
    if assembly is not None:
        joints = list(assembly.joints.values())
        issues.append(
            SimReadyIssue(
                "assembly_joints_declared",
                bool(joints),
                "warning",
                f"{len(joints)} joint(s) declared",
                "Add fixed/revolute/prismatic joints for articulated assets.",
            )
        )
        moving_joint_limits = [r for r in joint_limits if r["kind"] in {"prismatic", "revolute"}]
        if moving_joint_limits:
            joints_ok = all(r["passed"] for r in moving_joint_limits)
            issues.append(
                SimReadyIssue(
                    "joint_limits_declared",
                    joints_ok,
                    "error" if not joints_ok else "info",
                    "all moving joints declare finite axes, limits, and in-range defaults"
                    if joints_ok
                    else "one or more moving joints are missing finite axes, limits, or defaults",
                    "Set min/max/default limits for every prismatic or revolute joint.",
                )
            )
    elif len(entries) > 1:
        warnings.append("multi-part scene has no Assembly joint metadata")

    pair_contacts, mating_surfaces, clearance_checks = _pair_contact_records(bbox_records)
    (
        metadata_contacts,
        metadata_mating_surfaces,
        metadata_clearance_checks,
        metadata_insertion_axes,
    ) = _scene_metadata_task_records(entry_records, bbox_records)
    contact_interfaces.extend(pair_contacts)
    contact_interfaces.extend(metadata_contacts)
    mating_surfaces.extend(metadata_mating_surfaces)
    clearance_checks.extend(metadata_clearance_checks)
    insertion_axes.extend(_semantic_insertion_axes(mating_surfaces, result))
    insertion_axes.extend(metadata_insertion_axes)
    grasp_affordances = _dedupe_named_records(
        [
            *_grasp_affordance_records(bbox_records),
            *_semantic_grasp_affordance_records(bbox_records),
        ]
    )

    if gdt_checks:
        gdt_ok = all(record["passed"] for record in gdt_checks)
        issues.append(
            SimReadyIssue(
                "gdt_tolerance_checks",
                gdt_ok,
                "error" if not gdt_ok else "info",
                "all GD&T annotations have valid tolerances/datums"
                if gdt_ok
                else "one or more GD&T annotations are missing tolerance or datum evidence",
                "Use datum, form, orientation, and position annotations with positive tolerances.",
            )
        )

    if contact_rich:
        pair_contact_count = sum(1 for c in contact_interfaces if c.get("type") != "bbox_floor")
        issues.append(
            SimReadyIssue(
                "contact_interfaces_declared",
                pair_contact_count > 0,
                "error",
                f"{pair_contact_count} pair contact interface hint(s)",
                "Expose mating/contact surfaces and insertion-facing regions for DEX tasks.",
            )
        )
        if insertion_like:
            issues.append(
                SimReadyIssue(
                    "contact_task_multiple_parts",
                    len(entries) >= 2,
                    "error",
                    f"{len(entries)} part(s) available for contact-rich task",
                    "Model both tool/object and fixture/receptacle when generating insertion tasks.",
                )
            )
            declared_axes = [
                a
                for a in insertion_axes
                if a.get("passed")
                and a.get("source")
                in {
                    "joint_prismatic",
                    "annotation_plus_mating_surface",
                    "scene_connector",
                    "semantic_male_female_pair",
                }
            ]
            issues.append(
                SimReadyIssue(
                    "insertion_axes_declared",
                    bool(declared_axes),
                    "error",
                    f"{len(declared_axes)} insertion axis record(s)",
                    "Declare a prismatic joint or insertion-axis annotation tied to mating surfaces.",
                )
            )
            issues.append(
                SimReadyIssue(
                    "mating_surfaces_declared",
                    bool(mating_surfaces),
                    "error",
                    f"{len(mating_surfaces)} inferred mating surface pair(s)",
                    "Expose opposing plug/socket, ram/slot, cap/thread, or drawer/rail surfaces.",
                )
            )
            clearance_evidence = bool(clearance_checks) or _has_result_evidence(
                result, _CLEARANCE_TOKENS
            )
            issues.append(
                SimReadyIssue(
                    "clearance_checks_declared",
                    clearance_evidence,
                    "error",
                    f"{len(clearance_checks)} bbox clearance check(s)",
                    "Add explicit clearance/fit/tolerance dimensions or GD&T for contact paths.",
                )
            )
        tolerance_evidence = (
            bool(getattr(result, "dimensions", None))
            or bool(getattr(result, "gdt_annotations", None))
            or _has_result_evidence(result, _CLEARANCE_TOKENS)
        )
        issues.append(
            SimReadyIssue(
                "contact_tolerance_metadata",
                tolerance_evidence,
                "error",
                "clearance/tolerance metadata is present"
                if tolerance_evidence
                else "no explicit dimensions, clearance labels, or GD&T annotations recorded",
                "Add clearance/tolerance annotations for contact-rich manipulation assets.",
            )
        )
        if grasp_required:
            issues.append(
                SimReadyIssue(
                    "grasp_affordances_declared",
                    bool(grasp_affordances),
                    "error",
                    f"{len(grasp_affordances)} grasp affordance hint(s)",
                    "Tag handles, pads, jaws, knurls, or suction/grip regions for DEX planners.",
                )
            )
        issues.append(
            SimReadyIssue(
                "mass_inertia_enforced",
                bool(mass_properties) and all(record["passed"] for record in mass_properties),
                "error",
                f"{len(mass_properties)} mass/inertia record(s)",
                "Every sim-ready part must have positive mass, volume, COM, and inertia.",
            )
        )

    passed = not any(not i.passed and i.severity == "error" for i in issues)
    return SimReadyReport(
        passed=passed,
        issues=issues,
        warnings=warnings,
        sim_budget=sim_budget,
        contact_interfaces=contact_interfaces,
        recommended_collision_proxies=collision_proxies,
        insertion_axes=insertion_axes,
        mating_surfaces=mating_surfaces,
        clearance_checks=clearance_checks,
        grasp_affordances=grasp_affordances,
        joint_limits=joint_limits,
        mass_properties=mass_properties,
        gdt_checks=gdt_checks,
        mesh_topology=mesh_topology,
        geometry_handoff=geometry_handoff,
        profile=profile.to_dict(),
    )


def _level_status_from_issues(issues: list[dict[str, Any]]) -> str:
    if any(
        not issue.get("passed", False) and str(issue.get("severity")) in {"error", "failure"}
        for issue in issues
    ):
        return "fail"
    if any(not issue.get("passed", True) or issue.get("severity") == "warning" for issue in issues):
        return "warn"
    return "pass"


def _usd_stage_level(
    report: SimReadyReport,
    usd_path: str | Path | None,
    *,
    name: str = "usd_valid",
    mandatory: bool = True,
) -> SimReadyLevel:
    usd_issues = [
        _legacy_issue_dict(issue)
        for issue in report.issues
        if issue.name.startswith("usd_") or issue.name == "usd_present"
    ]
    status = _level_status_from_issues(usd_issues)
    if not usd_issues:
        status = "unavailable"
    if mandatory and (not usd_path) and status in {"warn", "unavailable"}:
        status = "incomplete"
    return SimReadyLevel(
        name=name,
        status=status,
        mandatory=mandatory,
        message="USD stage units/up-axis passed"
        if status == "pass"
        else "USD stage units/up-axis need attention",
        issues=usd_issues,
        artifacts={"usd_path": str(usd_path)} if usd_path else {},
        metadata={"validator": "cad_verifier.usd_stage_checks"},
    )


def _static_cad_level(
    report: SimReadyReport,
    *,
    name: str = "static_cad",
    mandatory: bool = True,
) -> SimReadyLevel:
    def in_static_scope(issue: SimReadyIssue) -> bool:
        if issue.name.startswith("usd_") or issue.name == "usd_present":
            return False
        return issue.name not in _TASK_SEMANTIC_ISSUE_NAMES

    blocking = [
        _legacy_issue_dict(issue) for issue in report.blocking_issues if in_static_scope(issue)
    ]
    warning_issues = [
        _legacy_issue_dict(issue)
        for issue in report.issues
        if in_static_scope(issue) and not issue.passed and issue.severity == "warning"
    ]
    status = "fail" if blocking else ("warn" if warning_issues else "pass")
    return SimReadyLevel(
        name=name,
        status=status,
        mandatory=mandatory,
        message="CAD build and static geometry checks passed"
        if status == "pass"
        else "CAD build/static geometry checks produced findings",
        issues=blocking + warning_issues,
        metrics=dict(report.sim_budget),
    )


def _raw_audit_level(
    name: str,
    raw: dict[str, Any] | None,
    *,
    mandatory: bool,
    validator: str,
    unavailable_message: str,
) -> SimReadyLevel:
    if not raw:
        return SimReadyLevel(
            name=name,
            status="unavailable",
            mandatory=mandatory,
            message=unavailable_message,
            metadata={"validator": validator},
        )
    status = str(raw.get("status") or "unavailable")
    if status == "error":
        status = "fail"
    if mandatory and status == "unavailable":
        status = "incomplete"
    return SimReadyLevel(
        name=name,
        status=status,
        mandatory=mandatory,
        message=str(raw.get("message") or ""),
        issues=list(raw.get("issues") or []),
        artifacts=dict(raw.get("artifacts") or {}),
        metrics=dict(raw.get("metrics") or {}),
        metadata={"validator": validator},
    )


def _mesh_topology_level(raw: dict[str, Any] | None, *, mandatory: bool) -> SimReadyLevel:
    return _raw_audit_level(
        "mesh_topology",
        raw,
        mandatory=mandatory,
        validator="cad_verifier.audit_usd_mesh_topology",
        unavailable_message="USD mesh topology audit was not run.",
    )


def _physics_authoring_level(
    raw: dict[str, Any] | None,
    *,
    mandatory: bool = True,
    name: str = "physics_authored",
) -> SimReadyLevel:
    return _raw_audit_level(
        name,
        raw,
        mandatory=mandatory,
        validator="cad_verifier.audit_usd_physics_authoring",
        unavailable_message="USD physics authoring audit was not run.",
    )


def _physics_materials_level(raw: dict[str, Any] | None, *, mandatory: bool) -> SimReadyLevel:
    return _raw_audit_level(
        "physics_materials",
        raw,
        mandatory=mandatory,
        validator="cad_verifier.audit_usd_physics_materials",
        unavailable_message="USD physics material audit was not run.",
    )


def _articulation_level(
    report: SimReadyReport,
    physics_authoring: dict[str, Any] | None,
    *,
    mandatory: bool,
) -> SimReadyLevel:
    metrics = dict((physics_authoring or {}).get("metrics") or {})
    issues: list[dict[str, Any]] = []
    if not mandatory:
        return SimReadyLevel(
            name="articulation",
            status="pass",
            mandatory=False,
            message="Articulation semantics are not required for this profile.",
            metrics=metrics,
            metadata={"validator": "cad_verifier.articulation_profile"},
        )

    articulation_roots = int(metrics.get("articulation_root_count") or 0)
    joint_count = int(metrics.get("joint_count") or 0)
    moving_joint_count = int(metrics.get("moving_joint_count") or 0)
    binding_count = int(metrics.get("joint_binding_count") or 0)
    limited_count = int(metrics.get("limited_moving_joint_count") or 0)
    issues.extend(
        [
            _certificate_issue(
                "articulation_root_present",
                passed=articulation_roots > 0,
                severity="info" if articulation_roots > 0 else "error",
                message=f"{articulation_roots} articulation root prim(s)",
                suggestion="Apply UsdPhysics.ArticulationRootAPI to the assembly root.",
                source="articulation",
            ),
            _certificate_issue(
                "articulation_moving_joints_present",
                passed=moving_joint_count > 0,
                severity="info" if moving_joint_count > 0 else "error",
                message=f"{moving_joint_count} moving joint(s)",
                suggestion="Author revolute or prismatic joints for articulated assets.",
                source="articulation",
            ),
            _certificate_issue(
                "articulation_joint_bindings_present",
                passed=joint_count > 0 and binding_count == joint_count,
                severity="info" if joint_count > 0 and binding_count == joint_count else "error",
                message=(
                    f"{binding_count}/{joint_count} joint(s) bind body0/body1 "
                    "or pin a fixed joint to world"
                ),
                suggestion=(
                    "Bind every physics joint to body relationships; fixed root joints may "
                    "pin one side to world."
                ),
                source="articulation",
            ),
            _certificate_issue(
                "articulation_moving_joint_limits_present",
                passed=moving_joint_count > 0 and limited_count == moving_joint_count,
                severity="info"
                if moving_joint_count > 0 and limited_count == moving_joint_count
                else "error",
                message=f"{limited_count}/{moving_joint_count} moving joint(s) have limits",
                suggestion="Author lower and upper limits for every moving joint.",
                source="articulation",
            ),
        ]
    )

    if report.joint_limits:
        source_limits_ok = all(record.get("passed") for record in report.joint_limits)
        issues.append(
            _certificate_issue(
                "articulation_source_joint_limits_valid",
                passed=source_limits_ok,
                severity="info" if source_limits_ok else "error",
                message="source joint limits are finite and in range"
                if source_limits_ok
                else "one or more source joint limits are invalid",
                suggestion="Set finite axis/min/max/default values in the source assembly.",
                source="articulation",
            )
        )

    status = _level_status_from_issues(issues)
    return SimReadyLevel(
        name="articulation",
        status=status,
        mandatory=True,
        message="Articulation authoring passed"
        if status == "pass"
        else "Articulation authoring needs repair",
        issues=issues,
        metrics=metrics,
        metadata={"validator": "cad_verifier.articulation_profile"},
    )


def _external_usd_validator_level(
    raw: dict[str, Any] | None,
    *,
    mandatory: bool = False,
) -> SimReadyLevel:
    if not raw:
        return SimReadyLevel(
            name="usd_asset_validator",
            status="incomplete" if mandatory else "unavailable",
            mandatory=mandatory,
            message="Omni Asset Validator was not run.",
            metadata={"validator": "omniverse-asset-validator"},
        )
    status = str(raw.get("status") or "unavailable")
    if status == "unavailable":
        return SimReadyLevel(
            name="usd_asset_validator",
            status="incomplete" if mandatory else "unavailable",
            mandatory=mandatory,
            message=str(raw.get("message") or "Omni Asset Validator is unavailable."),
            issues=list(raw.get("issues") or []),
            metadata={"validator": "omniverse-asset-validator"},
        )
    if status != "success":
        return SimReadyLevel(
            name="usd_asset_validator",
            status="fail" if mandatory else "warn",
            mandatory=mandatory,
            message=str(raw.get("error") or raw.get("message") or "USD validation failed to run."),
            issues=[
                _certificate_issue(
                    "omni_asset_validator_run",
                    passed=False,
                    severity="warning",
                    message=str(raw.get("error") or raw.get("message") or "validator failed"),
                    suggestion="Install/repair omniverse-asset-validator or inspect the USD manually.",
                    source="usd_asset_validator",
                )
            ],
            metadata={"validator": "omniverse-asset-validator"},
        )

    summary = dict(raw.get("summary") or {})
    raw_issues = list(raw.get("issues") or [])
    issues = [
        _certificate_issue(
            f"omni_asset_validator:{issue.get('rule') or 'unknown'}",
            passed=str(issue.get("severity")) not in {"failure", "error"},
            severity="error" if str(issue.get("severity")) in {"failure", "error"} else "warning",
            message=str(issue.get("message") or ""),
            suggestion=issue.get("suggestion"),
            source="usd_asset_validator",
            metadata={
                "category": issue.get("category"),
                "at": issue.get("at"),
                "rule": issue.get("rule"),
            },
        )
        for issue in raw_issues
    ]
    status_out = "pass" if summary.get("is_valid", True) else "fail"
    if status_out == "pass" and issues:
        status_out = "warn"
    return SimReadyLevel(
        name="usd_asset_validator",
        status=status_out,
        mandatory=mandatory,
        message="Omni Asset Validator passed"
        if status_out == "pass"
        else f"Omni Asset Validator found {len(issues)} issue(s)",
        issues=issues,
        metrics=summary,
        metadata={
            "validator": "omniverse-asset-validator",
            "categories_checked": list(raw.get("categories_checked") or []),
        },
    )


def _simready_foundation_level(
    raw: dict[str, Any] | None,
    *,
    mandatory: bool = False,
) -> SimReadyLevel:
    if not raw:
        return SimReadyLevel(
            name="simready_foundation",
            status="incomplete" if mandatory else "unavailable",
            mandatory=mandatory,
            message="SimReady Foundation validator was not run.",
            metadata={"validator": "simready-foundation"},
        )
    status = str(raw.get("status") or "unavailable")
    if status == "pass":
        return SimReadyLevel(
            name="simready_foundation",
            status="pass",
            mandatory=mandatory,
            message=str(raw.get("message") or "SimReady Foundation validator passed."),
            artifacts=dict(raw.get("artifacts") or {}),
            metrics=dict(raw.get("metrics") or {}),
            metadata={"validator": "simready-foundation"},
        )
    if status == "fail":
        return SimReadyLevel(
            name="simready_foundation",
            status="fail",
            mandatory=mandatory,
            message=str(raw.get("message") or "SimReady Foundation validator failed."),
            issues=list(raw.get("issues") or []),
            artifacts=dict(raw.get("artifacts") or {}),
            metrics=dict(raw.get("metrics") or {}),
            metadata={"validator": "simready-foundation"},
        )
    return SimReadyLevel(
        name="simready_foundation",
        status="incomplete" if mandatory else "unavailable",
        mandatory=mandatory,
        message=str(raw.get("message") or "SimReady Foundation validator is unavailable."),
        issues=list(raw.get("issues") or []),
        metadata={"validator": "simready-foundation"},
    )


def _runtime_smoke_level(
    raw: dict[str, Any] | None,
    *,
    mandatory: bool = False,
    dynamic_required: bool = False,
) -> SimReadyLevel:
    if not raw:
        return SimReadyLevel(
            name="runtime_smoke",
            status="incomplete" if mandatory else "unavailable",
            mandatory=mandatory,
            message="Runtime smoke was not run.",
        )
    status = str(raw.get("status") or "unavailable")
    if status == "error":
        status = "fail"
    metadata = dict(raw.get("metadata") or {})
    metrics = dict(raw.get("metrics") or {})
    mode = metadata.get("mode") or metrics.get("mode")
    if mandatory and status == "unavailable":
        status = "incomplete"
    if dynamic_required and status == "pass" and mode == "local_structural":
        status = "incomplete"
    return SimReadyLevel(
        name="runtime_smoke",
        status=status,
        mandatory=mandatory,
        message="Dynamic Isaac/PhysX runtime smoke is required but only local structural load ran."
        if status == "incomplete" and dynamic_required and mode == "local_structural"
        else str(raw.get("message") or ""),
        issues=list(raw.get("issues") or []),
        artifacts=dict(raw.get("artifacts") or {}),
        metrics=metrics,
        metadata=metadata,
    )


def _task_semantics_level(
    report: SimReadyReport,
    task_type: str | None,
    profile: SimReadyProfile,
) -> SimReadyLevel:
    task_l = (task_type or "").lower()
    mandatory = profile.requires_task_semantics or _is_contact_rich(task_l)
    if not mandatory:
        return SimReadyLevel(
            name="task_semantics",
            status="pass",
            mandatory=False,
            message="No contact-rich task semantics were requested.",
            metadata={"task_type": task_type, "asset_type": profile.asset_type},
        )

    issues = [
        _legacy_issue_dict(issue)
        for issue in report.issues
        if issue.name in _TASK_SEMANTIC_ISSUE_NAMES
    ]
    status = _level_status_from_issues(issues) if issues else "incomplete"
    return SimReadyLevel(
        name="task_semantics",
        status=status,
        mandatory=True,
        message="Contact-rich task semantics passed"
        if status == "pass"
        else "Contact-rich task semantics were not available"
        if status == "incomplete"
        else "Contact-rich task semantics need repair",
        issues=issues,
        metrics={
            "contact_interfaces": len(report.contact_interfaces),
            "insertion_axes": len(report.insertion_axes),
            "mating_surfaces": len(report.mating_surfaces),
            "clearance_checks": len(report.clearance_checks),
            "grasp_affordances": len(report.grasp_affordances),
        },
        metadata={"task_type": task_type, "asset_type": profile.asset_type},
    )


def _certificate_status(levels: list[SimReadyLevel]) -> str:
    if any(level.mandatory and level.status in {"fail", "error"} for level in levels):
        return "fail"
    if any(level.mandatory and level.status in {"incomplete", "unavailable"} for level in levels):
        return "incomplete"
    if any(level.mandatory and level.status == "warn" for level in levels):
        return "warn"
    if any(level.status in {"fail", "error", "warn"} for level in levels):
        return "warn"
    return "pass"


def _fix_strategies_from_levels(levels: list[SimReadyLevel]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    strategies: list[dict[str, Any]] = []
    for level in levels:
        for issue in level.issues:
            if issue.get("passed", True):
                continue
            strategy = str(issue.get("fix_strategy") or _fix_strategy_for_issue(str(issue["name"])))
            key = (str(issue.get("name")), strategy)
            if key in seen:
                continue
            seen.add(key)
            strategies.append(
                {
                    "issue": issue.get("name"),
                    "level": level.name,
                    "strategy": strategy,
                    "suggestion": issue.get("suggestion"),
                }
            )
    return strategies


def _alias_level(
    level: SimReadyLevel, name: str, *, mandatory: bool | None = None
) -> SimReadyLevel:
    return SimReadyLevel(
        name=name,
        status=level.status,
        mandatory=level.mandatory if mandatory is None else mandatory,
        message=level.message,
        issues=list(level.issues),
        artifacts=dict(level.artifacts),
        metrics=dict(level.metrics),
        metadata={**dict(level.metadata), "alias_of": level.name},
    )


def build_sim_ready_certificate(
    report: SimReadyReport,
    *,
    result: Any | None = None,
    usd_path: str | Path | None = None,
    usd_validation_result: dict[str, Any] | None = None,
    simready_foundation_result: dict[str, Any] | None = None,
    runtime_smoke_result: dict[str, Any] | None = None,
    physics_authoring_result: dict[str, Any] | None = None,
    mesh_topology_result: dict[str, Any] | None = None,
    physics_material_result: dict[str, Any] | None = None,
    repair_result: dict[str, Any] | None = None,
    task_type: str | None = None,
    asset_type: str | None = None,
    profile_id: str = SIM_READY_PROFILE_ID,
    strict_external_validators: bool = False,
    runtime_smoke_required: bool | None = None,
) -> SimReadyCertificate:
    """Build the versioned Isaac/OpenUSD SimReady certificate."""

    profile = resolve_sim_ready_profile(
        result,
        task_type=task_type,
        asset_type=asset_type,
        profile_id=profile_id,
    )
    required = set(profile.required_levels)
    report.profile = profile.to_dict()
    validation_mode = (
        "strict_external" if strict_external_validators else "local_plus_optional_external"
    )
    mesh_topology = (
        mesh_topology_result or report.mesh_topology or audit_usd_mesh_topology(usd_path)
    )
    report.mesh_topology = dict(mesh_topology or {})
    physics_authoring = physics_authoring_result or audit_usd_physics_authoring(usd_path)
    physics_materials = physics_material_result or audit_usd_physics_materials(usd_path)
    runtime_smoke = runtime_smoke_result
    if runtime_smoke is None:
        runtime_smoke = local_runtime_smoke(usd_path, physics_authoring)
    runtime_required = (
        profile.requires_runtime_smoke
        if runtime_smoke_required is None
        else bool(runtime_smoke_required)
    )

    cad_preflight = _static_cad_level(
        report,
        name="cad_preflight",
        mandatory="cad_preflight" in required,
    )
    usd_core = _usd_stage_level(
        report,
        usd_path,
        name="usd_core",
        mandatory="usd_core" in required,
    )
    physics_authoring_level = _physics_authoring_level(
        physics_authoring,
        mandatory="physics_authoring" in required,
        name="physics_authoring",
    )
    levels = [
        cad_preflight,
        _mesh_topology_level(mesh_topology, mandatory="mesh_topology" in required),
        usd_core,
        physics_authoring_level,
        _physics_materials_level(physics_materials, mandatory="physics_materials" in required),
        _articulation_level(report, physics_authoring, mandatory="articulation" in required),
        _task_semantics_level(report, task_type, profile),
        _runtime_smoke_level(
            runtime_smoke,
            mandatory=runtime_required,
            dynamic_required=runtime_required,
        ),
        _external_usd_validator_level(
            usd_validation_result,
            mandatory=strict_external_validators,
        ),
        _simready_foundation_level(
            simready_foundation_result,
            mandatory=strict_external_validators,
        ),
        _alias_level(cad_preflight, "static_cad"),
        _alias_level(usd_core, "usd_valid"),
        _alias_level(physics_authoring_level, "physics_authored"),
    ]
    status = _certificate_status(levels)
    artifacts = {"usd_path": str(usd_path)} if usd_path else {}
    if repair_result:
        artifacts["repair"] = dict(repair_result)
        artifacts.update(dict(repair_result.get("artifacts") or {}))
    return SimReadyCertificate(
        profile_id=profile.profile_id,
        status=status,
        levels=levels,
        artifacts=artifacts,
        fix_strategies=_fix_strategies_from_levels(levels),
        assumptions=[
            "Primary target is Isaac/OpenUSD; URDF/MJCF/SDF exports are secondary side channels.",
            "Runtime smoke defaults to a local structural USD load unless an Isaac command is configured; profiles can require dynamic smoke explicitly.",
            "External validators are optional unless strict_external_validators is enabled.",
        ],
        asset_type=profile.asset_type,
        requested_profile_id=profile_id,
        resolved_profile_id=profile.profile_id,
        validation_mode=validation_mode,
    )
