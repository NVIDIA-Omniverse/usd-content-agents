# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contract tests for Phase 2 hard-local workers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from geometry_repair.hard_mesh_policy import (
    HARD_MESH_CAPABILITIES_SCHEMA,
    HARD_MESH_PROTOCOL,
    HardMeshResult,
    discover_hard_mesh_executable,
    select_hard_mesh_routes,
)
from geometry_repair.models import RepairOperation
from geometry_repair.workers.mcut_shadow import McutShadowWorker
from geometry_repair.workers.pmp_patch import _PMP_SPEC, PmpPatchWorker
from geometry_repair.workers.wildmeshing_shadow import WildmeshingShadowWorker


def _operation(
    worker: str,
    parameters: dict[str, object],
    *,
    issue_ids: list[str] | None = None,
) -> RepairOperation:
    return RepairOperation(
        operation_id=f"test-{worker}",
        worker=worker,
        implementation="phase2-test",
        parameters=parameters,
        issue_ids=issue_ids or ["mesh:test-defect"],
        drift_band="conservative",
        source_checkpoint="source.usda",
    )


def _pmp_unapproved_remesh_parameters() -> dict[str, object]:
    return {
        "operation": "pmp_remesh_generated_patch",
        "target_mesh_path": "/Asset/PartA",
        "region_intent": "generated_patch_refinement",
        "intent_evidence_id": "mesh:test-defect",
        "generated_face_ids": [0],
        "transition_face_ids": [],
        "frozen_vertex_ids": [0],
        "protected_edge_vertex_pairs": [[0, 1]],
        "transition_rings": 1,
        "target_edge_length_ratio": 0.01,
        "max_envelope_ratio": 0.001,
        "max_iterations": 5,
        "deterministic_seed": 7,
        "timeout_s": 10.0,
    }


def _wildmeshing_parameters() -> dict[str, object]:
    return {
        "operation": "wildmeshing_remesh_named_patch",
        "target_mesh_path": "/Asset/PartA",
        "region_intent": "named_local_remesh",
        "intent_evidence_id": "mesh:test-defect",
        "region_face_ids": [0],
        "transition_face_ids": [],
        "frozen_vertex_ids": [0],
        "protected_edge_vertex_pairs": [[0, 1]],
        "required_invariants": [
            "frozen_vertices",
            "manifoldness",
            "material_boundaries",
            "orientation",
            "part_boundaries",
            "protected_edges",
            "surface_envelope",
            "uv_seams",
        ],
        "correspondence_mode": "source_face_barycentric",
        "attribute_policy": "preserve_or_refuse",
        "rollback_on_invariant_failure": True,
        "target_edge_length_ratio": 0.01,
        "max_envelope_ratio": 0.001,
        "max_operations": 10,
        "deterministic_seed": 11,
        "timeout_s": 10.0,
    }


def _mcut_parameters() -> dict[str, object]:
    return {
        "operation": "mcut_partition_source_parts",
        "part_paths": ["/Asset/PartA", "/Asset/PartB"],
        "region_intent": "diagnosed_two_part_intersection",
        "intent_evidence_id": "mesh:test-defect",
        "protected_feature_ids": ["clearance:test"],
        "fragment_selection": "none",
        "selected_fragment_ids": [],
        "deleted_fragment_ids": [],
        "combine_fragments": False,
        "max_fragments": 32,
        "max_intersection_curves": 32,
        "deterministic_seed": 13,
        "timeout_s": 10.0,
    }


def _write_usd(path: Path, *, attributed: bool = False, two_parts: bool = False) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    part_names = ["PartA", "PartB"] if two_parts else ["PartA"]
    for index, name in enumerate(part_names):
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{name}")
        offset = float(index) * 0.25
        mesh.CreatePointsAttr(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(offset, 0.0, 0.0),
                    Gf.Vec3f(offset + 1.0, 0.0, 0.0),
                    Gf.Vec3f(offset, 1.0, 0.0),
                ]
            )
        )
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        if attributed and index == 0:
            primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.faceVarying,
            )
            primvar.Set(Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 1.0)]))
    stage.GetRootLayer().Save()
    return path


def test_pmp_requires_explicit_intent_and_protected_boundary(tmp_path: Path) -> None:
    parameters = {
        "operation": "pmp_fill_classified_hole",
        "target_mesh_path": "/Asset/PartA",
        "region_intent": "classified_accidental_hole",
        "intent_evidence_id": "mesh:test-defect",
        "boundary_loop_vertex_ids": [0, 1, 2],
        "frozen_boundary_vertex_ids": [0, 1, 3],
        "protected_edge_vertex_pairs": [[0, 1], [1, 3], [3, 0]],
    }

    result = PmpPatchWorker().execute_typed(
        source=tmp_path / "unused.usda",
        output=tmp_path / "out.usda",
        operation=_operation("pmp_patch", parameters),
    )

    assert result.status == "refused"
    assert "boundary-loop vertex" in result.failures[0]


@pytest.mark.parametrize(
    ("worker", "parameters", "expected"),
    [
        (
            PmpPatchWorker(),
            _pmp_unapproved_remesh_parameters(),
            "unsupported PMP operation",
        ),
        (
            WildmeshingShadowWorker(),
            {**_wildmeshing_parameters(), "required_invariants": ["manifoldness"]},
            "at least 8 items",
        ),
        (
            McutShadowWorker(),
            {**_mcut_parameters(), "max_fragments": 10_000},
            "less than or equal to 4096",
        ),
    ],
)
def test_hard_local_parameter_bounds_fail_before_execution(
    tmp_path: Path,
    worker,
    parameters: dict[str, object],
    expected: str,
) -> None:
    result = worker.execute_typed(
        source=tmp_path / "unused.usda",
        output=tmp_path / "out.usda",
        operation=_operation(worker.name, parameters),
    )

    assert result.status == "refused"
    assert expected in result.failures[0]


def test_unapproved_pmp_remesh_is_refused_before_execution(tmp_path: Path) -> None:
    result = PmpPatchWorker().execute_typed(
        source=tmp_path / "source.usda",
        output=tmp_path / "out.usda",
        operation=_operation("pmp_patch", _pmp_unapproved_remesh_parameters()),
    )

    assert result.status == "refused"
    assert "unsupported PMP operation" in result.failures[0]


@pytest.mark.parametrize(
    ("worker", "parameters"),
    [
        (WildmeshingShadowWorker(), _wildmeshing_parameters()),
    ],
)
def test_attributed_patch_is_refused_without_exact_transfer(
    tmp_path: Path,
    worker,
    parameters: dict[str, object],
) -> None:
    source = _write_usd(tmp_path / f"{worker.name}.usda", attributed=True)
    result = worker.execute_typed(
        source=source,
        output=tmp_path / f"{worker.name}-out.usda",
        operation=_operation(worker.name, parameters),
    )

    assert result.status == "refused"
    assert "face-varying data" in result.failures[0]


def test_mcut_refuses_attributed_parts_and_any_fragment_selection(tmp_path: Path) -> None:
    source = _write_usd(tmp_path / "parts.usda", attributed=True, two_parts=True)
    worker = McutShadowWorker()
    attributed = worker.execute_typed(
        source=source,
        output=tmp_path / "out.usda",
        operation=_operation(worker.name, _mcut_parameters()),
    )
    selecting = worker.execute_typed(
        source=source,
        output=tmp_path / "selected.usda",
        operation=_operation(
            worker.name,
            {**_mcut_parameters(), "selected_fragment_ids": ["fragment-0"]},
        ),
    )

    assert attributed.status == "refused"
    assert "exact transfer" in attributed.failures[0]
    assert selecting.status == "refused"
    assert "at most 0 items" in selecting.failures[0]


def test_mcut_requires_exactly_two_distinct_named_parts(tmp_path: Path) -> None:
    worker = McutShadowWorker()
    three_parts = worker.execute_typed(
        source=tmp_path / "unused.usda",
        output=tmp_path / "out.usda",
        operation=_operation(
            worker.name,
            {
                **_mcut_parameters(),
                "part_paths": ["/Asset/A", "/Asset/B", "/Asset/C"],
            },
        ),
    )
    duplicate_parts = worker.execute_typed(
        source=tmp_path / "unused.usda",
        output=tmp_path / "out.usda",
        operation=_operation(
            worker.name,
            {**_mcut_parameters(), "part_paths": ["/Asset/A", "/Asset/A"]},
        ),
    )

    assert three_parts.status == "refused"
    assert duplicate_parts.status == "refused"
    assert "distinct named source parts" in duplicate_parts.failures[0]


def test_missing_native_binaries_are_explicitly_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "GEOMETRY_REPAIR_PMP_EXECUTABLE",
        "GEOMETRY_REPAIR_WILDMESHING_EXECUTABLE",
        "GEOMETRY_REPAIR_MCUT_EXECUTABLE",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr("geometry_repair.hard_mesh_policy.shutil.which", lambda _name: None)

    for worker in (PmpPatchWorker(), WildmeshingShadowWorker(), McutShadowWorker()):
        available, reason = worker.available()
        assert available is False
        assert "unavailable" in (reason or "")


def test_capability_discovery_rejects_exact_build_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "pmp-adapter"
    executable.write_text("test adapter placeholder\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("GEOMETRY_REPAIR_PMP_EXECUTABLE", str(executable))
    capability_document = json.dumps(
        {
            "schema_version": HARD_MESH_CAPABILITIES_SCHEMA,
            "protocol_version": HARD_MESH_PROTOCOL,
            "worker": _PMP_SPEC.worker,
            "implementation_version": "drifted-build",
            "build_id": "pmp-library:unreviewed",
            "operations": sorted(_PMP_SPEC.operations),
            "capabilities": sorted(_PMP_SPEC.required_capabilities),
            "deterministic": True,
        }
    )
    monkeypatch.setattr(
        "geometry_repair.hard_mesh_policy.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=capability_document,
            stderr="",
        ),
    )

    executable_path, capabilities, reason = discover_hard_mesh_executable(_PMP_SPEC)

    assert executable_path is None
    assert capabilities is None
    assert "build drifted" in (reason or "")


def test_hard_mesh_result_preserves_refusal_status_in_worker_metadata() -> None:
    projected = HardMeshResult(
        status="refused",
        operation="test_operation",
        failures=["explicit refusal"],
    ).to_worker_result()

    assert projected.status == "unavailable"
    assert projected.metadata["hard_mesh_status"] == "refused"
    assert projected.failures == ["explicit refusal"]


def test_policy_keeps_existing_authorities_and_shadow_ordering() -> None:
    hole = select_hard_mesh_routes(
        defect="classified_hole",
        explicit_intent=True,
        protected_boundaries=True,
        attributes_supported=True,
        reconstructive_authorized=True,
        boundary_vertex_count=4,
    )
    intersection = select_hard_mesh_routes(
        defect="two_part_intersection",
        explicit_intent=True,
        protected_boundaries=True,
        attributes_supported=True,
        reconstructive_authorized=True,
    )
    unclassified = select_hard_mesh_routes(
        defect="two_part_intersection",
        explicit_intent=False,
        protected_boundaries=True,
        attributes_supported=True,
    )

    assert [route.worker for route in hole.routes] == [
        "trimesh_bounded_hole_fill",
        "pmp_patch",
        "wildmeshing_shadow",
        "sdf_rebuild",
    ]
    assert [route.worker for route in intersection.routes] == [
        "geogram_local_repair",
        "mcut_shadow",
        "sdf_rebuild",
    ]
    assert [route.worker for route in unclassified.routes] == ["geogram_local_repair"]
    assert "explicit classified defect intent" in unclassified.refusal_reasons[0]
    assert all(
        route.authority == "shadow" for route in intersection.routes if "mcut" in route.worker
    )


def test_shadow_workers_never_project_typed_success_as_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    typed = HardMeshResult(
        status="success",
        output_path=str(tmp_path / "candidate.usda"),
        output_sha256="0" * 64,
        changed=True,
        operation="shadow",
    )
    for worker in (WildmeshingShadowWorker(), McutShadowWorker()):
        monkeypatch.setattr(worker, "execute_typed", lambda **_kwargs: typed)
        projected = worker.execute(
            source=tmp_path / "source.usda",
            output=tmp_path / "out.usda",
            operation=_operation(worker.name, {}),
        )
        assert projected.status == "unavailable"
        assert projected.metadata["shadow_only"] is True
        assert projected.output_path is None
