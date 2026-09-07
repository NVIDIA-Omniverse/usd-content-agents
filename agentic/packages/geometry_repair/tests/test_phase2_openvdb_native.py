# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity, refusal, and integration tests for the in-process OpenVDB 13 worker."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import trimesh

from geometry_repair.artifacts import file_sha256
from geometry_repair.diagnosis import diagnose_asset
from geometry_repair.mesh_io import load_meshes
from geometry_repair.models import RepairOperation
from geometry_repair.workers import sdf_rebuild as openvdb_worker
from geometry_repair.workers import sdf_reconstruction
from geometry_repair.workers.sdf_rebuild import (
    OPENVDB_BACKEND_SOURCE_ID,
    OPENVDB_BUILD_ID,
    OPENVDB_DISTRIBUTION_VERSION,
    OPENVDB_IMPLEMENTATION_VERSION,
    SdfRebuildWorker,
    _block_reconstructed_normals,
    _bounded_voxel_size,
    _normal_regeneration_plan,
    _reauthor_extents,
    _topology_attribute_refusals,
    inspect_sdf_reconstruction_eligibility,
)

_RUNTIME_IDENTITY: dict[str, Any] = {
    "library_version": [13, 0, 0],
    "distribution_version": OPENVDB_DISTRIBUTION_VERSION,
    "file_format_version": 224,
    "module_path": "/runtime/openvdb.so",
    "module_sha256": "1" * 64,
    "source_lock_path": "/runtime/_source_lock.json",
    "source_lock_sha256": "2" * 64,
    "source_distribution_version": OPENVDB_DISTRIBUTION_VERSION,
    "policy_schema": "world-understanding.openvdb-runtime-policy.v1",
    "source_commit": OPENVDB_BACKEND_SOURCE_ID,
    "capabilities": [
        "active_value_mask",
        "extract_enclosed_region",
        "mesh_to_level_set",
        "mesh_to_unsigned_distance_field",
        "scalar_mean_filter",
        "topology_to_level_set",
        "volume_to_mesh",
    ],
}

_BACKEND_IDENTITY: dict[str, Any] = {
    "backend_id": "openvdb",
    "implementation_version": f"openvdb-runtime-{OPENVDB_DISTRIBUTION_VERSION}",
    "operations": sorted(
        {operation.value for operation in openvdb_worker._OPENVDB13_REQUIRED_OPERATIONS}
        | {"write_fields"}
    ),
    "execution_mode": "in_process",
    "read_formats": [],
    "write_formats": ["vdb"],
    "supported_formats": ["vdb"],
    "provenance": _RUNTIME_IDENTITY,
}


def _operation(
    source: Path,
    *,
    operation_id: str = "openvdb13-test",
    drift_band: str = "reconstructive",
    parameters: dict[str, object] | None = None,
) -> RepairOperation:
    values: dict[str, object] = {
        "max_grid_dimension": 128,
        "max_output_faces": 250_000,
        "half_width": 3.0,
        "offset_voxels": 0.75,
        "adaptivity": 0.0,
        "closing_steps": 2,
        "smoothing_steps": 1,
        "deterministic_seed": 29,
        "timeout_s": 30.0,
    }
    if parameters:
        values.update(parameters)
    return RepairOperation(
        operation_id=operation_id,
        worker="sdf_rebuild",
        implementation="geometry_repair.sdf_reconstruction-test",
        parameters=values,
        issue_ids=["mesh:boundary_edges"],
        drift_band=drift_band,
        source_checkpoint=str(source),
    )


def _define_mesh(
    stage: Any,
    path: str,
    mesh: trimesh.Trimesh,
    *,
    attributed: bool = False,
    normal_primvar: bool = False,
) -> None:
    from pxr import Gf, Sdf, UsdGeom, Vt

    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    usd_mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(*(float(value) for value in point)) for point in vertices])
    )
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(triangles)))
    usd_mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray([int(value) for triangle in triangles for value in triangle])
    )
    usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    usd_mesh.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*(float(value) for value in vertices.min(axis=0))),
                Gf.Vec3f(*(float(value) for value in vertices.max(axis=0))),
            ]
        )
    )
    if attributed:
        primvar = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        primvar.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(float(index % 2), float((index // 2) % 2))
                    for index in range(3 * len(triangles))
                ]
            )
        )
    if normal_primvar:
        normals = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            "normals",
            Sdf.ValueTypeNames.Normal3fArray,
            UsdGeom.Tokens.faceVarying,
        )
        normals.Set(
            Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0) for _index in range(3 * len(triangles))])
        )


def _write_mesh_usd(
    path: Path,
    mesh: trimesh.Trimesh,
    *,
    attributed: bool = False,
    normal_primvar: bool = False,
) -> Path:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    _define_mesh(
        stage,
        "/Asset/Part",
        mesh,
        attributed=attributed,
        normal_primvar=normal_primvar,
    )
    stage.GetRootLayer().Save()
    return path


def _write_two_mesh_usd(path: Path) -> Path:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = trimesh.creation.box(extents=(0.1, 0.08, 0.06))
    _define_mesh(stage, "/Asset/PartA", mesh)
    _define_mesh(stage, "/Asset/PartB", mesh)
    stage.GetRootLayer().Save()
    return path


def _write_instance_usd(path: Path) -> Path:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    prototype = UsdGeom.Xform.Define(stage, "/Prototype")
    _define_mesh(stage, "/Prototype/Part", trimesh.creation.box(extents=(0.1, 0.1, 0.1)))
    instance = UsdGeom.Xform.Define(stage, "/Asset/Instance")
    instance.GetPrim().GetReferences().AddInternalReference(prototype.GetPath())
    instance.GetPrim().SetInstanceable(True)
    stage.GetRootLayer().Save()
    return path


def _fake_result(
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: sdf_reconstruction.SdfReconstructionControls,
) -> sdf_reconstruction.SdfReconstructionResult:
    input_vertices = np.array(vertices, dtype=np.float32, order="C", copy=True)
    input_vertices[input_vertices == 0.0] = np.float32(0.0)
    input_triangles = np.array(triangles, dtype=np.int32, order="C", copy=True)
    surface = sdf_reconstruction._canonical_surface(
        SimpleNamespace(
            vertices=input_vertices,
            triangles=input_triangles,
            quads=np.empty((0, 4), dtype=np.int32),
        ),
        controls,
    )
    boundary_edges, non_manifold_edges, duplicate_faces = (
        sdf_reconstruction._topology_defect_counts(input_triangles)
    )
    topology_defects = boundary_edges + non_manifold_edges + duplicate_faces
    topology_closing = controls.mode == "signed" and topology_defects > 0
    evidence = sdf_reconstruction._execution_evidence(
        controls=controls,
        qualification_id="geometry-repair.openvdb13.v1",
        required_operations=sdf_reconstruction._RECONSTRUCTION_OPERATIONS,
        limits=sdf_reconstruction._execution_limits(controls),
        selected_backend_identity=copy.deepcopy(_BACKEND_IDENTITY),
        selection_rejections=(),
        algorithm=sdf_reconstruction._algorithm_evidence(
            controls,
            topology_closing=topology_closing,
        ),
        estimated_dimensions=sdf_reconstruction._estimated_grid_dimensions(
            input_vertices,
            controls,
            topology_closing=topology_closing,
        ),
        source_vertices=input_vertices,
        source_triangles=input_triangles,
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        duplicate_faces=duplicate_faces,
        backend_call_status="succeeded",
        candidate_validation_status="accepted",
        surface=surface,
    )
    return sdf_reconstruction.SdfReconstructionResult(
        vertices=surface.vertices,
        triangles=surface.triangles,
        evidence=evidence,
    )


def _tampered_result(
    result: sdf_reconstruction.SdfReconstructionResult,
    path: tuple[str, ...],
    replacement: object,
) -> sdf_reconstruction.SdfReconstructionResult:
    payload = result.evidence.model_dump(mode="json")
    target: dict[str, Any] = payload
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = replacement
    return sdf_reconstruction.SdfReconstructionResult(
        vertices=result.vertices,
        triangles=result.triangles,
        evidence=sdf_reconstruction.SdfExecutionEvidence.model_construct(**payload),
    )


def _install_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: Exception | None = None,
    tamper: bool = False,
    session_calls: list[object] | None = None,
    session_requests: list[dict[str, object]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, sdf_reconstruction.SdfReconstructionControls]]:
    calls: list[tuple[np.ndarray, np.ndarray, sdf_reconstruction.SdfReconstructionControls]] = []
    shared_session = SimpleNamespace(
        backend_info=SimpleNamespace(as_dict=lambda: copy.deepcopy(_BACKEND_IDENTITY))
    )

    def create_session(**kwargs: object) -> object:
        if session_requests is not None:
            session_requests.append(kwargs)
        return shared_session

    monkeypatch.setattr(openvdb_worker.sdf_tools, "create_session", create_session)

    def reconstruct(
        vertices: np.ndarray,
        triangles: np.ndarray,
        controls: sdf_reconstruction.SdfReconstructionControls,
        *,
        session: object | None = None,
    ) -> sdf_reconstruction.SdfReconstructionResult:
        calls.append((vertices.copy(), triangles.copy(), controls))
        if session_calls is not None:
            session_calls.append(session)
        if failure is not None:
            raise failure
        result = _fake_result(vertices, triangles, controls)
        if tamper:
            result = _tampered_result(
                result,
                ("output_digests", "vertices", "sha256"),
                "f" * 64,
            )
        return result

    monkeypatch.setattr(openvdb_worker, "reconstruct_sdf_mesh", reconstruct)
    return calls


def test_available_requires_source_locked_v13_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def create_session(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            backend_info=SimpleNamespace(as_dict=lambda: copy.deepcopy(_BACKEND_IDENTITY))
        )

    monkeypatch.setattr(openvdb_worker.sdf_tools, "create_session", create_session)

    available, reason = SdfRebuildWorker().available()

    assert available is True
    assert reason is None
    assert captured["backend"] == "openvdb"
    assert set(captured["require"]) == set(openvdb_worker._OPENVDB13_REQUIRED_OPERATIONS)


def test_available_reports_source_lock_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        openvdb_worker.sdf_tools,
        "create_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            openvdb_worker.sdf_tools.BackendUnavailableError("source lock mismatch")
        ),
    )

    available, reason = SdfRebuildWorker().available()

    assert available is False
    assert "source lock mismatch" in (reason or "")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("module_path", None),
        ("module_sha256", "not-a-digest"),
        ("source_lock_path", ""),
        ("source_lock_sha256", "A" * 64),
    ],
)
def test_available_rejects_incomplete_loaded_runtime_evidence(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: object,
) -> None:
    identity = copy.deepcopy(_BACKEND_IDENTITY)
    identity["provenance"][key] = value
    monkeypatch.setattr(
        openvdb_worker.sdf_tools,
        "create_session",
        lambda **_kwargs: SimpleNamespace(backend_info=SimpleNamespace(as_dict=lambda: identity)),
    )

    available, reason = SdfRebuildWorker().available()

    assert available is False
    assert "qualification" in (reason or "")


@pytest.mark.parametrize("source_kind", ["attributed", "instance"])
def test_attributed_and_instance_inputs_are_refused_before_backend_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    calls = _install_fake_backend(monkeypatch)
    if source_kind == "attributed":
        source = _write_mesh_usd(
            tmp_path / "attributed.usda",
            trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
            attributed=True,
        )
    else:
        source = _write_instance_usd(tmp_path / "instance.usda")

    result = SdfRebuildWorker().execute(
        source=source,
        output=tmp_path / "candidate.usda",
        operation=_operation(source),
    )

    assert result.status == "unavailable"
    assert result.metadata["sdf_status"] == "refused"
    assert "attributed or instanced topology" in result.failures[0]
    assert calls == []
    assert not (tmp_path / "candidate.usda").exists()


def test_eligibility_failure_preserves_v1_payload_shape(tmp_path: Path) -> None:
    result = inspect_sdf_reconstruction_eligibility(tmp_path / "missing.usda")

    assert result["schema_version"] == "geometry-repair.sdf-eligibility.v1"
    assert result["eligible"] is False
    assert result["normal_regeneration_plan"] == {}


def test_reconstructive_route_allows_only_regenerable_normal_primvars(tmp_path: Path) -> None:
    from pxr import Usd

    source = _write_mesh_usd(
        tmp_path / "normal_primvar.usda",
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
        normal_primvar=True,
    )
    meshes, _metadata = load_meshes(source)
    render_meshes = [mesh for mesh in meshes if mesh.role == "render"]

    assert _topology_attribute_refusals(source, render_meshes) == {}
    plan = _normal_regeneration_plan(source, render_meshes)
    assert plan == {"/Asset/Part": ["primvars:normals"]}

    candidate = tmp_path / "candidate.usda"
    shutil.copy2(source, candidate)
    _block_reconstructed_normals(candidate, plan)
    stage = Usd.Stage.Open(str(candidate))
    assert stage is not None
    prim = stage.GetPrimAtPath("/Asset/Part")
    assert prim.GetAttribute("primvars:normals").Get() is None
    assert (
        prim.GetCustomDataByKey("geometryRepairNormalPolicy")
        == "derive_from_reconstructed_topology"
    )


@pytest.mark.parametrize("authoring_step", ["normals", "extent"])
def test_candidate_authoring_refuses_instance_proxy_targets(
    tmp_path: Path,
    authoring_step: str,
) -> None:
    candidate = _write_instance_usd(tmp_path / "candidate.usda")
    prim_path = "/Asset/Instance/Part"

    with pytest.raises(RuntimeError, match="read-only instance proxy"):
        if authoring_step == "normals":
            _block_reconstructed_normals(candidate, {prim_path: ["normals"]})
        else:
            mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
            _reauthor_extents(
                candidate,
                {
                    prim_path: (
                        np.asarray(mesh.vertices, dtype=np.float32),
                        np.asarray(mesh.faces, dtype=np.int32),
                    )
                },
            )


@pytest.mark.parametrize(
    ("drift_band", "parameters", "expected"),
    [
        ("conservative", {}, "reconstructive drift band"),
        ("reconstructive", {"max_grid_dimension": 31}, "max_grid_dimension"),
        ("reconstructive", {"smoothing_steps": 2}, "smoothing_steps"),
        ("reconstructive", {"smoothing_steps": True}, "smoothing_steps"),
        ("reconstructive", {"closing_steps": 3}, "closing_steps"),
        ("reconstructive", {"closing_steps": True}, "closing_steps"),
        ("reconstructive", {"half_width": "3.0"}, "half_width"),
        ("reconstructive", {"backend_id": "Not Valid"}, "backend_id"),
        ("reconstructive", {"backend_id": "openvdb--driver"}, "backend_id"),
    ],
)
def test_reconstructive_and_resource_policy_is_fail_closed(
    tmp_path: Path,
    drift_band: str,
    parameters: dict[str, object],
    expected: str,
) -> None:
    source = _write_mesh_usd(
        tmp_path / "source.usda",
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
    )

    result = SdfRebuildWorker().execute(
        source=source,
        output=tmp_path / "candidate.usda",
        operation=_operation(source, drift_band=drift_band, parameters=parameters),
    )

    assert result.status == "unavailable"
    assert result.metadata["sdf_status"] == "refused"
    assert expected in result.failures[0]
    assert not (tmp_path / "candidate.usda").exists()


def test_unqualified_backend_is_reported_as_unavailable(
    tmp_path: Path,
) -> None:
    source = _write_mesh_usd(
        tmp_path / "source.usda",
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
    )

    result = SdfRebuildWorker().execute(
        source=source,
        output=tmp_path / "candidate.usda",
        operation=_operation(source, parameters={"backend_id": "unreviewed"}),
    )

    assert result.status == "unavailable"
    assert result.metadata["sdf_status"] == "unavailable"
    assert result.metadata["sdf_backend_id"] == "unreviewed"
    assert "not qualified" in result.failures[0]
    evidence = result.metadata["sdf_execution_evidence"]
    assert evidence["requested_backend_id"] == "unreviewed"
    assert evidence["backend_qualification_id"] is None
    assert evidence["selected_backend_identity"] is None
    assert evidence["backend_call_status"] == "unavailable"


def test_worker_passes_local_arrays_and_controls_without_obj_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_mesh_usd(
        tmp_path / "source.usda",
        trimesh.creation.box(extents=(0.1, 0.08, 0.06)),
    )
    source_sha256 = file_sha256(source)
    source_meshes, _metadata = load_meshes(source)
    calls = _install_fake_backend(monkeypatch)
    output = tmp_path / "candidate.usda"

    result = SdfRebuildWorker().execute(
        source=source,
        output=output,
        operation=_operation(source),
    )

    assert result.status == "completed", result.failures
    assert len(calls) == 1
    vertices, triangles, controls = calls[0]
    assert np.array_equal(vertices, source_meshes[0].local_vertices)
    assert np.array_equal(triangles, source_meshes[0].triangles)
    assert controls.mode == "signed"
    assert controls.backend_id == "openvdb"
    assert controls.max_grid_dimension == 128
    assert controls.max_output_faces == 250_000
    assert controls.deterministic_seed == 29
    assert controls.voxel_size > 0.0
    assert file_sha256(source) == source_sha256
    assert output.is_file()
    assert list(tmp_path.glob("*.obj")) == []
    assert list(tmp_path.glob("*.openvdb_bridge")) == []
    assert result.metadata["backend"] == "sdf_tools.in_process"
    assert result.metadata["implementation_build_id"] == OPENVDB_BUILD_ID
    assert result.metadata["implementation_version"] == OPENVDB_IMPLEMENTATION_VERSION
    assert (
        result.metadata["sdf_backend"]["provenance"]["distribution_version"]
        == OPENVDB_DISTRIBUTION_VERSION
    )
    assert result.metadata["backend_qualification_id"] == "geometry-repair.openvdb13.v1"
    assert "openvdb_status" not in result.metadata
    assert "openvdb_version" not in result.metadata
    invocation = result.metadata["invocations"][0]
    assert invocation["backend"] == "sdf_tools.in_process"
    assert invocation["sdf_status"] == "success"
    assert invocation["backend_report"]["smoothing_steps"] == 1
    assert invocation["backend_evidence"]["selected_backend_identity"]["backend_id"] == "openvdb"
    assert invocation["backend_evidence"]["selected_backend_identity"]["provenance"][
        "library_version"
    ] == [13, 0, 0]
    execution_evidence = invocation["sdf_execution_evidence"]
    assert execution_evidence["requested_backend_id"] == "openvdb"
    assert execution_evidence["backend_qualification_id"] == "geometry-repair.openvdb13.v1"
    assert execution_evidence["backend_call_status"] == "succeeded"
    assert execution_evidence["candidate_validation_status"] == "accepted"
    assert execution_evidence["selection_rejections"] == []
    assert execution_evidence["source_digests"]["vertices"]["sha256"]
    assert execution_evidence["output_digests"]["vertices"]["sha256"]
    assert result.metadata["backend_success_is_acceptance"] is False
    assert "remains conditional" in result.warnings[0]
    diagnosis = diagnose_asset(output, profile="rigid_pick_place")
    assert diagnosis.metrics.mesh.watertight is True


def test_worker_reuses_one_qualified_session_for_all_render_meshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_two_mesh_usd(tmp_path / "source.usda")
    session_calls: list[object] = []
    session_requests: list[dict[str, object]] = []
    calls = _install_fake_backend(
        monkeypatch,
        session_calls=session_calls,
        session_requests=session_requests,
    )

    result = SdfRebuildWorker().execute(
        source=source,
        output=tmp_path / "candidate.usda",
        operation=_operation(source),
    )

    assert result.status == "completed", result.failures
    assert len(calls) == len(session_calls) == 2
    assert len(session_requests) == 1
    assert session_calls[0] is session_calls[1]
    assert session_requests[0]["require"] == openvdb_worker._OPENVDB13_REQUIRED_OPERATIONS


def test_acceptance_gate_failure_does_not_claim_sdf_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_mesh_usd(
        tmp_path / "source.usda",
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
    )
    _install_fake_backend(monkeypatch)
    reconstruct = openvdb_worker.reconstruct_sdf_mesh

    def _oversized_result(*args: Any, **kwargs: Any) -> Any:
        result = reconstruct(*args, **kwargs)
        result.vertices[:] = result.vertices * np.float32(10.0)
        return result

    monkeypatch.setattr(openvdb_worker, "reconstruct_sdf_mesh", _oversized_result)
    monkeypatch.setattr(openvdb_worker, "_validate_sdf_result", lambda *_args, **_kwargs: None)

    result = SdfRebuildWorker().execute(
        source=source,
        output=tmp_path / "candidate.usda",
        operation=_operation(source),
    )

    assert result.status == "failed"
    assert "surface area ratio" in result.failures[0]
    assert result.metadata["invocations"][0]["sdf_status"] == "backend_returned"


def test_worker_is_deterministic_from_backend_array_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_mesh_usd(
        tmp_path / "source.usda",
        trimesh.creation.icosphere(subdivisions=2, radius=0.05),
    )
    calls = _install_fake_backend(monkeypatch)
    worker = SdfRebuildWorker()

    first = worker.execute(
        source=source,
        output=tmp_path / "candidate_a.usda",
        operation=_operation(source, operation_id="openvdb13-a"),
    )
    second = worker.execute(
        source=source,
        output=tmp_path / "candidate_b.usda",
        operation=_operation(source, operation_id="openvdb13-b"),
    )

    assert len(calls) == 2
    assert first.status == second.status == "completed"
    assert first.output_sha256 == second.output_sha256
    first_hashes = first.metadata["invocations"][0]["backend_evidence"]["output_digests"]
    second_hashes = second.metadata["invocations"][0]["backend_evidence"]["output_digests"]
    assert first_hashes == second_hashes


def test_tampered_backend_evidence_fails_without_publishing_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_mesh_usd(
        tmp_path / "source.usda",
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
    )
    _install_fake_backend(monkeypatch, tamper=True)
    output = tmp_path / "candidate.usda"

    result = SdfRebuildWorker().execute(
        source=source,
        output=output,
        operation=_operation(source),
    )

    assert result.status == "failed"
    assert "source or output digest evidence" in result.failures[0]
    assert not output.exists()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("algorithm", "route"), "signed_topology_closing"),
        (("algorithm", "explicit_closing_operator_applied"), True),
        (("algorithm", "unsigned_distance_sampling"), True),
        (("algorithm", "gap_closing_budget_voxels"), 1),
        (("algorithm", "filter_operator"), "smooth_scalar"),
        (("algorithm", "isovalue"), -0.01),
        (("algorithm", "repair_orientation"), False),
        (("limits", "execution_limits", "max_threads"), True),
        (("limits", "execution_limits", "max_voxel_extent"), 63),
        (("limits", "execution_limits", "max_topology_steps"), 3),
        (("limits", "execution_limits", "max_filter_iterations"), 2),
        (("limits", "execution_limits", "max_filter_work"), 1),
        (("resource_usage", "estimated_grid_dimensions"), [1, 1, 1]),
        (("resource_usage", "input_boundary_edges"), 1),
        (("resource_usage", "input_non_manifold_edges"), 1),
        (("resource_usage", "input_duplicate_faces"), 1),
        (("resource_usage", "input_topology_defects"), 1),
        (("resource_usage", "output_array_bytes"), 1),
        (("determinism", "normalized_signed_zero"), False),
        (("determinism", "shortest_quad_diagonal"), False),
    ],
)
def test_validator_rejects_tampered_policy_evidence(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    controls = sdf_reconstruction.SdfReconstructionControls(
        voxel_size=0.01,
        max_grid_dimension=64,
        max_output_faces=10_000,
    )
    result = _tampered_result(_fake_result(vertices, triangles, controls), path, replacement)

    failure = openvdb_worker._validate_sdf_result(
        result,
        controls=controls,
        input_vertices=vertices,
        input_triangles=triangles,
        backend_identity=copy.deepcopy(_BACKEND_IDENTITY),
    )

    assert failure is not None


@pytest.mark.parametrize("key", ["output_surface_area", "output_signed_volume"])
def test_validator_rejects_geometry_evidence_that_does_not_match_arrays(key: str) -> None:
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    controls = sdf_reconstruction.SdfReconstructionControls(
        voxel_size=0.01,
        max_grid_dimension=64,
        max_output_faces=10_000,
    )
    result = _fake_result(vertices, triangles, controls)
    result = _tampered_result(
        result,
        ("geometry", key),
        result.evidence["geometry"][key] * 1.01,
    )

    failure = openvdb_worker._validate_sdf_result(
        result,
        controls=controls,
        input_vertices=vertices,
        input_triangles=triangles,
        backend_identity=copy.deepcopy(_BACKEND_IDENTITY),
    )

    assert failure == "SDF backend geometry evidence does not match its arrays"


def test_validator_derives_topology_closing_route_from_boundary_edges() -> None:
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces[:-1], dtype=np.int64)
    controls = sdf_reconstruction.SdfReconstructionControls(
        voxel_size=0.01,
        max_grid_dimension=64,
        max_output_faces=10_000,
    )
    result = _fake_result(vertices, triangles, controls)

    assert result.evidence["algorithm"]["route"] == "signed_topology_closing"
    assert (
        openvdb_worker._validate_sdf_result(
            result,
            controls=controls,
            input_vertices=vertices,
            input_triangles=triangles,
            backend_identity=copy.deepcopy(_BACKEND_IDENTITY),
        )
        is None
    )

    result = _tampered_result(
        result,
        ("algorithm", "route"),
        "signed_level_set",
    )
    assert (
        openvdb_worker._validate_sdf_result(
            result,
            controls=controls,
            input_vertices=vertices,
            input_triangles=triangles,
            backend_identity=copy.deepcopy(_BACKEND_IDENTITY),
        )
        is not None
    )


def test_bounded_voxel_size_reserves_topology_closing_margin() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 0.05, 0.05))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    closed_triangles = np.asarray(mesh.faces, dtype=np.int64)
    open_triangles = np.asarray(mesh.faces[:-1], dtype=np.int64)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    common = {
        "diagonal": diagonal,
        "max_grid_dimension": 64,
        "mode": "signed",
        "half_width": 3.0,
        "offset_voxels": 0.75,
        "closing_steps": 2,
    }

    closed_voxel_size, closed_dimensions = _bounded_voxel_size(
        vertices,
        closed_triangles,
        **common,
    )
    open_voxel_size, open_dimensions = _bounded_voxel_size(
        vertices,
        open_triangles,
        **common,
    )

    topology_margin = 1 + 3 + 2 + 4
    assert open_voxel_size == pytest.approx(1.0 / (64 - 2 * topology_margin - 2))
    assert open_voxel_size > closed_voxel_size
    assert max(closed_dimensions) <= 64
    assert max(open_dimensions) <= 64


def test_runtime_and_backend_failures_do_not_publish_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_mesh_usd(
        tmp_path / "source.usda",
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
    )
    output = tmp_path / "candidate.usda"
    monkeypatch.setattr(
        openvdb_worker.sdf_tools,
        "create_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            openvdb_worker.sdf_tools.BackendUnavailableError("wheel missing")
        ),
    )

    unavailable = SdfRebuildWorker().execute(
        source=source,
        output=output,
        operation=_operation(source),
    )

    assert unavailable.status == "unavailable"
    assert "wheel missing" in unavailable.failures[0]
    unavailable_evidence = unavailable.metadata["sdf_execution_evidence"]
    assert unavailable_evidence["schema_version"] == (
        sdf_reconstruction.SDF_EXECUTION_EVIDENCE_SCHEMA_VERSION
    )
    assert unavailable_evidence["requested_backend_id"] == "openvdb"
    assert unavailable_evidence["backend_call_status"] == "unavailable"
    assert unavailable_evidence["candidate_validation_status"] == "not_evaluated"
    assert unavailable_evidence["output_digests"] is None
    assert not output.exists()

    _install_fake_backend(
        monkeypatch,
        failure=openvdb_worker.sdf_tools.ResourceLimitError("max_output_faces"),
    )
    failed = SdfRebuildWorker().execute(
        source=source,
        output=output,
        operation=_operation(source),
    )

    assert failed.status == "failed"
    assert "max_output_faces" in failed.failures[0]
    failed_evidence = failed.metadata["invocations"][0]["sdf_execution_evidence"]
    assert failed_evidence["schema_version"] == unavailable_evidence["schema_version"]
    assert failed_evidence["selected_backend_identity"]["backend_id"] == "openvdb"
    assert failed_evidence["backend_call_status"] == "failed"
    assert failed_evidence["candidate_validation_status"] == "not_evaluated"
    assert not output.exists()
