# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import pytest
from pxr import Gf, Usd, UsdGeom, Vt

from content_agent_workflows.geometry import segmentation_routing
from content_agent_workflows.geometry.segmentation_routing import route_segmentation


def _stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    assert stage is not None
    UsdGeom.Xform.Define(stage, "/Asset")
    return stage


def _mesh(
    stage: Usd.Stage,
    prim_path: str,
    *,
    points: list[tuple[float, float, float]],
    counts: list[int],
    indices: list[int],
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def _save(stage: Usd.Stage) -> Path:
    path = Path(stage.GetRootLayer().realPath)
    stage.GetRootLayer().Save()
    return path


def _write_connected_triangles(path: Path) -> Path:
    stage = _stage(path)
    _mesh(
        stage,
        "/Asset/Mesh",
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        counts=[3, 3],
        indices=[0, 1, 2, 2, 1, 3],
    )
    return _save(stage)


def _write_multiple_meshes(path: Path) -> Path:
    stage = _stage(path)
    # Author in reverse lexical order to verify that evidence is path-sorted.
    _mesh(
        stage,
        "/Asset/handle",
        points=[(2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)],
        counts=[3],
        indices=[0, 1, 2],
    )
    _mesh(
        stage,
        "/Asset/body",
        points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        counts=[3],
        indices=[0, 1, 2],
    )
    return _save(stage)


def test_completed_run_reference_has_first_precedence(tmp_path: Path) -> None:
    run_dir = tmp_path / "completed-run"

    decision = route_segmentation(
        requested=False,
        source_usd_path=tmp_path / "missing.usda",
        completed_run_reference=run_dir,
        required_semantic_names=["handle", "body"],
    )

    assert decision.route == "consume_completed_run"
    assert decision.completed_run_reference == str(run_dir)
    assert decision.required_semantic_names == ["body", "handle"]
    assert decision.metrics.inspected_prim_count == 0


def test_no_request_does_not_inspect_source(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.usda"
    malformed.write_text("not usd", encoding="utf-8")

    decision = route_segmentation(
        requested=False,
        source_usd_path=malformed,
    )

    assert decision.route == "not_requested"
    assert decision.metrics.source_file_bytes is None
    assert decision.metrics.inspected_prim_count == 0


def test_no_request_does_not_emit_source_identity_warning() -> None:
    decision = route_segmentation(
        requested=False,
        source_usd_path=None,
        supplied_semantic_parts=["body", "handle"],
    )

    assert decision.route == "not_requested"
    assert decision.observed_source_semantic_names == ["body", "handle"]
    assert "source_names_are_not_geometry_proof" not in decision.reason_codes


def test_multiple_named_mesh_prims_reuse_source_identity(tmp_path: Path) -> None:
    source = _write_multiple_meshes(tmp_path / "named-meshes.usda")

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "reuse_source_identity"
    assert decision.observed_source_semantic_names == ["body", "handle"]
    assert decision.metrics.mesh_prim_paths == ["/Asset/body", "/Asset/handle"]
    assert decision.metrics.mesh_prim_count == 2
    assert "multiple_named_mesh_prims" in decision.reason_codes


def test_long_valid_mesh_identity_name_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "long-name.usda"
    long_name = "x" * 257
    stage = _stage(source)
    _mesh(
        stage,
        f"/Asset/{long_name}",
        points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        counts=[3],
        indices=[0, 1, 2],
    )
    _mesh(
        stage,
        "/Asset/body",
        points=[(2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)],
        counts=[3],
        indices=[0, 1, 2],
    )
    _save(stage)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "reuse_source_identity"
    assert decision.observed_source_semantic_names == ["body", long_name]


def test_invalid_required_name_returns_rejected_decision(tmp_path: Path) -> None:
    decision = route_segmentation(
        requested=True,
        source_usd_path=tmp_path / "unused.usda",
        required_semantic_names=["body/handle"],
    )

    assert decision.route == "rejected"
    assert decision.reason_codes == ["invalid_required_semantic_names"]
    assert decision.metrics.inspected_prim_count == 0


def test_unloaded_payload_prevents_partial_stage_routing(tmp_path: Path) -> None:
    payload_path = _write_multiple_meshes(tmp_path / "payload.usda")
    source = _write_connected_triangles(tmp_path / "partial.usda")
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    payload_prim = stage.DefinePrim("/Asset/Deferred", "Xform")
    payload_prim.GetPayloads().AddPayload(str(payload_path))
    stage.GetRootLayer().Save()

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "undetermined"
    assert decision.reason_codes == ["unloaded_payloads"]
    assert decision.metrics.unloaded_payload_count == 1
    assert decision.should_invoke_agentic_segmentation is False


def test_face_geom_subset_reuses_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "subset.usda"
    stage = _stage(source)
    _mesh(
        stage,
        "/Asset/Mesh",
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        counts=[3, 3],
        indices=[0, 1, 2, 2, 1, 3],
    )
    subset = UsdGeom.Subset.Define(stage, "/Asset/Mesh/handle")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateIndicesAttr(Vt.IntArray([1]))
    _save(stage)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "reuse_source_identity"
    assert decision.observed_source_semantic_names == ["handle"]
    assert decision.metrics.geom_subset_count == 1
    assert decision.semantic_name_evidence[0].prim_paths == ["/Asset/Mesh/handle"]


def test_long_valid_geom_subset_name_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "long-subset.usda"
    stage = _stage(source)
    _mesh(
        stage,
        "/Asset/Mesh",
        points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        counts=[3],
        indices=[0, 1, 2],
    )
    long_name = "s" * 257
    subset = UsdGeom.Subset.Define(stage, f"/Asset/Mesh/{long_name}")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateIndicesAttr(Vt.IntArray([0]))
    _save(stage)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "reuse_source_identity"
    assert decision.observed_source_semantic_names == [long_name]


def test_supplied_semantic_part_metadata_reuses_identity_without_usd() -> None:
    decision = route_segmentation(
        requested=True,
        supplied_semantic_parts=[{"name": "body"}, {"name": "handle"}],
    )

    assert decision.route == "reuse_source_identity"
    assert decision.observed_source_semantic_names == ["body", "handle"]
    assert decision.metrics.supplied_semantic_part_count == 2
    assert decision.metrics.topology_inspected is False


def test_unbound_semantic_part_metadata_does_not_bypass_fused_mesh_segmentation(
    tmp_path: Path,
) -> None:
    source = _write_connected_triangles(tmp_path / "fused-with-names.usda")

    decision = route_segmentation(
        requested=True,
        source_usd_path=source,
        supplied_semantic_parts=[{"name": "body"}, {"name": "handle"}],
    )

    assert decision.route == "agentic_semantic_segmentation"
    assert decision.observed_source_semantic_names == []
    assert decision.metrics.supplied_semantic_part_count == 2
    assert "unbound_semantic_part_metadata_is_guidance_only" in (decision.reason_codes)


def test_malformed_supplied_semantic_part_metadata_is_rejected() -> None:
    decision = route_segmentation(
        requested=True,
        supplied_semantic_parts=[{"name": "body"}, 42],  # type: ignore[list-item]
    )

    assert decision.route == "rejected"
    assert decision.reason_codes == ["malformed_semantic_part_metadata"]


def test_multiple_edge_connected_shells_route_to_deterministic_split(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shells.usda"
    stage = _stage(source)
    _mesh(
        stage,
        "/Asset/Mesh",
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ],
        counts=[3, 3],
        indices=[0, 1, 2, 3, 4, 5],
    )
    _save(stage)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "deterministic_shell_split"
    assert decision.metrics.edge_connected_shell_count == 2
    assert decision.metrics.shell_face_counts == [1, 1]
    assert decision.should_invoke_agentic_segmentation is False


def test_vertex_contact_without_shared_edge_is_two_shells(tmp_path: Path) -> None:
    source = tmp_path / "vertex-contact.usda"
    stage = _stage(source)
    _mesh(
        stage,
        "/Asset/Mesh",
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ],
        counts=[3, 3],
        indices=[0, 1, 2, 0, 3, 4],
    )
    _save(stage)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "deterministic_shell_split"
    assert decision.metrics.edge_connected_shell_count == 2


def test_one_fused_connected_triangular_mesh_routes_to_agentic_segmentation(
    tmp_path: Path,
) -> None:
    source = _write_connected_triangles(tmp_path / "fused.usda")

    decision = route_segmentation(
        requested=True,
        source_usd_path=source,
        required_semantic_names=["handle", "body"],
    )

    assert decision.route == "agentic_semantic_segmentation"
    assert decision.should_invoke_agentic_segmentation is True
    assert decision.metrics.edge_connected_shell_count == 1
    assert decision.metrics.all_faces_triangular is True
    assert decision.missing_required_semantic_names == ["body", "handle"]


def test_routing_normalizes_each_point_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_connected_triangles(tmp_path / "single-point-pass.usda")
    original = segmentation_routing._point_coordinates
    calls = 0

    def counted(
        point: Any,
        metrics: segmentation_routing._MetricsBuilder,
    ) -> tuple[float, float, float]:
        nonlocal calls
        calls += 1
        return original(point, metrics)

    monkeypatch.setattr(segmentation_routing, "_point_coordinates", counted)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "agentic_semantic_segmentation"
    assert calls == decision.metrics.point_count == 4


def test_routing_rejects_before_expanding_topology_over_memory_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_connected_triangles(tmp_path / "over-budget.usda")
    monkeypatch.setattr(segmentation_routing, "_MAX_TOPOLOGY_WORKING_BYTES", 1)

    def unexpected_point_conversion(
        _point: Any,
        _metrics: segmentation_routing._MetricsBuilder,
    ) -> NoReturn:
        raise AssertionError("over-budget topology must fail before point expansion")

    monkeypatch.setattr(
        segmentation_routing,
        "_point_coordinates",
        unexpected_point_conversion,
    )

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "rejected"
    assert decision.reason_codes == ["inspection_memory_budget_exceeded"]
    assert decision.metrics.topology_inspected is False


def test_connected_non_triangular_mesh_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "quad.usda"
    stage = _stage(source)
    _mesh(
        stage,
        "/Asset/Mesh",
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        counts=[4],
        indices=[0, 1, 2, 3],
    )
    _save(stage)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "rejected"
    assert decision.reason_codes == ["unsupported_mesh_configuration"]
    assert decision.metrics.all_faces_triangular is False


def test_missing_source_is_undetermined(tmp_path: Path) -> None:
    decision = route_segmentation(
        requested=True,
        source_usd_path=tmp_path / "missing.usda",
    )

    assert decision.route == "undetermined"
    assert decision.reason_codes == ["source_usd_uninspectable"]


def test_malformed_mesh_topology_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "malformed-topology.usda"
    stage = _stage(source)
    _mesh(
        stage,
        "/Asset/Mesh",
        points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        counts=[3],
        indices=[0, 1, 9],
    )
    _save(stage)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "rejected"
    assert decision.reason_codes == ["malformed_mesh_topology"]
    assert decision.metrics.malformed_mesh_count == 1
    assert decision.should_invoke_agentic_segmentation is False


def test_pathological_single_face_is_rejected_before_edge_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert segmentation_routing._MAX_VERTICES_PER_FACE == 10_000
    monkeypatch.setattr(segmentation_routing, "_MAX_VERTICES_PER_FACE", 4)
    source = tmp_path / "oversized-face.usda"
    stage = _stage(source)
    _mesh(
        stage,
        "/Asset/Mesh",
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.5, 1.5, 0.0),
            (0.0, 1.0, 0.0),
        ],
        counts=[5],
        indices=[0, 1, 2, 3, 4],
    )
    _save(stage)

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "rejected"
    assert decision.reason_codes == ["malformed_mesh_topology"]
    assert decision.should_invoke_agentic_segmentation is False


def test_required_name_gaps_are_evidence_not_geometry_proof(tmp_path: Path) -> None:
    source = _write_multiple_meshes(tmp_path / "required-names.usda")

    decision = route_segmentation(
        requested=True,
        source_usd_path=source,
        required_semantic_names=["knob", "body"],
    )

    assert decision.route == "reuse_source_identity"
    assert decision.missing_required_semantic_names == ["knob"]
    assert "required_semantic_names_missing_from_source_identity" in (
        decision.reason_codes
    )
    body_evidence = next(
        item for item in decision.semantic_name_evidence if item.name == "body"
    )
    assert body_evidence.geometry_semantics_proven is False


def test_routing_result_is_stable_and_deterministic(tmp_path: Path) -> None:
    source = _write_multiple_meshes(tmp_path / "stable.usda")
    kwargs = {
        "requested": True,
        "source_usd_path": source,
        "required_semantic_names": ["handle", "body", "knob"],
        "supplied_semantic_parts": [{"name": "dial"}, {"name": "body"}],
    }

    first = route_segmentation(**kwargs)
    second = route_segmentation(**kwargs)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.observed_source_semantic_names == ["body", "handle"]
    assert "unbound_semantic_part_metadata_is_guidance_only" in first.reason_codes
    assert first.metrics.mesh_prim_paths == ["/Asset/body", "/Asset/handle"]
