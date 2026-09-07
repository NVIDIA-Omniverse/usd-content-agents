# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.geometry import (
    scene_ops,
    segmentation_routing,
)
from content_agent_workflows.geometry import (
    segmentation as segmentation_module,
)
from content_agent_workflows.geometry.workflow import (
    GeometryWorkflowInput,
    run_geometry_workflow,
)

_PROTECTED_PRIM_PATH = "/Asset/Parts/body"


def test_geometry_handoff_face_limit_matches_routing_authority() -> None:
    assert (
        segmentation_module._MAX_SOURCE_FACE_COUNT
        == segmentation_routing._MAX_TOTAL_FACES
    )


def test_label_atomicity_uses_bounded_dense_fragment_storage(tmp_path: Path) -> None:
    fragments = tmp_path / "fragment_labels.u32le"
    semantics = tmp_path / "face_labels.u32le"
    fragments.write_bytes(struct.pack("<II", 0, 1))
    semantics.write_bytes(struct.pack("<II", 3, 4))

    segmentation_module._validate_label_atomicity(
        fragment_labels=fragments,
        face_labels=semantics,
        source_face_count=2,
        segment_ids={3, 4},
        expected_fragment_count=2,
    )

    fragments.write_bytes(struct.pack("<II", 0, 2))
    with pytest.raises(ValueError, match="outside fragment_count"):
        segmentation_module._validate_label_atomicity(
            fragment_labels=fragments,
            face_labels=semantics,
            source_face_count=2,
            segment_ids={3, 4},
            expected_fragment_count=2,
        )


def test_label_atomicity_rejects_face_counts_above_the_audit_budget(
    tmp_path: Path,
) -> None:
    fragments = tmp_path / "fragment_labels.u32le"
    semantics = tmp_path / "face_labels.u32le"
    fragments.write_bytes(b"")
    semantics.write_bytes(b"")

    with pytest.raises(ValueError, match="source face count exceeds"):
        segmentation_module._validate_label_atomicity(
            fragment_labels=fragments,
            face_labels=semantics,
            source_face_count=segmentation_module._MAX_SOURCE_FACE_COUNT + 1,
            segment_ids={0},
            expected_fragment_count=1,
        )


def _write_protected_semantic_usd(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    assert stage is not None
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    UsdGeom.Xform.Define(stage, "/Asset/Parts")
    stage.SetDefaultPrim(asset.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, _PROTECTED_PRIM_PATH)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
                Gf.Vec3f(2.0, 0.0, 0.0),
                Gf.Vec3f(3.0, 0.0, 0.0),
                Gf.Vec3f(2.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3, 4, 5]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    assert (
        mesh.GetPrim()
        .CreateAttribute(
            "meshSegmentation:sourceFaceIds",
            Sdf.ValueTypeNames.UIntArray,
        )
        .Set(Vt.UIntArray([0, 1]))
    )
    stage.GetRootLayer().Save()
    return path


def _write_source_identity_usd(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    assert stage is not None
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    for index, name in enumerate(("body", "handle")):
        offset = float(index * 2)
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{name}")
        mesh.CreatePointsAttr(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(offset, 0.0, 0.0),
                    Gf.Vec3f(offset + 1.0, 0.0, 0.0),
                    Gf.Vec3f(offset, 1.0, 0.0),
                    Gf.Vec3f(offset, 0.0, 1.0),
                ]
            )
        )
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3, 3, 3]))
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3])
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return path


def _replace_source_face_ids(
    stage: Usd.Stage,
    value_type: Sdf.ValueTypeName,
    values: Any,
) -> None:
    prim = stage.GetPrimAtPath(_PROTECTED_PRIM_PATH)
    assert prim.RemoveProperty("meshSegmentation:sourceFaceIds")
    assert prim.CreateAttribute(
        "meshSegmentation:sourceFaceIds",
        value_type,
    ).Set(values)


def _install_fake_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[Usd.Stage], None] | None = None,
    *,
    write_shared_metadata: bool = False,
) -> None:
    def fake_run(_self: object, context: dict[str, Any]) -> dict[str, Any]:
        shutil.copy2(context["input_usd_path"], context["output_usd_path"])
        if mutation is not None:
            stage = Usd.Stage.Open(
                context["output_usd_path"],
                load=Usd.Stage.LoadNone,
            )
            assert stage is not None
            mutation(stage)
            stage.GetRootLayer().Save()
        if write_shared_metadata:
            Path(context["output_usd_path"]).with_suffix(".metadata.json").write_text(
                '{"status":"optimizer-output"}\n',
                encoding="utf-8",
            )
        return {
            **context,
            "optimization_success": True,
            "optimization_metadata": {"test_backend": "copy"},
        }

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", fake_run)
    monkeypatch.setattr(
        scene_ops,
        "_geometry_fidelity_check",
        lambda _source, _output: {"status": "pass", "passed": True},
    )


def _optimize_protected(
    source: Path,
    output: Path,
) -> dict[str, Any]:
    return scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy="runtime_efficiency",
        protected_semantic_prim_paths=[_PROTECTED_PRIM_PATH],
    )


def test_optimizer_accepts_exact_uint_source_face_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"
    _install_fake_optimizer(monkeypatch)

    result = _optimize_protected(source, output)

    assert result["status"] == "completed"
    assert result["semantic_prim_boundaries"]["passed"] is True


def test_optimizer_accepts_unrelated_composed_stage_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = tmp_path / "dependency.usda"
    dependency_stage = Usd.Stage.CreateNew(str(dependency))
    assert dependency_stage is not None
    UsdGeom.Xform.Define(dependency_stage, "/Unrelated")
    dependency_stage.GetRootLayer().Save()

    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    source_stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    assert source_stage is not None
    source_stage.GetRootLayer().subLayerPaths.append(dependency.name)
    source_stage.GetRootLayer().Save()
    _install_fake_optimizer(monkeypatch)

    result = _optimize_protected(source, tmp_path / "output.usda")

    assert result["status"] == "completed"
    assert result["semantic_prim_boundaries"]["passed"] is True


def test_optimizer_accepts_float32_round_trip_noise_on_protected_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")

    def nudge_one_float32_step(stage: Usd.Stage) -> None:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_PROTECTED_PRIM_PATH))
        points = Vt.Vec3fArray(mesh.GetPointsAttr().Get())
        points[1] = Gf.Vec3f(1.0000001192092896, 0.0, 0.0)
        assert mesh.GetPointsAttr().Set(points)

    _install_fake_optimizer(monkeypatch, nudge_one_float32_step)

    result = _optimize_protected(source, tmp_path / "output.usda")

    assert result["status"] == "completed"
    assert result["semantic_prim_boundaries"]["passed"] is True


def test_optimizer_rejects_geometry_drift_beyond_float32_round_trip_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")

    def move_protected_point(stage: Usd.Stage) -> None:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_PROTECTED_PRIM_PATH))
        points = Vt.Vec3fArray(mesh.GetPointsAttr().Get())
        points[1] = Gf.Vec3f(1.00001, 0.0, 0.0)
        assert mesh.GetPointsAttr().Set(points)

    _install_fake_optimizer(monkeypatch, move_protected_point)

    result = _optimize_protected(source, tmp_path / "output.usda")

    assert result["status"] == "semantic_fidelity_fallback"
    assert result["semantic_prim_boundaries"]["failures"] == [
        "optimized semantic source-face membership or ordered face geometry "
        f"changed at prim: {_PROTECTED_PRIM_PATH}"
    ]


def test_optimizer_rejects_added_unsupported_boundable_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"

    def add_points(stage: Usd.Stage) -> None:
        points = UsdGeom.Points.Define(stage, "/Asset/InjectedPoints")
        points.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(50.0, 0.0, 0.0)]))
        points.CreateWidthsAttr(Vt.FloatArray([20.0]))

    _install_fake_optimizer(monkeypatch, add_points)

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert result["artifact_role"] == "normalized_copy"
    assert output.read_bytes() == source.read_bytes()
    failures = result["semantic_prim_boundaries"]["failures"]
    assert any(
        "optimized semantic stage contains boundable geometry unsupported by "
        "the fidelity authority: ['/Asset/InjectedPoints (Points)']" in failure
        for failure in failures
    )
    assert any(
        "optimized boundable prim inventory changed; "
        "added=['/Asset/InjectedPoints']" in failure
        for failure in failures
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda mesh: mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark),
        lambda mesh: mesh.GetOrientationAttr().Set(UsdGeom.Tokens.leftHanded),
        lambda mesh: mesh.GetDoubleSidedAttr().Set(True),
        lambda mesh: mesh.CreateHoleIndicesAttr(Vt.IntArray([0])),
        lambda mesh: UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible(),
        lambda mesh: (
            UsdGeom.Imageable(mesh.GetPrim()).GetPurposeAttr().Set(UsdGeom.Tokens.proxy)
        ),
    ],
    ids=[
        "subdivision",
        "orientation",
        "double-sided",
        "hole-indices",
        "visibility",
        "purpose",
    ],
)
def test_optimizer_rejects_protected_render_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[UsdGeom.Mesh], object],
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"

    def mutate_mesh(stage: Usd.Stage) -> None:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_PROTECTED_PRIM_PATH))
        mutation(mesh)

    _install_fake_optimizer(monkeypatch, mutate_mesh)

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert output.read_bytes() == source.read_bytes()
    assert any(
        "optimized protected semantic mesh render state changed at prim: "
        f"{_PROTECTED_PRIM_PATH}" in failure
        for failure in result["semantic_prim_boundaries"]["failures"]
    )


def test_optimizer_rejects_digest_bound_source_mutation(
    tmp_path: Path,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    expected_sha256 = file_sha256(source)
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    assert stage is not None
    UsdGeom.Mesh.Define(stage, "/Asset/InjectedAfterValidation")
    stage.GetRootLayer().Save()

    with pytest.raises(
        ValueError,
        match="Digest-bound Geometry source changed before optimization",
    ):
        scene_ops.optimize_geometry(
            source_usd=source,
            output_usd=tmp_path / "output.usda",
            policy="runtime_efficiency",
            protected_semantic_prim_paths=[_PROTECTED_PRIM_PATH],
            expected_source_sha256=expected_sha256,
        )


def test_optimizer_rejects_source_mutation_during_backend_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"
    expected_sha256 = file_sha256(source)

    def mutate_source_during_run(
        _self: object,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        shutil.copy2(context["input_usd_path"], context["output_usd_path"])
        stage = Usd.Stage.Open(context["input_usd_path"], load=Usd.Stage.LoadNone)
        assert stage is not None
        stage.GetRootLayer().customLayerData = {"mutated_during_optimization": True}
        stage.GetRootLayer().Save()
        return {**context, "optimization_success": True}

    monkeypatch.setattr(
        scene_ops.OptimizeUSDTask,
        "run",
        mutate_source_during_run,
    )

    with pytest.raises(
        ValueError,
        match="Digest-bound Geometry source changed during optimization",
    ):
        scene_ops.optimize_geometry(
            source_usd=source,
            output_usd=output,
            policy="runtime_efficiency",
            protected_semantic_prim_paths=[_PROTECTED_PRIM_PATH],
            expected_source_sha256=expected_sha256,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("value_type", "values"),
    [
        (Sdf.ValueTypeNames.IntArray, Vt.IntArray([0, 1])),
        (Sdf.ValueTypeNames.Int64Array, Vt.Int64Array([0, 1])),
        (Sdf.ValueTypeNames.StringArray, Vt.StringArray(["0", "1"])),
    ],
    ids=["signed-int", "signed-int64", "numeric-strings"],
)
def test_optimizer_rejects_coercible_source_face_id_type_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value_type: Sdf.ValueTypeName,
    values: Any,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"
    _install_fake_optimizer(
        monkeypatch,
        lambda stage: _replace_source_face_ids(stage, value_type, values),
    )

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert output.read_bytes() == source.read_bytes()
    failures = result["semantic_prim_boundaries"]["failures"]
    assert any(
        "optimized semantic mesh source-face provenance must be exact UIntArray"
        in failure
        for failure in failures
    )
    assert Path(result["rejected_output_usd"]).is_file()


def test_optimizer_rejects_coordinated_source_id_and_face_reordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"

    def reorder_faces_without_changing_membership(stage: Usd.Stage) -> None:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_PROTECTED_PRIM_PATH))
        assert mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([3, 4, 5, 0, 1, 2]))
        assert (
            stage.GetPrimAtPath(_PROTECTED_PRIM_PATH)
            .GetAttribute("meshSegmentation:sourceFaceIds")
            .Set(Vt.UIntArray([1, 0]))
        )

    _install_fake_optimizer(monkeypatch, reorder_faces_without_changing_membership)

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert result["semantic_prim_boundaries"]["failures"] == [
        "optimized meshSegmentation:sourceFaceIds values changed at prim: "
        f"{_PROTECTED_PRIM_PATH}"
    ]
    assert output.read_bytes() == source.read_bytes()


def test_optimizer_rejects_time_sampled_source_face_id_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"

    def add_time_sample(stage: Usd.Stage) -> None:
        source_ids = stage.GetPrimAtPath(_PROTECTED_PRIM_PATH).GetAttribute(
            "meshSegmentation:sourceFaceIds"
        )
        assert source_ids.Set(Vt.UIntArray([1, 0]), Usd.TimeCode(1.0))

    _install_fake_optimizer(monkeypatch, add_time_sample)

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert result["semantic_prim_boundaries"]["failures"] == [
        "optimized semantic mesh must be static; time-sampled attribute found at "
        f"{_PROTECTED_PRIM_PATH}.meshSegmentation:sourceFaceIds"
    ]
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("mutation", ["points", "ancestor_transform"])
def test_optimizer_rejects_time_sampled_protected_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"

    def add_time_sample(stage: Usd.Stage) -> None:
        if mutation == "points":
            points = UsdGeom.Mesh(
                stage.GetPrimAtPath(_PROTECTED_PRIM_PATH)
            ).GetPointsAttr()
            values = Vt.Vec3fArray(points.Get())
            values[0] = Gf.Vec3f(10.0, 0.0, 0.0)
            assert points.Set(values, Usd.TimeCode(1.0))
        else:
            xform = UsdGeom.Xformable(stage.GetPrimAtPath("/Asset"))
            translate = xform.AddTranslateOp()
            assert translate.Set(Gf.Vec3d(10.0, 0.0, 0.0), Usd.TimeCode(1.0))

    _install_fake_optimizer(monkeypatch, add_time_sample)

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert any(
        "optimized semantic mesh must be static; time-sampled attribute found"
        in failure
        for failure in result["semantic_prim_boundaries"]["failures"]
    )
    assert output.read_bytes() == source.read_bytes()


def test_optimizer_rejects_invalid_protected_source_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    assert stage is not None
    _replace_source_face_ids(
        stage,
        Sdf.ValueTypeNames.StringArray,
        Vt.StringArray(["0", "1"]),
    )
    stage.GetRootLayer().Save()

    def unexpected_run(_self: object, _context: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("The optimizer must not run for an invalid protected source")

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", unexpected_run)

    with pytest.raises(
        ValueError,
        match="Protected semantic source failed the optimizer lock contract",
    ):
        _optimize_protected(source, tmp_path / "output.usda")


def test_optimizer_rejects_composed_protected_source_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = tmp_path / "dependency.usda"
    dependency_stage = Usd.Stage.CreateNew(str(dependency))
    assert dependency_stage is not None
    UsdGeom.Xform.Define(dependency_stage, "/Asset")
    UsdGeom.Xform.Define(dependency_stage, "/Asset/Parts")
    UsdGeom.Mesh.Define(dependency_stage, _PROTECTED_PRIM_PATH)
    dependency_stage.GetRootLayer().Save()

    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    source_stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    assert source_stage is not None
    source_stage.GetRootLayer().subLayerPaths.append(dependency.name)
    source_stage.GetRootLayer().Save()

    def unexpected_run(_self: object, _context: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("The optimizer must not run for a composed protected source")

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", unexpected_run)

    with pytest.raises(ValueError, match="composed instead of root-authored"):
        _optimize_protected(source, tmp_path / "output.usda")


def test_optimizer_rejects_composed_protected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = tmp_path / "dependency.usda"
    dependency_stage = Usd.Stage.CreateNew(str(dependency))
    assert dependency_stage is not None
    UsdGeom.Xform.Define(dependency_stage, "/Asset")
    UsdGeom.Xform.Define(dependency_stage, "/Asset/Parts")
    UsdGeom.Mesh.Define(dependency_stage, _PROTECTED_PRIM_PATH)
    dependency_stage.GetRootLayer().Save()
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"

    def add_dependency(stage: Usd.Stage) -> None:
        stage.GetRootLayer().subLayerPaths.append(dependency.name)

    _install_fake_optimizer(monkeypatch, add_dependency)

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert output.read_bytes() == source.read_bytes()
    assert any(
        "optimized semantic mesh is composed instead of root-authored" in failure
        for failure in result["semantic_prim_boundaries"]["failures"]
    )


def test_optimizer_rejects_internally_composed_protected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"

    def add_internal_reference(stage: Usd.Stage) -> None:
        UsdGeom.Mesh.Define(stage, "/TemplateMesh")
        assert (
            stage.GetPrimAtPath(_PROTECTED_PRIM_PATH)
            .GetReferences()
            .AddInternalReference("/TemplateMesh")
        )

    _install_fake_optimizer(monkeypatch, add_internal_reference)

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert output.read_bytes() == source.read_bytes()
    failures = result["semantic_prim_boundaries"]["failures"]
    assert (
        "optimized semantic mesh is composed instead of root-authored: "
        f"{_PROTECTED_PRIM_PATH}"
    ) in failures
    assert (
        "optimized boundable prim inventory changed; "
        "added=['/TemplateMesh'], removed=[], type_changed=[]"
    ) in failures


def test_required_parts_imply_required_segmentation(tmp_path: Path) -> None:
    params = GeometryWorkflowInput(
        source_path=tmp_path / "source.usda",
        output_dir=tmp_path / "output",
        segmentation_required=False,
        segmentation_required_parts=["body"],
    )

    assert params.segmentation_required is True
    assert params.model_dump()["segmentation_required"] is True


def test_required_parts_without_run_cannot_produce_ready_handoff(
    tmp_path: Path,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    params = GeometryWorkflowInput(
        source_path=source,
        output_dir=tmp_path / "workflow-output",
        optimization_policy="skip",
        run_shared_usd_validation=False,
        run_asset_audit=False,
    )
    params.segmentation_required_parts = ["body"]

    result = run_geometry_workflow(params)

    assert params.segmentation_required is False
    assert result.success is False
    assert result.segmentation_outcome == "rejected"
    assert result.segmentation_route == "deterministic_shell_split"
    assert result.validation_status == "fail"
    assert result.handoff_ready == "no"
    evidence = json.loads(
        Path(result.validation_evidence_path or "").read_text(encoding="utf-8")
    )
    segmentation_check = next(
        check for check in evidence["checks"] if check["name"] == "part_segregation"
    )
    assert segmentation_check["status"] == "fail"
    assert segmentation_check["metadata"]["required"] is True
    assert any(
        "deterministic shell-split artifact" in failure
        for failure in segmentation_check["failures"]
    )


def test_workflow_reuses_source_identity_without_agentic_segmentation(
    tmp_path: Path,
) -> None:
    source = _write_source_identity_usd(tmp_path / "source.usda")

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "workflow-output",
            segmentation_required=True,
            segmentation_required_parts=["body", "handle"],
            optimization_policy="runtime_efficiency",
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert result.success is True, result.error
    assert result.validation_status == "conditional"
    assert result.handoff_ready == "conditional"
    assert result.segmentation_route == "reuse_source_identity"
    assert result.segmentation_outcome == "conditional"
    routing_path = Path(result.segmentation_routing_path or "")
    assert routing_path.is_file()
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    assert routing["should_invoke_agentic_segmentation"] is False
    assert routing["observed_source_semantic_names"] == ["body", "handle"]

    manifest = json.loads(
        Path(result.handoff_manifest_path or "").read_text(encoding="utf-8")
    )
    assert manifest["provenance"]["shared_optimization"]["policy"] == "skip"
    assert manifest["optimization_policy"] == "runtime_efficiency"
    assert manifest["effective_optimization_policy"] == "skip"
    assert manifest["segmentation"]["routing"]["route"] == "reuse_source_identity"
    assert {
        (part["name"], part["source_fidelity_tier"])
        for part in manifest["semantic_parts"]
    } == {
        ("body", "source_asserted_identity"),
        ("handle", "source_asserted_identity"),
    }
    evidence_bundle = json.loads(
        Path(result.evidence_bundle_path or "").read_text(encoding="utf-8")
    )
    assert evidence_bundle["optimization_policy"] == "runtime_efficiency"
    assert evidence_bundle["effective_optimization_policy"] == "skip"


def test_optimizer_quarantines_rejected_shared_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_protected_semantic_usd(tmp_path / "source.usda")
    output = tmp_path / "output.usda"

    def reorder_points(stage: Usd.Stage) -> None:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_PROTECTED_PRIM_PATH))
        points = list(mesh.GetPointsAttr().Get())
        points[0] = Gf.Vec3f(-1.0, 0.0, 0.0)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

    _install_fake_optimizer(
        monkeypatch,
        reorder_points,
        write_shared_metadata=True,
    )

    result = _optimize_protected(source, output)

    assert result["status"] == "semantic_fidelity_fallback"
    assert output.read_bytes() == source.read_bytes()
    assert not output.with_suffix(".metadata.json").exists()
    rejected_metadata = Path(result["rejected_shared_metadata_path"])
    assert rejected_metadata.name == "output.semantic-boundary-rejected.metadata.json"
    assert rejected_metadata.read_text(encoding="utf-8") == (
        '{"status":"optimizer-output"}\n'
    )
