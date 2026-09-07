# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contract tests for CAD-to-SimReady workflow composition."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from content_agent_workflows.cad_to_simready import (
    CAD_TO_SIMREADY_FINAL_RENDER_SCHEMA_VERSION,
    CAD_TO_SIMREADY_STEPS,
    CadToSimReadyInvocation,
    CadToSimReadyRequest,
    bind_artifact,
    build_stage_record,
    execute_cad_to_simready_workflow,
    flatten_usd_for_physics,
    render_final_simready_evidence,
    source_format,
)
from content_agent_workflows.cad_to_simready import (
    final_render as final_render_module,
)
from content_agent_workflows.cad_to_simready.workflow import (
    _is_completed_profile_failure,
)
from content_agent_workflows.common.artifacts import file_sha256


def test_request_defaults_to_skill_routing_without_a_model_override(
    tmp_path: Path,
) -> None:
    source = tmp_path / "assembly.forge.step"
    source.write_bytes(b"STEP")
    materials = tmp_path / "materials.yaml"
    materials.write_text("materials: []\n", encoding="utf-8")

    request = CadToSimReadyRequest(
        asset_id="assembly",
        source=bind_artifact(source),
        source_format=source_format(source),
        requires_cad_converter=True,
        profile="Prop-Robotics-Neutral",
        profile_version="1.0.0",
        materials_yaml=bind_artifact(materials),
        max_iterations=3,
    )

    assert request.source_format == "step"
    assert request.prompt_mode == "skill-routed"
    assert request.model is None


def test_stage_contract_preserves_all_domain_handoffs_in_order() -> None:
    assert [(step.stage, step.name) for step in CAD_TO_SIMREADY_STEPS] == [
        ("convert", "convert-to-usd"),
        ("convert", "canonicalize-usd"),
        ("material", "assign-materials"),
        ("physics", "flatten-for-physics"),
        ("physics", "normalize-units-for-physics"),
        ("physics", "apply-physics"),
        ("validation", "initial-simready-validation"),
        ("validation", "conform-simready-profile"),
        ("validation", "final-simready-validation"),
        ("validation", "render-final-simready"),
    ]


def test_flatten_applies_and_records_explicit_source_unit_contract(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, Vt

    source = tmp_path / "materialized.usda"
    output = tmp_path / "physics-input.usda"
    report = tmp_path / "flatten.json"
    stage = Usd.Stage.CreateNew(str(source))
    prim = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(prim.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    stage.GetRootLayer().Save()

    payload = flatten_usd_for_physics(
        source,
        output,
        report,
        source_meters_per_unit=1.0,
    )

    flattened = Usd.Stage.Open(str(output))
    assert flattened is not None
    assert output.stat().st_mode & 0o7777 == source.stat().st_mode & 0o7777
    assert UsdGeom.GetStageMetersPerUnit(flattened) == pytest.approx(1.0)
    assert payload["input_meters_per_unit"] == pytest.approx(0.001)
    assert payload["input_meters_per_unit_authored"] is True
    assert payload["source_meters_per_unit_override"] == pytest.approx(1.0)
    assert payload["output_meters_per_unit"] == pytest.approx(1.0)
    assert payload["unit_source"] == "explicit-source-contract"
    assert json.loads(report.read_text(encoding="utf-8")) == payload


def test_flatten_materializes_issue_1304_abstract_prototype_instance(
    tmp_path: Path,
) -> None:
    """The physics handoff must not preserve the mixed prototype topology."""

    from pxr import Sdf, Usd, UsdGeom, Vt

    first_conversion = tmp_path / "adjustable_wrench_first_conversion.usda"
    materialized = tmp_path / "adjustable_wrench_materialized.usda"
    output = tmp_path / "adjustable_wrench_physics_input.usda"
    report = tmp_path / "flatten.json"

    stage = Usd.Stage.CreateNew(str(first_conversion))
    asset = UsdGeom.Xform.Define(stage, "/adjustable_wrench")
    stage.SetDefaultPrim(asset.GetPrim())
    prototype = Sdf.PrimSpec(
        stage.GetRootLayer().GetPrimAtPath("/adjustable_wrench"),
        "Prototypes",
        Sdf.SpecifierClass,
    )
    prototype.typeName = "Scope"
    mesh = Sdf.PrimSpec(prototype, "wrench_mesh", Sdf.SpecifierDef)
    mesh.typeName = "Mesh"
    Sdf.AttributeSpec(
        mesh, "points", Sdf.ValueTypeNames.Point3fArray
    ).default = Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    Sdf.AttributeSpec(
        mesh, "faceVertexCounts", Sdf.ValueTypeNames.IntArray
    ).default = Vt.IntArray([3])
    Sdf.AttributeSpec(
        mesh, "faceVertexIndices", Sdf.ValueTypeNames.IntArray
    ).default = Vt.IntArray([0, 1, 2])
    instance = stage.DefinePrim("/adjustable_wrench/body", "Xform")
    instance.GetReferences().AddInternalReference("/adjustable_wrench/Prototypes")
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()

    # Match the issue's intermediate structure: a normal traversal sees no mesh,
    # while proxy traversal reaches the mesh through Flattened_Prototype_1.
    assert stage.Flatten().Export(str(materialized))
    mixed = Usd.Stage.Open(str(materialized))
    assert mixed is not None
    assert [
        str(prim.GetPath()) for prim in mixed.Traverse() if prim.IsA(UsdGeom.Mesh)
    ] == []
    assert [
        str(prim.GetPath())
        for prim in mixed.Traverse(Usd.TraverseInstanceProxies())
        if prim.IsA(UsdGeom.Mesh)
    ] == ["/adjustable_wrench/body/wrench_mesh"]
    assert any(
        prim.name.startswith("Flattened_Prototype_")
        for prim in mixed.GetRootLayer().rootPrims
    )
    expected_prototype_paths = sorted(
        str(prim.path)
        for prim in mixed.GetRootLayer().rootPrims
        if prim.name.startswith("Flattened_Prototype_")
    )
    source_sha256 = file_sha256(materialized)

    payload = flatten_usd_for_physics(materialized, output, report)

    flattened = Usd.Stage.Open(str(output))
    assert flattened is not None
    assert file_sha256(materialized) == source_sha256
    assert flattened.GetDefaultPrim().GetPath() == Sdf.Path("/adjustable_wrench")
    assert [
        str(prim.GetPath()) for prim in flattened.Traverse() if prim.IsA(UsdGeom.Mesh)
    ] == ["/adjustable_wrench/body/wrench_mesh"]
    assert not flattened.GetPrototypes()
    assert not any(
        prim.IsInstance() or prim.IsInstanceable() for prim in flattened.TraverseAll()
    )
    assert not any(
        prim.name.startswith("Flattened_Prototype_")
        for prim in flattened.GetRootLayer().rootPrims
    )
    assert payload["materialized_instance_paths"] == ["/adjustable_wrench/body"]
    assert (
        sorted(payload["removed_flattened_prototype_paths"]) == expected_prototype_paths
    )
    assert payload["renderable_mesh_count"] == 1
    assert payload["renderable_mesh_paths"] == ["/adjustable_wrench/body/wrench_mesh"]


@pytest.mark.parametrize(
    ("relationship_name", "target_path"),
    [
        ("material:binding", "/Flattened_Prototype_1/Material"),
        ("physics:filteredPairs", "/Flattened_Prototype_1/Mesh"),
    ],
)
def test_flatten_rejects_relationship_target_to_removed_prototype(
    tmp_path: Path,
    relationship_name: str,
    target_path: str,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, Vt

    source = tmp_path / "materialized.usda"
    output = tmp_path / "physics-input.usda"
    report = tmp_path / "flatten.json"
    stage = Usd.Stage.CreateNew(str(source))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Flattened_Prototype_1/Mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    stage.DefinePrim("/Flattened_Prototype_1/Material", "Material")
    instance = stage.DefinePrim("/Asset/body", "Xform")
    instance.GetReferences().AddInternalReference("/Flattened_Prototype_1")
    instance.SetInstanceable(True)
    asset.GetPrim().CreateRelationship(relationship_name).SetTargets(
        [Sdf.Path(target_path)]
    )
    stage.GetRootLayer().Save()

    with pytest.raises(
        ValueError,
        match="relationships targeting removed prototypes",
    ):
        flatten_usd_for_physics(source, output, report)

    assert not output.exists()
    assert not report.exists()


def test_flatten_rejects_default_prim_without_visible_renderable_mesh(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, Vt

    source = tmp_path / "hidden.usda"
    output = tmp_path / "missing-output" / "physics-input.usda"
    report = tmp_path / "missing-report" / "flatten.json"
    stage = Usd.Stage.CreateNew(str(source))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/HiddenMesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    stage.GetRootLayer().Save()

    with pytest.raises(
        ValueError,
        match="default prim exposes no visible renderable meshes: /Asset",
    ):
        flatten_usd_for_physics(source, output, report)

    assert not output.exists()
    assert not output.parent.exists()
    assert not report.exists()
    assert not report.parent.exists()


def test_flatten_rejects_visible_mesh_without_topology(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "empty-mesh.usda"
    output = tmp_path / "missing-output" / "physics-input.usda"
    report = tmp_path / "missing-report" / "flatten.json"
    stage = Usd.Stage.CreateNew(str(source))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.Mesh.Define(stage, "/Asset/EmptyMesh")
    stage.GetRootLayer().Save()

    with pytest.raises(
        ValueError,
        match="default prim exposes no visible renderable meshes: /Asset",
    ):
        flatten_usd_for_physics(source, output, report)

    assert not output.parent.exists()
    assert not report.parent.exists()


@pytest.mark.parametrize("meters_per_unit", [0.0, float("nan")])
def test_flatten_rejects_invalid_input_meters_per_unit(
    tmp_path: Path,
    meters_per_unit: float,
) -> None:
    from pxr import Usd, UsdGeom, Vt

    source = tmp_path / "materialized.usda"
    output = tmp_path / "missing-output" / "physics-input.usda"
    report = tmp_path / "missing-report" / "flatten.json"
    stage = Usd.Stage.CreateNew(str(source))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    stage.GetRootLayer().Save()

    with pytest.raises(
        ValueError,
        match="input stage metersPerUnit must be finite and positive",
    ):
        flatten_usd_for_physics(source, output, report)

    assert not output.parent.exists()
    assert not report.parent.exists()


def test_flatten_cleans_created_directories_when_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd, UsdGeom, Vt

    source = tmp_path / "materialized.usda"
    output = tmp_path / "missing" / "nested" / "physics-input.usda"
    report = tmp_path / "missing-report" / "flatten.json"
    stage = Usd.Stage.CreateNew(str(source))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    stage.GetRootLayer().Save()

    original_replace = Path.replace

    def fail_output_replace(path: Path, target: Path) -> Path:
        if target == output:
            raise OSError("forced output replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_output_replace)

    with pytest.raises(ValueError, match="unable to export flattened USD"):
        flatten_usd_for_physics(source, output, report)

    assert not output.parent.exists()
    assert not output.parent.parent.exists()
    assert not report.parent.exists()


def _verified_validation_report(
    final_usd: Path, report: Path, *, passed: bool = True
) -> None:
    status = "PASS" if passed else "FAIL"
    report.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.simready-profile-validation.v3"
                ),
                "asset_path": str(final_usd),
                "asset_sha256": file_sha256(final_usd),
                "errors": [],
                "foundation_checkout_verified": True,
                "passed": passed,
                "profile_name": "Prop-Robotics-Neutral",
                "profile_version": "1.0.0",
                "status": status,
                "validator_runtime_verified": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_completed_profile_failure_is_bound_to_the_exact_final_asset(
    tmp_path: Path,
) -> None:
    final_usd = tmp_path / "final.usda"
    report = tmp_path / "validation.json"
    final_usd.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    _verified_validation_report(final_usd, report, passed=False)

    assert _is_completed_profile_failure(report, final_usd) is True

    final_usd.write_text('#usda 1.0\ndef Xform "Changed" {}\n', encoding="utf-8")
    assert _is_completed_profile_failure(report, final_usd) is False


class _FakeFinalRenderSession:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.session_id = "workflow-cad-final-test"
        self.route = SimpleNamespace(source_revision="a" * 40)
        self.receipt_file = project_dir / "raw/usd_cli_command_receipts.jsonl"
        self.receipt_checkpoint_file = (
            project_dir / "raw/usd_cli_command_receipts.checkpoint.json"
        )

    @classmethod
    def create(cls, **kwargs: Any) -> _FakeFinalRenderSession:
        return cls(Path(kwargs["project_dir"]))

    def require_ovrtx(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True)
        return {"schema_version": "usd-cli.render-probe.v1", "ok": True}

    def open(self, _source: Path, *, read_only: bool = False) -> dict[str, Any]:
        assert read_only is False
        return {"ok": True}

    @staticmethod
    def stage_up_axis_is_y(_source: Path) -> bool:
        return False

    def render_view(self, **kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        name = str(kwargs["name"])
        image_path = output_dir / f"{name}.png"
        response_path = output_dir / f"{name}_response.json"
        camera_path = output_dir / f"{name}_camera.json"
        Image.new(
            "RGB",
            (int(kwargs["width"]), int(kwargs["height"])),
            (sum(name.encode()) % 256, 80, 120),
        ).save(image_path)
        response_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        camera_path.write_text(
            json.dumps({"direction": kwargs["direction"]}), encoding="utf-8"
        )
        return {
            "name": name,
            "direction": kwargs["direction"],
            "image_path": str(image_path),
            "response_path": str(response_path),
            "camera_json_path": str(camera_path),
            "renderer": "ovrtx",
            "renderer_identity": {"engine": "ovrtx"},
        }

    def close(self) -> None:
        self.receipt_file.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_file.write_text(
            json.dumps(
                {
                    "tool": {
                        "name": "usd-cli",
                        "source_revision": self.route.source_revision,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.receipt_checkpoint_file.write_text(
            json.dumps({"receipt_sha256": file_sha256(self.receipt_file)}),
            encoding="utf-8",
        )


def test_final_render_evidence_is_bound_to_the_strictly_validated_usd(
    tmp_path: Path,
) -> None:
    final_usd = tmp_path / "final.usd"
    final_usd.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    validation = tmp_path / "validation.json"
    _verified_validation_report(final_usd, validation)

    payload = render_final_simready_evidence(
        final_usd,
        validation,
        tmp_path / "renders",
        tmp_path / "render-manifest.json",
        width=128,
        height=96,
        turntable_frame_count=4,
        turntable_fps=8,
        session_factory=_FakeFinalRenderSession.create,
    )

    assert payload["schema_version"] == CAD_TO_SIMREADY_FINAL_RENDER_SCHEMA_VERSION
    assert payload["source_usd_sha256"] == file_sha256(final_usd)
    assert set(payload["artifacts"]) == {
        "hero",
        "multiview",
        "turntable_image",
    }
    assert payload["artifacts"]["turntable_image"]["frame_count"] == 4
    assert Path(payload["artifacts"]["turntable_image"]["path"]).is_file()
    assert payload["scene_tool"] == "usd-cli"
    assert payload["render_engine"] == "ovrtx"
    assert payload["usd_cli"]["source_revision"] == "a" * 40
    assert Path(payload["usd_cli"]["command_receipt"]["path"]).is_file()


def test_turntable_gif_rejects_an_encoder_frame_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = [tmp_path / "frame_0000.png", tmp_path / "frame_0001.png"]
    for index, frame in enumerate(frames):
        Image.new("RGB", (32, 32), (index * 80, 40, 120)).save(frame)
    monkeypatch.setattr(
        final_render_module,
        "_gif_metadata",
        lambda _path: {"width": 32, "height": 32, "frame_count": 1},
    )

    with pytest.raises(RuntimeError, match="frame-count verification"):
        final_render_module._encode_turntable_gif(
            frames,
            tmp_path / "turntable.gif",
            fps=8,
        )


def test_final_render_rejects_a_validation_report_for_other_bytes(
    tmp_path: Path,
) -> None:
    final_usd = tmp_path / "final.usd"
    final_usd.write_bytes(b"simready")
    validation = tmp_path / "validation.json"
    _verified_validation_report(final_usd, validation)
    final_usd.write_bytes(b"mutated")

    with pytest.raises(ValueError, match="exact USD"):
        render_final_simready_evidence(
            final_usd,
            validation,
            tmp_path / "renders",
            tmp_path / "render-manifest.json",
            session_factory=_FakeFinalRenderSession.create,
        )


def test_final_render_preserves_visual_evidence_for_a_verified_profile_failure(
    tmp_path: Path,
) -> None:
    final_usd = tmp_path / "final.usd"
    final_usd.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    validation = tmp_path / "validation.json"
    _verified_validation_report(final_usd, validation, passed=False)

    payload = render_final_simready_evidence(
        final_usd,
        validation,
        tmp_path / "renders",
        tmp_path / "render-manifest.json",
        width=128,
        height=96,
        turntable_frame_count=4,
        turntable_fps=8,
        session_factory=_FakeFinalRenderSession.create,
    )

    assert payload["render_status"] == "PASS"
    assert payload["simready_validation"] == {
        "status": "FAIL",
        "passed": False,
        "profile_name": "Prop-Robotics-Neutral",
        "profile_version": "1.0.0",
    }
    assert Path(payload["artifacts"]["turntable_image"]["path"]).is_file()


def test_unstarted_downstream_stage_is_skipped(tmp_path: Path) -> None:
    record = build_stage_record(
        "physics",
        run_dir=tmp_path,
        invocations=[],
        artifacts={},
    )

    assert record.status == "skipped"


def test_preflight_failure_blocks_before_any_authoring_step(tmp_path: Path) -> None:
    source = tmp_path / "asset.stl"
    source.write_text("solid asset\nendsolid asset\n", encoding="utf-8")
    materials = tmp_path / "materials.yaml"
    materials.write_text("materials: []\n", encoding="utf-8")
    request = CadToSimReadyRequest(
        asset_id="asset",
        source=bind_artifact(source),
        source_format="stl",
        requires_cad_converter=False,
        profile="Prop-Robotics-Neutral",
        profile_version="1.0.0",
        materials_yaml=bind_artifact(materials),
        max_iterations=3,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    converter_report = tmp_path / "preflight/convert.json"
    physics_report = tmp_path / "preflight/physics.json"
    paths = {
        "source": source,
        "converter_preflight": converter_report,
        "physics_runtime_preflight": physics_report,
        "simready_preflight": tmp_path / "preflight/simready.json",
    }
    calls: list[str] = []

    def invoke(step: str) -> CadToSimReadyInvocation:
        calls.append(step)
        report = converter_report if len(calls) == 1 else physics_report
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}\n", encoding="utf-8")
        return CadToSimReadyInvocation(
            stage="preflight",
            step=step,
            argv=[step],
            exit_code=0 if len(calls) == 1 else 1,
        )

    result = execute_cad_to_simready_workflow(
        request=request,
        request_path=request_path,
        run_dir=tmp_path,
        artifact_paths=paths,
        invoke=invoke,
    )

    assert calls == ["convert-to-usd-preflight", "physics-runtime-preflight"]
    assert result.status == "blocked"
    assert all(stage.status == "skipped" for stage in result.stages.values())


def test_verified_profile_failure_still_runs_final_visual_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.stl"
    source.write_text("solid asset\nendsolid asset\n", encoding="utf-8")
    materials = tmp_path / "materials.yaml"
    materials.write_text("materials: []\n", encoding="utf-8")
    request = CadToSimReadyRequest(
        asset_id="asset",
        source=bind_artifact(source),
        source_format="stl",
        requires_cad_converter=False,
        profile="Prop-Robotics-Neutral",
        profile_version="1.0.0",
        materials_yaml=bind_artifact(materials),
        max_iterations=3,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    artifact_paths = {"source": source}
    for step in CAD_TO_SIMREADY_STEPS:
        for name in step.produced_artifacts:
            artifact_paths[name] = tmp_path / "artifacts" / name
    for name in (
        "converter_preflight",
        "physics_runtime_preflight",
        "simready_preflight",
    ):
        artifact_paths[name] = tmp_path / "artifacts" / name

    calls: list[str] = []

    def invoke(step_name: str) -> CadToSimReadyInvocation:
        calls.append(step_name)
        produced = next(
            (
                step.produced_artifacts
                for step in CAD_TO_SIMREADY_STEPS
                if step.name == step_name
            ),
            (step_name.removesuffix("-preflight").replace("-", "_"),),
        )
        if step_name == "convert-to-usd-preflight":
            produced = ("converter_preflight",)
        elif step_name == "physics-runtime-preflight":
            produced = ("physics_runtime_preflight",)
        elif step_name == "simready-foundation-preflight":
            produced = ("simready_preflight",)
        for name in produced:
            path = artifact_paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "final_validation_report":
                _verified_validation_report(
                    artifact_paths["final_usd"],
                    path,
                    passed=False,
                )
            else:
                path.write_text(f"{step_name}:{name}\n", encoding="utf-8")
        return CadToSimReadyInvocation(
            stage=(
                "preflight"
                if step_name.endswith("preflight")
                else next(
                    step.stage
                    for step in CAD_TO_SIMREADY_STEPS
                    if step.name == step_name
                )
            ),
            step=step_name,
            argv=[step_name],
            exit_code=1 if step_name == "final-simready-validation" else 0,
        )

    result = execute_cad_to_simready_workflow(
        request=request,
        request_path=request_path,
        run_dir=tmp_path,
        artifact_paths=artifact_paths,
        invoke=invoke,
    )

    assert calls[-2:] == ["final-simready-validation", "render-final-simready"]
    assert result.status == "completed"
    assert result.completed_stages == ["convert", "material", "physics", "validation"]
    assert result.stages["validation"].status == "completed"
    validation_invocation = next(
        invocation
        for invocation in result.stages["validation"].invocations
        if invocation.step == "final-simready-validation"
    )
    assert validation_invocation.exit_code == 1
    assert "final_turntable_render" in result.stages["validation"].artifacts
