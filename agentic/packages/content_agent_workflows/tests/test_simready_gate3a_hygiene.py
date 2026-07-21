# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial coverage for the Gate 3A hygiene derivative."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import content_agent_workflows.simready.conform_profile as conform_profile_module
import content_agent_workflows.simready.gate3a_hygiene as hygiene_module
from content_agent_workflows.simready import (
    GATE3A_HYGIENE_REQUIREMENT,
    SimReadyConformanceInput,
    inspect_gate3a_physics_inventory,
    run_simready_profile_conformance,
)
from content_agent_workflows.simready.models import SIMREADY_GRASP_PLAN_SCHEMA_VERSION


def _new_stage(path: Path):
    Usd = pytest.importorskip("pxr.Usd")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    body0 = stage.DefinePrim("/Asset/Physics/Body0", "Xform")
    body1 = stage.DefinePrim("/Asset/Physics/Body1", "Xform")
    for body in (body0, body1):
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.CollisionAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body0)
    joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/Physics/Joint")
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    UsdPhysics.FilteredPairsAPI.Apply(body0).CreateFilteredPairsRel().SetTargets(
        [body1.GetPath()]
    )
    UsdPhysics.FilteredPairsAPI.Apply(body1).CreateFilteredPairsRel().SetTargets(
        [body0.GetPath()]
    )
    physics_material = UsdShade.Material.Define(
        stage, "/Asset/Physics/PhysicsMaterial"
    ).GetPrim()
    UsdPhysics.MaterialAPI.Apply(physics_material)
    return stage, root


def _add_camera_helper(stage, path: str, *, hidden: bool = True):
    Gf = pytest.importorskip("pxr.Gf")
    Sdf = pytest.importorskip("pxr.Sdf")
    prim = stage.OverridePrim(path)
    if hidden:
        prim.SetCustomDataByKey("omni:kit:hide_in_stage_window", True)
        prim.SetCustomDataByKey("omni:kit:no_delete", True)
    prim.CreateAttribute("clippingRange", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(1.0, 1000.0)
    )
    prim.CreateAttribute("horizontalAperture", Sdf.ValueTypeNames.Float).Set(50.0)
    prim.CreateAttribute("projection", Sdf.ValueTypeNames.Token).Set("orthographic")
    prim.CreateAttribute("verticalAperture", Sdf.ValueTypeNames.Float).Set(50.0)
    prim.CreateAttribute("omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d).Set(
        Gf.Vec3d(0.0)
    )
    return prim


def _run_hygiene(asset: Path, output_dir: Path):
    expected = inspect_gate3a_physics_inventory(asset)
    return run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(output_dir),
            repair_requirements=[GATE3A_HYGIENE_REQUIREMENT],
            expected_physics_inventory_sha256=expected.sha256,
            foundation_root=str(output_dir / "missing-foundation"),
            force=True,
        )
    )


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_gate3a_hygiene_repairs_only_proven_findings_and_preserves_dependencies(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    Vt = pytest.importorskip("pxr.Vt")
    package = tmp_path / "package"
    package.mkdir()
    asset = package / "asset.usda"
    (package / "OmniPBR.mdl").write_text("mdl 1.0;", encoding="utf-8")
    (package / "shader.glslfx").write_text("-- glslfx version 0.1", encoding="utf-8")
    (package / "texture.png").write_bytes(b"texture")
    stage, root = _new_stage(asset)

    _add_camera_helper(stage, "/ValidHelper")
    _add_camera_helper(stage, "/MissingMetadata", hidden=False)
    missing_camera = stage.OverridePrim("/MissingCamera")
    missing_camera.SetCustomDataByKey("omni:kit:hide_in_stage_window", True)
    missing_camera.SetCustomDataByKey("omni:kit:no_delete", True)
    _add_camera_helper(stage, "/WithChild")
    stage.DefinePrim("/WithChild/Child", "Xform")
    _add_camera_helper(stage, "/Incoming")
    root.CreateRelationship("incomingHelper").AddTarget("/Incoming")
    outgoing = _add_camera_helper(stage, "/Outgoing")
    outgoing.CreateRelationship("target").AddTarget("/Asset")
    nested = _add_camera_helper(stage, "/Asset/NestedHelper")
    assert nested
    typed = UsdGeom.Camera.Define(stage, "/TypedCamera").GetPrim()
    typed.SetCustomDataByKey("omni:kit:hide_in_stage_window", True)
    typed.SetCustomDataByKey("omni:kit:no_delete", True)

    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    primvars = UsdGeom.PrimvarsAPI(mesh)
    unindexed = primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    unindexed.Set(Vt.Vec2fArray([(0, 0), (1, 0), (0, 0)]))
    unindexed.GetAttr().SetMetadata("documentation", "preserved unindexed metadata")
    indexed = primvars.CreatePrimvar(
        "st_1",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    indexed.Set(Vt.Vec2fArray([(0, 0), (0, 0), (1, 0), (1, 1)]))
    indexed.SetIndices(Vt.IntArray([0, 2, 1, 3, 0]))
    indexed.GetIndicesAttr().SetMetadata("documentation", "preserved index metadata")
    already_covered = primvars.CreatePrimvar(
        "st_covered",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    already_covered.Set(Vt.Vec2fArray([(0, 0), (0, 0), (1, 0)]))
    already_covered.SetIndices(Vt.IntArray([0, 0, 2]))
    unindexed_flat = list(unindexed.ComputeFlattened())
    indexed_flat = list(indexed.ComputeFlattened())

    valid_shader = UsdShade.Shader.Define(stage, "/Asset/Looks/Valid")
    valid_shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    valid_shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    valid_shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    valid_shader.CreateIdAttr("mdl:OmniPBR")

    missing_subid = UsdShade.Shader.Define(stage, "/Asset/Looks/MissingSubId")
    missing_subid.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    missing_subid.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    missing_subid.CreateIdAttr("keep-missing-subidentifier")

    id_shader = UsdShade.Shader.Define(stage, "/Asset/Looks/IdImplementation")
    id_shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    id_shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    id_shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.id)
    id_shader.CreateIdAttr("keep-id-implementation")

    non_mdl = UsdShade.Shader.Define(stage, "/Asset/Looks/NonMdl")
    non_mdl.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    non_mdl.SetSourceAsset(Sdf.AssetPath("shader.glslfx"), "mdl")
    non_mdl.SetSourceAssetSubIdentifier("Surface", "mdl")
    non_mdl.CreateIdAttr("keep-non-mdl")
    root.CreateAttribute("previewTexture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("texture.png")
    )
    assert stage.GetRootLayer().Save()
    del stage
    source_bytes = _file_bytes(package)

    output_dir = tmp_path / "conform"
    report = _run_hygiene(asset, output_dir)

    assert report.passed
    assert report.requirements_repaired == [GATE3A_HYGIENE_REQUIREMENT]
    output = Path(report.output_usd_path)
    output_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert output_stage is not None
    assert not output_stage.GetPrimAtPath("/ValidHelper")
    for path in (
        "/MissingMetadata",
        "/MissingCamera",
        "/WithChild",
        "/Incoming",
        "/Outgoing",
        "/Asset/NestedHelper",
        "/TypedCamera",
    ):
        assert output_stage.GetPrimAtPath(path), path

    output_unindexed = UsdGeom.Primvar(
        output_stage.GetPrimAtPath("/Asset/Mesh").GetAttribute("primvars:st")
    )
    output_indexed = UsdGeom.Primvar(
        output_stage.GetPrimAtPath("/Asset/Mesh").GetAttribute("primvars:st_1")
    )
    output_covered = UsdGeom.Primvar(
        output_stage.GetPrimAtPath("/Asset/Mesh").GetAttribute("primvars:st_covered")
    )
    assert list(output_unindexed.Get()) == [(0, 0), (1, 0)]
    assert list(output_unindexed.GetIndices()) == [0, 1, 0]
    assert list(output_unindexed.ComputeFlattened()) == unindexed_flat
    assert output_unindexed.GetAttr().GetMetadata("documentation") == (
        "preserved unindexed metadata"
    )
    assert list(output_indexed.Get()) == [(0, 0), (1, 0), (1, 1)]
    assert list(output_indexed.GetIndices()) == [0, 1, 0, 2, 0]
    assert list(output_indexed.ComputeFlattened()) == indexed_flat
    assert output_indexed.GetIndicesAttr().GetMetadata("documentation") == (
        "preserved index metadata"
    )
    assert list(output_covered.Get()) == [(0, 0), (0, 0), (1, 0)]
    assert list(output_covered.GetIndices()) == [0, 0, 2]

    assert (
        not UsdShade.Shader(output_stage.GetPrimAtPath("/Asset/Looks/Valid"))
        .GetIdAttr()
        .HasAuthoredValue()
    )
    for path in (
        "/Asset/Looks/MissingSubId",
        "/Asset/Looks/IdImplementation",
        "/Asset/Looks/NonMdl",
    ):
        assert UsdShade.Shader(output_stage.GetPrimAtPath(path)).GetIdAttr().Get()
    valid_output = UsdShade.Shader(output_stage.GetPrimAtPath("/Asset/Looks/Valid"))
    assert valid_output.GetSourceAsset("mdl").path == "OmniPBR.mdl"
    assert valid_output.GetSourceAssetSubIdentifier("mdl") == "OmniPBR"
    assert _file_bytes(package) == source_bytes
    assert (output.parent / "OmniPBR.mdl").read_bytes() == source_bytes["OmniPBR.mdl"]
    assert (output.parent / "shader.glslfx").read_bytes() == source_bytes[
        "shader.glslfx"
    ]
    assert (output.parent / "texture.png").read_bytes() == source_bytes["texture.png"]

    receipt = json.loads(
        Path(report.reports[GATE3A_HYGIENE_REQUIREMENT]).read_text(encoding="utf-8")
    )
    assert receipt["source_identity_verified"]
    assert receipt["dependencies_preserved"]
    assert receipt["readback_verified"]
    assert [change["kind"] for change in receipt["changes"]] == [
        "remove_kit_helper_over",
        "compact_primvar",
        "compact_primvar",
        "remove_stale_shader_id",
    ]


def test_gate3a_hygiene_blocks_time_varying_duplicate_primvar(tmp_path: Path) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    Vt = pytest.importorskip("pxr.Vt")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    primvar = UsdGeom.PrimvarsAPI(
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    ).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set(Vt.Vec2fArray([(0, 0), (1, 0), (0, 0)]), Usd.TimeCode(1))
    primvar.Set(Vt.Vec2fArray([(0, 0), (1, 1), (0, 0)]), Usd.TimeCode(2))
    assert stage.GetRootLayer().Save()
    del stage
    source_bytes = asset.read_bytes()

    output_dir = tmp_path / "conform"
    report = _run_hygiene(asset, output_dir)

    assert not report.passed
    assert report.requirements_blocked == [GATE3A_HYGIENE_REQUIREMENT]
    assert "time-varying primvar" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (output_dir / hygiene_module.GATE3A_HYGIENE_OUTPUT_DIR).exists()


def test_gate3a_hygiene_blocks_duplicate_default_with_unique_value_sample(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    Vt = pytest.importorskip("pxr.Vt")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    primvar = UsdGeom.PrimvarsAPI(
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    ).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set(Vt.Vec2fArray([(0, 0), (1, 0), (0, 0)]))
    primvar.Set(
        Vt.Vec2fArray([(0, 0), (1, 0), (1, 1)]),
        Usd.TimeCode(1),
    )
    assert stage.GetRootLayer().Save()
    stage = None
    source_bytes = asset.read_bytes()

    output_dir = tmp_path / "conform"
    report = _run_hygiene(asset, output_dir)

    assert not report.passed
    assert "default and every authored value/index sample" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (output_dir / hygiene_module.GATE3A_HYGIENE_OUTPUT_DIR).exists()


def test_gate3a_hygiene_inspects_index_sample_times_before_proof(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    Vt = pytest.importorskip("pxr.Vt")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    primvar = UsdGeom.PrimvarsAPI(
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    ).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set(Vt.Vec2fArray([(0, 0), (1, 0), (0, 0)]))
    primvar.SetIndices(Vt.IntArray([0, 1, 2]))
    primvar.SetIndices(Vt.IntArray([0, 1, 1]), Usd.TimeCode(1))
    assert stage.GetRootLayer().Save()
    stage = None

    report = _run_hygiene(asset, tmp_path / "conform")

    assert not report.passed
    assert "default and every authored value/index sample" in report.steps[0]["reason"]


def test_gate3a_hygiene_blocks_variant_scoped_primvar_before_publication(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    Vt = pytest.importorskip("pxr.Vt")
    asset = tmp_path / "asset.usda"
    stage, root = _new_stage(asset)
    variants = root.GetVariantSets().AddVariantSet("model")
    expected_values = {
        "A": [(0, 0), (1, 0), (0, 0)],
        "B": [(0, 1), (1, 1), (2, 1)],
    }
    for selection, values in expected_values.items():
        variants.AddVariant(selection)
        variants.SetVariantSelection(selection)
        with variants.GetVariantEditContext():
            mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.faceVarying,
            )
            primvar.Set(Vt.Vec2fArray(values))
    variants.SetVariantSelection("A")
    assert stage.GetRootLayer().Save()
    stage = None
    source_bytes = asset.read_bytes()

    output_dir = tmp_path / "conform"
    report = _run_hygiene(asset, output_dir)

    assert not report.passed
    assert report.status == "BLOCKED"
    assert "variant-qualified edit target" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (output_dir / hygiene_module.GATE3A_HYGIENE_OUTPUT_DIR).exists()
    layer = Sdf.Layer.FindOrOpen(str(asset))
    assert layer is not None
    for selection, values in expected_values.items():
        spec = layer.GetAttributeAtPath(
            Sdf.Path(f"/Asset{{model={selection}}}Mesh.primvars:st")
        )
        assert list(spec.default) == values
        assert not layer.GetAttributeAtPath(Sdf.Path("/Asset/Mesh.primvars:st"))


def test_gate3a_hygiene_retains_helper_targeted_from_inactive_variant(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    stage, root = _new_stage(asset)
    _add_camera_helper(stage, "/Helper")
    variants = root.GetVariantSets().AddVariantSet("model")
    for selection in ("off", "on"):
        variants.AddVariant(selection)
    variants.SetVariantSelection("on")
    with variants.GetVariantEditContext():
        root.CreateRelationship("helperTarget").SetTargets([Sdf.Path("/Helper")])
    variants.SetVariantSelection("off")
    assert stage.GetRootLayer().Save()
    stage = None

    report = _run_hygiene(asset, tmp_path / "conform")

    assert report.passed
    output_stage = Usd.Stage.Open(report.output_usd_path, load=Usd.Stage.LoadAll)
    assert output_stage is not None
    assert output_stage.GetPrimAtPath("/Helper")
    output_variants = (
        output_stage.GetDefaultPrim().GetVariantSets().GetVariantSet("model")
    )
    assert output_variants.SetVariantSelection("on")
    assert output_stage.GetDefaultPrim().GetRelationship(
        "helperTarget"
    ).GetTargets() == [Sdf.Path("/Helper")]


def test_gate3a_hygiene_removes_stale_shader_id_from_package_sublayer(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    package = tmp_path / "package"
    layers = package / "layers"
    layers.mkdir(parents=True)
    (package / "OmniPBR.mdl").write_text("mdl 1.0;", encoding="utf-8")
    sublayer = layers / "look.usda"
    sub_stage = Usd.Stage.CreateNew(str(sublayer))
    sub_stage.DefinePrim("/Asset", "Xform")
    shader = UsdShade.Shader.Define(sub_stage, "/Asset/Looks/Shader")
    shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("../OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("mdl:OmniPBR")
    assert sub_stage.GetRootLayer().Save()
    del sub_stage
    asset = package / "asset.usda"
    stage, _root = _new_stage(asset)
    stage.GetRootLayer().subLayerPaths.append("layers/look.usda")
    assert stage.GetRootLayer().Save()
    del stage
    source_mdl = (package / "OmniPBR.mdl").read_bytes()

    report = _run_hygiene(package, tmp_path / "conform")

    assert report.passed
    output = Path(report.output_usd_path)
    output_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert output_stage is not None
    output_shader = UsdShade.Shader(output_stage.GetPrimAtPath("/Asset/Looks/Shader"))
    assert not output_shader.GetIdAttr().HasAuthoredValue()
    assert output_shader.GetSourceAsset("mdl").path == "../OmniPBR.mdl"
    assert output_shader.GetSourceAssetSubIdentifier("mdl") == "OmniPBR"
    assert (output.parent / "OmniPBR.mdl").read_bytes() == source_mdl
    receipt = json.loads(
        Path(report.reports[GATE3A_HYGIENE_REQUIREMENT]).read_text(encoding="utf-8")
    )
    shader_change = next(
        change
        for change in receipt["changes"]
        if change["kind"] == "remove_stale_shader_id"
    )
    assert shader_change["layers"] == ["layers/look.usda"]


def test_gate3a_hygiene_blocks_invalid_primvar_indices(tmp_path: Path) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    Vt = pytest.importorskip("pxr.Vt")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    primvar = UsdGeom.PrimvarsAPI(
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    ).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set(Vt.Vec2fArray([(0, 0), (0, 0)]))
    primvar.SetIndices(Vt.IntArray([0, 2]))
    assert stage.GetRootLayer().Save()
    del stage
    source_bytes = asset.read_bytes()

    output_dir = tmp_path / "conform"
    report = _run_hygiene(asset, output_dir)

    assert not report.passed
    assert "out of bounds" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert not (output_dir / hygiene_module.GATE3A_HYGIENE_OUTPUT_DIR).exists()


def test_gate3a_hygiene_accepts_sparse_high_primvar_indices(tmp_path: Path) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    Vt = pytest.importorskip("pxr.Vt")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    primvar = UsdGeom.PrimvarsAPI(
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
    ).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    expected_values = [(0, 0), (1, 0), (1, 1), (0, 1)]
    primvar.Set(Vt.Vec2fArray(expected_values))
    primvar.SetIndices(Vt.IntArray([3, 3, 3]))
    assert stage.GetRootLayer().Save()
    stage = None

    report = _run_hygiene(asset, tmp_path / "conform")

    assert report.passed
    output_stage = Usd.Stage.Open(report.output_usd_path, load=Usd.Stage.LoadAll)
    assert output_stage is not None
    assert not output_stage.GetPrimAtPath("/ValidHelper")
    output_primvar = UsdGeom.Primvar(
        output_stage.GetPrimAtPath("/Asset/Mesh").GetAttribute("primvars:st")
    )
    assert list(output_primvar.Get()) == expected_values
    assert list(output_primvar.GetIndices()) == [3, 3, 3]


@pytest.mark.parametrize("use_symlink_alias", [False, True])
def test_gate3a_hygiene_rejects_output_nested_under_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_symlink_alias: bool,
) -> None:
    source_tree = tmp_path / "package"
    source_tree.mkdir()
    asset = source_tree / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    stage = None
    expected = inspect_gate3a_physics_inventory(asset)
    target_output = source_tree / "nested-output"
    if use_symlink_alias:
        target_output.mkdir()
        output_dir = tmp_path / "output-alias"
        try:
            output_dir.symlink_to(target_output, target_is_directory=True)
        except OSError as exc:  # pragma: no cover - platform permission dependent
            pytest.skip(f"symlink creation is unavailable: {exc}")
    else:
        output_dir = target_output
    monkeypatch.setattr(
        hygiene_module,
        "_hygiene_source_package",
        lambda **_kwargs: (source_tree, asset, None),
    )

    result = hygiene_module.repair_gate3a_hygiene(
        asset_path=asset,
        package_root=source_tree,
        output_dir=output_dir,
        expected_physics_inventory_sha256=expected.sha256,
    )

    assert not result.passed
    assert "must not equal or be nested under its source tree" in result.reason
    assert not list(source_tree.rglob(".gate3a-hygiene-build-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics are required")
def test_gate3a_hygiene_keeps_copied_build_private_writable_and_clean(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    asset = package / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    stage = None
    source_bytes = asset.read_bytes()
    asset.chmod(0o444)
    package.chmod(0o555)
    try:
        report = _run_hygiene(package, tmp_path / "conform")
    finally:
        package.chmod(0o755)
        asset.chmod(0o644)

    assert report.passed
    assert asset.read_bytes() == source_bytes
    output_root = Path(report.output_usd_path).parent
    for path in [output_root, *sorted(output_root.rglob("*"))]:
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        assert mode & (stat.S_IRWXG | stat.S_IRWXO) == 0
        assert mode & stat.S_IWUSR
        if path.is_dir():
            assert mode & stat.S_IXUSR
    publish_root = tmp_path / "conform" / hygiene_module.GATE3A_HYGIENE_OUTPUT_DIR
    assert not list(publish_root.glob(".gate3a-hygiene-build-*"))


def test_gate3a_hygiene_blocks_malformed_non_array_primvar(tmp_path: Path) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    asset = tmp_path / "asset.usda"
    stage, root = _new_stage(asset)
    primvar = UsdGeom.PrimvarsAPI(root).CreatePrimvar(
        "malformed",
        Sdf.ValueTypeNames.Float,
        UsdGeom.Tokens.vertex,
    )
    primvar.Set(1.0)
    assert stage.GetRootLayer().Save()
    del stage

    report = _run_hygiene(asset, tmp_path / "conform")

    assert not report.passed
    assert "malformed non-array primvar" in report.steps[0]["reason"]


def test_gate3a_hygiene_publish_failure_rolls_back_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    del stage
    source_bytes = asset.read_bytes()

    def fail_publish(**_kwargs):
        raise OSError("injected publish failure")

    monkeypatch.setattr(hygiene_module, "_publish_isa001_tree", fail_publish)
    output_dir = tmp_path / "conform"
    report = _run_hygiene(asset, output_dir)

    assert not report.passed
    assert "injected publish failure" in report.steps[0]["reason"]
    assert asset.read_bytes() == source_bytes
    assert (output_dir / "staged" / "asset.usda").read_bytes() == source_bytes
    publish_root = output_dir / hygiene_module.GATE3A_HYGIENE_OUTPUT_DIR
    assert publish_root.is_dir()
    assert list(publish_root.iterdir()) == []


def test_gate3a_hygiene_requires_expected_physics_inventory(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    del stage

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=[GATE3A_HYGIENE_REQUIREMENT],
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.requirements_blocked == [GATE3A_HYGIENE_REQUIREMENT]
    assert "requires expected_physics_inventory_sha256" in report.steps[0]["reason"]


def test_gate3a_hygiene_rejects_self_bound_unrigged_source(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "unrigged.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    del stage
    unrigged = inspect_gate3a_physics_inventory(asset)
    assert unrigged.payload["counts"] == {
        "rigid_bodies": 0,
        "colliders": 0,
        "joints": 0,
        "articulation_roots": 0,
        "filtered_pair_bodies": 0,
        "filtered_pair_directed_targets": 0,
        "filtered_pair_relationships": 0,
    }

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=[GATE3A_HYGIENE_REQUIREMENT],
            expected_physics_inventory_sha256=unrigged.sha256,
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert "not a generated Joint Agent physics asset" in report.steps[0]["reason"]
    assert not (tmp_path / "conform" / "gate3a-hygiene").exists()


def test_gate3a_hygiene_final_guard_detects_physics_loss_on_blocked_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    del stage
    expected = inspect_gate3a_physics_inventory(asset)
    original_repair = conform_profile_module._repair_atomic_asset_paths

    def repair_then_strip_joint(**kwargs):
        result = original_repair(**kwargs)
        if result.passed:
            output_stage = Usd.Stage.Open(
                str(result.output_path), load=Usd.Stage.LoadAll
            )
            assert output_stage is not None
            assert output_stage.RemovePrim("/Asset/Physics/Joint")
            assert output_stage.GetRootLayer().Save()
        return result

    monkeypatch.setattr(
        conform_profile_module,
        "_repair_atomic_asset_paths",
        repair_then_strip_joint,
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=[
                GATE3A_HYGIENE_REQUIREMENT,
                "AA.001",
                "NP.006",
            ],
            expected_physics_inventory_sha256=expected.sha256,
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert not report.passed
    assert report.status == "FAIL"
    assert report.requirements_blocked == []
    assert "NP.006" in report.requirements_repaired
    assert any(
        "Final conformance output changed the expected physics inventory" in error
        for error in report.errors
    )


def test_gate3a_hygiene_routes_digit_bearing_requirement_from_validation_report(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    assert (
        conform_profile_module._parse_requirement("rerun G3A.HYG.001 now")
        == GATE3A_HYGIENE_REQUIREMENT
    )
    assert conform_profile_module._parse_requirements("G3A.HYG.001 and RB.COL.001") == [
        GATE3A_HYGIENE_REQUIREMENT,
        "RB.COL.001",
    ]
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    stage = None
    expected = inspect_gate3a_physics_inventory(asset)
    validation_report = tmp_path / "validation.json"
    validation_report.write_text(
        json.dumps({"rerun_reasons": ["G3A.HYG.001: rerun required"]}),
        encoding="utf-8",
    )

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            validation_report_path=str(validation_report),
            expected_physics_inventory_sha256=expected.sha256,
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert report.failed_requirements == [GATE3A_HYGIENE_REQUIREMENT]
    assert report.requirements_repaired == [GATE3A_HYGIENE_REQUIREMENT]
    assert [step["requirement"] for step in report.steps] == [
        GATE3A_HYGIENE_REQUIREMENT
    ]
    output_stage = Usd.Stage.Open(report.output_usd_path)
    assert output_stage is not None
    assert not output_stage.GetPrimAtPath("/ValidHelper")


def test_gate3a_hygiene_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    del stage
    source_bytes = asset.read_bytes()
    output_dir = tmp_path / "conform"

    first = _run_hygiene(asset, output_dir)
    second = _run_hygiene(asset, output_dir)

    assert first.passed and second.passed
    assert second.output_usd_path == first.output_usd_path
    assert asset.read_bytes() == source_bytes
    outputs = list((output_dir / hygiene_module.GATE3A_HYGIENE_OUTPUT_DIR).iterdir())
    assert len(outputs) == 1
    receipt = json.loads(
        Path(second.reports[GATE3A_HYGIENE_REQUIREMENT]).read_text(encoding="utf-8")
    )
    assert receipt["reused_output"]
    output_stage = Usd.Stage.Open(second.output_usd_path)
    assert output_stage is not None
    assert not output_stage.GetPrimAtPath("/ValidHelper")


def test_gate3a_hygiene_orders_before_gsp_aa_and_isa(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    grasp = UsdGeom.BasisCurves.Define(stage, "/Asset/grasp_identifier_01")
    grasp.CreateTypeAttr(UsdGeom.Tokens.linear)
    grasp.CreateCurveVertexCountsAttr([2])
    grasp.CreatePointsAttr([(0, 0, 0), (0, 0, 1)])
    grasp.CreateWidthsAttr([0.01])
    assert stage.GetRootLayer().Save()
    del stage
    output_dir = tmp_path / "conform"

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(output_dir),
            repair_requirements=[
                "ISA.001",
                "AA.001",
                "GSP.001",
                GATE3A_HYGIENE_REQUIREMENT,
            ],
            expected_physics_inventory_sha256=(
                inspect_gate3a_physics_inventory(asset).sha256
            ),
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert [step["requirement"] for step in report.steps] == [
        "PMT.001",
        GATE3A_HYGIENE_REQUIREMENT,
        "GSP.001",
        "AA.001",
        "ISA.001",
    ]
    assert report.steps[1]["input_usd_path"] == report.steps[0]["output_usd_path"]
    assert report.steps[2]["input_usd_path"] == report.steps[1]["output_usd_path"]
    assert report.steps[3]["input_usd_path"] == report.steps[2]["output_usd_path"]
    assert report.steps[4]["input_usd_path"] == report.steps[3]["output_usd_path"]
    output_stage = Usd.Stage.Open(report.output_usd_path, load=Usd.Stage.LoadAll)
    assert output_stage is not None
    assert not output_stage.GetPrimAtPath("/ValidHelper")
    assert output_stage.GetPrimAtPath("/Asset/grasp_identifier_01")


def test_gate3a_hygiene_preserves_plan_lineage_into_byte_changing_gsp(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    physics_material_path = Sdf.Path("/Asset/Physics/PhysicsMaterial")
    for body_path in ("/Asset/Physics/Body0", "/Asset/Physics/Body1"):
        stage.GetPrimAtPath(body_path).CreateRelationship(
            "material:binding:physics"
        ).SetTargets([physics_material_path])
    assert stage.GetRootLayer().Save()
    stage = None
    source_bytes = asset.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    grasp_plan = tmp_path / "grasp-plan.json"
    grasp_plan.write_text(
        json.dumps(
            {
                "schema_version": SIMREADY_GRASP_PLAN_SCHEMA_VERSION,
                "source_asset_sha256": source_sha256,
                "default_prim_path": "/Asset",
                "provenance": {
                    "source": "owner_approved_plan",
                    "approved_by": "simready-owner@example.com",
                    "evidence": ["review://fixture/g3a-gsp-lineage"],
                },
                "grasp_lines": [
                    {
                        "prim_path": "/Asset/grasp_identifier_01",
                        "coordinate_space": "local",
                        "points": [[0, 0, 0], [0, 0, 1]],
                        "widths": [0.01],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    expected = inspect_gate3a_physics_inventory(asset)

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=[GATE3A_HYGIENE_REQUIREMENT, "GSP.001"],
            grasp_plan_path=str(grasp_plan),
            source_asset=str(asset),
            expected_physics_inventory_sha256=expected.sha256,
            foundation_root=str(tmp_path / "missing-foundation"),
            force=True,
        )
    )

    assert report.passed
    assert asset.read_bytes() == source_bytes
    assert [step["requirement"] for step in report.steps] == [
        "PMT.001",
        GATE3A_HYGIENE_REQUIREMENT,
        "GSP.001",
    ]
    output_stage = Usd.Stage.Open(report.output_usd_path, load=Usd.Stage.LoadAll)
    assert output_stage is not None
    assert not output_stage.GetPrimAtPath("/ValidHelper")
    assert output_stage.GetPrimAtPath("/Asset/grasp_identifier_01").IsA(
        UsdGeom.BasisCurves
    )
    gsp_receipt = json.loads(
        Path(report.reports["GSP.001"]).read_text(encoding="utf-8")
    )
    assert gsp_receipt["source_lineage"]["kind"] == "G3A.HYG.001-receipt"
    assert gsp_receipt["source_lineage"]["source_asset_sha256"] == source_sha256
    assert (
        gsp_receipt["source_lineage"]["derivative_asset_sha256"]
        == gsp_receipt["staged_asset_sha256"]
    )


def test_simready_conformance_entrypoint_forwards_required_physics_fingerprint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asset = tmp_path / "asset.usda"
    stage, _root = _new_stage(asset)
    _add_camera_helper(stage, "/ValidHelper")
    assert stage.GetRootLayer().Save()
    stage = None
    expected = inspect_gate3a_physics_inventory(asset)

    code = conform_profile_module.main(
        [
            str(asset),
            "--output-dir",
            str(tmp_path / "conform"),
            "--foundation-root",
            str(tmp_path / "missing-foundation"),
            "--repair",
            GATE3A_HYGIENE_REQUIREMENT,
            "--expected-physics-inventory-sha256",
            expected.sha256,
            "--force",
            "--strict",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["passed"] is True
    assert payload["requirements_repaired"] == [GATE3A_HYGIENE_REQUIREMENT]
