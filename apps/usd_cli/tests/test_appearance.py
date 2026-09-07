# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clean-slate appearance overlay, audit, atomicity, and undo/redo coverage."""

from __future__ import annotations


def _preview_material(stage, path: str, color):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _quad(stage, path: str):
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1, -1, 0),
            Gf.Vec3f(1, -1, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(-1, 1, 0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    return mesh


def _appearance_fixture(path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    direct = _quad(stage, "/World/Direct")
    _quad(stage, "/World/Inherited")
    group = UsdGeom.Xform.Define(stage, "/World/Group").GetPrim()
    collected = _quad(stage, "/World/Group/Collected")
    subset_mesh = _quad(stage, "/World/SubsetMesh")
    purpose_mesh = _quad(stage, "/World/PurposeMesh")
    mdl_mesh = _quad(stage, "/World/MdlMesh")
    direct_shader_mesh = _quad(stage, "/World/DirectShader")
    display = _quad(stage, "/World/Display")
    inherited_display_root = UsdGeom.Xform.Define(
        stage, "/World/InheritedDisplay"
    ).GetPrim()
    inherited_display = _quad(stage, "/World/InheritedDisplay/Mesh")

    red = _preview_material(stage, "/World/Looks/Red", (1, 0, 0))
    blue = _preview_material(stage, "/World/Looks/Blue", (0, 0, 1))
    green = _preview_material(stage, "/World/Looks/Green", (0, 1, 0))

    UsdShade.MaterialBindingAPI.Apply(world).Bind(red)
    UsdShade.MaterialBindingAPI.Apply(direct.GetPrim()).Bind(blue)

    collection = Usd.CollectionAPI.Apply(group, "greenParts")
    collection.CreateIncludesRel().AddTarget(collected.GetPath())
    UsdShade.MaterialBindingAPI.Apply(group).Bind(
        collection,
        green,
        "greenParts",
        UsdShade.Tokens.strongerThanDescendants,
    )

    subset = UsdGeom.Subset.CreateGeomSubset(
        UsdGeom.Imageable(subset_mesh.GetPrim()),
        "PaintedFace",
        UsdGeom.Tokens.face,
        [0],
    )
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(blue)
    purpose_api = UsdShade.MaterialBindingAPI.Apply(purpose_mesh.GetPrim())
    purpose_api.Bind(red, materialPurpose=UsdShade.Tokens.preview)
    purpose_api.Bind(blue, materialPurpose=UsdShade.Tokens.full)

    mdl_material = UsdShade.Material.Define(stage, "/World/Looks/MDL")
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/MDL/Shader")
    mdl_shader.CreateIdAttr("mdlMaterial")
    mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mdl_material.CreateSurfaceOutput("mdl").ConnectToSource(
        mdl_shader.ConnectableAPI(), "out"
    )
    UsdShade.MaterialBindingAPI.Apply(mdl_mesh.GetPrim()).Bind(mdl_material)

    direct_shader = UsdShade.Shader.Define(stage, "/World/Looks/DirectShader")
    direct_shader.CreateIdAttr("UsdPreviewSurface")
    shader_surface = direct_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    surface_output = direct_shader_mesh.GetPrim().CreateAttribute(
        "outputs:ri:surface",
        Sdf.ValueTypeNames.Token,
    )
    surface_output.AddConnection(shader_surface.GetAttr().GetPath())
    volume_output = direct_shader_mesh.GetPrim().CreateAttribute(
        "outputs:volume:renderer",
        Sdf.ValueTypeNames.Token,
    )
    volume_output.Set("direct-volume")
    volume_output.Set("time-sampled-volume", 1.0)

    color = display.CreateDisplayColorPrimvar()
    color.Set([Gf.Vec3f(0.8, 0.3, 0.1)])
    color.SetIndices([0])
    opacity = display.CreateDisplayOpacityPrimvar()
    opacity.Set([0.25])
    opacity.Set([0.5], 1.0)

    inherited_primvars = UsdGeom.PrimvarsAPI(inherited_display_root)
    inherited_color = inherited_primvars.CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.constant,
    )
    inherited_color.Set([Gf.Vec3f(0.1, 0.7, 0.2)])
    inherited_color.SetIndices([0])
    inherited_opacity = inherited_primvars.CreatePrimvar(
        "displayOpacity",
        Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.constant,
    )
    inherited_opacity.Set([0.35])
    inherited_opacity.Set([0.65], 1.0)
    assert UsdGeom.PrimvarsAPI(inherited_display.GetPrim()).FindPrimvarWithInheritance(
        "displayColor"
    ).Get() == [Gf.Vec3f(0.1, 0.7, 0.2)]

    stage.SetDefaultPrim(world)
    stage.GetRootLayer().Save()
    return path


def _instance_fixture(root):
    from pxr import Gf, Usd, UsdGeom, UsdShade

    asset_path = root / "instance_asset.usda"
    asset = Usd.Stage.CreateNew(str(asset_path))
    asset_root = UsdGeom.Xform.Define(asset, "/Asset").GetPrim()
    mesh = _quad(asset, "/Asset/Mesh")
    material = _preview_material(asset, "/Asset/Looks/Red", (1, 0, 0))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    mesh.CreateDisplayColorPrimvar().Set([Gf.Vec3f(1, 0, 0)])
    asset.SetDefaultPrim(asset_root)
    asset.GetRootLayer().Save()

    scene_path = root / "instance_scene.usda"
    scene = Usd.Stage.CreateNew(str(scene_path))
    world = UsdGeom.Xform.Define(scene, "/World").GetPrim()
    instance = scene.DefinePrim("/World/Instance", "Xform")
    instance.GetReferences().AddReference("./instance_asset.usda")
    instance.SetInstanceable(True)
    scene.SetDefaultPrim(world)
    scene.GetRootLayer().Save()
    return scene_path, asset_path


def test_clear_masks_all_effective_appearance_and_roundtrips(tmp_path):
    from pxr import Gf, UsdGeom, UsdShade
    from usd_core.appearance import audit_appearance
    from usd_core.session import Session

    source = _appearance_fixture(tmp_path / "source.usda")
    source_bytes = source.read_bytes()
    session = Session.open(source)
    root_before = session._stage.GetRootLayer().ExportToString()

    before = session.appearance_audit()
    assert before.ok and not before.summary["clear"]
    assert before.summary["effective_material_bindings"] >= 4
    assert before.summary["effective_shader_appearances"] >= 4
    assert before.summary["direct_shader_outputs"] == 2
    assert before.summary["display_values"] == 6
    assert any(
        row.get("shader") == "mdlMaterial" for row in before.data["effective_shaders"]
    )

    cleared = session.appearance_clear()
    assert cleared.ok, cleared.issues
    assert cleared.summary["clear"]
    assert cleared.summary["bindings_masked"] >= 6
    assert cleared.summary["display_attributes_masked"] == 6
    assert cleared.summary["direct_shader_outputs_masked"] == 2
    assert session._stage.GetEditTarget().GetLayer() == (
        session._stage.GetSessionLayer()
    )
    assert session._stage.GetRootLayer().ExportToString() == root_before
    assert source.read_bytes() == source_bytes

    audit = audit_appearance(session._stage)
    assert audit["clear"], audit
    assert all(value == 0 for value in audit["counts"].values())
    for prim in session._stage.TraverseAll():
        for rel in prim.GetRelationships():
            if rel.GetName() == "material:binding" or rel.GetName().startswith(
                "material:binding:"
            ):
                assert rel.GetTargets() == []
        for name in (
            "primvars:displayColor",
            "primvars:displayColor:indices",
            "primvars:displayOpacity",
        ):
            attr = prim.GetAttribute(name)
            if attr:
                assert attr.Get() is None
                assert attr.GetTimeSamples() == []
    direct_shader_prim = session._stage.GetPrimAtPath("/World/DirectShader")
    for name in ("outputs:ri:surface", "outputs:volume:renderer"):
        attr = direct_shader_prim.GetAttribute(name)
        assert attr.GetConnections() == []
        assert attr.Get() is None
        assert attr.GetTimeSamples() == []

    undone = session.undo()
    assert undone.ok and undone.data["undone"] == 1
    assert session._stage.GetEditTarget().GetLayer() == (session._stage.GetRootLayer())
    restored = audit_appearance(session._stage)
    assert not restored["clear"]
    assert restored["counts"]["direct_shader_outputs"] == 2
    assert direct_shader_prim.GetAttribute("outputs:ri:surface").GetConnections()
    assert direct_shader_prim.GetAttribute("outputs:volume:renderer").Get() == (
        "direct-volume"
    )
    assert direct_shader_prim.GetAttribute(
        "outputs:volume:renderer"
    ).GetTimeSamples() == [1.0]
    direct = session._stage.GetPrimAtPath("/World/Direct")
    material, _ = UsdShade.MaterialBindingAPI(direct).ComputeBoundMaterial()
    assert material.GetPath().pathString == "/World/Looks/Blue"
    assert session._stage.GetPrimAtPath("/World/Display").GetAttribute(
        "primvars:displayOpacity"
    ).Get() == [0.25]
    assert session._stage.GetPrimAtPath("/World/Display").GetAttribute(
        "primvars:displayOpacity"
    ).GetTimeSamples() == [1.0]
    inherited_color = (
        UsdGeom.PrimvarsAPI(
            session._stage.GetPrimAtPath("/World/InheritedDisplay/Mesh")
        )
        .FindPrimvarWithInheritance("displayColor")
        .Get()
    )
    assert inherited_color == [Gf.Vec3f(0.1, 0.7, 0.2)]

    redone = session.redo()
    assert redone.ok and redone.data["redone"] == 1
    assert audit_appearance(session._stage)["clear"]
    assert source.read_bytes() == source_bytes


def test_new_decision_authors_above_clear_mask_and_save_is_fail_closed(tmp_path):
    from pxr import Usd
    from usd_core.appearance import audit_appearance
    from usd_core.session import Session

    source = _appearance_fixture(tmp_path / "source.usda")
    source_bytes = source.read_bytes()
    session = Session.open(source)
    assert session.appearance_clear().ok

    direct_ref = session.refs.ref_for_path("/World/Direct")
    assigned = session.material(
        ref=direct_ref,
        color=[0.2, 0.4, 0.8],
        name="AcceptedDecision",
    )
    assert assigned.ok, assigned.issues
    after_decision = audit_appearance(session._stage)
    assert after_decision["counts"]["effective_material_bindings"] == 1
    assert {row["path"] for row in after_decision["effective_bindings"]} == {
        "/World/Direct"
    }

    derivative = tmp_path / "derivative.usda"
    rejected = session.save(str(derivative))
    assert not rejected.ok and "--flatten" in rejected.issues[0].message
    overwrite = session.save(str(source), flatten=True)
    assert not overwrite.ok and "source immutable" in overwrite.issues[0].message
    checkpoint = session.checkpoint_save("would-omit-overlay")
    assert not checkpoint.ok and "--full" in checkpoint.issues[0].message
    exported = session.export("usda", str(tmp_path / "exported.usda"))
    assert not exported.ok and "would omit" in exported.issues[0].message

    saved = session.save(str(derivative), flatten=True)
    assert saved.ok, saved.issues
    assert source.read_bytes() == source_bytes
    reopened = Usd.Stage.Open(str(derivative))
    assert reopened
    derivative_audit = audit_appearance(reopened)
    assert derivative_audit["counts"]["effective_material_bindings"] == 1
    assert {row["path"] for row in derivative_audit["effective_bindings"]} == {
        "/World/Direct"
    }


def test_undo_speculative_material_restores_clear_mask_and_flattened_result(tmp_path):
    """Undo must restore the explicit-empty overlay, not reveal source binding."""
    from pxr import Usd
    from usd_core.appearance import audit_appearance
    from usd_core.session import Session

    source = _appearance_fixture(tmp_path / "source.usda")
    session = Session.open(source)
    assert session.appearance_clear().ok

    direct_ref = session.refs.ref_for_path("/World/Direct")
    assigned = session.material(
        ref=direct_ref,
        color=[0.7, 0.7, 0.7],
        name="SpeculativeDecision",
    )
    assert assigned.ok, assigned.issues
    assert audit_appearance(session._stage)["counts"][
        "effective_material_bindings"
    ] == 1

    undone = session.undo()
    assert undone.ok, undone.issues
    assert audit_appearance(session._stage)["clear"]
    relationship = session._stage.GetPrimAtPath("/World/Direct").GetRelationship(
        "material:binding"
    )
    assert relationship and relationship.GetTargets() == []

    derivative = tmp_path / "undone.usda"
    saved = session.save(str(derivative), flatten=True)
    assert saved.ok, saved.issues
    reopened = Usd.Stage.Open(str(derivative))
    assert reopened and audit_appearance(reopened)["clear"]


def test_clear_deinstances_only_in_overlay_and_undo_restores_instances(tmp_path):
    from pxr import Usd
    from usd_core.appearance import audit_appearance
    from usd_core.session import Session

    source, asset = _instance_fixture(tmp_path)
    source_bytes = source.read_bytes()
    asset_bytes = asset.read_bytes()
    session = Session.open(source)
    assert session._stage.GetPrimAtPath("/World/Instance").IsInstance()
    assert any(
        prim.IsInstanceProxy()
        for prim in session._stage.Traverse(Usd.TraverseInstanceProxies())
    )

    result = session.appearance_clear()
    assert result.ok, result.issues
    assert result.summary["deinstanced_roots"] == 1
    assert not session._stage.GetPrimAtPath("/World/Instance").IsInstance()
    assert audit_appearance(session._stage)["clear"]
    assert source.read_bytes() == source_bytes
    assert asset.read_bytes() == asset_bytes

    assert session.undo().ok
    assert session._stage.GetPrimAtPath("/World/Instance").IsInstance()
    assert not audit_appearance(session._stage)["clear"]
    assert session.redo().ok
    assert audit_appearance(session._stage)["clear"]


def test_clear_failure_rolls_back_the_whole_session_layer(tmp_path, monkeypatch):
    from usd_core import appearance
    from usd_core.session import Session

    source = _appearance_fixture(tmp_path / "source.usda")
    source_bytes = source.read_bytes()
    session = Session.open(source)
    layer_before = session._stage.GetSessionLayer().ExportToString()
    audit_before = appearance.audit_appearance(session._stage)

    def fail_after_bindings(*_args, **_kwargs):
        raise RuntimeError("injected display-mask failure")

    monkeypatch.setattr(appearance, "_block_display_attribute", fail_after_bindings)
    result = session.appearance_clear()
    assert not result.ok
    assert "injected display-mask failure" in result.issues[0].message
    assert session._stage.GetSessionLayer().ExportToString() == layer_before
    assert session._stage.GetEditTarget().GetLayer() == (session._stage.GetRootLayer())
    assert appearance.audit_appearance(session._stage) == audit_before
    assert session.history.entries() == []
    assert source.read_bytes() == source_bytes


def test_clear_rejects_read_only_session_without_any_edit(tmp_path):
    from usd_core.session import Session

    source = _appearance_fixture(tmp_path / "source.usda")
    source_bytes = source.read_bytes()
    session = Session()
    assert session.open_stage(str(source), read_only=True).ok
    layer_before = session._stage.GetSessionLayer().ExportToString()

    result = session.appearance_clear()
    assert not result.ok and "read-only" in result.issues[0].message
    assert session._stage.GetSessionLayer().ExportToString() == layer_before
    assert session.history.entries() == []
    assert source.read_bytes() == source_bytes


def test_appearance_cli_and_server_wiring(monkeypatch):
    from typer.testing import CliRunner
    from usd_cli import main
    from usd_core.models import Response
    from usd_server.app import COMMANDS

    calls = []

    def dispatch(command, payload):
        calls.append((command, payload))
        return Response(command=command, summary={"clear": command.endswith("clear")})

    monkeypatch.setattr(main, "dispatch", dispatch)
    runner = CliRunner()
    assert runner.invoke(main.app, ["appearance", "clear"]).exit_code == 0
    assert runner.invoke(main.app, ["appearance", "audit"]).exit_code == 0
    assert calls == [("appearance.clear", {}), ("appearance.audit", {})]
    assert COMMANDS["appearance.clear"] == "appearance_clear"
    assert COMMANDS["appearance.audit"] == "appearance_audit"
