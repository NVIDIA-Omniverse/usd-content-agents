# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the general stage validator and the material audit —
`usd-cli validate [--fix]` and `usd-cli material audit`. Direct usd_core tests (need pxr).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")


def _messy_stage():
    """A stage exhibiting every defect class the validator reports:
    unbound mesh, dangling binding target, bound subset without family metadata,
    out-of-range subset indices, shader without a source, unresolved asset path,
    an unused material, and an MDL material (renderer-compat case)."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from usd_core import materials

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    board = UsdGeom.Mesh.Define(stage, "/World/Board")
    board.GetFaceVertexCountsAttr().Set([3, 3, 3, 3])
    UsdGeom.Mesh.Define(stage, "/World/Chip")

    fr4 = materials.create_preview_material(stage, name="FR4", color=(0.2, 0.5, 0.2))
    materials.bind_material(stage, fr4, "/World/Board")
    materials.create_preview_material(stage, name="Unused", color=(1, 1, 1))
    materials.create_mdl_material(stage, name="Solder", color=(0.7, 0.7, 0.75))

    # subset bound by hand: has a binding but no familyName, and one bad face index
    sub = UsdGeom.Subset.CreateGeomSubset(UsdGeom.Imageable(board.GetPrim()), "Pads",
                                          UsdGeom.Tokens.face, [0, 99])
    sub.GetPrim().CreateRelationship("material:binding").SetTargets([Sdf.Path(fr4)])
    # dangling binding target on the chip
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Chip"))
    stage.GetPrimAtPath("/World/Chip").CreateRelationship("material:binding").SetTargets(
        [Sdf.Path("/World/Looks/Missing")])
    # a shader with no id / source asset inside an otherwise empty material
    UsdShade.Material.Define(stage, "/World/Looks/Broken")
    UsdShade.Shader.Define(stage, "/World/Looks/Broken/Sh")
    # unresolved asset attribute
    board.GetPrim().CreateAttribute("myTex", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("/nowhere/missing.png"))
    return stage


def _issues(report, check):
    return [i for i in report["issues"] if i["check"] == check]


def test_validate_reports_every_defect_class():
    from usd_core.validate import validate_stage

    report = validate_stage(_messy_stage())
    assert not report["ok"]
    assert any("unresolved asset" in i["message"] for i in _issues(report, "assets"))
    assert any("/World/Looks/Missing" in i["message"] for i in _issues(report, "bindings"))
    subset_msgs = " | ".join(i["message"] for i in _issues(report, "subsets"))
    assert "familyName" in subset_msgs and "out of range" in subset_msgs
    shader_msgs = " | ".join(i["message"] for i in _issues(report, "shaders"))
    assert "no info:id or source asset" in shader_msgs


def test_validate_reports_renderer_compat():
    from usd_core.validate import validate_stage

    report = validate_stage(_messy_stage())
    by_kind = {r["shader_kind"]: r for r in report["renderer_compat"]}
    assert by_kind["UsdPreviewSurface"]["backends"]["ovrtx"] == "supported"
    mdl = by_kind["mdl"]
    assert mdl["mdl_module"] == "OmniPBR.mdl"
    assert mdl["backends"]["ovrtx"] == "supported"


def test_validate_warns_on_ignored_preview_inputs():
    from pxr import Usd, UsdGeom
    from usd_core import materials
    from usd_core.validate import validate_stage

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    materials.create_preview_material(stage, name="Typo", color=(1, 0, 0),
                                      inputs={"rougness": 0.3})  # not a real input
    report = validate_stage(stage)
    warns = [i for i in report["issues"] if i["severity"] == "warn"]
    assert any("rougness" in w["message"] for w in warns)


def test_validate_fix_repairs_what_it_safely_can():
    from usd_core.validate import validate_stage

    stage = _messy_stage()
    report = validate_stage(stage, fix=True)
    fixed = " | ".join(report["fixed"])
    assert "removed invalid binding target /World/Looks/Missing" in fixed
    assert "familyName=materialBind" in fixed

    again = validate_stage(stage)
    assert not _issues(again, "bindings")  # dangling target gone
    # the out-of-range face index is deliberately NOT auto-fixed, still reported
    assert any("out of range" in i["message"] for i in _issues(again, "subsets"))


def test_material_audit_counts_and_details():
    from usd_core.audit import material_audit

    report = material_audit(_messy_stage(), effective=True, include_subsets=True)
    c = report["counts"]
    assert c["renderables"] == 2 and c["bound_direct"] == 1 and c["unbound"] == 1
    assert c["invalid_binding_targets"] == 1
    assert report["unbound_paths"] == ["/World/Chip"]
    # Unused = authored materials nothing resolves to (Solder MDL + Unused + the broken one)
    assert set(report["unused_materials"]) == {"/World/Looks/Unused", "/World/Looks/Solder",
                                               "/World/Looks/Broken"}
    (fam,) = report["subset_families"]
    # the bound-but-family-less subset contributes nothing to materialBind coverage —
    # that missing metadata is exactly what the audit flags in subsets_detail
    assert fam["mesh"] == "/World/Board" and fam["faces_covered"] == 0
    (detail,) = fam["subsets_detail"]
    assert detail["name"] == "Pads" and detail["problems"]
    per_prim = {r["path"]: r for r in report["renderables"]}
    assert per_prim["/World/Board"]["binding"] == "direct"
    assert per_prim["/World/Chip"]["material"] is None
    assert not report["ok"]  # invalid target + subset problems
