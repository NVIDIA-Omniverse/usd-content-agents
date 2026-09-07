# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-10 verification regressions from the agv-lift field report.

  1  verify's warnings ride the verdict line ("PASS with warnings (…)") and
     `verify --strict` demotes them to FAIL — task-03 read "NOT self-contained"
     twice below a PASS headline and shipped anyway.
"""
from __future__ import annotations

from pxr import Sdf, Usd, UsdGeom, UsdShade

from usd_core.config import Config
from usd_core.session import Session


def _bind_preview(stage, prim, mat_path: str, color=(0.2, 0.4, 0.6)):
    mat = UsdShade.Material.Define(stage, mat_path)
    sh = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    return mat


#   1  verify: warnings on the verdict line; --strict demotes them to FAIL


def _hidden_mesh_scene(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    visible = UsdGeom.Cube.Define(stage, "/World/visible_cube")
    hidden = UsdGeom.Cube.Define(stage, "/World/hidden_cube")
    UsdGeom.Imageable(hidden.GetPrim()).MakeInvisible()
    # keep the rest of the verdict green: every renderable bound
    _bind_preview(stage, visible.GetPrim(), "/World/Looks/mat_a")
    _bind_preview(stage, hidden.GetPrim(), "/World/Looks/mat_b")
    path = tmp_path / "deliverable.usda"
    stage.GetRootLayer().Export(str(path))
    return path


def test_verify_warnings_ride_the_verdict_line(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="t")
    r = s.verify(file=str(_hidden_mesh_scene(tmp_path)))
    assert r.ok is True
    headline = r.data["text"].splitlines()[0]
    assert headline.startswith("PASS with warnings")
    assert "1 hidden renderables" in headline
    assert "--strict" in headline  # the escalation path is advertised in-line


def test_verify_strict_demotes_warnings_to_fail(tmp_path):
    s = Session(Config(project_dir=tmp_path), name="t")
    r = s.verify(file=str(_hidden_mesh_scene(tmp_path)), strict=True)
    assert r.ok is False
    assert r.summary["strict"] is True
    assert r.summary["error_type"] == "verification"
    headline = r.data["text"].splitlines()[0]
    assert headline.startswith("FAIL (strict:")
    assert "hidden renderables" in headline


def test_verify_clean_deliverable_still_plain_pass(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/cube")
    _bind_preview(stage, cube.GetPrim(), "/World/Looks/mat")
    path = tmp_path / "clean.usda"
    stage.GetRootLayer().Export(str(path))

    s = Session(Config(project_dir=tmp_path), name="t")
    for strict in (False, True):
        r = s.verify(file=str(path), strict=strict)
        assert r.ok is True, r.data["text"]
        assert r.data["text"].splitlines()[0].startswith("PASS — ")
