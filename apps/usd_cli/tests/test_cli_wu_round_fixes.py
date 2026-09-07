# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave-2 CLI/UX fixes from the 2026-07-09 wu-examples benchmark round.

Covers: global-flag hoisting (--json after the subcommand), find --name glob
patterns, and properties large-array elision.
"""
from __future__ import annotations

from usd_cli.main import _hoist_global_flags
from usd_core.query import _elide_large, find_prims


def _stage_with_prims():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Mesh.Define(stage, "/World/VENTANA_01")
    UsdGeom.Mesh.Define(stage, "/World/PANEL_VENTANA_TRAS")
    UsdGeom.Mesh.Define(stage, "/World/tn__layer")
    UsdGeom.Mesh.Define(stage, "/World/Top_Plate")
    return stage


def test_hoist_global_flags_moves_json_before_subcommand():
    assert _hoist_global_flags(["snapshot", "--json"]) == ["--json", "snapshot"]
    assert _hoist_global_flags(["-q", "find", "--type", "Mesh"]) == [
        "-q", "find", "--type", "Mesh"]
    # tokens after `--` stay put
    assert _hoist_global_flags(["set", "@n1", "a", "--", "--json"]) == [
        "set", "@n1", "a", "--", "--json"]


def test_find_name_substring_still_works():
    stage = _stage_with_prims()
    names = {r["name"] for r in find_prims(stage, name="VENTANA")}
    assert names == {"VENTANA_01", "PANEL_VENTANA_TRAS"}


def test_find_name_glob_pattern():
    stage = _stage_with_prims()
    names = {r["name"] for r in find_prims(stage, name="*VENTANA*")}
    assert names == {"VENTANA_01", "PANEL_VENTANA_TRAS"}
    # anchored glob does not substring-match
    names = {r["name"] for r in find_prims(stage, name="VENTANA*")}
    assert names == {"VENTANA_01"}


def test_elide_large_arrays():
    small = list(range(10))
    assert _elide_large(small) == small
    big = list(range(10000))
    out = _elide_large(big)
    # round 8: long arrays collapse to a single descriptor (the 8-value preview
    # + marker was still ~100 kB of noise per benchmark run)
    assert isinstance(out, str)
    assert "10000 values" in out and "elided" in out


def test_prim_attribute_single_value():
    from pxr import UsdGeom

    from usd_core.query import prim_attribute

    stage = _stage_with_prims()
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Top_Plate"))
    mesh.CreateDisplayColorAttr([(0.1, 0.2, 0.3)])
    # round 8: agents called `properties @ref primvars:displayColor` 22 times
    # before the ATTR positional existed
    out = prim_attribute(stage, "/World/Top_Plate", "primvars:displayColor")
    assert out["attr"] == "primvars:displayColor"
    assert out["authored"] is True
    assert out["value"] == [[0.1, 0.2, 0.3]]


def test_prim_attribute_unknown_suggests_close_matches():
    import pytest

    from usd_core.query import prim_attribute

    stage = _stage_with_prims()
    with pytest.raises(ValueError, match="close matches.*displayColor"):
        prim_attribute(stage, "/World/Top_Plate", "displaycolor")


def test_prim_attribute_full_arrays_below_limit():
    from pxr import UsdGeom, Vt

    from usd_core.query import prim_attribute

    stage = _stage_with_prims()
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Top_Plate"))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(200))))
    # whole-prim elision kicks in far earlier; the single-attr path returns
    # the real numbers up to its own 256-element limit
    out = prim_attribute(stage, "/World/Top_Plate", "faceVertexIndices")
    assert out["value"] == list(range(200))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(500))))
    out = prim_attribute(stage, "/World/Top_Plate", "faceVertexIndices")
    assert isinstance(out["value"], str) and "500 values" in out["value"]
