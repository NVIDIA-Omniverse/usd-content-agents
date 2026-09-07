# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repair of in-range but degenerate UV layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from texture_agent.functions.uv_generation import (
    DEGENERATE_UV_SPAN,
    normalize_uvs,
    repair_degenerate_uvs,
)


def _mesh_stage(tmp_path: Path, uvs: list[tuple[float, float]]):
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(tmp_path / "mesh.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    primvar.Set(Vt.Vec2fArray([Gf.Vec2f(u, v) for u, v in uvs]))
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Root"))
    return stage, mesh


def _span(mesh) -> tuple[float, float]:
    from pxr import UsdGeom

    uvs = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st").Get()
    us = [c[0] for c in uvs]
    vs = [c[1] for c in uvs]
    return max(us) - min(us), max(vs) - min(vs)


def test_repairs_degenerate_uvs_to_full_range(tmp_path: Path) -> None:
    """A tiny in-range UV island is rescaled so textures sample usefully."""
    pytest.importorskip("pxr")
    stage, mesh = _mesh_stage(
        tmp_path,
        [(0.500, 0.500), (0.5015, 0.500), (0.5015, 0.5015), (0.500, 0.5015)],
    )

    # normalize_uvs only handles out-of-range UVs, so it leaves this alone.
    assert normalize_uvs(stage) == 0
    assert _span(mesh)[0] == pytest.approx(0.0015, abs=1e-6)

    assert repair_degenerate_uvs(stage) == 1

    du, dv = _span(mesh)
    assert du == pytest.approx(0.95, abs=1e-3)
    assert dv == pytest.approx(0.95, abs=1e-3)


def test_leaves_healthy_uvs_untouched(tmp_path: Path) -> None:
    """A normal 0..1 unwrap is not rescaled."""
    pytest.importorskip("pxr")
    stage, mesh = _mesh_stage(
        tmp_path, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    )

    assert repair_degenerate_uvs(stage) == 0
    assert _span(mesh) == (pytest.approx(1.0), pytest.approx(1.0))


def test_respects_min_span_threshold(tmp_path: Path) -> None:
    """A span at or above the threshold is treated as intentional."""
    pytest.importorskip("pxr")
    stage, mesh = _mesh_stage(
        tmp_path, [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    )

    assert repair_degenerate_uvs(stage, min_span=0.1) == 0
    assert _span(mesh)[0] == pytest.approx(0.2)

    # Raising the threshold above the island's span makes it eligible.
    assert repair_degenerate_uvs(stage, min_span=0.5) == 1
    assert _span(mesh)[0] == pytest.approx(0.95, abs=1e-3)


def test_skips_zero_area_uv_bounds(tmp_path: Path) -> None:
    """A collapsed UV island has no footprint to rescale."""
    pytest.importorskip("pxr")
    stage, mesh = _mesh_stage(
        tmp_path, [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    )

    assert repair_degenerate_uvs(stage) == 0
    assert _span(mesh) == (pytest.approx(0.0), pytest.approx(0.0))


def test_skips_non_finite_uvs(tmp_path: Path) -> None:
    """Non-finite UVs are reported rather than rescaled into garbage."""
    pytest.importorskip("pxr")
    stage, mesh = _mesh_stage(
        tmp_path,
        [(0.5, 0.5), (float("nan"), 0.5), (0.5015, 0.5015), (0.5, 0.5015)],
    )

    assert repair_degenerate_uvs(stage) == 0


def test_honours_target_prim_paths(tmp_path: Path) -> None:
    """Scoped repair only touches the requested prims."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, UsdGeom, Vt

    stage, mesh = _mesh_stage(
        tmp_path,
        [(0.500, 0.500), (0.5015, 0.500), (0.5015, 0.5015), (0.500, 0.5015)],
    )
    other = UsdGeom.Mesh.Define(stage, "/Root/Other")
    other.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    other.CreateFaceVertexCountsAttr([4])
    other.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(other.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.2, 0.2),
                Gf.Vec2f(0.2015, 0.2),
                Gf.Vec2f(0.2015, 0.2015),
                Gf.Vec2f(0.2, 0.2015),
            ]
        )
    )

    assert repair_degenerate_uvs(stage, target_prim_paths=["/Root/Mesh"]) == 1
    assert _span(mesh)[0] == pytest.approx(0.95, abs=1e-3)
    other_uvs = UsdGeom.PrimvarsAPI(other.GetPrim()).GetPrimvar("st").Get()
    other_span = max(c[0] for c in other_uvs) - min(c[0] for c in other_uvs)
    assert other_span == pytest.approx(0.0015, abs=1e-6)


def test_default_threshold_is_exposed() -> None:
    """The default is a module constant so callers can reason about it."""
    assert DEGENERATE_UV_SPAN == pytest.approx(0.05)


def test_skips_empty_uv_array(tmp_path: Path) -> None:
    """A declared but empty st primvar has nothing to rescale."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(tmp_path / "empty.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec2fArray([]))
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Root"))

    assert repair_degenerate_uvs(stage) == 0


def test_skips_non_mutable_prims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prim the mutability guard rejects is left untouched."""
    pytest.importorskip("pxr")
    from texture_agent.functions import uv_generation as uv

    stage, mesh = _mesh_stage(
        tmp_path,
        [(0.500, 0.500), (0.5015, 0.500), (0.5015, 0.5015), (0.500, 0.5015)],
    )
    monkeypatch.setattr(uv, "_prepare_mutable_prim", lambda *_args: False)

    assert uv.repair_degenerate_uvs(stage) == 0
    assert _span(mesh)[0] == pytest.approx(0.0015, abs=1e-6)


def test_skips_instance_proxy_meshes(tmp_path: Path) -> None:
    """Instance proxies are read-only, so they are left to their prototype."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    model_path = tmp_path / "model.usda"
    model = Usd.Stage.CreateNew(str(model_path))
    mesh = UsdGeom.Mesh.Define(model, "/Model/Mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.5, 0.5),
                Gf.Vec2f(0.5015, 0.5),
                Gf.Vec2f(0.5015, 0.5015),
                Gf.Vec2f(0.5, 0.5015),
            ]
        )
    )
    model.SetDefaultPrim(model.GetPrimAtPath("/Model"))
    model.GetRootLayer().Save()

    root = Usd.Stage.CreateNew(str(tmp_path / "root.usda"))
    instance = root.DefinePrim("/Root/Instance", "Xform")
    instance.GetReferences().AddReference("./model.usda")
    instance.SetInstanceable(True)
    root.SetDefaultPrim(root.GetPrimAtPath("/Root"))

    # The proxy mesh is only reachable through instance-proxy traversal and
    # cannot be authored on, so the repair must decline it.
    assert repair_degenerate_uvs(root) == 0


def test_repair_reaches_meshes_behind_an_instanceable_reference(tmp_path: Path) -> None:
    """The repair must see geometry authored under a class and instanced.

    The default predicate yields neither abstract prims nor instance proxies. On
    the acceptance asset the source geometry sits under a class prim
    (``/Body/Prototypes``, ``Sdf.SpecifierClass``) that an instanceable prim
    references, so the repair saw zero of thirteen meshes and logged success.
    Reported as issue #916.
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(tmp_path / "instanced.usda"))
    stage.CreateClassPrim("/Body/Prototypes")
    UsdGeom.Xform.Define(stage, "/Body/Prototypes/Body")
    mesh = UsdGeom.Mesh.Define(stage, "/Body/Prototypes/Body/Part/Mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.5, 0.5),
                Gf.Vec2f(0.5015, 0.5),
                Gf.Vec2f(0.5015, 0.5015),
                Gf.Vec2f(0.5, 0.5015),
            ]
        )
    )
    inst = UsdGeom.Xform.Define(stage, "/Body/Body").GetPrim()
    inst.GetReferences().AddInternalReference("/Body/Prototypes/Body")
    inst.SetInstanceable(True)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Body"))

    assert not [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]

    repaired = repair_degenerate_uvs(stage)

    assert repaired == 1
    # Instancing must survive: de-instancing to reach the mesh is what broke
    # packaging in #918.
    assert stage.GetPrimAtPath("/Body/Body").IsInstanceable()
    assert stage.GetPrototypes()
    uvs = (
        UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/Body/Prototypes/Body/Part/Mesh"))
        .GetPrimvar("st")
        .Get()
    )
    span = max(uv[0] for uv in uvs) - min(uv[0] for uv in uvs)
    assert span == pytest.approx(0.95, rel=1e-3)


def test_repair_catches_a_footprint_collapsed_in_one_axis(tmp_path: Path) -> None:
    """A span usable in u but collapsed in v is still degenerate.

    Measured on the acceptance asset: Gunmetal_Dark spans 0.0523 x 0.0260 and
    samples ~1400 of a 1024px map's million texels, but cleared a threshold
    applied to the wider axis alone.
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(tmp_path / "aniso.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.5, 0.5),
                Gf.Vec2f(0.5523, 0.5),
                Gf.Vec2f(0.5523, 0.5260),
                Gf.Vec2f(0.5, 0.5260),
            ]
        )
    )

    assert repair_degenerate_uvs(stage) == 1


def test_repair_skips_inactive_meshes(tmp_path: Path) -> None:
    """Prims absent from the composed stage must not be rewritten.

    The traversal deliberately admits abstract and undefined prims, since CAD
    sources live under classes and are overridden by ``over`` prims. Inactive
    prims contribute nothing to the composed stage, so editing them would be a
    change nobody asked for.
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(tmp_path / "inactive.usda"))

    def _degenerate_mesh(path: str) -> UsdGeom.Mesh:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(
            [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0.5, 0.5),
                    Gf.Vec2f(0.5015, 0.5),
                    Gf.Vec2f(0.5015, 0.5015),
                    Gf.Vec2f(0.5, 0.5015),
                ]
            )
        )
        return mesh

    _degenerate_mesh("/Root/Live/Mesh")
    _degenerate_mesh("/Root/Dormant/Mesh")
    # Deactivate the mesh itself. Deactivating an ancestor instead would drop the
    # whole subtree from composition, so no predicate could reach it and the test
    # would pass without exercising anything.
    stage.GetPrimAtPath("/Root/Dormant/Mesh").SetActive(False)

    repaired = repair_degenerate_uvs(stage)

    assert repaired == 1

    def _span(path: str) -> float:
        prim = stage.GetPrimAtPath(path)
        uvs = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get()
        return max(uv[0] for uv in uvs) - min(uv[0] for uv in uvs)

    assert _span("/Root/Live/Mesh") == pytest.approx(0.95, rel=1e-3)

    # An inactive prim is unreachable by path, so reactivate it to confirm the
    # repair left it alone.
    stage.GetPrimAtPath("/Root/Dormant/Mesh").SetActive(True)
    assert _span("/Root/Dormant/Mesh") == pytest.approx(0.0015, rel=1e-2)


def test_repair_reaches_an_undefined_mesh(tmp_path: Path) -> None:
    """Undefined prims must stay in scope.

    One CAD asset carries 4283 meshes under a class and 4283 sibling specs that
    report ``IsDefined() == False``; their opinion is the one that composes.
    Restricting the traversal to defined prims repaired 4270 specs and left the
    composed stage exactly as degenerate as before -- a silent no-op.
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    layer = Sdf.Layer.CreateNew(str(tmp_path / "over.usda"))
    spec = Sdf.CreatePrimInLayer(layer, Sdf.Path("/Root/Part/Mesh"))
    spec.specifier = Sdf.SpecifierOver
    spec.typeName = "Mesh"
    stage = Usd.Stage.Open(layer)

    mesh = stage.GetPrimAtPath("/Root/Part/Mesh")
    assert mesh.IsA(UsdGeom.Mesh)
    assert not mesh.IsDefined(), "fixture must model an undefined prim"

    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.5, 0.5),
                Gf.Vec2f(0.5015, 0.5),
                Gf.Vec2f(0.5015, 0.5015),
                Gf.Vec2f(0.5, 0.5015),
            ]
        )
    )

    assert repair_degenerate_uvs(stage) == 1

    uvs = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()
    span = max(uv[0] for uv in uvs) - min(uv[0] for uv in uvs)
    assert span == pytest.approx(0.95, rel=1e-3)


def test_repair_skips_meshes_behind_an_unloaded_payload(tmp_path: Path) -> None:
    """An unloaded payload contributes nothing, so its meshes are left alone.

    This is the ``Usd.PrimIsLoaded`` half of the traversal predicate. Loading the
    payload afterwards must show the UVs untouched, which distinguishes "skipped"
    from "repaired but invisible".
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    tiny = Vt.Vec2fArray(
        [
            Gf.Vec2f(0.5, 0.5),
            Gf.Vec2f(0.5015, 0.5),
            Gf.Vec2f(0.5015, 0.5015),
            Gf.Vec2f(0.5, 0.5015),
        ]
    )

    payload_path = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    mesh = UsdGeom.Mesh.Define(payload_stage, "/Payload/Mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(tiny)
    payload_stage.SetDefaultPrim(payload_stage.GetPrimAtPath("/Payload"))
    payload_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(tmp_path / "host.usda"), load=Usd.Stage.LoadNone)
    host = UsdGeom.Xform.Define(stage, "/Root/Deferred").GetPrim()
    host.GetPayloads().AddPayload(str(payload_path))
    assert not host.IsLoaded(), "fixture must model an unloaded payload"

    assert repair_degenerate_uvs(stage) == 0

    # Load it and confirm the UVs were never rewritten.
    stage.Load("/Root/Deferred")
    loaded = stage.GetPrimAtPath("/Root/Deferred/Mesh")
    assert loaded.IsValid()
    uvs = UsdGeom.PrimvarsAPI(loaded).GetPrimvar("st").Get()
    span = max(uv[0] for uv in uvs) - min(uv[0] for uv in uvs)
    assert span == pytest.approx(0.0015, rel=1e-2)
