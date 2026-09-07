# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-level tests for the material workflow additions: bulk rule-based binding,
prototype-aware authoring, subset targeting, audit/validate/subsets commands, and
material inspection at instance refs. Direct Session tests (need pxr), no daemon.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")


@pytest.fixture()
def session():
    from pxr import UsdGeom

    from usd_core.config import Config
    from usd_core.session import Session

    s = Session(Config())
    s.new()
    st = s._stage
    for nm in ("Conductor_01", "Conductor_02", "Board"):
        mesh = UsdGeom.Mesh.Define(st, f"/World/{nm}")
        mesh.GetFaceVertexCountsAttr().Set([3, 3])
    UsdGeom.Xform.Define(st, "/World/Proto")
    UsdGeom.Mesh.Define(st, "/World/Proto/Chip")
    for nm in ("U1", "U2"):
        p = st.DefinePrim(f"/World/{nm}")
        p.GetReferences().AddInternalReference("/World/Proto")
        p.SetInstanceable(True)
    s._index_prims()
    return s


def _bound(session, path):
    from usd_core.materials import bound_material
    return bound_material(session._stage, path)["bound_material_path"]


def test_bulk_bind_by_rule(session):
    r = session.material(color=[0.9, 0.5, 0.1], metallic=1.0, name="Copper",
                         all=True, type=["Mesh"], where=["name~=Conductor*"])
    assert r.ok, r.issues
    assert r.summary["targets"] == 2
    assert _bound(session, "/World/Conductor_01") == "/World/Looks/Copper"
    assert _bound(session, "/World/Conductor_02") == "/World/Looks/Copper"
    assert _bound(session, "/World/Board") is None  # rule excluded it

    # one undo drops the whole bulk transaction
    assert session.undo(1).ok
    assert _bound(session, "/World/Conductor_01") is None
    assert session.redo(1).ok
    assert _bound(session, "/World/Conductor_01") == "/World/Looks/Copper"


def test_bulk_and_ref_are_mutually_exclusive(session):
    r = session.material(ref="@n2", all=True, color=[1, 0, 0])
    assert not r.ok and "not both" in r.issues[0].message


def test_bulk_bind_collapses_instances_to_one_authoring(session):
    # both proxies of both instances match; with --prototype they collapse to ONE
    # binding on the prototype source
    r = session.material(color=[0.1, 0.1, 0.1], name="Epoxy", all=True,
                         where=["name==Chip"], prototype=True)
    assert r.ok, r.issues
    assert r.summary["targets"] == 1 and r.summary["matched"] >= 2
    assert r.data["bound_paths"] == ["/World/Proto/Chip"]
    assert _bound(session, "/World/U1/Chip") == "/World/Looks/Epoxy"
    assert _bound(session, "/World/U2/Chip") == "/World/Looks/Epoxy"


def test_single_ref_prototype_bind(session):
    chip = session.refs.ref_for_path("/World/U1/Chip")
    r = session.material(ref=chip, color=[0, 0, 1], name="Blue", prototype=True)
    assert r.ok and "prototype source" in r.data["text"]
    assert _bound(session, "/World/U2/Chip") == "/World/Looks/Blue"


def test_subset_targeting_and_subsets_command(session):
    from pxr import UsdGeom

    board = session._stage.GetPrimAtPath("/World/Board")
    UsdGeom.Subset.CreateGeomSubset(UsdGeom.Imageable(board), "Pads",
                                    UsdGeom.Tokens.face, [0])
    ref = session.refs.ref_for_path("/World/Board")

    r = session.material(ref=ref, subset="Pads", color=[0.9, 0.8, 0.2], name="Gold")
    assert r.ok and r.data["bound_paths"] == ["/World/Board/Pads"]

    r = session.subsets(ref=ref)
    assert r.ok and r.summary["subsets"] == 1 and r.summary["problems"] == 0
    (info,) = r.data["subsets"]
    assert info["family"] == "materialBind" and info["bound_material_path"].endswith("Gold")
    assert "faces 1/2" in r.data["text"]

    r = session.material(ref=ref, subset="Nope", color=[1, 0, 0])
    assert not r.ok and "no GeomSubset 'Nope'" in r.issues[0].message


def test_bulk_unbind(session):
    session.material(color=[1, 0, 0], name="Red", all=True, where=["name~=Conductor*"])
    r = session.material(all=True, where=["name~=Conductor*"], unbind=True)
    assert r.ok and r.summary["unbound"] == 2
    assert _bound(session, "/World/Conductor_01") is None


def test_material_audit_command(session):
    session.material(color=[1, 0, 0], name="Red", all=True, where=["name~=Conductor*"])
    r = session.material_audit(effective=True, include_subsets=True)
    assert r.ok
    assert r.summary["unbound"] >= 1  # Board and the chips are unbound
    assert "unbound" in r.data["text"]
    assert any(row["binding"] == "direct" for row in r.data["renderables"])


def test_validate_command_clean_and_dirty(session):
    r = session.validate_stage()
    assert r.ok and "clean" in r.data["text"]

    from pxr import Sdf, UsdShade
    chip = session._stage.GetPrimAtPath("/World/Board")
    UsdShade.MaterialBindingAPI.Apply(chip)
    chip.CreateRelationship("material:binding").SetTargets([Sdf.Path("/World/Nope")])
    r = session.validate_stage()
    assert not r.ok and r.summary["errors"] >= 1

    r = session.validate_stage(fix=True)
    assert r.summary["fixed"] == 1
    assert session.validate_stage().ok


def test_instance_material_inspection(session):
    session.material(color=[0.2, 0.2, 0.2], name="Epoxy", all=True,
                     where=["name==Chip"], prototype=True)
    u1 = session.refs.ref_for_path("/World/U1")

    snap = session.snapshot(materials=True)
    assert "mats={Epoxy×1}" in snap.data["text"]  # collapsed instance, materials visible

    desc = session.describe(u1).data["text"]
    assert "instance containing 1 gprims" in desc and "Epoxy" in desc

    mb = session.material_binding(u1)
    assert mb.data["materials_within"]["materials"] == {"/World/Looks/Epoxy": 1}
    assert "within instance: Epoxy×1" in mb.data["text"]


def test_material_binding_inspects_a_material_ref(session):
    session.material(ref=session.refs.ref_for_path("/World/Board"),
                     mdl="OmniPBR.mdl", color=[0.5, 0.5, 0.5], name="Steel")
    session._index_prims()
    mref = session.refs.ref_for_path("/World/Looks/Steel")
    assert mref
    r = session.material_binding(mref)
    assert r.ok and r.summary["shader"] == "mdl"
    assert "OmniPBR.mdl" in r.data["text"]


def test_shader_input_parsing(session):
    ref = session.refs.ref_for_path("/World/Board")
    r = session.material(ref=ref, mdl="OmniGlass.mdl", name="Glass",
                         inputs=["glass_ior=1.5", "thin_walled=true"])
    assert r.ok, r.issues
    from usd_core.materials import describe_material
    d = describe_material(session._stage, "/World/Looks/Glass")
    assert d["inputs"]["glass_ior"] == 1.5 and d["inputs"]["thin_walled"] is True

    r = session.material(ref=ref, inputs=["oops"])
    assert not r.ok and "name=value" in r.issues[0].message
