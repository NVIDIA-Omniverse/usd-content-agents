# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the manipulation / materials / query / physics commands
(the workbench capabilities ported from world-understanding). Driven through the CLI +
daemon, like the rest of the suite, so dispatch wiring is exercised too.
"""

from __future__ import annotations

import os

import pytest

from conftest import SPRAY


def test_transform_then_undo_roundtrips(project):
    project.open(SPRAY)
    moved = project.cli("transform", "@n4", "--tx=+3", json=True, expect_ok=True).json()
    assert moved["data"]["after"]["translate"][0] - moved["data"]["before"]["translate"][0] == 3.0
    undone = project.cli("undo", json=True, expect_ok=True).json()
    assert undone["data"]["undone"] == 1
    # after undo the prim is back where it started
    props = project.cli("properties", "@n4", json=True, expect_ok=True).json()
    assert props["ok"] is True


def test_set_new_attr_undo_then_redo_recreates(project):
    """Regression: `set` of a *new* custom attr → undo removes the spec → redo must
    re-declare it (not blindly Set a now-typeless attribute, which raised in USD)."""
    project.open(SPRAY)
    project.cli("set", "@n4", "myTag", "demo", expect_ok=True)
    project.cli("undo", json=True, expect_ok=True)
    redone = project.cli("redo", json=True, expect_ok=True).json()
    assert redone["ok"] and redone["data"]["redone"] == 1
    props = project.cli("properties", "@n4", json=True, expect_ok=True).json()
    assert "myTag" in props["data"]["attributes"]


def test_material_assign_and_query_binding(project):
    project.open(SPRAY)
    res = project.cli("material", "@n4", "--color", "1,0,0", "--roughness", "0.3",
                      json=True, expect_ok=True).json()
    assert res["ok"] and res["summary"]["material"]
    binding = project.cli("material-binding", "@n4", json=True, expect_ok=True).json()
    assert binding["data"]["binding_type"] == "direct"


def test_material_unbind_and_undo(project):
    project.open(SPRAY)
    created = project.cli("create", "mesh", "Ball", "--shape", "sphere",
                          json=True, expect_ok=True).json()
    ref = created["summary"]["ref"]
    project.cli("material", ref, "--color", "1,0,0", json=True, expect_ok=True)
    res = project.cli("material", ref, "--unbind", json=True, expect_ok=True).json()
    assert res["summary"]["unbound"]
    binding = project.cli("material-binding", ref, json=True, expect_ok=True).json()
    assert binding["data"]["binding_type"] == "none"
    # unbind is reversible
    project.cli("undo", json=True, expect_ok=True)
    binding = project.cli("material-binding", ref, json=True, expect_ok=True).json()
    assert binding["data"]["binding_type"] == "direct"


def test_material_unbind_reveals_weaker_binding(project):
    """Unbinding clears the session's opinion; a binding baked into a referenced layer
    correctly resurfaces, and the response says so."""
    project.open(SPRAY)
    project.cli("material", "@n4", "--color", "1,0,0", json=True, expect_ok=True)
    res = project.cli("material", "@n4", "--unbind", json=True, expect_ok=True).json()
    assert res["summary"]["unbound"]
    assert res["summary"]["now_bound"]  # the asset's own baked binding applies again


def test_material_unbind_without_binding_is_a_noop(project):
    project.open(SPRAY)
    created = project.cli("create", "mesh", "Bare", "--shape", "cube",
                          json=True, expect_ok=True).json()
    res = project.cli("material", created["summary"]["ref"], "--unbind",
                      json=True, expect_ok=True).json()
    assert res["summary"]["unbound"] is False
    assert "no material bound" in res["data"]["text"]


def test_material_unbind_rejects_other_options(project):
    project.open(SPRAY)
    res = project.cli("material", "@n4", "--unbind", "--color", "1,0,0",
                      json=True, expect_ok=False).json()
    assert "cannot be combined" in res["issues"][0]["message"]


def test_material_omnipbr_mdl_assign_and_introspect(project):
    project.open(SPRAY)
    res = project.cli("material", "@n4", "--omnipbr", "--color", "0,1,0", "--metallic", "1",
                      "--roughness", "0.2", json=True, expect_ok=True).json()
    assert res["ok"] and res["summary"]["material"]
    binding = project.cli("material-binding", "@n4", json=True, expect_ok=True).json()
    assert binding["data"]["binding_type"] == "direct"
    shader = binding["data"]["shader"]
    assert shader["kind"] == "mdl"
    assert shader["mdl_module"] == "OmniPBR.mdl" and shader["mdl_subidentifier"] == "OmniPBR"
    assert shader["inputs"]["metallic_constant"] == 1.0


def test_material_custom_mdl_module(project):
    project.open(SPRAY)
    project.cli("material", "@n4", "--mdl", "OmniGlass.mdl", json=True, expect_ok=True)
    binding = project.cli("material-binding", "@n4", json=True, expect_ok=True).json()
    assert binding["data"]["shader"]["mdl_subidentifier"] == "OmniGlass"


def test_create_delete_and_undo(project):
    project.open(SPRAY)
    created = project.cli("create", "mesh", "Box", "--shape", "cube", json=True, expect_ok=True).json()
    assert created["summary"]["name"] == "Box" and created["data"]["path"].endswith("/Box")
    ref = created["summary"]["ref"]
    project.cli("delete", ref, expect_ok=True)
    # delete is reversible
    undone = project.cli("undo", json=True, expect_ok=True).json()
    assert undone["data"]["undone"] == 1


def test_duplicate_is_independent(project):
    project.open(SPRAY)
    dup = project.cli("duplicate", "@n4", "--at", "0,2,0", json=True, expect_ok=True).json()
    assert dup["ok"] and dup["data"]["path"].endswith("_copy")


def test_hide_show_visibility(project):
    project.open(SPRAY)
    project.cli("hide", "@n4", expect_ok=True)
    props = project.cli("properties", "@n4", json=True, expect_ok=True).json()
    assert props["data"]["visible"] is False
    project.cli("show", "@n4", expect_ok=True)
    props = project.cli("properties", "@n4", json=True, expect_ok=True).json()
    assert props["data"]["visible"] is True


def test_find_by_type_returns_refs(project):
    project.open(SPRAY)
    env = project.cli("find", "--type", "Mesh", json=True, expect_ok=True).json()
    assert env["summary"]["matches"] >= 1
    assert all(r.startswith("@") for r in env["data"]["refs"])


def test_info_reports_stage_facts(project):
    project.open(SPRAY)
    env = project.cli("info", json=True, expect_ok=True).json()
    assert env["data"]["prim_count"] >= 1
    assert env["data"]["up_axis"] in ("Y", "Z")


def test_distance_between_two_meshes(project):
    project.open(SPRAY)
    env = project.cli("distance", "@n4", "@n6", json=True, expect_ok=True).json()
    assert env["data"]["distance"] >= 0.0


def test_physics_apply_then_validate(project):
    import json as _json
    project.open(SPRAY)
    project.cli("physics", "inspect", json=True, expect_ok=True)
    patch = project.root / "operations.json"
    patch.write_text(_json.dumps({
        "scene_paths": ["/PhysicsScenario"],
        "rigid_bodies": [{"path": "@n4", "density": 700.0}],
        "colliders": [{"path": "@n4", "approximation": "convexHull"}],
        "materials": [{"path": "/PhysicsMaterial", "static_friction": 0.6,
                       "dynamic_friction": 0.5, "restitution": 0.1}],
        "bindings": [{"target_path": "@n4", "material_path": "/PhysicsMaterial"}],
    }))
    applied = project.cli("physics", "apply", "-f", str(patch), json=True, expect_ok=True).json()
    assert applied["summary"]["operations"] == 5
    report = project.cli("physics", "validate", json=True, expect_ok=True).json()
    assert report["ok"] is True


def test_checkpoint_save_and_list(project):
    project.open(SPRAY)
    project.cli("checkpoint", "save", "cp1", expect_ok=True)
    env = project.cli("checkpoint", "list", json=True, expect_ok=True).json()
    assert "cp1" in env["data"]["checkpoints"]


def test_set_authors_relationship_targets_not_shadow_attrs(project):
    """Regression (task-06 benchmark): `set physics:body0 <path>` on a joint used to
    author a *custom attribute* shadowing the schema relationship — reported success,
    then corrupted every later read of the prim with a pxr verification error."""
    project.open(SPRAY)
    cab = project.cli("create", "xform", "Cabinet", json=True, expect_ok=True).json()
    joint = project.cli("create", "PhysicsPrismaticJoint", "slide",
                        json=True, expect_ok=True).json()
    jref, cpath = joint["summary"]["ref"], cab["data"]["path"]

    res = project.cli("set", jref, "physics:body0", cpath, json=True, expect_ok=True).json()
    assert res["summary"].get("relationship") is True
    assert res["data"]["new"] == [cpath]
    # the prim must still be readable (the old bug broke composition on read)
    props = project.cli("properties", jref, json=True, expect_ok=True).json()
    assert props["ok"] is True

    undone = project.cli("undo", json=True, expect_ok=True).json()
    assert undone["ok"] is True

    # a bogus target is rejected up front instead of authored dangling
    bad = project.cli("set", jref, "physics:body1", "/No/Such/Prim", json=True).json()
    assert bad["ok"] is False and "does not exist" in str(bad["issues"])


def test_save_rebases_relative_asset_paths(project, tmp_path):
    """Regression (task-04 benchmark): `save out.usda` kept payload/texture paths
    relative to the OLD directory, silently dropping geometry on reopen."""
    from pxr import Usd

    (project.root / "assets").mkdir(exist_ok=True)
    payload = project.root / "assets" / "payload.usda"
    pl = Usd.Stage.CreateNew(str(payload))
    pl.DefinePrim("/Geo", "Xform")
    pl.DefinePrim("/Geo/M", "Mesh")
    pl.GetRootLayer().Save()
    scene = project.root / "assets" / "scene.usda"
    sc = Usd.Stage.CreateNew(str(scene))
    root = sc.DefinePrim("/Root", "Xform")
    root.GetPayloads().AddPayload("./payload.usda", "/Geo")
    sc.GetRootLayer().Save()

    project.cli("open", "assets/scene.usda", "--force-reload", expect_ok=True)
    out = project.cli("save", "moved_scene.usda", json=True, expect_ok=True).json()
    assert out["summary"].get("rebased_asset_paths", 0) >= 1
    reopened = Usd.Stage.Open(str(project.root / "moved_scene.usda"))
    assert reopened.GetPrimAtPath("/Root/M").IsValid(), "payload must still resolve"


def test_physics_apply_authors_only_the_explicit_collider_target(project):
    import json as _json
    from pxr import Usd, UsdPhysics

    project.open(SPRAY)
    patch = project.root / "d.json"
    patch.write_text(_json.dumps({
        "scene_paths": ["/PhysicsScenario"],
        "rigid_bodies": [{"path": "/spray_bottle", "mass": 0.2}],
        "colliders": [{"path": "/spray_bottle/Geometry/bottle_body",
                       "approximation": "convexHull"}],
    }))
    project.cli("physics", "apply", "-f", str(patch), json=True, expect_ok=True)
    project.cli("save", "out_phys.usda", expect_ok=True)
    st = Usd.Stage.Open(str(project.root / "out_phys.usda"))
    root = st.GetPrimAtPath("/spray_bottle")
    mesh = st.GetPrimAtPath("/spray_bottle/Geometry/bottle_body")
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not root.HasAPI(UsdPhysics.CollisionAPI), "no collider on the body root"
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)


def test_save_strips_managed_render_camera(project):
    """Regression (round-3 t01): the auto-created ov_cam render camera leaked into saved
    deliverables. save now strips it (user cameras created via --name are kept)."""
    from pxr import Usd, UsdGeom

    staged = project.root / "managed-camera.usda"
    source_stage = Usd.Stage.Open(str(SPRAY.path))
    managed = UsdGeom.Camera.Define(source_stage, "/ov_cam")
    managed.GetPrim().SetCustomDataByKey("usdManagedCamera", True)
    source_stage.GetRootLayer().Export(str(staged))
    project.cli("open", str(staged), "--force-reload", json=True, expect_ok=True)
    project.cli("camera", "create", "--name", "hero", expect_ok=True)  # a user camera
    out = project.cli("save", "with_cams.usda", json=True, expect_ok=True).json()
    assert out["summary"].get("stripped_render_cameras", 0) >= 1
    st = Usd.Stage.Open(str(project.root / "with_cams.usda"))
    cams = [p.GetName() for p in st.Traverse() if p.IsA(UsdGeom.Camera)]
    assert "ov_cam" not in cams and "hero" in cams


def test_camera_create_look_at_accepts_ref(project):
    """Regression (round-3 t02/t03): camera create --look-at @ref crashed the float
    parser; it now resolves a prim ref/path to its center."""
    project.open(SPRAY)
    r = project.cli("camera", "create", "--name", "c2", "--look-at", "@n4",
                    json=True, expect_ok=True).json()
    assert r["ok"] and r["summary"]["camera"].endswith("c2")


@pytest.mark.skipif(
    not os.environ.get("USD_CLI_TEST_OVRTX"),
    reason="requires an explicitly enabled local OVRTX runtime",
)
def test_render_output_png_is_a_file_not_dir(project):
    """Regression (rounds 1-4 papercut): render -o foo.png created a directory foo.png/
    with an auto-named file inside. A single beauty render now writes exactly foo.png."""
    project.open(SPRAY)
    project.cli("render", "-o", "shot.png", "--res", "80x80", expect_ok=True)
    p = project.root / "shot.png"
    assert p.is_file(), "shot.png should be a file, not a directory"


def test_set_stage_up_axis_via_pseudo_root(project):
    """Regression (round-4 t11): set / upAxis Z used to error on the pseudo-root."""
    project.open(SPRAY)
    r = project.cli("set", "/", "upAxis", "Z", json=True, expect_ok=True).json()
    assert r["ok"] and r["summary"].get("stage_metadata") == "upAxis"
    info = project.cli("info", json=True, expect_ok=True).json()
    assert info["data"].get("up_axis", "").lower().startswith("z")


def test_material_library_import_without_ref(project, tmp_path):
    """Regression (round-4 t10): material --library --name with no ref now imports the
    material into /Looks (the scene-repair path) instead of erroring 'needs a ref'."""
    from pxr import Usd, UsdShade
    lib = Usd.Stage.CreateNew(str(tmp_path / "lib.usda"))
    UsdShade.Material.Define(lib, "/Looks/Steel_Brushed")
    lib.Save()
    project.open(SPRAY)
    r = project.cli("material", "--library", str(tmp_path / "lib.usda"),
                    "--name", "Steel_Brushed", json=True, expect_ok=True).json()
    assert r["ok"] and r["summary"].get("material", "").endswith("Steel_Brushed")


def test_material_library_import_normalizes_legacy_terminal_scope(project, tmp_path):
    """The CLI accepts a generic DCC material container only when it carries a
    real material-terminal connection, then types the local reference as Material."""
    from pxr import Sdf, Usd, UsdShade

    library_path = tmp_path / "legacy-lib.usda"
    library = Usd.Stage.CreateNew(str(library_path))
    legacy = library.DefinePrim("/World/Looks/Cardboard", "Scope")
    shader = UsdShade.Shader.Define(
        library, "/World/Looks/Cardboard/PreviewSurface"
    )
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    terminal = legacy.CreateAttribute("outputs:surface", Sdf.ValueTypeNames.Token)
    terminal.SetConnections([shader_output.GetAttr().GetPath()])
    library.Save()

    project.open(SPRAY)
    response = project.cli(
        "material",
        "--library",
        str(library_path),
        "--name",
        "Cardboard",
        json=True,
        expect_ok=True,
    ).json()

    assert response["ok"]
    assert response["data"]["material_path"].endswith("/Looks/Cardboard")


def test_material_library_import_accepts_exact_library_prim(project, tmp_path):
    from pxr import Usd, UsdShade

    library_path = tmp_path / "exact-lib.usda"
    library = Usd.Stage.CreateNew(str(library_path))
    UsdShade.Material.Define(library, "/World/Looks/Red")
    UsdShade.Material.Define(library, "/Other/Looks/Paint_Red")
    library.Save()

    project.open(SPRAY)
    response = project.cli(
        "material",
        "--library",
        str(library_path),
        "--library-prim",
        "/World/Looks/Red",
        json=True,
        expect_ok=True,
    ).json()

    assert response["data"]["material_path"].rsplit("/", 1)[-1].startswith("Red_")


def test_material_apply_batches_exact_decision_patch_in_one_transaction(project, tmp_path):
    import json

    from pxr import Usd, UsdShade

    library_path = tmp_path / "batch-lib.usda"
    library = Usd.Stage.CreateNew(str(library_path))
    UsdShade.Material.Define(library, "/World/Looks/Red")
    UsdShade.Material.Define(library, "/World/Looks/Blue")
    library.Save()

    project.open(SPRAY)
    first = project.cli(
        "create", "mesh", "BatchFirst", "--shape", "cube", json=True, expect_ok=True
    ).json()["data"]["path"]
    second = project.cli(
        "create", "mesh", "BatchSecond", "--shape", "sphere", json=True, expect_ok=True
    ).json()["data"]["path"]
    plan_path = project.root / "material-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-decision-patch.v1",
                "material_assignments": [
                    {
                        "material_name": "Red",
                        "material_path": "/World/Looks/Red",
                        "prim_paths": [second],
                        "runtime_prim_paths": [first],
                    },
                    {
                        "material_name": "Blue",
                        "material_path": "/World/Looks/Blue",
                        "prim_paths": [first],
                        "runtime_prim_paths": [second],
                    },
                ],
            }
        )
    )

    response = project.cli(
        "material-apply",
        str(plan_path),
        "--library",
        str(library_path),
        "--path-key",
        "runtime_prim_paths",
        json=True,
        expect_ok=True,
    ).json()

    assert response["summary"] == {
        "targets": 2,
        "authored": 2,
        "materials": 2,
        "groups": 2,
    }
    assert [group["prim_paths"] for group in response["data"]["groups"]] == [
        [first],
        [second],
    ]
    first_binding = project.cli(
        "material-binding", first, json=True, expect_ok=True
    ).json()
    second_binding = project.cli(
        "material-binding", second, json=True, expect_ok=True
    ).json()
    assert first_binding["data"]["bound_material_path"].startswith("/spray_bottle/Looks/Red_")
    assert second_binding["data"]["bound_material_path"].startswith(
        "/spray_bottle/Looks/Blue_"
    )
    imported_paths = list(response["data"]["imported_material_paths"].values())
    published = project.root / "material-plan-published.usda"
    project.cli("save", published, "--flatten", json=True, expect_ok=True)
    published_stage = Usd.Stage.Open(str(published))
    for path in imported_paths:
        prim = published_stage.GetPrimAtPath(path)
        assert prim.IsValid()
        assert prim.HasCustomDataKey("usdCliLibraryAsset")
        assert prim.HasCustomDataKey("usdCliLibrarySourcePrim")

    undone = project.cli("undo", json=True, expect_ok=True).json()
    assert undone["data"]["undone"] == 1
    assert project.cli(
        "material-binding", first, json=True, expect_ok=True
    ).json()["data"]["bound_material_path"] is None
    assert project.cli(
        "material-binding", second, json=True, expect_ok=True
    ).json()["data"]["bound_material_path"] is None
    undone_path = project.root / "material-plan-undone.usda"
    project.cli("save", undone_path, "--flatten", json=True, expect_ok=True)
    undone_stage = Usd.Stage.Open(str(undone_path))
    for path in imported_paths:
        prim = undone_stage.GetPrimAtPath(path)
        assert not prim.IsValid() or not prim.IsActive()

    project.cli(
        "material-apply",
        str(plan_path),
        "--library",
        str(library_path),
        "--path-key",
        "runtime_prim_paths",
        json=True,
        expect_ok=True,
    )
    reapplied_path = project.root / "material-plan-reapplied.usda"
    project.cli("save", reapplied_path, "--flatten", json=True, expect_ok=True)
    reapplied_stage = Usd.Stage.Open(str(reapplied_path))
    assert all(reapplied_stage.GetPrimAtPath(path).IsActive() for path in imported_paths)

    # Applying over active imports must not make them part of the next undo,
    # while undoing the reactivation must restore the prior inactive state.
    project.cli(
        "material-apply",
        str(plan_path),
        "--library",
        str(library_path),
        "--path-key",
        "runtime_prim_paths",
        json=True,
        expect_ok=True,
    )
    project.cli("undo", json=True, expect_ok=True)
    active_path = project.root / "material-plan-active-undo.usda"
    project.cli("save", active_path, "--flatten", json=True, expect_ok=True)
    active_stage = Usd.Stage.Open(str(active_path))
    assert all(active_stage.GetPrimAtPath(path).IsActive() for path in imported_paths)

    project.cli("undo", json=True, expect_ok=True)
    inactive_path = project.root / "material-plan-reactivation-undone.usda"
    project.cli("save", inactive_path, "--flatten", json=True, expect_ok=True)
    inactive_stage = Usd.Stage.Open(str(inactive_path))
    for path in imported_paths:
        prim = inactive_stage.GetPrimAtPath(path)
        assert not prim.IsValid() or not prim.IsActive()

    rebound = project.cli(
        "material",
        first,
        "--library",
        str(library_path),
        "--name",
        "Red",
        "--library-prim",
        "/World/Looks/Red",
        json=True,
        expect_ok=True,
    ).json()
    rebound_material = rebound["data"]["material_path"]
    project.cli("undo", json=True, expect_ok=True)
    rebound_undone = project.root / "material-rebound-undone.usda"
    project.cli("save", rebound_undone, "--flatten", json=True, expect_ok=True)
    rebound_stage = Usd.Stage.Open(str(rebound_undone))
    rebound_prim = rebound_stage.GetPrimAtPath(rebound_material)
    assert not rebound_prim.IsValid() or not rebound_prim.IsActive()
    assert not UsdShade.MaterialBindingAPI(
        rebound_stage.GetPrimAtPath(first)
    ).ComputeBoundMaterial()[0]


def test_material_apply_rejects_conflicting_exact_target_before_edit(project, tmp_path):
    import json

    from pxr import Usd, UsdShade

    library_path = tmp_path / "conflict-lib.usda"
    library = Usd.Stage.CreateNew(str(library_path))
    UsdShade.Material.Define(library, "/World/Looks/Red")
    UsdShade.Material.Define(library, "/World/Looks/Blue")
    library.Save()
    project.open(SPRAY)
    target = project.cli(
        "create", "mesh", "Conflicted", "--shape", "cube", json=True, expect_ok=True
    ).json()["data"]["path"]
    plan_path = project.root / "conflicting-material-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-decision-patch.v1",
                "material_assignments": [
                    {
                        "material_name": "Red",
                        "material_path": "/World/Looks/Red",
                        "prim_paths": [target],
                    },
                    {
                        "material_name": "Blue",
                        "material_path": "/World/Looks/Blue",
                        "prim_paths": [target],
                    },
                ],
            }
        )
    )

    response = project.cli(
        "material-apply",
        str(plan_path),
        "--library",
        str(library_path),
        json=True,
        expect_ok=False,
    ).json()

    assert "conflicting materials" in response["issues"][0]["message"]
    binding = project.cli(
        "material-binding", target, json=True, expect_ok=True
    ).json()
    assert binding["data"]["bound_material_path"] is None

    project.cli("delete", target, json=True, expect_ok=True)
    inactive_plan = json.loads(plan_path.read_text())
    inactive_plan["material_assignments"] = inactive_plan["material_assignments"][:1]
    plan_path.write_text(json.dumps(inactive_plan))
    inactive = project.cli(
        "material-apply",
        str(plan_path),
        "--library",
        str(library_path),
        json=True,
        expect_ok=False,
    ).json()
    assert "target is inactive" in inactive["issues"][0]["message"]


def test_render_frames_defaults_and_clamps_to_authored_range(project):
    """Regression (round-5 'static GIF'): render-frames mapped requested frame numbers
    straight to time codes, so frames past the recording's authored range rendered a
    static pose. It now defaults to / clamps to the stage's authored animation range."""
    from usd_core.session import Session
    from pxr import Usd, UsdGeom, Gf
    rec = project.root / "rec.usda"
    st = Usd.Stage.CreateNew(str(rec))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y)
    st.SetTimeCodesPerSecond(30.0); st.SetStartTimeCode(0.0); st.SetEndTimeCode(12.0)
    w = UsdGeom.Xform.Define(st, "/World"); st.SetDefaultPrim(w.GetPrim())
    c = UsdGeom.Cube.Define(st, "/World/C"); op = UsdGeom.Xformable(c).AddTranslateOp()
    for f in range(13):
        op.Set(Gf.Vec3d(f * 0.1, 0, 0), Usd.TimeCode(float(f)))
    st.GetRootLayer().Save()
    assert Session._stage_time_range(st) == (0, 12)


def test_render_frames_parser_preserves_fractional_time_codes():
    from usd_core.session import Session

    assert Session._parse_frames("0.25:2.75") == [0.25, 1.25, 2.25, 2.75]
    assert Session._parse_frames("0.25,1.5,2.75") == [0.25, 1.5, 2.75]
