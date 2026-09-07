# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native USD instancing: instanced geometry must be *seen*, not silently skipped.

A flattened-but-instanced asset (internal `instanceable` references to prototype subtrees,
e.g. exported PCBs with thousands of repeated components) composes its real geometry behind
instance proxies, which `stage.Traverse()` deliberately skips. Before the fix, such a file
opened "successfully" but reported zero meshes, `find -t Mesh` was empty, and every analytic
AOV rendered blank.

Two layers, like test_aov.py:
  * Unit (imports usd_core): stage_info / find_prims / render_aovs see proxy geometry.
  * CLI (project fixture): info / find / snapshot / spatial / edit-rejection end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

INSTANCED_USDA = """\
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "comp_1" (
        instanceable = true
        references = </Prototypes/Widget>
    )
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Xform "comp_2" (
        instanceable = true
        references = </Prototypes/Widget>
    )
    {
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Xform "comp_3" (
        instanceable = true
        references = </Prototypes/Widget>
    )
    {
        double3 xformOp:translate = (6, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}

def Scope "Prototypes"
{
    def Xform "Widget"
    {
        def Mesh "body"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
            float3[] extent = [(0, 0, 0), (1, 1, 0)]
        }

        def Mesh "cap"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
            float3[] extent = [(0, 0, 1), (1, 1, 1)]
        }
    }
}
"""

# 3 instances × 2 proxy meshes + the 2 prototype-source meshes under /Prototypes
EXPECTED_MESHES = 8


@pytest.fixture(scope="module")
def instanced_usda(tmp_path_factory):
    path = tmp_path_factory.mktemp("instanced_asset") / "instanced.usda"
    path.write_text(INSTANCED_USDA)
    return path


def _open_stage(path):
    from pxr import Usd

    return Usd.Stage.Open(str(path))


# ── unit layer ────────────────────────────────────────────────────────────────────
def test_stage_info_counts_instanced_meshes(instanced_usda):
    from usd_core import query

    info = query.stage_info(_open_stage(instanced_usda), str(instanced_usda))
    assert info["mesh_count"] == EXPECTED_MESHES
    assert info["instance_count"] == 3
    assert info["instance_proxy_prim_count"] == 6  # body + cap per instance


def test_find_prims_sees_instance_proxies(instanced_usda):
    from usd_core import query

    hits = query.find_prims(_open_stage(instanced_usda), type_="Mesh")
    assert len(hits) == EXPECTED_MESHES
    proxies = [h for h in hits if h.get("instance_proxy")]
    assert len(proxies) == 6
    assert any(h["path"] == "/World/comp_2/body" for h in proxies)


def test_set_instanceable_false_exposes_external_reference_parts_for_authoring(
    project,
    tmp_path,
):
    external = tmp_path / "external_parts.usda"
    external.write_text(
        """#usda 1.0
def Xform "Template"
{
    def Mesh "Body" {}
    def Mesh "Trim" {}
}
"""
    )
    source = tmp_path / "external_instance.usda"
    source.write_text(
        f"""#usda 1.0
def Xform "ExternalCopy" (
    instanceable = true
    references = @{external}@</Template>
) {{}}
"""
    )
    project.cli("open", source, "--force-reload", json=True, expect_ok=True)

    changed = project.cli(
        "set",
        "/ExternalCopy",
        "instanceable",
        "false",
        json=True,
        expect_ok=True,
    ).json()
    assert changed["data"] == {"old": True, "new": False}
    project.cli(
        "material",
        "/ExternalCopy/Body",
        "--color",
        "1,0,0",
        "--name",
        "BodyMaterial",
        json=True,
        expect_ok=True,
    )
    project.cli(
        "material",
        "/ExternalCopy/Trim",
        "--color",
        "0,0,1",
        "--name",
        "TrimMaterial",
        json=True,
        expect_ok=True,
    )
    body = project.cli(
        "material-binding", "/ExternalCopy/Body", json=True, expect_ok=True
    ).json()
    trim = project.cli(
        "material-binding", "/ExternalCopy/Trim", json=True, expect_ok=True
    ).json()
    assert body["data"]["binding_type"] == "direct"
    assert trim["data"]["binding_type"] == "direct"
    assert body["data"]["bound_material_path"] != trim["data"]["bound_material_path"]


def test_aovs_rasterize_instanced_geometry(instanced_usda):
    from usd_core import raster
    from usd_core.camera import author_camera, fit_distance, orbit_position

    stage = _open_stage(instanced_usda)
    cam = author_camera(stage, "/ov_cam",
                        orbit_position((3.5, 0.5, 0.5), fit_distance([8, 2, 2]), 30, 20),
                        (3.5, 0.5, 0.5))
    out = raster.render_aovs(stage, cam.GetPrim().GetPath().pathString, 160, 120,
                             ["depth", "segmentation"], lambda p: p)
    depth = np.asarray(out["depth"])
    assert (depth > 0).sum() > 200, "instanced meshes missing from analytic depth"
    assert any(p.startswith("/World/comp_") for p in out["legend"]), \
        f"no instance-proxy geometry in segmentation legend: {list(out['legend'])[:5]}"


def test_mesh_candidates_see_instance_proxies(instanced_usda):
    from usd_core import query

    cands = query.mesh_candidates(_open_stage(instanced_usda))
    assert len(cands) == EXPECTED_MESHES
    proxies = [c for c in cands if c.get("instance_proxy")]
    assert len(proxies) == 6
    # proxy rows name the editable authoring target for physics apply
    assert {c["instance_root"] for c in proxies} == \
        {"/World/comp_1", "/World/comp_2", "/World/comp_3"}
    assert all(c["bounds"] for c in proxies), "proxy candidates must have world bounds"


# The same instanced layout, but the prototype carries colliders (SimReady-style:
# physics schemas live inside the instanceable payload) and one instance is a rigid body.
INSTANCED_PHYSICS_USDA = INSTANCED_USDA.replace(
    'def Mesh "body"',
    'def Mesh "body" (prepend apiSchemas = ["PhysicsCollisionAPI"])',
).replace(
    'def Xform "comp_1" (',
    'def PhysicsScene "physicsScene"\n    {\n    }\n\n    def Xform "comp_1" (\n'
    '        prepend apiSchemas = ["PhysicsRigidBodyAPI"]',
)


def test_validate_schema_sees_physics_inside_instances(tmp_path):
    from pxr import Usd

    from usd_core import physics

    path = tmp_path / "instanced_physics.usda"
    path.write_text(INSTANCED_PHYSICS_USDA)
    report = physics.validate_schema(Usd.Stage.Open(str(path)))
    # 3 proxy colliders (one per instance) + the prototype-source mesh under /Prototypes
    assert report["checks"]["colliders"] == 4, report
    assert report["checks"]["rigid_bodies"] == 1
    assert report["checks"]["scenes"] == 1
    assert report["ok"], report["issues"]


def test_capture_state_scoped_to_instance_expands_proxies(instanced_usda):
    from usd_core import snapdiff

    stage = _open_stage(instanced_usda)
    whole = snapdiff.capture_state(stage, "/")
    assert "/World/comp_1" in whole
    assert "/World/comp_1/body" not in whole  # compact: instances diff as units
    scoped = snapdiff.capture_state(stage, "/World/comp_1")
    assert "/World/comp_1/body" in scoped  # scoping into the instance opts in
    assert "/World/comp_1/cap" in scoped


# ── CLI layer ─────────────────────────────────────────────────────────────────────
@pytest.fixture()
def opened(project, instanced_usda):
    project.cli("open", instanced_usda, "--force-reload", json=True, expect_ok=True)
    return project


def test_cli_info_reports_instancing(opened):
    env = opened.cli("info", json=True, expect_ok=True).json()
    assert env["data"]["mesh_count"] == EXPECTED_MESHES
    assert env["data"]["instance_count"] == 3


def test_cli_find_meshes_have_refs(opened):
    env = opened.cli("find", "--type", "Mesh", json=True, expect_ok=True).json()
    assert env["summary"]["matches"] == EXPECTED_MESHES
    assert all(r["ref"] for r in env["data"]["results"]), "proxy meshes must be @ref-addressable"


def test_cli_snapshot_collapses_instances_until_scoped(opened):
    env = opened.cli("snapshot", json=True, expect_ok=True).json()
    tree = env["data"]["text"]
    assert env["summary"]["instances"] == 3
    comp_lines = [ln for ln in tree.splitlines() if '"comp_' in ln]
    assert len(comp_lines) == 3 and all("(instance)" in ln for ln in comp_lines)
    # whole-scene view stays compact: no proxy meshes (defaultPrim scope excludes /Prototypes)
    assert "body" not in tree

    comp_ref = comp_lines[0].split()[0]
    scoped = opened.cli("snapshot", comp_ref, json=True, expect_ok=True).json()
    assert scoped["data"]["text"].count("body") == 1  # scoping into the instance expands it
    assert "cap" in scoped["data"]["text"]


def test_cli_spatial_treats_instances_as_objects(opened):
    env = opened.cli("find", "--name", "comp_1", json=True, expect_ok=True).json()
    ref = env["data"]["refs"][0]
    # the two prototype-source meshes near the origin legitimately rank first
    near = opened.cli("nearest", ref, "--count", "3", json=True, expect_ok=True).json()
    names = [r["name"] for r in near["data"]["results"]]
    assert "comp_2" in names, f"instance roots missing from spatial results: {names}"


def test_cli_edit_inside_instance_is_rejected_clearly(opened):
    env = opened.cli("find", "--type", "Mesh", json=True, expect_ok=True).json()
    proxy = next(r for r in env["data"]["results"] if r.get("instance_proxy"))
    res = opened.cli("transform", proxy["ref"], "--tx", "1", json=True, expect_ok=False).json()
    msg = " ".join(i["message"] for i in res["issues"])
    assert "instance proxy" in msg and "instance root" in msg


def test_cli_material_bind_unbind_roundtrip_on_instance_proxy(opened):
    """Binding is the one edit allowed on proxies (redirected to the instanceable root),
    and --unbind reverses it through the same redirect."""
    env = opened.cli("find", "--type", "Mesh", json=True, expect_ok=True).json()
    proxy = next(r for r in env["data"]["results"] if r.get("instance_proxy"))
    opened.cli("material", proxy["ref"], "--color", "1,0,0", json=True, expect_ok=True)
    bound = opened.cli("material-binding", proxy["ref"], json=True, expect_ok=True).json()
    assert bound["data"]["binding_type"] == "inherited"  # authored on the instance root
    res = opened.cli("material", proxy["ref"], "--unbind", json=True, expect_ok=True).json()
    assert res["summary"]["unbound"]
    after = opened.cli("material-binding", proxy["ref"], json=True, expect_ok=True).json()
    assert after["data"]["binding_type"] == "none"


def test_cli_properties_work_on_instance_proxy(opened):
    env = opened.cli("find", "--type", "Mesh", json=True, expect_ok=True).json()
    proxy = next(r for r in env["data"]["results"] if r.get("instance_proxy"))
    props = opened.cli("properties", proxy["ref"], json=True, expect_ok=True).json()
    assert props["data"]["type"] == "Mesh"
    assert props["data"]["bounds"]["size"]


def test_cli_physics_inspect_reports_proxy_candidates(opened):
    env = opened.cli("physics", "inspect", json=True, expect_ok=True).json()
    cands = env["data"]["candidates"]
    assert env["summary"]["candidates"] == EXPECTED_MESHES
    proxies = [c for c in cands if c.get("instance_proxy")]
    assert len(proxies) == 6
    # every proxy candidate points at its editable instance root, by path and by @ref
    assert all(c["instance_root"].startswith("/World/comp_") for c in proxies)
    assert all(c.get("instance_root_ref") for c in proxies)


def test_cli_physics_apply_on_instance_root_then_validate(opened, tmp_path):
    """The workflow the proxy candidates point to: author on the instance root, then
    validate — which must see the authored schemas without de-instancing."""
    import json as jsonlib

    env = opened.cli("physics", "inspect", json=True, expect_ok=True).json()
    proxy = next(c for c in env["data"]["candidates"] if c.get("instance_proxy"))
    patch = tmp_path / "operations.json"
    patch.write_text(jsonlib.dumps({
        "scene_paths": ["/PhysicsScenario"],
        "rigid_bodies": [{"path": proxy["instance_root_ref"], "mass": 1.0}],
        "colliders": [{"path": proxy["instance_root_ref"],
                       "approximation": "boundingCube"}],
    }))
    opened.cli("physics", "apply", "-f", patch, json=True, expect_ok=True)
    report = opened.cli("physics", "validate", json=True, expect_ok=True).json()
    assert report["data"]["checks"]["rigid_bodies"] == 1
    assert report["data"]["checks"]["colliders"] == 1
    assert report["data"]["ok"] is True
