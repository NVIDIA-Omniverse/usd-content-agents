# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for composed instance-proxy UV target scopes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import texture_agent.tasks.prepare_uvs as prepare_uvs_module
from texture_agent.tasks.prepare_uvs import (
    PrepareUVsTask,
    UVPreparationError,
)

_TINY_UVS = Vt.Vec2fArray(
    [
        Gf.Vec2f(0.5, 0.5),
        Gf.Vec2f(0.5015, 0.5),
        Gf.Vec2f(0.5015, 0.5015),
        Gf.Vec2f(0.5, 0.5015),
    ]
)


def _tiny_quad(stage: Usd.Stage, path: str) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    ).Set(_TINY_UVS)


def _class_instance_stage(path: Path, *, instance_names: tuple[str, ...]) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    stage.CreateClassPrim("/Body/Prototypes")
    UsdGeom.Xform.Define(stage, "/Body/Prototypes/Body")
    _tiny_quad(stage, "/Body/Prototypes/Body/Part/Mesh")
    _tiny_quad(stage, "/Body/Unselected/Mesh")
    for name in instance_names:
        instance = UsdGeom.Xform.Define(stage, f"/Body/{name}").GetPrim()
        instance.GetReferences().AddInternalReference("/Body/Prototypes/Body")
        instance.SetInstanceable(True)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Body"))
    assert stage.GetRootLayer().Save()
    return path


def _u_span(stage: Usd.Stage, path: str) -> float:
    values = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).GetPrimvar("st").Get()
    return max(value[0] for value in values) - min(value[0] for value in values)


def test_scoped_prepare_repairs_single_instance_proxy_without_widening_scope(
    tmp_path: Path,
) -> None:
    source = _class_instance_stage(tmp_path / "source.usda", instance_names=("Body",))
    working_dir = tmp_path / "work"
    original_targets = ("/Body/Body/Part",)

    context = {
        "usd_path": str(source),
        "working_dir": str(working_dir),
        "texture_config": {
            "uv_policy": "preserve_or_fix",
            "uv_scope": "target_prims",
            "uv_target_prim_paths": list(original_targets),
        },
    }
    PrepareUVsTask().run(context)
    prepared_path = context["usd_path"]
    actions = context["uv_preparation"]

    assert actions["degenerate_repaired"] == 1
    assert actions["target_prim_paths"] == list(original_targets)
    prepared = Usd.Stage.Open(prepared_path)
    assert prepared is not None
    assert prepared.GetPrimAtPath("/Body/Body").IsInstanceable()
    assert _u_span(prepared, "/Body/Body/Part/Mesh") == pytest.approx(0.95)
    assert _u_span(prepared, "/Body/Unselected/Mesh") == pytest.approx(
        0.0015,
        abs=1e-6,
    )

    report = json.loads(Path(actions["uv_report_path"]).read_text(encoding="utf-8"))
    report_paths = {mesh["prim_path"] for mesh in report["meshes"]}
    assert report_paths == {
        "/Body/Body/Part/Mesh",
        "/Body/Unselected/Mesh",
    }
    assert not any(path.startswith("/Flattened_Prototype_") for path in report_paths)
    assert report["actions"]["target_prim_paths"] == list(original_targets)


def test_scoped_prepare_rejects_one_alias_of_shared_instance_prototype(
    tmp_path: Path,
) -> None:
    source = _class_instance_stage(
        tmp_path / "shared.usda",
        instance_names=("Selected", "Unselected"),
    )

    with pytest.raises(
        UVPreparationError,
        match="shared instance prototype used by 2 composed instances",
    ):
        PrepareUVsTask().run(
            {
                "usd_path": str(source),
                "working_dir": str(tmp_path / "work"),
                "texture_config": {
                    "uv_policy": "preserve_or_fix",
                    "uv_scope": "target_prims",
                    "uv_target_prim_paths": ["/Body/Selected/Part/Mesh"],
                },
            }
        )


def test_scoped_prepare_repairs_root_mesh_instance(tmp_path: Path) -> None:
    source = tmp_path / "root-mesh.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.CreateClassPrim("/Prototypes")
    _tiny_quad(stage, "/Prototypes/Mesh")
    instance = UsdGeom.Mesh.Define(stage, "/Instance").GetPrim()
    instance.GetReferences().AddInternalReference("/Prototypes/Mesh")
    instance.SetInstanceable(True)
    stage.SetDefaultPrim(instance)
    assert stage.GetRootLayer().Save()

    context = {
        "usd_path": str(source),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_policy": "preserve_or_fix",
            "uv_scope": "target_prims",
            "uv_target_prim_paths": ["/Instance"],
        },
    }
    PrepareUVsTask().run(context)

    assert context["uv_preparation"]["degenerate_repaired"] == 1
    prepared = Usd.Stage.Open(context["usd_path"])
    assert prepared.GetPrimAtPath("/Instance").IsInstanceable()
    assert _u_span(prepared, "/Instance") == pytest.approx(0.95)


def test_scene_optimizer_routes_instance_target_through_current_backing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _class_instance_stage(tmp_path / "source.usda", instance_names=("Body",))
    original_target = "/Body/Body/Part/Mesh"
    so_call: dict[str, Any] = {}

    def fake_generate_projection_uvs(
        input_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        so_call.update(kwargs)
        input_stage = Usd.Stage.Open(str(input_path))
        # Model SO returning another flattened generation so its synthetic
        # prototype path differs from the one sent in the request.
        reflattened = input_stage.Flatten(addSourceFileComment=False)
        assert reflattened.Export(str(output_path))
        return {"meshes_with_uvs": 0, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_module,
        "generate_projection_uvs",
        fake_generate_projection_uvs,
    )
    context = {
        "usd_path": str(source),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "generate_missing",
            "uv_scope": "target_prims",
            "uv_target_prim_paths": [original_target],
        },
    }

    PrepareUVsTask().run(context)

    assert len(so_call["paths"]) == 1
    so_path = so_call["paths"][0]
    assert so_path.startswith("/Flattened_Prototype_")
    assert so_path.endswith("/Part/Mesh")
    assert so_path != original_target
    actions = context["uv_preparation"]
    assert actions["backend"] == "scene_optimizer"
    assert actions["degenerate_repaired"] == 1
    assert actions["target_prim_paths"] == [original_target]
    prepared = Usd.Stage.Open(context["usd_path"])
    assert prepared.GetPrimAtPath("/Body/Body").IsInstanceable()
    assert _u_span(prepared, original_target) == pytest.approx(0.95)
    report = json.loads(Path(actions["uv_report_path"]).read_text(encoding="utf-8"))
    assert report["actions"]["target_prim_paths"] == [original_target]
    assert original_target in {mesh["prim_path"] for mesh in report["meshes"]}


def test_scene_optimizer_scoped_copyback_pairs_reordered_meshes_by_composed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "two-meshes.usda"
    source_stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(source_stage, "/Root").GetPrim()
    _tiny_quad(source_stage, "/Root/A")
    _tiny_quad(source_stage, "/Root/B")
    source_stage.SetDefaultPrim(root)
    assert source_stage.GetRootLayer().Save()

    expected_uvs = {
        "/Root/A": Vt.Vec2fArray(
            [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(0.2, 0.0),
                Gf.Vec2f(0.2, 0.2),
                Gf.Vec2f(0.0, 0.2),
            ]
        ),
        "/Root/B": Vt.Vec2fArray(
            [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(0.8, 0.0),
                Gf.Vec2f(0.8, 0.8),
                Gf.Vec2f(0.0, 0.8),
            ]
        ),
    }

    def fake_generate_projection_uvs(
        _input_path: Path,
        output_path: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        optimized = Usd.Stage.CreateNew(str(output_path))
        optimized_root = UsdGeom.Xform.Define(optimized, "/Root").GetPrim()
        for mesh_path in ("/Root/B", "/Root/A"):
            _tiny_quad(optimized, mesh_path)
            UsdGeom.PrimvarsAPI(optimized.GetPrimAtPath(mesh_path)).GetPrimvar(
                "st"
            ).Set(expected_uvs[mesh_path])
        optimized.SetDefaultPrim(optimized_root)
        assert [
            str(prim.GetPath())
            for prim in optimized.Traverse()
            if prim.IsA(UsdGeom.Mesh)
        ] == ["/Root/B", "/Root/A"]
        assert optimized.GetRootLayer().Save()
        return {"meshes_with_uvs": 2, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_module,
        "generate_projection_uvs",
        fake_generate_projection_uvs,
    )
    context = {
        "usd_path": str(source),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "force_projection",
            "uv_scope": "target_prims",
            "uv_target_prim_paths": ["/Root/A", "/Root/B"],
            "uv_repair_degenerate": False,
        },
    }

    PrepareUVsTask().run(context)

    prepared = Usd.Stage.Open(context["usd_path"])
    assert _u_span(prepared, "/Root/A") == pytest.approx(0.2)
    assert _u_span(prepared, "/Root/B") == pytest.approx(0.8)
    assert context["uv_preparation"]["uv_writeback_meshes"] == 2


def test_scene_optimizer_copyback_removes_stale_uv_spec_when_output_has_none() -> None:
    destination = Usd.Stage.CreateInMemory()
    optimized = Usd.Stage.CreateInMemory()
    mesh_path = "/Root/Mesh"
    _tiny_quad(destination, mesh_path)
    destination_st = UsdGeom.PrimvarsAPI(
        destination.GetPrimAtPath(mesh_path)
    ).GetPrimvar("st")
    assert destination_st.SetIndices([0, 1, 2, 3])
    optimized_mesh = UsdGeom.Mesh.Define(optimized, mesh_path)
    optimized_mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    optimized_mesh.CreateFaceVertexCountsAttr([4])
    optimized_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mapping = {mesh_path: mesh_path}

    copied = prepare_uvs_module._copy_scene_optimizer_uvs(
        destination,
        optimized,
        destination_backings_by_composed=mapping,
        optimized_backings_by_composed=mapping,
    )

    destination_prim = destination.GetPrimAtPath(mesh_path)
    assert copied == 0
    assert not destination_prim.HasProperty("primvars:st")
    assert not destination_prim.HasProperty("primvars:st:indices")


def test_scene_optimizer_copyback_removes_stale_indices_only() -> None:
    destination = Usd.Stage.CreateInMemory()
    optimized = Usd.Stage.CreateInMemory()
    mesh_path = "/Root/Mesh"
    _tiny_quad(destination, mesh_path)
    _tiny_quad(optimized, mesh_path)
    destination_st = UsdGeom.PrimvarsAPI(
        destination.GetPrimAtPath(mesh_path)
    ).GetPrimvar("st")
    assert destination_st.SetIndices([0, 1, 2, 3])
    mapping = {mesh_path: mesh_path}

    copied = prepare_uvs_module._copy_scene_optimizer_uvs(
        destination,
        optimized,
        destination_backings_by_composed=mapping,
        optimized_backings_by_composed=mapping,
    )

    destination_prim = destination.GetPrimAtPath(mesh_path)
    assert copied == 1
    assert destination_prim.HasProperty("primvars:st")
    assert not destination_prim.HasProperty("primvars:st:indices")


def test_scene_optimizer_stage_scope_uses_active_reflattened_backing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _class_instance_stage(tmp_path / "source.usda", instance_names=("Body",))
    composed_target = "/Body/Body/Part/Mesh"
    generated_uvs = Vt.Vec2fArray(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )

    def fake_generate_projection_uvs(
        input_path: Path,
        output_path: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        input_stage = Usd.Stage.Open(str(input_path))
        reflattened = input_stage.Flatten(addSourceFileComment=False)
        optimized = Usd.Stage.Open(reflattened)
        active_backing = prepare_uvs_module._active_flattened_mesh_backings(optimized)[
            composed_target
        ]
        UsdGeom.PrimvarsAPI(optimized.GetPrimAtPath(active_backing)).GetPrimvar(
            "st"
        ).Set(generated_uvs)
        assert optimized.GetRootLayer().Export(str(output_path))
        return {"meshes_with_uvs": 2, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_module,
        "generate_projection_uvs",
        fake_generate_projection_uvs,
    )
    context = {
        "usd_path": str(source),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "force_projection",
            "uv_scope": "stage",
            "uv_repair_degenerate": False,
        },
    }

    PrepareUVsTask().run(context)

    prepared = Usd.Stage.Open(context["usd_path"])
    assert _u_span(prepared, composed_target) == pytest.approx(1.0)
    assert _u_span(prepared, "/Body/Unselected/Mesh") == pytest.approx(
        0.0015,
        abs=1e-6,
    )
    assert context["uv_preparation"]["uv_writeback_meshes"] == 2


def test_scene_optimizer_missing_scoped_target_falls_back_to_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _class_instance_stage(tmp_path / "source.usda", instance_names=("Body",))
    original_target = "/Body/Body/Part/Mesh"

    def fake_generate_projection_uvs(
        _input_path: Path,
        output_path: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        invalid_output = Usd.Stage.CreateNew(str(output_path))
        dropped = UsdGeom.Xform.Define(invalid_output, "/Dropped").GetPrim()
        invalid_output.SetDefaultPrim(dropped)
        assert invalid_output.GetRootLayer().Save()
        return {"meshes_with_uvs": 0, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_module,
        "generate_projection_uvs",
        fake_generate_projection_uvs,
    )
    context = {
        "usd_path": str(source),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "generate_missing",
            "uv_scope": "target_prims",
            "uv_target_prim_paths": [original_target],
        },
    }

    PrepareUVsTask().run(context)

    actions = context["uv_preparation"]
    assert actions["backend"] == "python"
    assert actions["fallback_from"]["backend"] == "scene_optimizer"
    assert (
        "could not preserve the scoped target geometry"
        in actions["fallback_from"]["error"]
    )
    assert actions["degenerate_repaired"] == 1
    assert actions["target_prim_paths"] == [original_target]
    prepared = Usd.Stage.Open(context["usd_path"])
    assert prepared.GetPrimAtPath("/Body/Body").IsInstanceable()
    assert _u_span(prepared, original_target) == pytest.approx(0.95)
    assert _u_span(prepared, "/Body/Unselected/Mesh") == pytest.approx(
        0.0015,
        abs=1e-6,
    )
