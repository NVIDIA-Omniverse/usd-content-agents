# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise layer-only service configuration and downloads with real USD files."""

import asyncio
import io
import json
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from material_agent.api.builders import build_unified_pipeline_config
from material_agent.config.unified_config import UnifiedPipelineConfigTask
from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from ...service.routers import artifacts_router, pipeline_router
from ...service.session.manager import SessionManager


def _create_input(path: Path, sample: str) -> str:
    if sample == "ladder":
        source = (
            Path(__file__).resolve().parents[3]
            / "material_agent/data/examples/ladder/sources/usd/ladder.usd"
        )
        shutil.copyfile(source, path)
    else:
        stage = Usd.Stage.CreateNew(str(path))
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        if sample == "cube":
            UsdGeom.Cube.Define(stage, "/Asset/Body")
        else:
            UsdGeom.Xform.Define(stage, "/Asset/Assembly")
            UsdGeom.Sphere.Define(stage, "/Asset/Assembly/Part")
        stage.GetRootLayer().Save()

    stage = Usd.Stage.Open(str(path))
    return str(
        next(prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdGeom.Gprim))
    )


def _create_library(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/TestMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.4, 0.6)
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    stage.GetRootLayer().Save()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("sample", ["cube", "nested_sphere", "ladder"])
async def test_layer_only_apply_and_download_with_real_usd(
    tmp_path: Path, sample: str
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    artifacts_router.set_session_manager(manager)

    input_usd = session_dir / "input" / "scene.usd"
    input_usd.parent.mkdir(parents=True, exist_ok=True)
    target_path = _create_input(input_usd, sample)
    original_bytes = input_usd.read_bytes()

    library_usd = session_dir / "library.usd"
    _create_library(library_usd)
    output_usd = session_dir / "output" / "scene_with_materials.usd"
    config = build_unified_pipeline_config(
        project_name=sample,
        session_id=session_id,
        working_dir=str(session_dir / "cache"),
        input_usd_path=str(input_usd),
        output_usd_path=str(output_usd),
        materials_library_path=str(library_usd),
        materials_entries=[
            {
                "name": "TestMaterial",
                "description": "A test preview material",
                "binding": "/World/Looks/TestMaterial",
            }
        ],
        enabled_steps=["apply"],
    )
    pipeline_router._configure_apply_step(
        config, layer_only=True, request_context="Test pipeline"
    )
    loaded = UnifiedPipelineConfigTask().run({"config_dict": config})
    apply_config = loaded["step_configs"]["apply"]
    assert apply_config["layer_only"] is True
    assert apply_config["flatten_output"] is False

    predictions = session_dir / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"id": target_path, "materials": {"material": "TestMaterial"}})
        + "\n",
        encoding="utf-8",
    )
    apply_config.update(
        {
            "predictions_path": str(predictions),
            "resolved_materials": {"TestMaterial": "/World/Looks/TestMaterial"},
            "is_library_based_mapping": True,
            "material_library_path": str(library_usd),
        }
    )
    ApplyMaterialsToUSDTask().run(apply_config)

    assert input_usd.read_bytes() == original_bytes
    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert str(input_usd) in layer.subLayerPaths
    target_spec = layer.GetPrimAtPath(target_path)
    assert target_spec is not None
    assert target_spec.specifier == Sdf.SpecifierOver
    assert "material:binding" in target_spec.relationships
    composed = Usd.Stage.Open(str(output_usd))
    assert composed.GetPrimAtPath(target_path).IsValid()
    material, _ = UsdShade.MaterialBindingAPI(
        composed.GetPrimAtPath(target_path)
    ).ComputeBoundMaterial()
    assert material

    flat_usd = output_usd.with_name("scene_with_materials_flat.usd")
    composed.Flatten().Export(str(flat_usd))
    app = FastAPI()
    app.include_router(artifacts_router.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        rendered_response = await client.get(f"/artifacts/{session_id}/output")
        applied_response = await client.get(f"/artifacts/{session_id}/output-applied")
    assert rendered_response.status_code == 200
    assert applied_response.status_code == 200
    assert rendered_response.content == flat_usd.read_bytes()
    assert applied_response.content == output_usd.read_bytes()


@pytest.mark.api
@pytest.mark.real_executor
@pytest.mark.parametrize("sample", ["cube", "nested_sphere", "ladder"])
async def test_post_pipeline_runs_real_apply_workflow(
    tmp_path: Path,
    sample: str,
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run an apply-only REST job with cached predictions and a local library."""
    input_usd = tmp_path / f"{sample}.usd"
    target = _create_input(input_usd, sample)
    library_usd = tmp_path / "library.usd"
    _create_library(library_usd)

    async def skip_input_preview(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(pipeline_router, "_render_input_preview", skip_input_preview)
    upload = await client.post(
        "/pipeline/upload-usd",
        files={
            "usd_file": (
                "scene.usd",
                input_usd.read_bytes(),
                "application/octet-stream",
            )
        },
    )
    assert upload.status_code == 201, upload.text
    session_id = upload.json()["session_id"]
    manager = pipeline_router.get_session_manager()
    session_dir = manager.get_session_dir(session_id)
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(
        json.dumps({"id": target, "materials": {"material": "TestMaterial"}}) + "\n",
        encoding="utf-8",
    )

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(
            "materials.yaml",
            "materials:\n"
            "  library_path: library.usd\n"
            "  entries:\n"
            "    - name: TestMaterial\n"
            "      description: A test preview material\n"
            "      binding: /World/Looks/TestMaterial\n",
        )
        archive.write(library_usd, "library.usd")
    submit = await client.post(
        "/pipeline",
        files={
            "materials_zip": (
                "materials.zip",
                archive_bytes.getvalue(),
                "application/zip",
            )
        },
        data={
            "session_id": session_id,
            "layer_only": "true",
            "steps": "apply",
            "optimize_usd": "false",
        },
    )
    assert submit.status_code == 202, submit.text
    for _ in range(500):
        status = (await client.get(f"/pipeline/{session_id}/status")).json()
        if status["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.01)
    assert status["status"] == "completed", status
    output_usd = session_dir / "output" / "scene_with_materials.usd"
    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert layer.subLayerPaths
    assert layer.GetPrimAtPath(target).specifier == Sdf.SpecifierOver
    applied = await client.get(f"/artifacts/{session_id}/output-applied")
    assert applied.status_code == 200
    assert applied.content == output_usd.read_bytes()
