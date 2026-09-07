# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the focused provider-free Texture UV CLI entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import content_agent_workflows.texture as texture_runtime
import pytest
from content_agent_workflows.common.artifacts import atomic_write_json
from content_agent_workflows.texture import build_texture_uv_leaf_invocation
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import content_workflow_cli.texture_capability_runner as texture_capability_runner
import content_workflow_cli.texture_runner as texture_runner
from content_workflow_cli.cli import main


def _write_ready_source(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(1.0, 0.0),
                Gf.Vec2f(1.0, 1.0),
                Gf.Vec2f(0.0, 1.0),
            ]
        )
    )
    assert stage.GetRootLayer().Save()


def test_texture_package_lazy_facade_preserves_public_imports() -> None:
    from content_agent_workflows.texture.models import TextureWorkflowRequest
    from content_agent_workflows.texture.uv_authoring import (
        run_texture_uv_leaf as direct_run_texture_uv_leaf,
    )

    assert texture_runtime.TextureWorkflowRequest is TextureWorkflowRequest
    assert texture_runtime.run_texture_uv_leaf is direct_run_texture_uv_leaf
    assert "TextureWorkflowRequest" in texture_runtime.__all__
    assert "run_texture_uv_leaf" in texture_runtime.__all__


def test_texture_agentic_leaf_uv_prepare_bypasses_umbrella_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.usda"
    output_dir = tmp_path / "leaf-attempt"
    output_dir.mkdir()
    _write_ready_source(source)
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Mesh",),
        policy="inspect",
    )
    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    def unexpected_umbrella(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("focused Texture UV entrypoint entered an umbrella path")

    for provider_surface in (
        "TextureAgentServiceClient",
        "VlmTextureVisualAssessor",
        "ProvidedImageTextureApplyLeaf",
        "invoke_texture_generator",
        "request_texture_provider_proposal",
        "request_texture_critique",
        "run_texture_workflow",
    ):
        monkeypatch.setattr(
            texture_runtime,
            provider_surface,
            unexpected_umbrella,
        )
    monkeypatch.setattr(
        texture_capability_runner,
        "LiveUsdCliTextureValidator",
        unexpected_umbrella,
    )
    monkeypatch.setattr(texture_runner, "_handle_texture_run", unexpected_umbrella)
    monkeypatch.setattr(texture_runner, "_handle_texture_resume", unexpected_umbrella)
    monkeypatch.setattr(
        texture_runner,
        "_handle_texture_agent_step",
        unexpected_umbrella,
    )

    exit_code = main(
        [
            "texture",
            "agentic-leaf",
            "uv-prepare",
            "--invocation",
            str(invocation_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["native_disposition"] == "passed"
    assert payload["provider_invoked"] is False
    assert payload["fixed_pipeline_invoked"] is False
    assert payload["nested_coordinator_invoked"] is False
    assert Path(payload["saved_stage_readbacks"][0]["path"]).is_file()


def test_texture_agentic_leaf_uv_prepare_is_import_isolated_in_fresh_process(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    output_dir = tmp_path / "leaf-attempt"
    output_dir.mkdir()
    _write_ready_source(source)
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=output_dir,
        target_prim_paths=("/World/Mesh",),
        policy="inspect",
    )
    invocation_path = output_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    script = """
import contextlib
import io
import json
import sys

from content_workflow_cli.cli import main

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exit_code = main([
        "texture",
        "agentic-leaf",
        "uv-prepare",
        "--invocation",
        sys.argv[1],
    ])
blocked_exact = {
    "content_agent_workflows.texture.capabilities",
    "content_agent_workflows.texture.client",
    "content_agent_workflows.texture.workflow",
    "world_understanding.functions.graphics.uv_generation",
    "world_understanding.functions.graphics.scene_optimizer_nvcf",
}
blocked = sorted(
    name
    for name in sys.modules
    if name in blocked_exact or "texture_agent" in name
)
print(json.dumps({
    "blocked": blocked,
    "exit_code": exit_code,
    "native_disposition": json.loads(stdout.getvalue())["native_disposition"],
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)

    completed = subprocess.run(
        [sys.executable, "-c", script, str(invocation_path)],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    probe = json.loads(completed.stdout)
    assert probe == {
        "blocked": [],
        "exit_code": 0,
        "native_disposition": "passed",
    }
