# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the isolated VoMP runtime and OVRTX orchestration blueprint."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import numpy as np
import pytest
from PIL import Image
from typer.main import get_command
from typer.testing import CliRunner

pxr = pytest.importorskip("pxr", reason="USD (pxr) not available in this env")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

import physics_agent.integrations as integrations_package  # noqa: E402
from physics_agent.api.defaults import PIPELINE_STEP_NAMES  # noqa: E402
from physics_agent.cli import app  # noqa: E402
from physics_agent.config.schema import STEP_ORDER, get_step_defaults  # noqa: E402
from physics_agent.config.unified_config import UnifiedPipelineConfigTask  # noqa: E402
from physics_agent.config.validator import ConfigValidator  # noqa: E402
from physics_agent.integrations import vomp_runtime as vomp_runtime_module  # noqa: E402
from physics_agent.integrations.vomp import VompIntegrationError  # noqa: E402
from physics_agent.integrations.vomp_defaults import (  # noqa: E402
    DEFAULT_VOMP_ARTIFACT_SHA256,
    DEFAULT_VOMP_RENDER_CONFIG,
    DEFAULT_VOMP_REVISION,
)
from physics_agent.integrations.vomp_pipeline import (  # noqa: E402
    VompRenderConfig,
    _add_camera,
    _ovrtx_render_metadata,
    _prepare_artifact_directory,
    _target_world_mesh,
    _triangulate_faces,
    _validate_mesh_transform_association,
    _validate_render_config,
    prepare_vomp_evidence,
    run_vomp_mass_pipeline,
    run_vomp_volume_deformable_pipeline,
)
from physics_agent.integrations.vomp_runtime import (  # noqa: E402
    ExternalVompRunner,
    VompRunRequest,
    VompRunResult,
    VompRuntimeConfig,
    _unsafe_untracked_runtime_paths,
    _validate_runtime,
)
from physics_agent.integrations.vomp_worker import (  # noqa: E402
    _deduplicate_results,
    _resolve_checkpoint_paths,
    _validate_watertight_mesh,
)
from physics_agent.integrations.vomp_worker import (  # noqa: E402
    _git_revision as _worker_git_revision,
)
from physics_agent.integrations.vomp_worker import (  # noqa: E402
    _runtime_untracked_paths as _worker_runtime_untracked_paths,
)
from physics_agent.integrations.vomp_worker import (  # noqa: E402
    _sha256 as _worker_sha256,
)
from physics_agent.integrations.vomp_worker import (  # noqa: E402
    _unsafe_untracked_runtime_paths as _worker_unsafe_untracked_runtime_paths,
)
from physics_agent.integrations.vomp_worker import (  # noqa: E402
    _write_json_atomic as _worker_write_json_atomic,
)
from physics_agent.tasks.config_vomp_mass import VompMassConfigTask  # noqa: E402
from physics_agent.tasks.vomp_mass import VompMassTask  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_cli_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "physics_agent.cli.setup_logging",
        lambda **_kwargs: logging.getLogger("physics_agent.tests.vomp_runtime"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_git() -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available in this environment")
    return git


def _make_runtime_checkout(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, str], str]:
    _require_git()
    root = tmp_path / "vomp"
    (root / "vomp").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "vomp" / "__init__.py").write_text("VERSION = 1\n", encoding="ascii")
    artifacts = {
        "geometry_checkpoint_dir": root / "weights" / "geometry.pt",
        "matvae_checkpoint_dir": root / "weights" / "matvae.safetensors",
        "normalization_params_path": root / "weights" / "normalization.json",
    }
    for path in artifacts.values():
        path.write_bytes(path.name.encode("ascii"))
    config_path = root / "weights" / "inference.json"
    config_path.write_text(
        json.dumps(
            {key: str(path.relative_to(root)) for key, path in artifacts.items()}
        ),
        encoding="ascii",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "vomp"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "runtime",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hashes = {"config": _sha256(config_path)}
    hashes.update({key: _sha256(path) for key, path in artifacts.items()})
    return root, config_path, hashes, revision


def _write_body_stage(path: Path, *, author_kilograms_per_unit: bool = True) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if author_kilograms_per_unit:
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/World/Body")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Body/Geometry")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(x, y, z)
            for x, y, z in (
                (-0.5, -0.5, -0.5),
                (0.5, -0.5, -0.5),
                (0.5, 0.5, -0.5),
                (-0.5, 0.5, -0.5),
                (-0.5, -0.5, 0.5),
                (0.5, -0.5, 0.5),
                (0.5, 0.5, 0.5),
                (-0.5, 0.5, 0.5),
            )
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [
            0,
            1,
            2,
            3,
            4,
            7,
            6,
            5,
            0,
            4,
            5,
            1,
            1,
            5,
            6,
            2,
            2,
            6,
            7,
            3,
            4,
            0,
            3,
            7,
        ]
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.GetPrim().CreateAttribute(
        "physicsAgent:testMarker", Sdf.ValueTypeNames.String
    ).Set("preserve-me")
    material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/TestMaterial/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.5, 0.8)
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    UsdGeom.Cube.Define(stage, "/World/Other").CreateSizeAttr(1.0)
    stage.GetRootLayer().Save()


def _write_deformable_input_stage(path: Path) -> None:
    _write_body_stage(path)
    stage = Usd.Stage.Open(str(path))
    body = stage.GetPrimAtPath("/World/Body")
    geometry = stage.GetPrimAtPath("/World/Body/Geometry")
    assert body.RemoveAPI(UsdPhysics.RigidBodyAPI)
    assert geometry.RemoveAPI(UsdPhysics.CollisionAPI)
    assert stage.GetRootLayer().Save()


def _copy_mesh(
    stage: Usd.Stage, source_path: str, destination_path: str
) -> UsdGeom.Mesh:
    source = UsdGeom.Mesh(stage.GetPrimAtPath(source_path))
    destination = UsdGeom.Mesh.Define(stage, destination_path)
    destination.CreatePointsAttr(source.GetPointsAttr().Get())
    destination.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    destination.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    destination.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return destination


class _FakeOvrtxBackend:
    def render(self, *, stage: Usd.Stage, cameras: list[str], **_kwargs: Any) -> dict:
        assert stage.GetPrimAtPath("/__PhysicsAgentVompCameras/Camera_000")
        other = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Other"))
        assert other.ComputeVisibility() == UsdGeom.Tokens.invisible
        other_instance = stage.GetPrimAtPath("/World/OtherInstance")
        if other_instance:
            assert (
                UsdGeom.Imageable(other_instance).ComputeVisibility()
                == UsdGeom.Tokens.invisible
            )
        other_instancer = stage.GetPrimAtPath("/World/OtherInstancer")
        if other_instancer:
            assert (
                UsdGeom.Imageable(other_instancer).ComputeVisibility()
                == UsdGeom.Tokens.invisible
            )
        scene_light = stage.GetPrimAtPath("/World/SceneLight")
        if scene_light:
            assert (
                UsdGeom.Imageable(scene_light).ComputeVisibility()
                != UsdGeom.Tokens.invisible
            )
        results = []
        for camera in cameras:
            results.append(
                {
                    "camera": camera,
                    "images": [Image.new("RGB", (32, 32), (32, 64, 96))],
                    "status": "success",
                }
            )
        return {
            "successful_cameras": len(cameras),
            "failed_cameras": 0,
            "results": results,
        }


def _install_fake_ovrtx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering_backend_factory."
        "create_rendering_backend",
        lambda backend, config: _FakeOvrtxBackend(),
    )


def test_prepare_vomp_evidence_exports_real_cameras_and_isolates_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import usd_camera

    source = tmp_path / "body.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    asset = UsdGeom.Xform.Define(stage, "/World/AssetSource")
    _copy_mesh(stage, "/World/Body/Geometry", "/World/AssetSource/Geometry")
    instance = UsdGeom.Xform.Define(stage, "/World/OtherInstance")
    instance.GetPrim().GetReferences().AddInternalReference(asset.GetPath())
    instance.GetPrim().SetInstanceable(True)
    UsdGeom.PointInstancer.Define(stage, "/World/OtherInstancer")
    UsdLux.SphereLight.Define(stage, "/World/SceneLight")
    stage.GetRootLayer().Save()
    _install_fake_ovrtx(monkeypatch)
    extraction_stages: set[int] = set()
    extraction_caches: set[int] = set()
    real_extract = usd_camera.extract_camera_parameters_from_stage

    def recording_extract(**kwargs: Any) -> dict[str, Any]:
        extraction_stages.add(id(kwargs["stage"]))
        extraction_caches.add(id(kwargs["xform_cache"]))
        return real_extract(**kwargs)

    monkeypatch.setattr(
        usd_camera,
        "extract_camera_parameters_from_stage",
        recording_extract,
    )

    result = prepare_vomp_evidence(
        source,
        target_prim_path="/World/Body",
        artifact_dir=tmp_path / "evidence",
        render_config=VompRenderConfig(
            num_views=3,
            image_width=32,
            image_height=32,
            num_sensor_updates=1,
        ),
    )

    assert result.mesh_path.is_file()
    assert result.render_stage_path.is_file()
    assert result.manifest["geometry"]["uniqueMeshCount"] == 1
    assert result.manifest["render"]["successfulCameras"] == 3
    renderer_metadata = json.loads(
        (result.artifact_dir / "ovrtx_render_result.json").read_text(encoding="utf-8")
    )
    assert result.manifest["rendererMetadata"] == renderer_metadata
    assert len(extraction_stages) == 1
    assert len(extraction_caches) == 1
    frames = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert len(frames) == 3
    assert all(
        Path(result.artifact_dir / "renders" / frame["file_path"]).is_file()
        for frame in frames
    )
    for frame in frames:
        transform = np.asarray(frame["transform_matrix"])
        assert np.linalg.norm(transform[:3, 3]) == pytest.approx(2.0)


def test_prepare_vomp_evidence_rejects_animated_geometry_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "animated.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    geometry = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Body/Geometry"))
    translate = geometry.AddTranslateOp()
    translate.Set(Gf.Vec3d(0.0))
    translate.Set(Gf.Vec3d(1.0, 0.0, 0.0), Usd.TimeCode(0.0))
    stage.GetRootLayer().Save()
    _install_fake_ovrtx(monkeypatch)

    with pytest.raises(VompIntegrationError, match="transform is time-varying"):
        prepare_vomp_evidence(
            source,
            target_prim_path="/World/Body",
            artifact_dir=tmp_path / "evidence",
            render_config=VompRenderConfig(
                num_views=1,
                image_width=32,
                image_height=32,
                num_sensor_updates=1,
            ),
        )


def test_prepare_vomp_evidence_rejects_single_geometry_time_sample(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sampled_points.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    points = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Body/Geometry")).GetPointsAttr()
    points.Set(points.Get(), Usd.TimeCode(0.0))
    stage.GetRootLayer().Save()

    with pytest.raises(VompIntegrationError, match="time-sampled geometry"):
        prepare_vomp_evidence(
            source,
            target_prim_path="/World/Body",
            artifact_dir=tmp_path / "evidence",
            render_config=VompRenderConfig(num_views=1),
        )


def test_prepare_vomp_evidence_rejects_blank_ovrtx_frames_and_keeps_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BlankOvrtxBackend(_FakeOvrtxBackend):
        def render(self, **kwargs: Any) -> dict[str, Any]:
            result = super().render(**kwargs)
            blank_frame = {
                "frame": 0,
                "camera": kwargs["cameras"][0],
                "image_file": "Camera_000_f0.png",
                "stats": {
                    "reason": "uniform image",
                    "unique_colors": 1,
                    "dominant_color_ratio": 1.0,
                    "luma_std": 0.0,
                },
            }
            result["blank_render_frames"] = []
            result["warnings"] = ["blank frame"]
            result["results"][0]["blank_render_frames"] = [blank_frame]
            return result

    source = tmp_path / "body.usda"
    evidence = tmp_path / "evidence"
    _write_body_stage(source)
    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering_backend_factory."
        "create_rendering_backend",
        lambda _backend, _config: _BlankOvrtxBackend(),
    )

    with pytest.raises(VompIntegrationError, match="frames as blank"):
        prepare_vomp_evidence(
            source,
            target_prim_path="/World/Body",
            artifact_dir=evidence,
            render_config=VompRenderConfig(
                num_views=1,
                image_width=32,
                image_height=32,
                num_sensor_updates=1,
            ),
        )

    metadata = json.loads(
        (evidence / "ovrtx_render_result.json").read_text(encoding="utf-8")
    )
    assert metadata["blankRenderFrames"][0]["stats"]["reason"] == "uniform image"
    assert not (evidence / "renders_metadata.json").exists()


def test_prepare_vomp_evidence_rejects_subdivision_control_mesh(
    tmp_path: Path,
) -> None:
    source = tmp_path / "subdivision.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    UsdGeom.Mesh(
        stage.GetPrimAtPath("/World/Body/Geometry")
    ).CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
    stage.GetRootLayer().Save()

    with pytest.raises(VompIntegrationError, match="tessellate the evaluated surface"):
        prepare_vomp_evidence(
            source,
            target_prim_path="/World/Body",
            artifact_dir=tmp_path / "evidence",
            render_config=VompRenderConfig(num_views=1),
        )


def test_prepare_vomp_evidence_rejects_visible_non_mesh_geometry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "implicit_geometry.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    UsdGeom.Cube.Define(stage, "/World/Body/ImplicitCube")
    stage.GetRootLayer().Save()

    with pytest.raises(
        VompIntegrationError,
        match="visible unsupported renderable geometry.*ImplicitCube",
    ):
        prepare_vomp_evidence(
            source,
            target_prim_path="/World/Body",
            artifact_dir=tmp_path / "evidence",
            render_config=VompRenderConfig(num_views=1),
        )


def test_prepare_vomp_evidence_preflights_mass_authoring_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: list[tuple[Path, str]] = []

    missing_units = tmp_path / "missing_mass_units.usda"
    _write_body_stage(missing_units, author_kilograms_per_unit=False)
    sources.append((missing_units, "kilogramsPerUnit"))

    disabled = tmp_path / "disabled.usda"
    _write_body_stage(disabled)
    stage = Usd.Stage.Open(str(disabled))
    UsdPhysics.RigidBodyAPI(
        stage.GetPrimAtPath("/World/Body")
    ).GetRigidBodyEnabledAttr().Set(False)
    stage.GetRootLayer().Save()
    sources.append((disabled, "explicitly disabled"))

    nested = tmp_path / "nested.usda"
    _write_body_stage(nested)
    stage = Usd.Stage.Open(str(nested))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/Body/Geometry"))
    stage.GetRootLayer().Save()
    sources.append((nested, "contains enabled rigid body"))

    scaled = tmp_path / "scaled.usda"
    _write_body_stage(scaled)
    stage = Usd.Stage.Open(str(scaled))
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/Body")).AddScaleOp().Set(
        Gf.Vec3d(2.0, 1.0, 1.0)
    )
    stage.GetRootLayer().Save()
    sources.append((scaled, "scale, shear, or reflection"))

    def unexpected_backend(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("OVRTX must not be created for an invalid target")

    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering_backend_factory."
        "create_rendering_backend",
        unexpected_backend,
    )
    for index, (source, message) in enumerate(sources):
        with pytest.raises(VompIntegrationError, match=message):
            prepare_vomp_evidence(
                source,
                target_prim_path="/World/Body",
                artifact_dir=tmp_path / f"evidence_{index}",
                render_config=VompRenderConfig(num_views=1),
            )

    usdz = tmp_path / "body.usdz"
    usdz.write_bytes(missing_units.read_bytes())
    with pytest.raises(VompIntegrationError, match="USDZ packages"):
        prepare_vomp_evidence(
            usdz,
            target_prim_path="/World/Body",
            artifact_dir=tmp_path / "evidence_usdz",
            render_config=VompRenderConfig(num_views=1),
        )


def test_world_mesh_handles_supported_usd_geometry_branches(tmp_path: Path) -> None:
    source = tmp_path / "geometry_branches.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    base_path = "/World/Body/Geometry"

    invisible = _copy_mesh(stage, base_path, "/World/Body/Invisible")
    invisible.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    guide = _copy_mesh(stage, base_path, "/World/Body/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    invisible_cube = UsdGeom.Cube.Define(stage, "/World/Body/InvisibleCube")
    invisible_cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    guide_sphere = UsdGeom.Sphere.Define(stage, "/World/Body/GuideSphere")
    guide_sphere.CreatePurposeAttr(UsdGeom.Tokens.guide)
    inactive = _copy_mesh(stage, base_path, "/World/Body/Inactive")
    inactive.GetPrim().SetActive(False)
    abstract = stage.CreateClassPrim("/World/Body/Abstract")
    abstract.SetTypeName("Mesh")
    holes = _copy_mesh(stage, base_path, "/World/Body/Holes")
    holes.CreateHoleIndicesAttr(list(range(6)))
    reflected = _copy_mesh(stage, base_path, "/World/Body/Reflected")
    UsdGeom.Xformable(reflected).AddScaleOp().Set(Gf.Vec3d(-1.0, 1.0, 1.0))
    _copy_mesh(stage, base_path, "/World/Body/Duplicate")
    UsdGeom.Xformable(stage.GetPrimAtPath(base_path)).SetResetXformStack(True)

    world_mesh = _target_world_mesh(
        stage.GetPrimAtPath("/World/Body"),
        meters_per_unit=1.0,
    )

    assert world_mesh.source_mesh_count == 4
    assert world_mesh.unique_mesh_count == 2
    assert (
        _triangulate_faces(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            np.asarray([3]),
            np.asarray([0, 1, 2]),
            {0},
        ).size
        == 0
    )

    camera_stage = Usd.Stage.CreateInMemory()
    camera_path = _add_camera(
        camera_stage,
        {
            "index": 0,
            "yaw": 0.0,
            "pitch": np.pi / 2.0,
            "radius": 2.0,
            "fovDegrees": 40.0,
        },
        center_m=np.zeros(3),
        scale_m=1.0,
        meters_per_unit=1.0,
    )
    assert camera_stage.GetPrimAtPath(camera_path)


def test_triangulate_faces_accepts_only_safe_polygon_fans() -> None:
    convex = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    assert np.array_equal(
        _triangulate_faces(
            convex,
            np.asarray([4]),
            np.asarray([0, 1, 2, 3]),
            set(),
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]]),
    )

    concave = convex.tolist()
    concave.insert(3, [1.0, 0.5, 0.0])
    with pytest.raises(VompIntegrationError, match="concave polygon"):
        _triangulate_faces(
            np.asarray(concave),
            np.asarray([5]),
            np.asarray([0, 1, 2, 3, 4]),
            set(),
        )

    non_planar = convex.copy()
    non_planar[2, 2] = 0.25
    with pytest.raises(VompIntegrationError, match="non-planar polygon"):
        _triangulate_faces(
            non_planar,
            np.asarray([4]),
            np.asarray([0, 1, 2, 3]),
            set(),
        )

    angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    pentagon = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(5)))
    with pytest.raises(VompIntegrationError, match="self-intersecting polygon"):
        _triangulate_faces(
            pentagon,
            np.asarray([5]),
            np.asarray([0, 2, 4, 1, 3]),
            set(),
        )


def test_triangulate_faces_accepts_float32_quantized_planar_polygon() -> None:
    polygon = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.7, 1.3, 0.0],
            [-0.2, 1.1, 0.0],
        ]
    )
    axis = np.asarray([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross_matrix = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross_matrix
    )
    point3f_values = (polygon @ rotation.T + np.asarray([10.0, -20.0, 30.0])).astype(
        np.float32
    )

    assert np.array_equal(
        _triangulate_faces(
            point3f_values.astype(np.float64),
            np.asarray([4]),
            np.asarray([0, 1, 2, 3]),
            set(),
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]]),
    )


def test_triangulate_faces_vectorizes_triangle_only_meshes() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    indices = np.asarray([0, 1, 2, 0, 2, 3])
    assert np.array_equal(
        _triangulate_faces(points, np.asarray([3, 3]), indices, set()),
        indices.reshape((-1, 3)),
    )
    assert np.array_equal(
        _triangulate_faces(points, np.asarray([3, 3]), indices, {1}),
        np.asarray([[0, 1, 2]]),
    )
    assert not _triangulate_faces(points, np.asarray([3, 3]), indices, {0, 1}).size


@pytest.mark.parametrize(
    ("points", "indices", "message"),
    [
        (np.eye(3), np.asarray([0, 1]), "malformed face indices"),
        (np.eye(3), np.asarray([0, 1, 2, 0]), "trailing face indices"),
        (np.eye(3), np.asarray([0, 1, 4]), "invalid indices"),
        (np.eye(3), np.asarray([0, 1, 1]), "repeated vertex indices"),
        (np.zeros((3, 3)), np.asarray([0, 1, 2]), "degenerate face"),
        (
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([0, 1, 2]),
            "repeated vertex positions",
        ),
        (
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.asarray([0, 1, 2]),
            "zero-area face",
        ),
    ],
)
def test_triangle_fast_path_rejects_invalid_geometry(
    points: np.ndarray,
    indices: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(VompIntegrationError, match=message):
        _triangulate_faces(points, np.asarray([3]), indices, set())


def test_world_mesh_honors_left_handed_orientation(tmp_path: Path) -> None:
    source = tmp_path / "orientation.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    target = stage.GetPrimAtPath("/World/Body")
    right_handed = _target_world_mesh(target, meters_per_unit=1.0)

    UsdGeom.Mesh(stage.GetPrimAtPath("/World/Body/Geometry")).CreateOrientationAttr(
        UsdGeom.Tokens.leftHanded
    )
    left_handed = _target_world_mesh(target, meters_per_unit=1.0)

    assert left_handed.vertices_m == pytest.approx(right_handed.vertices_m)
    assert np.array_equal(
        left_handed.triangles,
        right_handed.triangles[:, [0, 2, 1]],
    )


class _FakeVompRunner:
    def run(self, request: VompRunRequest) -> VompRunResult:
        coordinates = np.asarray(
            [
                (x, y, z)
                for x in (-0.25, 0.25)
                for y in (-0.25, 0.25)
                for z in (-0.25, 0.25)
            ],
            dtype=np.float32,
        )
        dtype = [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("youngs_modulus", "<f4"),
            ("poissons_ratio", "<f4"),
            ("density", "<f4"),
            ("segment_id", "<U32"),
        ]
        values = np.zeros(len(coordinates), dtype=dtype)
        values["x"], values["y"], values["z"] = coordinates.T
        values["youngs_modulus"] = 1.0e6
        values["poissons_ratio"] = 0.3
        values["density"] = 1000.0
        values["segment_id"] = "voxel_material"
        np.savez_compressed(request.output_npz_path, voxel_data=values)
        log_path = request.output_dir / "fake_worker.log"
        log_path.write_text("fake worker\n", encoding="ascii")
        manifest = {
            "protocolVersion": 1,
            "status": "success",
            "outputNpzPath": str(request.output_npz_path),
            "outputNpzSha256": _sha256(request.output_npz_path),
            "sampleCount": 8,
            "completeVoxelField": True,
            "voxelSizeNormalized": 0.5,
            "meshTransform": {"centerM": [0.0, 0.0, 0.0], "scaleM": 1.0},
            "deduplication": {"removedRows": 0},
            "runtime": {
                "vompRevision": "a" * 40,
                "pythonVersion": "3.10",
                "torchVersion": "2.4.0",
                "cudaAvailable": True,
                "cudaVersion": "12.1",
                "artifacts": {"config": {"sha256": "b" * 64, "sizeBytes": 1}},
            },
        }
        return VompRunResult(
            output_npz_path=request.output_npz_path,
            sample_count=8,
            voxel_size_m=0.5,
            coordinate_unit_meters=1.0,
            coordinate_offset_m=(0.0, 0.0, 0.0),
            manifest=manifest,
            worker_log_path=log_path,
        )


class _Float32NonBinaryVompRunner(_FakeVompRunner):
    source_scale_m = 1.23456789

    def __init__(self) -> None:
        self.mesh_scale_m = self.source_scale_m
        self.voxel_size_m = self.source_scale_m / 64.0

    def run(self, request: VompRunRequest) -> VompRunResult:
        base = super().run(request)
        evidence_manifest = json.loads(
            (request.output_dir / "ovrtx_manifest.json").read_text(encoding="utf-8")
        )
        geometry = evidence_manifest["geometry"]
        self.mesh_scale_m = float(geometry["scaleM"])
        self.voxel_size_m = self.mesh_scale_m / 64.0
        center = np.asarray(geometry["centerM"], dtype=np.float64)
        center[0] += 0.123456789
        coordinates = np.zeros((5, 3), dtype=np.float32)
        coordinates[:] = center
        coordinates[:, 0] += (np.arange(5, dtype=np.float64) - 2.0) * self.voxel_size_m
        dtype = [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("youngs_modulus", "<f4"),
            ("poissons_ratio", "<f4"),
            ("density", "<f4"),
            ("segment_id", "<U32"),
        ]
        values = np.zeros(len(coordinates), dtype=dtype)
        values["x"], values["y"], values["z"] = coordinates.T
        values["youngs_modulus"] = 1.0e6
        values["poissons_ratio"] = 0.3
        values["density"] = 1000.0
        values["segment_id"] = "voxel_material"
        np.savez_compressed(request.output_npz_path, voxel_data=values)
        manifest = json.loads(json.dumps(base.manifest))
        manifest.update(
            {
                "outputNpzSha256": _sha256(request.output_npz_path),
                "sampleCount": 5,
                "voxelSizeNormalized": 1.0 / 64.0,
                "meshTransform": {
                    "centerM": geometry["centerM"],
                    "scaleM": self.mesh_scale_m,
                },
            }
        )
        return VompRunResult(
            output_npz_path=request.output_npz_path,
            sample_count=5,
            voxel_size_m=self.voxel_size_m,
            coordinate_unit_meters=1.0,
            coordinate_offset_m=(-0.123456789, 0.0, 0.0),
            manifest=manifest,
            worker_log_path=base.worker_log_path,
        )


def test_volume_deformable_pipeline_runs_inference_without_rigid_authoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "body.usda"
    output = tmp_path / "body_vomp_deformable.usda"
    _write_deformable_input_stage(source)
    _install_fake_ovrtx(monkeypatch)

    result = run_vomp_volume_deformable_pipeline(
        source,
        output,
        target_prim_path="/World/Body",
        work_dir=tmp_path / "runs",
        runner=_FakeVompRunner(),
        render_config=VompRenderConfig(
            num_views=2,
            image_width=32,
            image_height=32,
            num_sensor_updates=1,
        ),
    )

    assert result.apply_result.sample_count == 8
    assert result.apply_result.point_count == 27
    assert result.apply_result.tet_count == 48
    assert result.apply_result.mass_kg == pytest.approx(1000.0)
    assert result.worker_manifest_path.is_file()
    assert result.evidence.manifest["render"]["successfulCameras"] == 2
    stage = Usd.Stage.Open(str(output))
    body = stage.GetPrimAtPath("/World/Body")
    assert not body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not body.HasAPI(UsdPhysics.MassAPI)
    assert "PhysicsDeformableBodyAPI" in {
        str(schema) for schema in body.GetPrimTypeInfo().GetAppliedAPISchemas()
    }
    provenance = json.loads(
        result.apply_result.provenance_path.read_text(encoding="utf-8")
    )
    assert provenance["evidence"]["inference"]["runtime"]["vompRevision"] == ("a" * 40)


def test_volume_deformable_pipeline_accepts_official_float32_lattice_roundoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "body.usda"
    output = tmp_path / "body_vomp_deformable.usda"
    _write_deformable_input_stage(source)
    runner = _Float32NonBinaryVompRunner()
    stage = Usd.Stage.Open(str(source))
    geometry = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Body/Geometry"))
    geometry.AddScaleOp().Set(
        Gf.Vec3d(
            runner.source_scale_m,
            runner.source_scale_m / 64.0,
            runner.source_scale_m / 64.0,
        )
    )
    assert stage.GetRootLayer().Save()
    _install_fake_ovrtx(monkeypatch)

    result = run_vomp_volume_deformable_pipeline(
        source,
        output,
        target_prim_path="/World/Body",
        work_dir=tmp_path / "runs",
        runner=runner,
        render_config=VompRenderConfig(
            num_views=2,
            image_width=32,
            image_height=32,
            num_sensor_updates=1,
        ),
    )

    expected_mass = 5.0 * 1000.0 * runner.voxel_size_m**3
    assert result.apply_result.sample_count == 5
    assert result.apply_result.point_count == 24
    assert result.apply_result.tet_count == 30
    assert result.apply_result.mass_kg == pytest.approx(expected_mass, rel=2.0e-6)
    provenance = json.loads(
        result.apply_result.provenance_path.read_text(encoding="utf-8")
    )
    assert provenance["topology"]["maximumVoxelCenterSnapErrorM"] > 0.0
    assert provenance["topology"]["massCenterSnapErrorM"] > 0.0


def test_volume_deformable_pipeline_rejects_rigid_contract_before_inference(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rigid_body.usda"
    _write_body_stage(source)

    with pytest.raises(VompIntegrationError, match="rigid mass semantics"):
        run_vomp_volume_deformable_pipeline(
            source,
            tmp_path / "deformable.usda",
            target_prim_path="/World/Body",
            work_dir=tmp_path / "runs",
            runner=_FakeVompRunner(),
        )

    assert not (tmp_path / "runs" / "evidence" / "vomp_voxel_materials.npz").exists()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"material_reduction": "homogeneous"},
            "material_reduction must be 'reject' or 'homogeneous-volume-average'",
        ),
        (
            {"max_deformable_voxels": 0},
            "max_deformable_voxels must be a positive integer",
        ),
    ],
)
def test_volume_deformable_pipeline_validates_options_before_side_effects(
    tmp_path: Path,
    options: dict[str, Any],
    message: str,
) -> None:
    work_root = tmp_path / "runs"

    with pytest.raises(VompIntegrationError, match=message):
        run_vomp_volume_deformable_pipeline(
            tmp_path / "missing.usda",
            tmp_path / "output.usda",
            target_prim_path="/World/Body",
            work_dir=work_root,
            runner=_FakeVompRunner(),
            **options,
        )

    assert not work_root.exists()


def test_volume_deformable_pipeline_requires_runner_or_runtime(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"

    with pytest.raises(VompIntegrationError, match="requires a VoMP runtime"):
        run_vomp_volume_deformable_pipeline(
            tmp_path / "missing.usda",
            tmp_path / "output.usda",
            target_prim_path="/World/Body",
            work_dir=work_root,
        )

    assert not work_root.exists()


def test_volume_deformable_pipeline_clamps_runtime_cap_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_root = tmp_path / "runs"
    runtime_config = VompRuntimeConfig(
        runtime_root=tmp_path / "VoMP",
        python_executable=tmp_path / "python",
        config_path=tmp_path / "inference.json",
    )
    calls: list[tuple[str, VompRuntimeConfig]] = []

    class StopBeforeArtifactWork(Exception):
        pass

    def validate(config: VompRuntimeConfig) -> tuple[Path, Path, Path]:
        calls.append(("validate", config))
        return config.runtime_root, config.python_executable, config.config_path

    def build_runner(config: VompRuntimeConfig) -> None:
        calls.append(("construct", config))
        raise StopBeforeArtifactWork

    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline._validate_runtime",
        validate,
    )
    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline.ExternalVompRunner",
        build_runner,
    )

    with pytest.raises(StopBeforeArtifactWork):
        run_vomp_volume_deformable_pipeline(
            tmp_path / "missing.usda",
            tmp_path / "output.usda",
            target_prim_path="/World/Body",
            work_dir=work_root,
            runtime_config=runtime_config,
        )

    assert [name for name, _config in calls] == ["validate", "construct"]
    assert all(config.max_complete_voxels == 65_536 for _name, config in calls)
    assert runtime_config.max_complete_voxels == 262_144
    assert not work_root.exists()


def test_volume_deformable_pipeline_builds_runner_from_validated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_config = VompRuntimeConfig(
        runtime_root=tmp_path / "VoMP",
        python_executable=tmp_path / "python",
        config_path=tmp_path / "inference.json",
        max_complete_voxels=65_536,
    )
    calls: list[tuple[str, VompRuntimeConfig]] = []

    def validate(config: VompRuntimeConfig) -> tuple[Path, Path, Path]:
        calls.append(("validate", config))
        return config.runtime_root, config.python_executable, config.config_path

    def build_runner(config: VompRuntimeConfig) -> _FakeVompRunner:
        calls.append(("construct", config))
        return _FakeVompRunner()

    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline._validate_runtime",
        validate,
    )
    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline.ExternalVompRunner",
        build_runner,
    )

    with pytest.raises(VompIntegrationError, match="positive integer"):
        run_vomp_volume_deformable_pipeline(
            tmp_path / "missing.usda",
            tmp_path / "output.usda",
            target_prim_path="/World/Body",
            work_dir=tmp_path / "runs",
            runtime_config=runtime_config,
            render_config=VompRenderConfig(num_views=0),
        )

    assert calls == [("validate", runtime_config), ("construct", runtime_config)]
    assert not (tmp_path / "runs").exists()


def test_pipeline_preserves_existing_collision_and_material_opinions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "body.usda"
    output = tmp_path / "body_vomp.usda"
    _write_body_stage(source)
    _install_fake_ovrtx(monkeypatch)

    result = run_vomp_mass_pipeline(
        source,
        output,
        target_prim_path="/World/Body",
        work_dir=tmp_path / "runs",
        runner=_FakeVompRunner(),
        render_config=VompRenderConfig(
            num_views=2,
            image_width=32,
            image_height=32,
            num_sensor_updates=1,
        ),
    )

    assert result.apply_result.mass_properties.mass_kg == pytest.approx(1000.0)
    stage = Usd.Stage.Open(str(output))
    body = stage.GetPrimAtPath("/World/Body")
    geometry = stage.GetPrimAtPath("/World/Body/Geometry")
    assert body.HasAPI(UsdPhysics.MassAPI)
    assert geometry.HasAPI(UsdPhysics.CollisionAPI)
    assert geometry.GetAttribute("physicsAgent:testMarker").Get() == "preserve-me"
    bound_material, _ = UsdShade.MaterialBindingAPI(geometry).ComputeBoundMaterial()
    assert bound_material.GetPath() == Sdf.Path("/World/Looks/TestMaterial")
    report = json.loads(result.apply_result.provenance_path.read_text(encoding="utf-8"))
    assert report["evidence"]["rendering"]["backend"] == "world_understanding.ovrtx"
    assert (
        report["evidence"]["rendering"]["rendererMetadata"]
        == result.evidence.manifest["rendererMetadata"]
    )
    usd_provenance = body.GetCustomDataByKey("physicsAgentVomp")
    assert (
        json.loads(usd_provenance["evidence"]["rendering"]["rendererMetadataJson"])
        == result.evidence.manifest["rendererMetadata"]
    )
    assert report["evidence"]["inference"]["runtime"]["vompRevision"] == "a" * 40

    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline._validate_runtime",
        lambda _config: (tmp_path / "VoMP", tmp_path / "python", tmp_path / "config"),
    )
    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline.ExternalVompRunner",
        lambda _config: _FakeVompRunner(),
    )
    runtime_result = run_vomp_mass_pipeline(
        source,
        tmp_path / "body_runtime_vomp.usda",
        target_prim_path="/World/Body",
        work_dir=tmp_path / "runtime_runs",
        runtime_config=VompRuntimeConfig(
            runtime_root=tmp_path / "VoMP",
            python_executable=tmp_path / "python",
            config_path=tmp_path / "inference.json",
        ),
        render_config=VompRenderConfig(
            num_views=1,
            image_width=32,
            image_height=32,
            num_sensor_updates=1,
        ),
    )
    assert runtime_result.apply_result.sample_count == 8


def test_pipeline_rejects_composition_change_during_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "body.usda"
    sublayer = tmp_path / "support.usda"
    output = tmp_path / "body_vomp.usda"
    _write_body_stage(source)
    support_stage = Usd.Stage.CreateNew(str(sublayer))
    UsdGeom.Scope.Define(support_stage, "/Support")
    support_stage.GetRootLayer().Save()
    source_stage = Usd.Stage.Open(str(source))
    source_stage.GetRootLayer().subLayerPaths.append(sublayer.name)
    source_stage.GetRootLayer().Save()
    _install_fake_ovrtx(monkeypatch)

    class _MutatingRunner(_FakeVompRunner):
        def run(self, request: VompRunRequest) -> VompRunResult:
            result = super().run(request)
            sublayer.write_text(
                sublayer.read_text(encoding="utf-8") + "\n# changed during inference\n",
                encoding="utf-8",
            )
            return result

    with pytest.raises(VompIntegrationError, match="composition changed"):
        run_vomp_mass_pipeline(
            source,
            output,
            target_prim_path="/World/Body",
            work_dir=tmp_path / "runs",
            runner=_MutatingRunner(),
            render_config=VompRenderConfig(
                num_views=1,
                image_width=32,
                image_height=32,
                num_sensor_updates=1,
            ),
        )

    assert not output.exists()
    assert not (tmp_path / "runs/evidence/vomp_inference_manifest.json").exists()


def test_pipeline_binds_authoring_to_evidence_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "body.usda"
    sublayer = tmp_path / "support.usda"
    output = tmp_path / "body_vomp.usda"
    _write_body_stage(source)
    support_stage = Usd.Stage.CreateNew(str(sublayer))
    UsdGeom.Scope.Define(support_stage, "/Support")
    support_stage.GetRootLayer().Save()
    source_stage = Usd.Stage.Open(str(source))
    source_stage.GetRootLayer().subLayerPaths.append(sublayer.name)
    source_stage.GetRootLayer().Save()
    _install_fake_ovrtx(monkeypatch)

    def mutate_after_pipeline_verification(
        evidence_manifest: dict[str, Any],
        worker_manifest: dict[str, Any],
    ) -> None:
        _validate_mesh_transform_association(evidence_manifest, worker_manifest)
        sublayer.write_text(
            sublayer.read_text(encoding="utf-8") + "\n# changed before authoring\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline._validate_mesh_transform_association",
        mutate_after_pipeline_verification,
    )

    with pytest.raises(VompIntegrationError, match="composition changed"):
        run_vomp_mass_pipeline(
            source,
            output,
            target_prim_path="/World/Body",
            work_dir=tmp_path / "runs",
            runner=_FakeVompRunner(),
            render_config=VompRenderConfig(
                num_views=1,
                image_width=32,
                image_height=32,
                num_sensor_updates=1,
            ),
        )

    assert not output.exists()


def test_pipeline_attests_runtime_before_replacing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_root = tmp_path / "runs"
    evidence = work_root / "evidence"
    evidence.mkdir(parents=True)
    stale = evidence / "previous-result.json"
    stale.write_text("preserve on preflight failure\n", encoding="ascii")

    def reject_runtime(_config: VompRuntimeConfig) -> tuple[Path, Path, Path]:
        assert stale.is_file()
        raise VompIntegrationError("runtime preflight failed")

    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline._validate_runtime",
        reject_runtime,
    )
    with pytest.raises(VompIntegrationError, match="runtime preflight failed"):
        run_vomp_mass_pipeline(
            tmp_path / "missing.usda",
            tmp_path / "output.usda",
            target_prim_path="/World/Body",
            work_dir=work_root,
            runtime_config=VompRuntimeConfig(
                runtime_root=tmp_path / "VoMP",
                python_executable=tmp_path / "python",
                config_path=tmp_path / "inference.json",
            ),
        )

    assert stale.read_text(encoding="ascii") == "preserve on preflight failure\n"


def test_pipeline_validates_render_config_before_replacing_evidence(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "runs"
    evidence = work_root / "evidence"
    evidence.mkdir(parents=True)
    previous = evidence / "previous-result.json"
    previous.write_text("preserve on preflight failure\n", encoding="ascii")

    with pytest.raises(VompIntegrationError, match="positive integer"):
        run_vomp_mass_pipeline(
            tmp_path / "missing.usda",
            tmp_path / "output.usda",
            target_prim_path="/World/Body",
            work_dir=work_root,
            runner=_FakeVompRunner(),
            render_config=VompRenderConfig(num_views=0),
        )

    assert previous.read_text(encoding="ascii") == "preserve on preflight failure\n"


def test_pipeline_requires_runner_or_runtime(tmp_path: Path) -> None:
    with pytest.raises(VompIntegrationError, match="requires a VoMP runtime"):
        run_vomp_mass_pipeline(
            tmp_path / "input.usda",
            tmp_path / "output.usda",
            target_prim_path="/World/Body",
            work_dir=tmp_path / "work",
        )


def test_worker_deduplicates_only_identical_voxel_rows() -> None:
    base = {
        "voxel_coords_world": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "youngs_modulus": np.asarray([1.0, 1.0]),
        "poisson_ratio": np.asarray([0.3, 0.3]),
        "density": np.asarray([1000.0, 1000.0]),
        "segment_id": np.asarray(["steel", "steel"]),
        "global_metadata": {"source": "test"},
    }
    report = _deduplicate_results(base)
    assert report["removedRows"] == 1
    assert len(base["density"]) == 1
    assert base["segment_id"].tolist() == ["steel"]
    assert base["global_metadata"] == {"source": "test"}

    conflicting = {
        "voxel_coords_world": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "youngs_modulus": np.asarray([1.0, 1.0]),
        "poisson_ratio": np.asarray([0.3, 0.3]),
        "density": np.asarray([1000.0, 900.0]),
    }
    with pytest.raises(RuntimeError, match="conflicting density"):
        _deduplicate_results(conflicting)


def test_worker_rejects_non_watertight_meshes_and_unicode_runtime_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trimesh

    closed_path = tmp_path / "closed.ply"
    open_path = tmp_path / "open.ply"
    closed = trimesh.creation.box()
    closed.export(closed_path)
    open_mesh = closed.copy()
    open_mesh.update_faces(np.arange(len(open_mesh.faces) - 1))
    open_mesh.remove_unreferenced_vertices()
    open_mesh.export(open_path)

    _validate_watertight_mesh(closed_path)
    with pytest.raises(RuntimeError, match="watertight manifold"):
        _validate_watertight_mesh(open_path)
    monkeypatch.setattr(trimesh, "load", lambda _path: object())
    with pytest.raises(RuntimeError, match="polygon mesh"):
        _validate_watertight_mesh(closed_path)
    assert _worker_unsafe_untracked_runtime_paths(
        [
            "vomp/injected_\N{LATIN SMALL LETTER E WITH ACUTE}.py",
            ".venv/lib/injected.so",
            "venv/lib/injected.py",
            "outputs/report.py",
        ]
    ) == ["vomp/injected_\N{LATIN SMALL LETTER E WITH ACUTE}.py"]
    assert _unsafe_untracked_runtime_paths(
        [".venv/lib/injected.so", "outputs/report.py", "vomp/injected.py"]
    ) == ["vomp/injected.py"]


def test_mesh_frame_attestation_and_bounded_artifact_directory(tmp_path: Path) -> None:
    render_manifest = {"geometry": {"centerM": [1.0, 2.0, 3.0], "scaleM": 4.0}}
    inference_manifest = {"meshTransform": {"centerM": [1.0, 2.0, 3.0], "scaleM": 4.0}}
    _validate_mesh_transform_association(render_manifest, inference_manifest)
    inference_manifest["meshTransform"]["scaleM"] = 2.0
    with pytest.raises(VompIntegrationError, match="does not match"):
        _validate_mesh_transform_association(render_manifest, inference_manifest)
    with pytest.raises(VompIntegrationError, match="omitted"):
        _validate_mesh_transform_association(render_manifest, {})
    for invalid_transform in (
        {"centerM": [1.0, 2.0], "scaleM": 4.0},
        {"centerM": [1.0, 2.0, 3.0], "scaleM": True},
        {"centerM": [1.0, 2.0, 3.0], "scaleM": 10**400},
        {"centerM": [1.0, 2.0, 3.0], "scaleM": float("inf")},
    ):
        with pytest.raises(VompIntegrationError, match="invalid"):
            _validate_mesh_transform_association(
                render_manifest,
                {"meshTransform": invalid_transform},
            )

    work_root = tmp_path / "runs"
    work_root.mkdir()
    artifact_dir = _prepare_artifact_directory(work_root)
    (artifact_dir / "stale.png").write_bytes(b"stale")
    assert _prepare_artifact_directory(work_root) == artifact_dir
    assert not (artifact_dir / "stale.png").exists()

    artifact_dir.rmdir()
    artifact_dir.write_text("stale file", encoding="ascii")
    assert _prepare_artifact_directory(work_root).is_dir()
    artifact_dir.rmdir()
    external = tmp_path / "external"
    external.mkdir()
    artifact_dir.symlink_to(external, target_is_directory=True)
    assert _prepare_artifact_directory(work_root).is_dir()
    assert external.is_dir()


@pytest.mark.parametrize(
    "render_result",
    [
        {"results": {}},
        {"results": [], "blank_render_frames": {}},
        {"results": [None]},
        {"results": [{"blank_render_frames": {}}]},
        {"results": [{"images": None}]},
        {"results": [], "total_render_time": float("nan")},
    ],
)
def test_ovrtx_metadata_rejects_malformed_renderer_results(
    render_result: dict[str, Any],
) -> None:
    with pytest.raises(VompIntegrationError, match="metadata"):
        _ovrtx_render_metadata(render_result)


@pytest.mark.parametrize(
    "config",
    [
        VompRenderConfig(num_views=True),
        VompRenderConfig(image_width=0),
        VompRenderConfig(radius=0.5),
        VompRenderConfig(fov_degrees=180.0),
        VompRenderConfig(render_mode="invalid"),
        VompRenderConfig(material_target="invalid"),
    ],
)
def test_render_config_rejects_invalid_direct_api_values(
    config: VompRenderConfig,
) -> None:
    with pytest.raises(VompIntegrationError):
        _validate_render_config(config)


def test_runtime_attestation_rejects_modified_checkout(tmp_path: Path) -> None:
    root, config_path, hashes, revision = _make_runtime_checkout(tmp_path)
    runtime = VompRuntimeConfig(
        runtime_root=root,
        python_executable=Path(sys.executable),
        config_path=config_path,
        expected_revision=revision,
        expected_artifact_sha256=hashes,
    )
    assert _validate_runtime(runtime)[0] == root

    with pytest.raises(VompIntegrationError, match="lowercase commit SHA"):
        _validate_runtime(
            VompRuntimeConfig(
                runtime_root=root,
                python_executable=Path(sys.executable),
                config_path=config_path,
                expected_revision="A" * 40,
                expected_artifact_sha256=hashes,
            )
        )

    (root / "vomp" / "__init__.py").write_text("VERSION = 2\n", encoding="ascii")
    with pytest.raises(VompIntegrationError, match="tracked modifications"):
        _validate_runtime(runtime)


def test_runtime_attestation_resolves_relative_python_and_config(
    tmp_path: Path,
) -> None:
    root, _config_path, hashes, revision = _make_runtime_checkout(tmp_path)
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (root / "outputs").mkdir()
    (root / "outputs/analysis.py").write_text(
        "print('safe output')\n", encoding="ascii"
    )
    (root / ".gitignore").write_text("*.so\n.venv/\n", encoding="ascii")
    (root / ".venv/lib").mkdir(parents=True)
    (root / ".venv/lib/ignored.so").write_bytes(b"approved environment")

    runtime = VompRuntimeConfig(
        runtime_root=root,
        python_executable=Path(".venv/bin/python"),
        config_path=Path("weights/inference.json"),
        expected_revision=revision,
        expected_artifact_sha256=hashes,
    )
    validated_root, validated_python, validated_config = _validate_runtime(runtime)

    assert validated_root == root
    assert validated_python == python
    assert validated_config == root / "weights/inference.json"

    ignored_runtime = root / "vomp/ignored.so"
    ignored_runtime.write_bytes(b"ignored package code")
    assert "vomp/ignored.so" in _worker_runtime_untracked_paths(root)
    with pytest.raises(VompIntegrationError, match="untracked runtime code"):
        _validate_runtime(runtime)
    ignored_runtime.unlink()

    unicode_runtime = root / "vomp" / "injected_\N{LATIN SMALL LETTER E WITH ACUTE}.py"
    unicode_runtime.write_text("VALUE = 1\n", encoding="ascii")
    with pytest.raises(VompIntegrationError, match="untracked runtime code"):
        _validate_runtime(runtime)
    unicode_runtime.unlink()

    (root / "injected").mkdir()
    (root / "injected/__init__.py").write_text("VALUE = 1\n", encoding="ascii")
    with pytest.raises(VompIntegrationError, match="untracked runtime code"):
        _validate_runtime(runtime)


def test_worker_process_runs_with_isolated_standard_io(tmp_path: Path) -> None:
    log_path = tmp_path / "worker.log"

    with log_path.open("w", encoding="utf-8") as log_stream:
        result = vomp_runtime_module._run_worker_process(
            [sys.executable, "-I", "-c", "print('worker output')"],
            cwd=tmp_path,
            environment={},
            log_stream=log_stream,
            timeout_seconds=10.0,
        )

    assert result.returncode == 0
    assert log_path.read_text(encoding="utf-8") == "worker output\n"


def test_external_runner_success_and_worker_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path, _hashes, revision = _make_runtime_checkout(tmp_path)
    worker_json = tmp_path / "worker.json"
    _worker_write_json_atomic(worker_json, {"ok": True})
    assert json.loads(worker_json.read_text(encoding="utf-8")) == {"ok": True}
    for writer, name in (
        (vomp_runtime_module._write_json_atomic, "runtime_bad.json"),
        (_worker_write_json_atomic, "worker_bad.json"),
    ):
        with pytest.raises(ValueError, match="Out of range float"):
            writer(tmp_path / name, {"value": float("nan")})
        assert not list(tmp_path.glob(f".{Path(name).stem}_*"))
    assert _worker_sha256(worker_json) == _sha256(worker_json)
    assert _worker_git_revision(root) == revision
    monkeypatch.chdir(root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifacts = _resolve_checkpoint_paths(config_path, config)
    assert artifacts["config"]["sha256"] == _sha256(config_path)

    mesh = tmp_path / "mesh.ply"
    metadata = tmp_path / "metadata.json"
    mesh.write_text("ply\n", encoding="ascii")
    metadata.write_text("[]", encoding="ascii")
    output_dir = tmp_path / "output"
    output_npz = output_dir / "materials.npz"
    runtime = VompRuntimeConfig(
        runtime_root=root,
        python_executable=Path(sys.executable),
        config_path=config_path,
        expected_revision=revision,
        max_complete_voxels=10,
    )
    monkeypatch.setattr(
        vomp_runtime_module,
        "_validate_runtime",
        lambda _config: (root, Path(sys.executable), config_path),
    )
    worker_fails = False
    worker_returns_nonfinite = False

    def fake_subprocess_run(command: list[str], **kwargs: Any) -> Any:
        assert command[1] == "-I"
        assert "PYTHONPATH" not in kwargs["environment"]
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        output_npz.write_bytes(b"attested npz")
        response = {
            "protocolVersion": 1,
            "status": "success",
            "outputNpzPath": str(output_npz),
            "outputNpzSha256": _sha256(output_npz),
            "sampleCount": 2,
            "completeVoxelField": True,
            "voxelSizeM": 0.01,
            "coordinateUnitMeters": 1.0,
            "coordinateOffsetM": [0.0, 0.0, 0.0],
            "runtime": {"vompRevision": payload["expectedRevision"]},
        }
        if worker_fails:
            response.update({"status": "error", "errorType": "SyntheticError"})
        if worker_returns_nonfinite:
            response["meshTransform"] = {
                "centerM": [0.0, 0.0, 0.0],
                "scaleM": float("nan"),
            }
        response_path.write_text(
            json.dumps(response),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        vomp_runtime_module,
        "_run_worker_process",
        fake_subprocess_run,
    )
    result = ExternalVompRunner(runtime).run(
        VompRunRequest(
            mesh_path=mesh,
            metadata_path=metadata,
            output_dir=output_dir,
            output_npz_path=output_npz,
            num_views=2,
            seed=7,
        )
    )

    assert result.sample_count == 2
    assert result.voxel_size_m == pytest.approx(0.01)
    assert result.output_npz_path == output_npz

    worker_fails = True
    with pytest.raises(VompIntegrationError, match="SyntheticError"):
        ExternalVompRunner(runtime).run(
            VompRunRequest(
                mesh_path=mesh,
                metadata_path=metadata,
                output_dir=output_dir,
                output_npz_path=output_npz,
                num_views=2,
                seed=7,
            )
        )

    worker_fails = False
    worker_returns_nonfinite = True
    with pytest.raises(VompIntegrationError, match="valid response"):
        ExternalVompRunner(runtime).run(
            VompRunRequest(
                mesh_path=mesh,
                metadata_path=metadata,
                output_dir=output_dir,
                output_npz_path=output_npz,
                num_views=2,
                seed=7,
            )
        )


def test_public_surfaces_expose_vomp_commands_and_one_pipeline_step() -> None:
    assert STEP_ORDER == tuple(PIPELINE_STEP_NAMES)
    assert STEP_ORDER is not PIPELINE_STEP_NAMES
    assert isinstance(PIPELINE_STEP_NAMES, list)
    assert isinstance(STEP_ORDER, tuple)
    assert STEP_ORDER[-1] == "vomp_mass"
    defaults = get_step_defaults("vomp_mass")
    assert defaults["enabled"] is False
    assert defaults["render"] == DEFAULT_VOMP_RENDER_CONFIG
    assert VompRenderConfig(**defaults["render"]) == VompRenderConfig()
    assert defaults["expected_revision"] == DEFAULT_VOMP_REVISION
    assert defaults["expected_artifact_sha256"] == DEFAULT_VOMP_ARTIFACT_SHA256
    assert defaults["provenance_path"] is None
    command = get_command(app)
    assert isinstance(command, click.Group)
    run_command = command.commands["run-vomp"]
    deformable_command = command.commands["run-vomp-deformable"]
    option_parameters = [
        parameter
        for parameter in run_command.params
        if isinstance(parameter, click.Option)
    ]
    options = {option for parameter in option_parameters for option in parameter.opts}
    assert "--vomp-root" in options
    assert "--expected-artifact-sha256" in options
    deformable_options = {
        option
        for parameter in deformable_command.params
        if isinstance(parameter, click.Option)
        for option in parameter.opts
    }
    assert {
        "--vomp-root",
        "--material-reduction",
        "--max-deformable-voxels",
    } <= deformable_options
    deformable_option_defaults = {
        parameter.name: parameter.default
        for parameter in deformable_command.params
        if isinstance(parameter, click.Option)
    }
    assert deformable_option_defaults["max_complete_voxels"] == 262_144
    assert deformable_option_defaults["max_deformable_voxels"] == 65_536
    revision_option = next(
        parameter
        for parameter in option_parameters
        if "--expected-revision" in parameter.opts
    )
    assert revision_option.default == DEFAULT_VOMP_REVISION
    option_defaults = {
        parameter.name: parameter.default for parameter in option_parameters
    }
    assert option_defaults["num_views"] == DEFAULT_VOMP_RENDER_CONFIG["num_views"]
    assert option_defaults["image_size"] == DEFAULT_VOMP_RENDER_CONFIG["image_width"]
    assert (
        DEFAULT_VOMP_RENDER_CONFIG["image_width"]
        == DEFAULT_VOMP_RENDER_CONFIG["image_height"]
    )
    assert option_defaults["seed"] == DEFAULT_VOMP_RENDER_CONFIG["seed"]
    assert option_defaults["render_mode"] == DEFAULT_VOMP_RENDER_CONFIG["render_mode"]
    assert (
        option_defaults["num_sensor_updates"]
        == DEFAULT_VOMP_RENDER_CONFIG["num_sensor_updates"]
    )
    assert (
        option_defaults["material_target"]
        == DEFAULT_VOMP_RENDER_CONFIG["material_target"]
    )
    expected_choices = {
        "--render-mode": ("rt1", "rt2", "pt"),
        "--material-target": ("auto", "preview_surface", "openpbr_materialx"),
        "--attention-backend": ("xformers", "sdpa", "naive"),
    }
    for option_name, choices in expected_choices.items():
        option = next(
            parameter
            for parameter in option_parameters
            if option_name in parameter.opts
        )
        assert isinstance(option.type, click.Choice)
        assert tuple(option.type.choices) == choices
    surface_text = " ".join(
        [
            run_command.help or "",
            *options,
            *(parameter.help or "" for parameter in option_parameters),
        ]
    )
    assert "blender" not in surface_text.lower()


def test_integrations_package_is_lazy_for_config_imports() -> None:
    code = """
import sys
import physics_agent.integrations
import physics_agent.config.schema
for name in (
    'physics_agent.integrations.vomp',
    'physics_agent.integrations.vomp_deformable',
    'physics_agent.integrations.vomp_pipeline',
    'physics_agent.integrations.vomp_runtime',
):
    assert name not in sys.modules, f'{name} was imported eagerly'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    integrations_package.__dict__.pop("DEFAULT_VOMP_REVISION", None)
    assert integrations_package.DEFAULT_VOMP_REVISION == DEFAULT_VOMP_REVISION
    integrations_package.__dict__.pop("vomp_defaults", None)
    assert integrations_package.vomp_defaults.DEFAULT_VOMP_REVISION == (
        DEFAULT_VOMP_REVISION
    )
    assert "VompRuntimeConfig" in dir(integrations_package)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = integrations_package.missing_integration


def test_vomp_config_requires_complete_attestation_and_valid_render_values() -> None:
    defaults = get_step_defaults("vomp_mass")
    defaults.update({"target_prim": "/World/Body", "runtime_root": "/opt/VoMP"})
    validator = ConfigValidator()
    validator.validate_step_requirements("vomp_mass", defaults, {})

    missing_hash = dict(defaults)
    missing_hash["expected_artifact_sha256"] = dict(
        defaults["expected_artifact_sha256"]
    )
    missing_hash["expected_artifact_sha256"].pop("config")
    with pytest.raises(ValueError, match="pin every required artifact"):
        validator.validate_step_requirements("vomp_mass", missing_hash, {})

    uppercase_revision = dict(defaults)
    uppercase_revision["expected_revision"] = "A" * 40
    with pytest.raises(ValueError, match="lowercase"):
        validator.validate_step_requirements("vomp_mass", uppercase_revision, {})

    invalid_render = dict(defaults)
    invalid_render["render"] = {**defaults["render"], "fov_degrees": 180.0}
    with pytest.raises(ValueError, match="between 1 and 179"):
        validator.validate_step_requirements("vomp_mass", invalid_render, {})

    unknown_render = dict(defaults)
    unknown_render["render"] = {**defaults["render"], "num_view": 3}
    with pytest.raises(ValueError, match="unsupported key.*num_view"):
        validator.validate_step_requirements("vomp_mass", unknown_render, {})


@pytest.mark.parametrize("value", [1.5, True, "2", 0])
@pytest.mark.parametrize(
    "field",
    [
        "max_complete_voxels",
        "num_views",
        "image_width",
        "image_height",
        "num_sensor_updates",
    ],
)
def test_vomp_config_rejects_non_integer_counts(field: str, value: Any) -> None:
    defaults = get_step_defaults("vomp_mass")
    defaults.update({"target_prim": "/World/Body", "runtime_root": "/opt/VoMP"})
    if field == "max_complete_voxels":
        defaults[field] = value
    else:
        defaults["render"] = {**defaults["render"], field: value}

    with pytest.raises(ValueError, match="positive integer"):
        ConfigValidator().validate_step_requirements("vomp_mass", defaults, {})


def test_unified_config_builds_concrete_vomp_step(tmp_path: Path) -> None:
    source = tmp_path / "input.usda"
    _write_body_stage(source)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
project:
  name: vomp-test
  working_dir: work
input:
  usd_path: input.usda
steps:
  vomp_mass:
    enabled: true
    target_prim: /World/Body
    runtime_root: external/VoMP
    provenance_path: audit/vomp.json
    render:
      ovrtx_venv_dir: external/ovrtx-env
""".lstrip(),
        encoding="ascii",
    )

    result = UnifiedPipelineConfigTask().run(
        {"config_path": str(config_path), "only_steps": ["vomp_mass"]}
    )

    assert result["steps_to_run"] == ["vomp_mass"]
    step = result["step_configs"]["vomp_mass"]
    assert step["usd_path"] == str(source)
    assert step["target_prim"] == "/World/Body"
    assert step["runtime_root"] == str((tmp_path / "external/VoMP").resolve())
    assert step["provenance_path"] == str((tmp_path / "audit/vomp.json").resolve())
    assert step["render"]["ovrtx_venv_dir"] == str(
        (tmp_path / "external/ovrtx-env").resolve()
    )
    assert step["output_usd_path"].endswith("/vomp/input_vomp.usda")
    assert step["work_dir"].endswith("/vomp/artifacts")


def test_vomp_config_and_execution_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "vomp.yaml"
    config_path.write_text(
        """
usd_path: input.usda
output_usd_path: output.usda
target_prim: /World/Body
runtime_root: external/VoMP
python_executable: runtime/bin/python
config_path: weights/custom.json
work_dir: evidence
provenance_path: report.json
render:
  num_views: 3
  image_width: 32
  image_height: 24
  radius: 2.5
  fov_degrees: 45.0
  seed: 9
  render_mode: rt1
  num_sensor_updates: 2
  material_target: preview_surface
  ovrtx_venv_dir: /opt/ovrtx
""".lstrip(),
        encoding="ascii",
    )
    context = VompMassConfigTask().run({"config_path": str(config_path)})
    assert context["runtime_root"] == str((tmp_path / "external/VoMP").resolve())
    assert context["python_executable"].endswith("/external/VoMP/runtime/bin/python")
    assert context["vomp_config_path"].endswith("/external/VoMP/weights/custom.json")
    assert context["provenance_path"] == str((tmp_path / "report.json").resolve())

    captured: dict[str, Any] = {}

    def fake_pipeline(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            apply_result=SimpleNamespace(
                output_usd_path=tmp_path / "output.usda",
                provenance_path=tmp_path / "report.json",
                sample_count=17,
            ),
            evidence=SimpleNamespace(artifact_dir=tmp_path / "evidence/run"),
            vomp_npz_path=tmp_path / "evidence/run/materials.npz",
            worker_manifest_path=tmp_path / "evidence/run/manifest.json",
            worker_log_path=tmp_path / "evidence/run/worker.log",
        )

    monkeypatch.setattr(
        "physics_agent.tasks.vomp_mass.run_vomp_mass_pipeline", fake_pipeline
    )
    result = VompMassTask().run(context)

    assert result["vomp_sample_count"] == 17
    assert result["vomp_npz_path"].endswith("/materials.npz")
    assert captured["kwargs"]["render_config"].num_views == 3
    assert captured["kwargs"]["render_config"].ovrtx_venv_dir == "/opt/ovrtx"
    assert captured["kwargs"]["runtime_config"].expected_artifact_sha256

    context["render"] = {}
    VompMassTask().run(context)
    assert captured["kwargs"]["render_config"] == VompRenderConfig()


def test_vomp_config_task_rejects_missing_values_and_non_mapping_render() -> None:
    task = VompMassConfigTask()
    with pytest.raises(ValueError, match="missing required configuration"):
        task.run({"config_dict": {}})

    with pytest.raises(ValueError, match="render must be a mapping"):
        task.run(
            {
                "config_dict": {
                    "usd_path": "input.usda",
                    "output_usd_path": "output.usda",
                    "target_prim": "/World/Body",
                    "runtime_root": "VoMP",
                    "render": [],
                }
            }
        )

    defaulted = task.run(
        {
            "config_dict": {
                "usd_path": "input.usda",
                "output_usd_path": "output.usda",
                "target_prim": "/World/Body",
                "runtime_root": "VoMP",
            }
        }
    )
    assert defaulted["render"] == {}


def test_vomp_workflow_factory_has_concrete_tasks() -> None:
    from physics_agent.workflows.factory import create_vomp_mass_workflow_from_config

    workflow = create_vomp_mass_workflow_from_config()
    assert [task.name for task in workflow.tasks] == ["VompMassConfig", "VompMass"]


def test_run_vomp_cli_executes_default_and_explicit_runtime_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_pipeline(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            apply_result=SimpleNamespace(
                mass_properties=SimpleNamespace(mass_kg=2.5),
                sample_count=8,
                output_usd_path=tmp_path / "output.usda",
                provenance_path=tmp_path / "report.json",
            ),
            evidence=SimpleNamespace(artifact_dir=tmp_path / "evidence"),
        )

    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline.run_vomp_mass_pipeline",
        fake_pipeline,
    )
    cli = CliRunner()
    default_result = cli.invoke(
        app,
        [
            "run-vomp",
            str(tmp_path / "input.usda"),
            str(tmp_path / "output.usda"),
            "--target-prim",
            "/World/Body",
            "--vomp-root",
            str(tmp_path / "VoMP"),
        ],
    )
    assert default_result.exit_code == 0, default_result.stdout
    assert "2.5 kg" in default_result.stdout

    explicit_result = cli.invoke(
        app,
        [
            "run-vomp",
            str(tmp_path / "input.usda"),
            str(tmp_path / "output.usda"),
            "--target-prim",
            "/World/Body",
            "--vomp-root",
            str(tmp_path / "VoMP"),
            "--vomp-python",
            str(tmp_path / "python"),
            "--vomp-config",
            str(tmp_path / "inference.json"),
            "--work-dir",
            str(tmp_path / "work"),
            "--provenance",
            str(tmp_path / "custom.json"),
            "--expected-artifact-sha256",
            f"config={'c' * 64}",
        ],
    )
    assert explicit_result.exit_code == 0, explicit_result.stdout
    relative_result = cli.invoke(
        app,
        [
            "run-vomp",
            str(tmp_path / "input.usda"),
            str(tmp_path / "output.usda"),
            "--target-prim",
            "/World/Body",
            "--vomp-root",
            str(tmp_path / "VoMP"),
            "--vomp-python",
            ".runtime/bin/python",
        ],
    )
    assert relative_result.exit_code == 0, relative_result.stdout
    assert len(calls) == 3
    assert calls[0]["runtime_config"].python_executable == (
        tmp_path / "VoMP/.venv/bin/python"
    )
    assert calls[1]["runtime_config"].python_executable == tmp_path / "python"
    assert calls[2]["runtime_config"].python_executable == (
        tmp_path / "VoMP/.runtime/bin/python"
    )
    assert calls[0]["runtime_config"].expected_artifact_sha256 == (
        DEFAULT_VOMP_ARTIFACT_SHA256
    )
    assert calls[1]["runtime_config"].expected_artifact_sha256 == {
        **DEFAULT_VOMP_ARTIFACT_SHA256,
        "config": "c" * 64,
    }
    assert calls[1]["work_dir"] == tmp_path / "work"

    invalid_result = cli.invoke(
        app,
        [
            "run-vomp",
            str(tmp_path / "input.usda"),
            str(tmp_path / "output.usda"),
            "--target-prim",
            "/World/Body",
            "--vomp-root",
            str(tmp_path / "VoMP"),
            "--expected-artifact-sha256",
            "unknown=bad",
        ],
    )
    assert invalid_result.exit_code == 1
    assert "valid keys" in invalid_result.stdout


def test_run_vomp_deformable_cli_routes_runtime_and_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_pipeline(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            apply_result=SimpleNamespace(
                sample_count=8,
                point_count=27,
                tet_count=48,
                mass_kg=2.5,
                material_reduction=SimpleNamespace(validation_status="conditional"),
                output_usd_path=tmp_path / "output.usda",
                provenance_path=tmp_path / "report.json",
            ),
            evidence=SimpleNamespace(artifact_dir=tmp_path / "evidence"),
        )

    monkeypatch.setattr(
        "physics_agent.integrations.vomp_pipeline.run_vomp_volume_deformable_pipeline",
        fake_pipeline,
    )
    result = CliRunner().invoke(
        app,
        [
            "run-vomp-deformable",
            str(tmp_path / "input.usda"),
            str(tmp_path / "output.usda"),
            "--target-prim",
            "/World/Body",
            "--vomp-root",
            str(tmp_path / "VoMP"),
            "--material-reduction",
            "homogeneous-volume-average",
            "--max-deformable-voxels",
            "99",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "2.5 kg" in result.stdout
    assert "conditional" in result.stdout
    assert len(calls) == 1
    assert calls[0]["material_reduction"] == "homogeneous-volume-average"
    assert calls[0]["max_deformable_voxels"] == 99
    assert calls[0]["runtime_config"].python_executable == (
        tmp_path / "VoMP/.venv/bin/python"
    )
