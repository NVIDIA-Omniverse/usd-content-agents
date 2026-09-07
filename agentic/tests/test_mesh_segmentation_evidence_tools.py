# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for closest-visible ID and deterministic initialization tools."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pxr import Usd, UsdGeom, Vt

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_SCRIPTS = (
    REPO_ROOT / "agentic/.agents/skills/content-workflow-mesh-segmentation/scripts"
)
CANONICAL_SCRIPTS = (
    REPO_ROOT / "agentic/.agents/skills/content-workflow-mesh-segmentation/scripts"
)
CANONICAL_SKILL = (
    REPO_ROOT / "agentic/.agents/skills/content-workflow-mesh-segmentation"
)
PYTHON = Path(sys.executable)


def _script_env() -> dict[str, str]:
    env = os.environ.copy()
    roots = [str(EVIDENCE_SCRIPTS), str(CANONICAL_SCRIPTS)]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return env


def _write_triangle_usd(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/FusedMesh")
    mesh.CreatePointsAttr().Set(
        Vt.Vec3fArray(
            [
                (-1.0, -1.0, -5.0),
                (1.0, -1.0, -5.0),
                (0.0, 1.0, -5.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()


def test_appearance_images_prefers_the_longest_view_suffix(tmp_path: Path) -> None:
    sys.path.insert(0, str(CANONICAL_SCRIPTS))
    try:
        sys.modules.pop("build_topology_component_evidence", None)
        module = importlib.import_module("build_topology_component_evidence")
    finally:
        sys.path.pop(0)

    image_path = tmp_path / "source-upper-right.png"
    Image.new("RGB", (4, 4), "white").save(image_path)

    assert module._appearance_images(tmp_path, {"right", "upper-right"}) == {
        "upper-right": image_path.resolve()
    }


def _write_disconnected_triangles_usd(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/FusedMesh")
    mesh.CreatePointsAttr().Set(
        Vt.Vec3fArray(
            [
                (-2.0, -1.0, 0.0),
                (-1.0, -1.0, 0.0),
                (-1.5, 1.0, 0.0),
                (1.0, -1.0, 0.0),
                (2.0, -1.0, 0.0),
                (1.5, 1.0, 0.0),
                (-0.25, -0.25, 1.0),
                (0.25, -0.25, 1.0),
                (0.0, 0.25, 1.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3, 3, 3]))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3, 4, 5, 6, 7, 8]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()


def test_topology_component_evidence_maps_visible_shells_to_fragments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "disconnected.usda"
    _write_disconnected_triangles_usd(source)
    sys.path.insert(0, str(CANONICAL_SCRIPTS))
    try:
        mesh_geometry = importlib.import_module("mesh_geometry")
        mesh_data = mesh_geometry.load_usd(source, "/World/FusedMesh")
    finally:
        sys.path.pop(0)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    fragment_labels = tmp_path / "fragment_ids.u32le"
    np.asarray([0, 1, 2], dtype="<u4").tofile(fragment_labels)
    fragment_labels_sha256 = hashlib.sha256(fragment_labels.read_bytes()).hexdigest()

    neutral_dir = tmp_path / "neutral"
    appearance_dir = tmp_path / "references"
    id_dir = tmp_path / "ids"
    neutral_dir.mkdir()
    appearance_dir.mkdir()
    id_dir.mkdir()
    neutral = neutral_dir / "front.png"
    Image.new("RGB", (8, 8), (96, 96, 96)).save(neutral)
    appearance = appearance_dir / "01_front.png"
    Image.new("RGB", (8, 8), (180, 24, 24)).save(appearance)
    face_ids = np.full((8, 8), -1, dtype=np.int32)
    face_ids[2:6, 1:3] = 0
    face_ids[2:6, 5:7] = 1
    face_ids_path = id_dir / "front_face_ids.npy"
    np.save(face_ids_path, face_ids, allow_pickle=False)
    camera = neutral_dir / "front_camera.json"
    camera.write_text('{"view":"front"}', encoding="utf-8")
    camera_sha256 = hashlib.sha256(camera.read_bytes()).hexdigest()
    id_manifest = id_dir / "manifest.json"
    id_manifest.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-id-buffers.v1",
                "source_sha256": source_sha256,
                "source_face_count": 3,
                "target_prim_path": "/World/FusedMesh",
                "topology_digest": mesh_data.topology_digest,
                "fragment_labels_sha256": fragment_labels_sha256,
                "views": [
                    {
                        "name": "front",
                        "camera": str(camera),
                        "camera_sha256": camera_sha256,
                        "channels": {"face_ids": {"raw": str(face_ids_path)}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    render_manifest = neutral_dir / "render_manifest.json"
    render_manifest.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-render-evidence.v1",
                "scene_tool": "usd-cli",
                "scene_sha256": source_sha256,
                "renders": [
                    {
                        "name": "front",
                        "image": str(neutral),
                        "camera": str(camera),
                        "camera_sha256": camera_sha256,
                        "renderer": {"backend": "ovrtx-test"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "components"

    subprocess.run(
        [
            str(PYTHON),
            str(CANONICAL_SCRIPTS / "build_topology_component_evidence.py"),
            "--source-usd",
            str(source),
            "--target",
            "/World/FusedMesh",
            "--fragment-labels",
            str(fragment_labels),
            "--id-buffer-manifest",
            str(id_manifest),
            "--neutral-render-manifest",
            str(render_manifest),
            "--appearance-dir",
            str(appearance_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    statistics = json.loads(
        (output_dir / "component_statistics.json").read_text(encoding="utf-8")
    )
    fragment_map = json.loads(
        (output_dir / "component_fragment_map.json").read_text(encoding="utf-8")
    )
    overlay_manifest = json.loads(
        (output_dir / "overlay_manifest.json").read_text(encoding="utf-8")
    )
    gallery_manifest = json.loads(
        (output_dir / "component_gallery_manifest.json").read_text(encoding="utf-8")
    )
    assert statistics["component_count"] == 3
    assert [record["face_count"] for record in statistics["components"]] == [1, 1, 1]
    assert [record["fragment_ids"] for record in fragment_map["components"]] == [
        [0],
        [1],
        [2],
    ]
    assert np.array_equal(
        np.fromfile(output_dir / "component_ids.u32le", dtype="<u4"),
        np.asarray([0, 1, 2], dtype=np.uint32),
    )
    assert overlay_manifest["views"][0]["visible_component_ids"] == [0, 1]
    assert overlay_manifest["views"][0]["appearance_image"] == str(appearance)
    assert (output_dir / "overlays/front.png").is_file()
    assert [record["component_id"] for record in gallery_manifest["components"]] == [
        0,
        1,
        2,
    ]
    assert all(
        Path(record["crop"]).is_file() for record in gallery_manifest["components"]
    )
    assert gallery_manifest["presentation"] == (
        "source_appearance_zoom_with_local_and_full_registered_source_context"
    )
    assert gallery_manifest["components"][0]["appearance_image"] == str(appearance)
    assert all(
        isinstance(record["nearby_similar_scale_component_ids"], list)
        for record in gallery_manifest["components"]
    )
    appearance_crop = np.asarray(
        Image.open(gallery_manifest["components"][0]["crop"]).convert("RGB")
    )
    assert np.any(
        (appearance_crop[..., 0] > 140)
        & (appearance_crop[..., 0] > appearance_crop[..., 1] * 3)
    )
    assert gallery_manifest["card_size"] == [640, 384]
    assert gallery_manifest["source_sha256"] == source_sha256
    assert gallery_manifest["topology_digest"] == mesh_data.topology_digest
    assert gallery_manifest["fragment_labels_sha256"] == fragment_labels_sha256
    assert gallery_manifest["registered_views"] == [
        {
            "view_id": "front",
            "camera_sha256": camera_sha256,
            "renderer": {"backend": "ovrtx-test"},
        }
    ]
    assert gallery_manifest["contact_sheets"]
    assert gallery_manifest["contact_sheet_artifacts"]
    assert all(Path(path).is_file() for path in gallery_manifest["contact_sheets"])
    occluded = gallery_manifest["components"][2]
    assert occluded["visible_pixel_count"] == 0
    assert occluded["evidence_mode"] == "diagnostic_only_local_preview"
    assert occluded["semantic_decision_allowed"] is False
    assert occluded["isolated_view_id"]
    occluded_pixels = np.asarray(Image.open(occluded["crop"]).convert("RGB"))
    assert np.count_nonzero(occluded_pixels) > 0

    stale_id_manifest = json.loads(id_manifest.read_text(encoding="utf-8"))
    stale_id_manifest["views"][0]["camera_sha256"] = "0" * 64
    id_manifest.write_text(json.dumps(stale_id_manifest), encoding="utf-8")
    stale = subprocess.run(
        [
            str(PYTHON),
            str(CANONICAL_SCRIPTS / "build_topology_component_evidence.py"),
            "--source-usd",
            str(source),
            "--target",
            "/World/FusedMesh",
            "--fragment-labels",
            str(fragment_labels),
            "--id-buffer-manifest",
            str(id_manifest),
            "--neutral-render-manifest",
            str(render_manifest),
            "--output-dir",
            str(tmp_path / "stale-components"),
        ],
        check=False,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    assert stale.returncode != 0
    assert "Camera digest differs" in stale.stderr


def test_component_assignment_validation_requires_exact_coverage(
    tmp_path: Path,
) -> None:
    gallery = tmp_path / "gallery.json"
    gallery.write_text(
        json.dumps(
            {
                "components": [
                    {"component_id": 0},
                    {"component_id": 1},
                    {"component_id": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "assignments": [{"semantic_part": "body", "component_ids": [0, 1]}],
                "unresolved_components": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "validation.json"
    command = [
        str(PYTHON),
        str(CANONICAL_SCRIPTS / "validate_component_assignment_review.py"),
        "--gallery-manifest",
        str(gallery),
        "--review",
        str(review),
        "--output",
        str(output),
    ]

    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8"))["missing_component_ids"] == [
        2
    ]

    review.write_text(
        json.dumps(
            {
                "assignments": [{"semantic_part": "body", "component_ids": [0, 1]}],
                "unresolved_components": [2],
            }
        ),
        encoding="utf-8",
    )
    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert passed.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"


def test_warp_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(EVIDENCE_SCRIPTS))
    warp_raycast = importlib.import_module("warp_raycast")
    monkeypatch.setattr(warp_raycast.wp, "init", lambda: None)
    monkeypatch.setattr(warp_raycast.wp, "is_cuda_available", lambda: False)
    monkeypatch.setattr(warp_raycast.wp, "get_device", lambda name: name)

    assert warp_raycast.resolve_warp_device("auto") == "cpu"


def test_render_mesh_evidence_defaults_cover_all_cube_corners(
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(CANONICAL_SCRIPTS))
    sys.modules.pop("render_mesh_evidence", None)
    render_mesh_evidence = importlib.import_module("render_mesh_evidence")

    assert len(render_mesh_evidence.DEFAULT_Z_UP_CAMERAS) == 8
    assert len(set(render_mesh_evidence.DEFAULT_Z_UP_CAMERAS)) == 8
    assert len(render_mesh_evidence.DEFAULT_Y_UP_CAMERAS) == 8
    assert len(set(render_mesh_evidence.DEFAULT_Y_UP_CAMERAS)) == 8
    assert any(
        camera.endswith("-z") for camera in render_mesh_evidence.DEFAULT_Z_UP_CAMERAS
    )
    assert any("-y" in camera for camera in render_mesh_evidence.DEFAULT_Y_UP_CAMERAS)


def test_render_mesh_evidence_preserves_metric_depth_and_separate_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(CANONICAL_SCRIPTS))
    sys.modules.pop("render_mesh_evidence", None)
    render_mesh_evidence = importlib.import_module("render_mesh_evidence")
    source_dir = tmp_path / "usd-cli"
    source_dir.mkdir()
    raw_source = source_dir / "linear_depth.npy"
    preview_source = source_dir / "depth.png"
    expected = np.asarray([[1.25, np.nan], [2.5, 5.0]], dtype=np.float32)
    np.save(raw_source, expected, allow_pickle=False)
    Image.fromarray(np.asarray([[255, 0], [128, 64]], dtype=np.uint8)).save(
        preview_source
    )

    channel = render_mesh_evidence._save_metric_depth(
        raw_source,
        preview_source,
        unit="meter",
    )

    preserved = np.load(channel["raw"], allow_pickle=False)
    np.testing.assert_array_equal(preserved, expected)
    assert Path(channel["raw"]).read_bytes() == raw_source.read_bytes()
    assert channel["dtype"] == "float32"
    assert channel["unit"] == "meter"
    assert channel["evidence_role"] == "auxiliary_cpu_aov"
    assert channel["final_render_evidence"] is False
    assert channel["preview_encoding"] == "per_frame_normalized_uint8"


def test_render_mesh_evidence_rejects_normalized_png_as_metric_depth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(CANONICAL_SCRIPTS))
    sys.modules.pop("render_mesh_evidence", None)
    render_mesh_evidence = importlib.import_module("render_mesh_evidence")
    normalized_png = tmp_path / "depth.png"
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    Image.new("L", (2, 2), 128).save(normalized_png)

    with pytest.raises(RuntimeError, match="refusing to relabel"):
        render_mesh_evidence._save_metric_depth(
            normalized_png,
            normalized_png,
            unit="meter",
        )

    assert not (output_dir / "front_linear_depth.npy").exists()


def test_render_mesh_evidence_uses_attested_route_and_sealed_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(CANONICAL_SCRIPTS))
    sys.modules.pop("render_mesh_evidence", None)
    render_mesh_evidence = importlib.import_module("render_mesh_evidence")
    project_dir = tmp_path / "project"
    evidence_dir = project_dir / "evidence"
    project_dir.mkdir()
    evidence_dir.mkdir()

    class FakeRoute:
        wrapper = tmp_path / "owned" / "usd-cli-tel"
        target = tmp_path / "owned" / "usd-cli"
        source_revision = "abc123"

    route = FakeRoute()
    monkeypatch.setattr(render_mesh_evidence, "_package_owned_route", lambda: route)
    monkeypatch.setattr(
        render_mesh_evidence,
        "controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {"CONTROLLED": "1"},
    )
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ok": True, "artifacts": []}),
            stderr="",
        )

    monkeypatch.setattr(
        render_mesh_evidence,
        "run_bounded_usd_cli_subprocess",
        fake_run,
    )
    session = render_mesh_evidence._MeshEvidenceUsdCliSession.create(
        project_dir=project_dir,
        session_id="workflow-test",
        evidence_dir=evidence_dir,
    )

    assert session.run_json(["open", "scene.usd"])["ok"] is True
    assert calls[0]["command"] == [
        str(route.wrapper),
        "--json",
        "--session",
        "workflow-test",
        "open",
        "scene.usd",
    ]
    assert calls[0]["env"] == {
        "CONTROLLED": "1",
        "USD_CLI_LOCK_RENDER_CONFIG": "1",
    }
    receipts = [
        json.loads(line)
        for line in session.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert receipts[0]["status"] == "completed"
    assert receipts[0]["tool"] == {
        "name": "usd-cli",
        "source_revision": "abc123",
    }
    checkpoint = json.loads(session.checkpoint_path.read_text(encoding="utf-8"))
    assert (
        checkpoint["receipt_sha256"]
        == hashlib.sha256(session.journal_path.read_bytes()).hexdigest()
    )
    assert checkpoint["receipt_size_bytes"] == session.journal_path.stat().st_size

    session.journal_path.write_text("tampered\n", encoding="utf-8")
    os.chmod(session.journal_path, 0o600)
    with pytest.raises(RuntimeError, match="receipt journal changed"):
        session.run_json(["snapshot"])
    assert len(calls) == 1


def test_render_mesh_evidence_resolves_route_from_installed_workflow_package(
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(CANONICAL_SCRIPTS))
    sys.modules.pop("render_mesh_evidence", None)
    render_mesh_evidence = importlib.import_module("render_mesh_evidence")
    expected_route = object()
    starts: list[Path] = []

    def fake_find(start: Path) -> Path:
        starts.append(start)
        return REPO_ROOT

    monkeypatch.setattr(
        render_mesh_evidence.usd_cli_backend,
        "find_usd_cli_repository_root",
        fake_find,
    )
    monkeypatch.setattr(
        render_mesh_evidence.usd_cli_backend,
        "resolve_package_owned_usd_cli_route",
        lambda root: expected_route if root == REPO_ROOT else None,
    )

    assert render_mesh_evidence._package_owned_route() is expected_route
    assert starts == [Path(render_mesh_evidence.usd_cli_backend.__file__).resolve()]


def test_render_mesh_evidence_main_emits_force_reload_and_quality_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(CANONICAL_SCRIPTS))
    sys.modules.pop("render_mesh_evidence", None)
    render_mesh_evidence = importlib.import_module("render_mesh_evidence")
    scene = tmp_path / "scene.usda"
    _write_triangle_usd(scene)
    output_dir = tmp_path / "evidence"
    (tmp_path / "request.json").write_text(
        json.dumps({"runtime": {"usd_cli_session_id": "workflow-test"}}),
        encoding="utf-8",
    )

    class FakeRoute:
        wrapper = tmp_path / "owned" / "usd-cli-tel"
        target = tmp_path / "owned" / "usd-cli"
        source_revision = "abc123"

    command_arguments: list[list[str]] = []

    def fake_run(command, **_kwargs):
        arguments = list(command[4:])
        command_arguments.append(arguments)
        if arguments[0] == "render-probe":
            payload = {
                "schema_version": "usd-cli.render-probe.v1",
                "ready": True,
            }
        elif arguments[0] == "render":
            render_dir = Path(arguments[arguments.index("--output") + 1])
            render_dir.mkdir(parents=True)
            rgb = render_dir / "rgb.png"
            normals = render_dir / "normals.png"
            depth = render_dir / "depth.png"
            linear_depth = render_dir / "linear_depth.npy"
            Image.new("RGB", (2, 2), "white").save(rgb)
            Image.new("RGB", (2, 2), (128, 128, 255)).save(normals)
            Image.new("L", (2, 2), 128).save(depth)
            np.save(linear_depth, np.ones((2, 2), dtype=np.float32))
            payload = {
                "ok": True,
                "summary": {
                    "backend": "ovrtx",
                    "ovrtx_render_mode": "rt2",
                    "ovrtx_num_sensor_updates": 64,
                    "active_aov": "LdrColor",
                },
                "data": {"linear_depth_unit": "meter"},
                "artifacts": [
                    {"label": "rgb:/World/usd_cam", "path": str(rgb)},
                    {"label": "normals", "path": str(normals)},
                    {"label": "depth", "path": str(depth)},
                    {"label": "linear_depth", "path": str(linear_depth)},
                ],
            }
        else:
            payload = {"ok": True}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(render_mesh_evidence, "_package_owned_route", FakeRoute)
    monkeypatch.setattr(
        render_mesh_evidence,
        "controlled_usd_cli_telemetry_env",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        render_mesh_evidence,
        "run_bounded_usd_cli_subprocess",
        fake_run,
    )
    monkeypatch.setattr(
        render_mesh_evidence,
        "parse_args",
        lambda: SimpleNamespace(
            scene=scene,
            focus="/World/FusedMesh",
            cameras=["+x+y+z"],
            camera_json=None,
            width=2,
            height=2,
            output_dir=output_dir,
        ),
    )

    render_mesh_evidence.main()

    open_arguments = next(args for args in command_arguments if args[0] == "open")
    render_arguments = next(args for args in command_arguments if args[0] == "render")
    assert open_arguments == ["open", str(scene.resolve()), "--force-reload"]
    mode_index = render_arguments.index("--mode")
    assert render_arguments[mode_index : mode_index + 2] == ["--mode", "quality"]
    manifest = json.loads(
        (output_dir / "render_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["renders"][0]["ovrtx_render_mode"] == "rt2"
    assert manifest["renders"][0]["ovrtx_num_sensor_updates"] == 64


def test_render_mesh_evidence_loads_payloaded_focus_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(CANONICAL_SCRIPTS))
    sys.modules.pop("render_mesh_evidence", None)
    render_mesh_evidence = importlib.import_module("render_mesh_evidence")
    payload_path = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    UsdGeom.Xform.Define(payload_stage, "/Payload")
    UsdGeom.Mesh.Define(payload_stage, "/Payload/FusedMesh")
    payload_stage.GetRootLayer().defaultPrim = "Payload"
    payload_stage.Save()
    scene = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    root = UsdGeom.Xform.Define(stage, "/World")
    root.GetPrim().GetPayloads().AddPayload(str(payload_path), "/Payload")
    stage.Save()

    opened, _up_axis_y = render_mesh_evidence._open_stage(scene)

    assert opened.GetPrimAtPath("/World/FusedMesh").IsValid()


@pytest.mark.parametrize(
    ("field", "value"),
    (("ovrtx_render_mode", None), ("ovrtx_num_sensor_updates", None)),
)
def test_render_mesh_evidence_rejects_missing_effective_ovrtx_metadata(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    monkeypatch.syspath_prepend(str(CANONICAL_SCRIPTS))
    sys.modules.pop("render_mesh_evidence", None)
    render_mesh_evidence = importlib.import_module("render_mesh_evidence")
    summary = {
        "backend": "ovrtx",
        "ovrtx_render_mode": "rt2",
        "ovrtx_num_sensor_updates": 64,
        "active_aov": "LdrColor",
    }
    summary[field] = value

    with pytest.raises(RuntimeError, match="complete executed OVRTX metadata"):
        render_mesh_evidence.validated_ovrtx_render_metadata({"summary": summary})


def test_render_face_id_buffer_uses_closest_visible_hit(tmp_path: Path) -> None:
    source = tmp_path / "triangle.usda"
    _write_triangle_usd(source)
    fragment_labels = tmp_path / "fragment_ids.u32le"
    np.asarray([0], dtype="<u4").tofile(fragment_labels)
    face_labels = tmp_path / "face_labels.u32le"
    np.asarray([7], dtype="<u4").tofile(face_labels)
    camera = tmp_path / "front_camera.json"
    camera.write_text(
        json.dumps(
            {
                "image_width": 3,
                "image_height": 3,
                "camera_state": {
                    "horizontal_aperture": 2.0,
                    "focal_length": 1.0,
                },
                "camera_world_transform": np.eye(4).tolist(),
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "ids"

    subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "render_face_id_buffers.py"),
            "--source-usd",
            str(source),
            "--target",
            "/World/FusedMesh",
            "--fragment-labels",
            str(fragment_labels),
            "--face-labels",
            str(face_labels),
            "--active-segment-id",
            "7",
            "--camera-json",
            str(camera),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    face_ids = np.load(output_dir / "front_face_ids.npy", allow_pickle=False)
    fragment_ids = np.load(output_dir / "front_fragment_ids.npy", allow_pickle=False)
    label_ids = np.load(output_dir / "front_label_ids.npy", allow_pickle=False)
    assert face_ids[1, 1] == 0
    assert fragment_ids[1, 1] == 0
    assert label_ids[1, 1] == 7
    assert np.count_nonzero(face_ids >= 0) == 1
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["views"][0]["closest_visible_hit_only"] is True
    assert manifest["views"][0]["visible_face_count"] == 1


def test_deterministic_mask_projection_is_independent_then_exact_union(
    tmp_path: Path,
) -> None:
    id_dir = tmp_path / "id"
    id_dir.mkdir()
    front_fragment_ids = np.tile(
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        (8, 1),
    )
    back_fragment_ids = np.tile(
        np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
        (8, 1),
    )
    front_fragment_path = id_dir / "front_fragment_ids.npy"
    back_fragment_path = id_dir / "back_fragment_ids.npy"
    np.save(front_fragment_path, front_fragment_ids, allow_pickle=False)
    np.save(back_fragment_path, back_fragment_ids, allow_pickle=False)
    id_manifest = id_dir / "manifest.json"
    id_manifest.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-id-buffers.v1",
                "views": [
                    {
                        "name": "front",
                        "closest_visible_hit_only": True,
                        "channels": {"fragment_ids": {"raw": str(front_fragment_path)}},
                    },
                    {
                        "name": "back",
                        "closest_visible_hit_only": True,
                        "channels": {"fragment_ids": {"raw": str(back_fragment_path)}},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    registration_dir = tmp_path / "registered"
    registration_dir.mkdir()
    source_dir = tmp_path / "neutral"
    source_dir.mkdir()
    front_mask_path = registration_dir / "front_aligned_mask.png"
    front_mask = np.zeros((8, 8), dtype=np.uint8)
    front_mask[:, :5] = 255
    Image.fromarray(front_mask, mode="L").save(front_mask_path)
    back_mask_path = registration_dir / "back_aligned_mask.png"
    back_mask = np.zeros((8, 8), dtype=np.uint8)
    back_mask[:, 3:] = 255
    Image.fromarray(back_mask, mode="L").save(back_mask_path)
    for view_id in ("front", "back"):
        Image.new("RGB", (8, 8), (128, 128, 128)).save(source_dir / f"{view_id}.png")
    registration_manifest = registration_dir / "manifest.json"
    registration_manifest.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-semantic-registration.v1",
                "views": [
                    {
                        "view_id": "front",
                        "status": "accepted",
                        "accepted": True,
                        "registered_mask": str(front_mask_path),
                        "source_render": str(source_dir / "front.png"),
                    },
                    {
                        "view_id": "back",
                        "status": "accepted",
                        "accepted": True,
                        "registered_mask": str(back_mask_path),
                        "source_render": str(source_dir / "back.png"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    fragment_labels = tmp_path / "fragment_ids.u32le"
    np.asarray([0, 1, 2, 2], dtype="<u4").tofile(fragment_labels)
    parent_labels = tmp_path / "parent_labels.u32le"
    np.zeros(4, dtype="<u4").tofile(parent_labels)
    union_dir = tmp_path / "union"
    subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "build_deterministic_fragment_union.py"),
            "--registration-manifest",
            str(registration_manifest),
            "--id-buffer-manifest",
            str(id_manifest),
            "--fragment-labels",
            str(fragment_labels),
            "--parent-labels",
            str(parent_labels),
            "--active-segment-id",
            "7",
            "--expected-view-count",
            "2",
            "--output-dir",
            str(union_dir),
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    union = json.loads((union_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((union_dir / "validation.json").read_text(encoding="utf-8"))
    edits = json.loads((union_dir / "edits.json").read_text(encoding="utf-8"))
    expected_labels = np.fromfile(
        union_dir / "expected_face_labels.u32le",
        dtype="<u4",
    )
    assert union["method"] == (
        "eight_view_two_pixel_eroded_closest_visible_fragment_set_union"
    )
    assert union["view_order"] == ["front", "back"]
    assert union["cross_view_reduction"] == "exact_set_union"
    assert union["negative_pixels_used"] is False
    assert union["cross_view_voting_used"] is False
    assert np.load(
        union_dir / "views/front/selected_fragment_ids.npy",
        allow_pickle=False,
    ).tolist() == [0]
    assert np.load(
        union_dir / "views/back/selected_fragment_ids.npy",
        allow_pickle=False,
    ).tolist() == [2]
    assert np.load(
        union_dir / "union_fragment_ids.npy",
        allow_pickle=False,
    ).tolist() == [0, 2]
    assert edits["edits"][0]["fragment_ids"] == [0, 2]
    assert expected_labels.tolist() == [7, 0, 7, 7]
    assert validation["set_union_replay_verified"] is True
    assert validation["agent_confirmation_before_rev_000"] is False


def test_region_mapping_is_initial_evidence_only(tmp_path: Path) -> None:
    id_dir = tmp_path / "id"
    id_dir.mkdir()
    fragment_ids = np.tile(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        (4, 1),
    )
    fragment_path = id_dir / "front_fragment_ids.npy"
    np.save(fragment_path, fragment_ids, allow_pickle=False)
    id_manifest = id_dir / "manifest.json"
    id_manifest.write_text(
        json.dumps(
            {
                "views": [
                    {
                        "name": "front",
                        "channels": {"fragment_ids": {"raw": str(fragment_path)}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    regions_path = tmp_path / "regions.json"
    regions_path.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "view_id": "front",
                        "polarity": "positive",
                        "shape": "box",
                        "box": [0, 0, 1, 3],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    region_output = tmp_path / "region_evidence.json"
    subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "regions_to_fragment_evidence.py"),
            "--id-buffer-manifest",
            str(id_manifest),
            "--regions",
            str(regions_path),
            "--output",
            str(region_output),
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    region_evidence = json.loads(region_output.read_text(encoding="utf-8"))
    assert region_evidence["closest_visible_hit_only"] is True
    assert region_evidence["events"][0]["candidate_fragment_ids"] == [0]


def test_build_revision_consistency_sheet_makes_standardized_crops(
    tmp_path: Path,
) -> None:
    neutral_dir = tmp_path / "neutral"
    flat_dir = tmp_path / "flat"
    selected_dir = tmp_path / "selected"
    neutral_dir.mkdir()
    flat_dir.mkdir()
    selected_dir.mkdir()
    for view_id, color in (("front", (255, 0, 255)), ("back", (0, 255, 255))):
        Image.new("RGB", (320, 240), (128, 128, 128)).save(
            neutral_dir / f"{view_id}.png"
        )
        Image.new("RGB", (320, 240), color).save(flat_dir / f"{view_id}.png")
        Image.new("RGB", (320, 240), (32, 32, 32)).save(selected_dir / f"{view_id}.png")
    neutral_manifest = neutral_dir / "render_manifest.json"
    flat_manifest = flat_dir / "render_manifest.json"
    selected_manifest = selected_dir / "render_manifest.json"
    neutral_manifest.write_text(
        json.dumps(
            {
                "renders": [
                    {
                        "name": view_id,
                        "image": str(neutral_dir / f"{view_id}.png"),
                    }
                    for view_id in ("front", "back")
                ]
            }
        ),
        encoding="utf-8",
    )
    flat_manifest.write_text(
        json.dumps(
            {
                "renders": [
                    {"name": view_id, "image": str(flat_dir / f"{view_id}.png")}
                    for view_id in ("front", "back")
                ]
            }
        ),
        encoding="utf-8",
    )
    selected_manifest.write_text(
        json.dumps(
            {
                "renders": [
                    {
                        "name": view_id,
                        "image": str(selected_dir / f"{view_id}.png"),
                    }
                    for view_id in ("front", "back")
                ]
            }
        ),
        encoding="utf-8",
    )
    regions = tmp_path / "regions.json"
    regions.write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-consistency-regions.v1"),
                "instances": [
                    {
                        "instance_id": "wheel-a",
                        "views": [
                            {
                                "view_id": "front",
                                "surface_role": "primary_surface",
                                "bbox": [40, 30, 180, 190],
                            },
                            {
                                "view_id": "back",
                                "surface_role": ("opposing_or_occlusion_revealing"),
                                "bbox": [100, 40, 260, 210],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "consistency"
    subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "build_revision_consistency_sheet.py"),
            "--neutral-manifest",
            str(neutral_manifest),
            "--flat-label-manifest",
            str(flat_manifest),
            "--selected-only-manifest",
            str(selected_manifest),
            "--regions",
            str(regions),
            "--output-dir",
            str(output_dir),
            "--crop-size",
            "256",
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["crop_size"] == [256, 256]
    assert manifest["layout"]["mode"] == "tiled_peer_grid"
    assert manifest["layout"]["columns"] == 1
    assert manifest["layout"]["rows"] == 1
    assert manifest["layout"]["instance_order"] == ["wheel-a"]
    assert len(manifest["instances"][0]["evidence_crops"]) == 2
    assert manifest["context_layout"]["mode"] == "full_view_triptych_pages"
    assert manifest["context_layout"]["channels"] == [
        "neutral",
        "flat_label",
        "selected_only",
    ]
    assert manifest["context_layout"]["view_order"] == ["front", "back"]
    assert len(manifest["context_pages"]) == 1
    assert Path(manifest["context_pages"][0]["image"]).is_file()
    for crop in manifest["instances"][0]["evidence_crops"]:
        with Image.open(crop["flat_label_crop"]) as image:
            assert image.size == (256, 256)
        with Image.open(crop["selected_only_crop"]) as image:
            assert image.size == (256, 256)
    assert Path(manifest["comparison_sheet"]).is_file()


def test_initializer_decision_records_direct_route_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    part_dir = run_dir / "vision_glass"
    evidence_dir = part_dir / "initializer-probe"
    evidence_dir.mkdir(parents=True)
    artifacts: dict[str, list[str]] = {
        "registration_manifests": [],
        "registered_masks": [],
        "chosen_pixel_overlays": [],
        "fragment_projections": [],
    }
    registration = evidence_dir / "registration.json"
    registration.write_text("{}\n", encoding="utf-8")
    artifacts["registration_manifests"].append(str(registration.relative_to(part_dir)))
    projection_manifest = evidence_dir / "projection.json"
    projection_manifest.write_text("{}\n", encoding="utf-8")
    for index in range(3):
        for role in (
            "registered_masks",
            "chosen_pixel_overlays",
            "fragment_projections",
        ):
            path = evidence_dir / f"{role}-{index}.png"
            Image.new("RGB", (2, 2), (index, index, index)).save(path)
            artifacts[role].append(str(path.relative_to(part_dir)))

    decision_path = part_dir / "initializer_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-initializer-decision.v1"),
                "semantic_part": "vision_glass",
                "segment_id": 3,
                "status": "accepted",
                "decision": "direct_agentic_selection",
                "probe_view_ids": ["a", "b", "c"],
                "evidence": {
                    **artifacts,
                    "projection_manifest": str(
                        projection_manifest.relative_to(part_dir)
                    ),
                },
                "assessment": {
                    "semantic_quality": "mixed",
                    "mask_scale": "marginal",
                    "eroded_interior_support": "marginal",
                    "cross_view_consistency": "mixed",
                    "mesh_projection_quality": "leaky",
                    "confuser_leakage": "major",
                    "rationale": "The projection amplifies small mask errors.",
                },
                "next_step": "direct_signed_fragment_selection",
            }
        ),
        encoding="utf-8",
    )
    output_path = part_dir / "initializer_decision_validation.json"
    subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_initializer_decision.py"),
            "--run-dir",
            str(run_dir),
            "--decision",
            str(decision_path),
            "--output",
            str(output_path),
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    validation = json.loads(output_path.read_text(encoding="utf-8"))
    assert validation["status"] == "passed"
    assert validation["decision"] == "direct_agentic_selection"
    assert validation["semantic_part"] == "vision_glass"


def test_initializer_decision_rejects_bad_mask_route(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    part_dir = run_dir / "glass"
    part_dir.mkdir(parents=True)
    artifact = part_dir / "evidence.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(artifact)
    decision_path = part_dir / "initializer_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-initializer-decision.v1"),
                "semantic_part": "glass",
                "segment_id": 1,
                "status": "accepted",
                "decision": "image_mask_seed",
                "probe_view_ids": ["a", "b", "c"],
                "evidence": {
                    "registration_manifests": ["evidence.png"],
                    "registered_masks": ["evidence.png"] * 3,
                    "chosen_pixel_overlays": ["evidence.png"] * 3,
                    "fragment_projections": ["evidence.png"] * 3,
                    "projection_manifest": "evidence.png",
                },
                "assessment": {
                    "semantic_quality": "poor",
                    "mask_scale": "too_small",
                    "eroded_interior_support": "empty",
                    "cross_view_consistency": "inconsistent",
                    "mesh_projection_quality": "leaky",
                    "confuser_leakage": "major",
                    "rationale": "No usable semantic support.",
                },
                "next_step": "eight_view_registered_mask_seed",
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_initializer_decision.py"),
            "--run-dir",
            str(run_dir),
            "--decision",
            str(decision_path),
            "--output",
            str(part_dir / "validation.json"),
        ],
        check=False,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "incompatible with the recorded assessment" in result.stderr


def test_initializer_decision_accepts_recorded_image_generation_fallback(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    part_dir.mkdir(parents=True)
    warning_path = part_dir / "image_generation_warning.json"
    warning_path.write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-image-generation-warning.v1"),
                "severity": "warning",
                "code": "image_generation_unavailable",
                "semantic_part": "tire",
                "reason": "The coding-agent session exposes no image editor.",
                "fallback": "direct_agentic_selection",
            }
        ),
        encoding="utf-8",
    )
    decision_path = part_dir / "initializer_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-initializer-decision.v1"),
                "semantic_part": "tire",
                "segment_id": 1,
                "status": "accepted",
                "probe_status": "skipped_image_generation_unavailable",
                "decision": "direct_agentic_selection",
                "probe_view_ids": [],
                "evidence": {
                    "image_generation_warning": "image_generation_warning.json"
                },
                "assessment": {
                    "semantic_quality": "unavailable",
                    "mask_scale": "unavailable",
                    "eroded_interior_support": "unavailable",
                    "cross_view_consistency": "unavailable",
                    "mesh_projection_quality": "unavailable",
                    "confuser_leakage": "unavailable",
                    "rationale": (
                        "No usable image generator was exposed; use direct picks."
                    ),
                },
                "next_step": "direct_signed_fragment_selection",
            }
        ),
        encoding="utf-8",
    )
    output_path = part_dir / "initializer_decision_validation.json"
    subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_initializer_decision.py"),
            "--run-dir",
            str(run_dir),
            "--decision",
            str(decision_path),
            "--output",
            str(output_path),
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    validation = json.loads(output_path.read_text(encoding="utf-8"))
    assert validation["status"] == "passed"
    assert validation["probe_status"] == ("skipped_image_generation_unavailable")
    assert validation["artifacts"][0]["role"] == "image_generation_warning"


def _write_prepared_mesh(run_dir: Path, *, face_count: int) -> None:
    """Write the neutral mesh the frontier recomputation derives adjacency from."""

    from pxr import Gf, Usd, UsdGeom, Vt

    prepare_dir = run_dir / "prepare"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    path = prepare_dir / "neutral.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    points = [(-1, 0, 0), (0, 0, 0), (0, 1, 0), (-1, 1, 0), (1, 0, 0), (1, 1, 0)]
    triangles = [(0, 1, 2), (0, 2, 3), (1, 4, 5), (1, 5, 2)][:face_count]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(triangles)))
    mesh.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray([index for triangle in triangles for index in triangle])
    )
    stage.Save()


def _write_falsification_fixture(
    tmp_path: Path,
    *,
    candidate_labels: list[int],
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    part_dir = run_dir / "tire"
    revision_dir = part_dir / "rev-001"
    render_dir = revision_dir / "renders"
    id_dir = revision_dir / "id-buffers"
    frontier_dir = part_dir / "review/frontier"
    for path in (render_dir, id_dir, frontier_dir):
        path.mkdir(parents=True, exist_ok=True)

    candidate_path = revision_dir / "face_labels.u32le"
    np.asarray(candidate_labels, dtype="<u4").tofile(candidate_path)

    # The validator recomputes the frontier from the prepared mesh and the
    # digest-bound fragment map rather than trusting the audit, so the fixture
    # carries both. Four triangles in a strip: faces 0-1 and 1-2 and 2-3 are
    # adjacent, and each face is its own fragment.
    _write_prepared_mesh(run_dir, face_count=len(candidate_labels))
    fragments_dir = run_dir / "fragments"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    fragment_ids_path = fragments_dir / "fragment_ids.u32le"
    np.arange(len(candidate_labels), dtype="<u4").tofile(fragment_ids_path)
    (fragments_dir / "fragment_manifest.json").write_text(
        json.dumps(
            {
                "fragment_ids_sha256": hashlib.sha256(
                    fragment_ids_path.read_bytes()
                ).hexdigest()
            }
        ),
        encoding="utf-8",
    )
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    camera_dir = part_dir / "cameras"
    camera_dir.mkdir()
    seed_camera = camera_dir / "seed.json"
    heldout_camera = camera_dir / "heldout.json"
    for camera in (seed_camera, heldout_camera):
        camera.write_text(
            json.dumps({"image_width": 100, "image_height": 100}),
            encoding="utf-8",
        )
    plan_path = part_dir / "falsification_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-falsification-plan.v1",
                "semantic_part": "tire",
                "segment_id": 7,
                "expected_instances": ["left", "right"],
                "confusers": [
                    {
                        "name": "rim",
                        "visual_test": "selection stops before the rim",
                    }
                ],
                "falsifiers": [
                    {
                        "id": "rim-leak",
                        "claim": "any selected rim rejects the candidate",
                    }
                ],
                "construction_view_ids": ["seed"],
                "held_out_view_ids": ["heldout"],
            }
        ),
        encoding="utf-8",
    )
    evidence_path = part_dir / "falsification_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-fragment-evidence.v1",
                "events": [
                    {
                        "polarity": "positive",
                        "face_id": 0,
                        "camera": str(seed_camera),
                        "view_id": "seed",
                        "instance_id": "left",
                        "coverage_role": "target_interior",
                        "probe_passed": True,
                        "rejection_reason": None,
                    },
                    {
                        "polarity": "positive",
                        "face_id": 0,
                        "camera": str(heldout_camera),
                        "view_id": "heldout",
                        "instance_id": "left",
                        "coverage_role": "target_extent",
                        "boundary_pair_id": "left-rim",
                        "pixel": [40, 50],
                        "probe_passed": True,
                        "rejection_reason": None,
                    },
                    {
                        "polarity": "positive",
                        "face_id": 0,
                        "camera": str(heldout_camera),
                        "view_id": "heldout",
                        "instance_id": "left",
                        "coverage_role": "opposing_or_occluded_surface",
                        "probe_passed": True,
                        "rejection_reason": None,
                    },
                    {
                        "polarity": "positive",
                        "face_id": 1,
                        "camera": str(seed_camera),
                        "view_id": "seed",
                        "instance_id": "right",
                        "coverage_role": "target_interior",
                        "probe_passed": True,
                        "rejection_reason": None,
                    },
                    {
                        "polarity": "positive",
                        "face_id": 1,
                        "camera": str(heldout_camera),
                        "view_id": "heldout",
                        "instance_id": "right",
                        "coverage_role": "target_extent",
                        "boundary_pair_id": "right-rim",
                        "pixel": [60, 50],
                        "probe_passed": True,
                        "rejection_reason": None,
                    },
                    {
                        "polarity": "positive",
                        "face_id": 1,
                        "camera": str(heldout_camera),
                        "view_id": "heldout",
                        "instance_id": "right",
                        "coverage_role": "opposing_or_occluded_surface",
                        "probe_passed": True,
                        "rejection_reason": None,
                    },
                    {
                        "polarity": "negative",
                        "face_id": 2,
                        "camera": str(heldout_camera),
                        "view_id": "heldout",
                        "instance_id": "left",
                        "confuser_id": "rim",
                        "boundary_pair_id": "left-rim",
                        "pixel": [42, 50],
                        "probe_passed": True,
                        "rejection_reason": None,
                    },
                    {
                        "polarity": "negative",
                        "face_id": 2,
                        "camera": str(heldout_camera),
                        "view_id": "heldout",
                        "instance_id": "right",
                        "confuser_id": "rim",
                        "boundary_pair_id": "right-rim",
                        "pixel": [62, 50],
                        "probe_passed": True,
                        "rejection_reason": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    comparison_path = revision_dir / "falsification_label_comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-fragment-label-comparison.v1",
                "status": "passed",
                "failures": [],
                "active_segment_id": 7,
                "candidate_labels_sha256": candidate_sha256,
                "unsatisfied_positive_face_ids": [],
                "violated_negative_face_ids": [],
                "locked_face_change_count": 0,
            }
        ),
        encoding="utf-8",
    )
    frontier_path = frontier_dir / "frontier_audit.json"
    unselected_frontier_path = frontier_dir / "unselected_frontier_fragment_ids.u32le"
    # Fragment 3 is the only unselected fragment adjacent to the selection in
    # the prepared mesh, so the audit has to say so -- the validator recomputes
    # it and will not take the audit's word.
    np.asarray([3], dtype="<u4").tofile(unselected_frontier_path)
    frontier_path.write_text(
        json.dumps(
            {
                "active_segment_id": 7,
                "face_labels_sha256": candidate_sha256,
                "unselected_frontier_fragment_ids": str(unselected_frontier_path),
                "unselected_frontier_fragment_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (id_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (8, 8), (255, 0, 255)).save(render_dir / "seed.png")
    Image.new("RGB", (8, 8), (255, 0, 255)).save(render_dir / "heldout.png")
    (render_dir / "render_manifest.json").write_text(
        json.dumps(
            {
                "renders": [
                    {"name": "seed", "image": str(render_dir / "seed.png")},
                    {
                        "name": "heldout",
                        "image": str(render_dir / "heldout.png"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    selected_dir = revision_dir / "selected-only"
    selected_render_dir = selected_dir / "renders"
    selected_render_dir.mkdir(parents=True)
    segmented_path = selected_dir / "segmented.usdc"
    segmented_path.write_bytes(b"segmented")
    selected_export_path = selected_dir / "export_manifest.json"
    selected_export_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-fragment-export.v1",
                "face_labels_sha256": candidate_sha256,
                "output_usd": str(segmented_path),
                "output_usd_sha256": hashlib.sha256(
                    segmented_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    selected_stage_path = selected_dir / "part-only.usdc"
    selected_stage_path.write_bytes(b"selected-only")
    selected_stage_manifest_path = selected_dir / "stage_manifest.json"
    selected_stage_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-selected-only-stage.v1",
                "status": "passed",
                "source_usd": str(segmented_path),
                "source_usd_sha256": hashlib.sha256(
                    segmented_path.read_bytes()
                ).hexdigest(),
                "target_prim": "/World/SegmentedAsset/Segments/tire",
                "output_usd": str(selected_stage_path),
                "output_usd_sha256": hashlib.sha256(
                    selected_stage_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    selected_render_manifest_path = selected_render_dir / "render_manifest.json"
    Image.new("RGB", (8, 8), (255, 0, 255)).save(selected_render_dir / "seed.png")
    Image.new("RGB", (8, 8), (255, 0, 255)).save(selected_render_dir / "heldout.png")
    selected_render_manifest_path.write_text(
        json.dumps(
            {
                "scene": str(selected_stage_path),
                "focus": "/World/SegmentedAsset/Segments/tire",
                "renders": [
                    {
                        "name": "seed",
                        "image": str(selected_render_dir / "seed.png"),
                    },
                    {
                        "name": "heldout",
                        "image": str(selected_render_dir / "heldout.png"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    consistency_sheet = revision_dir / "revision_consistency_sheet.png"
    Image.new("RGB", (32, 8), (255, 0, 255)).save(consistency_sheet)
    consistency_crop_dir = revision_dir / "revision-consistency"
    consistency_crop_dir.mkdir()
    context_sheet = consistency_crop_dir / "context_sheet_00.png"
    Image.new("RGB", (768, 560), (96, 96, 96)).save(context_sheet)
    consistency_instances: list[dict[str, object]] = []
    for instance_id in ("left", "right"):
        evidence_crops: list[dict[str, object]] = []
        for view_id, surface_role in (
            ("seed", "primary_surface"),
            ("heldout", "opposing_or_occlusion_revealing"),
        ):
            flat_crop = (
                consistency_crop_dir / f"{instance_id}__{view_id}__flat-label.png"
            )
            selected_crop = (
                consistency_crop_dir / f"{instance_id}__{view_id}__selected-only.png"
            )
            Image.new("RGB", (256, 256), (255, 0, 255)).save(flat_crop)
            Image.new("RGB", (256, 256), (32, 32, 32)).save(selected_crop)
            evidence_crops.append(
                {
                    "view_id": view_id,
                    "surface_role": surface_role,
                    "source_bbox": [0, 0, 8, 8],
                    "square_crop_box": [0, 0, 8, 8],
                    "flat_label_crop": str(flat_crop),
                    "flat_label_crop_sha256": hashlib.sha256(
                        flat_crop.read_bytes()
                    ).hexdigest(),
                    "selected_only_crop": str(selected_crop),
                    "selected_only_crop_sha256": hashlib.sha256(
                        selected_crop.read_bytes()
                    ).hexdigest(),
                    "crop_size": [256, 256],
                }
            )
        consistency_instances.append(
            {
                "instance_id": instance_id,
                "evidence_crops": evidence_crops,
            }
        )
    consistency_sheet_manifest = consistency_crop_dir / "manifest.json"
    consistency_sheet_manifest.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-consistency-sheet.v1",
                "status": "passed",
                "neutral_render_manifest": str(render_dir / "render_manifest.json"),
                "neutral_render_manifest_sha256": hashlib.sha256(
                    (render_dir / "render_manifest.json").read_bytes()
                ).hexdigest(),
                "flat_label_render_manifest": str(render_dir / "render_manifest.json"),
                "flat_label_render_manifest_sha256": hashlib.sha256(
                    (render_dir / "render_manifest.json").read_bytes()
                ).hexdigest(),
                "selected_only_render_manifest": str(selected_render_manifest_path),
                "selected_only_render_manifest_sha256": hashlib.sha256(
                    selected_render_manifest_path.read_bytes()
                ).hexdigest(),
                "crop_size": [256, 256],
                "layout": {
                    "mode": "tiled_peer_grid",
                    "columns": 2,
                    "rows": 1,
                    "tile_size": [16, 8],
                    "sheet_size": [32, 8],
                    "instance_order": ["left", "right"],
                    "channels_per_view": [
                        "flat_label",
                        "selected_only",
                    ],
                    "surface_role_order": [
                        "primary_surface",
                        "opposing_or_occlusion_revealing",
                    ],
                },
                "comparison_sheet": str(consistency_sheet),
                "comparison_sheet_sha256": hashlib.sha256(
                    consistency_sheet.read_bytes()
                ).hexdigest(),
                "context_layout": {
                    "mode": "full_view_triptych_pages",
                    "views_per_page": 2,
                    "image_size": [256, 256],
                    "channels": [
                        "neutral",
                        "flat_label",
                        "selected_only",
                    ],
                    "view_order": ["seed", "heldout"],
                },
                "context_pages": [
                    {
                        "page_id": "context-00",
                        "view_ids": ["seed", "heldout"],
                        "image": str(context_sheet),
                        "image_sha256": hashlib.sha256(
                            context_sheet.read_bytes()
                        ).hexdigest(),
                        "size": [768, 560],
                    }
                ],
                "instances": consistency_instances,
            }
        ),
        encoding="utf-8",
    )
    consistency_review_path = revision_dir / "revision_consistency_review.json"
    consistency_review_path.write_text(
        json.dumps(
            {
                "schema_version": ("mesh-segmentation-revision-consistency-review.v1"),
                "status": "passed",
                "semantic_part": "tire",
                "segment_id": 7,
                "revision": "rev-001",
                "candidate_labels": str(candidate_path),
                "candidate_labels_sha256": candidate_sha256,
                "neutral_render_manifest": str(render_dir / "render_manifest.json"),
                "neutral_render_manifest_sha256": hashlib.sha256(
                    (render_dir / "render_manifest.json").read_bytes()
                ).hexdigest(),
                "flat_label_render_manifest": str(render_dir / "render_manifest.json"),
                "flat_label_render_manifest_sha256": hashlib.sha256(
                    (render_dir / "render_manifest.json").read_bytes()
                ).hexdigest(),
                "selected_only_render_manifest": str(selected_render_manifest_path),
                "selected_only_render_manifest_sha256": hashlib.sha256(
                    selected_render_manifest_path.read_bytes()
                ).hexdigest(),
                "comparison_sheet": str(consistency_sheet),
                "comparison_sheet_sha256": hashlib.sha256(
                    consistency_sheet.read_bytes()
                ).hexdigest(),
                "consistency_sheet_manifest": str(consistency_sheet_manifest),
                "comparison_channels": ["flat_label", "selected_only"],
                "context_comparison_channels": [
                    "neutral",
                    "flat_label",
                    "selected_only",
                ],
                "context_comparison_sheets": [
                    {
                        "view_ids": ["seed", "heldout"],
                        "image": str(context_sheet),
                        "image_sha256": hashlib.sha256(
                            context_sheet.read_bytes()
                        ).hexdigest(),
                    }
                ],
                "context_outlier_regions": [],
                "comparison_mode": "repeated_instance_outlier",
                "expected_instances": ["left", "right"],
                "views_compared": ["seed", "heldout"],
                "outlier_regions": [],
                "instance_comparisons": [
                    {
                        "instance_id": "left",
                        "status": "consistent",
                        "peer_instance_ids": ["right"],
                        "unique_gap_regions": [],
                        "unique_protrusion_regions": [],
                        "unique_boundary_regions": [],
                        "assessment": "no left-only outlier is visible",
                    },
                    {
                        "instance_id": "right",
                        "status": "consistent",
                        "peer_instance_ids": ["left"],
                        "unique_gap_regions": [],
                        "unique_protrusion_regions": [],
                        "unique_boundary_regions": [],
                        "assessment": "no right-only outlier is visible",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def completeness_view(
        *,
        view_id: str,
        surface_role: str,
        frontier_fragment_id: int,
    ) -> dict[str, object]:
        return {
            "view_id": view_id,
            "surface_role": surface_role,
            "selected_only_render": (f"rev-001/selected-only/renders/{view_id}.png"),
            "flat_label_render": f"rev-001/renders/{view_id}.png",
            "coherent_semantic_shell": True,
            "outer_contour_continuous": True,
            "semantic_openings_plausible": True,
            "unexplained_boundaries": [],
            "boundary_classifications": [
                {
                    "region_id": f"{view_id}-outer-silhouette",
                    "classification": "silhouette_or_occlusion",
                    "rationale": "the visible boundary is the exterior silhouette",
                }
            ],
            "adjacent_unselected_frontier_fragment_ids_checked": [frontier_fragment_id],
            "frontier_decision": "confirmed_non_target",
            "assessment": "the selected-only shell is complete in this view",
        }

    review_path = part_dir / "falsification_review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": "mesh-segmentation-falsification-review.v1",
                "semantic_part": "tire",
                "segment_id": 7,
                "revision": "rev-001",
                "status": "accepted",
                "falsification_plan": "falsification_plan.json",
                "candidate_labels": "rev-001/face_labels.u32le",
                "signed_evidence": "falsification_evidence.json",
                "label_comparison": ("rev-001/falsification_label_comparison.json"),
                "frontier_audit": "review/frontier/frontier_audit.json",
                "latest_render_manifest": "rev-001/renders/render_manifest.json",
                "id_buffer_manifests": ["rev-001/id-buffers/manifest.json"],
                "selected_only_export_manifest": (
                    "rev-001/selected-only/export_manifest.json"
                ),
                "selected_only_stage_manifest": (
                    "rev-001/selected-only/stage_manifest.json"
                ),
                "selected_only_render_manifest": (
                    "rev-001/selected-only/renders/render_manifest.json"
                ),
                "selected_only_review": "passed",
                "revision_consistency_review": (
                    "rev-001/revision_consistency_review.json"
                ),
                "repeated_instance_consistency_checked": ["left", "right"],
                "instance_completeness_reviews": [
                    {
                        "instance_id": "left",
                        "status": "passed",
                        "views": [
                            completeness_view(
                                view_id="seed",
                                surface_role="primary_surface",
                                frontier_fragment_id=3,
                            ),
                            completeness_view(
                                view_id="heldout",
                                surface_role="opposing_or_occluded_surface",
                                frontier_fragment_id=3,
                            ),
                        ],
                    },
                    {
                        "instance_id": "right",
                        "status": "passed",
                        "views": [
                            completeness_view(
                                view_id="seed",
                                surface_role="primary_surface",
                                frontier_fragment_id=3,
                            ),
                            completeness_view(
                                view_id="heldout",
                                surface_role="opposing_or_occluded_surface",
                                frontier_fragment_id=3,
                            ),
                        ],
                    },
                ],
                "construction_view_ids": ["seed"],
                "held_out_view_ids": ["heldout"],
                "expected_instances_checked": ["left", "right"],
                "confusers_checked": ["rim"],
                "falsifiers": [
                    {
                        "id": "rim-leak",
                        "status": "passed",
                        "evidence": ["rev-001/renders/heldout.png"],
                        "rationale": "negative rim evidence remains unselected",
                    }
                ],
                "false_positive_review": "passed",
                "false_negative_review": "passed",
                "actionable_issues": [],
                "unresolved_uncertainties": [],
            }
        ),
        encoding="utf-8",
    )
    return run_dir, review_path


def test_falsification_review_validates_signed_confuser_evidence(
    tmp_path: Path,
) -> None:
    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    output_path = review_path.parent / "falsification_validation.json"
    subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_falsification_review.py"),
            "--run-dir",
            str(run_dir),
            "--review",
            str(review_path),
            "--output",
            str(output_path),
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    validation = json.loads(output_path.read_text(encoding="utf-8"))
    assert validation["status"] == "passed"
    assert validation["positive_face_count"] == 2
    assert validation["negative_face_count"] == 1
    assert validation["revision"] == "rev-001"
    assert validation["challenge_coverage"]["boundary_pair_count"] == 2
    assert validation["revision_consistency"]["instance_count"] == 2


def test_falsification_review_rejects_selected_negative_face(
    tmp_path: Path,
) -> None:
    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 7, 0],
    )
    result = subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_falsification_review.py"),
            "--run-dir",
            str(run_dir),
            "--review",
            str(review_path),
            "--output",
            str(review_path.parent / "falsification_validation.json"),
        ],
        check=False,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "candidate includes negative evidence faces" in result.stderr


def test_falsification_review_rejects_symlinked_artifact(tmp_path: Path) -> None:
    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    candidate_path = review_path.parent / review["candidate_labels"]
    symlink_path = candidate_path.with_name("symlinked-face-labels.u32le")
    symlink_path.symlink_to(candidate_path)
    review["candidate_labels"] = str(symlink_path.relative_to(review_path.parent))
    review_path.write_text(json.dumps(review), encoding="utf-8")

    result = subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_falsification_review.py"),
            "--run-dir",
            str(run_dir),
            "--review",
            str(review_path),
            "--output",
            str(review_path.parent / "falsification_validation.json"),
        ],
        check=False,
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Artifact must not be a symlink" in result.stderr


def test_falsification_review_rejects_incomplete_selected_only_shell(
    tmp_path: Path,
) -> None:
    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["instance_completeness_reviews"][1]["views"][1][
        "coherent_semantic_shell"
    ] = False
    review_path.write_text(json.dumps(review), encoding="utf-8")
    result = subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_falsification_review.py"),
            "--run-dir",
            str(run_dir),
            "--review",
            str(review_path),
            "--output",
            str(review_path.parent / "falsification_validation.json"),
        ],
        check=False,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires coherent_semantic_shell: true" in result.stderr


def test_falsification_review_rejects_non_frontier_completeness_check(
    tmp_path: Path,
) -> None:
    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["instance_completeness_reviews"][0]["views"][0][
        "adjacent_unselected_frontier_fragment_ids_checked"
    ] = [999]
    review_path.write_text(json.dumps(review), encoding="utf-8")
    result = subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_falsification_review.py"),
            "--run-dir",
            str(run_dir),
            "--review",
            str(review_path),
            "--output",
            str(review_path.parent / "falsification_validation.json"),
        ],
        check=False,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "claims non-frontier fragment IDs" in result.stderr


def test_falsification_review_accepts_complete_disconnected_components(
    tmp_path: Path,
) -> None:
    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    for instance_review in review["instance_completeness_reviews"]:
        for view in instance_review["views"]:
            view["adjacent_unselected_frontier_fragment_ids_checked"] = []
            view["frontier_decision"] = "not_applicable_complete_disconnected_component"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    _make_frontier_empty(run_dir, review_path)

    output_path = review_path.parent / "falsification_validation.json"
    subprocess.run(
        [
            str(PYTHON),
            str(CANONICAL_SCRIPTS / "validate_falsification_review.py"),
            "--run-dir",
            str(run_dir),
            "--review",
            str(review_path),
            "--output",
            str(output_path),
        ],
        check=True,
        env=_script_env(),
        capture_output=True,
        text=True,
    )
    validation = json.loads(output_path.read_text(encoding="utf-8"))
    assert validation["instance_completeness_coverage"]["frontier_check_mode"] == (
        "no_adjacent_unselected_frontier"
    )
    assert (
        validation["instance_completeness_coverage"][
            "checked_unselected_frontier_fragment_count"
        ]
        == 0
    )


def test_falsification_reference_documents_revision_consistency_contract() -> None:
    reference = (CANONICAL_SKILL / "references/falsification-and-locking.md").read_text(
        encoding="utf-8"
    )
    skill = (CANONICAL_SKILL / "SKILL.md").read_text(encoding="utf-8")

    for required_text in (
        "build_revision_consistency_sheet.py",
        "mesh-segmentation-consistency-regions.v1",
        "mesh-segmentation-consistency-sheet.v1",
        "mesh-segmentation-revision-consistency-review.v1",
        "candidate_labels_sha256",
        "neutral_render_manifest_sha256",
        "flat_label_render_manifest_sha256",
        "selected_only_render_manifest_sha256",
        "comparison_sheet_sha256",
        "consistency_sheet_manifest",
        "context_comparison_sheets",
        "context_outlier_regions",
        "instance_comparisons",
    ):
        assert required_text in reference

    assert "revision_consistency_regions.json" in skill
    assert "revision_consistency_review.json" in skill


def _write_disconnected_mesh(run_dir: Path, *, face_count: int) -> None:
    """Rewrite the neutral mesh so no two faces share an edge."""

    from pxr import Gf, Usd, UsdGeom, Vt

    path = run_dir / "prepare" / "neutral.usdc"
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    points = []
    indices = []
    for index in range(face_count):
        base = index * 3
        offset = float(index) * 10.0
        points.extend(
            [
                (offset, 0.0, 0.0),
                (offset + 1.0, 0.0, 0.0),
                (offset, 1.0, 0.0),
            ]
        )
        indices.extend([base, base + 1, base + 2])
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * face_count))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    stage.Save()


def _make_frontier_empty(run_dir: Path, review_path: Path) -> None:
    """Rewrite the fixture as a part with no adjacent unselected fragments.

    The geometry has to change too, not just the audit: the validator
    recomputes the frontier from the frozen fragment map, so a part is only
    vacuously satisfied when it genuinely has no unselected neighbour.
    """

    # The validator derives adjacency from the prepared mesh, so a vacuous
    # frontier has to be real geometry: rewrite the mesh as four triangles that
    # share no edges, which is the "complete disconnected source components"
    # case the waiver exists for.
    _write_disconnected_mesh(run_dir, face_count=4)
    frontier_dir = review_path.parent / "review" / "frontier"
    frontier = json.loads(
        (frontier_dir / "frontier_audit.json").read_text(encoding="utf-8")
    )
    ids_path = Path(frontier["unselected_frontier_fragment_ids"])
    np.asarray([], dtype="<u4").tofile(ids_path)
    frontier["unselected_frontier_fragment_count"] = 0
    (frontier_dir / "frontier_audit.json").write_text(
        json.dumps(frontier), encoding="utf-8"
    )


def _set_completeness_frontier(
    review_path: Path,
    *,
    fragment_ids: list[int],
    decision: str,
) -> None:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    for instance in review["instance_completeness_reviews"]:
        for view in instance["views"]:
            view["adjacent_unselected_frontier_fragment_ids_checked"] = list(
                fragment_ids
            )
            view["frontier_decision"] = decision
    review_path.write_text(json.dumps(review), encoding="utf-8")


def _run_falsification_validator(
    run_dir: Path, review_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(PYTHON),
            str(EVIDENCE_SCRIPTS / "validate_falsification_review.py"),
            "--run-dir",
            str(run_dir),
            "--review",
            str(review_path),
            "--output",
            str(review_path.parent / "falsification_validation.json"),
        ],
        check=False,
        env=_script_env(),
        capture_output=True,
        text=True,
    )


def test_falsification_review_accepts_vacuous_frontier(tmp_path: Path) -> None:
    """A part with no adjacent unselected fragment must still be lockable."""

    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    _make_frontier_empty(run_dir, review_path)
    _set_completeness_frontier(
        review_path,
        fragment_ids=[],
        decision="no_adjacent_unselected_frontier",
    )

    result = _run_falsification_validator(run_dir, review_path)
    assert result.returncode == 0, result.stderr
    validation = json.loads(
        (review_path.parent / "falsification_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["status"] == "passed"
    completeness = validation["instance_completeness_coverage"]
    assert completeness["frontier_challenge_required"] is False
    assert completeness["unselected_frontier_fragment_count"] == 0


def test_falsification_review_rejects_unearned_vacuous_frontier(
    tmp_path: Path,
) -> None:
    """The waiver is only available when the audit proves an empty frontier."""

    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    _set_completeness_frontier(
        review_path,
        fragment_ids=[],
        decision="no_adjacent_unselected_frontier",
    )

    result = _run_falsification_validator(run_dir, review_path)
    assert result.returncode != 0
    assert "must check at least one adjacent unselected frontier" in result.stderr


def test_falsification_review_requires_vacuous_decision_when_frontier_is_empty(
    tmp_path: Path,
) -> None:
    """An empty frontier cannot be reported as a confirmed challenge."""

    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    _make_frontier_empty(run_dir, review_path)
    _set_completeness_frontier(
        review_path,
        fragment_ids=[],
        decision="confirmed_non_target",
    )

    result = _run_falsification_validator(run_dir, review_path)
    assert result.returncode != 0
    assert "no_adjacent_unselected_frontier" in result.stderr


def test_falsification_review_rejects_a_truncated_frontier_ids_file(
    tmp_path: Path,
) -> None:
    """The waiver must not be forgeable by emptying one unsigned file.

    `unselected_frontier_fragment_ids.u32le` carries no digest, so deciding the
    waiver from its byte content alone let a candidate that genuinely
    under-selects truncate it to zero and waive the whole adjacent-frontier
    contract, while the audit that computed it still recorded a real count.
    """

    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    frontier_dir = review_path.parent / "review" / "frontier"
    audit_path = frontier_dir / "frontier_audit.json"
    frontier = json.loads(audit_path.read_text(encoding="utf-8"))
    # Empty the ids file *and* restate the count -- the child authors both, so
    # a validator that only cross-checks them against each other is satisfied.
    # The frozen fragment map still shows four unselected neighbours.
    np.asarray([], dtype="<u4").tofile(
        Path(frontier["unselected_frontier_fragment_ids"])
    )
    frontier["unselected_frontier_fragment_count"] = 0
    audit_path.write_text(json.dumps(frontier), encoding="utf-8")
    _set_completeness_frontier(
        review_path,
        fragment_ids=[],
        decision="no_adjacent_unselected_frontier",
    )

    result = _run_falsification_validator(run_dir, review_path)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "recomputed from the" in combined, combined[-400:]


def test_skill_scripts_accept_pre_rename_schemas() -> None:
    """The launcher and the scripts must agree on what they can read.

    The launcher tolerates the retired `-v5-` infix so a run recorded before
    the rename stays readable. The scripts compared exactly, so the same
    artifact was readable by the launcher and rejected by the tooling that
    re-validates it.
    """

    import sys

    scripts = (
        Path(__file__).resolve().parents[1]
        / ".agents"
        / "skills"
        / "content-workflow-mesh-segmentation"
        / "scripts"
    )
    sys.path.insert(0, str(scripts))
    try:
        from mesh_geometry import schema_matches
    finally:
        sys.path.remove(str(scripts))

    expected = "mesh-segmentation-falsification-plan.v1"
    assert schema_matches(expected, expected)
    assert schema_matches("mesh-segmentation-v5-falsification-plan.v1", expected)
    assert not schema_matches("mesh-segmentation-other-plan.v1", expected)
    assert not schema_matches(None, expected)

    # Two implementations exist -- one for the scripts, one in the launcher --
    # and they can only stay useful if they agree. Divergence would mean an
    # artifact the launcher reads and the validator rejects, or the reverse.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
    from content_workflow_cli.mesh_segmentation_runner import (  # noqa: PLC0415
        _schema_matches,
    )

    for value in (
        expected,
        "mesh-segmentation-v5-falsification-plan.v1",
        "mesh-segmentation-other-plan.v1",
        "content-agents.mesh-segmentation-x.v1",
        None,
        "",
    ):
        assert schema_matches(value, expected) == _schema_matches(value, expected), (
            value
        )


def test_falsification_review_ignores_a_tampered_adjacency_file(
    tmp_path: Path,
) -> None:
    """Emptying the adjacency file must not empty the frontier.

    `fragments/fragment_adjacency.npy` carries no digest, so deriving the
    frontier from it let a child empty that one file and waive the whole
    adjacent-frontier contract. Adjacency now comes from the prepared mesh and
    the digest-bound fragment map instead.
    """

    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    np.save(
        run_dir / "fragments" / "fragment_adjacency.npy",
        np.zeros((0, 2), dtype=np.int64),
    )
    _set_completeness_frontier(
        review_path,
        fragment_ids=[],
        decision="no_adjacent_unselected_frontier",
    )

    result = _run_falsification_validator(run_dir, review_path)

    assert result.returncode != 0


def test_falsification_review_rejects_a_swapped_fragment_map(tmp_path: Path) -> None:
    """The fragment map must match the manifest that froze it."""

    run_dir, review_path = _write_falsification_fixture(
        tmp_path,
        candidate_labels=[7, 7, 0, 0],
    )
    # Every face in one fragment: no unselected fragment, so no frontier.
    np.zeros(4, dtype="<u4").tofile(run_dir / "fragments" / "fragment_ids.u32le")

    result = _run_falsification_validator(run_dir, review_path)

    assert result.returncode != 0
    assert "manifest digest" in result.stdout + result.stderr
