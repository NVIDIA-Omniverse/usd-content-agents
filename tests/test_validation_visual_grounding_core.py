# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core coverage for visual grounding packet generation."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFont
from pxr import Usd, UsdGeom, UsdShade

from world_understanding.validation import visual_grounding as vg


def _write_panel_usd(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    looks = UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, f"{looks.GetPath()}/Painted_Gray")

    mesh = UsdGeom.Mesh.Define(stage, "/World/Panel")
    mesh.CreatePointsAttr(
        [
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    invisible = UsdGeom.Mesh.Define(stage, "/World/Invisible")
    invisible.CreatePointsAttr([(0.0, 0.0, 0.0)])
    invisible.CreateFaceVertexCountsAttr([1])
    invisible.CreateFaceVertexIndicesAttr([0])
    UsdGeom.Imageable(invisible.GetPrim()).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )

    guide = UsdGeom.Mesh.Define(stage, "/World/Guide")
    guide.CreatePointsAttr([(0.0, 0.0, 0.0)])
    guide.CreateFaceVertexCountsAttr([1])
    guide.CreateFaceVertexIndicesAttr([0])
    UsdGeom.Imageable(guide.GetPrim()).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    empty = UsdGeom.Mesh.Define(stage, "/World/Empty")
    empty.CreatePointsAttr([])
    empty.CreateFaceVertexCountsAttr([])
    empty.CreateFaceVertexIndicesAttr([])

    degenerate = UsdGeom.Mesh.Define(stage, "/World/Degenerate")
    degenerate.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    degenerate.CreateFaceVertexCountsAttr([2])
    degenerate.CreateFaceVertexIndicesAttr([0, 1])

    UsdGeom.Xform.Define(stage, "/World/EmptyRoot")
    assert stage.GetRootLayer().Save()
    return path


def _entry(
    entry_id: int,
    *,
    x: float,
    y: float,
    bbox: list[int],
) -> dict[str, object]:
    return {
        "id": entry_id,
        "visible_pixels": 10,
        "bbox_xyxy": bbox,
        "prim_path": f"/World/Mesh{entry_id}",
        "material_path": f"/World/Looks/Mat{entry_id}",
        "parent_path": "/World",
        "label_xy": [x, y],
        "color_rgb": [255, 0, 0],
    }


def test_generate_visual_grounding_packet_writes_artifacts_and_reports(
    tmp_path: Path,
) -> None:
    usd_path = _write_panel_usd(tmp_path / "panel.usda")
    beauty_path = tmp_path / "beauty.png"
    Image.new("RGB", (128, 96), color=(80, 80, 80)).save(beauty_path)

    packet = vg.generate_visual_grounding_packet(
        usd_path=usd_path,
        output_dir=tmp_path / "grounding",
        prim_path="/World",
        beauty_image_path=beauty_path,
        rasterizer="cpu",
        width=None,
        height=None,
        min_visible_pixels=1,
        max_labels=4,
        label_mode="callout",
    )

    artifacts = packet["artifacts"]
    for key in (
        "legend_json_path",
        "legend_csv_path",
        "html_report_path",
        "segmentation_image_path",
        "object_id_labeled_overlay_path",
        "materialized_labeled_overlay_path",
        "beauty_labeled_overlay_path",
    ):
        assert Path(artifacts[key]).exists()
    assert packet["image_size"] == [128, 96]
    assert packet["camera"]["mode"] == "cpu_triangle_rasterizer"
    assert packet["visible_entries"][0]["prim_path"] == "/World/Panel"
    assert packet["visible_entries"][0]["material_path"] == "/World/Looks/Painted_Gray"

    saved_packet = json.loads(Path(artifacts["legend_json_path"]).read_text())
    assert saved_packet["visible_entries"][0]["prim_path"] == "/World/Panel"
    csv_text = Path(artifacts["legend_csv_path"]).read_text()
    assert "visible_pixels" in csv_text
    html_text = Path(artifacts["html_report_path"]).read_text()
    assert "materialized_labeled_overlay.png" in html_text


def test_generate_visual_grounding_packet_center_labels_and_default_size(
    tmp_path: Path,
) -> None:
    usd_path = _write_panel_usd(tmp_path / "panel.usda")

    packet = vg.generate_visual_grounding_packet(
        usd_path=usd_path,
        output_dir=tmp_path / "grounding",
        rasterizer="cpu",
        min_visible_pixels=1,
        max_labels=1,
        label_mode="center",
    )

    assert packet["image_size"] == [768, 768]
    assert packet["artifacts"]["materialized_labeled_overlay_path"] is None
    html_text = Path(packet["artifacts"]["html_report_path"]).read_text()
    assert "materialized_labeled_overlay.png" not in html_text


def test_generate_visual_grounding_packet_rejects_invalid_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usd_path = _write_panel_usd(tmp_path / "panel.usda")

    with pytest.raises(ValueError, match="positive"):
        vg.generate_visual_grounding_packet(
            usd_path=usd_path,
            output_dir=tmp_path / "bad-size",
            width=0,
            height=96,
            rasterizer="cpu",
        )
    with pytest.raises(ValueError, match="rasterizer"):
        vg.generate_visual_grounding_packet(
            usd_path=usd_path,
            output_dir=tmp_path / "bad-rasterizer",
            rasterizer="bad",
        )
    with pytest.raises(ValueError, match="label_mode"):
        vg.generate_visual_grounding_packet(
            usd_path=usd_path,
            output_dir=tmp_path / "bad-label",
            label_mode="bad",
        )
    with pytest.raises(RuntimeError, match="No mesh records"):
        vg.generate_visual_grounding_packet(
            usd_path=usd_path,
            output_dir=tmp_path / "missing-root",
            prim_path="/Missing",
            rasterizer="cpu",
        )
    with pytest.raises(RuntimeError, match="No mesh records"):
        vg.generate_visual_grounding_packet(
            usd_path=usd_path,
            output_dir=tmp_path / "empty-root",
            prim_path="/World/EmptyRoot",
            rasterizer="cpu",
        )

    monkeypatch.setattr(vg.Usd.Stage, "Open", lambda _path: None)
    with pytest.raises(RuntimeError, match="Failed to open USD"):
        vg.generate_visual_grounding_packet(
            usd_path=usd_path,
            output_dir=tmp_path / "open-failed",
            rasterizer="cpu",
        )


def test_generate_visual_grounding_packet_uses_warp_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usd_path = _write_panel_usd(tmp_path / "panel.usda")
    record = vg.MeshRecord(
        numeric_id=1,
        prim_path="/World/Panel",
        material_path=None,
        parent_path="/World",
        points_world=np.zeros((0, 3), dtype=np.float64),
        triangles=np.zeros((0, 3), dtype=np.int32),
        color_rgb=(255, 0, 0),
    )

    def fake_render_id_buffer_with_warp(**kwargs):
        assert kwargs["device"] == "cuda:test"
        id_buffer = np.ones((8, 8), dtype=np.int32)
        return id_buffer, [record]

    monkeypatch.setattr(
        vg, "_render_id_buffer_with_warp", fake_render_id_buffer_with_warp
    )
    packet = vg.generate_visual_grounding_packet(
        usd_path=usd_path,
        output_dir=tmp_path / "warp",
        rasterizer="warp",
        device="cuda:test",
        width=8,
        height=8,
        min_visible_pixels=1,
    )

    assert packet["camera"] == {"mode": "warp_shape_index_image", "device": "cuda:test"}
    assert packet["visible_entries"][0]["prim_path"] == "/World/Panel"


def test_mesh_extraction_camera_and_raster_helpers(tmp_path: Path) -> None:
    usd_path = _write_panel_usd(tmp_path / "panel.usda")
    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None
    panel = stage.GetPrimAtPath("/World/Panel")

    assert vg._as_np3([1, 2, 3]).tolist() == [1.0, 2.0, 3.0]
    assert vg._is_under_root(panel, None)
    assert vg._is_under_root(panel, "/World")
    assert not vg._is_under_root(panel, "/Other")
    assert vg._is_visible(panel)
    assert vg._stable_color(1) == vg._stable_color(1)
    assert vg._triangulate([4, 2, 3], [0, 1, 2, 3, 4, 5, 0, 2, 3]).tolist() == [
        [0, 1, 2],
        [0, 2, 3],
        [0, 2, 3],
    ]

    records = vg._extract_mesh_records(stage, "/World")
    assert [record.prim_path for record in records] == ["/World/Panel"]
    assert vg._bound_material_path(stage.GetPrimAtPath("/World/Degenerate")) is None
    records_by_id = {record.numeric_id: record for record in records}
    image = vg._segmentation_image(np.ones((4, 4), dtype=np.int32), records_by_id)
    assert image.size == (4, 4)

    for direction in ("+x", "+z", "+x+y+z"):
        camera = vg._make_camera(
            stage=stage,
            root_path="/World",
            direction=direction,
            margin=1.0,
            focal_length=50.0,
            horizontal_aperture=36.0,
            vertical_aperture=36.0,
        )
        assert camera.position.shape == (3,)
        assert camera.tan_half_fov_x > 0
        assert camera.tan_half_fov_y > 0
    assert vg._target_prim_or_root(stage, None).IsValid()
    with pytest.raises(ValueError, match="Prim path not found"):
        vg._target_prim_or_root(stage, "/Missing")

    camera = vg.CameraSpec(
        position=np.array([0.0, 0.0, 2.0]),
        target=np.array([0.0, 0.0, 0.0]),
        right=np.array([1.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        forward=np.array([0.0, 0.0, -1.0]),
        focal_length=50.0,
        horizontal_aperture=36.0,
        vertical_aperture=36.0,
    )
    projected, depth = vg._project_points(records[0].points_world, camera, 32, 32)
    assert projected.shape[1] == 3
    assert np.all(depth > 0)
    id_buffer, z_buffer = vg._rasterize(records, camera, 32, 32)
    assert id_buffer.max() == 1
    assert np.isfinite(z_buffer[id_buffer == 1]).all()


def test_mesh_filter_and_material_exception_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePrim:
        def __init__(self, is_mesh: bool, is_proxy: bool):
            self._is_mesh = is_mesh
            self._is_proxy = is_proxy

        def IsA(self, _schema):
            return self._is_mesh

        def IsInstanceProxy(self):
            return self._is_proxy

    fake_stage = SimpleNamespace(
        TraverseAll=lambda: iter([FakePrim(False, False), FakePrim(True, True)])
    )
    assert vg._extract_mesh_records(fake_stage, None) == []

    def raise_binding_api(_prim):
        raise RuntimeError("binding failed")

    monkeypatch.setattr(vg.UsdShade, "MaterialBindingAPI", raise_binding_api)
    assert vg._bound_material_path(object()) is None


def test_probe_camera_helpers_choose_side_and_corner_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usd_path = _write_panel_usd(tmp_path / "panel.usda")
    stage = Usd.Stage.Open(str(usd_path))
    calls: list[tuple[str, dict[str, object]]] = []

    def recorder(name: str):
        def record(**kwargs):
            calls.append((name, kwargs))

        return record

    monkeypatch.setattr(vg, "add_focused_side_view_camera", recorder("focused_side"))
    monkeypatch.setattr(
        vg, "add_focused_corner_view_camera", recorder("focused_corner")
    )
    monkeypatch.setattr(vg, "add_side_view_camera", recorder("side"))
    monkeypatch.setattr(vg, "add_corner_view_camera", recorder("corner"))

    for root_path, direction in (
        ("/World", "+x"),
        ("/World", "+x+y+z"),
        (None, "+x"),
        (None, "+x+y+z"),
    ):
        assert (
            vg._add_probe_camera_to_stage(
                stage=stage,
                root_path=root_path,
                direction=direction,
                margin=1.0,
                focal_length=50.0,
                horizontal_aperture=36.0,
                vertical_aperture=36.0,
            )
            == "/VisualGroundingProbeCamera"
        )

    assert [name for name, _kwargs in calls] == [
        "focused_side",
        "focused_corner",
        "side",
        "corner",
    ]


def test_warp_mesh_extraction_and_render_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    usd_path = _write_panel_usd(tmp_path / "panel.usda")
    stage = Usd.Stage.Open(str(usd_path))

    class FakeWarpArray:
        def __init__(self, data):
            self._data = np.asarray(data)

        def reshape(self, shape):
            return self

    class FakeShapeImage:
        def numpy(self):
            max_uint = np.iinfo(np.uint32).max
            return np.array([[[[0, max_uint], [1, 0]]]], dtype=np.uint32)

    class FakeContext:
        def __init__(self):
            self.utils = SimpleNamespace(
                compute_pinhole_camera_rays=lambda *args, **kwargs: "rays"
            )

        def create_shape_index_image_output(self, width, height, channels):
            assert (width, height, channels) == (2, 2, 1)
            return FakeShapeImage()

        def render(self, **kwargs):
            assert kwargs["camera_rays"] == "rays"

    fake_wp = SimpleNamespace(
        vec3f="vec3f",
        int32="int32",
        uint32="uint32",
        float32="float32",
        transformf="transformf",
        Mesh=lambda **kwargs: {"mesh": kwargs},
        array=lambda data, **kwargs: FakeWarpArray(data),
        init=lambda: None,
        synchronize_device=lambda device: None,
    )
    monkeypatch.setattr(
        vg.render_warp, "_import_warp", lambda: (fake_wp, None, None, None)
    )

    warp_meshes, mesh_prims = vg._extract_world_warp_meshes(
        stage=stage,
        root_path="/World",
        time_code=vg.Usd.TimeCode.Default(),
        device="cpu",
    )
    assert len(warp_meshes) == 1
    assert [str(prim.GetPath()) for prim in mesh_prims] == ["/World/Panel"]
    assert vg._extract_world_warp_meshes(
        stage=stage,
        root_path="/Other",
        time_code=vg.Usd.TimeCode.Default(),
        device="cpu",
    ) == ([], [])

    monkeypatch.setattr(vg, "hide_prims_outside_subtree", lambda stage, root_path: None)
    monkeypatch.setattr(vg, "_add_probe_camera_to_stage", lambda **kwargs: "/Camera")
    monkeypatch.setattr(
        vg,
        "_extract_world_warp_meshes",
        lambda **kwargs: ([object()], [stage.GetPrimAtPath("/World/Panel")]),
    )
    monkeypatch.setattr(
        vg.render_warp,
        "_setup_render_context",
        lambda **kwargs: FakeContext(),
    )
    monkeypatch.setattr(vg.render_warp, "_is_visible", lambda prim, time_code: True)
    monkeypatch.setattr(vg.render_warp, "_compute_camera_fov", lambda *args: 45.0)
    monkeypatch.setattr(
        vg.render_warp,
        "_get_camera_transforms",
        lambda *args: [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]],
    )
    id_buffer, records = vg._render_id_buffer_with_warp(
        stage=stage,
        root_path="/World",
        width=2,
        height=2,
        direction="+x",
        margin=1.0,
        focal_length=50.0,
        horizontal_aperture=36.0,
        vertical_aperture=36.0,
        device="cpu",
    )
    assert id_buffer.tolist() == [[1, 0], [2, 1]]
    assert records[0].prim_path == "/World/Panel"

    monkeypatch.setattr(vg, "_extract_world_warp_meshes", lambda **kwargs: ([], []))
    with pytest.raises(RuntimeError, match="Warp extracted no meshes"):
        vg._render_id_buffer_with_warp(
            stage=stage,
            root_path=None,
            width=2,
            height=2,
            direction="+x",
            margin=1.0,
            focal_length=50.0,
            horizontal_aperture=36.0,
            vertical_aperture=36.0,
            device="cpu",
        )


def test_visible_entries_and_overlay_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_1 = vg.MeshRecord(
        numeric_id=1,
        prim_path="/World/A",
        material_path=None,
        parent_path="/World",
        points_world=np.zeros((0, 3), dtype=np.float64),
        triangles=np.zeros((0, 3), dtype=np.int32),
        color_rgb=(255, 0, 0),
    )
    record_2 = vg.MeshRecord(
        numeric_id=2,
        prim_path="/World/B",
        material_path="/World/Looks/B",
        parent_path="/World",
        points_world=np.zeros((0, 3), dtype=np.float64),
        triangles=np.zeros((0, 3), dtype=np.int32),
        color_rgb=(0, 255, 0),
    )
    id_buffer = np.zeros((20, 20), dtype=np.int32)
    id_buffer[2:5, 2:5] = 1
    id_buffer[14:19, 14:19] = 2
    entries = vg._visible_entries(id_buffer, {1: record_1, 2: record_2}, 1)
    assert [entry["id"] for entry in entries] == [2, 1]
    assert vg._visible_entries(id_buffer, {1: record_1}, 99) == []
    anchor_x, anchor_y = entries[0]["label_xy"]
    assert id_buffer[int(anchor_y), int(anchor_x)] == 2

    base = Image.new("RGB", (24, 24), "black")
    assert vg._draw_labeled_overlay(base, entries, 2, "center").size == (24, 24)
    assert vg._draw_labeled_overlay(base, entries, 2, "callout").width > 24
    assert vg._draw_callout_label_overlay(base.convert("RGBA"), [], True) is not None
    callout_canvas = base.convert("RGBA")
    vg._draw_callout_labels(callout_canvas, [], rounded=True)
    vg._draw_callout_labels(
        Image.new("RGBA", (32, 32), "black"),
        [_entry(i, x=16, y=10 + i, bbox=[14, 10, 18, 12]) for i in range(1, 5)],
        rounded=False,
    )

    beauty = tmp_path / "beauty.png"
    Image.new("RGB", (8, 8), "gray").save(beauty)
    assert vg._draw_beauty_label_overlay(
        beauty, entries, 1, (24, 24), "center"
    ).size == (24, 24)
    assert (
        vg._draw_beauty_label_overlay(
            tmp_path / "missing.png", entries, 1, (24, 24), "center"
        )
        is None
    )

    monkeypatch.setattr(
        vg.ImageFont,
        "truetype",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError),
    )
    monkeypatch.setattr(vg.ImageFont, "load_default", lambda: ImageFont.ImageFont())
    assert isinstance(vg._load_label_font(), ImageFont.ImageFont)
    shifted = vg._shift_overlay_entry(entries[0], x_offset=3, y_offset=4)
    assert shifted["label_xy"][0] == entries[0]["label_xy"][0] + 3
    assert vg._union_bbox(entries) == (2, 2, 18, 18)
    assert vg._spread_positions([], 0, 10, 2) == []
    assert vg._spread_positions([9, 9, 9], 0, 10, 4) == [2, 6, 10]
    assert vg._spread_positions([-5, -5], 0, 10, 4) == [0, 4]
    assert vg._spread_positions([5, 5, 5, 5], 5, 10, 4)[0] == 5


def test_rasterize_triangle_and_report_edge_branches(tmp_path: Path) -> None:
    z_buffer = np.full((8, 8), np.inf, dtype=np.float64)
    id_buffer = np.zeros((8, 8), dtype=np.int32)
    vg._rasterize_triangle(
        np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        1,
        z_buffer,
        id_buffer,
    )
    vg._rasterize_triangle(
        np.array([[20.0, 20.0, 1.0], [21.0, 20.0, 1.0], [20.0, 21.0, 1.0]]),
        1,
        z_buffer,
        id_buffer,
    )
    vg._rasterize_triangle(
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0]]),
        1,
        z_buffer,
        id_buffer,
    )
    vg._rasterize_triangle(
        np.array([[1.1, 1.1, 1.0], [1.2, 1.1, 1.0], [1.1, 1.2, 1.0]]),
        1,
        z_buffer,
        id_buffer,
    )
    vg._rasterize_triangle(
        np.array([[1.0, 1.0, 1.0], [6.0, 1.0, 1.0], [1.0, 6.0, 1.0]]),
        7,
        z_buffer,
        id_buffer,
    )
    assert id_buffer.max() == 7

    output_dir = tmp_path / "report"
    output_dir.mkdir()
    vg._write_legend_csv(
        output_dir / "legend.csv",
        [_entry(1, x=1, y=1, bbox=[0, 0, 2, 2])],
    )
    assert "Mesh1" in (output_dir / "legend.csv").read_text()
    vg._write_html_report(
        output_dir,
        Namespace(
            usd=Path("scene.usda"),
            prim_path=None,
            direction="+x",
            rasterizer="cpu",
            label_mode="center",
        ),
        [_entry(1, x=1, y=1, bbox=[0, 0, 2, 2])],
        elapsed_seconds=1.25,
        total_meshes=3,
    )
    assert "full stage" in (output_dir / "index.html").read_text()
