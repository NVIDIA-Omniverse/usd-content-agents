# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade
from world_understanding.validation.visual_grounding import (
    MeshRecord,
    _make_camera,
    _visible_entries,
    generate_visual_grounding_packet,
)

import material_agent.tasks.visual_grounding as visual_grounding_task_mod
from material_agent.tasks.visual_grounding import VisualGroundingTask


def _write_panel_usd(path: Path) -> None:
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

    stage.GetRootLayer().Save()


def test_generate_visual_grounding_packet_writes_artifacts(tmp_path: Path) -> None:
    usd_path = tmp_path / "panel.usda"
    _write_panel_usd(usd_path)
    beauty_path = tmp_path / "beauty.png"
    Image.new("RGB", (128, 128), color=(80, 80, 80)).save(beauty_path)

    packet = generate_visual_grounding_packet(
        usd_path=usd_path,
        output_dir=tmp_path / "grounding",
        beauty_image_path=beauty_path,
        rasterizer="cpu",
        width=128,
        height=128,
        min_visible_pixels=1,
        max_labels=4,
    )

    artifacts = packet["artifacts"]
    assert Path(artifacts["legend_json_path"]).exists()
    assert Path(artifacts["legend_csv_path"]).exists()
    assert Path(artifacts["html_report_path"]).exists()
    assert Path(artifacts["object_id_labeled_overlay_path"]).exists()
    assert Path(artifacts["materialized_labeled_overlay_path"]).exists()
    assert Path(artifacts["beauty_labeled_overlay_path"]).exists()
    assert (
        Path(artifacts["materialized_labeled_overlay_path"]).name
        == "materialized_labeled_overlay.png"
    )
    materialized_overlay = Image.open(artifacts["materialized_labeled_overlay_path"])
    assert materialized_overlay.width > 128
    assert materialized_overlay.height > 128
    assert packet["schema_version"] == "material-visual-grounding-packet/v1"
    assert packet["visible_entries"]
    assert packet["visible_entries"][0]["prim_path"] == "/World/Panel"
    assert packet["visible_entries"][0]["material_path"] == "/World/Looks/Painted_Gray"

    saved_packet = json.loads(Path(artifacts["legend_json_path"]).read_text())
    assert saved_packet["visible_entries"][0]["prim_path"] == "/World/Panel"


def test_generate_visual_grounding_packet_rejects_non_positive_dimensions(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "panel.usda"
    _write_panel_usd(usd_path)

    with pytest.raises(ValueError, match="positive"):
        generate_visual_grounding_packet(
            usd_path=usd_path,
            output_dir=tmp_path / "grounding",
            width=0,
            height=96,
            rasterizer="cpu",
        )


def test_generate_visual_grounding_packet_omits_missing_materialized_overlay(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "panel.usda"
    _write_panel_usd(usd_path)

    packet = generate_visual_grounding_packet(
        usd_path=usd_path,
        output_dir=tmp_path / "grounding",
        rasterizer="cpu",
        width=128,
        height=128,
        min_visible_pixels=1,
        max_labels=4,
    )

    artifacts = packet["artifacts"]
    assert artifacts["materialized_labeled_overlay_path"] is None
    html_report = Path(artifacts["html_report_path"]).read_text(encoding="utf-8")
    assert "materialized_labeled_overlay.png" not in html_report


def test_visible_entry_label_anchor_is_visible_surface_pixel() -> None:
    id_buffer = np.zeros((20, 20), dtype=np.int32)
    id_buffer[2:5, 2:5] = 1
    id_buffer[14:17, 14:17] = 1
    record = MeshRecord(
        numeric_id=1,
        prim_path="/World/DisconnectedVisiblePrim",
        material_path="/World/Looks/Black",
        parent_path="/World",
        points_world=np.zeros((0, 3), dtype=np.float64),
        triangles=np.zeros((0, 3), dtype=np.int32),
        color_rgb=(255, 0, 0),
    )

    entries = _visible_entries(
        id_buffer=id_buffer,
        records_by_id={1: record},
        min_visible_pixels=1,
    )

    assert len(entries) == 1
    entry = entries[0]
    anchor_x, anchor_y = entry["label_xy"]
    assert id_buffer[int(anchor_y), int(anchor_x)] == 1
    assert (
        id_buffer[int(entry["bbox_center_xy"][1]), int(entry["bbox_center_xy"][0])] == 0
    )
    assert entry["label_anchor_mode"] == "center_most_visible_pixel"


def test_axis_visual_grounding_cameras_are_not_corner_aliases(tmp_path: Path) -> None:
    usd_path = tmp_path / "panel.usda"
    _write_panel_usd(usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None

    positions = {
        direction: tuple(
            round(float(value), 4)
            for value in _make_camera(
                stage=stage,
                root_path="/World",
                direction=direction,
                margin=1.0,
                focal_length=50.0,
                horizontal_aperture=36.0,
                vertical_aperture=36.0,
            ).position
        )
        for direction in ["+x+y+z", "+x", "+y", "+z"]
    }

    assert len(set(positions.values())) == len(positions)


def test_visual_grounding_task_updates_context(tmp_path: Path) -> None:
    usd_path = tmp_path / "panel.usda"
    _write_panel_usd(usd_path)

    context = VisualGroundingTask().run(
        {
            "output_usd_path": str(usd_path),
            "visual_grounding_config": {
                "output_dir": str(tmp_path / "grounding"),
                "rasterizer": "cpu",
                "width": 96,
                "height": 96,
                "min_visible_pixels": 1,
            },
        }
    )

    assert Path(context["visual_grounding_packet_path"]).exists()
    assert Path(context["visual_grounding_html_path"]).exists()
    assert context["visual_grounding_packet"]["visible_entries"]


def test_visual_grounding_task_rejects_non_mapping_config(tmp_path: Path) -> None:
    usd_path = tmp_path / "panel.usda"
    _write_panel_usd(usd_path)

    with pytest.raises(TypeError, match="visual_grounding_config"):
        VisualGroundingTask().run(
            {
                "output_usd_path": str(usd_path),
                "visual_grounding_config": ["not", "a", "mapping"],
            }
        )


def test_visual_grounding_task_reports_counts(tmp_path: Path) -> None:
    usd_path = tmp_path / "panel.usda"
    _write_panel_usd(usd_path)

    output = VisualGroundingTask().run(
        {
            "output_usd_path": str(usd_path),
            "visual_grounding_config": {
                "output_dir": str(tmp_path / "grounding"),
                "rasterizer": "cpu",
                "width": 96,
                "height": 96,
                "min_visible_pixels": 1,
            },
        },
    )

    packet = output["visual_grounding_packet"]
    assert len(packet["visible_entries"]) >= 1
    assert packet["visible_entries"][0]["prim_path"] == "/World/Panel"
    assert Path(output["visual_grounding_html_path"]).exists()


def test_visual_grounding_task_uses_default_config_and_rendered_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "panel.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    beauty_path = tmp_path / "beauty.png"
    beauty_path.write_text("png", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_visual_grounding_packet(**kwargs):
        captured.update(kwargs)
        return {
            "artifacts": {
                "legend_json_path": str(tmp_path / "legend.json"),
                "html_report_path": str(tmp_path / "report.html"),
                "overlay_path": None,
                "mask_path": str(tmp_path / "mask.png"),
            },
            "visible_entries": [],
        }

    monkeypatch.setattr(
        visual_grounding_task_mod.validation_visual_grounding,
        "generate_visual_grounding_packet",
        fake_generate_visual_grounding_packet,
    )

    context = VisualGroundingTask().run(
        {
            "usd_path": str(usd_path),
            "rendered_image_paths": [str(beauty_path)],
        }
    )

    assert captured["output_dir"] == usd_path.resolve().parent / "visual_grounding"
    assert captured["beauty_image_path"] == beauty_path
    assert context["visual_grounding_overlay_paths"] == {
        "legend_json_path": str(tmp_path / "legend.json"),
        "html_report_path": str(tmp_path / "report.html"),
        "mask_path": str(tmp_path / "mask.png"),
    }
