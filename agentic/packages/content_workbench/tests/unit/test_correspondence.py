# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Scene Optimizer correspondence mapping."""

from __future__ import annotations

from pathlib import Path

from content_workbench.correspondence import SceneOptimizerPathMap


def _write_source_with_subsets(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Mesh "Panel"
    {
        int[] faceVertexCounts = [4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 0, 3, 2, 1]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]

        def GeomSubset "FaceA"
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [0]
        }

        def GeomSubset "FaceB"
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [1]
        }
    }
}
""",
        encoding="utf-8",
    )


def test_scene_optimizer_path_map_handles_split_geomsubsets(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    _write_source_with_subsets(source)
    metadata = {
        "correspondence_map": {
            "split_mapping": {
                "/World/Panel": ["/World/Panel_part_0", "/World/Panel_part_1"],
            },
            "full_mapping": {
                "original_to_prototype": {
                    "/World/Panel": ["/World/Panel_part_0", "/World/Panel_part_1"],
                },
            },
        },
    }

    path_map = SceneOptimizerPathMap.from_metadata(
        original_usd_path=source,
        optimization_metadata=metadata,
    )

    assert path_map.summary() == {
        "source_paths": 3,
        "inspection_paths": 2,
        "ambiguous_source_paths": 1,
        "ambiguous_inspection_paths": 0,
    }
    assert path_map.translate_source_to_inspection("/World/Panel").inspection_paths == [
        "/World/Panel_part_0",
        "/World/Panel_part_1",
    ]
    assert path_map.translate_source_to_inspection(
        "/World/Panel/FaceB"
    ).inspection_paths == ["/World/Panel_part_1"]
    assert path_map.translate_inspection_to_source(
        "/World/Panel_part_0"
    ).source_paths == ["/World/Panel/FaceA"]


def test_scene_optimizer_path_map_does_not_double_append_shared_suffix(
    tmp_path: Path,
) -> None:
    """Dedup targets can carry a suffix relative to their own key.

    ``mesh_I2`` and ``mesh_I7`` both dedup onto the same runtime mesh, whose
    path already has a "/Geometry" child segment relative to its own
    original key. Translating that already-suffixed path in the wrong
    direction (or re-translating an already-resolved path) must not
    re-append the suffix on top of itself and produce a nonexistent
    "/Geometry/Geometry" prim path.
    """
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    metadata = {
        "correspondence_map": {
            "full_mapping": {
                "original_to_prototype": {
                    "mesh_I2": ["mesh_I2/Geometry"],
                    "mesh_I7": ["mesh_I2/Geometry"],
                },
            },
        },
    }

    path_map = SceneOptimizerPathMap.from_metadata(
        original_usd_path=source,
        optimization_metadata=metadata,
    )

    assert path_map.translate_inspection_to_source("mesh_I2/Geometry").source_paths == [
        "mesh_I2",
        "mesh_I7",
    ]
    assert path_map.translate_source_to_inspection(
        "mesh_I2/Geometry"
    ).inspection_paths == ["mesh_I2/Geometry"]
    assert path_map.translate_source_to_inspection("mesh_I2").inspection_paths == [
        "mesh_I2/Geometry"
    ]


def test_scene_optimizer_path_map_sorts_mixed_numeric_and_alpha_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    metadata = {
        "correspondence_map": {
            "full_mapping": {
                "original_to_prototype": {
                    "/World/Panel": [
                        "/World/Panel_part_10",
                        "/World/Panel_part_alpha",
                        "/World/Panel_part_2",
                    ],
                },
            },
        },
    }

    path_map = SceneOptimizerPathMap.from_metadata(
        original_usd_path=source,
        optimization_metadata=metadata,
    )

    assert path_map.translate_source_to_inspection("/World/Panel").inspection_paths == [
        "/World/Panel_part_2",
        "/World/Panel_part_10",
        "/World/Panel_part_alpha",
    ]
