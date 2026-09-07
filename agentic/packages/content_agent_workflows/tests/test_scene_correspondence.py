# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for workflow-owned Scene Optimizer correspondence."""

from __future__ import annotations

from pathlib import Path

from content_agent_workflows.common.scene_correspondence import SceneOptimizerPathMap


def test_path_map_maps_split_instance_proxy_geomsubsets_to_source_subsets(
    tmp_path: Path,
) -> None:
    """Split instance-proxy meshes retain real GeomSubset authoring targets."""
    source = tmp_path / "source.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "Prototype"
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

def Xform "World"
{
    def Xform "Instance" (
        instanceable = true
        references = </Prototype>
    )
    {
    }
}
""",
        encoding="utf-8",
    )

    path_map = SceneOptimizerPathMap.from_metadata(
        original_usd_path=source,
        optimization_metadata={
            "correspondence_map": {
                "split_mapping": {
                    "/World/Instance/Panel": [
                        "/World/Panel_part_0",
                        "/World/Panel_part_1",
                    ],
                },
                "full_mapping": {
                    "original_to_prototype": {
                        "/World/Instance/Panel": [
                            "/World/Panel_part_0",
                            "/World/Panel_part_1",
                        ],
                    },
                },
            },
        },
    )

    assert path_map.translate_inspection_to_source(
        "/World/Panel_part_0"
    ).source_paths == ["/World/Instance/Panel/FaceA"]
    assert path_map.translate_inspection_to_source(
        "/World/Panel_part_1"
    ).source_paths == ["/World/Instance/Panel/FaceB"]
