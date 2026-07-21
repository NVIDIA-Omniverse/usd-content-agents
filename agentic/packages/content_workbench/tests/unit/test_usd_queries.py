# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for USD query support."""

from __future__ import annotations

from pathlib import Path

from content_workbench.usd_queries import UsdSceneQueries


def _write_sample_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Red"
        {
            token outputs:surface.connect = </World/Looks/Red/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1, 0, 0)
                asset inputs:sourceAsset = @omniverse://simready.ov.nvidia.com/Materials/Missing.mdl@
                token outputs:surface
            }
        }
    }

    def Mesh "Panel"
    {
        rel material:binding = </World/Looks/Red>
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    }
}
""",
        encoding="utf-8",
    )


def test_queries_return_root_children_and_properties(tmp_path: Path) -> None:
    stage_path = tmp_path / "sample.usda"
    _write_sample_stage(stage_path)

    queries = UsdSceneQueries(stage_path)

    assert queries.root_prim_path() == "/World"
    children = queries.get_children("/World")
    assert [child.path for child in children.children] == [
        "/World/Looks",
        "/World/Panel",
    ]

    properties = queries.get_properties("/World/Panel")
    assert properties.prim_path == "/World/Panel"
    assert properties.properties["type_name"] == "Mesh"
    assert properties.properties["bounds"]["min"] == [-1.0, -1.0, 0.0]
    assert properties.properties["relationships"]["material:binding"] == [
        "/World/Looks/Red"
    ]


def test_material_binding_and_remote_asset_diagnostics(tmp_path: Path) -> None:
    stage_path = tmp_path / "sample.usda"
    _write_sample_stage(stage_path)

    queries = UsdSceneQueries(stage_path)

    binding = queries.get_material_binding("/World/Panel")
    assert binding.binding_type == "direct"
    assert binding.bound_material_path == "/World/Looks/Red"
    assert binding.binding_source_path == "/World/Panel"
    assert binding.direct_targets == ["/World/Looks/Red"]

    diagnostics = queries.diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0].type == "remote_asset"
    assert diagnostics[0].source == (
        "omniverse://simready.ov.nvidia.com/Materials/Missing.mdl"
    )
    assert diagnostics[0].prim_path == "/World/Looks/Red/Shader"
    assert diagnostics[0].attribute == "inputs:sourceAsset"
