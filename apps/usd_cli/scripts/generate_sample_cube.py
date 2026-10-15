#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the `sample_assets/simready/Cube` sample from code.

The sample is a 0.2 m cube with one UsdPreviewSurface color per face, laid out
like a SimReady payload package: an interface layer, a payload `Contents`
layer that sublayers `Geometry.usdc` and `Materials.usda`, and an instanceable
`MaterialsLibrary.usda`. Every value comes from this script.

Usage: python scripts/generate_sample_cube.py [OUTPUT_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "sample_assets" / "simready" / "Cube"
CREATOR = "usd-cli scripts/generate_sample_cube.py"
HALF = 0.1
SPDX_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)

#: (material name, diffuse color, outward normal, quad corner indices)
FACES = (
    ("red", (1, 0, 0), (0, 0, -1), (0, 3, 2, 1)),
    ("green", (0, 1, 0), (0, 0, 1), (4, 5, 6, 7)),
    ("blue", (0, 0, 1), (0, -1, 0), (0, 1, 5, 4)),
    ("yellow", (1, 1, 0), (0, 1, 0), (3, 7, 6, 2)),
    ("cyan", (0, 1, 1), (-1, 0, 0), (0, 4, 7, 3)),
    ("magenta", (1, 0, 1), (1, 0, 0), (1, 2, 6, 5)),
)
LAYER_METADATA = """(
    customLayerData = {{
        string creator = "{creator}"
    }}
    defaultPrim = "{default_prim}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
{extra}    upAxis = "Z"
)
"""


def _usda(default_prim: str, body: str, extra: str = "") -> str:
    metadata = LAYER_METADATA.format(
        creator=CREATOR, default_prim=default_prim, extra=extra
    )
    return f"#usda 1.0\n{SPDX_HEADER}{metadata}\n{body}"


def _interface() -> str:
    return _usda(
        "cube",
        """def Xform "cube" (
    prepend apiSchemas = ["GeomModelAPI"]
    assetInfo = {
        string name = "cube"
    }
    kind = "component"
    prepend payload = @./Payload/Contents.usda@
)
{
    float3[] extentsHint = [(-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)]
}
""",
    )


def _contents() -> str:
    sublayers = "    subLayers = [\n        @./Geometry.usdc@,\n        @./Materials.usda@\n    ]\n"
    return _usda("cube", 'def Xform "cube"\n{\n}\n', extra=sublayers)


def _subset_name(index: int) -> str:
    return "Subset" if index == 0 else f"Subset_{index}"


def _materials() -> str:
    materials = "".join(
        f'        def Material "{name}" (\n'
        f"            prepend references = @./MaterialsLibrary.usda@</Materials/{name}>\n"
        "        )\n        {\n        }\n\n"
        for name, *_ in FACES
    ).rstrip("\n")
    bindings = "".join(
        f'            over "{_subset_name(index)}" (\n'
        '                prepend apiSchemas = ["MaterialBindingAPI"]\n'
        "            )\n            {\n"
        f"                rel material:binding = </cube/Materials/{name}>\n"
        "            }\n\n"
        for index, (name, *_) in enumerate(FACES)
    ).rstrip("\n")
    body = (
        'def Xform "cube"\n{\n    def Scope "Materials"\n    {\n'
        f"{materials}\n    }}\n\n"
        '    over "Geometry"\n    {\n        over "color_cube"\n        {\n'
        f"{bindings}\n        }}\n    }}\n}}\n"
    )
    return _usda("cube", body)


def _material_library() -> str:
    entries = []
    for name, color, _normal, _quad in FACES:
        path = f"/Materials/{name}"
        rgb = ", ".join(str(channel) for channel in color)
        entries.append(
            f'    def Material "{name}" (\n        instanceable = true\n    )\n    {{\n'
            f"        color3f inputs:diffuseColor = ({rgb})\n"
            "        float inputs:ior = 1\n"
            "        float inputs:metallic = 0\n"
            "        float inputs:opacity = 1\n"
            "        float inputs:roughness = 0\n"
            '        token inputs:wrapMode = "repeat"\n'
            f"        token outputs:displacement.connect = <{path}/PreviewSurface.outputs:displacement>\n"
            f"        token outputs:surface.connect = <{path}/PreviewSurface.outputs:surface>\n\n"
            '        def Shader "PreviewSurface"\n        {\n'
            '            uniform token info:id = "UsdPreviewSurface"\n'
            f"            color3f inputs:diffuseColor.connect = <{path}.inputs:diffuseColor>\n"
            f"            float inputs:ior.connect = <{path}.inputs:ior>\n"
            f"            float inputs:metallic.connect = <{path}.inputs:metallic>\n"
            f"            float inputs:opacity.connect = <{path}.inputs:opacity>\n"
            f"            float inputs:roughness.connect = <{path}.inputs:roughness>\n"
            "            token outputs:displacement\n"
            "            token outputs:surface\n"
            "        }\n    }\n"
        )
    body = 'class Scope "Materials"\n{\n' + "\n".join(entries) + "}\n"
    text = _usda("Materials", body)
    return text.replace("(\n    customLayerData", '(\n    "Material Library for cube. "\n    customLayerData', 1)


def _write_geometry(path: Path) -> None:
    corners = [
        Gf.Vec3f(x * HALF, y * HALF, z * HALF)
        for z in (-1, 1)
        for y, x in ((-1, -1), (-1, 1), (1, 1), (1, -1))
    ]
    stage = Usd.Stage.CreateNew(str(path))
    layer = stage.GetRootLayer()
    layer.customLayerData = {"creator": CREATOR}
    root = UsdGeom.Xform.Define(stage, "/cube")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    UsdGeom.Scope.Define(stage, "/cube/Geometry")
    mesh = UsdGeom.Mesh.Define(stage, "/cube/Geometry/color_cube")
    indices: list[int] = []
    for *_rest, quad in FACES:
        a, b, c, d = quad
        indices.extend((a, b, c, a, c, d))
    mesh.CreatePointsAttr(Vt.Vec3fArray(corners))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-HALF), Gf.Vec3f(HALF)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * 12))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    normals = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.faceVarying
    )
    normals.Set(Vt.Vec3fArray([Gf.Vec3f(*normal) for _n, _c, normal, _q in FACES]))
    normals.SetIndices(Vt.IntArray([face for face in range(len(FACES)) for _ in range(6)]))
    for index in range(len(FACES)):
        subset = UsdGeom.Subset.Define(
            stage, f"/cube/Geometry/color_cube/{_subset_name(index)}"
        )
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(Vt.IntArray([2 * index, 2 * index + 1]))
    layer.Save()


def generate(output_dir: Path) -> None:
    """Write the five Cube sample layers into ``output_dir``."""

    payload = output_dir / "Payload"
    payload.mkdir(parents=True, exist_ok=True)
    texts = {
        output_dir / "cube.usda": _interface(),
        payload / "Contents.usda": _contents(),
        payload / "Materials.usda": _materials(),
        payload / "MaterialsLibrary.usda": _material_library(),
    }
    for path, text in texts.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    geometry = payload / "Geometry.usdc"
    geometry.unlink(missing_ok=True)
    _write_geometry(geometry)


if __name__ == "__main__":
    generate(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT)
