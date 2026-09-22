# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit static glTF triangle intake preserving source vertices and faces.

This bounded route rejects unsupported geometry features. It never welds,
deduplicates, repairs or optimizes render geometry, including zero-area faces.
Material translation is USDPreviewSurface, not an exact glTF shader claim.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def import_static_gltf(source: Path, output: Path) -> dict:
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, Vt

    source = source.resolve()
    raw = source.read_bytes()
    binary = None
    if source.suffix.lower() == ".glb":
        if len(raw) < 20 or struct.unpack_from("<III", raw) != (
            0x46546C67,
            2,
            len(raw),
        ):
            raise ValueError("Invalid glTF2 binary header")
        chunks = {}
        offset = 12
        while offset < len(raw):
            size, kind = struct.unpack_from("<II", raw, offset)
            if offset + 8 + size > len(raw) or kind in chunks:
                raise ValueError("Invalid or repeated GLB chunk")
            chunks[kind] = raw[offset + 8 : offset + 8 + size]
            offset += 8 + size
        doc = json.loads(chunks[0x4E4F534A])
        binary = chunks.get(0x004E4942)
    else:
        doc = json.loads(raw)
    if doc.get("asset", {}).get("version") != "2.0":
        raise ValueError("Only glTF2 is supported")
    if doc.get("extensionsRequired") or doc.get("animations") or doc.get("skins"):
        raise ValueError(
            "Required extensions, animation and skinning require another explicit route"
        )
    dependencies = {source.name: _digest(raw)}

    def uri_bytes(uri):
        if uri.startswith("data:"):
            header, data = uri.split(",", 1)
            if not header.endswith(";base64"):
                raise ValueError("Only base64 embedded URIs are supported")
            return base64.b64decode(data, validate=True)
        parsed = urlsplit(uri)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("Only contained local glTF dependencies are supported")
        p = (source.parent / unquote(parsed.path)).resolve()
        if not p.is_relative_to(source.parent) or not p.is_file():
            raise ValueError("Missing or escaping glTF dependency")
        data = p.read_bytes()
        dependencies[str(p.relative_to(source.parent))] = _digest(data)
        return data

    buffers = []
    for index, item in enumerate(doc.get("buffers", [])):
        data = (
            uri_bytes(item["uri"]) if "uri" in item else binary if index == 0 else None
        )
        if data is None or len(data) < item["byteLength"]:
            raise ValueError("Missing or truncated glTF buffer")
        buffers.append(data[: item["byteLength"]])

    def view_data(index):
        view = doc["bufferViews"][index]
        data = buffers[view["buffer"]]
        offset = view.get("byteOffset", 0)
        end = offset + view["byteLength"]
        if offset < 0 or end > len(data):
            raise ValueError("Buffer view is outside its buffer")
        return data[offset:end], view

    def accessor(index):
        a = doc["accessors"][index]
        if "sparse" in a or a["type"] not in ("SCALAR", "VEC2", "VEC3", "VEC4"):
            raise ValueError("Sparse or matrix accessors are unsupported")
        dtype = np.dtype(
            {
                5120: "i1",
                5121: "u1",
                5122: "<i2",
                5123: "<u2",
                5125: "<u4",
                5126: "<f4",
            }[a["componentType"]]
        )
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        data, view = view_data(a["bufferView"])
        offset = a.get("byteOffset", 0)
        stride = view.get("byteStride", dtype.itemsize * width)
        count = a["count"]
        if (
            count < 1
            or stride < dtype.itemsize * width
            or offset < 0
            or offset + (count - 1) * stride + dtype.itemsize * width > len(data)
        ):
            raise ValueError("Invalid accessor extent or stride")
        values = np.ndarray(
            (count, width),
            dtype,
            buffer=data,
            offset=offset,
            strides=(stride, dtype.itemsize),
        ).copy()
        if a.get("normalized"):
            values = np.maximum(values.astype(float) / np.iinfo(dtype).max, -1)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite source accessor")
        return values

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    images = []
    for index, item in enumerate(doc.get("images", [])):
        data = (
            uri_bytes(item["uri"])
            if "uri" in item
            else view_data(item["bufferView"])[0]
        )
        suffix = (
            ".png"
            if data.startswith(b"\x89PNG")
            else ".jpg"
            if data.startswith(b"\xff\xd8")
            else None
        )
        if suffix is None:
            raise ValueError("Only PNG and JPEG textures are supported")
        rel = Path("textures") / f"image_{index:04d}{suffix}"
        (output.parent / rel).parent.mkdir(exist_ok=True)
        (output.parent / rel).write_bytes(data)
        images.append(rel.as_posix())
    materials = []
    if doc.get("materials"):
        UsdGeom.Scope.Define(stage, "/Asset/Looks")
    for index, item in enumerate(doc.get("materials", [])):
        path = f"/Asset/Looks/Material_{index:04d}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        uv = UsdShade.Shader.Define(stage, path + "/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        pbr = item.get("pbrMetallicRoughness", {})
        color = pbr.get("baseColorFactor", [1, 1, 1, 1])
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color[:3])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            pbr.get("roughnessFactor", 1)
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            pbr.get("metallicFactor", 1)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])

        def texture(info, name, space, scale):
            if info.get("texCoord", 0) != 0 or info.get("extensions"):
                raise ValueError(
                    "Alternate UV sets or texture transforms are unsupported"
                )
            tex = doc["textures"][info["index"]]
            sampler = (
                doc.get("samplers", [{}])[tex.get("sampler", 0)]
                if doc.get("samplers")
                else {}
            )
            t = UsdShade.Shader.Define(stage, path + "/" + name)
            t.CreateIdAttr("UsdUVTexture")
            t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(images[tex["source"]])
            )
            t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
            t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv.ConnectableAPI(), "result"
            )
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            for axis in ("S", "T"):
                t.CreateInput("wrap" + axis, Sdf.ValueTypeNames.Token).Set(
                    {10497: "repeat", 33071: "clamp", 33648: "mirror"}[
                        sampler.get("wrap" + axis, 10497)
                    ]
                )
            return t

        if "baseColorTexture" in pbr:
            t = texture(pbr["baseColorTexture"], "Color", "sRGB", color)
            shader.GetInput("diffuseColor").ConnectToSource(t.ConnectableAPI(), "rgb")
        if "metallicRoughnessTexture" in pbr:
            t = texture(
                pbr["metallicRoughnessTexture"],
                "MetalRough",
                "raw",
                [1, pbr.get("roughnessFactor", 1), pbr.get("metallicFactor", 1), 1],
            )
            shader.GetInput("roughness").ConnectToSource(t.ConnectableAPI(), "g")
            shader.GetInput("metallic").ConnectToSource(t.ConnectableAPI(), "b")
        if "normalTexture" in item:
            t = texture(item["normalTexture"], "Normal", "raw", [2, 2, 2, 1])
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(-1, -1, -1, 0)
            )
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                t.ConnectableAPI(), "rgb"
            )
        materials.append(material)
    parts = []
    visited = set()

    def visit(index, parent):
        if index in visited:
            raise ValueError("Repeated/cyclic node attachment is unsupported")
        visited.add(index)
        node = doc["nodes"][index]
        if node.get("skin") is not None or node.get("weights"):
            raise ValueError("Deformed meshes are unsupported")
        path = (
            parent
            + "/"
            + Tf.MakeValidIdentifier(node.get("name", "Node"))
            + f"_{index}"
        )
        xform = UsdGeom.Xform.Define(stage, path)
        if "matrix" in node:
            matrix = np.asarray(node["matrix"], float).reshape((4, 4), order="F")
        else:
            x, y, z, w = node.get("rotation", [0, 0, 0, 1])
            q = np.array([x, y, z, w], float)
            if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
                raise ValueError("Nonunit node quaternion")
            matrix = np.eye(4)
            matrix[:3, :3] = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            ) @ np.diag(node.get("scale", [1, 1, 1]))
            matrix[:3, 3] = node.get("translation", [0, 0, 0])
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Invalid affine node transform")
        xform.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
        if "mesh" in node:
            for pindex, primitive in enumerate(
                doc["meshes"][node["mesh"]]["primitives"]
            ):
                if (
                    primitive.get("mode", 4) != 4
                    or primitive.get("targets")
                    or primitive.get("extensions")
                ):
                    raise ValueError(
                        "Only undeformed, uncompressed triangles are supported"
                    )
                attributes = primitive["attributes"]
                positions = accessor(attributes["POSITION"])
                if (
                    positions.shape[1] != 3
                    or doc["accessors"][attributes["POSITION"]]["componentType"] != 5126
                ):
                    raise ValueError("Position must be FLOAT VEC3")
                faces = (
                    accessor(primitive["indices"]).reshape(-1)
                    if "indices" in primitive
                    else np.arange(len(positions))
                )
                if (
                    not np.issubdtype(faces.dtype, np.integer)
                    or len(faces) % 3
                    or np.any(faces < 0)
                    or np.any(faces >= len(positions))
                ):
                    raise ValueError("Invalid triangle indices")
                mesh = UsdGeom.Mesh.Define(stage, path + f"/Primitive_{pindex}")
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreatePointsAttr(
                    Vt.Vec3fArray.FromNumpy(positions.astype(np.float32))
                )
                mesh.CreateExtentAttr(
                    Vt.Vec3fArray.FromNumpy(
                        np.array([positions.min(0), positions.max(0)], dtype=np.float32)
                    )
                )
                mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 3))
                mesh.CreateFaceVertexIndicesAttr(faces.astype(int).tolist())
                for key in attributes:
                    if key not in ("POSITION", "NORMAL", "TEXCOORD_0"):
                        raise ValueError("Unsupported source vertex attribute " + key)
                if "NORMAL" in attributes:
                    normals = accessor(attributes["NORMAL"])
                    if normals.shape != positions.shape:
                        raise ValueError("Invalid source normal count")
                    mesh.CreateNormalsAttr(
                        Vt.Vec3fArray.FromNumpy(normals.astype(np.float32))
                    )
                    mesh.SetNormalsInterpolation("vertex")
                if "TEXCOORD_0" in attributes:
                    uv_values = accessor(attributes["TEXCOORD_0"])
                    if uv_values.shape != (len(positions), 2):
                        raise ValueError("Invalid source UV count")
                    # USD texture coordinates use the opposite vertical image convention.
                    uv_values[:, 1] = 1 - uv_values[:, 1]
                    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
                    ).Set(Vt.Vec2fArray.FromNumpy(uv_values.astype(np.float32)))
                if "material" in primitive:
                    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                        materials[primitive["material"]]
                    )
                    mesh.CreateDoubleSidedAttr(
                        doc["materials"][primitive["material"]].get(
                            "doubleSided", False
                        )
                    )
                mesh.GetPrim().SetCustomDataByKey("sourceGltfNode", index)
                parts.append(
                    {
                        "node": index,
                        "primitive": pindex,
                        "path": str(mesh.GetPath()),
                        "vertices": len(positions),
                        "triangles": len(faces) // 3,
                        "source_positions_sha256": _digest(
                            positions.astype("<f4").tobytes()
                        ),
                        "source_indices_sha256": _digest(faces.astype("<u4").tobytes()),
                    }
                )
        for child in node.get("children", []):
            visit(child, path)

    for index in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        visit(index, "/Asset")
    if not parts:
        raise ValueError("Source scene has no triangle geometry")
    stage.GetRootLayer().Save()
    record = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": _digest(raw),
        "source_dependencies": dependencies,
        "meters_per_unit": 1,
        "up_axis": "Y",
        "source_faces_preserved": True,
        "parts": parts,
        "limitations": [
            "Static triangle geometry only; unsupported geometry fails explicitly.",
            "Materials use USDPreviewSurface translation; shader equivalence is not certified.",
        ],
    }
    record_path = output.parent / "lossless_gltf_receipt.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return {
        "receipt": str(record_path),
        "source_sha256": record["source_sha256"],
        "parts": parts,
    }


def verify_preserved_render(reference: Path, candidate: Path) -> dict:
    """Verify source-derived prepared geometry survives handoff, before deferring topology."""
    from pxr import Usd, UsdGeom, UsdPhysics

    def read(path):
        stage = Usd.Stage.Open(str(path))
        cache = UsdGeom.XformCache()
        meshes = {}
        if stage is None:
            raise ValueError("Unreadable preserved render stage")
        for p in stage.Traverse():
            if any(attribute.GetNumTimeSamples() for attribute in p.GetAttributes()):
                raise ValueError("Static source-preserving handoff cannot admit time samples")
            if p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(
                    "Render-only handoff cannot admit collision or rigid-body authoring"
                )
            if not p.IsA(UsdGeom.Mesh):
                if p.IsA(UsdGeom.Gprim):
                    raise ValueError("Additional non-source visual geometry")
                continue
            m = UsdGeom.Mesh(p)
            v = np.asarray(m.GetPointsAttr().Get(), float)
            f = np.asarray(m.GetFaceVertexIndicesAttr().Get(), int)
            counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), int)
            if (
                not len(v)
                or not len(f)
                or not np.isfinite(v).all()
                or (counts < 3).any()
                or counts.sum() != len(f)
                or (f < 0).any()
                or (f >= len(v)).any()
            ):
                raise ValueError("Malformed render mesh")
            meshes[str(p.GetPath())] = (
                v,
                f,
                counts,
                np.asarray(cache.GetLocalToWorldTransform(p), float),
                np.array(
                    [
                        str(UsdGeom.Imageable(p).ComputeVisibility()),
                        str(UsdGeom.Imageable(p).ComputePurpose()),
                        str(m.GetSubdivisionSchemeAttr().Get()),
                        str(m.GetOrientationAttr().Get()),
                        str(m.GetDoubleSidedAttr().Get()),
                    ]
                ),
            )
        return (
            float(UsdGeom.GetStageMetersPerUnit(stage)),
            str(UsdGeom.GetStageUpAxis(stage)),
            meshes,
        )

    a = read(reference)
    b = read(candidate)
    if a[:2] != b[:2] or a[2].keys() != b[2].keys():
        raise ValueError("Source units, axes or part identity changed")
    for key, arrays in a[2].items():
        if any(
            not np.array_equal(x, y) for x, y in zip(arrays, b[2][key], strict=True)
        ):
            raise ValueError("Source geometry or placement changed: " + key)
    return {
        "passed": True,
        "source_mesh_count": len(a[2]),
        "claim_scope": "Exact prepared/source-derived render points, face arrays, part paths, visibility, purpose, surface orientation, units and default transforms only; collision and runtime validation remain required downstream.",
    }
