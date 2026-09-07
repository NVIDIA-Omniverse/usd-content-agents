# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import pytest

from geometry_agent_service.source_dependencies import (
    SourceDependencyError,
    validate_3mf_package_dependencies,
    validate_materialized_source_dependencies,
)


def _validate(root: Path, source: Path) -> None:
    validate_materialized_source_dependencies(
        source,
        package_root=root,
        package_files=tuple(path for path in root.rglob("*") if path.is_file()),
    )


def test_self_contained_gltf_and_glb_are_accepted(tmp_path: Path) -> None:
    gltf = tmp_path / "scene.gltf"
    gltf.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [
                    {
                        "uri": "data:application/octet-stream;base64,AAAA",
                        "byteLength": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _validate(tmp_path, gltf)

    document = json.dumps(
        {"asset": {"version": "2.0"}},
        separators=(",", ":"),
    ).encode("utf-8")
    document += b" " * (-len(document) % 4)
    glb = tmp_path / "scene.glb"
    glb.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, 20 + len(document))
        + struct.pack("<II", len(document), 0x4E4F534A)
        + document
    )
    _validate(tmp_path, glb)


def test_obj_dependency_closure_is_recursive_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "root.obj"
    child = tmp_path / "child.obj"
    material = tmp_path / "surface.mtl"
    texture = tmp_path / "surface.png"
    root.write_text("trace_obj child.obj\nv 0 0 0\nf 1 1 1\n", encoding="utf-8")
    child.write_text("mtllib surface.mtl\nv 0 0 0\nf 1 1 1\n", encoding="utf-8")
    material.write_text("newmtl surface\nmap_Kd surface.png\n", encoding="utf-8")
    texture.write_bytes(b"bounded texture bytes")

    _validate(tmp_path, root)

    child.write_text("mtllib ../../outside.mtl\nv 0 0 0\nf 1 1 1\n", encoding="utf-8")
    with pytest.raises(SourceDependencyError, match="traversal"):
        _validate(tmp_path, root)


def test_obj_and_mtl_reject_executable_or_ambiguous_paths(tmp_path: Path) -> None:
    source = tmp_path / "scene.obj"
    material = tmp_path / "surface.mtl"
    texture = tmp_path / "texture.png"
    material.write_text("newmtl surface\nmap_Kd texture.png\n", encoding="utf-8")
    texture.write_bytes(b"texture")

    for content in (
        "call helper.script\nv 0 0 0\nf 1 1 1\n",
        "csh echo unsafe\nv 0 0 0\nf 1 1 1\n",
        "mtllib ..\\surface.mtl\nv 0 0 0\nf 1 1 1\n",
    ):
        source.write_text(content, encoding="utf-8")
        with pytest.raises(SourceDependencyError):
            _validate(tmp_path, source)


def test_usd_sublayers_must_be_inside_the_admitted_package(tmp_path: Path) -> None:
    root = tmp_path / "root.usda"
    child = tmp_path / "layers" / "child.usda"
    child.parent.mkdir()
    root.write_text(
        "#usda 1.0\n(\n    subLayers = [@layers/child.usda@]\n)\n",
        encoding="utf-8",
    )
    child.write_text('#usda 1.0\ndef Xform "Child" {}\n', encoding="utf-8")
    _validate(tmp_path, root)

    child.write_text(
        '#usda 1.0\ndef Xform "Child" (references = @../../../outside.usda@) {}\n',
        encoding="utf-8",
    )
    with pytest.raises(SourceDependencyError, match="traversal"):
        _validate(tmp_path, root)


def test_usd_udim_textures_must_resolve_inside_the_package(tmp_path: Path) -> None:
    root = tmp_path / "root.usda"
    texture = tmp_path / "textures" / "surface.1001.png"
    texture.parent.mkdir()
    texture.write_bytes(b"texture")
    root.write_text(
        '#usda 1.0\ndef Shader "Texture" {\n'
        "    asset inputs:file = @textures/surface.<UDIM>.png@\n}\n",
        encoding="utf-8",
    )
    _validate(tmp_path, root)

    texture.unlink()
    with pytest.raises(SourceDependencyError, match="absent from its package"):
        _validate(tmp_path, root)


def test_3mf_relationships_and_textures_stay_inside_the_package(
    tmp_path: Path,
) -> None:
    relationships = tmp_path / "_rels" / ".rels"
    model = tmp_path / "3D" / "3dmodel.model"
    texture = tmp_path / "3D" / "Textures" / "surface.png"
    relationships.parent.mkdir()
    model.parent.mkdir()
    texture.parent.mkdir()
    relationships.write_text(
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rel0" Target="/3D/3dmodel.model" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>",
        encoding="utf-8",
    )
    model.write_text(
        '<model xmlns="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">'
        '<resources><texture2d id="1" path="/3D/Textures/surface.png" '
        'contenttype="image/png"/></resources></model>',
        encoding="utf-8",
    )
    texture.write_bytes(b"texture")
    package_files = tuple(path for path in tmp_path.rglob("*") if path.is_file())

    validate_3mf_package_dependencies(tmp_path, package_files)

    relationships.write_text(
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rel0" Target="https://assets.example/scene.model" '
        'TargetMode="External" Type="model"/>'
        "</Relationships>",
        encoding="utf-8",
    )
    with pytest.raises(SourceDependencyError, match="external relationship"):
        validate_3mf_package_dependencies(tmp_path, package_files)


def test_3mf_archive_dependency_closure_rejects_external_relationships(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.3mf"
    relationships = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rel0" Target="/3D/3dmodel.model" Type="model"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("_rels/.rels", relationships)
        archive.writestr(
            "3D/3dmodel.model",
            '<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"/>',
        )
    _validate(tmp_path, source)

    external = relationships.replace(
        'Target="/3D/3dmodel.model"',
        'Target="https://assets.example/model" TargetMode="External"',
    )
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("_rels/.rels", external)
        archive.writestr(
            "3D/3dmodel.model",
            '<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"/>',
        )

    with pytest.raises(SourceDependencyError, match="external relationship"):
        _validate(tmp_path, source)


def test_dependency_parsers_reject_oversized_single_lines(tmp_path: Path) -> None:
    source = tmp_path / "scene.obj"
    source.write_bytes(b"#" + b"x" * (1024 * 1024) + b"\n")

    with pytest.raises(SourceDependencyError, match="oversized line"):
        _validate(tmp_path, source)


def test_ply_texture_directive_rejects_whitespace_obfuscated_traversal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene.ply"
    source.write_text(
        "ply\nformat ascii 1.0\ncomment\tTextureFile\t../../outside.png\n"
        "element vertex 0\nend_header\n",
        encoding="ascii",
    )

    with pytest.raises(SourceDependencyError, match="traversal"):
        _validate(tmp_path, source)
