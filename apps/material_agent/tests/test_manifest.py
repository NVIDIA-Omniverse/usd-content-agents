# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material_agent.manifest module.

Covers: prim_path helpers, _find_mdl_root, write_materials_yaml,
discover_materials, and run_generate_manifest error/list paths.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from material_agent.manifest import (
    GenerateManifestInput,
    GenerateManifestResult,
    _find_mdl_root,
    discover_materials,
    prim_path_to_filename,
    prim_path_to_name,
    run_generate_manifest,
    write_materials_yaml,
)

# ---------------------------------------------------------------------------
# GenerateManifestInput defaults
# ---------------------------------------------------------------------------


def test_generate_manifest_input_uses_public_vlm_defaults():
    params = GenerateManifestInput(usd_file=Path("scene.usd"), output_dir=Path("out"))

    assert params.vlm_backend == "nim"
    assert params.vlm_model == "google/gemma-4-31b-it"


# ---------------------------------------------------------------------------
# prim_path_to_name
# ---------------------------------------------------------------------------


class TestPrimPathToName:
    def test_simple_path(self):
        assert prim_path_to_name("/World/Looks/Aluminum_Brushed") == "Aluminum Brushed"

    def test_single_segment(self):
        assert prim_path_to_name("/Gold") == "Gold"

    def test_no_underscores(self):
        assert prim_path_to_name("/World/Looks/Steel") == "Steel"

    def test_multiple_underscores(self):
        assert (
            prim_path_to_name("/World/Looks/Car_Paint_Metallic_Blue")
            == "Car Paint Metallic Blue"
        )

    def test_deeply_nested(self):
        assert prim_path_to_name("/A/B/C/D/My_Material") == "My Material"


# ---------------------------------------------------------------------------
# prim_path_to_filename
# ---------------------------------------------------------------------------


class TestPrimPathToFilename:
    def test_simple_path(self):
        assert (
            prim_path_to_filename("/World/Looks/Aluminum_Brushed") == "Aluminum_Brushed"
        )

    def test_single_segment(self):
        assert prim_path_to_filename("/Gold") == "Gold"

    def test_preserves_underscores(self):
        assert prim_path_to_filename("/World/Looks/Car_Paint_Blue") == "Car_Paint_Blue"

    def test_deeply_nested(self):
        assert prim_path_to_filename("/A/B/C/D/My_Material") == "My_Material"


# ---------------------------------------------------------------------------
# _find_mdl_root
# ---------------------------------------------------------------------------


class TestFindMdlRoot:
    def test_empty_list_returns_none(self):
        assert _find_mdl_root([]) is None

    def test_single_file_no_relative_imports(self, tmp_path: Path):
        mdl = tmp_path / "sub" / "Material.mdl"
        mdl.parent.mkdir()
        mdl.write_text("mdl 1.0;\nexport material M() = material() {};")
        result = _find_mdl_root([mdl])
        # No relative imports → root is the file's parent dir
        assert result == tmp_path / "sub"

    def test_single_file_with_parent_import(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        mdl = sub / "Material.mdl"
        mdl.write_text("mdl 1.0;\nimport ..::Templates::Glass;")
        result = _find_mdl_root([mdl])
        # One level of ..:: → root goes up one from parent
        assert result == tmp_path

    def test_two_files_common_ancestor(self, tmp_path: Path):
        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        a_dir.mkdir()
        b_dir.mkdir()
        a = a_dir / "MatA.mdl"
        b = b_dir / "MatB.mdl"
        a.write_text("mdl 1.0;\nexport material A() = material() {};")
        b.write_text("mdl 1.0;\nexport material B() = material() {};")
        result = _find_mdl_root([a, b])
        assert result == tmp_path

    def test_double_relative_import_depth(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        mdl = deep / "Mat.mdl"
        mdl.write_text("mdl 1.0;\nimport ..::..::Templates::Glass;")
        result = _find_mdl_root([mdl])
        # Two levels of ..:: from parent of deep (c) → goes up 2
        assert result == tmp_path / "a"


# ---------------------------------------------------------------------------
# write_materials_yaml
# ---------------------------------------------------------------------------


class TestWriteMaterialsYaml:
    def test_basic_output(self, tmp_path: Path):
        usd_file = tmp_path / "mats.usd"
        usd_file.touch()

        prim_paths = ["/World/Looks/Gold", "/World/Looks/Silver_Brushed"]
        thumbnails = {"/World/Looks/Gold": tmp_path / "Gold.png"}
        descriptions = {"/World/Looks/Gold": "Gold is a warm metallic surface."}

        result = write_materials_yaml(
            output_dir=tmp_path,
            usd_file=usd_file,
            prim_paths=prim_paths,
            thumbnails=thumbnails,
            descriptions=descriptions,
            image_size=256,
            library_path=None,
        )

        assert result == tmp_path / "materials.yaml"
        assert result.exists()

        with open(result) as f:
            content = f.read()
        # Skip comment lines for YAML parsing
        data = yaml.safe_load(content)

        assert "library_path" in data
        assert len(data["entries"]) == 2

        gold = data["entries"][0]
        assert gold["name"] == "Gold"
        assert gold["binding"] == "/World/Looks/Gold"
        assert gold["description"] == "Gold is a warm metallic surface."
        assert gold["icon"] == "thumbs/256x256/Gold.png"

        silver = data["entries"][1]
        assert silver["name"] == "Silver Brushed"
        assert silver["binding"] == "/World/Looks/Silver_Brushed"
        assert silver["description"] == ""
        assert silver["icon"] == ""  # not in thumbnails dict

    def test_custom_library_path(self, tmp_path: Path):
        usd_file = tmp_path / "mats.usd"
        usd_file.touch()

        result = write_materials_yaml(
            output_dir=tmp_path,
            usd_file=usd_file,
            prim_paths=["/Mat"],
            thumbnails={},
            descriptions={},
            image_size=256,
            library_path="/custom/path/materials.usd",
        )

        data = yaml.safe_load(open(result))
        assert data["library_path"] == "/custom/path/materials.usd"

    def test_relative_library_path_when_none(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        usd_file = sub / "mats.usd"
        usd_file.touch()

        result = write_materials_yaml(
            output_dir=tmp_path,
            usd_file=usd_file,
            prim_paths=["/Mat"],
            thumbnails={},
            descriptions={},
            image_size=512,
            library_path=None,
        )

        data = yaml.safe_load(open(result))
        # Should be relative from output_dir to usd_file
        assert data["library_path"] == "sub/mats.usd"

    def test_image_size_in_icon_path(self, tmp_path: Path):
        usd_file = tmp_path / "mats.usd"
        usd_file.touch()

        write_materials_yaml(
            output_dir=tmp_path,
            usd_file=usd_file,
            prim_paths=["/World/Mat"],
            thumbnails={"/World/Mat": tmp_path / "Mat.png"},
            descriptions={},
            image_size=512,
            library_path=None,
        )

        data = yaml.safe_load(open(tmp_path / "materials.yaml"))
        assert data["entries"][0]["icon"] == "thumbs/512x512/Mat.png"


# ---------------------------------------------------------------------------
# discover_materials
# ---------------------------------------------------------------------------


class TestDiscoverMaterials:
    def test_discovers_materials_from_usda(self, tmp_path: Path):
        usda = tmp_path / "test_materials.usda"
        usda.write_text(
            """\
#usda 1.0

def Scope "Looks"
{
    def Material "Gold"
    {
    }

    def Material "Silver"
    {
    }

    def Shader "NotAMaterial"
    {
    }
}
"""
        )

        result = discover_materials(usda)

        assert len(result) == 2
        assert "/Looks/Gold" in result
        assert "/Looks/Silver" in result

    def test_empty_stage_returns_empty(self, tmp_path: Path):
        usda = tmp_path / "empty.usda"
        usda.write_text("#usda 1.0\n")

        result = discover_materials(usda)

        assert result == []

    def test_nested_materials(self, tmp_path: Path):
        usda = tmp_path / "nested.usda"
        usda.write_text(
            """\
#usda 1.0

def Scope "Library"
{
    def Scope "Metals"
    {
        def Material "Copper"
        {
        }
    }

    def Scope "Plastics"
    {
        def Material "ABS_White"
        {
        }
    }
}
"""
        )

        result = discover_materials(usda)

        assert len(result) == 2
        assert "/Library/Metals/Copper" in result
        assert "/Library/Plastics/ABS_White" in result

    def test_invalid_usd_raises(self, tmp_path: Path):
        from pxr import Tf

        bad_file = tmp_path / "bad.usda"
        bad_file.write_text("this is not valid USD")

        with pytest.raises(Tf.ErrorException):
            discover_materials(bad_file)


# ---------------------------------------------------------------------------
# run_generate_manifest — error & list paths
# ---------------------------------------------------------------------------


class TestRunGenerateManifestErrors:
    def test_missing_usd_file(self, tmp_path: Path):
        params = GenerateManifestInput(
            usd_file=tmp_path / "nonexistent.usd",
            output_dir=tmp_path / "out",
        )
        result = run_generate_manifest(params)

        assert not result.success
        assert "not found" in result.error

    def test_no_materials_found(self, tmp_path: Path):
        usda = tmp_path / "empty.usda"
        usda.write_text("#usda 1.0\n")

        params = GenerateManifestInput(
            usd_file=usda,
            output_dir=tmp_path / "out",
        )
        result = run_generate_manifest(params)

        assert not result.success
        assert "No materials" in result.error

    def test_missing_template(self, tmp_path: Path):
        usda = tmp_path / "mats.usda"
        usda.write_text('#usda 1.0\ndef Material "Gold"\n{\n}\n')

        params = GenerateManifestInput(
            usd_file=usda,
            output_dir=tmp_path / "out",
            template=tmp_path / "nonexistent_template.usd",
        )
        result = run_generate_manifest(params)

        assert not result.success
        assert "Template USD not found" in result.error

    def test_list_materials_mode(self, tmp_path: Path):
        usda = tmp_path / "mats.usda"
        usda.write_text(
            """\
#usda 1.0

def Material "Gold"
{
}

def Material "Silver"
{
}
"""
        )

        params = GenerateManifestInput(
            usd_file=usda,
            output_dir=tmp_path / "out",
            list_materials=True,
        )
        result = run_generate_manifest(params)

        assert result.success
        assert result.materials_count == 2
        assert len(result.material_paths) == 2
        assert "/Gold" in result.material_paths
        assert "/Silver" in result.material_paths
        # list mode should not create output dir or yaml
        assert result.yaml_path is None

    def test_creates_output_dir(self, tmp_path: Path):
        """run_generate_manifest creates output_dir if it doesn't exist."""
        usda = tmp_path / "mats.usda"
        usda.write_text('#usda 1.0\ndef Material "Gold"\n{\n}\n')

        out_dir = tmp_path / "nested" / "output"
        assert not out_dir.exists()

        params = GenerateManifestInput(
            usd_file=usda,
            output_dir=out_dir,
            template=tmp_path / "no_template.usd",  # will fail at template check
        )
        result = run_generate_manifest(params)

        # It should fail at template, but output_dir should have been created
        assert not result.success
        assert out_dir.exists()


class TestGenerateManifestResult:
    def test_default_values(self):
        result = GenerateManifestResult(success=True)
        assert result.yaml_path is None
        assert result.materials_count == 0
        assert result.thumbnails_count == 0
        assert result.descriptions_count == 0
        assert result.error is None
        assert result.material_paths == []

    def test_populated_values(self):
        result = GenerateManifestResult(
            success=True,
            yaml_path=Path("/out/materials.yaml"),
            materials_count=5,
            thumbnails_count=4,
            descriptions_count=3,
            material_paths=["/A", "/B"],
        )
        assert result.materials_count == 5
        assert result.thumbnails_count == 4
        assert result.descriptions_count == 3
        assert len(result.material_paths) == 2
