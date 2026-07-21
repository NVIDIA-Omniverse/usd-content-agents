# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scene simulate module: mock prediction generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom

from material_agent.scene.simulate import (
    generate_mock_predictions,
    generate_mock_predictions_append,
    load_material_names_from_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_stage_with_meshes(
    path: Path,
    mesh_names: list[str],
    root: str = "/Root",
) -> Path:
    """Create a minimal USD stage with Mesh prims under *root*."""
    stage = Usd.Stage.CreateNew(str(path))
    root_prim = UsdGeom.Xform.Define(stage, root)
    stage.SetDefaultPrim(root_prim.GetPrim())
    for name in mesh_names:
        UsdGeom.Mesh.Define(stage, f"{root}/{name}")
    stage.GetRootLayer().Save()
    return path


def _read_predictions(path: Path) -> list[dict]:
    """Read a JSONL predictions file and return list of dicts."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# generate_mock_predictions
# ---------------------------------------------------------------------------


class TestGenerateMockPredictions:
    """Tests for generate_mock_predictions."""

    def test_basic_round_robin(self, tmp_path: Path) -> None:
        """Multiple meshes get materials assigned round-robin."""
        usd_path = _create_stage_with_meshes(
            tmp_path / "scene.usda",
            ["MeshA", "MeshB", "MeshC", "MeshD", "MeshE"],
        )
        materials = ["MatX", "MatY"]
        out = tmp_path / "predictions.jsonl"

        count = generate_mock_predictions(usd_path, materials, out)

        assert count == 5
        preds = _read_predictions(out)
        assert len(preds) == 5
        # Round-robin: 0->MatX, 1->MatY, 2->MatX, 3->MatY, 4->MatX
        assigned = [p["materials"]["material"] for p in preds]
        assert assigned == ["MatX", "MatY", "MatX", "MatY", "MatX"]

    def test_single_material(self, tmp_path: Path) -> None:
        """All prims get the same material when only one is provided."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["A", "B", "C"])
        out = tmp_path / "preds.jsonl"

        count = generate_mock_predictions(usd_path, ["OnlyMat"], out)

        assert count == 3
        preds = _read_predictions(out)
        assert all(p["materials"]["material"] == "OnlyMat" for p in preds)

    def test_jsonl_format(self, tmp_path: Path) -> None:
        """Each line is valid JSON with expected keys."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["Mesh1"])
        out = tmp_path / "preds.jsonl"
        generate_mock_predictions(usd_path, ["Mat1"], out)

        preds = _read_predictions(out)
        assert len(preds) == 1
        pred = preds[0]
        assert "id" in pred
        assert "materials" in pred
        assert "material" in pred["materials"]
        assert pred["id"] == "/Root/Mesh1"
        assert pred["materials"]["material"] == "Mat1"

    def test_empty_stage_no_meshes(self, tmp_path: Path) -> None:
        """Stage with no geometry prims produces zero predictions."""
        stage_path = tmp_path / "empty.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        xform = UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(xform.GetPrim())
        stage.GetRootLayer().Save()

        out = tmp_path / "preds.jsonl"
        count = generate_mock_predictions(stage_path, ["Mat1"], out)

        assert count == 0
        assert out.exists()
        preds = _read_predictions(out)
        assert preds == []

    def test_empty_material_names_raises(self, tmp_path: Path) -> None:
        """Empty material_names raises ValueError."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["Mesh1"])
        out = tmp_path / "preds.jsonl"

        with pytest.raises(ValueError, match="material_names must not be empty"):
            generate_mock_predictions(usd_path, [], out)

    def test_prim_path_scope(self, tmp_path: Path) -> None:
        """Only prims under prim_path_scope are included."""
        stage_path = tmp_path / "scoped.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        root = UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Xform.Define(stage, "/Root/GroupA")
        UsdGeom.Mesh.Define(stage, "/Root/GroupA/Mesh1")
        UsdGeom.Mesh.Define(stage, "/Root/GroupA/Mesh2")
        UsdGeom.Xform.Define(stage, "/Root/GroupB")
        UsdGeom.Mesh.Define(stage, "/Root/GroupB/Mesh3")
        stage.GetRootLayer().Save()

        out = tmp_path / "preds.jsonl"
        count = generate_mock_predictions(
            stage_path, ["Mat1"], out, prim_path_scope="/Root/GroupA"
        )

        assert count == 2
        preds = _read_predictions(out)
        ids = [p["id"] for p in preds]
        assert all(i.startswith("/Root/GroupA") for i in ids)
        assert "/Root/GroupB/Mesh3" not in ids

    def test_multiple_geometry_types(self, tmp_path: Path) -> None:
        """Non-Mesh geometry types (Cube, Sphere, etc.) are included."""
        stage_path = tmp_path / "multi_geom.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        root = UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/Root/MyMesh")
        UsdGeom.Cube.Define(stage, "/Root/MyCube")
        UsdGeom.Sphere.Define(stage, "/Root/MySphere")
        UsdGeom.Cylinder.Define(stage, "/Root/MyCylinder")
        # Xform should NOT be included
        UsdGeom.Xform.Define(stage, "/Root/MyXform")
        stage.GetRootLayer().Save()

        out = tmp_path / "preds.jsonl"
        count = generate_mock_predictions(stage_path, ["Mat1", "Mat2"], out)

        assert count == 4
        preds = _read_predictions(out)
        ids = {p["id"] for p in preds}
        assert "/Root/MyMesh" in ids
        assert "/Root/MyCube" in ids
        assert "/Root/MySphere" in ids
        assert "/Root/MyCylinder" in ids
        assert "/Root/MyXform" not in ids

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Output path's parent directories are created automatically."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["Mesh1"])
        out = tmp_path / "nested" / "deep" / "preds.jsonl"

        count = generate_mock_predictions(usd_path, ["Mat1"], out)

        assert count == 1
        assert out.exists()

    def test_returns_count(self, tmp_path: Path) -> None:
        """Return value matches number of predictions written."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["A", "B", "C"])
        out = tmp_path / "preds.jsonl"

        count = generate_mock_predictions(usd_path, ["M1", "M2"], out)

        assert count == 3
        assert len(_read_predictions(out)) == count


# ---------------------------------------------------------------------------
# generate_mock_predictions_append
# ---------------------------------------------------------------------------


class TestGenerateMockPredictionsAppend:
    """Tests for generate_mock_predictions_append."""

    def test_append_skips_existing(self, tmp_path: Path) -> None:
        """Already-predicted prim paths are not duplicated."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["A", "B", "C"])
        out = tmp_path / "preds.jsonl"

        # Write initial predictions for A only
        with open(out, "w") as f:
            f.write(
                json.dumps({"id": "/Root/A", "materials": {"material": "Existing"}})
                + "\n"
            )

        count = generate_mock_predictions_append(usd_path, ["NewMat"], out)

        assert count == 2  # B and C only
        preds = _read_predictions(out)
        assert len(preds) == 3  # original A + appended B, C
        ids = [p["id"] for p in preds]
        assert "/Root/A" in ids
        assert "/Root/B" in ids
        assert "/Root/C" in ids
        # A should keep original material
        a_pred = next(p for p in preds if p["id"] == "/Root/A")
        assert a_pred["materials"]["material"] == "Existing"

    def test_append_to_nonexistent_file(self, tmp_path: Path) -> None:
        """Works when output file doesn't exist yet."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["X", "Y"])
        out = tmp_path / "preds.jsonl"

        count = generate_mock_predictions_append(usd_path, ["Mat1"], out)

        assert count == 2
        preds = _read_predictions(out)
        assert len(preds) == 2

    def test_append_empty_materials_returns_zero(self, tmp_path: Path) -> None:
        """Empty material_names returns 0 immediately."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["A"])
        out = tmp_path / "preds.jsonl"

        count = generate_mock_predictions_append(usd_path, [], out)

        assert count == 0

    def test_append_null_stage_returns_zero(self, monkeypatch, tmp_path: Path) -> None:
        """Null USD stages do not append predictions."""
        out = tmp_path / "preds.jsonl"
        monkeypatch.setattr(Usd.Stage, "Open", lambda *_args: None)

        count = generate_mock_predictions_append(
            tmp_path / "null.usda",
            ["Mat1"],
            out,
        )

        assert count == 0

    def test_append_no_new_prims(self, tmp_path: Path) -> None:
        """Returns 0 when all prims already exist in predictions."""
        usd_path = _create_stage_with_meshes(tmp_path / "scene.usda", ["A", "B"])
        out = tmp_path / "preds.jsonl"

        # Pre-populate with all prims
        with open(out, "w") as f:
            f.write(
                json.dumps({"id": "/Root/A", "materials": {"material": "M1"}}) + "\n"
            )
            f.write(
                json.dumps({"id": "/Root/B", "materials": {"material": "M2"}}) + "\n"
            )

        count = generate_mock_predictions_append(usd_path, ["NewMat"], out)

        assert count == 0
        # File should be unchanged (2 lines)
        preds = _read_predictions(out)
        assert len(preds) == 2

    def test_append_round_robin_for_new_prims(self, tmp_path: Path) -> None:
        """New prims get round-robin materials starting from index 0."""
        usd_path = _create_stage_with_meshes(
            tmp_path / "scene.usda", ["A", "B", "C", "D"]
        )
        out = tmp_path / "preds.jsonl"

        # Pre-populate A and C
        with open(out, "w") as f:
            f.write(
                json.dumps({"id": "/Root/A", "materials": {"material": "Old"}}) + "\n"
            )
            f.write(
                json.dumps({"id": "/Root/C", "materials": {"material": "Old"}}) + "\n"
            )

        count = generate_mock_predictions_append(usd_path, ["MatX", "MatY"], out)

        assert count == 2  # B and D
        preds = _read_predictions(out)
        new_preds = [p for p in preds if p["materials"]["material"] != "Old"]
        assigned = [p["materials"]["material"] for p in new_preds]
        assert assigned == ["MatX", "MatY"]

    def test_append_with_prim_path_scope(self, tmp_path: Path) -> None:
        """prim_path_scope is respected during append."""
        stage_path = tmp_path / "scoped.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        root = UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Xform.Define(stage, "/Root/GroupA")
        UsdGeom.Mesh.Define(stage, "/Root/GroupA/M1")
        UsdGeom.Xform.Define(stage, "/Root/GroupB")
        UsdGeom.Mesh.Define(stage, "/Root/GroupB/M2")
        stage.GetRootLayer().Save()

        out = tmp_path / "preds.jsonl"

        count = generate_mock_predictions_append(
            stage_path, ["Mat1"], out, prim_path_scope="/Root/GroupA"
        )

        assert count == 1
        preds = _read_predictions(out)
        assert preds[0]["id"] == "/Root/GroupA/M1"


# ---------------------------------------------------------------------------
# load_material_names_from_config
# ---------------------------------------------------------------------------


class TestLoadMaterialNamesFromConfig:
    """Tests for load_material_names_from_config."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Loads material names from a YAML referenced by config."""
        mat_yaml = tmp_path / "materials.yaml"
        mat_yaml.write_text(
            "entries:\n  - name: Steel\n  - name: Wood\n  - name: Glass\n"
        )
        config_path = tmp_path / "config.yaml"
        config_path.touch()

        scene_config = {"materials": {"path": "materials.yaml"}}
        names = load_material_names_from_config(scene_config, config_path)

        assert names == ["Steel", "Wood", "Glass"]

    def test_absolute_path(self, tmp_path: Path) -> None:
        """Absolute materials path works."""
        mat_yaml = tmp_path / "subdir" / "mats.yaml"
        mat_yaml.parent.mkdir()
        mat_yaml.write_text("entries:\n  - name: Concrete\n")

        config_path = tmp_path / "config.yaml"
        config_path.touch()

        scene_config = {"materials": {"path": str(mat_yaml)}}
        names = load_material_names_from_config(scene_config, config_path)

        assert names == ["Concrete"]

    def test_no_materials_path_raises(self, tmp_path: Path) -> None:
        """Missing materials.path raises ValueError."""
        config_path = tmp_path / "config.yaml"
        config_path.touch()

        with pytest.raises(ValueError, match="No materials.path configured"):
            load_material_names_from_config({}, config_path)

    def test_missing_materials_section_raises(self, tmp_path: Path) -> None:
        """Empty materials section raises ValueError."""
        config_path = tmp_path / "config.yaml"
        config_path.touch()

        with pytest.raises(ValueError, match="No materials.path configured"):
            load_material_names_from_config({"materials": {}}, config_path)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Non-existent materials YAML raises FileNotFoundError."""
        config_path = tmp_path / "config.yaml"
        config_path.touch()

        scene_config = {"materials": {"path": "nonexistent.yaml"}}
        with pytest.raises(FileNotFoundError, match="Materials YAML not found"):
            load_material_names_from_config(scene_config, config_path)

    def test_no_entries_raises(self, tmp_path: Path) -> None:
        """YAML with no entries raises ValueError."""
        mat_yaml = tmp_path / "materials.yaml"
        mat_yaml.write_text("entries: []\n")
        config_path = tmp_path / "config.yaml"
        config_path.touch()

        scene_config = {"materials": {"path": "materials.yaml"}}
        with pytest.raises(ValueError, match="No material entries found"):
            load_material_names_from_config(scene_config, config_path)

    def test_entries_with_empty_names_skipped(self, tmp_path: Path) -> None:
        """Entries with empty or missing name are filtered out."""
        mat_yaml = tmp_path / "materials.yaml"
        mat_yaml.write_text(
            'entries:\n  - name: Valid\n  - name: ""\n  - other_key: no_name\n'
        )
        config_path = tmp_path / "config.yaml"
        config_path.touch()

        scene_config = {"materials": {"path": "materials.yaml"}}
        names = load_material_names_from_config(scene_config, config_path)

        assert names == ["Valid"]

    def test_relative_path_resolution(self, tmp_path: Path) -> None:
        """Relative path is resolved relative to config_path's parent."""
        subdir = tmp_path / "configs"
        subdir.mkdir()
        config_path = subdir / "scene.yaml"
        config_path.touch()

        mat_yaml = subdir / "libs" / "materials.yaml"
        mat_yaml.parent.mkdir()
        mat_yaml.write_text("entries:\n  - name: Rubber\n")

        scene_config = {"materials": {"path": "libs/materials.yaml"}}
        names = load_material_names_from_config(scene_config, config_path)

        assert names == ["Rubber"]
