# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import errno
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from material_agent.scene.config_gen import (
    CredentialOverlayShapeError,
    MissingCredentialSourceError,
    _drop_redacted_credentials,
    _has_nonsecret_context,
    _inject_split_context,
    _missing_credential_error,
    _overlay_inline_credentials,
    _rebase_paths,
    _sanitize_name,
    _unique_safe_names,
    _value_at_path,
    generate_all_configs,
    generate_all_payload_configs,
    generate_payload_config,
    generate_sub_asset_config,
    prepare_payload_runtime_configs,
    prepare_sub_asset_runtime_config,
    prepare_sub_asset_runtime_configs,
)
from material_agent.scene.manifest import PayloadGroup, SceneManifest, SubAsset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_scene_config() -> dict:
    """Minimal scene config template used across tests."""
    return {
        "project": {"name": "test"},
        "input": {"usd_path": "scene.usda"},
        "output": {"format": "usda"},
        "steps": {
            "apply": {"enabled": True},
            "render": {"enabled": True},
            "restore_usd": {"enabled": False},
        },
        "scene": {"analyze": {"some_key": True}},
    }


def _make_sub_asset(
    *,
    id: str = "sa1",
    name: str = "Ladder",
    prim_path: str = "/Root/Ladder",
    **kwargs,
) -> SubAsset:
    return SubAsset(id=id, name=name, prim_path=prim_path, **kwargs)


def _make_payload_group(
    *,
    id: str = "pg1",
    group_name: str = "Tray",
    payload_file: str = "/assets/Tray/Tray.usd",
    **kwargs,
) -> PayloadGroup:
    return PayloadGroup(
        id=id, group_name=group_name, payload_file=payload_file, **kwargs
    )


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_simple_name(self):
        assert _sanitize_name("Ladder") == "ladder"

    def test_special_characters_stripped(self):
        assert _sanitize_name("My Object!@#$%") == "my_object"

    def test_spaces_become_underscores(self):
        assert _sanitize_name("hello world") == "hello_world"

    def test_consecutive_underscores_collapsed(self):
        assert _sanitize_name("a___b") == "a_b"

    def test_leading_trailing_underscores_stripped(self):
        assert _sanitize_name("__foo__") == "foo"

    def test_empty_string_returns_unnamed(self):
        assert _sanitize_name("") == "unnamed"

    def test_only_special_chars_returns_unnamed(self):
        assert _sanitize_name("@#$") == "unnamed"

    def test_hyphens_preserved(self):
        assert _sanitize_name("UR-5e") == "ur-5e"

    def test_digits_preserved(self):
        assert _sanitize_name("Part_007") == "part_007"

    def test_slash_replaced(self):
        result = _sanitize_name("/Root/Obj")
        assert "/" not in result


# ---------------------------------------------------------------------------
# _unique_safe_names
# ---------------------------------------------------------------------------


class TestUniqueSafeNames:
    def test_no_collisions(self):
        assets = [
            _make_sub_asset(id="a1", name="Alpha"),
            _make_sub_asset(id="a2", name="Beta"),
        ]
        result = _unique_safe_names(assets)
        assert result == {"a1": "alpha", "a2": "beta"}

    def test_collisions_get_id_suffix(self):
        assets = [
            _make_sub_asset(id="id_100", name="Widget"),
            _make_sub_asset(id="id_200", name="Widget"),
        ]
        result = _unique_safe_names(assets)
        # Both should have unique names with ID suffix
        assert result["id_100"] != result["id_200"]
        assert result["id_100"].startswith("widget_")
        assert result["id_200"].startswith("widget_")

    def test_unique_names_no_suffix(self):
        assets = [_make_sub_asset(id="x", name="Solo")]
        result = _unique_safe_names(assets)
        assert result["x"] == "solo"


# ---------------------------------------------------------------------------
# _rebase_paths
# ---------------------------------------------------------------------------


class TestRebasePaths:
    def test_relative_path_rebased(self, tmp_path: Path):
        old_base = tmp_path / "configs"
        new_base = tmp_path / "configs" / "sub"
        old_base.mkdir()
        new_base.mkdir()
        config = {"input": {"usd_path": "scene.usda"}}
        _rebase_paths(config, old_base, new_base)
        # From new_base, we need to go up one level to reach old_base/scene.usda
        assert config["input"]["usd_path"] == str(Path("..") / "scene.usda")

    def test_absolute_path_unchanged(self, tmp_path: Path):
        config = {"input": {"usd_path": "/absolute/scene.usda"}}
        _rebase_paths(config, tmp_path, tmp_path / "sub")
        assert config["input"]["usd_path"] == "/absolute/scene.usda"

    def test_nested_dict_rebased(self, tmp_path: Path):
        old_base = tmp_path / "a"
        new_base = tmp_path / "a" / "b"
        old_base.mkdir()
        new_base.mkdir()
        config = {"steps": {"optimize_usd": {"path": "data/model.usd"}}}
        _rebase_paths(config, old_base, new_base)
        assert config["steps"]["optimize_usd"]["path"] == str(
            Path("..") / "data" / "model.usd"
        )

    def test_path_list_keys_rebased(self, tmp_path: Path):
        old_base = tmp_path / "a"
        new_base = tmp_path / "a" / "b"
        old_base.mkdir()
        new_base.mkdir()
        config = {"reference_images": ["img1.png", "img2.png"]}
        _rebase_paths(config, old_base, new_base)
        for val in config["reference_images"]:
            assert val.startswith("..")

    def test_step1x_and_creation_reference_paths_rebased(self, tmp_path: Path):
        old_base = tmp_path / "source"
        new_base = tmp_path / "generated"
        old_base.mkdir()
        new_base.mkdir()
        config = {
            "steps": {
                "create_materials": {
                    "step1x": {
                        "runtime_dir": "runtime",
                        "model_dir": "models",
                        "edit_script": "scripts/edit.py",
                    },
                    "step1x_material_anything": {
                        "cache_dir": "cache",
                        "output_dir": "",
                        "python_executable": "venv/bin/python",
                    },
                    "creation_requests": [
                        {
                            "reference_image_uris": [
                                "references/request.png",
                                "https://example.test/request.png",
                            ],
                            "recipe": {"reference_image_uris": "references/recipe.png"},
                        }
                    ],
                }
            }
        }

        _rebase_paths(config, old_base, new_base)

        create_materials = config["steps"]["create_materials"]
        assert create_materials["step1x"] == {
            "runtime_dir": str(Path("..") / "source" / "runtime"),
            "model_dir": str(Path("..") / "source" / "models"),
            "edit_script": str(Path("..") / "source" / "scripts" / "edit.py"),
        }
        assert create_materials["step1x_material_anything"] == {
            "cache_dir": str(Path("..") / "source" / "cache"),
            "output_dir": "",
            "python_executable": str(Path("..") / "source" / "venv" / "bin" / "python"),
        }
        request = create_materials["creation_requests"][0]
        assert request["reference_image_uris"] == [
            str(Path("..") / "source" / "references" / "request.png"),
            "https://example.test/request.png",
        ]
        assert request["recipe"]["reference_image_uris"] == str(
            Path("..") / "source" / "references" / "recipe.png"
        )

    def test_non_path_keys_untouched(self, tmp_path: Path):
        config = {"input": {"description": "relative/looking/string"}}
        _rebase_paths(config, tmp_path, tmp_path / "sub")
        assert config["input"]["description"] == "relative/looking/string"

    def test_working_dir_rebased(self, tmp_path: Path):
        old_base = tmp_path / "root"
        new_base = tmp_path / "root" / "sub"
        old_base.mkdir()
        new_base.mkdir()
        config = {"project": {"working_dir": ".workdir"}}
        _rebase_paths(config, old_base, new_base)
        assert ".." in config["project"]["working_dir"]

    def test_rebase_paths_uses_absolute_fallback_on_relpath_failure(
        self, monkeypatch, tmp_path: Path
    ):
        old_base = tmp_path / "root"
        new_base = tmp_path / "other"
        old_base.mkdir()
        new_base.mkdir()
        config = {
            "input": {"usd_path": "scene.usda"},
            "reference_images": ["ref.png"],
        }
        monkeypatch.setattr(
            "material_agent.scene.config_gen.os.path.relpath",
            lambda *_args: (_ for _ in ()).throw(ValueError("cross drive")),
        )

        _rebase_paths(config, old_base, new_base)

        assert config["input"]["usd_path"] == str((old_base / "scene.usda").resolve())
        assert config["reference_images"] == [str((old_base / "ref.png").resolve())]


# ---------------------------------------------------------------------------
# generate_sub_asset_config
# ---------------------------------------------------------------------------


class TestGenerateSubAssetConfig:
    def test_basic_generation(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        out = tmp_path / "configs" / "ladder.yaml"

        result = generate_sub_asset_config(sa, config, out)
        assert result == out
        assert out.exists()

        data = _read_yaml(out)
        # scene section removed
        assert "scene" not in data
        # prim_path set
        assert data["input"]["prim_path"] == "/Root/Ladder"
        # layer_only forced
        assert data["output"]["layer_only"] is True
        assert data["output"]["flatten_output"] is False
        # apply and render disabled
        assert data["steps"]["apply"]["enabled"] is False
        assert data["steps"]["render"]["enabled"] is False
        # restore_usd enabled
        assert data["steps"]["restore_usd"]["enabled"] is True
        assert data["project"]["working_dir"] == ".ladder"

    def test_project_name_sanitized(self, tmp_path: Path):
        sa = _make_sub_asset(name="My Object!!")
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out)
        data = _read_yaml(out)
        assert data["project"]["name"] == "my_object"
        assert data["project"]["session_id"] == "my_object"

    def test_session_id_override(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out, session_id="custom_session")
        data = _read_yaml(out)
        assert data["project"]["name"] == "custom_session"
        assert data["project"]["session_id"] == "custom_session"
        assert data["project"]["working_dir"] == ".custom_session"

    def test_scene_working_dir_replaced_with_per_asset_dir(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        config["project"]["working_dir"] = str(tmp_path / "scene")
        out = tmp_path / "configs" / "out.yaml"

        generate_sub_asset_config(sa, config, out, scene_config_dir=tmp_path)
        data = _read_yaml(out)
        assert data["project"]["working_dir"] == ".ladder"

    def test_path_rebasing(self, tmp_path: Path):
        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        configs_dir = tmp_path / "scene" / "configs"
        configs_dir.mkdir()

        sa = _make_sub_asset()
        config = {"input": {"usd_path": "model.usda"}, "steps": {}}
        out = configs_dir / "asset.yaml"

        generate_sub_asset_config(sa, config, out, scene_config_dir=scene_dir)
        data = _read_yaml(out)
        # usd_path should be rebased: from configs/, go up to scene/
        assert data["input"]["usd_path"] == str(Path("..") / "model.usda")

    def test_does_not_mutate_original_config(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        original_keys = set(config.keys())
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out)
        # Original config should still have "scene" key
        assert "scene" in config
        assert set(config.keys()) == original_keys

    def test_extracted_usd_used_when_exists(self, tmp_path: Path):
        extracted = tmp_path / "extracted" / "ladder.usda"
        extracted.parent.mkdir()
        extracted.write_text("# extracted")

        sa = _make_sub_asset(extracted_usd=str(extracted))
        config = _base_scene_config()
        out = tmp_path / "configs" / "ladder.yaml"

        generate_sub_asset_config(sa, config, out)
        data = _read_yaml(out)
        # Should use a relative path to the extracted USD
        assert "extracted" in data["input"]["usd_path"]

    def test_extracted_usd_uses_absolute_fallback_on_relpath_failure(
        self, monkeypatch, tmp_path: Path
    ):
        extracted = tmp_path / "extracted" / "ladder.usda"
        extracted.parent.mkdir()
        extracted.write_text("# extracted")
        monkeypatch.setattr(
            "material_agent.scene.config_gen.os.path.relpath",
            lambda *_args: (_ for _ in ()).throw(ValueError("cross drive")),
        )

        sa = _make_sub_asset(extracted_usd=str(extracted))
        out = tmp_path / "configs" / "ladder.yaml"

        generate_sub_asset_config(sa, _base_scene_config(), out)

        assert _read_yaml(out)["input"]["usd_path"] == str(extracted.resolve())

    def test_split_context_injected(self, tmp_path: Path):
        sa = _make_sub_asset(
            split_context={
                "parent_name": "BigMachine",
                "sibling_names": ["Arm", "Ladder"],
                "ancestors": ["Factory", "Line_01"],
            }
        )
        config = _base_scene_config()
        config["steps"]["build_dataset_prepare_dataset"] = {
            "prompts": {"vlm_system": "You are a material expert."}
        }
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out)
        data = _read_yaml(out)
        vlm_system = data["steps"]["build_dataset_prepare_dataset"]["prompts"][
            "vlm_system"
        ]
        assert "extracted from a larger structure" in vlm_system
        assert "Factory" in vlm_system

    def test_output_is_valid_yaml(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out)
        # Should not raise
        data = _read_yaml(out)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# generate_all_configs
# ---------------------------------------------------------------------------


class TestGenerateAllConfigs:
    def test_generates_configs_for_all_assets(self, tmp_path: Path):
        manifest = SceneManifest(
            sub_assets=[
                _make_sub_asset(id="a1", name="Alpha", prim_path="/Root/Alpha"),
                _make_sub_asset(id="a2", name="Beta", prim_path="/Root/Beta"),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        result = generate_all_configs(manifest, config, configs_dir)
        assert len(list(configs_dir.glob("*.yaml"))) == 2

        # Each asset should have config_path and working_dir set
        for sa in result.sub_assets:
            assert sa.config_path is not None
            assert sa.working_dir is not None

    def test_names_filter_limits_output(self, tmp_path: Path):
        manifest = SceneManifest(
            sub_assets=[
                _make_sub_asset(id="a1", name="Alpha", prim_path="/Root/Alpha"),
                _make_sub_asset(id="a2", name="Beta", prim_path="/Root/Beta"),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        generate_all_configs(manifest, config, configs_dir, names_filter=["Alpha"])
        # Only Alpha should have a config
        assert manifest.sub_assets[0].config_path is not None
        assert manifest.sub_assets[1].config_path is None

    def test_collision_handling(self, tmp_path: Path):
        manifest = SceneManifest(
            sub_assets=[
                _make_sub_asset(id="id_1", name="Widget", prim_path="/Root/W1"),
                _make_sub_asset(id="id_2", name="Widget", prim_path="/Root/W2"),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        generate_all_configs(manifest, config, configs_dir)
        files = list(configs_dir.glob("*.yaml"))
        assert len(files) == 2
        # File names should be different
        names = {f.stem for f in files}
        assert len(names) == 2

    def test_generation_failure_marks_asset_failed(self, monkeypatch, tmp_path: Path):
        manifest = SceneManifest(
            sub_assets=[_make_sub_asset(id="a1", name="Alpha", prim_path="/Root/Alpha")]
        )
        monkeypatch.setattr(
            "material_agent.scene.config_gen.generate_sub_asset_config",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        result = generate_all_configs(
            manifest, _base_scene_config(), tmp_path / "configs"
        )

        assert result.sub_assets[0].status == "failed"


# ---------------------------------------------------------------------------
# generate_payload_config
# ---------------------------------------------------------------------------


class TestGeneratePayloadConfig:
    def test_basic_generation(self, tmp_path: Path):
        payload_file = tmp_path / "assets" / "Tray.usd"
        payload_file.parent.mkdir(parents=True)
        payload_file.write_text("# payload")

        pg = _make_payload_group(payload_file=str(payload_file))
        config = _base_scene_config()
        out = tmp_path / "configs" / "tray.yaml"

        result = generate_payload_config(pg, config, out)
        assert result == out
        assert out.exists()

        data = _read_yaml(out)
        # scene section removed
        assert "scene" not in data
        # prim_path removed (no scoping for payloads)
        assert "prim_path" not in data.get("input", {})
        # layer_only forced
        assert data["output"]["layer_only"] is True
        # apply enabled with layer_only
        assert data["steps"]["apply"]["enabled"] is True
        assert data["steps"]["apply"]["layer_only"] is True
        assert data["steps"]["apply"]["skip_instance_check"] is True
        # render disabled
        assert data["steps"]["render"]["enabled"] is False
        # restore_usd enabled
        assert data["steps"]["restore_usd"]["enabled"] is True
        assert data["project"]["working_dir"] == ".tray"

    def test_project_identity_set(self, tmp_path: Path):
        payload_file = tmp_path / "Tray.usd"
        payload_file.write_text("# payload")

        pg = _make_payload_group(group_name="my_tray", payload_file=str(payload_file))
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        assert data["project"]["name"] == "my_tray"
        assert data["project"]["session_id"] == "my_tray"
        assert data["project"]["working_dir"] == ".my_tray"

    def test_container_payload_disables_so(self, tmp_path: Path):
        payload_file = tmp_path / "Parent.usd"
        payload_file.write_text("# parent")

        pg = _make_payload_group(
            payload_file=str(payload_file),
            child_payload_files=["/child1.usd", "/child2.usd"],
        )
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        assert data["steps"]["optimize_usd"]["enabled"] is False

    def test_representative_payload_sets_so_options(self, tmp_path: Path):
        payload_file = tmp_path / "Tray.usd"
        payload_file.write_text("# payload")
        rep_file = tmp_path / "Tray_rep.usd"
        rep_file.write_text("# representative")

        pg = _make_payload_group(
            payload_file=str(payload_file),
            representative_path=str(rep_file),
        )
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        so_settings = data["steps"]["optimize_usd"]["scene_optimizer_settings"]
        assert so_settings["enableDeinstance"] is False
        assert so_settings["enableSplitMeshes"] is True
        assert so_settings["enableDeduplicate"] is False
        # Should use representative path as input
        assert "Tray_rep" in data["input"]["usd_path"]
        # Should store original payload path
        assert "_original_payload_file" in data

    def test_modified_input_path_used(self, tmp_path: Path):
        payload_file = tmp_path / "Original.usd"
        payload_file.write_text("# orig")
        modified_file = tmp_path / "Modified.usd"
        modified_file.write_text("# modified")

        pg = _make_payload_group(
            payload_file=str(payload_file),
            modified_input_path=str(modified_file),
        )
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        assert "Modified" in data["input"]["usd_path"]

    def test_payload_config_rebases_scene_paths(self, tmp_path: Path):
        scene_dir = tmp_path / "scene"
        payload_file = scene_dir / "payloads" / "Tray.usd"
        payload_file.parent.mkdir(parents=True)
        payload_file.write_text("# payload")
        configs_dir = scene_dir / "configs"

        pg = _make_payload_group(payload_file=str(payload_file))
        config = {"input": {"usd_path": "scene.usda"}, "steps": {}}
        out = configs_dir / "tray.yaml"

        generate_payload_config(pg, config, out, scene_config_dir=scene_dir)
        data = _read_yaml(out)

        assert data["input"]["usd_path"] == str(Path("..") / "payloads" / "Tray.usd")

    def test_payload_config_uses_absolute_fallback_on_relpath_failure(
        self, monkeypatch, tmp_path: Path
    ):
        payload_file = tmp_path / "Tray.usd"
        payload_file.write_text("# payload")
        monkeypatch.setattr(
            "material_agent.scene.config_gen.os.path.relpath",
            lambda *_args: (_ for _ in ()).throw(ValueError("cross drive")),
        )

        out = tmp_path / "configs" / "tray.yaml"
        generate_payload_config(
            _make_payload_group(payload_file=str(payload_file)),
            _base_scene_config(),
            out,
        )

        assert _read_yaml(out)["input"]["usd_path"] == str(payload_file.resolve())

    def test_payload_context_injected(self, tmp_path: Path):
        payload_file = (
            tmp_path / "Assets" / "Phase_01" / "Machine" / "Tray" / "Tray.usd"
        )
        payload_file.parent.mkdir(parents=True)
        payload_file.write_text("# tray")

        pg = _make_payload_group(payload_file=str(payload_file), group_name="Tray")
        config = _base_scene_config()
        config["steps"]["build_dataset_prepare_dataset"] = {
            "prompts": {"vlm_system": "You are a material expert."}
        }
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        vlm = data["steps"]["build_dataset_prepare_dataset"]["prompts"]["vlm_system"]
        assert "industrial/warehouse" in vlm

    def test_split_context_helper_returns_when_absent(self) -> None:
        config = {}

        _inject_split_context(config, _make_sub_asset(split_context=None))

        assert config == {}

    def test_does_not_mutate_original_config(self, tmp_path: Path):
        payload_file = tmp_path / "Tray.usd"
        payload_file.write_text("# payload")

        pg = _make_payload_group(payload_file=str(payload_file))
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        assert "scene" in config


# ---------------------------------------------------------------------------
# generate_all_payload_configs
# ---------------------------------------------------------------------------


class TestGenerateAllPayloadConfigs:
    def test_generates_configs_for_all_payloads(self, tmp_path: Path):
        pf1 = tmp_path / "A.usd"
        pf2 = tmp_path / "B.usd"
        pf1.write_text("# a")
        pf2.write_text("# b")

        manifest = SceneManifest(
            payload_groups=[
                _make_payload_group(id="p1", group_name="A", payload_file=str(pf1)),
                _make_payload_group(id="p2", group_name="B", payload_file=str(pf2)),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        result = generate_all_payload_configs(manifest, config, configs_dir)
        payload_dir = configs_dir / "payloads"
        assert len(list(payload_dir.glob("*.yaml"))) == 2

        for pg in result.payload_groups:
            assert pg.config_path is not None
            assert pg.working_dir is not None

    def test_empty_payloads_returns_manifest(self, tmp_path: Path):
        manifest = SceneManifest(payload_groups=[])
        config = _base_scene_config()

        result = generate_all_payload_configs(manifest, config, tmp_path / "c")
        assert result is manifest

    def test_skipped_payloads_excluded(self, tmp_path: Path):
        pf = tmp_path / "A.usd"
        pf.write_text("# a")

        manifest = SceneManifest(
            payload_groups=[
                _make_payload_group(id="p1", group_name="A", payload_file=str(pf)),
                PayloadGroup(
                    id="p2",
                    group_name="B",
                    payload_file=str(pf),
                    status="skipped",
                ),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        generate_all_payload_configs(manifest, config, configs_dir)
        payload_dir = configs_dir / "payloads"
        files = list(payload_dir.glob("*.yaml"))
        assert len(files) == 1
        assert files[0].stem == "A"

    def test_payload_sibling_context_and_generation_failure(
        self, monkeypatch, tmp_path: Path
    ):
        parent_file = tmp_path / "Parent.usd"
        child_a = tmp_path / "Child_A.usd"
        child_b = tmp_path / "Child_B.usd"
        for path in (parent_file, child_a, child_b):
            path.write_text("# payload")

        manifest = SceneManifest(
            payload_groups=[
                PayloadGroup(
                    id="parent",
                    group_name="Parent",
                    payload_file=str(parent_file),
                    child_payload_files=[str(child_a), str(child_b)],
                    status="skipped",
                ),
                _make_payload_group(
                    id="a", group_name="Child_A", payload_file=str(child_a)
                ),
                _make_payload_group(
                    id="b", group_name="Child_B", payload_file=str(child_b)
                ),
            ]
        )
        config = _base_scene_config()
        config["steps"]["build_dataset_prepare_dataset"] = {
            "prompts": {"vlm_system": "Base prompt."}
        }

        generate_all_payload_configs(manifest, config, tmp_path / "configs")

        child_config = _read_yaml(tmp_path / "configs" / "payloads" / "Child_A.yaml")
        vlm_system = child_config["steps"]["build_dataset_prepare_dataset"]["prompts"][
            "vlm_system"
        ]
        assert "Sibling components in same system: Child B" in vlm_system

        monkeypatch.setattr(
            "material_agent.scene.config_gen.generate_payload_config",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        failed_manifest = SceneManifest(
            payload_groups=[
                _make_payload_group(
                    id="fail", group_name="Fail", payload_file=str(child_a)
                )
            ]
        )

        result = generate_all_payload_configs(
            failed_manifest, _base_scene_config(), tmp_path / "failed"
        )

        assert result.payload_groups[0].status == "failed"


class TestSceneCredentialTransport:
    """Security regressions for generated scene and payload configs."""

    @staticmethod
    def _credential_config() -> dict[str, Any]:
        config = _base_scene_config()
        config["steps"]["predict"] = {
            "vlm": {
                "backend": "nim",
                "api_key": "q7Z9",
                "providers": [{"name": "fallback", "token": "m4P8"}],
            }
        }
        return config

    def test_explicit_env_reference_survives_offline_generation_then_binding(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_name = "CUSTOM_OFFLINE_API_KEY"
        reference = "${CUSTOM_OFFLINE_API_KEY}"
        resolved_secret = "runtime-only-scene-key"
        monkeypatch.delenv(env_name, raising=False)
        config = _base_scene_config()
        config["steps"]["predict"] = {
            "vlm": {
                "backend": "openai",
                "api_key_env": reference,
            }
        }
        config_path = tmp_path / "generated.yaml"
        sub_asset = _make_sub_asset()

        generate_sub_asset_config(sub_asset, config, config_path)
        sub_asset.config_path = str(config_path)

        assert sub_asset.config_credential_paths == []
        persisted = config_path.read_text(encoding="utf-8")
        assert reference in persisted
        assert resolved_secret not in persisted

        monkeypatch.setenv(env_name, resolved_secret)
        runtime = prepare_sub_asset_runtime_config(sub_asset, config)

        assert runtime["steps"]["predict"]["vlm"]["api_key_env"] == reference
        assert sub_asset.config_credential_paths == []
        assert resolved_secret not in config_path.read_text(encoding="utf-8")

    def test_resume_normalizes_yaml_errors_without_rendering_source(
        self, tmp_path: Path
    ) -> None:
        sentinel = "never-render-this-scene-credential"
        config_path = tmp_path / "malformed-generated.yaml"
        config_path.write_text(
            f"api_key: [{sentinel}\n",
            encoding="utf-8",
        )
        sub_asset = _make_sub_asset(config_path=str(config_path))

        with pytest.raises(ValueError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, _base_scene_config())

        message = str(exc_info.value)
        assert message == f"Unable to parse generated scene config: {config_path}"
        assert sentinel not in message

    def test_resume_redacts_credential_bearing_path_from_yaml_error(
        self, tmp_path: Path
    ) -> None:
        path_secret = "material-generated-path-secret"
        config_dir = tmp_path / f"api_key={path_secret}"
        config_dir.mkdir()
        config_path = config_dir / "malformed-generated.yaml"
        config_path.write_text("steps: [unterminated\n", encoding="utf-8")
        sub_asset = _make_sub_asset(config_path=str(config_path))

        with pytest.raises(ValueError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, _base_scene_config())

        message = str(exc_info.value)
        assert message == "Unable to parse generated scene config: <redacted>"
        assert path_secret not in message
        assert "material-generated-path" not in message

    def test_resume_redacts_credential_bearing_path_from_mapping_error(
        self, tmp_path: Path
    ) -> None:
        path_secret = "material-nonmapping-path-secret"
        config_dir = tmp_path / f"token={path_secret}"
        config_dir.mkdir()
        config_path = config_dir / "legacy-generated.yaml"
        config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
        sub_asset = _make_sub_asset(config_path=str(config_path))

        with pytest.raises(ValueError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, _base_scene_config())

        message = str(exc_info.value)
        assert message == "Generated config must contain a mapping: <redacted>"
        assert path_secret not in message
        assert "material-nonmapping-path" not in message

    def test_legacy_scrub_redacts_path_in_warning_error_and_runtime_result(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path_secret = "material-scrub-path-secret"
        config_dir = tmp_path / f"api_key={path_secret}"
        config_dir.mkdir()
        config_path = config_dir / "legacy.yaml"
        unsafe = self._credential_config()
        unsafe.pop("scene")
        config_path.write_text(yaml.safe_dump(unsafe), encoding="utf-8")
        sub_asset = _make_sub_asset(config_path=str(config_path))
        caplog.set_level(logging.WARNING, logger="material_agent.scene.config_gen")

        runtime = prepare_sub_asset_runtime_config(
            sub_asset,
            self._credential_config(),
        )
        with pytest.raises(ValueError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, _base_scene_config())

        diagnostic_text = f"{exc_info.value}\n{caplog.text}\n{runtime!r}"
        assert "Removed legacy inline credentials" in caplog.text
        assert "Cannot rehydrate generated scene config <redacted>" in str(
            exc_info.value
        )
        assert "steps.predict.vlm.api_key" in str(exc_info.value)
        assert runtime["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"
        assert path_secret not in diagnostic_text
        assert "material-scrub-path" not in diagnostic_text

        persisted = config_path.read_text(encoding="utf-8")
        assert "q7Z9" not in persisted
        assert "m4P8" not in persisted

    def test_legacy_scrub_keeps_benign_path_diagnostic_useful(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        config_path = tmp_path / "ordinary" / "legacy.yaml"
        config_path.parent.mkdir()
        unsafe = self._credential_config()
        unsafe.pop("scene")
        config_path.write_text(yaml.safe_dump(unsafe), encoding="utf-8")
        sub_asset = _make_sub_asset(config_path=str(config_path))
        caplog.set_level(logging.WARNING, logger="material_agent.scene.config_gen")

        runtime = prepare_sub_asset_runtime_config(sub_asset, self._credential_config())

        assert str(config_path) in caplog.text
        assert "Removed legacy inline credentials" in caplog.text
        assert runtime["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"

    def test_runtime_overlay_error_redacts_credential_bearing_config_path(
        self, tmp_path: Path
    ) -> None:
        path_secret = "material-overlay-path-secret"
        config_dir = tmp_path / f"api_key={path_secret}"
        config_dir.mkdir()
        config_path = config_dir / "shape-drift.yaml"
        durable = _base_scene_config()
        durable.pop("scene")
        durable["steps"]["predict"] = {"vlm": {"providers": []}}
        config_path.write_text(yaml.safe_dump(durable), encoding="utf-8")

        source = _base_scene_config()
        source["steps"]["predict"] = {
            "vlm": {"providers": [{"name": "provider", "api_key": "q7Z9"}]}
        }
        credential_path = "steps.predict.vlm.providers[0].api_key"
        sub_asset = _make_sub_asset(
            config_path=str(config_path),
            config_credential_paths=[credential_path],
        )

        with pytest.raises(CredentialOverlayShapeError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, source)

        message = str(exc_info.value)
        assert exc_info.value.code == "credential_overlay_shape_mismatch"
        assert "Cannot rehydrate generated scene config <redacted>" in message
        assert "generated config structure" in message
        assert credential_path in message
        assert path_secret not in message
        assert "material-overlay-path" not in message

    def test_generated_config_read_os_error_preserves_type_and_redacts_path(
        self, tmp_path: Path
    ) -> None:
        path_secret = "material-read-path-secret"
        config_path = tmp_path / f"api_key={path_secret}"
        config_path.mkdir()
        sub_asset = _make_sub_asset(config_path=str(config_path))

        with pytest.raises(IsADirectoryError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, _base_scene_config())

        error = exc_info.value
        assert error.errno == errno.EISDIR
        assert error.filename == "<redacted>"
        assert path_secret not in str(error)
        assert "material-read-path" not in str(error)

    def test_generated_config_write_os_error_preserves_type_and_redacts_path(
        self, tmp_path: Path
    ) -> None:
        path_secret = "material-write-path-secret"
        blocked_parent = tmp_path / f"api_key={path_secret}"
        blocked_parent.write_text("not a directory", encoding="utf-8")
        output_path = blocked_parent / "generated.yaml"

        with pytest.raises(FileExistsError) as exc_info:
            generate_sub_asset_config(
                _make_sub_asset(),
                _base_scene_config(),
                output_path,
            )

        error = exc_info.value
        assert error.errno == errno.EEXIST
        assert error.filename == "<redacted>"
        assert path_secret not in str(error)
        assert "material-write-path" not in str(error)

    def test_success_logs_redact_generated_config_paths(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path_secret = "material-log-path-secret"
        output_dir = tmp_path / f"api_key={path_secret}"
        asset_path = output_dir / "asset.yaml"
        payload_path = output_dir / "payload.yaml"
        caplog.set_level(logging.INFO, logger="material_agent.scene.config_gen")

        generate_sub_asset_config(
            _make_sub_asset(),
            _base_scene_config(),
            asset_path,
        )
        generate_payload_config(
            _make_payload_group(),
            _base_scene_config(),
            payload_path,
        )

        assert caplog.text.count("<redacted>") >= 2
        assert path_secret not in caplog.text
        assert "material-log-path" not in caplog.text

    def test_generated_artifacts_omit_secrets_and_runtime_maps_are_isolated(
        self, tmp_path: Path
    ) -> None:
        payload_file = tmp_path / "inputs" / "payload.usd"
        payload_file.parent.mkdir()
        payload_file.write_text("# payload")
        manifest = SceneManifest(
            sub_assets=[
                _make_sub_asset(id="a1", name="Alpha"),
                _make_sub_asset(id="a2", name="Beta"),
            ],
            payload_groups=[
                _make_payload_group(
                    id="p1", group_name="Payload", payload_file=str(payload_file)
                )
            ],
        )
        config = self._credential_config()
        configs_dir = tmp_path / "generated" / "configs"

        generate_all_configs(manifest, config, configs_dir)
        generate_all_payload_configs(manifest, config, configs_dir)
        manifest_path = tmp_path / "generated" / "manifest.json"
        manifest.save(manifest_path)

        for artifact in (tmp_path / "generated").rglob("*"):
            if not artifact.is_file():
                continue
            artifact_text = artifact.read_text(errors="ignore")
            for sentinel in ("q7Z9", "m4P8", "q7Z", "m4P"):
                assert sentinel not in artifact_text

        asset_runtime = prepare_sub_asset_runtime_configs(manifest, config)
        payload_runtime = prepare_payload_runtime_configs(manifest, config)
        assert asset_runtime["a1"]["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"
        assert (
            asset_runtime["a2"]["steps"]["predict"]["vlm"]["providers"][0]["token"]
            == "m4P8"
        )
        assert payload_runtime["p1"]["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"

        asset_runtime["a1"]["steps"]["predict"]["vlm"]["api_key"] = "changed"
        assert asset_runtime["a2"]["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"
        assert payload_runtime["p1"]["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"

        loaded = SceneManifest.load(manifest_path)
        assert loaded.sub_assets[0].config_credential_paths == [
            "steps.predict.vlm.api_key",
            "steps.predict.vlm.providers[0].token",
        ]
        assert loaded.payload_groups[0].config_credential_paths == [
            "steps.predict.vlm.api_key",
            "steps.predict.vlm.providers[0].token",
        ]

        loaded_asset_runtime = prepare_sub_asset_runtime_configs(loaded, config)
        loaded_payload_runtime = prepare_payload_runtime_configs(loaded, config)
        assert set(loaded_asset_runtime) == {"a1", "a2"}
        assert set(loaded_payload_runtime) == {"p1"}
        assert (
            loaded_asset_runtime["a1"]["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"
        )
        assert (
            loaded_payload_runtime["p1"]["steps"]["predict"]["vlm"]["providers"][0][
                "token"
            ]
            == "m4P8"
        )

    def test_generated_artifacts_omit_url_credentials_and_rehydrate_in_memory(
        self,
        tmp_path: Path,
    ) -> None:
        sentinel = "sentinel-material-url-password"
        url = f"https://user:{sentinel}@vlm.example.test/v1"
        config = _base_scene_config()
        config["steps"]["predict"] = {"vlm": {"base_url": url}}
        manifest = SceneManifest(sub_assets=[_make_sub_asset(id="a1", name="Alpha")])
        configs_dir = tmp_path / "generated" / "configs"

        generate_all_configs(manifest, config, configs_dir)
        manifest_path = tmp_path / "generated" / "manifest.json"
        manifest.save(manifest_path)

        for artifact in (tmp_path / "generated").rglob("*"):
            if artifact.is_file():
                assert sentinel not in artifact.read_text(errors="ignore")
        assert manifest.sub_assets[0].config_credential_paths == [
            "steps.predict.vlm.base_url"
        ]
        runtime = prepare_sub_asset_runtime_config(manifest.sub_assets[0], config)
        assert runtime["steps"]["predict"]["vlm"]["base_url"] == url

    def test_signed_url_lists_use_null_tombstones_and_rehydrate_in_memory(
        self,
        tmp_path: Path,
    ) -> None:
        signed_urls = [
            "https://images.example.test/a.png?sig=sentinel-signed-a",
            "https://images.example.test/b.png?token=sentinel-signed-b",
            "https://images.example.test/public.png",
        ]
        config = _base_scene_config()
        config["steps"]["predict"] = {"reference_image_uris": signed_urls}
        payload_file = tmp_path / "inputs" / "payload.usd"
        payload_file.parent.mkdir()
        payload_file.write_text("# payload")
        manifest = SceneManifest(
            sub_assets=[_make_sub_asset(id="a1", name="Alpha")],
            payload_groups=[
                _make_payload_group(
                    id="p1", group_name="Payload", payload_file=str(payload_file)
                )
            ],
        )
        configs_dir = tmp_path / "generated" / "configs"

        generate_all_configs(manifest, config, configs_dir)
        generate_all_payload_configs(manifest, config, configs_dir)

        for config_path in (
            Path(manifest.sub_assets[0].config_path),
            Path(manifest.payload_groups[0].config_path),
        ):
            durable = _read_yaml(config_path)
            assert durable["steps"]["predict"]["reference_image_uris"] == [
                None,
                None,
                signed_urls[2],
            ]
            persisted = config_path.read_text()
            assert "sentinel-signed-a" not in persisted
            assert "sentinel-signed-b" not in persisted

        assert manifest.sub_assets[0].config_credential_paths == [
            "steps.predict.reference_image_uris[0]",
            "steps.predict.reference_image_uris[1]",
        ]
        assert manifest.payload_groups[0].config_credential_paths == [
            "steps.predict.reference_image_uris[0]",
            "steps.predict.reference_image_uris[1]",
        ]
        asset_runtime = prepare_sub_asset_runtime_config(manifest.sub_assets[0], config)
        payload_runtime = prepare_payload_runtime_configs(manifest, config)["p1"]
        assert asset_runtime["steps"]["predict"]["reference_image_uris"] == signed_urls
        assert (
            payload_runtime["steps"]["predict"]["reference_image_uris"] == signed_urls
        )

    def test_resume_rejects_list_credential_with_drifted_sibling_endpoint(
        self,
        tmp_path: Path,
    ) -> None:
        original = _base_scene_config()
        original["steps"]["predict"] = {
            "vlm": {
                "backend": "nim",
                "base_url": "https://old-vlm.example.test/v1",
                "reference_image_uris": [
                    "https://images.example.test/old.png?sig=sentinel-old-list"
                ],
            }
        }
        payload_file = tmp_path / "inputs" / "payload.usd"
        payload_file.parent.mkdir()
        payload_file.write_text("# payload")
        manifest = SceneManifest(
            sub_assets=[_make_sub_asset(id="a1", name="Alpha")],
            payload_groups=[
                _make_payload_group(
                    id="p1", group_name="Payload", payload_file=str(payload_file)
                )
            ],
        )
        configs_dir = tmp_path / "generated" / "configs"
        generate_all_configs(manifest, original, configs_dir)
        generate_all_payload_configs(manifest, original, configs_dir)

        current = _base_scene_config()
        current["steps"]["predict"] = {
            "vlm": {
                "backend": "openai",
                "base_url": "https://new-vlm.example.test/v1",
                "reference_image_uris": [
                    "https://images.example.test/new.png?sig=sentinel-current-list"
                ],
            }
        }

        with pytest.raises(ValueError) as asset_error:
            prepare_sub_asset_runtime_config(manifest.sub_assets[0], current)
        with pytest.raises(ValueError) as payload_error:
            prepare_payload_runtime_configs(manifest, current)

        for error in (asset_error.value, payload_error.value):
            message = str(error)
            assert "generated config structure" in message
            assert "steps.predict.vlm.reference_image_uris[0]" in message
            assert "sentinel-old-list" not in message
            assert "sentinel-current-list" not in message
            assert "old-vlm.example.test" not in message
            assert "new-vlm.example.test" not in message

        for config_path in (
            Path(manifest.sub_assets[0].config_path),
            Path(manifest.payload_groups[0].config_path),
        ):
            persisted = config_path.read_text()
            assert "https://old-vlm.example.test/v1" in persisted
            assert "sentinel-old-list" not in persisted
            assert "sentinel-current-list" not in persisted

    def test_resume_rejects_credential_context_endpoint_drift(
        self,
        tmp_path: Path,
    ) -> None:
        original = _base_scene_config()
        original["steps"]["predict"] = {
            "vlm": {
                "backend": "nim",
                "base_url": "https://old-vlm.example.test/v1",
                "auth": {"api_key": "sentinel-old-key"},
            }
        }
        payload_file = tmp_path / "inputs" / "payload.usd"
        payload_file.parent.mkdir()
        payload_file.write_text("# payload")
        manifest = SceneManifest(
            sub_assets=[_make_sub_asset(id="a1", name="Alpha")],
            payload_groups=[
                _make_payload_group(
                    id="p1", group_name="Payload", payload_file=str(payload_file)
                )
            ],
        )
        configs_dir = tmp_path / "generated" / "configs"
        generate_all_configs(manifest, original, configs_dir)
        generate_all_payload_configs(manifest, original, configs_dir)

        current = _base_scene_config()
        current["steps"]["predict"] = {
            "vlm": {
                "backend": "openai",
                "base_url": "https://new-vlm.example.test/v1",
                "auth": {"api_key": "sentinel-current-key"},
            }
        }

        with pytest.raises(ValueError) as asset_error:
            prepare_sub_asset_runtime_config(manifest.sub_assets[0], current)
        with pytest.raises(ValueError) as payload_error:
            prepare_payload_runtime_configs(manifest, current)

        for error in (asset_error.value, payload_error.value):
            message = str(error)
            assert "generated config structure" in message
            assert "steps.predict.vlm.auth.api_key" in message
            assert "sentinel-old-key" not in message
            assert "sentinel-current-key" not in message
            assert "old-vlm.example.test" not in message
            assert "new-vlm.example.test" not in message

        for config_path in (
            Path(manifest.sub_assets[0].config_path),
            Path(manifest.payload_groups[0].config_path),
        ):
            persisted = config_path.read_text()
            assert "https://old-vlm.example.test/v1" in persisted
            assert "sentinel-old-key" not in persisted
            assert "sentinel-current-key" not in persisted

    def test_credential_only_context_rejects_projection_drift(self) -> None:
        target = {"auth": {"endpoint": "evil"}}

        runtime = _overlay_inline_credentials(
            target,
            {"auth": {"api_key": "q7Z9"}},
            {"auth": {"api_key": "<redacted>"}},
        )

        assert runtime == target

    def test_resume_rejects_tampered_credential_only_context(
        self,
        tmp_path: Path,
    ) -> None:
        sentinel = "sentinel-credential-only-key"
        scene_config = _base_scene_config()
        scene_config["steps"]["predict"] = {"vlm": {"auth": {"api_key": sentinel}}}
        manifest = SceneManifest(sub_assets=[_make_sub_asset(id="a1", name="Alpha")])

        generate_all_configs(manifest, scene_config, tmp_path / "configs")
        sub_asset = manifest.sub_assets[0]
        config_path = Path(sub_asset.config_path)
        durable = _read_yaml(config_path)
        durable["steps"]["predict"]["vlm"]["auth"]["endpoint"] = "evil"
        config_path.write_text(yaml.safe_dump(durable), encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, scene_config)

        message = str(exc_info.value)
        assert "generated config structure" in message
        assert "steps.predict.vlm.auth.api_key" in message
        assert sentinel not in message
        assert sentinel not in config_path.read_text(encoding="utf-8")

    def test_credential_transport_edge_shapes_and_bounded_errors(self) -> None:
        missing = _value_at_path({"nested": {}}, ("nested", "missing"))
        assert missing is _value_at_path({"other": {}}, ("missing",))
        assert missing is _value_at_path({"nested": "scalar"}, ("nested", "key"))
        assert _has_nonsecret_context([None, {"nested": "value"}])
        assert not _has_nonsecret_context([None, {}])
        assert _overlay_inline_credentials(
            {"items": {}},
            {"items": ["https://example.test/image.png?sig=q7Z9"]},
            {"items": ["<redacted>"]},
        ) == {"items": {}}
        assert _overlay_inline_credentials(
            {"auth": {}},
            {"auth": {"api_key": "q7Z9"}},
            {"auth": {"api_key": "<redacted>"}},
        ) == {"auth": {"api_key": "q7Z9"}}
        assert (
            _overlay_inline_credentials(
                "durable", {"plain": "source"}, {"plain": "source"}
            )
            == "durable"
        )
        assert _drop_redacted_credentials(
            ({"api_key": "q7Z9"},),
            ({"api_key": "<redacted>"},),
        ) == ({},)
        assert _drop_redacted_credentials(
            ["https://example.test/image.png?sig=q7Z9"],
            ["<redacted>"],
        ) == [None]
        assert _overlay_inline_credentials("<redacted>", "q7Z9", "<redacted>") == "q7Z9"
        assert _overlay_inline_credentials(
            [None, "https://example.test/durable.png"],
            [
                "https://example.test/image.png?sig=q7Z9",
                "https://example.test/changed.png",
            ],
            ["<redacted>", "https://example.test/changed.png"],
        ) == [None, "https://example.test/durable.png"]
        assert (
            _overlay_inline_credentials(
                [],
                [{"name": "fallback"}],
                [{"name": "fallback"}],
            )
            == []
        )
        assert (
            _overlay_inline_credentials(
                None,
                [{"name": "fallback"}],
                [{"name": "fallback"}],
            )
            is None
        )
        assert (
            _overlay_inline_credentials(
                [],
                [{"name": "credential-provider", "api_key": "q7Z9"}],
                [{"name": "credential-provider", "api_key": "<redacted>"}],
            )
            == []
        )
        assert _overlay_inline_credentials(
            [{"name": "credential-provider", "endpoint": "durable"}],
            [{"name": "credential-provider", "api_key": "q7Z9"}],
            [{"name": "credential-provider", "api_key": "<redacted>"}],
        ) == [{"name": "credential-provider", "endpoint": "durable"}]
        assert _overlay_inline_credentials(
            [{"name": "credential-provider", "endpoint": "durable"}],
            [
                {
                    "name": "credential-provider",
                    "endpoint": "durable",
                    "api_key": "q7Z9",
                }
            ],
            [
                {
                    "name": "credential-provider",
                    "endpoint": "durable",
                    "api_key": "<redacted>",
                }
            ],
        ) == [
            {
                "name": "credential-provider",
                "endpoint": "durable",
                "api_key": "q7Z9",
            }
        ]
        assert _overlay_inline_credentials(
            [{"name": "provider-b"}, {"name": "provider-a"}],
            [
                {"name": "provider-a", "api_key": "q7Z9"},
                {"name": "provider-b"},
            ],
            [
                {"name": "provider-a", "api_key": "<redacted>"},
                {"name": "provider-b"},
            ],
        ) == [{"name": "provider-b"}, {"name": "provider-a"}]
        assert _overlay_inline_credentials(
            [{}],
            [{"api_key": "q7Z9"}],
            [{"api_key": "<redacted>"}],
        ) == [{"api_key": "q7Z9"}]
        assert _overlay_inline_credentials(
            ["mismatched-shape"],
            [{"name": "credential-provider", "api_key": "q7Z9"}],
            [{"name": "credential-provider", "api_key": "<redacted>"}],
        ) == ["mismatched-shape"]
        assert _overlay_inline_credentials(
            [{"name": "duplicate"}, {"name": "duplicate"}],
            [
                {"name": "duplicate", "api_key": "q7Z9"},
                {"name": "duplicate"},
            ],
            [
                {"name": "duplicate", "api_key": "<redacted>"},
                {"name": "duplicate"},
            ],
        ) == [
            {"name": "duplicate", "api_key": "q7Z9"},
            {"name": "duplicate"},
        ]
        assert _overlay_inline_credentials(
            [{"name": "shared-prefix"}],
            [
                {"name": "shared-prefix"},
                {"name": "credential-provider", "api_key": "q7Z9"},
            ],
            [
                {"name": "shared-prefix"},
                {"name": "credential-provider", "api_key": "<redacted>"},
            ],
        ) == [{"name": "shared-prefix"}]
        assert _overlay_inline_credentials(
            [{"name": "durable-prefix"}],
            [
                {"name": "different-source-prefix"},
                {"name": "credential-provider", "api_key": "q7Z9"},
            ],
            [
                {"name": "different-source-prefix"},
                {"name": "credential-provider", "api_key": "<redacted>"},
            ],
        ) == [{"name": "durable-prefix"}]
        assert (
            _overlay_inline_credentials(
                [],
                [
                    {"name": "source-only-noncredential"},
                    {"name": "credential-provider", "api_key": "q7Z9"},
                ],
                [
                    {"name": "source-only-noncredential"},
                    {"name": "credential-provider", "api_key": "<redacted>"},
                ],
            )
            == []
        )
        assert _overlay_inline_credentials(
            {"provider": {"endpoint": "old", "auth": {}}},
            {
                "provider": {
                    "endpoint": "new",
                    "auth": {"api_key": "q7Z9"},
                }
            },
            {
                "provider": {
                    "endpoint": "new",
                    "auth": {"api_key": "<redacted>"},
                }
            },
        ) == {"provider": {"endpoint": "old", "auth": {}}}
        assert _overlay_inline_credentials(
            {"provider": {"endpoint": "same", "auth": {}}},
            {
                "provider": {
                    "endpoint": "same",
                    "auth": {"api_key": "q7Z9"},
                }
            },
            {
                "provider": {
                    "endpoint": "same",
                    "auth": {"api_key": "<redacted>"},
                }
            },
        ) == {"provider": {"endpoint": "same", "auth": {"api_key": "q7Z9"}}}
        assert _overlay_inline_credentials(
            [[None], [None]],
            [
                ["https://example.test/a.png?sig=q7Z9"],
                ["https://example.test/b.png?sig=m4P8"],
            ],
            [["<redacted>"], ["<redacted>"]],
        ) == [
            ["https://example.test/a.png?sig=q7Z9"],
            ["https://example.test/b.png?sig=m4P8"],
        ]
        assert _overlay_inline_credentials("target", "source", "source") == "target"

        error = _missing_credential_error(
            config_path=Path("generated.yaml"),
            missing_paths={f"secret.path.{index}" for index in range(9)},
        )
        assert isinstance(error, MissingCredentialSourceError)
        assert error.code == "credential_source_missing"
        assert "and 1 more" in str(error)

    def test_resume_scrubs_legacy_yaml_and_rehydrates_from_current_source(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "legacy.yaml"
        unsafe = self._credential_config()
        unsafe.pop("scene")
        config_path.write_text(yaml.safe_dump(unsafe))
        sub_asset = _make_sub_asset(config_path=str(config_path))

        runtime = prepare_sub_asset_runtime_config(sub_asset, self._credential_config())

        persisted = config_path.read_text()
        assert "q7Z9" not in persisted
        assert "m4P8" not in persisted
        assert runtime["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"
        assert runtime["steps"]["predict"]["vlm"]["providers"][0]["token"] == "m4P8"

    def test_resume_fails_closed_by_path_when_source_secret_is_missing(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "legacy.yaml"
        unsafe = self._credential_config()
        unsafe.pop("scene")
        config_path.write_text(yaml.safe_dump(unsafe))
        sub_asset = _make_sub_asset(config_path=str(config_path))

        with pytest.raises(ValueError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, _base_scene_config())

        message = str(exc_info.value)
        assert "steps.predict.vlm.api_key" in message
        assert "q7Z9" not in message
        assert "m4P8" not in message
        persisted = config_path.read_text()
        assert "q7Z9" not in persisted
        assert "m4P8" not in persisted

    def test_sub_asset_resume_fails_closed_when_list_shape_drops_credential(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "shape-drift.yaml"
        durable = _base_scene_config()
        durable.pop("scene")
        durable["steps"]["predict"] = {"vlm": {"providers": []}}
        config_path.write_text(yaml.safe_dump(durable))

        source = _base_scene_config()
        source["steps"]["predict"] = {
            "vlm": {
                "providers": [
                    {"name": "source-only-noncredential"},
                    {"name": "credential-provider", "api_key": "q7Z9"},
                ]
            }
        }
        credential_path = "steps.predict.vlm.providers[1].api_key"
        sub_asset = _make_sub_asset(
            config_path=str(config_path),
            config_credential_paths=[credential_path],
        )

        with pytest.raises(ValueError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, source)

        message = str(exc_info.value)
        assert "generated config structure" in message
        assert credential_path in message
        assert "q7Z9" not in message
        assert sub_asset.config_credential_paths == [credential_path]

    def test_sub_asset_resume_fails_closed_when_provider_list_is_reordered(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "reordered-providers.yaml"
        durable = _base_scene_config()
        durable.pop("scene")
        durable["steps"]["predict"] = {
            "vlm": {
                "providers": [
                    {"name": "provider-b"},
                    {"name": "provider-a"},
                ]
            }
        }
        config_path.write_text(yaml.safe_dump(durable))

        source = _base_scene_config()
        source["steps"]["predict"] = {
            "vlm": {
                "providers": [
                    {"name": "provider-a", "api_key": "q7Z9"},
                    {"name": "provider-b"},
                ]
            }
        }
        credential_path = "steps.predict.vlm.providers[0].api_key"
        sub_asset = _make_sub_asset(
            config_path=str(config_path),
            config_credential_paths=[credential_path],
        )

        with pytest.raises(ValueError) as exc_info:
            prepare_sub_asset_runtime_config(sub_asset, source)

        message = str(exc_info.value)
        assert "generated config structure" in message
        assert credential_path in message
        assert "q7Z9" not in message
        assert "provider-a" not in message
        assert "provider-b" not in message
        assert sub_asset.config_credential_paths == [credential_path]

    def test_payload_resume_fails_closed_when_list_shape_drops_credential(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "payload-shape-drift.yaml"
        durable = _base_scene_config()
        durable.pop("scene")
        durable["steps"]["predict"] = {"vlm": {"providers": []}}
        config_path.write_text(yaml.safe_dump(durable))

        source = _base_scene_config()
        source["steps"]["predict"] = {
            "vlm": {
                "providers": [
                    {"name": "source-only-noncredential"},
                    {"name": "credential-provider", "api_key": "m4P8"},
                ]
            }
        }
        credential_path = "steps.predict.vlm.providers[1].api_key"
        payload = _make_payload_group(
            config_path=str(config_path),
            config_credential_paths=[credential_path],
        )

        with pytest.raises(ValueError) as exc_info:
            prepare_payload_runtime_configs(
                SceneManifest(payload_groups=[payload]),
                source,
            )

        message = str(exc_info.value)
        assert "generated config structure" in message
        assert credential_path in message
        assert "m4P8" not in message
        assert payload.config_credential_paths == [credential_path]

    def test_payload_resume_fails_closed_when_provider_item_type_changes(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "payload-provider-type-drift.yaml"
        durable = _base_scene_config()
        durable.pop("scene")
        durable["steps"]["predict"] = {
            "vlm": {"providers": ["mismatched-provider-shape"]}
        }
        config_path.write_text(yaml.safe_dump(durable))

        source = _base_scene_config()
        source["steps"]["predict"] = {
            "vlm": {"providers": [{"name": "credential-provider", "api_key": "m4P8"}]}
        }
        credential_path = "steps.predict.vlm.providers[0].api_key"
        payload = _make_payload_group(
            config_path=str(config_path),
            config_credential_paths=[credential_path],
        )

        with pytest.raises(ValueError) as exc_info:
            prepare_payload_runtime_configs(
                SceneManifest(payload_groups=[payload]),
                source,
            )

        message = str(exc_info.value)
        assert "generated config structure" in message
        assert credential_path in message
        assert "m4P8" not in message
        assert "credential-provider" not in message
        assert "mismatched-provider-shape" not in message
        assert payload.config_credential_paths == [credential_path]

    def test_payload_resume_fails_closed_when_source_secret_is_missing(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "legacy-payload.yaml"
        unsafe = self._credential_config()
        unsafe.pop("scene")
        config_path.write_text(yaml.safe_dump(unsafe))
        payload = _make_payload_group(config_path=str(config_path))

        with pytest.raises(ValueError) as exc_info:
            prepare_payload_runtime_configs(
                SceneManifest(payload_groups=[payload]),
                _base_scene_config(),
            )

        assert "steps.predict.vlm.api_key" in str(exc_info.value)
        assert "q7Z9" not in str(exc_info.value)
        persisted = config_path.read_text()
        assert "q7Z9" not in persisted
        assert "m4P8" not in persisted
