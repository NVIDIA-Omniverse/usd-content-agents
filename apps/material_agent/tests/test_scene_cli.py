# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for material_agent.scene.cli helper functions."""

from __future__ import annotations

from pathlib import Path

import click.exceptions
import pytest
import typer
import yaml


# ---------------------------------------------------------------------------
# _load_scene_config
# ---------------------------------------------------------------------------
class TestLoadSceneConfig:
    """Tests for _load_scene_config."""

    def test_valid_yaml(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _load_scene_config

        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"project": {"name": "test"}}))
        result = _load_scene_config(cfg)
        assert result == {"project": {"name": "test"}}

    def test_missing_file_raises_exit(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _load_scene_config

        with pytest.raises(click.exceptions.Exit):
            _load_scene_config(tmp_path / "nonexistent.yaml")

    def test_empty_yaml_raises_bad_parameter(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _load_scene_config

        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")
        with pytest.raises(typer.BadParameter, match="YAML mapping"):
            _load_scene_config(cfg)

    def test_complex_yaml(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _load_scene_config

        data = {
            "project": {"name": "scene1", "session_id": "sess01"},
            "input": {"usd_path": "model.usd"},
            "scene": {"analyze": {"skip_geometry": True}},
        }
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump(data))
        result = _load_scene_config(cfg)
        assert result["project"]["session_id"] == "sess01"
        assert result["scene"]["analyze"]["skip_geometry"] is True


# ---------------------------------------------------------------------------
# _resolve_usd_path
# ---------------------------------------------------------------------------
class TestResolveUsdPath:
    """Tests for _resolve_usd_path."""

    def test_relative_path_resolved_to_config_dir(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_usd_path

        usd = tmp_path / "model.usd"
        usd.write_text("dummy")
        config_path = tmp_path / "config.yaml"
        scene_config = {"input": {"usd_path": "model.usd"}}
        result = _resolve_usd_path(scene_config, config_path)
        assert result == usd.resolve()

    def test_absolute_path_used_directly(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_usd_path

        usd = tmp_path / "abs_model.usd"
        usd.write_text("dummy")
        config_path = tmp_path / "config.yaml"
        scene_config = {"input": {"usd_path": str(usd)}}
        result = _resolve_usd_path(scene_config, config_path)
        assert result == usd

    def test_missing_usd_path_key_raises_exit(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_usd_path

        config_path = tmp_path / "config.yaml"
        with pytest.raises(click.exceptions.Exit):
            _resolve_usd_path({}, config_path)

    def test_empty_usd_path_raises_exit(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_usd_path

        config_path = tmp_path / "config.yaml"
        with pytest.raises(click.exceptions.Exit):
            _resolve_usd_path({"input": {"usd_path": ""}}, config_path)

    def test_nonexistent_usd_file_raises_exit(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_usd_path

        config_path = tmp_path / "config.yaml"
        scene_config = {"input": {"usd_path": "missing.usd"}}
        with pytest.raises(click.exceptions.Exit):
            _resolve_usd_path(scene_config, config_path)

    def test_relative_path_in_subdir(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_usd_path

        subdir = tmp_path / "assets"
        subdir.mkdir()
        usd = subdir / "scene.usd"
        usd.write_text("dummy")
        config_path = tmp_path / "config.yaml"
        scene_config = {"input": {"usd_path": "assets/scene.usd"}}
        result = _resolve_usd_path(scene_config, config_path)
        assert result == usd.resolve()


# ---------------------------------------------------------------------------
# _resolve_material_library_yaml
# ---------------------------------------------------------------------------
class TestResolveMaterialLibraryYaml:
    """Tests for _resolve_material_library_yaml."""

    def test_no_materials_section_returns_none(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_material_library_yaml

        config_path = tmp_path / "config.yaml"
        assert _resolve_material_library_yaml({}, config_path) is None

    def test_no_path_key_returns_none(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_material_library_yaml

        config_path = tmp_path / "config.yaml"
        assert _resolve_material_library_yaml({"materials": {}}, config_path) is None

    def test_relative_path_resolved(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_material_library_yaml

        config_path = tmp_path / "config.yaml"
        scene_config = {"materials": {"path": "mats/library.yaml"}}
        result = _resolve_material_library_yaml(scene_config, config_path)
        assert result == (tmp_path / "mats" / "library.yaml").resolve()

    def test_absolute_path_used_directly(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_material_library_yaml

        config_path = tmp_path / "config.yaml"
        abs_path = "/some/absolute/library.yaml"
        scene_config = {"materials": {"path": abs_path}}
        result = _resolve_material_library_yaml(scene_config, config_path)
        assert result == Path(abs_path)

    def test_empty_path_returns_none(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _resolve_material_library_yaml

        config_path = tmp_path / "config.yaml"
        scene_config = {"materials": {"path": ""}}
        result = _resolve_material_library_yaml(scene_config, config_path)
        assert result is None


# ---------------------------------------------------------------------------
# _get_working_dir
# ---------------------------------------------------------------------------
class TestGetWorkingDir:
    """Tests for _get_working_dir."""

    def test_uses_session_id(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _get_working_dir

        config_path = tmp_path / "config.yaml"
        scene_config = {"project": {"session_id": "my_session"}}
        result = _get_working_dir(scene_config, config_path)
        assert result == tmp_path / ".my_session_scene"

    def test_falls_back_to_name(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _get_working_dir

        config_path = tmp_path / "config.yaml"
        scene_config = {"project": {"name": "proj_name"}}
        result = _get_working_dir(scene_config, config_path)
        assert result == tmp_path / ".proj_name_scene"

    def test_falls_back_to_default(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _get_working_dir

        config_path = tmp_path / "config.yaml"
        result = _get_working_dir({}, config_path)
        assert result == tmp_path / ".scene_scene"

    def test_session_id_takes_precedence_over_name(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _get_working_dir

        config_path = tmp_path / "config.yaml"
        scene_config = {"project": {"session_id": "sid", "name": "pname"}}
        result = _get_working_dir(scene_config, config_path)
        assert result == tmp_path / ".sid_scene"


# ---------------------------------------------------------------------------
# _get_manifest_path
# ---------------------------------------------------------------------------
class TestGetManifestPath:
    """Tests for _get_manifest_path."""

    def test_returns_manifest_json(self, tmp_path: Path) -> None:
        from material_agent.scene.cli import _get_manifest_path

        result = _get_manifest_path(tmp_path / "work")
        assert result == tmp_path / "work" / "manifest.json"


# ---------------------------------------------------------------------------
# _parse_assets_filter
# ---------------------------------------------------------------------------
class TestParseAssetsFilter:
    """Tests for _parse_assets_filter."""

    def test_none_returns_none(self) -> None:
        from material_agent.scene.cli import _parse_assets_filter

        assert _parse_assets_filter(None) is None

    def test_empty_string_returns_none(self) -> None:
        from material_agent.scene.cli import _parse_assets_filter

        assert _parse_assets_filter("") is None

    def test_single_asset(self) -> None:
        from material_agent.scene.cli import _parse_assets_filter

        assert _parse_assets_filter("chair") == ["chair"]

    def test_multiple_assets(self) -> None:
        from material_agent.scene.cli import _parse_assets_filter

        assert _parse_assets_filter("chair,table,lamp") == ["chair", "table", "lamp"]

    def test_strips_whitespace(self) -> None:
        from material_agent.scene.cli import _parse_assets_filter

        assert _parse_assets_filter(" chair , table ") == ["chair", "table"]

    def test_ignores_empty_entries(self) -> None:
        from material_agent.scene.cli import _parse_assets_filter

        assert _parse_assets_filter("chair,,table,") == ["chair", "table"]


# ---------------------------------------------------------------------------
# _steps_before
# ---------------------------------------------------------------------------
class TestStepsBefore:
    """Tests for _steps_before."""

    def test_first_step_returns_empty(self) -> None:
        from material_agent.scene.cli import _steps_before

        assert _steps_before("optimize_usd") == []

    def test_second_step_returns_first(self) -> None:
        from material_agent.scene.cli import _steps_before

        assert _steps_before("render_preview") == ["optimize_usd"]

    def test_predict_returns_earlier_steps(self) -> None:
        from material_agent.scene.cli import _steps_before

        result = _steps_before("predict")
        assert "optimize_usd" in result
        assert "render_preview" in result
        assert "build_dataset_prepare_dataset" in result
        assert "predict" not in result

    def test_last_step_returns_all_but_last(self) -> None:
        from material_agent.scene.cli import _ASSET_PIPELINE_STEPS, _steps_before

        result = _steps_before("render")
        assert len(result) == len(_ASSET_PIPELINE_STEPS) - 1
        assert "render" not in result

    def test_unknown_step_raises_bad_parameter(self) -> None:
        from material_agent.scene.cli import _steps_before

        with pytest.raises(typer.BadParameter, match="Unknown step 'bogus'"):
            _steps_before("bogus")


# ---------------------------------------------------------------------------
# _setup_logging
# ---------------------------------------------------------------------------
class TestSetupLogging:
    """Tests for _setup_logging."""

    def test_verbose_sets_debug(self) -> None:
        import logging

        from material_agent.scene.cli import _setup_logging

        root = logging.getLogger()
        # Remove existing handlers so basicConfig takes effect
        for h in root.handlers[:]:
            root.removeHandler(h)
        root.setLevel(logging.WARNING)

        _setup_logging(verbose=True)
        assert root.level == logging.DEBUG

    def test_non_verbose_sets_info(self) -> None:
        import logging

        from material_agent.scene.cli import _setup_logging

        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        root.setLevel(logging.WARNING)

        _setup_logging(verbose=False)
        assert root.level == logging.INFO


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------
class TestRunCommand:
    """Tests for the public scene run CLI wrapper."""

    def test_run_command_uses_public_scene_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from material_agent.api import ScenePipelineOutput
        from material_agent.scene.cli import run_cmd

        config = tmp_path / "scene.yaml"
        config.write_text("project:\n  name: test\n")
        captured = {}

        def fake_run_scene_pipeline(params):
            captured["params"] = params
            return ScenePipelineOutput(
                success=True,
                output_usd_path=str(tmp_path / "out.usd"),
                completed_assets=2,
                failed_assets=0,
            )

        monkeypatch.setattr(
            "material_agent.api.run_scene_pipeline",
            fake_run_scene_pipeline,
        )

        run_cmd(
            config=config,
            assets="AssetA,/Root/AssetB",
            skip="render",
            only="predict",
            from_step="predict",
            workers=2,
            skip_existing=True,
            no_render=True,
            resume=True,
            predict_max_workers=3,
        )

        params = captured["params"]
        assert params.config == config
        assert params.assets == ["AssetA", "/Root/AssetB"]
        assert params.skip_steps == ["render"]
        assert params.only_steps == ["predict"]
        assert params.from_step == "predict"
        assert params.max_workers == 2
        assert params.skip_existing is True
        assert params.no_render is True
        assert params.fail_on_validation_error is True
        assert params.resume is True
        assert params.predict_max_workers == 3

    def test_run_command_exits_on_validation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from material_agent.api import ScenePipelineOutput
        from material_agent.scene.cli import run_cmd

        config = tmp_path / "scene.yaml"
        config.write_text("project:\n  name: test\n")
        captured = {}

        def fake_run_scene_pipeline(params):
            captured["params"] = params
            return ScenePipelineOutput(
                success=False,
                error="Output validation failed",
                validation_passed=False,
            )

        monkeypatch.setattr(
            "material_agent.api.run_scene_pipeline",
            fake_run_scene_pipeline,
        )

        with pytest.raises(typer.Exit) as exc_info:
            run_cmd(config=config)

        params = captured["params"]
        output = capsys.readouterr().out
        assert exc_info.value.exit_code == 1
        assert params.fail_on_validation_error is True
        assert "Scene pipeline failed" in output
        assert "Scene pipeline complete" not in output

    def test_run_command_can_warn_on_validation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from material_agent.api import ScenePipelineOutput
        from material_agent.scene.cli import run_cmd

        config = tmp_path / "scene.yaml"
        config.write_text("project:\n  name: test\n")
        captured = {}

        def fake_run_scene_pipeline(params):
            captured["params"] = params
            return ScenePipelineOutput(
                success=True,
                validation_passed=False,
                output_usd_path=str(tmp_path / "out.usd"),
                completed_assets=1,
                failed_assets=0,
            )

        monkeypatch.setattr(
            "material_agent.api.run_scene_pipeline",
            fake_run_scene_pipeline,
        )

        run_cmd(config=config, fail_on_validation=False)

        params = captured["params"]
        output = capsys.readouterr().out
        assert params.fail_on_validation_error is False
        assert "Scene pipeline complete" in output
