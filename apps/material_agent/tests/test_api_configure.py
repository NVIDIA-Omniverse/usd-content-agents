# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Configure API."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.configure import (
    ConfigureInput,
    ConfigureOutput,
    run_configure,
)


class TestConfigureInput:
    """Tests for ConfigureInput validation."""

    def test_configure_input_valid(self, tmp_path):
        """Test creating valid ConfigureInput."""
        output_config = tmp_path / "new_config.yaml"

        params = ConfigureInput(
            output_config_path=output_config,
            force=False,
            verbose=True,
        )

        assert params.output_config_path == output_config
        assert params.force is False
        assert params.verbose is True

    def test_configure_input_file_exists_without_force(self, tmp_path):
        """Test ConfigureInput raises error when file exists and force is False."""
        output_config = tmp_path / "existing.yaml"
        output_config.write_text("# existing")

        with pytest.raises(FileExistsError, match="Configuration file already exists"):
            ConfigureInput(output_config_path=output_config, force=False)

    def test_configure_input_existing_credential_path_is_not_disclosed(self, tmp_path):
        sentinel = "api_key=configure-existing-path-secret-713"
        output_config = tmp_path / sentinel
        output_config.write_text("# existing", encoding="utf-8")

        with pytest.raises(FileExistsError) as exc_info:
            ConfigureInput(output_config_path=output_config, force=False)

        assert str(exc_info.value) == (
            "Configuration file already exists; use force=True"
        )
        assert sentinel not in repr(exc_info.value)

    def test_configure_input_file_exists_with_force(self, tmp_path):
        """Test ConfigureInput allows overwrite when force is True."""
        output_config = tmp_path / "existing.yaml"
        output_config.write_text("# existing")

        params = ConfigureInput(output_config_path=output_config, force=True)

        assert params.output_config_path == output_config
        assert params.force is True


class TestRunConfigure:
    """Tests for run_configure function."""

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_success(self, mock_create_workflow, tmp_path):
        """Test successful configuration creation."""
        # Setup
        output_config = tmp_path / "new_config.yaml"
        materials_manifest = tmp_path / "materials.yaml"
        materials_manifest.write_text("materials: []")
        reference_image = tmp_path / "ref.png"
        reference_image.write_text("png")
        sentinel = "api_key=configure-public-result-secret-713"

        # Mock workflow
        mock_workflow = Mock()
        runtime_result = {
            "config_created": True,
            "config_path": str(output_config),
            "pipeline_name": "test_pipeline",
            "input_usd_path": "/path/to/input.usd",
            "materials_library_path": "/path/to/materials",
            "output_usd_path": f"/path/{sentinel}/output.usd",
            "dataset_dir": "/path/to/dataset",
            "predictions_dir": "/path/to/predictions",
            "config_dict": {"vlm": {"api_key": sentinel}},
            "runtime_collaborator": object(),
        }
        mock_workflow.arun = AsyncMock(return_value=runtime_result)
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = ConfigureInput(
            output_config_path=output_config,
            materials_manifest=materials_manifest,
            reference_images=[str(reference_image)],
        )
        result = run_configure(params)

        # Verify
        assert result.success is True
        assert result.config_path == output_config
        assert result.pipeline_name == "test_pipeline"
        assert result.input_usd_path == "/path/to/input.usd"
        assert result.materials_library_path == "/path/to/materials"
        assert result.output_usd_path is None
        assert result.raw_result is not runtime_result
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result
        assert "runtime_collaborator" not in result.raw_result
        assert sentinel not in repr(result)
        assert runtime_result["config_dict"]["vlm"]["api_key"] == sentinel
        initial_context = mock_workflow.arun.call_args.kwargs["initial_context"]
        assert initial_context["materials_manifest"] == str(materials_manifest)
        assert initial_context["reference_images"] == [str(reference_image)]

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_not_created(self, mock_create_workflow, tmp_path):
        """Test configure when workflow doesn't create config."""
        # Setup
        output_config = tmp_path / "new_config.yaml"

        # Mock workflow that doesn't complete
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value={"config_created": False})
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = ConfigureInput(output_config_path=output_config)
        result = run_configure(params)

        # Verify
        assert result.success is False
        assert "did not complete" in result.error.lower()

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_file_exists_error(self, mock_create_workflow, tmp_path):
        """Test configure when file exists and force is False."""
        # Setup - create existing file
        output_config = tmp_path / "existing.yaml"
        output_config.write_text("# existing")

        # This should raise during input validation, not during run
        with pytest.raises(FileExistsError, match="Configuration file already exists"):
            ConfigureInput(output_config_path=output_config, force=False)

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_exception(self, mock_create_workflow, tmp_path, caplog):
        """Test configure when exception occurs."""
        # Setup
        output_config = tmp_path / "new_config.yaml"

        # Mock workflow that raises exception
        mock_workflow = Mock()
        sentinel = "api_key=configure-exception-secret-713"
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError(sentinel))
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = ConfigureInput(output_config_path=output_config)
        result = run_configure(params)

        # Verify
        assert result.success is False
        assert result.error == "Configuration creation failed"
        assert sentinel not in repr(result)
        assert sentinel not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    @patch("material_agent.workflows.create_configure_workflow")
    def test_run_configure_reraises_file_exists_error(
        self, mock_create_workflow, tmp_path
    ):
        """Test FileExistsError from workflow is not swallowed."""
        sentinel = "api_key=configure-file-exists-frame-secret-713"
        output_config = tmp_path / sentinel / "new_config.yaml"
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=FileExistsError("exists"))
        mock_create_workflow.return_value = mock_workflow

        with pytest.raises(FileExistsError) as exc_info:
            run_configure(ConfigureInput(output_config_path=output_config))

        assert str(exc_info.value) == (
            "Configuration file already exists; use force=True"
        )
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        cursor = exc_info.value.__traceback__
        configure_frames = []
        while cursor is not None:
            if cursor.tb_frame.f_code.co_name in {"__post_init__", "arun_configure"}:
                configure_frames.append(dict(cursor.tb_frame.f_locals))
            cursor = cursor.tb_next
        assert configure_frames
        assert sentinel not in repr(configure_frames)


class TestConfigureOutput:
    """Tests for ConfigureOutput dataclass."""

    def test_configure_output_success(self, tmp_path):
        """Test creating successful ConfigureOutput."""
        output = ConfigureOutput(
            success=True,
            config_path=tmp_path / "config.yaml",
            pipeline_name="test_pipeline",
            input_usd_path="/path/to/input.usd",
            materials_library_path="/path/to/materials",
            output_usd_path="/path/to/output.usd",
            dataset_dir="/path/to/dataset",
            predictions_dir="/path/to/predictions",
        )

        assert output.success is True
        assert output.config_path == tmp_path / "config.yaml"
        assert output.pipeline_name == "test_pipeline"

    def test_configure_output_error(self):
        """Test creating error ConfigureOutput."""
        output = ConfigureOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.config_path is None
