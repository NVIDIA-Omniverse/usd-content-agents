# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Apply API."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from material_agent.api.apply import ApplyInput, ApplyOutput, run_apply
from material_agent.api.types import AssignmentStats, DownloadStats


class TestApplyInput:
    """Tests for ApplyInput validation."""

    def test_apply_input_valid(self, tmp_path):
        """Test creating valid ApplyInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = ApplyInput(config=config_file, render_enabled=True)

        assert params.config == config_file
        assert params.render_enabled is True
        assert params.layer_only is False

    def test_apply_input_missing_config(self, tmp_path):
        """Test ApplyInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ApplyInput(config=config_file)

    def test_apply_input_hides_config_path_inspection_failure(self) -> None:
        sentinel = "never-expose-apply-config-path-713"
        config_file = Path("/tmp") / (f"api_key={sentinel}" + "x" * 5000)

        with pytest.raises(OSError, match="^Unable to inspect config file$") as exc:
            ApplyInput(config=config_file)

        assert sentinel not in str(exc.value)
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None

    def test_apply_input_rejects_empty_config_dict(self):
        """Test ApplyInput raises error for empty config dictionaries."""
        with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
            ApplyInput(config={})

    def test_apply_input_with_overrides(self, tmp_path):
        """Test ApplyInput with all overrides."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        input_usd = tmp_path / "input.usd"
        predictions = tmp_path / "predictions.jsonl"
        output_usd = tmp_path / "output.usd"

        params = ApplyInput(
            config=config_file,
            input_usd_override=input_usd,
            predictions_override=predictions,
            output_usd_override=output_usd,
            layer_only=True,
            render_enabled=False,
            verbose=True,
        )

        assert params.input_usd_override == input_usd
        assert params.predictions_override == predictions
        assert params.output_usd_override == output_usd
        assert params.layer_only is True


class TestRunApply:
    """Tests for run_apply function."""

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_apply_success(self, mock_arun_pipeline, tmp_path):
        """Test successful apply execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock pipeline result
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=True,
            step_results={
                "apply": {
                    "output_usd_path": str(tmp_path / "output.usd"),
                    "unique_materials": ["steel", "rubber", "plastic"],
                    "matched_materials": {
                        "steel": [{"source_path": "/path/to/steel.mdl"}]
                    },
                    "resolved_materials": {"steel": "/local/steel.mdl"},
                    "materials_applied": {"steel": {"prims": ["prim1", "prim2"]}},
                    "material_profile_result": {
                        "requested_profile": "auto",
                        "resolved_profile": "omnipbr_mdl",
                        "warnings": [],
                        "errors": [],
                    },
                    "resolved_material_profile": "omnipbr_mdl",
                    "material_profile_warnings": [],
                    "material_profile_errors": [],
                    "assignment_stats": {
                        "materials_created": 3,
                        "materials_applied": 3,
                        "total_prims": 10,
                        "failed": 0,
                        "bound_prim_ids": ["/Root/PartA", "/Root/PartB"],
                        "unbound_prim_ids": ["/Root/PartC"],
                    },
                    "download_stats": {
                        "found_local": 2,
                        "downloaded": 1,
                        "failed": 0,
                        "skipped": 0,
                    },
                    "rendered_image_paths": [str(tmp_path / "render.png")],
                    "rendering_skipped": False,
                    "layer_only": False,
                }
            },
        )

        # Execute
        params = ApplyInput(config=config_file)
        result = run_apply(params)

        # Verify
        assert result.success is True
        assert result.output_usd_path == tmp_path / "output.usd"
        assert len(result.unique_materials) == 3
        assert result.assignment_stats.materials_created == 3
        assert result.assignment_stats.bound_prim_ids == [
            "/Root/PartA",
            "/Root/PartB",
        ]
        assert result.assignment_stats.unbound_prim_ids == ["/Root/PartC"]
        assert result.download_stats.found_local == 2
        assert len(result.rendered_image_paths) == 1
        assert result.resolved_material_profile == "omnipbr_mdl"
        assert result.material_profile_result["requested_profile"] == "auto"
        assert result.material_profile_warnings == []
        assert result.material_profile_errors == []

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_apply_logs_dict_config_and_overrides(
        self, mock_arun_pipeline, tmp_path
    ):
        """Test apply accepts in-memory config and path overrides."""
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=True,
            step_results={"apply": {}},
        )

        params = ApplyInput(
            config={"pipeline": {"steps": []}},
            input_usd_override=tmp_path / "input.usd",
            predictions_override=tmp_path / "predictions.jsonl",
            output_usd_override=tmp_path / "output.usd",
        )
        result = run_apply(params)

        assert result.success is True
        assert result.output_usd_path is None
        assert result.assignment_stats is None
        assert result.download_stats is None

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_apply_projects_public_metadata_but_keeps_runtime_config_raw(
        self,
        mock_arun_pipeline: AsyncMock,
        tmp_path: Path,
    ) -> None:
        from material_agent.api.pipeline import PipelineOutput

        sentinel = "api_key=apply-public-result-secret-713"
        runtime_apply_result: dict[str, Any] = {
            "output_usd_path": str(tmp_path / sentinel / "output.usd"),
            "unique_materials": ["steel"],
            "resolved_material_profile": "omnipbr_mdl",
            "assignment_stats": {
                "materials_created": 1,
                "bound_prim_ids": [f"/Root?access_token={sentinel}"],
            },
            "config_dict": {"vlm": {"api_key": sentinel}},
            "details": {"api_key": sentinel},
            "runtime_collaborator": object(),
        }
        mock_arun_pipeline.return_value = PipelineOutput(
            success=True,
            step_results={"apply": runtime_apply_result},
        )
        config = {"pipeline": {"steps": []}, "vlm": {"api_key": sentinel}}

        result = run_apply(ApplyInput(config=config))

        assert result.success is True
        assert result.output_usd_path is None
        assert result.unique_materials == ["steel"]
        assert result.assignment_stats is not None
        assert result.assignment_stats.materials_created == 1
        assert all(
            isinstance(prim_id, str)
            for prim_id in result.assignment_stats.bound_prim_ids
        )
        assert result.raw_result is not runtime_apply_result
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result
        assert "runtime_collaborator" not in result.raw_result
        assert sentinel not in repr(result)
        assert runtime_apply_result["config_dict"]["vlm"]["api_key"] == sentinel
        await_args = mock_arun_pipeline.await_args
        assert await_args is not None
        pipeline_input = await_args.args[0]
        assert pipeline_input.config is config
        assert pipeline_input.config["vlm"]["api_key"] == sentinel

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_apply_pipeline_failure(self, mock_arun_pipeline, tmp_path):
        """Test apply when pipeline fails."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock pipeline failure
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=False,
            error="api_key=never-replay-apply-result-error-713",
        )

        # Execute
        params = ApplyInput(config=config_file)
        result = run_apply(params)

        # Verify
        assert result.success is False
        assert result.error == "Material apply failed"

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_apply_exception(self, mock_arun_pipeline, tmp_path, caplog):
        """Test apply when exception occurs."""
        # Setup
        sentinel = "never-replay-apply-exception-713"
        config_dir = tmp_path / f"api_key={sentinel}"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("# test config")

        # Mock exception
        mock_arun_pipeline.side_effect = RuntimeError(f"backend reflected {sentinel}")

        # Execute
        params = ApplyInput(config=config_file)
        result = run_apply(params)

        # Verify
        assert result.success is False
        assert result.error == "Material apply failed"
        assert sentinel not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)


class TestApplyOutput:
    """Tests for ApplyOutput dataclass."""

    def test_apply_output_success(self, tmp_path):
        """Test creating successful ApplyOutput."""
        assignment_stats = AssignmentStats(materials_created=5, total_prims=20)
        download_stats = DownloadStats(found_local=3, downloaded=2)

        output = ApplyOutput(
            success=True,
            output_usd_path=tmp_path / "output.usd",
            unique_materials=["steel", "rubber"],
            assignment_stats=assignment_stats,
            download_stats=download_stats,
        )

        assert output.success is True
        assert output.output_usd_path == tmp_path / "output.usd"
        assert len(output.unique_materials) == 2
        assert output.assignment_stats.materials_created == 5

    def test_apply_output_error(self):
        """Test creating error ApplyOutput."""
        output = ApplyOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.output_usd_path is None
