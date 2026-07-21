# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for session ID functionality."""

import logging
import uuid
from pathlib import Path

import pytest

from material_agent.api.builders import build_unified_pipeline_config
from material_agent.config.path_resolver import ProjectPathResolver
from material_agent.config.unified_config import UnifiedPipelineConfigTask


def test_session_id_auto_generation(tmp_path):
    """Test that session ID is auto-generated when not provided."""
    config_file = tmp_path / "config.yaml"

    config = {
        "project": {
            "name": "test_project",
            # No session_id provided
        },
        "input": {
            "usd_path": str(tmp_path / "input.usd"),
        },
        "output": {
            # No usd_path - will be auto-derived
        },
    }

    # Create a dummy input file
    (tmp_path / "input.usd").touch()

    # Create path resolver
    resolver = ProjectPathResolver(config, config_file)

    # Verify session ID was auto-generated
    assert resolver.session_id is not None
    assert len(resolver.session_id) == 36  # UUID format
    # Verify it's a valid UUID
    uuid.UUID(resolver.session_id)

    # Verify working directory uses session ID: .{session_id}
    assert f".{resolver.session_id}" in str(resolver.working_dir)

    # Verify output USD is auto-derived
    assert "output/output.usd" in str(resolver.output_usd)


def test_session_id_provided(tmp_path):
    """Test that provided session ID is used."""
    config_file = tmp_path / "config.yaml"
    test_session_id = "my-custom-session-123"

    config = {
        "project": {
            "name": "test_project",
            "session_id": test_session_id,
        },
        "input": {
            "usd_path": str(tmp_path / "input.usd"),
        },
        "output": {
            # No usd_path - will be auto-derived
        },
    }

    # Create a dummy input file
    (tmp_path / "input.usd").touch()

    # Create path resolver
    resolver = ProjectPathResolver(config, config_file)

    # Verify the provided session ID is used
    assert resolver.session_id == test_session_id

    # Verify working directory uses the provided session ID: .{session_id}
    assert f".{test_session_id}" in str(resolver.working_dir)


def test_session_id_with_custom_working_dir(tmp_path):
    """Test that custom working_dir overrides session-based path."""
    config_file = tmp_path / "config.yaml"
    test_session_id = "my-session"
    custom_working_dir = ".my_custom_dir"

    config = {
        "project": {
            "name": "test_project",
            "session_id": test_session_id,
            "working_dir": custom_working_dir,
        },
        "input": {
            "usd_path": str(tmp_path / "input.usd"),
        },
        "output": {
            # No usd_path - will be auto-derived
        },
    }

    # Create a dummy input file
    (tmp_path / "input.usd").touch()

    # Create path resolver
    resolver = ProjectPathResolver(config, config_file)

    # Verify session ID is set
    assert resolver.session_id == test_session_id

    # Verify custom working directory is used
    assert custom_working_dir in str(resolver.working_dir)
    # Should NOT have session_id in working_dir path when custom working_dir is used
    assert test_session_id not in str(resolver.working_dir)


def test_session_id_in_context(tmp_path):
    """Test that session ID is added to workflow context."""
    tmp_path / "config.yaml"

    config = {
        "project": {
            "name": "test_project",
        },
        "input": {
            "usd_path": str(tmp_path / "input.usd"),
        },
        "output": {
            # No usd_path - will be auto-derived
        },
        "materials": {
            "library_path": str(tmp_path / "materials.usd"),
            "entries": [],
        },
        "steps": {
            "build_dataset_usd": {"enabled": True},
        },
    }

    # Create dummy files
    (tmp_path / "input.usd").touch()
    (tmp_path / "materials.usd").touch()

    # Run config task
    task = UnifiedPipelineConfigTask()
    context = {"config_dict": config}
    result = task.run(context)

    # Verify session_id is in context
    assert "session_id" in result
    assert result["session_id"] is not None
    # Verify it's a valid UUID
    uuid.UUID(result["session_id"])


def test_build_unified_pipeline_config_with_session_id():
    """Test builder function with session_id parameter."""
    test_session_id = "test-session-abc123"

    config = build_unified_pipeline_config(
        project_name="test_project",
        input_usd_path="input.usd",
        materials_library_path="materials.usd",
        materials_entries=[
            {"name": "Steel", "binding": "/Materials/Steel"},
        ],
        session_id=test_session_id,
    )

    # Verify session_id is in config
    assert "project" in config
    assert "session_id" in config["project"]
    assert config["project"]["session_id"] == test_session_id

    # Verify working_dir is NOT set (will be auto-derived from session_id)
    assert "working_dir" not in config["project"]

    # Verify output.usd_path is NOT set (will be auto-derived)
    assert "output" in config
    assert "usd_path" not in config["output"]


def test_build_unified_pipeline_config_without_session_id():
    """Test builder without session_id (will be auto-generated)."""
    config = build_unified_pipeline_config(
        project_name="test_project",
        input_usd_path="input.usd",
        materials_library_path="materials.usd",
        materials_entries=[
            {"name": "Steel", "binding": "/Materials/Steel"},
        ],
    )

    # Verify session_id is NOT in config initially
    # (will be auto-generated by ProjectPathResolver)
    assert "project" in config
    assert "session_id" not in config["project"]

    # Verify output.usd_path is NOT set (will be auto-derived)
    assert "output" in config
    assert "usd_path" not in config["output"]


def test_backward_compatibility_no_session(tmp_path):
    """Test backward compatibility when neither session_id nor working_dir is provided."""
    config_file = tmp_path / "config.yaml"

    config = {
        "project": {
            "name": "test_project",
            # Neither session_id nor working_dir provided
        },
        "input": {
            "usd_path": str(tmp_path / "input.usd"),
        },
        "output": {
            # No usd_path - will be auto-derived
        },
    }

    # Create a dummy input file
    (tmp_path / "input.usd").touch()

    # Create path resolver
    resolver = ProjectPathResolver(config, config_file)

    # Should auto-generate a session ID
    assert resolver.session_id is not None

    # Working directory should use session-based structure: .{session_id}
    assert f".{resolver.session_id}" in str(resolver.working_dir)


def test_project_path_resolver_legacy_paths_and_summary(tmp_path, caplog):
    """Cover material-agent specific path helpers and legacy output override."""
    config_file = tmp_path / "config.yaml"
    input_usd = tmp_path / "input.usd"
    input_usd.touch()
    existing_ref = tmp_path / "existing.png"
    existing_ref.touch()

    config = {
        "project": {
            "name": "test_project",
            "session_id": "session-paths",
        },
        "input": {
            "usd_path": "input.usd",
            "prim_path": "/World/Robot",
            "reference_images": ["existing.png", "missing.png", None],
            "reference_pdfs": ["docs/spec.pdf", ""],
        },
        "output": {
            "usd_path": "legacy/output.usd",
            "layer_only": True,
            "flatten_output": False,
            "shader_target": "preview",
        },
    }

    resolver = ProjectPathResolver(config, config_file)

    assert resolver.input_usd == input_usd.resolve()
    assert resolver.prim_path == "/World/Robot"
    assert resolver.reference_images == [
        existing_ref.resolve(),
        (tmp_path / "missing.png").resolve(),
    ]
    assert resolver.reference_pdfs == [(tmp_path / "docs/spec.pdf").resolve()]
    assert resolver.output_usd == (tmp_path / "legacy/output.usd").resolve()
    assert resolver.layer_only is True
    assert resolver.flatten_output is False
    assert resolver.material_profile == "preview"

    assert resolver.get_step_dataset_file("predict").name == "dataset.jsonl"
    assert resolver.get_step_predictions_file().name == "predictions.jsonl"
    assert resolver.get_usd_dataset_dir().name == "usd"
    assert resolver.get_vectorstore_dir().name == "vectorstore"
    assert resolver.get_dataset_dir().name == "dataset"
    assert resolver.get_predictions_dir().name == "predictions"

    resolver.create_working_directories()
    assert resolver.working_dir.exists()
    assert resolver.get_output_dir().exists()
    assert resolver.get_temp_dir().exists()

    caplog.set_level(logging.WARNING)
    resolver.validate_input_paths()
    assert "Reference image not found" in caplog.text

    summary = resolver.get_path_summary()
    assert summary["project_name"] == "test_project"
    assert summary["input"]["usd_path"] == str(input_usd.resolve())
    assert summary["input"]["reference_images"] == [
        str(existing_ref.resolve()),
        str((tmp_path / "missing.png").resolve()),
    ]
    assert summary["output"]["usd_path"] == str(
        (tmp_path / "legacy/output.usd").resolve()
    )
    assert summary["output"]["layer_only"] is True
    assert summary["output"]["flatten_output"] is False
    assert summary["output"]["material_profile"] == "preview"
    assert Path(summary["step_outputs"]["build_dataset_usd"]).name == "usd"


def test_project_path_resolver_redacts_missing_input_diagnostics(
    tmp_path: Path,
) -> None:
    secret = "material-input-path-secret-713"
    config_dir = tmp_path / f"user:{secret}@assets.example.test"
    config_dir.mkdir()
    resolver = ProjectPathResolver(
        {
            "project": {"name": "safe", "working_dir": "work"},
            "input": {"usd_path": "missing.usd"},
            "output": {},
        },
        config_dir / "config.yaml",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        resolver.validate_input_paths()

    observable = f"{exc_info.value}\n{resolver!r}\n{resolver.get_path_summary()!r}"
    assert secret not in observable
    assert str(exc_info.value) == "Input USD file not found: <redacted>"
