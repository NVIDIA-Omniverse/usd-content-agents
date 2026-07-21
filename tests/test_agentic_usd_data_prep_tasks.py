# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD data preparation tasks."""

import builtins
import os
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import yaml
from PIL import Image, ImageDraw

from world_understanding.agentic.config import ConfigSourceError
from world_understanding.agentic.usd_tasks import (
    USDDataPrepConfigTask,
    USDDatasetManifestTask,
    USDLoadingTask,
    USDPrimTraversalAndRenderingTask,
    USDRendererProvisioningTask,
)
from world_understanding.agentic.usd_tasks.prim_traversal import (
    _zero_image_render_error_message,
    prim_path_to_directory_structure,
)
from world_understanding.functions.graphics.render_remote import RenderingStatus
from world_understanding.functions.graphics.rendering import (
    CameraViewType,
    RemoteRenderingBackend,
    RenderingConfig,
)
from world_understanding.functions.graphics.rendering_backend_factory import (
    create_rendering_backend,
)
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.usd.stage import MAX_PATH_COMPONENT_LEN


class TestUSDDataPrepTasks:
    """Basic tests for USD data preparation tasks."""

    def test_usd_data_prep_config_task_init(self):
        """Test USDDataPrepConfigTask initialization."""
        task = USDDataPrepConfigTask()
        assert task.name == "USDDataPrepConfig"
        assert (
            task.description == "Load and validate USD data preparation configuration"
        )
        assert hasattr(task, "run")

    def test_usd_renderer_provisioning_task_init(self):
        """Test USDRendererProvisioningTask initialization."""
        task = USDRendererProvisioningTask()
        assert task.name == "USDRendererProvisioning"
        assert task.description == "Provision USD renderer backend and resources"
        assert hasattr(task, "run")

    def test_usd_loading_task_init(self):
        """Test USDLoadingTask initialization."""
        task = USDLoadingTask()
        assert task.name == "USDLoading"
        assert task.description == "Load USD stage and apply configurations"
        assert hasattr(task, "run")

    def test_usd_prim_traversal_task_init(self):
        """Test USDPrimTraversalAndRenderingTask initialization."""
        task = USDPrimTraversalAndRenderingTask()
        assert task.name == "USDPrimTraversalAndRendering"
        assert task.description == "Traverse USD prims and render configured views"
        assert hasattr(task, "run")

    def test_usd_dataset_manifest_task_init(self):
        """Test USDDatasetManifestTask initialization."""
        task = USDDatasetManifestTask()
        assert task.name == "USDDatasetManifest"
        assert (
            task.description == "Create dataset.json and prims.jsonl files for USD data"
        )
        assert hasattr(task, "run")

    def test_all_tasks_are_importable(self):
        """Test that all USD data prep tasks can be imported."""
        # All tasks are already imported at the top of the file
        # This test verifies they are accessible
        assert USDDataPrepConfigTask is not None
        assert USDDatasetManifestTask is not None
        assert USDLoadingTask is not None
        assert USDPrimTraversalAndRenderingTask is not None
        assert USDRendererProvisioningTask is not None

    def test_tasks_inherit_from_base_task(self):
        """Test that all USD tasks inherit from the base Task class."""
        from world_understanding.agentic.tasks import Task

        assert issubclass(USDDataPrepConfigTask, Task)
        assert issubclass(USDDatasetManifestTask, Task)
        assert issubclass(USDLoadingTask, Task)
        assert issubclass(USDPrimTraversalAndRenderingTask, Task)
        assert issubclass(USDRendererProvisioningTask, Task)

    def test_config_task_propagates_max_concurrent_requests(self, tmp_path):
        """Config should pass render concurrency through to traversal."""
        config_path = tmp_path / "build_dataset_usd.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "usd_path: scene.usd",
                    "output_dir: output",
                    "batch_size: 64",
                    "num_workers: 1",
                    "max_concurrent_requests: 1",
                ]
            )
        )

        task = USDDataPrepConfigTask()
        result = task.run({"config_path": str(config_path)}, Mock(spec=ObjectStore))

        assert result["batch_size"] == 64
        assert result["num_workers"] == 1
        assert result["max_concurrent_requests"] == 1

    def test_config_task_uses_shared_missing_source_contract(self) -> None:
        with pytest.raises(
            ConfigSourceError,
            match="Either config_path or config_dict must be provided",
        ):
            USDDataPrepConfigTask().run({}, Mock(spec=ObjectStore))

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("max_concurrent_requests", 0),
            ("max_concurrent_requests", -1),
            ("max_concurrent_requests", "1"),
            ("max_concurrent_requests", True),
            ("num_workers", 0),
            ("num_workers", -1),
            ("num_workers", "1"),
            ("num_workers", False),
        ],
    )
    def test_config_task_rejects_invalid_render_concurrency(
        self, tmp_path, field_name, value
    ):
        """Render concurrency settings must be positive integers."""
        config_path = tmp_path / "build_dataset_usd.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "usd_path": "scene.usd",
                    "output_dir": "output",
                    field_name: value,
                }
            )
        )

        task = USDDataPrepConfigTask()
        with pytest.raises(
            ValueError, match=f"{field_name} must be a positive integer"
        ):
            task.run({"config_path": str(config_path)}, Mock(spec=ObjectStore))


class TestUSDDataPrepConfigTask:
    """Functional tests for USDDataPrepConfigTask."""

    def test_run_plumbs_max_concurrent_requests(self, tmp_path):
        """Config should carry async render request concurrency into context."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "usd_path": "scene.usd",
                    "output_dir": "dataset",
                    "batch_size": 16,
                    "max_concurrent_requests": 4,
                }
            ),
            encoding="utf-8",
        )
        context = {"config_path": str(config_path)}
        object_store = Mock(spec=ObjectStore)

        result = USDDataPrepConfigTask().run(context, object_store)

        assert result["usd_path"] == tmp_path / "scene.usd"
        assert result["output_dir"] == tmp_path / "dataset"
        assert result["batch_size"] == 16
        assert result["max_concurrent_requests"] == 4

    def test_run_plumbs_render_evidence_failure_policy(self, tmp_path):
        """Config should carry render-evidence failure policy into context."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "usd_path": "scene.usd",
                    "output_dir": "dataset",
                    "fail_on_blank_dataset_renders": False,
                    "fail_on_missing_prim_images": False,
                }
            ),
            encoding="utf-8",
        )
        context = {"config_path": str(config_path)}
        object_store = Mock(spec=ObjectStore)

        result = USDDataPrepConfigTask().run(context, object_store)

        assert result["fail_on_blank_dataset_renders"] is False
        assert result["fail_on_missing_prim_images"] is False

    @pytest.mark.parametrize(
        "policy_key",
        ["fail_on_blank_dataset_renders", "fail_on_missing_prim_images"],
    )
    def test_run_rejects_non_boolean_render_evidence_failure_policy(
        self,
        tmp_path,
        policy_key,
    ):
        """Evidence-policy flags must be booleans, not truthy strings."""
        sentinel = "api_key=never-echo-render-policy-713"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "usd_path": "scene.usd",
                    "output_dir": "dataset",
                    policy_key: sentinel,
                }
            ),
            encoding="utf-8",
        )
        object_store = Mock(spec=ObjectStore)

        with pytest.raises(
            ValueError,
            match=f"{policy_key} must be a boolean",
        ) as exc_info:
            USDDataPrepConfigTask().run(
                {"config_path": str(config_path)},
                object_store,
            )
        assert sentinel not in str(exc_info.value)

    def test_run_rejects_non_mapping_yaml_root(self, tmp_path):
        """Config YAML root must be a mapping so config errors stay clear."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not-a-mapping\n", encoding="utf-8")
        object_store = Mock(spec=ObjectStore)

        with pytest.raises(
            ValueError,
            match="must be a mapping at the document root",
        ):
            USDDataPrepConfigTask().run(
                {"config_path": str(config_path)},
                object_store,
            )

    @pytest.mark.parametrize(
        ("config_extra", "error_match"),
        [
            ({"batch_size": 0}, "batch_size must be a positive integer"),
            ({"renderer": {"image_width": 0}}, "renderer.image_width must"),
            ({"renderer": {"image_height": 0}}, "renderer.image_height must"),
        ],
    )
    def test_run_rejects_non_positive_integer_config(
        self,
        tmp_path,
        config_extra,
        error_match,
    ):
        """Render dimensions and batch sizes should fail at config load time."""
        config = {
            "usd_path": "scene.usd",
            "output_dir": "dataset",
            **config_extra,
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        object_store = Mock(spec=ObjectStore)

        with pytest.raises(ValueError, match=error_match):
            USDDataPrepConfigTask().run(
                {"config_path": str(config_path)},
                object_store,
            )

    def test_run_does_not_echo_invalid_integer_config_value(self, tmp_path):
        """Invalid scalar values must not cross the public error boundary."""
        sentinel = "api_key=never-echo-batch-size-713"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "usd_path": "scene.usd",
                    "output_dir": "dataset",
                    "batch_size": sentinel,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="batch_size must be a positive integer, got str",
        ) as exc_info:
            USDDataPrepConfigTask().run(
                {"config_path": str(config_path)},
                Mock(spec=ObjectStore),
            )
        assert sentinel not in str(exc_info.value)

    def test_run_accepts_config_dict(self, tmp_path, monkeypatch):
        """Config dictionaries should use the same loader path as YAML files."""
        monkeypatch.chdir(tmp_path)
        context = {
            "config_dict": {
                "usd_path": "scene.usd",
                "output_dir": "dataset",
                "batch_size": 32,
                "max_concurrent_requests": 8,
                "renderer": {
                    "backend": "remote",
                    "image_width": 256,
                    "image_height": 256,
                },
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = USDDataPrepConfigTask().run(context, object_store)

        assert result["usd_path"] == tmp_path / "scene.usd"
        assert result["output_dir"] == tmp_path / "dataset"
        assert result["batch_size"] == 32
        assert result["max_concurrent_requests"] == 8
        assert result["renderer_config"]["image_width"] == 256

    def test_config_dict_nested_values_are_isolated_from_the_caller(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        source = {
            "usd_path": "scene.usd",
            "output_dir": "dataset",
            "prim_filters": {"types": ["Mesh"]},
            "renderer": {"backend": "remote"},
        }

        result = USDDataPrepConfigTask().run(
            {"config_dict": source},
            Mock(spec=ObjectStore),
        )
        result["prim_filters"]["types"].append("Xform")
        result["renderer_config"]["backend"] = "mutated"

        assert source["prim_filters"] == {"types": ["Mesh"]}
        assert source["renderer"] == {"backend": "remote"}

    def test_malformed_yaml_diagnostic_cannot_render_source_credentials(self, tmp_path):
        sentinel = "ZXCV713SharedUSDConfigAliasSecretQWER"
        config_path = tmp_path / "malformed.yaml"
        config_path.write_text(
            f"renderer:\n  api_key: *{sentinel}\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError) as exc:
            USDDataPrepConfigTask().run(
                {"config_path": config_path},
                Mock(spec=ObjectStore),
            )

        observable = "".join(traceback.format_exception(exc.value))
        assert sentinel not in str(exc.value)
        assert sentinel not in observable
        assert exc.value.__cause__ is None

    def test_run_uses_missing_config_defaults_with_source_override(
        self, tmp_path, monkeypatch, caplog
    ):
        """A missing config file can still be driven by explicit CLI-style paths."""
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "missing.yaml"
        context = {
            "config_path": str(config_path),
            "source_override": "scene.usd",
        }
        object_store = Mock(spec=ObjectStore)

        result = USDDataPrepConfigTask().run(context, object_store)

        assert result["usd_path"] == tmp_path / "scene.usd"
        assert result["output_dir"] == tmp_path / "output" / "dataset"
        assert result["render_output_dir"].is_dir()
        assert "Configuration file not found; using defaults" in caplog.text

    def test_run_resolves_relative_output_dir_override(self, tmp_path, monkeypatch):
        """CLI-style output overrides are resolved from the current directory."""
        monkeypatch.chdir(tmp_path)
        context = {
            "config_dict": {
                "usd_path": "scene.usd",
            },
            "output_dir_override": "override-dataset",
        }
        object_store = Mock(spec=ObjectStore)

        result = USDDataPrepConfigTask().run(context, object_store)

        assert result["output_dir"] == tmp_path / "override-dataset"
        assert result["render_output_dir"] == tmp_path / "override-dataset" / "renders"


class TestUSDRendererProvisioningTask:
    """Functional tests for USDRendererProvisioningTask."""

    def test_run_with_nvcf_backend(self):
        """Test running with NVCF backend configuration."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": 1920,
                "image_height": 1080,
            }
        }
        object_store = Mock(spec=ObjectStore)

        with patch.dict(os.environ, {"NGC_API_KEY": "test-api-key"}):
            result = task.run(context, object_store)

        # Verify NVCF backend was created
        assert "rendering_backend" in result
        assert isinstance(result["rendering_backend"], RemoteRenderingBackend)
        assert result["rendering_backend"].api_key == "test-api-key"

        # Verify custom image dimensions
        assert result["image_height"] == 1080

    def test_run_with_nvcf_endpoint_and_transfer_options(self):
        """Remote renderer config should pass endpoint and transfer options through."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "base_url": "http://localhost:8001",
                "use_data_uri": True,
                "s3_bucket": "test-bucket",
                "s3_region": "us-west-2",
                "s3_profile": "test-profile",
                "bundle_mdl_assets": False,
                "timeout": 42,
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        backend = result["rendering_backend"]
        assert isinstance(backend, RemoteRenderingBackend)
        assert backend.base_url == "http://localhost:8001"
        assert backend.use_data_uri is True
        assert backend.s3_bucket == "test-bucket"
        assert backend.s3_region == "us-west-2"
        assert backend.s3_profile == "test-profile"
        assert backend.bundle_mdl_assets is False
        assert backend.timeout == 42

    def test_run_with_ovrtx_forwards_material_options(self):
        """OvRTX renderer config should pass material options to backend."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "ovrtx",
                "log_level": "info",
                "add_preview_fallbacks": True,
                "material_target": "auto",
            }
        }
        object_store = Mock(spec=ObjectStore)
        calls: list[dict[str, Any]] = []

        class FakeOvRTXRenderingBackend:
            def __init__(self, **kwargs: Any):
                calls.append(kwargs)

        with patch(
            "world_understanding.functions.graphics.rendering_backend_factory.OvRTXRenderingBackend",
            FakeOvRTXRenderingBackend,
        ):
            result = task.run(context, object_store)

        assert calls == [
            {
                "log_level": "info",
                "ovrtx_venv_dir": None,
                "num_sensor_updates": 32,
                "render_mode": "rt2",
                "add_preview_fallbacks": True,
                "material_target": "auto",
            }
        ]
        assert isinstance(result["rendering_backend"], FakeOvRTXRenderingBackend)

    def test_shared_ovrtx_backend_forwards_material_options(self):
        """The shared factory should forward preview material options."""
        calls: list[dict[str, Any]] = []

        class FakeOvRTXRenderingBackend:
            def __init__(self, **kwargs: Any):
                calls.append(kwargs)

        with patch(
            "world_understanding.functions.graphics.rendering_backend_factory.OvRTXRenderingBackend",
            FakeOvRTXRenderingBackend,
        ):
            backend = create_rendering_backend(
                "ovrtx",
                {
                    "add_preview_fallbacks": True,
                    "material_target": "preview_surface",
                },
            )

        assert calls == [
            {
                "log_level": "warn",
                "ovrtx_venv_dir": None,
                "num_sensor_updates": 32,
                "render_mode": "rt2",
                "add_preview_fallbacks": True,
                "material_target": "preview_surface",
            }
        ]
        assert isinstance(backend, FakeOvRTXRenderingBackend)

    def test_run_with_unknown_backend_raises_error(self):
        """Test that unknown backend raises ValueError."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "unknown_backend",
            }
        }
        object_store = Mock(spec=ObjectStore)

        with pytest.raises(
            ValueError, match="Unknown rendering backend: unknown_backend"
        ):
            task.run(context, object_store)

    def test_run_with_custom_configuration(self):
        """Test running with custom configuration values."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": 2048,
                "image_height": 1536,
                "cull_style": "front",
                "should_highlight_prim": False,
                "should_assign_random_colors": True,
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify custom configuration was applied
        config = result["rendering_config"]
        assert config.image_width == 2048
        assert config.cull_style == "front"
        assert config.should_highlight_prim is False
        assert config.should_assign_random_colors is True
        assert result["image_height"] == 1536

    def test_run_with_minimal_configuration(self):
        """Test running with minimal configuration (only backend specified)."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify defaults were applied
        config = result["rendering_config"]
        assert config.image_width == 512  # default
        assert config.cull_style == "back"  # default
        assert config.should_highlight_prim is False  # default
        assert config.should_assign_random_colors is True  # default
        assert result["image_height"] == 512  # default

    def test_run_with_empty_renderer_config(self):
        """Test running with empty renderer_config dict."""
        task = USDRendererProvisioningTask()
        context = {"renderer_config": {}}
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify all defaults were applied
        config = result["rendering_config"]
        assert config.image_width == 512
        assert config.cull_style == "back"
        assert config.should_highlight_prim is False
        assert config.should_assign_random_colors is True
        assert result["image_height"] == 512

    def test_run_without_renderer_config_key(self):
        """Test running when renderer_config key is missing from context."""
        task = USDRendererProvisioningTask()
        context = {}  # No renderer_config key
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify defaults were applied
        config = result["rendering_config"]
        assert config.image_width == 512
        assert config.cull_style == "back"
        assert config.should_highlight_prim is False
        assert config.should_assign_random_colors is True
        assert result["image_height"] == 512

    def test_run_with_corner_camera_view_type(self):
        """Test running with corner camera view type (default)."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "corner",
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify corner view configuration
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.CORNER
        assert config.camera_name_prefix == "CornerViewCamera"

        # Verify default corner camera directions (8 corners)
        expected_directions = [
            "+x+y+z",
            "-x+y+z",
            "-x-y+z",
            "+x-y+z",
            "+x+y-z",
            "-x+y-z",
            "-x-y-z",
            "+x-y-z",
        ]
        assert config.camera_ordering == expected_directions

    def test_run_with_side_camera_view_type(self):
        """Test running with side camera view type."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "side",
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify side view configuration
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.SIDE
        assert config.camera_name_prefix == "SideViewCamera"

        # Verify default side camera directions (6 cardinal directions)
        expected_directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
        assert config.camera_ordering == expected_directions

    def test_run_with_custom_camera_directions(self):
        """Test running with custom camera directions."""
        task = USDRendererProvisioningTask()
        custom_directions = ["+x", "+y", "+z"]
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "corner",
                "camera_directions": custom_directions,
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify custom directions were used
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.CORNER
        assert config.camera_ordering == custom_directions

    def test_run_with_invalid_camera_view_type_falls_back_to_corner(self):
        """Test that invalid camera view type falls back to corner."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "invalid_type",
            }
        }
        object_store = Mock(spec=ObjectStore)

        # Mock the event listener to capture warning calls
        mock_listener = Mock()
        with patch(
            "world_understanding.agentic.usd_tasks.renderer.get_listener",
            return_value=mock_listener,
        ):
            result = task.run(context, object_store)

        # Verify fallback to corner view
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.CORNER
        assert config.camera_name_prefix == "CornerViewCamera"

        # Verify warning was logged through the event listener
        mock_listener.warning.assert_called_once()
        warning_call = mock_listener.warning.call_args[0][0]
        assert "Invalid camera_view_type 'invalid_type', using 'corner'" in warning_call

    def test_run_with_case_insensitive_camera_view_type(self):
        """Test that camera view type is case insensitive."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "SIDE",  # uppercase
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify side view was recognized despite case
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.SIDE
        assert config.camera_name_prefix == "SideViewCamera"

    def test_run_with_nvcf_api_key_from_environment(self):
        """Test running with NVCF API key from environment."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        api_key = "test-nvcf-api-key-12345"
        with patch.dict(os.environ, {"NGC_API_KEY": api_key}):
            result = task.run(context, object_store)

        # Verify API key was used
        backend = result["rendering_backend"]
        assert isinstance(backend, RemoteRenderingBackend)
        assert backend.api_key == api_key

    def test_run_with_nvcf_no_api_key(self):
        """Test running with NVCF backend when API key is not set."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        # Ensure NGC_API_KEY is not set
        with patch.dict(os.environ, {}, clear=True):
            result = task.run(context, object_store)

        # Verify backend was created with None API key
        backend = result["rendering_backend"]
        assert isinstance(backend, RemoteRenderingBackend)
        assert backend.api_key is None

    def test_run_with_multiple_environment_variables(self):
        """Test running with multiple environment variables set."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        env_vars = {
            "NGC_API_KEY": "nvcf-key-123",
            "OTHER_VAR": "ignored",
        }

        with patch.dict(os.environ, env_vars):
            result = task.run(context, object_store)

        # Verify only relevant environment variable was used
        backend = result["rendering_backend"]
        assert isinstance(backend, RemoteRenderingBackend)
        assert backend.api_key == "nvcf-key-123"

    def test_run_with_none_context_raises_error(self):
        """Test that None context raises appropriate error."""
        task = USDRendererProvisioningTask()
        object_store = Mock(spec=ObjectStore)

        with pytest.raises(AttributeError):
            task.run(None, object_store)

    def test_run_with_none_object_store_raises_error(self):
        """Test that None object store raises appropriate error."""
        task = USDRendererProvisioningTask()
        context = {"renderer_config": {}}

        with pytest.raises(AttributeError):
            task.run(context, None)

    def test_run_with_invalid_renderer_config_type(self):
        """Test that invalid renderer_config type raises AttributeError."""
        task = USDRendererProvisioningTask()
        context = {"renderer_config": "invalid_string"}  # Should be dict
        object_store = Mock(spec=ObjectStore)

        # This should raise an AttributeError because strings don't have .get() method
        with pytest.raises(AttributeError, match="'str' object has no attribute 'get'"):
            task.run(context, object_store)

    def test_run_with_malformed_camera_directions(self):
        """Test handling of malformed camera directions."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "corner",
                "camera_directions": "not_a_list",  # Should be list
            }
        }
        object_store = Mock(spec=ObjectStore)

        # This should not raise an error, but should use the malformed value
        result = task.run(context, object_store)

        # Verify the malformed value was used as-is
        config = result["rendering_config"]
        assert config.camera_ordering == "not_a_list"

    def test_run_with_empty_camera_directions(self):
        """Test handling of empty camera directions list."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "corner",
                "camera_directions": [],  # Empty list
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify empty list was used
        config = result["rendering_config"]
        assert config.camera_ordering == []

    def test_run_with_negative_image_dimensions(self):
        """Test handling of negative image dimensions."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": -100,  # Negative width
                "image_height": -50,  # Negative height
            }
        }
        object_store = Mock(spec=ObjectStore)

        # This should not raise an error, but should use the negative values
        result = task.run(context, object_store)

        # Verify negative values were used (validation happens elsewhere)
        config = result["rendering_config"]
        assert config.image_width == -100
        assert result["image_height"] == -50

    def test_run_with_zero_image_dimensions(self):
        """Test handling of zero image dimensions."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": 0,
                "image_height": 0,
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify zero values were used
        config = result["rendering_config"]
        assert config.image_width == 0
        assert result["image_height"] == 0

    def test_run_stores_backend_in_object_store(self):
        """Test that backend is properly stored in object store."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        task.run(context, object_store)

        # Verify backend was stored in object store
        backend_calls = [
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_backend"
        ]
        assert len(backend_calls) == 1
        assert isinstance(backend_calls[0][0][1], RemoteRenderingBackend)

    def test_run_stores_config_in_object_store(self):
        """Test that config is properly stored in object store."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": 1024,
            }
        }
        object_store = Mock(spec=ObjectStore)

        task.run(context, object_store)

        # Verify config was stored in object store
        config_calls = [
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_config"
        ]
        assert len(config_calls) == 1
        assert isinstance(config_calls[0][0][1], RenderingConfig)
        assert config_calls[0][0][1].image_width == 1024

    def test_run_updates_context_with_backend_and_config(self):
        """Test that context is updated with backend and config references."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify context was updated
        assert "rendering_backend" in result
        assert "rendering_config" in result
        assert "image_height" in result

        # Verify the objects are the same as those stored in object store
        backend_call = next(
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_backend"
        )
        config_call = next(
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_config"
        )

        assert result["rendering_backend"] is backend_call[0][1]
        assert result["rendering_config"] is config_call[0][1]

    def test_run_preserves_existing_context_keys(self):
        """Test that existing context keys are preserved."""
        task = USDRendererProvisioningTask()
        context = {
            "existing_key": "existing_value",
            "another_key": 123,
            "renderer_config": {
                "backend": "remote",
            },
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify existing keys are preserved
        assert result["existing_key"] == "existing_value"
        assert result["another_key"] == 123

        # Verify new keys were added
        assert "rendering_backend" in result
        assert "rendering_config" in result
        assert "image_height" in result

    def test_run_object_store_called_twice(self):
        """Test that object store set method is called exactly twice."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        task.run(context, object_store)

        # Verify set was called exactly twice
        assert object_store.set.call_count == 2

        # Verify the keys that were stored
        stored_keys = [call[0][0] for call in object_store.set.call_args_list]
        assert "rendering_backend" in stored_keys
        assert "rendering_config" in stored_keys

    def test_run_with_nvcf_stores_correct_backend_type(self):
        """Test that NVCF backend is stored with correct type."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        with patch.dict(os.environ, {"NGC_API_KEY": "test-key"}):
            task.run(context, object_store)

        # Verify NVCF backend was stored
        backend_call = next(
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_backend"
        )
        stored_backend = backend_call[0][1]
        assert isinstance(stored_backend, RemoteRenderingBackend)
        assert stored_backend.api_key == "test-key"


class TestUSDPrimTraversalZeroImages:
    """Tests that USDPrimTraversalAndRenderingTask fails when no images are rendered."""

    @staticmethod
    def _save_nonblank_image(path: Path) -> None:
        image = Image.new("RGB", (32, 32), (220, 220, 220))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, 12, 31], fill=(255, 0, 0))
        draw.rectangle([13, 0, 23, 31], fill=(0, 255, 0))
        draw.rectangle([24, 0, 31, 31], fill=(0, 0, 255))
        image.save(path)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [0, -1, True, "not-an-int"])
    async def test_arun_rejects_invalid_max_concurrent_requests(
        self, tmp_path, value
    ) -> None:
        """Async render path must reject invalid concurrency before semaphore use."""
        task = USDPrimTraversalAndRenderingTask()
        rendering_config = RenderingConfig()
        object_store = Mock(spec=ObjectStore)
        object_store.get.side_effect = lambda key, default=None: {
            "usd_stage": object(),
            "rendering_backend": Mock(),
            "rendering_config": rendering_config,
            "usd_model": None,
        }.get(key, default)

        context: dict = {
            "prim_filters": {"types": ["UsdGeom.Mesh"]},
            "render_output_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "batch_size": 10,
            "rendering_modes": ["prim_only"],
            "max_concurrent_requests": value,
        }

        with pytest.raises(ValueError, match="max_concurrent_requests"):
            await task.arun(context, object_store)

    def test_no_matching_prims_fail_before_stage_preparation(self, tmp_path):
        """A filter mismatch must fail before preparing or uploading stages."""
        from pxr import Usd, UsdGeom

        task = USDPrimTraversalAndRenderingTask()
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Cube.Define(stage, "/World/Cube")
        mock_backend = Mock()
        rendering_config = RenderingConfig()

        object_store = Mock(spec=ObjectStore)
        object_store.get.side_effect = lambda key, default=None: {
            "usd_stage": stage,
            "rendering_backend": mock_backend,
            "rendering_config": rendering_config,
            "usd_model": None,
        }.get(key, default)

        context: dict = {
            "prim_filters": {"types": ["UsdGeom.Mesh"]},
            "render_output_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "skip_existing": False,
            "batch_size": 10,
            "rendering_modes": ["prim_only"],
        }

        prepare = Mock()
        upload = Mock()
        with (
            patch.object(task, "_prepare_stages", prepare),
            patch.object(task, "_upload_stages_to_s3", upload),
        ):
            with pytest.raises(RuntimeError, match="No USD prims matched") as exc:
                task.run(context, object_store)

        message = str(exc.value)
        assert '"types": ["UsdGeom.Mesh"]' in message
        assert '"Cube": 1' in message
        assert '"Xform": 1' in message
        prepare.assert_not_called()
        upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_no_matching_prims_fail_before_stage_preparation(
        self, tmp_path
    ) -> None:
        from pxr import Usd, UsdGeom

        task = USDPrimTraversalAndRenderingTask()
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Sphere.Define(stage, "/World/Sphere")
        object_store = Mock(spec=ObjectStore)
        object_store.get.side_effect = lambda key, default=None: {
            "usd_stage": stage,
            "rendering_backend": Mock(),
            "rendering_config": RenderingConfig(),
            "usd_model": None,
        }.get(key, default)
        context = {
            "prim_filters": {"types": ["UsdGeom.Mesh"]},
            "render_output_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "batch_size": 10,
            "rendering_modes": ["prim_only"],
            "max_concurrent_requests": 1,
        }
        prepare = Mock()
        upload = Mock()

        with (
            patch.object(task, "_prepare_stages", prepare),
            patch.object(task, "_upload_stages_to_s3_async", upload),
        ):
            with pytest.raises(RuntimeError, match="No USD prims matched") as exc:
                await task.arun(context, object_store)

        message = str(exc.value)
        assert '"Sphere": 1' in message
        assert '"Xform": 1' in message
        prepare.assert_not_called()
        upload.assert_not_called()

    def test_zero_image_render_error_uses_local_rest_endpoint(self, monkeypatch):
        """Local REST renderer failures should not be described as NVCF failures."""
        monkeypatch.setenv("RENDER_ENDPOINT", "http://host.docker.internal:8001")
        monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)

        message = _zero_image_render_error_message(RemoteRenderingBackend())

        assert "NVCF" not in message
        assert "configured render endpoint" in message
        assert "RENDER_ENDPOINT" in message
        assert "http://host.docker.internal:8001" in message

    def test_zero_image_render_error_mentions_nvcf_for_function_id(self, monkeypatch):
        """NVCF wording should remain when the function-id fallback is used."""
        monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
        monkeypatch.setenv("NVCF_RENDER_FUNCTION_ID", "render-function")

        message = _zero_image_render_error_message(RemoteRenderingBackend())

        assert "NVCF render function" in message
        assert "NVCF_RENDER_FUNCTION_ID" in message
        assert "https://render-function.invocation.api.nvcf.nvidia.com" in message

    def test_zero_image_render_error_sanitizes_endpoint_details(self):
        """Endpoint diagnostics should omit credentials and query parameters."""
        backend = RemoteRenderingBackend(
            base_url="https://user:secret@renderer.example.test:8443/render?token=abc"
        )

        message = _zero_image_render_error_message(backend)

        assert "https://renderer.example.test:8443/render" in message
        assert "secret" not in message
        assert "token=abc" not in message

    def test_zero_image_render_error_sanitizes_scheme_less_endpoint_details(self):
        """Scheme-less endpoint diagnostics should also omit URL secrets."""
        backend = RemoteRenderingBackend(
            base_url="user:secret@renderer.example.test:8443/render?token=abc"
        )

        message = _zero_image_render_error_message(backend)

        assert "http://renderer.example.test:8443/render" in message
        assert "secret" not in message
        assert "token=abc" not in message

    def test_zero_image_render_error_preserves_ipv6_endpoint_details(self):
        """Endpoint sanitization should preserve IPv6 brackets and ports."""
        backend = RemoteRenderingBackend(
            base_url="http://user:secret@[2001:db8::1]:8080/render?token=abc"
        )

        message = _zero_image_render_error_message(backend)

        assert "http://[2001:db8::1]:8080/render" in message
        assert "secret" not in message
        assert "token=abc" not in message

    def test_zero_image_render_error_mentions_nvcf_for_base_url(self):
        """Explicit NVCF invocation URLs should keep NVCF-specific wording."""
        backend = RemoteRenderingBackend(
            base_url="https://render-fn.invocation.api.nvcf.nvidia.com"
        )

        message = _zero_image_render_error_message(backend)

        assert "NVCF render function" in message
        assert "base_url" in message

    def test_zero_image_render_error_uses_generic_local_renderer_message(self):
        """Non-REST renderers should use generic renderer wording."""
        message = _zero_image_render_error_message(object())

        assert message == (
            "Rendering produced 0 images. Check renderer availability and logs above."
        )

    def test_zero_image_render_error_preserves_uppercase_https_endpoint(self):
        """URL detection should not depend on lowercase schemes."""
        backend = RemoteRenderingBackend(base_url="HTTPS://renderer.example.test")

        message = _zero_image_render_error_message(backend)

        assert "NVCF" not in message
        assert "renderer.example.test" in message

    def test_zero_image_render_error_preserves_scheme_less_host_port(self):
        """Scheme-less host:port endpoints should be treated as REST endpoints."""
        backend = RemoteRenderingBackend(base_url="host.docker.internal:8001")

        message = _zero_image_render_error_message(backend)

        assert "NVCF" not in message
        assert "http://host.docker.internal:8001" in message

    def test_zero_image_render_error_handles_malformed_endpoint(self):
        """Malformed endpoint values should not mask the zero-image failure."""
        backend = RemoteRenderingBackend(base_url="http://[::1")

        message = _zero_image_render_error_message(backend)

        assert message.startswith("Rendering produced 0 images.")
        assert "NVCF" not in message
        assert "http://[::1" in message

    def test_upload_stages_to_s3_forwards_bundle_mdl_assets(self, tmp_path):
        """Reusable stage upload must honor the remote renderer asset bundle flag."""
        task = USDPrimTraversalAndRenderingTask()
        backend = RemoteRenderingBackend(
            api_key="test-api-key",
            use_data_uri=True,
            bundle_mdl_assets=False,
        )
        prepared_stages = {
            "prim_only": {
                "type": "prim_only",
                "data": (Mock(), ["/Camera"], 1),
            },
            "composition": {
                "type": "composition",
                "data": (
                    (Mock(), ["/HighlightCamera"], 1),
                    (Mock(), ["/PlainCamera"], 1),
                ),
            },
        }
        listener = Mock()
        calls: list[dict[str, Any]] = []

        def fake_export_stage_to_s3(stage, **kwargs):
            calls.append(kwargs)
            return "data:application/usd;base64,test", None

        with patch(
            "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
            fake_export_stage_to_s3,
        ):
            task._upload_stages_to_s3(
                prepared_stages,
                backend,
                {"usd_path": str(tmp_path / "scene.usd")},
                listener,
            )

        assert len(calls) == 3
        assert all(call["bundle_mdl_assets"] is False for call in calls)

    @pytest.mark.asyncio
    async def test_upload_stages_to_s3_async_forwards_bundle_mdl_assets(self, tmp_path):
        """Async reusable stage upload must honor the remote renderer asset flag."""
        task = USDPrimTraversalAndRenderingTask()
        backend = RemoteRenderingBackend(
            api_key="test-api-key",
            use_data_uri=True,
            bundle_mdl_assets=False,
        )
        prepared_stages = {
            "prim_only": {
                "type": "prim_only",
                "data": (Mock(), ["/Camera"], 1),
            },
            "composition": {
                "type": "composition",
                "data": (
                    (Mock(), ["/HighlightCamera"], 1),
                    (Mock(), ["/PlainCamera"], 1),
                ),
            },
        }
        listener = Mock()
        calls: list[dict[str, Any]] = []

        def fake_export_stage_to_s3(stage, **kwargs):
            calls.append(kwargs)
            return "data:application/usd;base64,test", None

        with patch(
            "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
            fake_export_stage_to_s3,
        ):
            await task._upload_stages_to_s3_async(
                prepared_stages,
                backend,
                {"usd_path": str(tmp_path / "scene.usd")},
                listener,
            )

        assert len(calls) == 3
        assert all(call["bundle_mdl_assets"] is False for call in calls)

    @pytest.mark.asyncio
    async def test_process_batch_async_forwards_backend_options_to_composition_render(
        self, tmp_path
    ):
        """Remote composition batches must forward backend request options."""
        task = USDPrimTraversalAndRenderingTask()
        backend = RemoteRenderingBackend(
            api_key="test-api-key",
            retry_jitter=0.42,
            material_target="openpbr_materialx",
        )
        prepared_stages = {
            "composition": {
                "type": "composition",
                "config": RenderingConfig(image_width=16),
                "data": (
                    (Mock(), ["/Camera"], 1),
                    (Mock(), ["/Camera"], 1),
                ),
                "highlight_url": "data:application/usd;base64,highlight",
                "plain_url": "data:application/usd;base64,plain",
            },
        }
        calls: list[dict[str, Any]] = []

        async def fake_render_composition_from_url(**kwargs):
            calls.append(kwargs)
            return {"results": []}, {"results": []}

        with patch(
            "world_understanding.functions.graphics.render_remote_async.render_composition_from_url",
            fake_render_composition_from_url,
        ):
            await task._process_batch_async(
                0,
                1,
                ["/World/Mesh"],
                {},
                prepared_stages,
                backend,
                "composition",
                tmp_path,
                tmp_path,
                1,
                1,
                Mock(),
            )

        assert calls[0]["retry_jitter"] == 0.42
        assert calls[0]["material_target"] == "openpbr_materialx"

    @pytest.mark.asyncio
    async def test_process_batch_async_forwards_backend_options_to_prim_render(
        self, tmp_path
    ):
        """Remote prim batches must forward backend request options."""
        task = USDPrimTraversalAndRenderingTask()
        backend = RemoteRenderingBackend(
            api_key="test-api-key",
            retry_jitter=0.42,
            material_target="openpbr_materialx",
        )
        prepared_stages = {
            "prim_only": {
                "type": "prim_only",
                "config": RenderingConfig(image_width=16),
                "data": (Mock(), ["/Camera"], 1),
                "stage_url": "data:application/usd;base64,stage",
            },
        }
        calls: list[dict[str, Any]] = []

        async def fake_render_cameras_from_url(**kwargs):
            calls.append(kwargs)
            return {"results": []}

        with patch(
            "world_understanding.functions.graphics.render_remote_async.render_cameras_from_url",
            fake_render_cameras_from_url,
        ):
            await task._process_batch_async(
                0,
                1,
                ["/World/Mesh"],
                {},
                prepared_stages,
                backend,
                "prim_only",
                tmp_path,
                tmp_path,
                1,
                1,
                Mock(),
            )

        assert calls[0]["retry_jitter"] == 0.42
        assert calls[0]["material_target"] == "openpbr_materialx"

    def test_raises_when_majority_composition_renders_are_blank(self, tmp_path):
        """Dataset rendering must fail before VLM inference on mostly blank views."""
        task = USDPrimTraversalAndRenderingTask()
        blank_a = tmp_path / "a_composition.png"
        blank_b = tmp_path / "b_composition.png"
        nonblank = tmp_path / "c_composition.png"
        Image.new("RGB", (32, 32), (0, 0, 0)).save(blank_a)
        Image.new("RGB", (32, 32), (255, 255, 255)).save(blank_b)
        self._save_nonblank_image(nonblank)

        prim_data = [
            {
                "prim_path": "/World/A",
                "images": [
                    {
                        "path": blank_a.name,
                        "render_mode": "composition",
                        "view": "a",
                    },
                    {
                        "path": blank_b.name,
                        "render_mode": "composition",
                        "view": "b",
                    },
                    {
                        "path": nonblank.name,
                        "render_mode": "composition",
                        "view": "c",
                    },
                ],
            }
        ]

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                prim_data,
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context={},
            )

    def test_can_warn_on_majority_blank_renders_when_configured(self, tmp_path):
        """Low-signal geometry can continue when blank dataset renders are allowed."""
        task = USDPrimTraversalAndRenderingTask()
        blank_a = tmp_path / "a_composition.png"
        blank_b = tmp_path / "b_composition.png"
        nonblank = tmp_path / "c_composition.png"
        Image.new("RGB", (32, 32), (0, 0, 0)).save(blank_a)
        Image.new("RGB", (32, 32), (255, 255, 255)).save(blank_b)
        self._save_nonblank_image(nonblank)
        listener = Mock()
        context = {"fail_on_blank_dataset_renders": False}

        task._check_blank_dataset_renders(
            [
                {
                    "prim_path": "/World/A",
                    "images": [
                        {
                            "path": blank_a.name,
                            "render_mode": "composition",
                            "view": "a",
                        },
                        {
                            "path": blank_b.name,
                            "render_mode": "composition",
                            "view": "b",
                        },
                        {
                            "path": nonblank.name,
                            "render_mode": "composition",
                            "view": "c",
                        },
                    ],
                }
            ],
            tmp_path,
            rgb_modes=["composition"],
            sensor_modes=[],
            listener=listener,
            context=context,
        )

        listener.warning.assert_called_once()
        assert context["blank_render_checked_count"] == 3
        assert len(context["blank_renders"]) == 2

    def test_remote_blank_render_failures_count_without_image_payloads(self, tmp_path):
        """HTTP 422 blank_render frames must still trip the dataset guardrail."""
        task = USDPrimTraversalAndRenderingTask()
        context = {
            "failed_batches": [
                {
                    "blank_render": True,
                    "render_mode": "composition",
                    "prim_path": "/World/A",
                    "camera": "CornerViewCamera_posx",
                    "frame": 0,
                    "stats": {"blank": True, "reason": "remote_blank_render"},
                    "error": "1/1 OVRTX render frames are blank or near-blank.",
                }
            ]
        }

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                [],
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context=context,
            )

        assert context["blank_render_checked_count"] == 1
        assert context["blank_renders"][0]["prim_path"] == "/World/A"

    def test_overlapping_blank_failures_are_not_double_counted(self, tmp_path):
        """Remote blank metadata should not inflate counts for an existing render."""
        task = USDPrimTraversalAndRenderingTask()
        blank_path = tmp_path / "a_composition.png"
        Image.new("RGB", (16, 16), (0, 0, 0)).save(blank_path)
        context = {
            "failed_batches": [
                {
                    "blank_render": True,
                    "render_mode": "composition",
                    "prim_path": "/World/A",
                    "camera": "CornerViewCamera_posx",
                    "frame": 0,
                    "stats": {"blank": True, "reason": "remote_blank_render"},
                }
            ]
        }

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                [
                    {
                        "prim_path": "/World/A",
                        "images": [
                            {
                                "path": blank_path.name,
                                "render_mode": "composition",
                                "view": "a",
                            }
                        ],
                    }
                ],
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context=context,
            )

        assert context["blank_render_checked_count"] == 1
        assert len(context["blank_renders"]) == 1

    def test_overlapping_blank_failure_still_marks_candidate_key(self, tmp_path):
        """Overlapping remote blank metadata must still count as blank once."""
        task = USDPrimTraversalAndRenderingTask()
        nonblank_path = tmp_path / "a_composition.png"
        self._save_nonblank_image(nonblank_path)
        context = {
            "failed_batches": [
                {
                    "blank_render": True,
                    "render_mode": "composition",
                    "prim_path": "/World/A",
                    "camera": "CornerViewCamera_posx",
                    "frame": 0,
                    "stats": {"blank": True, "reason": "remote_blank_render"},
                }
            ]
        }

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                [
                    {
                        "prim_path": "/World/A",
                        "images": [
                            {
                                "path": nonblank_path.name,
                                "render_mode": "composition",
                                "view": "a",
                            }
                        ],
                    }
                ],
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context=context,
            )

        assert context["blank_render_checked_count"] == 1
        assert context["blank_renders"][0]["stats"]["reason"] == "remote_blank_render"

    def test_unreadable_render_counts_against_blank_threshold(self, tmp_path):
        """Corrupt images must not depress the blank-render failure ratio."""
        task = USDPrimTraversalAndRenderingTask()
        corrupt = tmp_path / "bad_composition.png"
        corrupt.write_text("not an image", encoding="utf-8")

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                [
                    {
                        "prim_path": "/World/Bad",
                        "images": [
                            {
                                "path": corrupt.name,
                                "render_mode": "composition",
                                "view": "bad",
                            }
                        ],
                    }
                ],
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context={},
            )

    def test_renderer_blank_stats_skip_disk_reanalysis(self, tmp_path):
        task = USDPrimTraversalAndRenderingTask()
        context: dict[str, Any] = {}

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                [
                    {
                        "prim_path": "/World/Blank",
                        "images": [
                            {
                                "path": "missing_composition.png",
                                "render_mode": "composition",
                                "view": "front",
                                "blank_render": True,
                                "stats": {"blank": True, "reason": "solid_color"},
                            }
                        ],
                    }
                ],
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context=context,
            )

        assert context["blank_renders"][0]["stats"]["reason"] == "solid_color"
        assert "analysis_error" not in context["blank_renders"][0]

    def test_blank_dataset_warning_path_does_not_raise(self, tmp_path):
        task = USDPrimTraversalAndRenderingTask()
        listener = Mock()
        context: dict[str, Any] = {}

        blank_path = tmp_path / "blank_composition.png"
        nonblank_path = tmp_path / "nonblank_composition.png"
        Image.new("RGB", (16, 16), (255, 255, 255)).save(blank_path)
        nonblank = Image.new("RGB", (16, 16), (255, 255, 255))
        draw = ImageDraw.Draw(nonblank)
        draw.rectangle([4, 4, 12, 12], fill=(0, 0, 0))
        nonblank.save(nonblank_path)

        task._check_blank_dataset_renders(
            [
                {
                    "prim_path": "/World/Blank",
                    "images": [
                        {
                            "path": blank_path.name,
                            "render_mode": "composition",
                            "view": "blank",
                        }
                    ],
                },
                {
                    "prim_path": "/World/Visible",
                    "images": [
                        {
                            "path": nonblank_path.name,
                            "render_mode": "composition",
                            "view": "visible",
                        }
                    ],
                },
            ],
            tmp_path,
            rgb_modes=["composition"],
            sensor_modes=[],
            listener=listener,
            context=context,
        )

        assert context["blank_render_checked_count"] == 2
        assert len(context["blank_renders"]) == 1
        listener.warning.assert_called_once()

    def test_blank_failures_match_selected_candidate_modes(self, tmp_path):
        task = USDPrimTraversalAndRenderingTask()
        listener = Mock()
        context: dict[str, Any] = {
            "failed_batches": [
                {
                    "blank_render": True,
                    "render_mode": "albedo",
                    "prim_path": "/World/A",
                    "stats": {"blank": True, "reason": "remote_blank_render"},
                }
            ]
        }
        nonblank_path = tmp_path / "a_composition.png"
        self._save_nonblank_image(nonblank_path)

        task._check_blank_dataset_renders(
            [
                {
                    "prim_path": "/World/A",
                    "images": [
                        {
                            "path": nonblank_path.name,
                            "render_mode": "composition",
                            "view": "a",
                        }
                    ],
                }
            ],
            tmp_path,
            rgb_modes=["composition", "albedo"],
            sensor_modes=[],
            listener=listener,
            context=context,
        )

        assert "blank_renders" not in context
        listener.warning.assert_not_called()

    def test_extracts_blank_render_failures_from_remote_result(self):
        task = USDPrimTraversalAndRenderingTask()

        failures = task._blank_render_failures_from_results(
            {
                "results": [
                    {
                        "camera": "Camera",
                        "status": RenderingStatus.blank_render,
                        "error": "blank",
                        "blank_render_frames": [
                            {
                                "frame": 4,
                                "stats": {"blank": True, "reason": "solid_color"},
                            }
                        ],
                    }
                ]
            },
            batch_start=4,
            batch_prims=["/World/A"],
            render_mode="composition",
        )

        assert failures == [
            {
                "batch_start": 4,
                "batch_prims": ["/World/A"],
                "render_mode": "composition",
                "camera": "Camera",
                "prim_path": "/World/A",
                "frame": 4,
                "blank_render": True,
                "stats": {"blank": True, "reason": "solid_color"},
                "error": "blank",
            }
        ]

    def test_blank_render_failure_without_frame_details_marks_batch_prims(self):
        task = USDPrimTraversalAndRenderingTask()

        failures = task._blank_render_failures_from_results(
            {
                "results": [
                    {
                        "camera": "Camera",
                        "status": "blank_render",
                        "error": "blank",
                        "blank_render_frames": [],
                    }
                ]
            },
            batch_start=7,
            batch_prims=["/World/A", "/World/B"],
            render_mode="composition",
        )

        assert [failure["prim_path"] for failure in failures] == [
            "/World/A",
            "/World/B",
        ]
        assert [failure["frame"] for failure in failures] == [7, 8]

    def test_success_blank_render_frames_do_not_create_failure_candidates(self):
        task = USDPrimTraversalAndRenderingTask()

        failures = task._blank_render_failures_from_results(
            {
                "results": [
                    {
                        "camera": "Camera",
                        "status": "success",
                        "blank_render_frames": [
                            {
                                "frame": 4,
                                "stats": {"blank": True, "reason": "solid_color"},
                            }
                        ],
                    }
                ]
            },
            batch_start=4,
            batch_prims=["/World/A"],
            render_mode="composition",
        )

        assert failures == []


class TestUSDPrimCollectionFilters:
    """Tests for USD prim collection filters."""

    def test_collect_prims_can_skip_invisible_meshes(self):
        """Hidden source/library meshes can be excluded from render datasets."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/World/VisibleMesh")
        hidden_parent = UsdGeom.Xform.Define(stage, "/World/HiddenLibrary")
        UsdGeom.Imageable(hidden_parent.GetPrim()).MakeInvisible()
        UsdGeom.Mesh.Define(stage, "/World/HiddenLibrary/HiddenMesh")

        task = USDPrimTraversalAndRenderingTask()
        prims = task._collect_prims(
            stage,
            {
                "types": ["UsdGeom.Mesh"],
                "skip_instances": False,
                "skip_invisible": True,
            },
            Mock(),
        )

        assert prims == ["/World/VisibleMesh"]

    def test_collect_prims_can_skip_invisible_specific_paths(self):
        """The skip_invisible filter also applies to explicit path lists."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/World/VisibleMesh")
        hidden_parent = UsdGeom.Xform.Define(stage, "/World/HiddenLibrary")
        UsdGeom.Imageable(hidden_parent.GetPrim()).MakeInvisible()
        UsdGeom.Mesh.Define(stage, "/World/HiddenLibrary/HiddenMesh")

        task = USDPrimTraversalAndRenderingTask()
        prims = task._collect_prims(
            stage,
            {
                "paths": [
                    "/World/VisibleMesh",
                    "/World/HiddenLibrary/HiddenMesh",
                ],
                "skip_instances": False,
                "skip_invisible": True,
            },
            Mock(),
        )

        assert prims == ["/World/VisibleMesh"]

    def test_collect_prims_falls_back_when_usdvol_module_missing(self):
        """UsdVol filters still match concrete typeNames without pxr.UsdVol."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        stage.DefinePrim("/World/Fog", "Volume")
        UsdGeom.Mesh.Define(stage, "/World/Mesh")

        real_import = builtins.__import__

        def import_without_usdvol(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "pxr" and fromlist and "UsdVol" in fromlist:
                raise ImportError("No module named 'pxr.UsdVol'")
            return real_import(name, globals, locals, fromlist, level)

        task = USDPrimTraversalAndRenderingTask()
        listener = Mock()

        with patch("builtins.__import__", side_effect=import_without_usdvol):
            prims = task._collect_prims(
                stage,
                {
                    "types": ["UsdVol.Volume"],
                    "skip_instances": False,
                },
                listener,
            )

        assert prims == ["/World/Fog"]
        listener.warning.assert_any_call(
            "pxr.UsdVol is not available from the active OpenUSD provider; "
            "falling back to exact typeName matching for 'UsdVol.Volume'."
        )


class TestUSDDatasetManifestZeroImages:
    """Tests that USDDatasetManifestTask fails when no images are in the dataset."""

    def test_raises_on_zero_images(self, tmp_path):
        """Task must raise RuntimeError when dataset has 0 images."""
        task = USDDatasetManifestTask()

        # prim_data with a prim that has no rendered images
        prim_data = [
            {
                "prim_path": "/World/Mesh",
                "images": {},
                "metadata": {"type": "UsdGeom.Mesh"},
            }
        ]

        context: dict = {
            "output_dir": str(tmp_path),
            "export_usd_model": False,
            "usd_path": "test.usd",
        }
        object_store = Mock(spec=ObjectStore)
        object_store.get.side_effect = lambda key, default=None: {
            "prim_data": prim_data,
            "usd_model": None,
            "usd_stage": None,
        }.get(key, default)

        with pytest.raises(RuntimeError, match="Dataset has 0 images"):
            task.run(context, object_store)


class TestPrimPathToDirectoryStructure:
    """Tests for prim path to filesystem path conversion."""

    def test_long_path_segments_are_bounded(self, tmp_path):
        """Long prim path segments should be truncated and still create a file path."""
        long_segment = "segment_" + ("x" * 500)
        prim_path = f"/World/{long_segment}/{long_segment}"
        output_path = prim_path_to_directory_structure(
            prim_path=prim_path,
            base_dir=tmp_path,
            filename="render.png",
        )

        assert output_path.parent.exists()
        assert output_path.name == "render.png"
        assert len(output_path.parent.name) <= MAX_PATH_COMPONENT_LEN
        assert len(output_path.parent.parent.name) <= MAX_PATH_COMPONENT_LEN
