# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for material_agent.scene.run module."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from material_agent.scene.config_gen import (
    generate_all_configs,
    generate_all_payload_configs,
    prepare_payload_runtime_config,
    prepare_sub_asset_runtime_config,
)
from material_agent.scene.manifest import (
    InstanceGroup,
    PayloadGroup,
    SceneManifest,
    SubAsset,
)
from material_agent.scene.run import (
    SubAssetHarnessConfig,
    _run_parallel,
    run_all,
    run_all_payloads_bottomup,
    run_payload,
    run_sub_asset,
    run_sub_asset_with_harness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sub_asset(
    name: str = "asset_a",
    prim_path: str = "/World/AssetA",
    config_path: str | None = "/tmp/cfg.yaml",
    status: str = "pending",
    mesh_count: int = 10,
    instance_group: str | None = None,
    asset_id: str | None = None,
) -> SubAsset:
    return SubAsset(
        id=asset_id or str(uuid.uuid4()),
        name=name,
        prim_path=prim_path,
        config_path=config_path,
        status=status,
        mesh_count=mesh_count,
        instance_group=instance_group,
    )


def _make_payload_group(
    group_name: str = "payload_a",
    config_path: str | None = "/tmp/pg_cfg.yaml",
    status: str = "pending",
    instance_count: int = 5,
    pg_id: str | None = None,
) -> PayloadGroup:
    return PayloadGroup(
        id=pg_id or str(uuid.uuid4()),
        group_name=group_name,
        payload_file="/tmp/payload.usd",
        config_path=config_path,
        status=status,
        instance_count=instance_count,
    )


@dataclass
class FakePipelineOutput:
    """Lightweight stand-in for PipelineOutput in tests."""

    success: bool
    error: str | None = None
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


def _write_config(path: Path, session_id: str = "test_session") -> None:
    """Write a minimal YAML config file."""
    cfg = {"project": {"session_id": session_id}, "steps": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))


def _scene_config_with_inline_api_key(secret: str) -> dict[str, Any]:
    return {
        "project": {"name": "scene"},
        "input": {"usd_path": "scene.usda"},
        "output": {"format": "usda"},
        "steps": {
            "predict": {
                "vlm": {
                    "backend": "nim",
                    "api_key": secret,
                }
            },
            "apply": {"enabled": False},
            "render": {"enabled": False},
            "restore_usd": {"enabled": False},
        },
    }


# ---------------------------------------------------------------------------
# run_sub_asset
# ---------------------------------------------------------------------------


class TestRunSubAsset:
    """Tests for run_sub_asset including SO retry logic."""

    def test_no_config_path_raises(self) -> None:
        sa = _make_sub_asset(config_path=None)
        with pytest.raises(ValueError, match="no config_path"):
            run_sub_asset(sa)

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        sa = _make_sub_asset(config_path=str(tmp_path / "nonexistent.yaml"))
        with pytest.raises(FileNotFoundError):
            run_sub_asset(sa)

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_success_no_retry(
        self,
        mock_run: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline succeeds on first try without SO — no retry."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=True,
            completed_steps=["build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 5}},
        )

        result = run_sub_asset(sa)
        assert result.status == "completed"
        assert mock_run.call_count == 1

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_so_retry_when_pipeline_failed_after_so(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SO ran but pipeline failed -> retry without SO."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        # First call: SO ran but pipeline failed
        fail_result = FakePipelineOutput(
            success=False,
            error="predict failed",
            completed_steps=["optimize_usd", "build_dataset_usd"],
        )
        # Second call (retry): succeeds
        success_result = FakePipelineOutput(
            success=True,
            completed_steps=["build_dataset_usd", "predict"],
        )
        mock_run.side_effect = [fail_result, success_result]

        result = run_sub_asset(sa)
        assert result.status == "completed"
        assert mock_run.call_count == 2
        mock_clean.assert_called_once()

        # Verify retry call has optimize_usd in skip_steps
        retry_call = mock_run.call_args_list[1]
        retry_input = retry_call[0][0]  # first positional arg = PipelineInput
        assert "optimize_usd" in retry_input.skip_steps

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_so_retry_when_zero_predictions(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SO ran, pipeline succeeded, but 0 predictions -> retry without SO."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        zero_pred = FakePipelineOutput(
            success=True,
            completed_steps=["optimize_usd", "build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 0}},
        )
        success_result = FakePipelineOutput(
            success=True,
            completed_steps=["build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 3}},
        )
        mock_run.side_effect = [zero_pred, success_result]

        result = run_sub_asset(sa)
        assert result.status == "completed"
        assert mock_run.call_count == 2
        mock_clean.assert_called_once()

    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_no_retry_when_non_so_step_fails(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline failed but SO was NOT in completed_steps -> no retry."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=False,
            error="build_dataset_usd failed",
            completed_steps=["build_dataset_usd"],
        )

        result = run_sub_asset(sa)
        assert result.status == "failed"
        assert mock_run.call_count == 1
        mock_clean.assert_not_called()

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_no_retry_when_so_succeeded_with_predictions(
        self,
        mock_run: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SO ran, pipeline succeeded, predictions > 0 -> no retry needed."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=True,
            completed_steps=["optimize_usd", "build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 5}},
        )

        result = run_sub_asset(sa)
        assert result.status == "completed"
        assert mock_run.call_count == 1

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_predict_max_workers_stays_in_runtime_config(
        self,
        mock_run: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        persisted = cfg_path.read_text()
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=True, completed_steps=["predict"]
        )

        run_sub_asset(sa, predict_max_workers=4)
        pipeline_input = mock_run.call_args.args[0]
        assert pipeline_input.config["steps"]["predict"]["max_workers"] == 4
        assert pipeline_input.config_path == cfg_path
        assert cfg_path.read_text() == persisted

    @patch("material_agent.api.pipeline.run_pipeline")
    def test_failed_pipeline_sets_status_failed(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=False, error="boom", completed_steps=[]
        )

        result = run_sub_asset(sa)
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# run_sub_asset_with_harness
# ---------------------------------------------------------------------------


class TestRunSubAssetWithHarness:
    """Tests for the retired sub-asset harness adapter."""

    def test_direct_call_reports_retired_workflow(self) -> None:
        with pytest.raises(RuntimeError, match="content-workflow-cli materials assign"):
            run_sub_asset_with_harness(
                _make_sub_asset(name="AssetA", asset_id="asset_a"),
                SubAssetHarnessConfig(enabled=True),
            )

    def test_direct_call_accepts_legacy_kwargs_before_retirement_error(self) -> None:
        with pytest.raises(RuntimeError, match="content-workflow-cli materials assign"):
            run_sub_asset_with_harness(
                _make_sub_asset(name="AssetA", asset_id="asset_a"),
                SubAssetHarnessConfig(enabled=True),
                verbose=True,
                predict_max_workers=2,
                cancel_checker=lambda: False,
            )

    def test_run_all_harness_config_writes_retired_error_sidecar(
        self,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "asset.yaml"
        _write_config(cfg_path, session_id="asset_a")
        sub_asset = _make_sub_asset(
            name="AssetA",
            config_path=str(cfg_path),
            asset_id="asset_a",
        )
        manifest = SceneManifest(sub_assets=[sub_asset])
        manifest_path = tmp_path / "manifest.json"

        run_all(
            manifest,
            manifest_path,
            harness_config=SubAssetHarnessConfig(enabled=True),
        )

        error_path = tmp_path / ".asset_a/harness_scene_adapter_error.json"
        assert sub_asset.status == "failed"
        assert error_path.exists()
        error = yaml.safe_load(error_path.read_text())
        assert error == {
            "schema": "world-understanding-durable-diagnostic-v1",
            "code": "material_scene_adapter_failed",
            "phase": "pipeline_execution",
            "retryable": False,
        }
        assert "content-workflow-cli materials assign" not in error_path.read_text()


# ---------------------------------------------------------------------------
# _run_parallel
# ---------------------------------------------------------------------------


class TestRunParallel:
    """Tests for _run_parallel: thread-safe saves, error handling, sorting."""

    @patch("material_agent.scene.run._run_sub_asset_worker")
    def test_manifest_saved_per_completion(
        self, mock_worker: MagicMock, tmp_path: Path
    ) -> None:
        """Manifest is saved once per completed future."""
        sa1 = _make_sub_asset(name="a1", asset_id="id1", mesh_count=20)
        sa2 = _make_sub_asset(name="a2", asset_id="id2", mesh_count=10)

        manifest = SceneManifest(sub_assets=[sa1, sa2])
        manifest_path = tmp_path / "manifest.json"

        # Workers return updated copies
        def worker_side_effect(sa, *args, **kwargs):
            sa.status = "completed"
            return sa

        mock_worker.side_effect = worker_side_effect

        manifest.save = MagicMock()  # type: ignore[method-assign]

        completed, failed = _run_parallel(
            [sa1, sa2],
            manifest,
            manifest_path,
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=2,
        )

        assert completed == 2
        assert failed == 0
        assert manifest.save.call_count == 2

    @patch("material_agent.scene.run._run_sub_asset_worker")
    def test_worker_exception_counts_as_failed(
        self, mock_worker: MagicMock, tmp_path: Path
    ) -> None:
        """If the worker future raises, the asset is marked failed."""
        sa = _make_sub_asset(name="boom", asset_id="id_boom", mesh_count=5)
        manifest = SceneManifest(sub_assets=[sa])
        manifest_path = tmp_path / "manifest.json"
        manifest.save = MagicMock()  # type: ignore[method-assign]

        mock_worker.side_effect = RuntimeError("worker crashed")

        completed, failed = _run_parallel(
            [sa],
            manifest,
            manifest_path,
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=1,
        )

        assert completed == 0
        assert failed == 1
        assert manifest.sub_assets[0].status == "failed"

    @patch("material_agent.scene.run._run_sub_asset_worker")
    def test_largest_first_sorting(
        self, mock_worker: MagicMock, tmp_path: Path
    ) -> None:
        """Assets should be submitted largest mesh_count first."""
        sa_small = _make_sub_asset(name="small", asset_id="id_s", mesh_count=5)
        sa_large = _make_sub_asset(name="large", asset_id="id_l", mesh_count=100)

        manifest = SceneManifest(sub_assets=[sa_small, sa_large])
        manifest_path = tmp_path / "manifest.json"
        manifest.save = MagicMock()  # type: ignore[method-assign]

        call_order: list[str] = []

        def worker_side_effect(sa, *args, **kwargs):
            call_order.append(sa.name)
            sa.status = "completed"
            return sa

        mock_worker.side_effect = worker_side_effect

        # Use max_workers=1 to force sequential submission order
        _run_parallel(
            [sa_small, sa_large],
            manifest,
            manifest_path,
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=1,
        )

        assert call_order[0] == "large"
        assert call_order[1] == "small"


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    """Tests for run_all: skip completed, sequential vs parallel routing."""

    @patch("material_agent.scene.run._run_sequential")
    def test_skip_completed_assets(self, mock_seq: MagicMock, tmp_path: Path) -> None:
        sa_done = _make_sub_asset(
            name="done", config_path="/tmp/c.yaml", status="completed"
        )
        sa_pending = _make_sub_asset(
            name="pending", config_path="/tmp/c2.yaml", status="pending"
        )

        manifest = SceneManifest(sub_assets=[sa_done, sa_pending])
        manifest_path = tmp_path / "manifest.json"

        mock_seq.return_value = (1, 0)

        run_all(manifest, manifest_path, skip_existing=True)

        # Only the pending asset should be passed to _run_sequential
        args = mock_seq.call_args
        to_process = args[0][0]
        assert len(to_process) == 1
        assert to_process[0].name == "pending"

    @patch("material_agent.scene.run._run_parallel")
    def test_parallel_routing_when_workers_gt_1(
        self, mock_par: MagicMock, tmp_path: Path
    ) -> None:
        sa = _make_sub_asset(config_path="/tmp/c.yaml")
        manifest = SceneManifest(sub_assets=[sa])
        manifest_path = tmp_path / "manifest.json"

        mock_par.return_value = (1, 0)

        run_all(manifest, manifest_path, max_workers=4)

        mock_par.assert_called_once()

    @patch("material_agent.scene.run._run_sequential")
    def test_sequential_routing_when_workers_eq_1(
        self, mock_seq: MagicMock, tmp_path: Path
    ) -> None:
        sa = _make_sub_asset(config_path="/tmp/c.yaml")
        manifest = SceneManifest(sub_assets=[sa])
        manifest_path = tmp_path / "manifest.json"

        mock_seq.return_value = (1, 0)

        run_all(manifest, manifest_path, max_workers=1)

        mock_seq.assert_called_once()

    @patch("material_agent.scene.run._run_sequential")
    def test_emits_initial_progress_before_processing(
        self, mock_seq: MagicMock, tmp_path: Path
    ) -> None:
        assets = [
            _make_sub_asset(name="a", config_path="/tmp/a.yaml"),
            _make_sub_asset(name="b", config_path="/tmp/b.yaml"),
        ]
        manifest = SceneManifest(sub_assets=assets)
        manifest_path = tmp_path / "manifest.json"
        progress_events: list[dict[str, Any]] = []
        mock_seq.return_value = (2, 0)

        run_all(
            manifest,
            manifest_path,
            progress_callback=progress_events.append,
        )

        assert progress_events
        assert progress_events[0] == {
            "current": 0,
            "total": 2,
            "completed": 0,
            "failed": 0,
            "asset_id": None,
            "asset_name": None,
            "asset_status": "pending",
            "message": "Processing 2 scene assets",
        }

    def test_returns_immediately_when_nothing_to_process(self, tmp_path: Path) -> None:
        manifest = SceneManifest(sub_assets=[])
        manifest_path = tmp_path / "manifest.json"

        result = run_all(manifest, manifest_path)
        assert result is manifest

    @patch("material_agent.scene.run._run_sequential")
    def test_skips_assets_without_config_path(
        self, mock_seq: MagicMock, tmp_path: Path
    ) -> None:
        sa_no_cfg = _make_sub_asset(name="no_cfg", config_path=None)
        sa_with_cfg = _make_sub_asset(name="has_cfg", config_path="/tmp/c.yaml")
        manifest = SceneManifest(sub_assets=[sa_no_cfg, sa_with_cfg])
        manifest_path = tmp_path / "manifest.json"

        mock_seq.return_value = (1, 0)

        run_all(manifest, manifest_path)

        to_process = mock_seq.call_args[0][0]
        assert len(to_process) == 1
        assert to_process[0].name == "has_cfg"


# ---------------------------------------------------------------------------
# run_payload
# ---------------------------------------------------------------------------


class TestRunPayload:
    """Tests for run_payload: SO retry for payload groups."""

    def test_no_config_path_raises(self) -> None:
        pg = _make_payload_group(config_path=None)
        with pytest.raises(ValueError, match="no config_path"):
            run_payload(pg)

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        pg = _make_payload_group(config_path=str(tmp_path / "nope.yaml"))
        with pytest.raises(FileNotFoundError):
            run_payload(pg)

    @patch("material_agent.scene.run._update_payload_output_paths")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_success_no_retry(
        self,
        mock_run: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "pg.yaml"
        _write_config(cfg_path)
        pg = _make_payload_group(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=True,
            completed_steps=["build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 5}},
        )

        result = run_payload(pg)
        assert result.status == "completed"
        assert mock_run.call_count == 1

    @patch("material_agent.scene.run._update_payload_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_so_retry_on_failure_after_so(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "pg.yaml"
        _write_config(cfg_path)
        pg = _make_payload_group(config_path=str(cfg_path))

        fail = FakePipelineOutput(
            success=False,
            error="predict boom",
            completed_steps=["optimize_usd"],
        )
        ok = FakePipelineOutput(
            success=True, completed_steps=["build_dataset_usd", "predict"]
        )
        mock_run.side_effect = [fail, ok]

        result = run_payload(pg)
        assert result.status == "completed"
        assert mock_run.call_count == 2
        mock_clean.assert_called_once()

    @patch("material_agent.scene.run._update_payload_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_so_retry_on_zero_predictions(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "pg.yaml"
        _write_config(cfg_path)
        pg = _make_payload_group(config_path=str(cfg_path))

        zero = FakePipelineOutput(
            success=True,
            completed_steps=["optimize_usd", "predict"],
            step_results={"predict": {"predictions_count": 0}},
        )
        ok = FakePipelineOutput(
            success=True,
            completed_steps=["predict"],
            step_results={"predict": {"predictions_count": 2}},
        )
        mock_run.side_effect = [zero, ok]

        result = run_payload(pg)
        assert result.status == "completed"
        assert mock_run.call_count == 2

    @patch("material_agent.api.pipeline.run_pipeline")
    def test_no_retry_when_no_so(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No SO in completed_steps + failure -> no retry."""
        cfg_path = tmp_path / "pg.yaml"
        _write_config(cfg_path)
        pg = _make_payload_group(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=False,
            error="build failed",
            completed_steps=["build_dataset_usd"],
        )

        result = run_payload(pg)
        assert result.status == "failed"
        assert mock_run.call_count == 1


class TestSceneRuntimeCredentialTransport:
    def test_per_item_runtime_configs_exclude_scene_analyze(
        self, tmp_path: Path
    ) -> None:
        """run-agent never executes scene analyze; that stage is absent here."""
        secret = "nvapi-live-scene-sentinel-713"
        scene_config = _scene_config_with_inline_api_key(secret)
        scene_config["scene"] = {
            "analyze": {
                "llm": {
                    "backend": "nim",
                    "api_key": "nvapi-scene-analyze-sentinel-713",
                }
            }
        }
        asset = SubAsset(id="asset", name="asset", prim_path="/World/Asset")
        payload = PayloadGroup(
            id="payload",
            group_name="payload",
            payload_file=str(tmp_path / "payload.usda"),
        )
        manifest = SceneManifest(sub_assets=[asset], payload_groups=[payload])
        generate_all_configs(manifest, scene_config, tmp_path / "configs")
        generate_all_payload_configs(manifest, scene_config, tmp_path / "configs")

        asset_runtime = prepare_sub_asset_runtime_config(asset, scene_config)
        payload_runtime = prepare_payload_runtime_config(payload, scene_config)

        assert "scene" not in asset_runtime
        assert "scene" not in payload_runtime

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_extracted_scene_simulate_rehydrates_then_drops_inline_key(
        self,
        max_workers: int,
        tmp_path: Path,
    ) -> None:
        secret = "nvapi-live-scene-sentinel-713"
        scene_config = _scene_config_with_inline_api_key(secret)
        original_config = deepcopy(scene_config)
        manifest = SceneManifest(
            sub_assets=[
                SubAsset(
                    id="asset-a",
                    name="asset-a",
                    prim_path="/World/A",
                    status="extracted",
                ),
                SubAsset(
                    id="asset-b",
                    name="asset-b",
                    prim_path="/World/B",
                    status="extracted",
                ),
            ]
        )
        generate_all_configs(manifest, scene_config, tmp_path / "configs")
        generated_paths = [
            Path(asset.config_path or "") for asset in manifest.sub_assets
        ]
        assert all(secret not in path.read_text() for path in generated_paths)
        assert all(
            asset.config_credential_paths == ["steps.predict.vlm.api_key"]
            for asset in manifest.sub_assets
        )

        execution_configs: list[dict[str, Any]] = []

        def capture_simulation_config(*_args: Any, **kwargs: Any) -> FakePipelineOutput:
            execution_configs.append(kwargs["config_dict"])
            return FakePipelineOutput(success=True, completed_steps=["predict"])

        with (
            patch(
                "material_agent.scene.run._run_simulate",
                side_effect=capture_simulation_config,
            ),
            patch("material_agent.scene.run._update_output_paths"),
        ):
            run_all(
                manifest,
                tmp_path / "manifest.json",
                max_workers=max_workers,
                simulate=True,
                material_names=["Steel"],
                scene_config=scene_config,
            )

        assert [asset.status for asset in manifest.sub_assets] == [
            "completed",
            "completed",
        ]
        assert len(execution_configs) == 2
        for execution_config in execution_configs:
            vlm = execution_config["steps"]["predict"]["vlm"]
            assert vlm["backend"] == "mock"
            assert vlm["api_key"] == "not-used"
            assert secret not in repr(execution_config)
        assert secret not in (tmp_path / "manifest.json").read_text()
        assert scene_config == original_config

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_sub_asset_rehydration_failure_is_item_local_and_value_free(
        self,
        max_workers: int,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret = "nvapi-live-scene-sentinel-713"
        scene_config = _scene_config_with_inline_api_key(secret)
        manifest = SceneManifest(
            sub_assets=[
                SubAsset(id="bad", name="bad", prim_path="/World/Bad"),
                SubAsset(id="good", name="good", prim_path="/World/Good"),
            ]
        )
        generate_all_configs(manifest, scene_config, tmp_path / "configs")
        manifest.sub_assets[0].config_credential_paths.append(
            "steps.predict.vlm.client_secret"
        )
        caplog.set_level(logging.ERROR, logger="material_agent.scene.run")

        with (
            patch("material_agent.api.pipeline.run_pipeline") as mock_run,
            patch("material_agent.scene.run._update_output_paths"),
        ):
            mock_run.return_value = FakePipelineOutput(
                success=True,
                completed_steps=["predict"],
            )
            run_all(
                manifest,
                tmp_path / "manifest.json",
                max_workers=max_workers,
                scene_config=scene_config,
            )

        status_by_id = {asset.id: asset.status for asset in manifest.sub_assets}
        assert status_by_id == {"bad": "failed", "good": "completed"}
        mock_run.assert_called_once()
        assert (
            mock_run.call_args.args[0].config["steps"]["predict"]["vlm"]["api_key"]
            == secret
        )
        preparation_records = [
            record
            for record in caplog.records
            if "Unable to prepare an in-memory scene configuration" in record.message
        ]
        assert len(preparation_records) == 1
        assert preparation_records[0].exc_info is None
        assert secret not in caplog.text

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_simulated_sub_asset_failure_does_not_publish_exception_graph(
        self,
        max_workers: int,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret = "nvapi-live-scene-sentinel-713"
        scene_config = _scene_config_with_inline_api_key(secret)
        asset = SubAsset(id="asset", name="asset", prim_path="/World/Asset")
        manifest = SceneManifest(sub_assets=[asset])
        generate_all_configs(manifest, scene_config, tmp_path / "configs")
        caplog.set_level(logging.ERROR, logger="material_agent.scene.run")

        with patch(
            "material_agent.scene.run._run_simulate",
            side_effect=RuntimeError(f"api_key={secret}"),
        ):
            run_all(
                manifest,
                tmp_path / "manifest.json",
                max_workers=max_workers,
                simulate=True,
                material_names=["Steel"],
                scene_config=scene_config,
            )

        assert asset.status == "failed"
        failure_records = [
            record
            for record in caplog.records
            if "Unable to process one scene sub-asset" in record.message
        ]
        assert len(failure_records) == 1
        assert failure_records[0].exc_info is None
        assert secret not in caplog.text

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_extracted_payload_simulate_rehydrates_then_drops_inline_key(
        self,
        max_workers: int,
        tmp_path: Path,
    ) -> None:
        secret = "nvapi-live-payload-sentinel-713"
        scene_config = _scene_config_with_inline_api_key(secret)
        original_config = deepcopy(scene_config)
        payloads = [
            PayloadGroup(
                id=f"payload-{index}",
                group_name=f"payload-{index}",
                payload_file=str(tmp_path / f"payload-{index}.usda"),
            )
            for index in range(2)
        ]
        manifest = SceneManifest(payload_groups=payloads)
        generate_all_payload_configs(manifest, scene_config, tmp_path / "configs")
        generated_paths = [
            Path(payload.config_path or "") for payload in manifest.payload_groups
        ]
        assert all(secret not in path.read_text() for path in generated_paths)

        execution_configs: list[dict[str, Any]] = []

        def capture_simulation_config(*_args: Any, **kwargs: Any) -> FakePipelineOutput:
            execution_configs.append(kwargs["config_dict"])
            return FakePipelineOutput(success=True, completed_steps=["predict"])

        with (
            patch(
                "material_agent.scene.run._run_simulate",
                side_effect=capture_simulation_config,
            ),
            patch("material_agent.scene.run._update_payload_output_paths"),
            patch("material_agent.scene.run._set_payload_output_usd"),
            patch("material_agent.scene.run._fix_output_material_scope"),
        ):
            run_all_payloads_bottomup(
                manifest,
                tmp_path / "manifest.json",
                scene_config,
                tmp_path / "configs",
                max_workers=max_workers,
                simulate=True,
                material_names=["Steel"],
            )

        assert [payload.status for payload in manifest.payload_groups] == [
            "completed",
            "completed",
        ]
        assert len(execution_configs) == 2
        for execution_config in execution_configs:
            vlm = execution_config["steps"]["predict"]["vlm"]
            assert vlm["backend"] == "mock"
            assert vlm["api_key"] == "not-used"
            assert secret not in repr(execution_config)
        assert secret not in (tmp_path / "manifest.json").read_text()
        assert scene_config == original_config

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_payload_rehydration_failure_is_item_local_and_value_free(
        self,
        max_workers: int,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret = "nvapi-live-payload-sentinel-713"
        scene_config = _scene_config_with_inline_api_key(secret)
        payloads = [
            PayloadGroup(
                id="bad",
                group_name="bad",
                payload_file=str(tmp_path / "bad.usda"),
            ),
            PayloadGroup(
                id="good",
                group_name="good",
                payload_file=str(tmp_path / "good.usda"),
            ),
        ]
        manifest = SceneManifest(payload_groups=payloads)
        generate_all_payload_configs(manifest, scene_config, tmp_path / "configs")
        manifest.payload_groups[0].config_credential_paths.append(
            "steps.predict.vlm.client_secret"
        )
        caplog.set_level(logging.ERROR, logger="material_agent.scene.run")

        with (
            patch("material_agent.api.pipeline.run_pipeline") as mock_run,
            patch("material_agent.scene.run._update_payload_output_paths"),
            patch("material_agent.scene.run._set_payload_output_usd"),
            patch("material_agent.scene.run._fix_output_material_scope"),
        ):
            mock_run.return_value = FakePipelineOutput(
                success=True,
                completed_steps=["predict"],
            )
            run_all_payloads_bottomup(
                manifest,
                tmp_path / "manifest.json",
                scene_config,
                tmp_path / "configs",
                max_workers=max_workers,
            )

        status_by_id = {
            payload.id: payload.status for payload in manifest.payload_groups
        }
        assert status_by_id == {"bad": "failed", "good": "completed"}
        mock_run.assert_called_once()
        assert (
            mock_run.call_args.args[0].config["steps"]["predict"]["vlm"]["api_key"]
            == secret
        )
        preparation_records = [
            record
            for record in caplog.records
            if "Unable to prepare an in-memory scene configuration" in record.message
        ]
        assert len(preparation_records) == 1
        assert preparation_records[0].exc_info is None
        assert secret not in caplog.text

    @pytest.mark.parametrize("max_workers", [1, 2])
    def test_simulated_payload_failure_does_not_publish_exception_graph(
        self,
        max_workers: int,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        secret = "nvapi-live-payload-sentinel-713"
        scene_config = _scene_config_with_inline_api_key(secret)
        payload = PayloadGroup(
            id="payload",
            group_name="payload",
            payload_file=str(tmp_path / "payload.usda"),
        )
        manifest = SceneManifest(payload_groups=[payload])
        generate_all_payload_configs(manifest, scene_config, tmp_path / "configs")
        caplog.set_level(logging.ERROR, logger="material_agent.scene.run")

        with patch(
            "material_agent.scene.run._run_simulate",
            side_effect=RuntimeError(f"api_key={secret}"),
        ):
            run_all_payloads_bottomup(
                manifest,
                tmp_path / "manifest.json",
                scene_config,
                tmp_path / "configs",
                max_workers=max_workers,
                simulate=True,
                material_names=["Steel"],
            )

        assert payload.status == "failed"
        failure_records = [
            record
            for record in caplog.records
            if "Unable to process one scene payload" in record.message
        ]
        assert len(failure_records) == 1
        assert failure_records[0].exc_info is None
        assert secret not in caplog.text

    @patch("material_agent.api.pipeline.run_pipeline")
    def test_concurrent_scene_runs_keep_distinct_source_credentials_isolated(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        overlap_barrier = Barrier(2)

        def run_pipeline_with_overlap(
            *_args: Any, **_kwargs: Any
        ) -> FakePipelineOutput:
            overlap_barrier.wait(timeout=5)
            return FakePipelineOutput(
                success=True,
                completed_steps=["predict"],
                step_results={"predict": {"predictions_count": 1}},
            )

        mock_run.side_effect = run_pipeline_with_overlap
        run_inputs: list[tuple[SceneManifest, Path, dict[str, Any]]] = []
        for name, credential in (
            ("alpha", {"api_key": "q7Z9"}),
            ("beta", {"providers": [{"token": "m4P8"}]}),
        ):
            scene_config = {
                "project": {"name": name},
                "input": {"usd_path": f"{name}.usd"},
                "steps": {"predict": {"vlm": credential}},
            }
            manifest = SceneManifest(
                sub_assets=[
                    SubAsset(
                        id=name,
                        name=name,
                        prim_path=f"/World/{name}",
                    )
                ]
            )
            run_root = tmp_path / name
            generate_all_configs(manifest, scene_config, run_root / "configs")
            run_inputs.append((manifest, run_root / "manifest.json", scene_config))

        def execute_scene(
            run_input: tuple[SceneManifest, Path, dict[str, Any]],
        ) -> SceneManifest:
            manifest, manifest_path, scene_config = run_input
            return run_all(
                manifest,
                manifest_path,
                scene_config=scene_config,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(execute_scene, run_inputs))

        assert all(result.sub_assets[0].status == "completed" for result in results)
        captured_by_session = {
            call_item.args[0].config["project"]["session_id"]: call_item.args[0].config
            for call_item in mock_run.call_args_list
        }
        assert captured_by_session["alpha"]["steps"]["predict"]["vlm"] == {
            "api_key": "q7Z9"
        }
        assert captured_by_session["beta"]["steps"]["predict"]["vlm"] == {
            "providers": [{"token": "m4P8"}]
        }
        assert "m4P8" not in str(captured_by_session["alpha"])
        assert "q7Z9" not in str(captured_by_session["beta"])

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_retry_receives_in_memory_secret_without_mutating_caller(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "asset.yaml"
        _write_config(config_path)
        persisted = config_path.read_text()
        sub_asset = _make_sub_asset(config_path=str(config_path))
        sub_asset.config_credential_paths = ["steps.predict.vlm.api_key"]
        runtime_config = {
            "project": {"session_id": "test_session"},
            "steps": {"predict": {"vlm": {"api_key": "q7Z9"}}},
        }
        failed = FakePipelineOutput(
            success=False,
            error="predict failed",
            completed_steps=["optimize_usd"],
        )
        succeeded = FakePipelineOutput(success=True, completed_steps=["predict"])
        mock_run.side_effect = [failed, succeeded]

        result = run_sub_asset(
            sub_asset,
            config_dict=runtime_config,
            predict_max_workers=3,
        )

        assert result.status == "completed"
        assert "max_workers" not in runtime_config["steps"]["predict"]
        assert config_path.read_text() == persisted
        assert mock_run.call_count == 2
        for pipeline_call in mock_run.call_args_list:
            pipeline_input = pipeline_call.args[0]
            assert pipeline_input.config["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"
            assert pipeline_input.config["steps"]["predict"]["max_workers"] == 3
            assert pipeline_input.config_path == config_path

    @patch("material_agent.api.pipeline.run_pipeline")
    def test_direct_asset_runner_scrubs_legacy_yaml_and_fails_closed(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "legacy_asset.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "project": {"session_id": "legacy"},
                    "steps": {"predict": {"vlm": {"api_key": "q7Z9"}}},
                }
            )
        )
        sub_asset = _make_sub_asset(config_path=str(config_path))

        with pytest.raises(ValueError) as exc_info:
            run_sub_asset(sub_asset)

        assert "steps.predict.vlm.api_key" in str(exc_info.value)
        assert "q7Z9" not in str(exc_info.value)
        assert "q7Z9" not in config_path.read_text()
        mock_run.assert_not_called()

    @patch("material_agent.api.pipeline.run_pipeline")
    def test_direct_payload_runner_scrubs_nested_legacy_secret_and_fails_closed(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "legacy_payload.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "project": {"session_id": "legacy"},
                    "steps": {"predict": {"vlm": {"providers": [{"token": "m4P8"}]}}},
                }
            )
        )
        payload_group = _make_payload_group(config_path=str(config_path))

        with pytest.raises(ValueError) as exc_info:
            run_payload(payload_group)

        assert "steps.predict.vlm.providers[0].token" in str(exc_info.value)
        assert "m4P8" not in str(exc_info.value)
        assert "m4P8" not in config_path.read_text()
        mock_run.assert_not_called()

    @patch("material_agent.api.pipeline.run_pipeline")
    def test_direct_asset_runner_rejects_incomplete_runtime_credentials(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "asset.yaml"
        _write_config(config_path)
        sub_asset = _make_sub_asset(config_path=str(config_path))
        sub_asset.config_credential_paths = ["steps.predict.vlm.api_key"]

        with pytest.raises(ValueError) as exc_info:
            run_sub_asset(
                sub_asset,
                config_dict={"project": {"session_id": "test"}, "steps": {}},
            )

        assert "steps.predict.vlm.api_key" in str(exc_info.value)
        mock_run.assert_not_called()

    @patch("material_agent.scene.run._update_payload_output_paths")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_direct_payload_runner_validates_and_forwards_runtime_credentials(
        self,
        mock_run: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "payload.yaml"
        _write_config(config_path)
        persisted = config_path.read_text()
        payload = _make_payload_group(config_path=str(config_path))
        payload.config_credential_paths = ["steps.predict.vlm.api_key"]

        with pytest.raises(ValueError) as exc_info:
            run_payload(
                payload,
                config_dict={"project": {"session_id": "test"}, "steps": {}},
            )
        assert "steps.predict.vlm.api_key" in str(exc_info.value)
        mock_run.assert_not_called()

        runtime_config = {
            "project": {"session_id": "test"},
            "steps": {"predict": {"vlm": {"api_key": "q7Z9"}}},
        }
        mock_run.return_value = FakePipelineOutput(
            success=True,
            completed_steps=["predict"],
        )

        result = run_payload(
            payload,
            config_dict=runtime_config,
            predict_max_workers=3,
        )

        assert result.status == "completed"
        pipeline_input = mock_run.call_args.args[0]
        assert pipeline_input.config["steps"]["predict"]["vlm"]["api_key"] == "q7Z9"
        assert pipeline_input.config["steps"]["predict"]["max_workers"] == 3
        assert "max_workers" not in runtime_config["steps"]["predict"]
        assert config_path.read_text() == persisted
        assert mock_run.call_count == 1
