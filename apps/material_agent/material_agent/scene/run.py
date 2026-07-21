# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline runner for large-scene sub-assets.

Calls the existing run_pipeline(PipelineInput(...)) for each sub-asset,
updating the manifest after each completion for resume safety.
Supports parallel execution via ThreadPoolExecutor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from world_understanding.agentic.base_pipeline_executor import save_pipeline_checkpoint
from world_understanding.agentic.config import clone_config_containers
from world_understanding.utils.credentials import find_inline_secret_paths
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    durable_diagnostic,
)

if TYPE_CHECKING:  # pragma: no cover
    from material_agent.api.pipeline import PipelineOutput

from .manifest import PayloadGroup, SceneManifest, SubAsset

logger = logging.getLogger(__name__)

CancelChecker = Callable[[], bool]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class SubAssetHarnessConfig:
    """Retired compatibility settings for the sub-asset harness path.

    The Material Agent task-first harness workflow has been retired. The
    supported agentic asset workflow is now ``content-workflow-cli`` with Content
    Workbench.
    """

    enabled: bool = False
    render_mode: Literal["remote"] = "remote"
    inference_mode: Literal["vlm"] = "vlm"
    vlm_backend: str = "nim"
    vlm_model: str = "qwen/qwen3.5-397b-a17b"
    max_iterations: int = 1
    max_rendered_prims: int | None = None
    apply_source_mode: Literal["copy", "overlay"] = "copy"
    expected_min_unique_materials: int | None = None
    render_width: int | None = None
    render_height: int | None = None
    dataset_render_batch_size: int | None = None
    dataset_render_max_concurrent_requests: int | None = None
    dataset_render_num_workers: int | None = None
    fail_on_blank_dataset_renders: bool = False
    fail_on_missing_prim_images: bool = False
    scene_render_width: int = 1024
    scene_render_height: int = 1024
    optimize_usd: bool = True
    restore_usd: bool = True
    generate_reference_images: bool = False
    optimizer_backend: Literal["local", "remote"] | None = None
    optimizer_config: dict[str, Any] | None = field(default_factory=dict)
    harness_request: str | None = None
    harness_metadata: dict[str, Any] = field(default_factory=dict)


def _raise_if_cancelled(cancel_checker: CancelChecker | None) -> None:
    """Raise ``CancelledError`` when the caller requests cancellation."""
    if cancel_checker and cancel_checker():
        raise asyncio.CancelledError("Scene pipeline cancellation requested")


def _prepare_sub_asset_runtime_config_for_run(
    sub_asset: SubAsset,
    scene_config: dict[str, Any] | None,
    scene_config_dir: Path | None,
) -> tuple[dict[str, Any] | None, bool]:
    """Prepare one runtime config without publishing rejected exception graphs."""
    if scene_config is None or not (
        sub_asset.config_credential_paths or find_inline_secret_paths(scene_config)
    ):
        return None, True

    runtime_config: dict[str, Any] | None = None
    preparation_failed = False
    try:
        from .config_gen import prepare_sub_asset_runtime_config

        runtime_config = prepare_sub_asset_runtime_config(
            sub_asset,
            scene_config,
            scene_config_dir=scene_config_dir,
        )
    except Exception:
        # Rehydration frames may retain the source credential. Consume the
        # exception here without binding, logging, or returning it; report only
        # a fixed failure after leaving the exception suite.
        preparation_failed = True

    if preparation_failed:
        logger.error(
            "Unable to prepare an in-memory scene configuration; "
            "marking one sub-asset failed"
        )
        return None, False
    return runtime_config, True


def _prepare_payload_runtime_config_for_run(
    payload_group: PayloadGroup,
    scene_config: dict[str, Any] | None,
    scene_config_dir: Path | None,
    sibling_names: list[str] | None,
) -> tuple[dict[str, Any] | None, bool]:
    """Prepare one payload runtime config behind the same safe boundary."""
    if scene_config is None or not (
        payload_group.config_credential_paths or find_inline_secret_paths(scene_config)
    ):
        return None, True

    runtime_config: dict[str, Any] | None = None
    preparation_failed = False
    try:
        from .config_gen import prepare_payload_runtime_config

        runtime_config = prepare_payload_runtime_config(
            payload_group,
            scene_config,
            scene_config_dir=scene_config_dir,
            sibling_names=sibling_names,
        )
    except Exception:
        preparation_failed = True

    if preparation_failed:
        logger.error(
            "Unable to prepare an in-memory scene configuration; "
            "marking one payload failed"
        )
        return None, False
    return runtime_config, True


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def _sub_asset_working_dir(config: dict[str, Any], config_path: Path) -> Path:
    project = config.get("project", {})
    if not isinstance(project, dict):
        project = {}

    configured = project.get("working_dir")
    if isinstance(configured, str) and configured:
        working_dir = Path(configured)
        if not working_dir.is_absolute():
            working_dir = config_path.parent / working_dir
        return working_dir.resolve()

    session_id = str(project.get("session_id") or "").strip()
    if not session_id:
        import re

        session_id = re.sub(r"[^\w\-]", "_", config_path.stem).strip("_").lower()
    return (config_path.parent / f".{session_id}").resolve()


def _run_sub_asset_selected(
    sub_asset: SubAsset,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    verbose: bool = False,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    harness_config: SubAssetHarnessConfig | None = None,
    config_dict: dict[str, Any] | None = None,
) -> SubAsset:
    if harness_config and harness_config.enabled:
        return run_sub_asset_with_harness(sub_asset, harness_config)

    return run_sub_asset(
        sub_asset,
        skip_steps,
        only_steps,
        verbose,
        simulate,
        material_names,
        resume=resume,
        from_step=from_step,
        predict_max_workers=predict_max_workers,
        cancel_checker=cancel_checker,
        config_dict=config_dict,
    )


def run_sub_asset_with_harness(
    sub_asset: SubAsset,
    harness_config: SubAssetHarnessConfig,
    verbose: bool = False,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
) -> SubAsset:
    """Reject the retired Material Agent harness loop.

    The execution kwargs are retained so older direct callers still receive the
    migration error instead of a Python signature error.
    """
    _ = (verbose, predict_max_workers, cancel_checker)
    raise RuntimeError(
        "The Material Agent task-first harness workflow has been retired. "
        "Use `content-workflow-cli materials assign` with Content Workbench for "
        "agentic asset workflows."
    )


def _write_sub_asset_harness_error(
    sub_asset: SubAsset,
    exc: BaseException,
) -> None:
    del exc
    if not sub_asset.config_path:
        return

    config_path = Path(sub_asset.config_path)
    try:
        config = _load_yaml_mapping(config_path)
        working_dir = _sub_asset_working_dir(config, config_path)
    except Exception:
        working_dir = config_path.parent / f".{config_path.stem}"

    sub_asset.working_dir = str(working_dir)
    error_path = working_dir / "harness_scene_adapter_error.json"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(
            durable_diagnostic(
                "material_scene_adapter_failed",
                phase=FailurePhase.PIPELINE_EXECUTION,
                retryable=False,
            ).to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _clean_working_dir_for_so_retry(config_path: Path) -> None:
    """Remove SO artifacts and pipeline state so the retry uses the original USD.

    Deletes the ``optimized/`` directory, all downstream outputs (dataset,
    predictions, restored), and the pipeline state file so
    ``build_dataset_usd`` falls back to the original extracted USD.
    """
    import shutil

    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    session_id = cfg.get("project", {}).get("session_id", "")
    working_dir = config_path.parent / f".{session_id}"

    dirs_to_clean = [
        "optimized",
        "dataset",
        "predictions",
        "restored",
        ".pipeline_temp",
    ]
    for d in dirs_to_clean:
        p = working_dir / d
        if p.exists():
            shutil.rmtree(p)

    state_file = working_dir / ".pipeline_state.json"
    if state_file.exists():
        state_file.unlink()

    logger.info(f"Cleaned working dir for SO retry: {working_dir}")


def _clear_pipeline_state_from_step(config_path: Path, from_step: str) -> None:
    """Remove *from_step* and all downstream steps from the pipeline state file.

    This allows ``resume=True`` to re-run those steps while keeping earlier
    steps (like optimize_usd, build_dataset) marked as completed.
    """
    import json

    # Derive working dir from config (same convention as unified_pipeline_executor)
    import yaml

    from material_agent.api.defaults import PIPELINE_STEP_NAMES

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    session_id = cfg.get("project", {}).get("session_id", "")
    working_dir = config_path.parent / f".{session_id}"
    state_file = working_dir / ".pipeline_state.json"

    if not state_file.exists():
        return

    with open(state_file) as f:
        state = json.load(f)

    if from_step not in PIPELINE_STEP_NAMES:
        return

    idx = PIPELINE_STEP_NAMES.index(from_step)
    steps_to_clear = set(PIPELINE_STEP_NAMES[idx:])

    completed = state.get("completed_steps", [])
    original_len = len(completed)
    state["completed_steps"] = [s for s in completed if s not in steps_to_clear]

    # Also remove from step_outputs
    original_outputs = set(state.get("step_outputs", {}))
    for s in steps_to_clear:
        state.get("step_outputs", {}).pop(s, None)
    cleared_outputs = original_outputs - set(state.get("step_outputs", {}))

    had_failed_steps = bool(state.get("failed_steps"))
    state.get("failed_steps", []).clear()
    step_errors = state.get("step_errors")
    cleared_errors = False
    if isinstance(step_errors, dict):
        for s in steps_to_clear:
            if s in step_errors:
                step_errors.pop(s, None)
                cleared_errors = True

    state_changed = (
        len(state["completed_steps"]) < original_len
        or bool(cleared_outputs)
        or had_failed_steps
        or cleared_errors
    )

    if state_changed:
        save_pipeline_checkpoint(state, state_file)
        logger.info(
            f"Cleared pipeline state from '{from_step}' for session '{session_id}'"
        )


def run_sub_asset(
    sub_asset: SubAsset,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    verbose: bool = False,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    config_dict: dict[str, Any] | None = None,
) -> SubAsset:
    """Run the material-agent pipeline on one sub-asset.

    Calls the existing pipeline API with the sub-asset's generated config.

    Args:
        sub_asset: SubAsset with config_path set.
        skip_steps: Steps to skip.
        only_steps: Steps to run exclusively.
        verbose: Enable verbose logging.
        simulate: If True, skip rendering/VLM and use mock predictions.
        material_names: Material names for mock predictions (required if simulate=True).
        resume: If True, resume from last checkpoint (skip completed steps).
        from_step: If set, clear this step and all downstream from pipeline state
            before resuming. Requires resume=True.
        predict_max_workers: Override predict step's max_workers in the per-asset
            config. Useful for scene runs where many assets run in parallel and the
            default (64) causes excessive concurrent VLM calls.
        cancel_checker: Optional callback returning True when the run should stop.

    Returns:
        Updated SubAsset with predictions and material layer paths.
    """
    from material_agent.api.pipeline import PipelineInput, run_pipeline

    if not sub_asset.config_path:
        raise ValueError(f"Sub-asset '{sub_asset.name}' has no config_path set")

    config_path = Path(sub_asset.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    from .config_gen import _load_and_sanitize_generated_config

    durable_config, legacy_credential_paths = _load_and_sanitize_generated_config(
        config_path
    )
    sub_asset.config_credential_paths = sorted(
        set(sub_asset.config_credential_paths) | set(legacy_credential_paths)
    )
    source_config_provided = config_dict is not None
    effective_config = (
        clone_config_containers(config_dict)
        if config_dict is not None
        else durable_config
    )
    if sub_asset.config_credential_paths and not source_config_provided:
        credential_paths = ", ".join(sub_asset.config_credential_paths[:8])
        raise ValueError(
            "Source scene config is required to rehydrate credentials at "
            f"{credential_paths}"
        )
    if source_config_provided:
        missing_paths = set(sub_asset.config_credential_paths) - set(
            find_inline_secret_paths(effective_config)
        )
        if missing_paths:
            credential_paths = ", ".join(sorted(missing_paths)[:8])
            raise ValueError(
                "Runtime scene config is missing inline credentials at "
                f"{credential_paths}"
            )

    # Apply per-asset concurrency only to this isolated runtime configuration.
    if predict_max_workers is not None:
        effective_config.setdefault("steps", {}).setdefault("predict", {})[
            "max_workers"
        ] = predict_max_workers

    # Clear pipeline state from the target step so resume re-runs it
    if from_step and resume:
        _clear_pipeline_state_from_step(config_path, from_step)

    logger.info(f"Running pipeline for '{sub_asset.name}' ({sub_asset.prim_path})")

    simulation_config = effective_config
    if simulate:
        from material_agent.api.simulate_config import patch_config_for_simulate

        # Credential continuity is validated above against the original source
        # configuration. Only then replace provider credentials in this
        # per-item copy so simulation cannot execute or retain a live key.
        simulation_config = patch_config_for_simulate(effective_config)

    simulate_kwargs: dict[str, Any] = {
        "verbose": verbose,
        "cancel_checker": cancel_checker,
        "config_dict": simulation_config,
    }

    if simulate:
        if not material_names:
            raise ValueError("material_names is required when simulate=True")
        result = _run_simulate(
            config_path,
            material_names,
            **simulate_kwargs,
        )
    else:
        params = PipelineInput(
            config=clone_config_containers(effective_config),
            skip_steps=skip_steps or [],
            only_steps=only_steps or [],
            verbose=verbose,
            resume=resume,
            simulate=False,
            cancel_checker=cancel_checker,
            config_path=config_path,
        )
        result = run_pipeline(params)

    # SO fallback: retry without SO optimization when:
    #  (a) optimize_usd step failed outright, OR
    #  (b) pipeline succeeded but SO produced 0 predictions (SO split meshes
    #      in a way that left no renderable geometry for the VLM).
    _needs_so_retry = False
    _had_so = "optimize_usd" in result.completed_steps
    if not result.success and _had_so:
        # SO step ran but the pipeline still failed — retry without it
        _needs_so_retry = True
        logger.warning(
            f"optimize_usd failed for '{sub_asset.name}', "
            f"retrying without SO optimization"
        )
    elif result.success and _had_so and "predict" in result.completed_steps:
        pred_count = result.step_results.get("predict", {}).get("predictions_count", -1)
        if pred_count == 0:
            _needs_so_retry = True
            logger.warning(
                f"0 predictions after SO for '{sub_asset.name}', "
                f"retrying without SO optimization"
            )

    if _needs_so_retry:
        # Clean SO artifacts so build_dataset_usd uses the original USD
        _clean_working_dir_for_so_retry(config_path)

        skip_no_so = list(set((skip_steps or []) + ["optimize_usd"]))
        if simulate:
            result = _run_simulate(
                config_path,
                material_names or [],
                **simulate_kwargs,
            )
        else:
            params_no_so = PipelineInput(
                config=clone_config_containers(effective_config),
                skip_steps=skip_no_so,
                only_steps=only_steps or [],
                verbose=verbose,
                resume=False,  # clean start for retry
                simulate=False,
                cancel_checker=cancel_checker,
                config_path=config_path,
            )
            result = run_pipeline(params_no_so)

    if result.success:
        logger.info(f"Pipeline completed for '{sub_asset.name}'")
        sub_asset.status = "completed"

        # Derive output paths from working directory convention
        _update_output_paths(sub_asset, config_path)
    else:
        logger.error(f"Pipeline failed for '{sub_asset.name}': {result.error}")
        sub_asset.status = "failed"

    return sub_asset


def _run_sub_asset_worker(
    sub_asset: SubAsset,
    skip_steps: list[str] | None,
    only_steps: list[str] | None,
    verbose: bool,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    harness_config: SubAssetHarnessConfig | None = None,
    config_dict: dict[str, Any] | None = None,
    scene_config: dict[str, Any] | None = None,
    scene_config_dir: Path | None = None,
) -> SubAsset:
    """Worker function for parallel execution.

    Uses asyncio.run() in its own event loop per thread.

    Returns:
        Updated SubAsset.
    """
    try:
        runtime_config = config_dict
        if runtime_config is None:
            runtime_config, config_ready = _prepare_sub_asset_runtime_config_for_run(
                sub_asset,
                scene_config,
                scene_config_dir,
            )
            if not config_ready:
                sub_asset.status = "failed"
                return sub_asset
        return _run_sub_asset_selected(
            sub_asset,
            skip_steps,
            only_steps,
            verbose,
            simulate,
            material_names,
            resume=resume,
            from_step=from_step,
            predict_max_workers=predict_max_workers,
            cancel_checker=cancel_checker,
            harness_config=harness_config,
            config_dict=runtime_config,
        )
    except Exception as exc:
        logger.error("Unable to process one scene sub-asset")
        sub_asset.status = "failed"
        if harness_config and harness_config.enabled:
            _write_sub_asset_harness_error(sub_asset, exc)
        return sub_asset


def run_all(
    manifest: SceneManifest,
    manifest_path: Path,
    names_filter: list[str] | None = None,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    skip_existing: bool = False,
    max_workers: int = 1,
    verbose: bool = False,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    progress_callback: ProgressCallback | None = None,
    harness_config: SubAssetHarnessConfig | None = None,
    scene_config: dict[str, Any] | None = None,
    scene_config_dir: Path | None = None,
) -> SceneManifest:
    """Run pipelines for all processable assets.

    When max_workers > 1, assets are processed in parallel using
    ThreadPoolExecutor. The manifest is saved after each batch of
    completions for resume safety.

    Args:
        manifest: Scene manifest with configs generated.
        manifest_path: Path to save manifest updates.
        names_filter: Optional name/path filter for assets.
        skip_steps: Steps to skip for all assets.
        only_steps: Steps to run exclusively for all assets.
        skip_existing: Skip assets with status == "completed".
        max_workers: Number of parallel workers (default: 1 = sequential).
        verbose: Enable verbose logging.
        resume: If True, resume per-asset pipelines from last checkpoint.
        from_step: If set, clear this step and downstream from pipeline state.
        harness_config: Retired compatibility settings. If enabled, sub-asset
            processing records a retired-workflow error instead of running the
            old harness path.

    Returns:
        Updated SceneManifest.
    """
    _raise_if_cancelled(cancel_checker)
    assets = manifest.get_processable_assets(names_filter)

    # Build set of representative asset IDs so they get processed
    representative_ids: set[str] = set()
    for ig in manifest.instance_groups:
        if ig.representative_id:
            representative_ids.add(ig.representative_id)

    # Filter out skipped/unconfigured/duplicate assets
    to_process: list[SubAsset] = []
    skipped = 0
    instance_group_members: list[SubAsset] = []
    for sa in assets:
        if skip_existing and sa.status == "completed":
            logger.info(f"Skipping '{sa.name}' (already completed)")
            skipped += 1
            continue
        if not sa.config_path:
            logger.warning(f"Skipping '{sa.name}' (no config generated)")
            skipped += 1
            continue
        if sa.instance_group and sa.id not in representative_ids:
            # Skip non-representative instance group members (duplicates).
            # Representatives must be processed so their results can be
            # copied to all other members of the group.
            instance_group_members.append(sa)
            skipped += 1
            continue
        to_process.append(sa)

    total = len(to_process)
    logger.info(
        f"Running pipeline for {total} sub-assets "
        f"(workers={max_workers}, skipped={skipped}, "
        "retired_harness_requested="
        f"{bool(harness_config and harness_config.enabled)})"
    )

    if total == 0:
        return manifest

    if progress_callback:
        progress_callback(
            {
                "current": 0,
                "total": total,
                "completed": 0,
                "failed": 0,
                "asset_id": None,
                "asset_name": None,
                "asset_status": "pending",
                "message": f"Processing {total} scene assets",
            }
        )

    if max_workers <= 1:
        # Sequential execution
        completed, failed = _run_sequential(
            to_process,
            manifest,
            manifest_path,
            skip_steps,
            only_steps,
            verbose,
            simulate=simulate,
            material_names=material_names,
            resume=resume,
            from_step=from_step,
            predict_max_workers=predict_max_workers,
            cancel_checker=cancel_checker,
            progress_callback=progress_callback,
            harness_config=harness_config,
            scene_config=scene_config,
            scene_config_dir=scene_config_dir,
        )
    else:
        # Parallel execution
        completed, failed = _run_parallel(
            to_process,
            manifest,
            manifest_path,
            skip_steps,
            only_steps,
            verbose,
            max_workers,
            simulate=simulate,
            material_names=material_names,
            resume=resume,
            from_step=from_step,
            predict_max_workers=predict_max_workers,
            cancel_checker=cancel_checker,
            progress_callback=progress_callback,
            harness_config=harness_config,
            scene_config=scene_config,
            scene_config_dir=scene_config_dir,
        )

    # Copy results from representatives to structural duplicate members
    if instance_group_members:
        _copy_results_to_duplicates(manifest, instance_group_members)

    logger.info(
        f"Pipeline run complete: {completed} completed, "
        f"{failed} failed, {skipped} skipped"
    )
    return manifest


def _copy_results_to_duplicates(
    manifest: SceneManifest,
    members: list[SubAsset],
) -> None:
    """Copy predictions and status from representative to duplicate members.

    For structural duplicate groups, only the representative was processed.
    This copies its predictions_path, material_layer_path, and status to
    all members in the same instance group.

    Args:
        manifest: The scene manifest (to look up representatives).
        members: List of sub-assets that were skipped as duplicates.
    """
    import shutil

    # Build representative lookup: instance_group name -> representative SubAsset
    rep_map: dict[str, SubAsset] = {}
    for ig in manifest.instance_groups:
        if ig.representative_id:
            for sa in manifest.sub_assets:
                if sa.id == ig.representative_id:
                    rep_map[ig.group_name] = sa
                    break

    copied = 0
    for member in members:
        group_name = member.instance_group
        if not group_name or group_name not in rep_map:
            continue
        rep = rep_map[group_name]
        if rep.status != "completed":
            logger.warning(
                f"Representative '{rep.name}' not completed, "
                f"skipping copy to '{member.name}'"
            )
            continue

        # Copy predictions file if it exists
        if rep.predictions_path and member.working_dir:
            src = Path(rep.predictions_path)
            dst_dir = Path(member.working_dir) / "predictions"
            if src.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                member.predictions_path = str(dst)

        # Copy material layer if it exists
        if rep.material_layer_path and member.working_dir:
            src = Path(rep.material_layer_path)
            dst_dir = Path(member.working_dir) / "output"
            if src.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                member.material_layer_path = str(dst)

        member.status = "completed"
        copied += 1

    if copied:
        logger.info(f"Copied results to {copied} structural duplicate members")


def _run_sequential(
    assets: list[SubAsset],
    manifest: SceneManifest,
    manifest_path: Path,
    skip_steps: list[str] | None,
    only_steps: list[str] | None,
    verbose: bool,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    progress_callback: ProgressCallback | None = None,
    harness_config: SubAssetHarnessConfig | None = None,
    runtime_configs: dict[str, dict[str, Any]] | None = None,
    scene_config: dict[str, Any] | None = None,
    scene_config_dir: Path | None = None,
) -> tuple[int, int]:
    """Run assets sequentially, saving manifest after each."""
    completed = 0
    failed = 0
    total = len(assets)

    for i, sa in enumerate(assets, 1):
        _raise_if_cancelled(cancel_checker)
        logger.info(f"[{i}/{total}] Processing '{sa.name}'...")
        try:
            runtime_config = (runtime_configs or {}).get(sa.id)
            config_ready = True
            if runtime_config is None:
                runtime_config, config_ready = (
                    _prepare_sub_asset_runtime_config_for_run(
                        sa,
                        scene_config,
                        scene_config_dir,
                    )
                )
                if not config_ready:
                    sa.status = "failed"
                    failed += 1
            if config_ready:
                _run_sub_asset_selected(
                    sa,
                    skip_steps,
                    only_steps,
                    verbose,
                    simulate,
                    material_names,
                    resume=resume,
                    from_step=from_step,
                    predict_max_workers=predict_max_workers,
                    cancel_checker=cancel_checker,
                    harness_config=harness_config,
                    config_dict=runtime_config,
                )
                if sa.status == "completed":
                    completed += 1
                else:
                    failed += 1
        except Exception as exc:
            logger.error("Unable to process one scene sub-asset")
            sa.status = "failed"
            if harness_config and harness_config.enabled:
                _write_sub_asset_harness_error(sa, exc)
            failed += 1

        manifest.save(manifest_path)
        if progress_callback:
            progress_callback(
                {
                    "current": completed + failed,
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                    "asset_id": sa.id,
                    "asset_name": sa.name,
                    "asset_status": sa.status,
                }
            )
        _raise_if_cancelled(cancel_checker)

    return completed, failed


def _run_parallel(
    assets: list[SubAsset],
    manifest: SceneManifest,
    manifest_path: Path,
    skip_steps: list[str] | None,
    only_steps: list[str] | None,
    verbose: bool,
    max_workers: int,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    progress_callback: ProgressCallback | None = None,
    harness_config: SubAssetHarnessConfig | None = None,
    runtime_configs: dict[str, dict[str, Any]] | None = None,
    scene_config: dict[str, Any] | None = None,
    scene_config_dir: Path | None = None,
) -> tuple[int, int]:
    """Run assets in parallel using ThreadPoolExecutor.

    Threads are used instead of processes because each pipeline call uses
    asyncio.run() internally and the heavy work (NVCF rendering, VLM API
    calls) is I/O-bound.  This avoids the memory cost of forking the full
    USD stage into multiple processes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = 0
    failed = 0
    total = len(assets)

    # Build a mapping from asset id -> index in manifest.sub_assets
    # so we can update the right SubAsset after workers return
    asset_index_map: dict[str, int] = {}
    for asset_idx, sa in enumerate(manifest.sub_assets):
        asset_index_map[sa.id] = asset_idx

    # Sort largest assets first so big jobs start early and small ones fill gaps
    sorted_assets = sorted(assets, key=lambda sa: sa.mesh_count, reverse=True)
    if sorted_assets:
        logger.info(
            f"Launching {min(max_workers, total)} parallel workers for {total} assets "
            f"(sorted by mesh count: {sorted_assets[0].mesh_count} → "
            f"{sorted_assets[-1].mesh_count})"
        )
    else:
        logger.info("No assets to process")
        return completed, failed

    manifest_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
        _raise_if_cancelled(cancel_checker)
        future_to_asset = {
            executor.submit(
                _run_sub_asset_worker,
                sa,
                skip_steps,
                only_steps,
                verbose,
                simulate,
                material_names,
                resume,
                from_step,
                predict_max_workers,
                cancel_checker,
                harness_config,
                (runtime_configs or {}).get(sa.id),
                scene_config,
                scene_config_dir,
            ): sa
            for sa in sorted_assets
        }

        for future in as_completed(future_to_asset):
            try:
                _raise_if_cancelled(cancel_checker)
            except asyncio.CancelledError:
                for pending in future_to_asset:
                    pending.cancel()
                raise
            original_sa = future_to_asset[future]
            progress_asset = original_sa
            with manifest_lock:
                try:
                    result_sa = future.result()
                    progress_asset = result_sa
                    # Update the manifest's sub_asset in-place
                    result_asset_idx = asset_index_map.get(result_sa.id)
                    if result_asset_idx is not None:
                        manifest.sub_assets[result_asset_idx] = result_sa

                    if result_sa.status == "completed":
                        completed += 1
                        logger.info(
                            f"[{completed + failed}/{total}] Completed '{result_sa.name}'"
                        )
                    else:
                        failed += 1
                        logger.error(
                            f"[{completed + failed}/{total}] Failed '{result_sa.name}'"
                        )
                except asyncio.CancelledError:
                    for pending in future_to_asset:
                        pending.cancel()
                    raise
                except Exception:
                    failed += 1
                    logger.error("Unable to collect one scene sub-asset result")
                    original_asset_idx = asset_index_map.get(original_sa.id)
                    if original_asset_idx is not None:
                        manifest.sub_assets[original_asset_idx].status = "failed"

                # Save manifest periodically (every completion)
                manifest.save(manifest_path)
                if progress_callback:
                    progress_callback(
                        {
                            "current": completed + failed,
                            "total": total,
                            "completed": completed,
                            "failed": failed,
                            "asset_id": progress_asset.id,
                            "asset_name": progress_asset.name,
                            "asset_status": progress_asset.status,
                        }
                    )

    return completed, failed


def _run_simulate(
    config_path: Path,
    material_names: list[str],
    verbose: bool,
    cancel_checker: CancelChecker | None = None,
    config_dict: dict[str, Any] | None = None,
) -> PipelineOutput:
    """Run a two-phase simulate pipeline: SO (real) + mock predictions + apply.

    Phase 1: Run optimize_usd step only (if enabled in config).
    Phase 2: Generate mock predictions from the optimized (or original) USD.
    Phase 3: Run restore_usd + apply steps with resume=True.

    Args:
        config_path: Path to the per-asset/payload config YAML.
        material_names: Material names for round-robin mock predictions.
        verbose: Enable verbose logging.
        cancel_checker: Optional callback returning True when the run should stop.
        config_dict: Optional isolated runtime config; ``config_path`` remains its
            relative-path anchor.

    Returns:
        PipelineOutput from the final phase.
    """
    from material_agent.api.pipeline import PipelineInput, PipelineOutput, run_pipeline

    config = (
        clone_config_containers(config_dict)
        if config_dict is not None
        else _load_yaml_mapping(config_path)
    )
    steps = config.get("steps", {})
    so_enabled = steps.get("optimize_usd", {}).get("enabled", False)

    session_id = config.get("project", {}).get("session_id", "")
    working_dir = config_path.parent / f".{session_id}"

    # Phase 1: Run SO if enabled
    if so_enabled:
        logger.info(f"simulate phase 1: running optimize_usd for {config_path.name}")
        result = run_pipeline(
            PipelineInput(
                config=clone_config_containers(config),
                only_steps=["optimize_usd"],
                verbose=verbose,
                cancel_checker=cancel_checker,
                config_path=config_path,
            )
        )
        if not result.success or "optimize_usd" not in result.completed_steps:
            # SO failed — fall through to phase 2/3 without SO.
            # Predictions will be generated from the original USD.
            logger.warning(
                f"simulate phase 1: SO failed for {config_path.name}, "
                f"continuing without optimization"
            )
            so_enabled = False

    # Phase 2: Generate mock predictions
    logger.info(f"simulate phase 2: generating mock predictions for {config_path.name}")
    pred_count = _generate_simulate_predictions(
        config, config_path, working_dir, material_names
    )

    # Write simulate marker
    marker = working_dir / ".simulate"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("mock predictions — simulate mode\n")

    # If no predictions were generated (container-only payload with 0 direct
    # meshes), skip phase 3 — the optimized output is already the deliverable.
    if pred_count == 0:
        logger.info(
            f"simulate: 0 predictions for {config_path.name}, "
            f"skipping restore/apply (container-only payload)"
        )
        return PipelineOutput(success=True, completed_steps=["optimize_usd"])

    apply_config = steps.get("apply", {})
    apply_enabled = False
    if isinstance(apply_config, dict):
        explicit_enabled = apply_config.get("enabled")
        if explicit_enabled is None:
            apply_enabled = any(k != "enabled" for k in apply_config)
        else:
            apply_enabled = bool(explicit_enabled)

    if not apply_enabled:
        logger.info(
            f"simulate: wrote {pred_count} predictions for {config_path.name}; "
            "skipping per-asset apply because apply.enabled is false"
        )
        return PipelineOutput(
            success=True,
            completed_steps=["predict"],
            step_results={"predict": {"predictions_count": pred_count}},
        )

    # Phase 3: Run remaining steps
    phase3_steps = ["restore_usd", "apply"] if so_enabled else ["apply"]
    logger.info(f"simulate phase 3: running {phase3_steps} for {config_path.name}")
    result = run_pipeline(
        PipelineInput(
            config=clone_config_containers(config),
            only_steps=phase3_steps,
            resume=True,
            verbose=verbose,
            cancel_checker=cancel_checker,
            config_path=config_path,
        )
    )

    # Phase 4: When SO ran, append original-USD predictions to the restored
    # file so that parent Mesh prims (which SO splits into per-subset meshes)
    # get predictions too.  This must happen after restore_usd which only
    # keeps predictions that match its SO→original path mapping.
    if so_enabled:
        restored_path = working_dir / "restored" / "restored_predictions.jsonl"
        target = (
            restored_path
            if restored_path.exists()
            else (working_dir / "predictions" / "predictions.jsonl")
        )
        input_usd = config.get("input", {}).get("usd_path", "")
        if input_usd and not Path(input_usd).is_absolute():
            original_usd = (config_path.parent / input_usd).resolve()
        else:
            original_usd = Path(input_usd)

        if original_usd.exists():
            from .simulate import generate_mock_predictions_append

            extra = generate_mock_predictions_append(
                usd_path=original_usd,
                material_names=material_names,
                output_path=target,
                prim_path_scope=config.get("input", {}).get("prim_path"),
            )
            if extra:
                logger.info(
                    f"simulate: appended {extra} original-USD predictions "
                    f"to {target.name}"
                )

    return result


def _generate_simulate_predictions(
    config: dict,
    config_path: Path,
    working_dir: Path,
    material_names: list[str],
) -> int:
    """Generate mock predictions for simulate mode.

    Enumerates prims from both the original input USD and the SO-optimized
    USD (if it exists), merging them so that parent Mesh prims that SO split
    into GeomSubset-level meshes still get predictions.

    Returns:
        Number of predictions written.
    """
    from .simulate import generate_mock_predictions

    # Use SO-optimized USD if available, otherwise original input
    optimized_usd = working_dir / "optimized" / "optimized_input.usd"
    if optimized_usd.exists():
        usd_path = optimized_usd
    else:
        input_usd = config.get("input", {}).get("usd_path", "")
        if input_usd and not Path(input_usd).is_absolute():
            usd_path = (config_path.parent / input_usd).resolve()
        else:
            usd_path = Path(input_usd)

    prim_path_scope = config.get("input", {}).get("prim_path")

    predictions_dir = working_dir / "predictions"
    predictions_path = predictions_dir / "predictions.jsonl"

    return generate_mock_predictions(
        usd_path=usd_path,
        material_names=material_names,
        output_path=predictions_path,
        prim_path_scope=prim_path_scope,
    )


def _update_output_paths(sub_asset: SubAsset, config_path: Path) -> None:
    """Derive predictions and material layer paths from pipeline output convention.

    The pipeline writes to:
    - .{session_id}/predictions/predictions.jsonl
    - .{session_id}/restored/restored_predictions.jsonl (if restore_usd ran)
    - .{session_id}/output/output.usd (only if apply step ran)

    In scene mode, per-asset apply is disabled. The asset is considered
    complete when predictions exist (the collect step handles unified apply).
    """
    import re

    import yaml

    # Read the config to get session_id
    with open(config_path) as f:
        config = yaml.safe_load(f)

    session_id = config.get("project", {}).get("session_id", "")
    if not session_id:
        safe_name = re.sub(r"[^\w\-]", "_", sub_asset.name).strip("_").lower()
        session_id = safe_name

    config_dir = config_path.parent
    working_dir = config_dir / f".{session_id}"

    sub_asset.working_dir = str(working_dir)

    # Check for restored predictions first (restore_usd remaps SO paths
    # back to original scene paths), then fall back to raw predictions.
    restored_path = working_dir / "restored" / "restored_predictions.jsonl"
    predictions_path = working_dir / "predictions" / "predictions.jsonl"

    if restored_path.exists():
        sub_asset.predictions_path = str(restored_path)
    elif predictions_path.exists():
        sub_asset.predictions_path = str(predictions_path)

    # Material layer is optional — in scene mode, per-asset apply is
    # disabled and the collect step creates a unified layer instead.
    material_layer_path = working_dir / "output" / "output.usd"
    if material_layer_path.exists():
        sub_asset.material_layer_path = str(material_layer_path)


# ---------------------------------------------------------------------------
# Payload pipeline
# ---------------------------------------------------------------------------


def run_payload(
    payload_group: PayloadGroup,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    verbose: bool = False,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    config_dict: dict[str, Any] | None = None,
) -> PayloadGroup:
    """Run the material-agent pipeline on one payload group.

    Args:
        payload_group: PayloadGroup with config_path set.
        skip_steps: Steps to skip.
        only_steps: Steps to run exclusively.
        verbose: Enable verbose logging.
        simulate: If True, skip rendering/VLM and use mock predictions.
        material_names: Material names for mock predictions (required if simulate=True).
        resume: If True, resume from last checkpoint (skip completed steps).
        from_step: If set, clear this step and downstream from pipeline state.
        predict_max_workers: Override predict step's max_workers in per-asset config.
        cancel_checker: Optional callback returning True when the run should stop.

    Returns:
        Updated PayloadGroup with predictions path.
    """
    from material_agent.api.pipeline import PipelineInput, run_pipeline

    if not payload_group.config_path:
        raise ValueError(
            f"Payload group '{payload_group.group_name}' has no config_path set"
        )

    config_path = Path(payload_group.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    from .config_gen import _load_and_sanitize_generated_config

    durable_config, legacy_credential_paths = _load_and_sanitize_generated_config(
        config_path
    )
    payload_group.config_credential_paths = sorted(
        set(payload_group.config_credential_paths) | set(legacy_credential_paths)
    )
    source_config_provided = config_dict is not None
    effective_config = (
        clone_config_containers(config_dict)
        if config_dict is not None
        else durable_config
    )
    if payload_group.config_credential_paths and not source_config_provided:
        credential_paths = ", ".join(payload_group.config_credential_paths[:8])
        raise ValueError(
            "Source scene config is required to rehydrate credentials at "
            f"{credential_paths}"
        )
    if source_config_provided:
        missing_paths = set(payload_group.config_credential_paths) - set(
            find_inline_secret_paths(effective_config)
        )
        if missing_paths:
            credential_paths = ", ".join(sorted(missing_paths)[:8])
            raise ValueError(
                "Runtime scene config is missing inline credentials at "
                f"{credential_paths}"
            )

    # Apply per-payload concurrency only to this isolated runtime configuration.
    if predict_max_workers is not None:
        effective_config.setdefault("steps", {}).setdefault("predict", {})[
            "max_workers"
        ] = predict_max_workers

    # Clear pipeline state from the target step so resume re-runs it
    if from_step and resume:
        _clear_pipeline_state_from_step(config_path, from_step)

    logger.info(
        f"Running pipeline for payload '{payload_group.group_name}' "
        f"({payload_group.payload_file})"
    )

    simulation_config = effective_config
    if simulate:
        from material_agent.api.simulate_config import patch_config_for_simulate

        simulation_config = patch_config_for_simulate(effective_config)

    simulate_kwargs: dict[str, Any] = {
        "verbose": verbose,
        "cancel_checker": cancel_checker,
        "config_dict": simulation_config,
    }

    if simulate:
        if not material_names:
            raise ValueError("material_names is required when simulate=True")
        result = _run_simulate(
            config_path,
            material_names,
            **simulate_kwargs,
        )
    else:
        params = PipelineInput(
            config=clone_config_containers(effective_config),
            skip_steps=skip_steps or [],
            only_steps=only_steps or [],
            verbose=verbose,
            resume=resume,
            simulate=False,
            cancel_checker=cancel_checker,
            config_path=config_path,
        )
        result = run_pipeline(params)

    # SO fallback: retry without SO when it failed or produced 0 predictions.
    _needs_so_retry = False
    _had_so = "optimize_usd" in result.completed_steps
    if not result.success and _had_so:
        _needs_so_retry = True
        logger.warning(
            f"optimize_usd failed for '{payload_group.group_name}', "
            f"retrying without SO optimization"
        )
    elif result.success and _had_so and "predict" in result.completed_steps:
        pred_count = result.step_results.get("predict", {}).get("predictions_count", -1)
        if pred_count == 0:
            _needs_so_retry = True
            logger.warning(
                f"0 predictions after SO for '{payload_group.group_name}', "
                f"retrying without SO optimization"
            )

    if _needs_so_retry:
        # Clean SO artifacts so build_dataset_usd uses the original USD
        _clean_working_dir_for_so_retry(config_path)

        skip_no_so = list(set((skip_steps or []) + ["optimize_usd"]))
        if simulate:
            result = _run_simulate(
                config_path,
                material_names or [],
                **simulate_kwargs,
            )
        else:
            params_no_so = PipelineInput(
                config=clone_config_containers(effective_config),
                skip_steps=skip_no_so,
                only_steps=only_steps or [],
                verbose=verbose,
                resume=False,  # clean start for retry
                simulate=False,
                cancel_checker=cancel_checker,
                config_path=config_path,
            )
            result = run_pipeline(params_no_so)

    if result.success:
        logger.info(f"Pipeline completed for payload '{payload_group.group_name}'")
        payload_group.status = "completed"
        _update_payload_output_paths(payload_group, config_path)
    else:
        logger.error(
            f"Pipeline failed for payload '{payload_group.group_name}': {result.error}"
        )
        payload_group.status = "failed"

    return payload_group


def _run_payload_worker(
    payload_group: PayloadGroup,
    skip_steps: list[str] | None,
    only_steps: list[str] | None,
    verbose: bool,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    config_dict: dict[str, Any] | None = None,
    scene_config: dict[str, Any] | None = None,
    scene_config_dir: Path | None = None,
    sibling_names: list[str] | None = None,
) -> PayloadGroup:
    """Worker function for parallel payload execution."""
    try:
        runtime_config = config_dict
        if runtime_config is None:
            runtime_config, config_ready = _prepare_payload_runtime_config_for_run(
                payload_group,
                scene_config,
                scene_config_dir,
                sibling_names,
            )
            if not config_ready:
                payload_group.status = "failed"
                return payload_group
        return run_payload(
            payload_group,
            skip_steps,
            only_steps,
            verbose,
            simulate,
            material_names,
            resume=resume,
            from_step=from_step,
            predict_max_workers=predict_max_workers,
            cancel_checker=cancel_checker,
            config_dict=runtime_config,
        )
    except Exception:
        logger.error("Unable to process one scene payload")
        payload_group.status = "failed"
        return payload_group


def run_all_payloads_bottomup(
    manifest: SceneManifest,
    manifest_path: Path,
    scene_config: dict,
    configs_dir: Path,
    scene_config_dir: Path | None = None,
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    skip_existing: bool = False,
    max_workers: int = 1,
    verbose: bool = False,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
) -> SceneManifest:
    """Run payload pipelines in bottom-up topological order.

    Processing order:
    1. All depth-0 payloads (leaves) — can run in parallel
    2. Create modified copies for depth-1 parents (rewrite child refs)
    3. Generate configs for depth-1 parents, run them
    4. Repeat for depth-2, 3, ... up to max depth

    After each payload completes, ``output_usd_path`` is set to the
    apply step's ``output.usd``. Parent payloads at the next level use
    these paths when creating their modified input copies.

    Args:
        manifest: Scene manifest with payload configs generated.
        manifest_path: Path to save manifest updates.
        scene_config: Scene-level config dict (for regenerating parent configs).
        configs_dir: Directory for generated configs.
        scene_config_dir: Original scene config dir (for path rebasing).
        skip_steps: Steps to skip for all payloads.
        only_steps: Steps to run exclusively for all payloads.
        skip_existing: Skip payloads with status == "completed".
        max_workers: Number of parallel workers.
        verbose: Enable verbose logging.
        predict_max_workers: Override predict step's max_workers in per-asset config.

    Returns:
        Updated SceneManifest.
    """
    from .config_gen import (
        _build_payload_sibling_map,
        generate_payload_config,
    )

    _raise_if_cancelled(cancel_checker)
    payloads_by_depth = manifest.get_payloads_by_depth()
    if not payloads_by_depth:
        logger.info("No processable payload groups")
        return manifest

    max_depth = max(payloads_by_depth.keys())
    total_all = sum(len(v) for v in payloads_by_depth.values())
    total_completed = 0
    total_failed = 0

    logger.info(
        f"Running {total_all} payload groups bottom-up "
        f"(max depth={max_depth}, workers={max_workers})"
    )
    payload_sibling_map = _build_payload_sibling_map(manifest)

    for depth in range(0, max_depth + 1):
        _raise_if_cancelled(cancel_checker)
        level_payloads = payloads_by_depth.get(depth, [])
        if not level_payloads:
            continue

        logger.info(f"--- Depth {depth}: {len(level_payloads)} payloads ---")

        if depth > 0:
            # Create modified copies for parent payloads and regenerate configs
            for pg in level_payloads:
                _raise_if_cancelled(cancel_checker)
                if skip_existing and pg.status == "completed":
                    continue
                try:
                    _create_modified_parent_copy(pg, manifest, configs_dir)
                    # Regenerate config pointing to modified copy
                    payload_configs_dir = configs_dir / "payloads"
                    config_path = payload_configs_dir / f"{pg.group_name}.yaml"
                    generate_payload_config(
                        pg, scene_config, config_path, scene_config_dir
                    )
                    pg.config_path = str(config_path)
                    pg.working_dir = str(payload_configs_dir / f".{pg.group_name}")
                except Exception:
                    logger.error("Unable to prepare one parent scene payload")
                    pg.status = "failed"

        # Filter payloads to run at this level
        to_process: list[PayloadGroup] = []
        for pg in level_payloads:
            if skip_existing and pg.status == "completed":
                # Still update output_usd_path for already-completed payloads
                _set_payload_output_usd(pg)
                continue
            if not pg.config_path:
                logger.warning(f"Skipping payload '{pg.group_name}' (no config)")
                continue
            to_process.append(pg)

        if not to_process:
            continue

        if max_workers <= 1:
            completed, failed = _run_payloads_sequential(
                to_process,
                manifest,
                manifest_path,
                skip_steps,
                only_steps,
                verbose,
                simulate=simulate,
                material_names=material_names,
                resume=resume,
                from_step=from_step,
                predict_max_workers=predict_max_workers,
                cancel_checker=cancel_checker,
                scene_config=scene_config,
                scene_config_dir=scene_config_dir,
                payload_sibling_map=payload_sibling_map,
            )
        else:
            completed, failed = _run_payloads_parallel(
                to_process,
                manifest,
                manifest_path,
                skip_steps,
                only_steps,
                verbose,
                max_workers,
                simulate=simulate,
                material_names=material_names,
                resume=resume,
                from_step=from_step,
                predict_max_workers=predict_max_workers,
                cancel_checker=cancel_checker,
                scene_config=scene_config,
                scene_config_dir=scene_config_dir,
                payload_sibling_map=payload_sibling_map,
            )

        # Update output_usd_path for completed payloads and fix
        # material paths for payloads with non-World defaultPrim
        for pg in level_payloads:
            if pg.status == "completed":
                _set_payload_output_usd(pg)
                _fix_output_material_scope(pg)
                # For representative payloads, swap the sublayer in output.usd
                # from the representative file back to the original payload.
                if pg.representative_path:
                    _fix_representative_sublayer(pg)

        total_completed += completed
        total_failed += failed
        manifest.save(manifest_path)

        logger.info(f"Depth {depth} complete: {completed} completed, {failed} failed")

    logger.info(
        f"Bottom-up payload pipeline complete: "
        f"{total_completed} completed, {total_failed} failed"
    )
    return manifest


def _create_modified_parent_copy(
    payload_group: PayloadGroup,
    manifest: SceneManifest,
    working_dir: Path,
) -> None:
    """Create a modified copy of a parent payload with updated child references.

    Copies the payload file (and its sublayers) to the working directory,
    then rewrites all payload/reference arcs that point to child payload
    files to instead point to the child's ``output_usd_path`` (the updated
    version with materials applied).

    Args:
        payload_group: The parent payload group.
        manifest: Scene manifest (for looking up child output paths).
        working_dir: Base working directory for modified copies.
    """
    import shutil

    from pxr import Sdf

    from material_agent.scene.payload_dag_utils import rewrite_arcs_in_layer

    payload_dir = working_dir / "payloads" / payload_group.group_name
    payload_dir.mkdir(parents=True, exist_ok=True)

    original = Path(payload_group.payload_file)
    modified = payload_dir / f"{original.stem}_modified{original.suffix}"

    # Copy original payload file
    shutil.copy2(str(original), str(modified))

    # Build child mapping: original child abs path -> updated output.usd path
    child_map: dict[str, str] = {}
    for child_file in payload_group.child_payload_files:
        child_pg = manifest.get_payload_by_file(child_file)
        if child_pg and child_pg.output_usd_path:
            child_map[str(Path(child_file).resolve())] = child_pg.output_usd_path
        elif child_pg and child_pg.modified_input_path:
            # Container payloads (0 direct meshes) have no output.usd but
            # their modified_input_path already references materialized
            # grandchildren — use it as the rewrite target.
            child_map[str(Path(child_file).resolve())] = child_pg.modified_input_path
        elif child_pg and child_pg.status == "skipped":
            pass  # Skip empty payloads — keep original reference
        else:
            logger.warning(
                f"Child payload not ready for '{payload_group.group_name}': "
                f"{Path(child_file).name}"
            )

    if not child_map:
        # No children to update — use the copy as-is
        payload_group.modified_input_path = str(modified)
        return

    # Open the copy and rewrite arcs
    layer = Sdf.Layer.FindOrOpen(str(modified))
    if not layer:
        raise RuntimeError(f"Failed to open modified copy: {modified}")

    # Resolve arcs from the ORIGINAL location (not the copy's), because
    # relative paths in the layer are relative to where the file was
    # authored, not where we copied it.
    count = rewrite_arcs_in_layer(layer, child_map, resolve_from=str(original))

    # Also check sublayers of this file for arcs to rewrite.
    # Use the ORIGINAL layer for resolving sublayer paths too.
    orig_layer = Sdf.Layer.FindOrOpen(str(original))
    for sl_path in list(layer.subLayerPaths):
        # Resolve sublayer from original location
        resolved = orig_layer.ComputeAbsolutePath(sl_path) if orig_layer else None
        if not resolved or not Path(resolved).exists():
            continue

        sl_orig = Path(resolved).resolve()
        sl_copy = payload_dir / sl_orig.name
        if not sl_copy.exists():
            shutil.copy2(str(sl_orig), str(sl_copy))

        sl_layer = Sdf.Layer.FindOrOpen(str(sl_copy))
        if sl_layer:
            sl_count = rewrite_arcs_in_layer(
                sl_layer, child_map, resolve_from=str(sl_orig)
            )
            if sl_count > 0:
                sl_layer.Save()
                # Update parent's sublayer path to point to copy
                idx = list(layer.subLayerPaths).index(sl_path)
                layer.subLayerPaths[idx] = str(sl_copy)
                count += sl_count

    layer.Save()
    payload_group.modified_input_path = str(modified)

    if count:
        logger.info(
            f"Created modified copy for '{payload_group.group_name}': "
            f"{count} arcs rewritten"
        )


def _fix_output_material_scope(payload_group: PayloadGroup) -> None:
    """Relocate materials in output.usd to be under the payload's defaultPrim.

    The apply step always writes materials at ``/World/Looks/...``.  For
    payloads whose ``defaultPrim`` is not ``World`` (e.g.,
    ``warehouse_h10m_straight``), the material defs and binding targets
    are outside the payload scope when loaded as a payload arc.

    This function:
    1. Checks if the output.usd's defaultPrim differs from ``World``.
    2. If so, copies material specs from ``/World/Looks/...`` to
       ``/<defaultPrim>/Looks/...``.
    3. Rewrites all binding targets to the new paths.
    4. Removes the orphaned ``/World/Looks`` tree.
    """
    from pxr import Sdf
    from world_understanding.utils.usd.material import ensure_looks_scope_spec

    if not payload_group.output_usd_path:
        return

    output_path = Path(payload_group.output_usd_path)
    if not output_path.exists():
        return

    layer = Sdf.Layer.FindOrOpen(str(output_path))
    if not layer or not layer.defaultPrim:
        return

    dp = layer.defaultPrim
    if dp == "World":
        return  # Already correct

    # Need to relocate /World/Looks/* to /<defaultPrim>/Looks/*
    world_looks = "/World/Looks"
    target_looks = f"/{dp}/Looks"

    world_looks_spec = layer.GetPrimAtPath(world_looks)
    if not world_looks_spec:
        return  # No materials to relocate

    logger.info(
        f"Relocating materials in '{payload_group.group_name}': "
        f"{world_looks} -> {target_looks}"
    )

    # Ensure target parent exists
    target_looks_parent = Sdf.Path(target_looks).GetParentPath()
    while target_looks_parent != Sdf.Path.absoluteRootPath:
        if not layer.GetPrimAtPath(target_looks_parent):
            parent_spec = Sdf.CreatePrimInLayer(layer, target_looks_parent)
            if parent_spec:
                parent_spec.specifier = Sdf.SpecifierOver
        target_looks_parent = target_looks_parent.GetParentPath()

    # Copy /World/Looks to /<defaultPrim>/Looks
    Sdf.CopySpec(layer, Sdf.Path(world_looks), layer, Sdf.Path(target_looks))
    ensure_looks_scope_spec(layer, target_looks)

    # Rewrite all binding targets from /World/Looks/* to /<defaultPrim>/Looks/*
    def _rewrite_bindings(spec: Sdf.PrimSpec) -> None:
        for rel_name in list(spec.relationships.keys()):
            if "material:binding" not in rel_name:
                continue
            rel = spec.relationships[rel_name]
            targets = list(rel.targetPathList.explicitItems)
            new_targets = []
            changed = False
            for t in targets:
                t_str = str(t)
                if t_str.startswith(world_looks + "/") or t_str == world_looks:
                    new_t = target_looks + t_str[len(world_looks) :]
                    new_targets.append(Sdf.Path(new_t))
                    changed = True
                else:
                    new_targets.append(t)
            if changed:
                rel.targetPathList.explicitItems = new_targets
        for child in spec.nameChildren:
            _rewrite_bindings(spec.nameChildren[child.name])

    for root_prim in layer.rootPrims:
        _rewrite_bindings(root_prim)

    # Remove /World/Looks (now orphaned)
    world_spec = layer.GetPrimAtPath("/World")
    if world_spec:
        looks_spec = layer.GetPrimAtPath(world_looks)
        if looks_spec:
            del world_spec.nameChildren[looks_spec.name]
        # Remove /World if now empty
        if not list(world_spec.nameChildren.keys()):
            del layer.pseudoRoot.nameChildren[world_spec.name]

    layer.Save()


def _fix_representative_sublayer(payload_group: PayloadGroup) -> None:
    """Fix output.usd sublayer to point to original payload instead of representative.

    When a payload was processed via its representative file (smaller
    extraction of prototype source prims), the apply step sets output.usd's
    sublayer to the representative. For the drop-in replacement chain to
    work, it must sublayer the original payload file instead.

    The material bindings in output.usd target prototype source prim paths
    which exist in both the representative and original files, so swapping
    the sublayer preserves all bindings.

    Args:
        payload_group: Payload group with representative_path set.
    """
    from pxr import Sdf

    if not payload_group.output_usd_path or not payload_group.representative_path:
        return

    output_path = Path(payload_group.output_usd_path)
    if not output_path.exists():
        return

    layer = Sdf.Layer.FindOrOpen(str(output_path))
    if not layer:
        return

    original_payload = str(Path(payload_group.payload_file).resolve())
    representative = str(Path(payload_group.representative_path).resolve())

    # Compute relative path from output.usd to original payload
    try:
        rel_original = os.path.relpath(original_payload, str(output_path.parent))
    except ValueError:
        rel_original = original_payload

    changed = False
    new_sublayers = list(layer.subLayerPaths)
    for i, sl in enumerate(new_sublayers):
        # Resolve the sublayer path to compare
        resolved = layer.ComputeAbsolutePath(sl)
        resolved = str(Path(resolved).resolve()) if resolved else ""
        if resolved == representative:
            new_sublayers[i] = rel_original
            changed = True

    if changed:
        layer.subLayerPaths = new_sublayers
        layer.Save()
        logger.info(
            f"Fixed sublayer for '{payload_group.group_name}': "
            f"representative → original payload"
        )
    else:
        logger.debug(f"No sublayer swap needed for '{payload_group.group_name}'")


def _set_payload_output_usd(payload_group: PayloadGroup) -> None:
    """Set output_usd_path to the apply step's output.usd."""
    if not payload_group.working_dir:
        return
    working_dir = Path(payload_group.working_dir)
    output_path = working_dir / "output" / "output.usd"
    if output_path.exists():
        payload_group.output_usd_path = str(output_path)
    else:
        logger.debug(
            f"No output.usd yet for '{payload_group.group_name}': {output_path}"
        )


def _run_payloads_sequential(
    payloads: list[PayloadGroup],
    manifest: SceneManifest,
    manifest_path: Path,
    skip_steps: list[str] | None,
    only_steps: list[str] | None,
    verbose: bool,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    runtime_configs: dict[str, dict[str, Any]] | None = None,
    scene_config: dict[str, Any] | None = None,
    scene_config_dir: Path | None = None,
    payload_sibling_map: dict[str, list[str]] | None = None,
) -> tuple[int, int]:
    """Run payload groups sequentially."""
    completed = 0
    failed = 0
    total = len(payloads)

    for i, pg in enumerate(payloads, 1):
        _raise_if_cancelled(cancel_checker)
        logger.info(f"[{i}/{total}] Processing payload '{pg.group_name}'...")
        try:
            runtime_config = (runtime_configs or {}).get(pg.id)
            config_ready = True
            if runtime_config is None:
                runtime_config, config_ready = _prepare_payload_runtime_config_for_run(
                    pg,
                    scene_config,
                    scene_config_dir,
                    (payload_sibling_map or {}).get(pg.group_name),
                )
                if not config_ready:
                    pg.status = "failed"
                    failed += 1
            if config_ready:
                run_payload(
                    pg,
                    skip_steps,
                    only_steps,
                    verbose,
                    simulate,
                    material_names,
                    resume=resume,
                    from_step=from_step,
                    predict_max_workers=predict_max_workers,
                    cancel_checker=cancel_checker,
                    config_dict=runtime_config,
                )
                if pg.status == "completed":
                    completed += 1
                else:
                    failed += 1
        except Exception:
            logger.error("Unable to process one scene payload")
            pg.status = "failed"
            failed += 1

        manifest.save(manifest_path)
        _raise_if_cancelled(cancel_checker)

    return completed, failed


def _run_payloads_parallel(
    payloads: list[PayloadGroup],
    manifest: SceneManifest,
    manifest_path: Path,
    skip_steps: list[str] | None,
    only_steps: list[str] | None,
    verbose: bool,
    max_workers: int,
    simulate: bool = False,
    material_names: list[str] | None = None,
    resume: bool = False,
    from_step: str | None = None,
    predict_max_workers: int | None = None,
    cancel_checker: CancelChecker | None = None,
    runtime_configs: dict[str, dict[str, Any]] | None = None,
    scene_config: dict[str, Any] | None = None,
    scene_config_dir: Path | None = None,
    payload_sibling_map: dict[str, list[str]] | None = None,
) -> tuple[int, int]:
    """Run payload groups in parallel using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = 0
    failed = 0
    total = len(payloads)

    # Build index map for updating manifest in-place
    pg_index_map: dict[str, int] = {}
    for payload_idx, pg in enumerate(manifest.payload_groups):
        pg_index_map[pg.id] = payload_idx

    # Sort largest payloads first (by instance count) to avoid long tails
    sorted_payloads = sorted(payloads, key=lambda pg: pg.instance_count, reverse=True)

    logger.info(
        f"Launching {min(max_workers, total)} parallel workers "
        f"for {total} payload groups (sorted by instance count)"
    )

    manifest_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
        _raise_if_cancelled(cancel_checker)
        future_to_pg = {
            executor.submit(
                _run_payload_worker,
                pg,
                skip_steps,
                only_steps,
                verbose,
                simulate,
                material_names,
                resume,
                from_step,
                predict_max_workers,
                cancel_checker,
                (runtime_configs or {}).get(pg.id),
                scene_config,
                scene_config_dir,
                (payload_sibling_map or {}).get(pg.group_name),
            ): pg
            for pg in sorted_payloads
        }

        for future in as_completed(future_to_pg):
            try:
                _raise_if_cancelled(cancel_checker)
            except asyncio.CancelledError:
                for pending in future_to_pg:
                    pending.cancel()
                raise
            original_pg = future_to_pg[future]
            with manifest_lock:
                try:
                    result_pg = future.result()
                    result_payload_idx = pg_index_map.get(result_pg.id)
                    if result_payload_idx is not None:
                        manifest.payload_groups[result_payload_idx] = result_pg

                    if result_pg.status == "completed":
                        completed += 1
                        logger.info(
                            f"[{completed + failed}/{total}] "
                            f"Completed payload '{result_pg.group_name}'"
                        )
                    else:
                        failed += 1
                        logger.error(
                            f"[{completed + failed}/{total}] "
                            f"Failed payload '{result_pg.group_name}'"
                        )
                except asyncio.CancelledError:
                    for pending in future_to_pg:
                        pending.cancel()
                    raise
                except Exception:
                    failed += 1
                    logger.error("Unable to collect one scene payload result")
                    original_payload_idx = pg_index_map.get(original_pg.id)
                    if original_payload_idx is not None:
                        manifest.payload_groups[original_payload_idx].status = "failed"

                manifest.save(manifest_path)

    return completed, failed


def _update_payload_output_paths(
    payload_group: PayloadGroup, config_path: Path
) -> None:
    """Derive predictions path from pipeline output convention for a payload group.

    Payloads don't use optimize_usd or restore_usd, so predictions are
    always the raw predictions file.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    session_id = config.get("project", {}).get("session_id", "")
    if not session_id:
        session_id = payload_group.group_name

    config_dir = config_path.parent
    working_dir = config_dir / f".{session_id}"

    payload_group.working_dir = str(working_dir)

    # Payload configs disable restore_usd, so raw predictions are the output
    predictions_path = working_dir / "predictions" / "predictions.jsonl"
    if predictions_path.exists():
        payload_group.predictions_path = str(predictions_path)
