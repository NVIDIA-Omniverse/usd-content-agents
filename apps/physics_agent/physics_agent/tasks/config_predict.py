# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict configuration task for Physics Agent."""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    path_exists_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)
from world_understanding.utils.object_store import ObjectStore

from physics_agent.api.defaults import PREDICT_DEFAULTS, apply_defaults

logger = logging.getLogger(__name__)


class PredictConfigTask(Task):
    """Load and validate prediction configuration.

    Input context keys:
        - config_path: Path to YAML config file
        OR
        - config_dict: Configuration dictionary

    Output context keys:
        - dataset: List of dataset entries
        - dataset_path: Path to dataset file
        - output_dir: Output directory for predictions
        - vlm_config: VLM configuration
        - llm_config: LLM configuration (optional)
        - system_prompt: System prompt for VLM
        - output_key: Key for classification output
    """

    def __init__(self):
        """Initialize the config task."""
        self.name = "PredictConfig"
        self.description = "Load and validate prediction configuration"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Load and validate configuration.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with configuration
        """
        raw_config = self._load_config(context)

        # Unified-config compatibility: when callers pass the same YAML file
        # they'd give to `physics-agent run` (i.e. the file has `project` and
        # `steps.predict` sections), unwrap the predict step config and merge
        # auto-wired paths derived from `project.working_dir`. This makes
        # `physics-agent predict CONFIG` (direct API path) produce the same
        # artifacts as `physics-agent run CONFIG --only predict` (unified
        # pipeline path) — see issue #42 acceptance criteria.
        if "project" in raw_config and "steps" in raw_config:
            config = self._unwrap_unified_predict_config(raw_config, context)
        else:
            config = raw_config

        # Apply defaults
        config = apply_defaults(config, PREDICT_DEFAULTS)

        # Resolve paths
        config_path = context.get("config_path")
        if config_path:
            config_dir = Path(config_path).parent
        else:
            config_dir = Path.cwd()

        # Load dataset.
        #
        # Resolution order (matches the docstring of PredictInput):
        # 1. context["dataset_override"]  — explicit override from PredictInput.
        # 2. config["dataset"]             — value baked into the YAML/dict config.
        # 3. Unified pipeline fallback     — when run as part of the unified
        #    pipeline, build_dataset_prepare_dataset writes
        #    {working_dir}/dataset/dataset.jsonl. Auto-resolve from
        #    project.working_dir so /predict in Mode B (and `run --only predict`
        #    in unified configs that omit a top-level `dataset:`) keep working.
        dataset_override = context.get("dataset_override")
        dataset_path: Path | None = None
        if dataset_override:
            # Explicit overrides come from the CLI invocation (e.g.
            # `physics-agent predict CONFIG --dataset data/foo.jsonl`).
            # Resolve those relative to the caller's cwd, not the config
            # file's directory, so a path that exists under the invocation
            # cwd isn't reported missing because the resolver looked under
            # configs/.
            dataset_path = self._resolve_path(str(dataset_override), Path.cwd())
        elif config.get("dataset"):
            dataset_path = self._resolve_path(str(config["dataset"]), config_dir)
        else:
            project_working_dir = (config.get("project") or {}).get("working_dir")
            if project_working_dir:
                candidate = (
                    self._resolve_path(str(project_working_dir), config_dir)
                    / "dataset"
                    / "dataset.jsonl"
                )
                if path_exists_with_safe_diagnostics(
                    candidate,
                    label="derived prediction dataset",
                ):
                    dataset_path = candidate

        if dataset_path is None:
            raise ValueError(
                "No dataset specified in configuration "
                "(set top-level 'dataset', pass dataset_override via "
                "PredictInput, or run after build_dataset_prepare_dataset)"
            )

        dataset = self._load_dataset(dataset_path)

        # Resolve output directory.
        #
        # Same order as dataset: explicit override > config value > derived.
        output_dir_override = context.get("output_dir_override")
        if output_dir_override:
            # CLI override — resolve relative to the caller's cwd so
            # `--output predictions/` lands under the user's pwd, not
            # under configs/.
            output_dir = self._resolve_path(str(output_dir_override), Path.cwd())
        elif config.get("output_dir"):
            output_dir = self._resolve_path(str(config["output_dir"]), config_dir)
        else:
            output_dir = dataset_path.parent / "output"
        create_directory_with_safe_diagnostics(
            output_dir,
            label="predict output directory",
            parents=True,
            exist_ok=True,
        )

        # Extract system prompt (if dataset.json exists with v0.2 format)
        system_prompt = self._extract_system_prompt(dataset_path)

        # Get output_key (configurable)
        output_key = config.get("output_key", "classification")
        allow_empty_predictions = config.get("allow_empty_predictions", False)
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "predict.allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )

        # Update context
        context["config"] = config  # Required for ModelProvisioningTask
        context.update(
            {
                "dataset": dataset,
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "image_base_dir": str(dataset_path.parent),
                "vlm_config": config.get("vlm", {}),
                "llm_config": config.get("llm", {}),
                "system_prompt": system_prompt,
                "output_key": output_key,
                "max_workers": config.get("max_workers"),
                "allow_empty_predictions": allow_empty_predictions,
                "resume": context.get("resume", False),
                "stream_predictions": context.get("stream_predictions", True),
            }
        )

        # Extract report compression configuration if present
        report_config = config.get("report", {})
        if isinstance(report_config, dict):
            if "image_max_size" in report_config:
                context["report_image_max_size"] = report_config["image_max_size"]
            if "image_format" in report_config:
                context["report_image_format"] = report_config["image_format"]
            if "image_quality" in report_config:
                context["report_image_quality"] = report_config["image_quality"]

        logger.info("Loaded configuration for prediction")
        logger.info(
            "Dataset: %s (%d entries)",
            redact_sensitive_path(dataset_path),
            len(dataset),
        )
        logger.info("Output directory: %s", redact_sensitive_path(output_dir))
        logger.info("Output key: %s", redact_sensitive_config(output_key))

        return context

    def _unwrap_unified_predict_config(
        self, unified: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Flatten a unified pipeline config into a flat predict config.

        Mirrors `UnifiedPipelineConfigTask._autowire_paths` for the predict
        step so a unified config can drive run_predict directly without the
        full pipeline executor.

        - Lifts `steps.predict.*` to the top level
        - Auto-wires `dataset` to `{working_dir}/dataset/dataset.jsonl`
          (matches `path_resolver.get_step_dataset_file('build_dataset_prepare_dataset')`)
        - Auto-wires `output_dir` to `{working_dir}/predictions`
          (matches `path_resolver.get_predictions_dir()`)

        Returns a flat dict suitable for the legacy predict config schema.
        """
        flat: dict[str, Any] = {}

        steps_section = unified.get("steps", {}) or {}
        predict_section = steps_section.get("predict", {}) or {}
        if isinstance(predict_section, dict):
            flat.update(
                {k: v for k, v in predict_section.items() if k not in ("enabled",)}
            )

        # Resolve working_dir via ProjectPathResolver so configs that omit
        # project.working_dir (and rely on the .{session_id} fallback) wire
        # dataset/output_dir the same way `physics-agent run --only predict`
        # does. Without this, the new direct path would raise "No dataset
        # specified" on every default config.
        project = unified.get("project", {}) or {}
        config_path = context.get("config_path")
        working_dir = self._derive_working_dir(unified, config_path)
        if working_dir is not None:
            flat.setdefault("dataset", str(working_dir / "dataset" / "dataset.jsonl"))
            flat.setdefault("output_dir", str(working_dir / "predictions"))

        # Preserve the `project` block too — _resolve_path() and the
        # downstream fallback logic in run() reads working_dir from there
        # for the auto-resolution branch.
        flat["project"] = project

        return flat

    def _derive_working_dir(
        self, unified: dict[str, Any], config_path: str | Path | None
    ) -> Path | None:
        """Derive the effective working directory for unified configs.

        Honors an explicit ``project.working_dir`` first; otherwise falls
        back to ``ProjectPathResolver`` so the ``.{session_id}`` default
        (used everywhere else in the pipeline) also applies to the direct
        ``physics-agent predict`` path. Returns None when nothing usable
        can be derived (e.g. a flat legacy config with no project block
        and no config file path).
        """
        project = unified.get("project", {}) or {}
        config_dir = Path(config_path).parent if config_path else Path.cwd()
        working_dir_raw = project.get("working_dir")
        if working_dir_raw:
            working_dir = Path(working_dir_raw)
            if not working_dir.is_absolute():
                working_dir = resolve_path_with_safe_diagnostics(
                    config_dir / working_dir,
                    label="predict working directory",
                )
            return working_dir

        if config_path is None:
            return None

        try:
            from physics_agent.config.path_resolver import ProjectPathResolver

            resolver = ProjectPathResolver(unified, Path(config_path))
            return Path(resolver.working_dir)
        except Exception:
            logger.debug(
                "ProjectPathResolver fallback failed for predict config; "
                "returning None for working_dir derivation"
            )
            return None

    def _load_config(self, context: dict[str, Any]) -> dict[str, Any]:
        """Load configuration from file or dict.

        Args:
            context: Workflow context

        Returns:
            Configuration dictionary
        """
        config, _ = load_config_mapping_from_context(
            context,
            allow_empty=context.get("config_dict") is not None,
            missing_path_message="No config_path or config_dict in context",
            parse_error_message="Unable to parse configuration file: {config_path}",
        )
        return config

    def _resolve_path(self, path: str, config_dir: Path) -> Path:
        """Resolve path relative to config directory.

        Args:
            path: Path string
            config_dir: Configuration directory

        Returns:
            Resolved Path
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return resolve_path_with_safe_diagnostics(
            config_dir / path_obj,
            label="predict configuration path",
        )

    def _load_dataset(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load dataset from JSONL file.

        Args:
            dataset_path: Path to dataset file

        Returns:
            List of dataset entries
        """
        safe_dataset_path = redact_sensitive_path(dataset_path)
        if not path_exists_with_safe_diagnostics(
            dataset_path,
            label="prediction dataset path",
        ):
            raise FileNotFoundError(
                f"Dataset file not found: {safe_dataset_path}"
            ) from None

        dataset = []
        try:
            with open(dataset_path, encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            dataset.append(json.loads(line))
                        except json.JSONDecodeError:
                            raise ValueError(
                                "Malformed prediction dataset JSONL at "
                                f"{safe_dataset_path}, line {line_number}"
                            ) from None
        except OSError as error:
            raise type(error)(
                error.errno,
                "Unable to read prediction dataset",
                safe_dataset_path,
            ) from None

        return dataset

    def _extract_system_prompt(self, dataset_path: Path) -> str | None:
        """Extract system prompt from dataset.json (v0.2 format).

        Args:
            dataset_path: Path to dataset JSONL file

        Returns:
            System prompt string or None
        """
        # Check for dataset.json in same directory
        dataset_json = dataset_path.parent / "dataset.json"
        if path_exists_with_safe_diagnostics(
            dataset_json,
            label="prediction dataset metadata",
        ):
            try:
                with open(dataset_json, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("system_prompt")
            except Exception:
                logger.warning(
                    "Failed to load system prompt from %s",
                    redact_sensitive_path(dataset_json),
                )

        return None
