# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict configuration task for Joint Agent."""

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

from joint_agent.api.defaults import PREDICT_DEFAULTS, apply_defaults

logger = logging.getLogger(__name__)


class DatasetPromptConfigError(ValueError):
    """Raised when dataset.json prompt metadata is invalid or ambiguous."""


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

    def __init__(self) -> None:
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
        config = self._load_config(context)

        # Apply defaults
        config = apply_defaults(config, PREDICT_DEFAULTS)

        # Resolve paths
        config_path = context.get("config_path")
        if config_path:
            config_dir = Path(config_path).parent
        else:
            config_dir = Path.cwd()

        # Load dataset
        dataset_path = config.get("dataset")
        if dataset_path:
            dataset_path = self._resolve_path(dataset_path, config_dir)
            dataset = self._load_dataset(dataset_path)
        else:
            raise ValueError("No dataset specified in configuration")

        # Resolve output directory
        output_dir = config.get("output_dir")
        if output_dir:
            output_dir = self._resolve_path(output_dir, config_dir)
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
                "completion_retries": config.get("completion_retries", 3),
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
            config_dict_non_mapping_message="config_dict must be a dictionary",
            file_non_mapping_message=("Configuration file must contain a dictionary"),
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
        """Extract and validate the system prompt from dataset.json.

        Args:
            dataset_path: Path to dataset JSONL file

        Returns:
            System prompt string or None

        Raises:
            DatasetPromptConfigError: If prompt metadata is malformed or ambiguous
        """
        dataset_json = dataset_path.parent / "dataset.json"
        if not path_exists_with_safe_diagnostics(
            dataset_json,
            label="prediction dataset metadata",
        ):
            return None

        try:
            with open(dataset_json, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise DatasetPromptConfigError(
                "Invalid prompt configuration in "
                f"{redact_sensitive_path(dataset_json)}: malformed JSON "
                f"at line {exc.lineno}, column {exc.colno}"
            ) from None
        except OSError as error:
            raise type(error)(
                error.errno,
                "Unable to read prediction dataset metadata",
                redact_sensitive_path(dataset_json),
            ) from None

        return self._validate_system_prompt_metadata(data, dataset_json)

    def _validate_system_prompt_metadata(
        self, data: Any, dataset_json: Path
    ) -> str | None:
        """Validate canonical and legacy prompt metadata without ambiguity."""
        safe_dataset_json = redact_sensitive_path(dataset_json)
        if not isinstance(data, dict):
            raise DatasetPromptConfigError(
                f"Invalid prompt configuration in {safe_dataset_json}: "
                "the document root must be an object"
            )

        has_legacy_prompt = "system_prompt" in data
        legacy_prompt = data.get("system_prompt")
        if has_legacy_prompt and not isinstance(legacy_prompt, str):
            raise DatasetPromptConfigError(
                f"Invalid prompt configuration in {safe_dataset_json}: legacy "
                'top-level "system_prompt" must be a string'
            )

        if "inference" not in data:
            return legacy_prompt

        inference = data["inference"]
        if not isinstance(inference, dict):
            raise DatasetPromptConfigError(
                f"Invalid prompt configuration in {safe_dataset_json}: "
                '"inference" must be an object'
            )

        prompts = inference.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise DatasetPromptConfigError(
                f"Invalid prompt configuration in {safe_dataset_json}: "
                '"inference.prompts" must be a non-empty list'
            )

        seen_indices: dict[int, int] = {}
        seen_names: dict[str, int] = {}
        nested_prompt: str | None = None
        for position, prompt_entry in enumerate(prompts):
            entry_path = f"inference.prompts[{position}]"
            if not isinstance(prompt_entry, dict):
                raise DatasetPromptConfigError(
                    f"Invalid prompt configuration in {safe_dataset_json}: "
                    f'"{entry_path}" must be an object'
                )

            step_name = prompt_entry.get("step_name")
            if not isinstance(step_name, str) or not step_name.strip():
                raise DatasetPromptConfigError(
                    f"Invalid prompt configuration in {safe_dataset_json}: "
                    f'"{entry_path}.step_name" must be a non-empty string'
                )

            step_index = prompt_entry.get("step_index")
            if (
                not isinstance(step_index, int)
                or isinstance(step_index, bool)
                or step_index < 0
            ):
                raise DatasetPromptConfigError(
                    f"Invalid prompt configuration in {safe_dataset_json}: "
                    f'"{entry_path}.step_index" must be a non-negative integer'
                )

            system_prompt = prompt_entry.get("system_prompt")
            if not isinstance(system_prompt, str):
                raise DatasetPromptConfigError(
                    f"Invalid prompt configuration in {safe_dataset_json}: "
                    f'"{entry_path}.system_prompt" must be a string'
                )

            if step_index in seen_indices:
                first_position = seen_indices[step_index]
                raise DatasetPromptConfigError(
                    f"Invalid prompt configuration in {safe_dataset_json}: duplicate "
                    f"step_index {step_index} at {entry_path}; already used at "
                    f"inference.prompts[{first_position}]"
                )

            normalized_name = step_name.strip()
            if normalized_name in seen_names:
                first_position = seen_names[normalized_name]
                raise DatasetPromptConfigError(
                    f"Invalid prompt configuration in {safe_dataset_json}: duplicate "
                    f"step_name at {entry_path}; already used at "
                    f"inference.prompts[{first_position}]"
                )

            if step_index != position:
                raise DatasetPromptConfigError(
                    f"Invalid prompt configuration in {safe_dataset_json}: "
                    f'"{entry_path}.step_index" must be {position} to match its '
                    "list position"
                )

            seen_indices[step_index] = position
            seen_names[normalized_name] = position
            if position == 0:
                nested_prompt = system_prompt

        if nested_prompt is None:
            raise DatasetPromptConfigError(
                f"Invalid prompt configuration in {safe_dataset_json}: "
                '"inference.prompts[0].system_prompt" is required'
            )

        if has_legacy_prompt and nested_prompt != legacy_prompt:
            raise DatasetPromptConfigError(
                f"Invalid prompt configuration in {safe_dataset_json}: "
                '"inference.prompts[0].system_prompt" conflicts with legacy '
                'top-level "system_prompt"'
            )

        return nested_prompt
