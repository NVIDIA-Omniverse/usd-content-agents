# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for restore_usd step."""

import logging
from dataclasses import dataclass
from typing import Any, NoReturn, TextIO

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import ScalarNode, SequenceNode

from world_understanding.agentic.config import (
    load_config_mapping_from_context,
    log_config_source,
)
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.agentic.usd_tasks.optimizer_models import UsdFormat
from world_understanding.utils.credentials import redact_sensitive_config
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RestoreConfigFailure:
    error_type: type[Exception]
    args: tuple[Any, ...]
    os_error: tuple[int | None, str | None, str | None] | None = None


def _capture_restore_failure(error: Exception) -> _RestoreConfigFailure:
    os_error = (
        (error.errno, error.strerror, error.filename)
        if isinstance(error, OSError)
        else None
    )
    return _RestoreConfigFailure(type(error), error.args, os_error)


def _raise_restore_failure(failure: _RestoreConfigFailure) -> NoReturn:
    if failure.os_error is not None:
        error_number, message, filename = failure.os_error
        if filename is not None:
            raise failure.error_type(error_number, message, filename)
        if error_number is not None:
            raise failure.error_type(error_number, message)
    raise failure.error_type(*failure.args)


class _LegacyRestoreConfigLoader(yaml.SafeLoader):
    """Safe loader for explicitly allowlisted legacy optimizer values."""


def _construct_legacy_python_none(
    loader: yaml.SafeLoader,
    node: ScalarNode,
) -> None:
    """Consume a legacy ``!!python/none`` scalar without enabling object tags."""
    loader.construct_scalar(node)
    return None


_LegacyRestoreConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/none",
    _construct_legacy_python_none,
)


_LEGACY_USD_FORMAT_TAG = (
    "tag:yaml.org,2002:python/object/apply:"
    "world_understanding.agentic.usd_tasks.optimizer_models.UsdFormat"
)


def _construct_legacy_usd_format(
    loader: yaml.SafeLoader,
    node: SequenceNode,
) -> UsdFormat:
    """Decode the exact enum representation emitted by historical PyYAML."""
    del loader
    if (
        type(node) is not SequenceNode
        or len(node.value) != 1
        or type(node.value[0]) is not ScalarNode
        or node.value[0].tag != "tag:yaml.org,2002:str"
        or node.value[0].value not in {member.value for member in UsdFormat}
    ):
        raise ConstructorError(
            None,
            None,
            "invalid legacy UsdFormat value",
            node.start_mark,
        )
    return UsdFormat(node.value[0].value)


_LegacyRestoreConfigLoader.add_constructor(
    _LEGACY_USD_FORMAT_TAG,
    _construct_legacy_usd_format,
)


def _load_restore_yaml(stream: TextIO) -> Any:
    """Load current YAML or the exact safe tags used by legacy handoffs."""
    try:
        return yaml.safe_load(stream)
    except yaml.constructor.ConstructorError:
        stream.seek(0)
        return yaml.load(  # noqa: S506 - loader subclasses yaml.SafeLoader
            stream,
            Loader=_LegacyRestoreConfigLoader,
        )


class RestoreUSDConfigTask(Task):
    """Load and validate configuration for predictions restoration step.

    Input context keys:
        - config_dict: In-memory configuration dictionary (preferred)
        - config_path: Path to YAML config file (fallback)

    Output context keys:
        - original_usd_path: Path to original USD (auto-wired by executor)
        - predictions_path: Path to input predictions.jsonl (auto-wired by executor)
        - output_predictions_path: Path for restored predictions.jsonl
        - optimization_metadata: Metadata from optimize_usd (injected by executor)
    """

    def run(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """Run without retaining request configuration in public tracebacks."""
        failure: _RestoreConfigFailure | None = None
        try:
            return self._run_impl(context, object_store)
        except (OSError, ValueError) as error:
            failure = _capture_restore_failure(error)

        del context, object_store
        assert failure is not None
        _raise_restore_failure(failure)

    def _run_impl(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """Load restoration configuration.

        Args:
            context: Workflow context with config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with configuration values

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If required fields are missing
        """
        listener = get_listener(context)

        config = self._load_config(context, listener)

        # Validate required fields
        if "original_usd_path" not in config:
            raise ValueError("original_usd_path is required in restore_usd config")
        if "predictions_path" not in config:
            raise ValueError("predictions_path is required in restore_usd config")
        if "output_predictions_path" not in config:
            raise ValueError(
                "output_predictions_path is required in restore_usd config"
            )
        if "optimization_metadata" not in config:
            raise ValueError("optimization_metadata is required in restore_usd config")

        # Extract paths (already resolved by UnifiedPipelineConfigTask)
        context["original_usd_path"] = config["original_usd_path"]
        context["predictions_path"] = config["predictions_path"]
        context["output_predictions_path"] = config["output_predictions_path"]
        context["optimization_metadata"] = config["optimization_metadata"]

        safe_paths = redact_sensitive_config(
            {
                "original_usd_path": str(context["original_usd_path"]),
                "predictions_path": str(context["predictions_path"]),
                "output_predictions_path": str(context["output_predictions_path"]),
            }
        )
        listener.info(f"Original USD: {safe_paths['original_usd_path']}")
        listener.info(f"Input predictions: {safe_paths['predictions_path']}")
        listener.info(f"Output predictions: {safe_paths['output_predictions_path']}")

        return context

    def _load_config(self, context: dict[str, Any], listener: Any) -> dict[str, Any]:
        """Load config without retaining request values in public tracebacks."""
        failure: _RestoreConfigFailure | None = None
        try:
            return self._load_config_impl(context, listener)
        except (OSError, ValueError) as error:
            failure = _capture_restore_failure(error)

        del context, listener
        assert failure is not None
        _raise_restore_failure(failure)

    def _load_config_impl(
        self, context: dict[str, Any], listener: Any
    ) -> dict[str, Any]:
        """Load an isolated mapping without rendering configuration values."""
        config, _ = load_config_mapping_from_context(
            context,
            missing_path_message="config_dict or config_path is required in context",
            missing_file_message="Configuration file not found: {config_path}",
            read_error_message=(
                "Unable to read restore_usd configuration file: {config_path}"
            ),
            parse_error_message=(
                "Unable to parse restore_usd configuration file: {config_path}"
            ),
            empty_message=(
                "Empty restore_usd configuration dictionary"
                if context.get("config_dict") is not None
                else "Empty configuration file: {config_path}"
            ),
            config_dict_non_mapping_message=(
                "restore_usd config_dict must be a mapping"
            ),
            file_non_mapping_message=("restore_usd configuration must be a mapping"),
            file_loader=_load_restore_yaml,
        )
        log_config_source(context, listener.info, label="restore_usd")
        return config
