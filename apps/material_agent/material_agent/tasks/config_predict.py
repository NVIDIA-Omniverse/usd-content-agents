# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for prediction workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import log_config_source
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    drop_stale_endpoint_credentials,
    path_exists_with_safe_diagnostics,
    read_text_with_safe_diagnostics,
    redact_sensitive_path,
)

from material_agent.prompt_security import format_material_names_for_prompt
from material_agent.tasks.config_loader import load_config_from_context
from material_agent.tasks.prepare_dataset import (
    _PERSISTED_DEFAULT_SYSTEM_PROMPT_SCHEMA,
    _TRUSTED_PREPARE_SYSTEM_PROMPT_TEMPLATE_CONFIG_KEY,
    _VLM_SYSTEM_PROMPT_TEMPLATE,
    PromptTemplateConfigurationError,
    PromptTemplateTypeError,
    render_vlm_system_prompt_template,
)

logger = logging.getLogger(__name__)


class UnsafePersistedSystemPromptError(ValueError):
    """Raised when a reused dataset predates prompt trust-boundary enforcement."""


def _validate_persisted_system_prompt(
    prompt_config: object,
    *,
    trusted_custom_template: object = None,
) -> str:
    """Rebuild and completely validate a persisted system prompt."""
    if not isinstance(prompt_config, dict):
        raise UnsafePersistedSystemPromptError(
            "Unsafe system prompt record in dataset.json. Regenerate the dataset "
            "with the current material-agent version."
        )

    schema = prompt_config.get("system_prompt_schema")
    material_names = prompt_config.get("material_names")
    persisted_prompt = prompt_config.get("system_prompt")
    if (
        not isinstance(material_names, list)
        or not all(isinstance(name, str) for name in material_names)
        or not isinstance(persisted_prompt, str)
    ):
        raise UnsafePersistedSystemPromptError(
            "Unsafe system prompt record in dataset.json. Regenerate the dataset "
            "with the current material-agent version, or provide the custom prompt "
            "as an explicit trusted system_prompt in the prediction configuration."
        )

    validated_custom_template: str | None
    if trusted_custom_template is None:
        validated_custom_template = None
    elif not isinstance(trusted_custom_template, str):
        raise PromptTemplateTypeError(
            "steps.build_dataset_prepare_dataset.prompts.vlm_system",
            trusted_custom_template,
        )
    else:
        validated_custom_template = trusted_custom_template

    trusted_template: str
    if schema == _PERSISTED_DEFAULT_SYSTEM_PROMPT_SCHEMA:
        if validated_custom_template not in (None, _VLM_SYSTEM_PROMPT_TEMPLATE):
            raise UnsafePersistedSystemPromptError(
                "Persisted default system prompt does not match the configured custom "
                "prompt template. Regenerate the dataset with the current configuration."
            )
        trusted_template = _VLM_SYSTEM_PROMPT_TEMPLATE
    elif schema == "custom" and validated_custom_template is not None:
        trusted_template = validated_custom_template
    else:
        raise UnsafePersistedSystemPromptError(
            "Unsafe legacy or custom system prompt in dataset.json. Regenerate the "
            "dataset with the current material-agent version, or configure the same "
            "trusted custom prompt used to prepare the dataset."
        )

    expected_prompt: str = render_vlm_system_prompt_template(
        trusted_template,
        materials_list=format_material_names_for_prompt(
            {"name": name} for name in material_names
        ),
    )
    if persisted_prompt != expected_prompt:
        raise UnsafePersistedSystemPromptError(
            "Unsafe modified system prompt in dataset.json. Regenerate the dataset "
            "with the current material-agent version, or provide an explicit trusted "
            "system_prompt in the prediction configuration."
        )
    return expected_prompt


class PredictConfigTask(Task):
    """Compatibility config task for prediction workflows.

    Standalone workflows load YAML from ``config_path``. Unified pipelines pass
    ``config_dict`` in memory and retain ``config_path`` only as a path anchor.
    """

    def __init__(self):
        """Initialize the predict config loading task."""
        self.name = "PredictConfigLoading"
        self.description = "Load prediction configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load prediction configuration.

        Args:
            context: Workflow context containing config_dict or config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config, _ = load_config_from_context(context)
        log_config_source(context, listener.info, label="prediction")

        # Simply pass through the config - it's already been resolved
        # by UnifiedPipelineConfigTask if coming from unified system
        context["config"] = config

        # Extract key fields for backward compatibility
        context["dataset_path"] = config.get("dataset")
        context["output_dir"] = config.get("output_dir")
        vlm_config = config.get("vlm", {})
        # Inject local NIM endpoint if configured via env var.
        # Setting MA_VLM_NIM_BASE_URL forces backend=nim regardless of config
        # (same pattern as material_agent_service pipeline_router.py).
        nim_base_url = os.environ.get("MA_VLM_NIM_BASE_URL")
        vlm_backend = (vlm_config.get("backend") or "").strip().lower()
        if nim_base_url and vlm_backend not in ("", "echo", "mock"):
            if vlm_backend != "nim":
                listener.info(
                    f"MA_VLM_NIM_BASE_URL set - overriding VLM backend "
                    f"from '{vlm_config.get('backend', '')}' to 'nim'"
                )
            drop_stale_endpoint_credentials(
                vlm_config, preserve_local_nim_placeholder=True
            )
            vlm_config["backend"] = "nim"
            vlm_config["base_url"] = nim_base_url
        config["vlm"] = vlm_config

        llm_config = config.get("llm", {})
        llm_nim_base_url = os.environ.get("MA_LLM_NIM_BASE_URL")
        llm_uses_vlm_sidecar = False
        if not llm_nim_base_url:
            llm_nim_base_url = os.environ.get("MA_VLM_NIM_BASE_URL")
            llm_uses_vlm_sidecar = bool(llm_nim_base_url)
        llm_backend = (llm_config.get("backend") or "").strip().lower()
        if llm_nim_base_url and llm_backend not in ("", "echo", "mock"):
            if llm_backend != "nim":
                listener.info(
                    f"MA_LLM_NIM_BASE_URL/MA_VLM_NIM_BASE_URL set - overriding "
                    f"LLM backend from '{llm_config.get('backend', '')}' to 'nim'"
                )
            drop_stale_endpoint_credentials(
                llm_config, preserve_local_nim_placeholder=True
            )
            llm_config["backend"] = "nim"
            llm_config["base_url"] = llm_nim_base_url
            if llm_uses_vlm_sidecar and vlm_config.get("model"):
                llm_config["model"] = vlm_config["model"]
        config["llm"] = llm_config

        context["vlm_config"] = vlm_config
        context["llm_config"] = llm_config
        context["max_workers"] = config.get("max_workers", 64)
        context["prediction_batch_size"] = config.get("prediction_batch_size", 1)
        allow_empty_predictions = config.get("allow_empty_predictions", False)
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "predict.allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )
        context["allow_empty_predictions"] = allow_empty_predictions

        # Iterative harness/refine pass-through. These keys are consumed by
        # VLMInferenceTask to carry forward previous predictions and re-predict
        # only the prims that the judge explicitly flagged.
        for key in (
            "predictions_path",
            "previous_predictions_path",
            "previous_prim_feedback",
            "resolved_assignments",
            "visual_refinement_context_by_prim",
        ):
            if key in config:
                context[key] = config[key]

        # Load system prompt from multiple sources (priority order):
        # 1. Direct system_prompt in config
        # 2. A fully validated default or configured custom prompt record in
        #    dataset.json (v0.2 format)
        # 3. system_prompt_file (legacy fallback only)
        system_prompt = config.get("system_prompt")
        trusted_custom_template = config.pop(
            _TRUSTED_PREPARE_SYSTEM_PROMPT_TEMPLATE_CONFIG_KEY,
            None,
        )

        # Try loading from dataset.json (v0.2 format) first if no direct system_prompt
        if not system_prompt:
            dataset_path = config.get("dataset")
            if dataset_path:
                # Dataset path points to dataset.jsonl, get the config file
                dataset_dir = Path(dataset_path).parent
                dataset_config_path = dataset_dir / "dataset.json"

                if path_exists_with_safe_diagnostics(
                    dataset_config_path,
                    label="prediction dataset metadata",
                ):
                    try:
                        dataset_config = json.loads(
                            read_text_with_safe_diagnostics(
                                dataset_config_path,
                                label="prediction dataset metadata",
                            )
                        )

                        # Extract system prompt from v0.2 format
                        prompts = dataset_config.get("inference", {}).get("prompts", [])
                        if prompts and len(prompts) > 0:
                            prompt_config = prompts[0]
                            persisted_system_prompt = (
                                prompt_config.get("system_prompt", "")
                                if isinstance(prompt_config, dict)
                                else ""
                            )
                            if persisted_system_prompt:
                                system_prompt = _validate_persisted_system_prompt(
                                    prompt_config,
                                    trusted_custom_template=trusted_custom_template,
                                )
                                listener.info(
                                    "Loaded system prompt from dataset.json (v0.2 format)"
                                )
                                config["system_prompt"] = system_prompt
                    except (
                        UnsafePersistedSystemPromptError,
                        PromptTemplateConfigurationError,
                        PromptTemplateTypeError,
                    ):
                        raise
                    except Exception:
                        listener.warning(
                            "Failed to load system prompt from "
                            f"{redact_sensitive_path(dataset_config_path)}"
                        )

        # Legacy fallback: try system_prompt_file only if still no system prompt
        if not system_prompt:
            system_prompt_file = config.get("system_prompt_file")
            if system_prompt_file:
                # Load from file (legacy support)
                system_prompt_path = Path(system_prompt_file)
                if path_exists_with_safe_diagnostics(
                    system_prompt_path,
                    label="system prompt file",
                ):
                    system_prompt = read_text_with_safe_diagnostics(
                        system_prompt_path,
                        label="system prompt file",
                    )
                    listener.info(
                        "Loaded system prompt from file (legacy): "
                        f"{redact_sensitive_path(system_prompt_path)}"
                    )
                    config["system_prompt"] = system_prompt
                else:
                    # Only warn if we couldn't load from dataset.json either
                    # (i.e., we actually need this file)
                    listener.warning(
                        "System prompt file not found: "
                        f"{redact_sensitive_path(system_prompt_path)}. "
                        "Unable to load system prompt from either dataset.json or file. "
                        "Will use default system prompt."
                    )

        context["system_prompt"] = system_prompt

        # Extract report compression configuration if present
        report_config = config.get("report", {})
        if isinstance(report_config, dict):
            if "image_max_size" in report_config:
                context["report_image_max_size"] = report_config["image_max_size"]
            if "image_format" in report_config:
                context["report_image_format"] = report_config["image_format"]
            if "image_quality" in report_config:
                context["report_image_quality"] = report_config["image_quality"]

        return context
