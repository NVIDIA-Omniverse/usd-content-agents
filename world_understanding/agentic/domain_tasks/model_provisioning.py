# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model provisioning task for creating VLM and LLM instances.

This is a shared task used by both material-agent, physics-agent, and joint-agent.
"""

import logging
from typing import Any

from world_understanding.agentic.config import get_api_key_for_model_config
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.models.backends.registry import (
    list_chat_backends,
    list_vlm_backends,
)
from world_understanding.functions.models.chat_models import create_chat_model
from world_understanding.functions.models.vision_language_models import create_vlm
from world_understanding.utils.credentials import (
    apply_llm_nim_env_override,
    apply_vlm_nim_env_override,
)

logger = logging.getLogger(__name__)

# Inference-time parameters that should NOT be passed to model constructors.
# They are extracted from config and forwarded at inference time instead.
_INFERENCE_TIME_KEYS = {
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "max_retries",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "reasoning_effort",
}


class ModelProvisioningTask(Task):
    """Task to create model instances from configuration.

    This task creates VLM, LLM, VLM Judge, and LLM Judge instances based on the
    configuration loaded by the appropriate config task.

    Input context keys:
        - config: Configuration dictionary with model settings
            - vlm: VLM configuration for predictions
            - llm: LLM configuration for structured parsing (optional)
            - vlm_judge: VLM configuration for judge (optional)
            - llm_judge: LLM configuration for text-based judge (optional)

    Output context keys:
        - vlm: VLM instance (if configured)
        - llm: LLM instance (if configured)
        - vlm_judge: VLM Judge instance (if configured)
        - llm_judge: LLM Judge instance (if configured)
        - vlm_config: VLM configuration dict
        - llm_config: LLM configuration dict
        - vlm_judge_config: VLM Judge configuration dict (if present)
        - llm_judge_config: LLM Judge configuration dict (if present)
    """

    def __init__(self):
        """Initialize the model provisioning task."""
        self.name = "ModelProvisioning"
        self.description = "Create VLM and LLM instances from configuration"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Create model instances from configuration.

        Args:
            context: Workflow context containing config
            object_store: Optional object store for persisting models

        Returns:
            Updated context with model instances

        Raises:
            ValueError: If config is not found in context
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Get configuration from context
        if "config" not in context:
            raise ValueError(
                "'config' not found in context. "
                "Model provisioning requires a 'config' key with model settings."
            )

        config = context["config"]

        models = {}

        def _has_backend(cfg: Any) -> bool:
            """Check if a model config dict has a backend specified."""
            return bool(
                isinstance(cfg, dict) and (cfg.get("backend") or cfg.get("provider"))
            )

        # Create VLM if configured
        if "vlm" in config and _has_backend(config["vlm"]):
            listener.info("Creating VLM instance...")
            try:
                vlm = self.create_vlm(config["vlm"])
                models["vlm"] = vlm
                backend = config["vlm"].get("backend") or config["vlm"].get("provider")
                listener.info(f"✓ VLM created: {backend}")
            except Exception as e:
                listener.error(f"Failed to create VLM: {e}")
                raise
        elif "vlm" in config:
            listener.debug("VLM config present but no backend specified — skipping")

        # Create LLM if configured
        if "llm" in config and _has_backend(config["llm"]):
            listener.info("Creating LLM instance...")
            try:
                llm = self._create_llm(config["llm"])
                if llm is not None:
                    models["llm"] = llm
                    backend = config["llm"].get("backend") or config["llm"].get(
                        "provider"
                    )
                    listener.info(f"✓ LLM created: {backend}")
                else:
                    listener.info("LLM configuration is empty - skipping LLM creation")
            except Exception as e:
                listener.warning(f"Failed to create LLM: {e}")
                # LLM is optional, so we don't raise
        elif "llm" in config:
            listener.debug("LLM config present but no backend specified — skipping")

        # Create VLM Judge if configured
        if "vlm_judge" in config and _has_backend(config["vlm_judge"]):
            listener.info("Creating VLM Judge instance...")
            try:
                vlm_judge = self.create_vlm(config["vlm_judge"])
                models["vlm_judge"] = vlm_judge
                backend = config["vlm_judge"].get("backend") or config["vlm_judge"].get(
                    "provider"
                )
                listener.info(f"✓ VLM Judge created: {backend}")
            except Exception as e:
                listener.error(f"Failed to create VLM Judge: {e}")
                raise
        elif "vlm_judge" in config:
            listener.debug(
                "VLM Judge config present but no backend specified — skipping"
            )

        # Create LLM Judge if configured
        if "llm_judge" in config and _has_backend(config["llm_judge"]):
            listener.info("Creating LLM Judge instance...")
            try:
                llm_judge = self._create_llm(config["llm_judge"])
                models["llm_judge"] = llm_judge
                backend = config["llm_judge"].get("backend") or config["llm_judge"].get(
                    "provider"
                )
                listener.info(f"✓ LLM Judge created: {backend}")
            except Exception as e:
                listener.error(f"Failed to create LLM Judge: {e}")
                raise
        elif "llm_judge" in config:
            listener.debug(
                "LLM Judge config present but no backend specified — skipping"
            )

        # Note: VLM/LLM models are not stored in object_store because:
        # 1. They are not picklable (contain local classes, thread locks, etc.)
        # 2. All tasks access them from context, not object_store
        # 3. Storing in context is sufficient and avoids serialization issues

        # Build vlm_invoke_kwargs from inference-time parameters in the VLM config.
        # These are NOT passed to the VLM constructor but must reach the inference call.
        vlm_cfg = config.get("vlm", {})
        vlm_invoke_kwargs = {
            k: v
            for k, v in vlm_cfg.items()
            if k in _INFERENCE_TIME_KEYS and v is not None
        }

        # Update context with models and configurations
        context["vlm"] = models.get("vlm")
        context["llm"] = models.get("llm")
        context["vlm_judge"] = models.get("vlm_judge")
        context["llm_judge"] = models.get("llm_judge")
        context["vlm_config"] = vlm_cfg
        context["llm_config"] = config.get("llm", {})
        context["vlm_judge_config"] = config.get("vlm_judge", {})
        context["llm_judge_config"] = config.get("llm_judge", {})
        context["vlm_invoke_kwargs"] = vlm_invoke_kwargs

        return context

    def create_vlm(self, vlm_config: dict[str, Any]) -> Any:
        """Create a VLM instance from configuration.

        Args:
            vlm_config: VLM configuration dictionary

        Returns:
            VLM instance

        Raises:
            ValueError: If required configuration is missing
        """
        vlm_config = apply_vlm_nim_env_override(vlm_config)
        backend = vlm_config.get("backend") or vlm_config.get("provider")
        if not backend:
            raise ValueError("VLM backend/provider not specified")

        kwargs = {"backend": backend}

        api_key = get_api_key_for_model_config(backend, vlm_config, "VLM")
        if api_key:
            kwargs["api_key"] = api_key

        # Pass through standard config keys
        if vlm_config.get("model"):
            kwargs["model"] = vlm_config["model"]
        if vlm_config.get("base_url"):
            kwargs["base_url"] = vlm_config["base_url"]
        if vlm_config.get("endpoint"):
            kwargs["endpoint"] = vlm_config["endpoint"]
        if vlm_config.get("api_name"):
            kwargs["api_name"] = vlm_config["api_name"]

        # Pass through any additional, backend-specific kwargs from config.
        # This allows passing along custom parameters (e.g., use_single_image_api)
        known_keys = {
            "backend",
            "provider",  # Alias for backend
            "model",
            "endpoint",
            "api_key",
            "api_key_env",
            "api_name",
            "base_url",
            "llm",  # Filter out llm if accidentally included in vlm_config
        }
        # These are inference-time parameters that should NOT be passed to VLM constructor
        # They will be used at inference time via vlm_config context
        _vlm_inference_keys = _INFERENCE_TIME_KEYS | {
            "reference_images",  # Judge-specific parameter, not for model constructor
        }
        extra_kwargs = {
            k: v
            for k, v in vlm_config.items()
            if k not in known_keys and k not in _vlm_inference_keys
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        kwargs.pop(
            "reference_images", None
        )  # Judge-specific parameter, not for model constructor

        # Check backend is registered before calling create_vlm
        available = list_vlm_backends()
        if backend not in available:
            raise ValueError(
                f"VLM backend '{backend}' is not registered. "
                "You may be missing an extra package.\n"
                f"Available backends: {', '.join(available)}"
            )

        return create_vlm(**kwargs)

    def _create_vlm(self, vlm_config: dict[str, Any]) -> Any:
        """Compatibility wrapper for older internal call sites."""
        return self.create_vlm(vlm_config)

    def _create_llm(self, llm_config: dict[str, Any]) -> Any | None:
        """Create an LLM instance from configuration.

        Args:
            llm_config: LLM configuration dictionary

        Returns:
            LLM instance, or None if no backend specified (optional LLM)

        Raises:
            ValueError: If required configuration is missing
        """
        # Apply the same MA_LLM_NIM_BASE_URL / MA_VLM_NIM_BASE_URL override
        # that ``create_chat_model_from_config`` and the CLI preflight apply,
        # so judge/evaluate provisioning agrees with runtime routing.
        llm_config = apply_llm_nim_env_override(llm_config)
        backend = llm_config.get("backend") or llm_config.get("provider")
        if not backend:
            # LLM is optional - return None if no backend specified
            return None

        kwargs = {"backend": backend}

        api_key = get_api_key_for_model_config(backend, llm_config, "LLM")
        if api_key:
            kwargs["api_key"] = api_key

        # Pass through standard config keys
        if llm_config.get("model"):
            kwargs["model"] = llm_config["model"]
        if llm_config.get("base_url"):
            kwargs["base_url"] = llm_config["base_url"]

        # Pass through any additional, backend-specific kwargs from config
        # to allow custom parameters such as include_thinking, thinking, etc.
        known_keys = {
            "backend",
            "provider",  # Alias for backend
            "model",
            "base_url",
            "endpoint",
            "api_key",
            "api_key_env",
            "vlm",  # Filter out vlm if accidentally included in llm_config
        }
        # These are inference-time parameters that should NOT be passed to LLM constructor
        # They will be used at inference time via llm_config context
        _llm_inference_keys = _INFERENCE_TIME_KEYS | {
            "reference_images",  # Judge-specific parameter, not for model constructor
        }
        extra_kwargs = {
            k: v
            for k, v in llm_config.items()
            if k not in known_keys and k not in _llm_inference_keys
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        kwargs.pop(
            "reference_images", None
        )  # Judge-specific parameter, not for model constructor

        # Check backend is registered before calling create_chat_model
        available_chat = list_chat_backends()
        if backend not in available_chat:
            raise ValueError(
                f"LLM backend '{backend}' is not registered. "
                "You may be missing an extra package.\n"
                f"Available backends: "
                f"{', '.join(available_chat)}"
            )

        return create_chat_model(**kwargs)
