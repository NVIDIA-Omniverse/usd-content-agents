# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify asset task for the Physics Agent pipeline.

Runs VLM inference on composition images to identify the whole asset
(type, subtype, description) before per-component classification.
"""

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.llm_parsing import extract_json_from_llm_response
from world_understanding.utils.model_auth import raise_for_model_authentication
from world_understanding.utils.model_timeout import (
    NON_RETRYABLE_VLM_TIMEOUT_MESSAGE,
    NonRetryableVLMTimeoutError,
    is_model_timeout_error,
)
from world_understanding.utils.object_store import ObjectStore

from physics_agent.api.defaults import DEFAULT_VLM_TEMPERATURE
from physics_agent.functions.inference import get_fibonacci_delay

logger = logging.getLogger(__name__)

_IDENTIFY_ASSET_VLM_MAX_RETRIES = 3
_UNBOUNDED_IDENTIFY_VLM_MESSAGE = (
    "Asset identification requires a VLM backend with a verified bounded "
    "request timeout"
)


class IdentifyAssetTask(Task):
    """Run VLM inference on composition images to identify the asset.

    Input context keys:
        - composition_images or rendered_preview_paths: List of image paths
        - vlm: VLM instance (from model provisioning)
        - llm: LLM instance (optional)
        - identify_system_prompt: System prompt for identification
        - output_dir: Output directory for results

    Output context keys:
        - identification: Dict with asset_type, asset_subtype, etc.
        - identification_path: Path to identification.json
    """

    def __init__(self) -> None:
        """Initialize the identify asset task."""
        self.name = "IdentifyAsset"
        self.description = "Identify whole asset from composition images"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Run asset identification via VLM.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with identification results
        """
        listener = get_listener(context, logger_name=__name__)

        vlm = context.get("vlm")
        composition_images = context.get("composition_images") or context.get(
            "rendered_preview_paths", []
        )
        system_prompt = context.get("identify_system_prompt", "")
        output_dir = Path(context.get("output_dir", "."))

        if vlm is None:
            raise ValueError("VLM not provided in context")

        if not composition_images:
            logger.warning("No composition images available for identification")
            identification = {
                "asset_type": "unknown",
                "asset_subtype": "unknown",
                "asset_description": "No composition images available",
                "confidence": "low",
                "reasoning": "No images to analyze",
            }
            self._save_identification(identification, output_dir)
            context["identification"] = identification
            context["identification_path"] = str(output_dir / "identification.json")
            return context

        listener.info(
            f"Identifying asset from {len(composition_images)} composition images"
        )

        # Limit to a reasonable number of images
        images_to_use = composition_images[:6]

        # Build the user prompt
        user_prompt = (
            "What is this 3D object? Analyze the composition views and identify "
            "what this object is.\n\n"
            "Respond with JSON:\n"
            '{"asset_type": "category (e.g., vehicle, tool, appliance, robot, '
            'furniture, industrial_equipment)", '
            '"asset_subtype": "specific type (e.g., forklift, drill, sedan)", '
            '"asset_description": "brief description of the object", '
            '"confidence": "high/medium/low", '
            '"reasoning": "explanation of identification"}'
        )

        # Get invoke kwargs (temperature, max_tokens, etc.)
        raw_vlm_invoke_kwargs = context.get("vlm_invoke_kwargs") or {}
        vlm_invoke_kwargs: dict[str, Any] = (
            dict(raw_vlm_invoke_kwargs)
            if isinstance(raw_vlm_invoke_kwargs, Mapping)
            else {}
        )
        raw_vlm_config = context.get("vlm_config") or {}
        vlm_config: dict[str, Any] = (
            dict(raw_vlm_config) if isinstance(raw_vlm_config, Mapping) else {}
        )
        max_retries = vlm_config.get("max_retries", _IDENTIFY_ASSET_VLM_MAX_RETRIES)
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 1
        ):
            raise ValueError(
                "identify_asset.vlm.max_retries must be a positive integer"
            )

        if getattr(vlm, "has_bounded_request_timeout", False) is not True:
            raise RuntimeError(_UNBOUNDED_IDENTIFY_VLM_MESSAGE)

        # Provisioned VLM backends own their timeout behavior. Keep attempts
        # synchronous so a retry cannot start until the previous request has
        # terminated; abandoning a running call in a worker leaks the request.
        extra_vlm_invoke_kwargs = {
            key: value
            for key, value in vlm_invoke_kwargs.items()
            if key
            not in {
                "max_completion_tokens",
                "max_retries",
                "max_tokens",
                "temperature",
            }
        }

        def _do_generate() -> Any:
            return vlm.generate(
                prompt=user_prompt,
                images=images_to_use,
                system_prompt=system_prompt if system_prompt else None,
                temperature=vlm_invoke_kwargs.get(
                    "temperature",
                    vlm_config.get("temperature", DEFAULT_VLM_TEMPERATURE),
                ),
                max_tokens=vlm_invoke_kwargs.get("max_tokens", 4096),
                **extra_vlm_invoke_kwargs,
            )

        response_text: str | None = None
        for attempt in range(max_retries):
            try:
                response_text = _do_generate()
                if not isinstance(response_text, str) or not response_text.strip():
                    raise RuntimeError("VLM returned an empty response")
                break
            except Exception as error:
                # Authentication errors are not transient and retain the
                # shared stable public diagnostic.
                raise_for_model_authentication(error)
                if is_model_timeout_error(error):
                    logger.error(
                        "Asset identification VLM timed out with unverified "
                        "remote completion; suppressing retries"
                    )
                    if isinstance(error, NonRetryableVLMTimeoutError):
                        raise
                    raise NonRetryableVLMTimeoutError(
                        NON_RETRYABLE_VLM_TIMEOUT_MESSAGE
                    ) from None
                if attempt == max_retries - 1:
                    logger.error(
                        "Asset identification VLM failed after %d attempts",
                        max_retries,
                    )
                    raise RuntimeError(
                        f"Asset identification VLM failed after {max_retries} attempts"
                    ) from None
                retry_delay = get_fibonacci_delay(attempt, base_delay=1.0)
                logger.warning(
                    "Asset identification VLM attempt %d/%d failed; retrying "
                    "in %.1f seconds",
                    attempt + 1,
                    max_retries,
                    retry_delay,
                )
                time.sleep(retry_delay)

        assert response_text is not None
        identification = self._parse_identification(response_text)

        listener.info(
            f"Identified asset: {identification.get('asset_type', 'unknown')} "
            f"/ {identification.get('asset_subtype', 'unknown')} "
            f"(confidence: {identification.get('confidence', 'unknown')})"
        )

        # Save results
        self._save_identification(identification, output_dir)

        # Update context
        context["identification"] = identification
        context["identification_path"] = str(output_dir / "identification.json")

        return context

    def _parse_identification(self, response_text: str) -> dict[str, Any]:
        """Parse VLM response into identification dict.

        Handles JSON embedded in markdown code blocks, <answer> tags, or raw JSON.
        """
        text = response_text.strip()
        result = extract_json_from_llm_response(
            text,
            expected_keys=["asset_type", "asset_subtype"],
        )
        if isinstance(result, dict):
            result.setdefault("asset_type", "unknown")
            result.setdefault("asset_subtype", "unknown")
            result.setdefault("asset_description", "")
            result.setdefault("confidence", "medium")
            result.setdefault("reasoning", "")
            return result

        # Fallback: return raw text as description, flagged machine-readably
        # so downstream scoring can exclude or annotate the row.
        logger.warning("Could not parse identification JSON from VLM response")
        return {
            "asset_type": "unknown",
            "asset_subtype": "unknown",
            "asset_description": text[:500],
            "confidence": "low",
            "reasoning": "Could not parse structured response",
            "identification_failed": True,
        }

    def _save_identification(
        self, identification: dict[str, Any], output_dir: Path
    ) -> None:
        """Save identification results to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "identification.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(identification, f, indent=2, ensure_ascii=False)
        logger.info("Saved identification to %s", output_path)
