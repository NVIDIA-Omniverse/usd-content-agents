# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify asset task for the Physics Agent pipeline.

Runs VLM inference on composition images to identify the whole asset
(type, subtype, description) before per-component classification.
"""

import concurrent.futures as _cf
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.llm_parsing import extract_json_from_llm_response
from world_understanding.utils.object_store import ObjectStore

from physics_agent.api.defaults import DEFAULT_VLM_TEMPERATURE

logger = logging.getLogger(__name__)

# Hard deadline for the VLM call in identify_asset. ChatNVIDIA's `timeout`
# kwarg is silently forwarded to model_kwargs and does NOT set an HTTP
# timeout, so a slow/hung NIM endpoint would otherwise block the pipeline
# forever. Override via PA_IDENTIFY_ASSET_VLM_TIMEOUT env var.
_IDENTIFY_ASSET_VLM_TIMEOUT = float(
    os.environ.get("PA_IDENTIFY_ASSET_VLM_TIMEOUT", "180")
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

    def __init__(self):
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

        try:
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

            # Wrap the VLM call in a hard deadline. ChatNVIDIA's timeout
            # kwarg does not set an HTTP timeout, so on a slow/hung NIM
            # endpoint this call could otherwise block indefinitely.
            def _do_generate() -> str:
                return vlm.generate(
                    prompt=user_prompt,
                    images=images_to_use,
                    system_prompt=system_prompt if system_prompt else None,
                    temperature=vlm_invoke_kwargs.get(
                        "temperature",
                        vlm_config.get("temperature", DEFAULT_VLM_TEMPERATURE),
                    ),
                    max_tokens=vlm_invoke_kwargs.get("max_tokens", 4096),
                )

            _executor = _cf.ThreadPoolExecutor(max_workers=1)
            _fut = _executor.submit(_do_generate)
            try:
                response_text = _fut.result(timeout=_IDENTIFY_ASSET_VLM_TIMEOUT)
            except _cf.TimeoutError as _te:
                _fut.cancel()
                raise TimeoutError(
                    f"VLM did not respond within {_IDENTIFY_ASSET_VLM_TIMEOUT:.0f}s"
                ) from _te
            finally:
                _executor.shutdown(wait=False, cancel_futures=True)

            # Parse JSON from response
            identification = self._parse_identification(response_text)

            listener.info(
                f"Identified asset: {identification.get('asset_type', 'unknown')} "
                f"/ {identification.get('asset_subtype', 'unknown')} "
                f"(confidence: {identification.get('confidence', 'unknown')})"
            )

        except Exception as e:
            logger.error("Asset identification failed: %s", e, exc_info=True)
            identification = {
                "asset_type": "unknown",
                "asset_subtype": "unknown",
                "asset_description": f"Identification failed: {e}",
                "confidence": "low",
                "reasoning": str(e),
            }

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

        # Fallback: return raw text as description
        logger.warning("Could not parse identification JSON from VLM response")
        return {
            "asset_type": "unknown",
            "asset_subtype": "unknown",
            "asset_description": text[:500],
            "confidence": "low",
            "reasoning": "Could not parse structured response",
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
