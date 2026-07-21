# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision-Language Model (VLM) tool for image analysis and understanding."""

import logging
from typing import Any

from pydantic import Field
from rich.console import Console

from world_understanding.functions.cv.vlm import generate_vlm_response
from world_understanding.functions.models.vision_language_models import create_vlm
from world_understanding.tools.base import (
    ExecutionPolicy,
    ToolInput,
    ToolOutput,
    register_tool,
)
from world_understanding.utils.credentials import (
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
)

logger = logging.getLogger(__name__)

# Default VLM configurations
_DEFAULT_NIM_VLM_MODEL = "qwen/qwen3.5-397b-a17b"
_DEFAULT_OPENAI_VLM_MODEL = "gpt-5.4"
_DEFAULT_ANTHROPIC_VLM_MODEL = "claude-opus-4-6"
_DEFAULT_GEMINI_VLM_MODEL = "gemini-3-pro-preview"


class VLMInput(ToolInput):
    """Input for VLM tool."""

    prompt: str = Field(..., description="User prompt/question about the image(s)")
    images: list[str] = Field(
        ...,
        description="List of image file paths to analyze (supports multiple images)",
    )
    backend: str = Field(
        default="nim",
        description=(
            "VLM backend to use (see list_vlm_backends() for available options)"
        ),
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the backend (uses env var if not provided)",
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Override the API base URL. Required when pointing at a custom "
            "OpenAI-compatible or NIM endpoint so the credential resolver "
            "can validate the explicit api_key + base_url pairing."
        ),
    )
    model: str | None = Field(
        default=None,
        description=("Model to use (backend-specific, uses default if not provided)"),
    )
    system_prompt: str = Field(
        default="You are a helpful AI assistant that can analyze images.",
        description="System instructions for the model",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation",
    )
    max_tokens: int = Field(
        default=1024, ge=1, le=8192, description="Maximum tokens in response"
    )


class VLMOutput(ToolOutput):
    """Output for VLM tool."""

    response: str = Field(..., description="Generated text response about the image(s)")
    backend_used: str = Field(..., description="VLM backend that was used")
    model_used: str | None = Field(
        default=None, description="Model that was used (if available)"
    )
    images_analyzed: int = Field(..., description="Number of images analyzed")


def _display_vlm_response(
    outputs: dict[str, Any], console: Console, indent: str = ""
) -> None:
    """Display VLM response in a formatted way."""
    console.print(f"{indent}[bold]VLM Analysis Results:[/bold]")
    console.print(f"{indent}Backend: {outputs.get('backend_used', 'unknown')}")
    if outputs.get("model_used"):
        console.print(f"{indent}Model: {outputs['model_used']}")
    console.print(f"{indent}Images Analyzed: {outputs.get('images_analyzed', 0)}")
    console.print(f"{indent}[bold]Response:[/bold]")
    console.print(f"{indent}{outputs.get('response', 'No response')}")


@register_tool(
    name="vlm",
    version="0.1.0",
    description="Analyze images using Vision-Language Models",
    tags=["vision", "language", "analysis", "gpu"],
    input_model=VLMInput,
    output_model=VLMOutput,
    policy=ExecutionPolicy(timeout_s=60.0, device="cuda"),
)
def vlm_tool(inputs: VLMInput) -> VLMOutput:
    """Analyze images using Vision-Language Models."""
    # Resolve credentials with endpoint awareness so a hosted ``OPENAI_API_KEY``
    # or ``NVIDIA_API_KEY`` is not silently forwarded to a non-provider URL
    # via ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` (OpenAI SDK env fallback)
    # or a custom NIM endpoint.
    if inputs.backend == "openai":
        api_key = get_openai_api_key_for_base_url(inputs.base_url, inputs.api_key)
    elif inputs.backend == "nim":
        api_key = get_nim_api_key_for_base_url(inputs.base_url, inputs.api_key)
    else:
        api_key = get_env_api_key_for_backend(inputs.backend, inputs.api_key)

    if not api_key:
        raise ValueError(
            f"API key required for backend '{inputs.backend}'. "
            "Set via parameter or environment variable."
        )

    # Select model based on backend
    model = inputs.model
    if not model:
        if inputs.backend == "nim":
            model = _DEFAULT_NIM_VLM_MODEL
        elif inputs.backend == "openai":
            model = _DEFAULT_OPENAI_VLM_MODEL
        elif inputs.backend == "anthropic":
            model = _DEFAULT_ANTHROPIC_VLM_MODEL
        elif inputs.backend == "gemini":
            model = _DEFAULT_GEMINI_VLM_MODEL

    # Create VLM instance
    vlm_kwargs: dict[str, Any] = {
        "backend": inputs.backend,
        "api_key": api_key,
        "model": model,
    }
    if inputs.base_url:
        vlm_kwargs["base_url"] = inputs.base_url
    vlm = create_vlm(**vlm_kwargs)

    # Call the function - it accepts list of image paths
    try:
        response = generate_vlm_response(
            vlm=vlm,
            prompt=inputs.prompt,
            images=inputs.images,  # Pass paths directly
            system_prompt=inputs.system_prompt,
            temperature=inputs.temperature,
            max_tokens=inputs.max_tokens,
        )

        return VLMOutput(
            response=response["response"],
            backend_used=inputs.backend,
            model_used=model,
            images_analyzed=len(inputs.images),
        )
    except Exception as e:
        logger.error(f"VLM generation failed: {e}")
        raise


# Attach display function to the tool
vlm_tool._display_function = _display_vlm_response
