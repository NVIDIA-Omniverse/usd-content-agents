# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable function for vision-language model (VLM) response generation."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from world_understanding.functions.models.vision_language_models import (
    BaseVisionLanguageModel,
    create_vlm,
)
from world_understanding.utils.credentials import get_env_api_key_for_backend


def generate_vlm_response(
    vlm: BaseVisionLanguageModel,
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant.",
    images: list[str | Path | Image.Image | np.ndarray] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generate text response using provided VLM with image inputs.

    Args:
        vlm: Initialized BaseVisionLanguageModel instance
        prompt: User prompt/question
        system_prompt: System instructions for the model
        images: List of images - can be:
                - File paths (str or Path)
                - PIL Image objects
                - NumPy arrays
                - None (for text-only prompts)
        kwargs: Additional arguments to pass to the VLM

    Returns:
        Dict containing:
            - response: Generated text response
            - error: Error message (only if an error occurred)
    """
    try:
        response = vlm.generate(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            **kwargs,
        )
        return {"response": response}
    except Exception as e:
        return {"error": f"Failed to generate response: {e}"}


def create_vlm_instance(
    backend: str,
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    api_name: str | None = None,
    **extra_kwargs: Any,
) -> BaseVisionLanguageModel:
    """Create a VLM instance with the specified backend and configuration.

    Args:
        backend: Registered backend for VLM creation
        model: VLM model name for backends that accept a model
        api_key: API key for the VLM backend (uses env var if None)
        endpoint: Server endpoint (for 'gradio')
        api_name: API endpoint name (for 'gradio')

    Returns:
        VLM instance

    Raises:
        ValueError: If required parameters are not set
    """
    kwargs: dict[str, Any] = {"backend": backend, **extra_kwargs}

    if model:
        kwargs["model"] = model

    if backend == "gradio":
        # For Gradio, use endpoint and api_name instead of api_key and model.
        if endpoint:
            kwargs["endpoint"] = endpoint
        if api_name:
            kwargs["api_name"] = api_name
    elif backend in {"nim", "openai"}:
        # NIM and OpenAI-compatible factories are endpoint-aware. Let them
        # decide whether env credentials are safe for any supplied base_url.
        if api_key is not None:
            kwargs["api_key"] = api_key
    else:
        resolved_api_key = get_env_api_key_for_backend(backend, api_key)
        if resolved_api_key:
            kwargs["api_key"] = resolved_api_key

    # Create and return VLM instance
    return create_vlm(**kwargs)


def get_image_caption(
    image: (
        str
        | Path
        | Image.Image
        | np.ndarray
        | list[str | Path | Image.Image | np.ndarray]
    ),
    caption_prompt: str = "Describe this image in detail.",
    system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
    vlm_backend: str = "nim",
    vlm_model: str | None = None,
    vlm_api_key: str | None = None,
) -> str:
    """Get a caption for an image using VLM.

    Args:
        image: Single image or List of images to caption (single caption for multiple images)
            either path or PIL Image object
        caption_prompt: Prompt to use for image captioning
        system_prompt: System instructions for the VLM
        vlm_backend: Registered VLM backend to use
        vlm_model: Model to use (uses backend default if None)
        vlm_api_key: API key for the VLM backend (uses env var if None)

    Returns:
        Caption for the images

    Raises:
        ValueError: If the image cannot be processed
        RuntimeError: If VLM captioning fails
    """
    if isinstance(image, str | Path | Image.Image | np.ndarray):
        images = [image]
    else:
        images = image

    return generate_vlm_response(
        create_vlm_instance(backend=vlm_backend, model=vlm_model, api_key=vlm_api_key),
        caption_prompt,
        system_prompt,
        images,
    )["response"]
