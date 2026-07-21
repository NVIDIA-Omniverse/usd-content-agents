# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image generation model implementations.

This module provides interfaces for image generation models that take text prompts
and optional conditioning images (reference, depth, segmentation, etc.) and generate
output images.
"""

import base64
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from world_understanding.utils.credentials import (
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
)
from world_understanding.utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)

# Default configurations
_DEFAULT_GEMINI_MODEL = "gemini-3-pro-image-preview"
_DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1"
_DEFAULT_NIM_IMAGE_MODEL = "black-forest-labs/flux_2-klein-4b"
_DEFAULT_NIM_IMAGE_BASE_URL = "https://ai.api.nvidia.com/v1/genai"
_DEFAULT_TIMEOUT_SECONDS = 120.0


class BaseImageGenerationModel(ABC):
    """Base class for image generation models.

    Image generation models take text prompts and optional conditioning images
    and generate output images. Unlike VLMs which return text, these models
    return PIL Images.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image from text prompt and optional conditioning images.

        Args:
            prompt: Text prompt describing the desired output
            images: Optional conditioning images (reference, depth, segmentation, etc.)
            **kwargs: Model-specific parameters (temperature, etc.)

        Returns:
            Generated PIL Image
        """
        pass

    @abstractmethod
    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image from interleaved image-prompt pairs.

        This method supports workflows where each conditioning image has an
        associated description (e.g., "This is the target image", "This is the
        depth map for shape retention").

        Args:
            image_prompt_pairs: List of (description, image) tuples where each
                description introduces or describes its corresponding image
            final_prompt: Final generation instruction after all images
            **kwargs: Model-specific parameters

        Returns:
            Generated PIL Image
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of the backend being used."""
        pass

    @property
    def supports_image_conditioning(self) -> bool:
        """Whether ``generate(images=...)`` respects the conditioning images.

        Most backends either run an img2img model or fold the reference
        images into a multimodal prompt. A handful of endpoints (notably
        the cloud NIM GenAI endpoint) are text-only and silently drop any
        provided images. Callers that build multi-pass pipelines whose
        coherence depends on conditioning (e.g. PBR albedo → normal →
        roughness) can probe this flag to degrade gracefully instead of
        paying for discarded round-trips.

        Defaults to ``True``; override to ``False`` in subclasses whose
        underlying endpoint does not accept reference images.
        """
        return True

    def _load_image(
        self, image: str | Path | PILImage.Image | np.ndarray
    ) -> PILImage.Image:
        """Load image from various input formats.

        Args:
            image: Image as file path, PIL Image, or numpy array

        Returns:
            PIL Image object
        """
        if isinstance(image, str | Path):
            return PILImage.open(image).convert("RGB")
        elif isinstance(image, PILImage.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return PILImage.fromarray(image).convert("RGB")
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, PIL Image, or numpy array."
            )


class GeminiImageGenerationModel(BaseImageGenerationModel):
    """Google Gemini image generation model.

    This model uses Google's Gemini API for image generation tasks including
    style transfer, image editing, and conditional image generation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_GEMINI_MODEL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize Gemini image generation model.

        Args:
            api_key: Google API key (loads from GOOGLE_API_KEY or GEMINI_API_KEY
                env var if None)
            model: Model name (default: gemini-3-pro-image-preview)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options

        Raises:
            ImportError: If google-genai is not installed
        """
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for GeminiImageGenerationModel. "
                "Install with: pip install google-genai"
            ) from e

        api_key = get_env_api_key_for_backend("gemini", api_key)
        if api_key is None:
            raise ValueError(
                "API key is required. Provide via api_key parameter or "
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        self.client = genai.Client(api_key=api_key)
        self._model_name = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image using Gemini.

        Args:
            prompt: Text prompt describing the desired output
            images: Optional list of conditioning images
            **kwargs: Additional arguments to pass to the API

        Returns:
            Generated PIL Image

        Raises:
            ValueError: If no image is generated in the response
        """
        # Build contents list with interleaved text and images
        contents: list[str | PILImage.Image] = [prompt]

        if images:
            for img in images:
                pil_img = self._load_image(img)
                contents.append(pil_img)

        request_kwargs = self._normalize_generate_content_kwargs(kwargs)

        # Call Gemini API
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,  # type: ignore[arg-type]
            **request_kwargs,
        )

        image = self._extract_inline_image(response)
        if image is not None:
            return image

        raise ValueError("No image generated in response")

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image with interleaved image-prompt pairs.

        This method is particularly useful for multi-conditioning workflows where
        each conditioning image needs a description (e.g., style transfer with
        depth and segmentation guidance).

        Args:
            image_prompt_pairs: List of (description, image) tuples
            final_prompt: Final generation instruction
            **kwargs: Additional arguments to pass to the API

        Returns:
            Generated PIL Image

        Raises:
            ValueError: If no image is generated in the response

        Examples:
            >>> model = GeminiImageGenerationModel()
            >>> generated = model.generate_with_image_prompt_pairs(
            ...     image_prompt_pairs=[
            ...         ("Target image to apply materials to:", target_img),
            ...         ("Depth map for shape retention:", depth_img),
            ...         ("Segmentation for material boundaries:", seg_img),
            ...     ],
            ...     final_prompt="Apply realistic materials matching the style.",
            ... )
        """
        # Build contents with interleaved descriptions and images
        contents: list[str | PILImage.Image] = []

        for description, img in image_prompt_pairs:
            contents.append(description)
            pil_img = self._load_image(img)
            contents.append(pil_img)

        # Add final prompt
        contents.append(final_prompt)

        request_kwargs = self._normalize_generate_content_kwargs(kwargs)

        # Call Gemini API
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,  # type: ignore[arg-type]
            **request_kwargs,
        )

        image = self._extract_inline_image(response)
        if image is not None:
            return image

        raise ValueError("No image generated in response")

    @staticmethod
    def _normalize_generate_content_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate common chat-style kwargs to google-genai config kwargs."""
        normalized = dict(kwargs)
        raw_config = normalized.pop("config", None)
        if raw_config is None:
            raw_config = {}
        config_is_mapping = isinstance(raw_config, Mapping)
        config = dict(raw_config) if config_is_mapping else raw_config

        def set_config_default(key: str, value: Any) -> None:
            if config_is_mapping:
                config.setdefault(key, value)
            elif getattr(config, key, None) is None:
                setattr(config, key, value)

        max_tokens = normalized.pop("max_tokens", None)
        max_completion_tokens = normalized.pop("max_completion_tokens", None)
        max_output_tokens = (
            max_tokens if max_tokens is not None else max_completion_tokens
        )
        if max_output_tokens is not None:
            set_config_default("max_output_tokens", max_output_tokens)

        for key in ("temperature", "top_p", "top_k", "candidate_count"):
            if key not in normalized:
                continue
            value = normalized.pop(key)
            set_config_default(key, value)

        if (config_is_mapping and config) or (not config_is_mapping and raw_config):
            normalized["config"] = config
        return normalized

    @staticmethod
    def _extract_inline_image(response: Any) -> PILImage.Image | None:
        """Extract the first inline image from a google-genai response."""
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None:
                    return PILImage.open(BytesIO(inline_data.data))  # type: ignore[arg-type]
        return None

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "gemini"


class OpenAICompatibleChatImageGenerationModel(BaseImageGenerationModel):
    """Image generation through an OpenAI-compatible chat endpoint.

    The model is called via OpenAI ``chat.completions.create``. Input images
    are sent as ``image_url`` content parts (base64 data URIs) and the
    generated image is extracted from the assistant response content parts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        backend_name: str = "openai_compatible",
        **kwargs: Any,
    ) -> None:
        """Initialise the OpenAI-compatible image generation model.

        Args:
            api_key: Endpoint-scoped API key.
            model: Model identifier.
            base_url: API base URL.
            timeout: Request timeout in seconds.
            **kwargs: Reserved for future options.
        """
        if not base_url:
            raise ValueError(
                "base_url is required for OpenAICompatibleChatImageGenerationModel"
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAICompatibleChatImageGenerationModel. "
                "Install with: pip install openai"
            ) from e

        if not api_key:
            raise ValueError("An endpoint-scoped api_key is required")

        self._model_name = model or _DEFAULT_GEMINI_MODEL
        self._backend_name = backend_name
        self._base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate an image from a text prompt and optional conditioning images.

        Args:
            prompt: Text prompt describing the desired output.
            images: Optional list of conditioning / reference images.
            **kwargs: Extra arguments forwarded to the API call
                (e.g. ``temperature``, ``max_tokens``).

        Returns:
            Generated PIL Image.
        """
        content = self._build_content(prompt, images)
        return self._call_and_extract_image(content, **kwargs)

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate an image from interleaved (description, image) pairs.

        Args:
            image_prompt_pairs: List of ``(description, image)`` tuples.
            final_prompt: Final generation instruction appended after all pairs.
            **kwargs: Extra arguments forwarded to the API call.

        Returns:
            Generated PIL Image.
        """
        content: list[dict[str, Any]] = []
        for description, img in image_prompt_pairs:
            content.append({"type": "text", "text": description})
            pil_img = self._load_image(img)
            b64 = image_to_base64(pil_img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})
        return self._call_and_extract_image(content, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return self._backend_name

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_content(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None,
    ) -> list[dict[str, Any]]:
        """Build an OpenAI-style content list from prompt + images."""
        content: list[dict[str, Any]] = []
        content.append({"type": "text", "text": prompt})
        if images:
            for img in images:
                pil_img = self._load_image(img)
                b64 = image_to_base64(pil_img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        return content

    def _call_and_extract_image(
        self,
        content: list[dict[str, Any]],
        **kwargs: Any,
    ) -> PILImage.Image:
        """Send chat completion request and extract generated image.

        The assistant response may contain a mix of text and image parts.
        We scan for the first image part and return it.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": content},
        ]

        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        # Forward caller kwargs (temperature, max_tokens, etc.)
        for key, value in kwargs.items():
            if value is not None:
                request_kwargs[key] = value

        logger.info(
            "Calling OpenAI-compatible image generation: model=%s",
            self._model_name,
        )
        response = self.client.chat.completions.create(**request_kwargs)

        # Extract image from response content parts. OpenAI-compatible gateways
        # preserve image responses in a few shapes depending on SDK/schema
        # support, including Gemini-style inlineData parts in model_extra.
        message = response.choices[0].message
        raw_content = message.content

        # Case 1: content is a list of parts (structured response)
        img = self._try_extract_image_from_part(raw_content)
        if img is not None:
            return img

        # Case 2: images in a separate "images" field on the message
        # (some OpenAI-compatible gateways return images here instead of content)
        raw_msg = response.choices[0].message
        images_list = getattr(raw_msg, "images", None)
        # Also check model_extra for fields the SDK doesn't recognise
        if images_list is None and hasattr(raw_msg, "model_extra"):
            images_list = raw_msg.model_extra.get("images")
        if isinstance(images_list, list):
            for part in images_list:
                img = self._try_extract_image_from_part(part)
                if img is not None:
                    return img

        # Case 3: full Gemini-style response content in SDK overflow fields.
        for extra in (
            getattr(raw_msg, "model_extra", None),
            getattr(response, "model_extra", None),
        ):
            img = self._try_extract_image_from_part(extra)
            if img is not None:
                return img

        # Log the raw response for debugging
        logger.warning(
            "No image found in response. content type=%s, "
            "finish_reason=%s, raw_content=%s",
            type(raw_content).__name__,
            getattr(response.choices[0], "finish_reason", "unknown"),
            repr(raw_content)[:500] if raw_content else "None",
        )

        raise ValueError(
            "No image found in response. "
            f"Response content type: {type(raw_content)}, "
            f"model: {self._model_name}"
        )

    @staticmethod
    def _try_extract_image_from_part(part: Any) -> PILImage.Image | None:
        """Try to extract a PIL Image from a single content part."""
        if part is None:
            return None

        if isinstance(part, str):
            return (
                OpenAICompatibleChatImageGenerationModel._try_decode_data_uri(part)
                or OpenAICompatibleChatImageGenerationModel._try_extract_image_from_json_string(
                    part
                )
            )

        if isinstance(part, list | tuple):
            for item in part:
                img = OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
                    item
                )
                if img is not None:
                    return img
            return None

        if isinstance(part, dict):
            # {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
            image_url = part.get("image_url")
            if part.get("type") == "image_url" and image_url:
                img = OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
                    image_url
                )
                if img is not None:
                    return img

            for key in ("url", "data_uri"):
                value = part.get(key)
                if isinstance(value, str):
                    img = OpenAICompatibleChatImageGenerationModel._try_decode_data_uri(
                        value
                    )
                    if img is not None:
                        return img

            for key in ("inline_data", "inlineData", "image", "source"):
                img = OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
                    part.get(key)
                )
                if img is not None:
                    return img

            mime_type = part.get("mime_type") or part.get("mimeType")
            if isinstance(mime_type, str) and mime_type.startswith("image/"):
                for key in ("data", "base64", "b64_json"):
                    img = OpenAICompatibleChatImageGenerationModel._try_decode_image_payload(
                        part.get(key)
                    )
                    if img is not None:
                        return img

            for key in ("b64_json", "base64"):
                img = (
                    OpenAICompatibleChatImageGenerationModel._try_decode_image_payload(
                        part.get(key)
                    )
                )
                if img is not None:
                    return img

            for value in part.values():
                img = OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
                    value
                )
                if img is not None:
                    return img
            return None

        # OpenAI SDK may return typed objects with attributes
        if hasattr(part, "type") and getattr(part, "type", None) == "image_url":
            image_url_obj = getattr(part, "image_url", None)
            if image_url_obj:
                img = OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
                    image_url_obj
                )
                if img is not None:
                    return img

        if hasattr(part, "model_dump"):
            try:
                dumped = part.model_dump()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                img = OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
                    dumped
                )
                if img is not None:
                    return img

        for attr in ("url", "data_uri"):
            value = getattr(part, attr, None)
            if isinstance(value, str):
                img = OpenAICompatibleChatImageGenerationModel._try_decode_data_uri(
                    value
                )
                if img is not None:
                    return img

        for attr in (
            "inline_data",
            "inlineData",
            "image_url",
            "image",
            "images",
            "parts",
            "content",
            "model_extra",
        ):
            if not hasattr(part, attr):
                continue
            img = OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
                getattr(part, attr)
            )
            if img is not None:
                return img

        mime_type = getattr(part, "mime_type", None) or getattr(part, "mimeType", None)
        if isinstance(mime_type, str) and mime_type.startswith("image/"):
            for attr in ("data", "base64", "b64_json"):
                img = (
                    OpenAICompatibleChatImageGenerationModel._try_decode_image_payload(
                        getattr(part, attr, None)
                    )
                )
                if img is not None:
                    return img
        return None

    @staticmethod
    def _try_decode_data_uri(text: str) -> PILImage.Image | None:
        """Decode a ``data:image/...;base64,...`` URI to a PIL Image."""
        if "base64," in text:
            b64_data = text.split("base64,", 1)[1].strip()
            return OpenAICompatibleChatImageGenerationModel._try_decode_image_payload(
                b64_data
            )
        return None

    @staticmethod
    def _try_extract_image_from_json_string(text: str) -> PILImage.Image | None:
        """Decode image content when a gateway serializes parts as JSON text."""
        stripped = text.strip()
        if not stripped or stripped[0] not in "[{":
            return None
        try:
            import json

            parsed = json.loads(stripped)
        except Exception:
            return None
        return OpenAICompatibleChatImageGenerationModel._try_extract_image_from_part(
            parsed
        )

    @staticmethod
    def _try_decode_image_payload(payload: Any) -> PILImage.Image | None:
        """Decode raw image bytes or base64 text into a loaded PIL image."""
        if payload is None:
            return None
        candidates: list[bytes] = []
        if isinstance(payload, bytes):
            candidates.append(payload)
            try:
                candidates.append(base64.b64decode(payload))
            except Exception:
                pass
        elif isinstance(payload, str):
            try:
                candidates.append(base64.b64decode(payload.strip()))
            except Exception:
                return None
        else:
            return None

        for raw in candidates:
            try:
                image = PILImage.open(BytesIO(raw))
                image.load()
                return image
            except Exception:
                continue
        return None


class _NamedBytesIO(BytesIO):
    """BytesIO subclass with a name attribute for OpenAI SDK file uploads."""

    def __init__(self, data: bytes, name: str = "image.png") -> None:
        super().__init__(data)
        self.name = name


class OpenAIImageGenerationModel(BaseImageGenerationModel):
    """OpenAI image generation model using the Images API (gpt-image-1).

    Uses ``images.generate`` for text-to-image and ``images.edit`` when
    conditioning images are provided.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_OPENAI_IMAGE_MODEL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI image generation model.

        Args:
            api_key: OpenAI API key (loads from OPENAI_API_KEY env var if None).
                Use ``not-used`` explicitly when ``base_url`` points at a local
                endpoint that does not require auth.
            model: Model name (default: gpt-image-1)
            timeout: Request timeout in seconds
            base_url: Override API base URL. Useful for OpenAI-compatible
                servers such as a locally-hosted NIM image generation container
                (e.g. ``http://localhost:8000/v1``).
            **kwargs: Additional configuration options
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIImageGenerationModel. "
                "Install with: pip install openai"
            ) from e

        api_key = get_openai_api_key_for_base_url(base_url, api_key)
        if api_key is None:
            raise ValueError(
                "API key is required. Provide via api_key parameter or "
                "OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
        self._model_name = model

    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image using OpenAI Images API.

        Args:
            prompt: Text prompt describing the desired output
            images: Optional conditioning images (uses images.edit when provided)
            **kwargs: Additional arguments forwarded to the API call

        Returns:
            Generated PIL Image
        """
        if images:
            image_files = [self._to_named_bytes_io(img) for img in images]
            response = self.client.images.edit(
                model=self._model_name,
                image=image_files,  # type: ignore[arg-type]
                prompt=prompt,
                n=1,
                **kwargs,
            )
        else:
            response = self.client.images.generate(
                model=self._model_name,
                prompt=prompt,
                n=1,
                **kwargs,
            )
        return self._extract_image(response)

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image from interleaved (description, image) pairs.

        The descriptions are combined into a single prompt since the OpenAI
        Images API does not natively support interleaved text/image inputs.

        Args:
            image_prompt_pairs: List of (description, image) tuples
            final_prompt: Final generation instruction
            **kwargs: Additional arguments forwarded to the API call

        Returns:
            Generated PIL Image
        """
        descriptions = [desc for desc, _ in image_prompt_pairs]
        combined_prompt = "\n".join(descriptions + [final_prompt])
        conditioning_images = [img for _, img in image_prompt_pairs]
        return self.generate(
            prompt=combined_prompt, images=conditioning_images, **kwargs
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return "openai"

    def _to_named_bytes_io(
        self, image: str | Path | PILImage.Image | np.ndarray
    ) -> "_NamedBytesIO":
        """Convert an image to a named BytesIO for OpenAI SDK upload."""
        pil_img = self._load_image(image)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return _NamedBytesIO(buf.getvalue())

    def _extract_image(self, response: Any) -> PILImage.Image:
        """Extract PIL Image from an OpenAI ImagesResponse."""
        if response.data:
            item = response.data[0]
            if item.b64_json:
                return PILImage.open(BytesIO(base64.b64decode(item.b64_json)))
            if item.url:
                import urllib.request

                with urllib.request.urlopen(  # noqa: S310
                    item.url, timeout=_DEFAULT_TIMEOUT_SECONDS
                ) as resp:
                    return PILImage.open(BytesIO(resp.read()))
        raise ValueError("No image found in OpenAI response")


class NIMImageGenerationModel(BaseImageGenerationModel):
    """NVIDIA NIM image generation model using the GenAI REST API.

    Calls ``https://ai.api.nvidia.com/v1/genai/{model}`` with a JSON body and
    returns JPEG images decoded from the ``artifacts[].base64`` response field.

    Image conditioning (ref_images) is not supported by the NIM GenAI endpoint;
    when conditioning images are provided their descriptions are folded into the
    text prompt only — no pixel data is sent to the API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_NIM_IMAGE_MODEL,
        base_url: str = _DEFAULT_NIM_IMAGE_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ) -> None:
        """Initialize NIM image generation model.

        Args:
            api_key: NVIDIA API key (loads from NVIDIA_API_KEY env var if None)
            model: Model name using underscore format, e.g.
                ``black-forest-labs/flux_2-klein-4b`` (default)
            base_url: Base URL for the NIM GenAI endpoint
            timeout: Request timeout in seconds
            **kwargs: Reserved for future options
        """
        api_key = get_nim_api_key_for_base_url(base_url, api_key)
        if api_key is None:
            raise ValueError(
                "API key is required. Provide via api_key parameter or "
                "NVIDIA_API_KEY environment variable."
            )

        self._api_key = api_key
        self._model_name = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @staticmethod
    def _model_to_url_slug(model: str) -> str:
        """Convert internal model name to URL slug.

        The NIM GenAI endpoint uses dots instead of the first underscore in the
        model name portion (after the org prefix).
        e.g. ``black-forest-labs/flux_2-klein-4b`` → ``black-forest-labs/flux.2-klein-4b``
        """
        if "/" in model:
            org, name = model.split("/", 1)
            return f"{org}/{name.replace('_', '.', 1)}"
        return model.replace("_", ".", 1)

    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image using NIM GenAI REST API.

        Args:
            prompt: Text prompt describing the desired output.
            images: Ignored (NIM has no img2img endpoint); present only to
                satisfy the base class interface. If provided, a warning is
                logged and the images are not sent to the API.
            **kwargs: Additional JSON body fields forwarded to the API
                (e.g. ``height``, ``width``).

        Returns:
            Generated PIL Image (JPEG decoded from base64).
        """
        import json
        import urllib.request

        if images:
            logger.warning(
                "NIMImageGenerationModel does not support image conditioning. "
                "The provided images will be ignored."
            )

        slug = self._model_to_url_slug(self._model_name)
        url = f"{self._base_url}/{slug}"

        body: dict[str, Any] = {"prompt": prompt, "height": 1024, "width": 1024}
        body.update(kwargs)

        data = json.dumps(body).encode()
        req = urllib.request.Request(  # noqa: S310
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        logger.info("Calling NIM image generation: model=%s", self._model_name)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            result = json.loads(resp.read())

        artifacts = result.get("artifacts", [])
        if not artifacts:
            raise ValueError("No artifacts in NIM image generation response")

        b64_data = artifacts[0].get("base64", "")
        if not b64_data:
            raise ValueError("Empty base64 in NIM image generation response")

        return PILImage.open(BytesIO(base64.b64decode(b64_data)))

    def generate_with_image_prompt_pairs(
        self,
        image_prompt_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        **kwargs: Any,
    ) -> PILImage.Image:
        """Generate image from interleaved (description, image) pairs.

        NIM has no interleaved image+text generation endpoint.  The image
        descriptions are concatenated into the text prompt; pixel data is
        discarded.

        Args:
            image_prompt_pairs: List of (description, image) tuples; only the
                descriptions are used.
            final_prompt: Final generation instruction appended after descriptions.
            **kwargs: Extra fields forwarded to the API body.

        Returns:
            Generated PIL Image.
        """
        descriptions = [desc for desc, _ in image_prompt_pairs]
        combined_prompt = "\n".join(descriptions + [final_prompt])
        return self.generate(prompt=combined_prompt, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return "nim"

    @property
    def supports_image_conditioning(self) -> bool:
        # The cloud NIM GenAI endpoint is text-only; any ``images=...`` is
        # dropped (see ``generate``). Downstream PBR pipelines rely on
        # this flag to avoid wasting a round-trip producing an albedo
        # reference that will never reach the model.
        return False


def create_image_generation_model(
    backend: str,
    **kwargs: Any,
) -> BaseImageGenerationModel:
    """Create an image generation model for the specified backend.

    Available backends depend on the installation. Public providers are always
    available and optional packages contribute factories through entry points.

    Args:
        backend: Backend name (use ``list_image_gen_backends()`` to see available)
        **kwargs: Backend-specific arguments (api_key, model, base_url, etc.)

    Returns:
        Configured image generation model instance

    Raises:
        ValueError: If backend is not supported
        ImportError: If required packages are not installed
    """
    from world_understanding.functions.models.backends.registry import (
        get_image_gen_factory,
    )

    factory = get_image_gen_factory(backend)
    return factory(**kwargs)
