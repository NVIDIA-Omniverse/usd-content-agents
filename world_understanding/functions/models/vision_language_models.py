# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision-Language Model implementations."""

import asyncio
import logging
import math
import os
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import CancelledError
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image as PILImage
from requests.exceptions import Timeout as RequestsTimeout

from world_understanding.functions.models.nim_timeout import _apply_nim_chat_timeout
from world_understanding.functions.models.token_limits import (
    ensure_model_output_token_budget,
    normalize_openai_token_kwargs,
)
from world_understanding.telemetry import GenAIAttributes, get_current_span, traced_vlm
from world_understanding.utils.credentials import get_env_api_key_for_backend
from world_understanding.utils.image_utils import image_to_base64
from world_understanding.utils.model_timeout import (
    NON_RETRYABLE_VLM_TIMEOUT_MESSAGE,
    NonRetryableVLMTimeoutError,
)
from world_understanding.utils.response_content import extract_text_content
from world_understanding.utils.token_tracking import TokenUsage

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Default configurations
_DEFAULT_NIM_VLM_MODEL = "moonshotai/kimi-k3"
_DEFAULT_AZURE_VLM_MODEL = "gpt-5"
_DEFAULT_OPENAI_MODEL = "gpt-5.4"
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
_DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
_DEFAULT_GRADIO_API_NAME = "/process_media"
# No default endpoint — callers must pass `endpoint=...` for the gradio backend.
_DEFAULT_GRADIO_ENDPOINT = ""
_DEFAULT_TIMEOUT_SECONDS = 120.0
_GRADIO_CANCEL_SETTLE_TIMEOUT_SECONDS = 1.0
_DEFAULT_MAX_TOKENS = None
_DEFAULT_TEMPERATURE = None


def _record_effective_openai_max_tokens(request_kwargs: dict[str, Any]) -> None:
    """Replace the traced request limit with the value sent to the provider."""
    effective_max_tokens = request_kwargs.get("max_tokens")
    if effective_max_tokens is None:
        effective_max_tokens = request_kwargs.get("max_completion_tokens")
    if effective_max_tokens is None:
        return

    span = get_current_span()
    if span is not None:
        span.set_attribute(GenAIAttributes.REQUEST_MAX_TOKENS, effective_max_tokens)


def _invoke_nim_chat_model(
    chat_model: Any,
    messages: list[Any],
    invoke_kwargs: dict[str, Any],
) -> "BaseMessage":
    """Invoke ChatNVIDIA while preserving retry-unsafe timeout semantics."""
    try:
        return cast("BaseMessage", chat_model.invoke(messages, **invoke_kwargs))
    except (TimeoutError, RequestsTimeout):
        # The pinned SDK raises built-in TimeoutError when HTTP 202 polling
        # exhausts its deadline, while requests raises its own Timeout type for
        # a transport deadline. Neither proves that remote work stopped.
        raise NonRetryableVLMTimeoutError(NON_RETRYABLE_VLM_TIMEOUT_MESSAGE) from None


class BaseVisionLanguageModel(ABC):
    """Base class for vision-language models.

    Subclasses should set self._last_token_usage after each model invocation
    to enable token tracking.
    """

    def __init__(self) -> None:
        """Initialize base VLM with token tracking."""
        self._last_token_usage: TokenUsage | None = None
        self._bounded_request_timeout_seconds: float | None = None

    @staticmethod
    def _normalize_request_timeout(timeout: Any, *, label: str) -> float:
        """Return a positive finite timeout without accepting booleans."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(f"{label} timeout must be a positive finite number")
        return float(timeout)

    def _set_bounded_request_timeout(self, timeout: Any, *, label: str) -> float:
        """Validate and record the synchronous request deadline contract."""
        timeout_s = self._normalize_request_timeout(timeout, label=label)
        self._bounded_request_timeout_seconds = timeout_s
        return timeout_s

    @property
    def last_token_usage(self) -> TokenUsage | None:
        """Get token usage from the last model invocation.

        Returns:
            TokenUsage object if available, None otherwise
        """
        return self._last_token_usage

    @property
    def has_bounded_request_timeout(self) -> bool:
        """Return whether the backend itself bounds synchronous requests."""
        return self._bounded_request_timeout_seconds is not None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from text and optional images synchronously.

        Args:
            prompt: User prompt/question
            images: Optional list of images as paths, PIL Images, or arrays
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        pass

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from text and optional images asynchronously.

        Default implementation delegates to sync generate() via asyncio.to_thread.
        Subclasses can override for true async behavior.

        Args:
            prompt: User prompt/question
            images: Optional list of images as paths, PIL Images, or arrays
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        return await asyncio.to_thread(
            self.generate,
            prompt,
            images,
            system_prompt,
            temperature,
            max_tokens,
            **kwargs,
        )

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs followed by a final prompt.

        Args:
            image_caption_pairs: List of tuples (caption, image) where each caption
                                describes or introduces its corresponding image
            final_prompt: The final prompt/question after all images
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        # Default implementation: concatenate all captions with final prompt
        # and provide all images together (for backward compatibility)
        all_captions = []
        all_images = []

        for caption, image in image_caption_pairs:
            all_captions.append(caption)
            all_images.append(image)

        combined_prompt = "\n".join(all_captions) + "\n" + final_prompt
        return self.generate(
            prompt=combined_prompt,
            images=all_images if all_images else None,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def agenerate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs asynchronously.

        Args:
            image_caption_pairs: List of tuples (caption, image) where each caption
                                describes or introduces its corresponding image
            final_prompt: The final prompt/question after all images
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        # Default implementation: concatenate all captions with final prompt
        # and provide all images together (for backward compatibility)
        all_captions = []
        all_images = []

        for caption, image in image_caption_pairs:
            all_captions.append(caption)
            all_images.append(image)

        combined_prompt = "\n".join(all_captions) + "\n" + final_prompt
        return await self.agenerate(
            prompt=combined_prompt,
            images=all_images if all_images else None,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

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

    def _images_to_base64(
        self, images: list[str | Path | PILImage.Image | np.ndarray]
    ) -> list[str]:
        """Convert images to base64 strings.

        Args:
            images: List of images in various formats

        Returns:
            List of base64 encoded image strings
        """
        base64_images = []
        for image in images:
            pil_image = self._load_image(image)
            base64_images.append(image_to_base64(pil_image))
        return base64_images


class GradioVLM(BaseVisionLanguageModel):
    """Vision-Language Model using Gradio client backend.

    Note: Gradio endpoints may require specific network access to function properly.
    The default endpoint is only accessible via NVIDIA San Jose VPN.
    """

    def __init__(
        self,
        endpoint: str = _DEFAULT_GRADIO_ENDPOINT,
        api_name: str = _DEFAULT_GRADIO_API_NAME,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        use_single_image_api: bool | None = None,
        **kwargs: Any,
    ):
        """Initialize Gradio VLM client.

        Args:
            endpoint: Gradio server endpoint URL
            api_name: API endpoint name for the VLM service
            timeout: Maximum job runtime before cancellation begins, in seconds
            use_single_image_api: If True, call endpoint with a single image
                input; if False, call an endpoint with multi-image. Defaults
                to True for backward compatibility when not provided.
            **kwargs: Additional configuration options
        """

        super().__init__()  # Initialize token tracking

        try:
            from gradio_client import Client
        except ImportError as e:
            raise ImportError(
                "gradio_client is required for GradioVLM. "
                "Install with: pip install gradio_client"
            ) from e

        self.endpoint = endpoint
        self.api_name = api_name
        self.timeout = self._set_bounded_request_timeout(timeout, label="Gradio VLM")
        self.client = Client(endpoint, verbose=False)
        self._model_name = kwargs.get("model_name", "gradio-vlm")
        # Configurable pathway for single vs multi image request formatting
        self.use_single_image_api: bool = (
            use_single_image_api if use_single_image_api is not None else True
        )

    @staticmethod
    def _cleanup_owned_temp_files(paths: tuple[str, ...] | list[str]) -> None:
        """Remove temporary inputs still owned by this backend."""
        for path in paths:
            try:
                os.unlink(path)
            except Exception:
                # Cleanup must never mask the provider result. This also keeps
                # completion callbacks from surfacing filesystem-hook errors.
                pass

    def _defer_owned_temp_file_cleanup(
        self,
        job: Any,
        owned_temp_paths: list[str],
    ) -> None:
        """Transfer temporary-input cleanup to an unsettled Gradio job."""
        if not owned_temp_paths or job.done():
            return

        deferred_paths = tuple(owned_temp_paths)
        owned_temp_paths.clear()

        def cleanup_after_completion(_job: Any) -> None:
            self._cleanup_owned_temp_files(deferred_paths)

        try:
            job.add_done_callback(cleanup_after_completion)
        except Exception:
            # Keep the files rather than racing a worker that may still be
            # uploading them. Pinned Gradio jobs are Future subclasses and
            # support callbacks; this is a fail-safe for incompatible clients.
            logger.warning(
                "Could not register cleanup for unsettled Gradio inputs; "
                "retaining %d temporary file(s)",
                len(deferred_paths),
                exc_info=True,
            )

    def _predict_with_timeout(
        self,
        owned_temp_paths: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run one Gradio job with a bounded, retry-unsafe local deadline."""
        job = self.client.submit(*args, **kwargs)
        try:
            return job.result(timeout=self.timeout)
        except TimeoutError as error:
            # A completed job may itself raise a provider TimeoutError. Only
            # cancel when our wait deadline expired while the job remained live.
            if job.done():
                return job.result()
            try:
                job.cancel()
            except Exception:
                logger.warning(
                    "Failed to cancel timed-out Gradio VLM job; waiting for it "
                    "to settle",
                    exc_info=True,
                )

            # Bound local cancellation settlement too. Gradio cancellation is
            # best-effort for a non-generator endpoint and cannot prove that
            # remote provider work stopped, so callers must not retry this
            # distinct timeout.
            try:
                settled_result = job.result(
                    timeout=min(
                        self.timeout,
                        _GRADIO_CANCEL_SETTLE_TIMEOUT_SECONDS,
                    )
                )
            except CancelledError:
                settled_result = None
            except TimeoutError:
                if job.done():
                    return job.result()
                settled_result = None
            except Exception:
                if job.done():
                    return job.result()
                settled_result = None
            else:
                return settled_result

            self._defer_owned_temp_file_cleanup(job, owned_temp_paths)
            raise NonRetryableVLMTimeoutError(
                f"Gradio VLM request exceeded {self.timeout:g} seconds; "
                "remote completion could not be verified"
            ) from error

    @traced_vlm(name="vlm.generate", system="gradio", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Gradio VLM service."""
        try:
            from gradio_client import file as gradio_f
        except ImportError as e:
            raise ImportError("gradio_client is required for file handling") from e

        if self.use_single_image_api:
            # Prepare the request
            tmp_file_paths: list[str] = []
            try:
                if images and len(images) > 0:
                    # For now, support single image (can be extended for multiple)
                    image = images[0]

                    # Handle different image types
                    if isinstance(image, str | Path):
                        image_input = gradio_f(str(image))
                    else:
                        # For PIL Image or numpy array, save to temp file
                        pil_image = self._load_image(image)
                        with tempfile.NamedTemporaryFile(
                            suffix=".png", delete=False
                        ) as tmp_file:
                            pil_image.save(tmp_file.name)
                            tmp_file_paths.append(tmp_file.name)
                            image_input = gradio_f(tmp_file.name)
                else:
                    image_input = None

                # Combine system and user prompts
                full_prompt = (
                    f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                )

                predict_kwargs = {
                    "api_name": self.api_name,
                }
                if temperature is not None:
                    predict_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    predict_kwargs["max_tokens"] = max_tokens
                predict_kwargs.update(kwargs)

                # Call Gradio API
                result = self._predict_with_timeout(
                    tmp_file_paths,
                    image_input,  # image_input
                    None,  # video_input (not used for images)
                    full_prompt,
                    **predict_kwargs,
                )

                # Extract response (usually first element for text output)
                if isinstance(result, list | tuple) and len(result) > 0:
                    return str(result[0])
                else:
                    return str(result)
            finally:
                self._cleanup_owned_temp_files(tmp_file_paths)
        else:
            tmp_file_paths: list[str] = []
            try:
                # Build a list of gradio file handles if images are provided
                images_arg = None
                if images and len(images) > 0:
                    image_inputs: list[Any] = []
                    for img in images:
                        if isinstance(img, str | Path):
                            image_inputs.append(gradio_f(str(img)))
                        else:
                            # For PIL Image or numpy array, save to temp file
                            pil_image = self._load_image(img)
                            with tempfile.NamedTemporaryFile(
                                suffix=".png", delete=False
                            ) as tmp_file:
                                pil_image.save(tmp_file.name)
                                tmp_file_paths.append(tmp_file.name)
                                image_inputs.append(gradio_f(tmp_file.name))
                    images_arg = image_inputs

                # Combine system and user prompts
                full_prompt = (
                    f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                )

                predict_kwargs = {
                    "api_name": self.api_name,
                }
                if temperature is not None:
                    predict_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    predict_kwargs["max_tokens"] = max_tokens
                predict_kwargs.update(kwargs)

                result = self._predict_with_timeout(
                    tmp_file_paths,
                    images_arg,
                    None,
                    full_prompt,
                    **predict_kwargs,
                )

                # Extract response (usually first element for text output)
                if isinstance(result, list | tuple) and len(result) > 0:
                    return str(result[0])
                else:
                    return str(result)
            finally:
                self._cleanup_owned_temp_files(tmp_file_paths)

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs using Gradio.

        Note: Gradio endpoints may not natively support true interleaved content.
        This implementation combines captions with the final prompt and
        provides images in order.

        Args:
            image_caption_pairs: List of tuples (caption, image)
            final_prompt: The final prompt after all images
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (None uses default)
            max_tokens: Maximum tokens in response (None uses default)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        # Build combined prompt with all captions
        combined_captions = []
        images = []

        for caption, image in image_caption_pairs:
            combined_captions.append(caption)
            images.append(image)

        # Combine all text parts
        full_user_prompt = "\n".join(combined_captions) + "\n" + final_prompt

        # Use the regular generate method with the combined prompt and ordered images
        return self.generate(
            prompt=full_user_prompt,
            images=images if images else None,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "gradio"

    @property
    def has_bounded_request_timeout(self) -> bool:
        """Gradio bounds both its initial wait and local cancellation wait."""
        return super().has_bounded_request_timeout


class AzureOpenAIVLM(BaseVisionLanguageModel):
    """Azure OpenAI VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        api_version: str = "2025-03-01-preview",
        azure_endpoint: str = "",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        backend_name: str = "azure_openai",
        **kwargs: Any,
    ):
        """Initialize an Azure OpenAI VLM.

        Args:
            api_key: Azure OpenAI API key
            model: Model name (defaults to gpt-5)
            api_version: API version
            azure_endpoint: Azure endpoint URL (required; no default)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        if not azure_endpoint:
            raise ValueError(
                "azure_endpoint is required for AzureOpenAIVLM (no default)."
            )
        super().__init__()  # Initialize token tracking
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for AzureOpenAIVLM. "
                "Install with: pip install langchain-openai"
            ) from e

        self._model_name = model or _DEFAULT_AZURE_VLM_MODEL
        self._backend_name = backend_name
        timeout_s = self._set_bounded_request_timeout(timeout, label="Azure OpenAI VLM")
        constructor_kwargs = normalize_openai_token_kwargs(
            self._model_name, None, kwargs
        )
        self.chat_model = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model=self._model_name,
            api_key=api_key,  # type: ignore[arg-type]
            timeout=timeout_s,
            **constructor_kwargs,
        )

    @traced_vlm(name="vlm.generate", system="azure_openai", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Azure OpenAI."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Debug logging for images
        import logging

        logger = logging.getLogger(__name__)
        if images is not None:
            logger.debug(
                f"AzureOpenAIVLM.generate received {len(images)} images (type: {type(images).__name__})"
            )
        else:
            logger.warning("AzureOpenAIVLM.generate received images=None")

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs = normalize_openai_token_kwargs(
            self._model_name, max_tokens, kwargs
        )
        _record_effective_openai_max_tokens(invoke_kwargs)
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        response = self.chat_model.invoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Azure OpenAI asynchronously."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Debug logging for images
        import logging

        logger = logging.getLogger(__name__)
        if images is not None:
            logger.debug(
                f"AzureOpenAIVLM.agenerate received {len(images)} images (type: {type(images).__name__})"
            )
        else:
            logger.warning("AzureOpenAIVLM.agenerate received images=None")

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs = normalize_openai_token_kwargs(
            self._model_name, max_tokens, kwargs
        )
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        invoke_kwargs = normalize_openai_token_kwargs(
            self._model_name, max_tokens, kwargs
        )
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        response = self.chat_model.invoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    async def agenerate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs asynchronously with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        invoke_kwargs = normalize_openai_token_kwargs(
            self._model_name, max_tokens, kwargs
        )
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return self._backend_name


def _openai_usage_details(details: Any) -> dict[str, int] | None:
    """Normalize usage details while preserving explicit zero token counts."""
    if details is None:
        return None
    if isinstance(details, dict):
        values = details
    elif callable(getattr(details, "model_dump", None)):
        values = details.model_dump()
    else:
        values = vars(details) if hasattr(details, "__dict__") else {}
    normalized = {
        str(key): value
        for key, value in values.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return normalized or None


def _token_usage_from_openai_response(
    usage: Any,
    *,
    model_name: str | None,
    invocation_type: str,
) -> TokenUsage | None:
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        input_token_details=_openai_usage_details(
            getattr(usage, "prompt_tokens_details", None)
        ),
        output_token_details=_openai_usage_details(
            getattr(usage, "completion_tokens_details", None)
        ),
        model_name=model_name,
        invocation_type=invocation_type,
    )


class OpenAICompatibleVLM(BaseVisionLanguageModel):
    """VLM backed by an operator-selected OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = "",
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        backend_name: str = "openai_compatible",
        **kwargs: Any,
    ):
        """Initialize an OpenAI-compatible VLM.

        Args:
            api_key: Endpoint-scoped API key.
            model: Model name.
            base_url: OpenAI-compatible API base URL.
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        if not base_url:
            raise ValueError("base_url is required for OpenAICompatibleVLM")
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAICompatibleVLM. "
                "Install with: pip install openai"
            ) from e

        self._model_name = model or _DEFAULT_OPENAI_MODEL
        self._backend_name = backend_name
        self._base_url = base_url
        timeout_s = self._set_bounded_request_timeout(
            timeout, label="OpenAI-compatible VLM"
        )
        self._timeout = timeout_s
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

        from openai import AsyncOpenAI

        self.aclient = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_s
        )

    @traced_vlm(name="vlm.generate", system="openai_compatible", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate a response through the configured endpoint."""
        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content if images else prompt},
        ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        request_kwargs.update(
            normalize_openai_token_kwargs(self._model_name, max_tokens, kwargs)
        )
        _record_effective_openai_max_tokens(request_kwargs)

        response = self.client.chat.completions.create(**request_kwargs)

        # Track token usage
        if response.usage:
            self._last_token_usage = _token_usage_from_openai_response(
                response.usage,
                model_name=self._model_name,
                invocation_type="vlm",
            )

        return response.choices[0].message.content or ""

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate a response through the configured endpoint asynchronously."""
        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content if images else prompt},
        ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        request_kwargs.update(
            normalize_openai_token_kwargs(self._model_name, max_tokens, kwargs)
        )

        response = await self.aclient.chat.completions.create(**request_kwargs)

        # Track token usage
        if response.usage:
            self._last_token_usage = _token_usage_from_openai_response(
                response.usage,
                model_name=self._model_name,
                invocation_type="vlm",
            )

        return response.choices[0].message.content or ""

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with interleaved support."""
        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        request_kwargs.update(
            normalize_openai_token_kwargs(self._model_name, max_tokens, kwargs)
        )

        response = self.client.chat.completions.create(**request_kwargs)

        # Track token usage
        if response.usage:
            self._last_token_usage = _token_usage_from_openai_response(
                response.usage,
                model_name=self._model_name,
                invocation_type="vlm",
            )

        return response.choices[0].message.content or ""

    async def agenerate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs asynchronously with interleaved support."""
        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature

        request_kwargs.update(
            normalize_openai_token_kwargs(self._model_name, max_tokens, kwargs)
        )

        response = await self.aclient.chat.completions.create(**request_kwargs)

        # Track token usage
        if response.usage:
            self._last_token_usage = _token_usage_from_openai_response(
                response.usage,
                model_name=self._model_name,
                invocation_type="vlm",
            )

        return response.choices[0].message.content or ""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return self._backend_name


class NvidiaNIMVLM(BaseVisionLanguageModel):
    """NVIDIA NIM VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize NVIDIA NIM VLM.

        Args:
            api_key: NVIDIA API key
            model: Model name (defaults to moonshotai/kimi-k3)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()  # Initialize token tracking
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as e:
            raise ImportError(
                "langchain-nvidia-ai-endpoints is required for NvidiaNIMVLM. "
                "Install with: pip install langchain-nvidia-ai-endpoints"
            ) from e

        self._model_name = model or _DEFAULT_NIM_VLM_MODEL
        timeout_s = self._normalize_request_timeout(timeout, label="NvidiaNIMVLM")
        self.chat_model = ChatNVIDIA(
            model=self._model_name,
            nvidia_api_key=api_key,
            **kwargs,
        )
        # Clear ChatNVIDIA's built-in max_tokens default so it doesn't conflict
        # with per-call max_tokens passed via invoke kwargs.
        self.chat_model.max_tokens = None
        # Cloud NIM rejects `timeout` when ChatNVIDIA serializes constructor
        # fields into the request body. Apply it to the underlying HTTP client
        # instead when the installed SDK exposes one.
        self._has_bounded_request_timeout = _apply_nim_chat_timeout(
            self.chat_model,
            timeout_s,
            label="NvidiaNIMVLM",
        )

    @traced_vlm(name="vlm.generate", system="nim", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using NVIDIA NIM."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content
        content: list[dict[str, Any]] = []

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Debug logging for images
        import logging

        logger = logging.getLogger(__name__)
        if images is not None:
            logger.debug(
                f"NvidiaNIMVLM.generate received {len(images)} images (type: {type(images).__name__})"
            )
        else:
            logger.warning("NvidiaNIMVLM.generate received images=None")

        # Add images if provided
        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        # Set temperature and max_tokens if provided
        invoke_kwargs = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        # Handle max_tokens for NIM models:
        # NIM uses max_tokens (not max_completion_tokens). ChatNVIDIA's built-in
        # default is cleared at construction; pass max_tokens per-call for thread safety.
        effective_max_tokens = kwargs.get("max_completion_tokens")
        if effective_max_tokens is None:
            effective_max_tokens = kwargs.get("max_tokens")
        if effective_max_tokens is None:
            effective_max_tokens = max_tokens
        if effective_max_tokens is not None:
            invoke_kwargs["max_tokens"] = ensure_model_output_token_budget(
                self._model_name,
                effective_max_tokens,
            )

        # Add remaining kwargs (excluding max_tokens and max_completion_tokens which we handled)
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        # Remove None values - they shouldn't be passed to the API
        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

        response = _invoke_nim_chat_model(
            self.chat_model,
            messages,
            invoke_kwargs,
        )

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Build content with interleaved captions and images
        content: list[dict[str, Any]] = []

        for caption, image in image_caption_pairs:
            # Add caption text
            content.append({"type": "text", "text": caption})

            # Add image
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        # Add final prompt
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        # Set temperature and max_tokens if provided
        invoke_kwargs = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature

        # Handle max_tokens for NIM models:
        # NIM uses max_tokens (not max_completion_tokens). ChatNVIDIA's built-in
        # default is cleared at construction; pass max_tokens per-call for thread safety.
        effective_max_tokens = kwargs.get("max_completion_tokens")
        if effective_max_tokens is None:
            effective_max_tokens = kwargs.get("max_tokens")
        if effective_max_tokens is None:
            effective_max_tokens = max_tokens
        if effective_max_tokens is not None:
            invoke_kwargs["max_tokens"] = ensure_model_output_token_budget(
                self._model_name,
                effective_max_tokens,
            )

        # Add remaining kwargs (excluding max_tokens and max_completion_tokens which we handled)
        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        # Remove None values - they shouldn't be passed to the API
        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}

        response = _invoke_nim_chat_model(
            self.chat_model,
            messages,
            invoke_kwargs,
        )

        # Track token usage
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )

        return response.content  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "nim"

    @property
    def has_bounded_request_timeout(self) -> bool:
        """Return whether ChatNVIDIA's synchronous HTTP client was bounded."""
        return bool(getattr(self, "_has_bounded_request_timeout", False))


class OpenAIVLM(BaseVisionLanguageModel):
    """OpenAI VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize OpenAI VLM.

        Args:
            api_key: OpenAI API key
            model: Model name (defaults to gpt-5.4)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAIVLM. "
                "Install with: pip install langchain-openai"
            ) from e

        self._model_name = model or _DEFAULT_OPENAI_MODEL
        timeout_s = self._set_bounded_request_timeout(timeout, label="OpenAI VLM")
        constructor_kwargs = normalize_openai_token_kwargs(
            self._model_name, None, kwargs
        )
        self.chat_model = ChatOpenAI(
            model=self._model_name,
            api_key=api_key,  # type: ignore[arg-type]
            timeout=timeout_s,
            **constructor_kwargs,
        )

    @traced_vlm(name="vlm.generate", system="openai", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using OpenAI."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs = normalize_openai_token_kwargs(
            self._model_name,
            max_tokens,
            kwargs,
            prefer_max_tokens_argument=True,
        )
        _record_effective_openai_max_tokens(invoke_kwargs)
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using OpenAI asynchronously."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs = normalize_openai_token_kwargs(
            self._model_name,
            max_tokens,
            kwargs,
            prefer_max_tokens_argument=True,
        )
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = []
        for caption, image in image_caption_pairs:
            content.append({"type": "text", "text": caption})
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        invoke_kwargs = normalize_openai_token_kwargs(
            self._model_name,
            max_tokens,
            kwargs,
            prefer_max_tokens_argument=True,
        )
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "openai"


class AnthropicVLM(BaseVisionLanguageModel):
    """Anthropic VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize Anthropic VLM.

        Args:
            api_key: Anthropic API key
            model: Model name (defaults to claude-opus-4-6)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is required for AnthropicVLM. "
                "Install with: pip install langchain-anthropic"
            ) from e

        self._model_name = model or _DEFAULT_ANTHROPIC_MODEL
        timeout_s = self._set_bounded_request_timeout(timeout, label="Anthropic VLM")
        self.chat_model = ChatAnthropic(
            model_name=self._model_name,
            api_key=api_key,  # type: ignore[arg-type]
            timeout=timeout_s,
            **kwargs,
        )

    @traced_vlm(name="vlm.generate", system="anthropic", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Anthropic."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Anthropic asynchronously."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = []
        for caption, image in image_caption_pairs:
            content.append({"type": "text", "text": caption})
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_completion_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return response.content  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "anthropic"


class GeminiVLM(BaseVisionLanguageModel):
    """Google Gemini VLM with support for interleaved text and images."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        **kwargs: Any,
    ):
        """Initialize Gemini VLM.

        Args:
            api_key: Google API key (loads from GOOGLE_API_KEY or GEMINI_API_KEY
                env var if None)
            model: Model name (defaults to gemini-3-pro-preview)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for GeminiVLM. "
                "Install with: pip install langchain-google-genai"
            ) from e

        api_key = get_env_api_key_for_backend("gemini", api_key)
        if api_key is None:
            raise ValueError(
                "API key is required. Provide via api_key parameter or "
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        self._model_name = model or _DEFAULT_GEMINI_MODEL
        timeout_s = self._set_bounded_request_timeout(timeout, label="Gemini VLM")
        self.chat_model = ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=api_key,
            timeout=timeout_s,
            **kwargs,
        )

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Extract text from response content (handles thinking model list responses)."""
        return extract_text_content(content)

    @traced_vlm(name="vlm.generate", system="gemini", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Gemini."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_output_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_output_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response using Gemini asynchronously."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if images:
            for image in images:
                pil_image = self._load_image(image)
                base64_image = image_to_base64(pil_image)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content if images else prompt),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_output_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_output_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = await self.chat_model.ainvoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        """Generate response from image-caption pairs with true interleaved support."""
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = []
        for caption, image in image_caption_pairs:
            content.append({"type": "text", "text": caption})
            pil_image = self._load_image(image)
            base64_image = image_to_base64(pil_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        invoke_kwargs: dict[str, Any] = {}
        if temperature is not None:
            invoke_kwargs["temperature"] = temperature
        if max_tokens is not None:
            invoke_kwargs["max_output_tokens"] = max_tokens

        for key, value in kwargs.items():
            if key not in ("max_tokens", "max_output_tokens"):
                invoke_kwargs[key] = value

        invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if v is not None}
        response = self.chat_model.invoke(messages, **invoke_kwargs)

        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "gemini"


def create_gradio_vlm(
    endpoint: str | None = None,
    api_name: str = _DEFAULT_GRADIO_API_NAME,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    use_single_image_api: bool = True,
    **kwargs: Any,
) -> GradioVLM:
    """Create a Gradio-based VLM.

    Args:
        endpoint: Gradio server endpoint (uses default if not provided)
        api_name: API endpoint name
        timeout: Request timeout in seconds
        use_single_image_api: If True, force single-image request format; if
            False, try multi-image first.
        **kwargs: Additional configuration

    Returns:
        Configured GradioVLM instance
    """
    return GradioVLM(
        endpoint=endpoint or _DEFAULT_GRADIO_ENDPOINT,
        api_name=api_name,
        timeout=timeout,
        use_single_image_api=use_single_image_api,
        **kwargs,
    )


class LangChainChatVLM(BaseVisionLanguageModel):
    """Generic multimodal adapter for a plugin-provided LangChain chat model."""

    def __init__(
        self,
        chat_model: Any,
        *,
        model_name: str,
        backend_name: str,
        request_style: str = "openai",
        request_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__()
        if request_timeout_seconds is not None:
            self._set_bounded_request_timeout(
                request_timeout_seconds,
                label="LangChain chat VLM",
            )
        self.chat_model = chat_model
        self._model_name = model_name
        self._backend_name = backend_name
        self._request_style = request_style

    def _content(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None,
    ) -> str | list[dict[str, Any]]:
        if not images:
            return prompt
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = image_to_base64(self._load_image(image))
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        return content

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Extract text from plain or Responses API content."""
        return extract_text_content(content)

    def _invoke_kwargs(
        self,
        temperature: float | None,
        max_tokens: int | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        options = dict(kwargs)
        if self._request_style == "bedrock":
            inference_config: dict[str, Any] = {}
            if temperature is not None:
                inference_config["temperature"] = float(temperature)
            top_p = options.pop("top_p", None)
            if top_p is not None:
                inference_config["topP"] = float(top_p)
            selected_max_tokens = options.pop(
                "max_completion_tokens", options.pop("max_tokens", max_tokens)
            )
            if selected_max_tokens is not None:
                inference_config["maxTokens"] = int(selected_max_tokens)
            stop_sequences = options.pop("stop_sequences", None)
            if stop_sequences is not None:
                inference_config["stopSequences"] = stop_sequences
            if inference_config:
                options["inference_config"] = inference_config
            return options

        if self._request_style == "openai_responses_reasoning":
            options.pop("temperature", None)
            options.pop("top_p", None)
            temperature = None

        if temperature is not None:
            options["temperature"] = temperature
        return normalize_openai_token_kwargs(self._model_name, max_tokens, options)

    @traced_vlm(name="vlm.generate", system="langchain_chat", operation="generate")
    def generate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self._content(prompt, images)),
        ]
        response = self.chat_model.invoke(
            messages,
            **self._invoke_kwargs(temperature, max_tokens, kwargs),
        )
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    @traced_vlm(name="vlm.generate", system="langchain_chat", operation="generate")
    async def agenerate(
        self,
        prompt: str,
        images: list[str | Path | PILImage.Image | np.ndarray] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self._content(prompt, images)),
        ]
        response = await self.chat_model.ainvoke(
            messages,
            **self._invoke_kwargs(temperature, max_tokens, kwargs),
        )
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    @traced_vlm(
        name="vlm.generate",
        system="langchain_chat",
        operation="generate_with_pairs",
    )
    def generate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = []
        for caption, image in image_caption_pairs:
            content.append({"type": "text", "text": caption})
            encoded = image_to_base64(self._load_image(image))
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        response = self.chat_model.invoke(
            messages,
            **self._invoke_kwargs(temperature, max_tokens, kwargs),
        )
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    @traced_vlm(
        name="vlm.generate",
        system="langchain_chat",
        operation="generate_with_pairs",
    )
    async def agenerate_with_image_caption_pairs(
        self,
        image_caption_pairs: list[tuple[str, str | Path | PILImage.Image | np.ndarray]],
        final_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = _DEFAULT_TEMPERATURE,
        max_tokens: int | None = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[dict[str, Any]] = []
        for caption, image in image_caption_pairs:
            content.append({"type": "text", "text": caption})
            encoded = image_to_base64(self._load_image(image))
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        content.append({"type": "text", "text": final_prompt})
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        response = await self.chat_model.ainvoke(
            messages,
            **self._invoke_kwargs(temperature, max_tokens, kwargs),
        )
        self._last_token_usage = TokenUsage.from_langchain_response(
            response, model_name=self._model_name, invocation_type="vlm"
        )
        return self._extract_text_content(response.content)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return self._backend_name


@traced_vlm(name="vlm.create", system="multi", operation="create")
def create_vlm(
    backend: str,
    **kwargs: Any,
) -> BaseVisionLanguageModel:
    """Create a Vision-Language Model for the specified backend.

    Available backends depend on the installation. Public providers are always
    available; optional provider packages contribute factories via entry points.

    Args:
        backend: Backend name (use ``list_vlm_backends()`` to see available)
        **kwargs: Backend-specific arguments.

    Returns:
        Configured VLM instance

    Raises:
        ValueError: If backend is not supported or required parameters missing

    Examples:
        ```python
        vlm = create_vlm("nim", api_key="your-key")
        response = vlm.generate(
            prompt="What is in this image?",
            images=["path/to/image.jpg"]
        )
        ```
    """
    from world_understanding.functions.models.backends.registry import get_vlm_factory

    factory = get_vlm_factory(backend)
    return factory(**kwargs)
