# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base embedding model implementation.

This module provides the base class for all embedding models (text, image, multimodal).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image as PILImage

# Default configurations
_DEFAULT_EMBEDDING_DIMENSION = 1024
_DEFAULT_TIMEOUT_SECONDS = 120.0


class BaseEmbeddingModel(ABC):
    """Base class for all embedding models.

    This class provides common functionality and interface for text, image,
    and multimodal embedding models.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        embedding_dimension: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the base embedding model.

        Args:
            api_key: API key for the service
            model: Model ID to use for embeddings
            base_url: Base URL for the API (optional)
            timeout: Request timeout in seconds
            embedding_dimension: Dimension of embeddings (optional)
            **kwargs: Additional configuration options
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._embedding_dim = embedding_dimension or _DEFAULT_EMBEDDING_DIMENSION

        # Initialize OpenAI client (most services use OpenAI-compatible API)
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return int(self._embedding_dim)

    def _update_embedding_dimension(self, actual_dimension: int) -> None:
        """Update the embedding dimension based on actual API response.

        Args:
            actual_dimension: The actual dimension returned by the API
        """
        self._embedding_dim = actual_dimension

    @classmethod
    @abstractmethod
    def list_available_models(cls) -> list[str]:
        """List all available models for this embedding type.

        Returns:
            List of available model names
        """
        pass

    def _validate_model(self, model: str, available_models: list[str]) -> None:
        """Validate that the model is supported.

        Args:
            model: Model name to validate
            available_models: List of available models

        Raises:
            ValueError: If model is not supported
        """
        if model not in available_models:
            available = ", ".join(available_models)
            raise ValueError(
                f"Unsupported model: {model}. Available models: {available}"
            )

    def _get_embedding_vectors_from_response(self, response) -> list[np.ndarray]:
        """Extract embedding vectors from API response.

        Args:
            response: API response object

        Returns:
            List of embedding vectors as numpy arrays
        """
        embedding_vectors = [data.embedding for data in response.data]
        return [np.array(vector, dtype=np.float32) for vector in embedding_vectors]

    def _init_embedding_dimension(self) -> None:
        """Initialize the embedding dimension based on the model."""
        embedding_vectors = self.embed_text("dummy")
        self._update_embedding_dimension(embedding_vectors.shape[0])

    def embed_text(self, text: str | Path, **kwargs: Any) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text string or file path to embed
            **kwargs: Additional arguments to pass to the model

        Returns:
            Embedding vector as np.ndarray

        Raises:
            NotImplementedError: If the model doesn't support text embedding
        """
        try:
            result = self.embed_texts([text], **kwargs)
            return result[0]
        except AttributeError as e:
            raise NotImplementedError(
                "This model doesn't support text embedding"
            ) from e

    def embed_texts(self, texts: list[str | Path], **kwargs: Any) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of text strings or file paths to embed
            **kwargs: Additional arguments to pass to the model

        Returns:
            List of embedding vectors as numpy arrays

        Raises:
            NotImplementedError: If the model doesn't support text embedding
        """
        raise NotImplementedError("This model doesn't support text embedding")

    def _load_text(self, text: str | Path) -> str:
        """Load text from various input formats."""
        if isinstance(text, str):
            return text
        elif isinstance(text, Path):
            if text.exists() and text.is_file():
                return text.read_text(encoding="utf-8")
            else:
                raise ValueError(f"Invalid text path: {text}")
        else:
            raise ValueError(
                f"Unsupported text type: {type(text)}. Expected str or Path."
            )

    def embed_image(
        self, image: str | Path | PILImage.Image | np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Generate embedding for a single image.

        Args:
            image: Image to embed (path, PIL Image, or np.ndarray)
            **kwargs: Additional arguments to pass to the model

        Returns:
            Embedding vector as np.ndarray

        Raises:
            NotImplementedError: If the model doesn't support image embedding
        """
        try:
            result = self.embed_images([image], **kwargs)
            return result[0]
        except AttributeError as e:
            raise NotImplementedError(
                "This model doesn't support image embedding"
            ) from e

    def embed_images(
        self, images: list[str | Path | PILImage.Image | np.ndarray], **kwargs: Any
    ) -> list[np.ndarray]:
        """Generate embeddings for multiple images.

        Args:
            images: List of images (path, PIL Image, or np.ndarray) to embed
            **kwargs: Additional arguments to pass to the model

        Returns:
            List of embedding vectors as numpy arrays

        Raises:
            NotImplementedError: If the model doesn't support image embedding
        """
        raise NotImplementedError("This model doesn't support image embedding")

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
            if Path(image).exists() and Path(image).is_file():
                return PILImage.open(image).convert("RGB")
            else:
                raise ValueError("Unsupported image type")
        elif isinstance(image, PILImage.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return PILImage.fromarray(image).convert("RGB")
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, PIL Image, or numpy array."
            )

    @classmethod
    def create_from_env(
        cls,
        backend: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> "BaseEmbeddingModel":
        """Create an embedding model from environment variables.

        Args:
            backend: Registered backend name.
            model: Model ID (optional, defaults used if not specified)
            **kwargs: Additional backend-specific arguments

        Returns:
            Configured embedding model instance

        Raises:
            ValueError: If backend is not supported or required parameters missing
        """
        import world_understanding.functions.models.backends  # noqa: F401
        from world_understanding.utils.credentials import get_env_api_key_for_backend

        api_key = get_env_api_key_for_backend(backend)

        if api_key is None:
            raise ValueError(f"API key is required for {backend} backend")

        # Set default base URL for NIM
        if backend == "nim" and "base_url" not in kwargs:
            kwargs["base_url"] = "https://integrate.api.nvidia.com/v1"

        return cls(api_key=api_key, model=model or cls.DEFAULT_MODEL, **kwargs)
