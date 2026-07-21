# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text embedding model implementations."""

# For type hints
import os
from pathlib import Path
from typing import Any

import numpy as np
from openai import AzureOpenAI

from world_understanding.functions.models.base_embedding_model import BaseEmbeddingModel


class BaseTextEmbeddingModel(BaseEmbeddingModel):
    """Base text embedding model."""

    pass


class OpenAITextEmbeddingModel(BaseTextEmbeddingModel):
    """OpenAI-compatible text embedding model."""

    # Available OpenAI models with their specifications
    AVAILABLE_MODELS = [
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ]

    DEFAULT_MODEL = "text-embedding-3-large"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI text embedding model.

        Args:
            api_key: API key for the service
            model: Model ID to use for embeddings
            base_url: Base URL for the API (optional)
            **kwargs: Additional configuration options
        """
        if not self.AVAILABLE_MODELS:
            raise NotImplementedError(
                "Currently provide no dedicated text embedding models. "
            )

        # Validate model is supported
        self._validate_model(model, self.AVAILABLE_MODELS)

        # Initialize parent class
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs,
        )

        self._init_embedding_dimension()

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available OpenAI text embedding models.

        Returns:
            List of available OpenAI model names
        """
        return cls.AVAILABLE_MODELS.copy()

    def embed_texts(self, texts: list[str | Path]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        texts = [self._load_text(text) for text in texts]
        # Call embeddings API for all texts at once
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float",
        )

        return self._get_embedding_vectors_from_response(response)


class NIMTextEmbeddingModel(BaseTextEmbeddingModel):
    """NVIDIA NIM text embedding model."""

    # Available NIM models with their requirements
    AVAILABLE_MODELS = [
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "nvidia/llama-nemotron-embed-vl-1b-v2",  # Multimodal model
        "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",  # Multimodal model
    ]

    DEFAULT_MODEL = "nvidia/nv-embedqa-mistral-7b-v2"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ):
        """Initialize NIM text embedding model.

        Args:
            api_key: NVIDIA API key
            model: Model ID to use for embeddings
            **kwargs: Additional configuration options
        """
        if not self.AVAILABLE_MODELS:
            raise NotImplementedError(
                "Currently provide no dedicated text embedding models. "
            )

        # Validate model is supported
        self._validate_model(model, self.AVAILABLE_MODELS)

        # Override base_url if provided in kwargs, otherwise use NIM default
        base_url = kwargs.pop("base_url", "https://integrate.api.nvidia.com/v1")

        # Initialize parent class
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs,
        )

        self._init_embedding_dimension()

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available NIM models with their requirements.

        Returns:
            List of available NIM model names
        """
        return cls.AVAILABLE_MODELS.copy()

    def embed_texts(
        self, texts: list[str | Path], input_type: str = "passage"
    ) -> list[np.ndarray]:
        """Generate embeddings for multiple texts using NIM-specific parameters.

        Args:
            texts: List of text strings or file paths to embed
            input_type: Type of input - 'passage' or 'query'

        Returns:
            List of embedding vectors as numpy arrays

        Raises:
            ValueError: If the model is not supported
        """
        texts = [self._load_text(text) for text in texts]

        # Handle different NIM model types
        if "vlm-embed" in self.model or "embed-vl" in self.model:
            # VLM models require modality parameter - must match input length
            modalities = ["text"] * len(texts)
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format="float",
                extra_body={
                    "modality": modalities,
                    "input_type": input_type,
                    "truncate": "NONE",
                },
            )
        else:
            # Other NIM models require input_type in extra_body
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "NONE"},
            )

        return self._get_embedding_vectors_from_response(response)


class AzureOpenAITextEmbeddingModel(BaseTextEmbeddingModel):
    """Azure OpenAI text embedding model."""

    # Available OpenAI models with their specifications
    AVAILABLE_MODELS = ["text-embedding-3-large", "text-embedding-3-small", "embedding"]

    DEFAULT_MODEL = "text-embedding-3-large"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ):
        """Initialize an Azure OpenAI text embedding model.

        Args:
            api_key: Azure OpenAI API key
            model: Model ID to use for embeddings
            **kwargs: Additional configuration options
        """
        if not self.AVAILABLE_MODELS:
            raise NotImplementedError(
                "Currently provide no dedicated text embedding models. "
            )

        # Validate model is supported
        self._validate_model(model, self.AVAILABLE_MODELS)

        self.api_key = api_key
        self.model = model
        self.base_url = kwargs.pop("base_url", "")
        if not self.base_url:
            raise ValueError("base_url is required for AzureOpenAITextEmbeddingModel")
        self.timeout = kwargs.pop("timeout", 120.0)
        self._embedding_dim = kwargs.pop("embedding_dimension", 1024)

        # Initialize OpenAI client for Azure endpoint
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "azure_endpoint": self.base_url,
            "api_version": "2025-03-01-preview",
        }
        self.client = AzureOpenAI(**client_kwargs)

        self._init_embedding_dimension()

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available OpenAI models with their specifications.

        Returns:
            List of available OpenAI model names
        """
        return cls.AVAILABLE_MODELS.copy()

    def embed_texts(self, texts: list[str | Path]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        texts = [self._load_text(text) for text in texts]

        # Call embeddings API for all texts at once
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float",
        )

        return self._get_embedding_vectors_from_response(response)


def create_nim_text_embedding_model(
    api_key: str,
    model: str = NIMTextEmbeddingModel.DEFAULT_MODEL,
    **kwargs: Any,
) -> NIMTextEmbeddingModel:
    """Create NVIDIA NIM text embedding model.

    Args:
        api_key: NVIDIA API key
        model: Model ID to use for embeddings
        **kwargs: Additional arguments to pass to NIMTextEmbeddingModel

    Returns:
        Configured NIMTextEmbeddingModel instance
    """
    return NIMTextEmbeddingModel(api_key=api_key, model=model, **kwargs)


def list_nim_text_models() -> list[str]:
    """List all available NIM models with their requirements.

    Returns:
        List of available NIM model names
    """
    return NIMTextEmbeddingModel.list_available_models()


def create_openai_text_embedding_model(
    api_key: str,
    model: str = OpenAITextEmbeddingModel.DEFAULT_MODEL,
    base_url: str | None = None,
    **kwargs: Any,
) -> OpenAITextEmbeddingModel:
    """Create OpenAI text embedding model.

    Args:
        api_key: OpenAI API key
        model: Model ID to use for embeddings
        base_url: Base URL for the API (optional)
        **kwargs: Additional arguments to pass to OpenAITextEmbeddingModel

    Returns:
        Configured OpenAITextEmbeddingModel instance
    """
    return OpenAITextEmbeddingModel(
        api_key=api_key,
        model=model,
        base_url=base_url,
        **kwargs,
    )


def create_azure_openai_text_embedding_model(
    api_key: str,
    model: str = AzureOpenAITextEmbeddingModel.DEFAULT_MODEL,
    **kwargs: Any,
) -> AzureOpenAITextEmbeddingModel:
    """Create an Azure OpenAI text embedding model.

    Args:
        api_key: Azure OpenAI API key
        model: Model ID to use for embeddings
        **kwargs: Additional arguments to pass to AzureOpenAITextEmbeddingModel

    Returns:
        Configured AzureOpenAITextEmbeddingModel instance
    """
    return AzureOpenAITextEmbeddingModel(
        api_key=api_key,
        model=model,
        **kwargs,
    )


def list_azure_openai_text_models() -> list[str]:
    """List all available Azure OpenAI models with their specifications.

    Returns:
        List of available Azure OpenAI model names
    """
    return AzureOpenAITextEmbeddingModel.list_available_models()


def create_text_embedding_model(
    backend: str, api_key: str | None = None, model: str | None = None, **kwargs: Any
) -> BaseTextEmbeddingModel:
    """Create a text embedding model for the specified backend.

    Args:
        backend: Registered backend name.
        api_key: API key for the backend
        model: Model ID (optional, defaults used if not specified)
        **kwargs: Additional backend-specific arguments

    Returns:
        Configured text embedding model instance

    Raises:
        ValueError: If backend is not supported or required parameters missing
    """
    if backend == "nim":
        if api_key is None:
            api_key = os.getenv("NVIDIA_API_KEY")
        if api_key is None:
            raise ValueError("API key is required for NIM backend")
        return create_nim_text_embedding_model(
            api_key=api_key,
            model=model or NIMTextEmbeddingModel.DEFAULT_MODEL,
            **kwargs,
        )

    elif backend == "openai":
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key is required for OpenAI backend")
        return create_openai_text_embedding_model(
            api_key=api_key,
            model=model or OpenAITextEmbeddingModel.DEFAULT_MODEL,
            **kwargs,
        )

    import world_understanding.functions.models.backends  # noqa: F401
    from world_understanding.functions.models.backends.registry import (
        get_text_embedding_factory,
    )

    factory = get_text_embedding_factory(backend)
    factory_kwargs: dict[str, Any] = {"api_key": api_key, **kwargs}
    if model is not None:
        factory_kwargs["model"] = model
    return factory(**factory_kwargs)
