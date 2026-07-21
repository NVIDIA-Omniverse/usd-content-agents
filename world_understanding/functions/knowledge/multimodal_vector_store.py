# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vector store for documents with either text or image content using FAISS.

This module provides a vector store implementation for documents containing
either text OR image content (not both), enabling efficient similarity search
using FAISS (Facebook AI Similarity Search) with unified embeddings.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from world_understanding.functions.knowledge.base_vector_store import (
    BaseDocument,
    BaseSearchResult,
    BaseVectorStore,
)
from world_understanding.functions.models.multimodal_embedding_models import (
    BaseMultimodalEmbeddingModel,
    create_multimodal_embedding_model,
)

# Configure logger
logger = logging.getLogger(__name__)


class MultimodalVectorStore(BaseVectorStore):
    """Vector store for documents with either text or image content using FAISS.

    This class provides functionality to:
    - Add documents containing either text OR image content to a vector index
    - Search for similar documents using unified embeddings
    - Save and load the index for persistence
    - Update or remove documents from the index

    Attributes:
        embedding_model: The multimodal model used to generate embeddings
        dimension: The dimension of the embeddings
        index: The FAISS index for similarity search
        metadata_store: Storage for document metadata
    """

    def __init__(
        self,
        embedding_model: BaseMultimodalEmbeddingModel,
        index_type: str = "IndexFlatL2",
        normalize_embeddings: bool = False,
        nlist: int = 100,
        M: int = 32,
    ):
        """Initialize the multimodal vector store.

        Args:
            embedding_model: Multimodal model to generate embeddings
            index_type: Type of FAISS index to use. Options:
                - "IndexFlatL2": Exact search using L2 distance
                - "IndexFlatIP": Exact search using inner product
                - "IndexIVFFlat": Approximate search with inverted file index
                - "IndexHNSWFlat": Approximate search with HNSW graph
            normalize_embeddings: Whether to normalize embeddings before indexing
            nlist: Number of clusters for IVF index
            M: Number of bi-directional links for HNSW index
        """
        super().__init__(
            embedding_model=embedding_model,
            index_type=index_type,
            normalize_embeddings=normalize_embeddings,
            nlist=nlist,
            M=M,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MultimodalVectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory path containing the saved index

        Returns:
            Loaded MultimodalVectorStore instance

        Raises:
            FileNotFoundError: If the saved files don't exist
            ValueError: If model compatibility validation fails
        """
        return cls._load(path, create_multimodal_embedding_model)


def build_multimodal_vector_store(
    text_source: str | Path | list[str | Path] | None = None,
    image_source: (
        str | Path | PILImage.Image | list[str | Path | PILImage.Image] | None
    ) = None,
    embedding_model: BaseMultimodalEmbeddingModel | None = None,
    index_type: str = "IndexFlatL2",
    normalize_embeddings: bool = False,
    metadata_extractor: Callable[[str | Path], dict[str, Any]] | None = None,
    text_extensions: tuple[str, ...] | None = (
        ".txt",
        ".md",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".json",
        ".xml",
        ".csv",
    ),
    image_extensions: tuple[str, ...] | None = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".gif",
        ".webp",
        ".svg",
    ),
    recursive: bool = True,
    image_embedding_type: str = "image",
    caption_prompt: str = "Describe this image in detail.",
    system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
    vlm_backend: str = "nim",
    vlm_model: str | None = None,
    vlm_api_key: str | None = None,
) -> MultimodalVectorStore:
    """Build a multimodal vector store from texts and/or images.

    This method provides a convenient way to create and populate a
    vector store from various sources:
    - Text strings or file paths
    - Image file paths or PIL Image objects
    - Mixed sources

    Args:
        text_source: Source of texts - can be:
            - str: Text string or file path
            - Path: Text file path
            - list: List of text strings or file paths
            - None: No text sources
        image_source: Source of images - can be:
            - str: Image file path
            - Path: Image file path
            - PILImage.Image: PIL Image object
            - list: List of image paths or PIL Images
            - None: No image sources
        embedding_model: Multimodal model to generate embeddings. If None, uses
            default NIM embedding model.
        index_type: Type of FAISS index (default: "IndexFlatL2")
        normalize_embeddings: Whether to normalize embeddings (default: False)
        metadata_extractor: Optional function to extract metadata from
            file paths. Should accept a path and return a dict.
        text_extensions: Tuple of valid text extensions (with dots).
        image_extensions: Tuple of valid image extensions (with dots).
        recursive: If True and sources are directories, scan recursively
            (default: True)
        image_embedding_type: How to handle images ("text" for captioning,
            "image" for direct embedding)
        caption_prompt: Prompt to use for image captioning
        system_prompt: System instructions for the VLM
        vlm_backend: Registered VLM backend to use
        vlm_model: Model to use (uses backend default if None)
        vlm_api_key: API key for the VLM backend (uses env var if None)

    Returns:
        MultimodalVectorStore populated with the provided texts and images

    Raises:
        ValueError: If sources are invalid or no content found
        FileNotFoundError: If directories don't exist

    Example:
        >>> # From text and image lists
        >>> store = build_multimodal_vector_store(
        ...     text_source=["Text 1", "Text 2", "path/to/text3.txt"],
        ...     image_source=["image1.jpg", "image2.png"]
        ... )
        >>>
        >>> # From directories
        >>> store = build_multimodal_vector_store(
        ...     text_source="path/to/texts/",
        ...     image_source="images/"
        ... )
        >>>
        >>> # Mixed sources
        >>> store = build_multimodal_vector_store(
        ...     text_source={"doc1": "Content 1", "doc2": "Content 2", "doc3": "path/to/text3.txt"},
        ...     image_source={"img1": "path/to/image1.jpg"}
        ... )
    """
    # Use the base class method to build the vector store
    if embedding_model is None:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable must be set when "
                "embedding_model is not provided"
            )
        embedding_model = create_multimodal_embedding_model(
            backend="nim", api_key=api_key
        )
        if not embedding_model:
            raise ValueError(
                "Failed to create embedding model. "
                "Please check your API key and try again."
            )

    return MultimodalVectorStore.build_vector_store(
        embedding_model=embedding_model,
        text_source=text_source,
        image_source=image_source,
        index_type=index_type,
        normalize_embeddings=normalize_embeddings,
        metadata_extractor=metadata_extractor,
        text_extensions=text_extensions,
        image_extensions=image_extensions,
        recursive=recursive,
        image_embedding_type=image_embedding_type,
        caption_prompt=caption_prompt,
        system_prompt=system_prompt,
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
    )


def find_similar_documents_from_vector_store(
    query: str | np.ndarray | Path | PILImage.Image,
    query_type: str,
    store: str | Path | MultimodalVectorStore,
    k: int = 5,
    filter_metadata: dict[str, Any] | None = None,
    embedding_type: str = "image",
    caption_prompt: str = "Describe this image in detail.",
    system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
    vlm_backend: str = "nim",
    vlm_model: str | None = None,
    vlm_api_key: str | None = None,
) -> list[BaseSearchResult]:
    """Find similar documents from a multimodal vector store.

    This function searches for documents similar to the query using vector
    similarity. It can work with either a saved vector store (by path) or
    an existing MultimodalVectorStore instance.

    Args:
        query: Query to find similar documents for. Can be:
            - str: Text string to search for or Path to an image file
            - np.ndarray: Pre-computed embedding vector
            - Path: Path to an image file
            - PILImage.Image: PIL Image object
        query_type: Type of query ("text", "image", "embedding")
        store: Vector store to search in. Can be:
            - str/Path: Path to a saved vector store directory
            - MultimodalVectorStore: An existing vector store instance
        k: Number of similar documents to return (default: 5)
        filter_metadata: Optional metadata filters for matching.
            For string values, uses case-insensitive contains matching.
            For non-string values, uses exact matching.
        embedding_type: Type of embedding to use for image queries ("text" or "image").
            If "text", the image will be captioned first, then embedded as text.
            If "image", the image will be embedded directly.
        caption_prompt: Prompt to use for image captioning
        system_prompt: System instructions for the VLM
        vlm_backend: VLM backend to use ("azure_openai" or "nim")
        vlm_model: Model to use (uses backend default if None)
        vlm_api_key: API key for the VLM backend (uses env var if None)

    Returns:
        List of MultimodalSearchResult objects ordered by similarity (most similar first)

    Raises:
        FileNotFoundError: If the vector store path doesn't exist
        ValueError: If NVIDIA_API_KEY not set when loading from path

    Example:
        >>> # Find similar documents using a saved store path
        >>> results = find_similar_documents_from_vector_store(
        ...     query="machine learning algorithms",
        ...     query_type="text",
        ...     store="my_store/",
        ...     k=10
        ... )
        >>>
        >>> # Using an existing store instance
        >>> store = build_multimodal_vector_store(
        ...     text_sources=["Text 1", "Text 2", "path/to/text3.txt"],
        ...     image_sources=["image1.jpg"]
        ... )
        >>> results = find_similar_documents_from_vector_store(
        ...     query="machine learning algorithms",
        ...     query_type="text",
        ...     store=store,
        ...     k=10
        ... )
        >>>
        >>> # Search by image
        >>> results = find_similar_documents_from_vector_store(
        ...     query="path/to/query_image.jpg",
        ...     store="my_store/",
        ...     query_type="image",
        ...     embedding_type="image"
        ... )
        >>>
        >>> # Search by image with text embedding (caption-based)
        >>> results = find_similar_documents_from_vector_store(
        ...     query="path/to/query_image.jpg",
        ...     store="my_store/",
        ...     query_type="image",
        ...     embedding_type="text"
        ... )
    """
    if isinstance(store, str | Path):
        vector_store = MultimodalVectorStore.load(store)
    else:
        vector_store = store
    return vector_store.find_similar_documents(
        query=query,
        query_type=query_type,
        k=k,
        filter_metadata=filter_metadata,
        embedding_type=embedding_type,
        caption_prompt=caption_prompt,
        system_prompt=system_prompt,
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
    )


def collect_documents_from_vector_store(
    store: str | Path | MultimodalVectorStore,
    filter_metadata: dict[str, Any] | None = None,
) -> list[BaseDocument]:
    """Collect documents from a multimodal vector store with optional metadata filtering.

    This function provides a convenient way to collect documents from a vector store
    with optional metadata filtering. It can work with either a saved vector store
    (by path) or an existing MultimodalVectorStore instance.

    Args:
        store: Vector store to collect documents from. Can be:
            - str/Path: Path to a saved vector store directory
            - MultimodalVectorStore: An existing vector store instance
        filter_metadata: Optional metadata filters for matching.
            For string values, uses case-insensitive contains matching.
            For non-string values, uses exact matching.
            If None, returns all documents.

    Returns:
        List of MultimodalDocument objects that match the filter criteria

    Raises:
        FileNotFoundError: If the vector store path doesn't exist
        ValueError: If NVIDIA_API_KEY not set when loading from path

    Example:
        >>> # Collect all documents from a saved store
        >>> all_docs = collect_documents_from_vector_store("my_store/")
        >>>
        >>> # Collect documents with specific metadata
        >>> filtered_docs = collect_documents_from_vector_store(
        ...     "my_store/",
        ...     {"category": "research", "year": 2023}
        ... )
        >>>
        >>> # Using an existing store instance
        >>> store = build_multimodal_vector_store(
        ...     text_sources=["Text 1", "Text 2"],
        ...     image_sources=["image1.jpg"]
        ... )
        >>> docs = collect_documents_from_vector_store(
        ...     store,
        ...     {"type": "text"}
        ... )
        >>>
        >>> # Collect documents with string contains matching
        >>> docs = collect_documents_from_vector_store(
        ...     "my_store/",
        ...     {"title": "machine learning"}
        ... )
    """
    if isinstance(store, str | Path):
        vector_store = MultimodalVectorStore.load(store)
    else:
        vector_store = store
    return vector_store.collect_documents(filter_metadata=filter_metadata)
