# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base vector store implementation for similarity search using FAISS.

This module provides the core vector store functionality that can be shared
across different types of vector stores (text, image, multimodal).
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from world_understanding.functions.models.base_embedding_model import BaseEmbeddingModel

# Configure logger
logger = logging.getLogger(__name__)


class BaseDocument(BaseModel):
    """Base document class for vector store documents."""

    model_config = {"arbitrary_types_allowed": True}

    text_content: str | None = Field(None, description="Text content of the document")
    text_path: str | None = Field(None, description="Path to the text file")
    image_path: str | None = Field(None, description="Path to the image file")
    image_data: PILImage.Image | None = Field(
        None, description="Image data if available in memory"
    )
    document_id: str = Field(..., description="Unique identifier for the document")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the document"
    )

    def has_text(self) -> bool:
        """Check if document has text content."""
        return (
            self.text_content is not None
            and bool(self.text_content.strip())
            or self.text_path is not None
        )

    def has_image(self) -> bool:
        """Check if document has image content."""
        return self.image_path is not None or self.image_data is not None

    def get_content_type(self) -> str:
        """Get the content type of this document."""
        if self.has_text() and self.has_image():
            return "multimodal"  # has both text and image
        elif self.has_text():
            return "text"
        elif self.has_image():
            return "image"
        else:
            return "none"


class BaseMetadata(BaseModel):
    """Base metadata associated with a document in the vector store."""

    document: BaseDocument = Field(..., description="The document")
    embedding_id: int | None = Field(
        None, description="Index of the embedding in the FAISS index"
    )


class BaseSearchResult(BaseModel):
    """Base result from a similarity search."""

    model_config = {"arbitrary_types_allowed": True}

    document: BaseDocument = Field(..., description="The matched document")
    score: float = Field(..., description="Similarity score (lower is more similar)")
    rank: int = Field(..., description="Rank in the search results (0-based)")


class BaseVectorStore:
    """Base vector store implementation for documents using FAISS.

    This class provides the core functionality that can be shared across
    different types of vector stores (text, image, multimodal).

    Attributes:
        embedding_model: The model used to generate embeddings
        dimension: The dimension of the embeddings
        index: The FAISS index for similarity search
        metadata_store: Storage for document metadata
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        index_type: str = "IndexFlatL2",
        normalize_embeddings: bool = False,
        nlist: int = 100,
        M: int = 32,
    ):
        """Initialize the base vector store.

        Args:
            embedding_model: Model to generate embeddings
            index_type: Type of FAISS index to use. Options:
                - "IndexFlatL2": Exact search using L2 distance
                - "IndexFlatIP": Exact search using inner product
                - "IndexIVFFlat": Approximate search with inverted file index
                - "IndexHNSWFlat": Approximate search with HNSW graph
            normalize_embeddings: Whether to normalize embeddings before indexing
            nlist: Number of clusters for IVF index
            M: Number of bi-directional links for HNSW index
        """
        self.embedding_model = embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.index_type = index_type

        self.dimension = self.embedding_model.embedding_dimension
        self.nlist = nlist
        self.M = M

        # Create FAISS index based on type
        if index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexIVFFlat":
            # For IVF, we need to train on data first
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self._needs_training = True
        elif index_type == "IndexHNSWFlat":
            # HNSW parameters
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Metadata storage (ID -> metadata)
        self.metadata_store: dict[int, BaseMetadata] = {}
        self._next_id = 0
        self._needs_training = getattr(self, "_needs_training", False)

    def add_document(
        self,
        document: BaseDocument,
        embedding_type: str = "text",
    ) -> int:
        """Add a document to the vector store.

        Args:
            document: Document to add
            embedding_type: Type of embedding to generate ("text" or "image")

        Returns:
            The ID assigned to the document in the index

        Raises:
            ValueError: If the document cannot be processed
        """
        if not document.has_text() and not document.has_image():
            raise ValueError("Document must have at least text or image content")

        # Generate embedding based on document content
        if embedding_type == "text":
            if not document.has_text():
                raise ValueError("Document must have text content")
            embedding = self.embedding_model.embed_text(
                document.text_content or Path(document.text_path)
            )

        elif embedding_type == "image":
            if not document.has_image():
                raise ValueError("Document must have image content")
            embedding = self.embedding_model.embed_image(
                document.image_data or Path(document.image_path)
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        # Normalize if requested
        if self.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Ensure embedding is the right shape
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

        # Add to index
        embedding_id = self._next_id

        # Train IVF index before first add
        if self._needs_training and self.index.ntotal == 0:
            self._train_index(initial_vectors=[embedding])

        self.index.add(embedding)

        # Store metadata
        self.metadata_store[embedding_id] = BaseMetadata(
            document=document,
            embedding_id=embedding_id,
        )

        self._next_id += 1
        return embedding_id

    def add_documents(
        self,
        documents: list[BaseDocument],
        embedding_type: str | list[str],
    ) -> list[int]:
        """Add multiple documents to the vector store.

        Args:
            documents: List of documents to add
            embedding_type: Embedding type

        Returns:
            List of IDs assigned to the documents

        Raises:
            ValueError: If any document cannot be processed
        """
        ids = []
        if isinstance(embedding_type, str):
            embedding_type = [embedding_type] * len(documents)

        if len(embedding_type) != len(documents):
            raise ValueError(
                f"Number of embedding types ({len(embedding_type)}) "
                f"doesn't match number of documents ({len(documents)})"
            )

        for document, doc_embedding_type in zip(
            documents, embedding_type, strict=False
        ):
            doc_id = self.add_document(document, doc_embedding_type)
            ids.append(doc_id)

        return ids

    def add_text(
        self,
        text: str | Path,
        text_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a text-only document to the vector store.

        Args:
            text: Text content or file path to add
            text_id: Optional unique identifier for the text
            metadata: Optional metadata to associate with the text

        Returns:
            The ID assigned to the text in the index
        """
        if text_id is None:
            text_id = f"text_{self._next_id}"

        is_file = isinstance(text, Path)
        text_content = text.read_text(encoding="utf-8") if is_file else text

        document = BaseDocument(
            text_content=text_content,
            text_path=str(text) if is_file else None,
            document_id=text_id,
            metadata=metadata or {},
        )

        return self.add_document(document=document, embedding_type="text")

    def add_texts(
        self,
        texts: list[str | Path],
        text_ids: list[str] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add multiple text documents to the vector store.

        Args:
            texts: List of text contents or file paths to add
            text_ids: Optional list of unique identifiers (one per text)
            metadata_list: Optional list of metadata dicts (one per text)

        Returns:
            List of IDs assigned to the texts

        Raises:
            ValueError: If the number of items doesn't match
        """
        if text_ids is not None and len(text_ids) != len(texts):
            raise ValueError(
                f"Number of text IDs ({len(text_ids)}) "
                f"doesn't match number of texts ({len(texts)})"
            )

        if metadata_list is not None and len(metadata_list) != len(texts):
            raise ValueError(
                f"Number of metadata items ({len(metadata_list)}) "
                f"doesn't match number of texts ({len(texts)})"
            )

        ids = []
        for i, text in enumerate(texts):
            text_id = text_ids[i] if text_ids else None
            metadata = metadata_list[i] if metadata_list else None
            doc_id = self.add_text(text, text_id, metadata)
            ids.append(doc_id)

        return ids

    def add_image(
        self,
        image: str | Path | PILImage.Image,
        image_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_type: str = "image",
        caption_prompt: str = "Describe this image in detail.",
        system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
        vlm_backend: str = "nim",
        vlm_model: str | None = None,
        vlm_api_key: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Add an image-only document to the vector store.

        Args:
            image: Image to add (file path, or PIL Image)
            image_id: Optional unique identifier for the image
            metadata: Optional metadata to associate with the image
            embedding_type: Embedding type ("text" or "image")
            caption_prompt: Prompt to use for image captioning
            system_prompt: System instructions for the VLM
            vlm_backend: Registered VLM backend to use
            vlm_model: Model to use (uses backend default if None)
            vlm_api_key: API key for the VLM backend (uses env var if None)
            kwargs: Additional arguments to pass to the VLM

        Returns:
            The ID assigned to the image in the index
        """
        # Convert to PIL Image if needed
        if isinstance(image, PILImage.Image):
            image_path = None
            image_data = image
        elif isinstance(image, str | Path):
            image_path = str(image)
            image_data = None
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. Expected str (file path) or PIL.Image"
            )

        # Generate image ID if not provided
        if image_id is None:
            image_id = f"image_{self._next_id}"

        if embedding_type not in ["text", "image"]:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

        text_content = None
        if embedding_type == "text":
            # Generate text content from image caption
            from world_understanding.functions.cv.vlm import get_image_caption

            text_content = get_image_caption(
                image=image,
                caption_prompt=caption_prompt,
                system_prompt=system_prompt,
                vlm_backend=vlm_backend,
                vlm_model=vlm_model,
                vlm_api_key=vlm_api_key,
                **kwargs,
            )

        document = BaseDocument(
            text_content=text_content,
            image_path=image_path,
            image_data=image_data,
            document_id=image_id,
            metadata=metadata or {},
        )

        return self.add_document(document=document, embedding_type=embedding_type)

    def add_images(
        self,
        images: list[str | Path | PILImage.Image],
        image_ids: list[str] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
        embedding_type: str | list[str] = "image",
        caption_prompt: str = "Describe this image in detail.",
        system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
        vlm_backend: str = "nim",
        vlm_model: str | None = None,
        vlm_api_key: str | None = None,
        **kwargs: Any,
    ) -> list[int]:
        """Add multiple image documents to the vector store.

        Args:
            images: List of images to add (file paths, or PIL Images)
            image_ids: Optional list of unique identifiers (one per image)
            metadata_list: Optional list of metadata dicts (one per image)
            embedding_type: List of embedding types (one per image)
            caption_prompt: Prompt to use for image captioning
            system_prompt: System instructions for the VLM
            vlm_backend: Registered VLM backend to use
            vlm_model: Model to use (uses backend default if None)
            vlm_api_key: API key for the VLM backend (uses env var if None)
            kwargs: Additional arguments to pass to the VLM

        Returns:
            List of IDs assigned to the images

        Raises:
            ValueError: If the number of items doesn't match
        """
        if image_ids is not None and len(image_ids) != len(images):
            raise ValueError(
                f"Number of image IDs ({len(image_ids)}) "
                f"doesn't match number of images ({len(images)})"
            )

        if metadata_list is not None and len(metadata_list) != len(images):
            raise ValueError(
                f"Number of metadata items ({len(metadata_list)}) "
                f"doesn't match number of images ({len(images)})"
            )

        if isinstance(embedding_type, str):
            embedding_type = [embedding_type] * len(images)

        if len(embedding_type) != len(images):
            raise ValueError(
                f"Number of embedding types ({len(embedding_type)}) "
                f"doesn't match number of images ({len(images)})"
            )

        ids = []
        for i, image in enumerate(images):
            image_id = image_ids[i] if image_ids else None
            metadata = metadata_list[i] if metadata_list else None
            doc_id = self.add_image(
                image,
                image_id,
                metadata,
                embedding_type[i],
                caption_prompt,
                system_prompt,
                vlm_backend,
                vlm_model,
                vlm_api_key,
                **kwargs,
            )
            ids.append(doc_id)

        return ids

    def search(
        self,
        query_embedding: np.ndarray | None = None,
        query_text: str | None = None,
        query_image: str | Path | PILImage.Image | dict[str, Any] | None = None,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        embedding_type: str = "image",
        caption_prompt: str = "Describe this image in detail.",
        system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
        vlm_backend: str = "nim",
        vlm_model: str | None = None,
        vlm_api_key: str | None = None,
    ) -> list[BaseSearchResult]:
        """Search for similar documents in the vector store.

        Args:
            query_embedding: Query embedding
            query_text: Text query
            query_image: Image query
            k: Number of results to return
            filter_metadata: Optional metadata filters. For string values, uses case-insensitive
                contains matching. For non-string values, uses exact matching.
            embedding_type: Type of embedding to use for image queries ("text" or "image").
                If "text", the image will be captioned first, then embedded as text.
                If "image", the image will be embedded directly.
            caption_prompt: Prompt to use for image captioning
            system_prompt: System instructions for the VLM
            vlm_backend: VLM backend to use ("azure_openai" or "nim")
            vlm_model: Model to use (uses backend default if None)
            vlm_api_key: API key for the VLM backend (uses env var if None)

        Returns:
            List of search results ordered by similarity

        Raises:
            ValueError: If the index is empty or query is invalid
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add documents before searching.")

        # Generate query embedding
        if query_embedding is None:
            if query_text is not None:
                query_embedding = self.embedding_model.embed_text(query_text)
            elif query_image is not None:
                if embedding_type == "text":
                    # Generate text content from image caption
                    from world_understanding.functions.cv.vlm import get_image_caption

                    text_content = get_image_caption(
                        query_image,
                        caption_prompt,
                        system_prompt,
                        vlm_backend,
                        vlm_model,
                        vlm_api_key,
                    )
                    query_embedding = self.embedding_model.embed_text(text_content)
                elif embedding_type == "image":
                    query_embedding = self.embedding_model.embed_image(query_image)
                else:
                    raise ValueError(f"Unknown embedding type: {embedding_type}")
            else:
                raise ValueError("Query must be a text or image")
        elif not isinstance(query_embedding, np.ndarray):
            raise ValueError("Unsupported query type")

        # Normalize if needed
        if self.normalize_embeddings:
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        # Ensure correct shape
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Search in index (get more results for filtering)
        search_k = min(k * 3, self.index.ntotal) if filter_metadata else k
        distances, indices = self.index.search(query_embedding, search_k)

        # Build results
        results: list[BaseSearchResult] = []
        for _i, (dist, idx) in enumerate(zip(distances[0], indices[0], strict=False)):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            metadata_entry = self.metadata_store.get(idx)
            if metadata_entry is None:
                continue

            # Apply metadata filter if provided
            if filter_metadata:
                match = all(
                    self._matches_metadata_filter(
                        metadata_entry.document.metadata.get(key), value
                    )
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue

            # Create result
            results.append(
                BaseSearchResult(
                    document=metadata_entry.document,
                    score=float(dist),
                    rank=len(results),
                )
            )

            if len(results) >= k:
                break

        return results

    def search_by_text(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[BaseSearchResult]:
        """Search for documents similar to a text query.

        Args:
            query: Text query to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters (see search method for details)

        Returns:
            List of search results ordered by similarity
        """
        return self.search(query_text=query, k=k, filter_metadata=filter_metadata)

    def search_by_image(
        self,
        query: str | PILImage.Image,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        embedding_type: str = "image",
        caption_prompt: str = "Describe this image in detail.",
        system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
        vlm_backend: str = "nim",
        vlm_model: str | None = None,
        vlm_api_key: str | None = None,
    ) -> list[BaseSearchResult]:
        """Search for documents similar to an image query.

        Args:
            query: Image query to search for
            k: Number of results to return
            filter_metadata: Optional metadata filters (see search method for details)
            embedding_type: Type of embedding to use ("text" or "image").
                If "text", the image will be captioned first, then embedded as text.
                If "image", the image will be embedded directly.
            caption_prompt: Prompt to use for image captioning
            system_prompt: System instructions for the VLM
            vlm_backend: VLM backend to use ("azure_openai" or "nim")
            vlm_model: Model to use (uses backend default if None)
            vlm_api_key: API key for the VLM backend (uses env var if None)

        Returns:
            List of search results ordered by similarity
        """
        return self.search(
            query_image=query,
            k=k,
            filter_metadata=filter_metadata,
            embedding_type=embedding_type,
            caption_prompt=caption_prompt,
            system_prompt=system_prompt,
            vlm_backend=vlm_backend,
            vlm_model=vlm_model,
            vlm_api_key=vlm_api_key,
        )

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[BaseSearchResult]:
        """Search for documents similar to an embedding query.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters (see search method for details)

        Returns:
            List of search results ordered by similarity
        """
        return self.search(
            query_embedding=query_embedding, k=k, filter_metadata=filter_metadata
        )

    def remove_document(self, embedding_id: int) -> bool:
        """Remove a document from the vector store.

        Args:
            embedding_id: ID of the document to remove

        Returns:
            True if removed, False if not found

        Note:
            FAISS doesn't support direct removal, so this marks the entry
            as deleted in metadata. The index will need to be rebuilt
            periodically to actually remove the vectors.
        """
        if embedding_id in self.metadata_store:
            del self.metadata_store[embedding_id]
            return True
        return False

    def update_metadata(self, embedding_id: int, metadata: dict[str, Any]) -> bool:
        """Update metadata for an existing document.

        Args:
            embedding_id: ID of the document to update
            metadata: New metadata (replaces existing)

        Returns:
            True if updated, False if not found
        """
        if embedding_id in self.metadata_store:
            self.metadata_store[embedding_id].document.metadata = metadata
            return True
        return False

    def save(self, path: str | Path) -> None:
        """Save the vector store to disk.

        Args:
            path: Directory path to save the index and metadata

        The index will be saved as 'index.faiss' and metadata as 'metadata.json'
        in the specified directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata including embedding model information
        metadata_path = path / "metadata.json"
        metadata_dict = {
            "dimension": self.dimension,
            "nlist": self.nlist,
            "M": self.M,
            "index_type": self.index_type,
            "normalize_embeddings": self.normalize_embeddings,
            "next_id": self._next_id,
            "embedding_model": self._serialize_embedding_model(),
            "metadata_store": {
                str(k): {
                    "document": {
                        "text_content": v.document.text_content,
                        "text_path": v.document.text_path,
                        "image_path": v.document.image_path,
                        "document_id": v.document.document_id,
                        "metadata": v.document.metadata,
                    },
                    "embedding_id": v.embedding_id,
                }
                for k, v in self.metadata_store.items()
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

    @classmethod
    def _load(
        cls, path: str | Path, embedding_model_creator: Callable[[], BaseEmbeddingModel]
    ) -> "BaseVectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory path containing the saved index

        Returns:
            Loaded MultimodalVectorStore instance

        Raises:
            FileNotFoundError: If the saved files don't exist
            ValueError: If model compatibility validation fails
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, encoding="utf-8") as f:
            metadata_dict = json.load(f)

        # Create instance with embedding model automatically loaded from metadata
        store = cls(
            embedding_model=embedding_model_creator(
                backend=metadata_dict["embedding_model"]["service"],
                model=metadata_dict["embedding_model"]["model"],
            ),
            nlist=metadata_dict["nlist"],
            M=metadata_dict["M"],
            index_type=metadata_dict["index_type"],
            normalize_embeddings=metadata_dict["normalize_embeddings"],
        )

        # Load FAISS index
        index_path = path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        store.index = faiss.read_index(str(index_path))

        # Restore metadata
        store._next_id = metadata_dict["next_id"]
        store.metadata_store = {}
        for k, v in metadata_dict["metadata_store"].items():
            doc_data = v["document"]
            document = BaseDocument(
                text_content=doc_data["text_content"],
                text_path=doc_data["text_path"],
                image_path=doc_data["image_path"],
                image_data=None,  # Image data not preserved in save/load
                document_id=doc_data["document_id"],
                metadata=doc_data["metadata"],
            )
            store.metadata_store[int(k)] = BaseMetadata(
                document=document,
                embedding_id=v["embedding_id"],
            )

        return store

    def _serialize_embedding_model(self) -> dict[str, Any]:
        """Serialize embedding model configuration for storage.

        Returns:
            Dictionary containing model configuration information
        """
        model_info = {
            "class_name": self.embedding_model.__class__.__name__,
            "module": self.embedding_model.__class__.__module__,
        }

        # Determine service from class name
        class_name = self.embedding_model.__class__.__name__
        if "NIM" in class_name:
            model_info["service"] = "nim"
        elif "OpenAI" in class_name:
            model_info["service"] = "openai"
        else:
            # Fallback: try to extract from module path
            module_parts = self.embedding_model.__class__.__module__.split(".")
            if "nim" in module_parts:
                model_info["service"] = "nim"
            elif "openai" in module_parts:
                model_info["service"] = "openai"
            else:
                model_info["service"] = "nim"  # Default fallback

        # Add model-specific configuration
        if hasattr(self.embedding_model, "model"):
            model_info["model"] = self.embedding_model.model
        if hasattr(self.embedding_model, "base_url"):
            model_info["base_url"] = self.embedding_model.base_url
        if hasattr(self.embedding_model, "timeout"):
            model_info["timeout"] = self.embedding_model.timeout
        if hasattr(self.embedding_model, "embedding_dimension"):
            model_info["embedding_dimension"] = self.embedding_model.embedding_dimension

        # Add service-specific information
        if hasattr(self.embedding_model, "AVAILABLE_MODELS"):
            model_info["available_models"] = self.embedding_model.AVAILABLE_MODELS
        if hasattr(self.embedding_model, "DEFAULT_MODEL"):
            model_info["default_model"] = self.embedding_model.DEFAULT_MODEL

        return model_info

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        # Recreate index
        if self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self._needs_training = True
        elif self.index_type == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M)

        # Clear metadata
        self.metadata_store.clear()
        self._next_id = 0

    def _train_index(self, initial_vectors: list[np.ndarray] | None = None) -> None:
        """Train the index if needed (for IVF indices).

        Args:
            initial_vectors: Optional initial vectors to use for training
                           (useful when training before first add)
        """
        if hasattr(self.index, "train"):
            vectors = []

            # Use initial vectors if provided
            if initial_vectors:
                vectors.extend(initial_vectors)

            # Add any existing vectors from the index
            if self.index.ntotal > 0:
                # Sample vectors for training to avoid O(N²) complexity
                # For IVF indices, 10k samples is typically sufficient
                sample_size = min(10000, self.index.ntotal)
                sampled_vectors = self.index.reconstruct_n(0, sample_size)
                vectors.extend(sampled_vectors)

            if vectors:
                training_vectors = np.array(vectors, dtype=np.float32)
                if training_vectors.ndim == 3:
                    # Reshape from (n, 1, d) to (n, d) if needed
                    training_vectors = training_vectors.reshape(len(vectors), -1)
                self.index.train(training_vectors)
                self._needs_training = False

    def _matches_metadata_filter(self, metadata_value: Any, filter_value: Any) -> bool:
        """Check if a metadata value matches the filter criteria.

        For string values, uses case-insensitive contains matching.
        For non-string values, uses exact matching.

        Args:
            metadata_value: The value from document metadata
            filter_value: The value to filter by

        Returns:
            True if the metadata value matches the filter criteria
        """
        if metadata_value is None:
            return filter_value is None

        # For string values, use case-insensitive contains matching
        if isinstance(metadata_value, str) and isinstance(filter_value, str):
            return filter_value.lower() in metadata_value.lower()

        # For non-string values, use exact matching
        return metadata_value == filter_value

    @property
    def num_documents(self) -> int:
        """Get the number of documents in the store."""
        return len(self.metadata_store)

    def get_all_metadata(self) -> dict[int, BaseMetadata]:
        """Get all metadata entries.

        Returns:
            Dictionary mapping embedding IDs to metadata
        """
        return self.metadata_store.copy()

    def collect_documents(
        self, filter_metadata: dict[str, Any] | None = None
    ) -> list[BaseDocument]:
        """Collect all documents that match the given metadata filter.

        Args:
            filter_metadata: Optional metadata filters. For string values, uses case-insensitive
                contains matching. For non-string values, uses exact matching.
                If None, returns all documents.

        Returns:
            List of documents that match the filter criteria

        Example:
            >>> # Get all documents
            >>> all_docs = store.collect_documents()
            >>>
            >>> # Get documents with specific metadata
            >>> filtered_docs = store.collect_documents({"category": "research"})
            >>>
            >>> # Get documents with string contains matching
            >>> docs = store.collect_documents({"title": "machine learning"})
        """
        if filter_metadata is None:
            # Return all documents if no filter is provided
            return [metadata.document for metadata in self.metadata_store.values()]

        # Filter documents based on metadata criteria
        matching_documents = []
        for metadata_entry in self.metadata_store.values():
            # Check if document matches all filter criteria
            match = all(
                self._matches_metadata_filter(
                    metadata_entry.document.metadata.get(key), value
                )
                for key, value in filter_metadata.items()
            )
            if match:
                matching_documents.append(metadata_entry.document)

        return matching_documents

    @classmethod
    def build_vector_store(
        cls,
        embedding_model: BaseEmbeddingModel,
        text_source: str | Path | list[str | Path] | None = None,
        image_source: (
            str | Path | PILImage.Image | list[str | Path | PILImage.Image] | None
        ) = None,
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
    ) -> "BaseVectorStore":
        """Build a vector store from texts and/or images.

        This method provides a convenient way to create and populate a
        vector store from various sources:
        - Text strings
        - Image file paths or PIL Image objects
        - Mixed sources

        Args:
            embedding_model: Model to generate embeddings.
            text_source: Source of texts - can be:
                - str: Text string or file path
                - list: List of text strings or file paths
                - None: No text sources
            image_source: Source of images - can be:
                - str: Image file path
                - PILImage.Image: PIL Image object
                - list: List of image paths or PIL Images
                - None: No image sources
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
            Vector store populated with the provided texts and images

        Raises:
            ValueError: If sources are invalid or no content found
            FileNotFoundError: If directories don't exist
        """
        # Create the vector store
        vector_store = cls(
            embedding_model=embedding_model,
            index_type=index_type,
            normalize_embeddings=normalize_embeddings,
        )

        text_sources: list[str | Path] = []
        image_sources: list[str | Path | PILImage.Image] = []

        # Process text sources
        if text_source:
            # Collect all text sources
            if isinstance(text_source, str | Path):
                text_source = [text_source]

            if isinstance(text_source, list):
                # Process each item in the list
                for item in text_source:
                    # String - check if it's a file path or text content
                    source_path = Path(item)
                    if source_path.exists():
                        if source_path.is_dir():
                            if recursive:
                                for ext in text_extensions:
                                    text_sources.extend(source_path.rglob(f"*{ext}"))
                            else:
                                for ext in text_extensions:
                                    text_sources.extend(source_path.glob(f"*{ext}"))
                        else:
                            # It's a file path
                            text_sources.append(source_path)
                    else:
                        # It's text content, keep as string
                        text_sources.append(item)
            else:
                raise ValueError(
                    f"Invalid text source type: {type(text_source)}. "
                    "Expected string, list, or None"
                )

            # Add texts to the vector store
            for text_source in text_sources:
                # Extract metadata if extractor provided
                metadata = None
                if metadata_extractor is not None and isinstance(text_source, Path):
                    try:
                        metadata = metadata_extractor(text_source)
                    except Exception as e:
                        # Log warning but continue
                        logger.warning(
                            f"Failed to extract metadata for {text_source}: {e}"
                        )

                # Add text to store
                try:
                    vector_store.add_text(text_source, metadata=metadata)
                except Exception as e:
                    # Log warning but continue with other texts
                    logger.warning(f"Failed to add text {text_source}: {e}")

        # Process image sources
        if image_source:
            # Collect all image sources
            if isinstance(image_source, str | Path | PILImage.Image):
                image_source = [image_source]

            if isinstance(image_source, list):
                # Process each item in the list
                for item in image_source:
                    # Check if it's a PIL Image object
                    if isinstance(item, PILImage.Image):
                        image_sources.append(item)
                    else:
                        # String or Path - check if it's a file path or directory
                        source_path = Path(item)
                        if source_path.exists():
                            if source_path.is_dir():
                                if recursive:
                                    for ext in image_extensions:
                                        image_sources.extend(
                                            source_path.rglob(f"*{ext}")
                                        )
                                else:
                                    for ext in image_extensions:
                                        image_sources.extend(
                                            source_path.glob(f"*{ext}")
                                        )
                            else:
                                # It's a file path
                                image_sources.append(source_path)
                        else:
                            # It's a string that doesn't exist as a file - treat as invalid
                            raise FileNotFoundError(
                                f"Image source path does not exist: {item}"
                            )
            else:
                raise ValueError(
                    f"Invalid image source type: {type(image_source)}. "
                    "Expected string, Path, PIL Image, list, or None"
                )

            # Add images to the vector store
            for img_source in image_sources:
                # Extract metadata if extractor provided
                metadata = None
                if metadata_extractor is not None and isinstance(
                    img_source, str | Path
                ):
                    try:
                        metadata = metadata_extractor(img_source)
                    except Exception as e:
                        # Log warning but continue
                        logger.warning(
                            f"Failed to extract metadata for {img_source}: {e}"
                        )

                # Add image to store
                try:
                    vector_store.add_image(
                        img_source,
                        metadata=metadata,
                        embedding_type=image_embedding_type,
                        caption_prompt=caption_prompt,
                        system_prompt=system_prompt,
                        vlm_backend=vlm_backend,
                        vlm_model=vlm_model,
                        vlm_api_key=vlm_api_key,
                    )
                except Exception as e:
                    # Log warning but continue with other images
                    logger.warning(f"Failed to add image {img_source}: {e}")

        # Check if any content was successfully added
        if vector_store.num_documents == 0:
            raise ValueError("No content could be added to the vector store")

        logger.info(
            f"Successfully added {vector_store.num_documents} documents to vector store"
        )

        return vector_store

    def find_similar_documents(
        self,
        query: str | np.ndarray | Path | PILImage.Image,
        query_type: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        embedding_type: str = "image",
        caption_prompt: str = "Describe this image in detail.",
        system_prompt: str = "You are a helpful AI assistant that can analyze images. Provide detailed, accurate descriptions.",
        vlm_backend: str = "nim",
        vlm_model: str | None = None,
        vlm_api_key: str | None = None,
    ) -> list[BaseSearchResult]:
        """Find similar documents from a vector store.

        This method searches for documents similar to the query using vector
        similarity. It can work with either a saved vector store (by path) or
        an existing vector store instance.

        Args:
            query: Query to find similar documents for. Can be:
                - str: Text string to search for or Path to an image file
                - np.ndarray: Pre-computed embedding vector
                - Path: Path to an image file
                - PILImage.Image: PIL Image object
            query_type: Type of query ("text", "image", "embedding")
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
            List of BaseSearchResult objects ordered by similarity (most similar first)

        Raises:
            FileNotFoundError: If the vector store path doesn't exist
            ValueError: If NVIDIA_API_KEY not set when loading from path
        """
        # Validate query_type
        if query_type not in ["text", "image", "embedding"]:
            raise ValueError(
                f"Invalid query_type: {query_type}. Must be 'text', 'image', or 'embedding'"
            )

        # Validate embedding_type
        if embedding_type not in ["text", "image"]:
            raise ValueError(
                f"Invalid embedding_type: {embedding_type}. Must be 'text', or 'image'"
            )

        # Perform the search
        logger.info(f"Searching for {k} similar documents using {query_type} query")

        if query_type == "text":
            results = self.search_by_text(query, k, filter_metadata)
        elif query_type == "image":
            results = self.search_by_image(
                query,
                k,
                filter_metadata,
                embedding_type,
                caption_prompt,
                system_prompt,
                vlm_backend,
                vlm_model,
                vlm_api_key,
            )
        else:  # query_type == "embedding"
            results = self.search_by_embedding(query, k, filter_metadata)

        logger.info(f"Found {len(results)} similar documents")

        return results
