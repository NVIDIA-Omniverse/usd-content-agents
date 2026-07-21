# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal vector store functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from world_understanding.functions.knowledge.base_vector_store import BaseSearchResult
from world_understanding.functions.knowledge.multimodal_vector_store import (
    MultimodalVectorStore,
    build_multimodal_vector_store,
    collect_documents_from_vector_store,
    find_similar_documents_from_vector_store,
)
from world_understanding.functions.models.multimodal_embedding_models import (
    BaseMultimodalEmbeddingModel,
)


class MockMultimodalEmbeddingModel(BaseMultimodalEmbeddingModel):
    """Mock multimodal embedding model for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self.model = "mock_model"  # Required for serialization
        self.base_url = "http://mock.url"  # Optional but good to have

    def embed_text(
        self, text: str, input_type: str | None = None, **kwargs
    ) -> np.ndarray:
        """Generate mock text embedding."""
        # Generate deterministic embedding based on text length and content
        np.random.seed(hash(text + str(input_type)) % 2**32)
        return np.random.randn(self._dimension).astype(np.float32)

    def embed_image(
        self,
        image: str | Path | PILImage.Image | np.ndarray,
        input_type: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate mock image embedding."""
        # Generate deterministic embedding based on input type
        if isinstance(image, np.ndarray):
            seed = int(np.mean(image) * 1000) % 1000
        elif isinstance(image, PILImage.Image):
            seed = image.width + image.height
        elif isinstance(image, str | Path):
            seed = len(str(image))
        else:
            seed = 42

        np.random.seed(seed + hash(str(input_type)) % 2**32)
        return np.random.rand(self._dimension).astype(np.float32)

    def embed_texts(
        self, texts: list[str], input_type: str | None = None, **kwargs
    ) -> list[np.ndarray]:
        """Generate mock embeddings for multiple texts."""
        return [self.embed_text(text, input_type, **kwargs) for text in texts]

    def embed_images(
        self,
        images: list[str | Path | PILImage.Image | np.ndarray],
        input_type: str | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Generate mock embeddings for multiple images."""
        return [self.embed_image(img, input_type, **kwargs) for img in images]

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List available models."""
        return ["mock_model"]

    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class TestMultimodalVectorStore:
    """Test MultimodalVectorStore class."""

    @pytest.fixture
    def mock_embedding_model(self) -> MockMultimodalEmbeddingModel:
        """Create a mock embedding model."""
        return MockMultimodalEmbeddingModel(dimension=384)

    @pytest.fixture
    def vector_store(self, mock_embedding_model) -> MultimodalVectorStore:
        """Create a vector store instance."""
        return MultimodalVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatL2",
            normalize_embeddings=False,
        )

    @pytest.fixture
    def sample_images(self, tmp_path: Path) -> list[Path]:
        """Create sample test images."""
        images = []
        for i in range(3):
            # Create different colored images
            color = (i * 80, 100, 255 - i * 80)
            img = PILImage.new("RGB", (100, 100), color)
            path = tmp_path / f"test_image_{i}.png"
            img.save(path)
            images.append(path)
        return images

    def test_initialization(self, mock_embedding_model):
        """Test MultimodalVectorStore initialization."""
        store = MultimodalVectorStore(mock_embedding_model)

        assert store.embedding_model == mock_embedding_model
        assert store.dimension == 384
        assert store.index_type == "IndexFlatL2"
        assert store.normalize_embeddings is False
        assert store.num_documents == 0

    def test_add_single_text(self, vector_store):
        """Test adding single text to vector store."""
        text_id = vector_store.add_text("Test text", "test_1", {"category": "test"})

        assert text_id == 0
        assert vector_store.num_documents == 1

        metadata = vector_store.metadata_store[0]
        assert metadata.document.has_text()
        assert metadata.document.text_content == "Test text"
        assert metadata.document.document_id == "test_1"
        assert metadata.document.metadata == {"category": "test"}
        assert metadata.embedding_id == 0

    def test_add_single_image(self, vector_store, sample_images):
        """Test adding single image to vector store."""
        image_path = sample_images[0]
        image_id = vector_store.add_image(
            image_path, "image_1", {"category": "test"}, "image"
        )

        assert image_id == 0
        assert vector_store.num_documents == 1

        metadata = vector_store.metadata_store[0]
        assert metadata.document.has_image()
        assert metadata.document.image_path == str(image_path)
        assert metadata.document.document_id == "image_1"
        assert metadata.document.metadata == {"category": "test"}
        assert metadata.embedding_id == 0

    def test_add_multiple_texts(self, vector_store):
        """Test adding multiple texts."""
        texts = ["Text 1", "Text 2", "Text 3"]
        text_ids = ["id_1", "id_2", "id_3"]
        metadata_list = [{"cat": "a"}, {"cat": "b"}, {"cat": "c"}]

        ids = vector_store.add_texts(texts, text_ids, metadata_list)

        assert ids == [0, 1, 2]
        assert vector_store.num_documents == 3

        for i, (text, text_id, metadata) in enumerate(
            zip(texts, text_ids, metadata_list, strict=False)
        ):
            stored_metadata = vector_store.metadata_store[i]
            assert stored_metadata.document.has_text()
            assert stored_metadata.document.text_content == text
            assert stored_metadata.document.document_id == text_id
            assert stored_metadata.document.metadata == metadata

    def test_add_multiple_images(self, vector_store, sample_images):
        """Test adding multiple images."""
        image_ids = ["img_1", "img_2", "img_3"]
        metadata_list = [{"cat": "a"}, {"cat": "b"}, {"cat": "c"}]
        embedding_types = ["image", "image", "image"]  # List of embedding types

        ids = vector_store.add_images(
            sample_images, image_ids, metadata_list, embedding_types
        )

        assert ids == [0, 1, 2]
        assert vector_store.num_documents == 3

        for i, (image_path, image_id, metadata) in enumerate(
            zip(sample_images, image_ids, metadata_list, strict=False)
        ):
            stored_metadata = vector_store.metadata_store[i]
            assert stored_metadata.document.has_image()
            assert stored_metadata.document.image_path == str(image_path)
            assert stored_metadata.document.document_id == image_id
            assert stored_metadata.document.metadata == metadata

    def test_search_similar_texts(self, vector_store):
        """Test text search functionality."""
        # Add some texts
        vector_store.add_text("Machine learning is AI", "ml", {"category": "tech"})
        vector_store.add_text(
            "Natural language processing", "nlp", {"category": "tech"}
        )
        vector_store.add_text(
            "The weather is sunny", "weather", {"category": "general"}
        )

        # Search
        results = vector_store.search_by_text("artificial intelligence", k=2)

        assert len(results) == 2
        assert all(isinstance(result, BaseSearchResult) for result in results)
        assert all(result.document.has_text() for result in results)

    def test_search_similar_images(self, vector_store, sample_images):
        """Test image search functionality."""
        # Add some images
        vector_store.add_image(sample_images[0], "img1", {"category": "test"}, "image")
        vector_store.add_image(sample_images[1], "img2", {"category": "test"}, "image")

        # Search
        results = vector_store.search_by_image(sample_images[0], k=2)

        assert len(results) == 2
        assert all(isinstance(result, BaseSearchResult) for result in results)
        assert all(result.document.has_image() for result in results)

    def test_search_with_mixed_content(self, vector_store, sample_images):
        """Test general search functionality with mixed content."""
        # Add mixed content
        vector_store.add_text("Machine learning", "ml", {"category": "tech"})
        vector_store.add_image(sample_images[0], "img1", {"category": "test"}, "image")

        # Text search
        text_results = vector_store.search(query_text="machine learning", k=2)
        assert len(text_results) >= 1
        assert any(result.document.has_text() for result in text_results)

        # Image search
        image_results = vector_store.search(query_image=sample_images[0], k=2)
        assert len(image_results) >= 1
        assert any(result.document.has_image() for result in image_results)

    def test_update_metadata(self, vector_store):
        """Test updating document metadata."""
        text_id = vector_store.add_text("Test text", "test_1", {"category": "test"})

        # Update metadata
        success = vector_store.update_metadata(
            text_id, {"category": "updated", "new_field": "value"}
        )
        assert success is True

        # Verify update
        metadata = vector_store.metadata_store[text_id]
        assert metadata.document.metadata["category"] == "updated"
        assert metadata.document.metadata["new_field"] == "value"

    def test_remove_document(self, vector_store):
        """Test removing a document."""
        text_id = vector_store.add_text("Test text", "test_1", {"category": "test"})

        assert vector_store.num_documents == 1

        # Remove document
        success = vector_store.remove_document(text_id)
        assert success is True
        assert vector_store.num_documents == 0

    def test_get_all_metadata(self, vector_store):
        """Test getting all metadata."""
        vector_store.add_text("Text 1", "id1", {"category": "test"})
        vector_store.add_text("Text 2", "id2", {"category": "test"})

        metadata = vector_store.get_all_metadata()
        assert len(metadata) == 2
        assert 0 in metadata
        assert 1 in metadata

    def test_clear(self, vector_store):
        """Test clearing the vector store."""
        vector_store.add_text("Test text", "test_1", {"category": "test"})

        assert vector_store.num_documents == 1

        vector_store.clear()

        assert vector_store.num_documents == 0
        assert len(vector_store.metadata_store) == 0

    def test_save_and_load(self, vector_store, tmp_path):
        """Test saving and loading vector store."""
        # Add some content
        vector_store.add_text("Test text", "test_1", {"category": "test"})

        # Save store
        save_path = tmp_path / "test_store"
        vector_store.save(save_path)

        # Load store - this will create a default embedding model
        # We need to mock the create_multimodal_embedding_model function
        with patch(
            "world_understanding.functions.knowledge.multimodal_vector_store.create_multimodal_embedding_model"
        ) as mock_create:
            # Create a mock model that matches our test model
            mock_model = MockMultimodalEmbeddingModel(dimension=384)
            mock_create.return_value = mock_model

            loaded_store = MultimodalVectorStore.load(save_path)

            assert loaded_store.num_documents == 1
            assert loaded_store.dimension == vector_store.dimension

    def test_different_index_types(self, mock_embedding_model):
        """Test different index types."""
        # Test IndexFlatL2
        store_l2 = MultimodalVectorStore(
            embedding_model=mock_embedding_model, index_type="IndexFlatL2"
        )
        store_l2.add_text("Test", "test", {})
        assert store_l2.index_type == "IndexFlatL2"

        # Test IndexFlatIP
        store_ip = MultimodalVectorStore(
            embedding_model=mock_embedding_model, index_type="IndexFlatIP"
        )
        store_ip.add_text("Test", "test", {})
        assert store_ip.index_type == "IndexFlatIP"

    def test_error_handling(self, vector_store):
        """Test error handling for invalid operations."""
        # Test searching empty store
        with pytest.raises(ValueError, match="Index is empty"):
            vector_store.search_by_text("test", k=1)

    def test_mixed_content_search(self, vector_store, sample_images):
        """Test searching across mixed text and image content."""
        # Add mixed content
        vector_store.add_text("Red color", "red_text", {"color": "red"})
        vector_store.add_text("Blue color", "blue_text", {"color": "blue"})
        vector_store.add_image(sample_images[0], "red_img", {"color": "red"}, "image")
        vector_store.add_image(sample_images[1], "blue_img", {"color": "blue"}, "image")

        # Search for red content
        red_results = vector_store.search_by_text("red", k=4)
        assert len(red_results) >= 2

        # Should find both text and image results
        has_text = any(result.document.has_text() for result in red_results)
        has_image = any(result.document.has_image() for result in red_results)
        assert has_text and has_image


class TestBuildMultimodalVectorStore:
    """Test build_multimodal_vector_store function."""

    @pytest.fixture
    def mock_embedding_model(self) -> MockMultimodalEmbeddingModel:
        """Create a mock embedding model."""
        return MockMultimodalEmbeddingModel(dimension=384)

    def test_build_from_text_sources(self, mock_embedding_model):
        """Test building vector store from text sources."""
        # Create test text content as strings (not file paths)
        texts = ["Test text content 1", "Test text content 2", "Test text content 3"]

        # Build vector store
        vector_store = build_multimodal_vector_store(
            text_source=texts,
            embedding_model=mock_embedding_model,
        )

        assert vector_store.num_documents == 3

    def test_build_from_image_sources(self, mock_embedding_model, tmp_path):
        """Test building vector store from image sources."""
        # Create test images
        image_files = []
        for i in range(3):
            img = PILImage.new("RGB", (100, 100), (i * 80, 100, 255 - i * 80))
            image_file = tmp_path / f"image_{i}.png"
            img.save(image_file)
            image_files.append(image_file)

        # Build vector store
        vector_store = build_multimodal_vector_store(
            image_source=image_files,
            embedding_model=mock_embedding_model,
        )

        assert vector_store.num_documents == 3

    def test_build_from_mixed_sources(self, mock_embedding_model, tmp_path):
        """Test building vector store from mixed text and image sources."""
        # Create test text content as strings
        texts = ["Test text content"]

        # Create test image
        img = PILImage.new("RGB", (100, 100), (255, 0, 0))
        image_file = tmp_path / "image.png"
        img.save(image_file)

        # Build vector store
        vector_store = build_multimodal_vector_store(
            text_source=texts,
            image_source=[image_file],
            embedding_model=mock_embedding_model,
        )

        assert vector_store.num_documents == 2

    def test_build_creates_embedding_model_from_api_key(self, monkeypatch):
        """Test default embedding model creation when an API key is configured."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
        mock_model = MockMultimodalEmbeddingModel(dimension=256)
        expected_store = object()

        with (
            patch(
                "world_understanding.functions.knowledge.multimodal_vector_store.create_multimodal_embedding_model",
                return_value=mock_model,
            ) as mock_create_model,
            patch.object(
                MultimodalVectorStore,
                "build_vector_store",
                return_value=expected_store,
            ) as mock_build_vector_store,
        ):
            vector_store = build_multimodal_vector_store(text_source=["hello"])

        assert vector_store is expected_store
        mock_create_model.assert_called_once_with(backend="nim", api_key="test-api-key")
        assert mock_build_vector_store.call_args.kwargs["embedding_model"] is mock_model


class TestFindSimilarDocumentsFromVectorStore:
    """Test find_similar_documents_from_vector_store helper dispatch."""

    def test_find_similar_documents_loads_path_store(self):
        """Test finding documents from a saved store path."""
        mock_store = MagicMock()
        expected_results = [object()]
        mock_store.find_similar_documents.return_value = expected_results
        store_path = Path("/tmp/multimodal-store")

        with patch.object(
            MultimodalVectorStore, "load", return_value=mock_store
        ) as mock_load:
            results = find_similar_documents_from_vector_store(
                query="chair",
                query_type="text",
                store=store_path,
                k=3,
                filter_metadata={"category": "furniture"},
                embedding_type="text",
                caption_prompt="caption",
                system_prompt="system",
                vlm_backend="azure_openai",
                vlm_model="gpt-4o-mini",
                vlm_api_key="key",
            )

        assert results == expected_results
        mock_load.assert_called_once_with(store_path)
        mock_store.find_similar_documents.assert_called_once_with(
            query="chair",
            query_type="text",
            k=3,
            filter_metadata={"category": "furniture"},
            embedding_type="text",
            caption_prompt="caption",
            system_prompt="system",
            vlm_backend="azure_openai",
            vlm_model="gpt-4o-mini",
            vlm_api_key="key",
        )

    def test_find_similar_documents_uses_store_instance(self):
        """Test finding documents from an existing store instance."""
        mock_store = MagicMock()
        expected_results = [object()]
        mock_store.find_similar_documents.return_value = expected_results

        results = find_similar_documents_from_vector_store(
            query=np.ones(3),
            query_type="embedding",
            store=mock_store,
        )

        assert results == expected_results
        mock_store.find_similar_documents.assert_called_once()


class TestCollectDocuments:
    """Test collect_documents functionality."""

    @pytest.fixture
    def mock_embedding_model(self) -> MockMultimodalEmbeddingModel:
        """Create a mock embedding model."""
        return MockMultimodalEmbeddingModel(dimension=384)

    @pytest.fixture
    def populated_vector_store(self, mock_embedding_model) -> MultimodalVectorStore:
        """Create a vector store with test documents."""
        store = MultimodalVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatL2",
            normalize_embeddings=False,
        )

        # Add test documents with different metadata
        store.add_text(
            "Machine learning text", "ml_doc", {"category": "tech", "year": 2023}
        )
        store.add_text(
            "Weather report", "weather_doc", {"category": "news", "year": 2024}
        )
        store.add_text(
            "Cooking recipe", "cooking_doc", {"category": "lifestyle", "year": 2023}
        )

        return store

    def test_collect_all_documents(self, populated_vector_store):
        """Test collecting all documents without filter."""
        documents = populated_vector_store.collect_documents()

        assert len(documents) == 3
        assert all(
            doc.document_id in ["ml_doc", "weather_doc", "cooking_doc"]
            for doc in documents
        )

    def test_collect_documents_with_filter(self, populated_vector_store):
        """Test collecting documents with metadata filter."""
        # Filter by category
        tech_docs = populated_vector_store.collect_documents({"category": "tech"})
        assert len(tech_docs) == 1
        assert tech_docs[0].document_id == "ml_doc"

        # Filter by year
        year_2023_docs = populated_vector_store.collect_documents({"year": 2023})
        assert len(year_2023_docs) == 2
        assert all(
            doc.document_id in ["ml_doc", "cooking_doc"] for doc in year_2023_docs
        )

        # Filter by multiple criteria
        tech_2023_docs = populated_vector_store.collect_documents(
            {"category": "tech", "year": 2023}
        )
        assert len(tech_2023_docs) == 1
        assert tech_2023_docs[0].document_id == "ml_doc"

    def test_collect_documents_string_matching(self, populated_vector_store):
        """Test string matching in metadata filters."""
        # Case-insensitive string matching
        tech_docs = populated_vector_store.collect_documents({"category": "TECH"})
        assert len(tech_docs) == 1
        assert tech_docs[0].document_id == "ml_doc"

        # Partial string matching
        tech_docs_partial = populated_vector_store.collect_documents(
            {"category": "ech"}
        )
        assert len(tech_docs_partial) == 1
        assert tech_docs_partial[0].document_id == "ml_doc"

    def test_collect_documents_no_matches(self, populated_vector_store):
        """Test collecting documents with filter that matches nothing."""
        no_docs = populated_vector_store.collect_documents({"category": "nonexistent"})
        assert len(no_docs) == 0

    def test_collect_documents_empty_store(self, mock_embedding_model):
        """Test collecting documents from empty vector store."""
        empty_store = MultimodalVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatL2",
        )

        all_docs = empty_store.collect_documents()
        assert len(all_docs) == 0

        filtered_docs = empty_store.collect_documents({"category": "tech"})
        assert len(filtered_docs) == 0


class TestCollectDocumentsFromVectorStore:
    """Test collect_documents_from_vector_store function."""

    def test_collect_documents_uses_store_instance(self):
        """Test collecting documents from an existing vector store instance."""
        mock_store = MagicMock()
        expected_documents = [object()]
        mock_store.collect_documents.return_value = expected_documents

        documents = collect_documents_from_vector_store(
            mock_store, filter_metadata={"category": "test"}
        )

        assert documents == expected_documents
        mock_store.collect_documents.assert_called_once_with(
            filter_metadata={"category": "test"}
        )

    def test_collect_documents_from_vector_store_function(self, tmp_path):
        """Test the standalone collect_documents_from_vector_store function."""
        # Create a mock embedding model
        mock_model = MockMultimodalEmbeddingModel()

        # Create a vector store with test data
        store = MultimodalVectorStore(
            embedding_model=mock_model,
            index_type="IndexFlatL2",
        )

        # Add test documents
        store.add_text(
            "Test document 1", "doc1", {"category": "test", "priority": "high"}
        )
        store.add_text(
            "Test document 2", "doc2", {"category": "test", "priority": "low"}
        )
        store.add_text(
            "Other document", "doc3", {"category": "other", "priority": "high"}
        )

        # Save the store
        store_path = tmp_path / "test_store"
        store.save(store_path)

        # Mock the create_multimodal_embedding_model function to return our mock model
        with patch(
            "world_understanding.functions.knowledge.multimodal_vector_store.create_multimodal_embedding_model"
        ) as mock_create:
            mock_create.return_value = mock_model

            # Test collecting all documents
            all_docs = collect_documents_from_vector_store(str(store_path))
            assert len(all_docs) == 3

            # Test collecting with filter
            test_docs = collect_documents_from_vector_store(
                str(store_path), {"category": "test"}
            )
            assert len(test_docs) == 2
            assert all(doc.document_id in ["doc1", "doc2"] for doc in test_docs)

            # Test collecting with multiple filters
            high_priority_test_docs = collect_documents_from_vector_store(
                str(store_path), {"category": "test", "priority": "high"}
            )
            assert len(high_priority_test_docs) == 1
            assert high_priority_test_docs[0].document_id == "doc1"
