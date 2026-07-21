# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for text vector store functions."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from world_understanding.functions.knowledge.base_vector_store import BaseSearchResult
from world_understanding.functions.knowledge.text_vector_store import (
    TextVectorStore,
    build_text_vector_store,
    find_similar_texts_from_vector_store,
)
from world_understanding.functions.models.base_embedding_model import BaseEmbeddingModel


class MockTextEmbeddingModel(BaseEmbeddingModel):
    """Mock text embedding model for testing."""

    def __init__(
        self,
        api_key: str = "dummy",
        model: str = "nvidia/llama-nemotron-embed-vl-1b-v2",
        embedding_dimension: int = 384,
    ):
        super().__init__(
            api_key=api_key, model=model, embedding_dimension=embedding_dimension
        )

    def embed_text(self, text: str, **kwargs) -> np.ndarray:
        """Generate mock embedding."""
        # Generate deterministic embedding based on text length
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(self.embedding_dimension).astype(np.float32)

    def embed_texts(self, texts: list[str], **kwargs) -> list[np.ndarray]:
        """Generate mock embeddings for multiple texts."""
        return [self.embed_text(text, **kwargs) for text in texts]

    def embed_image(self, image: str | Path | np.ndarray, **kwargs) -> np.ndarray:
        """Generate a mock image embedding (not used for text model)."""
        np.random.seed(42)
        return np.random.randn(self.embedding_dimension).astype(np.float32)

    def embed_images(
        self, images: list[str | Path | np.ndarray], **kwargs
    ) -> list[np.ndarray]:
        """Generate mock image embeddings (not used for text model)."""
        return [self.embed_image(img, **kwargs) for img in images]

    def list_available_models(self) -> list[str]:
        """Return available models."""
        return ["nvidia/llama-nemotron-embed-vl-1b-v2"]


class TestTextVectorStore:
    """Test TextVectorStore class."""

    @pytest.fixture
    def mock_embedding_model(self) -> MockTextEmbeddingModel:
        """Create a mock embedding model."""
        return MockTextEmbeddingModel(api_key="dummy", embedding_dimension=384)

    @pytest.fixture
    def vector_store(self, mock_embedding_model) -> TextVectorStore:
        """Create a vector store instance."""
        return TextVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatL2",
        )

    def test_initialization(self, mock_embedding_model):
        """Test TextVectorStore initialization."""
        store = TextVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatL2",
        )

        assert store.embedding_model == mock_embedding_model
        assert store.dimension == 384
        assert store.index_type == "IndexFlatL2"
        assert store.normalize_embeddings is False
        assert store.num_documents == 0

    def test_add_single_text(self, vector_store):
        """Test adding single text to vector store."""
        text_id = vector_store.add_text(
            text="Test text", text_id="test_1", metadata={"category": "test"}
        )

        assert text_id == 0
        assert vector_store.num_documents == 1

        metadata = vector_store.metadata_store[0]
        assert metadata.document.text_content == "Test text"
        assert metadata.document.document_id == "test_1"
        assert metadata.document.metadata == {"category": "test"}
        assert metadata.embedding_id == 0

    def test_add_multiple_texts(self, vector_store):
        """Test adding multiple texts."""
        texts = ["Text 1", "Text 2", "Text 3"]
        text_ids = ["id_1", "id_2", "id_3"]
        metadata_list = [{"cat": "a"}, {"cat": "b"}, {"cat": "c"}]

        ids = vector_store.add_texts(
            texts=texts, text_ids=text_ids, metadata_list=metadata_list
        )

        assert ids == [0, 1, 2]
        assert vector_store.num_documents == 3

        for i, (text, text_id, metadata) in enumerate(
            zip(texts, text_ids, metadata_list, strict=False)
        ):
            stored_metadata = vector_store.metadata_store[i]
            assert stored_metadata.document.text_content == text
            assert stored_metadata.document.document_id == text_id
            assert stored_metadata.document.metadata == metadata

    def test_search_similar_texts(self, vector_store):
        """Test text search functionality."""
        # Add some texts
        vector_store.add_text(
            text="Machine learning is AI", text_id="ml", metadata={"category": "tech"}
        )
        vector_store.add_text(
            text="Natural language processing",
            text_id="nlp",
            metadata={"category": "tech"},
        )
        vector_store.add_text(
            text="The weather is sunny",
            text_id="weather",
            metadata={"category": "general"},
        )

        # Search
        results = vector_store.search(query_text="artificial intelligence", k=2)

        assert len(results) == 2
        assert all(isinstance(r, BaseSearchResult) for r in results)
        assert results[0].rank == 0
        assert results[1].rank == 1

    def test_search_with_metadata_filter(self, vector_store):
        """Test search with metadata filter."""
        # Add texts with different categories
        vector_store.add_text(
            text="Machine learning", text_id="ml", metadata={"category": "tech"}
        )
        vector_store.add_text(
            text="Natural language", text_id="nlp", metadata={"category": "tech"}
        )
        vector_store.add_text(
            text="Weather report", text_id="weather", metadata={"category": "general"}
        )

        # Search with filter
        results = vector_store.search(
            query_text="learning", k=3, filter_metadata={"category": "tech"}
        )

        assert len(results) == 2  # Only tech category
        assert all(r.document.metadata["category"] == "tech" for r in results)

    def test_search_by_text(self, vector_store):
        """Test search_by_text method."""
        # Add some texts
        vector_store.add_text(
            text="Machine learning is AI", text_id="ml", metadata={"category": "tech"}
        )
        vector_store.add_text(
            text="Natural language processing",
            text_id="nlp",
            metadata={"category": "tech"},
        )
        vector_store.add_text(
            text="The weather is sunny",
            text_id="weather",
            metadata={"category": "general"},
        )

        # Search using search_by_text
        results = vector_store.search_by_text("artificial intelligence", k=2)

        assert len(results) == 2
        assert all(isinstance(r, BaseSearchResult) for r in results)
        assert results[0].rank == 0
        assert results[1].rank == 1

    def test_search_with_embedding(self, vector_store):
        """Test search_by_embedding method."""
        # Add some texts
        vector_store.add_text(
            text="Machine learning is AI", text_id="ml", metadata={"category": "tech"}
        )
        vector_store.add_text(
            text="Natural language processing",
            text_id="nlp",
            metadata={"category": "tech"},
        )

        # Create a mock embedding with correct dimension
        mock_embedding = np.random.randn(384).astype(np.float32)

        # Search using search_by_embedding
        results = vector_store.search_by_embedding(mock_embedding, k=2)

        assert len(results) == 2
        assert all(isinstance(r, BaseSearchResult) for r in results)
        assert results[0].rank == 0
        assert results[1].rank == 1

    def test_clear(self, vector_store):
        """Test clearing vector store."""
        # Add texts
        vector_store.add_text(text="Text 1", text_id="id1")
        vector_store.add_text(text="Text 2", text_id="id2")

        assert vector_store.num_documents == 2

        # Clear
        vector_store.clear()

        assert vector_store.num_documents == 0
        assert len(vector_store.metadata_store) == 0

    def test_save_and_load(self, vector_store, tmp_path):
        """Test saving and loading vector store."""
        # Add some texts
        vector_store.add_text("Machine learning", "ml", metadata={"category": "tech"})
        vector_store.add_text("Natural language", "nlp", metadata={"category": "tech"})
        vector_store.add_text(
            "Weather report", "weather", metadata={"category": "general"}
        )

        # Save
        save_path = tmp_path / "test_store"
        vector_store.save(save_path)

        # Load into new instance - mock the create_text_embedding_model function
        with patch(
            "world_understanding.functions.knowledge.text_vector_store.create_text_embedding_model"
        ) as mock_create:
            # Create a mock model that matches our test model
            mock_model = MockTextEmbeddingModel(
                api_key="dummy", embedding_dimension=384
            )
            mock_create.return_value = mock_model

            loaded_store = TextVectorStore.load(save_path)

            # Verify loaded data
            assert loaded_store.index_type == vector_store.index_type
            assert loaded_store.num_documents == 3

    def test_load_without_embedding_model_info(self, vector_store, tmp_path):
        """Test loading store without embedding model info."""
        # Add some texts
        vector_store.add_text("Test text", "test")

        # Save
        save_path = tmp_path / "test_store"
        vector_store.save(save_path)

        # Load - mock the create_text_embedding_model function
        with patch(
            "world_understanding.functions.knowledge.text_vector_store.create_text_embedding_model"
        ) as mock_create:
            # Create a mock model that matches our test model
            mock_model = MockTextEmbeddingModel(
                api_key="dummy", embedding_dimension=384
            )
            mock_create.return_value = mock_model

            loaded_store = TextVectorStore.load(save_path)
            assert loaded_store.num_documents == 1

    def test_different_index_types(self, mock_embedding_model):
        """Test different index types."""
        # Test IndexFlatL2
        store_l2 = TextVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatL2",
        )
        assert store_l2.index_type == "IndexFlatL2"

        # Test IndexFlatIP
        store_ip = TextVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatIP",
            normalize_embeddings=True,
        )
        assert store_ip.index_type == "IndexFlatIP"
        assert store_ip.normalize_embeddings is True

    def test_error_handling(self, vector_store):
        """Test error handling."""
        # Test invalid search parameters
        with pytest.raises(ValueError):
            vector_store.search(query_text="test", k=0)

        # Test search with no documents - should raise ValueError
        with pytest.raises(ValueError, match="Index is empty"):
            vector_store.search(query_text="test", k=1)


class TestBuildTextVectorStore:
    """Test build_text_vector_store function."""

    @pytest.fixture
    def mock_embedding_model(self) -> MockTextEmbeddingModel:
        """Create a mock embedding model."""
        return MockTextEmbeddingModel(api_key="dummy", embedding_dimension=384)

    def test_build_creates_embedding_model_from_api_key(self, monkeypatch):
        """Test default embedding model creation when an API key is configured."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
        mock_model = MockTextEmbeddingModel(
            api_key="test-api-key", embedding_dimension=256
        )
        expected_store = object()
        captured_kwargs = {}

        def create_model(**kwargs):
            captured_kwargs["factory"] = kwargs
            return mock_model

        def build_store(**kwargs):
            captured_kwargs["build"] = kwargs
            return expected_store

        monkeypatch.setattr(
            "world_understanding.functions.knowledge.text_vector_store.create_text_embedding_model",
            create_model,
        )
        monkeypatch.setattr(TextVectorStore, "build_vector_store", build_store)

        vector_store = build_text_vector_store(text_source=["hello"])

        assert vector_store is expected_store
        assert captured_kwargs["factory"] == {
            "backend": "nim",
            "api_key": "test-api-key",
        }
        assert captured_kwargs["build"]["embedding_model"] is mock_model

    def test_build_from_list(self, mock_embedding_model):
        """Test building vector store from list of texts."""
        texts = ["Text 1", "Text 2", "Text 3"]

        store = build_text_vector_store(
            text_source=texts, embedding_model=mock_embedding_model
        )

        assert store.num_documents == 3
        assert store.dimension == 384

    def test_build_from_dict(self, mock_embedding_model):
        """Test building vector store from dictionary."""
        # Convert dict to list of texts for the function
        text_dict = {"doc1": "Content 1", "doc2": "Content 2"}
        texts = list(text_dict.values())

        store = build_text_vector_store(
            text_source=texts, embedding_model=mock_embedding_model
        )

        assert store.num_documents == 2
        # Check that the texts were added
        text_contents = [m.document.text_content for m in store.metadata_store.values()]
        assert "Content 1" in text_contents
        assert "Content 2" in text_contents

    def test_build_with_different_index_types(self, mock_embedding_model):
        """Test building with different index types."""
        texts = ["Text 1", "Text 2"]

        # Test IndexFlatL2
        store_l2 = build_text_vector_store(
            text_source=texts,
            embedding_model=mock_embedding_model,
            index_type="IndexFlatL2",
        )
        assert store_l2.index_type == "IndexFlatL2"

        # Test IndexFlatIP
        store_ip = build_text_vector_store(
            text_source=texts,
            embedding_model=mock_embedding_model,
            index_type="IndexFlatIP",
            normalize_embeddings=True,
        )
        assert store_ip.index_type == "IndexFlatIP"
        assert store_ip.normalize_embeddings is True

    def test_build_empty_list(self, mock_embedding_model):
        """Test building from empty list raises error."""
        with pytest.raises(
            ValueError, match="No content could be added to the vector store"
        ):
            build_text_vector_store(
                text_source=[], embedding_model=mock_embedding_model
            )


class TestFindSimilarTextsFromVectorStore:
    """Test find_similar_texts_from_vector_store function."""

    @pytest.fixture
    def mock_embedding_model(self) -> MockTextEmbeddingModel:
        """Create a mock embedding model."""
        return MockTextEmbeddingModel(api_key="dummy", embedding_dimension=384)

    def test_find_similar_from_store_instance(self, mock_embedding_model):
        """Test finding similar texts from store instance."""
        store = TextVectorStore(embedding_model=mock_embedding_model)

        # Add texts
        store.add_text(
            "Machine learning algorithms", "ml", metadata={"category": "tech"}
        )
        store.add_text(
            "Natural language processing", "nlp", metadata={"category": "tech"}
        )
        store.add_text("Computer vision systems", "cv", metadata={"category": "tech"})

        # Find similar texts
        results = find_similar_texts_from_vector_store(
            query="artificial intelligence", query_type="text", store=store, k=2
        )

        assert len(results) == 2
        assert all(isinstance(r, BaseSearchResult) for r in results)
        assert results[0].rank == 0
        assert results[1].rank == 1

    def test_find_similar_with_different_query_types(
        self, mock_embedding_model, tmp_path
    ):
        """Test finding similar texts with different query types."""
        store = TextVectorStore(embedding_model=mock_embedding_model)

        # Add texts
        store.add_text(
            "Machine learning algorithms", "ml", metadata={"category": "tech"}
        )
        store.add_text(
            "Natural language processing", "nlp", metadata={"category": "tech"}
        )
        store.add_text("Computer vision systems", "cv", metadata={"category": "tech"})

        # Test with text query using store instance directly
        results_text = find_similar_texts_from_vector_store(
            query="artificial intelligence", query_type="text", store=store, k=2
        )
        assert len(results_text) == 2

        # Test with embedding query using store instance directly
        mock_embedding = np.random.randn(384).astype(np.float32)
        results_embedding = find_similar_texts_from_vector_store(
            query=mock_embedding, query_type="embedding", store=store, k=2
        )
        assert len(results_embedding) == 2

    def test_find_similar_nonexistent_store(self, mock_embedding_model):
        """Test finding similar texts from non-existent store."""
        with pytest.raises(FileNotFoundError):
            find_similar_texts_from_vector_store(
                query="test", query_type="text", store="/nonexistent/path", k=2
            )


if __name__ == "__main__":
    pytest.main([__file__])
