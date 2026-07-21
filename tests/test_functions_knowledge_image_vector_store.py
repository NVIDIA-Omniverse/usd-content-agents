# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the image vector store."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

# Image artifact no longer used - functions use PIL Images directly
from world_understanding.functions.knowledge.base_vector_store import BaseSearchResult
from world_understanding.functions.knowledge.image_vector_store import (
    ImageVectorStore,
    build_image_vector_store,
    find_similar_images_from_vector_store,
)
from world_understanding.functions.models.base_embedding_model import BaseEmbeddingModel


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model for testing."""

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/llama-nemotron-embed-vl-1b-v2",
        embedding_dimension: int = 512,
    ):
        super().__init__(
            api_key=api_key, model=model, embedding_dimension=embedding_dimension
        )

    def embed_image(
        self, image: str | Path | PILImage.Image | np.ndarray, **kwargs
    ) -> np.ndarray:
        """Generate a mock embedding based on image content."""
        # For testing, generate different embeddings based on input type
        if isinstance(image, np.ndarray):
            # Use mean of array as seed
            seed = int(np.mean(image) * 1000) % 1000
        elif isinstance(image, PILImage.Image):
            # Use image size as seed
            seed = image.width + image.height
        elif isinstance(image, str | Path):
            # Use path length as seed
            seed = len(str(image))
        else:
            seed = 42

        np.random.seed(seed)
        return np.random.rand(self.embedding_dimension).astype(np.float32)

    def embed_images(
        self, images: list[str | Path | PILImage.Image | np.ndarray], **kwargs
    ) -> list[np.ndarray]:
        """Embed multiple images."""
        return [self.embed_image(img, **kwargs) for img in images]

    def embed_text(self, text: str, **kwargs) -> np.ndarray:
        """Generate a mock text embedding."""
        np.random.seed(len(text) % 1000)
        return np.random.rand(self.embedding_dimension).astype(np.float32)

    def embed_texts(self, texts: list[str], **kwargs) -> list[np.ndarray]:
        """Generate mock text embeddings."""
        return [self.embed_text(text, **kwargs) for text in texts]

    def list_available_models(self) -> list[str]:
        """Return available models."""
        return ["nvidia/llama-nemotron-embed-vl-1b-v2"]


class TestImageVectorStore:
    """Test cases for ImageVectorStore."""

    @pytest.fixture
    def mock_embedding_model(self) -> MockEmbeddingModel:
        """Create a mock embedding model."""
        return MockEmbeddingModel(api_key="dummy", embedding_dimension=512)

    @pytest.fixture
    def vector_store(self, mock_embedding_model) -> ImageVectorStore:
        """Create a vector store instance."""
        return ImageVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatL2",
        )

    @pytest.fixture
    def sample_images(self, tmp_path: Path) -> list[Path]:
        """Create sample test images."""
        images = []
        for i in range(5):
            # Create different colored images
            color = (i * 50, 100, 255 - i * 50)
            img = PILImage.new("RGB", (100, 100), color)
            path = tmp_path / f"test_image_{i}.png"
            img.save(path)
            images.append(path)
        return images

    def test_initialization(self, mock_embedding_model):
        """Test vector store initialization."""
        # Test with explicit dimension
        store = ImageVectorStore(
            embedding_model=mock_embedding_model,
        )
        assert store.embedding_model.embedding_dimension == 512
        assert store.index_type == "IndexFlatL2"
        assert store.num_documents == 0

        # Test with auto-detected dimension
        store_auto = ImageVectorStore(embedding_model=mock_embedding_model)
        assert store_auto.embedding_model.embedding_dimension == 512

    def test_add_single_image(self, vector_store, sample_images):
        """Test adding a single image."""
        # Add image from path
        image_path = sample_images[0]
        metadata = {"name": "test_image", "category": "sample"}
        image_id = vector_store.add_image(image=image_path, metadata=metadata)

        assert image_id == 0
        assert vector_store.num_documents == 1
        assert image_id in vector_store.metadata_store

        # Verify metadata
        stored_metadata = vector_store.metadata_store[image_id]
        assert stored_metadata.document.image_path == str(image_path)
        assert stored_metadata.document.metadata == metadata

    def test_add_multiple_images(self, vector_store, sample_images):
        """Test adding multiple images."""
        metadata_list = [
            {"name": f"image_{i}", "index": i} for i in range(len(sample_images))
        ]

        ids = vector_store.add_images(images=sample_images, metadata_list=metadata_list)

        assert len(ids) == len(sample_images)
        assert vector_store.num_documents == len(sample_images)
        assert ids == list(range(len(sample_images)))

        # Verify all metadata
        for i, image_id in enumerate(ids):
            stored_metadata = vector_store.metadata_store[image_id]
            assert stored_metadata.document.metadata == metadata_list[i]

    def test_search_similar_images(self, vector_store, sample_images):
        """Test searching for similar images."""
        # Add images
        vector_store.add_images(sample_images)

        # Search with the first image
        results = vector_store.search(query_image=sample_images[0], k=3)

        assert len(results) == 3
        assert all(isinstance(r, BaseSearchResult) for r in results)

        # First result should be the query image itself (exact match)
        assert results[0].score < 0.01  # Very small distance
        assert results[0].rank == 0

        # Results should be ordered by similarity
        for i in range(1, len(results)):
            assert results[i].score >= results[i - 1].score
            assert results[i].rank == i

    def test_search_with_metadata_filter(self, vector_store, sample_images):
        """Test searching with metadata filters."""
        # Add images with different categories
        for i, img in enumerate(sample_images):
            metadata = {"category": "even" if i % 2 == 0 else "odd", "index": i}
            vector_store.add_image(img, metadata=metadata)

        # Search for similar images with category filter
        results = vector_store.search(
            query_image=sample_images[0], k=5, filter_metadata={"category": "even"}
        )

        # Should only return images with matching category
        assert all(r.document.metadata["category"] == "even" for r in results)
        assert len(results) <= 3  # Only 3 even-indexed images

    def test_search_with_embedding(self, vector_store, mock_embedding_model):
        """Test searching with a pre-computed embedding."""
        # Add some images
        dummy_images = [np.random.rand(100, 100, 3) for _ in range(5)]
        for i, img_array in enumerate(dummy_images):
            # Convert numpy array to PIL Image
            pil_img = PILImage.fromarray((img_array * 255).astype(np.uint8))
            vector_store.add_image(pil_img, metadata={"index": i})

        # Search with an embedding directly
        query_embedding = mock_embedding_model.embed_image(dummy_images[0])
        results = vector_store.search(query_embedding=query_embedding, k=3)

        assert len(results) == 3
        assert results[0].document.metadata["index"] == 0  # Should match first image

    def test_update_metadata(self, vector_store, sample_images):
        """Test updating metadata for an existing image."""
        # Add image
        image_id = vector_store.add_image(
            sample_images[0], metadata={"original": "metadata"}
        )

        # Update metadata
        new_metadata = {"updated": "metadata", "version": 2}
        success = vector_store.update_metadata(image_id, new_metadata)

        assert success
        assert vector_store.metadata_store[image_id].document.metadata == new_metadata

        # Try updating non-existent image
        assert not vector_store.update_metadata(999, {})

    def test_remove_image(self, vector_store, sample_images):
        """Test removing an image from the store."""
        # Add images
        ids = vector_store.add_images(sample_images[:3])

        # Remove middle image
        success = vector_store.remove_document(ids[1])
        assert success
        assert ids[1] not in vector_store.metadata_store

        # Try removing non-existent image
        assert not vector_store.remove_document(999)

    def test_save_and_load(self, vector_store, sample_images, tmp_path, monkeypatch):
        """Test saving and loading the vector store."""
        # Add images with metadata
        metadata_list = [{"name": f"img_{i}", "idx": i} for i in range(3)]
        vector_store.add_images(sample_images[:3], metadata_list=metadata_list)

        # Save to disk
        save_path = tmp_path / "vector_store"
        vector_store.save(save_path)

        # Verify files were created
        assert (save_path / "index.faiss").exists()
        assert (save_path / "metadata.json").exists()

        # Mock the create_image_embedding_model to return our mock model

        def mock_create_model(*args, **kwargs):
            return MockEmbeddingModel(api_key="dummy", embedding_dimension=512)

        monkeypatch.setattr(
            "world_understanding.functions.knowledge.image_vector_store.create_image_embedding_model",
            mock_create_model,
        )

        # Load into new instance (embedding model is automatically loaded from metadata)
        loaded_store = ImageVectorStore.load(save_path)

        # Verify loaded data
        assert (
            loaded_store.embedding_model.embedding_dimension
            == vector_store.embedding_model.embedding_dimension
        )
        assert loaded_store.index_type == vector_store.index_type
        assert loaded_store.num_documents == 3
        assert len(loaded_store.metadata_store) == 3

    def test_load_without_embedding_model_info(
        self, vector_store, sample_images, tmp_path
    ):
        """Test loading a store that doesn't have embedding model information."""
        # Add images and save
        vector_store.add_images(sample_images[:2])
        save_path = tmp_path / "vector_store"
        vector_store.save(save_path)

        # Manually remove embedding model info from metadata
        metadata_path = save_path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Remove embedding_model field
        if "embedding_model" in metadata:
            del metadata["embedding_model"]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Should fail when trying to load without embedding model info
        with pytest.raises(KeyError, match="embedding_model"):
            ImageVectorStore.load(save_path)

    def test_clear(self, vector_store, sample_images):
        """Test clearing the vector store."""
        # Add images
        vector_store.add_images(sample_images)
        assert vector_store.num_documents == len(sample_images)

        # Clear store
        vector_store.clear()
        assert vector_store.num_documents == 0
        assert len(vector_store.metadata_store) == 0
        assert vector_store.index.ntotal == 0

    def test_different_index_types(self, mock_embedding_model):
        """Test initialization with different index types."""
        # Test IndexFlatIP
        store_ip = ImageVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexFlatIP",
            normalize_embeddings=True,
        )
        assert store_ip.index_type == "IndexFlatIP"

        # Test IndexHNSWFlat
        store_hnsw = ImageVectorStore(
            embedding_model=mock_embedding_model,
            index_type="IndexHNSWFlat",
        )
        assert store_hnsw.index_type == "IndexHNSWFlat"

    def test_image_from_pil(self, vector_store):
        """Test adding PIL Image objects."""
        # Create PIL image
        pil_img = PILImage.new("RGB", (200, 200), color=(255, 0, 0))

        # Add to store directly using PIL image
        image_id = vector_store.add_image(pil_img, metadata={"type": "pil"})
        assert image_id == 0
        assert vector_store.num_documents == 1

        # Search should work with PIL image
        results = vector_store.search(query_image=pil_img, k=1)
        assert len(results) == 1
        assert results[0].document.metadata["type"] == "pil"

    def test_error_handling(self, vector_store):
        """Test error handling in various scenarios."""
        # Test searching empty index
        with pytest.raises(ValueError, match="Index is empty"):
            vector_store.search(query_image="dummy.jpg", k=5)

        # Test invalid image type
        with pytest.raises(ValueError, match="Unsupported image type"):
            vector_store.add_image(123)  # type: ignore

        # Test mismatched metadata list
        with pytest.raises(ValueError, match="doesn't match number of images"):
            vector_store.add_images(
                ["img1.jpg", "img2.jpg"], metadata_list=[{"meta": 1}]
            )

        # Test loading from non-existent path
        with pytest.raises(FileNotFoundError):
            ImageVectorStore.load("/non/existent/path")

    def test_automatic_embedding_model_loading(
        self, vector_store, sample_images, tmp_path, monkeypatch
    ):
        """Test that embedding model is automatically loaded from metadata."""
        # Add images and save
        vector_store.add_images(sample_images[:2])
        save_path = tmp_path / "vector_store"
        vector_store.save(save_path)

        # Mock the create_image_embedding_model to return our mock model
        def mock_create_model(*args, **kwargs):
            return MockEmbeddingModel(api_key="dummy", embedding_dimension=512)

        monkeypatch.setattr(
            "world_understanding.functions.knowledge.image_vector_store.create_image_embedding_model",
            mock_create_model,
        )

        # Load the store (should automatically recreate the embedding model)
        loaded_store = ImageVectorStore.load(save_path)

        # Verify that the embedding model was loaded
        assert loaded_store.embedding_model is not None
        assert hasattr(loaded_store.embedding_model, "embed_image")
        assert hasattr(loaded_store.embedding_model, "embedding_dimension")


class TestBuildImageVectorStore:
    """Tests for build_image_vector_store function."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(api_key="dummy", embedding_dimension=512)

    def test_build_creates_embedding_model_from_api_key(self, monkeypatch):
        """Test default embedding model creation when an API key is configured."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
        mock_model = MockEmbeddingModel(api_key="test-api-key", embedding_dimension=256)
        expected_store = object()
        captured_kwargs = {}

        def create_model(**kwargs):
            captured_kwargs["factory"] = kwargs
            return mock_model

        def build_store(**kwargs):
            captured_kwargs["build"] = kwargs
            return expected_store

        monkeypatch.setattr(
            "world_understanding.functions.knowledge.image_vector_store.create_image_embedding_model",
            create_model,
        )
        monkeypatch.setattr(ImageVectorStore, "build_vector_store", build_store)

        vector_store = build_image_vector_store(source=["image.png"])

        assert vector_store is expected_store
        assert captured_kwargs["factory"] == {
            "backend": "nim",
            "api_key": "test-api-key",
        }
        assert captured_kwargs["build"]["embedding_model"] is mock_model

    @pytest.fixture
    def temp_image_dir(self, tmp_path):
        """Create a temporary directory with test images."""
        image_dir = tmp_path / "test_images"
        image_dir.mkdir()

        # Create test images
        colors = ["red", "green", "blue", "yellow"]
        for _i, color in enumerate(colors):
            img = PILImage.new("RGB", (100, 100), color=color)
            img.save(image_dir / f"{color}.png")

            # Also create a jpg
            img.save(image_dir / f"{color}.jpg")

        # Create a subdirectory with more images
        subdir = image_dir / "subdir"
        subdir.mkdir()
        for _i, color in enumerate(["purple", "orange"]):
            img = PILImage.new("RGB", (100, 100), color=color)
            img.save(subdir / f"{color}.png")

        return image_dir

    def test_build_from_directory(self, temp_image_dir, mock_embedding_model):
        """Test building vector store from a directory."""
        store = build_image_vector_store(
            source=temp_image_dir,
            embedding_model=mock_embedding_model,
            recursive=True,
        )

        assert isinstance(store, ImageVectorStore)
        assert store.num_documents == 10  # 4 colors x 2 formats + 2 in subdir

    def test_build_from_directory_non_recursive(
        self, temp_image_dir, mock_embedding_model
    ):
        """Test building vector store from a directory without recursion."""
        store = build_image_vector_store(
            source=temp_image_dir,
            embedding_model=mock_embedding_model,
            recursive=False,
        )

        assert isinstance(store, ImageVectorStore)
        assert store.num_documents == 8  # 4 colors x 2 formats, no subdir

    def test_build_from_file_list(self, temp_image_dir, mock_embedding_model):
        """Test building vector store from a list of files."""
        image_files = [
            temp_image_dir / "red.png",
            temp_image_dir / "green.png",
            temp_image_dir / "blue.png",
        ]

        store = build_image_vector_store(
            source=image_files,
            embedding_model=mock_embedding_model,
        )

        assert isinstance(store, ImageVectorStore)
        assert store.num_documents == 3

    def test_build_with_metadata_extractor(self, temp_image_dir, mock_embedding_model):
        """Test building with custom metadata extraction."""

        def extract_metadata(path):
            return {
                "filename": Path(path).name,
                "color": Path(path).stem,
                "extension": Path(path).suffix,
            }

        store = build_image_vector_store(
            source=temp_image_dir,
            embedding_model=mock_embedding_model,
            metadata_extractor=extract_metadata,
            recursive=False,
        )

        # Check that metadata was extracted
        metadata_store = store.get_all_metadata()
        for _, meta in metadata_store.items():
            assert "filename" in meta.document.metadata
            assert "color" in meta.document.metadata
            assert "extension" in meta.document.metadata

    def test_build_with_specific_extensions(self, temp_image_dir, mock_embedding_model):
        """Test building with specific image extensions."""
        store = build_image_vector_store(
            source=temp_image_dir,
            embedding_model=mock_embedding_model,
            image_extensions=(".png",),
            recursive=True,
        )

        assert store.num_documents == 6  # Only PNG files

    def test_build_with_different_index_types(
        self, temp_image_dir, mock_embedding_model
    ):
        """Test building with different FAISS index types."""
        index_types = ["IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat"]

        for index_type in index_types:
            store = build_image_vector_store(
                source=[temp_image_dir / "red.png"],
                embedding_model=mock_embedding_model,
                index_type=index_type,
                normalize_embeddings=(index_type == "IndexFlatIP"),
            )

            assert isinstance(store, ImageVectorStore)
            assert store.index_type == index_type
            assert store.num_documents == 1

    def test_build_empty_directory(self, tmp_path, mock_embedding_model):
        """Test building from empty directory raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(
            ValueError, match="No content could be added to the vector store"
        ):
            build_image_vector_store(
                source=empty_dir,
                embedding_model=mock_embedding_model,
            )

    def test_build_nonexistent_path(self, mock_embedding_model):
        """Test building from non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            build_image_vector_store(
                source="nonexistent/path",
                embedding_model=mock_embedding_model,
            )


class TestFindSimilarImagesFromVectorStore:
    """Tests for find_similar_images_from_vector_store function."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        return MockEmbeddingModel(api_key="dummy", embedding_dimension=512)

    @pytest.fixture
    def populated_store(self, tmp_path, mock_embedding_model):
        """Create a populated vector store with test images."""
        # Create test images
        image_paths = []
        for _i, color in enumerate(["red", "green", "blue", "yellow", "purple"]):
            img = PILImage.new("RGB", (100, 100), color=color)
            img_path = tmp_path / f"{color}.png"
            img.save(img_path)
            image_paths.append(img_path)

        # Build store with metadata
        def extract_metadata(path):
            return {"color": Path(path).stem, "index": Path(path).stem[0]}

        store = build_image_vector_store(
            source=image_paths,
            embedding_model=mock_embedding_model,
            metadata_extractor=extract_metadata,
        )

        # Save the store
        store_path = tmp_path / "test_store"
        store.save(store_path)

        return store, store_path, image_paths

    def test_find_similar_from_store_instance(self, populated_store):
        """Test finding similar images from a store instance."""
        store, _, image_paths = populated_store

        results = find_similar_images_from_vector_store(
            query_image=image_paths[0],
            store=store,  # Pass store instance directly
            k=3,
        )

        assert len(results) == 3
        assert all(isinstance(r, BaseSearchResult) for r in results)

    def test_find_similar_with_different_query_types(self, populated_store):
        """Test finding similar images with different query types."""
        store, store_path, image_paths = populated_store

        # Test with file path
        results1 = find_similar_images_from_vector_store(
            query_image=str(image_paths[0]),
            store=store,
            k=2,
        )

        # Test with PIL Image object
        pil_img = PILImage.open(image_paths[0])
        results2 = find_similar_images_from_vector_store(
            query_image=pil_img,
            store=store,
            k=2,
        )

        # Test with numpy array (embedding) - need to get embedding from loaded store
        embedding = store.embedding_model.embed_image(image_paths[0])
        results3 = find_similar_images_from_vector_store(
            query_image=embedding,
            store=store,
            k=2,
        )

        assert len(results1) == len(results2) == len(results3) == 2

    def test_find_similar_nonexistent_store(self):
        """Test finding similar images from non-existent store raises error."""
        with pytest.raises(FileNotFoundError):
            find_similar_images_from_vector_store(
                query_image="dummy.jpg",
                store="nonexistent/store",
            )
