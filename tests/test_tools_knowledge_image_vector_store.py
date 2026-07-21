# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for image vector store tools.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from world_understanding.tools.knowledge.image_vector_store import (
    BuildImageVectorStoreInput,
    BuildImageVectorStoreOutput,
    FindSimilarImagesInput,
    FindSimilarImagesOutput,
    build_image_vector_store_tool,
    find_similar_images_tool,
)


class TestBuildImageVectorStoreTool:
    """Test suite for build_image_vector_store_tool."""

    def test_input_model_validation(self):
        """Test that input model validates correctly."""
        # Valid single path input
        input1 = BuildImageVectorStoreInput(
            source="/path/to/images",
            index_type="IndexFlatL2",
            normalize_embeddings=False,
        )
        assert input1.source == "/path/to/images"
        assert input1.index_type == "IndexFlatL2"
        assert input1.normalize_embeddings is False

        # Valid list of paths input
        input2 = BuildImageVectorStoreInput(
            source=["/path/to/image1.jpg", "/path/to/image2.png"],
            save_path="/path/to/store",
        )
        assert len(input2.source) == 2
        assert input2.save_path == "/path/to/store"

    def test_output_model_structure(self):
        """Test that output model has correct structure."""
        output = BuildImageVectorStoreOutput(
            success=True,
            num_images_indexed=10,
            index_type="IndexFlatL2",
            embedding_dimension=512,
            save_path="/path/to/store",
            errors=[],
        )

        assert output.success is True
        assert output.num_images_indexed == 10
        assert output.embedding_dimension == 512
        assert len(output.errors) == 0

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.build_image_vector_store_func"
    )
    def test_successful_build_from_directory(self, mock_build_func):
        """Test successful building of vector store from directory."""
        # Mock the vector store
        mock_store = MagicMock()
        mock_store.metadata_store = {"img1": {}, "img2": {}, "img3": {}}
        mock_store.dimension = 512
        mock_build_func.return_value = mock_store

        # Create input
        inputs = BuildImageVectorStoreInput(
            source="/path/to/images", index_type="IndexHNSWFlat", recursive=True
        )

        # Call tool
        output = build_image_vector_store_tool(inputs)

        # Verify results
        assert output.success is True
        assert output.num_images_indexed == 3
        assert output.index_type == "IndexHNSWFlat"
        assert output.embedding_dimension == 512
        assert len(output.errors) == 0

        # Verify function was called correctly
        mock_build_func.assert_called_once_with(
            source="/path/to/images",
            index_type="IndexHNSWFlat",
            normalize_embeddings=False,
            recursive=True,
        )

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.build_image_vector_store_func"
    )
    def test_successful_build_with_save(self, mock_build_func):
        """Test building and saving vector store."""
        # Mock the vector store
        mock_store = MagicMock()
        mock_store.metadata_store = {"img1": {}}
        mock_store.dimension = 768
        mock_store.save = MagicMock()
        mock_build_func.return_value = mock_store

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = str(Path(temp_dir) / "test_store")

            # Create input with save_path
            inputs = BuildImageVectorStoreInput(
                source="/path/to/images", save_path=save_path
            )

            # Call tool
            output = build_image_vector_store_tool(inputs)

            # Verify results
            assert output.success is True
            assert output.save_path == save_path
            mock_store.save.assert_called_once_with(save_path)

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.build_image_vector_store_func"
    )
    def test_file_not_found_error(self, mock_build_func):
        """Test handling of FileNotFoundError."""
        mock_build_func.side_effect = FileNotFoundError("Directory not found")

        inputs = BuildImageVectorStoreInput(source="/nonexistent/path")
        output = build_image_vector_store_tool(inputs)

        assert output.success is False
        assert output.num_images_indexed == 0
        assert len(output.errors) == 1
        assert "File not found" in output.errors[0]

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.build_image_vector_store_func"
    )
    def test_value_error(self, mock_build_func):
        """Test handling of ValueError."""
        mock_build_func.side_effect = ValueError("Invalid index type")

        inputs = BuildImageVectorStoreInput(source="/path/to/images")
        output = build_image_vector_store_tool(inputs)

        assert output.success is False
        assert len(output.errors) == 1
        assert "Invalid input" in output.errors[0]

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.build_image_vector_store_func"
    )
    def test_build_accepts_source_list(self, mock_build_func):
        """Test list source normalization before calling the build function."""
        mock_store = MagicMock()
        mock_store.metadata_store = {"img1": MagicMock(), "img2": MagicMock()}
        mock_store.dimension = 512
        mock_build_func.return_value = mock_store

        inputs = BuildImageVectorStoreInput(
            source=["/path/to/image1.png", "/path/to/image2.png"],
            index_type="IndexFlatIP",
            normalize_embeddings=True,
            recursive=False,
        )

        output = build_image_vector_store_tool(inputs)

        assert output.success is True
        assert output.num_images_indexed == 2
        mock_build_func.assert_called_once_with(
            source=["/path/to/image1.png", "/path/to/image2.png"],
            index_type="IndexFlatIP",
            normalize_embeddings=True,
            recursive=False,
        )


class TestFindSimilarImagesTool:
    """Test suite for find_similar_images_tool."""

    def test_input_model_validation(self):
        """Test that input model validates correctly."""
        inputs = FindSimilarImagesInput(
            query_image="/path/to/query.jpg",
            store_path="/path/to/store",
            k=10,
            filter_metadata={"category": "landscape"},
        )

        assert inputs.query_image == "/path/to/query.jpg"
        assert inputs.store_path == "/path/to/store"
        assert inputs.k == 10
        assert inputs.filter_metadata["category"] == "landscape"

    def test_output_model_structure(self):
        """Test that output model has correct structure."""
        output = FindSimilarImagesOutput(
            results=[
                {
                    "doc_id": "img1",
                    "score": 0.95,
                    "metadata": {"category": "landscape"},
                    "image_path": "/path/to/img1.jpg",
                }
            ],
            num_results=1,
            query_image="/path/to/query.jpg",
            search_errors=[],
        )

        assert output.num_results == 1
        assert len(output.results) == 1
        assert output.results[0]["score"] == 0.95
        assert output.query_image == "/path/to/query.jpg"

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.find_similar_images_func"
    )
    def test_successful_search(self, mock_find_func):
        """Test successful image similarity search."""
        # Mock search results
        mock_result1 = MagicMock()
        mock_result1.doc_id = "img1"
        mock_result1.score = 0.95
        mock_result1.metadata = {"category": "landscape"}
        mock_result1.document.image_path = "/path/to/img1.jpg"
        mock_result1.document.text_content = None

        mock_result2 = MagicMock()
        mock_result2.doc_id = "img2"
        mock_result2.score = 0.87
        mock_result2.metadata = {"category": "landscape"}
        mock_result2.document.image_path = "/path/to/img2.jpg"
        mock_result2.document.text_content = "Image description"

        mock_find_func.return_value = [mock_result1, mock_result2]

        # Create input
        inputs = FindSimilarImagesInput(
            query_image="/path/to/query.jpg", store_path="/path/to/store", k=5
        )

        # Call tool
        output = find_similar_images_tool(inputs)

        # Verify results
        assert output.num_results == 2
        assert len(output.results) == 2

        # Check first result
        assert output.results[0]["doc_id"] == "img1"
        assert output.results[0]["score"] == 0.95
        assert output.results[0]["image_path"] == "/path/to/img1.jpg"
        assert "text_content" not in output.results[0]  # None is not included

        # Check second result
        assert output.results[1]["doc_id"] == "img2"
        assert output.results[1]["score"] == 0.87
        assert output.results[1]["text_content"] == "Image description"

        # Verify function was called correctly
        mock_find_func.assert_called_once_with(
            query_image="/path/to/query.jpg",
            store="/path/to/store",
            k=5,
            filter_metadata=None,
        )

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.find_similar_images_func"
    )
    def test_search_with_metadata_filter(self, mock_find_func):
        """Test search with metadata filtering."""
        mock_find_func.return_value = []

        inputs = FindSimilarImagesInput(
            query_image="/path/to/query.jpg",
            store_path="/path/to/store",
            k=3,
            filter_metadata={"category": "portrait"},
        )

        output = find_similar_images_tool(inputs)

        assert output.num_results == 0
        mock_find_func.assert_called_once_with(
            query_image="/path/to/query.jpg",
            store="/path/to/store",
            k=3,
            filter_metadata={"category": "portrait"},
        )

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.find_similar_images_func"
    )
    def test_file_not_found_error_search(self, mock_find_func):
        """Test handling of FileNotFoundError in search."""
        mock_find_func.side_effect = FileNotFoundError("Store not found")

        inputs = FindSimilarImagesInput(
            query_image="/path/to/query.jpg", store_path="/nonexistent/store"
        )

        output = find_similar_images_tool(inputs)

        assert output.num_results == 0
        assert len(output.results) == 0
        assert len(output.search_errors) == 1
        assert "File not found" in output.search_errors[0]

    @patch(
        "world_understanding.tools.knowledge.image_vector_store.find_similar_images_func"
    )
    def test_value_error_search(self, mock_find_func):
        """Test handling of ValueError in search."""
        mock_find_func.side_effect = ValueError("Invalid query image")

        inputs = FindSimilarImagesInput(
            query_image="/invalid/query.jpg", store_path="/path/to/store"
        )

        output = find_similar_images_tool(inputs)

        assert len(output.search_errors) == 1
        assert "Invalid input" in output.search_errors[0]


def test_tool_registration():
    """Test that tools are properly registered."""
    from world_understanding.tools.base import get_tool_registry

    registry = get_tool_registry()

    # Check that our tools are registered
    assert "build_image_vector_store" in registry
    assert "find_similar_images" in registry

    # Verify tool specifications
    build_tool = registry["build_image_vector_store"]
    assert build_tool.spec.version == "0.1.0"
    assert build_tool.spec.input_model == BuildImageVectorStoreInput
    assert build_tool.spec.output_model == BuildImageVectorStoreOutput

    find_tool = registry["find_similar_images"]
    assert find_tool.spec.version == "0.1.0"
    assert find_tool.spec.input_model == FindSimilarImagesInput
    assert find_tool.spec.output_model == FindSimilarImagesOutput
