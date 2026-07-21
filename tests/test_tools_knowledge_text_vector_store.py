# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for text vector store tools.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from world_understanding.tools.knowledge.text_vector_store import (
    BuildTextVectorStoreInput,
    BuildTextVectorStoreOutput,
    FindSimilarTextsInput,
    FindSimilarTextsOutput,
    build_text_vector_store_tool,
    find_similar_texts_tool,
)


class TestBuildTextVectorStoreTool:
    """Test suite for build_text_vector_store_tool."""

    def test_input_model_validation(self):
        """Test that input model validates correctly."""
        # Valid text only input
        input1 = BuildTextVectorStoreInput(
            text_source="/path/to/texts",
            index_type="IndexFlatL2",
            normalize_embeddings=False,
        )
        assert input1.text_source == "/path/to/texts"
        assert input1.image_source is None
        assert input1.index_type == "IndexFlatL2"

        # Valid image input (will be captioned)
        input2 = BuildTextVectorStoreInput(
            image_source="/path/to/images",
            recursive=False,
        )
        assert input2.text_source is None
        assert input2.image_source == "/path/to/images"
        assert input2.recursive is False

        # Valid mixed input
        input3 = BuildTextVectorStoreInput(
            text_source=["/path/to/text1.txt", "/path/to/text2.md"],
            image_source=["/path/to/img1.jpg", "/path/to/img2.png"],
            save_path="/path/to/store",
        )
        assert len(input3.text_source) == 2
        assert len(input3.image_source) == 2
        assert input3.save_path == "/path/to/store"

    def test_output_model_structure(self):
        """Test that output model has correct structure."""
        output = BuildTextVectorStoreOutput(
            success=True,
            num_documents_indexed=15,
            num_texts=10,
            num_images_captioned=5,
            index_type="IndexFlatL2",
            embedding_dimension=768,
            save_path="/path/to/store",
            errors=[],
        )

        assert output.success is True
        assert output.num_documents_indexed == 15
        assert output.num_texts == 10
        assert output.num_images_captioned == 5
        assert output.embedding_dimension == 768
        assert len(output.errors) == 0

    @patch(
        "world_understanding.tools.knowledge.text_vector_store.build_text_vector_store_func"
    )
    def test_successful_build_mixed_sources(self, mock_build_func):
        """Test successful building of vector store from mixed sources."""
        # Mock the vector store
        mock_store = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.image_path = None
        mock_doc2 = MagicMock()
        mock_doc2.image_path = None
        mock_doc3 = MagicMock()
        mock_doc3.image_path = "/path/to/img1.jpg"
        mock_doc4 = MagicMock()
        mock_doc4.image_path = "/path/to/img2.jpg"
        mock_doc5 = MagicMock()
        mock_doc5.image_path = None

        mock_store.metadata_store = {
            "doc1": MagicMock(document=mock_doc1),
            "doc2": MagicMock(document=mock_doc2),
            "img1": MagicMock(document=mock_doc3),
            "img2": MagicMock(document=mock_doc4),
            "doc3": MagicMock(document=mock_doc5),
        }
        mock_store.dimension = 768
        mock_build_func.return_value = mock_store

        # Create input
        inputs = BuildTextVectorStoreInput(
            text_source="/path/to/texts",
            image_source="/path/to/images",
            index_type="IndexHNSWFlat",
            recursive=True,
        )

        # Call tool
        output = build_text_vector_store_tool(inputs)

        # Verify results
        assert output.success is True
        assert output.num_documents_indexed == 5
        assert output.num_texts == 3
        assert output.num_images_captioned == 2
        assert output.index_type == "IndexHNSWFlat"
        assert output.embedding_dimension == 768
        assert len(output.errors) == 0

        # Verify function was called correctly
        mock_build_func.assert_called_once_with(
            text_source="/path/to/texts",
            image_source="/path/to/images",
            index_type="IndexHNSWFlat",
            normalize_embeddings=False,
            recursive=True,
        )

    @patch(
        "world_understanding.tools.knowledge.text_vector_store.build_text_vector_store_func"
    )
    def test_successful_build_with_save(self, mock_build_func):
        """Test building and saving text vector store."""
        # Mock the vector store
        mock_store = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.image_path = None
        mock_store.metadata_store = {
            "doc1": MagicMock(document=mock_doc1),
        }
        mock_store.dimension = 1024
        mock_store.save = MagicMock()
        mock_build_func.return_value = mock_store

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = str(Path(temp_dir) / "test_store")

            # Create input with save_path
            inputs = BuildTextVectorStoreInput(
                text_source=["text1", "text2"],
                save_path=save_path,
            )

            # Call tool
            output = build_text_vector_store_tool(inputs)

            # Verify results
            assert output.success is True
            assert output.save_path == save_path
            mock_store.save.assert_called_once_with(save_path)

    @patch(
        "world_understanding.tools.knowledge.text_vector_store.build_text_vector_store_func"
    )
    def test_file_not_found_error(self, mock_build_func):
        """Test handling of FileNotFoundError."""
        mock_build_func.side_effect = FileNotFoundError("Directory not found")

        inputs = BuildTextVectorStoreInput(text_source="/nonexistent/path")
        output = build_text_vector_store_tool(inputs)

        assert output.success is False
        assert output.num_documents_indexed == 0
        assert len(output.errors) == 1
        assert "File not found" in output.errors[0]

    @patch(
        "world_understanding.tools.knowledge.text_vector_store.build_text_vector_store_func"
    )
    def test_build_accepts_image_only_list(self, mock_build_func):
        """Test empty text source and list image source normalization."""
        mock_store = MagicMock()
        mock_doc = MagicMock()
        mock_doc.image_path = "/path/to/image1.png"
        mock_store.metadata_store = {"img1": MagicMock(document=mock_doc)}
        mock_store.dimension = 512
        mock_build_func.return_value = mock_store

        inputs = BuildTextVectorStoreInput(
            image_source=["/path/to/image1.png"],
            index_type="IndexFlatIP",
            normalize_embeddings=True,
            recursive=False,
        )

        output = build_text_vector_store_tool(inputs)

        assert output.success is True
        assert output.num_texts == 0
        assert output.num_images_captioned == 1
        mock_build_func.assert_called_once_with(
            text_source=None,
            image_source=["/path/to/image1.png"],
            index_type="IndexFlatIP",
            normalize_embeddings=True,
            recursive=False,
        )


class TestFindSimilarTextsTool:
    """Test suite for find_similar_texts_tool."""

    def test_input_model_validation(self):
        """Test that input model validates correctly."""
        # Text query
        inputs1 = FindSimilarTextsInput(
            query="search for similar text",
            query_type="text",
            store_path="/path/to/store",
            k=10,
        )
        assert inputs1.query == "search for similar text"
        assert inputs1.query_type == "text"
        assert inputs1.k == 10

        # Image query (will be captioned before search)
        inputs2 = FindSimilarTextsInput(
            query="/path/to/query.jpg",
            query_type="image",
            store_path="/path/to/store",
            filter_metadata={"category": "technical"},
        )
        assert inputs2.query == "/path/to/query.jpg"
        assert inputs2.query_type == "image"
        assert inputs2.filter_metadata["category"] == "technical"

    def test_output_model_structure(self):
        """Test that output model has correct structure."""
        output = FindSimilarTextsOutput(
            results=[
                {
                    "doc_id": "doc1",
                    "score": 0.95,
                    "text_content": "Sample text content",
                    "source_path": "/path/to/doc1.txt",
                },
                {
                    "doc_id": "img1",
                    "score": 0.87,
                    "text_content": "Caption of the image",
                    "source_image": "/path/to/img1.jpg",
                    "is_captioned": True,
                },
            ],
            num_results=2,
            query="search query",
            query_type="text",
            search_errors=[],
        )

        assert output.num_results == 2
        assert len(output.results) == 2
        assert "source_path" in output.results[0]
        assert "is_captioned" in output.results[1]

    @patch(
        "world_understanding.tools.knowledge.text_vector_store.find_similar_texts_func"
    )
    def test_successful_text_search(self, mock_find_func):
        """Test successful text similarity search."""
        # Mock search results
        mock_result1 = MagicMock()
        mock_result1.score = 0.95
        mock_doc1 = MagicMock()
        mock_doc1.text_content = "Document text content"
        mock_doc1.text_path = "/path/to/doc1.txt"
        mock_doc1.image_path = None
        mock_result1.document = mock_doc1
        mock_result1.metadata = {"source": "doc1.txt"}

        mock_result2 = MagicMock()
        mock_result2.score = 0.82
        mock_doc2 = MagicMock()
        mock_doc2.text_content = "Captioned image content"
        mock_doc2.text_path = None
        mock_doc2.image_path = "/path/to/img.jpg"
        mock_result2.document = mock_doc2
        mock_result2.metadata = {"source": "img1.jpg"}

        mock_find_func.return_value = [mock_result1, mock_result2]

        # Create input
        inputs = FindSimilarTextsInput(
            query="machine learning",
            query_type="text",
            store_path="/path/to/store",
            k=5,
        )

        # Call tool
        output = find_similar_texts_tool(inputs)

        # Verify results
        assert output.num_results == 2
        assert len(output.results) == 2

        # Check first result (text document)
        assert output.results[0]["score"] == 0.95
        assert output.results[0]["text_content"] == "Document text content"
        assert output.results[0]["source_path"] == "/path/to/doc1.txt"
        assert "is_captioned" not in output.results[0]

        # Check second result (captioned image)
        assert output.results[1]["score"] == 0.82
        assert output.results[1]["text_content"] == "Captioned image content"
        assert output.results[1]["source_image"] == "/path/to/img.jpg"
        assert output.results[1]["is_captioned"] is True

        # Verify function was called correctly
        mock_find_func.assert_called_once_with(
            query="machine learning",
            query_type="text",
            store="/path/to/store",
            k=5,
            filter_metadata=None,
        )

    @patch(
        "world_understanding.tools.knowledge.text_vector_store.find_similar_texts_func"
    )
    def test_image_query(self, mock_find_func):
        """Test image query (will be captioned)."""
        mock_find_func.return_value = []

        inputs = FindSimilarTextsInput(
            query="/path/to/query.jpg",
            query_type="image",
            store_path="/path/to/store",
            k=3,
        )

        output = find_similar_texts_tool(inputs)

        assert output.num_results == 0
        mock_find_func.assert_called_once_with(
            query="/path/to/query.jpg",
            query_type="image",
            store="/path/to/store",
            k=3,
            filter_metadata=None,
        )

    @patch(
        "world_understanding.tools.knowledge.text_vector_store.find_similar_texts_func"
    )
    def test_file_not_found_error_search(self, mock_find_func):
        """Test handling of FileNotFoundError in search."""
        mock_find_func.side_effect = FileNotFoundError("Store not found")

        inputs = FindSimilarTextsInput(
            query="test query",
            query_type="text",
            store_path="/nonexistent/store",
        )

        output = find_similar_texts_tool(inputs)

        assert output.num_results == 0
        assert len(output.results) == 0
        assert len(output.search_errors) == 1
        assert "File not found" in output.search_errors[0]


def test_tool_registration():
    """Test that tools are properly registered."""
    from world_understanding.tools.base import get_tool_registry

    registry = get_tool_registry()

    # Check that our tools are registered
    assert "build_text_vector_store" in registry
    assert "find_similar_texts" in registry

    # Verify tool specifications
    build_tool = registry["build_text_vector_store"]
    assert build_tool.spec.version == "0.1.0"
    assert build_tool.spec.input_model == BuildTextVectorStoreInput
    assert build_tool.spec.output_model == BuildTextVectorStoreOutput

    find_tool = registry["find_similar_texts"]
    assert find_tool.spec.version == "0.1.0"
    assert find_tool.spec.input_model == FindSimilarTextsInput
    assert find_tool.spec.output_model == FindSimilarTextsOutput
