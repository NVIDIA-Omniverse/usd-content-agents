# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for multimodal vector store tools.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from world_understanding.tools.knowledge.multimodal_vector_store import (
    BuildMultimodalVectorStoreInput,
    BuildMultimodalVectorStoreOutput,
    FindSimilarDocumentsInput,
    FindSimilarDocumentsOutput,
    build_multimodal_vector_store_tool,
    find_similar_documents_tool,
)


class TestBuildMultimodalVectorStoreTool:
    """Test suite for build_multimodal_vector_store_tool."""

    def test_input_model_validation(self):
        """Test that input model validates correctly."""
        # Valid text only input
        input1 = BuildMultimodalVectorStoreInput(
            text_source="/path/to/texts",
            index_type="IndexFlatL2",
            normalize_embeddings=False,
        )
        assert input1.text_source == "/path/to/texts"
        assert input1.image_source is None
        assert input1.index_type == "IndexFlatL2"

        # Valid image only input
        input2 = BuildMultimodalVectorStoreInput(
            image_source="/path/to/images",
            image_embedding_type="text",  # Use captions
        )
        assert input2.text_source is None
        assert input2.image_source == "/path/to/images"
        assert input2.image_embedding_type == "text"

        # Valid mixed input
        input3 = BuildMultimodalVectorStoreInput(
            text_source=["/path/to/text1.txt", "/path/to/text2.md"],
            image_source=["/path/to/img1.jpg", "/path/to/img2.png"],
            save_path="/path/to/store",
        )
        assert len(input3.text_source) == 2
        assert len(input3.image_source) == 2
        assert input3.save_path == "/path/to/store"

    def test_output_model_structure(self):
        """Test that output model has correct structure."""
        output = BuildMultimodalVectorStoreOutput(
            success=True,
            num_documents_indexed=15,
            num_texts=10,
            num_images=5,
            num_multimodal=0,
            index_type="IndexFlatL2",
            embedding_dimension=768,
            save_path="/path/to/store",
            errors=[],
        )

        assert output.success is True
        assert output.num_documents_indexed == 15
        assert output.num_texts == 10
        assert output.num_images == 5
        assert output.embedding_dimension == 768
        assert len(output.errors) == 0

    @patch(
        "world_understanding.tools.knowledge.multimodal_vector_store.build_multimodal_vector_store_func"
    )
    def test_successful_build_mixed_sources(self, mock_build_func):
        """Test successful building of vector store from mixed sources."""
        # Mock the vector store
        mock_store = MagicMock()

        # Create mock documents with get_content_type method
        mock_doc1 = MagicMock()
        mock_doc1.get_content_type.return_value = "text"
        mock_doc2 = MagicMock()
        mock_doc2.get_content_type.return_value = "text"
        mock_doc3 = MagicMock()
        mock_doc3.get_content_type.return_value = "image"
        mock_doc4 = MagicMock()
        mock_doc4.get_content_type.return_value = "image"
        mock_doc5 = MagicMock()
        mock_doc5.get_content_type.return_value = "text"

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
        inputs = BuildMultimodalVectorStoreInput(
            text_source="/path/to/texts",
            image_source="/path/to/images",
            index_type="IndexHNSWFlat",
            recursive=True,
        )

        # Call tool
        output = build_multimodal_vector_store_tool(inputs)

        # Verify results
        assert output.success is True
        assert output.num_documents_indexed == 5
        assert output.num_texts == 3
        assert output.num_images == 2
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
            image_embedding_type="image",
            metadata_extractor=None,
        )

    @patch(
        "world_understanding.tools.knowledge.multimodal_vector_store.build_multimodal_vector_store_func"
    )
    def test_image_only_build_with_filename_metadata_and_multimodal_count(
        self, mock_build_func
    ):
        """Test image-only build, filename metadata extraction, and multimodal count."""
        mock_store = MagicMock()

        mock_image_doc = MagicMock()
        mock_image_doc.get_content_type.return_value = "image"
        mock_multimodal_doc = MagicMock()
        mock_multimodal_doc.get_content_type.return_value = "multimodal"

        mock_store.metadata_store = {
            "img1": MagicMock(document=mock_image_doc),
            "multi1": MagicMock(document=mock_multimodal_doc),
        }
        mock_store.dimension = 512
        mock_build_func.return_value = mock_store

        inputs = BuildMultimodalVectorStoreInput(
            image_source=["/assets/080-2418-000_1_0001_structured.txt"],
            include_filename_metadata=True,
        )

        output = build_multimodal_vector_store_tool(inputs)

        assert output.success is True
        assert output.num_texts == 0
        assert output.num_images == 1
        assert output.num_multimodal == 1
        assert output.embedding_dimension == 512

        call_kwargs = mock_build_func.call_args.kwargs
        assert call_kwargs["text_source"] is None
        assert call_kwargs["image_source"] == [
            "/assets/080-2418-000_1_0001_structured.txt"
        ]

        metadata_extractor = call_kwargs["metadata_extractor"]
        structured_metadata = metadata_extractor(
            "/docs/080-2418-000_1_0001_structured.txt"
        )
        assert structured_metadata == {
            "filename": "080-2418-000_1_0001_structured.txt",
            "file_stem": "080-2418-000_1_0001_structured",
            "extension": ".txt",
            "parent_dir": "docs",
            "relative_path": "/docs/080-2418-000_1_0001_structured.txt",
            "document_id": "080-2418-000",
            "page_or_section": "0001",
            "content_type": "structured",
            "is_structured": True,
        }

        text_metadata = metadata_extractor(Path("/docs/guide_text.md"))
        assert text_metadata["filename"] == "guide_text.md"
        assert text_metadata["document_id"] == "guide"
        assert text_metadata["is_text"] is True

    @patch(
        "world_understanding.tools.knowledge.multimodal_vector_store.build_multimodal_vector_store_func"
    )
    def test_successful_build_with_save(self, mock_build_func):
        """Test building and saving multimodal vector store."""
        # Mock the vector store
        mock_store = MagicMock()

        # Create mock documents with get_content_type method
        mock_doc1 = MagicMock()
        mock_doc1.get_content_type.return_value = "text"
        mock_doc2 = MagicMock()
        mock_doc2.get_content_type.return_value = "image"

        mock_store.metadata_store = {
            "doc1": MagicMock(document=mock_doc1),
            "img1": MagicMock(document=mock_doc2),
        }
        mock_store.dimension = 1024
        mock_store.save = MagicMock()
        mock_build_func.return_value = mock_store

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = str(Path(temp_dir) / "test_store")

            # Create input with save_path
            inputs = BuildMultimodalVectorStoreInput(
                text_source=["text1", "text2"],
                image_source=["img1.jpg"],
                save_path=save_path,
            )

            # Call tool
            output = build_multimodal_vector_store_tool(inputs)

            # Verify results
            assert output.success is True
            assert output.save_path == save_path
            mock_store.save.assert_called_once_with(save_path)

    @patch(
        "world_understanding.tools.knowledge.multimodal_vector_store.build_multimodal_vector_store_func"
    )
    def test_text_only_store(self, mock_build_func):
        """Test building a text-only vector store."""
        # Mock the vector store
        mock_store = MagicMock()

        # Create mock documents with get_content_type method
        mock_doc1 = MagicMock()
        mock_doc1.get_content_type.return_value = "text"
        mock_doc2 = MagicMock()
        mock_doc2.get_content_type.return_value = "text"

        mock_store.metadata_store = {
            "doc1": MagicMock(document=mock_doc1),
            "doc2": MagicMock(document=mock_doc2),
        }
        mock_store.dimension = 768
        mock_build_func.return_value = mock_store

        # Create input with only text sources
        inputs = BuildMultimodalVectorStoreInput(
            text_source=["/path/to/text1.txt", "/path/to/text2.md"]
        )

        # Call tool
        output = build_multimodal_vector_store_tool(inputs)

        # Verify results
        assert output.success is True
        assert output.num_texts == 2
        assert output.num_images == 0
        assert output.num_documents_indexed == 2

    @patch(
        "world_understanding.tools.knowledge.multimodal_vector_store.build_multimodal_vector_store_func"
    )
    def test_file_not_found_error(self, mock_build_func):
        """Test handling of FileNotFoundError."""
        mock_build_func.side_effect = FileNotFoundError("Directory not found")

        inputs = BuildMultimodalVectorStoreInput(text_source="/nonexistent/path")
        output = build_multimodal_vector_store_tool(inputs)

        assert output.success is False
        assert output.num_documents_indexed == 0
        assert len(output.errors) == 1
        assert "File not found" in output.errors[0]


class TestFindSimilarDocumentsTool:
    """Test suite for find_similar_documents_tool."""

    def test_input_model_validation(self):
        """Test that input model validates correctly."""
        # Text query
        inputs1 = FindSimilarDocumentsInput(
            query="search for similar text",
            query_type="text",
            store_path="/path/to/store",
            k=10,
        )
        assert inputs1.query == "search for similar text"
        assert inputs1.query_type == "text"
        assert inputs1.k == 10

        # Image query
        inputs2 = FindSimilarDocumentsInput(
            query="/path/to/query.jpg",
            query_type="image",
            store_path="/path/to/store",
            embedding_type="text",  # Use caption-based search
            filter_metadata={"category": "landscape"},
        )
        assert inputs2.query == "/path/to/query.jpg"
        assert inputs2.query_type == "image"
        assert inputs2.embedding_type == "text"
        assert inputs2.filter_metadata["category"] == "landscape"

    def test_output_model_structure(self):
        """Test that output model has correct structure."""
        output = FindSimilarDocumentsOutput(
            results=[
                {
                    "doc_id": "doc1",
                    "score": 0.95,
                    "metadata": {"content_type": "text"},
                    "text_content": "Sample text",
                    "content_type": "text",
                },
                {
                    "doc_id": "img1",
                    "score": 0.87,
                    "metadata": {"content_type": "image"},
                    "image_path": "/path/to/img1.jpg",
                    "content_type": "image",
                },
            ],
            num_results=2,
            query="search query",
            query_type="text",
            search_errors=[],
        )

        assert output.num_results == 2
        assert len(output.results) == 2
        assert output.results[0]["content_type"] == "text"
        assert output.results[1]["content_type"] == "image"

    @patch(
        "world_understanding.tools.knowledge.multimodal_vector_store.find_similar_documents_func"
    )
    def test_successful_text_search(self, mock_find_func):
        """Test successful text similarity search."""
        # Mock search results
        mock_result1 = MagicMock()
        mock_result1.score = 0.95
        mock_result1.document.text_content = "Document text content"
        mock_result1.document.image_path = None
        mock_result1.document.get_content_type.return_value = "text"
        mock_result1.metadata = {"content_type": "text", "source": "doc1.txt"}

        mock_result2 = MagicMock()
        mock_result2.score = 0.82
        mock_result2.document.text_content = None
        mock_result2.document.image_path = "/path/to/img.jpg"
        mock_result2.document.get_content_type.return_value = "image"
        mock_result2.metadata = {"content_type": "image", "source": "img1.jpg"}

        mock_find_func.return_value = [mock_result1, mock_result2]

        # Create input
        inputs = FindSimilarDocumentsInput(
            query="machine learning",
            query_type="text",
            store_path="/path/to/store",
            k=5,
        )

        # Call tool
        output = find_similar_documents_tool(inputs)

        # Verify results
        assert output.num_results == 2
        assert len(output.results) == 2

        # Check first result (text)
        assert output.results[0]["score"] == 0.95
        assert output.results[0]["text_content"] == "Document text content"
        assert output.results[0]["content_type"] == "text"
        assert "image_path" not in output.results[0]

        # Check second result (image)
        assert output.results[1]["score"] == 0.82
        assert output.results[1]["image_path"] == "/path/to/img.jpg"
        assert output.results[1]["content_type"] == "image"
        assert "text_content" not in output.results[1]

        # Verify function was called correctly
        mock_find_func.assert_called_once_with(
            query="machine learning",
            query_type="text",
            store="/path/to/store",
            k=5,
            filter_metadata=None,
            embedding_type="image",
        )

    @patch(
        "world_understanding.tools.knowledge.multimodal_vector_store.find_similar_documents_func"
    )
    def test_image_query_with_caption(self, mock_find_func):
        """Test image query using caption-based embedding."""
        mock_find_func.return_value = []

        inputs = FindSimilarDocumentsInput(
            query="/path/to/query.jpg",
            query_type="image",
            store_path="/path/to/store",
            k=3,
            embedding_type="text",  # Use caption
        )

        output = find_similar_documents_tool(inputs)

        assert output.num_results == 0
        mock_find_func.assert_called_once_with(
            query="/path/to/query.jpg",
            query_type="image",
            store="/path/to/store",
            k=3,
            filter_metadata=None,
            embedding_type="text",
        )

    @patch(
        "world_understanding.tools.knowledge.multimodal_vector_store.find_similar_documents_func"
    )
    def test_file_not_found_error_search(self, mock_find_func):
        """Test handling of FileNotFoundError in search."""
        mock_find_func.side_effect = FileNotFoundError("Store not found")

        inputs = FindSimilarDocumentsInput(
            query="test query",
            query_type="text",
            store_path="/nonexistent/store",
        )

        output = find_similar_documents_tool(inputs)

        assert output.num_results == 0
        assert len(output.results) == 0
        assert len(output.search_errors) == 1
        assert "File not found" in output.search_errors[0]


def test_tool_registration():
    """Test that tools are properly registered."""
    from world_understanding.tools.base import get_tool_registry

    registry = get_tool_registry()

    # Check that our tools are registered
    assert "build_multimodal_vector_store" in registry
    assert "find_similar_documents" in registry

    # Verify tool specifications
    build_tool = registry["build_multimodal_vector_store"]
    assert build_tool.spec.version == "0.1.0"
    assert build_tool.spec.input_model == BuildMultimodalVectorStoreInput
    assert build_tool.spec.output_model == BuildMultimodalVectorStoreOutput

    find_tool = registry["find_similar_documents"]
    assert find_tool.spec.version == "0.1.0"
    assert find_tool.spec.input_model == FindSimilarDocumentsInput
    assert find_tool.spec.output_model == FindSimilarDocumentsOutput
