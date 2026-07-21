# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This module handles preprocessing of collected data into structured formats using NVIDIA's nv_ingest framework.
"""

import base64
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Type checking imports - these are only imported for type hints
if TYPE_CHECKING:  # pragma: no cover - static type checker only.
    pass


class DataPreprocessor:
    """Preprocesses collected data into structured formats."""

    # Constants
    DEFAULT_TIMEOUT = 3600
    DEFAULT_PORT = 7671
    DEFAULT_HOSTNAME = "localhost"
    DEFAULT_EXTRACT_METHOD = "nemoretriever_parse"
    DEFAULT_TEXT_DEPTH = "page"

    # Class-level variables to track initialization
    _initialization_lock = threading.Lock()  # Initialize lock at class definition time
    _nv_ingest_client = None

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the data preprocessor.

        Args:
            config: Configuration dictionary containing preprocessing parameters.
                   Can include:
                   - 'extract_text': extract text (default: True)
                   - 'extract_tables': extract tables (default: True)
                   - 'extract_charts': extract charts (default: True)
                   - 'extract_images': extract images (default: True)
                   - 'extract_infographics': extract infographics (default: True)
                   - 'extract_method': extract method (default: "nemoretriever_parse")
                   - 'text_depth': text depth (default: "page")
                   - 'timeout': timeout in seconds (default: 3600)
                   - 'show_progress': show progress (default: True)
        """
        self.config = config or {}

        # Validate configuration
        self._validate_config()

        # Initialize nv_ingest components (only once per program)
        self._ensure_nv_ingest_initialized()

    def _validate_config(self):
        """Validate configuration parameters."""
        try:
            # Lazy import for validation
            from nv_ingest_client.primitives.tasks.extract import ExtractTaskSchema

            extract_method = self.config.get(
                "extract_method", self.DEFAULT_EXTRACT_METHOD
            )
            text_depth = self.config.get("text_depth", self.DEFAULT_TEXT_DEPTH)
            # Use a default document type for validation (PDF is commonly supported)
            ExtractTaskSchema(
                document_type="pdf",
                extract_method=extract_method,
                text_depth=text_depth,
            )
        except ImportError:
            logger.warning("nv_ingest is not installed, skipping validation")
        except Exception as e:
            logger.error("Failed to validate configuration: %s", e)
            raise RuntimeError("Invalid DataPreprocessor configuration") from e

    def _ensure_nv_ingest_initialized(self):
        """Ensure nv_ingest is initialized only once per program."""
        if DataPreprocessor._nv_ingest_client is None:
            try:
                with DataPreprocessor._initialization_lock:
                    # Double-check pattern to ensure thread safety
                    if DataPreprocessor._nv_ingest_client is None:
                        self._setup_nv_ingest()
            except Exception as e:
                # Ensure lock is released even if setup fails
                logger.error("Failed to initialize nv_ingest: %s", e)
                # Reset state to allow retry
                DataPreprocessor._nv_ingest_client = None
                raise  # Re-raise to let caller handle the error

        # Use class-level nv_ingest components
        self.nv_ingest_client = DataPreprocessor._nv_ingest_client

    def _setup_nv_ingest(self):
        """Set up nv_ingest pipeline and client."""
        try:
            # Lazy imports - only import when actually setting up nv_ingest
            from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
                PipelineCreationSchema,
                run_pipeline,
            )
            from nv_ingest_api.util.message_brokers.simple_message_broker import (
                SimpleClient,
            )
            from nv_ingest_client.client import NvIngestClient

            # Check for required API key
            api_key = os.getenv("NVIDIA_API_KEY")
            if api_key is None:
                raise OSError(
                    "NVIDIA_API_KEY environment variable is required for nv_ingest."
                )

            # Set environment variables for nv_ingest
            os.environ["NVIDIA_BUILD_API_KEY"] = api_key

            # Read environment variables for nv_ingest
            pipeline_config = PipelineCreationSchema()

            # Start the pipeline subprocess for library mode
            run_pipeline(
                pipeline_config,
                block=False,
                disable_dynamic_scaling=True,
                run_in_subprocess=True,
            )

            # Create nv_ingest client
            DataPreprocessor._nv_ingest_client = NvIngestClient(
                message_client_allocator=SimpleClient,
                message_client_port=self.DEFAULT_PORT,
                message_client_hostname=self.DEFAULT_HOSTNAME,
                message_client_kwargs={
                    "connection_timeout": self.config.get(
                        "timeout", self.DEFAULT_TIMEOUT
                    )
                },
            )

            logger.info("nv_ingest pipeline and client initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize nv_ingest: %s", e)
            DataPreprocessor._nv_ingest_client = None
            raise RuntimeError("nv_ingest setup failed") from e

    def cleanup(self):
        """Clean up nv_ingest resources."""
        if DataPreprocessor._nv_ingest_client:
            try:
                # Close client connection if it has a close method
                if hasattr(DataPreprocessor._nv_ingest_client, "close"):
                    DataPreprocessor._nv_ingest_client.close()
                DataPreprocessor._nv_ingest_client = None
                logger.info("nv_ingest client cleaned up successfully")
            except Exception as e:
                logger.error("Error during cleanup: %s", e)

    @classmethod
    def _validate_file_paths(cls, file_paths: list[str]):
        """Validate that all file paths exist and are readable."""
        try:
            for file_path in file_paths:
                file_path = Path(file_path)
                if not file_path.is_file():
                    raise FileNotFoundError(
                        f"{file_path} does not exist or is not a file"
                    )
                if not os.access(file_path, os.R_OK):
                    raise PermissionError(f"{file_path} is not readable")

                # Get file extension and convert to document type
                doc_type = file_path.suffix[1:] if file_path.suffix else ""

                if not doc_type:
                    raise ValueError(f"No document type found for file {file_path}")

                # Lazy import and use ExtractTaskSchema to validate the document type
                try:
                    from nv_ingest_client.primitives.tasks.extract import (
                        ExtractTaskSchema,
                    )
                except ImportError:
                    logger.warning(
                        "nv_ingest is not installed, skipping document type validation"
                    )
                    continue

                ExtractTaskSchema(document_type=doc_type)

        except ValueError as e:
            raise ValueError(f"Invalid file path: {file_path}") from e
        except Exception as e:
            logger.warning(
                "Error checking document type support for %s: %s", file_path, e
            )
            raise ValueError(
                f"Error checking document type support for {file_path}"
            ) from e

    def preprocess_documents(
        self,
        file_paths: list[str],
        save_content_only: bool = True,
        batch_size: int = 32,
        max_retries: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """Preprocess multiple documents using nv_ingest batch processing.

        Args:
            file_paths: List of document file paths: ["path/to/file.pdf", ...]
            save_content_only: If True, only save the content of the document to a file.
                If False, save the entire document to a file.
            batch_size: Batch size for processing the files.
            max_retries: Maximum number of retries for processing the files.

        Returns:
            Dictionary mapping file paths to lists of processing results.
            Each result contains extracted document_type and metadata.
            Format: {"path/to/file.pdf": [{"document_type": "...", "metadata": {...}}, ...]}
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        # Validate file paths
        DataPreprocessor._validate_file_paths(file_paths)

        if self.nv_ingest_client is None:
            raise RuntimeError("No nv_ingest client available")

        # Lazy import Ingestor when actually using it
        from nv_ingest_client.client import Ingestor

        # Use nv_ingest for advanced batch processing
        start_time = time.time()

        results = []
        for i in range(max_retries):
            logger.info(
                "Processing files in batch (try %d/%d): %s",
                i + 1,
                max_retries,
                file_paths,
            )
            ingestor = (
                Ingestor(client=self.nv_ingest_client)
                .files(file_paths)
                .extract(
                    extract_text=self.config.get("extract_text", True),
                    extract_tables=self.config.get("extract_tables", True),
                    extract_charts=self.config.get("extract_charts", True),
                    extract_images=self.config.get("extract_images", True),
                    # paddle_output_format="markdown",
                    extract_infographics=self.config.get("extract_infographics", True),
                    extract_method=self.config.get(
                        "extract_method", self.DEFAULT_EXTRACT_METHOD
                    ),  # Slower, but maximally accurate, especially for PDFs with pages that are scanned images
                    text_depth=self.config.get("text_depth", self.DEFAULT_TEXT_DEPTH),
                )
            )

            output: tuple[list[dict[str, Any]], list[tuple[str, str]]] = (
                ingestor.ingest(
                    show_progress=self.config.get("show_progress", True),
                    return_failures=True,
                    timeout=self.config.get("timeout", self.DEFAULT_TIMEOUT),
                    batch_size=batch_size,
                )
            )
            batch_results, failures = output
            results.extend(batch_results)

            if len(failures) == 0:
                break

            if i == max_retries - 1:
                raise RuntimeError("nv_ingest returned failures for batch processing")

            file_paths = [file_path.split(":")[1] for file_path, _ in failures]

        processing_time = time.time() - start_time

        if not results or len(results) == 0:
            raise RuntimeError("nv_ingest returned no results for batch processing")

        # Add preprocessing results to each document
        processed_documents = {}

        # order of results is not guaranteed, so we need to check the file path
        for file_path in file_paths:
            # Create a copy of the document to avoid modifying the original
            found = False
            for result in results:
                if len(result) == 0:
                    raise RuntimeError(f"No result found for document {file_path}")

                if result[0]["metadata"]["source_metadata"]["source_name"] == file_path:
                    if save_content_only:
                        contents = []
                        for doc in result:
                            document_type, content = (
                                DataPreprocessor.get_document_content(doc)
                            )
                            contents.append(
                                {"document_type": document_type, "content": content}
                            )
                        processed_documents[file_path] = contents
                    else:
                        processed_documents[file_path] = result
                    found = True
                    break

            if not found:
                raise RuntimeError(f"No result found for document {file_path}")

        logger.info(
            "Successfully processed %d documents with nv_ingest in %.2f seconds",
            len(file_paths),
            processing_time,
        )

        return processed_documents

    @classmethod
    def get_document_content(cls, document: dict[str, Any]) -> tuple[str, str]:
        """Preview a document."""
        document_type = document.get("document_type", "Unknown")

        metadata = document["metadata"]
        content = ""
        if "structured" in document_type:
            content = metadata["table_metadata"]["table_content"]
        elif "text" in document_type:
            content = metadata["content"]
        elif "image" in document_type:
            # base64 image
            content = metadata["content"]
            # caption or text
            # content = metadata["image_metadata"]["caption"]
        elif "audio" in document_type:
            content = metadata["audio_metadata"]["audio_transcript"]
        else:
            raise NotImplementedError(f"Unknown document type: {document_type}")

        return document_type, content


def extract_document_content(
    source: str | Path | list[str | Path],
    save_content_only: bool = True,
    batch_size: int = 32,
    max_retries: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Extract content from documents using nv_ingest.

    Args:
        source: Source of documents - can be:
            - str/Path: Directory to scan for documents
            - list: List of document file paths
        save_content_only: If True, only save the content of the document to a file.
            If False, save the entire document to a file.
        batch_size: Batch size for processing the files.
        max_retries: Maximum number of retries for processing the files.

    Returns:
        Dictionary mapping file paths to lists of processing results.
        Each result contains extracted content and metadata.

    Raises:
        ValueError: If no documents found or validation fails
        FileNotFoundError: If any file doesn't exist
        PermissionError: If any file is not readable
        RuntimeError: If nv_ingest processing fails
    """
    # Collect all document sources
    file_paths: list[str] = []

    # Lazy import _DEFAULT_EXTRACTOR_MAP when needed
    try:
        from nv_ingest_client.primitives.tasks.extract import _DEFAULT_EXTRACTOR_MAP
    except ImportError as e:
        raise RuntimeError(
            "nv_ingest is required for document processing but is not installed"
        ) from e

    if isinstance(source, str | Path):
        # Directory path - scan for documents
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")

        if source_path.is_file():
            # Single file
            file_paths = [str(source_path)]
        elif source_path.is_dir():
            # Directory - scan for documents

            # Common document extensions
            file_paths = list(source_path.rglob("*"))
            file_paths = [
                str(p)
                for p in file_paths
                if p.suffix.lower()[1:]
                in _DEFAULT_EXTRACTOR_MAP.keys()  # remove the dot
            ]
        else:
            raise ValueError(f"Invalid source path: {source}")
    elif isinstance(source, list):
        # List of document paths
        file_paths = [
            str(p)
            for p in source
            if Path(p).suffix.lower()[1:]
            in _DEFAULT_EXTRACTOR_MAP.keys()  # remove the dot
        ]
    else:
        raise ValueError(
            f"Invalid source type: {type(source)}. "
            "Expected directory path, file path, or list of document paths."
        )

    logger.debug("Processing file paths: %s", file_paths)

    if not file_paths:
        raise ValueError("No document files found in the provided source")

    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_documents(
        file_paths,
        save_content_only=save_content_only,
        batch_size=batch_size,
        max_retries=max_retries,
    )


def split_document_content_by_type(
    input_file_path: str, output_dir: str
) -> dict[str, list[str]]:
    """Split extracted document content by type for multimodal vector store.
    Assume that the input file is a json file of the content only output from
    extract_document_content function.

    Args:
        input_file_path: Output from extract_document_content function
        output_dir: Directory to save split content files

    Returns:
        Dictionary mapping document file names to lists of file paths created
    """
    input_file_path = Path(input_file_path)
    try:
        if not input_file_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        if not input_file_path.is_file():
            raise ValueError(f"Input file is not a file: {input_file_path}")

        with open(input_file_path, encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        raise ValueError(f"Invalid input file: {e}") from e

    output_dir = Path(output_dir)

    created_files = {}

    for doc_file_path, documents in data.items():
        doc_file_path = Path(doc_file_path)
        doc_file_name = doc_file_path.stem
        created_files[doc_file_name] = []
        for i, doc in enumerate(documents):
            document_type = doc["document_type"]
            content = doc["content"]
            output_path = (
                output_dir
                / doc_file_name
                / f"{doc_file_name}_{i:04d}_{document_type}.xxx"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if document_type == "structured" or document_type == "text":
                output_path = output_path.with_suffix(".txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                created_files[doc_file_name].append(output_path)
            elif document_type == "image":
                output_path = output_path.with_suffix(".png")
                try:
                    with open(output_path, "wb") as f:
                        f.write(base64.b64decode(content))
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {e}")
                    continue

                created_files[doc_file_name].append(output_path)

    return created_files
