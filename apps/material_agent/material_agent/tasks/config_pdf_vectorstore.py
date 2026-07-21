# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for PDF vectorstore workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import log_config_source
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from material_agent.tasks.config_loader import load_config_from_context

logger = logging.getLogger(__name__)


class PDFVectorstoreConfigTask(Task):
    """Compatibility config task for PDF vectorstore workflows."""

    def __init__(self):
        """Initialize the PDF vectorstore config loading task."""
        self.name = "PDFVectorstoreConfigLoading"
        self.description = "Load PDF vectorstore configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load PDF vectorstore configuration.

        Args:
            context: Workflow context containing config_path or config_dict
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(
            context,
            missing_path_message=(
                "Either config_path or config_dict must be provided in context"
            ),
        )
        log_config_source(context, listener.info, label="PDF vectorstore")
        config_dir = config_path.parent

        # Handle source path with override support
        source = context.get("source_override") or config.get("source")
        if not source:
            raise ValueError("source not specified in config or override")

        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = config_dir / source_path

        # Handle output directory with override support
        output_dir = context.get("output_dir_override") or config.get("output_dir")
        if not output_dir:
            raise ValueError("output_dir not specified in config or override")

        output_dir_path = Path(output_dir)
        if not output_dir_path.is_absolute():
            output_dir_path = config_dir / output_dir_path

        # Set up directory structure for pipeline stages
        extraction_output_dir = output_dir_path / "extracted"
        split_output_dir = output_dir_path / "split"
        vectorstore_save_path = output_dir_path / "vectorstore"
        extracted_content_path = extraction_output_dir / "document_content.json"

        # Set context keys expected by the workflow
        context["config"] = config
        context["source_path"] = str(source_path)
        context["output_dir"] = str(output_dir_path)
        context["extraction_output_dir"] = str(extraction_output_dir)
        context["split_output_dir"] = str(split_output_dir)
        context["vectorstore_save_path"] = str(vectorstore_save_path)
        context["extracted_content_path"] = str(extracted_content_path)
        context["save_content_only"] = True  # Save only the content JSON file
        context["embedding_config"] = config.get("embedding", {})
        context["chunk_size"] = config.get("chunk_size", 512)
        context["chunk_overlap"] = config.get("chunk_overlap", 50)
        context["image_embedding_type"] = config.get("image_embedding_type", "text")
        context["include_filename_metadata"] = config.get(
            "include_filename_metadata", True
        )

        # Construct embedding model string if config is provided
        embedding_config = context["embedding_config"]
        if embedding_config:
            service = embedding_config.get("service", "nim")
            model = embedding_config.get(
                "model", "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
            )
            context["embedding_model"] = f"{service}/{model}"
        else:
            context["embedding_model"] = None

        return context
