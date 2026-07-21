# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path resolution service for the unified config system.

This module provides centralized path resolution based on the unified config.
All paths are derived from project.working_dir, input.usd_path, and output.usd_path.

Session ID Support:
- Uses world_understanding.agentic.session.SessionManager for session management
- If session_id is provided, working_dir defaults to .{session_id}
- This allows automatic path management without manual configuration
- Sessions can be resumed by providing the same session_id
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import BasePathResolver
from world_understanding.utils.credentials import redact_sensitive_path

from material_agent.config.schema import STEP_OUTPUT_DIRS

logger = logging.getLogger(__name__)


class ProjectPathResolver(BasePathResolver):
    """Centralized path resolution for the entire pipeline.

    This class handles all path resolution based on the unified config structure.
    It enforces the convention that all intermediate files go into the working
    directory, with predictable subdirectory structure.

    All paths are resolved relative to the config file location unless they are
    absolute paths.
    """

    def __init__(self, config: dict[str, Any], config_file_path: Path):
        """Initialize the path resolver.

        Args:
            config: Unified configuration dictionary
            config_file_path: Path to the configuration file (for relative path resolution)
        """
        # Initialize base class (handles session management, working_dir, etc.)
        super().__init__(
            config=config,
            config_file_path=config_file_path,
            default_project_name="material_agent_project",
        )

        # Material-agent specific: Resolve input paths (relative to config directory)
        input_config = config.get("input") or {}
        self.input_usd: Path | None = self._resolve_path(input_config.get("usd_path"))
        self.prim_path: str | None = input_config.get("prim_path")

        reference_images = input_config.get("reference_images", [])
        self.reference_images: list[Path] = [
            resolved
            for img in reference_images
            if img and (resolved := self._resolve_path(img)) is not None
        ]

        # Resolve reference PDFs (relative to config directory)
        reference_pdfs = input_config.get("reference_pdfs", [])
        self.reference_pdfs: list[Path] = [
            resolved
            for pdf in reference_pdfs
            if pdf and (resolved := self._resolve_path(pdf)) is not None
        ]

        # Output configuration - only for options, NOT paths
        output_config = config.get("output") or {}
        self.layer_only = output_config.get("layer_only", False)
        self.flatten_output = output_config.get("flatten_output", True)
        self.material_profile = (
            output_config.get("material_profile")
            or output_config.get("shader_target")
            or output_config.get("material_authoring_target")
            or "auto"
        )

        # Output paths are now DERIVED from session structure
        # Structure: .{session_id}/output/
        output_dir = self.get_output_dir()
        self.output_usd: Path = output_dir / "output.usd"

        # Legacy support: if output.usd_path is explicitly provided and not None, use it
        if "usd_path" in output_config and output_config["usd_path"] is not None:
            logger.warning(
                "output.usd_path is deprecated. Paths are now auto-derived from session_id. "
                "Using provided path for backward compatibility."
            )
            resolved_output = self._resolve_path(output_config["usd_path"])
            if resolved_output is not None:
                self.output_usd = resolved_output

        logger.info(
            "Input USD: %s",
            (
                redact_sensitive_path(self.input_usd)
                if self.input_usd is not None
                else None
            ),
        )
        logger.info("Output USD: %s", redact_sensitive_path(self.output_usd))

    # Material-agent specific path methods

    def get_step_output_dir(self, step_name: str) -> Path:
        """Get the output directory for a specific step.

        All step outputs go into predictable subdirectories of working_dir.

        Args:
            step_name: Name of the step

        Returns:
            Path to the step's output directory
        """
        subdir = STEP_OUTPUT_DIRS.get(step_name, step_name)
        return self.working_dir / subdir

    def get_step_dataset_file(self, step_name: str) -> Path:
        """Get the dataset file path for a step.

        Args:
            step_name: Name of the step

        Returns:
            Path to the step's dataset file
        """
        if step_name == "build_dataset_prepare_dataset":
            return self.get_step_output_dir(step_name) / "dataset.jsonl"
        else:
            return self.get_step_output_dir(step_name) / "dataset.jsonl"

    def get_step_predictions_file(self, step_name: str = "predict") -> Path:
        """Get the predictions file path.

        Args:
            step_name: Name of the prediction step (predict or benchmark)

        Returns:
            Path to the predictions file
        """
        return self.get_step_output_dir(step_name) / "predictions.jsonl"

    def get_usd_dataset_dir(self) -> Path:
        """Get the USD dataset directory (build_dataset_usd output).

        Returns:
            Path to the USD dataset directory
        """
        return self.get_step_output_dir("build_dataset_usd")

    def get_vectorstore_dir(self) -> Path:
        """Get the vectorstore directory.

        Returns:
            Path to the vectorstore directory
        """
        return self.get_step_output_dir("build_dataset_pdf_vectorstore")

    def get_dataset_dir(self) -> Path:
        """Get the prepared dataset directory.

        Returns:
            Path to the prepared dataset directory
        """
        return self.get_step_output_dir("build_dataset_prepare_dataset")

    def get_predictions_dir(self) -> Path:
        """Get the predictions directory.

        Returns:
            Path to the predictions directory
        """
        return self.get_step_output_dir("predict")

    def create_working_directories(self) -> None:
        """Create all necessary working directories.

        This creates the working_dir and its standard subdirectories.
        """
        # Call base class to create working_dir, output, and temp
        super().create_working_directories()

    def validate_input_paths(self) -> None:
        """Validate that required input paths exist.

        Raises:
            FileNotFoundError: If required input files don't exist
        """
        if not self.input_usd or not self._path_exists_for_validation(
            self.input_usd, label="input USD"
        ):
            raise FileNotFoundError(
                f"Input USD file not found: {redact_sensitive_path(self.input_usd)}"
            )

        for img in self.reference_images:
            if not self._path_exists_for_validation(img, label="reference image"):
                logger.warning(
                    "Reference image not found: %s",
                    redact_sensitive_path(img),
                )

    def get_path_summary(self) -> dict[str, Any]:
        """Get a summary of all resolved paths.

        Returns:
            Dictionary with path information
        """
        # Get base class summary
        summary = super().get_path_summary()

        # Add material-agent specific paths
        summary.update(
            {
                "input": {
                    "usd_path": (
                        redact_sensitive_path(self.input_usd)
                        if self.input_usd
                        else None
                    ),
                    "reference_images": [
                        redact_sensitive_path(img) for img in self.reference_images
                    ],
                    "reference_pdfs": [
                        redact_sensitive_path(pdf) for pdf in self.reference_pdfs
                    ],
                },
                "output": {
                    "usd_path": (
                        redact_sensitive_path(self.output_usd)
                        if self.output_usd
                        else None
                    ),
                    "layer_only": self.layer_only,
                    "flatten_output": self.flatten_output,
                    "material_profile": self.material_profile,
                },
                "step_outputs": {
                    step: redact_sensitive_path(self.get_step_output_dir(step))
                    for step in STEP_OUTPUT_DIRS.keys()
                },
            }
        )

        return summary
