# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for pipeline workflows."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import (
    load_config_mapping_from_context,
    log_config_source,
)
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)

from material_agent.api.defaults import PIPELINE_STEP_NAMES
from material_agent.materials import (
    material_entries_with_fallback,
    material_mapping_with_fallback,
)
from material_agent.prompt_security import format_material_names_for_prompt
from material_agent.tasks.config_loader import load_config_from_context
from material_agent.tasks.prepare_dataset import (
    render_system_prompt_from_prepare_config,
)

logger = logging.getLogger(__name__)


class PipelineConfigTask(Task):
    """Task to load and validate configuration for pipeline workflows.

    This task loads a pipeline configuration dictionary (or standalone YAML) and
    validates its structure.
    The pipeline config can contain sections for each step: build_dataset_usd,
    build_dataset_pdf_vectorstore, build_dataset_prepare_dataset, predict/benchmark,
    apply, refine.

    Available steps:
    - build_dataset_usd: Prepare USD data with rendering
    - build_dataset_prepare_dataset: Create dataset from rendered USD
    - predict/benchmark: VLM-based material prediction
    - apply: Apply predicted materials to USD (single pass)
    - refine: Iterative material assignment with VLM judge refinement

    Each step section can either:
    1. Reference an external config file via 'config' key
    2. Provide inline configuration directly

    Path Resolution:
        - All relative paths in config file are treated as relative to the config file location
        - working_dir in pipeline section overrides the base directory for path resolution
        - Absolute paths are used as-is

    Input context keys:
        - config_dict: In-memory pipeline configuration (preferred)
        - config_path: YAML path or relative-path anchor
        - skip_steps: Optional list of step names to skip
        - only_steps: Optional list of step names to run exclusively

    Output context keys:
        - pipeline_config: Parsed and validated pipeline configuration
        - pipeline_name: Name of the pipeline
        - pipeline_description: Description of the pipeline
        - working_dir: Working directory for relative path resolution
        - steps_to_run: Ordered list of steps to execute
        - step_configs: Dictionary of resolved configurations for each step
        - keep_temp_files: Whether to preserve temporary files after completion
    """

    # Use centralized step names
    VALID_STEPS = PIPELINE_STEP_NAMES

    def __init__(self):
        """Initialize the pipeline config loading task."""
        self.name = "PipelineConfigLoading"
        self.description = "Load and validate pipeline configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load and validate pipeline configuration.

        Args:
            context: Workflow context containing config_dict or config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If configuration file not found
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(
            context,
            missing_file_message="Pipeline configuration file not found: {config_path}",
            empty_message="Pipeline configuration file is empty",
        )
        log_config_source(context, listener.info, label="pipeline")

        # Extract pipeline metadata
        pipeline_meta = config.get("pipeline", {})
        pipeline_name = pipeline_meta.get("name", "unnamed_pipeline")
        pipeline_description = pipeline_meta.get("description", "")

        # Determine working directory for path resolution
        working_dir = pipeline_meta.get("working_dir", ".")
        working_dir = Path(working_dir)
        if not working_dir.is_absolute():
            working_dir = config_path.parent / working_dir
        working_dir = resolve_path_with_safe_diagnostics(
            working_dir,
            label="pipeline working directory",
        )

        # Check if temporary files should be preserved (default: True)
        keep_temp_files = pipeline_meta.get("keep_temp_files", True)

        listener.info(f"Pipeline: {redact_sensitive_config(pipeline_name)}")
        if pipeline_description:
            listener.info(
                f"Description: {redact_sensitive_config(pipeline_description)}"
            )
        listener.info(f"Working directory: {redact_sensitive_path(working_dir)}")
        if keep_temp_files:
            listener.info("Temporary files will be preserved after completion")

        # Parse unified materials section
        materials_data = self._parse_materials(config, config_path)
        if materials_data:
            listener.info(
                f"Loaded {len(materials_data['entries'])} materials from unified definition"
            )
            if materials_data.get("library_path"):
                listener.info(
                    "  Material library: "
                    f"{redact_sensitive_path(materials_data['library_path'])}"
                )

        # Validate and extract step configurations
        steps_to_run, step_configs = self._process_steps(
            config, config_path, working_dir, context, materials_data, listener
        )

        if not steps_to_run:
            raise ValueError("No valid steps found in pipeline configuration")

        listener.info(f"Steps to execute: {', '.join(steps_to_run)}")

        # Update context
        context["pipeline_config"] = config
        context["pipeline_name"] = pipeline_name
        context["pipeline_description"] = pipeline_description
        context["working_dir"] = working_dir
        context["steps_to_run"] = steps_to_run
        context["step_configs"] = step_configs
        context["materials_data"] = materials_data  # Store for use by steps
        context["keep_temp_files"] = keep_temp_files

        return context

    def _parse_materials(
        self, config: dict[str, Any], config_path: Path
    ) -> dict[str, Any] | None:
        """Parse unified materials section from pipeline config.

        Args:
            config: Full pipeline configuration
            config_path: Path to the pipeline config file

        Returns:
            Parsed materials data or None if not present
        """
        materials_section = config.get("materials")
        if not materials_section:
            return None

        if not isinstance(materials_section, dict):
            raise ValueError("'materials' section must be a dictionary")

        # Parse library path (optional)
        library_path = materials_section.get("library_path")
        if library_path:
            library_path = Path(library_path)
            # Resolve relative to config file location
            if not library_path.is_absolute():
                library_path = config_path.parent / library_path
            library_path = str(
                resolve_path_with_safe_diagnostics(
                    library_path,
                    label="material library path",
                )
            )

        # Parse material entries
        entries = materials_section.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("'materials.entries' must be a list")

        parsed_entries = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Material entry {i} must be a dictionary")

            name = entry.get("name")
            description = entry.get("description", "")
            binding = entry.get("binding", "")

            if not name:
                raise ValueError(f"Material entry {i} missing 'name' field")

            parsed_entries.append(
                {
                    "name": name,
                    "description": description,
                    "binding": binding,
                    **{
                        str(key): value
                        for key, value in entry.items()
                        if key not in {"name", "description", "binding"}
                    },
                }
            )

        parsed_materials = {
            "library_path": library_path,
            "entries": parsed_entries,
        }
        simready = materials_section.get("simready")
        if isinstance(simready, dict):
            parsed_materials["simready"] = dict(simready)
        return parsed_materials

    def _process_steps(
        self,
        config: dict[str, Any],
        config_path: Path,
        working_dir: Path,
        context: dict[str, Any],
        materials_data: dict[str, Any] | None = None,
        listener: Any = None,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Process and resolve step configurations.

        Args:
            config: Full pipeline configuration
            config_path: Path to the pipeline config file
            working_dir: Working directory for relative paths
            context: Workflow context
            materials_data: Optional materials data from pipeline config
            listener: Event listener for logging

        Returns:
            Tuple of (steps_to_run, step_configs)
        """
        skip_steps = set(context.get("skip_steps", []))
        only_steps = context.get("only_steps", [])

        steps_to_run = []
        step_configs = {}

        # Process steps in order
        for step_name in self.VALID_STEPS:
            # Skip if not in config
            if step_name not in config:
                continue

            # Handle mutually exclusive predict/benchmark
            if step_name == "predict" and "benchmark" in config:
                listener.warning(
                    "Both 'predict' and 'benchmark' found. Using 'predict'."
                )
            elif step_name == "benchmark" and "predict" in steps_to_run:
                listener.info("Skipping 'benchmark' as 'predict' is already configured")
                continue

            # Apply skip/only filters
            if skip_steps and step_name in skip_steps:
                listener.info(f"Skipping step: {step_name} (--skip)")
                continue

            if only_steps and step_name not in only_steps:
                listener.debug(f"Skipping step: {step_name} (not in --only)")
                continue

            # Get step configuration
            step_config = config[step_name]
            if not isinstance(step_config, dict):
                raise ValueError(f"Step '{step_name}' must be a dictionary")

            # Resolve configuration (external reference or inline)
            resolved_config = self._resolve_step_config(
                step_name, step_config, config_path, working_dir, listener
            )

            # Inject materials data into specific steps
            if materials_data:
                resolved_config = self._inject_materials_into_step(
                    step_name, resolved_config, materials_data, listener
                )

            if (
                step_name in ("predict", "benchmark")
                and "system_prompt" not in resolved_config
                and "build_dataset_prepare_dataset" in step_configs
            ):
                resolved_config["system_prompt"] = (
                    render_system_prompt_from_prepare_config(
                        step_configs["build_dataset_prepare_dataset"]
                    )
                )

            steps_to_run.append(step_name)
            step_configs[step_name] = resolved_config

        return steps_to_run, step_configs

    def _resolve_step_config(
        self,
        step_name: str,
        step_config: dict[str, Any],
        config_path: Path,
        working_dir: Path,
        listener: Any = None,
    ) -> dict[str, Any]:
        """Resolve step configuration from external file or inline.

        Args:
            step_name: Name of the step
            step_config: Step configuration section
            config_path: Path to the pipeline config file
            working_dir: Working directory for relative paths
            listener: Event listener for logging

        Returns:
            Resolved configuration dictionary
        """
        # If 'config' key exists, load external config file
        if "config" in step_config:
            external_config_path = step_config["config"]
            external_config_path = Path(external_config_path)

            # Resolve relative to pipeline config file (not working_dir)
            if not external_config_path.is_absolute():
                external_config_path = config_path.parent / external_config_path

            listener.info(f"Loading external config for {step_name}")

            resolved_config, _ = load_config_mapping_from_context(
                {"config_path": external_config_path},
                missing_file_message=(
                    f"External config for step '{step_name}' not found: {{config_path}}"
                ),
                parse_error_message=(
                    f"Unable to parse external config for step '{step_name}': "
                    "{config_path}"
                ),
                empty_message=f"External config for '{step_name}' is empty",
                file_non_mapping_message=(
                    f"External config for '{step_name}' must contain a mapping, "
                    "got {type_name}"
                ),
            )

            # Store reference to external config path for path resolution
            resolved_config["_external_config_path"] = external_config_path

            return resolved_config

        # Otherwise, use inline configuration
        listener.info(f"Using inline config for {step_name}")

        # Make a copy to avoid modifying original
        resolved_config = dict(step_config)

        # Mark as inline config
        resolved_config["_inline_config"] = True
        resolved_config["_pipeline_config_path"] = config_path

        return resolved_config

    def _inject_materials_into_step(
        self,
        step_name: str,
        step_config: dict[str, Any],
        materials_data: dict[str, Any],
        listener: Any = None,
    ) -> dict[str, Any]:
        """Inject materials data into step configuration.

        Args:
            step_name: Name of the step
            step_config: Step configuration
            materials_data: Parsed materials data from pipeline config
            listener: Event listener for logging

        Returns:
            Updated step configuration with materials injected
        """
        # For build_dataset_prepare_dataset: inject materials_list
        if step_name == "build_dataset_prepare_dataset":
            entries = material_entries_with_fallback(materials_data["entries"])
            injected_materials_list = False
            if "materials_list" not in step_config:
                # Extract material names for prompt and dataset metadata.
                materials_list = [entry["name"] for entry in entries]
                step_config["materials_list"] = materials_list
                injected_materials_list = True
                listener.debug(
                    f"Injected {len(materials_list)} materials into {step_name}"
                )

            # Keep the prompt payload and persisted material_names sourced from the
            # same list. Explicit materials_list values are formatted later by the
            # prepare task; only synthesize the structured payload for an injected
            # list.
            if (
                "prompts" in step_config
                and injected_materials_list
                and "_materials_formatted" not in step_config
            ):
                materials_formatted = self._format_materials_for_prompt(entries)
                step_config["_materials_formatted"] = materials_formatted

        # For validate/harmonize: inject material_names
        elif step_name in ("validate_predictions", "harmonize_predictions"):
            if "material_names" not in step_config:
                entries = material_entries_with_fallback(materials_data["entries"])
                step_config["material_names"] = [entry["name"] for entry in entries]
                listener.debug(
                    f"Injected {len(step_config['material_names'])} material_names into {step_name}"
                )

        # For apply and refine: inject materials_mapping
        elif step_name in ["apply", "refine"]:
            # Build materials_mapping from entries
            materials_mapping = {}

            # Add library path if present
            if materials_data.get("library_path"):
                materials_mapping["material_library_path"] = materials_data[
                    "library_path"
                ]

            # Add name -> binding mappings
            for entry in material_entries_with_fallback(materials_data["entries"]):
                materials_mapping[entry["name"]] = entry["binding"]
            materials_mapping = material_mapping_with_fallback(materials_mapping)

            # For refine step, inject into the 'apply' subsection
            if step_name == "refine":
                if "apply" not in step_config:
                    step_config["apply"] = {}
                if "materials_mapping" not in step_config["apply"]:
                    step_config["apply"]["materials_mapping"] = materials_mapping
                    listener.debug(
                        f"Injected materials_mapping with {len(materials_data['entries'])} entries into refine.apply"
                    )
            # For apply step, inject at top level
            elif step_name == "apply":
                if "materials_mapping" not in step_config:
                    step_config["materials_mapping"] = materials_mapping
                    listener.debug(
                        f"Injected materials_mapping with {len(materials_data['entries'])} entries into {step_name}"
                    )

        return step_config

    def _format_materials_for_prompt(self, entries: list[dict[str, Any]]) -> str:
        """Format material names as untrusted prompt data.

        Args:
            entries: List of material entries

        Returns:
            Formatted string ready for {materials_list} substitution
        """
        return format_material_names_for_prompt(entries)
