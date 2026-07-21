# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for loading generated-material-library configuration from YAML."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)

from material_agent.tasks.config_loader import load_config_from_context

logger = logging.getLogger(__name__)


class GenerateMaterialLibraryConfigTask(Task):
    """Load generate-material-library configuration.

    The executable path is driven by either:
      - ``material_generation_plan_path``: path to a YAML plan, or
      - ``material_generation_plan``: inline plan data.

    Output context keys are consumed by ``GenerateMaterialLibraryTask``.
    """

    def __init__(self) -> None:
        self.name = "GenerateMaterialLibraryConfig"
        self.description = "Load generated-material-library configuration"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(
            context,
            missing_path_message="config_path is required in context",
            non_mapping_message=(
                "generate material library config must be a mapping, got {type_name}"
            ),
            allow_empty=True,
        )
        if context.get("config_dict") is not None:
            listener.info("Using in-memory generate-material-library config")
        else:
            listener.info("Loading generate-material-library config from file")

        output_dir = config.get("output_dir")
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.is_absolute():
                output_dir = resolve_path_with_safe_diagnostics(
                    config_path.parent / output_dir,
                    label="generated-material output directory",
                )
        else:
            output_dir = config_path.parent / "generated_material_library"
        create_directory_with_safe_diagnostics(
            output_dir,
            label="generated-material output directory",
        )

        context["output_dir"] = str(output_dir)

        input_usd_path = config.get("input_usd_path") or config.get("usd_path")
        if input_usd_path:
            input_usd_path = Path(input_usd_path)
            if not input_usd_path.is_absolute():
                input_usd_path = resolve_path_with_safe_diagnostics(
                    config_path.parent / input_usd_path,
                    label="generated-material input USD path",
                )
            context["input_usd_path"] = str(input_usd_path)

        plan_path = config.get("material_generation_plan_path")
        if plan_path:
            plan_path = Path(plan_path)
            if not plan_path.is_absolute():
                plan_path = resolve_path_with_safe_diagnostics(
                    config_path.parent / plan_path,
                    label="material-generation plan path",
                )
            context["material_generation_plan_path"] = str(plan_path)

        if "material_generation_plan" in config:
            context["material_generation_plan"] = config["material_generation_plan"]
            context["material_generation_plan_base_dir"] = str(config_path.parent)

        context["texture_generation"] = config.get("texture_generation", {})
        context["material_authoring"] = config.get("material_authoring", {})
        prototype_materials_path = config.get("prototype_materials_path")
        if prototype_materials_path:
            prototype_materials_path = Path(prototype_materials_path)
            if not prototype_materials_path.is_absolute():
                prototype_materials_path = resolve_path_with_safe_diagnostics(
                    config_path.parent / prototype_materials_path,
                    label="prototype-materials path",
                )
            context["prototype_materials_path"] = str(prototype_materials_path)
        if config.get("prototype_materials_data"):
            prototype_data = dict(config["prototype_materials_data"])
            prototype_library_path = prototype_data.get("library_path")
            if prototype_library_path:
                prototype_library_path = Path(prototype_library_path)
                if not prototype_library_path.is_absolute():
                    prototype_library_path = resolve_path_with_safe_diagnostics(
                        config_path.parent / prototype_library_path,
                        label="prototype-material library path",
                    )
                prototype_data["library_path"] = str(prototype_library_path)
            context["prototype_materials_data"] = prototype_data
        context["write_material_generation_plan"] = config.get(
            "write_material_generation_plan", True
        )
        context["include_generation_metadata"] = config.get(
            "include_generation_metadata", True
        )

        def resolve_path_list(key: str) -> list[str]:
            raw_paths = config[key]
            if isinstance(raw_paths, str):
                raw_paths = [raw_paths]
            if not isinstance(raw_paths, list | tuple):
                raise ValueError(f"{key} must be a list of paths")
            resolved_paths: list[str] = []
            for raw_path in raw_paths:
                path = Path(str(raw_path))
                resolved_paths.append(
                    str(
                        resolve_path_with_safe_diagnostics(
                            config_path.parent / path,
                            label="reference image path",
                        )
                    )
                    if not path.is_absolute()
                    else str(path)
                )
            return resolved_paths

        for key in (
            "reference_images",
            "generated_reference_image_paths",
            "rendered_preview_paths",
            "composition_images",
        ):
            if config.get(key):
                context[key] = resolve_path_list(key)

        if config.get("identification"):
            context["identification"] = config["identification"]

        if config.get("material_guidance"):
            context["material_guidance"] = config["material_guidance"]
        elif config.get("planning_guidance"):
            context["planning_guidance"] = config["planning_guidance"]

        if config.get("vlm"):
            context["vlm_config"] = config["vlm"]

        listener.info(f"  Output: {redact_sensitive_path(output_dir)}")
        if context.get("material_generation_plan_path"):
            listener.info(
                "  Plan: "
                f"{redact_sensitive_path(context['material_generation_plan_path'])}"
            )
        elif context.get("material_generation_plan"):
            listener.info("  Plan: inline")

        return context
