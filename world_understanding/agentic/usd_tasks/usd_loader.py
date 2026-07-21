# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD stage loading task."""

import logging
from pathlib import Path
from typing import Any

from pxr import Usd, UsdGeom

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.usd_model import USDModel
from world_understanding.utils.credentials import redact_sensitive_path
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.usd.stage import load_stage

logger = logging.getLogger(__name__)

_USD_FILE_NOT_FOUND_MESSAGE = "USD file not found"
_USD_LOADING_FAILURE_MESSAGE = "USD stage loading failed"


class _USDFileNotFoundError(FileNotFoundError):
    """A value-free missing-file error owned by the loading task."""


class USDLoadingTask(Task):
    """Load USD stage and prepare for processing."""

    def __init__(self):
        self.name = "USDLoading"
        self.description = "Load USD stage and apply configurations"

    def run(self, context: dict[str, Any], object_store: ObjectStore) -> dict[str, Any]:
        """Load USD stage from file.

        Expected context inputs:
            - usd_path: Path to USD file
            - build_usd_model: Whether to build USDModel (default: True)

        Updates context with:
            - stage_loaded: Boolean indicating success
            - num_prims: Total number of prims in stage
            - stage_info: Basic stage information
            - usd_model_built: Boolean indicating if USDModel was built

        Stores in object_store:
            - usd_stage: The loaded Usd.Stage object
            - usd_model: The USDModel instance (if build_usd_model is True)
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        usd_path = context.get("usd_path")
        if not usd_path:
            raise ValueError("usd_path not found in context")

        usd_path = Path(usd_path)
        build_usd_model = context.get("build_usd_model", True)

        load_failed = False

        try:
            if not usd_path.exists():
                raise _USDFileNotFoundError(_USD_FILE_NOT_FOUND_MESSAGE)

            # Keep the raw path for filesystem and USD operations, but never
            # project it into an event-listener diagnostic.
            listener.info("Loading USD stage")

            # Load the stage using the utility function
            stage = load_stage(str(usd_path))

            # Apply prototype conversion if requested (handles both class and over specifiers)
            convert_prototypes = context.get("convert_prototypes_to_xforms", False)
            if convert_prototypes:
                from world_understanding.utils.usd.prim import (
                    convert_abstract_prototypes_to_def,
                )

                prototype_names = context.get("prototype_names", None)
                listener.info("Converting abstract prototypes (class/over) to def...")
                converted_count = convert_abstract_prototypes_to_def(
                    stage, prototype_names=prototype_names
                )
                context["prototypes_converted"] = converted_count
                if converted_count > 0:
                    listener.info(
                        f"Converted {converted_count} abstract prototype(s) to def"
                    )
                else:
                    listener.info("No abstract prototypes found to convert")
            else:
                context["prototypes_converted"] = 0

            # Store stage in object store (it's a large object)
            object_store.set("usd_stage", stage)

            # Build USDModel if requested
            usd_model = None
            if build_usd_model:
                listener.info("Building USD model for hierarchy and collections...")
                usd_model = USDModel(str(usd_path))
                object_store.set("usd_model", usd_model)
                context["usd_model_built"] = True
                listener.info(
                    f"USD model built: {len(usd_model.prims)} prims, "
                    f"{len(usd_model.collections)} collections"
                )
            else:
                context["usd_model_built"] = False

            # Get basic stage information
            if usd_model:
                # Use USDModel stats if available
                num_prims = len(usd_model.prims)
                mesh_count = len(usd_model.get_all_meshes())
                xform_count = len(usd_model.get_all_xforms())
            else:
                # Fall back to manual counting
                root_prim = stage.GetPseudoRoot()
                all_prims = list(Usd.PrimRange(root_prim))
                num_prims = len(all_prims)
                mesh_count = sum(1 for p in all_prims if p.IsA(UsdGeom.Mesh))
                xform_count = sum(1 for p in all_prims if p.IsA(UsdGeom.Xform))

            # Update context
            context["stage_loaded"] = True
            context["num_prims"] = num_prims
            context["stage_info"] = {
                "total_prims": num_prims,
                "mesh_prims": mesh_count,
                "xform_prims": xform_count,
                "root_layer": redact_sensitive_path(stage.GetRootLayer().identifier),
                "up_axis": UsdGeom.GetStageUpAxis(stage),
                "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            }

            listener.info("USD stage loaded successfully:")
            listener.info(f"  Total prims: {num_prims}")
            listener.info(f"  Mesh prims: {mesh_count}")
            listener.info(f"  Xform prims: {xform_count}")
            listener.info(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}")
            listener.info(f"  Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")

        except _USDFileNotFoundError:
            context["stage_loaded"] = False
            context["error"] = _USD_FILE_NOT_FOUND_MESSAGE
            listener.error(_USD_FILE_NOT_FOUND_MESSAGE)
            raise
        except Exception:
            # USD and filesystem exceptions can contain source paths, backend
            # payloads, or credentialed locators. Record only the task-owned
            # category and raise a fresh exception after leaving the handler so
            # the untrusted exception is not retained as a cause or context.
            context["stage_loaded"] = False
            context["error"] = _USD_LOADING_FAILURE_MESSAGE
            listener.error(_USD_LOADING_FAILURE_MESSAGE)
            load_failed = True

        if load_failed:
            raise RuntimeError(_USD_LOADING_FAILURE_MESSAGE)

        return context
