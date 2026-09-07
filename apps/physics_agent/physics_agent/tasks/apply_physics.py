# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply physics properties from predictions to a USD stage."""

import logging
from typing import Any

from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from physics_agent.functions.apply_physics import apply_physics

logger = logging.getLogger(__name__)


class ApplyPhysicsTask(Task):
    """Apply UsdPhysics schemas to a USD stage using VLM predictions.

    Reads a predictions JSONL produced by the predict step and writes
    RigidBodyAPI, CollisionAPI, MassAPI, and MaterialAPI onto the matching
    prims, producing a simulation-ready USD file.

    Input context keys:
        - usd_path: Input USD file path
        - predictions_path: Predictions JSONL from predict (or restore_usd) step
        - output_usd_path: Output path for the physics-augmented USD
        - collision_approx: Collision approximation method (default: "convexHull")
        - output_key: Key under which the VLM classification dict lives in
          each prediction entry (default: "classification")
        - mass_scale_policy: warn | skip_mass | fail for mass/scale QA warnings
        - allow_empty_predictions: Allow authoring a rigid-body-only USD from
          an empty predictions file (default: False)
        - approved_dependency_roots: Filesystem roots from which portable
          exports may copy dependencies

    Output context keys:
        - output_usd_path: Absolute path to the written USD file
    """

    def __init__(self) -> None:
        self.name = "ApplyPhysics"
        self.description = "Apply UsdPhysics schemas from predictions to USD stage"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        usd_path = context.get("usd_path")
        predictions_path = context.get("predictions_path")
        output_usd_path = context.get("output_usd_path")
        collision_approx = context.get("collision_approx", "convexHull")
        output_key = context.get("output_key", "classification")
        mass_scale_policy = context.get("mass_scale_policy", "skip_mass")
        allow_empty_predictions = context.get("allow_empty_predictions", False)
        approved_dependency_roots = context.get("approved_dependency_roots")

        if not usd_path:
            raise ValueError("usd_path not in context")
        if not predictions_path:
            raise ValueError("predictions_path not in context")
        if not output_usd_path:
            raise ValueError("output_usd_path not in context")

        logger.info(
            "Applying physics: predictions=%s -> USD=%s", predictions_path, usd_path
        )

        output = apply_physics(
            usd_path=usd_path,
            predictions_path=predictions_path,
            output_path=output_usd_path,
            collision_approx=collision_approx,
            output_key=output_key,
            mass_scale_policy=mass_scale_policy,
            allow_empty_predictions=allow_empty_predictions,
            approved_dependency_roots=approved_dependency_roots,
        )

        context["output_usd_path"] = output
        logger.info("Physics-applied USD saved to %s", output)

        return context
