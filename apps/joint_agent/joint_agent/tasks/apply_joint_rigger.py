# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Rigger apply-step task."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from joint_agent.functions.joint_rigger_adapter import apply_joint_rigger
from joint_agent.joint_rigger_options import (
    DEFAULT_CANDIDATE_READINESS_POLICY,
    DEFAULT_JOINT_RIGGER_ADAPTER,
    DEFAULT_MISSING_DEPENDENCY_POLICY,
    DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class ApplyJointRiggerTask(Task):
    """Run the disabled-by-default Joint Rigger adapter boundary."""

    def __init__(self) -> None:
        self.name = "ApplyJointRigger"
        self.description = "Apply Joint Rigger adapter boundary"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        input_usd_path = Path(context["input_usd_path"])
        predictions_value = context.get("predictions_path")
        predictions_path = Path(predictions_value) if predictions_value else None
        output_usd_path = Path(context["output_usd_path"])
        diagnostics_path = Path(context["diagnostics_path"])
        validation_path = Path(context["validation_path"])

        listener.info(
            "Running Joint Rigger adapter boundary: "
            f"{context.get('adapter', DEFAULT_JOINT_RIGGER_ADAPTER)}"
        )

        result = apply_joint_rigger(
            input_usd_path=input_usd_path,
            predictions_path=predictions_path,
            output_usd_path=output_usd_path,
            diagnostics_path=diagnostics_path,
            validation_path=validation_path,
            articulation_candidates_path=context.get("articulation_candidates_path"),
            adapter=context.get("adapter", DEFAULT_JOINT_RIGGER_ADAPTER),
            on_missing_dependency=context.get(
                "on_missing_dependency",
                DEFAULT_MISSING_DEPENDENCY_POLICY,
            ),
            on_unready_candidates=context.get(
                "on_unready_candidates",
                DEFAULT_CANDIDATE_READINESS_POLICY,
            ),
            joint_rigger_template=context.get(
                "joint_rigger_template",
                DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
            ),
            apply_masses=context.get("apply_masses"),
            apply_collision=context.get("apply_collision"),
        )

        context.update(result)
        listener.info(
            "Joint Rigger adapter boundary complete: "
            f"{result.get('joint_rigger_status')}, "
            f"authored_joint_count={result.get('authored_joint_count', 0)}"
        )
        return context
