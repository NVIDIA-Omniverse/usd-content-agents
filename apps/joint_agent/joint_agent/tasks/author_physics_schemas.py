# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task wrapper for explicit post-rigger physics schema authoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from joint_agent.functions.physics_schema_authoring import author_physics_schemas

logger = logging.getLogger(__name__)


class AuthorPhysicsSchemasTask(Task):
    """Apply an explicit, evidence-backed physics schema authoring plan."""

    def __init__(self) -> None:
        self.name = "AuthorPhysicsSchemas"
        self.description = "Author explicit post-rigger physics schemas"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)
        listener.info("Running explicit post-rigger physics schema authoring")
        result = author_physics_schemas(
            input_usd_path=Path(context["input_usd_path"]),
            stage2_diagnostics_path=Path(context["stage2_diagnostics_path"]),
            stage2_validation_path=Path(context["stage2_validation_path"]),
            authoring_plan_path=Path(context["authoring_plan_path"]),
            output_usd_path=Path(context["output_usd_path"]),
            diagnostics_path=Path(context["diagnostics_path"]),
            validation_path=Path(context["validation_path"]),
        )
        context.update(result)
        listener.info(
            f"Physics schema authoring complete: {result['physics_authoring_status']}"
        )
        return context
