# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: construct and persist the immutable bounded texture plan."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

from texture_agent.planning import (
    TEXTURE_UNIT_DEFAULT_CAP,
    TEXTURE_UV_AWARE_DEFAULT_CAP,
    TexturePlan,
    TexturePlanRequest,
    TexturePlanSource,
    build_texture_plan,
    validate_texture_plan_payload,
)

logger = logging.getLogger(__name__)


class TexturePlanRejectedError(RuntimeError):
    """Raised after a non-executable plan has been durably written."""

    def __init__(self, plan: TexturePlan) -> None:
        self.plan = plan
        reasons = " ".join(plan.decision.reasons)
        actions = " ".join(plan.decision.recommended_actions)
        super().__init__(f"Texture plan rejected: {reasons} {actions}".strip())


def require_executable_texture_plan(context: dict[str, Any]) -> TexturePlan:
    """Load and validate the approved plan required by backend-facing tasks."""
    raw_plan = context.get("texture_plan")
    plan_path = Path(
        context.get("texture_plan_path")
        or Path(context.get("working_dir", ".")) / "texture_plan.json"
    )
    if raw_plan is None:
        if not plan_path.is_file():
            raise RuntimeError(
                "Prompt and image-generation work requires texture_plan.json. "
                "Run discovery and plan_textures first."
            )
        raw_plan = plan_path.read_bytes()
    plan = validate_texture_plan_payload(raw_plan)
    context["texture_plan"] = plan
    context["texture_plan_path"] = str(plan_path)
    if not plan.decision.execution_allowed:
        raise TexturePlanRejectedError(plan)
    return plan


def backend_default_texture_cap(texture_config: dict[str, Any]) -> int:
    """Return the contract default for a concrete generation backend."""
    backend = str(texture_config.get("backend", "simple_image_gen")).lower()
    engine = str(texture_config.get("engine", "")).lower()
    uv_policy = str(texture_config.get("uv_policy", "")).lower()
    simple_remote_engine = engine in {"simple_image_gen", "simple", "image_gen"}
    if (
        (backend == "service" and not simple_remote_engine)
        or "step1x" in backend
        or "step1x" in engine
        or uv_policy == "force_projection"
    ):
        return int(TEXTURE_UV_AWARE_DEFAULT_CAP)
    return int(TEXTURE_UNIT_DEFAULT_CAP)


def _planning_request(context: dict[str, Any]) -> TexturePlanRequest:
    planning_config = context.get("planning_config") or {}
    texture_config = context.get("texture_config") or {}
    input_config = (context.get("config") or {}).get("input") or {}
    backend_cap = planning_config.get("backend_default_cap")
    if backend_cap is None:
        backend_cap = backend_default_texture_cap(texture_config)

    return TexturePlanRequest(
        source=TexturePlanSource(
            source_asset=str(
                planning_config.get("source_asset") or context["usd_path"]
            ),
            upstream_assignment_artifact=planning_config.get(
                "upstream_assignment_artifact"
            ),
        ),
        discovery_mode=planning_config.get("discovery_mode", "effective_bound"),
        unit_mode=planning_config.get(
            "unit_mode",
            texture_config.get("mode", "per_material"),
        ),
        explicit_material_paths=tuple(
            planning_config.get("explicit_material_paths") or ()
        ),
        explicit_prim_paths=tuple(
            planning_config.get("explicit_prim_paths")
            or input_config.get("prim_paths")
            or ()
        ),
        detail_policy=texture_config.get("detail_policy", "default"),
        texture_size=int(texture_config.get("size", 1024)),
        backend=str(texture_config.get("backend", "simple_image_gen")),
        backend_default_cap=int(backend_cap),
        operator_override_cap=planning_config.get("operator_override_cap"),
        max_concurrency=int(
            planning_config.get(
                "max_concurrency",
                texture_config.get(
                    "workers",
                    (context.get("steps") or {})
                    .get("generate_textures", {})
                    .get("max_workers", 4),
                ),
            )
        ),
        unit_timeout_seconds=int(
            planning_config.get(
                "unit_timeout_seconds",
                texture_config.get("job_timeout_sec", 600),
            )
        ),
    )


class PlanTexturesTask(Task):
    """Build ``texture_plan.json`` before any prompt or image backend call.

    Context keys read:
        discovered_materials: Legacy authored-material list.
        effective_material_discovery: Optional WP1 typed discovery result.
        material_textures: Explicit material prompt scope.
        auto_prompt_config: Determines whether all discovered candidates are selected.
        planning_config: Discovery/unit modes, caps, and plan-only behavior.

    Context keys written:
        texture_plan: Validated immutable :class:`TexturePlan`.
        texture_plan_path: Path to the persisted JSON artifact.
    """

    def __init__(self) -> None:
        self.name = "PlanTextures"
        self.description = "Build and validate the bounded texture-generation plan"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        request = _planning_request(context)
        auto_prompt_config = context.get("auto_prompt_config") or {}
        plan = build_texture_plan(
            request,
            discovered_materials=context.get("discovered_materials") or (),
            effective_discovery=context.get("effective_material_discovery"),
            material_textures=context.get("material_textures") or {},
            auto_prompt_enabled=bool(auto_prompt_config.get("enabled", False)),
        )

        working_dir = Path(context["working_dir"])
        working_dir.mkdir(parents=True, exist_ok=True)
        plan_path = working_dir / "texture_plan.json"
        plan_path.write_text(
            json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        context["texture_plan"] = plan
        context["texture_plan_path"] = str(plan_path)
        logger.info(
            "Texture plan %s: %d selected unit(s), effective cap %d, hard cap %d",
            plan.decision.state,
            plan.counts.selected_unit_count,
            plan.limits.effective_cap,
            plan.limits.hard_cap,
        )

        planning_config = context.get("planning_config") or {}
        if not plan.decision.execution_allowed and not planning_config.get(
            "plan_only", False
        ):
            raise TexturePlanRejectedError(plan)
        return context
