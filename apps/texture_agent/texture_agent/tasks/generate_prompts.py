# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: expand configured texture prompts and optionally auto-generate more.

``material_textures`` is strict by default: only materials listed there are
expanded into texture units. Set ``auto_prompt.enabled: true`` to generate
prompts for discovered materials that do not have explicit specs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

from texture_agent.api.defaults import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
)
from texture_agent.functions.cached_apply import is_cached_apply_context
from texture_agent.functions.material_discovery import (
    MaterialInfo,
    expand_to_prim_units,
)
from texture_agent.functions.prompt_generation import (
    _fallback_prompts,
    generate_texture_prompts,
)
from texture_agent.planning import TexturePlan, TexturePlanUnit, TextureUnitMode
from texture_agent.planning.contracts import validate_texture_plan_payload

logger = logging.getLogger(__name__)


def _load_resumed_material_textures(
    *,
    material_textures: dict[str, Any],
    working_dir: str | Path | None,
    resume: bool,
) -> dict[str, Any]:
    """Merge cached prompt specs into the current config for resumed runs.

    ``material_prompts.json`` is written after prompt generation, so it is the
    durable source needed to reconstruct texture units without repeating an
    auto-prompt backend call. Current explicit config entries remain
    authoritative when a resumed config intentionally overrides a cached spec.
    """
    if not resume or not working_dir:
        return material_textures

    prompts_path = Path(working_dir) / "prompts" / "material_prompts.json"
    if not prompts_path.is_file():
        return material_textures

    try:
        cached = json.loads(prompts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(
            f"Failed to load cached material prompts from {prompts_path}: {err}"
        ) from err

    if not isinstance(cached, dict) or any(
        not isinstance(key, str) or not isinstance(spec, dict)
        for key, spec in cached.items()
    ):
        raise ValueError(
            f"Cached material prompts must be a mapping of names to specs: "
            f"{prompts_path}"
        )

    merged = {**cached, **material_textures}
    logger.info(
        "Loaded %d cached material prompt specs from %s (%d explicit overrides)",
        len(cached),
        prompts_path,
        len(set(cached).intersection(material_textures)),
    )
    return merged


def _load_resumed_texture_plan(
    context: dict[str, Any],
    *,
    working_dir: str | Path | None,
) -> TexturePlan | None:
    """Load a persisted plan for stable-key reconstruction on resume.

    Legacy sessions created before texture plans existed intentionally return
    ``None`` and retain their display-derived texture keys.
    """
    cached_apply = is_cached_apply_context(context)
    if not context.get("resume") and not cached_apply:
        return None

    planning_config = context.get("planning_config") or {}
    if planning_config.get("resume_apply_textures") and not planning_config.get(
        "apply_texture_plan_unit_ids"
    ):
        # The service creates a fresh plan while hydrating pre-plan sessions,
        # but their existing cache filenames still use legacy display keys.
        return None

    if context.get("texture_plan") is None:
        configured_path = context.get("texture_plan_path")
        if configured_path:
            plan_path = Path(str(configured_path))
        elif working_dir:
            plan_path = Path(working_dir) / "texture_plan.json"
        else:
            return None
        if not plan_path.is_file():
            return None
        context["texture_plan_path"] = str(plan_path)

    from texture_agent.tasks.plan_textures import require_executable_texture_plan

    return require_executable_texture_plan(context)


def _material_sample_label(material: MaterialInfo) -> str:
    """Return a rejection-message label that includes the USD material path."""
    if material.prim_path:
        return f"{material.name} ({material.prim_path})"
    return material.name


def _auto_prompt_material_limit(auto_prompt_config: dict) -> int | None:
    """Return the configured auto-prompt material cap, or None when disabled."""
    raw_limit = auto_prompt_config.get("max_generated_materials")
    if raw_limit is None:
        return None
    if isinstance(raw_limit, bool):
        raise ValueError("auto_prompt.max_generated_materials must be an integer")
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "auto_prompt.max_generated_materials must be an integer"
        ) from exc
    return limit if limit > 0 else None


def _texture_unit_limit(texture_config: dict) -> int | None:
    """Return the expanded texture-unit cap, or None when disabled."""
    raw_limit = texture_config.get("max_texture_units")
    if raw_limit is None:
        return None
    if isinstance(raw_limit, bool):
        raise ValueError("texture.max_texture_units must be an integer")
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("texture.max_texture_units must be an integer") from exc
    return limit if limit > 0 else None


def _material_aliases(material: MaterialInfo) -> set[str]:
    aliases = set(getattr(material, "material_alias_paths", ()) or ())
    aliases.add(material.prim_path)
    return {alias for alias in aliases if alias}


def _plan_material_paths(units: tuple[TexturePlanUnit, ...]) -> set[str]:
    return {path for unit in units for path in unit.material_prim_paths}


def _plan_member_paths(units: tuple[TexturePlanUnit, ...]) -> tuple[set[str], set[str]]:
    prims = {path for unit in units for path in unit.member_prim_paths}
    subsets = {path for unit in units for path in unit.member_subset_paths}
    return prims, subsets


def _spec_for_material(
    material: MaterialInfo,
    material_textures: dict,
) -> dict[str, Any] | None:
    direct = material_textures.get(material.name)
    if isinstance(direct, dict):
        return direct
    for alias in _material_aliases(material):
        scoped = material_textures.get(alias)
        if isinstance(scoped, dict):
            return scoped
    return None


def _apply_texture_plan_scope(
    materials: list[MaterialInfo],
    material_textures: dict,
    context: dict[str, Any],
) -> tuple[list[MaterialInfo], dict]:
    raw_plan = context.get("texture_plan")
    if raw_plan is None:
        return materials, material_textures

    plan = validate_texture_plan_payload(raw_plan)
    selected_material_paths = _plan_material_paths(plan.selected_units)
    selected_prims, selected_subsets = _plan_member_paths(plan.selected_units)
    scoped_materials: list[MaterialInfo] = []
    scoped_textures = dict(material_textures)

    for material in materials:
        if not _material_aliases(material).intersection(selected_material_paths):
            continue
        scoped = material
        if plan.request.unit_mode is TextureUnitMode.PER_PRIM:
            scoped = replace(
                material,
                bound_prim_paths=[
                    path for path in material.bound_prim_paths if path in selected_prims
                ],
                bound_subset_paths=[
                    path
                    for path in material.bound_subset_paths
                    if path in selected_subsets
                ],
            )
        scoped_materials.append(scoped)

    return scoped_materials, scoped_textures


class GeneratePromptsTask(Task):
    """Generate texture prompts for materials missing explicit specs.

    If material_textures config already provides prompts for all selected
    materials, this step is a no-op (no LLM call).

    If auto_prompt.enabled is true and material_textures is empty or some
    materials lack specs, calls an LLM to generate prompts for uncovered
    materials.

    After prompt generation, expands materials into PrimTextureUnits.

    Context keys read:
        discovered_materials (list[MaterialInfo]): From DiscoverMaterialsTask.
        material_textures (dict): Per-material specs from config.
        auto_prompt_config (dict): LLM config and user_prompt.
        texture_config (dict): For mode (per_material / per_prim).
        blend_config (dict): Fallback default opacity for generated specs.
        working_dir (str): Working directory.
        resume (bool): Reuse cached prompt specs before expanding texture units.
        cached_apply_only (bool): Rebuild units strictly from cached artifacts.
        planning_config (dict): Cached-apply and stable plan-ID behavior.
        texture_plan (TexturePlan): Optional selected-unit scope and stable IDs.
        texture_plan_path (str): Optional persisted plan loaded during resume.

    Context keys written:
        material_textures (dict): Updated with auto-generated prompts.
        auto_prompt_additions (dict): Specs added by auto-prompt generation.
        prim_texture_units (list[PrimTextureUnit]): Expanded generation units.
    """

    def __init__(self) -> None:
        self.name = "GeneratePrompts"
        self.description = "Generate texture prompts for materials via LLM"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        materials: list[MaterialInfo] = context.get("discovered_materials", [])
        material_textures: dict = context.get("material_textures", {})
        auto_prompt_config: dict = context.get("auto_prompt_config", {})
        texture_config: dict = context.get("texture_config", {})
        working_dir = context.get("working_dir")
        cached_apply = is_cached_apply_context(context)
        resumed_plan = _load_resumed_texture_plan(
            context,
            working_dir=working_dir,
        )
        material_textures = _load_resumed_material_textures(
            material_textures=material_textures,
            working_dir=working_dir,
            resume=bool(context.get("resume") or cached_apply),
        )
        materials, material_textures = _apply_texture_plan_scope(
            materials,
            material_textures,
            context,
        )
        context["texture_plan_scoped_materials"] = materials
        context["material_textures"] = material_textures

        if not materials and resumed_plan is None:
            logger.info("No materials discovered -- skipping prompt generation")
            context["prim_texture_units"] = []
            return context

        # Determine which materials need auto-prompts
        needs_prompt = [
            material
            for material in materials
            if _spec_for_material(material, material_textures) is None
        ]
        nested_auto_prompt = texture_config.get("auto_prompt", {})
        auto_prompt_enabled = bool(
            auto_prompt_config.get("enabled", nested_auto_prompt.get("enabled", False))
        )

        auto_specs: dict[str, dict[str, Any]] = {}
        if needs_prompt and cached_apply and auto_prompt_enabled:
            missing = ", ".join(_material_sample_label(item) for item in needs_prompt)
            raise RuntimeError(
                "Cached apply requires prompt specs for every selected material; "
                f"missing cached or explicit specs for: {missing}"
            )
        if needs_prompt and auto_prompt_enabled:
            max_generated = _auto_prompt_material_limit(auto_prompt_config)
            if max_generated is not None and len(needs_prompt) > max_generated:
                sample = ", ".join(
                    _material_sample_label(material) for material in needs_prompt[:10]
                )
                if len(needs_prompt) > 10:
                    sample = f"{sample}, ..."
                raise ValueError(
                    "Auto-prompt would select "
                    f"{len(needs_prompt)} discovered materials, exceeding "
                    "auto_prompt.max_generated_materials="
                    f"{max_generated}. Provide explicit material_textures for a "
                    "bounded material/prim subset, disable auto_prompt, or raise "
                    "the limit intentionally. Sample materials: "
                    f"{sample}"
                )

            user_prompt = auto_prompt_config.get("user_prompt", "")
            llm_config = auto_prompt_config.get("llm", {})
            default_opacity = auto_prompt_config.get(
                "default_opacity",
                context.get("blend_config", {}).get("default_opacity", 0.80),
            )

            from world_understanding.functions.models.chat_models import (
                create_chat_model_from_config,
            )

            try:
                llm = create_chat_model_from_config(
                    llm_config,
                    defaults={
                        "backend": DEFAULT_LLM_BACKEND,
                        "model": DEFAULT_LLM_MODEL,
                        "max_tokens": DEFAULT_LLM_MAX_TOKENS,
                        "temperature": DEFAULT_LLM_TEMPERATURE,
                    },
                )
            except Exception as err:
                logger.warning(
                    "Auto-prompt LLM could not be created (%s) -- using "
                    "fallback prompts composed from user_prompt + material name",
                    err,
                )
                llm = None

            if llm is None:
                # create_chat_model_from_config returns None (no warning
                # above) when the backend has no API key available; the
                # try/except handles the other failure modes.
                auto_specs = _fallback_prompts(
                    needs_prompt, user_prompt, default_opacity
                )
            else:
                auto_specs = generate_texture_prompts(
                    materials=needs_prompt,
                    llm=llm,
                    user_prompt=user_prompt,
                    default_opacity=default_opacity,
                )

            # Merge auto-generated specs into material_textures
            # (explicit configs take precedence -- they're already in the dict)
            material_textures.update(auto_specs)
            context["material_textures"] = material_textures
            context["auto_prompt_additions"] = auto_specs

            logger.info(
                "Auto-generated prompts for %d materials "
                "(%d explicit + %d auto = %d total)",
                len(auto_specs),
                len(material_textures) - len(auto_specs),
                len(auto_specs),
                len(material_textures),
            )

            for name, spec in auto_specs.items():
                prompt = spec["prompt"]
                display = prompt[:60] + "..." if len(prompt) > 60 else prompt
                logger.info(
                    "  [auto] %-30s prompt=%r opacity=%.2f",
                    name,
                    display,
                    spec["opacity"],
                )
        elif needs_prompt:
            context["auto_prompt_additions"] = {}
            logger.info(
                "Auto-prompt disabled; %d discovered materials without explicit "
                "material_textures specs will be skipped",
                len(needs_prompt),
            )
        else:
            context["auto_prompt_additions"] = {}
            logger.info(
                "All %d materials have explicit prompts -- skipping LLM",
                len(materials),
            )

        # Save prompts to working dir
        if working_dir:
            out_dir = Path(working_dir) / "prompts"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "material_prompts.json").write_text(
                json.dumps(material_textures, indent=2)
            )

        # Expand to prim texture units
        mode = texture_config.get("mode", "per_material")
        units = expand_to_prim_units(
            materials,
            material_textures,
            mode,
            default_detail_policy=texture_config.get("detail_policy", "default"),
        )
        if resumed_plan is not None:
            from texture_agent.execution import bind_prim_texture_units_to_plan

            units = bind_prim_texture_units_to_plan(resumed_plan, units)
        max_texture_units = _texture_unit_limit(texture_config)
        if max_texture_units is not None and len(units) > max_texture_units:
            sample = ", ".join(u.key for u in units[:10])
            if len(units) > 10:
                sample = f"{sample}, ..."
            raise ValueError(
                "Texture generation would create "
                f"{len(units)} expanded texture units, exceeding "
                f"texture.max_texture_units={max_texture_units}. Provide explicit "
                "material_textures for a bounded material/prim subset, use "
                "per_material mode, or raise the limit intentionally. Sample "
                f"units: {sample}"
            )
        context["prim_texture_units"] = units

        logger.info("Expanded to %d texture units (mode=%s)", len(units), mode)
        for u in units:
            logger.info("  %-40s prim=%s", u.key, u.prim_path or "(all)")

        return context
