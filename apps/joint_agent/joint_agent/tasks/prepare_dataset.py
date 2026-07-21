# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for preparing dataset for asset classification."""

import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.rendering import (
    parse_camera_angle_from_view_name,
)

from joint_agent.functions.consistency import normalized_path_signature

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RigidBodySourceIndex:
    """Source-authored rigid bodies plus hierarchy-only Xform choices."""

    endpoint_paths: tuple[str, ...] = ()
    owner_by_prim_path: dict[str, str] | None = None
    hierarchy_gaps_by_prim_path: dict[str, tuple[str, ...]] | None = None
    hierarchy_xform_paths: tuple[str, ...] = ()
    hierarchy_ancestor_xforms_by_prim_path: dict[str, tuple[str, ...]] | None = None

    def owner_for(self, prim_path: str) -> str | None:
        return (self.owner_by_prim_path or {}).get(prim_path)

    def hierarchy_gaps_for(self, prim_path: str) -> tuple[str, ...]:
        return (self.hierarchy_gaps_by_prim_path or {}).get(prim_path, ())

    def hierarchy_ancestors_for(self, prim_path: str) -> tuple[str, ...]:
        return (self.hierarchy_ancestor_xforms_by_prim_path or {}).get(prim_path, ())

    def hierarchy_choices_for(
        self,
        prim_path: str,
        *,
        path_limit: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return bounded Xform choices, prioritizing this prim's ancestors."""
        if path_limit < 1:
            raise ValueError("scene_prim_path_limit must be >= 1")
        choices: list[str] = []
        ancestors = self.hierarchy_ancestors_for(prim_path)
        for path in (*ancestors, *self.hierarchy_xform_paths):
            if path not in choices:
                choices.append(path)
            if len(choices) >= path_limit:
                break
        shown_ancestors = tuple(path for path in ancestors if path in choices)
        return tuple(choices), shown_ancestors


def _lexical_parent_prim_path(prim_path: str) -> str | None:
    """Return the lexical USD parent without requiring an exported prim row."""
    normalized = prim_path.rstrip("/")
    if not normalized:
        return None
    parent_path = normalized.rsplit("/", 1)[0]
    return parent_path or "/"


def _build_rigid_body_source_index(
    usd_model_data: Mapping[str, Any],
    *,
    rendered_prim_paths: list[str],
) -> _RigidBodySourceIndex:
    """Build a structural endpoint index from one exported ``usd_model.json``.

    The index records authored ``PhysicsRigidBodyAPI`` membership, nearest
    rigid-body ownership, and exact exported Xform ancestry for an unrigged
    fallback vocabulary. It deliberately does not infer which Xform moves,
    joint connectivity, parent direction, or motion axes.
    """
    raw_prims = usd_model_data.get("prims")
    if not isinstance(raw_prims, Mapping):
        return _RigidBodySourceIndex()

    prims = {
        str(path): value
        for path, value in raw_prims.items()
        if isinstance(path, str) and path.startswith("/") and isinstance(value, Mapping)
    }
    endpoint_paths = tuple(
        sorted(
            path
            for path, prim_data in prims.items()
            if _has_physics_rigid_body_api(prim_data.get("api_schemas"))
        )
    )
    hierarchy_xform_paths = tuple(
        sorted(
            path
            for path, prim_data in prims.items()
            if prim_data.get("type_name") == "Xform"
            and not bool(prim_data.get("is_in_prototype"))
        )
    )
    hierarchy_xform_set = set(hierarchy_xform_paths)
    endpoint_set = set(endpoint_paths)
    owner_by_prim_path: dict[str, str] = {}
    hierarchy_gaps_by_prim_path: dict[str, tuple[str, ...]] = {}
    hierarchy_ancestor_xforms_by_prim_path: dict[str, tuple[str, ...]] = {}
    for prim_path in rendered_prim_paths:
        ancestor_xforms: list[str] = []
        current_path = prim_path
        visited: set[str] = set()
        while current_path and current_path not in visited:
            visited.add(current_path)
            prim_data = prims.get(current_path)
            lexical_parent = _lexical_parent_prim_path(current_path)
            if lexical_parent is None or lexical_parent == "/":
                break
            parent_path = (
                prim_data.get("parent_path") if prim_data is not None else None
            )
            if parent_path != lexical_parent:
                break
            current_path = lexical_parent
            if current_path in hierarchy_xform_set:
                ancestor_xforms.append(current_path)
        if ancestor_xforms:
            hierarchy_ancestor_xforms_by_prim_path[prim_path] = tuple(ancestor_xforms)

        current_path = prim_path
        visited = set()
        hierarchy_gaps: list[str] = []
        while current_path and current_path not in visited:
            visited.add(current_path)
            if current_path in endpoint_set:
                owner_by_prim_path[prim_path] = current_path
                break
            prim_data = prims.get(current_path)
            parent_path = (
                prim_data.get("parent_path") if prim_data is not None else None
            )
            lexical_parent = _lexical_parent_prim_path(current_path)
            if lexical_parent is None:
                break
            if parent_path == lexical_parent:
                current_path = parent_path
                continue
            if prim_data is not None and lexical_parent == "/":
                # Exported top-level prims legitimately have no authored
                # parent row beyond the pseudo-root.
                break
            if current_path != "/":
                hierarchy_gaps.append(current_path)
            current_path = lexical_parent
        if hierarchy_gaps:
            hierarchy_gaps_by_prim_path[prim_path] = tuple(
                dict.fromkeys(hierarchy_gaps)
            )

    if hierarchy_gaps_by_prim_path:
        gap_count = len(
            {gap for gaps in hierarchy_gaps_by_prim_path.values() for gap in gaps}
        )
        logger.warning(
            "Source USD hierarchy omitted or malformed %d prim row(s) across "
            "%d rendered prim(s); rigid-body ownership used lexical ancestry",
            gap_count,
            len(hierarchy_gaps_by_prim_path),
        )

    return _RigidBodySourceIndex(
        endpoint_paths=endpoint_paths,
        owner_by_prim_path=owner_by_prim_path,
        hierarchy_gaps_by_prim_path=hierarchy_gaps_by_prim_path,
        hierarchy_xform_paths=hierarchy_xform_paths,
        hierarchy_ancestor_xforms_by_prim_path=(hierarchy_ancestor_xforms_by_prim_path),
    )


def _has_physics_rigid_body_api(raw_api_schemas: Any) -> bool:
    if not isinstance(raw_api_schemas, list):
        return False
    return any(
        str(schema).split(":", 1)[0] == "PhysicsRigidBodyAPI"
        for schema in raw_api_schemas
    )


def _load_rigid_body_source_index(
    usd_input_dir: Path,
    dataset_metadata: Mapping[str, Any],
    *,
    rendered_prim_paths: list[str],
) -> _RigidBodySourceIndex:
    """Load source structure referenced by dataset metadata, when available."""
    usd_model_file = dataset_metadata.get("usd_model_file")
    if not isinstance(usd_model_file, str) or not usd_model_file.strip():
        return _RigidBodySourceIndex()
    usd_model_path = Path(usd_model_file)
    if not usd_model_path.is_absolute():
        usd_model_path = usd_input_dir / usd_model_path
    with usd_model_path.open(encoding="utf-8") as f:
        usd_model_data = json.load(f)
    if not isinstance(usd_model_data, Mapping):
        return _RigidBodySourceIndex()
    return _build_rigid_body_source_index(
        usd_model_data,
        rendered_prim_paths=rendered_prim_paths,
    )


def _format_rigid_body_endpoint_context(
    source_index: _RigidBodySourceIndex,
    *,
    current_prim_path: str,
    path_limit: int,
) -> str:
    """Format exact source-authored endpoint choices without asserting a joint."""
    if path_limit < 1:
        raise ValueError("scene_prim_path_limit must be >= 1")
    if not source_index.endpoint_paths:
        return ""

    endpoint_paths = list(source_index.endpoint_paths)
    current_owner = source_index.owner_for(current_prim_path)
    shown_paths = endpoint_paths[:path_limit]
    if current_owner and current_owner not in shown_paths:
        shown_paths = sorted([*shown_paths[:-1], current_owner])
    omitted_count = len(endpoint_paths) - len(shown_paths)
    path_lines = [f"  - {prim_path}" for prim_path in shown_paths]
    if omitted_count > 0:
        path_lines.append(f"  - ... ({omitted_count} more)")

    owner_text = current_owner or "none"
    return "\n".join(
        [
            "Source-authored rigid-body endpoint vocabulary:",
            f"  - current rendered prim: {current_prim_path}",
            f"  - nearest authored rigid-body owner: {owner_text}",
            f"  - total authored rigid-body endpoints: {len(endpoint_paths)}",
            "  - exact body0/body1 endpoint choices:",
            *path_lines,
            (
                "When explicit visual evidence supports an articulation, use only "
                "these exact endpoint paths for rigger_evidence body0/body1. The "
                "owner identifies which rigid body contains this rendered prim; "
                "ownership alone does not establish a joint, fixed parent, axis, "
                "or candidate status."
            ),
        ]
    )


def _format_hierarchy_endpoint_context(
    source_index: _RigidBodySourceIndex,
    *,
    current_prim_path: str,
    path_limit: int,
) -> str:
    """Format exact Xform choices without mechanically selecting a body."""
    if source_index.endpoint_paths or not source_index.hierarchy_xform_paths:
        return ""
    choices, ancestors = source_index.hierarchy_choices_for(
        current_prim_path,
        path_limit=path_limit,
    )

    ancestor_lines = [f"  - {path}" for path in ancestors] or ["  - none"]
    choice_lines = [f"  - {path}" for path in choices]
    omitted_count = len(source_index.hierarchy_xform_paths) - len(choices)
    if omitted_count > 0:
        choice_lines.append(f"  - ... ({omitted_count} more source Xforms)")
    return "\n".join(
        [
            "Source hierarchy endpoint choices:",
            f"  - current rendered prim: {current_prim_path}",
            "  - exact ancestor Xforms, nearest first:",
            *ancestor_lines,
            "  - bounded exact Xform vocabulary:",
            *choice_lines,
            (
                "The source hierarchy proves only that these paths exist and "
                "which listed Xforms contain the rendered prim. The exact current "
                "rendered prim may be rigger_evidence.body1 when visual and "
                "semantic evidence identifies that prim itself as independently "
                "moving, even when it is a Mesh or another Gprim. This row-local "
                "body1 permission does not make the current prim a body0 or "
                "authorize sibling or nearby Gprims merely because their paths "
                "are listed. A different rendered Gprim may be body0 for the "
                "direct edge whose body1 is the current rendered prim only when "
                "its own prediction independently identifies it as a fixed, "
                "non-articulating support in the same source-exported assembly; "
                "names, proximity, and containment are insufficient. A listed "
                "Xform may be body0 only when independent visual or semantic evidence "
                "identifies it as the fixed/support body. A listed ancestor may be "
                "body1 only when evidence shows that the whole ancestor assembly "
                "moves as one body. Never select or lift to an ancestor merely "
                "because it contains the current prim. Hierarchy alone does not "
                "prove articulation, connectivity, a motion axis, or which "
                "ancestor moves."
            ),
        ]
    )


def _build_repetition_contexts(
    prims_data: list[dict[str, Any]],
    *,
    signature_depth: int,
    sibling_limit: int,
) -> dict[str, str]:
    """Build per-prim prompt hints for repeated path-signature groups."""
    if signature_depth < 1:
        raise ValueError("repetition_signature_depth must be >= 1")
    if sibling_limit < 1:
        raise ValueError("repetition_sibling_limit must be >= 1")

    groups: dict[str, list[str]] = defaultdict(list)
    for prim_data in prims_data:
        prim_path = prim_data.get("prim_path")
        if not isinstance(prim_path, str) or not prim_path:
            continue
        signature = normalized_path_signature(prim_path, depth=signature_depth)
        if signature:
            groups[signature].append(prim_path)

    contexts: dict[str, str] = {}
    for signature, group_paths in groups.items():
        if len(group_paths) < 2:
            continue

        sorted_group_paths = sorted(group_paths)
        for prim_path in sorted_group_paths:
            related_paths = [path for path in sorted_group_paths if path != prim_path]
            shown_paths = related_paths[:sibling_limit]
            omitted_count = len(related_paths) - len(shown_paths)
            related_text = "; ".join(shown_paths)
            if omitted_count > 0:
                related_text = f"{related_text}; ... ({omitted_count} more)"

            contexts[prim_path] = "\n".join(
                [
                    "Repeated/symmetric part hint:",
                    f"  - normalized path signature: {signature}",
                    f"  - repeated group size: {len(sorted_group_paths)}",
                    f"  - related prim paths: {related_text}",
                    (
                        "Use this as supporting evidence that the current prim may "
                        "be one instance of a repeated articulated part."
                    ),
                ]
            )

    return contexts


def _build_scene_prim_path_context(
    prims_data: list[dict[str, Any]],
    *,
    path_limit: int,
    current_prim_path: str | None = None,
) -> str:
    """Build a bounded scene path inventory for exact rigger evidence."""
    return _format_scene_prim_path_context(
        _collect_scene_prim_paths(prims_data),
        path_limit=path_limit,
        current_prim_path=current_prim_path,
    )


def _collect_scene_prim_paths(prims_data: list[dict[str, Any]]) -> list[str]:
    """Collect sorted absolute prim paths eligible for scene path context."""
    return sorted(
        {
            prim_path
            for prim_data in prims_data
            if isinstance((prim_path := prim_data.get("prim_path")), str)
            and prim_path.startswith("/")
        }
    )


def _source_prim_type(prim_data: Mapping[str, Any]) -> str:
    """Return the authored render-target type without assuming it is a Mesh."""
    for container_name, key in (("metadata", "type"), ("hierarchy", "type_name")):
        container = prim_data.get(container_name)
        if not isinstance(container, Mapping):
            continue
        value = container.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unknown"


def _format_scene_prim_path_context(
    prim_paths: list[str],
    *,
    path_limit: int,
    current_prim_path: str | None = None,
) -> str:
    """Format a bounded scene path inventory for exact rigger evidence."""
    if path_limit < 1:
        raise ValueError("scene_prim_path_limit must be >= 1")

    if not prim_paths:
        return ""

    shown_paths = prim_paths[:path_limit]
    if (
        current_prim_path
        and current_prim_path in prim_paths
        and current_prim_path not in shown_paths
    ):
        shown_paths = sorted([*shown_paths[:-1], current_prim_path])
    omitted_count = len(prim_paths) - len(shown_paths)
    path_lines = [f"  - {prim_path}" for prim_path in shown_paths]
    if omitted_count > 0:
        path_lines.append(f"  - ... ({omitted_count} more)")

    return "\n".join(
        [
            "Scene prim path inventory:",
            f"  - total listed render-target prim paths: {len(prim_paths)}",
            "  - available exact prim paths:",
            *path_lines,
            (
                "Use these exact absolute paths as supporting prim_paths in "
                "rigger_evidence. When a source-authored rigid-body endpoint "
                "vocabulary is provided below, body0/body1 must use that endpoint "
                "vocabulary instead of substituting a render-target prim path. "
                "Otherwise, "
                "if the needed exact endpoint is not listed or not explicit, omit "
                "that endpoint claim."
            ),
        ]
    )


def _format_stage_coordinate_context(dataset_metadata: dict[str, Any]) -> str:
    """Format stage coordinate-frame metadata for axis-grounded prompts."""
    stage_up_axis = dataset_metadata.get("stage_up_axis")
    if not isinstance(stage_up_axis, str) or not stage_up_axis.strip():
        return ""

    parts = [
        f"  - stage up axis: {stage_up_axis.strip().upper()} (the vertical scene axis)"
    ]

    meters_per_unit = dataset_metadata.get("meters_per_unit")
    if isinstance(meters_per_unit, int | float) and not isinstance(
        meters_per_unit, bool
    ):
        parts.append(f"  - meters per unit: {float(meters_per_unit):g}")

    return "\n".join(
        [
            "USD coordinate frame:",
            *parts,
            (
                "Use x/y/z axis tokens in this USD stage frame. If visual evidence "
                "uses words like vertical, up, or down, map them through the stage "
                "up axis above before emitting an axis_hint or motion_axis value."
            ),
            (
                "Geometric extents below are labeled extent_x, extent_y, and "
                "extent_z in this same stage frame; they are not screen-space "
                "width, height, or depth. Camera descriptions such as +X identify "
                "the stage-space side from which the camera looks toward the target."
            ),
        ]
    )


class PrepareDatasetTask(Task):
    """Task to prepare dataset for asset classification.

    This task creates dataset entries from USD renderings, combining images
    with configurable prompts for VLM classification.

    Input context keys:
        - usd_dir: Path to input USD dataset directory
        - dataset_path: Path to output dataset directory
        - models: List of model numbers to process
        - config: Configuration dictionary with optional flags:
            * 'include_prim_path_context' (bool): Include prim path in context
            * 'include_stage_coordinate_context' (bool): Include USD stage
              coordinate-frame context when stage metadata is available
            * 'include_scene_prim_path_context' (bool): Include scene path inventory
            * 'scene_prim_path_limit' (int): Max scene prim paths to show
            * Source-authored rigid-body endpoints and rendered-prim ownership are
              included automatically when ``usd_model.json`` is available
            * 'include_geometric_context' (bool): Include geometric info
            * 'include_repetition_context' (bool): Include repeated part hints
            * 'repetition_signature_depth' (int): Path suffix depth for grouping
            * 'repetition_sibling_limit' (int): Max related prim paths to show
            * 'prompts' (dict): Custom prompt templates
            * 'render_mode_filter' (list[str]): Optional filter for render modes

    Output context keys:
        - dataset_entries: List of prepared dataset entries
        - failed_models: List of model numbers that failed to process
        - dataset_jsonl_path: Path where dataset.jsonl was saved
    """

    def __init__(self) -> None:
        """Initialize the prepare dataset task."""
        self.name = "PrepareDataset"
        self.description = "Prepare dataset for asset classification"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        """Prepare dataset entries for the specified models.

        Args:
            context: Workflow context containing required parameters
            object_store: Optional object store (not used)

        Returns:
            Updated context with prepared dataset entries
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        usd_dir = context.get("usd_dir")
        dataset_path = context.get("dataset_path")
        models = context.get("models", [])
        config = context.get("config", {})

        if not usd_dir:
            raise ValueError("usd_dir not provided in context")
        if not dataset_path:
            raise ValueError("dataset_path not provided in context")
        if not models:
            raise ValueError("models not provided in context")

        usd_dir = Path(usd_dir)
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        listener.info(f"Preparing dataset for {len(models)} models")

        # Get configuration options
        include_prim_path_context = config.get("include_prim_path_context", False)
        include_scene_prim_path_context = config.get(
            "include_scene_prim_path_context",
            context.get("include_scene_prim_path_context", False),
        )
        scene_prim_path_limit = int(
            config.get(
                "scene_prim_path_limit",
                context.get("scene_prim_path_limit", 80),
            )
        )
        include_geometric_context = config.get("include_geometric_context", True)
        include_repetition_context = config.get("include_repetition_context", True)
        include_stage_coordinate_context = config.get(
            "include_stage_coordinate_context",
            context.get("include_stage_coordinate_context"),
        )
        if include_stage_coordinate_context is None:
            include_stage_coordinate_context = any(
                [
                    include_prim_path_context,
                    include_scene_prim_path_context,
                    include_geometric_context,
                    include_repetition_context,
                ]
            )
        repetition_signature_depth = int(config.get("repetition_signature_depth", 2))
        repetition_sibling_limit = int(config.get("repetition_sibling_limit", 5))

        # Load structure assignments if available (from analyze_structure step)
        structure_assignments: dict[str, str] = {}
        structure_assignments_path = config.get("structure_assignments_path")
        if structure_assignments_path:
            try:
                with open(structure_assignments_path, encoding="utf-8") as f:
                    sa_data = json.load(f)
                for prim_path, info in sa_data.get("assignments", {}).items():
                    if isinstance(info, dict):
                        structure_assignments[prim_path] = info.get(
                            "component_name", ""
                        )
                    elif isinstance(info, str):
                        structure_assignments[prim_path] = info
                listener.info(
                    f"Loaded {len(structure_assignments)} structure assignments from {structure_assignments_path}"
                )
            except Exception as e:
                listener.warning(f"Failed to load structure assignments: {e}")

        # Get custom prompt templates from config if provided
        prompt_config = config.get("prompts", {})
        system_prompt = prompt_config.get("system", "")
        user_prompt_template = prompt_config.get(
            "user", "Please analyze this asset and provide your classification."
        )

        # Get VLM image prompts if provided
        vlm_image_prompts = prompt_config.get("vlm_image_prompts", {})
        if isinstance(vlm_image_prompts, list):
            merged = {}
            for item in vlm_image_prompts:
                if isinstance(item, dict):
                    merged.update(item)
            vlm_image_prompts = merged

        # Get reference images from context
        reference_images = context.get("reference_images", [])
        # Support per-image prompts: string (shared) or list (per-image)
        reference_image_prompts_config = vlm_image_prompts.get(
            "reference_images", "This is a reference image of the asset."
        )
        if isinstance(reference_image_prompts_config, str):
            reference_image_prompts_list = [reference_image_prompts_config] * len(
                reference_images
            )
        else:
            reference_image_prompts_list = list(reference_image_prompts_config)

        dataset_entries = []
        failed_models = []

        for model_number in models:
            try:
                listener.info(f"Processing model: {model_number}")

                # Check for USD dataset structure in input directory
                usd_input_dir = usd_dir / model_number
                dataset_json_path = usd_input_dir / "dataset.json"
                prims_jsonl_path = usd_input_dir / "prims.jsonl"

                # Create output directory for this model
                output_dir = dataset_path / model_number
                output_dir.mkdir(parents=True, exist_ok=True)

                if not dataset_json_path.exists():
                    raise ValueError(f"Dataset JSON not found for {model_number}")
                if not prims_jsonl_path.exists():
                    raise ValueError(f"Prims JSONL not found for {model_number}")

                # Load dataset metadata
                with open(dataset_json_path, encoding="utf-8") as f:
                    dataset_metadata = json.load(f)
                total_prims = dataset_metadata["statistics"]["total_prims"]
                listener.info(f"Loaded dataset metadata with {total_prims} prims")
                stage_coordinate_context = (
                    _format_stage_coordinate_context(dataset_metadata)
                    if include_stage_coordinate_context
                    else ""
                )

                # Load prims data
                prims_data = []
                with open(prims_jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        prims_data.append(json.loads(line))
                listener.info(f"Loaded {len(prims_data)} prims from prims.jsonl")

                rendered_prim_paths = _collect_scene_prim_paths(prims_data)
                rigid_body_source_index = _RigidBodySourceIndex()
                try:
                    rigid_body_source_index = _load_rigid_body_source_index(
                        usd_input_dir,
                        dataset_metadata,
                        rendered_prim_paths=rendered_prim_paths,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    listener.warning(
                        "Failed to load source-authored rigid-body endpoint "
                        f"metadata: {exc}"
                    )
                if rigid_body_source_index.endpoint_paths:
                    listener.info(
                        "Loaded "
                        f"{len(rigid_body_source_index.endpoint_paths)} "
                        "source-authored rigid-body endpoints and "
                        f"{len(rigid_body_source_index.owner_by_prim_path or {})} "
                        "rendered-prim owners"
                    )

                repetition_contexts: dict[str, str] = {}
                if include_repetition_context:
                    repetition_contexts = _build_repetition_contexts(
                        prims_data,
                        signature_depth=repetition_signature_depth,
                        sibling_limit=repetition_sibling_limit,
                    )
                    listener.info(
                        f"Built repeated-part context for "
                        f"{len(repetition_contexts)} prims"
                    )

                scene_prim_paths: list[str] = []
                has_scene_prim_path_context = False
                if include_scene_prim_path_context:
                    scene_prim_paths = rendered_prim_paths
                    has_scene_prim_path_context = bool(
                        _format_scene_prim_path_context(
                            scene_prim_paths,
                            path_limit=scene_prim_path_limit,
                        )
                    )
                    if has_scene_prim_path_context:
                        listener.info(
                            "Added scene prim-path inventory to prompt context"
                        )

                # Process each prim
                for prim_idx, prim_data in enumerate(prims_data):
                    prim_path = prim_data["prim_path"]
                    listener.debug(f"Processing prim {prim_idx}: {prim_path}")

                    # Build context for this prim
                    prim_context = ""

                    if stage_coordinate_context:
                        prim_context = stage_coordinate_context

                    # Add prim path to context if enabled
                    if include_prim_path_context:
                        prim_path_context = (
                            f"The prim path of this 3D asset is: {prim_path}"
                        )
                        if prim_context:
                            prim_context = f"{prim_context}\n\n{prim_path_context}"
                        else:
                            prim_context = prim_path_context

                    # Add geometric context if enabled and available
                    if include_geometric_context:
                        geometric_parts = []

                        # Add world bbox in meters if available
                        world_bbox_meters = prim_data.get("world_bbox_meters")
                        if world_bbox_meters:
                            size_m = world_bbox_meters["size"]
                            geometric_parts.append(
                                f"Stage-axis extents (meters): "
                                f"extent_x={size_m[0]:.3f}m, "
                                f"extent_y={size_m[1]:.3f}m, "
                                f"extent_z={size_m[2]:.3f}m"
                            )
                            center_m = world_bbox_meters.get("center")
                            if isinstance(center_m, list) and len(center_m) == 3:
                                geometric_parts.append(
                                    f"Stage-space center (meters): "
                                    f"x={center_m[0]:.3f}m, "
                                    f"y={center_m[1]:.3f}m, "
                                    f"z={center_m[2]:.3f}m"
                                )
                            bbox_volume = size_m[0] * size_m[1] * size_m[2]
                            geometric_parts.append(
                                f"Bounding box volume: {bbox_volume:.6f} m³"
                            )

                        # Add relative metrics if available
                        relative_metrics = prim_data.get("relative_metrics")
                        if relative_metrics:
                            rel_size = relative_metrics["relative_size"]
                            geometric_parts.append(
                                f"Relative stage-axis extents (% of whole): "
                                f"extent_x={rel_size[0] * 100:.1f}%, "
                                f"extent_y={rel_size[1] * 100:.1f}%, "
                                f"extent_z={rel_size[2] * 100:.1f}%"
                            )

                        if geometric_parts:
                            geometric_context = "Geometric info:\n" + "\n".join(
                                [f"  - {part}" for part in geometric_parts]
                            )
                            if prim_context:
                                prim_context = f"{prim_context}\n\n{geometric_context}"
                            else:
                                prim_context = geometric_context

                    # Add repeated/symmetric path context when available.
                    repetition_context = repetition_contexts.get(prim_path)
                    if repetition_context:
                        if prim_context:
                            prim_context = f"{prim_context}\n\n{repetition_context}"
                        else:
                            prim_context = repetition_context

                    if has_scene_prim_path_context:
                        scene_prim_path_context = _format_scene_prim_path_context(
                            scene_prim_paths,
                            path_limit=scene_prim_path_limit,
                            current_prim_path=prim_path,
                        )
                        if prim_context:
                            prim_context = (
                                f"{prim_context}\n\n{scene_prim_path_context}"
                            )
                        else:
                            prim_context = scene_prim_path_context

                    rigid_body_endpoint_context = _format_rigid_body_endpoint_context(
                        rigid_body_source_index,
                        current_prim_path=prim_path,
                        path_limit=scene_prim_path_limit,
                    )
                    if rigid_body_endpoint_context:
                        if prim_context:
                            prim_context = (
                                f"{prim_context}\n\n{rigid_body_endpoint_context}"
                            )
                        else:
                            prim_context = rigid_body_endpoint_context

                    hierarchy_endpoint_context = _format_hierarchy_endpoint_context(
                        rigid_body_source_index,
                        current_prim_path=prim_path,
                        path_limit=scene_prim_path_limit,
                    )
                    if hierarchy_endpoint_context:
                        if prim_context:
                            prim_context = (
                                f"{prim_context}\n\n{hierarchy_endpoint_context}"
                            )
                        else:
                            prim_context = hierarchy_endpoint_context

                    # Inject structure assignment if available
                    if prim_path in structure_assignments:
                        segment_name = structure_assignments[prim_path]
                        structure_context = (
                            f"Structure analysis: This component has been "
                            f"identified as part of the **{segment_name}** "
                            f"segment. Use this as the component_name."
                        )
                        if prim_context:
                            prim_context = f"{prim_context}\n\n{structure_context}"
                        else:
                            prim_context = structure_context

                    # Format the user prompt with context
                    if prim_context:
                        prompt = f"{user_prompt_template}\n\nContext:\n{prim_context}"
                    else:
                        prompt = user_prompt_template

                    # Extract all image paths from renders
                    image_paths = []
                    image_metadata = []
                    render_mode_filter = config.get("render_mode_filter")

                    for render in prim_data.get("renders", []):
                        # Filter by render mode if specified
                        render_mode = render.get("render_mode", "unknown")
                        if render_mode_filter and render_mode not in render_mode_filter:
                            continue

                        render_path = usd_input_dir / render["path"]
                        try:
                            relative_path = render_path.relative_to(dataset_path)
                        except ValueError:
                            relative_path = os.path.relpath(render_path, dataset_path)
                        image_paths.append(str(relative_path))

                        # Store metadata
                        view_name = render.get("view", "unknown")
                        metadata_entry = {
                            "path": str(relative_path),
                            "view": view_name,
                            "camera": render.get("camera", "default"),
                            "render_mode": render_mode,
                        }

                        # Add VLM prompt for this render mode if available
                        if render_mode in vlm_image_prompts:
                            base_prompt = vlm_image_prompts[render_mode]
                            camera_angle = parse_camera_angle_from_view_name(view_name)
                            metadata_entry["vlm_prompt"] = (
                                f"{base_prompt}\n\n"
                                f"Camera Position: Looking from {camera_angle}"
                            )

                        image_metadata.append(metadata_entry)

                    if not image_paths:
                        listener.warning(
                            f"No image paths found for {prim_path}, skipping"
                        )
                        continue

                    # Sort images for consistent ordering (keep metadata aligned)
                    paired = list(zip(image_paths, image_metadata, strict=True))
                    paired.sort(key=lambda x: x[0])
                    listener.debug(f"Using {len(paired)} renders for {prim_path}")

                    # Build data item in v0.2 format
                    # Prepend reference images so VLM sees them first
                    media_images = []
                    for ref_idx, ref_img in enumerate(reference_images):
                        ref_path = Path(ref_img)
                        try:
                            rel_ref = str(ref_path.relative_to(dataset_path))
                        except ValueError:
                            rel_ref = os.path.relpath(ref_path, dataset_path)
                        ref_prompt = (
                            reference_image_prompts_list[ref_idx]
                            if ref_idx < len(reference_image_prompts_list)
                            else "This is a reference image of the asset."
                        )
                        media_images.append(
                            {
                                "path": str(rel_ref),
                                "type": "reference",
                                "metadata": {
                                    "view": "reference",
                                    "camera": "reference",
                                    "render_mode": "reference_image",
                                    "reference_index": ref_idx,
                                    "vlm_prompt": ref_prompt,
                                },
                            }
                        )

                    for img_path, img_meta in paired:
                        image_obj: dict[str, Any] = {
                            "path": img_path,
                            "type": "render",
                        }
                        if img_meta:
                            image_obj["metadata"] = {
                                k: v for k, v in img_meta.items() if k != "path"
                            }
                        media_images.append(image_obj)

                    data_item = {
                        "id": prim_path,
                        "source": {
                            "usd_path": prim_path,
                            "prim_type": _source_prim_type(prim_data),
                        },
                        "user_prompt": prompt,
                        "media": {"images": media_images},
                    }
                    rigid_body_owner_path = rigid_body_source_index.owner_for(prim_path)
                    if rigid_body_source_index.endpoint_paths:
                        data_item["usd_metadata"] = {
                            "structure_provenance": "source_metadata",
                            "rigid_body_owner_path": rigid_body_owner_path,
                            "rigid_body_endpoint_paths": list(
                                rigid_body_source_index.endpoint_paths
                            ),
                        }
                        hierarchy_gaps = rigid_body_source_index.hierarchy_gaps_for(
                            prim_path
                        )
                        if hierarchy_gaps:
                            data_item["usd_metadata"].update(
                                {
                                    "rigid_body_owner_resolution": (
                                        "lexical_ancestor_fallback"
                                    ),
                                    "rigid_body_hierarchy_gap_paths": list(
                                        hierarchy_gaps
                                    ),
                                }
                            )
                    elif rigid_body_source_index.hierarchy_xform_paths:
                        hierarchy_choices, hierarchy_ancestors = (
                            rigid_body_source_index.hierarchy_choices_for(
                                prim_path,
                                path_limit=scene_prim_path_limit,
                            )
                        )
                        if hierarchy_choices:
                            data_item["usd_metadata"] = {
                                "structure_provenance": "source_hierarchy",
                                "hierarchy_xform_paths": list(hierarchy_choices),
                                "hierarchy_ancestor_xform_paths": list(
                                    hierarchy_ancestors
                                ),
                            }

                    # Save individual entry
                    output_file = (
                        output_dir / f"{model_number}_prim_{prim_idx:04d}.json"
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(data_item, f, indent=4)

                    dataset_entries.append(data_item)

                listener.info(
                    f"Prepared {len(dataset_entries)} entries for {model_number}"
                )

            except Exception as e:
                failed_models.append(model_number)
                listener.warning(f"Failed to prepare data for {model_number}: {e}")

        # Save dataset entries
        dataset_jsonl_path = dataset_path / "dataset.jsonl"
        with open(dataset_jsonl_path, "w", encoding="utf-8") as f:
            for entry in dataset_entries:
                f.write(json.dumps(entry) + "\n")
        listener.info(f"Saved dataset to {dataset_jsonl_path}")

        # Create dataset.json (v0.2 format)
        dataset_config = {
            "schema_version": "0.2",
            "metadata": {
                "created": datetime.now().isoformat(),
                "creator": "joint-agent",
                "description": "Asset classification dataset",
                "num_entries": len(dataset_entries),
            },
            "task": {
                "type": "asset_classification",
                "description": "Classify assets based on visual analysis",
            },
            "inference": {
                "prompts": [
                    {
                        "step_name": "classification",
                        "step_index": 0,
                        "system_prompt": system_prompt,
                    }
                ]
            },
            "prims_file": "dataset.jsonl",
        }

        dataset_config_path = dataset_path / "dataset.json"
        with open(dataset_config_path, "w", encoding="utf-8") as f:
            json.dump(dataset_config, f, indent=2)
        listener.info(f"Saved dataset config to {dataset_config_path}")

        # Update context with results
        context["dataset_entries"] = dataset_entries
        context["failed_models"] = failed_models
        context["dataset_jsonl_path"] = str(dataset_jsonl_path)
        context["dataset_config_path"] = str(dataset_config_path)

        return context
