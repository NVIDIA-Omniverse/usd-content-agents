# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for analyzing articulated body structure.

Uses LLM hierarchy analysis (primary) or geometric contact graph (fallback)
to determine segment assignments for each mesh prim.
"""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)

_PROMPT_LIBRARY_METADATA_KEYS = (
    "prompt_library_robot_id",
    "prompt_library_match_source",
    "prompt_library_error",
)


def _merge_prompt_library_metadata(*sources: dict[str, Any]) -> dict[str, Any]:
    """Merge prompt-library provenance without letting later False clobber True."""
    merged: dict[str, Any] = {}
    has_prompt_metadata = False
    prompt_library_used = False
    for source in sources:
        if "prompt_library_used" in source:
            has_prompt_metadata = True
            prompt_library_used = prompt_library_used or bool(
                source["prompt_library_used"]
            )
        for key in _PROMPT_LIBRARY_METADATA_KEYS:
            if key in source:
                has_prompt_metadata = True
                merged[key] = source[key]

    if has_prompt_metadata:
        merged["prompt_library_used"] = prompt_library_used
    return merged


class AnalyzeStructureTask(Task):
    """Analyze USD hierarchy structure to assign segment names.

    Tries LLM hierarchy analysis first. If the hierarchy is uninformative,
    falls back to geometric contact graph analysis.

    Input context keys:
        - usd_path: Path to USD file
        - output_dir: Output directory
        - strategy: "auto" | "hierarchy" | "geometric" | "skip"
        - segment_names: Optional list of segment names (auto-inferred if None)
        - use_prompt_library: Optional prompt-library opt-in for known robots
        - robot_id: Optional explicit prompt-library robot ID
        - vlm: VLM instance (from ModelProvisioningTask)
        - llm: LLM instance (optional, for fallback parsing)

    Output context keys:
        - structure_assignments: Dict mapping prim_path -> segment_name
        - structure_assignments_path: Path to saved assignments JSON
        - structure_metadata: Strategy details
    """

    def __init__(self) -> None:
        self.name = "AnalyzeStructure"
        self.description = "Analyze articulated body structure"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        usd_path = context["usd_path"]
        output_dir = Path(context["output_dir"])
        strategy = context.get("strategy", "auto")
        segment_names = context.get("segment_names")  # None = auto-infer
        vlm = context.get("vlm")
        if vlm is None:
            raise ValueError(
                "VLM not provided in context. "
                "ModelProvisioningTask must run before AnalyzeStructure."
            )
        asset_type = context.get("asset_type")
        asset_subtype = context.get("asset_subtype")
        asset_confidence = context.get("asset_confidence")
        use_prompt_library = context.get("use_prompt_library", False)
        if not isinstance(use_prompt_library, bool):
            raise ValueError("use_prompt_library must be a boolean")
        robot_id = context.get("robot_id")
        identification_path = context.get("identification_path")

        # Find preview images from identify_asset step
        preview_images: list[str] = []
        if identification_path:
            preview_dir = Path(identification_path).parent / "preview"
            if preview_dir.exists():
                preview_images = sorted(str(p) for p in preview_dir.glob("*.png"))
                if preview_images:
                    listener.info(
                        f"Found {len(preview_images)} preview images "
                        f"from identify_asset"
                    )

        seg_info = f"{len(segment_names)} segments" if segment_names else "auto-infer"
        listener.info(
            f"Analyzing structure: {usd_path} (strategy={strategy}, {seg_info})"
        )

        if strategy == "skip":
            listener.info("Strategy is 'skip' — no structure analysis")
            context["structure_assignments"] = {}
            context["structure_metadata"] = {
                "strategy": "skip",
                "prompt_library_used": False,
                "heuristic_paths_used": [],
            }
            return context

        # Build VLM generate functions from VLM instance
        def vlm_generate(system_prompt: str, user_prompt: str) -> str:
            return vlm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=8192,
            )

        def vlm_generate_with_images(
            system_prompt: str, user_prompt: str, image_paths: list[str]
        ) -> str:
            """Generate with preview images for visual robot identification."""
            image_caption_pairs = [
                ("Rendered preview of the robot.", img_path)
                for img_path in image_paths[:4]  # limit to 4 images
            ]
            return vlm.generate_with_image_caption_pairs(
                image_caption_pairs=image_caption_pairs,
                final_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=4096,
            )

        assignments: dict[str, str] = {}
        metadata: dict[str, Any] = {}

        # Try hierarchy analysis first (unless forced to geometric)
        if strategy in ("auto", "hierarchy"):
            from joint_agent.functions.hierarchy_analysis import analyze_hierarchy

            listener.info("Running LLM hierarchy analysis...")
            assignments, metadata = analyze_hierarchy(
                usd_path,
                segment_names,
                vlm_generate,
                asset_type=asset_type,
                asset_subtype=asset_subtype,
                asset_confidence=asset_confidence,
                vlm_generate_with_images_fn=(
                    vlm_generate_with_images if preview_images else None
                ),
                preview_images=preview_images or None,
                use_prompt_library=use_prompt_library,
                robot_id=robot_id,
            )

            if assignments:
                listener.info(
                    f"Hierarchy analysis succeeded: {len(assignments)} "
                    f"assignments ({metadata.get('hierarchy_pattern')})"
                )

        prompt_library_incomplete = (
            metadata.get("reason") == "prompt_library_incomplete"
        )
        prompt_library_error: str | None = None
        if prompt_library_incomplete and strategy in ("auto", "hierarchy"):
            prompt_library_error = (
                "Prompt-library hierarchy output was incomplete; "
                "structure analysis cannot continue safely"
            )

        # Fall back to geometric if hierarchy failed or was skipped. Prompt-library
        # incompleteness is terminal so regressions remain visible to callers.
        if (
            not assignments
            and strategy in ("auto", "geometric")
            and not (strategy == "auto" and prompt_library_incomplete)
        ):
            from joint_agent.functions.geometric_analysis import (
                analyze_geometry,
            )
            from joint_agent.functions.hierarchy_analysis import (
                _resolve_prompt_library_entry,
                extract_scene_tree,
                infer_segment_names,
            )

            # Ensure segment_names is not None for geometric fallback.
            # If hierarchy analysis inferred them, use those; otherwise
            # infer now (the hierarchy step may have been skipped).
            segment_name_metadata: dict[str, Any] = {}
            if not segment_names and metadata.get("segment_names"):
                segment_names = metadata["segment_names"]
            if not segment_names:
                hierarchy_inference_exhausted = bool(metadata.get("segments_inferred"))
                if not hierarchy_inference_exhausted:
                    prompt_entry, segment_name_metadata = _resolve_prompt_library_entry(
                        use_prompt_library=use_prompt_library,
                        robot_id=robot_id,
                        asset_type=asset_type,
                        asset_subtype=asset_subtype,
                        asset_confidence=asset_confidence,
                        usd_path=usd_path,
                    )
                    if prompt_entry:
                        segment_names = list(prompt_entry.component_names)
                        segment_name_metadata["prompt_library_used"] = True
                    else:
                        segment_names = infer_segment_names(
                            usd_path,
                            vlm_generate,
                            asset_type,
                            asset_subtype,
                            vlm_generate_with_images_fn=(
                                vlm_generate_with_images if preview_images else None
                            ),
                            preview_images=preview_images or None,
                            asset_confidence=asset_confidence,
                            use_prompt_library=False,
                            robot_id=None,
                        )
                metadata.update(
                    _merge_prompt_library_metadata(metadata, segment_name_metadata)
                )

            if not segment_names:
                listener.warning(
                    "No segment names available; geometric fallback requires "
                    "model-inferred, prompt-library, or configured segment names"
                )
                metadata = {
                    **metadata,
                    "strategy": "none",
                    "reason": "segment_names_unresolved",
                    "prompt_library_used": bool(
                        metadata.get("prompt_library_used", False)
                    ),
                    "heuristic_paths_used": metadata.get("heuristic_paths_used", []),
                    "num_assigned": 0,
                }
                metadata.update(
                    _merge_prompt_library_metadata(metadata, segment_name_metadata)
                )
            else:
                listener.info("Running geometric contact graph analysis...")
                _, mesh_paths = extract_scene_tree(usd_path)
                assignments, geometric_metadata = analyze_geometry(
                    usd_path,
                    mesh_paths,
                    segment_names,
                    vlm_generate,
                )
                metadata = {
                    **metadata,
                    **geometric_metadata,
                    **_merge_prompt_library_metadata(
                        metadata,
                        segment_name_metadata,
                        geometric_metadata,
                    ),
                }

                if assignments:
                    listener.info(
                        f"Geometric analysis succeeded: {len(assignments)} assignments"
                    )

        if not assignments:
            listener.warning("No structure assignments produced")

        # Save assignments
        output_path = output_dir / "structure_assignments.json"
        output_data = {
            "strategy_used": metadata.get("strategy", "none"),
            "metadata": metadata,
            "segment_names": segment_names,
            "assignments": {
                path: {"component_name": name} for path, name in assignments.items()
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        listener.info(f"Saved structure assignments to {output_path}")

        # Update context
        context["structure_assignments"] = assignments
        context["structure_assignments_path"] = str(output_path)
        context["structure_metadata"] = metadata

        if prompt_library_error:
            message = f"{prompt_library_error}; diagnostics written to {output_path}"
            listener.error(message)
            raise RuntimeError(message)

        return context
