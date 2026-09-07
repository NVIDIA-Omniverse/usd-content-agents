# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for analyzing articulated body structure.

Uses LLM hierarchy analysis (primary) or geometric contact graph (fallback)
to determine segment assignments for each mesh prim.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from joint_agent.functions.provider_response_conformance import (
    ProviderAttemptJournal,
    ProviderResponseConformanceTerminalError,
    WholeAssetStructureEvaluation,
    load_provider_attempt_journal,
    persist_provider_attempt_journal,
    project_provider_attempt_persistence,
    provider_attempt_persistence_error,
    require_provider_attempt_journal_persistence,
    require_whole_asset_structure,
    run_provider_call_with_journal,
)

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


def _require_structure_with_context(
    context: dict[str, Any],
    evaluation: WholeAssetStructureEvaluation | None,
    diagnostics_path: Path,
    *,
    diagnostics_sha256: str | None,
    attempt_diagnostics: tuple[Mapping[str, Any], ...] = (),
    diagnostics_persistence_error: BaseException | None = None,
) -> None:
    """Keep typed status visible across the generic nested Workflow boundary."""
    try:
        require_whole_asset_structure(
            evaluation,
            diagnostics_artifact_path=(
                str(diagnostics_path) if diagnostics_sha256 is not None else None
            ),
            diagnostics_artifact_sha256=diagnostics_sha256,
            attempt_diagnostics=(
                [dict(attempt) for attempt in attempt_diagnostics]
                if attempt_diagnostics
                else None
            ),
            diagnostics_persistence_error=diagnostics_persistence_error,
        )
    except ProviderResponseConformanceTerminalError as error:
        context["provider_response_conformance_terminal_status"] = dict(error.status)
        raise


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
        - structure_provider_response_diagnostics_path: Durable provider-attempt
          journal path
        - structure_provider_response_diagnostics_sha256: SHA-256 of the
          provider-attempt journal
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
        articulation_intended = context.get("articulation_intended", False)
        if not isinstance(articulation_intended, bool):
            raise ValueError("articulation_intended must be a boolean")

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
            return cast(
                str,
                vlm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                    max_tokens=8192,
                ),
            )

        def vlm_generate_with_images(
            system_prompt: str, user_prompt: str, image_paths: list[str]
        ) -> str:
            """Generate with preview images for visual robot identification."""
            image_caption_pairs = [
                ("Rendered preview of the robot.", img_path)
                for img_path in image_paths[:4]  # limit to 4 images
            ]
            return cast(
                str,
                vlm.generate_with_image_caption_pairs(
                    image_caption_pairs=image_caption_pairs,
                    final_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                    max_tokens=4096,
                ),
            )

        assignments: dict[str, str] = {}
        metadata: dict[str, Any] = {}
        diagnostics_path = output_dir / "structure_provider_responses.json"
        resume_enabled = context.get("resume", False)
        if not isinstance(resume_enabled, bool):
            raise ValueError("resume must be a boolean")
        expected_diagnostics_sha256 = context.get(
            "provider_response_diagnostics_expected_sha256"
        )

        def bind_persisted_journal(path: Path, digest: str) -> None:
            project_provider_attempt_persistence(
                context,
                path=path,
                digest=digest,
                path_key="structure_provider_response_diagnostics_path",
                digest_key="structure_provider_response_diagnostics_sha256",
            )

        def bind_terminal(
            error: ProviderResponseConformanceTerminalError,
        ) -> None:
            context["provider_response_conformance_terminal_status"] = dict(
                error.status
            )

        if resume_enabled and (
            diagnostics_path.is_file() or expected_diagnostics_sha256 is not None
        ):
            attempt_journal = load_provider_attempt_journal(
                diagnostics_path,
                on_persisted=bind_persisted_journal,
                expected_sha256=expected_diagnostics_sha256,
                failure_stage="whole_asset_structure_evidence",
                on_terminal=bind_terminal,
            )
            if attempt_journal.snapshot().whole_asset_structure is not None:
                try:
                    attempt_journal.clear_whole_asset_structure()
                except Exception as persistence_error:
                    require_provider_attempt_journal_persistence(
                        attempt_journal,
                        failure_stage="whole_asset_structure_evidence",
                        diagnostics_artifact_path=str(diagnostics_path),
                        on_terminal=bind_terminal,
                        prior_error=(
                            provider_attempt_persistence_error(persistence_error)
                            or persistence_error
                        ),
                    )
        else:
            attempt_journal = ProviderAttemptJournal(
                path=diagnostics_path,
                on_persisted=bind_persisted_journal,
            )
        current_run_first_attempt_sequence = attempt_journal.attempt_count() + 1
        require_provider_attempt_journal_persistence(
            attempt_journal,
            failure_stage="whole_asset_structure_evidence",
            diagnostics_artifact_path=str(diagnostics_path),
            on_terminal=bind_terminal,
        )

        def run_provider_call(
            call: Any,
            *,
            failure_stage: str,
            capture_attempt: bool = False,
            entry_id: str | None = None,
        ) -> Any:
            return run_provider_call_with_journal(
                call,
                journal=attempt_journal,
                failure_stage=failure_stage,
                diagnostics_artifact_path=str(diagnostics_path),
                on_terminal=bind_terminal,
                capture_attempt=capture_attempt,
                entry_id=entry_id,
            )

        def persist_and_require_provider_conformance(
            *,
            prior_persistence_error: BaseException | None = None,
        ) -> None:
            if not attempt_journal.has_evidence():
                if articulation_intended and assignments:
                    _require_structure_with_context(
                        context,
                        None,
                        diagnostics_path,
                        diagnostics_sha256=None,
                    )
                return
            persistence = persist_provider_attempt_journal(
                attempt_journal,
                prior_error=prior_persistence_error,
            )
            current_attempts = persistence.snapshot.attempts[
                current_run_first_attempt_sequence - 1 :
            ]
            if articulation_intended and (
                persistence.snapshot.whole_asset_structure is not None or assignments
            ):
                _require_structure_with_context(
                    context,
                    persistence.snapshot.whole_asset_structure,
                    diagnostics_path,
                    diagnostics_sha256=persistence.artifact_sha256,
                    attempt_diagnostics=current_attempts[-1:],
                    diagnostics_persistence_error=persistence.error,
                )
            if persistence.error is not None:
                require_provider_attempt_journal_persistence(
                    attempt_journal,
                    failure_stage="whole_asset_structure_evidence",
                    diagnostics_artifact_path=str(diagnostics_path),
                    on_terminal=bind_terminal,
                    attempt_diagnostics=current_attempts[-1:],
                    prior_error=persistence.error,
                )

        def accepted_non_articulated_evaluation() -> (
            WholeAssetStructureEvaluation | None
        ):
            evaluation = attempt_journal.snapshot().whole_asset_structure
            if (
                evaluation is not None
                and evaluation.accepted
                and evaluation.dof == 0
                and not evaluation.segment_names
            ):
                return evaluation
            return None

        # Try hierarchy analysis first (unless forced to geometric)
        if strategy in ("auto", "hierarchy"):
            from joint_agent.functions.hierarchy_analysis import analyze_hierarchy

            listener.info("Running LLM hierarchy analysis...")
            assignments, metadata = run_provider_call(
                lambda: analyze_hierarchy(
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
                    articulation_intended=articulation_intended,
                    attempt_journal=attempt_journal,
                ),
                failure_stage="hierarchy_analysis",
            )

            if assignments:
                listener.info(
                    f"Hierarchy analysis succeeded: {len(assignments)} "
                    f"assignments ({metadata.get('hierarchy_pattern')})"
                )

            persist_and_require_provider_conformance()

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
            and accepted_non_articulated_evaluation() is None
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
                        segment_names = run_provider_call(
                            lambda: infer_segment_names(
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
                                articulation_intended=articulation_intended,
                                attempt_journal=attempt_journal,
                            ),
                            failure_stage="segment_inference",
                        )
                        persist_and_require_provider_conformance()
                metadata.update(
                    _merge_prompt_library_metadata(metadata, segment_name_metadata)
                )

            if not segment_names:
                non_articulated = accepted_non_articulated_evaluation()
                if non_articulated is not None:
                    metadata = {
                        **metadata,
                        "strategy": "none",
                        "reason": "provider_reported_zero_dof",
                        "structure_outcome": "not_articulated",
                        "prompt_library_used": bool(
                            metadata.get("prompt_library_used", False)
                        ),
                        "heuristic_paths_used": metadata.get(
                            "heuristic_paths_used", []
                        ),
                        "num_assigned": 0,
                    }
                else:
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
                        "heuristic_paths_used": metadata.get(
                            "heuristic_paths_used", []
                        ),
                        "num_assigned": 0,
                    }
                metadata.update(
                    _merge_prompt_library_metadata(metadata, segment_name_metadata)
                )
            else:
                listener.info("Running geometric contact graph analysis...")
                _, mesh_paths = extract_scene_tree(usd_path)

                def geometric_vlm_generate(
                    system_prompt: str,
                    user_prompt: str,
                ) -> str:
                    return cast(
                        str,
                        run_provider_call(
                            lambda: vlm_generate(system_prompt, user_prompt),
                            failure_stage="geometric_analysis",
                            capture_attempt=True,
                            entry_id="geometric_assignment",
                        ),
                    )

                assignments, geometric_metadata = analyze_geometry(
                    usd_path,
                    mesh_paths,
                    segment_names,
                    geometric_vlm_generate,
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
                persist_and_require_provider_conformance()

        if not assignments:
            listener.warning("No structure assignments produced")

        non_articulated = accepted_non_articulated_evaluation()
        if non_articulated is not None:
            metadata = {
                **metadata,
                "strategy": "none",
                "reason": "provider_reported_zero_dof",
                "structure_outcome": "not_articulated",
                "reasoning": (
                    "Provider-backed whole-asset analysis reported zero degrees "
                    "of freedom and no articulated segments for the identified asset."
                ),
                "evidence": {
                    "accepted": True,
                    "robot_type": non_articulated.robot_type,
                    "dof": non_articulated.dof,
                    "segment_names": list(non_articulated.segment_names),
                    "source_prim_inventory": list(
                        non_articulated.source_prim_inventory
                    ),
                    "provider_response_diagnostics_path": context.get(
                        "structure_provider_response_diagnostics_path"
                    ),
                    "provider_response_diagnostics_sha256": context.get(
                        "structure_provider_response_diagnostics_sha256"
                    ),
                },
                "num_assigned": 0,
            }
            listener.info(
                "Structure analysis completed successfully: asset is not articulated"
            )

        if articulation_intended and not assignments and non_articulated is None:
            evaluation: WholeAssetStructureEvaluation
            journal_snapshot = attempt_journal.snapshot()
            if journal_snapshot.whole_asset_structure is not None:
                evaluation = journal_snapshot.whole_asset_structure
                if evaluation.accepted:
                    evaluation = replace(
                        evaluation,
                        accepted=False,
                        reason_codes=("zero_whole_asset_assignments",),
                    )
            else:
                evaluation = WholeAssetStructureEvaluation(
                    accepted=False,
                    reason_codes=("zero_whole_asset_assignments",),
                    robot_type=None,
                    dof=None,
                    segment_names=tuple(segment_names or ()),
                    source_prim_inventory=(),
                )
            verdict_persistence_error: BaseException | None = None
            try:
                attempt_journal.set_whole_asset_structure(evaluation)
            except Exception as error:
                verdict_persistence_error = (
                    provider_attempt_persistence_error(error) or error
                )
            persist_and_require_provider_conformance(
                prior_persistence_error=verdict_persistence_error
            )

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
