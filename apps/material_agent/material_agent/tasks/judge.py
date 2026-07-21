# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge task for iterative workflows.

Combines two evaluation approaches:
1. **VLM image judge**: Compares rendered output against reference images (color matching)
2. **Prediction analysis**: Checks prediction symmetry and consistency programmatically

The combined score is a weighted average: 40% image judge + 60% prediction analysis.
The image judge must still approve before that blended score can approve an iteration.
"""

import logging
from pathlib import Path
from typing import Any, NamedTuple

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.llm_parsing import (
    extract_labeled_choice,
    extract_labeled_score,
)

from material_agent.tasks.prediction_analyzer import (
    PredictionAnalyzer,
    load_predictions,
    load_prims_metadata,
)

logger = logging.getLogger(__name__)


class _ParsedVlmCritique(NamedTuple):
    decision: str
    score: float
    reasoning: str
    decision_parsed: bool


class _VlmJudgeResult(NamedTuple):
    score: float
    critique: str
    decision: str
    decision_parsed: bool


# Default judge prompt for material assignment evaluation
DEFAULT_JUDGE_PROMPT = """You are an expert judge evaluating material assignment quality in 3D models.

**Your task:**
Evaluate the rendered 3D model for material assignment quality. You are given reference images for loose context — they may show a larger scene, a different object, or a rough style guide. They are NOT exact ground truth.

**Critical Rules — BE CONSERVATIVE:**
1. **Do NOT suggest wholesale color palette changes.** If the current materials look reasonable and internally consistent, score them well even if they don't exactly match the reference.
2. **The reference images are loose guidance only.** They may show an entire factory scene, a different angle, or a different object. Do NOT copy colors from unrelated objects in the reference (e.g., yellow cranes, blue walls) onto the model.
3. **Consistency and symmetry matter most.** Symmetric parts should have the same material. Similar structural parts should share materials. A coherent look is more important than matching a reference.
4. **Preserve what already works.** Only suggest changes for parts that are clearly wrong or inconsistent. If 90% of the model looks good, suggest fixes for the 10% — do NOT ask to redo everything.
5. **Small targeted fixes only.** Never suggest changing more than 3-4 parts at once. Each suggestion should be specific: "Change part X from material A to material B because..."

**Available Materials:**
{materials_list}

**Evaluation Criteria (in order of importance):**

1. **INTERNAL CONSISTENCY (50%)**: Are symmetric parts assigned the same material? Do similar structural elements share materials? Is the overall look coherent?
2. **PLAUSIBILITY (30%)**: Do the material choices make physical sense for the object type? (e.g., metal for structural joints, plastic for casings)
3. **REFERENCE ALIGNMENT (20%)**: Does the general color scheme loosely align with the reference? Only penalize if the colors are wildly different AND the reference clearly shows the same object.

**Provide your evaluation in the following format:**

**Critique:**
[Evaluate in order: consistency, plausibility, then reference alignment.
- Are symmetric/similar parts consistent?
- Do materials make sense for each part type?
- Only if the reference clearly shows the same object: does the palette loosely match?]

**Score:** [0-10 score. 7+ means the assignment is good enough — only fix obvious issues. Score generously if the model looks internally consistent.]

**Decision:** [APPROVE if score >= 7, or CONTINUE if targeted fixes are needed]

**Improvement Suggestions:**
[If CONTINUE, list at most 3-4 specific, conservative changes. Each must explain WHY the change improves consistency or plausibility. Do NOT suggest changes based solely on reference image colors.]
"""


class JudgeTask(Task):
    """Judge that evaluates material assignment quality using VLM.

    This judge uses a Vision Language Model to evaluate the quality of material
    assignments by comparing rendered USD output against reference images.

    Input context keys:
        - iteration_count: Current iteration number
        - materials_applied: Dictionary of applied materials
        - assignment_stats: Statistics from material assignment
        - output_usd_path: Path to the USD file with applied materials
        - rendered_image_paths: List of paths to rendered images (from RenderTask)
        - rendered_image_path: Single rendered image path (backward compatibility)
        - judge_config: Judge configuration dictionary
            - reference_images: List of reference image paths (required)
            - prompt: Optional custom judge prompt
            - temperature: Optional temperature override for VLM
            - max_tokens: Optional max_tokens override for VLM
        - vlm_judge: VLM instance for judge (if configured separately)
        - vlm: VLM instance from predict step (used if vlm_judge not available)
        - vlm_judge_config: VLM Judge configuration dict
        - vlm_config: VLM configuration dict
        - materials_mapping: Dictionary of available materials
        - config_path: Path to config file (for resolving relative image paths)

    Output context keys:
        - continue_iteration: Boolean flag for iteration continuation
        - judge_reasoning: Explanation of the decision (critique from VLM)
        - judge_score: Quality score (0-1)
        - judge_decision: "approve" or "continue"
        - judge_image_decision: Parsed VLM image-judge decision before score blending
        - judge_image_decision_parsed: Whether the VLM image decision field parsed
        - judge_critique: Full critique text from VLM
        - previous_prim_feedback: Dict of per-prim feedback for next iteration
        - symmetry_violations: List of symmetry violations found
        - consistency_violations: List of consistency violations found
        - prediction_consistency_score: Float 0-1 from prediction analysis
    """

    def __init__(self) -> None:
        """Initialize the judge."""
        self.name = "Judge"
        self.description = "Evaluate material assignment quality using VLM"

    def run(
        self,
        context: dict[str, Any],
        object_store: Any = None,
    ) -> dict[str, Any]:
        """Evaluate material assignment quality and decide iteration continuation.

        Runs two evaluation stages:
        1. Prediction analysis (symmetry + consistency) — programmatic, fast
        2. VLM image judge (color matching) — visual, uses VLM

        The raw combined score is retained for telemetry. Final approval still
        requires both a parseable image-judge approval and a score at/above the
        configured threshold.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with judge decision
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Get iteration info
        iteration_count = context.get("iteration_count", 1)
        materials_applied = context.get("materials_applied", {})
        assignment_stats = context.get("assignment_stats", {})

        # Get judge configuration from context
        judge_config = context.get("judge_config", {})

        listener.info("Running judge evaluation...")
        listener.info(f"  Current iteration: {iteration_count}")
        listener.info(
            f"  Materials applied: {len(materials_applied)} to "
            f"{assignment_stats.get('total_prims', 0)} prims"
        )

        # Stage 1: Prediction analysis (symmetry + consistency)
        prediction_analysis_config = judge_config.get("prediction_analysis", {})
        prediction_score: float | None = None
        prediction_critique = ""

        if prediction_analysis_config.get("enabled", True):
            prediction_score, prediction_critique = self._run_prediction_analysis(
                context, prediction_analysis_config
            )
        else:
            listener.info("Prediction analysis disabled in config")

        # Stage 2: VLM image judge
        image_judge_result = self._run_vlm_judge(context, judge_config, iteration_count)
        image_score = image_judge_result.score
        image_critique = image_judge_result.critique
        image_decision = image_judge_result.decision
        image_decision_parsed = image_judge_result.decision_parsed

        # Combine scores — only include prediction analysis if it ran
        if prediction_score is not None:
            prediction_analysis_config = judge_config.get("prediction_analysis", {})
            prediction_weight = prediction_analysis_config.get("weight", 0.6)
            image_weight = 1.0 - prediction_weight
            combined_score = (
                prediction_weight * prediction_score + image_weight * image_score
            )
        else:
            # Prediction analysis disabled — use image score only
            combined_score = image_score
            prediction_score = -1.0  # sentinel for logging

        # Combine critiques
        critique_parts = []
        if prediction_critique:
            critique_parts.append(
                "=== PREDICTION CONSISTENCY ANALYSIS ===\n" + prediction_critique
            )
        if image_critique:
            critique_parts.append("=== VISUAL QUALITY ANALYSIS ===\n" + image_critique)
        combined_critique = "\n\n".join(critique_parts)

        # Require the image judge to approve, then use the blended score threshold.
        score_threshold = judge_config.get("score_threshold", 0.7)
        if image_decision != "approve":
            decision = "continue"
            if not image_decision_parsed:
                listener.warning(
                    "VLM judge decision was unparseable; forcing iteration to continue"
                )
            else:
                listener.info("VLM judge requested another iteration")
        elif combined_score < score_threshold:
            decision = "continue"
        else:
            decision = "approve"

        # Extract reasoning
        reasoning_lines = combined_critique.split("\n")
        reasoning = " ".join(reasoning_lines[:3]).strip()
        if len(reasoning) > 200:
            reasoning = reasoning[:197] + "..."

        # Set context values
        context["continue_iteration"] = decision == "continue"
        context["judge_decision"] = decision
        context["judge_reasoning"] = reasoning
        context["judge_score"] = round(combined_score, 3)
        context["judge_critique"] = combined_critique
        context["judge_image_decision"] = image_decision
        context["judge_image_decision_parsed"] = image_decision_parsed
        context["prediction_consistency_score"] = prediction_score
        context["evaluation_signals"] = {
            "schema_version": "material-self-evaluation-signals/v1",
            "prediction_analysis": {
                "enabled": prediction_score >= 0.0,
                "status": ("completed" if prediction_score >= 0.0 else "disabled"),
                "symmetry_pair_count": None,
                "symmetry_violations": context.get("symmetry_violations", []),
                "consistency_violations": context.get("consistency_violations", []),
                "previous_prim_feedback": context.get("previous_prim_feedback", {}),
                "resolved_assignments": context.get("resolved_assignments", {}),
                "critique": prediction_critique,
            },
            "visual_evaluation": {
                "enabled": True,
                "status": "completed",
                "reference_image_paths": judge_config.get("reference_images", []),
                "rendered_image_paths": context.get(
                    "rendered_image_paths",
                    [context["rendered_image_path"]]
                    if context.get("rendered_image_path")
                    else [],
                ),
                "critique": image_critique,
                "issues": [],
            },
        }

        # Log decision
        listener.info("")
        listener.info("Judge Decision:")
        listener.info(f"  Decision: {decision.upper()}")
        listener.info(f"  Continue iteration: {decision == 'continue'}")
        listener.info(
            f"  Combined score: {combined_score:.3f} "
            f"(prediction: {prediction_score:.3f}, image: {image_score:.3f})"
        )
        listener.info(f"  Reasoning: {reasoning}")
        listener.info("")
        listener.info("Full Critique:")
        for line in combined_critique.split("\n"):
            listener.info(f"  {line}")

        return context

    def _run_prediction_analysis(
        self,
        context: dict[str, Any],
        prediction_analysis_config: dict[str, Any],
    ) -> tuple[float, str]:
        """Run prediction-level symmetry and consistency analysis.

        Args:
            context: Workflow context
            prediction_analysis_config: Config for prediction analysis

        Returns:
            Tuple of (score, critique_text)
        """
        listener = get_listener(context, logger_name=__name__)
        listener.info("Running prediction consistency analysis...")

        predictions_path = context.get("predictions_path")
        if not predictions_path or not Path(predictions_path).exists():
            listener.warning(
                "No predictions file found for analysis. Skipping prediction judge."
            )
            return 1.0, ""

        # Load predictions
        predictions = load_predictions(predictions_path)
        if not predictions:
            listener.warning("Predictions file is empty. Skipping prediction judge.")
            return 1.0, ""

        # Load prim metadata for spatial analysis
        dataset_path = context.get("dataset_path")
        prims_metadata: list[dict[str, Any]] = []
        if dataset_path:
            prims_metadata = load_prims_metadata(dataset_path)

        # Run analysis
        analyzer = PredictionAnalyzer(
            predictions=predictions,
            prims_metadata=prims_metadata,
            symmetry_tolerance=prediction_analysis_config.get(
                "symmetry_tolerance", 5.0
            ),
            consistency_threshold=prediction_analysis_config.get(
                "consistency_threshold", 0.6
            ),
            resolve_symmetry_directly=prediction_analysis_config.get(
                "resolve_symmetry_directly", True
            ),
            resolve_consistency_directly=prediction_analysis_config.get(
                "resolve_consistency_directly", True
            ),
            detect_numbered_path_symmetry=prediction_analysis_config.get(
                "detect_numbered_path_symmetry", True
            ),
        )
        result = analyzer.analyze()

        # Store detailed results in context
        context["symmetry_violations"] = [
            {
                "prim_a": v.prim_a,
                "prim_b": v.prim_b,
                "material_a": v.material_a,
                "material_b": v.material_b,
                "suggested": v.suggested,
            }
            for v in result.symmetry_violations
        ]
        context["consistency_violations"] = [
            {
                "group_name": v.group_name,
                "prims": v.prims,
                "materials": v.materials,
                "suggested": v.suggested,
            }
            for v in result.consistency_violations
        ]
        context["previous_prim_feedback"] = result.prim_feedback
        context["resolved_assignments"] = result.resolved_assignments

        # Log summary
        listener.info(f"  Symmetric pairs detected: {len(result.symmetry_pairs)}")
        listener.info(f"  Symmetry violations: {len(result.symmetry_violations)}")
        listener.info(f"  Consistency violations: {len(result.consistency_violations)}")
        listener.info(f"  Prediction consistency score: {result.score:.3f}")
        if result.resolved_assignments:
            listener.info(
                f"  Resolved assignments for {len(result.resolved_assignments)} prims "
                f"(will be applied directly, no VLM re-prediction needed)"
            )
        if result.prim_feedback:
            listener.info(
                f"  Per-prim feedback generated for {len(result.prim_feedback)} prims"
            )

        return result.score, result.critique

    def _run_vlm_judge(
        self,
        context: dict[str, Any],
        judge_config: dict[str, Any],
        iteration_count: int,
    ) -> _VlmJudgeResult:
        """Run VLM-based visual evaluation of material assignment.

        Args:
            context: Workflow context
            judge_config: Judge configuration
            iteration_count: Current iteration number

        Returns:
            Tuple of (score, critique_text, parsed_decision)
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Get VLM instance - prefer vlm_judge if available, otherwise use vlm
        vlm = context.get("vlm_judge") or context.get("vlm")
        if not vlm:
            listener.error("VLM not available for judge evaluation")
            raise ValueError(
                "VLM is required for judge evaluation but not found in context. "
                "Make sure VLM is configured in the predict section or "
                "judge.vlm section."
            )

        # Determine which VLM config to use for defaults
        if context.get("vlm_judge"):
            vlm_config = context.get("vlm_judge_config", {})
            listener.debug("Using dedicated judge VLM")
        else:
            vlm_config = context.get("vlm_config", {})
            listener.debug("Using predict VLM for judge")

        # Get rendered images
        rendered_image_paths = context.get("rendered_image_paths", [])
        if not rendered_image_paths:
            # Fall back to single image for backward compatibility
            rendered_image_path = context.get("rendered_image_path")
            if rendered_image_path:
                rendered_image_paths = [rendered_image_path]

        if not rendered_image_paths:
            listener.error(
                "No rendered images found for judge evaluation. "
                "Make sure rendering is enabled in the configuration."
            )
            raise ValueError(
                "Rendered images are required for judge evaluation but not found. "
                "Enable rendering in the refine step: refine.render.enabled = true"
            )

        # Get reference images
        reference_images = judge_config.get("reference_images", [])
        if not reference_images:
            listener.error(
                "No reference images provided for judge evaluation. "
                "Add reference_images to the judge configuration."
            )
            raise ValueError(
                "Reference images are required for judge evaluation. "
                "Add reference_images list to the judge section in your config."
            )

        # Resolve reference image paths (may be relative to config)
        config_path = context.get("config_path")
        if config_path:
            config_dir = Path(config_path).parent
            reference_images = [
                str(config_dir / ref) if not Path(ref).is_absolute() else ref
                for ref in reference_images
            ]

        # Validate image paths
        valid_reference_images = []
        for ref_img in reference_images:
            if Path(ref_img).exists():
                valid_reference_images.append(ref_img)
            else:
                listener.warning(f"Reference image not found: {ref_img}")

        if not valid_reference_images:
            raise ValueError(
                "No valid reference images found. Check paths in judge configuration."
            )

        valid_rendered_images = []
        for rend_img in rendered_image_paths:
            if Path(rend_img).exists():
                valid_rendered_images.append(rend_img)
            else:
                listener.warning(f"Rendered image not found: {rend_img}")

        if not valid_rendered_images:
            raise ValueError(
                "No valid rendered images found. Check rendering configuration."
            )

        # Build materials list for prompt
        materials_mapping = context.get("materials_mapping", {})
        if materials_mapping:
            materials_list = "\n".join(
                [f"- {name}" for name in sorted(materials_mapping.keys())]
            )
        else:
            materials_list = "(No materials list available)"

        # Get or build judge prompt
        judge_prompt_template = judge_config.get("prompt", DEFAULT_JUDGE_PROMPT)
        judge_prompt = judge_prompt_template.format(materials_list=materials_list)

        # Build image-caption pairs for VLM
        image_caption_pairs = []

        # Add reference images first
        for i, ref_img in enumerate(valid_reference_images, 1):
            caption = f"Reference Image {i}:"
            image_caption_pairs.append((caption, ref_img))

        # Add rendered images
        for i, rend_img in enumerate(valid_rendered_images, 1):
            caption = f"Rendered 3D Model (Current Result) - View {i}:"
            image_caption_pairs.append((caption, rend_img))

        # Get VLM configuration (temperature and max_tokens can override)
        temperature = judge_config.get(
            "temperature", vlm_config.get("temperature", 0.1)
        )
        max_tokens = judge_config.get("max_tokens", vlm_config.get("max_tokens", 2048))

        listener.info("Evaluating material assignment with VLM judge...")
        listener.info(f"  Reference images: {len(valid_reference_images)}")
        listener.info(f"  Rendered images: {len(valid_rendered_images)}")
        listener.info(f"  Temperature: {temperature}")
        listener.info(f"  Max tokens: {max_tokens}")

        try:
            # Use VLM to evaluate
            critique = vlm.generate_with_image_caption_pairs(
                image_caption_pairs=image_caption_pairs,
                final_prompt=judge_prompt,
                system_prompt="You are an expert judge evaluating 3D material assignments.",
                temperature=temperature,
                max_tokens=max_tokens,
            )

            listener.info("VLM image evaluation complete")
            listener.debug(f"Raw critique: {critique}")

            # Parse the critique to extract score and fail-closed image decision.
            parsed_critique = self._parse_vlm_critique(
                context,
                critique,
                iteration_count,
                score_threshold=judge_config.get("score_threshold", 0.7),
            )
            decision = parsed_critique.decision

            return _VlmJudgeResult(
                score=parsed_critique.score,
                critique=critique,
                decision=decision,
                decision_parsed=parsed_critique.decision_parsed,
            )

        except Exception as e:
            listener.error(f"VLM judge evaluation failed: {e}", exc_info=True)
            raise RuntimeError(
                f"Judge evaluation failed: {e}. "
                "Check VLM configuration and that all images are accessible."
            ) from e

    def _parse_vlm_critique(
        self,
        context: dict[str, Any],
        critique: str,
        iteration_count: int,
        *,
        score_threshold: float = 0.7,
    ) -> _ParsedVlmCritique:
        """Parse VLM critique to extract decision and score.

        Args:
            context: Workflow context
            critique: Raw critique text from VLM
            iteration_count: Current iteration number

        Returns:
            Parsed critique fields:
            - decision: "approve" or "continue"
            - score: 0-1 score
            - reasoning: Short summary for logging
            - decision_parsed: Whether the response supplied a parseable decision
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        parsed_score = extract_labeled_score(critique)
        if parsed_score is None:
            score = 0.5
            listener.warning("Could not extract VLM judge score; defaulting to 0.5")
        else:
            score = parsed_score
            listener.debug(f"Extracted score: {score}")

        # Try to extract decision
        decision_value = extract_labeled_choice(
            critique,
            "Decision",
            ("continue", "approve"),
            boundary_labels=(
                "Critique",
                "Improvement Suggestion",
                "Improvement Suggestions",
                "Recommendation",
                "Recommendations",
                "Score",
            ),
        )
        if decision_value:
            decision = decision_value
            decision_parsed = True
        else:
            decision = "continue"
            decision_parsed = False
            listener.warning(
                "Could not extract VLM judge decision; defaulting to 'continue'"
            )

        # Also consider score threshold.
        if score < score_threshold:
            decision = "continue"
            listener.debug(
                f"Score {score} < {score_threshold}, setting decision to 'continue'"
            )

        # Extract reasoning (first few sentences of critique)
        reasoning_lines = critique.split("\n")
        reasoning = " ".join(reasoning_lines[:3]).strip()
        if len(reasoning) > 200:
            reasoning = reasoning[:197] + "..."

        return _ParsedVlmCritique(
            decision=decision,
            score=score,
            reasoning=reasoning,
            decision_parsed=decision_parsed,
        )
