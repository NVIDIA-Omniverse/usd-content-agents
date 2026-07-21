# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt builders for content-workflow-cli workflows."""

from __future__ import annotations

import json
from pathlib import Path

CHILD_MATERIAL_ASSIGNMENT_ARTIFACTS = (
    "raw/material_decision_patch.json",
    "trace/",
)
WRAPPER_MATERIAL_ASSIGNMENT_ARTIFACTS = (
    "assignments.json",
    "visual_quality_assessment.json",
    "api_operation_counts.json",
    "validation_evidence.json",
    "final_summary.md",
)
CHILD_PHYSICS_APPLY_ARTIFACTS = (
    "raw/physics_decision_patch.json",
    "trace/",
)
WRAPPER_PHYSICS_APPLY_ARTIFACTS = (
    "physics_assignments.json",
    "physics_behavior_assessment.json",
    "validation_evidence.json",
    "final_summary.md",
    "runtime/",
)
DEFAULT_MATERIAL_CANDIDATE_POLICY: dict[str, object] = {
    "material_candidate_space": "source",
    "root_prim_path": None,
    "skip_instances": True,
    "skip_prototypes": False,
    "skip_invisible": False,
}


def _material_candidate_policy(
    explicit_policy: dict[str, object] | None,
    preflight_packet: dict[str, object] | None = None,
) -> dict[str, object]:
    packet_policy = None
    if isinstance(preflight_packet, dict):
        raw_packet_policy = preflight_packet.get("material_candidate_policy")
        if isinstance(raw_packet_policy, dict):
            packet_policy = raw_packet_policy
    raw_policy = packet_policy or explicit_policy or {}
    policy = dict(DEFAULT_MATERIAL_CANDIDATE_POLICY)
    policy.update({str(key): value for key, value in raw_policy.items()})
    if policy.get("material_candidate_space") not in {"source", "inspection"}:
        policy["material_candidate_space"] = "source"
    return policy


def build_physics_apply_prompt(
    *,
    repo_root: Path,
    run_dir: Path,
    usd_path: Path,
    workbench_url: str,
    session_id: str,
    reference_images: list[Path] | None = None,
    reference_files: list[Path] | None = None,
    additional_instructions: str | None = None,
    collision_approximation: str = "convexHull",
    visual_validation_max_iterations: int = 3,
) -> str:
    """Build the initial physics decision-patch prompt."""

    task = {
        "schema_version": "content-agents.physics-apply-task.v2",
        "workflow": "physics.apply",
        "required_skills": ["content-workbench", "content-workflow-physics"],
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "asset_path": str(usd_path),
        "reference_images": [str(path) for path in reference_images or []],
        "reference_files": [str(path) for path in reference_files or []],
        "workbench": {
            "endpoint": workbench_url,
            "session_id": session_id,
            "transport": "rest",
            "physics_run_packet_path": str(run_dir / "raw" / "physics_run_packet.json"),
            "components_path": str(run_dir / "raw" / "physics_components.json"),
            "topology_path": str(run_dir / "raw" / "physics_topology.json"),
        },
        "constraints": {
            "source_usd_edits_allowed": False,
            "collision_approximation_default": collision_approximation,
            "visual_validation_max_iterations": visual_validation_max_iterations,
        },
        "child_required_artifacts": list(CHILD_PHYSICS_APPLY_ARTIFACTS),
        "wrapper_final_artifacts": list(WRAPPER_PHYSICS_APPLY_ARTIFACTS),
    }
    if additional_instructions and additional_instructions.strip():
        task["additional_instructions"] = additional_instructions.strip()

    return f"""You are running a skill-routed agentic physics authoring workflow.

Load and follow these skills:
- `content-workbench`
- `content-workflow-physics`

Use the prepared Workbench session and component packet. The child agent owns
physics reasoning, the decision patch, and any topology-plan proposal; the
wrapper owns topology-plan application, schema application, ovphysx runtime
validation, frame rendering, visual review orchestration, canonical final
artifacts, and final validation evidence.
ovphysx/runtime metrics are authoritative for hard physics validation failures;
visual behavior review can make otherwise-passing results conditional but cannot
override a runtime failure.

Structured task:
```json
{json.dumps(task, indent=2)}
```

Decision task:
- Inspect `raw/physics_components.json`, `raw/physics_topology.json`, and any
  reference evidence. Treat visual, collider, and helper paths as distinct roles.
- For each logical component, write exactly one `decisions` entry or one
  `unresolved_components` entry. Never target helper paths.
- Infer density, estimated mass, static/dynamic friction, restitution, collider
  approximation, confidence, and rationale.
- Write `{run_dir}/raw/physics_decision_patch.json`. If and only if user intent
  resolves mobility and a topology repair is required, also write
  `{run_dir}/raw/physics_topology_plan.json` using the source digest and
  invariants from inspection. Otherwise preserve topology.

Patch schema:
```json
{{
  "schema_version": "content-agent-workflows.physics-decision-patch.v2",
  "asset": "{usd_path}",
  "source_digest": "copy from raw/physics_components.json",
  "decisions": [
    {{
      "decision_id": "stable-id",
      "component_id": "component_001",
      "body_root_path": "/Asset/Body",
      "visual_evidence_paths": ["/Asset/Body/Visual"],
      "collider_paths": ["/Asset/Body/Collision"],
      "collision_mode": "preserve_existing|author_on_targets",
      "mass_authoring_path": "/Asset/Body",
      "inferred_material_family": "glass|metal|plastic|rubber|wood|generic",
      "inferred_material_name": "optional existing material name or null",
      "collision_approximation": "{collision_approximation}",
      "physical_properties": {{
        "density": 1000.0,
        "estimated_mass_kg": 0.1,
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.1
      }},
      "confidence": 0.7,
      "rationale": "reasoning grounded in component roles, material, bounds, and topology"
    }}
  ],
  "unresolved_components": [
    {{"component_id": "component_002", "reason": "specific missing evidence"}}
  ]
}}
```

Do not write `physics_assignments.json`, `validation_evidence.json`,
`physics_behavior_assessment.json`, final summaries, or runtime validation
artifacts in this turn. The wrapper will apply the patch, run ovphysx, render
the simulation recording, and launch visual behavior review turns.

Finish with a short response pointing to the run directory and decision patch.
"""


def build_physics_visual_refinement_prompt(
    *,
    repo_root: Path,
    run_dir: Path,
    usd_path: Path,
    workbench_url: str,
    session_id: str,
    iteration: int,
    max_iterations: int,
    decision_patch_path: Path,
    validation_evidence_path: Path,
    runtime_report_path: Path | None,
    rendered_frames: list[str],
    previous_assessment_path: Path | None = None,
    issue_packet_path: Path | None = None,
) -> str:
    """Build a post-runtime visual behavior review/refinement prompt."""

    task = {
        "schema_version": "content-agents.physics-visual-review-task.v1",
        "workflow": "physics.apply.visual_review",
        "required_skills": ["content-workbench", "content-workflow-physics"],
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "asset_path": str(usd_path),
        "workbench": {
            "endpoint": workbench_url,
            "session_id": session_id,
        },
        "iteration": iteration,
        "max_iterations": max_iterations,
        "decision_patch_path": str(decision_patch_path),
        "validation_evidence_path": str(validation_evidence_path),
        "runtime_report_path": str(runtime_report_path)
        if runtime_report_path
        else None,
        "rendered_frames": rendered_frames,
        "previous_assessment_path": (
            str(previous_assessment_path) if previous_assessment_path else None
        ),
        "issue_packet_path": str(issue_packet_path) if issue_packet_path else None,
    }

    return f"""You are reviewing rendered physics simulation behavior for an
agentic physics authoring run.

Load and follow these skills:
- `content-workbench`
- `content-workflow-physics`

This is visual validation/refinement iteration {iteration} of {max_iterations}.
ovphysx/runtime metrics are authoritative for hard failures. Your visual review
is a semantic check over the rendered simulation frames and runtime report.

Structured task:
```json
{json.dumps(task, indent=2)}
```

Review contract:
- Inspect the rendered frame images directly.
- Read the runtime report and validation evidence.
- Check for parts visibly separating or moving independently when they should be
  one rigid object, no visible motion under gravity, implausible bounce/sliding,
  obvious interpenetration/tunneling, stale or blank renders, misframed renders,
  and mismatches between runtime metrics and visible behavior.
- Write `{run_dir}/physics_behavior_assessment.json` with:
  `schema_version`, `status`, `checked_views`, `runtime_report`,
  `rendered_frames`, `issues_found`, `issues_fixed`, `unresolved_issues`, and
  `assessment_notes`.
- Use `status: "pass"` when the behavior is plausible, `status: "fixed"` when
  you updated `raw/physics_decision_patch.json` to address a fixable issue, and
  `status: "unresolved_issues"` when issues remain after the available fix.
- If a visual issue is fixable by changing body grouping, collider
  approximation, density/mass, friction, or restitution, update
  `raw/physics_decision_patch.json` in the same schema as the initial patch.
- Do not edit source USD files or write canonical final artifacts. The wrapper
  will reapply any changed patch, rerun ovphysx, rerender, and merge evidence.

Finish with a short response listing visual status, any patch changes, and
remaining limitations.
"""


def build_skill_routed_material_assignment_prompt(
    *,
    repo_root: Path,
    run_dir: Path,
    usd_path: Path,
    reference_images: list[Path],
    materials_yaml: Path,
    materials_usd: Path,
    workbench_url: str,
    reference_files: list[Path] | None = None,
    optimize: bool = True,
    optimizer_options: dict[str, object] | None = None,
    material_candidate_policy: dict[str, object] | None = None,
    respect_existing_material_bindings: bool = False,
    additional_instructions: str | None = None,
    preflight_packet: dict[str, object] | None = None,
    vqa_refinement_max_iterations: int = 3,
) -> str:
    """Build a compact child-agent prompt that routes method through skills."""

    packet_path = run_dir / "raw" / "material_run_packet.json"
    preflight_enabled = preflight_packet is not None
    candidate_policy = _material_candidate_policy(
        material_candidate_policy,
        preflight_packet,
    )
    image_inputs = []
    if preflight_enabled:
        initial_renders = preflight_packet.get("initial_evidence_renders")
        if isinstance(initial_renders, list):
            for record in initial_renders:
                if isinstance(record, dict) and record.get("image_path"):
                    image_inputs.append(
                        {
                            "label": f"Workbench initial render: {record.get('name')}",
                            "path": str(record.get("image_path")),
                        }
                    )
    task = {
        "schema_version": "content-agents.skill-routed-task.v1",
        "workflow": "materials.assign",
        "required_skills": [
            "content-workbench",
            "content-workflow-material",
        ],
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "asset_path": str(usd_path),
        "reference_images": [str(path) for path in reference_images],
        "reference_files": [str(path) for path in reference_files or []],
        "material_library": {
            "materials_yaml": str(materials_yaml),
            "materials_usd": str(materials_usd),
        },
        "workbench": {
            "endpoint": workbench_url,
            "transport": "rest",
            "optimize": optimize,
            "optimizer_options": optimizer_options or {},
            "preflight_packet_path": str(packet_path) if preflight_enabled else None,
            "preflight_initial_render_inputs": image_inputs,
        },
        "material_candidate_policy": candidate_policy,
        "constraints": {
            "use_workbench_only": True,
            "source_usd_edits_allowed": False,
            "respect_existing_material_bindings": respect_existing_material_bindings,
            "clear_materials": not respect_existing_material_bindings,
            "predict_only_canonical_material_candidates": True,
            "vqa_refinement_max_iterations": vqa_refinement_max_iterations,
        },
        "child_required_artifacts": list(CHILD_MATERIAL_ASSIGNMENT_ARTIFACTS),
        "wrapper_final_artifacts": list(WRAPPER_MATERIAL_ASSIGNMENT_ARTIFACTS),
    }
    if additional_instructions and additional_instructions.strip():
        task["additional_instructions"] = additional_instructions.strip()

    preflight_note = (
        "A Workbench material-run packet has already been prepared. Reuse it; do "
        "not refetch Workbench docs, recreate the session, resnapshot the scene, "
        "or rerender initial evidence unless a targeted ambiguity requires it."
        if preflight_enabled
        else "No preflight packet was prepared. Use the skills to start or connect "
        "to Workbench, create the session, inspect the scene, and gather evidence."
    )

    return f"""You are running a skill-routed agentic asset workflow.

Load and follow these skills:
- `content-workbench`
- `content-workflow-material`

Use the skills as the source of truth for Workbench API mechanics, object/path
mapping, and material assignment policy. In this wrapper-mediated batch mode,
the child agent writes the material decision patch and focused evidence; the
wrapper owns bounded VQA refinement orchestration, canonical final artifacts,
and final validation evidence after the child exits.

{preflight_note}

Structured task:
```json
{json.dumps(task, indent=2)}
```

Required behavior:
- Inspect attached reference images directly before making material decisions.
- Inspect non-image reference files through targeted local file reads.
- Treat `raw/visible_candidate_prims.json` as the canonical material candidate
  universe. With the default source policy, predict source/prototype targets and
  use runtime paths only as visual evidence or preview fan-out; do not predict
  separate instance-proxy rows when they collapse to the same source target.
- Apply material decisions through Workbench, not by editing source USD files.
- Write `raw/material_decision_patch.json`; the wrapper finalizer owns canonical
  `assignments.json`, `visual_quality_assessment.json`,
  `api_operation_counts.json`, `validation_evidence.json`, final renders, and
  `final_summary.md`.
- Preserve concise trace evidence for any targeted extra renders, picks, or
  ambiguity resolution.

Finish with a short response pointing to the run directory and the decision
patch or evidence artifacts you wrote.
"""


def build_material_optimizer_selection_prompt(
    *,
    asset_path: Path,
    run_dir: Path,
    analysis_run_dir: Path,
    workbench_url: str,
    session_id: str,
    decision_path: Path,
    additional_instructions: str | None = None,
) -> str:
    """Build the agent turn that chooses per-asset optimizer settings."""

    extra = (
        additional_instructions.strip()
        if additional_instructions and additional_instructions.strip()
        else "none"
    )
    return f"""You are selecting Content Workbench Scene Optimizer settings for a material-assignment workflow.

Inspect the supplied unoptimized Workbench evidence and choose settings for the `material_assignment` task. Optimize for complete visible-material coverage, independently assignable appearance regions, and reliable source-space authoring. This is a task-and-asset decision, not a file-extension lookup or a fixed preset.

Inputs:
- Asset: `{asset_path}`
- Workflow run: `{run_dir}`
- Unoptimized analysis packet: `{analysis_run_dir / "raw" / "material_run_packet.json"}`
- Compact scene context: `{analysis_run_dir / "raw" / "material_authoring_context.md"}`
- Candidate table: `{analysis_run_dir / "raw" / "visible_candidate_table.tsv"}`
- Workbench endpoint: `{workbench_url}`
- Unoptimized session: `{session_id}`
- Additional task guidance: {extra}

Decision criteria:
- Enable optimization only when it improves inspection or path handling for this asset.
- Flatten prototypes when prototype composition otherwise hides or aliases visible material-authoring targets, but preserve reusable source inheritance when it already provides correct assignment fan-out.
- Deinstance when instances need different visible materials or instance proxies prevent legal assignment; preserve instancing when repeated parts intentionally share appearance.
- Treat prototype candidates plus visible runtime instance proxies as an unresolved
  authoring risk, not proof that no optimization is needed. Before selecting a
  no-op for such a scene, establish evidence that every visible runtime region
  maps to a legal source target and that source-space authoring propagates to
  the intended instances. If that correspondence is absent or aliased, compare
  a flatten/deinstance variant and prefer it when it restores reliable preview
  and durable assignment behavior.
- Enable splitting only when merged geometry prevents separate assignment of visibly distinct material regions. Disable it when it merely fragments one coherent appearance region.
- Enable deduplication only when geometrically repeated parts also share material intent. Disable it when similar geometry needs independent color, finish, or source identity.
- Preserve source-to-inspection path correspondence and avoid collapsing distinct material candidates.
- Judge variants by material candidate coverage, rejected/ambiguous target count, source-path translation, and whether one override affects only its intended visible region.
- Use the attached unoptimized renders and compact topology/candidate evidence.
  For prototype- or instance-heavy scenes, targeted path translation, pick, or
  optimizer-variant queries are required unless the supplied packet already
  proves complete source-to-runtime correspondence.
- The analysis session is disposable. You may use `/scene/optimize`, reload the source, and compare targeted optimizer variants when structural evidence alone is insufficient. Record any trial settings and observed candidate/path effects in `evidence`.

Do not edit the source USD or assign materials during this turn. Write exactly one decision artifact to `{decision_path}` with this shape:

```json
{{
  "schema_version": "content-agents.optimizer-decision.v1",
  "task": "material_assignment",
  "optimize": true,
  "flatten_prototypes": true,
  "enable_deinstance": true,
  "enable_split": false,
  "enable_deduplicate": false,
  "rationale": "Concise asset-specific reasoning grounded in inspected evidence.",
  "evidence": ["Paths or observable facts used for the decision."]
}}
```

Each optimizer option may be `true`, `false`, or `null`; `null` delegates that option to the backend default. When `optimize` is false, set every option to `null`. Finish after writing and validating the JSON artifact.
"""


def build_physics_optimizer_selection_prompt(
    *,
    asset_path: Path,
    run_dir: Path,
    analysis_run_dir: Path,
    workbench_url: str,
    session_id: str,
    decision_path: Path,
    collision_approximation: str,
    runtime_validation_enabled: bool,
    additional_instructions: str | None = None,
) -> str:
    """Build the physics-task agent turn that chooses optimizer settings."""

    extra = (
        additional_instructions.strip()
        if additional_instructions and additional_instructions.strip()
        else "none"
    )
    return f"""You are selecting Content Workbench Scene Optimizer settings for the `physics_authoring` task.

Choose settings for this asset's physics operation, not for material assignment or generic visual simplification. The optimized inspection must preserve logical components, rigid-body roots, joints/articulations, existing collider/helper roles, and legal source authoring targets used by schema application and runtime validation.

Inputs:
- Asset: `{asset_path}`
- Workflow run: `{run_dir}`
- Unoptimized component evidence: `{analysis_run_dir / "raw" / "physics_components.json"}`
- Unoptimized topology evidence: `{analysis_run_dir / "raw" / "physics_topology.json"}`
- Workbench packet: `{analysis_run_dir / "raw" / "physics_run_packet.json"}`
- Workbench endpoint: `{workbench_url}`
- Disposable unoptimized session: `{session_id}`
- Requested collision approximation: `{collision_approximation}`
- Runtime validation enabled: `{str(runtime_validation_enabled).lower()}`
- Additional task guidance: {extra}

Decision criteria:
- Prefer no optimization when the source already exposes stable component, body, joint, collider, helper, and visual-evidence roles.
- Flatten prototypes only when composition prevents reliable component inspection or legal collider/body authoring. Do not flatten when it destroys joint ownership, articulation structure, or meaningful component boundaries.
- Deinstance when a physical instance needs independent body/collider properties or instance proxies are not legal authoring targets. Preserve instancing when repeated instances intentionally share one physical definition and mapping remains valid.
- Enable splitting only when a merged mesh incorrectly combines distinct physical components or collider roles. Disable it when splitting would invent false rigid bodies or fragment one rigid component.
- Enable deduplication only when duplicate geometry has identical physical role and shared authoring is valid. Disable it across independently moving bodies, different collider roles, or separate joint participants.
- Evaluate variants by component count and membership, body/joint/articulation preservation, collider/helper classification, authoring-target legality, source-path correspondence, and expected runtime behavior. Prim-count reduction alone is not success.
- The analysis session is disposable. You may compare targeted optimizer variants through Workbench when the unoptimized topology is ambiguous; record observed structural differences in `evidence`.

Do not edit the source USD or author physics during this turn. Write exactly one decision artifact to `{decision_path}`:

```json
{{
  "schema_version": "content-agents.optimizer-decision.v1",
  "task": "physics_authoring",
  "optimize": false,
  "flatten_prototypes": null,
  "enable_deinstance": null,
  "enable_split": null,
  "enable_deduplicate": null,
  "rationale": "Concise physics-task reasoning grounded in component and topology evidence.",
  "evidence": ["Observable component, topology, path, or variant facts used for the decision."]
}}
```

Each optimizer option may be `true`, `false`, or `null`; `null` delegates to the backend default. When `optimize` is false, set every option to `null`. Finish after writing and validating the JSON artifact.
"""


def _build_preflight_material_assignment_prompt(
    *,
    request: dict[str, object],
    run_dir: Path,
    materials_usd: Path,
    preflight_packet: dict[str, object],
    extra_block: str,
    vqa_refinement_max_iterations: int,
) -> str:
    packet_path = run_dir / "raw" / "material_run_packet.json"
    session_id = str(preflight_packet.get("session_id") or "")
    initial_renders = preflight_packet.get("initial_evidence_renders")
    if not isinstance(initial_renders, list):
        initial_renders = []
    render_lines = []
    for record in initial_renders:
        if not isinstance(record, dict):
            continue
        render_lines.append(
            f"- {record.get('name')}: {record.get('image_path')} "
            f"(direction={record.get('direction')}, quality={record.get('render_quality')})"
        )
    render_block = "\n".join(render_lines) if render_lines else "- none"
    respect_existing = bool(request.get("respect_existing_material_bindings"))
    policy_rule = (
        "Existing materials that already match the references should stay "
        "`preserved_existing`; do not issue Workbench commands for them."
        if respect_existing
        else "The Workbench session was created with clean-slate materials: "
        "existing material bindings, shader colors, and display colors were "
        "cleared before snapshot/render and redacted from material surveys "
        "unless request.appearance_evidence_policy authorizes scoped evidence. "
        "Assign explicit library materials to visible material candidates; do "
        "not preserve source appearance."
    )

    return f"""You are running a content-workflow-cli material assignment workflow.

Goal:
Assign material-library materials through Content Workbench so the asset render matches the supplied visual and document references.

Refinement contract:
- The wrapper treats this child review/remediation pass as VQA refinement iteration 1 of {vqa_refinement_max_iterations}.
- If canonical final artifacts still contain unresolved visual quality or final-review issues after wrapper validation, the wrapper may launch bounded refinement turns in the same run directory and Workbench session.
- Preserve enough artifact history for those turns through `raw/material_decision_patch.json`, concise evidence artifacts, and trace events instead of relying on final response prose. The wrapper owns canonical `assignments.json`, `visual_quality_assessment.json`, operation counts, final summary, and final verification renders.

Performance contract:
- A Workbench material-run packet has already been prepared by the wrapper. Reuse it.
- Do not fetch Workbench docs, create a session, snapshot the scene, or render initial evidence views again.
- The model input includes reference image(s), generic reference file path(s), and the Workbench initial render image(s). Inspect attached images directly before using shell/API tools. For non-image `reference_files`, read or inspect the local file path with an appropriate targeted tool before making material decisions.
- Use shell only for targeted Workbench operations and the patch handoff. Avoid `jq`, `sed`, large raw JSON reads, local skill docs, README files, previous run summaries, OpenAPI/schema spelunking, and manual rewrites of broad final artifacts.
- Use additional picks/renders only for a named visual ambiguity.
- Default caps are 6 child-issued renders and 8 pick calls. There is no material-assignment target cap in clean-slate mode: every canonical material candidate must receive an explicit material-library assignment unless it is genuinely unassignable with evidence.

Inputs:
{json.dumps(request, indent=2)}
{extra_block}
Prepared packet:
- Packet path: `{packet_path}`
- Workbench session: `{session_id}`
- Compact context: `{run_dir / "raw" / "material_authoring_context.md"}`
- Assignment seed: `{run_dir / "raw" / "material_assignment_seed.json"}`
- Candidate table: `{run_dir / "raw" / "visible_candidate_table.tsv"}`
- Material palette: `{run_dir / "raw" / "material_palette.json"}`

Attached Workbench initial renders:
{render_block}

Hard constraints:
- Use only Content Workbench API calls and the supplied material library.
- Do not edit source USD files.
- Material edits must be non-destructive Workbench `material_override` commands.
- Material assignments must cover every canonical material candidate from `raw/visible_candidate_prims.json`. By default this is a source-space list: instance-proxy/runtime evidence is collapsed to authorable source/prototype targets so repeated instances can inherit through USD composition. Iterate that list as the authoritative checklist, decide a material for each candidate, then group candidates that share the same material family in `raw/material_decision_patch.json`.
- Material manifest names/descriptions are authoritative, but visual match is the primary goal when the exact substance is absent. If no exact material exists for a high-salience color mismatch, use the closest opaque/surface-compatible visual proxy from the supplied palette when it clearly improves the render, and record the substance/finish limitation in the patch. For example, a blue plastic proxy is acceptable for blue upholstery when the alternatives are leaving the chair gray/white or using glass/metal/paint that is visibly worse.
- Do not use a proxy that changes the target to the wrong color, transparency class, metalness class, or a worse finish than the current material.
- {policy_rule}
- Do not leave a candidate covered only by seed/default appearance in clean-slate mode. `reviewed_no_override` is valid only when `respect_existing_material_bindings` is true.
- Use grouped material assignments to keep the patch readable; it is acceptable for a group to contain many prim paths when they share the same visible material decision. The wrapper will apply the needed Workbench commands.
- Use the artifact `path_space` in `assignments.json`: source-space artifacts use authorable source/prototype paths in `prim_paths`; inspection-space artifacts use runtime inspection paths in `prim_paths` with source expansions in `source_prim_paths`.

Normal fast path:
1. Read `{packet_path}` only if you need exact paths for the prepared artifacts.
2. Use the attached reference images, generic reference files, and Workbench initial render images for visual comparison.
3. Read `raw/material_authoring_context.md` and `raw/material_assignment_seed.json` for compact material/path context.
4. Iterate `raw/visible_candidate_prims.json` or `raw/visible_candidate_table.tsv` and maintain a checklist so every canonical material candidate is assigned to exactly one material decision group.
5. Apply targeted `material_override` commands through `POST /sessions/{session_id}/commands`.
6. Render a small verification view only when needed to validate a changed group. Do not populate `{run_dir / "final_renders"}`; the wrapper renders canonical final views after your patch.
7. Write `raw/material_decision_patch.json` as the source-of-truth decision patch. The wrapper deterministically validates this patch, reapplies accepted material assignments, renders canonical final views, and rewrites the standard final artifacts.
8. Stop after the patch and any supporting evidence artifacts are written.

Workbench API quick contract:
- Material override:
  `POST /sessions/{session_id}/commands`
  with `{{"command":"material_override","payload":{{"prim_path":"<inspection-or-source path>","space":"<source-or-inspection>","unbind_existing":true,"material":{{"source":"material_library","library_path":"{materials_usd}","material_path":"<library path>","material_name":"<name>"}}}}}}`
  Use the artifact `path_space` to choose command target space. When it is
  `inspection`, use `runtime_prim_paths` with `space: "inspection"`. When it is
  `source`, use `prim_paths` with `space: "source"` and keep runtime paths as
  visual evidence.
- Render:
  `POST /sessions/{session_id}/render`
  with `width`, `height`, `use_session_camera:false`, `direction`, `margin`, `render_quality`, and `save_camera_json:true`.
- Pick only when required:
  `POST /sessions/{session_id}/pick`
  with `{{"x":<int>,"y":<int>,"width":<render width>,"height":<render height>,"update_selection":false,"mode":"replace","ovrtx_render_mode":"rt2","ovrtx_num_sensor_updates":1}}`.
  Pick uses the current session camera. Do not include `direction`, `camera`,
  `focus`, `margin`, `use_session_camera`, ray fields, or render-response
  camera JSON in the pick payload; Workbench rejects extra fields.

Patch-only handoff:
- `{run_dir}/raw/material_decision_patch.json`
- optional focused evidence renders or pick results under `{run_dir}/raw` or `{run_dir}/evidence_renders`
- optional concise observable trace events under `{run_dir}/trace/events.jsonl`
- Do not write `assignments.json`, `api_operation_counts.json`, `visual_quality_assessment.json`, `final_summary.md`, or canonical final render PNGs during the preflight child turn. The wrapper-owned finalizer writes those artifacts from the patch immediately after the child exits.

`raw/material_decision_patch.json` must include:
- `material_assignments`: list of changed material groups. Each item must include `family`, exact `material_name` from `raw/material_palette.json`, matching `material_path`, target paths, and `rationale`.
- `reviewed_no_override`: only valid when `respect_existing_material_bindings` is true. Each item must include `family`, target paths, and `rationale`.
- `final_review_issues_found`, `final_review_issues_fixed`, `final_review_notes`.
- `visual_quality_assessment` using the same fields required below.
- In inspection-space sessions, target paths for both lists should use `runtime_prim_paths` with source expansions in `source_prim_paths`. In source-space sessions, use `prim_paths`; runtime paths are evidence/preview fan-out, not prediction targets.
- Do not put broad seed coverage in `material_assignments`; include only explicit material-library material decisions. In clean-slate mode the union of `material_assignments[*].runtime_prim_paths` or `prim_paths` must cover all canonical candidates in `raw/visible_candidate_prims.json`.

Wrapper-owned final artifacts:
- After the child exits, the wrapper reads `raw/material_decision_patch.json`, validates material names/paths against `raw/material_palette.json`, rejects unsafe groups, reapplies accepted material assignments, renders canonical final views, and writes `assignments.json`, `api_operation_counts.json`, `visual_quality_assessment.json`, `final_summary.md`, `raw/final_render_records.json`, and final render PNGs.
- Do not manually reconstruct those wrapper-owned artifacts to preserve context. Put the decision, coverage rationale, and VQA status in the patch only.

Quality gate:
- The patch visual quality should be `pass` or `fixed` when the chosen override set is expected to resolve all fixable issues.
- Record unresolved issues only when Workbench/material granularity prevents a precise fix or every available visual proxy would make the render worse than the current state. Lack of an exact fabric/textile/paint subtype is not by itself a reason to leave a dominant color mismatch unfixed.
- Check for over-broad overrides, high-contrast color mistakes, finish mismatches, missed visible accents, and stale isolation/selection artifacts.
- If the final output is already correct after targeted overrides, stop promptly. Do not spend extra turns polishing trace prose.

Finish with a short final response pointing to the run directory, decision patch, and any focused evidence artifacts you wrote.
"""


def build_material_assignment_prompt(
    *,
    repo_root: Path,
    run_dir: Path,
    usd_path: Path,
    reference_images: list[Path],
    materials_yaml: Path,
    materials_usd: Path,
    workbench_url: str,
    reference_files: list[Path] | None = None,
    optimize: bool = True,
    optimizer_options: dict[str, object] | None = None,
    material_candidate_policy: dict[str, object] | None = None,
    respect_existing_material_bindings: bool = False,
    additional_instructions: str | None = None,
    preflight_packet: dict[str, object] | None = None,
    vqa_refinement_max_iterations: int = 3,
) -> str:
    """Build the child-agent prompt for material assignment."""

    request = {
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "usd_path": str(usd_path),
        "reference_images": [str(path) for path in reference_images],
        "reference_files": [str(path) for path in reference_files or []],
        "materials_yaml": str(materials_yaml),
        "materials_usd": str(materials_usd),
        "workbench_url": workbench_url,
        "workbench_optimize": optimize,
        "optimizer_options": optimizer_options or {},
        "material_candidate_policy": _material_candidate_policy(
            material_candidate_policy,
            preflight_packet,
        ),
        "respect_existing_material_bindings": respect_existing_material_bindings,
        "vqa_refinement_max_iterations": vqa_refinement_max_iterations,
    }
    extra = additional_instructions.strip() if additional_instructions else ""
    extra_block = f"\nAdditional user instructions:\n{extra}\n" if extra else ""
    if preflight_packet is not None:
        return _build_preflight_material_assignment_prompt(
            request=request,
            run_dir=run_dir,
            materials_usd=materials_usd,
            preflight_packet=preflight_packet,
            extra_block=extra_block,
            vqa_refinement_max_iterations=vqa_refinement_max_iterations,
        )

    return f"""You are running a content-workflow-cli material assignment workflow.

Goal:
Assign the best matching material-library material to each relevant prim in the USD asset so the Workbench render matches the supplied visual and document references.

Refinement contract:
- This initial child run includes VQA refinement iteration 1 of {vqa_refinement_max_iterations}.
- If canonical wrapper-validated artifacts still contain unresolved visual quality or final-review issues, the wrapper may launch additional bounded refinement turns in the same run directory/session.
- Preserve decisions and evidence for those turns by writing durable artifacts, especially `raw/material_decision_patch.json`, `assignments.json`, `visual_quality_assessment.json`, final renders, and trace events.

Termination goal:
Finish only when every visible/renderable material candidate prim is covered by an explicit material decision and the final visual quality assessment has no unresolved fixable defects. A prim is covered when it appears in exactly one `assignments.json` entry with `coverage_status: "material_assignment"`, `coverage_status: "preserved_existing"`, or `coverage_status: "ambiguous_unassigned"`. Do not treat non-renderable scopes, joints, material-library prims, or hidden collision helpers as material candidates unless they are visibly rendered. Once the coverage invariant is satisfied, final verification renders are written, and visual quality defects have been fixed or explicitly marked unfixable with evidence, stop promptly and write the final response.

Inputs:
{json.dumps(request, indent=2)}
{extra_block}
Hard constraints:
- Use Content Workbench as the scene interaction and rendering surface.
- Fetch the canonical Workbench docs from `{workbench_url}/agent-api`, `{workbench_url}/agent-api.json`, and `{workbench_url}/openapi.json` and save them under `raw/` for traceability. Use this prompt's workflow contract, the Workbench API quick contract below, and the compact helper artifacts first. Do not open, grep, `sed`, or `jq` saved Workbench docs during the normal path; inspect docs only after an API call fails or an endpoint schema is genuinely unclear.
- Use only Workbench API calls and the supplied material library for scene inspection, rendering, and material assignment.
- Do not use repository rendering CLIs or source USD edits.
- Do not reuse previous run artifacts from other directories.
- Existing authored appearance is {"respected as seed coverage" if respect_existing_material_bindings else "cleared from the Workbench inspection session and redacted from material surveys unless request.appearance_evidence_policy authorizes scoped evidence; do not preserve source material bindings, shader colors, or display colors"}.
- This prompt is the complete workflow contract for this run. Do not read local skill docs, README files, or prior run summaries unless a required input is missing.
- For non-image `reference_files`, read or inspect the local file path with an appropriate targeted tool before making material decisions. Treat PDFs/docs as reference evidence, not as renderable inputs.
- Material edits must be non-destructive Workbench `material_override` commands.
- If no exact material exists for a high-salience color mismatch, choose the closest opaque/surface-compatible visual proxy when it clearly improves the render and record the limitation. Do not leave dominant gray/white/black mismatches unresolved solely because the palette lacks the exact textile, fabric, paint subtype, or finish.
- In clean-slate mode, visible material candidates need explicit library material assignments unless they are genuinely ambiguous/unassignable. Use `preserved_existing` only when existing material bindings were explicitly respected.
- Treat `raw/visible_candidate_prims.json` as the authoritative coverage universe. With the default material-agent-compatible policy, these are canonical source/prototype material candidates; repeated runtime instance proxies are retained in `runtime_prim_paths` as evidence and preview fan-out. Literally iterate the candidate list during prediction, decide the intended material for each canonical candidate, and group candidates that share the same material decision.
- Do not leave any clean-slate canonical material candidate covered only by default material, display color, existing binding evidence, or seed coverage. The final patch must explicitly assign a material-library material to every candidate unless the candidate is genuinely unassignable with evidence.
- Use grouped material decisions for repeated parts and shared material families. A group may contain many target prims when the material decision is the same; the wrapper will apply the corresponding Workbench commands.
- Prefer actual materials from `materials_usd`; use `materials_yaml` to understand material names/categories.
- Use multiple Workbench views, camera moves, pixel picks, isolation, and final verification renders as needed.
- Use explicit bounded render dimensions: quick navigation at or below 640x480, evidence/final verification at or below 768x576 unless the user explicitly asks for larger output.
- Use `render_quality: "interactive"` for quick navigation, `render_quality: "inspection"` for evidence renders, and `render_quality: "final"` for final verification renders. Final renders should use Workbench's effective `ovrtx_render_mode` and update count from the render response.
- For standard visual inspection, omit `hdri_light`, `dome_light`, and `distant_light` so every workflow inherits Workbench's HDRI-600-only default. Override lighting only when the user or scene-specific evidence requires a different controlled rig; source-authored lights remain suppressed.
- If a `final` render fails, stalls, or triggers Workbench/GPU memory pressure, retry the same verification view with `render_quality: "inspection"`, `ovrtx_render_mode: "rt2"`, and `ovrtx_num_sensor_updates` no higher than 64; record this as a warning in the trace and artifacts instead of increasing resolution or update count.
- Create the Workbench session with `optimize: {str(optimize).lower()}`. If optimized, use `/optimization` and `/paths/translate` to recover source-space targets from inspection-space picks/selections.
- Prefer the packaged client tool `content-workbench-snapshot-scene` for initial scene inspection. It calls Workbench `/scene/snapshot`, writes standard raw artifacts plus compact material-authoring context files, and prints compact counts instead of dumping full scene JSON into the agent transcript.
- Start material reasoning from `raw/material_authoring_context.md`, `raw/material_assignment_seed.json`, `raw/visible_candidate_prims.json`, `raw/visible_candidate_table.tsv`, and `raw/material_palette.json`. Avoid broad `jq`, `sed`, or Python inspection over `raw/scene_snapshot.json`, `raw/properties_batch_all.json`, `raw/material_binding_batch_all.json`, `raw/path_translation_batch_all.json`, `materials_yaml`, or the Workbench docs unless a targeted follow-up is needed.
- Treat compact context candidate groups as coverage evidence, not an authoring plan. Groups marked `recommended_coverage_status: "preserved_existing"` require no Workbench command only when this run explicitly respects existing material bindings.
- Use `raw/material_assignment_seed.json` as a grouping hint, but use `raw/visible_candidate_prims.json` as the checklist that must be fully covered by explicit material decisions in clean-slate mode.
- Prefer Workbench batch endpoints for routine multi-prim reads: `/properties:batch`, `/material-binding:batch`, and `/paths/translate:batch`. Do not write one-off per-prim HTTP loops when an equivalent batch endpoint is available.
- Record observable evidence and rationale. Do not write hidden chain-of-thought.

Workbench API quick contract:
- Create a session with `POST /sessions` using `scene_path`, `width`, `height`, `optimize`, `clear_materials`, `optimizer_backend`, `flatten_prototypes`, `enable_deinstance`, `enable_split`, `enable_deduplicate`, and optional `optimization_config`. In default mode set `clear_materials: true`; set it false only when respecting existing material bindings. Read `session_id` from the JSON response.
- Snapshot the scene with `content-workbench-snapshot-scene`; do not manually rediscover snapshot, properties, material-binding, or path-translation schemas.
- Render with `POST /sessions/<session_id>/render` using `width`, `height`, `use_session_camera: false`, `direction`, `focus`, `margin`, `render_quality`, and `save_camera_json: true`. Download `image_url` from the render response.
- Material overrides with `POST /sessions/<session_id>/commands` and `{{"command":"material_override","payload":{{"prim_path":"<inspection-or-source path>","space":"<source-or-inspection>","unbind_existing":true,"material":{{"source":"material_library","library_path":"{materials_usd}","material_path":"<library path>","material_name":"<name>"}}}}}}`. Set `space` from `raw/visible_candidate_prims.json` `path_space`.
- Pick with `POST /sessions/<session_id>/pick` using `x`, `y`, optional `width`/`height`, `update_selection`, `mode`, `ovrtx_render_mode`, and `ovrtx_num_sensor_updates` only. Pick uses the current session camera; do not include `direction`, `camera`, `focus`, `margin`, `use_session_camera`, ray fields, or render-response camera JSON in the pick payload.
- Read accepted material assignment state with `GET /sessions/<session_id>/authoring/material-assignments` only when needed for final verification; do not poll it after every decision.

Completion discipline:
- Produce a material coverage report, not only a list of changed overrides.
- Every visible/renderable candidate prim you considered must end in one of these coverage states:
  `material_assignment`, `preserved_existing`, or `ambiguous_unassigned`.
- Build and maintain a visible-candidate checklist while inspecting. Before final response, verify that `candidate_visible_prim_count == material_decision_prim_count` and that every candidate path is represented in exactly one assignment entry.
- Major visible families such as body shell, frame, panels, wheels/feet, hands, labels/logos, glass/lenses, rubber, dark inserts, exposed metal, and fasteners must be explicitly represented in `assignments.json`; in clean-slate mode that representation should be a library material assignment or a justified ambiguous/unassignable decision.
- Prefer grouped family assignments for repeated visible parts when the same material decision clearly applies.
- Default caps are 10 evidence renders and 12 pick calls. Material assignments are not capped by prim count; in clean-slate mode completeness over all canonical material candidates is required.
- After confirming one material assignment works, apply the high-confidence visible material groups, render final verification views, and write the required artifacts.
- Before finishing, perform a final review pass against the final renders, material coverage table, assignment evidence, reference images, and generic reference files. If a visible family is missing, visibly wrong, over-broad, over-saturated, too bright/dark, or under-evidenced, fix it with additional Workbench inspection/material overrides or mark it as `ambiguous_unassigned`, then re-render the affected final view(s).
- Treat visual quality assessment as a required acceptance gate, not as a summary. Explicitly check for:
  - over-broad overrides where a material was applied to a prim that visually covers more surface area than the intended reference part;
  - high-contrast color mistakes, such as white logos that should be dark/subtle or black panels that should remain silver/gray;
  - material finish mistakes, such as rubber/plastic/paint/metal choices that make the final render visibly inconsistent with the reference;
  - missing separately bindable accents, labels, lenses, inserts, and connector hardware;
  - selection outlines, isolation leftovers, failed overrides, or stale final renders.
- For each visual defect, record the affected final render(s), affected prim paths, the observable mismatch, and the remediation. If Workbench prim granularity prevents a precise fix, prefer the least visually wrong whole-prim material and record the limitation as unresolved rather than silently accepting the bad look.
- If some geometry remains ambiguous, include its prim paths or family in the `ambiguous_unassigned` coverage state, record the uncertainty in `assignments.json` and `final_summary.md`, and finish with the best observable assignment map. Ambiguous coverage is acceptable only when it names the affected candidate paths/families and explains the limitation.

Required artifact layout:
- Write all run artifacts under `{run_dir}`.
- Save raw API/reference data under `{run_dir}/raw`.
- Save evidence renders under `{run_dir}/evidence_renders`.
- Save final renders under `{run_dir}/final_renders`.
- Write `{run_dir}/raw/material_decision_patch.json`.
- Write `{run_dir}/assignments.json`.
- Write `{run_dir}/api_operation_counts.json`.
- Write `{run_dir}/visual_quality_assessment.json`.
- Write `{run_dir}/final_summary.md`.
- Append observable trace events to `{run_dir}/trace/events.jsonl`.

Structured material decision patch:
- Before finishing, write `{run_dir}/raw/material_decision_patch.json`.
- The patch is the source-of-truth decision handoff. The wrapper deterministically validates it, reapplies accepted material assignments, renders canonical final views, and rewrites the standard final artifacts.
- Use this JSON shape:
{{
  "schema_version": "content-agents.material-decision-patch.v1",
  "material_assignments": [
    {{
      "family": "short visible family name",
      "material_name": "exact raw/material_palette.json name",
      "material_path": "matching raw/material_palette.json material path",
      "runtime_prim_paths": ["runtime paths when path_space is inspection"],
      "source_prim_paths": ["source expansions when available"],
      "prim_paths": ["source paths when path_space is source"],
      "rationale": "short observable reason"
    }}
  ],
  "reviewed_no_override": [
    {{
      "family": "short visible family name",
      "runtime_prim_paths": ["runtime paths when path_space is inspection"],
      "source_prim_paths": ["source expansions when available"],
      "prim_paths": ["source paths when path_space is source"],
      "rationale": "why the existing/default appearance is acceptable"
    }}
  ],
  "final_review_issues_found": [],
  "final_review_issues_fixed": [],
  "final_review_notes": "short final review notes",
  "visual_quality_assessment": {{
    "status": "pass|fixed|unresolved_issues",
    "checked_views": ["final render paths inspected"],
    "reference_images": ["reference image paths used"],
    "reference_files": ["non-image reference file paths used"],
    "issues_found": [],
    "issues_fixed": [],
    "unresolved_issues": [],
    "assessment_notes": "short notes"
  }}
}}
- Do not put broad seed coverage in `material_assignments`; only include families that should receive a material-library material assignment.
- In clean-slate mode, the union of `material_assignments[*].runtime_prim_paths` or `prim_paths` must cover every canonical candidate in `raw/visible_candidate_prims.json`; untouched seed rows are not valid material decisions.
- Material names and paths must match `raw/material_palette.json`.
   - In inspection-space sessions, material assignment and reviewed groups should use `runtime_prim_paths`; record source expansions in `source_prim_paths`. In source-space sessions, use `prim_paths`; keep `runtime_prim_paths` as visual evidence or preview fan-out.

Trace event schema:
Append one JSON object per meaningful step to `{run_dir}/trace/events.jsonl`.
Each event should use this shape:
{{
  "schema_version": "content-agents.trace.v1",
  "event_type": "evidence|decision|api|render|pick|assignment|verification|warning",
  "phase": "short phase name",
  "summary": "concise observable evidence or decision summary",
  "artifacts": ["paths to images/json files used as evidence"],
  "data": {{
    "api_calls": ["optional endpoint names"],
    "prim_paths": ["optional prim paths"],
    "material_names": ["optional material names"],
    "uncertainty": "optional uncertainty note"
  }}
}}

Required observable workflow:
1. Fetch Workbench API docs and save them in `raw/`.
2. Create a Workbench session for `usd_path` with `optimize: {str(optimize).lower()}` plus any non-null `optimizer_options` from Inputs, and save the session/optimization responses under `raw/`.
3. Snapshot the scene hierarchy and candidate hints.
   - Prefer `content-workbench-snapshot-scene --workbench-url {workbench_url} --session-id <session_id> --run-dir {run_dir} --materials-yaml {materials_yaml} --materials-usd {materials_usd}` with the Inputs `material_candidate_policy` flags (`--root-prim-path`, `--material-candidate-space`, `--skip-instances`/`--include-instances`, `--skip-prototypes`/`--include-prototypes`, `--skip-invisible`/`--include-invisible`) as applicable.
   - The tool calls `POST /sessions/{{session_id}}/scene/snapshot` and writes `raw/scene_snapshot.json`, `raw/tree_paths.json`, `raw/properties_batch_all.json`, `raw/material_binding_batch_all.json`, `raw/path_translation_batch_all.json`, `raw/visible_candidate_prims_preliminary.json`, `raw/visible_candidate_prims.json`, `raw/material_authoring_context.json`, `raw/material_authoring_context.md`, `raw/visible_candidate_table.tsv`, `raw/material_palette.json`, and `raw/material_assignment_seed.json`.
   - Use `raw/material_authoring_context.md` for the first overview, `raw/visible_candidate_prims.json` as the canonical material coverage universe, `raw/material_assignment_seed.json` as the editable starting ledger for `assignments.json`, and `raw/visible_candidate_table.tsv` for path lookup. Use raw snapshot/batch files only for targeted missing details. Do not turn all candidate rows into material assignment commands.
   - If the packaged tool or `/scene/snapshot` is unavailable, fall back to Workbench batch endpoints for properties, material bindings, and path translations when querying many prims.
   - Save or reuse the visible/renderable material-candidate checklist under `raw/visible_candidate_prims.json`.
   - Exclude non-rendered scopes, joints, material-library prims, and hidden collision helpers unless rendered evidence shows they are visible.
4. Render initial top, bottom, side/front/back, and oblique views.
5. Use Workbench camera commands, pixel picking, and isolation to resolve ambiguous geometry/material groups.
6. Apply a small material-library material assignment first to confirm binding works.
7. Apply library-backed material overrides for the selected prims. Use `path_space` to choose source-space authoring targets or inspection-space runtime targets; source-space runtime paths are evidence, not separate predictions.
8. Render final verification views from multiple directions with `render_quality: "final"` unless the Workbench reports that the requested mode is unavailable.
9. Run a final review/remediation pass:
   - Compare final renders against the reference images, generic reference files, and the material coverage table.
   - Identify missing visible families, obviously wrong colors/finishes, failed overrides, unresolved optimized/source path translation, and under-evidenced ambiguous selections.
   - Identify over-broad material bindings by asking whether each changed prim affects only the intended visible part in the final render. For example, do not accept a full upper-leg/hip mesh as black if the reference only shows a small dark socket or insert.
   - Re-evaluate high-impact black/white overrides. Do not accept a white logo, black upper-leg panel, black torso panel, or black head/lens choice just because the prim is separately bindable; verify that the color/finish matches the reference image.
   - Fix resolvable issues using more Workbench camera moves, picks, isolation, material-binding queries, and material assignments.
   - Verify that every path in `raw/visible_candidate_prims.json` appears in exactly one `assignments.json` assignment entry, either directly or through a clearly named family group.
   - Re-render any final view affected by a fix.
10. Save `assignments.json` with:
   - `session_id`
   - `source_usd`
   - `library_path`
   - `per_prim_material_assignment_count`
   - `coverage`: an object with `candidate_visible_prim_count`, `material_decision_prim_count`, `material_assignment_prim_count`, `preserved_existing_prim_count`, `ambiguous_unassigned_prim_count`, `missing_assignment_prim_count`, `rejected_assignment_prim_count`, `unassigned_visible_prim_count`, and short `coverage_notes`
   - `assignments`: list of objects with `family`, `coverage_status`, `material_name`, `material_path`, `prim_paths`, and a short observable `rationale`
   - `final_review`: an object with `issues_found`, `issues_fixed`, `unresolved_issues`, and short `review_notes`
   - `visual_quality_assessment`: an object with `status`, `issues_found`, `issues_fixed`, `unresolved_issues`, `checked_views`, and short `assessment_notes`
   - For `coverage_status: "material_assignment"`, `material_name` and `material_path` must be the applied library material.
   - For `coverage_status: "preserved_existing"`, `material_name` may describe the observed existing/default material family, `material_path` may be `null`, and `rationale` must explain why preserving it matches the reference.
   - For `coverage_status: "ambiguous_unassigned"`, `material_name` and `material_path` may be `null`, and `rationale` must explain the ambiguity or API limitation.
11. Save `api_operation_counts.json` with at least:
   - `api_operation_count_total`
   - `render_count_total`
   - `pick_calls`
   - `material_override_commands`
   - `final_renders`
   - `coverage_candidate_visible_prims`
   - `coverage_material_decision_prims`
   - `final_review_issues_found`
   - `final_review_issues_fixed`
   - `visual_quality_issues_found`
   - `visual_quality_issues_fixed`
12. Save `visual_quality_assessment.json` with:
   - `status`: `pass`, `fixed`, or `unresolved_issues`
   - `checked_views`: list of final render image paths inspected
   - `reference_images`: list of reference image paths used
   - `reference_files`: list of non-image reference file paths used
   - `issues_found`: list of visual mismatch objects, each with `severity`, `description`, `affected_prim_paths`, `evidence_artifacts`, `expected_appearance`, `actual_appearance`, and `status`
   - `issues_fixed`: list of fixes made before acceptance
   - `unresolved_issues`: list of remaining issues and why they are not fixable with current Workbench/material-library granularity
   - `assessment_notes`: short notes on perceptual match, over-broad overrides, high-contrast color mistakes, and material finish mismatches
13. Save `final_summary.md` with the final material map, a coverage summary table, visual quality assessment, final review findings/fixes, evidence used, uncertainty, and key artifact paths.

Finish with a short final response that points to the run directory, final renders, assignment file, and trace files.
"""


def build_material_refinement_prompt(
    *,
    run_dir: Path,
    usd_path: Path,
    reference_images: list[Path],
    materials_yaml: Path,
    materials_usd: Path,
    workbench_url: str,
    session_id: str,
    iteration: int,
    max_iterations: int,
    issue_summary: dict[str, object],
    history_path: Path,
    artifact_index_path: Path | None = None,
    issue_packet_path: Path | None = None,
    reference_files: list[Path] | None = None,
    repair_attempt_ledger: list[dict[str, object]] | None = None,
    previous_child_artifacts: list[dict[str, str]] | None = None,
    optimize: bool = True,
    respect_existing_material_bindings: bool = False,
    additional_instructions: str | None = None,
) -> str:
    """Build a compact VQA refinement prompt for a follow-up child turn."""

    request = {
        "run_dir": str(run_dir),
        "usd_path": str(usd_path),
        "reference_images": [str(path) for path in reference_images],
        "reference_files": [str(path) for path in reference_files or []],
        "materials_usd": str(materials_usd),
        "workbench_url": workbench_url,
        "workbench_session": session_id,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "optimized_session": optimize,
        "respect_existing_material_bindings": respect_existing_material_bindings,
        "artifact_index": str(artifact_index_path) if artifact_index_path else None,
        "issue_packet": str(issue_packet_path) if issue_packet_path else None,
    }
    extra = additional_instructions.strip() if additional_instructions else ""
    extra_block = f"\nAdditional user instructions:\n{extra}\n" if extra else ""
    attempt_ledger = repair_attempt_ledger or []
    compact_attempts = attempt_ledger[-3:]
    issue_summary_compact = {
        "status": issue_summary.get("status"),
        "active_issues": issue_summary.get("active_issues") or [],
        "issues_fixed": issue_summary.get("vqa_issues_fixed") or [],
        "coverage": issue_summary.get("coverage") or {},
        "current_material_decisions": issue_summary.get("current_material_decisions")
        or [],
        "assessment_notes": issue_summary.get("assessment_notes") or "",
    }
    allowed_artifact_reads = []
    if issue_packet_path is not None:
        allowed_artifact_reads.append(
            f"- `{issue_packet_path}` for the compact active issue packet and "
            "direct artifact pointers."
        )
    if artifact_index_path is not None:
        allowed_artifact_reads.append(
            f"- `{artifact_index_path}` for searchable paths to views, "
            "decisions, rejected paths, step snapshots, and trace state."
        )
    allowed_artifact_reads.extend(
        [
            f"- `{run_dir / 'raw' / 'material_decision_patch.json'}` for exact "
            "current groups and prim paths.",
            f"- `{run_dir / 'assignments.json'}` for current group-to-prim mapping.",
            f"- `{run_dir / 'raw' / 'material_palette.json'}` for a small targeted "
            "material lookup.",
            f"- `{run_dir / 'raw' / 'final_render_records.json'}` only if a pixel "
            "pick needs camera metadata.",
            f"- `{history_path}` only to avoid repeating a failed prior repair.",
        ]
    )
    allowed_artifact_reads_block = "\n".join(allowed_artifact_reads)

    return f"""You are doing a compact VQA refinement turn for an existing material-assignment run.

Goal:
Fix only the active non-systematic issue(s), or mark them unfixable with evidence. Do not redo the material plan.

Inputs:
{json.dumps(request, indent=2)}
{extra_block}
Active issue packet:
{json.dumps(issue_summary_compact, indent=2)}

Recent repair attempts:
{json.dumps(compact_attempts, indent=2)}

Allowed artifact reads, only if needed:
{allowed_artifact_reads_block}

Do not read skill docs, repository source, tests, README files, full child logs, broad raw scene snapshots, or previous run summaries. The API facts needed for this turn are below.

Workbench API facts:
- Reuse session `{session_id}` at `{workbench_url}`; do not create/restart/optimize/snapshot.
- Pick: `POST /sessions/<session_id>/pick` with only `x`, `y`, optional `width`/`height`, `update_selection`, `mode`, `ovrtx_render_mode`, `ovrtx_num_sensor_updates`.
- Material override: `POST /sessions/<session_id>/commands` with command `material_override`; use material names/paths from `raw/material_palette.json`.
- Render only the affected final view(s), at or below 768x576, if visual confirmation is needed. The wrapper/finalizer will regenerate canonical final renders after this turn.

Compact procedure:
1. Use the attached reference and current final-render images plus the active issue packet.
2. If an issue is already proven material-library or prim-granularity limited, write a concise unfixable trace event and leave it unresolved.
3. For a fixable issue, identify the smallest existing material group or exact prims. Prefer the inline active issue packet, then `issue_packet`/`artifact_index`; read only targeted JSON slices if exact prim paths are missing.
   - For `rejected_assignment` issues, treat the rejected paths as mandatory repair targets. Read the rejection reason and only the targeted rejected group if needed, then split overly broad/mixed groups, correct invalid material names or paths, or retarget invalid candidate paths before finalizing. Do not leave rejected paths uncovered unless you can show a material-library or geometry-addressability limitation.
4. Check only the small relevant material candidates. Do not browse the full palette unless the candidate is not already named.
5. Apply at most the needed material assignment(s), render only affected view(s) if needed, and update `raw/material_decision_patch.json` plus `visual_quality_assessment.json`.
6. Do not rewrite broad coverage or regenerate `assignments.json`; the wrapper finalizer owns canonical rewrites.

Output requirements:
- Final response must list changed group(s), affected prims, affected render(s), VQA status, and remaining limitations.
- Keep the trace and JSON edits concise and observable; no broad context preservation.
"""
