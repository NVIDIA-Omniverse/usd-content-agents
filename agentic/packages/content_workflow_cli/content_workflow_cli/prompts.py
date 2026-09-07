# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt builders for content-workflow-cli workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from content_agent_workflows.physics import PhysicsVompMassConfig

CHILD_MATERIAL_ASSIGNMENT_ARTIFACTS = (
    "raw/material_decision_patch.json",
    "trace/",
)
# Canonical saved-output location for usd-cli material runs; the runner's
# usd-cli finalizer enforces this same path, so keep the two in sync here.
USD_CLI_OUTPUT_USD_RELATIVE_PATH = Path("output") / "materialized.usda"
USD_CLI_WORKFLOW_COMMAND = "usd-cli-tel"
if os.name == "nt":
    JSON_SYNTAX_CHECK_GUIDANCE = (
        "The controlled JSON writer below performs the JSON syntax check before it "
        "publishes the artifact. Verify the published bytes with a direct "
        "`Get-Content -LiteralPath <path> -Raw` read; do not pipe through "
        "`ConvertFrom-Json`, because the managed Windows command policy rejects that "
        "pipeline."
    )
    CONTROLLED_JSON_ARTIFACT_WRITE = """Controlled JSON artifact writes: do not try `apply_patch` for generated JSON;
it is unavailable in this child session. Construct the complete compact JSON
document and pass it as a single-quoted PowerShell literal directly to
`content-workflow-cli artifact write-json --output <relative/path.json> --json '<document>'`.
PowerShell 5.1 strips ordinary JSON double quotes at the native executable
boundary. In the compact document, write every JSON double quote as `\\\"` and
encode every literal space inside a JSON string value as `\\u0020`; otherwise
PowerShell can split that value into extra CLI arguments. For example:
`content-workflow-cli artifact write-json --output raw/example.json --json '{\\\"note\\\":\\\"two\\u0020words\\\"}'`.
Escape an apostrophe inside the surrounding PowerShell literal by writing it twice.
Run the writer as its own command. Do not append a verification command with
`;`, `|`, or another shell operator; managed Windows policy evaluates the whole
shell command and rejects the combined form.
The checked-in writer rejects an existing destination instead of clobbering it
unless `--replace` is supplied. Replacement is confined to the child-owned
`raw/material_decision_patch.json` and `raw/physics_decision_patch.json` paths.
Do not use `--replace` for any other artifact.
It accepts only a canonical forward-slash path relative to this run, creates
required parent directories, parses the document, writes UTF-8 without a BOM
through the confined Windows artifact backend, and reports the published path.
In a separate command, verify it with the direct read
`Get-Content -LiteralPath <relative/path.json> -Raw`; the writer has already
validated its JSON syntax.
Invoke a parent-admitted checked-in Python helper as
`python '<absolute-helper.py>' ...` (or `python -E '<absolute-helper.py>' ...`).
Do not use PowerShell's `&` call operator, `python -m`, `python -c`, or a copied
helper; only the exact digest-bound helper paths are admitted.
If a command is declined by managed policy, do not retry an equivalent shell
spelling. Return to these exact admitted forms; if the required exact form is
also declined, report the system blocker immediately instead of continuing.
The wrapper performs the schema and reproducibility validation after the child exits."""
else:
    JSON_SYNTAX_CHECK_GUIDANCE = (
        "Use `jq -e` for JSON syntax checks and the documented checked-in CLI "
        "commands for operations."
    )
    CONTROLLED_JSON_ARTIFACT_WRITE = """Controlled JSON artifact writes: do not try `apply_patch` for generated JSON;
it is unavailable in this child session. Construct the complete document with
`jq -nS` (passing dynamic values with `--arg`/`--argjson`), write it to a
same-directory `.tmp` file, then atomically `mv` it to the required path.
Immediately run `jq -e . <path> >/dev/null`. The wrapper performs the
schema and reproducibility validation after the child exits."""

WINDOWS_CONTROLLED_JSON_ARTIFACT_WRITE = (
    CONTROLLED_JSON_ARTIFACT_WRITE if os.name == "nt" else ""
)
WINDOWS_CONTROLLED_MATERIAL_PATCH_REPLACE = (
    "Windows decision-patch update: run exactly "
    "`content-workflow-cli artifact write-json --replace --output "
    "raw/material_decision_patch.json --json '<document>'` as its own command. "
    "The confined writer validates and atomically replaces only this path."
    if os.name == "nt"
    else ""
)
WINDOWS_CONTROLLED_PHYSICS_PATCH_REPLACE = (
    "Windows decision-patch update: run exactly "
    "`content-workflow-cli artifact write-json --replace --output "
    "raw/physics_decision_patch.json --json '<document>'` as its own command. "
    "The confined writer validates and atomically replaces only this path."
    if os.name == "nt"
    else ""
)


def usd_cli_output_usd_path(run_dir: Path) -> Path:
    """Return the saved-output USD path the usd-cli child contract requires."""

    return run_dir / USD_CLI_OUTPUT_USD_RELATIVE_PATH


# usd-cli backend fallback: without wrapper preflight the initial child still
# owns discovery, scene authoring, and final-render production. Optimized
# preflight runs use the smaller decision-only contract above; the wrapper then
# executes that exact patch and owns all mechanical evidence production.
CHILD_MATERIAL_ASSIGNMENT_ARTIFACTS_USD_CLI = (
    "raw/visible_candidate_prims.json",
    "raw/material_decision_patch.json",
    "raw/appearance_clear_report.json",
    "final_renders/",
    "output/materialized.usda",
    "trace/",
)
WRAPPER_MATERIAL_ASSIGNMENT_ARTIFACTS = (
    "raw/material_post_apply_review.json",
    "raw/material_vqa_review_packet.json",
    "raw/material_vqa_review_sheets/",
    "final_renders/final_turntable.gif",
    "raw/turntable_assembly_receipt.json",
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
# Required skills live in two trees: workflow skills under agentic/.agents/skills
# and repo-global skills (e.g. usd-cli) under .agents/skills. Child agents launch
# from one working directory, so their skill catalog never covers both — the
# prompt must carry explicit SKILL.md paths. Agentic tree wins for duplicates.
_SKILL_TREE_SUBDIRS = (
    Path("agentic") / ".agents" / "skills",
    Path(".agents") / "skills",
)


def _resolve_required_skill_paths(repo_root: Path, skills: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for skill in skills:
        for subdir in _SKILL_TREE_SUBDIRS:
            candidate = repo_root / subdir / skill / "SKILL.md"
            if candidate.is_file():
                resolved[skill] = str(candidate)
                break
    return resolved


def _skills_prompt_list(
    repo_root: Path, skills: list[str]
) -> tuple[str, dict[str, str]]:
    paths = _resolve_required_skill_paths(repo_root, skills)
    lines = [
        f"- `{skill}` — `{paths[skill]}`" if skill in paths else f"- `{skill}`"
        for skill in skills
    ]
    return "\n".join(lines), paths


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
    reference_images: list[Path] | None = None,
    reference_files: list[Path] | None = None,
    additional_instructions: str | None = None,
    behavior_prompt: str | None = None,
    scenario_path: Path | None = None,
    tune: bool = False,
    refine: bool = False,
    tune_engine: str | None = None,
    optimizer: str = "auto",
    max_trials: int = 30,
    max_iterations: int = 5,
    vomp_mass: PhysicsVompMassConfig | None = None,
    collision_approximation: str = "convexHull",
    visual_validation_max_iterations: int = 3,
    validation_max_penetration_m: float | None = None,
    classification_view_labels: list[str] | None = None,
) -> str:
    """Build the initial physics decision-patch prompt."""

    from content_agent_workflows.physics import (
        material_volume_fractions,
        vomp_contract_payload,
    )

    required_skills = ["usd-cli", "content-workflow-physics"]
    skills_list, required_skill_paths = _skills_prompt_list(repo_root, required_skills)

    behavior_goal = (behavior_prompt or "").strip() or None
    attached_views = [label for label in classification_view_labels or [] if label]
    if attached_views:
        view_lines = "\n".join(f"- {label}" for label in attached_views)
        classification_views_note = f"""
ATTACHED ASSET VIEWS: {len(attached_views)} OVRTX renders of the source
asset are attached to this request:
{view_lines}
Ground the material and physical-property inference in what the asset visibly
is. Read texture, surface finish, and shape from the images to choose the
material family, density, and friction/restitution priors; use the JSON
evidence for exact paths, bounds, and topology. When an image contradicts a
prim or material name, trust the image and say so in the rationale.
"""
    else:
        classification_views_note = ""
    # The material-evidence bullets below must only reference rendered views
    # when views are actually attached; on the render-outage and dry-run paths
    # an instruction to inspect nonexistent images invites fabricated view
    # citations in the rationale.
    if attached_views:
        metal_vs_plastic_evidence = (
            "look at the attached views for\n"
            "  specular highlights, brushed or anodised finish, and crisp "
            "reflective edges\n"
            "  (metal) versus a matte, diffuse, slightly translucent surface "
            "(plastic)"
        )
        material_evidence_citation = (
            "name the view (oblique/top/bottom) or the component/material "
            "name it came\n  from"
        )
    else:
        metal_vs_plastic_evidence = (
            "weigh the component and\n"
            "  material names, geometry, bounds, and any reference evidence; "
            "no rendered\n"
            "  views are attached to this turn, so do not cite one"
        )
        material_evidence_citation = (
            "name the component/material name or the JSON evidence field it "
            "came\n  from"
        )
    fill_fractions = ", ".join(
        f"{family} {fraction:g}"
        for family, fraction in material_volume_fractions().items()
    )
    agentic_physics = {
        "schema_version": "content-agents.physics-agentic-contract.v1",
        "behavior_prompt": behavior_goal,
        "scenario_path": str(scenario_path) if scenario_path is not None else None,
        "mass_properties": vomp_contract_payload(vomp_mass),
        "tuning": {
            "enabled": bool(tune or refine),
            "mode": "refine" if refine else ("tune" if tune else "validate"),
            "engine": tune_engine,
            "optimizer": optimizer,
            "max_trials": max_trials,
            "max_iterations": max_iterations,
            "protected_parameters": ["mass_scale"] if vomp_mass is not None else [],
            "allow_revise_patch": vomp_mass is None,
        },
    }

    wrapper_final_artifacts = list(WRAPPER_PHYSICS_APPLY_ARTIFACTS)
    if vomp_mass is not None:
        wrapper_final_artifacts.extend(
            [
                "raw/physics_vomp_result.json",
                "raw/physics_vomp_mass_properties.json",
                "vomp/finalize-*/evidence/",
            ]
        )

    constraints: dict[str, Any] = {
        "source_usd_edits_allowed": False,
        # Named `..._default` historically, but visual-validation refinement
        # treated that as licence to coarsen the collider (convexHull ->
        # convexDecomposition -> boundingSphere on a RoboCasa apple) to make
        # the penetration check pass. Keep the old key for compatibility and
        # state the requirement explicitly below.
        "collision_approximation_default": collision_approximation,
        "collision_approximation_required": collision_approximation,
        "visual_validation_max_iterations": visual_validation_max_iterations,
    }
    # Emit the penetration limit only when the run overrides it. A literal
    # `"max_ground_penetration_m": null` in the task block reads as a value to
    # mirror, and an explicit null DISABLES the numeric gate at
    # workflow runtime evaluator rather than selecting its default — the
    # opposite of what a default run enforces.
    if validation_max_penetration_m is not None:
        constraints["max_ground_penetration_m"] = validation_max_penetration_m
    task = {
        "schema_version": "content-agents.physics-apply-task.v2",
        "workflow": "physics.apply",
        "required_skills": required_skills,
        "required_skill_paths": required_skill_paths,
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "asset_path": str(usd_path),
        "reference_images": [str(path) for path in reference_images or []],
        "reference_files": [str(path) for path in reference_files or []],
        "agentic_physics": agentic_physics,
        "classification_views_attached": [
            label for label in classification_view_labels or [] if label
        ],
        "scene_backend": "usd-cli",
        "scene_tool": {
            "name": "usd-cli",
            "command": USD_CLI_WORKFLOW_COMMAND,
            "responsibility": "low-level scene operations only",
            "physics_run_packet_path": str(run_dir / "raw" / "physics_run_packet.json"),
            "components_path": str(run_dir / "raw" / "physics_components.json"),
            "topology_path": str(run_dir / "raw" / "physics_topology.json"),
        },
        "physics_run_packet_path": str(run_dir / "raw" / "physics_run_packet.json"),
        "components_path": str(run_dir / "raw" / "physics_components.json"),
        "topology_path": str(run_dir / "raw" / "physics_topology.json"),
        "constraints": constraints,
        "child_required_artifacts": list(CHILD_PHYSICS_APPLY_ARTIFACTS),
        "wrapper_final_artifacts": wrapper_final_artifacts,
    }
    if additional_instructions and additional_instructions.strip():
        task["additional_instructions"] = additional_instructions.strip()

    return f"""You are running a skill-routed agentic physics authoring workflow.

Load and follow these skills (each entry lists its SKILL.md; read it directly —
do not rely on your skill catalog, which may not cover both skill trees):
{skills_list}

Execution safety: never invoke Python or PyPy, run inline code with `-c` or a
here-document, create a scratch `.py` file, or import repository internals from
the shell. {JSON_SYNTAX_CHECK_GUIDANCE} The wrapper validates the decision patch after this
turn; do not call its internal Python validators yourself.

{CONTROLLED_JSON_ARTIFACT_WRITE}

Use the prepared workflow-owned component/topology packet. The child agent owns
physics reasoning, the decision patch, and any topology-plan proposal. The
wrapper owns topology-plan application, low-level scene-tool dispatch through
`usd-cli`, ovphysx runtime acceptance, frame rendering, visual review
orchestration, canonical final artifacts, and final validation evidence.
The scene tool must not choose component grouping, topology policy, property
ranges, or an acceptance verdict.
ovphysx/runtime metrics are authoritative for hard physics validation failures;
visual behavior review can make otherwise-passing results conditional but cannot
override a runtime failure.

Behavior/tuning contract:
- If `agentic_physics.behavior_prompt` is set, treat it as the durable behavior
  goal for decision authoring, runtime validation review, and follow-up patch
  refinement.
- If a scenario path is provided, read it as supporting scenario constraints; do
  not execute the legacy `apps/physics_agent` pipeline directly.
- If `agentic_physics.mass_properties.enabled` is true, VoMP is the authoritative
  source for final mass, center of mass, and inertia. The wrapper runs the
  attested VoMP phase after applying this patch and before runtime validation.
  Keep `mass_authoring_path` unambiguous; when no explicit VoMP target is set,
  every accepted decision must resolve to one shared mass-authoring path. Treat
  `estimated_mass_kg` as provisional reasoning evidence, not the final value.
- If tuning mode is `tune` or `refine`, express the required parameter changes
  as decision-patch/topology-plan edits that the wrapper can apply and validate
  through usd-cli. Keep solver/runtime metrics authoritative.

Structured task:
```json
{json.dumps(task, indent=2)}
```
{classification_views_note}
Decision task:
- Inspect `raw/physics_components.json`, `raw/physics_topology.json`, any
  reference evidence, and the `agentic_physics` behavior/tuning contract. Treat
  visual, collider, and helper paths as distinct roles.
- Select collider targets only by copying `target_id` values from each
  component's `authoring_targets` list. Use `visual` target IDs with
  `author_on_targets` and `collider` target IDs with `preserve_existing`.
  Never type or reconstruct USD prim paths in the decision patch; the wrapper
  resolves target IDs to the inspected paths.
- For each logical component, write exactly one `decisions` entry or one
  `unresolved_components` entry. Never target helper paths.
- Infer density, estimated mass, static/dynamic friction, restitution, collider
  approximation, confidence, and rationale.
- `bounds_m.volume_m3` is an axis-aligned bounding-box volume, not the solid
  volume the part occupies. Estimate mass as density x bounding volume x the
  fraction of the box the part actually fills; typical fill fractions by
  family are {fill_fractions}. Adjust when the geometry is visibly solid or
  hollow, and state the fraction you used in the rationale.
- Exception: a component whose `component_role` is `unowned_static` (a floor,
  wall, or fixture the asset merely rests on) takes no mass -- author
  `density` 0.0 and `estimated_mass_kg` 0.0 for it, keeping its friction and
  restitution real values from its material, so fixture volume never inflates
  the simulated body's mass.
- Keep `quality_warnings` empty unless a concrete warning applies. A suspicious
  mass/scale warning must use `{{"code": "mass_scale_suspicious", "severity":
  "warning", "message": "..."}}`; do not use free-form strings.
- Decide the material family FIRST, from evidence, then derive every physical
  property from it -- never from the schema example below, which shows shape,
  not values. Density is the clearest tell: reusing a remembered plastic-range
  density instead of deriving it silently commits the part to plastic.
- Weight the component's own name heavily. A part called lens, mirror, window,
  glazing or screen is glass; blade, spring, screw, hinge or fastener is metal;
  gasket, seal, grip or bumper is rubber. Overriding an obvious name needs a
  stated reason in the rationale.
- Do not default a device or appliance housing to plastic. Consumer cameras,
  toasters, kettles, switch plates and appliance shells are frequently metal.
  Before choosing between metal and plastic, {metal_vs_plastic_evidence}.
- In the rationale, cite the evidence you actually used for the material call --
  {material_evidence_citation}. "Inferred from context" is not acceptable for a
  metal-vs-plastic call.
  Restitution in particular is a material property, and reference coefficients
  for an impact against a hard surface are: rubber 0.7-0.85, metal 0.5-0.65,
  hard plastic 0.5-0.65, glass 0.4-0.6, ceramic 0.4-0.55, wood 0.4-0.55.
  Author restitution below 0.1 only when the part is genuinely inelastic (soft
  foam, fabric, loose granular fill, a thin damped shell) -- a near-zero
  restitution means a body that lands dead with no rebound, which is wrong for
  ordinary rigid materials. State in the rationale which family drove the
  number.
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
      "collider_target_ids": ["copy-an-exact-authoring-target-id"],
      "collision_mode": "preserve_existing|author_on_targets",
      "inferred_material_family": "glass|metal|plastic|rubber|wood|soft|generic",
      "inferred_material_name": "optional existing material name or null",
      "collision_approximation": "{collision_approximation}",
      "physical_properties": {{
        "density": 0.0,
        "estimated_mass_kg": 0.0,
        "static_friction": 0.0,
        "dynamic_friction": 0.0,
        "restitution": 0.0
      }},
      "rigid_body_grouping": "optional grouping description or null",
      "quality_warnings": [],
      "confidence": 0.7,
      "rationale": "reasoning grounded in component roles, material, bounds, and topology"
    }}
  ],
  "unresolved_components": [
    {{"component_id": "component_002", "reason": "specific missing evidence"}}
  ]
}}
```

Every `physical_properties` value above is a `0.0` placeholder that only fixes
the JSON types: the schema requires plain numbers, and quoted strings fail
validation, and a decision whose values are all 0.0 is rejected as a copied
placeholder. Do not copy them: author each component's values from its
own inferred material family -- density in kg/m^3 from that family, estimated
mass as density x bounding volume x fill fraction (see the Decision task),
friction and restitution from the reference band above.

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
    iteration: int,
    max_iterations: int,
    decision_patch_path: Path,
    validation_evidence_path: Path,
    runtime_report_path: Path | None,
    rendered_frames: list[str],
    previous_assessment_path: Path | None = None,
    issue_packet_path: Path | None = None,
    collision_approximation: str | None = None,
    validation_max_penetration_m: float | None = None,
    defer_behavior_goal_until_tuning: bool = False,
) -> str:
    """Build a post-runtime visual behavior review/refinement prompt."""

    required_skills = ["usd-cli", "content-workflow-physics"]
    skills_list, required_skill_paths = _skills_prompt_list(repo_root, required_skills)
    task = {
        "schema_version": "content-agents.physics-visual-review-task.v1",
        "workflow": "physics.apply.visual_review",
        "required_skills": required_skills,
        "required_skill_paths": required_skill_paths,
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "asset_path": str(usd_path),
        "scene_backend": "usd-cli",
        "scene_tool": {
            "name": "usd-cli",
            "command": USD_CLI_WORKFLOW_COMMAND,
            "responsibility": "low-level scene operations only",
        },
        "iteration": iteration,
        "max_iterations": max_iterations,
        "decision_patch_path": str(decision_patch_path),
        "physics_agentic_contract_path": str(
            run_dir / "raw" / "physics_agentic_contract.json"
        ),
        "validation_evidence_path": str(validation_evidence_path),
        "runtime_report_path": str(runtime_report_path)
        if runtime_report_path
        else None,
        "rendered_frames": rendered_frames,
        "previous_assessment_path": (
            str(previous_assessment_path) if previous_assessment_path else None
        ),
        "issue_packet_path": str(issue_packet_path) if issue_packet_path else None,
        "authored_collision_approximation": collision_approximation,
        "validation_scope": (
            "pre_tuning_baseline"
            if defer_behavior_goal_until_tuning
            else "final_behavior"
        ),
    }
    # Emit the penetration limit only when the run overrides it: a literal
    # `"max_ground_penetration_m": null` in the task block reads as a value
    # to mirror, and an explicit null disables the numeric gate at
    # the workflow runtime evaluator instead of selecting its default.
    if validation_max_penetration_m is not None:
        task["max_ground_penetration_m"] = validation_max_penetration_m

    # Visual refinement may rewrite the decision patch, and it hits the same
    # degenerate optimum the tuning loop does: coarsening the collider makes the
    # penetration check pass by removing the body's ability to roll. Observed
    # convexHull -> convexDecomposition -> boundingSphere across two refinement
    # iterations on a RoboCasa apple, driven by a penetration limit the child
    # believed was 0.005 m when the run had configured 0.05 m.
    refinement_constraints = ""
    if validation_max_penetration_m is not None:
        refinement_constraints += (
            "\n- GROUND-PENETRATION LIMIT: this run is configured with "
            f"`max_ground_penetration_m={validation_max_penetration_m}`, NOT the "
            "workflow default (scale-relative on exact collider geometry, "
            "0.005 m on the conservative bbox fallback). Pass that acceptance "
            "override to the workflow runtime evaluator; do not treat a rest "
            "deeper than the default as a hard failure when it is inside the "
            "configured limit."
        )
    else:
        refinement_constraints += (
            "\n- GROUND-PENETRATION LIMIT: this run configures no override, so "
            "the workflow default applies (scale-relative on exact collider "
            "geometry, 0.005 m on the conservative bbox fallback). Run "
            "workflow runtime validation WITHOUT `max_ground_penetration_m` so your "
            "evidence measures the same gate; do not send "
            "`max_ground_penetration_m: null` — an explicit null disables the "
            "numeric gate rather than selecting the default."
        )
    if collision_approximation:
        refinement_constraints += (
            "\n- PRESERVE THE COLLISION SHAPE: the decision patch authored "
            f"collision approximation(s) `{collision_approximation}`. Fix "
            "behavior with "
            "material values (density/mass, friction, restitution); do NOT "
            "coarsen the collider to a bounding primitive to make a metric "
            "pass. A shape that cannot roll trivially satisfies settle and "
            "penetration checks while destroying the requested behavior."
        )

    behavior_scope = ""
    if defer_behavior_goal_until_tuning:
        behavior_scope = """

Pre-tuning validation scope:
- This pass validates baseline physics authoring and runtime safety only.
- Goal-level visual acceptance is deferred until candidate-specific tuning
  evidence is available and the wrapper's post-promotion assessment reviews
  the selected scenario rollout.
- Do not reject this baseline because it has not yet matched the behavior
  prompt, scenario objective, or reference images. In particular, missing
  target motion such as a slide or bounce is not an unresolved issue here.
- Still reject genuine baseline defects such as solver failure, non-finite or
  explosive motion, visible interpenetration, separated rigid parts, blank or
  stale renders, and invalid collision behavior.
"""

    return f"""You are reviewing rendered physics simulation behavior for an
agentic physics authoring run.

Load and follow these skills (each entry lists its SKILL.md; read it directly —
do not rely on your skill catalog, which may not cover both skill trees):
{skills_list}

Execution safety: never invoke Python or PyPy, run inline code with `-c` or a
here-document, create a scratch `.py` file, or import repository internals from
the shell. {JSON_SYNTAX_CHECK_GUIDANCE} The wrapper owns deterministic patch
validation and runtime evaluation.

{WINDOWS_CONTROLLED_JSON_ARTIFACT_WRITE}
{WINDOWS_CONTROLLED_PHYSICS_PATCH_REPLACE}

This is visual validation/refinement iteration {iteration} of {max_iterations}.
ovphysx/runtime metrics are authoritative for hard failures. Your visual review
is a semantic check over the rendered simulation frames and runtime report.

Structured task:
```json
{json.dumps(task, indent=2)}
```
{behavior_scope}

Review contract:
- Read `raw/physics_agentic_contract.json` before judging behavior so any
  behavior prompt, scenario, tuning/refine mode, reference images, and budgets
  remain durable across turns.
- Inspect the rendered frame images directly.
- Read the runtime report and validation evidence.
- Treat `scene_info.ground_clearance_support_decision` as validation
  measurement metadata, never as the authored collider. Multi-body runtime
  reports carry the same records at `ground_clearance_support_decisions` and
  at `per_body_results[].ground_clearance_support_decision`. When a record states
  `fallback_accepted: true`, the conservative support fallback is non-actionable:
  do not change collider targets or approximation unless another specific
  runtime or visual failure identifies the collider as the cause.
- Check for parts visibly separating or moving independently when they should be
  one rigid object, no visible motion under gravity, implausible bounce/sliding,
  obvious interpenetration/tunneling, stale or blank renders, misframed renders,
  and mismatches between runtime metrics and visible behavior.{refinement_constraints}
- Write `{run_dir}/physics_behavior_assessment.json` with:
  `schema_version`, `status`, `checked_views`, `runtime_report`,
  `rendered_frames`, `issues_found`, `issues_fixed`, `unresolved_issues`, and
  `assessment_notes`.
- In `checked_views`, write only exact paths copied from the task-provided
  `rendered_frames` list. Never put phase names or prose labels there; put those
  descriptions in `assessment_notes` instead.
- `runtime_report` and `rendered_frames` are optional evidence claims. Omit
  either when unused; otherwise reference only the task-provided runtime report
  and any subset of task-provided rendered frames, in any order.
- Use `status: "pass"` when the behavior is plausible, `status: "fixed"` when
  you updated `raw/physics_decision_patch.json` to address a fixable issue, and
  `status: "unresolved_issues"` when issues remain after the available fix.
- If a visual issue is fixable by changing body grouping, collider
  approximation, density/mass, friction, or restitution, update
  `raw/physics_decision_patch.json` in the same schema as the initial patch.
- Preserve `collider_target_ids` from the initial patch. If target selection
  must change, choose only IDs present in `raw/physics_components.json`; never
  replace them with handwritten USD paths.
- Do not edit source USD files or write canonical final artifacts. The wrapper
  will reapply any changed patch, rerun ovphysx, rerender, and merge evidence.

Finish with a short response listing visual status, any patch changes, and
remaining limitations.
"""


def build_physics_post_tuning_assessment_prompt(
    *,
    repo_root: Path,
    run_dir: Path,
    promoted_usd: Path,
    promoted_usd_sha256: str,
    validation_evidence_path: Path,
    runtime_report_path: Path,
    rendered_frames: list[str],
    reference_frames: list[dict[str, str]],
    assessment_path: Path,
    behavior_prompt: str | None,
    scenario_path: Path | None,
    additional_instructions: str | None = None,
    review_context: dict[str, Any] | None = None,
) -> str:
    """Build the final, tool-free assessment of a tuned physics USD."""

    # Keep the established public builder signature stable. The final reviewer
    # is deliberately tool-free, so the parent reduces the relevant checked-in
    # workflow rules and exact evidence into ``review_context`` instead of
    # asking the model to read skills or mutable run artifacts itself.
    del repo_root
    task = {
        "schema_version": "content-agents.physics-post-tuning-review-task.v1",
        "workflow": "physics.apply.agentic_tuning.final_visual_review",
        "run_dir": str(run_dir),
        "promoted_usd": str(promoted_usd),
        "promoted_usd_sha256": promoted_usd_sha256,
        "behavior_prompt": behavior_prompt,
        "additional_instructions": additional_instructions,
        "scenario_path": str(scenario_path) if scenario_path is not None else None,
        "reference_frames": reference_frames,
        "validation_evidence_path": str(validation_evidence_path),
        "runtime_report_path": str(runtime_report_path),
        "rendered_frames": rendered_frames,
        "wrapper_assessment_path": str(assessment_path),
        "review_context": review_context or {},
        "validation_scope": "post_tuning_final_behavior",
    }
    return f"""You are performing the final visual behavior assessment for the
exact physics USD selected by an agent-owned tuning loop. This is an
assessment-only turn: tuning is over and you must not edit any USD, scenario,
decision patch, or canonical workflow artifact.

This is a tool-free structured-output turn. The wrapper has embedded the
validated scenario, runtime report, and validation evidence in `review_context`
and attached the sampled images. Do not call tools or write files. The wrapper
will validate your structured response and author the assessment itself.

Structured task:
```json
{json.dumps(task, indent=2)}
```

Review contract:
- Inspect the attached task-provided rendered frames directly and use the exact
  runtime report, validation evidence, and accepted scenario embedded in
  `review_context`.
- Judge the final behavior against the complete user request: behavior prompt,
  additional instructions, scenario objective and conditions, and any attached
  reference frames. Reference frames and generated frames are labelled
  separately in the task and image attachments.
- Treat runtime failures, non-finite or explosive motion, visible penetration,
  separated rigid parts, stale/blank evidence, and goal mismatch as unresolved.
- Use `status: "pass"` only when the exact tuned candidate is safe and visibly
  satisfies the requested behavior. Otherwise use `status: "unresolved_issues"`
  and state the concrete mismatch. Never use `status: "fixed"` in this
  assessment-only turn.
- Return only the schema-constrained fields `status`, `unresolved_issues`, and
  `assessment_notes`. `status: "pass"` requires an empty `unresolved_issues`
  list; `status: "unresolved_issues"` requires at least one concrete issue.
  The wrapper binds the exact runtime report and reviewed frame paths.
- Do not write `physics_behavior_assessment.json`, `validation_evidence.json`,
  `physics_assignments.json`, `final_summary.md`, or any other file; the wrapper
  publishes the reviewed evidence only after it validates this response and its
  render receipt.

Return the structured response only.
"""


def build_skill_routed_material_assignment_prompt(
    *,
    repo_root: Path,
    run_dir: Path,
    usd_path: Path,
    reference_images: list[Path],
    materials_yaml: Path,
    materials_usd: Path,
    reference_files: list[Path] | None = None,
    optimize: bool = True,
    optimizer_options: dict[str, object] | None = None,
    material_candidate_policy: dict[str, object] | None = None,
    respect_existing_material_bindings: bool = False,
    additional_instructions: str | None = None,
    preflight_packet: dict[str, object] | None = None,
    vqa_refinement_max_iterations: int = 3,
    usd_cli_optimization: dict[str, object] | None = None,
) -> str:
    """Build a compact child-agent prompt that routes method through skills."""

    candidate_policy = _material_candidate_policy(
        material_candidate_policy,
        preflight_packet,
    )
    required_skills = ["usd-cli", "content-workflow-material"]
    skills_list, required_skill_paths = _skills_prompt_list(repo_root, required_skills)
    optimized_inspection = (
        isinstance(usd_cli_optimization, dict)
        and usd_cli_optimization.get("enabled") is True
    )
    material_apply_path_key = (
        "runtime_prim_paths" if optimized_inspection else "prim_paths"
    )
    candidate_path_space = "inspection" if optimized_inspection else "source"
    inspection_asset_path = (
        str(usd_cli_optimization.get("inspection_usd_path"))
        if optimized_inspection
        else str(usd_path)
    )
    scene_tool_block = {
        "scene_tool": {
            "name": "usd-cli",
            "command": USD_CLI_WORKFLOW_COMMAND,
            "skill": "usd-cli",
            "inspection_asset_path": inspection_asset_path,
            "authoring_asset_path": str(usd_path),
            "output_usd_path": str(usd_cli_output_usd_path(run_dir)),
        }
    }
    task = {
        "schema_version": "content-agents.skill-routed-task.v1",
        "workflow": "materials.assign",
        "scene_backend": "usd-cli",
        "required_skills": required_skills,
        "required_skill_paths": required_skill_paths,
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "asset_path": inspection_asset_path,
        "source_asset_path": str(usd_path),
        "reference_images": [str(path) for path in reference_images],
        "reference_files": [str(path) for path in reference_files or []],
        "material_library": {
            "materials_yaml": str(materials_yaml),
            "materials_usd": str(materials_usd),
        },
        **scene_tool_block,
        "material_candidate_policy": candidate_policy,
        "candidate_artifact_contract": {
            "path_space": candidate_path_space,
            "candidate_path_key": (
                "runtime_path"
                if candidate_path_space == "inspection"
                else "source_path"
            ),
        },
        "constraints": {
            "use_usd_cli_only": True,
            "source_usd_edits_allowed": False,
            "respect_existing_material_bindings": respect_existing_material_bindings,
            "clear_materials": not respect_existing_material_bindings,
            "predict_only_canonical_material_candidates": True,
            "vqa_refinement_max_iterations": vqa_refinement_max_iterations,
        },
        "child_required_artifacts": list(CHILD_MATERIAL_ASSIGNMENT_ARTIFACTS_USD_CLI),
        "wrapper_final_artifacts": list(WRAPPER_MATERIAL_ASSIGNMENT_ARTIFACTS),
    }
    if optimized_inspection:
        task["scene_optimization"] = usd_cli_optimization
    preflight_enabled = (
        isinstance(preflight_packet, dict) and preflight_packet.get("enabled") is True
    )
    decision_only = bool(
        preflight_enabled
        and preflight_packet.get("wrapper_executes_decision_patch") is True
    )
    if decision_only:
        task["execution_mode"] = "decision_only_wrapper_execution"
        task["child_required_artifacts"] = []
    if preflight_enabled:
        if decision_only:
            task["preflight"] = {
                "enabled": True,
                "candidate_visible_prim_count": preflight_packet.get(
                    "candidate_visible_prim_count"
                ),
                "decision_catalog": preflight_packet.get("decision_catalog") or [],
                "segmentation_candidate_legend": preflight_packet.get(
                    "segmentation_candidate_legend"
                )
                or [],
                "decision_palette": [
                    {key: value for key, value in row.items() if key != "material_path"}
                    for row in preflight_packet.get("decision_palette") or []
                    if isinstance(row, dict)
                ],
                "evidence_views": [
                    {
                        "name": row.get("name"),
                        "direction": row.get("direction"),
                    }
                    for row in preflight_packet.get("initial_evidence_renders") or []
                    if isinstance(row, dict)
                ],
            }
        else:
            task["preflight"] = preflight_packet
    if additional_instructions and additional_instructions.strip():
        task["additional_instructions"] = additional_instructions.strip()

    if preflight_enabled:
        inspection_instruction = """
- The wrapper already built the canonical candidate universe and attached three
  standardized OVRTX beauty views plus their segmentation views. Inspect every
  attached view directly; use `preflight.segmentation_candidate_legend` to map
  segmentation RGB values to candidate IDs, and use the candidate table/context
  for exact path and semantic metadata. Do not repeat the general scene survey, rebuild or
  overwrite a preflight artifact, or rerender the same standard views. At most
  one targeted additional render is allowed when a specific decision remains
  visually ambiguous; record the exact ambiguity it resolves.
"""
        candidate_instruction = """
- `raw/visible_candidate_prims.json` is wrapper-owned, hash-pinned canonical
  coverage evidence. Read it, but do not rewrite, replace, reconcile, or broaden
  it. Every decision-patch target must come from that artifact, and its
  `candidate_visible_prim_count` must be copied exactly as `candidate_count`.
"""
        post_clear_candidate_instruction = """
- In optimized preflight runs, candidate discovery was performed on the
  wrapper-owned clean inspection derivative. The authoring-source `appearance
  clear` remains mandatory, but do not run another visible-mesh query or rebuild
  the already hash-pinned candidate universe after the clear.
"""
    else:
        inspection_instruction = """
- Inspect the scene yourself with scoped usd-cli reads (`snapshot`, `find`,
  `render`, `material-binding`).
"""
        candidate_instruction = """
- Write `raw/visible_candidate_prims.json` before decisions. It must contain
  `schema_version: "content-agents.visible-candidate-prims.v1"`, the exact
  `path_space` and `candidate_path_key` from `candidate_artifact_contract`,
  `candidate_visible_prim_count`, and one unique absolute path under that key
  per object in `candidates`. The only valid `path_space` values are the literal
  strings `source` and `inspection`; never write `optimized_inspection`.
  `candidate_visible_prim_count` must equal `len(candidates)` exactly.
"""
        post_clear_candidate_instruction = """
- In a clean-slate authoring session, `appearance clear` deinstances native
  instance roots. Run a fresh visible-mesh query after the successful clear and
  build `raw/visible_candidate_prims.json` from that post-clear stage; never
  reuse a pre-clear proxy-only query.
"""

    if decision_only:
        return f"""You are the material-decision stage of a skill-routed asset workflow.

The wrapper pinned these skill sources as contract provenance:
{skills_list}

This is a decision-only turn. The wrapper already built and hash-pinned the
canonical candidate universe and attached three standardized OVRTX beauty views,
their segmentation views, and the user reference images. Inspect every attached
image directly, then return the material decision as the required structured
response. Do not call tools, read files, write artifacts, or run shell commands.
All required candidate and palette data is embedded below. The wrapper owns
candidate-ID expansion, exact-path validation, appearance clearing, checkpointing,
batch application, audits, save/reopen, verification renders, the turntable, and
fresh post-apply VQA after this turn.

Structured task:
```json
{json.dumps(task, indent=2)}
```

Decision procedure:
- Use `preflight.decision_catalog` as the complete candidate universe. Its
  zero-based `candidate_id` values are the only target identifiers you may
  return. Use `preflight.decision_palette` as the complete allowed material
  palette and copy each selected `name` exactly.
- Use the attached references, beauty views, segmentation views, semantic hints,
  shape hints, part hierarchy, repeated-part consistency, and material finish to
  choose the closest supplied library material for every candidate. Preserve
  deliberate color accents and functional distinctions; do not collapse the
  asset into one blanket material merely to shorten the response.
- `source_material_hints` preserve only the nearest authored binding names from
  the immutable source before appearance was cleared. Use them to recognize
  coherent part families and accelerate repeated-part grouping, but treat them
  as non-authoritative semantic hints: the attached reference and rendered
  geometry win whenever an old name conflicts with visible intent.
- Return schema version `content-agents.material-decision-intent.v1`. Each
  `material_assignments` group contains one exact palette `material_name`, a
  unique non-empty `candidate_ids` array, and a concise evidence-grounded
  rationale. The wrapper expands these IDs into all exact runtime and source
  paths without model-authored file mechanics.
- The union of `material_assignments[].candidate_ids` and
  `reviewed_no_override_candidate_ids` must cover every candidate ID exactly
  once. In clean-slate mode, `reviewed_no_override_candidate_ids` must be empty.
- Group candidates only when their selected material is identical. Keep the
  complete candidate-ID membership even for large repeated families.
- Return only the schema-constrained structured response. The wrapper validates,
  expands, and executes it deterministically.
"""

    return f"""You are running a skill-routed agentic asset workflow.

The wrapper pinned these skill sources as contract provenance:
{skills_list}

The complete applicable instructions and exact command forms are materialized
below. Do not reread those skill files, their references, repository docs, or
`--help` screens in this turn; doing so only duplicates the frozen prompt and
inflates every later model action. The wrapper has already verified the exact
package-owned CLI and renderer capabilities.

Execution safety: never invoke Python or PyPy, run inline code with `-c` or a
here-document, create a scratch `.py` file, or import repository internals from
the shell. {JSON_SYNTAX_CHECK_GUIDANCE} The wrapper validates canonical artifacts after this
turn.

{CONTROLLED_JSON_ARTIFACT_WRITE}

Use this frozen task contract for usd-cli mechanics and material assignment
policy. Invoke every scene command through the package-owned
`usd-cli-tel` executable (the command arguments are otherwise identical to
`usd-cli`) so the workflow trace is complete. Reuse the wrapper's inherited
session selection exactly: never set, unset, or override `USD_CLI_SESSION`, and
never pass `--session`. The wrapper supplies a named session when required and
otherwise selects the project default. You own scene state through applied
output and final-render production: inspect the supplied evidence, make material
decisions, apply material bindings under checkpoints, and render verification
evidence per the `content-workflow-material` skill. A fresh wrapper-launched turn owns the
post-apply VQA. The wrapper owns `assignments.json`,
`visual_quality_assessment.json`, `api_operation_counts.json`,
`validation_evidence.json`, `final_summary.md`, run-directory trace
composition, bounded extra refinement turns, and deterministic contract
validation after you exit. Do not write or patch those wrapper-owned files.

{inspection_instruction}

Structured task:
```json
{json.dumps(task, indent=2)}
```

Required behavior:
- Inspect attached reference images directly before making material decisions.
- Inspect non-image reference files through targeted local file reads.
- If inspecting a reference file requires a browser or document renderer, put
  its user-data/profile and all lock, socket, and scratch state in a temporary
  directory outside `run_dir`. Copy back only required regular-file evidence;
  never create browser profile state, symlinks, sockets, or special files under
  the run directory.
- When `scene_optimization.enabled` is true, use two explicit phases:
  1. Open only `inspection_asset_path` to inspect/render the wrapper-owned
     optimized representation. When `clear_materials` is true, the wrapper has
     already cleared, audited, flattened, reopened, and re-audited this
     derivative before child launch. Do not substitute another optimizer output,
     bind materials, or save the optimized stage.
  2. Translate every accepted inspection prim through the run-local
     `correspondence_path`. Each inspection path must resolve to exactly one
     source path; stop with a nonzero result if a path is absent or ambiguous.
     Then open `authoring_asset_path` (replacing the inspection stage) and, after
     the required clean-slate clear has de-instanced it, apply bindings to the
     live composed consumer paths matching `runtime_prim_paths`. Do not bind only
     the translated backing/prototype paths: `source_prim_paths`/`prim_paths`
     record the exact correspondence targets for wrapper-owned durable restore.
     Save only to `output_usd_path`.
- For an optimized run, record both `runtime_prim_paths` (optimized inspection
  paths) and `source_prim_paths`/`prim_paths` (authoring paths) in every
  decision and assignment group. Keep the two path spaces distinct even when
  the live audit succeeds on the runtime consumer: never replace a translated
  `source_prim_paths`/`prim_paths` entry with that runtime path. Never apply
  accepted bindings to the optimized inspection artifact.
- Every `prim_paths`, `source_prim_paths`, and `runtime_prim_paths` value must
  be a JSON array containing the exact absolute prim paths. Never replace that
  array with a count, prose such as "all candidates", a glob, or a reference
  to the candidate artifact. When one material covers every candidate, load
  the candidate artifact and write its complete path list into the group.
- Resolve the material library from `materials_yaml`/`materials_usd` and bind
  through `usd-cli-tel material --library`, never by editing source USD files.
  The wrapper has already validated the manifest and written the authoritative
  `raw/material_palette.json`; do not replace or reinterpret that palette.
{candidate_instruction}
- Candidate paths must be the composed renderable prims that visibly contribute
  to the primary asset. When a stage contains backing definitions such as a
  top-level `/meshes` library plus consumer prims under the asset root, select
  the consumer/render paths—not the backing definitions, and never both. Prove
  the chosen universe with the inspection render and scoped binding reads before
  writing the decision patch.
- When `clear_materials` is true, the optimized inspection derivative is already
  clean-slate evidence. Independently run `usd-cli-tel appearance clear`
  immediately after opening the authoring source and verify the result with
  `usd-cli-tel appearance audit` before using visual appearance as evidence.
  Save the raw successful clear response and post-clear audit in
  `raw/appearance_clear_report.json` with schema
  `content-agents.appearance-clear-report.v1`, capability
  `appearance.clear.v1`, status `pass`, the absolute source path,
  `source_sha256_before`, `source_sha256_after`, `source_unchanged:true`, and
  `cli_response`. Read the staged source digest from
  `request.json.material_contract.staged_source.staged_usd_sha256` (or the
  equivalent `raw/staged_input_source.json.source_sha256`); do not invent a
  shorter request path or write a null/string-null digest. The CLI response
  must retain schema `1`, command
  `appearance-clear`, `ok:true`, `summary.clear:true`, and
  `data.audit` with `clear:true`, `overlay_active:true`, and every effective
  appearance count equal to zero. Preserve all six required count keys exactly:
  `binding_relationships_with_targets`, `effective_material_bindings`,
  `effective_shader_appearances`, `direct_shader_outputs`, `display_values`,
  and `instance_proxies`.
{post_clear_candidate_instruction}
- `skip_instances:true` forbids unresolved instance-proxy decision targets—it
  does not waive their visible families or justify `candidate_count:0`.
- Apply bindings after `checkpoint save`; discard rejected previews with
  `undo`.
- Once `raw/material_decision_patch.json` is complete, apply its exact
  heterogeneous assignments in one serialized daemon transaction:
  `usd-cli-tel --json material-apply raw/material_decision_patch.json
  --library <absolute materials_usd path> --path-key {material_apply_path_key}`.
  This command
  imports each selected library material once and binds every exact tuple from
  the patch without target inference. Do not expand the patch into one shell or
  model tool call per prim. Keep `usd-cli-tel material` for a genuinely
  surgical one-target preview or repair only.
- Keep inspection output bounded. Never run `snapshot -a` on a mesh-heavy
  scope or print point, face-index, normal, or UV arrays into the agent
  context; query only the prims and named metadata needed for the decision.
- The project daemon is one serialized scene session. Never issue parallel
  shell/tool calls when any call contains `usd-cli-tel`; wait for the complete
  JSON response before issuing the next scene command. The OVRTX worker is a
  persistent backend process, so its PID is not render-progress evidence:
  never poll `ps` or `kill -0` to infer render completion.
- The wrapper has already passed the required renderer preflight and sealed its
  authoritative receipt at `raw/ovrtx_probe.json`. Do not invoke
  `usd-cli-tel render-probe` again from the child turn; use the existing scene
  session directly for the required renders.
- After the live binding audit passes, preserve a post-apply checkpoint, save
  `output_usd_path` with `--flatten`, reopen it, and verify durable binding
  coverage before starting the required long turntable render. Repeat this
  save/reopen gate after any accepted refinement. Treat every nonzero
  `usd-cli-tel` exit as a failed operation: inspect
  the error, retry or revise the decision, and never claim that candidate as
  covered until a successful binding is observable.
- Author `raw/material_decision_patch.json` before applying accepted bindings
  and use it as the single execution plan. Pass that patch directly to
  `material-apply`; never recalculate choices in a separate shell `case`, regex,
  default branch, generated command script, or ad hoc candidate loop.
  After binding, audit every tuple against the patch and correct any mismatch
  before render/save. After reopen, repeat that patch-to-audit comparison before
  claiming success.
- If visual review proves that the patch targeted a non-rendering backing path,
  restore the pre-bind checkpoint (or undo the whole attempted patch), rewrite
  both the candidate artifact and decision patch to the corrected renderable
  paths, and only then bind again. Never layer a second target universe on top
  of the first attempt; the saved USD may contain only bindings authorized by
  the final patch.
- Serialize every `usd-cli-tel` call in this session. Wait for one command to
  exit before starting the next; never launch overlapping tool calls, background
  jobs, parallel shells, or concurrent pipelines against the stage. For a
  multi-step shell sequence, use `set -euo pipefail` followed by a command
  separator so the shell stops at the first nonzero exit. Do not use `;` to
  continue into render, save, reopen, or audit after an earlier command fails.
- Prefer material verification renders at or below 768x768. Initial OVRTX
  shader compilation can legitimately take many minutes after fast clean-slate
  inspection renders. While the renderer process remains alive and is making
  CPU or GPU progress, wait for the current `usd-cli-tel render` command to
  return or reach its configured timeout. Do not cancel its client, signal the
  sidecar or OVRTX worker, or start a competing sidecar merely because no PNG
  has appeared yet.
- Run every agent render as one daemon-owned detached job. Invoke
  `usd-cli-tel --json render ... --detach` (including `--orbit 24` for the
  turntable), capture its job ID, inspect it only with
  `usd-cli-tel --json jobs`, and collect it with
  `usd-cli-tel --json wait <job>`. Finish one render job before submitting the
  next. Never launch a render as an untracked synchronous/background shell
  command, and never enqueue a competing render, history, save, or inspection
  command while a synchronous render owns the session.
- Write every artifact in `child_required_artifacts` under the run directory,
  including final render PNGs in `final_renders/`. Do not hand-assemble
  `raw/material_binding_audit.json`, `raw/material_operation_receipts.json`,
  `raw/final_render_records.json`, or `raw/material_application_receipt.json`;
  the wrapper derives those receipts from the saved output, decision patch,
  exact render set, and captured usd-cli command evidence.
- After all required final render jobs complete, return immediately. Do not
  inspect the 24 turntable frames one by one, search for an image compositor,
  or write `raw/material_post_apply_review.json`; the wrapper deterministically
  builds hash-bound review sheets and launches a fresh compact VQA turn. Do not
  inventory history/jobs again, construct wrapper-owned manifests, reformat
  receipts into alternate schemas, or spend another turn asserting
  mechanically checkable counts.
- Save the materialized result with
  `usd-cli-tel save <output_usd_path> --flatten`; clean-slate appearance is a
  session overlay and must never be saved back to the source.
- Treat checkpoints only as recovery state. Never copy or rename checkpoint
  bytes into `output_usd_path`, and never substitute a different extension or
  encoding. The durable output must be produced by the successful
  `usd-cli-tel save` command above and reopened with
  `usd-cli-tel open <output_usd_path> --force-reload` before success. Do not
  synthesize a passing reopened audit after a failed reopen.
- Preserve concise trace evidence under `trace/` for renders, binds, and
  ambiguity resolution.
- Write `raw/material_decision_patch.json` with exact schema version
  `content-agents.material-decision-patch.v1`, both
  `material_assignments` and `reviewed_no_override` lists, exact material names
  and paths from `raw/material_palette.json`, and exactly one decision for each
  canonical candidate. In clean-slate mode `reviewed_no_override` must be empty.
  A free-form `coverage` description never substitutes for target paths. For a
  source-space run, every assignment group has this exact minimum shape:
  `{{"material_name":"<palette name>","material_path":"<palette path>","prim_paths":["/absolute/candidate/path"]}}`.
  For an optimized inspection run, every group additionally has matching
  non-empty `runtime_prim_paths` plus translated `source_prim_paths` and
  `prim_paths`. Set `candidate_count` to the exact canonical candidate count.
  A genuine zero-candidate result must set `candidate_count:0`.

Finish with a short response pointing to the run directory, decision patch,
final renders, and saved output USD. State that post-apply VQA is pending in the
wrapper-owned fresh review turn.
"""


def build_material_post_apply_review_prompt(
    *,
    review_packet: dict[str, object],
    review_packet_sha256: str,
) -> str:
    """Build the tool-free prompt for one independent post-apply VQA turn."""

    return f"""You are the independent post-apply visual-quality reviewer for a material-assignment workflow.

Review goal:
- Compare the attached current-output review sheets against the attached reference images and the requested visual intent.
- Inspect every tile in every sheet. The packet maps each tile to an exact hash-bound final PNG, so reviewing the sheets covers the complete final render set.
- Judge visible material appearance, including brightness, hue, saturation, finish, metalness, transparency, part hierarchy, repeated-part consistency, accents, logos, and defects.
- Judge material identity at comparable visible orientations. Sparse reference
  views may use different lighting, exposure, environment, and camera angles
  from the turntable. Physically based metals and glossy dielectrics naturally
  show dark reflected bands, bright highlights, and orientation-dependent value
  changes; those effects alone are not material-assignment defects when the
  substance, base color, finish, symmetry, and well-lit comparable views agree.
- Report only issues that can reasonably be improved by choosing a different
  supplied library material. Do not ask a repair turn to replace physically
  correct metal with paint or plastic merely to imitate one reference view's
  illumination. Renderer lighting, black-background reflections, exposure,
  tone mapping, missing geometry, and camera differences are not material
  assignment issues.
- Do not perform structural coverage or file-integrity checks; the wrapper owns those deterministic gates.
- Do not call tools, read files, run shell commands, edit scene state, or write artifacts. Return only the schema-constrained semantic JSON response.

Issue policy:
- Use status `pass` when no prior issue existed and the result is acceptable.
- Use status `fixed` when the packet lists prior active issues and all are visibly resolved.
- Use status `unresolved_issues` when any fixable or unfixable visual issue remains.
- Every `issues_found` entry must contain severity, description, expected_appearance, actual_appearance, status, affected_prim_paths, and evidence_artifacts.
- For every active issue, `affected_prim_paths` must contain one or more exact
  `group_id` values from `current_material_decisions`. Select every current
  material group whose rendered parts exhibit the issue. These group identifiers
  are the surgical handoff to the repair turn; do not leave them empty and do not
  invent mesh paths from pixels.
- `unresolved_issues` contains the exact descriptions of all active issue entries. Keep assessment notes concise but specific.

Review packet SHA-256: `{review_packet_sha256}`

Review packet:
```json
{json.dumps(review_packet, indent=2)}
```
"""


def build_material_refinement_prompt(
    *,
    run_dir: Path,
    usd_path: Path,
    reference_images: list[Path],
    materials_yaml: Path,
    materials_usd: Path,
    iteration: int,
    max_iterations: int,
    issue_summary: dict[str, object],
    history_path: Path,
    artifact_index_path: Path | None = None,
    issue_packet_path: Path | None = None,
    repair_scope_path: Path | None = None,
    reference_files: list[Path] | None = None,
    repair_attempt_ledger: list[dict[str, object]] | None = None,
    previous_child_artifacts: list[dict[str, str]] | None = None,
    optimize: bool = True,
    respect_existing_material_bindings: bool = False,
    additional_instructions: str | None = None,
    decision_only: bool = False,
    repair_decision_context: dict[str, object] | None = None,
) -> str:
    """Build a compact VQA refinement prompt for a follow-up child turn."""

    if repair_scope_path is None:
        repair_scope_path = run_dir / "raw" / f"material_repair_scope_{iteration}.json"

    request = {
        "run_dir": str(run_dir),
        "usd_path": str(usd_path),
        "reference_images": [str(path) for path in reference_images],
        "reference_files": [str(path) for path in reference_files or []],
        "materials_usd": str(materials_usd),
        "scene_backend": "usd-cli",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "respect_existing_material_bindings": respect_existing_material_bindings,
        "artifact_index": str(artifact_index_path) if artifact_index_path else None,
        "issue_packet": str(issue_packet_path) if issue_packet_path else None,
        "repair_scope": str(repair_scope_path),
    }
    if optimize:
        request["scene_optimization"] = {
            "enabled": True,
            "manifest_path": str(run_dir / "raw" / "scene_optimizer" / "result.json"),
            "correspondence_path": str(
                run_dir / "raw" / "scene_optimizer" / "correspondence.json"
            ),
            "authoring_asset_path": str(usd_path),
            "output_usd_path": str(usd_cli_output_usd_path(run_dir)),
        }
    extra = additional_instructions.strip() if additional_instructions else ""
    extra_block = f"\nAdditional user instructions:\n{extra}\n" if extra else ""
    attempt_ledger = repair_attempt_ledger or []
    compact_attempts = attempt_ledger[-3:]
    issue_summary_compact = {
        "status": issue_summary.get("status"),
        "issue_signature": issue_summary.get("signature"),
        "active_issues": issue_summary.get("active_issues") or [],
        "issues_fixed": issue_summary.get("vqa_issues_fixed") or [],
        "coverage": issue_summary.get("coverage") or {},
        "current_material_decisions": issue_summary.get("current_material_decisions")
        or [],
        "assessment_notes": issue_summary.get("assessment_notes") or "",
    }
    if decision_only and repair_decision_context is not None:
        verified_rule = (
            "\n- Because the current canonical VQA gate is already satisfied, "
            'use `status: "verified"` with no material changes when the '
            "independent render review confirms that no repair is needed."
            if repair_decision_context.get("verified_no_change_allowed") is True
            else ""
        )
        affected_material_paths = {
            str(row["material_path"])
            for row in repair_decision_context.get("current_assignments") or []
            if isinstance(row, dict) and row.get("material_path")
        }
        issue_summary_compact["current_material_decisions"] = [
            {key: value for key, value in row.items() if key != "material_path"}
            for row in issue_summary_compact["current_material_decisions"]
            if isinstance(row, dict)
            and (
                not affected_material_paths
                or str(row.get("material_path") or "") in affected_material_paths
            )
        ]
        prompt_repair_context = {
            **repair_decision_context,
            "current_assignments": [
                {key: value for key, value in row.items() if key != "material_path"}
                for row in repair_decision_context.get("current_assignments") or []
                if isinstance(row, dict)
            ],
            "material_palette": [
                {key: value for key, value in row.items() if key != "material_path"}
                for row in repair_decision_context.get("material_palette") or []
                if isinstance(row, dict)
            ],
        }
        return f"""You are the semantic material-repair stage in a bounded VQA loop.

This is refinement iteration {iteration} of {max_iterations}. Inspect the attached
current renders and user reference images directly. Return only the required
schema-constrained repair response. Do not call tools, read files, write artifacts,
or run shell/scene/render commands. The wrapper preserves loop context below and
owns exact-path expansion, surgical-scope validation, full patch reconstruction,
scene execution, save/reopen audits, rendering, and the next independent VQA turn.

Active VQA state:
```json
{json.dumps(issue_summary_compact, indent=2)}
```

Authoritative repair context:
```json
{json.dumps(prompt_repair_context, indent=2)}
```

Recent repair attempts:
```json
{json.dumps(compact_attempts, indent=2)}
```
{extra_block}
Repair rules:
- Fix only the active issues. Keep every unaffected candidate unchanged and do
  not redo the full material plan.
- `candidate_catalog` is the exact surgically allowed candidate-ID universe,
  `current_assignments` is the currently rendered material grouping, and
  `material_palette` is the complete allowed replacement palette.
- For `status: "applied"`, return one or more `material_changes`. Each change
  uses one exact palette `material_name`, a unique non-empty `candidate_ids`
  array, and an evidence-grounded rationale. Every selected candidate must
  actually change material. Choose the smallest current group or candidate
  subset that fixes the mismatch and preserve repeated-part symmetry.
- Do not repeat a material change that a recent VQA attempt already rejected.
- For `status: "unfixable"`, return no material changes and explain the proven
  palette or geometry limitation in `decision_notes`.{verified_rule}
- Return schema version `content-agents.material-repair-intent.v1`. The wrapper
  expands candidate IDs to exact source/runtime paths and rejects any broadened,
  duplicate, unchanged, or out-of-universe edit before mutating the scene.
"""
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
            f"- `{run_dir / 'raw' / 'material_application_receipt.json'}` for "
            "the exact saved-USD and final-render identities under review.",
            f"- `{history_path}` only to avoid repeating a failed prior repair.",
        ]
    )
    allowed_artifact_reads_block = "\n".join(allowed_artifact_reads)

    optimizer_refinement_fact = (
        f"""
- This is a source-authoring turn after optimized inspection. Preserve both
  `runtime_prim_paths` and their uniquely translated
  `source_prim_paths`/`prim_paths` in every group. If retargeting is required,
  use `{run_dir / "raw" / "scene_optimizer" / "correspondence.json"}` and stop
  if the mapping is absent or ambiguous. Apply the edit to the live composed
  consumer named by `runtime_prim_paths`; the translated source paths are
  immutable provenance for deterministic durable restore, not replacements for
  the live edit target. Never reopen or edit the optimized inspection artifact."""
        if optimize
        else ""
    )
    material_apply_path_key = "runtime_prim_paths" if optimize else "prim_paths"
    if decision_only:
        tool_facts_block = """Execution ownership:
- This is a semantic repair-decision turn. Do not call `usd-cli`, `usd-cli-tel`,
  any scene tool, `--help`, or a render command. Do not read skill docs,
  repository source/tests, broad logs, or general scene snapshots.
- Update only the exact decision patch and repair-scope artifact. The wrapper
  reopens the staged source, clears appearance when required, validates and
  batch-applies the complete updated patch, saves/reopens/audits the output, and
  regenerates the complete verification/turntable render set before launching a
  fresh independent VQA turn.
- Current final renders, the active issue packet, the current decision patch,
  recent repair attempts, and the material palette are the complete handoff.
  They preserve the needed loop context without carrying prior model turns."""
    else:
        tool_facts_block = f"""usd-cli facts:
- Drive the scene through the package-owned `usd-cli-tel` executable (same command arguments as `usd-cli`; skill `usd-cli`) from the same working directory as the initial turn so the existing stage, checkpoints, history, and trace context are reused; do not restart or re-plan.
- Under a fresh checkpoint, use `material --library {materials_usd}` for one prim or update the patch and run `material-apply raw/material_decision_patch.json --library {materials_usd} --path-key {material_apply_path_key}` once for multiple prims. The absolute library path is mandatory; `inputs/material_library/...` is invalid. Undo rejected previews.
- After an accepted change, `usd-cli-tel --json save {run_dir / "output" / "materialized.usda"} --flatten`; reopen and audit it before rendering. `--flatten` is mandatory on every repair save; plain save is invalid.
- Regenerate the complete required verification and 24-frame turntable (4+24 frames) under `final_renders/`. For each batch, use a separate usd-cli-tel render command: `usd-cli-tel --json render ... --detach`, then `usd-cli-tel --json wait <job>`. Numeric collision suffixes are expected; the wrapper selects the newest complete orbit generation from the captured response. Do not review pixels; the wrapper launches a fresh review turn.{optimizer_refinement_fact}"""
    canonical_ownership_step = (
        "6. Before editing, write the declared repair scope to "
        f"`{repair_scope_path}` using schema "
        "`content-agents.material-repair-scope.v1`, the exact active "
        "`issue_signature`, status `applied` or `unfixable`, absolute "
        "source-space `target_prim_paths`, and a nonempty rationale. For an "
        "applied repair, change only those declared paths. For `unfixable`, "
        "`target_prim_paths` must be empty and no material decision may "
        "change; put the affected-but-uneditable prims in the evidence event "
        "and rationale instead. Do not write "
        "`raw/material_post_apply_review.json`; the wrapper owns the fresh "
        "independent post-repair review and all canonical projections."
    )

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

Do not read skill docs, repository source, tests, README files, full child logs, broad raw scene snapshots, or previous run summaries. The tool facts needed for this turn are below.

{tool_facts_block}

Execution safety: never invoke Python or PyPy, run inline code with `-c` or a
here-document, create a scratch `.py` file, or import repository internals from
the shell. {JSON_SYNTAX_CHECK_GUIDANCE} The wrapper owns deterministic validation.

{WINDOWS_CONTROLLED_MATERIAL_PATCH_REPLACE}

Compact procedure:
1. Use the attached reference and current final-render images plus the active issue packet.
2. If an issue is already proven material-library or prim-granularity limited, write a concise unfixable trace event and leave it unresolved. In clean-slate mode, keep `reviewed_no_override` empty and leave an already valid assignment group unchanged; record the limitation only in VQA and trace artifacts.
3. For a fixable issue, identify the smallest existing material group or exact prims. Prefer the inline active issue packet, then `issue_packet`/`artifact_index`; read only targeted JSON slices if exact prim paths are missing.
   - For `rejected_assignment` issues, treat the rejected paths as mandatory repair targets. Read the rejection reason and only the targeted rejected group if needed, then split overly broad/mixed groups, correct invalid material names or paths, or retarget invalid candidate paths before finalizing. Do not leave rejected paths uncovered unless you can show a material-library or geometry-addressability limitation.
4. Check only the small relevant material candidates. Do not browse the full palette unless the candidate is not already named.
5. {"Update only `raw/material_decision_patch.json` for the declared target paths; the wrapper performs all scene execution and rendering after you exit." if decision_only else "Apply at most the needed material assignment(s), update `raw/material_decision_patch.json`, re-save/reopen the output, and regenerate the complete final render set. Do not review the new pixels in this repair turn."}
{canonical_ownership_step}

Output requirements:
- List changed groups, affected prims/renders, VQA status, and limitations.
- Keep trace and JSON edits concise and observable.
"""


def build_physics_tuning_session_prompt(
    *,
    repo_root: Path,
    run_dir: Path,
    physics_usd: Path,
    broker_url: str,
    sweep_client_path: str,
    contract_path: Path,
    max_iterations: int,
    max_trials_per_sweep: int,
    sweep_deadline_seconds: float,
    engine: str,
    optimizer: str,
    scenario_path: Path | None = None,
    behavior_prompt: str | None = None,
    revalidation_max_penetration_m: float | None = None,
    revalidation_duration_s: float | None = None,
    revalidation_dt: float | None = None,
    revalidation_sample_fps: float | None = None,
    revalidation_drop_height_m: float | None = None,
    collision_approximation: str | None = None,
    protected_parameters: list[str] | None = None,
    allow_revise_patch: bool = True,
    candidate_suffix: str = ".usd",
) -> str:
    """Build the agent-owned physics tuning loop session prompt.

    The coding agent owns the outer loop: it authors/revises scenarios,
    requests budgeted judge-free sweeps from the wrapper-owned broker,
    inspects top-K candidate evidence, and decides accept / revise / stop.
    The wrapper enforces budgets, verifies the digest-bound decision chain,
    provisionally promotes the accepted candidate, and requires a separate
    assessment-only final visual review before publication succeeds.
    """

    protected = sorted(set(protected_parameters or []))
    promotion_gate: dict[str, Any] = {
        "engine": engine,
        "duration_s": revalidation_duration_s,
        "dt": revalidation_dt,
        "sample_fps": revalidation_sample_fps,
        "drop_height_m": revalidation_drop_height_m,
    }
    # Emit the penetration limit only when the run overrides it. The prompt
    # says to mirror the promotion_gate block exactly, and an explicit
    # `"max_ground_penetration_m": null` disables the numeric gate at
    # workflow runtime evaluator instead of selecting the runtime default — the
    # opposite of what a default run's gate enforces.
    if revalidation_max_penetration_m is not None:
        promotion_gate["max_ground_penetration_m"] = revalidation_max_penetration_m
    task = {
        "schema_version": "content-agents.physics-tuning-session-task.v1",
        "workflow": "physics.apply.agentic_tuning",
        "required_skills": ["usd-cli", "content-workflow-physics"],
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "finalized_physics_usd": str(physics_usd),
        "candidate_export_suffix": candidate_suffix,
        "scene_tool": {"name": "usd-cli", "renderer": "ovrtx"},
        "sweep_broker": {
            "endpoint": broker_url,
            "client": sweep_client_path,
            "engine": engine,
            "optimizer": optimizer,
            "budget": {
                "max_sweeps": max_iterations,
                "max_trials_per_sweep": max_trials_per_sweep,
                "sweep_deadline_seconds": sweep_deadline_seconds,
            },
        },
        "physics_agentic_contract_path": str(contract_path),
        "behavior_prompt": behavior_prompt,
        "initial_scenario_path": str(scenario_path) if scenario_path else None,
        "scenario_examples_dir": str(
            repo_root / "apps" / "physics_agent" / "configs" / "tuning"
        ),
        "decision_artifact_template": str(
            run_dir / "raw" / "physics_tuning_decision_<iteration>.json"
        ),
        "result_artifact_path": str(run_dir / "raw" / "physics_tuning_result.json"),
        "promotion_gate": promotion_gate,
        "authored_collision_approximation": collision_approximation,
        "protected_parameters": protected,
        "allow_revise_patch": allow_revise_patch,
    }

    # `revise_patch` may rewrite any field of the decision patch, and nothing
    # told the agent the collision approximation was not one of them. The loop's
    # proxy metrics (settle by the window, no backtracking, zero final angular
    # velocity) are all maximised by a collider that cannot rotate, so degrading
    # the authored shape is the degenerate optimum. Seen on a RoboCasa apple:
    # convexHull -> boundingSphere -> boundingCube, which passed every gate and
    # promoted physics that slides and stops dead instead of rolling.
    collider_note = ""
    if collision_approximation:
        collider_note = (
            "\n     PRESERVE THE COLLISION SHAPE: the decision patch authored "
            f"collision approximation(s) `{collision_approximation}`. A revised patch "
            "retunes material values (density/mass, friction, restitution); it "
            "must NOT degrade the collision approximation to a coarser bounding "
            "primitive. Replacing a hull with a box or sphere trivially satisfies "
            "the settle and no-backtracking metrics by removing the body's "
            "ability to roll, which is metric gaming, not a physics fix. If you "
            "genuinely believe the collision shape itself is wrong, say so in the "
            "rationale and justify it against the behavior goal — never swap it in "
            "silently to make a metric pass."
        )

    protected_parameter_note = ""
    if protected:
        protected_parameter_note = (
            " Protected parameters are wrapper-enforced: do not include "
            + ", ".join(f"`{name}`" for name in protected)
            + " in any scenario. The broker rejects a sweep before reservation "
            "when one is present."
        )

    if allow_revise_patch:
        revise_patch_instruction = f"""   - revise_patch: the physics authoring itself is wrong (not just parameter
     values). Write a revised decision patch, apply it through usd-cli
     `apply-schema` to a CLEAN derivative of the ORIGINAL authored USD (never
     a previously tuned USD — no mutation accumulation), record the rebuilt
     USD path plus its `rebuilt_physics_usd_sha256`, and use that as the
     next sweep's input (citing this decision via `--rebuilt-decision`).{collider_note}"""
    else:
        revise_patch_instruction = """   - revise_patch: PROHIBITED for this run. The finalized USD carries protected
     VoMP mass, center-of-mass, inertia, and principal-axis evidence; applying a
     decision patch directly could replace that attested contract. Use
     `revise_scenario`, `accept`, or `stop`. The wrapper rejects rebuilt sweep
     inputs and any decision chain containing `revise_patch`."""

    # The wrapper revalidates an accepted candidate on its own terms: its own
    # acceptance override AND its own drop-settle setup. The agent's evidence
    # calls otherwise default to the runtime penetration default and to
    # whatever its scenario happens to
    # use, so the two sides measure different experiments and disagree. Seen on a
    # RoboCasa apple: the agent measured 0.0283 m from a ~0.2 m scenario drop and
    # accepted, while the wrapper measured 0.0588 m from its configured 0.9 m drop
    # and refused to promote.
    gate_terms = [
        f"`max_ground_penetration_m={revalidation_max_penetration_m}`"
        if revalidation_max_penetration_m is not None
        else "",
        f"`drop_height_m={revalidation_drop_height_m}`"
        if revalidation_drop_height_m is not None
        else "",
        f"`duration_s={revalidation_duration_s}`"
        if revalidation_duration_s is not None
        else "",
        f"`sample_fps={revalidation_sample_fps}`"
        if revalidation_sample_fps is not None
        else "",
        f"`dt={revalidation_dt}`" if revalidation_dt is not None else "",
    ]
    gate_terms = [term for term in gate_terms if term]
    # `duration_s`/`sample_fps`/`dt` always carry a runner default, so gate_terms
    # is never empty on a real run. The "not the default" claims below are only
    # true when the run actually overrides them, so state each one conditionally:
    # asserting a non-default limit on a default run is the same agent/gate
    # divergence this note exists to remove, inverted.
    if revalidation_max_penetration_m is not None:
        penetration_clause = (
            "Its penetration limit is "
            f"`{revalidation_max_penetration_m}` m, NOT the runtime default."
        )
    else:
        penetration_clause = (
            "Its penetration limit is the runtime default — scale-relative, "
            "exactly `min(max(0.005, min(0.025 * bbox_diagonal_m, "
            "0.5 * smallest_bbox_extent_m)), 1.0)`, when penetration is "
            "measured from exact collider mesh vertices, the absolute "
            "0.005 m when it falls back to conservative bbox corners. This "
            "run configures no override, so do not invent one: run workflow "
            "runtime validation without `max_ground_penetration_m` and the "
            "same default applies to your evidence and to the gate "
            "(an explicit `max_ground_penetration_m: null` disables the "
            "numeric gate — that is not the default)."
        )
    drop_clause = (
        " Its drop height is NOT whatever your scenario uses."
        if revalidation_drop_height_m is not None
        else ""
    )
    revalidation_note = ""
    if gate_terms:
        revalidation_note = (
            "\n   PROMOTION GATE — MIRROR THESE EXACTLY: the wrapper revalidates "
            "the candidate you accept with " + ", ".join(gate_terms) + " (see "
            "`promotion_gate` in the task block). "
            + penetration_clause
            + drop_clause
            + " Use these same values when the workflow authors the scenario, calls "
            "usd-cli `physics simulate`, and evaluates the returned facts, or your "
            "evidence measures a different experiment "
            "than the gate: a shorter drop lands softer and under-reports "
            "penetration, so a candidate you accept can still fail promotion."
        )

    return f"""You are running the agent-owned physics behavior tuning loop for
a finalized physics USD. YOU own the outer loop: judge evidence, revise the
scenario or physics decision patch, and decide when to stop. The wrapper
enforces the sweep and publication contract; after your acceptance it
independently revalidates and visually reviews the exact promoted rollout.

Load and follow these skills:
- `usd-cli`
- `content-workflow-physics`

Structured task:
```json
{json.dumps(task, indent=2)}
```

{WINDOWS_CONTROLLED_JSON_ARTIFACT_WRITE}

Tuning loop (repeat up to {max_iterations} sweeps; the broker refuses more):
1. Scenario. Use the initial scenario when provided; otherwise author
   `{run_dir}/tuning/iter_<i>/scenario.yaml` yourself from the behavior goal in
   the contract. Copy the structure of the examples in the scenario examples
   directory (`name`, `metric`, `target`, `parameters` with bounds). Choose
   tunable parameters and bounds that serve the behavior goal. Do not set
   `target.vlm_check` (the broker forces it off) and do not rely on any
   VLM/LLM inside the sweep — the sweep is pure optimization.{protected_parameter_note}
2. Sweep. Request exactly one budgeted sweep per iteration:
   `{sweep_client_path} --broker-url {broker_url} \\
      --sweep-deadline-seconds {sweep_deadline_seconds} run \\
      --scenario <scenario.yaml> --physics-usd <sweep input USD> \\
      --output-dir {run_dir}/tuning/iter_<i>`
   Invoke that absolute client path directly. Do not wrap it in `uv run`,
   `pip install`, or any package manager — package tooling writes caches and
   symlinks into the run directory, which fails the wrapper's artifact
   safety gate.
   The tool blocks until the sweep finishes and prints the sweep record with
   `sweep_id`, digests, `evidence_path`, and the inline `evidence` object.
   `--output-dir` is a logical iteration path only: all unsandboxed broker and
   engine writes stay in a broker-private workspace. Exit code 3 means the
   budget or phase deadline refused the reservation — write your result
   artifact and stop. Never invoke `run_tune` or `physics-agent` directly:
   sweeps without a broker record are rejected by wrapper verification.
   The engine is broker-enforced ({engine}); do not pass `--engine`.
   The optimizer is broker-configured as `{optimizer}`; omit `--optimizer`
   from sweep calls to use it, or pass `--optimizer <name>` to override
   for one sweep. The sweep input must be the finalized physics USD — after a revise_patch
   rebuild, also pass `--rebuilt-decision <that revise_patch decision file>`
   so the broker can verify its complete canonical decision chain and
   digest-verify the rebuilt input; any other USD is rejected.
3. Inspect evidence. Use the sweep response's inline `evidence` object:
   top-K candidates with params, scores, recordings, and metrics. The
   `evidence_path` is the broker-private digest-bound copy for the decision
   record; do not try to rewrite it. The proxy score ranks candidates, but YOU
   judge semantics. Export every top candidate needed for that judgement into
   the confined run before accepting it:
   `{sweep_client_path} --broker-url {broker_url} materialize \\
      --sweep-id <id> --trial-index <n> \\
      --output-usd {run_dir}/tuning/iter_<i>/review/trial_<n>{candidate_suffix} \\
      --output-recording \\
        {run_dir}/tuning/iter_<i>/review/trial_<n>_recording<recording-suffix>`
   Read `<recording-suffix>` from that candidate's `recording` path in the
   inline evidence (`.usd` or `.usda`). The client rejects a destination that
   does not preserve the broker artifact suffix.
   The response keeps the broker-private `usd_path` used by the decision chain
   and adds `exported_usd_path` plus `exported_recording_path` for review.
   Render that exported recording directly through OVRTX: it is the exact
   scenario rollout that produced the candidate's score, including scenario
   conditions such as initial velocity. Inspect those frames before accepting
   any goal that depends on visible motion, tipping,
   sliding, or final rest. You may prefer a lower-ranked candidate whose
   behavior better matches the goal.{revalidation_note}
4. Decide. Write `{run_dir}/raw/physics_tuning_decision_<i>.json`:
   {{"schema_version": "content-agents.physics-tuning-decision.v1",
     "iteration": <i>, "decision": "accept" | "revise_scenario" |
     "revise_patch" | "stop", "sweep_id": ..., "scenario_path": ...,
     "scenario_sha256": ..., "evidence_path": ..., "evidence_sha256": ...,
     "selected": {{"sweep_id": ..., "trial_index": ..., "usd_path": ...,
     "usd_sha256": ...}} (accept only), "next_scenario_path": ...
     (revise_scenario only), "revised_patch_path": ... and
     "rebuilt_physics_usd": ... (revise_patch only),
     "prior_decision_sha256": <sha256 of the previous decision FILE, null for
     iteration 1>, "rationale": "..."}}
   Digest-bind every reference: copy `scenario_sha256` / `evidence_sha256`
   from the sweep record; compute file SHA-256 with `sha256sum`. Digests are
   MANDATORY, not advisory — accept/revise decisions without
   `scenario_sha256` (and accept without `evidence_sha256` +
   `selected.usd_sha256`, revise_patch without
   `rebuilt_physics_usd_sha256`) are rejected, and every digest is compared
   against the broker ledger with the referenced files rehashed at
   conclusion.
   A `stop` decision may cite the last sweep it judged, but when it does it
   must copy every scenario/evidence digest present in that sweep record. A
   sweep-less `stop` claims no sweep and leaves all sweep artifact fields null.
   Every nonterminal `revise_scenario` or `revise_patch` decision must cite a
   newly completed sweep. Only a terminal `accept` or `stop` may reuse the
   immediately preceding sweep id.
   - accept: materialize the chosen candidate (reuse the response from the
     review export when it is already materialized):
     `{sweep_client_path} --broker-url {broker_url} \\
        materialize --sweep-id <id> --trial-index <n>`
     and copy its `usd_path`/`usd_sha256` into `selected`.
   - revise_scenario: author the next scenario file yourself (adjust bounds,
     metric, or target setup based on the evidence) and reference it.
{revise_patch_instruction}
   - stop: the goal is unreachable within budget; explain why.
5. Result. After accept/stop (or a refused reservation), write
   `{run_dir}/raw/physics_tuning_result.json`:
   {{"schema_version": "content-agents.physics-tuning-result.v1",
     "status": "accepted" | "stopped" | "budget_exhausted" | "tool_failure",
     "selected": <same object as the accepting decision, accepted only>,
     "decision_paths": [<decision files in order>],
     "final_decision_sha256": <sha256 of the last decision file>,
     "rationale": "..."}}
   Only "accepted" can promote a candidate; never claim it unless the
   decision chain ENDS with the accept decision, and `selected` must match
   that final accept decision's candidate exactly (same sweep_id, trial_index,
   usd_path, and usd_sha256) — any divergence is a tool failure. The wrapper
   revalidates your selected USD independently, replays the exact promoted
   bytes under the accepted scenario, renders that replay through OVRTX, and
   runs a final goal-level behavior review. Any digest mismatch rejects
   promotion. A failed final visual assessment rejects and rolls back promotion.

Do not edit canonical run artifacts (`physics_assignments.json`,
`final_summary.md`, the output USD) — the wrapper promotes after verification.

Finish with a short response: iterations used, final status, selected
candidate (if any), and remaining limitations.
"""


def build_physics_external_refine_prompt(
    *,
    run_dir: Path,
    broker_url: str,
    sweep_client_path: str,
    contract_path: Path,
    user_prompt: str,
    task_name: str,
    objective: dict[str, Any],
    parameter_catalog: list[dict[str, Any]],
    initial_active_search: dict[str, dict[str, float]],
    nominal_params: dict[str, float],
    max_iterations: int,
    max_trials_per_sweep: int,
    sweep_deadline_seconds: float,
    reference_images: list[Path] | None = None,
    additional_instructions: str | None = None,
) -> str:
    """Build the agent-owned external-runtime (BYOR) refinement session prompt.

    The coding agent owns the outer loop that ``physics-agent
    refine-external`` runs with a VLM judge and an LLM refiner: it reviews
    each sweep's rendered winner frames against the user's behavior goal,
    revises the active parameter search, and decides accept / revise / stop.
    The wrapper owns qualification approval, enforces the sweep budget, and
    verifies the digest-bound decision chain against the broker ledger.
    """

    task = {
        "schema_version": "content-agents.physics-external-tuning-session-task.v1",
        "workflow": "physics.refine_external.agentic",
        "required_skills": ["content-workflow-physics-external-tuning"],
        "run_dir": str(run_dir),
        "external_task": task_name,
        "behavior_goal": user_prompt,
        "objective": objective,
        "parameter_catalog": parameter_catalog,
        "initial_active_search": initial_active_search,
        "nominal_params": nominal_params,
        "sweep_broker": {
            "endpoint": broker_url,
            "client": sweep_client_path,
            "budget": {
                "max_sweeps": max_iterations,
                "max_trials_per_sweep": max_trials_per_sweep,
                "sweep_deadline_seconds": sweep_deadline_seconds,
            },
        },
        "external_contract_path": str(contract_path),
        "reference_images": [str(path) for path in reference_images or []],
        "decision_artifact_template": str(
            run_dir / "raw" / "physics_external_tuning_decision_<iteration>.json"
        ),
        "result_artifact_path": str(
            run_dir / "raw" / "physics_external_tuning_result.json"
        ),
    }
    extra_note = ""
    if additional_instructions:
        extra_note = f"\nAdditional operator instructions:\n{additional_instructions}\n"
    reference_clause = (
        ", comparing against the provided reference images" if reference_images else ""
    )

    return f"""You are running the agent-owned external-runtime (BYOR) physics
refinement loop for a qualified customer simulation runtime. YOU own the
outer loop: visually judge each sweep's rendered rollout evidence against the
behavior goal, revise the active parameter search, and decide when to stop.
The wrapper owns qualification approval, enforces the sweep budget, and
independently verifies every digest you cite. There is no VLM judge and no
LLM refiner on this path — your own review replaces both.

Load and follow this skill:
- `content-workflow-physics-external-tuning`

Structured task:
```json
{json.dumps(task, indent=2)}
```

{WINDOWS_CONTROLLED_JSON_ARTIFACT_WRITE}

Refinement loop (repeat up to {max_iterations} sweeps; the broker refuses
more, and sweeps are sequential — never request one before the previous
finished):
1. Search. For the first sweep, the runtime config's declared search
   (`initial_active_search`) is usually right — omit `--active-search` to use
   it. For later sweeps, choose the next active parameter subset and bounds
   from the evidence: narrow bounds around a promising region, widen a rail
   (best value pinned at a bound), or activate a different catalog parameter.
   Only parameters in `parameter_catalog` are allowed; parameters you omit
   are pinned by the broker to the previous sweep's winning values (the
   qualified nominal values before the first sweep). The objective is
   adapter-owned and fixed — you cannot change what is measured, only where
   to search.
2. Sweep. Request exactly one budgeted sweep per iteration:
   `{sweep_client_path} --broker-url {broker_url} \\
      --sweep-deadline-seconds {sweep_deadline_seconds} run \\
      --output-dir {run_dir}/tuning/iter_<i> \\
      [--active-search '<json>'] [--max-trials <n>]`
   Invoke that absolute client path directly. Do not wrap it in `uv run`,
   `pip install`, or any package manager — package tooling writes caches and
   symlinks into the run directory, which fails the wrapper's artifact
   safety gate. The tool blocks until the sweep finishes (external trials
   start a customer simulator process each — expect minutes per trial) and
   prints the sweep record with `sweep_id`, `evidence_path`,
   `evidence_sha256`, `best_params`, `best_objective`, and the published
   frame paths with digests. Exit code 3 means the budget or the phase
   deadline refused the reservation — write your result artifact and stop.
   Exit code 7 means the previous sweep's runtime is still terminating —
   wait roughly a minute and retry the same command; it does not consume
   budget. Never invoke `run_external_tune`, `physics-agent
   tune-external`, or `physics-agent refine-external` directly: sweeps
   without a broker record are rejected by wrapper verification.
3. Review evidence — LOOK AT THE FRAMES. The broker publishes the winning
   trial's rendered rollout frames under
   `{run_dir}/tuning/iter_<i>/evidence-<sweep_id>/frames/` and lists them in
   `evidence.json` (`frames` with per-file SHA-256; `distinct_frames` is the
   deduplicated subset — review those to avoid submitting byte-identical
   images). Read the frames as images and judge whether the motion matches
   the behavior goal{reference_clause}. The recording renders
   the adapter's recorded proxy geometry, not the customer renderer — judge
   motion and configuration, not appearance. Also weigh the numeric
   objective (`best_objective`, direction in `objective`) and the per-trial
   `history`, but the frames are the ground truth for behavior. A sweep can
   minimize the objective while moving wrongly — that is a revise, not an
   accept.
4. Decide. Write `{run_dir}/raw/physics_external_tuning_decision_<i>.json`:
   {{"schema_version": "content-agents.physics-external-tuning-decision.v1",
     "iteration": <i>, "decision": "accept" | "revise_search" | "stop",
     "sweep_id": ..., "evidence_path": ..., "evidence_sha256": ...,
     "reviewed_frames": [{{"path": ..., "sha256": ...}}, ...] (the frames you
     actually reviewed, digests copied from evidence.json; MANDATORY for
     accept), "selected": {{"sweep_id": ..., "best_params": {{...}},
     "evidence_sha256": ..., "recording_sha256": ...}} (accept only, copied
     exactly from the sweep record), "next_active_search": {{"param":
     {{"min": ..., "max": ...}}}} (revise_search only),
     "prior_decision_sha256": <sha256 of the previous decision FILE, null
     for iteration 1>, "rationale": "..."}}
   Digest-bind every reference: copy `evidence_sha256` and frame digests
   from the sweep record; compute file SHA-256 with `sha256sum`. Digests are
   MANDATORY — the wrapper compares them against the broker ledger and
   rehashes the referenced files at conclusion, so a swapped or edited
   artifact fails the whole run.
   - accept: the reviewed behavior matches the goal.
   - revise_search: explain what the evidence showed and why the next
     bounds/subset should fix it.
   - stop: the goal is unreachable within budget or the adapter's objective
     cannot express it; explain why.
5. Result. After accept/stop (or a refused reservation), write
   `{run_dir}/raw/physics_external_tuning_result.json`:
   {{"schema_version": "content-agents.physics-external-tuning-result.v1",
     "status": "accepted" | "stopped" | "budget_exhausted" | "tool_failure",
     "selected": <same object as the accepting decision, accepted only>,
     "decision_paths": [<decision files in order>],
     "final_decision_sha256": <sha256 of the last decision file>,
     "rationale": "..."}}
   Only "accepted" publishes a final result; never claim it unless the
   decision chain ENDS with the accept decision and `selected` matches it
   exactly. The wrapper publishes `final/` (best params, the exact recorded
   rollout, render frames, declared adapter outputs) from broker-verified
   artifacts after your session exits — do not author `final/` yourself.
{extra_note}
Finish with a short response: sweeps used, final status, best parameters and
objective (if any), what you observed in the frames, and remaining
limitations.
"""
