---
name: content-workflow-physics
description: Use with content-workbench when a long-running coding agent needs to infer physics properties, author USD physics schema, run simulation-backed validation, refine failures, and produce canonical physics artifacts and validation evidence.
---

# content-workflow-physics

Use this skill with `content-workbench` when a long-running coding agent needs
to infer physics properties for a USD asset, author physics schema, run
simulation-backed validation, and produce canonical physics artifacts.

This skill owns workflow method. Workbench owns scene state, rendering, picking,
optimization, edit mechanics, restore, export, and runtime validation APIs.

## Workflow

1. Load the Workbench skill.
2. Start or connect to Workbench.
3. Inspect unoptimized component and topology evidence first. When optimizer
   settings are not supplied explicitly, choose them for physics authoring:
   preserve body, joint, articulation, collider, helper, and component roles;
   optimize only when it improves legal inspection or authoring targets.
4. Inspect logical components and topology. Keep visual evidence, collider
   targets, helper geometry, rigid-body roots, and joints as separate roles.
5. Resolve mobility intent before topology repair. Preserve existing topology
   when intent is absent or ambiguous. Apply only an explicit, digest-bound
   topology plan to a derivative asset.
6. Make exactly one physics decision per component. Existing colliders are
   preserved and targeted directly; visible geometry is a collider target only
   when the decision explicitly uses `author_on_targets`. Never target helpers.
7. Infer density, mass, friction, restitution, collision approximation,
   confidence, rationale, and quality warnings for each decision.
8. Write the V2 `raw/physics_decision_patch.json` and, when required by resolved
   mobility intent, `raw/physics_topology_plan.json`.
9. Apply accepted topology and physics decisions through Workbench APIs.
10. Run deterministic schema checks and runtime simulation validation. Treat
    initial pose discontinuity, excessive ground penetration, missing gravity
    response, and expected body-count mismatch as hard failures.
11. Render frames from the runtime `recording.usda` through Workbench's generic
    frame-sequence render API.
12. Write `physics_behavior_assessment.json` after visually reviewing the
    rendered simulation frames and runtime report.
13. Refine physics decisions when runtime or visual validation finds fixable
    issues.
14. Produce canonical artifacts and restore/export accepted edits when required.

## Required Artifacts

- `request.json`
- `raw/physics_decision_patch.json`
- `physics_assignments.json`
- authored USD/USDZ with physics schema when the workflow claims a durable output
- `physics_behavior_assessment.json`
- `validation_evidence.json`
- runtime validation artifacts
- operation counts where available
- final summary
- trace/events where available

## Runtime Validation Refinement

Runtime validation is part of this skill, not a separate workflow skill.
ovphysx/runtime metrics are authoritative for hard failures. Visual behavior
review can make an otherwise passing runtime result conditional, but it must not
override non-finite trajectories, missing rigid bodies, runtime load failures, or
other solver-backed failures.

After runtime validation, render the `recording.usda` with Workbench
`render-frames` and review the frames for:

- parts visibly separating or moving independently when intended as one rigid
  body;
- no visible motion under gravity;
- implausible bounce, sliding, or settling;
- obvious interpenetration or tunneling;
- stale, blank, or misframed validation renders;
- mismatch between runtime metrics and visible behavior.

Write `physics_behavior_assessment.json` with `status`, `checked_views`,
`runtime_report`, `rendered_frames`, `issues_found`, `issues_fixed`,
`unresolved_issues`, and `assessment_notes`.

When simulation or schema validation reports fixable issues:

- preserve the previous decision patch;
- inspect affected prims, properties, body grouping, collider approximation, and
  trajectory metrics;
- apply targeted corrections through Workbench;
- rerun only the needed validation scenarios when possible;
- update physics assignments and validation evidence;
- stop when remaining issues are unfixable with available evidence/runtime
  support or when the configured iteration cap is reached.

## Boundaries

Do not use the fixed `apps/physics_agent` pipeline as this workflow's engine.
Learn from its property vocabulary, authoring rules, simulator contracts, and
validation metrics, but keep the workflow agent-driven through Workbench.

Call Workbench `inspect-components`, `inspect-topology`,
`apply-topology-plan`, `apply-schema`, and `validate-runtime` operations. The
`inspect-mesh-candidates` route is V1 compatibility only and must not drive new
workflow decisions. When the workflow needs visual review of a generated
time-sampled recording, call Workbench's generic frame-sequence render
operation with the recording as `scene_path`. A
solver such as ovphysx may run in an isolated helper process behind Workbench,
but the workflow agent should consume Workbench artifacts and metrics rather
than importing the solver daemon directly.

Move reusable USD physics authoring, simulator execution, trajectory metrics, and
validation evidence helpers into `world_understanding` when they are useful
outside this workflow. Keep workflow-specific policy, schemas, finalizers, and
repair rules in `content_agent_workflows.physics`.

## References

- `references/physics-policy.md`
- `references/output-artifacts.md`
- `references/runtime-validation.md`
