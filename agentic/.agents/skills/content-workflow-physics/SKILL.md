---
name: content-workflow-physics
description: Own agent-driven physics planning, policy, component decisions, guarded repair, authoring, optional attested VoMP mass properties, simulation-backed validation, refinement, acceptance, and durable artifacts for a USD asset, using usd-cli only for low-level scene operations.
metadata:
  author: NVIDIA Omniverse
---

# content-workflow-physics

Use this skill when a long-running coding agent must infer physics properties,
author USD physics schema, validate behavior, refine failures, and produce
canonical physics artifacts.

This skill and `content_agent_workflows.physics` own the workflow. The selected
scene backend is DCC-like: it may inspect authored properties, apply accepted
edits, run simulation, return raw observations, and render evidence. It must not
select workflow policy or issue the final verdict.

## Runtime Preflight

Before starting Workbench or a child-agent turn, prepare the isolated OvPhysX
runtime used by simulation-backed validation:

```bash
content-workflow-cli preflight physics-runtime --repo-root REPO_ROOT
```

Missing reviewed dependencies are installed from the architecture-specific
lock by default, followed by a CPU construction/release probe. Use
`--no-install-missing` for check-only operation; a blocked report includes the
exact install commands. Do not begin physics authoring when this required
runtime gate is blocked.

## Workflow

1. Read the resolved request, source identity, output contract, user guidance,
   runtime target, and validation policy.
2. Load the repo-root `usd-cli` skill for low-level operations.
3. Inspect the workflow-owned usd-cli session's declared inspection
   scene, starting from unoptimized component and topology evidence. When
   optimization is enabled, use inspection path space, retain source-path
   expansions, and use the wrapper's durable inspection copy for every later
   turn. When optimizer settings are not supplied explicitly, choose them for
   physics authoring: preserve body, joint, articulation, collider, helper, and
   component roles; optimize only when it improves legal inspection or
   authoring targets.
4. When `raw/physics_agentic_contract.json` exists, read it before planning; it
   is the durable behavior/tuning/refine contract for the long-running workflow.
   Never invoke Python or PyPy, execute inline code or a scratch `.py` file, or
   import repository internals from the child shell. Use `jq -e` and documented
   checked-in CLI commands; let the wrapper run deterministic validators.
5. Inspect scene hierarchy, logical components, authored physics state, visual
   evidence, collider targets, rigid-body roots, helpers, and joints. Keep those
   roles separate.
6. Resolve whether physics is expected and determine mobility intent in the
   workflow layer, before topology repair. Preserve existing topology when
   intent is absent or ambiguous.
7. Build and validate any digest-bound topology repair plan with the existing
   workflow helpers. Use the retained guarded topology helper for primitives not
   yet available through usd-cli. Apply only an explicit,
   digest-bound topology plan to a derivative asset.
8. Make exactly one accepted physics decision per logical component. Existing
   colliders are preserved and targeted directly; visible geometry is a collider
   target only when the decision explicitly uses `author_on_targets`. Select
   targets using the component's stable `authoring_targets[].target_id` values;
   never retype USD prim paths into the decision patch. Never target helpers.
9. Infer density or mass, friction, restitution, collision approximation,
   confidence, rationale, and quality warnings for each decision according to
   workflow policy.
10. Write the V2 `raw/physics_decision_patch.json` with
    `collider_target_ids` and, when required by resolved mobility intent,
    `raw/physics_topology_plan.json`. The wrapper resolves IDs to inspected
    paths.
11. Do not send the target-ID patch directly to `physics/apply-schema`. The
    wrapper resolves IDs to inspected paths, then applies the accepted topology
    plan and physics decisions through the selected low-level backend/helper to
    a derivative asset after the child turn.
12. Do not author `mass_authoring_path` in a target-ID patch. When
    `agentic_physics.mass_properties.enabled` is true, the wrapper derives the
    mass target from the inspected body root, then runs calibrated OVRTX
    evidence capture, the pinned official VoMP runtime, and MassAPI authoring
    after schema application. The VoMP-authored USD becomes the canonical input
    to runtime validation; do not invoke `physics-agent run-vomp` from the child
    session.
13. Run deterministic schema checks and raw runtime simulation validation.
    Treat solver load failures, non-finite trajectories, missing expected
    bodies, expected body-count mismatch, initial pose discontinuity, excessive
    ground penetration, and required gravity-response failures as hard failures
    under workflow policy.
14. Render frames from the runtime `recording.usda` through OVRTX using the
    usd-cli frame-sequence render route.
15. Write `physics_behavior_assessment.json` after visually reviewing the
    rendered simulation frames and runtime report.
16. When `raw/physics_agentic_contract.json` requests `tune` or `refine`, run
    the agent-owned tuning loop described in "Agent-Owned Tuning Loop" below:
    the agent judges evidence, revises scenarios or the decision patch, and
    decides when to stop; the wrapper only enforces budgets and promotes an
    accepted candidate after independent revalidation.
    Tuning and refinement remain workflow capabilities and reuse the same
    workflow-owned usd-cli session for scene operations.
17. Refine targeted fixable decisions when runtime, visual validation, or
    tuning evidence finds fixable issues, preserving prior patches, simulation
    reports, and renders.
18. Reopen the saved derivative, validate authored state and the full artifact
    contract, produce canonical artifacts and restore/export accepted edits
    when required, then issue the workflow acceptance/failure classification.

For launcher-managed execution, use the existing entry point:

```bash
content-workflow-cli physics apply \
  --usd INPUT.usd \
  --output-dir RUN
```

The launcher performs a package-owned `usd-cli-tel`/OVRTX preflight before
child launch, then uses `physics apply`, `physics validate`, and `save` for the
accepted low-level patch. Workflow tuning/refinement policy stays outside
usd-cli.

## VoMP Mass Properties

VoMP is opt-in through `--vomp-root`. The checkout revision and model artifact
hashes are pinned by the wrapper. The selected prim must be one enabled,
non-instance-backed rigid body with static mesh geometry and authored USD length
and mass units; the deterministic adapter fails closed when these or calibrated
render/inference requirements are not met.

The schema-authored layer is retained as `raw/physics_pre_vomp.<ext>`. Canonical
`physics.<ext>`, runtime validation, assignments, and final summaries all refer
to the VoMP-authored derivative. Every visual-refinement iteration that changes
the decision patch reruns this phase before validation.

During agentic tuning, `mass_scale` is wrapper-protected and `revise_patch` is
disabled because either can replace the attested mass contract. Scenarios may
still tune friction, restitution, or backend contact parameters. Before an
accepted candidate can be promoted, the wrapper independently verifies its
mass, center of mass, diagonal inertia, principal axes, target, and provenance
against the immutable VoMP result named by the latest finalize record. After
promotion, `raw/physics_vomp_result.json` is republished with the promoted USD
path and digest.

## Agent-Owned Tuning Loop

The tuning outer loop is part of this skill, not a wrapper feature. Each
iteration:

- Author or revise a tuning `scenario.yaml` (structure: `name`, `metric`,
  `target`, `parameters` with bounds — see
  `apps/physics_agent/configs/tuning/` for examples). Scenario authoring and
  revision are agent decisions; there is no interpreter or refiner model.
  When the durable contract lists protected parameters, omit them; the broker
  rejects them before reserving sweep budget.
- Request exactly one budgeted sweep through the wrapper-owned broker client
  `content-workflow-physics-tune-sweep` (broker URL comes from the task).
  The broker atomically reserves budget before the sweep, enforces the tune
  engine (never pass `--engine` overrides), forces the engine judge and
  `target.vlm_check` off, and caps the sweep deadline at the remaining phase
  time. The sweep input must be the finalized physics USD, or a
  `revise_patch` rebuild cited via `--rebuilt-decision`, whose complete
  canonical decision chain is broker-verified and digest-binds that USD via
  `rebuilt_physics_usd_sha256`. A refused reservation
  (exit 3) means the budget or phase deadline is spent — write the result
  artifact and stop. Never invoke `run_tune`, `physics-agent tune`, or
  `physics-agent refine` directly: the budget is protocol-enforced, and
  sweeps without a broker record fail wrapper verification and can never be
  promoted.
- Inspect the evidence packet: top-K candidates with parameters, proxy
  scores, recordings, and metrics. The proxy score ranks candidates; the
  agent judges semantics. Render
  candidate recordings through usd-cli, author the candidate scenario in the
  workflow, run usd-cli `physics simulate`, and apply the workflow-owned
  acceptance criteria to the raw facts. You may select a lower-ranked candidate.
- Write a digest-bound `raw/physics_tuning_decision_<i>.json`
  (`accept` | `revise_scenario` | `revise_patch` | `stop`), carrying the
  sweep id, scenario/evidence SHA-256 digests from the sweep record, the
  selected candidate's USD digest (accept), and the SHA-256 of the previous
  decision file (`prior_decision_sha256`, null for iteration 1). The digests
  are mandatory: decisions missing them are rejected, every digest is
  compared against the broker ledger, and the referenced artifacts are
  rehashed at conclusion. Each nonterminal revision must cite a newly
  completed sweep; only a terminal `accept` or `stop` may reuse the
  immediately preceding sweep id.
- `revise_patch` means the authored physics itself is wrong: write a revised
  decision patch, apply it through usd-cli `apply-schema` to a CLEAN
  derivative of the ORIGINAL authored USD (never a previously tuned USD),
  record the rebuilt USD's path and `rebuilt_physics_usd_sha256` in the
  decision, and sweep that rebuilt USD next (citing the decision via
  `--rebuilt-decision`). This action is unavailable when the durable contract
  sets `allow_revise_patch` to false, including VoMP mass-authoring runs.
- Finish with `raw/physics_tuning_result.json` using honest terminal
  semantics: `accepted` (only with a chain ending in accept), `stopped`,
  `budget_exhausted`, or `tool_failure`. Only `accepted` can promote, and
  `selected` must exactly match the final accept decision's candidate
  (sweep id, trial index, USD path, USD digest); the wrapper independently
  revalidates the exact selected USD and rejects any divergence.

## Workflow-Owned Policy

Keep all of the following outside usd-cli and in the Physics workflow:

- whether physics is expected;
- acceptable ranges and defaults;
- logical component grouping;
- mobility intent and authoring policy;
- topology-repair planning and guarded-repair authorization;
- `physics_sanity` interpretation;
- expected body counts and required validation scenarios;
- behavior assessment, workflow failure classification, and final acceptance.

Low-level authored-state checks, simulation trajectories, metrics, command
outcomes, and OVRTX renders are observations consumed by this policy.

## Required Artifacts

### Controlled JSON artifact writes

For generated JSON, do not probe `apply_patch`: it is unavailable in controlled
child sessions. Construct the complete document with `jq -nS`, pass dynamic
values through `--arg`/`--argjson`, write a same-directory `.tmp`, then `mv` it
to the canonical artifact path and verify it with `jq -e .`. The wrapper
continues to validate schema and reproducibility after the child exits.

Every invocation path preserves:
- `request.json`;
- `raw/physics_decision_patch.json`;
- optional `raw/physics_topology_plan.json`;
- `physics_assignments.json`;
- authored derivative USD/USDZ when durable output is claimed;
- `physics_behavior_assessment.json`;
- `validation_evidence.json`;
- runtime validation reports, trajectories, recordings, and reviewed OVRTX
  frames;
- when VoMP is enabled: `raw/physics_pre_vomp.<ext>`,
  `raw/physics_vomp_result.json`, and `raw/physics_vomp_mass_properties.json`;
- for each long-running VoMP finalization: immutable input/output, result, and
  provenance under `raw/vomp/finalize-<i>/`, plus render/inference evidence
  under `vomp/finalize-<i>/`;
- when tuning: `tuning/iter_<i>/scenario.yaml`, `tuning/iter_<i>/evidence.json`,
  `raw/physics_tuning_decision_<i>.json` (digest chain), and
  `raw/physics_tuning_result.json`;
- operation counts;
- request, decisions, operation evidence, validation, renders, final output,
  failure state, and resumable trace artifacts;
- final summary.

Backend history/checkpoints may supply low-level evidence but never replace the
workflow trace or artifact contract.

## Runtime Refinement

Solver-backed hard failures cannot be overridden by visual judgment. Visual
review may make an otherwise passing runtime result conditional when frames
show separation, missing motion, implausible bounce/sliding/settling,
interpenetration, tunneling, or stale/blank evidence.

For a fixable issue:

- preserve the previous patch and validation artifacts;
- inspect affected properties, grouping, collider approximation, and
  trajectories;
- validate a targeted replacement patch under workflow policy;
- apply it through usd-cli;
- rerun only the necessary scenarios when safe;
- refresh assignments, behavior assessment, validation, trace, and summary;
- stop at the configured cap or when evidence/runtime support is insufficient.

## Boundaries

- Do not move the physics workflow into usd-cli.
- `usd-cli physics apply` accepts an already-decided low-level patch; it does
  not own the workflow decision schema or policy.
- Reuse the workflow-owned usd-cli session during tuning/refinement; do not
  create a second sidecar or move tuning policy into usd-cli.
- Do not use a missing low-level primitive, including guarded topology-plan
  application, as a reason to block, deprecate, or redefine the workflow.
  Retain existing workflow helpers for unavailable primitives and record that
  fallback in the run manifest.
- A solver may run behind usd-cli, but raw results return to this
  workflow for interpretation.
- Reusable deterministic USD authoring, simulation, and metric helpers may live
  in `world_understanding`; workflow policy, schemas, finalizers, repair rules,
  and verdicts remain in `content_agent_workflows.physics`.

The child agent drives usd-cli for low-level inspection, schema authoring,
explicit-scenario simulation, and visual review, and owns
every tuning-loop judgment (scenario authoring/revision, candidate selection,
stop decisions). Deterministic sweep compute stays behind the wrapper-owned
sweep broker: one judge-free `run_tune` sweep per budgeted reservation,
isolated in the selected backend (`ovphysx`, `newton`, or `fake`). The fixed
`physics_agent.api.refine` loop (engine-internal VLM judge + LLM scenario
refiner) is not used by this workflow, and no wrapper or engine code invokes
an LLM/VLM — all model calls happen in the coding-agent session.

Use usd-cli `physics inspect`, `physics topology`, `physics apply`,
`physics validate`, and `physics simulate` only after the workflow has selected
the explicit targets or authored the explicit scenario. Component grouping,
topology planning, acceptance thresholds, and verdicts remain workflow
operations. When the workflow needs visual review of a generated
time-sampled recording, call the backend's generic frame-sequence render
operation with the recording as `scene_path`. The workflow agent consumes
backend and tuning artifacts/metrics; solver execution stays behind the scene
backend or the wrapper-owned backend adapter rather than inside ad hoc child
agent code.

The agentic wrapper may call the deterministic
`physics_agent.integrations.vomp_pipeline` adapter for calibrated evidence,
isolated official-runtime execution, and MassAPI publication. That adapter is a
low-level runtime boundary, not the legacy fixed Physics Agent workflow: target
selection, ordering, refinement, validation, artifacts, and promotion remain
owned by this agentic workflow.

Move reusable USD physics authoring, simulator execution, trajectory metrics, and
validation evidence helpers into `world_understanding` when they are useful
outside this workflow. Keep workflow-specific policy, schemas, finalizers, and
repair rules in `content_agent_workflows.physics`.

## References

- `references/physics-policy.md`
- `references/output-artifacts.md`
- `references/runtime-validation.md`
