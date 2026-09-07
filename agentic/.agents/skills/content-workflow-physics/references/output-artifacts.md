# Physics Workflow Output Artifacts

Canonical physics workflow outputs should include:

```text
physics_assignments.json
physics_behavior_assessment.json
validation_evidence.json
final_summary.md
raw/physics_decision_patch.json
raw/physics_components.json
raw/physics_topology.json
raw/physics_topology_plan.json        # only when intent-gated repair is used
raw/physics_topology_report.json      # only when a plan is applied
raw/physics_apply_report.json
raw/physics_inspection/               # portable optimized scene bundle, when used
runtime_validation/
trace/
```

For optimized runs, `raw/physics_inspection/` is the workflow-owned portable
bundle opened by usd-cli and pinned by the run packet. The workflow must stage
and export it within the run directory; low-level scene operations must not
write outside that boundary.

When the workflow claims a durable sim-ready output, it should also include the
authored USD/USDZ path, for example:

```text
physics.usdc
```

The default durable physics output uses binary crate encoding to avoid the
large storage and transfer overhead of ASCII USDA on geometry-heavy assets.
An explicit `.usd`, `.usda`, or `.usdz` path still preserves the requested
format. Under `--tune`/`--refine`, tuned candidates are materialized and
promoted byte-identically in the requested format; `.usdz` outputs are
rejected up front because a package cannot be authored by a plain layer
export, so tuning supports `.usd`, `.usda`, and `.usdc` targets only.

`physics_assignments.json` should include:

- schema version;
- source asset and authored output asset;
- exact source and prepared-asset SHA-256 digests;
- target runtime;
- path space;
- per-decision runtime prim paths;
- source path expansions when optimized;
- component/material classification;
- physical properties;
- collision approximation;
- rigid body grouping;
- confidence, evidence, and rationale;
- quality warnings;
- coverage status;
- unresolved issues.

`validation_evidence.json` should include checks for:

- physics properties;
- collisions;
- non-visual materials;
- runtime loadability;
- no explosions;
- no penetration when evaluated;
- stable settle or task-specific behavior;
- simulation visual review when rendered validation frames were reviewed;
- unresolved warnings and repair hints.

`physics_behavior_assessment.json` should include:

- `status`: `pass`, `fixed`, or `unresolved_issues`;
- `checked_views` and `rendered_frames`;
- `runtime_report`;
- `issues_found`, `issues_fixed`, and `unresolved_issues`;
- short `assessment_notes`.

`runtime_validation/` should include simulation evidence when available:

- derivative simulation scene USD;
- trajectory JSONL;
- recording USD with time samples;
- metric summary JSON;
- optional rendered PNG frames;
- simulator logs or unavailable-runtime diagnostics.

The final summary should state whether the authored USD is sim-ready, conditional,
failed, or not evaluated, and should point to the blocking evidence instead of
hiding limitations in prose.
