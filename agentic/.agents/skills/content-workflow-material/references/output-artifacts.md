# Material Assignment Output Artifacts

`request.json` is a launcher-owned, immutable workflow input. Read it for the
frozen task and integrity contract, but never create, replace, or edit it while
assembling output artifacts.

Canonical material assignment outputs should include:

```text
assignments.json
visual_quality_assessment.json
api_operation_counts.json
final_summary.md
raw/material_decision_patch.json
raw/material_applied_decision_patch.json
raw/material_application_receipt.json
raw/material_post_apply_review.json
raw/material_binding_audit.json
raw/ovrtx_render_probe.json
raw/material_operation_receipts.json
raw/final_render_records.json
final_renders/
trace/
```

`final_renders/` includes the four segmented verification views, 24 sealed
OVRTX turntable frame PNGs, and `final_turntable.gif`. A separate post-apply VQA
must bind the exact receipt identity of every verification view, turntable
frame, and GIF before publication. Static-only output is incomplete unless the
frozen workflow request explicitly retires the turntable requirement.

When the workflow claims a durable source-space output, prefer a `.usdc`
filename for geometry-heavy assets. Explicit `.usd`, `.usda`, and `.usdz`
outputs remain supported when interoperability or packaging requires them.

Assignments should include:

- schema version `content-agents.assignments.v1`;
- `source_usd` set to the immutable staged source path from the workflow task;
- `material_library` as an object with `materials_usd` set to the staged USD
  library path and `materials_yaml` set to the task's manifest path (not a
  string-valued shortcut);
- path space;
- an `assignments` list (not `assignment_groups`);
- coverage status `material_assignment` or `preserved_existing`;
- assigned material name and path;
- target inspection paths;
- source path expansions when optimized;
- a `coverage` object with the exact nonnegative integer fields
  `candidate_visible_prim_count`, `material_decision_prim_count`,
  `material_assignment_prim_count`, `preserved_existing_prim_count`,
  `missing_assignment_prim_count`, `rejected_assignment_prim_count`, and
  `unassigned_visible_prim_count`;
- evidence and rationale;
- unresolved issues.

Visual quality assessment should include:

- schema version `content-agents.visual-quality-assessment.v1`;
- status `pass`, `fixed`, or `unresolved_issues`;
- unique run-contained `checked_views` plus separate `reference_images` and
  `reference_files` lists;
- typed `issues_found` entries with `severity`, `description`,
  `expected_appearance`, `actual_appearance`, `status`,
  `affected_prim_paths`, and `evidence_artifacts`;
- string lists for `issues_fixed` and `unresolved_issues`;
- a nonempty string `assessment_notes`;
- affected renders;
- affected prims or objects;

Use `unresolved_issues` status exactly when its list is nonempty. A documented
palette or renderer limitation is an unresolved issue, not a separate status.

Operation counts should use schema version
`content-agents.api-operation-counts.v1`. `material_override_commands` is the
number of accepted final per-prim material decisions and must match the
decision patch; record rejected A/B previews separately. `final_renders` is the
exact final PNG count, and `render_count_total` may include additional
inspection or rejected-preview renders.
