---
name: content-workflow-material
description: Use with content-workbench when a long-running coding agent needs to assign visual or non-visual materials to a USD asset, use material libraries, run iterative visual quality assessment refinement, and produce canonical material artifacts and validation evidence.
---

# content-workflow-material

Use this skill with `content-workbench` when a long-running coding agent needs
to assign visual or non-visual materials to a USD asset and produce validated
material-assignment artifacts.

This skill owns workflow method. Workbench owns scene state, rendering, picking,
optimization, edit mechanics, restore, and export.

## Workflow

1. Load the Workbench skill.
2. Start or connect to Workbench.
3. Inspect the unoptimized asset first. When optimizer settings were not supplied
   explicitly, choose whether to optimize and select prototype flattening,
   deinstancing, splitting, and deduplication from observable asset structure.
   Record the decision before loading the material-authoring session.
4. Read durable `additional_instructions` before planning, then inspect
   reference images and reference files before assigning materials. Apply the
   instructions once as task-wide policy; do not expand them into per-prim
   prompts.
5. Inspect visible/renderable material candidates through Workbench snapshots,
   renders, picks, and material-binding queries.
6. Query or inspect the material library.
7. Group candidates by visible material family.
8. Apply material assignments through Workbench preview/edit APIs.
9. Render verification views.
10. Run visual quality assessment.
11. Refine material assignments when VQA finds fixable issues.
12. Produce canonical artifacts and restore/export accepted edits when required.

User instructions constrain interpretation of references, semantic names,
palette, transparency, and negative rules. Preserve them in `request.json` for
single-asset runs. For large-scene work, cite the frozen material-task request
digest in every decision and result so later collection can use the same
acceptance policy.

When instructions explicitly promote authored `primvars:displayColor` or
existing material bindings for named source roots, treat that as scoped
permission only after the frozen task request records an
`appearance_evidence_policy` for those roots. The default policy is clean-slate:
old source bindings, shader colors, and display colors are unavailable evidence.
For approved display-color scopes, render source colors and library swatches
under the same Workbench setup, rank their measured appearance, and let the
agent choose the final material using semantic class and scene evidence. The
ranking is advisory and must not become an RGB-to-material lookup table. For
decomposed large scenes, follow the `match-display-color` evidence step in
`content-workflow-asset-task-processing` and cite its request digest and match
artifacts in the decision.

## Required Artifacts

- `request.json`
- material decision patch or equivalent edit record
- `assignments.json`
- `visual_quality_assessment.json`
- render records and final render PNGs
- operation counts
- final summary
- trace/events where available
- restore/export artifacts when the workflow claims a source-space output

## VQA Refinement

VQA refinement is part of this skill, not a separate workflow skill.

When VQA reports fixable issues:

- preserve the previous decision patch;
- inspect the affected render(s), prims, and materials;
- apply targeted corrections through Workbench;
- re-render the affected views;
- update material assignment and VQA artifacts;
- stop when remaining issues are unfixable with available material granularity
  or when the configured iteration cap is reached.

## References

- `references/material-policy.md`
- `references/output-artifacts.md`
- `references/iterative-vqa-refinement.md`
