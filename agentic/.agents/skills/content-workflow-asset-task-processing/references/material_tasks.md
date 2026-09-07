# Material Task Processing

> **Scene tool:** the `material_task` adapters below (`survey`,
> `match-display-color`, `run-batch`) use usd-cli for low-level scene operations
> and still commit typed survey, retrieval-evidence, and authoring artifacts
> through the same pure-Python
> `asset_task_processing` state machine. A missing low-level primitive is not a
> reason to drop the batch orchestration or resume contract.

Survey source Mesh/GeomSubset candidates without changing the assets. The
survey uses computed USD visibility and excludes invisible meshes. An
invisible-only work item is an upstream material-view error, not an empty
assignment to waive:

```bash
content-workflow-cli scene material-task survey \
  --processing-dir RUN/02-asset-tasks \
  --render-index OPTIONAL_RENDER_INDEX.json
```

Review the surveys and references, author one complete material decision per
work item, and choose execution order in an agent-authored batch plan. The
survey index exposes the material task request, its SHA-256, and its
`additional_instructions`. Set every material decision's `task_request_digest`
to that SHA-256.

## Appearance Evidence

Treat source-authored appearance as untrusted CAD visualization metadata by
default. This includes existing material bindings, shader diffuse/base colors,
and `primvars:displayColor`. Never bulk-map CAD RGB values to material IDs, use
display color as the fallback decision rule, or preserve a source palette only
because it is authored. Evidence priority is explicit user guidance and
accepted references; rendered spatial/functional role and geometry; semantic
names; then prompt-approved authored appearance as scoped weak evidence.

The survey exposes authored appearance only when the frozen task request's
`appearance_evidence_policy` authorizes it. If `additional_instructions`
promotes display colors or existing material bindings for an exact list of
source prim roots, materialize that prose into
`appearance_evidence_policy.scopes` before running `survey`. Use
`sources: ["display_color"]` for authored display color and
`sources: ["material_binding"]` for existing bound material/shader hints. Keep
the default policy as `ignore`; add no global sources unless the user requested
a preservation-first task.

Within an approved scope, use authored appearance to infer base hue and color
segmentation. Infer material class from role, geometry, names, and references,
then choose the closest rendered library material in that plausible class.
Keep distinct color regions distinct and author source prims so instances
inherit. The exception must not affect candidates outside listed roots. If a
survey omits authored appearance, treat it as unavailable evidence, not
permission to inspect source USD materials manually.

## Display-Color Retrieval

When the prompt activates this policy, extract exact source roots and generate
rendered retrieval evidence before authoring the decision:

```bash
content-workflow-cli scene material-task \
  match-display-color \
  --processing-dir RUN/02-asset-tasks \
  --work-item-id TASK_ID:MANIFEST_ID:ASSET_ID \
  --scope /EXACT/SOURCE/ROOT \
  --top-k 5
```

Repeat `--scope` for multiple prompt-approved roots. The command rejects scopes
outside the work-item root. It renders every library material and each unique
scoped display color on the same neutral sphere, converts measured
center-patch sRGB to CIE Lab, and writes a CIE76-ranked shortlist to
`display_color_matches.json`. The artifact records the frozen task-request path
and digest, exact scopes, render configuration, target swatches, library
swatches, and appearance-index path. Shared library and target-swatch indexes
are content- and render-config-addressed so camera or shading changes cannot
reuse stale images.

This route produces evidence only and never assigns a material. Inspect role,
geometry, names, references, and candidate descriptions, then select a
semantically plausible material from the color-near shortlist. A lower-ranked
plausible finish may beat an unrelated rank-one finish. Increase `--top-k` when
the shortlist has no plausible class. Cite `display_color_matches.json` and its
appearance index in the decision. Invoke this route only when the prompt or an
accepted reference promotes display color for the requested roots and the
frozen policy includes `display_color` for those roots.

## Batch And Visual Gates

For reference-driven work, render high-impact or uncertain families before
committing. Check positive and negative guidance visually, not only candidate
coverage and blankness. A nonblank render does not prove requested colors,
transparency, or semantic exceptions are correct. Revise decisions when a
prominent feature is absent or reads as the wrong material.

```bash
content-workflow-cli scene material-task run-batch \
  --processing-dir RUN/02-asset-tasks \
  --batch-plan MATERIAL_BATCH_PLAN.json
```

The adapter validates exact candidate coverage and material-library names,
creates and closes one usd-cli session scope per asset to bound memory, requires
an OVRTX readiness probe, authors a
task-owned preview layer, validates it, and commits the standard result and
ledger entry. A requested blank render fails the work item even if the API call
succeeded. The adapter never chooses a material or invokes another model.

Use `content-agent-workflows.material-task-request.v2` for material requests.
It accepts reference paths, material-library paths, processing policy, and one
scene/task-level `additional_instructions` string. Workflow 2 freezes every
task request at preparation; changing one requires phase invalidation and a new
preparation. Every committed result and decision-ledger entry must cite the
matching task-request digest.
