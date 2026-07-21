---
name: content-workflow-asset-task-processing
description: Plan and execute one or more domain tasks over finalized decomposed scene assets, with an immutable eligible work matrix, independent per-item results, agent-chosen ordering, durable decision memory, and a sealed Workflow 2 handoff. Use after scene decomposition for material, physics, articulation, geometry, labeling, or mixed asset processing.
---

# content-workflow-asset-task-processing

Run Workflow 2 over qualified `(manifest_id, asset_id, task_id)` work items.
Let the driving agent choose its plan; do not bake domain archetypes, task
ordering, or automatic result propagation into the runtime.

## Prepare

1. Load `manifest_catalog.json` and every selected finalized manifest.
2. Load every requested domain skill and write `task_catalog.json` using
   `TaskCatalog` from
   `content_agent_workflows.asset_task_processing.contracts`.
   Write one task request per catalog entry. Put user guidance in the request's
   `additional_instructions` field, copying scene-level instructions exactly
   from `large_scene_run.json` when present. Keep this guidance task-wide; do
   not duplicate it per asset or prim.
3. Mechanically expand each task over processable assets in its selected view:

   ```bash
   python -m content_agent_workflows.asset_task_processing prepare \
     --manifest-catalog MANIFEST_CATALOG.json \
     --task-catalog TASK_CATALOG.json \
     --output-dir RUN/02-asset-tasks \
     --input-digest PHASE_INPUT_DIGEST
   ```

   This writes immutable `asset_task_inventory.json` and separate mutable
   `asset_task_run_state.json`. Do not edit either manually.
4. Survey the complete work matrix and available evidence before committing the
   first result. Read every task request, summarize the applicable user guidance
   in the plan, and cite its frozen SHA-256. Write a plan draft, then preserve it
   as an immutable revision:

   ```bash
   python -m content_agent_workflows.asset_task_processing record-plan \
     --output-dir RUN/02-asset-tasks \
     --plan-file PLAN_DRAFT.md
   ```

## Process

1. Choose asset-major, task-major, mixed, or concurrent progression from the
   evidence and resource limits. Revise the plan when new evidence warrants it.
2. Process only eligible representatives. Use one Workbench sidecar where
   practical and serialize scene loads and OvRTX renders.
3. Consult prior results as explicit evidence when useful. Record fully
   qualified citations in `informed_by_results`; never copy an earlier result
   merely because an asset looks similar.
4. Write one independent `AssetTaskResult` per completed work item, including
   original-path mapping, domain payload paths, plan revision, validation, and
   warnings.
5. Commit the result, validator report, and matching `DecisionLedgerEntry` in
   one `commit-item` call. Do not write the ledger and result index as separate
   manual steps; partial commits intentionally fail the handoff gate.
6. Keep each domain's preview layer separate and provisional. Preserve failed
   or deferred items for explicit retry. Use an `AcceptedWaiver` for any
   required item intentionally omitted.

Use the phase module's `status`, `show-item`, `begin-item`, `commit-item`,
`fail-item`, and `waive-item` operations for state transitions. These are a
thin script over the package runtime, not another product-level CLI.

## Material Tasks

For a material task, survey source Mesh/GeomSubset candidates without changing
the assets. The survey uses computed USD visibility and excludes invisible
meshes. An invisible-only work item is an upstream material-view error, not an
empty assignment to waive:

```bash
python -m content_agent_workflows.asset_task_processing.material_task survey \
  --processing-dir RUN/02-asset-tasks \
  --render-index OPTIONAL_RENDER_INDEX.json
```

Review the surveys and references, author one complete material decision per
work item, and choose the execution order in an agent-authored batch plan. The
survey index exposes the material task request, its SHA-256, and its
`additional_instructions`. Set every material decision's
`task_request_digest` to that SHA-256. Run the plan through one Workbench
sidecar:

Resolve the Workbench URL once from the immutable scene-run request. This keeps
all material-task commands on the exact default or custom remote endpoint that
the launcher froze for the run. If `request.json` is missing or malformed, the
resolver stops with guidance to point `RUN` at a prepared scene workflow or
recreate the run:

```bash
WORKBENCH_URL="$(
  python .agents/skills/content-workflow-asset-task-processing/scripts/resolve_workbench_url.py \
    "$RUN/request.json"
)"
```

Treat source-authored appearance as untrusted CAD visualization metadata by
default. This includes existing material bindings, shader diffuse/base colors,
and `primvars:displayColor`. Never bulk-map CAD RGB values to material IDs, use
display color as the fallback decision rule, or preserve a source palette
merely because it is authored. Evidence priority is: explicit user guidance and
accepted references; rendered spatial/functional role and geometry; semantic
names; then prompt-approved authored appearance as scoped weak supporting
evidence.

The survey exposes authored appearance only when the frozen task request's
`appearance_evidence_policy` authorizes it. If `additional_instructions`
promotes display colors or existing material bindings for an exact list of
source prim roots, first materialize that prose into
`appearance_evidence_policy.scopes` before running `survey`. Use
`sources: ["display_color"]` for authored display color and
`sources: ["material_binding"]` for existing bound material/shader hints. Keep
the default policy as `ignore` and do not add broad/global sources unless the
user explicitly requested a preservation-first task.

Within an approved scope, use authored appearance to infer base hue and color
segmentation, infer material class from role/geometry/names/references, and then
choose the closest rendered library material within that plausible class. Keep
distinct color regions distinct and author source prims so instances inherit.
The scoped exception must not affect candidates outside the listed roots and
does not require per-prim prompts or a hard-coded material mapper. If a survey
does not expose an authored appearance field, treat it as unavailable evidence,
not as permission to inspect source USD materials manually.

When the prompt activates this policy, the driving agent extracts the exact
source roots and generates rendered retrieval evidence before authoring the
decision:

```bash
python -m content_agent_workflows.asset_task_processing.material_task \
  match-display-color \
  --processing-dir RUN/02-asset-tasks \
  --work-item-id TASK_ID:MANIFEST_ID:ASSET_ID \
  --scope /EXACT/SOURCE/ROOT \
  --top-k 5 \
  --workbench-url "$WORKBENCH_URL"
```

Repeat `--scope` for multiple prompt-approved roots. The command rejects scopes
outside the work-item root. It renders every library material and each unique
scoped display color on the same neutral Workbench sphere, converts measured
center-patch sRGB to CIE Lab, and writes a CIE76-ranked shortlist to
`display_color_matches.json`. The artifact records the frozen task-request path
and digest, exact scopes, render configuration, target swatches, library
swatches, and appearance-index path. The shared library index is content- and
render-config-addressed; target swatches are also render-config-addressed so a
camera or shading change cannot reuse stale images.

This command produces evidence only. It never assigns a material. The agent
must inspect role, geometry, names, references, and candidate descriptions,
then select a semantically plausible library material from the color-near
shortlist. A lower-ranked plausible finish may beat an unrelated rank-one
finish. Increase `--top-k` when the shortlist contains no plausible class.
Cite `display_color_matches.json` and its appearance index in the material
decision. Do not invoke this path unless the task prompt or accepted reference
explicitly promotes display color for the requested roots and the frozen
`appearance_evidence_policy` includes `display_color` for those roots.

For reference-driven work, render high-impact or uncertain families before
committing them. Check explicit positive and negative guidance visually, not
only candidate coverage and blankness. A nonblank render is necessary but does
not prove that requested colors, transparency, or semantic exceptions are
correct. Revise the decision when a prominent requested feature is absent or
visually reads as the wrong material.

```bash
python -m content_agent_workflows.asset_task_processing.material_task run-batch \
  --processing-dir RUN/02-asset-tasks \
  --batch-plan MATERIAL_BATCH_PLAN.json \
  --workbench-url "$WORKBENCH_URL"
```

The adapter validates exact candidate coverage and material-library names,
creates and closes one Workbench session per asset to bound memory, authors a
task-owned preview layer, validates it, and commits the standard result and
ledger entry. When rendering is requested, a blank image fails the work item
even if the Workbench API call succeeded. It never chooses a material or
invokes another model.

Use `content-agent-workflows.material-task-request.v2` for material requests.
It accepts reference paths, material-library paths, processing policy, and one
scene/task-level `additional_instructions` string. Workflow 2 freezes every
task request at preparation; changing one requires phase invalidation and a new
preparation. Every committed result and decision-ledger entry must cite the
matching task-request digest.

## Complete

Finalize and seal the phase only after required work is completed or waived:

```bash
python -m content_agent_workflows.asset_task_processing finalize \
  --output-dir RUN/02-asset-tasks
```

The runtime computes `processing_result.json` and its output digest from the
catalogs, referenced manifests, immutable inventory, mutable state, plan
revisions, ledger, index, completed results, validator reports, and domain
outputs. Return that result to the umbrella handoff validator.

Do not author durable opinions onto the original scene here. Workflow 3 owns
collection and harmonization.

When changing processing behavior, preserve immutable inventory semantics,
task-request digests, per-item result independence, decision ledger integrity,
and deterministic handoff validation.
