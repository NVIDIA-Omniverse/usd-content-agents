---
name: content-workflow-asset-task-processing
description: Plan and execute one or more domain tasks over finalized decomposed scene assets, with an immutable eligible work matrix, independent per-item results, agent-chosen ordering, durable decision memory, and a sealed Workflow 2 handoff. Use after scene decomposition for material, physics, articulation, geometry, labeling, or mixed asset processing.
metadata:
  author: NVIDIA Omniverse
---

# content-workflow-asset-task-processing

> **Workflow-owned processing:** Immutable inventory, plan revisions,
> per-item failure isolation, decision memory, commit gates, and the Workflow 2
> handoff remain in
> `content-workflow-cli scene process`. Use the scene
> tool frozen in the large-scene request only for per-asset inspection,
> rendering, and accepted authoring. Existing
> usd-cli material-task adapters remain supported where selected; changing a
> low-level backend must not remove batch orchestration or resume.

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
   content-workflow-cli scene process prepare \
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
   content-workflow-cli scene process record-plan \
     --output-dir RUN/02-asset-tasks \
     --plan-file PLAN_DRAFT.md
   ```

## Process

1. Choose asset-major, task-major, mixed, or concurrent progression from the
   evidence and resource limits. Revise the plan when new evidence warrants it.
2. Process only eligible representatives. Do per-item scene inspection,
   OVRTX rendering, and accepted material/physics authoring through the
   workflow-owned usd-cli session. Serialize scene loads and GPU renders where
   resources are constrained. Copy the parent scene task's frozen
   `scene_backend` and `scene_session_scope` into each domain task request's
   `processing_policy`. With `scene_session_scope: per_asset`, create an
   independent usd-cli named session and checkpoint
   lineage for every work item. Close that state before moving to the next
   item; a retry starts fresh state for only the failed/deferred item and never
   reopens a completed item's sealed artifacts.
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

Before processing material work, read
[`references/material_tasks.md`](references/material_tasks.md). It defines the
survey and batch commands, authored-appearance evidence policy, display-color
matching route, visual review gates, and frozen task-request digest contract.
Run those steps through usd-cli for low-level scene operations, then commit
through the pure-Python `asset_task_processing` state machine:

```bash
content-workflow-cli scene material-task \
  match-display-color \
  --processing-dir RUN/02-asset-tasks \
  --work-item-id TASK_ID:MANIFEST_ID:ASSET_ID \
  --scope /EXACT/SOURCE/ROOT \
  --top-k 5
content-workflow-cli scene material-task run-batch \
  --processing-dir RUN/02-asset-tasks \
  --batch-plan MATERIAL_BATCH_PLAN.json
```

## Complete

Finalize and seal the phase only after required work is completed or waived:

```bash
content-workflow-cli scene process finalize \
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
