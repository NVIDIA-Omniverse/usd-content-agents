---
name: content-workflow-scene-decomposition
description: Decompose a large OpenUSD scene into finalized processable representatives, instance/prototype/payload family mappings, optional extracted assets, and a sealed Workflow 1 handoff. Use for the decomposition phase of a large-scene run, for repeated or heavily instanced scenes, or when material and physics tasks need explicit scene partitions.
---

# content-workflow-scene-decomposition

Produce finalized manifest views and topology mappings. Do not make material or
physics assignments and do not mark the umbrella phase complete.

## Workflow

1. Read the source scene, requested tasks, phase input digest, and output
   directory from `large_scene_run.json`.
2. Inspect scene hierarchy, instances, prototypes, payloads, geometry counts,
   and prior diagnostics. Use `content-workbench` only for ambiguity that
   structural evidence cannot resolve.
3. Choose a decomposition intent and stable manifest ID. Create separate views
   only when requested domains genuinely need different processable boundaries.
4. Run deterministic decomposition:

   ```bash
   content-workflow-scene-decompose SCENE.usd \
     --output-dir RUN/01-decomposition \
     --manifest-id MANIFEST_ID \
     --intent INTENT \
     --extract-assets \
     --input-digest PHASE_INPUT_DIGEST
   ```

   Add include/exclude paths, payload/prototype controls, structural duplicate
   detection, and extraction worker limits as scene evidence requires. Keep
   extraction concurrency conservative for large scenes.

   For a `material_processing` view, default to
   `--exclude-invisible-assets --min-mesh-count 1` unless the task explicitly
   requires hidden variants. A physics or other domain may need a separate
   view with a different visibility policy.

5. Do not enable external LLM refinement. The driving agent owns ambiguous
   boundary decisions and may revise explicit decomposition controls and rerun
   after inspecting the frontier.
6. Inspect `scene_manifest.json`, `manifest_catalog.json`, extracted assets, and
   diagnostics. Require one processable source representative per validated
   family; keep non-representative instance members non-processable.
7. Confirm `decomposition_result.json` reports `success: true`, no unresolved
   issues, and a non-empty output digest.
8. Return the result path to `content-workflow-large-scene` for handoff
   validation and phase completion.

## Required Outputs

Preserve:

- `scene_decomposition_request.json`
- `manifest_catalog.json`
- every finalized `scene_manifest.json`
- extracted representative USDs when requested
- `decomposition_result.json`

The catalog and manifests must identify the same original scene. Never expose
an instance proxy or temporary extraction path as a durable authoring target.

When changing decomposition behavior, preserve manifest schema compatibility,
source-scene identity, original-path mappings, and deterministic handoff
validation.
