---
name: content-workflow-scene-collection
description: Collect validated decomposed-asset task results back onto original OpenUSD topology through domain-specific projection, representative propagation, conflict harmonization, independent domain layers, composition, and a sealed Workflow 3 handoff. Use after asset-task processing or when repairing collection for material, physics, articulation, geometry, or mixed-domain outputs.
---

# content-workflow-scene-collection

Route and validate durable collection while leaving authoring semantics to each
domain collector. Do not treat collection as a generic file merge.

## Preflight

1. Load the original scene identity, manifest and task catalogs, immutable work
   inventory, result index, and every completed domain payload.
   Load each indexed task request and its `additional_instructions`; treat that
   task-wide guidance as the acceptance policy for harmonization and final
   validation.
2. Load each required domain skill. Reject missing or failed required results,
   mismatched qualified IDs, unresolved source mappings, temporary authoring
   targets, and instance proxies.
3. Persist a collection input index so unchanged domains can resume without
   repeating work. Include every task request and its SHA-256 in the index.

## Collect By Domain

1. Project task-local targets into canonical original-path space through their
   selected manifest mappings.
2. Propagate representative opinions only through validated instance,
   prototype, payload, or structural-family mappings. Prefer source/prototype
   authoring that preserves native instancing. Resolve a family from its
   `representative_asset_id` and complete `member_paths`; do not require every
   member to have been processable or retained as a manifest asset.
3. Invoke the domain collector to resolve overlaps, nested representatives,
   explicit member overrides, and global constraints. Record every remap,
   merge, lift, rejection, and unresolved conflict.
4. Author and validate one independently inspectable domain layer or repaired
   USD. Seal one `DomainCollectionResult` per domain.
5. Compose requested domain layers only after their independent validators
   pass. Run topology preservation and required cross-domain validation. Final
   renders and VQA must evaluate the composed scene against the same task
   guidance used to produce the per-asset results. Turn that guidance into a
   short observable checklist and record pass/fail evidence for each item.
   Blankness, topology preservation, and valid bindings are not visual
   acceptance. Do not complete collection while a prominent requested color,
   transparency treatment, named exception, or negative constraint visibly
   fails; invalidate the earliest responsible phase instead.

## Complete

Write `collection_result.json` with `CollectionPhaseResult`. Cover every domain
result and artifact, topology report, composition output, cross-domain report,
and final deliverable in `artifact_paths`. Set `completion_policy_satisfied`
only when every required collector and requested validation policy passes.

Seal the draft and return it to the umbrella skill:

```bash
content-workflow-large-scene seal-result \
  --phase collection \
  --result RUN/03-collection/collection_result.json
```

For the implemented material collector, write a typed `CollectionRequest` and
run:

```bash
python -m content_agent_workflows.scene_collection \
  --request RUN/03-collection/request.json
```

This produces the projected-binding index, harmonization decisions,
material-only layer, composed scene, topology and cross-domain reports, and a
sealed `collection_result.json`. It does not mutate the source scene.

For a material survey with `visibility_policy: visible_only`, leave unmatched
computed-invisible regions unchanged. A visible representative candidate may
still propagate to the corresponding region of a hidden family member.

When collection exposes an upstream decomposition or task error, call the
umbrella's `invalidate-from` at the earliest affected phase. Do not silently
repair another domain's result or mutate a completed phase artifact.

When changing collection behavior, preserve original-topology projection,
representative propagation rules, independent domain layers, topology reports,
and deterministic handoff validation.
