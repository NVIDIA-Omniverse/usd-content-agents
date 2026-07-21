---
name: content-workflow-large-scene
description: Coordinate a large OpenUSD scene through decomposition, per-asset domain processing, and original-topology collection using durable run state and deterministic handoff gates. Use when a scene is too large or repetitive to process monolithically, when material/physics/other tasks must run over decomposed representatives, or when resuming or repairing a three-phase large-scene run.
---

# content-workflow-large-scene

Own phase selection, handoff validation, and recovery. Delegate phase methods
and domain judgment to their own skills. Never advance a phase by editing
`large_scene_run.json` directly.

Batch users launch this workflow through `content-workflow-cli scene run`. The
wrapper creates the resolved request and run state, then launches an agent with
this skill. The `content-workflow-large-scene` console commands below are
internal transition helpers for the running agent, interactive setup, tests,
and recovery; they are not the public batch entrypoint.

## Create A Run

Create one canonical run directory and state file:

```bash
content-workflow-large-scene create \
  --run-state RUN/large_scene_run.json \
  --run-id RUN_ID \
  --source-scene SCENE.usd \
  --additional-instructions-file USER_GUIDANCE.md \
  --task material \
  --task physics
```

Add each request, reference index, or source dependency snapshot that defines
the run with `--input-artifact PATH`. Creation makes `decomposition` ready and
records a source-input digest.

Use `--additional-instructions` or `--additional-instructions-file` for user
guidance that must survive every phase and agent-session boundary. Store it
once at scene scope; do not expand it per prim. Before Workflow 2 preparation,
copy the exact text into each applicable task request's
`additional_instructions` field. The processing handoff fails if scene-level
guidance is omitted from a task request.

For material tasks, create an explicit `appearance_evidence_policy` in the
material task request. The default is clean-slate:

```json
{
  "schema_version": "content-agent-workflows.appearance-evidence-policy.v1",
  "default": "ignore",
  "global_sources": [],
  "scopes": []
}
```

If the user guidance explicitly says to use display colors or existing
materials for named source roots, resolve those roots from the finalized
decomposition and add scoped entries with `sources` such as `display_color` and
`material_binding`. Do not enable broad/global authored-appearance evidence
from scene wording alone.

## Execute The Ready Phase

1. Read state with:

   ```bash
   content-workflow-large-scene status --run-state RUN/large_scene_run.json
   ```

2. Select `current_phase`. Stop when it is `null`; the run is complete.
3. Begin only a `ready` phase:

   ```bash
   content-workflow-large-scene begin-phase \
     --run-state RUN/large_scene_run.json \
     --phase PHASE
   ```

4. Load `content-workbench`, the mapped phase skill, and every requested domain
   skill needed in that phase:

   - `decomposition`: `content-workflow-scene-decomposition`
   - `asset_task_processing`: `content-workflow-asset-task-processing` plus
     material, physics, articulation, geometry, or other task skills
   - `collection`: `content-workflow-scene-collection` plus each required
     collector's domain skill

   Also load `additional_instructions` from `large_scene_run.json`. In Workflow
   2, apply it while planning and authoring every applicable task result. In
   Workflow 3, retain it as the acceptance policy for harmonization and final
   visual review.

5. Execute until the phase skill writes its concrete result artifact.
6. Seal a draft Workflow 2 or 3 result when needed:

   ```bash
   content-workflow-large-scene seal-result --phase PHASE --result RESULT.json
   ```

   Workflow 1 seals `decomposition_result.json` itself.

7. Inspect the deterministic gate before completion:

   ```bash
   content-workflow-large-scene validate-handoff \
     --run-state RUN/large_scene_run.json \
     --phase PHASE \
     --result RESULT.json
   ```

8. Complete only after `valid` is `true`:

   ```bash
   content-workflow-large-scene complete-phase \
     --run-state RUN/large_scene_run.json \
     --phase PHASE \
     --result RESULT.json
   ```

`validate-handoff` MUST return `valid: true` before `complete-phase` is
called. `complete-phase` re-runs the gate and atomically marks the successor
ready only on success; calling it against an invalid handoff durably marks the
phase `failed` and requires explicit recovery. Never infer completion from
files merely existing.

## Recover

Record execution failures with `fail-phase`. When a later phase disproves an
earlier assumption, return to the earliest affected phase:

```bash
content-workflow-large-scene invalidate-from \
  --run-state RUN/large_scene_run.json \
  --phase PHASE \
  --reason "CONCRETE REASON"
```

Do not mutate completed artifacts in place or patch around a failed handoff.
Keep superseded files for audit; the run-state transition history records which
digests were invalidated.

When user guidance changes after a valid decomposition, revise it through the
coordinator instead of editing run state or rerunning Workflow 1:

```bash
content-workflow-large-scene revise-instructions \
  --run-state RUN/large_scene_run.json \
  --additional-instructions-file USER_GUIDANCE.md \
  --reason "CONCRETE REASON"
```

This preserves the completed decomposition and invalidates Workflow 2 and its
collection successor. Prepare Workflow 2 again so every task request receives
a new frozen digest.

## Boundaries

- Keep semantic decomposition, task ordering, material/physics decisions, and
  harmonization out of this skill.
- Treat output digests as immutable identities. A changed artifact requires
  invalidation and a new sealed phase result.
- Preserve original topology as the durable authoring target. Extracted assets
  and preview layers are working evidence only.
- Keep extensions compatible with the existing run-state schema, handoff gates,
  and phase result contracts before adding a phase or new task domain.
