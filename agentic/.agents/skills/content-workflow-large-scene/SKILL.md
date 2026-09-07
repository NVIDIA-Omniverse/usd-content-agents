---
name: content-workflow-large-scene
description: Coordinate a large OpenUSD scene through decomposition, per-asset domain processing, and original-topology collection using durable run state and deterministic handoff gates. Use when a scene is too large or repetitive to process monolithically, when material/physics/other tasks must run over decomposed representatives, or when resuming or repairing a three-phase large-scene run.
metadata:
  author: NVIDIA Omniverse
---

# content-workflow-large-scene

> **Launcher-owned run envelope:** User-facing runs always start with
> `content-workflow-cli scene run` and continue with
> `content-workflow-cli scene resume`. The launcher owns the run envelope:
> request freezing, run-directory preparation, child launch, process
> observation, and resume entry. This skill and the
> `content_agent_workflows.large_scene` package own phase selection and method,
> per-asset workflow execution, handoff gates, checkpoint and artifact
> contracts, failure isolation, and recovery semantics inside that envelope.
> For low-level scene operations, use usd-cli as frozen in the run request.
> Changing the low-level tool configuration must
> never move workflow ownership into the launcher or scene tool.
> Backend history or hand-written notes cannot replace launcher-owned request,
> transition, checkpoint, artifact, recovery, and terminal-validation records.
> Diagnostic low-level previews are not acceptance evidence; final visual
> evidence remains subject to the workflow OVRTX contract.

Inside a launcher-owned child turn, never invoke `content-workflow-cli scene
run` or `content-workflow-cli scene resume`. They are parent-only entry points;
calling either recursively can replace active phase state and collide with the
parent-owned usd-cli daemon. Continue or recover with the `scene phase`,
`scene decompose`, `scene process`, `scene material-task`, and `scene collect`
commands documented below.

Own phase selection, handoff validation, and recovery. Delegate phase methods
and domain judgment to their own skills. Never advance a phase by editing
`large_scene_run.json` directly.

The `content-workflow-cli scene phase` commands below are transition helpers
for the running agent, tests, interactive repair, and recovery. They are not
the user-facing batch launcher and must not replace `scene run/resume`.

## Create A Run

Normally `content-workflow-cli scene run` creates the canonical run directory
and state file. For an explicit interactive repair/test that has no launcher
request yet, create the same contract with:

```bash
content-workflow-cli scene phase create \
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
   content-workflow-cli scene phase status --run-state RUN/large_scene_run.json
   ```

2. Select `current_phase`. Stop when it is `null`; the run is complete.
3. Begin only a `ready` phase:

   ```bash
   content-workflow-cli scene phase begin-phase \
     --run-state RUN/large_scene_run.json \
     --phase PHASE
   ```

4. Read the frozen run request, load the skill for its configured scene
   backend, then load the mapped phase skill and every requested domain skill
   needed in that phase:

   - `decomposition`: `content-workflow-scene-decomposition`
   - `asset_task_processing`: `content-workflow-asset-task-processing` plus
     material, physics, articulation, geometry, or other task skills
   - `collection`: `content-workflow-scene-collection` plus each required
     collector's domain skill

   In the public staged workflow, use the `usd-cli` skill for scene
   inspection, rendering, and edits. In an internal request explicitly
   configured for usd-cli, use the repo-root `usd-cli` skill only for those
   low-level operations. Do not route phase policy or state transitions into
   either scene tool. The request runtime, each task, and
   `large_scene_run.json` must name the same frozen backend. During Workflow 2,
   copy the task's backend and `scene_session_scope` into its frozen domain
   task request; do not silently select a different backend during resume. Copy
   task inputs exactly from the frozen launcher request. In particular, use
   `tasks[].inputs.materials_usd` as `material_library_path`; a `*_source` path
   is provenance, not the run-confined working input.

   Also load `additional_instructions` from `large_scene_run.json`. In Workflow
   2, apply it while planning and authoring every applicable task result. In
   Workflow 3, retain it as the acceptance policy for harmonization and final
   visual review.

5. Execute until the phase skill writes its concrete result artifact.
6. Seal a draft Workflow 2 or 3 result when needed:

   ```bash
   content-workflow-cli scene phase seal-result --phase PHASE --result RESULT.json
   ```

   Workflow 1 seals `decomposition_result.json` itself.

7. Inspect the deterministic gate before completion:

   ```bash
   content-workflow-cli scene phase validate-handoff \
     --run-state RUN/large_scene_run.json \
     --phase PHASE \
     --result RESULT.json
   ```

8. Complete only after `valid` is `true`:

   ```bash
   content-workflow-cli scene phase complete-phase \
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
content-workflow-cli scene phase invalidate-from \
  --run-state RUN/large_scene_run.json \
  --phase PHASE \
  --reason "CONCRETE REASON"
```

Do not mutate completed artifacts in place or patch around a failed handoff.
Keep superseded files for audit; the run-state transition history records which
digests were invalidated.

Do not infer a command-policy blocker. Attempt the exact required
`content-workflow-cli scene ...` command and cite its retained denial result
before recording a policy failure. Author declarative JSON, YAML, and Markdown
with the child file-editing tools rather than inline or module Python.

When user guidance changes after a valid decomposition, revise it through the
coordinator instead of editing run state or rerunning Workflow 1:

```bash
content-workflow-cli scene phase revise-instructions \
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
- Preserve `scene run/resume`, per-asset failure isolation, checkpoints,
  operation evidence, and terminal validation while low-level scene operations
  migrate between backends.
