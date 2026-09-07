---
name: content-workflow-cli
description: >-
  Default launcher for supported Content Agent tasks from the repository root,
  including deterministic Geometry handoff, conversion, material or physics
  authoring, texture, segmentation, validation, large scenes, durable
  articulation, composed assets, review, and resume. Use fixed-pipeline app CLIs
  only when the user explicitly requests fixed control flow, YAML configuration,
  REST, benchmarking, or a named fixed-pipeline
  command.
metadata:
  author: NVIDIA Omniverse
---

# content-workflow-cli

Use this skill from the repository root for supported Content Agent tasks unless
the user explicitly requests a fixed pipeline app pipeline.

> **Layering:** `content-workflow-cli` launches an agent and owns the durable
> run envelope. The selected `content-workflow-*` skill owns workflow policy,
> sequencing, validation, evidence, artifacts, and recovery. Every workflow
> uses the repo-owned usd-cli skill for low-level scene operations.

Use this skill when the user wants the batch shortcut for a prepared agentic
asset workflow.

`content-workflow-cli` is a launcher. It is not the source of truth for
workflow methodology or usd-cli command mechanics.

## Geometry Workflow

Use the public Geometry-only launcher for a fresh, deterministic handoff from
one admitted CAD, mesh, immutable authoring-provider bundle, or USD source:

```bash
content-workflow-cli geometry run path/to/source.step \
  --prompt-file path/to/geometry-requirements.md \
  --reference-image path/to/reference.png \
  --output-dir runs/source-geometry-001
```

The launcher freezes `geometry_request.json`, calls the shared typed Geometry
workflow once, and writes `geometry_workflow_result.json`. It defaults to
digest-bound six-view OVRTX evidence, correspondence-preserving optimization,
path-traced rendering with 64 accumulation iterations, and Geometry-only
validation. It does not launch a child coding agent or own Geometry
methodology. Exit `0` means `handoff_ready=yes`, `3` means
`conditional`, `1` means `no`, and `2` means the request or execution failed.

Use `--dry-run --json` to validate and freeze a request without execution. Use
`--no-render-evidence` only for explicitly nonvisual diagnostics; it is not
accepted visual evidence. Existing output directories are rejected so prior
evidence cannot be overwritten or ambiguously resumed. Apply the
`content-workflow-geometry` skill for workflow policy and evidence decisions.

## Current Primary Workflow

```bash
content-workflow-cli materials assign \
  --usd path/to/asset.usdc \
  --reference-image path/to/reference.png \
  --additional-instructions-file path/to/material-guidance.md \
  --optimizer-selection agent \
  --materials-yaml path/to/materials.yaml \
  --output-dir runs/example
```

Use `--optimizer-selection agent` when optimizer behavior should be chosen from
the asset rather than fixed by the launcher. The wrapper first provides an
unoptimized inspection through the backend frozen in the request, validates the resulting
`raw/optimizer_decision.json`, and only then creates the material-authoring
session with the selected optimization, prototype flattening, deinstancing,
splitting, and deduplication settings. Fixed optimizer flags remain available
for reproducible diagnostics and explicit user overrides, but do not combine
them with agent selection.

Optimizer decisions are task-scoped. `materials assign` selects settings against
visible material coverage, independent appearance authoring, and source-path
mapping. `physics apply` runs a separate unoptimized inspection and selects
settings against component membership, rigid-body and joint topology,
collider/helper roles, legal authoring targets, and runtime behavior. Each
`optimizer_decision.json` declares `task`; the wrapper rejects a decision for a
different operation.

Use `--additional-instructions` for short inline guidance or
`--additional-instructions-file` for durable multi-line policy. The wrapper
stores the normalized text in `request.json`, includes it in the child-agent
task, and reuses it during bounded VQA refinement. Provide task-wide guidance
once; do not generate per-prim prompt copies.

## Large-Scene Workflow

Use the public scene launcher rather than invoking the internal transition tool
or agent runtime directly:

```bash
content-workflow-cli scene run \
  --usd path/to/scene.usd \
  --task material \
  --materials-yaml path/to/materials.yaml \
  --reference-dir path/to/references \
  --reference-image path/to/accepted-render.png \
  --additional-instructions-file path/to/material-guidance.md
```

The wrapper writes a resolved request and `large_scene_run.json`, launches one
long-running agent with the root workflow skills, and returns success
only after decomposition, asset-task processing, and collection complete their
handoff gates. Resume a prepared or interrupted run with:

```bash
content-workflow-cli scene resume --run-dir runs/RUN_ID
```

`content-workflow-cli scene phase` is the transition utility used by the
umbrella skill, tests, and recovery. Do not expose it as the user launch command.

## Articulation Workflow

Use the public articulation commands for a prompt-driven Joint Agent run:

```bash
ARTICULATION_INTENT="Present the drawer candidates for review, then author only "
ARTICULATION_INTENT+="approved prismatic joints."

content-workflow-cli articulation run \
  --usd path/to/asset.usdz \
  --joint-config runs/configs/joint.yaml \
  --output-dir runs/articulation \
  --intent "$ARTICULATION_INTENT"
```

The default command freezes the request and launches one long-running child
routed through `content-workflow-articulation` and its four atomic skills. Use
`--execution-mode fixed` only for an explicit compatibility or baseline run.
When `--embedded-run-state` is present, the outer asset coordinator remains the
only reasoning loop and drives the same focused prepare/apply contract without
launching a child.

Inspect the reported candidates and usd-cli evidence, then submit one
decision for every review-required candidate:

```bash
content-workflow-cli articulation review \
  --run-dir runs/articulation \
  --decisions-json runs/articulation-decisions.json \
  --reviewer asset-owner
```

If an embedded human review uses `revise`, preserve the parent graph and bind a
typed, run-confined revision patch before any authoring:

```bash
content-workflow-cli articulation revise-graph \
  --run-dir runs/articulation \
  --revision-patch runs/articulation/graph-revision-patch.json
```

This command publishes an immutable parent/child revision receipt, resets the
outer Joint gate to `needs_review`, and requires a complete decision file for
the new graph digest. It rejects stale parent acceptance and does not invoke a
provider or authorer.

Resume an interrupted CLI-created run with:

```bash
content-workflow-cli articulation resume --run-dir runs/articulation
```

For the public articulation commands:

- Inspect `articulation_candidates.json` and
  `scene_evidence/manifest.json`, then submit one decision for every
  `review_required_candidate_ids` value. Accept only native-ready revolute or
  prismatic candidates.
- Treat candidate and usd-cli evidence digests as immutable. The review
  receipt binds each decision to those exact artifacts; do not edit them in
  place.
- Resume with the same run directory and stored request. Completed phases and
  a complete verified usd-cli manifest are reused; incomplete evidence
  capture may be retried. Changed source, intent, Joint config, evidence
  policy, or launcher metadata requires a new run.
- Treat only `completed` as final success. It means reviewed articulation-graph
  authoring left source mass and collider APIs unchanged and the accepted IDs
  exactly match the self-contained saved USDZ joint graph. Physics authoring is
  a separate workflow.
- Treat `cancelled` and `failed` checkpoints as terminal diagnostic records.
  Preserve the run, correct the underlying problem, and start with a new run
  directory rather than expecting `resume` to repeat a backend phase.

## Composed Asset Workflow

Use one durable prompt when Geometry, Joint, Material, Texture, Physics, and
Validation must work on the same asset in order:

```bash
content-workflow-cli asset run \
  --usd path/to/cabinet.usdz \
  --prompt "Prepare the geometry, find the drawer joints, assign painted metal with light wear, add physics, validate, and package the result." \
  --joint-config path/to/joint.yaml \
  --materials-yaml path/to/materials.yaml \
  --materials-usd path/to/materials.usda \
  --output-dir runs/cabinet-composed
```

The same single long-running coordinator is callable from an interactive
repository-root session and from this CLI. It owns prompt interpretation, typed
per-stage plans, evidence review, bounded refinement/revisits, and stop
decisions. Geometry, Joint, Material, Texture, Physics, and Validation remain
typed executors/validators/finalizers and must not launch nested reasoning
agents. Geometry must publish a digest-bound durable USDC and evidence before
Articulation can begin.

The command pauses after publishing digest-bound Joint candidates. Submit one
decision per required candidate, then continue the same run:

```bash
content-workflow-cli asset review \
  --run-dir runs/cabinet-composed \
  --decisions-json path/to/decisions.json \
  --reviewer asset-owner
```

Use `asset resume --run-dir ...` after an interruption. A failed or cancelled
stage additionally requires `--recover "reason"`. Accepted stage outputs are
rehash-verified and never rerun. Success requires a canonical final USDZ, a
combined digest report, and a passing top-level terminal validation.

## Validation Workflow

Use the public validation launcher to turn a prompt into one evidence-backed
report. Validation never modifies the asset it inspects:

```bash
content-workflow-cli validate run \
  --usd path/to/asset.usdc \
  --task "Validate that this renders successfully and looks like the reference. \
Do not modify the asset; save a report with evidence and recommended actions." \
  --reference-image path/to/reference.png \
  --output-dir runs/validation-example \
  --runner codex \
  --model gpt-5.6-sol \
  --render-backend remote
```

Validation selects no planning provider, model, provider execution mode, or
rendering backend by default. Name the runner and model explicitly; Claude also
requires `--claude-execution-mode sdk|cli`. Name `--render-backend` only when
current rendering is requested.

The explicit runner authors only a preparation-bound check-selection proposal;
trusted outer code accepts and executes the exact checks, then publishes
`validation_request.json`, `validation_plan.json`, `validation_result.json`,
`validation_evidence.json`, and `final_summary.json` into the run directory.
Exit status is `0` for pass, `1` for fail (or warn with `--fail-on-warn`), `2`
for a configuration or identity error, and `3` for a cancelled run.

Resume an interrupted run at its next unfinished check:

```bash
content-workflow-cli validate resume --output-dir runs/validation-example
```

Resume reloads the published request, so a changed prompt, asset, or reference
fails closed instead of silently revalidating something else. An accepted
`render_valid` result is reused and only `look_right` executes.

If the previous run was interrupted mid-check, its claim is still live and
resume refuses to run beside it. Once that runner is confirmed gone — after a
Ctrl-C, for example — release the claim explicitly:

```bash
content-workflow-cli validate resume \
  --output-dir runs/validation-example \
  --recover-orphaned-claims
```

Do not pass `--recover-orphaned-claims` while another runner may still be
working; the default refusal is what prevents two runners from executing the
same check.

## Wrapper Responsibilities

- parse workflow inputs;
- create a run directory;
- write `request.json`;
- prepare the workflow-owned usd-cli session;
- launch Codex or Claude Code for standalone skill-routed Texture and
  Articulation workflows, while embedded composed execution launches no nested
  child;
- pass a compact structured task request;
- validate required workflow artifacts;
- require terminal phase validation for a large-scene run;
- preserve logs, traces, and partial outputs.

The wrapper and owning domain skills also retain conversion routing, SimReady
Foundation execution, optimizer configuration and source-space restoration,
large-scene transition policy, material-manifest selection, validation,
artifact completeness, recovery, and terminal success semantics. A scene
backend cannot relax or replace those contracts.

## Agent Responsibilities

The launched agent should load the owning workflow skill first, especially:

- `content-workflow-material`
- `content-workflow-physics`
- `content-workflow-simready`
- `content-workflow-texture` and its four atomic Texture skills
- `content-workflow-articulation` and its four atomic Articulation skills
- `content-workflow-large-scene` (umbrella orchestration skill)

It then loads the repo-root `usd-cli` skill for low-level operations.

## Rules

- User-facing large-scene work always starts or resumes through
  `content-workflow-cli scene run/resume`; the transition CLI is internal to the
  running agent, tests, and recovery.
- SimReady and Convert-to-USD use their existing `content-workflow-cli` /
  `content_agent_workflows` paths; usd-cli remains the low-level scene tool.
- A missing low-level primitive must not remove or redefine its owning
  workflow. Use the workflow-selected supported helper or usd-cli primitive.
- Backend history, checkpoints, and command telemetry are supplemental
  low-level evidence. They never replace the workflow request, decisions,
  validation evidence, durable trace, failure artifacts, or final verdict.
- Final workflow visual evidence must satisfy the configured OVRTX render
  contract; diagnostic backend previews are not acceptance evidence.
- If a task needs custom reasoning, inspection, or recovery, start an
  interactive long-running agent from `agentic/` with the domain workflow skill
  and usd-cli.

If a task needs custom reasoning, inspection, or recovery, prefer an interactive
long-running agent session from the repository root using the usd-cli and
workflow skills directly. Do not require the user to enter `agentic/`.
