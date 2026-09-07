---
name: content-workflow-material
description: Default material-authoring workflow for supported, unqualified USD material tasks. Use with usd-cli when a long-running coding agent assigns visual or non-visual materials, uses libraries, performs iterative visual review, and produces canonical artifacts and validation evidence; use material-agent-* only for explicit fixed pipeline requests.
metadata:
  author: NVIDIA Omniverse
---

# content-workflow-material

Use this skill when a long-running coding agent must
assign visual or non-visual materials and produce validated
material-assignment artifacts.

This skill owns workflow method, material-library semantics, selection policy,
sequencing, visual review, validation, evidence, artifacts, and recovery. The
usd-cli owns only low-level scene state, inspection, binding,
rendering, and save/export mechanics.

## Workflow

1. Read the resolved request, including the immutable
   source path, references, instructions, `materials.yaml`, output path, and
   clean-slate policy.
2. Load the repo-root `usd-cli` skill for low-level operations only.
   Before invoking it, install the ordinary in-tree package into the active
   environment with `uv pip install -e "apps/usd_cli[cli,server]" --overrides
   apps/usd_cli/requirements/usd-exchange-override.txt`, then verify `usd-cli
   --version`. The override is mandatory so another package cannot supply the
   native `pxr` modules.
   Never invoke Python or PyPy, execute inline code or a scratch `.py` file, or
   import repository internals from the child shell. Use `jq -e` and documented
   checked-in CLI commands; let the wrapper run deterministic validators.
3. Inspect the unoptimized asset first. When optimizer settings were not supplied
   explicitly, choose whether to optimize and select prototype flattening,
   deinstancing, splitting, and deduplication from observable asset structure.
   Record the decision before loading the material-authoring session.
4. Read durable `additional_instructions` before planning, then inspect
   reference images and reference files before assigning materials. Apply the
   instructions once as task-wide policy; do not expand them into per-prim
   prompts. If a non-image reference needs a browser or document renderer,
   keep its profile, lock files, sockets, and other scratch state in a temporary
   directory outside `run_dir`; copy only ordinary-file evidence back into the
   run. Browser-created symlinks and sockets are not durable workflow artifacts.
5. In clean-slate mode, open the authoring asset and run `appearance clear`
   before freezing the candidate set. The clear operation deinstances native
   instance roots; rerun the visible-mesh query afterward and build the
   candidate artifact from that post-clear stage. Treat `skip_instances` as a
   prohibition on unresolved instance-proxy targets, not permission to drop a
   visible instance family. A pre-clear proxy-only query never proves a genuine
   zero-candidate result. Inspect visible/renderable material candidates through
   usd-cli hierarchy, render, selection, and material-binding commands. Keep hierarchy reads
   bounded: never use `snapshot -a` on mesh-heavy scopes or dump point,
   face-index, normal, or UV arrays into the agent context. Query only the
   named metadata or prims needed for the decision. Treat the project daemon as
   one serialized scene session: never issue parallel shell/tool calls when any
   of them invokes usd-cli. Wait for the complete JSON response before the next
   scene command. An OVRTX worker PID is a persistent backend process, not a
   render-progress signal; never poll `ps`/`kill -0` to decide whether a render
   finished. The launcher-owned `raw/ovrtx_probe.json` receipt is authoritative
   for the already completed renderer preflight; do not run `render-probe`
   again from the child workflow.
6. Query or inspect the material library.
7. Group candidates by visible material family.
8. Write the complete exact decision patch, then apply its heterogeneous
   assignments once with `usd-cli material-apply`; never expand it into one
   model or shell action per prim. Keep `usd-cli material` for a genuinely
   surgical one-target preview or repair. Audit the live bindings, save the
   exact requested output with `--flatten`, reopen it, and
   verify the durable binding coverage before starting a long render. Preserve
   a post-apply checkpoint. After every accepted refinement, repeat this
   save/reopen verification before replacing canonical visual evidence.
   For an optimized inspection run, clean-slate clearing de-instances the
   authoring stage: bind the live composed consumer named by each
   `runtime_prim_paths` entry, while preserving the unique
   `inspection_to_source` translation separately in
   `source_prim_paths`/`prim_paths`. The latter may name a backing/prototype
   prim and is wrapper-owned durable-restore provenance. A successful live
   consumer audit never authorizes rewriting it to the runtime path, and
   binding only that backing path does not satisfy the live apply gate.
9. Render verification views plus the required 24-frame OVRTX turntable and
   seal every PNG frame/backend response. For structured content-workflow runs,
   return as soon as those renders and the saved output are complete. Do not
   inspect the turntable frames one by one or search for an image compositor:
   the wrapper builds hash-bound review sheets, assembles the GIF, and launches
   a fresh compact VQA turn. Run every agent
   render as one daemon-owned detached job: capture the job ID returned by
   `usd-cli render ... --detach`, inspect progress only with `usd-cli jobs`, and
   collect the completed response with `usd-cli wait <job>`. Finish one job
   before submitting the next. Never run an orbit as an untracked synchronous
   shell command, put it in the shell background, or enqueue another render,
   history, save, or inspection command while a synchronous render owns the
   session.
   Prefer final verification views at or below 768x768. The first materialized
   OVRTX render may spend many minutes compiling shaders even when earlier
   clean-slate inspection renders were fast. While the renderer process is
   alive and making CPU or GPU progress, let the `usd-cli` render command run
   to its configured timeout; elapsed time alone is not a failed render.
   Do not cancel the client, signal the sidecar/OVRTX worker, or start a second
   sidecar for the run.
   When a dark candidate is difficult to distinguish from the background,
   compare the default render with one temporary usd-cli dome-fill render;
   adjust the fill from visual evidence and restore the default rig for final
   evidence.
10. Run visual quality assessment in a separate post-apply turn whose receipt
    binds the exact saved output, every final render, and deterministic sheet
    tile mapping. Compare the visible result to the reference, not the intended
    material names: a dominant surface cannot pass when its rendered
    brightness, saturation, metalness, or transparency class is visibly
    different from the reference. This review turn is tool-free and returns
    only the semantic assessment; the wrapper writes mechanical bindings.
11. When VQA finds fixable issues, launch a fresh compact repair turn with the
    active issue packet. Require a declared source-prim repair scope, reject
    unrelated decision changes, use the exact absolute material-library path,
    regenerate the complete final render set (the wrapper selects usd-cli's
    newest collision-suffixed orbit generation), and
    run another fresh independent VQA before accepting the repair.
12. Pass the structural completion gates, then restore/export accepted edits
    when required. Never publish from the pre-apply assessment. If visual
    evidence times out, retain the verified durable output and post-apply
    checkpoint, report unresolved VQA, and return nonzero; never discard or
    misrepresent the completed material authoring.

## Authored-Appearance Evidence

The default evidence policy is clean-slate: old bindings, shader colors,
`primvars:displayColor`, indices, and opacity are unavailable.

When instructions explicitly promote authored appearance for named source
roots, record scoped permission in the frozen
`appearance_evidence_policy`. Within an approved scope, authored appearance is
weak supporting evidence only. Render source colors and candidate library
materials under the same OVRTX setup, rank measured appearance, and let the
agent decide from semantic class, role, geometry, names, and accepted
references. Never turn RGB values into a hard-coded material lookup.

For decomposed scenes, follow the evidence contract in
`content-workflow-asset-task-processing` and cite the frozen task-request
digest.

## Required Artifacts

Before writing canonical JSON, read `references/output-artifacts.md` and use
its exact field names and status vocabulary. Do not invent workflow-prefixed
schema names or aliases such as `assignment_groups`, `assigned`, or
`pass_with_limitations`; those are not accepted canonical fields.

When the structured task separates `child_required_artifacts` from
`wrapper_final_artifacts`, honor that ownership boundary exactly. The initial
child writes only its decision, scene outputs, and exact tool/trace evidence.
The fresh review turn returns only a structured semantic assessment, and the
wrapper writes the compact post-apply review with exact output/render bindings.
Neither model turn duplicates wrapper-owned assignments, VQA, receipts, render
manifests, operation counts, validation, or summaries.

### Controlled JSON artifact writes

For generated JSON, do not probe `apply_patch`: it is unavailable in controlled
child sessions. Construct the complete document with `jq -nS`, pass dynamic
values through `--arg`/`--argjson`, write a same-directory `.tmp`, then `mv` it
to the canonical artifact path and verify it with `jq -e .`. The wrapper
continues to validate schema and reproducibility after the child exits.

- launcher-owned `request.json` input (read it, but never create, replace, or edit it)
- material decision patch or equivalent edit record
- `assignments.json`
- `visual_quality_assessment.json`
- render records and final render PNGs
- operation counts
- normalized material palette/manifest evidence
- appearance-clear/audit records for every inspection and authoring
  derivative used as decision evidence
- `validation_evidence.json`
- saved derivative USD when durable output is claimed
- final summary
- trace/events where available
- restore/export artifacts when the workflow claims a source-space output
- `raw/material_binding_audit.json` when a durable USD is produced
- `raw/material_application_receipt.json` and
  `raw/material_post_apply_review.json` for the apply/review boundary

## Completion Gates

Visual approval and assignment coverage are separate requirements. Do not
report success merely because the renders look good.

- Every visible candidate must have an accepted material assignment or an
  explicitly preserved existing assignment. Missing, rejected, or ambiguous
  candidates keep the workflow incomplete.
- In clean-slate mode, derive candidate coverage after `appearance clear` has
  deinstanced native instance roots. Never report zero candidates merely
  because a pre-clear query returned only instance proxies.
- When the workflow claims a durable source-space result, request it with
  `--output-usd`. Open the composed output and require every active, visible,
  render-purpose mesh to resolve a direct or inherited material binding, or
  complete material-subset coverage of all faces.
- Record exact unbound and partially bound mesh paths in
  `raw/material_binding_audit.json`. Any such path is a failed delivery gate,
  not a warning-only result.
- A mesh whose material GeomSubsets cover every face is fully bound. Do not add
  a direct fallback binding merely because a low-level audit also lists the
  parent mesh as unbound; that extra binding can mask the subset decisions and
  is invalid unless it is an explicit accepted decision target.
- A partial restore or less than 100% final visible-mesh binding coverage must
  return a nonzero workflow status so callers can retry, repair, or move on.
- Publish the durable output only with `usd-cli save` to the exact requested
  path. A checkpoint is recovery state, not a deliverable: never copy or
  rename a `.usd`/`.usdc` checkpoint to a `.usda` output path because the file
  extension selects the USD encoding. If a render really fails, preserve the
  checkpoint and use ordinary `usd-cli` checkpoint/session recovery before
  saving.

## VQA Refinement

When VQA reports fixable issues:

- preserve the previous decision patch and renders;
- in clean-slate mode, never add `reviewed_no_override` to document an
  unfixable limitation; keep an already valid assignment unchanged and record
  the limitation only in VQA and trace artifacts;
- inspect only affected prims, views, and selected materials;
- apply a validated targeted patch through the selected backend;
- rerender affected views through OVRTX;
- use temporary dome-fill inspection lighting when low contrast prevents a
  confident diagnosis, without changing the accepted material solely to
  compensate for the inspection rig;
- refresh assignments, effective-binding audit, VQA, validation evidence, and
  trace;
- stop at the configured iteration cap or when remaining issues require new
  evidence, policy, or material-library granularity.

## Boundaries

- `materials.yaml` is a workflow contract, not a usd-cli-era artifact and not
  a usd-cli feature. Resolve it in the workflow layer and validate every
  proposed selection against its palette.
  Never ask usd-cli to parse the manifest or choose a material.
- The workflow selects and validates a material; usd-cli only applies
  the accepted USD-library binding.
- Treat source assets as immutable unless the user explicitly requests
  otherwise.
- A missing low-level backend primitive must not remove or redefine material
  selection, VQA, evidence, or artifact requirements.

## Observation Memory

When the frozen request enables `agent_memory`, use only the launcher-provided
`content-agent-memory` broker contract. Never open or modify the wrapper-owned
memory root directly.

Treat memory as a checkpoint, not a material selector:

- do not search before the initial material decision because current-run memory
  is empty;
- let the launcher checkpoint each finalized decision patch, VQA assessment,
  and representative final render;
- in a fresh VQA refinement turn, search once for the current asset before
  changing the patch, then inspect only the artifact needed for the active
  issue;
- use recalled evidence to avoid repeating an unchanged failed repair, while
  keeping current renders, references, candidate coverage, and material-library
  constraints authoritative;
- retry only when current evidence suggests a concrete change; otherwise mark
  the limitation or move on according to the normal refinement policy;
- do not duplicate routine launcher checkpoints or delay completion for memory.

When `agent_memory.enabled` is false, follow the same workflow without durable
recall and do not claim that memory influenced the result.

## References

- `references/material-policy.md`
- `references/output-artifacts.md`
- `references/iterative-vqa-refinement.md`
