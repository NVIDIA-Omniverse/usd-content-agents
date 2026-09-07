---
name: content-workflow-mesh-segmentation
description: Segment one fused triangular USD mesh into logical mesh prims by identifying visible parts, freezing deterministic geometry fragments, choosing per part between a registered image-mask seed produced through the image-generation skill and direct agentic signed fragment selection, and falsifying each provisional result with signed confuser evidence and held-out multi-view review before locking it. Use for fused-mesh semantic or material segmentation when Codex must control fallible image-generated masks and refine whole-fragment labels.
metadata:
  author: NVIDIA Omniverse
---

# Mesh Segmentation

Recognize the parts, generate one immutable fragment map, then finish exactly
one part at a time through direct visual decisions. Before `rev-000`, inspect a
small registered mask probe and choose either the deterministic eight-view mask
initializer or direct signed fragment selection. After `rev-000`, both routes
use the same fragment review and the agent is responsible for every refinement
decision.

This is a clean workflow. Do not inherit strategies from an older segmentation
skill. Reuse the existing scripts only as general-purpose rendering, fragment,
picking, editing, audit, and export infrastructure.

## Guardrails

- A provider retry can re-enter the same run directory with a fresh agent
  thread. Inspect existing state before creating anything. If a part has a
  locked `part_completion.json`, never regenerate its initializer decision,
  initializer validation, `rev-000`, or earlier evidence. Resume from the
  newest numbered state. Every accepted lock is face-immutable in every run
  mode; continue only from unlocked, provisional, or residual faces.
- Do not inspect golden, benchmark, or previously accepted segment labels during
  a run.
- Do not use annulus fitting, PCA, flood filling, category-specific predicates,
  remembered face IDs, fixed coordinates, or asset-specific scripts.
- Image-generation masks are optional, fallible initialization proposals. Use
  the `image-generation` skill for every generated mask. It selects an
  explicitly configured World Understanding backend or defaults to the coding
  agent's companion image-generation capability. If the selected mode is
  unavailable, record and report a warning and continue with direct agentic
  selection; never select another provider implicitly or fabricate mask
  evidence. When masks are available, a
  part-specific three-view probe must show that they survive erosion and
  project cleanly to fragments before the mask route is allowed.
- Record and validate `PART/initializer_decision.json` before creating
  `rev-000`. Part size or category alone never determines the route.
- If the mask route is chosen, do not vote over mask pixels, fragments, or
  views. Erode by exactly 2 px, select closest-visible fragments independently,
  and take the exact eight-view set union without visual alteration.
- If the direct route is chosen, discard mask-derived fragment IDs. Build
  `rev-000` only from focused neutral/ID evidence and visually confirmed signed
  fragment picks.
- Do not edit individual triangles. A pick resolves the closest visible face,
  but the include or exclude operation applies to its entire frozen fragment.
- Do not treat a script result as semantic validation. Only visual inspection
  determines whether a fragment belongs to a part.
- Batch obvious decisions. Do not serialize clicks that can be made from the
  same evidence.
- Treat a validated lock as immutable. Never extend, reassign, or otherwise
  change any of its faces in any run mode. Continue only from unlocked,
  provisional, or residual faces.
- Finish a part cleanly when practical. When a retry is unlikely to change the
  evidence, follow the recovery policy below instead of getting stuck.
- Treat every `rev-000` as provisional regardless of initializer route. Never
  let acceptance of an image-mask probe substitute for post-seed
  falsification.
- Before locking, collect signed positive target evidence and signed negative
  confuser evidence, then mechanically verify both against the exact candidate
  labels. A prose `passed` assertion is not validation.
- Challenge boundaries rather than sampling only easy interiors. For every
  expected instance, positive evidence must cover its interior, visible extent,
  and an opposing or occluded surface from at least two views, with a close
  signed pair straddling a visible target/confuser boundary. Every held-out view
  must contribute both polarities.
- Before acceptance, render the exact active segment by itself from every
  held-out view. Use selected-only evidence for both error directions: find
  embedded caps, internal plates, and floaters, but also reject truncated
  shells, discontinuous contours, implausible openings, and unexplained jagged
  cuts caused by missing faces. Compare repeated instances against one another
  so neither extra nor missing geometry can hide behind residual geometry.
- Connected-component count proves instance presence, not completeness. Before
  locking, review every expected instance from at least two selected-only views,
  including one opposing or occlusion-revealing view. Classify every visible
  selection boundary as an intended semantic boundary or a
  silhouette/occlusion; any unexplained boundary rejects the candidate.
- Expand any explicit count or repeated collection into separate stable
  `expected_instances` IDs before review (for example, `wing-1` through
  `wing-4`, not one entry named `four wings`). The number of
  `instance_completeness_reviews` entries must match that explicit count. If an
  instance is missing or unresolved, inspect and record the disposition of the
  selected components' `nearby_similar_scale_component_ids`; those hints are
  candidates to falsify, not semantic proof. An instance-count mismatch cannot
  pass: revise the candidate when evidence localizes the missing instance, or
  defer it when ambiguity remains.
- For each selected-only completeness view with a nonempty deterministic
  frontier, inspect at least one actual adjacent unselected fragment in the
  matching flat-label and ID evidence. A candidate cannot pass until every
  checked frontier fragment is confirmed non-target; include any confirmed
  target fragment and rerender. If the audit frontier is empty, record the
  complete-disconnected-component not-applicable decision instead.
- If one fragment visibly crosses a semantic boundary, do not split it with
  triangle edits. Before any part is locked, regenerate a finer fragment map.
  After locking begins, report the representational limit rather than silently
  changing the map.

## Prerequisites

The input is a USD stage containing one static triangulated fused mesh. Record
the target mesh prim path. Work in a fresh run directory; never overwrite a
revision. Use the workflow-owned usd-cli session and a fresh host Codex
session, not Docker.

Locate the canonical tool directory:

```bash
if test -d agentic/.agents/skills/content-workflow-mesh-segmentation/scripts; then
  MESH_TOOL_DIR=agentic/.agents/skills/content-workflow-mesh-segmentation/scripts
elif test -d .agents/skills/content-workflow-mesh-segmentation/scripts; then
  MESH_TOOL_DIR=.agents/skills/content-workflow-mesh-segmentation/scripts
else
  echo "mesh segmentation tools not found" >&2
  exit 1
fi
```

Use only the scripts named below. The canonical directory contains both the
general-purpose fragment tools and the evidence tools.

The launcher records the image-generation mode in `request.json` and requires
the `image-generation` skill. With an explicit World Understanding backend it
also records the model, optional base URL, and API-key environment-variable
name. Without a backend, the shared skill defaults to the coding agent's
companion image-generation capability and treats any model as a preference,
not an implicit provider. Read those values from the frozen request. Never
persist or print a key value.

## Use Observation Memory

Read `runtime.agent_memory` from the frozen request. When enabled, use
`content-agent-memory` with exactly its `run_id` and pass its `broker_url` via
`--broker-url`; do not open the wrapper-owned root, object store, locks, or
SQLite index directly. Memory is required-capture for that run.
When disabled, continue normally and report that durable recall is unavailable.

Current-run memory is empty before the first plan and before a part's first
attempt, so do not query it then. Search only when resuming or deliberately
revisiting the active part:

```bash
content-agent-memory --run-id RUN_ID --broker-url MEMORY_BROKER_URL search \
  --workflow mesh-segmentation --target "PART" --limit 4
```

Search a named confuser separately only when it is relevant. Do not load global
context or unrelated part cards. Inspect explicit observation IDs and artifact
roles only when their summaries affect the current decision.

Memory is a checkpoint, not a second decision-maker. Do not search memory before
every revision or lock, and do not reload unchanged observations. Memory capture
is post-decision bookkeeping: first create and review the same candidate you
would create with memory disabled. The launcher records an evidence-only
checkpoint as soon as that exact revision has selected-only and frontier
evidence; do not duplicate it or wait until the end. The checkpoint deliberately
does not claim a semantic result and cannot satisfy your required-capture duty.
The broker marks agent records separately from launcher checkpoints for terminal
audit. Final export is forbidden while any requested
part lacks selected-only evidence, a frontier audit, or its launcher checkpoint;
finish the missing review first. Record a materially changed diagnosis or
correction, a deferral, or a validated lock yourself; do not record routine
successful commands or repeat an unchanged backend warning.

A memory summary is not semantic evidence. Never select, reject, transfer, or
preserve geometry merely to agree with a previous observation. On revisit, use
memory to recover the prior candidate and unresolved question, then let current
signed and visual evidence decide whether to keep or change it.

Use absolute artifact paths. Put the exact semantic part in
`interaction.target_object_ids` and tags; `interaction` has no `semantic_part`
field. Chain changed decisions with `parent_observation_id`.

Use this minimal record shape; do not rediscover the schema during a run:

```json
{
  "workflow": "mesh-segmentation",
  "phase": "revision_review",
  "interaction": {
    "operation": "review_candidate",
    "target_object_ids": ["PART"],
    "target_prim_paths": ["/World/FusedMesh"]
  },
  "outcome": {"classification": "ambiguous", "summary": "EVIDENCE SUMMARY"},
  "expectation": {
    "expected_changes": ["COMPLETE TARGET SHELL"],
    "forbidden_changes": ["EXCLUDE NAMED CONFUSERS"]
  },
  "artifacts": [{
    "path": "/ABSOLUTE/RUN/PART/rev-NNN/face_labels.u32le",
    "role": "candidate_labels",
    "media_type": "application/octet-stream"
  }],
  "importance": "high",
  "tags": ["mesh-segmentation", "PART"]
}
```

Classify an initializer or unvalidated revision as `not_checked`, `ambiguous`,
or `contradicted`; never call it `matched` merely because routing or edit
validation passed. Use `matched` and pinning only after both false-positive and
false-negative validation passes. Preserve expected completeness as strongly
as forbidden leakage.

Memory must not reduce cameras, renders, signed evidence, revisions, or visual
inspection. It is advisory; dense labels and validations remain authoritative.
If immediate required capture fails, retry the capture once. If it still fails,
continue producing the best workflow artifacts, report the missing capture,
and do not claim that the run satisfied its memory contract.

## Recover, Defer, and Finish

Use ordinary operator judgment; do not turn recovery into another state
machine.

- Retry while a changed camera, evidence source, or edit still produces
  measurable progress: it covers a previously uncovered required positive,
  removes at least one signed mismatch, resolves an unexplained boundary, or
  adds a registered view that can decide a named ambiguity. Never repeat an
  attempt unchanged. Defer only after distinct changed attempts repeat the same
  defect without any of those gains.
- Make revisions evidence-monotonic. Replace the current best candidate only
  when new signed, registered, selected-only, or component-context evidence
  identifies a specific false positive or false negative and the proposed edit
  addresses it. An ambiguous reinterpretation is not new evidence: keep the
  earlier candidate unchanged, gather a genuinely discriminating view, or
  defer. Never swap one plausible semantic guess for another merely to try
  something different.
- Do not defer an unstarted part. For every requested part with inspectable
  evidence, create and visually review at least one real `rev-000` covering
  every inspectable expected instance and visible extent. A token seed does not
  count. Complexity, remaining time, or strict validation cost alone is not a
  deferral reason. A part proven invisible in all available views may remain
  residual with that limitation recorded.
- Treat every requested target as a positive assertion that the part should be
  found. Before calling one absent or uninspectable, audit every still-unlocked
  topology component, not only the components whose thumbnail or shape looks
  likely. Tiny assembled-view crops and semantic shape guesses cannot rule a
  component out. For each hidden, occluded, or too-small component that remains
  unclassified, render its isolated geometry from at least two fitted opposing
  views and compare those views with its registered source-context locator.
  Record every component's disposition. Deferral is allowed only when this
  exhaustive residual audit leaves no component or coherent component assembly
  consistent with the requested target.
- Only validated locks reserve faces. A deferred or provisional candidate must
  not consume geometry from later parts; preserve it separately and rebuild or
  reconcile it from the latest locked parent when revisited.
- A validated lock freezes both its evidence history and assigned faces in
  every run mode. If later evidence suggests that it is wrong, preserve the
  lock, record the conflict or unresolved boundary, and continue without
  changing its faces. This workflow does not advertise lock correction until a
  launcher-owned transactional validator can bind every affected revision to
  the final state.
- When the same problem remains, or the evidence says another retry is unlikely
  to help, preserve the best valid full-face labels, record the exact unresolved
  issue in memory, defer the part and continue with another part. A deferred
  candidate is provisional; do not promote it as a validated lock.
- Revisit deferred parts once, each after every other non-deferred requested
  part has been attempted. The remaining geometry and accumulated observations
  often make the boundary clearer. If one is still unresolved, keep the best
  provisional assignment and report the uncertainty.
- A zero unselected frontier is legitimate when the selected target is already
  a complete disconnected component. Record the frontier check as not
  applicable. Do not invent adjacent geometry or borrow faces merely to satisfy
  a checklist.
- Never rewrite the frozen fragment artifacts during semantic work. If their
  integrity changes, stop label edits and reproduce the deterministic fragment
  map from the source once. Stop the run only when a clean retry cannot restore
  face provenance, label count, fragment atomicity, or the source mesh.
- Always export the best complete face partition, even when some semantic parts
  remain deferred. Preserve one label per source face, use the documented
  residual for genuinely unresolved faces, and produce the final USD, labels,
  manifest, renders, and report. Do not call a deferred part validated or
  locked.

## 1. Identify the Parts

Prepare a neutral copy and render the source and frozen fragments from all eight
cube-corner cameras. The script defaults to these eight views; do not reduce the
set when memory is enabled. Reuse the saved cameras for registered ID evidence:

```bash
python "$MESH_TOOL_DIR/prepare_mesh.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --output-dir RUN/prepare

python "$MESH_TOOL_DIR/render_mesh_evidence.py" \
  --scene INPUT.usd \
  --focus /World/FusedMesh \
  --camera-json CAMERA_A.json \
  --camera-json CAMERA_B.json \
  --output-dir RUN/part-identification
```

Inspect the source references and neutral renders. Write `RUN/part_plan.json`
with:

- a stable segment ID and semantic name for each visible logical/material part;
- separate stable IDs for repeated instances expected to share that label; if
  the description gives an explicit count, write exactly that many entries and
  never collapse the collection into one phrase such as `four wings`;
- nearby parts likely to be confused with it;
- a short visual description that can be checked in renders.

When `request.json` provides non-empty `inputs.target_semantic_parts`, treat
that list as the complete user vocabulary and skip open-ended part
identification. Use exactly those strings as `semantic_name` values; do not
add, rename, split, or merge non-residual categories. Inspect the object only
to record expected instances, confusers, visual descriptions, and a safe
processing order with broad provided body categories last. Use `other` only as
the final residual for genuinely unassigned faces.

For targeted runs, create each part directory directly under `RUN` using the
exact case-sensitive vocabulary string, for example `RUN/Base` and
`RUN/Fan Blades`. Do not lowercase, slug, number, or nest these directories.
Vocabulary strings may contain spaces. Set `PART` to the exact string and quote
every shell path containing it, for example
`--clicks "RUN/$PART/rev-NNN/clicks.json"`. Record the same string in every
`semantic_part` field.

Do not segment yet. The inventory may be corrected when new views reveal an
unstarted part, but do not redefine a completed part to hide an error. Reserve
broad residual categories such as `body` or `other` until the more specific
parts are complete.

## 2. Generate and Freeze Fragments

Generate deterministic Python/SciPy over-segmentation once:

```bash
python "$MESH_TOOL_DIR/oversegment_mesh.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --output-dir RUN/fragments
```

Render `RUN/fragments/fragments.usdc` from the same cameras as the neutral
source. Check that the fragment coloring is registered to the source and that
fragments are small enough to represent visible semantic boundaries. More
fragments are acceptable; premature merging is not.

Freeze these artifacts for the full run:

- `fragment_ids.u32le`
- `fragment_adjacency.npy`
- `fragments.usdc`
- `fragment_manifest.json`

Create `RUN/state/labels-000.u32le` with one background segment ID (`0`) for
each source face. This is the parent state for the first part. Every later state
must preserve one label per source face and keep each fragment atomic.
State history is append-only. Once a `state/labels-NNN.u32le` file exists, never
rewrite it, especially after an initializer gate has sealed it. For any later
correction, use the newest state as the revision parent, write the result to a
new `rev-NNN`, then promote it to a new monotonically numbered state file. Do
not rebuild or reapply unchanged later parts. A cross-label transfer is two
explicit append-only revisions: exclude from the source part, then include in
the destination from the resulting newest state.
Treat `state/final_labels.u32le` as the terminal promotion marker, not a rolling
working alias. During construction, use the newest numbered state. Never create,
replace, rename, or write `final_labels.u32le`. After every requested part has a
complete validated lock and the bounded uncertainty pass is complete, end the
turn. The launcher validates the exact lock-bound state and promotes it. A
continuation reads `raw/final_labels_promotion.json` and uses its lock-bound
`source_labels` numbered state for final export work; it must not access or
change the launcher-owned terminal marker.

The launcher prompt binds every executable helper to an absolute checked-in
path outside the writable run. Use those exact paths. Never execute a helper
copy under `RUN/scripts/` or `RUN/.agents/`; those copies are child-writable and
are not trusted execution inputs.

Render zero-label face-ID buffers for all eight saved neutral cameras in one
`render_face_id_buffers.py` call, then build the reusable component evidence:

The neutral render manifest, ID buffers, and component builder must all bind to
the exact immutable `INPUT.usd`, not `prepare/neutral.usdc`. The prepared stage
is useful diagnostic output, but its file digest is intentionally different.
Select camera JSON paths from `.renders[].camera` in the render manifest; do
not glob every JSON file in the render directory because response JSON files
are not cameras.

```bash
python "$MESH_TOOL_DIR/build_topology_component_evidence.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --fragment-labels RUN/fragments/fragment_ids.u32le \
  --id-buffer-manifest RUN/topology-id-buffers/manifest.json \
  --neutral-render-manifest RUN/part-identification/render_manifest.json \
  --appearance-dir RUN/inputs/references \
  --output-dir RUN/topology-components
```

Visible cards preserve registered source appearance in a close crop, a wider
local crop, and a readable full-view locator. Each card also lists similarly
scaled components whose 3D bounds nearly touch; treat that list only as a prompt
to inspect possible assembly peers, never as proof that they share a semantic
label. Components hidden in every assembled OVRTX view get an isolated-geometry
card with a diagnostic-only local preview and global-position locator. Never use
that PIL preview to make a semantic or inspectability decision. Obtain a focused
shared OVRTX render before declaring an expected part uninspectable. Use this
corpus only for the active part:
inspect likely registered cards and its named confusers. Do not assign the rest
of the asset speculatively. In particular, do not make a global
component-to-semantics assignment before part work; that turns an early visual
guess into a sticky taxonomy and makes memory a second decision-maker.

If the likely-card pass does not locate a requested target, switch from likely
triage to an exhaustive residual audit before deferring it. Inspect every
still-unlocked component and classify it in a target-specific audit artifact.
An assembled-view thumbnail can be too small or occluded to reveal an interior
part, so render isolated geometry from at least two fitted opposing views for
each unresolved component and compare it with the registered source-context
locator. Do not let an early semantic guess about a component's shape stand in
for this evidence.

Before starting the first semantic revision, use the component corpus to
challenge the plan's assembly granularity. Topology is a clue, not semantic
proof. Prefer the coarsest logical-assembly partition consistent with the
request and registered evidence; a semantic part is not every surface that can
be named. If the request supplies target names but no annotated boundary or
explicit part description, do not infer a fine cut inside a welded component
from the name alone. Surface orientation, a narrow visible band, and even a
local seam are supporting clues, not sufficient evidence. Split the component
only when registered multi-view evidence establishes a repeatable closed
boundary around a coherent 3D subassembly, or the user explicitly defines that
boundary. Otherwise revise the still-unstarted plan at the logical-assembly
level and keep the component together.

## 3. Segment One Part

Choose one unfinished part from `part_plan.json`. Give it its recorded nonzero
segment ID. Already completed segment IDs in the parent labels are locked.

### Choose the initializer and create `rev-000`

Read [initializer-routing.md](references/initializer-routing.md) completely
before initializing the active part.

Resolve image generation through the `image-generation` skill as specified by
the routing reference. When it is available, create the three-view image-mask
probe, preserve each `agentic-image-generation-result.v1` manifest, and inspect
the registered masks, 2 px-eroded pixels, and resulting whole-fragment
projections. When it is unavailable, record the required warning and use the
direct route without fabricating a probe. Then write and validate:

- `RUN/PART/initializer_decision.json`
- `RUN/PART/initializer_decision_validation.json`

Create the validation artifact by running
the launcher-provided absolute `validate_initializer_decision.py` path against
the decision before recording
initializer memory or creating `rev-000`. The launcher-owned
`RUN/PART/initializer_runtime_gate.json` is watchdog state: it never substitutes
for the decision validation and must not be cited as if it did.
After validation passes, treat the decision and every cited evidence artifact as
immutable. If any of them is rewritten, delete the now-stale validation, rerun
the validator, and wait for the runtime gate to accept the refreshed corpus
before continuing. Do not start or review `rev-000` against a stale digest.

Choose exactly one route:

- **`image_mask_seed`**: reuse the accepted probe views, generate the remaining
  five fixed cube-corner views, and apply the exact deterministic eight-view
  fragment union unchanged as `rev-000`.
- **`direct_agentic_selection`**: discard every mask-derived fragment ID, add
  focused views, batch closest-hit positive and negative fragment picks, inspect
  each whole fragment, and apply only confirmed direct selections as `rev-000`.

Do not blend routes within one `rev-000`. Both routes must preserve the same
immutable parent labels and frozen fragment map.

### Observe

Read [falsification-and-locking.md](references/falsification-and-locking.md)
completely before reviewing `rev-000`. Write the falsification plan before
looking for reasons to accept the initializer. Predeclare expected instances,
nearby confusers, observable falsifiers, construction views, and held-out
review views.

Render a small set of informative views of:

1. the neutral source;
2. the frozen fragment colors;
3. a flat, unlit binary label view, with the active part highlighted and every
   unselected visible face neutral;
4. the closest-visible face and fragment ID buffers.
5. a selected-only export of the exact candidate, with every other segment
   hidden.

Use matching saved cameras so evidence can be compared directly. Add closer,
opposing, top, bottom, or occlusion-revealing views whenever the existing views
leave uncertainty. Normal and depth renders may clarify geometry, but they do
not decide semantics. Lit renders provide semantic context only; never infer
label membership from lighting or shadow.

Generate all registered ID and label channels from the same saved camera:

```bash
python "$MESH_TOOL_DIR/render_face_id_buffers.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --fragment-labels RUN/fragments/fragment_ids.u32le \
  --face-labels "RUN/$PART/rev-NNN/face_labels.u32le" \
  --active-segment-id PART_ID \
  --camera-json RUN/cameras/right.json \
  --output-dir "RUN/$PART/rev-NNN/id-buffers"
```

### Batch-pick and decide

Create a click batch against the rendered evidence:

```json
{
  "events": [
    {
      "camera": "RUN/cameras/right.json",
      "view_id": "right",
      "instance_id": "target-part",
      "pixel": [320, 240],
      "polarity": "positive",
      "note": "interior of a fragment visibly belonging to the target"
    }
  ]
}
```

Pick several unmistakable fragment interiors in one call:

```bash
python "$MESH_TOOL_DIR/pick_face_evidence.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --fragment-labels RUN/fragments/fragment_ids.u32le \
  --clicks "RUN/$PART/rev-NNN/clicks.json" \
  --output "RUN/$PART/rev-NNN/picks.json"
```

Use the closest visible hit only. Avoid silhouettes and fragment boundaries.
Deduplicate returned fragment IDs.

For a broad or thin visible mistake, mark a batch polygon or scribble over the
same-camera ID buffer instead of issuing thousands of point rays:

```bash
python "$MESH_TOOL_DIR/regions_to_fragment_evidence.py" \
  --id-buffer-manifest "RUN/$PART/rev-NNN/id-buffers/manifest.json" \
  --regions "RUN/$PART/rev-NNN/regions.json" \
  --output "RUN/$PART/rev-NNN/region-fragments.json"
```

The mapping is deterministic and returns the unique closest-visible fragment
IDs under each marked region. Region evidence is still a proposal: inspect each
whole fragment before applying it.

Classify each picked fragment by direct inspection:

- **include** when the whole visible fragment belongs to the target part;
- **exclude** when a selected fragment belongs to another part;
- **undecided** when current evidence is insufficient.

Never guess an undecided fragment. Render another view of it. Write one batch
of supported decisions:

```json
{
  "edits": [
    {
      "operation": "include",
      "fragment_ids": [12, 34],
      "reason": "visible target-part surfaces in two views",
      "evidence_render_ids": ["right", "upper-right"]
    },
    {
      "operation": "exclude",
      "fragment_ids": [56],
      "reason": "visible neighboring part",
      "evidence_render_ids": ["front-close"]
    }
  ]
}
```

Apply the batch as a new immutable revision:

```bash
python "$MESH_TOOL_DIR/apply_fragment_edits.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --fragment-labels RUN/fragments/fragment_ids.u32le \
  --parent-labels "RUN/$PART/rev-NNN/parent_labels.u32le" \
  --edits "RUN/$PART/rev-NNN/edits.json" \
  --active-segment-id PART_ID \
  --output-dir "RUN/$PART/rev-NEXT"
```

For `rev-000`, pass `RUN/PART/rev-000` as the actual script output directory so
the generated `face_labels.u32le`, `diagnostic.usdc`, and `edit_manifest.json`
all live there. Put the edit request outside that output directory when needed.
Do not create `RUN/PART/rev-000` first: `apply_fragment_edits.py` owns that
directory and refuses to overwrite or populate a pre-existing revision.
Never output to `rev-000-applied` and copy selected files afterward.

### Inspect and repeat

Render the new diagnostic USD with the same cameras. Conduct two explicit,
separate searches in every review.

**False-positive search**

- Find highlighted pixels that the neutral or reference evidence shows belong
  to a neighboring part.
- Map the region through the same-camera ID buffer and inspect the returned
  whole fragments.
- Exclude only confirmed non-target fragments.

**False-negative search**

- Find target pixels that remain neutral in the flat label view.
- Map the region through the same-camera ID buffer and inspect the returned
  whole fragments.
- Include only confirmed target fragments.

Also inspect missing repeated instances, leaks at contact boundaries, isolated
selected fragments, and hidden-side mistakes. If a decision is unclear, change
the view before changing labels.

Repeat:

```text
render current state
  -> batch-pick visible mistakes or missing fragments
  -> include / exclude / retain undecided
  -> create a new revision
  -> render again
```

Do not replace this loop with a geometry hypothesis or automatic semantic
expansion. Large batches are fine when every decision is visually grounded;
small batches are fine near difficult boundaries.

Before declaring the current revision clean, export its full partition, then
isolate the active segment:

Write the candidate's segment metadata from the segment IDs actually present
in that revision's `face_labels.u32le`. Do not pass future labels or residual
label `0` after it has no faces: `export_segmented_usd.py` rejects every
configured segment with zero faces. Keep this revision-local segment JSON next
to the selected-only evidence at
`RUN/$PART/rev-NNN/selected-only/segments.json`.

```bash
python "$MESH_TOOL_DIR/export_segmented_usd.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --fragment-labels RUN/fragments/fragment_ids.u32le \
  --face-labels "RUN/$PART/rev-NNN/face_labels.u32le" \
  --segments "RUN/$PART/rev-NNN/selected-only/segments.json" \
  --output-usd "RUN/$PART/rev-NNN/selected-only/segmented.usdc" \
  --manifest "RUN/$PART/rev-NNN/selected-only/export_manifest.json"

python "$MESH_TOOL_DIR/build_selected_only_stage.py" \
  --source-usd "RUN/$PART/rev-NNN/selected-only/segmented.usdc" \
  --target-prim /World/SegmentedAsset/Segments/PART \
  --output-usd "RUN/$PART/rev-NNN/selected-only/part-only.usdc" \
  --manifest "RUN/$PART/rev-NNN/selected-only/stage_manifest.json"
```

Render that revision-bound `part-only.usdc` from every held-out camera into
`RUN/$PART/rev-NNN/selected-only/renders`. Keep the renders beside the revision
they validate: the render manifest must cite the exact `part-only.usdc` in that
same `rev-NNN/selected-only` directory so the launcher can checkpoint the
reviewed candidate without overwriting an earlier lock's evidence.
For repeated instances,
compare silhouettes, shells, holes/openings, and visible internal surfaces.
Inspect the isolated part as an object that must look semantically complete by
itself. Any truncated wall, discontinuous contour, unexplained jagged opening,
one-sided structure, or subset-only structure rejects the candidate. Inspect
the corresponding selected and adjacent unselected frontier fragments in the
matching flat-label and ID views, then include or exclude only confirmed whole
fragments.

For the current candidate, maintain one signed falsification evidence set:

- positive clicks covering `target_interior`, `target_extent`, and
  `opposing_or_occluded_surface` for every expected instance from at least two
  views;
- negative clicks tagged with every visible named confuser;
- one close positive/negative boundary pair per expected instance, tagged with
  the same `boundary_pair_id`.

Resolve these clicks independently of image-generation masks. Run
`compare_face_labels.py` against the immutable state from before the active
part. A selected negative face or omitted positive face rejects the candidate
even when the rendered result looks broadly plausible. Repair it and rerun the
comparison.

Apply the recovery policy after a failed review: use changed evidence or method
while it produces measurable progress; otherwise defer the part and continue.
Preserve every rejected revision. If the review is merely ambiguous and does
not localize a false positive or false negative, keep the earlier candidate
unchanged instead of editing from a new semantic guess.

## 4. Review, Lock, and Move On

When memory is enabled, let the launcher checkpoint the exact candidate as soon
as its deterministic selected-only and frontier evidence exists. This
checkpoint makes no semantic claim. If the part is revisited, search this
checkpoint before gathering the new discriminating evidence. Record a changed
semantic diagnosis or correction after deciding it. This is the point of
memory: resume the prior decision without redoing it, while the new evidence—not
the summary—still decides the correction.

When an inspection pass finds no issue, perform a separate falsification and
completion review:

- inspect all sides and any relevant underside or occluded region;
- inspect close views of every visible semantic boundary;
- compare neutral, flat binary label, fragment-color, face-ID, and fragment-ID
  evidence from matching cameras;
- inspect the selected-only candidate from every held-out view and compare all
  repeated instances for both inconsistent extra geometry and incomplete
  shells, discontinuous contours, or unexplained openings;
- record one structured `instance_completeness_reviews` entry for every
  expected instance, with at least two visible selected-only views spanning
  `primary_surface` and `opposing_or_occluded_surface`;
- reject a collective phrase used as one `instance_id`; an explicit expected
  count must have the same number of separate completeness entries;
- for every missing or unresolved instance, inspect each selected component's
  `nearby_similar_scale_component_ids` and record why every hinted component is
  target or non-target before accepting, revising, or deferring;
- for every completeness view, classify the visible selected-only boundaries,
  require `coherent_semantic_shell`, `outer_contour_continuous`, and
  `semantic_openings_plausible`, and record zero `unexplained_boundaries`;
- when the exact frontier audit contains unselected fragments, check at least
  one of them per completeness view and record `confirmed_non_target`; when the
  audit is empty because the target is a complete disconnected component,
  record an empty checked-fragment list and
  `not_applicable_complete_disconnected_component` instead;
- explicitly record that both the false-positive and false-negative searches
  found no actionable issue;
- check every expected repeated instance;
- include at least one useful view that was not used to make the last edit;
- verify every predeclared falsifier and named confuser;
- verify that every held-out view contributed accepted positive and negative
  evidence and that boundary pairs challenge the candidate locally rather than
  merely sampling distant interiors;
- rerun signed evidence against the exact latest candidate labels;
- run the deterministic frontier audit:

```bash
python "$MESH_TOOL_DIR/audit_selection_frontier.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --fragment-labels RUN/fragments/fragment_ids.u32le \
  --face-labels "RUN/$PART/rev-FINAL/face_labels.u32le" \
  --active-segment-id PART_ID \
  --output-dir "RUN/$PART/review"
```

The audit locates boundaries; it does not approve them. If review or signed
evidence exposes one coherent mistake, write a diagnosis and return to the same
render-and-pick loop. Continue with changed evidence while coverage or leakage
still improves. If distinct changed attempts repeat the same defect without
progress, defer the part rather than blocking the rest of the asset. Preserve
the rejected revision.

Lock the part only after one complete review following the last edit produces
no further edit and the launcher-provided absolute
`validate_falsification_review.py` path passes. Preserve:

- `RUN/PART/falsification_plan.json`;
- `RUN/PART/falsification_evidence.json`;
- `RUN/PART/falsification_review.json`;
- `RUN/PART/falsification_validation.json`.
- `RUN/PART/rev-FINAL/revision_consistency_regions.json`;
- `RUN/PART/rev-FINAL/revision-consistency/manifest.json` and every generated
  crop, comparison sheet, and context sheet;
- `RUN/PART/rev-FINAL/revision_consistency_review.json`.

Never hand-write `falsification_validation.json`, copy one from another part,
or edit `falsification_review.json` after validation. The validator binds the
review digest and exact evidence digests. If the review or any bound artifact
changes, delete the stale validation and run the validator again. Immediately
after it passes, write `part_completion.json`, record the validated-lock memory
observation when memory is enabled, and only then promote the part labels to the
next numbered state. Record every requested part's validated lock, finish the
bounded uncertainty pass, and end the turn. The launcher validates the complete
lock corpus and owns creation of `state/final_labels.u32le`; its first promotion
closes the timely-memory window.

Record `RUN/PART/part_completion.json` with the same final revision, both
falsification artifact paths, the validation digest, views inspected, expected
instances checked, remaining blind spots, and the agent's confidence rationale.
The record must list the ID-buffer manifests and state
`false_positive_review: passed` and `false_negative_review: passed`. These
strings are summaries; they have no authority without the passing validation
artifact. Promote its face labels to the parent state for the next part. Locked
assignments remain face-immutable in every run mode.

After a validated lock, begin the next part. If validation cannot pass after the
recovery policy, preserve the candidate as provisional and begin another part
without claiming a lock.

## 5. Resolve Residuals and Export

After all specific parts have either locked or completed their deferred revisit,
inspect the remaining background fragments. Segment a broad `body` part through
the same visual loop. Assign genuinely ambiguous leftovers to a documented
`other` part only after inspection; do not silently absorb them into the nearest
segment.

Before export, make one bounded uncertainty pass. Revisit at most the three
weakest candidates: prefer an instance-count mismatch, a multi-piece assembly
whose nearby peers remain in another segment or the residual, or a candidate
whose selected-only evidence did not resolve its planned confuser. With memory
enabled, search that exact part's checkpoint and inspect its candidate labels
and frontier artifact before gathering new evidence or creating any revisit
revision; the memory search must be the first action of the revisit. With memory
disabled, read the equivalent local candidate artifacts. For a cross-label
removal or transfer, search both affected parts first and require a
source-context view that shows the component's assembled attachment or
appearance; selected-only geometry alone is insufficient. If source context
remains ambiguous, keep the candidate. Create at most one new revision per
revisited part in this pass, and only when the comparison localizes a false
positive or false negative. Otherwise keep the candidate and move on.
Never implement a revisit by rewriting the original sequential
`state/labels-*` chain; append only the changed part revisions to the newest
state.
This pass is a human-style retry, not a mandate to churn every provisional
segment.

Give each segment an intentional diagnostic `color` in `segments.json` when a
specific palette matters. If a color is omitted, the exporter assigns a stable,
distinct fallback color by segment ID so final renders still expose the
partition instead of presenting every segment in the same gray.

Residual is not a shortcut. Before export, revisit every visible named part
still absorbed by `body` or `other`, especially repeated or occluded topology
components. A bookkeeping-only revisit is insufficient when a changed view,
component proposal, or fragment edit can still expand an incomplete candidate.

Finalize the root `RUN/segments.json` from the segment IDs actually present in
`RUN/state/final_labels.u32le`, including residual label `0` only when it still
has faces. Unlike the revision-local candidate exports above, the final export
intentionally consumes this root manifest because it describes the complete
final partition. Every configured ID must have faces and every observed ID must
be configured; prune superseded or emptied IDs before invoking the exporter.
Then export:

```bash
python "$MESH_TOOL_DIR/export_segmented_usd.py" \
  --source-usd INPUT.usd \
  --target /World/FusedMesh \
  --fragment-labels RUN/fragments/fragment_ids.u32le \
  --face-labels RUN/state/final_labels.u32le \
  --segments RUN/segments.json \
  --output-usd RUN/final/segmented.usdc \
  --manifest RUN/final/export_manifest.json
```

The final report must list the identified parts, revision count per part,
locked fragment and face counts, views used for each completion review, every
deferred or provisional part, and any unresolved fragment-boundary limitation.
Never claim validation for a part that could not be visually inspected.

## Run Directory Layout

Read [run-layout.md](references/run-layout.md) before writing any run artifact.
Every consumer of a run -- the launcher's gates and terminal validation --
locates evidence by exact path, so an invented layout makes correct work
unreadable and fails the run. In particular,
`PART/rev-NNN/selected-only/renders/` must hold renders of that exact revision's
part alone, not a whole-object render with the target tinted.

## Required Run Artifacts

Preserve:

- every exact path in the frozen request's `required_final_artifacts` array,
  including `state/final_labels.u32le`, `final/segmented.usdc`,
  `final/export_manifest.json`, `final/renders/render_manifest.json`, and
  `final/report.md`;
- recognition-mode queue and lock evidence (`part_work_queue.json`,
  `part_lock_manifest.json`, `live_sequential_gate.json`, and
  `part_review_manifest.json`) or targeted-mode evidence
  (`final/target_evidence.json` and its per-target falsification bindings), as
  applicable;
- the part plan and source references;
- neutral renders, saved cameras, and frozen fragment artifacts;
- for every part: `initializer_decision.json` and its validation, plus either
  the raw mask-probe results, prompts, affine transforms, registration metrics,
  closest-visible ID buffers, eroded pixel overlays, and fragment projections,
  or a validated `image_generation_warning.json`;
- for a mask-routed part: the remaining five view artifacts, exact eight-view
  union manifest, replay validation, and unmodified `rev-000` edit;
- for a direct-routed part: focused neutral/ID views, signed clicks, resolved
  fragment evidence, and the direct `rev-000` edit;
- for every part and revision: renders, clicks, picked IDs, edit batch,
  region marks, ID-buffer mappings, diagnostic USD, face labels, and edit
  manifest;
- for every part: falsification plan, signed positive/negative evidence,
  latest-candidate label comparison, held-out falsifier evidence, falsification
  review, selected-only export/stage/renders, and deterministic falsification
  validation;
- for every part: final-revision consistency regions, the
  `build_revision_consistency_sheet.py` output manifest and all referenced
  crop/context images, and the digest-bound `revision_consistency_review.json`;
- per-part completion reviews and immutable parent states;
- final segment metadata, segmented USD, export manifest, and report.

These artifacts must make every semantic fragment decision replayable and
auditable.

## Reproducible Fresh-Session Launch

Run the canonical workflow from an isolated host Codex session through the CLI.
Always pass the Codex Responses endpoint and its environment-variable name.
Optionally pass an explicit image-editing multimodal base URL and its provider
settings. The launcher normalizes the Codex Responses URL for the SDK and
records the selected image-generation mode in `request.json`.

```bash
content-workflow-cli mesh-segmentation run \
  --workflow-skill content-workflow-mesh-segmentation \
  --asset INPUT.usd \
  --reference-image REFERENCE.png \
  --target-semantic-part tire \
  --runner codex \
  --model provider/coding-agent-model \
  --model-reasoning-effort high \
  --codex-responses-url https://provider.example/v1/responses \
  --codex-api-key-env MESH_SEGMENTATION_API_KEY \
  --codex-execution-mode host \
  --image-gen-backend openai_compatible \
  --image-gen-model provider/image-edit-model \
  --image-gen-base-url https://provider.example/v1 \
  --image-gen-api-key-env MESH_SEGMENTATION_API_KEY
```

`--image-gen-model` and `--image-gen-base-url` select the explicit external
provider path in this example. Do not treat the shown Gemini model and NVIDIA
endpoint as workflow defaults. If `--image-gen-base-url` is omitted, the child
first tries its companion image-generation capability. If that is absent or
unusable, the run continues with direct agentic initialization and reports a
warning.

Never add `--codex-execution-mode container` and never start Docker for this
workflow.
