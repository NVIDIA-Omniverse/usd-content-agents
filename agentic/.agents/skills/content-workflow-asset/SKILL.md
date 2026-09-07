---
name: content-workflow-asset
description: >-
  Run one prompt-specific asset workflow through a sole outer reasoner that
  supplies and freezes a typed public-leaf dependency graph. Use for selective,
  resumable asset composition without a built-in domain or stage order. Explicit
  compatibility-fixed mode still composes Geometry preparation, articulation
  review, material assignment, bounded texture
  generation, physics authoring, validation, and final USDZ packaging as one
  durable single-asset workflow.
version: "0.3.5"
author: NVIDIA Omniverse
tags:
  - content-agents
  - asset
  - composition
  - workflow
tools:
  - Shell
  - Filesystem
  - Python
compatibility: >-
  Requires Python >=3.12 and content-workflow-cli. Each outer-selected opaque
  leaf retains its own prerequisites and native terminal contract. Fixed
  compatibility mode also requires deterministic source/render/usd-cli
  evidence, a material library, and the selected generation, proposal, render,
  and simulation backends.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - asset
    - composition
    - workflow
---

# Agentic Asset Composition

Use this skill from a repository-root call to
`content_workflow_cli.asset_runner.run_interactive_asset_workflow` or
`resume_interactive_asset_workflow`, or from `content-workflow-cli asset run`
and `asset resume`. The public launchers expose one `AssetCoordinatorSession`
to exactly one outer reasoner for one prompt while owning the sole parent
usd-cli lifecycle. There is no selector, domain planner, nested coordinator,
provider-owned controller, or built-in total domain/stage order.

## Public Agentic Contract

- New public runs are `selected_mode=agentic` and request schema
  `content-agents.asset-composition-request.v3`. They bind the staged source,
  prompt/source/config/reference digests, the deterministic repository-owned
  `content-agents.asset-leaf-catalog.v3`, and one durable
  `asset-sole-coordinator-identity.v1`. Never author or substitute run-local
  descriptors; `--leaf-catalog` accepts only an exact repository-catalog
  readback.
- The sole outer reasoner interprets the prompt and supplies one immutable
  `content-agents.asset-execution-graph.v2`. Use stable catalog leaf IDs only.
  List the prompt-specific selected set, every explicit dependency,
  required/optional requirement, terminal-output leaves, and every omitted ID.
  Nodes, dependency IDs, and omissions use canonical lexical storage only; that
  storage order is never execution policy.
- Bind each selected node to its catalog descriptor digest. Static catalog
  requirements must appear in `depends_on`; incompatible selections are invalid.
  A descriptor's `required_dependents` are conditional producer constraints: if
  both leaves are selected, each named dependent node must include the descriptor
  leaf in `depends_on`. Omitting the dependent imposes no global order.
  Unknown, missing, duplicate, cyclic, incompatible, mixed-mode, or stale graph
  identities fail closed.
- State validates and follows the frozen graph. It never chooses, inserts,
  removes, or reorders a leaf. No agentic transition can enter the historical
  `STAGE_ORDER`, fixed pipeline, classic mode, or a fallback path.
- Omitted catalog leaves are durably `not_requested`. A selected required leaf
  must pass. A selected optional leaf may be `not_evaluated` only with native
  result/evidence and only when no selected dependent requires it.
- If `articulation.proposal-provider.v1` is selected, its predecessor
  preparation must use `proposal_status=not_evaluated` with no proposal. Use
  `proposal_status=not_requested` only when that proposal leaf is omitted.
- Every leaf receipt binds the graph/catalog/coordinator identities, dependency
  receipts, invocation, native result and terminal receipt, operation/evidence
  indexes, evidence, saved-stage readbacks, native disposition, timing, resource
  claims/releases, and supersession. A failed or cancelled opaque result stays
  failed or cancelled with those exact artifacts; composition must never upgrade
  it to `passed` or `not_evaluated`. The graph terminal receipt binds all selected
  receipts, all omissions, terminal artifacts, total timing, and parent/leaf
  resource release.

Discover the capability catalog with `content-workflow-cli asset catalog` and
use the identical frozen catalog in `request.json`. Each descriptor resolves to
repository-owned invocation/result schemas and a declared deterministic
projector; catalog discovery advertises capabilities but never selects or
orders them. Create the prompt-specific `AssetExecutionNode` list and
`AssetExecutionGraph.create(...)` inside the sole reasoning invocation. The
factory canonicalizes storage lexically; only explicit dependency edges govern
readiness.

## Agentic Execution

1. Read and verify the frozen `request.json` and `asset_run.json`. Never edit
   either. Confirm the request coordinator identity matches this invocation and
   `next_action` is `freeze_graph` for a fresh run. A public launcher session
   also supplies the immutable typed `ParentUsdCliSessionIdentity` and its exact
   artifact binding. Reuse that parent session; never start, replace, or stop a
   daemon from the reasoning loop.
2. Inspect the prompt, references, staged-source identity, and frozen public
   catalog. Read back `content-workflow-cli asset catalog` and require its exact
   descriptor and catalog digests to match the request. Author the exact graph
   JSON below the run root. Include every catalog ID in exactly one of
   `selected_leaf_ids` or `omitted_leaf_ids`; freeze it once with `freeze-graph`.
   Absence of a graph is not permission to use a fixed order. When the request
   carries `required_leaf_ids`, select every one as `required`; mark exactly
   `required_terminal_leaf_ids` as terminal outputs; and include every edge in
   `required_leaf_dependencies`. Missing catalog capabilities or any partial
   graph contract must fail before provider, renderer, authoring, or runtime
   work begins.
3. When `next_action=begin_leaf`, select only a dependency-ready leaf already
   frozen in the graph, call `begin-leaf`, then
   `leaf-dir`; put every invocation/result/evidence/readback/release file below
   that attempt directory. On managed Windows, invoke the descriptor through
   `content-workflow-asset-state invoke-leaf` with the exact run state, active
   leaf ID, absolute invocation path, and a new absolute result JSON path in the
   attempt directory. Do not invoke a domain executable or `usd-cli-tel`
   directly. Use the launcher-owned readiness artifact, command
   journal/checkpoint, staged source, and exact session identity for retained
   pre-native preparation facts; the trusted domain leaf owns authenticated
   scene commands. Other hosts invoke the descriptor's opaque public entrypoint
   directly. The entrypoint
   retains its own native validation and cannot have a
   conditional or failed disposition upgraded by composition state.
   For every invocation artifact inherited from a predecessor leaf, copy its
   absolute `path`, `sha256`, and `size_bytes` verbatim from that predecessor's
   verified `leaf_receipt.json`; never reconstruct, abbreviate, or relocate the
   path. Before calling the opaque entrypoint, require that each copied path is
   an existing regular file below the declared run directory and rehashes to the
   copied identity. Correct an invocation-construction error before the native
   call; after the native call, preserve any failed result exactly.
   When `articulation.preparation-publisher.v1` feeds the standalone
   `articulation.author.v1` leaf, require its retained inspector configuration
   to set `capabilities.canonical_output_evidence_required=true` before
   publication. That capability declares the mandatory post-author evidence
   policy; keep the author invocation's `canonical_visual_envelope` null until
   `validation.canonical-ovrtx-evidence.v1` runs after authoring.
   Load the exact bound preparation publication before constructing the author
   invocation, and copy its complete `source` binding into the invocation.
   Never reuse the request's staged-source binding: the retained publication
   source is authoritative even when both paths contain byte-identical files.
   Author its preparation readback as
   `content-agent-workflows.articulation-preparation-readback-draft.v2` and
   omit both `source_dependency_bundle_sha256` and
   `saved_stage_dependency_bundle_sha256`. Never calculate, guess, or recover
   those package-aware identities from a publisher error. The trusted
   preparation publisher derives and seals them; v1 readbacks remain only for
   trusted producers that already own both exact identities. A typed failed
   preparation result is terminal for that attempt and must be passed unchanged
   to `fail-leaf`, not repaired by retrying with a guessed digest.
   `articulation.author.v1` has one explicit two-call pause inside the same
   active leaf attempt. Its immutable invocation prescribes
   `articulation_decision_patch.json`. The first entrypoint call returns typed
   `awaiting_decision` progress with the exact identity and observation. Write
   only the complete decision patch at that prescribed path, then call the same
   `content-workflow-cli articulation agentic-leaf author --invocation ...`
   entrypoint with the same invocation bytes. Never substitute the standalone
   `articulation _agent-apply` command, import an internal Python API, or create
   or execute a Python helper at this pause. Never create, pre-create, or write
   the entrypoint-owned `native/` directory. Only the second, terminal result,
   which releases its normalized, ledger-bound native patch, may be passed to
   `complete-leaf`. In that patch, set the top-level `evidence_requirements` to
   a non-empty list and set non-empty `evidence_ids` on every
   `candidate_decisions` entry and every entry in `canonical_graph.groups`,
   `canonical_graph.memberships`, `canonical_graph.joints`, and
   `canonical_graph.rigid_link_operations`. Those lists may contain only exact
   `evidence_id` values from the current
   `articulation_author_observation.json.evidence_records`. Verify every
   required list before the first native call; never submit an empty list and
   try to repair it after a native failure, and never put future post-author
   validation goals in these fields. When canonical OVRTX
   evidence is selected with Articulation authoring, the canonical evidence node
   must depend on `articulation.author.v1`; do not render the pre-author stage.
   This is the sole outer coordinator's decision pause; it does not launch a
   Joint child or another coordinator.
   `material.assignment.v1` has two typed pauses inside one active attempt.
   Keep the invocation bytes unchanged. The first call seals preparation and
   returns `awaiting_decision`; write only the prescribed
   `native/raw/material_decision_patch.json`, then call the same entrypoint.
   The second call seals the applied result and exact post-apply OVRTX evidence
   and returns `awaiting_review`; visually review every prescribed binding,
   write only `native/raw/material_post_apply_review.json`, and call the same
   entrypoint a third time. Only a terminal `passed` result with exact output
   bytes and a resource-release receipt may reach `complete-leaf`; a
   conditional Material result is a failed required leaf.
   Keep Articulation-review and final-package visual identities separate.
   `validation.canonical-ovrtx-evidence.v1` stays bound to the authored
   Articulation bytes. Carry the accepted publication through Material,
   Texture, Physics inspection/apply, SimReady conformance, and portable
   packaging. Run `asset.final-ovrtx-evidence.v1` and SimReady validation on the
   exact portable package. `asset.combined-result.v1` must receive every
   required same-run predecessor receipt and can pass only when its projector
   proves the complete cross-domain byte chain.
4. Call `complete-leaf` with only the exact typed invocation and result. The
   repository binding executes its declared projector API v2 with an immutable
   `content-agents.asset-leaf-projection-context.v1` that binds the selected
   invocation/result path, digest, and size. The domain projector must compare
   that context with its native receipt chain before returning every exact
   digest/size-qualified terminal, index, evidence, readback, and release
   artifact. State exposes newly dependency-ready leaves without inventing a
   successor order.
   On native interruption or failure, use `cancel-leaf` or `fail-leaf` with the
   exact typed invocation and result; their declared projectors retain native
   terminal, evidence, readback, and release artifacts. Recovery
   uses `recover-leaf`, retains and supersedes the prior immutable receipt, and
   retries the same leaf without changing the graph. A required failed or
   cancelled leaf cannot aggregate or finalize successfully. Selected-leaf path
   fields must be absolute so projector replay is independent of process CWD.
   For `validation.canonical-ovrtx-evidence.v1`, a native exception or
   interruption uses the adapter-owned
   `CanonicalOvrtxEvidenceLeafTerminalResult`; bind its exact native terminal
   receipt, retained canonical request, and distinct evidence/readback files
   below the invocation `output_dir`, and pass the model's exact `error` value
   as `--reason`. Focused and provided Validation interruptions use
   `SharedValidationLeafTerminalResult`, bind the exact selected invocation,
   and follow the same distinct native artifact and exact-reason rules.
5. Continue until `next_action=finalize_receipts`, then return from the outer
   reasoning loop. Both public interactive and batch launchers own the parent
   usd-cli/session teardown, seal the exact release receipt plus command
   journal/checkpoint, and call `finalize_graph_run` only after actual release.
   A launcher-bound session rejects `finalize_graph` while that parent lifecycle
   is live. Then require `validate-terminal` to pass.

```bash
content-workflow-cli asset catalog
content-workflow-asset-state freeze-graph --run-state "$RUN_STATE" --graph "$GRAPH"
content-workflow-asset-state begin-leaf --run-state "$RUN_STATE" --leaf "$LEAF_ID"
content-workflow-asset-state leaf-dir --run-state "$RUN_STATE" --leaf "$LEAF_ID"
content-workflow-asset-state invoke-leaf --run-state "$RUN_STATE" \
  --leaf "$LEAF_ID" --invocation "$INVOCATION" --result "$RESULT"
content-workflow-asset-state complete-leaf --run-state "$RUN_STATE" \
  --leaf "$LEAF_ID" --invocation "$INVOCATION" --result "$RESULT"

content-workflow-asset-state fail-leaf --run-state "$RUN_STATE" \
  --leaf "$LEAF_ID" --reason "$REASON" --invocation "$INVOCATION" \
  --result "$RESULT"
```

Interactive resume uses `resume_interactive_asset_workflow` with the same sole
outer callback. Resume rehashes the request, graph, catalog, completed receipts,
and artifacts.
Any prompt/source/config/reference/catalog/graph digest drift, changed edge or
dependency, missing receipt, or released-resource mismatch stops the run. Never
regenerate or replace the graph during resume. Do not expose credentials or
copy parent renderer configuration into run artifacts.

`asset run --dry-run` stages and digests the full source closure but invokes no
provider, renderer, simulator, or leaf. It leaves the run at `freeze_graph`.
URDF/MJCF, OBJ, glTF, and Collada references are discovered and staged with
their relative layout. Pass the smallest complete package directory with
`--source-root` for an opaque multi-file CAD or mesh format, or when a parsed
format intentionally references files above its own directory. Missing, remote,
symlinked, or out-of-root dependencies fail before child launch.

## Frozen Graph v1 Compatibility

Historical runs frozen with `content-agents.asset-execution-graph.v1`, a v1
catalog/descriptors, v1 leaf receipts, and a v1 graph terminal receipt may be
read, resumed, recovered, and finished without changing their exact graph bytes
or caller-supplied topological order. The historical `complete-leaf`,
`fail-leaf`, `cancel-leaf`, and `finalize-graph --resource-release` artifact
flags remain compatibility-only aliases for those v1 targets.

The `finalize-graph` transition CLI is graph-v1 compatibility only. Graph v2
returns from the sole reasoning loop and can be finalized only by the launcher
after the exact parent teardown receipt, journal, and checkpoint are sealed.

The version matrix is closed. Graph v1 accepts only v1 catalog, descriptor,
leaf-receipt, and graph-terminal-receipt identities. Graph v2 accepts only the
repository catalog/descriptor v3 identity plus projector/context/projection,
leaf-receipt v2, and graph-terminal-receipt v2 identities. No mixed receipt,
catalog, dependency, upgrade, or downgrade is valid. Graph v2 also rejects all
legacy completion/finalization flags and requires exact descriptor-resolved
invocation/result artifacts plus the launcher-owned parent release receipt,
command journal, and command checkpoint.

A graph-v1 run is historical completion evidence only. It cannot be promoted,
reinterpreted, or upgraded into graph-v2 qualification. Any v2 qualification
requires a fresh request, repository catalog v3, graph v2, and new artifact
identity chain.

## Explicit Fixed Compatibility Mode

The remainder of this document describes only
`selected_mode=compatibility_fixed`, entered solely with
`content-workflow-cli asset run --compatibility-fixed-order` plus its historical
Joint and Material inputs. It retains `STAGE_ORDER` for existing callers. It is
unreachable from agentic qualification and must never be used as an agentic
fallback. `asset review`, coordinator plan/review drafts, stage revisits, and the
fixed sequence below are compatibility-only.

### Fixed Compatibility Sequence

`Geometry -> Articulation review -> Material -> Texture -> Physics -> Validation -> USDZ package`

### Fixed Compatibility Guardrails

- A fixed-compatibility request starts with an immutable CAD, mesh, USD, URDF,
  MJCF, or `geometry.source.v1` provider export and enters Geometry first.
- Text/image authoring is a separate Geometry Agent operation. Complete it
  before starting asset composition; never execute provider-native source from
  the composed workflow.
- Geometry may run bounded repair and may consume an already completed Part
  Segregation handoff; it does not start a second reasoning loop.
- A conditional Geometry handoff remains conditional downstream. This workflow
  does not turn Geometry readiness into final SimReady conformance.
- Joint owns reviewed articulation inference and the authored joint graph. The
  later Physics stage preserves that graph while adding mass, colliders, rigid
  bodies, runtime simulation, and physics refinement.
- Texture work must name an explicit material or renderable prim scope and may
  select at most 64 texture units.
- An articulation review gate is mandatory. Never auto-accept candidates.
- Each domain command retains its own supported capability and terminal
  semantics. The coordinator cannot turn a conditional domain result into a
  pass.
- The asset coordinator is the only cross-domain reasoning loop. It loads each
  domain's atomic skills and invokes the same focused step surfaces that a
  standalone domain child would use.
- Bounded refinement and backward stage revisits are allowed only after an
  evidence review records the reason and repair scope. A frozen Joint review
  receipt cannot be invalidated.
- The default global ceilings are 48 plan revisions, 48 evidence reviews,
  three refinements per stage, and three revisits. When a budget is exhausted,
  record a concrete stop decision; do not retry the same transition.

### Fixed Compatibility Prerequisites

- Read the frozen `request.json` and verified `asset_run.json` paths from the
  launcher prompt.
- Use the `repository_root`, `joint_config`, `materials_yaml`,
  `materials_usd`, references, and prompt exactly as stored in the request.
- For a provided-source request, use the staged source closure and any staged
  segmentation evidence exactly as bound in the request. Do not substitute
  external paths.
- If the source came from Geometry Agent, preserve the complete source bundle,
  provider receipt, selected representation, and source revision as immutable
  input evidence. Do not substitute another export or provider revision.
- A prompt-only request contains a launcher-generated text reference under
  `inputs/prompt-reference.md`; pass it to Material exactly like an external
  `--reference` file. Never regenerate or edit it during resume.
- Do not expose credentials or copy their values into workflow artifacts.
- The launcher owns one usd-cli session and sidecar for the workflow. Nested
  workflow commands reuse that session contract.

### Fixed Compatibility Instructions

1. Run `content-workflow-asset-state status --run-state <state>` before every
   action. Never edit the state or request JSON directly. Act only on
   `current_stage` and the coordinator's `next_action`.
2. Inspect the frozen prompt, exact stage input, accepted predecessor evidence,
   and any current-attempt evidence, but never bind unreviewed current-attempt
   files into a plan because the resumed executor may rewrite them. Write an
   `asset-coordinator-plan-draft.v1` JSON file and seal it with `record-plan`.
   The plan must state the stage objective, typed executor steps, acceptance
   evidence, and why this revision follows from inspected evidence. Never put
   mutable coordinator control files such as `asset_run.json`, locks,
   `terminal_validation.json`, launcher request/result files, domain
   checkpoints/progress, or live child logs in `evidence_paths`; the state API
   rejects them before sealing. Checkpoint exceptions are limited to the
   terminal `validation_checkpoint.json` in an accepted Validation review and
   terminal `workflow_checkpoint.json` in an accepted Texture review. The
   state API first proves these are terminal and binds their native identities
   and artifact manifests.
3. When `next_action` is `begin_stage`, call `begin-stage`, then ask for the
   canonical `stage-dir`. Revisited attempts use `attempts/02`, `attempts/03`,
   and so on; never overwrite a superseded attempt. Every reviewed output and
   evidence file must be a regular file below that exact current stage-attempt
   directory, including coordinator-authored Physics and cross-stage receipts.
4. Execute the matching typed domain path below using the exact
   `input_asset.path`. Domain code may inspect, execute, validate, and
   finalize, but it must not delegate the full cross-domain goal to another
   agent. External authoring is completed before this workflow and is not a
   delegation exception.
5. Inspect the native result and evidence yourself. Write an
   `asset-coordinator-review-draft.v1` with findings and exactly one decision:
   `accept`, `await_review`, `refine`, `revisit`, `stop_failed`, or
   `stop_cancelled`. Seal it with `record-evidence-review`. Stage-specific
   Geometry, Physics, and final-package acceptance checks run at this boundary,
   so a bad `accept` remains `execute_stage` and can be replaced with `refine`,
   `revisit`, or a stop decision without recovering the whole attempt.
6. For `accept`, call `complete-stage` with the exact reviewed output and the
   exact evidence list. For `refine`, record a revised plan and rerun only the
   named repair scope. For `revisit`, target an earlier non-frozen stage; the
   state archives that stage and all downstream work before replanning. For a
   stop decision, retain concrete failure evidence when available and stop. If
   an executor fails before producing an artifact, call the session's typed
   `fail_stage` or `cancel_stage` transition with the concrete reason.
7. Continue until finalization completes and `validate-terminal` returns zero.

Every sealed plan, review, and handoff is immutable. Never overwrite an
artifact already named by one of those records. Put each refinement in a new
numbered subdirectory with a distinct output asset path. An interrupted
executor may resume its current attempt while the coordinator still records
`next_action: execute_stage`. The receipt-bound Joint continuation is also
preserved if the launcher fails while the stage is `ready`, after exact human
decisions are bound but before `begin-stage`. Recovery after an accept,
await-review, or stop review starts the next numbered attempt because that
attempt's evidence is already immutable.

### Coordinator Draft Schemas

Use the exact plan/review shapes and decision constraints in
[coordinator-run-records.md](references/coordinator-run-records.md). Both
schemas reject unknown fields.

### Geometry

Follow [geometry-stage.md](references/geometry-stage.md) using the frozen
policy and exact input. For a provider-produced source, preserve its immutable
`geometry.source.v1` manifest, selected representation, parameters, parts,
rights, provenance, and revision through the Geometry handoff. Only its typed,
digest-bound durable USDC handoff may enter Articulation; never continue with
the original source after rejection.

When the request names a completed Part Segregation run, consume only the
validated `geometry.segmentation_run` binding. The launcher stages the complete
producer file closure below `inputs/geometry-segmentation/`; never substitute
the original external run path or accept unbound segmentation evidence.

### Articulation

Build a validated `EmbeddedArticulationPreparation` from exact source/dependency
identity plus deterministic hierarchy/member/owner/capability/render/usd-cli records:

```bash
content-workflow-cli articulation run \
  --usd "$INPUT_ASSET" --embedded-preparation "$ARTICULATION_PREPARATION" \
  --output-dir "$STAGE_DIR/domain-run" --intent "$PROMPT" \
  --embedded-run-state "$RUN_STATE"
```

The first result is `awaiting_decision`. Provider-backed compatibility instead
uses `--joint-config`; its Stage 2 output is optional proposal evidence. Read
`articulation_agent_observation.json` and author one
`EmbeddedArticulationDecisionPatch`. The canonical graph must cover every source
member/owner and explicitly decide grouping, membership/co-rigid disposition,
endpoints, type, signed axes, limits/units, frames, roles, and required fact
states. Unknown, unresolved, explicit-fixed, stale, or incomplete facts block
authoring. Persist exact outer `accept`, `reject`, or `revise` plus human policy:
default `not_requested`, or `human_required` for frozen policy, ambiguity,
unsupported facts, or contradictory evidence. Apply through `articulation
_agent-apply`; never reuse the provider Stage 2 document. Reject/revise blocks.

For `human_required`, bind `canonical_articulation_graph.json` in an
`await_review` evidence review and call `require-review` on that exact file.
After `asset review`, use the bound reviewer and decisions; every frozen ID must
appear exactly once as `accept`, `reject`, or `revise`, and only all-accept may
author. Resume with:

```bash
content-workflow-cli articulation review \
  --run-dir "$STAGE_DIR/domain-run" \
  --decisions-json "$BOUND_DECISIONS" --reviewer "$REVIEWER"
```

For `not_requested`, create no human receipt; exact outer acceptance remains
mandatory. The adapter authorizes only the exact accepted graph, plus matching
human acceptance when selected, and binds `predictions_path: null`. After saved
readback proves exact topology, frames, grouping, ownership, membership, and
co-rigid disposition, author an `EmbeddedArticulationPostReviewPatch` bound to
the execution-result and accepted-decision digests:

```bash
content-workflow-cli articulation _agent-finalize \
  --run-dir "$STAGE_DIR/domain-run" --post-review-patch "$POST_REVIEW_PATCH"
```

Complete only an accepted post-readback review. Bind the output, terminal
checkpoint/request/summary, preparation/evidence/proposal,
`canonical_articulation_graph.json`, graph-only request/result/readback,
`approved_articulation_candidates.json`, outer reviews, optional human, authorization, and receipt.

### Material

Use the prepare/apply-review/publish Material executor so this coordinator
remains the only reasoning loop. First prepare candidates and visual evidence without launching
`materials assign`:

```bash
# Repeat the typed reference flags for every frozen reference in request.json.
python -m content_workflow_cli.material_coordinator prepare \
  --usd "$INPUT_ASSET" \
  --materials-yaml "$MATERIALS_YAML" \
  --materials-usd "$MATERIALS_USD" \
  --reference-file "$PROMPT_REFERENCE" \
  --output-dir "$STAGE_DIR/domain-run/iteration-01" \
  --output-usd "$STAGE_DIR/domain-run/iteration-01/material.usdz" \
  --repo-root "$REPOSITORY_ROOT" \
  --no-optimize
```

Immediately copy the literal `preparation_sha256` value from prepare's
structured stdout into parent-owned coordinator state outside the run
directory. Do not recompute it from the child-writable run in a later tool call.

Pass every frozen reference with its original kind. Inspect the prepared
candidate inventory, palette, initial renders, prompt, and preserved Joint
graph. Author the canonical typed decision patch at
`$STAGE_DIR/domain-run/iteration-01/raw/material_decision_patch.json` using
`references/coordinator-decision-schemas.md`. Then apply the accepted workflow
decision and generate post-apply OVRTX evidence. This command must stop with
`review_required`; it does not publish the requested output or close the
usd-cli session:

```bash
python -m content_workflow_cli.material_coordinator finalize \
  --run-dir "$STAGE_DIR/domain-run/iteration-01" \
  --decision-patch "$STAGE_DIR/domain-run/iteration-01/raw/material_decision_patch.json" \
  --preparation-sha256 "<literal preparation_sha256 returned by prepare>"
```

Immediately copy the literal `application_receipt_sha256` value from
finalize's structured stdout into the same parent-owned coordinator state. Do
not persist either phase seal in the run directory and do not recompute it.

Inspect every digest-bound final render named by
`raw/material_application_receipt.json`, including the 24-frame turntable and
GIF. Author the exact post-apply assessment at
`raw/material_post_apply_review.json` using
`references/coordinator-decision-schemas.md`; its `checked_view_bindings` must
copy every final binding from the application receipt without modification.
Then publish and close the shared session:

```bash
python -m content_workflow_cli.material_coordinator review \
  --run-dir "$STAGE_DIR/domain-run/iteration-01" \
  --review-patch "$STAGE_DIR/domain-run/iteration-01/raw/material_post_apply_review.json" \
  --preparation-sha256 "<literal preparation_sha256 returned by prepare>" \
  --application-receipt-sha256 "<literal application_receipt_sha256 returned by finalize>"
```

Accept only the canonical restored/exported USD after `review` returns a clean
`pass`. Bind `coordinator_result.json`
and every digest-bound entry in its `evidence` list exactly once. That list
includes the coordinator request/preparation, applied decision patch,
application receipt, post-apply review, assignments, operation counts, VQA,
validation evidence, final summary, restore response, material-binding audit,
OVRTX probe, run packet, visible candidates, palette, initial and final renders,
turntable/GIF, exact backend responses, final-render records, and trace/replay
outputs. Preserve the Joint graph. If visual evidence calls for a bounded
change, record `refine`, revise the plan, author a new decision, and run all
three phases again in an isolated refinement subdirectory rather than starting
another agent or reapplying the sealed attempt.

If the coordinator stops, revisits, or is cancelled after `prepare` but before
`finalize`, or after `finalize` returns `review_required` but before `review`,
release the prepared usd-cli session explicitly:

```bash
python -m content_workflow_cli.material_coordinator release \
  --run-dir "$STAGE_DIR/domain-run/iteration-01" \
  --preparation-sha256 "<literal preparation_sha256 returned by prepare>"
```

For cancellation after `finalize`, also pass
`--application-receipt-sha256 "<literal application_receipt_sha256 returned by finalize>"`.
The coordinator
must retain the originally captured parent-side digests; recomputing either
digest from a child-writable run directory destroys the phase-boundary seal.

### Texture

Derive the narrow texture scope from the frozen prompt plus accepted material
assignments. Load the Texture scope, candidate, quality, and publication atomic
skills. Pass explicit `--material-path` or `--prim-path` values to
`content-workflow-cli texture run`; embedded mode persists source inspection
and a service proposal, then yields a focused observation without a child.
Never select the whole asset implicitly. Use the exact material output, prompt,
nested `domain-run`, and workflow-owned usd-cli session. Before review,
pass `--embedded-run-state "$RUN_STATE"`; the typed request binds the exact outer
request, plan, attempt, input, and canonical nested output root. New v2 runs require this binding; pre-upgrade v1 runs may finish an immutable legacy request through their prior acceptance contract. Drive each next capability with
`content-workflow-cli texture _agent-step` and an exact typed decision patch.
Resume an interrupted attempt only before review, and start post-review
refinements in a new numbered stage attempt.
The first outer patch authors exact targets, appearance, generator inputs, preservation constraints, and acceptance criteria.
Generation produces a non-mutating candidate. Inspect fresh per-unit visual
evidence and record outer accept/reject/revise dispositions; usd-cli/VQA
findings are critique only. Publish the exact accepted candidate once, review
that mutation separately, and never promote a conditional or rejected result.

Bind `request.json`, `embedded-domain-decision-artifacts/journal.json`, every
artifact named by that journal, `validation_evidence.json`, `final_summary.json`,
the terminal `workflow_checkpoint.json`, `workflow_progress.json`, every
candidate unit artifact named by that checkpoint, and every visual-evidence
render named by it. The final summary must name the shared completed mutation
receipt. The progress index must match the terminal checkpoint. The checkpoint source digest must equal the exact Material handoff; its output path and digest must equal the reviewed Texture publication and receipt.

### Physics

Run `content-workflow-cli physics apply` against the exact textured output. The asset coordinator owns the Physics reasoning. Inspect the prompt, reviewed Joint
graph, component inventory, appearance handoffs, and behavior references; author
a typed `physics_decision_patch.json` and, for hierarchy changes, a typed
`physics_topology_plan.json`. If inventory paths are instance proxies, stop and
hand the asset back for offline de-instancing; the topology-plan executor cannot
target instance proxies. Invoke only the lower-level executor with a nested `domain-run`,
`--output-usd "$STAGE_DIR/physics.usdz"`, `--direct-executor`,
`--decision-patch`, optional `--topology-plan`, simulation enabled,
`--fail-on-validation-error`, and `--no-optimize`.
Do not use the default Physics command path because it launches a nested
reasoning turn. The deterministic CLI reuses the exact request-bound parent
editing session named by the launcher identity. It must not create or close a
session or the parent sidecar.

Name the coordinator-owned files
`$STAGE_DIR/coordinator_physics_decision_patch.json` and, when needed,
`$STAGE_DIR/coordinator_physics_topology_plan.json`; pass those exact paths to
`--decision-patch` and `--topology-plan`. Use the exact schemas in
`references/coordinator-decision-schemas.md`.

Preserve Joint topology and appearance. Physics owns rigid bodies, masses,
colliders, physical materials, and runtime validation. Bind the authored
self-contained USDZ, assignments, coordinator decision/topology files, component
inventory, the separate runtime report, and validation evidence. Bind trajectory
JSONL and rendered frames only when evaluated. For topology changes, also bind
`assignments.prepared_asset`. Never accept a bare layer extracted from that
package: its texture dependencies are not part of the digest-bound handoff.

The accepted evidence list must include the coordinator patch, the native
`raw/physics_decision_patch.json`, `physics_assignments.json`,
`raw/physics_components.json`, `raw/physics_apply_report.json`, the applied
patch when distinct, the runtime simulation report, the trajectory JSONL named
by an evaluated report, and native validation evidence. A schema-readback skip
report has `not_evaluated: true`, no trajectory, and a warning. Both patches must
match, the coordinator patch must bind the exact Texture handoff, and every
assignment path must be sealed. A topology-changing run must also seal
`raw/physics_topology_report.json`; its input/output digests, applied operations,
and invariants must exactly match the coordinator topology plan.

The native Physics validation evidence must parse as the public typed schema and
name the exact published Physics asset plus its byte SHA-256. Do not reuse a
passing evidence file from an earlier derivative or repackage the validated
asset after that receipt is written.

With `runtime_required`, do not complete Physics for a `conditional` or
`not_evaluated` runtime result. With `schema_readback`, require a passing
`physics_properties` check and preserve runtime gaps as warnings without a
dynamic claim. Every enabled Joint endpoint must name its exact distinct rigid
body. Nested bodies require a world-preserving reset xform stack recorded as
`reset_xform_stack: preserve_world`; otherwise fail and retain the evidence.

Do not pass `--behavior-prompt`, `--behavior-prompt-file`, `--tune`, `--refine`,
or `--scenario` to this deterministic invocation; those flags select the
nested agentic branch. Use the frozen prompt and reference images directly when
authoring the decision patch and judging simulation evidence. If
runtime evidence disproves the plan, record a bounded Physics refinement or
revisit Material/Texture as warranted; never mutate the frozen input or Joint
review receipt.

### Validation

Translate the frozen user goal and accepted cross-stage claims into the
Validation task. Select supported templates from evidence: use `render_valid`
for canonical load/render validation and add `look_right` only when appearance
or reference matching is part of the goal. Do not hardcode the same template
pair for every prompt, and do not request unsupported `physics_sane` or
`physical_behavior` templates. Invoke `content-workflow-cli validate run` on the
exact Physics output with the coordinator-authored `--task`, selected template
flags, exact references, isolated `domain-run`, `--direct-executor`, and an
explicit `--render-backend <frozen-outer-policy-backend>` whose literal value is
named by frozen outer policy; this is the explicit deterministic compatibility
path and does not launch a coding agent. Never infer a render backend from
installed services or ambient environment variables. If outer policy selected
none, stop before requesting `render_valid` rather than substituting a provider.

The composed stage is the cross-domain gate. Before completing it, verify and
bind all of the following into `cross_stage_validation.json`:

- the reviewed joint count/type and preserved Joint graph;
- requested material and texture appearance with references;
- the accepted Physics assignments, schema-application report, and runtime
  behavior evidence from the immediately preceding handoff;
- render/load validity and packaging readiness;
- preservation of all non-target content.

Validation never modifies the asset. Copy the input bytes to a new regular file
inside the validation stage and use that copy as the stage output. Bind
`validation_result.json`, `validation_evidence.json`, `final_summary.json`, the
terminal `validation_checkpoint.json`, and `cross_stage_validation.json`. Treat
warn according to the frozen user goal, but refine, revisit, or stop on native
`needs_refinement` or `fail`; never erase warnings. A passing visual result
cannot override failed or missing Joint or Physics evidence.

Author `cross_stage_validation.json` with schema
`content-agent-workflows.asset-cross-stage-validation.v1`. Bind the exact
Validation input and accepted Articulation, Material, Texture, and Physics
handoffs. Include exactly five claims named `joint_graph`, `appearance`,
`physics_behavior`, `render_and_package`, and `non_target_preservation`; each
claim is `pass` or `warn`, has a concrete summary, and references only evidence
already sealed in this exact partition: `joint_graph` uses both Articulation
and native Validation evidence after exact graph readback;
`appearance` uses Material or Texture; `physics_behavior` uses Physics;
`render_and_package` uses native Validation; and `non_target_preservation` may
use any accepted upstream/native evidence but must include native Validation.
Set `physics_behavior` to `warn` for schema-readback and state runtime was not
qualified; use `pass` only for runtime-required evidence.
Set `source_unchanged` to `true`. Copy every binding as the full immutable
`{"path": ..., "sha256": ..., "size_bytes": ...}` object from state; do not
reconstruct or omit binding fields.

If a prior Validation attempt was interrupted after publishing
`validation_request.json` and `validation_checkpoint.json`, resume that exact
domain run with `content-workflow-cli validate resume --output-dir
"$STAGE_DIR/domain-run" --recover-orphaned-claims`; do not create a competing
request in the same directory.

### Finalization

Package the exact validation output with the public converter into a canonical
self-contained USDZ under the finalization stage. Do not claim that an
arbitrary ZIP archive is USDZ:

```bash
content-workflow-cli convert-to-usd \
  "$INPUT_ASSET" "$STAGE_DIR/final_asset.usdz" \
  --output-format usdz \
  --output-dir "$STAGE_DIR/package"
```

Build the digest-bound combined report from the already accepted validation
summary:

```bash
content-workflow-asset-state build-report \
  --run-state "$RUN_STATE" \
  --final-asset "$STAGE_DIR/final_asset.usdz" \
  --validation-summary "$VALIDATION_SUMMARY" \
  --output "$STAGE_DIR/combined_report.json"
```

Complete finalization with the USDZ output, combined report, and converter
report as evidence. The state command independently checks canonical USDZ
layout and that the combined report binds the request, source, final bytes,
every prior handoff, and an accepted validation summary.

## Fixed Compatibility Command Reference

Only compatibility mode uses the stage-command forms in
`references/coordinator-control.md`.

```bash
content-workflow-asset-state status --run-state RUN/asset_run.json
content-workflow-asset-state stage-dir --run-state RUN/asset_run.json --stage STAGE
content-workflow-asset-state record-plan --run-state RUN/asset_run.json \
  --plan-file PLAN_DRAFT.json
content-workflow-asset-state begin-stage --run-state RUN/asset_run.json --stage STAGE
content-workflow-asset-state execute-geometry --run-state RUN/asset_run.json
content-workflow-asset-state record-evidence-review \
  --run-state RUN/asset_run.json --review-file REVIEW_DRAFT.json
content-workflow-asset-state require-review --run-state RUN/asset_run.json --candidates FILE
content-workflow-asset-state complete-stage --run-state RUN/asset_run.json \
  --stage STAGE --output-asset ASSET --evidence FILE --summary TEXT
content-workflow-asset-state fail-stage --run-state RUN/asset_run.json \
  --stage STAGE --reason TEXT
content-workflow-asset-state cancel-stage --run-state RUN/asset_run.json \
  --stage STAGE --reason TEXT
content-workflow-asset-state build-report --run-state RUN/asset_run.json \
  --final-asset ASSET --validation-summary FILE --output FILE
content-workflow-asset-state validate-terminal --run-state RUN/asset_run.json
```

Repeat `--evidence` for every accepted artifact. Use `asset resume --recover
"reason"` only after the previous runner is confirmed gone and the concrete
cause is corrected.

## Output Format

- `request.json` with `selected_mode=agentic`, sole coordinator identity,
  repository-owned v3 catalog, and prompt/source/config/reference/catalog digests
- immutable `execution_graph.json` with selected set, dependency DAG,
  required/optional requirements, omissions, descriptor bindings, and graph digest
- `asset_run.json` with graph-bound leaf states, transitions, attempts, immutable
  receipt bindings, supersession, and no fixed stages
- `leaves/<lexical-index>-<leaf-id>/attempts/<attempt>/leaf_receipt.json`
- `graph_terminal_receipt.json` covering every selected and omitted leaf, terminal
  artifacts, timing, and resource releases
- `terminal_validation.json`
- child logs and final messages retained by the launcher

Fixed compatibility output instead retains:

- `request.json` with frozen prompt, staged source, references, typed Geometry
  policy when selected, and runtime settings
- `asset_run.json` with ordered stages, attempts, transitions, and digests
- immutable `coordinator/plans` and `coordinator/reviews` chains recording
  prompt interpretation, evidence findings, revisions, revisits, and stops
- ordered attempt directories below `stages/`, beginning with Geometry, then
  Articulation, Material, Texture, Physics, Validation, and Finalization
- one digest-bound `handoff.json` per completed stage, exact persisted
  articulation review decisions, `final_asset.usdz`, and `combined_report.json`

## Troubleshooting

For agentic runs, preserve the failing leaf's native disposition, native result,
terminal receipt, indexes, evidence, readbacks, and releases; never replace the
graph, upgrade a non-passing result, or enter fixed compatibility. Fixed-mode failure
dispositions remain in `references/coordinator-control.md`. In either mode,
never bypass launcher lifecycle, resource release, or artifact custody.
