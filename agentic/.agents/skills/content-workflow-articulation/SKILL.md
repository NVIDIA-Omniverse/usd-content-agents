---
name: content-workflow-articulation
description: >-
  Run durable Joint Agent articulation-v1 workflows that bind usd-cli
  inspection evidence, pause for explicit review, resume without repeating
  completed phases, author only approved revolute or prismatic joints, and
  verify exact saved USDZ readback. Use when prompt-driven joint, drawer,
  hinge, slider, mechanism, or articulation work needs the isolated agentic
  workspace.
version: "0.2.8"
author: NVIDIA Omniverse
tags:
  - content-agents
  - articulation
  - joint
  - workflow
tools:
  - Shell
  - Filesystem
  - Python
compatibility: Requires the isolated agentic workspace and Python >=3.12; provider-backed live runs require Joint Agent and its selected backends, while provider-neutral standalone or embedded preparation accepts pre-bound deterministic source, render, and usd-cli evidence without a Joint model or service.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - articulation
    - joint
    - workflow
---

# content-workflow-articulation

Use this workflow skill to sequence four reusable atomic skills:
`content-articulation-inspection`, `content-articulation-proposal`,
`content-articulation-review`, and `content-articulation-authoring`. usd-cli
provides source inspection; Joint Agent is an optional proposal leaf and owns
Stage 2 readiness only when selected; deterministic workflow code owns exact
saved-readback preparation, review bindings, accepted-only authoring, resume,
cancellation, saved-stage readback, and publication.

The default public CLI launches exactly one long-running child with a compact
task. The child invokes focused prepare and apply steps through the shared
runner. `--execution-mode fixed` is an explicit compatibility controller only.

For a shared outer-selected asset graph, obtain the stable
`articulation.preparation-publisher.v1` and
`articulation.proposal-provider.v1` descriptors from
`articulation_asset_leaf_catalog()`. The domain adapter binds their exact
invocation/result schemas and the proposal-to-preparation dependency only. It
does not select, order, invoke, or terminalize graph nodes;
those operations remain owned by the sole outer coordinator. Put the selected
typed invocation below its leaf-attempt directory and call the descriptor
entrypoint with `--invocation`; preparation derives
`articulation-preparation-publication/` and proposal derives
`articulation-proposal-attempt/` beside that host-chosen envelope. Both are
create-only, so retry or replacement requires a fresh host-selected attempt
directory. Never put an output path, HTTP endpoint, or credential source in
selected-leaf data, and never mix that artifact with direct CLI flags.

When embedded under `content-workflow-asset`, the outer asset coordinator is
the only reasoning loop. Invoke the first `articulation run` with the active
`--embedded-run-state`, inspect `articulation_agent_observation.json`, and use
the same focused apply step. The native request binds the exact outer request,
coordinator plan, attempt, input handoff, and nested `domain-run` root. Never
launch a nested coding agent or replace the bound request after its exact outer
review or any selected human review.

## When to Use

Use for prompt-driven articulation work that must expose uncertain candidates,
survive interruption, or prove that the saved joint graph exactly matches the
approved candidate set. Typical requests mention drawers, hinges, sliders,
doors, mechanisms, joints, or articulated components.

## Limitations

- Articulation-v1 authors only revolute and prismatic joints.
- Provider-neutral preparation does not create render or usd-cli evidence. Its
  public producer derives all six preparation records from an exact retained
  source/dependency/configuration closure plus complete saved readback. Exact
  per-prim ownership/disposition authority comes from the retained
  configuration and must match the saved observation. The create-only publisher
  fails closed on unsafe paths, links, stale bytes, incomplete coverage,
  destination reuse, or readback drift.
- A selected replacement proposal provider is a separate public leaf. Its
  invocation binds one retained `artifact-json` payload; trusted standalone
  operators may instead call the explicit `http-json` direct adapter. Both are
  advisory only and never fall back to Joint inference or classic execution.
  Every invoked attempt writes one immutable terminal receipt for success,
  provider failure, invalid response, or bound replacement while create-only
  storage remains writable. Irrecoverable terminal storage failure is an
  infrastructure error that poisons the root. Preserve a failed attempt root
  and use a new root for every retry or replacement.
- Current provider-neutral runs must produce canonical post-authoring visual evidence
  through the shared OVRTX leaf, bind its exact output/dependency/report/image
  identities, and include those image digests in the outer post-readback review.
  This evidence is a review input, never a deterministic visual pass.
- Human approval does not promote a native Stage 2 `review_required` candidate.
  Re-run Joint Agent adjudication or reject that candidate.
- Authoring is topology-only: masses and colliders remain false.
- Provider-backed preparation requires a healthy workflow-owned usd-cli session and captures
  digest-bound source, topology, property, camera, response, and focused-render
  evidence before review.
- Source inspection is fail-closed at 4,096 snapshot prims. Split or simplify a
  larger inspection scope before retrying; renders do not start for a truncated
  snapshot.
- One collection supports at most 8 configured view directions and 256 focused
  renders. Reduce directions or candidate moving-part coverage before retrying.
- Skill-routed execution sends every candidate through its evidence-bound agent
  decision patch. `--review-policy none` publishes that decision as the review
  receipt; both `uncertain` and `all` retain the fail-closed all-candidate
  operator review gate. Frozen v1/v2 deserialization retains its historical
  review semantics; the current provider-neutral leaf does not emit that legacy
  compatibility shape.
- usd-cli and the workflow host must resolve the same source and run paths.
- Provider-backed inference and rendering still depend on the backends selected by the
  copied Joint Agent config.

## Prerequisites

- From the repository root, set up and activate the Python 3.12 environment:

  ```bash
  ./scripts/setup_content_agent.sh
  uv pip install --python .venv/bin/python -e "apps/joint_agent[dev]"
  mkdir -p runs/configs
  cp apps/joint_agent/configs/byoa_joint_rigger.yaml \
    runs/configs/file-cabinet-joint.yaml
  source .venv/bin/activate
  ```

- For provider-backed runs, load `usd-cli` for source inspection and evidence
  renders. The workflow owns one local sidecar for the session.
- Load usd-cli and the four atomic Articulation skills named above.
- Provide a local USD-family source. For provider-backed runs, also provide a
  copied Joint Agent BYOA config and configure its model/render backends. For a
  provider-neutral standalone or embedded run, provide either a validated
  `EmbeddedArticulationPreparation` or publish one from a complete retained
  `ArticulationPreparationInspectionReadback`.

## Instructions

Before the numbered workflow, provider-neutral callers that begin from saved
inspection readback run:

```bash
content-workflow-cli articulation publish-preparation \
  --readback path/to/articulation_preparation_readback.json \
  --retained-root path/to/retained-inspection \
  --output-dir runs/preparation-publication

content-workflow-cli articulation validate-preparation \
  --publication runs/preparation-publication/articulation_preparation_publication.json
```

The producer takes no `source_members` or `authoritative_owners` arguments; it
derives both from exact retained configuration membership rows, rejects a saved
readback observation that differs from them, and publishes
`articulation_preparation_publication.json` beside the preparation in a fresh
directory. Revalidate the returned publication path before passing its
preparation onward. The exact models, artifacts, and terminal rules are
specified in `agentic/docs/articulation_preparation_attempt_contract.md`.

1. If the outer reasoner explicitly selected an alternate proposal provider,
   start from a preparation with `proposal_status=not_evaluated` and run
   `content-workflow-cli articulation propose` with exactly one adapter. Use its
   separately written bound preparation and terminal receipt. A provider,
   transport, HTTP, response-schema, request-drift, or local-publication failure
   is a typed terminal failure, not unavailable or pass. Validate the terminal
   through `content-workflow-cli articulation validate-attempt`. For an
   explicit replacement, bind the exact prior terminal receipt and use a fresh
   attempt root. If no provider was selected, retain `not_requested` and skip
   the proposal leaf.
2. Run `content-workflow-cli articulation run` with the exact prompt, source,
   run directory, review policy, motion types, and candidate-count bounds. Use
   `--joint-config` for provider-backed compatibility, `--preparation` for a
   provider-neutral standalone run, or `--embedded-preparation` together with
   `--embedded-run-state` for outer Asset custody. Standalone launches exactly
   one child through `workflow=articulation.author`; embedded launches none.
   Neither provider-neutral route constructs `JointAgentLocalClient` or invokes
   `joint_agent.api.pipeline`.
3. Inspect `articulation_agent_observation.json` and its bound evidence. In
   standalone mode the sole child writes the complete typed decision patch. In
   embedded mode the outer coordinator authors the complete ordered canonical
   graph. Any provider candidate document is optional proposal evidence only.
4. Persist an explicit outer `accept`, `reject`, or `revise` disposition in the
   patch for the exact canonical graph. Also select `not_requested` human review
   or `human_required` with a frozen task-policy or fail-closed escalation
   reason. Neither human status is an outer-review disposition or implicit pass.
5. Apply the patch through the focused step. It fails before publication on a
   stale revision, changed digest, incomplete IDs, unsafe edit, unbound path, or
   absent exact outer review. Outer rejection or revision blocks mutation.
   Standalone `parent_unresolved`, `axis_missing`, endpoint, frame/limit,
   membership, or readback issues produce an immutable candidate-bound packet
   and permit only one focused replacement attempt. Exhausting that cap is a
   conditional terminal result, never success.
6. If `human_required` leaves the result at `needs_review`, inspect
   `agent_reviewed_articulation_candidates.json` together with the original
   `articulation_candidates.json` and `scene_evidence/manifest.json`.
   Human review must cover the exact agent-reviewed document that authoring
   will consume. Review each candidate's endpoints, axis, confidence,
   readiness, unresolved reasons, properties, and focused renders.
7. Write one human `accept`, `reject`, or `revise` decision for every ID printed
   under `review_required_candidate_ids`. Never accept a candidate that is not
   native-ready for its requested articulation-v1 motion type.
8. Run `content-workflow-cli articulation review`. It binds the decisions, the
   exact agent-reviewed candidate digest, and the usd-cli evidence digest
   into one receipt and immediately resumes. An all-accept decision proceeds;
   reject remains terminal. For revise, preserve the conditional checkpoint and
   use the refreshed v4 observation's `graph_revision_inputs` to bind the exact
   persisted human decision and complete outer decision file, then create a
   typed immutable graph revision patch. Run
   `content-workflow-cli articulation revise-graph --run-dir <run>
   --revision-patch <patch>`; it archives the exact parent graph/review,
   publishes a revised graph without authoring, and reopens complete human
   review at the revised digest. Do not edit the parent graph or reuse its
   acceptance.
9. After an interruption, run `content-workflow-cli articulation resume` with
   the same run directory. Repeating the exact original `run` command also
   resumes. Do not change source, intent, config, or evidence policy in place.
10. After deterministic authoring and saved-stage readback pause at
   `awaiting_post_review`, run
   `content-workflow-cli validate produce-canonical-visual-evidence` on the
   exact authored output with the original source and an explicitly selected
   `remote` or `ovrtx` shared backend. Then run
   `content-workflow-cli articulation bind-output-evidence` with the emitted
   `verified_operation_envelope.json`. A standalone visual payload is not
   sufficient because Joint must retain and reverify the shared producer,
   tool, backend, verifier, and projector chain.
11. Inspect every bound canonical image. Persist the mode-specific post-review
   patch that
   accepts, rejects, or revises the exact execution result, accepted graph,
   output-evidence digest, and complete ordered image-digest list; apply it
   through the focused finalization step. Absence or substitution of any output,
   dependency, render report, image, or metadata binding fails closed.
   Populate every digest from the exact bound artifact or receipt with
   deterministic shell or structured-JSON tooling; never retype, reconstruct,
   or copy a shortened digest from prose or terminal display. Before invoking
   finalization, validate that every required SHA-256 value is exactly 64
   lowercase hexadecimal characters and matches the current artifact bytes.
12. Treat only `completed` as success. Provider-neutral standalone completion
   uses the v5 checkpoint and terminal receipt. It means `owned_core` authored the exact
   accepted IDs, the saved output passed exact graph readback, canonical OVRTX
   evidence remained intact through outer review, and the terminal receipt
   binds all of them. Mock completion uses simulated validation and is test
   evidence only.
13. When separately selected, project completed graph/apply, retained Gate 3A,
   retained Gate 3B, or trusted dynamic results with the corresponding public
   `articulation project-*` leaf. Feed each emitted envelope to
   `validate ingest-verified-operation-result`; projection revalidates native
   Joint facts but never runs inference, simulation, rendering, or shared
   Validation execution. Gate 3A/3B projection requires the canonical retained
   root layout, exact plan, intake, authoring receipt, closeout, and report. The
   authoring receipt supplies the generated identities published by the real
   static workflow; do not synthesize a standalone identities document or
   substitute source/output bytes.
14. For `conditional`, retain the output and inspect unresolved IDs plus
   `validation_evidence.json` when validation ran; pre-authoring conditionals
   have no output or validation artifact. For `cancelled` or `failed`, preserve
   the run for diagnosis and restart with a new run directory; those
   checkpoints are terminal.

## File-Cabinet Scenario

Run the practical six-drawer acceptance scenario:

```bash
ARTICULATION_INTENT="Identify the six drawers, present every candidate for "
ARTICULATION_INTENT+="review, then author exactly six prismatic drawer joints "
ARTICULATION_INTENT+="and no masses or colliders."

content-workflow-cli articulation run \
  --usd /data/sm_filecabinet_d01_01.usdz \
  --joint-config runs/configs/file-cabinet-joint.yaml \
  --output-dir runs/file-cabinet-articulation \
  --intent "$ARTICULATION_INTENT" \
  --review-policy all \
  --allowed-motion-type prismatic \
  --expected-candidate-count 6 \
  --max-candidate-count 6
```

After inspecting the candidates and usd-cli renders, create
`runs/file-cabinet-decisions.json` with every printed candidate ID:

```json
{
  "candidate_0001": "accept",
  "candidate_0002": "accept",
  "candidate_0003": "accept",
  "candidate_0004": "accept",
  "candidate_0005": "accept",
  "candidate_0006": "accept"
}
```

Use the exact IDs printed by the run; the file must contain one decision for
every `review_required_candidate_ids` value. Resume through the review command:

```bash
content-workflow-cli articulation review \
  --run-dir runs/file-cabinet-articulation \
  --decisions-json runs/file-cabinet-decisions.json \
  --reviewer asset-owner
```

The `--review-policy all` setting prevents silent auto-approval. A native Stage
2 `review_required` candidate remains non-authorable even if the receipt says
`accept`; reject it or rerun Joint Agent adjudication with stronger evidence.

## Python Interactive API

For a long-running Python host, call
`content_agent_workflows.articulation.run_interactive_articulation_workflow`,
build digest-bound decisions with `build_articulation_review_receipt`, and pass
an `Articulationusd-cliEvidenceCollector` implementation when usd-cli
evidence is required. Keep the same request, client configuration, collector
configuration, and run directory across calls.

CLI-created runs are batch-mode checkpoints and are resumed with the CLI
commands above. The CLI does not adopt a separately constructed Python
interactive checkpoint.

Embedded interactive and batch callers use the same public
`EmbeddedArticulationPreparation` and
`prepare_embedded_articulation_workflow` API. Proposal fields may be absent; the
preparation records that optional leaf as `not_requested` or `not_evaluated`,
never as a pass. The outer coordinator must explicitly accept, reject, or revise
the exact persisted graph. Human review is `not_requested` by default and is
`human_required` only when frozen task policy selects it or ambiguity,
unsupported facts, or contradictory evidence require fail-closed escalation.
Graph-derived authoring, saved-stage readback, and outer post-review remain
mandatory in both modes. Current public embedded runs additionally require the
shared canonical OVRTX payload and a v2 output-bound post-review; v1/v2
compatibility artifacts retain their historical post-review contract.

Use `select_embedded_articulation_human_review_policy` from deterministic
Python workflow code for the current v3 outer patch; it is an SDK policy
helper, not a launcher subcommand. It returns `not_requested` when no live task-policy, ambiguity,
unsupported-fact, or contradictory-evidence gate exists, and `human_required`
with exact reasons otherwise. Do not serialize those current fields into frozen
v1/v2 patches; their compatibility behavior remains an unconditional human
gate.

## Output Format

- `request.json`
- `articulation_preparation_publication.json` and
  `embedded_articulation_preparation.json` when preparation is published from
  complete retained saved readback
- `embedded_articulation_preparation.json` for provider-neutral embedded runs
- replacement-provider request, native payload, v2 proposal, and bound
  preparation artifacts when the optional public proposal leaf was selected
- `articulation_proposal_attempt_terminal.json` for every invoked provider
  attempt, including failures and explicitly bound replacement attempts
- optional `inference_result.json` and provider proposal for provider-backed runs
- Joint Agent predictions and candidate report referenced by
  `inference_result.json`
- `articulation_candidates.json`
- `scene_evidence/manifest.json`, source snapshot, topology and property
  inspection, and candidate-focused image, response, and camera artifacts
- `articulation_agent_observation.json`, decision patch, reviewed candidates,
  and immutable decision ledger
- `embedded_articulation_outer_review.json` after exact outer graph review
- `graph_revisions/revision-NNN/graph_revision.json` plus immutable revised
  graph, outer review, and coordinator decision after a human revision
- optional `review_receipt.json` after selected human review
- `approved_articulation_candidates.json` after approval
- `authoring_request.json` and `authoring_result.json` after authoring starts
- `joint_rigger/authoring_attempt.json`, `diagnostics.json`, and `result.json`
- `validation_evidence.json` after validation
- `embedded_articulation_output_evidence.json` after binding shared canonical
  OVRTX output evidence
- `embedded_articulation_post_review_patch.json` and
  `embedded_articulation_terminal_receipt.json` after accepted output review
- optional Joint verified-operation projection/envelope artifacts for
  graph/apply, Gate 3A, Gate 3B, or dynamic results when those leaves are
  separately selected
- `checkpoint.json`
- `workflow_progress.json`
- `final_summary.json`
- `joint_rigger/rigged.usdz` plus Joint Rigger diagnostics and result evidence

## Troubleshooting

- If a receipt or usd-cli evidence digest differs, do not edit it in place.
  Reopen the current evidence and build a new run or valid receipt.
- If a proposal attempt is terminally failed, invalid, or replaced, preserve
  that root. Never add a success artifact to it or reuse it; start a fresh root
  and bind the prior terminal receipt when replacement is explicit.
- A failed or interrupted usd-cli capture may retry from checkpointed Joint
  Agent inference and candidate artifacts. Only a complete, verified manifest
  skips collection; incomplete capture artifacts may be replaced during retry.
- A workflow `failed` result is a terminal, diagnostic checkpoint. `resume`
  returns the same verified failure instead of retrying a possibly
  non-idempotent backend phase. Preserve that run and use a new run directory
  after correcting a transient backend problem.
- For a Joint provider terminal, follow the exact `recovery_action`. `resume`
  with `resume: true` and `clean: false` reuses the named checkpoint and its
  digest-bound provider journal in the same work directory. `restart` with
  `resume: false` and `clean: true` means the evidence is unavailable or
  untrusted: preserve the failed directory for diagnosis and start the same
  request in a new clean run directory. Never reinterpret one mode as the other
  or delete the failed evidence in place.
- Changing the usd-cli evidence policy or another digest-bound collector
  setting invalidates evidence reuse. The collector configuration digest
  is bound into the request. Preserve the old run and start a new request when
  changing it.
- If a candidate is native `review_required`, do not flip its status. Correct
  the source evidence through Joint Agent adjudication.
- If resume reports source, configuration, candidate, or output drift, preserve
  the run for diagnosis and start a new run for changed inputs.
- If exact readback fails, keep the conditional artifact but do not claim
  completion.
