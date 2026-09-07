---
name: content-workflow-validation
description: Validate USD assets through deterministic preparation, one decision-only Codex or Claude planning child, outer-validated exact adapters, independent assessment/review, and non-mutating terminal receipts. Use for agentic standalone Validation; use focused commands for explicit compatibility operations and the fixed-pipeline umbrella for config-driven Validation Agent runs.
version: "0.3.1"
author: NVIDIA Omniverse
tags:
  - content-agents
  - validation
  - agentic-workflow
tools:
  - Shell
  - Filesystem
  - Python
compatibility: Requires content-workflow-cli and the repo Python environment. Rendering, simulation, or advisory judging requires only the explicitly selected capability and its configured runtime.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - validation
    - agentic-workflow
---

# Content Workflow Validation

Use one outer reasoner to supervise a single decision-only planning child,
validate its digest-bound proposal before any operation, invoke exact
Validation adapters, inspect their typed facts and actual images, author the
assessment, and review the exact saved result. Deterministic code owns
preparation, acceptance, execution boundaries, readback, and receipts.

## When to Use

- Let one planning child select only the static, render, runtime-evidence, or
  advisory capabilities justified by explicit claims and acceptance criteria.
- Run a documented named profile without inferring additional work.
- Perform a comprehensive final agentic validation after staged checks.
- Assess a composed asset with upstream handoffs and cross-stage evidence bound
  before semantic review.
- Ingest a domain-verified operation without rerunning its provider, renderer,
  simulator, or domain workflow.

Use the `$fixed-pipeline` umbrella's `validation-agent-cli` reference for YAML/config-driven
Validation Agent runs, services, and legacy template automation.

## Limitations

- The planning child proposes checks but has no execution or publication
  authority. The outer process rejects stale, unknown, duplicate, cyclic,
  undeclared, mistargeted, mistyped, or policy-invalid plans before any call.
- V0.6 permits one instance per capability and no revision loop, carry-forward,
  or follow-up proposals. Those are post-v0.6 behavior.
- Do not mutate the source USD. A changed source/dependency identity fails.
- Final `render_valid` geometry evidence must come from the shared OVRTX USD
  render path and retain the exact source USD digest plus OVRTX render metadata.
  Local preview images are reference-only and cannot satisfy this gate.
- A precomputed OVRTX image may satisfy `render_valid` only through an explicit
  `qualified_render_evidence` record that verifies its source digest, image
  digest, backend, and retained OVRTX metadata. A generic reference image or a
  qualification string without that metadata remains reference-only. The v1
  record binds one source digest, so multi-source requests cannot use this
  precomputed-evidence path.
- `physical_behavior` consumes existing simulator/runtime/recording evidence;
  it does not launch Newton or another simulator.
- `look_right` is optional advisory critique. It is never canonical visual
  authority and never substitutes for direct outer inspection.
- Architecture readiness is not asset qualification. Do not reuse historical
  Keyboard, Dishwasher, or Texture-reference runs as fresh acceptance evidence.

## Prerequisites

1. Activate the repository Python environment and make
   `content-workflow-cli` available.
2. Resolve the exact source USD and its dependency closure. A bare unresolved
   `OmniPBR.mdl` is the canonical public renderer-runtime module declaration:
   bind its external identity without injecting bytes, and still require the
   selected OVRTX runtime to resolve it. Other missing MDL modules, including
   package-relative `./OmniPBR.mdl`, fail closed.
3. Gather task-required references, renderer receipts, simulator/runtime
   receipts, and package evidence.
4. For composed mode, prepare the accepted Articulation, Material, Texture, and
   Physics handoffs plus `cross_stage_validation.json` before collecting
   assessment evidence.
5. For canonical post-mutation visual evidence, make the package-owned in-tree
   `usd-cli` available with an explicit local or remote OVRTX backend. The leaf
   fails closed if the usd-cli provenance or OVRTX probe/render cannot be
   verified.

## Command Families

Choose one Validation command family and a fresh run directory:

- **Agentic coordinator mode (default):** use `validate run` with an explicit
  runner and model. Claude also requires an explicit `--claude-execution-mode
  sdk|cli`. Then run
  `collect-evidence`, `assess`, and `review-assessment`. One child authors only
  `validation_coordinator_plan_patch.json`; trusted code accepts and executes
  it through exact adapters.
- **Focused execute mode (compatibility):** use `prepare`, one `check` call per
  explicitly selected capability, `finalize`, `collect-evidence`, `assess`,
  and `review-assessment`. This keeps explicit outer selection available.
- **Provided mode:** use `ingest-verified-operation-result`,
  `collect-evidence`, `assess`, and `review-assessment`. It verifies an existing
  domain result and executes no Validation template, renderer, simulator,
  provider, or domain workflow.
- **All-in-one execute compatibility:** pass `validate run --direct-executor`
  for the legacy ordered-template workflow; `validate resume` resumes only that
  path. Existing composed-asset calls with `--embedded-run-state` retain their
  implicit compatibility routing. Its `validation_plan.json` is never agentic
  decision authority.

If `validate resume` is pointed at an agentic coordinator run, it writes the
digest-bound `validation_safe_restart.json` disposition with
`safe_restart_required`, preserves the prior run, and invokes neither the
legacy executor nor another child. Start a fresh coordinator run in a new empty
output directory.

The separate `validation-agent` CLI remains the fixed-pipeline config/Python
surface reached through `$fixed-pipeline`; none of these command families
replace it. Validation selects no provider runner, model, provider execution
mode, or rendering backend by default; each provider-bearing invocation must
name its choices explicitly.

## Instructions

1. Run `validate run` with the exact source, task, references, focus prims, and
   runtime configuration. Name `--runner` and `--model` explicitly. The Claude
   request path also names `--claude-execution-mode sdk|cli`. Use `--dry-run`
   to inspect the selection-free preparation and prompt without launching the
   child.
2. Verify `validation_coordinator_preparation.json` binds the source dependency
   closure, task/request/config identities, all approved capabilities and
   requirements, evidence inventory, mandatory constraints, and preparation
   digest while `selected_checks` remains empty.
3. The child writes only `validation_coordinator_plan_patch.json`: plan ID,
   bound preparation digest, claims/criteria, finite unique checks, exact
   capability/template/rule IDs, targets, focus, typed parameters,
   required/advisory disposition, dependencies, evidence requirements, and
   completion policy. Before launch, the parent freezes the complete child
   launch descriptor, including runner/model/mode and output paths; acceptance
   rejects any digest change. Select at least two distinct capabilities
   identified by the preparation as provider-free and include at least one
   explicit dependency. It must not call a provider, renderer, simulator,
   operation, nested agent, or arbitrary code.
4. Trusted acceptance validates the entire proposal and source identity before
   creating `validation_request.json`, `validation_plan.json`, or operation
   artifacts. The classic plan is an accepted-plan compatibility projection,
   never decision input or fallback.
5. Trusted execution calls `run_validation_operation` in exact dependency
   order and then finalizes. Only a selected check may invoke its exact leaf;
   an omitted capability remains explicit and a required failure or
   unevaluated check is non-success.
6. Run `validate collect-evidence`. In composed mode pass
   `--embedded-run-state`; in standalone mode omit it. This binds saved native
   results, source readback, actual render/reference images, package evidence,
   and, when embedded, upstream handoffs and cross-stage claims.
7. Inspect the evidence index and open the actual bound render/reference image
   paths. Author `ValidationCoordinatorAssessment` with independent static,
   runtime, visual, package, and cross-stage dispositions. Cite evidence IDs,
   findings, remediation requirements, confidence/limitations in rationale,
   and one terminal disposition. Deterministic aggregation is non-authoritative.
8. Run `validate assess`, read back the canonical assessment, author an exact
   `ValidationCoordinatorReviewDraft`, and run `validate review-assessment`.
   Accept only a passing canonical assessment whose bytes and identity match.
9. Treat `validation_terminal_receipt.json` as the mode-neutral terminal
   readback. For coordinator runs it binds preparation, child proposal,
   accepted-plan receipt, operation index/results, assessment/review,
   `publication_kind: validation_assessment`, and `source_mutated: false`.

For focused compatibility, choose exactly one explicit request form (`--template`,
`--rule`, or `--profile`), then run `prepare`, exact `check` calls, and
`finalize`. Missing mandatory selected operations fail closed.
   Unselected leaves remain `not_requested`; an unexecuted optional critique
   remains `not_evaluated`. Neither state is a pass.

## Provided-operation method

Provided mode is separate from focused and all-in-one execute modes and requires
its own run directory. The outer reasoner selects a domain-owned
verifier/projector, which emits one
`content-agent-workflows.verified-validation-operation-envelope.v1`. Shared
Validation re-reads the envelope, projection, source, output, dependencies,
artifacts, report/payload, and every component contract/configuration binding.
It does not execute, normalize, rerender, simulate, call a provider, or infer
another operation.

1. Have the domain verifier persist its native report and optional typed
   payload, then a projection manifest binding the exact bytes and native
   status.
2. Run `validate ingest-verified-operation-result` once per explicitly chosen
   envelope. Operation IDs and gate IDs must be stable, hierarchical, and
   unique within the run.
3. Run `validate collect-evidence`. Inspect each independent record and its
   bound native artifacts.
4. Author `VerifiedOperationCoordinatorAssessment` against the exact
   `assessment_identity_sha256`. Preserve every operation ID, gate ID, ingest
   receipt digest, and native status; optional advisory critique remains
   `advisory` or `not_evaluated`, never pass.
5. Run `validate assess`, then author the existing
   `ValidationCoordinatorReviewDraft` and run `validate review-assessment`.
   The terminal receipt retains one disposition per imported operation while
   all five fixed-pipeline gates remain explicit `not_evaluated` in provided mode.
   Its `receipt_status: completed` records successful deterministic
   publication, not semantic acceptance. Read `review_disposition` separately:
   only `accept` authorizes, while `reject`, `revise`, `retry`, `stop`, and
   `cancelled` remain exact outer decisions and never launch a hidden loop.

Do not pass path-only `common.validation_evidence.EvidenceArtifact` data to
this method. A domain projector must bind the source/output/dependency/report
bytes plus producer, tool, profile/backend, verifier, and projector identity.

Canonical post-mutation visual evidence is produced separately:

```bash
content-workflow-cli validate produce-canonical-visual-evidence \
  --usd post-mutation.usda \
  --source-usd source.usda \
  --output-dir visual-evidence \
  --render-backend ovrtx
```

This leaf renders through the package-owned usd-cli and its explicit local or
remote OVRTX backend. It binds the dependency closure, image bytes, render
responses, camera records, usd-cli command journal/checkpoint and source
revision, render report/metadata, and tool/backend identities. Its completed
render is evidence production, not semantic visual acceptance. The outer
organizer reviews the exact image digests. Ingress may carry the emitted
envelope but never renders or judges it.

## Saved-run compatibility

- Embedded receipt indexes with the legacy v1 shape predate the required
  terminal-receipt and independent-gate bindings. They are intentionally
  invalidated rather than migrated or relabeled; produce a fresh direct
  evidence chain and v2 receipt index.
- Checkpoints created before `look_right` was digest-bound to its explicit
  `render_valid` dependency are also intentionally invalidated. Start a fresh
  run identity; do not migrate the prior plan digest.

## Command Reference

Run the default agentic coordinator:

```bash
content-workflow-cli validate run \
  --usd asset.usda \
  --task "Select and execute the checks needed for release readiness." \
  --output-dir validation-run \
  --render-backend remote \
  --runner codex \
  --model gpt-5.6-sol
content-workflow-cli validate collect-evidence --output-dir validation-run
```

Use `--runner claude --model <model> --claude-execution-mode sdk|cli` for the
equivalent Claude child request. Add `--direct-executor` only when the caller
intentionally owns legacy template selection; `--template` is rejected by the
default coordinator.

Prepare exact templates:

```bash
content-workflow-cli validate prepare \
  --usd asset.usda \
  --task "Validate static physics and canonical render evidence." \
  --output-dir validation-run \
  --template physics_sane \
  --template render_valid
```

Prepare one named profile:

```bash
content-workflow-cli validate prepare \
  --usd asset.usda \
  --task "Run comprehensive final validation." \
  --output-dir validation-run \
  --profile comprehensive
```

Named profiles are `static`, `visual`, `runtime`, `comprehensive`, and
`comprehensive-with-advisory-critique`. Advisory critique is present only in
the last profile.

Atomic rule IDs are:

- `render.runtime_evidence`
- `physics.usd_schema_sanity`
- `physics.behavior_evidence`
- `visual.optional_advisory_critique`

When choosing the advisory rule/template, also choose
`render.runtime_evidence`/`render_valid` earlier in the same explicit request.

Run and finalize selected operations:

```bash
content-workflow-cli validate check \
  --output-dir validation-run \
  --template physics_sane
content-workflow-cli validate check \
  --output-dir validation-run \
  --template render_valid
content-workflow-cli validate finalize --output-dir validation-run
content-workflow-cli validate collect-evidence --output-dir validation-run
```

Assess and seal standalone evidence:

```bash
content-workflow-cli validate assess \
  --output-dir validation-run \
  --assessment outer-assessment.json
content-workflow-cli validate review-assessment \
  --output-dir validation-run \
  --review outer-review.json
```

Ingest and assess provided native results:

```bash
content-workflow-cli validate ingest-verified-operation-result \
  --envelope physics-mass-envelope.json \
  --output-dir validation-provided
content-workflow-cli validate collect-evidence \
  --output-dir validation-provided
content-workflow-cli validate assess \
  --output-dir validation-provided \
  --assessment outer-provided-assessment.json
content-workflow-cli validate review-assessment \
  --output-dir validation-provided \
  --review outer-review.json
```

For composed mode, add the exact `--embedded-run-state asset_run.json` to
`collect-evidence`, `assess`, and `review-assessment`.

## Shared benchmark report

When scoring a Validation bundle, read and follow
[references/benchmark-report-and-replay.md](references/benchmark-report-and-replay.md).
Replay-video generation is internal-only and is not part of the public
Validation workflow.

## Output Format

The agentic coordinator and focused compatibility paths publish:

- `validation_coordinator_preparation.json`
- `validation_coordinator_plan_patch.json`
- `validation_coordinator_accepted_plan.json`
- `validation_coordinator_execution_receipt.json`
- `validation_operation_preparation.json`
- `operations/<template>/operation_result.json`
- `validation_operation_index.json`
- standard `validation_request.json`, `validation_plan.json`,
  `validation_result.json`, `validation_evidence.json`, and
  `final_summary.json`
- `standalone_validation_evidence.json` or
  `embedded_validation_evidence.json`
- `canonical_validation_assessment.json`
- mode-specific execution/receipt indexes
- `validation_terminal_receipt.json`

Provided mode instead publishes `verified_operation_ingest_index.json`, one
immutable `verified_operations/<envelope-digest>/ingest_receipt.json`,
`verified_operation_evidence_index.json`, the exact canonical outer
assessment, `verified_operation_execution_index.json`, and the same terminal
receipt filename. Execute and provided artifacts cannot share a run directory.

Use `validation_result.json` and its `ValidationIssue` objects for #38 contract
compatibility. Use the terminal receipt for outer-authority and independent
gate readback.

## Troubleshooting

- **Exactly one explicit choice required**: use templates, rules, or one named
  profile, never a mixture, and only on focused/direct compatibility paths.
- **Plan rejected before execution**: inspect the exact preparation digest,
  capability/template/rule IDs, targets, focus, parameter types, dependency
  DAG, and policy constraints. Do not edit an accepted projection or fall back
  to classic planning.
- **Execute/provided conflict**: use a fresh run directory. There is no
  fallback or automatic conversion between modes.
- **Projection differs**: reproduce the domain projection from current native
  bytes and identities. Do not edit the envelope or migrate its digest.
- **Operation was not requested**: rerun preparation with a new output
  directory and an explicit outer choice. Do not modify the frozen plan.
- **Required operation not evaluated**: execute the missing mandatory leaf;
  finalization will not convert it to a pass.
- **Evidence identity is stale**: start a fresh run from current source,
  references, receipts, and handoffs. Do not relabel old artifacts.
- **Optional judge unavailable**: leave `look_right` `not_evaluated` and perform
  direct outer visual assessment from current images.
- **Cross-stage evidence missing**: create and bind the coordinator-reviewed
  cross-stage receipt before embedded evidence collection.
