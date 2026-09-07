# Verified Validation Operation Ingress v1

This public protocol lets one outer Codex or Claude organizer carry an
already-produced, domain-verified result into shared Validation without
rerunning the domain workflow. It is a byte/provenance boundary, not a planner,
provider, renderer, simulator, judge, or report normalizer.

The durable envelope schema is
`content-agent-workflows.verified-validation-operation-envelope.v1`. The public
Python verifier entry point is `ingest_verified_operation_result`; the public
CLI leaf is:

```bash
content-workflow-cli validate ingest-verified-operation-result \
  --envelope domain-envelope.json \
  --output-dir provided-validation-run
```

## Domain projector handoff

A domain-owned verifier/projector must publish these immutable inputs:

| Envelope field | Required binding |
| --- | --- |
| `operation_id`, `gate_id` | Stable hierarchical IDs, unique within the provided run. |
| `evidence_type`, `claim_scope` | Native evidence/report family and bounded claim; shared Validation does not reinterpret either. |
| `native_report_type`, `native_payload_type` | Type identities paired with the exact report/payload bindings. |
| `native_status` | One of `pass`, `warn`, `fail`, `not_requested`, `not_evaluated`, `blocked`, or `error`. |
| `required`, `authority` | Required-gate policy and `deterministic_fact`, `outer_review_input`, or `advisory_critique`. |
| `source`, `output` | Exact pre/post artifact `ExecutionArtifactBinding` values. |
| `dependencies`, `artifacts` | Complete ordered exact-file bindings for inputs and evidence outputs. |
| `native_report`, `native_payload` | At least one bound native result. Neither is parsed into a generic broad gate. |
| `producer`, `tool`, `profile`, `backend`, `verifier`, `projector` | Versioned component IDs whose contract file and applicable configuration file carry SHA-256 and byte size. Profile/backend may be absent only when genuinely inapplicable. |
| `projection` | A v1 projection manifest repeating result/artifact identity and the canonical digests of every component identity. |

All `ExecutionArtifactBinding.path` values must be absolute paths to one regular
file with one hard link. Shared Validation opens every parent without following
symlinks, opens each file with no-follow semantics, re-reads all bytes, checks
device/inode/size stability, and compares SHA-256 plus byte size. It performs
the same readback on every transitive component contract and configuration.

A fixed-pipeline path-only `common.validation_evidence.EvidenceArtifact` is not a
canonical provided-result input. Domains must project it to this envelope only
after producing and verifying the native report and all exact bindings.

## Explicit modes

Focused template/rule/profile preparation publishes `execution_mode: execute`.
Ingress publishes `execution_mode: provided`. Their artifacts cannot coexist
in one run directory. There is no fallback, automatic selection, provider
lookup, rendering, simulation, report migration, or normalization between the
modes.

After all explicitly chosen envelopes are ingested:

```bash
content-workflow-cli validate collect-evidence \
  --output-dir provided-validation-run
content-workflow-cli validate assess \
  --output-dir provided-validation-run \
  --assessment outer-assessment.json
content-workflow-cli validate review-assessment \
  --output-dir provided-validation-run \
  --review outer-review.json
```

The evidence index binds the complete ingest index into one
`assessment_identity_sha256`. The outer-authored
`VerifiedOperationCoordinatorAssessment` must cover every imported operation
exactly once, in order, with its gate ID, ingest-receipt SHA-256, and unchanged
native status. Required failure/block/error states cannot pass;
`not_requested`/`not_evaluated` cannot be relabeled; advisory critique cannot
become pass.

The terminal receipt retains one independent disposition per native operation.
The fixed-pipeline five Validation gates remain present as `not_evaluated` in provided
mode; shared Validation never silently collapses domain operations into them.
The assessment is the canonical semantic/visual authority. Optional critique
remains advisory.

`receipt_status: completed` means deterministic terminal publication and
readback succeeded; it is not a semantic pass. The exact outer review
disposition remains separately typed as `accept`, `reject`, `revise`, `retry`,
`stop`, or `cancelled`. Only `accept` of a passing assessment authorizes the
result. Every other disposition is preserved verbatim and launches no hidden
retry or revision loop; the outer organizer decides whether to stop or begin a
new explicit attempt.

## Canonical post-mutation visual evidence

Rendering is deliberately separate from ingress:

```bash
content-workflow-cli validate produce-canonical-visual-evidence \
  --usd post-mutation.usda \
  --source-usd source.usda \
  --output-dir canonical-visual \
  --render-backend ovrtx
```

This leaf uses the package-owned usd-cli with the explicitly selected local or
remote OVRTX backend and requires a complete USD dependency closure. Its v2
payload binds the exact source/output USD, dependencies, image bytes, render
responses, camera records, usd-cli command journal/checkpoint and source
revision, native render report, metadata, tool/backend, verifier, and projector
identity. A completed render proves evidence production only; the outer
organizer makes the exact-digest semantic visual decision. The emitted v1
envelope can then enter provided mode. Ingress itself never imports or invokes
the rendering leaf.

This is a reusable architecture protocol, not asset qualification. Material,
Physics, and Joint retain ownership of native report production, verification,
meaning, and projection.
