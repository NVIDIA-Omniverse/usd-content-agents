# Articulation preparation and proposal-attempt contract

The provider-neutral Articulation contract adds two pre-qualification leaves.
They do not run Joint inference, author USD, render, score, attest, or
coordinate an asset workflow.

The domain-local `articulation_asset_leaf_catalog()` adapts these leaves to the
shared outer-selected graph API without selecting or executing either one. It
publishes stable `articulation.preparation-publisher.v1` and
`articulation.proposal-provider.v1` descriptors, exact invocation/result schema
digests, and the proposal-to-preparation dependency. Only an outer coordinator
may choose graph membership, order, optionality, or terminal-output status.
Write the selected descriptor's typed invocation below its shared leaf-attempt
directory and pass it through the entrypoint's `--invocation` option. The
domain adapter re-establishes every invocation-bound input before execution;
preparation derives the create-only `articulation-preparation-publication/`
directory beside that envelope, and a proposal attempt derives the create-only
`articulation-proposal-attempt/` directory there. Neither output path is
caller-authored data, and retry requires a fresh host-selected attempt directory.
The selected-leaf proposal envelope
accepts only a retained `artifact-json` payload. Live HTTP endpoints and
credential sources remain trusted standalone/deployment configuration, never
model-authored selected-leaf fields. On success the selected proposal
entrypoint emits the terminal receipt advertised by its descriptor.
The selected preparation entrypoint emits its existing publication JSON on
success or a generic typed failed result plus an invocation-bound terminal
receipt when deterministic input validation fails. It never emits dependency
identity values in an error.
Direct CLI options remain available for standalone calls and cannot be mixed
with a graph-selected invocation.

## Deterministic preparation publication

`publish_embedded_articulation_preparation` consumes one complete readback
beneath a retained root. A public outer coordinator authors
`ArticulationPreparationInspectionReadbackDraft` with schema
`content-agent-workflows.articulation-preparation-readback-draft.v2`. The draft
binds the exact source, complete dependency inventory, inspector configuration
and implementation, saved stage, render records, and scene records, but omits
both `source_dependency_bundle_sha256` and
`saved_stage_dependency_bundle_sha256`. The coordinator must not calculate,
guess, or recover those package-aware values from diagnostic errors. The
trusted producer independently derives and seals the source and saved-stage USD
dependency identities, cross-checks every hierarchy path, parent, and type
against the saved stage, verifies that every claimed member is active, reopens
every retained artifact after inspection, and writes a new create-only
publication directory. A typeless USD `def`, `over`, or `class` prim is
represented by the exact empty `type_name` token; it is not omitted or assigned
a synthetic type. An inactive prim fails publication rather than being
certified as active source evidence.

The v1 readback remains a compatibility input only for a trusted producer that
already owns its two exact package-aware identities. Both readback versions
have one exact retained dependency inventory shared by source and saved-stage
verification. A distinct saved stage must therefore resolve through the same
retained dependency paths; a flattened stage or a stage with a different
closure requires a future schema and fails closed here.

The API intentionally has no caller-supplied `source_members` or
`authoritative_owners` arguments. It derives both from the exact membership rows
in the retained `ArticulationPreparationInspectorConfiguration`. Those rows
must use `retained-explicit-membership-v1`, exactly cover the stage-verified
hierarchy, name only known self-owned authoritative owners, and equal the saved
readback observations.
Structurally valid fabricated readback rows are rejected. The six required
`EmbeddedArticulationPreparation` records bind the saved readback and every
retained artifact. Symlinks, hard links, non-regular entries, path escape,
stale claims, incomplete coverage, readback drift, and reused or mutated
publication roots fail closed.

Current publications always retain exact `membership_rows` together with the
`retained-explicit-membership-v1` policy marker, so current graph validation
compares every membership disposition and rejects either fact without the
other. Frozen v1/v2 preparations that predate both facts remain readable and
intentionally use their historical compatibility checks; omitting either fact
is not a way to create a current publication. Saved-stage hierarchy traversal
is bounded by the readback row count. Native instances and prims marked
instanceable are rejected explicitly because this v1 producer cannot expand
instance proxies; those assets need a future versioned contract.

Set the readback's `proposal_status` to `not_evaluated` only when outer policy
selected a provider attempt. Use `not_requested` when no attempt was selected;
the proposal leaf rejects that state rather than treating it as pass.

Publish and revalidate through the public launcher:

```bash
content-workflow-cli articulation publish-preparation \
  --readback path/to/articulation_preparation_readback.json \
  --retained-root path/to/retained-inspection \
  --output-dir runs/preparation-publication

# An outer-selected graph uses its exact typed invocation instead:
content-workflow-cli articulation publish-preparation \
  --invocation runs/leaf-attempt/preparation-invocation.json

content-workflow-cli articulation validate-preparation \
  --publication runs/preparation-publication/articulation_preparation_publication.json
```

The publication contains only:

- `embedded_articulation_preparation.json`
- `articulation_preparation_publication.json`

Exact preparation bytes are reproducible for the same retained paths and
bytes; publication metadata also binds the create-only destination-specific
preparation artifact. The publisher seals the directory read-only after exact
write readback and completes full validation before returning; a changed mode
or entry inventory invalidates publication. The
publisher implementation digest records the producing version and is not
compared with a later installed version during readback. If local storage
fails before sealing, preserve the poisoned root for diagnosis and retry only
with a fresh output directory.

## Terminal provider attempts

`request_embedded_articulation_provider_proposal` accepts only an explicitly
selected `not_evaluated` leaf and never falls back. Once its fresh attempt root
and exact request exist, it writes
`articulation_proposal_attempt_terminal.json` for every invoked attempt while
the create-only storage remains writable. An irrecoverable terminal-write
failure such as exhausted or read-only storage cannot manufacture a receipt;
it poisons that root and is an infrastructure error, never a provider result.
The receipt preserves the request payload, preparation, evidence, exact
artifact-backed provider inputs, provider, capability, configuration, and
source identities, and distinguishes:

- `succeeded`
- `provider_failure`
- `invalid_response`
- `input_drift`
- `publication_failure`

The public request-v1 wire is frozen. HTTP calls, which have no local provider
input artifacts, continue to transmit exactly that v1 shape without a
`provider_inputs` field. Artifact-backed calls publish request v2 so the exact
payload binding is part of the request and terminal identities. Later proposal
validation reopens the exact original preparation for both v1 and v2 requests,
so that preparation must remain at its recorded absolute path with unchanged
bytes and file identity. V2 validation also reopens the original provider
payload and fails closed on path-safety, byte, or file-identity drift.

A replacement passes the exact prior terminal receipt and an explicit reason.
The new receipt binds that prior attempt's ID, request digest, provider, and
complete terminal-receipt lineage; every lineage binding is rechecked during
the attempt. The prior roots remain unchanged. A failed root is never retried
or converted to success in place. `not_requested` and unavailable providers
are not success dispositions. Drift of any bound terminal during the
replacement attempt is itself a verifiable `input_drift` disposition, never a
false local publication failure.

Revalidate a terminal and its complete replacement lineage through:

```bash
content-workflow-cli articulation validate-attempt \
  --terminal-receipt runs/proposal-attempt/articulation_proposal_attempt_terminal.json
```

The graph-selected proposal entrypoint likewise accepts exactly one
`ArticulationProposalLeafInvocation` through `--invocation`; it binds one exact
retained artifact payload and cannot carry an HTTP endpoint, environment
variable name, credential, or arbitrary output path. The standalone direct CLI
retains its explicit HTTP adapter for trusted operator/deployment use. That
adapter transmits the complete typed request, including bound absolute local
artifact paths, to its one configured endpoint; deployments must authorize
that metadata disclosure as well as the endpoint and credential source.

Extra, changed, escaped, or unbound entries invalidate the receipt. The
canonical terminal filename is mandatory, and successful validation
reconstructs the complete provider wrapper from the exact native payload,
request, preparation, evidence, and capability identities. The directory is
sealed read-only and completely self-validated before success returns or a
typed terminal failure is raised. If local storage
or exact readback fails after an output was created, `publication_failure`
binds that partial output rather than attributing the fault to the provider.
The terminal retains at most 128 deterministic forensic partial outputs. If a
trusted in-process adapter writes more unauthorized files into the otherwise
private attempt root, excess files are discarded and the receipt preserves the
original provider, invalid-response, drift, or publication disposition and
failure code. The failure summary records the bounded-inventory overflow.
Unsafe, unreadable, vanished, or excess partials are discarded rather than
retained as unverifiable evidence, and the receipt sets
`partial_outputs_discarded=true`; every retained artifact remains bound.
An `input_drift` receipt skips only exact paths recorded as drifted, including
the request when necessary, and verifies every other binding. An invoked
attempt failure with a written receipt exits 1 and reports its identity as JSON
on stderr. Exit 2 covers pre-invocation contract errors and irrecoverable
terminal-storage failures that leave a poisoned root; interruption preserves
its terminal when storage permits and re-raises the interrupt. The HTTP adapter
streams one explicit response through the same bounded provider-input limit;
an oversized or invalid typed body terminates as `invalid_response`.

## Human-review policy

Current v3 deterministic workflow code provides the Python SDK helper
`select_embedded_articulation_human_review_policy`. With no live task-policy,
ambiguity, unsupported-fact, or contradictory-evidence gate, it selects
`not_requested`. Any selected reason produces `human_required`, and the
existing workflow cannot finalize that path without exact accepted human
evidence. All four live facts are required keyword arguments so callers cannot
obtain a permissive answer without stating the current policy evidence.

Frozen v1/v2 patches retain their historical unconditional human-review
compatibility behavior. Current policy fields must not be mixed into those
wire formats.
