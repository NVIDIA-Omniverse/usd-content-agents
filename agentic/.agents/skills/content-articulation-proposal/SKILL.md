---
name: content-articulation-proposal
description: Propose bounded revolute or prismatic articulation candidates from exact inspection evidence. Use to reason about endpoints, axes, limits, confidence, and native readiness before review.
version: "0.2.1"
author: NVIDIA Omniverse
tags:
  - content-agents
  - articulation
  - proposal
tools:
  - Shell
  - Filesystem
compatibility: Requires the isolated Agentic workspace and Python >=3.12. Built-in Joint inference is optional; a selected-leaf replacement binds artifact-json, while trusted direct calls may use artifact-json or http-json.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - articulation
    - proposal
---

# Articulation Proposal

Review advisory proposals against exact inspection evidence. A proposal may
come from built-in Joint inference or one explicitly selected replacement
provider. This atomic skill proposes or edits candidates; it never authors
joints or selects a provider fallback.

## When to Use

Use after `content-articulation-inspection` and before candidate selection. If
the outer reasoner did not select a proposal provider, retain `not_requested`
and skip this leaf.

## Limitations

- Articulation v1 supports only native-ready revolute and prismatic joints.
- Candidate ID, endpoints, motion type, and native readiness are immutable in
  an agent decision edit. Undeclared pass-through fields from Joint are also
  immutable.
- Provider proposals are advisory. They never constrain or substitute for the
  outer-authored canonical graph.
- A selected replacement provider requires a preparation whose proposal status
  is `not_evaluated`; `not_requested` and already-available modes fail closed.
- Selected-leaf data binds only `artifact-json`; trusted direct calls may use
  the mutually exclusive `artifact-json` or `http-json` adapters. A failed
  provider never falls back to Joint inference, the classic controller, or the
  other adapter.
- Every provider attempt requires a fresh `--output-dir`. An existing output
  directory fails closed so a retry cannot overwrite or mix attempt evidence.
- Every invoked attempt writes a typed terminal receipt while its create-only
  storage remains writable. Provider, transport, HTTP, invalid-response,
  input-drift, and local-publication failures are not unavailable or pass;
  local faults are never attributed to the provider. Irrecoverable terminal
  storage failure poisons the root as infrastructure failure. The root binds
  every retained regular output and self-validates after sealing.

## Prerequisites

Read the observation-bound candidate document and usd-cli evidence. Verify
their digests before reasoning about a candidate.

## Instructions

1. If a replacement provider was selected, invoke exactly one public proposal
   leaf and retain its provider, capability, non-secret provider/service alias,
   adapter, configuration, implementation, preparation, evidence digests, and
   complete prior-terminal lineage.
2. Review each candidate in canonical order.
3. Compare moving parts, fixed parent, axis, limits, confidence, and unresolved
   reasons against hierarchy, properties, picks, and focused renders.
4. Keep rejected candidates in the complete decision set.
5. Hand advisory findings, never proposal authority, to
   `content-articulation-review`.

## Command Reference

Invoke an exact external-provider payload without Joint inference:

```bash
content-workflow-cli articulation propose \
  --preparation path/to/selected-pending-preparation.json \
  --output-dir path/to/proposal-leaf \
  --intent "Suggest articulation candidates from the exact evidence." \
  --provider-adapter artifact-json \
  --provider-id replacement-provider \
  --capability-id articulation-proposal-v1 \
  --provider-payload path/to/provider-domain-proposal.json
```

Use `--provider-adapter http-json` with an explicit credential-free
`--provider-url`, `--provider-endpoint-alias`, and optional
`--provider-token-env` for a live endpoint. The command emits the exact request,
native payload, v2 `EmbeddedArticulationProviderProposal`, and a separately
written proposal-bound preparation. Pass that bound preparation to the normal
embedded run. No authoring command belongs to this skill.

Replace a prior terminal attempt only from a new root and bind the exact
lineage:

```bash
content-workflow-cli articulation propose \
  --preparation path/to/selected-pending-preparation.json \
  --output-dir path/to/replacement-attempt \
  --intent "Retry through the explicitly selected provider." \
  --provider-adapter artifact-json \
  --provider-id replacement-provider \
  --capability-id articulation-proposal-v1 \
  --provider-payload path/to/provider-domain-proposal.json \
  --replaces-terminal-receipt path/to/prior/articulation_proposal_attempt_terminal.json \
  --replacement-reason "The explicit provider payload was corrected."

content-workflow-cli articulation validate-attempt \
  --terminal-receipt path/to/replacement-attempt/articulation_proposal_attempt_terminal.json
```

## Common Workflows

For a drawer bank, review every prismatic candidate separately even when their
evidence and axes look similar.

## Output Format

The replacement-provider leaf produces
`ArticulationProposalProviderAttemptPublication`; its v2 proposal binds exact
provider, capability, and readable non-secret provider/service alias provenance
plus preparation/evidence/request digests, and its terminal receipt binds the
complete attempt or exact partial outputs after a storage/readback failure.
`publication_failure` identifies a local create-only storage/readback fault.
`ArticulationProposalProviderPublication` remains an import/deserialization-only
frozen compatibility shape; the current leaf does not emit it. Semantic review
still produces the complete
outer-owned canonical graph. See
`agentic/docs/articulation_preparation_attempt_contract.md`.

## Troubleshooting

Reject rather than promote unsupported or contradictory proposal facts. An
invoked-attempt failure with a written receipt exits 1 and prints its identity
as JSON on stderr. Exit 2 covers pre-invocation contract errors and
irrecoverable terminal-storage failures that leave a poisoned root;
interruptions preserve a terminal when storage permits and re-raise. Every
path stops without fallback. Preserve the failed root; a retry uses the
replacement flags above. Gather new evidence in a new run when immutable
inputs are wrong.
