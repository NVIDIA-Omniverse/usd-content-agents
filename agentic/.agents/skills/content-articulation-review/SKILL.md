---
name: content-articulation-review
description: Select, reject, or safely edit every articulation candidate through a digest-bound decision patch. Use before human review or accepted-only authoring.
version: "0.1.0"
author: NVIDIA Omniverse
tags:
  - content-agents
  - articulation
  - review
tools:
  - Shell
  - Filesystem
compatibility: Requires an Articulation step observation and Python >=3.12.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - articulation
    - review
---

# Articulation Candidate Review

Bind one complete ordered candidate decision to the exact request, source,
dependency bundle, candidate document, usd-cli evidence, and checkpoint.

## When to Use

Use after inspection/proposal and before any authoring or human review receipt.

## Limitations

- Every candidate ID must appear exactly once and in canonical order.
- Evidence must be observation-bound and remain inside the run directory.
- An agent patch never bypasses a configured human review gate.
- A human `revise` decision never permits in-place graph edits or authoring.

## Prerequisites

Load `content-articulation-inspection` and
`content-articulation-proposal`. Read the current observation and schema.

## Instructions

1. Copy every digest and checkpoint revision from the current observation.
2. Decide `accept` or `reject` for each exact candidate.
3. Complete-candidate edits may change only `confidence`, `parent_hint`,
   `child_hint`, `component_name`, `component_type`, or the explanatory
   `evidence` text. Candidate identity, topology, axis, limits, readiness, and
   every Joint-owned declared or pass-through provenance/evidence field are
   immutable. Include concrete bound evidence paths.
4. Write the patch to the observation's `decision_patch_path`.
5. Apply it through the focused command selected by the active workflow and
   respect `needs_review` when the requested policy requires human approval.
   In an Asset `articulation.author.v1` leaf, the first opaque entrypoint call
   returns `content-agent-workflows.articulation-author-leaf-progress.v1`.
   After writing the patch to its prescribed path, call that same
   `content-workflow-cli articulation agentic-leaf author --invocation ...`
   entrypoint again with the exact immutable invocation bytes. Do not call
   `_agent-apply`, import an internal Python API, or create or execute a Python
   helper in this Asset-leaf pause.

## Command Reference

For a standalone `content-workflow-cli articulation run`, use
`content-workflow-cli articulation _agent-apply --run-dir <run>
--decision-patch <patch>`.

For an Asset `articulation.author.v1` leaf, re-run the descriptor's same opaque
`content-workflow-cli articulation agentic-leaf author --invocation <invocation>`
entrypoint. It validates and durably binds the prescribed outer patch inside
the active leaf attempt.

## Common Workflows

With review policy `all` or `uncertain`, stop after the agent ledger is bound;
the asset owner supplies the separate human receipt.

For an embedded canonical graph, the human receipt may use `revise` for exact
candidate IDs. Preserve the parent graph and review bytes, author a
`content-agent-workflows.embedded-articulation-graph-revision-patch.v1` that
binds the parent state revision, identity/evidence/proposal digests, parent
graph file and semantic digests, human decision and complete decisions file,
reviewer, reason, timestamp, exact supported field changes, and revised graph.
Supported changes are joint type, axis, lower/upper limit, limit unit, and frame
policy; topology, membership, coverage, candidate identity, and evidence remain
immutable. Apply it with `content-workflow-cli articulation revise-graph`, then
stop for a complete new human decision over every ID at the revised graph
digest. Never reuse the parent acceptance.

## Output Format

Use `content-agent-workflows.articulation-decision-patch.v1`. The deterministic
step publishes the patch, reviewed candidate document, and immutable ledger.

## Troubleshooting

Stale revision, digest drift, missing IDs, unsafe edits, and unbound evidence
fail before authoring. Rebuild from the latest observation; never edit state.
