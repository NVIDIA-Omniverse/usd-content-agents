---
name: content-texture-publish
description: Validate a Texture decision chain and publish a portable scoped USD result. Use after all bounded generation and visual-review steps reach a terminal decision.
version: "0.2.0"
author: NVIDIA Omniverse
tags:
  - content-agents
  - texture
  - publication
tools:
  - Shell
  - Filesystem
compatibility: Requires the isolated Agentic workspace, canonical Texture finalizer, and Python >=3.12.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - texture
    - publication
---

# Texture Publication

Finalize only a digest-bound Texture decision chain and verify portable output
dependency closure before publishing canonical artifacts.

## When to Use

Use only when the frozen selection requests `publish` and an outer review has
accepted every exact candidate unit. Fixed-pipeline finalization remains
explicit.

## Limitations

- Publication cannot change decisions, plan scope, or accepted artifacts.
- Embedded deterministic publication is the sole
  `execution_effect=mutation` operation in the Texture stage attempt.
- Schema/readback success cannot override visual rejection.
- Conditional output must retain unresolved exact unit IDs.
- Source assets remain immutable.

## Prerequisites

The request, preparation, outer plan, generation, candidate evidence, outer
review, source closure, reference bindings, and accepted candidate must all be
available at their exact recorded digests.

## Instructions

1. Verify every unit has an outer `accept` receipt for the exact persisted
   candidate generation result and output digest.
2. Reject publication when `publish=not_requested`, any review is reject/revise,
   or an optional leaf is mislabeled as passing.
3. Recheck request, source bytes and dependency closure, reference bytes,
   deterministic scope, generator result, evidence, accepted candidate, and
   non-target preservation before copying.
4. Publish once, then reopen the saved stage and repeat deterministic scope
   readback. Require the saved bytes to equal the accepted candidate bytes.
5. Write the publication validation and receipt binding every upstream packet,
   exact references, accepted candidate, published asset, and status table.

## Command Reference

Run `content-workflow-cli texture publish` with the exact request, preparation,
outer plan, generation, evidence, outer-review paths, and a new output path.
The legacy `_agent-step` finalizer remains available to fixed-pipeline callers.

## Common Workflows

Standalone and embedded workflows use the same patch and finalizer. Only the
standalone wrapper launches a child.

## Output Format

Retain `publication_validation.json`, `texture_publication_receipt.json`, the
portable asset, and every upstream packet the receipt binds.

## Troubleshooting

Any stale patch, missing dependency, digest drift, or out-of-root path fails
before publication. Start a new run for changed inputs.
