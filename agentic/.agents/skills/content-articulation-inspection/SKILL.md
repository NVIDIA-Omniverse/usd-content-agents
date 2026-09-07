---
name: content-articulation-inspection
description: Inspect bounded USD hierarchy, existing articulation, and usd-cli evidence before proposing joints. Use for exact Articulation candidate preparation and targeted reinspection.
version: "0.1.3"
author: NVIDIA Omniverse
tags:
  - content-agents
  - articulation
  - inspection
tools:
  - Shell
  - Filesystem
compatibility: Requires the isolated Agentic workspace, usd-cli, and Python >=3.12.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - articulation
    - inspection
---

# Articulation Inspection

Inspect geometry, hierarchy, existing joints, properties, picks, and focused
renders without authoring or changing the source asset.

## When to Use

Use before candidate proposal and after validation identifies a specific joint
or moving-part scope that needs more evidence.

## Limitations

- Never mutate the source or infer joints from labels alone.
- Inspect only the request-bound asset and candidate moving-part scope.
- A preview render is diagnostic; saved-stage readback is final evidence.

## Prerequisites

Load `usd-cli` and the frozen Articulation request. Preserve exact retained
artifact bindings throughout the run. The coordinator does not derive the
package-aware dependency-bundle identity.

## Instructions

1. Inspect hierarchy, xforms, bounds, mesh topology, and existing joint graph.
2. Capture request-bound properties, picks, cameras, and focused renders.
3. Verify fixed-parent and moving-part paths are distinct and present.
4. Record exact evidence paths and digests for proposal/review.
5. For provider-neutral preparation, persist one complete typed
   `ArticulationPreparationInspectionReadbackDraft` with schema
   `content-agent-workflows.articulation-preparation-readback-draft.v2`: exact
   retained source and dependency claims, inspector configuration and
   implementation, saved stage, lexical hierarchy parents, per-prim
   membership/owner observations, renders, and scene records. Omit
   `source_dependency_bundle_sha256` and
   `saved_stage_dependency_bundle_sha256`; do not calculate, guess, copy, or
   recover either value from an error. The trusted publisher independently
   derives and seals both package-aware identities. The v1 readback is only a
   compatibility input for a trusted producer that already owns both exact
   identities. The retained inspector configuration is the exact
   membership authority; its rows must cover the saved hierarchy and equal the
   readback observations, and every referenced authoritative owner must be
   self-owned. Set `proposal_status` to `not_evaluated` only when the outer
   policy selected a proposal provider; otherwise retain `not_requested`. Do
   not hand-author separate caller member or owner lists. When this preparation
   feeds the public standalone/asset Articulation author leaf, the retained
   inspector configuration must set
   `capabilities.canonical_output_evidence_required` to `true`. This declares
   the required post-author evidence policy; it does not supply a visual
   envelope, render the pre-author stage, or change the author-before-canonical-
   render dependency. Save the inspected stage directly into the retained leaf
   directory with `usd-cli-tel --json save <retained-path>.usda --flatten`,
   verify that exact regular file and digest, and bind it as `saved_stage`.
   Never bind a path returned by `checkpoint save`: parent-owned `.usd-cli`
   checkpoint state is session-ephemeral and may be removed before the
   create-only preparation publisher captures it.
6. Stop on truncated, missing, or cross-run evidence.

## Command Reference

`content-workflow-cli articulation _agent-prepare --run-dir <run>` performs
deterministic inspection, inference, and evidence collection, then pauses.

## Common Workflows

On targeted reinspection, retain accepted candidate decisions and collect new
evidence only for the failed candidate scope.

## Output Format

Use the request, candidate document, usd-cli manifest, exact candidate IDs,
and evidence digest map named by `articulation_agent_observation.json`. The
provider-neutral leaf additionally consumes the typed saved readback described
in `agentic/docs/articulation_preparation_attempt_contract.md`.

## Troubleshooting

If a prim path or evidence digest does not match, stop before proposal review;
do not repair identity by editing an artifact in place.
