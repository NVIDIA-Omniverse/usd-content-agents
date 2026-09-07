---
name: content-texture-quality
description: Collect matched Texture evidence, optionally request advisory critique, and record the outer multimodal review. Use after explicit candidate generation.
version: "0.2.0"
author: NVIDIA Omniverse
tags:
  - content-agents
  - texture
  - validation
tools:
  - Shell
  - Filesystem
compatibility: Evidence requires the canonical USD rendering surface; advisory critique requires an explicitly configured visual assessor; Python >=3.12.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - texture
    - validation
---

# Texture Quality Review

Collect fresh paired source/candidate evidence for the exact target set, then
let the outer multimodal reasoner decide accept, reject, or revise. Domain VLM
critique is an independently selected advisory leaf.

## When to Use

Use after every initial or refined candidate application.

## Limitations

- Semantic claims require actual rendered evidence.
- Validation cannot alter the output asset.
- Validator/VQA findings cannot directly partition canonical accepted and
  remaining IDs.
- Every disposition must be authored in an outer review artifact bound to the
  exact candidate result and fresh evidence.

## Prerequisites

The frozen operation selection must request `evidence` and `review`, and the
generation packet must bind the output asset. `critique` may be requested or
explicitly `not_requested`.

## Instructions

1. Render every matching selected and canonical view through the shared OVRTX
   USD path, preserving the exact source USD digest and OVRTX render metadata
   with each evidence record.
2. Verify unchanged geometry, bindings outside scope, source dependency
   identity, candidate closure, target UV/material state, and reference bytes.
3. Stop after `texture evidence`. It must report semantic assessment as
   `not_evaluated`, not pass.
4. Only when selected, run `texture critique` with explicit VLM backend and
   model. Treat findings as advisory evidence, never the semantic decision.
5. Have the outer model directly inspect every exact reference, then every
   outer-provided image, then every matched source/candidate image, in that
   exact order. Author one `accept`, `reject`, or `revise` for every exact
   unit, binding the candidate result/output digest and evidence digest.
6. Record that decision with `texture review`. Fail closed on visual rejection,
   stale evidence, missing units, or candidate
   substitution. Only explicit outer `revise` units may be regenerated.
7. Stop when all units have outer acceptance or the bounded budget is exhausted.

## Command Reference

Run `content-workflow-cli texture evidence` first. If selected, run
`content-workflow-cli texture critique` with explicit provider configuration.
Finally author `texture-outer-review-input.v1` and run
`content-workflow-cli texture review`.

## Common Workflows

If one unit fails, load `content-texture-scope` and
`content-texture-candidate` for only that unit; do not replay successful work.

## Output Format

Retain `texture_candidate_evidence.json`, optional `texture_critique.json`, and
`texture_outer_review.json` with paired renders, references, renderer metadata,
static findings, provider provenance, candidate identity, and outer dispositions.

## Troubleshooting

Reject validation output that omits requested IDs, adds IDs, changes order,
references a different asset, or reports a stale iteration.
