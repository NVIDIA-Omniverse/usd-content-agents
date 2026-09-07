---
name: content-texture-candidate
description: Generate, compare, and preview one bounded texture candidate set for exact texture-unit IDs. Use for initial generation or failed-unit regeneration without unrelated mutation.
version: "0.3.0"
author: NVIDIA Omniverse
tags:
  - content-agents
  - texture
  - generation
tools:
  - Shell
  - Filesystem
compatibility: Requires the isolated Agentic workspace, Python >=3.12, and either a configured generator leaf or exact outer-provided PNG artifacts.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - texture
    - generation
---

# Texture Candidate Generation

Generate or regenerate candidate artifacts for exact plan unit IDs. Embedded
generation is non-mutating: it cannot publish or replace the canonical asset.

## When to Use

Use after provider-free preparation and an outer-authored plan selects
`generate`. For refinement, use only exact outer `revise` unit IDs.

## Limitations

- A remote Texture Agent may be a leaf backend, never the workflow controller.
- Provider generation and deterministic supplied-image apply are mutually
  exclusive. Neither mode falls back to the other.
- Reviewed unit artifacts are immutable preserved inputs during refinement.
- Generation cannot mutate geometry, physics, collision, or unrelated material state.
- A leaf backend cannot change the outer-accepted targets, appearance,
  generator inputs, preservation constraints, or acceptance criteria.

## Prerequisites

Load `texture_preparation.json`, the exact outer plan, and any explicitly
requested advisory proposal. A proposal is not required for other generator
leaves.

## Instructions

1. Verify `generate=requested`; reject `not_requested` rather than silently
   generating.
2. Bind the outer plan's exact targets, semantic intent, reference identities,
   and generator inputs. Do not replace them with advisory proposal values.
3. Select exactly one execution mode. For `provider_generate`, select the
   provider/capability explicitly and record its provenance. For
   `apply_provided`, bind one ordered `outer_generated_candidate` albedo PNG to
   each exact unit, including its producer capability and invocation
   provenance.
4. Invoke only the selected leaf. `apply_provided` performs deterministic USD
   authoring and must not call Texture Agent, simple image generation, a model,
   or another provider.
5. Persist the generation result, candidate asset, unit artifacts, and digests
   before deterministic evidence collection.
6. Preserve reviewed artifacts byte-for-byte during bounded revision.
7. Stop after generation. Do not render, critique, publish, or update the
   canonical asset.

## Command Reference

Run `content-workflow-cli texture generate --preparation ... --outer-plan ...`
with an explicitly exposed generator provider, or run
`content-workflow-cli texture apply-provided --preparation ... --outer-plan ...`
for outer-created images. The bundled Texture-service adapter opens an
execution session from the deterministic scope and exact outer generator
inputs. Pass the optional `--provider-proposal` to either command only when
`propose=requested`; it remains advisory and never drives mutation. The remote
service adapter requires a byte-self-contained source and fails closed when the
prepared USD has an external dependency closure it cannot upload. Typed Python
adapters can implement `TextureGeneratorLeaf`, while the public CLI supports
only its concrete named adapters. Do not claim arbitrary adapters are callable
until they are explicitly exposed.

## Common Workflows

After `refine`, route immediately to `content-texture-quality`; never replay the
initial plan or reviewed candidates. Resume must reuse the exact persisted
candidate/result and cannot call the generator again.

## Output Format

Retain `texture_generation.json`, the exact outer-plan and optional-proposal
bindings, execution mode, generator/apply capability, generator inputs,
candidate binding, per-unit artifacts, references, outer-provided image
bindings and producer provenance, and operation statuses.

## Troubleshooting

If the provider returns unknown, missing, duplicated, or reordered unit IDs, or
if supplied images are missing, duplicated, stale, symlinked, reordered,
scope-mismatched, or not exact-size PNG albedo artifacts, stop. Do not translate
display names into plan-unit identities or treat references as candidates.
