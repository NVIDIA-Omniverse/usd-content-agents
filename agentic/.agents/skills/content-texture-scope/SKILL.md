---
name: content-texture-scope
description: Inspect an explicit USD material or prim scope, existing bindings, and UV readiness before texture generation. Use for bounded Texture workflow inspection and UV-preparation decisions.
version: "0.2.0"
author: NVIDIA Omniverse
tags:
  - content-agents
  - texture
  - inspection
tools:
  - Shell
  - Filesystem
compatibility: Requires the isolated Agentic workspace, the canonical USD interaction skill, and Python >=3.12.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - texture
    - inspection
---

# Texture Scope Inspection

Inspect only the request-bound material or prim scope and produce evidence for
the next Texture decision. This atomic skill never generates or applies images.

## When to Use

Use before candidate generation and whenever a revised unit needs targeted
reinspection of bindings or UV state.

## Limitations

- Do not expand beyond explicit material or prim roots.
- Do not treat valid UV syntax as proof of acceptable visual quality.
- Do not edit source USD or accepted output artifacts.

## Prerequisites

Load `usd-cli` or its canonical USD-tool successor. Freeze the source,
explicit target scope, reference roles, and requested operation selection.

## Instructions

1. Run provider-free `texture prepare` before constructing a Texture service or
   VLM assessor. The existing deterministic Texture planner may derive stable
   units; it must use the frozen explicit scope, not a provider proposal.
2. Inspect hierarchy, dependency closure, bindings, shader inputs, existing
   textures, and target membership for only the deterministic unit IDs.
3. Inspect UV primvars, interpolation, indices, seams, and target coverage.
4. Record source-render evidence and the renderer/tool metadata that produced
   it, bound to the exact source USD digest.
5. Bind every reference as `ROLE=PATH`. Preserve role, path, SHA-256, and size
   through preparation and the later outer plan.
6. Record all seven operation statuses. An unselected proposal, generator, or
   critique is `not_requested`, never `pass`; a selected leaf not yet called is
   `not_evaluated`.
7. If the frozen selection requests advisory planning, invoke `texture propose`
   separately. Do not request it merely to select targets.
8. Let the outer reasoner author the complete typed plan. Its appearance and
   generator inputs need not equal an advisory service proposal.

## Command Reference

Use `content-workflow-cli texture prepare` with explicit `--usd`, scope,
references, and `--request-*` selections. Invoke
`content-workflow-cli texture propose` only when `propose=requested`; its
provider ID and Texture service endpoint are mandatory arguments.

`texture _agent-step` remains a legacy Agentic compatibility adapter, not the
canonical focused boundary.

## Common Workflows

For refinement, inspect only exact unit IDs from the outer `revise` decision
and preserve every accepted unit. Persisted-observation target selection is
limited to the legacy `_agent-step` compatibility adapter.

## Output Format

Retain `capability_request.json`, `texture_preparation.json`, the deterministic
scope plan, inspection facts, dependency bindings, reference bindings, initial
renders, renderer metadata, and the explicit operation status table.

## Troubleshooting

If scope or UV evidence is missing, stop before mutation and gather a focused
canonical USD inspection rather than broadening the request.
