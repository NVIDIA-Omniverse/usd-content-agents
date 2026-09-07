---
name: content-articulation-authoring
description: Author only approved articulation candidates, then validate saved-stage identity and joint-graph readback. Use after agent and required human review are durably bound.
version: "0.1.0"
author: NVIDIA Omniverse
tags:
  - content-agents
  - articulation
  - authoring
  - validation
tools:
  - Shell
  - Filesystem
compatibility: Requires the canonical Articulation finalizer, Joint Rigger owned_core, and Python >=3.12.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - articulation
    - authoring
    - validation
---

# Approved Articulation Authoring

Continue deterministic accepted-only authoring and verify the exact saved USDZ
joint graph. The workflow code, not the reasoning agent, performs mutation.

## When to Use

Use only after the agent decision ledger and any configured human review
receipt validate against the same candidate evidence.

## Limitations

- Author exactly once and only accepted native-ready candidates.
- Do not author masses or colliders in the articulation stage.
- Render evidence cannot replace saved-stage identity and graph readback.

## Prerequisites

Verify the decision ledger, reviewed candidate document, review receipt when
required, and approved candidate document before continuing.

## Instructions

1. Let the deterministic controller build the accepted-only authoring request.
2. Verify authoring result IDs match approved IDs exactly.
3. Reopen the saved artifact and validate source identity and complete joint
   graph readback before rendering.
4. Render focused validation evidence for the authored result.
5. If validation fails, retain accepted work and route only failed candidate
   scope through targeted reinspection in a new run or supported refinement.
6. Finalize `completed` only for exact readback; otherwise retain a conditional
   result with unresolved IDs and evidence.

## Command Reference

After an agent patch with requested review policy `none`, standalone runs use
`_agent-apply` to continue deterministic authoring. An Asset
`articulation.author.v1` leaf instead re-runs its same opaque
`content-workflow-cli articulation agentic-leaf author --invocation ...`
entrypoint with the exact immutable invocation. Do not import an internal
Python API or create or execute a Python helper to cross either decision pause.
Human-gated runs continue with the public `articulation review` command.

## Common Workflows

In a composed asset workflow, the outer coordinator remains the only reasoning
loop and re-invokes the selected Asset leaf's opaque entrypoint; it does not
substitute the standalone prepare/apply surfaces.

## Output Format

Retain approved candidates, receipt, authoring request/result, saved USDZ,
validation evidence, checkpoint, progress, trace, and final summary.

## Troubleshooting

On graph mismatch or artifact drift, keep the run for diagnosis and do not
reauthor over immutable evidence. Start a new run after correcting inputs.
