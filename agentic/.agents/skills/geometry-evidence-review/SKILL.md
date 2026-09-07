---
name: geometry-evidence-review
description: Gather and interpret geometry inspection, OVRTX render, topology, USD, projection, source-fidelity, and protected-feature evidence for one candidate. Use when an interactive session or workflow must decide whether geometry is reviewable, what additional evidence is needed, which defect is present, or whether to accept, refine, repair, or block a candidate.
metadata:
  author: NVIDIA Omniverse
---

# Geometry Evidence Review

Use this atomic skill to make one evidence-backed geometry decision. Typed
inspection, render, and validation modules produce facts; this skill teaches
which facts to gather and how to interpret them. It does not mutate geometry or
declare the enclosing workflow complete. Ownership is atomic: this skill does
not own run state, evidence acceptance, artifact lifecycle, or completion. The
calling workflow owns those contracts.

## Procedure

1. Bind the candidate digest, source-bundle and representation digests,
   parameter row, request, and reference evidence before inspection.
2. Gather deterministic evidence first: loadability, extents, components,
   manifold/topology state, openings, interfaces, protected features, semantic
   parts, artifact bindings, and validator availability.
3. Use `usd-cli` for interactive scene inspection, picking, isolation,
   normal/depth channels, and reversible previews. Use the shared Geometry
   inspection and USD validation contracts for batch evidence.
4. Request OVRTX evidence through the typed render operation. Use individual
   digest-bound views as evidence; treat a six-view grid as presentation only.
   Reject blank, clipped, distorted, fallback, wrong-camera, or artifact-mismatched
   evidence before judging shape.
5. Compare the request and references in this order: primary silhouette,
   negative space, part count and ownership, interfaces and affordances,
   relative dimensions, continuity, repetition, then secondary detail.
6. Separate three root causes:
   - geometry defect: send a bounded revision to the selected external
     authoring provider, or route imported geometry to
     `content-workflow-geometry-repair`;
   - evidence defect: request a corrected view, channel, crop, or validator;
   - underdetermined request: retain the raw result, expose the missing or
     conflicting fact, and forbid selection from hidden reference geometry.
7. Return exactly one recommendation: `accept`, `refine`, `repair`,
   `more_evidence`, or `blocked`, with every finding linked to evidence.

## Evidence Rules

- File existence is not validation; consume the typed result and digest.
- A local preview is not accepted OVRTX evidence.
- A successful build does not prove prompt fidelity or realistic topology.
- An unavailable validator is `not_evaluated`, never pass.
- Do not use this skill as the independent benchmark scoring oracle for a skill
  that generated the candidate.
- Do not repair camera or evidence defects by changing correct geometry.

## Return

Return candidate and evidence digests, evidence sufficiency, findings with
source paths or view IDs, unevaluated checks, the recommendation, and the next
atomic skill or workflow. The parent workflow owns promotion and completion.
