# Geometry Evidence Contract

## Status Semantics

- Geometry-only evidence uses `T1_basic_stability`.
- Optional skipped runtime and SimReady checks remain `not_evaluated` and do
  not degrade otherwise valid geometry.
- `optimization_unavailable` means the output is a normalized source copy,
  not optimized geometry.
- Temporary runtime proxy success is conditional loadability evidence only.
- SimReady Foundation `BLOCKED` maps to `sim_ready_status=not_evaluated`.
- Only formal Foundation PASS or FAIL sets formal SimReady status.

## Accepted Producers

| Evidence | Producer |
| --- | --- |
| Source conversion | `content_agent_workflows.convert_to_usd` through `content-workflow-cli convert-to-usd`, retaining its probe, validation, reports, manifest, and output-USD handoff |
| Scene optimization | Workflow-owned `OptimizeUSDTask` plus correspondence metadata and source-space projection; use usd-cli for inspection and final low-level edits without routing workflow policy through the low-level tool |
| Final render views | Geometry `render_geometry_evidence` through `world_understanding.validation.usd_rendering.render_usd_visual_evidence`, backed by explicit local OVRTX or a remote/configured backend where the workflow observes and retains protocol evidence from the exact render endpoint identifying OVRTX and proving readiness plus a successful no-redirect render; endpoint userinfo/query/fragment are rejected, automatic `NGC_API_KEY` use is limited to canonical HTTPS NVCF hosts, custom bearer secrets are explicit and endpoint-scoped, unauthenticated use requires loopback or explicit operator trust, and the audited individual image must be named by that report; generic `world_understanding.validation.usd_rendering` remote transport without that protocol evidence is insufficient |
| Image health | `world_understanding.functions.graphics.render_validation` |
| Generic USD validation | Configured scene backend or `world_understanding.functions.graphics.validate_usd` |
| Runtime | `content_agent_workflows.runtime_validation` with the configured solver; the current OvPhysX path runs on CPU, while OVRTX/GPU is required separately for rendered review of its recording |
| Formal SimReady | `content_agent_workflows.simready` / Foundation |
| CAD source and product checks | Geometry `cad_preflight` and asset audit |

Direct usd-cli renders are interactive diagnostics unless the caller binds both the
render result and shared validation to the exact source USD digest. Batch/final
Geometry evidence always uses the digest-bound producer listed above.
Its `geometry_render_evidence_<preset>.json` report records the source USD path,
its SHA-256 before and after rendering, and an `image_bindings` entry containing
the image SHA-256 and source USD SHA-256 for every retained view and presentation
image. A missing or changed source or image fails the evidence instead of
silently accepting a stale pairing.
When the source is already unavailable before rendering, the failed report
records a null source digest plus the blocking issue; it never invents a digest
or invokes the renderer.

## Bundle

`geometry_evidence_bundle.json` is a composition index. Every entry records:

- artifact kind and absolute path;
- SHA-256 digest;
- producing shared system;
- original status and normalized severity;
- schema version when present;
- exact claim scope.

Do not collapse upstream reports into one invented validator result. Preserve
the original report and severity even when Geometry applies a documented
nonblocking policy, such as treating ASCII storage performance as a warning.

The workflow layer always owns the run directory and durable trace. Scene-tool
history, checkpoints, renders, and raw reports may contribute low-level
evidence, but they never replace the request, decisions, validation, final
output, failure state, and resumable workflow artifacts.
