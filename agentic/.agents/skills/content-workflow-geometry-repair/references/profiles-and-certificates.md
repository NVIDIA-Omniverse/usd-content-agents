# Profiles And Certificates

## Initial Profiles

| Profile | Render geometry | Collision geometry | Mandatory geometry gate |
| --- | --- | --- | --- |
| `visual_only` | Sheets permitted | None | Parse, finite geometry, preserved hierarchy/material identity |
| `static_environment` | Intentional sheets permitted | Static triangle mesh | Authored collider plus authoritative real `ovphysx` cook-and-contact canary |
| `rigid_pick_place` | Watertight exterior and preserved affordances | Bounded primitive/single hull when accurate, otherwise pinned seeded CoACD | Watertightness, collider volume/hull/vertex budgets, protected probes, CoACD warning review, and four-pose real `ovphysx` cook/drop/settle evidence |
| `articulated_rigid` | Per-link render mappings | Per-link collision mappings | Caller-supplied link geometry and sampled sweep evidence; complete profile remains conditional |
| `contact_rich` | Accepted task geometry | Explicit receiver collision paths | Caller-supplied axis, path, clearance, stop, and applicable SDF-resolution evidence; complete profile remains conditional |

`articulated_rigid` and `contact_rich` now emit geometry-only advanced-profile
evidence. Do not emit a complete-profile `certified` outcome for either: route
joint authoring, physical properties, target-runtime behavior, and SimReady
conformance downstream. `deformable_or_cae` remains reserved and fail-closed
until element-quality and solver gates exist.

## Drift Bands

- `identity`: no geometry change.
- `conservative`: p99 sampled surface drift at most `0.001 * bbox_diagonal` and
  volume drift at most 1%.
- `moderate`: collision-only or explicitly lossy output, p99 at most 0.5% and
  volume drift at most 3%.
- `reconstructive`: explicit opt-in, p99 at most 1% and volume drift at most
  5%; never silently replace source-preserved render geometry.

Real-unit protected clearances override percentage thresholds.
Threshold selection alone does not measure a protected opening or clearance;
without an explicit collision-space probe, that protection remains
`conditional` even when CoACD and `ovphysx` succeed.

## Escalate

Reject or remain conditional when units/profile are uncertain, a part could be
deleted or fused, a protected opening could close, indexed attributes cannot be
remapped, local and global repairs disagree, or all candidates exceed drift or
collision budgets.

Certification additionally requires a caller-confirmed profile. Production
use requires source URI, license, and provenance. An unavailable official
source-format validator remains `not_evaluated`. Any accepted render-geometry
change requires source-digest-bound OVRTX evidence from the shared Geometry
renderer plus recorded human review; numeric fidelity checks alone do not clear
that review.

Geometry certification does not cover material assignment, density, mass,
inertia, friction, restitution, joints, drives, controls, or complete SimReady
conformance.

`fake` runtime evidence is deterministic test evidence only. It cannot produce
a production `certified` outcome.
