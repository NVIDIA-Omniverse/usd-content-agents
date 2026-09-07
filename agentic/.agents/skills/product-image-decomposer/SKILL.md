---
name: product-image-decomposer
description: Decompose product reference images into CAD-ready component crops, semantic part-plan evidence, exploded-view evidence, six-view prompts, and a reassembly manifest before geometry generation or reconstruction.
metadata:
  author: NVIDIA Omniverse
---

# Product Image Decomposer

Produce planning evidence, not final geometry. Preserve uncertainty and do not
invent hidden components.

## Workflow

1. Inventory every binding image and camera confidence. Record separate typed
   facts for `visual_silhouette`, `visual_topology` (including contiguous
   plate/ribbon vs connected frame vs separate-body connectivity), `visual_negative_space`,
   `visual_repetition` with exact `observed_count` or explicit ambiguity,
   `visual_boundary`, and
   `visual_depth_order`; keep uncertain hidden construction non-binding. For a
   repeated family, enumerate member centerlines in the clearest view and
   cross-check `N` members against `N-1` internal gaps before binding the count.
   Continue each centerline through foreground occluders and include partially
   visible boundary members. If plausible independent readings disagree, keep
   the count non-binding, set `observed_count` to null, record sorted
   `plausible_counts`, and name a bounded integer `count_parameter` for
   downstream visual selection. In maximum mode, preserve two independent
   source-only inventories and their deterministic consensus; do not let one
   pass silently erase the other's topology or count uncertainty.
2. Define a component tree with stable IDs. Split only where physical part,
   material, motion, or manufacturing boundaries justify it. Do not split a
   perforated plate into bars or fuse a visible skeletal frame into a slab. A
   rigid assembly may still contain separate manufactured members: preserve
   visible member terminations, butt joints, fastener seams, and material
   boundaries instead of adding hidden overlap merely to force one body.
3. Create padded crops and optional masks with
   `scripts/split_product_image.py`. Preserve the original image as binding
   context for every crop.
4. For each target leaf, write its local frame, scale evidence, silhouette
   landmarks, signature details, functional interfaces, material evidence,
   uncertainty, construction class, and `do_not_invent` list. Classify the
   dominant geometry as a constant-thickness planar profile, primitive
   prismatic/revolved form, swept section, lofted transition, connected-member
   assembly, or separate body. For a planar profile, identify the view normal
   to thickness, the complete outer material loop, and every same-frame void;
   do not decompose that loop into convenient primitive solids.
5. Produce a provider-neutral authoring brief per leaf: part names, bounded
   semantic controls, interfaces, required representation roles, and
   invariants. Every binding visual fact must name its source image and be
   linked to all planned features that implement it.
6. Produce an assembly plan with parent/child contacts, transforms, and typed
   joints. Every movable relation needs axis, origin, limits, and default.
7. Submit each component brief and crop to one explicitly selected
   `GeometryAuthoringProvider`. Require an immutable `geometry.source.v1`
   bundle in the component's local frame, then validate requested parameter
   variants, exported representations, and OVRTX evidence before reassembly.
8. Reassemble only accepted components. Validate support/contact, scale,
   symmetry, clearance, no floating parts, and min/mid/max joint poses.
9. Compare qualified deterministic projections and same-camera OVRTX renders
   against the original and crops. Deterministic projection is hard only at
   segmentation confidence `>=0.80` and camera-pose margin `>=0.03`; otherwise
   preserve it as advisory. Treat a
   feature that exists only in metadata but is hidden, contained, coplanar, or
   visually unreadable as missing geometry. Component
   acceptance requires silhouette, proportions, negative space, signature
   details, and material-zone plausibility; numeric checks alone are insufficient.

## Outputs

- `decomposition_report.json`
- original plus component crops/masks
- per-leaf provider-neutral authoring briefs and prompts
- assembly/transform/joint plan
- traceability from every visible signature item to a component or explicit
  deferral

Use a digest-bound provider revision request for source correction and
`content-workflow-geometry` for accepted geometry handoff.
