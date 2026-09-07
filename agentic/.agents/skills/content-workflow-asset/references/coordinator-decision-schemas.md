# Coordinator Decision Schemas

Use these shapes when the asset coordinator, rather than a nested domain agent,
authors Material and Physics decisions. Copy exact candidate/component paths and
digests from prepared artifacts; do not invent identifiers.

## Material

```json
{
  "schema_version": "content-agents.material-decision-patch.v1",
  "material_assignments": [{
    "family": "drawer fronts",
    "material_name": "exact palette name",
    "material_path": "/Looks/ExactPaletteMaterial",
    "runtime_prim_paths": [],
    "source_prim_paths": [],
    "prim_paths": ["/Asset/Drawer"],
    "rationale": "observable prompt, palette, and render evidence"
  }],
  "reviewed_no_override": [],
  "final_review_issues_found": [],
  "final_review_issues_fixed": [],
  "final_review_notes": "short evidence summary",
  "visual_quality_assessment": {
    "status": "pass",
    "checked_views": ["/absolute/run/evidence_renders/initial.png"],
    "reference_images": [],
    "reference_files": [],
    "issues_found": [],
    "issues_fixed": [],
    "unresolved_issues": [],
    "assessment_notes": "short observable assessment"
  }
}
```

`checked_views` must exactly reproduce every path in the preparation's
`initial_render_bindings` plus every frozen reference image/file path. This
includes the initial beauty, camera, OVRTX response, segmentation, legend, and
segmentation-response evidence; an arbitrary child-created in-run image is not
review evidence. Use
`reviewed_no_override` only when the frozen request explicitly preserves
existing bindings. Inspect the deterministic final renders after finalization
and refine instead of accepting contradictory evidence.

### Material post-apply review

After `material_coordinator finalize` returns `review_required`, inspect the
exact final OVRTX artifacts and copy every `final_render_bindings` entry from
`raw/material_application_receipt.json` into this separate patch. Do not author
future paths before finalization and do not omit turntable frames or the GIF.

```json
{
  "schema_version": "content-agents.material-post-apply-review.v1",
  "status": "pass",
  "checked_views": [
    "/absolute/run/final_renders/final_top.png",
    "/absolute/run/final_renders/final_turntable.gif"
  ],
  "checked_view_bindings": [{
    "path": "/absolute/run/final_renders/final_top.png",
    "sha256": "<exact receipt digest>",
    "size_bytes": 123
  }, {
    "path": "/absolute/run/final_renders/final_turntable.gif",
    "sha256": "<exact receipt digest>",
    "size_bytes": 456
  }],
  "issues_found": [],
  "issues_fixed": [],
  "unresolved_issues": [],
  "assessment_notes": "Reviewed every exact post-apply render binding."
}
```

The abbreviated example shows two entries only; the real patch must reproduce
the entire receipt binding set exactly. Use `unresolved_issues` only with status
`unresolved_issues`; a conditional review cannot be accepted as a clean pass.

## Physics decision patch

```json
{
  "schema_version": "content-agent-workflows.physics-decision-patch.v2",
  "asset": "/absolute/path/to/exact-texture-handoff.usdz",
  "source_digest": "sha256:<dependency-aware digest of the exact stage input>",
  "decisions": [{
    "decision_id": "stable-id",
    "component_id": "component_001",
    "body_root_path": "/Asset/Body",
    "visual_evidence_paths": ["/Asset/Body/Visual"],
    "collider_paths": ["/Asset/Body/Collision"],
    "collision_mode": "preserve_existing",
    "mass_authoring_path": "/Asset/Body",
    "inferred_material_family": "wood",
    "inferred_material_name": null,
    "collision_approximation": "convexHull",
    "physical_properties": {
      "density": 700.0,
      "estimated_mass_kg": 10.0,
      "static_friction": 0.5,
      "dynamic_friction": 0.4,
      "restitution": 0.1
    },
    "confidence": 0.8,
    "rationale": "component, material, bounds, and topology evidence"
  }],
  "unresolved_components": []
}
```

Cover each exact component once in either `decisions` or
`unresolved_components`. Composed acceptance requires no unresolved components.
Compute `source_digest` with `physics_topology.sha256_file` on the exact Texture
handoff before topology repair. Do not copy the post-repair digest from
`raw/physics_components.json`.

## Physics topology plan

Write this only when the frozen intent requires an allowed topology repair:

```json
{
  "schema_version": "content-workflows.physics-topology-plan.v1",
  "expected_source_digest": "sha256:<dependency-aware digest of the exact texture handoff>",
  "mobility_intent": "movable",
  "operations": [{
    "op": "ensure_rigid_body_api",
    "prim_path": "/Asset/Body"
  }],
  "joint_endpoint_owner_promotions": [{
    "joint_prim_path": "/Asset/Joints/DrawerJoint",
    "relationship": "body1",
    "relationship_target_path": "/Asset/Drawer/Frame",
    "requested_rigid_body_ancestor_path": "/Asset/Drawer"
  }],
  "invariants": {
    "enabled_collider_count": 1,
    "reject_articulation_changes": true
  }
}
```

`mobility_intent` is `preserve`, `movable`, or `static`. Operations are limited
to `ensure_rigid_body_api`, `remove_rigid_body_api`, and `remove_fixed_joint`;
`preserve` forbids removals. The applied report must repeat each plan operation,
reject no operation, preserve collider count, and reject articulation changes.
When an ensure operation changes the resolved owner of a non-fixed joint endpoint,
list every affected relationship in `joint_endpoint_owner_promotions`. Each entry
binds one existing single relationship target to either that same path or a strict
ancestor. While this allowlist is present, every ensure operation must be covered;
the native report records each verified before/after owner change. The relationship
target and joint prim/type/axis/limit signatures remain immutable.
When an ensured body is nested below another enabled body, the executor preserves
its world transforms and annotates that applied operation with
`reset_xform_stack: preserve_world`; this annotation is report-only.
