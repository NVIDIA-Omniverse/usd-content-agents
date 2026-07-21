# SimReady Profile Conformance

Use profile conformance to route failing SimReady requirements to the
appropriate Foundation FET conformance skill or helper. Conformance works on
staged outputs and must not silently mutate the source asset.

Command shape:

```bash
content-workflow-simready-conform-profile asset.usda \
  --output-dir simready-conform \
  --validation-report simready-profile.json \
  --profile Prop-Robotics-Neutral \
  --profile-version 1.0.0 \
  --report simready-conform/simready-conform-profile.json
```

`G3A.HYG.001` is restricted to generated Joint Agent physics assets. Its caller
must supply the trusted inventory fingerprint captured before hygiene; the
installed command and wrapper use the same option:

```bash
content-workflow-simready-conform-profile asset.usda \
  --output-dir simready-conform \
  --repair G3A.HYG.001 \
  --expected-physics-inventory-sha256 "$PHYSICS_INVENTORY_SHA256"
```

Do not derive this value from a physics-stripped fallback or substitute source
asset. Obtain it from `inspect_gate3a_physics_inventory(...)` on the trusted
Joint Agent output that the conformance request is authorized to modify.

The conformance report records:

- input and output USD paths;
- selected profile and version;
- failed, repaired, blocked, and skipped requirements;
- the Foundation FET skill selected for each routed requirement;
- per-step reason, status, and handoff metadata;
- next step, usually formal profile validation.

Automatic repairs must be mechanical, staged, and locally verifiable. If repair
requires user intent, source data, visual judgement, material identity, mass
policy, joint semantics, texture edits, or unsupported package mutation, report
the requirement as blocked and hand off to the matching workflow or user.

Common routing:

| Requirement area | Foundation skill |
| --- | --- |
| Core naming, metadata, package layout | `simready-foundation-conform-fet-000-core` |
| Units and minimal/base-neutral USD | `simready-foundation-conform-fet-001-minimal` |
| Rigid body physics | `simready-foundation-conform-fet-003-rigid-body-physics` |
| Multibody physics | `simready-foundation-conform-fet-004-simulate-multi-body-physics` |
| Grasp vector | `simready-foundation-conform-fet-005-simulate-grasp-physics` |
| Visual materials | `simready-foundation-conform-fet-006-materials` |
| Nonvisual materials | `simready-foundation-conform-fet-007-nonvisual-materials` |
| Robot core | `simready-foundation-conform-fet-021-robot-core` (explicit user request only) |
| Robot materials | `simready-foundation-conform-fet-023-robot-materials` (explicit user request only) |
| Base articulation | `simready-foundation-conform-fet-024-base-articulation` (explicit user request only) |
