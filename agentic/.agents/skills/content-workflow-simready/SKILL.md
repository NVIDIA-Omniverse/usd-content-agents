---
name: content-workflow-simready
description: Use after USD conversion or Joint Agent publication to run SimReady Foundation profile conformance, formal Gate 3B validation, and non-blocking remediation reporting for staged USD or USDZ assets.
---

# content-workflow-simready

Use this skill after an asset has a meaningful USD-family output and the user
wants SimReady profile conformance or formal SimReady Foundation validation.
Run material and physics workflows first unless the user explicitly asks for a
validation-only inspection of the current USD.

This skill owns SimReady profile selection, Foundation toolchain preflight,
staged conformance routing, formal static profile validation, and remediation
handoff. It does not own CAD conversion, material prediction, physics
prediction, texture generation, joint inference, Workbench scene mechanics, or
runtime simulation.

## Normal Order

1. Convert the source asset to USD when needed with `content-workflow-convert-to-usd`.
2. Run `content-workflow-material` against the latest USD unless the user
   explicitly supplied an already-authored material result or asked for
   validation-only behavior.
3. Run `content-workflow-physics` against the latest USD unless the user
   explicitly supplied an already-authored physics result or asked for
   validation-only behavior.
4. Run any other requested authoring workflows such as joint, texture, or
   geometry.
5. Run SimReady Foundation preflight.
6. Run `simready-conform-profile` when there are known profile failures or the
   workflow requested staged conformance.
7. Run `simready-validate` against the latest staged USD.
8. If validation fails after a meaningful USD exists, record conditional status,
   rerun reasons, repair hints, and the next conformance handoff.

## Profile Selection

Use the user-provided profile when present. Otherwise:

- use `Prop-Robotics-Neutral@1.0.0`.

Do not infer Isaac, PhysX, Runnable, Package, or candidate profiles from file
extension, converter route, runtime target, validation findings, or available
physics APIs. Use those profiles only when the user or calling workflow
explicitly requests them. Do not select robot or articulated-body profiles by
default; robot SimReady workflows are not supported here yet.

Profiles, versions, requirements, features, and validators come from SimReady
Foundation. Do not invent local profile presets.

For an explicit Joint Agent Gate 3B request, use
`Prop-Robotics-Isaac@1.0.0`. This is a user-requested validation target, not a
claim that the Research Preview output is already simulation-ready.

## Commands

Preflight Foundation dependencies:

```bash
content-workflow-simready-preflight
content-workflow-cli preflight simready-foundation
```

Run staged conformance routing:

```bash
content-workflow-simready-conform-profile asset.usda \
  --output-dir ./simready-conform \
  --profile Prop-Robotics-Neutral \
  --profile-version 1.0.0 \
  --validation-report ./simready-profile.json \
  --report ./simready-conform/simready-conform-profile.json
```

Run formal profile validation:

```bash
content-workflow-simready-validate-profile asset.usda \
  --profile Prop-Robotics-Neutral \
  --profile-version 1.0.0 \
  --report ./simready-profile.json
```

Joint Agent USDZ example:

```bash
content-workflow-simready-validate-profile joint-output.usdz \
  --profile Prop-Robotics-Isaac \
  --profile-version 1.0.0 \
  --report ./joint-validation/gate3b.json
```

Equivalent wrapper shortcuts:

```bash
content-workflow-cli simready conform-profile asset.usda --output-dir ./simready-conform
content-workflow-cli simready validate-profile asset.usda --report ./simready-profile.json
```

When routing `G3A.HYG.001`, pass the trusted pre-hygiene Joint Agent inventory
fingerprint to either conformance command with
`--expected-physics-inventory-sha256 SHA256`. Missing or mismatched proof must
remain blocked.

Pass `--strict` when a failed profile or blocked conformance step should return
a non-zero process status. Without `--strict`, validation/conformance findings
are reported as workflow diagnostics whenever a meaningful USD artifact exists.

## Artifact Contract

Preserve:

- SimReady preflight report;
- conformance report and any staged USD output;
- validation report;
- raw Foundation validation output;
- stdout/stderr logs for subprocess-backed validation;
- latest authored USD path;
- failed requirements, ignored issues, rerun reasons, and repair hints.

## Boundaries

- Treat source assets as immutable unless the user explicitly asks for in-place
  edits.
- Run Foundation validation from the managed SimReady adapter, not by importing
  Foundation internals directly into Workbench.
- Preserve Joint Agent USDZ inputs unchanged. The adapter stages a temporary
  validation target for Foundation and records that staging in the report.
- Use SimReady Foundation as the source of truth for requirements, feature IDs,
  profile versions, validators, and FET conformance policy.
- When conformance requires visual judgement, source data, material identity,
  mass intent, joint semantics, texture edits, or unsafe mutation, report the
  step as blocked and hand off to the matching authoring workflow or the user.

Read the references only when needed:

- `references/foundation-toolchain.md`
- `references/profile-conformance.md`
- `references/profile-validation.md`
