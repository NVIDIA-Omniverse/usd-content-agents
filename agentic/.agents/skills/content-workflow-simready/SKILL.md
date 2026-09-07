---
name: content-workflow-simready
description: Use after USD conversion, material and physics authoring, or Joint Agent publication to preflight SimReady Foundation, formally validate the current staged USD or USDZ, conditionally conform it from the failed validation report, revalidate the conformed output, run Gate 3B validation, and preserve durable non-blocking remediation evidence.
metadata:
  author: NVIDIA Omniverse
---

# content-workflow-simready

Use this skill after an asset has a meaningful USD-family output and the user
wants SimReady profile conformance or formal SimReady Foundation validation.
Run material and physics workflows first unless the user explicitly asks for a
validation-only inspection of the current USD.

This skill owns SimReady profile selection, Foundation toolchain preflight,
staged conformance routing, formal static profile validation, optional external
SimReady Benchmark execution, and remediation handoff. It does not implement a
simulator or runtime-test framework. It delegates requested runtime tests to the
pinned `simready-benchmark` package and preserves that package's native results.
It also does not own CAD conversion, material prediction, physics prediction,
texture generation, joint inference, usd-cli or Workbench scene mechanics, or
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
5. Run the existing SimReady Foundation dependency preflight automatically.
6. Run formal profile validation against the current staged USD through
   `content-workflow-cli simready validate-profile`.
7. If validation reports failed requirements, run staged conformance through
   `content-workflow-cli simready conform-profile`, passing that failed
   validation report.
8. Run formal profile validation again against the conformed USD.
9. When the user requests runtime validation, run the external SimReady
   Benchmark adapter after static validation has identified the intended
   profile and features.
10. If validation or revalidation fails after a meaningful USD exists, record
    conditional status, rerun reasons, repair hints, and the next conformance
    handoff.

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

SimReady is a workflow capability implemented by
`content_agent_workflows.simready`; it does not depend on a native usd-cli
command or on usd-cli. Use the existing wrapper path automatically:

```bash
content-workflow-cli preflight simready-foundation \
  --report RUN/simready_preflight.json

content-workflow-cli simready validate-profile STAGED.usd \
  --profile Prop-Robotics-Neutral \
  --profile-version 1.0.0 \
  --report RUN/prior_validation.json \
  --stdout-log RUN/prior_validation_stdout.log \
  --stderr-log RUN/prior_validation_stderr.log

# Run only when prior_validation.json reports failed requirements.
content-workflow-cli simready conform-profile STAGED.usd \
  --output-dir RUN/conformed \
  --profile Prop-Robotics-Neutral \
  --profile-version 1.0.0 \
  --validation-report RUN/prior_validation.json \
  --report RUN/simready_conformance.json

# Read output_usd_path from simready_conformance.json before revalidation.
content-workflow-cli simready validate-profile CONFORMED_OUTPUT.usd \
  --profile Prop-Robotics-Neutral \
  --profile-version 1.0.0 \
  --report RUN/simready_profile.json \
  --stdout-log RUN/simready_stdout.log \
  --stderr-log RUN/simready_stderr.log
```

Consult each command's `--help` for the exact output path and optional
Foundation checkout/venv arguments. The wrapper performs dependency preflight,
routes conformance and validation through the external SimReady Foundation
toolchain, and preserves normalized reports. Do not prompt merely because
`usd-cli simready` is absent.

Run runtime validation through the external benchmark environment:

```bash
content-workflow-cli simready validate-runtime asset.usda \
  --output-dir ./simready-runtime \
  --sr-specs /path/to/simready-foundation/nv_core/sr_specs \
  --engines-toml /path/to/engines.toml \
  --tests-path /path/to/foundation-test-pack \
  --feature FET003 \
  --runtime isaac_sim \
  --benchmark-executable /path/to/benchmark-venv/bin/simready-benchmark
```

`content-workflow-simready-runtime-validate` remains the standalone compatibility
entry point for the same adapter.

Joint Agent USDZ example:

```bash
content-workflow-simready-validate-profile joint-output.usdz \
  --profile Prop-Robotics-Isaac \
  --profile-version 1.0.0 \
  --report ./joint-validation/gate3b.json
```

Equivalent installed console commands:

```bash
content-workflow-simready-preflight
content-workflow-simready-conform-profile asset.usda --output-dir ./simready-conform
content-workflow-simready-validate-profile asset.usda --report ./simready-profile.json
content-workflow-simready-runtime-validate asset.usda \
  --output-dir ./simready-runtime \
  --sr-specs /path/to/sr_specs \
  --engines-toml /path/to/engines.toml
```

`usd-cli validate` and `usd-cli physics validate` are generic USD/physics
checks, not SimReady profile conformance, and must not be reported as SimReady
status.

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
- the validation report's canonical `asset_dependency_manifest`, which binds
  the root layer and every resolved local USD composition or asset dependency
  by path, size, and SHA-256 (or binds the complete archive for USDZ);
- raw Foundation validation output;
- stdout/stderr logs for subprocess-backed validation;
- the durable `workflow_run_manifest.json` record reported by each result's
  `workflow_run_manifest_path`, including its sealed checkpoint history;
- the native SimReady Benchmark report, run summary, event stream, and logs;
- the normalized runtime report and `physical_behavior` template projection,
  retaining every native per-test disposition;
- one independently ingestible verified-operation v1 envelope per native runtime
  test, with the exact native report, per-test payload, source/dependencies,
  metrics, media, logs, engine identity, and adapter/tool/projector identity;
- latest authored USD path;
- failed requirements, ignored issues, rerun reasons, and repair hints.

## Boundaries

- Treat source assets as immutable unless the user explicitly asks for in-place
  edits.
- Run Foundation validation through the managed SimReady adapter in
  `content_agent_workflows.simready`, which invokes the external Foundation
  validator as a tool. Do not import Foundation internals into usd-cli or
  usd-cli.
- Run runtime tests through the public `simready-benchmark` CLI in its separate
  Python 3.12 environment. Do not reproduce its planner, engine adapters,
  execution lifecycle, batch/resume behavior, reporter, or stamping logic.
- Treat benchmark exit code `4`, engine readiness failures, zero-test plans,
  engine crashes, malformed reports, and changed source dependencies as
  non-passing results. Do not reinterpret them as asset passes.
- Keep static Foundation/Isaac conformance, dynamic Benchmark execution, and
  shared final assessment separate. The SimReady projector consumes retained
  files only; it must never rerun Benchmark or a simulator. Preserve
  `skipped` as `not_evaluated` and `incomplete` as `error` through ingress.
- Reject unresolved, remote, or changing USD dependencies. A root-layer hash
  alone is not sufficient validation identity for a composed stage.
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
- `references/runtime-validation.md`
