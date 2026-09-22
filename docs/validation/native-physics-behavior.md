# Native Physics evidence in Validation

`physical_behavior` can consume a native `PhysicsBehaviorAssessment` without
converting it into a legacy judge `decision: approve`. The native route is an
explicit, provider-free, read-only consumer; it does not run Physics, render,
call a model, repair files, or weaken a required check. The scaffold template
version is v2 so old prepared/check caches do not silently reuse new semantics.

The initial supported scope is **one asset, one measured rigid body, native
runtime stability and an agent review of recorded OVRTX frames**. It does not
establish arbitrary task success, articulated-joint behavior, source fidelity,
or calibrated actuator/load realism. Those claims require separate measured
checks. A passing synthetic contract fixture is not physical-task evidence.

Prepare a digest closure only after the native Physics run is terminal:

```python
import json
from pathlib import Path
from world_understanding.functions.physics.native_behavior_validation import (
    prepare_native_physics_bundle,
)

run = Path('/absolute/native/physics_run')
bundle = prepare_native_physics_bundle(
    asset=run / 'physics.usda',
    assessment=run / 'physics_behavior_assessment.json',
    validation_evidence=run / 'validation_evidence.json',
    run_dir=run,
)
(run / 'validation_behavior_bundle.json').write_text(json.dumps(bundle, indent=2))
```

Preparation fails for incomplete or unsuccessful native evidence. It freezes
existing bytes and never creates a success/approval record. Retain the bundle
before beginning Validation. Supply it through the existing policy interface:

```json
{
  "behavior_evidence_required": true,
  "physical_behavior_evidence": [{
    "path": "/absolute/native/physics_run/validation_behavior_bundle.json",
    "kind": "simulation_json",
    "role": "native_physics_bundle",
    "required": true
  }]
}
```

A normal agentic `content-workflow-cli validate run` may use this policy with
`--runner codex --model gpt-6-astra --model-reasoning-effort ultra` and explicit
`--required-capability validation.physical_behavior` plus the required geometry/
physics/render capabilities appropriate to the task. The adapter does not select
or waive checks. The existing coordinator still validates the child's plan and
runs exact checks; `look_right` still requires its explicit provider if selected.
A deterministic scaffold invocation is sufficient for a compatibility regression,
but is not a substitute for an end-to-end agentic Validation run.

The consumer requires the exact native schema; pass/fixed visual status with no
unresolved issues; native `sim_ready_status: pass`; no warnings or failures; clean
physics-properties/loadability/no-explosions/visual-review checks; the same asset
path and native asset SHA; matching runtime and render inputs; finite, complete
trajectory coverage; coherent raw solver facts; and the existing native OVRTX
frame/response/dependency attestation checks. The closure binds all consumed
files. Missing, stale, mismatched, unsupported, or incomplete evidence fails the
required check. The native producer remains the authority for its declared
runtime acceptance thresholds; this consumer does not invent new task metrics.
The closure protects retained bytes after preparation, not against a malicious
producer fabricating an entire internally consistent evidence set.

Raw native assessments without a complete bundle are rejected. An unrelated
legacy approval cannot override native rejection, and a native pass cannot hide
an explicitly supplied legacy failure. Existing approved-refine behavior remains
available when no native evidence is supplied. No `approve` field is synthesized.

The implementation reuses read-only producer attestation helpers in
`content_workflow_cli.runner`; that package and `content_agent_workflows` must be
installed. Missing dependencies fail closed. Tests use synthetic files with the
real native verifier and launch no providers, renderer, or simulator:

```sh
python -m pytest -o addopts= tests/test_native_physics_behavior_validation.py tests/test_agentic_validation_scaffold.py -q
```
