# Native Validation policy and separate physical task gate

Use only after the native Physics run is terminal and successful. This reads the
actual typed output; it never substitutes an approval or clears a warning.

```python
import json
from pathlib import Path
from world_understanding.functions.physics.native_behavior_validation import prepare_native_physics_bundle

run = Path('/absolute/capstone/runs/drawer_physics_08')
asset = run / 'physics.usda'
assessment_path = run / 'physics_behavior_assessment.json'
review = json.loads(assessment_path.read_text())
bundle = prepare_native_physics_bundle(
    asset=asset, assessment=assessment_path,
    validation_evidence=run / 'validation_evidence.json', run_dir=run,
)
bundle_path = run / 'validation_behavior_bundle.json'
bundle_path.write_text(json.dumps(bundle, indent=2))
policy = {
    'behavior_evidence_required': True,
    'physical_behavior_evidence': [{
        'path': str(bundle_path), 'kind': 'simulation_json',
        'role': 'native_physics_bundle', 'required': True,
    }],
    'render_image_paths': review['rendered_frames'],
}
(run / 'native_validation_policy.json').write_text(json.dumps(policy, indent=2))
```

Example supported agentic CLI shape (replace paths; do not run while Physics is
active). The three capabilities are mandatory. A provider-free rules/adapter
execution is selected by the agentic planning child; no external VLM is assumed:

```sh
content-workflow-cli validate run \
  --usd /absolute/capstone/runs/drawer_physics_08/physics.usda \
  --task 'Validate the authored drawer USD using the native measured single-body runtime stability and recorded OVRTX visual review. Require render_valid, physics_sane, and physical_behavior, with explicit dependencies. This check does not establish the separately evaluated opening, load retention, or five-seed task metrics. Preserve Geometry conditional warnings and all earlier physical failures in the outcome description; do not waive required checks or claim universal SimReady readiness.' \
  --output-dir /absolute/capstone/runs/drawer_validation_01 \
  --base-dir /absolute/capstone/runs/drawer_physics_08 \
  --policy-file /absolute/capstone/runs/drawer_physics_08/native_validation_policy.json \
  --runner codex --model gpt-6-astra --model-reasoning-effort ultra \
  --required-capability validation.render_valid \
  --required-capability validation.physics_sane \
  --required-capability validation.physical_behavior \
  --fail-on-warn
```

The final physical-task claim needs a separate, explicit conjunction:

1. Native Validation succeeds without waiving required warnings/failures.
2. The independent, frozen source-clear fixture's five-seed task acceptance
   succeeds, and its submitted asset SHA matches the native Validation asset SHA.
3. The source-clear fixture freeze and qualification digests are the predeclared
   ones, with original-source versus collision-proxy floor initialization and
   original-fixture failure reported separately.

Do not insert task JSON/trajectories into `physical_behavior_evidence` and infer
that their metrics have been checked. The native adapter consumes native smoke
and visual evidence only. Retain the task reports, trajectories, replay renders,
fixture hashes, and same-asset identity in a separate capstone outcome receipt.
A failed or unavailable native run cannot create the bundle above; preserve the
failure instead of writing a synthetic success record.
