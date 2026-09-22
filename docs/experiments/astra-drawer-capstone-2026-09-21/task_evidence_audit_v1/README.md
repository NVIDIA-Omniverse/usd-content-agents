# Independent task evidence audit

This create-only auditor does not run a solver, renderer or model, and never imports
the frozen evaluator/solver. It binds their exact SHA256 values plus the source-clear
spec, submitted asset/bindings, structural/aggregate reports and every seed's
request, scene, report, full trace, replay and native log. Native USD dependency
closure is resolved and hashed; repeated/before/after reads must retain the same
bytes. The submitted composed source and every authored source property are checked
against the task scenes, allowing only the evaluator's declared interventions.
Payload mass, size, inertia, material, independent-body status, seeded placement,
gravity, body iterations, disabled drives, controller and all 2460 finite contiguous
post-step samples per seed are checked.

The metric formulas are separate code: prescribed half-cosine force PD and seeded
noise, opening and closing windows, final settling, payload center retention,
raw pair-contact impulses, global contact penetration, speed, rotation and off-axis
motion. Acceptance thresholds are unchanged. Small documented tolerances reconcile
float32 USD/trace storage with float64 metrics; they never relax task gates.

A consistent audit is **not** task PASS or native Validation acceptance. Its receipt
retains the reported task result and lists directly corroborated failures. Exact
native-relative off-axis/orientation cannot be independently reconstructed because
the original trace omits initial native XY/quaternion. The redundant Z-origin is
checked against the retained import-error bound; authored-frame XY is bounded by
that error, and authored-frame normalized and legacy quaternion rotations are
reported explicitly. The original legacy quaternion check did not normalize.

`payload_drawer_contact_force_n` is actually impulse N*s in this runtime. The frozen
contact predicate remains norm >0.01Ns (equivalent2.4N at240Hz). Global contact rows
lack actor IDs; penetration therefore cannot be independently attributed to a body
pair. Interior query rows preceded the first native pose read, so their saved zero
hits are only cross-checked, not requalified as initialized queries. Separate
positive-controlled cooking preflight is required. Prior original-source geometry
checks are bound, not re-run. Hashes and numerical agreement cannot prove a hostile
producer truly executed a solver.

Run with an expected final asset hash independently obtained beforehand:

```sh
TASK_ROOT=/opt/astra-content-value-20260921
nice -n 15 ionice -c3 "$TASK_ROOT/repo/.venv/bin/python" \
  "$TASK_ROOT/capstone/task_evidence_audit_v1/audit.py" \
  --experiment-root "$TASK_ROOT" \
  --evaluator "$TASK_ROOT/capstone/evaluator/source_clear_v1" \
  --evaluation "$TASK_ROOT/capstone/evaluations/native10_source_clear_v1" \
  --bindings "$TASK_ROOT/capstone/evidence/native10_task_bindings.json" \
  --asset "$TASK_ROOT/capstone/runs/drawer_physics_10/physics.usda" \
  --expected-asset-sha256 "$REVIEWED_ASSET_SHA256" \
  --output "$TASK_ROOT/capstone/task_evidence_audit_v1/native10"
```

Exit0 means consistent within the stated evidence limits. Exit2 means inconsistent
or incomplete evidence, not an automatic physical task failure. An existing output
directory is always refused. This tool only assesses completed five-seed reports.
No prior task, evaluator, source or native output is modified.

Qualification uses analytical stationary traces with known failure predicates,
finite/count/time/force/phase/contact/penetration/retention/rotation mutations,
actual retained scene readback and separate copied-USD mass mutations. Synthetic
traces are unit fixtures, never claimed solver runs. Retained native09 is the
full-path negative physical example: all five original task FAILs must remain FAIL.
