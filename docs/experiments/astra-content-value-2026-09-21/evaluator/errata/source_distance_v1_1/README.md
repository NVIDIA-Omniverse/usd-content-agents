# Source-distance numerical erratum, v1.1

The frozen source-surface routine calls `trimesh.proximity.closest_point` in meter coordinates. With the installed trimesh 4.12.2, an analytically on-surface point on a 0.5 mm triangle incorrectly measured 0.125 mm from that triangle. Uniformly evaluating the same geometry in millimeters returned zero. This can falsely reject retained small CAD features. Frozen v1 files and receipts remain unchanged.

The correction multiplies both meshes and the existing length tolerance by 1000 before calling the **unchanged frozen sampling/distance function**, then divides measured lengths by 1000. Its source-function SHA256 is `e402e7a4511ead7db0ef45f28c7de7ac8c58a1ec59279ddbce7abd827e2c99dd` in all nine affected snapshots: conveyor, general03–07/09, and08/10. No source, assembly transform, mesh, sampling count, seed, tolerance, contact predicate, controller, or solver is changed.

The worker independently recalculates bounds and area in meters, requires agreement with both the frozen v1 evidence and corrected computation, and carries the original v1 bounds, area ratio, and tolerance values into the replacement checks exactly. The original tolerance remains `max(0.00005 m, source diagonal * 0.001)` with area ratio deviation at most0.03.

`qualification_r2.json` passes50 numerical regressions: five triangle scales, three origins, exact-on-surface and known25/75 µm offsets, unchanged50 µm acceptance boundaries, area/bounds checks, and reproduction of the frozen identity failure. `non_submission_qualification.json` verifies that both synthetic null-submission arms preserve their original contractual rejection and allow the batch to continue. These are unscored fixtures.

Revision2 adds null-submission handling and discovers conveyor NPZ files beside `evaluator/conveyor/reference/source_inventory.json`. Numerical correction is identical to the first qualified worker, which is retained as `worker_initial.py` with `qualification_initial.json`; that initial worker produced the completed04 adjudications.

Run a single completed arm on a Linux machine with the complete retained submission, `output_manifest.json`, regenerated or retained reference NPZ files, and full original native evaluation directory. The public subset alone omits those non-drawer artifacts and is insufficient for this command. Choose a fresh output directory and explicitly bind revision2 qualification:

```sh
export EXP_ROOT=/absolute/path/to/complete-staged-experiment
export STRUCT_PY=/absolute/path/to/structural-env/bin/python
env TMPDIR=/absolute/path/to/writable-scratch UV_NO_CACHE=1 \
  nice -n 15 ionice -c 3 \
  "$STRUCT_PY" \
  "$EXP_ROOT/evaluator/errata/source_distance_v1_1/worker.py" \
  --root "$EXP_ROOT" --case 03_hinge --arm plain_astra \
  --qualification "$EXP_ROOT/evaluator/errata/source_distance_v1_1/qualification_r2.json" \
  --output /absolute/path/to/new-03-plain-adjudication
```

Use `--snapshot`, `--inventory`, `--source-root`, or `--v1-output` for an explicitly staged runtime. Defaults cover the standard evaluator directories and the04 staged runtime. `dispatcher.py --cases ... --label ...` schedules one worker at a time on a host, waits for completed author/v1 records, and records every job. It supports the authorized fresh conveyor run via its worker flag. Do not launch a second owner for the same case/arm output directory.

Receipts are separate from frozen v1:

- `evaluation_adjudications/pilot-v1/{case}/{arm}/source_distance_v1_1/acceptance.json`: `accepted`, `status`, `concrete_failures`, `inconclusive_checks`, `false_pass`, input hashes, original v1 receipt hash, geometry differences, and native-evidence hashes.
- `conservative_v1.json`: known-invalid v1 distance measurements treated as inconclusive; independently failed bounds/area and other concrete failures remain failures.
- `provenance.json`: exact report/artifact hashes and original native evidence hashes.
- Null submissions receive `adjudication_action="skipped_non_submission"`; their original contractual failure is preserved without inventing geometry or physics evidence.

General-case acceptance requires all five complete v1 native seed results, unchanged author scene/bindings, and unchanged runtime configuration/scenario/results throughout adjudication. The worker does not rerun general physics. Conveyor v1 gated physics on all structural checks; when corrected geometry is its only blocker, the explicitly authorized option `--fresh-conveyor-if-needed` runs the unchanged frozen conveyor evaluator/controller/solver in `fresh_native/`, substituting only the qualified distance function in memory. No acceptance is promoted without five complete native trials.

Raw v1 `false_pass` flags caused solely by the invalid distance check must not be counted. Publish frozen, conservative, and corrected outcomes distinctly. A post-freeze adjudication is not a preregistered result or a new author repair.
