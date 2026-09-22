# Drawer evaluator implementation

`DRAWER_CONTRACT.md` is the arm submission contract. `drawer_acceptance.json` is the machine-readable acceptance specification. Both arms must use the same frozen files and evaluator hashes recorded in `drawer_freeze.json`.

The evaluator has two processes. `drawer_evaluate.py` runs in the repository's USD environment to inspect the actual composed submission, compare all five source components against the publisher's untouched glTF triangles, and create per-seed test copies. `drawer_solver.py` runs in the isolated ovphysx environment and loads each USD copy directly. It applies a bounded world-frame force and reads native body poses, velocities and contacts. It never imports submitted Python, reads an arm's success JSON, plays a submitted animation, or writes body poses/velocities while stepping.

The force controller is an explicitly modeled handle-force abstraction. It applies force at the drawer body, without a robot hand or grasp controller. Mass, inertia, material and collision choices remain authored parameters subject to validity checks; they are not measurements of physical furniture. The evaluator initializes the drawer at rest, fixes gravity, enables contact reports and consistent solver iteration counts, removes drives, and independently adds its free 0.5kg payload. These interventions are identical for both arms.

`PASS` means source/structural checks and every one of five seeds passed. `FAIL` means a concrete structural or behavioral criterion failed. `INCONCLUSIVE` means the native backend, binding, import, or evaluator failed to produce trustworthy observations; this is never counted as evidence of an asset false pass. Aggregate output retains these statuses and errors separately.

## Evidence and tests

- `source_direction_evidence.json`: nine ray tests on the original cabinet establish positive Z as the open face and negative Z as the rear wall.
- `force_units_evidence.json`: three small synthetic cubes verify freefall and 1N force scaling with 1kg versus 2kg mass. Integration error is bounded against analytic motion.
- `selftest/test_report.json`: expected-negative kinematic, missing-collider and wrong-joint inputs; unchanged and deformed source surface comparisons; independent native positive tray, blocked/no-motion tray, and filled-collider tray runs.
- `mesh_collision_evidence.json`: a synthetic convex-mesh floor additionally exercises USD mesh collision cooking and native payload contact, beyond primitive-collider tests.
- `selftest_initial/` preserves the first successful fixture run before provenance/initial-rest hardening. `selftest/` records the final rerun and independent traces.

The synthetic tray uses a handful of cuboids and one prismatic joint. It is deliberately not the Poly Haven cabinet, fails the actual source-fidelity requirement, and is never a scored arm result. No benchmark arm output or source-cabinet solution was used to tune the evaluator.

## Output and limits

Each real invocation creates a fresh output directory containing `structural_report.json` and aggregate `report.json`; each seed has the actual imported `scene.usda`, a scene-hashed request, raw solver log, one-row-per-step `trace.jsonl`, native `trial_report.json`, and `replay.usda`. The replay is written after simulation from observed native poses and serves only as evidence playback. Geometry bounds, triangle counts, area, vertex/centroid distances and welded boundary/nonmanifold-edge counts are reported per component. Input USD, bindings, composed stage, specification and implementation digests identify the evaluated version.

Interior collision probes cover six predefined open-space points, and the free payload exercises the floor and physical transport. These tests do not prove collision accuracy at every point of the cabinet or every possible manipulation. The five deterministic perturbations test bounded payload placements and controller-force jitter; they are not a statistical estimate of general reliability or a real-world validation.

No extra packages were installed for this evaluator. USD inspection uses the existing repository environment; native simulation uses ovphysx0.4.13 in its separate environment because mixing its bundled USD25.11 with repository USD25.5 in one process is unsupported. All simulation and reference-geometry calculations ran on the assigned Horde node.
