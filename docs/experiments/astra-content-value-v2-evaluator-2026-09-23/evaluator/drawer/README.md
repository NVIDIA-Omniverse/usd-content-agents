# Drawer evaluator v2 (not frozen)

This package preserves the original five source meshes and the bounded0.5kg drawer task. It carries the source-derived, collision-free payload start from the separate capstone and corrects query initialization, contact units, recorded initial quaternion and rotation-aware payload coordinates. Opening, closure, force, mass, geometry-fidelity and physical tolerances remain unchanged.

Before any scored author, qualify the unchanged-source checks, mathematical metrics, synthetic native good/blocked/filled tasks and initialized cooked-collision queries. A static cube in a separate query-only derivative provides a known positive witness. The proposed payload is absent from that query scene, so its own collider cannot contaminate the envelope test. No time step is used in the query preflight.

A missing positive witness or solver evidence is INCONCLUSIVE. An observed blocked envelope, missing required physical structure or completed physical test failure is FAIL. A passing task requires all checks and all five seeds. Native query/cooking logs remain evidence even on failure.

Run the pure numeric tests with `python -m unittest discover -s . -p test_drawer_metrics.py`. Set `OVPHYSX_TEST_PYTHON` to the pinned native runtime Python and `DRAWER_QUALIFICATION_DIR` to a new directory before running `drawer_test_fixtures.py` with the OpenUSD environment. Run an authored submission with `drawer_evaluate.py --bindings ... --output NEW_DIRECTORY --solver NATIVE_PYTHON`.

This is development-set simulated task acceptance. Estimated mass/inertia/friction and ideal rail collision exclusions do not establish hardware fidelity. Contact-point penetration is a global scene measure; the payload/drawer contact impulse uses the explicit pair binding. Retention uses payload center, not full-volume containment.
