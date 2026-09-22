# Supplementary static delivery audit

This independent, read-only audit checks necessary physical-delivery conditions
for the Content Agents submissions in cases07–10. It does not replace or modify
the frozen evaluations, and does not complete source-surface comparison or native
physics tests. The author acceptance claim is not used as observation evidence.

The auditor hashes the output manifest, submission pointer, saved scene,
bindings, launch receipt and frozen public task. The public task hash must match
the launched task. Every composed USD layer must be inside the submitted run
and match its output-manifest hash. Missing dependencies stop the audit instead
of being mistaken for absent delivered physics. Input and used-layer hashes are
checked again afterward.

The exact submitted USD is opened with `LoadAll`. Actual `UsdPhysics` APIs and
joint schema types are enumerated, including instance proxies. The receipt
lists every rigid body, joint and collider, including enabled/kinematic state.
Required roles come from the frozen public task's `public_acceptance` object.
Each required role must map uniquely to an actual prim. In these four scoped
tasks, each non-base body is explicitly required to move, so it must have an
enabled, nonkinematic rigid-body API and an enabled collider. Each required
joint must have the specified actual joint type and be enabled. Declared role
names or binding `moving` flags cannot substitute for those APIs.

`DECISIVE_NON_DELIVERY` means one or more necessary static requirements is
contradicted by the submitted artifact. This can establish task non-delivery
even while source-fidelity evaluation remains unfinished. It is not a measured
native solver failure. `NO_DECISIVE_NON_DELIVERY_FROM_THESE_CHECKS` means only
that this small static subset did not establish rejection; it is never a
physical acceptance verdict. Mass, inertia, joint endpoints, source fidelity,
loads, limits, contacts and successful motion still require their independent
checks.

The positive/negative synthetic qualification is run independently on both
recorded structural runtimes. It tests geometry-only input, disabled and
kinematic bodies, absent/disabled colliders, wrong or disabled joints, absent
or fake bindings, duplicate roles, and unresolved dependencies. The positive
fixture intentionally demonstrates schema presence only, not a valid physics
task. Qualification receipts bind the exact auditor bytes and USD version.

All commands use CPU only, `nice15`, idle I/O and a single OpenMP/BLAS thread.
They start no renderer, solver, model worker or proximity query. Choose fresh
output paths; evidence is never overwritten.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n15 ionice -c3 \
  /path/to/structural-env/bin/python qualify.py --output-dir qualification_new

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n15 ionice -c3 \
  /path/to/structural-env/bin/python audit.py \
  --root /path/to/retained-experiment --case 07_robot_arm \
  --qualification qualification_new/qualification.json \
  --output 07_robot_arm_content_agents_new.json
```

The findings remain separate supplementary evidence. No source evaluation is
claimed complete, no existing job is stopped, and no scored receipt, frozen
implementation or aggregate result is changed by this audit.

An independent Astra Ultra reviewer inspected the auditor and found no blocking
issue for this necessary-condition-only scope. The descendant-collider check
could miss an ownership error when a nested separate rigid body owns the
collider; therefore positive presence still cannot establish acceptance.
Malformed bindings, parser failures or hash/dependency assertions are audit
errors with no decisive verdict, not evidence of physical rejection. The four
observed zero-schema submissions do not depend on those subtleties. The review
is retained in `independent_review.json`.
