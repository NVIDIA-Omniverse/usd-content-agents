# Rerun v2 independent evaluator: cases02–10

Final qualification is complete. `freeze_manifest.json` binds the frozen cases02–10 implementation, all original-source references, public tasks and qualification evidence. Root owns the separate `drawer/` package and global process/model eligibility; this package alone does not authorize scored authors.

No previous author's USD, controller, bindings, traces, or capstone solved drawer
is used as a fixture or author input. `provenance.json` records every retained v1
evaluator and source-reference input. Source IDs, source-to-body role requirements,
five seeds **11,23,47,83,131**, controller gains/effort limits, motion/holding/contact
thresholds, and original task scope remain unchanged except for the explicit
measurement and auxiliary-policy changes below. Every source NPZ was rehashed
against its original frozen inventory. No v1 file is modified.

## Explicit changes

* Cases02–10 compare changed topology using deterministic area samples **and all
  referenced vertices in both directions, plus the separately retained loose-point sets**, recentered in millimetres. The independently
  established meter-coordinate nearest-triangle error is removed. Area tolerance
  remains3%; bounds and surface tolerances remain `max(50µm,0.001×source diagonal)`.
* A fast path proves corresponding triangles stay within the maximum unrounded
  paired-vertex distance. Convexity of barycentric interpolation makes this a
  bound over **every triangle point**, stronger than sampling. It also checks
  every loose point. Sorting only proposes correspondence; it never rounds or
  substitutes a digest for the actual distance proof. Arbitrary retopology falls
  back to the indexed symmetric measurement.
  Candidate triangle surface-landmark queries accelerate it conservatively: a distance to
  any actual triangle below the unchanged tolerance certifies that point; every
  uncertified point still receives the complete threshold-AABB triangle query. Reported
  candidate distances are labelled upper bounds, not exact nearest distances.
* The full3092-part printer float32 source roundtrip initially exposed an original
  loose tessellation point that lies275µm from its own triangle surface. The old
  criterion rejected unchanged source geometry. V2 explicitly preserves the point
  array as well as the surface; added, moved, and deleted loose points are tested.
  Loose-point counts must agree and their point sets are compared symmetrically
  within the same metric tolerance, separately from the triangle surface. Surface
  vertex/face counts may differ under permitted retessellation.
  The first failed qualification is retained beside the corrected run.
* Source measurement writes per-part progress. Complete inspection is bounded at
  1800s; input USD loading has a separate900s deadline; each native trial is bounded at900s (case02 retains300s). A timeout kills only that isolated
  child process group and retains a process receipt/log. Missing observation is
  **INCONCLUSIVE**, never a demonstrated author failure or a pass. The full3092-part retessellation check passed1266.87s under the actual4CPU/32GiB evaluation budget, with USD build/read48.02s and fresh read38.07s.
* The evaluator's gravity witness moves from `(1000,1000,1000)` to `(1,1,1)`m and
  has no collision API. This avoids the confirmed distant-origin/high-iteration
  measurement interaction without touching source bodies or accepting collisions
  against a test fixture. Witness failure is evaluator insufficiency; wrong
  authored Earth gravity remains a concrete failure. Failed physics predicates from a seed with a failed witness remain raw observations but are classified INCONCLUSIVE; independent static/source/schema failures remain concrete. The engine paired-load response requires both witnesses. Fresh native calibration passes default and128-iteration companion scenes, with a zero-gravity negative.
* Case06 explicitly permits at most eight declared `passive_constraint` auxiliary
  moving bodies. Each must have at least two distinct joint neighbors, no incident
  controlled joint, finite positive observed dynamics and colliders, and no required
  source role. Original parts still map exactly once. This replaces a hidden
  universal source-ownership gate; it does not permit auxiliary actuators.
* General traces now label pre-step pose/joint/control time separately from
  post-step velocity/contact time, and retain actual native initial poses. The
  measured trajectories, controller sampling, and thresholds are unchanged.
* The controller is clamped after seeded0.98–1.02 jitter, preventing the prior2%
  excess above its published actuator cap. Gains, jitter distribution, targets,
  and separately declared external test loads remain unchanged; this is not a
  retuning of the prior joint5 holding failure.
* Required named body roles are unambiguous. Existing per-case structural and
  source-role checks remain in their copied snapshots; no universal completeness
  claim is inferred from a schema or fixture.

## Scope of the nine tasks

| Case | Required physical observation | Deliberate limit |
|---|---|---|
|02 conveyor|Bounded rail torque transfers free carrier and payload through the declared path fraction with measured contacts.|No externally prescribed carrier/payload motion.|
|03 hinge|Door opens≥60°, closes≤5°, stays within actual joint limits.|No refrigeration function.|
|04 gripper|Source-linked jaws close, make bilateral contact, retain the independent block under load, and release.|Toy contact calibration does not substitute for source linkage/role checks.|
|05 vise|Slider travels, holds the target under bounded load, returns, and exercises its limit.|No generic workpiece clamp claim.|
|06 engine|Crank completes≥2π; piston stroke≥2mm; beam swing≥0.05rad; closure/repeat and paired shaft-load response.|Ideal joints and explicitly passive auxiliary constraints; no steam thermodynamics.|
|07 robot arm|All six joints move to targets, hold under bounded torque, return, and maintain end-effector/chain consistency.|The previous joint5 holding failure is not waived or retuned.|
|08 excavator|Boom/stick/bucket source roles articulate through three links under load and maintain closure.|Conversion-relative reference; no digging or native-CAD completeness certification.|
|09 printer|X/Y/Z travel/hold/return; original bed carries a freely contacting payload with≤3mm relative drift and≥90% post-settle contact.|No extrusion or print accuracy claim.|
|10 hand|All14 rotary joints across five source-grounded digit branches flex, hold, and reopen; every distal part moves.|69 saved source shells, not five rigid finger compounds; no tendon accuracy or dexterous grasp claim.|

Exact numeric contracts are in `cases/<case>/cases.json` (or case02 `contract.json`),
case04/09 companion contracts, and proposed `tasks/`. Final task files bind their v2 snapshot hashes and completed qualification receipt.
Public inventories are addressed as `/input/<case>_source_inventory.json`; the orchestrator mounts only the current case. Missing/empty/malformed author artifacts or invalid bindings schemas are concrete rejection. Missing/corrupt evaluator references, loader deadlines and native infrastructure errors are inconclusive.

## Local controls and remote qualification

The isolated local `.venv` is a test dependency environment, not a distributable
runtime. The measurement suite tests analytic small triangles at several world
origins, known offsets, changed topology, surface certificates, thin missing
geometry, area duplication, loose-point preservation/negatives, auxiliary-policy
refusals, nonfinite/index errors, and bounded process/inspection timeouts.

```sh
python -m pytest tests -q
python qualification/source_roundtrip.py --case 09_printer --output /new/reference-check
```

Source roundtrip performance is not full USD stage/runtime performance. The unchanged original
STEP need not be retessellated for scoring: complete independently frozen NPZ
references are included. Original03 reference regeneration still needs the
documented exact-hash alignment of Trimesh's random split-primitive names; source
IDs and expected hashes are not regenerated or rewritten by this package.

Native qualification entrypoint, **only on a root-authorized isolated Linux slot**:

```sh
"$STRUCT_PY" qualification/native_qualify.py --case 03_hinge --variant positive \
  --output /new/synthetic03 --run-solver --solver-python "$SOLVER_PY" --seed 11
```

`STRUCT_PY` and `SOLVER_PY` name the independently qualified executables.
Omitting `--run-solver` prepares and inspects synthetic USD only; it explicitly
reports native qualification pending. It never imports `ovphysx` into the USD
parser process. Each actual native invocation uses a new child process and retains
all inputs, logs, timing, result and observed checks.

Before freezing, require fresh positive native controls for all five seeds in each
case, meaningful blocked-motion/contact negatives, the structural negatives, torque
and impulse/dt calibration, and the near-origin witness at default and128 position
iterations. Case06 additionally needs its paired unloaded response. Case09 needs a
full source-derived stage roundtrip timing and independent synthetic XYZ/payload
native cooking/runtime timing on the actual isolated worker. Cases04/09 require retained independent payload placement
proof; cases08/10 require role/branch negatives. `qualification_plan.json` enumerates
the gate. Remote job concurrency and author/evaluator isolation are root's broker
responsibility; an evaluator cannot certify global compliance from its own process.

Accepted means every predeclared case check and all five fresh trials passed.
Concrete failure and unavailable measurement remain separate. A false-pass claim
also requires an explicit author acceptance claim and a concrete independent
failure. Neither synthetic positives nor successful file hashing certify a submitted
mechanism. No frozen v1 outcome is rescored or overwritten by this v2 package.

## Qualification boundaries

The production gripper torque path is also tested on a separate analytical two-slider closed linkage. The first toy motor inertia (1e-6kgm²) produced controller oscillation and failed; those bytes remain. The corrected fixture uses a200g cylinder of32mm radius and4mm height with analytic inertia. It changes no controller or task threshold, and qualifies that operating regime rather than every possible authored mass/inertia. No original CAD geometry is used in this toy linkage.

Printer source preservation is timed both as a complete3092-part changed-triangulation comparison and as a fresh-process original-source USD readback. Native XYZ/payload fixtures separately qualify the physical observer. These controls do not bound arbitrary submitted collider decompositions; actual trials still have the predeclared900s deadline and infrastructure/resource exhaustion stays inconclusive.

The original2CPU/8GiB preparation check completed3092-mesh USD build/read in47.77s and a fresh read in36.46s, but the full changed-triangulation comparison timed out at900s after1448 passing parts. It is retained as INCONCLUSIVE. Before scored authoring, the complete structural/source deadline is set to1800s and qualified under the actual4CPU/32GiB evaluation lane. Input USD loading remains900s and native-trial caps/geometry tolerances are unchanged.

Final controls:182 local tests;45 positive native trials across all nine cases and five seeds;13 expected native negatives;five additional production-cam positives and a broken-linkage negative;37 structural controls;36 branch checks;12 source-role/placement controls; fresh force, torque, impulse/dt and gravity calibration. `qualification/final_receipt.json` and `qualification_manifest.json` bind all evidence, including retained unsuccessful preparation attempts.
