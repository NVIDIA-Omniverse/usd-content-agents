# Evidence notes: scope, history, privacy and reproduction

This bundle is a dated subset of a separate follow-up. Frozen pilot assets,
evaluators and scores are unchanged. Geometry04 preserves all five original
CC0 source meshes and has a conditional handoff; Joint02 has an accepted native
articulation receipt. Native Physics03,06 and07 fail. Physics08 also completed
with workflow status `fail`, ValidationEvidence `conditional`, and the typed
visual assessment `unresolved_issues`; its measured native smoke gates passed.
The independent five-seed source-clear task also failed all five seeds. Physics09 also ended unresolved and failed all five independent task trials.
Physics10 passed its native mounted-rest workflow and all five independent source-clear tasks on the same asset. The later actual native Validation terminal passed all four required gates and accepted the independent review. This completes the bounded conjunction; original Geometry conditions and optional unsupported cross-stage integrity remain unchanged.

The [complete task report](evaluations/native08_source_clear_v1/report.json) binds
the unchanged08 asset, bindings, specification and evaluator. The [full-range
plot](task_plots/native08_source_clear_v1/drawer_task.png), five CSVs and five
losslessly compressed full traces show failed opening, penetration and closed
hold gates. The [export receipt](task_plots/native08_source_clear_v1/export_verification.json)
and manifest bind both compressed and original uncompressed trace hashes.
The [source contact diagnostic](evidence/source_contact_diagnostic_08.json) measures
distance to original source surfaces at a recorded pose; proximity does not identify
a contact actor, prove solid overlap or establish the failure cause.

The [09 task report](evaluations/native09_source_clear_v1/report.json),
[09 full-range plot](task_plots/native09_source_clear_v1/drawer_task.png),
[09 exact trace export verification](task_plots/native09_source_clear_v1/export_verification.json),
and [independent09 startup review](evidence/native09_startup_review/review.json)
preserve the separate failed outcome and its unchanged asset. The [brief installed
runtime documentation excerpt](evidence/ovphysx_0_4_13_registry_excerpt.json)
binds package version/document SHA and exact lines137–138 supporting one narrow
nonfatal initialization-message exception. The full SDK document is not distributed.
The [initial strict audit](evidence/native09_critical_cooking_log_audit.json) remains
false and the [actual cooking log](evidence/native09_cooked_clearance_v1.log)
preserves every warning; this exception grants no workflow/task acceptance. A later
[static endpoint repair protocol](evidence/static_endpoint_repair_protocol_v1.json)
and [synthetic qualification](evidence/static_joint_endpoint_v1/review_receipt.json)
support the bounded static Mesh endpoint hypothesis. The separately completed
[Joint03 native terminal receipt](runs/drawer_joint_03/standalone_articulation_terminal_receipt.json)
is accepted; this establishes neither Physics10 nor physical-task success. The [native Validation launcher](validation_launch/README.md)
and [qualification](validation_launch/qualification.json) qualified the strict
consumer and staged native closure before execution. That qualification itself
launched no native workflow. Its task precheck
binds reports, not a new independent verification of solver traces. The launcher
later rejected actual Geometry04 because it assumed the wrong handoff field. The
[real refusal](validation_launch/history/geometry_handoff_assumption/original_assumption_refusal.json)
and old source/review remain exact in history. The corrected launcher consumes
Geometry's actual typed schema and conditional handoff, qualified by22 tests;
no Geometry receipt or policy was changed. The earlier independent review applies
only to its explicitly preserved old code, not this correction. Historical path
redirects in the publication manifest point to exact old code bytes. Any superseded
historical documentation that is not provided is labeled digest-attested only.

The [independent08 preflight review](evidence/native08_independent_preflight_review.json)
retains an actual plugin warning: requested hullVertexLimit128 lies outside the
observed supported8..64 range. Authored USD readback does not prove that128 took
effect. Clear initial cooked space and measured smoke stability do not establish
loaded motion or payload retention. Previous code qualification is preserved as
a historical result and does not qualify every cooking option at runtime. A
[later range correction](evidence/convex_options_bounds_v2/qualification.json)
restricts future authoring to8..64. The code patch includes that later correction;
it was not applied to, or used to reinterpret, the unchanged08 asset or trials.

The [outcome](outcome.json) and [publication manifest](publication_manifest.json)
separate exact bytes from explicitly selected/projected reports. Selected Joint03
terminal, identity, readback and preparation receipts remain exact where their
reviewed fields contain no private data. Mechanical endpoint fields are retained;
service/session metadata and decision reasoning are projected out when present. Embedded native
receipt hashes identify original retained files; consult the manifest to see
whether the target is published exactly or as a projection. Projected receipts
are not original native inputs to the strict Validation adapter. The native10
assessment, ValidationEvidence, runtime reports, scene/recording/trajectory, and
review images remain exact. Its render receipt and camera metadata need privacy
projection because nested commands contain workflow identifiers. Therefore the
public subset can verify exact measured/typed bytes and frame hashes but does not
recreate every original strict native producer hash binding. Original digest
attestation is explicitly distinct from exact public bytes.

[Source-clear initialization](SOURCE_CLEAR_FIXTURE.md) documents the predeclared
source-only payload placement correction and original source/proxy-floor limit.
Fixture qualification is not authored-asset acceptance. Native smoke stability
and visual review also do not establish opening or loaded-payload retention.
Physics10 separately completed its native workflow with all four checks passing
and a typed visual pass for three seconds of mounted rest. Its exact assessment
explicitly does not establish opening, closing, cavity fidelity or payload retention.
[Postflight](evidence/native10_postflight_execution.json) and the
[actual warnings](evidence/native10_cooking_log_review.json) preserve the initial
strict cooking audit failure, the exact documented nonfatal-message exception,
and CPU collision fallback limitations. No GPU, particle or deformable collision
qualification is inferred. The later final receipt verifies both native Validation
and independent five-seed task acceptance bound to the same authored USD digest.

The [original source](source/drawer_cabinet_1k.gltf) includes its exact BIN and
three texture dependencies. [License provenance](source/license_provenance.json)
retains the CC0 source attribution. Geometry's [native six-view render](runs/drawer_geometry_04/render_evidence/geometry_six_view.png)
and the [Joint02 canonical render](evidence/joint02_canonical.png) are evidence
of the retained geometry, not physical success. Physics07's eight retained
review frames accompany its failed native measured trace. Physics08 retains both
bounded review-frame sets and its unchanged unresolved assessment. Earlier frame
receipts may bind intermediate files subsequently superseded within that run;
the manifest distinguishes published bytes from original retained references.

Run `python tools/verify_public_bundle.py .` for checksum, privacy and reference
review; add `--decode-usd` with `usd-core` installed for decoded layer/dependency
review. The verified CPU environment used OpenUSD25.5; tools import `Usd` before
`Sdf` to initialize file-format plugins. The independently retained [25-candidate decoded scan](publication_review/usd_privacy_review.json)
[08 extension](publication_review/usd_privacy_review_08.json), and
[09/static-fixture extension](publication_review/usd_privacy_review_09.json), and
[Joint03/package/topology extension](publication_review/usd_privacy_review_joint03.json), and
[native10/retained09-test-fixture extension](publication_review/usd_privacy_review_10.json) cover source-host
closure, not portable dependency resolution. Run `python -m unittest discover -s
tools -p test_publication_tools.py` for provider-free negative controls.
`tools/verify_source_arrays.py` compares the original glTF arrays against
a selected USD. The [rebuild qualification](publication_review/portable_public_rebuild_review.json)
reproduced13source/evaluator/specification digests; the historical freeze remains
an explicitly disclosed exception. The [portable dependency qualification](publication_review/portable_public_bundle_r2_portable_closure.json)
found no missing or external dependencies across17USD/package files. Its relative
paths and hashes identify the separate relocated derivatives, not original files
at matching paths in this bundle. Five source-array checks on those derivatives
verified Geometry04, Joint02 and Physics03/07/08 against the original glTF.
Exact native USDs can retain generic absolute runtime locators;
`tools/relocate_usd_paths.py` creates separate portable copies without rewriting
original bytes. A relocated scene is a derivative and its new digest is reported.
A retained trajectory replay is presentation evidence, not a new simulation.
The [native10 replay receipt](task_replay/native10_seed11_v2/presentation_receipt.json)
checks three recorded steps, both drawer/payload positions and
quaternion agreement, and unchanged input/dependency bytes. The native render
command reports OVRTX and its probe is ready, but per-frame renderer identities
are null. Three PNGs are provided separately; the automatically generated
three-frame GIF is deliberately excluded. The exact seed11 request, scene and
replay are included; decompress its complete published trace to restore the
fourth input expected by [render_task_replay.py](render_task_replay.py). Generic
absolute native paths require the same separate relocation or original filesystem
layout described above. Running that script with rendering enabled is a new
presentation render, never a new physical trial or native Validation evidence.
[Decoded replay review](publication_review/usd_privacy_review_native10_replay.json)
retains the source-host dependency and privacy limits.
The Joint03 decoded scan likewise found only generic absolute locators across
its eight selected USD/package/render-stage inputs, with no unresolved source-host
dependencies. The original helper `package_for_joint_03.py` contained a private
GPU identifier: its published executable derivative changes only that selector
to the caller's `CUDA_VISIBLE_DEVICES` environment value. The manifest binds both
helper digests and labels the transformation. It has not been executed as part of
publication checks. All other selected exact helpers retain their original bytes.
The source-preserving Joint03 canonical image has been viewed; it does not display
joint motion or establish physics. Terminal receipts may reference intentionally
omitted private request/trace files; their digests remain retained-only references,
not a claim that the public bundle contains the whole original runtime context.

For a new independent task execution, first verify the bundle and create relocated
assets. Preserve the original binding and write a separate binding that changes
only its filesystem locator, then use the unchanged evaluator with an explicitly
chosen compatible ovphysx Python environment:

```sh
python tools/verify_public_bundle.py . --decode-usd
python tools/relocate_usd_paths.py . ../portable-native
python - <<'PYCODE'
import json
from pathlib import Path
binding = json.loads(Path("evidence/native08_task_bindings.json").read_text())
binding["final_usd"] = str(Path("../portable-native/runs/drawer_physics_08/physics.usda").resolve())
with Path("../portable-native/native08.replay-bindings.json").open("x") as f:
    json.dump(binding, f, indent=2)
PYCODE
python evaluator/source_clear_v1/drawer_evaluate.py   --bindings ../portable-native/native08.replay-bindings.json   --output ../fresh-native08-evaluation --solver /path/to/ovphysx-env/bin/python
```

This launches a new simulation only when explicitly executed. The source-only
rebuild and publication checks do not launch simulation. The copied USD retains
all non-asset opinions; relocation creates a different digest, so new results
must report that digest and must not be substituted for the historical08 receipt.
The evaluator requires its scientific Python dependencies and a compatible native
ovphysx installation; no backend or credential is redistributed here.

The historical `original_drawer_freeze.json` is a privacy projection: its original
digest is attested by the publication manifest but its original bytes are not
provided. The source-clear freeze remains exact. Reconstruct the old12-file input
from11unchanged files plus the exact `original_spec/drawer_acceptance.json` with
`python tools/rebuild_source_clear_fixture.py . /fresh/output`. The helper verifies
all original file hashes and runs the unchanged source-only builder. Its new
freeze differs in timestamp and projected historical-receipt hash; compare the
actual regenerated specification, source and evaluator bytes, not overall freeze
digests. This performs no simulation or authored-asset acceptance.

The [qualified task evidence auditor](task_evidence_audit_v1/README.md) independently
recomputes recorded formulas without launching a solver. Its [20-test qualification](task_evidence_audit_v1/qualification.json)
and [retained09 audit](task_evidence_audit_v1/native09_qualified/audit.json) corroborate
all five historical09 failures; audit consistency is not acceptance. Historical
failed auditor attempts and their code snapshots are retained, not rewritten.
The [independent native10 review](evidence/native10_independent_review/review.json)
and [completed native10 trace audit](task_evidence_audit_v1/native10/audit.json)
corroborate the separate five-seed PASS on the unchanged10 asset. Neither is a
native Validation terminal result. Only seed11's small09 request/scene fixtures are included for the three retained-USD
tests; a full audit rerun additionally needs every seed's exact request, scene,
replay and native log, as well as the decompressed full traces. Missing retained
inputs cannot be replaced with generated records. The test suite can use a
separate relocated copy arranged as `<test-root>/capstone` via `TASK_AUDIT_ROOT`;
its qualification receipt records testing on original retained bytes. A separate
[public-copy check](publication_review/task_audit_portable_tests.json) reproduced
all20 tests using independently relocated published assets, with no dependency
outside that portable directory. From the public bundle, with compatible OpenUSD,
NumPy and pytest installed:

```sh
python tools/relocate_usd_paths.py . ../audit-test-root/capstone
python - <<'PYTESTFILES'
from pathlib import Path
import shutil
for relative in [
    "task_evidence_audit_v1/audit.py", "task_evidence_audit_v1/test_audit.py",
    "evaluator/source_clear_v1/drawer_acceptance.json",
    "evidence/native09_task_bindings.json",
    "evaluations/native09_source_clear_v1/seed_11/request.json",
]:
    destination = Path("../audit-test-root/capstone") / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(relative, destination)
PYTESTFILES
TASK_AUDIT_ROOT="$(cd ../audit-test-root && pwd)" python -m pytest -q ../audit-test-root/capstone/task_evidence_audit_v1/test_audit.py
```

These are analytical and copied-scene tests only. They do not rerun physical trials.

[Completed native leaf accounting](evidence/completed_native_leaf_accounting.json)
records only its named native workflow invocations. It excludes outer development
and coordinator work, independent reviewers, native Validation, replay, tests,
probes and GPU cost. Dollar cost is unknown; missing times and counts are not
filled in. This partial accounting is not an end-to-end cost or workflow-efficiency
comparison.


The [actual native Validation terminal](runs/drawer_validation_01_prepared/native_run/validation_terminal_receipt.json)
now records completed/pass/review-accept. Its four required gates pass; optional
standalone cross-stage integrity remains not_evaluated. The [final assessment](runs/drawer_validation_01_prepared/native_run/canonical_validation_assessment.json)
and [independent native review](runs/drawer_validation_01_prepared/independent_review.json)
are exact safe final records. Schema-required gate/finding rationale strings are
reviewed final evidence-based explanations intended for readers; this narrow
exception does not publish private prompts or raw model reasoning. A boolean
base_url_configured only states whether configuration exists and contains no URL.
The accepted plan and plan patch remove the private child session identifier and
are explicit projections. The exact terminal retains original hashes for those
targets: the public verifier labels them ORIGINAL_RETAINED_BYTES_TARGET_IS_PROJECTION,
not exact public reproduction of the entire original native input closure. The
[export manifest](evidence/validation01_terminal_export.json) attests the retained
original42-file export, not that every export member is selected for publication.
The [final conjunction](evidence/final_capstone_conjunction.json) and its
[independent review](evidence/final_conjunction_independent_review.json) bind the
same authored asset, native terminal and separately audited five-seed task.
Geometry's original conditional handoff and formal SimReady not_evaluated remain
unchanged. Original topology warnings, estimated mass/materials, ideal connected-actor
collision exclusion, CPU-only scope and missing initial native pose/contact actor
identifiers remain limits. Extensive repairs are outside the frozen pilot budget.
The original Physics03 root console was overwritten by the later Physics07
launcher; surviving native receipts are retained and that missing console is not
reconstructed.

[Validation leaf accounting](evidence/validation01_leaf_accounting.json) is a
separate partial counter receipt. Outer development, assessor/reviewer work and
compute dollars remain outside these named native-leaf counters; it is not a total
cost or workflow-efficiency comparison.

[Code changes](code/capstone_changes.patch) are separate from run outputs.
[Validation policy example](validation_contract_review/POLICY_EXAMPLE.md) keeps
native smoke and the independently frozen physical task gate distinct.
