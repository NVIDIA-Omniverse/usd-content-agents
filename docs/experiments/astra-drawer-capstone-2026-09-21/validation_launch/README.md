# Native Validation launcher: prepare first, execute only after genuine evidence

These files sit outside the shared capstone checkout. Preparation calls the actual
strict native consumer from commit78004ff/current e640b8d, the installed native CLI
parser and real typed assessment schemas. It does not launch a model, renderer or
solver and does not author an assessment, approval or terminal success receipt.

The current09 workflow is unresolved and its five task trials failed. It cannot
prepare this launcher. Preserve03/06/07/08/09 failures. A repaired asset needs a
new native Physics output and a new complete five-seed task on that exact digest.
If runtime/render/visual evidence is stale or contaminated by an earlier failed
aggregate, obtain fresh genuine runtime, OVRTX frames and an actual AstraUltra
review in a new native Physics output. Do not delete failed checks, relabel a
failed aggregate, manufacture `approve`, or reuse a task report from another USD.

Preparation requires the original source-clear specification SHA
`90cde935ce539607c6a770f6ff0ee8cc8c5bc37ba2f18c76802e5cdfe70bf1cd`, exactly
seeds11/23/47/83/131, matching aggregate and per-trial reports, the actual submitted
bindings SHA/path, the reviewed final asset SHA, native workflow `pass`, the strict
native runtime/render/typed-review closure, and original Geometry's conditional
handoff. A completed failed task may be retained alongside a narrow native check;
it always keeps the final capstone conjunction false. Preparation is not acceptance. Task prechecking binds the aggregate and per-seed
reports, asset, bindings and specification. It does not independently consume or
verify each task request, scene, full trace or evaluator implementation bytes;
root's separate independent task audit and public trace manifest own that evidence.
The native behavior bundle has its own stricter measured runtime/render closure.

The Geometry v3 receipt represents this handoff with top-level
`sim_ready_status: not_evaluated` and `metadata.handoff_ready: conditional`.
Preparation validates the strict ValidationEvidence schema, `workflow: geometry`,
and `metadata.schema_version: content-agent-workflows.geometry.v3`; it preserves
the original warnings and unresolved topology issues. The earlier launcher
incorrectly expected `sim_ready_status: conditional` and refused Geometry04.
Its source, qualification and reproduced refusal are retained under
`history/geometry_handoff_assumption/`. This correction does not change Geometry
evidence, grant a waiver, or turn its handoff into a SimReady pass.

The installed agentic coordinator defaults to canonical USD visual evidence.
This launcher keeps that supported route: actual launch will create fresh native
OVRTX canonical renders on the same USD. It requires `render_valid`, `physics_sane`
and `physical_behavior`, with all-required evidence and dependencies. These three
capabilities are declared provider-free by the native coordinator; the only model
planning child is `gpt-6-astra` with effort `ultra`. No VLM provider is configured.
If a child nevertheless selects `look_right`, retain the actual refusal/warning;
never waive it or pretend a provider ran. `physics_sane` uses the real USD inspector
with physics expected; no relaxed scene, body, collider or joint criteria are added.

After the final asset is independently hashed and its new native/task evidence is
terminal, copy this folder to the host (outside the active repository). Set the
asset hash from the reviewed retained receipt, not by silently replacing a stale
expected value at launch. Example for the separate future Physics10 output:

```sh
CAPSTONE=/opt/astra-content-value-20260921/capstone
PY="$CAPSTONE/repo/.venv/bin/python"
"$PY" -B "$CAPSTONE/validation_launch/prepare_validation.py" \
  --repo "$CAPSTONE/repo" \
  --expected-commit e640b8d6aa667745830db5e9bf87fd1b4cd763cb \
  --asset "$CAPSTONE/runs/drawer_physics_10/physics.usda" \
  --expected-asset-sha256 "$REVIEWED_FINAL_ASSET_SHA256" \
  --physics-run "$CAPSTONE/runs/drawer_physics_10" \
  --task-report "$CAPSTONE/evaluations/native10_source_clear_v1/report.json" \
  --bindings "$CAPSTONE/evidence/native10_task_bindings.json" \
  --spec "$CAPSTONE/evaluator/source_clear_v1/drawer_acceptance.json" \
  --geometry-evidence "$CAPSTONE/runs/drawer_geometry_04/geometry_validation_evidence.json" \
  --output "$CAPSTONE/runs/drawer_validation_01_prepared"
```

Names for a later candidate must identify its actual outputs. The expected commit
must identify the reviewed code; a changed commit or any changed bound code file
requires new preparation. GPU isolation/runtime environment remains the existing
explicitly allocated capstone lane; this launcher does not select another GPU or
install/change a runtime. `commands.json` records the supported native commands.
The stage launcher explicitly sets cwd and prepends the reviewed repository and
agentic package paths to PYTHONPATH, so a shared installed .venv cannot silently
select the old repository. It inherits runtime credentials without recording them.
The preparation prints its receipt hash. Copy that exact hash into subsequent
commands. Preview defaults to no execution:

```sh
"$PY" -B "$CAPSTONE/validation_launch/run_stage.py" \
  --prepared "$CAPSTONE/runs/drawer_validation_01_prepared" \
  --preparation-sha256 "$REVIEWED_PREPARATION_SHA256" --stage run
```

Append `--execute` only when root authorizes the actual lane/model run. A stage is
write-once. Every invocation rechecks the source, native closure, task reports,
code/commit, policy and command bytes. Logs are retained as `.private.log` and are
not publication inputs. Run the stages in this order:

1. `--stage run --execute`: actual AstraUltra planning plus exact native checks and
   canonical OVRTX render. Return0 is **not** terminal Validation acceptance.
2. `--stage collect --execute`: native `validate collect-evidence`, producing
   `native_run/standalone_validation_evidence.json` and its native evidence index.
3. Root's actual AstraUltra outer assessor reads that exact index and its bound
   evidence, preparation/task/Geometry receipts, and actual native results. It
   writes `outer_assessment.json` conforming to `assessment.schema.json`. No sample
   pass record is supplied. Required gates and findings cannot be waived/deferred.
   Then `--stage assess --execute` calls native `validate assess` on those bytes.
4. An independent actual AstraUltra reviewer reads the exact assessment and the
   evidence, then writes `independent_review.json` conforming to `review.schema.json`.
   `--stage review --execute` calls native `validate review-assessment`. Inspect its
   real terminal receipt and accepted disposition; a completed review of a failure
   is not a passing asset.

Assessment and review are human-visible native typed records authored by the
actual AstraUltra agents, not outputs synthesized by these scripts. Preserve
Geometry's conditional source/visual handoff and all earlier failures in their
scope. Native single-body smoke/visual behavior does not establish joint motion,
loaded opening or retention. Final capstone success requires both a genuinely
passing, unwaived native terminal assessment/review and five task PASS results on
the same exact final asset SHA. No universal SimReady or actuator realism claim.

For provider-free qualification, reuse the repository's qualified synthetic native
fixture and actual parser/consumer. Set `CAPSTONE_TEST_REPO` to the reviewed checkout,
`CAPSTONE_TEST_SPEC` to the exact source-clear spec, add this directory, repository,
its `tests`, both agentic packages and `apps/usd_cli/src` to `PYTHONPATH`, then run:

```sh
python -B -m pytest -o addopts= -q test_precheck.py
```

No authored USD is mutated and no model/solver/render is invoked by these tests.
The actual adapter's larger99-test qualification remains a separate earlier record.
