# Astra and Content Agents: ten-source development pilot

This is a measured experiment on original online assets, with a saved physical drawer task and independently checked trajectories. Read the [results and per-asset outcomes](report/RESULTS.md) together with the [method and limitations](report/METHOD.md). The configured installation, evaluator errata and observed isolation failures limit causal conclusions; this pilot is not proof of a workflow advantage.

The two arms use Astra Ultra, identical source bytes and task briefs per pair, the same installed low-level tools, matched host/GPU placement, a 40-minute authoring budget and two allowed repairs. The intervention is the pinned public Content Agents workflow. Four full 48 GB NVIDIA L40 GPUs on Horde were verified and used as the environment. Ten curators acquired the sources; each source then received two fresh scored authoring contexts. Five physics seeds test each completed authored asset, not five independent authoring attempts.

## What can be inspected

- [Four-GPU verification and concurrent renderer preflight](report/hardware_verification.json). This synthetic fixture establishes environment readiness, separately from physical asset acceptance.
- [Structured results](report/results.json), [CSV](report/results.csv), [frozen protocol](protocol/protocol.json), and [global protocol audit](report/global_protocol_audit.json).
- [Ten-source catalog, upstream links and licensing notes](report/sources.json), [exact source closure hashes](protocol/dataset.json), and [pinned download URLs](protocol/source_downloads.json).
- Frozen independent evaluator code under `evaluator/`, complete source reference inventories, original and versioned corrected acceptance receipts, author claims, sanitized execution/usage receipts and protocol audits.
- The original CC0 drawer, unchanged accepted USD/bindings, five compressed measured traces, [trajectory plot](report/drawer/drawer_validation.png), and [per-seed measurements](report/drawer/summary.json).
- Three actual OVRTX images of independent trace poses: [closed](visuals/pilot-v1/01_drawer/plain_astra/closed_initial.png), [open with the free payload](visuals/pilot-v1/01_drawer/plain_astra/open_loaded.png), and [returned closed](visuals/pilot-v1/01_drawer/plain_astra/closed_returned.png), with [portable reproduction instructions](report/drawer_replay_reproduction.md) and projected renderer/visual-review receipts.
- [Geometry workflow diagnosis](report/workflow_geometry_diagnosis.md), [reference regeneration evidence](report/reference_regeneration.md), and [dependency review](report/independent_method_review.md).
- The [engine contract/witness measurement review](evaluation_adjudications/pilot-v1/06_engine/plain_astra/measurement_review/assessment.json) and [synthetic diagnostic instructions](evaluator/errata/engine_auxiliary_body_review/README.md), preserved separately from the original scored receipts.
- The [supplementary static delivery audit](performance_research/structural_delivery_v1/README.md) for CA07–10, with exact receipts, qualified code, synthetic fixtures and sanitized input manifests. It proves absence of mandatory physics schemas; source fidelity and native runtime remain unassessed by that audit.

Physical task acceptance and experimental eligibility are different columns. A passing physical artifact does not cure a contaminated authoring run. Known evaluator numerical defects do not establish false passes; frozen and corrected measurements remain separate. Dollar rates and unmeasured human review time are unknown, not zero.

The [drawer source/proxy clearance diagnostic](report/drawer/source_proxy_clearance.json) also bounds the physical example: its accepted floor collider is 4.50 mm below the original visible floor at three checked positions. Frozen acceptance measures the authored collision approximation, not exact contact fidelity to that rendered surface.

## Reproduce the drawer

Use Linux and the recorded structural and native solver environments in [the reproduction guide](evaluator/general/REBUILD.md). The structural environment and OvPhysX environment must remain separate because they provide different USD runtimes. The public repository baseline is `a96faf9cb2f5c1f655fe0d60c0ccf57e3477b1aa`; dependency pins are recorded, not a claim that arbitrary future resolutions are equivalent.

From this experiment directory:

```sh
python3 scripts/verify_public_bundle.py --root "$PWD"
python3 scripts/fetch_sources.py --root "$PWD" \
  --manifest protocol/source_downloads.json --case 01_drawer --verify-only

export STRUCT_PY=/absolute/path/to/structural-env/bin/python
export SOLVER_PY=/absolute/path/to/ovphysx-env/bin/python
# The unchanged frozen adapter sets this scratch path for its child process.
# Provision it writable, or mount writable scratch there in a container.
test -d /opt/astra-content-value-20260921/tmp
test -w /opt/astra-content-value-20260921/tmp
"$STRUCT_PY" evaluator/drawer_evaluate.py \
  --bindings "$PWD/runs/pilot-v1/01_drawer/plain_astra/bindings.json" \
  --solver "$SOLVER_PY" --output "$PWD/new-drawer-evaluation"
```

Choose a new output directory each time. The evaluator never imports author-written Python. It checks original source triangles and authored joint/body structure, removes authored drives, adds its own free payload, and applies bounded external forces in fresh native simulation for five seeds. The drawer contact-unit [erratum](evaluator/drawer_contact_units_errata.md) explains the unchanged raw impulse predicate and force presentation.

The unchanged scored `final.usd` uses relative `assets/textures/` paths. Its [relocation receipt](report/drawer_portability_receipt.json) confirms that the complete submission opens outside the original run directory. To check another relocation without editing those bytes:

```sh
"$STRUCT_PY" scripts/check_drawer_portability.py \
  --source "$PWD/runs/pilot-v1/01_drawer/plain_astra" \
  --destination /absolute/path/to/new-drawer-relocation
```

Rendering the packaged snapshots is separate from rerunning physics. Follow the [replay guide](report/drawer_replay_reproduction.md), which copies only immutable snapshot inputs into a fresh output directory before invoking `render_portable_drawer_snapshots.py`. The original preparation script is retained for provenance and requires the complete retained evaluation directory; it is not the public rendering entrypoint.

## Reproduce the wider pilot

Fetch sources with `scripts/fetch_sources.py`, selecting a case with repeated `--case` arguments or all cases by omitting the filter. Downloads are checked against exact sizes and SHA256; altered upstream bytes are rejected. Check each upstream license before reuse.

Non-drawer CAD geometry and most author submissions are retained locally, not redistributed in this bundle. The full frozen inventories contain the geometry digests needed for regeneration. Cases 03 and 05 were independently regenerated and verified; case 03 needs the [hash-alignment helper](report/reference_regeneration.md) because a loader assigns nondeterministic primitive suffixes. This qualification does not prove byte-stable regeneration for every large CAD case. The included drawer is the self-contained physical replay; the wider source/evaluator release exposes the experiment and its limitations without claiming that all twenty submissions are publicly replayable.

The [source-distance erratum guide](evaluator/errata/source_distance_v1_1/README.md) gives a staged-retention command with an explicit `--qualification .../qualification_r2.json`. It additionally requires the full original submission, output manifest, reference NPZ files and native evaluation records; the public summary receipts alone cannot run that adjudication.

Authoring harnesses are included for inspection. A future clean comparison must add operating-system isolation for author/evaluator files and process lists, a global simulation semaphore, and complete workflow-dependency/Git-provenance preflight before a new freeze. These controls were not retroactively applied to the recorded experiment.

## Accounting and evidence integrity

In a disposable copy of the bundle, run `python3 scripts/summarize_results.py --root "$PWD"` to regenerate the table from published receipts. This writes `report/results.json` and `report/results.csv`, so the copied publication manifest will no longer describe those newly generated bytes. To calculate dollars, fill a copy of `protocol/rates.template.json` with unit rates, their source and effective date, then pass `--rates /path/to/rates.json`. Cached input is separated from uncached input; reasoning output is not billed twice. Ratios include failed attempts and are undefined when the denominator is zero. Shared setup, evaluation and idle reservations remain separate from per-arm authoring time.

`publication_manifest.json` lists exact published digests and original retained digests. A projected receipt removes runtime identifiers or private session detail and is explicitly marked; it is not presented as the original hashed receipt. Scored USD, frozen evaluator code and original acceptance records remain unchanged. Credentials, internal service addresses, private model reasoning and raw private tool/session records are excluded.

The fresh case03 audits retain hashes of original evidence even when the public file is a projection. The manifest's `audit_reference_bindings` index classifies every canonical case03 evidence reference as an exact published file, a published projection with its own digest, or evidence retained outside this bundle. Omitted evidence is not publicly byte-verifiable. The two prior current audit receipts are preserved separately under `audit_history`; these are not reconstructions of any lost original receipt.

For the supplementary CA07–10 delivery findings, the aggregator accepts only four pinned receipt digests and their exact qualified auditor/qualification digests. Task, submission and output manifests are published unchanged; those manifests bind the retained USD and bindings without requiring those geometry files in the public bundle. A launch projection must match its published hash and retained-original digest in the publication manifest. If a retained scene or layer is present, its bytes are also checked. This rule can establish FAIL from missing mandatory physics, cannot grant or override PASS, and preserves the original frozen status and unassessed source/runtime scope.

The drawer source is Ulan Cabanilla's [Drawer Cabinet on Poly Haven](https://polyhaven.com/a/drawer_cabinet), released under CC0. Other sources remain governed by their upstream licenses and the qualifications in the catalog. Neither this experiment nor its simulations certify real-world mechanical properties or robot behavior.

To reproduce the read-only source/proxy diagnostic in the structural environment, choose a new output file:

```sh
"$STRUCT_PY" scripts/check_drawer_payload_clearance.py --root "$PWD" \
  --output /absolute/path/to/new-source-proxy-clearance.json
```
