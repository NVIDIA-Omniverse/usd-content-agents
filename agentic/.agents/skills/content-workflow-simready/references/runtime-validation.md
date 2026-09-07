# SimReady Runtime Validation

World Understanding integrates SimReady runtime testing as a consumer of the
published SimReady Benchmark package. The package owns test discovery, planning,
engine sessions, execution, reporting, batch/resume support, and result stamping.
The workflow adapter owns explicit invocation, source identity checks, artifact
containment, a compatibility projection into the Validation Agent
`physical_behavior` contract, and a lossless per-test projection into the shared
verified-operation v1 ingress contract.

## Environment

Keep the benchmark and Kit plugin outside the main World Understanding
environment because their OpenUSD and Omniverse dependencies can conflict with
the application's active `pxr` provider.

```bash
uv venv --python 3.12 /path/to/benchmark-venv
uv pip install \
  --python /path/to/benchmark-venv/bin/python \
  "simready-benchmark[kit]==2026.6.5"
/path/to/benchmark-venv/bin/simready-benchmark --version
```

Both pinned wheels are public PyPI releases:
[`simready-benchmark==2026.6.5`](https://pypi.org/project/simready-benchmark/2026.6.5/)
and
[`simready-benchmark-engine-kit==2026.6.5`](https://pypi.org/project/simready-benchmark-engine-kit/2026.6.5/).
The `kit` extra above installs the latter package; Isaac Sim itself remains a
separate runtime installation.

The adapter accepts only `simready-benchmark==2026.6.5` and, when visible in
the same environment, `simready-benchmark-engine-kit==2026.6.5`. Set
`CONTENT_WORKFLOW_SIMREADY_BENCHMARK_EXECUTABLE` or pass
`--benchmark-executable` explicitly. The benchmark package does not install an
Isaac Sim runtime; `engines.toml` must reference a working engine or remote
worker configuration.

The SimReady Benchmark QA guide for 2026.6.4 remains useful for setup, but the
adapter's pinned package and installed `--help` output are authoritative.
The checked
`agentic/packages/content_agent_workflows/tests/fixtures/simready_benchmark_2026_6_5_contract.json`
fixture records the 2026.6.5 wheel's version banner, relevant `nargs` shapes,
public wheel digests, exit-code mapping, report schema and paths, per-status
`result.json` persistence, rollup rules, and mirror-only stamping contract.
Update that fixture from the installed wheels before changing any adapter
assumption.

## Required Inputs

- one local `.usd`, `.usda`, or `.usdc` asset;
- the runtime Foundation `sr_specs` directory from a user-provided compatible
  checkout, such as the public
  [`NVIDIA/simready-foundation`](https://github.com/NVIDIA/simready-foundation)
  repository;
- an explicit `engines.toml`;
- a discoverable runtime test pack, either configured in `engines.toml` or
  passed with `--tests-path`;
- at least one applicable stamped feature or an explicit `--feature` override.

Static Foundation validation and runtime Benchmark validation are separate:

```text
Static gates (selected)       -> Isaac Asset Validator / SimReady Foundation
Runtime behavior (selected)   -> simready-benchmark -> Kit/Isaac Sim
Final shared assessment       -> verified-operation ingestion, no simulator rerun
```

## Result Handling

The benchmark runs under `<output-dir>/simready-benchmark`. The adapter writes:

- `simready-runtime-validation.json`, the WU envelope;
- `validation-template-result.json`, the `physical_behavior` projection;
- `simready-benchmark.stdout.log` and `simready-benchmark.stderr.log`;
- `simready-runtime-verified-operations/simready_runtime_verified_operations.json`,
  indexing one verified-operation envelope per native test;
- `simready-runtime-verified-operations/runtime_input_bindings.json`, retaining
  the exact `sr_specs`, `engines.toml`, project-config, and custom-test file/tree
  identities used by the run;
- the exact native `simready-benchmark/plan.json` used to bind the requested
  asset to the report's directory-safe asset key;
- the untouched native report at
  `simready-benchmark/report/test_results_index.json`;
- the native `run_summary.json`, `state/events.jsonl`, engine logs, results, and
  media.

Each verified operation carries a stable operation and gate identity derived
from the native profile, feature, test, engine, and engine-version contract. It
binds the exact source and dependency closure, unmodified aggregate report, raw
per-test result, metrics, media, logs, benchmark executable, profile, backend,
runtime-input manifest and files, verifier, and projector identities. The
projector reverifies the runtime inputs after Benchmark exits and before
publication. The projector is provider-free and non-executing. Shared
Validation can ingest each envelope independently without launching Benchmark,
Kit, or Isaac Sim and without collapsing sibling test dispositions.

In pinned Benchmark 2026.6.5, multiple `--runtime` values filter the eligible
engine configurations; they do not form a test-by-engine execution product.
Each work item is consumed once, and the reporter stores one `result.json` under
the asset-and-test path. A report containing the same asset/test pair more than
once is therefore rejected as malformed instead of ambiguously binding one
per-test file to multiple engine results.

The `physical_behavior` file remains a compatibility view. The verified-operation
envelopes are the canonical carry-through for already produced runtime evidence.

An explicit `--report` path must remain under `<output-dir>` and outside the
managed `simready-benchmark` directory. This prevents report publication from
overwriting the source asset or benchmark inputs.

Treat `<output-dir>` as a single-run workspace. Starting a new invocation in an
existing directory invalidates and removes any prior
`simready-runtime-verified-operations` publication before input validation or
benchmark executable resolution completes. Invalidation also rotates a validity
token bound into every envelope, so a copied envelope from the prior publication
cannot bypass revocation through direct ingestion. Use a fresh output directory
when a previous accepted publication must remain loadable.

The adapter maps native test statuses without deleting the original value:

| Native status | WU disposition |
|---|---|
| `pass` | `pass` |
| `fail` | `fail` |
| `skipped` | `not_evaluated` |
| `incomplete` | `error` |
| `blocked` | `blocked` |

Engine crash/stuck/error events produce an adapter `error`, even when the native
synthetic test result says `fail`. Exit code `4` or readiness `not_ready`
produces `blocked`. A zero-test plan is an integration error, never a pass.

Source asset dependencies are digested before execution and verified again
afterward. Native stamping is off by default. In the pinned 2026.6.5 runtime,
enabling it copies the source USD into `simready-benchmark/results/<asset-key>`
and stamps that mirror; the source asset remains read-only. This behavior is
pinned by the recorded package contract, and the adapter still rejects the run
if any source or dependency identity changes.

The canonical CLI surface is:

```bash
content-workflow-cli simready validate-runtime asset.usda \
  --output-dir ./simready-runtime \
  --sr-specs /path/to/sr_specs \
  --engines-toml /path/to/engines.toml \
  --tests-path /path/to/runtime-tests \
  --benchmark-executable /path/to/benchmark-venv/bin/simready-benchmark
```

The standalone `content-workflow-simready-runtime-validate` entry point remains
available for compatibility and invokes the same adapter.
