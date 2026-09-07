# Content Agents Dashboard

Web UI for browsing standardized benchmark results across agentic content-agent
workflows.

Use Node.js 20 or newer to install, test, and run the dashboard.

The dashboard discovers runs under the benchmark artifact root and consumes each
run's files directly:

```text
<artifact-root>/<workflow>/<run-id>/benchmark_run.json
<artifact-root>/<workflow>/<run-id>/bundle/run.json
<artifact-root>/<workflow>/<run-id>/score/suite_result.json  # optional
```

By default, `<artifact-root>` is the repository's `.benchmark/workflows` directory.
Set `CONTENT_BENCHMARK_ROOT` when the runs live elsewhere. The Vite server exposes
the files under `/benchmark-data` and synthesizes `/benchmark-data/index.json` for
run discovery. This URL layout can be mirrored by static or S3 hosting later.

Only workflows with standardized artifacts appear in the UI. Material Agentic
and Material Fixed-Pipeline Workflow share the Material regression dataset, and
Physics Fixed Pipeline and Physics Agentic share the PhysX-Mobility property
suite (same manifest, prediction parser, and scorer, so the two execution modes
are directly comparable). Public CAD comparisons, CAD-to-SimReady, and its
Geometry alias are supported when publishers emit the standard run files and
Geometry extension documented in
[`GEOMETRY_BENCHMARK_BUNDLES.md`](GEOMETRY_BENCHMARK_BUNDLES.md).

The browser treats `score/suite_result.json` as authoritative. Workflow status,
handoff state, and bundled findings remain descriptive and never replace suite
scoring. A refined result must use a new run ID and retain the prior published
result under the publisher's storage policy. The dashboard's 30-second polling
only discovers published artifacts and does not execute or resume benchmarks.
For public CAD comparisons, prompt-identifiability audits are shown as
digest-bound evidence and a separate detected-conflict-free reporting stratum;
they never rewrite the raw pairwise verdict.

Geometry runs are exposed only after exact integrity coverage is established for
their authoritative documents and every local artifact path the dashboard can
consume. The loopback artifact server rejects noncanonical or unbound URLs and
serves requested bytes from the same bounded snapshot used for verification.
Geometry artifact serving requires Linux procfs so each opened descriptor can
be proven to resolve to the expected file under the run root. Missing or
malformed snapshot tokens are rejected before verification; valid concurrent
requests for one run share a single verification, and process-wide verification
concurrency and pending work are bounded.

Run locally against the default artifact root:

```bash
cd apps/content_agents_dashboard/web
npm ci
npm run dev
```

Pass `-- --host 0.0.0.0` only when LAN access to the local benchmark artifacts
is intentional.

Run against another artifact root:

```bash
cd apps/content_agents_dashboard/web
CONTENT_BENCHMARK_ROOT=/path/to/workflows npm run dev
```

For shared hosting, export a curated copy instead of serving the raw benchmark
tree. The export contains standardized run documents plus only the evidence and
reports linked by each bundle. Note: for digest-verified physics references the
export also ships `*.provenance.exact.json` sidecars carrying the VERBATIM
OVRTX records — including producing-machine absolute paths — because the
fail-closed evidence rule requires the exact render metadata to stay
recoverable and integrity-bound beside the shared artifact. Records whose
endpoints could carry credentials are refused rather than shipped:

```bash
python apps/content_agents_dashboard/export_benchmark_runs.py \
  --artifact-root .benchmark/workflows \
  --output-root .benchmark/dashboard-workflows \
  --workflow mesh_segmentation \
  --run-id <run-id> \
  --allow-source-root .data/regression
```

A bundle may only pull in files from its own run directory or from a root named
with `--allow-source-root`; anything else fails the export naming the offending
path. Since the export is built to be shared, an unbounded path in a malformed
bundle would otherwise copy local files straight into a hosted artifact. Pass
the dataset root as shown, because `source_usd` and the reference renders are
recorded outside the run directory.

`--artifact-root` expects the `<workflow>/<run-id>/benchmark_run.json` layout
the benchmark CLI writes.

Bind Vite to a specific Tailscale address when the dashboard should be visible
only over the tailnet:

```bash
CONTENT_BENCHMARK_ROOT="$PWD/.benchmark/dashboard-workflows" \
  npm --prefix apps/content_agents_dashboard/web run dev -- \
  --host "$(tailscale ip -4)"
```

Build:

```bash
cd apps/content_agents_dashboard/web
npm run build
```
