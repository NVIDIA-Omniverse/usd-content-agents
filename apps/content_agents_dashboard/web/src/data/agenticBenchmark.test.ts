// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clearReferenceVerificationCache,
  compareRunsNewestFirst,
  loadBenchmarkBundle,
  loadBenchmarkIndexFingerprint,
  sha256HexSync,
} from "./agenticBenchmark";

const indexUrl = "/benchmark-data/index.json";
const geometryArtifactVersion = `sha256:${"6".repeat(64)}`;
const geometryArtifactVersionQuery = `?artifact_version=${encodeURIComponent(
  geometryArtifactVersion,
)}`;

function fixtureDocuments() {
  return {
    [indexUrl]: {
      schema_version: "content-agent-benchmark-index.v1",
      generated_at: "2026-07-22T00:00:00Z",
      runs: [
        {
          artifact_base_url: "/artifacts/run",
          artifact_version: "v1",
          bundle_url: "/bundle.json",
          created_at: "2026-07-22T00:00:00Z",
          evaluation_url: null,
          manifest_url: "/manifest.json",
          run_id: "material-agentic-run",
          score_url: "/score.json",
          status: "complete",
          workflow: "material-agentic",
        },
      ],
    },
    "/manifest.json": {
      schema_version: "content-agent-benchmark-run.v1",
      run_id: "material-agentic-run",
      workflow: "material-agentic",
      status: "complete",
      created_at: "2026-07-22T00:00:00Z",
      case_count: 1,
      metadata: { git: { branch: "main", commit: "abcdef12" } },
    },
    "/bundle.json": {
      schema_version: "material-agent-benchmark-run.v1",
      run_id: "material-agentic-run",
      label: "Material Agentic material-agentic-run",
      workflow: "material-agentic",
      created_at: "2026-07-22T00:00:00Z",
      assets: [
        {
          asset_id: "asset_a",
          status: "completed",
          metrics: { runtime_seconds: 12 },
          renders: {
            final: "assets/asset_a/final image.png",
            turntable: "assets/asset_a/final turntable.mp4",
          },
          references: [{ path: "assets/asset_a/reference.png" }],
        },
      ],
    },
    "/score.json": {
      run_id: "material-agentic-run",
      created_at: "2026-07-22T00:00:00Z",
      status: "fail",
      cases: [
        {
          workflow: "material-agentic",
          asset_id: "asset_a",
          status: "fail",
          signals: [
            {
              name: "finished",
              ok: false,
              severity: "blocking",
              detail: "Workflow exited nonzero.",
            },
          ],
        },
      ],
    },
  };
}

function geometryFixtureDocuments(workflow = "geometry-cad-to-simready") {
  const runId = "geometry-cad-to-simready-run";
  const artifactBase = "/artifacts/geometry";
  const sha256 = (character: string) => character.repeat(64);
  return {
    [indexUrl]: {
      schema_version: "content-agent-benchmark-index.v1",
      generated_at: "2026-08-04T12:00:00Z",
      runs: [
        {
          artifact_base_url: artifactBase,
          artifact_directory: runId,
          artifact_version: geometryArtifactVersion,
          bundle_url: "/geometry-bundle.json",
          created_at: "2026-08-04T11:00:00Z",
          evaluation_url: null,
          manifest_url: "/geometry-manifest.json",
          run_id: runId,
          score_url: "/geometry-score.json",
          status: "complete",
          workflow,
          geometry_integrity: {
            schema_version: "content-agent-benchmark-geometry-integrity.v1",
            artifacts: [
              {
                path: "benchmark_run.json",
                sha256: sha256("3"),
              },
              {
                path: "bundle/run.json",
                sha256: sha256("4"),
              },
              {
                path: "bundle/assets/fixture_valve/content_agents_manifest.json",
                sha256: sha256("a"),
              },
              {
                path: "bundle/assets/fixture_valve/geometry_validation_evidence.json",
                sha256: sha256("b"),
              },
              {
                path: "bundle/assets/fixture_valve/output.usdc",
                sha256: sha256("9"),
              },
              {
                path: "bundle/assets/fixture_valve/renders/front.png",
                sha256: sha256("c"),
              },
              {
                path: "bundle/assets/fixture_valve/renders/front_camera.json",
                sha256: sha256("d"),
              },
              {
                path: "bundle/assets/fixture_valve/renders/render_manifest.json",
                sha256: sha256("e"),
              },
              {
                path: "bundle/assets/fixture_valve/source.usdc",
                sha256: sha256("f"),
              },
              {
                path: "score/suite_result.json",
                sha256: sha256("5"),
              },
            ],
          },
        },
      ],
    },
    "/geometry-manifest.json": {
      schema_version: "content-agent-benchmark-run.v1",
      run_id: runId,
      workflow,
      status: "complete",
      created_at: "2026-08-04T11:00:00Z",
      case_count: 1,
      metadata: {
        git: { branch: "geometry-release", commit: "12345678", dirty: false },
        provenance: {
          target: {
            requested_ref: "refs/heads/geometry-release",
            resolved_ref: "12345678",
            branch: "geometry-release",
            commit: "12345678",
            commit_timestamp: "2026-08-04T10:00:00Z",
            remote: "origin",
            remote_url: "https://example.test/target.git",
            dirty: false,
          },
          harness: {
            requested_ref: "main",
            resolved_ref: "abcdef12",
            branch: "main",
            commit: "abcdef12",
            commit_timestamp: "2026-08-04T09:00:00Z",
            remote: "origin",
            remote_url: "https://example.test/harness.git",
            dirty: false,
          },
          dataset: {
            manifest: "cad_to_simready.jsonl",
            manifest_sha256: sha256("1"),
            content_sha256: sha256("2"),
            case_count: 1,
            file_count: 1,
          },
        },
        execution: {
          schema_version: "content-agent-execution-metadata.v1",
          runner_hardware: {
            architecture: "x86_64",
            os: "Linux",
            os_release: "test",
            cpu_model: "Test CPU",
            logical_cpu_count: 8,
            memory_total_bytes: 17179869184,
            gpus: [
              {
                name: "Test GPU",
                memory_total_bytes: 8589934592,
                driver_version: "test",
              },
            ],
          },
          renderers: {
            workflow: null,
            benchmark_evidence: {
              backend: "usd-cli",
              deployment: "local",
              service: "World Understanding OVRTX",
              function_id: null,
              endpoint_host: null,
              renderer: "ovrtx",
              settings: {
                image_width: 640,
                image_height: 640,
                render_quality: "final",
                ovrtx_render_mode: "pt",
                ovrtx_num_sensor_updates: 500,
              },
            },
          },
          usage: {
            input_tokens: 1200,
            cached_input_tokens: 300,
            output_tokens: 240,
            total_tokens: 1440,
            cost_estimate_status: "estimated",
            cost_estimate: {
              actual_billing: false,
              pricing_schema_version: "test-pricing.v1",
              pricing_model: "test-model",
              model_source: "recorded",
              currency: "USD",
              standard_api_equivalent_usd: 0.12,
              long_context_upper_bound_usd: 0.18,
              cached_input_tokens_reported: true,
              pricing_as_of: "2026-08-04",
              pricing_source: "test",
              caveat: "Fixture estimate.",
            },
          },
        },
        vlm: { backend: "test", model: "test-model" },
      },
    },
    "/geometry-bundle.json": {
      schema_version: "content-agent-benchmark-bundle.v1",
      run_id: runId,
      label: `CAD-to-SimReady ${runId}`,
      workflow,
      created_at: "2026-08-04T11:00:00Z",
      source: "geometry-evidence-publisher",
      assets: [
        {
          asset_id: "fixture_valve",
          name: "fixture valve",
          prompt: "Build a fixture valve with two protected openings.",
          status: "certified",
          tags: ["public_canary", "rigid_body"],
          source_usd: "assets/fixture_valve/source.usdc",
          output_usd: "assets/fixture_valve/output.usdc",
          metrics: {
            runtime_seconds: 42.5,
            watertight: true,
            protected_opening_count: 2,
          },
          renders: { front: "assets/fixture_valve/renders/front.png" },
          geometry: {
            schema_version: "content-agent-benchmark-geometry.v1",
            benchmark_family: "cad_to_simready",
            outcome: "certified",
            handoff_ready: "yes",
            artifacts: [
              {
                kind: "source_usd",
                label: "Source USD",
                path: "assets/fixture_valve/source.usdc",
                sha256: sha256("f"),
                media_type: "application/octet-stream",
              },
              {
                kind: "output_usd",
                label: "Output USD",
                path: "assets/fixture_valve/output.usdc",
                sha256: sha256("9"),
                media_type: "application/octet-stream",
              },
              {
                kind: "content_agents_manifest",
                label: "Content Agents manifest",
                path: "assets/fixture_valve/content_agents_manifest.json",
                sha256: sha256("a"),
                media_type: "application/json",
              },
              {
                kind: "geometry_validation_evidence",
                label: "Geometry validation evidence",
                path: "assets/fixture_valve/geometry_validation_evidence.json",
                sha256: sha256("b"),
                media_type: "application/json",
              },
              {
                kind: "ovrtx_render_manifest",
                label: "OVRTX render manifest",
                path: "assets/fixture_valve/renders/render_manifest.json",
                sha256: sha256("e"),
                media_type: "application/json",
              },
            ],
            render_evidence: [
              {
                view: "front",
                image: "assets/fixture_valve/renders/front.png",
                image_sha256: sha256("c"),
                camera: "assets/fixture_valve/renders/front_camera.json",
                camera_sha256: sha256("d"),
                renderer: "ovrtx",
                render_quality: "final",
                ovrtx_render_mode: "pt",
                ovrtx_num_sensor_updates: 500,
                active_aov: "LdrColor",
                width: 640,
                height: 640,
                fallback: false,
                elapsed_seconds: 12.5,
              },
            ],
            findings: [
              {
                severity: "warning",
                title: "Downstream runtime pending",
                detail: "Runtime cooking remains downstream-owned.",
              },
            ],
          },
        },
      ],
    },
    "/geometry-score.json": {
      run_id: runId,
      created_at: "2026-08-04T11:01:00Z",
      status: "fail",
      cases: [
        {
          workflow,
          asset_id: "fixture_valve",
          status: "fail",
          metrics: { deterministic_validation_count: 9 },
          signals: [
            {
              name: "simready_validation",
              ok: false,
              severity: "blocking",
              detail: "The authoritative SimReady gate failed.",
            },
          ],
        },
      ],
    },
  };
}

function stubFetch(documents: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const document = documents[url] ?? documents[url.split("?", 1)[0]!];
      return {
        ok: document !== undefined,
        status: document === undefined ? 404 : 200,
        headers: new Headers(
          document instanceof Uint8Array ? { "content-type": "image/png" } : {},
        ),
        json: async () => document,
        arrayBuffer: async () => {
          if (document instanceof Uint8Array) {
            return document.buffer.slice(
              document.byteOffset,
              document.byteOffset + document.byteLength,
            );
          }
          throw new Error(`not a binary document: ${url}`);
        },
      };
    }),
  );
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

afterEach(() => {
  vi.unstubAllGlobals();
  clearReferenceVerificationCache();
});

describe("sha256HexSync", () => {
  it("matches WebCrypto and the FIPS 180-4 vectors", async () => {
    // NIST vector: sha256("abc").
    expect(sha256HexSync(new TextEncoder().encode("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(sha256HexSync(new Uint8Array())).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    // Multi-block input (>64 bytes) agrees with WebCrypto.
    const long = new Uint8Array(150).map((_, index) => index % 251);
    expect(sha256HexSync(long)).toBe(await sha256Hex(long));
  });
});

describe("compareRunsNewestFirst", () => {
  it("sorts valid timestamps ahead of missing or invalid timestamps", () => {
    const runs = [
      { run_id: "invalid", created_at: "not-a-date" },
      { run_id: "new", created_at: "2026-07-22T00:00:00Z" },
      { run_id: "missing", created_at: "" },
      { run_id: "old", created_at: "2026-07-21T00:00:00Z" },
    ];

    expect(runs.sort(compareRunsNewestFirst).map((run) => run.run_id)).toEqual([
      "new",
      "old",
      "missing",
      "invalid",
    ]);
  });
});

describe("loadBenchmarkBundle", () => {
  it("adapts authoritative Geometry scores, OVRTX evidence, and handoff artifacts", async () => {
    stubFetch(geometryFixtureDocuments());

    const result = await loadBenchmarkBundle(indexUrl);
    const workflow = result.workflows[0]!;
    const run = result.runs[0]!;
    const asset = run.assets[0]!;

    expect(workflow.id).toBe("cad-to-simready");
    expect(run.suite_id).toBe("geometry-cad-to-simready");
    expect(run.provenance?.dataset.content_sha256).toBe("2".repeat(64));
    expect(run.execution?.renderers.benchmark_evidence?.renderer).toBe("ovrtx");
    expect(run.summary).toMatchObject({
      fail: 1,
      pass: 0,
      median_runtime_seconds: 42.5,
      input_tokens: 1200,
      cached_input_tokens: 300,
      output_tokens: 240,
      total_tokens: 1440,
      cost_estimate_status: "estimated",
    });
    expect(asset.status).toBe("fail");
    expect(asset.prompt).toBe(
      "Build a fixture valve with two protected openings.",
    );
    expect(asset.status_label).toBe("Fail · Certified · Handoff Yes");
    expect(asset.metrics).toMatchObject({
      deterministic_validation_count: 9,
      watertight: true,
      protected_opening_count: 2,
      geometry_outcome: "certified",
      handoff_ready: "yes",
      ovrtx_render_count: 1,
    });
    expect(asset.renders[0]).toMatchObject({
      path:
        "/artifacts/geometry/bundle/assets/fixture_valve/renders/front.png" +
        geometryArtifactVersionQuery,
      renderer: "OVRTX",
      sha256: "c".repeat(64),
      settings: {
        render_quality: "final",
        ovrtx_render_mode: "pt",
        ovrtx_num_sensor_updates: 500,
        active_aov: "LdrColor",
        elapsed_seconds: 12.5,
      },
      camera: {
        path:
          "/artifacts/geometry/bundle/assets/fixture_valve/renders/front_camera.json" +
          geometryArtifactVersionQuery,
        sha256: "d".repeat(64),
      },
    });
    expect(asset.reports.map((report) => report.kind)).toEqual([
      "suite_result",
      "source_usd",
      "output_usd",
      "content_agents_manifest",
      "geometry_validation_evidence",
      "ovrtx_render_manifest",
    ]);
    expect(
      asset.reports.find((report) => report.kind === "output_usd"),
    ).toMatchObject({
      sha256: "9".repeat(64),
      media_type: "application/octet-stream",
    });
    expect(asset.findings).toEqual([
      {
        severity: "blocking",
        title: "Simready Validation",
        detail: "The authoritative SimReady gate failed.",
      },
      {
        severity: "warning",
        title: "Downstream runtime pending",
        detail: "Runtime cooking remains downstream-owned.",
      },
    ]);
  });

  it("loads digest-bound public CAD comparisons as a dedicated workflow", async () => {
    stubFetch(geometryFixtureDocuments("geometry-public-cad"));

    const result = await loadBenchmarkBundle(indexUrl);

    expect(result.workflows).toEqual([
      expect.objectContaining({
        id: "cad-benchmark",
        name: "CAD Agent Benchmark",
      }),
    ]);
    expect(result.runs[0]).toMatchObject({
      workflow_id: "cad-benchmark",
      suite_id: "geometry-public-cad",
    });
  });

  it("does not present legacy renders as authoritative Geometry evidence", async () => {
    const documents = geometryFixtureDocuments() as Record<string, any>;
    documents["/geometry-bundle.json"].assets[0].geometry.render_evidence = [];
    documents[indexUrl].runs[0].geometry_integrity.artifacts = documents[
      indexUrl
    ].runs[0].geometry_integrity.artifacts.filter(
      (artifact: { path: string }) =>
        !artifact.path.endsWith("renders/front_camera.json"),
    );
    stubFetch(documents);

    const result = await loadBenchmarkBundle(indexUrl);
    const asset = result.runs[0]!.assets[0]!;

    expect(asset.renders).toEqual([]);
    expect(asset.preview.image_url).toBeUndefined();
  });

  it("preserves generic render evidence for legacy Geometry bundles", async () => {
    const documents = geometryFixtureDocuments("cad_to_simready") as Record<
      string,
      any
    >;
    delete documents["/geometry-bundle.json"].assets[0].geometry;
    stubFetch(documents);

    const result = await loadBenchmarkBundle(indexUrl);
    const asset = result.runs[0]!.assets[0]!;

    expect(asset.renders).toEqual([
      expect.objectContaining({
        path:
          "/artifacts/geometry/bundle/assets/fixture_valve/renders/front.png" +
          geometryArtifactVersionQuery,
      }),
    ]);
    expect(asset.preview.image_url).toBe(asset.renders[0]!.path);
  });

  it("rejects Geometry bundle identity drift instead of loading a partial contract", async () => {
    const documents = geometryFixtureDocuments() as Record<string, any>;
    documents["/geometry-bundle.json"].run_id = "rewritten-run";
    stubFetch(documents);

    await expect(loadBenchmarkBundle(indexUrl)).rejects.toThrow(
      "Run ID mismatch in benchmark artifacts",
    );
  });

  it("rejects a Geometry run whose index integrity does not cover its bundle", async () => {
    const documents = geometryFixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs[0].geometry_integrity.artifacts = [];
    stubFetch(documents);

    await expect(loadBenchmarkBundle(indexUrl)).rejects.toThrow(
      "Geometry artifact integrity coverage mismatch",
    );
  });

  it("rejects a Geometry index entry without a digest-shaped artifact version", async () => {
    const documents = geometryFixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs[0].artifact_version = "mutable-v1";
    stubFetch(documents);

    await expect(loadBenchmarkBundle(indexUrl)).rejects.toThrow(
      "Invalid Geometry artifact version",
    );
  });

  it("adapts indexed artifacts, evidence URLs, and scoring findings", async () => {
    stubFetch(fixtureDocuments());

    const result = await loadBenchmarkBundle(indexUrl);
    const run = result.runs[0]!;
    const asset = run.assets[0]!;

    expect(result.workflows[0]!.run_ids).toEqual(["material-agentic-run"]);
    expect(run.branch).toBe("main");
    expect(run.summary.fail).toBe(1);
    expect(run.summary.median_runtime_seconds).toBe(12);
    expect(asset.status).toBe("fail");
    expect(asset.renders[0]!.path).toBe(
      "/artifacts/run/bundle/assets/asset_a/final%20image.png",
    );
    expect(asset.renders).toEqual([
      expect.objectContaining({ view: "Final", media_type: "image" }),
      expect.objectContaining({ view: "Turntable", media_type: "video" }),
    ]);
    expect(asset.findings).toEqual([
      {
        severity: "blocking",
        title: "Finished",
        detail: "Workflow exited nonzero.",
      },
    ]);
  });

  it("loads aggregate driver and delegated-vision usage", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/manifest.json"].metadata.execution = {
      usage: {
        input_tokens: 100,
        cached_input_tokens: 30,
        output_tokens: 10,
        total_tokens: 110,
        cost_estimate: {
          actual_billing: false,
          provider_reported_cost: true,
          pricing_schema_version: "provider-reported-cost.v1",
          pricing_model: "claude-opus-5",
          model_source: "run_configuration",
          currency: "USD",
          standard_api_equivalent_usd: 2.5,
          long_context_upper_bound_usd: 2.5,
          cached_input_tokens_reported: true,
          pricing_as_of: "provider-reported",
          pricing_source: "Model result total_cost_usd",
          caveat: "Delegated VLM cost excluded.",
        },
        cost_estimate_status: "provider_reported",
        cost_estimate_scope: "driver_only",
      },
    };
    documents["/bundle.json"].summary = {
      metrics: {
        driver_input_tokens: 100,
        driver_output_tokens: 10,
        driver_total_tokens: 110,
        vision_input_tokens: 20,
        vision_output_tokens: 5,
        vision_total_tokens: 25,
        vision_invocation_count: 2,
        combined_total_tokens: 135,
      },
    };
    documents["/bundle.json"].assets[0].metrics = {
      runtime_seconds: 12,
      driver_cached_input_tokens: 30,
      vision_cached_input_tokens: 4,
    };
    stubFetch(documents);

    const summary = (await loadBenchmarkBundle(indexUrl)).runs[0]!.summary;

    expect(summary).toMatchObject({
      driver_input_tokens: 100,
      driver_cached_input_tokens: 30,
      driver_output_tokens: 10,
      driver_total_tokens: 110,
      vision_input_tokens: 20,
      vision_cached_input_tokens: 4,
      vision_output_tokens: 5,
      vision_total_tokens: 25,
      vision_invocation_count: 2,
      combined_total_tokens: 135,
      cost_estimate_scope: "driver_only",
    });
  });

  it("reads asset metrics from the exported root when the id was sanitized", async () => {
    // The exporter keeps `asset_id` verbatim as the join key but sanitizes the
    // directory, so guessing `assets/${asset_id}` would 404 for any id with a
    // character outside [A-Za-z0-9._-].
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets[0].asset_id = "part/a";
    documents["/bundle.json"].assets[0].export_root = "assets/part_a";
    documents["/score.json"].cases[0].asset_id = "part/a";
    stubFetch(documents);

    const bundle = await loadBenchmarkBundle(indexUrl);
    const metrics = bundle.runs[0]!.assets[0]!.reports.find(
      (report) => report.kind === "metrics",
    );

    expect(metrics!.path).toContain("assets/part_a/metrics.json");
    expect(metrics!.path).not.toContain("assets/part/a/metrics.json");
  });

  it.each(["../../protected", "assets\\protected", "https:protected"])(
    "rejects an unsafe exported asset root: %s",
    async (exportRoot) => {
      const documents = fixtureDocuments() as Record<string, any>;
      documents["/bundle.json"].assets[0].export_root = exportRoot;
      stubFetch(documents);

      await expect(loadBenchmarkBundle(indexUrl)).rejects.toThrow(
        "Unsafe benchmark artifact path",
      );
    },
  );

  it("accepts a historical bundle that declares an aliased workflow", async () => {
    // score_bundle accepts `agentic_workflow` through the material adapter's
    // bundle_workflow_aliases and emits canonical cases, so rejecting it here
    // would make a run the backend scored perfectly well unreadable.
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].workflow = "agentic_workflow";
    stubFetch(documents);

    const bundle = await loadBenchmarkBundle(indexUrl);

    expect(bundle.runs[0]!.assets.length).toBeGreaterThan(0);
  });

  it("discovers canonical CAD-to-SimReady runs", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    const workflow = "cad-to-simready";
    const runId = "cad-to-simready-run";
    documents[indexUrl].runs[0].workflow = workflow;
    documents[indexUrl].runs[0].run_id = runId;
    documents["/manifest.json"].workflow = workflow;
    documents["/manifest.json"].run_id = runId;
    documents["/bundle.json"].workflow = workflow;
    documents["/bundle.json"].run_id = runId;
    documents["/bundle.json"].config = {
      profile: "Prop-Robotics-Neutral",
      profile_version: "1.0.0",
    };
    documents["/score.json"].run_id = runId;
    documents["/score.json"].cases[0].workflow = workflow;
    documents["/score.json"].cases[0].status = "pass";
    documents["/score.json"].cases[0].signals = [
      {
        name: "simready_profile_validation",
        ok: true,
        severity: "blocking",
        value: {
          profile: "Prop-Robotics-Neutral",
          profile_version: "1.0.0",
          status: "PASS",
        },
      },
    ];
    stubFetch(documents);

    const bundle = await loadBenchmarkBundle(indexUrl);

    expect(bundle.workflows).toEqual([
      expect.objectContaining({
        id: workflow,
        run_ids: [runId],
      }),
    ]);
    expect(bundle.runs[0]).toEqual(
      expect.objectContaining({
        run_id: runId,
        workflow_id: workflow,
      }),
    );
    expect(bundle.runs[0]!.simready_profiles).toEqual([
      {
        profile_target: "Prop-Robotics-Neutral@1.0.0",
        passed: 1,
        failed: 0,
        unverified: 0,
        total: 1,
      },
    ]);
    expect(bundle.runs[0]!.assets[0]!.simready_validation).toEqual(
      expect.objectContaining({
        profile_target: "Prop-Robotics-Neutral@1.0.0",
        status: "PASS",
        passed: true,
        evidence_verified: true,
        failed_features: [],
      }),
    );
  });

  it("keeps bundle-only SimReady passes explicitly unverified", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs[0].workflow = "cad-to-simready";
    documents["/manifest.json"].workflow = "cad-to-simready";
    documents["/bundle.json"].workflow = "cad-to-simready";
    documents["/bundle.json"].assets[0].simready_validation = {
      profile: "Prop-Robotics-Neutral",
      profile_version: "1.0.0",
      profile_target: "Prop-Robotics-Neutral@1.0.0",
      status: "PASS",
      passed: true,
      evidence_verified: null,
      failed_features: [],
      failed_requirements: [],
      errors: [],
    };
    documents["/score.json"].cases[0].workflow = "cad-to-simready";
    documents["/score.json"].cases[0].signals = [];
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;
    const validation = run.assets[0]!.simready_validation;

    expect(validation).toEqual(
      expect.objectContaining({
        status: "PASS",
        passed: true,
        evidence_verified: null,
      }),
    );
    expect(run.simready_profiles).toEqual([
      {
        profile_target: "Prop-Robotics-Neutral@1.0.0",
        passed: 0,
        failed: 0,
        unverified: 1,
        total: 1,
      },
    ]);
  });

  it("derives SimReady status and verdict from the same effective source", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs[0].workflow = "cad-to-simready";
    documents["/manifest.json"].workflow = "cad-to-simready";
    documents["/bundle.json"].workflow = "cad-to-simready";
    documents["/bundle.json"].assets[0].simready_validation = {
      status: "PASS",
      passed: true,
    };
    documents["/score.json"].cases[0].workflow = "cad-to-simready";
    documents["/score.json"].cases[0].signals = [
      {
        name: "simready_profile_validation",
        ok: false,
        severity: "blocking",
      },
    ];
    stubFetch(documents);

    const validation = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!
      .simready_validation;

    expect(validation?.status).toBe("PASS");
    expect(validation?.passed).toBe(true);
  });

  it("shows failed SimReady features and requirement IDs", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    const workflow = "cad-to-simready";
    const runId = "cad-to-simready-failed";
    documents[indexUrl].runs[0].workflow = workflow;
    documents[indexUrl].runs[0].run_id = runId;
    documents["/manifest.json"].workflow = workflow;
    documents["/manifest.json"].run_id = runId;
    documents["/bundle.json"].workflow = workflow;
    documents["/bundle.json"].run_id = runId;
    documents["/score.json"].run_id = runId;
    documents["/score.json"].cases[0].workflow = workflow;
    documents["/score.json"].cases[0].signals = [
      {
        name: "simready_profile_validation",
        ok: false,
        severity: "blocking",
        detail:
          "SimReady profile Prop-Robotics-Neutral@1.0.0 failed features: FET000_CORE (NP.005, NP.006)",
        value: {
          profile: "Prop-Robotics-Neutral",
          profile_version: "1.0.0",
          profile_target: "Prop-Robotics-Neutral@1.0.0",
          status: "FAIL",
          passed: false,
          failed_features: [
            {
              feature_id: "FET000_CORE",
              requirements: ["NP.005", "NP.006"],
              messages: ["Missing SimReady metadata."],
            },
          ],
          failed_requirements: ["NP.005", "NP.006"],
          errors: [],
        },
      },
    ];
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;

    expect(run.simready_profiles).toEqual([
      {
        profile_target: "Prop-Robotics-Neutral@1.0.0",
        passed: 0,
        failed: 1,
        unverified: 0,
        total: 1,
      },
    ]);
    expect(run.assets[0]!.simready_validation?.failed_features).toEqual([
      {
        feature_id: "FET000_CORE",
        requirements: ["NP.005", "NP.006"],
        messages: ["Missing SimReady metadata."],
      },
    ]);
    expect(run.assets[0]!.findings[0]!.detail).toContain("FET000_CORE");
  });

  it("keeps metrics on a case the bundle never collected", async () => {
    // A scored case absent from the bundle is synthesised for display. It
    // carries no asset-side metrics of its own, so the scored case must remain
    // the source -- otherwise a case that failed to collect loses the very
    // numbers explaining why.
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets = [];
    documents["/score.json"].cases[0].metrics = {
      runtime_seconds: 42,
      total_tokens: 1234,
    };
    stubFetch(documents);

    const bundle = await loadBenchmarkBundle(indexUrl);
    const asset = bundle.runs[0]!.assets[0]!;

    expect(asset.status_label).toContain("Not Collected");
    expect(asset.metrics.total_tokens).toBeDefined();
  });

  it("keeps segmentation quality when the scored case omits its workflow", async () => {
    // The artifact guard deliberately tolerates a missing workflow field, and
    // adaptAsset then presents the asset as mesh-segmentation because the run
    // is. The evaluation gate has to agree, or the asset renders as a
    // segmentation asset with its quality badges silently absent.
    const documents = fixtureDocuments() as Record<string, any>;
    const assetId =
      "mesh_segmentation_partobjaverse_73fc1424ee9c4dd9bee17702aab2b7b3";
    documents[indexUrl].runs[0].workflow = "mesh_segmentation";
    documents[indexUrl].runs[0].run_id = "mesh-run";
    documents["/manifest.json"].workflow = "mesh_segmentation";
    documents["/manifest.json"].run_id = "mesh-run";
    documents["/bundle.json"].workflow = "mesh_segmentation";
    documents["/bundle.json"].run_id = "mesh-run";
    documents["/bundle.json"].assets = [
      { asset_id: assetId, status: "completed", tags: ["mesh_segmentation"] },
    ];
    documents["/score.json"].run_id = "mesh-run";
    documents["/score.json"].cases = [
      {
        // No `workflow` field at all.
        asset_id: assetId,
        status: "warn",
        tags: ["mesh_segmentation"],
        metrics: { face_accuracy: 0.9, macro_iou: 0.8 },
      },
    ];
    stubFetch(documents);

    const bundle = await loadBenchmarkBundle(indexUrl);
    const asset = bundle.runs[0]!.assets[0]!;

    expect(asset.family).toBe("PartObjaverse");
    expect(asset.mesh_segmentation_evaluation).toBeDefined();
    expect(asset.mesh_segmentation_evaluation!.macro_iou).toBe(0.8);
  });

  it("rejects a bundle that declares a different workflow", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].workflow = "mesh_segmentation";
    stubFetch(documents);

    await expect(loadBenchmarkBundle(indexUrl)).rejects.toThrow(
      /Workflow mismatch/,
    );
  });

  it("rejects a scored case that declares a different workflow", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/score.json"].cases[0].workflow = "mesh_segmentation";
    stubFetch(documents);

    await expect(loadBenchmarkBundle(indexUrl)).rejects.toThrow(
      /Workflow mismatch/,
    );
  });

  it("still loads when a document omits its workflow", async () => {
    // An absent field is missing provenance, not a mismatch. Treating it as
    // one would make otherwise readable runs fail to load entirely.
    const documents = fixtureDocuments() as Record<string, any>;
    delete documents["/bundle.json"].workflow;
    delete documents["/score.json"].cases[0].workflow;
    stubFetch(documents);

    const bundle = await loadBenchmarkBundle(indexUrl);

    expect(bundle.runs[0]!.assets.length).toBeGreaterThan(0);
  });

  it("names an unscored asset by its run workflow, not as a material asset", async () => {
    // A bundle asset with no matching scored case used to fall through to the
    // material presentation -- "Material" family and a raw title-cased id --
    // inside a mesh-segmentation run, because naming keyed on the per-case
    // workflow rather than the run's.
    const documents = fixtureDocuments() as Record<string, any>;
    const assetId =
      "mesh_segmentation_partobjaverse_73fc1424ee9c4dd9bee17702aab2b7b3";
    documents[indexUrl].runs[0].workflow = "mesh_segmentation";
    documents[indexUrl].runs[0].run_id = "mesh-run";
    documents["/manifest.json"].workflow = "mesh_segmentation";
    documents["/manifest.json"].run_id = "mesh-run";
    documents["/bundle.json"].workflow = "mesh_segmentation";
    documents["/bundle.json"].run_id = "mesh-run";
    documents["/bundle.json"].assets = [
      {
        asset_id: assetId,
        status: "failed",
        tags: ["mesh_segmentation", "partobjaverse_tiny"],
        metrics: { runtime_seconds: 90 },
      },
    ];
    documents["/score.json"].run_id = "mesh-run";
    documents["/score.json"].cases = [];
    stubFetch(documents);

    const bundle = await loadBenchmarkBundle(indexUrl);
    const asset = bundle.runs[0]!.assets[0]!;

    expect(asset.family).toBe("PartObjaverse");
    expect(asset.name).not.toContain("Mesh Segmentation Partobjaverse");
  });

  it("adapts mesh segmentation quality separately from contract status", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    const assetId =
      "mesh_segmentation_partobjaverse_73fc1424ee9c4dd9bee17702aab2b7b3";
    documents[indexUrl].runs[0].workflow = "mesh_segmentation";
    documents[indexUrl].runs[0].run_id = "mesh-run";
    documents[indexUrl].runs[0].status = "failed";
    documents["/manifest.json"].workflow = "mesh_segmentation";
    documents["/manifest.json"].run_id = "mesh-run";
    documents["/manifest.json"].status = "failed";
    documents["/bundle.json"].workflow = "mesh_segmentation";
    documents["/bundle.json"].run_id = "mesh-run";
    documents["/bundle.json"].assets = [
      {
        asset_id: assetId,
        status: "failed",
        tags: ["mesh_segmentation", "partobjaverse_tiny", "food", "control"],
        output_usd: "assets/pumpkin/segmented.usdc",
        metrics: { runtime_seconds: 90 },
        renders: { final: "assets/pumpkin/final.png" },
      },
    ];
    documents["/score.json"].run_id = "mesh-run";
    documents["/score.json"].cases = [
      {
        workflow: "mesh_segmentation",
        asset_id: assetId,
        status: "fail",
        tags: ["mesh_segmentation", "partobjaverse_tiny", "food", "control"],
        metrics: {
          face_count: 846,
          ground_truth_segment_count: 6,
          predicted_segment_count: 6,
          face_accuracy: 0.974,
          macro_iou: 0.915,
          area_accuracy: 0.994,
          area_weighted_iou: 0.99,
          matches: [{ ground_truth_id: 0, predicted_id: 4, iou: 0.98 }],
        },
        signals: [
          {
            name: "workflow_finished",
            ok: false,
            severity: "blocking",
          },
          { name: "face_accuracy", ok: true, severity: "warning" },
          { name: "macro_iou", ok: true, severity: "warning" },
        ],
      },
    ];
    stubFetch(documents);

    const result = await loadBenchmarkBundle(indexUrl);
    const run = result.runs[0]!;
    const asset = run.assets[0]!;

    expect(result.workflows[0]!.id).toBe("mesh-segmentation");
    expect(run.run_status).toBe("failed");
    expect(run.mesh_segmentation_evaluation).toMatchObject({
      scoreable_asset_count: 1,
      quality_pass_asset_count: 1,
      final_export_asset_count: 1,
      mean_face_accuracy: 0.974,
      mean_macro_iou: 0.915,
    });
    expect(asset.name).toBe("Food · 73fc1424");
    expect(asset.status).toBe("fail");
    expect(asset.mesh_segmentation_evaluation).toMatchObject({
      status: "pass",
      face_accuracy: 0.974,
      macro_iou: 0.915,
      matches: [{ ground_truth_id: 0, predicted_id: 4, iou: 0.98 }],
    });
    expect(asset.metrics).toMatchObject({
      face_accuracy: "97.4%",
      macro_iou: "0.915",
    });
  });

  it("loads persisted asset reviews and counts reviewed assets", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets[0].review_path =
      "assets/asset_a/review.json";
    documents["/artifacts/run/bundle/assets/asset_a/review.json"] = {
      schema_version: "material-agent-review.v1",
      asset_id: "asset_a",
      reviewer: "reviewer",
      score: 4,
      comment: "Material identity matches the reference.",
      tags: ["verified"],
      updated_at: "2026-07-22T01:00:00Z",
    };
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;
    const asset = run.assets[0]!;

    expect(asset.review).toMatchObject({
      reviewer: "reviewer",
      score: 4,
      comment: "Material identity matches the reference.",
      tags: ["verified"],
    });
    expect(run.summary.reviewed).toBe(1);
    expect(run.summary.needs_review).toBe(0);
  });

  it("rejects unsafe review paths before building report links", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets[0].review_path =
      "../../../../attacker-controlled.html";
    stubFetch(documents);

    await expect(loadBenchmarkBundle(indexUrl)).rejects.toThrow(
      "Unsafe review path for asset_a",
    );
  });

  it("rejects cross-origin URLs supplied by the benchmark index", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs[0].bundle_url = "https://example.com/bundle.json";
    stubFetch(documents);

    await expect(loadBenchmarkBundle(indexUrl)).rejects.toThrow(
      "Unsafe bundle_url",
    );
  });

  it("changes the index fingerprint when an artifact version changes", async () => {
    const documents = fixtureDocuments();
    stubFetch(documents);
    const first = await loadBenchmarkIndexFingerprint(indexUrl);
    documents[indexUrl].runs[0]!.artifact_version = "v2";
    const second = await loadBenchmarkIndexFingerprint(indexUrl);

    expect(second).not.toBe(first);
  });

  it("changes the index fingerprint when the artifact directory changes", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    stubFetch(documents);
    const first = await loadBenchmarkIndexFingerprint(indexUrl);
    documents[indexUrl].runs[0]!.artifact_directory = "relocated-run";
    const second = await loadBenchmarkIndexFingerprint(indexUrl);

    expect(second).not.toBe(first);
  });

  it("does not accept the index fingerprint when a supported run fails to load", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs.push({
      ...documents[indexUrl].runs[0],
      bundle_url: "/missing-bundle.json",
      manifest_url: "/missing-manifest.json",
      run_id: "material-agentic-missing",
    });
    stubFetch(documents);
    vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const result = await loadBenchmarkBundle(indexUrl);

    expect(result.runs.map((run) => run.run_id)).toEqual([
      "material-agentic-run",
    ]);
    expect(result.index_fingerprint).toBe("");
  });

  it("surfaces failed evaluations and rejects unsafe local artifact paths", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs[0].evaluation_url = "/evaluation.json";
    documents["/score.json"].status = "pass";
    documents["/score.json"].cases[0].status = "pass";
    documents["/score.json"].cases[0].signals = [];
    documents["/bundle.json"].assets[0].output_usd =
      "/tmp/material-agentic-run/../secret.usd";
    documents["/evaluation.json"] = {
      schema_version: "content-agent-material-evaluation.v3",
      run_id: "material-agentic-run",
      workflow: "material-agentic",
      method: { prompt_version: "v1", caveat: "test" },
      judges: [],
      aggregate: {
        asset_count: 1,
        evaluated_asset_count: 0,
        mean_scores: {},
        mean_agreement_scores: {},
        acceptable_rate: null,
        high_fidelity_rate: null,
        mean_judge_confidence: null,
        mean_assignment_coverage_score: null,
        authoring_coverage_score: null,
        rejected_assignment_rate: null,
        missing_assignment_rate: null,
        final_output_asset_count: 0,
        final_visible_mesh_count: null,
        final_bound_visible_mesh_count: null,
        final_partially_bound_visible_mesh_count: null,
        final_unbound_visible_mesh_count: null,
        final_mesh_binding_coverage_score: null,
      },
      assets: [
        {
          asset_id: "asset_a",
          status: "error",
          scores: null,
          issues: ["Judge request timed out."],
          rationale: "The material judge did not return a valid response.",
        },
      ],
    };
    stubFetch(documents);

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.status).toBe("pass");
    expect(asset.material_evaluation).toBeUndefined();
    expect(asset.reports.some((report) => report.label === "Output USD")).toBe(
      false,
    );
    expect(asset.findings).toEqual([
      {
        severity: "warning",
        title: "Material Evaluation Error",
        detail: "The material judge did not return a valid response.",
      },
      {
        severity: "minor",
        title: "Judge Finding",
        detail: "Judge request timed out.",
      },
    ]);
  });

  it("preserves run evaluation summaries with partial aggregate scores", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs[0].evaluation_url = "/evaluation.json";
    documents["/evaluation.json"] = {
      schema_version: "content-agent-material-evaluation.v3",
      run_id: "material-agentic-run",
      workflow: "material-agentic",
      method: { prompt_version: "v1", caveat: "test" },
      judges: [{ backend: "test", model: "judge" }],
      aggregate: {
        asset_count: 1,
        evaluated_asset_count: 1,
        mean_scores: {
          overall_assignment_quality_score: 82,
          material_identity_score: 84,
          color_palette_score: 80,
          material_consistency_score: 78,
        },
        mean_agreement_scores: {
          overall_assignment_quality_score: 91,
        },
        acceptable_rate: 1,
        high_fidelity_rate: 0,
        mean_judge_confidence: 0.85,
        mean_assignment_coverage_score: 100,
      },
      assets: [],
    };
    stubFetch(documents);

    const summary = (await loadBenchmarkBundle(indexUrl)).runs[0]!
      .material_evaluation;

    expect(summary).not.toBeNull();
    expect(summary!.mean_scores.overall_assignment_quality_score).toBe(82);
    expect(summary!.mean_scores.surface_finish_score).toBeNull();
    expect(
      summary!.mean_agreement_scores.overall_assignment_quality_score,
    ).toBe(91);
    expect(summary!.acceptable_rate).toBe(1);
    expect(summary!.judges).toEqual([{ backend: "test", model: "judge" }]);
  });

  it("sums per-asset token details when run-level usage is absent", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets[0].metrics = {
      runtime_seconds: 12,
      input_tokens: 20,
      cached_input_tokens: 8,
      output_tokens: 3,
      total_tokens: 23,
    };
    stubFetch(documents);

    const summary = (await loadBenchmarkBundle(indexUrl)).runs[0]!.summary;

    expect(summary.input_tokens).toBe(20);
    expect(summary.cached_input_tokens).toBe(8);
    expect(summary.output_tokens).toBe(3);
    expect(summary.total_tokens).toBe(23);
  });

  it("keeps the driver and delegated vision models distinct", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].summary = {
      common_config: {
        runner: "codex",
        model: "nvidia/nvidia/nemotron-3-ultra",
        vision_backend: "delegated_vision",
        vision_model: "openai/openai/gpt-5.6-sol",
      },
    };
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;

    expect(run.model).toBe("nvidia/nvidia/nemotron-3-ultra");
    expect(run.vision_backend).toBe("delegated_vision");
    expect(run.vision_model).toBe("openai/openai/gpt-5.6-sol");
  });

  it("shows the executed CAD model and resource metrics from the bundle", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].config = {
      runner: "codex",
      model: "gpt-5.6-sol",
      model_source: "stage_execution",
      models_used: ["gpt-5.6-sol"],
    };
    documents["/bundle.json"].assets[0].metrics = {
      runtime_seconds: 45.25,
      agent_runtime_seconds: 32.5,
      model: "gpt-5.6-sol",
      material_model: "gpt-5.6-sol",
      physics_model: "gpt-5.6-sol",
      input_tokens: 3000,
      cached_input_tokens: 2300,
      output_tokens: 300,
      reasoning_output_tokens: 120,
      total_tokens: 3300,
      potential_model_cost_usd: 0.01365,
      potential_model_cost_upper_bound_usd: 0.0228,
    };
    documents["/manifest.json"].metadata.execution = {
      usage: {
        input_tokens: 3000,
        cached_input_tokens: 2300,
        output_tokens: 300,
        reasoning_output_tokens: 120,
        total_tokens: 3300,
        cost_estimate_status: "estimated",
        cost_estimate: {
          actual_billing: false,
          pricing_schema_version: "token-cost-estimate.v1",
          pricing_model: "gpt-5.6-sol",
          model_source: "stage_execution",
          currency: "USD",
          standard_api_equivalent_usd: 0.01365,
          long_context_upper_bound_usd: 0.0228,
          cached_input_tokens_reported: true,
          pricing_as_of: "2026-08-13",
          pricing_source:
            "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
          caveat: "API-equivalent estimate only",
        },
      },
    };
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;
    const metrics = run.assets[0]!.metrics;

    expect(run.runner).toBe("codex");
    expect(run.model).toBe("gpt-5.6-sol");
    expect(run.summary.median_runtime_seconds).toBe(45.25);
    expect(run.summary.reasoning_output_tokens).toBe(120);
    expect(run.summary.cost_estimate?.pricing_model).toBe("gpt-5.6-sol");
    expect(metrics.agent_runtime_seconds).toBe(32.5);
    expect(metrics.model).toBe("gpt-5.6-sol");
    expect(metrics.potential_model_cost_usd).toBe("$0.0137");
    expect(metrics.potential_model_cost_upper_bound_usd).toBe("$0.0228");
  });

  it("keeps total tokens coherent when usage fields need asset fallbacks", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets[0].metrics = {
      input_tokens: 20,
      output_tokens: 3,
    };
    documents["/manifest.json"].metadata.execution = {
      usage: { input_tokens: null, output_tokens: 3, total_tokens: 999 },
    };
    stubFetch(documents);

    const summary = (await loadBenchmarkBundle(indexUrl)).runs[0]!.summary;

    expect(summary.input_tokens).toBe(20);
    expect(summary.output_tokens).toBe(3);
    expect(summary.total_tokens).toBe(23);
  });

  it("does not downgrade unknown high-severity benchmark signals", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/score.json"].cases[0].signals[0].severity = "critical";
    stubFetch(documents);

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.findings[0]!.severity).toBe("blocking");
  });

  it("counts bundle assets absent from a partial score as unscored", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/score.json"].cases = [];
    stubFetch(documents);

    const summary = (await loadBenchmarkBundle(indexUrl)).runs[0]!.summary;

    expect(summary.total).toBe(1);
    expect(summary.pass).toBe(0);
    expect(summary.unscored).toBe(1);
  });

  it("maps local artifacts through the indexed output directory", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents[indexUrl].runs[0].artifact_directory = "custom-output";
    documents["/bundle.json"].assets[0].local_artifacts = {
      operation_trace:
        "/repo/.benchmark/workflows/material-agentic/custom-output/raw/trace.md",
    };
    stubFetch(documents);

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.trace_links).toEqual([
      {
        label: "Operation Trace",
        path: "/artifacts/run/raw/trace.md",
        kind: "workflow",
      },
    ]);
  });

  it("does not invent an undefined marker for malformed index entries", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    delete documents[indexUrl].runs[0].run_id;
    documents["/bundle.json"].assets[0].local_artifacts = {
      operation_trace: "/repo/undefined/raw/trace.md",
    };
    stubFetch(documents);

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.trace_links).toEqual([]);
  });

  it("leaves material source references unflagged by the OVRTX gate", async () => {
    // Material (and mesh-segmentation) references are dataset/source images
    // without provenance records; the OVRTX evidence gate applies only to
    // the physics workflows, so these must not demote to Diagnostic preview.
    stubFetch(fixtureDocuments());

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.references).toHaveLength(1);
    expect(asset.references[0]!.diagnostic).toBeUndefined();
    expect(asset.references[0]!.provenance).toBeUndefined();
  });

  it("propagates validated reference provenance and flags the rest as diagnostic", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    // The OVRTX provenance gate applies to physics workflows only.
    for (const key of [indexUrl, "/manifest.json", "/bundle.json"]) {
      const doc = documents[key];
      if (doc.runs) doc.runs[0].workflow = "physics-fixed";
      else doc.workflow = "physics-fixed";
    }
    documents["/score.json"].cases[0].workflow = "physics-fixed";
    const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    documents["/artifacts/run/bundle/assets/asset_a/reference.png"] =
      imageBytes;
    documents["/bundle.json"].assets[0].references = [
      {
        path: "assets/asset_a/reference.png",
        provenance: {
          renderer: "ovrtx",
          render_metadata: { image_width: 1024 },
          source_usd_sha256: "a".repeat(64),
          image_sha256: await sha256Hex(imageBytes),
        },
      },
      {
        label: "curated",
        path: "assets/asset_a/curated.png",
        provenance: { renderer: "ovrtx" }, // digests missing → not evidence
      },
      {
        label: "malformed",
        path: "assets/asset_a/malformed.png",
        provenance: {
          renderer: "ovrtx",
          render_metadata: { image_width: 1024 },
          source_usd_sha256: "not-a-digest",
          image_sha256: "b".repeat(64),
        },
      },
      {
        label: "local-renderer",
        path: "assets/asset_a/local.png",
        provenance: {
          renderer: "blender", // not the shared OVRTX path → not evidence
          render_metadata: { image_width: 1024 },
          source_usd_sha256: "a".repeat(64),
          image_sha256: "b".repeat(64),
        },
      },
      {
        label: "no-metadata",
        path: "assets/asset_a/no-metadata.png",
        provenance: {
          renderer: "ovrtx", // render_metadata missing → not evidence
          source_usd_sha256: "a".repeat(64),
          image_sha256: "b".repeat(64),
        },
      },
      {
        label: "noncanonical-renderer",
        path: "assets/asset_a/preview.png",
        provenance: {
          renderer: "ovrtx-preview", // not the canonical identity
          render_metadata: { image_width: 1024 },
          source_usd_sha256: "a".repeat(64),
          image_sha256: "b".repeat(64),
        },
      },
      {
        label: "array-metadata",
        path: "assets/asset_a/array.png",
        provenance: {
          renderer: "ovrtx",
          render_metadata: ["not", "a", "record"], // array → not evidence
          source_usd_sha256: "a".repeat(64),
          image_sha256: "b".repeat(64),
        },
      },
      { label: "bare", path: "assets/asset_a/bare.png" },
    ];
    stubFetch(documents);

    const propagatedBundle = await loadBenchmarkBundle(indexUrl);
    await propagatedBundle.verify_references?.();
    const asset = propagatedBundle.runs[0]!.assets[0]!;

    expect(asset.references[0]).toMatchObject({
      diagnostic: false,
      provenance: `reference render: ovrtx · usd sha256 ${"a".repeat(64)} · image sha256 ${await sha256Hex(imageBytes)}`,
    });
    expect(asset.references[1]).toMatchObject({
      diagnostic: true,
      provenance: null,
    });
    for (const index of [2, 3, 4, 5, 6, 7]) {
      expect(asset.references[index]).toMatchObject({
        diagnostic: true,
        provenance: null,
      });
    }
  });

  it("requires the full attested OVRTX identity for remote references", async () => {
    // "remote" is a transport, not an engine: the remote backend permits
    // any compatible renderer, so a remote record without the complete
    // attested identity (endpoint, engine, protocol version, status) must
    // stay diagnostic — and a complete identity must promote.
    const documents = fixtureDocuments() as Record<string, any>;
    for (const key of [indexUrl, "/manifest.json", "/bundle.json"]) {
      const doc = documents[key];
      if (doc.runs) doc.runs[0].workflow = "physics-fixed";
      else doc.workflow = "physics-fixed";
    }
    documents["/score.json"].cases[0].workflow = "physics-fixed";
    const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    documents["/artifacts/run/bundle/assets/asset_a/reference.png"] =
      imageBytes;
    const makeReference = (identity: Record<string, unknown> | null) => [
      {
        label: "reference",
        path: "assets/asset_a/reference.png",
        provenance: {
          renderer: "remote",
          render_metadata: { image_width: 1024 },
          source_usd_sha256: "a".repeat(64),
          image_sha256: sha256HexSync(imageBytes),
          ...(identity
            ? {
                render_response: {
                  data: { results: [{ renderer_identity: identity }] },
                },
              }
            : {}),
        },
      },
    ];

    // Engine-only identity -> not attestation -> stays diagnostic.
    documents["/bundle.json"].assets[0].references = makeReference({
      engine: "ovrtx",
    });
    stubFetch(documents);
    clearReferenceVerificationCache();
    const engineOnly = await loadBenchmarkBundle(indexUrl);
    await engineOnly.verify_references?.();
    expect(engineOnly.runs[0]!.assets[0]!.references[0]!.diagnostic).toBe(true);

    // Complete attested identity -> promoted after byte verification.
    documents["/bundle.json"].assets[0].references = makeReference({
      endpoint: "https://render.internal:8443/v1",
      engine: "ovrtx",
      protocol_version: 1,
      status: "ready",
    });
    stubFetch(documents);
    clearReferenceVerificationCache();
    const attested = await loadBenchmarkBundle(indexUrl);
    await attested.verify_references?.();
    expect(attested.runs[0]!.assets[0]!.references[0]!.diagnostic).toBe(false);
  });

  it("requires the exact-record sidecar before promoting an exported reference", async () => {
    // An exported provenance record is only a projection: the exact OVRTX
    // metadata lives in the bound *.provenance.exact.json sidecar. A
    // missing or altered sidecar means the evidence contract cannot be
    // satisfied, so the reference must stay demoted even when the image
    // bytes themselves verify.
    const documents = fixtureDocuments() as Record<string, any>;
    for (const key of [indexUrl, "/manifest.json", "/bundle.json"]) {
      const doc = documents[key];
      if (doc.runs) doc.runs[0].workflow = "physics-fixed";
      else doc.workflow = "physics-fixed";
    }
    documents["/score.json"].cases[0].workflow = "physics-fixed";
    const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    // The verifier compares the displayed projection against the parsed
    // sidecar, so the exact record must agree on renderer and digests.
    const exactBytes = new TextEncoder().encode(
      JSON.stringify({
        renderer: "ovrtx",
        render_metadata: { image_width: 1024 },
        source_usd_sha256: "a".repeat(64),
        image_sha256: await sha256Hex(imageBytes),
      }),
    );
    documents["/artifacts/run/bundle/assets/asset_a/reference.png"] =
      imageBytes;
    documents["/bundle.json"].assets[0].references = [
      {
        label: "reference",
        path: "assets/asset_a/reference.png",
        provenance: {
          renderer: "ovrtx",
          render_metadata: { image_width: 1024 },
          source_usd_sha256: "a".repeat(64),
          image_sha256: await sha256Hex(imageBytes),
          exact_record_sha256: await sha256Hex(exactBytes),
          exact_record_path: "assets/asset_a/reference.provenance.exact.json",
        },
      },
    ];

    // Sidecar missing → image verifies but the reference stays demoted.
    stubFetch(documents);
    clearReferenceVerificationCache();
    const missing = await loadBenchmarkBundle(indexUrl);
    await missing.verify_references?.();
    expect(missing.runs[0]!.assets[0]!.references[0]!.diagnostic).toBe(true);

    // Sidecar altered → digest mismatch, still demoted.
    documents[
      "/artifacts/run/bundle/assets/asset_a/reference.provenance.exact.json"
    ] = new TextEncoder().encode('{"renderer":"tampered"}');
    stubFetch(documents);
    clearReferenceVerificationCache();
    const tampered = await loadBenchmarkBundle(indexUrl);
    await tampered.verify_references?.();
    expect(tampered.runs[0]!.assets[0]!.references[0]!.diagnostic).toBe(true);

    // Sidecar present and digest-matching → promoted.
    documents[
      "/artifacts/run/bundle/assets/asset_a/reference.provenance.exact.json"
    ] = exactBytes;
    stubFetch(documents);
    clearReferenceVerificationCache();
    const verified = await loadBenchmarkBundle(indexUrl);
    await verified.verify_references?.();
    expect(verified.runs[0]!.assets[0]!.references[0]!.diagnostic).toBe(false);
  });

  it("suppresses the scored-thumbnail preview when no physics reference is validated", async () => {
    // The physics scorer fills thumbnails from bundle references regardless
    // of provenance; an image the evidence grid labels "Diagnostic preview"
    // must not reappear unlabeled as the asset card/hero preview.
    const documents = fixtureDocuments() as Record<string, any>;
    for (const key of [indexUrl, "/manifest.json", "/bundle.json"]) {
      const doc = documents[key];
      if (doc.runs) doc.runs[0].workflow = "physics-fixed";
      else doc.workflow = "physics-fixed";
    }
    documents["/score.json"].cases[0].workflow = "physics-fixed";
    documents["/bundle.json"].assets[0].renders = {};
    documents["/bundle.json"].assets[0].references = [
      { label: "bare", path: "assets/asset_a/reference.png" },
    ];
    documents["/score.json"].cases[0].thumbnails = [
      "data:image/png;base64,QUJD",
    ];
    stubFetch(documents);

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.references[0]!.diagnostic).toBe(true);
    expect(asset.preview.image_url).toBeUndefined();

    // With a validated reference (bytes served and digest-matching) the
    // preview shows that reference's own provenance-bound image — never the
    // scorer's re-encoded thumbnail.
    const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    documents["/artifacts/run/bundle/assets/asset_a/reference.png"] =
      imageBytes;
    documents["/bundle.json"].assets[0].references = [
      {
        label: "reference",
        path: "assets/asset_a/reference.png",
        provenance: {
          renderer: "ovrtx",
          render_metadata: { image_width: 1024 },
          source_usd_sha256: "a".repeat(64),
          image_sha256: await sha256Hex(imageBytes),
        },
      },
    ];
    stubFetch(documents);

    clearReferenceVerificationCache();
    const validatedBundle = await loadBenchmarkBundle(indexUrl);
    const validated = validatedBundle.runs[0]!.assets[0]!;

    // Before verification completes the reference stays demoted (fail
    // closed) so first paint never presents unverified bytes as evidence.
    expect(validated.references[0]!.diagnostic).toBe(true);
    await validatedBundle.verify_references?.();

    expect(validated.references[0]!.diagnostic).toBe(false);
    expect(validated.preview.image_url).toBe(validated.references[0]!.path);
    // The reference is rebound to the verified bytes (blob URL), so a later
    // request cannot fetch different bytes than the digest-checked ones.
    expect(validated.references[0]!.path).toMatch(/^blob:/);

    // Same well-formed record, but the served bytes do not match the digest:
    // the loader must demote the reference and drop the preview.
    documents["/artifacts/run/bundle/assets/asset_a/reference.png"] =
      new Uint8Array([0x00, 0x01]);
    stubFetch(documents);
    clearReferenceVerificationCache();

    const tamperedBundle = await loadBenchmarkBundle(indexUrl);
    await tamperedBundle.verify_references?.();
    const tampered = tamperedBundle.runs[0]!.assets[0]!;

    expect(tampered.references[0]!.diagnostic).toBe(true);
    expect(tampered.references[0]!.provenance).toBeNull();
    expect(tampered.preview.image_url).toBeUndefined();
  });

  it("drops unsafe evidence paths", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets[0].references = [
      { path: "../secret-reference.png" },
    ];
    documents["/bundle.json"].assets[0].renders = {
      final: "../secret-render.png",
    };
    stubFetch(documents);

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.references).toEqual([]);
    expect(asset.renders).toEqual([]);
  });

  it("drops an off-origin score thumbnail used as a preview fallback", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets[0].renders = {};
    documents["/score.json"].cases[0].thumbnails = [
      "https://attacker.example/beacon.png",
    ];
    stubFetch(documents);

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.preview.image_url).toBeUndefined();
  });

  it("preserves generated raster data thumbnails for score-only previews", async () => {
    const documents = fixtureDocuments() as Record<string, any>;
    documents["/bundle.json"].assets[0].renders = {};
    documents["/score.json"].cases[0].thumbnails = [
      "data:image/png;base64,AAAA",
    ];
    stubFetch(documents);

    const asset = (await loadBenchmarkBundle(indexUrl)).runs[0]!.assets[0]!;

    expect(asset.preview.image_url).toBe("data:image/png;base64,AAAA");
  });

  function physicsDocuments(workflow: "physics-fixed" | "physics-agentic") {
    const documents = fixtureDocuments() as Record<string, any>;
    const runId = `${workflow}-run`;
    documents[indexUrl].runs[0].workflow = workflow;
    documents[indexUrl].runs[0].run_id = runId;
    documents["/manifest.json"].workflow = workflow;
    documents["/manifest.json"].run_id = runId;
    documents["/bundle.json"].workflow = workflow;
    documents["/bundle.json"].run_id = runId;
    documents["/bundle.json"].assets[0].asset_id = "physx_45516";
    documents["/score.json"].run_id = runId;
    documents["/score.json"].status = "pass";
    documents["/score.json"].cases = [
      {
        workflow,
        asset_id: "physx_45516",
        status: "pass",
        signals: [
          {
            name: "material_family_accuracy",
            ok: true,
            severity: "warning",
            value: 0.75,
            detail: "3/4 parts in the correct material family",
          },
        ],
        metrics: {
          material_accuracy: 0.75,
          gt_parts_total: 4,
          gt_parts_covered: 4,
          density_log_mae: 0.15,
          density_median_error_factor: 1.16,
          friction_mae: 0.05,
          restitution_mae: 0.08,
          predictions_count: 4,
          zero_mass_prims: 0,
          property_rows: [
            {
              prim_path: "/Root/Drive/Base_Body",
              gt_part: "Base_Body",
              predicted_material: "plastic",
              gt_material: "ABS Plastic",
              material_match: true,
              predicted_density: 1100,
              gt_density: 1060,
              density_log_ratio: 0.037,
              predicted_static_friction: 0.4,
              gt_static_friction: 0.45,
              predicted_dynamic_friction: 0.3,
              gt_dynamic_friction: 0.35,
              friction_abs_error: 0.05,
              predicted_restitution: 0.45,
              gt_restitution: 0.5,
              restitution_abs_error: 0.05,
              predicted_mass_kg: 0.01,
            },
          ],
        },
      },
    ];
    return { documents, runId };
  }

  it("adapts physics-fixed property scoring into an asset evaluation", async () => {
    const { documents, runId } = physicsDocuments("physics-fixed");
    stubFetch(documents);

    const result = await loadBenchmarkBundle(indexUrl);
    const run = result.runs[0]!;
    const asset = run.assets[0]!;

    expect(result.workflows[0]!.id).toBe("physics-fixed");
    expect(result.workflows[0]!.run_ids).toEqual([runId]);
    expect(asset.family).toBe("PhysX-Mobility");
    expect(asset.physics_evaluation).toMatchObject({
      material_accuracy: 0.75,
      gt_parts_total: 4,
      gt_parts_covered: 4,
      density_log_mae: 0.15,
      friction_mae: 0.05,
      restitution_mae: 0.08,
    });
    expect(asset.physics_evaluation!.parts).toHaveLength(1);
    expect(asset.physics_evaluation!.parts[0]).toMatchObject({
      prim_path: "/Root/Drive/Base_Body",
      material_match: true,
      predicted_density: 1100,
      gt_density: 1060,
    });
    expect(asset.metrics.material_accuracy).toBe("75.0%");
    expect(asset.metrics.density_log_mae).toBe("0.150");
    expect(run.physics_evaluation).toEqual({
      asset_count: 1,
      scored_asset_count: 1,
      mean_material_accuracy: 0.75,
      mean_density_log_mae: 0.15,
      mean_friction_mae: 0.05,
      mean_restitution_mae: 0.08,
      gt_full_coverage_asset_count: 1,
    });
  });

  it("adapts physics-agentic runs through the same physics evaluation path", async () => {
    const { documents } = physicsDocuments("physics-agentic");
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;

    expect(run.workflow_id).toBe("physics-agentic");
    expect(run.assets[0]!.physics_evaluation?.material_accuracy).toBe(0.75);
    expect(run.physics_evaluation?.scored_asset_count).toBe(1);
  });

  it("loads a historical run whose documents use the aliased 'physics' id", async () => {
    // Exported historical runs carry the aliased workflow id everywhere —
    // index entry, manifest, bundle, and scored cases. The dashboard must
    // resolve them exactly like the adapters' bundle_workflow_aliases, not
    // silently drop the run as an unsupported workflow.
    const { documents } = physicsDocuments("physics-fixed");
    documents[indexUrl].runs[0].workflow = "physics";
    documents["/manifest.json"].workflow = "physics";
    documents["/bundle.json"].workflow = "physics";
    documents["/score.json"].cases[0].workflow = "physics";
    stubFetch(documents);

    const bundle = await loadBenchmarkBundle(indexUrl);
    const run = bundle.runs[0]!;

    expect(run.workflow_id).toBe("physics-fixed");
    // The physics gates key on the canonical id, so evaluation still runs.
    expect(run.physics_evaluation?.scored_asset_count).toBe(1);
  });

  it("labels partial per-asset token coverage for physics runs", async () => {
    const { documents } = physicsDocuments("physics-fixed");
    // Two bundle assets; only one recorded token metrics. Physics summary
    // metrics are themselves per-asset sums, so the coverage note applies
    // even though run-level values exist.
    documents["/bundle.json"].assets[0].metrics = {
      input_tokens: 800,
      output_tokens: 200,
      total_tokens: 1000,
    };
    documents["/bundle.json"].assets.push({
      ...documents["/bundle.json"].assets[0],
      asset_id: "physx_no_tokens",
      metrics: {},
    });
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;

    expect(run.summary.token_asset_coverage).toBe("1/2");
    expect(run.summary.input_tokens).toBe(800);
  });

  it("omits the physics evaluation for a predictions-only case without ground truth", async () => {
    const { documents } = physicsDocuments("physics-fixed");
    // The scorer's no-ground-truth branch records only predictions_count;
    // such a case must not surface an all-n/a panel or count as scored.
    documents["/score.json"].cases[0].signals = [];
    documents["/score.json"].cases[0].metrics = { predictions_count: 4 };
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;

    expect(run.assets[0]!.physics_evaluation).toBeUndefined();
    expect(run.physics_evaluation?.scored_asset_count).toBe(0);
  });

  it("omits the physics evaluation when a case was never property-scored", async () => {
    const { documents } = physicsDocuments("physics-fixed");
    documents["/score.json"].cases[0].status = "error";
    documents["/score.json"].cases[0].signals = [];
    documents["/score.json"].cases[0].metrics = {};
    stubFetch(documents);

    const run = (await loadBenchmarkBundle(indexUrl)).runs[0]!;

    expect(run.assets[0]!.physics_evaluation).toBeUndefined();
    expect(run.physics_evaluation).toEqual({
      asset_count: 1,
      scored_asset_count: 0,
      mean_material_accuracy: null,
      mean_density_log_mae: null,
      mean_friction_mae: null,
      mean_restitution_mae: null,
      gt_full_coverage_asset_count: 0,
    });
  });
});
