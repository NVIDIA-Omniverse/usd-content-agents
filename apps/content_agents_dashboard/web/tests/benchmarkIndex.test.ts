// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import {
  mkdtemp,
  mkdir,
  rm,
  symlink,
  truncate,
  unlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  benchmarkArtifactsPlugin,
  createBenchmarkIndex,
} from "../vite.config";

const temporaryRoots: string[] = [];

afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

function sha256(value: string) {
  return createHash("sha256").update(value).digest("hex");
}

async function writeGeometryRun(
  root: string,
  workflow = "geometry-cad-to-simready",
) {
  const runId = "run-1";
  const runRoot = join(root, workflow, runId);
  const assetRoot = join(runRoot, "bundle", "assets", "valve");
  const renderPath = join(assetRoot, "renders", "front.png");
  const cameraPath = join(assetRoot, "renders", "front_camera.json");
  const declaredArtifactPath = join(assetRoot, "content_agents_manifest.json");
  const referencePath = join(assetRoot, "reference.png");
  const reviewPath = join(assetRoot, "review.json");
  const sourcePath = join(assetRoot, "source.usda");
  const tracePath = join(assetRoot, "trace.json");
  const thumbnailPath = join(assetRoot, "thumbnail.png");
  const debugPath = join(assetRoot, "debug.json");
  const renderBytes = "published-render";
  const cameraBytes = '{"camera":"front"}';
  const artifactBytes = '{"asset_id":"valve"}';
  const manifestBytes = JSON.stringify({
    schema_version: "content-agent-benchmark-run.v1",
    run_id: runId,
    workflow,
    created_at: "2026-08-04T00:00:00Z",
  });
  await mkdir(join(assetRoot, "renders"), { recursive: true });
  await Promise.all([
    writeFile(join(runRoot, "benchmark_run.json"), manifestBytes),
    writeFile(renderPath, renderBytes),
    writeFile(cameraPath, cameraBytes),
    writeFile(declaredArtifactPath, artifactBytes),
    writeFile(referencePath, "reference"),
    writeFile(reviewPath, '{"asset_id":"valve"}'),
    writeFile(sourcePath, "#usda 1.0"),
    writeFile(tracePath, '{"trace":true}'),
    writeFile(thumbnailPath, "thumbnail"),
    writeFile(debugPath, '{"debug":true}'),
  ]);
  const bundleBytes = JSON.stringify({
    schema_version: "content-agent-benchmark-bundle.v1",
    run_id: runId,
    workflow,
    assets: [
      {
        asset_id: "valve",
        source_usd: "assets/valve/source.usda",
        review_path: "assets/valve/review.json",
        references: [{ path: "assets/valve/reference.png" }],
        renders: { front: "assets/valve/renders/front.png" },
        local_artifacts: {
          trace: "assets/valve/trace.json",
          run_dir: "assets/valve",
        },
        geometry: {
          schema_version: "content-agent-benchmark-geometry.v1",
          artifacts: [
            {
              kind: "content_agents_manifest",
              label: "Content Agents manifest",
              path: "assets/valve/content_agents_manifest.json",
              sha256: sha256(artifactBytes),
              media_type: "application/json",
            },
          ],
          render_evidence: [
            {
              view: "front",
              image: "assets/valve/renders/front.png",
              image_sha256: sha256(renderBytes),
              camera: "assets/valve/renders/front_camera.json",
              camera_sha256: sha256(cameraBytes),
              renderer: "ovrtx",
              render_quality: "final",
              ovrtx_render_mode: "pt",
              ovrtx_num_sensor_updates: 500,
              active_aov: "LdrColor",
              width: 640,
              height: 640,
            },
          ],
        },
      },
    ],
  });
  await writeFile(
    join(runRoot, "bundle", "run.json"),
    bundleBytes,
  );
  const scoreBytes = JSON.stringify({
    run_id: runId,
    cases: [
      {
        asset_id: "valve",
        workflow,
        artifacts: { debug: "assets/valve/debug.json" },
        thumbnails: ["assets/valve/thumbnail.png"],
      },
    ],
  });
  const evaluationBytes = JSON.stringify({
    schema_version: "content-agent-material-evaluation.v3",
    run_id: runId,
    workflow,
  });
  await mkdir(join(runRoot, "score"), { recursive: true });
  await mkdir(join(runRoot, "evaluation"), { recursive: true });
  await Promise.all([
    writeFile(join(runRoot, "score", "suite_result.json"), scoreBytes),
    writeFile(
      join(runRoot, "evaluation", "material_evaluation.json"),
      evaluationBytes,
    ),
  ]);
  return {
    artifactBytes,
    bundleBytes,
    declaredArtifactPath,
    manifestBytes,
    renderBytes,
    renderPath,
    runId,
    runRoot,
    scoreBytes,
    sourcePath,
    workflow,
  };
}

async function writeLegacyCadToSimReadyRun(root: string, workflow: string) {
  const runId = "legacy-run-1";
  const runRoot = join(root, workflow, runId);
  await mkdir(join(runRoot, "bundle"), { recursive: true });
  await writeFile(
    join(runRoot, "benchmark_run.json"),
    JSON.stringify({
      schema_version: "content-agent-benchmark-run.v1",
      run_id: runId,
      workflow,
      created_at: "2026-08-04T00:00:00Z",
    }),
  );
  await writeFile(
    join(runRoot, "bundle", "run.json"),
    JSON.stringify({
      schema_version: "content-agent-benchmark-bundle.v1",
      run_id: runId,
      workflow,
      assets: [{ asset_id: "legacy-valve", status: "pass" }],
    }),
  );
  return { runId, workflow };
}

async function requestArtifact(
  root: string,
  url: string,
  options: {
    artifactVersion?: string | null;
    middleware?: BenchmarkMiddleware;
    method?: "GET" | "HEAD";
    onSetHeader?: (name: string, value: string | number) => void;
  } = {},
) {
  const middleware = options.middleware ?? (await installArtifactMiddleware(root));

  const headers = new Map<string, string | number>();
  let body: unknown;
  let nextError: Error | undefined;
  const response = {
    statusCode: 0,
    setHeader(name: string, value: string | number) {
      headers.set(name, value);
      options.onSetHeader?.(name, value);
    },
    end(value?: unknown) {
      body = value;
    },
  };
  let requestUrl = url;
  if (!requestUrl.includes("artifact_version=") && options.artifactVersion !== null) {
    const index = await createBenchmarkIndex(root);
    const geometryRun = index.runs.find((run) => run.geometry_integrity !== undefined);
    const artifactVersion = options.artifactVersion ?? geometryRun?.artifact_version;
    if (artifactVersion) {
      const separator = requestUrl.includes("?") ? "&" : "?";
      requestUrl = `${requestUrl}${separator}artifact_version=${encodeURIComponent(
        artifactVersion,
      )}`;
    }
  }
  await middleware(
    { method: options.method ?? "GET", url: requestUrl },
    response,
    (error) => {
      nextError = error;
    },
  );
  return { body, headers, nextError, statusCode: response.statusCode };
}

type BenchmarkMiddleware = (
  request: any,
  response: any,
  next: (error?: Error) => void,
) => Promise<void>;

async function installArtifactMiddleware(root: string) {
  const plugin = benchmarkArtifactsPlugin(root);
  let middleware: BenchmarkMiddleware | undefined;
  const install = plugin.configureServer;
  if (typeof install !== "function") throw new Error("Missing benchmark middleware");
  await install({
    middlewares: {
      use(handler: typeof middleware) {
        middleware = handler;
      },
    },
  } as any);
  if (!middleware) throw new Error("Benchmark middleware was not installed");
  return middleware;
}

describe("createBenchmarkIndex", () => {
  it.each(["cad_to_simready", "cad-to-simready"])(
    "keeps legacy %s runs browsable without a Geometry extension",
    async (workflow) => {
      const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
      temporaryRoots.push(root);
      const fixture = await writeLegacyCadToSimReadyRun(root, workflow);

      const index = await createBenchmarkIndex(root);

      expect(index.runs).toHaveLength(1);
      expect(index.runs[0]).toMatchObject({
        run_id: fixture.runId,
        workflow,
      });
      expect(index.runs[0]?.geometry_integrity).toBeUndefined();
    },
  );

  it("indexes a digest-bound public CAD comparison run", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    await writeGeometryRun(root, "geometry-public-cad");

    const index = await createBenchmarkIndex(root);

    expect(index.runs).toHaveLength(1);
    expect(index.runs[0]).toMatchObject({
      workflow: "geometry-public-cad",
      artifact_version: expect.stringMatching(/^sha256:[0-9a-f]{64}$/),
    });
  });

  it("reports the contract reason when a Geometry run is omitted", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    const bundlePath = join(fixture.runRoot, "bundle", "run.json");
    const bundle = JSON.parse(fixture.bundleBytes);
    delete bundle.assets[0].geometry;
    await writeFile(bundlePath, JSON.stringify(bundle));
    const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    expect((await createBenchmarkIndex(root)).runs).toEqual([]);
    expect(warning).toHaveBeenCalledWith(
      expect.stringContaining(
        "Unsupported Geometry asset schema for valve: missing",
      ),
    );
  });

  it("omits an oversized benchmark manifest without allocating its contents", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const runRoot = join(root, "material-agentic", "run-1");
    await mkdir(join(runRoot, "bundle"), { recursive: true });
    await writeFile(join(runRoot, "bundle", "run.json"), '{"assets":[]}');
    await writeFile(join(runRoot, "benchmark_run.json"), "{}");
    await truncate(join(runRoot, "benchmark_run.json"), 64 * 1024 * 1024 + 1);

    expect((await createBenchmarkIndex(root)).runs).toEqual([]);
  });

  it("omits a Geometry run with an oversized declared artifact", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    await truncate(fixture.renderPath, 256 * 1024 * 1024 + 1);

    expect((await createBenchmarkIndex(root)).runs).toEqual([]);
  });

  it("changes the artifact version when a persisted review changes", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const runRoot = join(root, "material-agentic", "run-1");
    const reviewPath = join(runRoot, "bundle", "assets", "asset", "review.json");
    await mkdir(join(runRoot, "bundle", "assets", "asset"), { recursive: true });
    await writeFile(
      join(runRoot, "benchmark_run.json"),
      JSON.stringify({
        run_id: "run-1",
        workflow: "material-agentic",
        created_at: "2026-07-23T00:00:00Z",
      }),
    );
    await writeFile(
      join(runRoot, "bundle", "run.json"),
      JSON.stringify({
        assets: [{ asset_id: "asset", review_path: "assets/asset/review.json" }],
      }),
    );
    await writeFile(reviewPath, JSON.stringify({ comment: "first" }));

    const first = await createBenchmarkIndex(root);
    await writeFile(reviewPath, JSON.stringify({ comment: "updated review" }));
    const second = await createBenchmarkIndex(root);

    expect(second.runs[0]!.artifact_version).not.toBe(
      first.runs[0]!.artifact_version,
    );
  });

  it("omits and rejects a Geometry run after published image replacement", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);

    const first = await createBenchmarkIndex(root);
    expect(first.runs).toHaveLength(1);
    expect(first.runs[0]!.artifact_version).toMatch(/^sha256:[0-9a-f]{64}$/);
    const integrity = first.runs[0]!.geometry_integrity!.artifacts;
    expect(integrity.map((artifact) => artifact.path)).toEqual([
      "benchmark_run.json",
      "bundle/assets/valve/content_agents_manifest.json",
      "bundle/assets/valve/debug.json",
      "bundle/assets/valve/reference.png",
      "bundle/assets/valve/renders/front_camera.json",
      "bundle/assets/valve/renders/front.png",
      "bundle/assets/valve/review.json",
      "bundle/assets/valve/source.usda",
      "bundle/assets/valve/thumbnail.png",
      "bundle/assets/valve/trace.json",
      "bundle/run.json",
      "evaluation/material_evaluation.json",
      "score/suite_result.json",
    ]);
    expect(
      integrity.find((artifact) => artifact.path === "benchmark_run.json")?.sha256,
    ).toBe(sha256(fixture.manifestBytes));
    expect(
      integrity.find((artifact) => artifact.path === "bundle/run.json")?.sha256,
    ).toBe(sha256(fixture.bundleBytes));
    expect(
      integrity.find((artifact) => artifact.path === "score/suite_result.json")
        ?.sha256,
    ).toBe(sha256(fixture.scoreBytes));

    await writeFile(fixture.renderPath, "replacement-render");
    const response = await requestArtifact(
      root,
      `/benchmark-data/${fixture.workflow}/${fixture.runId}/benchmark_run.json`,
      { artifactVersion: first.runs[0]!.artifact_version },
    );
    const second = await createBenchmarkIndex(root);

    expect(response).toMatchObject({
      body: "Geometry artifact integrity check failed",
      nextError: undefined,
      statusCode: 409,
    });
    expect(second.runs).toEqual([]);
  });

  it.each([
    "score/suite_result.json",
    "evaluation/material_evaluation.json",
  ])("omits a Geometry run whose optional document is literal null: %s", async (path) => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    await writeFile(join(fixture.runRoot, path), "null");

    const index = await createBenchmarkIndex(root);

    expect(index.runs).toEqual([]);
  });

  it("omits a Geometry run with a missing declared artifact", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    await unlink(fixture.declaredArtifactPath);

    expect((await createBenchmarkIndex(root)).runs).toEqual([]);
  });

  it("binds standard consumer files that have no Geometry declaration", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);

    const first = await createBenchmarkIndex(root);
    const firstSource = first.runs[0]!.geometry_integrity!.artifacts.find(
      (artifact) => artifact.path === "bundle/assets/valve/source.usda",
    );
    await writeFile(fixture.sourcePath, "#usda 1.0\ndef Xform Changed {}\n");
    const second = await createBenchmarkIndex(root);
    const secondSource = second.runs[0]!.geometry_integrity!.artifacts.find(
      (artifact) => artifact.path === "bundle/assets/valve/source.usda",
    );

    expect(firstSource?.sha256).toBe(sha256("#usda 1.0"));
    expect(secondSource?.sha256).not.toBe(firstSource?.sha256);
    expect(second.runs[0]!.artifact_version).not.toBe(
      first.runs[0]!.artifact_version,
    );
  });

  it("rejects a stale Geometry index version after authoritative JSON changes", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    const first = await createBenchmarkIndex(root);
    const firstVersion = first.runs[0]!.artifact_version;
    const scorePath = join(fixture.runRoot, "score", "suite_result.json");
    const rewrittenScore = JSON.parse(fixture.scoreBytes);
    rewrittenScore.status = "pass";
    await writeFile(scorePath, JSON.stringify(rewrittenScore));

    const staleResponse = await requestArtifact(
      root,
      `/benchmark-data/${fixture.workflow}/${fixture.runId}/score/suite_result.json`,
      { artifactVersion: firstVersion },
    );
    const second = await createBenchmarkIndex(root);

    expect(second.runs[0]!.artifact_version).not.toBe(firstVersion);
    expect(staleResponse).toMatchObject({
      body: "Geometry artifact version mismatch",
      nextError: undefined,
      statusCode: 409,
    });
  });

  it("requires a Geometry artifact version on every artifact request", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);

    const response = await requestArtifact(
      root,
      `/benchmark-data/${fixture.workflow}/${fixture.runId}/benchmark_run.json`,
      { artifactVersion: null },
    );

    expect(response).toMatchObject({
      body: "Geometry artifact version mismatch",
      nextError: undefined,
      statusCode: 409,
    });
  });

  it("rejects a missing token before scanning a corrupted Geometry run", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    await writeFile(fixture.renderPath, "replacement-render");

    const response = await requestArtifact(
      root,
      `/benchmark-data/${fixture.workflow}/${fixture.runId}/benchmark_run.json`,
      { artifactVersion: null },
    );

    expect(response).toMatchObject({
      body: "Geometry artifact version mismatch",
      nextError: undefined,
      statusCode: 409,
    });
  });

  it("omits a Geometry run whose declared artifact symlink escapes its bundle", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    const outsidePath = join(root, "outside-content-agents-manifest.json");
    await writeFile(outsidePath, fixture.artifactBytes);
    await unlink(fixture.declaredArtifactPath);
    await symlink(outsidePath, fixture.declaredArtifactPath);

    expect((await createBenchmarkIndex(root)).runs).toEqual([]);
  });

  it("rejects symlinks even when they resolve to a regular file inside the run", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    const duplicatePath = join(fixture.runRoot, "bundle", "assets", "manifest-copy.json");
    await writeFile(duplicatePath, fixture.artifactBytes);
    await unlink(fixture.declaredArtifactPath);
    await symlink(duplicatePath, fixture.declaredArtifactPath);

    expect((await createBenchmarkIndex(root)).runs).toEqual([]);
  });

  it("rejects a symlinked authoritative score document", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    const scorePath = join(fixture.runRoot, "score", "suite_result.json");
    const scoreCopyPath = join(fixture.runRoot, "score-copy.json");
    await writeFile(scoreCopyPath, fixture.scoreBytes);
    await unlink(scorePath);
    await symlink(scoreCopyPath, scorePath);

    expect((await createBenchmarkIndex(root)).runs).toEqual([]);
  });

  it("rejects canonical but unbound files in an otherwise valid Geometry run", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    await writeFile(join(fixture.runRoot, "bundle", "unbound.txt"), "private");

    const response = await requestArtifact(
      root,
      `/benchmark-data/${fixture.workflow}/${fixture.runId}/bundle/unbound.txt`,
    );

    expect(response).toMatchObject({
      body: "Unbound Geometry artifact path",
      nextError: undefined,
      statusCode: 403,
    });
  });

  it.each([
    "bundle//run.json",
    "bundle/./run.json",
    "bundle/%2e/run.json",
    "bundle/%2e%2e/benchmark_run.json",
  ])("rejects a noncanonical decoded artifact URL: %s", async (path) => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);

    const response = await requestArtifact(
      root,
      `/benchmark-data/${fixture.workflow}/${fixture.runId}/${path}`,
    );

    expect(response).toMatchObject({
      body: "Invalid benchmark artifact URL",
      nextError: undefined,
      statusCode: 400,
    });
  });

  it("rejects encoded separators instead of accepting an alternate run URL", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);

    const response = await requestArtifact(
      root,
      `/benchmark-data/${fixture.workflow}%2F${fixture.runId}/benchmark_run.json`,
    );

    expect(response.statusCode).toBe(400);
  });

  it("serves the exact verified bytes when the file changes after verification", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    let replaced = false;

    const response = await requestArtifact(
      root,
      `/benchmark-data/${fixture.workflow}/${fixture.runId}/bundle/assets/valve/renders/front.png`,
      {
        onSetHeader(name) {
          if (name === "Content-Length" && !replaced) {
            replaced = true;
            writeFileSync(fixture.renderPath, "replacement-after-verification");
          }
        },
      },
    );

    expect(response.statusCode).toBe(200);
    expect(Buffer.isBuffer(response.body)).toBe(true);
    expect((response.body as Buffer).toString("utf8")).toBe(fixture.renderBytes);
    expect(response.headers.get("Content-Length")).toBe(
      Buffer.byteLength(fixture.renderBytes),
    );
  });

  it("shares one exact snapshot across concurrent requests for a Geometry run", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    const artifactVersion = (await createBenchmarkIndex(root)).runs[0]!
      .artifact_version;
    const middleware = await installArtifactMiddleware(root);
    const url = `/benchmark-data/${fixture.workflow}/${fixture.runId}/bundle/assets/valve/renders/front.png`;
    let replaced = false;

    const [first, second] = await Promise.all([
      requestArtifact(root, url, {
        artifactVersion,
        middleware,
        onSetHeader(name) {
          if (name === "Content-Length" && !replaced) {
            replaced = true;
            writeFileSync(fixture.renderPath, "replacement-after-verification");
          }
        },
      }),
      requestArtifact(root, url, { artifactVersion, middleware }),
    ]);

    for (const response of [first, second]) {
      expect(response.statusCode).toBe(200);
      expect(Buffer.isBuffer(response.body)).toBe(true);
      expect((response.body as Buffer).toString("utf8")).toBe(fixture.renderBytes);
    }
  });

  it("verifies Geometry integrity for HEAD without sending the artifact body", async () => {
    const root = await mkdtemp(join(tmpdir(), "content-benchmark-index-"));
    temporaryRoots.push(root);
    const fixture = await writeGeometryRun(root);
    const url = `/benchmark-data/${fixture.workflow}/${fixture.runId}/bundle/assets/valve/renders/front.png`;
    const artifactVersion = (await createBenchmarkIndex(root)).runs[0]!
      .artifact_version;

    const valid = await requestArtifact(root, url, {
      artifactVersion,
      method: "HEAD",
    });
    expect(valid).toMatchObject({ body: undefined, statusCode: 200 });
    expect(valid.headers.get("Content-Length")).toBe(
      Buffer.byteLength(fixture.renderBytes),
    );

    await writeFile(fixture.renderPath, "replacement-render");
    const invalid = await requestArtifact(root, url, {
      artifactVersion,
      method: "HEAD",
    });
    expect(invalid).toMatchObject({
      body: "Geometry artifact integrity check failed",
      statusCode: 409,
    });
  });
});
