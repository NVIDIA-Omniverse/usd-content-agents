// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { createHash } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import type { Dirent } from "node:fs";
import { lstat, open, readdir, realpath, stat } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import {
  defineConfig,
  type Plugin,
  type PreviewServer,
  type ViteDevServer,
} from "vite";
import react from "@vitejs/plugin-react";
import {
  GEOMETRY_ARTIFACT_INTEGRITY_SCHEMA,
  geometryRunArtifactBindings,
  isGeometryIntegrityWorkflow,
  validateGeometryRunContract,
  type GeometryArtifactIntegrity,
  type GeometryRunContract,
} from "./src/data/geometryBenchmark";

const BENCHMARK_URL_PREFIX = "/benchmark-data";
const GEOMETRY_ARTIFACT_VERSION_PARAM = "artifact_version";
const GEOMETRY_ARTIFACT_VERSION_PATTERN = /^sha256:[0-9a-f]{64}$/;
const MAX_BENCHMARK_JSON_BYTES = 64 * 1024 * 1024;
const MAX_GEOMETRY_ARTIFACT_BYTES = 256 * 1024 * 1024;
const MAX_GEOMETRY_RUN_SNAPSHOT_BYTES = 512 * 1024 * 1024;
const MAX_CONCURRENT_GEOMETRY_VERIFICATIONS = 1;
const MAX_PENDING_GEOMETRY_VERIFICATIONS = 16;
const DEFAULT_BENCHMARK_ROOT = fileURLToPath(
  new URL("../../../.benchmark/workflows", import.meta.url),
);

const CONTENT_TYPES: Record<string, string> = {
  ".gif": "image/gif",
  ".html": "text/html; charset=utf-8",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".mp4": "video/mp4",
  ".m4v": "video/x-m4v",
  ".mov": "video/quicktime",
  ".png": "image/png",
  ".txt": "text/plain; charset=utf-8",
  ".usd": "application/octet-stream",
  ".usda": "text/plain; charset=utf-8",
  ".usdc": "application/octet-stream",
  ".webm": "video/webm",
};

interface BenchmarkManifest {
  created_at?: string;
  run_id?: string;
  schema_version?: string;
  status?: string;
  workflow?: string;
}

interface BenchmarkIndexRun {
  artifact_base_url: string;
  artifact_directory: string;
  artifact_version: string;
  bundle_url: string;
  created_at: string;
  evaluation_url: string | null;
  manifest_url: string;
  run_id: string;
  score_url: string | null;
  status: string;
  workflow: string;
  geometry_integrity?: GeometryArtifactIntegrity;
}

async function fileExists(path: string) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

function pathIsWithin(root: string, candidate: string) {
  return candidate === root || candidate.startsWith(`${root}${sep}`);
}

function reportGeometryIndexFailure(
  workflow: string,
  runId: string,
  error: unknown,
) {
  const detail = error instanceof Error ? error.message : "unknown validation error";
  console.warn(
    `Content Agents Dashboard omitted Geometry run ${workflow}/${runId}: ${detail}`,
  );
}

async function readExactBoundedBytes(
  handle: Awaited<ReturnType<typeof open>>,
  initialStat: Awaited<ReturnType<typeof handle.stat>>,
  maximumBytes: number,
  label: string,
) {
  if (
    typeof initialStat.size !== "number" ||
    !Number.isSafeInteger(initialStat.size) ||
    initialStat.size < 0 ||
    initialStat.size > maximumBytes
  ) {
    throw new Error(`${label} exceeds the snapshot limit`);
  }

  const bytes = Buffer.alloc(initialStat.size);
  let offset = 0;
  while (offset < bytes.byteLength) {
    const result = await handle.read(
      bytes,
      offset,
      bytes.byteLength - offset,
      offset,
    );
    if (result.bytesRead === 0) {
      throw new Error(`${label} changed during snapshot`);
    }
    offset += result.bytesRead;
  }

  const trailingByte = Buffer.allocUnsafe(1);
  const trailingRead = await handle.read(trailingByte, 0, 1, bytes.byteLength);
  const finalStat = await handle.stat();
  if (
    trailingRead.bytesRead !== 0 ||
    finalStat.dev !== initialStat.dev ||
    finalStat.ino !== initialStat.ino ||
    finalStat.size !== initialStat.size ||
    finalStat.mtimeMs !== initialStat.mtimeMs ||
    finalStat.ctimeMs !== initialStat.ctimeMs
  ) {
    throw new Error(`${label} changed during snapshot`);
  }
  return bytes;
}

async function readBoundedRegularFile(
  path: string,
  maximumBytes: number,
  label: string,
) {
  const handle = await open(
    path,
    constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
  );
  try {
    const fileStat = await handle.stat();
    if (!fileStat.isFile()) throw new Error(`${label} is not a regular file`);
    return await readExactBoundedBytes(handle, fileStat, maximumBytes, label);
  } finally {
    await handle.close();
  }
}

async function readBoundedJson<T>(path: string, label: string): Promise<T> {
  const bytes = await readBoundedRegularFile(
    path,
    MAX_BENCHMARK_JSON_BYTES,
    label,
  );
  try {
    return JSON.parse(bytes.toString("utf8")) as T;
  } catch {
    throw new Error(`${label} is not valid JSON`);
  }
}

interface GeometrySnapshotBudget {
  bytes: number;
}

interface VerifiedArtifactSnapshot {
  absolutePath: string;
  bytes: Buffer;
  runPath: string;
  sha256: string;
}

interface VerifiedGeometryRun {
  bundle: GeometryRunContract["bundle"];
  evaluation: unknown | null;
  integrity: GeometryArtifactIntegrity;
  manifest: GeometryRunContract["manifest"] & BenchmarkManifest;
  realRunRoot: string;
  score: GeometryRunContract["score"];
  snapshots: Map<string, VerifiedArtifactSnapshot>;
}

let activeGeometryVerifications = 0;
const pendingGeometryVerifications: Array<() => void> = [];

async function withGeometryVerificationSlot<T>(operation: () => Promise<T>) {
  if (activeGeometryVerifications >= MAX_CONCURRENT_GEOMETRY_VERIFICATIONS) {
    if (
      pendingGeometryVerifications.length >= MAX_PENDING_GEOMETRY_VERIFICATIONS
    ) {
      throw new Error("Geometry verification capacity exceeded");
    }
    await new Promise<void>((resolveSlot) => {
      pendingGeometryVerifications.push(resolveSlot);
    });
  } else {
    activeGeometryVerifications += 1;
  }
  try {
    return await operation();
  } finally {
    const next = pendingGeometryVerifications.shift();
    if (next) next();
    else activeGeometryVerifications -= 1;
  }
}

function geometryArtifactVersion(integrity: GeometryArtifactIntegrity) {
  const canonical = [...integrity.artifacts]
    .sort((left, right) => left.path.localeCompare(right.path))
    .map(({ path, sha256 }) => `${path}\0${sha256}`)
    .join("\n");
  return `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
}

function requireCanonicalRunPath(path: string) {
  if (
    !path ||
    path.includes("\\") ||
    path.startsWith("/") ||
    path
      .split("/")
      .some((segment) => !segment || segment === "." || segment === "..")
  ) {
    throw new Error(`Noncanonical Geometry run path: ${path || "<empty>"}`);
  }
}

async function verifiedGeometryRunRoot(
  benchmarkRoot: string,
  directoryWorkflow: string,
  runDirectory: string,
) {
  requireCanonicalRunPath(directoryWorkflow);
  requireCanonicalRunPath(runDirectory);
  if (directoryWorkflow.includes("/") || runDirectory.includes("/")) {
    throw new Error(`Geometry workflow and run must be single path segments`);
  }

  const realBenchmarkRoot = await realpath(benchmarkRoot);
  const workflowRoot = resolve(realBenchmarkRoot, directoryWorkflow);
  const runRoot = resolve(workflowRoot, runDirectory);
  for (const [label, path] of [
    ["workflow", workflowRoot],
    ["run", runRoot],
  ] as const) {
    const pathStat = await lstat(path);
    if (pathStat.isSymbolicLink() || !pathStat.isDirectory()) {
      throw new Error(`Geometry ${label} root is not a real directory: ${path}`);
    }
    if ((await realpath(path)) !== path) {
      throw new Error(`Geometry ${label} root is not canonical: ${path}`);
    }
  }
  if (!pathIsWithin(realBenchmarkRoot, runRoot)) {
    throw new Error(`Geometry run escapes the benchmark root: ${runRoot}`);
  }
  return runRoot;
}

async function snapshotGeometryArtifact(
  realRunRoot: string,
  runPath: string,
  budget: GeometrySnapshotBudget,
): Promise<VerifiedArtifactSnapshot> {
  requireCanonicalRunPath(runPath);
  const segments = runPath.split("/");
  let candidate = realRunRoot;
  let lexicalFileStat: Awaited<ReturnType<typeof lstat>> | undefined;
  for (const [index, segment] of segments.entries()) {
    candidate = resolve(candidate, segment);
    if (!pathIsWithin(realRunRoot, candidate)) {
      throw new Error(`Geometry artifact escapes its real run root: ${runPath}`);
    }
    const candidateStat = await lstat(candidate);
    if (candidateStat.isSymbolicLink()) {
      throw new Error(`Geometry artifact path contains a symlink: ${runPath}`);
    }
    const finalSegment = index === segments.length - 1;
    if (!finalSegment && !candidateStat.isDirectory()) {
      throw new Error(`Geometry artifact parent is not a directory: ${runPath}`);
    }
    if (finalSegment && !candidateStat.isFile()) {
      throw new Error(`Geometry artifact is not a regular file: ${runPath}`);
    }
    if (finalSegment) lexicalFileStat = candidateStat;
  }
  if ((await realpath(candidate)) !== candidate) {
    throw new Error(`Geometry artifact target is not canonical: ${runPath}`);
  }

  const handle = await open(
    candidate,
    constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
  );
  try {
    await assertGeometryDescriptorPath(handle.fd, realRunRoot, candidate, runPath);
    const fileStat = await handle.stat();
    if (!fileStat.isFile()) {
      throw new Error(`Geometry artifact is not a regular file: ${runPath}`);
    }
    if (
      !lexicalFileStat ||
      fileStat.dev !== lexicalFileStat.dev ||
      fileStat.ino !== lexicalFileStat.ino
    ) {
      throw new Error(`Geometry artifact target changed during verification: ${runPath}`);
    }
    if (fileStat.size > MAX_GEOMETRY_ARTIFACT_BYTES) {
      throw new Error(`Geometry artifact exceeds the snapshot limit: ${runPath}`);
    }
    if (budget.bytes + fileStat.size > MAX_GEOMETRY_RUN_SNAPSHOT_BYTES) {
      throw new Error(`Geometry run exceeds the snapshot limit`);
    }
    const bytes = await readExactBoundedBytes(
      handle,
      fileStat,
      MAX_GEOMETRY_ARTIFACT_BYTES,
      `Geometry artifact ${runPath}`,
    );
    await assertGeometryDescriptorPath(handle.fd, realRunRoot, candidate, runPath);
    if (budget.bytes + bytes.byteLength > MAX_GEOMETRY_RUN_SNAPSHOT_BYTES) {
      throw new Error(`Geometry run exceeds the snapshot limit`);
    }
    budget.bytes += bytes.byteLength;
    return {
      absolutePath: candidate,
      bytes,
      runPath,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    };
  } finally {
    await handle.close();
  }
}

async function assertGeometryDescriptorPath(
  descriptor: number,
  realRunRoot: string,
  expectedPath: string,
  runPath: string,
) {
  if (process.platform !== "linux") {
    throw new Error(
      "Geometry descriptor-backed artifact confinement requires Linux procfs",
    );
  }
  let descriptorPath: string;
  try {
    descriptorPath = await realpath(`/proc/self/fd/${descriptor}`);
  } catch {
    throw new Error(
      "Geometry descriptor-backed artifact confinement is unavailable",
    );
  }
  if (
    descriptorPath !== expectedPath ||
    !pathIsWithin(realRunRoot, descriptorPath)
  ) {
    throw new Error(
      `Geometry artifact descriptor escapes its verified run root: ${runPath}`,
    );
  }
}

async function optionalGeometryArtifact(
  realRunRoot: string,
  runPath: string,
  budget: GeometrySnapshotBudget,
) {
  try {
    return await snapshotGeometryArtifact(realRunRoot, runPath, budget);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT" || code === "ENOTDIR") return null;
    throw error;
  }
}

function parseGeometryJson<T>(snapshot: VerifiedArtifactSnapshot): T {
  try {
    return JSON.parse(snapshot.bytes.toString("utf8")) as T;
  } catch {
    throw new Error(`Geometry artifact is not valid JSON: ${snapshot.runPath}`);
  }
}

async function verifyGeometryRunWithoutSlot(
  benchmarkRoot: string,
  directoryWorkflow: string,
  runDirectory: string,
): Promise<VerifiedGeometryRun> {
  const realRunRoot = await verifiedGeometryRunRoot(
    benchmarkRoot,
    directoryWorkflow,
    runDirectory,
  );
  const budget = { bytes: 0 };
  const snapshots = new Map<string, VerifiedArtifactSnapshot>();
  const capture = async (runPath: string) => {
    const existing = snapshots.get(runPath);
    if (existing) return existing;
    const snapshot = await snapshotGeometryArtifact(realRunRoot, runPath, budget);
    snapshots.set(runPath, snapshot);
    return snapshot;
  };

  const manifest = parseGeometryJson<
    GeometryRunContract["manifest"] & BenchmarkManifest
  >(await capture("benchmark_run.json"));
  const bundle = parseGeometryJson<GeometryRunContract["bundle"]>(
    await capture("bundle/run.json"),
  );
  const scoreSnapshot = await optionalGeometryArtifact(
    realRunRoot,
    "score/suite_result.json",
    budget,
  );
  if (scoreSnapshot) snapshots.set(scoreSnapshot.runPath, scoreSnapshot);
  const evaluationSnapshot = await optionalGeometryArtifact(
    realRunRoot,
    "evaluation/material_evaluation.json",
    budget,
  );
  if (evaluationSnapshot) {
    snapshots.set(evaluationSnapshot.runPath, evaluationSnapshot);
  }
  const score = scoreSnapshot
    ? parseGeometryJson<GeometryRunContract["score"]>(scoreSnapshot)
    : null;
  const evaluation = evaluationSnapshot
    ? parseGeometryJson<unknown>(evaluationSnapshot)
    : null;
  if (scoreSnapshot && score === null) {
    throw new Error("Geometry score document is null");
  }
  if (evaluationSnapshot && evaluation === null) {
    throw new Error("Geometry evaluation document is null");
  }

  if (
    directoryWorkflow !== manifest.workflow ||
    !isGeometryIntegrityWorkflow(directoryWorkflow)
  ) {
    throw new Error(`Geometry workflow directory mismatch: ${directoryWorkflow}`);
  }
  if (runDirectory !== manifest.run_id) {
    throw new Error(`Geometry run directory mismatch: ${runDirectory}`);
  }

  const entry = {
    artifact_directory: runDirectory,
    evaluation_url: evaluationSnapshot
      ? "evaluation/material_evaluation.json"
      : null,
    run_id: manifest.run_id,
    score_url: scoreSnapshot ? "score/suite_result.json" : null,
    workflow: manifest.workflow,
  };
  const contract = { entry, manifest, bundle, score, evaluation };
  validateGeometryRunContract(contract, { requireIntegrity: false });

  const bindings = geometryRunArtifactBindings(contract);
  for (const binding of bindings) {
    const snapshot = await capture(binding.path);
    if (binding.sha256 !== undefined && snapshot.sha256 !== binding.sha256) {
      throw new Error(`Geometry artifact SHA-256 mismatch: ${binding.path}`);
    }
  }

  const integrity: GeometryArtifactIntegrity = {
    schema_version: GEOMETRY_ARTIFACT_INTEGRITY_SCHEMA,
    artifacts: bindings.map(({ path }) => ({
      path,
      sha256: snapshots.get(path)!.sha256,
    })),
  };
  validateGeometryRunContract({
    ...contract,
    entry: { ...entry, geometry_integrity: integrity },
  });
  return {
    bundle,
    evaluation,
    integrity,
    manifest,
    realRunRoot,
    score,
    snapshots,
  };
}

async function verifyGeometryRun(
  benchmarkRoot: string,
  directoryWorkflow: string,
  runDirectory: string,
) {
  return withGeometryVerificationSlot(() =>
    verifyGeometryRunWithoutSlot(benchmarkRoot, directoryWorkflow, runDirectory),
  );
}

interface VersionedBundleAsset {
  review_path?: unknown;
  source_usd?: unknown;
  output_usd?: unknown;
  references?: Array<{ path?: unknown }>;
  renders?: Record<string, unknown>;
  artifact_links?: Array<{ uri?: unknown }>;
  geometry?: {
    artifacts?: Array<{ path?: unknown }>;
    render_evidence?: Array<{ image?: unknown; camera?: unknown }>;
  };
}

async function bundleVersionPaths(bundlePath: string) {
  let bundle: { assets?: VersionedBundleAsset[] };
  try {
    bundle = await readBoundedJson(bundlePath, "Benchmark bundle");
  } catch {
    return [];
  }

  const bundleRoot = await realpath(dirname(bundlePath));
  const assets = Array.isArray(bundle.assets) ? bundle.assets : [];
  const declaredPaths = assets.flatMap((asset) => [
    asset.review_path,
    asset.source_usd,
    asset.output_usd,
    ...(Array.isArray(asset.references) ? asset.references : []).map(
      (reference) => reference.path,
    ),
    ...Object.values(
      asset.renders && typeof asset.renders === "object" ? asset.renders : {},
    ),
    ...(Array.isArray(asset.artifact_links) ? asset.artifact_links : []).map(
      (artifact) => artifact.uri,
    ),
    ...(Array.isArray(asset.geometry?.artifacts)
      ? asset.geometry.artifacts
      : []
    ).map((artifact) => artifact.path),
    ...(Array.isArray(asset.geometry?.render_evidence)
      ? asset.geometry.render_evidence
      : []
    ).flatMap((render) => [render.image, render.camera]),
  ]);
  const paths = await Promise.all(
    declaredPaths.map(async (declaredPath) => {
      if (typeof declaredPath !== "string" || !declaredPath) return null;
      const candidate = resolve(bundleRoot, declaredPath);
      if (!pathIsWithin(bundleRoot, candidate)) return null;
      try {
        const resolvedReview = await realpath(candidate);
        return pathIsWithin(bundleRoot, resolvedReview) ? resolvedReview : null;
      } catch {
        return null;
      }
    }),
  );
  return paths.filter((path): path is string => path !== null);
}

export async function createBenchmarkIndex(benchmarkRoot: string) {
  const runs: BenchmarkIndexRun[] = [];
  let workflowEntries: Dirent[];
  try {
    workflowEntries = await readdir(benchmarkRoot, { withFileTypes: true });
  } catch {
    workflowEntries = [];
  }

  for (const workflowEntry of workflowEntries) {
    if (!workflowEntry.isDirectory()) continue;
    const workflow = workflowEntry.name;
    const workflowRoot = resolve(benchmarkRoot, workflow);
    let runEntries;
    try {
      runEntries = await readdir(workflowRoot, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const runEntry of runEntries) {
      if (!runEntry.isDirectory()) continue;
      const runRoot = resolve(workflowRoot, runEntry.name);
      const manifestPath = resolve(runRoot, "benchmark_run.json");
      const bundlePath = resolve(runRoot, "bundle", "run.json");
      let manifest: BenchmarkManifest;
      let verifiedGeometry: VerifiedGeometryRun | null = null;
      if (isGeometryIntegrityWorkflow(workflow)) {
        try {
          verifiedGeometry = await verifyGeometryRun(
            benchmarkRoot,
            workflow,
            runEntry.name,
          );
        } catch (error) {
          reportGeometryIndexFailure(workflow, runEntry.name, error);
          continue;
        }
        manifest = verifiedGeometry.manifest;
      } else {
        if (!(await fileExists(bundlePath))) continue;
        try {
          manifest = await readBoundedJson(manifestPath, "Benchmark manifest");
        } catch {
          continue;
        }
        if (isGeometryIntegrityWorkflow(manifest.workflow ?? "")) {
          try {
            verifiedGeometry = await verifyGeometryRun(
              benchmarkRoot,
              workflow,
              runEntry.name,
            );
          } catch (error) {
            reportGeometryIndexFailure(workflow, runEntry.name, error);
            continue;
          }
          manifest = verifiedGeometry.manifest;
        }
      }
      if (!manifest.run_id || !manifest.workflow) continue;
      const encodedWorkflow = encodeURIComponent(workflow);
      const encodedRunId = encodeURIComponent(runEntry.name);
      const artifactBaseUrl = `${BENCHMARK_URL_PREFIX}/${encodedWorkflow}/${encodedRunId}`;
      const scorePath = resolve(runRoot, "score", "suite_result.json");
      const evaluationPath = resolve(
        runRoot,
        "evaluation",
        "material_evaluation.json",
      );
      let artifactVersion: string;
      if (verifiedGeometry) {
        artifactVersion = geometryArtifactVersion(verifiedGeometry.integrity);
      } else {
        let versionedBundlePaths: string[];
        try {
          versionedBundlePaths = await bundleVersionPaths(bundlePath);
        } catch {
          continue;
        }
        const versionPaths = [
          ...new Set([
            manifestPath,
            bundlePath,
            scorePath,
            evaluationPath,
            ...versionedBundlePaths,
          ]),
        ];
        const versions = await Promise.all(
          versionPaths.map(async (path) => {
            try {
              const fileStat = await stat(path);
              return `${fileStat.mtimeMs}:${fileStat.size}`;
            } catch {
              return "missing";
            }
          }),
        );
        artifactVersion = versions.join("|");
      }
      runs.push({
        artifact_base_url: artifactBaseUrl,
        artifact_directory: runEntry.name,
        artifact_version: artifactVersion,
        bundle_url: `${artifactBaseUrl}/bundle/run.json`,
        created_at: manifest.created_at ?? "",
        evaluation_url: (verifiedGeometry
          ? verifiedGeometry.evaluation !== null
          : await fileExists(evaluationPath))
          ? `${artifactBaseUrl}/evaluation/material_evaluation.json`
          : null,
        manifest_url: `${artifactBaseUrl}/benchmark_run.json`,
        run_id: manifest.run_id,
        score_url: (verifiedGeometry
          ? verifiedGeometry.score !== null
          : await fileExists(scorePath))
          ? `${artifactBaseUrl}/score/suite_result.json`
          : null,
        status: manifest.status ?? "unknown",
        workflow: manifest.workflow,
        ...(verifiedGeometry
          ? { geometry_integrity: verifiedGeometry.integrity }
          : {}),
      });
    }
  }

  runs.sort((left, right) => right.created_at.localeCompare(left.created_at));
  return {
    schema_version: "content-agent-benchmark-index.v1",
    generated_at: new Date().toISOString(),
    runs,
  };
}

async function verifyGeometryRequest(
  benchmarkRoot: string,
  pathSegments: string[],
) {
  const [directoryWorkflow, runDirectory] = pathSegments;
  if (!directoryWorkflow || !runDirectory) return;

  if (isGeometryIntegrityWorkflow(directoryWorkflow)) {
    return verifyGeometryRun(benchmarkRoot, directoryWorkflow, runDirectory);
  }

  const runRoot = resolve(benchmarkRoot, directoryWorkflow, runDirectory);
  const manifestPath = resolve(runRoot, "benchmark_run.json");
  let manifest: BenchmarkManifest;
  try {
    manifest = await readBoundedJson(manifestPath, "Benchmark manifest");
  } catch {
    return;
  }
  if (!isGeometryIntegrityWorkflow(manifest.workflow ?? "")) {
    return;
  }
  return verifyGeometryRun(benchmarkRoot, directoryWorkflow, runDirectory);
}

function decodeCanonicalArtifactPath(requestPath: string) {
  const encodedPath = requestPath.slice(`${BENCHMARK_URL_PREFIX}/`.length);
  const relativePath = decodeURIComponent(encodedPath);
  const pathSegments = relativePath.split("/");
  if (
    pathSegments.length < 3 ||
    pathSegments.some(
      (segment) =>
        !segment ||
        segment === "." ||
        segment === ".." ||
        segment.includes("\\") ||
        /[\0-\x1f\x7f]/.test(segment),
    ) ||
    pathSegments.map(encodeURIComponent).join("/") !== encodedPath
  ) {
    throw new Error(`Noncanonical benchmark artifact URL`);
  }
  return { pathSegments, relativePath };
}

export function benchmarkArtifactsPlugin(benchmarkRoot: string): Plugin {
  const resolvedRoot = resolve(benchmarkRoot);
  let indexInFlight: ReturnType<typeof createBenchmarkIndex> | null = null;
  const createBenchmarkIndexSingleFlight = () => {
    if (indexInFlight) return indexInFlight;
    const index = (async () => {
      try {
        return await createBenchmarkIndex(resolvedRoot);
      } finally {
        indexInFlight = null;
      }
    })();
    indexInFlight = index;
    return index;
  };
  const inFlightGeometryVerifications = new Map<
    string,
    Promise<VerifiedGeometryRun | undefined>
  >();
  const verifyGeometryRequestSingleFlight = (pathSegments: string[]) => {
    const [workflow, runDirectory] = pathSegments;
    if (
      !workflow ||
      !runDirectory ||
      !isGeometryIntegrityWorkflow(workflow)
    ) {
      return verifyGeometryRequest(resolvedRoot, pathSegments);
    }
    const key = `${workflow}\0${runDirectory}`;
    const existing = inFlightGeometryVerifications.get(key);
    if (existing) return existing;
    const verification = (async () => {
      try {
        return await verifyGeometryRequest(resolvedRoot, pathSegments);
      } finally {
        inFlightGeometryVerifications.delete(key);
      }
    })();
    inFlightGeometryVerifications.set(key, verification);
    return verification;
  };

  // Serves both hooks, so it must be typed by what it actually uses.
  // `NonNullable<Plugin["configureServer"]>` is the dev-server hook and
  // does not describe `configurePreviewServer`; nothing caught that while
  // this file sat outside the type gate.
  const installMiddleware = (server: ViteDevServer | PreviewServer) => {
    server.middlewares.use(async (request, response, next) => {
      const rawRequestUrl = request.url ?? "";
      const queryOffset = rawRequestUrl.indexOf("?");
      const requestPath =
        queryOffset === -1 ? rawRequestUrl : rawRequestUrl.slice(0, queryOffset);
      const requestSearchParams = new URLSearchParams(
        queryOffset === -1 ? "" : rawRequestUrl.slice(queryOffset + 1),
      );
      if (
        !requestPath ||
        (requestPath !== BENCHMARK_URL_PREFIX &&
          !requestPath.startsWith(`${BENCHMARK_URL_PREFIX}/`))
      ) {
        next();
        return;
      }

      if (requestPath === `${BENCHMARK_URL_PREFIX}/index.json`) {
        try {
          const index = await createBenchmarkIndexSingleFlight();
          response.statusCode = 200;
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          response.setHeader("Cache-Control", "no-store");
          response.end(JSON.stringify(index));
        } catch (error) {
          next(error as Error);
        }
        return;
      }

      let artifactRequest: ReturnType<typeof decodeCanonicalArtifactPath>;
      try {
        artifactRequest = decodeCanonicalArtifactPath(requestPath);
      } catch {
        response.statusCode = 400;
        response.setHeader("Content-Type", "text/plain; charset=utf-8");
        response.setHeader("Cache-Control", "no-store");
        response.end("Invalid benchmark artifact URL");
        return;
      }

      try {
        const { pathSegments, relativePath } = artifactRequest;
        const artifactPath = resolve(resolvedRoot, relativePath);
        if (
          artifactPath !== resolvedRoot &&
          !artifactPath.startsWith(`${resolvedRoot}${sep}`)
        ) {
          response.statusCode = 403;
          response.end("Forbidden");
          return;
        }

        const requestedVersions = requestSearchParams.getAll(
          GEOMETRY_ARTIFACT_VERSION_PARAM,
        );
        if (
          isGeometryIntegrityWorkflow(pathSegments[0] ?? "") &&
          (requestedVersions.length !== 1 ||
            !GEOMETRY_ARTIFACT_VERSION_PATTERN.test(requestedVersions[0] ?? ""))
        ) {
          response.statusCode = 409;
          response.setHeader("Content-Type", "text/plain; charset=utf-8");
          response.setHeader("Cache-Control", "no-store");
          response.end("Geometry artifact version mismatch");
          return;
        }

        let verifiedGeometry: VerifiedGeometryRun | undefined;
        try {
          verifiedGeometry = await verifyGeometryRequestSingleFlight(pathSegments);
        } catch {
          response.statusCode = 409;
          response.setHeader("Content-Type", "text/plain; charset=utf-8");
          response.setHeader("Cache-Control", "no-store");
          response.end("Geometry artifact integrity check failed");
          return;
        }

        if (verifiedGeometry) {
          const currentVersion = geometryArtifactVersion(verifiedGeometry.integrity);
          if (
            requestedVersions.length !== 1 ||
            requestedVersions[0] !== currentVersion
          ) {
            response.statusCode = 409;
            response.setHeader("Content-Type", "text/plain; charset=utf-8");
            response.setHeader("Cache-Control", "no-store");
            response.end("Geometry artifact version mismatch");
            return;
          }
          const runPath = pathSegments.slice(2).join("/");
          const snapshot = verifiedGeometry.snapshots.get(runPath);
          const canonicalTarget = resolve(verifiedGeometry.realRunRoot, runPath);
          if (!snapshot || snapshot.absolutePath !== canonicalTarget) {
            response.statusCode = 403;
            response.setHeader("Content-Type", "text/plain; charset=utf-8");
            response.setHeader("Cache-Control", "no-store");
            response.end("Unbound Geometry artifact path");
            return;
          }

          response.statusCode = 200;
          response.setHeader(
            "Content-Type",
            CONTENT_TYPES[extname(snapshot.absolutePath).toLowerCase()] ??
              "application/octet-stream",
          );
          response.setHeader("Content-Length", snapshot.bytes.byteLength);
          response.setHeader("Cache-Control", "no-store");
          response.end(request.method === "HEAD" ? undefined : snapshot.bytes);
          return;
        }

        const [realRoot, realArtifactPath] = await Promise.all([
          realpath(resolvedRoot),
          realpath(artifactPath),
        ]);
        if (
          realArtifactPath !== realRoot &&
          !realArtifactPath.startsWith(`${realRoot}${sep}`)
        ) {
          response.statusCode = 403;
          response.end("Forbidden");
          return;
        }

        const artifactStat = await stat(realArtifactPath);
        if (!artifactStat.isFile()) {
          response.statusCode = 404;
          response.end("Not found");
          return;
        }
        response.statusCode = 200;
        response.setHeader(
          "Content-Type",
          CONTENT_TYPES[extname(realArtifactPath).toLowerCase()] ??
            "application/octet-stream",
        );
        response.setHeader("Content-Length", artifactStat.size);
        response.setHeader("Cache-Control", "no-store");
        if (request.method === "HEAD") {
          response.end();
          return;
        }
        createReadStream(realArtifactPath).pipe(response);
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === "ENOENT" || code === "ENOTDIR") {
          response.statusCode = 404;
          response.end("Not found");
          return;
        }
        next(error as Error);
      }
    });
  };

  return {
    name: "content-agent-benchmark-artifacts",
    configureServer: installMiddleware,
    configurePreviewServer: installMiddleware,
  };
}

export default defineConfig({
  plugins: [
    react(),
    benchmarkArtifactsPlugin(
      process.env.CONTENT_BENCHMARK_ROOT ?? DEFAULT_BENCHMARK_ROOT,
    ),
  ],
  build: {
    outDir: "dist",
    assetsDir: "_static",
  },
  server: {
    port: 3001,
  },
});
