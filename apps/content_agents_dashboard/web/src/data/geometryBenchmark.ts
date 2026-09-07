// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  EvidenceItem,
  Finding,
  ReportLink,
  ReviewSeverity,
} from "../types/benchmark";

export const STANDARD_BENCHMARK_BUNDLE_SCHEMA =
  "content-agent-benchmark-bundle.v1";
export const GEOMETRY_ASSET_SCHEMA = "content-agent-benchmark-geometry.v1";
export const GEOMETRY_ARTIFACT_INTEGRITY_SCHEMA =
  "content-agent-benchmark-geometry-integrity.v1";

const GEOMETRY_WORKFLOW_IDS = new Set([
  "cad_to_simready",
  "cad-to-simready",
  "geometry-cad-to-simready",
  "geometry-public-cad",
]);
const GEOMETRY_INTEGRITY_WORKFLOW_IDS = new Set([
  "geometry-cad-to-simready",
  "geometry-public-cad",
]);
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const JSON_ARTIFACT_KINDS = new Set([
  "content_agents_manifest",
  "geometry_validation_evidence",
  "ovrtx_render_manifest",
]);
const USD_MEDIA_TYPES = new Set(["application/octet-stream", "text/plain"]);

export type BundleMetric = string | number | boolean | null;

export interface BundleArtifactLink {
  kind?: string;
  label?: string;
  uri?: string;
  sha256?: string;
  media_type?: string;
}

export interface GeometryArtifact {
  kind: string;
  label: string;
  path: string;
  sha256: string;
  media_type: string;
}

export interface GeometryRenderEvidence {
  view: string;
  image: string;
  image_sha256: string;
  camera: string;
  camera_sha256: string;
  renderer: string;
  render_quality: string;
  ovrtx_render_mode: string;
  ovrtx_num_sensor_updates: number;
  active_aov: string;
  width: number;
  height: number;
  fallback?: boolean;
  elapsed_seconds?: number;
}

export interface GeometryBundleFinding {
  severity: string;
  title: string;
  detail: string;
}

export interface GeometryAssetExtension {
  schema_version: string;
  benchmark_family?: string;
  outcome?: string;
  handoff_ready?: string;
  artifacts?: GeometryArtifact[];
  render_evidence?: GeometryRenderEvidence[];
  findings?: GeometryBundleFinding[];
}

export interface GeometryCompatibleAsset {
  asset_id: string;
  metrics?: Record<string, BundleMetric>;
  source_usd?: string;
  output_usd?: string;
  references?: Array<{ path?: string }>;
  renders?: Record<string, string>;
  artifact_links?: BundleArtifactLink[];
  local_artifacts?: Record<string, string>;
  review_path?: string;
  geometry?: GeometryAssetExtension;
}

export interface GeometryCompatibleSuiteCase {
  asset_id: string;
  workflow: string;
  metrics?: Record<string, unknown>;
  artifacts?: Record<string, string>;
  thumbnails?: string[];
}

export interface GeometryArtifactDigest {
  path: string;
  sha256: string;
}

export interface GeometryArtifactIntegrity {
  schema_version: typeof GEOMETRY_ARTIFACT_INTEGRITY_SCHEMA;
  artifacts: GeometryArtifactDigest[];
}

export interface GeometryRunContract {
  entry: {
    artifact_directory?: string;
    evaluation_url?: string | null;
    run_id: string;
    score_url?: string | null;
    workflow: string;
    geometry_integrity?: unknown;
  };
  manifest: { schema_version: string; run_id: string; workflow: string };
  bundle: {
    schema_version: string;
    run_id: string;
    workflow: string;
    assets: GeometryCompatibleAsset[];
  };
  score: {
    run_id: string;
    cases: GeometryCompatibleSuiteCase[];
  } | null;
  evaluation?: unknown | null;
}

interface GeometryAssetAdapterInput {
  asset: GeometryCompatibleAsset;
  scoredCase?: GeometryCompatibleSuiteCase;
  resolveArtifact: (path: string) => string | null;
}

export interface GeometryAssetAdapterResult {
  family: string | null;
  workflowStatus: string | null;
  renders: EvidenceItem[];
  reports: ReportLink[];
  metrics: Record<string, string | number | boolean>;
  findings: Finding[];
}

export function isGeometryWorkflow(workflow: string) {
  return GEOMETRY_WORKFLOW_IDS.has(workflow);
}

export function isGeometryIntegrityWorkflow(workflow: string) {
  return GEOMETRY_INTEGRITY_WORKFLOW_IDS.has(workflow);
}

function isPortableArtifactPath(path: unknown): path is string {
  if (typeof path !== "string") return false;
  if (!path || path.includes("\\") || /[\0-\x1f\x7f]/.test(path)) return false;
  if (path.startsWith("/") || /^[a-z][a-z0-9+.-]*:/i.test(path)) return false;
  return !path
    .split("/")
    .some((segment) => !segment || segment === "." || segment === "..");
}

function requirePortablePath(path: unknown, field: string): asserts path is string {
  if (!isPortableArtifactPath(path)) {
    throw new Error(
      `Geometry bundle has unsafe ${field}: ${typeof path === "string" && path ? path : "<empty>"}`,
    );
  }
}

function requireSha256(value: unknown, field: string): asserts value is string {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
    throw new Error(`Geometry bundle has invalid ${field}`);
  }
}

function requireNonempty(value: unknown, field: string): asserts value is string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Geometry bundle has empty ${field}`);
  }
}

function assertUniqueIds(values: string[], label: string) {
  const seen = new Set<string>();
  for (const value of values) {
    requireNonempty(value, label);
    if (seen.has(value)) throw new Error(`Duplicate ${label}: ${value}`);
    seen.add(value);
  }
}

function validateGeometryExtension(asset: GeometryCompatibleAsset) {
  const geometry = asset.geometry;
  if (!geometry || geometry.schema_version !== GEOMETRY_ASSET_SCHEMA) {
    throw new Error(
      `Unsupported Geometry asset schema for ${asset.asset_id}: ${geometry?.schema_version ?? "missing"}`,
    );
  }
  if (geometry.benchmark_family !== undefined) {
    requireNonempty(geometry.benchmark_family, "geometry.benchmark_family");
  }
  if (geometry.outcome !== undefined) {
    requireNonempty(geometry.outcome, "geometry.outcome");
  }
  if (geometry.handoff_ready !== undefined) {
    requireNonempty(geometry.handoff_ready, "geometry.handoff_ready");
  }

  const artifactIdentities: string[] = [];
  if (geometry.artifacts !== undefined && !Array.isArray(geometry.artifacts)) {
    throw new Error(`Geometry bundle has invalid geometry.artifacts for ${asset.asset_id}`);
  }
  for (const artifact of geometry.artifacts ?? []) {
    if (!artifact || typeof artifact !== "object") {
      throw new Error(`Geometry bundle has invalid artifact for ${asset.asset_id}`);
    }
    requireNonempty(artifact.kind, "geometry.artifacts[].kind");
    requireNonempty(artifact.label, "geometry.artifacts[].label");
    requirePortablePath(artifact.path, "geometry.artifacts[].path");
    requireSha256(artifact.sha256, "geometry.artifacts[].sha256");
    requireNonempty(artifact.media_type, "geometry.artifacts[].media_type");
    if (
      JSON_ARTIFACT_KINDS.has(artifact.kind) &&
      (artifact.media_type !== "application/json" ||
        !artifact.path.toLowerCase().endsWith(".json"))
    ) {
      throw new Error(`Geometry artifact ${artifact.kind} must be JSON`);
    }
    if (artifact.kind === "source_usd" || artifact.kind === "output_usd") {
      if (!/\.(?:usd|usda|usdc)$/i.test(artifact.path)) {
        throw new Error(`Geometry artifact ${artifact.kind} must be USD`);
      }
      if (!USD_MEDIA_TYPES.has(artifact.media_type)) {
        throw new Error(`Geometry artifact ${artifact.kind} has invalid USD media type`);
      }
    }
    artifactIdentities.push(`${artifact.kind}:${artifact.path}`);
  }
  assertUniqueIds(artifactIdentities, "Geometry artifact identity");

  const renderViews: string[] = [];
  if (
    geometry.render_evidence !== undefined &&
    !Array.isArray(geometry.render_evidence)
  ) {
    throw new Error(
      `Geometry bundle has invalid geometry.render_evidence for ${asset.asset_id}`,
    );
  }
  for (const render of geometry.render_evidence ?? []) {
    if (!render || typeof render !== "object") {
      throw new Error(`Geometry bundle has invalid render evidence for ${asset.asset_id}`);
    }
    requireNonempty(render.view, "geometry.render_evidence[].view");
    requirePortablePath(render.image, "geometry.render_evidence[].image");
    if (!/\.(?:png|jpe?g)$/i.test(render.image)) {
      throw new Error(`Geometry render ${render.view} must be a raster image`);
    }
    requireSha256(render.image_sha256, "geometry.render_evidence[].image_sha256");
    requirePortablePath(render.camera, "geometry.render_evidence[].camera");
    if (!render.camera.toLowerCase().endsWith(".json")) {
      throw new Error(`Geometry render ${render.view} camera must be JSON`);
    }
    requireSha256(render.camera_sha256, "geometry.render_evidence[].camera_sha256");
    requireNonempty(render.renderer, "geometry.render_evidence[].renderer");
    if (render.renderer.toLowerCase() !== "ovrtx") {
      throw new Error(
        `Geometry render ${render.view} is not exact OVRTX evidence: ${render.renderer}`,
      );
    }
    requireNonempty(render.render_quality, "geometry.render_evidence[].render_quality");
    requireNonempty(
      render.ovrtx_render_mode,
      "geometry.render_evidence[].ovrtx_render_mode",
    );
    requireNonempty(render.active_aov, "geometry.render_evidence[].active_aov");
    if (!Number.isInteger(render.ovrtx_num_sensor_updates) || render.ovrtx_num_sensor_updates < 1) {
      throw new Error(
        `Geometry render ${render.view} has invalid ovrtx_num_sensor_updates`,
      );
    }
    if (!Number.isInteger(render.width) || render.width < 1) {
      throw new Error(`Geometry render ${render.view} has invalid width`);
    }
    if (!Number.isInteger(render.height) || render.height < 1) {
      throw new Error(`Geometry render ${render.view} has invalid height`);
    }
    if (render.fallback !== undefined && typeof render.fallback !== "boolean") {
      throw new Error(`Geometry render ${render.view} has invalid fallback state`);
    }
    if (render.fallback === true) {
      throw new Error(
        `Geometry render ${render.view} is fallback output, not exact OVRTX evidence`,
      );
    }
    if (
      render.elapsed_seconds !== undefined &&
      (!Number.isFinite(render.elapsed_seconds) || render.elapsed_seconds < 0)
    ) {
      throw new Error(`Geometry render ${render.view} has invalid elapsed_seconds`);
    }
    const indexedPath = asset.renders?.[render.view];
    if (indexedPath !== undefined && indexedPath !== render.image) {
      throw new Error(`Geometry render path mismatch for ${asset.asset_id}/${render.view}`);
    }
    renderViews.push(render.view);
  }
  assertUniqueIds(renderViews, "Geometry render view");

  if (geometry.findings !== undefined && !Array.isArray(geometry.findings)) {
    throw new Error(`Geometry bundle has invalid geometry.findings for ${asset.asset_id}`);
  }
  for (const finding of geometry.findings ?? []) {
    if (!finding || typeof finding !== "object") {
      throw new Error(`Geometry bundle has invalid finding for ${asset.asset_id}`);
    }
    requireNonempty(finding.severity, "geometry.findings[].severity");
    requireNonempty(finding.title, "geometry.findings[].title");
    requireNonempty(finding.detail, "geometry.findings[].detail");
  }
}

export function geometryArtifactDeclarations(
  assets: GeometryCompatibleAsset[],
): GeometryArtifactDigest[] {
  const declarations = new Map<string, string>();
  const addDeclaration = (path: unknown, sha256: unknown, field: string) => {
    requirePortablePath(path, `${field}.path`);
    requireSha256(sha256, `${field}.sha256`);
    const existing = declarations.get(path);
    if (existing !== undefined && existing !== sha256) {
      throw new Error(`Conflicting Geometry artifact SHA-256 for ${path}`);
    }
    declarations.set(path, sha256);
  };

  for (const asset of assets) {
    if (!asset.geometry) {
      throw new Error(`Geometry extension missing for ${asset.asset_id}`);
    }
    for (const artifact of asset.geometry.artifacts ?? []) {
      addDeclaration(
        artifact.path,
        artifact.sha256,
        "geometry.artifacts[]",
      );
    }
    for (const render of asset.geometry.render_evidence ?? []) {
      addDeclaration(
        render.image,
        render.image_sha256,
        "geometry.render_evidence[].image",
      );
      addDeclaration(
        render.camera,
        render.camera_sha256,
        "geometry.render_evidence[].camera",
      );
    }
  }

  return [...declarations.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([path, sha256]) => ({ path, sha256 }));
}

function isExternalConsumerReference(path: string) {
  return /^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith("//");
}

export interface GeometryRunArtifactBinding {
  path: string;
  sha256?: string;
}

export function geometryRunArtifactBindings({
  entry,
  bundle,
  score,
  evaluation,
}: GeometryRunContract): GeometryRunArtifactBinding[] {
  const bindings = new Map<string, string | undefined>();
  const addBinding = (path: string, sha256?: string) => {
    const existing = bindings.get(path);
    if (
      bindings.has(path) &&
      existing !== undefined &&
      sha256 !== undefined &&
      existing !== sha256
    ) {
      throw new Error(`Conflicting Geometry run artifact SHA-256 for ${path}`);
    }
    bindings.set(path, existing ?? sha256);
  };
  const addBundlePath = (
    path: unknown,
    field: string,
    options: { allowExternal?: boolean; sha256?: unknown } = {},
  ) => {
    if (
      options.allowExternal &&
      typeof path === "string" &&
      isExternalConsumerReference(path)
    ) {
      return;
    }
    requirePortablePath(path, field);
    if (options.sha256 !== undefined) {
      requireSha256(options.sha256, `${field}.sha256`);
    }
    addBinding(`bundle/${path}`, options.sha256 as string | undefined);
  };

  const artifactDirectory = entry.artifact_directory ?? entry.run_id;
  requirePortablePath(artifactDirectory, "artifact_directory");
  if (artifactDirectory.includes("/")) {
    throw new Error(`Geometry run has unsafe artifact_directory`);
  }

  addBinding("benchmark_run.json");
  addBinding("bundle/run.json");
  if (score !== null) addBinding("score/suite_result.json");
  if (evaluation !== undefined && evaluation !== null) {
    addBinding("evaluation/material_evaluation.json");
  }

  for (const asset of bundle.assets) {
    if (asset.references !== undefined && !Array.isArray(asset.references)) {
      throw new Error(`Geometry bundle has invalid assets[].references`);
    }
    if (
      asset.renders !== undefined &&
      (!asset.renders ||
        typeof asset.renders !== "object" ||
        Array.isArray(asset.renders))
    ) {
      throw new Error(`Geometry bundle has invalid assets[].renders`);
    }
    if (
      asset.artifact_links !== undefined &&
      !Array.isArray(asset.artifact_links)
    ) {
      throw new Error(`Geometry bundle has invalid assets[].artifact_links`);
    }
    if (
      asset.local_artifacts !== undefined &&
      (!asset.local_artifacts ||
        typeof asset.local_artifacts !== "object" ||
        Array.isArray(asset.local_artifacts))
    ) {
      throw new Error(`Geometry bundle has invalid assets[].local_artifacts`);
    }
    if (asset.review_path !== undefined) {
      addBundlePath(asset.review_path, "assets[].review_path");
    }
    if (asset.source_usd !== undefined) {
      addBundlePath(asset.source_usd, "assets[].source_usd");
    }
    if (asset.output_usd !== undefined) {
      addBundlePath(asset.output_usd, "assets[].output_usd");
    }
    for (const reference of asset.references ?? []) {
      if (!reference || typeof reference !== "object") {
        throw new Error(`Geometry bundle has invalid assets[].references[]`);
      }
      addBundlePath(reference.path, "assets[].references[].path");
    }
    for (const path of Object.values(asset.renders ?? {})) {
      addBundlePath(path, "assets[].renders[]");
    }
    for (const artifact of asset.artifact_links ?? []) {
      if (!artifact || typeof artifact !== "object") {
        throw new Error(`Geometry bundle has invalid assets[].artifact_links[]`);
      }
      if (artifact.uri === undefined) continue;
      addBundlePath(artifact.uri, "assets[].artifact_links[].uri", {
        allowExternal: true,
        sha256: artifact.sha256,
      });
    }
    for (const [label, path] of Object.entries(asset.local_artifacts ?? {})) {
      if (label === "run_dir") continue;
      addBundlePath(path, `assets[].local_artifacts.${label}`);
    }
  }

  for (const declaration of geometryArtifactDeclarations(bundle.assets)) {
    addBundlePath(declaration.path, "geometry declaration", {
      sha256: declaration.sha256,
    });
  }
  for (const scoredCase of score?.cases ?? []) {
    if (
      scoredCase.artifacts !== undefined &&
      (!scoredCase.artifacts ||
        typeof scoredCase.artifacts !== "object" ||
        Array.isArray(scoredCase.artifacts))
    ) {
      throw new Error(`Geometry score has invalid cases[].artifacts`);
    }
    if (
      scoredCase.thumbnails !== undefined &&
      !Array.isArray(scoredCase.thumbnails)
    ) {
      throw new Error(`Geometry score has invalid cases[].thumbnails`);
    }
    for (const path of Object.values(scoredCase.artifacts ?? {})) {
      addBundlePath(path, "score.cases[].artifacts[]", { allowExternal: true });
    }
    for (const path of scoredCase.thumbnails ?? []) {
      if (
        typeof path === "string" &&
        /^data:image\/(?:png|jpeg|gif);base64,[a-z0-9+/=]+$/i.test(path.trim())
      ) {
        continue;
      }
      addBundlePath(path, "score.cases[].thumbnails[]", { allowExternal: true });
    }
  }

  return [...bindings.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([path, sha256]) => ({
      path,
      ...(sha256 === undefined ? {} : { sha256 }),
    }));
}

function validateGeometryIntegrity(contract: GeometryRunContract) {
  const { entry } = contract;
  const value = entry.geometry_integrity;
  if (!value || typeof value !== "object") {
    throw new Error(`Geometry artifact integrity missing for ${entry.run_id}`);
  }
  const integrity = value as Partial<GeometryArtifactIntegrity>;
  if (integrity.schema_version !== GEOMETRY_ARTIFACT_INTEGRITY_SCHEMA) {
    throw new Error(`Unsupported Geometry artifact integrity for ${entry.run_id}`);
  }
  if (!Array.isArray(integrity.artifacts)) {
    throw new Error(`Geometry artifact integrity list is invalid for ${entry.run_id}`);
  }

  for (const artifact of integrity.artifacts) {
    if (!artifact || typeof artifact !== "object") {
      throw new Error(`Geometry artifact integrity record is invalid for ${entry.run_id}`);
    }
    requirePortablePath(artifact.path, "geometry_integrity.artifacts[].path");
    requireSha256(artifact.sha256, "geometry_integrity.artifacts[].sha256");
  }
  assertUniqueIds(
    integrity.artifacts.map((artifact) => artifact.path),
    "Geometry integrity artifact path",
  );

  const expected = geometryRunArtifactBindings(contract);
  const verifiedByPath = new Map(
    integrity.artifacts.map((artifact) => [artifact.path, artifact.sha256]),
  );
  if (verifiedByPath.size !== expected.length) {
    throw new Error(`Geometry artifact integrity coverage mismatch for ${entry.run_id}`);
  }
  for (const declaration of expected) {
    const verifiedSha256 = verifiedByPath.get(declaration.path);
    if (verifiedSha256 === undefined) {
      throw new Error(
        `Geometry artifact integrity missing ${declaration.path} for ${entry.run_id}`,
      );
    }
    if (
      declaration.sha256 !== undefined &&
      verifiedSha256 !== declaration.sha256
    ) {
      throw new Error(
        `Geometry artifact integrity mismatch for ${declaration.path}`,
      );
    }
  }
}

export function validateGeometryRunContract({
  entry,
  manifest,
  bundle,
  score,
  evaluation,
}: GeometryRunContract, options: { requireIntegrity?: boolean } = {}) {
  if (manifest.schema_version !== "content-agent-benchmark-run.v1") {
    throw new Error(`Unsupported Geometry run manifest: ${manifest.schema_version}`);
  }
  if (bundle.schema_version !== STANDARD_BENCHMARK_BUNDLE_SCHEMA) {
    throw new Error(`Unsupported Geometry benchmark bundle: ${bundle.schema_version}`);
  }
  if (
    entry.run_id !== manifest.run_id ||
    entry.run_id !== bundle.run_id ||
    (score !== null && entry.run_id !== score.run_id)
  ) {
    throw new Error(`Run ID mismatch in Geometry benchmark artifacts for ${entry.run_id}`);
  }
  if (entry.workflow !== manifest.workflow || entry.workflow !== bundle.workflow) {
    throw new Error(`Workflow mismatch in Geometry benchmark artifacts for ${entry.run_id}`);
  }
  if (!Array.isArray(bundle.assets)) {
    throw new Error(`Geometry benchmark bundle assets must be an array`);
  }
  if (score !== null && !Array.isArray(score.cases)) {
    throw new Error(`Geometry suite result cases must be an array`);
  }
  for (const asset of bundle.assets) {
    if (!asset || typeof asset !== "object") {
      throw new Error(`Geometry benchmark bundle has an invalid asset record`);
    }
  }
  for (const scoredCase of score?.cases ?? []) {
    if (!scoredCase || typeof scoredCase !== "object") {
      throw new Error(`Geometry suite result has an invalid case record`);
    }
  }
  assertUniqueIds(
    bundle.assets.map((asset) => asset.asset_id),
    "Geometry bundle asset ID",
  );
  assertUniqueIds(
    (score?.cases ?? []).map((scoredCase) => scoredCase.asset_id),
    "Geometry score case ID",
  );
  for (const scoredCase of score?.cases ?? []) {
    if (scoredCase.workflow !== entry.workflow) {
      throw new Error(
        `Geometry score workflow mismatch for ${entry.run_id}/${scoredCase.asset_id}`,
      );
    }
  }
  for (const asset of bundle.assets) validateGeometryExtension(asset);
  if (options.requireIntegrity !== false) {
    validateGeometryIntegrity({ entry, manifest, bundle, score, evaluation });
  }
}

function titleCase(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function familyLabel(value: string) {
  return value === "cad_to_simready" || value === "cad-to-simready"
    ? "CAD-to-SimReady"
    : titleCase(value);
}

function findingSeverity(value: string): ReviewSeverity {
  const severity = value.toLowerCase();
  if (["blocking", "critical", "error"].includes(severity)) return "blocking";
  if (severity === "major") return "major";
  if (severity === "warning") return "warning";
  if (["none", "minor"].includes(severity)) return severity as ReviewSeverity;
  return "warning";
}

function displayMetrics(
  asset: GeometryCompatibleAsset,
  scoredCase: GeometryCompatibleSuiteCase | undefined,
) {
  const values: Record<string, unknown> = {
    ...(scoredCase?.metrics ?? {}),
    ...(asset.metrics ?? {}),
  };
  if (asset.geometry?.outcome) values.geometry_outcome = asset.geometry.outcome;
  if (asset.geometry?.handoff_ready) {
    values.handoff_ready = asset.geometry.handoff_ready;
  }
  if (asset.geometry?.render_evidence) {
    values.ovrtx_render_count = asset.geometry.render_evidence.length;
  }
  return Object.fromEntries(
    Object.entries(values).filter(
      (entry): entry is [string, string | number | boolean] =>
        typeof entry[1] === "string" ||
        typeof entry[1] === "number" ||
        typeof entry[1] === "boolean",
    ),
  );
}

export function adaptGeometryAsset({
  asset,
  scoredCase,
  resolveArtifact,
}: GeometryAssetAdapterInput): GeometryAssetAdapterResult {
  const geometry = asset.geometry;
  if (!geometry) {
    throw new Error(`Geometry extension missing for ${asset.asset_id}`);
  }
  const resolveRequired = (path: string) => {
    const resolved = resolveArtifact(path);
    if (!resolved) throw new Error(`Unsafe Geometry artifact path for ${asset.asset_id}: ${path}`);
    return resolved;
  };
  const renders = (geometry.render_evidence ?? []).map((render) => ({
    label: `${titleCase(render.view)} · OVRTX`,
    view: titleCase(render.view),
    path: resolveRequired(render.image),
    kind: "render" as const,
    sha256: render.image_sha256,
    renderer: "OVRTX",
    settings: {
      render_quality: render.render_quality,
      ovrtx_render_mode: render.ovrtx_render_mode,
      ovrtx_num_sensor_updates: render.ovrtx_num_sensor_updates,
      active_aov: render.active_aov,
      width: render.width,
      height: render.height,
      fallback: render.fallback === true,
      elapsed_seconds: render.elapsed_seconds,
    },
    camera: {
      label: `${titleCase(render.view)} camera`,
      path: resolveRequired(render.camera),
      kind: "ovrtx_camera",
      sha256: render.camera_sha256,
    },
  } satisfies EvidenceItem));
  const reports = (geometry.artifacts ?? []).map((artifact) => ({
    label: artifact.label,
    path: resolveRequired(artifact.path),
    kind: artifact.kind,
    sha256: artifact.sha256,
    media_type: artifact.media_type,
  } satisfies ReportLink));
  const findings = (geometry.findings ?? []).map((finding) => ({
    severity: findingSeverity(finding.severity),
    title: finding.title,
    detail: finding.detail,
  } satisfies Finding));
  const state = [
    geometry.outcome ? titleCase(geometry.outcome) : null,
    geometry.handoff_ready
      ? `Handoff ${titleCase(geometry.handoff_ready)}`
      : null,
  ].filter((value): value is string => value !== null);

  return {
    family: geometry.benchmark_family
      ? familyLabel(geometry.benchmark_family)
      : null,
    workflowStatus: state.length ? state.join(" · ") : null,
    renders,
    reports,
    metrics: displayMetrics(asset, scoredCase),
    findings,
  };
}
