// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AssetFeedback,
  AssetMaterialEvaluation,
  AssetMeshSegmentationEvaluation,
  AssetPhysicsEvaluation,
  AssetReview,
  AssetResult,
  AssetSimReadyValidation,
  BenchmarkBundle,
  BenchmarkProvenance,
  BenchmarkRun,
  EvidenceItem,
  Finding,
  MaterialAgreementScores,
  MaterialJudge,
  MaterialJudgeResult,
  MaterialScores,
  MeshSegmentationEvaluationSummary,
  MeshSegmentationMatch,
  PhysicsEvaluationSummary,
  PhysicsPartRow,
  ReportLink,
  ReviewSeverity,
  ReviewVerdict,
  ResultStatus,
  RunExecutionMetadata,
  SignalSummary,
  WorkflowSummary,
} from "../types/benchmark";
import {
  adaptGeometryAsset,
  isGeometryIntegrityWorkflow,
  isGeometryWorkflow,
  validateGeometryRunContract,
  type BundleArtifactLink,
  type GeometryAssetExtension,
} from "./geometryBenchmark";

type JsonValue = string | number | boolean | null;
type JsonRecord = Record<string, JsonValue>;

interface BenchmarkIndex {
  schema_version: "content-agent-benchmark-index.v1";
  generated_at: string;
  runs: BenchmarkIndexRun[];
}

interface BenchmarkIndexRun {
  artifact_base_url: string;
  artifact_directory?: string;
  artifact_version: string;
  bundle_url: string;
  created_at: string;
  evaluation_url: string | null;
  geometry_integrity?: unknown;
  manifest_url: string;
  run_id: string;
  score_url: string | null;
  status: string;
  workflow: string;
}

const GEOMETRY_ARTIFACT_VERSION_PATTERN = /^sha256:[0-9a-f]{64}$/;
const GEOMETRY_ARTIFACT_VERSION_PARAM = "artifact_version";

interface MaterialEvaluationDocument {
  schema_version: "content-agent-material-evaluation.v3";
  run_id: string;
  workflow: string;
  method: {
    prompt_version: string;
    caveat: string;
  };
  judges: MaterialJudge[];
  aggregate: {
    asset_count: number;
    evaluated_asset_count: number;
    mean_scores: Partial<Record<keyof MaterialScores, number | null>>;
    mean_agreement_scores: Partial<Record<keyof MaterialScores, number | null>>;
    acceptable_rate: number | null;
    high_fidelity_rate: number | null;
    mean_judge_confidence: number | null;
    mean_assignment_coverage_score: number | null;
    authoring_coverage_score: number | null;
    rejected_assignment_rate: number | null;
    missing_assignment_rate: number | null;
    final_output_asset_count: number;
    final_visible_mesh_count: number | null;
    final_bound_visible_mesh_count: number | null;
    final_partially_bound_visible_mesh_count: number | null;
    final_unbound_visible_mesh_count: number | null;
    final_mesh_binding_coverage_score: number | null;
  };
  assets: MaterialEvaluationAsset[];
}

export function compareRunsNewestFirst(
  left: Pick<BenchmarkRun, "created_at" | "run_id">,
  right: Pick<BenchmarkRun, "created_at" | "run_id">,
): number {
  const leftTime = Date.parse(left.created_at);
  const rightTime = Date.parse(right.created_at);
  const leftValid = Number.isFinite(leftTime);
  const rightValid = Number.isFinite(rightTime);
  if (leftValid !== rightValid) return rightValid ? 1 : -1;
  if (leftValid && leftTime !== rightTime) return rightTime - leftTime;
  return right.run_id.localeCompare(left.run_id);
}

interface MaterialEvaluationAsset {
  asset_id: string;
  status: string;
  scores: Partial<Record<keyof MaterialScores, number>> | null;
  agreement_scores?: Partial<
    Record<keyof MaterialScores, number | null>
  > | null;
  confidence?: number;
  issues?: string[];
  semantic_consistency_findings?: string[];
  judge_results?: Array<{
    judge: MaterialJudge;
    scores: Partial<Record<keyof MaterialScores, number>> | null;
    confidence?: number;
  }>;
  rationale?: string;
  structural?: {
    assignment_coverage_score?: number | null;
    authoring_coverage_score?: number | null;
    rejected_assignment_rate?: number | null;
    missing_assignment_rate?: number | null;
    coverage_limitation?: string | null;
    final_output_available?: boolean;
    final_visible_mesh_count?: number | null;
    final_bound_visible_mesh_count?: number | null;
    final_partially_bound_visible_mesh_count?: number | null;
    final_unbound_visible_mesh_count?: number | null;
    final_mesh_binding_coverage_score?: number | null;
    final_unbound_visible_mesh_paths?: string[];
    unique_material_count?: number | null;
    material_distribution_entropy?: number | null;
    unknown_assignment_rate?: number | null;
  };
}

interface AgenticRunManifest {
  schema_version: string;
  run_id: string;
  workflow: string;
  status: string;
  created_at: string;
  case_count: number;
  metadata?: {
    git?: GitMetadata;
    provenance?: BenchmarkProvenance;
    execution?: RunExecutionMetadata;
    vlm?: {
      backend?: string;
      model?: string;
    };
  };
}

interface GitMetadata {
  branch?: string;
  commit?: string;
  dirty?: boolean | string;
}

interface BenchmarkRunBundle {
  schema_version: string;
  run_id: string;
  label: string;
  workflow: string;
  created_at: string;
  source?: string;
  git?: GitMetadata;
  config?: {
    provenance?: BenchmarkProvenance;
    [key: string]: unknown;
  };
  summary?: {
    asset_count?: number;
    common_config?: JsonRecord;
    execution_mode?: string;
    metrics?: JsonRecord;
    statuses?: Record<string, number>;
  };
  assets: BenchmarkBundleAsset[];
}

interface BenchmarkBundleAsset {
  asset_id: string;
  name?: string;
  prompt?: string;
  status?: string;
  source_usd?: string;
  output_usd?: string;
  tags?: string[];
  config?: JsonRecord;
  metrics?: JsonRecord;
  simready_validation?: unknown;
  references?: Array<{
    label?: string;
    path: string;
    provenance?: JsonRecord;
  }>;
  renders?: Record<string, string>;
  artifact_links?: BundleArtifactLink[];
  local_artifacts?: Record<string, string>;
  review_path?: string;
  // Written by export_benchmark_runs.py: where this asset's files actually
  // landed, since the exporter sanitizes the directory but not `asset_id`.
  export_root?: string;
  geometry?: GeometryAssetExtension;
}

interface PersistedReviewDocument {
  asset_id?: string;
  reviewer?: string;
  verdict?: string;
  severity?: string;
  score?: number | null;
  tags?: string[];
  comment?: string;
  needs_rerun?: boolean;
  updated_at?: string | null;
  feedback?: AssetFeedback[];
}

interface SuiteResult {
  run_id: string;
  created_at: string;
  status: string;
  counts?: Partial<Record<ResultStatus, number>>;
  cases: SuiteCase[];
}

interface SuiteCase {
  workflow: string;
  asset_id: string;
  status: string;
  signals?: SuiteSignal[];
  metrics?: Record<string, unknown>;
  artifacts?: Record<string, string>;
  thumbnails?: string[];
  tags?: string[];
  run_dir?: string;
  error?: string | null;
  skipped?: boolean;
}

interface SuiteSignal {
  name: string;
  ok: boolean;
  severity: "blocking" | "warning" | string;
  value?: unknown;
  detail?: string;
}

// Historical bundles name the workflow differently from the canonical id the
// artifact directory uses. `score_bundle` accepts these through each adapter's
// `bundle_workflow_aliases` and emits canonical scored cases, so rejecting them
// here would make a run the backend scored perfectly well unreadable.
const BUNDLE_WORKFLOW_ALIASES: Record<string, string> = {
  agentic_workflow: "material-agentic",
  fixed_pipeline: "material-fixed",
  // Historical bundle ids the physics benchmark adapters accept through
  // bundle_workflow_aliases; the dashboard must resolve them the same way.
  physics: "physics-fixed",
  physics_agentic: "physics-agentic",
};

function canonicalWorkflow(workflow: string) {
  return BUNDLE_WORKFLOW_ALIASES[workflow] ?? workflow;
}

const WORKFLOW_METADATA: Record<
  string,
  Pick<WorkflowSummary, "id" | "name" | "description">
> = {
  "material-agentic": {
    id: "material-agentic",
    name: "Material Agentic",
    description:
      "Agentic material assignment results, visual evidence, and scoring signals.",
  },
  "material-fixed": {
    id: "material-fixed",
    name: "Material Fixed Pipeline",
    description:
      "Fixed-pipeline Material Agent Service results and downloaded artifacts.",
  },
  mesh_segmentation: {
    id: "mesh-segmentation",
    name: "Mesh Segmentation",
    description:
      "PartObjaverse semantic mesh partitioning, contract validation, and ground-truth quality metrics.",
  },
  "physics-fixed": {
    id: "physics-fixed",
    name: "Physics Fixed Pipeline",
    description:
      "Fixed VLM pipeline physics-property predictions scored against PhysX-Mobility ground truth.",
  },
  "physics-agentic": {
    id: "physics-agentic",
    name: "Physics Agentic",
    description:
      "Agentic physics-authoring predictions scored against the same PhysX-Mobility ground truth as the fixed pipeline.",
  },
  public_cad: {
    id: "cad-benchmark",
    name: "CAD Agent Benchmark",
    description:
      "Same-model public CAD comparisons with prompts, native baselines, OVRTX evidence, and authoritative scores.",
  },
  cad_to_simready: {
    id: "cad-to-simready",
    name: "CAD-to-SimReady",
    description: "CAD conversion and SimReady validation results.",
  },
};

const WORKFLOW_ALIASES: Record<string, keyof typeof WORKFLOW_METADATA> = {
  "cad-to-simready": "cad_to_simready",
  "geometry-cad-to-simready": "cad_to_simready",
  "geometry-public-cad": "public_cad",
};

function workflowMetadata(workflow: string) {
  return WORKFLOW_METADATA[WORKFLOW_ALIASES[workflow] ?? workflow];
}

const STATUS_LABELS: Record<ResultStatus, string> = {
  pass: "Pass",
  warn: "Warn",
  fail: "Fail",
  error: "Error",
  skipped: "Skipped",
  unknown: "Unknown",
};

const STATUS_TONES: Record<ResultStatus, AssetResult["preview"]["tone"]> = {
  pass: "good",
  warn: "warning",
  fail: "bad",
  error: "bad",
  skipped: "blocked",
  unknown: "blocked",
};

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function indexFingerprint(index: BenchmarkIndex) {
  return JSON.stringify(
    [...index.runs]
      .sort((left, right) => left.run_id.localeCompare(right.run_id))
      .map((entry) => ({
        run_id: entry.run_id,
        workflow: entry.workflow,
        artifact_version: entry.artifact_version,
        artifact_directory: entry.artifact_directory,
        status: entry.status,
        created_at: entry.created_at,
        manifest_url: entry.manifest_url,
        bundle_url: entry.bundle_url,
        evaluation_url: entry.evaluation_url,
        score_url: entry.score_url,
      })),
  );
}

async function loadBenchmarkIndex(indexUrl: string) {
  const index = await fetchJson<BenchmarkIndex>(indexUrl);
  if (index.schema_version !== "content-agent-benchmark-index.v1") {
    throw new Error(`Unsupported benchmark index: ${index.schema_version}`);
  }
  const fallbackBase =
    typeof window === "undefined" ? "http://localhost/" : window.location.href;
  const indexBase = new URL(indexUrl, fallbackBase);
  const validateUrl = (value: unknown, field: string) => {
    if (typeof value !== "string" || !value.trim()) {
      throw new Error(`Invalid ${field} in benchmark index`);
    }
    const resolved = new URL(value, indexBase);
    if (
      !["http:", "https:"].includes(resolved.protocol) ||
      resolved.origin !== indexBase.origin ||
      resolved.username ||
      resolved.password
    ) {
      throw new Error(`Unsafe ${field} in benchmark index: ${value}`);
    }
  };
  for (const entry of index.runs) {
    validateUrl(entry.artifact_base_url, "artifact_base_url");
    validateUrl(entry.manifest_url, "manifest_url");
    validateUrl(entry.bundle_url, "bundle_url");
    if (entry.evaluation_url !== null) {
      validateUrl(entry.evaluation_url, "evaluation_url");
    }
    if (entry.score_url !== null) validateUrl(entry.score_url, "score_url");
    if (
      entry.geometry_integrity !== undefined &&
      !GEOMETRY_ARTIFACT_VERSION_PATTERN.test(entry.artifact_version)
    ) {
      throw new Error(`Invalid Geometry artifact version for ${entry.run_id}`);
    }
  }
  return index;
}

export async function loadBenchmarkIndexFingerprint(indexUrl: string) {
  return indexFingerprint(await loadBenchmarkIndex(indexUrl));
}

function resultStatus(value: string | undefined): ResultStatus {
  if (
    value === "pass" ||
    value === "warn" ||
    value === "fail" ||
    value === "error" ||
    value === "skipped"
  ) {
    return value;
  }
  return "unknown";
}

function versionedGeometryArtifactUrl(entry: BenchmarkIndexRun, url: string) {
  if (entry.geometry_integrity === undefined) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}${GEOMETRY_ARTIFACT_VERSION_PARAM}=${encodeURIComponent(
    entry.artifact_version,
  )}`;
}

function runArtifactUrl(entry: BenchmarkIndexRun, relativePath: string) {
  const segments = artifactPathSegments(relativePath);
  return versionedGeometryArtifactUrl(
    entry,
    `${entry.artifact_base_url}/${segments.map(encodeURIComponent).join("/")}`,
  );
}

function artifactPathSegments(path: string) {
  if (
    path.startsWith("/") ||
    path.includes("\\") ||
    /^[a-z][a-z0-9+.-]*:/i.test(path)
  ) {
    throw new Error(`Unsafe benchmark artifact path: ${path}`);
  }
  const segments = path.split("/");
  if (
    segments.some(
      (segment) =>
        !segment ||
        segment === "." ||
        segment === ".." ||
        /^[a-z][a-z0-9+.-]*:/i.test(segment),
    )
  ) {
    throw new Error(`Unsafe benchmark artifact path: ${path}`);
  }
  return segments;
}

function hasUnsafeArtifactSegment(path: string) {
  try {
    artifactPathSegments(path);
    return false;
  } catch {
    return true;
  }
}

function localArtifactUrl(entry: BenchmarkIndexRun, path: string) {
  if (path.includes("\\")) return null;
  const normalizedPath = path;
  const artifactDirectory = (entry.artifact_directory ?? entry.run_id)?.trim();
  if (
    artifactDirectory &&
    !artifactDirectory.includes("/") &&
    !hasUnsafeArtifactSegment(artifactDirectory)
  ) {
    const marker = `/${artifactDirectory}/`;
    const markerIndex = normalizedPath.lastIndexOf(marker);
    if (markerIndex >= 0) {
      const relativePath = normalizedPath.slice(markerIndex + marker.length);
      return hasUnsafeArtifactSegment(relativePath)
        ? null
        : runArtifactUrl(entry, relativePath);
    }
  }
  if (
    normalizedPath.startsWith("/") ||
    /^[a-z][a-z0-9+.-]*:/i.test(normalizedPath) ||
    hasUnsafeArtifactSegment(normalizedPath)
  ) {
    return null;
  }
  return runArtifactUrl(entry, `bundle/${normalizedPath}`);
}

function scoreThumbnailUrl(entry: BenchmarkIndexRun, path: string) {
  const value = path.trim();
  if (/^data:image\/(?:png|jpeg|gif);base64,[a-z0-9+/=]+$/i.test(value)) {
    return value;
  }
  return localArtifactUrl(entry, value);
}

function titleCase(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function meshSegmentationAssetName(assetId: string, tags: string[]) {
  const uid = assetId.match(/partobjaverse_([a-f0-9]+)$/i)?.[1];
  const category = tags.find(
    (tag) => !["mesh_segmentation", "partobjaverse_tiny"].includes(tag),
  );
  const prefix = category ? titleCase(category) : "PartObjaverse";
  return uid ? `${prefix} · ${uid.slice(0, 8)}` : titleCase(assetId);
}

function displayRunLabel(bundle: BenchmarkRunBundle, workflowName: string) {
  const withoutRunId = bundle.label.replace(bundle.run_id, "").trim();
  return withoutRunId || `${workflowName} benchmark`;
}

function median(values: number[]) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const right = sorted[middle] ?? 0;
  if (sorted.length % 2) return right;
  return ((sorted[middle - 1] ?? 0) + right) / 2;
}

function numericMetric(metrics: JsonRecord | undefined, name: string) {
  const value = metrics?.[name];
  return typeof value === "number" ? value : 0;
}

function optionalNumericMetric(metrics: JsonRecord | undefined, name: string) {
  const value = metrics?.[name];
  return typeof value === "number" ? value : null;
}

function scoredNumericMetric(
  metrics: Record<string, unknown> | undefined,
  name: string,
) {
  const value = metrics?.[name];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function meshSegmentationMatches(
  metrics: Record<string, unknown> | undefined,
): MeshSegmentationMatch[] {
  const matches = metrics?.matches;
  if (!Array.isArray(matches)) return [];
  return matches.flatMap((match) => {
    if (!match || typeof match !== "object") return [];
    const value = match as Record<string, unknown>;
    const groundTruthId = value.ground_truth_id;
    const predictedId = value.predicted_id;
    const iou = value.iou;
    if (
      typeof groundTruthId !== "number" ||
      typeof predictedId !== "number" ||
      typeof iou !== "number" ||
      !Number.isFinite(groundTruthId) ||
      !Number.isFinite(predictedId) ||
      !Number.isFinite(iou)
    ) {
      return [];
    }
    return [
      {
        ground_truth_id: groundTruthId,
        predicted_id: predictedId,
        iou,
      },
    ];
  });
}

function meshSegmentationEvaluation(
  scoredCase: SuiteCase | undefined,
  isMeshSegmentation: boolean,
): AssetMeshSegmentationEvaluation | undefined {
  // Must agree with how `adaptAsset` decides the asset is a segmentation
  // asset. Gating on the scored case alone meant an asset presented as
  // mesh-segmentation -- because the run is -- lost its quality badges and
  // threshold checks whenever the scored case omitted its workflow, which the
  // artifact guard deliberately tolerates.
  if (!isMeshSegmentation || !scoredCase) return undefined;
  const faceAccuracy = scoredNumericMetric(scoredCase.metrics, "face_accuracy");
  const macroIou = scoredNumericMetric(scoredCase.metrics, "macro_iou");
  const scoreable = faceAccuracy !== null && macroIou !== null;
  const faceAccuracySignal = scoredCase.signals?.find(
    (signal) => signal.name === "face_accuracy",
  );
  const macroIouSignal = scoredCase.signals?.find(
    (signal) => signal.name === "macro_iou",
  );
  const faceAccuracyPassed = scoreable && faceAccuracySignal?.ok === true;
  const macroIouPassed = scoreable && macroIouSignal?.ok === true;
  return {
    status: !scoreable
      ? "not_scoreable"
      : faceAccuracyPassed && macroIouPassed
        ? "pass"
        : "below_threshold",
    face_accuracy: faceAccuracy,
    macro_iou: macroIou,
    area_accuracy: scoredNumericMetric(scoredCase.metrics, "area_accuracy"),
    area_weighted_iou: scoredNumericMetric(
      scoredCase.metrics,
      "area_weighted_iou",
    ),
    face_count: scoredNumericMetric(scoredCase.metrics, "face_count"),
    ground_truth_segment_count: scoredNumericMetric(
      scoredCase.metrics,
      "ground_truth_segment_count",
    ),
    predicted_segment_count: scoredNumericMetric(
      scoredCase.metrics,
      "predicted_segment_count",
    ),
    face_accuracy_passed: faceAccuracyPassed,
    macro_iou_passed: macroIouPassed,
    matches: meshSegmentationMatches(scoredCase.metrics),
  };
}

function mean(values: number[]) {
  return values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : null;
}

function meshSegmentationSummary(
  assets: AssetResult[],
): MeshSegmentationEvaluationSummary {
  const evaluations = assets.flatMap((asset) =>
    asset.mesh_segmentation_evaluation
      ? [asset.mesh_segmentation_evaluation]
      : [],
  );
  const scoreable = evaluations.filter(
    (evaluation) => evaluation.status !== "not_scoreable",
  );
  return {
    asset_count: assets.length,
    scoreable_asset_count: scoreable.length,
    quality_pass_asset_count: scoreable.filter(
      (evaluation) => evaluation.status === "pass",
    ).length,
    final_export_asset_count: assets.filter((asset) =>
      Boolean(asset.output_usd),
    ).length,
    mean_face_accuracy: mean(
      scoreable.flatMap((evaluation) =>
        evaluation.face_accuracy === null ? [] : [evaluation.face_accuracy],
      ),
    ),
    mean_macro_iou: mean(
      scoreable.flatMap((evaluation) =>
        evaluation.macro_iou === null ? [] : [evaluation.macro_iou],
      ),
    ),
  };
}

// Both physics execution modes share the PhysX-Mobility scorer, so they share
// one extraction path; only the workflow id (and therefore the dashboard tab)
// differs.
const PHYSICS_WORKFLOWS = new Set(["physics-fixed", "physics-agentic"]);

function optionalNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function physicsPartRows(
  metrics: Record<string, unknown> | undefined,
): PhysicsPartRow[] {
  const rows = metrics?.property_rows;
  if (!Array.isArray(rows)) return [];
  return rows.flatMap((row) => {
    if (!row || typeof row !== "object") return [];
    const record = row as Record<string, unknown>;
    return [
      {
        prim_path: typeof record.prim_path === "string" ? record.prim_path : "",
        gt_part: typeof record.gt_part === "string" ? record.gt_part : null,
        predicted_material:
          typeof record.predicted_material === "string"
            ? record.predicted_material
            : null,
        gt_material:
          typeof record.gt_material === "string" ? record.gt_material : null,
        material_match:
          typeof record.material_match === "boolean"
            ? record.material_match
            : null,
        predicted_density: optionalNumber(record.predicted_density),
        gt_density: optionalNumber(record.gt_density),
        density_log_ratio: optionalNumber(record.density_log_ratio),
        predicted_static_friction: optionalNumber(
          record.predicted_static_friction,
        ),
        gt_static_friction: optionalNumber(record.gt_static_friction),
        predicted_dynamic_friction: optionalNumber(
          record.predicted_dynamic_friction,
        ),
        gt_dynamic_friction: optionalNumber(record.gt_dynamic_friction),
        friction_abs_error: optionalNumber(record.friction_abs_error),
        predicted_restitution: optionalNumber(record.predicted_restitution),
        gt_restitution: optionalNumber(record.gt_restitution),
        restitution_abs_error: optionalNumber(record.restitution_abs_error),
        predicted_mass_kg: optionalNumber(record.predicted_mass_kg),
      },
    ];
  });
}

function physicsEvaluation(
  scoredCase: SuiteCase | undefined,
  isPhysics: boolean,
): AssetPhysicsEvaluation | undefined {
  // Mirrors meshSegmentationEvaluation: gate on how `adaptAsset` classified
  // the asset (run workflow first), not on the scored case alone.
  if (!isPhysics || !scoredCase) return undefined;
  const metrics = scoredCase.metrics;
  const parts = physicsPartRows(metrics);
  const materialAccuracy = scoredNumericMetric(metrics, "material_accuracy");
  const predictionsCount = scoredNumericMetric(metrics, "predictions_count");
  // A case the scorer never property-scored (no GT, workflow error) carries
  // no part rows and no accuracy. predictions_count alone is not enough: the
  // no-ground-truth branch of the scorer sets only that metric, and an
  // all-n/a panel would read as a scored asset with empty results.
  if (parts.length === 0 && materialAccuracy === null) {
    return undefined;
  }
  return {
    material_accuracy: materialAccuracy,
    gt_parts_total: scoredNumericMetric(metrics, "gt_parts_total"),
    gt_parts_covered: scoredNumericMetric(metrics, "gt_parts_covered"),
    density_log_mae: scoredNumericMetric(metrics, "density_log_mae"),
    density_median_error_factor: scoredNumericMetric(
      metrics,
      "density_median_error_factor",
    ),
    friction_mae: scoredNumericMetric(metrics, "friction_mae"),
    restitution_mae: scoredNumericMetric(metrics, "restitution_mae"),
    predictions_count: predictionsCount,
    zero_mass_prims: scoredNumericMetric(metrics, "zero_mass_prims"),
    parts,
  };
}

function physicsSummary(assets: AssetResult[]): PhysicsEvaluationSummary {
  const evaluations = assets.flatMap((asset) =>
    asset.physics_evaluation ? [asset.physics_evaluation] : [],
  );
  const metricMean = (
    pick: (evaluation: AssetPhysicsEvaluation) => number | null,
  ) =>
    mean(
      evaluations.flatMap((evaluation) => {
        const value = pick(evaluation);
        return value === null ? [] : [value];
      }),
    );
  return {
    asset_count: assets.length,
    scored_asset_count: evaluations.length,
    mean_material_accuracy: metricMean(
      (evaluation) => evaluation.material_accuracy,
    ),
    mean_density_log_mae: metricMean(
      (evaluation) => evaluation.density_log_mae,
    ),
    mean_friction_mae: metricMean((evaluation) => evaluation.friction_mae),
    mean_restitution_mae: metricMean(
      (evaluation) => evaluation.restitution_mae,
    ),
    gt_full_coverage_asset_count: evaluations.filter(
      (evaluation) =>
        evaluation.gt_parts_total !== null &&
        evaluation.gt_parts_covered === evaluation.gt_parts_total,
    ).length,
  };
}

const MATERIAL_SCORE_KEYS: Array<keyof MaterialScores> = [
  "overall_assignment_quality_score",
  "material_identity_score",
  "color_palette_score",
  "surface_finish_score",
  "material_consistency_score",
];

function materialScores(
  values:
    Partial<Record<keyof MaterialScores, number | null>> | null | undefined,
): MaterialScores | null {
  if (
    !values ||
    !MATERIAL_SCORE_KEYS.every((key) => Number.isFinite(values[key]))
  ) {
    return null;
  }
  return Object.fromEntries(
    MATERIAL_SCORE_KEYS.map((key) => [key, values[key] as number]),
  ) as unknown as MaterialScores;
}

function materialAgreementScores(
  values:
    Partial<Record<keyof MaterialScores, number | null>> | null | undefined,
): MaterialAgreementScores {
  return Object.fromEntries(
    MATERIAL_SCORE_KEYS.map((key) => [
      key,
      Number.isFinite(values?.[key]) ? values?.[key] : null,
    ]),
  ) as MaterialAgreementScores;
}

function assetMaterialEvaluation(
  evaluation: MaterialEvaluationAsset | undefined,
): AssetMaterialEvaluation | undefined {
  const scores = materialScores(evaluation?.scores);
  if (!evaluation || evaluation.status !== "evaluated" || !scores)
    return undefined;
  const judgeResults = (evaluation.judge_results ?? []).flatMap((result) => {
    const judgeScores = materialScores(result.scores);
    if (!judgeScores) return [];
    return [
      {
        judge: result.judge,
        scores: judgeScores,
        confidence: result.confidence ?? 0,
      } satisfies MaterialJudgeResult,
    ];
  });
  return {
    scores,
    agreement_scores: materialAgreementScores(evaluation.agreement_scores),
    confidence: evaluation.confidence ?? 0,
    issues: evaluation.issues ?? [],
    semantic_consistency_findings:
      evaluation.semantic_consistency_findings ?? [],
    rationale: evaluation.rationale ?? "",
    judge_results: judgeResults,
    structural: {
      assignment_coverage_score:
        evaluation.structural?.assignment_coverage_score ?? null,
      authoring_coverage_score:
        evaluation.structural?.authoring_coverage_score ??
        evaluation.structural?.assignment_coverage_score ??
        null,
      rejected_assignment_rate:
        evaluation.structural?.rejected_assignment_rate ?? null,
      missing_assignment_rate:
        evaluation.structural?.missing_assignment_rate ?? null,
      coverage_limitation: evaluation.structural?.coverage_limitation ?? null,
      final_output_available:
        evaluation.structural?.final_output_available ?? false,
      final_visible_mesh_count:
        evaluation.structural?.final_visible_mesh_count ?? null,
      final_bound_visible_mesh_count:
        evaluation.structural?.final_bound_visible_mesh_count ?? null,
      final_partially_bound_visible_mesh_count:
        evaluation.structural?.final_partially_bound_visible_mesh_count ?? null,
      final_unbound_visible_mesh_count:
        evaluation.structural?.final_unbound_visible_mesh_count ?? null,
      final_mesh_binding_coverage_score:
        evaluation.structural?.final_mesh_binding_coverage_score ?? null,
      final_unbound_visible_mesh_paths:
        evaluation.structural?.final_unbound_visible_mesh_paths ?? [],
      unique_material_count:
        evaluation.structural?.unique_material_count ?? null,
      material_distribution_entropy:
        evaluation.structural?.material_distribution_entropy ?? null,
      unknown_assignment_rate:
        evaluation.structural?.unknown_assignment_rate ?? null,
    },
  };
}

function displayMetrics(
  asset: BenchmarkBundleAsset,
  scoredCase: SuiteCase | undefined,
  evaluation: AssetMaterialEvaluation | undefined,
) {
  const metrics = {
    ...(scoredCase?.metrics ?? {}),
    ...(asset.metrics ?? {}),
  };
  const selectedNames = [
    "face_accuracy",
    "macro_iou",
    "area_accuracy",
    "area_weighted_iou",
    "face_count",
    "ground_truth_segment_count",
    "predicted_segment_count",
    "material_accuracy",
    "gt_parts_covered",
    "gt_parts_total",
    "density_log_mae",
    "density_median_error_factor",
    "friction_mae",
    "restitution_mae",
    "zero_mass_prims",
    "predictions_count",
    "runtime_seconds",
    "agent_runtime_seconds",
    "model",
    "material_model",
    "physics_model",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
    "driver_input_tokens",
    "driver_cached_input_tokens",
    "driver_output_tokens",
    "driver_total_tokens",
    "vision_input_tokens",
    "vision_cached_input_tokens",
    "vision_output_tokens",
    "vision_total_tokens",
    "vision_invocation_count",
    "combined_total_tokens",
    "render_calls",
    "workbench_api_calls",
    "command_calls",
    "failed_command_calls",
    "model_turn_count",
    "potential_model_cost_usd",
    "potential_model_cost_upper_bound_usd",
    "simready_profile",
    "simready_validation_status",
    "simready_validation_passed",
    "simready_failed_features",
    "simready_failed_requirements",
    "iterations",
    "vqa_status",
  ];
  const ratioMetrics = new Set([
    "face_accuracy",
    "area_accuracy",
    "material_accuracy",
  ]);
  const threeDecimalMetrics = new Set([
    "macro_iou",
    "area_weighted_iou",
    "density_log_mae",
    "friction_mae",
    "restitution_mae",
  ]);
  const workflowMetrics: Record<string, string | number | boolean> =
    Object.fromEntries(
      selectedNames.flatMap((name) => {
        const value = metrics[name];
        if (value === undefined || value === null) return [];
        if (typeof value === "number" && ratioMetrics.has(name)) {
          return [[name, `${(value * 100).toFixed(1)}%`]];
        }
        if (typeof value === "number" && threeDecimalMetrics.has(name)) {
          return [[name, value.toFixed(3)]];
        }
        if (typeof value === "number" && name.includes("model_cost")) {
          return [[name, `$${value.toFixed(4)}`]];
        }
        return typeof value === "string" ||
          typeof value === "number" ||
          typeof value === "boolean"
          ? [[name, value]]
          : [];
      }),
    );
  if (!evaluation) return workflowMetrics;
  const visualScore = (value: number) => `${value}/100`;
  const agreementScore = (value: number | null) =>
    value === null ? "n/a" : visualScore(value);
  return {
    overall_assignment_quality_score: visualScore(
      evaluation.scores.overall_assignment_quality_score,
    ),
    material_identity_score: visualScore(
      evaluation.scores.material_identity_score,
    ),
    color_palette_score: visualScore(evaluation.scores.color_palette_score),
    surface_finish_score: visualScore(evaluation.scores.surface_finish_score),
    material_consistency_score: visualScore(
      evaluation.scores.material_consistency_score,
    ),
    overall_assignment_quality_agreement: agreementScore(
      evaluation.agreement_scores.overall_assignment_quality_score,
    ),
    material_identity_agreement: agreementScore(
      evaluation.agreement_scores.material_identity_score,
    ),
    color_palette_agreement: agreementScore(
      evaluation.agreement_scores.color_palette_score,
    ),
    surface_finish_agreement: agreementScore(
      evaluation.agreement_scores.surface_finish_score,
    ),
    material_consistency_agreement: agreementScore(
      evaluation.agreement_scores.material_consistency_score,
    ),
    final_mesh_binding_coverage_score:
      evaluation.structural.final_mesh_binding_coverage_score === null
        ? "n/a"
        : `${evaluation.structural.final_mesh_binding_coverage_score}%`,
    final_bound_visible_mesh_count:
      evaluation.structural.final_bound_visible_mesh_count ?? "n/a",
    unbound_final_mesh_prims:
      evaluation.structural.final_unbound_visible_mesh_count === null ||
      evaluation.structural.final_visible_mesh_count === null
        ? "n/a"
        : `${evaluation.structural.final_unbound_visible_mesh_count.toLocaleString("en-US")} / ${evaluation.structural.final_visible_mesh_count.toLocaleString("en-US")}`,
    missing_assignment_rate:
      evaluation.structural.missing_assignment_rate === null
        ? "n/a"
        : `${evaluation.structural.missing_assignment_rate}%`,
    unique_material_count: evaluation.structural.unique_material_count ?? "n/a",
    material_distribution_entropy:
      evaluation.structural.material_distribution_entropy ?? "n/a",
    unknown_assignment_rate:
      evaluation.structural.unknown_assignment_rate === null
        ? "n/a"
        : `${evaluation.structural.unknown_assignment_rate}%`,
    judge_confidence: `${Math.round(evaluation.confidence * 100)}%`,
    ...workflowMetrics,
  };
}

// A reference image is reproducible evidence only when its recorded
// provenance carries the source USD and image digests; anything less is a
// diagnostic preview and must be labeled as such (AGENTS.md fail-closed
// evidence rule).
function isSha256Digest(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/i.test(value);
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).sort();
    return `{${keys
      .map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

function sanitizedMetadataFingerprint(record: Record<string, unknown>): string {
  // Mirror the exporter's projection: drop asset_base_dir and per-stage
  // usd_path so a verbatim record and its projection fingerprint equally.
  const metadata = record.render_metadata;
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    return "";
  }
  const source = metadata as Record<string, unknown>;
  const sanitized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(source)) {
    if (key === "asset_base_dir") continue;
    if (key === "stage_preparation" && Array.isArray(value)) {
      sanitized[key] = value.map((entry) =>
        entry && typeof entry === "object" && !Array.isArray(entry)
          ? Object.fromEntries(
              Object.entries(entry as Record<string, unknown>).filter(
                ([entryKey]) =>
                  entryKey !== "usd_path" && entryKey !== "asset_base_dir",
              ),
            )
          : entry,
      );
      continue;
    }
    sanitized[key] = value;
  }
  return stableStringify(sanitized);
}

function remoteIdentityIsOvrtx(record: Record<string, unknown>): boolean {
  const response = record.render_response;
  const data =
    response && typeof response === "object"
      ? (response as Record<string, unknown>).data
      : null;
  const results =
    data && typeof data === "object"
      ? (data as Record<string, unknown>).results
      : null;
  if (response && typeof response === "object") {
    // A render_response is present, so its identity is AUTHORITATIVE —
    // including when it is malformed (not exactly one object result, or a
    // null identity): a stale top-level or metadata identity must not vouch
    // for a response that fails to attest.
    let responseIdentity: unknown = undefined;
    if (Array.isArray(results) && results.length === 1) {
      const first = results[0];
      if (first && typeof first === "object") {
        responseIdentity = (first as Record<string, unknown>).renderer_identity;
      }
    }
    return isCompleteOvrtxIdentity(responseIdentity);
  }
  const identities: unknown[] = [record.renderer_identity];
  const metadata = record.render_metadata;
  if (metadata && typeof metadata === "object" && !Array.isArray(metadata)) {
    identities.push((metadata as Record<string, unknown>).renderer_identity);
  }
  return identities.some(isCompleteOvrtxIdentity);
}

function isCompleteOvrtxIdentity(identity: unknown): boolean {
  // The full attested remote OVRTX identity, matching
  // validated_ovrtx_render_metadata's remote contract — an engine name
  // alone is not attestation.
  if (!identity || typeof identity !== "object" || Array.isArray(identity)) {
    return false;
  }
  const record = identity as Record<string, unknown>;
  const endpoint = record.endpoint;
  let endpointOk = false;
  if (typeof endpoint === "string") {
    try {
      const parsed = new URL(endpoint);
      endpointOk =
        (parsed.protocol === "http:" || parsed.protocol === "https:") &&
        parsed.host.length > 0 &&
        // Userinfo, query, and fragment can all carry credentials (signed
        // URLs, ?token=...) wherever the identity travels.
        parsed.username === "" &&
        parsed.password === "" &&
        parsed.search === "" &&
        parsed.hash === "";
    } catch {
      endpointOk = false;
    }
  }
  return (
    endpointOk &&
    record.engine === "ovrtx" &&
    typeof record.protocol_version === "number" &&
    Number.isInteger(record.protocol_version) &&
    record.protocol_version >= 1 &&
    typeof record.status === "string" &&
    record.status.length > 0
  );
}

function referenceProvenanceSummary(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const renderer = record.renderer;
  const metadata = record.render_metadata;
  const usd = record.source_usd_sha256;
  const image = record.image_sha256;
  // Fail closed: final visual evidence must come from the shared OVRTX
  // render path and carry its render metadata and digests. The recognized
  // OVRTX-backed identities from the render-backend contract are "ovrtx"
  // (local runtime) and "remote" (configured remote OVRTX service).
  // Anything else — a local/preview renderer, a noncanonical identity such
  // as "ovrtx-preview", a missing/empty/non-object render_metadata record,
  // or a malformed digest — must not present as reproducible evidence.
  if (renderer !== "ovrtx" && renderer !== "remote") return null;
  // "remote" is a transport, not an engine: the remote backend permits any
  // compatible renderer, so it counts only with the response's attested
  // OVRTX engine identity.
  if (renderer === "remote" && !remoteIdentityIsOvrtx(record)) return null;
  if (
    !metadata ||
    typeof metadata !== "object" ||
    Array.isArray(metadata) ||
    Object.keys(metadata).length === 0
  ) {
    return null;
  }
  if (!isSha256Digest(usd) || !isSha256Digest(image)) return null;
  return `reference render: ${renderer} · usd sha256 ${usd} · image sha256 ${image}`;
}

// Pure-JS SHA-256 (FIPS 180-4) fallback for insecure origins where
// crypto.subtle is unavailable (the documented plain-HTTP dev hosting): the
// byte check must still run — annotating instead of verifying would let
// replaced bytes present as validated evidence.
export function sha256HexSync(bytes: Uint8Array): string {
  const K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  const state = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
    0x1f83d9ab, 0x5be0cd19,
  ];
  const bitLength = bytes.length * 8;
  const paddedLength = (((bytes.length + 8) >> 6) + 1) << 6;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  new DataView(padded.buffer).setUint32(paddedLength - 4, bitLength >>> 0);
  new DataView(padded.buffer).setUint32(
    paddedLength - 8,
    Math.floor(bitLength / 0x100000000),
  );
  const view = new DataView(padded.buffer);
  const w = new Uint32Array(64);
  const rotr = (value: number, bits: number) =>
    (value >>> bits) | (value << (32 - bits));
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let i = 0; i < 16; i += 1) w[i] = view.getUint32(offset + i * 4);
    for (let i = 16; i < 64; i += 1) {
      const s0 =
        rotr(w[i - 15]!, 7) ^ rotr(w[i - 15]!, 18) ^ (w[i - 15]! >>> 3);
      const s1 = rotr(w[i - 2]!, 17) ^ rotr(w[i - 2]!, 19) ^ (w[i - 2]! >>> 10);
      w[i] = (w[i - 16]! + s0 + w[i - 7]! + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = state as [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ];
    for (let i = 0; i < 64; i += 1) {
      const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + s1 + ch + K[i]! + w[i]!) >>> 0;
      const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + maj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }
    state[0] = (state[0]! + a) >>> 0;
    state[1] = (state[1]! + b) >>> 0;
    state[2] = (state[2]! + c) >>> 0;
    state[3] = (state[3]! + d) >>> 0;
    state[4] = (state[4]! + e) >>> 0;
    state[5] = (state[5]! + f) >>> 0;
    state[6] = (state[6]! + g) >>> 0;
    state[7] = (state[7]! + h) >>> 0;
  }
  return state.map((word) => word.toString(16).padStart(8, "0")).join("");
}

async function sha256HexOfBuffer(buffer: ArrayBuffer): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    const digest = await subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
  }
  return sha256HexSync(new Uint8Array(buffer));
}

// Bound how many reference images are fetched and hashed at once — shared
// across every concurrently loading run, so a multi-run index cannot fan out
// 4×N full-resolution downloads during initial load.
const REFERENCE_VERIFICATION_CONCURRENCY = 4;
let activeVerificationSlots = 0;
const verificationWaiters: Array<() => void> = [];

async function withVerificationSlot<T>(task: () => Promise<T>): Promise<T> {
  if (activeVerificationSlots >= REFERENCE_VERIFICATION_CONCURRENCY) {
    // The releaser transfers its slot directly to the waiter (no decrement),
    // so a newcomer racing this continuation cannot double-claim capacity.
    await new Promise<void>((resolve) => verificationWaiters.push(resolve));
  } else {
    activeVerificationSlots += 1;
  }
  try {
    return await task();
  } finally {
    const waiter = verificationWaiters.shift();
    if (waiter) waiter();
    else activeVerificationSlots -= 1;
  }
}

interface VerifiedReference {
  ok: boolean;
  // Object URL over the exact verified bytes, when the platform supports it;
  // rendering from it removes the fetch-then-display TOCTOU window.
  blobUrl: string | null;
}

// Memoized per (url, digest): dev serving is Cache-Control: no-store and the
// 30s refresh can reload the index, so repeat verifications must not
// re-download and re-hash the same bytes.
const verifiedReferenceCache = new Map<string, Promise<VerifiedReference>>();
const verifiedExactRecordCache = new Map<
  string,
  Promise<Record<string, unknown> | null>
>();

export function clearReferenceVerificationCache(): void {
  // Revoke the object URLs before dropping the entries — the blobs pin the
  // full-resolution image buffers for the page lifetime otherwise.
  for (const verification of verifiedReferenceCache.values()) {
    void verification.then((outcome) => {
      if (outcome.blobUrl) URL.revokeObjectURL(outcome.blobUrl);
    });
  }
  verifiedReferenceCache.clear();
  verifiedExactRecordCache.clear();
}

// A stalled server must not park a verification slot forever: four hung
// fetches would exhaust the global pool and every later reference would stay
// demoted until a page reload. Timed-out verifications resolve ok:false,
// are not cached, and retry on the next refresh.
const VERIFICATION_FETCH_TIMEOUT_MS = 30_000;

function verificationFetchSignal(): AbortSignal | undefined {
  return typeof AbortSignal !== "undefined" && "timeout" in AbortSignal
    ? AbortSignal.timeout(VERIFICATION_FETCH_TIMEOUT_MS)
    : undefined;
}

async function fetchAndVerifyReference(
  path: string,
  expectedDigest: string,
): Promise<VerifiedReference> {
  try {
    const response = await fetch(path, { signal: verificationFetchSignal() });
    if (!response.ok) return { ok: false, blobUrl: null };
    const buffer = await response.arrayBuffer();
    if ((await sha256HexOfBuffer(buffer)) !== expectedDigest) {
      return { ok: false, blobUrl: null };
    }
    let blobUrl: string | null = null;
    try {
      // An untyped blob downloads instead of displaying when "Open original"
      // targets it in a new tab; keep the served type, defaulting to PNG.
      blobUrl = URL.createObjectURL(
        new Blob([buffer], {
          type: response.headers.get("content-type") ?? "image/png",
        }),
      );
    } catch {
      blobUrl = null;
    }
    if (blobUrl === null) {
      // Without an immutable representation of the verified bytes, a later
      // <img>/"Open original" request could fetch DIFFERENT bytes than the
      // ones that passed the digest check — the reference must stay
      // demoted rather than be labeled validated against a mutable URL.
      return { ok: false, blobUrl: null };
    }
    return { ok: true, blobUrl };
  } catch {
    return { ok: false, blobUrl: null };
  }
}

// A shape-valid provenance record is not enough for the live dashboard: the
// served image may have been replaced after collection. Verify the actual
// bytes against the record's image digest and demote the reference (and any
// preview built from it) on mismatch or fetch failure — fail closed. On a
// match, rebind the reference to the verified bytes themselves (a blob URL)
// so a later <img> or "Open original" request cannot fetch different bytes
// than the ones that passed the digest check.
async function fetchAndVerifyExactRecord(
  url: string,
  expectedDigest: string,
): Promise<Record<string, unknown> | null> {
  try {
    const response = await fetch(url, { signal: verificationFetchSignal() });
    if (!response.ok) return null;
    const buffer = await response.arrayBuffer();
    if ((await sha256HexOfBuffer(buffer)) !== expectedDigest) return null;
    const parsed: unknown = JSON.parse(new TextDecoder().decode(buffer));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function deferValidatedReferenceVerification(
  assets: AssetResult[],
): () => Promise<void> {
  const pending = assets.flatMap((asset) =>
    asset.references
      .filter(
        (reference) => reference.diagnostic === false && reference.image_sha256,
      )
      .map((reference) => ({
        asset,
        reference,
        provenance: reference.provenance,
        path: reference.path,
        wasPreview: asset.preview.image_url === reference.path,
      })),
  );
  // Fail closed without blocking first paint: a reference is not validated
  // evidence until its served bytes match the recorded digest, so every
  // candidate starts demoted and is upgraded asynchronously once its bytes
  // verify. The dashboard no longer waits on fetching and SHA-256-hashing
  // every reference image across every run before rendering anything.
  for (const entry of pending) {
    entry.reference.diagnostic = true;
    entry.reference.provenance = null;
    if (entry.wasPreview) {
      entry.asset.preview.image_url = undefined;
    }
  }
  return async () => {
    await Promise.all(
      pending.map(async (entry) => {
        const cacheKey = `${entry.path}\0${entry.reference.image_sha256}`;
        let verification = verifiedReferenceCache.get(cacheKey);
        if (!verification) {
          verification = withVerificationSlot(() =>
            fetchAndVerifyReference(entry.path, entry.reference.image_sha256!),
          );
          verifiedReferenceCache.set(cacheKey, verification);
          // Only a successful verification is cacheable: a transient fetch
          // error or timeout also resolves ok:false, and pinning that would
          // demote the reference for the page lifetime with no retry path.
          void verification.then((outcome) => {
            if (
              !outcome.ok &&
              verifiedReferenceCache.get(cacheKey) === verification
            ) {
              verifiedReferenceCache.delete(cacheKey);
            }
          });
        }
        const result = await verification;
        if (!result.ok) return; // stays demoted — fail closed
        // An exported record is only a projection: the exact OVRTX metadata
        // lives in the bound sidecar. A missing, unreachable, or altered
        // sidecar — or a projection whose binding fields are incomplete —
        // means the evidence contract cannot be satisfied, so the reference
        // stays demoted.
        if (entry.reference.exact_record_broken) return;
        if (entry.reference.exact_record_sha256) {
          const exactUrl = entry.reference.exact_record_url;
          if (!exactUrl) return;
          const exactKey = `${exactUrl}\0${entry.reference.exact_record_sha256}`;
          let exactVerification = verifiedExactRecordCache.get(exactKey);
          if (!exactVerification) {
            exactVerification = withVerificationSlot(() =>
              fetchAndVerifyExactRecord(
                exactUrl,
                entry.reference.exact_record_sha256!,
              ),
            );
            verifiedExactRecordCache.set(exactKey, exactVerification);
            // Cache successes only, same as the image check: a transient
            // failure must retry on the next refresh, not pin a demotion.
            void exactVerification.then((record) => {
              if (
                record === null &&
                verifiedExactRecordCache.get(exactKey) === exactVerification
              ) {
                verifiedExactRecordCache.delete(exactKey);
              }
            });
          }
          const exactRecord = await exactVerification;
          if (!exactRecord) return;
          // The displayed projection must agree with the verified exact
          // record: a valid-but-different sidecar must not vouch for a
          // stale projection.
          const lower = (value: unknown) => String(value ?? "").toLowerCase();
          if (
            exactRecord.renderer !== entry.reference.renderer ||
            lower(exactRecord.image_sha256) !==
              lower(entry.reference.image_sha256) ||
            lower(exactRecord.source_usd_sha256) !==
              lower(entry.reference.source_usd_sha256)
          ) {
            return;
          }
          // The exact record itself must satisfy the OVRTX evidence
          // predicate (renderer allowlist, metadata, digests, remote
          // attestation), and its sanitized metadata must agree with the
          // displayed projection — a digest-valid sidecar with divergent
          // or missing metadata must not vouch for the projection.
          if (referenceProvenanceSummary(exactRecord) === null) return;
          if (
            entry.reference.render_metadata_fingerprint &&
            sanitizedMetadataFingerprint(exactRecord) !==
              entry.reference.render_metadata_fingerprint
          ) {
            return;
          }
        }
        entry.reference.diagnostic = false;
        entry.reference.provenance = entry.provenance;
        const verifiedPath = result.blobUrl ?? entry.path;
        entry.reference.path = verifiedPath;
        if (entry.wasPreview) {
          entry.asset.preview.image_url = verifiedPath;
        }
      }),
    );
  };
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is string => typeof item === "string" && Boolean(item),
      )
    : [];
}

function simreadyValidationForAsset(
  asset: BenchmarkBundleAsset,
  scoredCase: SuiteCase | undefined,
): AssetSimReadyValidation | undefined {
  const signal = scoredCase?.signals?.find(
    (candidate) => candidate.name === "simready_profile_validation",
  );
  const signalValue =
    signal?.value && typeof signal.value === "object"
      ? (signal.value as Record<string, unknown>)
      : undefined;
  const bundleValue =
    asset.simready_validation && typeof asset.simready_validation === "object"
      ? (asset.simready_validation as Record<string, unknown>)
      : undefined;
  const value = signalValue ?? bundleValue;
  if (!value) return undefined;

  const profile =
    typeof value.profile === "string" ? value.profile : "unknown";
  const profileVersion =
    typeof value.profile_version === "string"
      ? value.profile_version
      : "unknown";
  const profileTarget =
    typeof value.profile_target === "string"
      ? value.profile_target
      : `${profile}@${profileVersion}`;
  const failedFeatures = Array.isArray(value.failed_features)
    ? value.failed_features.flatMap((candidate) => {
        if (!candidate || typeof candidate !== "object") return [];
        const feature = candidate as Record<string, unknown>;
        if (typeof feature.feature_id !== "string" || !feature.feature_id)
          return [];
        return [
          {
            feature_id: feature.feature_id,
            requirements: stringArray(feature.requirements),
            messages: stringArray(feature.messages),
          },
        ];
      })
    : [];
  return {
    profile,
    profile_version: profileVersion,
    profile_target: profileTarget,
    status:
      typeof value.status === "string"
        ? value.status.toUpperCase()
        : signal?.ok
          ? "PASS"
          : signal
            ? "FAIL"
            : "UNAVAILABLE",
    passed:
      typeof value.passed === "boolean"
        ? value.passed
        : signal
          ? signal.ok
          : false,
    evidence_verified:
      typeof value.evidence_verified === "boolean"
        ? value.evidence_verified
        : signal
          ? true
          : null,
    failed_features: failedFeatures,
    failed_requirements: stringArray(value.failed_requirements),
    errors: stringArray(value.errors),
  };
}

function evidenceForAsset(
  entry: BenchmarkIndexRun,
  asset: BenchmarkBundleAsset,
): { references: EvidenceItem[]; renders: EvidenceItem[] } {
  // The OVRTX provenance requirement applies to rendered-geometry evidence,
  // which only the physics workflows bundle as references. Material and
  // mesh-segmentation references are dataset/source images with a label and
  // path only — flagging them diagnostic would mislabel legitimate source
  // references as "not evidence".
  const requireProvenance = PHYSICS_WORKFLOWS.has(entry.workflow);
  const evidenceMediaType = (path: string): "image" | "video" =>
    /\.(?:mp4|mov|m4v|webm)(?:$|[?#])/i.test(path) ? "video" : "image";
  const references = (asset.references ?? []).flatMap((reference, index) => {
    const path = localArtifactUrl(entry, reference.path);
    if (!path) return [];
    const base = {
      label: reference.label ?? `reference ${index + 1}`,
      view: reference.label ?? `reference ${index + 1}`,
      path,
      kind: "reference" as const,
      media_type: evidenceMediaType(path),
    };
    if (!requireProvenance) return [base];
    const provenance = referenceProvenanceSummary(reference.provenance);
    const expectedDigest =
      provenance !== null
        ? String(
            (reference.provenance as Record<string, unknown>).image_sha256,
          ).toLowerCase()
        : null;
    // An exported record is a projection; the exact OVRTX metadata lives in
    // the bound sidecar named by exact_record_path. Carry the binding so the
    // verifier can require the sidecar's bytes to hash to
    // exact_record_sha256 before promoting the reference.
    const record =
      provenance !== null
        ? (reference.provenance as Record<string, unknown>)
        : null;
    const exactRecordSha =
      record && isSha256Digest(record.exact_record_sha256)
        ? String(record.exact_record_sha256).toLowerCase()
        : null;
    const exactRecordUrl =
      record && typeof record.exact_record_path === "string"
        ? localArtifactUrl(entry, record.exact_record_path)
        : null;
    // Fail closed on a corrupted binding: a record that carries either half
    // of the sidecar binding but not a well-formed pair must not skip the
    // sidecar check and promote on the image digest alone.
    const brokenExactBinding =
      record !== null &&
      (record.exact_record_sha256 !== undefined ||
        record.exact_record_path !== undefined) &&
      (exactRecordSha === null || exactRecordUrl === null);
    if (brokenExactBinding) {
      return [{ ...base, provenance: null, diagnostic: true }];
    }
    // A projected record must carry BOTH binding fields; one without the
    // other is unverifiable and can never be promoted.
    const hasExactBinding =
      record !== null &&
      (record.exact_record_sha256 !== undefined ||
        record.exact_record_path !== undefined);
    const exactRecordBroken =
      hasExactBinding && (exactRecordSha === null || exactRecordUrl === null);
    return [
      {
        ...base,
        provenance,
        diagnostic: provenance === null,
        image_sha256: expectedDigest,
        renderer: record ? String(record.renderer ?? "") : undefined,
        render_metadata_fingerprint: record
          ? sanitizedMetadataFingerprint(record)
          : null,
        source_usd_sha256:
          record && isSha256Digest(record.source_usd_sha256)
            ? String(record.source_usd_sha256).toLowerCase()
            : null,
        exact_record_sha256: exactRecordSha,
        exact_record_url: exactRecordUrl,
        exact_record_broken: exactRecordBroken,
      },
    ];
  });
  const renderOrder = [
    "final",
    "oblique",
    "turntable",
    "pos_y",
    "pos_x",
    "pos_z",
    "neg_z",
  ];
  const renderRank = (view: string) => {
    if (view === "before") return Number.MAX_SAFE_INTEGER;
    const rank = renderOrder.indexOf(view);
    return rank === -1 ? renderOrder.length : rank;
  };
  const renderEntries = Object.entries(asset.renders ?? {}).sort(
    ([left], [right]) => renderRank(left) - renderRank(right),
  );
  const renders = renderEntries.flatMap(([view, artifactPath]) => {
    const path = localArtifactUrl(entry, artifactPath);
    if (!path) return [];
    return [
      {
        label: titleCase(view),
        view: titleCase(view),
        path,
        kind: "render" as const,
        media_type: evidenceMediaType(path),
      },
    ];
  });
  return { references, renders };
}

function reportsForAsset(
  entry: BenchmarkIndexRun,
  asset: BenchmarkBundleAsset,
): { reports: ReportLink[]; traces: ReportLink[] } {
  const reports: ReportLink[] = [
    {
      label: "Scoring result",
      path: entry.score_url
        ? versionedGeometryArtifactUrl(entry, entry.score_url)
        : runArtifactUrl(entry, "bundle/run.json"),
      kind: entry.score_url ? "suite_result" : "run_bundle",
    },
  ];
  if (!isGeometryWorkflow(entry.workflow)) {
    reports.push({
      label: "Asset metrics",
      // The exporter sanitizes the directory name but keeps `asset_id`
      // verbatim, so an id containing anything outside [A-Za-z0-9._-] lands
      // somewhere this cannot guess. It records the real location.
      path: runArtifactUrl(
        entry,
        `bundle/${asset.export_root ?? `assets/${asset.asset_id}`}/metrics.json`,
      ),
      kind: "metrics",
    });
  }
  if (asset.review_path) {
    const reviewUrl = localArtifactUrl(entry, asset.review_path);
    if (reviewUrl) {
      reports.push({
        label: "Visual review",
        path: reviewUrl,
        kind: "review",
      });
    }
  }
  if (asset.source_usd) {
    const sourceUrl = localArtifactUrl(entry, asset.source_usd);
    if (sourceUrl) {
      reports.push({
        label: "Source USD",
        path: sourceUrl,
        kind: "source_usd",
      });
    }
  }
  if (asset.output_usd) {
    const outputUrl = localArtifactUrl(entry, asset.output_usd);
    if (outputUrl) {
      reports.push({
        label: "Output USD",
        path: outputUrl,
        kind: "output_usd",
      });
    }
  }
  for (const artifact of asset.artifact_links ?? []) {
    if (!artifact.uri) continue;
    const artifactUrl = localArtifactUrl(entry, artifact.uri);
    if (!artifactUrl) continue;
    reports.push({
      label: artifact.label ?? titleCase(artifact.kind ?? "artifact"),
      path: artifactUrl,
      kind: artifact.kind ?? "artifact",
      sha256: artifact.sha256,
      media_type: artifact.media_type,
    });
  }

  const traces = Object.entries(asset.local_artifacts ?? {}).flatMap(
    ([label, path]) => {
      const url = localArtifactUrl(entry, path);
      if (!url || label === "run_dir") return [];
      return [{ label: titleCase(label), path: url, kind: "workflow" }];
    },
  );
  const uniqueReports = [
    ...new Map(reports.map((report) => [report.path, report])).values(),
  ];
  return { reports: uniqueReports, traces };
}

function findingsForCase(
  scoredCase: SuiteCase | undefined,
  evaluation: MaterialEvaluationAsset | undefined,
): Finding[] {
  const findings = (scoredCase?.signals ?? [])
    .filter((signal) => !signal.ok)
    .map((signal) => ({
      severity: findingSeverity(signal.severity),
      title: titleCase(signal.name),
      detail:
        signal.detail ??
        (signal.value === undefined
          ? "Benchmark signal did not pass."
          : String(signal.value)),
    }));
  if (scoredCase?.error) {
    findings.unshift({
      severity: "blocking",
      title: "Workflow Error",
      detail: scoredCase.error,
    });
  }
  if (evaluation && evaluation.status !== "evaluated") {
    findings.push({
      severity: "warning",
      title:
        evaluation.status === "error"
          ? "Material Evaluation Error"
          : "Material Evaluation Incomplete",
      detail:
        evaluation.rationale ||
        `Material evaluation status: ${titleCase(evaluation.status)}`,
    });
  }
  for (const issue of evaluation?.issues ?? []) {
    findings.push({
      severity: "minor",
      title: "Judge Finding",
      detail: issue,
    });
  }
  for (const issue of evaluation?.semantic_consistency_findings ?? []) {
    findings.push({
      severity: "minor",
      title: "Semantic Consistency",
      detail: issue,
    });
  }
  return findings;
}

function findingSeverity(value: string): ReviewSeverity {
  const severity = value.toLowerCase();
  if (["blocking", "critical", "error"].includes(severity)) return "blocking";
  if (severity === "major") return "major";
  if (severity === "warning") return "warning";
  if (["none", "minor"].includes(severity)) return severity as ReviewSeverity;
  return "warning";
}

const REVIEW_VERDICTS = new Set<ReviewVerdict>([
  "unreviewed",
  "pass",
  "issue",
  "regression",
  "dataset_issue",
  "unclear",
]);

function persistedAssetReview(document: PersistedReviewDocument | undefined): {
  review: AssetReview;
  feedback: AssetFeedback[];
} {
  const verdict = document?.verdict;
  const severity = document?.severity;
  const score = document?.score;
  const feedback = Array.isArray(document?.feedback)
    ? document.feedback.filter(
        (entry) =>
          entry &&
          typeof entry.id === "string" &&
          typeof entry.reviewer === "string" &&
          typeof entry.comment === "string" &&
          typeof entry.created_at === "string",
      )
    : [];
  return {
    feedback,
    review: {
      reviewer: typeof document?.reviewer === "string" ? document.reviewer : "",
      verdict:
        typeof verdict === "string" &&
        REVIEW_VERDICTS.has(verdict as ReviewVerdict)
          ? (verdict as ReviewVerdict)
          : "unreviewed",
      severity:
        typeof severity === "string" ? findingSeverity(severity) : "none",
      score: typeof score === "number" && Number.isFinite(score) ? score : null,
      tags: Array.isArray(document?.tags)
        ? document.tags.filter((tag): tag is string => typeof tag === "string")
        : [],
      comment: typeof document?.comment === "string" ? document.comment : "",
      needs_rerun: document?.needs_rerun === true,
      updated_at:
        typeof document?.updated_at === "string" ? document.updated_at : null,
    },
  };
}

function adaptAsset(
  entry: BenchmarkIndexRun,
  asset: BenchmarkBundleAsset,
  scoredCase: SuiteCase | undefined,
  evaluationResult: MaterialEvaluationAsset | undefined,
  persistedReview: PersistedReviewDocument | undefined,
): AssetResult {
  const status = resultStatus(scoredCase?.status);
  const evidence = evidenceForAsset(entry, asset);
  const links = reportsForAsset(entry, asset);
  const isGeometry = isGeometryWorkflow(entry.workflow);
  const geometry =
    isGeometry && asset.geometry
      ? adaptGeometryAsset({
          asset,
          scoredCase,
          resolveArtifact: (path) => localArtifactUrl(entry, path),
        })
      : null;
  const tags = scoredCase?.tags ?? asset.tags ?? [];
  const workflowStatus =
    geometry?.workflowStatus ??
    (asset.status ? titleCase(asset.status) : "Not reported");
  const evaluation = assetMaterialEvaluation(evaluationResult);
  // Key naming on the run's workflow, not the per-case one. `scoredCase` is
  // optional, so a bundle asset without a matching scored case would otherwise
  // be presented as a Material asset inside a mesh-segmentation run.
  const isMeshSegmentation =
    entry.workflow === "mesh_segmentation" ||
    scoredCase?.workflow === "mesh_segmentation";
  const meshEvaluation = meshSegmentationEvaluation(
    scoredCase,
    isMeshSegmentation,
  );
  const isPhysics =
    PHYSICS_WORKFLOWS.has(entry.workflow) ||
    PHYSICS_WORKFLOWS.has(scoredCase?.workflow ?? "");
  const physicsEval = physicsEvaluation(scoredCase, isPhysics);
  const review = persistedAssetReview(persistedReview);
  // The physics scorer fills thumbnails from bundle references regardless of
  // provenance, and even for a validated reference the scorer's thumbnail is
  // a Pillow re-encode — not the provenance-bound bytes. Fail closed: a
  // physics preview uses the validated reference image itself (its digests
  // describe exactly those bytes), and shows nothing when no reference
  // validates, so an image the evidence grid labels "Diagnostic preview"
  // cannot reappear unlabeled on the asset card or hero.
  const validatedReference = isPhysics
    ? evidence.references.find((reference) => reference.diagnostic === false)
    : undefined;
  const scoredThumbnail = isPhysics ? undefined : scoredCase?.thumbnails?.[0];
  const fallbackThumbnail = scoredThumbnail
    ? scoreThumbnailUrl(entry, scoredThumbnail)
    : null;
  const reportsByPath = new Map<string, ReportLink>();
  for (const report of [...links.reports, ...(geometry?.reports ?? [])]) {
    const existing = reportsByPath.get(report.path);
    reportsByPath.set(
      report.path,
      existing ? { ...existing, ...report } : report,
    );
  }
  const reports = [...reportsByPath.values()];
  const renders = geometry?.renders ?? evidence.renders;
  const previewImage = geometry
    ? renders[0]?.path
    : (renders.find((item) => item.media_type === "image")?.path ??
      validatedReference?.path ??
      fallbackThumbnail ??
      undefined);

  return {
    asset_id: asset.asset_id,
    name: isMeshSegmentation
      ? meshSegmentationAssetName(asset.asset_id, tags)
      : titleCase(asset.name ?? asset.asset_id),
    prompt:
      typeof asset.prompt === "string" && asset.prompt.trim()
        ? asset.prompt.trim()
        : null,
    status,
    status_label: `${STATUS_LABELS[status]} · ${workflowStatus}`,
    family: isMeshSegmentation
      ? "PartObjaverse"
      : isPhysics
        ? "PhysX-Mobility"
        : (geometry?.family ??
          (tags[0]
            ? titleCase(tags[0])
            : isGeometry
              ? "Geometry"
              : "Material")),
    tags,
    source_usd: asset.source_usd ?? "",
    output_usd: asset.output_usd ?? "",
    references: evidence.references,
    renders,
    reports,
    trace_links: links.traces,
    metrics: geometry?.metrics ?? displayMetrics(asset, scoredCase, evaluation),
    simready_validation: simreadyValidationForAsset(asset, scoredCase),
    findings: [
      ...findingsForCase(scoredCase, evaluationResult),
      ...(geometry?.findings ?? []),
    ],
    material_evaluation: evaluation,
    mesh_segmentation_evaluation: meshEvaluation,
    physics_evaluation: physicsEval,
    feedback: review.feedback,
    review: review.review,
    preview: {
      variant: "generic",
      tone: STATUS_TONES[status],
      image_url: previewImage,
    },
  };
}

function adaptScoreOnlyCase(
  entry: BenchmarkIndexRun,
  scoredCase: SuiteCase,
  evaluationResult: MaterialEvaluationAsset | undefined,
) {
  const asset = adaptAsset(
    entry,
    {
      asset_id: scoredCase.asset_id,
      name: scoredCase.asset_id,
      status: "not_collected",
      tags: scoredCase.tags,
    },
    scoredCase,
    evaluationResult,
    undefined,
  );
  return {
    ...asset,
    reports: [
      {
        label: "Scoring result",
        path: entry.score_url
          ? versionedGeometryArtifactUrl(entry, entry.score_url)
          : runArtifactUrl(entry, "bundle/run.json"),
        kind: entry.score_url ? "suite_result" : "run_bundle",
      },
    ],
    trace_links: [],
  };
}

function signalSummary(cases: SuiteCase[]) {
  const summaries: Record<string, SignalSummary> = {};
  for (const scoredCase of cases) {
    for (const signal of scoredCase.signals ?? []) {
      const summary = summaries[signal.name] ?? { passed: 0, total: 0 };
      summary.total += 1;
      if (signal.ok) summary.passed += 1;
      summaries[signal.name] = summary;
    }
  }
  return summaries;
}

function countStatus(assets: AssetResult[], status: ResultStatus) {
  return assets.filter((asset) => asset.status === status).length;
}

async function loadRun(entry: BenchmarkIndexRun): Promise<BenchmarkRun> {
  const [manifest, bundle, score, evaluation] = await Promise.all([
    fetchJson<AgenticRunManifest>(
      versionedGeometryArtifactUrl(entry, entry.manifest_url),
    ),
    fetchJson<BenchmarkRunBundle>(
      versionedGeometryArtifactUrl(entry, entry.bundle_url),
    ),
    entry.score_url
      ? fetchJson<SuiteResult>(
          versionedGeometryArtifactUrl(entry, entry.score_url),
        )
      : Promise.resolve(null),
    entry.evaluation_url
      ? fetchJson<MaterialEvaluationDocument>(
          versionedGeometryArtifactUrl(entry, entry.evaluation_url),
        )
      : Promise.resolve(null),
  ]);

  if (
    manifest.run_id !== bundle.run_id ||
    (score && score.run_id !== bundle.run_id)
  ) {
    throw new Error(
      `Run ID mismatch in benchmark artifacts for ${entry.run_id}`,
    );
  }
  // The entry workflow is canonicalized at the index boundary; a historical
  // manifest still declares the aliased id, which is agreement, not a
  // mismatch.
  if (canonicalWorkflow(manifest.workflow) !== entry.workflow) {
    throw new Error(
      `Workflow mismatch in benchmark artifacts for ${entry.run_id}`,
    );
  }
  if (isGeometryIntegrityWorkflow(entry.workflow)) {
    validateGeometryRunContract({ entry, manifest, bundle, score, evaluation });
  }
  if (
    evaluation &&
    (evaluation.run_id !== bundle.run_id ||
      evaluation.workflow !== entry.workflow)
  ) {
    throw new Error(
      `Evaluation mismatch in benchmark artifacts for ${entry.run_id}`,
    );
  }

  // `adaptAsset` classifies and evaluates by `entry.workflow`, so artifacts
  // belonging to a different workflow must not reach it just because the run
  // IDs line up. Compare only when a document actually declares a workflow --
  // an omitted field is missing provenance, not a mismatch, and must not make
  // an otherwise loadable run unreadable.
  const declaredWorkflows: Array<string | undefined> = [
    bundle.workflow,
    ...(score?.cases ?? []).map((item) => item.workflow),
  ];
  if (
    declaredWorkflows.some(
      (declared) =>
        declared !== undefined &&
        canonicalWorkflow(declared) !== entry.workflow,
    )
  ) {
    throw new Error(
      `Workflow mismatch in benchmark artifacts for ${entry.run_id}`,
    );
  }

  const workflow = workflowMetadata(entry.workflow);
  if (!workflow) throw new Error(`Unsupported workflow: ${entry.workflow}`);
  const casesByAsset = new Map(
    (score?.cases ?? []).map((item) => [item.asset_id, item]),
  );
  const evaluationByAsset = new Map(
    (evaluation?.assets ?? []).map((item) => [item.asset_id, item]),
  );
  const persistedReviews = await Promise.all(
    bundle.assets.flatMap((asset) => {
      if (!asset.review_path) return [];
      const reviewUrl = localArtifactUrl(entry, asset.review_path);
      if (!reviewUrl) {
        throw new Error(`Unsafe review path for ${asset.asset_id}`);
      }
      return [
        fetchJson<PersistedReviewDocument>(reviewUrl).then((document) => {
          if (document.asset_id && document.asset_id !== asset.asset_id) {
            throw new Error(`Review asset mismatch for ${asset.asset_id}`);
          }
          return [asset.asset_id, document] as const;
        }),
      ];
    }),
  );
  const reviewsByAsset = new Map(persistedReviews);
  const bundleAssetIds = new Set(bundle.assets.map((asset) => asset.asset_id));
  const assets = [
    ...bundle.assets.map((asset) =>
      adaptAsset(
        entry,
        asset,
        casesByAsset.get(asset.asset_id),
        evaluationByAsset.get(asset.asset_id),
        reviewsByAsset.get(asset.asset_id),
      ),
    ),
    ...(score?.cases ?? [])
      .filter((scoredCase) => !bundleAssetIds.has(scoredCase.asset_id))
      .map((scoredCase) =>
        adaptScoreOnlyCase(
          entry,
          scoredCase,
          evaluationByAsset.get(scoredCase.asset_id),
        ),
      ),
  ];
  const verifyReferences = deferValidatedReferenceVerification(assets);
  const runtimes = assets
    .map((asset) => numericMetric(asset.metrics, "runtime_seconds"))
    .filter((runtime) => runtime > 0);
  const totalTokens = assets.reduce(
    (total, asset) => total + numericMetric(asset.metrics, "total_tokens"),
    0,
  );
  const summedAssetMetric = (name: string) => {
    const values = assets
      .map((asset) => optionalNumericMetric(asset.metrics, name))
      .filter((value): value is number => value !== null);
    return values.length
      ? values.reduce((total, value) => total + value, 0)
      : null;
  };
  const summaryMetrics = bundle.summary?.metrics;
  const aggregateMetric = (name: string) =>
    optionalNumericMetric(summaryMetrics, name) ?? summedAssetMetric(name);
  // When per-asset token sums drive the run totals, a session that died
  // before recording token stats silently drops out of the sum. Surface the
  // measured-asset coverage so a partial total is labeled, mirroring the
  // standalone report. Only meaningful when the displayed values actually
  // came from asset sums — a run with complete run-level usage is complete
  // regardless of per-asset coverage (computed below once the run-level
  // sources are known). The numerator counts the same per-asset signals the
  // displayed sums are built from (input/output tokens), not the synthesized
  // total_tokens, so the coverage note and the sum cannot disagree.
  const tokenMeasuredAssets = assets.filter(
    (asset) =>
      optionalNumericMetric(asset.metrics, "input_tokens") !== null ||
      optionalNumericMetric(asset.metrics, "output_tokens") !== null,
  ).length;
  const execution = manifest.metadata?.execution ?? null;
  const usage = execution?.usage ?? null;
  const runLevelInputTokens =
    usage?.input_tokens ??
    optionalNumericMetric(summaryMetrics, "input_tokens") ??
    null;
  const inputTokens = runLevelInputTokens ?? summedAssetMetric("input_tokens");
  const cachedInputTokens =
    usage?.cached_input_tokens ??
    optionalNumericMetric(summaryMetrics, "cached_input_tokens") ??
    summedAssetMetric("cached_input_tokens");
  const runLevelOutputTokens =
    usage?.output_tokens ??
    optionalNumericMetric(summaryMetrics, "output_tokens") ??
    null;
  const outputTokens =
    runLevelOutputTokens ?? summedAssetMetric("output_tokens");
  // Physics bundles' "run-level" summary metrics are themselves per-asset
  // sums (both adapters aggregate asset metrics), so partial per-asset
  // coverage must be labeled for them even when run-level values exist.
  // Other workflows carry genuinely independent run-level usage; for those
  // the label applies only when the display fell through to asset sums.
  const tokenSumsAreAssetDerived =
    PHYSICS_WORKFLOWS.has(entry.workflow) ||
    (runLevelInputTokens === null && runLevelOutputTokens === null);
  const tokenAssetCoverage =
    tokenSumsAreAssetDerived &&
    tokenMeasuredAssets > 0 &&
    tokenMeasuredAssets < assets.length
      ? `${tokenMeasuredAssets}/${assets.length}`
      : null;
  const reasoningOutputTokens =
    usage?.reasoning_output_tokens ??
    optionalNumericMetric(summaryMetrics, "reasoning_output_tokens") ??
    summedAssetMetric("reasoning_output_tokens");
  const coherentTotalTokens =
    inputTokens !== null && outputTokens !== null
      ? inputTokens + outputTokens
      : (usage?.total_tokens ?? totalTokens);
  const git = bundle.git ?? manifest.metadata?.git ?? {};
  const commonConfig = bundle.summary?.common_config ?? {};
  const bundleConfig = bundle.config ?? {};
  // Physics-fixed bundles record their VLM at config.options.vlm_model and
  // their execution mode at summary.execution_mode rather than the material
  // bundle's common_config, so read both shapes.
  const bundleOptions =
    bundleConfig.options && typeof bundleConfig.options === "object"
      ? (bundleConfig.options as Record<string, unknown>)
      : {};
  const recordedModels = Array.isArray(bundleConfig.models_used)
    ? bundleConfig.models_used.filter(
        (value): value is string => typeof value === "string" && Boolean(value),
      )
    : [];
  const runner =
    (typeof commonConfig.runner === "string" && commonConfig.runner) ||
    (typeof bundleConfig.runner === "string" && bundleConfig.runner) ||
    (typeof bundle.summary?.execution_mode === "string" &&
      bundle.summary.execution_mode) ||
    bundle.source ||
    "unknown";
  const model =
    (typeof commonConfig.model === "string" && commonConfig.model) ||
    (typeof commonConfig.vlm_model === "string" && commonConfig.vlm_model) ||
    (typeof bundleConfig.model === "string" && bundleConfig.model) ||
    (typeof bundleOptions.vlm_model === "string" && bundleOptions.vlm_model) ||
    (recordedModels.length ? recordedModels.join(", ") : null) ||
    manifest.metadata?.vlm?.model ||
    null;
  const visionModel =
    (typeof commonConfig.vision_model === "string" &&
      commonConfig.vision_model) ||
    (typeof bundleConfig.vision_model === "string" &&
      bundleConfig.vision_model) ||
    null;
  const visionBackend =
    (typeof commonConfig.vision_backend === "string" &&
      commonConfig.vision_backend) ||
    (typeof bundleConfig.vision_backend === "string" &&
      bundleConfig.vision_backend) ||
    null;
  const provenance =
    bundle.config?.provenance ?? manifest.metadata?.provenance ?? null;
  const reviewedAssetCount = assets.filter(
    (asset) =>
      Boolean(asset.review.reviewer.trim() && asset.review.comment.trim()) ||
      Boolean(asset.feedback?.length),
  ).length;
  const meshEvaluation =
    entry.workflow === "mesh_segmentation"
      ? meshSegmentationSummary(assets)
      : null;
  const physicsEvaluationSummary = PHYSICS_WORKFLOWS.has(entry.workflow)
    ? physicsSummary(assets)
    : null;
  const simreadyProfileCounts = new Map<
    string,
    { passed: number; failed: number; unverified: number; total: number }
  >();
  for (const asset of assets) {
    const validation = asset.simready_validation;
    if (!validation) continue;
    const counts = simreadyProfileCounts.get(validation.profile_target) ?? {
      passed: 0,
      failed: 0,
      unverified: 0,
      total: 0,
    };
    counts.total += 1;
    if (validation.evidence_verified !== true) counts.unverified += 1;
    else if (validation.passed) counts.passed += 1;
    else counts.failed += 1;
    simreadyProfileCounts.set(validation.profile_target, counts);
  }
  if (
    entry.workflow === "cad-to-simready" &&
    simreadyProfileCounts.size === 0 &&
    typeof bundleConfig.profile === "string" &&
    typeof bundleConfig.profile_version === "string"
  ) {
    simreadyProfileCounts.set(
      `${bundleConfig.profile}@${bundleConfig.profile_version}`,
      { passed: 0, failed: 0, unverified: 0, total: 0 },
    );
  }
  const simreadyProfiles = [...simreadyProfileCounts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([profileTarget, counts]) => ({
      profile_target: profileTarget,
      ...counts,
    }));

  return {
    run_id: bundle.run_id,
    workflow_id: workflow.id,
    verify_references: verifyReferences,
    label: displayRunLabel(bundle, workflow.name),
    suite_id: entry.workflow,
    created_at: manifest.created_at || bundle.created_at,
    commit: git.commit ?? "unknown",
    branch: git.branch ?? "unknown",
    dirty: git.dirty === true || git.dirty === "true",
    scored: score !== null,
    model,
    vision_model: visionModel,
    vision_backend: visionBackend,
    runner,
    run_status: manifest.status,
    source: bundle.source ?? "unknown",
    artifact_root: entry.artifact_base_url,
    provenance,
    execution,
    material_evaluation: evaluation
      ? {
          evaluated_asset_count: evaluation.aggregate.evaluated_asset_count,
          asset_count: evaluation.aggregate.asset_count,
          mean_scores: materialAgreementScores(
            evaluation.aggregate.mean_scores,
          ),
          mean_agreement_scores: materialAgreementScores(
            evaluation.aggregate.mean_agreement_scores,
          ),
          acceptable_rate: evaluation.aggregate.acceptable_rate,
          high_fidelity_rate: evaluation.aggregate.high_fidelity_rate,
          mean_judge_confidence: evaluation.aggregate.mean_judge_confidence,
          mean_assignment_coverage_score:
            evaluation.aggregate.mean_assignment_coverage_score,
          authoring_coverage_score:
            evaluation.aggregate.authoring_coverage_score ??
            evaluation.aggregate.mean_assignment_coverage_score,
          rejected_assignment_rate:
            evaluation.aggregate.rejected_assignment_rate ?? null,
          missing_assignment_rate:
            evaluation.aggregate.missing_assignment_rate ?? null,
          final_output_asset_count:
            evaluation.aggregate.final_output_asset_count ?? 0,
          final_visible_mesh_count:
            evaluation.aggregate.final_visible_mesh_count ?? null,
          final_bound_visible_mesh_count:
            evaluation.aggregate.final_bound_visible_mesh_count ?? null,
          final_partially_bound_visible_mesh_count:
            evaluation.aggregate.final_partially_bound_visible_mesh_count ??
            null,
          final_unbound_visible_mesh_count:
            evaluation.aggregate.final_unbound_visible_mesh_count ?? null,
          final_mesh_binding_coverage_score:
            evaluation.aggregate.final_mesh_binding_coverage_score ?? null,
          judges: evaluation.judges,
          prompt_version: evaluation.method.prompt_version,
          caveat: evaluation.method.caveat,
        }
      : null,
    mesh_segmentation_evaluation: meshEvaluation,
    physics_evaluation: physicsEvaluationSummary,
    simready_profiles: simreadyProfiles,
    summary: {
      total: assets.length,
      pass: countStatus(assets, "pass"),
      warn: countStatus(assets, "warn"),
      fail: countStatus(assets, "fail"),
      error: countStatus(assets, "error"),
      skipped: countStatus(assets, "skipped"),
      unscored: countStatus(assets, "unknown"),
      reviewed: reviewedAssetCount,
      needs_review: assets.length - reviewedAssetCount,
      median_runtime_seconds: median(runtimes),
      token_asset_coverage: tokenAssetCoverage,
      input_tokens: inputTokens,
      cached_input_tokens: cachedInputTokens,
      output_tokens: outputTokens,
      reasoning_output_tokens: reasoningOutputTokens,
      driver_input_tokens: aggregateMetric("driver_input_tokens"),
      driver_cached_input_tokens: aggregateMetric("driver_cached_input_tokens"),
      driver_output_tokens: aggregateMetric("driver_output_tokens"),
      driver_total_tokens: aggregateMetric("driver_total_tokens"),
      vision_input_tokens: aggregateMetric("vision_input_tokens"),
      vision_cached_input_tokens: aggregateMetric("vision_cached_input_tokens"),
      vision_output_tokens: aggregateMetric("vision_output_tokens"),
      vision_total_tokens: aggregateMetric("vision_total_tokens"),
      vision_invocation_count: aggregateMetric("vision_invocation_count"),
      combined_total_tokens: aggregateMetric("combined_total_tokens"),
      total_tokens: coherentTotalTokens,
      cost_estimate: usage?.cost_estimate ?? null,
      cost_estimate_status:
        usage?.cost_estimate_status ??
        (usage?.cost_estimate
          ? "estimated"
          : model
            ? "model_not_in_pricing_table"
            : "model_unreported"),
      cost_estimate_scope: usage?.cost_estimate_scope,
      signals: signalSummary(score?.cases ?? []),
    },
    assets,
  };
}

export async function loadBenchmarkBundle(
  indexUrl: string,
): Promise<BenchmarkBundle> {
  const index = await loadBenchmarkIndex(indexUrl);

  // Canonicalize historical workflow ids ("physics", "physics_agentic", ...)
  // once at the boundary: every downstream gate — supported-workflow
  // filtering, PHYSICS_WORKFLOWS evidence gating, declared-workflow checks —
  // keys on the canonical id, and a historical export must not silently
  // vanish or bypass the physics provenance gates.
  const entries = index.runs.map((entry) => ({
    ...entry,
    workflow: canonicalWorkflow(entry.workflow),
  }));
  const supportedEntries = entries.filter((entry) =>
    workflowMetadata(entry.workflow),
  );
  const settledRuns = await Promise.allSettled(supportedEntries.map(loadRun));
  const runs = settledRuns.flatMap((result) =>
    result.status === "fulfilled" ? [result.value] : [],
  );
  const failures = settledRuns.flatMap((result) =>
    result.status === "rejected" ? [String(result.reason)] : [],
  );
  if (!runs.length) {
    const detail = failures.length ? ` ${failures.join("; ")}` : "";
    throw new Error(`No supported benchmark runs were found.${detail}`);
  }
  if (failures.length)
    console.warn("Some benchmark runs could not be loaded", failures);

  const workflows = Object.values(WORKFLOW_METADATA).flatMap((metadata) => {
    const runIds = runs
      .filter((run) => run.workflow_id === metadata.id)
      .sort(compareRunsNewestFirst)
      .map((run) => run.run_id);
    return runIds.length ? [{ ...metadata, run_ids: runIds }] : [];
  });

  return {
    schema_version: index.schema_version,
    generated_at: index.generated_at,
    dashboard_id: "agentic/benchmark",
    // Do not accept the complete-index fingerprint for a partial load. An empty
    // value makes the refresh loop retry omitted runs even when the index itself
    // has not changed.
    index_fingerprint: failures.length ? "" : indexFingerprint(index),
    workflows,
    runs,
    // References start demoted (fail closed); awaiting this upgrades every
    // byte-verified reference in place. Callers re-render afterwards.
    verify_references: async () => {
      await Promise.all(runs.map((run) => run.verify_references?.()));
    },
  };
}
