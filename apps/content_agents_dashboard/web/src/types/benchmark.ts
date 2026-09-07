// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type WorkflowId =
  | "material-agentic"
  | "material-fixed"
  | "mesh-segmentation"
  | "physics-fixed"
  | "physics-agentic"
  | "cad-benchmark"
  | "cad-to-simready";

export type ResultStatus =
  "pass" | "warn" | "fail" | "error" | "skipped" | "unknown";

export type ReviewVerdict =
  "unreviewed" | "pass" | "issue" | "regression" | "dataset_issue" | "unclear";

export type ReviewSeverity =
  "none" | "minor" | "warning" | "major" | "blocking";

export interface BenchmarkBundle {
  schema_version: string;
  generated_at: string;
  dashboard_id: string;
  index_fingerprint: string;
  workflows: WorkflowSummary[];
  runs: BenchmarkRun[];
  // References start demoted (fail closed); awaiting this upgrades every
  // byte-verified reference in place. Callers re-render afterwards.
  verify_references?: () => Promise<void>;
}

export interface WorkflowSummary {
  id: WorkflowId;
  name: string;
  description: string;
  run_ids: string[];
}

export interface BenchmarkRun {
  run_id: string;
  workflow_id: WorkflowId;
  verify_references?: () => Promise<void>;
  label: string;
  suite_id: string;
  created_at: string;
  commit: string;
  branch: string;
  dirty: boolean;
  scored: boolean;
  model: string | null;
  vision_model: string | null;
  vision_backend: string | null;
  runner: string;
  run_status: string;
  source: string;
  artifact_root: string;
  provenance: BenchmarkProvenance | null;
  execution: RunExecutionMetadata | null;
  material_evaluation: MaterialEvaluationSummary | null;
  mesh_segmentation_evaluation: MeshSegmentationEvaluationSummary | null;
  physics_evaluation: PhysicsEvaluationSummary | null;
  simready_profiles?: SimReadyProfileSummary[];
  summary: RunSummary;
  assets: AssetResult[];
}

// Per-prim ground-truth comparison row emitted by the physics scorer as
// `property_rows` in CaseResult.metrics. Both physics-fixed and
// physics-agentic share the scorer, so one shape covers both workflows.
export interface PhysicsPartRow {
  prim_path: string;
  gt_part: string | null;
  predicted_material: string | null;
  gt_material: string | null;
  material_match: boolean | null;
  predicted_density: number | null;
  gt_density: number | null;
  density_log_ratio: number | null;
  predicted_static_friction: number | null;
  gt_static_friction: number | null;
  predicted_dynamic_friction: number | null;
  gt_dynamic_friction: number | null;
  friction_abs_error: number | null;
  predicted_restitution: number | null;
  gt_restitution: number | null;
  restitution_abs_error: number | null;
  predicted_mass_kg: number | null;
}

export interface AssetPhysicsEvaluation {
  material_accuracy: number | null;
  gt_parts_total: number | null;
  gt_parts_covered: number | null;
  density_log_mae: number | null;
  density_median_error_factor: number | null;
  friction_mae: number | null;
  restitution_mae: number | null;
  predictions_count: number | null;
  zero_mass_prims: number | null;
  parts: PhysicsPartRow[];
}

export interface PhysicsEvaluationSummary {
  asset_count: number;
  scored_asset_count: number;
  mean_material_accuracy: number | null;
  mean_density_log_mae: number | null;
  mean_friction_mae: number | null;
  mean_restitution_mae: number | null;
  gt_full_coverage_asset_count: number;
}

export interface SimReadyProfileSummary {
  profile_target: string;
  passed: number;
  failed: number;
  unverified: number;
  total: number;
}

export interface SimReadyFeatureFailure {
  feature_id: string;
  requirements: string[];
  messages: string[];
}

export interface AssetSimReadyValidation {
  profile: string;
  profile_version: string;
  profile_target: string;
  status: string;
  passed: boolean;
  evidence_verified: boolean | null;
  failed_features: SimReadyFeatureFailure[];
  failed_requirements: string[];
  errors: string[];
}

export type MeshSegmentationQualityStatus =
  "pass" | "below_threshold" | "not_scoreable";

export interface MeshSegmentationMatch {
  ground_truth_id: number;
  predicted_id: number;
  iou: number;
}

export interface AssetMeshSegmentationEvaluation {
  status: MeshSegmentationQualityStatus;
  face_accuracy: number | null;
  macro_iou: number | null;
  area_accuracy: number | null;
  area_weighted_iou: number | null;
  face_count: number | null;
  ground_truth_segment_count: number | null;
  predicted_segment_count: number | null;
  face_accuracy_passed: boolean;
  macro_iou_passed: boolean;
  matches: MeshSegmentationMatch[];
}

export interface MeshSegmentationEvaluationSummary {
  asset_count: number;
  scoreable_asset_count: number;
  quality_pass_asset_count: number;
  final_export_asset_count: number;
  mean_face_accuracy: number | null;
  mean_macro_iou: number | null;
}

export interface GitProvenance {
  requested_ref: string;
  resolved_ref: string;
  branch: string;
  commit: string;
  commit_timestamp: string;
  remote: string;
  remote_url: string;
  dirty: boolean;
}

export interface DatasetProvenance {
  manifest: string;
  manifest_sha256: string;
  content_sha256: string;
  case_count: number;
  file_count: number;
}

export interface BenchmarkProvenance {
  target: GitProvenance;
  harness: GitProvenance;
  dataset: DatasetProvenance;
}

export interface RunnerGpu {
  name: string;
  memory_total_bytes: number | null;
  driver_version: string | null;
}

export interface RunnerHardware {
  architecture: string | null;
  os: string | null;
  os_release: string | null;
  cpu_model: string | null;
  logical_cpu_count: number | null;
  memory_total_bytes: number | null;
  gpus: RunnerGpu[];
}

export interface RendererDeployment {
  backend: string;
  deployment: string;
  service: string;
  function_id: string | null;
  endpoint_host: string | null;
  renderer?: string | null;
  settings?: {
    image_width?: number | null;
    image_height?: number | null;
    render_quality?: string | null;
    ovrtx_render_mode?: string | null;
    ovrtx_num_sensor_updates?: number | null;
  };
}

export interface TokenCostEstimate {
  actual_billing: boolean;
  provider_reported_cost?: boolean;
  pricing_schema_version: string;
  pricing_model: string;
  model_source: string;
  currency: string;
  standard_api_equivalent_usd: number;
  long_context_upper_bound_usd: number;
  cached_input_tokens_reported: boolean;
  pricing_as_of: string;
  pricing_source: string;
  caveat: string;
}

export interface RunTokenUsage {
  input_tokens: number | null;
  cached_input_tokens: number | null;
  uncached_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  output_tokens: number | null;
  reasoning_output_tokens: number | null;
  total_tokens: number | null;
  cost_estimate: TokenCostEstimate | null;
  cost_estimate_status:
    | "estimated"
    | "actual"
    | "provider_reported"
    | "partial_estimate"
    | "model_not_in_pricing_table"
    | "model_unreported";
  cost_estimate_scope?: "all_usage" | "driver_only" | "predict_only";
}

export interface RunExecutionMetadata {
  schema_version: string;
  runner_hardware: RunnerHardware | null;
  renderers: {
    workflow: RendererDeployment | null;
    benchmark_evidence: RendererDeployment | null;
  };
  usage: RunTokenUsage | null;
}

export interface MaterialEvaluationSummary {
  evaluated_asset_count: number;
  asset_count: number;
  mean_scores: MaterialAgreementScores;
  mean_agreement_scores: MaterialAgreementScores;
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
  judges: MaterialJudge[];
  prompt_version: string;
  caveat: string;
}

export interface MaterialScores {
  overall_assignment_quality_score: number;
  material_identity_score: number;
  color_palette_score: number;
  surface_finish_score: number;
  material_consistency_score: number;
}

export type MaterialAgreementScores = Record<
  keyof MaterialScores,
  number | null
>;

export interface MaterialJudge {
  backend: string;
  model: string;
}

export interface MaterialJudgeResult {
  judge: MaterialJudge;
  scores: MaterialScores;
  confidence: number;
}

export interface AssetMaterialEvaluation {
  scores: MaterialScores;
  agreement_scores: MaterialAgreementScores;
  confidence: number;
  issues: string[];
  semantic_consistency_findings: string[];
  rationale: string;
  judge_results: MaterialJudgeResult[];
  structural: {
    assignment_coverage_score: number | null;
    authoring_coverage_score: number | null;
    rejected_assignment_rate: number | null;
    missing_assignment_rate: number | null;
    coverage_limitation: string | null;
    final_output_available: boolean;
    final_visible_mesh_count: number | null;
    final_bound_visible_mesh_count: number | null;
    final_partially_bound_visible_mesh_count: number | null;
    final_unbound_visible_mesh_count: number | null;
    final_mesh_binding_coverage_score: number | null;
    final_unbound_visible_mesh_paths: string[];
    unique_material_count: number | null;
    material_distribution_entropy: number | null;
    unknown_assignment_rate: number | null;
  };
}

export interface RunSummary {
  total: number;
  pass: number;
  warn: number;
  fail: number;
  error: number;
  skipped: number;
  unscored: number;
  reviewed: number;
  needs_review: number;
  median_runtime_seconds: number;
  /** "n/m" when token metrics cover only a subset of assets (failed,
   * resumed, or malformed sessions); null when coverage is complete. */
  token_asset_coverage: string | null;
  input_tokens: number | null;
  cached_input_tokens: number | null;
  output_tokens: number | null;
  reasoning_output_tokens: number | null;
  driver_input_tokens: number | null;
  driver_cached_input_tokens: number | null;
  driver_output_tokens: number | null;
  driver_total_tokens: number | null;
  vision_input_tokens: number | null;
  vision_cached_input_tokens: number | null;
  vision_output_tokens: number | null;
  vision_total_tokens: number | null;
  vision_invocation_count: number | null;
  combined_total_tokens: number | null;
  total_tokens: number;
  cost_estimate: TokenCostEstimate | null;
  cost_estimate_status:
    | "estimated"
    | "actual"
    | "provider_reported"
    | "partial_estimate"
    | "model_not_in_pricing_table"
    | "model_unreported";
  cost_estimate_scope?: "all_usage" | "driver_only" | "predict_only";
  signals?: Record<string, SignalSummary>;
}

export interface SignalSummary {
  passed: number;
  total: number;
}

export interface AssetResult {
  asset_id: string;
  name: string;
  prompt: string | null;
  status: ResultStatus;
  status_label: string;
  family: string;
  tags: string[];
  source_usd: string;
  output_usd: string;
  references: EvidenceItem[];
  renders: EvidenceItem[];
  reports: ReportLink[];
  trace_links: ReportLink[];
  metrics: Record<string, string | number | boolean>;
  simready_validation?: AssetSimReadyValidation;
  material_evaluation?: AssetMaterialEvaluation;
  mesh_segmentation_evaluation?: AssetMeshSegmentationEvaluation;
  physics_evaluation?: AssetPhysicsEvaluation;
  findings: Finding[];
  feedback?: AssetFeedback[];
  review: AssetReview;
  preview: PreviewModel;
}

export interface EvidenceItem {
  label: string;
  view: string;
  path: string;
  kind: "reference" | "render" | "runtime" | "validation";
  /** Validated render provenance summary (renderer + source USD and image
   * digests). Null/absent means the image is not reproducible evidence. */
  provenance?: string | null;
  /** True when a reference image lacks validated provenance and must be
   * presented as a diagnostic preview, never as reference evidence. */
  diagnostic?: boolean;
  /** The provenance record's expected image digest; the loader verifies the
   * served bytes against it and demotes the item on mismatch. */
  image_sha256?: string | null;
  /** Exported (projected) records bind an exact-record sidecar: the loader
   * must fetch it and verify its bytes against exact_record_sha256 before
   * promoting the reference — the projection alone is not the exact OVRTX
   * metadata the evidence rule requires. */
  exact_record_sha256?: string | null;
  exact_record_url?: string | null;
  /** True when a projected record carries an incomplete exact-record
   * binding (one of digest/path missing) — the reference can never be
   * promoted because its exact metadata is unverifiable. */
  exact_record_broken?: boolean;
  /** The projection's source USD digest, compared against the verified
   * exact record before promotion. */
  source_usd_sha256?: string | null;
  /** Canonical fingerprint of the projection's sanitized render_metadata;
   * the verified exact record's sanitized metadata must match it before
   * promotion. */
  render_metadata_fingerprint?: string | null;
  sha256?: string;
  renderer?: string;
  settings?: {
    render_quality: string;
    ovrtx_render_mode: string;
    ovrtx_num_sensor_updates: number;
    active_aov: string;
    width: number;
    height: number;
    fallback: boolean;
    elapsed_seconds?: number;
  };
  camera?: ReportLink;
  media_type?: "image" | "video";
}

export interface ReportLink {
  label: string;
  path: string;
  kind: string;
  sha256?: string;
  media_type?: string;
}

export interface Finding {
  severity: ReviewSeverity;
  title: string;
  detail: string;
}

export interface AssetReview {
  reviewer: string;
  verdict: ReviewVerdict;
  severity: ReviewSeverity;
  score: number | null;
  tags: string[];
  comment: string;
  needs_rerun: boolean;
  updated_at: string | null;
}

export interface AssetFeedback {
  id: string;
  reviewer: string;
  comment: string;
  created_at: string;
}

export interface PreviewModel {
  variant:
    | "agv"
    | "pcb"
    | "valve"
    | "bin"
    | "lightbulb"
    | "roller"
    | "caster"
    | "bracket"
    | "hinge"
    | "pump"
    | "sheet"
    | "gearbox"
    | "generic";
  tone: "reference" | "good" | "warning" | "bad" | "blocked";
  image_url?: string;
}
