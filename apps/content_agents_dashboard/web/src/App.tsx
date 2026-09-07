// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import {
  Component,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react";
import { AppBarLogo } from "./components/NvidiaLogo";
import {
  compareRunsNewestFirst,
  loadBenchmarkBundle,
  loadBenchmarkIndexFingerprint,
} from "./data/agenticBenchmark";
import {
  compareFeedbackNewestFirst,
  limitStoredFeedback,
  mergeFeedback,
  parseStoredFeedback,
} from "./data/feedback";
import {
  AppBar,
  Badge,
  Button,
  Card,
  Switch,
  TextArea,
  TextInput,
  ThemeProvider,
  Tooltip,
  cx,
} from "./ui";
import type {
  AssetPhysicsEvaluation,
  AssetResult,
  AssetFeedback,
  BenchmarkBundle,
  BenchmarkRun,
  EvidenceItem,
  RendererDeployment,
  ResultStatus,
  ReviewSeverity,
  RunnerHardware,
  WorkflowId,
  WorkflowSummary,
} from "./types/benchmark";

const BENCHMARK_INDEX_URL = "/benchmark-data/index.json";
const FEEDBACK_STORAGE_KEY = "content-agents-dashboard-feedback";

type StatusFilter = ResultStatus | "all";
type ThemeMode = "light" | "dark";
type FeedbackDraft = Pick<AssetFeedback, "reviewer" | "comment">;
type RunView = "results" | "feedback";

export interface RunFeedbackEntry {
  asset: AssetResult;
  feedback: AssetFeedback;
}

interface ExpandedEvidence {
  category: "Reference" | "Render";
  item: EvidenceItem;
}

const STATUS_FILTERS: Array<{ value: StatusFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "pass", label: "Pass" },
  { value: "warn", label: "Warn" },
  { value: "fail", label: "Fail" },
  { value: "error", label: "Error" },
  { value: "skipped", label: "Skipped" },
  { value: "unknown", label: "Unscored" },
];

const WORKFLOW_ORDER: WorkflowId[] = [
  "material-agentic",
  "material-fixed",
  "mesh-segmentation",
  "physics-fixed",
  "physics-agentic",
  "cad-benchmark",
  "cad-to-simready",
];

function getInitialTheme(): ThemeMode {
  try {
    const stored = localStorage.getItem("content-agents-dashboard-theme");
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // Storage can be unavailable in restricted browser contexts.
  }
  return "light";
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true
  );
}

function statusBadgeColor(status: ResultStatus) {
  if (status === "pass") return "green";
  if (status === "warn") return "yellow";
  if (status === "fail") return "red";
  if (status === "error") return "red";
  if (status === "skipped") return "blue";
  return "gray";
}

function severityBadgeColor(severity: ReviewSeverity) {
  if (severity === "blocking") return "red";
  if (severity === "major" || severity === "warning") return "yellow";
  if (severity === "minor") return "blue";
  return "gray";
}

function formatDate(value: string) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "Date unavailable";
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
}

function formatTokenCount(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "n/a";
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatBytes(value: number | null) {
  if (value === null || !Number.isFinite(value) || value <= 0) return null;
  return `${(value / 1024 ** 3).toFixed(1)} GiB`;
}

function hardwareLabel(hardware: RunnerHardware | null) {
  if (!hardware) return "Not reported";
  const cpu = hardware.cpu_model
    ? `${hardware.cpu_model}${
        hardware.logical_cpu_count
          ? ` (${hardware.logical_cpu_count} logical CPUs)`
          : ""
      }`
    : hardware.logical_cpu_count
      ? `${hardware.logical_cpu_count} logical CPUs`
      : null;
  const memory = formatBytes(hardware.memory_total_bytes);
  const gpus = hardware.gpus.length
    ? hardware.gpus
        .map((gpu) => {
          const gpuMemory = formatBytes(gpu.memory_total_bytes);
          return gpuMemory ? `${gpu.name} (${gpuMemory})` : gpu.name;
        })
        .join(", ")
    : "No local GPU detected";
  return [cpu, memory && `${memory} RAM`, gpus].filter(Boolean).join(" · ");
}

function rendererLabel(renderer: RendererDeployment | null) {
  if (!renderer) return "Not reported";
  const deployment =
    renderer.deployment.toLowerCase() === "nvcf"
      ? "NVCF deployment"
      : renderer.deployment
          .split("-")
          .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
          .join(" ");
  return `${renderer.service} · ${deployment}`;
}

function evidenceRendererLabel(renderer: RendererDeployment | null) {
  if (!renderer) return "Not reported";
  const settings = renderer.settings;
  const dimensions =
    settings?.image_width && settings.image_height
      ? `${settings.image_width}x${settings.image_height}`
      : null;
  return [
    renderer.service,
    renderer.renderer?.toUpperCase(),
    settings?.ovrtx_render_mode,
    settings?.render_quality,
    dimensions,
  ]
    .filter(Boolean)
    .join(" · ");
}

function evidenceSettingsLabel(evidence: EvidenceItem) {
  const settings = evidence.settings;
  if (!settings) return evidence.renderer ?? null;
  return [
    evidence.renderer,
    settings.render_quality,
    settings.ovrtx_render_mode,
    settings.active_aov,
    `${settings.ovrtx_num_sensor_updates} sensor updates`,
    `${settings.width}x${settings.height}`,
    settings.elapsed_seconds === undefined
      ? null
      : `${settings.elapsed_seconds}s render`,
    settings.fallback ? "fallback" : "exact artifact",
  ]
    .filter(Boolean)
    .join(" · ");
}

async function copyText(value: string) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(value);
      return;
    } catch {
      // Plain HTTP origins may expose the API but reject writes.
    }
  }

  const input = document.createElement("textarea");
  input.value = value;
  input.setAttribute("readonly", "");
  input.style.position = "fixed";
  input.style.opacity = "0";
  document.body.append(input);
  input.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } finally {
    input.remove();
  }
  if (!copied) throw new Error("Clipboard access was unavailable.");
}

function feedbackKey(
  run: Pick<BenchmarkRun, "workflow_id" | "run_id">,
  assetId: string,
) {
  return `${run.workflow_id}:${run.run_id}:${assetId}`;
}

export function createFeedbackId() {
  const crypto = globalThis.crypto;
  if (!crypto) throw new Error("Secure random generation is unavailable.");
  if (crypto.randomUUID) return crypto.randomUUID();
  const words = new Uint32Array(4);
  crypto.getRandomValues(words);
  return `${Date.now()}-${Array.from(words, (word) =>
    word.toString(36).padStart(7, "0"),
  ).join("")}`;
}

function initialSubmittedFeedback(): Record<string, AssetFeedback[]> {
  try {
    return limitStoredFeedback(
      parseStoredFeedback(localStorage.getItem(FEEDBACK_STORAGE_KEY)),
    ).feedback;
  } catch {
    return {};
  }
}

function bundledFeedback(asset: AssetResult): AssetFeedback[] {
  const feedback = asset.feedback ?? [];
  const legacyReview = asset.review;
  if (!legacyReview.reviewer.trim() || !legacyReview.comment.trim()) {
    return feedback;
  }
  return mergeFeedback(feedback, {
    id: `legacy-${asset.asset_id}`,
    reviewer: legacyReview.reviewer,
    comment: legacyReview.comment,
    created_at: legacyReview.updated_at ?? "",
  });
}

function feedbackForAsset(
  run: BenchmarkRun,
  asset: AssetResult,
  submittedFeedback: Record<string, AssetFeedback[]>,
) {
  return [
    ...bundledFeedback(asset),
    ...(submittedFeedback[feedbackKey(run, asset.asset_id)] ?? []),
  ].sort(compareFeedbackNewestFirst);
}

function feedbackForRun(
  run: BenchmarkRun,
  submittedFeedback: Record<string, AssetFeedback[]>,
): RunFeedbackEntry[] {
  return run.assets
    .flatMap((asset) =>
      feedbackForAsset(run, asset, submittedFeedback).map((feedback) => ({
        asset,
        feedback,
      })),
    )
    .sort((left, right) =>
      compareFeedbackNewestFirst(left.feedback, right.feedback),
    );
}

class DashboardErrorBoundary extends Component<
  { children: ReactNode; resetKey: string },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Dashboard content failed to render", error, info);
  }

  componentDidUpdate(
    previous: Readonly<{ children: ReactNode; resetKey: string }>,
  ) {
    if (previous.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return (
        <section className="dashboard-error" role="alert">
          <h2>Could not display this benchmark run</h2>
          <p>{this.state.error.message}</p>
          <Button color="brand" onClick={this.reset}>
            Try again
          </Button>
        </section>
      );
    }
    return this.props.children;
  }
}

function escapeMarkdown(value: string) {
  return value.replace(/([\\`*_{}[\]()#+\-.!|<>])/g, "\\$1");
}

export function createIssueDigest(
  workflow: WorkflowSummary,
  run: BenchmarkRun,
  feedback: RunFeedbackEntry[],
) {
  const grouped = new Map<string, RunFeedbackEntry[]>();
  for (const entry of feedback) {
    const entries = grouped.get(entry.asset.asset_id) ?? [];
    entries.push(entry);
    grouped.set(entry.asset.asset_id, entries);
  }

  const header = [
    `# ${escapeMarkdown(`${workflow.name}: ${run.label}`)}`,
    "",
    `- Run ID: \`${run.run_id}\``,
    `- Branch: \`${run.branch}\``,
    `- Commit: \`${run.commit}\``,
    `- Suite: \`${run.suite_id}\``,
    "",
  ];

  if (!grouped.size)
    return [...header, "No reviewer feedback recorded."].join("\n");

  const sections = [...grouped.values()].flatMap((entries) => {
    const [first] = entries;
    if (!first) return [];
    const findings = first.asset.findings
      .map((finding) => escapeMarkdown(finding.title))
      .join("; ");
    return [
      `## ${escapeMarkdown(first.asset.name)} (\`${first.asset.asset_id}\`)`,
      `Status: ${escapeMarkdown(first.asset.status_label)}${findings ? ` · Findings: ${findings}` : ""}`,
      ...entries.map(
        (entry) =>
          `- ${escapeMarkdown(entry.feedback.reviewer)} (${formatDate(entry.feedback.created_at)}): ${escapeMarkdown(entry.feedback.comment)}`,
      ),
      "",
    ];
  });

  return [...header, ...sections].join("\n");
}

export function getRunForWorkflow(
  runs: BenchmarkRun[],
  workflowId: WorkflowId,
  selectedRunId: string,
) {
  const workflowRuns = runs
    .filter((run) => run.workflow_id === workflowId)
    .sort(compareRunsNewestFirst);
  const selected = workflowRuns.find((run) => run.run_id === selectedRunId);
  // Prefer a run that both scored and finished. Dropping the completeness
  // condition made a failed or still-running run the default view for every
  // workflow. The `?? workflowRuns[0]` fallback below still shows something
  // when no complete run exists, so requiring it here costs no visibility --
  // which is what I got wrong when I first argued against restoring it.
  const latestScoredRun = workflowRuns.find(
    (run) => run.scored && run.run_status === "complete",
  );
  return selected ?? latestScoredRun ?? workflowRuns[0] ?? null;
}

export function nextBaselineRunId(current: string, selectedRunId: string) {
  // A run cannot be its own baseline -- `MeshSegmentationStrip` filters the
  // selected run out of the candidates, so keeping it would leave a dangling
  // selection and silently drop the delta. Everything else survives: clearing
  // unconditionally discarded a deliberate choice every time the reader opened
  // another run, which is exactly the workflow of comparing several runs
  // against one baseline.
  return current === selectedRunId ? "" : current;
}

export default function App() {
  const [bundle, setBundle] = useState<BenchmarkBundle | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [selectedWorkflowId, setSelectedWorkflowId] =
    useState<WorkflowId>("material-agentic");
  const [selectedRunId, setSelectedRunId] = useState("");
  // A benchmark run only means something next to another run, so the
  // segmentation strip can report a delta against a chosen baseline.
  const [baselineRunId, setBaselineRunId] = useState("");
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [query, setQuery] = useState("");
  const [runView, setRunView] = useState<RunView>("results");
  const [submittedFeedback, setSubmittedFeedback] = useState(
    initialSubmittedFeedback,
  );
  const [feedbackDrafts, setFeedbackDrafts] = useState<
    Record<string, FeedbackDraft>
  >({});
  const [exportNotice, setExportNotice] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    let loading = false;
    let loaded = false;
    let fingerprint = "";
    let currentBundle: BenchmarkBundle | null = null;

    const refresh = async (force = false) => {
      if (loading) return;
      loading = true;
      try {
        if (!force && fingerprint) {
          const latestFingerprint =
            await loadBenchmarkIndexFingerprint(BENCHMARK_INDEX_URL);
          if (latestFingerprint === fingerprint) return;
        }
        const data = await loadBenchmarkBundle(BENCHMARK_INDEX_URL);
        if (!mounted) return;
        fingerprint = data.index_fingerprint;
        if (!loaded) {
          const latestRun = [...data.runs].sort(compareRunsNewestFirst)[0];
          if (latestRun) setSelectedWorkflowId(latestRun.workflow_id);
        }
        loaded = true;
        currentBundle = data;
        setBundle(data);
        setLoadError(null);
        // References render demoted until their bytes verify (fail closed);
        // upgrade them off the critical path and re-render once done, so
        // first paint never blocks on hashing every reference image.
        // A completion from an earlier refresh must not restore its bundle
        // after a later refresh installed a newer one. Compare bundle
        // identity, not the fingerprint string: a degraded index loads with
        // an empty fingerprint on every poll, so two different bundles can
        // share "" and the string compare would let the stale restore
        // through.
        void data.verify_references?.().then(() => {
          if (mounted && currentBundle === data) {
            setBundle({ ...data });
          }
        });
      } catch (error: unknown) {
        if (!mounted) return;
        if (!loaded) {
          setLoadError(error instanceof Error ? error.message : String(error));
        } else {
          console.warn("Benchmark refresh failed", error);
        }
      } finally {
        loading = false;
      }
    };

    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") void refresh();
    };

    void refresh(true);
    const interval = window.setInterval(() => void refresh(), 30_000);
    window.addEventListener("focus", refreshWhenVisible);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      mounted = false;
      window.clearInterval(interval);
      window.removeEventListener("focus", refreshWhenVisible);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("content-agents-dashboard-theme", theme);
    } catch {
      // Theme state remains usable even when persistence is unavailable.
    }
  }, [theme]);

  useEffect(() => {
    try {
      localStorage.setItem(
        FEEDBACK_STORAGE_KEY,
        JSON.stringify(submittedFeedback),
      );
    } catch {
      setExportNotice(
        "Feedback is available in this tab but browser storage failed.",
      );
    }
  }, [submittedFeedback]);

  const workflows = useMemo(() => {
    if (!bundle) return [];
    return [...bundle.workflows].sort(
      (a, b) => WORKFLOW_ORDER.indexOf(a.id) - WORKFLOW_ORDER.indexOf(b.id),
    );
  }, [bundle]);

  const selectedWorkflow =
    workflows.find((workflow) => workflow.id === selectedWorkflowId) ??
    workflows[0];
  const activeWorkflowId = selectedWorkflow?.id ?? selectedWorkflowId;

  const selectedRun = bundle
    ? getRunForWorkflow(bundle.runs, activeWorkflowId, selectedRunId)
    : null;

  const workflowRuns = useMemo(() => {
    if (!bundle) return [];
    return bundle.runs
      .filter((run) => run.workflow_id === activeWorkflowId)
      .sort(compareRunsNewestFirst);
  }, [activeWorkflowId, bundle]);

  useEffect(() => {
    if (selectedWorkflow && selectedWorkflow.id !== selectedWorkflowId) {
      setSelectedWorkflowId(selectedWorkflow.id);
    }
  }, [selectedWorkflow, selectedWorkflowId]);

  useEffect(() => {
    if (!selectedRun) return;
    if (selectedRun.run_id !== selectedRunId) {
      setSelectedRunId(selectedRun.run_id);
    }
  }, [selectedRun, selectedRunId]);

  useEffect(() => {
    setExportNotice(null);
  }, [selectedRun?.run_id]);

  const filteredAssets = useMemo(() => {
    if (!selectedRun) return [];
    const normalizedQuery = query.trim().toLowerCase();
    return selectedRun.assets.filter((asset) => {
      const matchesStatus =
        statusFilter === "all" || asset.status === statusFilter;
      const searchable = [
        asset.name,
        asset.asset_id,
        asset.family,
        asset.status_label,
        asset.prompt ?? "",
        ...asset.tags,
      ]
        .join(" ")
        .toLowerCase();
      const matchesQuery =
        !normalizedQuery || searchable.includes(normalizedQuery);
      return matchesStatus && matchesQuery;
    });
  }, [query, selectedRun, statusFilter]);

  const selectedAsset = useMemo(() => {
    if (!filteredAssets.length) return null;
    return (
      filteredAssets.find((asset) => asset.asset_id === selectedAssetId) ??
      filteredAssets[0] ??
      null
    );
  }, [filteredAssets, selectedAssetId]);

  useEffect(() => {
    if (selectedAsset && selectedAsset.asset_id !== selectedAssetId) {
      setSelectedAssetId(selectedAsset.asset_id);
    }
  }, [selectedAsset, selectedAssetId]);

  const selectedFeedbackKey =
    selectedAsset && selectedRun
      ? feedbackKey(selectedRun, selectedAsset.asset_id)
      : "";
  const selectedFeedback =
    selectedAsset && selectedRun
      ? feedbackForAsset(selectedRun, selectedAsset, submittedFeedback)
      : [];
  const runFeedback = useMemo(
    () => (selectedRun ? feedbackForRun(selectedRun, submittedFeedback) : []),
    [selectedRun, submittedFeedback],
  );
  const feedbackDraft = feedbackDrafts[selectedFeedbackKey] ?? {
    reviewer: "",
    comment: "",
  };

  const updateFeedbackDraft = (patch: Partial<FeedbackDraft>) => {
    if (!selectedFeedbackKey) return;
    setFeedbackDrafts((previous) => ({
      ...previous,
      [selectedFeedbackKey]: {
        reviewer: "",
        comment: "",
        ...previous[selectedFeedbackKey],
        ...patch,
      },
    }));
  };

  const submitSelectedFeedback = () => {
    if (!selectedAsset || !selectedRun) return;
    const reviewer = feedbackDraft.reviewer.trim();
    const comment = feedbackDraft.comment.trim();
    if (!reviewer || !comment) return;
    const entry: AssetFeedback = {
      id: createFeedbackId(),
      reviewer,
      comment,
      created_at: new Date().toISOString(),
    };
    const limited = limitStoredFeedback({
      ...submittedFeedback,
      [selectedFeedbackKey]: [
        ...(submittedFeedback[selectedFeedbackKey] ?? []),
        entry,
      ],
    });
    setSubmittedFeedback(limited.feedback);
    if (limited.dropped) {
      setExportNotice(
        `Browser storage retained the newest feedback and removed ${limited.dropped} old entr${limited.dropped === 1 ? "y" : "ies"}.`,
      );
    }
    setFeedbackDrafts((previous) => ({
      ...previous,
      [selectedFeedbackKey]: { reviewer: "", comment: "" },
    }));
  };

  const copyIssueDigest = async () => {
    if (!selectedRun || !selectedWorkflow) return;
    const digest = createIssueDigest(
      selectedWorkflow,
      selectedRun,
      runFeedback,
    );
    try {
      await copyText(digest);
      setExportNotice("Issue digest copied.");
    } catch {
      setExportNotice("Clipboard access was unavailable.");
    }
  };

  const downloadFeedbackJson = () => {
    if (!selectedRun || !selectedWorkflow) return;
    const payload = {
      workflow: {
        id: selectedWorkflow.id,
        name: selectedWorkflow.name,
      },
      run: {
        run_id: selectedRun.run_id,
        branch: selectedRun.branch,
        commit: selectedRun.commit,
        suite_id: selectedRun.suite_id,
        created_at: selectedRun.created_at,
      },
      feedback: runFeedback.map(({ asset, feedback }) => ({
        ...feedback,
        asset_id: asset.asset_id,
        asset_name: asset.name,
        status: asset.status,
        status_label: asset.status_label,
        findings: asset.findings,
      })),
    };
    const objectUrl = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = `${selectedRun.run_id}-feedback.json`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
    setExportNotice("Feedback JSON downloaded.");
  };

  if (loadError) {
    return (
      <ThemeProvider theme={theme} global>
        <div className="empty-state">
          <h1>Content Agents Benchmarks</h1>
          <p>{loadError}</p>
          <Button color="brand" onClick={() => window.location.reload()}>
            Reload
          </Button>
        </div>
      </ThemeProvider>
    );
  }

  if (!bundle || !selectedWorkflow || !selectedRun) {
    return (
      <ThemeProvider theme={theme} global>
        <div className="empty-state">
          <div className="loading-mark" />
          <p>Loading benchmark bundle</p>
        </div>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme} global>
      <div className="app-frame">
        <AppBar
          slotStart={
            <>
              <AppBarLogo />
              <div className="app-title">
                <span>Content Agents Benchmarks</span>
                <small>{bundle.dashboard_id}</small>
              </div>
            </>
          }
          slotEnd={
            <div className="app-bar-actions">
              <Switch
                checked={theme === "dark"}
                slotLabel={theme === "dark" ? "Dark" : "Light"}
                onCheckedChange={(checked) =>
                  setTheme(checked ? "dark" : "light")
                }
              />
              <Badge color="brand" kind="outline">
                Live artifacts
              </Badge>
            </div>
          }
        />
        <WorkflowTabs
          workflows={workflows}
          runs={bundle.runs}
          selectedWorkflowId={selectedWorkflow.id}
          onSelectWorkflow={(workflowId) => {
            setSelectedWorkflowId(workflowId);
            setSelectedRunId("");
            setSelectedAssetId("");
            setStatusFilter("all");
            setQuery("");
          }}
        />
        <div className="workspace">
          <RunHistoryNav
            workflow={selectedWorkflow}
            runs={workflowRuns}
            selectedRunId={selectedRun.run_id}
            generatedAt={bundle.generated_at}
            onSelectRun={(runId) => {
              setSelectedRunId(runId);
              setBaselineRunId((current) => nextBaselineRunId(current, runId));
              setSelectedAssetId("");
              setStatusFilter("all");
              setQuery("");
            }}
          />
          <main className="dashboard-main">
            <DashboardErrorBoundary
              resetKey={`${selectedWorkflow.id}:${selectedRun.run_id}:${runView}`}
            >
              <RunHeader workflow={selectedWorkflow} run={selectedRun} />
              <ExecutionSummary run={selectedRun} />
              <KpiStrip run={selectedRun} />
              <MeshSegmentationStrip
                run={selectedRun}
                baselineRuns={workflowRuns.filter(
                  (candidate) => candidate.run_id !== selectedRun.run_id,
                )}
                baselineRunId={baselineRunId}
                onSelectBaseline={setBaselineRunId}
              />
              <MaterialEvaluationStrip run={selectedRun} />
              <PhysicsEvaluationStrip run={selectedRun} />
              <RunContentTabs
                feedbackCount={runFeedback.length}
                selectedView={runView}
                onSelectView={setRunView}
              />
              {runView === "results" ? (
                <section className="results-layout">
                  <div className="results-column">
                    <Toolbar
                      query={query}
                      statusFilter={statusFilter}
                      onQueryChange={setQuery}
                      onStatusFilterChange={setStatusFilter}
                    />
                    <AssetList
                      assets={filteredAssets}
                      selectedAssetId={selectedAsset?.asset_id ?? ""}
                      onSelect={setSelectedAssetId}
                    />
                  </div>
                  <AssetDetail
                    key={`${selectedRun.run_id}-${selectedAsset?.asset_id ?? "none"}`}
                    asset={selectedAsset}
                    feedback={selectedFeedback}
                    feedbackDraft={feedbackDraft}
                    onFeedbackDraftChange={updateFeedbackDraft}
                    onSubmitFeedback={submitSelectedFeedback}
                  />
                </section>
              ) : (
                <RunFeedbackView
                  workflow={selectedWorkflow}
                  run={selectedRun}
                  feedback={runFeedback}
                  exportNotice={exportNotice}
                  onCopyIssueDigest={() => void copyIssueDigest()}
                  onDownloadJson={downloadFeedbackJson}
                  onOpenAsset={(assetId) => {
                    setSelectedAssetId(assetId);
                    setRunView("results");
                  }}
                />
              )}
            </DashboardErrorBoundary>
          </main>
        </div>
      </div>
    </ThemeProvider>
  );
}

interface WorkflowNavProps {
  workflows: WorkflowSummary[];
  runs: BenchmarkRun[];
  selectedWorkflowId: WorkflowId;
  onSelectWorkflow: (workflowId: WorkflowId) => void;
}

function WorkflowTabs({
  workflows,
  runs,
  selectedWorkflowId,
  onSelectWorkflow,
}: WorkflowNavProps) {
  return (
    <nav
      className="workflow-tabs"
      aria-label="Benchmark workflows"
      role="tablist"
    >
      {workflows.map((workflow) => {
        const runCount = runs.filter(
          (run) => run.workflow_id === workflow.id,
        ).length;
        return (
          <button
            key={workflow.id}
            aria-selected={workflow.id === selectedWorkflowId}
            className={cx(
              "workflow-tab",
              workflow.id === selectedWorkflowId && "is-active",
            )}
            role="tab"
            type="button"
            onClick={() => onSelectWorkflow(workflow.id)}
          >
            <span>{workflow.name}</span>
            <Badge kind="outline">{runCount} runs</Badge>
          </button>
        );
      })}
    </nav>
  );
}

interface RunHistoryNavProps {
  workflow: WorkflowSummary;
  runs: BenchmarkRun[];
  selectedRunId: string;
  generatedAt: string;
  onSelectRun: (runId: string) => void;
}

function RunHistoryNav({
  workflow,
  runs,
  selectedRunId,
  generatedAt,
  onSelectRun,
}: RunHistoryNavProps) {
  return (
    <aside className="run-nav">
      <div className="run-nav-heading">
        <div>
          <div className="nav-heading">Run history</div>
          <strong>{workflow.name}</strong>
        </div>
        <Badge kind="outline">{runs.length}</Badge>
      </div>
      <div className="run-history-list">
        {runs.map((run, index) => (
          <RunHistoryItem
            key={run.run_id}
            run={run}
            latest={index === 0}
            selected={run.run_id === selectedRunId}
            onSelect={() => onSelectRun(run.run_id)}
          />
        ))}
      </div>
      <div className="nav-footer">
        <span>Generated</span>
        <strong>{formatDate(generatedAt)}</strong>
      </div>
    </aside>
  );
}

function RunHistoryItem({
  run,
  latest,
  selected,
  onSelect,
}: {
  run: BenchmarkRun;
  latest: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const scoredTotal = run.summary.total - run.summary.unscored;
  const status = benchmarkRunResultStatus(run);
  const statusLabel = run.scored
    ? status
    : run.run_status === "failed"
      ? "error"
      : "unscored";

  return (
    <button
      className={cx("run-history-item", selected && "is-selected")}
      type="button"
      onClick={onSelect}
    >
      <span className="run-history-title">
        <code title={run.run_id}>{run.run_id}</code>
        {latest && <Badge color="brand">latest</Badge>}
      </span>
      <span className="run-history-time">{formatDate(run.created_at)}</span>
      <span className="run-history-ref">
        <span title={run.branch}>{run.branch}</span>
        <code title={run.commit}>{run.commit}</code>
      </span>
      <span className="run-history-counts">
        <Badge color={statusBadgeColor(status)}>{statusLabel}</Badge>
        <span>
          {scoredTotal > 0
            ? `${run.summary.pass}/${scoredTotal} pass`
            : "No scored cases"}
          {run.summary.unscored > 0
            ? ` · ${run.summary.unscored} unscored`
            : ""}
        </span>
      </span>
    </button>
  );
}

export function benchmarkRunResultStatus(run: BenchmarkRun): ResultStatus {
  if (!run.scored) return run.run_status === "failed" ? "error" : "unknown";
  if (run.summary.error) return "error";
  if (run.summary.fail) return "fail";
  if (run.summary.warn) return "warn";
  if (run.summary.skipped) return "skipped";
  if (run.summary.pass) return "pass";
  return "unknown";
}

function RunHeader({
  workflow,
  run,
}: {
  workflow: WorkflowSummary;
  run: BenchmarkRun;
}) {
  return (
    <section className="run-header">
      <div>
        <div className="eyebrow">{workflow.name}</div>
        <h1>{run.label}</h1>
        <p>{workflow.description}</p>
      </div>
      <div className="run-meta">
        <Meta copyable label="Run ID" value={run.run_id} mono wide />
        <Meta copyable label="Branch" value={run.branch} mono wide />
        <Meta copyable label="Commit" value={run.commit} mono wide />
        {run.provenance && (
          <>
            <Meta
              copyable
              label="Harness branch"
              value={run.provenance.harness.branch}
              mono
              wide
            />
            <Meta
              copyable
              label="Harness commit"
              value={run.provenance.harness.commit}
              mono
              wide
            />
            <Meta
              copyable
              label="Dataset fingerprint"
              value={run.provenance.dataset.content_sha256}
              mono
              wide
            />
          </>
        )}
        <Meta label="Worktree" value={run.dirty ? "Dirty" : "Clean"} />
        <Meta label="Suite" value={run.suite_id} />
        {(run.simready_profiles ?? []).length > 0 && (
          <Meta
            label="SimReady profile"
            value={(run.simready_profiles ?? [])
              .map((profile) => {
                if (!profile.total)
                  return `${profile.profile_target} · validation not reported`;
                const verifiedTotal = profile.total - profile.unverified;
                const results = [profile.profile_target];
                if (verifiedTotal) {
                  results.push(`${profile.passed}/${verifiedTotal} PASS`);
                  if (profile.failed) results.push(`${profile.failed} FAIL`);
                }
                if (profile.unverified)
                  results.push(`${profile.unverified} UNVERIFIED`);
                return results.join(" · ");
              })
              .join("; ")}
            wide
          />
        )}
        <Meta
          label="Runner"
          value={run.model ? `${run.runner} / ${run.model}` : run.runner}
        />
        {run.vision_model && (
          <Meta
            label="Vision"
            value={
              run.vision_backend
                ? `${run.vision_backend} / ${run.vision_model}`
                : run.vision_model
            }
          />
        )}
        <Meta label="Source" value={run.source} />
        <Meta label="Run status" value={run.run_status} />
        <Meta label="Created" value={formatDate(run.created_at)} />
      </div>
    </section>
  );
}

function ExecutionSummary({ run }: { run: BenchmarkRun }) {
  const execution = run.execution;
  if (!execution) return null;
  const workflowRenderer = execution.renderers.workflow;
  const evidenceRenderer = execution.renderers.benchmark_evidence;
  const hardware = execution.runner_hardware;
  return (
    <section className="execution-summary" aria-label="Execution environment">
      <div className="execution-summary-heading">Execution environment</div>
      <div className="execution-summary-grid">
        <Meta
          label="Workflow renderer"
          value={rendererLabel(workflowRenderer)}
        />
        <Meta
          label="Benchmark evidence renderer"
          value={evidenceRendererLabel(evidenceRenderer)}
        />
        <Meta
          label="Runner OS"
          value={
            hardware?.os
              ? `${hardware.os} ${hardware.os_release ?? ""} · ${
                  hardware.architecture ?? "unknown architecture"
                }`
              : "Not reported"
          }
        />
        {workflowRenderer?.function_id && (
          <Meta
            copyable
            label="NVCF render function"
            value={workflowRenderer.function_id}
            mono
            wide
          />
        )}
        <Meta label="Runner hardware" value={hardwareLabel(hardware)} wide />
      </div>
    </section>
  );
}

function Meta({
  label,
  value,
  mono = false,
  copyable = false,
  wide = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
  copyable?: boolean;
  wide?: boolean;
}) {
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "error">(
    "idle",
  );
  const resetTimer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (resetTimer.current !== null) window.clearTimeout(resetTimer.current);
    },
    [],
  );

  const onCopy = async () => {
    try {
      await copyText(value);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("error");
    }
    if (resetTimer.current !== null) window.clearTimeout(resetTimer.current);
    resetTimer.current = window.setTimeout(() => setCopyStatus("idle"), 1600);
  };

  return (
    <div className={cx("meta-item", wide && "meta-item-wide")}>
      <div className="meta-item-label">
        <span>{label}</span>
        {copyable && (
          <button
            aria-label={`Copy ${label}`}
            className="meta-copy-button"
            onClick={() => void onCopy()}
            type="button"
          >
            {copyStatus === "copied"
              ? "Copied"
              : copyStatus === "error"
                ? "Unavailable"
                : "Copy"}
          </button>
        )}
      </div>
      <strong className={cx("meta-item-value", mono && "mono")} title={value}>
        {value}
      </strong>
    </div>
  );
}

/**
 * Reasoning tokens are billed as output but are invisible in the transcript,
 * so reporting them as a share of output explains cost that reading the run
 * would never account for.
 */
export function reasoningShare(
  reasoning: number | null,
  output: number | null,
) {
  if (reasoning === null) return undefined;
  if (output === null || output <= 0) {
    return `${formatTokenCount(reasoning)} reasoning`;
  }
  return `${formatTokenCount(reasoning)} reasoning (${Math.round(
    (reasoning / output) * 100,
  )}% of output)`;
}

function KpiStrip({ run }: { run: BenchmarkRun }) {
  const formatRuntime = (seconds: number) => {
    if (!Number.isFinite(seconds) || seconds <= 0) return "n/a";
    const roundedSeconds = Math.round(seconds);
    if (roundedSeconds < 60) return `${roundedSeconds}s`;
    const minutes = Math.floor(roundedSeconds / 60);
    const remainingSeconds = roundedSeconds % 60;
    return remainingSeconds
      ? `${minutes}m ${remainingSeconds}s`
      : `${minutes}m`;
  };
  const cost = run.summary.cost_estimate;
  const providerReportedCost = Boolean(cost?.provider_reported_cost);
  const costValue = cost
    ? cost.actual_billing || providerReportedCost
      ? `$${cost.standard_api_equivalent_usd.toFixed(2)}`
      : `$${cost.standard_api_equivalent_usd.toFixed(2)}-$${cost.long_context_upper_bound_usd.toFixed(2)}`
    : "n/a";
  const costModelSource = cost?.model_source.includes("assumption")
    ? "assumed"
    : "recorded";
  const costDetail = cost
    ? providerReportedCost
      ? run.summary.cost_estimate_scope === "driver_only"
        ? `Driver-only provider-reported cost · delegated VLM cost excluded · not verified invoice billing · ${cost.pricing_model.toUpperCase()}`
        : `Provider-reported cost · subscription or API accounting · not verified invoice billing · ${cost.pricing_model.toUpperCase()}`
      : cost.actual_billing
        ? run.summary.cost_estimate_scope === "driver_only"
          ? `Driver-only provider billing value · delegated VLM cost excluded · verify against invoice · ${cost.pricing_model.toUpperCase()}`
          : `Provider billing value · verify against invoice · ${cost.pricing_model.toUpperCase()}`
        : run.summary.cost_estimate_status === "partial_estimate"
          ? `Driver only · delegated VLM cost excluded · ${cost.pricing_model.toUpperCase()} ${costModelSource} · not billing`
          : `Standard to all-long-context upper · ${cost.pricing_model.toUpperCase()} ${costModelSource} · not billing`
    : run.summary.cost_estimate_status === "model_not_in_pricing_table"
      ? "Run model has no configured pricing table"
      : "Run model was not reported";
  // An estimate computed from per-asset token sums inherits their caveats;
  // without repeating them here the dollar figure reads as a whole-run
  // estimate. The predict-only caveat keys on the usage scope the bundle
  // declares (not a workflow-id string): a fixed pipeline that later
  // instruments identification drops the caveat by declaring all_usage, and
  // any other predict-only bundle gains it. ANY run whose
  // token_asset_coverage is set (e.g. an agentic run where some assets
  // failed before writing cost metrics) adds the partial-coverage note.
  const predictOnly = run.summary.cost_estimate_scope === "predict_only";
  const predictOnlyCaveat =
    "Predict step only — identification VLM usage is untracked";
  const coverageNote = run.summary.token_asset_coverage
    ? `${run.summary.token_asset_coverage} assets measured`
    : undefined;
  const coverageCaveat = coverageNote ? ` · ${coverageNote}` : "";
  const predictOnlyDetail = `${predictOnlyCaveat}${coverageCaveat}`;
  const costCaveat =
    cost && !providerReportedCost && !cost.actual_billing
      ? predictOnly
        ? ` · predict step only — identification VLM usage is untracked${coverageCaveat}`
        : coverageCaveat
      : "";

  return (
    <div className="kpi-strip">
      <KpiCard label="Pass" value={run.summary.pass} color="green" />
      <KpiCard label="Warn" value={run.summary.warn} color="yellow" />
      <KpiCard label="Fail" value={run.summary.fail} color="red" />
      <KpiCard label="Error" value={run.summary.error} color="red" />
      <KpiCard label="Skipped" value={run.summary.skipped} color="blue" />
      <KpiCard label="Unscored" value={run.summary.unscored} color="gray" />
      <KpiCard
        label="Median runtime"
        value={formatRuntime(run.summary.median_runtime_seconds)}
        color="gray"
      />
      <KpiCard
        label="Input tokens"
        value={formatTokenCount(run.summary.input_tokens)}
        color="gray"
        detail={
          predictOnly
            ? predictOnlyDetail
            : `${
                run.summary.cached_input_tokens === null
                  ? "Cache usage not reported"
                  : "Includes cached input"
              }${coverageCaveat}`
        }
      />
      <KpiCard
        label="Cached input"
        value={formatTokenCount(run.summary.cached_input_tokens)}
        color="gray"
      />
      <KpiCard
        label="Output tokens"
        value={formatTokenCount(run.summary.output_tokens)}
        color="gray"
        detail={
          predictOnly
            ? predictOnlyDetail
            : [
                reasoningShare(
                  run.summary.reasoning_output_tokens,
                  run.summary.output_tokens,
                ),
                coverageNote,
              ]
                .filter(Boolean)
                .join(" · ") || undefined
        }
      />
      <KpiCard
        label="Potential model cost"
        value={costValue}
        color="gray"
        detail={`${costDetail}${costCaveat}`}
      />
    </div>
  );
}

function KpiCard({
  label,
  value,
  color,
  detail,
}: {
  label: string;
  value: ReactNode;
  color: "green" | "yellow" | "red" | "blue" | "gray";
  detail?: ReactNode;
}) {
  return (
    <Card className={cx("kpi-card", `kpi-${color}`)}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </Card>
  );
}

function segmentationPercent(value: number | null) {
  return value === null ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function segmentationIou(value: number | null) {
  return value === null ? "n/a" : value.toFixed(3);
}

/**
 * Macro IoU is reported to three decimals, so anything under half of the last
 * printed digit would render as "+0.000" -- a change the reader cannot see and
 * that no downstream comparison should act on. Treat it as no change.
 */
const SEGMENTATION_DELTA_EPSILON = 0.0005;

/**
 * Cases whose blocking signals all passed.
 *
 * `warn` means no blocking signal failed -- only a warning one did, which for
 * this suite is a quality threshold. Counting `pass` alone therefore reported
 * a contract failure for a contract-valid case that merely segmented poorly,
 * which the adjacent quality KPI already reports on its own terms.
 */
export function contractPassCount(run: BenchmarkRun) {
  return run.summary.pass + run.summary.warn;
}

export function segmentationDelta(
  current: number | null,
  baseline: number | null,
) {
  if (current === null || baseline === null) return undefined;
  const delta = current - baseline;
  if (Math.abs(delta) < SEGMENTATION_DELTA_EPSILON)
    return "no change vs baseline";
  return `${delta > 0 ? "+" : "−"}${Math.abs(delta).toFixed(3)} vs baseline`;
}

export interface SegmentationComparisonRow {
  asset_id: string;
  name: string;
  baseline: number | null;
  current: number | null;
  delta: number | null;
}

/**
 * Pair the two runs by asset so a mean shift can be traced to the cases that
 * caused it. Assets present in only one run still appear, because a case that
 * newly failed to score is exactly the kind of regression a mean hides.
 */
export function segmentationComparisonRows(
  current: BenchmarkRun,
  baseline: BenchmarkRun,
): SegmentationComparisonRow[] {
  const baselineAssets = new Map(
    baseline.assets.map((asset) => [asset.asset_id, asset]),
  );
  const rows: SegmentationComparisonRow[] = current.assets.map((asset) => {
    const before =
      baselineAssets.get(asset.asset_id)?.mesh_segmentation_evaluation
        ?.macro_iou ?? null;
    const after = asset.mesh_segmentation_evaluation?.macro_iou ?? null;
    return {
      asset_id: asset.asset_id,
      name: asset.name,
      baseline: before,
      current: after,
      delta: before === null || after === null ? null : after - before,
    };
  });
  const currentIds = new Set(current.assets.map((asset) => asset.asset_id));
  for (const asset of baseline.assets) {
    if (currentIds.has(asset.asset_id)) continue;
    rows.push({
      asset_id: asset.asset_id,
      name: asset.name,
      baseline: asset.mesh_segmentation_evaluation?.macro_iou ?? null,
      current: null,
      delta: null,
    });
  }
  // An asset the baseline scored and this run did not has no delta, but it is
  // the regression the docstring above says a mean hides -- so it sorts first,
  // not into the middle of the table as an apparent no-change.
  const rank = (row: SegmentationComparisonRow) =>
    row.delta === null && row.baseline !== null ? 0 : 1;
  return rows.sort(
    (first, second) =>
      rank(first) - rank(second) || (first.delta ?? 0) - (second.delta ?? 0),
  );
}

function SegmentationComparisonTable({
  current,
  baseline,
}: {
  current: BenchmarkRun;
  baseline: BenchmarkRun;
}) {
  const rows = segmentationComparisonRows(current, baseline);
  if (!rows.length) return null;
  return (
    <div className="segmentation-comparison-table">
      <table>
        <caption>
          Per-asset macro IoU, {baseline.run_id} → {current.run_id}
        </caption>
        <thead>
          <tr>
            <th scope="col">Asset</th>
            <th scope="col">Before</th>
            <th scope="col">After</th>
            <th scope="col">Δ</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.asset_id}>
              <th scope="row">{row.name}</th>
              <td>{segmentationIou(row.baseline)}</td>
              <td>{segmentationIou(row.current)}</td>
              <td
                className={cx(
                  "segmentation-delta",
                  row.delta !== null &&
                    row.delta > SEGMENTATION_DELTA_EPSILON &&
                    "is-better",
                  row.delta !== null &&
                    row.delta < -SEGMENTATION_DELTA_EPSILON &&
                    "is-worse",
                )}
              >
                {row.delta === null
                  ? "n/a"
                  : `${row.delta > 0 ? "+" : "−"}${Math.abs(row.delta).toFixed(3)}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MeshSegmentationStrip({
  run,
  baselineRuns,
  baselineRunId,
  onSelectBaseline,
}: {
  run: BenchmarkRun;
  baselineRuns: BenchmarkRun[];
  baselineRunId: string;
  onSelectBaseline: (runId: string) => void;
}) {
  const evaluation = run.mesh_segmentation_evaluation;
  if (!evaluation) return null;
  const baselineRun = baselineRuns.find(
    (candidate) => candidate.run_id === baselineRunId,
  );
  const baseline = baselineRun?.mesh_segmentation_evaluation ?? null;
  return (
    <section className="material-evaluation-summary">
      <div className="material-evaluation-heading">
        <div>
          <h2>Segmentation quality</h2>
          <p>
            Strict artifact-contract verdicts and label-invariant PartObjaverse
            partition quality are reported independently.
          </p>
        </div>
        <div className="segmentation-compare">
          <label htmlFor="segmentation-baseline">Compare against</label>
          <select
            id="segmentation-baseline"
            value={baselineRunId}
            onChange={(event) => onSelectBaseline(event.target.value)}
          >
            <option value="">No baseline</option>
            {baselineRuns
              .filter((candidate) => candidate.mesh_segmentation_evaluation)
              .map((candidate) => (
                <option key={candidate.run_id} value={candidate.run_id}>
                  {candidate.run_id}
                </option>
              ))}
          </select>
          <Badge kind="outline">Hungarian label matching</Badge>
        </div>
      </div>
      <div className="evaluation-kpi-strip">
        <KpiCard
          label="Contract passes"
          value={`${contractPassCount(run)} / ${evaluation.asset_count}`}
          color={
            contractPassCount(run) === evaluation.asset_count ? "green" : "red"
          }
          detail="Blocking signals only; quality is the next card"
        />
        <KpiCard
          label="Scoreable assets"
          value={`${evaluation.scoreable_asset_count} / ${evaluation.asset_count}`}
          color={
            evaluation.scoreable_asset_count === evaluation.asset_count
              ? "green"
              : "yellow"
          }
        />
        <KpiCard
          label="Quality passes"
          value={`${evaluation.quality_pass_asset_count} / ${evaluation.scoreable_asset_count}`}
          color={
            evaluation.quality_pass_asset_count ===
              evaluation.scoreable_asset_count &&
            evaluation.scoreable_asset_count > 0
              ? "green"
              : "yellow"
          }
          detail="Passes face-accuracy and macro-IoU signals"
        />
        <KpiCard
          label="Mean face accuracy"
          value={segmentationPercent(evaluation.mean_face_accuracy)}
          color={evaluationColor(
            evaluation.mean_face_accuracy === null
              ? null
              : evaluation.mean_face_accuracy * 100,
          )}
          detail={segmentationDelta(
            evaluation.mean_face_accuracy,
            baseline?.mean_face_accuracy ?? null,
          )}
        />
        <KpiCard
          label="Mean macro IoU"
          value={segmentationIou(evaluation.mean_macro_iou)}
          color={evaluationColor(
            evaluation.mean_macro_iou === null
              ? null
              : evaluation.mean_macro_iou * 100,
          )}
          detail={segmentationDelta(
            evaluation.mean_macro_iou,
            baseline?.mean_macro_iou ?? null,
          )}
        />
        <KpiCard
          label="Final exports"
          value={`${evaluation.final_export_asset_count} / ${evaluation.asset_count}`}
          color={
            evaluation.final_export_asset_count === evaluation.asset_count
              ? "green"
              : "yellow"
          }
        />
      </div>
      {baselineRun && (
        <SegmentationComparisonTable current={run} baseline={baselineRun} />
      )}
    </section>
  );
}

function evaluationColor(score: number | null) {
  if (score === null) return "gray" as const;
  if (score >= 85) return "green" as const;
  if (score >= 70) return "blue" as const;
  if (score >= 40) return "yellow" as const;
  return "red" as const;
}

function evaluationValue(value: number | null, suffix = "") {
  if (value === null) return "n/a";
  const formatted = Number.isInteger(value) ? String(value) : value.toFixed(1);
  return `${formatted}${suffix}`;
}

function evaluationScoreValue(value: number | null) {
  return value === null ? "n/a" : `${evaluationValue(value)}/100`;
}

function meshCountFraction(unbound: number | null, total: number | null) {
  if (unbound === null || total === null) return "n/a";
  return `${unbound.toLocaleString("en-US")} / ${total.toLocaleString("en-US")}`;
}

function judgeModelLabel(model: string) {
  const pathParts = model.split("/").filter(Boolean);
  const modelName = pathParts[pathParts.length - 1] ?? model;
  if (/^gemini-3\.1-pro/i.test(modelName)) return "Gemini 3.1 Pro";
  return modelName.replace(/^gpt-/i, "GPT-");
}

function physicsErrorValue(value: number | null) {
  return value === null ? "n/a" : value.toFixed(3);
}

function physicsFactorValue(value: number | null) {
  return value === null ? "n/a" : `×${value.toFixed(2)}`;
}

function PhysicsEvaluationStrip({ run }: { run: BenchmarkRun }) {
  const evaluation = run.physics_evaluation;
  if (!evaluation) return null;
  return (
    <section className="material-evaluation-summary">
      <div className="material-evaluation-heading">
        <div>
          <h2>Physics property accuracy</h2>
          <p>
            Material family is classification accuracy; density, friction, and
            restitution are regression errors against PhysX-Mobility ground
            truth (lower is better).
          </p>
        </div>
        <Badge kind="outline">PhysX-Mobility ground truth</Badge>
      </div>
      <div className="evaluation-kpi-strip">
        <KpiCard
          label="Scored assets"
          value={`${evaluation.scored_asset_count} / ${evaluation.asset_count}`}
          color={
            evaluation.scored_asset_count === evaluation.asset_count
              ? "green"
              : "yellow"
          }
          detail="Assets with property scoring against ground truth"
        />
        <KpiCard
          label="Mean material accuracy"
          value={segmentationPercent(evaluation.mean_material_accuracy)}
          color={evaluationColor(
            evaluation.mean_material_accuracy === null
              ? null
              : evaluation.mean_material_accuracy * 100,
          )}
          detail="Coarse material-family classification"
        />
        <KpiCard
          label="Mean density error"
          value={physicsErrorValue(evaluation.mean_density_log_mae)}
          color={evaluation.mean_density_log_mae === null ? "gray" : "blue"}
          detail="MAE of ln(pred/gt), scale-normalized"
        />
        <KpiCard
          label="Mean friction error"
          value={physicsErrorValue(evaluation.mean_friction_mae)}
          color={evaluation.mean_friction_mae === null ? "gray" : "blue"}
          detail="Mean |error| over both coefficients"
        />
        <KpiCard
          label="Mean restitution error"
          value={physicsErrorValue(evaluation.mean_restitution_mae)}
          color={evaluation.mean_restitution_mae === null ? "gray" : "blue"}
          detail="Mean |error| against material lookup"
        />
        <KpiCard
          label="Full GT coverage"
          value={`${evaluation.gt_full_coverage_asset_count} / ${evaluation.scored_asset_count}`}
          color={
            evaluation.scored_asset_count > 0 &&
            evaluation.gt_full_coverage_asset_count ===
              evaluation.scored_asset_count
              ? "green"
              : "yellow"
          }
          detail="Every expected part matched by a prediction"
        />
      </div>
    </section>
  );
}

function MaterialEvaluationStrip({ run }: { run: BenchmarkRun }) {
  const evaluation = run.material_evaluation;
  if (!evaluation) return null;
  const scores = evaluation.mean_scores;
  const agreements = evaluation.mean_agreement_scores;
  const judgeLabels = evaluation.judges.map((judge) =>
    judgeModelLabel(judge.model),
  );
  const judgeTooltip = evaluation.judges
    .map(
      (judge) =>
        `Judge backend: ${judge.backend}\nModel identifier: ${judge.model}`,
    )
    .join("\n\n");

  return (
    <section className="material-evaluation-summary">
      <div className="material-evaluation-heading">
        <div>
          <h2>Reference-grounded evaluation</h2>
          <p>
            Visual scores use a 0–100 scale · {evaluation.evaluated_asset_count}
            /{evaluation.asset_count} assets
            {evaluation.acceptable_rate !== null && (
              <>
                {" "}
                · {evaluationValue(evaluation.acceptable_rate, "%")} of all
                assets scored 70 or higher
              </>
            )}{" "}
            · judge confidence{" "}
            {evaluationValue(
              evaluation.mean_judge_confidence === null
                ? null
                : evaluation.mean_judge_confidence * 100,
              "%",
            )}
          </p>
        </div>
        <Tooltip slotContent={`${judgeTooltip}\n\n${evaluation.caveat}`}>
          <Badge className="judge-model-badge" kind="outline">
            Judges: {judgeLabels.join(" + ")}
          </Badge>
        </Tooltip>
      </div>
      <div className="evaluation-kpi-strip">
        <KpiCard
          label="Overall assignment quality"
          value={evaluationScoreValue(scores.overall_assignment_quality_score)}
          color={evaluationColor(scores.overall_assignment_quality_score)}
          detail={`Agreement ${evaluationScoreValue(
            agreements.overall_assignment_quality_score,
          )}`}
        />
        <KpiCard
          label="Final mesh bindings"
          value={evaluationValue(
            evaluation.final_mesh_binding_coverage_score,
            "%",
          )}
          color={evaluationColor(evaluation.final_mesh_binding_coverage_score)}
        />
        <KpiCard
          label="Unbound final mesh prims"
          value={meshCountFraction(
            evaluation.final_unbound_visible_mesh_count,
            evaluation.final_visible_mesh_count,
          )}
          color={
            evaluation.final_unbound_visible_mesh_count === null
              ? "gray"
              : evaluation.final_unbound_visible_mesh_count === 0
                ? "green"
                : "yellow"
          }
        />
        <KpiCard
          label="Material identity score"
          value={evaluationScoreValue(scores.material_identity_score)}
          color={evaluationColor(scores.material_identity_score)}
          detail={`Agreement ${evaluationScoreValue(
            agreements.material_identity_score,
          )}`}
        />
        <KpiCard
          label="Color palette score"
          value={evaluationScoreValue(scores.color_palette_score)}
          color={evaluationColor(scores.color_palette_score)}
          detail={`Agreement ${evaluationScoreValue(
            agreements.color_palette_score,
          )}`}
        />
        <KpiCard
          label="Surface finish score"
          value={evaluationScoreValue(scores.surface_finish_score)}
          color={evaluationColor(scores.surface_finish_score)}
          detail={`Agreement ${evaluationScoreValue(
            agreements.surface_finish_score,
          )}`}
        />
        <KpiCard
          label="Material consistency score"
          value={evaluationScoreValue(scores.material_consistency_score)}
          color={evaluationColor(scores.material_consistency_score)}
          detail={`Agreement ${evaluationScoreValue(
            agreements.material_consistency_score,
          )}`}
        />
      </div>
    </section>
  );
}

function RunContentTabs({
  feedbackCount,
  selectedView,
  onSelectView,
}: {
  feedbackCount: number;
  selectedView: RunView;
  onSelectView: (view: RunView) => void;
}) {
  return (
    <nav className="run-content-tabs" aria-label="Run content" role="tablist">
      <button
        aria-selected={selectedView === "results"}
        className={cx(
          "run-content-tab",
          selectedView === "results" && "is-active",
        )}
        onClick={() => onSelectView("results")}
        role="tab"
        type="button"
      >
        Results
      </button>
      <button
        aria-selected={selectedView === "feedback"}
        className={cx(
          "run-content-tab",
          selectedView === "feedback" && "is-active",
        )}
        onClick={() => onSelectView("feedback")}
        role="tab"
        type="button"
      >
        Feedback <Badge kind="outline">{feedbackCount}</Badge>
      </button>
    </nav>
  );
}

function RunFeedbackView({
  workflow,
  run,
  feedback,
  exportNotice,
  onCopyIssueDigest,
  onDownloadJson,
  onOpenAsset,
}: {
  workflow: WorkflowSummary;
  run: BenchmarkRun;
  feedback: RunFeedbackEntry[];
  exportNotice: string | null;
  onCopyIssueDigest: () => void;
  onDownloadJson: () => void;
  onOpenAsset: (assetId: string) => void;
}) {
  const assetCount = new Set(feedback.map((entry) => entry.asset.asset_id))
    .size;

  return (
    <section className="run-feedback-view">
      <header className="run-feedback-header">
        <div>
          <div className="eyebrow">{workflow.name}</div>
          <h2>Run feedback</h2>
          <p>
            {feedback.length
              ? `${feedback.length} comments across ${assetCount} assets.`
              : "No feedback recorded for this run."}
          </p>
        </div>
        <div className="run-feedback-actions">
          <Button kind="secondary" onClick={onCopyIssueDigest}>
            Copy issue digest
          </Button>
          <Button kind="secondary" onClick={onDownloadJson}>
            Download JSON
          </Button>
        </div>
      </header>
      {exportNotice && (
        <p className="run-feedback-notice" role="status">
          {exportNotice}
        </p>
      )}
      {feedback.length ? (
        <div className="run-feedback-list">
          {feedback.map(({ asset, feedback: entry }) => (
            <article className="run-feedback-item" key={entry.id}>
              <div className="run-feedback-context">
                <button
                  className="run-feedback-asset"
                  onClick={() => onOpenAsset(asset.asset_id)}
                  type="button"
                >
                  <strong>{asset.name}</strong>
                  <code>{asset.asset_id}</code>
                </button>
                <Badge color={statusBadgeColor(asset.status)}>
                  {asset.status}
                </Badge>
              </div>
              <div className="feedback-entry-heading">
                <strong>{entry.reviewer}</strong>
                <time dateTime={entry.created_at}>
                  {formatDate(entry.created_at)}
                </time>
              </div>
              <p>{entry.comment}</p>
            </article>
          ))}
        </div>
      ) : (
        <div className="run-feedback-empty">
          Add feedback from any result to collect it here.
        </div>
      )}
      <footer className="run-feedback-footer">
        <span>Run ID</span>
        <code>{run.run_id}</code>
      </footer>
    </section>
  );
}

interface ToolbarProps {
  query: string;
  statusFilter: StatusFilter;
  onQueryChange: (value: string) => void;
  onStatusFilterChange: (value: StatusFilter) => void;
}

function Toolbar({
  query,
  statusFilter,
  onQueryChange,
  onStatusFilterChange,
}: ToolbarProps) {
  return (
    <section className="toolbar">
      <TextInput
        aria-label="Search assets"
        placeholder="Search assets, tags, families"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
      />
      <div className="segmented-row" aria-label="Status filter">
        {STATUS_FILTERS.map((item) => (
          <button
            key={item.value}
            className={cx(
              "segmented-button",
              statusFilter === item.value && "active",
            )}
            type="button"
            onClick={() => onStatusFilterChange(item.value)}
          >
            {item.label}
          </button>
        ))}
      </div>
    </section>
  );
}

interface AssetListProps {
  assets: AssetResult[];
  selectedAssetId: string;
  onSelect: (assetId: string) => void;
}

function meshQualityLabel(asset: AssetResult) {
  const evaluation = asset.mesh_segmentation_evaluation;
  if (!evaluation || evaluation.status === "not_scoreable")
    return "Not scoreable";
  return evaluation.status === "pass" ? "Quality pass" : "Below threshold";
}

function meshQualityColor(asset: AssetResult) {
  const status = asset.mesh_segmentation_evaluation?.status;
  if (status === "pass") return "green" as const;
  if (status === "below_threshold") return "yellow" as const;
  return "gray" as const;
}

function AssetList({ assets, selectedAssetId, onSelect }: AssetListProps) {
  if (!assets.length) {
    return (
      <section className="asset-grid empty-results">
        <p>No assets match the current filters.</p>
      </section>
    );
  }

  return (
    <section
      className="asset-grid"
      aria-label={`${assets.length} benchmark assets`}
    >
      {assets.map((asset) => {
        return (
          <button
            key={asset.asset_id}
            className={cx(
              "asset-tile",
              asset.asset_id === selectedAssetId && "is-selected",
            )}
            type="button"
            onClick={() => onSelect(asset.asset_id)}
          >
            <EvidencePreview asset={asset} />
            <span className="asset-tile-main">
              <span className="asset-tile-title">
                <strong>{asset.name}</strong>
                <Badge color={statusBadgeColor(asset.status)}>
                  {asset.status}
                </Badge>
              </span>
              <span className="asset-tile-subtitle">{asset.status_label}</span>
              {asset.material_evaluation && (
                <span className="asset-evaluation-score">
                  <Badge kind="outline">
                    {evaluationScoreValue(
                      asset.material_evaluation.scores
                        .overall_assignment_quality_score,
                    )}{" "}
                    overall score
                  </Badge>
                  {asset.material_evaluation.structural
                    .final_unbound_visible_mesh_count !== null && (
                    <Badge
                      color={
                        asset.material_evaluation.structural
                          .final_unbound_visible_mesh_count === 0
                          ? "green"
                          : "yellow"
                      }
                    >
                      {meshCountFraction(
                        asset.material_evaluation.structural
                          .final_unbound_visible_mesh_count,
                        asset.material_evaluation.structural
                          .final_visible_mesh_count,
                      )}{" "}
                      unbound
                    </Badge>
                  )}
                </span>
              )}
              {asset.mesh_segmentation_evaluation && (
                <span className="asset-evaluation-score">
                  <Badge color={meshQualityColor(asset)}>
                    {meshQualityLabel(asset)}
                  </Badge>
                  {asset.mesh_segmentation_evaluation.face_accuracy !==
                    null && (
                    <Badge kind="outline">
                      {segmentationPercent(
                        asset.mesh_segmentation_evaluation.face_accuracy,
                      )}{" "}
                      face accuracy
                    </Badge>
                  )}
                  {asset.mesh_segmentation_evaluation.macro_iou !== null && (
                    <Badge kind="outline">
                      {segmentationIou(
                        asset.mesh_segmentation_evaluation.macro_iou,
                      )}{" "}
                      mIoU
                    </Badge>
                  )}
                </span>
              )}
              {asset.physics_evaluation && (
                <span className="asset-evaluation-score">
                  <Badge
                    color={
                      asset.physics_evaluation.material_accuracy === null
                        ? "gray"
                        : asset.physics_evaluation.material_accuracy >= 0.5
                          ? "green"
                          : "yellow"
                    }
                  >
                    {segmentationPercent(
                      asset.physics_evaluation.material_accuracy,
                    )}{" "}
                    material
                  </Badge>
                  {asset.physics_evaluation.density_log_mae !== null && (
                    <Badge kind="outline">
                      {physicsErrorValue(
                        asset.physics_evaluation.density_log_mae,
                      )}{" "}
                      density MAE
                    </Badge>
                  )}
                </span>
              )}
              <span className="asset-tags">
                {asset.tags.slice(0, 2).map((tag) => (
                  <Badge key={tag} kind="outline">
                    {tag}
                  </Badge>
                ))}
              </span>
            </span>
          </button>
        );
      })}
    </section>
  );
}

function physicsPropertyPair(predicted: number | null, gt: number | null) {
  const format = (value: number | null) =>
    value === null
      ? "n/a"
      : Number.isInteger(value)
        ? value.toLocaleString("en-US")
        : value.toFixed(2);
  return `${format(predicted)} / ${format(gt)}`;
}

function PhysicsPartsTable({
  evaluation,
}: {
  evaluation: AssetPhysicsEvaluation;
}) {
  if (!evaluation.parts.length) return null;
  return (
    <div className="segmentation-comparison-table">
      <table>
        <caption>Per-part predictions vs. ground truth (pred / GT)</caption>
        <thead>
          <tr>
            <th scope="col">Prim</th>
            <th scope="col">Material</th>
            <th scope="col">Density kg/m³</th>
            <th scope="col">Friction s/d</th>
            <th scope="col">Restitution</th>
          </tr>
        </thead>
        <tbody>
          {evaluation.parts.map((part, partIndex) => (
            <tr key={`${part.prim_path}-${partIndex}`}>
              <th scope="row" title={part.prim_path}>
                {part.prim_path.split("/").filter(Boolean).pop() ??
                  part.prim_path}
              </th>
              <td
                className={cx(
                  "segmentation-delta",
                  part.material_match === true && "is-better",
                  part.material_match === false && "is-worse",
                )}
              >
                {part.predicted_material ?? "n/a"} /{" "}
                {part.gt_material ?? "no GT"}
              </td>
              <td>
                {physicsPropertyPair(part.predicted_density, part.gt_density)}
              </td>
              <td>
                {physicsPropertyPair(
                  part.predicted_static_friction,
                  part.gt_static_friction,
                )}{" "}
                ·{" "}
                {physicsPropertyPair(
                  part.predicted_dynamic_friction,
                  part.gt_dynamic_friction,
                )}
              </td>
              <td>
                {physicsPropertyPair(
                  part.predicted_restitution,
                  part.gt_restitution,
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface AssetDetailProps {
  asset: AssetResult | null;
  feedback: AssetFeedback[];
  feedbackDraft: FeedbackDraft;
  onFeedbackDraftChange: (patch: Partial<FeedbackDraft>) => void;
  onSubmitFeedback: () => void;
}

function AssetDetail({
  asset,
  feedback,
  feedbackDraft,
  onFeedbackDraftChange,
  onSubmitFeedback,
}: AssetDetailProps) {
  const [expandedEvidence, setExpandedEvidence] =
    useState<ExpandedEvidence | null>(null);

  useEffect(() => {
    setExpandedEvidence(null);
  }, [asset?.asset_id]);

  useEffect(() => {
    if (!expandedEvidence) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpandedEvidence(null);
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [expandedEvidence]);

  if (!asset) {
    return (
      <aside className="asset-detail">
        <p>No asset selected.</p>
      </aside>
    );
  }

  const evidence = [
    ...asset.references.map((item) => ({
      category: "Reference" as const,
      item,
    })),
    ...asset.renders.map((item) => ({ category: "Render" as const, item })),
  ];

  return (
    <>
      <aside className="asset-detail">
        <section className="detail-section detail-heading">
          <div>
            <div className="eyebrow">{asset.family}</div>
            <h2>{asset.name}</h2>
          </div>
          <Badge color={statusBadgeColor(asset.status)}>
            {asset.status_label}
          </Badge>
        </section>

        {asset.simready_validation && (
          <section className="detail-section asset-evaluation-summary">
            <div>
              <span>SimReady profile</span>
              <strong>{asset.simready_validation.profile_target}</strong>
            </div>
            <p>
              {asset.simready_validation.passed &&
              asset.simready_validation.evidence_verified === null
                ? "UNVERIFIED · The bundle reports a profile PASS, but no scored evidence verified it."
                : asset.simready_validation.passed
                  ? "PASS · The formal profile validation passed for this asset."
                : `${asset.simready_validation.status} · The formal profile validation did not pass.`}
            </p>
            {asset.simready_validation.failed_features.length ? (
              <div className="finding-list">
                {asset.simready_validation.failed_features.map((feature) => (
                  <div className="finding-item" key={feature.feature_id}>
                    <strong>{feature.feature_id}</strong>
                    <p>
                      {feature.requirements.length
                        ? `Failed requirements: ${feature.requirements.join(", ")}`
                        : "The validator did not report a requirement ID for this failed feature."}
                    </p>
                    {feature.messages.map((message) => (
                      <small key={message}>{message}</small>
                    ))}
                  </div>
                ))}
              </div>
            ) : !asset.simready_validation.passed ? (
              <small>
                {asset.simready_validation.failed_requirements.length
                  ? `Failed requirements without a mapped feature: ${asset.simready_validation.failed_requirements.join(", ")}`
                  : asset.simready_validation.errors.length
                    ? `Validator errors: ${asset.simready_validation.errors.join("; ")}`
                    : "No failed feature was reported by the validator."}
              </small>
            ) : (
              <small>No active feature failures.</small>
            )}
          </section>
        )}

        {asset.material_evaluation && (
          <section className="detail-section asset-evaluation-summary">
            <div>
              <span>Overall assignment quality</span>
              <strong>
                {evaluationScoreValue(
                  asset.material_evaluation.scores
                    .overall_assignment_quality_score,
                )}
              </strong>
            </div>
            <p>{asset.material_evaluation.rationale}</p>
            <small>
              Judge confidence{" "}
              {Math.round(asset.material_evaluation.confidence * 100)}%
              {asset.material_evaluation.agreement_scores
                .overall_assignment_quality_score !== null && (
                <>
                  {" "}
                  · judge agreement{" "}
                  {evaluationScoreValue(
                    asset.material_evaluation.agreement_scores
                      .overall_assignment_quality_score,
                  )}
                </>
              )}
              {asset.material_evaluation.structural
                .final_mesh_binding_coverage_score !== null && (
                <>
                  {" "}
                  · final mesh bindings{" "}
                  {Math.round(
                    asset.material_evaluation.structural
                      .final_mesh_binding_coverage_score,
                  )}
                  % (
                  {meshCountFraction(
                    asset.material_evaluation.structural
                      .final_unbound_visible_mesh_count,
                    asset.material_evaluation.structural
                      .final_visible_mesh_count,
                  )}{" "}
                  unbound)
                </>
              )}
            </small>
          </section>
        )}

        {asset.mesh_segmentation_evaluation && (
          <section className="detail-section asset-evaluation-summary">
            <div>
              <span>Segmentation quality</span>
              <strong>{meshQualityLabel(asset)}</strong>
            </div>
            {asset.mesh_segmentation_evaluation.status === "not_scoreable" ? (
              <p>No final face-label prediction was available for scoring.</p>
            ) : (
              <p>
                Face accuracy{" "}
                {segmentationPercent(
                  asset.mesh_segmentation_evaluation.face_accuracy,
                )}{" "}
                · macro IoU{" "}
                {segmentationIou(asset.mesh_segmentation_evaluation.macro_iou)}
              </p>
            )}
            <small>
              {asset.mesh_segmentation_evaluation.face_count === null
                ? "Ground-truth comparison unavailable"
                : `${asset.mesh_segmentation_evaluation.face_count.toLocaleString("en-US")} faces · ${asset.mesh_segmentation_evaluation.ground_truth_segment_count ?? "n/a"} ground-truth segments · ${asset.mesh_segmentation_evaluation.predicted_segment_count ?? "n/a"} predicted segments`}
            </small>
          </section>
        )}

        {asset.physics_evaluation && (
          <section className="detail-section asset-evaluation-summary">
            <div>
              <span>Physics property accuracy</span>
              <strong>
                {segmentationPercent(
                  asset.physics_evaluation.material_accuracy,
                )}{" "}
                material family
              </strong>
            </div>
            <p>
              Density MAE{" "}
              {physicsErrorValue(asset.physics_evaluation.density_log_mae)}
              {asset.physics_evaluation.density_median_error_factor !==
                null && (
                <>
                  {" "}
                  (median factor{" "}
                  {physicsFactorValue(
                    asset.physics_evaluation.density_median_error_factor,
                  )}
                  )
                </>
              )}{" "}
              · friction MAE{" "}
              {physicsErrorValue(asset.physics_evaluation.friction_mae)} ·
              restitution MAE{" "}
              {physicsErrorValue(asset.physics_evaluation.restitution_mae)}
            </p>
            <small>
              {asset.physics_evaluation.gt_parts_covered ?? "n/a"} /{" "}
              {asset.physics_evaluation.gt_parts_total ?? "n/a"} expected parts
              covered · {asset.physics_evaluation.predictions_count ?? "n/a"}{" "}
              predictions
              {asset.physics_evaluation.zero_mass_prims !== null &&
                asset.physics_evaluation.zero_mass_prims > 0 && (
                  <>
                    {" "}
                    · {asset.physics_evaluation.zero_mass_prims} zero-mass prims
                  </>
                )}
            </small>
            <PhysicsPartsTable evaluation={asset.physics_evaluation} />
          </section>
        )}

        {asset.material_evaluation?.judge_results.length ? (
          <section className="detail-section judge-breakdown">
            <h3>Independent judges</h3>
            {asset.material_evaluation.judge_results.map((result) => (
              <article key={`${result.judge.backend}-${result.judge.model}`}>
                <header>
                  <strong>{judgeModelLabel(result.judge.model)}</strong>
                  <span>
                    {evaluationScoreValue(
                      result.scores.overall_assignment_quality_score,
                    )}
                  </span>
                </header>
                <dl>
                  <div>
                    <dt>Identity</dt>
                    <dd>
                      {evaluationValue(result.scores.material_identity_score)}
                    </dd>
                  </div>
                  <div>
                    <dt>Palette</dt>
                    <dd>
                      {evaluationValue(result.scores.color_palette_score)}
                    </dd>
                  </div>
                  <div>
                    <dt>Finish</dt>
                    <dd>
                      {evaluationValue(result.scores.surface_finish_score)}
                    </dd>
                  </div>
                  <div>
                    <dt>Consistency</dt>
                    <dd>
                      {evaluationValue(
                        result.scores.material_consistency_score,
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Confidence</dt>
                    <dd>{evaluationValue(result.confidence * 100, "%")}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </section>
        ) : null}

        {asset.prompt && (
          <section className="detail-section benchmark-prompt">
            <h3>Prompt</h3>
            <p>{asset.prompt}</p>
          </section>
        )}

        <section className="detail-section">
          <div className="evidence-section-heading">
            <h3>Visual evidence</h3>
            <span>
              {asset.references.length} reference
              {asset.references.length === 1 ? "" : "s"} ·{" "}
              {asset.renders.length} render
              {asset.renders.length === 1 ? "" : "s"}
            </span>
          </div>
          {evidence.length ? (
            <div className="evidence-gallery">
              {evidence.map(({ category, item }) => (
                <EvidenceFrame
                  key={`${category}-${item.view}-${item.path}`}
                  category={category}
                  evidence={item}
                  asset={asset}
                  onOpen={() => setExpandedEvidence({ category, item })}
                  toneOverride={
                    category === "Reference" ? "reference" : undefined
                  }
                />
              ))}
            </div>
          ) : (
            <div className="evidence-unavailable">
              No visual evidence supplied.
            </div>
          )}
        </section>

        <section className="detail-section">
          <h3>Findings</h3>
          {asset.findings.length ? (
            <div className="finding-list">
              {asset.findings.map((finding, findingIndex) => (
                <div
                  className="finding-item"
                  key={`${finding.title}-${finding.severity}-${findingIndex}`}
                >
                  <Badge color={severityBadgeColor(finding.severity)}>
                    {finding.severity}
                  </Badge>
                  <strong>{finding.title}</strong>
                  <p>{finding.detail}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No findings recorded.</p>
          )}
        </section>

        <section className="detail-section metric-grid">
          {Object.entries(asset.metrics).map(([key, value]) => (
            <div className="metric-chip" key={key}>
              <span>{key.replace(/_/g, " ")}</span>
              <strong>{String(value)}</strong>
            </div>
          ))}
        </section>

        <section className="detail-section link-section">
          <h3>Artifacts</h3>
          <div className="artifact-list">
            {[...asset.reports, ...asset.trace_links].map((link) => (
              <Tooltip
                key={`${link.kind}-${link.path}`}
                slotContent={[
                  link.path,
                  link.media_type,
                  link.sha256 ? `sha256:${link.sha256}` : null,
                ]
                  .filter(Boolean)
                  .join("\n")}
              >
                <a
                  className="artifact-link"
                  href={link.path}
                  rel="noreferrer"
                  target="_blank"
                >
                  <span>
                    {link.label}
                    {link.sha256 && (
                      <small>sha256:{link.sha256.slice(0, 12)}</small>
                    )}
                  </span>
                  <Badge kind="outline">{link.kind}</Badge>
                </a>
              </Tooltip>
            ))}
          </div>
        </section>

        <section className="detail-section feedback-section">
          <h3>Feedback</h3>
          <div className="feedback-thread" aria-live="polite">
            {feedback.length ? (
              feedback.map((entry) => (
                <article className="feedback-entry" key={entry.id}>
                  <div className="feedback-entry-heading">
                    <strong>{entry.reviewer}</strong>
                    <time dateTime={entry.created_at}>
                      {formatDate(entry.created_at)}
                    </time>
                  </div>
                  <p>{entry.comment}</p>
                </article>
              ))
            ) : (
              <p className="muted">No feedback yet.</p>
            )}
          </div>
          <form
            className="feedback-form"
            onSubmit={(event) => {
              event.preventDefault();
              onSubmitFeedback();
            }}
          >
            <label>
              Reviewer
              <TextInput
                maxLength={100}
                value={feedbackDraft.reviewer}
                onChange={(event) =>
                  onFeedbackDraftChange({ reviewer: event.target.value })
                }
              />
            </label>
            <label className="full-field">
              Comment
              <TextArea
                maxLength={1_000}
                rows={3}
                value={feedbackDraft.comment}
                onChange={(event) =>
                  onFeedbackDraftChange({ comment: event.target.value })
                }
              />
            </label>
            <div className="feedback-actions">
              <Button
                color="brand"
                disabled={
                  !feedbackDraft.reviewer.trim() ||
                  !feedbackDraft.comment.trim()
                }
                type="submit"
              >
                Submit feedback
              </Button>
            </div>
          </form>
        </section>
      </aside>
      {expandedEvidence && (
        <EvidenceLightbox
          asset={asset}
          category={expandedEvidence.category}
          evidence={expandedEvidence.item}
          onClose={() => setExpandedEvidence(null)}
        />
      )}
    </>
  );
}

export function EvidenceFrame({
  category,
  evidence,
  asset,
  onOpen,
  toneOverride,
}: {
  category: ExpandedEvidence["category"];
  evidence: EvidenceItem;
  asset: AssetResult;
  onOpen: () => void;
  toneOverride?: AssetResult["preview"]["tone"];
}) {
  const isDiagnostic = category === "Reference" && evidence.diagnostic;
  return (
    <button
      aria-label={`View ${asset.name} ${evidence.view} ${
        isDiagnostic ? "diagnostic preview" : category.toLowerCase()
      } full size`}
      className="evidence-frame"
      onClick={onOpen}
      title={
        evidence.provenance ??
        (isDiagnostic
          ? "diagnostic preview — no validated OVRTX render provenance; not reference evidence"
          : undefined)
      }
      type="button"
    >
      <div className="evidence-frame-header">
        <span>
          {isDiagnostic ? "Diagnostic preview" : category}
          {evidence.renderer ? ` · ${evidence.renderer}` : ""}
        </span>
        <small>{evidence.view}</small>
      </div>
      <EvidencePreview
        asset={asset}
        controls={false}
        imageUrl={evidence.path}
        loading={category === "Reference" ? "eager" : "lazy"}
        mediaType={evidence.media_type}
        toneOverride={toneOverride}
      />
    </button>
  );
}

export function EvidenceLightbox({
  asset,
  category,
  evidence,
  onClose,
}: {
  asset: AssetResult;
  category: ExpandedEvidence["category"];
  evidence: EvidenceItem;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  const accessibleKind =
    category === "Reference" && evidence.diagnostic
      ? "diagnostic preview"
      : category.toLowerCase();

  useEffect(() => {
    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const dialog = dialogRef.current;
    if (!dialog) return;
    const focusable = Array.from(
      dialog.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    );
    (focusable[0] ?? dialog).focus();

    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      if (!focusable.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", trapFocus);
    return () => {
      document.removeEventListener("keydown", trapFocus);
      previouslyFocused?.focus();
    };
  }, []);

  return (
    <div
      className="evidence-lightbox-backdrop"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) onClose();
      }}
    >
      <section
        aria-label={`${asset.name} ${evidence.view} ${accessibleKind}`}
        aria-modal="true"
        className="evidence-lightbox"
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        <header className="evidence-lightbox-header">
          <div>
            <div className="eyebrow">
              {category === "Reference" && evidence.diagnostic
                ? "Diagnostic preview"
                : category}
            </div>
            <h2>
              {asset.name} · {evidence.view}
            </h2>
          </div>
          <Button autoFocus kind="secondary" onClick={onClose} size="small">
            Close
          </Button>
        </header>
        <div className="evidence-lightbox-media">
          {evidence.media_type === "video" ? (
            <video
              aria-label={`${asset.name} ${evidence.view} ${accessibleKind}`}
              autoPlay
              controls
              loop
              muted
              playsInline
              src={evidence.path}
            />
          ) : (
            <img
              alt={`${asset.name} ${evidence.view} ${accessibleKind}`}
              src={evidence.path}
            />
          )}
        </div>
        <footer className="evidence-lightbox-footer">
          <div className="evidence-lightbox-details">
            <span>
              {evidence.provenance ??
                (category === "Reference" && evidence.diagnostic
                  ? `${evidence.label} — diagnostic preview; no validated OVRTX render provenance`
                  : evidence.label)}
            </span>
            {evidenceSettingsLabel(evidence) && (
              <small>{evidenceSettingsLabel(evidence)}</small>
            )}
            {evidence.sha256 && <code>sha256:{evidence.sha256}</code>}
          </div>
          <div className="evidence-lightbox-links">
            {evidence.camera && (
              <Tooltip
                slotContent={`sha256:${evidence.camera.sha256 ?? "not reported"}`}
              >
                <a href={evidence.camera.path} rel="noreferrer" target="_blank">
                  Camera
                </a>
              </Tooltip>
            )}
            <a href={evidence.path} rel="noreferrer" target="_blank">
              Open original
            </a>
          </div>
        </footer>
      </section>
    </div>
  );
}

export function EvidencePreview({
  asset,
  controls = true,
  imageUrl,
  loading = "lazy",
  mediaType = "image",
  toneOverride,
}: {
  asset: AssetResult;
  controls?: boolean;
  imageUrl?: string;
  loading?: "eager" | "lazy";
  mediaType?: "image" | "video";
  toneOverride?: AssetResult["preview"]["tone"];
}) {
  const instanceId = useId();
  const gradientId = `metal-${instanceId.replace(/:/g, "")}`;
  const tone = toneOverride ?? asset.preview.tone;
  const previewUrl = imageUrl ?? asset.preview.image_url;
  if (previewUrl) {
    return (
      <div className={cx("evidence-preview", `tone-${tone}`)}>
        {mediaType === "video" ? (
          <video
            key={previewUrl}
            aria-label={`${asset.name} benchmark video evidence`}
            autoPlay={!prefersReducedMotion()}
            className="evidence-preview-image"
            controls={controls}
            loop
            muted
            playsInline
            preload={loading === "eager" ? "auto" : "metadata"}
            src={previewUrl}
          />
        ) : (
          <img
            key={previewUrl}
            alt={`${asset.name} benchmark render`}
            className="evidence-preview-image"
            loading={loading}
            src={previewUrl}
          />
        )}
      </div>
    );
  }

  return (
    <div className={cx("evidence-preview", `tone-${tone}`)}>
      <svg aria-label={`${asset.name} evidence preview`} viewBox="0 0 160 110">
        <defs>
          <linearGradient id={gradientId} x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%" stopColor="currentColor" stopOpacity="0.25" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0.68" />
          </linearGradient>
        </defs>
        <rect
          className="svg-floor"
          x="0"
          y="0"
          width="160"
          height="110"
          rx="6"
        />
        <g className="svg-shadow">
          <ellipse cx="80" cy="88" rx="52" ry="10" />
        </g>
        <PreviewShape variant={asset.preview.variant} gradientId={gradientId} />
      </svg>
    </div>
  );
}

function PreviewShape({
  variant,
  gradientId,
}: {
  variant: AssetResult["preview"]["variant"];
  gradientId: string;
}) {
  const gradient = `url(#${gradientId})`;
  if (variant === "agv") {
    return (
      <g>
        <rect
          className="svg-accent"
          x="31"
          y="45"
          width="98"
          height="28"
          rx="7"
        />
        <rect
          className="svg-body"
          x="43"
          y="30"
          width="76"
          height="23"
          rx="5"
          fill={gradient}
        />
        <circle className="svg-dark" cx="52" cy="75" r="8" />
        <circle className="svg-dark" cx="108" cy="75" r="8" />
      </g>
    );
  }
  if (variant === "pcb") {
    return (
      <g>
        <rect
          className="svg-accent"
          x="34"
          y="25"
          width="92"
          height="58"
          rx="5"
        />
        <rect
          className="svg-body"
          x="48"
          y="36"
          width="30"
          height="19"
          rx="2"
          fill={gradient}
        />
        <rect
          className="svg-body"
          x="86"
          y="35"
          width="24"
          height="12"
          rx="2"
          fill={gradient}
        />
        <circle className="svg-light" cx="53" cy="68" r="3" />
        <circle className="svg-light" cx="72" cy="68" r="3" />
        <circle className="svg-light" cx="96" cy="68" r="3" />
      </g>
    );
  }
  if (variant === "valve") {
    return (
      <g>
        <rect
          className="svg-body"
          x="48"
          y="49"
          width="64"
          height="27"
          rx="6"
          fill={gradient}
        />
        <circle className="svg-accent" cx="80" cy="48" r="20" />
        <rect
          className="svg-dark"
          x="73"
          y="17"
          width="14"
          height="24"
          rx="3"
        />
        <rect
          className="svg-light"
          x="37"
          y="58"
          width="86"
          height="8"
          rx="4"
        />
      </g>
    );
  }
  if (variant === "lightbulb") {
    return (
      <g>
        <circle className="svg-light" cx="80" cy="42" r="23" />
        <rect
          className="svg-body"
          x="67"
          y="61"
          width="26"
          height="19"
          rx="5"
          fill={gradient}
        />
        <rect className="svg-dark" x="69" y="76" width="22" height="8" rx="2" />
      </g>
    );
  }
  if (variant === "roller") {
    return (
      <g>
        <rect
          className="svg-body"
          x="35"
          y="43"
          width="90"
          height="29"
          rx="14"
          fill={gradient}
        />
        <circle className="svg-accent" cx="48" cy="58" r="15" />
        <circle className="svg-accent" cx="112" cy="58" r="15" />
        <line className="svg-stroke" x1="52" y1="58" x2="108" y2="58" />
      </g>
    );
  }
  if (variant === "caster") {
    return (
      <g>
        <path className="svg-body" d="M63 28h34l-7 31H70z" fill={gradient} />
        <circle className="svg-accent" cx="80" cy="70" r="18" />
        <circle className="svg-dark" cx="80" cy="70" r="8" />
      </g>
    );
  }
  if (variant === "hinge") {
    return (
      <g>
        <rect
          className="svg-body"
          x="38"
          y="39"
          width="35"
          height="40"
          rx="5"
          fill={gradient}
        />
        <rect
          className="svg-body"
          x="87"
          y="39"
          width="35"
          height="40"
          rx="5"
          fill={gradient}
        />
        <rect
          className="svg-accent"
          x="74"
          y="32"
          width="12"
          height="54"
          rx="5"
        />
        <circle className="svg-dark" cx="55" cy="52" r="4" />
        <circle className="svg-dark" cx="105" cy="66" r="4" />
      </g>
    );
  }
  if (variant === "pump") {
    return (
      <g>
        <circle className="svg-body" cx="76" cy="54" r="29" fill={gradient} />
        <rect
          className="svg-accent"
          x="95"
          y="44"
          width="33"
          height="19"
          rx="4"
        />
        <rect className="svg-dark" x="48" y="78" width="55" height="8" rx="4" />
      </g>
    );
  }
  if (variant === "sheet") {
    return (
      <g>
        <path
          className="svg-body"
          d="M47 29h65v18H84v34H61V47H47z"
          fill={gradient}
        />
        <circle className="svg-dark" cx="62" cy="38" r="4" />
        <circle className="svg-dark" cx="97" cy="38" r="4" />
      </g>
    );
  }
  if (variant === "gearbox") {
    return (
      <g>
        <rect
          className="svg-body"
          x="42"
          y="35"
          width="76"
          height="47"
          rx="8"
          fill={gradient}
        />
        <circle className="svg-accent" cx="80" cy="59" r="18" />
        <circle className="svg-dark" cx="80" cy="59" r="8" />
        <rect
          className="svg-light"
          x="56"
          y="28"
          width="48"
          height="12"
          rx="3"
        />
      </g>
    );
  }
  return (
    <g>
      <rect
        className="svg-body"
        x="48"
        y="32"
        width="64"
        height="50"
        rx="8"
        fill={gradient}
      />
      <circle className="svg-accent" cx="80" cy="57" r="17" />
    </g>
  );
}
