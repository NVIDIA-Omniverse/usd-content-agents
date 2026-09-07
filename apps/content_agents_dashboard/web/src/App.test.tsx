// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import App, {
  benchmarkRunResultStatus,
  segmentationDelta,
  contractPassCount,
  createFeedbackId,
  createIssueDigest,
  EvidenceLightbox,
  getRunForWorkflow,
  nextBaselineRunId,
  reasoningShare,
  segmentationComparisonRows,
  EvidenceFrame,
  EvidencePreview,
} from "./App";
import type {
  AssetResult,
  BenchmarkRun,
  EvidenceItem,
  WorkflowSummary,
} from "./types/benchmark";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the benchmark loading state before artifacts arrive", () => {
    const markup = renderToStaticMarkup(<App />);

    expect(markup).toContain("Loading benchmark bundle");
    expect(markup).toContain('class="empty-state"');
  });

  it("gives repeated fallback previews unique SVG gradient IDs", () => {
    const asset = {
      asset_id: "repeated-asset",
      name: "Repeated asset",
      prompt: null,
      status: "pass",
      status_label: "Pass",
      family: "test",
      tags: [],
      source_usd: "source.usda",
      output_usd: "output.usda",
      references: [],
      renders: [],
      reports: [],
      trace_links: [],
      metrics: {},
      findings: [],
      review: {
        reviewer: "",
        verdict: "unreviewed",
        severity: "none",
        score: null,
        tags: [],
        comment: "",
        needs_rerun: false,
        updated_at: null,
      },
      preview: { tone: "good", variant: "agv" },
    } satisfies AssetResult;

    const markup = renderToStaticMarkup(
      <>
        <EvidencePreview asset={asset} />
        <EvidencePreview asset={asset} />
      </>,
    );
    const gradientIds = Array.from(
      markup.matchAll(/<linearGradient id="([^"]+)"/g),
      (match) => match[1],
    );

    expect(gradientIds).toHaveLength(2);
    expect(new Set(gradientIds).size).toBe(2);
    for (const gradientId of gradientIds) {
      expect(markup).toContain(`fill="url(#${gradientId})"`);
    }
  });

  it("labels a provenance-less physics reference as Diagnostic preview", () => {
    // UI half of the fail-closed evidence rule: an unvalidated reference
    // must render as "Diagnostic preview", never as plain "Reference".
    const asset = {
      name: "Asset",
      preview: { tone: "good", variant: "generic" },
    } as unknown as AssetResult;
    const diagnostic: EvidenceItem = {
      label: "reference 1",
      view: "reference 1",
      path: "/img.png",
      kind: "reference",
      provenance: null,
      diagnostic: true,
    };
    const validated: EvidenceItem = {
      ...diagnostic,
      provenance: "reference render: ovrtx · usd sha256 abc · image sha256 def",
      diagnostic: false,
    };

    const diagnosticMarkup = renderToStaticMarkup(
      <>
        <EvidenceFrame
          asset={asset}
          category="Reference"
          evidence={diagnostic}
          onOpen={() => {}}
        />
        <EvidenceLightbox
          asset={asset}
          category="Reference"
          evidence={diagnostic}
          onClose={() => {}}
        />
      </>,
    );
    expect(diagnosticMarkup).toContain("Diagnostic preview");
    expect(diagnosticMarkup).not.toContain(">Reference<");
    expect(diagnosticMarkup).toContain(
      "no validated OVRTX render provenance",
    );

    const validatedMarkup = renderToStaticMarkup(
      <EvidenceFrame
        asset={asset}
        category="Reference"
        evidence={validated}
        onOpen={() => {}}
      />,
    );
    expect(validatedMarkup).toContain(">Reference<");
    expect(validatedMarkup).not.toContain("Diagnostic preview");
    expect(validatedMarkup).toContain("usd sha256 abc");
  });

  it("renders turntable evidence as inline video", () => {
    const asset = {
      asset_id: "video-asset",
      name: "Video asset",
      prompt: null,
      status: "pass",
      status_label: "Pass",
      family: "test",
      tags: [],
      source_usd: "source.usda",
      output_usd: "output.usda",
      references: [],
      renders: [],
      reports: [],
      trace_links: [],
      metrics: {},
      findings: [],
      review: {
        reviewer: "",
        verdict: "unreviewed",
        severity: "none",
        score: null,
        tags: [],
        comment: "",
        needs_rerun: false,
        updated_at: null,
      },
      preview: { tone: "good", variant: "agv" },
    } satisfies AssetResult;

    const markup = renderToStaticMarkup(
      <EvidencePreview
        asset={asset}
        imageUrl="/artifacts/final-turntable.mp4"
        mediaType="video"
      />,
    );

    expect(markup).toContain("<video");
    expect(markup).toContain('src="/artifacts/final-turntable.mp4"');
    expect(markup).toContain("controls");
    expect(markup).not.toContain("<img");
  });

  it("supports non-interactive video thumbnails inside evidence buttons", () => {
    const asset = {
      asset_id: "video-thumbnail",
      name: "Video thumbnail",
      preview: { tone: "good", variant: "agv" },
    } as AssetResult;

    const markup = renderToStaticMarkup(
      <EvidencePreview
        asset={asset}
        controls={false}
        imageUrl="/artifacts/final-turntable.mp4"
        mediaType="video"
      />,
    );

    expect(markup).toContain("<video");
    expect(markup).not.toContain("controls");
  });

  it("escapes reviewer comments in issue digests", () => {
    const workflow = {
      id: "material-agentic",
      name: "Material Agentic",
      description: "test",
      run_ids: ["run"],
    } satisfies WorkflowSummary;
    const run = {
      run_id: "run",
      label: "Run",
      branch: "main",
      commit: "abcdef12",
      suite_id: "material-agentic",
    } as BenchmarkRun;
    const asset = {
      asset_id: "asset",
      name: "Asset",
      status_label: "Pass",
      findings: [],
    } as unknown as AssetResult;

    const digest = createIssueDigest(workflow, run, [
      {
        asset,
        feedback: {
          id: "feedback",
          reviewer: "reviewer",
          comment: "**bold** [link](https://example.test) <details>",
          created_at: "2026-07-22T00:00:00Z",
        },
      },
    ]);

    expect(digest).toContain(
      "\\*\\*bold\\*\\* \\[link\\]\\(https://example\\.test\\) \\<details\\>",
    );
    expect(digest).not.toContain("**bold**");
  });

  it("generates distinct secure fallback feedback IDs without module state", () => {
    let seed = 0;
    vi.stubGlobal("crypto", {
      getRandomValues: (words: Uint32Array) => {
        words.fill((seed += 1));
        return words;
      },
    });
    vi.spyOn(Date, "now").mockReturnValue(123);

    expect(createFeedbackId()).not.toBe(createFeedbackId());
  });

  it("fails closed when secure random generation is unavailable", () => {
    vi.stubGlobal("crypto", undefined);

    expect(() => createFeedbackId()).toThrow(
      "Secure random generation is unavailable.",
    );
  });

  it("reports a scored failing suite as fail rather than infrastructure error", () => {
    const run = {
      scored: true,
      run_status: "failed",
      summary: {
        pass: 0,
        warn: 0,
        fail: 12,
        error: 0,
        skipped: 0,
      },
    } as BenchmarkRun;

    expect(benchmarkRunResultStatus(run)).toBe("fail");
  });
});

describe("segmentationDelta", () => {
  it("reports a signed delta against the baseline", () => {
    expect(segmentationDelta(0.657, 0.556)).toBe("+0.101 vs baseline");
    expect(segmentationDelta(0.5, 0.6)).toBe("−0.100 vs baseline");
  });

  it("calls out an unchanged metric rather than showing +0.000", () => {
    expect(segmentationDelta(0.556, 0.556)).toBe("no change vs baseline");
  });

  it("shows nothing when either run lacks the metric", () => {
    expect(segmentationDelta(0.5, null)).toBeUndefined();
    expect(segmentationDelta(null, 0.5)).toBeUndefined();
  });
});

describe("segmentationComparisonRows", () => {
  function runWith(
    runId: string,
    scores: Array<[string, number | null]>,
  ): BenchmarkRun {
    return {
      run_id: runId,
      assets: scores.map(([assetId, macroIou]) => ({
        asset_id: assetId,
        name: assetId,
        mesh_segmentation_evaluation:
          macroIou === null ? undefined : { macro_iou: macroIou },
      })),
    } as unknown as BenchmarkRun;
  }

  it("pairs assets by id and sorts regressions first", () => {
    const rows = segmentationComparisonRows(
      runWith("after", [
        ["gain", 0.9],
        ["loss", 0.2],
      ]),
      runWith("before", [
        ["gain", 0.5],
        ["loss", 0.8],
      ]),
    );

    expect(rows.map((row) => row.asset_id)).toEqual(["loss", "gain"]);
    expect(rows.map((row) => row.delta ?? 0)[0]).toBeCloseTo(-0.6);
    expect(rows.map((row) => row.delta ?? 0)[1]).toBeCloseTo(0.4);
  });

  it("keeps an asset that only the baseline scored so a new failure is visible", () => {
    const rows = segmentationComparisonRows(
      runWith("after", [["dropped", null]]),
      runWith("before", [["dropped", 0.7]]),
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      baseline: 0.7,
      current: null,
      delta: null,
    });
  });

  it("sorts an asset that stopped scoring ahead of measured regressions", () => {
    // It has no delta, so a plain numeric sort placed it mid-table as an
    // apparent no-change -- the exact regression this table exists to surface.
    const rows = segmentationComparisonRows(
      runWith("after", [
        ["gain", 0.9],
        ["loss", 0.2],
        ["stopped", null],
      ]),
      runWith("before", [
        ["gain", 0.5],
        ["loss", 0.8],
        ["stopped", 0.7],
      ]),
    );

    expect(rows.map((row) => row.asset_id)).toEqual([
      "stopped",
      "loss",
      "gain",
    ]);
  });

  it("keeps an asset missing from the baseline run entirely", () => {
    const rows = segmentationComparisonRows(
      runWith("after", [["kept", 0.4]]),
      runWith("before", [["retired", 0.6]]),
    );

    expect(rows.map((row) => row.asset_id).sort()).toEqual(["kept", "retired"]);
  });
});

describe("reasoningShare", () => {
  it("reports reasoning tokens as a share of output", () => {
    expect(reasoningShare(194470, 889907)).toBe(
      "194.5K reasoning (22% of output)",
    );
  });

  it("omits the share when output is unknown or zero", () => {
    expect(reasoningShare(1200, null)).toBe("1.2K reasoning");
    expect(reasoningShare(1200, 0)).toBe("1.2K reasoning");
  });

  it("shows nothing when the run never reported reasoning tokens", () => {
    expect(reasoningShare(null, 889907)).toBeUndefined();
  });
});

describe("contractPassCount", () => {
  it("counts a case that only missed a quality threshold as contract-valid", () => {
    // `warn` means no blocking signal failed, so the contract held; the
    // segmentation was merely poor, which the quality KPI reports separately.
    const run = {
      summary: { pass: 3, warn: 8, fail: 1, error: 0 },
    } as BenchmarkRun;

    expect(contractPassCount(run)).toBe(11);
  });

  it("excludes cases with a blocking failure", () => {
    const run = {
      summary: { pass: 0, warn: 0, fail: 12, error: 0 },
    } as BenchmarkRun;

    expect(contractPassCount(run)).toBe(0);
  });
});

describe("getRunForWorkflow", () => {
  const run = (id: string, status: string, scored = true) =>
    ({
      run_id: id,
      workflow_id: "mesh-segmentation",
      run_status: status,
      scored,
      created_at: `2026-08-0${id.slice(-1)}T00:00:00Z`,
    }) as unknown as BenchmarkRun;

  it("does not default to a run that has not finished", () => {
    // A failed or still-running run became the representative view for every
    // workflow, showing stale renders and misleading pass counts.
    const runs = [
      run("r3", "failed"),
      run("r2", "complete"),
      run("r1", "running"),
    ];

    expect(getRunForWorkflow(runs, "mesh-segmentation", "")!.run_id).toBe("r2");
  });

  it("still shows something when no run has finished", () => {
    // The fallback is why requiring completeness costs no visibility.
    const runs = [run("r2", "failed"), run("r1", "failed")];

    expect(getRunForWorkflow(runs, "mesh-segmentation", "")).not.toBeNull();
  });

  it("honours an explicit selection regardless of status", () => {
    const runs = [run("r2", "complete"), run("r1", "failed")];

    expect(getRunForWorkflow(runs, "mesh-segmentation", "r1")!.run_id).toBe(
      "r1",
    );
  });
});

describe("nextBaselineRunId", () => {
  it("keeps a deliberately chosen baseline when another run is opened", () => {
    // Comparing several runs against one baseline is the whole workflow;
    // clearing on every selection silently discarded that choice.
    expect(nextBaselineRunId("merge-ab-before", "merge-ab-after")).toBe(
      "merge-ab-before",
    );
  });

  it("drops the baseline when that run becomes the selected run", () => {
    // A run cannot be its own baseline -- the strip filters it out of the
    // candidates, so keeping it would leave a dangling selection.
    expect(nextBaselineRunId("merge-ab-before", "merge-ab-before")).toBe("");
  });

  it("leaves an unset baseline unset", () => {
    expect(nextBaselineRunId("", "merge-ab-after")).toBe("");
  });
});

describe("EvidenceLightbox", () => {
  it("shows exact OVRTX settings, digest, and camera evidence", () => {
    const imageSha256 = "a".repeat(64);
    const cameraSha256 = "b".repeat(64);
    const markup = renderToStaticMarkup(
      <EvidenceLightbox
        asset={{ name: "Fixture valve" } as AssetResult}
        category="Render"
        evidence={{
          label: "Front · OVRTX",
          view: "Front",
          path: "/benchmark-data/run/bundle/assets/valve/front.png",
          kind: "render",
          sha256: imageSha256,
          renderer: "OVRTX",
          settings: {
            render_quality: "final",
            ovrtx_render_mode: "pt",
            ovrtx_num_sensor_updates: 500,
            active_aov: "LdrColor",
            width: 640,
            height: 640,
            fallback: false,
            elapsed_seconds: 12.5,
          },
          camera: {
            label: "Front camera",
            path: "/benchmark-data/run/bundle/assets/valve/front_camera.json",
            kind: "ovrtx_camera",
            sha256: cameraSha256,
          },
        }}
        onClose={() => undefined}
      />,
    );

    expect(markup).toContain(
      "OVRTX · final · pt · LdrColor · 500 sensor updates · 640x640 · 12.5s render",
    );
    expect(markup).toContain(`sha256:${imageSha256}`);
    expect(markup).toContain("front_camera.json");
    expect(markup).toContain(`sha256:${cameraSha256}`);
  });
});
