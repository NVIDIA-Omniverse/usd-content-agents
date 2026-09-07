// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  adaptGeometryAsset,
  GEOMETRY_ARTIFACT_INTEGRITY_SCHEMA,
  GEOMETRY_ASSET_SCHEMA,
  STANDARD_BENCHMARK_BUNDLE_SCHEMA,
  validateGeometryRunContract,
} from "./geometryBenchmark";

const digest = (character: string) => character.repeat(64);

function contract() {
  const workflow = "geometry-cad-to-simready";
  const runId = "geometry-run-1";
  const asset = {
    asset_id: "valve",
    metrics: { runtime_seconds: 12.5, watertight: true },
    renders: { front: "assets/valve/renders/front.png" },
    geometry: {
      schema_version: GEOMETRY_ASSET_SCHEMA,
      benchmark_family: "cad_to_simready",
      outcome: "conditional",
      handoff_ready: "conditional",
      artifacts: [
        {
          kind: "content_agents_manifest",
          label: "Content Agents manifest",
          path: "assets/valve/content_agents_manifest.json",
          sha256: digest("a"),
          media_type: "application/json",
        },
        {
          kind: "geometry_validation_evidence",
          label: "Geometry validation evidence",
          path: "assets/valve/geometry_validation_evidence.json",
          sha256: digest("b"),
          media_type: "application/json",
        },
      ],
      render_evidence: [
        {
          view: "front",
          image: "assets/valve/renders/front.png",
          image_sha256: digest("c"),
          camera: "assets/valve/renders/front_camera.json",
          camera_sha256: digest("d"),
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
          title: "Runtime validation pending",
          detail: "Runtime cooking is downstream-owned.",
        },
      ],
    },
  };
  return {
    entry: {
      run_id: runId,
      workflow,
      geometry_integrity: {
        schema_version: GEOMETRY_ARTIFACT_INTEGRITY_SCHEMA,
        artifacts: [
          {
            path: "benchmark_run.json",
            sha256: digest("1"),
          },
          {
            path: "bundle/run.json",
            sha256: digest("2"),
          },
          {
            path: "bundle/assets/valve/content_agents_manifest.json",
            sha256: digest("a"),
          },
          {
            path: "bundle/assets/valve/geometry_validation_evidence.json",
            sha256: digest("b"),
          },
          {
            path: "bundle/assets/valve/renders/front.png",
            sha256: digest("c"),
          },
          {
            path: "bundle/assets/valve/renders/front_camera.json",
            sha256: digest("d"),
          },
          {
            path: "score/suite_result.json",
            sha256: digest("3"),
          },
        ],
      },
    },
    manifest: {
      schema_version: "content-agent-benchmark-run.v1",
      run_id: runId,
      workflow,
    },
    bundle: {
      schema_version: STANDARD_BENCHMARK_BUNDLE_SCHEMA,
      run_id: runId,
      workflow,
      assets: [asset],
    },
    score: {
      run_id: runId,
      cases: [
        {
          workflow,
          asset_id: "valve",
          metrics: { validation_count: 7 },
        },
      ],
    },
  };
}

describe("Geometry benchmark contract", () => {
  it("validates and adapts portable Geometry evidence without metric allowlists", () => {
    const fixture = contract();

    expect(() => validateGeometryRunContract(fixture)).not.toThrow();
    const adapted = adaptGeometryAsset({
      asset: fixture.bundle.assets[0]!,
      scoredCase: fixture.score.cases[0],
      resolveArtifact: (path) => `/benchmark-data/run/bundle/${path}`,
    });

    expect(adapted.workflowStatus).toBe("Conditional · Handoff Conditional");
    expect(adapted.family).toBe("CAD-to-SimReady");
    expect(adapted.metrics).toMatchObject({
      runtime_seconds: 12.5,
      watertight: true,
      validation_count: 7,
      geometry_outcome: "conditional",
      handoff_ready: "conditional",
      ovrtx_render_count: 1,
    });
    expect(adapted.reports.map((report) => report.kind)).toEqual([
      "content_agents_manifest",
      "geometry_validation_evidence",
    ]);
    expect(adapted.renders[0]).toMatchObject({
      renderer: "OVRTX",
      sha256: digest("c"),
      settings: {
        render_quality: "final",
        ovrtx_render_mode: "pt",
        ovrtx_num_sensor_updates: 500,
        active_aov: "LdrColor",
        elapsed_seconds: 12.5,
      },
      camera: { sha256: digest("d") },
    });
    expect(adapted.findings).toEqual([
      {
        severity: "warning",
        title: "Runtime validation pending",
        detail: "Runtime cooking is downstream-owned.",
      },
    ]);
  });

  it("rejects duplicate case identities and non-OVRTX render claims", () => {
    const duplicate = contract();
    duplicate.score.cases.push({ ...duplicate.score.cases[0]! });
    expect(() => validateGeometryRunContract(duplicate)).toThrow(
      "Duplicate Geometry score case ID: valve",
    );

    const wrongRenderer = contract();
    wrongRenderer.bundle.assets[0]!.geometry.render_evidence[0]!.renderer = "storm";
    expect(() => validateGeometryRunContract(wrongRenderer)).toThrow(
      "is not exact OVRTX evidence",
    );

    const fallbackRender = contract();
    fallbackRender.bundle.assets[0]!.geometry.render_evidence[0]!.fallback = true;
    expect(() => validateGeometryRunContract(fallbackRender)).toThrow(
      "is fallback output, not exact OVRTX evidence",
    );
  });

  it("rejects unsafe artifact paths and render-index drift", () => {
    const unsafe = contract();
    unsafe.bundle.assets[0]!.geometry.artifacts[0]!.path = "../private.json";
    expect(() => validateGeometryRunContract(unsafe)).toThrow(
      "unsafe geometry.artifacts[].path",
    );

    const emptySegment = contract();
    emptySegment.bundle.assets[0]!.geometry.artifacts[0]!.path =
      "assets//private.json";
    expect(() => validateGeometryRunContract(emptySegment)).toThrow(
      "unsafe geometry.artifacts[].path",
    );

    const drifted = contract();
    drifted.bundle.assets[0]!.renders.front = "assets/valve/renders/other.png";
    expect(() => validateGeometryRunContract(drifted)).toThrow(
      "Geometry render path mismatch",
    );
  });

  it("rejects missing or mismatched publication integrity", () => {
    const missing = contract();
    delete (missing.entry as { geometry_integrity?: unknown }).geometry_integrity;
    expect(() => validateGeometryRunContract(missing)).toThrow(
      "Geometry artifact integrity missing",
    );

    const mismatched = contract();
    const renderIntegrity = mismatched.entry.geometry_integrity.artifacts.find(
      (artifact) => artifact.path === "bundle/assets/valve/renders/front.png",
    )!;
    renderIntegrity.sha256 = digest("e");
    expect(() => validateGeometryRunContract(mismatched)).toThrow(
      "Geometry artifact integrity mismatch for bundle/assets/valve/renders/front.png",
    );

    const unbound = contract();
    unbound.entry.geometry_integrity.artifacts.push({
      path: "bundle/assets/valve/unbound.txt",
      sha256: digest("f"),
    });
    expect(() => validateGeometryRunContract(unbound)).toThrow(
      "Geometry artifact integrity coverage mismatch",
    );
  });

  it("requires integrity for standard consumer-reachable files", () => {
    const fixture = contract();
    const asset = fixture.bundle.assets[0]! as typeof fixture.bundle.assets[0] & {
      local_artifacts: Record<string, string>;
      references: Array<{ path: string }>;
      review_path: string;
    };
    asset.references = [{ path: "assets/valve/reference.png" }];
    asset.review_path = "assets/valve/review.json";
    asset.local_artifacts = {
      trace: "assets/valve/trace.json",
      run_dir: "assets/valve",
    };
    (
      fixture.score.cases[0]! as typeof fixture.score.cases[0] & {
        thumbnails: string[];
      }
    ).thumbnails = ["assets/valve/thumbnail.png"];

    expect(() => validateGeometryRunContract(fixture)).toThrow(
      "Geometry artifact integrity coverage mismatch",
    );

    fixture.entry.geometry_integrity.artifacts.push(
      {
        path: "bundle/assets/valve/reference.png",
        sha256: digest("4"),
      },
      {
        path: "bundle/assets/valve/review.json",
        sha256: digest("5"),
      },
      {
        path: "bundle/assets/valve/thumbnail.png",
        sha256: digest("6"),
      },
      {
        path: "bundle/assets/valve/trace.json",
        sha256: digest("7"),
      },
    );
    expect(() => validateGeometryRunContract(fixture)).not.toThrow();
  });
});
