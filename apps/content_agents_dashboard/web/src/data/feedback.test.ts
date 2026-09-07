// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import {
  compareFeedbackNewestFirst,
  limitStoredFeedback,
  mergeFeedback,
  parseStoredFeedback,
} from "./feedback";

const feedback = {
  id: "feedback-1",
  reviewer: "reviewer",
  comment: "Looks correct.",
  created_at: "2026-07-22T00:00:00Z",
};

describe("parseStoredFeedback", () => {
  it("keeps valid entries and drops malformed storage values", () => {
    expect(
      parseStoredFeedback(
        JSON.stringify({
          valid: [feedback],
          mixed: [feedback, { reviewer: "missing fields" }],
          invalid: "not an array",
        }),
      ),
    ).toEqual({ valid: [feedback], mixed: [feedback] });
    expect(parseStoredFeedback("{")).toEqual({});
    expect(parseStoredFeedback(JSON.stringify([]))).toEqual({});
  });
});

describe("mergeFeedback", () => {
  it("does not duplicate a migrated legacy review", () => {
    expect(mergeFeedback([feedback], feedback)).toEqual([feedback]);
    expect(
      mergeFeedback([feedback], { ...feedback, id: "legacy-asset" }),
    ).toEqual([feedback]);
  });

  it("retains a distinct legacy review", () => {
    const legacy = { ...feedback, id: "legacy-asset", comment: "Legacy note" };
    expect(mergeFeedback([feedback], legacy)).toEqual([feedback, legacy]);
  });
});

describe("feedback ordering and persistence limits", () => {
  it("sorts invalid or missing timestamps after dated feedback", () => {
    const old = { ...feedback, id: "old", created_at: "2026-07-21T00:00:00Z" };
    const missing = { ...feedback, id: "missing", created_at: "" };
    const invalid = { ...feedback, id: "invalid", created_at: "not-a-date" };

    expect(
      [missing, old, invalid, feedback]
        .sort(compareFeedbackNewestFirst)
        .map((entry) => entry.id),
    ).toEqual(["feedback-1", "old", "missing", "invalid"]);
  });

  it("retains the newest entries when browser persistence is bounded", () => {
    const old = { ...feedback, id: "old", created_at: "2026-07-20T00:00:00Z" };
    const newest = { ...feedback, id: "new", created_at: "2026-07-23T00:00:00Z" };
    const result = limitStoredFeedback({ first: [old, feedback], second: [newest] }, 2);

    expect(result.dropped).toBe(1);
    expect(Object.values(result.feedback).flat().map((entry) => entry.id)).toEqual([
      "new",
      "feedback-1",
    ]);
  });

  it("also bounds serialized feedback size", () => {
    const large = { ...feedback, comment: "x".repeat(500) };
    const result = limitStoredFeedback({ first: [large] }, 10, 100);

    expect(result).toEqual({ feedback: {}, dropped: 1 });
  });
});
