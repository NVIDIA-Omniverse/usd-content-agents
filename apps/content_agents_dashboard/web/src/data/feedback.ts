// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AssetFeedback } from "../types/benchmark";

export const MAX_STORED_FEEDBACK_ENTRIES = 2_000;
export const MAX_STORED_FEEDBACK_CODE_UNITS = 2_000_000;

function feedbackTimestamp(value: string) {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : null;
}

export function compareFeedbackNewestFirst(
  left: AssetFeedback,
  right: AssetFeedback,
) {
  const leftTime = feedbackTimestamp(left.created_at);
  const rightTime = feedbackTimestamp(right.created_at);
  if (leftTime === rightTime) return 0;
  if (leftTime === null) return 1;
  if (rightTime === null) return -1;
  return rightTime - leftTime;
}

export function limitStoredFeedback(
  feedback: Record<string, AssetFeedback[]>,
  maxEntries = MAX_STORED_FEEDBACK_ENTRIES,
  maxCodeUnits = MAX_STORED_FEEDBACK_CODE_UNITS,
) {
  const entries = Object.entries(feedback).flatMap(([key, values]) =>
    values.map((entry) => ({ key, entry })),
  );
  if (
    entries.length <= maxEntries &&
    JSON.stringify(feedback).length <= maxCodeUnits
  ) {
    return { feedback, dropped: 0 };
  }

  entries.sort((left, right) =>
    compareFeedbackNewestFirst(left.entry, right.entry),
  );
  const limited: Record<string, AssetFeedback[]> = {};
  let estimatedCodeUnits = 2;
  let retained = 0;
  for (const { key, entry } of entries) {
    if (retained >= maxEntries) break;
    const entryCodeUnits =
      JSON.stringify(key).length + JSON.stringify(entry).length + 4;
    if (estimatedCodeUnits + entryCodeUnits > maxCodeUnits) continue;
    (limited[key] ??= []).push(entry);
    estimatedCodeUnits += entryCodeUnits;
    retained += 1;
  }
  return { feedback: limited, dropped: entries.length - retained };
}

function isAssetFeedback(value: unknown): value is AssetFeedback {
  if (!value || typeof value !== "object") return false;
  const feedback = value as Record<string, unknown>;
  return (
    typeof feedback.id === "string" &&
    typeof feedback.reviewer === "string" &&
    typeof feedback.comment === "string" &&
    typeof feedback.created_at === "string"
  );
}

export function parseStoredFeedback(
  value: string | null,
): Record<string, AssetFeedback[]> {
  if (!value) return {};
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return Object.fromEntries(
      Object.entries(parsed).flatMap(([key, entries]) => {
        if (!Array.isArray(entries)) return [];
        const validEntries = entries.filter(isAssetFeedback);
        return validEntries.length ? [[key, validEntries]] : [];
      }),
    );
  } catch {
    return {};
  }
}

export function mergeFeedback(
  feedback: AssetFeedback[],
  legacyReview: AssetFeedback | null,
): AssetFeedback[] {
  if (!legacyReview) return feedback;
  const duplicate = feedback.some(
    (entry) =>
      entry.id === legacyReview.id ||
      (entry.reviewer === legacyReview.reviewer &&
        entry.comment === legacyReview.comment &&
        entry.created_at === legacyReview.created_at),
  );
  return duplicate ? feedback : [...feedback, legacyReview];
}
