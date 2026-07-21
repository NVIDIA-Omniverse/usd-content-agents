# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a robot_id from upstream signals (asset_type, asset_subtype, USD path).

Strategy: build a flat token set from the inputs and check it against each
entry's ``robot_id`` and ``aliases``. Tokens are normalized (lowercase, with
hyphens/spaces collapsed to underscores) so ``"UR-10e"``, ``"ur 10e"``, and
``"ur10e"`` all match the same entry.
"""

from __future__ import annotations

import re
from pathlib import Path

from joint_agent.prompts.library import load_all


def _normalize(s: str) -> str:
    """Lowercase and strip all non-alphanumeric characters for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _candidate_tokens(
    asset_type: str | None,
    asset_subtype: str | None,
    usd_path: str | None,
) -> set[str]:
    """Build a normalized token set from matching signals."""
    tokens: set[str] = set()
    for raw in (asset_type, asset_subtype):
        if raw:
            tokens.add(_normalize(raw))
    if usd_path:
        stem = Path(usd_path).stem
        tokens.add(_normalize(stem))
    return tokens


def lookup_robot_id(
    asset_type: str | None = None,
    asset_subtype: str | None = None,
    usd_path: str | None = None,
) -> str | None:
    """Return the matching robot_id, or None if no entry matches.

    Resolution order, first hit wins:
      1. Exact match against any entry's ``robot_id`` (normalized).
      2. Exact match against any alias (normalized).
      3. Separator-bounded match in the USD filename stem using only
         ``robot_id`` and ``filename_aliases`` (longest robot_id match wins to
         avoid ``ur10`` shadowing ``ur10e``). This handles stems like
         ``so_arm_101_anonymous`` without matching arbitrary substrings such as
         ``redpanda`` or broad vendor aliases such as ``kuka``.

    Args:
        asset_type: Asset type from the identify_asset step (e.g. ``"robot_arm"``).
        asset_subtype: Asset subtype (e.g. ``"UR10e"``, ``"Franka Panda"``).
        usd_path: Path to the input USD file; the filename stem is searched.
    """
    tokens = _candidate_tokens(asset_type, asset_subtype, usd_path)
    if not tokens:
        return None

    entries = load_all()

    # Pass 1: exact matches against robot_id or aliases.
    for entry in entries:
        normalized_aliases = {_normalize(a) for a in entry.aliases}
        normalized_id = _normalize(entry.robot_id)
        if normalized_id in tokens or normalized_aliases & tokens:
            return entry.robot_id

    # Pass 2: separator-bounded containment in the USD filename stem.
    # Prefer longer matches (so "ur10e" beats "ur10").
    if usd_path:
        stem_parts = [
            p for p in re.split(r"[^a-z0-9]+", Path(usd_path).stem.lower()) if p
        ]
        candidates: list[tuple[int, str]] = []
        for entry in entries:
            keys = [entry.robot_id, *entry.filename_aliases]
            for raw_key in keys:
                key_parts = [p for p in re.split(r"[^a-z0-9]+", raw_key.lower()) if p]
                if not key_parts or len(key_parts) > len(stem_parts):
                    continue
                # Generic one-word aliases without digits must match the whole
                # stem. This lets panda.usda resolve while avoiding
                # panda_gripper.usda or kinova_gen3.usda false positives.
                if (
                    len(key_parts) == 1
                    and not any(ch.isdigit() for ch in key_parts[0])
                    and stem_parts != key_parts
                ):
                    continue
                for start in range(len(stem_parts) - len(key_parts) + 1):
                    if stem_parts[start : start + len(key_parts)] == key_parts:
                        candidates.append((len(_normalize(raw_key)), entry.robot_id))
                        break
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

    return None
