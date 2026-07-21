# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test parallel LLM conflict resolution in harmonize.

Uses synthetic predictions to create conflicts, then benchmarks
sequential vs parallel resolution.
"""

from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from material_agent.scene.harmonize import (
    _build_harmonize_prompt,
    _find_conflicts,
    _majority_vote_fallback,
    _resolve_conflicts,
    _resolve_single_group,
    apply_prim_remap,
)


def _make_prediction(prim_id: str, material: str, reasoning: str = "") -> dict:
    return {
        "id": prim_id,
        "materials": {
            "material": material,
            "original_response": f"<reasoning>{reasoning}</reasoning>",
        },
    }


def _make_conflict_predictions(
    n_groups: int = 10,
    members_per_group: int = 4,
) -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[int, list[str]]]:
    """Create n conflict groups as indexed predictions.

    Returns (predictions, conflicts, group_signals_map) matching the current API.
    """
    predictions: list[dict[str, Any]] = []
    conflicts: dict[int, list[int]] = {}
    group_signals_map: dict[int, list[str]] = {}

    for g in range(n_groups):
        rep_idx = len(predictions)
        members = []
        for m in range(members_per_group):
            mat = "Steel Brushed" if m % 2 == 0 else "Aluminum Anodized"
            reasoning = (
                f"The part has a brushed metallic surface with "
                f"{'visible grain' if m % 2 == 0 else 'anodized finish'}"
            )
            predictions.append(
                _make_prediction(
                    f"/Root/Assembly_{g}/Part_{m}",
                    mat,
                    reasoning,
                )
            )
            members.append(len(predictions) - 1)
        conflicts[rep_idx] = members
        group_signals_map[rep_idx] = ["name_template", "geometry"]
    return predictions, conflicts, group_signals_map


class TestBuildHarmonizePrompt:
    def test_builds_prompt_with_options(self):
        predictions, conflicts, signals_map = _make_conflict_predictions(1, 4)
        rep = next(iter(conflicts))
        members = [predictions[i] for i in conflicts[rep]]
        signals = signals_map[rep]
        prompt = _build_harmonize_prompt(members, signals)
        assert "Steel Brushed" in prompt
        assert "Aluminum Anodized" in prompt
        assert "unify" in prompt

    def test_deduplicates_materials(self):
        """Members with same material should be merged into one entry."""
        predictions, conflicts, signals_map = _make_conflict_predictions(1, 6)
        rep = next(iter(conflicts))
        members = [predictions[i] for i in conflicts[rep]]
        signals = signals_map[rep]
        prompt = _build_harmonize_prompt(members, signals)
        # Only 2 unique materials mentioned as separate options
        assert prompt.count('Material: "') == 2


class TestFindConflicts:
    def test_detects_disagreements(self):
        predictions, conflicts, _ = _make_conflict_predictions(3, 4)
        # All groups should be conflicts since half have Steel, half Aluminum
        all_groups = conflicts  # these ARE the conflicts
        found = _find_conflicts(all_groups, predictions)
        assert len(found) == 3

    def test_no_conflict_when_unanimous(self):
        predictions = [
            _make_prediction("/a", "Steel Brushed"),
            _make_prediction("/b", "Steel Brushed"),
        ]
        groups = {0: [0, 1]}
        found = _find_conflicts(groups, predictions)
        assert len(found) == 0


class TestResolveSingleGroup:
    def test_returns_remap_on_unify(self):
        predictions, conflicts, signals_map = _make_conflict_predictions(1, 4)
        rep = next(iter(conflicts))

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"action": "unify", "material": "Steel Brushed", "reason": "clear grain"}'
        )

        remap = _resolve_single_group(
            conflicts[rep], predictions, signals_map[rep], mock_llm
        )
        # Members with "Aluminum Anodized" should be remapped
        assert len(remap) == 2  # 2 out of 4 had different material
        for _prim_id, mat in remap.items():
            assert mat == "Steel Brushed"

    def test_returns_empty_on_keep(self):
        predictions, conflicts, signals_map = _make_conflict_predictions(1, 4)
        rep = next(iter(conflicts))

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"action": "keep", "reason": "different parts"}'
        )

        remap = _resolve_single_group(
            conflicts[rep], predictions, signals_map[rep], mock_llm
        )
        assert remap == {}

    def test_returns_empty_on_llm_failure(self):
        predictions, conflicts, signals_map = _make_conflict_predictions(1, 4)
        rep = next(iter(conflicts))

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API error")

        remap = _resolve_single_group(
            conflicts[rep], predictions, signals_map[rep], mock_llm
        )
        assert remap == {}


class TestResolveConflictsParallel:
    def test_parallel_resolves_all_groups(self):
        """All conflict groups should be resolved."""
        n_groups = 5
        predictions, conflicts, signals_map = _make_conflict_predictions(n_groups, 4)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"action": "unify", "material": "Steel Brushed", "reason": "grain"}'
        )

        with patch(
            "world_understanding.functions.models.chat_models.create_chat_model_from_config",
            return_value=mock_llm,
        ):
            remap = _resolve_conflicts(
                conflicts, predictions, signals_map, llm_config={"backend": "mock"}
            )

        # Each group has 2 members needing remap (the Aluminum ones)
        assert len(remap) == n_groups * 2
        assert mock_llm.invoke.call_count == n_groups

    def test_parallel_faster_than_sequential(self):
        """Conflict groups should invoke the LLM concurrently."""
        n_groups = 8
        sleep_time = 0.2
        predictions, conflicts, signals_map = _make_conflict_predictions(n_groups, 4)
        lock = threading.Lock()
        active_calls = 0
        max_active_calls = 0

        def slow_invoke(*args, **kwargs):
            nonlocal active_calls, max_active_calls
            with lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            try:
                time.sleep(sleep_time)
                return MagicMock(
                    content='{"action": "unify", "material": "Steel Brushed", "reason": "test"}'
                )
            finally:
                with lock:
                    active_calls -= 1

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = slow_invoke

        with patch(
            "world_understanding.functions.models.chat_models.create_chat_model_from_config",
            return_value=mock_llm,
        ):
            remap = _resolve_conflicts(
                conflicts, predictions, signals_map, llm_config={"backend": "mock"}
            )

        assert max_active_calls > 1
        assert len(remap) == n_groups * 2

    def test_single_group_no_threadpool(self):
        """Single group should skip thread pool overhead."""
        predictions, conflicts, signals_map = _make_conflict_predictions(1, 4)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"action": "unify", "material": "Steel Brushed", "reason": "test"}'
        )

        with patch(
            "world_understanding.functions.models.chat_models.create_chat_model_from_config",
            return_value=mock_llm,
        ):
            remap = _resolve_conflicts(
                conflicts, predictions, signals_map, llm_config={"backend": "mock"}
            )

        assert len(remap) == 2

    def test_partial_failure_continues(self):
        """If one group fails, others should still resolve."""
        n_groups = 4
        predictions, conflicts, signals_map = _make_conflict_predictions(n_groups, 4)

        call_count = 0

        def sometimes_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated API failure")
            return MagicMock(
                content='{"action": "unify", "material": "Steel Brushed", "reason": "test"}'
            )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = sometimes_fail

        with patch(
            "world_understanding.functions.models.chat_models.create_chat_model_from_config",
            return_value=mock_llm,
        ):
            remap = _resolve_conflicts(
                conflicts, predictions, signals_map, llm_config={"backend": "mock"}
            )

        # 3 out of 4 groups should succeed (each contributing 2 remaps)
        assert len(remap) == (n_groups - 1) * 2


class TestMajorityVoteFallback:
    def test_majority_wins(self):
        """Groups with clear majority should resolve without LLM."""
        predictions = [
            _make_prediction("/a", "Steel Brushed"),
            _make_prediction("/b", "Steel Brushed"),
            _make_prediction("/c", "Steel Brushed"),
            _make_prediction("/d", "Aluminum Anodized"),
        ]
        conflicts = {0: [0, 1, 2, 3]}
        remap = _majority_vote_fallback(conflicts, predictions)
        assert remap == {"/d": "Steel Brushed"}

    def test_tie_no_remap(self):
        """50/50 split with no majority should not remap."""
        predictions = [
            _make_prediction("/a", "Steel Brushed"),
            _make_prediction("/b", "Aluminum Anodized"),
        ]
        conflicts = {0: [0, 1]}
        remap = _majority_vote_fallback(conflicts, predictions)
        assert remap == {}

    def test_resolve_conflicts_falls_back_without_llm(self):
        """No LLM config should trigger majority vote fallback."""
        predictions = [
            _make_prediction("/a", "Steel Brushed"),
            _make_prediction("/b", "Steel Brushed"),
            _make_prediction("/c", "Steel Brushed"),
            _make_prediction("/d", "Aluminum Anodized"),
        ]
        conflicts = {0: [0, 1, 2, 3]}
        signals_map = {0: ["name_template"]}
        remap = _resolve_conflicts(conflicts, predictions, signals_map, llm_config=None)
        assert remap == {"/d": "Steel Brushed"}


class TestApplyPrimRemap:
    def test_applies_remap_and_tracks_original(self):
        preds = [
            _make_prediction("/a", "Steel Brushed"),
            _make_prediction("/b", "Aluminum Anodized"),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
            tmp = Path(f.name)

        remap = {"/b": "Steel Brushed"}
        updated = apply_prim_remap(tmp, remap, trusted_root=tmp.parent)
        assert updated == 1

        # Verify the file
        result = []
        for line in tmp.read_text().splitlines():
            if line.strip():
                result.append(json.loads(line))
        assert result[1]["materials"]["material"] == "Steel Brushed"
        assert result[1]["materials"]["harmonized_from"] == "Aluminum Anodized"
        tmp.unlink()
