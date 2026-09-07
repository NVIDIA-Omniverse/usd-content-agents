# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for identify-asset fail-closed persistence and failure flagging.

Complements ``test_tasks_identify_asset.py`` (retry serialization, provider
timeout normalization, auth handling): these tests pin down that exhausted
retries never persist a fabricated ``identification.json`` and that the
degraded parse fallback is flagged machine-readably for downstream scoring.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from physics_agent.tasks import identify_asset
from physics_agent.tasks.identify_asset import IdentifyAssetTask


class _FlakyVLM:
    """Bounded-timeout VLM stub that raises the given errors before succeeding."""

    has_bounded_request_timeout = True

    def __init__(self, errors: Sequence[BaseException], response_text: str) -> None:
        self._errors = list(errors)
        self.response_text = response_text
        self.calls = 0

    def generate(self, **kwargs: Any) -> str:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return self.response_text


_GOOD_RESPONSE = json.dumps(
    {
        "asset_type": "vehicle",
        "asset_subtype": "forklift",
        "asset_description": "A forklift",
        "confidence": "high",
        "reasoning": "Visible forks and mast",
    }
)


def _context(vlm: _FlakyVLM, tmp_path: Path) -> dict[str, Any]:
    return {
        "vlm": vlm,
        "composition_images": ["/tmp/view.png"],
        "output_dir": str(tmp_path),
    }


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real backoff delays in tests."""
    monkeypatch.setattr(identify_asset.time, "sleep", lambda _s: None)


def test_retry_exhaustion_fails_closed_without_persisting(tmp_path: Path) -> None:
    """Persistent transient errors exhaust retries and persist nothing."""
    vlm = _FlakyVLM(errors=[ConnectionError("boom")] * 5, response_text=_GOOD_RESPONSE)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        IdentifyAssetTask().run(_context(vlm, tmp_path))

    assert vlm.calls == 3
    # Fail closed: no fabricated identification.json is persisted.
    assert not (tmp_path / "identification.json").exists()


def test_transient_error_recovers_within_retry_budget(tmp_path: Path) -> None:
    """A flapping endpoint succeeds on a later attempt with a clean result."""
    vlm = _FlakyVLM(
        errors=[ConnectionError("reset"), ConnectionError("reset again")],
        response_text=_GOOD_RESPONSE,
    )

    context = IdentifyAssetTask().run(_context(vlm, tmp_path))

    assert vlm.calls == 3
    identification = context["identification"]
    assert identification["asset_type"] == "vehicle"
    assert "identification_failed" not in identification


def test_parse_failure_sets_identification_failed_flag(tmp_path: Path) -> None:
    """Unparseable VLM output is saved degraded with the failure flag."""
    vlm = _FlakyVLM(errors=[], response_text="not json at all")

    context = IdentifyAssetTask().run(_context(vlm, tmp_path))

    identification = context["identification"]
    assert identification["identification_failed"] is True
    assert identification["asset_type"] == "unknown"
    saved = json.loads((tmp_path / "identification.json").read_text())
    assert saved["identification_failed"] is True


def test_success_path_unchanged(tmp_path: Path) -> None:
    """A first-try success produces a clean, unflagged result."""
    vlm = _FlakyVLM(errors=[], response_text=_GOOD_RESPONSE)

    context = IdentifyAssetTask().run(_context(vlm, tmp_path))

    assert vlm.calls == 1
    identification = context["identification"]
    assert identification["asset_type"] == "vehicle"
    assert identification["asset_subtype"] == "forklift"
    assert "identification_failed" not in identification
    saved = json.loads((tmp_path / "identification.json").read_text())
    assert saved == identification
