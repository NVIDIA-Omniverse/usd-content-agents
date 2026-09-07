# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for process resource-limit helpers."""

from __future__ import annotations

import pytest

from geometry_repair import process_limits


def test_temporary_cpu_affinity_restores_original_mask_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(process_limits.os, "sched_getaffinity", lambda _pid: {7, 3, 5})
    monkeypatch.setattr(
        process_limits.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((pid, set(cpus))),
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        with process_limits.temporary_cpu_affinity(1):
            assert calls == [(0, {3})]
            raise RuntimeError("backend failed")

    assert calls == [(0, {3}), (0, {3, 5, 7})]
