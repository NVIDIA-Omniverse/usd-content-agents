# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the collision-worker soft address-space boundary."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from geometry_repair import process_limits


def test_soft_address_space_limit_preserves_inherited_hard_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[tuple[int, tuple[int, int]]] = []
    resource = SimpleNamespace(
        RLIMIT_AS=9,
        RLIM_INFINITY=-1,
        getrlimit=lambda _kind: (2048 * 1024 * 1024, 4096 * 1024 * 1024),
        setrlimit=lambda kind, limits: applied.append((kind, limits)),
    )
    monkeypatch.setitem(sys.modules, "resource", resource)

    process_limits.limit_address_space_soft(640)

    assert applied == [(9, (640 * 1024 * 1024, 4096 * 1024 * 1024))]
