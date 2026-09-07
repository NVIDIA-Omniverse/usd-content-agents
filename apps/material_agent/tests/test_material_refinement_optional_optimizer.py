# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the optional Material refinement optimizer dependency."""

from __future__ import annotations

import pytest
from world_understanding.optimization import TunableParam, get_runner, resolve_optimizer

from material_agent.material_refinement.runner import MaterialSearchSpace


def test_auto_optimizer_resolves_and_runs_with_refinement_extra() -> None:
    pytest.importorskip("botorch")
    optimizer = resolve_optimizer("auto")
    observed: list[float] = []

    get_runner(optimizer)(
        MaterialSearchSpace((TunableParam("value", 0.0, 1.0),)),
        lambda candidate: observed.append(candidate["value"]) or candidate["value"],
        max_trials=2,
        seed=42,
    )

    assert optimizer == "botorch"
    assert len(observed) == 2
    assert all(0.0 <= value <= 1.0 for value in observed)
