# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable worker operation allow-list tests."""

from __future__ import annotations

import pytest

from geometry_repair.models import RepairOperation
from geometry_repair.policy import (
    assert_worker_operations_enabled,
    worker_operations,
    worker_record,
)
from geometry_repair.worker_runner import (
    _WORKERS,
    _assert_worker_binding,
    _requested_operations,
)
from geometry_repair.workers.pmp_patch import _PMP_SPEC, PmpPatchWorker


def test_every_isolated_worker_declares_only_registered_operations() -> None:
    for name, worker in _WORKERS.items():
        assert worker.operations
        assert worker.operations <= worker_operations(name)
        assert_worker_operations_enabled(name, worker.operations)


def test_pmp_registry_records_weak_copyleft_transitive_boundary() -> None:
    assert "Eigen MPL-2.0" in worker_record("pmp_patch")["license"]


def test_unregistered_operation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unapproved operations"):
        assert_worker_operations_enabled("pmp_patch", "pmp_remesh_generated_patch")


def test_explicit_request_operation_is_checked_before_worker_execution() -> None:
    worker = PmpPatchWorker()
    operation = RepairOperation(
        operation_id="unapproved-pmp-remesh",
        worker="pmp_patch",
        implementation="test",
        parameters={"operation": "pmp_remesh_generated_patch"},
        issue_ids=["mesh:test"],
        drift_band="conservative",
        source_checkpoint="source.usda",
    )

    with pytest.raises(ValueError, match="unapproved operations"):
        assert_worker_operations_enabled(
            "pmp_patch",
            _requested_operations(operation, worker),
        )


def test_operation_worker_must_match_selected_runner_worker() -> None:
    operation = RepairOperation(
        operation_id="mismatched-worker",
        worker="pmp_patch",
        implementation="test",
        parameters={"operation": "pmp_fill_classified_hole"},
        issue_ids=["mesh:test"],
        drift_band="conservative",
        source_checkpoint="source.usda",
    )

    with pytest.raises(ValueError, match="does not match"):
        _assert_worker_binding(operation, "trimesh_conservative_cleanup")


def test_pmp_generated_patch_remesh_is_not_executable(tmp_path) -> None:
    assert _PMP_SPEC.operations == frozenset({"pmp_fill_classified_hole"})
    operation = RepairOperation(
        operation_id="unapproved-pmp-remesh",
        worker="pmp_patch",
        implementation="test",
        parameters={"operation": "pmp_remesh_generated_patch"},
        issue_ids=["mesh:test"],
        drift_band="conservative",
        source_checkpoint="source.usda",
    )

    result = PmpPatchWorker().execute_typed(
        source=tmp_path / "source.usda",
        output=tmp_path / "output.usda",
        operation=operation,
    )

    assert result.status == "refused"
    assert "unsupported PMP operation" in result.failures[0]
