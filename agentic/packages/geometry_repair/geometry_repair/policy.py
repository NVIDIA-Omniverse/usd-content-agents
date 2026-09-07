# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable worker allow-list and dependency policy checks."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from .worker_ids import canonical_worker_name

_ENABLED_STATUSES = {"enabled", "enabled_with_notice", "enabled_external"}
_ALLOWED_COMPONENT_CLASSES = {"nvidia_first_party", "open_source"}


def worker_registry() -> dict[str, Any]:
    """Load the checked-in worker provenance registry."""

    path = resources.files("geometry_repair").joinpath("worker_registry.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("workers"), list):
        raise RuntimeError("Invalid geometry repair worker registry")
    return payload


def worker_record(name: str) -> dict[str, Any]:
    """Return one worker record or reject an undeclared implementation."""

    name = canonical_worker_name(name)
    for record in worker_registry()["workers"]:
        if isinstance(record, dict) and record.get("name") == name:
            return record
    raise ValueError(f"Repair worker {name!r} is absent from the provenance registry")


def assert_worker_enabled(name: str, *, allow_external: bool = False) -> None:
    """Reject workers that are not approved OSS or NVIDIA first-party components."""

    record = worker_record(name)
    status = str(record.get("approval_status") or "")
    component_class = str(record.get("component_class") or "")
    if status not in _ENABLED_STATUSES:
        raise ValueError(f"Repair worker {name!r} is not enabled: {status}")
    if component_class not in _ALLOWED_COMPONENT_CLASSES:
        raise ValueError(
            f"Repair worker {name!r} is not classified as approved open source or NVIDIA first-party"
        )
    if status == "enabled_external" and not allow_external:
        raise ValueError(f"Repair worker {name!r} is owned by an external shared workflow boundary")


def worker_operations(name: str) -> frozenset[str]:
    """Return the exact operation allow-list for one registered worker."""

    record = worker_record(name)
    raw_operations = record.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError(f"Repair worker {name!r} has no registered operations")
    if not all(isinstance(operation, str) and operation for operation in raw_operations):
        raise ValueError(f"Repair worker {name!r} has an invalid operation allow-list")
    operations = frozenset(raw_operations)
    if len(operations) != len(raw_operations):
        raise ValueError(f"Repair worker {name!r} has duplicate registered operations")
    return operations


def assert_worker_operations_enabled(
    name: str,
    operations: str | set[str] | frozenset[str],
    *,
    allow_external: bool = False,
) -> None:
    """Reject executable capabilities absent from the worker operation allow-list."""

    assert_worker_enabled(name, allow_external=allow_external)
    requested = {operations} if isinstance(operations, str) else set(operations)
    if not requested or not all(
        isinstance(operation, str) and operation for operation in requested
    ):
        raise ValueError(f"Repair worker {name!r} requested an invalid empty operation set")
    unapproved = requested - worker_operations(name)
    if unapproved:
        raise ValueError(
            f"Repair worker {name!r} requested unapproved operations: "
            f"{', '.join(sorted(unapproved))}"
        )
