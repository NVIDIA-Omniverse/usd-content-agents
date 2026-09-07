# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic repair ordering derived from checked-in benchmark evidence."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def route_policy() -> dict[str, Any]:
    path = resources.files("geometry_repair").joinpath("route_policy.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "geometry-repair.route-policy.v1":
        raise RuntimeError("Invalid geometry repair route policy")
    if not isinstance(payload.get("workers"), dict) or not isinstance(
        payload.get("issue_routes"), dict
    ):
        raise RuntimeError("Geometry repair route policy is incomplete")
    return payload


def ranked_workers(issue_ids: set[str], enabled: set[str]) -> tuple[list[str], dict[str, Any]]:
    """Return applicable workers in evidence-backed least-destructive order."""

    policy = route_policy()
    routes = policy["issue_routes"]
    applicable = {
        worker for issue_id in issue_ids for worker in routes.get(issue_id, []) if worker in enabled
    }
    worker_records = policy["workers"]
    ordered = sorted(
        applicable,
        key=lambda worker: (
            int(worker_records.get(worker, {}).get("priority", 10_000)),
            int(worker_records.get(worker, {}).get("destructive_cost", 10_000)),
            worker,
        ),
    )
    evidence = {
        "policy_id": policy["policy_id"],
        "source_evidence": policy["source_evidence"],
        "blocking_issue_ids": sorted(issue_ids),
        "ranked_workers": [
            {
                "worker": worker,
                **worker_records.get(worker, {}),
                "target_issue_ids": sorted(
                    issue_id for issue_id in issue_ids if worker in routes.get(issue_id, [])
                ),
            }
            for worker in ordered
        ],
    }
    return ordered, evidence
