# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated entry point for bounded source-collision audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collision_audit import CollisionAuditLimits, audit_source_collision_proxy
from .models import ProtectedFeature
from .process_limits import limit_address_space


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    limit_address_space(int(request["memory_mb"]))
    audit_source_collision_proxy(
        request["render_path"],
        request["collision_path"],
        protected_features=[
            ProtectedFeature.model_validate(item) for item in request.get("protected_features", [])
        ],
        limits=CollisionAuditLimits.model_validate(request["limits"]),
        surface_sample_limit=int(request["surface_sample_limit"]),
        occupancy_grid_resolution=int(request["occupancy_grid_resolution"]),
        occupancy_face_point_limit=int(request["occupancy_face_point_limit"]),
        report_path=request["report_path"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
