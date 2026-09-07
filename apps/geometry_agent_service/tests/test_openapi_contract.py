# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.metadata import version as distribution_version
from pathlib import Path

import yaml

from geometry_agent_service.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def test_checked_in_openapi_matches_application_contract() -> None:
    checked_in = yaml.safe_load(
        (PROJECT_ROOT / "openapi.yaml").read_text(encoding="utf-8")
    )
    assert checked_in == app.openapi()

    paths = checked_in["paths"]
    assert set(paths) == {
        "/api/health",
        "/api/info",
        "/api/geometry/sources",
        "/api/geometry/generations",
        "/api/geometry/revisions",
        "/api/geometry/families",
        "/api/geometry/provider-exports",
        "/api/geometry/exports",
        "/api/geometry/runs",
        "/api/geometry/jobs/{job_id}",
        "/api/geometry/providers",
        "/api/geometry/artifacts/{artifact_id}",
    }
    for path in set(paths) - {"/api/health", "/api/info"}:
        operation = next(iter(paths[path].values()))
        assert operation["security"] == [
            {"APIKeyHeader": []},
            {"HTTPBearer": []},
        ]


def test_service_versions_match_repository_release() -> None:
    repository_version = (
        (REPOSITORY_ROOT / "VERSION.md").read_text(encoding="utf-8").strip()
    )

    assert distribution_version("geometry-agent-service") == repository_version
    assert app.version == repository_version
