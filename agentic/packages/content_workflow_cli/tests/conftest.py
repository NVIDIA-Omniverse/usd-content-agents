# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for content-workflow-cli tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from world_understanding.functions.physics.physics_topology import sha256_file

from content_workflow_cli import runner

PhysicsPacketWriter = Callable[..., dict[str, Any]]


@pytest.fixture
def physics_packet_writer() -> PhysicsPacketWriter:
    """Write a valid pinned physics packet and return its payload."""

    def write(
        run_dir: Path,
        source_usd: Path,
        *,
        inspection_usd: Path | None = None,
        session_id: str = "session-1",
        source_path_expansions: dict[str, list[str]] | None = None,
        initial_evidence_renders: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        inspection = (inspection_usd or source_usd).resolve()
        source = source_usd.resolve()
        packet = {
            "schema_version": "content-agents.physics-run-packet.v2",
            "session_id": session_id,
            "source_usd_path": str(source),
            "source_asset_sha256": runner._file_digest(source),
            "inspection_usd_path": str(inspection),
            "inspection_asset_sha256": runner._file_digest(inspection),
            "inspection_bundle_sha256": (
                runner._physics_inspection_bundle_digest(
                    inspection,
                    allow_hardlinks=inspection == source,
                )
            ),
            "inspection_source_digest": sha256_file(inspection),
            "path_space": "inspection" if inspection != source else "source",
            "source_path_expansions": source_path_expansions or {},
            "initial_evidence_renders": initial_evidence_renders or [],
        }
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "physics_run_packet.json").write_text(
            json.dumps(packet),
            encoding="utf-8",
        )
        return packet

    return write
