# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stable response envelope shared by core, server, and CLI.

Mirrors the Content Agents `contracts.py` shapes (TaskArtifact / TaskIssue) and carries
a `schema_version` so agents can depend on the shape (content-agents-mapping.md §3).
Plain dataclasses — no pydantic — to keep `usd_core` dependency-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "1"


@dataclass
class Artifact:
    """A produced file: a render, an overlay, a JSON summary."""

    path: str
    kind: str  # "render" | "overlay" | "summary" | "export" | ...
    label: str = ""


@dataclass
class Issue:
    """A non-fatal observation surfaced to the agent."""

    severity: str  # "info" | "warn" | "error"
    message: str
    detail: str = ""


@dataclass
class Response:
    """Every command returns one of these. `--json` serializes it verbatim."""

    command: str
    ok: bool = True
    schema_version: str = SCHEMA_VERSION
    summary: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Response":
        return cls(
            command=d.get("command", ""),
            ok=d.get("ok", True),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            summary=d.get("summary", {}),
            data=d.get("data", {}),
            artifacts=[Artifact(**a) for a in d.get("artifacts", [])],
            issues=[Issue(**i) for i in d.get("issues", [])],
        )
