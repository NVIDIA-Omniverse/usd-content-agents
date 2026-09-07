# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure command contracts for static 0.6 stages.

The models describe reviewed commands; they never resolve a path, inspect a
launcher, construct a subprocess environment, or execute a command.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_validator,
)

from joint_agent.static_qualification_contracts import (
    StaticQualificationValidationStageName,
)

ARTICULATION_V2_STATIC_COMMAND_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-command-v1"
] = "joint-agent-articulation-v2-static-command-v1"
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?$")


class _CommandModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )


class ArticulationV2StaticCommandV1(_CommandModel):
    """Canonical argv and reviewed tool identity for one planned stage."""

    schema_version: Literal["joint-agent-articulation-v2-static-command-v1"]
    stage: StaticQualificationValidationStageName
    tool_id: str
    tool_version: str
    profile_id: str
    result_id: str
    argv: tuple[str, ...]

    @field_validator("tool_id", "profile_id", "result_id")
    @classmethod
    def _nonblank_identity(cls, value: str, info: Any) -> str:
        if not value or value.strip() != value or "\x00" in value:
            raise ValueError(f"{info.field_name} must be exact and nonblank")
        return value

    @field_validator("tool_version")
    @classmethod
    def _semantic_tool_version(cls, value: str) -> str:
        if _VERSION_RE.fullmatch(value) is None:
            raise ValueError("tool_version must be a canonical semantic version")
        return value

    @field_validator("argv")
    @classmethod
    def _canonical_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("command argv must not be empty")
        if any(
            type(token) is not str
            or not token
            or token.strip() != token
            or "\x00" in token
            for token in value
        ):
            raise ValueError("command argv tokens must be exact and nonblank")
        return value


_COMMAND_ADAPTER = TypeAdapter(ArticulationV2StaticCommandV1)


def canonical_articulation_v2_static_command_bytes(
    command: ArticulationV2StaticCommandV1,
) -> bytes:
    """Return the deterministic JSON preimage for one inert command."""

    if type(command) is not ArticulationV2StaticCommandV1:
        raise TypeError("command must be an exact ArticulationV2StaticCommandV1")
    encoded = json.dumps(
        _COMMAND_ADAPTER.dump_python(command, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    validated = _COMMAND_ADAPTER.validate_json(encoded, strict=True)
    return json.dumps(
        _COMMAND_ADAPTER.dump_python(validated, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def articulation_v2_static_command_sha256(
    command: ArticulationV2StaticCommandV1,
) -> str:
    return hashlib.sha256(
        canonical_articulation_v2_static_command_bytes(command)
    ).hexdigest()


__all__ = [
    "ARTICULATION_V2_STATIC_COMMAND_SCHEMA_VERSION",
    "ArticulationV2StaticCommandV1",
    "articulation_v2_static_command_sha256",
    "canonical_articulation_v2_static_command_bytes",
]
