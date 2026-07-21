# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YAML loader for robot prompt entries.

Reads ``apps/joint_agent/joint_agent/prompts/data/<robot_id>.yaml`` files and
returns validated ``RobotPromptEntry`` objects. Results are cached.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources
from pathlib import Path

import yaml

from joint_agent.prompts.schema import RobotPromptEntry

_ROBOT_ID_RE = re.compile(r"[A-Za-z0-9_-]+")


def _data_dir() -> Path:
    """Return the path to the bundled ``data/`` directory."""
    pkg = resources.files("joint_agent.prompts")
    return Path(str(pkg / "data"))


@cache
def load_prompt(robot_id: str) -> RobotPromptEntry:
    """Load and validate a robot prompt entry by ID.

    Args:
        robot_id: The robot identifier matching a YAML filename.

    Raises:
        ValueError: If ``robot_id`` is not a safe prompt-library slug.
        FileNotFoundError: If no matching YAML exists in ``data/``.
        pydantic.ValidationError: If the YAML fails schema validation.
    """
    if not _ROBOT_ID_RE.fullmatch(robot_id):
        raise ValueError(f"Invalid robot_id: {robot_id!r}")

    path = _data_dir() / f"{robot_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No prompt entry found for robot_id={robot_id!r} (looked for {path})"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a mapping")
    if "robot_id" not in data:
        data["robot_id"] = robot_id
    if data["robot_id"] != robot_id:
        raise ValueError(
            f"robot_id mismatch in {path}: file declares "
            f"{data['robot_id']!r} but filename suggests {robot_id!r}"
        )
    return RobotPromptEntry.model_validate(data)


def list_robots() -> list[str]:
    """Return the sorted list of robot IDs available in the library."""
    return sorted(p.stem for p in _data_dir().glob("*.yaml"))


@cache
def load_all() -> tuple[RobotPromptEntry, ...]:
    """Load every entry; convenience for matching and listing tools."""
    return tuple(load_prompt(rid) for rid in list_robots())
