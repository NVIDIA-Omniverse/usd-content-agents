# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard: usd-cli operates on refs — commands must not leak prim SdfPaths in their default
(plain-text) output. Full paths are only exposed by `resolve` (the explicit reveal) and in
the structured `--json` `data`. Filesystem paths (save/export/open/convert) are exempt.
"""

from __future__ import annotations

import json

from conftest import SPRAY

# Commands whose plain-text output legitimately carries a path: `resolve` reveals SdfPaths
# on request; save/export deal in filesystem paths.
_PATH_OK = {"resolve", "save", "export"}


def _plain(project, *args) -> str:
    """Run a command WITHOUT --json and return its stdout (the agent-facing text)."""
    return project.cli(*args).out


def test_spatial_and_query_outputs_use_refs_not_paths(project):
    project.open(SPRAY)
    # exercise the commands that return object lists / introspection
    cmds = [
        ("snapshot", "-d", "2", "-m", "-b"),
        ("find", "--type", "Mesh"),
        ("properties", "@n4"),
        ("describe", "@n4"),
        ("material", "@n4", "--color", "1,0,0"),
        ("material-binding", "@n4"),
        ("nearest", "@n4", "--count", "3"),
        ("within", "@n4", "1.0"),
        ("overlapping", "@n4"),
        ("above", "@n4"),
        ("distance", "@n4", "@n7"),
        ("raycast", "0,0,1", "0,0,-1"),
    ]
    for args in cmds:
        out = _plain(project, *args)
        assert "/spray_bottle/" not in out, f"`{args[0]}` leaked a prim path:\n{out}"


def test_resolve_still_reveals_paths(project):
    project.open(SPRAY)
    out = _plain(project, "resolve", "@n4")
    assert "/spray_bottle/" in out  # resolve is the explicit path-reveal command


def test_json_still_carries_paths(project):
    """--json `data` keeps full paths for callers that want them (it's 'requested')."""
    project.open(SPRAY)
    env = project.cli("nearest", "@n4", "--count", "2", json=True, expect_ok=True).json()
    assert all("/spray_bottle/" in r["path"] for r in env["data"]["results"])
    assert all(r["ref"].startswith("@") for r in env["data"]["results"])
