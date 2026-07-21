# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-boundary tests for the shared Joint Rigger package."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "world_understanding" / "functions" / "physics" / "joint_rigger"


def test_joint_rigger_core_has_no_app_or_external_rigger_imports() -> None:
    forbidden_roots = {"joint_agent", "usd_joint_rigger"}
    for path in sorted(CORE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = {node.module.split(".", 1)[0]}
            else:
                continue
            assert imported.isdisjoint(forbidden_roots), (
                f"{path.relative_to(REPO_ROOT)} imports an app/external package: "
                f"{sorted(imported & forbidden_roots)}"
            )


def test_joint_rigger_core_imports_when_app_and_external_package_are_blocked() -> None:
    script = r"""
import importlib.abc
import sys

class BlockForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "joint_agent" or fullname.startswith("joint_agent."):
            raise AssertionError(f"unexpected app import: {fullname}")
        if fullname == "usd_joint_rigger" or fullname.startswith("usd_joint_rigger."):
            raise AssertionError(f"unexpected external import: {fullname}")
        return None

sys.meta_path.insert(0, BlockForbidden())
import world_understanding.functions.physics.joint_rigger as joint_rigger

assert callable(joint_rigger.author_joint_rig)
assert callable(joint_rigger.author_joint_rig_with_physics)
assert callable(joint_rigger.author_joint_topology)
assert callable(joint_rigger.author_physics_schemas)
assert callable(joint_rigger.capture_joint_rigger_physics_schema_snapshot)
assert callable(joint_rigger.capture_joint_rigger_stage_snapshot)
assert callable(joint_rigger.identify_usd_artifact)
assert callable(joint_rigger.local_usd_dependency_paths)
assert callable(joint_rigger.physics_schema_counts)
assert callable(joint_rigger.validate_authored_joint_topology)
assert callable(joint_rigger.validate_authored_joint_rig_with_physics)
assert callable(joint_rigger.validate_authored_physics_schemas)
assert callable(joint_rigger.validate_joint_rigger_stage_preservation)
assert callable(joint_rigger.validate_joint_topology_plan)
assert callable(joint_rigger.validate_physics_plan_evidence)
assert joint_rigger.JointRiggerInputV1.__name__ == "JointRiggerInputV1"
assert joint_rigger.JointFrictionV1.__name__ == "JointFrictionV1"
assert joint_rigger.JointType is not None
assert joint_rigger.ProvenanceSource is not None
assert joint_rigger.ResultStatus is not None
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_facade_exports_every_public_error_type() -> None:
    from world_understanding.functions.physics.joint_rigger import facade

    assert {
        "JointRiggerArtifactError",
        "JointRiggerBackendIncompatibleError",
        "JointRiggerBackendUnavailableError",
        "JointRiggerFacadeError",
        "JointRiggerPostCommitCleanupError",
    } <= set(facade.__all__)
