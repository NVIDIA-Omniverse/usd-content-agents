# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for requests-only service clients that cannot import core."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PHYSICS_CONCURRENCY_SCRIPT = (
    REPO_ROOT / "apps/physics_agent_service/internal/scripts/test_concurrency.py"
)
STANDALONE_CLIENT_ENTRYPOINTS = (
    (REPO_ROOT / "apps/joint_agent_service/client/client.py", "JointAgentClient"),
    (REPO_ROOT / "apps/joint_agent_service/client/client_v2.py", "JointAgentClient"),
    (REPO_ROOT / "apps/physics_agent_service/client/client.py", "PhysicsAgentClient"),
    (
        REPO_ROOT / "apps/physics_agent_service/client/client_v2.py",
        "PhysicsAgentClient",
    ),
)
EXECUTABLE_CONCURRENCY_SCRIPTS = (
    REPO_ROOT / "apps/joint_agent_service/scripts/test_concurrency.py",
)
CONCURRENCY_SELECTOR_SCRIPTS = (
    *EXECUTABLE_CONCURRENCY_SCRIPTS,
    PHYSICS_CONCURRENCY_SCRIPT,
)
PUBLIC_CLIENT_SKILLS = (
    REPO_ROOT / ".agents/skills/joint-agent-client/SKILL.md",
    REPO_ROOT / ".agents/skills/physics-agent-client/SKILL.md",
)


def _requests_only_environment(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "requests.py").write_text(
        "class Session:\n    pass\n\nclass Response:\n    pass\n",
        encoding="utf-8",
    )
    return {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(tmp_path),
    }


@pytest.mark.parametrize(
    ("script", "client_global"),
    STANDALONE_CLIENT_ENTRYPOINTS,
    ids=lambda value: (
        str(value.relative_to(REPO_ROOT)) if isinstance(value, Path) else value
    ),
)
def test_requests_only_client_forwards_open_backend_name(
    script: Path,
    client_global: str,
    tmp_path: Path,
) -> None:
    """Client CLIs must forward names without importing or copying core."""
    program = textwrap.dedent(
        f"""
        import runpy

        data = runpy.run_path({str(script)!r})

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def run_and_monitor(self, **kwargs):
                print("FORWARDED=" + str(kwargs["render_backend"]))
                return "session", None

        main = data["main"]
        main.__globals__[{client_global!r}] = FakeClient
        main(["asset.usdz", "--render-backend", "future-backend", "--quiet"])
        """
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", program],
        cwd=tmp_path,
        env=_requests_only_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "FORWARDED=future-backend" in result.stdout


@pytest.mark.parametrize(
    "script",
    EXECUTABLE_CONCURRENCY_SCRIPTS,
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_requests_only_concurrency_script_forwards_open_backend_name(
    script: Path,
    tmp_path: Path,
) -> None:
    """Concurrency CLIs must pass open names to their request workers."""
    program = textwrap.dedent(
        f"""
        import runpy

        data = runpy.run_path({str(script)!r})

        def fake_run_single_pipeline(**kwargs):
            print("FORWARDED=" + str(kwargs["render_backend"]))
            return object()

        def fake_print_summary(results):
            return True

        main = data["main"]
        main.__globals__["run_single_pipeline"] = fake_run_single_pipeline
        main.__globals__["print_summary"] = fake_print_summary
        code = main([
            "--local",
            "--s3-only",
            "--num-jobs",
            "1",
            "--render-backend",
            "future-backend",
        ])
        print("EXIT=" + str(code))
        """
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", program],
        cwd=tmp_path,
        env=_requests_only_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "FORWARDED=future-backend" in result.stdout
    assert "EXIT=0" in result.stdout


@pytest.mark.parametrize(
    "script",
    CONCURRENCY_SELECTOR_SCRIPTS,
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_concurrency_script_keeps_render_backend_as_open_passthrough(
    script: Path,
) -> None:
    """Requests-only helpers delegate canonical validation to the service."""
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    render_backend_arguments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--render-backend"
    ]

    assert len(render_backend_arguments) == 1
    assert all(
        keyword.arg != "choices" for keyword in render_backend_arguments[0].keywords
    )


@pytest.mark.parametrize(
    "skill_path",
    PUBLIC_CLIENT_SKILLS,
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_requests_only_client_skill_delegates_backend_validation(
    skill_path: Path,
) -> None:
    """Public client guidance must not duplicate the canonical backend list."""
    backend_rows = [
        line
        for line in skill_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `render_backend` |")
    ]

    assert len(backend_rows) == 1
    assert (
        "validated by the server against the current canonical registry"
        in backend_rows[0]
    )
