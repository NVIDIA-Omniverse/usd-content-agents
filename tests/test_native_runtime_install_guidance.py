# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard public OVRTX/OvPhysX guidance against floating installs."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OVRTX_LOCK = "world_understanding/functions/graphics/pylock.ovrtx-runtime.toml"
OVPHYSX_LOCK = "apps/physics_agent/runtime/pylock.ovphysx-runtime.toml"
OVPHYSX_ARM64_LOCK = "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml"

_GUIDANCE_ROOTS = (
    REPO_ROOT / "world_understanding",
    REPO_ROOT / "apps/physics_agent",
    REPO_ROOT / "apps/ovrtx_rendering_api",
    REPO_ROOT / ".agents/skills/physics-agent-cli",
    REPO_ROOT / ".agents/skills/deploy-ovrtx-docker",
)
_GUIDANCE_SUFFIXES = frozenset({".md", ".py", ".sh"})
_OVRTX_REPRO = REPO_ROOT / "apps/ovrtx_rendering_api/tests/renders/ovrtx_bug_repro.py"
_NATIVE_INSTALL = re.compile(
    r"\b(?:uv\s+pip|python3?\s+-m\s+pip|pip)\s+install\b[^\n]*"
    r"\b(?:ovrtx|ovphysx)\b",
    flags=re.IGNORECASE,
)


def _public_guidance_files() -> Iterator[Path]:
    for root in _GUIDANCE_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in _GUIDANCE_SUFFIXES:
                continue
            relative = path.relative_to(REPO_ROOT)
            if "internal" in relative.parts:
                continue
            if "tests" in relative.parts and path != _OVRTX_REPRO:
                continue
            yield path


def test_public_native_runtime_guidance_has_no_floating_installs() -> None:
    violations: list[str] = []
    for path in _public_guidance_files():
        text = path.read_text(encoding="utf-8").replace("\\\n", " ")
        for line in text.splitlines():
            match = _NATIVE_INSTALL.search(line)
            if match is None:
                continue
            uses_lock = "pylock.ovrtx" in line or "pylock.ovphysx" in line
            enforces_hashes = "--require-hashes" in line and "--no-deps" in line
            if not (uses_lock and enforces_hashes):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}: {match.group(0).strip()}"
                )

    assert not violations, (
        "OVRTX/OvPhysX install guidance must use the checked-in PEP 751 locks:\n"
        + "\n".join(violations)
    )


def test_native_runtime_bootstrap_hints_name_the_reviewed_locks() -> None:
    ovphysx_guidance = (
        REPO_ROOT / "world_understanding/functions/physics/ovphysx_daemon.py",
        REPO_ROOT / "world_understanding/functions/physics/_ovphysx_daemon_script.py",
        REPO_ROOT / "apps/physics_agent/docs/tuning.md",
    )
    for path in ovphysx_guidance:
        text = path.read_text(encoding="utf-8")
        assert OVPHYSX_LOCK in text, path
        assert OVPHYSX_ARM64_LOCK in text, path

    locked_commands = (*ovphysx_guidance[2:], _OVRTX_REPRO)
    for path in locked_commands:
        text = path.read_text(encoding="utf-8")
        assert "--require-hashes" in text, path
        assert "--no-deps" in text, path
        if path != _OVRTX_REPRO:
            assert ".wu-ovphysx-runtime-ready" in text, path

    assert OVRTX_LOCK in _OVRTX_REPRO.read_text(encoding="utf-8")
