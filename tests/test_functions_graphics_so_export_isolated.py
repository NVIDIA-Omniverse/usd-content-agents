# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""so_export must work inside the ABI-isolated Scene Optimizer worker.

The worker copies ``so_export.py`` into a temp dir and runs it under
``python -S`` with a replaced ``PYTHONPATH``, so ``world_understanding`` is
unimportable there by design. A regression (#1172 follow-up) added a
``world_understanding`` import on the MDL-token path, which made every
textured-asset optimization fail with "Failed to enumerate USD asset
dependencies". This test executes the real module file in that exact
environment.
"""

import importlib.abc
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SO_EXPORT = (
    Path(__file__).resolve().parents[1]
    / "world_understanding"
    / "functions"
    / "graphics"
    / "so_export.py"
)


def test_is_bare_mdl_token_without_world_understanding(tmp_path: Path) -> None:
    shutil.copy2(_SO_EXPORT, tmp_path / "so_export.py")
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import sys

            sys.path.insert(0, sys.argv[1])
            try:
                import world_understanding
            except ModuleNotFoundError:
                pass
            else:
                raise SystemExit("test invalid: world_understanding importable")

            import so_export

            assert so_export.is_bare_mdl_token("@OmniPBR.mdl@")
            assert so_export.is_bare_mdl_token("NvidiaMetal.mdl")
            assert not so_export.is_bare_mdl_token("textures/albedo.png")
            assert not so_export.is_bare_mdl_token("./materials/Steel.mdl")
            assert not so_export.is_bare_mdl_token("omniverse://server/x.mdl")
            print("ok")
            """
        )
    )
    # -S plus an emptied PYTHONPATH mirrors _subprocess_env: no site-packages,
    # so the world_understanding package must not be reachable.
    result = subprocess.run(
        [sys.executable, "-S", str(probe), str(tmp_path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path)},
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


class _BlockImport(importlib.abc.MetaPathFinder):
    """Raise ModuleNotFoundError for one module, mimicking the isolated worker."""

    def __init__(self, blocked: str) -> None:
        self._blocked = blocked

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        if fullname == self._blocked:
            raise ModuleNotFoundError(fullname)
        return None


def test_is_bare_mdl_token_fallback_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the inline fallback when ``asset_paths`` cannot be imported."""
    from world_understanding.functions.graphics import so_export

    # Normal path first: the package import succeeds and delegates.
    assert so_export.is_bare_mdl_token("@OmniPBR.mdl@")
    assert not so_export.is_bare_mdl_token("textures/albedo.png")

    blocked = "world_understanding.utils.usd.asset_paths"
    monkeypatch.delitem(sys.modules, blocked, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockImport(blocked), *sys.meta_path])

    assert so_export.is_bare_mdl_token("@OmniPBR.mdl@")
    assert so_export.is_bare_mdl_token("NvidiaMetal.mdl")
    assert not so_export.is_bare_mdl_token("textures/albedo.png")
    assert not so_export.is_bare_mdl_token("./materials/Steel.mdl")
    assert not so_export.is_bare_mdl_token("omniverse://server/x.mdl")
    assert not so_export.is_bare_mdl_token("")
