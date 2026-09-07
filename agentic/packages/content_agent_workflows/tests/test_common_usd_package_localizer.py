# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

import content_agent_workflows.common.usd_package_localizer as localizer


def _fail_temporary_directory_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    real_temporary_directory = localizer.tempfile.TemporaryDirectory

    class TemporaryDirectoryWithFailingExit:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._delegate = real_temporary_directory(*args, **kwargs)

        def __enter__(self) -> str:
            return self._delegate.__enter__()

        def __exit__(self, *args: Any) -> None:
            self._delegate.__exit__(*args)
            raise OSError("synthetic temporary cleanup failure")

    monkeypatch.setattr(
        localizer.tempfile,
        "TemporaryDirectory",
        TemporaryDirectoryWithFailingExit,
    )


def test_package_localizer_absorbs_only_rewritten_member_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "localized.usdz"
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        environment = kwargs.pop("env")
        assert environment["TMPDIR"] == environment["TMP"]
        assert environment["TMPDIR"] == environment["TEMP"]
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        output.write_bytes(b"localized-package")
        return subprocess.CompletedProcess(
            command,
            0,
            "",
            "Warning: in _EnqueueDependency at line 99 of /build/usdUtils/"
            "assetLocalization.cpp -- Failed to resolve reference @0/source.usdc@ "
            "with computed asset path @0/source.usdc@ found in layer "
            f"@{source}@.\n",
        )

    monkeypatch.setattr(localizer.subprocess, "run", run)

    localizer.create_localized_usdz_package(
        source,
        output,
        "localized.usdc",
    )

    assert output.read_bytes() == b"localized-package"
    assert commands[0][1:3] == [
        str(Path(localizer.__file__).resolve()),
        "--localize-child",
    ]


def test_package_localizer_uses_distinct_private_temporary_roots_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    temporary_roots: list[Path] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        temporary_root = Path(environment["TMPDIR"])
        assert environment["TMP"] == str(temporary_root)
        assert environment["TEMP"] == str(temporary_root)
        assert temporary_root.is_dir()
        assert not temporary_root.is_symlink()
        assert temporary_root.parent == tmp_path
        if os.name != "nt":
            assert stat.S_IMODE(temporary_root.lstat().st_mode) == 0o700
            assert temporary_root.lstat().st_uid == os.geteuid()
        temporary_roots.append(temporary_root)
        Path(command[4]).write_bytes(b"localized-package")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(localizer.subprocess, "run", run)

    for index in range(2):
        localizer.create_localized_usdz_package(
            source,
            tmp_path / f"localized-{index}.usdz",
            "localized.usdc",
        )

    assert len(set(temporary_roots)) == 2
    assert all(not root.exists() for root in temporary_roots)


def test_package_localizer_rejects_other_diagnostics_and_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "localized.usdz"

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output.write_bytes(b"partial-package")
        return subprocess.CompletedProcess(
            command,
            0,
            "",
            "Warning: unresolved original dependency @missing.png@\n",
        )

    monkeypatch.setattr(localizer.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="unresolved original dependency"):
        localizer.create_localized_usdz_package(
            source,
            output,
            "localized.usdc",
        )

    assert not output.exists()


def test_package_localizer_preserves_child_failure_when_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "localized.usdz"

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output.write_bytes(b"partial-package")
        return subprocess.CompletedProcess(command, 17, "", "")

    _fail_temporary_directory_exit(monkeypatch)
    monkeypatch.setattr(localizer.subprocess, "run", run)

    with pytest.raises(
        RuntimeError,
        match="localizer exited with code 17",
    ) as exc_info:
        localizer.create_localized_usdz_package(
            source,
            output,
            "localized.usdc",
        )

    assert isinstance(exc_info.value.__cause__, OSError)
    assert "synthetic temporary cleanup failure" in str(exc_info.value.__cause__)
    assert not output.exists()


def test_package_localizer_removes_successful_output_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "localized.usdz"

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output.write_bytes(b"complete-package")
        return subprocess.CompletedProcess(command, 0, "", "")

    _fail_temporary_directory_exit(monkeypatch)
    monkeypatch.setattr(localizer.subprocess, "run", run)

    with pytest.raises(OSError, match="synthetic temporary cleanup failure"):
        localizer.create_localized_usdz_package(
            source,
            output,
            "localized.usdc",
        )

    assert not output.exists()


def test_package_localizer_absorbs_rewritten_member_notice_with_windows_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "localized.usdz"

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        output.write_bytes(b"localized-package")
        return subprocess.CompletedProcess(
            command,
            0,
            "",
            r"Warning: in _EnqueueDependency at line 99 of "
            r"C:\build\usdUtils\assetLocalization.cpp -- Failed to resolve "
            r"reference @0/source.usdc@ with computed asset path "
            r"@0/source.usdc@ found in layer @C:\run\source.usda@."
            "\n",
        )

    monkeypatch.setattr(localizer.subprocess, "run", run)

    localizer.create_localized_usdz_package(
        source,
        output,
        "localized.usdc",
    )

    assert output.read_bytes() == b"localized-package"
