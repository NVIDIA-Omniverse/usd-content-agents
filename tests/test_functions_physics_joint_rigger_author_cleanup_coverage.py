# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused cleanup-error semantics for the owned topology author."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from world_understanding.functions.physics.joint_rigger import author as author_module


def _install_projection_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    close_binding: Any,
    remove_directory: Any,
) -> tuple[
    Path,
    SimpleNamespace,
    SimpleNamespace,
    Path,
    Path,
    dict[Path, Path],
]:
    source_path = tmp_path / "source.usda"
    source_identity = object()
    binding = SimpleNamespace(descriptor=object(), sha256="0" * 64)
    request = SimpleNamespace(source_asset=source_identity)
    dependency_snapshots = (("dependency.usda", 23, "1" * 64, "dependency.usda", True),)
    bound_directory = tmp_path / "bound-projection"
    bound_source = bound_directory / "source.usda"
    restore_paths = {bound_source: source_path}

    def create_binding(path: Path, *, expected: Any) -> SimpleNamespace:
        assert path is source_path
        assert expected is source_identity
        return binding

    def snapshot_dependencies(observed_binding: Any) -> tuple[Any, ...]:
        assert observed_binding is binding
        return dependency_snapshots

    def materialize_projection(
        *,
        descriptor: Any,
        expected_sha256: str,
        logical_input_path: Path,
        dependencies: tuple[Any, ...],
        editable_root: bool,
    ) -> tuple[Path, Path, dict[Path, Path]]:
        assert descriptor is binding.descriptor
        assert expected_sha256 == binding.sha256
        assert logical_input_path is source_path
        assert dependencies is dependency_snapshots
        assert editable_root is False
        return bound_source, bound_directory, restore_paths

    def checked_close_binding(observed_binding: Any) -> list[Exception]:
        assert observed_binding is binding
        return close_binding(observed_binding)

    def checked_remove_directory(directory: Path) -> None:
        assert directory is bound_directory
        remove_directory(directory)

    monkeypatch.setattr(
        author_module,
        "create_sealed_source_binding",
        create_binding,
    )
    monkeypatch.setattr(
        author_module,
        "bound_input_dependency_snapshots",
        snapshot_dependencies,
    )
    monkeypatch.setattr(
        author_module,
        "materialize_bound_input",
        materialize_projection,
    )
    monkeypatch.setattr(
        author_module,
        "close_source_binding",
        checked_close_binding,
    )
    monkeypatch.setattr(
        author_module,
        "remove_bound_input_directory",
        checked_remove_directory,
    )
    return (
        source_path,
        binding,
        request,
        bound_source,
        bound_directory,
        restore_paths,
    )


def test_bound_projection_reraises_single_ordinary_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_error = RuntimeError("projection cleanup failed")
    close_calls = 0

    def remove_directory(path: Path) -> None:
        raise cleanup_error

    def close_binding(binding: Any) -> list[Exception]:
        nonlocal close_calls
        close_calls += 1
        return []

    (
        source_path,
        binding,
        request,
        bound_source,
        bound_directory,
        restore_paths,
    ) = _install_projection_stubs(
        monkeypatch,
        tmp_path=tmp_path,
        close_binding=close_binding,
        remove_directory=remove_directory,
    )

    with pytest.raises(RuntimeError) as raised:
        with author_module._bound_source_projection(
            source_path,
            request,
            editable_root=False,
        ) as projection:
            assert projection[0] is binding
            assert projection[1] is bound_source
            assert projection[2] is bound_directory
            assert projection[3] is restore_paths

    assert raised.value is cleanup_error
    assert close_calls == 1
    assert getattr(cleanup_error, "__notes__", ()) == ()


def test_bound_projection_groups_two_ordinary_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_error = RuntimeError("projection directory cleanup failed")
    descriptor_error = OSError("source descriptor cleanup failed")
    cleanup_calls: list[tuple[str, Any]] = []

    def remove_directory(path: Path) -> None:
        cleanup_calls.append(("remove", path))
        raise projection_error

    def close_binding(binding: Any) -> list[Exception]:
        cleanup_calls.append(("close", binding))
        raise descriptor_error

    (
        source_path,
        binding,
        request,
        bound_source,
        bound_directory,
        restore_paths,
    ) = _install_projection_stubs(
        monkeypatch,
        tmp_path=tmp_path,
        close_binding=close_binding,
        remove_directory=remove_directory,
    )

    with pytest.raises(ExceptionGroup) as raised:
        with author_module._bound_source_projection(
            source_path,
            request,
            editable_root=False,
        ) as projection:
            assert projection == (
                binding,
                bound_source,
                bound_directory,
                restore_paths,
            )

    assert raised.value.message == "Bound source cleanup failed"
    assert len(raised.value.exceptions) == 2
    assert raised.value.exceptions[0] is projection_error
    assert raised.value.exceptions[1] is descriptor_error
    assert [label for label, _ in cleanup_calls] == ["remove", "close"]
    assert cleanup_calls[0][1] is bound_directory
    assert cleanup_calls[1][1] is binding


def test_bound_projection_preserves_mixed_cleanup_only_baseexception_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fatal_cleanup = KeyboardInterrupt("projection cleanup interrupted")
    descriptor_error = OSError("descriptor close failed")
    descriptor_error.add_note("descriptor fd 42 remained open")
    close_calls = 0

    def remove_directory(path: Path) -> None:
        raise fatal_cleanup

    def close_binding(binding: Any) -> list[Exception]:
        nonlocal close_calls
        close_calls += 1
        return [descriptor_error]

    (
        source_path,
        binding,
        request,
        bound_source,
        bound_directory,
        restore_paths,
    ) = _install_projection_stubs(
        monkeypatch,
        tmp_path=tmp_path,
        close_binding=close_binding,
        remove_directory=remove_directory,
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        with author_module._bound_source_projection(
            source_path,
            request,
            editable_root=False,
        ) as projection:
            assert projection == (
                binding,
                bound_source,
                bound_directory,
                restore_paths,
            )

    assert raised.value is fatal_cleanup
    assert close_calls == 1
    assert getattr(fatal_cleanup, "__notes__", ()) == [
        "Bound source descriptor cleanup failed: OSError: descriptor close failed",
        "Bound source descriptor cleanup failed detail: descriptor fd 42 remained open",
    ]
