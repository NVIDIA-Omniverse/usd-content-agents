# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for Joint Rigger source-binding trust boundaries."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd

from world_understanding.functions.physics.joint_rigger import (
    artifacts as artifacts_module,
)
from world_understanding.functions.physics.joint_rigger import (
    source_binding,
)
from world_understanding.functions.physics.joint_rigger.facade import (
    JointRiggerArtifactError,
    JointRiggerBackendIncompatibleError,
)
from world_understanding.functions.physics.joint_rigger.reference import (
    identify_usd_artifact,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_asset_layer(path: Path, locators: list[str]) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    prim = stage.DefinePrim("/Root")
    attribute = prim.CreateAttribute(
        "sourceBinding:testAssets",
        Sdf.ValueTypeNames.AssetArray,
        custom=True,
    )
    assert attribute.Set([Sdf.AssetPath(locator) for locator in locators])
    assert stage.GetRootLayer().Save()
    del stage


def _write_usdz(path: Path, root_payload: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        root_entry = zipfile.ZipInfo(
            "root.usda",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        root_entry.compress_type = zipfile.ZIP_STORED
        root_entry.create_system = 3
        root_entry.external_attr = 0o100644 << 16
        archive.writestr(root_entry, root_payload)


def test_materialize_expanduser_failure_removes_created_private_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    payload = b"#usda 1.0\n"
    source.write_bytes(payload)
    descriptor = os.open(source, os.O_RDONLY)
    monkeypatch.setattr(
        source_binding.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    try:
        with pytest.raises(RuntimeError, match="home directory"):
            source_binding.materialize_bound_input(
                descriptor=descriptor,
                expected_sha256=_sha256(payload),
                logical_input_path=Path(
                    "~joint-rigger-user-that-cannot-exist/source.usda"
                ),
            )
    finally:
        os.close(descriptor)

    assert not list(tmp_path.glob("joint-rigger-bound-input-*"))


def test_duplicate_projection_alias_rejects_conflicting_restore_path(
    tmp_path: Path,
) -> None:
    root_payload = b"#usda 1.0\n"
    dependency_payload = b"#usda 1.0\n"
    root = tmp_path / "root.usda"
    dependency = tmp_path / "dependency.usda"
    root.write_bytes(root_payload)
    dependency.write_bytes(dependency_payload)
    root_descriptor = os.open(root, os.O_RDONLY)
    dependency_descriptor = os.open(dependency, os.O_RDONLY)
    logical_dependency = tmp_path / "logical" / "dependency.usda"

    try:
        with pytest.raises(ValueError, match="Conflicting bound dependencies"):
            source_binding.materialize_bound_input(
                descriptor=root_descriptor,
                expected_sha256=_sha256(root_payload),
                logical_input_path=tmp_path / "logical" / "root.usda",
                dependencies=(
                    (
                        str(logical_dependency),
                        dependency_descriptor,
                        _sha256(dependency_payload),
                        str(tmp_path / "restore-a" / "dependency.usda"),
                        False,
                    ),
                    (
                        str(logical_dependency),
                        dependency_descriptor,
                        _sha256(dependency_payload),
                        str(tmp_path / "restore-b" / "dependency.usda"),
                        True,
                    ),
                ),
            )
    finally:
        os.close(dependency_descriptor)
        os.close(root_descriptor)


@pytest.mark.parametrize(
    "layer_roles",
    [(False, True), (True, False)],
    ids=["false-then-true", "true-then-false"],
)
def test_duplicate_projection_alias_merges_layer_roles_with_logical_or(
    tmp_path: Path,
    layer_roles: tuple[bool, bool],
) -> None:
    root_payload = b"#usda 1.0\n"
    root = tmp_path / "root.usda"
    dependency = tmp_path / "dependency.usda"
    root.write_bytes(root_payload)
    remote_locator = "https://example.test/dependency.usda"
    _write_asset_layer(dependency, [remote_locator])
    dependency_payload = dependency.read_bytes()
    root_descriptor = os.open(root, os.O_RDONLY)
    dependency_descriptor = os.open(dependency, os.O_RDONLY)
    logical_root = tmp_path / "logical" / "root.usda"
    logical_dependency = logical_root.parent / "dependency.usda"
    restore_dependency = tmp_path / "source" / "dependency.usda"

    try:
        with pytest.raises(JointRiggerBackendIncompatibleError) as caught:
            source_binding.materialize_bound_input(
                descriptor=root_descriptor,
                expected_sha256=_sha256(root_payload),
                logical_input_path=logical_root,
                dependencies=(
                    (
                        str(logical_dependency),
                        dependency_descriptor,
                        _sha256(dependency_payload),
                        str(restore_dependency),
                        layer_roles[0],
                    ),
                    (
                        str(logical_dependency),
                        dependency_descriptor,
                        _sha256(dependency_payload),
                        str(restore_dependency),
                        layer_roles[1],
                    ),
                ),
            )

        assert "'remote'" in str(caught.value)
        assert remote_locator in str(caught.value)
    finally:
        os.close(dependency_descriptor)
        os.close(root_descriptor)


def test_restore_projection_preserves_empty_and_remote_asset_paths(
    tmp_path: Path,
) -> None:
    projection_root = tmp_path / "projection"
    projection_root.mkdir()
    output = projection_root / "root.usda"
    remote = "https://example.test/assets/remote.usda"
    _write_asset_layer(output, ["", remote])

    source_binding.restore_bound_projection_paths(
        output,
        projection_root=projection_root,
        logical_output_parent=tmp_path / "published",
        restore_paths={},
    )

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    observed = (
        stage.GetPrimAtPath("/Root").GetAttribute("sourceBinding:testAssets").Get()
    )
    assert [str(item.path) for item in observed] == ["", remote]


@pytest.mark.parametrize("descriptor_rebound", [False, True])
def test_restore_projection_preserves_already_correct_lexical_asset_path(
    tmp_path: Path,
    descriptor_rebound: bool,
) -> None:
    projection_root = tmp_path / "projection"
    projected_asset = projection_root / "SubUSDs" / "materials" / "Material.mdl"
    projected_asset.parent.mkdir(parents=True)
    projected_asset.write_text("mdl 1.7;\n", encoding="utf-8")
    output = projection_root / "root.usda"
    locator = "./SubUSDs/materials/Material.mdl"
    _write_asset_layer(output, [locator])
    logical_output_parent = tmp_path / "published"
    original_asset = logical_output_parent / "SubUSDs" / "materials" / "Material.mdl"
    output_descriptor = os.open(output, os.O_RDWR) if descriptor_rebound else None

    try:
        source_binding.restore_bound_projection_paths(
            output,
            projection_root=projection_root,
            logical_output_parent=logical_output_parent,
            restore_paths={projected_asset: original_asset},
            output_descriptor=output_descriptor,
        )
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)

    layer = Sdf.Layer.OpenAsAnonymous(str(output))
    assert layer is not None
    attribute = layer.GetPropertyAtPath("/Root.sourceBinding:testAssets")
    assert attribute.default[0].path == locator
    assert f"@{locator}@" in attribute.GetAsText()


def test_restore_projection_rebases_asset_path_for_different_logical_parent(
    tmp_path: Path,
) -> None:
    projection_root = tmp_path / "projection"
    projected_asset = projection_root / "sidecars" / "Material.mdl"
    projected_asset.parent.mkdir(parents=True)
    projected_asset.write_text("mdl 1.7;\n", encoding="utf-8")
    output = projection_root / "root.usda"
    _write_asset_layer(output, ["./sidecars/Material.mdl"])
    logical_output_parent = tmp_path / "published"
    original_asset = tmp_path / "source" / "materials" / "Material.mdl"

    source_binding.restore_bound_projection_paths(
        output,
        projection_root=projection_root,
        logical_output_parent=logical_output_parent,
        restore_paths={projected_asset: original_asset},
    )

    layer = Sdf.Layer.OpenAsAnonymous(str(output))
    assert layer is not None
    attribute = layer.GetPropertyAtPath("/Root.sourceBinding:testAssets")
    assert attribute.default[0].path == os.path.relpath(
        original_asset,
        logical_output_parent,
    ).replace("\\", "/")


def test_restore_projection_preserves_stale_informational_identifier(
    tmp_path: Path,
) -> None:
    projection_root = tmp_path / "projection"
    projection_root.mkdir()
    output = projection_root / "root.usda"
    stage = Usd.Stage.CreateNew(str(output))
    root = stage.DefinePrim("/Root")
    stage.SetDefaultPrim(root)
    identifier = Sdf.AssetPath("./missing-export-source.usd")
    root.SetAssetInfoByKey("identifier", identifier)
    root.SetAssetInfoByKey("name", "source model")
    assert stage.GetRootLayer().Save()
    del stage

    source_binding.restore_bound_projection_paths(
        output,
        projection_root=projection_root,
        logical_output_parent=tmp_path / "published",
        restore_paths={},
    )

    restored = Usd.Stage.Open(str(output))
    assert restored is not None
    restored_root = restored.GetPrimAtPath("/Root")
    assert restored_root.GetAssetInfoByKey("identifier") == identifier
    assert restored_root.GetAssetInfoByKey("name") == "source model"


def test_informational_identifier_restore_re_resolves_replaced_prim_spec() -> None:
    layer = Sdf.Layer.CreateAnonymous("replaced-identifier.usda")
    original = Sdf.PrimSpec(layer, "Root", Sdf.SpecifierDef, "Xform")
    identifier = Sdf.AssetPath("./original-source.usd")
    original.SetInfo(
        "assetInfo",
        {"identifier": identifier, "name": "source model"},
    )

    removed = source_binding._take_informational_asset_identifiers(layer)
    del layer.rootPrims["Root"]
    replacement = Sdf.PrimSpec(layer, "Root", Sdf.SpecifierDef, "Xform")

    source_binding._restore_informational_asset_identifiers(layer, removed)

    assert replacement.GetInfo("assetInfo") == {
        "identifier": identifier,
        "name": "source model",
    }


def test_restore_projection_still_rejects_matching_runtime_dependency(
    tmp_path: Path,
) -> None:
    projection_root = tmp_path / "projection"
    projection_root.mkdir()
    output = projection_root / "root.usda"
    missing = "./missing-export-source.usd"
    _write_asset_layer(output, [missing])
    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    stage.GetPrimAtPath("/Root").SetAssetInfoByKey(
        "identifier",
        Sdf.AssetPath(missing),
    )
    assert stage.GetRootLayer().Save()
    del stage

    with pytest.raises(RuntimeError, match="could not be resolved"):
        source_binding.restore_bound_projection_paths(
            output,
            projection_root=projection_root,
            logical_output_parent=tmp_path / "published",
            restore_paths={},
        )


def test_remove_bound_input_directory_accepts_already_removed_path(
    tmp_path: Path,
) -> None:
    owned = source_binding._create_bound_input_directory(
        parent_path=tmp_path,
        prefix="already-removed-",
    )
    owned.path.rmdir()

    source_binding.remove_bound_input_directory(owned)
    source_binding.remove_bound_input_directory(owned)

    assert not owned.path.exists()
    assert owned.closed


def test_remove_bound_input_directory_never_chmods_symlink_target(
    tmp_path: Path,
) -> None:
    owned = source_binding._create_bound_input_directory(
        parent_path=tmp_path,
        prefix="projection-",
    )
    projection = owned.path
    victim = tmp_path / "external-victim"
    victim.write_bytes(b"external")
    victim.chmod(0o444)
    (projection / "escape").symlink_to(victim)

    source_binding.remove_bound_input_directory(owned)

    assert not projection.exists()
    assert victim.read_bytes() == b"external"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o444


def test_remove_bound_input_directory_preserves_precleanup_replacement(
    tmp_path: Path,
) -> None:
    owned = source_binding._create_bound_input_directory(
        parent_path=tmp_path,
        prefix="projection-",
    )
    (owned.path / "owned.txt").write_text("owned", encoding="utf-8")
    moved = tmp_path / "moved-owned-projection"
    owned.path.rename(moved)
    owned.path.mkdir()
    (owned.path / "foreign.txt").write_text("foreign", encoding="utf-8")

    with pytest.raises(
        JointRiggerArtifactError,
        match="replacement preserved",
    ):
        source_binding.remove_bound_input_directory(owned)

    assert (moved / "owned.txt").read_text(encoding="utf-8") == "owned"
    assert (owned.path / "foreign.txt").read_text(encoding="utf-8") == "foreign"


def test_bound_input_directory_child_open_emfile_removes_created_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "a" * 24
    name = f"projection-{token}"
    real_open = source_binding.os.open
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)

    def fail_child_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == name and dir_fd is not None:
            raise OSError(errno.EMFILE, "synthetic child descriptor exhaustion")
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(source_binding.os, "open", fail_child_open)

    with pytest.raises(OSError, match="synthetic child descriptor exhaustion"):
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert not (tmp_path / name).exists()
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_bound_input_directory_failed_open_preserves_raced_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "b" * 24
    name = f"projection-{token}"
    original = tmp_path / name
    moved = tmp_path / "moved-created-directory"
    real_open = source_binding.os.open
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)
    swapped = False

    def swap_before_child_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == name and dir_fd is not None:
            swapped = True
            original.rename(moved)
            original.mkdir()
            (original / "foreign.txt").write_text("foreign", encoding="utf-8")
            raise OSError(errno.EMFILE, "synthetic raced child open failure")
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(source_binding.os, "open", swap_before_child_open)

    with pytest.raises(OSError, match="synthetic raced child open failure") as caught:
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert swapped
    assert moved.is_dir()
    assert (original / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert "replacement preserved" in "\n".join(caught.value.__notes__)
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_bound_input_directory_stat_failure_without_baseline_preserves_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "c" * 24
    name = f"projection-{token}"
    real_stat = source_binding.os.stat
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)
    failed = False

    def fail_first_child_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal failed
        if path == name and dir_fd is not None and not failed:
            failed = True
            raise OSError(errno.EIO, "synthetic transient child stat failure")
        return real_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(source_binding.os, "stat", fail_first_child_stat)

    with pytest.raises(
        OSError,
        match="synthetic transient child stat failure",
    ) as caught:
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert failed
    assert stat.S_ISDIR(os.lstat(tmp_path / name).st_mode)
    assert "identity was never retained" in "\n".join(caught.value.__notes__)
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_bound_input_directory_first_stat_failure_preserves_raced_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "1" * 24
    name = f"projection-{token}"
    original = tmp_path / name
    moved = tmp_path / "moved-before-first-stat"
    real_stat = source_binding.os.stat
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)
    swapped = False

    def swap_and_fail_first_child_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal swapped
        if path == name and dir_fd is not None and not swapped:
            swapped = True
            original.rename(moved)
            original.mkdir()
            (original / "foreign.txt").write_text("foreign", encoding="utf-8")
            raise OSError(errno.EIO, "synthetic raced first child stat failure")
        return real_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(
        source_binding.os,
        "stat",
        swap_and_fail_first_child_stat,
    )

    with pytest.raises(
        OSError,
        match="synthetic raced first child stat failure",
    ) as caught:
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert swapped
    assert moved.is_dir()
    assert (original / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert "identity was never retained" in "\n".join(caught.value.__notes__)
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_bound_input_directory_transient_fstat_failure_removes_created_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "d" * 24
    name = f"projection-{token}"
    child_path = tmp_path / name
    real_fstat = source_binding.os.fstat
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)
    failed = False

    def fail_first_child_fstat(descriptor: int) -> os.stat_result:
        nonlocal failed
        metadata = real_fstat(descriptor)
        if (
            child_path.exists()
            and metadata.st_ino == child_path.stat().st_ino
            and not failed
        ):
            failed = True
            raise OSError(errno.EIO, "synthetic transient child fstat failure")
        return metadata

    monkeypatch.setattr(source_binding.os, "fstat", fail_first_child_fstat)

    with pytest.raises(OSError, match="synthetic transient child fstat failure"):
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert failed
    assert not child_path.exists()
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_bound_input_directory_persistent_fstat_failure_uses_retained_name_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "f" * 24
    name = f"projection-{token}"
    child_path = tmp_path / name
    real_fstat = source_binding.os.fstat
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)

    def fail_child_fstat(descriptor: int) -> os.stat_result:
        metadata = real_fstat(descriptor)
        if child_path.exists() and metadata.st_ino == child_path.stat().st_ino:
            raise OSError(errno.EIO, "synthetic persistent child fstat failure")
        return metadata

    monkeypatch.setattr(source_binding.os, "fstat", fail_child_fstat)

    with pytest.raises(OSError, match="synthetic persistent child fstat failure"):
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert not child_path.exists()
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_bound_input_directory_unknown_identity_failure_preserves_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "0" * 24
    name = f"projection-{token}"
    child_path = tmp_path / name
    real_stat = source_binding.os.stat
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)

    def fail_every_child_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if path == name and dir_fd is not None:
            raise OSError(errno.EIO, "synthetic persistent child stat failure")
        return real_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(source_binding.os, "stat", fail_every_child_stat)

    with pytest.raises(
        OSError, match="synthetic persistent child stat failure"
    ) as caught:
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert stat.S_ISDIR(os.lstat(child_path).st_mode)
    assert "identity was never retained" in "\n".join(caught.value.__notes__)
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_bound_input_directory_post_open_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "e" * 24
    name = f"projection-{token}"
    original = tmp_path / name
    moved = tmp_path / "moved-retained-directory"
    real_stat = source_binding.os.stat
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)
    child_stats = 0

    def swap_on_final_child_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal child_stats
        if path == name and dir_fd is not None:
            child_stats += 1
            if child_stats == 2:
                original.rename(moved)
                original.mkdir()
                (original / "foreign.txt").write_text("foreign", encoding="utf-8")
        return real_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(source_binding.os, "stat", swap_on_final_child_stat)

    with pytest.raises(
        JointRiggerArtifactError,
        match="name changed after retention",
    ) as caught:
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert child_stats >= 2
    assert moved.is_dir()
    assert (original / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert "replacement preserved" in "\n".join(caught.value.__notes__)
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_bound_input_directory_rechecks_name_after_final_parent_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "2" * 24
    name = f"projection-{token}"
    original = tmp_path / name
    moved = tmp_path / "moved-during-parent-validation"
    real_require_parent = source_binding._require_bound_directory_unchanged
    monkeypatch.setattr(source_binding.secrets, "token_hex", lambda _size: token)
    validations = 0

    def swap_during_final_parent_validation(parent: object) -> None:
        nonlocal validations
        real_require_parent(parent)
        validations += 1
        if validations == 2:
            original.rename(moved)
            original.mkdir()
            (original / "foreign.txt").write_text("foreign", encoding="utf-8")

    monkeypatch.setattr(
        source_binding,
        "_require_bound_directory_unchanged",
        swap_during_final_parent_validation,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="name changed after retention",
    ) as caught:
        source_binding._create_bound_input_directory(
            parent_path=tmp_path,
            prefix="projection-",
        )

    assert validations == 2
    assert moved.is_dir()
    assert (original / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert "replacement preserved" in "\n".join(caught.value.__notes__)
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))


def test_frozen_projection_root_rejects_stable_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    moved_source = tmp_path / "validated-source.bin"
    target_directory = tmp_path / "target"
    target_directory.mkdir()
    target = target_directory / "target.bin"
    source.write_bytes(b"validated")
    real_copy = source_binding._copy_descriptor_bytes

    def copy_then_replace_source(
        source_descriptor: int,
        target_descriptor: int,
        *,
        expected_source: os.stat_result,
        label: str,
    ) -> None:
        real_copy(
            source_descriptor,
            target_descriptor,
            expected_source=expected_source,
            label=label,
        )
        source.parent.chmod(0o700)
        source.rename(moved_source)
        source.write_bytes(b"unvalidated replacement")

    monkeypatch.setattr(
        source_binding,
        "_copy_descriptor_bytes",
        copy_then_replace_source,
    )

    with pytest.raises(
        JointRiggerArtifactError,
        match="Frozen projected root changed",
    ):
        with source_binding.freeze_bound_projection_root(source) as frozen_source:
            source_binding.copy_regular_file_to_new_path(
                source,
                target,
                label="frozen source",
                frozen_source=frozen_source,
            )

    assert not target.exists()
    assert moved_source.read_bytes() == b"validated"
    assert source.read_bytes() == b"unvalidated replacement"


def test_copy_regular_file_reports_missing_source_without_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.bin"

    with pytest.raises(FileNotFoundError):
        source_binding.copy_regular_file_to_new_path(
            tmp_path / "missing.bin",
            target,
            label="missing fixture",
        )

    assert not target.exists()


def test_copy_regular_file_preserves_primary_when_target_was_unlinked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"source")
    primary_error = JointRiggerArtifactError("forced descriptor-copy failure")

    def unlink_target_then_fail(*_args: object, **_kwargs: object) -> None:
        target.unlink()
        raise primary_error

    monkeypatch.setattr(
        source_binding,
        "_copy_descriptor_bytes",
        unlink_target_then_fail,
    )

    with pytest.raises(JointRiggerArtifactError) as caught:
        source_binding.copy_regular_file_to_new_path(
            source,
            target,
            label="adversarial fixture",
        )

    assert caught.value is primary_error
    assert not target.exists()


def test_copy_regular_file_notes_close_failure_and_removes_owned_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"source")
    primary_error = JointRiggerArtifactError("forced descriptor-copy failure")
    real_close = source_binding.os.close
    close_count = 0

    def fail_first_close_after_closing(descriptor: int) -> None:
        nonlocal close_count
        real_close(descriptor)
        close_count += 1
        if close_count == 1:
            raise OSError("forced target close audit failure")

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise primary_error

    monkeypatch.setattr(source_binding, "_copy_descriptor_bytes", fail_copy)
    monkeypatch.setattr(source_binding.os, "close", fail_first_close_after_closing)

    with pytest.raises(JointRiggerArtifactError) as caught:
        source_binding.copy_regular_file_to_new_path(
            source,
            target,
            label="adversarial fixture",
        )

    assert caught.value is primary_error
    assert not target.exists()
    assert close_count == 3
    assert "forced target close audit failure" in "\n".join(
        getattr(caught.value, "__notes__", ())
    )


def test_copy_regular_file_surfaces_close_failure_after_successful_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    payload = b"stable source"
    source.write_bytes(payload)
    real_close = source_binding.os.close
    close_count = 0

    def fail_first_close_after_closing(descriptor: int) -> None:
        nonlocal close_count
        real_close(descriptor)
        close_count += 1
        if close_count == 1:
            raise OSError("forced target close audit failure")

    monkeypatch.setattr(source_binding.os, "close", fail_first_close_after_closing)

    with pytest.raises(OSError, match="forced target close audit failure"):
        source_binding.copy_regular_file_to_new_path(
            source,
            target,
            label="stable fixture",
        )

    assert close_count == 3
    assert target.read_bytes() == payload


@pytest.mark.parametrize("operation", ["copy", "write"])
def test_failed_new_file_cleanup_preserves_same_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    moved_owned = tmp_path / "owned-moved.bin"
    source.write_bytes(b"source")
    primary_error = JointRiggerArtifactError(f"forced {operation} failure")
    real_rename = artifacts_module._rename_descriptor_entry_noreplace
    swapped = False

    def swap_before_cleanup_quarantine(
        source_parent_descriptor: int,
        source_name: str,
        target_parent_descriptor: int,
        target_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal swapped
        if (
            not swapped
            and source_name == target.name
            and target_name.startswith(".joint-rigger.cleanup-")
        ):
            os.rename(
                source_name,
                moved_owned.name,
                src_dir_fd=source_parent_descriptor,
                dst_dir_fd=source_parent_descriptor,
            )
            foreign_descriptor = os.open(
                source_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=source_parent_descriptor,
            )
            try:
                os.write(foreign_descriptor, b"foreign")
            finally:
                os.close(foreign_descriptor)
            swapped = True
        real_rename(
            source_parent_descriptor,
            source_name,
            target_parent_descriptor,
            target_name,
            label=label,
        )

    monkeypatch.setattr(
        artifacts_module,
        "_rename_descriptor_entry_noreplace",
        swap_before_cleanup_quarantine,
    )
    if operation == "copy":

        def fail_copy(*_args: object, **_kwargs: object) -> None:
            raise primary_error

        monkeypatch.setattr(source_binding, "_copy_descriptor_bytes", fail_copy)
    else:

        def fail_fsync(_descriptor: int) -> None:
            raise primary_error

        monkeypatch.setattr(source_binding.os, "fsync", fail_fsync)

    with pytest.raises(JointRiggerArtifactError) as caught:
        if operation == "copy":
            source_binding.copy_regular_file_to_new_path(
                source,
                target,
                label="adversarial copy",
            )
        else:
            source_binding.write_new_text_file(
                target,
                "report",
                label="adversarial report",
            )

    assert caught.value is primary_error
    assert swapped
    assert moved_owned.exists()
    assert target.read_bytes() == b"foreign"
    assert not list(tmp_path.glob(".joint-rigger.cleanup-*"))
    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    assert "cleanup also failed" in notes


def test_remove_empty_created_directory_accepts_already_missing_name(
    tmp_path: Path,
) -> None:
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        source_binding._remove_empty_created_bound_input_directory(
            parent_descriptor=parent_descriptor,
            parent_path=tmp_path,
            name="already-missing",
            expected_identity=(0, 0),
        )
    finally:
        os.close(parent_descriptor)


def test_rebound_output_preserves_primary_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output.usda"
    output.write_text("#usda 1.0\n", encoding="utf-8")
    output_descriptor = os.open(output, os.O_RDWR)
    primary_error = RuntimeError("forced rebound export failure")
    cleanup_error = OSError("forced rebound cleanup failure")
    real_remove = source_binding.remove_bound_input_directory

    def fail_export_format(*_args: object, **_kwargs: object) -> str:
        raise primary_error

    def remove_then_fail(directory: source_binding.BoundInputDirectory) -> None:
        real_remove(directory)
        raise cleanup_error

    monkeypatch.setattr(
        source_binding,
        "_concrete_usd_export_format",
        fail_export_format,
    )
    monkeypatch.setattr(
        source_binding,
        "remove_bound_input_directory",
        remove_then_fail,
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            source_binding.restore_bound_projection_paths(
                output,
                projection_root=tmp_path,
                logical_output_parent=tmp_path,
                restore_paths={},
                output_descriptor=output_descriptor,
            )
    finally:
        os.close(output_descriptor)

    assert caught.value is primary_error
    assert "forced rebound cleanup failure" in "\n".join(
        getattr(caught.value, "__notes__", ())
    )


def test_rebound_output_surfaces_cleanup_failure_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output.usda"
    output.write_text("#usda 1.0\n", encoding="utf-8")
    output_descriptor = os.open(output, os.O_RDWR)
    cleanup_error = OSError("forced rebound cleanup failure")
    real_remove = source_binding.remove_bound_input_directory

    def remove_then_fail(directory: source_binding.BoundInputDirectory) -> None:
        real_remove(directory)
        raise cleanup_error

    monkeypatch.setattr(
        source_binding,
        "remove_bound_input_directory",
        remove_then_fail,
    )
    try:
        with pytest.raises(OSError) as caught:
            source_binding.restore_bound_projection_paths(
                output,
                projection_root=tmp_path,
                logical_output_parent=tmp_path,
                restore_paths={},
                output_descriptor=output_descriptor,
            )
    finally:
        os.close(output_descriptor)

    assert caught.value is cleanup_error


@pytest.mark.parametrize("remove_alias", [False, True])
def test_validation_alias_open_failure_cleans_unretained_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remove_alias: bool,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    source_descriptor = os.open(source, os.O_RDONLY)
    parent = source_binding._open_bound_directory(tmp_path)
    primary_error = OSError("forced validation alias open failure")
    real_open = source_binding.os.open
    real_close = source_binding.os.close
    real_fchmod = source_binding.os.fchmod
    alias_open_failed = False

    def fail_alias_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal alias_open_failed
        if (
            isinstance(path, str)
            and path.startswith(".joint-rigger-validation-")
            and dir_fd == parent.descriptor
            and not alias_open_failed
        ):
            alias_open_failed = True
            if remove_alias:
                os.unlink(path, dir_fd=dir_fd)
            raise primary_error
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(source_binding.os, "open", fail_alias_open)
    try:
        with pytest.raises(OSError) as caught:
            with source_binding._descriptor_projection_validation_path(
                path=source,
                parent=parent,
                descriptor=source_descriptor,
            ):
                pytest.fail("validation alias open should fail before yield")
    finally:
        real_fchmod(parent.descriptor, 0o700)
        real_close(parent.descriptor)
        real_close(source_descriptor)

    assert caught.value is primary_error
    assert alias_open_failed
    assert not list(tmp_path.glob(".joint-rigger-validation-*"))
    assert getattr(caught.value, "__notes__", ()) == ()


def test_validation_alias_cleanup_failure_is_noted_on_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    source_descriptor = os.open(source, os.O_RDONLY)
    parent = source_binding._open_bound_directory(tmp_path)
    primary_error = RuntimeError("forced validation failure")
    real_close = source_binding.os.close
    real_fchmod = source_binding.os.fchmod

    def fail_restore_write_mode(descriptor: int, mode: int) -> None:
        real_fchmod(descriptor, mode)
        if descriptor == parent.descriptor and mode == 0o700:
            raise OSError("forced validation cleanup chmod failure")

    monkeypatch.setattr(source_binding.os, "fchmod", fail_restore_write_mode)
    try:
        with pytest.raises(RuntimeError) as caught:
            with source_binding._descriptor_projection_validation_path(
                path=source,
                parent=parent,
                descriptor=source_descriptor,
            ):
                raise primary_error
    finally:
        real_fchmod(parent.descriptor, 0o700)
        real_close(parent.descriptor)
        real_close(source_descriptor)

    assert caught.value is primary_error
    assert "forced validation cleanup chmod failure" in "\n".join(
        getattr(caught.value, "__notes__", ())
    )


def test_validation_alias_surfaces_cleanup_failure_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    source_descriptor = os.open(source, os.O_RDONLY)
    parent = source_binding._open_bound_directory(tmp_path)
    real_close = source_binding.os.close
    real_fchmod = source_binding.os.fchmod
    readonly_calls = 0

    def fail_second_readonly_mode(descriptor: int, mode: int) -> None:
        nonlocal readonly_calls
        real_fchmod(descriptor, mode)
        if descriptor == parent.descriptor and mode == 0o500:
            readonly_calls += 1
            if readonly_calls == 2:
                raise OSError("forced final validation chmod failure")

    monkeypatch.setattr(source_binding.os, "fchmod", fail_second_readonly_mode)
    try:
        with pytest.raises(OSError, match="forced final validation chmod failure"):
            with source_binding._descriptor_projection_validation_path(
                path=source,
                parent=parent,
                descriptor=source_descriptor,
            ) as validation_path:
                assert validation_path.name.endswith(".usda")
    finally:
        real_fchmod(parent.descriptor, 0o700)
        real_close(parent.descriptor)
        real_close(source_descriptor)

    assert readonly_calls == 2
    assert not list(tmp_path.glob(".joint-rigger-validation-*"))


def test_freeze_rejects_reopened_root_with_different_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = tmp_path / "projection"
    projection.mkdir()
    source = projection / "source.usda"
    replacement = projection / "replacement.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    replacement.write_text("#usda 1.0\n", encoding="utf-8")
    real_open = source_binding.os.open
    reopened_descriptors: list[int] = []

    def reopen_different_inode(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            path == source.name
            and dir_fd is not None
            and flags & os.O_ACCMODE == os.O_RDONLY
        ):
            descriptor = real_open(replacement.name, flags, mode, dir_fd=dir_fd)
            reopened_descriptors.append(descriptor)
            return descriptor
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(source_binding.os, "open", reopen_different_inode)
    try:
        with pytest.raises(
            JointRiggerArtifactError,
            match="changed while reopened",
        ):
            with source_binding.freeze_bound_projection_root(
                source,
                prepare_before_freeze=lambda _descriptor: None,
            ):
                pytest.fail("mismatched reopened root should fail before yield")
    finally:
        source.chmod(0o600)

    assert len(reopened_descriptors) == 1
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(reopened_descriptors[0])
    assert closed_descriptor.value.errno == errno.EBADF


def test_freeze_parent_open_failure_skips_unowned_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    primary_error = OSError("forced parent open failure")
    close_calls: list[int] = []
    real_close = source_binding.os.close

    def fail_parent_open(_path: Path) -> object:
        raise primary_error

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(source_binding, "_open_bound_directory", fail_parent_open)
    monkeypatch.setattr(source_binding.os, "close", track_close)
    with pytest.raises(OSError) as caught:
        with source_binding.freeze_bound_projection_root(source):
            pytest.fail("parent open should fail before yield")

    assert caught.value is primary_error
    assert close_calls == []


@pytest.mark.parametrize("raise_primary", [False, True])
def test_freeze_descriptor_close_failures_preserve_exception_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raise_primary: bool,
) -> None:
    projection = tmp_path / "projection"
    projection.mkdir()
    source = projection / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    primary_error = RuntimeError("forced frozen body failure")
    real_close = source_binding.os.close
    close_calls = 0

    def fail_first_close_after_closing(descriptor: int) -> None:
        nonlocal close_calls
        real_close(descriptor)
        close_calls += 1
        if close_calls == 1:
            raise OSError("forced frozen descriptor close failure")

    monkeypatch.setattr(source_binding.os, "close", fail_first_close_after_closing)
    try:
        if raise_primary:
            with pytest.raises(RuntimeError) as caught:
                with source_binding.freeze_bound_projection_root(source):
                    raise primary_error
            assert caught.value is primary_error
            assert "forced frozen descriptor close failure" in "\n".join(
                getattr(caught.value, "__notes__", ())
            )
        else:
            with pytest.raises(
                OSError,
                match="forced frozen descriptor close failure",
            ):
                with source_binding.freeze_bound_projection_root(source):
                    pass
    finally:
        projection.chmod(0o700)
        source.chmod(0o600)

    assert close_calls == 2


def test_remove_bound_directory_skips_invalid_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = source_binding._create_bound_input_directory(
        parent_path=tmp_path,
        prefix="invalid-descriptor-",
    )
    os.close(owned.descriptor)
    owned.descriptor = -1
    close_calls: list[int] = []
    real_close = source_binding.os.close

    def track_close(descriptor: int) -> None:
        close_calls.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(source_binding.os, "close", track_close)

    with pytest.raises(OSError) as caught:
        source_binding.remove_bound_input_directory(owned)

    assert caught.value.errno == errno.EBADF
    assert -1 not in close_calls
    assert getattr(caught.value, "__notes__", ()) == ()
    owned.path.rmdir()


@pytest.mark.parametrize("raise_primary", [False, True])
def test_remove_bound_directory_close_failures_preserve_exception_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raise_primary: bool,
) -> None:
    owned = source_binding._create_bound_input_directory(
        parent_path=tmp_path,
        prefix="close-failure-",
    )
    owned_descriptor = owned.descriptor
    primary_error = RuntimeError("forced bound cleanup failure")
    real_close = source_binding.os.close

    def fail_owned_close_after_closing(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == owned_descriptor:
            raise OSError("forced bound descriptor close failure")

    monkeypatch.setattr(
        source_binding.os,
        "close",
        fail_owned_close_after_closing,
    )
    if raise_primary:

        def fail_validation(_parent: object) -> None:
            raise primary_error

        monkeypatch.setattr(
            source_binding,
            "_require_bound_directory_unchanged",
            fail_validation,
        )

    if raise_primary:
        with pytest.raises(RuntimeError) as caught:
            source_binding.remove_bound_input_directory(owned)
        assert caught.value is primary_error
        assert "forced bound descriptor close failure" in "\n".join(
            getattr(caught.value, "__notes__", ())
        )
        owned.path.rmdir()
    else:
        with pytest.raises(OSError, match="forced bound descriptor close failure"):
            source_binding.remove_bound_input_directory(owned)
        assert not owned.path.exists()


def test_write_text_accepts_target_removed_during_failure_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "report.txt"
    primary_error = RuntimeError("forced post-write validation failure")
    validations = 0
    real_require = source_binding._require_bound_directory_unchanged

    def fail_second_validation(parent: object) -> None:
        nonlocal validations
        validations += 1
        real_require(parent)
        if validations == 2:
            raise primary_error

    def unlink_then_report_missing(
        parent_descriptor: int,
        name: str,
        **_kwargs: object,
    ) -> None:
        os.unlink(name, dir_fd=parent_descriptor)
        raise FileNotFoundError(name)

    monkeypatch.setattr(
        source_binding,
        "_require_bound_directory_unchanged",
        fail_second_validation,
    )
    monkeypatch.setattr(
        source_binding,
        "_remove_descriptor_entry",
        unlink_then_report_missing,
    )

    with pytest.raises(RuntimeError) as caught:
        source_binding.write_new_text_file(target, "payload", label="report")

    assert caught.value is primary_error
    assert validations == 2
    assert not target.exists()
    assert getattr(caught.value, "__notes__", ()) == ()


def test_write_text_surfaces_descriptor_close_failure_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "report.txt"
    real_close = source_binding.os.close
    close_calls = 0

    def fail_first_close_after_closing(descriptor: int) -> None:
        nonlocal close_calls
        real_close(descriptor)
        close_calls += 1
        if close_calls == 1:
            raise OSError("forced report descriptor close failure")

    monkeypatch.setattr(source_binding.os, "close", fail_first_close_after_closing)

    with pytest.raises(OSError, match="forced report descriptor close failure"):
        source_binding.write_new_text_file(target, "payload", label="report")

    assert close_calls == 2
    assert target.read_text(encoding="utf-8") == "payload"


def test_failed_sealed_binding_closes_the_owned_memfd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    created_descriptors: list[int] = []
    real_memfd_create = source_binding._MEMFD_CREATE
    if real_memfd_create is None:
        pytest.skip("platform does not provide os.memfd_create")

    def tracked_memfd_create(_name: bytes, flags: int) -> int:
        descriptor = int(real_memfd_create(b"joint-rigger-coverage", flags))
        created_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(source_binding, "_MEMFD_CREATE", tracked_memfd_create)

    with pytest.raises(JointRiggerArtifactError, match="expected identity"):
        source_binding._create_sealed_file_binding(
            source,
            expected_sha256="0" * 64,
        )

    assert len(created_descriptors) == 1
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(created_descriptors[0])
    assert closed_descriptor.value.errno == errno.EBADF


def test_missing_memfd_support_uses_a_read_only_pinned_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text('#usda 1.0\n\ndef Xform "Root" {}\n', encoding="utf-8")
    monkeypatch.setattr(source_binding, "_MEMFD_CREATE", None)

    binding = source_binding._create_sealed_file_binding(
        source,
        expected_sha256=_sha256(source.read_bytes()),
    )
    try:
        assert binding.storage_kind == "pinned_file"
        assert binding.descriptor_state is not None
        assert fcntl.fcntl(binding.descriptor, fcntl.F_GETFL) & os.O_ACCMODE == 0
        source_binding._require_sealed_file_binding(binding)
    finally:
        os.close(binding.descriptor)


def test_large_source_uses_pinned_file_without_product_byte_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text(
        '#usda 1.0\n\ndef Xform "Root" {\n    string note = "'
        + ("x" * 2048)
        + '"\n}\n',
        encoding="utf-8",
    )
    identity = identify_usd_artifact(source, uri=str(source))
    monkeypatch.setattr(source_binding, "_MAX_MEMFD_SNAPSHOT_BYTES", 1024)
    binding = source_binding.create_sealed_source_binding(source, expected=identity)
    try:
        assert os.fstat(binding.descriptor).st_size == source.stat().st_size
        assert source.stat().st_size > source_binding._MAX_MEMFD_SNAPSHOT_BYTES
        assert binding.storage_kind == "pinned_file"
        assert binding.descriptor_state is not None
        source_binding.require_sealed_source_binding(binding)
    finally:
        source_binding.close_source_binding(binding)


def test_pinned_file_reports_source_byte_mutation_before_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    payload = b'#usda 1.0\n\ndef Xform "Root" {}\n'
    source.write_bytes(payload)
    monkeypatch.setattr(source_binding, "_MAX_MEMFD_SNAPSHOT_BYTES", 0)
    binding = source_binding._create_sealed_file_binding(
        source,
        expected_sha256=_sha256(payload),
    )
    try:
        source.write_bytes(payload + b"#")
        with pytest.raises(JointRiggerArtifactError, match="snapshot changed"):
            source_binding._require_sealed_file_binding(binding)
    finally:
        os.close(binding.descriptor)


def test_sealed_memfd_stays_immutable_after_chmod_and_proc_reopen(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text('#usda 1.0\n\ndef Xform "Root" {}\n', encoding="utf-8")
    identity = identify_usd_artifact(source, uri=str(source))
    if source_binding._MEMFD_CREATE is None:
        pytest.skip("platform does not provide os.memfd_create")

    binding = source_binding.create_sealed_source_binding(source, expected=identity)
    reopened_descriptor = -1
    try:
        os.fchmod(binding.descriptor, 0o600)
        reopened_descriptor = os.open(
            f"/proc/self/fd/{binding.descriptor}",
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        )
        with pytest.raises(OSError) as sealed_write:
            os.pwrite(reopened_descriptor, b"changed", 0)
        assert sealed_write.value.errno == errno.EPERM
        source_binding.require_sealed_source_binding(binding)
    finally:
        if reopened_descriptor >= 0:
            os.close(reopened_descriptor)
        source_binding.close_source_binding(binding)


def test_sealed_source_root_can_exceed_memfd_selection_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text(
        '#usda 1.0\n\ndef Xform "Root" {\n    string note = "'
        + ("x" * 2048)
        + '"\n}\n',
        encoding="utf-8",
    )
    identity = identify_usd_artifact(source, uri=str(source))
    monkeypatch.setattr(source_binding, "_MAX_MEMFD_SNAPSHOT_BYTES", 1024)

    binding = source_binding.create_sealed_source_binding(source, expected=identity)
    try:
        assert os.fstat(binding.descriptor).st_size == source.stat().st_size
        assert source.stat().st_size > 1024
        assert binding.storage_kind == "pinned_file"
    finally:
        source_binding.close_source_binding(binding)


def test_large_source_dependency_uses_exact_disk_snapshot_instead_of_memfd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = tmp_path / "large.bin"
    original_dependency = b"x" * 2048
    dependency.write_bytes(original_dependency)
    source = tmp_path / "source.usda"
    _write_asset_layer(source, [dependency.name])
    identity = identify_usd_artifact(source, uri=str(source))
    created_descriptors: list[int] = []
    real_memfd_create = source_binding._MEMFD_CREATE
    if real_memfd_create is None:
        pytest.skip("platform does not provide os.memfd_create")

    def tracked_memfd_create(name: bytes, flags: int) -> int:
        descriptor = int(real_memfd_create(name, flags))
        created_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(source_binding, "_MEMFD_CREATE", tracked_memfd_create)
    monkeypatch.setattr(source_binding, "_MAX_MEMFD_SNAPSHOT_BYTES", 1024)

    binding = source_binding.create_sealed_source_binding(source, expected=identity)
    directory = None
    try:
        assert len(created_descriptors) == 1
        assert len(binding.dependencies) == 1
        assert binding.dependencies[0].storage_kind == "anonymous_snapshot"
        assert binding.dependencies[0].descriptor_state is not None
        assert os.fstat(binding.dependencies[0].descriptor).st_nlink == 0
        assert (
            fcntl.fcntl(binding.dependencies[0].descriptor, fcntl.F_GETFL)
            & os.O_ACCMODE
            == os.O_RDONLY
        )

        dependency.write_bytes(b"y" * len(original_dependency))
        source_binding.require_sealed_source_binding(binding)
        _, directory, _ = source_binding.materialize_bound_input(
            descriptor=binding.descriptor,
            expected_sha256=binding.sha256,
            logical_input_path=binding.path,
            dependencies=source_binding.bound_input_dependency_snapshots(binding),
        )
        projected_dependency = (
            directory.path
            / "filesystem"
            / dependency.resolve().relative_to(dependency.anchor)
        )
        assert projected_dependency.read_bytes() == original_dependency
    finally:
        if directory is not None:
            source_binding.remove_bound_input_directory(directory)
        assert source_binding.close_source_binding(binding) == []
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(created_descriptors[0])
    assert closed_descriptor.value.errno == errno.EBADF


def test_dependency_snapshot_skips_memory_backed_temp_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_directory = tmp_path / "memory"
    disk_directory = tmp_path / "disk"
    memory_directory.mkdir()
    disk_directory.mkdir()
    source = tmp_path / "dependency.bin"
    source.write_bytes(b"dependency bytes")
    opened_directories: list[Path] = []
    observed_filesystems = iter((frozenset({"tmpfs"}), frozenset({"ext4"})))
    real_temporary_file = source_binding.tempfile.TemporaryFile

    def tracked_temporary_file(*args: Any, **kwargs: Any) -> Any:
        opened_directories.append(Path(str(kwargs["dir"])))
        return real_temporary_file(*args, **kwargs)

    monkeypatch.delenv(source_binding._DISK_SNAPSHOT_DIRECTORY_ENV, raising=False)
    monkeypatch.setattr(
        source_binding,
        "_disk_snapshot_candidate_directories",
        lambda _source_path: (memory_directory, disk_directory),
    )
    monkeypatch.setattr(
        source_binding,
        "_descriptor_filesystem_types",
        lambda _descriptor: next(observed_filesystems),
    )
    monkeypatch.setattr(
        source_binding.tempfile,
        "TemporaryFile",
        tracked_temporary_file,
    )

    binding = source_binding._create_sealed_file_binding(
        source,
        prefer_disk_snapshot=True,
    )
    try:
        assert binding.storage_kind == "anonymous_snapshot"
        assert opened_directories == [memory_directory, disk_directory]
    finally:
        os.close(binding.descriptor)


def test_dependency_snapshot_honors_configured_disk_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_directory = tmp_path / "snapshots"
    snapshot_directory.mkdir()
    source = tmp_path / "dependency.bin"
    source.write_bytes(b"dependency bytes")
    opened_directories: list[Path] = []
    real_temporary_file = source_binding.tempfile.TemporaryFile

    def tracked_temporary_file(*args: Any, **kwargs: Any) -> Any:
        opened_directories.append(Path(str(kwargs["dir"])))
        return real_temporary_file(*args, **kwargs)

    monkeypatch.setenv(
        source_binding._DISK_SNAPSHOT_DIRECTORY_ENV,
        str(snapshot_directory),
    )
    monkeypatch.setattr(
        source_binding.tempfile,
        "TemporaryFile",
        tracked_temporary_file,
    )

    binding = source_binding._create_sealed_file_binding(
        source,
        prefer_disk_snapshot=True,
    )
    try:
        assert binding.storage_kind == "anonymous_snapshot"
        assert opened_directories == [snapshot_directory]
    finally:
        os.close(binding.descriptor)


def test_dependency_snapshot_rejects_configured_memory_backed_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_directory = tmp_path / "snapshots"
    snapshot_directory.mkdir()
    source = tmp_path / "dependency.bin"
    source.write_bytes(b"dependency bytes")
    monkeypatch.setenv(
        source_binding._DISK_SNAPSHOT_DIRECTORY_ENV,
        str(snapshot_directory),
    )
    monkeypatch.setattr(
        source_binding,
        "_descriptor_filesystem_types",
        lambda _descriptor: frozenset({"tmpfs"}),
    )

    with pytest.raises(
        JointRiggerBackendIncompatibleError,
        match=source_binding._DISK_SNAPSHOT_DIRECTORY_ENV,
    ) as caught:
        source_binding._create_sealed_file_binding(
            source,
            prefer_disk_snapshot=True,
        )

    assert "memory-backed filesystem tmpfs" in str(caught.value)


def test_sealed_source_has_no_corpus_shaped_aggregate_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"a" * 400)
    second.write_bytes(b"b" * 400)
    source = tmp_path / "source.usda"
    _write_asset_layer(source, [first.name, second.name])
    identity = identify_usd_artifact(source, uri=str(source))
    monkeypatch.setattr(source_binding, "_MAX_MEMFD_SNAPSHOT_BYTES", 1024)
    binding = source_binding.create_sealed_source_binding(source, expected=identity)
    try:
        assert len(binding.dependencies) == 2
        assert all(
            dependency.storage_kind == "anonymous_snapshot"
            for dependency in binding.dependencies
        )
        assert (
            sum(
                os.fstat(dependency.descriptor).st_size
                for dependency in binding.dependencies
            )
            == 800
        )
    finally:
        assert source_binding.close_source_binding(binding) == []


@pytest.mark.parametrize(
    ("constant", "limit", "message"),
    [
        ("_MAX_BOUND_DEPENDENCY_FILES", 1, "dependency-file limit"),
        ("_MAX_BOUND_DEPENDENCY_REFERENCES", 2, "reference limit"),
    ],
)
def test_sealed_source_closure_count_limits_close_root_memfd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    limit: int,
    message: str,
) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    source = tmp_path / "source.usda"
    _write_asset_layer(source, [first.name, second.name])
    identity = identify_usd_artifact(source, uri=str(source))
    created_descriptors: list[int] = []
    real_memfd_create = source_binding._MEMFD_CREATE
    if real_memfd_create is None:
        pytest.skip("platform does not provide os.memfd_create")

    def tracked_memfd_create(name: bytes, flags: int) -> int:
        descriptor = int(real_memfd_create(name, flags))
        created_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(source_binding, "_MEMFD_CREATE", tracked_memfd_create)
    monkeypatch.setattr(source_binding, constant, limit)

    with pytest.raises(JointRiggerBackendIncompatibleError, match=message):
        source_binding.create_sealed_source_binding(source, expected=identity)

    assert len(created_descriptors) == 1
    with pytest.raises(OSError) as closed_descriptor:
        os.fstat(created_descriptors[0])
    assert closed_descriptor.value.errno == errno.EBADF


def test_sealed_binding_expands_home_before_no_follow_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    payload = b"#usda 1.0\n"
    source.write_bytes(payload)
    monkeypatch.setenv("HOME", str(tmp_path))

    binding = source_binding._create_sealed_file_binding(
        Path("~/source.usda"),
        expected_sha256=_sha256(payload),
    )
    try:
        assert binding.path == source.resolve(strict=True)
        assert binding.sha256 == _sha256(payload)
        source_binding._require_sealed_file_binding(binding)
    finally:
        os.close(binding.descriptor)


def test_usdz_projection_validation_recurses_through_extracted_package(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    _write_usdz(package, "#usda 1.0\n")

    source_binding._validate_bound_projection_dependencies(
        package,
        projection_root=tmp_path,
        materialized_paths=frozenset({package}),
        layer_paths=frozenset(),
        restore_paths={},
    )

    assert package.is_file()
    assert not list(tmp_path.glob(".asset.usdz.sealed-validation-*"))


def test_usdz_projection_preserves_specific_locator_rejection(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    remote = "https://example.test/assets/remote.usda"
    layer = tmp_path / "root.usda"
    _write_asset_layer(layer, [remote])
    _write_usdz(package, layer.read_text(encoding="utf-8"))

    with pytest.raises(JointRiggerBackendIncompatibleError) as caught:
        source_binding._validate_bound_projection_dependencies(
            package,
            projection_root=tmp_path,
            materialized_paths=frozenset({package}),
            layer_paths=frozenset(),
            restore_paths={},
        )

    assert "'remote'" in str(caught.value)
    assert remote in str(caught.value)
    assert "Could not validate sealed USDZ dependency closure" not in str(caught.value)


def test_usdz_projection_ignores_informational_asset_identifier(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    _write_usdz(
        package,
        """#usda 1.0

def Xform \"Root\" (
    assetInfo = {
        asset identifier = @./missing-original.usd@
        string name = \"Root\"
    }
)
{
}

def Xform \"IdentifierOnly\" (
    assetInfo = {
        asset identifier = @./missing-identifier-only.usd@
    }
)
{
}
""",
    )

    source_binding._validate_bound_projection_dependencies(
        package,
        projection_root=tmp_path,
        materialized_paths=frozenset({package}),
        layer_paths=frozenset(),
        restore_paths={},
    )


def test_remove_informational_asset_identifiers_recurses_nested_variants() -> None:
    layer = Sdf.Layer.CreateAnonymous("variant-identifiers.usda")
    root = Sdf.PrimSpec(layer, "Root", Sdf.SpecifierDef, "Xform")
    look = Sdf.VariantSetSpec(root, "look")
    painted = Sdf.VariantSpec(look, "painted")
    variant_child = Sdf.PrimSpec(
        painted.primSpec,
        "VariantChild",
        Sdf.SpecifierDef,
        "Xform",
    )
    quality = Sdf.VariantSetSpec(variant_child, "quality")
    high = Sdf.VariantSpec(quality, "high")
    nested_child = Sdf.PrimSpec(
        high.primSpec,
        "NestedChild",
        Sdf.SpecifierDef,
        "Xform",
    )
    prim_specs = (
        (root, "root"),
        (painted.primSpec, "variant"),
        (variant_child, "variant-child"),
        (high.primSpec, "nested-variant"),
        (nested_child, "nested-child"),
    )
    for prim_spec, label in prim_specs:
        prim_spec.SetInfo(
            "assetInfo",
            {
                "identifier": Sdf.AssetPath(f"./missing-{label}.usd"),
                "name": label,
                "version": "1",
            },
        )
    variant_asset = Sdf.AttributeSpec(
        painted.primSpec,
        "sourceBinding:variantAsset",
        Sdf.ValueTypeNames.Asset,
    )
    variant_asset.default = Sdf.AssetPath("./variant-runtime.png")
    nested_asset = Sdf.AttributeSpec(
        nested_child,
        "sourceBinding:nestedAsset",
        Sdf.ValueTypeNames.Asset,
    )
    nested_asset.default = Sdf.AssetPath("./nested-runtime.png")

    source_binding._remove_informational_asset_identifiers(layer)

    for prim_spec, label in prim_specs:
        assert prim_spec.GetInfo("assetInfo") == {
            "name": label,
            "version": "1",
        }
    assert variant_asset.default == Sdf.AssetPath("./variant-runtime.png")
    assert nested_asset.default == Sdf.AssetPath("./nested-runtime.png")


def test_usdz_projection_ignores_identifiers_in_nested_variants(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    _write_usdz(
        package,
        """#usda 1.0

def Xform "Root"
{
    variantSet "look" = {
        "painted" (
            assetInfo = {
                asset identifier = @./missing-variant.usd@
                string name = "painted"
            }
        ) {
            def Xform "VariantChild"
            {
                variantSet "quality" = {
                    "high" (
                        assetInfo = {
                            asset identifier = @./missing-nested-variant.usd@
                            string name = "high"
                        }
                    ) {
                    }
                }
            }
        }
    }
}
""",
    )

    source_binding._validate_bound_projection_dependencies(
        package,
        projection_root=tmp_path,
        materialized_paths=frozenset({package}),
        layer_paths=frozenset(),
        restore_paths={},
    )


def test_usdz_projection_preserves_nested_variant_runtime_asset_rejection(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    informational = "./missing-nested-identifier.usd"
    runtime = "./missing-nested-runtime.png"
    _write_usdz(
        package,
        f"""#usda 1.0

def Xform "Root"
{{
    variantSet "look" = {{
        "painted" (
            assetInfo = {{
                asset identifier = @{informational}@
            }}
        ) {{
            def Xform "VariantChild"
            {{
                variantSet "quality" = {{
                    "high" (
                        assetInfo = {{
                            asset identifier = @./missing-inner-identifier.usd@
                        }}
                    ) {{
                        custom asset sourceBinding:runtimeAsset = @{runtime}@
                    }}
                }}
            }}
        }}
    }}
}}
""",
    )

    with pytest.raises(JointRiggerBackendIncompatibleError) as caught:
        source_binding._validate_bound_projection_dependencies(
            package,
            projection_root=tmp_path,
            materialized_paths=frozenset({package}),
            layer_paths=frozenset(),
            restore_paths={},
        )

    assert runtime in str(caught.value)
    assert informational not in str(caught.value)
    assert "./missing-inner-identifier.usd" not in str(caught.value)


def test_usdz_projection_still_rejects_missing_runtime_asset(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    _write_usdz(
        package,
        """#usda 1.0

def Xform \"Root\"
{
    custom asset sourceBinding:runtimeAsset = @./missing-texture.png@
}
""",
    )

    with pytest.raises(JointRiggerBackendIncompatibleError) as caught:
        source_binding._validate_bound_projection_dependencies(
            package,
            projection_root=tmp_path,
            materialized_paths=frozenset({package}),
            layer_paths=frozenset(),
            restore_paths={},
        )

    assert "'missing'" in str(caught.value)
    assert "./missing-texture.png" in str(caught.value)


def test_raw_projection_classifies_every_unsupported_locator_boundary(
    tmp_path: Path,
) -> None:
    projection_root = tmp_path / "projection"
    nested = projection_root / "nested"
    nested.mkdir(parents=True)
    root = projection_root / "root.usda"
    dependency = nested / "dependency.usda"
    data = nested / "data.bin"
    remote = "https://example.test/assets/remote.usda"
    package_relative = "archive.usdz[root.usda]"
    windows_absolute = "C:/absolute.usda"
    posix_absolute = "/absolute.usda"
    escaped = "../escape.usda"
    nested_package = "payload.usdz"
    missing = "missing.usda"
    _write_asset_layer(
        root,
        [
            "",
            package_relative,
            windows_absolute,
            remote,
            posix_absolute,
            escaped,
            nested_package,
            missing,
            "nested/dependency.usda",
        ],
    )
    _write_asset_layer(dependency, ["data.bin"])
    data.write_bytes(b"data")

    with pytest.raises(JointRiggerBackendIncompatibleError) as caught:
        source_binding._validate_bound_projection_dependencies(
            root,
            projection_root=projection_root,
            materialized_paths=frozenset({root, dependency, data}),
            layer_paths=frozenset({root, dependency}),
            restore_paths={
                root: Path("/original/root.usda"),
                dependency: Path("/different-parent/dependency.usda"),
                data: Path("/different-parent/data.bin"),
            },
        )

    detail = str(caught.value)
    for category in (
        "absolute",
        "remote",
        "package",
        "escaped",
        "missing",
        "nested_symlink_alias",
        "cross_parent_layer_alias",
    ):
        assert f"'{category}'" in detail
    for locator in (
        package_relative,
        windows_absolute,
        remote,
        posix_absolute,
        escaped,
        nested_package,
        missing,
        "nested/dependency.usda",
        "data.bin",
    ):
        assert locator in detail
    assert "''" not in detail


def test_sealed_source_preserves_literal_tilde_dependency_alias(
    tmp_path: Path,
) -> None:
    literal_directory = tmp_path / "~"
    literal_directory.mkdir()
    dependency = literal_directory / "dependency.bin"
    dependency.write_bytes(b"literal tilde dependency")
    source = tmp_path / "source.usda"
    _write_asset_layer(source, ["~/dependency.bin"])
    identity = identify_usd_artifact(source, uri=str(source))

    binding = source_binding.create_sealed_source_binding(source, expected=identity)
    try:
        assert len(binding.dependencies) == 1
        assert binding.dependencies[0].path == dependency.resolve(strict=True)
        assert binding.dependencies[0].projection_paths == (dependency,)
    finally:
        assert source_binding.close_source_binding(binding) == []


def test_projection_validation_preserves_literal_tilde_locator(
    tmp_path: Path,
) -> None:
    projection_root = tmp_path / "projection"
    literal_directory = projection_root / "~"
    literal_directory.mkdir(parents=True)
    dependency = literal_directory / "dependency.bin"
    dependency.write_bytes(b"literal tilde dependency")
    source = projection_root / "source.usda"
    _write_asset_layer(source, ["~/dependency.bin"])

    source_binding._validate_bound_projection_dependencies(
        source,
        projection_root=projection_root,
        materialized_paths=frozenset({source, dependency}),
        layer_paths=frozenset({source}),
        restore_paths={},
    )


def test_cleanup_error_note_preserves_primary_error_and_all_details() -> None:
    primary = KeyboardInterrupt("primary interruption")
    cleanup_errors = [OSError("close one"), RuntimeError("close two")]

    source_binding._add_cleanup_error_note(
        primary,
        label="binding cleanup failed",
        errors=cleanup_errors,
    )

    assert getattr(primary, "__notes__", ()) == [
        "binding cleanup failed: close one; close two"
    ]
