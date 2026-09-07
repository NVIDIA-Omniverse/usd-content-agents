# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-import admission, identity inspection, and capability checks for OpenVDB 13.

The already-imported :mod:`openvdb_runtime` package and its build manifest are
the policy boundary.  Native-wheel bytes are untrusted until this module binds
them to the independently reviewed release lock and completes a post-import
identity check.
"""

from __future__ import annotations

import _imp
import hashlib
import importlib.metadata
import importlib.util
import json
import numbers
import os
import platform
import re
import stat
import sys
import tempfile
import threading
import weakref
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.machinery import (
    EXTENSION_SUFFIXES,
    ExtensionFileLoader,
    PathFinder,
    SourceFileLoader,
)
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

from .errors import (
    CapabilityUnavailableError,
    NativeOperationError,
    OpenVDBRuntimeError,
    RuntimeUnavailableError,
    RuntimeVersionError,
)
from .types import Capability, RuntimeInfo

_POLICY_PATH = Path(__file__).with_name("_build_manifest.json")
_POLICY_SHA256 = "4bb75e47c1dfd5504191a6a5afa549f8b44774c66e85d01cf79a94d2c9cd07ba"
_PACKAGED_RELEASE_LOCK_PATH = Path(__file__).with_name("_release_lock.json")
_SOURCE_RELEASE_LOCK_PATH = Path(__file__).parents[1] / "native" / "release-lock.json"
_SOURCE_LOCK_SCHEMA = "world-understanding.openvdb-native-source-lock.v1"
_RELEASE_LOCK_SCHEMA = "world-understanding.sdf-native-release-lock.v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_EXTENSION_MEMBER_PATTERN = re.compile(r"openvdb/lib/openvdb[^/]*[.]so\Z")
_PROVENANCE_MEMBER_PATTERN = re.compile(
    r"openvdb-[^/]+[.]dist-info/licenses/NATIVE_DEPENDENCY_PROVENANCE[.]json\Z"
)
_METADATA_MEMBER_PATTERN = re.compile(r"openvdb-[^/]+[.]dist-info/METADATA\Z")
_ADMISSION_LOCK = threading.RLock()
_ADMITTED_MODULE: ModuleType | None = None
_ADMITTED_SYS_MODULES: dict[str, ModuleType] | None = None
_ADMITTED_DISTRIBUTION_VERSION: str | None = None


def _cleanup_runtime_snapshot(
    directory: tempfile.TemporaryDirectory[str], descriptors: tuple[int, ...]
) -> None:
    try:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
    finally:
        directory.cleanup()


class _RuntimeSnapshot:
    """Process-private, read-only copy of authenticated runtime members."""

    def __init__(
        self,
        directory: tempfile.TemporaryDirectory[str],
        root: Path,
        directory_fd: int,
        member_fds: tuple[tuple[str, int], ...],
        identities: tuple[tuple[str, int, int, int], ...],
        extension_member: str,
    ) -> None:
        self._directory = directory
        self.root = root
        self.directory_fd = directory_fd
        self.member_fds = member_fds
        self.identities = identities
        self.extension_member = extension_member
        self._finalizer = weakref.finalize(
            self,
            _cleanup_runtime_snapshot,
            directory,
            (directory_fd, *(descriptor for _member, descriptor in member_fds)),
        )

    def loader_path(self, member: str) -> Path:
        if self.directory_fd < 0:
            raise RuntimeVersionError("authenticated OpenVDB runtime snapshot is closed")
        return Path(f"/proc/self/fd/{self.directory_fd}").joinpath(*PurePosixPath(member).parts)

    @property
    def package_root(self) -> Path:
        return self.root / "openvdb"

    @property
    def wrapper_path(self) -> Path:
        return self.package_root / "__init__.py"

    @property
    def extension_path(self) -> Path:
        return self.root.joinpath(*PurePosixPath(self.extension_member).parts)

    def close(self) -> None:
        self.directory_fd = -1
        self.member_fds = ()
        self._finalizer()


@dataclass(frozen=True)
class _AdmissionPlan:
    distribution_root: Path
    package_root: Path
    wrapper_path: Path
    wrapper_source: bytes
    extension_path: Path
    extension_member: str
    source_lock_sha256: str
    distribution_version: str
    runtime_members: tuple[tuple[str, str], ...]
    native_members: tuple[tuple[str, str], ...]
    snapshot: _RuntimeSnapshot


_ADMITTED_PLAN: _AdmissionPlan | None = None


def _load_policy() -> dict[str, Any]:
    try:
        payload = _POLICY_PATH.read_bytes()
        if hashlib.sha256(payload).hexdigest() != _POLICY_SHA256:
            raise RuntimeVersionError(
                "OpenVDB runtime policy differs from the driver-embedded identity"
            )
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeVersionError(f"unable to read OpenVDB runtime policy: {exc}") from exc
    if value.get("schema") != "world-understanding.openvdb-runtime-policy.v1":
        raise RuntimeVersionError("unsupported OpenVDB runtime policy schema")
    return value


def _load_release_lock(policy: dict[str, Any]) -> dict[str, Any]:
    path = (
        _PACKAGED_RELEASE_LOCK_PATH
        if _PACKAGED_RELEASE_LOCK_PATH.is_file()
        else _SOURCE_RELEASE_LOCK_PATH
    )
    try:
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeVersionError(f"unable to read OpenVDB native release lock: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != _RELEASE_LOCK_SCHEMA:
        raise RuntimeVersionError("unsupported OpenVDB native release-lock schema")
    expected = policy.get("openvdb_release_lock_sha256")
    actual = hashlib.sha256(data).hexdigest()
    if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
        raise RuntimeVersionError("OpenVDB runtime policy has no release-lock identity")
    if actual != expected:
        raise RuntimeVersionError(
            f"OpenVDB release-lock digest mismatch: expected {expected}, got {actual}"
        )
    return value


def _normalized_architecture() -> str:
    raw = platform.machine().lower()
    architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(raw, raw)
    if architecture not in {"x86_64", "aarch64"}:
        raise RuntimeVersionError(f"unsupported OpenVDB release architecture: {raw or 'unknown'}")
    return architecture


def _selected_promoted_release(policy: dict[str, Any]) -> dict[str, Any]:
    """Return the reviewed platform record before any native package code runs."""

    release_lock = _load_release_lock(policy)
    platforms = release_lock.get("platforms")
    selected = platforms.get(_normalized_architecture()) if isinstance(platforms, dict) else None
    if not isinstance(selected, dict) or selected.get("status") != "promoted":
        raise RuntimeVersionError("OpenVDB native artifact is not promoted for this architecture")
    return selected


def _validated_member_map(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise RuntimeVersionError(f"OpenVDB promoted artifact has no {label} inventory")
    result: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_digest, str):
            raise RuntimeVersionError(f"OpenVDB promoted {label} inventory is malformed")
        member = PurePosixPath(raw_name)
        if (
            not raw_name
            or "\\" in raw_name
            or member.is_absolute()
            or member.as_posix() != raw_name
            or any(part in {"", ".", ".."} for part in member.parts)
            or not _SHA256_PATTERN.fullmatch(raw_digest)
        ):
            raise RuntimeVersionError(f"OpenVDB promoted {label} inventory is malformed")
        result[raw_name] = raw_digest
    return result


def _distribution() -> importlib.metadata.Distribution:
    distributions = list(importlib.metadata.distributions(name="openvdb"))
    if not distributions:
        raise RuntimeUnavailableError(
            "OpenVDB 13 is unavailable; install the repository-built openvdb native wheel"
        )
    if len(distributions) != 1:
        raise RuntimeVersionError(
            "exactly one installed repository-built OpenVDB distribution is required"
        )
    return distributions[0]


def _distribution_root(distribution: importlib.metadata.Distribution) -> Path:
    try:
        raw_root = Path(distribution.locate_file(""))
        root_stat = raw_root.lstat()
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeVersionError(
            f"could not locate installed OpenVDB distribution: {exc}"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeVersionError("installed OpenVDB distribution root is not a regular directory")
    try:
        return raw_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeVersionError(
            f"could not resolve installed OpenVDB distribution: {exc}"
        ) from exc


def _read_regular_member(root: Path, member: str) -> bytes:
    current = root
    try:
        for index, part in enumerate(PurePosixPath(member).parts):
            current = current / part
            member_stat = current.lstat()
            if stat.S_ISLNK(member_stat.st_mode):
                raise RuntimeVersionError(f"installed OpenVDB member is a symlink: {member}")
            if index < len(PurePosixPath(member).parts) - 1:
                if not stat.S_ISDIR(member_stat.st_mode):
                    raise RuntimeVersionError(
                        f"installed OpenVDB member has a non-directory parent: {member}"
                    )
            elif not stat.S_ISREG(member_stat.st_mode):
                raise RuntimeVersionError(f"installed OpenVDB member is not regular: {member}")
        resolved = current.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RuntimeVersionError(
                f"installed OpenVDB member escapes its distribution: {member}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(current, flags)
        try:
            opened_stat = os.fstat(descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise RuntimeVersionError(f"installed OpenVDB member is not regular: {member}")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return stream.read()
        finally:
            os.close(descriptor)
    except RuntimeVersionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeVersionError(
            f"could not verify installed OpenVDB member {member}: {exc}"
        ) from exc


def _is_native_member(member: str) -> bool:
    name = PurePosixPath(member).name.lower()
    return name.endswith((".pyd", ".dll", ".dylib")) or ".so." in name or name.endswith(".so")


def _verify_runtime_members(
    root: Path,
    runtime_members: dict[str, str],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for member, expected_digest in runtime_members.items():
        payload = _read_regular_member(root, member)
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise RuntimeVersionError(
                f"installed OpenVDB member differs from release lock: {member}"
            )
        payloads[member] = payload
    return payloads


def _write_snapshot_member(root: Path, member: str, payload: bytes) -> tuple[int, int, int]:
    parts = PurePosixPath(member).parts
    parent = root
    try:
        for part in parts[:-1]:
            parent /= part
            try:
                parent.mkdir(mode=0o700)
            except FileExistsError:
                parent_stat = parent.lstat()
                if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
                    raise RuntimeVersionError(
                        f"authenticated OpenVDB snapshot parent is unsafe: {member}"
                    ) from None
        path = parent / parts[-1]
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags, 0o600)
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("snapshot write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
            member_stat = os.fstat(descriptor)
            if not stat.S_ISREG(member_stat.st_mode) or member_stat.st_size != len(payload):
                raise RuntimeVersionError(
                    f"authenticated OpenVDB snapshot member is malformed: {member}"
                )
            return member_stat.st_dev, member_stat.st_ino, member_stat.st_size
        finally:
            os.close(descriptor)
    except RuntimeVersionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeVersionError(
            f"could not create authenticated OpenVDB snapshot member {member}: {exc}"
        ) from exc


def _lock_snapshot_directories(root: Path) -> None:
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        try:
            directory_stat = directory.lstat()
            if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
                raise RuntimeVersionError(
                    f"authenticated OpenVDB snapshot directory is unsafe: {directory}"
                )
            os.chmod(directory, 0o500, follow_symlinks=False)
        except RuntimeVersionError:
            raise
        except OSError as exc:
            raise RuntimeVersionError(
                f"could not lock authenticated OpenVDB snapshot directory: {directory}"
            ) from exc


def _create_runtime_snapshot(
    payloads: dict[str, bytes],
    runtime_members: dict[str, str],
    extension_member: str,
) -> _RuntimeSnapshot:
    """Copy authenticated bytes away from mutable installed pathnames."""

    if payloads.keys() != runtime_members.keys():
        raise RuntimeVersionError("authenticated OpenVDB snapshot inventory is incomplete")
    directory = tempfile.TemporaryDirectory(prefix="openvdb-runtime-")
    root = Path(directory.name).resolve(strict=True)
    directory_fd = -1
    member_fds: list[tuple[str, int]] = []
    try:
        identities = tuple(
            (member, *_write_snapshot_member(root, member, payloads[member]))
            for member in sorted(runtime_members)
        )
        _verify_runtime_members(root, runtime_members)
        _scan_runtime_trees(root, set(runtime_members))
        _lock_snapshot_directories(root)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(root, flags)
        descriptor_stat = os.fstat(directory_fd)
        root_stat = root.lstat()
        if not stat.S_ISDIR(descriptor_stat.st_mode) or (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) != (root_stat.st_dev, root_stat.st_ino):
            raise RuntimeVersionError("authenticated OpenVDB snapshot root changed during creation")
        loader_root = Path(f"/proc/self/fd/{directory_fd}")
        if loader_root.resolve(strict=True) != root:
            raise RuntimeVersionError(
                "the host cannot provide descriptor-bound OpenVDB runtime loading"
            )
        expected_identities = {
            member: (device, inode, size) for member, device, inode, size in identities
        }
        member_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        for member in sorted(runtime_members):
            descriptor = os.open(root.joinpath(*PurePosixPath(member).parts), member_flags)
            member_fds.append((member, descriptor))
            member_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(member_stat.st_mode)
                or stat.S_IMODE(member_stat.st_mode) != 0o400
                or (member_stat.st_dev, member_stat.st_ino, member_stat.st_size)
                != expected_identities[member]
            ):
                raise RuntimeVersionError(
                    f"authenticated OpenVDB snapshot member changed during creation: {member}"
                )
        return _RuntimeSnapshot(
            directory,
            root,
            directory_fd,
            tuple(member_fds),
            identities,
            extension_member,
        )
    except BaseException:
        for _member, descriptor in member_fds:
            os.close(descriptor)
        if directory_fd >= 0:
            os.close(directory_fd)
        directory.cleanup()
        raise


def _verify_snapshot_handle_identity(snapshot: _RuntimeSnapshot) -> None:
    """Check retained snapshot handles without reading member contents."""

    try:
        descriptor_stat = os.fstat(snapshot.directory_fd)
        root_stat = snapshot.root.lstat()
    except OSError as exc:
        raise RuntimeVersionError("authenticated OpenVDB runtime snapshot is unavailable") from exc
    if (
        not stat.S_ISDIR(descriptor_stat.st_mode)
        or stat.S_IMODE(descriptor_stat.st_mode) != 0o500
        or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (root_stat.st_dev, root_stat.st_ino)
    ):
        raise RuntimeVersionError("authenticated OpenVDB runtime snapshot root was replaced")
    loader_root = Path(f"/proc/self/fd/{snapshot.directory_fd}")
    try:
        if loader_root.resolve(strict=True) != snapshot.root:
            raise RuntimeVersionError("authenticated OpenVDB runtime snapshot root was redirected")
    except OSError as exc:
        raise RuntimeVersionError("authenticated OpenVDB runtime snapshot is unavailable") from exc

    expected_identities = {
        member: (device, inode, size) for member, device, inode, size in snapshot.identities
    }
    if {member for member, _descriptor in snapshot.member_fds} != set(expected_identities):
        raise RuntimeVersionError("authenticated OpenVDB runtime snapshot handles changed")
    for member, descriptor in snapshot.member_fds:
        try:
            member_stat = os.fstat(descriptor)
        except OSError as exc:
            raise RuntimeVersionError(
                f"authenticated OpenVDB runtime snapshot handle is unavailable: {member}"
            ) from exc
        if (
            not stat.S_ISREG(member_stat.st_mode)
            or stat.S_IMODE(member_stat.st_mode) != 0o400
            or (member_stat.st_dev, member_stat.st_ino, member_stat.st_size)
            != expected_identities[member]
        ):
            raise RuntimeVersionError(
                f"authenticated OpenVDB runtime snapshot handle changed: {member}"
            )


def _verify_runtime_snapshot(plan: _AdmissionPlan) -> None:
    snapshot = plan.snapshot
    runtime_members = dict(plan.runtime_members)
    _verify_snapshot_handle_identity(snapshot)
    expected_identities = {
        member: (device, inode, size) for member, device, inode, size in snapshot.identities
    }
    actual_members: set[str] = set()
    for current, directories, filenames in os.walk(snapshot.root, followlinks=False):
        current_path = Path(current)
        current_stat = current_path.lstat()
        if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(current_stat.st_mode):
            raise RuntimeVersionError(
                "authenticated OpenVDB runtime snapshot contains unsafe paths"
            )
        if stat.S_IMODE(current_stat.st_mode) != 0o500:
            raise RuntimeVersionError(
                "authenticated OpenVDB runtime snapshot directory became writable"
            )
        for directory in directories:
            directory_stat = (current_path / directory).lstat()
            if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
                raise RuntimeVersionError(
                    "authenticated OpenVDB runtime snapshot contains unsafe paths"
                )
        for filename in filenames:
            path = current_path / filename
            member = path.relative_to(snapshot.root).as_posix()
            member_stat = path.lstat()
            if (
                stat.S_ISLNK(member_stat.st_mode)
                or not stat.S_ISREG(member_stat.st_mode)
                or stat.S_IMODE(member_stat.st_mode) != 0o400
                or (member_stat.st_dev, member_stat.st_ino, member_stat.st_size)
                != expected_identities.get(member)
            ):
                raise RuntimeVersionError(
                    f"authenticated OpenVDB runtime snapshot member was replaced: {member}"
                )
            actual_members.add(member)
    if actual_members != set(runtime_members):
        raise RuntimeVersionError("authenticated OpenVDB runtime snapshot inventory changed")
    try:
        _verify_runtime_members(snapshot.root, runtime_members)
    except RuntimeVersionError as exc:
        raise RuntimeVersionError(
            "authenticated OpenVDB runtime snapshot differs from promoted bytes"
        ) from exc


def _recorded_distribution_members(distribution: importlib.metadata.Distribution) -> set[str]:
    files = distribution.files
    if files is None:
        raise RuntimeVersionError("installed OpenVDB distribution has no wheel RECORD inventory")
    members: set[str] = set()
    for entry in files:
        try:
            name = entry.as_posix()
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeVersionError("installed OpenVDB wheel RECORD is malformed") from exc
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or path.as_posix() != name
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise RuntimeVersionError("installed OpenVDB wheel RECORD is malformed")
        members.add(name)
    return members


def _scan_runtime_trees(root: Path, expected_members: set[str]) -> None:
    """Reject executable additions while ignoring inert installer bytecode caches."""

    for tree_name in ("openvdb", "openvdb.libs"):
        tree = root / tree_name
        try:
            tree_stat = tree.lstat()
        except OSError as exc:
            raise RuntimeVersionError(
                f"installed OpenVDB runtime tree is missing: {tree_name}"
            ) from exc
        if stat.S_ISLNK(tree_stat.st_mode) or not stat.S_ISDIR(tree_stat.st_mode):
            raise RuntimeVersionError(f"installed OpenVDB runtime tree is unsafe: {tree_name}")
        for current, directories, filenames in os.walk(tree, followlinks=False):
            current_path = Path(current)
            for directory in directories:
                path = current_path / directory
                try:
                    mode = path.lstat().st_mode
                except OSError as exc:
                    raise RuntimeVersionError(
                        f"could not inspect installed OpenVDB directory: {path}"
                    ) from exc
                if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                    raise RuntimeVersionError(f"installed OpenVDB runtime path is unsafe: {path}")
            for filename in filenames:
                path = current_path / filename
                try:
                    mode = path.lstat().st_mode
                except OSError as exc:
                    raise RuntimeVersionError(
                        f"could not inspect installed OpenVDB runtime member: {path}"
                    ) from exc
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise RuntimeVersionError(f"installed OpenVDB runtime path is unsafe: {path}")
                member = path.relative_to(root).as_posix()
                suffix = path.suffix.lower()
                executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
                if suffix in {".pyc", ".pyo"} and not executable:
                    continue
                if (
                    suffix in {".py", ".pth"} or _is_native_member(member) or executable
                ) and member not in expected_members:
                    raise RuntimeVersionError(
                        f"installed OpenVDB runtime contains an unrecorded executable member: {member}"
                    )


def _zip_contains_openvdb(path: Path) -> bool:
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as archive:
            return any(
                name == "openvdb.py"
                or name.startswith("openvdb/")
                or any(name == f"openvdb{suffix}" for suffix in EXTENSION_SUFFIXES)
                for name in archive.namelist()
            )
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _reject_import_shadowing(package_root: Path) -> None:
    foreign_candidates: set[str] = set()
    for raw_entry in sys.path:
        if not isinstance(raw_entry, str):
            raise RuntimeVersionError("sys.path contains a non-filesystem OpenVDB import source")
        entry = Path.cwd() if raw_entry == "" else Path(raw_entry)
        try:
            if entry.is_dir():
                candidates = [entry / "openvdb", entry / "openvdb.py"]
                candidates.extend(entry / f"openvdb{suffix}" for suffix in EXTENSION_SUFFIXES)
                for candidate in candidates:
                    if not candidate.exists() and not candidate.is_symlink():
                        continue
                    try:
                        resolved = candidate.resolve(strict=True)
                    except OSError:
                        foreign_candidates.add(str(candidate))
                        continue
                    if resolved != package_root:
                        foreign_candidates.add(str(resolved))
            elif entry.is_file() and _zip_contains_openvdb(entry):
                foreign_candidates.add(str(entry.resolve()))
        except OSError as exc:
            raise RuntimeVersionError(
                f"could not inspect OpenVDB import path {entry}: {exc}"
            ) from exc
    if foreign_candidates:
        joined = ", ".join(sorted(foreign_candidates))
        raise RuntimeVersionError(f"OpenVDB import shadowing or multiple roots detected: {joined}")


def _validate_top_level_spec(wrapper_path: Path, package_root: Path) -> Any:
    try:
        spec = PathFinder.find_spec("openvdb")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeVersionError(f"could not resolve installed OpenVDB package: {exc}") from exc
    if (
        spec is None
        or not isinstance(spec.loader, SourceFileLoader)
        or spec.origin is None
        or spec.submodule_search_locations is None
    ):
        raise RuntimeVersionError(
            "installed OpenVDB must resolve to one filesystem source package, not a namespace or zip"
        )
    try:
        origin = Path(spec.origin).resolve(strict=True)
        locations = tuple(
            Path(location).resolve(strict=True) for location in spec.submodule_search_locations
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeVersionError(f"installed OpenVDB import spec is malformed: {exc}") from exc
    if origin != wrapper_path or locations != (package_root,):
        raise RuntimeVersionError("installed OpenVDB import spec is shadowed or has multiple roots")
    return spec


def _validate_source_lock(payload: bytes, policy: dict[str, Any]) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    expected_digest = policy.get("openvdb_source_lock_sha256")
    if digest != expected_digest:
        raise RuntimeVersionError(
            f"OpenVDB source-lock digest mismatch: expected {expected_digest}, got {digest}"
        )
    try:
        lock = json.loads(payload)
        commit = lock["source"]["commit"]
        distribution_version = lock["distribution"]["version"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeVersionError("embedded OpenVDB source lock is malformed") from exc
    if not isinstance(lock, dict) or lock.get("schema") != _SOURCE_LOCK_SCHEMA:
        raise RuntimeVersionError("embedded OpenVDB source lock has an unsupported schema")
    if commit != policy.get("openvdb_source_commit"):
        raise RuntimeVersionError("embedded OpenVDB source lock has an unexpected commit")
    if distribution_version != policy.get("openvdb_distribution_version"):
        raise RuntimeVersionError("embedded OpenVDB source lock has an unexpected version")
    return digest


def _plan_runtime_admission(policy: dict[str, Any]) -> _AdmissionPlan:
    selected = _selected_promoted_release(policy)
    runtime_members = _validated_member_map(selected.get("wheel_runtime_members"), "runtime-member")
    native_members = _validated_member_map(selected.get("wheel_native_members"), "native-member")
    if any(runtime_members.get(name) != digest for name, digest in native_members.items()):
        raise RuntimeVersionError(
            "OpenVDB promoted native inventory is not bound by its runtime inventory"
        )
    if set(runtime_members).intersection({"openvdb/__init__.py", "openvdb/_source_lock.json"}) != {
        "openvdb/__init__.py",
        "openvdb/_source_lock.json",
    }:
        raise RuntimeVersionError(
            "OpenVDB promoted runtime inventory is missing package policy files"
        )
    metadata_members = [
        name for name in runtime_members if _METADATA_MEMBER_PATTERN.fullmatch(name)
    ]
    provenance_members = [
        name for name in runtime_members if _PROVENANCE_MEMBER_PATTERN.fullmatch(name)
    ]
    extension_members = [
        name for name in native_members if _EXTENSION_MEMBER_PATTERN.fullmatch(name)
    ]
    if len(metadata_members) != 1 or len(provenance_members) != 1:
        raise RuntimeVersionError(
            "OpenVDB promoted runtime inventory must bind one metadata and provenance file"
        )
    if len(extension_members) != 1:
        raise RuntimeVersionError("OpenVDB promoted native inventory must bind one extension")

    source_lock_digest = selected.get("source_lock_sha256")
    if source_lock_digest != policy.get("openvdb_source_lock_sha256"):
        raise RuntimeVersionError("OpenVDB promoted artifact uses a different source lock")
    distribution = _distribution()
    root = _distribution_root(distribution)
    recorded_members = _recorded_distribution_members(distribution)
    if not set(runtime_members).issubset(recorded_members):
        raise RuntimeVersionError(
            "installed OpenVDB wheel RECORD does not own every promoted runtime member"
        )
    payloads = _verify_runtime_members(root, runtime_members)
    actual_source_lock = _validate_source_lock(payloads["openvdb/_source_lock.json"], policy)
    if actual_source_lock != source_lock_digest:
        raise RuntimeVersionError("OpenVDB promoted artifact uses a different source lock")
    distribution_version = distribution.version
    if distribution_version != policy.get("openvdb_distribution_version"):
        raise RuntimeVersionError(
            "OpenVDB distribution identity mismatch: "
            f"expected {policy.get('openvdb_distribution_version')}, got {distribution_version}"
        )
    _scan_runtime_trees(root, set(runtime_members))
    package_root = root / "openvdb"
    wrapper_path = package_root / "__init__.py"
    extension_member = extension_members[0]
    extension_path = root.joinpath(*PurePosixPath(extension_member).parts)
    _reject_import_shadowing(package_root)
    _validate_top_level_spec(wrapper_path, package_root)
    snapshot = _create_runtime_snapshot(payloads, runtime_members, extension_member)
    return _AdmissionPlan(
        distribution_root=root,
        package_root=package_root,
        wrapper_path=wrapper_path,
        wrapper_source=payloads["openvdb/__init__.py"],
        extension_path=extension_path,
        extension_member=extension_member,
        source_lock_sha256=actual_source_lock,
        distribution_version=distribution_version,
        runtime_members=tuple(sorted(runtime_members.items())),
        native_members=tuple(sorted(native_members.items())),
        snapshot=snapshot,
    )


def _new_namespace_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    if spec is None:
        raise RuntimeVersionError(f"could not create locked namespace for {name}")
    spec.submodule_search_locations = [str(path)]
    module = ModuleType(name)
    module.__spec__ = spec
    module.__loader__ = None
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    return module


def _import_locked_distribution(plan: _AdmissionPlan) -> ModuleType:
    """Execute only source and native files from the authenticated snapshot."""

    _verify_runtime_snapshot(plan)
    snapshot = plan.snapshot
    loader_wrapper_path = snapshot.loader_path("openvdb/__init__.py")
    loader_package_root = snapshot.loader_path("openvdb")
    wrapper_loader = SourceFileLoader("openvdb", str(loader_wrapper_path))
    top_spec = spec_from_file_location(
        "openvdb",
        loader_wrapper_path,
        loader=wrapper_loader,
        submodule_search_locations=[str(loader_package_root)],
    )
    if top_spec is None:
        raise RuntimeVersionError("could not create locked OpenVDB wrapper spec")
    wrapper = ModuleType("openvdb")
    wrapper.__file__ = str(loader_wrapper_path)
    wrapper.__cached__ = None
    wrapper.__loader__ = top_spec.loader
    wrapper.__package__ = "openvdb"
    wrapper.__path__ = [str(loader_package_root)]  # type: ignore[attr-defined]
    wrapper.__spec__ = top_spec
    library_namespace = _new_namespace_module("openvdb.lib", snapshot.loader_path("openvdb/lib"))
    extension_name = "openvdb.lib.openvdb"
    loader_extension_path = snapshot.loader_path(plan.extension_member)
    extension_loader = ExtensionFileLoader(extension_name, str(loader_extension_path))
    extension_spec = spec_from_file_location(
        extension_name,
        loader_extension_path,
        loader=extension_loader,
    )
    if extension_spec is None:
        raise RuntimeVersionError("could not create locked OpenVDB extension spec")
    sys.modules["openvdb"] = wrapper
    sys.modules["openvdb.lib"] = library_namespace
    try:
        extension = module_from_spec(extension_spec)
        sys.modules[extension_name] = extension
        extension_loader.exec_module(extension)
        library_namespace.openvdb = extension  # type: ignore[attr-defined]
        wrapper.lib = library_namespace  # type: ignore[attr-defined]
        code = compile(plan.wrapper_source, str(loader_wrapper_path), "exec", dont_inherit=True)
        exec(code, wrapper.__dict__)
    except BaseException:
        for name in tuple(sys.modules):
            if name == "openvdb" or name.startswith("openvdb."):
                sys.modules.pop(name, None)
        raise
    return wrapper


def _verify_post_import(plan: _AdmissionPlan, module: ModuleType) -> dict[str, ModuleType]:
    if sys.modules.get("openvdb") is not module:
        raise RuntimeVersionError("loaded OpenVDB wrapper was replaced during initialization")
    native = sys.modules.get("openvdb.lib.openvdb")
    if not isinstance(native, ModuleType):
        raise RuntimeVersionError("loaded OpenVDB extension is missing after initialization")
    try:
        native_path = Path(native.__file__).resolve(strict=True)  # type: ignore[arg-type]
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise RuntimeVersionError("loaded OpenVDB extension path is malformed") from exc
    if native_path != plan.snapshot.extension_path:
        raise RuntimeVersionError("loaded OpenVDB extension differs from the admitted path")
    runtime_members = dict(plan.runtime_members)
    if (
        hashlib.sha256(_read_regular_member(plan.snapshot.root, plan.extension_member)).hexdigest()
        != runtime_members[plan.extension_member]
    ):
        raise RuntimeVersionError("loaded OpenVDB extension differs from promoted bytes")
    _verify_runtime_snapshot(plan)
    _verify_runtime_members(plan.distribution_root, runtime_members)
    loaded: dict[str, ModuleType] = {}
    for name, value in sys.modules.items():
        if name != "openvdb" and not name.startswith("openvdb."):
            continue
        if not isinstance(value, ModuleType):
            raise RuntimeVersionError(f"loaded OpenVDB module entry is malformed: {name}")
        if name not in {"openvdb", "openvdb.lib", "openvdb.lib.openvdb"} and not name.startswith(
            "openvdb.lib.openvdb."
        ):
            raise RuntimeVersionError(f"unexpected OpenVDB module was loaded: {name}")
        loaded[name] = value
    return loaded


def _cached_admitted_module() -> ModuleType | None:
    if _ADMITTED_MODULE is None:
        return None
    if not _ADMITTED_SYS_MODULES:
        raise RuntimeVersionError("cached OpenVDB admission state is malformed")
    if any(sys.modules.get(name) is not value for name, value in _ADMITTED_SYS_MODULES.items()):
        raise RuntimeVersionError("admitted OpenVDB modules were replaced")
    _revalidate_cached_admission()
    return _ADMITTED_MODULE


def _revalidate_cached_admission() -> None:
    """Validate retained snapshot identity without re-reading native contents."""

    if _ADMITTED_PLAN is None:
        raise RuntimeVersionError("cached OpenVDB admission plan is missing")
    _verify_snapshot_handle_identity(_ADMITTED_PLAN.snapshot)


def _deep_revalidate_admission() -> None:
    """Re-hash the private snapshot and installed wheel for diagnostics."""

    if _ADMITTED_PLAN is None:
        raise RuntimeVersionError("cached OpenVDB admission plan is missing")
    current = {
        name: value
        for name, value in sys.modules.items()
        if name == "openvdb" or name.startswith("openvdb.")
    }
    if current != _ADMITTED_SYS_MODULES:
        raise RuntimeVersionError("admitted OpenVDB modules were replaced or extended")
    _verify_runtime_snapshot(_ADMITTED_PLAN)
    runtime_members = dict(_ADMITTED_PLAN.runtime_members)
    _verify_runtime_members(_ADMITTED_PLAN.distribution_root, runtime_members)
    _scan_runtime_trees(_ADMITTED_PLAN.distribution_root, set(runtime_members))


def load_openvdb() -> ModuleType:
    """Admit, import, and return the exact promoted in-process runtime.

    Failures are not cached.  No ``openvdb`` wrapper or extension code runs
    before the full installed runtime inventory has been admitted. Cached reuse
    validates retained snapshot handles without re-reading native contents;
    :func:`inspect_runtime` performs explicit deep revalidation.
    """

    global _ADMITTED_DISTRIBUTION_VERSION, _ADMITTED_MODULE, _ADMITTED_PLAN
    global _ADMITTED_SYS_MODULES
    with _ADMISSION_LOCK:
        cached = _cached_admitted_module()
        if cached is not None:
            return cached
        existing = [
            name for name in sys.modules if name == "openvdb" or name.startswith("openvdb.")
        ]
        if existing:
            raise RuntimeVersionError(
                "OpenVDB was imported before runtime admission; start a clean interpreter"
            )
        policy = _load_policy()
        plan = _plan_runtime_admission(policy)
        try:
            _imp.acquire_lock()
        except BaseException:
            plan.snapshot.close()
            raise
        try:
            existing = [
                name for name in sys.modules if name == "openvdb" or name.startswith("openvdb.")
            ]
            if not existing:
                try:
                    module = _import_locked_distribution(plan)
                    loaded = _verify_post_import(plan, module)
                except (ImportError, OSError) as exc:
                    plan.snapshot.close()
                    raise RuntimeUnavailableError(
                        "OpenVDB 13 is unavailable; install the repository-built openvdb native wheel"
                    ) from exc
                except BaseException:
                    for name in tuple(sys.modules):
                        if name == "openvdb" or name.startswith("openvdb."):
                            sys.modules.pop(name, None)
                    plan.snapshot.close()
                    raise
                _ADMITTED_MODULE = module
                _ADMITTED_SYS_MODULES = loaded
                _ADMITTED_DISTRIBUTION_VERSION = plan.distribution_version
                _ADMITTED_PLAN = plan
                return module
        finally:
            _imp.release_lock()
        plan.snapshot.close()
        raise RuntimeVersionError(
            "OpenVDB was imported before runtime admission; start a clean interpreter"
        )


def _callable_attribute(value: object, name: str) -> bool:
    return callable(getattr(value, name, None))


def _toolset(module: ModuleType) -> object | None:
    return getattr(module, "tools", None)


def _call_native(operation: str, function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Invoke a binding entry point without leaking backend exception types."""

    try:
        return function(*args, **kwargs)
    except OpenVDBRuntimeError:
        raise
    except (ArithmeticError, IndexError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise NativeOperationError(f"OpenVDB {operation} failed: {exc}") from exc


def _tools_api_version(module: ModuleType) -> int:
    value = getattr(_toolset(module), "API_VERSION", 0)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return 0
    return max(0, int(value))


def detect_capabilities(module: ModuleType | None = None) -> frozenset[Capability]:
    """Inspect the loaded module without invoking native geometry operations."""

    module = module or load_openvdb()
    capabilities: set[Capability] = set()
    float_grid = getattr(module, "FloatGrid", None)
    tools = _toolset(module)

    if _callable_attribute(module, "createLinearTransform"):
        capabilities.add(Capability.TRANSFORMS)
    if all(
        _callable_attribute(module, name)
        for name in (
            "read",
            "readAll",
            "readGridMetadata",
            "readAllGridMetadata",
            "write",
        )
    ):
        capabilities.add(Capability.VDB_IO)
    if float_grid is not None and all(
        _callable_attribute(float_grid, name) for name in ("copyFromArray", "copyToArray")
    ):
        capabilities.add(Capability.NUMPY_TRANSFER)
    if (
        float_grid is not None and _callable_attribute(float_grid, "createLevelSetFromPolygons")
    ) or (tools is not None and _callable_attribute(tools, "mesh_to_level_set")):
        capabilities.add(Capability.MESH_TO_LEVEL_SET)
    if (float_grid is not None and _callable_attribute(float_grid, "convertToPolygons")) or (
        tools is not None and _callable_attribute(tools, "volume_to_mesh")
    ):
        capabilities.add(Capability.VOLUME_TO_MESH)
    tool_capabilities = {
        Capability.MESH_TO_UNSIGNED_DISTANCE_FIELD: "mesh_to_unsigned_distance_field",
        Capability.LEVEL_SET_NORMALIZE: "level_set_normalize",
        Capability.LEVEL_SET_REBUILD: "level_set_rebuild",
        Capability.RESAMPLE_TO_MATCH: "resample_to_match",
        Capability.SAMPLE_VALUES: "sample_values",
        Capability.SAMPLE_GRADIENTS: "sample_gradients",
        Capability.ACTIVE_VALUE_MASK: "active_value_mask",
        Capability.TOPOLOGY_TO_LEVEL_SET: "topology_to_level_set",
        Capability.EXTRACT_ENCLOSED_REGION: "extract_enclosed_region",
        Capability.SCALAR_MEAN_FILTER: "scalar_mean",
    }
    if tools is not None:
        for capability, name in tool_capabilities.items():
            if _callable_attribute(tools, name):
                capabilities.add(capability)
        if _callable_attribute(tools, "volume_to_mesh") and _tools_api_version(module) >= 2:
            capabilities.add(Capability.EXTENDED_VOLUME_TO_MESH)
    if tools is not None and all(
        _callable_attribute(tools, name)
        for name in ("csg_union", "csg_intersection", "csg_difference")
    ):
        capabilities.add(Capability.CSG)
    if tools is not None and _callable_attribute(tools, "level_set_offset"):
        capabilities.add(Capability.LEVEL_SET_OFFSET)
    if tools is not None and _callable_attribute(tools, "level_set_mean"):
        capabilities.add(Capability.LEVEL_SET_FILTER)
    return frozenset(capabilities)


def _native_module(module: ModuleType) -> ModuleType:
    """Resolve the nanobind extension behind the public ``openvdb`` wrapper."""

    if getattr(module, "__name__", None) == "openvdb.lib.openvdb":
        return module
    if getattr(module, "__name__", None) != "openvdb":
        return module
    candidate = sys.modules.get("openvdb.lib.openvdb")
    if not isinstance(candidate, ModuleType):
        return module
    wrapper_path = getattr(module, "__file__", None)
    candidate_path = getattr(candidate, "__file__", None)
    if not wrapper_path or not candidate_path:
        return module
    try:
        if not Path(candidate_path).resolve().is_relative_to(Path(wrapper_path).resolve().parent):
            return module
    except (OSError, ValueError):
        return module
    return candidate


def _module_digest(module: ModuleType) -> tuple[str | None, str | None]:
    native_module = _native_module(module)
    raw_path = getattr(native_module, "__file__", None)
    if not raw_path:
        return None, None
    path = Path(raw_path).resolve()
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    return str(path), digest


def _source_lock_identity(
    module: ModuleType,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Read identity from the source lock embedded in the native wheel."""

    native_module = _native_module(module)
    candidates: list[Path] = []
    for loaded_module in (module, native_module):
        raw_path = getattr(loaded_module, "__file__", None)
        if not raw_path:
            continue
        parent = Path(raw_path).resolve().parent
        candidates.extend((parent / "_source_lock.json", parent.parent / "_source_lock.json"))
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            lock = json.loads(data)
            if not isinstance(lock, dict):
                raise RuntimeVersionError("embedded OpenVDB source lock must be an object")
            if lock.get("schema") != _SOURCE_LOCK_SCHEMA:
                raise RuntimeVersionError("embedded OpenVDB source lock has an unsupported schema")
            commit = lock["source"]["commit"]
            distribution_version = lock["distribution"]["version"]
            if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise RuntimeVersionError("embedded OpenVDB source lock has an invalid commit")
            if not isinstance(distribution_version, str) or not distribution_version:
                raise RuntimeVersionError(
                    "embedded OpenVDB source lock has an invalid distribution version"
                )
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeVersionError("embedded OpenVDB source lock is malformed") from exc
        return str(path), hashlib.sha256(data).hexdigest(), commit, distribution_version
    return None, None, None, None


def _distribution_version() -> str | None:
    return _ADMITTED_DISTRIBUTION_VERSION


def inspect_runtime(module: ModuleType | None = None) -> RuntimeInfo:
    """Return identity and capability evidence for the admitted runtime.

    Supplying a module is only valid when it is the module previously admitted
    by :func:`load_openvdb`. Tests that inspect intentionally malformed module
    objects use the private ``_inspect_unadmitted_module_for_tests`` helper.
    """

    with _ADMISSION_LOCK:
        if module is None:
            module = load_openvdb()
        elif module is not _ADMITTED_MODULE:
            raise RuntimeVersionError("explicit OpenVDB module was not admitted by this loader")
        if _cached_admitted_module() is not module:
            raise RuntimeVersionError("cached OpenVDB admission state is malformed")
        _deep_revalidate_admission()
        return _inspect_runtime_module(module)


def _library_version(module: ModuleType) -> tuple[int, int, int]:
    raw_version = getattr(module, "LIBRARY_VERSION", ())
    if (
        not isinstance(raw_version, tuple | list)
        or len(raw_version) != 3
        or any(
            isinstance(part, bool) or not isinstance(part, numbers.Integral) for part in raw_version
        )
    ):
        raise RuntimeVersionError(
            "openvdb.LIBRARY_VERSION is missing or malformed; expected three integers"
        )
    return tuple(int(part) for part in raw_version)  # type: ignore[return-value]


def _inspect_runtime_module(module: ModuleType) -> RuntimeInfo:
    """Inspect a module after admission; kept separate for isolated unit fixtures."""

    policy = _load_policy()
    library_version = _library_version(module)
    module_path, module_sha256 = _module_digest(module)
    source_lock_path, source_lock_sha256, locked_commit, locked_distribution = (
        _source_lock_identity(module)
    )
    file_format = getattr(module, "FILE_FORMAT_VERSION", None)
    if file_format is not None and (
        isinstance(file_format, bool) or not isinstance(file_format, numbers.Integral)
    ):
        raise RuntimeVersionError("openvdb.FILE_FORMAT_VERSION is malformed; expected an integer")
    return RuntimeInfo(
        library_version=library_version,
        distribution_version=_distribution_version(),
        file_format_version=int(file_format) if file_format is not None else None,
        module_path=module_path,
        module_sha256=module_sha256,
        source_lock_path=source_lock_path,
        source_lock_sha256=source_lock_sha256,
        source_distribution_version=locked_distribution,
        policy_schema=str(policy["schema"]),
        source_commit=locked_commit,
        capabilities=detect_capabilities(module),
    )


def require_runtime(
    capabilities: Iterable[Capability] = (),
    *,
    require_distribution_identity: bool = False,
) -> ModuleType:
    """Validate runtime policy and return the native module.

    Distribution identity is always required. ``require_distribution_identity``
    remains as a compatibility keyword for callers that previously opted in to
    the check; passing ``False`` no longer weakens admission.
    """

    del require_distribution_identity
    module = load_openvdb()
    policy = _load_policy()
    library_version = _library_version(module)
    available_capabilities = detect_capabilities(module)
    expected_library = tuple(int(part) for part in policy["openvdb_library_version"])
    if library_version != expected_library:
        expected = ".".join(str(part) for part in expected_library)
        actual = ".".join(str(part) for part in library_version)
        raise RuntimeVersionError(f"OpenVDB {expected} is required, but {actual} is loaded")
    required_policy_capabilities = frozenset(
        Capability(value)
        for value in (
            *policy["required_core_capabilities"],
            *policy["required_repository_capabilities"],
        )
    )
    missing_policy_capabilities = required_policy_capabilities.difference(available_capabilities)
    if missing_policy_capabilities:
        raise CapabilityUnavailableError(
            capability.value for capability in missing_policy_capabilities
        )
    requested = frozenset(capabilities)
    missing = requested.difference(available_capabilities)
    if missing:
        raise CapabilityUnavailableError(capability.value for capability in missing)
    return module


def is_available(*capabilities: Capability) -> bool:
    """Return whether a policy-compatible runtime provides the requested API."""

    try:
        require_runtime(capabilities)
    except OpenVDBRuntimeError:
        return False
    return True


def _inspect_unadmitted_module_for_tests(module: ModuleType) -> RuntimeInfo:
    """Test-only module-shape inspection with no native operation entry point."""

    return _inspect_runtime_module(module)
