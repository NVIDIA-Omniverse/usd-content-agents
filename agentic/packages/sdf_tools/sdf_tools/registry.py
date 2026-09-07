# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, policy-bound SDF backend registration and selection."""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import re
import stat
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

from .backend import BackendDescriptor, SdfBackend, SdfBackendExtension
from .errors import (
    BackendNotRegisteredError,
    BackendUnavailableError,
    CapabilityUnavailableError,
)
from .licensing import validate_license_manifest
from .types import BackendInfo, Operation

_POLICY_PATH = Path(__file__).with_name("backend_policy.json")
_ADMISSION_FIELDS = {
    "authenticated_data_files",
    "authenticated_files",
    "distribution",
    "distribution_version",
    "entry_point",
    "entry_point_module_path",
    "entry_point_module_sha256",
    "implementation_version",
    "license_manifest_sha256",
    "native_closure_attestation",
    "production_qualified",
}
_DISTRIBUTION_NORMALIZER = re.compile(r"[-_.]+")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FILE_CHUNK_SIZE = 1024 * 1024
_IGNORED_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})
_PROHIBITED_IMPORT_ARTIFACT_SUFFIXES = frozenset({".pth"})
_NATIVE_IMPORT_ARTIFACT_SUFFIXES = frozenset({".pyd", ".so"})
_UNRECORDED_EXECUTABLE_SUFFIXES = frozenset(
    {".bat", ".cmd", ".com", ".dll", ".dylib", ".exe", ".ps1", ".sh"}
)
_ENTRY_POINT_LOAD_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class BackendRejection:
    """One deterministic reason a backend was not selected."""

    backend_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class _AuthenticatedSourceBinding:
    module_name: str
    path: Path
    sha256: str
    is_package: bool


@dataclass(frozen=True, slots=True)
class _AuthenticatedImportPlan:
    backend_id: str
    target_module: str
    target_attributes: tuple[str, ...]
    top_level_name: str
    modules: Mapping[str, _AuthenticatedSourceBinding]


class _AuthenticatedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, binding: _AuthenticatedSourceBinding, backend_id: str) -> None:
        super().__init__(binding.module_name, str(binding.path))
        self._binding = binding
        self._backend_id = backend_id

    def get_code(self, fullname: str) -> Any:
        if fullname != self._binding.module_name:
            raise ImportError("authenticated SDF source loader received another module")
        source = _read_authenticated_source_file(
            self._binding.path,
            self._binding.sha256,
            self._backend_id,
        )
        return compile(
            source,
            str(self._binding.path),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )

    def set_data(self, _path: str, _data: bytes, *, _mode: int = 0o666) -> None:
        # Authenticated modules never materialize interpreter bytecode caches.
        return None


class _AuthenticatedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._plans: dict[str, _AuthenticatedImportPlan] = {}
        self._lock = RLock()

    def bind(self, plan: _AuthenticatedImportPlan, backend_id: str) -> bool:
        with self._lock:
            existing = self._plans.get(plan.top_level_name)
            if existing is not None:
                if existing != plan:
                    raise BackendUnavailableError(
                        f"SDF backend {backend_id!r} conflicts with an authenticated import binding"
                    )
                return False
            self._plans[plan.top_level_name] = plan
            return True

    def unbind(self, plan: _AuthenticatedImportPlan) -> None:
        with self._lock:
            if self._plans.get(plan.top_level_name) == plan:
                del self._plans[plan.top_level_name]

    def has_bindings(self) -> bool:
        with self._lock:
            return bool(self._plans)

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> Any:
        del path, target
        top_level_name = fullname.partition(".")[0]
        with self._lock:
            plan = self._plans.get(top_level_name)
            if plan is None:
                return None
            binding = plan.modules.get(fullname)
        if binding is None:
            raise ModuleNotFoundError(
                f"module {fullname!r} is outside the authenticated SDF backend package"
            )
        loader = _AuthenticatedSourceLoader(binding, plan.backend_id)
        locations = [str(binding.path.parent)] if binding.is_package else None
        return importlib.util.spec_from_file_location(
            fullname,
            binding.path,
            loader=loader,
            submodule_search_locations=locations,
        )


_AUTHENTICATED_SOURCE_FINDER = _AuthenticatedSourceFinder()


def _load_policy() -> dict[str, Any]:
    try:
        policy = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendUnavailableError(f"unable to read SDF backend policy: {exc}") from exc
    if policy.get("schema") != "world-understanding.sdf-backend-policy.v1":
        raise BackendUnavailableError("unsupported SDF backend policy schema")
    if not isinstance(policy.get("backends"), dict):
        raise BackendUnavailableError("SDF backend policy is malformed")
    return policy


def _normalized_distribution_name(value: str) -> str:
    return _DISTRIBUTION_NORMALIZER.sub("-", value).lower()


def _entry_point_module_path(
    backend_id: str,
    admission: Mapping[str, Any],
) -> PurePosixPath:
    target = admission["entry_point"]
    raw_path = admission["entry_point_module_path"]
    if not isinstance(target, str) or not isinstance(raw_path, str):
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} has malformed module authentication policy"
        )
    module_name, separator, attribute = target.partition(":")
    module_parts = module_name.split(".")
    if (
        not separator
        or not attribute
        or not module_parts
        or any(not part.isidentifier() for part in module_parts)
    ):
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} has malformed entry point policy"
        )
    module_path = PurePosixPath(raw_path)
    if (
        not raw_path
        or "\\" in raw_path
        or module_path.is_absolute()
        or str(module_path) != raw_path
        or any(part in {"", ".", ".."} for part in module_path.parts)
    ):
        raise BackendUnavailableError(f"SDF backend {backend_id!r} has unsafe module path policy")

    source_path = PurePosixPath(*module_parts[:-1], f"{module_parts[-1]}.py")
    package_path = PurePosixPath(*module_parts, "__init__.py")
    extension_parent = PurePosixPath(*module_parts[:-1])
    extension_name = module_path.name
    extension_module = (
        module_path.parent == extension_parent
        and extension_name.startswith(f"{module_parts[-1]}.")
        and module_path.suffix in {".pyd", ".so"}
    )
    if module_path not in {source_path, package_path} and not extension_module:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} module path does not match its entry point"
        )
    return module_path


def _editable_distribution_root(distribution: object) -> Path | None:
    reader = getattr(distribution, "read_text", None)
    if not callable(reader):
        return None
    try:
        raw = reader("direct_url.json")
        direct_url = json.loads(raw) if isinstance(raw, str) else None
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(direct_url, dict):
        return None
    directory_info = direct_url.get("dir_info")
    if not isinstance(directory_info, dict) or directory_info.get("editable") is not True:
        return None
    url = direct_url.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlsplit(url)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    return Path(unquote(parsed.path))


def _locate_distribution_module(
    distribution: object,
    module_path: PurePosixPath,
    backend_id: str,
) -> Path:
    files = getattr(distribution, "files", None)
    matching_files = [item for item in files or () if PurePosixPath(str(item)) == module_path]
    if len(matching_files) > 1:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} distribution records duplicate module files"
        )
    try:
        if matching_files:
            locator = distribution.locate_file  # type: ignore[attr-defined]
            root = Path(locator(""))
            candidate = Path(locator(matching_files[0]))
        else:
            root = _editable_distribution_root(distribution)  # type: ignore[assignment]
            if root is None:
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} distribution does not record its entry "
                    "point module"
                )
            candidate = root.joinpath(*module_path.parts)
        if candidate.is_symlink():
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} entry point module may not be a symbolic link"
            )
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        expected_candidate = resolved_root.joinpath(*module_path.parts).resolve(strict=True)
    except BackendUnavailableError:
        raise
    except (AttributeError, OSError, RuntimeError) as exc:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} entry point module cannot be authenticated"
        ) from exc
    if not resolved_root.is_dir() or resolved_candidate != expected_candidate:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} entry point module is outside its distribution"
        )
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} entry point module is outside its distribution"
        ) from exc
    return resolved_candidate


def _sha256_regular_file(path: Path, backend_id: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} authenticated path is not a regular file"
                )
            digest = hashlib.sha256()
            while chunk := stream.read(_FILE_CHUNK_SIZE):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat(follow_symlinks=False)
    except BackendUnavailableError:
        raise
    except OSError as exc:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} authenticated file cannot be read"
        ) from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != current_identity:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} file changed during authentication"
        )
    return digest.hexdigest()


def _read_authenticated_source_file(
    path: Path,
    expected_sha256: str,
    backend_id: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} authenticated source is not a regular file"
                )
            source = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat(follow_symlinks=False)
    except BackendUnavailableError:
        raise
    except OSError as exc:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} authenticated source cannot be read"
        ) from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != current_identity:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} source changed during authenticated loading"
        )
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} authenticated source is not policy-approved"
        )
    return source


def _distribution_root_for_file(path: Path, relative_path: PurePosixPath) -> Path:
    root = path
    for _part in relative_path.parts:
        root = root.parent
    return root


def _authenticated_file_policy(
    admission: Mapping[str, Any],
    module_path: PurePosixPath,
    backend_id: str,
) -> dict[PurePosixPath, str]:
    raw_files = admission["authenticated_files"]
    raw_data_files = admission["authenticated_data_files"]
    if not isinstance(raw_files, dict) or not raw_files or not isinstance(raw_data_files, dict):
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} has malformed authenticated file policy"
        )
    authenticated_files: dict[PurePosixPath, str] = {}
    module_name = str(admission["entry_point"]).partition(":")[0]
    top_level_name = module_name.partition(".")[0]
    package_prefix = PurePosixPath(top_level_name)
    package_style = module_path != PurePosixPath(f"{top_level_name}.py")
    for file_map, python_code in ((raw_files, True), (raw_data_files, False)):
        for raw_path, digest in file_map.items():
            if not isinstance(raw_path, str) or not isinstance(digest, str):
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} has malformed authenticated file policy"
                )
            path = PurePosixPath(raw_path)
            top_level_data_file = len(path.parts) == 1 and path.name.startswith(
                f"{top_level_name}."
            )
            if (
                not raw_path
                or "\\" in raw_path
                or path.is_absolute()
                or str(path) != raw_path
                or any(part in {"", ".", ".."} for part in path.parts)
                or (python_code and path.suffix != ".py")
                or (not python_code and path.suffix == ".py")
                or _SHA256.fullmatch(digest) is None
                or (package_style and path.parts[0] != package_prefix.name)
                or (not package_style and python_code and path != module_path)
                or (not package_style and not python_code and not top_level_data_file)
                or path in authenticated_files
            ):
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} has unsafe authenticated file policy"
                )
            authenticated_files[path] = digest
    if module_path not in authenticated_files:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} entry point module is missing from authenticated files"
        )
    if authenticated_files[module_path] != admission["entry_point_module_sha256"]:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} entry point module digests disagree in policy"
        )
    return authenticated_files


def _installed_python_files(
    root: Path,
    module_path: PurePosixPath,
    admission: Mapping[str, Any],
    backend_id: str,
) -> dict[PurePosixPath, Path]:
    module_name = str(admission["entry_point"]).partition(":")[0]
    top_level_name = module_name.partition(".")[0]
    top_level_module = root / f"{top_level_name}.py"
    package_root = root / top_level_name
    package_style = module_path != PurePosixPath(f"{top_level_name}.py")
    if package_style and top_level_module.exists():
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} has an ambiguous top-level module"
        )
    if not package_style:
        if package_root.exists():
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} has an ambiguous top-level package"
            )
        return {module_path: top_level_module} if top_level_module.is_file() else {}
    if not package_root.is_dir() or package_root.is_symlink():
        return {}

    installed: dict[PurePosixPath, Path] = {}
    for directory, directory_names, filenames in os.walk(package_root, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(directory_names):
            child = directory_path / name
            if child.is_symlink():
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} package contains a symbolic-link directory"
                )
        for filename in filenames:
            if not filename.casefold().endswith(".py"):
                continue
            path = directory_path / filename
            relative = PurePosixPath(path.relative_to(root).as_posix())
            installed[relative] = path
    return installed


def _is_native_import_artifact(filename: str) -> bool:
    folded_name = filename.casefold()
    return PurePosixPath(folded_name).suffix in _NATIVE_IMPORT_ARTIFACT_SUFFIXES or any(
        folded_name.endswith(extension_suffix.casefold())
        for extension_suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def _reject_prohibited_import_artifact(
    relative_path: PurePosixPath,
    backend_id: str,
) -> None:
    raise BackendUnavailableError(
        f"SDF backend {backend_id!r} package contains prohibited executable import "
        f"artifact {str(relative_path)!r}"
    )


def _validate_executable_package_member(
    path: Path,
    relative_path: PurePosixPath,
    authenticated_files: Mapping[PurePosixPath, str],
    backend_id: str,
) -> None:
    suffix = path.suffix.casefold()
    if suffix in _IGNORED_BYTECODE_SUFFIXES:
        return
    if suffix in _PROHIBITED_IMPORT_ARTIFACT_SUFFIXES or _is_native_import_artifact(path.name):
        _reject_prohibited_import_artifact(relative_path, backend_id)
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} package member cannot be authenticated"
        ) from exc
    if (
        relative_path not in authenticated_files
        and stat.S_ISREG(metadata.st_mode)
        and (
            metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            or suffix in _UNRECORDED_EXECUTABLE_SUFFIXES
        )
    ):
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} package contains an unrecorded executable member"
        )


def _authenticate_executable_package_closure(
    root: Path,
    module_path: PurePosixPath,
    admission: Mapping[str, Any],
    authenticated_files: Mapping[PurePosixPath, str],
    backend_id: str,
) -> None:
    module_name = str(admission["entry_point"]).partition(":")[0]
    top_level_name = module_name.partition(".")[0]
    package_root = root / top_level_name
    package_style = module_path != PurePosixPath(f"{top_level_name}.py")

    def walk_error(error: OSError) -> None:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} package closure cannot be authenticated"
        ) from error

    if package_style and package_root.is_dir() and not package_root.is_symlink():
        for directory, directory_names, filenames in os.walk(
            package_root,
            followlinks=False,
            onerror=walk_error,
        ):
            directory_path = Path(directory)
            directory_names[:] = [
                name for name in directory_names if name.casefold() != "__pycache__"
            ]
            for filename in filenames:
                path = directory_path / filename
                relative = PurePosixPath(path.relative_to(root).as_posix())
                _validate_executable_package_member(
                    path,
                    relative,
                    authenticated_files,
                    backend_id,
                )

    try:
        top_level_members = tuple(root.iterdir())
    except OSError as exc:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} package closure cannot be authenticated"
        ) from exc
    for path in top_level_members:
        if path.name.startswith(f"{top_level_name}."):
            _validate_executable_package_member(
                path,
                PurePosixPath(path.name),
                authenticated_files,
                backend_id,
            )


def _authenticate_python_file_closure(
    distribution: object,
    installed_module_path: Path,
    module_path: PurePosixPath,
    admission: Mapping[str, Any],
    backend_id: str,
) -> dict[PurePosixPath, str]:
    authenticated_files = _authenticated_file_policy(admission, module_path, backend_id)
    root = _distribution_root_for_file(installed_module_path, module_path)
    _authenticate_executable_package_closure(
        root,
        module_path,
        admission,
        authenticated_files,
        backend_id,
    )
    installed_python_files = _installed_python_files(
        root,
        module_path,
        admission,
        backend_id,
    )
    authenticated_python_files = {path for path in authenticated_files if path.suffix == ".py"}
    if set(installed_python_files) != authenticated_python_files:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} executable package files differ from policy"
        )
    installed_files = dict(installed_python_files)
    for relative_path in authenticated_files.keys() - authenticated_python_files:
        candidate = root.joinpath(*relative_path.parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} authenticated data file is unavailable"
            )
        installed_files[relative_path] = candidate

    editable_root = _editable_distribution_root(distribution)
    if editable_root is None:
        recorded_files = {
            PurePosixPath(str(item))
            for item in getattr(distribution, "files", None) or ()
            if PurePosixPath(str(item)) in authenticated_files
            or (
                str(item).endswith(".py")
                and (
                    PurePosixPath(str(item)) == PurePosixPath(f"{module_path.parts[0]}.py")
                    or PurePosixPath(str(item)).parts[0] == module_path.parts[0]
                )
            )
        }
        if recorded_files != set(authenticated_files):
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} distribution file records differ from policy"
            )

    for relative_path, expected_sha256 in authenticated_files.items():
        installed_sha256 = _sha256_regular_file(installed_files[relative_path], backend_id)
        if installed_sha256 != expected_sha256:
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} authenticated package file is not policy-approved"
            )
    return authenticated_files


def _build_authenticated_import_plan(
    root: Path,
    module_path: PurePosixPath,
    admission: Mapping[str, Any],
    authenticated_files: Mapping[PurePosixPath, str],
    backend_id: str,
) -> _AuthenticatedImportPlan:
    target_module, _separator, target_attribute = str(admission["entry_point"]).partition(":")
    target_attributes = tuple(target_attribute.split("."))
    if not target_attributes or any(not part.isidentifier() for part in target_attributes):
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} has malformed entry point attributes"
        )

    modules: dict[str, _AuthenticatedSourceBinding] = {}
    for relative_path, expected_sha256 in authenticated_files.items():
        if relative_path.suffix != ".py":
            continue
        module_parts = list(relative_path.parts)
        is_package = module_parts[-1] == "__init__.py"
        if is_package:
            module_parts.pop()
        else:
            module_parts[-1] = PurePosixPath(module_parts[-1]).stem
        if not module_parts or any(not part.isidentifier() for part in module_parts):
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} has an invalid authenticated module path"
            )
        module_name = ".".join(module_parts)
        binding = _AuthenticatedSourceBinding(
            module_name=module_name,
            path=root.joinpath(*relative_path.parts),
            sha256=expected_sha256,
            is_package=is_package,
        )
        if module_name in modules:
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} has ambiguous authenticated modules"
            )
        modules[module_name] = binding

    target_binding = modules.get(target_module)
    expected_target_path = root.joinpath(*module_path.parts)
    if target_binding is None or target_binding.path != expected_target_path:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} entry point is not an authenticated source module"
        )
    target_parts = target_module.split(".")
    for index in range(1, len(target_parts)):
        package_name = ".".join(target_parts[:index])
        package_binding = modules.get(package_name)
        if package_binding is None or not package_binding.is_package:
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} entry point package is not authenticated"
            )
    return _AuthenticatedImportPlan(
        backend_id=backend_id,
        target_module=target_module,
        target_attributes=target_attributes,
        top_level_name=target_parts[0],
        modules=MappingProxyType(modules),
    )


def _resolved_import_path(value: object, backend_id: str) -> Path:
    if not isinstance(value, str | os.PathLike):
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} has a malformed preloaded module binding"
        )
    try:
        return Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} preloaded module binding cannot be authenticated"
        ) from exc


def _validate_preloaded_import_bindings(plan: _AuthenticatedImportPlan) -> None:
    namespace_prefix = f"{plan.top_level_name}."
    for module_name, module in tuple(sys.modules.items()):
        if module_name != plan.top_level_name and not module_name.startswith(namespace_prefix):
            continue
        binding = plan.modules.get(module_name)
        if binding is None or module is None:
            raise BackendUnavailableError(
                f"SDF backend {plan.backend_id!r} has an unapproved preloaded package module"
            )
        spec = getattr(module, "__spec__", None)
        loader = getattr(spec, "loader", None)
        if not isinstance(loader, importlib.machinery.SourceFileLoader):
            raise BackendUnavailableError(
                f"SDF backend {plan.backend_id!r} preloaded module is not source-bound"
            )
        origins = (
            getattr(module, "__file__", None),
            getattr(spec, "origin", None),
            getattr(loader, "path", None),
        )
        if any(
            _resolved_import_path(origin, plan.backend_id) != binding.path for origin in origins
        ):
            raise BackendUnavailableError(
                f"SDF backend {plan.backend_id!r} preloaded module came from another root"
            )
        search_locations = getattr(spec, "submodule_search_locations", None)
        module_path = getattr(module, "__path__", None)
        if binding.is_package:
            expected_location = (binding.path.parent,)
            if search_locations is None or module_path is None:
                raise BackendUnavailableError(
                    f"SDF backend {plan.backend_id!r} preloaded package binding is malformed"
                )
            resolved_search = tuple(
                _resolved_import_path(location, plan.backend_id) for location in search_locations
            )
            resolved_module_path = tuple(
                _resolved_import_path(location, plan.backend_id) for location in module_path
            )
            if resolved_search != expected_location or resolved_module_path != expected_location:
                raise BackendUnavailableError(
                    f"SDF backend {plan.backend_id!r} preloaded package came from another root"
                )
        elif search_locations is not None or module_path is not None:
            raise BackendUnavailableError(
                f"SDF backend {plan.backend_id!r} preloaded module binding is malformed"
            )
        if type(loader) is not _AuthenticatedSourceLoader or loader._binding != binding:  # type: ignore[attr-defined]
            raise BackendUnavailableError(
                f"SDF backend {plan.backend_id!r} was imported before authenticated discovery"
            )


def _authenticate_entry_point_distribution(
    entry_point: importlib.metadata.EntryPoint,
    admission: Mapping[str, Any],
    backend_id: str,
) -> _AuthenticatedImportPlan:
    expected_name = admission["distribution"]
    expected_version = admission["distribution_version"]
    expected_sha256 = admission["entry_point_module_sha256"]
    if (
        not isinstance(expected_name, str)
        or not expected_name
        or not isinstance(expected_version, str)
        or not expected_version
        or not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} has malformed distribution authentication policy"
        )

    distribution = getattr(entry_point, "dist", None)
    distribution_name = getattr(distribution, "name", None)
    if not isinstance(distribution_name, str) or _normalized_distribution_name(
        distribution_name
    ) != _normalized_distribution_name(expected_name):
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} came from an unapproved distribution"
        )
    distribution_version = getattr(distribution, "version", None)
    if distribution_version != expected_version:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} distribution version is not policy-approved"
        )

    module_path = _entry_point_module_path(backend_id, admission)
    installed_path = _locate_distribution_module(distribution, module_path, backend_id)
    installed_sha256 = _sha256_regular_file(installed_path, backend_id)
    if installed_sha256 != expected_sha256:
        raise BackendUnavailableError(
            f"SDF backend {backend_id!r} entry point module is not policy-approved"
        )
    authenticated_files = _authenticate_python_file_closure(
        distribution,
        installed_path,
        module_path,
        admission,
        backend_id,
    )
    root = _distribution_root_for_file(installed_path, module_path)
    return _build_authenticated_import_plan(
        root,
        module_path,
        admission,
        authenticated_files,
        backend_id,
    )


def _load_authenticated_extension(
    entry_point: importlib.metadata.EntryPoint,
    plan: _AuthenticatedImportPlan,
) -> object:
    with _ENTRY_POINT_LOAD_LOCK:
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            if not isinstance(entry_point, importlib.metadata.EntryPoint):
                loaded = entry_point.load()
                return loaded() if callable(loaded) else loaded

            _validate_preloaded_import_bindings(plan)
            preloaded_names = frozenset(sys.modules)
            added_binding = _AUTHENTICATED_SOURCE_FINDER.bind(plan, plan.backend_id)
            while _AUTHENTICATED_SOURCE_FINDER in sys.meta_path:
                sys.meta_path.remove(_AUTHENTICATED_SOURCE_FINDER)
            sys.meta_path.insert(0, _AUTHENTICATED_SOURCE_FINDER)
            try:
                loaded = importlib.import_module(plan.target_module)
                for attribute in plan.target_attributes:
                    loaded = getattr(loaded, attribute)
                extension = loaded() if callable(loaded) else loaded
                if not isinstance(extension, SdfBackendExtension):
                    raise BackendUnavailableError(
                        f"SDF backend {plan.backend_id!r} entry point returned an invalid extension"
                    )
            except Exception:
                if added_binding:
                    prefix = f"{plan.top_level_name}."
                    for module_name in tuple(sys.modules):
                        if module_name in preloaded_names:
                            continue
                        if module_name == plan.top_level_name or module_name.startswith(prefix):
                            sys.modules.pop(module_name, None)
                    _AUTHENTICATED_SOURCE_FINDER.unbind(plan)
                    if not _AUTHENTICATED_SOURCE_FINDER.has_bindings():
                        while _AUTHENTICATED_SOURCE_FINDER in sys.meta_path:
                            sys.meta_path.remove(_AUTHENTICATED_SOURCE_FINDER)
                raise
            return extension
        finally:
            sys.dont_write_bytecode = previous


def _validate_runtime_info(
    info: object,
    descriptor: BackendDescriptor,
) -> BackendInfo:
    if not isinstance(info, BackendInfo):
        raise BackendUnavailableError("SDF backend returned an invalid runtime descriptor")
    if (
        info.backend_id != descriptor.backend_id
        or info.implementation_version != descriptor.implementation_version
        or info.execution_mode != descriptor.execution_mode
        or not info.operations.issubset(descriptor.operations)
        or not info.read_formats.issubset(descriptor.read_formats)
        or not info.write_formats.issubset(descriptor.write_formats)
    ):
        raise BackendUnavailableError(
            f"SDF backend {descriptor.backend_id!r} runtime identity differs from policy"
        )
    try:
        json.dumps(info.as_dict(), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise BackendUnavailableError(
            f"SDF backend {descriptor.backend_id!r} returned non-canonical provenance"
        ) from exc
    return info


def _format_identifiers(value: Iterable[str], *, label: str) -> frozenset[str]:
    if isinstance(value, str):
        raise TypeError(f"{label} must be an iterable of format identifiers")
    values = tuple(value)
    for format_name in values:
        if not isinstance(format_name, str):
            raise TypeError(f"{label} must contain strings")
        if not format_name:
            raise ValueError(f"{label} must contain nonempty identifiers")
    return frozenset(values)


def _directional_format_requirements(
    required: frozenset[Operation],
    require_formats: Iterable[str] | Mapping[Operation | str, Iterable[str]],
) -> tuple[frozenset[str], frozenset[str]]:
    if isinstance(require_formats, Mapping):
        read_formats: frozenset[str] = frozenset()
        write_formats: frozenset[str] = frozenset()
        for raw_operation, values in require_formats.items():
            try:
                operation = Operation(raw_operation)
            except (TypeError, ValueError) as exc:
                raise ValueError("format requirements name an unknown operation") from exc
            if operation not in {Operation.READ_FIELDS, Operation.WRITE_FIELDS}:
                raise ValueError("format requirements may name only read_fields or write_fields")
            if operation not in required:
                raise ValueError(
                    f"format requirements for {operation.value} require that operation"
                )
            formats = _format_identifiers(
                values,
                label=f"{operation.value} format requirements",
            )
            if operation is Operation.READ_FIELDS:
                read_formats = formats
            else:
                write_formats = formats
        return read_formats, write_formats

    formats = _format_identifiers(require_formats, label="require_formats")
    if formats and not required.intersection({Operation.READ_FIELDS, Operation.WRITE_FIELDS}):
        raise ValueError("require_formats requires read_fields or write_fields")
    return (
        formats if Operation.READ_FIELDS in required else frozenset(),
        formats if Operation.WRITE_FIELDS in required else frozenset(),
    )


class SdfBackendRegistry:
    """Thread-safe, policy-bound registry with immutable public snapshots."""

    def __init__(self) -> None:
        self._extensions: dict[str, SdfBackendExtension] = {}
        self._lock = RLock()

    @staticmethod
    def _validate_extension(extension: SdfBackendExtension) -> None:
        if not isinstance(extension, SdfBackendExtension):
            raise TypeError("extension must be an sdf_tools.SdfBackendExtension")
        validate_license_manifest(extension.descriptor.license_manifest)

    def snapshot(self) -> Mapping[str, SdfBackendExtension]:
        with self._lock:
            return MappingProxyType(dict(self._extensions))

    def _instance(self, backend_id: str) -> SdfBackend:
        with self._lock:
            extension = self._extensions.get(backend_id)
            if extension is None:
                raise BackendNotRegisteredError(
                    f"SDF backend {backend_id!r} is not admitted by the registry"
                )
        # A driver owns every opaque field it creates. A fresh driver per
        # session prevents fields from one session being accepted by another.
        return extension.create()

    def resolve(
        self,
        *,
        backend: str,
        require: Iterable[Operation] = (),
        require_formats: Iterable[str] | Mapping[Operation | str, Iterable[str]] = (),
    ) -> tuple[SdfBackend, BackendInfo, tuple[BackendRejection, ...]]:
        required = frozenset(Operation(operation) for operation in require)
        required_read_formats, required_write_formats = _directional_format_requirements(
            required,
            require_formats,
        )
        if backend != "auto":
            try:
                instance = self._instance(backend)
            except Exception as exc:
                raise BackendUnavailableError(
                    f"SDF backend {backend!r} could not be created: {exc}"
                ) from exc
            declared = instance.descriptor.operations
            missing = required - declared
            missing_read_formats = required_read_formats - instance.descriptor.read_formats
            missing_write_formats = required_write_formats - instance.descriptor.write_formats
            if missing or missing_read_formats or missing_write_formats:
                raise CapabilityUnavailableError(
                    backend,
                    (operation.value for operation in missing),
                    read_formats=missing_read_formats,
                    write_formats=missing_write_formats,
                )
            try:
                info = _validate_runtime_info(instance.inspect(), instance.descriptor)
            except Exception as exc:
                raise BackendUnavailableError(
                    f"SDF backend {backend!r} is unavailable: {exc}"
                ) from exc
            runtime_missing = required - info.operations
            runtime_missing_read_formats = required_read_formats - info.read_formats
            runtime_missing_write_formats = required_write_formats - info.write_formats
            if runtime_missing or runtime_missing_read_formats or runtime_missing_write_formats:
                raise CapabilityUnavailableError(
                    backend,
                    (operation.value for operation in runtime_missing),
                    read_formats=runtime_missing_read_formats,
                    write_formats=runtime_missing_write_formats,
                )
            return instance, info, ()

        with self._lock:
            candidates = sorted(
                self._extensions.values(),
                key=lambda extension: (
                    -extension.descriptor.priority,
                    extension.descriptor.backend_id,
                    extension.descriptor.implementation_version,
                ),
            )
        rejections: list[BackendRejection] = []
        for extension in candidates:
            descriptor = extension.descriptor
            missing = required - descriptor.operations
            missing_read_formats = required_read_formats - descriptor.read_formats
            missing_write_formats = required_write_formats - descriptor.write_formats
            static_reasons = []
            if missing:
                static_reasons.append(
                    "missing operations: "
                    + ", ".join(sorted(operation.value for operation in missing))
                )
            if missing_read_formats:
                static_reasons.append(
                    "missing read formats: " + ", ".join(sorted(missing_read_formats))
                )
            if missing_write_formats:
                static_reasons.append(
                    "missing write formats: " + ", ".join(sorted(missing_write_formats))
                )
            if static_reasons:
                rejections.append(
                    BackendRejection(
                        descriptor.backend_id,
                        "; ".join(static_reasons),
                    )
                )
                continue
            try:
                instance = self._instance(descriptor.backend_id)
                info = _validate_runtime_info(instance.inspect(), descriptor)
            except Exception as exc:
                rejections.append(BackendRejection(descriptor.backend_id, str(exc)))
                continue
            runtime_missing = required - info.operations
            runtime_missing_read_formats = required_read_formats - info.read_formats
            runtime_missing_write_formats = required_write_formats - info.write_formats
            runtime_reasons = []
            if runtime_missing:
                runtime_reasons.append(
                    "runtime missing operations: "
                    + ", ".join(sorted(operation.value for operation in runtime_missing))
                )
            if runtime_missing_read_formats:
                runtime_reasons.append(
                    "runtime missing read formats: "
                    + ", ".join(sorted(runtime_missing_read_formats))
                )
            if runtime_missing_write_formats:
                runtime_reasons.append(
                    "runtime missing write formats: "
                    + ", ".join(sorted(runtime_missing_write_formats))
                )
            if runtime_reasons:
                rejections.append(
                    BackendRejection(
                        descriptor.backend_id,
                        "; ".join(runtime_reasons),
                    )
                )
                continue
            return instance, info, tuple(rejections)
        detail = "; ".join(
            f"{rejection.backend_id}: {rejection.reason}" for rejection in rejections
        )
        raise BackendUnavailableError(
            "no admitted SDF backend satisfies the request" + (f" ({detail})" if detail else "")
        )


def discover_installed_backends(
    registry: SdfBackendRegistry,
    *,
    backend: str | None = None,
    entry_points: Iterable[importlib.metadata.EntryPoint] | None = None,
) -> tuple[str, ...]:
    """Load selected installed entry points named exactly by checked-in policy.

    Automatic discovery considers only production-qualified policy entries.
    Explicit selection loads only the named policy entry, so an unrelated bad
    plugin cannot prevent a healthy backend from being used.
    """

    policy = _load_policy()
    group = str(policy["entry_point_group"])
    if entry_points is None:
        discovered = importlib.metadata.entry_points()
        entry_points = discovered.select(group=group)
    by_name: dict[str, list[importlib.metadata.EntryPoint]] = {}
    for entry_point in entry_points:
        if entry_point.group != group:
            continue
        by_name.setdefault(entry_point.name, []).append(entry_point)

    policy_backends = policy["backends"]
    if backend is None:
        candidates = [
            (backend_id, admission)
            for backend_id, admission in sorted(policy_backends.items())
            if isinstance(admission, dict) and admission.get("production_qualified") is True
        ]
    else:
        admission = policy_backends.get(backend)
        candidates = [] if admission is None else [(backend, admission)]

    extensions: list[SdfBackendExtension] = []
    registered = registry.snapshot()
    for backend_id, admission in candidates:
        if not isinstance(admission, dict) or set(admission) != _ADMISSION_FIELDS:
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} has malformed admission policy"
            )
        if not isinstance(admission["production_qualified"], bool):
            raise BackendUnavailableError(
                f"SDF backend {backend_id!r} has invalid qualification policy"
            )
        if admission["production_qualified"] is not True:
            raise BackendUnavailableError(f"SDF backend {backend_id!r} is not production-qualified")
        if backend_id in registered:
            continue
        matches = by_name.get(backend_id, [])
        if not matches:
            continue
        if len(matches) != 1:
            raise BackendUnavailableError(
                f"multiple installed entry points claim SDF backend {backend_id!r}"
            )
        entry_point = matches[0]
        expected_target = admission["entry_point"]
        if entry_point.value != expected_target:
            failure = BackendUnavailableError(
                f"SDF backend {backend_id!r} entry point target is not policy-approved"
            )
            if backend is not None:
                raise failure
            continue
        try:
            plan = _authenticate_entry_point_distribution(entry_point, admission, backend_id)
            extension = _load_authenticated_extension(entry_point, plan)
            if not isinstance(extension, SdfBackendExtension):
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} entry point returned an invalid extension"
                )
            registry._validate_extension(extension)
            if extension.descriptor.backend_id != backend_id:
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} entry point changed its identifier"
                )
            if extension.descriptor.implementation_version != admission["implementation_version"]:
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} implementation version is not policy-approved"
                )
            manifest = extension.descriptor.license_manifest
            if manifest.sha256 != admission["license_manifest_sha256"]:
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} license manifest is not policy-approved"
                )
            if manifest.native_closure_attestation != admission["native_closure_attestation"]:
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} native closure is not policy-approved"
                )
        except Exception as exc:
            if backend is not None:
                if isinstance(exc, BackendUnavailableError):
                    raise
                raise BackendUnavailableError(
                    f"SDF backend {backend_id!r} failed during authenticated discovery"
                ) from exc
            continue
        extensions.append(extension)

    identifiers = [extension.descriptor.backend_id for extension in extensions]
    if len(identifiers) != len(set(identifiers)):
        raise BackendUnavailableError(
            "authenticated SDF backend batch contains duplicate identifiers"
        )
    with registry._lock:
        conflicts = sorted(set(identifiers).intersection(registry._extensions))
        if conflicts:
            raise BackendUnavailableError(
                f"SDF backends already registered: {', '.join(conflicts)}"
            )
        registry._extensions.update(
            (extension.descriptor.backend_id, extension) for extension in extensions
        )
    return tuple(extension.descriptor.backend_id for extension in extensions)
