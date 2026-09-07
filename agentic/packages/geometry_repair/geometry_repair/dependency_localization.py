# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-approved, source-preserving OpenUSD dependency localization."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEPENDENCY_LOCALIZATION_SCHEMA_VERSION = "geometry-repair.dependency-localization.v1"
DEPENDENCY_REMAP_SCHEMA_VERSION = "geometry-repair.dependency-remap.v1"
_CHUNK_BYTES = 1024 * 1024
_DYNAMIC_ASSET_MARKERS = ("<UDIM>", "%(UDIM)d", "${")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PORTABLE_FILENAME_CHARACTER = re.compile(r"[^A-Za-z0-9._-]")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_USD_SUFFIXES = {".usd", ".usda", ".usdc"}

DependencyResolutionMethod = Literal["source_directory", "approved_root", "exact_remap"]
MaterialFidelityStatus = Literal[
    "not_evaluated_dependency_complete",
    "not_claimed_unresolved_dependencies",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DependencyRemap(_FrozenModel):
    """One exact authored-identifier to caller-provided local-file remap."""

    authored_path: str = Field(min_length=1)
    local_path: str = Field(min_length=1)
    source_layer: str | None = None
    sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _validate_digest(self) -> DependencyRemap:
        if self.sha256 is not None and not _HASH_PATTERN.fullmatch(self.sha256):
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
        return self


class DependencyRemapManifest(_FrozenModel):
    """Exact remaps; no basename, prefix, glob, or search matching is performed."""

    schema_version: Literal["geometry-repair.dependency-remap.v1"] = DEPENDENCY_REMAP_SCHEMA_VERSION
    remaps: list[DependencyRemap] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reject_duplicate_selectors(self) -> DependencyRemapManifest:
        selectors = [(item.source_layer, item.authored_path) for item in self.remaps]
        if len(selectors) != len(set(selectors)):
            raise ValueError("dependency remap selectors must be unique")
        return self


type DependencyRemapInput = DependencyRemapManifest | Mapping[str, Any] | str | Path | None


class DependencySourceDecision(_FrozenModel):
    """Safe local-source decision consumed by USD intake and localization."""

    resolved: bool
    local_path: str | None = None
    package_member: str | None = None
    remote_scheme: str | None = None
    resolution_method: DependencyResolutionMethod | None = None
    expected_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    reason_code: str | None = None
    reason: str | None = None


class LocalizedDependencyMapping(_FrozenModel):
    """One deterministic authored-path to portable-package mapping."""

    source_layer: str
    kind: str
    authored_path: str
    resolution_method: DependencyResolutionMethod
    source_path: str
    source_sha256: str = Field(min_length=64, max_length=64)
    source_size_bytes: int = Field(ge=0)
    package_path: str
    package_asset_path: str
    package_member: str | None = None
    localized_sha256: str = Field(min_length=64, max_length=64)
    localized_size_bytes: int = Field(ge=0)
    rewrite_applied: bool

    @model_validator(mode="after")
    def _validate_package_paths(self) -> LocalizedDependencyMapping:
        _validate_portable_package_path(self.package_path)
        _validate_portable_asset_path(self.package_asset_path)
        if self.package_member is not None:
            _validate_package_member(self.package_member)
        return self


class UnresolvedDependencyEvidence(_FrozenModel):
    """One dependency that was not safely approved, copied, or rewritten."""

    source_layer: str
    kind: str
    authored_path: str
    status: str
    reason_code: str
    reason: str


class DependencyLocalizationReport(_FrozenModel):
    """Portable-bundle mapping plus explicit unresolved and fidelity evidence."""

    schema_version: Literal["geometry-repair.dependency-localization.v1"] = (
        DEPENDENCY_LOCALIZATION_SCHEMA_VERSION
    )
    source_path: str
    source_sha256: str = Field(min_length=64, max_length=64)
    source_size_bytes: int = Field(ge=0)
    source_unchanged: bool
    package_root: str
    localized_source_path: str
    localized_source_package_path: str
    localized_source_sha256: str = Field(min_length=64, max_length=64)
    mappings: list[LocalizedDependencyMapping] = Field(default_factory=list)
    unresolved: list[UnresolvedDependencyEvidence] = Field(default_factory=list)
    dependency_complete: bool
    portable_package_complete: bool
    material_fidelity_claimed: Literal[False] = False
    material_fidelity_status: MaterialFidelityStatus
    evidence_path: str
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_claim_boundary(self) -> DependencyLocalizationReport:
        _validate_portable_package_path(self.localized_source_package_path)
        if self.unresolved and self.dependency_complete:
            raise ValueError("dependency_complete cannot be true with unresolved evidence")
        if self.unresolved and self.material_fidelity_status != (
            "not_claimed_unresolved_dependencies"
        ):
            raise ValueError("unresolved dependencies must block a material-fidelity claim")
        if self.portable_package_complete and not self.dependency_complete:
            raise ValueError("a portable package cannot be complete with unresolved dependencies")
        return self


@dataclass(frozen=True)
class _RootApproval:
    path: Path
    method: DependencyResolutionMethod


def _path_within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current /= part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _portable_filename(name: str, *, max_length: int = 120) -> str:
    portable = _PORTABLE_FILENAME_CHARACTER.sub("_", name)
    portable = portable.strip(".")
    if not portable or portable in {".", ".."}:
        portable = "dependency"
    if len(portable) > max_length:
        suffix = Path(portable).suffix[:16]
        identity = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
        stem_length = max_length - len(suffix) - len(identity) - 1
        portable = f"{portable[:stem_length]}_{identity}{suffix}"
    return portable


def _validate_portable_package_path(value: str) -> None:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"package path is not portable: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"package path must be normalized and relative: {value!r}")
    if urlsplit(value).scheme:
        raise ValueError(f"package path cannot contain a URI scheme: {value!r}")


def _split_package_asset_path(value: str) -> tuple[str, str | None]:
    try:
        from pxr import Ar
    except ImportError:
        return value, None
    if not Ar.IsPackageRelativePath(value):
        return value, None
    return (
        str(Ar.SplitPackageRelativePathOuter(value)),
        str(Ar.SplitPackageRelativePathInner(value)),
    )


def _validate_portable_asset_path(value: str) -> None:
    outer, member = _split_package_asset_path(value)
    _validate_portable_package_path(outer)
    if member is not None:
        _validate_package_member(member)


def _validate_package_member(member: str) -> None:
    if not member or "\\" in member or "\x00" in member:
        raise ValueError(f"package member is not portable: {member!r}")
    path = PurePosixPath(member)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"package member must be normalized and relative: {member!r}")


def split_asset_identifier(identifier: str) -> tuple[str, str | None, str | None]:
    """Return a local outer path, package member, or non-file URI scheme."""

    outer, member = _split_package_asset_path(identifier)
    parsed = urlsplit(outer)
    if parsed.scheme and not _WINDOWS_ABSOLUTE.match(outer):
        scheme = parsed.scheme.lower()
        if scheme == "file" and parsed.netloc in {"", "localhost"}:
            return unquote(parsed.path), member, None
        return "", member, scheme
    return outer, member, None


def _normalize_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "mappings" in payload:
        normalized = dict(payload)
        normalized["remaps"] = normalized.pop("mappings")
        return normalized
    if "remaps" in payload or "schema_version" in payload:
        return dict(payload)
    remaps: list[dict[str, Any]] = []
    for authored_path, target in payload.items():
        if isinstance(target, Mapping):
            item = dict(target)
            item.setdefault("authored_path", str(authored_path))
        else:
            item = {"authored_path": str(authored_path), "local_path": str(target)}
        remaps.append(item)
    return {"remaps": remaps}


def _anchor_manifest_paths(
    manifest: DependencyRemapManifest,
    *,
    base_dir: Path | None,
) -> DependencyRemapManifest:
    anchored: list[DependencyRemap] = []
    for remap in manifest.remaps:
        local = Path(remap.local_path).expanduser()
        if ".." in local.parts:
            raise ValueError("dependency remap paths cannot contain parent traversal")
        if not local.is_absolute():
            if base_dir is None:
                raise ValueError("in-memory dependency remap paths must be absolute")
            local = base_dir / local
        source_layer = remap.source_layer
        if source_layer is not None:
            layer_outer, layer_member, layer_scheme = split_asset_identifier(source_layer)
            if layer_scheme is not None:
                raise ValueError("dependency remap source_layer must be a local identifier")
            layer_path = Path(layer_outer).expanduser()
            if ".." in layer_path.parts:
                raise ValueError("remap source_layer cannot contain parent traversal")
            if not layer_path.is_absolute():
                if base_dir is None:
                    raise ValueError("in-memory remap source_layer paths must be absolute")
                layer_path = base_dir / layer_path
            source_layer = str(layer_path.absolute())
            if layer_member:
                _validate_package_member(layer_member)
                source_layer = f"{source_layer}[{layer_member}]"
        anchored.append(
            remap.model_copy(
                update={
                    "local_path": str(local.absolute()),
                    "source_layer": source_layer,
                }
            )
        )
    return DependencyRemapManifest(
        remaps=sorted(
            anchored,
            key=lambda item: (item.source_layer or "", item.authored_path, item.local_path),
        )
    )


def load_dependency_remap_manifest(value: DependencyRemapInput) -> DependencyRemapManifest:
    """Load and canonicalize an exact remap manifest without resolving any URI."""

    if value is None:
        return DependencyRemapManifest()
    if isinstance(value, DependencyRemapManifest):
        return _anchor_manifest_paths(value, base_dir=None)

    base_dir = None
    if isinstance(value, str | Path):
        manifest_path = Path(value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = manifest_path.absolute()
        if _has_symlink_component(manifest_path):
            raise ValueError("dependency remap manifest cannot be read through a symlink")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("dependency remap manifest must contain a JSON object")
        base_dir = manifest_path.parent
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise TypeError(f"unsupported dependency remap manifest type: {type(value).__name__}")
    manifest = DependencyRemapManifest.model_validate(_normalize_manifest_payload(payload))
    return _anchor_manifest_paths(manifest, base_dir=base_dir)


class DependencyApprovalPolicy:
    """Resolve only exact remaps or paths contained by caller-approved roots."""

    def __init__(
        self,
        *,
        approved_roots: Sequence[str | Path] = (),
        remap_manifest: DependencyRemapInput = None,
        source_directory_root: str | Path | None = None,
    ) -> None:
        roots: list[_RootApproval] = []
        if source_directory_root is not None:
            roots.append(
                _RootApproval(
                    Path(source_directory_root).expanduser().resolve(),
                    "source_directory",
                )
            )
        for raw_root in approved_roots:
            root = Path(raw_root).expanduser().resolve()
            if not root.is_dir():
                raise NotADirectoryError(f"approved dependency root is not a directory: {root}")
            roots.append(_RootApproval(root, "approved_root"))
        self._roots = tuple(
            sorted(
                {(item.path, item.method): item for item in roots}.values(),
                key=lambda item: (str(item.path), item.method),
            )
        )
        self.remap_manifest = load_dependency_remap_manifest(remap_manifest)
        self._remaps: dict[tuple[str | None, str], DependencyRemap] = {}
        for item in self.remap_manifest.remaps:
            source_layer = (
                _canonical_layer_identifier(item.source_layer) if item.source_layer else None
            )
            selector = (source_layer, item.authored_path)
            if selector in self._remaps:
                raise ValueError("dependency remap selectors must remain unique when canonicalized")
            self._remaps[selector] = item

    @property
    def approved_roots(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self._roots)

    def _exact_remap(self, source_layer: str, authored_path: str) -> DependencyRemap | None:
        canonical_layer = _canonical_layer_identifier(source_layer)
        return self._remaps.get((canonical_layer, authored_path)) or self._remaps.get(
            (None, authored_path)
        )

    def resolve(self, *, source_layer: str, authored_path: str) -> DependencySourceDecision:
        """Resolve one occurrence without search, globbing, or network access."""

        if any(marker in authored_path for marker in _DYNAMIC_ASSET_MARKERS):
            return DependencySourceDecision(
                resolved=False,
                reason_code="dynamic_pattern",
                reason="authored asset path contains a dynamic expansion marker",
            )
        remap = self._exact_remap(source_layer, authored_path)
        if remap is not None:
            candidate = Path(remap.local_path)
            if _has_symlink_component(candidate):
                return DependencySourceDecision(
                    resolved=False,
                    reason_code="symlink_escape",
                    reason="exact remap targets cannot contain symlink components",
                )
            candidate = candidate.resolve(strict=False)
            if not candidate.is_file():
                return DependencySourceDecision(
                    resolved=False,
                    local_path=str(candidate),
                    reason_code="remap_target_missing",
                    reason="exact remap target is not a regular file",
                )
            _outer, package_member, _scheme = split_asset_identifier(authored_path)
            if package_member is not None:
                try:
                    _validate_package_member(package_member)
                except ValueError as exc:
                    return DependencySourceDecision(
                        resolved=False,
                        reason_code="unsafe_package_member",
                        reason=str(exc),
                    )
            return DependencySourceDecision(
                resolved=True,
                local_path=str(candidate),
                package_member=package_member,
                resolution_method="exact_remap",
                expected_sha256=remap.sha256,
            )

        local_identifier, package_member, remote_scheme = split_asset_identifier(authored_path)
        if remote_scheme is not None:
            return DependencySourceDecision(
                resolved=False,
                remote_scheme=remote_scheme,
                reason_code="remote_not_fetched",
                reason="remote asset identifiers are recorded but never fetched",
            )
        if package_member is not None:
            try:
                _validate_package_member(package_member)
            except ValueError as exc:
                return DependencySourceDecision(
                    resolved=False,
                    reason_code="unsafe_package_member",
                    reason=str(exc),
                )
        if not local_identifier or "\x00" in local_identifier:
            return DependencySourceDecision(
                resolved=False,
                reason_code="invalid_local_path",
                reason="authored dependency path is empty or contains a null byte",
            )

        source_outer, _source_member, source_scheme = split_asset_identifier(source_layer)
        if source_scheme is not None:
            return DependencySourceDecision(
                resolved=False,
                reason_code="non_local_source_layer",
                reason="dependency source layer is not a local identifier",
            )
        raw_candidate = Path(local_identifier).expanduser()
        if not raw_candidate.is_absolute():
            raw_candidate = Path(source_outer).parent / raw_candidate
        lexical_candidate = Path(os.path.abspath(raw_candidate))
        resolved_candidate = raw_candidate.resolve(strict=False)
        approved_paths = tuple(item.path for item in self._roots)
        lexical_approved = _path_within(lexical_candidate, approved_paths)
        resolved_approval = next(
            (
                item
                for item in self._roots
                if resolved_candidate == item.path or resolved_candidate.is_relative_to(item.path)
            ),
            None,
        )
        if resolved_approval is None:
            if lexical_approved:
                reason_code = "symlink_escape"
                reason = "dependency path escapes an approved root through a symlink"
            elif ".." in Path(local_identifier.replace("\\", "/")).parts:
                reason_code = "traversal_escape"
                reason = "dependency path traversal escapes all approved roots"
            else:
                reason_code = "not_approved"
                reason = "resolved dependency path is outside caller-approved roots"
            return DependencySourceDecision(
                resolved=False,
                local_path=str(resolved_candidate),
                package_member=package_member,
                reason_code=reason_code,
                reason=reason,
            )
        if not resolved_candidate.is_file():
            return DependencySourceDecision(
                resolved=False,
                local_path=str(resolved_candidate),
                package_member=package_member,
                resolution_method=resolved_approval.method,
                reason_code="local_file_missing",
                reason="approved local dependency is not a regular file",
            )
        return DependencySourceDecision(
            resolved=True,
            local_path=str(resolved_candidate),
            package_member=package_member,
            resolution_method=resolved_approval.method,
        )


def _canonical_layer_identifier(identifier: str) -> str:
    outer, member, scheme = split_asset_identifier(identifier)
    if scheme is not None:
        return identifier
    canonical = str(Path(outer).expanduser().resolve(strict=False))
    return f"{canonical}[{member}]" if member else canonical


def _stable_hash(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"file changed while it was hashed: {path}")
    return digest.hexdigest(), after.st_size


def _safe_destination(package_root: Path, package_path: str) -> Path:
    _validate_portable_package_path(package_path)
    target = package_root.joinpath(*PurePosixPath(package_path).parts)
    current = package_root
    for part in PurePosixPath(package_path).parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"package destination contains a symlink: {current}")
    if target.is_symlink():
        raise ValueError(f"package destination is a symlink: {target}")
    if not target.parent.resolve(strict=False).is_relative_to(package_root):
        raise ValueError(f"package destination escapes package root: {package_path}")
    return target


def _copy_verified(source: Path, target: Path) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    before = source.stat()
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_stream:
                while chunk := input_stream.read(_CHUNK_BYTES):
                    digest.update(chunk)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = source.stat()
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise RuntimeError(f"source changed while it was localized: {source}")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    localized_digest, localized_size = _stable_hash(target)
    if localized_digest != digest.hexdigest() or localized_size != before.st_size:
        raise RuntimeError(f"localized dependency does not match its source: {source}")
    return digest.hexdigest(), before.st_size


def _atomic_write_json(path: Path, payload: BaseModel) -> None:
    document = (
        json.dumps(
            payload.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _package_asset_path(package_path: str, package_member: str | None) -> str:
    return f"{package_path}[{package_member}]" if package_member else package_path


def _relative_asset_path(
    *,
    layer_package_path: str,
    dependency_package_path: str,
    package_member: str | None,
) -> str:
    layer_parent = PurePosixPath(layer_package_path).parent.as_posix()
    if layer_parent == ".":
        layer_parent = ""
    relative = posixpath.relpath(dependency_package_path, start=layer_parent or ".")
    normalized = posixpath.normpath(posixpath.join(layer_parent, relative))
    if normalized != dependency_package_path:
        raise ValueError("rewritten dependency path would escape the portable package")
    if "\\" in relative or urlsplit(relative).scheme or relative.startswith("/"):
        raise ValueError("rewritten dependency path is not portable")
    return _package_asset_path(relative, package_member)


def _rewrite_localized_layers(
    *,
    package_root: Path,
    layer_package_paths: Mapping[str, str],
    dependency_package_paths: Mapping[tuple[str, str], tuple[str, str | None]],
) -> tuple[set[tuple[str, str]], list[UnresolvedDependencyEvidence]]:
    from pxr import Sdf, UsdUtils

    applied: set[tuple[str, str]] = set()
    failures: list[UnresolvedDependencyEvidence] = []
    for original_layer, layer_package_path in sorted(layer_package_paths.items()):
        authored_for_layer = {
            authored: target
            for (source_layer, authored), target in dependency_package_paths.items()
            if source_layer == original_layer
        }
        if not authored_for_layer:
            continue
        localized_layer = _safe_destination(package_root, layer_package_path)
        if localized_layer.suffix.lower() not in _USD_SUFFIXES:
            for authored_path in sorted(authored_for_layer):
                failures.append(
                    UnresolvedDependencyEvidence(
                        source_layer=original_layer,
                        kind="asset",
                        authored_path=authored_path,
                        status="rewrite_unavailable",
                        reason_code="unsupported_layer_package",
                        reason="dependencies inside packaged USD layers cannot be rewritten safely",
                    )
                )
            continue
        try:
            layer = Sdf.Layer.FindOrOpen(str(localized_layer))
            if layer is None:
                raise RuntimeError("Sdf.Layer.FindOrOpen returned None")

            def rewrite(
                authored_path: str,
                *,
                authored_targets: Mapping[str, tuple[str, str | None]] = authored_for_layer,
                source_layer: str = original_layer,
                source_package_path: str = layer_package_path,
            ) -> str:
                target = authored_targets.get(authored_path)
                if target is None:
                    return authored_path
                package_path, package_member = target
                applied.add((source_layer, authored_path))
                return _relative_asset_path(
                    layer_package_path=source_package_path,
                    dependency_package_path=package_path,
                    package_member=package_member,
                )

            UsdUtils.ModifyAssetPaths(layer, rewrite)
            if not layer.Save():
                raise RuntimeError("Sdf.Layer.Save returned false")
        except Exception as exc:
            for authored_path in sorted(authored_for_layer):
                applied.discard((original_layer, authored_path))
                failures.append(
                    UnresolvedDependencyEvidence(
                        source_layer=original_layer,
                        kind="asset",
                        authored_path=authored_path,
                        status="rewrite_failed",
                        reason_code="layer_rewrite_failed",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
    return applied, failures


def _unresolved_from_record(record: Any) -> UnresolvedDependencyEvidence:
    reason_code = str(getattr(record, "reason_code", None) or record.status)
    reason = str(getattr(record, "reason", None) or "dependency was not localized")
    return UnresolvedDependencyEvidence(
        source_layer=str(record.source_layer),
        kind=str(record.kind),
        authored_path=str(record.authored_path),
        status=str(record.status),
        reason_code=reason_code,
        reason=reason,
    )


def localize_usd_dependencies(
    source_path: str | Path,
    package_root: str | Path,
    *,
    approved_roots: Sequence[str | Path] = (),
    remap_manifest: DependencyRemapInput = None,
    max_dependencies: int = 20_000,
    max_hash_bytes: int = 8 * 1024 * 1024 * 1024,
) -> DependencyLocalizationReport:
    """Copy approved USD dependencies into a deterministic portable bundle.

    The source file and approved dependency files are never edited. Relative
    paths are accepted only when their resolved file remains inside an explicit
    approved root. Exact remaps match the full authored identifier, optionally
    scoped to the full source-layer identifier. Remote identifiers are never
    opened; they remain explicit unresolved evidence unless exactly remapped to
    a caller-provided local file.
    """

    from .usd_intake import inventory_usd_stage

    source = Path(source_path).expanduser().resolve()
    output = Path(package_root).expanduser()
    if not output.is_absolute():
        output = output.absolute()
    if output.exists() and output.is_symlink():
        raise ValueError("dependency package root cannot be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("dependency package root cannot contain the source file")

    manifest = load_dependency_remap_manifest(remap_manifest)
    intake = inventory_usd_stage(
        source,
        allowed_dependency_roots=approved_roots,
        dependency_remap_manifest=manifest,
        require_explicit_dependency_approval=True,
        max_dependencies=max_dependencies,
        max_hash_bytes=max_hash_bytes,
    )
    source_package_path = _portable_filename(source.name)
    if source_package_path == "dependency_localization.json":
        source_package_path = f"source_{source_package_path}"
    localized_source = _safe_destination(output, source_package_path)
    copied_source_digest, copied_source_size = _copy_verified(source, localized_source)
    if (
        copied_source_digest != intake.source_sha256
        or copied_source_size != intake.source_size_bytes
    ):
        raise RuntimeError("source changed between intake and localization")

    resolved_records = [
        record
        for record in intake.dependencies
        if record.status in {"resolved_local", "resolved_package"}
        and record.local_path is not None
        and record.sha256 is not None
        and record.size_bytes is not None
        and record.resolution_method is not None
    ]
    package_path_by_source: dict[str, str] = {str(source): source_package_path}
    source_facts: dict[str, tuple[str, int]] = {
        str(source): (intake.source_sha256, intake.source_size_bytes)
    }
    for record in resolved_records:
        dependency = Path(record.local_path).resolve()
        if dependency == output or dependency.is_relative_to(output):
            raise ValueError("approved dependency sources cannot be inside the package root")
        source_key = str(dependency)
        source_facts[source_key] = (record.sha256, record.size_bytes)
        if source_key == str(source):
            continue
        filename = _portable_filename(dependency.name)
        source_identity = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
        package_path_by_source.setdefault(
            source_key,
            f"dependencies/{record.sha256}_{source_identity}_{filename}",
        )

    for source_key, package_path in sorted(package_path_by_source.items()):
        if source_key == str(source):
            continue
        dependency = Path(source_key)
        target = _safe_destination(output, package_path)
        digest, size_bytes = _copy_verified(dependency, target)
        expected_digest, expected_size = source_facts[source_key]
        if digest != expected_digest or size_bytes != expected_size:
            raise RuntimeError(f"dependency changed between intake and localization: {dependency}")

    dependency_package_paths: dict[tuple[str, str], tuple[str, str | None]] = {}
    conflicting_keys: set[tuple[str, str]] = set()
    for record in resolved_records:
        key = (_canonical_layer_identifier(record.source_layer), record.authored_path)
        target = (
            package_path_by_source[str(Path(record.local_path).resolve())],
            record.package_member,
        )
        previous = dependency_package_paths.get(key)
        if previous is not None and previous != target:
            conflicting_keys.add(key)
        dependency_package_paths[key] = target

    layer_package_paths = {_canonical_layer_identifier(str(source)): source_package_path}
    for record in resolved_records:
        dependency = Path(record.local_path).resolve()
        if dependency.suffix.lower() in _USD_SUFFIXES:
            layer_package_paths[_canonical_layer_identifier(str(dependency))] = (
                package_path_by_source[str(dependency)]
            )
    applied, rewrite_failures = _rewrite_localized_layers(
        package_root=output,
        layer_package_paths=layer_package_paths,
        dependency_package_paths=dependency_package_paths,
    )

    unresolved = [
        _unresolved_from_record(record)
        for record in intake.dependencies
        if record.status not in {"resolved_local", "resolved_package"}
    ]
    for warning in intake.warnings:
        if warning.startswith(
            ("dependency inventory stopped", "could not inspect dependency layer")
        ):
            unresolved.append(
                UnresolvedDependencyEvidence(
                    source_layer=str(source),
                    kind="inventory",
                    authored_path="<dependency inventory>",
                    status="inventory_incomplete",
                    reason_code="inventory_incomplete",
                    reason=warning,
                )
            )
    unresolved.extend(rewrite_failures)
    for source_layer, authored_path in sorted(conflicting_keys):
        unresolved.append(
            UnresolvedDependencyEvidence(
                source_layer=source_layer,
                kind="asset",
                authored_path=authored_path,
                status="conflicting_mapping",
                reason_code="conflicting_mapping",
                reason="one source-layer identifier resolved to multiple local files",
            )
        )

    localized_facts = {
        source_key: _stable_hash(_safe_destination(output, package_path))
        for source_key, package_path in sorted(package_path_by_source.items())
    }
    mappings: list[LocalizedDependencyMapping] = []
    for record in resolved_records:
        source_key = str(Path(record.local_path).resolve())
        package_path = package_path_by_source[source_key]
        localized_sha256, localized_size = localized_facts[source_key]
        layer_key = _canonical_layer_identifier(record.source_layer)
        rewrite_applied = (layer_key, record.authored_path) in applied
        if not rewrite_applied and (layer_key, record.authored_path) not in conflicting_keys:
            unresolved.append(
                UnresolvedDependencyEvidence(
                    source_layer=record.source_layer,
                    kind=record.kind,
                    authored_path=record.authored_path,
                    status="rewrite_not_applied",
                    reason_code="rewrite_not_applied",
                    reason="localized source layer did not expose the authored asset path for rewrite",
                )
            )
        mappings.append(
            LocalizedDependencyMapping(
                source_layer=record.source_layer,
                kind=record.kind,
                authored_path=record.authored_path,
                resolution_method=record.resolution_method,
                source_path=source_key,
                source_sha256=record.sha256,
                source_size_bytes=record.size_bytes,
                package_path=package_path,
                package_asset_path=_package_asset_path(package_path, record.package_member),
                package_member=record.package_member,
                localized_sha256=localized_sha256,
                localized_size_bytes=localized_size,
                rewrite_applied=rewrite_applied,
            )
        )

    mappings.sort(key=lambda item: (item.source_layer, item.kind, item.authored_path))
    unresolved = sorted(
        set(unresolved),
        key=lambda item: (
            item.source_layer,
            item.kind,
            item.authored_path,
            item.reason_code,
            item.reason,
        ),
    )
    source_final_digest, _source_final_size = _stable_hash(source)
    source_unchanged = source_final_digest == intake.source_sha256
    if not source_unchanged:
        raise RuntimeError("source changed during dependency localization")
    localized_source_digest, _localized_source_size = _stable_hash(localized_source)
    dependency_complete = not unresolved
    warnings = list(intake.warnings)
    if unresolved:
        warnings.append(
            "material fidelity is not claimed because dependency localization is incomplete"
        )
    evidence_path = _safe_destination(output, "dependency_localization.json")
    report = DependencyLocalizationReport(
        source_path=str(source),
        source_sha256=intake.source_sha256,
        source_size_bytes=intake.source_size_bytes,
        source_unchanged=source_unchanged,
        package_root=str(output),
        localized_source_path=str(localized_source),
        localized_source_package_path=source_package_path,
        localized_source_sha256=localized_source_digest,
        mappings=mappings,
        unresolved=unresolved,
        dependency_complete=dependency_complete,
        portable_package_complete=dependency_complete
        and all(item.rewrite_applied for item in mappings),
        material_fidelity_status=(
            "not_evaluated_dependency_complete"
            if dependency_complete
            else "not_claimed_unresolved_dependencies"
        ),
        evidence_path=str(evidence_path),
        warnings=sorted(set(warnings)),
    )
    _atomic_write_json(evidence_path, report)
    return report


__all__ = [
    "DEPENDENCY_LOCALIZATION_SCHEMA_VERSION",
    "DEPENDENCY_REMAP_SCHEMA_VERSION",
    "DependencyApprovalPolicy",
    "DependencyLocalizationReport",
    "DependencyRemap",
    "DependencyRemapInput",
    "DependencyRemapManifest",
    "DependencyResolutionMethod",
    "DependencySourceDecision",
    "LocalizedDependencyMapping",
    "UnresolvedDependencyEvidence",
    "load_dependency_remap_manifest",
    "localize_usd_dependencies",
    "split_asset_identifier",
]
