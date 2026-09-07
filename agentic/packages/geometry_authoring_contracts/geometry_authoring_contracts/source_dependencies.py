# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed dependency checks for immutable geometry source packages."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import struct
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

_GLTF_JSON_CHUNK = 0x4E4F534A
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSON_DEPTH = 32
_MAX_JSON_VALUES = 250_000
_MAX_XML_BYTES = 16 * 1024 * 1024
_MAX_XML_DEPTH = 64
_MAX_XML_ELEMENTS = 100_000
_MAX_TEXT_LINE_BYTES = 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 256
_MAX_ARCHIVE_UNPACKED_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_COMPRESSION_RATIO = 200.0
_MAX_PLY_HEADER_BYTES = 1024 * 1024
_USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc"})
_NESTED_ARCHIVES = frozenset({".3mf", ".usdz", ".zip"})
_ARCHIVE_MEMBER_EXTENSIONS = frozenset(
    {
        ".bin",
        ".exr",
        ".glb",
        ".gltf",
        ".hdr",
        ".jpeg",
        ".jpg",
        ".json",
        ".model",
        ".mtl",
        ".obj",
        ".ply",
        ".png",
        ".rels",
        ".stl",
        ".tif",
        ".tiff",
        ".txt",
        ".usd",
        ".usda",
        ".usdc",
        ".webp",
        ".xml",
    }
)
_MTL_DEPENDENCY_COMMANDS = frozenset(
    {
        "bump",
        "decal",
        "disp",
        "map_bump",
        "map_d",
        "map_disp",
        "map_ka",
        "map_kd",
        "map_ke",
        "map_ks",
        "map_ns",
        "map_pr",
        "map_ps",
        "map_refl",
        "map_tr",
        "norm",
        "refl",
    }
)
PROCESSABLE_REPRESENTATION_ROLES = frozenset(
    {"design_exchange", "render_geometry", "collision_candidate", "reference"}
)
DEPENDENCY_REPRESENTATION_ROLES = PROCESSABLE_REPRESENTATION_ROLES | {"supporting_asset"}


class SourceDependencyError(ValueError):
    """A source may cause a downstream loader to access an unbound resource."""


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(maximum + 1)
    if len(content) > maximum:
        raise SourceDependencyError(f"{label} exceeds the dependency-check limit")
    return content


def _iter_bounded_text_lines(path: Path, label: str) -> Iterable[str]:
    with path.open("rb") as stream:
        while raw_line := stream.readline(_MAX_TEXT_LINE_BYTES + 1):
            if len(raw_line) > _MAX_TEXT_LINE_BYTES:
                raise SourceDependencyError(f"{label} contains an oversized line")
            try:
                yield raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SourceDependencyError(f"{label} must be UTF-8 text") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_bounded_json(content: bytes) -> dict[str, Any]:
    if len(content) > _MAX_JSON_BYTES:
        raise SourceDependencyError("geometry JSON exceeds the dependency-check limit")
    try:
        document = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise SourceDependencyError("geometry JSON is not bounded standard UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise SourceDependencyError("geometry JSON must contain an object")

    pending: list[tuple[object, int]] = [(document, 0)]
    values = 0
    while pending:
        value, depth = pending.pop()
        values += 1
        if values > _MAX_JSON_VALUES or depth > _MAX_JSON_DEPTH:
            raise SourceDependencyError("geometry JSON exceeds structural limits")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    return document


def _iter_uri_values(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "uri":
                if not isinstance(item, str):
                    raise SourceDependencyError("geometry URI fields must be strings")
                yield item
            else:
                yield from _iter_uri_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_uri_values(item)


def _normalized_files(paths: Iterable[Path]) -> frozenset[Path]:
    return frozenset(Path(os.path.abspath(path)) for path in paths)


def _resolve_bound_reference(
    reference: str,
    *,
    owner: Path,
    root: Path,
    allowed_files: frozenset[Path],
    allow_package_absolute: bool = False,
    allow_udim: bool = False,
) -> Path | None:
    if not reference or any(ord(character) < 32 for character in reference):
        raise SourceDependencyError("geometry contains an invalid dependency reference")
    if reference[:5].lower() == "data:":
        return None
    try:
        parsed = urlsplit(reference)
        decoded = unquote(parsed.path, errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise SourceDependencyError("geometry contains an invalid dependency reference") from exc
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise SourceDependencyError("geometry contains a remote or qualified dependency")
    if "\\" in decoded or any(ord(character) < 32 for character in decoded):
        raise SourceDependencyError("geometry contains an unsafe dependency path")
    if allow_package_absolute and decoded.startswith("/"):
        decoded = decoded.removeprefix("/")
        base = root
    else:
        base = owner.parent
    relative = PurePosixPath(decoded)
    if (
        not decoded
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SourceDependencyError("geometry contains an absolute or traversal dependency")
    if "<UDIM>" in decoded:
        if not allow_udim or decoded.count("<UDIM>") != 1:
            raise SourceDependencyError("geometry contains an unsupported asset pattern")
        pattern_path = Path(os.path.abspath(base / Path(*relative.parts)))
        if not pattern_path.parent.is_relative_to(root):
            raise SourceDependencyError("geometry contains an absolute or traversal dependency")
        filename_pattern = re.compile(
            re.escape(pattern_path.name).replace(re.escape("<UDIM>"), r"[1-9][0-9]{3}")
        )
        admitted_tiles = {
            item
            for item in allowed_files
            if item.parent == pattern_path.parent
            and filename_pattern.fullmatch(item.name) is not None
        }
        materialized_tiles = {
            Path(os.path.abspath(item))
            for item in pattern_path.parent.iterdir()
            if filename_pattern.fullmatch(item.name) is not None
        }
        if not admitted_tiles or materialized_tiles != admitted_tiles:
            raise SourceDependencyError("geometry references a dependency absent from its package")
        return None
    if "<" in decoded or ">" in decoded:
        raise SourceDependencyError("geometry contains an unsupported asset pattern")
    target = Path(os.path.abspath(base / Path(*relative.parts)))
    if not target.is_relative_to(root) or target not in allowed_files:
        raise SourceDependencyError("geometry references a dependency absent from its package")
    return target


def _validate_gltf(
    path: Path,
    *,
    root: Path,
    allowed_files: frozenset[Path],
    binary: bool,
) -> None:
    if binary:
        file_size = path.stat().st_size
        json_content: bytes | None = None
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"glTF":
                raise SourceDependencyError("GLB source has an invalid header")
            version, declared_length = struct.unpack_from("<II", header, 4)
            if version != 2 or declared_length != file_size:
                raise SourceDependencyError("GLB source has an invalid version or length")
            offset = 12
            while offset < file_size:
                chunk_header = stream.read(8)
                if len(chunk_header) != 8:
                    raise SourceDependencyError("GLB source has a truncated chunk header")
                chunk_length, chunk_type = struct.unpack("<II", chunk_header)
                offset += 8
                if chunk_length % 4 or offset + chunk_length > file_size:
                    raise SourceDependencyError("GLB source has an invalid chunk length")
                if chunk_type == _GLTF_JSON_CHUNK:
                    if json_content is not None:
                        raise SourceDependencyError("GLB source has multiple JSON chunks")
                    if chunk_length > _MAX_JSON_BYTES:
                        raise SourceDependencyError("GLB JSON exceeds the dependency-check limit")
                    raw_json = stream.read(chunk_length)
                    if len(raw_json) != chunk_length:
                        raise SourceDependencyError("GLB source has a truncated JSON chunk")
                    json_content = raw_json.rstrip(b" \t\r\n\x00")
                else:
                    stream.seek(chunk_length, os.SEEK_CUR)
                offset += chunk_length
        if json_content is None:
            raise SourceDependencyError("GLB source has no JSON chunk")
        document = _load_bounded_json(json_content)
    else:
        document = _load_bounded_json(_read_bounded(path, _MAX_JSON_BYTES, "glTF JSON"))
    asset = document.get("asset")
    if not isinstance(asset, dict) or not isinstance(asset.get("version"), str):
        raise SourceDependencyError("glTF source is missing asset version metadata")
    for uri in _iter_uri_values(document):
        _resolve_bound_reference(
            uri,
            owner=path,
            root=root,
            allowed_files=allowed_files,
        )


def _split_text_line(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=True, posix=True)
    except ValueError as exc:
        raise SourceDependencyError("geometry text contains malformed quoting") from exc


def _validate_mtl(
    path: Path,
    *,
    root: Path,
    allowed_files: frozenset[Path],
) -> None:
    for line in _iter_bounded_text_lines(path, "MTL dependency"):
        tokens = _split_text_line(line)
        if not tokens or tokens[0].casefold() not in _MTL_DEPENDENCY_COMMANDS:
            continue
        if "\\" in line:
            raise SourceDependencyError("MTL contains an unsafe dependency path")
        if len(tokens) < 2:
            raise SourceDependencyError("MTL contains an incomplete texture reference")
        _resolve_bound_reference(
            tokens[-1],
            owner=path,
            root=root,
            allowed_files=allowed_files,
        )


def _validate_obj(
    path: Path,
    *,
    root: Path,
    allowed_files: frozenset[Path],
    visited: set[Path],
) -> None:
    if path in visited:
        return
    visited.add(path)
    for line in _iter_bounded_text_lines(path, "OBJ source"):
        tokens = _split_text_line(line)
        if not tokens:
            continue
        command = tokens[0].casefold()
        if command in {"call", "csh"}:
            raise SourceDependencyError("OBJ source contains an executable command")
        if command not in {"maplib", "mtllib", "shadow_obj", "trace_obj"}:
            continue
        if "\\" in line:
            raise SourceDependencyError("OBJ source contains an unsafe dependency path")
        if len(tokens) < 2:
            raise SourceDependencyError("OBJ source contains an incomplete dependency")
        for reference in tokens[1:]:
            dependency = _resolve_bound_reference(
                reference,
                owner=path,
                root=root,
                allowed_files=allowed_files,
            )
            if dependency is not None and dependency.suffix.lower() == ".mtl":
                _validate_mtl(dependency, root=root, allowed_files=allowed_files)
            elif dependency is not None and dependency.suffix.lower() == ".obj":
                _validate_obj(
                    dependency,
                    root=root,
                    allowed_files=allowed_files,
                    visited=visited,
                )


def _validate_ply(
    path: Path,
    *,
    root: Path,
    allowed_files: frozenset[Path],
) -> None:
    lines: list[str] = []
    consumed = 0
    found_terminator = False
    with path.open("rb") as stream:
        while consumed < _MAX_PLY_HEADER_BYTES:
            raw_line = stream.readline(
                min(_MAX_TEXT_LINE_BYTES, _MAX_PLY_HEADER_BYTES - consumed) + 1
            )
            if not raw_line:
                break
            consumed += len(raw_line)
            if consumed > _MAX_PLY_HEADER_BYTES or len(raw_line) > _MAX_TEXT_LINE_BYTES:
                break
            try:
                line = raw_line.decode("ascii").removesuffix("\n").removesuffix("\r")
            except UnicodeDecodeError as exc:
                raise SourceDependencyError("PLY header must be ASCII") from exc
            if line == "end_header":
                found_terminator = True
                break
            lines.append(line)
    if not found_terminator:
        raise SourceDependencyError("PLY source has no bounded header terminator")
    for line in lines:
        fields = line.split(maxsplit=2)
        if (
            len(fields) >= 2
            and fields[0].casefold() == "comment"
            and fields[1].casefold() == "texturefile"
        ):
            if len(fields) != 3:
                raise SourceDependencyError("PLY texture has no dependency path")
            _resolve_bound_reference(
                fields[2],
                owner=path,
                root=root,
                allowed_files=allowed_files,
            )


def _validate_usd(
    path: Path,
    *,
    root: Path,
    allowed_files: frozenset[Path],
    visited: set[Path],
) -> None:
    if path in visited:
        return
    visited.add(path)
    try:
        from pxr import Sdf, UsdUtils

        layer = Sdf.Layer.OpenAsAnonymous(str(path))
    except Exception as exc:
        raise SourceDependencyError("USD source could not be inspected safely") from exc
    if not layer:
        raise SourceDependencyError("USD source could not be inspected safely")
    locators: list[str] = []

    def collect(locator: str) -> str:
        if locator:
            locators.append(str(locator))
        return locator

    try:
        UsdUtils.ModifyAssetPaths(layer, collect, keepEmptyPathsInArrays=True)
    except Exception as exc:
        raise SourceDependencyError("USD dependency metadata could not be inspected") from exc
    for locator in locators:
        dependency = _resolve_bound_reference(
            locator,
            owner=path,
            root=root,
            allowed_files=allowed_files,
            allow_udim=True,
        )
        if dependency is not None and dependency.suffix.lower() in _USD_EXTENSIONS:
            _validate_usd(
                dependency,
                root=root,
                allowed_files=allowed_files,
                visited=visited,
            )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _package_suffix(path: Path) -> str:
    return ".rels" if path.name.casefold() == ".rels" else path.suffix.lower()


def _relationship_owner(path: Path, package_root: Path) -> Path:
    relative = path.relative_to(package_root)
    if relative == Path("_rels/.rels"):
        return package_root / "package"
    if relative.parent.name != "_rels" or not relative.name.endswith(".rels"):
        raise SourceDependencyError("3MF relationship part has an invalid package path")
    source_name = relative.name.removesuffix(".rels")
    return package_root / relative.parent.parent / source_name


def _parse_package_xml(path: Path) -> ET.Element:
    content = _read_bounded(path, _MAX_XML_BYTES, "package XML")
    lowered = content.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise SourceDependencyError("package XML declarations are not allowed")
    try:
        document = ET.fromstring(content)
    except ET.ParseError as exc:
        raise SourceDependencyError("package XML is invalid") from exc
    pending: list[tuple[ET.Element, int]] = [(document, 0)]
    element_count = 0
    while pending:
        element, depth = pending.pop()
        element_count += 1
        if element_count > _MAX_XML_ELEMENTS or depth > _MAX_XML_DEPTH:
            raise SourceDependencyError("package XML exceeds structural limits")
        pending.extend((child, depth + 1) for child in element)
    return document


def validate_3mf_package_dependencies(
    package_root: Path,
    package_files: Iterable[Path],
) -> None:
    """Require every 3MF relationship and texture to stay inside the package."""

    root = Path(os.path.abspath(package_root))
    allowed_files = _normalized_files(package_files)
    for path in sorted(allowed_files):
        suffix = _package_suffix(path)
        if suffix not in {".rels", ".model"}:
            continue
        document = _parse_package_xml(path)
        for element in document.iter():
            name = _local_name(element.tag)
            if name == "relationship":
                if element.attrib.get("TargetMode", "").casefold() == "external":
                    raise SourceDependencyError("3MF package contains an external relationship")
                target = element.attrib.get("Target")
                if not target:
                    raise SourceDependencyError("3MF package contains an incomplete relationship")
                _resolve_bound_reference(
                    target,
                    owner=_relationship_owner(path, root),
                    root=root,
                    allowed_files=allowed_files,
                    allow_package_absolute=True,
                )
            elif name == "texture2d":
                target = element.attrib.get("path")
                if not target:
                    raise SourceDependencyError("3MF texture has no package path")
                _resolve_bound_reference(
                    target,
                    owner=path,
                    root=root,
                    allowed_files=allowed_files,
                    allow_package_absolute=True,
                )


def _validated_archive_members(archive: zipfile.ZipFile) -> tuple[str, ...]:
    infos = archive.infolist()
    if len(infos) > _MAX_ARCHIVE_MEMBERS:
        raise SourceDependencyError("geometry package contains too many members")
    names: set[str] = set()
    files: list[str] = []
    unpacked = 0
    compressed = 0
    for info in infos:
        raw_name = info.filename
        if not raw_name or "\x00" in raw_name or "\\" in raw_name:
            raise SourceDependencyError("geometry package contains an unsafe member name")
        member = PurePosixPath(raw_name)
        if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
            raise SourceDependencyError("geometry package contains a traversal member")
        name = member.as_posix()
        if name in names:
            raise SourceDependencyError("geometry package contains duplicate members")
        names.add(name)
        if info.flag_bits & 0x1:
            raise SourceDependencyError("encrypted geometry packages are unsupported")
        mode = (info.external_attr >> 16) & 0o170000
        if stat.S_ISLNK(mode):
            raise SourceDependencyError("geometry package links are not allowed")
        if info.is_dir():
            continue
        suffix = ".rels" if member.name.casefold() == ".rels" else member.suffix.lower()
        if suffix not in _ARCHIVE_MEMBER_EXTENSIONS:
            raise SourceDependencyError(
                f"geometry package member type is not allowed: {suffix or '<none>'}"
            )
        unpacked += info.file_size
        compressed += info.compress_size
        files.append(name)
    if unpacked > _MAX_ARCHIVE_UNPACKED_BYTES:
        raise SourceDependencyError("geometry package exceeds the unpacked byte limit")
    ratio = float("inf") if compressed == 0 and unpacked else unpacked / max(compressed, 1)
    if ratio > _MAX_ARCHIVE_COMPRESSION_RATIO:
        raise SourceDependencyError("geometry package exceeds the compression-ratio limit")
    return tuple(files)


def _validate_packaged_source_dependencies(path: Path) -> None:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceDependencyError("geometry package is not a valid ZIP container") from exc
    with archive:
        ordered_files = _validated_archive_members(archive)
        with tempfile.TemporaryDirectory(prefix="geometry-dependency-") as directory:
            root = Path(directory)
            root_resolved = root.resolve()
            for info in archive.infolist():
                member = PurePosixPath(info.filename).as_posix()
                target = root / Path(*PurePosixPath(member).parts)
                if not target.resolve().is_relative_to(root_resolved):
                    raise SourceDependencyError(
                        "geometry package member escapes its extraction root"
                    )
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            package_files = tuple(root / Path(*PurePosixPath(name).parts) for name in ordered_files)
            suffix = path.suffix.lower()
            if suffix == ".3mf":
                validate_3mf_package_dependencies(root, package_files)
                return
            if suffix == ".usdz":
                if (
                    not ordered_files
                    or Path(ordered_files[0]).suffix.lower() not in _USD_EXTENSIONS
                ):
                    raise SourceDependencyError("USDZ package must begin with its root USD layer")
                validate_materialized_source_dependencies(
                    root / Path(*PurePosixPath(ordered_files[0]).parts),
                    package_root=root,
                    package_files=package_files,
                )
                return
    raise SourceDependencyError("unsupported nested geometry package")


def validate_materialized_source_dependencies(
    source_path: Path,
    *,
    package_root: Path,
    package_files: Iterable[Path],
) -> None:
    """Validate one materialized root against an exact allowed file closure."""

    path = Path(os.path.abspath(source_path))
    root = Path(os.path.abspath(package_root))
    allowed_files = _normalized_files(package_files)
    if path not in allowed_files or not path.is_relative_to(root):
        raise SourceDependencyError("geometry root is absent from its admitted package")
    suffix = path.suffix.lower()
    if suffix in {".3mf", ".usdz"}:
        _validate_packaged_source_dependencies(path)
    elif suffix in _NESTED_ARCHIVES:
        raise SourceDependencyError("nested geometry packages are not supported")
    elif suffix == ".gltf":
        _validate_gltf(path, root=root, allowed_files=allowed_files, binary=False)
    elif suffix == ".glb":
        _validate_gltf(path, root=root, allowed_files=allowed_files, binary=True)
    elif suffix == ".obj":
        _validate_obj(path, root=root, allowed_files=allowed_files, visited=set())
    elif suffix == ".ply":
        _validate_ply(path, root=root, allowed_files=allowed_files)
    elif suffix in _USD_EXTENSIONS:
        _validate_usd(
            path,
            root=root,
            allowed_files=allowed_files,
            visited=set(),
        )


__all__ = [
    "DEPENDENCY_REPRESENTATION_ROLES",
    "PROCESSABLE_REPRESENTATION_ROLES",
    "SourceDependencyError",
    "validate_3mf_package_dependencies",
    "validate_materialized_source_dependencies",
]
