# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic dependency discovery for staged non-USD geometry sources."""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

MAX_SOURCE_CLOSURE_FILES = 20_000
MAX_SOURCE_CLOSURE_BYTES = 64 * 1024 * 1024 * 1024
MAX_SOURCE_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_ANCESTOR_DEPTH = 4

# These formats carry their complete geometry in one file. Formats that may
# contain opaque external references require an explicit source tree instead.
_SELF_CONTAINED_SUFFIXES = frozenset(
    {
        ".3mf",
        ".brep",
        ".brp",
        ".e57",
        ".glb",
        ".iges",
        ".igs",
        ".ifczip",
        ".ply",
        ".pts",
        ".sab",
        ".sat",
        ".step",
        ".stl",
        ".stp",
        ".x_b",
        ".x_t",
        ".xmt",
        ".xmt_txt",
    }
)
_PARSED_SUFFIXES = frozenset({".dae", ".gltf", ".mjcf", ".obj", ".urdf", ".xml"})
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\\\/]")
_MTL_TEXTURE_COMMANDS = frozenset(
    {
        "bump",
        "decal",
        "disp",
        "map_bump",
        "map_d",
        "map_ka",
        "map_kd",
        "map_ke",
        "map_ks",
        "map_ns",
        "map_pm",
        "map_pr",
        "map_ps",
        "norm",
        "refl",
    }
)


@dataclass(frozen=True)
class NonUsdSourceClosure:
    """One verified source-root-relative file closure, including its root file."""

    source: Path
    source_root: Path
    files: tuple[Path, ...]
    strategy: str


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _bounded_bytes(path: Path) -> bytes:
    size = path.stat().st_size
    if size > MAX_SOURCE_DOCUMENT_BYTES:
        raise ValueError(
            f"Source dependency document exceeds {MAX_SOURCE_DOCUMENT_BYTES} bytes: {path}"
        )
    return path.read_bytes()


def _xml_root(path: Path) -> ET.Element:
    try:
        payload = _bounded_bytes(path)
        upper_payload = payload.upper()
        if b"<!DOCTYPE" in upper_payload or b"<!ENTITY" in upper_payload:
            raise ValueError(
                f"Source dependency XML must not declare a DTD or entity: {path}"
            )
        return ET.fromstring(payload)
    except (ET.ParseError, OSError, UnicodeError) as exc:
        raise ValueError(f"Source dependency XML is invalid: {path}: {exc}") from exc


def _package_identity_at_root(candidate_root: Path) -> tuple[Path, str] | None:
    manifest = candidate_root / "package.xml"
    if not manifest.is_file() or manifest.is_symlink():
        return None
    root = _xml_root(manifest)
    if _local_name(root.tag) != "package":
        return None
    name = next(
        (
            (item.text or "").strip()
            for item in root
            if _local_name(item.tag) == "name" and (item.text or "").strip()
        ),
        "",
    )
    if name:
        return candidate_root.resolve(strict=True), name
    return None


def _package_identity(source: Path) -> tuple[Path, str] | None:
    # ROS packages are normally one or two levels above their URDF. Keep this
    # discovery bounded so an unrelated package.xml near the filesystem root
    # cannot silently widen the approved dependency root.
    for candidate_root in source.parents[:MAX_PACKAGE_ANCESTOR_DEPTH]:
        if candidate_root == candidate_root.parent:
            break
        identity = _package_identity_at_root(candidate_root)
        if identity is not None:
            return identity
    return None


def source_root_stages_whole_tree(source_path: Path) -> bool:
    """Return whether an explicit root must be frozen as an opaque tree."""

    suffix = source_path.suffix.lower()
    return suffix not in _SELF_CONTAINED_SUFFIXES and suffix not in _PARSED_SUFFIXES


def _xml_kind(source: Path) -> str:
    root_name = _local_name(_xml_root(source).tag)
    expected = {
        ".urdf": "robot",
        ".mjcf": "mujoco",
    }.get(source.suffix.lower())
    if expected is not None and root_name != expected:
        raise ValueError(
            f"Source {source.suffix.lower()} document must have a {expected!r} "
            f"root element: {source}"
        )
    if root_name == "robot":
        return "urdf"
    if root_name == "mujoco":
        return "mjcf"
    raise ValueError(
        "Source XML must have a 'robot' (URDF) or 'mujoco' (MJCF) root "
        f"element: {source}"
    )


def _reject_symlink_components(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(
                f"Source dependency must not traverse a symlink: {current}"
            )


def _contained_regular_file(root: Path, candidate: Path, *, reference: str) -> Path:
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Source dependency escapes its approved root: {reference!r}"
        ) from exc
    try:
        _reject_symlink_components(root, lexical)
        metadata = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"Source dependency is missing: {reference!r}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Source dependency is not a regular file: {reference!r}")
    return lexical.resolve(strict=True)


def _is_portable_absolute_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        Path(normalized).is_absolute()
        or normalized.startswith("//")
        or _WINDOWS_ABSOLUTE_PATH.match(normalized) is not None
    )


def _is_portable_absolute_reference(value: str) -> bool:
    """Recognize literal or once-encoded absolute local references."""

    if _is_portable_absolute_path(value):
        return True
    parsed = urlsplit(value)
    decoded_path = unquote(parsed.path)
    if len(parsed.scheme) == 1:
        return _is_portable_absolute_path(f"{parsed.scheme}:{decoded_path}")
    return not parsed.scheme and _is_portable_absolute_path(decoded_path)


def _resolve_reference(
    reference: str,
    *,
    declaring_file: Path,
    source_root: Path,
    package_name: str | None = None,
    package_root: Path | None = None,
    literal_local_fragment: bool = False,
) -> Path | None:
    value = reference.strip()
    if not value:
        raise ValueError(f"Empty source dependency reference in {declaring_file}")
    if value.lower().startswith("data:"):
        return None
    if "$" in value:
        raise ValueError(
            f"Dynamic source dependency references are not supported: {value!r}"
        )
    # Reject literal absolute paths before URI parsing. Encoded path data is
    # decoded exactly once from ``parsed.path`` and checked below.
    if _is_portable_absolute_reference(value):
        raise ValueError(f"Absolute source dependency is not allowed: {value!r}")

    # OBJ and MTL declarations are filesystem paths rather than URI references.
    # Preserve a literal hash there while retaining strict URI fragment handling
    # for glTF, XML, and COLLADA callers.
    parsed = urlsplit(value.replace("#", "%23") if literal_local_fragment else value)
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"Source dependency URI query/fragment is not supported: {value!r}"
        )
    decoded_path = unquote(parsed.path)
    if parsed.scheme.lower() == "package":
        requested_package = unquote(parsed.netloc)
        current_package_alias = not requested_package and parsed.path.startswith("/")
        if not package_name:
            raise ValueError(
                "Source package dependency has no verified package identity; "
                "provide --source-root for package layouts beyond automatic "
                f"discovery: {value!r}"
            )
        if requested_package != package_name and not current_package_alias:
            raise ValueError(
                f"Source package dependency is outside the approved package: {value!r}"
            )
        encoded_relative = (
            parsed.path[1:] if parsed.path.startswith("/") else parsed.path
        )
        relative = unquote(encoded_relative).replace("\\", "/")
        if _is_portable_absolute_path(relative):
            raise ValueError(f"Absolute source dependency is not allowed: {value!r}")
        if package_root is None:
            raise ValueError(
                f"Source package dependency has no verified package root: {value!r}"
            )
        candidate = package_root / relative
    elif parsed.scheme:
        raise ValueError(
            f"Remote or absolute source dependency is not allowed: {value!r}"
        )
    else:
        relative = decoded_path.replace("\\", "/")
        if _is_portable_absolute_path(relative):
            raise ValueError(f"Absolute source dependency is not allowed: {value!r}")
        candidate = declaring_file.parent / relative
    return _contained_regular_file(source_root, candidate, reference=value)


def _xml_file_references(
    path: Path,
    *,
    source_root: Path,
    package_name: str | None,
    package_root: Path | None,
    kind: str,
    mjcf_directories: dict[str, str] | None = None,
    mjcf_root_document: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    root = _xml_root(path)
    dependencies: list[Path] = []
    includes: list[Path] = []
    directories = mjcf_directories or {}

    for element in root.iter():
        tag = _local_name(element.tag)
        references: list[tuple[str, str]] = []
        if tag == "include":
            for attribute in ("file", "filename"):
                if element.get(attribute):
                    references.append(("include", str(element.get(attribute))))
        if kind == "urdf" and tag in {"mesh", "texture"}:
            for attribute in ("filename", "file", "url"):
                if element.get(attribute):
                    references.append((tag, str(element.get(attribute))))
        if kind == "urdf" and tag == "uri" and (element.text or "").strip():
            references.append(("mesh", str(element.text)))
        if kind == "mjcf" and tag in {"hfield", "mesh", "skin", "texture"}:
            if element.get("file"):
                references.append((tag, str(element.get("file"))))

        for reference_kind, raw_reference in references:
            reference = raw_reference
            reference_document = path
            if kind == "mjcf":
                if mjcf_root_document is None:
                    raise ValueError(
                        "MJCF dependency discovery requires its root document"
                    )
                # MuJoCo first merges every include into the main document. Include
                # paths and compiler asset directories therefore remain relative to
                # the main MJCF file, never to the declaring include or approved root.
                reference_document = mjcf_root_document
                if reference_kind != "include":
                    # Compiler directories apply only to relative asset names.
                    # Reject the raw reference first so prefixing cannot turn an
                    # absolute path into an apparently contained relative path.
                    if _is_portable_absolute_reference(reference):
                        raise ValueError(
                            f"Absolute source dependency is not allowed: {reference!r}"
                        )
                    directory_kind = (
                        "mesh"
                        if reference_kind in {"hfield", "skin"}
                        else reference_kind
                    )
                    directory = directories.get(directory_kind) or directories.get(
                        "asset"
                    )
                    if directory:
                        reference = f"{directory.rstrip('/')}/{reference.lstrip('/')}"
            resolved = _resolve_reference(
                reference,
                declaring_file=reference_document,
                source_root=source_root,
                package_name=package_name,
                package_root=package_root,
            )
            if resolved is None:
                continue
            dependencies.append(resolved)
            if reference_kind == "include":
                includes.append(resolved)
    return dependencies, includes


def _xml_closure(
    source: Path,
    *,
    source_root: Path,
    kind: str,
    package: tuple[Path, str] | None,
) -> set[Path]:
    package_name = package[1] if package is not None else None
    package_root = package[0] if package is not None else None
    files: set[Path] = {source}
    if package is not None:
        files.add(
            _contained_regular_file(
                source_root,
                package[0] / "package.xml",
                reference="package.xml",
            )
        )

    mjcf_directories: dict[str, str] = {}
    if kind == "mjcf":
        root = _xml_root(source)
        for element in root.iter():
            if _local_name(element.tag) != "compiler":
                continue
            asset_dir = str(element.get("assetdir") or "").strip()
            if asset_dir:
                mjcf_directories["asset"] = asset_dir
            for key, attribute in (("mesh", "meshdir"), ("texture", "texturedir")):
                value = str(element.get(attribute) or "").strip()
                if value:
                    mjcf_directories[key] = value

    pending = [source]
    inspected: set[Path] = set()
    while pending:
        document = pending.pop()
        if document in inspected:
            continue
        inspected.add(document)
        dependencies, includes = _xml_file_references(
            document,
            source_root=source_root,
            package_name=package_name,
            package_root=package_root,
            kind=kind,
            mjcf_directories=mjcf_directories,
            mjcf_root_document=source if kind == "mjcf" else None,
        )
        files.update(dependencies)
        pending.extend(includes)
    return files


def _shell_tokens(line: str, *, path: Path) -> list[str]:
    quote: str | None = None
    comment_index: int | None = None
    for index, character in enumerate(line):
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "#" and (index == 0 or line[index - 1].isspace()):
            comment_index = index
            break
    declaration = line if comment_index is None else line[:comment_index]
    try:
        lexer = shlex.shlex(declaration, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        # OBJ and MTL files in cross-platform datasets frequently spell paths
        # with Windows separators. Backslashes and embedded hash characters are
        # path data here, not shell escapes or unconditional comment markers.
        lexer.escape = ""
        return list(lexer)
    except ValueError as exc:
        raise ValueError(
            f"Malformed source dependency declaration in {path}: {line!r}"
        ) from exc


def _obj_closure(source: Path, *, source_root: Path) -> set[Path]:
    files: set[Path] = {source}
    materials: list[Path] = []
    for line in _bounded_bytes(source).decode("utf-8", errors="strict").splitlines():
        tokens = _shell_tokens(line, path=source)
        if not tokens or tokens[0].lower() != "mtllib":
            continue
        if len(tokens) == 1:
            raise ValueError(f"OBJ mtllib declaration has no path: {source}")
        for value in tokens[1:]:
            resolved = _resolve_reference(
                value,
                declaring_file=source,
                source_root=source_root,
                literal_local_fragment=True,
            )
            if resolved is not None:
                materials.append(resolved)
                files.add(resolved)

    for material in materials:
        for line in (
            _bounded_bytes(material).decode("utf-8", errors="strict").splitlines()
        ):
            tokens = _shell_tokens(line, path=material)
            if not tokens or tokens[0].lower() not in _MTL_TEXTURE_COMMANDS:
                continue
            if len(tokens) == 1:
                raise ValueError(f"MTL texture declaration has no path: {material}")
            resolved = _resolve_reference(
                tokens[-1],
                declaring_file=material,
                source_root=source_root,
                literal_local_fragment=True,
            )
            if resolved is not None:
                files.add(resolved)
    return files


def _gltf_closure(source: Path, *, source_root: Path) -> set[Path]:
    try:
        payload = json.loads(_bounded_bytes(source))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"glTF dependency manifest is invalid: {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"glTF dependency manifest is not an object: {source}")
    files: set[Path] = {source}
    for collection_name in ("buffers", "images"):
        collection = payload.get(collection_name, [])
        if not isinstance(collection, list):
            raise ValueError(f"glTF {collection_name} must be an array: {source}")
        for item in collection:
            if not isinstance(item, dict) or "uri" not in item:
                continue
            uri = item["uri"]
            if not isinstance(uri, str):
                raise ValueError(
                    f"glTF {collection_name} URI must be a string: {source}"
                )
            resolved = _resolve_reference(
                uri,
                declaring_file=source,
                source_root=source_root,
            )
            if resolved is not None:
                files.add(resolved)
    return files


def _collada_reference(
    value: str,
    *,
    declaring_file: Path,
    source_root: Path,
) -> Path | None:
    parsed = urlsplit(value.strip())
    if parsed.fragment:
        if not parsed.path:
            return None
        value = parsed._replace(fragment="").geturl()
    return _resolve_reference(
        value,
        declaring_file=declaring_file,
        source_root=source_root,
    )


def _collada_closure(source: Path, *, source_root: Path) -> set[Path]:
    files: set[Path] = {source}
    pending = [source]
    inspected: set[Path] = set()
    while pending:
        document = pending.pop()
        if document in inspected:
            continue
        inspected.add(document)
        root = _xml_root(document)
        references: list[str] = []
        for library in root.iter():
            if _local_name(library.tag) != "library_images":
                continue
            for element in library.iter():
                if _local_name(element.tag) != "init_from":
                    continue
                # COLLADA 1.4 stores the image URI directly in init_from;
                # COLLADA 1.5 wraps it in a ref child. Embedded hex payloads do
                # not add an external source dependency.
                direct_reference = (element.text or "").strip()
                if direct_reference:
                    references.append(direct_reference)
                references.extend(
                    child_reference
                    for child in element.iter()
                    if child is not element and _local_name(child.tag) == "ref"
                    if (child_reference := (child.text or "").strip())
                )
        for element in root.iter():
            # COLLADA's instance elements use URI-valued ``url`` attributes.
            # Follow every external document URI and ignore local ``#id`` arcs
            # in _collada_reference instead of maintaining an incomplete tag
            # allowlist as the schema evolves.
            url = element.get("url")
            if url:
                references.append(str(url))
        for value in references:
            resolved = _collada_reference(
                value,
                declaring_file=document,
                source_root=source_root,
            )
            if resolved is None:
                continue
            files.add(resolved)
            if resolved.suffix.lower() == ".dae":
                pending.append(resolved)
    return files


def _explicit_tree_files(source_root: Path) -> set[Path]:
    def raise_walk_error(error: OSError) -> None:
        raise error

    files: set[Path] = set()
    total_bytes = 0
    for current_text, directory_names, file_names in os.walk(
        source_root,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current / name
            if path.is_symlink():
                raise ValueError(f"Explicit source tree contains a symlink: {path}")
        for name in file_names:
            path = current / name
            resolved = _contained_regular_file(
                source_root,
                path,
                reference=str(path),
            )
            files.add(resolved)
            if len(files) > MAX_SOURCE_CLOSURE_FILES:
                raise ValueError(
                    f"Source dependency closure exceeds {MAX_SOURCE_CLOSURE_FILES} files"
                )
            total_bytes += resolved.stat().st_size
            if total_bytes > MAX_SOURCE_CLOSURE_BYTES:
                raise ValueError(
                    "Source dependency closure exceeds "
                    f"{MAX_SOURCE_CLOSURE_BYTES} bytes"
                )
    return files


def _validate_closure(files: set[Path], *, source: Path) -> tuple[Path, ...]:
    if source not in files:
        raise ValueError("Source dependency closure lost its root file")
    ordered = tuple(sorted(files, key=str))
    if len(ordered) > MAX_SOURCE_CLOSURE_FILES:
        raise ValueError(
            f"Source dependency closure exceeds {MAX_SOURCE_CLOSURE_FILES} files"
        )
    total_bytes = sum(path.stat().st_size for path in ordered)
    if total_bytes > MAX_SOURCE_CLOSURE_BYTES:
        raise ValueError(
            f"Source dependency closure exceeds {MAX_SOURCE_CLOSURE_BYTES} bytes"
        )
    return ordered


def discover_non_usd_source_closure(
    source_path: Path,
    *,
    explicit_source_root: Path | None = None,
) -> NonUsdSourceClosure:
    """Discover a complete, confined dependency closure for one non-USD source.

    Parsed formats copy only references reachable from the root document. ROS
    package discovery is deliberately bounded; deeper package layouts use an
    explicit source root. Opaque multi-file formats also require an explicit
    source root, whose regular files are frozen as the approved closure.
    """

    source = source_path.expanduser().resolve(strict=True)
    suffix = source.suffix.lower()
    explicit_root: Path | None = None
    if explicit_source_root is not None:
        explicit_root = explicit_source_root.expanduser().resolve(strict=True)
        if not explicit_root.is_dir():
            raise ValueError(
                f"Explicit source root is not a directory: {explicit_root}"
            )
        try:
            source.relative_to(explicit_root)
        except ValueError as exc:
            raise ValueError("Source file is outside the explicit source root") from exc
    if explicit_root is not None and source_root_stages_whole_tree(source):
        files = _explicit_tree_files(explicit_root)
        return NonUsdSourceClosure(
            source=source,
            source_root=explicit_root,
            files=_validate_closure(files, source=source),
            strategy="explicit_source_tree",
        )

    xml_kind = _xml_kind(source) if suffix in {".urdf", ".mjcf", ".xml"} else None
    package = None
    if xml_kind == "urdf":
        package = _package_identity_at_root(explicit_root) if explicit_root else None
        if package is None:
            discovered_package = _package_identity(source)
            if discovered_package is not None and (
                explicit_root is None
                or discovered_package[0] == explicit_root
                or explicit_root in discovered_package[0].parents
            ):
                package = discovered_package
    source_root = explicit_root or (
        package[0] if package is not None else source.parent.resolve(strict=True)
    )
    if suffix in _SELF_CONTAINED_SUFFIXES:
        files = {source}
        strategy = "verified_single_file_format"
    elif suffix == ".obj":
        files = _obj_closure(source, source_root=source_root)
        strategy = "obj_material_texture_graph"
    elif suffix == ".gltf":
        files = _gltf_closure(source, source_root=source_root)
        strategy = "gltf_uri_graph"
    elif suffix == ".dae":
        files = _collada_closure(source, source_root=source_root)
        strategy = "collada_reference_graph"
    elif suffix in {".urdf", ".mjcf", ".xml"}:
        if xml_kind is None:
            raise ValueError(f"Could not classify source XML: {source}")
        files = _xml_closure(
            source,
            source_root=source_root,
            kind=xml_kind,
            package=package,
        )
        strategy = f"{xml_kind}_xml_reference_graph"
    else:
        parsed = ", ".join(sorted(_PARSED_SUFFIXES))
        raise ValueError(
            "Cannot prove a closed dependency set for this non-USD source format "
            f"({suffix or '<none>'}). Provide --source-root to freeze its bounded "
            f"source tree, or use a self-contained/parsed format ({parsed})."
        )

    return NonUsdSourceClosure(
        source=source,
        source_root=source_root,
        files=_validate_closure(files, source=source),
        strategy=strategy,
    )
