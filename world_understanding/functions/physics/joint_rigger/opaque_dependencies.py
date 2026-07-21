# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded parsing for local MDL and MaterialX dependency references."""

from __future__ import annotations

import os
import re
from collections.abc import Collection
from pathlib import Path
from urllib.parse import unquote, urlparse

OPAQUE_DEPENDENCY_EXTENSIONS = frozenset({".mdl", ".mtlx"})
OPAQUE_RESOURCE_SUFFIXES = frozenset(
    {
        ".bmp",
        ".dds",
        ".exr",
        ".hdr",
        ".ies",
        ".jpeg",
        ".jpg",
        ".mbsdf",
        ".mdl",
        ".mtlx",
        ".png",
        ".tga",
        ".tif",
        ".tiff",
        ".tx",
        ".vdb",
    }
)
MDL_RUNTIME_MODULE_PREFIXES = frozenset(
    {
        "anno",
        "base",
        "builtins",
        "debug",
        "df",
        "limits",
        "math",
        "neuray",
        "state",
        "tex",
        "nvidia::core_definitions",
    }
)
MDL_RUNTIME_EXACT_MODULES = frozenset({"OmniPBR"})

_MDL_DEPENDENCY_KEYWORD_PATTERN = re.compile(r"\b(?:import|using)\b")
_MDL_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MDL_COMPAT_USING_ALIAS_PATTERN = re.compile(r'\s*m_mdl\s*=\s*"mdl"\s*')
_MDL_COMPAT_USING_ALIAS_USE_PATTERN = re.compile(r"\bm_mdl\s*::")
_MDL_RESOURCE_PATTERN = re.compile(
    r"\b(?:texture_(?:2d|3d|cube)|light_profile|bsdf_measurement)"
    r'\s*\(\s*"((?:[^"\\]|\\.)*)"'
)
_MDL_STRING_PATTERN = re.compile(r'"((?:[^"\\]|\\.)*)"')


class OpaqueDependencyError(ValueError):
    """An opaque material document has an unsafe or unbounded dependency."""


def strip_mdl_comments(text: str, *, document: Path) -> str:
    """Remove C-style comments without treating markers in strings as code."""

    output: list[str] = []
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == '"':
                state = "string"
                output.append(char)
            elif char == "/" and following == "/":
                state = "line_comment"
                output.extend((" ", " "))
                index += 1
            elif char == "/" and following == "*":
                state = "block_comment"
                output.extend((" ", " "))
                index += 1
            else:
                output.append(char)
        elif state == "string":
            output.append(char)
            if char == "\\" and following:
                output.append(following)
                index += 1
            elif char == '"':
                state = "code"
        elif state == "line_comment":
            output.append("\n" if char == "\n" else " ")
            if char == "\n":
                state = "code"
        else:
            output.append("\n" if char == "\n" else " ")
            if char == "*" and following == "/":
                output.append(" ")
                index += 1
                state = "code"
        index += 1
    if state in {"string", "block_comment"}:
        raise OpaqueDependencyError(
            f"Opaque MDL dependency has unterminated syntax: {document}"
        )
    return "".join(output)


def mdl_local_references(text: str, *, document: Path) -> tuple[str, ...]:
    """Extract provable local imports and resource paths from one MDL module."""

    stripped = strip_mdl_comments(text, document=document)
    import_source = re.sub(
        r'"(?:[^"\\]|\\.)*"',
        lambda match: '"' + " " * (len(match.group()) - 2) + '"',
        stripped,
    )
    compat_using_alias_used = (
        _MDL_COMPAT_USING_ALIAS_USE_PATTERN.search(import_source) is not None
    )
    references: list[str] = []
    dependency_matches = _MDL_DEPENDENCY_KEYWORD_PATTERN.finditer(import_source)
    match = next(dependency_matches, None)
    while match is not None:
        keyword = match.group()
        clause_end = import_source.find(";", match.end())
        next_match = next(dependency_matches, None)
        remainder_start = match.end()
        if (
            clause_end < 0
            or remainder_start >= clause_end
            or not import_source[remainder_start].isspace()
        ):
            raise OpaqueDependencyError(
                "Opaque MDL dependency contains an unbounded "
                f"{keyword} clause: {document}"
            )
        if keyword == "using":
            if next_match is None or next_match.start() >= clause_end:
                if _MDL_COMPAT_USING_ALIAS_PATTERN.fullmatch(
                    stripped,
                    remainder_start,
                    clause_end,
                ):
                    if compat_using_alias_used:
                        raise OpaqueDependencyError(
                            "Opaque MDL dependency has an unsupported using alias "
                            f"use: {document}"
                        )
                    match = next_match
                    continue
                raise OpaqueDependencyError(
                    "Opaque MDL dependency contains an unbounded using clause: "
                    f"{document}"
                )
            if next_match.group() != "import":
                raise OpaqueDependencyError(
                    "Opaque MDL dependency contains an unbounded using clause: "
                    f"{document}"
                )
            module = import_source[remainder_start : next_match.start()].strip()
            selector_start = next_match.end()
            following_match = next(dependency_matches, None)
            if (
                not module
                or selector_start >= clause_end
                or not import_source[selector_start].isspace()
                or (
                    following_match is not None and following_match.start() < clause_end
                )
            ):
                raise OpaqueDependencyError(
                    "Opaque MDL dependency contains an unbounded using clause: "
                    f"{document}"
                )
            _validate_mdl_using_selectors(
                import_source,
                start=selector_start,
                end=clause_end,
                document=document,
            )
            local_reference = _mdl_module_reference(
                module,
                document=document,
                context="using import",
            )
            if local_reference is not None:
                references.append(local_reference)
            match = following_match
            continue
        if next_match is not None and next_match.start() < clause_end:
            raise OpaqueDependencyError(
                f"Opaque MDL dependency contains an unbounded import clause: {document}"
            )
        target = import_source[remainder_start:clause_end].strip()
        match = next_match
        if not target:
            raise OpaqueDependencyError(
                f"Opaque MDL dependency contains an unbounded import clause: {document}"
            )
        if "," in target:
            raise OpaqueDependencyError(
                f"Opaque MDL dependency has an unsupported import list: {document}"
            )
        if target.startswith("::"):
            module = _mdl_runtime_import_module(target)
            if module is None or not _is_approved_mdl_runtime_module(module):
                raise OpaqueDependencyError(
                    "Opaque MDL dependency imports an unapproved runtime module "
                    f"{target!r}: {document}"
                )
            continue
        if not target.endswith("::*"):
            raise OpaqueDependencyError(
                "Opaque MDL dependency has an unprovable local import "
                f"{target!r}: {document}"
            )
        local_reference = _mdl_module_reference(
            target.removesuffix("::*"),
            document=document,
            context="import",
        )
        assert local_reference is not None
        references.append(local_reference)
    for match in _MDL_RESOURCE_PATTERN.finditer(stripped):
        value = match.group(1)
        if value:
            if "\\" in value:
                raise OpaqueDependencyError(
                    f"Opaque MDL dependency has an escaped resource path: {document}"
                )
            references.append(value)
    seen_references = set(references)
    for match in _MDL_STRING_PATTERN.finditer(stripped):
        value = match.group(1)
        if not value or value in seen_references:
            continue
        parsed = urlparse(value)
        suffix = Path(parsed.path).suffix.lower()
        if suffix in OPAQUE_RESOURCE_SUFFIXES:
            if "\\" in value:
                raise OpaqueDependencyError(
                    f"Opaque MDL dependency has an escaped resource path: {document}"
                )
            references.append(value)
            seen_references.add(value)
    return tuple(references)


def _is_approved_mdl_runtime_module(module: str) -> bool:
    """Return whether an absolute MDL module is supplied by the runtime."""

    return module in MDL_RUNTIME_EXACT_MODULES or any(
        module == prefix or module.startswith(prefix + "::")
        for prefix in MDL_RUNTIME_MODULE_PREFIXES
    )


def _mdl_runtime_import_module(target: str) -> str | None:
    """Return the runtime module named by one absolute MDL import target."""

    if not target.startswith("::"):
        return None
    qualified = target[2:]
    if qualified.endswith("::*"):
        module = qualified.removesuffix("::*")
    else:
        module, separator, symbol = qualified.rpartition("::")
        if not separator or _MDL_IDENTIFIER_PATTERN.fullmatch(symbol) is None:
            return None
    parts = module.split("::")
    if not parts or any(
        _MDL_IDENTIFIER_PATTERN.fullmatch(part) is None for part in parts
    ):
        return None
    return module


def _validate_mdl_using_selectors(
    source: str,
    *,
    start: int,
    end: int,
    document: Path,
) -> None:
    """Validate a bounded MDL using-selector range with constant extra memory."""

    index = start
    count = 0
    while True:
        separator = source.find(",", index, end)
        selector_end = end if separator < 0 else separator
        while index < selector_end and source[index].isspace():
            index += 1
        while selector_end > index and source[selector_end - 1].isspace():
            selector_end -= 1
        count += 1
        wildcard = selector_end == index + 1 and source[index:selector_end] == "*"
        if (
            index >= selector_end
            or (
                not wildcard
                and _MDL_IDENTIFIER_PATTERN.fullmatch(
                    source,
                    index,
                    selector_end,
                )
                is None
            )
            or (wildcard and (count != 1 or separator >= 0))
        ):
            raise OpaqueDependencyError(
                "Opaque MDL dependency has an unsupported using import list: "
                f"{document}"
            )
        if separator < 0:
            return
        index = separator + 1


def _mdl_module_reference(
    module: str,
    *,
    document: Path,
    context: str,
) -> str | None:
    """Validate one MDL module path and return its local sibling locator."""

    if module.startswith("::"):
        runtime_module = module[2:]
        if not _is_approved_mdl_runtime_module(runtime_module):
            raise OpaqueDependencyError(
                "Opaque MDL dependency imports an unapproved runtime module "
                f"{module!r}: {document}"
            )
        return None
    if module.startswith(".::"):
        module = module[3:]
    elif module.startswith("."):
        raise OpaqueDependencyError(
            f"Opaque MDL dependency has an invalid local {context} {module!r}: "
            f"{document}"
        )
    parts = module.split("::")
    if not parts or any(not part.isidentifier() for part in parts):
        raise OpaqueDependencyError(
            f"Opaque MDL dependency has an invalid local {context} {module!r}: "
            f"{document}"
        )
    return "/".join(parts) + ".mdl"


def materialx_local_references(text: str, *, document: Path) -> tuple[str, ...]:
    """Extract every supported path-bearing attribute from one MaterialX file."""

    import xml.etree.ElementTree as ET

    lowered = text.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise OpaqueDependencyError(
            f"Opaque MaterialX dependency contains a DTD or entity: {document}"
        )
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise OpaqueDependencyError(
            f"Opaque MaterialX dependency is invalid XML: {document}: {exc}"
        ) from exc
    references: list[str] = []
    path_attributes = {"file", "filename", "href", "sourceuri"}
    for element in root.iter():
        value_type = element.attrib.get("type", "").lower()
        for raw_name, value in element.attrib.items():
            name = raw_name.rsplit("}", 1)[-1].lower()
            if name in {"fileprefix", "geomprefix"} and value:
                raise OpaqueDependencyError(
                    "Opaque MaterialX dependency uses an unprovable path prefix "
                    f"attribute: {document}"
                )
            if name in path_attributes or (
                name == "value" and value_type in {"filename", "filepath"}
            ):
                if value:
                    references.append(value)
    return tuple(references)


def opaque_local_references(text: str, *, document: Path) -> tuple[str, ...]:
    """Extract local references from one supported opaque material document."""

    suffix = document.suffix.lower()
    if suffix == ".mdl":
        return mdl_local_references(text, document=document)
    if suffix == ".mtlx":
        return materialx_local_references(text, document=document)
    raise OpaqueDependencyError(
        f"Unsupported opaque material dependency format: {document}"
    )


def resolve_local_reference(
    document: Path,
    value: str,
    *,
    allowed_root: Path | None = None,
    allowed_files: Collection[Path] | None = None,
) -> Path:
    """Resolve one exact relative opaque-material reference lexically."""

    raw = value.strip()
    parsed = urlparse(raw)
    if (
        not raw
        or raw != value
        or unquote(raw) != raw
        or "\\" in raw
        or "\x00" in raw
        or any(marker in raw for marker in ("<", ">", "$"))
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or Path(raw).is_absolute()
        or Path(raw) == Path(".")
    ):
        raise OpaqueDependencyError(
            "Opaque material dependency has an external or ambiguous path "
            f"{value!r}: {document}"
        )
    target = Path(os.path.abspath(document.parent / raw))
    if ".." not in Path(raw).parts:
        return target
    normalized_document = Path(os.path.abspath(document))
    root_proven = allowed_root is None
    if allowed_root is not None:
        normalized_root = Path(os.path.abspath(allowed_root))
        try:
            normalized_document.relative_to(normalized_root)
            target.relative_to(normalized_root)
        except ValueError:
            pass
        else:
            root_proven = True
    files_proven = allowed_files is None
    if allowed_files is not None:
        files_proven = all(
            any(
                candidate == path or Path(os.path.abspath(candidate)) == path
                for candidate in allowed_files
            )
            for path in (normalized_document, target)
        )
    if (allowed_root is not None or allowed_files is not None) and all(
        (root_proven, files_proven)
    ):
        return target
    raise OpaqueDependencyError(
        "Opaque material dependency has an external or ambiguous path "
        f"{value!r}: {document}"
    )
