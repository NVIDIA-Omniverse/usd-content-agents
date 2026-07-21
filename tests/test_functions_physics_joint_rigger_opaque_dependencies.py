# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct adversarial coverage for bounded opaque-material references."""

from __future__ import annotations

import os
import time
import tracemalloc
from pathlib import Path

import pytest

from world_understanding.functions.physics.joint_rigger import opaque_dependencies


def test_mdl_references_ignore_comments_and_preserve_bounded_resources(
    tmp_path: Path,
) -> None:
    document = tmp_path / "Main.mdl"
    source = r"""
mdl 1.7;
string important = "not-an-import";
string marker = "literal \" // marker";
// import Commented::*; texture_2d("commented.png");
/* import Blocked::*; "blocked.mtlx" */
import ::df::*;
import Materials::Peer::*;
texture_2d("textures/explicit.png");
string generic = "textures/generic.mtlx";
string repeated = "textures/explicit.png";
string import_text = "import Ghost::*;";
"""

    stripped = opaque_dependencies.strip_mdl_comments(source, document=document)

    assert '"literal \\" // marker"' in stripped
    assert "Commented" not in stripped
    assert "Blocked" not in stripped
    assert opaque_dependencies.mdl_local_references(
        source,
        document=document,
    ) == (
        "Materials/Peer.mdl",
        "textures/explicit.png",
        "textures/generic.mtlx",
    )


def test_mdl_references_parse_runtime_and_local_using_imports(
    tmp_path: Path,
) -> None:
    source = """mdl 1.7;
string prefix = "string before compatibility alias";
using m_mdl = "mdl";
export using ::OmniPBR import OmniPBR;
import ::OmniPBR::OmniPBR;
using Materials::Peer import Surface, Tint;
using .::Sibling import *;
import ::tex::gamma_mode;
"""

    assert opaque_dependencies.mdl_local_references(
        source,
        document=tmp_path / "Main.mdl",
    ) == ("Materials/Peer.mdl", "Sibling.mdl")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('mdl 1.7;\nstring value = "unterminated;\n', "unterminated syntax"),
        ("mdl 1.7;\n/* unterminated\n", "unterminated syntax"),
        ("mdl 1.7;\nimport Peer::*\n", "unbounded import clause"),
        ("mdl 1.7;\nimport::df::*;\n", "unbounded import clause"),
        ("mdl 1.7;\nimport Peer::*, Other::*;\n", "unsupported import list"),
        (
            "mdl 1.7;\nimport ::unpackaged_vendor::Material::*;\n",
            "unapproved runtime module",
        ),
        ("mdl 1.7;\nimport Peer::Material;\n", "unprovable local import"),
        ("mdl 1.7;\nimport Peer::::Material::*;\n", "invalid local import"),
        ("mdl 1.7;\nimport ant;\n", "unprovable local import"),
        (
            "mdl 1.7;\nusing ::unpackaged_vendor::Material import Material;\n",
            "unapproved runtime module",
        ),
        (
            "mdl 1.7;\nusing ::OmniPBRExtra import Material;\n",
            "unapproved runtime module",
        ),
        (
            "mdl 1.7;\nusing ::OmniPBR::Nested import Material;\n",
            "unapproved runtime module",
        ),
        (
            "mdl 1.7;\nimport ::OmniPBR::Nested::*;\n",
            "unapproved runtime module",
        ),
        (
            'mdl 1.7;\nusing 1alias = "mdl";\n',
            "unbounded using clause",
        ),
        (
            'mdl 1.7;\nusing pkg = "materials";\n',
            "unbounded using clause",
        ),
        (
            'mdl 1.7;\nusing m_mdl = "materials";\n',
            "unbounded using clause",
        ),
        (
            'mdl 1.7;\nusing m_mdl = "mdl";\nimport m_mdl::Peer::*;\n',
            "unsupported using alias use",
        ),
        (
            "mdl 1.7;\nusing Local import One,,Two;\n",
            "unsupported using import list",
        ),
        (
            "mdl 1.7;\nusing Local import *, One;\n",
            "unsupported using import list",
        ),
        (
            "mdl 1.7;\nusing ..::Local import One;\n",
            "invalid local using import",
        ),
        ("mdl 1.7;\nusing Local import One\n", "unbounded using clause"),
        ("mdl 1.7;\nusing Local importOne;\n", "unbounded using clause"),
        (
            'mdl 1.7;\ntexture_2d("textures\\\\albedo.png");\n',
            "escaped resource path",
        ),
    ],
    ids=[
        "unterminated-string",
        "unterminated-block-comment",
        "missing-semicolon",
        "missing-whitespace",
        "import-list",
        "unapproved-runtime-module",
        "missing-wildcard",
        "empty-module-component",
        "import-keyword-boundary",
        "using-unapproved-runtime-module",
        "using-runtime-exact-name-collision",
        "using-runtime-nested-name",
        "import-runtime-nested-module",
        "invalid-using-alias",
        "unsupported-using-alias",
        "wrong-compat-using-alias-target",
        "compat-using-alias-use",
        "using-empty-selector",
        "using-wildcard-list",
        "using-parent-module",
        "using-missing-semicolon",
        "using-missing-import-whitespace",
        "backslash-resource",
    ],
)
def test_mdl_references_reject_adversarial_syntax(
    tmp_path: Path,
    source: str,
    message: str,
) -> None:
    with pytest.raises(opaque_dependencies.OpaqueDependencyError, match=message):
        opaque_dependencies.mdl_local_references(
            source,
            document=tmp_path / "Main.mdl",
        )


@pytest.mark.parametrize(
    "whitespace",
    (" ", "\t", "\u00a0"),
    ids=("space", "tab", "unicode-no-break-space"),
)
def test_mdl_references_reject_whitespace_only_import_in_bounded_time(
    tmp_path: Path,
    whitespace: str,
) -> None:
    source = "mdl 1.7;\nimport " + whitespace * 32_000
    started = time.process_time()

    with pytest.raises(
        opaque_dependencies.OpaqueDependencyError,
        match="unbounded import clause",
    ):
        opaque_dependencies.mdl_local_references(
            source,
            document=tmp_path / "Main.mdl",
        )

    assert time.process_time() - started < 1.0


def test_mdl_references_reject_repeated_unterminated_imports_in_bounded_time(
    tmp_path: Path,
) -> None:
    source = "mdl 1.7;\n" + "import x " * 8_000
    started = time.process_time()

    with pytest.raises(
        opaque_dependencies.OpaqueDependencyError,
        match="unbounded import clause",
    ):
        opaque_dependencies.mdl_local_references(
            source,
            document=tmp_path / "Main.mdl",
        )

    assert time.process_time() - started < 1.0


def test_mdl_references_scan_repeated_compat_aliases_in_bounded_time(
    tmp_path: Path,
) -> None:
    source = "mdl 1.7;\n" + 'using m_mdl = "mdl";\n' * 3_000
    started = time.process_time()

    assert (
        opaque_dependencies.mdl_local_references(
            source,
            document=tmp_path / "Main.mdl",
        )
        == ()
    )

    assert time.process_time() - started < 1.5


def test_mdl_references_scan_semicolon_clauses_with_bounded_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "value;" * 166_667
    monkeypatch.setattr(
        opaque_dependencies,
        "strip_mdl_comments",
        lambda text, *, document: text,
    )
    tracemalloc.start()
    try:
        references = opaque_dependencies.mdl_local_references(
            source,
            document=tmp_path / "Main.mdl",
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert references == ()
    assert peak_bytes < 1_000_000


def test_mdl_using_selector_scan_has_bounded_peak_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "mdl 1.7;\nusing Local import " + "Name," * 50_000 + "Name;\n"
    monkeypatch.setattr(
        opaque_dependencies,
        "strip_mdl_comments",
        lambda text, *, document: text,
    )
    tracemalloc.start()
    try:
        references = opaque_dependencies.mdl_local_references(
            source,
            document=tmp_path / "Main.mdl",
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert references == ("Local.mdl",)
    assert peak_bytes < 1_000_000


def test_mdl_using_selector_range_validation_uses_constant_extra_memory(
    tmp_path: Path,
) -> None:
    selectors = "Name," * 400_000 + "Name"
    tracemalloc.start()
    try:
        opaque_dependencies._validate_mdl_using_selectors(
            selectors,
            start=0,
            end=len(selectors),
            document=tmp_path / "Main.mdl",
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < 100_000


def test_materialx_references_cover_supported_path_attributes(tmp_path: Path) -> None:
    document = tmp_path / "surface.mtlx"
    source = """<materialx xmlns:custom="urn:custom">
  <image file="textures/file.png" />
  <image filename="textures/filename.exr" />
  <include href="includes/library.mtlx" />
  <include sourceUri="includes/source.mtlx" />
  <include custom:href="includes/namespaced.mtlx" />
  <input type="filename" value="textures/value.tx" />
  <input type="filepath" value="textures/path.vdb" />
  <input type="string" value="ignored.png" />
</materialx>"""

    assert opaque_dependencies.materialx_local_references(
        source,
        document=document,
    ) == (
        "textures/file.png",
        "textures/filename.exr",
        "includes/library.mtlx",
        "includes/source.mtlx",
        "includes/namespaced.mtlx",
        "textures/value.tx",
        "textures/path.vdb",
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            '<!DOCTYPE materialx SYSTEM "external.dtd"><materialx />',
            "DTD or entity",
        ),
        ("<!ENTITY unsafe 'value'><materialx />", "DTD or entity"),
        ("<materialx>", "invalid XML"),
        (
            '<materialx><input fileprefix="textures/" /></materialx>',
            "unprovable path prefix",
        ),
        (
            '<materialx xmlns:c="urn:c"><input c:geomprefix="/World" /></materialx>',
            "unprovable path prefix",
        ),
    ],
    ids=["doctype", "entity", "malformed", "file-prefix", "geometry-prefix"],
)
def test_materialx_references_reject_adversarial_xml(
    tmp_path: Path,
    source: str,
    message: str,
) -> None:
    with pytest.raises(opaque_dependencies.OpaqueDependencyError, match=message):
        opaque_dependencies.materialx_local_references(
            source,
            document=tmp_path / "surface.mtlx",
        )


def test_resolve_local_reference_accepts_only_unambiguous_relative_paths(
    tmp_path: Path,
) -> None:
    document = tmp_path / "materials" / "Main.mdl"

    assert opaque_dependencies.resolve_local_reference(
        document,
        "textures/./albedo.png",
    ) == Path(os.path.abspath(document.parent / "textures/albedo.png"))


def test_resolve_local_reference_allows_parent_path_only_inside_bound_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    document = root / "materials" / "Main.mdl"

    assert (
        opaque_dependencies.resolve_local_reference(
            document,
            "../textures/albedo.png",
            allowed_root=root,
        )
        == root / "textures" / "albedo.png"
    )

    with pytest.raises(
        opaque_dependencies.OpaqueDependencyError,
        match="external or ambiguous",
    ):
        opaque_dependencies.resolve_local_reference(
            document,
            "../../outside.png",
            allowed_root=root,
        )


def test_resolve_local_reference_allows_parent_path_only_in_exact_file_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    document = root / "materials" / "Main.mdl"
    target = root / "textures" / "albedo.png"

    assert opaque_dependencies.resolve_local_reference(
        document,
        "../textures/albedo.png",
        allowed_files={document, target},
    ) == Path(os.path.abspath(target))

    with pytest.raises(
        opaque_dependencies.OpaqueDependencyError,
        match="external or ambiguous",
    ):
        opaque_dependencies.resolve_local_reference(
            document,
            "../textures/albedo.png",
            allowed_files={document},
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " textures/albedo.png",
        "textures/albedo.png ",
        "../albedo.png",
        ".",
        "/textures/albedo.png",
        "C:/textures/albedo.png",
        "https://example.invalid/albedo.png",
        "//server/share/albedo.png",
        "textures/albedo.png?version=1",
        "textures/albedo.png#fragment",
        "textures\\albedo.png",
        "textures/%2e%2e/albedo.png",
        "textures/<UDIM>.png",
        "$ASSET_ROOT/albedo.png",
        "textures/albedo\x00.png",
    ],
    ids=[
        "empty",
        "leading-space",
        "trailing-space",
        "parent-traversal",
        "current-directory",
        "absolute",
        "drive-style",
        "url",
        "network-path",
        "query",
        "fragment",
        "backslash",
        "percent-encoded",
        "angle-marker",
        "expansion-marker",
        "nul",
    ],
)
def test_resolve_local_reference_rejects_external_or_ambiguous_paths(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(
        opaque_dependencies.OpaqueDependencyError,
        match="external or ambiguous",
    ):
        opaque_dependencies.resolve_local_reference(
            tmp_path / "Main.mdl",
            value,
        )


def test_opaque_reference_dispatch_parses_materialx_documents(tmp_path: Path) -> None:
    source = """<materialx>
  <image file="textures/base_color.png" />
</materialx>"""

    assert opaque_dependencies.opaque_local_references(
        source,
        document=tmp_path / "surface.mtlx",
    ) == ("textures/base_color.png",)


def test_opaque_reference_dispatch_rejects_unsupported_formats(tmp_path: Path) -> None:
    with pytest.raises(
        opaque_dependencies.OpaqueDependencyError,
        match="Unsupported opaque material dependency format",
    ):
        opaque_dependencies.opaque_local_references(
            "ignored",
            document=tmp_path / "material.txt",
        )
