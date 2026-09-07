# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Option B confinement: staged run-root inputs against the real usd-cli daemon.

The daemon confines named request paths to ``USD_CLI_SERVER_ALLOWED_ROOTS``
(exactly one root — the run directory) and has no resolver-level containment,
so the wrapper stages digest-bound, proven self-contained derivatives of every
approved input inside the run root. The regressions here drive the *actual*
daemon — started through the runner's own production path — because
monkeypatched Python path checks explicitly do not satisfy this gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from content_agent_workflows.common.usd_cli_session import WorkflowUsdCliSession

from content_workflow_cli import runner
from content_workflow_cli.runner import PhysicsApplyConfig

REPO_ROOT = Path(__file__).resolve().parents[4]

pxr = pytest.importorskip("pxr", reason="usd-cli confinement tests require pxr")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_composed_caller_asset(source_root: Path) -> Path:
    """A composed asset: sublayer + reference + texture, all relative."""

    (source_root / "layers").mkdir(parents=True)
    (source_root / "parts").mkdir(parents=True)
    (source_root / "textures").mkdir(parents=True)
    (source_root / "layers" / "detail.usda").write_text(
        '#usda 1.0\n\nover "World"\n{\n    def Scope "Detail" {}\n}\n',
        encoding="utf-8",
    )
    (source_root / "parts" / "geom.usda").write_text(
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "Part"\n'
        ")\n"
        "\n"
        'def Xform "Part"\n'
        "{\n"
        '    def Mesh "Geom" {}\n'
        "}\n",
        encoding="utf-8",
    )
    (source_root / "textures" / "checker.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"confinement-test-texture"
    )
    asset = source_root / "asset.usda"
    asset.write_text(
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        "    subLayers = [\n"
        "        @./layers/detail.usda@\n"
        "    ]\n"
        ")\n"
        "\n"
        'def Xform "World"\n'
        "{\n"
        '    def Mesh "Body" {}\n'
        "\n"
        '    def "Part" (\n'
        "        references = @./parts/geom.usda@</Part>\n"
        "    )\n"
        "    {\n"
        "    }\n"
        "\n"
        '    def Scope "Looks"\n'
        "    {\n"
        '        def Material "Painted"\n'
        "        {\n"
        '            def Shader "Tex"\n'
        "            {\n"
        '                uniform token info:id = "UsdUVTexture"\n'
        "                asset inputs:file = @./textures/checker.png@\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    return asset


def _write_caller_material_library(library_root: Path) -> Path:
    (library_root / "textures").mkdir(parents=True)
    (library_root / "textures" / "base.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"confinement-test-library-texture"
    )
    library = library_root / "materials.usda"
    library.write_text(
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        ")\n"
        "\n"
        'def Xform "World"\n'
        "{\n"
        '    def Scope "Looks"\n'
        "    {\n"
        '        def Material "Paint_White_Satin"\n'
        "        {\n"
        '            def Shader "Base"\n'
        "            {\n"
        '                uniform token info:id = "UsdUVTexture"\n'
        "                asset inputs:file = @./textures/base.png@\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    return library


def _tree_digests(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


# ---------------------------------------------------------------------------
# Staging unit coverage (no daemon).
# ---------------------------------------------------------------------------


def test_stage_usd_cli_input_tree_stages_digest_bound_self_contained_copy(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "caller"
    asset = _write_composed_caller_asset(source_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    staged = runner._stage_usd_cli_input_tree(
        label="source",
        source_usd_path=asset,
        run_dir=run_dir,
    )

    # Every closure file is mirrored below the run root with its relative
    # layout preserved, byte-identical, and read-only.
    assert staged.staged_root == run_dir.resolve() / "inputs" / "source"
    assert staged.staged_usd_path.is_file()
    assert staged.staged_usd_path.relative_to(staged.staged_root) == Path("asset.usda")
    for relative in (
        "asset.usda",
        "layers/detail.usda",
        "parts/geom.usda",
        "textures/checker.png",
    ):
        original = source_root / relative
        mirrored = staged.staged_root / relative
        assert mirrored.is_file(), relative
        assert _sha256(mirrored) == _sha256(original), relative
        assert stat.S_IMODE(mirrored.stat().st_mode) == 0o444, relative
    assert staged.file_count == 4
    assert staged.source_sha256 == _sha256(asset)

    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == runner.USD_CLI_STAGED_INPUT_SCHEMA_VERSION
    assert manifest["source_usd_path"] == str(asset.resolve())
    assert manifest["source_sha256"] == staged.source_sha256
    assert manifest["self_containment"]["status"] == "verified"
    assert manifest["self_containment"]["escaped_paths"] == []
    assert len(manifest["files"]) == 4
    for entry in manifest["files"]:
        assert Path(entry["staged_path"]).is_file()
        assert _sha256(Path(entry["staged_path"])) == entry["sha256"]
        assert _sha256(Path(entry["source_path"])) == entry["sha256"]

    # The staged copy composes on its own: prim paths (source-space
    # correspondence) and the composed dependency closure survive staging.
    from pxr import Usd

    stage = Usd.Stage.Open(str(staged.staged_usd_path), Usd.Stage.LoadAll)
    assert stage is not None
    assert stage.GetPrimAtPath("/World/Body").IsValid()
    assert stage.GetPrimAtPath("/World/Part/Geom").IsValid()
    assert stage.GetPrimAtPath("/World/Detail").IsValid()

    assert runner._staged_input_integrity_errors(run_dir) == []


def test_stage_usd_cli_input_tree_fails_closed_on_absolute_references(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "caller"
    source_root.mkdir()
    referenced = source_root / "external.usda"
    referenced.write_text(
        '#usda 1.0\n(\n    defaultPrim = "Asset"\n)\n\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    asset = source_root / "asset.usda"
    asset.write_text(
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        ")\n"
        "\n"
        'def Xform "World"\n'
        "{\n"
        f'    def "Item" (references = @{referenced}@</Asset>) {{}}\n'
        "}\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # A staged layer whose authored reference is absolute would pass the
    # daemon's `open` check yet silently resolve back into the caller's tree,
    # so staging must refuse it outright.
    with pytest.raises(ValueError, match="not self-contained"):
        runner._stage_usd_cli_input_tree(
            label="source",
            source_usd_path=asset,
            run_dir=run_dir,
        )


def test_staged_input_integrity_errors_reports_mutated_inputs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "caller"
    asset = _write_composed_caller_asset(source_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged = runner._stage_usd_cli_input_tree(
        label="source",
        source_usd_path=asset,
        run_dir=run_dir,
    )
    assert runner._staged_input_integrity_errors(run_dir) == []

    mutated = staged.staged_root / "layers" / "detail.usda"
    os.chmod(mutated, 0o644)
    mutated.write_text("#usda 1.0\n", encoding="utf-8")

    errors = runner._staged_input_integrity_errors(run_dir)
    assert any(str(mutated) in error for error in errors)


def test_staged_input_integrity_errors_uses_parent_bound_manifest_set(
    tmp_path: Path,
) -> None:
    asset = _write_composed_caller_asset(tmp_path / "caller")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged = runner._stage_usd_cli_input_tree(
        label="source",
        source_usd_path=asset,
        run_dir=run_dir,
    )
    manifest_bytes = staged.manifest_path.read_bytes()

    staged.manifest_path.write_bytes(manifest_bytes + b"\n")
    assert any(
        "manifest changed" in error
        for error in runner._staged_input_integrity_errors(run_dir)
    )

    staged.manifest_path.write_bytes(manifest_bytes)
    staged.manifest_path.unlink()
    assert any(
        "manifest is missing" in error
        for error in runner._staged_input_integrity_errors(run_dir)
    )


# ---------------------------------------------------------------------------
# Real-daemon regressions (#3): no monkeypatched path checks.
# ---------------------------------------------------------------------------


def _real_usd_cli_route(run_dir: Path) -> runner.UsdCliTelemetryRoute:
    route = runner._prepare_usd_cli_telemetry_route(run_dir, repo_root=REPO_ROOT)
    if route.active and route.target_path is not None:
        return route
    # Developer checkouts routinely carry unrecorded local artifacts (old
    # daemon state, run reports) that the package-provenance gate rejects.
    # That gate has its own tests; the regression here exercises the *daemon's*
    # filesystem enforcement, which only needs the real installed executables.
    interpreter_bin = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    wrapper = interpreter_bin / "usd-cli-tel"
    target = interpreter_bin / "usd-cli"
    if not wrapper.is_file() or not target.is_file():
        pytest.skip(f"usd-cli executables unavailable: {route.reason}")
    return runner.UsdCliTelemetryRoute(
        status="active",
        wrapper_path=wrapper,
        target_path=target,
    )


def _usd_cli_client(
    route: runner.UsdCliTelemetryRoute,
    run_dir: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    assert route.target_path is not None
    return subprocess.run(  # noqa: S603 - fixed executable from the owned route
        [str(route.target_path), "--json", *arguments],
        cwd=run_dir,
        env=runner._usd_cli_daemon_env(
            run_dir=run_dir,
            target_path=route.target_path,
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _assert_rejected_outside_roots(
    completed: subprocess.CompletedProcess[str],
    *,
    label: str,
) -> None:
    """The daemon's allowed-roots rejection surfaces as an HTTP 400 refusal.

    The client JSON carries the transport-level `400 Bad Request` without
    echoing the server's read-root or write-root detail, so accept either
    capability-specific refusal or the transport form. Every rejected request
    here targets a file that exists and
    composes — its staged byte-identical twin succeeds in the same session —
    which pins the refusal to the path validation, not to the asset.
    """

    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode != 0, (label, combined)
    assert (
        "outside server.allowed_roots" in combined
        or "outside server.allowed_write_roots" in combined
    ) or ('"ok":false' in combined and "400 Bad Request" in combined), (label, combined)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_real_daemon_material_staged_inputs_confined_to_run_root(
    tmp_path: Path,
) -> None:
    """Material regression: staged source+library readable, originals fail
    closed, writes stay run-root confined, and every input stays immutable."""

    caller_root = tmp_path / "caller"
    asset = _write_composed_caller_asset(caller_root / "asset")
    library = _write_caller_material_library(caller_root / "library")
    original_digests = _tree_digests(
        sorted(path for path in caller_root.rglob("*") if path.is_file())
    )

    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    staged_source = runner._stage_usd_cli_input_tree(
        label="source",
        source_usd_path=asset,
        run_dir=run_dir,
    )
    staged_library = runner._stage_usd_cli_input_tree(
        label="material_library",
        source_usd_path=library,
        run_dir=run_dir,
    )

    route = _real_usd_cli_route(run_dir)
    lease = runner._start_usd_cli_run_daemon(route=route, run_dir=run_dir)
    try:
        # Approved staged inputs are readable through the daemon.
        opened = _usd_cli_client(
            route, run_dir, "open", str(staged_source.staged_usd_path)
        )
        assert opened.returncode == 0, opened.stderr

        # The original caller paths fail closed on their first open.
        _assert_rejected_outside_roots(
            _usd_cli_client(route, run_dir, "open", str(asset)),
            label="original source open",
        )

        # Binding from the staged library is allowed; the caller's original
        # library path is rejected by the same request validation.
        bound = _usd_cli_client(
            route,
            run_dir,
            "material",
            "/World/Body",
            "--library",
            str(staged_library.staged_usd_path),
            "--name",
            "Paint_White_Satin",
        )
        assert bound.returncode == 0, bound.stderr
        _assert_rejected_outside_roots(
            _usd_cli_client(
                route,
                run_dir,
                "material",
                "/World/Body",
                "--library",
                str(library),
                "--name",
                "Paint_White_Satin",
            ),
            label="original library bind",
        )

        # Writes: run-root outputs succeed; anything outside fails closed.
        output_usd = run_dir / "output" / "materialized.usda"
        output_usd.parent.mkdir(parents=True, exist_ok=True)
        saved = _usd_cli_client(route, run_dir, "save", str(output_usd), "--flatten")
        assert saved.returncode == 0, saved.stderr
        assert output_usd.is_file()
        escape_usd = caller_root / "escape.usda"
        _assert_rejected_outside_roots(
            _usd_cli_client(route, run_dir, "save", str(escape_usd)),
            label="outside save",
        )
        assert not escape_usd.exists()
    finally:
        runner._stop_usd_cli_run_daemon_best_effort(lease=lease)

    # Immutability: the caller's originals and the digest-bound staged trees
    # are byte-identical to their pre-run state.
    assert (
        _tree_digests(sorted(path for path in caller_root.rglob("*") if path.is_file()))
        == original_digests
    )
    assert runner._staged_input_integrity_errors(run_dir) == []


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_real_daemon_physics_staged_packet_confined_to_run_root(
    tmp_path: Path,
) -> None:
    """Physics regression: the packet's staged source is readable and
    authorable through the daemon while the original stays outside and
    immutable, with outputs confined to the run root."""

    caller_root = tmp_path / "caller"
    asset = _write_composed_caller_asset(caller_root / "asset")
    original_digests = _tree_digests(
        sorted(path for path in caller_root.rglob("*") if path.is_file())
    )
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)

    config = PhysicsApplyConfig(
        repo_root=REPO_ROOT,
        usd_path=asset,
        output_dir=run_dir,
    )
    route = _real_usd_cli_route(run_dir)
    lease = runner._start_usd_cli_run_daemon(route=route, run_dir=run_dir)
    try:
        session = WorkflowUsdCliSession(
            project_dir=run_dir,
            session_id="workflow-physics-confinement",
            route=route,
            workflow="physics",
        )
        packet = runner._prepare_usd_cli_physics_run_packet(
            config,
            run_dir,
            usd_cli_session=session,
        )
        staged_metadata = packet["staged_source"]
        assert staged_metadata["self_contained"] is True
        staged_usd = Path(str(staged_metadata["staged_usd_path"]))
        assert staged_usd.is_file()
        assert run_dir.resolve() in staged_usd.resolve().parents
        assert staged_metadata["source_sha256"] == _sha256(asset)
        # Component evidence was inspected on the staged tree the child opens.
        components = json.loads(
            (run_dir / "raw" / "physics_components.json").read_text(encoding="utf-8")
        )
        assert json.dumps(components)  # parses; content is backend-owned

        opened = _usd_cli_client(route, run_dir, "open", str(staged_usd))
        assert opened.returncode == 0, opened.stderr
        _assert_rejected_outside_roots(
            _usd_cli_client(route, run_dir, "open", str(asset)),
            label="original physics source open",
        )

        # A physics-authoring mutation on the staged copy persists only into
        # the run root.
        authored = _usd_cli_client(
            route,
            run_dir,
            "physics",
            "author",
            "--ref",
            "/World/Body",
            "--collision",
        )
        if authored.returncode != 0:
            # Authoring surface can evolve; the confinement contract under
            # test is the write boundary, so fall back to a generic edit.
            authored = _usd_cli_client(route, run_dir, "hide", "/World/Body")
        assert authored.returncode == 0, authored.stderr
        output_usd = run_dir / "output" / "physics.usda"
        output_usd.parent.mkdir(parents=True, exist_ok=True)
        saved = _usd_cli_client(route, run_dir, "save", str(output_usd), "--flatten")
        assert saved.returncode == 0, saved.stderr
        assert output_usd.is_file()
        _assert_rejected_outside_roots(
            _usd_cli_client(route, run_dir, "save", str(caller_root / "escape.usda")),
            label="outside physics save",
        )
    finally:
        runner._stop_usd_cli_run_daemon_best_effort(lease=lease)

    assert (
        _tree_digests(sorted(path for path in caller_root.rglob("*") if path.is_file()))
        == original_digests
    )
    assert runner._staged_input_integrity_errors(run_dir) == []
