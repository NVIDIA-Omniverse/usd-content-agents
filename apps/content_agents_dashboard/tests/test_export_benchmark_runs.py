# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dashboard benchmark-run exporter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "export_benchmark_runs.py"
_spec = importlib.util.spec_from_file_location("export_benchmark_runs", _MODULE_PATH)
assert _spec and _spec.loader
export_benchmark_runs = importlib.util.module_from_spec(_spec)
sys.modules["export_benchmark_runs"] = export_benchmark_runs
_spec.loader.exec_module(export_benchmark_runs)

export_run = export_benchmark_runs.export_run
discover_runs = export_benchmark_runs.discover_runs
UnsafeSourcePathError = export_benchmark_runs.UnsafeSourcePathError


def _write_run(
    root: Path, *, run_id: str = "run-1", asset_overrides: dict | None = None
):
    run = root / "mesh_segmentation" / run_id
    (run / "bundle").mkdir(parents=True)
    (run / "raw").mkdir(parents=True)
    render = run / "raw" / "final.png"
    render.write_bytes(b"\x89PNG-not-really")
    (run / "benchmark_run.json").write_text(
        json.dumps({"workflow": "mesh_segmentation", "run_id": run_id}),
        encoding="utf-8",
    )
    asset = {
        "asset_id": "asset-a",
        "status": "completed",
        "renders": {"final": str(render)},
        "references": [],
        "local_artifacts": {},
    }
    asset.update(asset_overrides or {})
    (run / "bundle" / "run.json").write_text(
        json.dumps(
            {"workflow": "mesh_segmentation", "run_id": run_id, "assets": [asset]}
        ),
        encoding="utf-8",
    )
    return run


def test_export_copies_only_linked_files(tmp_path: Path) -> None:
    source_root = tmp_path / "artifacts"
    run = _write_run(source_root)
    # A large intermediate the export must not carry.
    (run / "raw" / "scratch.bin").write_bytes(b"x" * 64)

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    assert exported["assets"][0]["renders"]["final"]
    copied = {path.name for path in destination.rglob("*") if path.is_file()}
    assert "scratch.bin" not in copied


def test_export_preserves_reference_provenance(tmp_path: Path) -> None:
    """A digest-verified provenance record travels with the exported image.

    Dropping it demotes an OVRTX-validated reference to an unverifiable
    diagnostic preview in the hosted dashboard; conversely a provenance
    whose image digest no longer matches the exported bytes must be omitted
    so a stale claim cannot publish as validated OVRTX evidence.
    """
    source_root = tmp_path / "artifacts"
    image_bytes = b"\x89PNG-not-really"
    provenance = {
        "renderer": "ovrtx",
        # render_metadata itself carries producing-machine paths
        # (stage_preparation[].usd_path, asset_base_dir); only the path-free
        # remainder may travel.
        "render_metadata": {
            "image_width": 1024,
            "asset_base_dir": "/home/someone/.data/assets",
            "stage_preparation": [
                {
                    "usd_path": "/home/someone/run/input.usdc",
                    "usd_sha256": "c" * 64,
                    "up_axis": "Z",
                }
            ],
        },
        "source_usd_sha256": "a" * 64,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        # Producing-machine absolute paths that must NOT survive the export.
        "render_response": {"results": [{"usd_path": "/home/someone/run/input.usdc"}]},
    }
    stale_provenance = {**provenance, "image_sha256": "b" * 64}
    reference = source_root / "mesh_segmentation" / "run-1" / "raw" / "reference.png"
    asset_overrides = {
        "references": [
            {
                "label": "reference",
                "path": str(reference),
                "provenance": provenance,
            },
            {
                "label": "stale",
                "path": str(reference),
                "provenance": stale_provenance,
            },
            {"label": "bare", "path": str(reference)},
        ]
    }
    run = _write_run(source_root, asset_overrides=asset_overrides)
    reference.write_bytes(image_bytes)

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    exported_references = exported["assets"][0]["references"]
    # Shareable projection: digest-bound fields and path-free metadata
    # travel; the render_response and render_metadata's producing-machine
    # paths must not, and the projection is bound to the exact record by
    # its canonical-JSON digest.
    exact_digest = hashlib.sha256(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    exact_record_path = "assets/asset-a/references/01_reference.provenance.exact.json"
    assert exported_references[0]["provenance"] == {
        "renderer": "ovrtx",
        "render_metadata": {
            "image_width": 1024,
            "stage_preparation": [{"usd_sha256": "c" * 64, "up_axis": "Z"}],
        },
        "source_usd_sha256": "a" * 64,
        "image_sha256": provenance["image_sha256"],
        "exact_record_sha256": exact_digest,
        "exact_record_path": exact_record_path,
    }
    assert "/home/someone" not in json.dumps(exported_references[0])
    # The exact verbatim record travels in an explicitly-labeled sidecar
    # whose canonical bytes hash to exact_record_sha256, so the exact OVRTX
    # metadata stays recoverable even when the source run is unavailable.
    sidecar = destination / "bundle" / exact_record_path
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == exact_digest
    assert json.loads(sidecar.read_text(encoding="utf-8")) == provenance
    assert "provenance" not in exported_references[1]
    assert "provenance" not in exported_references[2]


def test_export_sanitizes_suite_result_provenance(tmp_path: Path) -> None:
    """The scorer stores the verbatim reference provenance in suite_result.

    A byte-for-byte copy publishes the render response's absolute image/USD
    paths — the exact producing-machine paths the exporter exists to remove.
    The projected record keeps the digests and path-free metadata and stays
    bound to the exact record by its canonical-JSON digest.
    """
    run = _write_run(tmp_path / "artifacts")
    provenance = {
        "renderer": "ovrtx",
        "render_metadata": {
            "image_width": 1024,
            "asset_base_dir": "/home/someone/.data/assets",
            "stage_preparation": [
                {"usd_path": "/home/someone/run/input.usdc", "usd_sha256": "c" * 64}
            ],
        },
        "source_usd_sha256": "a" * 64,
        "image_sha256": "b" * 64,
        "render_response": {"results": [{"usd_path": "/home/someone/run/x.usdc"}]},
    }
    score_dir = run / "score"
    score_dir.mkdir()
    (score_dir / "suite_result.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "asset_id": "asset-a",
                        "metrics": {"reference_provenance": provenance},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "score" / "suite_result.json").read_text())
    projected = exported["cases"][0]["metrics"]["reference_provenance"]
    assert "/home/someone" not in json.dumps(exported)
    assert projected["source_usd_sha256"] == "a" * 64
    assert projected["render_metadata"]["stage_preparation"] == [
        {"usd_sha256": "c" * 64}
    ]
    assert (
        projected["exact_record_sha256"]
        == hashlib.sha256(
            json.dumps(
                provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()
    )


def test_export_refuses_an_absolute_path_outside_the_run(tmp_path: Path) -> None:
    """A shared export must never carry a file the bundle merely names.

    The exporter's output is meant for hosting, so an absolute path such as
    /etc/passwd in a malformed or hostile bundle would otherwise be copied
    verbatim into a shareable artifact.
    """

    secret = tmp_path / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    run = _write_run(
        tmp_path / "artifacts",
        asset_overrides={"renders": {"final": str(secret)}},
    )

    with pytest.raises(UnsafeSourcePathError, match="outside the permitted roots"):
        export_run(run, tmp_path / "out")


def test_export_refuses_a_relative_escape(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    run = _write_run(
        tmp_path / "artifacts",
        asset_overrides={"renders": {"final": "../../../secret.txt"}},
    )

    with pytest.raises(UnsafeSourcePathError):
        export_run(run, tmp_path / "out")


def test_export_allows_an_explicitly_permitted_dataset_root(tmp_path: Path) -> None:
    """source_usd legitimately lives outside the run, so it must stay exportable."""

    dataset = tmp_path / "data"
    dataset.mkdir()
    source_usd = dataset / "source.usdc"
    source_usd.write_bytes(b"usd")
    run = _write_run(
        tmp_path / "artifacts",
        asset_overrides={"source_usd": str(source_usd)},
    )

    destination = export_run(run, tmp_path / "out", allowed_roots=[dataset])

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    assert exported["assets"][0]["source_usd"] == "assets/asset-a/input/source.usdc"
    assert (destination / "bundle" / "assets/asset-a/input/source.usdc").is_file()


def test_export_refuses_to_overwrite_an_existing_export(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "artifacts")
    export_run(run, tmp_path / "out")

    with pytest.raises(FileExistsError):
        export_run(run, tmp_path / "out")


def test_discover_runs_filters_by_workflow_and_run_id(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _write_run(root, run_id="run-1")
    _write_run(root, run_id="run-2")

    assert len(discover_runs(root, workflow="mesh_segmentation", run_id=None)) == 2
    assert len(discover_runs(root, workflow="material-agentic", run_id=None)) == 0
    only = discover_runs(root, workflow=None, run_id="run-2")
    assert [path.name for path in only] == ["run-2"]


def test_export_records_where_a_sanitized_asset_landed(tmp_path: Path) -> None:
    """`asset_id` stays verbatim, so the export must say where files went.

    The directory name is sanitized but `asset_id` is the join key with the
    scored cases and cannot be rewritten. Without an explicit pointer the
    dashboard would have to reimplement this sanitization and could drift from
    it, requesting `assets/part/a/metrics.json` for a directory named
    `assets/part_a`.
    """

    run = _write_run(
        tmp_path / "artifacts",
        asset_overrides={"asset_id": "part/a", "renders": {}},
    )

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    asset = exported["assets"][0]
    assert asset["asset_id"] == "part/a"
    assert asset["export_root"] == "assets/part_a"
    assert (destination / "bundle" / asset["export_root"] / "metrics.json").is_file()


def test_export_keeps_the_source_usd_extension(tmp_path: Path) -> None:
    """USD picks its format plugin by extension, and the copy is byte-for-byte.

    The material workflow records `output_usd` as an ASCII `.usda` layer.
    Forcing `.usdc` on it produced a file nothing could open, published by the
    dashboard as the run's "Output USD" download.
    """

    run = _write_run(tmp_path / "artifacts")
    ascii_layer = run / "raw" / "output.usda"
    ascii_layer.write_text("#usda 1.0\n", encoding="utf-8")

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["output_usd"] = str(ascii_layer)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    assert exported["assets"][0]["output_usd"].endswith("/segmented.usda")
    copied = destination / "bundle" / exported["assets"][0]["output_usd"]
    assert copied.read_text(encoding="utf-8").startswith("#usda")


def test_export_copies_the_review_document(tmp_path: Path) -> None:
    """The dashboard fetches every review with Promise.all.

    A retained but uncopied review_path 404s and rejects the whole exported
    run, not just the asset it belongs to.
    """

    run = _write_run(tmp_path / "artifacts")
    review = run / "bundle" / "review.json"
    review.write_text(json.dumps({"asset_id": "asset-a"}), encoding="utf-8")

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["review_path"] = "review.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    review_path = exported["assets"][0]["review_path"]
    assert (destination / "bundle" / review_path).is_file()


def test_export_clears_a_review_path_it_cannot_copy(tmp_path: Path) -> None:
    """A dangling pointer must not survive into the export."""

    run = _write_run(tmp_path / "artifacts")
    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["review_path"] = "missing-review.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    assert "review_path" not in exported["assets"][0]


def test_main_skips_an_unexportable_run_and_keeps_going(tmp_path: Path, capsys) -> None:
    """One bad run must not abandon the batch or block its own retry.

    The traversal guard turned a previously silent copy into a hard error, so
    without this a single malformed bundle aborts every remaining run and
    leaves a partial directory that the FileExistsError guard then refuses to
    overwrite.
    """

    root = tmp_path / "artifacts"
    secret = tmp_path / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    _write_run(root, run_id="bad", asset_overrides={"renders": {"final": str(secret)}})
    _write_run(root, run_id="good")
    out = tmp_path / "out"

    argv = [
        "export_benchmark_runs.py",
        "--artifact-root",
        str(root),
        "--output-root",
        str(out),
    ]
    original = sys.argv
    sys.argv = argv
    try:
        code = export_benchmark_runs.main()
    finally:
        sys.argv = original

    assert code == 1
    assert (out / "mesh_segmentation" / "good").is_dir()
    # The failed run left nothing behind, so a retry is not blocked.
    assert not (out / "mesh_segmentation" / "bad").exists()
    assert "skipped" in capsys.readouterr().err
    # Nor any staging debris.
    assert [path.name for path in (out / "mesh_segmentation").iterdir()] == ["good"]


def test_export_disambiguates_colliding_asset_roots(tmp_path: Path) -> None:
    """Two ids that sanitize alike must not share one directory.

    `part/a` and `part_a` both fold to `part_a`, so the later asset silently
    overwrote the earlier one's renders, reports and metrics while both records
    advertised the same export_root.
    """

    run = _write_run(tmp_path / "artifacts")
    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    first = dict(bundle["assets"][0])
    first["asset_id"] = "part/a"
    second = dict(bundle["assets"][0])
    second["asset_id"] = "part_a"
    bundle["assets"] = [first, second]
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    roots = [asset["export_root"] for asset in exported["assets"]]
    assert len(set(roots)) == 2, roots
    for root in roots:
        assert (destination / "bundle" / root / "metrics.json").is_file()


def test_rerunning_into_an_existing_root_preserves_the_prior_export(
    tmp_path: Path, capsys
) -> None:
    """A refused re-export must never destroy the export it refused to replace.

    FileExistsError is an OSError, so the batch handler caught the "already
    exists" guard and then deleted the complete, previously successful export
    it was protecting.
    """

    root = tmp_path / "artifacts"
    _write_run(root, run_id="run-1")
    out = tmp_path / "out"

    argv = [
        "export_benchmark_runs.py",
        "--artifact-root",
        str(root),
        "--output-root",
        str(out),
    ]
    original = sys.argv
    sys.argv = argv
    try:
        assert export_benchmark_runs.main() == 0
        exported = out / "mesh_segmentation" / "run-1"
        marker = json.loads((exported / "bundle" / "run.json").read_text())

        # Second pass against the same root: refused, and non-destructive.
        assert export_benchmark_runs.main() == 1
    finally:
        sys.argv = original

    assert exported.is_dir()
    assert json.loads((exported / "bundle" / "run.json").read_text()) == marker
    assert "already exists" in capsys.readouterr().err


def test_export_disambiguates_colliding_render_view_names(tmp_path: Path) -> None:
    """Agent-authored view names can fold to one segment.

    `front oblique` and `front-oblique` both sanitize to `front_oblique`, so
    the second copy overwrote the first and both entries pointed at the
    surviving image -- one camera shown under two labels.
    """

    run = _write_run(tmp_path / "artifacts")
    second = run / "raw" / "second.png"
    second.write_bytes(b"\x89PNG-second")

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["renders"] = {
        "front oblique": str(run / "raw" / "final.png"),
        "front-oblique": str(second),
    }
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    paths = list(exported["assets"][0]["renders"].values())
    assert len(set(paths)) == 2, paths
    contents = {(destination / "bundle" / path).read_bytes() for path in paths}
    assert contents == {b"\x89PNG-not-really", b"\x89PNG-second"}


def test_export_refuses_a_symlink_that_escapes_the_run(tmp_path: Path) -> None:
    """A symlink inside the run must not be a way out of it.

    The guard resolves before comparing, so following the link is exactly what
    makes it safe -- the resolved target lands outside every permitted root and
    is refused. Pinned here because "resolve() follows symlinks" reads like a
    weakness and is the opposite.
    """

    secret = tmp_path / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    run = _write_run(tmp_path / "artifacts")
    escape = run / "raw" / "innocent.png"
    escape.symlink_to(secret)

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["renders"] = {"final": str(escape)}
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(UnsafeSourcePathError, match="outside the permitted roots"):
        export_run(run, tmp_path / "out")


def test_export_allows_a_symlink_that_stays_inside_the_run(tmp_path: Path) -> None:
    """The guard bounds where files come from, not how they are referenced."""

    run = _write_run(tmp_path / "artifacts")
    real = run / "raw" / "final.png"
    link = run / "raw" / "aliased.png"
    link.symlink_to(real)

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["renders"] = {"final": str(link)}
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    assert exported["assets"][0]["renders"]["final"]


def test_export_copies_render_manifest_dependencies(tmp_path: Path) -> None:
    """Provenance links must not dangle in a shared export.

    The manifest references per-view camera and response files by absolute
    path. Copying the manifest alone left every one of those links pointing at
    a path that does not exist outside the machine that produced the run.
    """

    run = _write_run(tmp_path / "artifacts")
    raw = run / "raw"
    camera = raw / "camera.json"
    camera.write_text(json.dumps({"fov": 30}), encoding="utf-8")
    response = raw / "response.json"
    response.write_text(json.dumps({"renderer": "ovrtx"}), encoding="utf-8")
    (raw / "depth.png").write_bytes(b"\x89PNG-depth")
    (raw / "depth.npy").write_bytes(b"\x93NUMPY-depth")
    manifest = raw / "render_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "renders": [
                    {
                        "name": "front",
                        "image": str(raw / "final.png"),
                        "camera": str(camera),
                        "response": str(response),
                        # The shape `render_mesh_evidence.py` actually writes:
                        # a PNG rendering under `preview` and the
                        # full-resolution AOV under `raw`.
                        "channels": {
                            "linear_depth": {
                                "aov": "DistanceToImagePlaneSD",
                                "preview": str(raw / "depth.png"),
                                "raw": str(raw / "depth.npy"),
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["local_artifacts"] = {"render_manifest": str(manifest)}
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    manifest_rel = exported["assets"][0]["local_artifacts"]["render_manifest"]
    manifest_path = destination / "bundle" / manifest_rel
    copied = json.loads(manifest_path.read_text())
    record = copied["renders"][0]

    # Links resolve from the manifest's own directory: it lands beside the
    # asset, so a bundle-relative link would not resolve from where a reader
    # opens it. `image` counts too -- omitting it left the export pointing at
    # the producing machine.
    for field in ("image", "camera", "response"):
        assert not Path(record[field]).is_absolute(), field
        assert (manifest_path.parent / record[field]).resolve().is_file(), field
    # `raw` is the full-resolution AOV the render produced; `preview` is only
    # its PNG rendering. Copying the preview alone left every normal and depth
    # buffer pointing at the producing machine, so the hosted export could not
    # reproduce the evidence it advertises.
    channel = copied["renders"][0]["channels"]["linear_depth"]
    for field in ("preview", "raw"):
        assert not Path(channel[field]).is_absolute(), field
        assert (manifest_path.parent / channel[field]).resolve().is_file(), field
    assert channel["aov"] == "DistanceToImagePlaneSD", "untouched fields must survive"


def test_export_rewrites_the_manifest_scene_to_the_copied_usd(tmp_path: Path) -> None:
    """The manifest names the scene it rendered by absolute host path.

    Left alone, an export advertises final evidence whose subject cannot be
    opened. The digest must survive so the binding still verifies.
    """

    run = _write_run(tmp_path / "artifacts")
    raw = run / "raw"
    final_usd = raw / "segmented.usdc"
    final_usd.write_bytes(b"usd-final")
    manifest = raw / "render_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scene": str(final_usd),
                "scene_sha256": "a" * 64,
                "renders": [{"name": "front", "image": str(raw / "final.png")}],
            }
        ),
        encoding="utf-8",
    )

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["output_usd"] = str(final_usd)
    bundle["assets"][0]["local_artifacts"] = {"render_manifest": str(manifest)}
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    manifest_rel = exported["assets"][0]["local_artifacts"]["render_manifest"]
    manifest_path = destination / "bundle" / manifest_rel
    copied = json.loads(manifest_path.read_text())

    assert not Path(copied["scene"]).is_absolute()
    assert (manifest_path.parent / copied["scene"]).resolve().is_file()
    assert copied["scene_sha256"] == "a" * 64, "the digest must not be rewritten"


def test_export_copies_usd_sublayer_packages(tmp_path: Path) -> None:
    """A root layer that sublayers siblings must arrive with them.

    The material workflow writes output.usda referencing material_library.usdz
    and source_scene.usdz beside it; copying the root alone produced an
    advertised Output USD that cannot open.
    """

    pytest.importorskip("pxr")
    from pxr import Usd

    run = _write_run(tmp_path / "artifacts")
    raw = run / "raw"
    Usd.Stage.CreateNew(str(raw / "material_library.usdc")).Save()
    root_path = raw / "output.usdc"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.GetRootLayer().subLayerPaths.append("material_library.usdc")
    root_stage.Save()

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["output_usd"] = str(root_path)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    copied_root = destination / "bundle" / exported["assets"][0]["output_usd"]
    assert copied_root.is_file()
    assert (copied_root.parent / "material_library.usdc").is_file()


def test_export_copies_usd_dependency_sidecars(tmp_path: Path) -> None:
    """The localized "<name>_assets/" sidecar must travel with the root layer.

    Both physics collectors localize texture/composition dependencies into a
    sibling "<name>_assets/" directory the root layer references relatively;
    copying the root alone hosts an Output USD with dangling references. The
    sidecar keeps its source name because the exporter renames the root layer
    while the references inside it still name the original directory.
    """

    run = _write_run(tmp_path / "artifacts")
    raw = run / "raw"
    root_path = raw / "physics.usdc"
    root_path.write_bytes(b"binary-usd-root")
    sidecar = raw / "physics.usdc_assets" / "textures"
    sidecar.mkdir(parents=True)
    (sidecar / "albedo.png").write_bytes(b"png-bytes")
    # A symlinked member must not be dereferenced into the export.
    (raw / "physics.usdc_assets" / "escape").symlink_to(tmp_path / "outside")

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["output_usd"] = str(root_path)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    copied_root = destination / "bundle" / exported["assets"][0]["output_usd"]
    assert copied_root.is_file()
    copied_sidecar = copied_root.parent / "physics.usdc_assets"
    assert (copied_sidecar / "textures" / "albedo.png").read_bytes() == b"png-bytes"
    assert not (copied_sidecar / "escape").exists()


def test_export_refuses_a_sublayer_that_escapes_the_staged_asset(
    tmp_path: Path,
) -> None:
    """Bounding the source is not enough; the destination must be bounded too.

    A layer stored deeply under an allowed root can name `../../..` and still
    resolve inside that root, while the same components walk the *destination*
    out of the staging tree -- overwriting another staged asset, or a file
    outside the export entirely.
    """

    pytest.importorskip("pxr")
    from pxr import Usd

    run = _write_run(tmp_path / "artifacts")
    deep = run / "raw" / "a" / "b" / "c"
    deep.mkdir(parents=True)
    # Sits inside the run (so the source check passes) but above the layer.
    escaped = run / "raw" / "escaped.usdc"
    Usd.Stage.CreateNew(str(escaped)).Save()
    root_path = deep / "output.usdc"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.GetRootLayer().subLayerPaths.append("../../../escaped.usdc")
    root_stage.Save()

    bundle_path = run / "bundle" / "run.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["assets"][0]["output_usd"] = str(root_path)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    destination = export_run(run, tmp_path / "out")

    exported = json.loads((destination / "bundle" / "run.json").read_text())
    copied_root = destination / "bundle" / exported["assets"][0]["output_usd"]
    assert copied_root.is_file()
    # Nothing written above the asset's own output directory.
    strays = [
        path
        for path in (tmp_path / "out").rglob("escaped.usdc")
        if path.parent != copied_root.parent
    ]
    assert strays == [], strays


def test_export_directory_is_readable_by_other_accounts(tmp_path: Path) -> None:
    """Staging is 0o700 and `replace` promotes it verbatim.

    An export served by a different service account would then 403 on every
    asset, report and manifest it advertises.
    """

    run = _write_run(tmp_path / "artifacts")
    # The artifacts this copies are written owner-only at the source: the
    # mesh-segmentation runner's `_write_json` defaults to 0o600 and
    # `content_benchmark.util.write_json` renames a NamedTemporaryFile into
    # place. `shutil.copy2` reproduces that mode, so chmoding the staging root
    # alone left every copied file unreadable -- including the
    # `benchmark_run.json` the dashboard fetches first, whose 403 drops the
    # whole run.
    for path in run.rglob("*"):
        if path.is_file():
            path.chmod(0o600)

    destination = export_run(run, tmp_path / "out")

    assert destination.stat().st_mode & 0o055 == 0o055
    copied = [path for path in destination.rglob("*") if path.is_file()]
    assert copied, "export produced no files to check"
    unreadable = [
        str(path.relative_to(destination))
        for path in copied
        if path.stat().st_mode & 0o044 != 0o044
    ]
    assert unreadable == [], f"copied files are not world/group readable: {unreadable}"
    directories = [path for path in destination.rglob("*") if path.is_dir()]
    untraversable = [
        str(path.relative_to(destination))
        for path in directories
        if path.stat().st_mode & 0o055 != 0o055
    ]
    assert untraversable == [], (
        f"copied directories are not traversable: {untraversable}"
    )


def test_reexport_preserves_original_exact_record(tmp_path: Path) -> None:
    """Re-exporting an already exported run must not promote the sanitized
    projection to the 'exact' record: the original exact sidecar is
    recovered and shipped, keeping the exact OVRTX metadata recoverable
    across export generations."""
    source_root = tmp_path / "artifacts"
    image_bytes = b"\x89PNG-not-really"
    provenance = {
        "renderer": "ovrtx",
        "render_metadata": {
            "image_width": 1024,
            "asset_base_dir": "/home/someone/.data/assets",
            "stage_preparation": [
                {
                    "usd_path": "/home/someone/run/input.usdc",
                    "usd_sha256": "c" * 64,
                    "up_axis": "Z",
                }
            ],
        },
        "source_usd_sha256": "a" * 64,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "render_response": {"results": [{"usd_path": "/home/someone/run/input.usdc"}]},
    }
    reference = source_root / "mesh_segmentation" / "run-1" / "raw" / "reference.png"
    run = _write_run(
        source_root,
        asset_overrides={
            "references": [
                {"label": "reference", "path": str(reference), "provenance": provenance}
            ]
        },
    )
    reference.write_bytes(image_bytes)

    first = export_run(run, tmp_path / "out-1")
    second = export_run(first, tmp_path / "out-2")

    exported = json.loads((second / "bundle" / "run.json").read_text())
    record = exported["assets"][0]["references"][0]
    exact_digest = hashlib.sha256(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    # The projection still binds to the ORIGINAL exact record...
    assert record["provenance"]["exact_record_sha256"] == exact_digest
    # ...and the re-exported sidecar carries the original verbatim record,
    # not the first export's projection.
    sidecar = second / "bundle" / record["provenance"]["exact_record_path"]
    assert json.loads(sidecar.read_text(encoding="utf-8")) == provenance


def test_export_copies_report_exact_provenance_sidecar(tmp_path: Path) -> None:
    """The report HTML documents its exact records at the sibling filename;
    the export must carry that sidecar in lockstep or the hosted report
    404s on its own contract."""
    run = _write_run(tmp_path / "artifacts")
    (run / "score").mkdir(exist_ok=True)
    (run / "score" / "report.html").write_text("<html></html>", encoding="utf-8")
    (run / "score" / "report.html.provenance.exact.json").write_text(
        "{}", encoding="utf-8"
    )

    destination = export_run(run, tmp_path / "out")

    assert (destination / "score" / "report.html.provenance.exact.json").is_file()


def test_reexport_rejects_tampered_exact_sidecar(tmp_path: Path) -> None:
    """A replaced sidecar whose canonical digest no longer matches the
    projection's exact_record_sha256 must not be re-signed as canonical:
    the re-export ships the image without provenance (fail closed)."""
    source_root = tmp_path / "artifacts"
    image_bytes = b"\x89PNG-not-really"
    provenance = {
        "renderer": "ovrtx",
        "render_metadata": {"image_width": 1024},
        "source_usd_sha256": "a" * 64,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    reference = source_root / "mesh_segmentation" / "run-1" / "raw" / "reference.png"
    run = _write_run(
        source_root,
        asset_overrides={
            "references": [
                {"label": "reference", "path": str(reference), "provenance": provenance}
            ]
        },
    )
    reference.write_bytes(image_bytes)

    first = export_run(run, tmp_path / "out-1")
    exported = json.loads((first / "bundle" / "run.json").read_text())
    sidecar_rel = exported["assets"][0]["references"][0]["provenance"][
        "exact_record_path"
    ]
    # Replace the sidecar with a different (but image-digest-consistent)
    # record; its canonical digest no longer matches the projection binding.
    (first / "bundle" / sidecar_rel).write_text(
        json.dumps({**provenance, "renderer": "tampered"}), encoding="utf-8"
    )

    second = export_run(first, tmp_path / "out-2")

    record = json.loads((second / "bundle" / "run.json").read_text())["assets"][0][
        "references"
    ][0]
    assert "provenance" not in record


def test_sanitize_strips_paths_despite_binding_marker(tmp_path: Path) -> None:
    """A binding marker is data, not proof: a score record carrying
    exact_record_sha256 ALONGSIDE producing-machine fields is still
    projected, and an unverifiable binding is dropped instead of
    re-signed."""
    from apps.content_agents_dashboard.export_benchmark_runs import (
        _sanitize_suite_result_provenance,
    )

    tampered = {
        "renderer": "ovrtx",
        "render_metadata": {
            "image_width": 1024,
            "asset_base_dir": "/home/producer/.data/assets",
        },
        "source_usd_sha256": "a" * 64,
        "image_sha256": "b" * 64,
        "render_response": {"results": [{"usd_path": "/home/producer/in.usdc"}]},
        "exact_record_sha256": "c" * 64,
        "exact_record_path": "assets/x/references/01_x.provenance.exact.json",
    }
    score = tmp_path / "suite_result.json"
    score.write_text(
        json.dumps({"cases": [{"metrics": {"reference_provenance": tampered}}]}),
        encoding="utf-8",
    )

    _sanitize_suite_result_provenance(score, source_bundle_dir=tmp_path / "bundle")

    record = json.loads(score.read_text(encoding="utf-8"))["cases"][0]["metrics"][
        "reference_provenance"
    ]
    assert "/home/producer" not in json.dumps(record)
    assert "render_response" not in record
    assert "exact_record_sha256" not in record
    assert "exact_record_path" not in record


def test_sanitize_preserves_verified_binding(tmp_path: Path) -> None:
    """A prior binding whose named sidecar digests to it in the source
    bundle is preserved through re-projection."""
    from apps.content_agents_dashboard.export_benchmark_runs import (
        _sanitize_suite_result_provenance,
    )

    verbatim = {"renderer": "ovrtx", "render_metadata": {"image_width": 1024}}
    canonical = json.dumps(
        verbatim, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    sha = hashlib.sha256(canonical).hexdigest()
    rel = "assets/x/references/01_x.provenance.exact.json"
    sidecar = tmp_path / "bundle" / rel
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(canonical)
    projection = {
        "renderer": "ovrtx",
        "render_metadata": {"image_width": 1024},
        "source_usd_sha256": "a" * 64,
        "image_sha256": "b" * 64,
        "exact_record_sha256": sha,
        "exact_record_path": rel,
    }
    score = tmp_path / "suite_result.json"
    score.write_text(
        json.dumps({"cases": [{"metrics": {"reference_provenance": projection}}]}),
        encoding="utf-8",
    )

    _sanitize_suite_result_provenance(score, source_bundle_dir=tmp_path / "bundle")

    record = json.loads(score.read_text(encoding="utf-8"))["cases"][0]["metrics"][
        "reference_provenance"
    ]
    assert record["exact_record_sha256"] == sha
    assert record["exact_record_path"] == rel


def test_shareable_provenance_carries_attested_remote_identity() -> None:
    """The attested renderer_identity fields (path-free) survive the
    projection: dropping them would demote every exported remote reference
    at the dashboard's attestation gate."""
    from apps.content_agents_dashboard.export_benchmark_runs import (
        _shareable_provenance,
    )

    identity = {
        "endpoint": "https://render.internal:8443/v1",
        "engine": "ovrtx",
        "protocol_version": 1,
        "status": "ready",
        "hostname": "producer-box",  # not part of the schema; must not travel
    }
    record = {
        "renderer": "remote",
        "render_metadata": {"image_width": 1024},
        "source_usd_sha256": "a" * 64,
        "image_sha256": "b" * 64,
        "render_response": {
            "data": {"results": [{"renderer_identity": identity}]},
            "results": [{"usd_path": "/home/producer/in.usdc"}],
        },
    }

    projected = _shareable_provenance(record)

    assert projected["renderer_identity"] == {
        "endpoint": "https://render.internal:8443/v1",
        "engine": "ovrtx",
        "protocol_version": 1,
        "status": "ready",
    }
    assert "render_response" not in projected
    assert "/home/producer" not in json.dumps(projected)


def test_export_refuses_credentialed_exact_record(tmp_path: Path) -> None:
    """A verbatim record whose identity endpoint carries userinfo must not
    ship: the digest-bound sidecar would publish the credential, so the
    reference exports without provenance."""
    source_root = tmp_path / "artifacts"
    image_bytes = b"\x89PNG-not-really"
    provenance = {
        "renderer": "remote",
        "render_metadata": {"image_width": 1024},
        "source_usd_sha256": "a" * 64,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "render_response": {
            "data": {
                "results": [
                    {
                        "renderer_identity": {
                            "endpoint": "https://user:token@render.internal/v1",
                            "engine": "ovrtx",
                            "protocol_version": 1,
                            "status": "ready",
                        }
                    }
                ]
            }
        },
    }
    reference = source_root / "mesh_segmentation" / "run-1" / "raw" / "reference.png"
    run = _write_run(
        source_root,
        asset_overrides={
            "references": [
                {"label": "reference", "path": str(reference), "provenance": provenance}
            ]
        },
    )
    reference.write_bytes(image_bytes)

    destination = export_run(run, tmp_path / "out")

    record = json.loads((destination / "bundle" / "run.json").read_text())["assets"][0][
        "references"
    ][0]
    assert "provenance" not in record
    assert "user:token" not in "".join(
        str(path) + path.read_text(encoding="utf-8", errors="ignore")
        for path in destination.rglob("*.json")
    )


def test_export_screens_report_sidecar(tmp_path: Path) -> None:
    """The report's exact-record sidecar is screened before copying: a
    credential-bearing or unparseable sidecar (stale pre-gate renderer,
    replaced file) is omitted from the hosted export."""
    run = _write_run(tmp_path / "artifacts")
    (run / "score").mkdir(exist_ok=True)
    (run / "score" / "report.html").write_text("<html></html>", encoding="utf-8")
    (run / "score" / "report.html.provenance.exact.json").write_text(
        json.dumps(
            {
                "asset": {
                    "renderer": "remote",
                    "render_response": {
                        "data": {
                            "results": [
                                {
                                    "renderer_identity": {
                                        "endpoint": "https://u:tok@host/v1"
                                    }
                                }
                            ]
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    destination = export_run(run, tmp_path / "out")

    assert (destination / "score" / "report.html").is_file()
    assert not (destination / "score" / "report.html.provenance.exact.json").exists()

    # A clean sidecar still travels.
    run2 = _write_run(tmp_path / "artifacts2", run_id="run-2")
    (run2 / "score").mkdir(exist_ok=True)
    (run2 / "score" / "report.html").write_text("<html></html>", encoding="utf-8")
    (run2 / "score" / "report.html.provenance.exact.json").write_text(
        json.dumps({"asset": {"renderer": "ovrtx"}}), encoding="utf-8"
    )
    destination2 = export_run(run2, tmp_path / "out2")
    assert (destination2 / "score" / "report.html.provenance.exact.json").is_file()
