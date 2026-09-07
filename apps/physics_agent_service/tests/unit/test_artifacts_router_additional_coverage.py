# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pxr import Sdf, Usd
from world_understanding.utils.held_file_response import open_held_artifact_file

from ...service.routers import artifacts_router
from ...service.session.manager import SessionManager


class _Store:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def list_keys(self, _session_id: str, *, prefix: str = "") -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]

    async def open_read(self, _session_id: str, key: str) -> io.BytesIO:
        return io.BytesIO(self.objects[key])


class _BoundedReadBytesIO(io.BytesIO):
    """Reject unbounded reads while recording requested chunk sizes."""

    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise AssertionError("bundle writer attempted an unbounded read")
        self.read_sizes.append(size)
        return super().read(size)


def _write_test_usd_layer(
    path: Path,
    *,
    asset_paths: tuple[str, ...] = (),
    sublayer_paths: tuple[str, ...] = (),
) -> None:
    """Write one real USDA/USDC layer with selected authored dependencies."""

    _ = Usd.GetVersion()
    layer = Sdf.Layer.CreateNew(str(path))
    assert layer is not None
    prim = Sdf.CreatePrimInLayer(layer, "/Root")
    prim.specifier = Sdf.SpecifierDef
    for index, asset_path in enumerate(asset_paths):
        attribute = Sdf.AttributeSpec(
            prim,
            f"asset_{index}",
            Sdf.ValueTypeNames.Asset,
        )
        attribute.default = Sdf.AssetPath(asset_path)
    layer.subLayerPaths = list(sublayer_paths)
    layer.Save()


class _Manager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.storage_path = root
        self.store = _Store()
        self.exists = True
        self.local_artifacts: dict[str, Path | None] = {}
        self.stream_artifacts: dict[str, io.BytesIO | None] = {}
        self.store_keys: dict[str, list[str]] = {}
        self.sync_calls: list[str] = []

    async def session_exists(self, _session_id: str) -> bool:
        return self.exists

    def get_session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def get_artifact_path(self, _session_id: str, artifact_type: str):
        return self.local_artifacts.get(artifact_type)

    async def open_local_artifact_key(self, session_id: str, key: str):
        try:
            return open_held_artifact_file(self.root, f"{session_id}/{key}")
        except (OSError, RuntimeError, ValueError):
            return None

    async def get_local_artifact_stream(
        self,
        _session_id: str,
        artifact_type: str,
    ):
        path = self.local_artifacts.get(artifact_type)
        if path is None:
            return None
        try:
            relative_key = path.relative_to(self.root).as_posix()
            return open_held_artifact_file(self.root, relative_key), relative_key
        except (OSError, RuntimeError, ValueError):
            return None

    async def get_artifact_stream(
        self,
        _session_id: str,
        artifact_type: str,
        key: str | None = None,
    ):
        if key is not None and key in self.store.objects:
            return io.BytesIO(self.store.objects[key])
        return self.stream_artifacts.get(artifact_type)

    async def list_artifact_keys(
        self, _session_id: str, artifact_type: str
    ) -> list[str]:
        return self.store_keys.get(artifact_type, [])

    async def sync_from_store(self, _session_id: str, *, prefix: str = "") -> int:
        self.sync_calls.append(prefix)
        return 0

    async def _expected_output_usd_suffix(
        self,
        _session_id: str,
        _session_dir: Path | None = None,
    ) -> str:
        return ".usda"


@pytest.mark.asyncio
async def test_generate_report_on_demand_uses_reporting_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    predictions.write_text(json.dumps({"id": "/a"}) + "\n", encoding="utf-8")
    dataset.write_text(json.dumps({"id": "/a"}) + "\n", encoding="utf-8")
    calls = []

    reporting = ModuleType("physics_agent.tasks.reporting")

    class FakeTask:
        def run(self, context, _unused) -> None:
            calls.append(context)

    reporting.GeneratePredictionReportTask = FakeTask
    monkeypatch.setitem(sys.modules, "physics_agent.tasks.reporting", reporting)
    service_dir = artifacts_router.Path(artifacts_router.__file__).parent.parent.parent
    for path in (str(service_dir.parent), str(service_dir.parent.parent)):
        while path in sys.path:
            sys.path.remove(path)

    await artifacts_router._generate_report_on_demand(tmp_path, predictions, dataset)

    assert calls[0]["predictions"] == [{"id": "/a"}]
    assert calls[0]["dataset"] == [{"id": "/a"}]


@pytest.mark.asyncio
async def test_s3_report_prefers_published_html_without_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"
    published = b"<html>published-with-images-and-metadata</html>"

    async def sync_to_local(
        _session_id: str,
        local_session_dir: str,
        *,
        prefix,
    ) -> int:
        assert prefix == (
            "cache/predictions/",
            "cache/dataset/dataset.jsonl",
        )
        report = Path(local_session_dir) / "cache" / "predictions" / "report.html"
        report.parent.mkdir(parents=True)
        report.write_bytes(published)
        return 1

    async def fail_generate(*_args, **_kwargs) -> None:
        raise AssertionError("published reports must not be regenerated")

    manager.store.sync_to_local = sync_to_local
    monkeypatch.setattr(artifacts_router, "_generate_report_on_demand", fail_generate)

    response = await artifacts_router._serve_s3_prediction_report(manager, "sid")
    try:
        assert response._stream.read() == published
    finally:
        response._stream.close()
        assert response.background is not None
        await response.background()


@pytest.mark.asyncio
async def test_s3_report_generates_from_published_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"

    async def sync_to_local(
        _session_id: str,
        local_session_dir: str,
        *,
        prefix,
    ) -> int:
        root = Path(local_session_dir)
        predictions = root / "cache" / "predictions" / "predictions.jsonl"
        dataset = root / "cache" / "dataset" / "dataset.jsonl"
        predictions.parent.mkdir(parents=True)
        dataset.parent.mkdir(parents=True)
        predictions.write_text("{}\n", encoding="utf-8")
        dataset.write_text("{}\n", encoding="utf-8")
        return 2

    async def generate(
        session_dir: Path,
        _predictions_path: Path,
        _dataset_path: Path,
    ) -> None:
        (session_dir / "cache" / "predictions" / "report.html").write_text(
            "generated",
            encoding="utf-8",
        )

    manager.store.sync_to_local = sync_to_local
    monkeypatch.setattr(artifacts_router, "_generate_report_on_demand", generate)

    response = await artifacts_router._serve_s3_prediction_report(manager, "sid")
    try:
        assert response._stream.read() == b"generated"
    finally:
        response._stream.close()
        assert response.background is not None
        await response.background()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("write_predictions", "write_dataset", "detail"),
    [
        (False, False, "Predictions not available yet"),
        (True, False, "Dataset not available"),
    ],
)
async def test_s3_report_missing_published_inputs_are_404(
    tmp_path: Path,
    write_predictions: bool,
    write_dataset: bool,
    detail: str,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"

    async def sync_to_local(
        _session_id: str,
        local_session_dir: str,
        *,
        prefix,
    ) -> int:
        root = Path(local_session_dir)
        if write_predictions:
            path = root / "cache" / "predictions" / "predictions.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
        if write_dataset:
            path = root / "cache" / "dataset" / "dataset.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
        return 0

    manager.store.sync_to_local = sync_to_local
    with pytest.raises(HTTPException, match=detail) as error:
        await artifacts_router._serve_s3_prediction_report(manager, "sid")
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_s3_report_sync_failure_is_sanitized_500(tmp_path: Path) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"

    async def fail_sync(*_args, **_kwargs) -> int:
        raise RuntimeError("remote secret")

    manager.store.sync_to_local = fail_sync
    with pytest.raises(HTTPException, match="Report generation failed") as error:
        await artifacts_router._serve_s3_prediction_report(manager, "sid")
    assert error.value.status_code == 500


@pytest.mark.asyncio
async def test_s3_report_cancellation_cleans_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"
    snapshot_dir = tmp_path / "cancelled-report-snapshot"
    snapshot_dir.mkdir()

    async def cancel_sync(*_args, **_kwargs) -> int:
        raise asyncio.CancelledError

    manager.store.sync_to_local = cancel_sync
    monkeypatch.setattr(
        artifacts_router.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(snapshot_dir),
    )

    with pytest.raises(asyncio.CancelledError):
        await artifacts_router._serve_s3_prediction_report(manager, "sid")
    assert not snapshot_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_error", "expected_type", "expected_status"),
    [
        (RuntimeError("response construction failed"), HTTPException, 500),
        (
            HTTPException(status_code=409, detail="response conflict"),
            HTTPException,
            409,
        ),
        (asyncio.CancelledError(), asyncio.CancelledError, None),
    ],
)
async def test_s3_report_response_failure_closes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_error: BaseException,
    expected_type: type[BaseException],
    expected_status: int | None,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"
    snapshot_dir = tmp_path / "failed-report-response"
    snapshot_dir.mkdir()
    opened_streams: list[io.BufferedReader] = []

    async def sync_to_local(
        _session_id: str,
        local_session_dir: str,
        *,
        prefix,
    ) -> int:
        report = Path(local_session_dir) / "cache" / "predictions" / "report.html"
        report.parent.mkdir(parents=True)
        report.write_text("report", encoding="utf-8")
        return 1

    def fail_response(artifact, **_kwargs):
        opened_streams.append(artifact.stream)
        raise response_error

    manager.store.sync_to_local = sync_to_local
    monkeypatch.setattr(
        artifacts_router.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(snapshot_dir),
    )
    monkeypatch.setattr(artifacts_router, "HeldFileResponse", fail_response)

    with pytest.raises(expected_type) as error:
        await artifacts_router._serve_s3_prediction_report(manager, "sid")
    if expected_status is not None:
        assert isinstance(error.value, HTTPException)
        assert error.value.status_code == expected_status
    assert len(opened_streams) == 1
    assert opened_streams[0].closed
    assert not snapshot_dir.exists()


@pytest.mark.asyncio
async def test_serve_artifact_local_stream_and_missing(tmp_path: Path) -> None:
    manager = _Manager(tmp_path)
    local = manager.get_session_dir("sid") / "predictions.jsonl"
    local.write_text("{}\n", encoding="utf-8")
    manager.local_artifacts["predictions"] = local
    response = await artifacts_router._serve_artifact(
        manager,
        "sid",
        "predictions",
        "application/x-ndjson",
        "predictions.jsonl",
    )
    assert isinstance(response, FileResponse)

    manager.local_artifacts["predictions"] = None
    manager.stream_artifacts["predictions"] = io.BytesIO(b"{}\n")
    response = await artifacts_router._serve_artifact(
        manager,
        "sid",
        "predictions",
        "application/x-ndjson",
        "predictions.jsonl",
    )
    assert isinstance(response, StreamingResponse)

    manager.stream_artifacts["predictions"] = None
    with pytest.raises(HTTPException, match="not available"):
        await artifacts_router._serve_artifact(
            manager,
            "sid",
            "predictions",
            "application/x-ndjson",
            "predictions.jsonl",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("alias_kind", ["leaf", "ancestor"])
async def test_physics_local_artifact_rejects_pipeline_temp_aliases(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    secret_dir = session_dir / "cache" / ".pipeline_temp"
    secret_dir.mkdir(parents=True)
    secret = secret_dir / "credential.jsonl"
    secret.write_bytes(b"physics-secret-sentinel")
    local_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    if alias_kind == "leaf":
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.symlink_to(secret)
    else:
        local_path.parent.parent.mkdir(parents=True, exist_ok=True)
        local_path.parent.rmdir()
        local_path.parent.symlink_to(secret_dir, target_is_directory=True)

    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router._serve_artifact(
            manager,
            session_id,
            "predictions",
            "application/x-ndjson",
            "predictions.jsonl",
        )

    assert exc_info.value.status_code == 404
    assert secret.read_bytes() == b"physics-secret-sentinel"


@pytest.mark.asyncio
async def test_physics_local_response_holds_inode_across_path_swap(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    local_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"safe-physics-bytes")

    response = await artifacts_router._serve_artifact(
        manager,
        session_id,
        "predictions",
        "application/x-ndjson",
        "predictions.jsonl",
    )
    assert isinstance(response, FileResponse)

    detached = local_path.with_name("predictions.safe")
    local_path.rename(detached)
    secret = session_dir / "cache" / ".pipeline_temp" / "credential.jsonl"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b"physics-secret-sentinel")
    local_path.symlink_to(secret)

    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await response(
        {"type": "http", "method": "GET", "headers": [], "extensions": {}},
        receive,
        send,
    )
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    assert body == b"safe-physics-bytes"
    assert b"physics-secret-sentinel" not in body


def test_artifact_zip_helpers_cover_safe_and_unsafe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert artifacts_router._usd_media_type("scene.usda") == "text/plain"
    assert (
        artifacts_router._usd_media_type("scene.unknown") == "application/octet-stream"
    )
    assert artifacts_router._output_usd_bundle_filename("scene_physics.usda") == (
        "scene_physics_bundle.zip"
    )
    with pytest.raises(ValueError):
        artifacts_router._archive_name_for_output_file("../bad.usda")
    with pytest.raises(ValueError):
        artifacts_router._archive_name_for_sidecar("assets", Path("bad:name.txt"))

    output = tmp_path / "scene_physics.usda"
    _write_test_usd_layer(output)
    assert artifacts_router._write_local_output_usd_bundle(output) is None

    sidecar = tmp_path / "scene_physics.usda_assets"
    sidecar.mkdir()
    assert artifacts_router._write_local_output_usd_bundle(output) is None

    empty_output = tmp_path / "empty_sidecar.usda"
    _write_test_usd_layer(
        empty_output,
        asset_paths=("empty_sidecar.usda_assets/missing.png",),
    )
    (tmp_path / "empty_sidecar.usda_assets").mkdir()
    assert artifacts_router._write_local_output_usd_bundle(empty_output) is None

    (sidecar / "texture.png").write_bytes(b"png")
    (sidecar / "bad:name.txt").write_bytes(b"bad")
    legacy_shadow = tmp_path / "scene_physics_assets"
    legacy_shadow.mkdir()
    (legacy_shadow / "stale.png").write_bytes(b"stale")
    assert artifacts_router._write_local_output_usd_bundle(output) is None
    output.unlink()
    _write_test_usd_layer(
        output,
        asset_paths=("scene_physics.usda_assets/texture.png",),
    )
    zip_path = artifacts_router._write_local_output_usd_bundle(output)
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as archive:
        assert "scene_physics.usda" in archive.namelist()
        assert "scene_physics.usda_assets/texture.png" in archive.namelist()
        assert "scene_physics.usda_assets/bad:name.txt" not in archive.namelist()
        assert "scene_physics_assets/stale.png" not in archive.namelist()
    artifacts_router._cleanup_temp_file(zip_path)

    legacy_output = tmp_path / "legacy.usda"
    _write_test_usd_layer(
        legacy_output,
        asset_paths=("legacy_assets/texture.png",),
    )
    legacy_sidecar = tmp_path / "legacy_assets"
    legacy_sidecar.mkdir()
    (legacy_sidecar / "texture.png").write_bytes(b"legacy")
    legacy_zip = artifacts_router._write_local_output_usd_bundle(legacy_output)
    assert legacy_zip is not None
    with zipfile.ZipFile(legacy_zip) as archive:
        assert archive.read("legacy_assets/texture.png") == b"legacy"
    artifacts_router._cleanup_temp_file(legacy_zip)

    real_zipfile = zipfile.ZipFile

    class BadZip:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def write(self, *_args, **_kwargs) -> None:
            raise RuntimeError("zip failed")

    monkeypatch.setattr(zipfile, "ZipFile", BadZip)
    with pytest.raises(RuntimeError, match="zip failed"):
        artifacts_router._write_local_output_usd_bundle(output)
    monkeypatch.setattr(zipfile, "ZipFile", real_zipfile)

    def fail_unlink(self: Path, *, missing_ok: bool = False) -> None:
        raise OSError("cannot remove")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    artifacts_router._cleanup_temp_file(tmp_path / "leftover.zip")


@pytest.mark.asyncio
async def test_store_output_usd_bundle_and_helpers(tmp_path: Path) -> None:
    manager = _Manager(tmp_path)
    output_key = "cache/physics/scene_physics.usda"
    good_sidecar = "cache/physics/scene_physics.usda_assets/texture.png"
    bad_sidecar = "cache/physics/other_assets/skip.png"
    current_store_root = tmp_path / "current-store-root.usda"
    _write_test_usd_layer(
        current_store_root,
        asset_paths=("scene_physics.usda_assets/texture.png",),
    )
    manager.store.objects[output_key] = current_store_root.read_bytes()
    manager.store.objects[good_sidecar] = b"png"
    manager.store.objects["cache/physics/scene_physics_assets/stale.png"] = b"stale"
    manager.store.objects[bad_sidecar] = b"bad"

    assert artifacts_router._store_output_sidecar_prefix(output_key) == (
        "cache/physics/scene_physics.usda_assets/"
    )
    assert await artifacts_router._list_store_output_sidecar_keys(
        manager, "sid", output_key
    ) == [good_sidecar]
    assert (
        await artifacts_router._list_store_output_sidecar_keys(
            manager,
            "sid",
            "cache/physics/extensionless",
        )
        == []
    )

    zip_path = await artifacts_router._write_store_output_usd_bundle(
        manager,
        "sid",
        output_key,
        [good_sidecar, bad_sidecar],
    )
    with zipfile.ZipFile(zip_path) as archive:
        assert "scene_physics.usda" in archive.namelist()
        assert "scene_physics.usda_assets/texture.png" in archive.namelist()
        assert "other_assets/skip.png" not in archive.namelist()
    artifacts_router._cleanup_temp_file(zip_path)

    with pytest.raises(KeyError):
        await artifacts_router._write_store_output_usd_bundle(
            manager,
            "sid",
            output_key,
            ["cache/physics/scene_physics.usda_assets/missing.png"],
        )

    legacy_output_key = "cache/physics/legacy.usda"
    legacy_sidecar = "cache/physics/legacy_assets/texture.png"
    legacy_store_root = tmp_path / "legacy-store-root.usda"
    _write_test_usd_layer(
        legacy_store_root,
        asset_paths=("legacy_assets/texture.png",),
    )
    manager.store.objects[legacy_output_key] = legacy_store_root.read_bytes()
    manager.store.objects[legacy_sidecar] = b"legacy"
    assert await artifacts_router._list_store_output_sidecar_keys(
        manager,
        "sid",
        legacy_output_key,
    ) == [legacy_sidecar]
    legacy_zip = await artifacts_router._write_store_output_usd_bundle(
        manager,
        "sid",
        legacy_output_key,
        [legacy_sidecar],
    )
    with zipfile.ZipFile(legacy_zip) as archive:
        assert archive.read("legacy_assets/texture.png") == b"legacy"
    artifacts_router._cleanup_temp_file(legacy_zip)


@pytest.mark.parametrize(
    "sidecar_name",
    ["scene.usda_assets", "scene_assets"],
    ids=["current", "legacy"],
)
def test_open_output_ignores_unreferenced_sidecar(
    tmp_path: Path,
    sidecar_name: str,
) -> None:
    root = tmp_path / "store"
    output = root / "sid" / "cache" / "physics" / "scene.usda"
    output.parent.mkdir(parents=True)
    _write_test_usd_layer(output)
    sidecar_file = output.parent / sidecar_name / "stale.png"
    sidecar_file.parent.mkdir()
    sidecar_file.write_bytes(b"stale")

    artifact = open_held_artifact_file(root, "sid/cache/physics/scene.usda")
    try:
        assert artifact.stream.tell() == 0
        assert (
            artifacts_router._write_open_output_usd_bundle(
                root,
                "sid",
                artifact,
                "cache/physics/scene.usda",
            )
            is None
        )
        assert artifact.stream.tell() == 0
        assert artifacts_router._write_local_output_usd_bundle(output) is None
    finally:
        artifact.stream.close()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "fifo"])
def test_open_output_ignores_unreferenced_unsafe_sidecar(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    root = tmp_path / "store"
    output = root / "sid" / "cache" / "physics" / "scene.usda"
    output.parent.mkdir(parents=True)
    _write_test_usd_layer(output)
    unsafe_entry = output.parent / "scene.usda_assets" / "unsafe"
    unsafe_entry.parent.mkdir()
    if unsafe_kind == "symlink":
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"unsafe")
        unsafe_entry.symlink_to(outside)
    else:
        os.mkfifo(unsafe_entry)

    artifact = open_held_artifact_file(root, "sid/cache/physics/scene.usda")
    try:
        assert (
            artifacts_router._write_open_output_usd_bundle(
                root,
                "sid",
                artifact,
                "cache/physics/scene.usda",
            )
            is None
        )
        assert artifact.stream.tell() == 0
    finally:
        artifact.stream.close()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "fifo"])
def test_open_output_rejects_referenced_unsafe_sidecar(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    root = tmp_path / "store"
    output = root / "sid" / "cache" / "physics" / "scene.usda"
    output.parent.mkdir(parents=True)
    _write_test_usd_layer(
        output,
        asset_paths=("scene.usda_assets/unsafe",),
    )
    unsafe_entry = output.parent / "scene.usda_assets" / "unsafe"
    unsafe_entry.parent.mkdir()
    if unsafe_kind == "symlink":
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"unsafe")
        unsafe_entry.symlink_to(outside)
    else:
        os.mkfifo(unsafe_entry)

    artifact = open_held_artifact_file(root, "sid/cache/physics/scene.usda")
    try:
        with pytest.raises(artifacts_router.ArtifactPathError):
            artifacts_router._write_open_output_usd_bundle(
                root,
                "sid",
                artifact,
                "cache/physics/scene.usda",
            )
        assert artifact.stream.tell() == 0
    finally:
        artifact.stream.close()


def test_open_output_legacy_parse_failure_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    output = root / "sid" / "cache" / "physics" / "scene.usda"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"not-a-usd-layer")
    legacy_file = output.parent / "scene_assets" / "texture.png"
    legacy_file.parent.mkdir()
    legacy_file.write_bytes(b"stale-or-required")

    artifact = open_held_artifact_file(root, "sid/cache/physics/scene.usda")
    try:
        with pytest.raises(artifacts_router.ArtifactPathError):
            artifacts_router._write_open_output_usd_bundle(
                root,
                "sid",
                artifact,
                "cache/physics/scene.usda",
            )
        assert artifact.stream.tell() == 0
        with pytest.raises(artifacts_router.ArtifactPathError):
            artifacts_router._write_local_output_usd_bundle(output)
    finally:
        artifact.stream.close()


@pytest.mark.parametrize(
    "sidecar_name",
    ["scene.usda_assets", "scene_assets"],
    ids=["current", "legacy"],
)
def test_open_output_bundles_authored_composition_sidecar(
    tmp_path: Path,
    sidecar_name: str,
) -> None:
    root = tmp_path / "store"
    output = root / "sid" / "cache" / "physics" / "scene.usda"
    output.parent.mkdir(parents=True)
    _write_test_usd_layer(
        output,
        sublayer_paths=(f"{sidecar_name}/sub.usda",),
    )
    sidecar_file = output.parent / sidecar_name / "sub.usda"
    sidecar_file.parent.mkdir()
    sidecar_file.write_bytes(b"#usda 1.0\n")

    artifact = open_held_artifact_file(root, "sid/cache/physics/scene.usda")
    zip_path = artifacts_router._write_open_output_usd_bundle(
        root,
        "sid",
        artifact,
        "cache/physics/scene.usda",
    )
    assert zip_path is not None
    try:
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.read(f"{sidecar_name}/sub.usda") == b"#usda 1.0\n"
    finally:
        artifact.stream.close()
        artifacts_router._cleanup_temp_file(zip_path)


@pytest.mark.asyncio
async def test_store_legacy_sidecar_handles_extension_collision_exactly(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    output_key = "cache/physics/scene.usda"
    stale_legacy = "cache/physics/scene_assets/from-usdc.png"
    root = tmp_path / "scene-root.usda"
    _write_test_usd_layer(
        root,
        asset_paths=(
            "scene.usdc_assets/texture.png",
            "nested/scene_assets/not-root-relative.png",
        ),
    )
    manager.store.objects[output_key] = root.read_bytes()
    manager.store.objects[stale_legacy] = b"stale"
    assert not artifacts_router._authored_asset_uses_sidecar_directory(
        "../scene_assets/escape.png",
        "scene_assets",
    )

    assert (
        await artifacts_router._list_store_output_sidecar_keys(
            manager,
            "sid",
            output_key,
        )
        == []
    )


@pytest.mark.asyncio
async def test_store_ignores_unreferenced_current_sidecar(tmp_path: Path) -> None:
    manager = _Manager(tmp_path)
    output_key = "cache/physics/scene.usda"
    current_key = "cache/physics/scene.usda_assets/stale.png"
    root = tmp_path / "scene-root.usda"
    _write_test_usd_layer(root)
    manager.store.objects[output_key] = root.read_bytes()
    manager.store.objects[current_key] = b"stale"

    assert (
        await artifacts_router._list_store_output_sidecar_keys(
            manager,
            "sid",
            output_key,
        )
        == []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sidecar_name",
    ["scene.usda_assets", "scene_assets"],
    ids=["current", "legacy"],
)
async def test_store_sidecar_parse_failure_fails_closed_and_closes_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_name: str,
) -> None:
    manager = _Manager(tmp_path)
    output_key = "cache/physics/scene.usda"
    sidecar_key = f"cache/physics/{sidecar_name}/texture.png"
    manager.store.objects[output_key] = b"not-a-usd-layer"
    manager.store.objects[sidecar_key] = b"stale-or-required"
    output_stream = _BoundedReadBytesIO(manager.store.objects[output_key])

    async def open_output(_session_id: str, key: str) -> _BoundedReadBytesIO:
        assert key == output_key
        return output_stream

    monkeypatch.setattr(manager.store, "open_read", open_output)
    with pytest.raises(artifacts_router.ArtifactPathError):
        await artifacts_router._list_store_output_sidecar_keys(
            manager,
            "sid",
            output_key,
        )

    assert output_stream.closed
    assert output_stream.read_sizes


@pytest.mark.asyncio
async def test_store_bundles_authored_legacy_usdc_sidecar(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    output_key = "cache/physics/scene.usdc"
    legacy_key = "cache/physics/scene_assets/sub.usda"
    root = tmp_path / "scene-root.usdc"
    _write_test_usd_layer(
        root,
        sublayer_paths=("scene_assets/sub.usda",),
    )
    manager.store.objects[output_key] = root.read_bytes()
    manager.store.objects[legacy_key] = b"#usda 1.0\n"

    sidecar_keys = await artifacts_router._list_store_output_sidecar_keys(
        manager,
        "sid",
        output_key,
    )
    assert sidecar_keys == [legacy_key]
    zip_path = await artifacts_router._write_store_output_usd_bundle(
        manager,
        "sid",
        output_key,
        sidecar_keys,
    )
    try:
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.read("scene_assets/sub.usda") == b"#usda 1.0\n"
    finally:
        artifacts_router._cleanup_temp_file(zip_path)


def test_open_output_usd_bundle_uses_bounded_stream_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "store"
    root.mkdir()
    output_file = root / "output-metadata"
    sidecar_file = root / "sidecar-metadata"
    authored_root = tmp_path / "authored-root.usda"
    _write_test_usd_layer(
        authored_root,
        asset_paths=("scene.usda_assets/texture.png",),
    )
    output_file.write_bytes(authored_root.read_bytes())
    sidecar_file.write_bytes(b"metadata")

    output_stream = _BoundedReadBytesIO(output_file.read_bytes())
    sidecar_stream = _BoundedReadBytesIO(b"texture-bytes")
    output_artifact = artifacts_router.OpenArtifactFile(
        "sid/cache/physics/scene.usda",
        output_stream,
        output_file.stat(),
    )
    sidecar_artifact = artifacts_router.OpenArtifactFile(
        "sid/cache/physics/scene.usda_assets/texture.png",
        sidecar_stream,
        sidecar_file.stat(),
    )

    def fake_sidecars(_root_descriptor: int, *, prefix: str = ""):
        if prefix.endswith("scene.usda_assets/"):
            try:
                yield sidecar_artifact
            finally:
                sidecar_stream.close()

    monkeypatch.setattr(
        artifacts_router,
        "iter_open_regular_files",
        fake_sidecars,
    )

    zip_path = artifacts_router._write_open_output_usd_bundle(
        root,
        "sid",
        output_artifact,
        "cache/physics/scene.usda",
    )
    assert zip_path is not None
    try:
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.read("scene.usda") == output_file.read_bytes()
            assert archive.read("scene.usda_assets/texture.png") == b"texture-bytes"
        assert output_stream.read_sizes
        assert sidecar_stream.read_sizes
        assert max(output_stream.read_sizes) <= artifacts_router._ZIP_COPY_CHUNK_SIZE
        assert max(sidecar_stream.read_sizes) <= artifacts_router._ZIP_COPY_CHUNK_SIZE
    finally:
        output_stream.close()
        sidecar_stream.close()
        artifacts_router._cleanup_temp_file(zip_path)


@pytest.mark.asyncio
async def test_store_output_usd_bundle_uses_bounded_stream_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    output_key = "cache/physics/scene.usda"
    sidecar_key = "cache/physics/scene.usda_assets/texture.png"
    streams: dict[str, _BoundedReadBytesIO] = {}
    objects = {
        output_key: b"root-usd-bytes",
        sidecar_key: b"texture-bytes",
    }

    async def open_read(_session_id: str, key: str) -> _BoundedReadBytesIO:
        stream = _BoundedReadBytesIO(objects[key])
        streams[key] = stream
        return stream

    monkeypatch.setattr(manager.store, "open_read", open_read)
    zip_path = await artifacts_router._write_store_output_usd_bundle(
        manager,
        "sid",
        output_key,
        [sidecar_key],
    )
    try:
        with zipfile.ZipFile(zip_path) as archive:
            assert archive.read("scene.usda") == b"root-usd-bytes"
            assert archive.read("scene.usda_assets/texture.png") == b"texture-bytes"
        assert set(streams) == {output_key, sidecar_key}
        for stream in streams.values():
            assert stream.closed
            assert stream.read_sizes
            assert max(stream.read_sizes) <= artifacts_router._ZIP_COPY_CHUNK_SIZE
    finally:
        artifacts_router._cleanup_temp_file(zip_path)


@pytest.mark.asyncio
async def test_store_output_usd_bundle_cancellation_cleans_temp_zip_and_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    output_key = "cache/physics/scene.usda"
    sidecar_key = "cache/physics/scene.usda_assets/texture.png"
    output_stream = _BoundedReadBytesIO(b"root-usd-bytes")

    async def cancel_during_sidecar_open(
        _session_id: str,
        key: str,
    ) -> _BoundedReadBytesIO:
        if key == output_key:
            return output_stream
        raise asyncio.CancelledError

    zip_path = tmp_path / "cancelled.zip"
    zip_path.write_bytes(b"named-temp-placeholder")
    monkeypatch.setattr(manager.store, "open_read", cancel_during_sidecar_open)
    monkeypatch.setattr(artifacts_router, "_new_temp_zip_path", lambda: zip_path)

    with pytest.raises(asyncio.CancelledError):
        await artifacts_router._write_store_output_usd_bundle(
            manager,
            "sid",
            output_key,
            [sidecar_key],
        )

    assert output_stream.closed
    assert not zip_path.exists()


@pytest.mark.asyncio
async def test_artifact_endpoints_session_and_report_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    artifacts_router.set_session_manager(manager)

    manager.exists = False
    with pytest.raises(HTTPException, match="Session not found"):
        await artifacts_router.download_predictions("sid")
    with pytest.raises(HTTPException, match="Session not found"):
        await artifacts_router.view_prediction_report("sid")
    with pytest.raises(HTTPException, match="Session not found"):
        await artifacts_router.download_dataset("sid")
    manager.exists = True

    session_dir = manager.get_session_dir("sid")
    report = session_dir / "cache" / "predictions" / "report.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html></html>", encoding="utf-8")
    response = await artifacts_router.view_prediction_report("sid")
    assert isinstance(response, FileResponse)

    report.unlink()
    called = []

    async def fake_generate(_session_dir, _predictions_path, _dataset_path):
        called.append(True)
        report.write_text("<html>generated</html>", encoding="utf-8")

    monkeypatch.setattr(artifacts_router, "_generate_report_on_demand", fake_generate)
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")
    response = await artifacts_router.view_prediction_report("sid")
    assert called
    assert isinstance(response, FileResponse)

    report.unlink()
    predictions.unlink()
    dataset.unlink()

    async def sync_and_restore(_session_id: str, *, prefix: str = "") -> int:
        manager.sync_calls.append(prefix)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        predictions.write_text("{}\n", encoding="utf-8")
        dataset.write_text("{}\n", encoding="utf-8")
        return 1

    manager.sync_from_store = sync_and_restore
    monkeypatch.setattr(artifacts_router, "_generate_report_on_demand", fake_generate)
    response = await artifacts_router.view_prediction_report("sid")
    assert isinstance(response, FileResponse)

    predictions.unlink()
    report.unlink()

    async def no_sync(_session_id: str, *, prefix: str = "") -> int:
        manager.sync_calls.append(prefix)
        return 0

    manager.sync_from_store = no_sync
    with pytest.raises(HTTPException, match="Predictions not available"):
        await artifacts_router.view_prediction_report("sid")

    predictions.write_text("{}\n", encoding="utf-8")
    dataset.unlink()
    with pytest.raises(HTTPException, match="Dataset not available"):
        await artifacts_router.view_prediction_report("sid")

    sentinel = "physics-report-publication-sentinel-727"

    async def fail_generate(*_args, **_kwargs):
        raise RuntimeError(sentinel)

    dataset.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(artifacts_router, "_generate_report_on_demand", fail_generate)
    with caplog.at_level(logging.ERROR, logger=artifacts_router.__name__):
        with pytest.raises(HTTPException) as exc_info:
            await artifacts_router.view_prediction_report("sid")
    assert exc_info.value.detail == "Report generation failed"
    assert "physics_prediction_report_publication_failed" in caplog.text
    assert "phase=local_publication" in caplog.text
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_download_output_usd_local_store_and_missing(tmp_path: Path) -> None:
    manager = _Manager(tmp_path)
    artifacts_router.set_session_manager(manager)
    output = tmp_path / "scene_physics.usda"
    _write_test_usd_layer(output)

    manager.local_artifacts["output_usd"] = output
    response = await artifacts_router.download_output_usd("sid")
    assert isinstance(response, FileResponse)

    sidecar = tmp_path / "scene_physics.usda_assets"
    sidecar.mkdir()
    (sidecar / "texture.png").write_bytes(b"png")
    _write_test_usd_layer(
        output,
        asset_paths=("scene_physics.usda_assets/texture.png",),
    )
    response = await artifacts_router.download_output_usd("sid")
    assert isinstance(response, FileResponse)

    manager.local_artifacts["output_usd"] = None
    key = "cache/physics/scene_physics.usda"
    manager.store_keys["output_usd"] = [key]
    _write_test_usd_layer(output)
    manager.store.objects[key] = output.read_bytes()
    response = await artifacts_router.download_output_usd("sid")
    assert isinstance(response, StreamingResponse)

    sidecar_key = "cache/physics/scene_physics.usda_assets/texture.png"
    manager.store.objects[sidecar_key] = b"png"
    _write_test_usd_layer(
        output,
        asset_paths=("scene_physics.usda_assets/texture.png",),
    )
    manager.store.objects[key] = output.read_bytes()
    response = await artifacts_router.download_output_usd("sid")
    assert isinstance(response, FileResponse)

    manager.store_keys["output_usd"] = []
    with pytest.raises(HTTPException, match="Output USD not available"):
        await artifacts_router.download_output_usd("sid")


def test_open_output_bundle_cleans_partial_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "store"
    output = root / "sid" / "cache" / "physics" / "scene.usda"
    sidecar = root / "sid" / "cache" / "physics" / "scene_assets" / "tex.png"
    output.parent.mkdir(parents=True)
    sidecar.parent.mkdir(parents=True)
    _write_test_usd_layer(
        output,
        asset_paths=("scene_assets/tex.png",),
    )
    sidecar.write_bytes(b"png")
    artifact = open_held_artifact_file(
        root,
        "sid/cache/physics/scene.usda",
    )
    zip_path = tmp_path / "partial.zip"

    class FailingZip:
        def __init__(self, *_args, **_kwargs) -> None:
            zip_path.write_bytes(b"partial")

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def open(self, *_args, **_kwargs) -> None:
            raise OSError("zip write failed")

    monkeypatch.setattr(artifacts_router, "_new_temp_zip_path", lambda: zip_path)
    monkeypatch.setattr(artifacts_router.zipfile, "ZipFile", FailingZip)
    with pytest.raises(OSError, match="zip write failed"):
        artifacts_router._write_open_output_usd_bundle(
            root,
            "sid",
            artifact,
            "cache/physics/scene.usda",
        )
    artifact.stream.close()
    assert not zip_path.exists()


@pytest.mark.asyncio
async def test_local_output_bundle_fails_closed_after_unsafe_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path / "store")
    artifacts_router.set_session_manager(manager)
    output = manager.get_session_dir("sid") / "cache" / "physics" / "scene.usda"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"complete-root-usd")
    sidecar = output.parent / "scene.usda_assets"
    sidecar.mkdir()
    (sidecar / "a-safe.png").write_bytes(b"safe")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"unsafe")
    (sidecar / "z-unsafe.png").symlink_to(outside)
    captured: dict[str, object] = {}

    async def open_local_output(*_args, **_kwargs):
        artifact = open_held_artifact_file(
            manager.storage_path,
            "sid/cache/physics/scene.usda",
        )
        captured["artifact"] = artifact
        return artifact, "cache/physics/scene.usda"

    zip_path = tmp_path / "partial.zip"
    monkeypatch.setattr(manager, "get_local_artifact_stream", open_local_output)
    monkeypatch.setattr(artifacts_router, "_new_temp_zip_path", lambda: zip_path)

    with pytest.raises(artifacts_router.ArtifactPathError):
        await artifacts_router.download_output_usd("sid")

    assert captured["artifact"].stream.closed
    assert not zip_path.exists()


@pytest.mark.asyncio
async def test_report_and_local_output_response_defensive_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    artifacts_router.set_session_manager(manager)
    report = manager.get_session_dir("sid") / "cache" / "predictions" / "report.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html></html>", encoding="utf-8")

    async def missing_report(*_args, **_kwargs):
        return None

    monkeypatch.setattr(manager, "open_local_artifact_key", missing_report)
    with pytest.raises(HTTPException, match="Prediction report not available"):
        await artifacts_router.view_prediction_report("sid")

    output = manager.get_session_dir("sid") / "cache" / "physics" / "scene.usda"
    output.parent.mkdir(parents=True)
    output.write_text("#usda\n", encoding="utf-8")
    manager.local_artifacts["output_usd"] = output

    def unsafe_sidecars(*_args, **_kwargs):
        raise artifacts_router.ArtifactPathError("unsafe sidecars")

    monkeypatch.setattr(
        artifacts_router,
        "_write_open_output_usd_bundle",
        unsafe_sidecars,
    )
    with pytest.raises(artifacts_router.ArtifactPathError, match="unsafe sidecars"):
        await artifacts_router.download_output_usd("sid")

    captured: dict[str, object] = {}
    original_get = manager.get_local_artifact_stream

    async def capture_artifact(*args, **kwargs):
        value = await original_get(*args, **kwargs)
        captured["artifact"] = value[0]
        return value

    def fail_response(*_args, **_kwargs):
        raise RuntimeError("response construction failed")

    monkeypatch.setattr(manager, "get_local_artifact_stream", capture_artifact)
    monkeypatch.setattr(
        artifacts_router,
        "_write_open_output_usd_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(artifacts_router, "HeldFileResponse", fail_response)
    with pytest.raises(RuntimeError, match="response construction failed"):
        await artifacts_router.download_output_usd("sid")
    assert captured["artifact"].stream.closed


@pytest.mark.asyncio
async def test_s3_output_snapshot_missing_and_direct_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"

    async def empty_sync(*_args, **_kwargs) -> int:
        return 0

    manager.store.sync_to_local = empty_sync
    with pytest.raises(HTTPException, match="Output USD not available") as error:
        await artifacts_router._serve_s3_output_usd_snapshot(manager, "sid")
    assert error.value.status_code == 404

    async def sync_candidates(
        _session_id: str,
        local_session_dir: str,
        *,
        prefix: str,
    ) -> int:
        assert prefix == "cache/physics/"
        physics = Path(local_session_dir) / "cache" / "physics"
        physics.mkdir(parents=True)
        (physics / "scene_physics.usda").write_bytes(b"preferred-usda")
        (physics / "scene_physics.usdc").write_bytes(b"secondary-usdc")
        return 2

    manager.store.sync_to_local = sync_candidates
    monkeypatch.setattr(
        artifacts_router,
        "_write_open_output_usd_bundle",
        lambda *_args, **_kwargs: None,
    )
    response = await artifacts_router._serve_s3_output_usd_snapshot(manager, "sid")
    try:
        assert response._stream.read() == b"preferred-usda"
        assert response.headers["content-disposition"].endswith(
            'filename="scene_physics.usda"'
        )
    finally:
        response._stream.close()
        assert response.background is not None
        await response.background()


@pytest.mark.asyncio
async def test_s3_output_snapshot_returns_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"

    async def sync_output(
        _session_id: str,
        local_session_dir: str,
        *,
        prefix: str,
    ) -> int:
        output = Path(local_session_dir) / "cache" / "physics" / "scene_physics.usda"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"usd")
        return 1

    manager.store.sync_to_local = sync_output
    bundle = tmp_path / "output-bundle.zip"
    bundle.write_bytes(b"zip")
    monkeypatch.setattr(
        artifacts_router,
        "_write_open_output_usd_bundle",
        lambda *_args, **_kwargs: bundle,
    )

    response = await artifacts_router._serve_s3_output_usd_snapshot(manager, "sid")
    assert isinstance(response, FileResponse)
    assert response.background is not None
    await response.background()
    assert not bundle.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["http", "base"])
async def test_s3_output_snapshot_closes_selected_artifact_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"

    async def sync_output(
        _session_id: str,
        local_session_dir: str,
        *,
        prefix: str,
    ) -> int:
        output = Path(local_session_dir) / "cache" / "physics" / "scene_physics.usda"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"usd")
        return 1

    manager.store.sync_to_local = sync_output

    def fail_bundle(*_args, **_kwargs):
        if failure_kind == "http":
            raise HTTPException(status_code=409, detail="bundle conflict")
        raise KeyboardInterrupt("bundle interrupted")

    monkeypatch.setattr(
        artifacts_router,
        "_write_open_output_usd_bundle",
        fail_bundle,
    )
    if failure_kind == "http":
        with pytest.raises(HTTPException, match="bundle conflict"):
            await artifacts_router._serve_s3_output_usd_snapshot(manager, "sid")
    else:
        with pytest.raises(KeyboardInterrupt, match="bundle interrupted"):
            await artifacts_router._serve_s3_output_usd_snapshot(manager, "sid")


@pytest.mark.asyncio
async def test_s3_artifact_endpoints_delegate_to_pinned_snapshot_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.store.kind = "s3"
    artifacts_router.set_session_manager(manager)

    async def report(*_args, **_kwargs) -> Response:
        return Response(b"report")

    async def output(*_args, **_kwargs) -> Response:
        return Response(b"output")

    monkeypatch.setattr(artifacts_router, "_serve_s3_prediction_report", report)
    monkeypatch.setattr(artifacts_router, "_serve_s3_output_usd_snapshot", output)
    assert (await artifacts_router.view_prediction_report("sid")).body == b"report"
    assert (await artifacts_router.download_output_usd("sid")).body == b"output"


def test_cleanup_temp_tree_logs_remove_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_remove(_path: Path) -> None:
        raise OSError("busy")

    monkeypatch.setattr(artifacts_router.shutil, "rmtree", fail_remove)
    with caplog.at_level(logging.WARNING):
        artifacts_router._cleanup_temp_tree(tmp_path)
    assert "Failed to remove temporary artifact snapshot" in caplog.text
