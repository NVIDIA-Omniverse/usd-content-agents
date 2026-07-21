# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import json
import logging
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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
    output.write_text("#usda\n", encoding="utf-8")
    assert artifacts_router._write_local_output_usd_bundle(output) is None

    sidecar = tmp_path / "scene_physics_assets"
    sidecar.mkdir()
    assert artifacts_router._write_local_output_usd_bundle(output) is None
    (sidecar / "texture.png").write_bytes(b"png")
    (sidecar / "bad:name.txt").write_bytes(b"bad")
    zip_path = artifacts_router._write_local_output_usd_bundle(output)
    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as archive:
        assert "scene_physics.usda" in archive.namelist()
        assert "scene_physics_assets/texture.png" in archive.namelist()
        assert "scene_physics_assets/bad:name.txt" not in archive.namelist()
    artifacts_router._cleanup_temp_file(zip_path)

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
    good_sidecar = "cache/physics/scene_physics_assets/texture.png"
    bad_sidecar = "cache/physics/other_assets/skip.png"
    manager.store.objects[output_key] = b"#usda\n"
    manager.store.objects[good_sidecar] = b"png"
    manager.store.objects[bad_sidecar] = b"bad"

    assert artifacts_router._store_output_sidecar_prefix(output_key) == (
        "cache/physics/scene_physics_assets/"
    )
    assert await artifacts_router._list_store_output_sidecar_keys(
        manager, "sid", output_key
    ) == [good_sidecar]

    zip_path = await artifacts_router._write_store_output_usd_bundle(
        manager,
        "sid",
        output_key,
        [good_sidecar, bad_sidecar],
    )
    with zipfile.ZipFile(zip_path) as archive:
        assert "scene_physics.usda" in archive.namelist()
        assert "scene_physics_assets/texture.png" in archive.namelist()
        assert "other_assets/skip.png" not in archive.namelist()
    artifacts_router._cleanup_temp_file(zip_path)

    with pytest.raises(KeyError):
        await artifacts_router._write_store_output_usd_bundle(
            manager,
            "sid",
            output_key,
            ["cache/physics/scene_physics_assets/missing.png"],
        )


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
    output.write_text("#usda\n", encoding="utf-8")

    manager.local_artifacts["output_usd"] = output
    response = await artifacts_router.download_output_usd("sid")
    assert isinstance(response, FileResponse)

    sidecar = tmp_path / "scene_physics_assets"
    sidecar.mkdir()
    (sidecar / "texture.png").write_bytes(b"png")
    response = await artifacts_router.download_output_usd("sid")
    assert isinstance(response, FileResponse)

    manager.local_artifacts["output_usd"] = None
    key = "cache/physics/scene_physics.usda"
    manager.store_keys["output_usd"] = [key]
    manager.store.objects[key] = b"#usda\n"
    response = await artifacts_router.download_output_usd("sid")
    assert isinstance(response, StreamingResponse)

    sidecar_key = "cache/physics/scene_physics_assets/texture.png"
    manager.store.objects[sidecar_key] = b"png"
    response = await artifacts_router.download_output_usd("sid")
    assert isinstance(response, FileResponse)

    manager.store_keys["output_usd"] = []
    with pytest.raises(HTTPException, match="Output USD not available"):
        await artifacts_router.download_output_usd("sid")
