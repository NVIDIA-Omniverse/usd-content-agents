# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from apps.texture_gen_step1x_service.scripts import smoke


def _completed_status(*, include_orm: bool) -> dict[str, Any]:
    maps: dict[str, dict[str, Any]] = {
        "albedo": {
            "uri": "file:///tmp/albedo.png",
            "width": 128,
            "height": 128,
            "colorspace": "srgb",
        }
    }
    generated = {"albedo": "file:///tmp/albedo.png"}
    metadata: dict[str, Any] = {"degraded_channels": ["normal"]}
    diagnostics = [
        {
            "code": "STEP1X_MAPS_DEGRADED",
            "severity": "warning",
            "channels": ["normal"],
        }
    ]
    if include_orm:
        maps["orm"] = {
            "uri": "file:///tmp/orm.png",
            "width": 128,
            "height": 128,
            "packing": "occlusion_roughness_metallic",
        }
        generated["orm"] = "file:///tmp/orm.png"
    else:
        metadata["degraded_channels"].append("orm")
        diagnostics[0]["channels"].append("orm")
    return {
        "job_id": "vj-smoke",
        "status": "completed",
        "result": {
            "generated_textures": generated,
            "maps": maps,
            "metadata": metadata,
            "diagnostics": diagnostics,
        },
    }


def test_smoke_require_orm_accepts_full_pbr_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    status = _completed_status(include_orm=True)
    monkeypatch.setattr(
        sys, "argv", ["smoke.py", "--request", str(request_path), "--require-orm"]
    )
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": True})
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(smoke, "_artifact_visible", lambda _uri: True)

    assert smoke.main() == 0


def test_smoke_require_orm_rejects_degraded_orm_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    status = _completed_status(include_orm=False)
    monkeypatch.setattr(
        sys, "argv", ["smoke.py", "--request", str(request_path), "--require-orm"]
    )
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": True})
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(smoke, "_artifact_visible", lambda _uri: True)

    assert smoke.main() == 13


def test_smoke_allows_not_ready_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "--request", str(request_path), "--allow-not-ready"],
    )
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": False})

    assert smoke.main() == 0


def test_smoke_rejects_not_ready_without_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["smoke.py", "--request", str(request_path)])
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": False})

    assert smoke.main() == 2


def test_smoke_can_require_gpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "--request", str(request_path), "--require-gpu"],
    )
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": True})

    assert smoke.main() == 7


def test_smoke_times_out_while_job_is_processing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "--request", str(request_path), "--timeout-sec", "1"],
    )
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": True})
    monkeypatch.setattr(
        smoke,
        "_request_json",
        lambda *_args, **_kwargs: {"job_id": "vj", "status": "processing"},
    )
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(ticks))

    assert smoke.main() == 3


def test_smoke_rejects_failed_final_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["smoke.py", "--request", str(request_path)])
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": True})
    monkeypatch.setattr(
        smoke,
        "_request_json",
        lambda *_args, **_kwargs: {"job_id": "vj", "status": "failed"},
    )

    assert smoke.main() == 4


def test_smoke_polls_until_completed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    statuses = [
        {"job_id": "vj", "status": "queued"},
        {"job_id": "vj", "status": "processing", "progress": 0.5},
        _completed_status(include_orm=True),
    ]

    def fake_get_json(url: str) -> dict[str, Any]:
        if url.endswith("/health"):
            return {"ready": True}
        return statuses.pop(0)

    monkeypatch.setattr(sys, "argv", ["smoke.py", "--request", str(request_path)])
    monkeypatch.setattr(smoke, "_get_json", fake_get_json)
    monkeypatch.setattr(
        smoke, "_request_json", lambda *_args, **_kwargs: statuses.pop(0)
    )
    monkeypatch.setattr(smoke, "_artifact_visible", lambda _uri: True)
    monkeypatch.setattr(smoke.time, "sleep", lambda _seconds: None)

    assert smoke.main() == 0


def test_smoke_rejects_incomplete_albedo_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["smoke.py", "--request", str(request_path)])
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": True})

    status = _completed_status(include_orm=True)
    status["result"]["maps"]["albedo"].pop("uri")
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    assert smoke.main() == 5

    status = _completed_status(include_orm=True)
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(smoke, "_artifact_visible", lambda _uri: False)
    assert smoke.main() == 6

    status = _completed_status(include_orm=True)
    status["result"]["maps"]["albedo"].pop("width")
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(smoke, "_artifact_visible", lambda _uri: True)
    assert smoke.main() == 8

    status = _completed_status(include_orm=True)
    status["result"]["maps"]["albedo"]["colorspace"] = "linear"
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    assert smoke.main() == 9


def test_smoke_rejects_missing_degradation_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["smoke.py", "--request", str(request_path)])
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": True})
    monkeypatch.setattr(smoke, "_artifact_visible", lambda _uri: True)

    status = _completed_status(include_orm=False)
    status["result"]["metadata"]["degraded_channels"].remove("normal")
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    assert smoke.main() == 10

    status = _completed_status(include_orm=True)
    status["result"]["generated_textures"].pop("orm")
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    assert smoke.main() == 11

    status = _completed_status(include_orm=False)
    status["result"]["diagnostics"] = []
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    assert smoke.main() == 12


def test_smoke_require_orm_validates_orm_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke.py", "--request", str(request_path), "--require-orm"],
    )
    monkeypatch.setattr(smoke, "_get_json", lambda _url: {"ready": True})
    monkeypatch.setattr(smoke, "_artifact_visible", lambda _uri: True)

    status = _completed_status(include_orm=True)
    status["result"]["maps"]["orm"].pop("uri")
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    assert smoke.main() == 14

    status = _completed_status(include_orm=True)
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(
        smoke,
        "_artifact_visible",
        lambda uri: not str(uri).endswith("orm.png"),
    )
    assert smoke.main() == 15

    status = _completed_status(include_orm=True)
    status["result"]["maps"]["orm"]["packing"] = "bad"
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(smoke, "_artifact_visible", lambda _uri: True)
    assert smoke.main() == 16

    status = _completed_status(include_orm=True)
    status["result"]["metadata"]["degraded_channels"].append("orm")
    monkeypatch.setattr(smoke, "_request_json", lambda *_args, **_kwargs: status)
    assert smoke.main() == 17


def test_request_json_handles_success_and_transport_errors(monkeypatch) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return b'{"ok": true}'

    requests: list[Any] = []

    def fake_urlopen(request: Any, *, timeout: int) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(smoke, "urlopen", fake_urlopen)

    assert smoke._request_json("http://service/jobs", method="POST", body={"x": 1}) == {
        "ok": True
    }
    assert requests[0][1] == 30

    def fake_http_error(_request: Any, *, timeout: int) -> None:
        raise HTTPError(
            "http://service/jobs",
            500,
            "server error",
            hdrs=None,
            fp=io.BytesIO(b"bad"),
        )

    monkeypatch.setattr(smoke, "urlopen", fake_http_error)
    try:
        smoke._get_json("http://service/jobs")
    except RuntimeError as exc:
        assert "HTTP 500: bad" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    monkeypatch.setattr(
        smoke,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    try:
        smoke._get_json("http://service/jobs")
    except RuntimeError as exc:
        assert "failed:" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_artifact_visible_handles_file_http_and_unknown_schemes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(b"x")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")

    assert smoke._artifact_visible(artifact.as_uri()) is True
    assert smoke._artifact_visible(empty.as_uri()) is False
    assert smoke._artifact_visible("s3://bucket/key") is False

    class FakeResponse:
        status = 204

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(smoke, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    assert smoke._artifact_visible("https://example.test/artifact.png") is True

    monkeypatch.setattr(
        smoke,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    assert smoke._artifact_visible("https://example.test/artifact.png") is False
