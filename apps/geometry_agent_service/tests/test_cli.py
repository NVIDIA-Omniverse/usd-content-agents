# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from geometry_agent_client import cli
from geometry_agent_client import http as client_http
from geometry_agent_client.http import GeometryAgentClient, GeometryAgentClientError


class _FakeClient:
    def __init__(self) -> None:
        self.generated: dict[str, Any] | None = None
        self.revised: dict[str, Any] | None = None
        self.family_request: dict[str, Any] | None = None
        self.exported: dict[str, Any] | None = None
        self.provider_exported: dict[str, Any] | None = None
        self.ran: dict[str, Any] | None = None

    def upload(self, _path: Path, **_kwargs: Any) -> dict[str, str]:
        return {"source_id": "src_" + "a" * 64}

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.generated = payload
        return {
            "status": "succeeded",
            "job_id": "job_generation",
            "result": {"source": {"source_id": "src_" + "b" * 64}},
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ran = payload
        return {
            "status": "succeeded",
            "job_id": "job_run",
            "result": {"handoff_ready": True},
        }

    def revise(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.revised = payload
        return {
            "status": "succeeded",
            "job_id": "job_revision",
            "result": {"source": {"source_id": "src_" + "c" * 64}},
        }

    def family(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.family_request = payload
        return {
            "status": "succeeded",
            "job_id": "job_family",
            "result": {"succeeded_count": 2, "failed_count": 0},
        }

    def export(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.exported = payload
        return {
            "status": "succeeded",
            "job_id": "job_export",
            "result": {"source": {"source_id": "src_" + "d" * 64}},
        }

    def provider_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.provider_exported = payload
        return {
            "status": "succeeded",
            "job_id": "job_provider_export",
            "result": {"source": {"source_id": "src_" + "e" * 64}},
        }

    def wait(self, job: dict[str, Any]) -> dict[str, Any]:
        return job


def test_http_client_owned_session_ignores_ambient_proxy_configuration() -> None:
    client = GeometryAgentClient(
        base_url="http://127.0.0.1:8776",
        api_key="test-key",
    )

    assert client._session.trust_env is False


def test_http_client_preserves_injected_session_policy() -> None:
    session = requests.Session()
    session.trust_env = True

    client = GeometryAgentClient(
        base_url="http://127.0.0.1:8776",
        api_key="test-key",
        session=session,
    )

    assert client._session is session
    assert session.trust_env is True


def test_http_client_rejects_dangling_download_destination_symlink(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "download.usda"
    destination.symlink_to(tmp_path / "missing.usda")
    client = GeometryAgentClient(
        base_url="http://127.0.0.1:8776",
        api_key="test-key",
    )

    with pytest.raises(FileExistsError, match="destination already exists"):
        client.download("sha256:" + "a" * 64, destination)

    assert destination.is_symlink()


def test_http_client_rejects_upload_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.step"
    source.write_bytes(b"STEP")
    linked_source = tmp_path / "linked.step"
    linked_source.symlink_to(source)
    client = GeometryAgentClient(
        base_url="http://127.0.0.1:8776",
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="not a regular file"):
        client.upload(
            linked_source,
            role="geometry",
            media_type="model/step",
        )


@pytest.mark.parametrize("dangling", (False, True))
def test_provider_output_directory_rejects_symlink(
    tmp_path: Path,
    dangling: bool,
) -> None:
    target = tmp_path / "target"
    if not dangling:
        target.mkdir()
    output = tmp_path / "output"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError, match="already exists"):
        cli._new_output_directory(output)


def test_http_client_streams_and_stops_at_the_json_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {"chunks": 0}

    class StreamingResponse:
        status_code = 200
        headers: dict[str, str] = {}

        def iter_content(self, *, chunk_size: int):
            assert chunk_size == 64 * 1024
            for chunk in (b'{"value":', b'"too-large"}', b"must-not-be-read"):
                observed["chunks"] += 1
                yield chunk

        def close(self) -> None:
            observed["closed"] = True

    class StreamingSession:
        def request(self, *_args: Any, **kwargs: Any) -> StreamingResponse:
            observed["stream"] = kwargs["stream"]
            return StreamingResponse()

    monkeypatch.setattr(client_http, "_MAX_JSON_BYTES", 12)
    client = GeometryAgentClient(
        base_url="http://127.0.0.1:8776",
        api_key="test-key",
        session=StreamingSession(),  # type: ignore[arg-type]
    )

    with pytest.raises(GeometryAgentClientError, match="oversized JSON"):
        client.providers()

    assert observed == {"chunks": 2, "closed": True, "stream": True}


def test_http_error_body_is_parsed_without_response_json() -> None:
    observed: dict[str, Any] = {}

    class ErrorResponse:
        status_code = 422
        headers: dict[str, str] = {}

        def iter_content(self, *, chunk_size: int):
            assert chunk_size == 64 * 1024
            yield b'{"detail":{"message":"Bounded service error"}}'

        def json(self) -> None:
            raise AssertionError("response.json() must not buffer the body")

        def close(self) -> None:
            observed["closed"] = True

    class ErrorSession:
        def request(self, *_args: Any, **kwargs: Any) -> ErrorResponse:
            observed["stream"] = kwargs["stream"]
            return ErrorResponse()

    client = GeometryAgentClient(
        base_url="http://127.0.0.1:8776",
        api_key="test-key",
        session=ErrorSession(),  # type: ignore[arg-type]
    )

    with pytest.raises(GeometryAgentClientError, match="Bounded service error"):
        client.providers()

    assert observed == {"closed": True, "stream": True}


def test_generate_and_run_preserves_both_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(cli, "_client", lambda _args: fake)
    output = tmp_path / "nested" / "result.json"
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--provider",
            "build123d-worker",
            "--prompt",
            "A mounting bracket",
            "--parameter",
            "width_mm=80",
            "--format",
            "step",
            "--run",
            "--output",
            str(output),
        ]
    )

    assert args.handler(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generation"]["job_id"] == "job_generation"
    assert payload["geometry_run"]["job_id"] == "job_run"
    assert fake.generated is not None
    assert fake.generated["parameters"] == {"width_mm": 80}
    assert fake.ran is not None
    assert fake.ran["source_id"] == "src_" + "b" * 64


def test_generate_preserves_receipts_when_geometry_run_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient()

    def failed_run(payload: dict[str, Any]) -> dict[str, Any]:
        fake.ran = payload
        return {
            "status": "failed",
            "job_id": "job_run",
            "error": {"message": "OVRTX evidence could not be certified"},
        }

    monkeypatch.setattr(fake, "run", failed_run)
    monkeypatch.setattr(cli, "_client", lambda _args: fake)
    output = tmp_path / "result.json"
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--provider",
            "build123d-http",
            "--prompt",
            "A mounting bracket",
            "--run",
            "--output",
            str(output),
        ]
    )

    with pytest.raises(RuntimeError, match="OVRTX evidence could not be certified"):
        args.handler(args)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generation"]["status"] == "succeeded"
    assert payload["geometry_run"]["status"] == "failed"


def test_duplicate_semantic_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="only once"):
        cli._parameters([("width", 20), ("width", 30)])


def test_semantic_parameter_accepts_an_explicit_unit() -> None:
    assert cli._parameter('width_mm={"value":80,"unit":"mm"}') == (
        "width_mm",
        {"value": 80, "unit": "mm"},
    )


def test_revision_binds_source_provider_and_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(cli, "_client", lambda _args: fake)
    output = tmp_path / "revision.json"
    args = cli.build_parser().parse_args(
        [
            "revise",
            "src_" + "b" * 64,
            "--provider",
            "build123d-http",
            "--prompt",
            "Increase the width.",
            "--parameter",
            "width_mm=90",
            "--format",
            "step",
            "--output",
            str(output),
        ]
    )

    assert args.handler(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["job_id"] == "job_revision"
    assert fake.revised == {
        "provider_id": "build123d-http",
        "source_id": "src_" + "b" * 64,
        "instructions": "Increase the width.",
        "image_source_ids": [],
        "parameter_overrides": {"width_mm": 90},
        "requested_formats": ["step"],
    }


def test_family_and_export_commands_use_explicit_provider_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(cli, "_client", lambda _args: fake)
    variants = tmp_path / "variants.json"
    variants.write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "variant_id": "compact",
                        "parameter_overrides": {"width_mm": 60},
                    },
                    {
                        "variant_id": "wide",
                        "parameter_overrides": {"width_mm": 90},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    family_output = tmp_path / "family.json"
    family_args = cli.build_parser().parse_args(
        [
            "family",
            "src_" + "b" * 64,
            "--provider",
            "parametric-authoring-http",
            "--variants",
            str(variants),
            "--format",
            "step",
            "--output",
            str(family_output),
        ]
    )
    assert family_args.handler(family_args) == 0
    assert fake.family_request is not None
    assert fake.family_request["variants"][1]["parameter_overrides"] == {"width_mm": 90}

    export_output = tmp_path / "export.json"
    export_args = cli.build_parser().parse_args(
        [
            "export",
            "src_" + "b" * 64,
            "--provider",
            "external-cad-http",
            "--format",
            "step",
            "--output",
            str(export_output),
        ]
    )
    assert export_args.handler(export_args) == 0
    assert fake.exported == {
        "provider_id": "external-cad-http",
        "source_id": "src_" + "b" * 64,
        "requested_formats": ["step"],
    }


def test_provider_revision_export_binds_external_identity_and_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(cli, "_client", lambda _args: fake)
    output = tmp_path / "provider-export.json"
    args = cli.build_parser().parse_args(
        [
            "export-provider-revision",
            "--provider",
            "external-cad-http",
            "--source-revision",
            "immutable-revision-42",
            "--meters-per-unit",
            "0.001",
            "--up-axis",
            "Z",
            "--forward-axis",
            "+Y",
            "--rights-assertion",
            "Authorized project export.",
            "--format",
            "step",
            "--output",
            str(output),
        ]
    )

    assert args.handler(args) == 0
    assert fake.provider_exported is not None
    assert fake.provider_exported["coordinate_system"] == {
        "meters_per_unit": 0.001,
        "up_axis": "Z",
        "forward_axis": "+Y",
        "handedness": "right",
    }
    assert fake.provider_exported["requested_formats"] == ["step"]


def test_nonterminal_job_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="non-terminal"):
        cli._terminal({"status": "running"})


def test_prompt_file_symlink_is_rejected(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A bracket", encoding="utf-8")
    link = tmp_path / "prompt-link.txt"
    link.symlink_to(prompt)
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--provider",
            "build123d-worker",
            "--prompt-file",
            str(link),
        ]
    )

    with pytest.raises(ValueError, match="regular file"):
        cli._prompt(args)


def test_inline_prompt_enforces_service_character_limit() -> None:
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--provider",
            "build123d-worker",
            "--prompt",
            "x" * 32_769,
        ]
    )

    with pytest.raises(ValueError, match="32,768-character service limit"):
        cli._prompt(args)


def test_onshape_export_requires_local_api_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONSHAPE_API_KEY", raising=False)
    monkeypatch.delenv("ONSHAPE_API_SECRET", raising=False)
    monkeypatch.delenv("ONSHAPE_SECRET", raising=False)
    args = cli.build_parser().parse_args(
        [
            "export-onshape",
            "--document-id",
            "1" * 24,
            "--version-id",
            "2" * 24,
            "--element-id",
            "3" * 24,
            "--element-kind",
            "partstudio",
            "--rights-assertion",
            "Authorized test export.",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(RuntimeError, match="do not pass credentials in chat"):
        args.handler(args)


@pytest.mark.parametrize("credential_flag", ("--api-key", "--api-secret"))
def test_onshape_export_has_no_credential_arguments(
    tmp_path: Path,
    credential_flag: str,
) -> None:
    arguments = [
        "export-onshape",
        "--document-id",
        "1" * 24,
        "--version-id",
        "2" * 24,
        "--element-id",
        "3" * 24,
        "--element-kind",
        "partstudio",
        "--rights-assertion",
        "Authorized test export.",
        "--output-dir",
        str(tmp_path / "out"),
        credential_flag,
        "must-not-be-accepted",
    ]

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


def test_onshape_compatible_secret_environment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONSHAPE_API_KEY", "access-key")
    monkeypatch.delenv("ONSHAPE_API_SECRET", raising=False)
    monkeypatch.setenv("ONSHAPE_SECRET", "secret-key")

    assert cli._onshape_api_credentials() == ("access-key", "secret-key")


def test_onshape_secret_environment_names_must_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONSHAPE_API_KEY", "access-key")
    monkeypatch.setenv("ONSHAPE_API_SECRET", "secret-one")
    monkeypatch.setenv("ONSHAPE_SECRET", "secret-two")

    with pytest.raises(RuntimeError, match="disagree"):
        cli._onshape_api_credentials()


def test_onshape_workspace_export_requires_an_explicit_snapshot_name(
    tmp_path: Path,
) -> None:
    args = cli.build_parser().parse_args(
        [
            "export-onshape",
            "--document-id",
            "1" * 24,
            "--workspace-id",
            "4" * 24,
            "--element-id",
            "3" * 24,
            "--element-kind",
            "partstudio",
            "--rights-assertion",
            "Authorized test export.",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(ValueError, match="requires --snapshot-name"):
        args.handler(args)


def test_onshape_workspace_is_snapshotted_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geometry_authoring_connectors

    observed: dict[str, Any] = {}

    class FakeOnshapeConnector:
        def __init__(self, **credentials: str) -> None:
            observed["credential_names"] = sorted(credentials)

        def create_immutable_version(self, request: Any) -> str:
            observed["snapshot"] = request
            return "2" * 24

        def export(self, request: Any, *, output_dir: Path) -> None:
            observed["export"] = request
            observed["output_dir"] = output_dir
            raise RuntimeError("stop after export request")

    monkeypatch.setenv("ONSHAPE_API_KEY", "access-key")
    monkeypatch.setenv("ONSHAPE_API_SECRET", "secret-key")
    monkeypatch.delenv("ONSHAPE_SECRET", raising=False)
    monkeypatch.setattr(
        geometry_authoring_connectors,
        "OnshapeConnector",
        FakeOnshapeConnector,
    )
    args = cli.build_parser().parse_args(
        [
            "export-onshape",
            "--document-id",
            "1" * 24,
            "--workspace-id",
            "4" * 24,
            "--snapshot-name",
            "Geometry Agent test export",
            "--element-id",
            "3" * 24,
            "--element-kind",
            "partstudio",
            "--rights-assertion",
            "Authorized test export.",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(RuntimeError, match="stop after export request"):
        args.handler(args)

    assert observed["credential_names"] == ["api_access_key", "api_secret_key"]
    assert observed["snapshot"].workspace_id == "4" * 24
    assert observed["snapshot"].version_name == "Geometry Agent test export"
    assert observed["export"].version_id == "2" * 24


def test_onshape_export_validates_element_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geometry_authoring_connectors

    snapshot_created = False

    class FakeOnshapeConnector:
        def __init__(self, **_credentials: str) -> None:
            pass

        def create_immutable_version(self, _request: Any) -> str:
            nonlocal snapshot_created
            snapshot_created = True
            return "2" * 24

    monkeypatch.setattr(
        geometry_authoring_connectors,
        "OnshapeConnector",
        FakeOnshapeConnector,
    )
    args = cli.build_parser().parse_args(
        [
            "export-onshape",
            "--document-id",
            "1" * 24,
            "--workspace-id",
            "4" * 24,
            "--snapshot-name",
            "Geometry Agent test export",
            "--element-id",
            "malformed-element-id",
            "--element-kind",
            "partstudio",
            "--rights-assertion",
            "Authorized test export.",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(ValueError, match="element_id"):
        args.handler(args)

    assert snapshot_created is False
