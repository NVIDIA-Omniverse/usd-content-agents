# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import pytest

from geometry_authoring_connectors import (
    ConnectorConfigurationError,
    InvalidProviderResponseError,
    OnshapeConnector,
    OnshapeVersionExportRequest,
    OnshapeWorkspaceSnapshotRequest,
    ProviderTransportError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
    WireArtifact,
    WireSourceBundle,
    canonicalize_materialized_bundle,
)

from .conftest import FakeResponse, QueueTransport

DOCUMENT_ID = "1" * 24
VERSION_ID = "2" * 24
ELEMENT_ID = "3" * 24
TRANSLATION_ID = "translation-1"
EXTERNAL_ID = "external-data-1"
WORKSPACE_ID = "4" * 24
STEP_BYTES = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
GLTF_BYTES = b'{"asset":{"version":"2.0"}}'
OBJ_BYTES = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
MTL_BYTES = b"newmtl Default\nKd 0.8 0.8 0.8\n"


def _connector(**kwargs: Any) -> OnshapeConnector:
    return OnshapeConnector(
        api_access_key="test-access-key",
        api_secret_key="test-secret-key",
        **kwargs,
    )


def _request(
    format_name: Literal["step", "gltf", "obj"] = "step",
) -> OnshapeVersionExportRequest:
    return OnshapeVersionExportRequest(
        document_id=DOCUMENT_ID,
        version_id=VERSION_ID,
        element_id=ELEMENT_ID,
        element_kind="partstudio",
        format=format_name,
        output_filename=f"onshape_part.{format_name}",
    )


def _active() -> dict[str, object]:
    return {"id": TRANSLATION_ID, "requestState": "ACTIVE"}


def _done(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "id": TRANSLATION_ID,
        "requestState": "DONE",
        "documentId": DOCUMENT_ID,
        "resultDocumentId": DOCUMENT_ID,
        "versionId": VERSION_ID,
        "requestElementId": ELEMENT_ID,
        "resultExternalDataIds": [EXTERNAL_ID],
    }
    result.update(overrides)
    return result


def test_workspace_snapshot_creates_an_immutable_version() -> None:
    transport = QueueTransport(
        FakeResponse.json(
            {
                "id": VERSION_ID,
                "documentId": DOCUMENT_ID,
                "workspaceId": WORKSPACE_ID,
            }
        )
    )
    connector = _connector(transport=transport)

    version_id = connector.create_immutable_version(
        OnshapeWorkspaceSnapshotRequest(
            document_id=DOCUMENT_ID,
            workspace_id=WORKSPACE_ID,
            version_name="Geometry Agent export",
        )
    )

    assert version_id == VERSION_ID
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url == (f"https://cad.onshape.com/api/v12/documents/d/{DOCUMENT_ID}/versions")
    assert json.loads(request.kwargs["data"]) == {
        "documentId": DOCUMENT_ID,
        "workspaceId": WORKSPACE_ID,
        "name": "Geometry Agent export",
    }


@pytest.mark.parametrize("version_name", ("invalid\x7f", "invalid\x80"))
def test_workspace_snapshot_rejects_non_printing_names(version_name: str) -> None:
    with pytest.raises(ValueError, match="trimmed printable text"):
        OnshapeWorkspaceSnapshotRequest(
            document_id=DOCUMENT_ID,
            workspace_id=WORKSPACE_ID,
            version_name=version_name,
        )


def test_workspace_snapshot_rejects_identity_drift() -> None:
    connector = _connector(
        transport=QueueTransport(
            FakeResponse.json(
                {
                    "id": VERSION_ID,
                    "documentId": "f" * 24,
                    "workspaceId": WORKSPACE_ID,
                }
            )
        )
    )

    with pytest.raises(InvalidProviderResponseError, match="changed documentId"):
        connector.create_immutable_version(
            OnshapeWorkspaceSnapshotRequest(
                document_id=DOCUMENT_ID,
                workspace_id=WORKSPACE_ID,
                version_name="Geometry Agent export",
            )
        )


def _zip(*members: tuple[str, bytes]) -> bytes:
    destination = io.BytesIO()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, content)
    return destination.getvalue()


def _multipart_gltf(part_name: str, x_position: float) -> bytes:
    buffer = base64.b64encode(b"\x00" * 12).decode("ascii")
    return json.dumps(
        {
            "asset": {"version": "2.0", "generator": "Onshape test export"},
            "extensionsUsed": ["PTC_onshape_metadata"],
            "buffers": [
                {
                    "byteLength": 12,
                    "uri": f"data:application/octet-stream;base64,{buffer}",
                }
            ],
            "bufferViews": [{"buffer": 0, "byteLength": 12}],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 1,
                    "type": "VEC3",
                }
            ],
            "materials": [{"name": f"{part_name} material"}],
            "meshes": [
                {
                    "name": f"{part_name} mesh",
                    "primitives": [{"attributes": {"POSITION": 0}, "material": 0}],
                }
            ],
            "nodes": [
                {
                    "extensions": {"PTC_onshape_metadata": {"entity_type": "Body"}},
                    "mesh": 0,
                    "name": part_name,
                    "translation": [x_position, 0.0, 0.0],
                }
            ],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        },
        separators=(",", ":"),
    ).encode()


def _multipart_obj(part_name: str, material_name: str) -> bytes:
    return (
        f"mtllib {part_name}.mtl\n"
        f"g {part_name}\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vn 0 0 1\n"
        f"usemtl {material_name}\n"
        "f -3//-1 -2//-1 -1//-1\n"
    ).encode()


def test_official_version_export_records_immutable_provenance(tmp_path: Path) -> None:
    transport = QueueTransport(
        FakeResponse.json(_active()),
        FakeResponse.json(_done()),
        FakeResponse(STEP_BYTES, content_type="application/octet-stream"),
    )
    sleeps: list[float] = []
    connector = _connector(
        transport=transport,
        sleeper=sleeps.append,
    )

    bundle = connector.export(_request(), output_dir=tmp_path / "result")

    assert bundle.source_revision == (f"onshape:partstudio:{DOCUMENT_ID}:{VERSION_ID}:{ELEMENT_ID}")
    assert bundle.metadata["immutable_version"] is True
    assert bundle.metadata["document_id"] == DOCUMENT_ID
    assert bundle.metadata["version_id"] == VERSION_ID
    assert bundle.metadata["translation_id"] == TRANSLATION_ID
    assert bundle.verification_assertions[0].assertion_id == "onshape-export-step"
    assert bundle.verification_assertions[0].metrics == bundle.metadata
    assert bundle.artifacts[0].path.read_bytes() == STEP_BYTES
    assert sleeps == [1.0]
    assert [request.method for request in transport.requests] == ["POST", "GET", "GET"]
    assert (
        transport.requests[0].url
        == f"https://cad.onshape.com/api/v12/partstudios/d/{DOCUMENT_ID}/v/{VERSION_ID}/e/{ELEMENT_ID}/export/step"
    )
    assert (
        transport.requests[2].url
        == f"https://cad.onshape.com/api/v12/documents/d/{DOCUMENT_ID}/externaldata/{EXTERNAL_ID}"
    )
    for recorded in transport.requests:
        assert "test-secret-key" not in recorded.url
        assert recorded.kwargs["headers"]["Authorization"].startswith(
            "On test-access-key:HmacSHA256:"
        )
        assert "test-secret-key" not in recorded.kwargs["headers"]["Authorization"]
        assert recorded.kwargs["allow_redirects"] is False
    request_body = json.loads(transport.requests[0].kwargs["data"])
    assert request_body["storeInDocument"] is False
    assert request_body["grouping"] is True
    assert request_body["isYAxisUp"] is False
    assert request_body["stepUnit"] == "METER"


def test_gltf_version_export_uses_y_up_millimeter_mesh_coordinates(
    tmp_path: Path,
) -> None:
    transport = QueueTransport(
        FakeResponse.json(_active()),
        FakeResponse.json(_done()),
        FakeResponse(GLTF_BYTES, content_type="model/gltf+json"),
    )
    connector = _connector(
        transport=transport,
        sleeper=lambda _seconds: None,
    )

    bundle = connector.export(_request("gltf"), output_dir=tmp_path / "result")

    assert bundle.up_axis == "Y"
    assert bundle.forward_axis == "+Z"
    assert bundle.units == "millimeter"
    request_body = json.loads(transport.requests[0].kwargs["data"])
    assert request_body["grouping"] is True
    assert request_body["isYAxisUp"] is True
    assert request_body["meshParams"]["unit"] == "MILLIMETER"


@pytest.mark.parametrize(
    "export_options",
    (
        {"grouping": False},
        {"isYAxisUp": True},
        {"stepUnit": "INCH"},
        {"storeInDocument": True},
        {"meshParams": {"unit": "MILLIMETER"}},
        {"meshParams": "not-an-object"},
    ),
)
def test_export_request_rejects_caller_override_of_package_coordinates(
    export_options: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="connector-controlled|must be an object"):
        OnshapeVersionExportRequest(
            document_id=DOCUMENT_ID,
            version_id=VERSION_ID,
            element_id=ELEMENT_ID,
            element_kind="partstudio",
            format="gltf",
            output_filename="onshape_part.gltf",
            export_options=export_options,
        )


def test_gltf_archive_extracts_entrypoint_and_rewrites_bounded_dependencies(
    tmp_path: Path,
) -> None:
    gltf = json.dumps(
        {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 4, "uri": "CONNECTING ROD.bin"}],
        },
        separators=(",", ":"),
    ).encode()
    connector = _connector(
        transport=QueueTransport(
            FakeResponse.json(_done()),
            FakeResponse(
                _zip(
                    ("CONNECTING ROD.gltf", gltf),
                    ("CONNECTING ROD.bin", b"mesh"),
                ),
                content_type="application/zip",
            ),
        ),
    )

    bundle = connector.export(_request("gltf"), output_dir=tmp_path / "result")

    assert [item.filename for item in bundle.artifacts] == [
        "onshape_part.gltf",
        "onshape_part-resource-001.bin",
    ]
    assert [item.role for item in bundle.artifacts] == [
        "render_geometry",
        "supporting_asset",
    ]
    document = json.loads(bundle.artifacts[0].path.read_text())
    assert document["buffers"][0]["uri"] == "onshape_part-resource-001.bin"
    assert bundle.artifacts[1].path.read_bytes() == b"mesh"
    assert bundle.metadata["archive_members"] == [
        "CONNECTING ROD.bin",
        "CONNECTING ROD.gltf",
    ]
    assert bundle.metadata["supporting_archive_members"] == ["CONNECTING ROD.bin"]

    canonical = canonicalize_materialized_bundle(
        bundle,
        request_digest="a" * 64,
        rights_assertion="Authorized test export.",
    )
    assert [item.role for item in canonical.representations] == [
        "render_geometry",
        "supporting_asset",
    ]
    assert canonical.representations[1].format == "bin"


def test_multipart_gltf_archive_merges_part_documents_into_one_scene(
    tmp_path: Path,
) -> None:
    members = tuple(
        (f"Part Studio - Part {index:02d}.gltf", _multipart_gltf(f"Part {index}", index))
        for index in range(40)
    )
    connector = _connector(
        transport=QueueTransport(
            FakeResponse.json(_done()),
            FakeResponse(_zip(*members), content_type="application/zip"),
        ),
    )

    bundle = connector.export(_request("gltf"), output_dir=tmp_path / "result")

    assert [item.filename for item in bundle.artifacts] == ["onshape_part.gltf"]
    document = json.loads(bundle.artifacts[0].path.read_text())
    assert len(document["buffers"]) == 40
    assert len(document["bufferViews"]) == 40
    assert len(document["accessors"]) == 40
    assert len(document["materials"]) == 40
    assert len(document["meshes"]) == 40
    assert len(document["nodes"]) == 40
    assert document["scenes"][0]["nodes"] == list(range(40))
    assert document["bufferViews"][39]["buffer"] == 39
    assert document["accessors"][39]["bufferView"] == 39
    assert document["meshes"][39]["primitives"][0]["attributes"]["POSITION"] == 39
    assert document["meshes"][39]["primitives"][0]["material"] == 39
    assert document["nodes"][39]["mesh"] == 39
    assert bundle.metadata["supporting_archive_members"] == []


def test_obj_archive_extracts_geometry_and_material_library(tmp_path: Path) -> None:
    obj = b"mtllib CONNECTING ROD.mtl\n" + OBJ_BYTES
    connector = _connector(
        transport=QueueTransport(
            FakeResponse.json(_done()),
            FakeResponse(
                _zip(
                    ("CONNECTING ROD.obj", obj),
                    ("CONNECTING ROD.mtl", MTL_BYTES),
                ),
                content_type="application/zip",
            ),
        ),
    )

    bundle = connector.export(_request("obj"), output_dir=tmp_path / "result")

    assert [item.filename for item in bundle.artifacts] == [
        "onshape_part.obj",
        "onshape_part-material-001.mtl",
    ]
    assert [item.role for item in bundle.artifacts] == [
        "mesh_geometry",
        "supporting_asset",
    ]
    assert bundle.artifacts[0].path.read_text().startswith("mtllib onshape_part-material-001.mtl\n")
    assert bundle.artifacts[1].path.read_bytes() == MTL_BYTES
    assert bundle.metadata["supporting_archive_members"] == ["CONNECTING ROD.mtl"]


def test_multipart_obj_archive_merges_parts_and_rebases_indices(tmp_path: Path) -> None:
    members: list[tuple[str, bytes]] = []
    for index in range(40):
        part_name = f"Part Studio - Part {index:02d}"
        material_name = f"material {index:02d}"
        members.extend(
            (
                (f"{part_name}.obj", _multipart_obj(part_name, material_name)),
                (
                    f"{part_name}.mtl",
                    f"newmtl{' ' if index else chr(9)}{material_name}\nKd 0.8 0.8 0.8\n".encode(),
                ),
            )
        )
    connector = _connector(
        transport=QueueTransport(
            FakeResponse.json(_done()),
            FakeResponse(_zip(*members), content_type="application/zip"),
        ),
    )

    bundle = connector.export(_request("obj"), output_dir=tmp_path / "result")

    assert [item.filename for item in bundle.artifacts] == [
        "onshape_part.obj",
        "onshape_part-material-001.mtl",
    ]
    obj = bundle.artifacts[0].path.read_text()
    material = bundle.artifacts[1].path.read_text()
    assert obj.count("mtllib ") == 1
    assert "f 1//1 2//1 3//1" in obj
    assert "f 4//2 5//2 6//2" in obj
    assert "f 118//40 119//40 120//40" in obj
    assert material.count("newmtl ") == 40
    assert len(bundle.metadata["supporting_archive_members"]) == 40


@pytest.mark.parametrize(
    ("format_name", "archive", "message"),
    (
        (
            "gltf",
            _zip(
                ("one.gltf", _multipart_gltf("one", 0.0)),
                (
                    "two.gltf",
                    b'{"asset":{"version":"2.0"},"buffers":'
                    b'[{"byteLength":4,"uri":"external.bin"}],'
                    b'"meshes":[{}],"nodes":[{}],"scenes":[{"nodes":[0]}]}',
                ),
            ),
            "bounded data URIs",
        ),
        (
            "gltf",
            _zip(
                ("one.gltf", _multipart_gltf("one", 0.0)),
                (
                    "two.gltf",
                    _multipart_gltf("two", 1.0).replace(
                        b'"PTC_onshape_metadata"',
                        b'"unsupported_extension"',
                    ),
                ),
            ),
            "unsupported extension",
        ),
        (
            "gltf",
            _zip(
                ("one.gltf", _multipart_gltf("one", 0.0)),
                (
                    "two.gltf",
                    _multipart_gltf("two", 1.0).replace(
                        b'"byteLength":12',
                        b'"byteLength":11',
                    ),
                ),
            ),
            "buffer length is inconsistent",
        ),
        (
            "gltf",
            _zip(("model.gltf", GLTF_BYTES), ("unused.bin", b"data")),
            "unreferenced supporting artifact",
        ),
        (
            "obj",
            _zip(("model.obj", OBJ_BYTES), ("unused.mtl", MTL_BYTES)),
            "unreferenced material library",
        ),
        (
            "obj",
            _zip(
                ("model.obj", b"mtllib material.mtl\n" + OBJ_BYTES),
                ("material.mtl", b"newmtl x\nmap_Kd external.png\n"),
            ),
            "unsupported texture dependency",
        ),
        (
            "obj",
            _zip(
                ("model.obj", b"mtllib\tmaterial.mtl\n" + OBJ_BYTES),
                ("material.mtl", b"newmtl x\nmap_Pr roughness.png\n"),
            ),
            "unsupported texture dependency",
        ),
        (
            "obj",
            _zip(
                (
                    "one.obj",
                    _multipart_obj("one", "one").replace(b"-1//-1", b"-4//-1"),
                ),
                ("one.mtl", b"newmtl one\nKd 0.8 0.8 0.8\n"),
                ("two.obj", _multipart_obj("two", "two")),
                ("two.mtl", b"newmtl two\nKd 0.8 0.8 0.8\n"),
            ),
            "out-of-range vertex index",
        ),
        (
            "obj",
            _zip(("nested/model.obj", OBJ_BYTES)),
            "unsafe or unsupported member",
        ),
    ),
)
def test_mesh_archives_reject_ambiguous_or_incomplete_bundles(
    tmp_path: Path,
    format_name: Literal["gltf", "obj"],
    archive: bytes,
    message: str,
) -> None:
    connector = _connector(
        transport=QueueTransport(
            FakeResponse.json(_done()),
            FakeResponse(archive, content_type="application/zip"),
        ),
    )

    with pytest.raises(InvalidProviderResponseError, match=message):
        connector.export(_request(format_name), output_dir=tmp_path / "result")


def test_supporting_asset_role_rejects_executable_formats() -> None:
    geometry = WireArtifact.from_bytes(
        filename="geometry.step",
        role="cad_geometry",
        media_type="model/step",
        content=STEP_BYTES,
    )
    executable = WireArtifact.from_bytes(
        filename="helper.py",
        role="supporting_asset",
        media_type="text/x-python",
        content=b"print('not executed')\n",
    )

    with pytest.raises(InvalidProviderResponseError, match="supporting-asset format"):
        WireSourceBundle(
            provider_id="onshape-official-api",
            provider_version="v12",
            source_revision="immutable-version",
            units="meter",
            up_axis="Z",
            artifacts=(geometry, executable),
        )


def test_supporting_asset_role_accepts_bounded_standard_json() -> None:
    geometry = WireArtifact.from_bytes(
        filename="geometry.step",
        role="cad_geometry",
        media_type="model/step",
        content=STEP_BYTES,
    )
    parameters = WireArtifact.from_bytes(
        filename="parameters.json",
        role="supporting_asset",
        media_type="application/json",
        content=b'{"width_mm": 42.0, "enabled": true}',
    )

    bundle = WireSourceBundle(
        provider_id="fixture-provider",
        provider_version="v1",
        source_revision="immutable-version",
        units="millimeter",
        up_axis="Z",
        artifacts=(geometry, parameters),
    )

    assert bundle.artifacts[1].filename == "parameters.json"


def _supporting_json_bundle(content: bytes) -> WireSourceBundle:
    geometry = WireArtifact.from_bytes(
        filename="geometry.step",
        role="cad_geometry",
        media_type="model/step",
        content=STEP_BYTES,
    )
    parameters = WireArtifact.from_bytes(
        filename="parameters.json",
        role="supporting_asset",
        media_type="application/json",
        content=content,
    )
    return WireSourceBundle(
        provider_id="fixture-provider",
        provider_version="v1",
        source_revision="immutable-version",
        units="millimeter",
        up_axis="Z",
        artifacts=(geometry, parameters),
    )


def test_supporting_asset_role_rejects_unsafe_json() -> None:
    for content in (
        b"not-json",
        b"NaN",
        b'"scalar"',
        b'{"overflow":1e400}',
        b'{"duplicate":1,"duplicate":2}',
    ):
        with pytest.raises(
            InvalidProviderResponseError,
            match="JSON supporting asset",
        ):
            _supporting_json_bundle(content)


def test_supporting_asset_role_enforces_json_resource_limits() -> None:
    cases = (
        (b'{"value":"' + b"x" * (4 * 1024 * 1024) + b'"}', "4-MiB"),
        (b"[" * 34 + b"]" * 34, "nesting-depth"),
        (b'{"' + b"k" * 1_025 + b'":null}', "oversized object key"),
        (b"[" + b",".join([b"null"] * 100_001) + b"]", "value-count"),
    )
    for content, message in cases:
        with pytest.raises(InvalidProviderResponseError, match=message):
            _supporting_json_bundle(content)


def test_raw_gltf_rejects_non_finite_json_constants(tmp_path: Path) -> None:
    connector = _connector(
        transport=QueueTransport(
            FakeResponse.json(_done()),
            FakeResponse(
                b'{"asset":{"version":"2.0"},"extras":{"invalid":NaN}}',
                content_type="model/gltf+json",
            ),
        ),
    )

    with pytest.raises(InvalidProviderResponseError, match="UTF-8 JSON"):
        connector.export(_request("gltf"), output_dir=tmp_path / "result")


def test_api_key_export_signs_every_request_without_leaking_credentials(
    tmp_path: Path,
) -> None:
    transport = QueueTransport(
        FakeResponse.json(_active()),
        FakeResponse.json(_done()),
        FakeResponse(STEP_BYTES, content_type="application/octet-stream"),
    )
    access_key = "public-access-key"
    secret_key = "private-secret-key"
    auth_date = "Sat, 22 Aug 2026 12:00:00 GMT"
    nonces = iter(
        (
            "nonce000000000001",
            "nonce000000000002",
            "nonce000000000003",
        )
    )
    connector = OnshapeConnector(
        api_access_key=access_key,
        api_secret_key=secret_key,
        transport=transport,
        sleeper=lambda _seconds: None,
        request_date_factory=lambda: auth_date,
        request_nonce_factory=lambda: next(nonces),
    )

    output_dir = tmp_path / "result"
    connector.export(_request(), output_dir=output_dir)

    for index, recorded in enumerate(transport.requests, start=1):
        headers = recorded.kwargs["headers"]
        nonce = f"nonce{index:012d}"
        parsed = urlparse(recorded.url)
        content_type = headers.get("Content-Type", "")
        signing_input = (
            f"{recorded.method}\n{nonce}\n{auth_date}\n{content_type}\n"
            f"{parsed.path}\n{parsed.query}\n"
        ).lower()
        expected = base64.b64encode(
            hmac.new(
                secret_key.encode(),
                signing_input.encode(),
                hashlib.sha256,
            ).digest()
        ).decode()
        assert headers["Authorization"] == (f"On {access_key}:HmacSHA256:{expected}")
        assert headers["Date"] == auth_date
        assert headers["On-Nonce"] == nonce
        assert secret_key not in recorded.url
        assert secret_key not in headers["Authorization"]

    for path in output_dir.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert access_key.encode() not in content
            assert secret_key.encode() not in content


def test_single_member_step_archive_is_safely_expanded(
    tmp_path: Path,
) -> None:
    transport = QueueTransport(
        FakeResponse.json(_done()),
        FakeResponse(
            _zip(
                ("FLYWHEEL.step", STEP_BYTES),
            ),
            content_type="application/zip",
        ),
    )
    connector = _connector(
        transport=transport,
    )

    bundle = connector.export(_request(), output_dir=tmp_path / "result")

    assert [artifact.filename for artifact in bundle.artifacts] == ["onshape_part.step"]
    assert [artifact.path.read_bytes() for artifact in bundle.artifacts] == [STEP_BYTES]
    assert bundle.metadata["archive_members"] == ["FLYWHEEL.step"]


@pytest.mark.parametrize(
    "archive",
    (
        _zip(("one.step", STEP_BYTES), ("two.step", STEP_BYTES)),
        _zip(("nested/part.step", STEP_BYTES)),
        _zip(("notes.txt", b"not geometry")),
        _zip(
            (
                "compressed.step",
                b"ISO-10303-21;\n" + b" " * (1024 * 1024) + b"END-ISO-10303-21;",
            )
        ),
    ),
    ids=("multiple-step", "nested-path", "non-step", "compression-ratio"),
)
def test_step_archive_rejects_unsafe_or_abusive_members(
    tmp_path: Path,
    archive: bytes,
) -> None:
    connector = _connector(
        transport=QueueTransport(
            FakeResponse.json(_done()),
            FakeResponse(archive, content_type="application/zip"),
        ),
    )

    with pytest.raises(InvalidProviderResponseError, match="Onshape"):
        connector.export(_request(), output_dir=tmp_path / "result")


@pytest.mark.parametrize(
    "credentials",
    (
        {"api_access_key": "", "api_secret_key": "secret-key"},
        {"api_access_key": "access-key", "api_secret_key": ""},
        {"api_access_key": " access-key", "api_secret_key": "secret-key"},
        {"api_access_key": "access-key", "api_secret_key": "secret\nkey"},
    ),
)
def test_onshape_rejects_invalid_api_key_authentication(
    credentials: dict[str, str],
) -> None:
    with pytest.raises(ConnectorConfigurationError):
        OnshapeConnector(**credentials)


@pytest.mark.parametrize(
    "api_base_url",
    (
        "https://api.example.com/api/v12",
        "http://cad.onshape.com/api/v12",
        "https://user:secret@cad.onshape.com/api/v12",
        "https://cad.onshape.com/api/v12?token=secret",
        "https://cad.onshape.com/not-api/v12",
    ),
)
def test_onshape_requires_credential_free_official_api_url(api_base_url: str) -> None:
    with pytest.raises(ConnectorConfigurationError):
        _connector(api_base_url=api_base_url)


def test_onshape_rejects_redirect_instead_of_forwarding_api_key(tmp_path: Path) -> None:
    transport = QueueTransport(
        FakeResponse(
            b"",
            status_code=307,
            headers={"Location": "https://storage.example/export.step"},
        )
    )
    connector = _connector(transport=transport)
    with pytest.raises(ProviderTransportError, match="redirects are not permitted"):
        connector.export(_request(), output_dir=tmp_path)
    assert len(transport.requests) == 1


def test_onshape_rejects_version_drift(tmp_path: Path) -> None:
    transport = QueueTransport(
        FakeResponse.json(_done(versionId="4" * 24)),
    )
    connector = _connector(transport=transport)
    with pytest.raises(InvalidProviderResponseError, match="immutable versionId"):
        connector.export(_request(), output_dir=tmp_path)


def test_onshape_rejects_ambiguous_external_artifacts(tmp_path: Path) -> None:
    transport = QueueTransport(
        FakeResponse.json(_done(resultExternalDataIds=["one", "two"])),
    )
    connector = _connector(transport=transport)
    with pytest.raises(InvalidProviderResponseError, match="exactly one"):
        connector.export(_request(), output_dir=tmp_path)


def test_onshape_polling_is_bounded(tmp_path: Path) -> None:
    connector = _connector(
        transport=QueueTransport(FakeResponse.json(_active())),
        max_poll_attempts=1,
        sleeper=lambda _seconds: pytest.fail("bounded single attempt must not sleep"),
    )
    with pytest.raises(ProviderUnavailableError, match="bounded poll window"):
        connector.export(_request(), output_dir=tmp_path)


def test_onshape_default_poll_budget_allows_slow_translations(tmp_path: Path) -> None:
    sleeps: list[float] = []
    transport = QueueTransport(
        *(FakeResponse.json(_active()) for _ in range(21)),
        FakeResponse.json(_done()),
        FakeResponse(STEP_BYTES, content_type="application/octet-stream"),
    )
    connector = _connector(
        transport=transport,
        sleeper=sleeps.append,
    )

    bundle = connector.export(_request(), output_dir=tmp_path / "result")

    assert bundle.artifacts[0].path.read_bytes() == STEP_BYTES
    assert len(transport.requests) == 23
    assert sleeps[:4] == [1.0, 2.0, 4.0, 8.0]
    assert max(sleeps) == 10.0
    assert sum(sleeps) > 120.0


def test_onshape_does_not_claim_geometry_generation_or_revision() -> None:
    connector = _connector(
        transport=QueueTransport(),
    )
    with pytest.raises(UnsupportedCapabilityError):
        connector.generate()
    with pytest.raises(UnsupportedCapabilityError):
        connector.revise()
    assert connector.capabilities().text is False
    assert connector.capabilities().revision is False
