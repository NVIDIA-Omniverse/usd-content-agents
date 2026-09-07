# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API-key-authenticated export of immutable Onshape geometry."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import math
import re
import secrets
import stat
import time
import zipfile
from collections.abc import Callable
from email.utils import formatdate
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._artifacts import materialize_wire_bundle
from ._formats import SUPPORTING_ASSET_MEDIA_TYPES
from ._http import BoundedHttpClient, HttpTransport
from .errors import (
    ConnectorConfigurationError,
    InvalidProviderResponseError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from .models import (
    MAX_OUTPUT_ARTIFACT_BYTES,
    MAX_SOURCE_BUNDLE_BYTES,
    SAFE_FILENAME_PATTERN,
    AuthoringCapabilities,
    MaterializedWireSourceBundle,
    WireArtifact,
    WireSourceBundle,
    WireVerificationAssertion,
    _validate_json_value,
)

ONSHAPE_PROVIDER_ID = "onshape-official-api"
ONSHAPE_EXPORT_REQUEST_SCHEMA_VERSION: Literal[
    "geometry-authoring-connectors.onshape-version-export.v1"
] = "geometry-authoring-connectors.onshape-version-export.v1"
ONSHAPE_WORKSPACE_SNAPSHOT_REQUEST_SCHEMA_VERSION: Literal[
    "geometry-authoring-connectors.onshape-workspace-snapshot.v1"
] = "geometry-authoring-connectors.onshape-workspace-snapshot.v1"
_ONSHAPE_ID_RE = re.compile(r"^[0-9a-f]{24}$")
_TRANSLATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_FORMAT_SUFFIX = {"step": ".step", "gltf": ".gltf", "obj": ".obj"}
_FORMAT_MEDIA_TYPE = {
    "step": "model/step",
    "gltf": "model/gltf+json",
    "obj": "model/obj",
}
_FORMAT_UNITS: dict[
    Literal["step", "gltf", "obj"],
    Literal["meter", "millimeter"],
] = {
    "step": "meter",
    "gltf": "millimeter",
    "obj": "millimeter",
}
_FORMAT_UP_AXIS: dict[Literal["step", "gltf", "obj"], Literal["Y", "Z"]] = {
    "step": "Z",
    "gltf": "Y",
    "obj": "Z",
}
_FORMAT_FORWARD_AXIS: dict[Literal["step", "gltf", "obj"], Literal["+Y", "+Z"]] = {
    "step": "+Y",
    "gltf": "+Z",
    "obj": "+Y",
}
_MAX_ARCHIVE_COMPRESSION_RATIO = 200.0
_MAX_ONSHAPE_ARCHIVE_MEMBERS = 256
_ARCHIVE_READ_CHUNK_BYTES = 64 * 1024
_GLTF_DEPENDENCY_MEDIA_TYPES = {
    suffix: media_type
    for suffix, media_type in SUPPORTING_ASSET_MEDIA_TYPES.items()
    if suffix != ".mtl"
}
_OBJ_MATERIAL_LIBRARY_DIRECTIVE = re.compile(r"^mtllib\s+", re.IGNORECASE)
_MTL_MATERIAL_DIRECTIVE = re.compile(r"^newmtl\s+", re.IGNORECASE)
_MTL_EXTERNAL_ASSET_DIRECTIVE = re.compile(
    r"^(?:map_[A-Za-z0-9_]+|bump|disp|decal|refl|norm)\s+",
    re.IGNORECASE,
)
_MULTIPART_GLTF_EXTENSIONS = frozenset({"PTC_onshape_metadata"})
_GLTF_EMBEDDED_BUFFER_PREFIX = "data:application/octet-stream;base64,"


class OnshapeVersionExportRequest(BaseModel):
    """Export request bound to an immutable Onshape document version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["geometry-authoring-connectors.onshape-version-export.v1"] = (
        ONSHAPE_EXPORT_REQUEST_SCHEMA_VERSION
    )
    document_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    version_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    element_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    element_kind: Literal["partstudio", "assembly"]
    format: Literal["step", "gltf", "obj"]
    output_filename: str = Field(pattern=SAFE_FILENAME_PATTERN)
    export_options: dict[str, Any] = Field(default_factory=dict, max_length=128)

    @field_validator("export_options")
    @classmethod
    def validate_export_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_value(value)
        controlled = {"grouping", "isYAxisUp", "stepUnit", "storeInDocument"}
        mesh_parameters = value.get("meshParams")
        if controlled.intersection(value) or (
            isinstance(mesh_parameters, dict) and "unit" in mesh_parameters
        ):
            raise ValueError(
                "Onshape storage, grouping, axis, and unit options are connector-controlled"
            )
        if mesh_parameters is not None and not isinstance(mesh_parameters, dict):
            raise ValueError("Onshape meshParams export option must be an object")
        return value

    @model_validator(mode="after")
    def validate_filename(self) -> Self:
        if Path(self.output_filename).suffix.lower() != _FORMAT_SUFFIX[self.format]:
            raise ValueError("Onshape output filename extension must match the export format")
        return self


class OnshapeWorkspaceSnapshotRequest(BaseModel):
    """Request an immutable named version from one mutable workspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["geometry-authoring-connectors.onshape-workspace-snapshot.v1"] = (
        ONSHAPE_WORKSPACE_SNAPSHOT_REQUEST_SCHEMA_VERSION
    )
    document_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    workspace_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    version_name: str = Field(min_length=1, max_length=128)

    @field_validator("version_name")
    @classmethod
    def validate_version_name(cls, value: str) -> str:
        if value != value.strip() or not value.isprintable():
            raise ValueError("Onshape snapshot name must be trimmed printable text")
        return value


def onshape_source_revision(request: OnshapeVersionExportRequest) -> str:
    """Return the complete immutable identity retained by Geometry Agent."""

    return (
        f"onshape:{request.element_kind}:{request.document_id}:"
        f"{request.version_id}:{request.element_id}"
    )


def _response_id(
    value: Any,
    *,
    label: str,
    pattern: re.Pattern[str] = _TRANSLATION_ID_RE,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise InvalidProviderResponseError(f"Onshape returned an invalid {label}")
    return value


def _request_date() -> str:
    return formatdate(usegmt=True)


def _request_nonce() -> str:
    return secrets.token_hex(16)


class _OnshapeApiKeySigner:
    """Create one Onshape HMAC authorization header set per HTTP request."""

    def __init__(
        self,
        *,
        access_key: str,
        secret_key: str,
        date_factory: Callable[[], str],
        nonce_factory: Callable[[], str],
    ) -> None:
        if not access_key or not secret_key:
            raise ConnectorConfigurationError(
                "Onshape API access key and secret key are both required"
            )
        for value in (access_key, secret_key):
            if (
                len(value) > 4096
                or not value.isascii()
                or value != value.strip()
                or any(ord(character) < 33 or ord(character) == 127 for character in value)
            ):
                raise ConnectorConfigurationError("Onshape API credentials are invalid")
        self._access_key = access_key
        self._secret_key = secret_key
        self._date_factory = date_factory
        self._nonce_factory = nonce_factory

    def __call__(self, method: str, url: str, content_type: str) -> dict[str, str]:
        nonce = self._nonce_factory()
        auth_date = self._date_factory()
        if re.fullmatch(r"[A-Za-z0-9]{16,128}", nonce) is None:
            raise ConnectorConfigurationError(
                "Onshape API nonce must be 16 to 128 alphanumeric characters"
            )
        try:
            auth_date.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ConnectorConfigurationError("Onshape API date must be ASCII") from exc
        if not auth_date or any(character in "\r\n" for character in auth_date):
            raise ConnectorConfigurationError("Onshape API date is invalid")
        parsed = urlparse(url)
        signing_input = (
            f"{method}\n{nonce}\n{auth_date}\n{content_type}\n{parsed.path}\n{parsed.query}\n"
        ).lower()
        signature = base64.b64encode(
            hmac.new(
                self._secret_key.encode("utf-8"),
                signing_input.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("ascii")
        return {
            "Authorization": f"On {self._access_key}:HmacSHA256:{signature}",
            "Date": auth_date,
            "On-Nonce": nonce,
        }


def _read_archive_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    label: str,
) -> bytes:
    result = bytearray()
    try:
        with archive.open(info, "r") as stream:
            while True:
                chunk = stream.read(_ARCHIVE_READ_CHUNK_BYTES)
                if not chunk:
                    break
                result.extend(chunk)
                if len(result) > MAX_OUTPUT_ARTIFACT_BYTES:
                    raise InvalidProviderResponseError(
                        f"Onshape {label} archive member exceeds the byte limit"
                    )
    except InvalidProviderResponseError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise InvalidProviderResponseError(
            f"Onshape {label} archive member could not be read"
        ) from exc
    if len(result) != info.file_size:
        raise InvalidProviderResponseError(
            f"Onshape {label} archive member differs from its declared size"
        )
    return bytes(result)


def _archive_members(
    content: bytes,
    *,
    label: str,
    allowed_suffixes: frozenset[str],
) -> tuple[tuple[str, bytes], ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members or len(members) > _MAX_ONSHAPE_ARCHIVE_MEMBERS:
                raise InvalidProviderResponseError(
                    f"Onshape {label} archive has an invalid member count"
                )
            observed_names: set[str] = set()
            total_size = 0
            for info in members:
                member_path = PurePosixPath(info.filename)
                normalized_name = info.filename.casefold()
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(unix_mode)
                if (
                    member_path.name != info.filename
                    or "\\" in info.filename
                    or any(
                        ord(character) < 0x20 or ord(character) == 0x7F
                        for character in info.filename
                    )
                    or normalized_name in observed_names
                    or member_path.suffix.lower() not in allowed_suffixes
                    or info.flag_bits & 0x1
                    or file_type not in {0, stat.S_IFREG}
                ):
                    raise InvalidProviderResponseError(
                        f"Onshape {label} archive contains an unsafe or unsupported member"
                    )
                observed_names.add(normalized_name)
                if info.file_size <= 0 or info.file_size > MAX_OUTPUT_ARTIFACT_BYTES:
                    raise InvalidProviderResponseError(
                        f"Onshape {label} archive member has an invalid size"
                    )
                total_size += info.file_size
                if total_size > MAX_SOURCE_BUNDLE_BYTES:
                    raise InvalidProviderResponseError(
                        f"Onshape {label} archive exceeds the aggregate byte limit"
                    )
                if info.file_size / max(1, info.compress_size) > _MAX_ARCHIVE_COMPRESSION_RATIO:
                    raise InvalidProviderResponseError(
                        f"Onshape {label} archive exceeds the compression-ratio limit"
                    )
            return tuple(
                (
                    info.filename,
                    _read_archive_member(archive, info, label=label),
                )
                for info in sorted(members, key=lambda item: item.filename.casefold())
            )
    except InvalidProviderResponseError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise InvalidProviderResponseError(f"Onshape returned an invalid {label} archive") from exc


def _step_artifacts(
    content: bytes,
    *,
    output_filename: str,
) -> tuple[tuple[WireArtifact, ...], tuple[str, ...]]:
    if not content.startswith(b"PK\x03\x04"):
        return (
            WireArtifact.from_bytes(
                filename=output_filename,
                role="cad_geometry",
                media_type=_FORMAT_MEDIA_TYPE["step"],
                content=content,
            ),
        ), ()
    members = _archive_members(
        content,
        label="STEP",
        allowed_suffixes=frozenset({".step", ".stp"}),
    )
    if len(members) != 1:
        raise InvalidProviderResponseError(
            "Onshape grouped STEP export returned multiple archive members"
        )
    return (
        WireArtifact.from_bytes(
            filename=output_filename,
            role="cad_geometry",
            media_type=_FORMAT_MEDIA_TYPE["step"],
            content=members[0][1],
        ),
    ), (members[0][0],)


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _gltf_document(content: bytes) -> dict[str, Any]:
    try:
        document = json.loads(
            content.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise InvalidProviderResponseError("Onshape glTF export must be UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise InvalidProviderResponseError("Onshape glTF export must contain a JSON object")
    return document


def _gltf_object_collection(document: dict[str, Any], name: str) -> list[dict[str, Any]]:
    value = document.get(name, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise InvalidProviderResponseError(f"Onshape glTF {name} must be an array of objects")
    return value


def _shift_gltf_index(
    container: dict[str, Any],
    key: str,
    *,
    offset: int,
    source_count: int,
    label: str,
    required: bool = True,
) -> None:
    if key not in container and not required:
        return
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < source_count:
        raise InvalidProviderResponseError(
            f"Onshape multipart glTF contains an invalid {label} index"
        )
    container[key] = value + offset


def _merge_gltf_documents(
    primaries: list[tuple[str, bytes]],
) -> bytes:
    if sum(len(content) for _name, content in primaries) > MAX_OUTPUT_ARTIFACT_BYTES:
        raise InvalidProviderResponseError(
            "Onshape multipart glTF exceeds the merged-artifact byte limit"
        )
    merged: dict[str, Any] = {
        "asset": {"version": "2.0"},
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
        "materials": [],
        "meshes": [],
        "nodes": [],
        "scenes": [{"name": "Onshape multipart export", "nodes": []}],
        "scene": 0,
    }
    extensions_used: set[str] = set()
    extensions_required: set[str] = set()
    asset_metadata: dict[str, Any] | None = None

    for member_name, content in primaries:
        document = _gltf_document(content)
        asset = document.get("asset")
        if not isinstance(asset, dict) or asset.get("version") != "2.0":
            raise InvalidProviderResponseError(
                "Onshape multipart glTF requires glTF 2.0 asset metadata"
            )
        if asset_metadata is None:
            asset_metadata = dict(asset)
            merged["asset"] = asset_metadata
        for unsupported in (
            "animations",
            "cameras",
            "images",
            "samplers",
            "skins",
            "textures",
        ):
            if document.get(unsupported):
                raise InvalidProviderResponseError(
                    f"Onshape multipart glTF contains unsupported {unsupported}"
                )
        for field, destination in (
            ("extensionsUsed", extensions_used),
            ("extensionsRequired", extensions_required),
        ):
            values = document.get(field, [])
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise InvalidProviderResponseError(
                    f"Onshape multipart glTF {field} must be an array of strings"
                )
            if not set(values) <= _MULTIPART_GLTF_EXTENSIONS:
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF contains an unsupported extension"
                )
            destination.update(values)
        if document.get("extensions"):
            raise InvalidProviderResponseError(
                "Onshape multipart glTF contains unsupported top-level extensions"
            )

        buffers = _gltf_object_collection(document, "buffers")
        buffer_views = _gltf_object_collection(document, "bufferViews")
        accessors = _gltf_object_collection(document, "accessors")
        materials = _gltf_object_collection(document, "materials")
        meshes = _gltf_object_collection(document, "meshes")
        nodes = _gltf_object_collection(document, "nodes")
        scenes = _gltf_object_collection(document, "scenes")
        if not buffers or not meshes or not nodes or not scenes:
            raise InvalidProviderResponseError(
                "Onshape multipart glTF member is missing renderable scene data"
            )
        for buffer in buffers:
            uri = buffer.get("uri")
            byte_length = buffer.get("byteLength")
            if (
                not isinstance(uri, str)
                or not uri.startswith(_GLTF_EMBEDDED_BUFFER_PREFIX)
                or isinstance(byte_length, bool)
                or not isinstance(byte_length, int)
                or byte_length <= 0
            ):
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF buffers must be bounded data URIs"
                )
            try:
                decoded = base64.b64decode(
                    uri.removeprefix(_GLTF_EMBEDDED_BUFFER_PREFIX),
                    validate=True,
                )
            except (binascii.Error, ValueError) as exc:
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF contains an invalid embedded buffer"
                ) from exc
            if len(decoded) != byte_length:
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF embedded buffer length is inconsistent"
                )
        for label, collection in (
            ("buffer", buffers),
            ("bufferView", buffer_views),
            ("accessor", accessors),
            ("material", materials),
            ("scene", scenes),
        ):
            if any(item.get("extensions") for item in collection):
                raise InvalidProviderResponseError(
                    f"Onshape multipart glTF {label} extensions are unsupported"
                )

        buffer_offset = len(merged["buffers"])
        buffer_view_offset = len(merged["bufferViews"])
        accessor_offset = len(merged["accessors"])
        material_offset = len(merged["materials"])
        mesh_offset = len(merged["meshes"])
        node_offset = len(merged["nodes"])

        for buffer_view in buffer_views:
            _shift_gltf_index(
                buffer_view,
                "buffer",
                offset=buffer_offset,
                source_count=len(buffers),
                label="bufferView buffer",
            )
        for accessor in accessors:
            _shift_gltf_index(
                accessor,
                "bufferView",
                offset=buffer_view_offset,
                source_count=len(buffer_views),
                label="accessor bufferView",
                required=False,
            )
            sparse = accessor.get("sparse")
            if sparse is not None:
                if not isinstance(sparse, dict):
                    raise InvalidProviderResponseError(
                        "Onshape multipart glTF accessor sparse data is invalid"
                    )
                for section in ("indices", "values"):
                    item = sparse.get(section)
                    if not isinstance(item, dict):
                        raise InvalidProviderResponseError(
                            "Onshape multipart glTF accessor sparse data is invalid"
                        )
                    _shift_gltf_index(
                        item,
                        "bufferView",
                        offset=buffer_view_offset,
                        source_count=len(buffer_views),
                        label="sparse accessor bufferView",
                    )
        for mesh in meshes:
            if mesh.get("extensions"):
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF mesh extensions are unsupported"
                )
            primitives = mesh.get("primitives")
            if not isinstance(primitives, list) or not primitives:
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF mesh primitives are invalid"
                )
            for primitive in primitives:
                if not isinstance(primitive, dict) or primitive.get("extensions"):
                    raise InvalidProviderResponseError(
                        "Onshape multipart glTF primitive is unsupported"
                    )
                attributes = primitive.get("attributes")
                if not isinstance(attributes, dict) or not attributes:
                    raise InvalidProviderResponseError(
                        "Onshape multipart glTF primitive attributes are invalid"
                    )
                for attribute in attributes:
                    _shift_gltf_index(
                        attributes,
                        attribute,
                        offset=accessor_offset,
                        source_count=len(accessors),
                        label="primitive attribute accessor",
                    )
                _shift_gltf_index(
                    primitive,
                    "indices",
                    offset=accessor_offset,
                    source_count=len(accessors),
                    label="primitive indices accessor",
                    required=False,
                )
                _shift_gltf_index(
                    primitive,
                    "material",
                    offset=material_offset,
                    source_count=len(materials),
                    label="primitive material",
                    required=False,
                )
                targets = primitive.get("targets", [])
                if not isinstance(targets, list) or any(
                    not isinstance(item, dict) for item in targets
                ):
                    raise InvalidProviderResponseError(
                        "Onshape multipart glTF morph targets are invalid"
                    )
                for target in targets:
                    for attribute in target:
                        _shift_gltf_index(
                            target,
                            attribute,
                            offset=accessor_offset,
                            source_count=len(accessors),
                            label="morph-target accessor",
                        )
        for node in nodes:
            if "camera" in node or "skin" in node:
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF node camera and skin references are unsupported"
                )
            node_extensions = node.get("extensions")
            if node_extensions is not None and (
                not isinstance(node_extensions, dict)
                or not set(node_extensions) <= _MULTIPART_GLTF_EXTENSIONS
                or any(not isinstance(value, dict) for value in node_extensions.values())
            ):
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF node extensions are unsupported"
                )
            _shift_gltf_index(
                node,
                "mesh",
                offset=mesh_offset,
                source_count=len(meshes),
                label="node mesh",
                required=False,
            )
            children = node.get("children", [])
            if not isinstance(children, list):
                raise InvalidProviderResponseError(
                    "Onshape multipart glTF node children are invalid"
                )
            shifted_children: list[int] = []
            for child in children:
                holder = {"child": child}
                _shift_gltf_index(
                    holder,
                    "child",
                    offset=node_offset,
                    source_count=len(nodes),
                    label="node child",
                )
                shifted_children.append(holder["child"])
            if "children" in node:
                node["children"] = shifted_children

        scene_index = document.get("scene", 0)
        scene_holder = {"scene": scene_index}
        _shift_gltf_index(
            scene_holder,
            "scene",
            offset=0,
            source_count=len(scenes),
            label="default scene",
        )
        scene_nodes = scenes[scene_holder["scene"]].get("nodes", [])
        if not isinstance(scene_nodes, list):
            raise InvalidProviderResponseError("Onshape multipart glTF scene nodes are invalid")
        for scene_node in scene_nodes:
            holder = {"node": scene_node}
            _shift_gltf_index(
                holder,
                "node",
                offset=node_offset,
                source_count=len(nodes),
                label=f"scene node in {member_name}",
            )
            merged["scenes"][0]["nodes"].append(holder["node"])

        merged["buffers"].extend(buffers)
        merged["bufferViews"].extend(buffer_views)
        merged["accessors"].extend(accessors)
        merged["materials"].extend(materials)
        merged["meshes"].extend(meshes)
        merged["nodes"].extend(nodes)

    if extensions_used:
        merged["extensionsUsed"] = sorted(extensions_used)
    if extensions_required:
        merged["extensionsRequired"] = sorted(extensions_required)
    normalized = json.dumps(
        merged,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(normalized) > MAX_OUTPUT_ARTIFACT_BYTES:
        raise InvalidProviderResponseError(
            "Onshape multipart glTF exceeds the merged-artifact byte limit"
        )
    return normalized


def _gltf_artifacts(
    content: bytes,
    *,
    output_filename: str,
) -> tuple[tuple[WireArtifact, ...], tuple[str, ...], tuple[str, ...]]:
    if not content.startswith(b"PK\x03\x04"):
        document = _gltf_document(content)
        for collection in ("buffers", "images"):
            values = document.get(collection, [])
            if not isinstance(values, list):
                raise InvalidProviderResponseError(
                    f"Onshape glTF {collection} must be a JSON array"
                )
            for item in values:
                if isinstance(item, dict) and isinstance(item.get("uri"), str):
                    uri = item["uri"]
                    if not uri.startswith("data:"):
                        raise InvalidProviderResponseError(
                            "Onshape raw glTF references an unavailable supporting artifact"
                        )
        return (
            (
                WireArtifact.from_bytes(
                    filename=output_filename,
                    role="render_geometry",
                    media_type=_FORMAT_MEDIA_TYPE["gltf"],
                    content=content,
                ),
            ),
            (),
            (),
        )

    members = _archive_members(
        content,
        label="glTF",
        allowed_suffixes=frozenset({".gltf", *_GLTF_DEPENDENCY_MEDIA_TYPES}),
    )
    primaries = [item for item in members if Path(item[0]).suffix.lower() == ".gltf"]
    if not primaries:
        raise InvalidProviderResponseError(
            "Onshape glTF archive must contain at least one glTF entrypoint"
        )
    dependencies = [item for item in members if Path(item[0]).suffix.lower() != ".gltf"]
    if len(primaries) > 1:
        if dependencies:
            raise InvalidProviderResponseError(
                "Onshape multipart glTF archive cannot mix entrypoints and dependencies"
            )
        normalized = _merge_gltf_documents(primaries)
        return (
            (
                WireArtifact.from_bytes(
                    filename=output_filename,
                    role="render_geometry",
                    media_type=_FORMAT_MEDIA_TYPE["gltf"],
                    content=normalized,
                ),
            ),
            tuple(name for name, _value in members),
            (),
        )
    stem = Path(output_filename).stem
    mapped_dependencies = {
        name: f"{stem}-resource-{index:03d}{Path(name).suffix.lower()}"
        for index, (name, _value) in enumerate(dependencies, start=1)
    }
    document = _gltf_document(primaries[0][1])
    referenced: set[str] = set()
    for collection in ("buffers", "images"):
        values = document.get(collection, [])
        if not isinstance(values, list):
            raise InvalidProviderResponseError(f"Onshape glTF {collection} must be a JSON array")
        for item in values:
            if not isinstance(item, dict) or not isinstance(item.get("uri"), str):
                continue
            uri = item["uri"]
            if uri.startswith("data:"):
                continue
            replacement = mapped_dependencies.get(uri)
            if replacement is None:
                raise InvalidProviderResponseError(
                    "Onshape glTF references an unavailable supporting artifact"
                )
            item["uri"] = replacement
            referenced.add(uri)
    if referenced != set(mapped_dependencies):
        raise InvalidProviderResponseError(
            "Onshape glTF archive contains an unreferenced supporting artifact"
        )
    normalized = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    artifacts = [
        WireArtifact.from_bytes(
            filename=output_filename,
            role="render_geometry",
            media_type=_FORMAT_MEDIA_TYPE["gltf"],
            content=normalized,
        )
    ]
    for name, dependency in dependencies:
        artifacts.append(
            WireArtifact.from_bytes(
                filename=mapped_dependencies[name],
                role="supporting_asset",
                media_type=_GLTF_DEPENDENCY_MEDIA_TYPES[Path(name).suffix.lower()],
                content=dependency,
            )
        )
    return (
        tuple(artifacts),
        tuple(name for name, _value in members),
        tuple(name for name, _value in dependencies),
    )


def _obj_source_index(
    raw: str,
    *,
    local_count: int,
    global_offset: int,
    label: str,
) -> str:
    try:
        value = int(raw)
    except ValueError as exc:
        raise InvalidProviderResponseError(
            f"Onshape multipart OBJ contains an invalid {label} index"
        ) from exc
    if value == 0:
        raise InvalidProviderResponseError(
            f"Onshape multipart OBJ contains an invalid {label} index"
        )
    local_index = value if value > 0 else local_count + value + 1
    if not 1 <= local_index <= local_count:
        raise InvalidProviderResponseError(
            f"Onshape multipart OBJ contains an out-of-range {label} index"
        )
    return str(global_offset + local_index)


def _rebase_obj_reference(
    token: str,
    *,
    vertex_count: int,
    texture_count: int,
    normal_count: int,
    vertex_offset: int,
    texture_offset: int,
    normal_offset: int,
    directive: str,
) -> str:
    components = token.split("/")
    maximum_components = 1 if directive == "p" else (2 if directive == "l" else 3)
    if not components[0] or len(components) > maximum_components:
        raise InvalidProviderResponseError(
            "Onshape multipart OBJ contains an invalid indexed primitive"
        )
    result = [
        _obj_source_index(
            components[0],
            local_count=vertex_count,
            global_offset=vertex_offset,
            label="vertex",
        )
    ]
    if len(components) >= 2:
        result.append(
            ""
            if not components[1]
            else _obj_source_index(
                components[1],
                local_count=texture_count,
                global_offset=texture_offset,
                label="texture-coordinate",
            )
        )
    if len(components) == 3:
        result.append(
            ""
            if not components[2]
            else _obj_source_index(
                components[2],
                local_count=normal_count,
                global_offset=normal_offset,
                label="normal",
            )
        )
    return "/".join(result)


def _validate_obj_numeric_record(directive: str, value: str) -> None:
    expected = {"v": (3, 4), "vt": (1, 3), "vn": (3, 3), "vp": (1, 3)}[directive]
    fields = value.split()
    if not expected[0] <= len(fields) <= expected[1]:
        raise InvalidProviderResponseError(
            f"Onshape multipart OBJ contains an invalid {directive} record"
        )
    try:
        values = [float(field) for field in fields]
    except ValueError as exc:
        raise InvalidProviderResponseError(
            f"Onshape multipart OBJ contains an invalid {directive} record"
        ) from exc
    if not all(math.isfinite(item) for item in values):
        raise InvalidProviderResponseError(
            f"Onshape multipart OBJ contains a non-finite {directive} record"
        )


def _obj_safe_label(value: str, *, default: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return (label or default)[:128]


def _material_definitions(
    content: bytes,
    *,
    part_index: int,
) -> tuple[dict[str, str], list[str]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidProviderResponseError(
            "Onshape OBJ material library must be UTF-8 text"
        ) from exc
    if any(_MTL_EXTERNAL_ASSET_DIRECTIVE.match(line.lstrip()) for line in text.splitlines()):
        raise InvalidProviderResponseError(
            "Onshape OBJ material library references an unsupported texture dependency"
        )
    mapping: dict[str, str] = {}
    normalized: list[str] = []
    material_index = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        match = _MTL_MATERIAL_DIRECTIVE.match(stripped)
        if match is None:
            if mapping:
                normalized.append(line)
            elif stripped and not stripped.startswith("#"):
                raise InvalidProviderResponseError(
                    "Onshape OBJ material library has content before newmtl"
                )
            continue
        source_name = stripped[match.end() :].strip()
        if not source_name or source_name in mapping:
            raise InvalidProviderResponseError(
                "Onshape OBJ material library has an invalid material name"
            )
        material_index += 1
        replacement = f"part-{part_index:03d}-material-{material_index:03d}"
        mapping[source_name] = replacement
        normalized.append(f"newmtl {replacement}")
    if not mapping:
        raise InvalidProviderResponseError(
            "Onshape OBJ material library contains no material definitions"
        )
    return mapping, normalized


def _merge_obj_documents(
    primaries: list[tuple[str, bytes]],
    materials: list[tuple[str, bytes]],
    *,
    output_filename: str,
) -> tuple[bytes, bytes | None]:
    if sum(len(content) for _name, content in (*primaries, *materials)) > MAX_OUTPUT_ARTIFACT_BYTES:
        raise InvalidProviderResponseError(
            "Onshape multipart OBJ exceeds the merged-artifact byte limit"
        )
    material_by_name = dict(materials)
    if len(material_by_name) != len(materials):
        raise InvalidProviderResponseError(
            "Onshape multipart OBJ contains duplicate material-library names"
        )
    referenced_material_files: set[str] = set()
    merged_material_lines: list[str] = ["# Merged multipart Onshape material library"]
    merged_obj_lines: list[str] = ["# Merged multipart Onshape OBJ export"]
    if materials:
        merged_obj_lines.append(f"mtllib {Path(output_filename).stem}-material-001.mtl")

    vertex_offset = 0
    texture_offset = 0
    normal_offset = 0
    for part_index, (member_name, content) in enumerate(primaries, start=1):
        try:
            source_lines = content.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise InvalidProviderResponseError("Onshape OBJ export must be UTF-8 text") from exc
        material_names = []
        for line in source_lines:
            stripped = line.lstrip()
            match = _OBJ_MATERIAL_LIBRARY_DIRECTIVE.match(stripped)
            if match is not None:
                source_name = stripped[match.end() :].strip()
                if not source_name:
                    raise InvalidProviderResponseError(
                        "Onshape multipart OBJ has an empty material-library reference"
                    )
                material_names.append(source_name)
        local_materials: dict[str, str] = {}
        for material_name in material_names:
            material_content = material_by_name.get(material_name)
            if material_content is None:
                raise InvalidProviderResponseError(
                    "Onshape OBJ references an unavailable material library"
                )
            referenced_material_files.add(material_name)
            mapping, normalized = _material_definitions(
                material_content,
                part_index=part_index,
            )
            if set(mapping).intersection(local_materials):
                raise InvalidProviderResponseError(
                    "Onshape multipart OBJ has ambiguous material names"
                )
            local_materials.update(mapping)
            merged_material_lines.extend(normalized)

        vertex_count = 0
        texture_count = 0
        normal_count = 0
        part_label = _obj_safe_label(Path(member_name).stem, default=f"part-{part_index:03d}")
        merged_obj_lines.append(f"g part-{part_index:03d}-{part_label}")
        for line in source_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pieces = stripped.split(maxsplit=1)
            directive = pieces[0].lower()
            value = pieces[1] if len(pieces) == 2 else ""
            if directive == "mtllib":
                continue
            if directive in {"v", "vt", "vn", "vp"}:
                _validate_obj_numeric_record(directive, value)
                if directive == "v":
                    vertex_count += 1
                elif directive == "vt":
                    texture_count += 1
                elif directive == "vn":
                    normal_count += 1
                merged_obj_lines.append(f"{directive} {value}")
                continue
            if directive in {"f", "l", "p"}:
                references = value.split()
                minimum = 3 if directive == "f" else (2 if directive == "l" else 1)
                if len(references) < minimum:
                    raise InvalidProviderResponseError(
                        "Onshape multipart OBJ contains an invalid indexed primitive"
                    )
                rebased = [
                    _rebase_obj_reference(
                        item,
                        vertex_count=vertex_count,
                        texture_count=texture_count,
                        normal_count=normal_count,
                        vertex_offset=vertex_offset,
                        texture_offset=texture_offset,
                        normal_offset=normal_offset,
                        directive=directive,
                    )
                    for item in references
                ]
                merged_obj_lines.append(f"{directive} {' '.join(rebased)}")
                continue
            if directive == "usemtl":
                replacement = local_materials.get(value.strip())
                if replacement is None:
                    raise InvalidProviderResponseError(
                        "Onshape multipart OBJ references an undefined material"
                    )
                merged_obj_lines.append(f"usemtl {replacement}")
                continue
            if directive in {"g", "o"}:
                label = _obj_safe_label(value, default=directive)
                merged_obj_lines.append(f"{directive} part-{part_index:03d}-{label}")
                continue
            if directive == "s":
                merged_obj_lines.append(f"s {value}")
                continue
            raise InvalidProviderResponseError(
                f"Onshape multipart OBJ contains unsupported directive: {directive}"
            )
        vertex_offset += vertex_count
        texture_offset += texture_count
        normal_offset += normal_count

    if referenced_material_files != set(material_by_name):
        raise InvalidProviderResponseError(
            "Onshape multipart OBJ contains an unreferenced material library"
        )
    if vertex_offset == 0:
        raise InvalidProviderResponseError("Onshape multipart OBJ contains no vertices")
    obj_content = ("\n".join(merged_obj_lines) + "\n").encode("utf-8")
    material_content = (
        ("\n".join(merged_material_lines) + "\n").encode("utf-8") if materials else None
    )
    if len(obj_content) > MAX_OUTPUT_ARTIFACT_BYTES or (
        material_content is not None and len(material_content) > MAX_OUTPUT_ARTIFACT_BYTES
    ):
        raise InvalidProviderResponseError(
            "Onshape multipart OBJ exceeds the merged-artifact byte limit"
        )
    return obj_content, material_content


def _obj_artifacts(
    content: bytes,
    *,
    output_filename: str,
) -> tuple[tuple[WireArtifact, ...], tuple[str, ...], tuple[str, ...]]:
    if not content.startswith(b"PK\x03\x04"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidProviderResponseError("Onshape OBJ export must be UTF-8 text") from exc
        if any(_OBJ_MATERIAL_LIBRARY_DIRECTIVE.match(line.lstrip()) for line in text.splitlines()):
            raise InvalidProviderResponseError(
                "Onshape raw OBJ references an unavailable material library"
            )
        return (
            (
                WireArtifact.from_bytes(
                    filename=output_filename,
                    role="mesh_geometry",
                    media_type=_FORMAT_MEDIA_TYPE["obj"],
                    content=content,
                ),
            ),
            (),
            (),
        )

    members = _archive_members(
        content,
        label="OBJ",
        allowed_suffixes=frozenset({".obj", ".mtl"}),
    )
    primaries = [item for item in members if Path(item[0]).suffix.lower() == ".obj"]
    if not primaries:
        raise InvalidProviderResponseError(
            "Onshape OBJ archive must contain at least one OBJ entrypoint"
        )
    materials = [item for item in members if Path(item[0]).suffix.lower() == ".mtl"]
    if len(primaries) > 1:
        merged_obj, merged_material = _merge_obj_documents(
            primaries,
            materials,
            output_filename=output_filename,
        )
        artifacts = [
            WireArtifact.from_bytes(
                filename=output_filename,
                role="mesh_geometry",
                media_type=_FORMAT_MEDIA_TYPE["obj"],
                content=merged_obj,
            )
        ]
        supporting_names: tuple[str, ...] = ()
        if merged_material is not None:
            material_filename = f"{Path(output_filename).stem}-material-001.mtl"
            artifacts.append(
                WireArtifact.from_bytes(
                    filename=material_filename,
                    role="supporting_asset",
                    media_type="model/mtl",
                    content=merged_material,
                )
            )
            supporting_names = tuple(name for name, _value in materials)
        return (
            tuple(artifacts),
            tuple(name for name, _value in members),
            supporting_names,
        )
    stem = Path(output_filename).stem
    mapped_materials = {
        name: f"{stem}-material-{index:03d}.mtl"
        for index, (name, _value) in enumerate(materials, start=1)
    }
    try:
        obj_text = primaries[0][1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidProviderResponseError("Onshape OBJ export must be UTF-8 text") from exc
    referenced: set[str] = set()
    normalized_lines: list[str] = []
    for line in obj_text.splitlines():
        stripped = line.lstrip()
        match = _OBJ_MATERIAL_LIBRARY_DIRECTIVE.match(stripped)
        if match is None:
            normalized_lines.append(line)
            continue
        indentation = line[: len(line) - len(stripped)]
        source_name = stripped[match.end() :].strip()
        replacement = mapped_materials.get(source_name)
        if replacement is None:
            raise InvalidProviderResponseError(
                "Onshape OBJ references an unavailable material library"
            )
        referenced.add(source_name)
        normalized_lines.append(f"{indentation}mtllib {replacement}")
    if referenced != set(mapped_materials):
        raise InvalidProviderResponseError(
            "Onshape OBJ archive contains an unreferenced material library"
        )
    artifacts = [
        WireArtifact.from_bytes(
            filename=output_filename,
            role="mesh_geometry",
            media_type=_FORMAT_MEDIA_TYPE["obj"],
            content=("\n".join(normalized_lines) + "\n").encode("utf-8"),
        )
    ]
    for name, material in materials:
        try:
            material_text = material.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidProviderResponseError(
                "Onshape OBJ material library must be UTF-8 text"
            ) from exc
        if any(
            _MTL_EXTERNAL_ASSET_DIRECTIVE.match(line.lstrip())
            for line in material_text.splitlines()
        ):
            raise InvalidProviderResponseError(
                "Onshape OBJ material library references an unsupported texture dependency"
            )
        artifacts.append(
            WireArtifact.from_bytes(
                filename=mapped_materials[name],
                role="supporting_asset",
                media_type="model/mtl",
                content=material,
            )
        )
    return (
        tuple(artifacts),
        tuple(name for name, _value in members),
        tuple(name for name, _value in materials),
    )


class OnshapeConnector:
    """Snapshot and export Onshape geometry with a caller-owned API key."""

    def __init__(
        self,
        *,
        api_access_key: str,
        api_secret_key: str,
        api_base_url: str = "https://cad.onshape.com/api/v12",
        endpoint_alias: str = "onshape-official-api",
        provider_id: str = ONSHAPE_PROVIDER_ID,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 120.0,
        max_poll_attempts: int = 30,
        poll_interval_seconds: float = 1.0,
        transport: HttpTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        request_date_factory: Callable[[], str] = _request_date,
        request_nonce_factory: Callable[[], str] = _request_nonce,
    ) -> None:
        signer = _OnshapeApiKeySigner(
            access_key=api_access_key,
            secret_key=api_secret_key,
            date_factory=request_date_factory,
            nonce_factory=request_nonce_factory,
        )
        parsed = urlparse(api_base_url.strip().rstrip("/"))
        if re.fullmatch(r"/api/v[1-9][0-9]*", parsed.path) is None:
            raise ConnectorConfigurationError(
                "Onshape base URL must end in an explicit official /api/vN path"
            )
        if not 1 <= max_poll_attempts <= 120:
            raise ConnectorConfigurationError("Onshape polling attempts must be in [1, 120]")
        if not 0 < poll_interval_seconds <= 30:
            raise ConnectorConfigurationError("Onshape polling interval must be in (0, 30] seconds")
        self._provider_id = provider_id
        self._max_poll_attempts = max_poll_attempts
        self._poll_interval_seconds = poll_interval_seconds
        self._sleeper = sleeper
        self._http = BoundedHttpClient(
            provider_id=provider_id,
            endpoint_alias=endpoint_alias,
            base_url=api_base_url,
            bearer_token=None,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            transport=transport,
            allowed_host_suffix="onshape.com",
            request_header_provider=signer,
        )

    def create_immutable_version(
        self,
        request: OnshapeWorkspaceSnapshotRequest,
    ) -> str:
        """Create a named version so later export cannot race workspace edits."""

        created = self._http.request_json(
            "POST",
            relative_path=f"/documents/d/{request.document_id}/versions",
            payload={
                "documentId": request.document_id,
                "workspaceId": request.workspace_id,
                "name": request.version_name,
            },
            expected_statuses=frozenset({200, 201}),
        )
        version_id = _response_id(
            created.get("id"),
            label="created version id",
            pattern=_ONSHAPE_ID_RE,
        )
        for field, expected in (
            ("documentId", request.document_id),
            ("workspaceId", request.workspace_id),
        ):
            observed = created.get(field)
            if observed is not None and observed != expected:
                raise InvalidProviderResponseError(
                    f"Onshape created version changed {field}",
                    provider_id=self._provider_id,
                )
        return version_id

    def capabilities(self) -> AuthoringCapabilities:
        return AuthoringCapabilities(
            provider_id=self._provider_id,
            text=False,
            image=False,
            revision=False,
            export=True,
            native_source=False,
            formats=("step", "gltf", "obj"),
        )

    def generate(self, *_args: Any, **_kwargs: Any) -> None:
        raise UnsupportedCapabilityError(
            "Onshape reference connector exports existing immutable versions; it does not generate",
            provider_id=self._provider_id,
        )

    def revise(self, *_args: Any, **_kwargs: Any) -> None:
        raise UnsupportedCapabilityError(
            "Onshape reference connector does not mutate user documents",
            provider_id=self._provider_id,
        )

    def export(
        self,
        request: OnshapeVersionExportRequest,
        *,
        output_dir: str | Path,
    ) -> MaterializedWireSourceBundle:
        collection = "partstudios" if request.element_kind == "partstudio" else "assemblies"
        export_path = (
            f"/{collection}/d/{request.document_id}/v/{request.version_id}"
            f"/e/{request.element_id}/export/{request.format}"
        )
        options = dict(request.export_options)
        options.update(
            {
                "grouping": True,
                "isYAxisUp": request.format == "gltf",
                "storeInDocument": False,
            }
        )
        if request.format == "step":
            options["stepUnit"] = "METER"
        else:
            options["meshParams"] = {
                **cast(dict[str, Any], options.get("meshParams", {})),
                "unit": "MILLIMETER",
            }
        translation = self._http.request_json(
            "POST",
            relative_path=export_path,
            payload=options,
            expected_statuses=frozenset({200, 201}),
        )
        translation_id = _response_id(translation.get("id"), label="translation id")
        terminal = self._poll_translation(
            translation_id,
            initial=translation,
        )
        self._validate_immutable_binding(request, terminal, translation_id=translation_id)
        external_ids = terminal.get("resultExternalDataIds")
        if not isinstance(external_ids, list) or len(external_ids) != 1:
            raise InvalidProviderResponseError(
                "Onshape export must produce exactly one external data artifact",
                provider_id=self._provider_id,
            )
        external_data_id = _response_id(
            external_ids[0],
            label="external data id",
        )
        content = self._http.request_bytes(
            "GET",
            relative_path=(f"/documents/d/{request.document_id}/externaldata/{external_data_id}"),
        )
        archive_members: tuple[str, ...] = ()
        supporting_archive_members: tuple[str, ...] = ()
        if request.format == "step":
            artifacts, archive_members = _step_artifacts(
                content,
                output_filename=request.output_filename,
            )
        elif request.format == "gltf":
            artifacts, archive_members, supporting_archive_members = _gltf_artifacts(
                content,
                output_filename=request.output_filename,
            )
        else:
            artifacts, archive_members, supporting_archive_members = _obj_artifacts(
                content,
                output_filename=request.output_filename,
            )
        export_metadata = {
            "source_system": "onshape",
            "document_id": request.document_id,
            "version_id": request.version_id,
            "element_id": request.element_id,
            "element_kind": request.element_kind,
            "translation_id": translation_id,
            "external_data_id": external_data_id,
            "immutable_version": True,
            "grouped_step_export": request.format == "step",
            "archive_members": list(archive_members),
            "supporting_archive_members": list(supporting_archive_members),
        }
        bundle = WireSourceBundle(
            provider_id=self._provider_id,
            provider_version=urlparse(self._http.base_url).path.rsplit("/", 1)[-1],
            source_revision=onshape_source_revision(request),
            units=_FORMAT_UNITS[request.format],
            up_axis=_FORMAT_UP_AXIS[request.format],
            forward_axis=_FORMAT_FORWARD_AXIS[request.format],
            artifacts=artifacts,
            verification_assertions=(
                WireVerificationAssertion(
                    assertion_id=f"onshape-export-{request.format}",
                    status="passed",
                    summary=(
                        f"Onshape exported the immutable {request.format.upper()} representation."
                    ),
                    metrics=export_metadata,
                ),
            ),
            metadata=export_metadata,
        )
        return materialize_wire_bundle(bundle, output_dir)

    def _poll_translation(
        self,
        translation_id: str,
        *,
        initial: dict[str, Any],
    ) -> dict[str, Any]:
        current = initial
        for attempt in range(self._max_poll_attempts):
            state = current.get("requestState")
            if state == "DONE":
                return current
            if state == "FAILED":
                raise ProviderUnavailableError(
                    "Onshape export translation failed without fallback",
                    provider_id=self._provider_id,
                )
            if state != "ACTIVE":
                raise InvalidProviderResponseError(
                    "Onshape returned an invalid translation state",
                    provider_id=self._provider_id,
                )
            if attempt + 1 == self._max_poll_attempts:
                break
            delay_seconds = min(
                self._poll_interval_seconds * (2 ** min(attempt, 10)),
                10.0,
            )
            self._sleeper(delay_seconds)
            current = self._http.request_json(
                "GET",
                relative_path=f"/translations/{translation_id}",
            )
        raise ProviderUnavailableError(
            "Onshape export did not complete within the bounded poll window",
            provider_id=self._provider_id,
        )

    def _validate_immutable_binding(
        self,
        request: OnshapeVersionExportRequest,
        terminal: dict[str, Any],
        *,
        translation_id: str,
    ) -> None:
        observed_translation_id = _response_id(
            terminal.get("id"),
            label="terminal translation id",
        )
        if observed_translation_id != translation_id:
            raise InvalidProviderResponseError(
                "Onshape terminal response identifies another translation",
                provider_id=self._provider_id,
            )
        for field, expected in (
            ("documentId", request.document_id),
            ("resultDocumentId", request.document_id),
            ("versionId", request.version_id),
            ("requestElementId", request.element_id),
        ):
            observed = terminal.get(field)
            if observed is not None and observed != expected:
                raise InvalidProviderResponseError(
                    f"Onshape terminal response changed immutable {field}",
                    provider_id=self._provider_id,
                )
