# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral Geometry Repair qualification policy tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sdf_tools

from geometry_repair.sdf_backend_qualification import (
    DEFAULT_SDF_BACKEND_ID,
    GEOMETRY_REPAIR_REQUIRED_SDF_OPERATIONS,
    SDF_REBUILD_BUILD_ID,
    SDF_REBUILD_IMPLEMENTATION_VERSION,
    get_sdf_backend_qualification,
    qualified_sdf_backend_ids,
)


def _qualified_identity() -> dict[str, object]:
    qualification = get_sdf_backend_qualification("openvdb")
    provenance = {
        **qualification.expected_provenance,
        "library_version": list(qualification.expected_provenance["library_version"]),
        "module_path": "/runtime/openvdb.so",
        "module_sha256": "1" * 64,
        "source_lock_path": "/runtime/source-lock.json",
        "source_lock_sha256": "2" * 64,
    }
    return {
        **qualification.expected_identity,
        "operations": sorted(
            {operation.value for operation in qualification.required_operations} | {"write_fields"}
        ),
        "read_formats": [],
        "write_formats": ["vdb"],
        "supported_formats": ["vdb"],
        "provenance": provenance,
    }


def test_geometry_repair_qualifies_backends_separately_from_the_sdf_facade() -> None:
    qualification = get_sdf_backend_qualification(DEFAULT_SDF_BACKEND_ID)

    assert qualified_sdf_backend_ids() == ("openvdb",)
    assert qualification.backend_id == "openvdb"
    assert qualification.qualification_id == "geometry-repair.openvdb13.v1"
    assert qualification.required_operations == GEOMETRY_REPAIR_REQUIRED_SDF_OPERATIONS
    assert SDF_REBUILD_BUILD_ID == "geometry-repair:sdf-rebuild:2"
    assert SDF_REBUILD_IMPLEMENTATION_VERSION == "geometry-repair-sdf-rebuild-2"
    with pytest.raises(TypeError):
        qualification.expected_identity["backend_id"] = "substituted"  # type: ignore[index]
    with pytest.raises(TypeError):
        qualification.expected_provenance["library_version"][0] = 12  # type: ignore[index]


def test_qualification_inspects_the_selected_driver_through_sdf_tools(monkeypatch) -> None:
    identity = _qualified_identity()
    captured: dict[str, object] = {}

    def create_session(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(backend_info=SimpleNamespace(as_dict=lambda: identity))

    monkeypatch.setattr(sdf_tools, "create_session", create_session)

    loaded = get_sdf_backend_qualification("openvdb").inspect()

    assert loaded == identity
    assert captured["backend"] == "openvdb"
    assert captured["require"] == get_sdf_backend_qualification("openvdb").required_operations


def test_unknown_or_identity_mismatched_backend_fails_closed(monkeypatch) -> None:
    with pytest.raises(sdf_tools.BackendUnavailableError, match="not qualified"):
        get_sdf_backend_qualification("unreviewed")

    identity = _qualified_identity()
    identity["backend_id"] = "substituted"
    monkeypatch.setattr(
        sdf_tools,
        "create_session",
        lambda **_kwargs: SimpleNamespace(backend_info=SimpleNamespace(as_dict=lambda: identity)),
    )

    with pytest.raises(sdf_tools.BackendUnavailableError, match="differs"):
        get_sdf_backend_qualification("openvdb").inspect()


@pytest.mark.parametrize("replacement", [False, 13.0])
def test_qualification_identity_comparison_is_json_type_strict(monkeypatch, replacement) -> None:
    identity = _qualified_identity()
    identity["provenance"]["library_version"][0] = replacement  # type: ignore[index]
    monkeypatch.setattr(
        sdf_tools,
        "create_session",
        lambda **_kwargs: SimpleNamespace(backend_info=SimpleNamespace(as_dict=lambda: identity)),
    )

    with pytest.raises(sdf_tools.BackendUnavailableError, match="provenance differs"):
        get_sdf_backend_qualification("openvdb").inspect()
