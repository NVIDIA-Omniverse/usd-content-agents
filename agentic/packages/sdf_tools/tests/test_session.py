# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import sdf_tools
import sdf_tools.session as session_module


class ArrayRows:
    def __init__(self, rows: list[tuple[object, ...]], width: int) -> None:
        self._rows = rows
        self.shape = (len(rows), width)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


class ArrayValues:
    def __init__(self, shape: tuple[int, ...], values: tuple[object, ...]) -> None:
        self.shape = shape
        self._values = values

    def __iter__(self):
        return iter(self._values)


def _testing_toolkit() -> sdf_tools.SdfToolkit:
    return sdf_tools.SdfToolkit(
        registry=sdf_tools.SdfBackendRegistry(),
        discover_installed=False,
    )


def _register_test_extension(
    toolkit: sdf_tools.SdfToolkit,
    extension: sdf_tools.SdfBackendExtension,
) -> None:
    """Test-local synthetic injection; production exposes no registration hook."""

    sdf_tools.validate_license_manifest(extension.descriptor.license_manifest)
    backend_id = extension.descriptor.backend_id
    assert backend_id not in toolkit.registry._extensions  # type: ignore[attr-defined]
    toolkit.registry._extensions[backend_id] = extension  # type: ignore[attr-defined]


def test_fake_alternate_backend_implements_public_sdf_flow(
    triangle_mesh, sdf_backend_factory
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(
        backend="alternate",
        require={sdf_tools.Operation.MESH_TO_SDF, sdf_tools.Operation.FIELD_TO_MESH},
    )

    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    result = session.field_to_mesh(field)

    assert field.backend_id == "alternate"
    assert isinstance(result, sdf_tools.Mesh)
    assert [call[0] for call in backend.calls] == [
        sdf_tools.Operation.MESH_TO_SDF,
        sdf_tools.Operation.FIELD_TO_MESH,
    ]
    assert backend.calls[0][2]["limits"] == sdf_tools.DEFAULT_LIMITS


def test_write_fields_rejects_a_backend_result_for_another_path(
    triangle_mesh,
    sdf_backend_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    original_execute = backend.execute

    def execute(operation, /, *args, **kwargs):
        if operation is sdf_tools.Operation.WRITE_FIELDS:
            return Path("other.sdf-test")
        return original_execute(operation, *args, **kwargs)

    monkeypatch.setattr(backend, "execute", execute)
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)

    with pytest.raises(
        sdf_tools.BackendOperationError,
        match="instead of the requested path",
    ):
        session.write_fields("requested.sdf-test", field, format="sdf-test")


@pytest.mark.parametrize("backend_id", ["dense-grid", "sparse-tree"])
def test_backend_neutral_conformance_flow(backend_id, triangle_mesh, sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory(backend_id)
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(
        backend=backend_id,
        require=frozenset(sdf_tools.Operation),
        require_formats={"sdf-test"},
    )

    signed = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    unsigned = session.mesh_to_udf(triangle_mesh, voxel_size=0.1)
    assert isinstance(session.field_to_mesh(signed), sdf_tools.Mesh)
    assert session.union(signed, signed).kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert session.intersection(signed, signed).kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert session.difference(signed, signed).kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert session.offset(signed, 0.1).kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert session.smooth(signed).kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert session.smooth(unsigned).kind is sdf_tools.FieldKind.UNSIGNED_DISTANCE
    assert session.normalize(signed).kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert session.rebuild(signed).kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert session.resample_to_match(unsigned, signed).kind is sdf_tools.FieldKind.UNSIGNED_DISTANCE
    assert session.sample_values(signed, [(0.0, 0.0, 0.0)]).shape == (1,)
    assert session.sample_gradients(signed, [(0.0, 0.0, 0.0)]).shape == (1, 3)
    assert session.active_value_mask(signed).kind is sdf_tools.FieldKind.MASK
    assert session.topology_to_sdf(signed).kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert session.extract_enclosed_region(signed).kind is sdf_tools.FieldKind.MASK

    output = session.write_fields(
        "field.sdf-test",
        (signed, unsigned),
        format="sdf-test",
        metadata={"backend_neutral": True},
    )
    contents = session.read_fields(output, format="sdf-test")
    scalar = session.read_field(output, "density", format="sdf-test")

    assert output == Path("field.sdf-test")
    assert tuple(field.kind for field in contents.fields) == (
        sdf_tools.FieldKind.SIGNED_DISTANCE,
        sdf_tools.FieldKind.UNSIGNED_DISTANCE,
    )
    assert contents.metadata == {"backend_neutral": True}
    assert scalar.kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert {call[0] for call in backend.calls} == set(sdf_tools.Operation)


def test_auto_selection_honors_operation_and_format_requirements(
    sdf_backend_factory,
) -> None:
    higher, higher_backend = sdf_backend_factory(
        "higher",
        priority=100,
        read_formats=frozenset({"sdf-test"}),
        write_formats=frozenset({"other"}),
    )
    compatible, compatible_backend = sdf_backend_factory(
        "compatible",
        priority=1,
        read_formats=frozenset({"other"}),
        write_formats=frozenset({"sdf-test"}),
    )
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, higher)
    _register_test_extension(toolkit, compatible)

    session = toolkit.create_session(
        backend="auto",
        require={sdf_tools.Operation.WRITE_FIELDS},
        require_formats={"sdf-test"},
    )

    assert session.backend_id == "compatible"
    assert session.selection_rejections == (
        sdf_tools.BackendRejection("higher", "missing write formats: sdf-test"),
    )
    assert higher_backend.calls == []
    assert compatible_backend.calls == []


def test_explicit_selection_reports_missing_format_requirement(sdf_backend_factory) -> None:
    extension_value, _backend = sdf_backend_factory(
        "alternate",
        write_formats=frozenset({"sdf-test"}),
    )
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)

    with pytest.raises(
        sdf_tools.CapabilityUnavailableError,
        match="required write formats: vdb",
    ) as raised:
        toolkit.create_session(
            backend="alternate",
            require={sdf_tools.Operation.WRITE_FIELDS},
            require_formats={"vdb"},
        )

    assert raised.value.operations == ()
    assert raised.value.formats == ("vdb",)
    assert raised.value.read_formats == ()
    assert raised.value.write_formats == ("vdb",)


@pytest.mark.parametrize(
    ("require_formats", "error", "message"),
    [
        ("sdf-test", TypeError, "iterable of format identifiers"),
        (("",), ValueError, "must contain nonempty identifiers"),
        ((1,), TypeError, "must contain strings"),
    ],
)
def test_selection_validates_format_requirements(
    sdf_backend_factory, require_formats, error, message
) -> None:
    extension_value, _backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)

    with pytest.raises(error, match=message):
        toolkit.create_session(
            backend="alternate",
            require={sdf_tools.Operation.WRITE_FIELDS},
            require_formats=require_formats,
        )


def test_directional_formats_select_and_preflight_asymmetric_backend(
    triangle_mesh, sdf_backend_factory
) -> None:
    reader_extension, reader = sdf_backend_factory(
        "reader",
        priority=100,
        read_formats=frozenset({"portable"}),
        write_formats=frozenset({"native"}),
    )
    writer_extension, writer = sdf_backend_factory(
        "writer",
        priority=1,
        read_formats=frozenset({"native"}),
        write_formats=frozenset({"portable"}),
    )
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, reader_extension)
    _register_test_extension(toolkit, writer_extension)

    read_session = toolkit.create_session(
        backend="auto",
        require={sdf_tools.Operation.READ_FIELDS},
        require_formats={"portable"},
    )
    write_session = toolkit.create_session(
        backend="auto",
        require={sdf_tools.Operation.WRITE_FIELDS},
        require_formats={"portable"},
    )

    assert read_session.backend_id == "reader"
    assert write_session.backend_id == "writer"
    assert read_session.backend_info.supported_formats == frozenset({"native", "portable"})

    read_session.read_fields("input.portable", format="portable")
    field = read_session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    with pytest.raises(
        sdf_tools.CapabilityUnavailableError,
        match="required write formats: portable",
    ):
        read_session.write_fields("output.portable", field, format="portable")
    with pytest.raises(
        sdf_tools.CapabilityUnavailableError,
        match="required read formats: portable",
    ):
        write_session.read_fields("input.portable", format="portable")

    field = write_session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    write_session.write_fields("output.portable", field, format="portable")

    assert [call[0] for call in reader.calls] == [
        sdf_tools.Operation.READ_FIELDS,
        sdf_tools.Operation.MESH_TO_SDF,
    ]
    assert [call[0] for call in writer.calls] == [
        sdf_tools.Operation.MESH_TO_SDF,
        sdf_tools.Operation.WRITE_FIELDS,
    ]


def test_mixed_backend_boolean_fails_before_driver_call(triangle_mesh, sdf_backend_factory) -> None:
    first_extension, first = sdf_backend_factory("first")
    second_extension, _ = sdf_backend_factory("second")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, first_extension)
    _register_test_extension(toolkit, second_extension)
    session = toolkit.create_session(backend="first")
    left = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    right = sdf_tools.Field("second", sdf_tools.FieldKind.SIGNED_DISTANCE, object(), object())

    with pytest.raises(sdf_tools.BackendMismatchError, match="second"):
        session.union(left, right)

    assert [call[0] for call in first.calls] == [sdf_tools.Operation.MESH_TO_SDF]


def test_boolean_requires_signed_fields(sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    left = sdf_tools.Field(
        "alternate", sdf_tools.FieldKind.UNSIGNED_DISTANCE, object(), backend.field_owner
    )
    right = sdf_tools.Field(
        "alternate", sdf_tools.FieldKind.UNSIGNED_DISTANCE, object(), backend.field_owner
    )

    with pytest.raises(sdf_tools.InvalidGeometryError, match="signed-distance"):
        session.union(left, right)

    assert backend.calls == []


def test_session_refuses_an_undeclared_operation_before_driver_call(
    triangle_mesh, sdf_backend_factory
) -> None:
    extension_value, backend = sdf_backend_factory(
        "narrow",
        operations=frozenset({sdf_tools.Operation.MESH_TO_SDF}),
    )
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="narrow")
    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)

    with pytest.raises(sdf_tools.CapabilityUnavailableError, match="sample_values"):
        session.sample_values(field, [(0.0, 0.0, 0.0)])

    assert [call[0] for call in backend.calls] == [sdf_tools.Operation.MESH_TO_SDF]


def test_smooth_rejects_mask_before_driver_call(sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    mask = sdf_tools.Field("alternate", sdf_tools.FieldKind.MASK, object(), backend.field_owner)

    with pytest.raises(sdf_tools.InvalidGeometryError, match="distance or scalar"):
        session.smooth(mask)

    assert backend.calls == []


def test_session_validates_sample_result_shape(triangle_mesh, sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    backend.execute = lambda *_args, **_kwargs: object()  # type: ignore[method-assign]

    with pytest.raises(sdf_tools.BackendOperationError, match="malformed samples"):
        session.sample_values(field, [(0.0, 0.0, 0.0)])


def test_session_rejects_nonintegral_sample_shape(triangle_mesh, sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    backend.execute = lambda *_args, **_kwargs: type(  # type: ignore[method-assign]
        "Samples", (), {"shape": (1.5,)}
    )()

    with pytest.raises(sdf_tools.BackendOperationError, match="malformed samples"):
        session.sample_values(field, [(0.0, 0.0, 0.0)])


@pytest.mark.parametrize(
    "result",
    [
        type("ShapeOnlySamples", (), {"shape": (1,)})(),
        ArrayValues((1,), (float("nan"),)),
        ArrayValues((1,), ("not-numeric",)),
        ArrayValues((1, 3), ((0.0, float("inf"), 0.0),)),
    ],
)
def test_session_rejects_unusable_or_nonfinite_sample_data(
    triangle_mesh, sdf_backend_factory, result
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    backend.execute = lambda *_args, **_kwargs: result  # type: ignore[method-assign]

    operation = session.sample_gradients if result.shape == (1, 3) else session.sample_values
    with pytest.raises(sdf_tools.BackendOperationError, match="malformed samples"):
        operation(field, [(0.0, 0.0, 0.0)])


def test_session_normalizes_unexpected_driver_exceptions(
    triangle_mesh, sdf_backend_factory
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    failure = RuntimeError("backend implementation detail")

    def fail(*_args, **_kwargs):
        raise failure

    backend.execute = fail  # type: ignore[method-assign]

    with pytest.raises(
        sdf_tools.BackendOperationError,
        match="failed while executing mesh_to_sdf",
    ) as raised:
        session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)

    assert raised.value.__cause__ is failure


def test_session_preserves_backend_operation_errors(triangle_mesh, sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    failure = sdf_tools.ResourceLimitError("driver limit")

    def fail(*_args, **_kwargs):
        raise failure

    backend.execute = fail  # type: ignore[method-assign]

    with pytest.raises(sdf_tools.ResourceLimitError, match="driver limit") as raised:
        session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)

    assert raised.value is failure


@pytest.mark.parametrize(
    "error_type",
    [
        sdf_tools.SdfToolsError,
        sdf_tools.BackendNotRegisteredError,
        sdf_tools.LicensePolicyError,
    ],
)
def test_session_normalizes_non_operation_sdf_tools_errors(
    triangle_mesh, sdf_backend_factory, error_type
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    failure = error_type("driver must not report this error category")

    def fail(*_args, **_kwargs):
        raise failure

    backend.execute = fail  # type: ignore[method-assign]

    with pytest.raises(
        sdf_tools.BackendOperationError,
        match="failed while executing mesh_to_sdf",
    ) as raised:
        session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)

    assert raised.value.__cause__ is failure


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ("not-a-field", "non-field result"),
        (
            sdf_tools.Field("different", sdf_tools.FieldKind.SIGNED_DISTANCE, object(), object()),
            "owned by 'different'",
        ),
    ],
)
def test_session_rejects_backend_results_that_violate_the_contract(
    triangle_mesh,
    sdf_backend_factory,
    result: object,
    message: str,
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    backend.execute = lambda *_args, **_kwargs: result  # type: ignore[method-assign]

    with pytest.raises(sdf_tools.BackendOperationError, match=message):
        session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)


def test_session_rejects_wrong_kind_from_the_selected_backend(
    triangle_mesh, sdf_backend_factory
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    result = sdf_tools.Field(
        "alternate",
        sdf_tools.FieldKind.UNSIGNED_DISTANCE,
        object(),
        backend.field_owner,
    )
    backend.execute = lambda *_args, **_kwargs: result  # type: ignore[method-assign]

    with pytest.raises(sdf_tools.BackendOperationError, match="expected 'signed_distance'"):
        session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)


@pytest.mark.parametrize(
    "malformed",
    [
        sdf_tools.Mesh(vertices=[(0.0, 1.0)], triangles=[(0, 0, 0)]),
        sdf_tools.Mesh(vertices=[("x", "y", "z")], triangles=[(0, 0, 0)]),
        sdf_tools.Mesh(vertices=[(0.0, float("nan"), 0.0)], triangles=[(0, 0, 0)]),
        sdf_tools.Mesh(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 1.5)],
        ),
        sdf_tools.Mesh(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 3)],
        ),
        sdf_tools.Mesh(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles=[(0, 1, 1)],
        ),
    ],
)
def test_session_rejects_malformed_mesh_result(
    triangle_mesh, sdf_backend_factory, malformed
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    backend.execute = lambda *_args, **_kwargs: malformed  # type: ignore[method-assign]

    with pytest.raises(sdf_tools.BackendOperationError, match="malformed mesh"):
        session.field_to_mesh(field)


def test_session_accepts_valid_array_like_mesh_result(triangle_mesh, sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    result = sdf_tools.Mesh(
        vertices=ArrayRows(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            3,
        ),
        triangles=ArrayRows([(0, 1, 2)], 3),
    )
    backend.execute = lambda *_args, **_kwargs: result  # type: ignore[method-assign]

    assert session.field_to_mesh(field) is result


def test_session_bounds_successful_mesh_result(triangle_mesh, sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(
        backend="alternate",
        limits=sdf_tools.ExecutionLimits(max_vertices=2),
    )
    field = session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)

    with pytest.raises(sdf_tools.ResourceLimitError, match="backend mesh has 3 vertices"):
        session.field_to_mesh(field)

    assert [call[0] for call in backend.calls] == [
        sdf_tools.Operation.MESH_TO_SDF,
        sdf_tools.Operation.FIELD_TO_MESH,
    ]


def test_session_rejects_same_id_field_from_another_implementation(
    triangle_mesh, sdf_backend_factory
) -> None:
    first_extension, first = sdf_backend_factory("alternate")
    second_extension, second = sdf_backend_factory("alternate")
    first_toolkit = _testing_toolkit()
    second_toolkit = _testing_toolkit()
    _register_test_extension(first_toolkit, first_extension)
    _register_test_extension(second_toolkit, second_extension)
    first_session = first_toolkit.create_session(backend="alternate")
    second_session = second_toolkit.create_session(backend="alternate")
    foreign = second_session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)

    with pytest.raises(sdf_tools.BackendMismatchError, match="selected 'alternate'"):
        first_session.field_to_mesh(foreign)

    assert first.calls == []
    assert len(second.calls) == 1


def test_sessions_from_one_toolkit_get_distinct_driver_ownership(
    triangle_mesh, sdf_backend_factory
) -> None:
    template_extension, template = sdf_backend_factory("alternate")
    instances = []

    def create_backend():
        backend = type(template)(template_extension.descriptor)
        instances.append(backend)
        return backend

    toolkit = _testing_toolkit()
    _register_test_extension(
        toolkit, sdf_tools.SdfBackendExtension(template_extension.descriptor, create_backend)
    )
    first_session = toolkit.create_session(backend="alternate")
    first_field = first_session.mesh_to_sdf(triangle_mesh, voxel_size=0.1)
    second_session = toolkit.create_session(backend="alternate")

    assert len(instances) == 2
    assert instances[0].field_owner is not instances[1].field_owner
    with pytest.raises(sdf_tools.BackendMismatchError, match="selected 'alternate'"):
        second_session.field_to_mesh(first_field)


def test_session_rejects_foreign_fields_in_read_contents(sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    backend.execute = lambda *_args, **_kwargs: sdf_tools.FieldContents(  # type: ignore[method-assign]
        fields=(sdf_tools.Field("different", sdf_tools.FieldKind.SCALAR, object(), object()),),
        metadata={},
    )

    with pytest.raises(sdf_tools.BackendOperationError, match="owned by 'different'"):
        session.read_fields("input.sdf-test", format="sdf-test")


def test_session_bounds_successful_field_collection(
    sdf_backend_factory,
) -> None:
    limits = sdf_tools.ExecutionLimits(max_fields=1)
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate", limits=limits)
    fields = tuple(
        sdf_tools.Field(
            "alternate",
            sdf_tools.FieldKind.SCALAR,
            object(),
            backend.field_owner,
        )
        for _ in range(2)
    )
    backend.execute = lambda *_args, **_kwargs: sdf_tools.FieldContents(  # type: ignore[method-assign]
        fields=fields,
        metadata={},
    )

    with pytest.raises(sdf_tools.ResourceLimitError, match="returned 2 fields"):
        session.read_fields("input.sdf-test", format="sdf-test")


def test_session_rejects_malformed_success_metadata(sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")
    field = sdf_tools.Field(
        "alternate",
        sdf_tools.FieldKind.SCALAR,
        object(),
        backend.field_owner,
    )
    backend.execute = lambda *_args, **_kwargs: sdf_tools.FieldContents(  # type: ignore[method-assign]
        fields=(field,),
        metadata={"invalid": object()},
    )

    with pytest.raises(
        sdf_tools.BackendOperationError,
        match="failed while executing read_fields",
    ):
        session.read_fields("input.sdf-test", format="sdf-test")


def test_session_rejects_unsupported_format_before_driver_call(
    sdf_backend_factory,
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate")

    with pytest.raises(
        sdf_tools.CapabilityUnavailableError,
        match="required read formats: vdb",
    ):
        session.read_fields("input.vdb", format="vdb")

    field = sdf_tools.Field(
        "alternate",
        sdf_tools.FieldKind.SIGNED_DISTANCE,
        object(),
        backend.field_owner,
    )
    with pytest.raises(
        sdf_tools.CapabilityUnavailableError,
        match="required write formats: vdb",
    ):
        session.write_fields("output.vdb", field, format="vdb")

    assert backend.calls == []


@pytest.mark.parametrize(
    ("metadata", "limits", "message"),
    [
        (
            {"a": 1, "b": 2},
            sdf_tools.ExecutionLimits(max_metadata_entries=1),
            "entry limit",
        ),
        (
            {"value": "12345678"},
            sdf_tools.ExecutionLimits(max_metadata_bytes=8),
            "byte limit",
        ),
    ],
)
def test_session_bounds_write_metadata_before_driver_call(
    sdf_backend_factory, metadata, limits, message
) -> None:
    extension_value, backend = sdf_backend_factory("alternate")
    toolkit = _testing_toolkit()
    _register_test_extension(toolkit, extension_value)
    session = toolkit.create_session(backend="alternate", limits=limits)
    field = sdf_tools.Field(
        "alternate", sdf_tools.FieldKind.SIGNED_DISTANCE, object(), backend.field_owner
    )

    with pytest.raises(sdf_tools.ResourceLimitError, match=message):
        session.write_fields(
            "asset.sdf-test",
            field,
            format="sdf-test",
            metadata=metadata,
        )

    assert backend.calls == []


def test_explicit_discovery_remains_available_after_auto_discovery(monkeypatch) -> None:
    calls: list[str | None] = []

    def discover(_registry, *, backend=None):
        calls.append(backend)
        return ()

    monkeypatch.setattr(session_module, "discover_installed_backends", discover)
    toolkit = sdf_tools.SdfToolkit()

    toolkit.load_installed_backends()
    toolkit.load_installed_backends("experimental")

    assert calls == [None, "experimental"]
