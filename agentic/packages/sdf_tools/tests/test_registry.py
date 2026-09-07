# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import py_compile
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import sdf_tools
import sdf_tools.registry as registry_module

_DRIVER_MODULE_BYTES = b"# authenticated synthetic SDF driver\n"
_DRIVER_MODULE_SHA256 = hashlib.sha256(_DRIVER_MODULE_BYTES).hexdigest()


@pytest.fixture
def isolated_authenticated_source_finder(monkeypatch: pytest.MonkeyPatch):
    original_meta_path = list(sys.meta_path)
    finder = registry_module._AuthenticatedSourceFinder()  # type: ignore[attr-defined]
    monkeypatch.setattr(registry_module, "_AUTHENTICATED_SOURCE_FINDER", finder)
    namespaces: list[str] = []
    yield namespaces
    sys.meta_path[:] = original_meta_path
    for namespace in namespaces:
        prefix = f"{namespace}."
        for module_name in tuple(sys.modules):
            if module_name == namespace or module_name.startswith(prefix):
                sys.modules.pop(module_name, None)


def _real_entry_point(entry) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(entry.name, entry.value, entry.group)._for(entry.dist)


def _testing_registry(
    *extensions: sdf_tools.SdfBackendExtension,
) -> sdf_tools.SdfBackendRegistry:
    """Test-private injection without a registration hook in shipped code."""

    registry = sdf_tools.SdfBackendRegistry()
    for extension in extensions:
        sdf_tools.validate_license_manifest(extension.descriptor.license_manifest)
        backend_id = extension.descriptor.backend_id
        if backend_id in registry._extensions:  # type: ignore[attr-defined]
            raise ValueError(f"duplicate synthetic backend identifier: {backend_id}")
        registry._extensions[backend_id] = extension  # type: ignore[attr-defined]
    return registry


def test_auto_selection_is_priority_then_identifier(sdf_backend_factory) -> None:
    low, _ = sdf_backend_factory("low", priority=1)
    beta, _ = sdf_backend_factory("beta", priority=5)
    alpha, alpha_backend = sdf_backend_factory("alpha", priority=5)
    registry = _testing_registry(low, beta, alpha)

    selected, info, rejections = registry.resolve(
        backend="auto", require={sdf_tools.Operation.MESH_TO_SDF}
    )

    assert selected is alpha_backend
    assert info.backend_id == "alpha"
    assert rejections == ()


def test_explicit_backend_never_falls_back(sdf_backend_factory) -> None:
    unavailable, _ = sdf_backend_factory("selected", available=False)
    fallback, _ = sdf_backend_factory("fallback", priority=100)
    registry = _testing_registry(unavailable, fallback)

    with pytest.raises(sdf_tools.BackendUnavailableError, match="selected"):
        registry.resolve(backend="selected")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_id", "substituted"),
        ("implementation_version", "2.0"),
        ("execution_mode", "helper_process"),
    ],
)
def test_runtime_identity_must_match_registered_descriptor(
    field: str, value: object, sdf_backend_factory
) -> None:
    registered, backend = sdf_backend_factory("selected")
    registry = _testing_registry(registered)
    original = backend.inspect

    def changed_identity() -> sdf_tools.BackendInfo:
        identity = original()
        values = {
            "backend_id": identity.backend_id,
            "implementation_version": identity.implementation_version,
            "operations": identity.operations,
            "execution_mode": identity.execution_mode,
            "read_formats": identity.read_formats,
            "write_formats": identity.write_formats,
            "provenance": identity.provenance,
        }
        values[field] = value
        return sdf_tools.BackendInfo(**values)

    backend.inspect = changed_identity  # type: ignore[method-assign]

    with pytest.raises(sdf_tools.BackendUnavailableError, match="differs from policy"):
        registry.resolve(backend="selected")


def test_runtime_provenance_is_immutable(sdf_backend_factory) -> None:
    extension_value, _backend = sdf_backend_factory("selected")
    registry = _testing_registry(extension_value)
    _driver, info, _rejections = registry.resolve(backend="selected")

    with pytest.raises(TypeError):
        info.provenance["test"] = False  # type: ignore[index]

    detached = info.as_dict()
    detached["provenance"]["test"] = False
    assert info.provenance["test"] is True


def test_runtime_formats_must_satisfy_selection_requirements(sdf_backend_factory) -> None:
    extension_value, backend = sdf_backend_factory(
        "selected",
        read_formats=frozenset({"sdf-test"}),
    )
    registry = _testing_registry(extension_value)
    original = backend.inspect

    def missing_runtime_format() -> sdf_tools.BackendInfo:
        identity = original()
        return sdf_tools.BackendInfo(
            backend_id=identity.backend_id,
            implementation_version=identity.implementation_version,
            operations=identity.operations,
            execution_mode=identity.execution_mode,
            read_formats=frozenset(),
            write_formats=identity.write_formats,
            provenance=identity.provenance,
        )

    backend.inspect = missing_runtime_format  # type: ignore[method-assign]

    with pytest.raises(
        sdf_tools.CapabilityUnavailableError,
        match="required read formats: sdf-test",
    ):
        registry.resolve(
            backend="selected",
            require={sdf_tools.Operation.READ_FIELDS},
            require_formats={"sdf-test"},
        )


def test_explicit_factory_failure_is_normalized(sdf_license_manifest_factory) -> None:
    descriptor = sdf_tools.BackendDescriptor(
        backend_id="broken",
        implementation_version="1.0",
        operations=frozenset({sdf_tools.Operation.MESH_TO_SDF}),
        priority=1,
        execution_mode="in_process",
        read_formats=frozenset(),
        write_formats=frozenset(),
        license_manifest=sdf_license_manifest_factory("broken"),
    )
    registry = _testing_registry(
        sdf_tools.SdfBackendExtension(
            descriptor,
            lambda: (_ for _ in ()).throw(ImportError("native module missing")),
        ),
    )

    with pytest.raises(sdf_tools.BackendUnavailableError, match="could not be created"):
        registry.resolve(backend="broken")


def _admission_policy(extension: sdf_tools.SdfBackendExtension) -> dict[str, object]:
    descriptor = extension.descriptor
    return {
        "schema": "world-understanding.sdf-backend-policy.v1",
        "entry_point_group": "sdf_tools.backends",
        "backends": {
            descriptor.backend_id: {
                "authenticated_data_files": {},
                "authenticated_files": {
                    f"{descriptor.backend_id}.py": _DRIVER_MODULE_SHA256,
                },
                "distribution": f"{descriptor.backend_id}-driver",
                "distribution_version": "1.0",
                "entry_point": f"{descriptor.backend_id}:extension",
                "entry_point_module_path": f"{descriptor.backend_id}.py",
                "entry_point_module_sha256": _DRIVER_MODULE_SHA256,
                "implementation_version": descriptor.implementation_version,
                "license_manifest_sha256": descriptor.license_manifest.sha256,
                "native_closure_attestation": (
                    descriptor.license_manifest.native_closure_attestation
                ),
                "production_qualified": True,
            }
        },
    }


def test_production_registry_has_no_direct_registration_surface() -> None:
    registry = sdf_tools.SdfBackendRegistry()
    toolkit = sdf_tools.SdfToolkit(registry=registry, discover_installed=False)

    assert not hasattr(registry, "register")
    assert not hasattr(registry, "register_many")
    assert not hasattr(registry, "_for_testing")
    assert not hasattr(registry, "_register_for_testing")
    assert not hasattr(registry, "_register_many_for_testing")
    assert not hasattr(toolkit, "_register_for_testing")


def test_toolkit_rejects_a_caller_supplied_registry_subclass() -> None:
    class CallerRegistry(sdf_tools.SdfBackendRegistry):
        def resolve(self, **_kwargs):
            raise AssertionError("caller-controlled resolution must not execute")

    with pytest.raises(TypeError, match="exact sdf_tools.SdfBackendRegistry"):
        sdf_tools.SdfToolkit(registry=CallerRegistry(), discover_installed=False)


@pytest.mark.parametrize(
    ("distribution_version", "module_bytes", "record_module", "message"),
    [
        ("9.9", _DRIVER_MODULE_BYTES, True, "distribution version"),
        ("1.0", b"tampered driver code\n", True, "module is not policy-approved"),
        ("1.0", _DRIVER_MODULE_BYTES, False, "does not record"),
    ],
)
def test_driver_authentication_fails_before_entry_point_code_executes(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
    distribution_version: str,
    module_bytes: bytes,
    record_module: bool,
    message: str,
) -> None:
    extension_value, _backend = sdf_backend_factory("authenticated")
    policy = _admission_policy(extension_value)
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    load_called = False

    def load():
        nonlocal load_called
        load_called = True
        return extension_value

    entry = _entry_point(
        "authenticated",
        "authenticated:extension",
        "authenticated-driver",
        extension_value,
        distribution_version=distribution_version,
        module_bytes=module_bytes,
        record_module=record_module,
        load=load,
    )

    with pytest.raises(sdf_tools.BackendUnavailableError, match=message):
        registry_module.discover_installed_backends(
            sdf_tools.SdfBackendRegistry(),
            backend="authenticated",
            entry_points=[entry],
        )

    assert load_called is False


def test_driver_authentication_rejects_an_unlisted_package_module_before_load(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
) -> None:
    extension_value, _backend = sdf_backend_factory("authenticated")
    policy = _admission_policy(extension_value)
    admission = policy["backends"]["authenticated"]  # type: ignore[index]
    admission["entry_point"] = "driver_package.backend:extension"  # type: ignore[index]
    admission["entry_point_module_path"] = "driver_package/backend.py"  # type: ignore[index]
    admission["authenticated_files"] = {  # type: ignore[index]
        "driver_package/backend.py": _DRIVER_MODULE_SHA256,
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    load_called = False

    def load():
        nonlocal load_called
        load_called = True
        return extension_value

    entry = _entry_point(
        "authenticated",
        "driver_package.backend:extension",
        "authenticated-driver",
        extension_value,
        load=load,
    )
    entry.dist.add_file("driver_package/hidden.py", b"raise RuntimeError('hidden')\n")

    with pytest.raises(sdf_tools.BackendUnavailableError, match="files differ from policy"):
        registry_module.discover_installed_backends(
            sdf_tools.SdfBackendRegistry(),
            backend="authenticated",
            entry_points=[entry],
        )

    assert load_called is False


def test_authenticated_source_loader_ignores_unchecked_hash_bytecode(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
    tmp_path: Path,
    isolated_authenticated_source_finder: list[str],
) -> None:
    extension_value, _backend = sdf_backend_factory("authenticated")
    backend_bytes = b"from _sdf_test_extension_holder import extension\n"
    package_bytes = b"# authenticated package\n"
    policy = _admission_policy(extension_value)
    admission = policy["backends"]["authenticated"]  # type: ignore[index]
    admission["entry_point"] = "driver_package.backend:extension"  # type: ignore[index]
    admission["entry_point_module_path"] = "driver_package/backend.py"  # type: ignore[index]
    admission["entry_point_module_sha256"] = hashlib.sha256(backend_bytes).hexdigest()  # type: ignore[index]
    admission["authenticated_files"] = {  # type: ignore[index]
        "driver_package/__init__.py": hashlib.sha256(package_bytes).hexdigest(),
        "driver_package/backend.py": hashlib.sha256(backend_bytes).hexdigest(),
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    monkeypatch.setitem(
        sys.modules,
        "_sdf_test_extension_holder",
        SimpleNamespace(extension=extension_value),
    )
    isolated_authenticated_source_finder.append("driver_package")

    entry = _entry_point(
        "authenticated",
        "driver_package.backend:extension",
        "authenticated-driver",
        extension_value,
        module_bytes=backend_bytes,
    )
    entry.dist.add_file("driver_package/__init__.py", package_bytes)
    approved_source = entry.dist.locate_file("driver_package/backend.py")
    bytecode_path = Path(importlib.util.cache_from_source(str(approved_source)))
    malicious_source = tmp_path / "backend.py"
    malicious_source.write_text(
        "raise AssertionError('unchecked malicious bytecode executed')\n",
        encoding="utf-8",
    )
    bytecode_path.parent.mkdir(parents=True)
    py_compile.compile(
        str(malicious_source),
        cfile=str(bytecode_path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    assert int.from_bytes(bytecode_path.read_bytes()[4:8], "little") == 1
    monkeypatch.setattr(
        importlib.metadata.EntryPoint,
        "load",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("ordinary entry-point loading must not execute")
        ),
    )

    loaded = registry_module.discover_installed_backends(
        sdf_tools.SdfBackendRegistry(),
        backend="authenticated",
        entry_points=[_real_entry_point(entry)],
    )

    assert loaded == ("authenticated",)
    assert bytecode_path.is_file()
    assert isinstance(
        sys.modules["driver_package.backend"].__loader__,
        registry_module._AuthenticatedSourceLoader,  # type: ignore[attr-defined]
    )


def test_authenticated_source_load_is_repeatable_with_a_bytecode_cache(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
    isolated_authenticated_source_finder: list[str],
) -> None:
    extension_value, _backend = sdf_backend_factory("authenticated")
    backend_bytes = b"from _sdf_test_extension_holder import extension\n"
    package_bytes = b"# authenticated package\n"
    policy = _admission_policy(extension_value)
    admission = policy["backends"]["authenticated"]  # type: ignore[index]
    admission["entry_point"] = "driver_package.backend:extension"  # type: ignore[index]
    admission["entry_point_module_path"] = "driver_package/backend.py"  # type: ignore[index]
    admission["entry_point_module_sha256"] = hashlib.sha256(backend_bytes).hexdigest()  # type: ignore[index]
    admission["authenticated_files"] = {  # type: ignore[index]
        "driver_package/__init__.py": hashlib.sha256(package_bytes).hexdigest(),
        "driver_package/backend.py": hashlib.sha256(backend_bytes).hexdigest(),
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    monkeypatch.setitem(
        sys.modules,
        "_sdf_test_extension_holder",
        SimpleNamespace(extension=extension_value),
    )
    isolated_authenticated_source_finder.append("driver_package")

    entry = _entry_point(
        "authenticated",
        "driver_package.backend:extension",
        "authenticated-driver",
        extension_value,
        module_bytes=backend_bytes,
    )
    entry.dist.add_file("driver_package/__init__.py", package_bytes)
    cache_path = Path(
        importlib.util.cache_from_source(str(entry.dist.locate_file("driver_package/backend.py")))
    )
    cache_path.parent.mkdir(parents=True)
    py_compile.compile(
        str(entry.dist.locate_file("driver_package/backend.py")),
        cfile=str(cache_path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    assert cache_path.is_file()

    original_dont_write_bytecode = sys.dont_write_bytecode
    for _attempt in range(2):
        loaded = registry_module.discover_installed_backends(
            sdf_tools.SdfBackendRegistry(),
            backend="authenticated",
            entry_points=[_real_entry_point(entry)],
        )
        assert loaded == ("authenticated",)
        assert cache_path.is_file()
        assert sys.dont_write_bytecode is original_dont_write_bytecode


def test_preloaded_module_from_a_restored_approved_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
    isolated_authenticated_source_finder: list[str],
) -> None:
    extension_value, _backend = sdf_backend_factory("authenticated")
    approved_bytes = b"from _sdf_test_extension_holder import extension\n"
    malicious_bytes = b"extension = 'preloaded-unapproved-value'\n"
    policy = _admission_policy(extension_value)
    admission = policy["backends"]["authenticated"]  # type: ignore[index]
    admission["entry_point"] = "preloaded_driver:extension"  # type: ignore[index]
    admission["entry_point_module_path"] = "preloaded_driver.py"  # type: ignore[index]
    admission["entry_point_module_sha256"] = hashlib.sha256(approved_bytes).hexdigest()  # type: ignore[index]
    admission["authenticated_files"] = {  # type: ignore[index]
        "preloaded_driver.py": hashlib.sha256(approved_bytes).hexdigest(),
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    monkeypatch.setitem(
        sys.modules,
        "_sdf_test_extension_holder",
        SimpleNamespace(extension=extension_value),
    )
    isolated_authenticated_source_finder.append("preloaded_driver")

    entry = _entry_point(
        "authenticated",
        "preloaded_driver:extension",
        "authenticated-driver",
        extension_value,
        module_bytes=malicious_bytes,
    )
    monkeypatch.syspath_prepend(str(entry.dist.locate_file("")))
    preloaded = importlib.import_module("preloaded_driver")
    assert preloaded.extension == "preloaded-unapproved-value"
    entry.dist.locate_file("preloaded_driver.py").write_bytes(approved_bytes)

    with pytest.raises(sdf_tools.BackendUnavailableError, match="imported before authenticated"):
        registry_module.discover_installed_backends(
            sdf_tools.SdfBackendRegistry(),
            backend="authenticated",
            entry_points=[_real_entry_point(entry)],
        )

    assert sys.modules["preloaded_driver"] is preloaded
    assert preloaded.extension == "preloaded-unapproved-value"


def test_authenticated_source_loader_prevents_an_earlier_sys_path_shadow(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
    tmp_path: Path,
    isolated_authenticated_source_finder: list[str],
) -> None:
    extension_value, _backend = sdf_backend_factory("authenticated")
    module_name = "path_bound_driver"
    holder_name = "_sdf_shadow_extension_holder"
    approved_bytes = f"from {holder_name} import extension\n".encode()
    policy = _admission_policy(extension_value)
    admission = policy["backends"]["authenticated"]  # type: ignore[index]
    admission["entry_point"] = f"{module_name}:extension"  # type: ignore[index]
    admission["entry_point_module_path"] = f"{module_name}.py"  # type: ignore[index]
    admission["entry_point_module_sha256"] = hashlib.sha256(approved_bytes).hexdigest()  # type: ignore[index]
    admission["authenticated_files"] = {  # type: ignore[index]
        f"{module_name}.py": hashlib.sha256(approved_bytes).hexdigest(),
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    monkeypatch.setitem(
        sys.modules,
        holder_name,
        SimpleNamespace(extension=extension_value),
    )
    isolated_authenticated_source_finder.append(module_name)

    entry = _entry_point(
        "authenticated",
        f"{module_name}:extension",
        "authenticated-driver",
        extension_value,
        module_bytes=approved_bytes,
    )
    approved_path = entry.dist.locate_file(f"{module_name}.py").resolve()
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir()
    marker = tmp_path / "shadow-executed"
    (shadow_root / f"{module_name}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(shadow_root))
    ordinary_spec = importlib.machinery.PathFinder.find_spec(module_name)
    assert ordinary_spec is not None
    assert Path(str(ordinary_spec.origin)).resolve() == shadow_root / f"{module_name}.py"

    loaded = registry_module.discover_installed_backends(
        sdf_tools.SdfBackendRegistry(),
        backend="authenticated",
        entry_points=[_real_entry_point(entry)],
    )

    assert loaded == ("authenticated",)
    assert not marker.exists()
    assert Path(sys.modules[module_name].__file__).resolve() == approved_path


@pytest.mark.parametrize(
    ("artifact_path", "executable", "message"),
    [
        ("driver_package/Mod.PY", False, "files differ from policy"),
        ("driver_package/import-hook.pth", False, "prohibited executable import artifact"),
        ("driver_package/native.so", False, "prohibited executable import artifact"),
        ("driver_package/launch", True, "unrecorded executable member"),
    ],
)
def test_driver_authentication_rejects_other_executable_package_members_before_load(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
    artifact_path: str,
    executable: bool,
    message: str,
) -> None:
    extension_value, _backend = sdf_backend_factory("authenticated")
    policy = _admission_policy(extension_value)
    admission = policy["backends"]["authenticated"]  # type: ignore[index]
    admission["entry_point"] = "driver_package.backend:extension"  # type: ignore[index]
    admission["entry_point_module_path"] = "driver_package/backend.py"  # type: ignore[index]
    admission["authenticated_files"] = {  # type: ignore[index]
        "driver_package/backend.py": _DRIVER_MODULE_SHA256,
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    load_called = False

    def load():
        nonlocal load_called
        load_called = True
        return extension_value

    entry = _entry_point(
        "authenticated",
        "driver_package.backend:extension",
        "authenticated-driver",
        extension_value,
        load=load,
    )
    entry.dist.add_file(artifact_path, b"unapproved executable content\n", record=False)
    if executable:
        entry.dist.locate_file(artifact_path).chmod(0o700)

    with pytest.raises(sdf_tools.BackendUnavailableError, match=message):
        registry_module.discover_installed_backends(
            sdf_tools.SdfBackendRegistry(),
            backend="authenticated",
            entry_points=[entry],
        )

    assert load_called is False


def test_driver_authentication_rejects_a_tampered_build_manifest_before_load(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
) -> None:
    extension_value, _backend = sdf_backend_factory("authenticated")
    manifest_bytes = b'{"source_lock_sha256":"approved"}\n'
    policy = _admission_policy(extension_value)
    admission = policy["backends"]["authenticated"]  # type: ignore[index]
    admission["entry_point"] = "driver_package.backend:extension"  # type: ignore[index]
    admission["entry_point_module_path"] = "driver_package/backend.py"  # type: ignore[index]
    admission["authenticated_files"] = {  # type: ignore[index]
        "driver_package/backend.py": _DRIVER_MODULE_SHA256,
    }
    admission["authenticated_data_files"] = {  # type: ignore[index]
        "driver_package/_build_manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    load_called = False

    def load():
        nonlocal load_called
        load_called = True
        return extension_value

    entry = _entry_point(
        "authenticated",
        "driver_package.backend:extension",
        "authenticated-driver",
        extension_value,
        load=load,
    )
    entry.dist.add_file(
        "driver_package/_build_manifest.json",
        b'{"source_lock_sha256":"redirected"}\n',
    )

    with pytest.raises(sdf_tools.BackendUnavailableError, match="not policy-approved"):
        registry_module.discover_installed_backends(
            sdf_tools.SdfBackendRegistry(),
            backend="authenticated",
            entry_points=[entry],
        )

    assert load_called is False


def test_auto_discovery_isolates_load_and_descriptor_failures(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
) -> None:
    healthy, _backend = sdf_backend_factory("healthy")
    load_failure_called = False

    def admission(
        backend_id: str,
        module_name: str,
        *,
        implementation_version: str,
        license_manifest_sha256: str,
    ) -> dict[str, object]:
        return {
            "authenticated_data_files": {},
            "authenticated_files": {f"{module_name}.py": _DRIVER_MODULE_SHA256},
            "distribution": f"{backend_id}-driver",
            "distribution_version": "1.0",
            "entry_point": f"{module_name}:extension",
            "entry_point_module_path": f"{module_name}.py",
            "entry_point_module_sha256": _DRIVER_MODULE_SHA256,
            "implementation_version": implementation_version,
            "license_manifest_sha256": license_manifest_sha256,
            "native_closure_attestation": None,
            "production_qualified": True,
        }

    policy = {
        "schema": "world-understanding.sdf-backend-policy.v1",
        "entry_point_group": "sdf_tools.backends",
        "backends": {
            "a-load-failure": admission(
                "a-load-failure",
                "load_failure",
                implementation_version="1.0",
                license_manifest_sha256="0" * 64,
            ),
            "b-descriptor-failure": admission(
                "b-descriptor-failure",
                "descriptor_failure",
                implementation_version="1.0",
                license_manifest_sha256="0" * 64,
            ),
            "healthy": admission(
                "healthy",
                "healthy",
                implementation_version=healthy.descriptor.implementation_version,
                license_manifest_sha256=healthy.descriptor.license_manifest.sha256,
            ),
        },
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)

    def fail_during_load():
        nonlocal load_failure_called
        load_failure_called = True
        raise ImportError("authenticated driver failed to import")

    entries = [
        _entry_point(
            "a-load-failure",
            "load_failure:extension",
            "a-load-failure-driver",
            object(),
            load=fail_during_load,
        ),
        _entry_point(
            "b-descriptor-failure",
            "descriptor_failure:extension",
            "b-descriptor-failure-driver",
            object(),
        ),
        _entry_point(
            "healthy",
            "healthy:extension",
            "healthy-driver",
            healthy,
        ),
    ]
    registry = sdf_tools.SdfBackendRegistry()

    loaded = registry_module.discover_installed_backends(registry, entry_points=entries)

    assert load_failure_called is True
    assert loaded == ("healthy",)
    assert tuple(registry.snapshot()) == ("healthy",)


def test_production_registration_rejects_a_nonqualified_policy_record(
    monkeypatch: pytest.MonkeyPatch,
    sdf_backend_factory,
) -> None:
    extension_value, _backend = sdf_backend_factory("experimental")
    policy = _admission_policy(extension_value)
    policy["backends"]["experimental"]["production_qualified"] = False  # type: ignore[index]
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    entries = [
        _entry_point(
            "experimental",
            "experimental:extension",
            "experimental-driver",
            extension_value,
        )
    ]

    with pytest.raises(sdf_tools.BackendUnavailableError, match="not production-qualified"):
        registry_module.discover_installed_backends(
            sdf_tools.SdfBackendRegistry(),
            backend="experimental",
            entry_points=entries,
        )


@pytest.mark.parametrize(
    "expression",
    [
        "LGPL-2.1-only",
        "LGPL-3.0-or-later",
        "MIT AND LGPL-2.1+",
        "GNU Lesser General Public License",
    ],
)
def test_backend_introduced_lgpl_is_rejected(expression: str) -> None:
    manifest = sdf_tools.BackendLicenseManifest(
        schema="world-understanding.sdf-backend-license.v1",
        claim="fixture",
        components=(
            sdf_tools.LicenseComponent(
                "forbidden",
                "1",
                expression,
                sdf_tools.DependencyScope.RUNTIME,
                True,
            ),
        ),
    )
    with pytest.raises(sdf_tools.LicensePolicyError, match="forbidden LGPL"):
        sdf_tools.validate_license_manifest(manifest)


def test_preexisting_platform_lgpl_is_recorded_but_not_misclassified(
    sdf_license_manifest_factory,
) -> None:
    manifest = sdf_license_manifest_factory()
    extended = sdf_tools.BackendLicenseManifest(
        schema=manifest.schema,
        claim=manifest.claim,
        components=manifest.components
        + (
            sdf_tools.LicenseComponent(
                "glibc",
                "platform",
                "LGPL-2.1-or-later",
                sdf_tools.DependencyScope.PLATFORM_ABI,
                False,
            ),
        ),
    )

    sdf_tools.validate_license_manifest(extended)


@pytest.mark.parametrize("expression", ["GPL-3.0-only", "AGPL-3.0-or-later"])
def test_backend_introduced_strong_copyleft_is_rejected(expression: str) -> None:
    manifest = sdf_tools.BackendLicenseManifest(
        schema="world-understanding.sdf-backend-license.v1",
        claim="fixture",
        components=(
            sdf_tools.LicenseComponent(
                "forbidden",
                "1",
                expression,
                sdf_tools.DependencyScope.RUNTIME,
                True,
            ),
        ),
    )

    with pytest.raises(sdf_tools.LicensePolicyError, match="forbidden copyleft"):
        sdf_tools.validate_license_manifest(manifest)


@pytest.mark.parametrize(
    ("scope", "introduced"),
    [
        (sdf_tools.DependencyScope.BUNDLED, False),
        (sdf_tools.DependencyScope.RUNTIME, False),
        (sdf_tools.DependencyScope.BUILD, False),
        (sdf_tools.DependencyScope.PLATFORM_ABI, True),
        (sdf_tools.DependencyScope.PREEXISTING_APPLICATION, True),
    ],
)
def test_dependency_scope_cannot_be_relabelled(scope, introduced: bool) -> None:
    manifest = sdf_tools.BackendLicenseManifest(
        schema="world-understanding.sdf-backend-license.v1",
        claim="fixture",
        native_closure_attestation=(
            "sha256:" + "1" * 64 if scope is sdf_tools.DependencyScope.BUNDLED else None
        ),
        components=(sdf_tools.LicenseComponent("misclassified", "1", "MIT", scope, introduced),),
    )

    with pytest.raises(sdf_tools.LicensePolicyError, match="must be classified"):
        sdf_tools.validate_license_manifest(manifest)


def test_unknown_introduced_license_is_rejected() -> None:
    manifest = sdf_tools.BackendLicenseManifest(
        schema="world-understanding.sdf-backend-license.v1",
        claim="fixture",
        components=(
            sdf_tools.LicenseComponent(
                "unreviewed",
                "1",
                "LicenseRef-Custom",
                sdf_tools.DependencyScope.RUNTIME,
                True,
            ),
        ),
    )

    with pytest.raises(sdf_tools.LicensePolicyError, match="unapproved SPDX"):
        sdf_tools.validate_license_manifest(manifest)


@pytest.mark.parametrize("attestation", [None, "fixture"])
def test_bundled_component_requires_native_closure_attestation(attestation: str | None) -> None:
    manifest = sdf_tools.BackendLicenseManifest(
        schema="world-understanding.sdf-backend-license.v1",
        claim="fixture",
        native_closure_attestation=attestation,
        components=(
            sdf_tools.LicenseComponent(
                "native-library",
                "1",
                "Apache-2.0",
                sdf_tools.DependencyScope.BUNDLED,
                True,
            ),
        ),
    )

    with pytest.raises(sdf_tools.LicensePolicyError, match="closure-policy attestation"):
        sdf_tools.validate_license_manifest(manifest)


class _FixtureDistribution:
    def __init__(
        self,
        name: str,
        module_path: str,
        *,
        version: str,
        module_bytes: bytes,
        record_module: bool,
    ) -> None:
        self.name = name
        self.version = version
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._root = Path(self._temporary_directory.name)
        path = self._root / module_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(module_bytes)
        self.files = (PurePosixPath(module_path),) if record_module else ()

    def add_file(self, module_path: str, content: bytes, *, record: bool = True) -> None:
        path = self._root / module_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if record:
            self.files = (*self.files, PurePosixPath(module_path))

    def locate_file(self, path: object) -> Path:
        return self._root / str(path)

    def read_text(self, _filename: str) -> None:
        return None


def _entry_point(
    name: str,
    value: str,
    distribution: str,
    loaded: object,
    *,
    distribution_version: str = "1.0",
    module_bytes: bytes = _DRIVER_MODULE_BYTES,
    record_module: bool = True,
    load=None,
):
    module_name = value.partition(":")[0]
    module_path = f"{module_name.replace('.', '/')}.py"
    return SimpleNamespace(
        name=name,
        value=value,
        group="sdf_tools.backends",
        dist=_FixtureDistribution(
            distribution,
            module_path,
            version=distribution_version,
            module_bytes=module_bytes,
            record_module=record_module,
        ),
        load=load or (lambda: loaded),
    )


def test_explicit_discovery_ignores_an_unrelated_broken_plugin(
    monkeypatch, sdf_backend_factory
) -> None:
    good, _backend = sdf_backend_factory("good")
    policy = {
        "schema": "world-understanding.sdf-backend-policy.v1",
        "entry_point_group": "sdf_tools.backends",
        "backends": {
            "broken": {
                "authenticated_data_files": {},
                "authenticated_files": {"broken.py": _DRIVER_MODULE_SHA256},
                "distribution": "broken-driver",
                "distribution_version": "1.0",
                "entry_point": "broken:extension",
                "entry_point_module_path": "broken.py",
                "entry_point_module_sha256": _DRIVER_MODULE_SHA256,
                "implementation_version": "1.0",
                "license_manifest_sha256": "0" * 64,
                "native_closure_attestation": None,
                "production_qualified": True,
            },
            "good": {
                "authenticated_data_files": {},
                "authenticated_files": {"good.py": _DRIVER_MODULE_SHA256},
                "distribution": "good-driver",
                "distribution_version": "1.0",
                "entry_point": "good:extension",
                "entry_point_module_path": "good.py",
                "entry_point_module_sha256": _DRIVER_MODULE_SHA256,
                "implementation_version": good.descriptor.implementation_version,
                "license_manifest_sha256": good.descriptor.license_manifest.sha256,
                "native_closure_attestation": None,
                "production_qualified": True,
            },
        },
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    entries = [
        _entry_point("broken", "wrong:target", "broken-driver", object()),
        _entry_point("good", "good:extension", "good-driver", good),
    ]
    registry = sdf_tools.SdfBackendRegistry()

    loaded = registry_module.discover_installed_backends(
        registry,
        backend="good",
        entry_points=entries,
    )

    assert loaded == ("good",)


def test_auto_discovery_loads_only_production_qualified_plugins(
    monkeypatch, sdf_backend_factory
) -> None:
    qualified, _backend = sdf_backend_factory("qualified")
    loaded_experimental = False

    def load_experimental():
        nonlocal loaded_experimental
        loaded_experimental = True
        raise AssertionError("experimental driver must not be loaded by auto discovery")

    policy = {
        "schema": "world-understanding.sdf-backend-policy.v1",
        "entry_point_group": "sdf_tools.backends",
        "backends": {
            "experimental": {
                "authenticated_data_files": {},
                "authenticated_files": {"experimental.py": _DRIVER_MODULE_SHA256},
                "distribution": "experimental-driver",
                "distribution_version": "1.0",
                "entry_point": "experimental:extension",
                "entry_point_module_path": "experimental.py",
                "entry_point_module_sha256": _DRIVER_MODULE_SHA256,
                "implementation_version": "1.0",
                "license_manifest_sha256": "0" * 64,
                "native_closure_attestation": None,
                "production_qualified": False,
            },
            "qualified": {
                "authenticated_data_files": {},
                "authenticated_files": {"qualified.py": _DRIVER_MODULE_SHA256},
                "distribution": "qualified-driver",
                "distribution_version": "1.0",
                "entry_point": "qualified:extension",
                "entry_point_module_path": "qualified.py",
                "entry_point_module_sha256": _DRIVER_MODULE_SHA256,
                "implementation_version": qualified.descriptor.implementation_version,
                "license_manifest_sha256": qualified.descriptor.license_manifest.sha256,
                "native_closure_attestation": None,
                "production_qualified": True,
            },
        },
    }
    monkeypatch.setattr(registry_module, "_load_policy", lambda: policy)
    entries = [
        _entry_point(
            "experimental",
            "experimental:extension",
            "experimental-driver",
            object(),
            load=load_experimental,
        ),
        _entry_point(
            "qualified",
            "qualified:extension",
            "qualified-driver",
            qualified,
        ),
    ]
    registry = sdf_tools.SdfBackendRegistry()

    loaded = registry_module.discover_installed_backends(registry, entry_points=entries)

    assert loaded == ("qualified",)
    assert loaded_experimental is False
