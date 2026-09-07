# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openvdb_runtime import (
    Capability,
    CapabilityUnavailableError,
    RuntimeUnavailableError,
    RuntimeVersionError,
    inspect_runtime,
    is_available,
    require_runtime,
    runtime,
)


class _DistributionPath(str):
    def as_posix(self) -> str:
        return str(self)


class _NativeDistribution:
    def __init__(
        self,
        root: Path,
        members: tuple[str, ...],
        version: str = "13.0.0+wu.3",
    ) -> None:
        self._root = root
        self.files = tuple(_DistributionPath(member) for member in members)
        self.version = version

    def locate_file(self, member: object) -> Path:
        return self._root / str(member)


def _promoted_release_record(
    *,
    source_lock_sha256: str,
    runtime_members: dict[str, str],
    native_members: dict[str, str],
) -> dict[str, object]:
    return {
        "schema": "world-understanding.sdf-native-release-lock.v1",
        "trust_model": "test fixture",
        "platforms": {
            "x86_64": {
                "status": "promoted",
                "source_lock_sha256": source_lock_sha256,
                "wheel_runtime_members": runtime_members,
                "wheel_native_members": native_members,
            }
        },
    }


def _write_member(root: Path, member: str, payload: bytes) -> None:
    path = root.joinpath(*member.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _admission_fixture(monkeypatch, tmp_path: Path) -> dict[str, Any]:
    policy = runtime._load_policy().copy()
    source_lock = json.dumps(
        {
            "schema": "world-understanding.openvdb-native-source-lock.v1",
            "source": {"commit": policy["openvdb_source_commit"]},
            "distribution": {"version": policy["openvdb_distribution_version"]},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    source_lock_sha256 = hashlib.sha256(source_lock).hexdigest()
    policy["openvdb_source_lock_sha256"] = source_lock_sha256
    payloads = {
        "openvdb/__init__.py": b"from .lib.openvdb import *\n",
        "openvdb/_source_lock.json": source_lock,
        "openvdb/lib/openvdb.cpython-312-x86_64-linux-gnu.so": b"extension",
        "openvdb/lib/libopenvdb.so.13.0.0": b"openvdb library",
        "openvdb.libs/libtbb-locked.so.12": b"tbb library",
        "openvdb-13.0.0+wu.3.dist-info/METADATA": (
            b"Metadata-Version: 2.4\nName: openvdb\nVersion: 13.0.0+wu.3\n"
        ),
        (
            "openvdb-13.0.0+wu.3.dist-info/licenses/NATIVE_DEPENDENCY_PROVENANCE.json"
        ): b'{"schema":"test"}\n',
    }
    for member, payload in payloads.items():
        _write_member(tmp_path, member, payload)
    runtime_members = {
        member: hashlib.sha256(payload).hexdigest() for member, payload in payloads.items()
    }
    native_members = {
        member: digest
        for member, digest in runtime_members.items()
        if runtime._is_native_member(member)
    }
    release = _promoted_release_record(
        source_lock_sha256=source_lock_sha256,
        runtime_members=runtime_members,
        native_members=native_members,
    )
    distribution = _NativeDistribution(tmp_path, tuple(payloads))
    monkeypatch.setattr(runtime, "_ADMITTED_MODULE", None)
    monkeypatch.setattr(runtime, "_ADMITTED_SYS_MODULES", None)
    monkeypatch.setattr(runtime, "_ADMITTED_DISTRIBUTION_VERSION", None)
    monkeypatch.setattr(runtime, "_ADMITTED_PLAN", None)
    for name in tuple(sys.modules):
        if name == "openvdb" or name.startswith("openvdb."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    import_paths = [
        entry
        for entry in sys.path
        if not (isinstance(entry, str) and (Path(entry or Path.cwd()) / "openvdb").exists())
    ]
    monkeypatch.setattr(sys, "path", [str(tmp_path), *import_paths])
    monkeypatch.setattr(runtime, "_load_policy", lambda: policy)
    monkeypatch.setattr(runtime, "_load_release_lock", lambda loaded_policy: release)
    monkeypatch.setattr(runtime, "_normalized_architecture", lambda: "x86_64")
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distributions",
        lambda **kwargs: iter((distribution,)),
    )
    return {
        "root": tmp_path,
        "policy": policy,
        "release": release,
        "distribution": distribution,
        "payloads": payloads,
        "runtime_members": runtime_members,
        "native_members": native_members,
    }


def _fake_locked_import(calls: list[str], monkeypatch):
    def load(plan):
        calls.append("import")
        wrapper = ModuleType("openvdb")
        wrapper.__file__ = str(plan.snapshot.loader_path("openvdb/__init__.py"))
        wrapper.LIBRARY_VERSION = (13, 0, 0)
        wrapper.FILE_FORMAT_VERSION = 224
        library = ModuleType("openvdb.lib")
        native = ModuleType("openvdb.lib.openvdb")
        native.__file__ = str(plan.snapshot.loader_path(plan.extension_member))
        monkeypatch.setitem(sys.modules, "openvdb", wrapper)
        monkeypatch.setitem(sys.modules, "openvdb.lib", library)
        monkeypatch.setitem(sys.modules, "openvdb.lib.openvdb", native)
        return wrapper

    return load


def test_inspect_runtime_reports_identity_and_capabilities(fake_openvdb):
    info = inspect_runtime()

    assert info.library_version == (13, 0, 0)
    assert info.file_format_version == 224
    assert info.module_path is None
    assert info.module_sha256 is None
    assert info.source_lock_path is None
    assert info.source_lock_sha256 is None
    assert info.source_distribution_version is None
    assert info.source_commit is None
    assert info.supports(
        Capability.TRANSFORMS,
        Capability.VDB_IO,
        Capability.NUMPY_TRANSFER,
        Capability.MESH_TO_LEVEL_SET,
        Capability.VOLUME_TO_MESH,
        Capability.CSG,
        Capability.LEVEL_SET_OFFSET,
        Capability.LEVEL_SET_FILTER,
        Capability.SCALAR_MEAN_FILTER,
        Capability.LEVEL_SET_NORMALIZE,
        Capability.LEVEL_SET_REBUILD,
        Capability.RESAMPLE_TO_MATCH,
        Capability.SAMPLE_VALUES,
        Capability.SAMPLE_GRADIENTS,
        Capability.MESH_TO_UNSIGNED_DISTANCE_FIELD,
        Capability.ACTIVE_VALUE_MASK,
        Capability.TOPOLOGY_TO_LEVEL_SET,
        Capability.EXTRACT_ENCLOSED_REGION,
        Capability.EXTENDED_VOLUME_TO_MESH,
    )
    assert info.as_dict()["capabilities"] == sorted(
        capability.value for capability in info.capabilities
    )


def test_require_runtime_rejects_other_openvdb_major(fake_openvdb):
    fake_openvdb.LIBRARY_VERSION = (12, 1, 0)

    with pytest.raises(RuntimeVersionError, match=r"13\.0\.0.*12\.1\.0"):
        require_runtime()


def test_require_runtime_reports_missing_capability(stock_openvdb):
    with pytest.raises(CapabilityUnavailableError, match="csg") as exc_info:
        require_runtime((Capability.CSG,))

    assert "csg" in exc_info.value.capabilities


def test_capabilities_are_granular(fake_openvdb):
    del fake_openvdb.tools.sample_gradients

    info = inspect_runtime()

    assert Capability.SAMPLE_VALUES in info.capabilities
    assert Capability.SAMPLE_GRADIENTS not in info.capabilities


def test_extended_meshing_requires_tools_api_v2(fake_openvdb):
    fake_openvdb.tools.API_VERSION = 1

    info = inspect_runtime()

    assert Capability.VOLUME_TO_MESH in info.capabilities
    assert Capability.EXTENDED_VOLUME_TO_MESH not in info.capabilities


def test_successful_admission_is_cached_only_after_post_check(monkeypatch, tmp_path):
    _admission_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )

    first = runtime.load_openvdb()
    second = runtime.load_openvdb()

    assert first is second
    assert calls == ["import"]


def test_admission_planning_precedes_global_import_lock_and_rechecks_modules(
    monkeypatch, tmp_path
) -> None:
    _admission_fixture(monkeypatch, tmp_path)
    original_plan = runtime._plan_runtime_admission
    plans = []
    foreign_module = ModuleType("openvdb")
    calls: list[str] = []

    def plan_runtime_admission(policy):
        assert not runtime._imp.lock_held()
        plan = original_plan(policy)
        plans.append(plan)
        monkeypatch.setitem(sys.modules, "openvdb", foreign_module)
        return plan

    monkeypatch.setattr(runtime, "_plan_runtime_admission", plan_runtime_admission)
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )

    with pytest.raises(RuntimeVersionError, match="imported before runtime admission"):
        runtime.load_openvdb()

    assert calls == []
    assert sys.modules["openvdb"] is foreign_module
    assert plans[0].snapshot.directory_fd == -1


def test_native_import_and_post_check_hold_global_import_lock(monkeypatch, tmp_path) -> None:
    _admission_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    fake_import = _fake_locked_import(calls, monkeypatch)
    original_verify = runtime._verify_post_import

    def import_locked_distribution(plan):
        assert runtime._imp.lock_held()
        return fake_import(plan)

    def verify_post_import(plan, module):
        assert runtime._imp.lock_held()
        return original_verify(plan, module)

    monkeypatch.setattr(runtime, "_import_locked_distribution", import_locked_distribution)
    monkeypatch.setattr(runtime, "_verify_post_import", verify_post_import)

    runtime.load_openvdb()

    assert calls == ["import"]
    assert not runtime._imp.lock_held()


def test_native_loader_uses_private_snapshot_during_installed_path_swap(
    monkeypatch, tmp_path
) -> None:
    fixture = _admission_fixture(monkeypatch, tmp_path)
    malicious_payload = b"malicious native replacement"
    observed_snapshot_members: dict[str, bytes] = {}
    malicious_code_executed: list[str] = []

    class SwappingExtensionLoader:
        def __init__(self, fullname: str, path: str) -> None:
            self.fullname = fullname
            self.path = path

        def create_module(self, spec):
            module = ModuleType(self.fullname)
            module.__file__ = self.path
            return module

        def exec_module(self, module: ModuleType) -> None:
            installed_native = {
                member: fixture["root"].joinpath(*member.split("/"))
                for member in fixture["native_members"]
            }
            original_payloads = {
                member: path.read_bytes() for member, path in installed_native.items()
            }
            try:
                for path in installed_native.values():
                    path.write_bytes(malicious_payload)
                snapshot_root = Path(self.path).parents[2]
                for member in fixture["native_members"]:
                    payload = snapshot_root.joinpath(*member.split("/")).read_bytes()
                    observed_snapshot_members[member] = payload
                    if payload == malicious_payload:
                        malicious_code_executed.append(member)
            finally:
                for member, path in installed_native.items():
                    path.write_bytes(original_payloads[member])
            module.__file__ = self.path

    monkeypatch.setattr(runtime, "ExtensionFileLoader", SwappingExtensionLoader)

    module = runtime.load_openvdb()

    assert module is sys.modules["openvdb"]
    assert malicious_code_executed == []
    assert observed_snapshot_members == {
        member: fixture["payloads"][member] for member in fixture["native_members"]
    }


def test_private_snapshot_mutation_is_rejected_before_native_loader_runs(
    monkeypatch, tmp_path
) -> None:
    fixture = _admission_fixture(monkeypatch, tmp_path)
    plan = runtime._plan_runtime_admission(fixture["policy"])
    extension = plan.snapshot.extension_path
    extension.chmod(0o600)
    extension.write_bytes(b"snapshot replacement")
    calls: list[str] = []

    class RejectingExtensionLoader:
        def __init__(self, fullname: str, path: str) -> None:
            calls.append(fullname)

    monkeypatch.setattr(runtime, "ExtensionFileLoader", RejectingExtensionLoader)

    with pytest.raises(RuntimeVersionError, match="snapshot handle changed"):
        runtime._import_locked_distribution(plan)

    assert calls == []


@pytest.mark.parametrize(
    "member",
    [
        "openvdb/__init__.py",
        "openvdb/_source_lock.json",
        "openvdb/lib/openvdb.cpython-312-x86_64-linux-gnu.so",
        "openvdb/lib/libopenvdb.so.13.0.0",
        "openvdb.libs/libtbb-locked.so.12",
        "openvdb-13.0.0+wu.3.dist-info/METADATA",
        "openvdb-13.0.0+wu.3.dist-info/licenses/NATIVE_DEPENDENCY_PROVENANCE.json",
    ],
)
def test_cached_runtime_is_isolated_from_mutated_installed_member(
    monkeypatch, tmp_path, member: str
) -> None:
    fixture = _admission_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )
    admitted = runtime.load_openvdb()
    monkeypatch.setattr(runtime, "detect_capabilities", lambda _module=None: frozenset(Capability))
    fixture["root"].joinpath(*member.split("/")).write_bytes(b"mutated after admission")

    assert runtime.require_runtime() is admitted

    with pytest.raises(
        RuntimeVersionError, match=rf"differs from release lock: {re.escape(member)}"
    ):
        runtime.inspect_runtime(admitted)

    assert runtime._ADMITTED_MODULE is admitted
    assert calls == ["import"]


def test_explicit_inspection_revalidates_cached_member_bytes(monkeypatch, tmp_path) -> None:
    fixture = _admission_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )
    admitted = runtime.load_openvdb()
    extension = fixture["root"] / "openvdb" / "lib" / ("openvdb.cpython-312-x86_64-linux-gnu.so")
    extension.write_bytes(b"mutated after admission")

    with pytest.raises(RuntimeVersionError, match="differs from release lock"):
        runtime.inspect_runtime(admitted)

    assert calls == ["import"]


def test_cached_runtime_is_isolated_from_new_installed_executable(monkeypatch, tmp_path) -> None:
    fixture = _admission_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )
    admitted = runtime.load_openvdb()
    monkeypatch.setattr(runtime, "detect_capabilities", lambda _module=None: frozenset(Capability))
    _write_member(fixture["root"], "openvdb/late.py", b"raise RuntimeError\n")

    assert runtime.require_runtime() is admitted

    with pytest.raises(RuntimeVersionError, match="unrecorded executable member"):
        runtime.inspect_runtime(admitted)

    assert calls == ["import"]


def test_repeated_require_runtime_avoids_deep_revalidation(monkeypatch, tmp_path) -> None:
    _admission_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )
    admitted = runtime.load_openvdb()
    monkeypatch.setattr(runtime, "detect_capabilities", lambda _module=None: frozenset(Capability))

    def reject_deep_revalidation(*_args, **_kwargs):
        raise AssertionError("cached require_runtime performed deep revalidation")

    monkeypatch.setattr(runtime, "_verify_runtime_snapshot", reject_deep_revalidation)
    monkeypatch.setattr(runtime, "_verify_runtime_members", reject_deep_revalidation)
    monkeypatch.setattr(runtime, "_scan_runtime_trees", reject_deep_revalidation)
    monkeypatch.setattr(runtime, "_module_digest", reject_deep_revalidation)
    monkeypatch.setattr(runtime, "_source_lock_identity", reject_deep_revalidation)

    for _ in range(20):
        assert runtime.require_runtime() is admitted

    with pytest.raises(AssertionError, match="deep revalidation"):
        runtime.inspect_runtime(admitted)
    assert calls == ["import"]


def test_failed_post_import_check_is_not_cached(monkeypatch, tmp_path):
    _admission_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )

    def reject_post_check(plan, module):
        raise RuntimeVersionError("post-check")

    monkeypatch.setattr(runtime, "_verify_post_import", reject_post_check)

    with pytest.raises(RuntimeVersionError, match="post-check"):
        runtime.load_openvdb()

    assert calls == ["import"]
    assert runtime._ADMITTED_MODULE is None
    assert not any(name == "openvdb" or name.startswith("openvdb.") for name in sys.modules)


@pytest.mark.parametrize(
    "tamper",
    [
        "wrapper",
        "native",
        "metadata",
        "provenance",
        "candidate",
        "foreign_module",
        "shadowing",
        "executable",
    ],
)
def test_no_import_callback_runs_before_complete_admission(monkeypatch, tmp_path, tamper: str):
    fixture = _admission_fixture(monkeypatch, tmp_path)
    root = fixture["root"]
    release = fixture["release"]
    if tamper == "wrapper":
        (root / "openvdb" / "__init__.py").write_bytes(b"raise RuntimeError\n")
    elif tamper == "native":
        (root / "openvdb" / "lib" / "openvdb.cpython-312-x86_64-linux-gnu.so").write_bytes(
            b"modified"
        )
    elif tamper == "metadata":
        (root / "openvdb-13.0.0+wu.3.dist-info" / "METADATA").write_bytes(b"changed")
    elif tamper == "provenance":
        (
            root
            / "openvdb-13.0.0+wu.3.dist-info"
            / "licenses"
            / "NATIVE_DEPENDENCY_PROVENANCE.json"
        ).write_bytes(b"changed")
    elif tamper == "candidate":
        release["platforms"]["x86_64"]["status"] = "candidate"
    elif tamper == "foreign_module":
        monkeypatch.setitem(sys.modules, "openvdb", ModuleType("openvdb"))
    elif tamper == "shadowing":
        shadow = tmp_path / "shadow"
        (shadow / "openvdb").mkdir(parents=True)
        (shadow / "openvdb" / "__init__.py").write_bytes(b"")
        sys.path.insert(0, str(shadow))
    elif tamper == "executable":
        executable = root / "openvdb" / "late-hook"
        executable.write_bytes(b"unrecorded executable")
        executable.chmod(0o700)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )

    with pytest.raises(RuntimeVersionError):
        runtime.load_openvdb()

    assert calls == []


def test_shadowing_admission_fixture_restores_sys_path(tmp_path: Path) -> None:
    original_path = sys.path
    original_entries = list(sys.path)

    with pytest.MonkeyPatch.context() as monkeypatch:
        fixture = _admission_fixture(monkeypatch, tmp_path)
        shadow = tmp_path / "shadow"
        (shadow / "openvdb").mkdir(parents=True)
        (shadow / "openvdb" / "__init__.py").write_bytes(b"")
        sys.path.insert(0, str(shadow))

        with pytest.raises(RuntimeVersionError, match="shadowing or multiple roots"):
            runtime._plan_runtime_admission(fixture["policy"])

    assert sys.path is original_path
    assert sys.path == original_entries


@pytest.mark.parametrize("entrypoint", ["inspect", "require", "available"])
def test_default_runtime_entrypoints_cannot_bypass_preimport_admission(
    monkeypatch, tmp_path, entrypoint: str
):
    fixture = _admission_fixture(monkeypatch, tmp_path)
    (fixture["root"] / "openvdb" / "__init__.py").write_bytes(b"tampered")
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )

    if entrypoint == "available":
        assert not is_available()
    else:
        function = inspect_runtime if entrypoint == "inspect" else require_runtime
        with pytest.raises(RuntimeVersionError, match="differs from release lock"):
            function()

    assert calls == []


@pytest.mark.parametrize(
    "member", ["openvdb/extra.py", "openvdb/extra.pth", "openvdb.libs/extra.so"]
)
def test_unrecorded_executable_members_are_rejected(monkeypatch, tmp_path, member: str):
    fixture = _admission_fixture(monkeypatch, tmp_path)
    _write_member(fixture["root"], member, b"unrecorded")

    with pytest.raises(RuntimeVersionError, match="unrecorded executable"):
        runtime._plan_runtime_admission(fixture["policy"])


def test_inert_installer_bytecode_caches_are_ignored(monkeypatch, tmp_path) -> None:
    fixture = _admission_fixture(monkeypatch, tmp_path)
    cache = fixture["root"] / "openvdb" / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"installer bytecode cache")
    (cache / "legacy.pyo").write_bytes(b"optimized installer bytecode cache")
    (fixture["root"] / "openvdb" / "legacy.pyc").write_bytes(b"legacy bytecode cache")

    plan = runtime._plan_runtime_admission(fixture["policy"])

    assert not (plan.snapshot.root / "openvdb" / "__pycache__").exists()
    assert not (plan.snapshot.root / "openvdb" / "legacy.pyc").exists()
    plan.snapshot.close()


def test_symlinked_installer_bytecode_cache_is_rejected(monkeypatch, tmp_path) -> None:
    fixture = _admission_fixture(monkeypatch, tmp_path)
    cache_target = fixture["root"] / "external-cache"
    cache_target.mkdir()
    (fixture["root"] / "openvdb" / "__pycache__").symlink_to(
        cache_target,
        target_is_directory=True,
    )

    with pytest.raises(RuntimeVersionError, match="runtime path is unsafe"):
        runtime._plan_runtime_admission(fixture["policy"])


def test_runtime_member_symlink_is_rejected(monkeypatch, tmp_path):
    fixture = _admission_fixture(monkeypatch, tmp_path)
    wrapper = fixture["root"] / "openvdb" / "__init__.py"
    wrapper.unlink()
    wrapper.symlink_to(fixture["root"] / "openvdb" / "_source_lock.json")

    with pytest.raises(RuntimeVersionError, match="symlink"):
        runtime._plan_runtime_admission(fixture["policy"])


def test_runtime_member_paths_must_be_canonical(monkeypatch, tmp_path):
    fixture = _admission_fixture(monkeypatch, tmp_path)
    selected = fixture["release"]["platforms"]["x86_64"]
    selected["wheel_runtime_members"]["openvdb/../escape.py"] = "0" * 64

    with pytest.raises(RuntimeVersionError, match="malformed"):
        runtime._plan_runtime_admission(fixture["policy"])


def test_multiple_distributions_are_rejected(monkeypatch, tmp_path):
    fixture = _admission_fixture(monkeypatch, tmp_path)
    distribution = fixture["distribution"]
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distributions",
        lambda **kwargs: iter((distribution, distribution)),
    )

    with pytest.raises(RuntimeVersionError, match="exactly one"):
        runtime._plan_runtime_admission(fixture["policy"])


def test_zip_shadow_root_is_rejected(monkeypatch, tmp_path):
    fixture = _admission_fixture(monkeypatch, tmp_path)
    shadow = tmp_path / "shadow.zip"
    with zipfile.ZipFile(shadow, "w") as archive:
        archive.writestr("openvdb/__init__.py", "")
    sys.path.append(str(shadow))

    with pytest.raises(RuntimeVersionError, match="shadowing or multiple roots"):
        runtime._plan_runtime_admission(fixture["policy"])


def test_release_lock_bytes_are_bound_by_driver_policy(monkeypatch, tmp_path):
    release_path = tmp_path / "release-lock.json"
    payload = (
        b'{"platforms":{},"schema":"world-understanding.sdf-native-release-lock.v1",'
        b'"trust_model":"test"}\n'
    )
    release_path.write_bytes(payload)
    policy = {"openvdb_release_lock_sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(runtime, "_PACKAGED_RELEASE_LOCK_PATH", release_path)

    assert runtime._load_release_lock(policy)["trust_model"] == "test"

    release_path.write_bytes(payload + b" ")
    with pytest.raises(RuntimeVersionError, match="release-lock digest mismatch"):
        runtime._load_release_lock(policy)


def test_runtime_policy_bytes_are_bound_by_authenticated_driver_source(
    monkeypatch, tmp_path
) -> None:
    policy_path = tmp_path / "_build_manifest.json"
    approved = runtime._POLICY_PATH.read_bytes()
    policy_path.write_bytes(approved)
    monkeypatch.setattr(runtime, "_POLICY_PATH", policy_path)

    assert runtime._load_policy()["schema"] == "world-understanding.openvdb-runtime-policy.v1"

    malicious = json.loads(approved)
    malicious["openvdb_release_lock_sha256"] = "0" * 64
    policy_path.write_text(json.dumps(malicious), encoding="utf-8")
    with pytest.raises(RuntimeVersionError, match="driver-embedded identity"):
        runtime._load_policy()


def test_inert_installer_metadata_extras_are_allowed(monkeypatch, tmp_path):
    fixture = _admission_fixture(monkeypatch, tmp_path)
    distribution = fixture["distribution"]
    installer = "openvdb-13.0.0+wu.3.dist-info/INSTALLER"
    direct_url = "openvdb-13.0.0+wu.3.dist-info/direct_url.json"
    _write_member(tmp_path, installer, b"uv\n")
    _write_member(tmp_path, direct_url, b"{}\n")
    distribution.files += (_DistributionPath(installer), _DistributionPath(direct_url))

    plan = runtime._plan_runtime_admission(fixture["policy"])

    assert plan.distribution_root == tmp_path


def test_prepopulated_child_module_is_rejected_before_import(monkeypatch, tmp_path):
    _admission_fixture(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "openvdb.lib", ModuleType("openvdb.lib"))
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_import_locked_distribution",
        _fake_locked_import(calls, monkeypatch),
    )

    with pytest.raises(RuntimeVersionError, match="imported before runtime admission"):
        runtime.load_openvdb()

    assert calls == []


def test_explicit_unadmitted_module_cannot_enter_public_inspection():
    module = ModuleType("openvdb")
    module.LIBRARY_VERSION = (13, 0, 0)

    with pytest.raises(RuntimeVersionError, match="not admitted"):
        inspect_runtime(module)


def test_inspection_reports_embedded_source_lock(fake_openvdb, tmp_path):
    package = tmp_path / "openvdb"
    package.mkdir()
    fake_openvdb.__file__ = str(package / "__init__.py")
    source_lock = package / "_source_lock.json"
    payload = (
        b'{"schema":"world-understanding.openvdb-native-source-lock.v1",'
        b'"source":{"commit":"1111111111111111111111111111111111111111"},'
        b'"distribution":{"version":"13.0.0+wu.3"}}'
    )
    source_lock.write_bytes(payload)

    info = inspect_runtime()

    assert info.source_lock_path == str(source_lock)
    assert info.source_lock_sha256 == hashlib.sha256(payload).hexdigest()
    assert info.source_distribution_version == "13.0.0+wu.3"
    assert info.source_commit == "1" * 40


@pytest.mark.parametrize("commit", ["", "A" * 40, "1" * 39, "not-a-commit"])
def test_inspection_rejects_malformed_embedded_source_commit(fake_openvdb, tmp_path, commit):
    package = tmp_path / "openvdb"
    package.mkdir()
    fake_openvdb.__file__ = str(package / "__init__.py")
    (package / "_source_lock.json").write_text(
        '{"schema":"world-understanding.openvdb-native-source-lock.v1",'
        f'"source":{{"commit":"{commit}"}},'
        '"distribution":{"version":"13.0.0+wu.3"}}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeVersionError, match="invalid commit"):
        inspect_runtime()


def test_inspection_rejects_non_object_embedded_source_lock(fake_openvdb, tmp_path):
    package = tmp_path / "openvdb"
    package.mkdir()
    fake_openvdb.__file__ = str(package / "__init__.py")
    (package / "_source_lock.json").write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeVersionError, match="must be an object"):
        inspect_runtime()


def test_inspection_hashes_loaded_extension(fake_openvdb, tmp_path):
    module_path = tmp_path / "openvdb.so"
    module_path.write_bytes(b"native module fixture")
    fake_openvdb.__file__ = str(module_path)

    info = inspect_runtime()

    assert info.module_path == str(module_path)
    assert info.module_sha256 == hashlib.sha256(b"native module fixture").hexdigest()


def test_inspection_hashes_nanobind_module_behind_wrapper(fake_openvdb, monkeypatch, tmp_path):
    wrapper_path = tmp_path / "openvdb" / "__init__.py"
    wrapper_path.parent.mkdir()
    wrapper_path.write_text("from .lib.openvdb import *\n", encoding="utf-8")
    native_path = tmp_path / "openvdb" / "lib" / "openvdb.cpython-312.so"
    native_path.parent.mkdir()
    native_path.write_bytes(b"nanobind extension fixture")
    fake_openvdb.__file__ = str(wrapper_path)
    native_module = ModuleType("openvdb.lib.openvdb")
    native_module.__file__ = str(native_path)
    monkeypatch.setitem(sys.modules, "openvdb.lib.openvdb", native_module)

    info = runtime._inspect_unadmitted_module_for_tests(fake_openvdb)

    assert info.module_path == str(native_path)
    assert info.module_sha256 == hashlib.sha256(b"nanobind extension fixture").hexdigest()


def test_is_available_is_fail_closed(fake_openvdb):
    assert is_available(Capability.CSG)
    fake_openvdb.LIBRARY_VERSION = (14, 0, 0)
    assert not is_available(Capability.CSG)


def test_missing_module_has_actionable_error(monkeypatch):
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda **kwargs: iter(()))

    with pytest.raises(RuntimeUnavailableError, match="native wheel"):
        runtime._distribution()


def test_malformed_library_version_is_rejected():
    module = ModuleType("openvdb")
    module.LIBRARY_VERSION = "thirteen"

    with pytest.raises(RuntimeVersionError, match="malformed"):
        runtime._inspect_unadmitted_module_for_tests(module)


@pytest.mark.parametrize("file_format", ["224", 224.0, True, object()])
def test_malformed_file_format_version_is_rejected(file_format):
    module = ModuleType("openvdb")
    module.LIBRARY_VERSION = (13, 0, 0)
    module.FILE_FORMAT_VERSION = file_format

    with pytest.raises(RuntimeVersionError, match=r"FILE_FORMAT_VERSION.*integer"):
        runtime._inspect_unadmitted_module_for_tests(module)


@pytest.mark.parametrize(
    "version",
    [
        (13.9, 0, 0),
        ("13", "0", "0"),
        (13, False, 0),
    ],
)
def test_coercible_library_versions_are_rejected(version):
    module = ModuleType("openvdb")
    module.LIBRARY_VERSION = version

    with pytest.raises(RuntimeVersionError, match="three integers"):
        runtime._inspect_unadmitted_module_for_tests(module)
