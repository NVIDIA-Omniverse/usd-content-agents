# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class _NoopFileLock:
    def __init__(self, path: str, timeout: float) -> None:
        self.path = path
        self.timeout = timeout

    def __enter__(self) -> "_NoopFileLock":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_runtime_lock_is_used_with_hash_enforcement() -> None:
    from world_understanding.functions.graphics import render_ovrtx

    lock_file = render_ovrtx._OVRTX_RUNTIME_LOCK_FILE
    assert lock_file.exists()
    assert render_ovrtx._ovrtx_runtime_lock_args() == [
        "--require-hashes",
        "--no-deps",
        "-r",
        str(lock_file),
        "--no-config",
        "--no-sources",
    ]

    with lock_file.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = lock["packages"]
    assert {package["name"]: package["version"] for package in packages} == {
        "numpy": "2.2.6",
        "ovrtx": "0.3.0.312915",
        "pillow": "12.3.0",
    }
    assert all(
        len(wheel["hashes"]["sha256"]) == 64
        for package in packages
        for wheel in package["wheels"]
    )
    packages_by_name = {package["name"]: package for package in packages}
    for package_name in ("numpy", "ovrtx", "pillow"):
        wheel_urls = [
            wheel["url"] for wheel in packages_by_name[package_name]["wheels"]
        ]
        assert any("x86_64" in url for url in wheel_urls)
        assert any("aarch64" in url for url in wheel_urls)


def test_content_workbench_does_not_install_ovrtx_in_shared_environment() -> None:
    pyproject_path = REPO_ROOT / "agentic/packages/content_workbench/pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    assert not any(
        dependency.partition(";")[0].strip().startswith("ovrtx")
        for dependency in dependencies
    )
    assert "ovrtx" not in pyproject.get("tool", {}).get("uv", {}).get("sources", {})


def test_bundled_python_runtime_libraries_are_removed_from_ovrtx_package(
    tmp_path: Path,
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
    plugins_dir = site_dir / "ovrtx" / "bin" / "plugins"
    plugins_dir.mkdir(parents=True)

    bundled_python_libraries = [
        plugins_dir / "libpython3.12.so",
        plugins_dir / "libpython3.12.so.1.0",
        site_dir / "ovrtx" / "lib" / "libpython3.12.so",
    ]
    for library in bundled_python_libraries:
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(b"python-runtime")
    keep_library = plugins_dir / "libovrtx.so"
    keep_library.write_bytes(b"ovrtx")

    removed = render_ovrtx._remove_ovrtx_bundled_python_libraries(venv_dir)

    assert removed == bundled_python_libraries
    assert all(not library.exists() for library in bundled_python_libraries)
    assert keep_library.exists()


def test_new_ovrtx_venv_removes_bundled_python_before_import_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    plugins_dir = (
        render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        / "ovrtx"
        / "bin"
        / "plugins"
    )
    bundled_python_library = plugins_dir / "libpython3.12.so.1.0"

    def fake_run_checked(cmd: list[str], label: str) -> None:
        if label == "uv venv creation":
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")
        if label == "locked OVRTX runtime install":
            plugins_dir.mkdir(parents=True, exist_ok=True)
            bundled_python_library.write_bytes(b"python-runtime")

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        assert python_path_arg == python_path
        assert venv_dir_arg == venv_dir
        assert not bundled_python_library.exists()
        return render_ovrtx._OVRTX_VERSION

    monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert not bundled_python_library.exists()


def test_auto_provision_without_uv_fails_before_unlocked_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: None)
    monkeypatch.setattr(render_ovrtx.os.path, "exists", lambda path: False)

    venv_dir = tmp_path / "ovrtx_venv"
    with pytest.raises(RuntimeError, match="requires the uv executable"):
        render_ovrtx._get_ovrtx_python_unlocked(venv_dir)
    assert not venv_dir.exists()


def test_get_ovrtx_python_uses_cross_process_file_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    lock_calls: list[tuple[str, float]] = []

    class FakeFileLock:
        def __init__(self, path: str, timeout: float) -> None:
            lock_calls.append((path, timeout))

        def __enter__(self) -> "FakeFileLock":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(render_ovrtx, "FileLock", FakeFileLock)
    monkeypatch.setattr(
        render_ovrtx,
        "_probe_ovrtx_version",
        lambda python_path_arg, venv_dir_arg: render_ovrtx._OVRTX_VERSION,
    )

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert lock_calls == [
        (
            str(render_ovrtx._ovrtx_provision_lock_path(venv_dir)),
            render_ovrtx._OVRTX_PROVISION_LOCK_TIMEOUT_S,
        )
    ]


def test_provision_lock_path_honors_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("WU_OVRTX_LOCK_DIR", str(lock_dir))

    venv_dir = tmp_path / "cache" / "ovrtx_venv"
    lock_path = render_ovrtx._ovrtx_provision_lock_path(venv_dir)

    assert lock_path.parent == lock_dir
    assert lock_path.name.startswith("ovrtx_venv-")
    assert lock_path.name.endswith(".lock")


def test_auto_provision_disabled_preserves_existing_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    rmtree_calls: list[tuple[Path, bool]] = []

    monkeypatch.setattr(
        render_ovrtx,
        "_probe_ovrtx_version",
        lambda python_path_arg, venv_dir_arg: "0.2.0.280040",
    )
    monkeypatch.setattr(
        render_ovrtx.shutil,
        "rmtree",
        lambda path, ignore_errors=False: rmtree_calls.append(
            (Path(path), ignore_errors)
        ),
    )

    with pytest.raises(RuntimeError, match="AUTO_PROVISION is disabled"):
        render_ovrtx._get_ovrtx_python(venv_dir=venv_dir)
    assert rmtree_calls == []
    assert python_path.exists()


def test_provision_only_cli_calls_provisioner(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(
        render_ovrtx,
        "_get_ovrtx_python",
        lambda venv_dir=None: "/tmp/ovrtx/bin/python",
    )

    assert render_ovrtx._main(["--provision-only"]) == 0
    assert capsys.readouterr().out == "OvRTX Python ready: /tmp/ovrtx/bin/python\n"


def test_daemon_start_clears_pythonpath_in_daemon_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setenv("PYTHONPATH", "/app/pythonpath")
    monkeypatch.delenv("DISPLAY", raising=False)
    captured_env: dict[str, str] = {}

    class FakeProcess:
        pid = 12345
        stdin = None
        stdout = None
        stderr: list[str] = []

        def poll(self) -> None:
            return None

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        captured_env.update(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
    monkeypatch.setattr(render_ovrtx.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        render_ovrtx._OvRTXDaemon,
        "_read_stdout_line",
        lambda self, timeout_s, phase: json.dumps({"status": "ready"}),
    )

    daemon = render_ovrtx._OvRTXDaemon(
        ovrtx_python=str(tmp_path / "python"),
        daemon_script_path=str(tmp_path / "daemon.py"),
    )
    daemon.ensure_running()

    assert "PYTHONPATH" not in captured_env
    assert captured_env["DISPLAY"] == ":0"
    assert os.environ["PYTHONPATH"] == "/app/pythonpath"


def test_existing_wrong_version_venv_is_recreated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")

    versions = iter(["0.2.0.280040", render_ovrtx._OVRTX_VERSION])
    probe_calls: list[tuple[Path, Path]] = []

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        probe_calls.append((python_path_arg, venv_dir_arg))
        return next(versions)

    rmtree_calls: list[tuple[Path, bool]] = []
    real_rmtree = render_ovrtx.shutil.rmtree

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        rmtree_calls.append((Path(path), ignore_errors))
        real_rmtree(path, ignore_errors=ignore_errors)

    run_checked_calls: list[tuple[list[str], str]] = []

    def fake_run_checked(cmd: list[str], label: str) -> None:
        run_checked_calls.append((cmd, label))
        if label == "uv venv creation":
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")

    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
    monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert probe_calls == [(python_path, venv_dir), (python_path, venv_dir)]
    assert rmtree_calls == [(venv_dir, True)]
    assert [label for _, label in run_checked_calls] == [
        "uv venv creation",
        "locked OVRTX runtime install",
    ]


def test_managed_marker_without_version_is_reprovisioned_before_version_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n",
        encoding="utf-8",
    )

    probe_calls: list[tuple[Path, Path]] = []

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        probe_calls.append((python_path_arg, venv_dir_arg))
        return render_ovrtx._OVRTX_VERSION

    rmtree_calls: list[tuple[Path, bool]] = []
    real_rmtree = render_ovrtx.shutil.rmtree

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        rmtree_calls.append((Path(path), ignore_errors))
        real_rmtree(path, ignore_errors=ignore_errors)

    def fake_run_checked(cmd: list[str], label: str) -> None:
        if label == "uv venv creation":
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")

    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
    monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert probe_calls == [(python_path, venv_dir)]
    assert rmtree_calls == [(venv_dir, True)]
    marker = (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).read_text(encoding="utf-8")
    assert f"ovrtx_version={render_ovrtx._OVRTX_VERSION}" in marker
    assert f"runtime_lock_sha256={render_ovrtx._ovrtx_runtime_lock_digest()}" in marker


@pytest.mark.parametrize("recorded_digest", [None, "0" * 64])
def test_auto_provision_disabled_rejects_stale_managed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_digest: str | None,
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    marker = (
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
    )
    if recorded_digest is not None:
        marker += f"runtime_lock_sha256={recorded_digest}\n"
    marker_path = venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER
    marker_path.write_text(
        marker,
        encoding="utf-8",
    )
    original_marker = marker_path.read_text(encoding="utf-8")

    probe_calls: list[tuple[Path, Path]] = []

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        probe_calls.append((python_path_arg, venv_dir_arg))
        return render_ovrtx._OVRTX_VERSION

    rmtree_calls: list[tuple[Path, bool]] = []

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        rmtree_calls.append((Path(path), ignore_errors))

    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
    monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)

    with pytest.raises(RuntimeError, match="does not match the current runtime lock"):
        render_ovrtx._get_ovrtx_python(venv_dir=venv_dir)
    assert probe_calls == []
    assert rmtree_calls == []
    assert python_path.exists()
    assert marker_path.read_text(encoding="utf-8") == original_marker


@pytest.mark.parametrize("recorded_digest", [None, "0" * 64])
def test_current_ovrtx_managed_runtime_with_stale_lock_is_reprovisioned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_digest: str | None,
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    marker = (
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
    )
    if recorded_digest is not None:
        marker += f"runtime_lock_sha256={recorded_digest}\n"
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        marker,
        encoding="utf-8",
    )

    probe_calls: list[tuple[Path, Path]] = []

    def fake_probe(python_path_arg: Path, venv_dir_arg: Path) -> str:
        probe_calls.append((python_path_arg, venv_dir_arg))
        return render_ovrtx._OVRTX_VERSION

    rmtree_calls: list[tuple[Path, bool]] = []
    real_rmtree = render_ovrtx.shutil.rmtree

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        rmtree_calls.append((Path(path), ignore_errors))
        real_rmtree(path, ignore_errors=ignore_errors)

    install_calls: list[str] = []

    def fake_run_checked(cmd: list[str], label: str) -> None:
        install_calls.append(label)
        if label == "uv venv creation":
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")

    monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
    monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)

    assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
    assert probe_calls == [(python_path, venv_dir)]
    assert rmtree_calls == [(venv_dir, True)]
    assert install_calls == ["uv venv creation", "locked OVRTX runtime install"]
    assert render_ovrtx._ovrtx_managed_marker_matches_runtime_lock(venv_dir)


@pytest.mark.parametrize("leave_python", [True, False])
def test_stale_managed_runtime_cleanup_failure_stops_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leave_python: bool,
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    stale_package = venv_dir / "lib" / "python3.12" / "site-packages" / "stale"
    stale_package.mkdir(parents=True)
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
        f"runtime_lock_sha256={'0' * 64}\n",
        encoding="utf-8",
    )
    install_calls: list[str] = []

    def partial_rmtree(path: Path, ignore_errors: bool = False) -> None:
        (path / render_ovrtx._OVRTX_MANAGED_MARKER).unlink(missing_ok=True)
        (path / render_ovrtx._OVRTX_PROVISIONING_MARKER).unlink(missing_ok=True)
        if not leave_python:
            python_path.unlink(missing_ok=True)

    monkeypatch.setattr(render_ovrtx.shutil, "rmtree", partial_rmtree)
    monkeypatch.setattr(
        render_ovrtx,
        "_run_checked",
        lambda cmd, label: install_calls.append(label),
    )

    with pytest.raises(RuntimeError, match="could not be completely removed"):
        render_ovrtx._get_ovrtx_python(venv_dir=venv_dir)
    cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
    assert cache_key in render_ovrtx._verified_managed_ovrtx_python_cache
    assert (venv_dir / render_ovrtx._OVRTX_PROVISIONING_MARKER).exists()
    with pytest.raises(RuntimeError, match="could not be completely removed"):
        render_ovrtx._get_ovrtx_python(venv_dir=venv_dir)
    assert stale_package.exists()
    assert install_calls == []
    assert not render_ovrtx._ovrtx_managed_marker_matches_runtime_lock(venv_dir)


def test_auto_provision_disabled_preserves_incomplete_managed_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())
    monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")

    venv_dir = tmp_path / "ovrtx_venv"
    stale_package = venv_dir / "lib" / "python3.12" / "site-packages" / "stale"
    stale_package.mkdir(parents=True)
    marker_path = venv_dir / render_ovrtx._OVRTX_PROVISIONING_MARKER
    marker_path.write_text("interrupted")
    rmtree_calls: list[Path] = []
    monkeypatch.setattr(
        render_ovrtx.shutil,
        "rmtree",
        lambda path, ignore_errors=False: rmtree_calls.append(Path(path)),
    )

    with pytest.raises(RuntimeError, match="AUTO_PROVISION is disabled"):
        render_ovrtx._get_ovrtx_python(venv_dir)
    assert rmtree_calls == []
    assert marker_path.read_text() == "interrupted"
    assert stale_package.exists()


@pytest.mark.parametrize("change_after_marker", [False, True])
def test_runtime_lock_change_during_provisioning_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_after_marker: bool,
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
    monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
    monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(render_ovrtx, "FileLock", _NoopFileLock)

    runtime_lock = tmp_path / "pylock.ovrtx-runtime.toml"
    runtime_lock.write_bytes(b"lock generation A")
    monkeypatch.setattr(render_ovrtx, "_OVRTX_RUNTIME_LOCK_FILE", runtime_lock)

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    lock_snapshots: list[Path] = []

    def fake_run_checked(cmd: list[str], label: str) -> None:
        if label == "uv venv creation":
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("")
        if label == "locked OVRTX runtime install":
            snapshot_path = Path(cmd[cmd.index("-r") + 1])
            lock_snapshots.append(snapshot_path)
            assert snapshot_path != runtime_lock
            assert snapshot_path.name.startswith("pylock.")
            assert snapshot_path.read_bytes() == b"lock generation A"
            if not change_after_marker:
                runtime_lock.write_bytes(b"lock generation B")

    original_write_marker = render_ovrtx._write_ovrtx_managed_marker

    def fake_write_marker(venv_dir_arg: Path, runtime_lock_digest: str) -> None:
        original_write_marker(venv_dir_arg, runtime_lock_digest)
        if change_after_marker:
            runtime_lock.write_bytes(b"lock generation B")

    monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
    monkeypatch.setattr(
        render_ovrtx,
        "_probe_ovrtx_version",
        lambda python_path_arg, venv_dir_arg: render_ovrtx._OVRTX_VERSION,
    )
    monkeypatch.setattr(
        render_ovrtx,
        "_write_ovrtx_managed_marker",
        fake_write_marker,
    )

    error = (
        "lock changed while marking"
        if change_after_marker
        else "lock changed during provisioning"
    )
    with pytest.raises(RuntimeError, match=error):
        render_ovrtx._get_ovrtx_python(venv_dir=venv_dir)
    assert not venv_dir.exists()
    assert len(lock_snapshots) == 1
    assert not lock_snapshots[0].exists()


def test_managed_marker_with_current_version_uses_fast_path(tmp_path: Path) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
        f"runtime_lock_sha256={render_ovrtx._ovrtx_runtime_lock_digest()}\n",
        encoding="utf-8",
    )

    assert render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)


def test_verified_managed_cache_does_not_bypass_changed_runtime_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    stale_digest = "0" * 64
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
        f"runtime_lock_sha256={stale_digest}\n",
        encoding="utf-8",
    )
    cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
    monkeypatch.setattr(
        render_ovrtx,
        "_verified_ovrtx_python_cache",
        {(cache_key, render_ovrtx._ovrtx_runtime_lock_digest())},
    )

    assert not render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)


def test_verified_managed_cache_rejects_lost_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    marker_path = venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER
    marker_path.write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
        f"runtime_lock_sha256={render_ovrtx._ovrtx_runtime_lock_digest()}\n",
        encoding="utf-8",
    )
    cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
    verified_key = (cache_key, render_ovrtx._ovrtx_runtime_lock_digest())
    monkeypatch.setattr(
        render_ovrtx,
        "_verified_ovrtx_python_cache",
        {verified_key},
    )
    monkeypatch.setattr(
        render_ovrtx,
        "_verified_managed_ovrtx_python_cache",
        set(),
    )
    monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
    monkeypatch.setattr(
        render_ovrtx,
        "_ovrtx_python_cache",
        {cache_key: str(python_path)},
    )
    monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")

    assert render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)
    assert cache_key in render_ovrtx._verified_managed_ovrtx_python_cache
    marker_path.unlink()
    assert not render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)
    with pytest.raises(RuntimeError, match="does not match the current runtime lock"):
        render_ovrtx._get_ovrtx_python(venv_dir)
    assert cache_key in render_ovrtx._verified_managed_ovrtx_python_cache


def test_verified_managed_cache_rejects_provisioning_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
        f"runtime_lock_sha256={render_ovrtx._ovrtx_runtime_lock_digest()}\n",
        encoding="utf-8",
    )
    (venv_dir / render_ovrtx._OVRTX_PROVISIONING_MARKER).write_text("")
    cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
    monkeypatch.setattr(
        render_ovrtx,
        "_verified_ovrtx_python_cache",
        {(cache_key, render_ovrtx._ovrtx_runtime_lock_digest())},
    )

    assert not render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)


def test_managed_marker_version_parser_is_line_based(tmp_path: Path) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    marker_path = tmp_path / render_ovrtx._OVRTX_MANAGED_MARKER
    marker_path.write_text(
        "Created by world_understanding.functions.graphics.render_ovrtx\n"
        f"note=ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
        f"runtime_lock_sha256={render_ovrtx._ovrtx_runtime_lock_digest()}\n",
        encoding="utf-8",
    )

    assert (
        render_ovrtx._read_ovrtx_managed_marker_version(marker_path)
        == render_ovrtx._OVRTX_VERSION
    )


def test_unreadable_managed_marker_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    marker_path = venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER
    marker_path.write_text(
        f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n",
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == marker_path:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert not render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)


def test_marker_version_mismatch_skips_bundled_library_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("")
    (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
        "ovrtx_version=0.0.0\n"
        f"runtime_lock_sha256={render_ovrtx._ovrtx_runtime_lock_digest()}\n",
        encoding="utf-8",
    )

    def fail_if_scanned(unused_venv_dir: Path) -> list[Path]:
        raise AssertionError("bundled libraries should not be scanned")

    monkeypatch.setattr(
        render_ovrtx,
        "_ovrtx_bundled_python_libraries",
        fail_if_scanned,
    )

    assert not render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)
