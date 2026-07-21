# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for local Scene Optimizer backend.

Unit tests run without the Scene Optimizer package — all subprocess calls
are mocked.
"""

import json
import subprocess
import sys
import sysconfig
from unittest.mock import MagicMock, patch

import pytest

from world_understanding.functions.graphics.scene_optimizer_local import (
    _build_operations_list,
    _python_libdir,
    _resolve_so_python,
    optimize_usd_local,
)
from world_understanding.functions.graphics.so_worker import (
    _merge_split_mappings,
    _natural_sort_key,
    build_correspondence_map,
    track_split_meshes,
)

# ---------------------------------------------------------------------------
# _build_operations_list tests
# ---------------------------------------------------------------------------


class TestBuildOperationsList:
    """Test _build_operations_list() operation ordering and params."""

    def test_defaults(self):
        """Default settings produce deinstance, splitMeshes, deduplicateGeometry."""
        settings = {
            "enable_deinstance": True,
            "enable_split_meshes": True,
            "enable_deduplicate": True,
            "deduplicate": {
                "tolerance": 0.001,
                "consider_deep_transforms": True,
            },
        }
        ops = _build_operations_list(settings)

        assert len(ops) == 3
        assert ops[0][0] == "utilityFunction"
        assert ops[0][1]["function"] == 0  # DEINSTANCE
        assert ops[1][0] == "splitMeshes"
        assert ops[1][1]["splitOn"] == 1
        assert ops[2][0] == "deduplicateGeometry"
        assert ops[2][1]["duplicateMethod"] == 2
        assert ops[2][1]["tolerance"] == 0.001

    def test_disabled(self):
        """All ops disabled — empty list."""
        settings = {
            "enable_deinstance": False,
            "enable_split_meshes": False,
            "enable_deduplicate": False,
        }
        ops = _build_operations_list(settings)

        assert len(ops) == 0

    def test_custom_dedup_params(self):
        """Deduplicate params are passed through correctly."""
        settings = {
            "enable_deinstance": False,
            "enable_split_meshes": False,
            "enable_deduplicate": True,
            "deduplicate": {
                "tolerance": 0.01,
                "consider_deep_transforms": False,
                "fuzzy": True,
                "use_gpu": True,
                "allow_scaling": True,
                "ignore_attributes": ["primvars:st"],
            },
        }
        ops = _build_operations_list(settings)

        assert len(ops) == 1
        dedup_params = ops[0][1]
        assert dedup_params["tolerance"] == 0.01
        assert dedup_params["considerDeepTransforms"] is False
        assert dedup_params["fuzzy"] is True
        assert dedup_params["useGpu"] is True
        assert dedup_params["allowScaling"] is True
        assert dedup_params["ignoreAttributes"] == ["primvars:st"]

    def test_deinstance_only(self):
        """enable_deinstance=True adds utilityFunction(DEINSTANCE) operation."""
        settings = {
            "enable_deinstance": True,
            "enable_split_meshes": False,
            "enable_deduplicate": False,
        }
        ops = _build_operations_list(settings)

        assert len(ops) == 1
        assert ops[0][0] == "utilityFunction"
        assert ops[0][1]["function"] == 0
        assert ops[0][1]["primPaths"] == []

    def test_empty_settings(self):
        """Empty settings dict uses defaults (deinstance + split + dedup enabled)."""
        ops = _build_operations_list({})

        assert len(ops) == 3
        assert ops[0][0] == "utilityFunction"
        assert ops[1][0] == "splitMeshes"
        assert ops[2][0] == "deduplicateGeometry"


# ---------------------------------------------------------------------------
# optimize_usd_local tests
# ---------------------------------------------------------------------------


class TestOptimizeUsdLocalMissingEnv:
    """Test that missing env var or bad layout raises RuntimeError."""

    def test_missing_so_package_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WU_SO_PACKAGE_DIR", raising=False)
        # Pin CWD so the auto-discovery fallback (.build-resources/...) points
        # somewhere that does not exist, forcing the "not found" error.
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError, match="WU_SO_PACKAGE_DIR"):
            optimize_usd_local("/tmp/in.usd", "/tmp/out.usd")

    def test_missing_subdirectory(self, monkeypatch, tmp_path):
        """Package dir without the expected subdirs raises with package path."""
        so_dir = tmp_path / "so_pkg"
        (so_dir / "python").mkdir(parents=True)
        (so_dir / "lib").mkdir()
        (so_dir / "extraLibs").mkdir()
        # Missing `usdpy/` — the new single-dir contract

        monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(so_dir))

        with pytest.raises(
            RuntimeError, match="Scene Optimizer package directory missing"
        ):
            optimize_usd_local("/tmp/in.usd", "/tmp/out.usd")

    def test_auto_discovers_build_resources_when_env_unset(self, monkeypatch, tmp_path):
        """With WU_SO_PACKAGE_DIR unset, resolver picks up the fetch-script default."""
        from world_understanding.functions.graphics.scene_optimizer_local import (
            _resolve_so_package_dir,
        )

        monkeypatch.delenv("WU_SO_PACKAGE_DIR", raising=False)

        so_dir = tmp_path / ".build-resources" / "scene_optimizer_core"
        (so_dir / "python").mkdir(parents=True)
        (so_dir / "lib").mkdir()
        (so_dir / "extraLibs").mkdir()
        (so_dir / "usdpy").mkdir()
        monkeypatch.chdir(tmp_path)

        assert _resolve_so_package_dir() == so_dir


class TestOptimizeUsdLocalSubprocess:
    """Test subprocess invocation with mocked subprocess.run."""

    @pytest.fixture()
    def env_setup(self, monkeypatch, tmp_path):
        """Set up valid SO Core package layout and env vars."""
        so_dir = tmp_path / "so_pkg"
        (so_dir / "python").mkdir(parents=True)
        (so_dir / "lib").mkdir()
        (so_dir / "extraLibs").mkdir()
        (so_dir / "usdpy").mkdir()

        monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(so_dir))
        # Pin WU_SO_PYTHON to the current interpreter so the libdir lookup
        # short-circuits (no extra subprocess.run call) and tests that mock
        # subprocess.run see only the worker invocation. This must NOT use
        # ``delenv`` — on Python 3.13+ runners the default resolver would
        # return ``python3.12`` and trigger an extra libdir probe, breaking
        # mocks that only expect the worker call.
        monkeypatch.setenv("WU_SO_PYTHON", sys.executable)

        return {"so_dir": so_dir}

    def test_subprocess_env_vars(self, env_setup, monkeypatch, tmp_path):
        """Verify LD_LIBRARY_PATH, PYTHONPATH, PXR_PLUGINPATH_NAME are set correctly."""
        so_dir = env_setup["so_dir"]

        # Prepare a manifest that the "subprocess" would write
        manifest = {
            "status": "success",
            "optimization_time": 1.5,
            "operations_executed": [{"name": "merge", "success": True, "time": 0.5}],
            "stage_size_bytes": 1024,
        }

        def mock_run(cmd, **kwargs):
            # Parse params from the command to find manifest_path
            params = json.loads(cmd[-1])
            manifest_path = params["manifest_path"]
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            # Verify environment variables
            env = kwargs.get("env", {})
            ld_path = env.get("LD_LIBRARY_PATH", "")
            py_path = env.get("PYTHONPATH", "")
            plug_path = env.get("PXR_PLUGINPATH_NAME", "")

            assert str(so_dir / "lib") in ld_path
            assert str(so_dir / "extraLibs") in ld_path

            assert str(so_dir / "python") in py_path
            assert str(so_dir / "usdpy") in py_path

            assert str(so_dir / "extraLibs" / "usd") == plug_path

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            result = optimize_usd_local(
                input_path=input_path,
                output_path=output_path,
                optimization_config={"scene_optimizer_settings": {}},
            )

        assert result["status"] == "success"
        assert result["optimization_time"] == 1.5
        assert result["stage_size_bytes"] == 1024
        assert len(result["operations_executed"]) == 1

    def test_subprocess_params_json(self, env_setup, tmp_path):
        """Verify the params JSON passed to the subprocess."""
        captured_params = {}

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            captured_params.update(params)

            manifest_path = params["manifest_path"]
            with open(manifest_path, "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            optimize_usd_local(
                input_path=input_path,
                output_path=output_path,
                optimization_config={
                    "scene_optimizer_settings": {
                        "enable_split_meshes": True,
                        "enable_deduplicate": False,
                        "enable_deinstance": False,
                    }
                },
            )

        assert captured_params["input_usd_path"] == str(input_path)
        assert captured_params["output_usd_path"] == str(output_path)
        # splitMeshes enabled, dedup disabled, no merge
        assert len(captured_params["operations"]) == 1
        assert captured_params["operations"][0][0] == "splitMeshes"

    def test_subprocess_failure(self, env_setup, tmp_path):
        """Verify RuntimeError is raised on subprocess failure."""

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = "some output"
            result.stderr = "segfault"
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            with pytest.raises(RuntimeError, match="Scene Optimizer subprocess failed"):
                optimize_usd_local(
                    input_path=input_path,
                    output_path=output_path,
                )

    def test_pythonpath_stripped(self, env_setup, monkeypatch, tmp_path):
        """Verify parent PYTHONPATH is replaced, not appended to."""
        monkeypatch.setenv("PYTHONPATH", "/some/parent/path")

        def mock_run(cmd, **kwargs):
            env = kwargs.get("env", {})
            py_path = env.get("PYTHONPATH", "")
            # Parent PYTHONPATH must NOT leak into the subprocess
            assert "/some/parent/path" not in py_path

            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            optimize_usd_local(input_path, output_path)

    def test_ld_library_path_does_not_inherit_parent(
        self, env_setup, monkeypatch, tmp_path
    ):
        """The worker's LD_LIBRARY_PATH does NOT inherit the parent's value.

        Inheriting would let unrelated host paths (Kit, rendering stack,
        system libs) satisfy missing transitive deps for the SO bundle's
        compiled extensions and silently mix ABIs. Only the SO bundle's
        ``lib/`` + ``extraLibs/`` plus the SO Python's ``LIBDIR`` should
        appear.
        """
        parent_ld = "/parent/site-libs"
        monkeypatch.setenv("LD_LIBRARY_PATH", parent_ld)

        def mock_run(cmd, **kwargs):
            env = kwargs.get("env", {})
            ld_path = env.get("LD_LIBRARY_PATH", "")
            so_lib = str(env_setup["so_dir"] / "lib")
            so_extra = str(env_setup["so_dir"] / "extraLibs")
            assert so_lib in ld_path
            assert so_extra in ld_path
            assert parent_ld not in ld_path

            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            optimize_usd_local(input_path, tmp_path / "output.usd")

    def test_subprocess_timeout_raises_runtime_error(self, env_setup, tmp_path):
        """TimeoutExpired from the worker is converted to a RuntimeError."""
        input_path = tmp_path / "input.usd"
        input_path.touch()

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["worker"], timeout=3),
        ):
            with pytest.raises(RuntimeError, match="timed out after 3s"):
                optimize_usd_local(
                    input_path,
                    tmp_path / "output.usd",
                    optimization_config={"timeout": 3},
                )

    def test_worker_sys_path_excludes_parent_site_packages(self, tmp_path):
        """Real subprocess: worker ``sys.path`` excludes parent venv's site-packages.

        End-to-end check that ``-S`` + the curated ``PYTHONPATH`` actually
        keep the parent venv's ``pxr`` provider (and any other package) out
        of the worker. Spawns a real ``sys.executable -S -c ...`` with the
        same env construction ``optimize_usd_local`` uses, then inspects
        the printed ``sys.path``.
        """
        from world_understanding.functions.graphics.scene_optimizer_local import (
            _subprocess_env,
        )

        # Fake SO package layout — paths just need to exist.
        so_dir = tmp_path / "so_pkg"
        for sub in ("python", "lib", "extraLibs", "usdpy"):
            (so_dir / sub).mkdir(parents=True)

        env = _subprocess_env(so_dir, sys.executable)

        proc = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys, json; print(json.dumps(sys.path))",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        worker_sys_path = json.loads(proc.stdout.strip())

        # SO bundle's PYTHONPATH entries are present.
        assert str(so_dir / "python") in worker_sys_path
        assert str(so_dir / "usdpy") in worker_sys_path

        # Parent's site-packages dirs are NOT present. We probe sysconfig
        # for the parent's "purelib" and "platlib" — these are the dirs
        # ``site.py`` would have added were ``-S`` not set.
        parent_purelib = sysconfig.get_path("purelib")
        parent_platlib = sysconfig.get_path("platlib")
        assert parent_purelib not in worker_sys_path, (
            f"parent purelib leaked into worker sys.path: {parent_purelib}"
        )
        assert parent_platlib not in worker_sys_path, (
            f"parent platlib leaked into worker sys.path: {parent_platlib}"
        )

    def test_worker_argv_includes_S_flag(self, env_setup, tmp_path):
        """Worker is launched with ``-S`` to keep parent site-packages off sys.path.

        Without ``-S``, ``site.py`` would auto-add the project venv's
        ``site-packages`` to ``sys.path``. Mixing the app's ``pxr`` provider
        with the SO bundle's stock USD 25.11 bindings triggers the exact ABI
        crash class this subprocess boundary exists to prevent.
        """
        captured: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            captured.append(list(cmd))
            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            optimize_usd_local(input_path, tmp_path / "output.usd")

        argv = captured[0]
        assert argv[1] == "-S"
        assert argv[2].endswith("_so_worker.py")
        assert json.loads(argv[3])  # parses as JSON

    def test_ld_library_path_includes_python_libdir(self, env_setup, tmp_path):
        """The SO Python's LIBDIR is injected into LD_LIBRARY_PATH.

        The SO Core bundle's compiled extensions ``DT_NEEDED libpython3.12.so``;
        with newer toolchains the python binary tags its private ``lib/`` with
        ``DT_RUNPATH`` (not searched for transitive deps), so the loader falls
        through to ``LD_LIBRARY_PATH`` — which is where this entry rescues it.
        """
        import sysconfig

        expected_libdir = sysconfig.get_config_var("LIBDIR")
        if not expected_libdir:
            pytest.skip("interpreter has no LIBDIR config var")

        def mock_run(cmd, **kwargs):
            env = kwargs.get("env", {})
            assert expected_libdir in env.get("LD_LIBRARY_PATH", "")

            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            optimize_usd_local(input_path, tmp_path / "output.usd")

    def test_default_so_python_is_sys_executable_on_312_host(
        self, env_setup, tmp_path, monkeypatch
    ):
        """On a 3.12 host with WU_SO_PYTHON unset, the subprocess runs sys.executable.

        Regression: the old default ``python3.12`` required that name on PATH,
        which fails for callers that invoke ``.venv/bin/python script.py``
        without activating.
        """
        # Override the fixture's pin and force the 3.12 resolver branch so
        # this test is meaningful on any supported runner (3.12 or 3.13+).
        monkeypatch.delenv("WU_SO_PYTHON", raising=False)
        monkeypatch.setattr(sys, "version_info", (3, 12, 5, "final", 0))

        captured_cmd = {}

        def mock_run(cmd, **kwargs):
            captured_cmd["argv"] = cmd

            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            optimize_usd_local(input_path, tmp_path / "output.usd")

        assert captured_cmd["argv"][0] == sys.executable

    def test_default_so_python_is_python312_on_non_312_host(
        self, env_setup, tmp_path, monkeypatch
    ):
        """On a 3.13+ host with WU_SO_PYTHON unset, the worker uses ``python3.12``.

        The SO Core bundle's compiled extensions are cpython-312-only, so
        defaulting to ``sys.executable`` (3.13 in this case) would load 3.12
        ABI bindings into a 3.13 interpreter — the resolver intentionally
        falls back to a PATH lookup of ``python3.12`` instead. The libdir
        probe targeting ``python3.12`` is mocked separately from the worker
        invocation.
        """
        monkeypatch.delenv("WU_SO_PYTHON", raising=False)
        monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))

        worker_argv: list[str] = []

        def mock_run(cmd, **kwargs):
            # The libdir probe runs first — distinguish by the ``sysconfig``
            # snippet at the tail of argv (probe argv is now ``[py, -S, -c,
            # snippet]`` after the round-4 hardening).
            if "sysconfig" in cmd[-1]:
                result = MagicMock()
                result.returncode = 0
                result.stdout = "/usr/lib\n"
                result.stderr = ""
                return result
            # Worker invocation
            worker_argv.extend(cmd)
            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            optimize_usd_local(input_path, tmp_path / "output.usd")

        assert worker_argv[0] == "python3.12"

    def test_error_field_passthrough(self, env_setup, tmp_path):
        """When the worker writes ``error`` to the manifest, it appears in the result.

        Regression: ``optimize_usd_local`` previously stripped this field, so
        the orchestrator surfaced its generic "Unknown optimization error"
        fallback instead of the worker's actual traceback (e.g.
        ``ImportError: libpython3.12.so.1.0``).
        """
        manifest = {
            "status": "error",
            "optimization_time": 0.05,
            "operations_executed": [],
            "stage_size_bytes": 0,
            "error": "ImportError: libpython3.12.so.1.0: cannot open shared object file",
        }

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump(manifest, f)

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            result = optimize_usd_local(input_path, tmp_path / "output.usd")

        assert result["status"] == "error"
        assert result["error"] == manifest["error"]

    def test_python_libdir_short_circuits_for_sys_executable(self):
        """``_python_libdir(sys.executable)`` returns the value without subprocessing.

        We patch ``subprocess.run`` to raise; if it is called the test fails,
        proving the parent's ``sysconfig`` is consulted directly.
        """
        import sysconfig

        with patch("subprocess.run", side_effect=AssertionError("must not subprocess")):
            assert _python_libdir(sys.executable) == sysconfig.get_config_var("LIBDIR")

    def test_python_libdir_subprocess_query_path(self):
        """When ``so_python != sys.executable``, the helper queries that interpreter.

        The probe argv must include ``-S`` so the queried interpreter doesn't
        run ``site.py``/``sitecustomize.py``/``.pth`` files before our snippet
        — those could pollute stdout or hang.
        """
        captured = {}

        def mock_run(cmd, **kwargs):
            captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = "/opt/python3.12/lib\n"
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            libdir = _python_libdir("/opt/python3.12/bin/python3.12")

        assert libdir == "/opt/python3.12/lib"
        assert captured["cmd"][0] == "/opt/python3.12/bin/python3.12"
        assert captured["cmd"][1] == "-S"
        assert captured["cmd"][2] == "-c"
        assert "sysconfig" in captured["cmd"][3]

    def test_python_libdir_strict_parse_takes_last_nonempty_line(self):
        """``_python_libdir`` returns the last non-empty stdout line.

        Defensive against a queried interpreter emitting extra lines (e.g.
        from a ``PYTHONPATH``-injected module's import-time print). The
        snippet's ``print(LIBDIR)`` is always the last write to stdout.
        """

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "noise from sitecustomize\n\n/opt/python3.12/lib\n"
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            libdir = _python_libdir("/opt/python3.12/bin/python3.12")

        assert libdir == "/opt/python3.12/lib"

    def test_python_libdir_rejects_non_absolute_output(self):
        """A non-absolute LIBDIR is rejected (would be unsafe in LD_LIBRARY_PATH)."""

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "relative/path\n"
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            assert _python_libdir("/opt/python3.12/bin/python3.12") is None

    def test_resolve_so_python_respects_explicit_override(self, monkeypatch):
        """``WU_SO_PYTHON`` set in the environment is returned verbatim.

        Uses ``sys.executable`` (a real existing absolute path) so the
        existence-check on absolute overrides is satisfied.
        """
        monkeypatch.setenv("WU_SO_PYTHON", sys.executable)
        assert _resolve_so_python() == sys.executable

    def test_resolve_so_python_passes_through_relative_override(self, monkeypatch):
        """Relative names are passed through unchanged for ``PATH`` lookup.

        Preserves the historical ``WU_SO_PYTHON=python3.12`` ergonomic where
        the user expects ``subprocess.run`` to do its normal ``PATH`` search.
        """
        monkeypatch.setenv("WU_SO_PYTHON", "python3.12")
        assert _resolve_so_python() == "python3.12"

    def test_resolve_so_python_rejects_nonexistent_absolute_override(
        self, monkeypatch, tmp_path
    ):
        """Absolute ``WU_SO_PYTHON`` pointing at a missing file fails fast.

        Without this check, the bad path would propagate to ``subprocess.run``
        and surface as a less-actionable ``FileNotFoundError`` deep in the
        worker launch path.
        """
        bogus = str(tmp_path / "nope" / "python3.12")
        monkeypatch.setenv("WU_SO_PYTHON", bogus)
        with pytest.raises(ValueError, match="absolute path that does not exist"):
            _resolve_so_python()

    def test_resolve_so_python_uses_sys_executable_on_312_host(self, monkeypatch):
        """On a Python 3.12 host with no override, returns ``sys.executable``."""
        monkeypatch.delenv("WU_SO_PYTHON", raising=False)
        monkeypatch.setattr(sys, "version_info", (3, 12, 5, "final", 0))
        assert _resolve_so_python() == sys.executable

    def test_resolve_so_python_falls_back_to_python312_on_non_312_host(
        self, monkeypatch
    ):
        """On Python 3.13+ hosts, default to PATH lookup of ``python3.12``.

        Codex flagged that an unconditional ``sys.executable`` default would
        have launched the SO worker under 3.13 with cpython-312-only ABI
        bindings on the PYTHONPATH — broken before doing any work.
        """
        monkeypatch.delenv("WU_SO_PYTHON", raising=False)
        monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))
        assert _resolve_so_python() == "python3.12"

    def test_python_libdir_returns_none_on_subprocess_failure(self):
        """Non-zero exit / empty stdout / FileNotFoundError all yield ``None``."""

        # Non-zero exit
        def mock_fail(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "boom"
            return result

        with patch("subprocess.run", side_effect=mock_fail):
            assert _python_libdir("/no/such/python") is None

        # Empty stdout (some configs return None for LIBDIR → empty string)
        def mock_empty(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "\n"
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_empty):
            assert _python_libdir("/no/such/python") is None

        # Binary not on PATH
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _python_libdir("python3.12") is None

        # Subprocess timeout
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
            assert _python_libdir("python3.12") is None

    def test_correspondence_map_passthrough(self, env_setup, tmp_path):
        """Verify correspondence_map from manifest is returned in result."""
        correspondence_map = {
            "summary": {
                "operations_run": {"split": True, "deduplicate": True},
                "total_original_prims": 3,
            },
            "split_mapping": {
                "/World/Mesh1": ["/World/Mesh1_part", "/World/Mesh1_part_1"],
            },
            "deduplication_mapping": {
                "instance_to_prototype": {
                    "/World/Mesh2/Geometry": "/World/Mesh1/Geometry",
                },
            },
            "full_mapping": {
                "original_to_prototype": {
                    "/World/Mesh1": [
                        "/World/Mesh1_part/Geometry",
                        "/World/Mesh1_part_1/Geometry",
                    ],
                    "/World/Mesh2": ["/World/Mesh1/Geometry"],
                    "/World/Mesh3": ["/World/Mesh3"],
                },
            },
        }
        manifest = {
            "status": "success",
            "optimization_time": 0.5,
            "operations_executed": [],
            "stage_size_bytes": 100,
            "correspondence_map": correspondence_map,
        }

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump(manifest, f)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            result = optimize_usd_local(input_path, output_path)

        assert result["correspondence_map"] == correspondence_map
        assert result["correspondence_map"]["split_mapping"]["/World/Mesh1"] == [
            "/World/Mesh1_part",
            "/World/Mesh1_part_1",
        ]
        full = result["correspondence_map"]["full_mapping"]["original_to_prototype"]
        assert "/World/Mesh1" in full
        assert "/World/Mesh2" in full
        assert "/World/Mesh3" in full

    def test_correspondence_map_defaults_to_empty(self, env_setup, tmp_path):
        """When manifest has no correspondence_map, result defaults to {}."""
        manifest = {
            "status": "success",
            "optimization_time": 0.1,
            "operations_executed": [],
        }

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump(manifest, f)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()

            result = optimize_usd_local(input_path, tmp_path / "output.usd")

        assert result["correspondence_map"] == {}


# ---------------------------------------------------------------------------
# Worker script helper function tests
# ---------------------------------------------------------------------------


class TestNaturalSortKey:
    """Test _natural_sort_key for correct numerical ordering."""

    def test_basic(self):
        paths = ["/World/Mesh_part_10", "/World/Mesh_part_2", "/World/Mesh_part_1"]
        result = sorted(paths, key=_natural_sort_key)
        assert result == [
            "/World/Mesh_part_1",
            "/World/Mesh_part_2",
            "/World/Mesh_part_10",
        ]

    def test_no_numbers(self):
        paths = ["/World/C", "/World/A", "/World/B"]
        result = sorted(paths, key=_natural_sort_key)
        assert result == ["/World/A", "/World/B", "/World/C"]

    def test_mixed(self):
        paths = ["/World/Mesh_part", "/World/Mesh_part_1", "/World/Mesh_part_0"]
        result = sorted(paths, key=_natural_sort_key)
        assert result[0] == "/World/Mesh_part"


class TestWorkerTrackSplitMeshes:
    """Test track_split_meshes from the worker module."""

    def test_no_changes(self):
        """No meshes removed or added — empty split mapping."""
        assert (
            track_split_meshes(["/World/A", "/World/B"], ["/World/A", "/World/B"]) == {}
        )

    def test_single_split(self):
        """One mesh split into two parts."""
        result = track_split_meshes(
            ["/World/A", "/World/B"],
            ["/World/A", "/World/B_part", "/World/B_part_1"],
        )
        assert result == {"/World/B": ["/World/B_part", "/World/B_part_1"]}

    def test_multiple_splits(self):
        """Multiple meshes split."""
        result = track_split_meshes(
            ["/World/A", "/World/B", "/World/C"],
            [
                "/World/A_part",
                "/World/A_part_1",
                "/World/B",
                "/World/C_part",
                "/World/C_part_1",
                "/World/C_part_2",
            ],
        )
        assert result["/World/A"] == ["/World/A_part", "/World/A_part_1"]
        assert len(result["/World/C"]) == 3
        assert "/World/B" not in result

    def test_nested_paths(self):
        """Split detection works with nested prim paths."""
        result = track_split_meshes(
            ["/World/Group/Mesh1"],
            ["/World/Group/Mesh1_part", "/World/Group/Mesh1_part_1"],
        )
        assert result == {
            "/World/Group/Mesh1": [
                "/World/Group/Mesh1_part",
                "/World/Group/Mesh1_part_1",
            ]
        }

    def test_repeated_split_chains_existing_mapping(self):
        """A second splitMeshes pass preserves the original mesh as the source."""
        first = {
            "/World/Mesh": ["/World/Mesh_part", "/World/Mesh_part_1"],
        }
        second = {
            "/World/Mesh_part": [
                "/World/Mesh_part_part",
                "/World/Mesh_part_part_1",
            ],
        }

        result = _merge_split_mappings(first, second)

        assert result == {
            "/World/Mesh": [
                "/World/Mesh_part_part",
                "/World/Mesh_part_part_1",
                "/World/Mesh_part_1",
            ],
        }


class TestWorkerBuildCorrespondenceMap:
    """Test build_correspondence_map from the worker module."""

    def test_identity_no_ops(self):
        """No split or dedup — all meshes map to themselves."""
        result = build_correspondence_map(
            ["/World/A", "/World/B"], {}, {}, False, False
        )

        o2p = result["full_mapping"]["original_to_prototype"]
        assert o2p == {"/World/A": ["/World/A"], "/World/B": ["/World/B"]}
        assert result["split_mapping"] == {}
        assert result["deduplication_mapping"]["instance_to_prototype"] == {}

    def test_split_only(self):
        """Split without dedup — parts map directly."""
        split = {"/World/A": ["/World/A_part", "/World/A_part_1"]}
        result = build_correspondence_map(
            ["/World/A", "/World/B"], split, {}, True, False
        )

        o2p = result["full_mapping"]["original_to_prototype"]
        assert o2p["/World/A"] == ["/World/A_part", "/World/A_part_1"]
        assert o2p["/World/B"] == ["/World/B"]

    def test_dedup_only(self):
        """Dedup without split — instances map to prototypes."""
        i2p = {"/World/B": "/World/A", "/World/C": "/World/A"}
        result = build_correspondence_map(
            ["/World/A", "/World/B", "/World/C"], {}, i2p, False, True
        )

        o2p = result["full_mapping"]["original_to_prototype"]
        assert o2p["/World/B"] == ["/World/A"]
        assert o2p["/World/C"] == ["/World/A"]

    def test_split_then_dedup(self):
        """Split + dedup — chained mapping."""
        split = {
            "/World/A": ["/World/A_part", "/World/A_part_1"],
            "/World/B": ["/World/B_part", "/World/B_part_1"],
        }
        i2p = {"/World/B_part": "/World/A_part", "/World/B_part_1": "/World/A_part_1"}
        result = build_correspondence_map(
            ["/World/A", "/World/B"], split, i2p, True, True
        )

        o2p = result["full_mapping"]["original_to_prototype"]
        assert o2p["/World/A"] == ["/World/A_part", "/World/A_part_1"]
        assert o2p["/World/B"] == ["/World/A_part", "/World/A_part_1"]

    def test_summary_structure(self):
        """Summary has expected keys and values."""
        split = {"/World/A": ["/World/A_part", "/World/A_part_1"]}
        i2p = {"/World/B": "/World/C"}
        result = build_correspondence_map(
            ["/World/A", "/World/B", "/World/C"], split, i2p, True, True
        )

        summary = result["summary"]
        assert summary["total_original_prims"] == 3
        assert summary["meshes_split"] == 1
        assert summary["instances_tracked"] == 1
