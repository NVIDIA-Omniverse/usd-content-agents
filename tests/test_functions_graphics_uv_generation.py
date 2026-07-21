# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for UV generation using Scene Optimizer operations.

Unit tests mock the subprocess; integration tests require an unpacked
Scene Optimizer Core package (see scripts/fetch_build_resources.sh and
``WU_SO_PACKAGE_DIR``). NVCF tests mock the async HTTP layer; A/B tests
require both a local SO package and NVCF access.
"""

import base64
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from world_understanding.functions.graphics.uv_generation import (
    ProjectionType,
    _build_uv_generation_settings,
    generate_atlas_uvs,
    generate_projection_uvs,
)

uv = importlib.import_module("world_understanding.functions.graphics.uv_generation")

# ---------------------------------------------------------------------------
# ProjectionType enum tests
# ---------------------------------------------------------------------------


class TestProjectionType:
    """Verify ProjectionType enum values match C++ enum."""

    def test_values(self):
        assert ProjectionType.PLANAR == 0
        assert ProjectionType.SPHERICAL == 1
        assert ProjectionType.CYLINDRICAL == 2
        assert ProjectionType.TRIPLANAR == 3
        assert ProjectionType.CUBE == 4


# ---------------------------------------------------------------------------
# Unit tests (mocked subprocess)
# ---------------------------------------------------------------------------


class TestGenerateProjectionUvsMocked:
    """Test generate_projection_uvs with mocked subprocess."""

    @pytest.fixture()
    def env_setup(self, monkeypatch, tmp_path):
        """Set up valid SO directory structure (single-dir public package layout)."""
        so_dir = tmp_path / "so_pkg"
        (so_dir / "python").mkdir(parents=True)
        (so_dir / "lib").mkdir()
        (so_dir / "extraLibs").mkdir()
        (so_dir / "usdpy").mkdir()

        monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(so_dir))
        # Pin WU_SO_PYTHON to the current interpreter so the libdir helper
        # short-circuits and tests that mock subprocess.run see only the
        # worker invocation. ``delenv`` would be version-fragile: on a 3.13
        # runner the resolver would return ``python3.12`` and trigger an
        # extra libdir-probe subprocess.run that the mocks do not handle.
        monkeypatch.setenv("WU_SO_PYTHON", sys.executable)

        return {"so_dir": so_dir}

    def test_default_params(self, env_setup, tmp_path):
        """Default call passes cube projection with correct camelCase params."""
        captured = {}

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            captured.update(params)
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateProjectionUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 1024,
                        "mesh_count": 5,
                        "meshes_with_uvs": 5,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            result = generate_projection_uvs(input_path, tmp_path / "out.usd")

        assert result["status"] == "success"
        assert result["meshes_with_uvs"] == 5
        assert captured["operation"] == "generateProjectionUVs"
        assert captured["op_params"]["projectionType"] == 4  # CUBE
        assert captured["op_params"]["useWorldSpaceScales"] is True
        assert captured["op_params"]["scaleFactor"] == 0.01
        assert captured["op_params"]["overwriteExisting"] is True

    def test_custom_projection(self, env_setup, tmp_path):
        """Custom projection type and scale."""
        captured = {}

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            captured.update(params)
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateProjectionUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 1024,
                        "mesh_count": 3,
                        "meshes_with_uvs": 3,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            generate_projection_uvs(
                input_path,
                tmp_path / "out.usd",
                projection_type=ProjectionType.PLANAR,
                scale_factor=0.05,
                overwrite_existing=False,
            )

        assert captured["op_params"]["projectionType"] == 0
        assert captured["op_params"]["scaleFactor"] == 0.05
        assert captured["op_params"]["overwriteExisting"] is False

    def test_preprojection_xform(self, env_setup, tmp_path):
        """preprojectionXform is passed when provided."""
        captured = {}
        xform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            captured.update(params)
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateProjectionUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 0,
                        "mesh_count": 0,
                        "meshes_with_uvs": 0,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            generate_projection_uvs(
                input_path, tmp_path / "out.usd", preprojection_xform=xform
            )

        assert captured["op_params"]["preprojectionXform"] == xform

    def test_invalid_xform_length(self, env_setup, tmp_path):
        """preprojectionXform with wrong length raises ValueError."""
        input_path = tmp_path / "input.usd"
        input_path.touch()
        with pytest.raises(ValueError, match="exactly 16 floats"):
            generate_projection_uvs(
                input_path, tmp_path / "out.usd", preprojection_xform=[1, 0, 0]
            )

    def test_paths_filter(self, env_setup, tmp_path):
        """Specific prim paths are passed through."""
        captured = {}

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            captured.update(params)
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateProjectionUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 0,
                        "mesh_count": 1,
                        "meshes_with_uvs": 1,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            generate_projection_uvs(
                input_path, tmp_path / "out.usd", paths=["/World/Mesh1"]
            )

        assert captured["op_params"]["paths"] == ["/World/Mesh1"]

    def test_subprocess_failure(self, env_setup, tmp_path):
        """RuntimeError on subprocess failure."""

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = "output"
            result.stderr = "segfault"
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            with pytest.raises(RuntimeError, match="UV generation subprocess failed"):
                generate_projection_uvs(input_path, tmp_path / "out.usd")

    def test_subprocess_timeout(self, env_setup, tmp_path):
        """TimeoutExpired is normalized to RuntimeError."""
        input_path = tmp_path / "input.usd"
        input_path.touch()

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(["so-python"], timeout=3),
        ):
            with pytest.raises(RuntimeError, match="timed out after 3s"):
                generate_projection_uvs(input_path, tmp_path / "out.usd", timeout=3)

    def test_subprocess_env_includes_python_libdir(self, env_setup, tmp_path):
        """The UV worker subprocess gets the SO Python's LIBDIR injected.

        Mirrors ``test_ld_library_path_includes_python_libdir`` from the
        scene-optimizer test file: confirms ``_run_uv_worker`` calls
        ``_python_libdir(so_python)`` rather than the parent's
        ``sysconfig.LIBDIR`` (which would be wrong if ``WU_SO_PYTHON``
        pointed at a different interpreter).
        """
        import sysconfig

        expected_libdir = sysconfig.get_config_var("LIBDIR")
        if not expected_libdir:
            pytest.skip("interpreter has no LIBDIR config var")

        captured_env = {}

        def mock_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateProjectionUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 0,
                        "mesh_count": 0,
                        "meshes_with_uvs": 0,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            generate_projection_uvs(input_path, tmp_path / "out.usd")

        lib_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        assert expected_libdir in captured_env.get(lib_var, "")

    def test_uv_worker_argv_includes_S_flag(self, env_setup, tmp_path):
        """UV worker is launched with ``-S`` (parity with the SO worker).

        Same isolation contract as ``scene_optimizer_local._subprocess_env``:
        ``-S`` keeps the parent venv's ``pxr`` provider off the worker's
        ``sys.path`` even when ``WU_SO_PYTHON`` resolves to the project venv.
        """
        captured: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            captured.append(list(cmd))
            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateProjectionUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 0,
                        "mesh_count": 0,
                        "meshes_with_uvs": 0,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            generate_projection_uvs(input_path, tmp_path / "out.usd")

        argv = captured[0]
        assert argv[1] == "-S"
        assert argv[2].endswith("_so_uv_worker.py")
        assert json.loads(argv[3])

    def test_uv_lib_path_does_not_inherit_parent(
        self, env_setup, monkeypatch, tmp_path
    ):
        """UV worker's LD_LIBRARY_PATH does NOT inherit the parent's value.

        Parity with ``test_ld_library_path_does_not_inherit_parent`` on the
        scene-optimizer side. Only the SO bundle's ``lib``/``extraLibs`` plus
        the SO Python's ``LIBDIR`` should appear; arbitrary host paths must
        not satisfy missing transitive deps.
        """
        parent_ld = "/parent/site-libs"
        lib_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        monkeypatch.setenv(lib_var, parent_ld)

        captured_env: dict[str, str] = {}

        def mock_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            params = json.loads(cmd[-1])
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateProjectionUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 0,
                        "mesh_count": 0,
                        "meshes_with_uvs": 0,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            generate_projection_uvs(input_path, tmp_path / "out.usd")

        ld_path = captured_env.get(lib_var, "")
        assert parent_ld not in ld_path

    def test_missing_so_package_falls_back_to_nvcf(self, monkeypatch, tmp_path):
        """Missing SO package triggers NVCF fallback (not a hard error)."""
        monkeypatch.delenv("WU_SO_PACKAGE_DIR", raising=False)
        # Pin CWD so the auto-discovery fallback (.build-resources/...) lands
        # in tmp_path where the SO bundle is absent — avoids picking up a
        # real bundle in the dev's worktree CWD.
        monkeypatch.chdir(tmp_path)

        fake_usd = b"fallback-usd"
        fake_b64 = base64.b64encode(fake_usd).decode()

        async def mock_execute(*args, **kwargs):
            return {
                "success": True,
                "operations_executed": ["generateProjectionUVs"],
                "generated_stage_base64": fake_b64,
            }

        input_path = tmp_path / "input.usd"
        input_path.write_bytes(b"input")
        output_path = tmp_path / "out.usdc"

        with (
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".should_use_data_uri",
                return_value=True,
            ),
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".create_data_uri_from_file",
                return_value="data:application/octet-stream;base64,AAAA",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_nvcf_api_key",
                return_value="test-key",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_base_url",
                return_value="https://api.nvcf.nvidia.com/v2/func/123",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.execute_nvcf_request_async",
                side_effect=mock_execute,
            ),
        ):
            result = generate_projection_uvs(input_path, output_path)

        assert result["status"] == "success"
        assert output_path.read_bytes() == fake_usd


class TestGenerateAtlasUvsMocked:
    """Test generate_atlas_uvs with mocked subprocess."""

    @pytest.fixture()
    def env_setup(self, monkeypatch, tmp_path):
        so_dir = tmp_path / "so_pkg"
        (so_dir / "python").mkdir(parents=True)
        (so_dir / "lib").mkdir()
        (so_dir / "extraLibs").mkdir()
        (so_dir / "usdpy").mkdir()
        monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(so_dir))
        monkeypatch.setenv("WU_SO_PYTHON", sys.executable)

    def test_default_params(self, env_setup, tmp_path):
        """Default atlas UV params are correct."""
        captured = {}

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            captured.update(params)
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateAtlasUVs",
                        "operation_time": 0.5,
                        "total_time": 0.6,
                        "stage_size_bytes": 2048,
                        "mesh_count": 10,
                        "meshes_with_uvs": 10,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            result = generate_atlas_uvs(input_path, tmp_path / "out.usd")

        assert result["status"] == "success"
        assert result["meshes_with_uvs"] == 10
        assert captured["operation"] == "generateAtlasUVs"
        op = captured["op_params"]
        assert op["distortionThreshold"] == 3.0
        assert op["enableAtlasPacking"] is True
        assert op["scaleFactor"] == 0.01
        assert op["overwriteExisting"] is True

    def test_custom_params(self, env_setup, tmp_path):
        """Custom atlas UV params are passed correctly."""
        captured = {}

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            captured.update(params)
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateAtlasUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 0,
                        "mesh_count": 0,
                        "meshes_with_uvs": 0,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            generate_atlas_uvs(
                input_path,
                tmp_path / "out.usd",
                distortion_threshold=2.0,
                enable_atlas_packing=False,
                scale_factor=0.02,
                scale_units=1.0,
                overwrite_existing=False,
            )

        op = captured["op_params"]
        assert op["distortionThreshold"] == 2.0
        assert op["enableAtlasPacking"] is False
        assert op["scaleFactor"] == 0.02
        assert op["scaleUnits"] == 1.0
        assert op["overwriteExisting"] is False

    def test_paths_filter(self, env_setup, tmp_path):
        """Specific prim paths are passed through for atlas UVs."""
        captured = {}

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[-1])
            captured.update(params)
            with open(params["manifest_path"], "w") as f:
                json.dump(
                    {
                        "status": "success",
                        "operation": "generateAtlasUVs",
                        "operation_time": 0.1,
                        "total_time": 0.2,
                        "stage_size_bytes": 0,
                        "mesh_count": 1,
                        "meshes_with_uvs": 1,
                    },
                    f,
                )
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            generate_atlas_uvs(input_path, tmp_path / "out.usd", paths=["/World/Mesh"])

        assert captured["op_params"]["paths"] == ["/World/Mesh"]


# ---------------------------------------------------------------------------
# Integration tests (require an unpacked Scene Optimizer Core package)
# ---------------------------------------------------------------------------


def _so_core_available() -> bool:
    """Return True when WU_SO_PACKAGE_DIR points at a usable SO Core layout."""
    pkg = os.environ.get("WU_SO_PACKAGE_DIR")
    if not pkg:
        return False
    root = Path(pkg)
    required_subdirs = ("python", "lib", "extraLibs", "usdpy")
    return all((root / sub).is_dir() for sub in required_subdirs)


HAS_SO_CORE = _so_core_available()

_DATA_DIR = (
    Path(__file__).parent.parent
    / "apps"
    / "material_agent"
    / "data"
    / "regression"
    / "scene_optimizer"
)
_TEST_SCENE = _DATA_DIR / "scene_optimizer_test.usda"


@pytest.mark.skipif(
    not HAS_SO_CORE,
    reason="scene_optimizer_core not unpacked (set WU_SO_PACKAGE_DIR)",
)
class TestUVGenerationIntegration:
    """Integration tests that run real SO UV generation."""

    def test_projection_uvs_cube(self, tmp_path):
        """generateProjectionUVs creates primvars:st on meshes."""
        output = tmp_path / "proj_uvs.usdc"
        result = generate_projection_uvs(
            _TEST_SCENE, output, projection_type=ProjectionType.CUBE
        )

        assert result["status"] == "success"
        assert result["meshes_with_uvs"] > 0
        assert output.exists()

        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(output))
        uv_count = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                st = prim.GetAttribute("primvars:st")
                if st and st.HasAuthoredValue():
                    uv_count += 1
        assert uv_count > 0

    def test_projection_uvs_planar(self, tmp_path):
        """Planar projection also works."""
        output = tmp_path / "planar_uvs.usdc"
        result = generate_projection_uvs(
            _TEST_SCENE, output, projection_type=ProjectionType.PLANAR
        )
        assert result["status"] == "success"
        assert result["meshes_with_uvs"] > 0

    def test_atlas_uvs(self, tmp_path):
        """generateAtlasUVs creates primvars:st on meshes."""
        output = tmp_path / "atlas_uvs.usdc"
        result = generate_atlas_uvs(_TEST_SCENE, output)

        assert result["status"] == "success"
        assert result["meshes_with_uvs"] > 0
        assert output.exists()

        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(output))
        uv_count = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                st = prim.GetAttribute("primvars:st")
                if st and st.HasAuthoredValue():
                    uv_count += 1
        assert uv_count > 0

    def test_overwrite_false_skips_existing(self, tmp_path):
        """overwrite_existing=False preserves existing UVs."""
        # First pass: generate UVs
        intermediate = tmp_path / "with_uvs.usdc"
        generate_projection_uvs(
            _TEST_SCENE, intermediate, projection_type=ProjectionType.CUBE
        )

        # Second pass: should skip (already have UVs)
        output = tmp_path / "skip_existing.usdc"
        result = generate_projection_uvs(intermediate, output, overwrite_existing=False)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# NVCF backend unit tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestBuildUvGenerationSettings:
    """Test _build_uv_generation_settings helper."""

    def test_projection_uvs(self):
        settings = _build_uv_generation_settings(
            "generateProjectionUVs",
            {"projectionType": 4, "scaleFactor": 0.01},
        )
        assert settings["enable_generate_projection_uvs"] is True
        assert settings["enable_generate_atlas_uvs"] is False
        assert settings["generate_projection_uvs"]["projectionType"] == 4

    def test_atlas_uvs(self):
        settings = _build_uv_generation_settings(
            "generateAtlasUVs",
            {"distortionThreshold": 3.0},
            output_format="usda",
        )
        assert settings["enable_generate_projection_uvs"] is False
        assert settings["enable_generate_atlas_uvs"] is True
        assert settings["generate_atlas_uvs"]["distortionThreshold"] == 3.0
        assert settings["output_format"] == "usda"


class TestNvcfBackendMocked:
    """Test NVCF backend with mocked execute_nvcf_request_async."""

    @pytest.mark.asyncio
    async def test_generate_uvs_from_url_reports_missing_stage_base64(self, tmp_path):
        """Successful transport without payload returns an error result."""

        async def mock_execute(*args, **kwargs):
            return {"success": True, "operations_executed": ["generateProjectionUVs"]}

        with (
            patch(
                "world_understanding.utils.nvcf_utils.get_nvcf_api_key",
                return_value="test-key",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_base_url",
                return_value="https://api.nvcf.nvidia.com/v2/func/123",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.execute_nvcf_request_async",
                side_effect=mock_execute,
            ),
        ):
            result = await uv._generate_uvs_from_url(
                "https://example.test/input.usd",
                tmp_path / "output.usdc",
                "generateProjectionUVs",
                {"projectionType": 4},
            )

        assert result["status"] == "error"
        assert result["error"] == "No generated_stage_base64 in response"

    @pytest.mark.asyncio
    async def test_generate_uvs_from_url_counts_meshes_with_uvs(self, tmp_path):
        """NVCF helper counts mesh UVs in the returned stage when pxr can open it."""
        generated_usd = b"""#usda 1.0
def Mesh "Mesh"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
        interpolation = "faceVarying"
    )
}
"""
        fake_b64 = base64.b64encode(generated_usd).decode()

        async def mock_execute(*args, **kwargs):
            return {
                "success": True,
                "operations_executed": ["generateProjectionUVs"],
                "generated_stage_base64": fake_b64,
            }

        with (
            patch(
                "world_understanding.utils.nvcf_utils.get_nvcf_api_key",
                return_value="test-key",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_base_url",
                return_value="https://api.nvcf.nvidia.com/v2/func/123",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.execute_nvcf_request_async",
                side_effect=mock_execute,
            ),
        ):
            result = await uv._generate_uvs_from_url(
                "https://example.test/input.usd",
                tmp_path / "output.usda",
                "generateProjectionUVs",
                {"projectionType": 4},
            )

        assert result["status"] == "success"
        assert result["mesh_count"] == 1
        assert result["meshes_with_uvs"] == 1

    @pytest.mark.asyncio
    async def test_generate_uvs_from_path_uploads_to_s3_and_cleans_up(self, tmp_path):
        """S3 upload path delegates to URL helper and deletes the temporary object."""
        input_path = tmp_path / "input.txt"
        input_path.write_bytes(b"usd")
        deleted: list[str] = []
        captured: dict[str, object] = {}

        async def fake_generate_from_url(**kwargs):
            captured.update(kwargs)
            return {"status": "success", "operation": kwargs["operation"]}

        with (
            patch(
                "world_understanding.utils.s3_utils.upload_file_to_s3",
                return_value="s3://bucket/key",
            ) as upload,
            patch(
                "world_understanding.utils.nvcf_utils.s3_uri_to_https_url",
                return_value="https://bucket.example/key",
            ),
            patch(
                "world_understanding.utils.s3_utils.delete_s3_path",
                side_effect=lambda s3_uri, profile_name=None: deleted.append(s3_uri),
            ),
            patch(
                "world_understanding.functions.graphics.uv_generation._generate_uvs_from_url",
                new=fake_generate_from_url,
            ),
        ):
            result = await uv._generate_uvs_from_path(
                input_path,
                tmp_path / "out.usd",
                "generateProjectionUVs",
                {"projectionType": 4},
                use_data_uri=False,
            )

        assert result == {"status": "success", "operation": "generateProjectionUVs"}
        assert upload.call_args.kwargs["s3_path"].endswith("/input.usd")
        assert captured["input_url"] == "https://bucket.example/key"
        assert deleted == ["s3://bucket/key"]

    def test_nvcf_projection_uvs(self, tmp_path):
        """backend='remote' calls NVCF /generate-uvs and writes output."""
        fake_usd = b"fake-usd-content"
        fake_b64 = base64.b64encode(fake_usd).decode()

        async def mock_execute(*args, **kwargs):
            return {
                "success": True,
                "operations_executed": ["generateProjectionUVs"],
                "generated_stage_base64": fake_b64,
            }

        input_path = tmp_path / "input.usd"
        input_path.write_bytes(b"input")
        output_path = tmp_path / "output.usdc"

        with (
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".should_use_data_uri",
                return_value=True,
            ),
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".create_data_uri_from_file",
                return_value="data:application/octet-stream;base64,AAAA",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_nvcf_api_key",
                return_value="test-key",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_base_url",
                return_value="https://api.nvcf.nvidia.com/v2/func/123",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.execute_nvcf_request_async",
                side_effect=mock_execute,
            ),
        ):
            result = generate_projection_uvs(input_path, output_path, backend="remote")

        assert result["status"] == "success"
        assert output_path.exists()
        assert output_path.read_bytes() == fake_usd

    def test_nvcf_atlas_uvs(self, tmp_path):
        """backend='remote' works for atlas UVs too."""
        fake_usd = b"atlas-usd"
        fake_b64 = base64.b64encode(fake_usd).decode()

        async def mock_execute(*args, **kwargs):
            return {
                "success": True,
                "operations_executed": ["generateAtlasUVs"],
                "generated_stage_base64": fake_b64,
            }

        input_path = tmp_path / "input.usd"
        input_path.write_bytes(b"input")
        output_path = tmp_path / "output.usdc"

        with (
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".should_use_data_uri",
                return_value=True,
            ),
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".create_data_uri_from_file",
                return_value="data:application/octet-stream;base64,AAAA",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_nvcf_api_key",
                return_value="test-key",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_base_url",
                return_value="https://api.nvcf.nvidia.com/v2/func/123",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.execute_nvcf_request_async",
                side_effect=mock_execute,
            ),
        ):
            result = generate_atlas_uvs(input_path, output_path, backend="remote")

        assert result["status"] == "success"
        assert output_path.read_bytes() == fake_usd

    def test_nvcf_failure_raises(self, tmp_path):
        """NVCF error raises RuntimeError."""

        async def mock_execute(*args, **kwargs):
            return {"success": False}

        input_path = tmp_path / "input.usd"
        input_path.write_bytes(b"input")

        with (
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".should_use_data_uri",
                return_value=True,
            ),
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".create_data_uri_from_file",
                return_value="data:application/octet-stream;base64,AAAA",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_nvcf_api_key",
                return_value="test-key",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_base_url",
                return_value="https://api.nvcf.nvidia.com/v2/func/123",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.execute_nvcf_request_async",
                side_effect=mock_execute,
            ),
        ):
            with pytest.raises(RuntimeError, match="NVCF UV generation failed"):
                generate_projection_uvs(
                    input_path, tmp_path / "out.usdc", backend="remote"
                )

    def test_invalid_backend(self, tmp_path):
        """Invalid backend raises ValueError."""
        input_path = tmp_path / "input.usd"
        input_path.touch()
        with pytest.raises(ValueError, match="Invalid backend"):
            generate_projection_uvs(input_path, tmp_path / "out.usdc", backend="bogus")

    def test_nvcf_returns_mesh_counts(self, tmp_path):
        """NVCF result includes mesh_count and meshes_with_uvs."""
        fake_usd = b"fake-usd"
        fake_b64 = base64.b64encode(fake_usd).decode()

        async def mock_execute(*args, **kwargs):
            return {
                "success": True,
                "operations_executed": ["generateProjectionUVs"],
                "generated_stage_base64": fake_b64,
            }

        input_path = tmp_path / "input.usd"
        input_path.write_bytes(b"input")
        output_path = tmp_path / "output.usdc"

        with (
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".should_use_data_uri",
                return_value=True,
            ),
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".create_data_uri_from_file",
                return_value="data:application/octet-stream;base64,AAAA",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_nvcf_api_key",
                return_value="test-key",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_base_url",
                return_value="https://api.nvcf.nvidia.com/v2/func/123",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.execute_nvcf_request_async",
                side_effect=mock_execute,
            ),
        ):
            result = generate_projection_uvs(input_path, output_path, backend="remote")

        # mesh_count and meshes_with_uvs must be present in result
        assert "mesh_count" in result
        assert "meshes_with_uvs" in result


class TestLocalAutoFallback:
    """Test that backend='local' auto-falls back to NVCF."""

    def test_fallback_on_missing_package(self, monkeypatch, tmp_path):
        """Local backend auto-falls back to NVCF when SO package is missing."""
        monkeypatch.delenv("WU_SO_PACKAGE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)

        fake_usd = b"nvcf-fallback"
        fake_b64 = base64.b64encode(fake_usd).decode()

        async def mock_execute(*args, **kwargs):
            return {
                "success": True,
                "operations_executed": ["generateProjectionUVs"],
                "generated_stage_base64": fake_b64,
            }

        input_path = tmp_path / "input.usd"
        input_path.write_bytes(b"input")
        output_path = tmp_path / "output.usdc"

        with (
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".should_use_data_uri",
                return_value=True,
            ),
            patch(
                "world_understanding.functions.graphics.uv_generation"
                ".create_data_uri_from_file",
                return_value="data:application/octet-stream;base64,AAAA",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_nvcf_api_key",
                return_value="test-key",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.get_base_url",
                return_value="https://api.nvcf.nvidia.com/v2/func/123",
            ),
            patch(
                "world_understanding.utils.nvcf_utils.execute_nvcf_request_async",
                side_effect=mock_execute,
            ),
        ):
            # backend="local" (default) should auto-fall back to NVCF
            result = generate_projection_uvs(input_path, output_path)

        assert result["status"] == "success"
        assert output_path.read_bytes() == fake_usd

    def test_disable_remote_fallback(self, tmp_path):
        """Local backend can be kept local-only for task-level fallback."""
        input_path = tmp_path / "input.usd"
        input_path.write_bytes(b"input")
        output_path = tmp_path / "output.usdc"

        with (
            patch(
                "world_understanding.functions.graphics.uv_generation._run_uv_worker",
                side_effect=RuntimeError("SO package missing directory: python"),
            ),
            patch(
                "world_understanding.functions.graphics.uv_generation._run_uv_nvcf"
            ) as nvcf,
        ):
            with pytest.raises(RuntimeError, match="SO package missing directory"):
                generate_projection_uvs(
                    input_path,
                    output_path,
                    allow_remote_fallback=False,
                )

        nvcf.assert_not_called()


# ---------------------------------------------------------------------------
# A/B comparison: local vs NVCF (require both SO bundle and NVCF credentials)
# ---------------------------------------------------------------------------

HAS_NVCF = bool(os.environ.get("NGC_API_KEY"))
HAS_NVCF_UV_ENDPOINT = bool(os.environ.get("NVCF_UV_GENERATION_ENABLED"))


@pytest.mark.skipif(
    not HAS_SO_CORE,
    reason="scene_optimizer_core not unpacked (set WU_SO_PACKAGE_DIR)",
)
@pytest.mark.skipif(not HAS_NVCF, reason="NGC_API_KEY not set")
@pytest.mark.skipif(
    not HAS_NVCF_UV_ENDPOINT,
    reason="NVCF_UV_GENERATION_ENABLED not set (endpoint not deployed yet)",
)
class TestLocalVsNvcfComparison:
    """A/B tests: local and NVCF backends must produce identical primvars:st."""

    @staticmethod
    def _get_mesh_uvs(usd_path: Path) -> dict[str, list]:
        """Extract primvars:st per mesh from a USD file."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(usd_path))
        mesh_uvs = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                st = prim.GetAttribute("primvars:st")
                if st and st.HasAuthoredValue():
                    mesh_uvs[str(prim.GetPath())] = list(st.Get())
        return mesh_uvs

    def test_projection_uvs_local_vs_nvcf(self, tmp_path):
        """Projection UVs from local and NVCF must match."""
        local_out = tmp_path / "local_proj.usdc"
        nvcf_out = tmp_path / "nvcf_proj.usdc"

        generate_projection_uvs(
            _TEST_SCENE,
            local_out,
            projection_type=ProjectionType.CUBE,
            backend="local",
        )
        generate_projection_uvs(
            _TEST_SCENE,
            nvcf_out,
            projection_type=ProjectionType.CUBE,
            backend="remote",
        )

        local_uvs = self._get_mesh_uvs(local_out)
        nvcf_uvs = self._get_mesh_uvs(nvcf_out)

        assert set(local_uvs.keys()) == set(nvcf_uvs.keys()), (
            f"Mesh sets differ: local={set(local_uvs.keys())}, "
            f"nvcf={set(nvcf_uvs.keys())}"
        )

        import numpy as np

        for prim_path in local_uvs:
            local_arr = np.array(local_uvs[prim_path])
            nvcf_arr = np.array(nvcf_uvs[prim_path])
            np.testing.assert_allclose(
                local_arr,
                nvcf_arr,
                atol=1e-5,
                err_msg=f"UV mismatch at {prim_path}",
            )

    def test_atlas_uvs_local_vs_nvcf(self, tmp_path):
        """Atlas UVs from local and NVCF must match."""
        local_out = tmp_path / "local_atlas.usdc"
        nvcf_out = tmp_path / "nvcf_atlas.usdc"

        generate_atlas_uvs(_TEST_SCENE, local_out, backend="local")
        generate_atlas_uvs(_TEST_SCENE, nvcf_out, backend="remote")

        local_uvs = self._get_mesh_uvs(local_out)
        nvcf_uvs = self._get_mesh_uvs(nvcf_out)

        assert set(local_uvs.keys()) == set(nvcf_uvs.keys()), (
            f"Mesh sets differ: local={set(local_uvs.keys())}, "
            f"nvcf={set(nvcf_uvs.keys())}"
        )

        import numpy as np

        for prim_path in local_uvs:
            local_arr = np.array(local_uvs[prim_path])
            nvcf_arr = np.array(nvcf_uvs[prim_path])
            np.testing.assert_allclose(
                local_arr,
                nvcf_arr,
                atol=1e-5,
                err_msg=f"UV mismatch at {prim_path}",
            )
