# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OptimizeUSDTask local backend dispatch and asyncio.to_thread wrapping."""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from world_understanding.agentic.usd_tasks import optimize_usd as optimize_usd_module
from world_understanding.agentic.usd_tasks.optimize_usd import OptimizeUSDTask


def _traceback_frame_locals(
    error: BaseException,
    function_name: str,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    cursor = error.__traceback__
    while cursor is not None:
        if cursor.tb_frame.f_code.co_name == function_name:
            frames.append(dict(cursor.tb_frame.f_locals))
        cursor = cursor.tb_next
    return frames


def test_derived_path_redaction_reuses_the_central_safe_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "ZQX713DerivedPathCredentialMNP9"
    replacement_marker = object()
    monkeypatch.setattr(
        optimize_usd_module,
        "redact_sensitive_config",
        lambda _value: replacement_marker,
    )

    safe_source, source_is_sensitive = optimize_usd_module._redacted_log_projection(
        f"https://user:{sentinel}@example.test/input.usd"
    )

    assert source_is_sensitive is True
    assert safe_source is replacement_marker
    assert (
        optimize_usd_module._redacted_derived_path(
            Path("derived.usd"), source_redaction=safe_source
        )
        is replacement_marker
    )


def _make_context(tmp_path: Path, backend: str = "local") -> dict:
    input_usd = tmp_path / "input.usd"
    input_usd.touch()
    output_usd = tmp_path / "output.usd"
    return {
        "input_usd_path": str(input_usd),
        "output_usd_path": str(output_usd),
        "optimization_config": {
            "backend": backend,
            "flatten_prototypes": False,
            "scene_optimizer_settings": {
                "enable_deinstance": False,
                "enable_split_meshes": True,
                "enable_deduplicate": True,
            },
        },
    }


def _mock_usd_open(prim_count: int = 3):
    """Return a patched pxr.Usd.Stage.Open that returns a mock stage."""
    mock_prim = MagicMock()
    mock_prim.IsA = MagicMock(return_value=True)
    mock_stage = MagicMock()
    mock_stage.Traverse.return_value = [mock_prim] * prim_count
    return mock_stage


def _restrict_stage_paths(stage: MagicMock, paths: set[str]) -> None:
    """Make a mock stage prove membership only for the supplied prim paths."""

    def get_prim(path: object) -> MagicMock:
        prim = MagicMock()
        prim.__bool__.return_value = str(path) in paths
        return prim

    stage.GetPrimAtPath.side_effect = get_prim


def test_durable_optimizer_config_projection_is_json_safe_and_deterministic() -> None:
    runtime_client = object()
    config = {
        "api_key": "<redacted>",
        "features": {"zeta", 2, ("nested", 1)},
        "frozen": frozenset({"beta", 1}),
        "tuple": (True, "value"),
        "runtime_client": runtime_client,
        "runtime_path": Path("runtime-only.usd"),
    }

    projected = optimize_usd_module._project_durable_optimization_config(config)

    assert projected == {
        "api_key": "<redacted>",
        "features": [2, "zeta", ["nested", 1]],
        "frozen": [1, "beta"],
        "tuple": [True, "value"],
    }
    assert optimize_usd_module._project_durable_optimization_config(config) == projected
    json.dumps(projected, allow_nan=False)


def test_durable_optimizer_config_omits_whole_sequence_with_runtime_member() -> None:
    projected = optimize_usd_module._project_durable_optimization_config(
        {
            "ordered": ["before", object(), "after"],
            "runtime_set": {"kept", object()},
            "ordinary_runtime_leaf": object(),
            "preserved": ["before", "after"],
        }
    )

    assert projected == {"preserved": ["before", "after"]}


@pytest.mark.parametrize(
    "unsupported",
    [
        float("nan"),
        float("inf"),
        b"binary",
        date(2026, 7, 15),
        datetime(2026, 7, 15, 12, 30),
        {7: "non-string key"},
    ],
)
def test_durable_optimizer_config_rejects_unsupported_config_values_value_free(
    unsupported: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="^Unsupported durable optimizer configuration$",
    ) as exc_info:
        optimize_usd_module._project_durable_optimization_config(
            {"unsupported": unsupported}
        )

    assert repr(unsupported) not in str(exc_info.value)


def test_durable_optimizer_config_rejects_recursive_containers_value_free() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    with pytest.raises(
        ValueError,
        match="^Unsupported durable optimizer configuration$",
    ):
        optimize_usd_module._project_durable_optimization_config(
            {"recursive": recursive}
        )


def test_atomic_metadata_write_preserves_existing_file_on_serialization_failure(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "output.metadata.json"
    metadata_path.write_text('{"status": "existing"}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        optimize_usd_module._write_json_atomic(
            metadata_path,
            {"non_finite": float("nan")},
        )

    assert metadata_path.read_text(encoding="utf-8") == '{"status": "existing"}\n'
    assert list(tmp_path.glob(f".{metadata_path.name}.*.tmp")) == []


def test_atomic_metadata_write_preserves_existing_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path = tmp_path / "output.metadata.json"
    metadata_path.write_text('{"status": "existing"}\n', encoding="utf-8")

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr(optimize_usd_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replacement failed"):
        optimize_usd_module._write_json_atomic(metadata_path, {"status": "new"})

    assert metadata_path.read_text(encoding="utf-8") == '{"status": "existing"}\n'
    assert list(tmp_path.glob(f".{metadata_path.name}.*.tmp")) == []


def test_malformed_backend_metadata_is_discarded_without_throwing() -> None:
    huge_integer = 10**10000

    metadata = optimize_usd_module._project_backend_metadata(
        {
            "optimization_time": huge_integer,
            "stage_size_bytes": huge_integer,
            "operations_executed": [
                {"name": []},
                {"name": "split", "time": huge_integer},
            ],
        },
        original_stage=MagicMock(),
        optimized_stage=MagicMock(),
    )

    assert metadata == {
        "optimization_time": None,
        "stage_size_bytes": None,
        "operations_executed": [{"name": "split"}],
        "correspondence_map": {},
    }


class TestOptimizeUSDTaskLocalBackend:
    """Tests for the local backend branch in OptimizeUSDTask.arun() (lines 196-207)."""

    def test_sync_run_wrapper_delegates_to_async_validation(self):
        with pytest.raises(ValueError, match="input_usd_path is required"):
            OptimizeUSDTask().run({})

    def test_get_enabled_operations_defaults_to_all_operations(self):
        assert OptimizeUSDTask()._get_enabled_operations({}) == [
            "deinstance",
            "split",
            "deduplicate",
        ]

    @pytest.mark.asyncio
    async def test_local_backend_uses_asyncio_to_thread(self, tmp_path):
        """backend='local' must wrap optimize_usd_local in asyncio.to_thread."""
        context = _make_context(tmp_path, backend="local")
        mock_stage = _mock_usd_open()

        local_result = {
            "status": "success",
            "optimization_time": 1.0,
            "operations_executed": ["split", "deduplicate"],
        }

        captured_calls: list = []

        async def fake_to_thread(fn: object, **kwargs: Any) -> dict[str, object]:
            captured_calls.append({"fn": fn, "kwargs": kwargs})
            Path(kwargs["output_path"]).touch()
            return local_result

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
            ) as mock_nvcf,
            patch(
                "world_understanding.functions.graphics.scene_optimizer_local.optimize_usd_local",
                return_value=local_result,
            ),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "_restore_optimized_stage_metadata",
                return_value=({"restored": True}, mock_stage),
            ),
            patch("asyncio.to_thread", side_effect=fake_to_thread),
        ):
            task = OptimizeUSDTask()
            result = await task.arun(context)

        # NVCF path must NOT have been invoked
        mock_nvcf.assert_not_called()

        # asyncio.to_thread must have been called exactly once
        assert len(captured_calls) == 1, (
            f"Expected asyncio.to_thread to be called once, got {len(captured_calls)}"
        )

        # The callable passed to to_thread must be optimize_usd_local.
        # When patched, it is a MagicMock whose repr contains the name.
        fn = captured_calls[0]["fn"]
        assert "optimize_usd_local" in str(fn), (
            f"Expected optimize_usd_local callable, got {fn!r}"
        )

        # Context should reflect success
        assert result["optimization_success"] is True

    @pytest.mark.asyncio
    async def test_plain_config_remote_failure_uses_constant_safe_error(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sentinel = "optimizer-live-secret"
        context = _make_context(tmp_path, backend="remote")
        context["optimization_config"] = {
            "backend": "remote",
            "flatten_prototypes": False,
            "custom_option": True,
            "api_key": sentinel,
            "nested": {"token": "xy"},
        }
        mock_stage = _mock_usd_open()

        with caplog.at_level(logging.INFO):
            with (
                patch("pxr.Usd.Stage.Open", return_value=mock_stage),
                patch("pxr.UsdGeom.Mesh", MagicMock()),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                    new_callable=AsyncMock,
                    return_value={
                        "status": "failed",
                        "error": (
                            f"request rejected for https://user:{sentinel}@example.test; "
                            f"key fragment={sentinel[:8]}"
                        ),
                    },
                ),
            ):
                with pytest.raises(
                    RuntimeError, match="^USD optimization failed$"
                ) as exc:
                    await OptimizeUSDTask().arun(context)

        assert context["optimization_success"] is False
        assert context["optimization_error"] == "USD optimization failed"
        observable = f"{exc.value}\n{caplog.text}\n{context['optimization_error']}"
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None
        assert sentinel not in observable
        assert sentinel[:8] not in observable
        assert sentinel[-8:] not in observable
        assert "<redacted>" in caplog.text

    @pytest.mark.asyncio
    async def test_optimizer_settings_use_redacted_diagnostics_but_raw_runtime_values(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sentinel = "ZQX713OptimizerSettingsCredentialMNP9"
        logged_fields = (
            "generate_report",
            "capture_stats",
            "verbose",
            "wait_for_assets",
            "stage_timeout",
            "extract_geom_subset_indices",
        )
        raw_settings: dict[str, object] = {
            "enable_deinstance": False,
            "enable_split_meshes": True,
            "enable_deduplicate": True,
            **{
                field: (
                    f"https://optimizer.example.test/settings?sig={sentinel}-{field}"
                )
                for field in logged_fields
            },
        }
        context = _make_context(tmp_path, backend="remote")
        context["optimization_config"]["scene_optimizer_settings"] = raw_settings
        mock_stage = _mock_usd_open()

        with caplog.at_level(logging.INFO):
            with (
                patch("pxr.Usd.Stage.Open", return_value=mock_stage),
                patch("pxr.UsdGeom.Mesh", MagicMock()),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd."
                    "optimize_usd_from_path",
                    new_callable=AsyncMock,
                    return_value={"status": "failed"},
                ) as mock_nvcf,
            ):
                with pytest.raises(RuntimeError, match="^USD optimization failed$"):
                    await OptimizeUSDTask().arun(context)

        runtime_config = mock_nvcf.await_args.kwargs["optimization_config"]
        assert runtime_config["scene_optimizer_settings"] == raw_settings
        observable = caplog.text
        assert sentinel not in observable
        assert sentinel[:10] not in observable
        assert sentinel[-10:] not in observable
        assert "optimizer.example.test" not in observable
        assert observable.count("<redacted>") >= len(logged_fields)

    @pytest.mark.asyncio
    async def test_config_diagnostics_do_not_repr_opaque_or_path_values(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sentinel = "optimizer-runtime-diagnostic-sentinel-727"

        class ExplosiveRepr:
            def __repr__(self) -> str:
                raise AssertionError(f"opaque repr rendered: {sentinel}")

        runtime_client = ExplosiveRepr()
        runtime_path = Path(f"client_secret={sentinel}")
        context = _make_context(tmp_path, backend="remote")
        context["optimization_config"] = {
            "backend": "remote",
            "flatten_prototypes": False,
            "api_key": "xy",
            "runtime_client": runtime_client,
            "runtime_path": runtime_path,
        }
        mock_stage = _mock_usd_open()

        async def successful_optimizer(**kwargs: Any) -> dict[str, object]:
            Path(kwargs["output_path"]).touch()
            assert kwargs["optimization_config"]["runtime_client"] is runtime_client
            assert kwargs["optimization_config"]["runtime_path"] is runtime_path
            return {"status": "success"}

        with caplog.at_level(logging.INFO):
            with (
                patch("pxr.Usd.Stage.Open", return_value=mock_stage),
                patch("pxr.UsdGeom.Mesh", MagicMock()),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd."
                    "optimize_usd_from_path",
                    side_effect=successful_optimizer,
                ),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd."
                    "_restore_optimized_stage_metadata",
                    return_value=({"restored": True}, mock_stage),
                ),
            ):
                result = await OptimizeUSDTask().arun(context)

        persisted = json.dumps(result["optimization_metadata"], sort_keys=True)
        assert sentinel not in f"{caplog.text}\n{persisted}"
        assert "runtime_client" not in persisted
        assert "Optimization config:" in caplog.text

    @pytest.mark.asyncio
    async def test_remote_exception_text_is_not_propagated(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sentinel = "ZQX713ReflectedCredentialMNP9"
        context = _make_context(tmp_path, backend="remote")
        mock_stage = _mock_usd_open()

        with caplog.at_level(logging.INFO):
            with (
                patch("pxr.Usd.Stage.Open", return_value=mock_stage),
                patch("pxr.UsdGeom.Mesh", MagicMock()),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError(
                        f"backend echoed key {sentinel} and {sentinel[-8:]}"
                    ),
                ),
            ):
                with pytest.raises(
                    RuntimeError, match="^USD optimization failed$"
                ) as exc:
                    await OptimizeUSDTask().arun(context)

        observable = f"{exc.value}\n{caplog.text}\n{context['optimization_error']}"
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None
        assert sentinel not in observable
        assert sentinel[:8] not in observable
        assert sentinel[-8:] not in observable

    @pytest.mark.asyncio
    async def test_nvcf_backend_does_not_use_to_thread(self, tmp_path):
        """backend='remote' must call optimize_usd_from_path and not asyncio.to_thread."""
        context = _make_context(tmp_path, backend="remote")
        context["optimization_config"]["api_key"] = "xy"
        Path(context["output_usd_path"]).touch()
        mock_stage = _mock_usd_open()

        nvcf_result = {
            "status": "success",
            "optimization_time": 2.0,
            "operations_executed": ["split", "deduplicate"],
        }

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                new_callable=AsyncMock,
                return_value=nvcf_result,
            ) as mock_nvcf,
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "_restore_optimized_stage_metadata",
                return_value=({"restored": True}, mock_stage),
            ),
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            task = OptimizeUSDTask()
            result = await task.arun(context)

        mock_nvcf.assert_called_once()
        mock_to_thread.assert_not_called()
        assert result["optimization_success"] is True
        metadata = json.loads(
            (tmp_path / "output.metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["optimization_config"]["api_key"] == "<redacted>"
        assert "xy" not in json.dumps(metadata)

    @pytest.mark.asyncio
    async def test_success_sidecar_omits_runtime_leaves_and_normalizes_sets(
        self,
        tmp_path: Path,
    ) -> None:
        context = _make_context(tmp_path, backend="remote")
        runtime_client = object()
        context["optimization_config"].update(
            {
                "api_key": "optimizer-live-secret",
                "features": {"zeta", 2},
                "frozen": frozenset({"beta", 1}),
                "runtime_client": runtime_client,
                "runtime_path": Path("runtime-only.usd"),
            }
        )
        mock_stage = _mock_usd_open()

        async def successful_optimizer(**kwargs: Any) -> dict[str, object]:
            Path(kwargs["output_path"]).touch()
            assert kwargs["optimization_config"]["runtime_client"] is runtime_client
            return {"status": "success", "optimization_time": 1.0}

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "optimize_usd_from_path",
                side_effect=successful_optimizer,
            ),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "_restore_optimized_stage_metadata",
                return_value=({"restored": True}, mock_stage),
            ),
        ):
            result = await OptimizeUSDTask().arun(context)

        persisted = json.loads(
            (tmp_path / "output.metadata.json").read_text(encoding="utf-8")
        )
        assert persisted["optimization_config"] == {
            "backend": "remote",
            "flatten_prototypes": False,
            "scene_optimizer_settings": {
                "enable_deinstance": False,
                "enable_split_meshes": True,
                "enable_deduplicate": True,
            },
            "api_key": "<redacted>",
            "features": [2, "zeta"],
            "frozen": [1, "beta"],
        }
        assert result["optimization_metadata"] == persisted

    @pytest.mark.asyncio
    async def test_success_without_output_stage_is_not_published(
        self,
        tmp_path: Path,
    ) -> None:
        context = _make_context(tmp_path, backend="remote")
        mock_stage = _mock_usd_open()

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "optimize_usd_from_path",
                new_callable=AsyncMock,
                return_value={"status": "success"},
            ),
            pytest.raises(RuntimeError, match="^USD optimization failed$"),
        ):
            await OptimizeUSDTask().arun(context)

        assert context["optimization_success"] is False
        assert "optimized_usd_path" not in context
        assert "optimization_metadata" not in context
        assert not (tmp_path / "output.metadata.json").exists()

    @pytest.mark.asyncio
    async def test_success_with_unreadable_output_stage_is_not_published(
        self,
        tmp_path: Path,
    ) -> None:
        context = _make_context(tmp_path, backend="remote")
        Path(context["output_usd_path"]).touch()
        source_stage = _mock_usd_open()

        with (
            patch("pxr.Usd.Stage.Open", side_effect=[source_stage, None]),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "optimize_usd_from_path",
                new_callable=AsyncMock,
                return_value={"status": "success"},
            ),
            pytest.raises(RuntimeError, match="^USD optimization failed$"),
        ):
            await OptimizeUSDTask().arun(context)

        assert context["optimization_success"] is False
        assert "optimized_usd_path" not in context
        assert "optimization_metadata" not in context
        assert not (tmp_path / "output.metadata.json").exists()

    @pytest.mark.asyncio
    async def test_sidecar_failure_does_not_publish_success_context(
        self,
        tmp_path: Path,
    ) -> None:
        context = _make_context(tmp_path, backend="remote")
        Path(context["output_usd_path"]).touch()
        mock_stage = _mock_usd_open()

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "optimize_usd_from_path",
                new_callable=AsyncMock,
                return_value={"status": "success"},
            ),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd._write_json_atomic",
                side_effect=OSError("replacement failed"),
            ),
            pytest.raises(RuntimeError, match="^USD optimization failed$"),
        ):
            await OptimizeUSDTask().arun(context)

        assert context["optimization_success"] is False
        assert "optimized_usd_path" not in context
        assert "optimization_metadata" not in context

    @pytest.mark.asyncio
    async def test_unsupported_durable_config_does_not_publish_success_context(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        context = _make_context(tmp_path, backend="remote")
        context["optimization_config"] = {
            "backend": "remote",
            "flatten_prototypes": False,
            "unsupported": float("nan"),
        }
        Path(context["output_usd_path"]).touch()
        mock_stage = _mock_usd_open()

        with caplog.at_level(logging.INFO):
            with (
                patch("pxr.Usd.Stage.Open", return_value=mock_stage),
                patch("pxr.UsdGeom.Mesh", MagicMock()),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd."
                    "optimize_usd_from_path",
                    new_callable=AsyncMock,
                    return_value={"status": "success"},
                ),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd."
                    "_restore_optimized_stage_metadata",
                    return_value=({"restored": True}, mock_stage),
                ),
                pytest.raises(RuntimeError, match="^USD optimization failed$"),
            ):
                await OptimizeUSDTask().arun(context)

        assert context["optimization_success"] is False
        assert context["optimization_error"] == "USD optimization failed"
        assert "optimized_usd_path" not in context
        assert "optimization_metadata" not in context
        assert not (tmp_path / "output.metadata.json").exists()
        assert "Optimization config: <unsupported>" in caplog.text
        assert "nan" not in caplog.text

    @pytest.mark.asyncio
    async def test_success_result_is_projected_before_any_durable_boundary(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sentinel = "ZQX713OptimizerSuccessCredentialMNP9"
        prefix = sentinel[:10]
        suffix = sentinel[-10:]
        signed_url = f"https://user:{sentinel}@example.test/output.usd"
        context = _make_context(tmp_path, backend="remote")
        Path(context["output_usd_path"]).touch()
        mock_stage = _mock_usd_open()
        _restrict_stage_paths(
            mock_stage,
            {"/World/Mesh", "/World/Mesh_part", "/World/Injected"},
        )
        backend_result = {
            "status": "success",
            "optimization_time": 2.5,
            "stage_size_bytes": 512,
            "operations_executed": [
                "split",
                {
                    "name": "deduplicateGeometry",
                    "success": True,
                    "time": 0.25,
                    "diagnostic": sentinel,
                },
                f"split-{prefix}",
                {"name": suffix, "success": True},
            ],
            "report": f"backend report contained {sentinel}, {prefix}, and {suffix}",
            "correspondence_map": {
                "summary": {
                    "operations_run": {"split": True},
                    "note": prefix,
                },
                "split_mapping": {
                    "/World/Mesh": ["/World/Mesh_part"],
                },
                "full_mapping": {
                    "original_to_prototype": {
                        "/World/Mesh": ["/World/Mesh_part"],
                        f"/World/{prefix}": ["/World/Injected"],
                    },
                },
                "nested_backend_data": {
                    f"api_key_{suffix}": sentinel,
                    "artifact_url": signed_url,
                },
            },
        }

        with caplog.at_level(logging.DEBUG):
            with (
                patch("pxr.Usd.Stage.Open", return_value=mock_stage),
                patch("pxr.UsdGeom.Mesh", MagicMock()),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                    new_callable=AsyncMock,
                    return_value=backend_result,
                ),
                patch(
                    "world_understanding.agentic.usd_tasks.optimize_usd."
                    "_restore_optimized_stage_metadata",
                    return_value=({"restored": True}, mock_stage),
                ),
            ):
                result = await OptimizeUSDTask().arun(context)

        metadata_path = tmp_path / "output.metadata.json"
        persisted = metadata_path.read_text(encoding="utf-8")
        observable = "\n".join(
            (
                json.dumps(result, sort_keys=True),
                caplog.text,
                persisted,
            )
        )
        assert sentinel not in observable
        assert prefix not in observable
        assert suffix not in observable

        metadata = json.loads(persisted)
        assert "report" not in metadata
        assert metadata["optimization_time"] == 2.5
        assert metadata["stage_size_bytes"] == 512
        assert metadata["operations_executed"] == [
            "split",
            {
                "name": "deduplicateGeometry",
                "success": True,
                "time": 0.25,
            },
        ]
        assert metadata["correspondence_map"] == {
            "summary": {"operations_run": {"split": True}},
            "split_mapping": {"/World/Mesh": ["/World/Mesh_part"]},
            "full_mapping": {
                "original_to_prototype": {
                    "/World/Mesh": ["/World/Mesh_part"],
                }
            },
        }

    @pytest.mark.asyncio
    async def test_valid_correspondence_map_remains_restore_compatible(
        self, tmp_path: Path
    ) -> None:
        context = _make_context(tmp_path, backend="remote")
        Path(context["output_usd_path"]).touch()
        mock_stage = _mock_usd_open()
        _restrict_stage_paths(
            mock_stage,
            {
                "/World/Mesh",
                "/World/Mesh_part",
                "/World/Mesh_part_1",
                "/World/Other",
            },
        )
        correspondence_map = {
            "summary": {
                "note": "free-form backend text is not durable metadata",
                "operations_run": {
                    "deinstance": False,
                    "split": True,
                    "deduplicate": True,
                    "unknown": True,
                },
                "total_original_prims": 2,
                "meshes_split": 1,
                "instances_tracked": 1,
                "unknown_count": 999,
            },
            "split_mapping": {
                "/World/Mesh": ["/World/Mesh_part", "/World/Mesh_part_1"],
            },
            "deduplication_mapping": {
                "instance_to_prototype": {
                    "/World/Mesh_part_1": "/World/Mesh_part",
                },
                "backend_debug": "discarded",
            },
            "full_mapping": {
                "original_to_prototype": {
                    "/World/Mesh": ["/World/Mesh_part", "/World/Mesh_part"],
                    "/World/Other": ["/World/Other"],
                },
            },
            "backend_extension": {"discarded": True},
        }
        backend_result = {
            "status": "success",
            "optimization_time": float("inf"),
            "stage_size_bytes": 512.5,
            "operations_executed": [
                {"name": "utilityFunction", "success": True, "time": 0.1},
                {"name": "splitMeshes", "success": True, "time": -1},
                {"name": "deduplicateGeometry", "success": True, "time": 0.2},
                {"name": "unknownOperation", "success": True},
            ],
            "correspondence_map": correspondence_map,
        }

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                new_callable=AsyncMock,
                return_value=backend_result,
            ),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "_restore_optimized_stage_metadata",
                return_value=({"restored": True}, mock_stage),
            ),
        ):
            result = await OptimizeUSDTask().arun(context)

        metadata = result["optimization_metadata"]
        assert metadata["optimization_time"] is None
        assert metadata["stage_size_bytes"] is None
        assert metadata["operations_executed"] == [
            {"name": "utilityFunction", "success": True, "time": 0.1},
            {"name": "splitMeshes", "success": True},
            {"name": "deduplicateGeometry", "success": True, "time": 0.2},
        ]
        assert metadata["correspondence_map"] == {
            "summary": {
                "operations_run": {
                    "deinstance": False,
                    "split": True,
                    "deduplicate": True,
                },
                "total_original_prims": 2,
                "meshes_split": 1,
                "instances_tracked": 1,
            },
            "split_mapping": {
                "/World/Mesh": ["/World/Mesh_part", "/World/Mesh_part_1"],
            },
            "deduplication_mapping": {
                "instance_to_prototype": {
                    "/World/Mesh_part_1": "/World/Mesh_part",
                }
            },
            "full_mapping": {
                "original_to_prototype": {
                    "/World/Mesh": ["/World/Mesh_part", "/World/Mesh_part"],
                    "/World/Other": ["/World/Other"],
                }
            },
        }

        from world_understanding.agentic.usd_tasks.restore_usd import RestoreUSDTask

        predictions_path = tmp_path / "predictions.jsonl"
        restored_path = tmp_path / "restored.jsonl"
        predictions_path.write_text(
            "\n".join(
                json.dumps({"id": prim_path, "material": material})
                for prim_path, material in (
                    ("/World/Mesh_part", "part-a"),
                    ("/World/Mesh_part_1", "part-b"),
                    ("/World/Other", "identity"),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        restore_task = RestoreUSDTask()
        with (
            patch.object(restore_task, "_open_stage", return_value=object()),
            patch.object(restore_task, "_get_geomsubset_paths", return_value=[]),
        ):
            restored = restore_task.run(
                {
                    "original_usd_path": tmp_path / "input.usd",
                    "predictions_path": predictions_path,
                    "output_predictions_path": restored_path,
                    "optimization_metadata": metadata,
                }
            )

        restored_rows = [
            json.loads(line)
            for line in restored_path.read_text(encoding="utf-8").splitlines()
        ]
        assert restored["restore_success"] is True
        assert any(row["id"] == "/World/Other" for row in restored_rows)

    @pytest.mark.asyncio
    async def test_remote_preflatten_preserves_face_varying_uvs(self, tmp_path):
        """The production pre-flatten path must not invalidate authored UVs."""
        from pxr import Sdf, Usd, UsdGeom, Vt

        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "optimized_input.usd"
        stage = Usd.Stage.CreateNew(str(input_usd))
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.GetRootLayer().customLayerData = {
            "SimReady_Metadata": '{"identifier":"test"}'
        }
        points = Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
        face_counts = Vt.IntArray([3, 3])
        face_indices = Vt.IntArray([0, 1, 2, 0, 2, 3])
        st_values = Vt.Vec2fArray([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)])
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set(face_counts)
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set(st_values)
        st.BlockIndices()
        stage.GetRootLayer().Save()

        optimizer_inputs: list[Path] = []

        async def passthrough_optimizer(*, input_path, output_path, **_kwargs):
            optimizer_input = Path(input_path)
            optimizer_inputs.append(optimizer_input)
            Path(output_path).write_bytes(optimizer_input.read_bytes())
            optimized_stage = Usd.Stage.Open(str(output_path))
            optimized_stage.ClearDefaultPrim()
            UsdGeom.SetStageUpAxis(optimized_stage, UsdGeom.Tokens.y)
            UsdGeom.SetStageMetersPerUnit(optimized_stage, 0.01)
            optimized_stage.GetRootLayer().customLayerData = {}
            optimized_stage.GetRootLayer().Save()
            return {
                "status": "success",
                "optimization_time": 0.1,
                "operations_executed": ["deinstance"],
            }

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "optimization_config": {
                "backend": "remote",
                "flatten_prototypes": True,
                "scene_optimizer_settings": {
                    "enable_deinstance": True,
                    "enable_split_meshes": False,
                    "enable_deduplicate": False,
                },
            },
        }

        with patch(
            "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
            side_effect=passthrough_optimizer,
        ):
            result = await OptimizeUSDTask().arun(context)

        assert result["optimization_success"] is True
        assert len(optimizer_inputs) == 1
        assert optimizer_inputs[0].parent == output_usd.parent
        assert not optimizer_inputs[0].exists()

        optimized_stage = Usd.Stage.Open(str(output_usd))
        assert optimized_stage.GetDefaultPrim().GetPath() == Sdf.Path("/World")
        assert UsdGeom.GetStageUpAxis(optimized_stage) == UsdGeom.Tokens.z
        assert UsdGeom.GetStageMetersPerUnit(optimized_stage) == 1.0
        assert (
            optimized_stage.GetRootLayer().customLayerData["SimReady_Metadata"]
            == '{"identifier":"test"}'
        )
        optimized_mesh = UsdGeom.Mesh(optimized_stage.GetPrimAtPath("/World/Mesh"))
        optimized_st = UsdGeom.PrimvarsAPI(optimized_mesh).GetPrimvar("st")
        assert optimized_mesh.GetPointsAttr().Get() == points
        assert optimized_mesh.GetFaceVertexCountsAttr().Get() == face_counts
        assert optimized_mesh.GetFaceVertexIndicesAttr().Get() == face_indices
        assert optimized_st.GetInterpolation() == UsdGeom.Tokens.faceVarying
        assert optimized_st.GetAttr().Get() == st_values
        assert optimized_st.GetIndicesAttr().GetResolveInfo().ValueIsBlocked()
        assert len(optimized_st.ComputeFlattened()) == sum(face_counts)

    @pytest.mark.asyncio
    async def test_local_failure_without_nvcf_raises_with_guidance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Local backend missing + no NVCF endpoint should raise, not fall through to NVCF."""
        monkeypatch.delenv("NVCF_OPTIMIZER_FUNCTION_ID", raising=False)
        monkeypatch.delenv("OPTIMIZER_ENDPOINT", raising=False)

        sentinel = "local-backend-context-secret-713"
        context = _make_context(tmp_path, backend="local")
        mock_stage = _mock_usd_open()

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                new_callable=AsyncMock,
            ) as mock_nvcf,
            patch(
                "world_understanding.functions.graphics.scene_optimizer_local.optimize_usd_local",
                side_effect=RuntimeError(
                    "WU_SO_PACKAGE_DIR environment variable is not set: " + sentinel
                ),
            ),
        ):
            task = OptimizeUSDTask()
            with pytest.raises(RuntimeError, match="fetch_build_resources") as exc_info:
                await task.arun(context)

        mock_nvcf.assert_not_called()
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert sentinel not in repr(exc_info.value)
        assert sentinel not in context["optimization_error"]
        assert not _traceback_frame_locals(exc_info.value, "_arun_impl")
        public_frames = _traceback_frame_locals(exc_info.value, "arun")
        assert public_frames
        assert sentinel not in repr(public_frames)

    @pytest.mark.asyncio
    async def test_local_fallback_remote_failure_severs_both_backend_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPTIMIZER_ENDPOINT", "https://optimizer.example.test")
        monkeypatch.delenv("NVCF_OPTIMIZER_FUNCTION_ID", raising=False)
        local_secret = "local-fallback-context-secret-713"
        remote_secret = "remote-fallback-context-secret-713"
        context = _make_context(tmp_path, backend="local")
        mock_stage = _mock_usd_open()

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.functions.graphics.scene_optimizer_local.optimize_usd_local",
                side_effect=RuntimeError(
                    "Scene Optimizer Core package not found: " + local_secret
                ),
            ),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                new_callable=AsyncMock,
                side_effect=RuntimeError("remote backend echoed " + remote_secret),
            ) as mock_nvcf,
        ):
            with pytest.raises(
                RuntimeError, match="^USD optimization failed$"
            ) as exc_info:
                await OptimizeUSDTask().arun(context)

        mock_nvcf.assert_awaited_once()
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        observable = f"{exc_info.value!r}\n{context!r}"
        assert local_secret not in observable
        assert remote_secret not in observable

    @pytest.mark.asyncio
    async def test_flattened_input_is_written_to_output_workspace(
        self, tmp_path: Path
    ) -> None:
        """Prototype flattening must not write temporary files beside read-only inputs."""
        source_dir = tmp_path / "readonly-source"
        source_dir.mkdir()
        input_usd = source_dir / "input.usd"
        input_usd.touch()
        output_dir = tmp_path / "optimizer-workspace"
        output_usd = output_dir / "output.usd"
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "optimization_config": {
                "backend": "local",
                "flatten_prototypes": True,
                "scene_optimizer_settings": {
                    "enable_deinstance": False,
                    "enable_split_meshes": True,
                    "enable_deduplicate": True,
                },
            },
        }
        mock_stage = _mock_usd_open()
        exported_paths: list[Path] = []
        local_inputs: list[Path] = []

        class FakeFlattenedLayer:
            def Export(self, path: str) -> None:
                exported_path = Path(path)
                exported_paths.append(exported_path)
                exported_path.parent.mkdir(parents=True, exist_ok=True)
                exported_path.write_text("#usda 1.0\n", encoding="utf-8")

        local_result: dict[str, object] = {
            "status": "success",
            "optimization_time": 1.0,
            "operations_executed": ["split"],
        }

        async def fake_to_thread(_fn: object, **kwargs: Any) -> dict[str, object]:
            local_inputs.append(Path(str(kwargs["input_path"])))
            Path(str(kwargs["output_path"])).touch()
            return local_result

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.utils.usd.prim.convert_abstract_prototypes_to_def",
                return_value=0,
            ),
            patch(
                "world_understanding.utils.usd.prim.flatten_prototype_references",
                return_value=FakeFlattenedLayer(),
            ),
            patch(
                "world_understanding.functions.graphics.scene_optimizer_local.optimize_usd_local",
                return_value=local_result,
            ),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "_restore_optimized_stage_metadata",
                return_value=({"restored": True}, mock_stage),
            ),
            patch("asyncio.to_thread", side_effect=fake_to_thread),
        ):
            task = OptimizeUSDTask()
            result = await task.arun(context)

        assert result["optimization_success"] is True
        assert len(exported_paths) == 1
        assert exported_paths[0].parent == output_dir
        assert exported_paths[0].name.startswith("_flattened_input_")
        assert exported_paths[0].suffix == ".usd"
        assert local_inputs == exported_paths
        assert not (source_dir / "_flattened_input.usd").exists()

    @pytest.mark.asyncio
    async def test_flatten_cleanup_ignores_missing_temp_file(
        self, tmp_path: Path
    ) -> None:
        context = _make_context(tmp_path, backend="local")
        context["optimization_config"]["flatten_prototypes"] = True
        mock_stage = _mock_usd_open()

        class FakeFlattenedLayer:
            def Export(self, path: str) -> None:
                Path(path).write_text("#usda 1.0\n", encoding="utf-8")

        local_result: dict[str, object] = {
            "status": "success",
            "optimization_time": 1.0,
            "operations_executed": ["split"],
        }

        async def fake_to_thread(_fn: object, **kwargs: Any) -> dict[str, object]:
            Path(str(kwargs["output_path"])).touch()
            return local_result

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.utils.usd.prim.convert_abstract_prototypes_to_def",
                return_value=2,
            ),
            patch(
                "world_understanding.utils.usd.prim.flatten_prototype_references",
                return_value=FakeFlattenedLayer(),
            ),
            patch(
                "world_understanding.functions.graphics.scene_optimizer_local.optimize_usd_local",
                return_value=local_result,
            ),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd."
                "_restore_optimized_stage_metadata",
                return_value=({"restored": True}, mock_stage),
            ),
            patch("asyncio.to_thread", side_effect=fake_to_thread),
            patch.object(Path, "unlink", side_effect=FileNotFoundError),
        ):
            result = await OptimizeUSDTask().arun(context)

        assert result["optimization_success"] is True
        assert result["optimization_metadata"]["prototypes_converted_pre"] == 2
