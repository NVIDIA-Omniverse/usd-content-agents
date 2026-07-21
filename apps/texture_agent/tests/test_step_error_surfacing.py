# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for per-unit error surfacing and failure-rate threshold gating.

Covers the public failure-surfacing gaps:

  * ``_classify_unit_failure`` extracts HTTP status codes from both wrapped
    ``HTTPError`` causes and from the message text used by per-unit
    ``RuntimeError`` wrappers.
  * ``_raise_if_above_threshold`` honours a configurable failure-rate
    threshold (default 1.0 = "raise only when 100% fail").
  * ``GenerateTexturesTask`` stashes structured per-unit errors on the
    pipeline context regardless of whether the threshold gate fires.
  * ``BlendTexturesTask`` surfaces per-unit errors and raises when every
    blend attempt fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest
from PIL import Image

import texture_agent.tasks.blend_textures as blend_textures_task
import texture_agent.tasks.generate_textures as generate_textures_task
from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.functions.texture_generation import GeneratedTextures
from texture_agent.tasks.thresholds import validate_failure_threshold


def _unit(name: str = "Steel") -> PrimTextureUnit:
    material = MaterialInfo(
        prim_path=f"/Root/Looks/{name}",
        name=name,
        bound_prim_paths=[f"/Root/{name}_Mesh"],
        base_color=(0.4, 0.5, 0.6),
        base_metalness=0.3,
        specular_roughness=0.2,
    )
    return PrimTextureUnit(
        prim_path="",
        material_info=material,
        key=name,
        prompt="prompt",
        opacity=0.8,
    )


def _save_png(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> str:
    Image.new("RGB", (8, 8), color).save(path)
    return str(path)


class TestClassifyUnitFailure:
    def test_extracts_status_from_message_when_no_cause(self) -> None:
        exc = RuntimeError(
            "Generation failed for Aluminum_Brushed: HTTP Error 403: Forbidden"
        )
        record = generate_textures_task._classify_unit_failure("Aluminum_Brushed", exc)

        assert record == {
            "material": "Aluminum_Brushed",
            "type": "RuntimeError",
            "status": 403,
            "message": (
                "Generation failed for Aluminum_Brushed: HTTP Error 403: Forbidden"
            ),
        }

    def test_prefers_wrapped_httperror_over_message_scrape(self) -> None:
        cause = HTTPError("http://x", 503, "Service Unavailable", {}, None)
        outer = RuntimeError("wrapper text mentions HTTP 200 but cause is 503")
        outer.__cause__ = cause

        record = generate_textures_task._classify_unit_failure("Steel", outer)

        assert record["type"] == "HTTPError"
        assert record["status"] == 503

    def test_extracts_status_from_httpx_status_error(self) -> None:
        """The service backend's REST polling path raises
        ``httpx.HTTPStatusError``. Its message ("Client error '403
        Forbidden' for url ...") doesn't contain a literal ``HTTP <NNN>``
        substring, so the regex fallback would miss it. Extraction must
        prefer ``response.status_code`` from the cause chain."""
        import httpx

        request = httpx.Request("GET", "http://service/v1/jobs/x")
        response = httpx.Response(403, request=request)
        cause = httpx.HTTPStatusError(
            "Client error '403 Forbidden' for url 'http://service/v1/jobs/x'",
            request=request,
            response=response,
        )
        outer = RuntimeError("polling failed")
        outer.__cause__ = cause

        record = generate_textures_task._classify_unit_failure("Steel", outer)

        assert record["type"] == "HTTPStatusError"
        assert record["status"] == 403

    def test_returns_none_status_when_no_http_indicator(self) -> None:
        record = generate_textures_task._classify_unit_failure(
            "Foo", ValueError("not an http failure")
        )
        assert record == {
            "material": "Foo",
            "type": "ValueError",
            "status": None,
            "message": "not an http failure",
        }


class TestValidateFailureThreshold:
    """Catch typo'd config values that would silently disable the gate.

    ``nan >= 1.0`` and ``0.5 >= 1.1`` both evaluate to ``False`` in Python,
    so an unvalidated threshold of ``"nan"`` / ``1.1`` would let a 100%-fail
    run flow through to ``apply_textures`` -- the original 6126254 silent-
    completed pattern.
    """

    @pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
    def test_rejects_out_of_range_or_non_finite(self, value: float) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            validate_failure_threshold(
                value, config_key="texture_config.failure_threshold"
            )

    @pytest.mark.parametrize("value", ["nan", "abc", None, object()])
    def test_rejects_uncoercible(self, value: object) -> None:
        with pytest.raises(ValueError, match="failure_threshold"):
            validate_failure_threshold(
                value, config_key="texture_config.failure_threshold"
            )

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 1, "0.5"])
    def test_accepts_in_range_values(self, value: object) -> None:
        result = validate_failure_threshold(
            value, config_key="texture_config.failure_threshold"
        )
        assert 0.0 <= result <= 1.0


class TestRaiseIfAboveThreshold:
    def test_raises_on_full_failure_at_default_threshold(self) -> None:
        attempted = [_unit("A"), _unit("B")]
        errors = [
            {"material": "A", "type": "RuntimeError", "status": 403, "message": "x"},
            {"material": "B", "type": "RuntimeError", "status": 403, "message": "y"},
        ]

        with pytest.raises(RuntimeError, match="2/2 texture generation requests"):
            generate_textures_task._raise_if_above_threshold(
                attempted, {}, errors, backend_label="nim", failure_threshold=1.0
            )

    def test_does_not_raise_on_partial_failure_at_default_threshold(self) -> None:
        attempted = [_unit("A"), _unit("B")]
        fresh = {"A": GeneratedTextures(albedo="/tmp/a.png", normal="", orm="")}
        errors = [
            {"material": "B", "type": "RuntimeError", "status": 500, "message": "x"},
        ]

        # Default threshold == 1.0 → 50% < 100% → no raise.
        generate_textures_task._raise_if_above_threshold(
            attempted, fresh, errors, backend_label="nim", failure_threshold=1.0
        )

    def test_raises_on_partial_failure_when_threshold_lowered(self) -> None:
        attempted = [_unit("A"), _unit("B"), _unit("C"), _unit("D")]
        fresh = {"A": GeneratedTextures(albedo="/tmp/a.png", normal="", orm="")}
        errors = [
            {"material": "B", "type": "RuntimeError", "status": 500, "message": "b"},
            {"material": "C", "type": "RuntimeError", "status": 500, "message": "c"},
            {"material": "D", "type": "RuntimeError", "status": 500, "message": "d"},
        ]

        # 75% failure rate ≥ 0.5 threshold → raise.
        with pytest.raises(RuntimeError, match=r"failure rate 75% >= threshold 50%"):
            generate_textures_task._raise_if_above_threshold(
                attempted, fresh, errors, backend_label="nim", failure_threshold=0.5
            )

    def test_counts_duplicate_error_records_once_per_failed_unit(self) -> None:
        attempted = [_unit("A"), _unit("B")]
        fresh = {"A": GeneratedTextures(albedo="/tmp/a.png", normal="", orm="")}
        errors = [
            {"material": "B", "type": "RuntimeError", "status": 500, "message": "b"},
            {"material": "B", "type": "HTTPError", "status": 502, "message": "b2"},
        ]

        generate_textures_task._raise_if_above_threshold(
            attempted, fresh, errors, backend_label="nim", failure_threshold=1.0
        )
        with pytest.raises(RuntimeError, match="1/2 texture generation requests"):
            generate_textures_task._raise_if_above_threshold(
                attempted, fresh, errors, backend_label="nim", failure_threshold=0.5
            )

    def test_no_raise_when_no_attempts(self) -> None:
        # Cached-only run path -- nothing to evaluate.
        generate_textures_task._raise_if_above_threshold(
            [], {}, [], backend_label="nim", failure_threshold=0.0
        )

    def test_raised_message_includes_per_unit_status_and_type(self) -> None:
        attempted = [_unit("A")]
        errors = [
            {
                "material": "A",
                "type": "HTTPError",
                "status": 403,
                "message": "Forbidden",
            }
        ]
        with pytest.raises(RuntimeError) as excinfo:
            generate_textures_task._raise_if_above_threshold(
                attempted, {}, errors, backend_label="nim", failure_threshold=1.0
            )
        assert "[HTTPError 403]" in str(excinfo.value)
        assert "A: [HTTPError 403] Forbidden" in str(excinfo.value)


class TestGenerateTexturesContextSurfacing:
    """``GenerateTexturesTask.run`` must publish per-unit errors on the
    context regardless of the threshold outcome, so the service layer can
    include them in completed-step SSE payloads even on partial-success
    runs."""

    def test_partial_failure_below_threshold_surfaces_errors_on_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        task = generate_textures_task.GenerateTexturesTask()

        ok_albedo = tmp_path / "generated" / "Good_albedo.png"
        ok_albedo.parent.mkdir(parents=True, exist_ok=True)
        _save_png(ok_albedo)

        def fake_simple(self, units, context, out_dir, texture_config):
            generated = {
                "Good": GeneratedTextures(albedo=str(ok_albedo), normal="", orm="")
            }
            errors = [
                {
                    "material": "Bad",
                    "type": "RuntimeError",
                    "status": 403,
                    "message": "HTTP 403",
                }
            ]
            return generated, errors, "fake-engine"

        monkeypatch.setattr(
            generate_textures_task.GenerateTexturesTask,
            "_run_simple_image_gen",
            fake_simple,
        )

        units = [_unit("Good"), _unit("Bad")]
        ctx = task.run(
            {
                "prim_texture_units": units,
                "texture_config": {
                    "backend": "simple_image_gen",
                    "skip_existing": False,
                    # Default threshold (1.0) so partial failure does not raise.
                },
                "working_dir": str(tmp_path),
                "usd_path": "/tmp/in.usd",
            }
        )

        assert "Good" in ctx["generated_textures"]
        assert ctx["generate_textures_failed_count"] == 1
        assert ctx["generate_textures_attempted_count"] == 2
        assert ctx["generate_textures_errors"] == [
            {
                "material": "Bad",
                "type": "RuntimeError",
                "status": 403,
                "message": "HTTP 403",
            }
        ]

    def test_invalid_threshold_fails_before_backend_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo'd ``failure_threshold`` must reject before any backend
        call -- otherwise we waste 8x network round-trips and only THEN
        report a config error to the customer."""
        task = generate_textures_task.GenerateTexturesTask()

        backend_called = []

        def fake_simple(self, units, context, out_dir, texture_config):
            backend_called.append(True)
            return {}, [], "fake-engine"

        monkeypatch.setattr(
            generate_textures_task.GenerateTexturesTask,
            "_run_simple_image_gen",
            fake_simple,
        )

        with pytest.raises(ValueError, match="failure_threshold"):
            task.run(
                {
                    "prim_texture_units": [_unit("A")],
                    "texture_config": {
                        "backend": "simple_image_gen",
                        "skip_existing": False,
                        "failure_threshold": "nan",
                    },
                    "working_dir": str(tmp_path),
                    "usd_path": "/tmp/in.usd",
                }
            )
        assert backend_called == [], (
            "backend was dispatched despite invalid failure_threshold"
        )

    def test_partial_failure_above_threshold_publishes_successes_to_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the threshold gate fires after partial success, the executor
        reads ``context["generated_textures"]`` to build failed-step stats.
        That dict must contain the units that DID succeed -- otherwise the
        failure diagnostics misreport ``textures_generated: 0`` even when
        files were written to disk."""
        task = generate_textures_task.GenerateTexturesTask()

        ok_albedo = tmp_path / "generated" / "Good_albedo.png"
        ok_albedo.parent.mkdir(parents=True, exist_ok=True)
        _save_png(ok_albedo)

        def fake_simple(self, units, context, out_dir, texture_config):
            generated = {
                "Good": GeneratedTextures(albedo=str(ok_albedo), normal="", orm="")
            }
            errors = [
                {
                    "material": "BadA",
                    "type": "RuntimeError",
                    "status": 500,
                    "message": "x",
                },
                {
                    "material": "BadB",
                    "type": "RuntimeError",
                    "status": 500,
                    "message": "y",
                },
                {
                    "material": "BadC",
                    "type": "RuntimeError",
                    "status": 500,
                    "message": "z",
                },
            ]
            return generated, errors, "fake-engine"

        monkeypatch.setattr(
            generate_textures_task.GenerateTexturesTask,
            "_run_simple_image_gen",
            fake_simple,
        )

        # Spy: capture the context as it looked at the moment the raise
        # would land (i.e. after the merge-into-context but before/at the
        # threshold check). We do that by intercepting the threshold
        # helper itself.
        captured: dict[str, Any] = {}
        original = generate_textures_task._raise_if_above_threshold

        def spy(attempted, fresh_generated, errors, **kwargs):
            captured["context_generated"] = task_context["generated_textures"]
            captured["context_errors"] = task_context["generate_textures_errors"]
            return original(attempted, fresh_generated, errors, **kwargs)

        monkeypatch.setattr(generate_textures_task, "_raise_if_above_threshold", spy)

        task_context: dict[str, Any] = {
            "prim_texture_units": [
                _unit("Good"),
                _unit("BadA"),
                _unit("BadB"),
                _unit("BadC"),
            ],
            "texture_config": {
                "backend": "simple_image_gen",
                "skip_existing": False,
                "failure_threshold": 0.5,
            },
            "working_dir": str(tmp_path),
            "usd_path": "/tmp/in.usd",
        }

        with pytest.raises(RuntimeError, match=r"3/4 texture generation requests"):
            task.run(task_context)

        # The successful ``Good`` unit is on context BEFORE the raise so
        # the executor's failed-step stats correctly report 1 textures_
        # generated alongside 3 textures_failed.
        assert "Good" in captured["context_generated"]
        assert len(captured["context_errors"]) == 3

    def test_full_failure_raises_with_structured_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        task = generate_textures_task.GenerateTexturesTask()

        def fake_simple(self, units, context, out_dir, texture_config):
            errors = [
                {
                    "material": u.key,
                    "type": "RuntimeError",
                    "status": 403,
                    "message": f"HTTP 403 for {u.key}",
                }
                for u in units
            ]
            return {}, errors, "fake-engine"

        monkeypatch.setattr(
            generate_textures_task.GenerateTexturesTask,
            "_run_simple_image_gen",
            fake_simple,
        )

        with pytest.raises(RuntimeError, match=r"2/2 texture generation requests"):
            task.run(
                {
                    "prim_texture_units": [_unit("A"), _unit("B")],
                    "texture_config": {
                        "backend": "simple_image_gen",
                        "skip_existing": False,
                    },
                    "working_dir": str(tmp_path),
                    "usd_path": "/tmp/in.usd",
                }
            )


class TestBlendTexturesErrorSurfacing:
    def test_raises_when_every_blend_fails(self, tmp_path: Path) -> None:
        task = blend_textures_task.BlendTexturesTask()
        unit = _unit("Steel")

        with pytest.raises(RuntimeError, match=r"1/1 blend operations failed"):
            task.run(
                {
                    "prim_texture_units": [unit],
                    "generated_textures": {
                        "Steel": GeneratedTextures(
                            albedo="/does/not/exist.png", normal="", orm=""
                        )
                    },
                    "blend_config": {"output_size": 16},
                    "working_dir": str(tmp_path),
                }
            )

    def test_partial_failure_below_threshold_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        task = blend_textures_task.BlendTexturesTask()
        good_albedo = tmp_path / "good_albedo.png"
        _save_png(good_albedo)

        ctx = task.run(
            {
                "prim_texture_units": [_unit("Good"), _unit("Bad")],
                "generated_textures": {
                    "Good": GeneratedTextures(
                        albedo=str(good_albedo), normal="", orm=""
                    ),
                    "Bad": GeneratedTextures(
                        albedo="/does/not/exist.png", normal="", orm=""
                    ),
                },
                "blend_config": {"output_size": 16},
                "working_dir": str(tmp_path),
            }
        )

        assert "Good" in ctx["blended_textures"]
        assert ctx["blend_textures_failed_count"] == 1
        assert ctx["blend_textures_attempted_count"] == 2
        assert ctx["blend_textures_errors"][0]["material"] == "Bad"
        assert ctx["blend_textures_errors"][0]["type"] == "MissingAlbedo"

    def test_corrupt_albedo_propagates_instead_of_being_swallowed(
        self, tmp_path: Path
    ) -> None:
        """Hard exceptions inside the blend ops (e.g. PIL refusing to open
        a corrupt PNG) must propagate. Without this, a single corrupt
        input would silently drop the material at the default threshold
        of 1.0 and let apply_textures emit a USD missing materials --
        defeating the customer's "the pipeline should fail loudly when
        something goes wrong" expectation."""
        task = blend_textures_task.BlendTexturesTask()
        # Write garbage at the albedo path so PIL.Image.open raises
        # UnidentifiedImageError when blend tries to load it.
        bad_albedo = tmp_path / "Steel_albedo.png"
        bad_albedo.write_bytes(b"\x00not-a-png\x00")

        from PIL import UnidentifiedImageError

        with pytest.raises(UnidentifiedImageError):
            task.run(
                {
                    "prim_texture_units": [_unit("Steel")],
                    "generated_textures": {
                        "Steel": GeneratedTextures(
                            albedo=str(bad_albedo), normal="", orm=""
                        )
                    },
                    "blend_config": {"output_size": 16},
                    "working_dir": str(tmp_path),
                }
            )

    def test_partial_failure_above_lowered_threshold_raises(
        self, tmp_path: Path
    ) -> None:
        """`failure_threshold: 0.5` must trip even when one blend succeeded.

        Guards against the original `not blended` short-circuit that ignored
        configured thresholds the moment any single unit succeeded.
        """
        task = blend_textures_task.BlendTexturesTask()
        good_albedo = tmp_path / "good_albedo.png"
        _save_png(good_albedo)

        with pytest.raises(RuntimeError, match=r"3/4 blend operations failed"):
            task.run(
                {
                    "prim_texture_units": [
                        _unit("Good"),
                        _unit("BadA"),
                        _unit("BadB"),
                        _unit("BadC"),
                    ],
                    "generated_textures": {
                        "Good": GeneratedTextures(
                            albedo=str(good_albedo), normal="", orm=""
                        ),
                        "BadA": GeneratedTextures(
                            albedo="/does/not/exist_a.png", normal="", orm=""
                        ),
                        "BadB": GeneratedTextures(
                            albedo="/does/not/exist_b.png", normal="", orm=""
                        ),
                        "BadC": GeneratedTextures(
                            albedo="/does/not/exist_c.png", normal="", orm=""
                        ),
                    },
                    "blend_config": {"output_size": 16, "failure_threshold": 0.5},
                    "working_dir": str(tmp_path),
                }
            )
