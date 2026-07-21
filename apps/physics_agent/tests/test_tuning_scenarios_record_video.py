# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune scenarios honor ``record_video`` independently of ``vlm_check``.

These tests exercise the render branch in
``physics_agent.tuning.scenarios.{drop_settle,freeform}.evaluate`` without
GPU. The render driver is monkey-patched so we can assert (a) it is
invoked when ``record_video`` opts in even with ``vlm_check`` off, and
(b) it is NOT invoked when both are off, and (c) the VLM judge is not
called when ``vlm_check`` is off.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest


def _make_scenario(
    name: str,
    *,
    target: dict[str, Any],
    metric: str = "settle_distance",
) -> Any:
    """Build a minimal valid Scenario (one tunable param)."""
    from physics_agent.tuning.types import Scenario, TunableParam

    return Scenario(
        name=name,
        params=(TunableParam(name="mass_scale", min_value=0.5, max_value=2.0),),
        target=target,
        metric=metric,
    )


def _install_fake_render_driver(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install a stub ``render_time_sampled_usd`` and capture call args.

    Patches the symbol directly on the real graphics package so the
    scenario's ``from world_understanding.functions.graphics import
    render_time_sampled_usd`` resolves to our stub. We don't replace
    the package itself — that would break unrelated submodule imports.
    """
    import world_understanding.functions.graphics as graphics_pkg

    calls: dict[str, Any] = {"count": 0, "args": []}

    def fake_render(usd_path, output_dir, **kwargs):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        calls["args"].append({"usd_path": usd_path, "output_dir": output_dir, **kwargs})
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        frame = Path(output_dir) / "frame_0000.png"
        frame.write_bytes(b"fake-png")
        return [frame]

    monkeypatch.setattr(
        graphics_pkg, "render_time_sampled_usd", fake_render, raising=False
    )
    return calls


def _drop_settle_target(*, vlm_check: str, record_video: str) -> dict[str, Any]:
    return {
        "duration_s": 0.1,
        "drop_height_m": 0.05,
        "vlm_check": vlm_check,
        "record_video": record_video,
        "sample_fps": 30,
        "cameras": ["+x+y+z"],
    }


@pytest.mark.parametrize(
    ("target", "expected"),
    (
        ({}, "ovrtx"),
        ({"video_renderer": None, "vlm_renderer": None}, "ovrtx"),
        ({"video_renderer": None, "vlm_renderer": "mock"}, "mock"),
        ({"video_renderer": "warp", "vlm_renderer": "mock"}, "warp"),
    ),
)
def test_resolve_video_renderer_uses_none_only_fallback(
    target: dict[str, Any], expected: str
) -> None:
    from physics_agent.tuning.video_rendering import resolve_video_renderer

    assert resolve_video_renderer(target) == expected


@pytest.mark.parametrize("invalid", ("", False, 0, [], " ", "nvcf", "typo"))
def test_resolve_video_renderer_rejects_explicit_invalid_values(invalid: Any) -> None:
    from physics_agent.tuning.video_rendering import resolve_video_renderer

    with pytest.raises(ValueError, match="Unknown rendering backend"):
        resolve_video_renderer({"video_renderer": invalid, "vlm_renderer": "mock"})


@pytest.mark.parametrize("scenario_name", ("drop_settle", "freeform"))
def test_invalid_active_video_renderer_fails_before_trial_side_effects(
    scenario_name: str,
    tmp_path: Path,
) -> None:
    physics_usd = tmp_path / f"{scenario_name}.usda"
    physics_usd.write_bytes(b"")
    work_dir = tmp_path / f"{scenario_name}_work"

    if scenario_name == "drop_settle":
        from physics_agent.tuning.scenarios.drop_settle import evaluate

        target = {
            **_drop_settle_target(vlm_check="off", record_video="always"),
            "video_renderer": "",
        }
        metric = "settle_distance"
        evaluator_kwargs: dict[str, Any] = {"final_state_judge": None}
    else:
        from physics_agent.tuning.scenarios.freeform import evaluate

        target = {
            "description": "spin",
            "record_video": "always",
            "video_renderer": "",
        }
        metric = "combined"
        evaluator_kwargs = {"judge_callback": None}

    scenario = _make_scenario(scenario_name, target=target, metric=metric)
    with pytest.raises(ValueError, match="Unknown rendering backend"):
        evaluate(
            params={},
            scenario=scenario,
            physics_usd=physics_usd,
            seed=99,
            simulator=_FakeDaemon(),  # type: ignore[arg-type]
            work_dir=work_dir,
            **evaluator_kwargs,
        )

    assert not work_dir.exists()


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return _install_fake_render_driver(monkeypatch)


class _FakeDaemon:
    """Minimal daemon stub that returns a flat trajectory."""

    def evaluate(self, **kwargs):  # type: ignore[no-untyped-def]
        # Three samples at 0.0/0.05/0.1 seconds, all at rest position.
        trajectory = [
            (
                t,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            for t in (0.0, 0.05, 0.1)
        ]
        return {"trajectory": trajectory, "final_pose": trajectory[-1][1]}


def _stub_drop_settle_supports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stub the side-effecting helpers drop_settle.evaluate calls.

    drop_settle.evaluate imports its dependencies lazily inside the
    function body, so we have to patch the source modules — not the
    drop_settle module's own namespace.
    """
    from world_understanding.functions.physics import (
        trajectory as trajectory_mod,
    )

    from physics_agent import recording as recording_pkg
    from physics_agent.tuning import usd_patch as usd_patch_mod
    from physics_agent.tuning.scenarios import _scene_builder as scene_builder_mod

    monkeypatch.setattr(
        usd_patch_mod,
        "patch_physics_usd",
        lambda src, dst, params: Path(dst).write_bytes(b""),
    )

    def fake_build_scene(_src, dst, **kwargs):  # type: ignore[no-untyped-def]
        Path(dst).write_bytes(b"")
        return {
            "body_pattern": "/Body",
            "body_prim_path": "/Body",
            "rest_position": [0.0, 0.0, 0.0],
            "drop_height_m_resolved": 0.05,
            "bbox_size_m": 0.1,
            "camera_paths": [],
        }

    monkeypatch.setattr(scene_builder_mod, "build_drop_settle_scene", fake_build_scene)
    monkeypatch.setattr(
        recording_pkg,
        "author_trajectory_usda",
        lambda scene, traj, body, out, fps, **kwargs: Path(out).write_bytes(b""),
    )
    monkeypatch.setattr(
        trajectory_mod, "settle_distance", lambda traj, rest_position: 0.0
    )

    physics_usd = tmp_path / "physics.usda"
    physics_usd.write_bytes(b"")
    return physics_usd


class TestDropSettleRecordVideo:
    def test_record_video_off_and_vlm_off_skips_render(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_render: dict[str, Any],
    ) -> None:
        from physics_agent.tuning.scenarios.drop_settle import evaluate

        physics_usd = _stub_drop_settle_supports(monkeypatch, tmp_path)
        scenario = _make_scenario(
            "drop_settle",
            target=_drop_settle_target(vlm_check="off", record_video="off"),
        )

        evaluate(
            params={},
            scenario=scenario,
            physics_usd=physics_usd,
            seed=0,
            simulator=_FakeDaemon(),  # type: ignore[arg-type]
            work_dir=tmp_path / "work",
        )
        assert fake_render["count"] == 0

    def test_record_video_always_triggers_render_without_vlm(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_render: dict[str, Any],
    ) -> None:
        from physics_agent.tuning.scenarios.drop_settle import evaluate

        physics_usd = _stub_drop_settle_supports(monkeypatch, tmp_path)
        scenario = _make_scenario(
            "drop_settle",
            target={
                **_drop_settle_target(vlm_check="off", record_video="always"),
                "video_renderer": "mock",
            },
        )

        # Pass final_state_judge=None to prove VLM is not invoked.
        out = evaluate(
            params={},
            scenario=scenario,
            physics_usd=physics_usd,
            seed=1,
            simulator=_FakeDaemon(),  # type: ignore[arg-type]
            work_dir=tmp_path / "work",
            final_state_judge=None,
        )
        assert fake_render["count"] == 1
        assert fake_render["args"][0]["renderer"] == "mock"
        # Result dict carries the video block; no vlm_check block.
        assert out.get("record_video", {}).get("status") == "ok"
        assert "vlm_check" not in out

    def test_vlm_check_alone_still_triggers_render(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_render: dict[str, Any],
    ) -> None:
        from physics_agent.tuning.scenarios.drop_settle import evaluate

        physics_usd = _stub_drop_settle_supports(monkeypatch, tmp_path)
        scenario = _make_scenario(
            "drop_settle",
            target=_drop_settle_target(vlm_check="always", record_video="off"),
        )

        judge_calls = []

        def fake_judge(frames, prompt, observations):  # type: ignore[no-untyped-def]
            judge_calls.append((frames, prompt, observations))
            return {"score": 0.9, "reasoning": "ok"}

        out = evaluate(
            params={},
            scenario=scenario,
            physics_usd=physics_usd,
            seed=2,
            simulator=_FakeDaemon(),  # type: ignore[arg-type]
            work_dir=tmp_path / "work",
            final_state_judge=fake_judge,
        )
        assert fake_render["count"] == 1
        assert len(judge_calls) == 1
        assert "vlm_check" in out
        # No record_video block when record_video is off.
        assert "record_video" not in out


class TestFreeformRecordVideo:
    def _stub_freeform_supports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> Path:
        from world_understanding.functions.physics import (
            trajectory as trajectory_mod,
        )

        from physics_agent import recording as recording_pkg
        from physics_agent.tuning import usd_patch as usd_patch_mod
        from physics_agent.tuning.scenarios import _scene_builder as scene_builder_mod

        monkeypatch.setattr(
            usd_patch_mod,
            "patch_physics_usd",
            lambda src, dst, params: Path(dst).write_bytes(b""),
        )

        def fake_build(_src, dst, *, target):  # type: ignore[no-untyped-def]
            Path(dst).write_bytes(b"")
            return {
                "body_pattern": "/Body",
                "body_prim_path": "/Body",
                "camera_paths": [],
            }

        monkeypatch.setattr(scene_builder_mod, "build_freeform_scene", fake_build)
        monkeypatch.setattr(
            recording_pkg,
            "author_trajectory_usda",
            lambda scene, traj, body, out, fps, **kwargs: Path(out).write_bytes(b""),
        )
        # Round 12 (CX P2#4): freeform now passes ``world_up`` through to
        # ``trajectory_summary`` so the stub must accept the kwarg.
        monkeypatch.setattr(
            trajectory_mod,
            "trajectory_summary",
            lambda traj, *, world_up=None: {
                "final_position": [0.0, 0.0, 0.0],
                "fell_over": False,
                "settle_time_s": 0.05,
                "duration_s": 0.1,
                "n_samples": 3,
            },
        )
        physics_usd = tmp_path / "physics.usda"
        physics_usd.write_bytes(b"")
        return physics_usd

    def test_record_video_only_renders_without_judge_callback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_render: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from physics_agent.tuning.scenarios.freeform import evaluate

        physics_usd = self._stub_freeform_supports(monkeypatch, tmp_path)
        scenario = _make_scenario(
            "freeform",
            target={
                "description": "spin a top",
                "duration_s": 0.1,
                "sample_fps": 30,
                "observations": ["stayed upright"],
                "record_video": "always",
                "vlm_renderer": None,
            },
            metric="combined",
        )

        with caplog.at_level(logging.INFO):
            out = evaluate(
                params={},
                scenario=scenario,
                physics_usd=physics_usd,
                seed=3,
                simulator=_FakeDaemon(),  # type: ignore[arg-type]
                work_dir=tmp_path / "work",
                judge_callback=None,  # No VLM judge wired up.
            )
        assert fake_render["count"] == 1
        assert fake_render["args"][0]["renderer"] == "ovrtx"
        assert out.get("record_video", {}).get("status") == "ok"
        assert out["vlm_score"] is None

    def test_record_video_with_judge_callback_scores_frames(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_render: dict[str, Any],
    ) -> None:
        from physics_agent.tuning.scenarios.freeform import evaluate

        physics_usd = self._stub_freeform_supports(monkeypatch, tmp_path)
        scenario = _make_scenario(
            "freeform",
            target={
                "description": "spin a top",
                "duration_s": 0.1,
                "sample_fps": 30,
                "observations": ["stayed upright"],
                "record_video": "always",
            },
            metric="combined",
        )
        judge_calls: list[tuple[list[Path], str | None, list[str]]] = []

        def fake_judge(
            frames: list[Path],
            prompt: str | None,
            observations: list[str],
        ) -> dict[str, Any]:
            judge_calls.append((frames, prompt, observations))
            return {"score": 0.75, "reasoning": "stable"}

        out = evaluate(
            params={},
            scenario=scenario,
            physics_usd=physics_usd,
            seed=4,
            simulator=_FakeDaemon(),  # type: ignore[arg-type]
            work_dir=tmp_path / "work",
            judge_callback=fake_judge,
        )

        assert fake_render["count"] == 1
        assert len(judge_calls) == 1
        assert out["vlm_score"] == 0.75
        assert out["reasoning"] == "stable"
        assert out["weights_used"]["vlm"] > 0
