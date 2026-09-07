# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the new ``user_prompt`` / ``enable_judge`` /
``judge_max_iterations`` form fields on POST /tune.

These tests focus on the request-validation surface and on the
session-metadata + on-disk persistence the router performs synchronously
during ``create_tune`` — the background tune executor itself is replaced
with a no-op stub so no real tune is queued.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


def _scenario_yaml() -> str:
    """Minimal valid drop_settle scenario YAML."""
    return """
name: drop_settle
parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
"""


def _multipart_files(usd_bytes: bytes = b"#usda 1.0\n# fake physics usd\n"):
    return [
        ("physics_usd", ("physics.usda", usd_bytes, "application/octet-stream")),
    ]


@pytest.fixture(autouse=True)
def _stub_tune_executor(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace ``execute_tune_async`` with a no-op coroutine.

    These tests assert on the router-side behaviour of POST /tune (validation,
    session-metadata config, files written under ``session_dir/input/``).
    None of them need the tune to actually run, so we install a stub that
    captures its kwargs and returns immediately.
    """
    captured: dict = {"kwargs": None}

    async def fake_execute_tune_async(**kwargs) -> None:
        captured["kwargs"] = kwargs
        # Yield once so the registered task gets a chance to run cleanly.
        await asyncio.sleep(0)

    from ...service.routers import tune_router as router_module

    monkeypatch.setattr(
        router_module,
        "execute_tune_async",
        fake_execute_tune_async,
        raising=False,
    )

    # The router does a lazy `from ..workers.tune_executor import
    # execute_tune_async` inside create_tune, so override the module the
    # import resolves to under both possible package paths.
    fake_module = type(sys)("tune_executor_stub")
    fake_module.execute_tune_async = fake_execute_tune_async
    monkeypatch.setitem(
        sys.modules,
        "service.workers.tune_executor",
        fake_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "apps.physics_agent_service.service.workers.tune_executor",
        fake_module,
    )

    return captured


async def _post_tune(client, *, files=None, data=None):
    """Convenience wrapper for POST /tune with multipart form."""
    return await client.post(
        "/tune",
        files=files if files is not None else _multipart_files(),
        data=data or {},
    )


@pytest.mark.api
async def test_post_tune_with_scenario_only_still_works(client) -> None:
    """Existing behaviour: scenario_yaml alone, no user_prompt -> 202."""
    r = await _post_tune(client, data={"scenario_yaml": _scenario_yaml()})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "pending"
    assert "session_id" in body


@pytest.mark.api
async def test_post_tune_with_user_prompt_only_returns_202(client) -> None:
    """user_prompt alone, no scenario_yaml -> 202; user_prompt.txt written;
    scenario.yaml is NOT written."""
    from ...service.routers import tune_router

    r = await _post_tune(client, data={"user_prompt": "make it bouncy"})
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]

    manager = tune_router.get_session_manager()
    session_dir = manager.get_session_dir(sid)
    user_prompt_path = session_dir / "input" / "user_prompt.txt"
    scenario_path = session_dir / "input" / "scenario.yaml"

    assert user_prompt_path.exists()
    assert user_prompt_path.read_text(encoding="utf-8") == "make it bouncy"
    assert not scenario_path.exists()


@pytest.mark.api
async def test_post_tune_with_both_user_prompt_and_scenario_returns_202(
    client,
) -> None:
    """Both user_prompt and scenario_yaml supplied -> both files written."""
    from ...service.routers import tune_router

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "user_prompt": "tighten the bounce",
        },
    )
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]

    manager = tune_router.get_session_manager()
    session_dir = manager.get_session_dir(sid)
    scenario_path = session_dir / "input" / "scenario.yaml"
    user_prompt_path = session_dir / "input" / "user_prompt.txt"

    assert scenario_path.exists()
    assert "drop_settle" in scenario_path.read_text(encoding="utf-8")
    assert user_prompt_path.exists()
    assert user_prompt_path.read_text(encoding="utf-8") == "tighten the bounce"


@pytest.mark.api
async def test_post_tune_reference_images_persisted(
    client,
    _stub_tune_executor,
) -> None:
    """Reference image uploads are copied and passed to the tune executor."""
    from ...service.routers import tune_router

    files = _multipart_files()
    files.append(("reference_images", ("ref.png", b"fake image", "image/png")))
    data = {
        "scenario_yaml": _scenario_yaml(),
        "reference_descriptions": json.dumps(["target visual"]),
        "judge_max_tokens": "1234",
        "judge_temperature": "0.25",
    }

    r = await _post_tune(client, files=files, data=data)
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]

    manager = tune_router.get_session_manager()
    meta = await manager.get_session_metadata(sid)
    assert meta is not None
    config = meta["config"]
    assert len(config["reference_images"]) == 1
    assert config["reference_descriptions"] == ["target visual"]
    assert config["judge_max_tokens"] == 1234
    assert config["judge_temperature"] == 0.25
    reference_path = Path(config["reference_images"][0])
    assert reference_path.exists()
    assert reference_path.read_bytes() == b"fake image"

    await asyncio.sleep(0.05)
    kwargs = _stub_tune_executor["kwargs"]
    assert kwargs is not None
    assert kwargs["reference_images"] == [reference_path]
    assert kwargs["reference_descriptions"] == ["target visual"]
    assert kwargs["judge_max_tokens"] == 1234
    assert kwargs["judge_temperature"] == 0.25


@pytest.mark.api
async def test_post_tune_rejects_too_many_reference_images(client) -> None:
    files = _multipart_files()
    for idx in range(17):
        files.append(
            (
                "reference_images",
                (f"ref_{idx}.png", b"fake image", "image/png"),
            )
        )

    r = await _post_tune(
        client,
        files=files,
        data={"scenario_yaml": _scenario_yaml()},
    )

    assert r.status_code == 400, r.text
    assert "Too many reference images" in r.json()["detail"]


@pytest.mark.api
async def test_post_tune_reference_description_size_limit(client) -> None:
    files = _multipart_files()
    files.append(("reference_images", ("ref.png", b"fake image", "image/png")))
    data = {
        "scenario_yaml": _scenario_yaml(),
        "reference_descriptions": json.dumps(["x" * (2 * 1024 + 1)]),
    }

    r = await _post_tune(client, files=files, data=data)

    assert r.status_code == 413, r.text
    assert "reference_descriptions[1]" in r.json()["detail"]


@pytest.mark.api
async def test_post_tune_rejects_removed_video_inputs(client) -> None:
    legacy_field = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "reference_video_frames": "8",
        },
    )
    assert legacy_field.status_code == 400, legacy_field.text
    assert "Video inputs are unsupported" in legacy_field.json()["detail"]
    assert "reference_video_frames" in legacy_field.json()["detail"]

    files = _multipart_files()
    files.append(("reference_images", ("motion.mp4", b"video", "video/mp4")))
    video_upload = await _post_tune(
        client,
        files=files,
        data={"scenario_yaml": _scenario_yaml()},
    )
    assert video_upload.status_code == 400, video_upload.text
    assert "video uploads: motion.mp4" in video_upload.json()["detail"]


@pytest.mark.api
async def test_post_tune_reference_copy_failure_deletes_session(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ...service.routers import tune_router

    manager = tune_router.get_session_manager()
    before = set(await manager.list_sessions())
    original_stream_copy = tune_router._stream_copy

    async def fail_reference_copy(upload, dest, *args, **kwargs):
        if "reference_images" in str(dest):
            raise OSError("disk full")
        return await original_stream_copy(upload, dest, *args, **kwargs)

    monkeypatch.setattr(tune_router, "_stream_copy", fail_reference_copy)
    files = _multipart_files()
    files.append(("reference_images", ("ref.png", b"fake image", "image/png")))

    r = await _post_tune(
        client,
        files=files,
        data={"scenario_yaml": _scenario_yaml()},
    )

    assert r.status_code == 500, r.text
    assert "Failed to copy reference media" in r.json()["detail"]
    assert set(await manager.list_sessions()) == before


@pytest.mark.api
async def test_post_tune_with_partial_scenario_yaml_and_user_prompt_returns_202(
    client,
) -> None:
    """Round 12 (CX P2#1): a partial ``scenario_yaml`` (e.g. an override
    that pins ``parameters`` and lets the interpreter author the rest)
    must be accepted when combined with ``user_prompt``. The full
    ``load_scenario`` validator only runs in YAML-only mode.
    """
    partial_yaml = (
        "name: drop_settle\n"
        "parameters:\n"
        "  - name: mass_scale\n"
        "    min: 0.5\n"
        "    max: 2.0\n"
    )
    r = await _post_tune(
        client,
        data={
            "scenario_yaml": partial_yaml,
            "user_prompt": "make it bouncy and override mass_scale",
        },
    )
    assert r.status_code == 202, r.text


@pytest.mark.api
async def test_post_tune_rejects_mixed_friction_override_before_session(
    client,
) -> None:
    from ...service.routers import tune_router

    manager = tune_router.get_session_manager()
    before = set(await manager.list_sessions())
    partial_yaml = (
        "parameters:\n"
        "  - name: static_friction\n"
        "    min: 0.01\n"
        "    max: 0.02\n"
        "  - name: dynamic_friction\n"
    )

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": partial_yaml,
            "user_prompt": "make it realistic",
        },
    )

    assert r.status_code == 400, r.text
    assert "must both use automatic bounds" in r.json()["detail"]
    assert set(await manager.list_sessions()) == before


@pytest.mark.api
async def test_post_tune_rejects_newton_unsupported_yaml_param_before_session(
    client,
) -> None:
    from ...service.routers import tune_router

    manager = tune_router.get_session_manager()
    before = set(await manager.list_sessions())
    scenario_yaml = """
name: drop_settle
metric: max_bounce_height
target:
  drop_height_m: 0.5
  duration_s: 1.0
parameters:
  - name: restitution
    min: 0.0
    max: 1.0
"""

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": scenario_yaml,
            "engine": "newton",
        },
    )

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "restitution" in detail
    assert set(await manager.list_sessions()) == before


@pytest.mark.api
async def test_post_tune_rejects_newton_unsupported_override_param_before_session(
    client,
) -> None:
    from ...service.routers import tune_router

    manager = tune_router.get_session_manager()
    before = set(await manager.list_sessions())
    partial_yaml = """
parameters:
  - name: static_friction
    min: 0.05
    max: 1.0
"""

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": partial_yaml,
            "user_prompt": "make the object stick hard",
            "engine": "newton",
        },
    )

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "static_friction" in detail
    assert set(await manager.list_sessions()) == before


@pytest.mark.api
async def test_post_tune_rejects_ovphysx_newton_contact_param_before_session(
    client,
) -> None:
    from ...service.routers import tune_router

    manager = tune_router.get_session_manager()
    before = set(await manager.list_sessions())
    scenario_yaml = """
name: drop_settle
metric: settle_distance
target:
  drop_height_m: 0.5
  duration_s: 1.0
parameters:
  - name: contact_ke
    min: 100.0
    max: 100000.0
"""

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": scenario_yaml,
            "engine": "ovphysx",
        },
    )

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "contact_ke" in detail
    assert "ovphysx" in detail
    assert set(await manager.list_sessions()) == before


@pytest.mark.api
async def test_post_tune_accepts_supported_newton_param_after_request_gate(
    client,
) -> None:
    scenario_yaml = """
name: drop_settle
metric: settle_distance
target:
  drop_height_m: 0.5
  duration_s: 1.0
parameters:
  - name: dynamic_friction
    min: 0.05
    max: 1.0
"""

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": scenario_yaml,
            "engine": "newton",
        },
    )

    assert r.status_code == 202, r.text


@pytest.mark.api
async def test_post_tune_rejects_unknown_engine_before_session(
    client,
) -> None:
    from ...service.routers import tune_router

    manager = tune_router.get_session_manager()
    before = set(await manager.list_sessions())

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "engine": "mujoco",
        },
    )

    assert r.status_code == 400, r.text
    assert "Unknown engine" in r.json()["detail"]
    assert set(await manager.list_sessions()) == before


@pytest.mark.api
async def test_post_tune_with_unknown_scenario_name_and_user_prompt_returns_400(
    client,
) -> None:
    """Round 12 (CX P2#1): even in user_prompt+override mode we still
    reject an unknown ``name`` up-front so the worker doesn't waste an
    LLM call on a guaranteed-bad scenario kind.
    """
    r = await _post_tune(
        client,
        data={
            "scenario_yaml": "name: not_a_real_kind\n",
            "user_prompt": "make it bouncy",
        },
    )
    assert r.status_code == 400, r.text
    assert "not_a_real_kind" in r.json()["detail"]


@pytest.mark.api
async def test_post_tune_with_neither_returns_400(client) -> None:
    """Empty scenario_yaml + empty user_prompt -> 400."""
    r = await _post_tune(
        client,
        data={"scenario_yaml": "", "user_prompt": ""},
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "scenario_yaml" in detail
    assert "user_prompt" in detail


@pytest.mark.api
async def test_post_tune_user_prompt_size_limit(client) -> None:
    """user_prompt of 16 KB + 1 byte -> 413 with size-limit detail."""
    oversize = "a" * (16 * 1024 + 1)
    r = await _post_tune(client, data={"user_prompt": oversize})
    assert r.status_code == 413, r.text
    detail = r.json()["detail"]
    assert "user_prompt" in detail
    assert "16" in detail  # mentions the 16 KB limit


@pytest.mark.api
async def test_post_tune_judge_max_iterations_too_low(client) -> None:
    """judge_max_iterations=0 -> 400."""
    r = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "judge_max_iterations": "0",
        },
    )
    assert r.status_code == 400, r.text
    assert "judge_max_iterations" in r.json()["detail"]


@pytest.mark.api
async def test_post_tune_judge_max_iterations_too_high(client) -> None:
    """judge_max_iterations=11 -> 400."""
    r = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "judge_max_iterations": "11",
        },
    )
    assert r.status_code == 400, r.text
    assert "judge_max_iterations" in r.json()["detail"]


@pytest.mark.api
async def test_post_tune_judge_max_tokens_too_low(client) -> None:
    """judge_max_tokens=0 -> 400."""
    r = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "judge_max_tokens": "0",
        },
    )
    assert r.status_code == 400, r.text
    assert "judge_max_tokens" in r.json()["detail"]


@pytest.mark.api
async def test_post_tune_judge_temperature_too_low(client) -> None:
    """judge_temperature=-0.1 -> 400."""
    r = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "judge_temperature": "-0.1",
        },
    )
    assert r.status_code == 400, r.text
    assert "judge_temperature" in r.json()["detail"]


@pytest.mark.api
async def test_post_tune_enable_judge_false_persisted(client) -> None:
    """enable_judge=false form value -> session config has enable_judge=False."""
    from ...service.routers import tune_router

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "enable_judge": "false",
        },
    )
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]

    manager = tune_router.get_session_manager()
    meta = await manager.get_session_metadata(sid)
    assert meta is not None
    config = meta["config"]
    assert config["enable_judge"] is False


@pytest.mark.api
async def test_post_tune_enable_judge_defaults_to_true(client) -> None:
    """Omitting enable_judge -> session config has enable_judge=True."""
    from ...service.routers import tune_router

    r = await _post_tune(client, data={"scenario_yaml": _scenario_yaml()})
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]

    manager = tune_router.get_session_manager()
    meta = await manager.get_session_metadata(sid)
    assert meta is not None
    config = meta["config"]
    assert config["enable_judge"] is True


@pytest.mark.api
async def test_post_tune_user_prompt_persisted_in_session_metadata(
    client,
) -> None:
    """user_prompt='X' -> session config dict has user_prompt='X'."""
    from ...service.routers import tune_router

    r = await _post_tune(
        client,
        data={
            "scenario_yaml": _scenario_yaml(),
            "user_prompt": "X",
        },
    )
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]

    manager = tune_router.get_session_manager()
    meta = await manager.get_session_metadata(sid)
    assert meta is not None
    config = meta["config"]
    assert config["user_prompt"] == "X"
    # user_prompt_path is the absolute path to the on-disk file.
    assert config["user_prompt_path"] is not None
    assert config["user_prompt_path"].endswith("user_prompt.txt")
    # Round 11 thread #4: lock down the absolute-path contract so a
    # regression to a relative path doesn't slip past the suffix check.
    from pathlib import Path as _Path

    assert _Path(config["user_prompt_path"]).is_absolute()


@pytest.mark.api
async def test_post_tune_invalid_scenario_yaml_with_user_prompt_returns_400(
    client,
) -> None:
    """Even with user_prompt, an explicit scenario_yaml must validate."""
    r = await _post_tune(
        client,
        data={
            "scenario_yaml": "name: not_a_real_kind\nparameters: []\n",
            "user_prompt": "make it bouncy",
        },
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    # The router prefixes scenario-validation failures with "Invalid scenario".
    assert "Invalid scenario" in detail


@pytest.mark.api
async def test_post_tune_blank_only_whitespace_user_prompt_treated_as_empty(
    client,
) -> None:
    """user_prompt='   ' is treated as empty; with no scenario_yaml -> 400."""
    r = await _post_tune(
        client,
        data={"scenario_yaml": "", "user_prompt": "   "},
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "scenario_yaml" in detail
    assert "user_prompt" in detail
