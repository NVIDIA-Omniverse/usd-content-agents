# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the isolated OvPhysX daemon script."""

from __future__ import annotations

import builtins
import importlib
import io
import json
import os
import sys
import types
from typing import Any

import numpy as np
import pytest


def _load_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(os, "dup", lambda _fd: 99)
    monkeypatch.setattr(os, "dup2", lambda _src, _dst: None)
    sys.modules.pop(
        "world_understanding.functions.physics._ovphysx_daemon_script", None
    )
    return importlib.import_module(
        "world_understanding.functions.physics._ovphysx_daemon_script"
    )


class FakeBinding:
    def __init__(
        self, shape: tuple[int, ...], fill: float, *, destroy_raises: bool = False
    ):
        self.shape = shape
        self.fill = fill
        self.destroy_raises = destroy_raises
        self.destroyed = False
        self.writes: list[np.ndarray] = []

    def write(self, value: np.ndarray) -> None:
        self.writes.append(value.copy())

    def read(self, buffer: np.ndarray) -> None:
        buffer[:] = self.fill
        self.fill += 1.0

    def destroy(self) -> None:
        self.destroyed = True
        if self.destroy_raises:
            raise RuntimeError("destroy failed")


class FakePhysX:
    def __init__(self, *, device: str):
        self.device = device
        self.removed: list[int] = []
        self.steps: list[tuple[int, float, float]] = []
        self.released = False
        self.pose_binding = FakeBinding((1, 7), 1.0)
        self.velocity_binding = FakeBinding((1, 6), 10.0)

    def add_usd(self, scene_usd: str) -> tuple[int, None]:
        self.scene_usd = scene_usd
        return 42, None

    def create_tensor_binding(self, *, pattern: str, tensor_type: str) -> FakeBinding:
        self.pattern = pattern
        if tensor_type == "pose":
            return self.pose_binding
        return self.velocity_binding

    def step_n_sync(self, chunk: int, dt: float, current_time: float) -> None:
        self.steps.append((chunk, dt, current_time))

    def remove_usd(self, handle: int) -> None:
        self.removed.append(handle)

    def release(self) -> None:
        self.released = True


def _request(**overrides: Any) -> dict[str, Any]:
    req: dict[str, Any] = {
        "scene_usd": "scene.usda",
        "body_pattern": "/World/Body",
        "duration_s": 0.1,
        "dt": 0.05,
        "sample_fps": 10,
        "initial_linear_velocity": [1, 2, 3],
        "initial_angular_velocity": [4, 5, 6],
    }
    req.update(overrides)
    return req


def test_emit_and_read_command_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    writes: list[bytes] = []

    def fake_write(_fd: int, data: memoryview | bytes) -> int:
        chunk = bytes(data[:3])
        writes.append(chunk)
        return len(chunk)

    monkeypatch.setattr(module.os, "write", fake_write)
    module._emit({"status": "ok", "value": 1})
    assert b"".join(writes).endswith(b"\n")
    assert json.loads(b"".join(writes)) == {"status": "ok", "value": 1}

    writes.clear()
    module._emit_error("bad", detail=2)
    assert json.loads(b"".join(writes))["error"] == "bad"

    attempts = {"count": 0}

    def interrupted_then_complete(_fd: int, data: memoryview | bytes) -> int:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise InterruptedError()
        return len(data)

    monkeypatch.setattr(module.os, "write", interrupted_then_complete)
    module._emit({"status": "after-interrupt"})
    assert attempts["count"] == 2

    monkeypatch.setattr(module.os, "write", lambda _fd, _data: 0)
    module._emit({"status": "zero-write"})

    monkeypatch.setattr(module.sys, "stdin", io.StringIO('{"command": "x"}\n'))
    assert module._read_command() == {"command": "x"}
    monkeypatch.setattr(module.sys, "stdin", io.StringIO("not-json\n"))
    assert module._read_command() == {}
    monkeypatch.setattr(module.sys, "stdin", io.StringIO("\n"))
    assert module._read_command() is None
    monkeypatch.setattr(module.sys, "stdin", io.StringIO(""))
    assert module._read_command() is None


def test_import_ovphysx_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    fake_ovphysx = types.ModuleType("ovphysx")
    fake_ovphysx.PhysX = FakePhysX
    fake_ovphysx.TensorType = types.SimpleNamespace(
        RIGID_BODY_POSE="pose",
        RIGID_BODY_VELOCITY="velocity",
    )
    monkeypatch.setitem(sys.modules, "ovphysx", fake_ovphysx)
    ovphysx_mod, physx_cls, tensor_type = module._import_ovphysx()
    assert ovphysx_mod is fake_ovphysx
    assert physx_cls is FakePhysX
    assert tensor_type.RIGID_BODY_POSE == "pose"

    emitted: list[dict[str, Any]] = []
    monkeypatch.delitem(sys.modules, "ovphysx", raising=False)
    monkeypatch.setattr(
        module,
        "_emit_error",
        lambda message, **extra: emitted.append({"message": message, **extra}),
    )
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ovphysx":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        module._import_ovphysx()
    assert emitted[0]["message"].startswith("ovphysx import failed")


def test_daemon_state_evaluate_reset_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    tensor_type = types.SimpleNamespace(
        RIGID_BODY_POSE="pose",
        RIGID_BODY_VELOCITY="velocity",
    )
    state = module._DaemonState(object(), FakePhysX, tensor_type)
    result = state.evaluate(_request())

    assert result["status"] == "ok"
    assert result["n_bodies"] == 1
    assert result["n_steps"] == 2
    assert len(result["trajectory"]) == 2
    assert state._physx.velocity_binding.writes[0][0].tolist() == [1, 2, 3, 4, 5, 6]
    assert state._physx.steps == [(2, 0.05, 0.0)]

    previous_pose = state._physx.pose_binding
    previous_velocity = state._physx.velocity_binding
    state.reset_state()
    assert previous_pose.destroyed
    assert previous_velocity.destroyed
    assert state._physx.removed == [42]

    state._previous_bindings = [FakeBinding((1, 7), 0, destroy_raises=True)]
    state._previous_usd_handle = 7
    monkeypatch.setattr(
        state._physx,
        "remove_usd",
        lambda _handle: (_ for _ in ()).throw(RuntimeError("remove failed")),
    )
    state.reset_state()
    assert state._previous_bindings == []
    assert state._previous_usd_handle is None

    monkeypatch.setattr(
        state._physx,
        "release",
        lambda: (_ for _ in ()).throw(RuntimeError("release failed")),
    )
    state.shutdown()


def test_daemon_state_evaluate_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    tensor_type = types.SimpleNamespace(
        RIGID_BODY_POSE="pose",
        RIGID_BODY_VELOCITY="velocity",
    )

    class ZeroBodyPhysX(FakePhysX):
        def __init__(self, *, device: str):
            super().__init__(device=device)
            self.pose_binding = FakeBinding((0, 7), 1.0)

    state = module._DaemonState(object(), ZeroBodyPhysX, tensor_type)
    with pytest.raises(RuntimeError, match="no rigid bodies matched"):
        state.evaluate(_request())

    for overrides, match in [
        ({"dt": 0}, "dt must be"),
        ({"sample_fps": 0}, "sample_fps must be"),
        ({"duration_s": 0}, "duration_s must be"),
    ]:
        state = module._DaemonState(object(), FakePhysX, tensor_type)
        with pytest.raises(ValueError, match=match):
            state.evaluate(_request(**overrides))


def test_build_state_reports_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    tensor_type = types.SimpleNamespace(
        RIGID_BODY_POSE="pose",
        RIGID_BODY_VELOCITY="velocity",
    )
    monkeypatch.setattr(
        module,
        "_import_ovphysx",
        lambda: (object(), FakePhysX, tensor_type),
    )

    def fail_initialization(*_args: Any) -> None:
        raise RuntimeError("native startup failed")

    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "_DaemonState", fail_initialization)
    monkeypatch.setattr(
        module,
        "_emit_error",
        lambda message, **extra: emitted.append({"message": message, **extra}),
    )

    assert module._build_state() is None
    assert emitted[0]["message"] == (
        "ovphysx initialization failed: native startup failed"
    )
    assert "RuntimeError: native startup failed" in emitted[0]["traceback"]


def test_build_state_and_main_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    tensor_type = types.SimpleNamespace(
        RIGID_BODY_POSE="pose",
        RIGID_BODY_VELOCITY="velocity",
    )
    monkeypatch.setattr(
        module,
        "_import_ovphysx",
        lambda: (object(), FakePhysX, tensor_type),
    )
    assert isinstance(module._build_state(), module._DaemonState)
    monkeypatch.setattr(
        module,
        "_import_ovphysx",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert module._build_state() is None

    class MainState:
        def __init__(self) -> None:
            self.reset_count = 0
            self.shutdown_count = 0

        def evaluate(self, cmd: dict[str, Any]) -> dict[str, Any]:
            if cmd.get("raise"):
                raise RuntimeError("evaluate failed")
            return {"status": "ok", "trajectory": []}

        def reset_state(self) -> None:
            self.reset_count += 1

        def shutdown(self) -> None:
            self.shutdown_count += 1

    state = MainState()
    commands = iter(
        [
            {"command": "evaluate"},
            {"command": "evaluate", "raise": True},
            {"command": "reset_only"},
            {"command": "unknown"},
            {},
            {"command": "shutdown"},
        ]
    )
    emitted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "_build_state", lambda: state)
    monkeypatch.setattr(module, "_read_command", lambda: next(commands))
    monkeypatch.setattr(module, "_emit", lambda payload: emitted.append(payload))
    monkeypatch.setattr(
        module,
        "_emit_error",
        lambda message, **extra: errors.append({"error": message, **extra}),
    )

    assert module.main() == 0
    assert emitted[0]["status"] == "ready"
    assert {"status": "ok"} in emitted
    assert state.reset_count == 1
    assert state.shutdown_count == 1
    assert any(error["error"] == "evaluate failed" for error in errors)
    assert any("unknown command" in error["error"] for error in errors)

    monkeypatch.setattr(module, "_build_state", lambda: None)
    assert module.main() == 1

    monkeypatch.setattr(module, "_build_state", lambda: MainState())
    monkeypatch.setattr(module, "_read_command", lambda: None)
    assert module.main() == 0
