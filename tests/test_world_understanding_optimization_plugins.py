# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral optimizer extension contract tests."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from world_understanding.optimization import optimizers, registry
from world_understanding.optimization.contracts import (
    SUPPORTED_OPTIMIZERS,
    OptimizerSettings,
)
from world_understanding.optimization.errors import OptimizerUnavailableError


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_optimizer_plugins", {})
    monkeypatch.setattr(registry, "_optimizer_plugins_scanned", False)
    monkeypatch.setattr(registry, "_loaded_optimizer_plugins", set())
    monkeypatch.setattr(registry, "_failed_optimizer_plugins", {})
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_kwargs: ())


def test_registered_optimizer_resolves_and_dispatches() -> None:
    def runner(*_args, **_kwargs) -> None:
        return None

    registry.register_optimizer("test-remote", runner)

    assert optimizers.resolve_optimizer("test-remote") == "test-remote"
    assert optimizers.get_runner("test-remote") is runner
    assert "test-remote" in optimizers.get_supported_optimizer_names()
    assert OptimizerSettings(name="test-remote").name == "test-remote"


def test_unavailable_optimizer_preserves_extension_message() -> None:
    registry.register_optimizer(
        "test-remote",
        lambda *_args, **_kwargs: None,
        is_available=lambda: False,
        unavailable_message="test remote runtime is unavailable",
    )

    with pytest.raises(
        OptimizerUnavailableError,
        match="test remote runtime is unavailable",
    ):
        optimizers.resolve_optimizer("test-remote")


def test_entry_point_registrar_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    discovery_calls = 0

    def registrar() -> None:
        calls.append("loaded")
        registry.register_optimizer(
            "test-remote",
            lambda *_args, **_kwargs: None,
        )

    entry_point = SimpleNamespace(
        name="test-provider",
        value="test_provider:register",
        load=lambda: registrar,
    )

    def entry_points(**_kwargs: object) -> tuple[SimpleNamespace, ...]:
        nonlocal discovery_calls
        discovery_calls += 1
        return (entry_point, entry_point)

    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        entry_points,
    )

    assert registry.list_registered_optimizer_names() == ("test-remote",)
    assert registry.list_registered_optimizer_names() == ("test-remote",)
    assert registry.list_loaded_optimizer_plugins() == (
        "test-provider:test_provider:register",
    )
    assert calls == ["loaded"]
    assert discovery_calls == 1


def test_concurrent_entry_point_registrar_loads_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_count = 8
    start = threading.Barrier(worker_count)
    calls = 0
    calls_lock = threading.Lock()

    def registrar() -> None:
        nonlocal calls
        with calls_lock:
            calls += 1

    def load_registrar() -> object:
        time.sleep(0.05)
        return registrar

    entry_point = SimpleNamespace(
        name="test-provider",
        value="test_provider:register",
        load=load_registrar,
    )
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda **_kwargs: (entry_point,),
    )

    def load_plugin(_index: int) -> tuple[str, ...]:
        start.wait()
        return registry.load_optimizer_plugins()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(load_plugin, range(worker_count)))

    expected = ("test-provider:test_provider:register",)
    assert results == [expected] * worker_count
    assert calls == 1


@pytest.mark.parametrize("name", ("", " ", *SUPPORTED_OPTIMIZERS))
def test_registry_rejects_empty_and_builtin_names(name: str) -> None:
    with pytest.raises(ValueError):
        registry.register_optimizer(name, lambda *_args, **_kwargs: None)


def test_broken_entry_point_does_not_break_builtins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_registrar() -> None:
        registry.register_optimizer(
            "partially-loaded",
            lambda *_args, **_kwargs: None,
        )
        raise RuntimeError("boom")

    def good_registrar() -> None:
        registry.register_optimizer(
            "test-remote",
            lambda *_args, **_kwargs: None,
        )

    broken = SimpleNamespace(
        name="broken-provider",
        value="broken_provider:register",
        load=lambda: broken_registrar,
    )
    good = SimpleNamespace(
        name="good-provider",
        value="good_provider:register",
        load=lambda: good_registrar,
    )
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda **_kwargs: (broken, good),
    )

    # The healthy plugin still registers and built-ins remain usable.
    assert registry.list_registered_optimizer_names() == ("test-remote",)
    assert registry.get_registered_optimizer("partially-loaded") is None
    assert optimizers.resolve_optimizer("random") == "random"
    assert "random" in optimizers.get_supported_optimizer_names()

    # The failure is recorded once for diagnostics, not retried per call.
    failures = registry.list_failed_optimizer_plugins()
    assert failures == {"broken-provider:broken_provider:register": "boom"}
    registry.list_registered_optimizer_names()
    assert registry.list_failed_optimizer_plugins() == failures
