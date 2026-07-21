# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for object store implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from world_understanding.utils.object_store import (
    InMemoryObjectStore,
    ObjectStore,
    TempDirObjectStore,
)


def test_object_store_abstract_method_bodies_are_callable() -> None:
    ObjectStore.get(object(), "missing", "default")
    ObjectStore.set(object(), "key", "value")
    ObjectStore.delete(object(), "key")
    ObjectStore.exists(object(), "key")
    ObjectStore.keys(object())
    ObjectStore.clear(object())


def test_in_memory_object_store_crud() -> None:
    store = InMemoryObjectStore()

    assert store.get("missing", "default") == "default"
    assert store.exists("item") is False

    store.set("item", {"value": 1})
    store.set("other", [1, 2])
    assert store.get("item") == {"value": 1}
    assert store.exists("item") is True
    assert store.keys() == ["item", "other"]

    store.delete("missing")
    store.delete("item")
    assert store.exists("item") is False

    store.clear()
    assert store.keys() == []


def test_temp_dir_object_store_json_round_trip_and_keys(tmp_path: Path) -> None:
    store = TempDirObjectStore(tmp_path / "json-store", serializer="json")

    assert store.get("missing", {"fallback": True}) == {"fallback": True}
    assert store._get_path("nested\\value").name == "nested_value.json"

    store.set("nested/value", {"ok": True})
    store.set("plain", [1, 2, 3])
    (tmp_path / "json-store" / "ignored.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "json-store" / "subdir").mkdir()

    assert store.get("nested/value") == {"ok": True}
    assert store.exists("plain") is True
    assert sorted(store.keys()) == ["nested/value", "plain"]

    store.delete("nested/value")
    store.delete("nested/value")
    assert store.exists("nested/value") is False
    assert "nested/value" not in store._metadata

    store.clear()
    assert store.keys() == []
    assert store._metadata == {}


def test_temp_dir_object_store_pickle_round_trip_and_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = TempDirObjectStore(tmp_path / "pickle-store")

    store.set("nested/value", {"ok": True})
    assert store.get("nested/value") == {"ok": True}
    assert store.exists("nested/value") is True
    assert store.keys() == ["nested/value"]

    store._get_path("broken").write_bytes(b"not a pickle")
    assert store.get("broken", "fallback") == "fallback"
    assert "Error reading broken" in capsys.readouterr().out

    bad_json = TempDirObjectStore(tmp_path / "bad-json", serializer="json")
    with pytest.raises(RuntimeError, match="Failed to store bad"):
        bad_json.set("bad", object())


def test_temp_dir_object_store_auto_cleanup_best_effort(tmp_path: Path) -> None:
    auto = TempDirObjectStore()
    auto_path = auto._temp_dir
    auto.set("value", {"ok": True})
    assert auto_path.exists()
    auto.__del__()
    assert not auto_path.exists()

    already_removed = TempDirObjectStore()
    removed_path = already_removed._temp_dir
    removed_path.rmdir()
    already_removed.__del__()
    assert not removed_path.exists()
