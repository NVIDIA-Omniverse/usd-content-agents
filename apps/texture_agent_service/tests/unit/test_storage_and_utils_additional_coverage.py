# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from pathlib import Path

from ...service import utils as utils_module
from ...service.storage.local_store import LocalSessionStore
from ...service.utils import AccessLogFilter, get_version


def test_utils_version_fallback_and_access_log_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        utils_module,
        "version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError("missing")),
    )
    assert get_version() == "0.0.1-dev"

    access_filter = AccessLogFilter()
    assert not access_filter.filter(
        logging.LogRecord("x", logging.INFO, "", 1, "GET /health", (), None)
    )
    assert not access_filter.filter(
        logging.LogRecord("x", logging.INFO, "", 1, "GET /metrics", (), None)
    )
    assert access_filter.filter(
        logging.LogRecord("x", logging.INFO, "", 1, "GET /api", (), None)
    )


def test_local_store_missing_and_noop_edges(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))

    assert store.list_sessions() == []
    store.invalidate_sessions_cache()
    assert store._prefix_matches("cache/a.txt", "") is True
    assert store.list_keys("missing") == []
    assert store.get_event_log("missing") == []
    assert store.sync_to_local("missing", str(tmp_path / "target")) == 0
    assert store.sync_from_local("copied", str(tmp_path / "missing-source")) == 0

    store.init_session("sid")
    store.put_bytes("sid", "cache/a.txt", b"a")
    store.put_bytes("sid", "other.txt", b"skip")
    target = tmp_path / "target"
    assert store.sync_to_local("sid", str(target), prefix="cache/") == 1
    assert (target / "cache" / "a.txt").read_bytes() == b"a"
    assert not (target / "other.txt").exists()

    source = tmp_path / "source"
    (source / "cache").mkdir(parents=True)
    (source / "cache" / "b.txt").write_bytes(b"b")
    (source / "other.txt").write_bytes(b"skip")
    assert store.sync_from_local("copied", str(source), prefix="cache/") == 1
    assert (store._session_dir("copied") / "cache" / "b.txt").read_bytes() == b"b"
    assert not (store._session_dir("copied") / "other.txt").exists()
