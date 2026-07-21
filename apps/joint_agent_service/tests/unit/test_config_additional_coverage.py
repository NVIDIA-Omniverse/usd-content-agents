# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ...service import config as config_module
from ...service import storage as storage_module
from ...service.storage.local_store import LocalSessionStore


def test_config_validates_vlm_max_workers_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JA_VLM_MAX_WORKERS", raising=False)
    assert config_module.ServiceConfig().vlm_max_workers == 64

    monkeypatch.setenv("JA_VLM_MAX_WORKERS", "1")
    assert config_module.ServiceConfig().vlm_max_workers == 1

    for invalid_value in ("0", "not-an-integer"):
        monkeypatch.setenv("JA_VLM_MAX_WORKERS", invalid_value)
        with pytest.raises(ValidationError):
            config_module.ServiceConfig()


def test_config_load_description_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingPath:
        def __init__(self, *_args: Any) -> None:
            pass

        @property
        def parent(self) -> MissingPath:
            return self

        def __truediv__(self, _name: str) -> MissingPath:
            return self

        def exists(self) -> bool:
            return False

    monkeypatch.setattr(config_module, "Path", MissingPath)

    assert config_module.ServiceConfig._load_description() == (
        "Joint Agent REST API Service"
    )


def test_config_build_session_store_local_and_s3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_cfg = config_module.ServiceConfig(
        session_storage_path=str(tmp_path / "local"),
        storage_kind="local",
    )
    assert isinstance(local_cfg.build_session_store(), LocalSessionStore)

    captured = {}

    def fake_from_config(storage_cfg):
        captured["storage_cfg"] = storage_cfg
        return "s3-store"

    monkeypatch.setattr(storage_module.S3SessionStore, "from_config", fake_from_config)

    s3_cfg = config_module.ServiceConfig(
        session_storage_path=str(tmp_path / "sessions"),
        storage_kind="s3",
        storage_s3_bucket="bucket",
        storage_s3_prefix="prefix",
    )
    assert s3_cfg.build_session_store() == "s3-store"
    assert captured["storage_cfg"].kind == "s3"
    assert captured["storage_cfg"].s3_bucket == "bucket"
    assert captured["storage_cfg"].s3_prefix == "prefix"


@pytest.mark.parametrize("bucket", ["", "   "])
def test_config_rejects_s3_storage_without_bucket(
    tmp_path: Path,
    bucket: str,
) -> None:
    s3_cfg = config_module.ServiceConfig(
        session_storage_path=str(tmp_path / "sessions"),
        storage_kind="s3",
        storage_s3_bucket=bucket,
    )

    with pytest.raises(
        ValueError,
        match="JA_STORAGE_S3_BUCKET is required when JA_STORAGE_KIND=s3",
    ):
        s3_cfg.build_session_store()
