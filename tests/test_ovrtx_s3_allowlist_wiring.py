# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep every concrete OVRTX Compose service fail-closed for S3 intake."""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILES = sorted((REPO_ROOT / "apps").glob("**/docker-compose*.yml"))
CONCRETE_OVRTX_COMPOSE_FILES = [
    path
    for path in COMPOSE_FILES
    if "OVRTX_LOG_LEVEL=" in path.read_text(encoding="utf-8")
]


def test_concrete_ovrtx_compose_discovery_is_not_empty() -> None:
    assert CONCRETE_OVRTX_COMPOSE_FILES


@pytest.mark.parametrize(
    "compose_path",
    CONCRETE_OVRTX_COMPOSE_FILES,
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_concrete_ovrtx_compose_service_passes_s3_allowlist(
    compose_path: Path,
) -> None:
    document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    environment = document["services"]["ovrtx-rendering-api"]["environment"]
    allowlist_values: list[str] = []
    for entry in environment:
        if isinstance(entry, str):
            name, _, value = entry.partition("=")
        else:
            name, value = next(iter(entry.items()))
        if name == "OVRTX_S3_ALLOWED_BUCKETS":
            allowlist_values.append(value)

    assert allowlist_values == ["${OVRTX_S3_ALLOWED_BUCKETS:-}"]
