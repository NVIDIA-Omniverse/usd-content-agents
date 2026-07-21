# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the bundled Material Agent skill reference client."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CLIENT_PATH = REPO_ROOT / ".agents/skills/material-agent-client/references/client.py"
SPEC = importlib.util.spec_from_file_location(
    "material_agent_skill_reference_client", CLIENT_PATH
)
assert SPEC is not None and SPEC.loader is not None
CLIENT_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLIENT_MODULE
_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    SPEC.loader.exec_module(CLIENT_MODULE)
finally:
    sys.dont_write_bytecode = _dont_write_bytecode
MaterialAgentClient = CLIENT_MODULE.MaterialAgentClient


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"session_id": "session-1"}


class _FakeSession:
    def __init__(self) -> None:
        self.posts: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.posts.append({"url": url, **kwargs})
        return _FakeResponse()


def _client() -> tuple[MaterialAgentClient, _FakeSession]:
    client = MaterialAgentClient(base_url="http://service")
    session = _FakeSession()
    client._http = session
    return client, session


def test_public_methods_preserve_legacy_positional_parameter_order() -> None:
    start_parameters = list(
        inspect.signature(MaterialAgentClient.start_pipeline).parameters
    )
    start_legacy_tail = [
        "render_num_workers",
        "generated_reference_id",
        "user_email",
        "layer_only",
    ]
    start_tail_index = start_parameters.index(start_legacy_tail[0])
    assert (
        start_parameters[start_tail_index : start_tail_index + 4] == start_legacy_tail
    )
    assert start_parameters.index("enable_prim_clustering") > start_parameters.index(
        "layer_only"
    )

    run_parameters = list(
        inspect.signature(MaterialAgentClient.run_and_monitor).parameters
    )
    run_legacy_tail = ["render_num_workers", "user_email", "layer_only"]
    run_tail_index = run_parameters.index(run_legacy_tail[0])
    assert run_parameters[run_tail_index : run_tail_index + 3] == run_legacy_tail
    assert run_parameters.index("enable_prim_clustering") > run_parameters.index(
        "layer_only"
    )


def test_start_pipeline_omits_optional_defaults(tmp_path: Path) -> None:
    client, session = _client()
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    client.start_pipeline(usd_path=str(usd_path))

    assert session.posts[0]["data"] == {}


def test_start_pipeline_omits_blank_email_and_normalizes_explicit_email(
    tmp_path: Path,
) -> None:
    client, session = _client()
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    client.start_pipeline(usd_path=str(usd_path), user_email="   ")
    client.start_pipeline(usd_path=str(usd_path), user_email="  user@example.com  ")

    assert session.posts[0]["data"] == {}
    assert session.posts[1]["data"] == {"user_email": "user@example.com"}


@pytest.mark.parametrize(("value", "serialized"), [(True, "true"), (False, "false")])
def test_start_pipeline_serializes_explicit_booleans(
    tmp_path: Path,
    value: bool,
    serialized: str,
) -> None:
    client, session = _client()
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    client.start_pipeline(
        usd_path=str(usd_path),
        enable_prim_clustering=value,
        cluster_report=value,
        layer_only=value,
    )

    assert session.posts[0]["data"] == {
        "enable_prim_clustering": serialized,
        "cluster_report": serialized,
        "layer_only": serialized,
    }


def test_start_pipeline_posts_exact_prim_clustering_payload(tmp_path: Path) -> None:
    client, session = _client()
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    client.start_pipeline(
        usd_path=str(usd_path),
        enable_prim_clustering=True,
        cluster_min_prims=25,
        cluster_embedding_backend="nim",
        cluster_embedding_model="embedding-model",
        cluster_embedding_base_url="http://embedding:8000/v1",
        cluster_embedding_max_workers=2,
        cluster_embedding_batch_size=8,
        cluster_max_size=11,
        cluster_similarity_threshold_low=0.97,
        cluster_similarity_threshold_medium=0.94,
        cluster_similarity_threshold_high=0.88,
        cluster_report=False,
    )

    assert session.posts[0]["data"] == {
        "enable_prim_clustering": "true",
        "cluster_min_prims": "25",
        "cluster_embedding_backend": "nim",
        "cluster_embedding_model": "embedding-model",
        "cluster_embedding_base_url": "http://embedding:8000/v1",
        "cluster_embedding_max_workers": "2",
        "cluster_embedding_batch_size": "8",
        "cluster_max_size": "11",
        "cluster_similarity_threshold_low": "0.97",
        "cluster_similarity_threshold_medium": "0.94",
        "cluster_similarity_threshold_high": "0.88",
        "cluster_report": "false",
    }


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("cluster_min_prims", 0, "cluster_min_prims must be at least 1"),
        (
            "cluster_similarity_threshold_low",
            1.5,
            "cluster_similarity_threshold_low must be between 0.0 and 1.0",
        ),
        (
            "cluster_similarity_threshold_low",
            True,
            "cluster_similarity_threshold_low must be an int or float",
        ),
        (
            "cluster_similarity_threshold_low",
            float("nan"),
            "cluster_similarity_threshold_low must be between 0.0 and 1.0",
        ),
        (
            "cluster_similarity_threshold_low",
            float("inf"),
            "cluster_similarity_threshold_low must be between 0.0 and 1.0",
        ),
        (
            "cluster_similarity_threshold_low",
            float("-inf"),
            "cluster_similarity_threshold_low must be between 0.0 and 1.0",
        ),
    ],
)
def test_start_pipeline_rejects_invalid_clustering_overrides(
    tmp_path: Path,
    override: str,
    value: object,
    message: str,
) -> None:
    client, session = _client()
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        client.start_pipeline(usd_path=str(usd_path), **{override: value})

    assert session.posts == []


@pytest.mark.parametrize(
    ("layer_only", "expected_body"),
    [
        (None, {"steps": ["apply"]}),
        (True, {"steps": ["apply"], "layer_only": True}),
        (False, {"steps": ["apply"], "layer_only": False}),
    ],
)
def test_regenerate_serializes_layer_only_tristate(
    layer_only: bool | None,
    expected_body: dict[str, object],
) -> None:
    client, session = _client()

    client.regenerate("session-1", ["apply"], layer_only=layer_only)

    assert session.posts[0]["json"] == expected_body


def test_run_and_monitor_forwards_initial_pipeline_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, session = _client()
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr(
        client, "get_status", lambda _session_id: {"status": "completed"}
    )

    client.run_and_monitor(
        usd_path=str(usd_path),
        enable_prim_clustering=True,
        cluster_min_prims=25,
        layer_only=False,
        print_stream=False,
    )

    assert session.posts[0]["data"] == {
        "enable_prim_clustering": "true",
        "cluster_min_prims": "25",
        "layer_only": "false",
    }


def test_cli_email_is_optional() -> None:
    args = CLIENT_MODULE.build_arg_parser().parse_args(["scene.usd"])

    assert args.email is None
    assert args.enable_prim_clustering is None
    assert args.cluster_report is None
    assert args.layer_only is None


@pytest.mark.parametrize(
    ("flag", "attribute", "expected"),
    [
        ("--enable-prim-clustering", "enable_prim_clustering", True),
        ("--disable-prim-clustering", "enable_prim_clustering", False),
        ("--cluster-report", "cluster_report", True),
        ("--no-cluster-report", "cluster_report", False),
        ("--layer-only", "layer_only", True),
        ("--no-layer-only", "layer_only", False),
    ],
)
def test_cli_boolean_overrides_accept_explicit_values(
    flag: str,
    attribute: str,
    expected: bool,
) -> None:
    args = CLIENT_MODULE.build_arg_parser().parse_args([flag, "scene.usd"])

    assert getattr(args, attribute) is expected
