# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any

import pytest

from ...client.client import JointAgentClient, build_arg_parser
from ...client.client_v2 import JointAgentClient as JointAgentClientV2
from ...client.client_v2 import build_arg_parser as build_v2_arg_parser

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def test_python_client_exposes_owned_core_adapter() -> None:
    calls: list[dict] = []

    class _Response:
        ok = True
        status_code = 202
        text = ""

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, str]:
            return {"session_id": "session-1", "run_id": "a" * 32}

    class _Http:
        def post(self, url: str, **kwargs: Any) -> _Response:
            calls.append({"url": url, **kwargs})
            return _Response()

    client = JointAgentClient(base_url=BASE_URL)
    client._http = _Http()
    session_id = client.start_pipeline(
        session_id="existing-session",
        apply_joint_rigger=True,
        joint_rigger_adapter="owned_core",
        joint_rigger_apply_masses=False,
        joint_rigger_apply_collision=False,
    )

    assert session_id == "session-1"
    assert calls[0]["data"] == {
        "session_id": "existing-session",
        "apply_joint_rigger": "true",
        "joint_rigger_adapter": "owned_core",
        "joint_rigger_apply_masses": "false",
        "joint_rigger_apply_collision": "false",
    }
    client.cancel(session_id)
    assert calls[1]["params"] == {"run_id": "a" * 32}
    client.cancel(session_id, run_id="")
    assert calls[2]["params"] == {"run_id": ""}
    with pytest.raises(ValueError, match="run_id is required"):
        client.cancel("external-session")
    adapter_action = next(
        action
        for action in build_arg_parser()._actions
        if action.dest == "joint_rigger_adapter"
    )
    assert adapter_action.choices == ["owned_core", "mock", "usd_joint_rigger"]


def test_v2_client_uses_exact_run_token() -> None:
    calls: list[dict] = []

    class _Response:
        ok = True
        status_code = 202
        text = ""

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, str]:
            return {"session_id": "session-v2", "run_id": "b" * 32}

    class _Http:
        def post(self, url: str, **kwargs: Any) -> _Response:
            calls.append({"url": url, **kwargs})
            return _Response()

    client = JointAgentClientV2(base_url=BASE_URL)
    client._http = _Http()
    session_id = client.start_pipeline(session_id="existing-session")

    assert session_id == "session-v2"
    client.cancel(session_id)
    assert calls[1]["params"] == {"run_id": "b" * 32}
    client.cancel(session_id, run_id="")
    assert calls[2]["params"] == {"run_id": ""}
    with pytest.raises(ValueError, match="run_id is required"):
        client.cancel("external-session")


@pytest.mark.parametrize("parser_factory", (build_arg_parser, build_v2_arg_parser))
def test_client_defers_render_backend_validation_to_server(parser_factory: Any) -> None:
    args = parser_factory().parse_args(
        ["--render-backend", "future-server-backend", "asset.usdz"]
    )

    assert args.render_backend == "future-server-backend"


@pytest.mark.parametrize("client_type", [JointAgentClient, JointAgentClientV2])
def test_clients_cancel_the_latest_regenerated_run(client_type) -> None:
    calls: list[dict] = []

    class _Response:
        ok = True
        status_code = 202
        text = ""

        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"session_id": "regenerated-session", "run_id": self.run_id}

    class _Http:
        def post(self, url: str, **kwargs: Any) -> _Response:
            calls.append({"url": url, **kwargs})
            run_id = "b" * 32 if url.endswith("/regenerate") else "a" * 32
            return _Response(run_id)

    client = client_type(base_url=BASE_URL)
    client._http = _Http()
    session_id = client.start_pipeline(session_id="existing-session")
    regenerated = client.regenerate(session_id, ["predict"])
    client.cancel(session_id)

    assert regenerated["run_id"] == "b" * 32
    assert calls[2]["params"] == {"run_id": "b" * 32}


@pytest.fixture
def api_client() -> JointAgentClient:
    client = JointAgentClient(base_url=BASE_URL)
    return client


@pytest.mark.parametrize(
    "kwargs",
    [
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../tests/test_data/simple_cube.usda",
                ),
            }
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../tests/test_data/simple_cube.usda",
                ),
                "user_prompt": "Classify the components of this articulated body.",
            }
        ),
    ],
)
@pytest.mark.skipif(
    os.getenv("RUN_CLIENT_TESTS", "false").lower() not in ["true", "1", "yes", "y"],
    reason="Skipping test in CI",
)
async def test_basic_pipeline(api_client: JointAgentClient, kwargs: dict):
    session_id, status = api_client.run_and_monitor(**kwargs)
    assert status is not None
    assert status["status"] == "completed"

    # Check predictions artifact
    predictions = api_client.download_predictions(session_id)
    assert len(predictions) > 0
    print(f"Predictions downloaded: {len(predictions)} bytes")

    # Check report artifact
    report = api_client._http.get(
        f"{api_client.base_url}/artifacts/{session_id}/report"
    )
    assert report.status_code == 200
    report_text = report.text
    assert len(report_text) > 0
    print(f"Report downloaded: {len(report_text)} bytes")

    if "user_prompt" in kwargs:
        assert kwargs["user_prompt"] in report_text
        print(f"user_prompt: {kwargs['user_prompt']} found in report")
