# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bearer-token enforcement wiring for the Texture Agent Service (issue #956).

Enforcement is opt-in: with TEXTURE_AGENT_TOKEN unset the service behaves
exactly as it did before, which is what keeps existing deployments working.

Unlike the sibling services, texture has no conftest and its other TestClient
suites build their own bare ``FastAPI()`` and mount routers directly. Those
would pass whether or not the guard works, so this module imports the real app
from ``service.main`` — the pattern used by test_s3_openapi_contract.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

TOKEN_ENV = "TEXTURE_AGENT_TOKEN"
PROBE = "/sessions"
OPEN_PATHS = ("/health", "/api", "/", "/openapi.json")
MIN_PROTECTED_PATHS = 17
PROTECTED_PREFIXES = ("/pipeline", "/artifacts", "/sessions")


@pytest.fixture(scope="module")
def app():
    from ...service.main import app as real_app

    return real_app


@pytest.fixture
def client(app) -> TestClient:
    """A client that surfaces downstream errors as responses.

    These tests assert only on the auth boundary. The real app's lifespan does
    not run under TestClient, so routes that need a SessionManager raise once
    the request gets past the guard. ``raise_server_exceptions=False`` turns
    that into a 500 response, which is exactly the signal these tests want: the
    request was *not* rejected with 401. A regression that wrongly rejected the
    request would still produce a 401 and fail the assertion.
    """
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token exported to drive the client would otherwise skew every test."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)


def test_default_deployment_needs_no_token(client: TestClient) -> None:
    assert client.get(PROBE).status_code != 401


def test_stale_client_token_is_ignored_when_disabled(client: TestClient) -> None:
    response = client.get(PROBE, headers={"Authorization": "Bearer stale-token"})
    assert response.status_code != 401


def test_missing_token_is_rejected_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "s3cret")
    response = client.get(PROBE)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_wrong_token_is_rejected_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "s3cret")
    assert (
        client.get(PROBE, headers={"Authorization": "Bearer nope"}).status_code == 401
    )


def test_correct_token_is_accepted_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "s3cret")
    response = client.get(PROBE, headers={"Authorization": "Bearer s3cret"})
    assert response.status_code != 401


@pytest.mark.parametrize("path", OPEN_PATHS)
def test_probe_paths_stay_open_under_enforcement(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "s3cret")
    assert client.get(path).status_code != 401, f"{path} must not require a token"


def test_health_reports_enforcement_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.get("/health").json()["auth_enforced"] is False
    monkeypatch.setenv(TOKEN_ENV, "s3cret")
    assert client.get("/health").json()["auth_enforced"] is True


def test_every_protected_operation_declares_the_guard(app) -> None:
    """Catches a future router included without the auth dependency.

    Reads the generated OpenAPI schema rather than ``app.routes``, which does
    not expose mounted router routes here and would silently pass over an
    empty list.
    """
    paths = app.openapi()["paths"]
    protected = {
        path: ops for path, ops in paths.items() if path.startswith(PROTECTED_PREFIXES)
    }
    assert len(protected) >= MIN_PROTECTED_PATHS, (
        f"expected at least {MIN_PROTECTED_PATHS} protected paths, found "
        f"{len(protected)} — this check is not seeing the mounted routers"
    )
    missing = [
        f"{method.upper()} {path}"
        for path, ops in protected.items()
        for method, op in ops.items()
        if method in {"get", "post", "put", "patch", "delete"}
        and not op.get("security")
    ]
    assert not missing, f"operations without the bearer guard: {missing}"


def test_open_operations_do_not_declare_the_guard(app) -> None:
    paths = app.openapi()["paths"]
    leaked = [
        f"{method.upper()} {path}"
        for path in OPEN_PATHS
        for method, op in paths.get(path, {}).items()
        if method in {"get", "post"} and op.get("security")
    ]
    assert not leaked, f"probe routes must stay open: {leaked}"
