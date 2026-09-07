# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bearer-token enforcement wiring for the Material Agent Service (issue #956).

Enforcement is opt-in: with MATERIAL_AGENT_TOKEN unset the service behaves exactly as it
did before, which is what keeps existing deployments working.
"""

from __future__ import annotations

import pytest

TOKEN_ENV = "MATERIAL_AGENT_TOKEN"
PROBE = "/materials/libraries"
OPEN_PATHS = ("/health", "/api", "/", "/openapi.json", "/manual")
MIN_PROTECTED_PATHS = 40
PROTECTED_PREFIXES = ("/pipeline", "/artifacts", "/assets", "/sessions", "/materials")


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token exported to drive the client would skew the disabled-mode tests.

    The shared conftest does not clear this variable and the guard reads it per
    request, so an ambient value would make the unauthenticated assertions 401.
    """
    monkeypatch.delenv(TOKEN_ENV, raising=False)


@pytest.fixture
async def auth_client(app):
    """Client that returns downstream errors instead of raising them.

    These tests assert only on the auth boundary. Sibling tests in the same
    session swap the module-level session manager for a mock, so a probe that
    gets *past* the guard can fail inside the handler. Surfacing that as a 500
    response is exactly the signal wanted here: the request was not rejected
    with 401. A regression that wrongly rejected it would still produce a 401
    and fail the assertion.
    """
    import httpx

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestServiceAuthWiring:
    async def test_default_deployment_needs_no_token(self, auth_client) -> None:
        """The backward-compatibility guarantee: unset means unchanged."""
        response = await auth_client.get(PROBE)
        assert response.status_code != 401

    async def test_stale_client_token_is_ignored_when_disabled(
        self, auth_client
    ) -> None:
        response = await auth_client.get(
            PROBE, headers={"Authorization": "Bearer stale-token"}
        )
        assert response.status_code != 401

    async def test_missing_token_is_rejected_when_enabled(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await client.get(PROBE)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    async def test_wrong_token_is_rejected_when_enabled(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await client.get(PROBE, headers={"Authorization": "Bearer nope"})
        assert response.status_code == 401

    async def test_correct_token_is_accepted_when_enabled(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await auth_client.get(
            PROBE, headers={"Authorization": "Bearer s3cret"}
        )
        assert response.status_code != 401

    @pytest.mark.parametrize("path", OPEN_PATHS)
    async def test_probe_paths_stay_open_under_enforcement(
        self, client, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        """Liveness probes and docs send no Authorization header."""
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await client.get(path)
        assert response.status_code != 401, f"{path} must not require a token"

    async def test_health_reports_enforcement_state(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (await client.get("/health")).json()["auth_enforced"] is False
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        assert (await client.get("/health")).json()["auth_enforced"] is True

    async def test_every_protected_operation_declares_the_guard(self, app) -> None:
        """Catches a future router included without the auth dependency.

        Reads the generated OpenAPI schema rather than ``app.routes``: the
        session-scoped fixtures do not expose mounted router routes through
        that attribute, so an introspection-based check silently passes over an
        empty list.
        """
        paths = app.openapi()["paths"]
        protected = {
            path: ops
            for path, ops in paths.items()
            if path.startswith(PROTECTED_PREFIXES)
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

    async def test_open_operations_do_not_declare_the_guard(self, app) -> None:
        paths = app.openapi()["paths"]
        leaked = [
            f"{method.upper()} {path}"
            for path in OPEN_PATHS
            for method, op in paths.get(path, {}).items()
            if method in {"get", "post"} and op.get("security")
        ]
        assert not leaked, f"probe routes must stay open: {leaked}"


class TestBrowserSessionExchange:
    """POST /auth/session lets a browser obtain a usable credential (#973)."""

    async def test_exchange_is_conflict_when_auth_is_off(self, auth_client) -> None:
        """Nothing to establish when the service takes no credential."""
        response = await auth_client.post("/auth/session")
        assert response.status_code == 409

    async def test_exchange_rejects_a_wrong_token(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await auth_client.post(
            "/auth/session", headers={"Authorization": "Bearer wrong"}
        )
        assert response.status_code == 401
        assert "set-cookie" not in response.headers

    async def test_exchange_requires_a_bearer_scheme(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await auth_client.post(
            "/auth/session", headers={"Authorization": "s3cret"}
        )
        assert response.status_code == 401

    async def test_exchange_sets_a_hardened_cookie(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await auth_client.post(
            "/auth/session", headers={"Authorization": "Bearer s3cret"}
        )
        assert response.status_code == 204
        raw = response.headers["set-cookie"]
        assert "HttpOnly" in raw
        assert "SameSite=strict" in raw.replace("samesite", "SameSite")
        # The token must never travel in the cookie value.
        assert "s3cret" not in raw

    async def test_cookie_then_authenticates_a_protected_route(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: no Authorization header on the follow-up request."""
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        assert (await auth_client.get(PROBE)).status_code == 401

        exchange = await auth_client.post(
            "/auth/session", headers={"Authorization": "Bearer s3cret"}
        )
        assert exchange.status_code == 204

        after = await auth_client.get(PROBE)
        assert after.status_code != 401

    async def test_logout_clears_the_cookie(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        await auth_client.post(
            "/auth/session", headers={"Authorization": "Bearer s3cret"}
        )
        assert (await auth_client.get(PROBE)).status_code != 401

        assert (await auth_client.delete("/auth/session")).status_code == 204
        assert (await auth_client.get(PROBE)).status_code == 401

    async def test_cookie_dies_with_token_rotation(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cookie must not outlive the credential it was derived from."""
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        await auth_client.post(
            "/auth/session", headers={"Authorization": "Bearer s3cret"}
        )
        assert (await auth_client.get(PROBE)).status_code != 401

        monkeypatch.setenv(TOKEN_ENV, "rotated")
        assert (await auth_client.get(PROBE)).status_code == 401

    async def test_exchange_endpoint_is_not_itself_protected(self, app) -> None:
        """It must stay open, or a browser could never obtain a credential."""
        paths = app.openapi()["paths"]
        assert not paths["/auth/session"]["post"].get("security")

    async def test_secure_flag_is_set_over_https(
        self, app, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct TLS: request.url.scheme is https."""
        import httpx

        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://secure.test"
        ) as client:
            response = await client.post(
                "/auth/session", headers={"Authorization": "Bearer s3cret"}
            )
        assert response.status_code == 204
        assert "Secure" in response.headers["set-cookie"]

    async def test_secure_flag_survives_upstream_tls_termination(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ingress and NVCF terminate TLS, so the scheme reaching us is http.

        Without honouring X-Forwarded-Proto the cookie would ship without
        Secure on every real HTTPS deployment.
        """
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await auth_client.post(
            "/auth/session",
            headers={
                "Authorization": "Bearer s3cret",
                "X-Forwarded-Proto": "https,http",
            },
        )
        assert response.status_code == 204
        assert "Secure" in response.headers["set-cookie"]

    async def test_secure_flag_is_omitted_on_plain_http(
        self, auth_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise localhost development could never hold a session."""
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        response = await auth_client.post(
            "/auth/session", headers={"Authorization": "Bearer s3cret"}
        )
        assert response.status_code == 204
        assert "Secure" not in response.headers["set-cookie"]
