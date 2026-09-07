# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared optional bearer-token service guard (issue #956)."""

from __future__ import annotations

import base64
import contextlib
import logging
import secrets
import time

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient

from world_understanding.utils.service_auth import (
    AUTH_REQUIRED_ENV,
    SESSION_COOKIE_NAME,
    auth_is_enforced,
    build_token_dependency,
    log_auth_posture,
    mint_session_cookie,
    request_is_same_origin,
    request_origin_is_foreign,
    request_scheme,
    resolve_expected_token,
    verify_session_cookie,
)

ENV = "TEST_AGENT_TOKEN"
ALT_ENV = "TEST_AGENT_TOKEN_ALT"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (ENV, ALT_ENV, AUTH_REQUIRED_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client() -> TestClient:
    """An app whose protected router carries the guard, plus an open route."""
    app = FastAPI()
    dependency = build_token_dependency([ENV], service_label="Test Service")

    router = APIRouter(prefix="/protected")

    @router.get("/thing")
    def _thing() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(router, dependencies=[Depends(dependency)])

    @app.get("/health")
    def _health() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


# --- token resolution ------------------------------------------------------


def test_absent_variable_disables_enforcement() -> None:
    assert resolve_expected_token([ENV]) is None
    assert auth_is_enforced([ENV]) is False


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_variable_counts_as_unset(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Compose ``${VAR:-}`` and Helm ``value: ""`` deliver empty strings.

    Treating those as enforcement-on would 401 every request in a deployment
    nobody meant to secure.
    """
    monkeypatch.setenv(ENV, value)
    assert resolve_expected_token([ENV]) is None
    assert auth_is_enforced([ENV]) is False


def test_value_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV, "  s3cret  ")
    assert resolve_expected_token([ENV]) == "s3cret"


def test_names_are_tried_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALT_ENV, "from-alt")
    assert resolve_expected_token([ENV, ALT_ENV]) == "from-alt"
    monkeypatch.setenv(ENV, "from-primary")
    assert resolve_expected_token([ENV, ALT_ENV]) == "from-primary"


# --- disabled (backward compatibility) -------------------------------------


def test_disabled_allows_request_without_header(client: TestClient) -> None:
    assert client.get("/protected/thing").status_code == 200


def test_disabled_ignores_a_bogus_header(client: TestClient) -> None:
    """The case that matters most: clients already send tokens today.

    With no server token configured, a stale or wrong client token must not
    start failing.
    """
    response = client.get(
        "/protected/thing", headers={"Authorization": "Bearer nonsense"}
    )
    assert response.status_code == 200


# --- enabled ---------------------------------------------------------------


def test_correct_token_is_accepted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    response = client.get(
        "/protected/thing", headers={"Authorization": "Bearer s3cret"}
    )
    assert response.status_code == 200


def test_missing_header_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    response = client.get("/protected/thing")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "header",
    [
        "Bearer wrong",
        "Bearer ",
        "Basic abc",
        "s3cret",
        "",
    ],
)
def test_malformed_or_wrong_credentials_are_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    response = client.get("/protected/thing", headers={"Authorization": header})
    assert response.status_code == 401


def test_open_route_stays_reachable_under_enforcement(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Liveness probes send no Authorization header."""
    monkeypatch.setenv(ENV, "s3cret")
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_token_is_read_per_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service test suites build the app once per session."""
    assert client.get("/protected/thing").status_code == 200
    monkeypatch.setenv(ENV, "s3cret")
    assert client.get("/protected/thing").status_code == 401
    monkeypatch.delenv(ENV)
    assert client.get("/protected/thing").status_code == 200


def test_non_ascii_configured_token_rejects_without_raising(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-ASCII *configured* token must yield 401, not a 500 crash oracle.

    ``secrets.compare_digest`` raises ``TypeError`` when given ``str`` values
    containing non-ASCII characters, so comparing the raw strings would turn an
    operator's accented password into a 500 on every request. Pins the UTF-8
    encoding in ``_tokens_match``.

    Only the configured side can be non-ASCII: HTTP header values must be
    ASCII/latin-1, so an HTTP client cannot transmit one as a credential.
    """
    monkeypatch.setenv(ENV, "pässwörd")
    response = client.get("/protected/thing", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_comparison_is_constant_time(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    calls: list[tuple[object, object]] = []
    real = secrets.compare_digest

    def _spy(a: object, b: object) -> bool:
        calls.append((a, b))
        return real(a, b)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "world_understanding.utils.service_auth.secrets.compare_digest", _spy
    )
    client.get("/protected/thing", headers={"Authorization": "Bearer s3cret"})
    assert calls, "compare_digest was not used; a plain == is a timing oracle"
    assert all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in calls)


def test_token_never_appears_in_the_response(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "sup3rs3cret")
    response = client.get("/protected/thing", headers={"Authorization": "Bearer wrong"})
    assert "sup3rs3cret" not in response.text
    assert "wrong" not in response.text


# --- startup posture -------------------------------------------------------


def test_posture_warns_when_unset(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test.posture.warn")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        log_auth_posture(logger, [ENV], service_label="Test Service")
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_posture_logs_info_when_set(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    logger = logging.getLogger("test.posture.info")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_auth_posture(logger, [ENV], service_label="Test Service")
    assert any(r.levelno == logging.INFO for r in caplog.records)
    assert all("s3cret" not in r.getMessage() for r in caplog.records)


def test_posture_can_fail_closed_on_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AUTH_REQUIRED_ENV, "1")
    logger = logging.getLogger("test.posture.required")
    with pytest.raises(RuntimeError, match=AUTH_REQUIRED_ENV):
        log_auth_posture(logger, [ENV], service_label="Test Service")


def test_posture_satisfied_when_required_and_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUTH_REQUIRED_ENV, "true")
    monkeypatch.setenv(ENV, "s3cret")
    logger = logging.getLogger("test.posture.both")
    log_auth_posture(logger, [ENV], service_label="Test Service")


def test_posture_ignores_non_truthy_required_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUTH_REQUIRED_ENV, "0")
    logger = logging.getLogger("test.posture.off")
    log_auth_posture(logger, [ENV], service_label="Test Service")


# --- session cookie (issue #973) -------------------------------------------


LABEL = "Test Service"


def test_cookie_roundtrips() -> None:
    value = mint_session_cookie("s3cret", service_label=LABEL)
    assert verify_session_cookie(value, "s3cret", service_label=LABEL)


def test_cookie_never_contains_the_token() -> None:
    value = mint_session_cookie("sup3rs3cret", service_label=LABEL)
    assert "sup3rs3cret" not in value
    # ...nor recoverable by decoding any component.
    for part in value.split("."):
        with contextlib.suppress(Exception):
            pad = part + "=" * (-len(part) % 4)
            assert b"sup3rs3cret" not in base64.urlsafe_b64decode(pad)


def test_cookie_is_rejected_after_token_rotation() -> None:
    """The signing key is derived from the token, so rotation invalidates it.

    This is the property that stops a leaked cookie outliving the credential it
    came from.
    """
    value = mint_session_cookie("old-token", service_label=LABEL)
    assert not verify_session_cookie(value, "new-token", service_label=LABEL)


def test_cookie_is_scoped_to_the_service() -> None:
    """A cookie from one service must not authenticate against another."""
    value = mint_session_cookie("shared", service_label="Material Agent Service")
    assert not verify_session_cookie(
        value, "shared", service_label="Physics Agent Service"
    )


def test_expired_cookie_is_rejected() -> None:
    value = mint_session_cookie("s3cret", service_label=LABEL, max_age=60)
    assert verify_session_cookie(value, "s3cret", service_label=LABEL)
    later = time.time() + 61
    assert not verify_session_cookie(value, "s3cret", service_label=LABEL, now=later)


def test_expiry_cannot_be_extended_without_the_key() -> None:
    """Tampering with the expiry must invalidate the signature."""
    value = mint_session_cookie("s3cret", service_label=LABEL, max_age=60)
    version, expiry, nonce, sig = value.split(".")
    forged = f"{version}.{int(expiry) + 10_000}.{nonce}.{sig}"
    assert not verify_session_cookie(forged, "s3cret", service_label=LABEL)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "garbage",
        "v1.1.2",
        "v1.1.2.3.4",
        "v2.9999999999.abc.def",
        "v1.notanint.abc.def",
        "v1..abc.def",
        "....",
        "v1.9999999999.abc.é",
    ],
)
def test_malformed_cookies_return_false_and_never_raise(value: str) -> None:
    assert verify_session_cookie(value, "s3cret", service_label=LABEL) is False


def test_each_mint_is_distinct() -> None:
    a = mint_session_cookie("s3cret", service_label=LABEL)
    b = mint_session_cookie("s3cret", service_label=LABEL)
    assert a != b


def test_cookie_authenticates_a_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    assert client.get("/protected/thing").status_code == 401
    client.cookies.set(
        SESSION_COOKIE_NAME, mint_session_cookie("s3cret", service_label="Test Service")
    )
    assert client.get("/protected/thing").status_code == 200


def test_forged_cookie_does_not_authenticate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    client.cookies.set(
        SESSION_COOKIE_NAME, mint_session_cookie("wrong", service_label="Test Service")
    )
    assert client.get("/protected/thing").status_code == 401


def test_cookie_is_ignored_when_enforcement_is_off(client: TestClient) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, "anything-at-all")
    assert client.get("/protected/thing").status_code == 200


# --- same-origin enforcement for cookie credentials ------------------------


@pytest.fixture
def write_client() -> TestClient:
    """An app with a state-changing route behind the guard."""
    app = FastAPI()
    dependency = build_token_dependency([ENV], service_label="Test Service")
    router = APIRouter(prefix="/protected")

    @router.post("/write")
    def _write() -> dict[str, bool]:
        return {"ok": True}

    @router.get("/read")
    def _read() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(router, dependencies=[Depends(dependency)])
    return TestClient(app, base_url="http://svc.test")


def _cookie() -> str:
    return mint_session_cookie("s3cret", service_label="Test Service")


def test_cookie_write_from_own_origin_is_allowed(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    write_client.cookies.set(SESSION_COOKIE_NAME, _cookie())
    response = write_client.post(
        "/protected/write", headers={"Origin": "http://svc.test"}
    )
    assert response.status_code == 200


def test_cookie_write_from_another_port_is_rejected(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case SameSite=Strict does NOT cover.

    A site is registrable-domain-plus-scheme; the port is not part of it. So a
    page on http://svc.test:3000 is same-site with http://svc.test and the
    browser attaches the cookie. Only an explicit origin check stops the write.
    """
    monkeypatch.setenv(ENV, "s3cret")
    write_client.cookies.set(SESSION_COOKIE_NAME, _cookie())
    response = write_client.post(
        "/protected/write", headers={"Origin": "http://svc.test:3000"}
    )
    assert response.status_code == 403


def test_cookie_write_without_an_origin_is_rejected(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    write_client.cookies.set(SESSION_COOKIE_NAME, _cookie())
    assert write_client.post("/protected/write").status_code == 403


def test_cookie_write_accepts_a_matching_referer(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "s3cret")
    write_client.cookies.set(SESSION_COOKIE_NAME, _cookie())
    response = write_client.post(
        "/protected/write", headers={"Referer": "http://svc.test/app/page"}
    )
    assert response.status_code == 200


def test_cookie_read_needs_no_origin(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """<img> and EventSource are GETs and must keep working."""
    monkeypatch.setenv(ENV, "s3cret")
    write_client.cookies.set(SESSION_COOKIE_NAME, _cookie())
    assert write_client.get("/protected/read").status_code == 200


def test_bearer_write_is_not_subject_to_the_origin_check(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cross-origin attacker cannot set an Authorization header anyway."""
    monkeypatch.setenv(ENV, "s3cret")
    response = write_client.post(
        "/protected/write",
        headers={"Authorization": "Bearer s3cret", "Origin": "http://evil.test"},
    )
    assert response.status_code == 200


def test_origin_rejection_is_403_not_401(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A distinct code so the cause is not mistaken for a bad credential."""
    monkeypatch.setenv(ENV, "s3cret")
    write_client.cookies.set(SESSION_COOKIE_NAME, _cookie())
    response = write_client.post(
        "/protected/write", headers={"Origin": "http://evil.test"}
    )
    assert response.status_code == 403
    assert "Cross-origin" in response.json()["detail"]


# --- defensive branches ----------------------------------------------------


def _request(headers: dict[str, str], scheme: str = "http") -> Request:
    """Build a bare Starlette request with exactly these headers."""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "scheme": scheme,
            "headers": [
                (k.lower().encode("latin-1"), v.encode("latin-1"))
                for k, v in headers.items()
            ],
        }
    )


def test_same_origin_requires_a_host_header() -> None:
    assert request_is_same_origin(_request({"origin": "http://svc.test"})) is False


def test_same_origin_rejects_a_referer_without_an_origin() -> None:
    for referer in ("", "/relative/path", "://"):
        assert (
            request_is_same_origin(_request({"host": "svc.test", "referer": referer}))
            is False
        )


def test_same_origin_accepts_a_matching_origin() -> None:
    request = _request({"host": "svc.test", "origin": "http://svc.test"})
    assert request_is_same_origin(request) is True


def test_same_origin_compares_the_scheme_too() -> None:
    """http://svc and https://svc are different origins.

    Comparing only the host would let a page served over plaintext on the same
    host drive a request the session cookie authenticates.
    """
    request = _request({"host": "svc.test", "origin": "https://svc.test"})
    assert request_is_same_origin(request) is False


def test_forwarded_proto_defines_the_expected_scheme() -> None:
    """Behind a TLS terminator the service's own origin is https."""
    request = _request(
        {
            "host": "svc.test",
            "origin": "https://svc.test",
            "x-forwarded-proto": "https",
        }
    )
    assert request_is_same_origin(request) is True


def test_foreign_origin_predicate_ignores_an_undeclared_origin() -> None:
    """A same-origin <img> may send neither Origin nor Referer."""
    assert request_origin_is_foreign(_request({"host": "svc.test"})) is False
    assert (
        request_origin_is_foreign(
            _request({"host": "svc.test", "origin": "http://evil.test"})
        )
        is True
    )


def test_signed_cookie_with_a_non_integer_expiry_is_rejected() -> None:
    """The signature is checked before the expiry is parsed.

    That ordering means this branch is unreachable with an unsigned value, so
    the test forges a *correctly signed* payload carrying a non-numeric expiry
    to prove the parse failure returns False rather than raising.
    """
    import hashlib
    import hmac

    from world_understanding.utils.service_auth import _b64, _cookie_key

    payload = "v1.notanint.nonce"
    key = _cookie_key("s3cret", "Test Service")
    signature = _b64(hmac.new(key, payload.encode("utf-8"), hashlib.sha256).digest())
    assert (
        verify_session_cookie(
            f"{payload}.{signature}", "s3cret", service_label="Test Service"
        )
        is False
    )


def test_cookie_read_from_a_foreign_origin_is_rejected(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some protected GETs have side effects.

    Report generation reserves cache runs and writes lineage, so a
    cross-origin <img> pointed at one must not be authenticated by the cookie.
    A cross-origin image load carries a Referer even though it sends no Origin.
    """
    monkeypatch.setenv(ENV, "s3cret")
    write_client.cookies.set(SESSION_COOKIE_NAME, _cookie())
    response = write_client.get(
        "/protected/read", headers={"Referer": "http://evil.test/page"}
    )
    assert response.status_code == 403


def test_cookie_read_without_any_declared_origin_is_allowed(
    write_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-origin <img> and EventSource may declare nothing at all."""
    monkeypatch.setenv(ENV, "s3cret")
    write_client.cookies.set(SESSION_COOKIE_NAME, _cookie())
    assert write_client.get("/protected/read").status_code == 200


def test_request_scheme_reports_direct_tls() -> None:
    assert request_scheme(_request({"host": "svc.test"}, scheme="https")) == "https"


def test_request_scheme_falls_back_to_the_transport_scheme() -> None:
    assert request_scheme(_request({"host": "svc.test"})) == "http"


def test_referer_with_an_empty_host_declares_no_origin() -> None:
    """``http:///path`` parses into a scheme and a path but no authority."""
    request = _request({"host": "svc.test", "referer": "http:///path"})
    assert request_is_same_origin(request) is False
    assert request_origin_is_foreign(request) is False


def test_null_origin_is_treated_as_undeclared() -> None:
    """Sandboxed iframes and some redirects send ``Origin: null``.

    Treating the literal string as an origin would let it be compared, and it
    can never legitimately match this service.
    """
    request = _request({"host": "svc.test", "origin": "null"})
    assert request_is_same_origin(request) is False
    # Both halves matter. "not same-origin" alone would also hold if null were
    # classified as foreign, which would 403 protected reads from a sandboxed
    # iframe instead of letting them through as undeclared.
    assert request_origin_is_foreign(request) is False
