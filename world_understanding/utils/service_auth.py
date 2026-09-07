# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional bearer-token enforcement shared by the agent REST services.

The Material, Physics, Joint, and Texture service clients have long read a
``<SERVICE>_AGENT_TOKEN`` environment variable and attached it as an
``Authorization: Bearer`` header, but no server ever validated it. The OpenAPI
specs advertised a ``bearerAuth`` scheme that did not exist. This module closes
that gap (GitHub issue #956).

Enforcement is **opt-in and per-request**:

* When the token variable is unset, empty, or whitespace-only, every request is
  accepted exactly as before. Existing Docker Compose, Helm, and NVCF
  deployments are unaffected until an operator sets the variable.
* When it is set on the **server**, requests to protected routes must present a
  matching bearer token or receive ``401``.

The value is read from the environment inside the dependency rather than
captured into a settings object at import time. Two reasons: the service
settings classes use prefixed environment variables (``MA_``, ``PA_``, ``JA_``,
``TA_``), so a settings field would bind ``MA_MATERIAL_AGENT_TOKEN`` rather than
the unprefixed name the clients already send; and the service test suites build
the app once per session, so a per-request read is what lets a test toggle
enforcement.

This is a single shared static secret. It closes the "advertised but unenforced
control" gap; it does not provide per-caller identity, rotation, rate limiting,
or audit, and it is not a replacement for gateway authentication.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections.abc import Callable, Sequence

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

__all__ = [
    "AUTH_REQUIRED_ENV",
    "SESSION_COOKIE_MAX_AGE",
    "SESSION_COOKIE_NAME",
    "auth_is_enforced",
    "build_token_dependency",
    "log_auth_posture",
    "mint_session_cookie",
    "request_is_same_origin",
    "request_origin_is_foreign",
    "request_scheme",
    "resolve_expected_token",
    "verify_session_cookie",
]

# Opt-in strictness: when set to a truthy value, a service refuses to start
# unless a token is configured. Off by default because every shipped deployment
# binds 0.0.0.0 without a token, so defaulting to fail-closed would refuse to
# start the entire installed base.
AUTH_REQUIRED_ENV = "WU_SERVICE_AUTH_REQUIRED"

# Browser clients cannot attach an Authorization header to EventSource streams,
# <img> sources, or top-level download navigations. A cookie is the only
# credential the browser sends on all of those, so services may offer a token
# exchange that mints one. See mint_session_cookie for the value format.
SESSION_COOKIE_NAME = "wu_service_session"
SESSION_COOKIE_MAX_AGE = 12 * 60 * 60

_COOKIE_VERSION = "v1"
_COOKIE_KEY_CONTEXT = b"wu-service-auth-cookie-v1|"

# Methods that cannot change state, so they need no cross-origin check.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_expected_token(env_names: Sequence[str]) -> str | None:
    """Return the configured bearer token, or ``None`` when auth is disabled.

    Whitespace-only and empty values count as unset. This matters because
    Docker Compose ``${VAR:-}`` passthrough and Helm ``value: ""`` defaults both
    deliver an empty string rather than an absent variable; treating those as
    "enforcement on" would reject every request with an unsatisfiable token.
    """
    for name in env_names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        token = raw.strip()
        if token:
            return token
    return None


def auth_is_enforced(env_names: Sequence[str]) -> bool:
    """Return whether a token is configured for these variable names."""
    return resolve_expected_token(env_names) is not None


def _tokens_match(supplied: str, expected: str) -> bool:
    """Constant-time comparison that tolerates non-ASCII tokens.

    ``secrets.compare_digest`` raises ``TypeError`` for ``str`` arguments
    containing non-ASCII characters, which would surface as a 500 and turn the
    guard into a crash oracle. Comparing UTF-8 bytes avoids that.
    """
    return secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _cookie_key(token: str, service_label: str) -> bytes:
    """Derive the cookie signing key from the service token.

    Deriving rather than storing means a cookie stops verifying the moment the
    token is rotated or cleared, and a leaked cookie can never be replayed as a
    bearer credential because the token is not recoverable from it.
    """
    return hmac.new(
        token.encode("utf-8"),
        _COOKIE_KEY_CONTEXT + service_label.encode("utf-8"),
        hashlib.sha256,
    ).digest()


def mint_session_cookie(
    token: str,
    *,
    service_label: str,
    max_age: int = SESSION_COOKIE_MAX_AGE,
    now: float | None = None,
) -> str:
    """Return a signed, expiring cookie value derived from ``token``.

    Format: ``v1.<expiry>.<nonce>.<signature>``. The token itself never appears
    in the value. The nonce makes each mint distinct so a value cannot be
    correlated across sessions.
    """
    expiry = int((time.time() if now is None else now) + max_age)
    nonce = _b64(secrets.token_bytes(16))
    payload = f"{_COOKIE_VERSION}.{expiry}.{nonce}"
    key = _cookie_key(token, service_label)
    signature = _b64(hmac.new(key, payload.encode("utf-8"), hashlib.sha256).digest())
    return f"{payload}.{signature}"


def verify_session_cookie(
    value: str,
    token: str,
    *,
    service_label: str,
    now: float | None = None,
) -> bool:
    """Return whether ``value`` is a live cookie minted from ``token``.

    Returns False rather than raising for every malformed input: this runs on
    attacker-controlled data, and an exception here would be a 500 instead of a
    401.
    """
    parts = value.split(".")
    if len(parts) != 4 or parts[0] != _COOKIE_VERSION:
        return False

    version, expiry_raw, nonce, signature = parts
    payload = f"{version}.{expiry_raw}.{nonce}"
    key = _cookie_key(token, service_label)
    expected = _b64(hmac.new(key, payload.encode("utf-8"), hashlib.sha256).digest())

    # Compare before parsing the expiry so a forged value cannot be
    # distinguished from an expired one by response timing.
    if not secrets.compare_digest(
        signature.encode("ascii", "ignore"), expected.encode("ascii")
    ):
        return False

    try:
        expiry = int(expiry_raw)
    except ValueError:
        return False

    return expiry > (time.time() if now is None else now)


def request_scheme(request: Request) -> str:
    """The client-facing scheme, honouring an upstream TLS terminator.

    Read from ``X-Forwarded-Proto`` directly rather than relying on uvicorn's
    ``--proxy-headers`` / ``FORWARDED_ALLOW_IPS`` handling, so the result does
    not depend on deployment-specific server flags.
    """
    if request.url.scheme == "https":
        return "https"
    forwarded = request.headers.get("x-forwarded-proto", "")
    # The header may carry a comma-separated chain; the first entry is the
    # client-facing protocol.
    if forwarded.split(",")[0].strip().lower() == "https":
        return "https"
    return request.url.scheme or "http"


def _declared_origin(request: Request) -> str | None:
    """The origin the client claims to be on, or ``None`` if it declared none.

    ``Origin`` is preferred; ``Referer`` is the fallback because browsers omit
    ``Origin`` on same-origin GETs and on ``<img>`` loads, but do send a
    ``Referer``.
    """
    declared = request.headers.get("origin")
    if declared and declared != "null":
        return declared.strip().lower().rstrip("/")

    referer = request.headers.get("referer")
    if not referer:
        return None
    scheme, sep, rest = referer.partition("://")
    if not sep or not rest:
        return None
    host = rest.split("/", 1)[0]
    if not host:
        return None
    return f"{scheme}://{host}".strip().lower()


def _own_origin(request: Request) -> str | None:
    host = request.headers.get("host")
    if not host:
        return None
    return f"{request_scheme(request)}://{host}".strip().lower()


def request_is_same_origin(request: Request) -> bool:
    """Whether the client declared an origin, and it is this service's own.

    ``SameSite=Strict`` is a same-*site* control and the port is not part of a
    site: ``http://localhost:3000`` and ``http://localhost:8000`` are same-site,
    so the browser attaches the cookie between them. This repo's own Vite dev
    server runs on port 3000, so cookie-authenticated state-changing requests
    need an explicit origin check on top of ``SameSite``.

    The comparison includes the **scheme**: ``http://svc`` and ``https://svc``
    are different origins, so a page served over plaintext on the same host
    must not drive a request the cookie authenticates.

    A request that declares nothing is *not* same-origin. Browsers send
    ``Origin`` on non-GET requests, and a non-browser caller should use the
    bearer header, which is exempt from this check.
    """
    own = _own_origin(request)
    declared = _declared_origin(request)
    return own is not None and declared is not None and declared == own


def request_origin_is_foreign(request: Request) -> bool:
    """Whether the client declared an origin that is *not* this service's.

    Distinct from ``not request_is_same_origin(...)``: this is False when the
    client declared nothing. Used for safe methods, where a cross-origin
    ``<img>`` carries a ``Referer`` that can be rejected, while a same-origin
    load that simply omits both headers must still be allowed.
    """
    declared = _declared_origin(request)
    if declared is None:
        return False
    own = _own_origin(request)
    return own is None or declared != own


def build_token_dependency(
    env_names: Sequence[str],
    *,
    service_label: str,
    cookie_name: str = SESSION_COOKIE_NAME,
) -> Callable[..., None]:
    """Build a FastAPI dependency enforcing an optional bearer token.

    Attach it to routers that should be protected::

        app.include_router(pipeline_router.router, dependencies=[Depends(dep)])

    Deliberately a route dependency rather than middleware: the services assert
    on ``app.user_middleware[0]``, and ``add_middleware`` prepends, so an auth
    middleware would displace that entry. It is attached at ``include_router``
    rather than inside the ``APIRouter`` constructors so that tests which mount
    routers onto their own bare app keep working.
    """
    env_names = tuple(env_names)
    # auto_error=False so a missing header reaches us as None and we can emit a
    # 401 with WWW-Authenticate rather than FastAPI's bare 403.
    scheme = HTTPBearer(auto_error=False, description=f"{service_label} bearer token")

    def require_service_token(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Security(scheme),
    ) -> None:
        expected = resolve_expected_token(env_names)
        if expected is None:
            return

        supplied: str | None = None
        if credentials is not None and credentials.scheme.lower() == "bearer":
            supplied = credentials.credentials
        if supplied is not None and _tokens_match(supplied, expected):
            return

        # Browser fallback. Only consulted when no valid bearer header was
        # presented, so it can never weaken the header path.
        cookie = request.cookies.get(cookie_name)
        if cookie and verify_session_cookie(
            cookie, expected, service_label=service_label
        ):
            if request.method.upper() in _SAFE_METHODS:
                # Safe methods must stay reachable for <img> and EventSource,
                # which cannot send an Authorization header. But some protected
                # GETs do have side effects (report generation reserves cache
                # runs and writes lineage), so reject a request that positively
                # declares a foreign origin. A cross-origin <img> carries a
                # Referer; a same-origin load that declares nothing is allowed.
                if not request_origin_is_foreign(request):
                    return
            elif request_is_same_origin(request):
                return
            # A valid cookie from a foreign origin. SameSite=Strict does not
            # stop this because same-site includes other ports on the same
            # host. Reject rather than fall through to the 401, so the cause is
            # not mistaken for a bad credential.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Cross-origin request rejected. Cookie credentials are "
                    "accepted only from this service's own origin; use an "
                    "Authorization: Bearer header instead."
                ),
            )

        # Never include the supplied or expected value in the response or in
        # logs.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing or invalid credentials. This service has "
                f"{env_names[0]} configured, so requests must send an "
                "Authorization: Bearer header, or a session cookie obtained "
                "from the token-exchange endpoint."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return require_service_token


def log_auth_posture(
    logger: logging.Logger,
    env_names: Sequence[str],
    *,
    service_label: str,
) -> None:
    """Record whether token enforcement is active, and honour opt-in strictness.

    Raises ``RuntimeError`` when :data:`AUTH_REQUIRED_ENV` is truthy and no
    token is configured, so an operator who wants fail-closed behaviour can ask
    for it explicitly without that becoming the default for everyone else.
    """
    env_names = tuple(env_names)
    primary = env_names[0]
    required = os.environ.get(AUTH_REQUIRED_ENV, "").strip().lower() in _TRUTHY

    if auth_is_enforced(env_names):
        logger.info(
            "%s: bearer-token enforcement is ACTIVE (%s is set).",
            service_label,
            primary,
        )
        return

    if required:
        raise RuntimeError(
            f"{AUTH_REQUIRED_ENV} is set but {primary} is not configured; "
            f"refusing to start {service_label} without bearer-token "
            "enforcement."
        )

    logger.warning(
        "%s: no bearer-token enforcement (%s is not set). The service accepts "
        "unauthenticated requests and must run behind a trusted network "
        "boundary that terminates authentication. Set %s to require a token.",
        service_label,
        primary,
        primary,
    )
