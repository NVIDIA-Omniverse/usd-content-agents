# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared credential resolution helpers."""

from __future__ import annotations

import ast
import asyncio
import ipaddress
import json
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import parse_qsl, unquote, urlparse

API_KEY_ENV_VAR_MAP: dict[str, tuple[str, ...]] = {
    "nim": ("NVIDIA_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def register_api_key_env_vars(backend: str, *env_vars: str) -> None:
    """Register credential aliases supplied by an optional backend plugin."""
    if not backend or not env_vars or any(not env_var for env_var in env_vars):
        raise ValueError("backend and env var names must be non-empty")
    API_KEY_ENV_VAR_MAP[backend] = tuple(env_vars)


LOCAL_NIM_API_KEY_PLACEHOLDER = "not-used"
_EXPLICIT_API_KEY_PLACEHOLDER_VALUES = {
    LOCAL_NIM_API_KEY_PLACEHOLDER,
    "your-api-key",
    "your-api-key-here",
    "your-account-key",
    "your-account-key-here",
    "your-google-api-key",
    "your-key",
    "your-key-here",
    "your-ngc-api-key",
    "your-nvcf-api-key",
    "your-nvidia-api-key",
    "your-shared-access-key",
    "your-shared-access-key-here",
    "your_actual_nvcf_api_key",
    "your_account_key",
    "your_account_key_here",
    "your_anthropic_api_key_here",
    "your_api_key",
    "your_api_key_here",
    "your_embedding_api_key_here",
    "your_nvidia_api_key",
    "your_nvidia_api_key_here",
    "your_openai_api_key",
    "your_openai_api_key_here",
    "your_shared_access_key",
    "your_shared_access_key_here",
    "your_anthropic_api_key",
    "your_google_api_key",
    "your_google_api_key_here",
    "your_gemini_api_key",
    "your_image_api_key_here",
    "your_ngc_api_key",
    "your_ngc_api_key_here",
    "your_nim_api_key",
    "your_nvcf_api_key",
    "your_nvcf_api_key_here",
    "replace_me",
    "changeme",
    "todo",
}
_NVIDIA_HOST_SUFFIXES = ("nvidia.com",)
_OPENAI_HOST_SUFFIXES = ("openai.com",)
_LOCAL_HOSTNAMES = {"localhost", "host.docker.internal", "gateway.docker.internal"}

OPENAI_ENV_REDIRECT_CREDENTIAL_MESSAGE = (
    "OPENAI_API_KEY was intentionally not forwarded because OPENAI_BASE_URL or "
    "OPENAI_API_BASE selects a non-OpenAI endpoint. Pair an endpoint-scoped "
    "api_key or api_key_env with base_url in config; documented local no-auth "
    "endpoints may use api_key: not-used."
)


class InlineSecretError(ValueError):
    """A durable value contains an inline credential.

    This subtype lets persistence boundaries distinguish a rejected credential
    from unrelated validation or serialization failures while preserving the
    existing ``ValueError`` compatibility contract.
    """

    def __init__(self, message: str, *, paths: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        # Paths are produced by the bounded diagnostic-path renderer. They do
        # not contain credential values and let persistence boundaries retain
        # a useful operator signal without parsing exception text.
        self.paths = paths[:8]


def _raise_missing_endpoint_env() -> NoReturn:
    """Raise a missing-env diagnostic from a frame with no configured value."""
    raise ValueError(
        "configured API key environment variable is not set or empty"
    ) from None


CREDENTIAL_SCAN_LIMIT_MESSAGE = "Credential safety scan exceeded its fixed work limits"


class CredentialScanLimitError(InlineSecretError):
    """The input could not be proven safe within fixed scanner work limits."""


@dataclass(frozen=True)
class CredentialScanLimits:
    """Total work limits for one durable credential scan.

    Limits cover the complete object graph, not each individual scalar. Zero is
    accepted for deterministic fail-closed tests; negative limits are invalid.
    """

    max_nodes: int = 100_000
    max_bytes: int = 8 * 1024 * 1024
    max_decode_work: int = 64 * 1024 * 1024
    max_uri_candidates: int = 16_384
    max_query_branches: int = 32_768
    max_container_depth: int = 64

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.max_nodes,
                self.max_bytes,
                self.max_decode_work,
                self.max_uri_candidates,
                self.max_query_branches,
                self.max_container_depth,
            )
        ):
            raise ValueError("credential scan limits must be nonnegative")


DEFAULT_CREDENTIAL_SCAN_LIMITS = CredentialScanLimits()


class _ScanBudget:
    """Mutable accounting for one scanner invocation."""

    __slots__ = (
        "bytes",
        "decode_work",
        "limits",
        "nodes",
        "query_branches",
        "uri_candidates",
    )

    def __init__(self, limits: CredentialScanLimits) -> None:
        self.limits = limits
        self.nodes = 0
        self.bytes = 0
        self.decode_work = 0
        self.uri_candidates = 0
        self.query_branches = 0

    @staticmethod
    def _consume(current: int, amount: int, limit: int) -> int:
        if amount < 0 or amount > limit - current:
            raise CredentialScanLimitError(CREDENTIAL_SCAN_LIMIT_MESSAGE)
        return current + amount

    def consume_node(self, *, depth: int) -> None:
        if depth > self.limits.max_container_depth:
            raise CredentialScanLimitError(CREDENTIAL_SCAN_LIMIT_MESSAGE)
        self.nodes = self._consume(self.nodes, 1, self.limits.max_nodes)

    def consume_scalar_bytes(self, value: Any) -> None:
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, str):
            size = len(value.encode("utf-8", errors="surrogatepass"))
        elif isinstance(value, bytes | bytearray | memoryview):
            size = len(value)
        elif type(value) is int:
            size = max(1, (value.bit_length() + 7) // 8)
        else:
            return
        self.bytes = self._consume(self.bytes, size, self.limits.max_bytes)

    def consume_decode(self, text: str) -> None:
        self.decode_work = self._consume(
            self.decode_work,
            len(text),
            self.limits.max_decode_work,
        )

    def consume_uri_candidate(self) -> None:
        self.uri_candidates = self._consume(
            self.uri_candidates,
            1,
            self.limits.max_uri_candidates,
        )

    def consume_query_branches(self, count: int) -> None:
        self.query_branches = self._consume(
            self.query_branches,
            count,
            self.limits.max_query_branches,
        )


def _is_provider_owned_base_url(base_url: Any, host_suffixes: tuple[str, ...]) -> bool:
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    parsed = urlparse(base_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False

    host = parsed.hostname
    if not host:
        return False
    host_lower = host.lower()
    return any(
        host_lower == suffix or host_lower.endswith(f".{suffix}")
        for suffix in host_suffixes
    )


def is_nvidia_provider_base_url(base_url: Any) -> bool:
    """Return True when ``base_url`` is unset or belongs to NVIDIA."""
    return not base_url or _is_provider_owned_base_url(base_url, _NVIDIA_HOST_SUFFIXES)


def is_openai_provider_base_url(base_url: Any) -> bool:
    """Return True when ``base_url`` is unset or belongs to OpenAI."""
    return not base_url or _is_provider_owned_base_url(base_url, _OPENAI_HOST_SUFFIXES)


def _is_explicit_api_key_placeholder(value: Any) -> bool:
    """Return True only for exact placeholders safe in durable artifacts."""
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return bool(normalized and normalized in _EXPLICIT_API_KEY_PLACEHOLDER_VALUES)


def is_placeholder_api_key(value: Any) -> bool:
    """Return True for explicit or conventional runtime template values.

    Runtime credential resolution accepts provider-specific ``your_`` and
    ``your-`` templates for backwards compatibility. Durable artifact guards
    intentionally use the stricter explicit-placeholder predicate above.
    """
    if _is_explicit_api_key_placeholder(value):
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized.startswith("your_") or normalized.startswith("your-")


_INLINE_SECRET_REDACTION = "<redacted>"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DURABLE_ENV_REFERENCE_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
# Match authority syntax from its fixed ``//`` delimiter. Including an
# optional variable-length scheme prefix made a non-match backtrack from every
# character in a long scalar. Consumers need only the authority and remainder,
# so the delimiter-first form preserves detection while keeping scans linear.
_URI_CANDIDATE_RE = re.compile(r"(?is)//[^\s\"'<>]+")
_PATH_NORMALIZED_URI_RE = re.compile(
    r"(?is)\b[a-z][a-z0-9+.-]*:/+(?P<authority>[^/?#\s\"'<>]+)"
)
_ENCODED_URI_PREFIX_RE = re.compile(
    r"(?is)(?:[a-z][a-z0-9+.-]*%(?:25)*3a)?"
    r"%(?:25)*2f%(?:25)*2f"
)
_ENCODED_AUTHORIZATION_RE = re.compile(
    r"(?i)authorization%(?:25)*3a|"
    r"bearer%(?:25)*(?:09|0a|0b|0c|0d|20)"
)
_AUTHORIZATION_BEARER_VALUE_RE = re.compile(
    r"(?i)\bauthorization\s*[:=]\s*bearer\s+([^\s,;]+)"
)
_AUTHORIZATION_VALUE_RE = re.compile(
    r"(?i)\bauthorization\s*[:=]\s*(?!bearer\b)([^\s,;]+)"
)
_BARE_BEARER_VALUE_RE = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]+)")
_CONFIG_ASSIGNMENT_RE = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9_.?&#-])"
    r"(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,127})"
    r"[ \t]*(?:=|:)(?![=:])[ \t]*"
    r"(?P<value>"
    r'"(?:\\.|[^"\\\r\n])*"'
    r"|'(?:\\.|[^'\\\r\n])*'"
    r"|`[^`\r\n]*`"
    r"|(?![#])[^\s,;]+"
    r")"
)
_QUOTED_CONFIG_ASSIGNMENT_RE = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9_.?&#-])"
    r"(?P<key>"
    r'"(?:\\(?:["\\/bfnrt]|u[0-9a-f]{4})|[^"\\\r\n]){1,768}"'
    r"|'(?:\\.|[^'\\\r\n]){1,768}'"
    r")"
    r"[ \t\r\n]*(?:=|:)(?![=:])[ \t\r\n]*"
    r"(?P<value>"
    r'"(?:\\.|[^"\\\r\n])*"'
    r"|'(?:\\.|[^'\\\r\n])*'"
    r"|`[^`\r\n]*`"
    r"|(?![#])[^\s,;]+"
    r")"
)
_PLAIN_YAML_CONFIG_ASSIGNMENT_RE = re.compile(
    r"(?im)"
    r"^[ \t]*(?:-[ \t]+)?"
    r"(?P<key>[A-Za-z](?:[A-Za-z0-9_.-]|[ \t]){0,126}[A-Za-z0-9_.-])"
    r"[ \t]*:(?![:=])[ \t]*"
    r"(?P<value>"
    r'"(?:\\.|[^"\\\r\n])*"'
    r"|'(?:\\.|[^'\\\r\n])*'"
    r"|`[^`\r\n]*`"
    r"|(?![#])[^\s,;]+"
    r")"
)
_YAML_CONTAINER_KEY_RE = re.compile(
    r"(?i)"
    r"(?P<indent>[ \t]*)"
    r"(?P<item>-[ \t]+)?"
    r"(?P<key>"
    r'"(?:\\(?:["\\/bfnrt]|u[0-9a-f]{4})|[^"\\\r\n]){1,768}"'
    r"|'(?:\\.|[^'\\\r\n]){1,768}'"
    r"|[A-Za-z](?:[A-Za-z0-9_.-]|[ \t]){0,126}[A-Za-z0-9_.-]"
    r")"
    r"[ \t]*:[ \t]*(?:#.*)?"
)
_ENCODED_CONFIG_ASSIGNMENT_RE = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9_.?&#-])"
    r"(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,127})"
    r"[ \t]*%(?:25)*(?:3a|3d)[ \t]*"
    r"(?P<value>[^\s,;]+)"
)
_ENCODED_QUOTED_CONFIG_ASSIGNMENT_RE = re.compile(
    r"(?ix)"
    r"%(?:25)*(?:22|27)"
    r"(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,127})"
    r"%(?:25)*(?:22|27)"
    r"[ \t]*%(?:25)*(?:3a|3d)"
)
_PRIVATE_KEY_PLACEHOLDER_RE = re.compile(
    r"(?ix)^(?:"
    r"\.{3}"
    r"|<\s*(?:redacted|private[ _-]*key(?:[ _-]*(?:data|here|placeholder))?)\s*>"
    r"|\{\{\s*[A-Z0-9_]*PRIVATE_KEY[A-Z0-9_]*\s*\}\}"
    r"|(?:YOUR|REPLACE(?:_WITH)?|PASTE)[ _-]*PRIVATE[ _-]*KEY"
    r"(?:[ _-]*(?:CONTENT|DATA|HERE|PLACEHOLDER))?"
    r"|(?:BASE64[ _-]*)?PRIVATE[ _-]*KEY[ _-]*(?:CONTENT|DATA|PLACEHOLDER)"
    r")$"
)
_DOCUMENTATION_PLACEHOLDER_RE = re.compile(
    r"(?ix)^<\s*(?:"
    r"(?:your|example|sample|placeholder|replace(?:[ _-]*with)?|insert|paste)"
    r"[ _-]*(?:api[ _-]*key|access[ _-]*token|token|credential|secret|"
    r"signature|password|username|user)"
    r"|(?:api[ _-]*key|access[ _-]*token|token|credential|secret|signature|"
    r"password|username|user)(?:[ _-]*(?:here|value|placeholder|example))?"
    r")\s*>$"
)
_PEM_BEGIN_PREFIX = "-----BEGIN"
_PEM_END_PREFIX = "-----END"
_MAX_PEM_LABEL_LENGTH = 128
_MAX_PEM_BOUNDARY_SPAN = 512
_MAX_PEM_PLACEHOLDER_BODY_LENGTH = 1024
_ENCODED_PEM_WHITESPACE_CODES = {"09", "0A", "0D", "20"}
_ENV_TEMPLATE_RE = re.compile(
    r"^\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})$"
)
_AUTH_DOCUMENTATION_WORDS = {
    "authentication",
    "authorization",
    "credential",
    "credentials",
    "example",
    "examples",
    "guidance",
    "header",
    "required",
    "scheme",
    "token",
    "tokens",
}
_DOCUMENTATION_USERINFO_VALUES = {
    "example",
    "sample-user",
    "sample_user",
    "user",
    "username",
    "your-user",
    "your_user",
}
_NON_USERINFO_PREFIXES = {"mailto", "tel", "urn"}
_USERINFO_COLON_RE = re.compile(r"(?i):|%(?:25)*3a")
_USERINFO_AT_RE = re.compile(r"(?i)@|%(?:25)*40")
_MAX_URI_NESTING_DEPTH = 8
_MAX_URI_DECODE_ROUNDS = 4
_REFERENCE_KEY_SUFFIXES = {
    "env",
    "file",
    "filename",
    "field",
    "fields",
    "name",
    "names",
    "path",
    "paths",
    "ref",
    "reference",
    "references",
    "var",
    "vars",
}
_VALUE_KEY_SUFFIXES = {"literal", "string", "value", "values"}
_PATH_REFERENCE_KEY_SUFFIXES = {
    "asset",
    "assets",
    "dataset",
    "dir",
    "directory",
    "endpoint",
    "file",
    "filename",
    "href",
    "hrefs",
    "image",
    "images",
    "media",
    "path",
    "paths",
    "pdf",
    "pdfs",
    "ref",
    "reference",
    "references",
    "source",
    "uri",
    "uris",
    "url",
    "urls",
}


def _config_key_tokens(key: Any) -> tuple[str, ...]:
    """Normalize snake, kebab, and camel-case config keys into tokens."""
    # Split both ordinary camel case (``apiKey``) and an acronym followed by a
    # word (``AWSAccessKeyId``). Without the second boundary, the latter became
    # ``awsaccess_key_id`` and bypassed the ``access_key_id`` credential suffix.
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", str(key))
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text).lower()
    return tuple(token for token in re.split(r"[^a-z0-9]+", text) if token)


def _strip_rotation_suffix(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Remove one conventional numeric/version suffix from a config key."""
    if not tokens:
        return tokens
    last = tokens[-1]
    suffix_start = len(last)
    while suffix_start and last[suffix_start - 1].isdigit():
        suffix_start -= 1
    if suffix_start == len(last):
        return tokens
    prefix = last[:suffix_start]
    if not prefix or prefix == "v":
        return tokens[:-1]
    if prefix.endswith("v"):
        prefix = prefix[:-1]
    return (*tokens[:-1], prefix)


def _sensitive_config_tokens(tokens: tuple[str, ...]) -> bool:
    """Classify already-normalized config-key tokens."""
    for _ in range(2):
        tokens = _strip_rotation_suffix(tokens)
        if tokens[-2:] in {("connection", "string"), ("connection", "strings")}:
            return True
        if not tokens or tokens[-1] not in _VALUE_KEY_SUFFIXES:
            break
        tokens = tokens[:-1]
    if tokens[-2:] in {("connection", "string"), ("connection", "strings")}:
        return True
    if not tokens or tokens[-1] in _REFERENCE_KEY_SUFFIXES:
        return False

    compact = "".join(tokens)
    if compact.endswith(("apikey", "apikeys")):
        return True
    # A bare plural field conventionally carries a collection of credentials.
    # Keep the match exact so token-count fields such as ``max_tokens`` and
    # domain data such as ``custom_tokens`` remain ordinary configuration.
    if tokens == ("tokens",):
        return True
    if tokens[-1] in {
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "passwords",
        "passwd",
        "passwds",
        "secret",
        "secrets",
        "token",
    }:
        return True
    sensitive_suffixes = (
        ("account", "key"),
        ("account", "keys"),
        ("access", "key"),
        ("access", "key", "id"),
        ("access", "key", "ids"),
        ("access", "keys"),
        ("access", "tokens"),
        ("api", "tokens"),
        ("auth", "header"),
        ("auth", "headers"),
        ("auth", "tokens"),
        ("authorization", "header"),
        ("authorization", "headers"),
        ("authorization", "tokens"),
        ("bearer", "tokens"),
        ("client", "key"),
        ("client", "keys"),
        ("client", "secrets"),
        ("client", "tokens"),
        ("consumer", "key"),
        ("consumer", "keys"),
        ("csrf", "tokens"),
        ("id", "tokens"),
        ("oauth", "tokens"),
        ("private", "key"),
        ("private", "keys"),
        ("refresh", "tokens"),
        ("session", "key"),
        ("session", "keys"),
        ("session", "tokens"),
        ("service", "account", "key"),
        ("service", "account", "keys"),
        ("shared", "access", "signature"),
        ("shared", "access", "signatures"),
        ("service", "tokens"),
        ("secret", "access", "key"),
        ("secret", "access", "keys"),
        ("secret", "key"),
        ("secret", "keys"),
        ("security", "tokens"),
        ("signing", "key"),
        ("signing", "keys"),
        ("storage", "key"),
        ("storage", "keys"),
        ("subscription", "key"),
        ("subscription", "keys"),
    )
    return any(tokens[-len(suffix) :] == suffix for suffix in sensitive_suffixes)


def _is_sensitive_config_key(key: Any) -> bool:
    """Return whether a config key conventionally stores a credential value.

    Reference and schema keys such as ``api_key_env``, ``password_file``, and
    ``credential_fields`` are deliberately excluded. Token matching is based
    on whole key components, so ordinary keys such as ``max_tokens``,
    ``tokenizer``, and ``secretary`` are not treated as credentials. A single
    numeric or ``vN`` suffix is ignored only when the remaining key is already
    an unambiguous credential name, covering common key-rotation fields such as
    ``api_key_2`` without classifying ``max_tokens_2``.
    """
    return _sensitive_config_tokens(_config_key_tokens(key))


def _is_path_reference_key(key: Any) -> bool:
    """Return whether a config key carries path-, URI-, or media-locator values."""
    tokens = _config_key_tokens(key)
    if not tokens:
        return False
    if tokens[-1] in _PATH_REFERENCE_KEY_SUFFIXES:
        return True
    if tokens == ("session", "id"):
        return True
    reference_tokens = {"ref", "reference", "references"}
    media_tokens = {"asset", "assets", "image", "images", "media", "pdf", "pdfs"}
    return bool(
        reference_tokens.intersection(tokens) and media_tokens.intersection(tokens)
    )


def _is_sensitive_env_reference_key(key: Any) -> bool:
    """Return whether ``key`` names an environment-backed credential field."""
    tokens = _config_key_tokens(key)
    while tokens and tokens[-1] in _VALUE_KEY_SUFFIXES:
        tokens = tokens[:-1]
    tokens = _strip_rotation_suffix(tokens)
    while tokens and tokens[-1] in _VALUE_KEY_SUFFIXES:
        tokens = tokens[:-1]
    if not tokens:
        return False

    suffix_length = 0
    if len(tokens) >= 2 and tokens[-2:] in {
        ("env", "name"),
        ("env", "names"),
        ("env", "var"),
        ("env", "vars"),
    }:
        suffix_length = 2
    elif tokens[-1] in {"env", "var", "vars"}:
        suffix_length = 1
    if not suffix_length:
        return False

    base_tokens = tokens[:-suffix_length]
    if not base_tokens:
        return False
    return _is_sensitive_config_key("_".join(base_tokens))


def parse_env_reference(value: Any, *, allow_legacy_bare: bool = False) -> str | None:
    """Return the environment name carried by an explicit reference.

    Durable and diagnostic boundaries accept only the unambiguous ``${NAME}``
    form. Runtime consumers may opt into legacy bare-name parsing while source
    configs migrate. Invalid inputs return ``None`` without rendering the
    potentially credential-bearing value in a diagnostic.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    match = _DURABLE_ENV_REFERENCE_RE.fullmatch(stripped)
    if match:
        return match.group(1)
    if allow_legacy_bare and _ENV_NAME_RE.fullmatch(stripped):
        return stripped
    return None


def format_env_reference(env_name: Any) -> str:
    """Return the canonical durable ``${NAME}`` form without reading its value."""
    parsed = parse_env_reference(env_name, allow_legacy_bare=True)
    if parsed is None:
        raise ValueError(
            "environment reference must be a bare variable name or ${NAME}"
        )
    return f"${{{parsed}}}"


def _is_valid_env_reference(value: Any) -> bool:
    """Return whether every value uses explicit durable-reference syntax."""
    if isinstance(value, list | tuple):
        return all(_is_valid_env_reference(item) for item in value)
    return parse_env_reference(value) is not None


def _is_invalid_durable_env_reference(value: Any) -> bool:
    """Reject ambiguous env fields while allowing empty or placeholder values."""
    return _is_inline_secret_value(value) and not _is_valid_env_reference(value)


def _is_inline_secret_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        if stripped.lower() in {"<redacted>", "redacted"}:
            return False
        if _DOCUMENTATION_PLACEHOLDER_RE.fullmatch(stripped):
            return False
        if _is_explicit_api_key_placeholder(stripped):
            return False
    return True


def _is_sensitive_url_parameter(key: str) -> bool:
    """Return whether a URL parameter conventionally carries a credential.

    Signed URLs use short or provider-specific parameter names that are too
    ambiguous to classify as credential fields in ordinary configuration
    mappings (for example, Azure's ``sig`` and Google's ``key``). Within a URL
    query or fragment those names represent bearer material and must not reach
    durable artifacts.
    """
    tokens = _config_key_tokens(key)
    return (
        _is_sensitive_config_key(key)
        or tokens == ("key",)
        or bool(tokens and tokens[-1] in {"sig", "signature"})
    )


def _decoded_scalar_variants(
    value: str, *, budget: _ScanBudget | None = None
) -> tuple[tuple[str, ...], bool]:
    """Return bounded percent-decoded variants and whether work was exhausted."""
    candidate_texts = [value.strip()]

    for _ in range(_MAX_URI_DECODE_ROUNDS):
        if budget is not None:
            budget.consume_decode(candidate_texts[-1])
        decoded = unquote(candidate_texts[-1])
        if decoded == candidate_texts[-1]:
            return tuple(candidate_texts), False
        candidate_texts.append(decoded)

    if budget is not None:
        budget.consume_decode(candidate_texts[-1])
    return tuple(candidate_texts), unquote(candidate_texts[-1]) != candidate_texts[-1]


def _decode_nested_percent_bytes_once(value: str) -> tuple[str, bool]:
    """Collapse one bounded pass of ordinary or repeatedly escaped bytes."""
    decoded: list[str] = []
    changed = False
    index = 0
    while index < len(value):
        if value[index] != "%":
            decoded.append(value[index])
            index += 1
            continue

        byte_index = index + 1
        nested_percent = False
        while value[byte_index : byte_index + 2].lower() == "25":
            nested_percent = True
            byte_index += 2
        encoded_byte = value[byte_index : byte_index + 2]
        if len(encoded_byte) == 2 and all(
            character in "0123456789abcdefABCDEF" for character in encoded_byte
        ):
            decoded.append(chr(int(encoded_byte, 16)))
            index = byte_index + 2
            changed = True
            continue
        if nested_percent:
            # A terminal ``%25`` chain canonically represents a literal percent
            # even when no following byte escape is present.
            decoded.append("%")
            index = byte_index
            changed = True
            continue
        decoded.append("%")
        index += 1
    return "".join(decoded), changed


def _canonicalize_terminal_percent_encoding(
    value: str, *, budget: _ScanBudget | None = None
) -> tuple[str, bool]:
    """Decode a fixed number of terminal probe passes and report ambiguity."""
    canonical = value
    for _ in range(_MAX_URI_DECODE_ROUNDS):
        if budget is not None:
            budget.consume_decode(canonical)
        canonical, changed = _decode_nested_percent_bytes_once(canonical)
        if not changed:
            return canonical, False
    if budget is not None:
        budget.consume_decode(canonical)
    _, still_changes = _decode_nested_percent_bytes_once(canonical)
    return canonical, still_changes


def _iter_uri_candidates(
    value: str, *, budget: _ScanBudget | None = None
) -> Iterator[re.Match[str]]:
    """Yield URI regex matches while accounting for every candidate branch."""
    for match in _URI_CANDIDATE_RE.finditer(value):
        if budget is not None:
            budget.consume_uri_candidate()
        yield match


def _uri_candidate_texts(
    value: str, *, budget: _ScanBudget | None = None
) -> tuple[tuple[str, ...], bool]:
    """Return bounded URI text variants and whether decoding ended ambiguously.

    Query parsing decodes one percent-encoding layer, but redirect parameters
    are commonly encoded more than once. Scan every bounded decoding layer: a
    visible public URI must not hide a separately encoded credential URI in the
    same string. If the work bound is exhausted while either an encoded URI
    prefix or unresolved encoded content inside a visible URI remains, callers
    fail closed instead of guessing what the next layer would reveal. The
    latter covers every URI component, including authority delimiters and query
    keys, rather than enumerating individual credential-bearing delimiters.
    """
    candidate_texts, exhausted = _decoded_scalar_variants(value, budget=budget)
    if not exhausted:
        return candidate_texts, False

    terminal_text = candidate_texts[-1]
    if _ENCODED_URI_PREFIX_RE.search(terminal_text):
        return candidate_texts, True

    # A visible URI with any still-decodable content cannot be proven safe at
    # the bounded-work limit. For example, an otherwise-plain authority can
    # hide only its ``@`` delimiter one layer beyond the limit. Checking the
    # complete matched URI also closes the same class for encoded query names,
    # separators, fragments, and future credential syntaxes.
    ambiguous_visible_uri = False
    for match in _iter_uri_candidates(terminal_text, budget=budget):
        if budget is not None:
            budget.consume_decode(match.group(0))
        if unquote(match.group(0)) != match.group(0):
            ambiguous_visible_uri = True
            break
    return candidate_texts, ambiguous_visible_uri


def _contains_uri_candidate(value: str, *, budget: _ScanBudget | None = None) -> bool:
    """Return whether text contains a visible or boundedly encoded URI."""
    candidate_texts, ambiguous = _uri_candidate_texts(value, budget=budget)
    return ambiguous or any(
        next(_iter_uri_candidates(candidate_text, budget=budget), None) is not None
        for candidate_text in candidate_texts
    )


def _url_component_has_inline_secret(
    component: str, *, depth: int, budget: _ScanBudget | None = None
) -> bool:
    # Both ampersand and semicolon are accepted as query separators by common
    # servers and signing implementations. Treating only ``&`` as a separator
    # lets ``?x=1;access_token=...`` bypass the durable-artifact guard.
    normalized_component = component.replace(";", "&")
    if budget is not None:
        budget.consume_query_branches(normalized_component.count("&") + 1)
    for key, value in parse_qsl(normalized_component, keep_blank_values=True):
        if _is_sensitive_url_parameter(key) and _is_inline_secret_value(value):
            return True
        # parse_qsl decodes percent-encoded redirect and callback values. Inspect
        # those recursively so a credential URI cannot hide behind an ordinary
        # parameter name such as ``redirect``.
        if depth >= _MAX_URI_NESTING_DEPTH:
            # Continue checking direct parameters at the depth boundary, but do
            # not recurse again. A further nested URI is unresolved bearer
            # material, so fail closed; ordinary scalar values remain allowed.
            if _contains_uri_candidate(value, budget=budget):
                return True
        elif _is_url_with_inline_secret(value, _depth=depth + 1, budget=budget):
            return True
    return False


def _userinfo_is_explicit_documentation(authority: str) -> bool:
    """Recognize only conventional username-only URI examples."""
    userinfo, separator, _host = authority.rpartition("@")
    return bool(
        separator and userinfo.strip().lower() in _DOCUMENTATION_USERINFO_VALUES
    )


def _authority_has_inline_secret(
    authority: str,
    *,
    budget: _ScanBudget | None = None,
) -> bool:
    """Inspect literal and boundedly encoded authority userinfo."""
    candidate_authorities, exhausted = _decoded_scalar_variants(
        authority,
        budget=budget,
    )
    for candidate_authority in candidate_authorities:
        if "@" in candidate_authority and not _userinfo_is_explicit_documentation(
            candidate_authority
        ):
            return True
    # An unresolved encoded at-sign can still turn the preceding authority
    # bytes into userinfo after the fixed decode bound. Fail closed without
    # rejecting unrelated encoded public path/query content.
    return bool(
        exhausted and _USERINFO_AT_RE.search(candidate_authorities[-1]) is not None
    )


def _is_url_with_inline_secret(
    value: Any, *, _depth: int = 0, budget: _ScanBudget | None = None
) -> bool:
    """Return whether a URI value embeds userinfo or bearer parameters."""
    if not isinstance(value, str) or not value.strip():
        return False

    candidate_texts, ambiguous = _uri_candidate_texts(value, budget=budget)
    if ambiguous:
        return True

    for candidate_text in candidate_texts:
        for candidate_match in _iter_uri_candidates(candidate_text, budget=budget):
            candidate = candidate_match.group(0)
            match = re.match(
                r"(?is)^(?:[a-z][a-z0-9+.-]*:)?//([^/?#]*)(.*)$",
                candidate,
            )
            if match is None:  # pragma: no cover - constrained by candidate regex
                continue
            authority, remainder = match.groups()
            if not authority:
                continue
            if _authority_has_inline_secret(authority, budget=budget):
                # Userinfo is authentication material as a whole. Even a
                # username-only form can carry a token, so fail closed rather than
                # guessing which component is confidential. Exempt only the
                # conventional documentation-only usernames below; arbitrary
                # username-only values still fail closed because they can be bearer
                # material. This lexical check does not depend on hostname parsing,
                # so malformed URIs cannot bypass it.
                return True

            before_fragment, separator, fragment = remainder.partition("#")
            _, query_separator, query = before_fragment.partition("?")
            if query_separator and _url_component_has_inline_secret(
                query, depth=_depth, budget=budget
            ):
                return True
            if separator and _url_component_has_inline_secret(
                fragment, depth=_depth, budget=budget
            ):
                return True
    return False


def _path_text_has_inline_secret(
    value: str, *, budget: _ScanBudget | None = None
) -> bool:
    """Inspect query/fragment credentials after URI-to-Path normalization.

    ``Path('https://host/object?sig=...')`` becomes ``https:/host/...`` and no
    longer matches an ordinary URI authority. Query and fragment bearer
    semantics survive that normalization, so inspect those components without
    requiring a URI prefix. Scan bounded decoded variants and fail closed if a
    path component remains ambiguous at the work limit.
    """
    candidate_texts, exhausted = _decoded_scalar_variants(value, budget=budget)
    for candidate_text in candidate_texts:
        if any(
            _authority_has_inline_secret(
                match.group("authority"),
                budget=budget,
            )
            for match in _PATH_NORMALIZED_URI_RE.finditer(candidate_text)
        ):
            return True

        before_fragment, fragment_separator, fragment = candidate_text.partition("#")
        _, query_separator, query = before_fragment.partition("?")
        if query_separator and _url_component_has_inline_secret(
            query, depth=0, budget=budget
        ):
            return True
        if fragment_separator and _url_component_has_inline_secret(
            fragment, depth=0, budget=budget
        ):
            return True

    if not exhausted:
        return False
    terminal_text = candidate_texts[-1]
    for match in _PATH_NORMALIZED_URI_RE.finditer(terminal_text):
        if budget is not None:
            budget.consume_uri_candidate()
            budget.consume_decode(match.group(0))
        if unquote(match.group(0)) != match.group(0):
            return True
    if budget is not None:
        budget.consume_decode(terminal_text)
    return ("?" in terminal_text or "#" in terminal_text) and (
        unquote(terminal_text) != terminal_text
    )


def _has_inline_authorization(value: Any, *, budget: _ScanBudget | None = None) -> bool:
    """Return whether free text contains a recognized live bearer credential."""
    if not isinstance(value, str) or not value.strip():
        return False

    def is_live_explicit_value(match_value: str) -> bool:
        stripped = match_value.strip()
        if not _is_inline_secret_value(stripped):
            return False
        return _ENV_TEMPLATE_RE.fullmatch(stripped) is None

    def is_live_bare_value(match: re.Match[str], candidate_text: str) -> bool:
        stripped = match.group(1).strip().strip(".,:;()[]{}\"'")
        if not is_live_explicit_value(stripped):
            return False
        if stripped.lower() in _AUTH_DOCUMENTATION_WORDS:
            return False
        # An exact ``Bearer <value>`` scalar or a value supplied by an explicit
        # usage phrase is a credential boundary regardless of token length or
        # alphabet. Keep documentation words and unrelated prose such as
        # ``The bearer must authenticate`` out of this high-confidence path.
        raw_prefix = candidate_text[: match.start()]
        prefix = raw_prefix.rstrip()
        if (
            not prefix
            or prefix.endswith((":", "="))
            or re.search(r"(?:^|[\r\n])\s*$", raw_prefix)
        ):
            return True
        if re.search(
            r"(?i)\b(?:pass|provide|send|supply|use|using|with)\s*$",
            prefix,
        ):
            return True

        # Other free-text mentions require an opaque token shape to avoid
        # classifying ordinary prose. Digits, token punctuation, or a long word
        # distinguish JWT/API-key shapes from grammar such as ``bearer must``.
        return (
            any(character.isdigit() for character in stripped)
            or any(character in "._~+/=-" for character in stripped)
            or len(stripped) >= 20
        )

    candidate_texts, exhausted = _decoded_scalar_variants(value, budget=budget)
    for candidate_text in candidate_texts:
        for pattern in (
            _AUTHORIZATION_BEARER_VALUE_RE,
            _AUTHORIZATION_VALUE_RE,
        ):
            for match in pattern.finditer(candidate_text):
                if is_live_explicit_value(match.group(1)):
                    return True
        for match in _BARE_BEARER_VALUE_RE.finditer(candidate_text):
            if is_live_bare_value(match, candidate_text):
                return True
    # Deeply encoded Authorization syntax is unresolved credential material at
    # the bounded-work limit. Fail closed without unbounded decoding.
    return bool(exhausted and _ENCODED_AUTHORIZATION_RE.search(candidate_texts[-1]))


def _has_inline_sensitive_assignment(
    value: Any, *, budget: _ScanBudget | None = None
) -> bool:
    """Detect live credentials assigned to sensitive keys inside free text."""
    if not isinstance(value, str) or not value.strip():
        return False

    def normalized_assignment_value(raw_value: str) -> str:
        stripped = raw_value.strip()
        if (
            len(stripped) >= 2
            and stripped[0] in {'"', "'", "`"}
            and stripped[-1] == stripped[0]
        ):
            stripped = stripped[1:-1].strip()
        return stripped

    def assignment_key(raw_key: str) -> str | None:
        if not raw_key.startswith(('"', "'")):
            return raw_key
        try:
            decoded_key = (
                json.loads(raw_key)
                if raw_key.startswith('"')
                else ast.literal_eval(raw_key)
            )
        except (SyntaxError, ValueError):
            return None
        if not isinstance(decoded_key, str) or len(decoded_key) > 128:
            return None
        return decoded_key

    def is_live_assignment(match: re.Match[str]) -> bool:
        key = assignment_key(match.group("key"))
        if key is None or not _is_sensitive_config_key(key):
            return False
        assigned_value = normalized_assignment_value(match.group("value"))
        if not _is_inline_secret_value(assigned_value):
            return False
        if _ENV_TEMPLATE_RE.fullmatch(assigned_value):
            return False
        if _PRIVATE_KEY_PLACEHOLDER_RE.fullmatch(assigned_value):
            return False
        if _is_explicit_private_key_placeholder_value(assigned_value):
            return False
        documentation_value = assigned_value.rstrip(".,:;!?)]}").lower()
        if (
            documentation_value == "bearer"
            or documentation_value in _AUTH_DOCUMENTATION_WORDS
        ):
            return False
        return documentation_value not in {"false", "none", "null", "~"}

    def has_live_assignment(candidate_text: str) -> bool:
        if any(
            is_live_assignment(match)
            for pattern in (
                _CONFIG_ASSIGNMENT_RE,
                _QUOTED_CONFIG_ASSIGNMENT_RE,
                _PLAIN_YAML_CONFIG_ASSIGNMENT_RE,
            )
            for match in pattern.finditer(candidate_text)
        ):
            return True

        # PyYAML emits nested mappings/sequences with a key-only parent line.
        # Track a pending sensitive parent in one pass and fail closed as soon
        # as a nonempty child record appears. A truly empty/schema-only key is
        # safe when the next content line is a peer or the document ends.
        pending_parent: tuple[int, bool] | None = None
        for line in candidate_text.splitlines():
            stripped = line.lstrip(" \t")
            if not stripped or stripped.startswith("#"):
                continue
            indentation = len(line) - len(stripped)
            if pending_parent is not None:
                parent_indent, allows_indentless_sequence = pending_parent
                is_indentless_sequence = allows_indentless_sequence and (
                    stripped == "-" or stripped.startswith("- ")
                )
                if indentation > parent_indent or (
                    indentation == parent_indent and is_indentless_sequence
                ):
                    return True
                pending_parent = None

            container_match = _YAML_CONTAINER_KEY_RE.fullmatch(line)
            if container_match is None:
                continue
            key = assignment_key(container_match.group("key"))
            if key is not None and _is_sensitive_config_key(key):
                pending_parent = (
                    len(container_match.group("indent")),
                    container_match.group("item") is None,
                )
        return False

    candidate_texts, exhausted = _decoded_scalar_variants(value, budget=budget)
    for candidate_text in candidate_texts:
        if has_live_assignment(candidate_text):
            return True

    if not exhausted:
        return False

    canonical_text, canonicalization_exhausted = (
        _canonicalize_terminal_percent_encoding(candidate_texts[-1], budget=budget)
    )
    if has_live_assignment(canonical_text):
        return True
    if not canonicalization_exhausted:
        return False

    # At the bounded decoding limit, a still-encoded assignment cannot be
    # proven credential-free. The key remains visible for conventional percent
    # encoding, so reuse the same classifier and fail closed without decoding
    # attacker-controlled text without a bound.
    return any(
        _is_sensitive_config_key(match.group("key"))
        for pattern in (
            _ENCODED_CONFIG_ASSIGNMENT_RE,
            _ENCODED_QUOTED_CONFIG_ASSIGNMENT_RE,
        )
        for match in pattern.finditer(canonical_text)
    )


def _consume_pem_separator(
    text: str, index: int, *, allow_encoded: bool
) -> tuple[int, bool]:
    """Consume literal or nested-percent-encoded PEM whitespace."""
    start = index
    used_encoded = False
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if not allow_encoded or text[index] != "%":
            break
        encoded_index = index + 1
        while text.startswith("25", encoded_index):
            encoded_index += 2
        if (
            text[encoded_index : encoded_index + 2].upper()
            not in _ENCODED_PEM_WHITESPACE_CODES
        ):
            break
        used_encoded = True
        index = encoded_index + 2
    return index if index > start else start, used_encoded


def _has_ascii_pem_prefix(text: str, index: int, prefix: str) -> bool:
    """Compare a fixed ASCII PEM prefix without changing string indices."""
    return text[index : index + len(prefix)].upper() == prefix


def _parse_pem_boundary(
    text: str,
    marker_index: int,
    prefix: str,
    *,
    allow_encoded_whitespace: bool,
) -> tuple[str, int, bool] | None:
    """Parse one length-bounded PEM boundary starting at ``marker_index``."""
    if not _has_ascii_pem_prefix(text, marker_index, prefix):
        return None

    index = marker_index + len(prefix)
    index, used_encoded = _consume_pem_separator(
        text,
        index,
        allow_encoded=allow_encoded_whitespace,
    )
    if index == marker_index + len(prefix):
        return None

    boundary_limit = min(len(text), marker_index + _MAX_PEM_BOUNDARY_SPAN)
    label: list[str] = []
    label_length = 0
    while index < boundary_limit:
        if text.startswith("-----", index):
            normalized_label = "".join(label).strip()
            if not normalized_label:
                return None
            return normalized_label, index + 5, used_encoded

        separator_end, separator_was_encoded = _consume_pem_separator(
            text,
            index,
            allow_encoded=allow_encoded_whitespace,
        )
        if separator_end > index:
            if label and label[-1] != " ":
                label.append(" ")
            used_encoded = used_encoded or separator_was_encoded
            index = separator_end
            continue

        character = text[index]
        if not (character.isascii() and (character.isalnum() or character in "_.-")):
            return None
        label.append(character.upper())
        label_length += 1
        if label_length > _MAX_PEM_LABEL_LENGTH:
            return None
        index += 1
    return None


def _is_private_key_pem_label(label: str) -> bool:
    words = label.split()
    return words[-2:] == ["PRIVATE", "KEY"] or words[-3:] == [
        "PRIVATE",
        "KEY",
        "BLOCK",
    ]


def _pem_body_is_explicit_placeholder(body: str) -> bool:
    """Recognize only an empty or explicit placeholder body."""
    if len(body) > _MAX_PEM_PLACEHOLDER_BODY_LENGTH:
        return False

    content_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        while stripped.startswith(">"):
            stripped = stripped[1:].lstrip()
        if stripped:
            content_lines.append(stripped)
    if not content_lines:
        return True
    if len(content_lines) != 1:
        return False
    placeholder = content_lines[0]
    return bool(
        _ENV_TEMPLATE_RE.fullmatch(placeholder)
        or _PRIVATE_KEY_PLACEHOLDER_RE.fullmatch(placeholder)
    )


def _pem_body_is_explicitly_safe(body: str) -> bool:
    """Recognize only explicit placeholders or narrow marker prose."""
    if _pem_body_is_explicit_placeholder(body):
        return True
    # Documentation commonly names the BEGIN and END markers on one line.
    # Keep that narrow connector safe without treating arbitrary same-line
    # payload bytes as documentation.
    return body.strip().casefold() in {"and", "or", "/", "to"}


def _inspection_text_has_private_key_block(text: str) -> bool:
    """Scan PEM blocks once without regex backtracking."""
    active_block: tuple[str, int] | None = None
    cursor = 0
    while True:
        marker_index = text.find("-----", cursor)
        if marker_index < 0:
            break

        prefix: str | None = None
        is_begin = False
        if _has_ascii_pem_prefix(text, marker_index, _PEM_BEGIN_PREFIX):
            prefix = _PEM_BEGIN_PREFIX
            is_begin = True
        elif _has_ascii_pem_prefix(text, marker_index, _PEM_END_PREFIX):
            prefix = _PEM_END_PREFIX
        if prefix is None:
            cursor = marker_index + 5
            continue

        parsed = _parse_pem_boundary(
            text,
            marker_index,
            prefix,
            allow_encoded_whitespace=False,
        )
        if parsed is None:
            cursor = marker_index + 5
            continue
        label, boundary_end, _ = parsed
        cursor = boundary_end
        if not _is_private_key_pem_label(label):
            continue
        if is_begin:
            if active_block is not None:
                # Nested/repeated private BEGIN markers are malformed partial
                # key material. Reject immediately rather than searching the
                # remainder once per marker.
                return True
            active_block = (label, boundary_end)
            continue

        if active_block is None or active_block[0] != label:
            continue
        _, body_start = active_block
        active_block = None
        if not _pem_body_is_explicitly_safe(text[body_start:marker_index]):
            return True

    # A truncated block is still partial credential material whenever anything
    # follows the BEGIN boundary. Only an empty trailing boundary remains safe;
    # placeholders require a matching END before they are accepted.
    if active_block is not None:
        _, body_start = active_block
        if text[body_start:].strip(" >\t\r\n"):
            return True
    return False


def _is_explicit_private_key_placeholder_value(value: str) -> bool:
    """Recognize a value that is exactly one safe private-key placeholder block."""
    inspection_text = (
        value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\r")
    )
    normalized_lines: list[str] = []
    for line in inspection_text.splitlines():
        normalized_line = line.lstrip()
        while normalized_line.startswith(">"):
            normalized_line = normalized_line[1:].lstrip()
        normalized_lines.append(normalized_line)
    normalized = "\n".join(normalized_lines).strip()

    begin = _parse_pem_boundary(
        normalized,
        0,
        _PEM_BEGIN_PREFIX,
        allow_encoded_whitespace=False,
    )
    if begin is None:
        return False
    begin_label, body_start, _ = begin
    if not _is_private_key_pem_label(begin_label):
        return False

    cursor = body_start
    while True:
        marker_index = normalized.find("-----", cursor)
        if marker_index < 0:
            return False
        end = _parse_pem_boundary(
            normalized,
            marker_index,
            _PEM_END_PREFIX,
            allow_encoded_whitespace=False,
        )
        if end is None:
            cursor = marker_index + 5
            continue
        end_label, boundary_end, _ = end
        if end_label != begin_label:
            return False
        return bool(
            _pem_body_is_explicit_placeholder(normalized[body_start:marker_index])
            and not normalized[boundary_end:].strip()
        )


def _has_unresolved_encoded_private_key_marker(text: str) -> bool:
    """Fail closed when bounded decoding leaves encoded private-key syntax."""
    cursor = 0
    while True:
        marker_index = text.find("-----", cursor)
        if marker_index < 0:
            return False
        if not _has_ascii_pem_prefix(text, marker_index, _PEM_BEGIN_PREFIX):
            cursor = marker_index + 5
            continue
        parsed = _parse_pem_boundary(
            text,
            marker_index,
            _PEM_BEGIN_PREFIX,
            allow_encoded_whitespace=True,
        )
        if parsed is None:
            cursor = marker_index + len(_PEM_BEGIN_PREFIX)
            continue
        label, boundary_end, label_used_encoding = parsed
        cursor = boundary_end
        if not _is_private_key_pem_label(label):
            continue
        _, body_used_encoding = _consume_pem_separator(
            text,
            boundary_end,
            allow_encoded=True,
        )
        if label_used_encoding or body_used_encoding:
            return True


def _has_inline_private_key_block(
    value: Any, *, budget: _ScanBudget | None = None
) -> bool:
    """Detect live PEM-style private-key blocks embedded in free text.

    Complete private-key boundaries are a high-confidence credential shape,
    independent of the surrounding durable field name. Documentation may
    legitimately show the same boundaries around an explicit placeholder, so
    exempt only a small, syntactic placeholder vocabulary rather than trying
    to validate or fingerprint the private-key payload.
    """
    if not isinstance(value, str) or not value.strip():
        return False

    candidate_texts, exhausted = _decoded_scalar_variants(value, budget=budget)
    for candidate_text in candidate_texts:
        inspection_text = (
            candidate_text.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\r", "\r")
        )
        if _inspection_text_has_private_key_block(inspection_text):
            return True
    if not exhausted:
        return False

    canonical_text, canonicalization_exhausted = (
        _canonicalize_terminal_percent_encoding(candidate_texts[-1], budget=budget)
    )
    canonical_inspection_text = (
        canonical_text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\r")
    )
    if _inspection_text_has_private_key_block(canonical_inspection_text):
        return True
    return bool(
        canonicalization_exhausted
        and _has_unresolved_encoded_private_key_marker(canonical_text)
    )


def _has_userinfo_without_authority(
    value: Any, *, budget: _ScanBudget | None = None
) -> bool:
    """Detect deterministic ``user:secret@host`` DSN/userinfo forms."""
    if not isinstance(value, str) or not value.strip():
        return False

    def is_clock_locator(user: str, secret: str) -> bool:
        """Preserve the established ``10:30@room`` scheduling syntax."""
        if (
            not user.isdigit()
            or not secret.isdigit()
            or not 1 <= len(user) <= 2
            or not 1 <= len(secret) <= 2
        ):
            return False
        return 0 <= int(user) <= 23 and 0 <= int(secret) <= 59

    def is_userinfo_character(character: str) -> bool:
        return not character.isspace() and character not in ":/@;"

    def has_host_after(candidate_text: str, at_index: int) -> bool:
        host_index = at_index + 1
        if host_index >= len(candidate_text):
            return False
        if candidate_text[host_index] == "[":
            host_index += 1
            host_start = host_index
            while (
                host_index < len(candidate_text)
                and candidate_text[host_index] != "]"
                and candidate_text[host_index] != "/"
                and not candidate_text[host_index].isspace()
            ):
                host_index += 1
            return (
                host_index > host_start
                and host_index < len(candidate_text)
                and candidate_text[host_index] == "]"
            )
        first = candidate_text[host_index]
        return first.isascii() and first.isalnum()

    def has_live_userinfo(candidate_text: str) -> bool:
        """Scan delimiter-first so long non-matches remain linear-time."""
        for colon_index, character in enumerate(candidate_text):
            if character != ":":
                continue

            user_start = colon_index
            while user_start > 0 and is_userinfo_character(
                candidate_text[user_start - 1]
            ):
                user_start -= 1
            if user_start == colon_index:
                continue

            at_index = colon_index + 1
            while at_index < len(candidate_text) and is_userinfo_character(
                candidate_text[at_index]
            ):
                at_index += 1
            if at_index == colon_index + 1:
                continue
            if at_index >= len(candidate_text) or candidate_text[at_index] != "@":
                continue
            if not has_host_after(candidate_text, at_index):
                continue

            user = candidate_text[user_start:colon_index]
            secret = candidate_text[colon_index + 1 : at_index]
            if user.lower() in _NON_USERINFO_PREFIXES:
                continue
            if is_clock_locator(user, secret):
                continue
            return True
        return False

    candidate_texts, exhausted = _decoded_scalar_variants(value, budget=budget)
    for candidate_text in candidate_texts:
        # A single-label host is valid for service discovery and local
        # infrastructure, so it is not evidence against userinfo. Colon/at
        # forms are indistinguishable from credentials and must fail closed
        # rather than relying on username/password heuristics.
        if has_live_userinfo(candidate_text):
            return True

    if not exhausted:
        return False

    # At the bounded-work limit, reject any still-encoded colon followed by an
    # at-sign (or their literal forms). This covers generic and single-label
    # DSNs across arbitrary remaining encoding depth without enumerating
    # schemes or decoding attacker-controlled input without a bound.
    terminal_text = candidate_texts[-1]
    colon = _USERINFO_COLON_RE.search(terminal_text)
    return bool(colon and _USERINFO_AT_RE.search(terminal_text, colon.end()))


def _string_has_inline_secret(
    value: Any,
    *,
    path_context: bool = False,
    budget: _ScanBudget | None = None,
) -> bool:
    """Return whether a scalar embeds a supported inline credential syntax."""
    is_path_like = isinstance(value, os.PathLike)
    if is_path_like:
        value = os.fspath(value)
    if isinstance(value, bytes | bytearray | memoryview):
        # A binary scalar can encode credential text as UTF-8, UTF-16/32, or an
        # application-specific encoding and later be serialized as ``!!binary``.
        # Its contents therefore cannot be proven credential-free at this
        # durable boundary. Configuration is text-shaped, so fail closed rather
        # than guessing one encoding and leaving another bypass.
        return True
    return (
        (
            isinstance(value, str)
            and (
                is_path_like
                or path_context
                or _PATH_NORMALIZED_URI_RE.search(value) is not None
            )
            and _path_text_has_inline_secret(value, budget=budget)
        )
        or _is_url_with_inline_secret(value, budget=budget)
        or _has_inline_authorization(value, budget=budget)
        or _has_inline_sensitive_assignment(value, budget=budget)
        or _has_inline_private_key_block(value, budget=budget)
        or _has_userinfo_without_authority(value, budget=budget)
    )


def _mapping_key_has_inline_secret(
    key: Any, *, budget: _ScanBudget | None = None, depth: int = 0
) -> bool:
    """Return whether a hashable mapping key recursively embeds a credential.

    Config mappings normally use scalar string keys, but YAML and direct Python
    callers can also supply hashable tuple or frozenset keys. Treat those
    containers as an atomic key and inspect their members without ever using a
    potentially secret-bearing key in a diagnostic path.
    """
    if budget is not None:
        budget.consume_node(depth=depth)
        budget.consume_scalar_bytes(key)
    if isinstance(key, tuple | frozenset):
        found = False
        for item in key:
            found = (
                _mapping_key_has_inline_secret(item, budget=budget, depth=depth + 1)
                or found
            )
        return found
    # Mapping keys are used verbatim when field paths are rendered. Treat them
    # as path-like so relative signed-query and fragment forms are rejected as
    # atomically as absolute credential-bearing URLs; otherwise the key itself
    # can become the leak when a nested value produces an error diagnostic.
    return _string_has_inline_secret(key, path_context=True, budget=budget)


_DIAGNOSTIC_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_CREDENTIAL_METADATA_KEY_TOKENS = {
    "audience",
    "backend",
    "id",
    "kind",
    "provider",
    "scheme",
    "scope",
    "source",
    "type",
    "version",
}


def _diagnostic_mapping_path(path: str, key: Any, ordinal: int) -> str:
    """Render a bounded, collision-free path segment without custom reprs."""
    if isinstance(key, str) and len(key) <= 128:
        if _DIAGNOSTIC_IDENTIFIER_RE.fullmatch(key):
            return f"{path}.{key}" if path else key
        segment = json.dumps(key, ensure_ascii=True)
    elif isinstance(key, bool):
        segment = f"bool:{str(key).lower()}"
    elif isinstance(key, int):
        segment = str(key) if key.bit_length() <= 256 else f"int#{ordinal}"
    elif key is None:
        segment = "null"
    else:
        segment = f"key#{ordinal}"
    return f"{path}[{segment}]" if path else f"[{segment}]"


def _credential_container_scalar_is_reference(key: Any, value: Any) -> bool:
    """Recognize metadata/reference leaves nested below credential containers."""
    if not isinstance(key, str):
        return False
    tokens = _config_key_tokens(key)
    if not tokens:
        return False
    final = tokens[-1]
    if final in {"env", "var", "vars"} or tokens[-2:] in {
        ("env", "name"),
        ("env", "names"),
        ("env", "var"),
        ("env", "vars"),
    }:
        return _is_valid_env_reference(value)
    return final in _REFERENCE_KEY_SUFFIXES | _CREDENTIAL_METADATA_KEY_TOKENS


def _is_credential_container_key(key: str) -> bool:
    """Return whether a sensitive key conventionally owns child descriptors."""
    tokens = _config_key_tokens(key)
    return bool(
        tokens
        and _is_sensitive_config_key(key)
        and (
            tokens[-1].endswith("s")
            or "".join(tokens).endswith(("apikeys", "accesskeyids"))
        )
    )


def find_inline_secret_paths(
    value: Any,
    *,
    _path_context: bool = False,
    limits: CredentialScanLimits | None = None,
    _credential_unordered_container_ids: set[int] | None = None,
    _scan_budget: _ScanBudget | None = None,
) -> tuple[str, ...]:
    """Return paths of live inline credentials without returning their values.

    Unordered containers are an atomic security boundary: when any member of a
    ``set`` or ``frozenset`` contains a credential, report only the container's
    stable parent path. Inventing positional paths for unordered members would
    make diagnostics and credential transport depend on hash iteration order.
    """
    findings: list[str] = []
    budget = _scan_budget or _ScanBudget(limits or DEFAULT_CREDENTIAL_SCAN_LIMITS)
    active_container_ids: set[int] = set()

    def walk(
        item: Any,
        path: str,
        *,
        depth: int,
        record_findings: bool = True,
        path_context: bool = False,
        credential_context: bool = False,
        already_accounted: bool = False,
    ) -> bool:
        if not already_accounted:
            budget.consume_node(depth=depth)
            budget.consume_scalar_bytes(item)
        found = False
        if isinstance(item, Mapping):
            container_id = id(item)
            if container_id in active_container_ids:
                return False
            active_container_ids.add(container_id)
            keys: list[Any] = []
            # A credential-bearing mapping key cannot be represented safely in
            # a field path or partially retained in a sanitized mapping. Treat
            # the whole mapping as one atomic security boundary and report only
            # its stable, value-free parent path.
            try:
                for key in item:
                    if _mapping_key_has_inline_secret(
                        key, budget=budget, depth=depth + 1
                    ):
                        if record_findings:
                            findings.append(path or "$")
                        return True
                    keys.append(key)
                for ordinal, key in enumerate(keys):
                    child = item[key]
                    child_path = _diagnostic_mapping_path(path, key, ordinal)
                    string_key = key if isinstance(key, str) else ""
                    # A path context applies to direct scalar values and scalar
                    # collection members. Structured records establish their own
                    # field roles so a benign description nested under ``paths``
                    # is not treated as a path; an explicit nested path key
                    # re-enables the context.
                    child_path_context = bool(
                        string_key and _is_path_reference_key(string_key)
                    )
                    sensitive_key = bool(
                        string_key and _is_sensitive_config_key(string_key)
                    )
                    credential_container_key = bool(
                        string_key and _is_credential_container_key(string_key)
                    )
                    structured_child = isinstance(
                        child, Mapping | list | tuple | set | frozenset
                    )
                    budget.consume_node(depth=depth + 1)
                    budget.consume_scalar_bytes(child)
                    if (
                        string_key
                        and _is_sensitive_env_reference_key(string_key)
                        and _is_invalid_durable_env_reference(child)
                    ):
                        child_has_secret = True
                    elif (
                        sensitive_key
                        and not structured_child
                        and _is_inline_secret_value(child)
                    ):
                        child_has_secret = True
                    elif (
                        sensitive_key
                        and structured_child
                        and not credential_container_key
                    ):
                        child_has_secret = True
                    elif (
                        credential_context
                        and not structured_child
                        and _is_inline_secret_value(child)
                        and not _credential_container_scalar_is_reference(key, child)
                    ):
                        child_has_secret = True
                    else:
                        child_has_secret = walk(
                            child,
                            child_path,
                            depth=depth + 1,
                            record_findings=record_findings,
                            path_context=child_path_context,
                            credential_context=(
                                credential_context or credential_container_key
                            ),
                            already_accounted=True,
                        )
                        found = child_has_secret or found
                        continue
                    if record_findings:
                        findings.append(child_path)
                    found = child_has_secret or found
            finally:
                active_container_ids.remove(container_id)
        elif isinstance(item, list | tuple):
            container_id = id(item)
            if container_id in active_container_ids:
                return False
            active_container_ids.add(container_id)
            try:
                for index, child in enumerate(item):
                    child_path = f"{path}[{index}]" if path else f"[{index}]"
                    budget.consume_node(depth=depth + 1)
                    budget.consume_scalar_bytes(child)
                    if (
                        credential_context
                        and not isinstance(
                            child, Mapping | list | tuple | set | frozenset
                        )
                        and _is_inline_secret_value(child)
                    ):
                        child_found = True
                        if record_findings:
                            findings.append(child_path)
                    else:
                        child_found = walk(
                            child,
                            child_path,
                            depth=depth + 1,
                            record_findings=record_findings,
                            path_context=path_context,
                            credential_context=credential_context,
                            already_accounted=True,
                        )
                    found = child_found or found
            finally:
                active_container_ids.remove(container_id)
        elif isinstance(item, set | frozenset):
            # A set has no stable member index. Inspect every member without
            # recording its traversal path, then collapse any finding to the
            # deterministic, value-free container path.
            for child in item:
                budget.consume_node(depth=depth + 1)
                budget.consume_scalar_bytes(child)
                if (
                    credential_context
                    and not isinstance(child, Mapping | list | tuple | set | frozenset)
                    and _is_inline_secret_value(child)
                ):
                    child_found = True
                else:
                    child_found = walk(
                        child,
                        path,
                        depth=depth + 1,
                        record_findings=False,
                        path_context=path_context,
                        credential_context=credential_context,
                        already_accounted=True,
                    )
                found = child_found or found
            if found:
                if _credential_unordered_container_ids is not None:
                    _credential_unordered_container_ids.add(id(item))
                if record_findings:
                    findings.append(path or "$")
        elif _string_has_inline_secret(item, path_context=path_context, budget=budget):
            found = True
            if record_findings:
                findings.append(path or "$")
        return found

    walk(value, "", depth=0, path_context=_path_context)
    return tuple(findings)


def ensure_no_inline_secrets(
    value: Any,
    *,
    context: str = "configuration",
    path_context: bool = False,
    limits: CredentialScanLimits | None = None,
) -> None:
    """Fail closed before a durable config or artifact stores credentials.

    The exception reports only field paths. It never includes credential
    values, fragments, lengths, or fingerprints.
    """
    paths = find_inline_secret_paths(value, _path_context=path_context, limits=limits)
    if not paths:
        return
    rendered = ", ".join(paths[:8])
    if len(paths) > 8:
        rendered += f", and {len(paths) - 8} more"
    raise InlineSecretError(
        f"{context} contains inline credential values at {rendered}; "
        "use environment-variable or managed-secret references",
        paths=paths,
    )


async def aensure_no_inline_secrets(
    value: Any,
    *,
    context: str = "configuration",
    path_context: bool = False,
    limits: CredentialScanLimits | None = None,
) -> None:
    """Run the bounded credential scan outside an async caller's event loop."""
    await asyncio.to_thread(
        ensure_no_inline_secrets,
        value,
        context=context,
        path_context=path_context,
        limits=limits,
    )


def redact_sensitive_config(
    value: Any,
    *,
    _path_context: bool = False,
    limits: CredentialScanLimits | None = None,
) -> Any:
    """Return a recursively redacted logging-safe copy of config-like data.

    Credential-bearing unordered containers are replaced atomically. Partial
    member redaction could collapse distinct set members and cannot preserve a
    stable path for later diagnostics. A complete bounded scan runs first. The
    scan and copy traversal spend from one aggregate budget, so neither repeated
    detector work nor projection can exceed the caller's limits after preflight.
    """
    scan_limits = limits or DEFAULT_CREDENTIAL_SCAN_LIMITS
    budget = _ScanBudget(scan_limits)
    credential_unordered_container_ids: set[int] = set()
    find_inline_secret_paths(
        value,
        _path_context=_path_context,
        limits=scan_limits,
        _credential_unordered_container_ids=credential_unordered_container_ids,
        _scan_budget=budget,
    )
    memo: dict[tuple[int, bool, bool], Any] = {}

    def redact(
        item: Any,
        *,
        depth: int,
        path_context: bool,
        credential_context: bool = False,
        already_accounted: bool = False,
    ) -> Any:
        if not already_accounted:
            budget.consume_node(depth=depth)
            budget.consume_scalar_bytes(item)
        structured_item = isinstance(item, Mapping | list | tuple | set | frozenset)
        if credential_context and not structured_item and _is_inline_secret_value(item):
            return _INLINE_SECRET_REDACTION
        if _string_has_inline_secret(
            item,
            path_context=path_context,
            budget=budget,
        ):
            return _INLINE_SECRET_REDACTION
        if isinstance(item, Mapping):
            item_id = id(item)
            memo_key = (item_id, credential_context, path_context)
            if memo_key in memo:
                return memo[memo_key]
            # Never retain a credential-bearing key, even with a redacted value.
            if any(
                _mapping_key_has_inline_secret(
                    key,
                    budget=budget,
                    depth=depth + 1,
                )
                for key in item
            ):
                return _INLINE_SECRET_REDACTION
            redacted_mapping: dict[Any, Any] = {}
            memo[memo_key] = redacted_mapping
            for key, child in item.items():
                string_key = key if isinstance(key, str) else ""
                child_path_context = bool(
                    string_key and _is_path_reference_key(string_key)
                )
                sensitive_key = bool(
                    string_key and _is_sensitive_config_key(string_key)
                )
                credential_container_key = bool(
                    string_key and _is_credential_container_key(string_key)
                )
                structured_child = isinstance(
                    child, Mapping | list | tuple | set | frozenset
                )
                budget.consume_node(depth=depth + 1)
                budget.consume_scalar_bytes(child)
                if (
                    string_key
                    and _is_sensitive_env_reference_key(string_key)
                    and _is_invalid_durable_env_reference(child)
                ):
                    redacted_mapping[key] = _INLINE_SECRET_REDACTION
                elif (
                    sensitive_key
                    and not structured_child
                    and _is_inline_secret_value(child)
                ):
                    redacted_mapping[key] = _INLINE_SECRET_REDACTION
                elif (
                    sensitive_key and structured_child and not credential_container_key
                ):
                    redacted_mapping[key] = _INLINE_SECRET_REDACTION
                elif (
                    credential_context
                    and not structured_child
                    and _is_inline_secret_value(child)
                    and not _credential_container_scalar_is_reference(key, child)
                ):
                    redacted_mapping[key] = _INLINE_SECRET_REDACTION
                else:
                    child_credential_context = (
                        credential_context or credential_container_key
                    )
                    if (
                        credential_context
                        and not structured_child
                        and _credential_container_scalar_is_reference(key, child)
                    ):
                        child_credential_context = False
                    redacted_mapping[key] = redact(
                        child,
                        depth=depth + 1,
                        path_context=child_path_context,
                        credential_context=child_credential_context,
                        already_accounted=True,
                    )
            return redacted_mapping
        if isinstance(item, list):
            item_id = id(item)
            memo_key = (item_id, credential_context, path_context)
            if memo_key in memo:
                return memo[memo_key]
            redacted_list: list[Any] = []
            memo[memo_key] = redacted_list
            redacted_list.extend(
                redact(
                    child,
                    depth=depth + 1,
                    path_context=path_context,
                    credential_context=credential_context,
                )
                for child in item
            )
            return redacted_list
        if isinstance(item, tuple):
            item_id = id(item)
            memo_key = (item_id, credential_context, path_context)
            cached = memo.get(memo_key)
            if cached is not None:
                return cached
            redacted_items = [
                redact(
                    child,
                    depth=depth + 1,
                    path_context=path_context,
                    credential_context=credential_context,
                )
                for child in item
            ]
            recursive_clone = memo.get(memo_key)
            if recursive_clone is not None:
                return recursive_clone
            redacted_tuple = tuple(redacted_items)
            memo[memo_key] = redacted_tuple
            return redacted_tuple
        if isinstance(item, set | frozenset):
            if id(item) in credential_unordered_container_ids:
                return _INLINE_SECRET_REDACTION
            for child in item:
                budget.consume_node(depth=depth + 1)
                budget.consume_scalar_bytes(child)
            return item.copy()
        return item

    return redact(value, depth=0, path_context=_path_context)


def redact_sensitive_path(value: str | os.PathLike[str] | None) -> str:
    """Return path text safe for logs and diagnostic metadata.

    Runtime path objects must retain their original value for resolution and
    I/O. Diagnostic surfaces should instead use this string projection so a
    path containing URI userinfo, a signed query, or inline authorization
    material cannot be logged or persisted verbatim.
    """
    path_text = os.fspath(value) if isinstance(value, os.PathLike) else value
    return str(redact_sensitive_config(path_text, _path_context=True))


def resolve_path_with_safe_diagnostics(
    value: str | os.PathLike[str], *, label: str
) -> Path:
    """Resolve a runtime path without exposing it through OS diagnostics.

    The returned path retains its original runtime value. Only resolution
    failures are projected to a diagnostic-safe representation so symlink
    loops and filesystem errors cannot echo credential-bearing path text.
    """
    path = Path(value)
    safe_path = redact_sensitive_path(path)
    runtime_failure = False
    os_failure: tuple[type[OSError], int | None] | None = None
    try:
        return path.resolve()
    except RuntimeError:
        runtime_failure = True
    except OSError as error:
        os_failure = (type(error), error.errno)
    if runtime_failure:
        raise RuntimeError(f"Unable to resolve {label}: {safe_path}")
    assert os_failure is not None
    error_type, error_number = os_failure
    raise error_type(
        error_number,
        f"Unable to resolve {label}",
        safe_path,
    )


def create_directory_with_safe_diagnostics(
    value: str | os.PathLike[str],
    *,
    label: str,
    parents: bool = True,
    exist_ok: bool = True,
) -> Path:
    """Create a runtime directory without exposing it through OS diagnostics."""
    path = Path(value)
    safe_path = redact_sensitive_path(path)
    runtime_failure = False
    os_failure: tuple[type[OSError], int | None] | None = None
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except RuntimeError:
        runtime_failure = True
    except OSError as error:
        os_failure = (type(error), error.errno)
    if runtime_failure:
        raise RuntimeError(f"Unable to create {label}: {safe_path}")
    if os_failure is not None:
        error_type, error_number = os_failure
        raise error_type(
            error_number,
            f"Unable to create {label}",
            safe_path,
        )
    return path


def path_exists_with_safe_diagnostics(
    value: str | os.PathLike[str], *, label: str
) -> bool:
    """Inspect a runtime path without exposing it through OS diagnostics."""
    path = Path(value)
    safe_path = redact_sensitive_path(path)
    os_failure: tuple[type[OSError], int | None] | None = None
    try:
        return path.exists()
    except OSError as error:
        os_failure = (type(error), error.errno)
    assert os_failure is not None
    error_type, error_number = os_failure
    raise error_type(
        error_number,
        f"Unable to inspect {label}",
        safe_path,
    )


def path_is_file_with_safe_diagnostics(
    value: str | os.PathLike[str], *, label: str
) -> bool:
    """Inspect whether a runtime path is a file with value-safe diagnostics."""
    path = Path(value)
    safe_path = redact_sensitive_path(path)
    os_failure: tuple[type[OSError], int | None] | None = None
    try:
        return path.is_file()
    except OSError as error:
        os_failure = (type(error), error.errno)
    assert os_failure is not None
    error_type, error_number = os_failure
    raise error_type(
        error_number,
        f"Unable to inspect {label}",
        safe_path,
    )


def read_text_with_safe_diagnostics(
    value: str | os.PathLike[str],
    *,
    label: str,
    encoding: str = "utf-8",
) -> str:
    """Read a runtime text file without exposing it through OS diagnostics."""
    path = Path(value)
    safe_path = redact_sensitive_path(path)
    os_failure: tuple[type[OSError], int | None] | None = None
    try:
        return path.read_text(encoding=encoding)
    except OSError as error:
        os_failure = (type(error), error.errno)
    assert os_failure is not None
    error_type, error_number = os_failure
    raise error_type(
        error_number,
        f"Unable to read {label}",
        safe_path,
    )


def is_local_nim_api_key_placeholder(value: Any) -> bool:
    """Return True only for the explicit local NIM no-auth opt-in value."""
    return (
        isinstance(value, str)
        and value.strip().lower() == LOCAL_NIM_API_KEY_PLACEHOLDER
    )


_VLM_NIM_ENV_BASE_URL_VARS = (
    "WU_VLM_NIM_BASE_URL",
    "PA_VLM_NIM_BASE_URL",
    "TA_VLM_NIM_BASE_URL",
    "MA_VLM_NIM_BASE_URL",
)
_LLM_NIM_ENV_BASE_URL_VARS = (
    "WU_LLM_NIM_BASE_URL",
    "PA_LLM_NIM_BASE_URL",
    "TA_LLM_NIM_BASE_URL",
    "MA_LLM_NIM_BASE_URL",
    *_VLM_NIM_ENV_BASE_URL_VARS,
)
_NIM_API_KEY_ENV_VARS = (
    "WU_NIM_API_KEY",
    "PA_NIM_API_KEY",
    "TA_NIM_API_KEY",
    "MA_NIM_API_KEY",
)


def _first_env_value(env_vars: tuple[str, ...]) -> str | None:
    for var in env_vars:
        value = os.getenv(var)
        if value:
            return value
    return None


def get_vlm_nim_env_base_url_override() -> str | None:
    """Return the runtime VLM NIM env base-URL override, if any."""
    return _first_env_value(_VLM_NIM_ENV_BASE_URL_VARS)


def get_llm_nim_env_base_url_override() -> str | None:
    """Return the runtime LLM NIM env base-URL override, if any.

    Agent-specific aliases are accepted for backwards compatibility:
    ``MA_*`` for material-agent, ``PA_*`` for physics-agent, ``TA_*`` for
    texture-agent, plus neutral ``WU_*`` variables. ``*_LLM_NIM_BASE_URL``
    is preferred over ``*_VLM_NIM_BASE_URL`` for LLM routing.
    """
    return _first_env_value(_LLM_NIM_ENV_BASE_URL_VARS)


def apply_vlm_nim_env_override(vlm_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``vlm_config`` with the runtime VLM NIM override applied."""
    config = dict(vlm_config)
    nim_base_url = get_vlm_nim_env_base_url_override()
    if not nim_base_url:
        return config
    backend = (config.get("backend") or config.get("provider") or "").strip().lower()
    if backend in ("", "echo", "mock"):
        return config
    drop_stale_endpoint_credentials(config, preserve_local_nim_placeholder=True)
    config["backend"] = "nim"
    config["base_url"] = nim_base_url
    return config


def apply_llm_nim_env_override(llm_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``llm_config`` with the runtime LLM NIM override applied.

    When a ``*_LLM_NIM_BASE_URL`` or ``*_VLM_NIM_BASE_URL`` alias is set, the
    section is forced to ``backend: nim`` and the env-supplied ``base_url``;
    any endpoint-scoped fields from the prior backend are dropped (with the
    explicit local-NIM no-auth placeholder preserved). When neither env var is
    set, ``llm_config`` is returned unchanged (still copied to avoid mutating
    the caller's dict).

    Mock / echo configs and configs with no ``backend`` are left alone — the
    override is a *runtime routing* hint that should not silently turn a
    deliberately-mocked simulate run into a real NIM call when the operator
    happens to also have ``MA_LLM_NIM_BASE_URL`` set in the environment.
    """
    config = dict(llm_config)
    nim_base_url = get_llm_nim_env_base_url_override()
    if not nim_base_url:
        return config
    backend = (config.get("backend") or config.get("provider") or "").strip().lower()
    if backend in ("", "echo", "mock"):
        return config
    drop_stale_endpoint_credentials(config, preserve_local_nim_placeholder=True)
    config["backend"] = "nim"
    config["base_url"] = nim_base_url
    return config


def drop_stale_endpoint_credentials(
    model_config: dict[str, Any],
    *,
    preserve_local_nim_placeholder: bool = False,
) -> None:
    """Drop endpoint-scoped keys left over from a previous backend.

    Whenever ``backend`` or ``base_url`` on a model section is rewritten
    (env override, runtime NIM-base-URL injection, service-side route
    selection), the prior ``api_key``/``api_key_env`` and ``base_url`` belonged
    to a different endpoint. Leaving them in place can forward one provider's
    credential to another endpoint or send traffic to the old URL while
    validation reports success.

    Args:
        model_config: Model section dict (mutated in place).
        preserve_local_nim_placeholder: If True, keep an existing ``api_key``
            equal to the local NIM no-auth placeholder. Use this when
            reusing the section for a NIM endpoint where the operator has
            explicitly opted into no-auth.
    """
    if not (
        preserve_local_nim_placeholder
        and is_local_nim_api_key_placeholder(model_config.get("api_key"))
    ):
        model_config.pop("api_key", None)
    model_config.pop("api_key_env", None)
    model_config.pop("base_url", None)


def resolve_endpoint_api_key(
    api_key: Any = None,
    api_key_env: Any = None,
    *,
    prefer_env: bool = False,
    require_env: bool = False,
) -> str | None:
    """Resolve an endpoint-scoped API key from an inline value or env reference.

    Resolution order:
    - ``prefer_env=True`` reads ``api_key_env`` first and returns ``None`` when
      the named variable is unset, so service readiness can fail fast on a
      configured-but-missing env key instead of falling back to inline config.
    - Default callers prefer a real inline ``api_key`` and then fall back to
      ``api_key_env``. Generic placeholders are treated as unset, while
      ``not-used`` remains a valid explicit local no-auth opt-in.
    - ``require_env=True`` raises a value-free diagnostic when
      ``api_key_env`` is configured but unset and no valid inline fallback is
      available. With ``prefer_env=True``, the configured environment value
      remains authoritative and a missing value does not fall back inline.
      The configured reference stays available to the runtime lookup but is
      not copied into the exception.
    """

    env_name = parse_env_reference(api_key_env, allow_legacy_bare=True)
    env_configured = env_name is not None
    env_value = os.getenv(env_name) if env_name is not None else None
    api_key_str: str | None = None
    if prefer_env:
        if env_configured:
            if require_env and not env_value:
                # This branch deliberately does not use an inline fallback:
                # prefer_env makes the configured endpoint reference
                # authoritative.
                del api_key, api_key_env, env_name, env_value
                _raise_missing_endpoint_env()
            return env_value

    if api_key is not None:
        api_key_str = str(api_key).strip()
        if api_key_str and (
            is_local_nim_api_key_placeholder(api_key_str)
            or not is_placeholder_api_key(api_key_str)
        ):
            return api_key_str

    if env_value:
        return env_value

    if not (require_env and env_configured):
        return None

    # This frame is retained by the replacement exception. Remove the legacy
    # reference, explicit reference, inline fallback, and resolved value before
    # delegating the value-free raise.
    del api_key, api_key_env, api_key_str, env_name, env_value
    _raise_missing_endpoint_env()


def is_local_base_url(base_url: Any) -> bool:
    """Return True for local or cluster-private service endpoints."""
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    parsed = urlparse(base_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False

    host = parsed.hostname
    if not host:
        return False
    host_lower = host.lower()
    if host_lower in _LOCAL_HOSTNAMES:
        return True
    if "." not in host_lower:
        return True
    if host_lower.endswith((".local", ".svc", ".svc.cluster.local")):
        return True

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def get_env_api_key_for_backend(
    backend: str,
    explicit_api_key: Any = None,
) -> str | None:
    """Resolve a backend API key from explicit input or environment aliases."""
    if explicit_api_key is not None:
        explicit_api_key_str = str(explicit_api_key).strip()
        if explicit_api_key_str and not is_placeholder_api_key(explicit_api_key_str):
            return explicit_api_key_str

    for env_var in API_KEY_ENV_VAR_MAP.get(backend, ()):
        api_key = os.getenv(env_var)
        if api_key and not is_placeholder_api_key(api_key):
            return api_key

    return None


def get_nim_api_key_for_base_url(
    base_url: Any,
    explicit_api_key: Any = None,
) -> str | None:
    """Resolve a NIM API key with explicit local sidecar no-auth opt-in.

    Hosted NVIDIA NIM endpoints require a real key from explicit config or
    ``NVIDIA_API_KEY``. Non-hosted NIM endpoints (in-cluster sidecars and
    operator-configured external NIM URLs such as the helm chart's
    ``vlmNim.endpointOverride``) use a NIM-scoped API key alias as the
    explicit opt-in; the hosted ``NVIDIA_API_KEY`` is never silently
    forwarded to a non-NVIDIA NIM endpoint. No-auth NIM endpoints may use the
    ``not-used`` placeholder, but only when explicitly supplied via config or
    one of the NIM-scoped API key aliases.
    """
    if explicit_api_key is not None:
        explicit_api_key_str = str(explicit_api_key).strip()
        if explicit_api_key_str and not is_placeholder_api_key(explicit_api_key_str):
            return explicit_api_key_str

    is_local_endpoint = is_local_base_url(base_url)
    if is_local_endpoint and is_local_nim_api_key_placeholder(explicit_api_key):
        return LOCAL_NIM_API_KEY_PLACEHOLDER

    is_nvidia_endpoint = is_nvidia_provider_base_url(base_url)
    nim_api_key = _first_env_value(_NIM_API_KEY_ENV_VARS)
    if not is_nvidia_endpoint:
        # Non-hosted NIM (local sidecar or custom remote NIM URL): the
        # operator must opt in with a NIM-scoped key (real value or the
        # ``not-used`` no-auth placeholder). ``NVIDIA_API_KEY`` is never
        # silently forwarded to non-NVIDIA endpoints.
        if nim_api_key and not is_placeholder_api_key(nim_api_key):
            return nim_api_key
        if is_local_nim_api_key_placeholder(nim_api_key):
            return LOCAL_NIM_API_KEY_PLACEHOLDER
        return None

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_api_key and not is_placeholder_api_key(nvidia_api_key):
        return nvidia_api_key

    return None


def resolve_effective_openai_base_url(explicit: Any = None) -> str | None:
    """Return the OpenAI base URL the SDK will actually hit.

    ``langchain_openai.ChatOpenAI`` (and the underlying ``openai`` SDK) fall
    back to ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` when the constructor
    receives no explicit ``base_url``. Endpoint-based credential checks must
    use this effective URL or the hosted ``OPENAI_API_KEY`` could be sent to
    an env-redirected custom endpoint.
    """
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")


def get_openai_api_key_for_base_url(
    base_url: Any,
    explicit_api_key: Any = None,
) -> str | None:
    """Resolve an OpenAI-compatible API key with local no-auth support.

    Hosted OpenAI endpoints require a real key from explicit config or
    ``OPENAI_API_KEY``. Custom remote OpenAI-compatible endpoints require
    explicit config so provider credentials are not sent to arbitrary compatible
    services. Local endpoints may use the documented ``not-used`` dummy key only
    when it is explicitly supplied in config, so authenticated private gateways
    do not get silently treated as no-auth services.

    The check is performed against the *effective* base URL — when no
    ``base_url`` is supplied the OpenAI SDK falls back to ``OPENAI_BASE_URL``
    / ``OPENAI_API_BASE``, and a hosted ``OPENAI_API_KEY`` must not be
    forwarded to that env-redirected endpoint. An explicit ``api_key`` is
    accepted only when the caller also explicitly chose the endpoint (config
    ``base_url``) or the effective endpoint is provider-owned/local; an
    env-redirected custom endpoint paired with an explicit ``api_key`` is
    rejected so the caller's hosted key cannot follow an unintended redirect.
    """
    config_supplied_base_url = isinstance(base_url, str) and bool(base_url.strip())
    effective_base_url = resolve_effective_openai_base_url(base_url)
    is_local_endpoint = is_local_base_url(effective_base_url)
    is_provider_endpoint = is_openai_provider_base_url(effective_base_url)

    explicit_api_key_str = (
        str(explicit_api_key).strip() if explicit_api_key is not None else None
    )
    if explicit_api_key_str is not None:
        if explicit_api_key_str and not is_placeholder_api_key(explicit_api_key_str):
            # Trust an explicit api_key only when the user paired it with an
            # explicit endpoint, or the resolved endpoint is provider-owned.
            # Local endpoints require explicit pairing too — a malicious
            # ``OPENAI_BASE_URL=http://attacker.local/v1`` would otherwise
            # exfiltrate a hosted key to a non-pair-validated host.
            if config_supplied_base_url or is_provider_endpoint:
                return explicit_api_key_str
            return None
        if is_local_endpoint and is_local_nim_api_key_placeholder(explicit_api_key_str):
            return LOCAL_NIM_API_KEY_PLACEHOLDER

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not is_provider_endpoint:
        # Non-provider endpoints (local, custom remote, env-redirected) never
        # silently inherit the hosted ``OPENAI_API_KEY``. The trust boundary
        # for forwarding the hosted key is the OpenAI provider URL set —
        # local servers, even when explicitly paired in config, must opt in
        # via an endpoint-scoped ``api_key`` or the ``not-used`` no-auth
        # placeholder. Callers like ``wu image-gen --base-url <local>``
        # inject ``not-used`` after this returns ``None``.
        return None

    if openai_api_key and not is_placeholder_api_key(openai_api_key):
        return openai_api_key

    return None


def openai_missing_credential_message(base_url: Any, model_type: str) -> str:
    """Return endpoint-aware, value-free OpenAI credential guidance."""
    explicit_base_url = isinstance(base_url, str) and bool(base_url.strip())
    effective_base_url = resolve_effective_openai_base_url(base_url)
    if effective_base_url and not is_openai_provider_base_url(effective_base_url):
        if not explicit_base_url:
            return OPENAI_ENV_REDIRECT_CREDENTIAL_MESSAGE
        if is_local_base_url(effective_base_url):
            return (
                "The configured local OpenAI-compatible endpoint requires an "
                "endpoint-scoped api_key or api_key_env; documented local "
                "no-auth endpoints may use api_key: not-used."
            )
        return (
            "The configured OpenAI-compatible endpoint requires an "
            "endpoint-scoped api_key or api_key_env paired with base_url."
        )
    return f"OPENAI_API_KEY not set for openai {model_type}"
