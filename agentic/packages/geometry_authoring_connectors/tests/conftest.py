# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        content_type: str = "application/json",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._content = content
        self.status_code = status_code
        self.headers = {
            "Content-Length": str(len(content)),
            "Content-Type": content_type,
            **dict(headers or {}),
        }
        self.closed = False

    @classmethod
    def json(cls, value: dict[str, Any], *, status_code: int = 200) -> FakeResponse:
        return cls(
            json.dumps(value, sort_keys=True).encode("utf-8"),
            status_code=status_code,
        )

    def iter_content(self, chunk_size: int) -> Any:
        for offset in range(0, len(self._content), chunk_size):
            yield self._content[offset : offset + chunk_size]

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    url: str
    kwargs: dict[str, Any]


class QueueTransport:
    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[RecordedRequest] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append(RecordedRequest(method=method, url=url, kwargs=kwargs))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
