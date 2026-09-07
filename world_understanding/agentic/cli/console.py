# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoding-safe text streams for agent command-line consoles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TextIO


class EncodingSafeTextIO:
    """Project writes into a dynamically selected stream's active encoding.

    The stream getter keeps Rich consoles compatible with Click's temporary
    stdout replacement during tests and command invocation. Unsupported text
    is escaped instead of raising ``UnicodeEncodeError`` and aborting a command.
    """

    def __init__(self, stream_getter: Callable[[], TextIO]) -> None:
        self._stream_getter = stream_getter

    @property
    def encoding(self) -> str:
        return getattr(self._stream_getter(), "encoding", None) or "utf-8"

    def write(self, value: str) -> int:
        stream = self._stream_getter()
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            value.encode(encoding)
        except LookupError:
            encoding = "ascii"
        except UnicodeEncodeError:
            pass
        else:
            return stream.write(value)
        projected = value.encode(encoding, errors="backslashreplace").decode(encoding)
        return stream.write(projected)

    def flush(self) -> None:
        self._stream_getter().flush()

    def isatty(self) -> bool:
        return self._stream_getter().isatty()

    def fileno(self) -> int:
        return self._stream_getter().fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream_getter(), name)
