# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secret-safe representations for mutable public result containers."""

from __future__ import annotations


class SecretSafeReprMixin:
    """Represent a result without reading or formatting any mutable field."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"

    def __str__(self) -> str:
        return SecretSafeReprMixin.__repr__(self)
