# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Agent FastAPI Service."""

from .utils import get_version

__version__ = get_version()

__all__ = ["__version__"]
