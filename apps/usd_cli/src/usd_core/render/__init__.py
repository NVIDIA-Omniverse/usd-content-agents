# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVRTX render backends.

- ovrtx  — vendored OVRTX subprocess daemon; native Linux or native Windows
  with a compatible NVIDIA RTX/Vulkan stack. WSL2 is not supported.
- remote — REST client to an OVRTX render service (same interface as local).

Local OVRTX is a daemon on localhost; remote is one at a configured URL.
"""

from usd_core.render.base import RenderBackend, RenderResult
from usd_core.render.factory import make_backend

__all__ = ["RenderBackend", "RenderResult", "make_backend"]
