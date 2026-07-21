# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model backends for chat, VLM, and image generation models.

Public providers are always registered. Optional provider packages are loaded
through the backend entry-point contract without exposing their implementation
or credentials in this package.
"""

from . import public  # noqa: F401 -- registers public backends
from .registry import load_backend_plugins

load_backend_plugins()
