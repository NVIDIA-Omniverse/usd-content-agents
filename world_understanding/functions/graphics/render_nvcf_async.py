# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated compatibility module for async REST API-based rendering.

Use :mod:`world_understanding.functions.graphics.render_remote_async` for new
code. This module name is kept because the REST renderer was originally
NVCF-only.
"""

from __future__ import annotations

import sys

from world_understanding.functions.graphics import render_remote_async as _remote_async

sys.modules[__name__] = _remote_async
