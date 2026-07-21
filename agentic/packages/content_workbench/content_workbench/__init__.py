# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content Workbench service package."""

from .main import app, create_app

__all__ = ["app", "create_app"]
