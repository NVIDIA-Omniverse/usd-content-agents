# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packaged entry point for texture-gen-simple-service."""

from apps.texture_gen_simple_service.app import app, main

__all__ = ["app", "main"]
