# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for joint_agent_service."""

# Exclude scripts/ from test collection — these are manual utilities,
# not pytest tests, and their module-level imports break collection.
collect_ignore_glob = ["scripts/*"]
