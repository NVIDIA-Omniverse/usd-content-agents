# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""usd-cli OVRTX rendering API — a thin HTTP wrapper over usd_core's local OVRTX backend.

Speaks the contract usd_core.render.remote.RemoteRenderBackend expects
(`POST /render`, `GET /health`). Deliberately simpler than world-understanding's
Kit-compatible service: single-modality RGB, no frame ranges / sensors / S3 / ZIP /
multi-GPU dispatch.
"""
