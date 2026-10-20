# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One default request deadline for every hosted model client.

Hosted endpoints can queue requests before generation starts. The shared
request deadline accounts for queueing as well as response generation.

Every VLM and chat client uses this value when its caller does not pass
``timeout``. ``WU_MODEL_REQUEST_TIMEOUT_SECONDS`` overrides it for all agents;
an explicit ``timeout`` in a model config or constructor call still wins.

NIM clients default to a longer deadline because the hosted NIM transport
retries a hung or failed attempt inside it: three 180 s
attempts plus backoff need about 550 s. The same environment override applies.
"""

from __future__ import annotations

import logging

from world_understanding.utils.environment import parse_float_env

logger = logging.getLogger(__name__)

MODEL_REQUEST_TIMEOUT_ENV = "WU_MODEL_REQUEST_TIMEOUT_SECONDS"
MODEL_REQUEST_TIMEOUT_FALLBACK_SECONDS = 300.0
HOSTED_NIM_REQUEST_TIMEOUT_FALLBACK_SECONDS = 600.0

DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = parse_float_env(
    MODEL_REQUEST_TIMEOUT_ENV,
    MODEL_REQUEST_TIMEOUT_FALLBACK_SECONDS,
    minimum=1.0,
    maximum=3600.0,
    logger=logger,
)
DEFAULT_HOSTED_NIM_REQUEST_TIMEOUT_SECONDS = parse_float_env(
    MODEL_REQUEST_TIMEOUT_ENV,
    HOSTED_NIM_REQUEST_TIMEOUT_FALLBACK_SECONDS,
    minimum=1.0,
    maximum=3600.0,
    logger=logger,
)
