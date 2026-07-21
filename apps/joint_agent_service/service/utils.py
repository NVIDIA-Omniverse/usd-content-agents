# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    try:
        return version("joint-agent-service")
    except PackageNotFoundError:
        return "0.0.1-dev"


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Exclude healthchecks from access logs
        return (
            record.getMessage().find("/health") == -1
            and record.getMessage().find("/metrics") == -1
        )
