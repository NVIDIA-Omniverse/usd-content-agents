# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import logging.config
import os
import sys
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def configure_service_standard_streams() -> None:
    """Configure Windows service streams without replacing or owning them."""

    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # Embedders own their streams. Unsupported adaptations must leave
            # those objects and their lifecycle untouched.
            continue


def get_logging_config() -> dict[str, Any]:
    with open(
        os.getenv("LOGGING_CONFIG", "/logging.yaml"), encoding="utf-8"
    ) as logging_config_file:
        logging_config: dict[str, Any] = yaml.load(
            logging_config_file, Loader=yaml.SafeLoader
        )
    return logging_config


def setup_logging() -> None:
    # We need to override propagate and level settings for existing loggers as some of them are configured via env
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger_item in loggers:
        logger_item.propagate = True
        logger_item.level = 0
        logger_item.debug(
            "Setting up logger %s, propagate: %s, parent: %s, level: %s",
            str(logger_item.name),
            str(logger_item.propagate),
            str(logger_item.parent),
            str(logger_item.level),
        )
    try:
        config = get_logging_config()
        logger.info("Setting up logging configuration: %s", config)
        logging.config.dictConfig(config)
    except FileNotFoundError as exc_info:
        logger.warning("logging configuration file not found: %s", str(exc_info))
