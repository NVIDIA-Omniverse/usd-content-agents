# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Geometry Agent service."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

try:
    __version__ = distribution_version("geometry-agent-service")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
