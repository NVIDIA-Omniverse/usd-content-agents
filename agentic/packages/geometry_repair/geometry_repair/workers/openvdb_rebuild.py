# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility module alias for the former OpenVDB-qualified import path."""

import sys

from . import sdf_rebuild as _implementation

sys.modules[__name__] = _implementation
