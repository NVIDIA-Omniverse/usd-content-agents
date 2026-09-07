# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""usd_cli — thin client. Parses args, POSTs to the daemon, prints stdout.

Carries no USD/render deps so it stays fast to import (cli-design.md §5.3). If sub-100ms
startup ever becomes a hard requirement, this is the only layer that gets reimplemented
in a compiled language; usd_core and usd_server stay put.
"""
