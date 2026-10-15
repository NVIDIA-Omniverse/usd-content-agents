# OVRTX runtime dependencies and distribution

The repository contains dependency declarations, pinned download URLs, hashes,
and license notices for `ovrtx` and `ovstage`. Their wheel archives and installed
native libraries are not vendored in the Git source tree. This statement is
specific to these packages: the repository does include binary USD assets,
textures, images, and environment maps.

## Installation and container images

The qualified renderer profile pins:

| Package | Version | Download source |
| --- | --- | --- |
| ovrtx | 0.4.1.364340 | NVIDIA Python package index, `https://pypi.nvidia.com/ovrtx/` |
| ovstage | 0.1.1.355824 | NVIDIA Python package index, `https://pypi.nvidia.com/ovstage/` |

The [runtime lock](../world_understanding/functions/graphics/pylock.ovrtx-runtime.toml)
records exact artifact URLs and SHA-256 hashes. The local renderer bootstrap in
[`render_ovrtx.py`](../world_understanding/functions/graphics/render_ovrtx.py)
downloads and installs this profile into an isolated environment. The USD CLI
renderer has its own [pinned profile](../apps/usd_cli/src/usd_core/render/pylock.ovrtx-runtime.toml).
A remote-rendering client uses packages installed on the rendering server.

The [WU rendering-service Dockerfile](../apps/ovrtx_rendering_api/Dockerfile)
and [USD CLI adapter Dockerfile](../apps/usd_cli/apps/ovrtx_rendering_api/Dockerfile)
install these dependencies during image construction. Resulting images contain
the installed packages. Consequently, source checkout distribution and renderer
container distribution must be described separately; neither package should be
classified as absent from the renderer runtime merely because its wheel is not
checked into Git. This document describes the build behavior, not which image
tags have been published.

## Licenses and notices

Both versions are identified as `LicenseRef-NvidiaProprietary` in the
[third-party notice](../THIRD_PARTY_NOTICE.md#ovrtx-rendering-worker). That file
reproduces the exact-wheel license texts and bundled component notices, including
the NVIDIA Software License Agreement and applicable product-specific terms.
The repository's Apache-2.0 code license does not replace those package terms.
Preserve the package license files and third-party notices when packaging the
installed runtime; redistribution review must include the resulting artifacts.

Platform artifacts recorded in a lock do not expand the supported platform
matrix. See the root README for the supported Linux and WSL2 workflows.
