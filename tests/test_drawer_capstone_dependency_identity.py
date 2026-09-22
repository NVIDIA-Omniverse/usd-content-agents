# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content binding for generated USD that retains textures inside source USDZ."""
from pathlib import Path
import zipfile

import pytest
from pxr import UsdUtils

from content_agent_workflows.simready.asset_identity import (
    AssetDependencyIdentityError,
    _dependency_path,
    asset_dependency_identity_errors,
    build_asset_dependency_manifest,
)


def packaged_texture(tmp_path):
    texture = tmp_path/'texture.bin'
    texture.write_bytes(b'original local texture bytes')
    source = tmp_path/'source.usda'
    source.write_text('#usda 1.0\ndef Xform "Asset" {\n'
                      'custom asset texture = @texture.bin@\n}\n')
    package = tmp_path/'source.usdz'
    assert UsdUtils.CreateNewUsdzPackage(str(source), str(package))
    with zipfile.ZipFile(package) as archive:
        member = next(n for n in archive.namelist() if n.endswith('texture.bin'))
    return package, member


def test_flattened_derivative_binds_packaged_texture_through_archive(tmp_path):
    package, member = packaged_texture(tmp_path)
    derivative = tmp_path/'physics.usda'
    derivative.write_text('#usda 1.0\ndef Xform "Asset" {\n'
                          f'custom asset texture = @{package}[{member}]@\n}}\n')
    manifest = build_asset_dependency_manifest(derivative)
    assert {Path(f['path']) for f in manifest['files']} == {derivative, package}
    assert asset_dependency_identity_errors(derivative, manifest) == []
    # Changing archive bytes must invalidate the previously frozen identity even
    # when the root USDA and its authored member locator are unchanged.
    package.write_bytes(package.read_bytes()+b'archive changed')
    assert asset_dependency_identity_errors(derivative, manifest)


def test_relative_packaged_texture_binds_same_local_archive(tmp_path):
    package, member = packaged_texture(tmp_path)
    assert _dependency_path(f'{package.name}[{member}]', tmp_path) == package


def test_missing_member_is_not_certified_by_existing_archive(tmp_path):
    package, _ = packaged_texture(tmp_path)
    derivative = tmp_path/'missing.usda'
    derivative.write_text('#usda 1.0\ndef Xform "Asset" {\n'
                          f'custom asset texture = @{package}[missing.bin]@\n}}\n')
    with pytest.raises(AssetDependencyIdentityError):
        build_asset_dependency_manifest(derivative)


@pytest.mark.parametrize('locator', [
    'https://example.invalid/source.usdz[texture.bin]',
    'omniverse://example.invalid/source.usdz[texture.bin]',
    'source.usdz[../texture.bin]',
    'source.usdz[/texture.bin]',
    'source.usdz[folder\\texture.bin]',
    'source.usdz[inner.usdz[texture.bin]]',
    'source.zip[texture.bin]',
    'source.usdz[texture.bin',
    'source.usdz[]',
])
def test_unsupported_packaged_locator_fails_closed(tmp_path, locator):
    with pytest.raises(AssetDependencyIdentityError):
        _dependency_path(locator, tmp_path)
