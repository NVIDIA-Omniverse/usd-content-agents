# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for apps/ovrtx_rendering_api/service/renderer.py.

Focus on the pure-Python helpers — the pxr and ovrtx imports in ``render()``
are not exercised here since they require a GPU-equipped environment.
"""

from __future__ import annotations

import base64
import socket
import threading
import zipfile
from pathlib import Path

import pytest

# ``service`` is on sys.path via ``pythonpath = ["apps/ovrtx_rendering_api"]``
# in the root pyproject.toml's [tool.pytest.ini_options].
from service.renderer import (
    _ZIP_MAX_FILES,
    Renderer,
    _extract_zip_bundle,
    _fetch_usd,
    _is_usdz_payload,
    _parse_zip_max_uncompressed_bytes,
    _validate_connected_socket_peer,
    _validate_url_target,
)


def _make_bundle(tmp_path: Path, names: list[str]) -> Path:
    """Build a bundle.zip containing the given entries at the archive root."""
    src = tmp_path / "src"
    src.mkdir()
    for name in names:
        p = src / name
        p.parent.mkdir(parents=True, exist_ok=True)
        # A one-line USDA is enough — we never open the stage in these tests.
        p.write_text('#usda 1.0\ndef Xform "Root" {}\n')

    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(src / name, name)
    return zip_path


class _DummySocket:
    def __init__(self, peer_host: str) -> None:
        self.peer_host = peer_host
        self.closed = False

    def getpeername(self):
        return (self.peer_host, 443)

    def close(self) -> None:
        self.closed = True


class _FailingPeerSocket(_DummySocket):
    def __init__(self) -> None:
        super().__init__("93.184.216.34")

    def getpeername(self):
        raise OSError("peer unavailable")


class _DummyResponse:
    def __init__(
        self,
        status_code: int,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _RecordingBackend:
    def __init__(self) -> None:
        self.render_kwargs: dict | None = None
        self.base_dir_name: str | None = None
        self.stage_exists_during_render = False
        self.texture_exists_during_render = False

    def render(self, **kwargs):
        self.render_kwargs = kwargs
        base_dir = Path(kwargs["base_dir"])
        self.base_dir_name = base_dir.name
        self.stage_exists_during_render = (base_dir / "stage.usda").exists()
        self.texture_exists_during_render = (
            base_dir / "textures" / "albedo.png"
        ).exists()
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "images": [],
                    "sensors": {},
                    "frame_count": 0,
                }
            ],
        }


class TestValidateUrlTarget:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/scene.usd",
            "http://localhost:8000/scene.usd",
            "http://localhost.:8000/scene.usd",
            "http://[::1]:8000/scene.usd",
            "http://[::ffff:169.254.169.254]/latest/meta-data",
            "http://10.0.0.5/scene.usd",
            "http://169.254.169.254/latest/meta-data",
            "http://0.0.0.0:8000/scene.usd",
            "http://0:8000/scene.usd",
            "http://[::]:8000/scene.usd",
            "http://2130706433:8000/scene.usd",
            "http://0x7f000001:8000/scene.usd",
            "http://0x7f.0.0.1:8000/scene.usd",
            "http://0177.0.0.1:8000/scene.usd",
            "http://0251.0376.0251.0376/latest/meta-data",
        ],
    )
    def test_blocks_private_loopback_and_metadata_ips(self, url: str) -> None:
        with pytest.raises(ValueError, match="URL blocked"):
            _validate_url_target(url)

    def test_allows_public_hostname(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "service.renderer.socket.getaddrinfo",
            lambda *args, **kwargs: [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("93.184.216.34", 443),
                )
            ],
        )
        _validate_url_target("https://example.com/scene.usd")

    def test_blocks_hostname_that_resolves_to_private_ip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "service.renderer.socket.getaddrinfo",
            lambda *args, **kwargs: [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    0,
                    "",
                    ("127.0.0.1", 80),
                )
            ],
        )

        with pytest.raises(ValueError, match="URL blocked"):
            _validate_url_target("http://private.example/scene.usd")

    def test_connected_socket_peer_check_blocks_rebound_private_ip(self) -> None:
        sock = _DummySocket("127.0.0.1")

        with pytest.raises(ValueError, match="URL blocked"):
            _validate_connected_socket_peer(sock, "https://rebound.example")

        assert sock.closed is True

    def test_connected_socket_peer_check_blocks_unspecified_ip(self) -> None:
        sock = _DummySocket("0.0.0.0")

        with pytest.raises(ValueError, match="URL blocked"):
            _validate_connected_socket_peer(sock, "https://rebound.example")

        assert sock.closed is True

    def test_connected_socket_peer_check_allows_public_ip(self) -> None:
        sock = _DummySocket("93.184.216.34")

        _validate_connected_socket_peer(sock, "https://example.com")

        assert sock.closed is False

    def test_connected_socket_peer_check_closes_on_peer_lookup_error(self) -> None:
        sock = _FailingPeerSocket()

        with pytest.raises(OSError, match="peer unavailable"):
            _validate_connected_socket_peer(sock, "https://example.com")

        assert sock.closed is True


class TestFetchUsd:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/scene.usd",
            "ssh://example.com/scene.usd",
        ],
    )
    def test_rejects_non_http_s3_data_schemes(
        self,
        url: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_request(*args, **kwargs):
            pytest.fail("unsupported schemes must not reach requests")

        monkeypatch.setattr("service.renderer._safe_requests_get", fail_request)

        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _fetch_usd(url, str(tmp_path / "scene.usd"))

        assert not (tmp_path / "scene.usd").exists()

    def test_http_redirects_are_validated_before_writing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        responses = [
            _DummyResponse(302, headers={"Location": "/final.usda"}),
            _DummyResponse(200, content=b"#usda 1.0\n"),
        ]
        requested_urls: list[str] = []
        validated_urls: list[str] = []

        def fake_get(url: str, *, timeout: float, allow_redirects: bool):
            requested_urls.append(url)
            assert timeout == 300
            assert allow_redirects is False
            return responses.pop(0)

        monkeypatch.setattr("service.renderer._safe_requests_get", fake_get)
        monkeypatch.setattr(
            "service.renderer._validate_url_target",
            lambda url: validated_urls.append(url),
        )

        dest = tmp_path / "scene.usd"
        _fetch_usd("https://assets.example/start.usd", str(dest))

        assert dest.read_bytes() == b"#usda 1.0\n"
        assert requested_urls == [
            "https://assets.example/start.usd",
            "https://assets.example/final.usda",
        ]
        assert validated_urls == requested_urls

    @pytest.mark.parametrize(
        ("headers", "message"),
        [
            ({}, "missing Location"),
            ({"Location": "file:///etc/passwd"}, "Unsupported redirect URL scheme"),
        ],
    )
    def test_http_redirects_reject_unsafe_targets(
        self,
        headers: dict[str, str],
        message: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        response = _DummyResponse(302, headers=headers)

        monkeypatch.setattr(
            "service.renderer._safe_requests_get",
            lambda *args, **kwargs: response,
        )
        monkeypatch.setattr("service.renderer._validate_url_target", lambda url: None)

        dest = tmp_path / "scene.usd"
        with pytest.raises(ValueError, match=message):
            _fetch_usd("https://assets.example/start.usd", str(dest))

        assert response.closed is True
        assert not dest.exists()


class TestExtractZipBundle:
    def test_picks_stage_usda_root_from_client_bundle(self, tmp_path: Path):
        """Matches the layout produced by render_remote._bundle_stage_with_local_assets."""
        zip_path = _make_bundle(
            tmp_path,
            ["stage.usda", "mdl_materials/wood/wood.mdl", "textures/albedo.png"],
        )
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).name == "stage.usda"
        assert Path(main_usd).exists()
        # Assets must be extracted alongside so relative paths resolve.
        assert (extracted / "bundle" / "mdl_materials" / "wood" / "wood.mdl").exists()
        assert (extracted / "bundle" / "textures" / "albedo.png").exists()

    def test_renderer_passes_extracted_bundle_base_dir_to_backend(
        self,
        tmp_path: Path,
    ):
        zip_path = _make_bundle(
            tmp_path,
            ["stage.usda", "textures/albedo.png"],
        )
        backend = _RecordingBackend()
        renderer = Renderer.__new__(Renderer)
        renderer._backend = backend
        renderer._initialized = False
        renderer._render_lock = threading.RLock()
        renderer._recovery_cooldown_until = 0.0
        payload = base64.b64encode(zip_path.read_bytes()).decode("ascii")

        result = renderer.render(
            f"data:application/zip;base64,{payload}",
            camera_paths=["/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )

        assert result["status"] == "success"
        assert backend.render_kwargs is not None
        assert backend.base_dir_name == "bundle"
        assert backend.stage_exists_during_render is True
        assert backend.texture_exists_during_render is True

    def test_prefers_main_over_scene_and_stage(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["main.usda", "scene.usd", "stage.usdc"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).name == "main.usda"

    def test_prefers_scene_over_stage(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["scene.usd", "stage.usdc"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).name == "scene.usd"

    def test_falls_back_to_alphabetical(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["zebra.usda", "alpha.usda"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).name == "alpha.usda"

    def test_usdz_mode_prefers_first_usd_in_archive_order(self, tmp_path: Path):
        """USDZ packages use their first USD layer as the package root."""
        zip_path = _make_bundle(tmp_path, ["zebra.usda", "alpha.usda"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(
            str(zip_path),
            str(extracted),
            prefer_first_usd=True,
        )

        assert Path(main_usd).name == "zebra.usda"

    def test_discovers_nested_usd(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["assets/sub/main.usda"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        assert Path(main_usd).relative_to(extracted / "bundle") == Path(
            "assets/sub/main.usda"
        )

    def test_discovers_uppercase_usd_extensions(self, tmp_path: Path):
        """USD files with uppercase extensions must be found on Linux too."""
        zip_path = _make_bundle(tmp_path, ["MAIN.USDA", "Scene.USD"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        main_usd = _extract_zip_bundle(str(zip_path), str(extracted))

        # Priority still picks "main" via stem.lower(), even with uppercase ext.
        assert Path(main_usd).name == "MAIN.USDA"

    def test_empty_bundle_raises(self, tmp_path: Path):
        zip_path = _make_bundle(tmp_path, ["README.md", "notes/info.txt"])
        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="No USD layer found"):
            _extract_zip_bundle(str(zip_path), str(extracted))

    def test_rejects_path_traversal_entries(self, tmp_path: Path):
        """A malicious bundle must not be able to write outside extract_dir."""
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            zf.writestr("../escape.usda", '#usda 1.0\ndef Xform "Bad" {}\n')

        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="unsafe entry path"):
            _extract_zip_bundle(str(zip_path), str(extracted))

        # And the escape target must not have been written.
        assert not (tmp_path / "escape.usda").exists()

    def test_rejects_zip_bomb_by_entry_count(self, tmp_path: Path):
        """Too many entries trips the ZIP-bomb guard before extraction."""
        zip_path = tmp_path / "bomb.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            # Writing _ZIP_MAX_FILES+1 real entries is slow; patch the
            # central directory by emitting tiny entries beyond the limit.
            for i in range(_ZIP_MAX_FILES):
                zf.writestr(f"pad_{i}.bin", b"")

        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="too many entries"):
            _extract_zip_bundle(str(zip_path), str(extracted))

    def test_rejects_zip_bomb_by_uncompressed_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Oversized uncompressed total trips the guard before extraction."""
        zip_path = tmp_path / "bomb.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')

        # Drive the threshold down to something we can exceed trivially.
        monkeypatch.setattr("service.renderer._ZIP_MAX_UNCOMPRESSED_BYTES", 8)

        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="uncompressed size too large"):
            _extract_zip_bundle(str(zip_path), str(extracted))

    def test_rejects_invalid_zip_uncompressed_size_env(self):
        """The ZIP size override must be a positive integer byte count."""
        with pytest.raises(ValueError, match="must be greater than zero"):
            _parse_zip_max_uncompressed_bytes("0")

    def test_rejects_symlink_entries(self, tmp_path: Path):
        """Symlink entries in the ZIP must not be materialized on disk."""
        zip_path = tmp_path / "symlinks.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            # Symlink entry pointing at an absolute host path. Encode the
            # Unix symlink mode (0xA1FF = S_IFLNK|0o777) in external_attr,
            # mirroring what unzip / Info-ZIP writes.
            link = zipfile.ZipInfo("link.usda")
            link.create_system = 3  # Unix
            link.external_attr = (0xA1FF) << 16
            zf.writestr(link, "/etc/passwd")

        extracted = tmp_path / "work"
        extracted.mkdir()

        with pytest.raises(ValueError, match="symlink entry"):
            _extract_zip_bundle(str(zip_path), str(extracted))

        assert not (extracted / "bundle" / "link.usda").exists()


class TestIsUsdzPayload:
    """Detection decides whether a ZIP is a .usdz package or a bundle to extract."""

    def _write_usdz_shape(self, zip_path: Path) -> None:
        """Write an archive matching the Pixar USDZ structural spec."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("root.usdc", b"PXR-USDC\x00\x00\x00\x00")
            zf.writestr("textures/albedo.png", b"\x89PNG\r\n\x1a\n")

    def _write_deflated_bundle(self, zip_path: Path) -> None:
        """Write a render_nvcf-style bundle (DEFLATED)."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')
            zf.writestr("textures/albedo.png", b"\x89PNG\r\n\x1a\n")

    def test_url_extension_wins_for_s3(self, tmp_path: Path):
        zip_path = tmp_path / "asset.usdz"
        self._write_usdz_shape(zip_path)

        assert _is_usdz_payload("s3://bucket/asset.usdz", str(zip_path)) is True
        assert _is_usdz_payload("https://host/path/asset.USDZ", str(zip_path)) is True

    def test_url_with_query_string(self, tmp_path: Path):
        zip_path = tmp_path / "asset.usdz"
        self._write_usdz_shape(zip_path)

        # Query parameters must not hide the .usdz suffix.
        assert (
            _is_usdz_payload(
                "https://host/asset.usdz?version=42&signed=yes", str(zip_path)
            )
            is True
        )

    def test_non_usdz_url_is_not_package(self, tmp_path: Path):
        """render_nvcf bundles upload as .zip; those must go through extraction."""
        zip_path = tmp_path / "bundle.zip"
        self._write_deflated_bundle(zip_path)

        assert _is_usdz_payload("https://bucket/bundle.zip", str(zip_path)) is False

    def test_url_wins_over_content(self, tmp_path: Path):
        """A .zip URL never takes the USDZ path even if contents look USDZ-shaped."""
        zip_path = tmp_path / "looks_like_usdz.zip"
        self._write_usdz_shape(zip_path)

        assert (
            _is_usdz_payload("https://host/looks_like_usdz.zip", str(zip_path)) is False
        )

    def test_data_uri_usdz_by_content(self, tmp_path: Path):
        """Data URIs have no path; fall back to structural signature."""
        zip_path = tmp_path / "payload.bin"
        self._write_usdz_shape(zip_path)

        assert _is_usdz_payload("data:application/zip;base64,", str(zip_path)) is True

    def test_data_uri_deflated_is_not_usdz(self, tmp_path: Path):
        """A DEFLATED client bundle delivered via data URI still extracts."""
        zip_path = tmp_path / "payload.bin"
        self._write_deflated_bundle(zip_path)

        assert _is_usdz_payload("data:application/zip;base64,", str(zip_path)) is False

    def test_data_uri_stored_but_non_usd_first(self, tmp_path: Path):
        """STORED alone is not enough — first entry must be a USD layer."""
        zip_path = tmp_path / "payload.bin"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("README.md", b"not usd")
            zf.writestr("root.usdc", b"PXR-USDC\x00\x00\x00\x00")

        assert _is_usdz_payload("data:application/zip;base64,", str(zip_path)) is False

    def test_data_uri_empty_zip(self, tmp_path: Path):
        zip_path = tmp_path / "payload.bin"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED):
            pass

        assert _is_usdz_payload("data:application/zip;base64,", str(zip_path)) is False
