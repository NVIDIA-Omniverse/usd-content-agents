# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for S3 utility functions."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import NoCredentialsError, ProfileNotFound


class TestCreateS3Client:
    """Tests for _create_s3_client helper."""

    def test_with_valid_profile(self) -> None:
        """Test client creation with a valid AWS profile."""
        from world_understanding.utils.s3_utils import _create_s3_client

        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        with patch("boto3.Session", return_value=mock_session) as mock_ctor:
            client = _create_s3_client("my-profile")
            mock_ctor.assert_called_once_with(profile_name="my-profile")
            assert client == mock_client

    def test_with_no_profile(self) -> None:
        """Test client creation without a profile uses default credentials."""
        from world_understanding.utils.s3_utils import _create_s3_client

        mock_client = MagicMock()

        with patch("boto3.client", return_value=mock_client) as mock_ctor:
            client = _create_s3_client(None)
            mock_ctor.assert_called_once_with("s3")
            assert client == mock_client

    def test_fallback_when_profile_not_found(self) -> None:
        """Test that missing profile falls back to default credentials."""
        from world_understanding.utils.s3_utils import _create_s3_client

        mock_client = MagicMock()

        with (
            patch(
                "boto3.Session",
                side_effect=ProfileNotFound(profile="missing-profile"),
            ),
            patch("boto3.client", return_value=mock_client) as mock_default,
        ):
            client = _create_s3_client("missing-profile")
            mock_default.assert_called_once_with("s3")
            assert client == mock_client

    def test_raises_when_no_credentials_at_all(self) -> None:
        """Test error when profile missing AND no default credentials."""
        from world_understanding.utils.s3_utils import _create_s3_client

        with (
            patch(
                "boto3.Session",
                side_effect=ProfileNotFound(profile="missing"),
            ),
            patch(
                "boto3.client",
                side_effect=NoCredentialsError(),
            ),
            pytest.raises(ValueError, match="No AWS credentials available"),
        ):
            _create_s3_client("missing")

    def test_raises_when_no_default_credentials(self) -> None:
        """Test error when no profile given and no default credentials."""
        from world_understanding.utils.s3_utils import _create_s3_client

        with (
            patch(
                "boto3.client",
                side_effect=NoCredentialsError(),
            ),
            pytest.raises(ValueError, match="No AWS credentials available"),
        ):
            _create_s3_client(None)


class TestUploadFileToS3Preconditions:
    """Tests for upload_file_to_s3 input validation (no live S3 calls)."""

    def test_raises_value_error_when_bucket_is_empty(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """An empty bucket in the S3 URI should raise ValueError before reaching boto3.

        Regression: `scene_optimizer_nvcf.py` interpolates `s3://{WU_S3_BUCKET}/{key}`
        even when WU_S3_BUCKET is empty, which used to surface as an opaque
        boto3 bucket-name regex error. It should surface as a descriptive
        ValueError instead.
        """
        from world_understanding.utils.s3_utils import upload_file_to_s3

        src = tmp_path / "payload.txt"
        src.write_text("hello")

        with pytest.raises(ValueError, match="S3 bucket is required"):
            upload_file_to_s3(str(src), "s3:///path/to/file.txt")


class TestS3BucketAuthorization:
    """Tests for the client-controlled S3 trust-boundary guard."""

    @pytest.mark.parametrize("allowed_buckets", [None, "", " , \t\n", []])
    def test_fails_closed_without_an_allowlist(
        self,
        allowed_buckets: str | list[str] | None,
    ) -> None:
        from world_understanding.utils.s3_utils import (
            S3BucketNotAllowedError,
            assert_s3_bucket_allowed,
        )

        with pytest.raises(
            S3BucketNotAllowedError,
            match="not permitted by the service's configured bucket allowlist",
        ):
            assert_s3_bucket_allowed(
                "s3://intake-bucket/path/scene.usdz", allowed_buckets
            )

    def test_rejects_foreign_bucket_with_the_same_stable_detail(self) -> None:
        from world_understanding.utils.s3_utils import (
            S3BucketNotAllowedError,
            assert_s3_bucket_allowed,
        )

        with pytest.raises(S3BucketNotAllowedError) as empty_policy:
            assert_s3_bucket_allowed("s3://foreign/private.usdz", "")
        with pytest.raises(S3BucketNotAllowedError) as foreign_bucket:
            assert_s3_bucket_allowed(
                "s3://foreign/private.usdz", "approved-a, approved-b"
            )

        assert str(foreign_bucket.value) == str(empty_policy.value)
        assert "foreign" not in str(foreign_bucket.value)
        assert "private.usdz" not in str(foreign_bucket.value)

    def test_accepts_only_exact_normalized_bucket_names(self) -> None:
        from world_understanding.utils.s3_utils import (
            S3BucketNotAllowedError,
            assert_s3_bucket_allowed,
        )

        assert (
            assert_s3_bucket_allowed(
                "s3://approved-b/path/scene.usdz",
                " approved-a,\napproved-b\tapproved-c ",
            )
            == "approved-b"
        )
        assert (
            assert_s3_bucket_allowed(
                "s3://approved-c/path/scene.usdz",
                ["approved-a", " approved-c "],
            )
            == "approved-c"
        )

        with pytest.raises(S3BucketNotAllowedError):
            assert_s3_bucket_allowed("s3://approved/path/scene.usdz", "approved-prefix")

    def test_extension_guard_authorizes_and_normalizes_suffix(self) -> None:
        from world_understanding.utils.s3_utils import (
            authorize_s3_uri_for_extensions,
        )

        assert (
            authorize_s3_uri_for_extensions(
                "s3://approved/path/SCENE.USDZ",
                "approved",
                allowed_extensions={"usd", ".usdz"},
            )
            == ".usdz"
        )

    def test_extension_guard_preserves_invalid_file_type_before_policy(self) -> None:
        from world_understanding.utils.s3_utils import (
            authorize_s3_uri_for_extensions,
        )

        with pytest.raises(ValueError, match="Invalid USD file type"):
            authorize_s3_uri_for_extensions(
                "s3://foreign/path/scene.txt",
                "approved",
                allowed_extensions={".usd", ".usdz"},
            )

    @pytest.mark.parametrize(
        ("s3_uri", "expected_error"),
        [
            ("https://approved/path/scene.usd", "Invalid S3 URI format"),
            ("s3://approved/", "Invalid USD file type"),
            ("s3://approved/path/scene.txt", "Invalid USD file type"),
        ],
    )
    def test_extension_guard_rejects_invalid_inputs(
        self,
        s3_uri: str,
        expected_error: str,
    ) -> None:
        from world_understanding.utils.s3_utils import (
            authorize_s3_uri_for_extensions,
        )

        with pytest.raises(ValueError, match=expected_error):
            authorize_s3_uri_for_extensions(
                s3_uri,
                "approved",
                allowed_extensions={".usd", ".usdz"},
            )

    @pytest.mark.parametrize(
        "s3_uri",
        [
            "s3://approved",
            "s3://approved/",
            "s3:///path/scene.usdz",
            "approved/path/scene.usdz",
        ],
    )
    def test_malformed_uri_is_rejected_before_authorization(self, s3_uri: str) -> None:
        from world_understanding.utils.s3_utils import assert_s3_bucket_allowed

        with pytest.raises(ValueError, match="Invalid S3 URI format"):
            assert_s3_bucket_allowed(s3_uri, "approved")
