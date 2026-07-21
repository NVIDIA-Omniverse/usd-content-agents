# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S3 utility functions for file upload and management."""

import logging
import mimetypes
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from world_understanding.telemetry.attributes import MAAttributes

logger = logging.getLogger(__name__)

# Get tracer at module level
_tracer = trace.get_tracer(__name__)

_S3_LOCATION_NOT_ALLOWED_MESSAGE = (
    "S3 URI is not permitted by the service's configured bucket allowlist"
)


def _create_s3_client(profile_name: str | None = None) -> Any:
    """Create an S3 client, falling back to default credentials if profile not found.

    Args:
        profile_name: AWS profile name. If the profile is not found,
            falls back to default credentials (env vars, instance role, etc.).

    Returns:
        A boto3 S3 client.

    Raises:
        ValueError: If no AWS credentials are available at all.
    """
    try:
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
            s3_client = session.client("s3")
            logger.info("Using AWS profile: %s", profile_name)
            return s3_client
    except ProfileNotFound:
        logger.warning(
            "AWS profile '%s' not found, falling back to default credentials",
            profile_name,
        )

    try:
        s3_client = boto3.client("s3")
        logger.info("Using default AWS credentials")
        return s3_client
    except NoCredentialsError as e:
        raise ValueError("No AWS credentials available") from e


class S3BucketNotAllowedError(Exception):
    """Raised when a client-supplied S3 URI is outside an explicit allowlist."""


def _normalize_bucket_names(
    allowed_buckets: str | Iterable[str] | None,
) -> set[str]:
    """Normalize a delimited string or iterable of bucket names for exact matching."""
    if allowed_buckets is None:
        return set()
    if isinstance(allowed_buckets, str):
        items: Iterable[str] = re.split(r"[,\s]+", allowed_buckets)
    else:
        items = allowed_buckets
    return {item.strip() for item in items if isinstance(item, str) and item.strip()}


def assert_s3_bucket_allowed(
    s3_uri: str,
    allowed_buckets: str | Iterable[str] | None,
) -> str:
    """Authorize a client-supplied S3 URI before any service-owned S3 request.

    The policy intentionally fails closed: an empty or unset allowlist rejects
    every bucket. Entries are exact bucket names, not prefixes or URI patterns.
    Services should expose the same generic rejection detail for both an empty
    policy and a foreign bucket so the authorization check cannot disclose
    whether an object exists.

    Args:
        s3_uri: Client-supplied URI in ``s3://bucket/key`` form.
        allowed_buckets: Bucket names as a comma/whitespace-separated string or
            an iterable of strings.

    Returns:
        The authorized bucket name.

    Raises:
        ValueError: If ``s3_uri`` is malformed.
        S3BucketNotAllowedError: If the bucket is not explicitly allowed.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI format; expected s3://bucket/key")
    bucket, key = _parse_s3_path(s3_uri)
    if not bucket or not key:
        raise ValueError("Invalid S3 URI format; expected s3://bucket/key")
    if bucket not in _normalize_bucket_names(allowed_buckets):
        raise S3BucketNotAllowedError(_S3_LOCATION_NOT_ALLOWED_MESSAGE)
    return bucket


def authorize_s3_uri_for_extensions(
    s3_uri: str,
    allowed_buckets: str | Iterable[str] | None,
    *,
    allowed_extensions: Iterable[str],
) -> str:
    """Authorize one client S3 URI and validate its object-key extension.

    URI shape and file type are validated before bucket authorization to retain
    the services' existing HTTP 400 contract for malformed inputs. Bucket
    authorization still completes before the caller performs any service-owned
    S3 or session-store operation. The caller must translate these domain
    exceptions into its transport contract.

    Args:
        s3_uri: Client-supplied URI in ``s3://bucket/key`` form.
        allowed_buckets: Exact bucket names accepted by the service.
        allowed_extensions: Accepted object suffixes, with or without a leading
            dot. Matching is case-insensitive and the returned suffix is lower-case.

    Returns:
        The normalized object-key extension, including its leading dot.

    Raises:
        ValueError: If the URI is malformed or its extension is not allowed.
        S3BucketNotAllowedError: If the bucket is not explicitly allowed.
    """
    if not s3_uri.startswith("s3://") or s3_uri.count("/") < 3:
        raise ValueError("Invalid S3 URI format; expected s3://bucket/key")
    filename = s3_uri.rstrip("/").rsplit("/", 1)[-1]
    extension = Path(filename).suffix.lower()
    normalized_extensions: set[str] = set()
    for item in allowed_extensions:
        value = item.strip().lower()
        if value:
            normalized_extensions.add(value if value.startswith(".") else f".{value}")
    if extension not in normalized_extensions:
        allowed = ", ".join(sorted(normalized_extensions))
        raise ValueError(
            f"Invalid USD file type in S3 URI: {extension}. Allowed: {allowed}"
        )
    assert_s3_bucket_allowed(s3_uri, allowed_buckets)
    return extension


def list_s3_folder(
    s3_folder_path: str,
    profile_name: str | None = None,
    bucket_name: str | None = None,
) -> list[str]:
    """
    List all files in an S3 folder/prefix.

    Args:
        s3_folder_path: S3 folder path. Can be:
            - Full S3 URI (s3://bucket/folder/)
            - Bucket and folder path (bucket/folder/)
            - Just the folder path if bucket_name is provided
        profile_name: AWS profile name to use for authentication.
            If None, uses default credentials
        bucket_name: Optional bucket name if not included in s3_folder_path

    Returns:
        List of full S3 URIs for all files in the folder

    Raises:
        ProfileNotFound: If the specified AWS profile doesn't exist
        NoCredentialsError: If no AWS credentials are available
        ClientError: If S3 listing fails

    Examples:
        >>> list_s3_folder("s3://my-bucket/materials/aluminum/")
        ['s3://my-bucket/materials/aluminum/texture1.png',
         's3://my-bucket/materials/aluminum/texture2.png']
    """
    # Parse S3 path
    bucket, prefix = _parse_s3_path(s3_folder_path, bucket_name)

    # Ensure prefix ends with / for folder listing
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    with _tracer.start_as_current_span("s3.list") as span:
        span.set_attribute(MAAttributes.S3_BUCKET, bucket)
        span.set_attribute(MAAttributes.S3_KEY, prefix)
        span.set_attribute(MAAttributes.S3_OPERATION, "list")

        # Create S3 client with specified profile
        try:
            s3_client = _create_s3_client(profile_name)
        except ValueError as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        # List objects in the folder
        file_uris = []
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Skip the folder itself (keys ending with /)
                        if not key.endswith("/"):
                            file_uris.append(f"s3://{bucket}/{key}")

            logger.debug(f"Found {len(file_uris)} files in s3://{bucket}/{prefix}")

        except ClientError as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchBucket":
                raise ClientError(
                    {
                        "Error": {
                            "Code": error_code,
                            "Message": f"Bucket '{bucket}' does not exist",
                        }
                    },
                    "list_objects_v2",
                ) from e
            raise

        return file_uris


def upload_file_to_s3(
    file_path: str | Path,
    s3_path: str,
    profile_name: str | None = None,
    bucket_name: str | None = None,
    extra_args: dict[str, Any] | None = None,
    callback: Any | None = None,
) -> str:
    """
    Upload a file to an S3 bucket at a specified path.

    Args:
        file_path: Local path to the file to upload
        s3_path: S3 path where the file should be uploaded. Can be:
            - Full S3 URI (s3://bucket/key/path)
            - Bucket and key path (bucket/key/path) if bucket_name is not provided
            - Just the key path if bucket_name is provided
        profile_name: AWS profile name to use for authentication.
            If None, uses default credentials
        bucket_name: Optional bucket name if not included in s3_path
        extra_args: Extra arguments for upload
            (e.g., {'ACL': 'public-read', 'ContentType': 'text/html'})
        callback: Optional callback for upload progress

    Returns:
        The S3 URI of the uploaded file

    Raises:
        FileNotFoundError: If the local file doesn't exist
        ValueError: If S3 path format is invalid
        ProfileNotFound: If the specified AWS profile doesn't exist
        NoCredentialsError: If no AWS credentials are available
        ClientError: If S3 upload fails

    Examples:
        # Upload with full S3 URI
        >>> upload_file_to_s3(
        ...     "local.txt",
        ...     "s3://my-bucket/path/to/file.txt",
        ...     profile_name="dev"
        ... )
        's3://my-bucket/path/to/file.txt'

        # Upload with bucket and key
        >>> upload_file_to_s3("local.txt", "my-bucket/path/to/file.txt")
        's3://my-bucket/path/to/file.txt'

        # Upload with separate bucket name
        >>> upload_file_to_s3(
        ...     "local.txt",
        ...     "path/to/file.txt",
        ...     bucket_name="my-bucket"
        ... )
        's3://my-bucket/path/to/file.txt'

        # Upload with extra arguments
        >>> upload_file_to_s3(
        ...     "index.html",
        ...     "s3://website-bucket/index.html",
        ...     extra_args={'ACL': 'public-read', 'ContentType': 'text/html'}
        ... )
        's3://website-bucket/index.html'
    """
    # Convert to Path object for easier handling
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Parse S3 path
    bucket, key = _parse_s3_path(s3_path, bucket_name)

    if not bucket:
        raise ValueError(
            "S3 bucket is required for upload but is empty. "
            "Set WU_S3_BUCKET or pass bucket_name, or switch to data-URI "
            "mode by setting MA_RENDERING_USE_DATA_URI=true."
        )

    with _tracer.start_as_current_span("s3.upload") as span:
        span.set_attribute(MAAttributes.S3_BUCKET, bucket)
        span.set_attribute(MAAttributes.S3_KEY, key)
        span.set_attribute(MAAttributes.S3_OPERATION, "upload")

        # Create S3 client with specified profile
        try:
            s3_client = _create_s3_client(profile_name)
        except ValueError as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        # Prepare extra arguments
        if extra_args is None:
            extra_args = {}

        # Auto-detect content type if not specified
        if "ContentType" not in extra_args:
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type:
                extra_args["ContentType"] = content_type

        # Upload the file
        try:
            logger.info("Uploading %s to s3://%s/%s", file_path, bucket, key)
            s3_client.upload_file(
                str(file_path),
                bucket,
                key,
                ExtraArgs=extra_args,
                Callback=callback,
            )
            s3_uri = f"s3://{bucket}/{key}"
            logger.info("Successfully uploaded to %s", s3_uri)
            return s3_uri

        except ClientError as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise ValueError(f"Bucket '{bucket}' does not exist") from e
            elif error_code == "AccessDenied":
                raise PermissionError(f"Access denied to bucket '{bucket}'") from e
            else:
                raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise RuntimeError(f"Unexpected error during upload: {e}") from e


def upload_directory_to_s3(
    directory_path: str | Path,
    s3_prefix: str,
    profile_name: str | None = None,
    bucket_name: str | None = None,
    extra_args: dict[str, Any] | None = None,
    recursive: bool = True,
    file_pattern: str | None = None,
) -> list[str]:
    """
    Upload all files from a directory to S3.

    Args:
        directory_path: Local directory path
        s3_prefix: S3 prefix for uploaded files
        profile_name: AWS profile name to use
        bucket_name: Optional bucket name if not included in s3_prefix
        extra_args: Extra arguments for upload
        recursive: Whether to upload subdirectories recursively
        file_pattern: Optional glob pattern to filter files (e.g., "*.txt")

    Returns:
        List of S3 URIs for uploaded files

    Examples:
        # Upload entire directory
        >>> upload_directory_to_s3(
        ...     "./data",
        ...     "s3://my-bucket/data/",
        ...     profile_name="dev"
        ... )
        ['s3://my-bucket/data/file1.txt', 's3://my-bucket/data/file2.txt']

        # Upload only specific files
        >>> upload_directory_to_s3(
        ...     "./images",
        ...     "s3://my-bucket/images/",
        ...     file_pattern="*.png"
        ... )
        ['s3://my-bucket/images/img1.png', 's3://my-bucket/images/img2.png']
    """
    directory_path = Path(directory_path)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Parse S3 prefix
    bucket, prefix = _parse_s3_path(s3_prefix, bucket_name)

    # Ensure prefix ends with /
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    uploaded_files = []

    # Get files to upload
    if recursive:
        if file_pattern:
            files = list(directory_path.rglob(file_pattern))
        else:
            files = [f for f in directory_path.rglob("*") if f.is_file()]
    else:
        if file_pattern:
            files = list(directory_path.glob(file_pattern))
        else:
            files = [f for f in directory_path.iterdir() if f.is_file()]

    # Upload each file
    for file_path in files:
        # Calculate relative path for S3 key
        relative_path = file_path.relative_to(directory_path)
        s3_key = prefix + str(relative_path).replace(os.sep, "/")
        s3_full_path = f"s3://{bucket}/{s3_key}"

        try:
            upload_file_to_s3(
                file_path,
                s3_full_path,
                profile_name=profile_name,
                extra_args=extra_args,
            )
            uploaded_files.append(s3_full_path)
        except Exception as e:
            logger.error("Failed to upload %s: %s", file_path, e)
            # Continue with other files

    logger.info("Uploaded %s files to S3", len(uploaded_files))
    return uploaded_files


def _parse_s3_path(s3_path: str, bucket_name: str | None = None) -> tuple[str, str]:
    """
    Parse S3 path into bucket and key components.

    Args:
        s3_path: S3 path in various formats
        bucket_name: Optional bucket name to use if not in path

    Returns:
        Tuple of (bucket, key)

    Raises:
        ValueError: If path format is invalid
    """
    s3_path = s3_path.strip()

    # Handle s3:// URI format
    if s3_path.startswith("s3://"):
        path = s3_path[5:]  # Remove s3:// prefix
        parts = path.split("/", 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid S3 URI format: {s3_path}")
        return parts[0], parts[1]

    # Handle bucket/key format
    if "/" in s3_path and not bucket_name:
        parts = s3_path.split("/", 1)
        return parts[0], parts[1]

    # Handle key-only format with separate bucket_name
    if bucket_name:
        return bucket_name, s3_path

    raise ValueError(
        f"Invalid S3 path format: {s3_path}. "
        "Use 's3://bucket/key', 'bucket/key', or provide "
        "bucket_name separately"
    )


def check_s3_file_exists(
    s3_path: str,
    profile_name: str | None = None,
    bucket_name: str | None = None,
) -> bool:
    """
    Check if a file exists in S3.

    Args:
        s3_path: S3 path to check
        profile_name: AWS profile name to use
        bucket_name: Optional bucket name if not included in s3_path

    Returns:
        True if file exists, False otherwise

    Examples:
        >>> check_s3_file_exists("s3://my-bucket/file.txt", profile_name="dev")
        True
    """
    bucket, key = _parse_s3_path(s3_path, bucket_name)

    try:
        s3_client = _create_s3_client(profile_name)
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            logger.error("Error checking S3 file: %s", e)
            return False
    except Exception as e:
        logger.error("Unexpected error checking S3 file: %s", e)
        return False


def download_file_from_s3(
    s3_path: str,
    local_path: str | Path,
    profile_name: str | None = None,
    bucket_name: str | None = None,
    callback: Any | None = None,
) -> str:
    """
    Download a file from S3 to a local path.

    Args:
        s3_path: S3 path to download from. Can be:
            - Full S3 URI (s3://bucket/key/path)
            - Bucket and key path (bucket/key/path) if bucket_name is not provided
            - Just the key path if bucket_name is provided
        local_path: Local path where the file should be downloaded
        profile_name: AWS profile name to use for authentication.
            If None, uses default credentials
        bucket_name: Optional bucket name if not included in s3_path
        callback: Optional callback for download progress

    Returns:
        The local file path as a string

    Raises:
        ValueError: If S3 path format is invalid
        ProfileNotFound: If the specified AWS profile doesn't exist
        NoCredentialsError: If no AWS credentials are available
        ClientError: If S3 download fails

    Examples:
        # Download with full S3 URI
        >>> download_file_from_s3(
        ...     "s3://my-bucket/path/to/file.txt",
        ...     "/local/path/file.txt",
        ...     profile_name="dev"
        ... )
        '/local/path/file.txt'
    """
    # Convert to Path object for easier handling
    local_path = Path(local_path)

    # Create parent directories if they don't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse S3 path
    bucket, key = _parse_s3_path(s3_path, bucket_name)

    with _tracer.start_as_current_span("s3.download") as span:
        span.set_attribute(MAAttributes.S3_BUCKET, bucket)
        span.set_attribute(MAAttributes.S3_KEY, key)
        span.set_attribute(MAAttributes.S3_OPERATION, "download")

        # Create S3 client with specified profile
        try:
            s3_client = _create_s3_client(profile_name)
        except ValueError as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        # Download the file
        try:
            logger.info("Downloading s3://%s/%s to %s", bucket, key, local_path)
            s3_client.download_file(
                bucket,
                key,
                str(local_path),
                Callback=callback,
            )
            logger.info("Successfully downloaded to %s", local_path)
            return str(local_path)

        except ClientError as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise ValueError(f"Bucket '{bucket}' does not exist") from e
            elif error_code == "NoSuchKey" or error_code == "404":
                raise FileNotFoundError(
                    f"S3 object 's3://{bucket}/{key}' does not exist"
                ) from e
            elif error_code == "AccessDenied":
                raise PermissionError(f"Access denied to bucket '{bucket}'") from e
            else:
                raise RuntimeError(f"Failed to download file: {e}") from e
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise RuntimeError(f"Unexpected error during download: {e}") from e


def delete_s3_path(
    s3_path: str,
    profile_name: str | None = None,
    bucket_name: str | None = None,
    recursive: bool = False,
    max_keys: int = 1000,
) -> dict[str, int] | bool:
    """
    Delete an S3 object or folder (prefix) and optionally all its contents.

    This unified function can delete either:
    1. A single S3 object (file)
    2. A folder/prefix and all objects within it (when recursive=True)

    Args:
        s3_path: S3 path to delete. Can be:
            - Full S3 URI (s3://bucket/path/to/file.txt or s3://bucket/folder/)
            - Bucket and path (bucket/path/to/file.txt) if bucket_name is not provided
            - Just the path if bucket_name is provided
        profile_name: AWS profile name to use for authentication.
            If None, uses default credentials
        bucket_name: Optional bucket name if not included in s3_path
        recursive: If True, delete all objects with this prefix (folder behavior).
            If False, delete only the exact object (file behavior). Default: False
        max_keys: Maximum number of objects to delete per batch when recursive=True (default 1000)

    Returns:
        - If recursive=False: bool (True if deleted, False if didn't exist)
        - If recursive=True: dict with 'deleted' and 'failed' counts

    Raises:
        ValueError: If S3 path format is invalid
        ProfileNotFound: If the specified AWS profile doesn't exist
        NoCredentialsError: If no AWS credentials are available
        RuntimeError: If an unexpected error occurs

    Examples:
        # Delete a single file
        >>> delete_s3_path("s3://my-bucket/path/to/file.txt")
        True

        # Delete a folder and all its contents
        >>> delete_s3_path("s3://my-bucket/temp/uploads/", recursive=True)
        {'deleted': 15, 'failed': 0}

        # Delete with custom profile
        >>> delete_s3_path(
        ...     "s3://my-bucket/data/",
        ...     profile_name="production",
        ...     recursive=True
        ... )
        {'deleted': 42, 'failed': 0}
    """
    # Parse S3 path
    bucket, key = _parse_s3_path(s3_path, bucket_name)

    with _tracer.start_as_current_span("s3.delete") as span:
        span.set_attribute(MAAttributes.S3_BUCKET, bucket)
        span.set_attribute(MAAttributes.S3_KEY, key)
        span.set_attribute(
            MAAttributes.S3_OPERATION, "delete_recursive" if recursive else "delete"
        )

        # Create S3 client with specified profile
        try:
            s3_client = _create_s3_client(profile_name)
        except ValueError as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

        if not recursive:
            # Single object deletion
            try:
                logger.info("Deleting s3://%s/%s", bucket, key)
                s3_client.delete_object(Bucket=bucket, Key=key)
                logger.info("Successfully deleted s3://%s/%s", bucket, key)
                return True

            except ClientError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                error_code = e.response["Error"]["Code"]
                if error_code == "NoSuchBucket":
                    raise ValueError(f"Bucket '{bucket}' does not exist") from e
                elif error_code == "NoSuchKey":
                    logger.info("Object s3://%s/%s does not exist", bucket, key)
                    return False
                elif error_code == "AccessDenied":
                    raise PermissionError(
                        f"Access denied to delete from bucket '{bucket}'"
                    ) from e
                else:
                    raise RuntimeError(f"Failed to delete object: {e}") from e
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise RuntimeError(f"Unexpected error during delete: {e}") from e

        else:
            # Recursive deletion (folder behavior)
            prefix = key
            # Ensure prefix ends with / for folder-like behavior unless it's empty
            if prefix and not prefix.endswith("/"):
                prefix += "/"

            deleted_count = 0
            failed_count = 0

            try:
                logger.info(
                    "Recursively deleting s3://%s/%s and all its contents",
                    bucket,
                    prefix,
                )

                # Use paginator to handle large numbers of objects
                paginator = s3_client.get_paginator("list_objects_v2")
                page_iterator = paginator.paginate(
                    Bucket=bucket,
                    Prefix=prefix,
                    PaginationConfig={"PageSize": max_keys},
                )

                for page in page_iterator:
                    if "Contents" not in page:
                        continue

                    # Prepare objects for batch deletion
                    objects_to_delete = []
                    for obj in page["Contents"]:
                        objects_to_delete.append({"Key": obj["Key"]})

                    if not objects_to_delete:
                        continue

                    # Delete objects in batch
                    try:
                        response = s3_client.delete_objects(
                            Bucket=bucket, Delete={"Objects": objects_to_delete}
                        )

                        # Count successful deletions
                        if "Deleted" in response:
                            batch_deleted = len(response["Deleted"])
                            deleted_count += batch_deleted
                            logger.info(
                                "Deleted %s objects from s3://%s/%s",
                                batch_deleted,
                                bucket,
                                prefix,
                            )

                        # Count failed deletions
                        if "Errors" in response:
                            batch_failed = len(response["Errors"])
                            failed_count += batch_failed
                            for error in response["Errors"]:
                                logger.error(
                                    "Failed to delete %s: %s - %s",
                                    error["Key"],
                                    error["Code"],
                                    error["Message"],
                                )

                    except ClientError as e:
                        logger.error("Failed to delete batch: %s", e)
                        failed_count += len(objects_to_delete)

                if deleted_count > 0:
                    logger.info(
                        "Successfully deleted %s objects from s3://%s/%s",
                        deleted_count,
                        bucket,
                        prefix,
                    )
                else:
                    logger.info("No objects found in s3://%s/%s", bucket, prefix)

                if failed_count > 0:
                    logger.warning(
                        "Failed to delete %s objects from s3://%s/%s",
                        failed_count,
                        bucket,
                        prefix,
                    )

                return {"deleted": deleted_count, "failed": failed_count}

            except ClientError as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                error_code = e.response["Error"]["Code"]
                if error_code == "NoSuchBucket":
                    raise ValueError(f"Bucket '{bucket}' does not exist") from e
                elif error_code == "AccessDenied":
                    raise PermissionError(
                        f"Access denied to delete from bucket '{bucket}'"
                    ) from e
                else:
                    raise RuntimeError(f"Failed to delete folder: {e}") from e
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise RuntimeError(f"Unexpected error during folder delete: {e}") from e


def get_s3_file_url(
    s3_path: str,
    profile_name: str | None = None,
    bucket_name: str | None = None,
    expiration: int = 3600,
    use_public_url: bool = False,
) -> str:
    """
    Get a URL for an S3 file (either public or presigned).

    Args:
        s3_path: S3 path to the file
        profile_name: AWS profile name to use
        bucket_name: Optional bucket name if not included in s3_path
        expiration: Expiration time in seconds for presigned URL (default 1 hour)
        use_public_url: If True, return public URL instead of presigned

    Returns:
        URL to access the S3 file

    Examples:
        # Get presigned URL
        >>> get_s3_file_url(
        ...     "s3://my-bucket/file.txt",
        ...     profile_name="dev"
        ... )
        'https://my-bucket.s3.amazonaws.com/file.txt?...'

        # Get public URL
        >>> get_s3_file_url(
        ...     "s3://public-bucket/file.txt",
        ...     use_public_url=True
        ... )
        'https://public-bucket.s3.amazonaws.com/file.txt'
    """
    bucket, key = _parse_s3_path(s3_path, bucket_name)

    if use_public_url:
        # Return public URL format
        return f"https://{bucket}.s3.amazonaws.com/{key}"

    try:
        s3_client = _create_s3_client(profile_name)
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration,
        )
        return url
    except Exception as e:
        raise RuntimeError(f"Failed to generate S3 URL: {e}") from e
