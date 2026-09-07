# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for immutable opaque-artifact captures."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from world_understanding.utils import captured_artifacts
from world_understanding.utils.captured_artifacts import (
    BoundedBinaryReader,
    CapturedArtifactCleanupError,
    CapturedArtifactError,
    CapturedOpaqueArtifactResolver,
    CapturedOpaqueFile,
    OpaqueArtifactRequest,
    capture_local_opaque_file,
    capture_resolved_opaque_file,
    capture_streamed_opaque_file,
)


def _request(
    payload: bytes,
    *,
    uri: str = "artifact://tests/payload.bin",
    max_bytes: int | None = None,
) -> OpaqueArtifactRequest:
    return OpaqueArtifactRequest(
        uri=uri,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        max_bytes=len(payload) if max_bytes is None else max_bytes,
    )


def _leaf_exceptions(error: BaseException) -> list[BaseException]:
    pending = [error]
    leaves: list[BaseException] = []
    while pending:
        candidate = pending.pop()
        if isinstance(candidate, BaseExceptionGroup):
            pending.extend(candidate.exceptions)
        else:
            leaves.append(candidate)
    return leaves


def test_captured_opaque_file_is_runtime_final() -> None:
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type(
            "MaliciousCapturedOpaqueFile", (captured_artifacts.CapturedOpaqueFile,), {}
        )


def test_opaque_artifact_request_is_runtime_final() -> None:
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("MaliciousOpaqueArtifactRequest", (OpaqueArtifactRequest,), {})


def test_unsupported_platform_fails_before_resolver_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"payload"
    resolver_called = False

    class ExplosiveResolver:
        def capture(self, request: OpaqueArtifactRequest) -> Any:
            nonlocal resolver_called
            resolver_called = True
            raise AssertionError("resolver must not run")

    monkeypatch.setattr(captured_artifacts, "_CAPTURE_PLATFORM_SUPPORTED", False)

    with pytest.raises(CapturedArtifactError) as raised:
        with capture_resolved_opaque_file(ExplosiveResolver(), _request(payload)):
            raise AssertionError("capture must not be published")

    assert raised.value.code == "unsupported_platform"
    assert resolver_called is False


class _ComparisonForgingSha256(str):
    def __eq__(self, _other: object) -> bool:
        return True

    def __ne__(self, _other: object) -> bool:
        return False


class _ComparisonForgingBudget(int):
    def __lt__(self, _other: object) -> bool:
        return False

    def __le__(self, _other: object) -> bool:
        return True

    def __gt__(self, _other: object) -> bool:
        return False

    def __ge__(self, _other: object) -> bool:
        return True


class _UnderreportedBytes(bytes):
    def __len__(self) -> int:
        return 1


def test_sha256_string_subclass_cannot_forge_digest_match() -> None:
    payload = b"trusted digest payload"
    forged_sha256 = _ComparisonForgingSha256("0" * 64)

    with pytest.raises(TypeError, match="sha256"):
        OpaqueArtifactRequest(
            uri="artifact://tests/forged-sha.bin",
            sha256=forged_sha256,
            size_bytes=len(payload),
            max_bytes=len(payload),
        )

    with pytest.raises(TypeError, match="sha256"):
        captured_artifacts._capture_reader_snapshot(
            lambda size, offset: payload[offset : offset + size],
            uri="artifact://tests/forged-sha.bin",
            expected_sha256=forged_sha256,
            size_bytes=len(payload),
            max_bytes=len(payload),
            prefer_disk_snapshot=True,
            memfd_max_bytes=0,
            chunk_size=4,
            snapshot_factory=captured_artifacts.tempfile.TemporaryFile,
            source_stability_check=None,
        )


def test_integer_subclass_cannot_forge_max_bytes_budget() -> None:
    payload = b"x" * 4096
    forged_max_bytes = _ComparisonForgingBudget(1)

    with pytest.raises(TypeError, match="max_bytes"):
        OpaqueArtifactRequest(
            uri="artifact://tests/forged-budget.bin",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            max_bytes=forged_max_bytes,
        )

    with pytest.raises(TypeError, match="max_bytes"):
        captured_artifacts._capture_reader_snapshot(
            lambda size, offset: payload[offset : offset + size],
            uri="artifact://tests/forged-budget.bin",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            max_bytes=forged_max_bytes,
            prefer_disk_snapshot=True,
            memfd_max_bytes=0,
            chunk_size=4,
            snapshot_factory=captured_artifacts.tempfile.TemporaryFile,
            source_stability_check=None,
        )


def test_bytes_subclass_cannot_underreport_before_budget_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    underreported = _UnderreportedBytes(b"x" * 4096)
    writes: list[bytes] = []

    with monkeypatch.context() as write_patch:
        write_patch.setattr(
            captured_artifacts,
            "_write_all",
            lambda _descriptor, payload: writes.append(payload),
        )
        with pytest.raises(CapturedArtifactError) as raised:
            captured_artifacts._capture_reader_snapshot(
                lambda _size, _offset: underreported,
                uri="artifact://tests/underreported-bytes.bin",
                expected_sha256=None,
                size_bytes=1,
                max_bytes=1,
                prefer_disk_snapshot=True,
                memfd_max_bytes=0,
                chunk_size=1,
                snapshot_factory=captured_artifacts.tempfile.TemporaryFile,
                source_stability_check=None,
            )

    assert raised.value.code == "stream_protocol_invalid"
    assert writes == []


def _direct_capture(
    payload: bytes,
    request: OpaqueArtifactRequest,
) -> CapturedOpaqueFile:
    return captured_artifacts._capture_reader_snapshot(
        lambda size, offset: payload[offset : offset + size],
        uri=request.uri,
        expected_sha256=request.sha256,
        size_bytes=request.size_bytes,
        max_bytes=request.max_bytes,
        prefer_disk_snapshot=True,
        memfd_max_bytes=0,
        chunk_size=max(1, min(4, request.size_bytes)),
        snapshot_factory=captured_artifacts.tempfile.TemporaryFile,
        source_stability_check=None,
    )


class _ResolverContext:
    def __init__(
        self,
        captured: CapturedOpaqueFile,
        *,
        close_capture: bool,
        exit_error: BaseException | None = None,
        transfer_descriptor: bool = False,
        reuse_same_inode: bool = False,
    ) -> None:
        self.captured = captured
        self.close_capture = close_capture
        self.exit_error = exit_error
        self.transfer_descriptor = transfer_descriptor
        self.reuse_same_inode = reuse_same_inode
        self.transferred_descriptor = -1
        self.reused_descriptor = -1
        self.replacement_descriptor = -1

    def __enter__(self) -> CapturedOpaqueFile:
        return self.captured

    def __exit__(self, *_args: object) -> bool:
        original_descriptor = self.captured._descriptor
        if self.reuse_same_inode:
            self.replacement_descriptor = os.dup(original_descriptor)
        if self.transfer_descriptor:
            self.transferred_descriptor = self.captured._take_descriptor()
        if self.close_capture:
            cleanup_errors = CapturedOpaqueFile._close_errors(self.captured)
            if cleanup_errors:
                raise BaseExceptionGroup("test resolver cleanup failed", cleanup_errors)
        if self.reuse_same_inode:
            os.dup2(self.replacement_descriptor, original_descriptor)
            self.reused_descriptor = original_descriptor
        if self.exit_error is not None:
            raise self.exit_error
        return True


class _DirectResolver:
    def __init__(
        self,
        payload: bytes,
        *,
        close_capture: bool,
        exit_error: BaseException | None = None,
        transfer_descriptor: bool = False,
        reuse_same_inode: bool = False,
    ) -> None:
        self.payload = payload
        self.close_capture = close_capture
        self.exit_error = exit_error
        self.transfer_descriptor = transfer_descriptor
        self.reuse_same_inode = reuse_same_inode
        self.contexts: list[_ResolverContext] = []

    def capture(self, request: OpaqueArtifactRequest) -> _ResolverContext:
        context = _ResolverContext(
            _direct_capture(self.payload, request),
            close_capture=self.close_capture,
            exit_error=self.exit_error,
            transfer_descriptor=self.transfer_descriptor,
            reuse_same_inode=self.reuse_same_inode,
        )
        self.contexts.append(context)
        return context


def test_resolved_capture_accepts_normal_provider_context() -> None:
    payload = b"normally managed provider capture"
    request = _request(payload)
    retained: list[CapturedOpaqueFile] = []

    class Resolver:
        @contextmanager
        def capture(
            self,
            exact_request: OpaqueArtifactRequest,
        ) -> Iterator[CapturedOpaqueFile]:
            with capture_streamed_opaque_file(
                io.BytesIO(payload),
                exact_request,
            ) as captured:
                retained.append(captured)
                yield captured

    with capture_resolved_opaque_file(Resolver(), request) as captured:
        assert len(retained) == 1
        assert captured is not retained[0]
        assert retained[0].closed
        assert retained[0]._descriptor == -1
        assert not captured.closed
        assert b"".join(captured.iter_chunks()) == payload

    assert len(retained) == 1
    assert retained[0].closed
    assert retained[0]._descriptor == -1
    assert not retained[0]._readers
    assert captured.closed
    assert captured._descriptor == -1


def test_resolved_anonymous_capture_isolated_from_provider_retained_writer() -> None:
    payload = b"provider-retained writer must not alias detached bytes"
    request = _request(payload)

    class ResolverContext:
        def __init__(self, captured: CapturedOpaqueFile) -> None:
            self.captured = captured

        def __enter__(self) -> CapturedOpaqueFile:
            return self.captured

        def __exit__(self, *_args: object) -> bool:
            cleanup_errors = CapturedOpaqueFile._close_errors(self.captured)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "provider capture cleanup failed",
                    cleanup_errors,
                )
            return True

    class Resolver:
        def __init__(self) -> None:
            self.writer: BinaryIO | None = None
            self.captured: CapturedOpaqueFile | None = None
            self.provider_identity: tuple[int, int] | None = None

        def capture(self, resolver_request: OpaqueArtifactRequest) -> ResolverContext:
            self.writer = captured_artifacts.tempfile.TemporaryFile()
            writer_descriptor = self.writer.fileno()
            captured_artifacts._write_all(writer_descriptor, payload)
            os.fsync(writer_descriptor)
            os.fchmod(writer_descriptor, 0o400)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            reader_descriptor = os.open(
                f"/proc/self/fd/{writer_descriptor}",
                flags,
            )
            metadata = os.fstat(reader_descriptor)
            self.provider_identity = (metadata.st_dev, metadata.st_ino)
            self.captured = CapturedOpaqueFile(
                request=resolver_request,
                descriptor=reader_descriptor,
                storage_kind="anonymous_snapshot",
                descriptor_state=captured_artifacts._descriptor_state(metadata),
            )
            self.captured.require_intact()
            return ResolverContext(self.captured)

    resolver = Resolver()
    try:
        with capture_resolved_opaque_file(resolver, request) as detached:
            assert resolver.captured is not None
            assert resolver.captured.closed
            assert resolver.provider_identity is not None
            assert detached._descriptor_identity != resolver.provider_identity
            assert resolver.writer is not None
            os.pwrite(resolver.writer.fileno(), b"X", 0)
            os.fsync(resolver.writer.fileno())
            assert b"".join(detached.iter_chunks(chunk_size=4)) == payload
            detached.require_intact()
    finally:
        if resolver.captured is not None:
            CapturedOpaqueFile._close_errors(resolver.captured)
        if resolver.writer is not None:
            resolver.writer.close()


def test_resolved_capture_rejects_resolver_request_mutation() -> None:
    admitted_payload = b"admitted"
    forged_payload = b"forged!!"
    request = _request(admitted_payload)
    original_snapshot = (
        request.uri,
        request.sha256,
        request.size_bytes,
        request.max_bytes,
    )
    contexts: list[_ResolverContext] = []
    resolver_requests: list[OpaqueArtifactRequest] = []

    class Resolver:
        def capture(self, resolver_request: OpaqueArtifactRequest) -> _ResolverContext:
            assert resolver_request is not request
            resolver_requests.append(resolver_request)
            object.__setattr__(
                resolver_request,
                "uri",
                "artifact://tests/forged-payload.bin",
            )
            object.__setattr__(
                resolver_request,
                "sha256",
                hashlib.sha256(forged_payload).hexdigest(),
            )
            context = _ResolverContext(
                _direct_capture(forged_payload, resolver_request),
                close_capture=True,
            )
            contexts.append(context)
            return context

    with pytest.raises(CapturedArtifactError) as raised:
        with capture_resolved_opaque_file(Resolver(), request):
            pytest.fail("mutated resolver request must not cross the trust boundary")

    assert raised.value.code == "capture_representation_invalid"
    assert (
        request.uri,
        request.sha256,
        request.size_bytes,
        request.max_bytes,
    ) == original_snapshot
    assert len(resolver_requests) == 1
    assert len(contexts) == 1
    assert contexts[0].captured.closed


def test_resolved_capture_rejects_live_caller_request_mutation() -> None:
    payload = b"caller request mutation"
    request = _request(payload)
    contexts: list[_ResolverContext] = []

    class Resolver:
        def capture(self, resolver_request: OpaqueArtifactRequest) -> _ResolverContext:
            object.__setattr__(
                request,
                "uri",
                "artifact://tests/mutated-by-resolver.bin",
            )
            context = _ResolverContext(
                _direct_capture(payload, resolver_request),
                close_capture=True,
            )
            contexts.append(context)
            return context

    with pytest.raises(CapturedArtifactError) as raised:
        with capture_resolved_opaque_file(Resolver(), request):
            pytest.fail("mutated caller request must not cross the trust boundary")

    assert raised.value.code == "capture_representation_invalid"
    assert len(contexts) == 1
    assert contexts[0].captured.closed


def test_resolved_capture_rejects_noop_provider_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"provider no-op cleanup"
    request = _request(payload)
    resolver = _DirectResolver(payload, close_capture=False)
    published = False
    detached_captures: list[CapturedOpaqueFile] = []
    real_detach = captured_artifacts._detach_resolved_capture

    def record_detached_capture(
        provider_capture: CapturedOpaqueFile,
        request_snapshot: tuple[str, str, int, int],
    ) -> CapturedOpaqueFile:
        detached = real_detach(provider_capture, request_snapshot)
        detached_captures.append(detached)
        return detached

    monkeypatch.setattr(
        captured_artifacts,
        "_detach_resolved_capture",
        record_detached_capture,
    )

    try:
        with pytest.raises(BaseExceptionGroup) as raised:
            with capture_resolved_opaque_file(resolver, request):
                published = True

        cleanup_codes = {
            failure.code
            for failure in _leaf_exceptions(raised.value)
            if isinstance(failure, CapturedArtifactCleanupError)
        }
        assert cleanup_codes == {
            "resolver_capture_ownership_retained",
            "descriptor_release_unverifiable",
        }
        assert not published
        captured = resolver.contexts[0].captured
        assert not captured.closed
        assert captured._descriptor >= 0
        assert not captured._readers
        os.fstat(captured._descriptor)
        assert len(detached_captures) == 1
        assert detached_captures[0].closed
        assert detached_captures[0]._descriptor == -1
        assert not detached_captures[0]._readers
    finally:
        for context in resolver.contexts:
            CapturedOpaqueFile._close_errors(context.captured)

    assert captured.closed
    assert captured._descriptor == -1


def test_resolved_capture_never_suppresses_exit_stack_primary() -> None:
    payload = b"non-suppressible provider context"
    request = _request(payload)
    resolver = _DirectResolver(payload, close_capture=True)
    primary = RuntimeError("primary verifier failure")

    with pytest.raises(RuntimeError) as raised:
        with ExitStack() as stack:
            stack.enter_context(capture_resolved_opaque_file(resolver, request))
            raise primary

    assert raised.value is primary
    assert getattr(primary, "__notes__", ()) == ()
    assert resolver.contexts[0].captured.closed


def test_outer_resolver_cannot_suppress_inner_resolver_cleanup_failure() -> None:
    payload = b"composed resolver teardown"
    request = _request(payload)
    outer = _DirectResolver(payload, close_capture=True)
    inner_error = RuntimeError("inner provider teardown failure")
    inner = _DirectResolver(
        payload,
        close_capture=True,
        exit_error=inner_error,
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        with ExitStack() as stack:
            stack.enter_context(capture_resolved_opaque_file(outer, request))
            stack.enter_context(capture_resolved_opaque_file(inner, request))

    leaves = _leaf_exceptions(raised.value)
    assert inner_error in leaves
    assert any(
        isinstance(failure, CapturedArtifactCleanupError)
        and failure.code == "resolver_capture_cleanup_failed"
        for failure in leaves
    )
    assert outer.contexts[0].captured.closed
    assert inner.contexts[0].captured.closed


def test_resolved_capture_recognizes_nested_typed_cleanup_failure() -> None:
    payload = b"nested provider cleanup"
    typed_cleanup = CapturedArtifactCleanupError(
        "provider_cleanup_retained",
        "provider retained cleanup ownership",
    )
    nested_cleanup = BaseExceptionGroup(
        "outer provider cleanup",
        [BaseExceptionGroup("inner provider cleanup", [typed_cleanup])],
    )
    resolver = _DirectResolver(
        payload,
        close_capture=True,
        exit_error=nested_cleanup,
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        with capture_resolved_opaque_file(resolver, _request(payload)):
            pytest.fail("provider teardown failure must block publication")

    assert raised.value is nested_cleanup
    cleanup_codes = {
        failure.code
        for failure in _leaf_exceptions(raised.value)
        if isinstance(failure, CapturedArtifactCleanupError)
    }
    assert cleanup_codes == {"provider_cleanup_retained"}
    assert resolver.contexts[0].captured.closed


def test_resolved_capture_blocks_publication_and_preserves_teardown_failures() -> None:
    payload = b"provider teardown failures"
    request = _request(payload)
    resolver = _DirectResolver(
        payload,
        close_capture=True,
        exit_error=OSError("forced provider teardown failure"),
    )
    published = False

    try:
        with pytest.raises(BaseExceptionGroup) as raised:
            with capture_resolved_opaque_file(resolver, request):
                published = True

        leaves = _leaf_exceptions(raised.value)
        cleanup_codes = {
            failure.code
            for failure in leaves
            if isinstance(failure, CapturedArtifactCleanupError)
        }
        assert cleanup_codes == {"resolver_capture_cleanup_failed"}
        assert any(
            isinstance(failure, OSError)
            and str(failure) == "forced provider teardown failure"
            for failure in leaves
        )
        assert not published
        assert resolver.contexts[0].captured.closed
    finally:
        for context in resolver.contexts:
            CapturedOpaqueFile._close_errors(context.captured)


def test_resolved_capture_revalidates_corruption_when_caller_body_fails() -> None:
    payload = b"primary plus capture corruption"
    request = _request(payload)
    resolver = _DirectResolver(payload, close_capture=True)
    primary = RuntimeError("primary verifier failure")

    with pytest.raises(RuntimeError) as raised:
        with capture_resolved_opaque_file(resolver, request) as captured:
            object.__setattr__(
                captured,
                "_descriptor_state",
                (-1, -1, -1, -1, -1, -1, -1),
            )
            raise primary

    assert raised.value is primary
    notes = "\n".join(getattr(primary, "__notes__", ()))
    assert "post-use detached capture validation also failed" in notes
    assert "invalid concrete capture representation" in notes
    assert resolver.contexts[0].captured.closed


def test_resolved_capture_reports_cleanup_only_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"resolved cleanup-only failure"
    resolver = _DirectResolver(payload, close_capture=True)
    real_close = captured_artifacts.os.close
    detached_descriptor = -1

    with pytest.raises(OSError, match="forced detached close failure"):
        with capture_resolved_opaque_file(
            resolver,
            _request(payload),
        ) as captured:
            detached_descriptor = captured._descriptor

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                if descriptor == detached_descriptor:
                    raise OSError("forced detached close failure")

            monkeypatch.setattr(captured_artifacts.os, "close", close_then_fail)

    assert captured.closed
    assert captured._descriptor == -1
    assert captured._readers == {}
    with pytest.raises(OSError) as closed:
        os.fstat(detached_descriptor)
    assert closed.value.errno == errno.EBADF


def test_resolved_capture_never_closes_descriptor_rebound_during_teardown() -> None:
    payload = b"provider descriptor rebind"
    request = _request(payload)
    published = False

    class RebindingContext:
        def __init__(self, captured: CapturedOpaqueFile) -> None:
            self.captured = captured
            self.rebound_descriptor = -1
            self.replacement_source = -1

        def __enter__(self) -> CapturedOpaqueFile:
            return self.captured

        def __exit__(self, *_args: object) -> bool:
            original_descriptor = self.captured._descriptor
            self.replacement_source = os.open("/dev/null", os.O_RDONLY)
            os.close(original_descriptor)
            os.dup2(self.replacement_source, original_descriptor)
            self.rebound_descriptor = original_descriptor
            object.__setattr__(self.captured, "_descriptor", -1)
            object.__setattr__(self.captured, "_closed", True)
            return True

    class Resolver:
        def __init__(self) -> None:
            self.context: RebindingContext | None = None

        def capture(self, resolver_request: OpaqueArtifactRequest) -> RebindingContext:
            self.context = RebindingContext(_direct_capture(payload, resolver_request))
            return self.context

    resolver = Resolver()
    try:
        with pytest.raises(CapturedArtifactCleanupError) as raised:
            with capture_resolved_opaque_file(resolver, request):
                published = True

        assert raised.value.code == "descriptor_release_unverifiable"
        assert not published
        assert resolver.context is not None
        os.fstat(resolver.context.rebound_descriptor)
        os.fstat(resolver.context.replacement_source)
    finally:
        if resolver.context is not None:
            for descriptor in {
                resolver.context.rebound_descriptor,
                resolver.context.replacement_source,
            }:
                if descriptor >= 0:
                    os.close(descriptor)


def test_resolved_capture_reports_but_does_not_close_transferred_descriptor() -> None:
    payload = b"provider descriptor transfer"
    request = _request(payload)
    resolver = _DirectResolver(
        payload,
        close_capture=False,
        transfer_descriptor=True,
    )

    context: _ResolverContext | None = None
    try:
        with pytest.raises(CapturedArtifactCleanupError) as raised:
            with capture_resolved_opaque_file(resolver, request):
                pass

        assert raised.value.code == "descriptor_release_unverifiable"
        context = resolver.contexts[0]
        assert context.captured.closed
        os.fstat(context.transferred_descriptor)
    finally:
        if context is not None and context.transferred_descriptor >= 0:
            os.close(context.transferred_descriptor)
    assert context is not None
    with pytest.raises(OSError):
        os.fstat(context.transferred_descriptor)


def test_resolved_capture_does_not_close_same_inode_reused_descriptor() -> None:
    payload = b"same inode descriptor reuse"
    request = _request(payload)
    resolver = _DirectResolver(
        payload,
        close_capture=True,
        reuse_same_inode=True,
    )
    context: _ResolverContext | None = None
    try:
        with pytest.raises(CapturedArtifactCleanupError) as raised:
            with capture_resolved_opaque_file(resolver, request):
                pass

        assert raised.value.code == "descriptor_release_unverifiable"
        context = resolver.contexts[0]
        reused_state = os.fstat(context.reused_descriptor)
        replacement_state = os.fstat(context.replacement_descriptor)
        assert (reused_state.st_dev, reused_state.st_ino) == (
            replacement_state.st_dev,
            replacement_state.st_ino,
        )
    finally:
        if context is not None:
            if context.reused_descriptor >= 0:
                os.close(context.reused_descriptor)
            if context.replacement_descriptor >= 0:
                os.close(context.replacement_descriptor)


def test_resolved_capture_never_closes_post_exit_poisoned_descriptors() -> None:
    payload = b"provider teardown ownership poison"
    request = _request(payload)

    class PoisoningContext:
        def __init__(self, captured: CapturedOpaqueFile) -> None:
            self.captured = captured
            self.rebound_descriptor = -1
            self.replacement_source = -1
            self.poisoned_reader = -1

        def __enter__(self) -> CapturedOpaqueFile:
            return self.captured

        def __exit__(self, *_args: object) -> bool:
            original_descriptor = self.captured._descriptor
            self.replacement_source = os.open("/dev/null", os.O_RDONLY)
            os.close(original_descriptor)
            os.dup2(self.replacement_source, original_descriptor)
            self.rebound_descriptor = original_descriptor
            rebound_metadata = os.fstat(self.rebound_descriptor)
            self.poisoned_reader = os.open("/dev/null", os.O_RDONLY)
            reader_metadata = os.fstat(self.poisoned_reader)
            object.__setattr__(self.captured, "_descriptor", self.rebound_descriptor)
            object.__setattr__(
                self.captured,
                "_descriptor_identity",
                (rebound_metadata.st_dev, rebound_metadata.st_ino),
            )
            object.__setattr__(self.captured, "_closed", False)
            object.__setattr__(
                self.captured,
                "_readers",
                {
                    self.poisoned_reader: (
                        reader_metadata.st_dev,
                        reader_metadata.st_ino,
                    )
                },
            )
            return True

    class Resolver:
        def __init__(self) -> None:
            self.context: PoisoningContext | None = None

        def capture(self, resolver_request: OpaqueArtifactRequest) -> PoisoningContext:
            self.context = PoisoningContext(_direct_capture(payload, resolver_request))
            return self.context

    resolver = Resolver()
    try:
        with pytest.raises(BaseExceptionGroup) as raised:
            with capture_resolved_opaque_file(resolver, request):
                pass

        failures: list[BaseException] = list(raised.value.exceptions)
        leaf_failures: list[BaseException] = []
        while failures:
            failure = failures.pop()
            if isinstance(failure, BaseExceptionGroup):
                failures.extend(failure.exceptions)
            else:
                leaf_failures.append(failure)
        cleanup_codes = {
            failure.code
            for failure in leaf_failures
            if isinstance(failure, CapturedArtifactCleanupError)
        }
        assert cleanup_codes == {
            "resolver_capture_ownership_retained",
            "descriptor_release_unverifiable",
        }

        assert resolver.context is not None
        os.fstat(resolver.context.rebound_descriptor)
        os.fstat(resolver.context.replacement_source)
        os.fstat(resolver.context.poisoned_reader)
    finally:
        if resolver.context is not None:
            for descriptor in {
                resolver.context.rebound_descriptor,
                resolver.context.replacement_source,
                resolver.context.poisoned_reader,
            }:
                if descriptor >= 0:
                    os.close(descriptor)


def test_retained_descriptor_release_rejects_reused_number(
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "original.bin"
    replacement_path = tmp_path / "replacement.bin"
    original_path.write_bytes(b"original")
    replacement_path.write_bytes(b"replacement")
    original = os.open(original_path, os.O_RDONLY)
    replacement = os.open(replacement_path, os.O_RDONLY)
    reused = os.dup(original)
    original_state = os.fstat(reused)
    try:
        os.dup2(replacement, reused)
        failures = captured_artifacts._verify_resolver_descriptor_release(
            reused,
            (original_state.st_dev, original_state.st_ino),
        )
        assert len(failures) == 1
        assert isinstance(failures[0], CapturedArtifactCleanupError)
        assert failures[0].code == "descriptor_release_unverifiable"
        assert os.fstat(reused).st_ino == os.fstat(replacement).st_ino
    finally:
        os.close(reused)
        os.close(replacement)
        os.close(original)


def test_resolved_capture_rejects_subclass_without_virtual_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    with monkeypatch.context() as subclass_patch:
        subclass_patch.setattr(
            CapturedOpaqueFile,
            "__init_subclass__",
            classmethod(lambda _cls, **_kwargs: None),
        )

        class MaliciousCapturedOpaqueFile(CapturedOpaqueFile):
            def __init__(self) -> None:
                pass

            def require_intact(self) -> None:
                calls.append("require_intact")

        malicious = MaliciousCapturedOpaqueFile()

    class Context:
        def __enter__(self) -> CapturedOpaqueFile:
            return malicious

        def __exit__(self, *_args: object) -> bool:
            return True

    class Resolver:
        def capture(self, _request: OpaqueArtifactRequest) -> Context:
            return Context()

    with pytest.raises(CapturedArtifactError) as raised:
        with capture_resolved_opaque_file(Resolver(), _request(b"payload")):
            pytest.fail("malicious subtype must not enter the trusted boundary")

    assert raised.value.code == "capture_type_mismatch"
    assert calls == []


def test_resolved_capture_rejects_poisoned_exact_type_without_entering_lock() -> None:
    payload = b"poisoned exact capture"
    request = _request(payload)
    calls: list[str] = []

    class SuppressingLock:
        def __enter__(self) -> SuppressingLock:
            calls.append("enter")
            return self

        def __exit__(self, *_args: object) -> bool:
            calls.append("exit")
            return True

    forged = object.__new__(CapturedOpaqueFile)
    object.__setattr__(
        forged,
        "_request_snapshot",
        (request.uri, request.sha256, request.size_bytes, request.max_bytes),
    )
    object.__setattr__(forged, "_descriptor", -1)
    object.__setattr__(forged, "_storage_kind", "sealed_memfd")
    object.__setattr__(forged, "_descriptor_state", (0, 0, 0, 0, 0, 0, 0))
    object.__setattr__(forged, "_descriptor_identity", (0, 0))
    object.__setattr__(forged, "_closed", False)
    object.__setattr__(forged, "_readers", {})
    object.__setattr__(forged, "_lock", SuppressingLock())

    class Context:
        def __enter__(self) -> CapturedOpaqueFile:
            return forged

        def __exit__(self, *_args: object) -> bool:
            return True

    class Resolver:
        def capture(self, _request: OpaqueArtifactRequest) -> Context:
            return Context()

    with pytest.raises(CapturedArtifactError) as raised:
        with capture_resolved_opaque_file(Resolver(), request):
            pytest.fail("poisoned exact capture must not cross the trust boundary")

    assert raised.value.code == "capture_representation_invalid"
    assert calls == []


def test_resolved_capture_never_closes_unvalidated_initial_descriptor() -> None:
    request = _request(b"untrusted initial descriptor")
    unrelated_descriptor = os.open("/dev/null", os.O_RDONLY)
    metadata = os.fstat(unrelated_descriptor)
    forged = object.__new__(CapturedOpaqueFile)
    object.__setattr__(
        forged,
        "_request_snapshot",
        (request.uri, request.sha256, request.size_bytes, request.max_bytes),
    )
    object.__setattr__(forged, "_descriptor", unrelated_descriptor)
    object.__setattr__(forged, "_storage_kind", "sealed_memfd")
    object.__setattr__(
        forged,
        "_descriptor_state",
        captured_artifacts._descriptor_state(metadata),
    )
    object.__setattr__(
        forged,
        "_descriptor_identity",
        (metadata.st_dev, metadata.st_ino),
    )
    object.__setattr__(forged, "_closed", False)
    object.__setattr__(forged, "_readers", {})
    object.__setattr__(forged, "_lock", captured_artifacts.threading.RLock())

    class Context:
        def __enter__(self) -> CapturedOpaqueFile:
            return forged

        def __exit__(self, *_args: object) -> bool:
            return True

    class Resolver:
        def capture(self, _request: OpaqueArtifactRequest) -> Context:
            return Context()

    try:
        with pytest.raises(CapturedArtifactError) as raised:
            with capture_resolved_opaque_file(Resolver(), request):
                pytest.fail("unvalidated descriptor must not cross the trust boundary")

        assert raised.value.code == "descriptor_reused"
        os.fstat(unrelated_descriptor)
    finally:
        os.close(unrelated_descriptor)


def test_resolved_capture_revalidates_provider_after_detachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"post-detachment provider mutation"
    request = _request(payload)
    resolver = _DirectResolver(payload, close_capture=False)
    real_detach = captured_artifacts._detach_resolved_capture
    original_descriptor = -1
    unrelated_descriptor = -1
    detached_descriptor = -1

    def poison_after_detachment(
        captured: CapturedOpaqueFile,
        request_snapshot: tuple[str, str, int, int],
    ) -> CapturedOpaqueFile:
        nonlocal original_descriptor
        nonlocal unrelated_descriptor
        nonlocal detached_descriptor
        detached = real_detach(captured, request_snapshot)
        detached_descriptor = detached._descriptor
        original_descriptor = captured._descriptor
        unrelated_descriptor = os.open("/dev/null", os.O_RDONLY)
        metadata = os.fstat(unrelated_descriptor)
        object.__setattr__(captured, "_descriptor", unrelated_descriptor)
        object.__setattr__(
            captured,
            "_descriptor_identity",
            (metadata.st_dev, metadata.st_ino),
        )
        return detached

    try:
        with monkeypatch.context() as detach_patch:
            detach_patch.setattr(
                captured_artifacts,
                "_detach_resolved_capture",
                poison_after_detachment,
            )
            with pytest.raises(CapturedArtifactError) as raised:
                with capture_resolved_opaque_file(resolver, request):
                    pytest.fail("capture mutated after detachment must not be yielded")

        assert raised.value.code == "capture_representation_invalid"
        os.fstat(original_descriptor)
        os.fstat(unrelated_descriptor)
        with pytest.raises(OSError):
            os.fstat(detached_descriptor)
    finally:
        for descriptor in {original_descriptor, unrelated_descriptor}:
            if descriptor >= 0:
                os.close(descriptor)


def test_failed_sealed_capture_detachment_closes_helper_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if captured_artifacts._MEMFD_CREATE is None:
        pytest.skip("platform does not provide memfd_create")

    payload = b"failed sealed capture detachment"
    expected = CapturedArtifactError(
        "capture_detach_failed",
        "forced detached capture validation failure",
    )
    detached_descriptor = -1
    real_require_intact = CapturedOpaqueFile.require_intact

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as provider_capture:
        assert provider_capture.storage_kind == "sealed_memfd"

        def fail_detached_validation(captured: CapturedOpaqueFile) -> None:
            nonlocal detached_descriptor
            if captured is provider_capture:
                real_require_intact(captured)
                return
            detached_descriptor = captured._descriptor
            raise expected

        monkeypatch.setattr(
            CapturedOpaqueFile,
            "require_intact",
            fail_detached_validation,
        )
        with pytest.raises(CapturedArtifactError) as raised:
            captured_artifacts._detach_resolved_capture(
                provider_capture,
                (
                    provider_capture.uri,
                    provider_capture.sha256,
                    provider_capture.size_bytes,
                    provider_capture.max_bytes,
                ),
            )

    assert raised.value is expected
    assert detached_descriptor >= 0
    with pytest.raises(OSError) as closed:
        os.fstat(detached_descriptor)
    assert closed.value.errno == errno.EBADF


def test_resolved_capture_rejects_incomplete_exact_type_representation() -> None:
    incomplete = object.__new__(CapturedOpaqueFile)

    class Context:
        def __enter__(self) -> CapturedOpaqueFile:
            return incomplete

        def __exit__(self, *_args: object) -> bool:
            return True

    class Resolver:
        def capture(self, _request: OpaqueArtifactRequest) -> Context:
            return Context()

    with pytest.raises(CapturedArtifactError) as raised:
        with capture_resolved_opaque_file(Resolver(), _request(b"payload")):
            pytest.fail("incomplete exact capture must not cross the trust boundary")

    assert raised.value.code == "capture_representation_invalid"
    assert "ownership could not be snapshotted" in "\n".join(
        getattr(raised.value, "__notes__", ())
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {
                "uri": "",
                "sha256": "0" * 64,
                "size_bytes": 0,
                "max_bytes": 0,
            },
            ValueError,
            "uri",
        ),
        (
            {
                "uri": _ComparisonForgingSha256("artifact://valid"),
                "sha256": "0" * 64,
                "size_bytes": 0,
                "max_bytes": 0,
            },
            TypeError,
            "uri",
        ),
        (
            {
                "uri": "artifact://valid",
                "sha256": "A" * 64,
                "size_bytes": 0,
                "max_bytes": 0,
            },
            ValueError,
            "sha256",
        ),
        (
            {
                "uri": "artifact://valid",
                "sha256": "0" * 64,
                "size_bytes": True,
                "max_bytes": 1,
            },
            TypeError,
            "size_bytes",
        ),
        (
            {
                "uri": "artifact://valid",
                "sha256": "0" * 64,
                "size_bytes": 2,
                "max_bytes": 1,
            },
            ValueError,
            "max_bytes",
        ),
    ],
)
def test_request_rejects_invalid_identity_before_capture(
    kwargs: dict[str, Any],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        OpaqueArtifactRequest(**kwargs)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"memfd_max_bytes": True}, "memfd_max_bytes"),
        (
            {"memfd_max_bytes": _ComparisonForgingBudget(1)},
            "memfd_max_bytes",
        ),
        ({"chunk_size": True}, "chunk_size"),
        ({"chunk_size": _ComparisonForgingBudget(1)}, "chunk_size"),
    ],
)
def test_stream_capture_options_reject_bool_and_integer_subclasses(
    options: dict[str, Any],
    message: str,
) -> None:
    payload = b"exact capture options"
    with pytest.raises(TypeError, match=message):
        with capture_streamed_opaque_file(
            io.BytesIO(payload),
            _request(payload),
            **options,
        ):
            pass


def test_published_request_mutation_cannot_rewrite_private_capture_identity() -> None:
    payload = b"private admitted capture identity"
    admitted = _request(payload)

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        admitted,
    ) as captured:
        exposed = captured.request
        object.__setattr__(
            exposed,
            "uri",
            "artifact://tests/forged-public-projection.bin",
        )
        object.__setattr__(
            exposed,
            "sha256",
            _ComparisonForgingSha256("0" * 64),
        )
        object.__setattr__(exposed, "size_bytes", 0)
        object.__setattr__(
            exposed,
            "max_bytes",
            _ComparisonForgingBudget(0),
        )

        fresh_projection = captured.request
        assert fresh_projection is not exposed
        assert fresh_projection == admitted
        assert captured.uri == admitted.uri
        assert captured.sha256 == admitted.sha256
        assert captured.size_bytes == admitted.size_bytes
        assert captured.max_bytes == admitted.max_bytes
        assert b"".join(captured.iter_chunks(chunk_size=4)) == payload
        captured.require_intact()

    assert captured.closed


def test_local_capture_yields_only_immutable_bounded_reader_surface(
    tmp_path: Path,
) -> None:
    payload = b"captured local bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)

    with capture_local_opaque_file(source, _request(payload)) as captured:
        assert captured.request == _request(payload)
        assert captured.uri == "artifact://tests/payload.bin"
        assert captured.sha256 == hashlib.sha256(payload).hexdigest()
        assert captured.size_bytes == len(payload)
        assert captured.max_bytes == len(payload)
        assert captured.storage_kind in {"sealed_memfd", "anonymous_snapshot"}
        assert not hasattr(captured, "path")
        assert not hasattr(captured, "descriptor")
        assert not hasattr(captured, "payload")
        assert not hasattr(captured, "read_bytes")
        assert list(captured.iter_chunks(chunk_size=4)) == [
            b"capt",
            b"ured",
            b" loc",
            b"al b",
            b"ytes",
        ]
        captured.require_intact()
        assert not captured.closed

    assert captured.closed
    with pytest.raises(CapturedArtifactError, match="already closed"):
        captured.require_intact()
    assert captured._close_errors() == []


def test_local_source_close_failure_closes_detached_snapshot_before_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"source cleanup failure must not orphan its detached snapshot"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    captures: list[Any] = []
    real_capture = captured_artifacts._capture_open_file_snapshot
    real_close = captured_artifacts.os.close

    def track_capture(*args: Any, **kwargs: Any) -> Any:
        captured = real_capture(*args, **kwargs)
        captures.append(captured)
        return captured

    def close_source_then_fail(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        real_close(descriptor)
        if (metadata.st_dev, metadata.st_ino) == source_identity:
            raise OSError("forced local source descriptor close failure")

    monkeypatch.setattr(
        captured_artifacts,
        "_capture_open_file_snapshot",
        track_capture,
    )
    monkeypatch.setattr(captured_artifacts.os, "close", close_source_then_fail)

    with pytest.raises(
        OSError,
        match="forced local source descriptor close failure",
    ):
        with capture_local_opaque_file(source, _request(payload)):
            pytest.fail("source cleanup failure must fail before yield")

    assert len(captures) == 1
    captured = captures[0]
    assert captured.closed
    assert captured._descriptor == -1


def test_small_capture_uses_sealed_memfd_when_available() -> None:
    if captured_artifacts._MEMFD_CREATE is None:
        pytest.skip("platform does not provide memfd_create")
    payload = b"small sealed capture"

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        assert captured.storage_kind == "sealed_memfd"


def test_public_capture_never_falls_back_to_pinned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"disk fallback remains detached"
    monkeypatch.setattr(captured_artifacts, "_MEMFD_CREATE", None)

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
        snapshot_directory=tmp_path,
    ) as captured:
        assert captured.storage_kind == "anonymous_snapshot"
        assert captured.storage_kind != "pinned_file"


def test_local_capture_detects_in_place_mutation_while_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"a" * 4096
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    real_pread = captured_artifacts.os.pread
    mutated = False

    def mutate_after_first_source_read(
        descriptor: int,
        size: int,
        offset: int,
    ) -> bytes:
        nonlocal mutated
        chunk = real_pread(descriptor, size, offset)
        metadata = os.fstat(descriptor)
        if not mutated and (metadata.st_dev, metadata.st_ino) == source_identity:
            mutated = True
            source.write_bytes(b"b" * len(payload))
        return chunk

    monkeypatch.setattr(
        captured_artifacts.os,
        "pread",
        mutate_after_first_source_read,
    )

    with pytest.raises(CapturedArtifactError) as caught:
        with capture_local_opaque_file(source, _request(payload)):
            pytest.fail("mutated source must fail before yield")

    assert caught.value.code == "source_changed"
    assert mutated


def test_local_capture_detects_path_replacement_after_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"same bytes, different inode"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    replaced = tmp_path / "replacement.bin"
    replaced.write_bytes(payload)
    real_capture = captured_artifacts._capture_open_file_snapshot

    def capture_then_replace(*args: Any, **kwargs: Any) -> Any:
        captured = real_capture(*args, **kwargs)
        source.unlink()
        replaced.rename(source)
        return captured

    monkeypatch.setattr(
        captured_artifacts,
        "_capture_open_file_snapshot",
        capture_then_replace,
    )

    with pytest.raises(CapturedArtifactError) as caught:
        with capture_local_opaque_file(source, _request(payload)):
            pytest.fail("replaced source path must fail before yield")

    assert caught.value.code == "source_changed"


@pytest.mark.parametrize("change", ["unlink", "broken_symlink"])
def test_local_capture_normalizes_disappearing_path_after_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    payload = b"captured before the path disappears"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    input_path = source
    if change == "broken_symlink":
        input_path = tmp_path / "input.bin"
        input_path.symlink_to(source.name)
    real_capture = captured_artifacts._capture_open_file_snapshot

    def capture_then_change_path(*args: Any, **kwargs: Any) -> Any:
        captured = real_capture(*args, **kwargs)
        input_path.unlink()
        if change == "broken_symlink":
            input_path.symlink_to("missing.bin")
        return captured

    monkeypatch.setattr(
        captured_artifacts,
        "_capture_open_file_snapshot",
        capture_then_change_path,
    )

    with pytest.raises(CapturedArtifactError) as caught:
        with capture_local_opaque_file(input_path, _request(payload)):
            pytest.fail("disappearing source path must fail before yield")

    assert caught.value.code == "source_changed"
    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
    ("payload", "declared_payload", "max_bytes", "code"),
    [
        (b"abc", b"abcd", 4, "size_mismatch"),
        (b"abcde", b"abcd", 5, "size_mismatch"),
        (b"abcde", b"abcd", 4, "max_bytes_exceeded"),
    ],
)
def test_stream_capture_rejects_truncation_growth_and_oversize(
    payload: bytes,
    declared_payload: bytes,
    max_bytes: int,
    code: str,
) -> None:
    request = _request(declared_payload, max_bytes=max_bytes)

    with pytest.raises(CapturedArtifactError) as caught:
        with capture_streamed_opaque_file(io.BytesIO(payload), request):
            pytest.fail("invalid streamed size must fail before yield")

    assert caught.value.code == code


def test_stream_capture_rejects_digest_mismatch_before_yield() -> None:
    request = _request(b"declared")

    with pytest.raises(CapturedArtifactError) as caught:
        with capture_streamed_opaque_file(io.BytesIO(b"observed"), request):
            pytest.fail("digest mismatch must fail before yield")

    assert caught.value.code == "digest_mismatch"


def test_interrupted_stream_closes_partial_memfd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if captured_artifacts._MEMFD_CREATE is None:
        pytest.skip("platform does not provide memfd_create")
    created_descriptors: list[int] = []
    real_memfd_create = captured_artifacts._MEMFD_CREATE

    def tracked_memfd_create(name: bytes, flags: int) -> int:
        descriptor = int(real_memfd_create(name, flags))
        created_descriptors.append(descriptor)
        return descriptor

    class InterruptedStream:
        calls = 0

        def read(self, size: int) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return b"ab"[:size]
            raise RuntimeError("transport interrupted")

    monkeypatch.setattr(
        captured_artifacts,
        "_MEMFD_CREATE",
        tracked_memfd_create,
    )
    request = _request(b"abcd")

    with pytest.raises(RuntimeError, match="transport interrupted"):
        with capture_streamed_opaque_file(InterruptedStream(), request):
            pytest.fail("interrupted stream must fail before yield")

    assert len(created_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(created_descriptors[0])


@pytest.mark.parametrize(
    "error_number",
    [errno.EACCES, errno.ENOSYS, errno.EPERM],
)
def test_runtime_unavailable_memfd_falls_back_before_stream_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    payload = b"detached disk fallback"

    def unavailable_memfd(_name: bytes, _flags: int) -> int:
        ctypes.set_errno(error_number)
        return -1

    monkeypatch.setattr(
        captured_artifacts,
        "_MEMFD_CREATE",
        unavailable_memfd,
    )

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
        snapshot_directory=tmp_path,
    ) as captured:
        assert captured.storage_kind == "anonymous_snapshot"
        assert b"".join(captured.iter_chunks()) == payload


@pytest.mark.parametrize(
    "error_number",
    [errno.EMFILE, errno.ENFILE, errno.ENOMEM],
)
def test_memfd_resource_exhaustion_remains_a_hard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    payload = b"must not hide resource exhaustion"
    stream = io.BytesIO(payload)

    def exhausted_memfd(_name: bytes, _flags: int) -> int:
        ctypes.set_errno(error_number)
        return -1

    monkeypatch.setattr(
        captured_artifacts,
        "_MEMFD_CREATE",
        exhausted_memfd,
    )

    with pytest.raises(OSError) as caught:
        with capture_streamed_opaque_file(
            stream,
            _request(payload),
            snapshot_directory=tmp_path,
        ):
            pytest.fail("resource exhaustion must fail before yield")

    assert caught.value.errno == error_number
    assert stream.tell() == 0


def test_open_descriptor_capture_uses_default_anonymous_snapshot_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"default compatibility snapshot factory"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    monkeypatch.setattr(
        captured_artifacts,
        "_create_anonymous_snapshot_file",
        lambda **_kwargs: captured_artifacts.tempfile.TemporaryFile(dir=tmp_path),
    )
    try:
        captured = captured_artifacts._capture_open_file_snapshot(
            source_descriptor,
            uri="artifact://tests/default-snapshot-factory.bin",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            max_bytes=len(payload),
            prefer_disk_snapshot=True,
        )
    finally:
        os.close(source_descriptor)

    transferred_descriptor = -1
    try:
        assert captured.storage_kind == "anonymous_snapshot"
        assert b"".join(captured.iter_chunks()) == payload
        transferred_descriptor = captured._take_descriptor()
        assert captured.closed
        assert os.fstat(transferred_descriptor).st_size == len(payload)
    finally:
        if transferred_descriptor >= 0:
            os.close(transferred_descriptor)
        assert captured._close_errors() == []


def test_disk_snapshot_hash_failure_closes_writer_and_partial_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"disk snapshot finalization failure"
    snapshot_file = captured_artifacts.tempfile.TemporaryFile(dir=tmp_path)
    binding_descriptors: list[int] = []
    real_open = captured_artifacts.os.open

    def track_binding_open(*args: Any, **kwargs: Any) -> int:
        descriptor = real_open(*args, **kwargs)
        binding_descriptors.append(descriptor)
        return descriptor

    def fail_final_hash(_descriptor: int) -> str:
        raise CapturedArtifactError(
            "snapshot_changed",
            "forced disk snapshot final hash failure",
        )

    monkeypatch.setattr(captured_artifacts.os, "open", track_binding_open)
    monkeypatch.setattr(
        captured_artifacts,
        "_stable_descriptor_sha256",
        fail_final_hash,
    )

    with pytest.raises(CapturedArtifactError) as caught:
        captured_artifacts._capture_reader_snapshot(
            lambda size, offset: payload[offset : offset + size],
            uri="artifact://tests/disk-finalization.bin",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            max_bytes=len(payload),
            prefer_disk_snapshot=True,
            memfd_max_bytes=0,
            chunk_size=4,
            snapshot_factory=lambda: snapshot_file,
            source_stability_check=None,
        )

    assert caught.value.code == "snapshot_changed"
    assert snapshot_file.closed
    assert len(binding_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(binding_descriptors[0])


def test_partial_snapshot_descriptor_is_closed_when_final_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"partially finalized snapshot"
    created_descriptor = -1

    def fail_final_validation(captured: Any) -> None:
        nonlocal created_descriptor
        created_descriptor = captured._descriptor
        raise CapturedArtifactError(
            "snapshot_changed",
            "forced final snapshot validation failure",
        )

    monkeypatch.setattr(
        captured_artifacts.CapturedOpaqueFile,
        "require_intact",
        fail_final_validation,
    )

    with pytest.raises(CapturedArtifactError) as caught:
        captured_artifacts._capture_reader_snapshot(
            lambda size, offset: payload[offset : offset + size],
            uri="artifact://tests/partial-finalization.bin",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            max_bytes=len(payload),
            prefer_disk_snapshot=False,
            memfd_max_bytes=len(payload),
            chunk_size=4,
            snapshot_factory=lambda: pytest.fail(
                "sealed memfd capture must not request disk storage"
            ),
            source_stability_check=None,
        )

    assert caught.value.code == "snapshot_changed"
    assert created_descriptor >= 0
    with pytest.raises(OSError):
        os.fstat(created_descriptor)


def test_descriptor_reuse_is_reported_without_closing_replacement(
    tmp_path: Path,
) -> None:
    payload = b"descriptor reuse"
    replacement_path = tmp_path / "replacement.bin"
    replacement_path.write_bytes(b"replacement")
    replacement_source = -1
    reused_descriptor = -1
    try:
        with pytest.raises(CapturedArtifactError) as caught:
            with capture_streamed_opaque_file(
                io.BytesIO(payload),
                _request(payload),
            ) as captured:
                reused_descriptor = captured._descriptor
                os.close(reused_descriptor)
                replacement_source = os.open(replacement_path, os.O_RDONLY)
                if replacement_source != reused_descriptor:
                    os.dup2(replacement_source, reused_descriptor)

        assert caught.value.code == "descriptor_reused"
        notes = "\n".join(getattr(caught.value, "__notes__", ()))
        assert "refusing to close" in notes
        assert os.fstat(reused_descriptor).st_size == len(b"replacement")
    finally:
        if reused_descriptor >= 0:
            try:
                os.close(reused_descriptor)
            except OSError:
                pass
        if replacement_source >= 0 and replacement_source != reused_descriptor:
            os.close(replacement_source)


def test_failed_legacy_descriptor_transfer_closes_retained_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"failed transfer must retain exactly one cleanup owner"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        captured = captured_artifacts._capture_open_file_snapshot(
            source_descriptor,
            uri="artifact://tests/legacy-transfer.bin",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            max_bytes=len(payload),
        )
    finally:
        os.close(source_descriptor)

    snapshot_descriptor = captured._descriptor

    def fail_transfer_integrity(_descriptor: int) -> str:
        raise CapturedArtifactError(
            "snapshot_changed",
            "forced transfer-time snapshot hash failure",
        )

    monkeypatch.setattr(
        captured_artifacts,
        "_stable_descriptor_sha256",
        fail_transfer_integrity,
    )

    with pytest.raises(CapturedArtifactError) as caught:
        captured._take_descriptor()

    assert caught.value.code == "snapshot_changed"
    assert captured.closed
    assert captured._descriptor == -1
    with pytest.raises(OSError):
        os.fstat(snapshot_descriptor)


def test_memory_backed_snapshot_candidates_are_closed_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Candidate:
        def __init__(self, descriptor: int) -> None:
            self.descriptor = descriptor
            self.closed = False

        def fileno(self) -> int:
            return self.descriptor

        def close(self) -> None:
            self.closed = True

    candidates = [Candidate(101), Candidate(102)]
    remaining = list(candidates)
    monkeypatch.setattr(
        captured_artifacts,
        "_snapshot_candidate_directories",
        lambda **_kwargs: (Path("/candidate-a"), Path("/candidate-b")),
    )
    monkeypatch.setattr(
        captured_artifacts.tempfile,
        "TemporaryFile",
        lambda **_kwargs: remaining.pop(0),
    )
    monkeypatch.setattr(
        captured_artifacts,
        "_descriptor_filesystem_types",
        lambda _descriptor: frozenset({"tmpfs"}),
    )

    with pytest.raises(CapturedArtifactError) as caught:
        captured_artifacts._create_anonymous_snapshot_file(
            source_path=None,
            snapshot_directory=None,
        )

    assert caught.value.code == "snapshot_storage_unavailable"
    assert all(candidate.closed for candidate in candidates)
    assert "memory-backed filesystem tmpfs" in str(caught.value)


def test_memory_backed_snapshot_candidate_cleanup_failure_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Candidate:
        def fileno(self) -> int:
            return 103

        def close(self) -> None:
            raise OSError("forced rejected candidate close failure")

    monkeypatch.setattr(
        captured_artifacts,
        "_snapshot_candidate_directories",
        lambda **_kwargs: (Path("/candidate"),),
    )
    monkeypatch.setattr(
        captured_artifacts.tempfile,
        "TemporaryFile",
        lambda **_kwargs: Candidate(),
    )
    monkeypatch.setattr(
        captured_artifacts,
        "_descriptor_filesystem_types",
        lambda _descriptor: frozenset({"tmpfs"}),
    )

    with pytest.raises(captured_artifacts.CapturedArtifactCleanupError) as caught:
        captured_artifacts._create_anonymous_snapshot_file(
            source_path=None,
            snapshot_directory=None,
        )

    assert caught.value.code == "snapshot_candidate_cleanup_failed"
    assert "forced rejected candidate close failure" in str(caught.value)


def test_anonymous_snapshot_candidate_is_private_and_unlinked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        captured_artifacts,
        "_descriptor_filesystem_types",
        lambda _descriptor: frozenset({"ext4"}),
    )

    candidate = captured_artifacts._create_anonymous_snapshot_file(
        source_path=None,
        snapshot_directory=tmp_path,
    )
    try:
        state = os.fstat(candidate.fileno())
        assert state.st_mode & 0o077 == 0
        assert state.st_nlink == 0
    finally:
        candidate.close()


def test_snapshot_directory_policy_covers_environment_default_and_source_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_directory = tmp_path / "environment"
    monkeypatch.setenv(
        captured_artifacts._SNAPSHOT_DIRECTORY_ENV,
        str(environment_directory),
    )
    assert captured_artifacts._snapshot_candidate_directories(
        source_path=None,
        snapshot_directory=None,
    ) == (environment_directory,)

    monkeypatch.setenv(captured_artifacts._SNAPSHOT_DIRECTORY_ENV, "   ")
    with pytest.raises(CapturedArtifactError) as caught:
        captured_artifacts._snapshot_candidate_directories(
            source_path=None,
            snapshot_directory=None,
        )
    assert caught.value.code == "snapshot_directory_invalid"

    monkeypatch.delenv(captured_artifacts._SNAPSHOT_DIRECTORY_ENV)
    source = tmp_path / "source" / "asset.bin"
    default_directory = tmp_path / "default"
    monkeypatch.setattr(
        captured_artifacts.tempfile,
        "gettempdir",
        lambda: str(default_directory),
    )
    observed = captured_artifacts._snapshot_candidate_directories(
        source_path=source,
        snapshot_directory=None,
    )
    assert observed == (
        default_directory,
        Path("/var/tmp"),
        source.parent,
    )


def test_mountinfo_parser_skips_matching_entry_without_separator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"mountinfo")
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    metadata = os.fstat(descriptor)
    device = f"{os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)}"
    mountinfo = (
        f"1 0 {device} / / rw shared:1 malformed\n"
        f"2 0 {device} / / rw - overlay overlay rw\n"
    )
    real_path_open = Path.open

    def open_mountinfo(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if path == Path("/proc/self/mountinfo"):
            return io.StringIO(mountinfo)
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_mountinfo)
    try:
        assert captured_artifacts._descriptor_filesystem_types(descriptor) == (
            frozenset({"overlay"})
        )
    finally:
        os.close(descriptor)


def test_partial_iterator_owns_no_reader_across_yield_and_fails_after_close() -> None:
    payload = b"ABCDEFGH"
    iterator: Any | None = None
    try:
        with capture_streamed_opaque_file(
            io.BytesIO(payload),
            _request(payload),
        ) as captured:
            iterator = captured.iter_chunks(chunk_size=4)
            assert next(iterator) == b"ABCD"
            assert captured._readers == {}

        with pytest.raises(CapturedArtifactError) as caught:
            next(iterator)

        assert caught.value.code == "capture_closed"
        assert captured.closed
        assert captured._descriptor == -1
        assert captured._readers == {}
    finally:
        if iterator is not None:
            iterator.close()


def test_explicit_partial_iterator_close_revalidates_without_reader_ownership() -> None:
    payload = b"ABCDEFGH"

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        iterator = captured.iter_chunks(chunk_size=4)
        assert next(iterator) == b"ABCD"
        assert captured._readers == {}

        iterator.close()

        assert captured._readers == {}
        captured.require_intact()


def test_resolved_partial_iterator_owns_no_reader_across_yield() -> None:
    payload = b"ABCDEFGH"
    resolver = _DirectResolver(payload, close_capture=True)
    iterator: Any | None = None
    try:
        with capture_resolved_opaque_file(resolver, _request(payload)) as captured:
            assert resolver.contexts[0].captured.closed
            assert captured is not resolver.contexts[0].captured
            iterator = captured.iter_chunks(chunk_size=4)
            assert next(iterator) == b"ABCD"
            assert captured._readers == {}

        assert captured.closed
        assert captured._descriptor == -1
        assert captured._readers == {}
        with pytest.raises(CapturedArtifactError) as caught:
            next(iterator)
        assert caught.value.code == "capture_closed"
    finally:
        if iterator is not None:
            iterator.close()


def test_iter_chunks_hashes_capture_only_before_and_after_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"ABCDEFGH"
    real_sha256 = captured_artifacts._stable_descriptor_sha256

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        hashed_descriptors: list[int] = []

        def record_hash(descriptor: int) -> str:
            hashed_descriptors.append(descriptor)
            return real_sha256(descriptor)

        monkeypatch.setattr(
            captured_artifacts,
            "_stable_descriptor_sha256",
            record_hash,
        )

        assert b"".join(captured.iter_chunks(chunk_size=1)) == payload
        assert hashed_descriptors == [captured._descriptor, captured._descriptor]
        assert captured._readers == {}


def test_bounded_read_cleanup_failure_prevents_yield_and_retains_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"reader cleanup remains attributable"
    reader_descriptor = -1
    real_close = captured_artifacts.os.close

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        iterator = captured.iter_chunks(chunk_size=4)

        def fail_reader_close(descriptor: int) -> None:
            nonlocal reader_descriptor
            if descriptor in captured._readers:
                reader_descriptor = descriptor
                raise OSError("forced pre-close failure")
            real_close(descriptor)

        monkeypatch.setattr(captured_artifacts.os, "close", fail_reader_close)
        with pytest.raises(OSError, match="forced pre-close failure"):
            next(iterator)

        assert reader_descriptor >= 0
        assert reader_descriptor in captured._readers
        assert os.fstat(reader_descriptor).st_size == len(payload)
        monkeypatch.setattr(captured_artifacts.os, "close", real_close)

    with pytest.raises(OSError) as closed:
        os.fstat(reader_descriptor)
    assert closed.value.errno == errno.EBADF


def test_transient_reader_fstat_failure_retains_cleanup_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"transient reader inspection failure"
    reader_descriptor = -1
    real_fstat = captured_artifacts.os.fstat

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        reader_descriptor = captured._open_reader()

        def fail_reader_fstat(descriptor: int) -> os.stat_result:
            if descriptor == reader_descriptor:
                raise OSError(errno.EINTR, "forced transient fstat failure")
            return real_fstat(descriptor)

        monkeypatch.setattr(captured_artifacts.os, "fstat", fail_reader_fstat)
        failures = captured._close_reader_errors(reader_descriptor)

        assert len(failures) == 1
        assert isinstance(failures[0], CapturedArtifactCleanupError)
        assert failures[0].code == "descriptor_state_unverifiable"
        assert reader_descriptor in captured._readers
        assert real_fstat(reader_descriptor).st_size == len(payload)
        monkeypatch.setattr(captured_artifacts.os, "fstat", real_fstat)

    with pytest.raises(OSError) as closed:
        os.fstat(reader_descriptor)
    assert closed.value.errno == errno.EBADF


def test_duplicate_reader_cleanup_is_idempotent() -> None:
    payload = b"idempotent reader cleanup"

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        reader_descriptor = captured._open_reader()
        assert reader_descriptor in captured._readers

        assert captured._close_reader_errors(reader_descriptor) == []
        assert reader_descriptor not in captured._readers
        assert captured._close_reader_errors(reader_descriptor) == []

        with pytest.raises(OSError) as closed:
            os.fstat(reader_descriptor)
        assert closed.value.errno == errno.EBADF


def test_transient_post_close_fstat_failure_retains_cleanup_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"post-close reader inspection failure"
    reader_descriptor = -1
    reader_fstat_calls = 0
    real_close = captured_artifacts.os.close
    real_fstat = captured_artifacts.os.fstat

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        reader_descriptor = captured._open_reader()

        def fail_reader_close(descriptor: int) -> None:
            if descriptor == reader_descriptor:
                raise OSError(errno.EIO, "forced reader close failure")
            real_close(descriptor)

        def fail_second_reader_fstat(descriptor: int) -> os.stat_result:
            nonlocal reader_fstat_calls
            if descriptor == reader_descriptor:
                reader_fstat_calls += 1
                if reader_fstat_calls == 2:
                    raise OSError(errno.EIO, "forced post-close fstat failure")
            return real_fstat(descriptor)

        monkeypatch.setattr(captured_artifacts.os, "close", fail_reader_close)
        monkeypatch.setattr(captured_artifacts.os, "fstat", fail_second_reader_fstat)
        failures = captured._close_reader_errors(reader_descriptor)

        assert len(failures) == 2
        assert isinstance(failures[0], OSError)
        assert isinstance(failures[1], CapturedArtifactCleanupError)
        assert failures[1].code == "descriptor_state_unverifiable"
        assert reader_descriptor in captured._readers
        assert real_fstat(reader_descriptor).st_size == len(payload)
        monkeypatch.setattr(captured_artifacts.os, "close", real_close)
        monkeypatch.setattr(captured_artifacts.os, "fstat", real_fstat)

    with pytest.raises(OSError) as closed:
        os.fstat(reader_descriptor)
    assert closed.value.errno == errno.EBADF


def test_body_error_precedes_snapshot_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"body error precedence"
    real_close = captured_artifacts.os.close

    with pytest.raises(RuntimeError, match="body failed") as caught:
        with capture_streamed_opaque_file(
            io.BytesIO(payload),
            _request(payload),
        ) as captured:
            owned_descriptor = captured._descriptor

            def fail_owned_close(descriptor: int) -> None:
                real_close(descriptor)
                if descriptor == owned_descriptor:
                    raise OSError("forced snapshot close failure")

            monkeypatch.setattr(captured_artifacts.os, "close", fail_owned_close)
            raise RuntimeError("body failed")

    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    assert "captured snapshot cleanup also failed" in notes
    assert "forced snapshot close failure" in notes


def test_stream_capture_reports_cleanup_only_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"stream cleanup-only failure"
    real_close = captured_artifacts.os.close
    snapshot_descriptor = -1

    with pytest.raises(OSError, match="forced stream snapshot close failure"):
        with capture_streamed_opaque_file(
            io.BytesIO(payload),
            _request(payload),
        ) as captured:
            snapshot_descriptor = captured._descriptor

            def close_then_fail(descriptor: int) -> None:
                real_close(descriptor)
                if descriptor == snapshot_descriptor:
                    raise OSError("forced stream snapshot close failure")

            monkeypatch.setattr(captured_artifacts.os, "close", close_then_fail)

    assert captured.closed
    assert captured._descriptor == -1
    assert captured._readers == {}
    with pytest.raises(OSError) as closed:
        os.fstat(snapshot_descriptor)
    assert closed.value.errno == errno.EBADF


def test_bounded_read_error_precedes_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"bounded read and cleanup failure"
    reader_descriptor = -1
    real_pread = captured_artifacts.os.pread
    real_close = captured_artifacts.os.close

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        iterator = captured.iter_chunks(chunk_size=5)

        def fail_reader_read(descriptor: int, size: int, offset: int) -> bytes:
            nonlocal reader_descriptor
            if descriptor in captured._readers:
                reader_descriptor = descriptor
                raise RuntimeError("forced bounded read failure")
            return real_pread(descriptor, size, offset)

        def close_reader_then_fail(descriptor: int) -> None:
            real_close(descriptor)
            if descriptor == reader_descriptor:
                raise OSError("forced reader cleanup failure")

        monkeypatch.setattr(captured_artifacts.os, "pread", fail_reader_read)
        monkeypatch.setattr(captured_artifacts.os, "close", close_reader_then_fail)

        with pytest.raises(RuntimeError, match="forced bounded read failure") as caught:
            next(iterator)

        notes = "\n".join(getattr(caught.value, "__notes__", ()))
        assert "duplicate-reader cleanup also failed" in notes
        assert "forced reader cleanup failure" in notes
        assert reader_descriptor >= 0
        assert captured._readers == {}
        with pytest.raises(OSError):
            os.fstat(reader_descriptor)


def test_bounded_read_revalidates_identity_before_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"ABCDEFGH"
    replacement_path = tmp_path / "replacement.bin"
    replacement_path.write_bytes(b"xxxxYYYY")
    real_pread = captured_artifacts.os.pread
    replacement_source = -1
    rebound_descriptor = -1

    with capture_streamed_opaque_file(
        io.BytesIO(payload),
        _request(payload),
    ) as captured:
        iterator = captured.iter_chunks(chunk_size=4)

        def rebind_during_read(
            descriptor: int,
            size: int,
            offset: int,
        ) -> bytes:
            nonlocal replacement_source
            nonlocal rebound_descriptor
            if descriptor in captured._readers and rebound_descriptor < 0:
                replacement_source = os.open(replacement_path, os.O_RDONLY)
                os.close(descriptor)
                os.dup2(replacement_source, descriptor)
                rebound_descriptor = descriptor
            return real_pread(descriptor, size, offset)

        monkeypatch.setattr(captured_artifacts.os, "pread", rebind_during_read)
        try:
            with pytest.raises(CapturedArtifactError) as caught:
                next(iterator)

            assert caught.value.code == "reader_identity_mismatch"
            assert captured._readers == {}
            assert rebound_descriptor >= 0
            assert os.fstat(rebound_descriptor).st_size == len(b"xxxxYYYY")
        finally:
            for descriptor in {rebound_descriptor, replacement_source}:
                if descriptor >= 0:
                    os.close(descriptor)


def test_post_use_integrity_check_rejects_anonymous_snapshot_mutation(
    tmp_path: Path,
) -> None:
    payload = b"anonymous snapshot must remain exact"

    with pytest.raises(CapturedArtifactError) as caught:
        with capture_streamed_opaque_file(
            io.BytesIO(payload),
            _request(payload),
            memfd_max_bytes=0,
            snapshot_directory=tmp_path,
        ) as captured:
            assert captured.storage_kind == "anonymous_snapshot"
            os.fchmod(captured._descriptor, 0o600)
            writer = os.open(
                f"/proc/self/fd/{captured._descriptor}",
                os.O_WRONLY,
            )
            try:
                os.pwrite(writer, b"changed", 0)
            finally:
                os.close(writer)

    assert caught.value.code == "snapshot_changed"


def test_large_stream_capture_and_duplicate_reader_are_chunk_bounded(
    tmp_path: Path,
) -> None:
    size_bytes = 5 * 1024 * 1024 + 123
    capture_chunk_size = 128 * 1024
    read_chunk_size = 64 * 1024
    digest = hashlib.sha256()
    remaining = size_bytes
    while remaining:
        chunk = b"x" * min(capture_chunk_size, remaining)
        digest.update(chunk)
        remaining -= len(chunk)

    class GeneratedStream:
        def __init__(self) -> None:
            self.remaining = size_bytes
            self.requests: list[int] = []
            self.largest_response = 0

        def read(self, size: int) -> bytes:
            self.requests.append(size)
            amount = min(size, self.remaining)
            self.remaining -= amount
            self.largest_response = max(self.largest_response, amount)
            return b"x" * amount

    stream = GeneratedStream()
    request = OpaqueArtifactRequest(
        uri="artifact://tests/large.bin",
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
        max_bytes=size_bytes,
    )

    with capture_streamed_opaque_file(
        stream,
        request,
        memfd_max_bytes=1024,
        chunk_size=capture_chunk_size,
        snapshot_directory=tmp_path,
    ) as captured:
        assert captured.storage_kind == "anonymous_snapshot"
        observed_size = 0
        largest_output = 0
        for chunk in captured.iter_chunks(chunk_size=read_chunk_size):
            observed_size += len(chunk)
            largest_output = max(largest_output, len(chunk))
        assert observed_size == size_bytes
        assert largest_output <= read_chunk_size

    assert stream.remaining == 0
    assert max(stream.requests) <= capture_chunk_size
    assert stream.largest_response <= capture_chunk_size


def test_resolver_protocol_requires_only_managed_capture() -> None:
    class Resolver:
        def capture(
            self,
            request: OpaqueArtifactRequest,
        ) -> Any:
            return capture_streamed_opaque_file(
                io.BytesIO(b"payload"),
                request,
            )

    resolver = Resolver()
    assert isinstance(resolver, CapturedOpaqueArtifactResolver)
    request = _request(b"payload")
    with resolver.capture(request) as captured:
        assert b"".join(captured.iter_chunks()) == b"payload"


def test_bounded_reader_protocol_matches_the_runtime_capture_contract() -> None:
    class MinimalBoundedReader:
        def __init__(self, payload: bytes) -> None:
            self._stream = io.BytesIO(payload)

        def read(self, size: int) -> bytes:
            return self._stream.read(size)

    payload = b"minimal bounded reader"
    reader = MinimalBoundedReader(payload)
    assert isinstance(reader, BoundedBinaryReader)

    with capture_streamed_opaque_file(reader, _request(payload)) as captured:
        assert b"".join(captured.iter_chunks()) == payload


def test_stream_remains_caller_owned_after_capture() -> None:
    stream: BinaryIO = io.BytesIO(b"payload")
    with capture_streamed_opaque_file(stream, _request(b"payload")):
        pass

    assert not stream.closed
    assert stream.tell() == len(b"payload")
