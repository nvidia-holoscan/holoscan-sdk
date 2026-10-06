# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Standalone Channel clients for processes that do not host a Graph."""

import enum

from datetime import timedelta as _timedelta
from typing import Any, Self, overload

from holoscan import QueuePolicy as QueuePolicy
from holoscan import Sample as Sample
from holoscan import SampleFlags as _SampleFlags
from holoscan import SampleMetadata as SampleMetadata
from holoscan.payload import PayloadType as _payload_PayloadType
from holoscan.tensor import DLDataType as _tensor_DLDataType
from holoscan.tensor import MemoryKind as _tensor_MemoryKind
from holoscan.time import Duration as _time_Duration
from holoscan.time import TimePoint as _time_TimePoint
from holoscan.topics import EndpointCatalog as _topics_EndpointCatalog

from . import aio as aio

type _Timeout = float | _timedelta | _time_Duration


class CancelToken:
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    @property
    def cancelled(self) -> bool: ...


class ReadinessHandle:
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def fileno(self) -> int: ...
    def acknowledge(self) -> None: ...
    def detach(self) -> None: ...


class TensorTransferMode(enum.Enum):
    SAME_PROCESS_SIDECAR = ...
    BUFFER_HANDLE = ...


class _TensorWritePhase:
    def __enter__(self) -> TensorLoan: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...


class TensorLoan:
    """One-shot loan consumed by every emit attempt, including a failed attempt."""
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> _tensor_DLDataType: ...
    @property
    def strides(self) -> tuple[int, ...] | None: ...
    @property
    def memory_kind(self) -> _tensor_MemoryKind: ...
    @property
    def nbytes(self) -> int: ...
    def __array__(self, dtype: object | None = None, *, copy: bool | None = None) -> object: ...
    def __dlpack__(
        self,
        *,
        stream: object | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object:
        """Export a writable view; CUDA consumers must use the write phase's stream."""
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def write(self, *, stream: object | None = None) -> _TensorWritePhase:
        """Enter the writable phase.

        CUDA rejects thread-relative ``cudaStreamPerThread`` (``stream=2``).
        """
    def cancel(self) -> None: ...


class ChannelOptions:
    def __init__(
        self,
        *,
        queue_policy: QueuePolicy = ...,
        max_subscriber_queue_depth: int = 0,
        subscriber_queue_depth: int = 8,
        max_subscriber_borrowed_samples: int = 0,
        max_borrowed_samples: int = 0,
        publisher_history_depth: int = 0,
        max_loaned_samples: int = 0,
        expected_recipient_count: int = 1,
        enable_event_wake: bool = False,
        replacement_attribution_required: bool = False,
        tensor_transfer_mode: TensorTransferMode = ...,
    ) -> None: ...
    queue_policy: QueuePolicy
    max_subscriber_queue_depth: int
    subscriber_queue_depth: int
    max_subscriber_borrowed_samples: int
    max_borrowed_samples: int
    publisher_history_depth: int
    max_loaned_samples: int
    expected_recipient_count: int
    enable_event_wake: bool
    replacement_attribution_required: bool
    tensor_transfer_mode: TensorTransferMode


class ClockHealthAssessment(enum.Enum):
    HEALTHY = ...
    DEGRADED = ...
    UNHEALTHY = ...


class ClockHealthReason(enum.Enum):
    NONE = ...
    SYNCHRONIZATION_LOCK_LOST = ...
    PHASE_ERROR_LIMIT_EXCEEDED = ...
    RATE_ERROR_LIMIT_EXCEEDED = ...
    CALIBRATION_EXPIRED = ...
    CLOCK_STALLED = ...
    TIMESTAMP_REGRESSION = ...
    TIMESTAMP_SOURCE_FAULT = ...
    MONITOR_FAULT = ...
    UNSPECIFIED = ...


class ClockHealthReport:
    def __init__(
        self,
        assessment: ClockHealthAssessment,
        reason: ClockHealthReason = ...,
    ) -> None: ...
    assessment: ClockHealthAssessment
    reason: ClockHealthReason
    @staticmethod
    def healthy() -> ClockHealthReport: ...
    @staticmethod
    def degraded(reason: ClockHealthReason) -> ClockHealthReport: ...
    @staticmethod
    def unhealthy(reason: ClockHealthReason) -> ClockHealthReport: ...


class Publisher:
    @property
    def payload_type(self) -> _payload_PayloadType: ...
    @property
    def has_subscribers(self) -> bool: ...
    def loan(
        self,
        shape: list[int] | tuple[int, ...],
        dtype: object,
        *,
        strides: list[int] | tuple[int, ...] | None = None,
    ) -> TensorLoan:
        """Borrow Tensor backing; scalar publishers reject this runtime-only operation."""
    def emit(
        self,
        value: object,
        *,
        capture_time: _time_TimePoint | None = None,
        frame_id: int | None = None,
        flags: _SampleFlags | None = None,
        timeout: _Timeout | None = None,
    ) -> int: ...
    def emit_captured(
        self,
        value: object,
        capture_timestamp_ns: int,
        *,
        frame_id: int | None = None,
        flags: _SampleFlags | None = None,
    ) -> int: ...
    def make_capture_time(self, timestamp_ns: int) -> _time_TimePoint: ...
    def report_clock_health(self, report: ClockHealthReport) -> None: ...
    def update_connections(self) -> None: ...
    def wait_for_subscribers(self, timeout: _Timeout | None = None) -> None: ...
    def close(self) -> None:
        """Publish ordered EOS; destruction and process exit do not imply EOS."""


class Subscriber:
    @property
    def payload_type(self) -> _payload_PayloadType: ...
    @property
    def has_samples(self) -> bool: ...
    @property
    def has_publishers(self) -> bool: ...
    def try_receive(self) -> Sample[Any] | None: ...
    def receive(
        self, timeout: _Timeout | None = None, *, cancel: CancelToken | None = None
    ) -> Sample[Any]: ...
    def wait(self, timeout: _Timeout | None = None, *, cancel: CancelToken | None = None) -> None: ...
    def readiness(self) -> ReadinessHandle: ...
    def refresh_connections(self) -> None: ...
    def close(self) -> None: ...


class Runtime:
    """Own endpoints; normal close publishes EOS, while abnormal exit abandons them."""
    def __init__(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def close(self) -> None: ...
    @overload
    def publisher(
        self,
        catalog: _topics_EndpointCatalog,
        name: str,
        *,
        timeout: _Timeout | None = None,
        payload_type: _payload_PayloadType | type[object] | None = None,
    ) -> Publisher: ...
    @overload
    def publisher(
        self,
        channel_name: str,
        payload_type: _payload_PayloadType | type[object],
        *,
        options: ChannelOptions | None = None,
    ) -> Publisher: ...
    @overload
    def subscriber(
        self,
        catalog: _topics_EndpointCatalog,
        name: str,
        *,
        timeout: _Timeout | None = None,
        payload_type: _payload_PayloadType | type[object] | None = None,
    ) -> Subscriber: ...
    @overload
    def subscriber(
        self,
        channel_name: str,
        payload_type: _payload_PayloadType | type[object],
        *,
        options: ChannelOptions | None = None,
    ) -> Subscriber: ...


__all__: list[str] = [
    "Runtime",
    "Publisher",
    "Subscriber",
    "CancelToken",
    "ReadinessHandle",
    "TensorLoan",
    "ChannelOptions",
    "QueuePolicy",
    "TensorTransferMode",
    "ClockHealthAssessment",
    "ClockHealthReason",
    "ClockHealthReport",
    "Sample",
    "SampleMetadata",
    "aio",
]
