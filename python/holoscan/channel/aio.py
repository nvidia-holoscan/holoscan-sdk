# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ``asyncio`` adapters for standalone Channel endpoints."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, NoReturn

from .._core import _ChannelTensorLoan as _TensorLoan
from .._core import _raise_channel_async_receive_timeout
from ..errors import (
    BackpressuredError,
    HoloscanInterruptedError,
    HoloscanTimeoutError,
    NotReadyError,
    NotSupportedError,
)
from ..time import Duration, TimePoint
from . import CancelToken, ReadinessHandle
from . import Publisher as _SyncPublisher
from . import Subscriber as _SyncSubscriber

if TYPE_CHECKING:
    from .. import Sample

_RETRY_INTERVAL_SECONDS = 0.005
_FALLBACK_WAIT_SECONDS = 1.0


def _timeout_seconds(value: float | timedelta | Duration) -> float:
    if isinstance(value, Duration):
        seconds = value.nanoseconds / 1_000_000_000
    elif isinstance(value, timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        seconds = float(value)
    else:
        raise TypeError("timeout must be seconds, datetime.timedelta, or holoscan.time.Duration")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("timeout must be finite and nonnegative")
    return seconds


def _raise_receive_timeout_error() -> NoReturn:
    """Raise a timeout through the native structured-error translator."""
    _raise_channel_async_receive_timeout()
    raise AssertionError("native timeout translator returned without raising")


class Subscriber:
    """Adapt one synchronous subscriber to an ``asyncio`` event loop."""

    def __init__(self, subscriber: _SyncSubscriber) -> None:
        self._subscriber = subscriber
        self._loop: asyncio.AbstractEventLoop | None = None
        self._readiness: ReadinessHandle | None = None
        self._reader_attached = False
        self._pending: deque[asyncio.Future[Sample[Any]]] = deque()
        self._terminal: Exception | None = None
        self._fallback: bool | None = None
        self._fallback_lock = asyncio.Lock()

    async def receive(self, timeout: float | timedelta | Duration | None = None) -> Sample[Any]:
        """Receive one sample without blocking the event loop."""
        timeout_seconds = None if timeout is None else _timeout_seconds(timeout)
        loop = asyncio.get_running_loop()
        self._bind_loop(loop)
        if self._terminal is not None:
            raise self._terminal
        if self._fallback is True:
            return await self._receive_fallback(timeout_seconds)

        future: asyncio.Future[Sample[Any]] = loop.create_future()
        self._pending.append(future)
        detach_after_cancellation = False
        try:
            if self._readiness is None:
                try:
                    self._attach_reader()
                except NotSupportedError:
                    self._fallback = True
                    self._pending.remove(future)
                    future.cancel()
                    return await self._receive_fallback(timeout_seconds)
            else:
                self._resume_reader()
            # A wake may already have been acknowledged before the reader was paused.  Prime one
            # demand-bounded drain so queued native work cannot depend on a future descriptor edge.
            self._on_readable()
            if timeout_seconds is None:
                return await future
            try:
                return await asyncio.wait_for(asyncio.shield(future), timeout_seconds)
            except TimeoutError as error:
                if future.done() and not future.cancelled():
                    return future.result()
                try:
                    _raise_receive_timeout_error()
                except HoloscanTimeoutError as timeout_error:
                    raise timeout_error from error
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                return future.result()
            detach_after_cancellation = True
            raise
        finally:
            if future in self._pending:
                self._pending.remove(future)
                future.cancel()
            if not self._pending:
                self._pause_reader()
            if detach_after_cancellation and not self._pending:
                self._detach_reader()

    def detach(self) -> None:
        """Detach this adapter's event-loop reader and readiness handle."""
        if self._pending:
            raise RuntimeError("cannot detach an asyncio Subscriber with pending receives")
        self._detach_reader()

    def _bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("a channel.aio.Subscriber cannot move between event loops")

    def _attach_reader(self) -> None:
        assert self._loop is not None
        handle = self._subscriber.readiness()
        self._readiness = handle
        try:
            self._resume_reader()
        except BaseException:
            self._readiness = None
            handle.detach()
            raise
        self._fallback = False

    def _resume_reader(self) -> None:
        handle = self._readiness
        if handle is None or self._reader_attached:
            return
        assert self._loop is not None
        self._loop.add_reader(handle.fileno(), self._on_readable)
        self._reader_attached = True

    def _discard_completed_waiters(self) -> None:
        while self._pending and self._pending[0].done():
            self._pending.popleft()

    def _on_readable(self) -> None:
        handle = self._readiness
        if handle is None:
            return
        try:
            handle.acknowledge()
            self._discard_completed_waiters()
            while self._pending:
                sample = self._subscriber.try_receive()
                if sample is None:
                    break
                self._pending.popleft().set_result(sample)
                self._discard_completed_waiters()
            if not self._pending:
                self._pause_reader()
        except Exception as error:
            self._terminal = error
            while self._pending:
                future = self._pending.popleft()
                if not future.done():
                    future.set_exception(error)
            self._detach_reader()
        except BaseException:
            self._detach_reader()
            raise

    def _detach_reader(self) -> None:
        handle = self._readiness
        if handle is None:
            return
        self._pause_reader()
        handle.detach()
        self._readiness = None

    def _pause_reader(self) -> None:
        handle = self._readiness
        if handle is None or not self._reader_attached:
            return
        if self._loop is not None:
            self._loop.remove_reader(handle.fileno())
        self._reader_attached = False

    async def _receive_fallback(self, timeout: float | None) -> Sample[Any]:
        assert self._loop is not None
        async with self._fallback_lock:
            while True:
                cancel = CancelToken()
                wait_timeout = _FALLBACK_WAIT_SECONDS if timeout is None else timeout
                worker = self._loop.run_in_executor(
                    None, partial(self._subscriber.receive, wait_timeout, cancel=cancel)
                )
                try:
                    return await asyncio.shield(worker)
                except HoloscanTimeoutError:
                    if timeout is not None:
                        raise
                except asyncio.CancelledError as cancelled:
                    cancel.cancel()
                    while True:
                        try:
                            return await asyncio.shield(worker)
                        except asyncio.CancelledError:
                            # Repeated task cancellation must not abandon the native worker.  Its
                            # successful receive still owns a sample that cannot be replayed.
                            if worker.cancelled():
                                raise cancelled from None
                            cancel.cancel()
                            continue
                        except HoloscanInterruptedError:
                            raise cancelled from None


class Publisher:
    """Adapt one synchronous publisher with nonblocking retry sleeps."""

    def __init__(self, publisher: _SyncPublisher) -> None:
        self._publisher = publisher

    async def emit(
        self,
        value: object,
        *,
        capture_time: TimePoint | None = None,
        frame_id: int | None = None,
        flags: object | None = None,
        timeout: float | timedelta | Duration | None = None,
    ) -> int:
        """Emit a value and return the number of subscribers that accepted it.

        Without ``timeout``, this makes one publication attempt. With ``timeout``, it retries
        only pre-submit ``NotReadyError`` and ``BackpressuredError`` outcomes against one absolute
        deadline. It never retries a submitted, partially submitted, or indeterminate outcome.
        """
        arguments = {
            "capture_time": capture_time,
            "frame_id": frame_id,
            "flags": flags,
        }
        if timeout is not None and isinstance(value, _TensorLoan):
            # The synchronous endpoint owns the canonical error token. A loan is a one-shot
            # authority consumed by its first publication attempt, so it cannot be retried.
            return self._publisher.emit(value, timeout=timeout, **arguments)
        if timeout is None:
            return self._publisher.emit(value, **arguments)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _timeout_seconds(timeout)
        while True:
            try:
                return self._publisher.emit(value, **arguments)
            except (NotReadyError, BackpressuredError):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return self._publisher.emit(value, timeout=0.0, **arguments)
                await asyncio.sleep(min(_RETRY_INTERVAL_SECONDS, remaining))


__all__ = ["Publisher", "Subscriber"]
