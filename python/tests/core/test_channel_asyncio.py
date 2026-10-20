# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Python Channel readiness and asyncio tests."""

import asyncio
import select
import threading

import pytest
from channel_test_utils import unique_channel

import holoscan as hs

_ASYNC_TIMEOUT_DIAGNOSTIC = "PYTHON_CHANNEL_ASYNC_RECEIVE_TIMEOUT"


# REQ: PY-CHANNEL-006
def test_readiness_handle_is_pollable_and_claims_the_wait_owner():
    # Key setup: Claim the descriptor before publishing a scalar sample.
    name = unique_channel("python-channel-readiness")
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)
        publisher = runtime.publisher(name, hs.payload.uint32, options=options)
        publisher.update_connections()
        subscriber.refresh_connections()

        with subscriber.readiness() as readiness:
            with pytest.raises(hs.errors.InvalidArgumentError, match="WAIT_ALREADY_ACTIVE"):
                subscriber.receive(0.1)
            assert publisher.emit(29) == 1
            readable, _, _ = select.select([readiness], [], [], 1.0)
            assert readable == [readiness]
            readiness.acknowledge()
            assert subscriber.try_receive().data == 29

        # Key assertion: Detaching releases the single wait-owner slot immediately.
        assert publisher.emit(31) == 1
        assert subscriber.receive(1.0).data == 31
        with pytest.raises(hs.errors.ClosedError, match="READINESS_DETACHED"):
            readiness.fileno()


# REQ: PY-CHANNEL-006
def test_asyncio_readiness_receives_fifo_without_helper_threads_and_surfaces_eos():
    async def scenario():
        name = unique_channel("python-channel-aio-fifo")
        options = hs.channel.ChannelOptions(enable_event_wake=True)
        with hs.channel.Runtime() as runtime:
            subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)
            publisher = runtime.publisher(name, hs.payload.uint32, options=options)
            publisher.update_connections()
            subscriber.refresh_connections()
            aio_subscriber = hs.channel.aio.Subscriber(subscriber)
            thread_count = threading.active_count()

            first_wait = asyncio.create_task(aio_subscriber.receive(1.0))
            second_wait = asyncio.create_task(aio_subscriber.receive(1.0))
            await asyncio.sleep(0)
            with pytest.raises(hs.errors.InvalidArgumentError, match="WAIT_ALREADY_ACTIVE"):
                subscriber.wait(0.0)
            assert publisher.emit(5) == 1
            assert publisher.emit(7) == 1
            first_sample, second_sample = await asyncio.gather(first_wait, second_wait)

            # Key assertion: Two waiters on one adapter retain FIFO order without an executor.
            assert (first_sample.data, second_sample.data) == (5, 7)
            assert threading.active_count() == thread_count
            with pytest.raises(hs.errors.InvalidArgumentError, match="WAIT_ALREADY_ACTIVE"):
                subscriber.wait(0.0)

            # The readiness registration remains attached across an idle interval and another
            # successful receive instead of rebuilding the native handle for every sample.
            assert publisher.emit(9) == 1
            assert (await aio_subscriber.receive(1.0)).data == 9

            publisher.close()
            with pytest.raises(hs.errors.ClosedError):
                await aio_subscriber.receive(1.0)
            aio_subscriber.detach()

    asyncio.run(scenario())


# REQ: PY-CHANNEL-006
def test_asyncio_idle_reader_preserves_native_queue_bounds_and_fifo():
    async def scenario():
        name = unique_channel("python-channel-aio-idle")
        options = hs.channel.ChannelOptions(
            enable_event_wake=True,
            subscriber_queue_depth=8,
        )
        with hs.channel.Runtime() as runtime:
            subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)
            publisher = runtime.publisher(name, hs.payload.uint32, options=options)
            publisher.update_connections()
            subscriber.refresh_connections()
            aio_subscriber = hs.channel.aio.Subscriber(subscriber)

            initial = asyncio.create_task(aio_subscriber.receive(1.0))
            await asyncio.sleep(0)
            assert publisher.emit(1) == 1
            assert (await initial).data == 1

            for value in range(2, 10):
                assert publisher.emit(value) == 1
            await asyncio.sleep(0.05)

            # Key assertion: With no admitted coroutine waiter, the adapter leaves samples in the
            # finite native queue instead of draining them into unbounded Python-owned storage.
            assert subscriber.has_samples
            receives = [asyncio.create_task(aio_subscriber.receive(1.0)) for _ in range(8)]
            samples = await asyncio.gather(*receives)
            assert [sample.data for sample in samples] == list(range(2, 10))
            publisher.close()
            with pytest.raises(hs.errors.ClosedError):
                await aio_subscriber.receive(1.0)
            aio_subscriber.detach()

    asyncio.run(scenario())


# REQ: PY-CHANNEL-006
@pytest.mark.parametrize(
    "invalid_timeout",
    [-1, True, "invalid", float("nan")],
    ids=["negative", "boolean", "string", "nan"],
)
@pytest.mark.parametrize("resumed", [False, True], ids=["initial", "resumed"])
def test_invalid_receive_timeout_preserves_queued_samples(invalid_timeout, resumed):
    async def scenario():
        name = unique_channel("python-channel-aio-invalid-timeout")
        options = hs.channel.ChannelOptions(enable_event_wake=True, subscriber_queue_depth=3)
        with hs.channel.Runtime() as runtime:
            subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)
            publisher = runtime.publisher(name, hs.payload.uint32, options=options)
            publisher.update_connections()
            subscriber.refresh_connections()
            aio_subscriber = hs.channel.aio.Subscriber(subscriber)

            if resumed:
                assert publisher.emit(0) == 1
                assert (await aio_subscriber.receive(1.0)).data == 0

            assert publisher.emit(11) == 1
            assert publisher.emit(22) == 1
            with pytest.raises((TypeError, ValueError)):
                await aio_subscriber.receive(invalid_timeout)

            # Key assertion: Rejecting the timeout happens before the adapter admits a waiter or
            # drains either queued sample, both initially and after reusing its readiness handle.
            assert (await aio_subscriber.receive(1.0)).data == 11
            assert (await aio_subscriber.receive(1.0)).data == 22
            aio_subscriber.detach()

    asyncio.run(scenario())


# REQ: PY-CHANNEL-006
def test_asyncio_completed_sample_wins_timeout_and_cancellation_boundary(monkeypatch):
    async def scenario():
        name = unique_channel("python-channel-aio-boundary")
        options = hs.channel.ChannelOptions(enable_event_wake=True)
        with hs.channel.Runtime() as runtime:
            subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)
            publisher = runtime.publisher(name, hs.payload.uint32, options=options)
            publisher.update_connections()
            subscriber.refresh_connections()
            aio_subscriber = hs.channel.aio.Subscriber(subscriber)
            samples = [object(), object()]

            async def complete_then_timeout(awaitable, timeout):
                del awaitable, timeout
                aio_subscriber._pending[0].set_result(samples[0])
                raise TimeoutError

            monkeypatch.setattr(asyncio, "wait_for", complete_then_timeout)
            assert await aio_subscriber.receive(0.1) is samples[0]

            async def complete_then_cancel(awaitable, timeout):
                del awaitable, timeout
                aio_subscriber._pending[0].set_result(samples[1])
                raise asyncio.CancelledError

            monkeypatch.setattr(asyncio, "wait_for", complete_then_cancel)

            # Key assertion: A result already delivered to the waiter is never discarded merely
            # because timeout or cancellation is observed at the same scheduling boundary.
            assert await aio_subscriber.receive(0.1) is samples[1]
            aio_subscriber.detach()

    asyncio.run(scenario())


# REQ: PY-CHANNEL-006
def test_asyncio_executor_fallback_uses_bounded_infinite_wait_slices():
    class FallbackSubscriber:
        def __init__(self):
            self.waits = []

        def readiness(self):
            raise hs.errors.NotSupportedError("readiness unavailable")

        def receive(self, timeout, *, cancel):
            del cancel
            self.waits.append(timeout)
            if len(self.waits) < 3:
                hs.channel.aio._raise_receive_timeout_error()
            return "sample"

    async def scenario():
        subscriber = FallbackSubscriber()
        adapter = hs.channel.aio.Subscriber(subscriber)
        assert await adapter.receive() == "sample"

        # Key assertion: An unbounded coroutine wait is implemented as cancellable finite native
        # waits, not one effectively uninterruptible 24-hour executor call.
        assert subscriber.waits == [1.0, 1.0, 1.0]

    asyncio.run(scenario())


def test_asyncio_timeout_uses_native_structured_error():
    with pytest.raises(hs.errors.HoloscanTimeoutError) as caught:
        hs.channel.aio._raise_receive_timeout_error()

    assert caught.value.code == hs.errors.ErrorCode.TIMEOUT
    assert caught.value.token == _ASYNC_TIMEOUT_DIAGNOSTIC


# REQ: PY-CHANNEL-006
def test_asyncio_fallback_completed_sample_wins_cancellation(monkeypatch):
    class FallbackSubscriber:
        def readiness(self):
            raise hs.errors.NotSupportedError("readiness unavailable")

        def receive(self, timeout, *, cancel):
            del timeout, cancel
            raise AssertionError("the controlled executor seam supplies the result")

    async def scenario():
        loop = asyncio.get_running_loop()
        workers = [loop.create_future(), loop.create_future()]
        issued = []

        def run_in_executor(executor, function):
            del executor, function
            worker = workers[len(issued)]
            issued.append(worker)
            return worker

        monkeypatch.setattr(loop, "run_in_executor", run_in_executor)
        adapter = hs.channel.aio.Subscriber(FallbackSubscriber())
        sample = object()
        receive = asyncio.create_task(adapter.receive())
        await asyncio.sleep(0)
        workers[0].set_result(sample)
        receive.cancel()

        # Key assertion: Once the executor has consumed a sample, cancellation cannot discard it.
        assert await receive is sample

        continuation = asyncio.create_task(adapter.receive())
        await asyncio.sleep(0)
        continuation.cancel()
        await asyncio.sleep(0)
        continuation.cancel()
        await asyncio.sleep(0)
        next_sample = object()
        workers[1].set_result(next_sample)

        # Repeated cancellation is also deferred until the worker establishes whether it consumed
        # a sample, so the next receive cannot silently skip that value.
        assert await continuation is next_sample

    asyncio.run(scenario())


# REQ: PY-CHANNEL-006
def test_asyncio_task_cancellation_detaches_and_async_emit_uses_nonblocking_retries():
    async def scenario():
        name = unique_channel("python-channel-aio-cancel")
        options = hs.channel.ChannelOptions(enable_event_wake=True)
        with hs.channel.Runtime() as runtime:
            subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)
            publisher = runtime.publisher(name, hs.payload.uint32, options=options)
            publisher.update_connections()
            subscriber.refresh_connections()
            aio_subscriber = hs.channel.aio.Subscriber(subscriber)
            aio_publisher = hs.channel.aio.Publisher(publisher)

            cancelled = asyncio.create_task(aio_subscriber.receive())
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled

            # Key assertion: Cancellation detached readiness, so another wait may claim ownership.
            emitted = asyncio.create_task(aio_publisher.emit(43, timeout=1.0))
            assert await emitted == 1
            assert subscriber.receive(1.0).data == 43

            retry_name = unique_channel("python-channel-aio-emit-retry")
            retry_publisher = runtime.publisher(retry_name, hs.payload.uint32, options=options)
            retry_aio_publisher = hs.channel.aio.Publisher(retry_publisher)
            retry = asyncio.create_task(retry_aio_publisher.emit(47, timeout=1.0))
            await asyncio.sleep(0.02)
            assert not retry.done()
            retry_subscriber = runtime.subscriber(retry_name, hs.payload.uint32, options=options)
            retry_publisher.update_connections()
            retry_subscriber.refresh_connections()

            # Key assertion: Retry cadence yields to the event loop until a peer attaches.
            assert await retry == 1
            assert retry_subscriber.receive(1.0).data == 47

    asyncio.run(scenario())
