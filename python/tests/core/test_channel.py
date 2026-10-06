# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Python Channel scalar and runtime tests."""

import gc
import os
import subprocess
import threading
import time

import control_payload
import pytest
from channel_test_utils import unique_channel

import holoscan as hs


# REQ: PY-CHANNEL-001
# REQ: PY-CHANNEL-009
def test_name_based_scalar_round_trip_and_ordered_eos():
    # Key setup: Attach a standalone publisher and subscriber without authoring a Graph.
    class IndexValue:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    name = unique_channel("python-channel-scalar")
    options = hs.channel.ChannelOptions(enable_event_wake=True)

    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)
        publisher = runtime.publisher(name, hs.payload.uint32, options=options)
        publisher.update_connections()
        subscriber.refresh_connections()

        # Key assertion: The registered scalar thunk preserves the exact value and metadata.
        assert subscriber.try_receive() is None
        assert publisher.has_subscribers
        assert subscriber.has_publishers
        assert publisher.emit(41, frame_id=IndexValue(7)) == 1
        sample = subscriber.receive(1.0)
        assert sample.data == 41
        assert sample.metadata.frame_id == 7
        assert sample.metadata.byte_size == 4

        with pytest.raises(
            hs.errors.InvalidArgumentError, match="PYTHON_CHANNEL_UNSIGNED_INTEGER_REQUIRED"
        ):
            publisher.emit(42, frame_id=True)
        for invalid_frame_id in (IndexValue(-1), IndexValue(1 << 64)):
            with pytest.raises(
                hs.errors.InvalidArgumentError,
                match="PYTHON_CHANNEL_UNSIGNED_INTEGER_OUT_OF_RANGE",
            ):
                publisher.emit(42, frame_id=invalid_frame_id)

        publisher.close()
        with pytest.raises(hs.errors.ClosedError):
            subscriber.receive(1.0)
        with pytest.raises(hs.errors.ClosedError):
            subscriber.try_receive()


# REQ: PY-CHANNEL-001
def test_cross_process_python_client_round_trips_with_cpp_channel_runtime():
    # Key setup: One independent C++ process owns the opposite endpoint in both directions.
    prefix = unique_channel("python-channel-cpp-peer")
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    peer = subprocess.Popen(
        [os.environ["HOLOSCAN_SCHEMA_PAYLOAD_PEER"], "--channel-client", prefix],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with hs.channel.Runtime() as runtime:
            scalar_subscriber = runtime.subscriber(
                f"{prefix}-scalar-from-cpp", hs.payload.uint32, options=options
            )
            scalar_publisher = runtime.publisher(
                f"{prefix}-scalar-to-cpp", hs.payload.uint32, options=options
            )
            schema_subscriber = runtime.subscriber(
                f"{prefix}-schema-from-cpp", control_payload.ControlPayload, options=options
            )
            schema_publisher = runtime.publisher(
                f"{prefix}-schema-to-cpp", control_payload.ControlPayload, options=options
            )
            scalar_publisher.wait_for_subscribers(5.0)
            schema_publisher.wait_for_subscribers(5.0)
            scalar_subscriber.refresh_connections()
            schema_subscriber.refresh_connections()

            assert scalar_subscriber.receive(5.0).data == 41
            assert schema_subscriber.receive(5.0).data == control_payload.ControlPayload(
                sequence=43, gain=2.5, labels=["cpp", "channel"]
            )
            assert scalar_publisher.emit(42, timeout=5.0) == 1
            assert (
                schema_publisher.emit(
                    control_payload.ControlPayload(
                        sequence=44, gain=3.5, labels=["python", "channel"]
                    ),
                    timeout=5.0,
                )
                == 1
            )

            stdout, stderr = peer.communicate(timeout=10)

            # Key assertion: Both payload families crossed real process boundaries both ways.
            assert peer.returncode == 0, stderr
            assert "standalone channel C++ peer complete" in stdout
            with pytest.raises(hs.errors.ClosedError):
                scalar_subscriber.receive(5.0)
            with pytest.raises(hs.errors.ClosedError):
                schema_subscriber.receive(5.0)
    finally:
        if peer.poll() is None:
            peer.kill()
            peer.communicate(timeout=5)


# REQ: PY-CHANNEL-003
def test_wait_releases_gil_and_cross_thread_cancel_keeps_subscriber_usable():
    # Key setup: Block one receive while another Python thread makes progress and then cancels it.
    name = unique_channel("python-channel-cancel")
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    cancel = hs.channel.CancelToken()
    progressed = threading.Event()

    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)

        def progress_and_cancel():
            progressed.set()
            time.sleep(0.05)
            cancel.cancel()

        thread = threading.Thread(target=progress_and_cancel)
        thread.start()
        with pytest.raises(hs.errors.HoloscanInterruptedError):
            subscriber.receive(cancel=cancel)
        thread.join(timeout=1)
        assert progressed.is_set()
        assert not thread.is_alive()

        # Key assertion: Cancellation is call-scoped and does not retire the subscriber.
        publisher = runtime.publisher(name, hs.payload.uint32, options=options)
        publisher.update_connections()
        subscriber.refresh_connections()
        assert publisher.emit(23) == 1
        assert subscriber.receive(1.0).data == 23


# REQ: PY-CHANNEL-003
def test_connection_snapshot_properties_release_gil_while_waiting_for_endpoint_locks():
    # Key setup: Make each property wait behind an endpoint call whose progress needs another
    # Python thread. Holding the GIL while waiting for the endpoint mutex would force the 2 s
    # deadline to expire.
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(
            unique_channel("python-channel-property-subscriber"),
            hs.payload.uint32,
            options=options,
        )
        cancel = hs.channel.CancelToken()
        waiter_started = threading.Event()
        waiter_result = []

        def blocked_receive():
            waiter_started.set()
            try:
                subscriber.receive(2.0, cancel=cancel)
            except hs.errors.HoloscanInterruptedError:
                waiter_result.append("cancelled")

        waiter = threading.Thread(target=blocked_receive)
        waiter.start()
        assert waiter_started.wait(timeout=1)
        time.sleep(0.05)
        canceller = threading.Thread(target=lambda: (time.sleep(0.05), cancel.cancel()))
        canceller.start()
        started = time.monotonic()
        assert not subscriber.has_publishers
        elapsed = time.monotonic() - started
        waiter.join(timeout=1)
        canceller.join(timeout=1)
        assert waiter_result == ["cancelled"]
        assert elapsed < 1.0

        publisher_name = unique_channel("python-channel-property-publisher")
        publisher = runtime.publisher(publisher_name, hs.payload.uint32, options=options)
        publisher_wait_started = threading.Event()
        publisher_wait_result = []
        peer = []

        def wait_for_peer():
            publisher_wait_started.set()
            try:
                publisher.wait_for_subscribers(2.0)
                publisher_wait_result.append("connected")
            except BaseException as error:  # preserve a useful assertion instead of a thread leak
                publisher_wait_result.append(error)

        def attach_peer():
            time.sleep(0.05)
            peer.append(runtime.subscriber(publisher_name, hs.payload.uint32, options=options))

        publisher_waiter = threading.Thread(target=wait_for_peer)
        publisher_waiter.start()
        assert publisher_wait_started.wait(timeout=1)
        time.sleep(0.05)
        connector = threading.Thread(target=attach_peer)
        connector.start()
        started = time.monotonic()
        assert publisher.has_subscribers
        elapsed = time.monotonic() - started
        publisher_waiter.join(timeout=1)
        connector.join(timeout=1)

        # Key assertion: Both advisory properties release the GIL before blocking on their
        # operation mutex, allowing cancellation and peer attachment to run promptly.
        assert publisher_wait_result == ["connected"]
        assert len(peer) == 1
        assert elapsed < 1.0


# REQ: PY-CHANNEL-003
def test_second_concurrent_wait_is_rejected_without_disturbing_first_waiter():
    # Key setup: Keep one wait active while the calling thread attempts another wait.
    name = unique_channel("python-channel-one-waiter")
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    cancel = hs.channel.CancelToken()
    started = threading.Event()
    result = []

    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)

        def first_waiter():
            started.set()
            try:
                subscriber.wait(2.0, cancel=cancel)
            except hs.errors.HoloscanInterruptedError:
                result.append("cancelled")

        thread = threading.Thread(target=first_waiter)
        thread.start()
        assert started.wait(timeout=1)
        time.sleep(0.05)
        with pytest.raises(hs.errors.InvalidArgumentError, match="WAIT_ALREADY_ACTIVE"):
            subscriber.receive(0.1)
        cancel.cancel()
        thread.join(timeout=1)

        # Key assertion: Rejecting the second waiter leaves the admitted first waiter cancellable.
        assert result == ["cancelled"]
        assert not thread.is_alive()


# REQ: PY-CHANNEL-004
def test_runtime_context_retires_endpoints_and_emit_validates_exact_scalar_range():
    # Key setup: Retain endpoint objects after leaving their owning Runtime context.
    name = unique_channel("python-channel-context")
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, hs.payload.uint8, options=options)
        publisher = runtime.publisher(name, hs.payload.uint8, options=options)
        publisher.update_connections()
        subscriber.refresh_connections()
        with pytest.raises(hs.errors.InvalidArgumentError, match="VALUE_OUT_OF_RANGE"):
            publisher.emit(256)

    # Key assertion: Context exit closes live endpoints deterministically.
    with pytest.raises(hs.errors.ClosedError):
        publisher.emit(1)
    with pytest.raises(hs.errors.ClosedError):
        subscriber.try_receive()

    runtime = hs.channel.Runtime()
    publisher_without_peer = runtime.publisher(
        unique_channel("python-channel-orphan"), hs.payload.uint8
    )
    runtime.close()
    with pytest.raises(hs.errors.ClosedError, match="PYTHON_CHANNEL_RUNTIME_CLOSED"):
        publisher_without_peer.emit(1)


# REQ: PY-CHANNEL-004
def test_runtime_publishes_eos_only_for_explicit_or_normal_context_close():
    # Key setup: Observe three independently owned publishers from a longer-lived Runtime.
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    with hs.channel.Runtime() as observer:
        normal_name = unique_channel("python-channel-normal-context")
        normal_subscriber = observer.subscriber(normal_name, hs.payload.uint32, options=options)
        with hs.channel.Runtime() as producer:
            publisher = producer.publisher(normal_name, hs.payload.uint32, options=options)
            publisher.wait_for_subscribers(1.0)
            normal_subscriber.refresh_connections()
            assert publisher.emit(1) == 1
            assert normal_subscriber.receive(1.0).data == 1
        with pytest.raises(hs.errors.ClosedError):
            normal_subscriber.receive(1.0)

        exceptional_name = unique_channel("python-channel-exception-context")
        exceptional_subscriber = observer.subscriber(
            exceptional_name, hs.payload.uint32, options=options
        )

        def abandon_publisher():
            with hs.channel.Runtime() as producer:
                publisher = producer.publisher(exceptional_name, hs.payload.uint32, options=options)
                publisher.wait_for_subscribers(1.0)
                exceptional_subscriber.refresh_connections()
                assert publisher.emit(2) == 1
                assert exceptional_subscriber.receive(1.0).data == 2
                raise RuntimeError("abandon producer")

        with pytest.raises(RuntimeError, match="abandon producer"):
            abandon_publisher()
        with pytest.raises(hs.errors.HoloscanTimeoutError):
            exceptional_subscriber.receive(0.1)

        destructor_name = unique_channel("python-channel-runtime-destructor")
        destructor_subscriber = observer.subscriber(
            destructor_name, hs.payload.uint32, options=options
        )
        producer = hs.channel.Runtime()
        publisher = producer.publisher(destructor_name, hs.payload.uint32, options=options)
        publisher.wait_for_subscribers(1.0)
        destructor_subscriber.refresh_connections()
        assert publisher.emit(3) == 1
        assert destructor_subscriber.receive(1.0).data == 3
        del producer
        gc.collect()

        # Key assertion: Exceptional exit and Runtime destruction retire publishers locally;
        # neither fabricates the explicit ordered EOS reserved for a successful close.
        with pytest.raises(hs.errors.HoloscanTimeoutError):
            destructor_subscriber.receive(0.1)


# REQ: PY-CHANNEL-005
def test_emit_timeout_retries_only_pre_submit_outcomes():
    # Key setup: Publish without a peer so the backend reports a retryable pre-submit outcome.
    name = unique_channel("python-channel-emit-retry")
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    with hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(name, hs.payload.uint32, options=options)
        started = time.monotonic()
        with pytest.raises(hs.errors.HoloscanTimeoutError):
            publisher.emit(1, timeout=0.05)
        assert 0.04 <= time.monotonic() - started < 0.5

        subscriber = runtime.subscriber(name, hs.payload.uint32, options=options)
        publisher.update_connections()
        subscriber.refresh_connections()
        assert publisher.emit(2, timeout=0.5) == 1
        assert subscriber.receive(0.5).data == 2

        publisher.close()
        started = time.monotonic()
        with pytest.raises(hs.errors.ClosedError):
            publisher.emit(3, timeout=0.5)

        # Key assertion: A terminal error is returned immediately rather than retried to timeout.
        assert time.monotonic() - started < 0.1


# REQ: PY-CHANNEL-003
def test_name_based_wait_without_event_wake_reports_not_supported():
    # Key setup: Preserve the C++ default that does not enable a kernel event lane.
    name = unique_channel("python-channel-no-event-wake")
    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, hs.payload.uint32)

        # Key assertion: Blocking wait fails explicitly while nonblocking receive remains valid.
        assert subscriber.try_receive() is None
        with pytest.raises(hs.errors.NotSupportedError):
            subscriber.receive(0.1)
        with pytest.raises(hs.errors.NotSupportedError):
            subscriber.readiness()
