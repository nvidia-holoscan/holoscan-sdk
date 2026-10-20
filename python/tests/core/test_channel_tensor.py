# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Python Channel Tensor client tests."""

import asyncio
import ctypes
import gc
import subprocess
import sys
import threading

import holoscan_test_support  # noqa: F401
import pytest
import stereo_payload
from channel_test_utils import unique_channel

import holoscan as hs


def host_u8_boundary(*, shape=(2, 4), copy_if_needed=True):
    """Build the bounded host Tensor contract used by Channel client tests."""
    arguments = dict(
        representation=hs.tensor.TensorRepresentation(
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, 8, 1),
            rank=len(shape),
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
        ),
        bounds=hs.tensor.TensorBounds(max_shape=shape),
    )
    if copy_if_needed:
        arguments["transfer"] = hs.tensor.TensorTransferPolicy(
            foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
        )
    return hs.tensor.TensorBoundaryContract(**arguments)


class TensorRelay(hs.Operator):
    """Relay one bounded host Tensor without replacing its backing."""

    inp = hs.Input("in", hs.payload.Tensor)
    out = hs.Output("out", hs.payload.Tensor)

    def setup(self, spec):
        tensor_input = hs.tensor.InputSpec(
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype="uint8",
            rank=2,
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
            bounds=hs.tensor.TensorBounds(max_shape=(2, 4)),
        )
        tensor_output = hs.tensor.OutputSpec(
            storage="pass_through",
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype="uint8",
            rank=2,
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
            bounds=hs.tensor.TensorBounds(max_shape=(2, 4)),
        )
        spec.input(self.inp).expects_tensor(tensor_input)
        spec.output(self.out).produces_tensor(tensor_output)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        self.out.forward(self.inp.receive())


def stereo_tensor_fields():
    """Build the two generated descriptor-field contracts used by client tests."""
    contract = host_u8_boundary(shape=(2, 2))
    return {"left": contract, "right": contract}


class StereoRelay(hs.Operator):
    """Relay one generated two-Tensor schema through bounded output pools."""

    inp = hs.Input("in", stereo_payload.StereoPayload)
    out = hs.Output("out", stereo_payload.StereoPayload)

    def setup(self, spec):
        spec.input(self.inp)
        output = spec.output(self.out)
        tensor_spec = hs.tensor.OutputSpec(
            storage="runtime_pool",
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype="uint8",
            rank=2,
            transfer=hs.tensor.TensorTransferPolicy(
                foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
            ),
        )
        output.produces_tensor(tensor_spec, hs.tensor.byte_span(4), field="left")
        output.produces_tensor(tensor_spec, hs.tensor.byte_span(4), field="right")

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        sample = self.inp.receive()
        sample.data.exposure_us += 1
        self.out.forward(sample, sample.data)


# REQ: PY-CHANNEL-007
# REQ: PY-CHANNEL-009
# REQ: TENSOR-PY-018
def test_catalog_tensor_loan_copy_receive_and_contract_view():
    # Key setup: Relay a bounded host Tensor between two catalog-attached Channel clients.
    np = pytest.importorskip("numpy")
    input_service = unique_channel("python-channel-tensor-in")
    output_service = unique_channel("python-channel-tensor-out")
    input_endpoint = f"{input_service}-publisher"
    output_endpoint = f"{output_service}-subscriber"
    graph = hs.Graph("channel-client-tensor-catalog", default_clock=hs.RealtimeClock)
    imported = graph.import_topic(
        f"holomq://{input_service}",
        hs.payload.TensorType,
        endpoint_name=input_endpoint,
        bounds=hs.topics.EndpointBounds(
            publisher=hs.topics.PublisherEndpointBounds(max_loaned_samples=1)
        ),
        tensor=host_u8_boundary(),
    )
    exported = graph.export_topic(
        f"holomq://{output_service}",
        hs.payload.TensorType,
        endpoint_name=output_endpoint,
        bounds=hs.topics.EndpointBounds(
            subscriber=hs.topics.SubscriberEndpointBounds(queue_depth=2, max_borrowed_samples=1)
        ),
        tensor=host_u8_boundary(),
    )
    relay = graph.op(TensorRelay, "relay")
    graph.add_flow(imported.output, relay["in"], queue_depth=2)
    graph.add_flow(relay.out, exported.input, queue_depth=2)
    plan = hs.compile(graph)
    assert plan.ok, plan.json()

    endpoint = plan.endpoint_catalog.endpoint(input_endpoint)
    requirements = endpoint.tensor_requirements
    assert endpoint.has_tensor_requirements
    assert endpoint.scalar_identity is None
    assert requirements.memory_kind is hs.tensor.MemoryKind.HOST
    assert requirements.dtype.code == hs.tensor.DLDataTypeCode.DLUINT
    assert requirements.dtype.bits == 8
    assert requirements.dtype.lanes == 1
    assert requirements.rank == 2
    assert requirements.max_byte_span == 8
    assert requirements.stride_policy is hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY
    assert requirements.foreign_emit_policy is hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED

    with hs.run_async(plan) as session, hs.channel.Runtime() as runtime:
        # Key assertion: Catalog inference recovers Tensor and its bounded loan provider.
        publisher = runtime.publisher(plan.endpoint_catalog, input_endpoint)
        subscriber = runtime.subscriber(plan.endpoint_catalog, output_endpoint)
        assert publisher.payload_type == hs.payload.TensorType
        assert subscriber.payload_type == hs.payload.TensorType
        publisher.wait_for_subscribers(1.0)
        subscriber.refresh_connections()

        loan = publisher.loan((2, 4), "uint8")
        assert loan.shape == (2, 4)
        assert loan.strides is None
        assert loan.memory_kind is hs.tensor.MemoryKind.HOST
        assert loan.nbytes == 8
        with pytest.raises(hs.errors.ResourceExhaustedError):
            publisher.loan((2, 4), "uint8")
        with loan.write() as writable:
            versioned_capsule = writable.__dlpack__(max_version=(1, 0))
            capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
            capsule_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
            capsule_is_valid.restype = ctypes.c_int
            assert capsule_is_valid(versioned_capsule, b"dltensor_versioned") == 1
            del versioned_capsule
            gc.collect()
            unsynchronized_capsule = writable.__dlpack__(stream=-1)
            assert capsule_is_valid(unsynchronized_capsule, b"dltensor") == 1
            del unsynchronized_capsule
            gc.collect()
            view = np.asarray(writable)
            view[...] = np.arange(8, dtype=np.uint8).reshape(2, 4)
            del view

        with pytest.raises(
            hs.errors.InvalidArgumentError, match="PYTHON_TENSOR_LOAN_TIMEOUT_UNSUPPORTED"
        ):
            asyncio.run(hs.channel.aio.Publisher(publisher).emit(loan, timeout=0.1))

        # Key assertion: Async publication rejects a retry deadline before consuming a one-shot
        # loan, so the caller can still choose one explicit synchronous attempt.
        assert publisher.emit(loan, frame_id=23) == 1
        sample = subscriber.receive(1.0)
        assert isinstance(sample.data, hs.Tensor)
        assert sample.data.is_read_only
        assert sample.metadata.frame_id == 23
        np.testing.assert_array_equal(
            np.asarray(sample.data), np.arange(8, dtype=np.uint8).reshape(2, 4)
        )
        del sample
        gc.collect()

        # Key assertion: Foreign NumPy input uses the contract-authorized copy path.
        copied = np.full((2, 4), 9, dtype=np.uint8)
        assert publisher.emit(copied, timeout=1.0) == 1
        copied_sample = subscriber.receive(1.0)
        assert copied_sample.data.is_read_only
        np.testing.assert_array_equal(np.asarray(copied_sample.data), copied)
        del copied_sample

        cancelled = publisher.loan((2, 4), "uint8")
        cancelled.cancel()
        with pytest.raises(hs.errors.InvalidArgumentError), cancelled.write():
            pass
        assert hs.channel.TensorLoan.__name__ == "TensorLoan"
        session.request_stop()


# REQ: PY-CHANNEL-007
# REQ: TENSOR-PY-018
def test_retained_numpy_loan_alias_blocks_transfer_cancel_and_reuse():
    np = pytest.importorskip("numpy")
    endpoint_name = unique_channel("python-channel-retained-host-alias")
    publisher_json, subscriber_json = holoscan_test_support._tensor_client_catalogs(
        endpoint_name, "host"
    )
    publisher_catalog = hs.topics.EndpointCatalog.parse(publisher_json)
    subscriber_catalog = hs.topics.EndpointCatalog.parse(subscriber_json)

    with hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(publisher_catalog, endpoint_name)
        subscriber = runtime.subscriber(subscriber_catalog, endpoint_name)
        publisher.wait_for_subscribers(2.0)
        subscriber.refresh_connections()
        loan = publisher.loan((2, 4), "uint8")

        phase = loan.write()
        writable = phase.__enter__()
        retained = np.asarray(writable)
        retained[...] = np.arange(8, dtype=np.uint8).reshape(2, 4)
        with pytest.raises(
            hs.errors.PermissionDeniedError, match="PYTHON_TENSOR_LOAN_VIEW_STILL_EXPORTED"
        ):
            phase.__exit__(None, None, None)

        # Key assertion: The ndarray's private base lease prevents publication, cancellation, and
        # reuse of the finite endpoint slot while writable authority remains reachable.
        with pytest.raises(
            hs.errors.PermissionDeniedError, match="PYTHON_TENSOR_LOAN_VIEW_STILL_EXPORTED"
        ):
            publisher.emit(loan)
        with pytest.raises(
            hs.errors.PermissionDeniedError, match="PYTHON_TENSOR_LOAN_VIEW_STILL_EXPORTED"
        ):
            loan.cancel()
        with pytest.raises(hs.errors.ResourceExhaustedError):
            publisher.loan((2, 4), "uint8")

        retained[...] = 19
        del retained
        gc.collect()
        loan.cancel()

        replacement = publisher.loan((2, 4), "uint8")
        with replacement.write() as writable:
            replacement_view = np.asarray(writable)
            replacement_view[...] = 23
            del replacement_view
        assert publisher.emit(replacement) == 1
        sample = subscriber.receive(2.0)
        np.testing.assert_array_equal(np.asarray(sample.data), np.full((2, 4), 23, dtype=np.uint8))


# REQ: PY-CHANNEL-007
# REQ: TENSOR-PY-018
def test_tensor_loan_operations_share_gil_before_mutex_order():
    endpoint_name = unique_channel("python-channel-loan-lock-order")
    publisher_json, subscriber_json = holoscan_test_support._tensor_client_catalogs(
        endpoint_name, "host"
    )
    publisher_catalog = hs.topics.EndpointCatalog.parse(publisher_json)
    subscriber_catalog = hs.topics.EndpointCatalog.parse(subscriber_json)

    with hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(publisher_catalog, endpoint_name)
        subscriber = runtime.subscriber(subscriber_catalog, endpoint_name)
        publisher.wait_for_subscribers(2.0)
        subscriber.refresh_connections()

        for _ in range(32):
            loan = publisher.loan((2, 4), "uint8")
            phase = loan.write()
            phase.__enter__()
            start = threading.Barrier(3)
            failures = []

            def finish(phase=phase, start=start, failures=failures):
                start.wait()
                try:
                    phase.__exit__(None, None, None)
                except hs.HoloscanError as caught:
                    failures.append(caught)

            def cancel(loan=loan, start=start, failures=failures):
                start.wait()
                try:
                    loan.cancel()
                except hs.HoloscanError as caught:
                    failures.append(caught)

            finish_thread = threading.Thread(target=finish)
            cancel_thread = threading.Thread(target=cancel)
            finish_thread.start()
            cancel_thread.start()
            start.wait()
            finish_thread.join(timeout=2.0)
            cancel_thread.join(timeout=2.0)

            # Key assertion: Competing transitions finish without the GIL/mutex inversion. The
            # losing transition may report that cancellation or write completion won first.
            assert not finish_thread.is_alive()
            assert not cancel_thread.is_alive()
            assert all(isinstance(failure, hs.errors.InvalidArgumentError) for failure in failures)


# REQ: PY-CHANNEL-007
# REQ: TENSOR-PY-018
@pytest.mark.parametrize(
    "memory_kind",
    [
        hs.tensor.MemoryKind.PINNED_HOST,
        hs.tensor.MemoryKind.CUDA_DEVICE,
    ],
)
def test_catalog_tensor_pinned_and_device_loan_and_copy_paths(memory_kind):
    # Key setup: Independently authored import/export catalogs use the same endpoint identity, so
    # their catalog-derived physical bindings name one bounded BufferHandle service.
    np = pytest.importorskip("numpy")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("pinned/device Tensor client coverage requires a CUDA device")
    device_memory = memory_kind is hs.tensor.MemoryKind.CUDA_DEVICE
    if memory_kind is hs.tensor.MemoryKind.CUDA_DEVICE:
        memory_kind_name = "cuda_device"
    else:
        memory_kind_name = "pinned_host"
    endpoint_name = unique_channel(f"python-channel-{memory_kind_name}")
    publisher_json, subscriber_json = holoscan_test_support._tensor_client_catalogs(
        endpoint_name, memory_kind_name
    )
    publisher_catalog = hs.topics.EndpointCatalog.parse(publisher_json)
    subscriber_catalog = hs.topics.EndpointCatalog.parse(subscriber_json)
    assert publisher_catalog.endpoint(endpoint_name).tensor_requirements.memory_kind is memory_kind
    with hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(publisher_catalog, endpoint_name)
        subscriber = runtime.subscriber(subscriber_catalog, endpoint_name)
        publisher.wait_for_subscribers(2.0)
        subscriber.refresh_connections()
        loan = publisher.loan((2, 4), "uint8")
        assert loan.memory_kind is memory_kind

        expected = np.arange(8, dtype=np.uint8).reshape(2, 4)
        if not device_memory:
            with loan.write() as writable:
                np.asarray(writable)[...] = expected
        else:
            # cudaStreamPerThread has the same pointer value on every host thread but names a
            # different actual stream. Reject it before opening the write phase so a worker cannot
            # publish readiness from a stream that did not perform its asynchronous writes.
            device_id = cp.cuda.runtime.getDevice()
            worker_ready = threading.Barrier(2)
            thread_stream_failures = []

            def request_thread_relative_stream():
                with cp.cuda.Device(device_id):
                    worker_ready.wait(timeout=2.0)
                    try:
                        loan.write(stream=2)
                    except hs.errors.InvalidArgumentError as error:
                        thread_stream_failures.append(error)

            worker = threading.Thread(target=request_thread_relative_stream)
            worker.start()
            worker_ready.wait(timeout=2.0)
            worker.join(timeout=2.0)
            assert not worker.is_alive()
            assert len(thread_stream_failures) == 1
            assert isinstance(thread_stream_failures[0], hs.errors.InvalidArgumentError)
            assert thread_stream_failures[0].token == "PYTHON_TENSOR_LOAN_THREAD_STREAM_UNSUPPORTED"

            producer_stream = cp.cuda.Stream(non_blocking=True)
            consumer_stream = cp.cuda.Stream(non_blocking=True)
            phase = loan.write(stream=producer_stream)
            writable = phase.__enter__()
            with producer_stream:
                assert not hasattr(writable, "__cuda_array_interface__")
                device_view = cp.from_dlpack(writable)
                device_view[...] = cp.asarray(expected)
                del device_view

            # Key assertion: A writable CUDA DLPack consumer must use the stream whose readiness
            # the loan will publish. Cross-stream and explicit no-sync exports would otherwise let
            # the endpoint recycle backing before the consumer's asynchronous writes complete.
            for requested_stream, error_token in (
                (None, "PYTHON_TENSOR_LOAN_DLPACK_STREAM_MISMATCH"),
                (1, "PYTHON_TENSOR_LOAN_DLPACK_STREAM_MISMATCH"),
                (consumer_stream.ptr, "PYTHON_TENSOR_LOAN_DLPACK_STREAM_MISMATCH"),
                (-1, "PYTHON_TENSOR_LOAN_DLPACK_STREAM_UNTRACKED"),
            ):
                with pytest.raises(RuntimeError, match=error_token):
                    writable.__dlpack__(stream=requested_stream)
            phase.__exit__(None, None, None)

        assert publisher.emit(loan, timeout=None) == 1
        received = subscriber.receive(2.0)
        assert received.data.is_read_only
        if not device_memory:
            np.testing.assert_array_equal(np.asarray(received.data), expected)
            copied = np.full((2, 4), 11, dtype=np.uint8)
        else:
            cp.testing.assert_array_equal(cp.asarray(received.data), cp.asarray(expected))
            copied = cp.full((2, 4), 11, dtype=cp.uint8)
        del received

        # Key assertion: Tensor-like publication applies the same endpoint policy to pinned and
        # device inputs, while the loan path remains a direct bounded BufferHandle publication.
        assert publisher.emit(copied, timeout=2.0) == 1
        copied_sample = subscriber.receive(2.0)
        if not device_memory:
            np.testing.assert_array_equal(np.asarray(copied_sample.data), copied)
        else:
            cp.testing.assert_array_equal(cp.asarray(copied_sample.data), copied)
            del copied_sample
            gc.collect()

            retained_loan = publisher.loan((2, 4), "uint8")
            with producer_stream:
                phase = retained_loan.write(stream=producer_stream)
                writable = phase.__enter__()
                retained_device_view = cp.from_dlpack(writable)
                retained_device_view[...] = cp.asarray(expected)
                with pytest.raises(
                    hs.errors.PermissionDeniedError,
                    match="PYTHON_TENSOR_LOAN_VIEW_STILL_EXPORTED",
                ):
                    phase.__exit__(None, None, None)
            producer_stream.synchronize()
            with pytest.raises(
                hs.errors.PermissionDeniedError, match="PYTHON_TENSOR_LOAN_VIEW_STILL_EXPORTED"
            ):
                retained_loan.cancel()
            with pytest.raises(hs.errors.ResourceExhaustedError):
                publisher.loan((2, 4), "uint8")
            del retained_device_view
            gc.collect()
            retained_loan.cancel()


# REQ: PY-CHANNEL-007
# REQ: PY-CHANNEL-009
def test_name_based_tensor_publisher_requires_catalog_contract():
    # Key setup: A plain endpoint name cannot express bounded Tensor loan requirements.

    # Key assertion: Tensor publishers reject name-only attachment instead of inventing capacity.
    with (
        hs.channel.Runtime() as runtime,
        pytest.raises(
            hs.errors.NotSupportedError, match="CHANNEL_NAME_TENSOR_PUBLISHER_REQUIRES_CATALOG"
        ),
    ):
        runtime.publisher(unique_channel("python-channel-tensor-name"), hs.payload.Tensor)


# REQ: PY-CHANNEL-008
# REQ: PY-CHANNEL-009
def test_catalog_descriptor_schema_tensor_round_trip():
    # Key setup: Use client endpoints on both sides of a generated two-Tensor schema relay.
    np = pytest.importorskip("numpy")
    input_service = unique_channel("python-channel-stereo-in")
    output_service = unique_channel("python-channel-stereo-out")
    input_endpoint = f"{input_service}-publisher"
    output_endpoint = f"{output_service}-subscriber"
    payload = hs.payload.resolve(stereo_payload.StereoPayload._registry_name)
    graph = hs.Graph("channel-client-stereo-catalog", default_clock=hs.RealtimeClock)
    imported = graph.import_topic(
        f"holomq://{input_service}",
        payload,
        endpoint_name=input_endpoint,
        tensor_fields=stereo_tensor_fields(),
    )
    exported = graph.export_topic(
        f"holomq://{output_service}",
        payload,
        endpoint_name=output_endpoint,
        bounds=hs.topics.EndpointBounds(
            subscriber=hs.topics.SubscriberEndpointBounds(queue_depth=2, max_borrowed_samples=1)
        ),
        tensor_fields=stereo_tensor_fields(),
    )
    relay = graph.op(StereoRelay, "relay")
    graph.add_flow(imported.output, relay["in"], queue_depth=2)
    graph.add_flow(relay.out, exported.input, queue_depth=2)
    plan = hs.compile(graph)
    assert plan.ok, plan.json()

    with hs.run_async(plan) as session, hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(plan.endpoint_catalog, input_endpoint)
        subscriber = runtime.subscriber(plan.endpoint_catalog, output_endpoint)
        publisher.wait_for_subscribers(1.0)
        subscriber.refresh_connections()
        value = stereo_payload.StereoPayload(
            header=stereo_payload.FrameHeader(sequence=31, source="channel-client"),
            exposure_us=1200,
            left=np.array([[1, 2], [3, 4]], dtype=np.uint8),
            right=np.array([[5, 6], [7, 8]], dtype=np.uint8),
        )

        # Key assertion: Metadata and canonical Tensor fields use the descriptor channel path.
        assert publisher.emit(value, frame_id=37, timeout=1.0) == 1
        sample = subscriber.receive(1.0)
        assert isinstance(sample.data, stereo_payload.StereoPayload)
        assert sample.data.header == stereo_payload.FrameHeader(
            sequence=31, source="channel-client"
        )
        assert sample.data.exposure_us == 1201
        assert set(sample.data.tensors) == {"left", "right"}
        assert sample.data.left.is_read_only
        assert sample.data.right.is_read_only
        assert sample.metadata.frame_id == 37
        np.testing.assert_array_equal(
            np.asarray(sample.data.left), np.array([[1, 2], [3, 4]], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            np.asarray(sample.data.right), np.array([[5, 6], [7, 8]], dtype=np.uint8)
        )
        session.request_stop()


# REQ: PY-CHANNEL-007
# REQ: PY-CHANNEL-008
def test_catalog_tensor_and_descriptor_schema_cross_process(tmp_path):
    # Key setup: Keep the Python-authored relay graph in this process while an independently
    # imported Python interpreter owns both standalone client endpoints.
    tensor_input_service = unique_channel("python-channel-ipc-tensor-in")
    tensor_output_service = unique_channel("python-channel-ipc-tensor-out")
    tensor_input_endpoint = f"{tensor_input_service}-publisher"
    tensor_output_endpoint = f"{tensor_output_service}-subscriber"
    stereo_input_service = unique_channel("python-channel-ipc-stereo-in")
    stereo_output_service = unique_channel("python-channel-ipc-stereo-out")
    stereo_input_endpoint = f"{stereo_input_service}-publisher"
    stereo_output_endpoint = f"{stereo_output_service}-subscriber"

    graph = hs.Graph("channel-client-tensor-cross-process", default_clock=hs.RealtimeClock)
    tensor_imported = graph.import_topic(
        f"holomq://{tensor_input_service}",
        hs.payload.TensorType,
        endpoint_name=tensor_input_endpoint,
        bounds=hs.topics.EndpointBounds(
            publisher=hs.topics.PublisherEndpointBounds(max_loaned_samples=1)
        ),
        tensor=host_u8_boundary(),
    )
    tensor_exported = graph.export_topic(
        f"holomq://{tensor_output_service}",
        hs.payload.TensorType,
        endpoint_name=tensor_output_endpoint,
        tensor=host_u8_boundary(),
    )
    tensor_relay = graph.op(TensorRelay, "tensor-relay")
    graph.add_flow(tensor_imported.output, tensor_relay["in"], queue_depth=2)
    graph.add_flow(tensor_relay.out, tensor_exported.input, queue_depth=2)

    stereo_payload_type = hs.payload.resolve(stereo_payload.StereoPayload._registry_name)
    stereo_imported = graph.import_topic(
        f"holomq://{stereo_input_service}",
        stereo_payload_type,
        endpoint_name=stereo_input_endpoint,
        tensor_fields=stereo_tensor_fields(),
    )
    stereo_exported = graph.export_topic(
        f"holomq://{stereo_output_service}",
        stereo_payload_type,
        endpoint_name=stereo_output_endpoint,
        tensor_fields=stereo_tensor_fields(),
    )
    stereo_relay = graph.op(StereoRelay, "stereo-relay")
    graph.add_flow(stereo_imported.output, stereo_relay["in"], queue_depth=2)
    graph.add_flow(stereo_relay.out, stereo_exported.input, queue_depth=2)
    plan = hs.compile(graph)
    assert plan.ok, plan.json()
    catalog_path = tmp_path / "tensor-channel-catalog.json"
    plan.endpoint_catalog.save_json(catalog_path)

    child_program = r"""
import sys

import numpy as np
import stereo_payload

import holoscan as hs

catalog = hs.topics.EndpointCatalog.load(sys.argv[1])
with hs.channel.Runtime() as runtime:
    tensor_publisher = runtime.publisher(catalog, sys.argv[2])
    tensor_subscriber = runtime.subscriber(catalog, sys.argv[3])
    stereo_publisher = runtime.publisher(catalog, sys.argv[4])
    stereo_subscriber = runtime.subscriber(catalog, sys.argv[5])
    tensor_publisher.wait_for_subscribers(5.0)
    stereo_publisher.wait_for_subscribers(5.0)
    tensor_subscriber.refresh_connections()
    stereo_subscriber.refresh_connections()

    with tensor_publisher.loan((2, 4), "uint8").write() as loan:
        np.asarray(loan)[...] = np.arange(8, dtype=np.uint8).reshape(2, 4)
    assert tensor_publisher.emit(loan, frame_id=71) == 1
    tensor_sample = tensor_subscriber.receive(5.0)
    assert tensor_sample.data.is_read_only
    assert tensor_sample.metadata.frame_id == 71
    np.testing.assert_array_equal(
        np.asarray(tensor_sample.data), np.arange(8, dtype=np.uint8).reshape(2, 4)
    )

    stereo_value = stereo_payload.StereoPayload(
        header=stereo_payload.FrameHeader(sequence=73, source="python-child"),
        exposure_us=1700,
        left=np.array([[1, 3], [5, 7]], dtype=np.uint8),
        right=np.array([[2, 4], [6, 8]], dtype=np.uint8),
    )
    assert stereo_publisher.emit(stereo_value, frame_id=79, timeout=5.0) == 1
    stereo_sample = stereo_subscriber.receive(5.0)
    assert stereo_sample.data.header == stereo_payload.FrameHeader(
        sequence=73, source="python-child"
    )
    assert stereo_sample.data.exposure_us == 1701
    assert stereo_sample.metadata.frame_id == 79
    assert stereo_sample.data.left.is_read_only
    assert stereo_sample.data.right.is_read_only
    np.testing.assert_array_equal(
        np.asarray(stereo_sample.data.left), np.array([[1, 3], [5, 7]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        np.asarray(stereo_sample.data.right), np.array([[2, 4], [6, 8]], dtype=np.uint8)
    )
print("cross-process Tensor clients complete")
"""

    with hs.run_async(plan) as session:
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                child_program,
                str(catalog_path),
                tensor_input_endpoint,
                tensor_output_endpoint,
                stereo_input_endpoint,
                stereo_output_endpoint,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        session.request_stop()

    # Key assertion: raw and generated multi-Tensor BufferHandle routes survive a real exec
    # boundary, including a fresh generated-schema module registration in the child process.
    assert child.returncode == 0, child.stderr
    assert "cross-process Tensor clients complete" in child.stdout
