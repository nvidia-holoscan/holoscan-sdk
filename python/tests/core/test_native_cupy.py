# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regress zero-copy CuPy ownership across native CUDA operator edges."""

import gc
import json
import os
import subprocess
import sys
import threading
import time
import warnings
import weakref
from pathlib import Path

import numpy as np
import pytest
from holoscan_test_support import (
    _configure_native_completion_probe,
    _configure_temporal_tensor_probe,
    _copy_tensor_to_host_async,
    _CUDATensorGate,
    _CUDATensorValueGate,
    _HoloMQTensorImportProbe,
    _mark_native_completion_source_advanced,
    _native_completion_callback_counts,
    _temporal_tensor_probe_status,
    _tensor_client_catalogs,
)

import holoscan as hs
from holoscan._core import (
    _test_drain_tensor_releases,
    _test_release_tensor_in_cuda_callback,
    _test_shutdown_tensor_releases,
    _test_tensor_stream_lifetime,
)


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    return cp


# REQ: TENSOR-PY-016
def test_cupy14_sdk_stream_context_orders_gpu_work_and_expires_protocol_owner(cupy, monkeypatch):
    cp = cupy
    captured_owners = []
    thread_failures = []
    received = []
    original_from_external = cp.cuda.Stream.from_external

    def capture_owner(_cls, owner):
        captured_owners.append(owner)
        return original_from_external(owner)

    monkeypatch.setattr(cp.cuda.Stream, "from_external", classmethod(capture_owner))

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="pass_through",
                    memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                    bounds=hs.tensor.TensorBounds(max_rank=1, max_byte_span=8),
                )
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, ctx):
            scope = ctx.stream()
            native_pointer = scope.__cuda_stream__()[1]
            assert native_pointer not in (0, 1, 2)
            previous_pointer = cp.cuda.get_current_stream().ptr
            with scope:
                assert cp.cuda.get_current_stream().ptr == native_pointer
                owner = captured_owners[-1]
                assert owner is not scope
                assert owner.__cuda_stream__() == (0, native_pointer)

                def probe_wrong_thread():
                    try:
                        owner.__cuda_stream__()
                    except Exception as error:
                        thread_failures.append(error)

                worker = threading.Thread(target=probe_wrong_thread)
                worker.start()
                worker.join(timeout=2.0)
                assert not worker.is_alive()

                array = cp.full((8,), 7, dtype=cp.uint8)
                array += 2
                self.out.emit(array)
            assert cp.cuda.get_current_stream().ptr == previous_pointer
            with pytest.raises(ValueError, match="scope body failed"), scope:
                raise ValueError("scope body failed")
            assert cp.cuda.get_current_stream().ptr == previous_pointer

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp)

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with ctx.stream():
                received.append(cp.asnumpy(self.inp.receive_data()))

    graph = hs.Graph("cupy14-sdk-stream-context", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=1)
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="source",
        output_port="out",
        device=hs.execution.DeviceId(cp.cuda.runtime.getDevice()),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        session = hs.run(hs.compile(graph, hs.CompileOptions(deployment=deployment)))

    assert session.diagnostics == []
    assert len(received) == 1
    np.testing.assert_array_equal(received[0], np.full((8,), 9, dtype=np.uint8))
    assert len(thread_failures) == 1
    assert isinstance(thread_failures[0], hs.errors.NotReadyError)
    assert thread_failures[0].token == "PYTHON_EXECUTION_CONTEXT_EXPIRED"
    assert captured_owners
    for owner in captured_owners:
        with pytest.raises(hs.errors.NotReadyError, match="PYTHON_EXECUTION_CONTEXT_EXPIRED"):
            owner.__cuda_stream__()


class _CudaOwner:
    """Expose only the CUDA array interface with the actual producer stream."""

    def __init__(self, array, stream):
        self.array = array
        self.stream = stream

    @property
    def __cuda_array_interface__(self):
        return {**self.array.__cuda_array_interface__, "stream": self.stream}


class _DLPackProducer:
    """Observe stream negotiation without exposing CuPy's other protocols."""

    def __init__(self, array, expected_stream):
        self.array = array
        self.expected_stream = expected_stream

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, *, stream):
        assert stream == self.expected_stream
        return self.array.__dlpack__(stream=stream)


# REQ: PY-OP-012
# REQ: TENSOR-STORAGE-010
@pytest.mark.parametrize("receive_mode", ["owning", "selected"])
def test_tensor_materialization_tracks_the_callback_stream(receive_mode, cupy):
    # Graph: imported Python CUDA owner -> distinct callback stream -> delayed native read.
    # Use raw CUDA pointers so no subsequent array conversion registers the consumer for us.
    cp = cupy
    gate = _CUDATensorGate()
    owner = _CudaOwner(cp.full((64,), 23, dtype=cp.float32), 1)
    reference = weakref.ref(owner)
    payloads = [owner]
    del owner
    result = cp.empty((64,), dtype=cp.float32)
    delivered = threading.Event()
    lifetimes = []
    streams = {}

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="pass_through", memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE
                )
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            if payloads:
                streams["source"] = ctx.stream().__cuda_stream__()[1]
                with ctx.stream():
                    self.out.emit(payloads.pop())

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            tensor = (
                self.inp.selected_sample().data
                if receive_mode == "selected"
                else self.inp.receive_data()
            )
            stream = ctx.stream().__cuda_stream__()[1]
            streams["sink"] = stream
            lifetimes.append(_test_tensor_stream_lifetime(tensor, stream))
            gate.block(stream)
            cp.cuda.runtime.memcpyAsync(
                result.data.ptr,
                tensor.data,
                result.nbytes,
                cp.cuda.runtime.memcpyDeviceToDevice,
                stream,
            )
            delivered.set()

    graph = hs.Graph("selected-cuda-callback-tracking", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=1)
    graph.partition("source").add(source)
    graph.partition("sink").add(sink)
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="source", output_port="out", device=hs.execution.DeviceId(0)
    )
    plan = hs.compile(
        graph,
        hs.CompileOptions(
            deployment=deployment,
            cuda_execution=hs.execution.CudaExecutionCompileOptions(
                compute_stream_policy=hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
            ),
        ),
    )
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        try:
            assert delivered.wait(5), session.diagnostics
            assert gate.wait_started(5)
            assert streams["source"] != streams["sink"]
            gc.collect()
            assert reference() is not None
        finally:
            gate.release()
            session.request_stop()
            assert session.wait(timeout=5)
    assert not session.diagnostics
    assert lifetimes[0]() == 2  # The callback stream retired after the delayed read.
    cp.testing.assert_array_equal(result, cp.full((64,), 23, dtype=cp.float32))
    _test_drain_tensor_releases()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "importer,error_cleanup",
    [
        ("array_interface", False),
        ("dlpack", False),
        ("python_dlpack", False),
        ("direct", False),
        ("converted_input", False),
        ("native_forward", False),
        ("dlpack", True),
    ],
)
@pytest.mark.parametrize("shared_stream", [True, False])
def test_owned_strided_tensors_cross_edges_without_copy_and_survive_session(
    importer, error_cleanup, shared_stream, cupy
):
    cp = cupy

    completed = threading.Event()
    retained = []
    checksums = []
    source_tensors = []
    source_owners = []
    pointers = []
    streams = set()

    def declare(spec, node):
        spec.output(node.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="pass_through",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                bounds=hs.tensor.TensorBounds(max_rank=8, max_byte_span=1024**3),
            )
        )

    def enter(ctx):
        pointer = ctx.stream().__cuda_stream__()[1]
        assert pointer not in (0, 1, 2)
        streams.add(pointer)
        return cp.cuda.Stream.from_external(ctx.stream())

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            declare(spec, self)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            if len(pointers) >= 8 or len(pointers) > len(retained):
                return
            with enter(ctx):
                owner = cp.full((2, 4, 5), len(pointers) + 1, dtype=cp.complex64)
                array = owner[:, ::2, :]
                pointers.append((array.data.ptr, array.strides))
                pointer = ctx.stream().__cuda_stream__()[1]
                if importer == "native_forward":
                    wrapper = _CudaOwner(array, pointer)
                    source_owners.append(weakref.ref(wrapper))
                    tensor = hs.Tensor.as_tensor(wrapper)
                    source_tensors.append(tensor)
                    self.out.emit(tensor)
                elif importer == "array_interface":
                    self.out.emit(hs.Tensor.as_tensor(_CudaOwner(array, pointer)))
                elif importer == "dlpack":
                    self.out.emit(hs.Tensor.as_tensor(array))
                elif importer == "python_dlpack":
                    self.out.emit(hs.Tensor.as_tensor(_DLPackProducer(array, pointer)))
                else:
                    self.out.emit(array)

    class Forward(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp if importer == "converted_input" else None)
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.input(self.inp, queue_depth=1)
            declare(spec, self)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with enter(ctx):
                tensor = self.inp.receive_data()
                view = tensor if importer == "converted_input" else cp.from_dlpack(tensor)
                index = len(retained)
                assert (view.data.ptr, view.strides) == pointers[index]
                self.out.emit(tensor)

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor)

        def setup(self, spec):
            spec.input(self.inp, queue_depth=1)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with enter(ctx):
                view = cp.from_dlpack(self.inp.receive_data())
                assert (view.data.ptr, view.strides) == pointers[len(retained)]
                retained.append(view)
                if error_cleanup:
                    checksums.append(cp.sum(view))
                    completed.set()
                    raise RuntimeError("deliberate CuPy consumer failure")
                if len(retained) == 8:
                    completed.set()

    # Key setup: Preserve the reported strided complex payload and bounded two-edge graph.
    graph = hs.Graph("owning-cuda-no-copy", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    if importer == "native_forward":
        forward = graph.op("holoscan.test.CUDATensorForward", "forward")
    else:
        forward = graph.op(Forward, "forward")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, forward["in"], queue_depth=1, max_inflight_samples=2)
    graph.add_flow(forward["out"], sink["in"], queue_depth=1, max_inflight_samples=2)
    # A linear chain would otherwise share one automatic partition, and so one stream.
    for name, node in (("source", source), ("forward", forward), ("sink", sink)):
        graph.partition(name).add(node)
    deployment = hs.execution.DeploymentBinding()
    for name in ("source", "forward"):
        deployment.bind_tensor_output_device(
            operator_path=name, output_port="out", device=hs.execution.DeviceId(0)
        )
    plan = hs.compile(
        graph,
        hs.CompileOptions(
            deployment=deployment,
            cuda_execution=hs.execution.CudaExecutionCompileOptions(
                compute_stream_policy=(
                    hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
                    if shared_stream
                    else hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
                )
            ),
        ),
    )
    assert plan.ok, plan.summary()
    assert len(plan.partitions) == 3, plan.summary()
    document = json.loads(plan.json())
    assert not document["tensor_allocations"]
    assert not document["tensor_interface_pools"]
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 10
        while not completed.wait(0.01) and time.monotonic() < deadline:
            if session.wait(timeout=0):
                break
        observed = completed.is_set()
        if observed and not error_cleanup:
            with cp.cuda.Device(0):
                for pointer in streams:
                    cp.cuda.runtime.streamSynchronize(pointer)
        session.request_stop()
        assert session.wait(timeout=5)
    # Key assertion: Every import path passes unchanged pointers/strides and retains storage.
    assert observed, session.diagnostics
    if error_cleanup:
        assert any(d.token == "RUNTIME_OPERATOR_CALLBACK_FAILED" for d in session.diagnostics)
        np.testing.assert_array_equal(cp.asnumpy(checksums[0]), 20)
    else:
        assert not session.diagnostics
    # Key assertion: Only the per-partition policy gives each Python operator its own stream.
    python_operators = 2 if importer == "native_forward" else 3
    assert len(streams) == (1 if shared_stream else python_operators)
    gc.collect()
    # Churn allocations after stream/session retirement; views must own the
    # original independent arrays, not a recycled runtime-pool slot.
    with cp.cuda.Stream(non_blocking=True):
        cp.full((1000,), -1, dtype=cp.complex64)
        for index, view in enumerate(retained):
            np.testing.assert_array_equal(cp.asnumpy(view), index + 1)

    if importer == "native_forward":
        # The native consumer's stream is gone. Its lease must already have snapshotted use.
        source_tensors.clear()
        retained.clear()
        del view
        gc.collect()
        _test_drain_tensor_releases()
        assert all(reference() is None for reference in source_owners)


@pytest.mark.parametrize("pending", ["producer", "fanout", "array_interface_consumer"])
def test_imported_cuda_owner_waits_for_pending_work(pending, cupy):
    cp = cupy
    slow = cp.cuda.Stream(non_blocking=True)
    fast = cp.cuda.Stream(non_blocking=True)
    gate = _CUDATensorGate()
    result = cp.empty((64,), dtype=cp.float32)
    fast_result = cp.empty_like(result)
    finalized = []

    class Owner(_CudaOwner):
        def __del__(self):
            # A Python allocator finalizer must run with the GIL on a regular host thread.
            # CUDA API calls here would fail if ownership ran in a CUDA host callback.
            cp.empty((1,), dtype=cp.uint8)
            finalized.append(True)

    try:
        # Key setup: Hold producer work or one of two consumers behind a native CUDA gate.
        with slow:
            owner = Owner(cp.full((64,), 7, dtype=cp.float32), slow.ptr)
            if pending == "producer":
                gate.block(slow.ptr)
            tensor = hs.Tensor.as_tensor(owner)
        reference = weakref.ref(owner)
        del owner
        if pending != "producer":
            with slow:
                slow_view = (
                    cp.asarray(tensor)
                    if pending == "array_interface_consumer"
                    else cp.from_dlpack(tensor)
                )
                gate.block(slow.ptr)
                cp.cuda.runtime.memcpyAsync(
                    result.data.ptr,
                    slow_view.data.ptr,
                    result.nbytes,
                    cp.cuda.runtime.memcpyDeviceToDevice,
                    slow.ptr,
                )
            with fast:
                fast_view = cp.from_dlpack(tensor)
                cp.cuda.runtime.memcpyAsync(
                    fast_result.data.ptr,
                    fast_view.data.ptr,
                    fast_result.nbytes,
                    cp.cuda.runtime.memcpyDeviceToDevice,
                    fast.ptr,
                )
            fast.synchronize()
            del slow_view, fast_view
        assert gate.wait_started(5)
        del tensor
        gc.collect()
        # Key assertion: Neither the last consumer registered nor dropping all views frees
        # storage while a different stream still needs it. Destruction must return promptly.
        assert reference() is not None
        assert not finalized
    finally:
        gate.release()
        slow.synchronize()
        fast.synchronize()
    _test_drain_tensor_releases()
    assert reference() is None
    assert finalized == [True]
    if pending != "producer":
        np.testing.assert_array_equal(cp.asnumpy(result), 7)
        np.testing.assert_array_equal(cp.asnumpy(fast_result), 7)


@pytest.mark.parametrize("method", ["as_tensor", "from_dlpack"])
@pytest.mark.parametrize("stream_form", ["pointer", "protocol"])
def test_dlpack_import_negotiates_explicit_stream_without_copy(method, stream_form, cupy):
    cp = cupy
    stream = cp.cuda.Stream(non_blocking=True)
    calls = []

    class Producer:
        def __init__(self, array):
            self.array = array

        def __dlpack_device__(self):
            return self.array.__dlpack_device__()

        def __dlpack__(self, *, stream):
            calls.append(stream)
            return self.array.__dlpack__(stream=stream)

    # Key setup: Use only Python DLPack so the negotiated stream is observable.
    with stream:
        array = cp.arange(16, dtype=cp.float32)[::2]
        producer = Producer(array)
        argument = stream.ptr if stream_form == "pointer" else stream
        tensor = getattr(hs.Tensor, method)(producer, stream=argument)
        result = cp.from_dlpack(tensor)
        assert calls == [stream.ptr]
        # Key assertion: Explicit stream negotiation preserves allocation and strided metadata.
        assert tensor.data == result.data.ptr == array.data.ptr
        assert tensor.strides == result.strides == array.strides
        np.testing.assert_array_equal(cp.asnumpy(result), np.arange(16, dtype=np.float32)[::2])
        with pytest.raises(RuntimeError, match="stream=-1"):
            getattr(hs.Tensor, method)(producer, stream=-1)
        del result, tensor
    stream.synchronize()


@pytest.mark.parametrize("stream_argument", [-1, 2])
def test_imported_cuda_untracked_export_waits_for_work(stream_argument, cupy):
    cp = cupy
    result = cp.empty((64,), dtype=cp.float32)
    owner = _CudaOwner(cp.full((64,), 7, dtype=cp.float32), 1)
    reference = weakref.ref(owner)
    tensor = hs.Tensor.as_tensor(owner)
    del owner
    gate = _CUDATensorGate()
    try:
        # Key setup: No-sync and thread-relative DLPack exports cannot provide a portable stream.
        capsule = tensor.__dlpack__(stream=stream_argument, max_version=(1, 0))
        gate.block(2)
        cp.cuda.runtime.memcpyAsync(
            result.data.ptr, tensor.data, result.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, 2
        )
        assert gate.wait_started(5)
        del capsule, tensor
        gc.collect()
        # Key assertion: Conservative retirement covers a different thread's default stream.
        assert reference() is not None
    finally:
        gate.release()
        cp.cuda.runtime.streamSynchronize(2)
    _test_drain_tensor_releases()
    assert reference() is None
    np.testing.assert_array_equal(cp.asnumpy(result), 7)


# REQ: TENSOR-STORAGE-010
@pytest.mark.parametrize("shared_consumers", [False, True])
def test_native_completion_snapshots_cover_later_reads_on_the_same_stream(shared_consumers, cupy):
    cp = cupy
    gate = _CUDATensorGate()
    result = cp.empty((64,), dtype=cp.float32)
    pinned = cp.cuda.alloc_pinned_memory(result.nbytes)
    host_copy = np.frombuffer(pinned, dtype=cp.float32, count=result.size)
    owners = [_CudaOwner(cp.full((64,), 7, dtype=cp.float32), 1)]
    reference = weakref.ref(owners[0])
    replacement = hs.Tensor.as_tensor(cp.full((64,), 9, dtype=cp.float32))
    paused = threading.Event()
    _configure_native_completion_probe(
        gate, result.data.ptr, host_copy.ctypes.data, owners[0].array.data.ptr, result.nbytes
    )

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            self.frames = 0
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="pass_through",
                    memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                    bounds=hs.tensor.TensorBounds(max_rank=1, max_byte_span=result.nbytes),
                )
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            if paused.is_set() or self.frames >= 16:
                return
            if self.frames:
                # The first Python callback and its imported-owner snapshot are now finished.
                _mark_native_completion_source_advanced()
            with ctx.stream():
                try:
                    self.out.emit(owners[0] if owners else replacement)
                except RuntimeError as error:
                    if "backpressure" in str(error).lower():
                        return
                    raise
            if owners:
                owners.clear()
            self.frames += 1

    graph = hs.Graph("native-cupy-completion-snapshots", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    graph.partition("source").add(source)
    names = (
        ["CUDATensorFirstCompletion", "CUDATensorSecondCompletion"]
        if shared_consumers
        else ["CUDATensorCopyThenRead"]
    )
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="source", output_port="out", device=hs.execution.DeviceId(0)
    )
    for index, name in enumerate(names):
        node = graph.op(f"holoscan.test.{name}", f"sink{index}")
        graph.partition(f"sink{index}").add(node)
        graph.add_flow(source.out, node["in"], queue_depth=1)

    plan = hs.compile(
        graph,
        hs.CompileOptions(
            deployment=deployment,
            cuda_execution=hs.execution.CudaExecutionCompileOptions(
                compute_stream_policy=hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
            ),
        ),
    )
    assert plan.ok, plan.summary()
    assert (
        plan.cuda_compute_stream_policy
        == hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
    )
    with hs.run_async(plan) as session:
        try:
            assert gate.wait_started(5), session.diagnostics
            # Replacement samples evict the first Tensor from all channel caches and leases.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                counts = _native_completion_callback_counts()
                if all(count >= 8 for count in (counts[1:] if shared_consumers else counts[:1])):
                    break
                time.sleep(0.001)
            else:
                pytest.fail(f"native replacements did not retire: {counts}; {session.diagnostics}")
            # Pause publication before the check so the fixture cannot build a retirement backlog.
            paused.set()
            time.sleep(0.05)
            # An early copy snapshot cannot free the owner before the subsequent gated read.
            assert reference() is not None
        finally:
            gate.release()
            session.request_stop()
            assert session.wait(timeout=5)
    assert not session.diagnostics
    _test_drain_tensor_releases()
    assert reference() is None
    np.testing.assert_array_equal(cp.asnumpy(result), 7)
    np.testing.assert_array_equal(host_copy, 7)


def test_imported_cuda_fanout_exceeds_eight_streams(cupy):
    cp = cupy
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(12)]
    tensor = hs.Tensor.as_tensor(cp.arange(8, dtype=cp.float32))
    views = []
    try:
        # Key setup: A long-lived import is consumed by more than eight distinct streams.
        for stream in streams:
            with stream:
                views.append(cp.from_dlpack(tensor))
        # Key assertion: Fan-out does not fail admission or copy imported storage.
        assert all(view.data.ptr == tensor.data for view in views)
    finally:
        views.clear()
        del tensor
        for stream in streams:
            stream.synchronize()
        _test_drain_tensor_releases()


@pytest.mark.usefixtures("cupy")
def test_cupy_per_thread_default_stream_environment():
    # Key setup: CuPy reads this policy at import time, so qualify a fresh interpreter.
    program = """
import cupy as cp
import holoscan as hs
from holoscan._core import (
    _test_drain_tensor_releases,
    _test_release_tensor_in_cuda_callback,
    _test_shutdown_tensor_releases,
    _test_tensor_stream_lifetime,
)
assert cp.cuda.get_current_stream().ptr == 2
array = cp.arange(8, dtype=cp.float32)
tensor = hs.Tensor.as_tensor(array, stream=cp.cuda.get_current_stream())
view = cp.from_dlpack(tensor)
assert view.data.ptr == array.data.ptr
cp.testing.assert_array_equal(view, array)
del view, tensor
_test_drain_tensor_releases()
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "CUPY_CUDA_PER_THREAD_DEFAULT_STREAM": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # Key assertion: Ordinary cp.from_dlpack works with CuPy's per-thread default policy.
    assert result.returncode == 0, result.stderr


def test_imported_cuda_readiness_outlives_producer_stream(cupy):
    cp = cupy
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        array = cp.full((64,), 9, dtype=cp.float32)
        tensor = hs.Tensor.as_tensor(array, stream=stream)
    stream.synchronize()
    del array, stream
    gc.collect()
    # Key assertion: A later consumer waits on a retained event, not a destroyed stream handle.
    consumer = cp.cuda.Stream(non_blocking=True)
    with consumer:
        view = cp.from_dlpack(tensor)
        np.testing.assert_array_equal(cp.asnumpy(view), 9)
        del view, tensor
    consumer.synchronize()
    _test_drain_tensor_releases()


@pytest.mark.parametrize("existing_tensor", [False, True])
@pytest.mark.usefixtures("cupy")
def test_array_interface_import_does_not_borrow_temporary_import_stream(existing_tensor):
    # Key setup: Isolate the destroyed-stream regression; recording on a dead handle can crash.
    program = """
import gc
import sys
import weakref

import cupy as cp
import holoscan as hs
from holoscan._core import (
    _test_drain_tensor_releases,
    _test_release_tensor_in_cuda_callback,
    _test_shutdown_tensor_releases,
    _test_tensor_stream_lifetime,
)

class Owner:
    def __init__(self, array, stream):
        self.array = array
        self.stream = stream

    @property
    def __cuda_array_interface__(self):
        return {**self.array.__cuda_array_interface__, "stream": self.stream}

producer_stream = cp.cuda.Stream(non_blocking=True)
with producer_stream:
    array = cp.full((64,), 9, dtype=cp.float32)
owner = Owner(array, producer_stream.ptr)
owner_ref = weakref.ref(owner)
tensor = hs.Tensor.as_tensor(owner) if sys.argv[1] == "True" else None
import_stream = cp.cuda.Stream(non_blocking=True)
# Import only orders readiness on this stream; no array read is queued there.
tensor = hs.Tensor.as_tensor(tensor if tensor is not None else owner, stream=import_stream)
import_stream.synchronize()
producer_stream.synchronize()
del import_stream, owner, array
gc.collect()
assert owner_ref() is not None
# Do not create another stream here: CUDA could recycle the destroyed handle.
del tensor
_test_drain_tensor_releases()
assert owner_ref() is None
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(existing_tensor)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # Key assertion: Release after import-stream destruction still frees the foreign owner.
    assert result.returncode == 0, result.stderr


# REQ: TENSOR-STORAGE-010
@pytest.mark.parametrize("shared_stream", [True, False])
@pytest.mark.parametrize(
    "consumer", ["python", "native", "python_stop_user_stream", "native_released_gate"]
)
def test_retained_previous_frame_is_fenced_at_late_callback_release(consumer, shared_stream, cupy):
    cp = cupy
    native_consumer = consumer.startswith("native")
    gate = _CUDATensorGate()
    result = cp.empty((64,), dtype=cp.float32)
    owners = [_CudaOwner(cp.full((64,), value, dtype=cp.float32), 1) for value in (7, 9)]
    reference = weakref.ref(owners[0])
    delivered = threading.Event()
    stopped = threading.Event()
    seen = []
    streams = {}
    _configure_temporal_tensor_probe(gate, result.data.ptr)
    if consumer == "native_released_gate":
        # The native probe must remain usable after its Python gate wrapper is destroyed.
        del gate
        gc.collect()
        gate = None

    def setup_cuda(spec, node):
        if hasattr(node, "out"):
            spec.output(node.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="pass_through",
                    memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                    bounds=hs.tensor.TensorBounds(max_rank=1, max_byte_span=256),
                )
            )

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            setup_cuda(spec, self)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            if owners:
                streams["source"] = ctx.stream().__cuda_stream__()[1]
                with ctx.stream():
                    self.out.emit(owners.pop(0))

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=None if native_consumer else cp)

        def setup(self, spec):
            setup_cuda(spec, self)
            spec.input(self.inp)
            self.previous = None
            self.user_stream = None

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def copy_previous(self, pointer):
            gate.block(pointer)
            cp.cuda.runtime.memcpyAsync(
                result.data.ptr,
                self.previous.data.ptr,
                result.nbytes,
                cp.cuda.runtime.memcpyDeviceToDevice,
                pointer,
            )
            if self.user_stream is not None:
                self.user_stream.synchronize()
            self.previous = None

        def compute(self, ctx):
            with ctx.stream():
                current = self.inp.receive_data()
                seen.append(True)
                self.pointer = ctx.stream().__cuda_stream__()[1]
                streams["sink"] = self.pointer
                if native_consumer and len(seen) == 1:
                    # Keep frame 1 owned on this partition until the relay has queued its
                    # later gated read and dropped its retained handle on another partition.
                    deadline = time.monotonic() + 5
                    while _temporal_tensor_probe_status()[0] < 2:
                        assert time.monotonic() < deadline, "native relay did not queue frame 2"
                        threading.Event().wait(0.001)
                if not native_consumer:
                    if self.previous is not None and consumer == "python":
                        self.copy_previous(self.pointer)
                    elif len(seen) == 1:
                        self.previous = current
                        if consumer == "python_stop_user_stream":
                            self.user_stream = cp.cuda.Stream(non_blocking=True)
                            ready = cp.cuda.Event()
                            ready.record()
                            self.user_stream.wait_event(ready)
                if len(seen) == 2:
                    delivered.set()

        def stop(self):
            if consumer == "python_stop_user_stream" and self.previous is not None:
                with self.user_stream:
                    self.copy_previous(self.user_stream.ptr)
            stopped.set()

    graph = hs.Graph("temporal-cupy-retirement", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    nodes = {"source": source, "sink": sink}
    bound_outputs = ["source"]
    # Both frames fit on every edge, so concurrent partitions never meet pressure. A relay's
    # forwarded sample keeps its input read claim until the sink releases it.
    edge = {"queue_depth": 2, "max_inflight_samples": 2}
    if native_consumer:
        forward = graph.op("holoscan.test.CUDATensorTemporal", "forward")
        graph.add_flow(source.out, forward["in"], **edge)
        graph.add_flow(forward["out"], sink["in"], **edge)
        nodes["forward"] = forward
        bound_outputs.append("forward")
    else:
        graph.add_flow(source.out, sink["in"], **edge)
    # A linear chain would otherwise share one automatic partition, and so one stream.
    partitions = {name: [node] for name, node in nodes.items()}
    for name, members in partitions.items():
        partition = graph.partition(name)
        for node in members:
            partition.add(node)
    deployment = hs.execution.DeploymentBinding()
    for name in bound_outputs:
        deployment.bind_tensor_output_device(
            operator_path=name, output_port="out", device=hs.execution.DeviceId(0)
        )
    plan = hs.compile(
        graph,
        hs.CompileOptions(
            deployment=deployment,
            cuda_execution=hs.execution.CudaExecutionCompileOptions(
                compute_stream_policy=(
                    hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
                    if shared_stream
                    else hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
                )
            ),
        ),
    )
    assert plan.ok, plan.summary()
    assert len(plan.partitions) == len(partitions), plan.summary()
    document = json.loads(plan.json())
    assert not document["tensor_allocations"]
    assert not document["tensor_interface_pools"]
    with hs.run_async(plan) as session:
        try:
            # Key setup: Drop a previous frame in a subsequent callback, behind pending GPU work.
            assert delivered.wait(5), session.diagnostics
            if consumer == "python_stop_user_stream":
                session.request_stop()
            if gate is not None:
                assert gate.wait_started(5)
                time.sleep(0.05)  # Give an incorrect early retirement time to run its deleter.
                # Earlier activation snapshots cannot retire the previous frame yet.
                assert reference() is not None
        finally:
            if gate is not None:
                gate.release()
            session.request_stop()
            assert session.wait(timeout=5)
    assert not session.diagnostics
    if native_consumer:
        callbacks, streams["forward"] = _temporal_tensor_probe_status()
        assert callbacks == 2
    assert len(set(streams.values())) == (1 if shared_stream else len(nodes))
    _test_drain_tensor_releases()
    assert reference() is None
    np.testing.assert_array_equal(cp.asnumpy(result), 7)


@pytest.mark.parametrize("memory_kind", ["pinned_host", "cuda_device"])
def test_channel_copy_keeps_import_alive_after_publisher_close(memory_kind, cupy):
    cp = cupy
    name = f"cupy-late-drop-{os.getpid()}-{time.monotonic_ns()}"
    publisher_json, subscriber_json = _tensor_client_catalogs(name, memory_kind)
    owner = _CudaOwner(cp.arange(8, dtype=cp.uint8).reshape(2, 4), 1)
    reference = weakref.ref(owner)
    tensor = hs.Tensor.as_tensor(owner)
    del owner
    with hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(hs.topics.EndpointCatalog.parse(publisher_json), name)
        subscriber = runtime.subscriber(hs.topics.EndpointCatalog.parse(subscriber_json), name)
        publisher.wait_for_subscribers(2.0)
        subscriber.refresh_connections()
        # Key setup: Emit an existing Tensor that remains alive after its copy stream closes.
        assert publisher.emit(tensor) == 1
        sample = subscriber.receive(2.0)
        assert sample is not None
        if memory_kind == "cuda_device":
            values = cp.asnumpy(cp.from_dlpack(sample.data))
        else:
            values = np.asarray(sample.data)
        np.testing.assert_array_equal(values, np.arange(8, dtype=np.uint8).reshape(2, 4))
        del sample
    # Key assertion: Final release uses the completed copy event, not a closed endpoint stream.
    del tensor
    gc.collect()
    _test_drain_tensor_releases()
    assert reference() is None


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), sys.float_info.max])
def test_retirement_drain_rejects_unrepresentable_deadlines(timeout):
    # Private drain-helper validation; this does not prove a product lifetime requirement.
    # Key assertion: Invalid deadlines are rejected without needing a worker or CUDA device.
    with pytest.raises(ValueError, match="timeout must be finite.*deadline range"):
        _test_drain_tensor_releases(timeout=timeout)


# REQ: TENSOR-STORAGE-010
def test_retirement_worker_sleeps_while_gpu_completion_is_pending(cupy):
    cp = cupy
    stream = cp.cuda.Stream(non_blocking=True)
    gate = _CUDATensorGate()
    with stream:
        owner = cp.full((64,), 7, dtype=cp.float32)
        reference = weakref.ref(owner)
        tensor = hs.Tensor.as_tensor(owner, stream=stream)
        del owner
        view = cp.from_dlpack(tensor)
        gate.block(stream.ptr)
    try:
        assert gate.wait_started(5)
        del view, tensor
        with pytest.raises(RuntimeError, match="Timed out waiting for Python Tensor retirement"):
            _test_drain_tensor_releases(timeout=0.01)
        assert reference() is not None
        _task = next(
            path
            for path in Path("/proc/self/task").iterdir()
            if (path / "comm").read_text().strip() == "hsdk-py-retire"
        )

        def cpu_ticks():
            fields = (_task / "stat").read_text().split(")", 1)[1].split()
            return int(fields[11]) + int(fields[12])

        # Key assertion: Blocking-sync retirement waits consume little CPU under pending work.
        before = cpu_ticks()
        time.sleep(0.3)
        cpu_seconds = (cpu_ticks() - before) / os.sysconf("SC_CLK_TCK")
        print(f"retirement worker: {cpu_seconds:.3f}s CPU during 0.3s pending work")
        assert cpu_seconds < 0.15, cpu_seconds
    finally:
        gate.release()
        stream.synchronize()
    _test_drain_tensor_releases(timeout=5)
    assert reference() is None


def test_async_host_copy_snapshot_outlives_copy_stream(cupy):
    cp = cupy
    pinned = cp.cuda.alloc_pinned_memory(64 * 4)
    result = np.ndarray((64,), dtype=np.float32, buffer=pinned)
    owner = _CudaOwner(cp.full((64,), 7, dtype=cp.float32), 1)
    reference = weakref.ref(owner)
    tensor = hs.Tensor.as_tensor(owner)
    del owner
    stream = cp.cuda.Stream(non_blocking=True)
    # Key setup: Keep the source Tensor alive after its C++ asynchronous copy stream is destroyed.
    _copy_tensor_to_host_async(tensor, result.ctypes.data, result.nbytes, stream.ptr)
    stream.synchronize()
    del stream
    gc.collect()
    del tensor
    _test_drain_tensor_releases()
    # Key assertion: Copy submission is snapshotted while the stream is still alive.
    assert reference() is None
    np.testing.assert_array_equal(result, 7)


def test_imported_cuda_graph_supports_twelve_partition_consumers(cupy):
    cp = cupy
    delivered = threading.Event()
    received = []
    payload = [hs.Tensor.as_tensor(cp.arange(8, dtype=cp.float32))]

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            self.emitted = False
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="pass_through",
                    memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                    bounds=hs.tensor.TensorBounds(max_rank=1, max_byte_span=32),
                )
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            if not self.emitted:
                with ctx.stream():
                    self.out.emit(payload[0])
                self.emitted = True

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp)

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with ctx.stream():
                assert self.inp.receive_data().data.ptr == payload[0].data
                received.append(True)
                if len(received) == 12:
                    delivered.set()

    class Relay(Source):
        inp = hs.Input("in", hs.payload.Tensor)

        def setup(self, spec):
            super().setup(spec)
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with ctx.stream():
                self.out.forward(self.inp.receive())

    # Key setup: Respect the channel's eight-recipient limit while reusing one allocation
    # across two relays and twelve independent consumer execution partitions.
    graph = hs.Graph("cupy-partition-fanout", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="source", output_port="out", device=hs.execution.DeviceId(0)
    )
    relays = []
    for index in range(2):
        name = f"relay{index}"
        relay = graph.op(Relay, name)
        graph.add_flow(source.out, relay["in"], queue_depth=1)
        deployment.bind_tensor_output_device(
            operator_path=name, output_port="out", device=hs.execution.DeviceId(0)
        )
        relays.append(relay)
    for index in range(12):
        name = f"sink{index}"
        sink = graph.op(Sink, name)
        graph.add_flow(relays[index // 6].out, sink["in"], queue_depth=1)
    plan = hs.compile(
        graph,
        hs.CompileOptions(
            deployment=deployment,
            cuda_execution=hs.execution.CudaExecutionCompileOptions(
                compute_stream_policy=hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
            ),
        ),
    )
    assert plan.ok, plan.summary()
    document = json.loads(plan.json())
    assert not document["tensor_allocations"]
    assert not document["tensor_interface_pools"]
    with hs.run_async(plan) as session:
        try:
            # Key assertion: The source stream does not reduce a lifetime-wide fan-out budget.
            assert delivered.wait(5), session.diagnostics
        finally:
            session.request_stop()
            assert session.wait(timeout=5)
    assert not session.diagnostics
    payload.clear()
    _test_drain_tensor_releases()


# REQ: TENSOR-STORAGE-010
@pytest.mark.usefixtures("cupy")
def test_release_worker_does_not_inherit_a_pinned_import_thread():
    # Key setup: Load the SDK before the first importer is restricted to one CPU.
    program = """
import os
import time
from pathlib import Path
import cupy as cp
import holoscan as hs
from holoscan._core import _test_drain_tensor_releases
allowed = os.sched_getaffinity(0)
if len(allowed) < 2:
    raise SystemExit(77)
isolated = set()
try:
    for part in Path('/sys/devices/system/cpu/isolated').read_text().strip().split(','):
        if part:
            first, _, last = part.partition('-')
            isolated.update(range(int(first), int(last or first) + 1))
except FileNotFoundError:
    pass
expected = (allowed - isolated) or allowed
array = cp.arange(8, dtype=cp.float32)
cp.cuda.runtime.deviceSynchronize()
try:
    os.sched_setaffinity(0, {min(allowed)})
    tensor = hs.Tensor.as_tensor(array)
    deadline = time.monotonic() + 5
    worker = None
    while worker is None and time.monotonic() < deadline:
        for path in Path('/proc/self/task').iterdir():
            try:
                if (path / 'comm').read_text().strip() == 'hsdk-py-retire':
                    worker = int(path.name)
                    break
            except FileNotFoundError:
                pass
        time.sleep(0.001)
    assert worker is not None, 'release worker did not start'
    assert os.sched_getaffinity(worker) == expected
    del tensor, array
    _test_drain_tensor_releases()
finally:
    os.sched_setaffinity(0, allowed)
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=20
    )
    if result.returncode == 77:
        pytest.skip("At least two allowed CPUs are needed to test inherited affinity")
    # Key assertion: The service remains schedulable outside the importing lane's CPU.
    assert result.returncode == 0, result.stdout + result.stderr


# REQ: TENSOR-STORAGE-010
# REQ: LIFECYCLE-011
@pytest.mark.parametrize(
    "release_mode",
    [
        "thread",
        "cycle",
        "closing_thread",
        "closing_cycle",
        "cuda_callback",
        "closing_cuda_callback",
        "closing_worker_shutdown",
    ],
)
def test_sdk_stream_retirement_covers_escaped_finalizers(release_mode, cupy):
    """Finalizers outside callbacks use SDK stream identity, including refused Closing pins."""
    if release_mode == "closing_worker_shutdown" and not os.getenv("HSDK_TEST_RETIREMENT_SHUTDOWN"):
        process = subprocess.run(
            [sys.executable, "-m", "pytest", __file__, "-k", "closing_worker_shutdown", "-q"],
            env={**os.environ, "HSDK_TEST_RETIREMENT_SHUTDOWN": "1"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        return

    cp = cupy
    gate = _CUDATensorGate()
    payloads = [_CudaOwner(cp.full((64,), value, dtype=cp.float32), 1) for value in (23, 97)]
    reference = weakref.ref(payloads[0])
    result = cp.empty((64,), dtype=cp.float32)
    delivered = threading.Event()
    holders = []
    monitors = []
    triggers = []

    class Cycle:
        def __init__(self, view):
            self.view = view
            self.loop = self

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="pass_through", memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE
                )
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            if payloads:
                with ctx.stream():
                    try:
                        self.out.emit(payloads[0])
                    except hs.errors.BackpressuredError:
                        return
                    payloads.pop(0)

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor)

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with ctx.stream():
                tensor = self.inp.receive_data()
                stream = ctx.stream().__cuda_stream__()[1]
                if not monitors:
                    view = cp.from_dlpack(tensor)
                    monitors.append(_test_tensor_stream_lifetime(tensor, stream))
                    holders.append(Cycle(view) if "cycle" in release_mode else view)
                    return
                # Keep the trigger Tensor out of the release queue: its fence must not
                # incidentally delay retirement of the older view under test.
                triggers.append(tensor)
                # Use the older view without another protocol export or input admission.
                # Its earlier callback snapshot cannot cover this later GPU read.
                view = holders[0].view if "cycle" in release_mode else holders[0]
                gate.block(stream)
                cp.cuda.runtime.memcpyAsync(
                    result.data.ptr,
                    view.data.ptr,
                    result.nbytes,
                    cp.cuda.runtime.memcpyDeviceToDevice,
                    stream,
                )
                if "cuda_callback" in release_mode:
                    # A fresh imported alias is owned solely by the queued native host function.
                    alias = hs.Tensor.as_tensor(view)
                    _test_release_tensor_in_cuda_callback(alias, stream)
                    holders.clear()
            delivered.set()

    graph = hs.Graph("escaped-cupy-finalizer", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=1)
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="source", output_port="out", device=hs.execution.DeviceId(0)
    )
    plan = hs.compile(
        graph,
        hs.CompileOptions(
            deployment=deployment,
            cuda_execution=hs.execution.CudaExecutionCompileOptions(
                compute_stream_policy=hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
            ),
        ),
    )
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        try:
            assert delivered.wait(5), session.diagnostics
            assert gate.wait_started(5)
            if release_mode.startswith("closing"):
                session.request_stop()
                deadline = time.monotonic() + 5
                while monitors[0]() == 0 and time.monotonic() < deadline:
                    time.sleep(0.001)
                assert monitors[0]() == 1  # Closing: admission refused, GPU still blocked

            def release():
                holders.clear()
                if "cycle" in release_mode:
                    gc.collect()

            finalizer = threading.Thread(target=release)
            finalizer.start()
            finalizer.join(timeout=5)
            assert not finalizer.is_alive()  # Closing finalizers never wait on teardown.
            time.sleep(0.05)
            assert reference() is not None
            if release_mode == "closing_worker_shutdown":
                _test_shutdown_tensor_releases()  # never join unresolved stream outcomes
        finally:
            gate.release()
            session.request_stop()
            assert session.wait(timeout=5)
    assert not session.diagnostics
    assert monitors[0]() == 2  # Retired, even after the native stream owner is gone
    triggers.clear()
    if release_mode == "closing_worker_shutdown":
        assert reference() is not None  # late outcomes deliberately retain their Python owner
    else:
        _test_drain_tensor_releases()
        assert reference() is None
    np.testing.assert_array_equal(cp.asnumpy(result), 23)


# REQ: TENSOR-STORAGE-010
@pytest.mark.parametrize("producers", [1, 4])
def test_external_retirement_wakes_after_idle_and_concurrent_bursts(producers, cupy):
    """Drain every owner across repeated idle-to-busy transitions and concurrent publishers."""
    cp = cupy
    barrier = threading.Barrier(producers + 1, timeout=10)
    owners = []
    failures = []

    def publish():
        try:
            with cp.cuda.Device(0):
                stream = cp.cuda.Stream(non_blocking=True)
                for _ in range(32):
                    with stream:
                        for _ in range(17):
                            array = cp.empty(128, dtype=cp.float32)
                            owners.append(weakref.ref(array))
                            tensor = hs.Tensor.as_tensor(array, stream=stream)
                            view = cp.from_dlpack(tensor)
                            del view, tensor, array
                    barrier.wait()
                    barrier.wait()  # Keep the caller stream alive until its owners are drained.
        except Exception as exc:
            failures.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=publish, daemon=True) for _ in range(producers)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(32):
            barrier.wait()
            _test_drain_tensor_releases(timeout=5)
            assert all(owner() is None for owner in owners)
            time.sleep(0.001)  # Allow the empty worker to sleep before the next burst.
            barrier.wait()
    except BaseException:
        barrier.abort()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=10)
    assert not failures
    assert all(not thread.is_alive() for thread in threads)


# REQ: TENSOR-STORAGE-012
@pytest.mark.parametrize("reverse_drop", [False, True])
def test_cupy_own_stream_fences_holomq_import_before_slot_reuse(cupy, reverse_drop):
    cp = cupy
    probe = _HoloMQTensorImportProbe()
    tensor = probe.import_tensor()
    fast = cp.cuda.Stream(non_blocking=True)
    slow = cp.cuda.Stream(non_blocking=True)
    gate = _CUDATensorValueGate()
    fast_result = cp.empty(64, dtype=cp.uint8)
    slow_result = cp.empty_like(fast_result)
    # Warm the ufunc before a GPU gate can block allocator or compiler work.
    cp.add(fast_result, 0, out=slow_result)
    cp.cuda.runtime.deviceSynchronize()
    with fast:
        fast_view = cp.from_dlpack(tensor)
        cp.add(fast_view, 0, out=fast_result)
    with slow:
        slow_view = cp.from_dlpack(tensor)
        gate.block(slow.ptr)
        cp.add(slow_view, 0, out=slow_result)
    try:
        fast.synchronize()
        # The GPU wait leaves fast-stream retirement callbacks free to run.
        del tensor
        if reverse_drop:
            del slow_view
            del fast_view
        else:
            del fast_view
            del slow_view
        gc.collect()
        for _ in range(20):
            assert not probe.slot_available()
            time.sleep(0.001)
    finally:
        gate.release()
        slow.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(fast_result), np.full(64, 7, np.uint8))
    np.testing.assert_array_equal(cp.asnumpy(slow_result), np.full(64, 7, np.uint8))
    deadline = time.monotonic() + 5
    while not probe.slot_available() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert probe.slot_available()
