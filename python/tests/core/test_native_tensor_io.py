# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Python Tensor declarations, ports, loans, views, and CUDA stream scopes."""

import ctypes
import json
import sys
import threading
import types

import pytest

import holoscan as hs


@pytest.mark.parametrize("field_name", ["memory_kind", "dtype", "strides", "bounds", "transfer"])
def test_tensor_output_spec_type_errors_name_the_invalid_field(field_name):
    # Key setup: Supply one invalid value while every other output-spec field remains valid.
    arguments = {"storage": "runtime_pool", "dtype": "uint8", "rank": 1}
    arguments[field_name] = object()

    # Key assertion: Construction identifies the exact field rather than the enclosing object.
    with pytest.raises(hs.errors.InvalidArgumentError, match=rf"field_name={field_name}"):
        hs.tensor.OutputSpec(**arguments)


class DerivedTensorSource(hs.Operator):
    """Declare a parameter-sized Tensor pool; used by test_tensor_capacity_expression_lowers."""

    out = hs.Output("out", hs.payload.Tensor)
    shape = hs.Param([8, 16], max_elements=2, recompile_required=True)

    def setup(self, spec):
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=2,
            ),
            hs.tensor.parameter_contiguous_byte_span(type(self).shape)
            .multiplied_by(hs.tensor.byte_span(3))
            .ceiling_divided_by(hs.tensor.byte_span(2))
            .maximum_with(hs.tensor.byte_span(200))
            .aligned_to(64),
        )

    def compute(self, context):
        pass


class InputSizedTensorTransform(hs.Operator):
    """Size an output from its input bound; used by test_tensor_input_capacity_lowers."""

    inp = hs.Input("in", hs.payload.Tensor)
    out = hs.Output("out", hs.payload.Tensor)

    def setup(self, spec):
        spec.input(self.inp)
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=2,
            ),
            hs.tensor.input_max_byte_span(self.inp),
        )

    def compute(self, context):
        pass


def tensor_bytes(tensor):
    """Copy a host Tensor view for NativeTensorSink's representative runtime test."""
    storage = (ctypes.c_uint8 * tensor.nbytes).from_address(tensor.data)
    return bytes(storage)


class TensorBytesView:
    """Resolve a Tensor-to-bytes converter; used by the bind-once Tensor port test."""

    resolution_count = 0

    class _FromDlpack:
        def __get__(self, instance, owner):
            del self, instance
            owner.resolution_count += 1
            return tensor_bytes

    from_dlpack = _FromDlpack()


class NativeTensorSource(hs.Operator):
    """Publish one provider-backed Tensor loan; used by test_native_tensor_ports_run."""

    out = hs.Output("out", hs.payload.Tensor)
    consumed = False
    consumed_code = None
    consumed_token = None
    double_enter_token = None
    stream_protocol = None
    stream_thread_failures = []
    expired_stream = None
    expired_context = None
    retained_tensor = None

    def setup(self, spec):
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=2,
            ),
            hs.tensor.byte_span(6),
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(3)

    def compute(self, ctx):
        stream = ctx.stream()
        type(self).stream_protocol = stream.__cuda_stream__()

        stream_thread_failures = []

        def use_stream():
            for operation in (stream.__cuda_stream__, stream.__enter__):
                try:
                    operation()
                except Exception as error:
                    stream_thread_failures.append(error)

        worker = threading.Thread(target=use_stream)
        worker.start()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        type(self).stream_thread_failures = stream_thread_failures
        with stream:
            pass

        type(self).expired_stream = stream
        type(self).expired_context = ctx
        with self.out.allocate_tensor(shape=(2, 3), dtype=hs.tensor.DLDataType(1, 8, 1)) as loan:
            type(self).retained_tensor = loan.tensor
            storage = (ctypes.c_uint8 * loan.tensor.nbytes).from_address(loan.tensor.data)
            storage[:] = range(6)
            try:
                loan.__enter__()
            except hs.errors.InvalidArgumentError as error:
                type(self).double_enter_token = error.token
        try:
            _ = loan.tensor
        except hs.errors.ConsumedLoanError as error:
            type(self).consumed = True
            type(self).consumed_code = error.code
            type(self).consumed_token = error.token


class NativeTensorSink(hs.Operator):
    """Receive a declared Tensor view; used by test_native_tensor_ports_run."""

    inp = hs.Input("in", hs.payload.Tensor, as_=TensorBytesView)
    received = []

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received.append(self.inp.receive_data())


# REQ: TENSOR-PY-014
def test_tensor_capacity_expression_lowers_to_an_ordinary_pool_plan():
    # Key setup: Compile a Python Tensor source whose bounded shape feeds every expression operator.
    #
    #   source[Python Tensor pool]
    graph = hs.Graph("python-derived-tensor-capacity")
    graph.op(DerivedTensorSource, "source")
    plan = hs.compile(graph)
    document = json.loads(plan.json())

    # Key assertion: Compile-only evaluation resolves 8*16*3/2, max(200), align(64) to 256 bytes.
    assert plan.ok
    assert len(document["tensor_byte_span_derivations"]) == 1
    assert document["tensor_allocations"][0]["bounds"]["max_byte_span"] == 256
    assert document["tensor_allocations"][0]["aggregate_byte_capacity"] % 256 == 0
    assert document["tensor_byte_span_derivations"][0]["update_disposition"] == 1


# REQ: TENSOR-PY-014
def test_tensor_input_capacity_expression_resolves_the_connected_producer_bound():
    # Key setup: Connect a parameter-derived Python Tensor source to an input-derived transform.
    #
    #   source[Python] --Tensor--> transform[Python]
    graph = hs.Graph("python-input-derived-tensor-capacity", default_clock=hs.RealtimeClock)
    source = graph.op(DerivedTensorSource, "source")
    transform = graph.op(InputSizedTensorTransform, "transform")
    graph.add_flow(source.out, transform["in"])
    plan = hs.compile(graph)
    document = json.loads(plan.json())

    # Key assertion: Both output pools have the same finite 256-byte per-slot capacity.
    assert plan.ok
    assert sorted(
        allocation["bounds"]["max_byte_span"] for allocation in document["tensor_allocations"]
    ) == [256, 256]


# REQ: TENSOR-PY-014
def test_tensor_parameter_capacity_rejects_foreign_descriptor_ownership():
    # Key setup: Reference another operator class's descriptor from this setup callback.
    class ForeignCapacity(hs.Operator):
        """Own the foreign descriptor used only by this rejection test."""

        extent = hs.Param(64)

        def compute(self, context):
            pass

    class InvalidSource(hs.Operator):
        """Misuse ForeignCapacity.extent for the enclosing ownership test."""

        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.HOST,
                    dtype=hs.tensor.DLDataType(1, 8, 1),
                    rank=1,
                ),
                hs.tensor.parameter_byte_span(ForeignCapacity.extent),
            )

        def compute(self, context):
            pass

    graph = hs.Graph("python-foreign-tensor-capacity")

    # Key assertion: Foreign descriptor identity fails with the stable dependency token.
    with pytest.raises(
        hs.errors.InvalidArgumentError, match="GRAPH_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"
    ):
        graph.op(InvalidSource, "invalid")


# REQ: TENSOR-PY-014
def test_tensor_capacity_expression_reports_structural_invalid_reason():
    # Key setup: Create invalid literal and alignment expressions without compiling a graph.
    non_positive = hs.tensor.byte_span(0)
    bad_alignment = hs.tensor.byte_span(64).aligned_to(48)

    # Key assertion: The immutable expressions expose the exact C++ structural reason.
    assert (
        non_positive.invalid_reason
        == hs.tensor.ByteSpanExpressionInvalidReason.NON_POSITIVE_CONSTANT
    )
    assert (
        bad_alignment.invalid_reason == hs.tensor.ByteSpanExpressionInvalidReason.ALIGNMENT_INVALID
    )


# REQ: TENSOR-PY-014
def test_tensor_parameter_capacity_requires_the_class_descriptor():
    # Key setup: Use an instance parameter value where the class-level declaration token is needed.
    class InvalidSource(hs.Operator):
        """Use the tempting instance spelling for this descriptor diagnostic test."""

        out = hs.Output("out", hs.payload.Tensor)
        extent = hs.Param(64)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.HOST,
                    dtype="uint8",
                    rank=1,
                ),
                hs.tensor.parameter_byte_span(self.extent),
            )

        def compute(self, context):
            pass

    graph = hs.Graph("python-tensor-parameter-descriptor-required")

    # Key assertion: The stable token explains that type(self).extent is required.
    with pytest.raises(hs.errors.InvalidArgumentError) as raised:
        graph.op(InvalidSource, "invalid")
    assert raised.value.token == "GRAPH_OPERATOR_SETUP_RUNTIME_ERROR"
    assert "PYTHON_TENSOR_PARAMETER_DESCRIPTOR_REQUIRED" in str(raised.value)


# REQ: TENSOR-PY-015
def test_tensor_payload_declaration_rejects_registry_name_lookalikes():
    # Key setup: Supply an arbitrary object that imitates Tensor's private registry attribute.
    class PayloadLookalike:
        """Imitate a private registry attribute for this declaration-boundary test."""

        _registry_name = "holoscan.Tensor"

    # Key assertion: Only an actual payload token or the exact Tensor type is accepted.
    with pytest.raises(hs.errors.InvalidArgumentError) as raised:
        hs.Input("in", PayloadLookalike)
    assert raised.value.token == "AUTHORING_PAYLOAD_TYPE_TOKEN_REQUIRED"


# REQ: TENSOR-PY-015
def test_tensor_input_view_is_resolved_once_at_binding():
    # Key setup: Declare one valid callable conversion and one non-callable input view.
    TensorBytesView.resolution_count = 0

    class ViewConsumer(hs.Operator):
        """Carry the valid as_= converter for this bind-time resolution test."""

        inp = hs.Input("in", hs.payload.Tensor, as_=TensorBytesView)

        def compute(self, context):
            pass

    class InvalidViewConsumer(hs.Operator):
        """Carry an invalid as_= value for this bind-time rejection test."""

        inp = hs.Input("in", hs.payload.Tensor, as_=object())

        def compute(self, context):
            pass

    graph = hs.Graph("python-tensor-input-view")
    graph.op(ViewConsumer, "valid")

    # Key assertion: Binding accepts the callable but rejects an unresolved view deterministically.
    assert TensorBytesView.resolution_count == 1
    with pytest.raises(hs.errors.InvalidArgumentError, match="PYTHON_TENSOR_INPUT_VIEW_INVALID"):
        graph.op(InvalidViewConsumer, "invalid")


# REQ: TENSOR-PY-015; TENSOR-PY-016
def test_native_tensor_ports_run_with_loans_views_and_callback_scoped_streams():
    # Key setup: Run three samples through a direct pool while retaining each callback-scoped view.
    #
    #   source[Python loan] --Tensor--> sink[Python as_=bytes]
    NativeTensorSource.consumed = False
    NativeTensorSource.consumed_code = None
    NativeTensorSource.consumed_token = None
    NativeTensorSource.double_enter_token = None
    NativeTensorSource.stream_protocol = None
    NativeTensorSource.stream_thread_failures = []
    NativeTensorSource.expired_stream = None
    NativeTensorSource.expired_context = None
    NativeTensorSource.retained_tensor = None
    NativeTensorSink.received = []
    TensorBytesView.resolution_count = 0
    graph = hs.Graph("native-tensor-ports")
    source = graph.op(NativeTensorSource, "source")
    sink = graph.op(NativeTensorSink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=1)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

    plan = hs.compile(graph)
    compile_resolution_count = TensorBytesView.resolution_count
    session = hs.run(plan)

    # Key assertion: Loans release their views between activations and callback facades expire.
    assert session.diagnostics == []
    assert NativeTensorSink.received == [bytes(range(6))] * 3
    assert NativeTensorSource.consumed
    assert NativeTensorSource.consumed_code == hs.errors.ErrorCode.INVALID_ARGUMENT
    assert NativeTensorSource.consumed_token == "PYTHON_TENSOR_LOAN_CONSUMED"
    assert NativeTensorSource.double_enter_token == "PYTHON_TENSOR_LOAN_ALREADY_ENTERED"
    assert NativeTensorSource.retained_tensor.data == 0
    assert NativeTensorSource.retained_tensor.device.device_type == hs.tensor.DLDeviceType.DLCPU
    assert NativeTensorSource.retained_tensor.device.device_id == 0
    assert NativeTensorSource.retained_tensor.dtype.bits == 0
    assert NativeTensorSource.retained_tensor.ndim == 0
    assert NativeTensorSource.retained_tensor.byte_offset == 0
    assert compile_resolution_count == 2
    assert TensorBytesView.resolution_count == compile_resolution_count + 1
    assert NativeTensorSource.stream_protocol is not None
    assert NativeTensorSource.stream_protocol[0] == 0
    assert len(NativeTensorSource.stream_thread_failures) == 2
    assert all(
        isinstance(error, hs.errors.NotReadyError)
        for error in NativeTensorSource.stream_thread_failures
    )
    expected_diagnostic = "PYTHON_EXECUTION_CONTEXT_EXPIRED"
    assert all(
        error.token == expected_diagnostic for error in NativeTensorSource.stream_thread_failures
    )
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_EXECUTION_CONTEXT_EXPIRED"):
        NativeTensorSource.expired_stream.__cuda_stream__()
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_EXECUTION_CONTEXT_EXPIRED"):
        NativeTensorSource.expired_context.stream()


# REQ: TENSOR-PY-016
def test_tensor_loan_exception_abandons_and_allows_another_allocation():
    # Key setup: Abandon one retained loan view, then allocate and publish again in the callback.
    #
    #   source[Python abandon then loan] --Tensor--> sink[Python]
    class AbandoningSource(hs.Operator):
        """Exercise exception abandonment and immediate reuse in this test."""

        out = hs.Output("out", hs.payload.Tensor)
        abandoned_tensor = None
        dropped_tensor = None

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.HOST,
                    dtype=hs.tensor.DLDataType(1, 8, 1),
                    rank=1,
                ),
                hs.tensor.byte_span(4),
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            dropped = self.out.allocate_tensor(shape=(4,), dtype="uint8")
            type(self).dropped_tensor = dropped.tensor
            del dropped
            try:
                with self.out.allocate_tensor(shape=(4,), dtype="uint8") as loan:
                    type(self).abandoned_tensor = loan.tensor
                    raise ValueError("abandon this loan")
            except ValueError:
                pass
            with self.out.allocate_tensor(shape=(4,), dtype="uint8") as loan:
                storage = (ctypes.c_uint8 * 4).from_address(loan.tensor.data)
                storage[:] = (4, 3, 2, 1)

    class AbandonSink(hs.Operator):
        """Observe the post-abandon publication in this test."""

        inp = hs.Input("in", hs.payload.Tensor, as_=tensor_bytes)
        received = None

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            type(self).received = self.inp.receive_data()

    AbandoningSource.abandoned_tensor = None
    AbandoningSource.dropped_tensor = None
    AbandonSink.received = None
    graph = hs.Graph("native-tensor-loan-abandon")
    source = graph.op(AbandoningSource, "source")
    sink = graph.op(AbandonSink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=1)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

    session = hs.run(hs.compile(graph))

    # Key assertion: Exceptional exit invalidates the old view and returns capacity for reuse.
    assert session.diagnostics == []
    assert AbandoningSource.dropped_tensor.data == 0
    assert AbandoningSource.abandoned_tensor.data == 0
    assert AbandonSink.received == bytes((4, 3, 2, 1))


# REQ: TENSOR-PY-016
def test_cuda_stream_context_manager_unwinds_every_loaded_library(monkeypatch):
    # Key setup: Install fake CuPy and PyTorch stream scopes with controlled enter/exit failures.
    events = []

    class Scope:
        """Record stream-scope cleanup order for this context-manager test."""

        def __init__(self, name, *, fail_enter=False, fail_exit=False):
            self.name = name
            self.fail_enter = fail_enter
            self.fail_exit = fail_exit

        def __enter__(self):
            events.append(f"{self.name}-enter")
            if self.fail_enter:
                raise RuntimeError(f"{self.name} enter failed")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            events.append(f"{self.name}-exit")
            if self.fail_exit:
                raise RuntimeError(f"{self.name} exit failed")
            return False

    def cupy_external_stream(pointer, device_id):
        assert device_id == 7
        return Scope(f"cupy-{pointer}")

    cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(ExternalStream=cupy_external_stream))

    class TorchCuda:
        """Provide the fake PyTorch external-stream API used by this test."""

        fail_enter = True
        fail_exit = False

        @classmethod
        def ExternalStream(cls, pointer, *, device):  # noqa: N802 - mirror PyTorch API
            assert pointer == 1234
            assert device == 7
            return (pointer, device)

        @classmethod
        def stream(cls, external):
            del external
            return Scope("torch", fail_enter=cls.fail_enter, fail_exit=cls.fail_exit)

    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=TorchCuda))
    stream = hs._core._test_cuda_stream_scope(1234, device=7)
    assert type(stream).__module__ == "holoscan._core"
    assert not hasattr(hs, "_CudaStream")

    with pytest.raises(RuntimeError, match="torch enter failed"):
        stream.__enter__()
    assert events == ["cupy-1234-enter", "torch-enter", "cupy-1234-exit"]

    events.clear()
    TorchCuda.fail_enter = False
    TorchCuda.fail_exit = True
    stream.__enter__()
    with pytest.raises(RuntimeError, match="torch exit failed"):
        stream.__exit__(None, None, None)

    # Key assertion: Entry rollback and normal exit both restore every earlier library scope.
    assert events == ["cupy-1234-enter", "torch-enter", "torch-exit", "cupy-1234-exit"]

    events.clear()
    TorchCuda.fail_exit = False
    stream = hs._core._test_cuda_stream_scope(1234, device=7)
    stream.__enter__()
    del stream
    assert events == ["cupy-1234-enter", "torch-enter", "torch-exit", "cupy-1234-exit"]


# REQ: TENSOR-PY-016
def test_cuda_stream_rejects_double_enter_without_array_libraries(monkeypatch):
    # Key setup: Enter a non-null stream while neither optional array library is loaded.
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    stream = hs._core._test_cuda_stream_scope(1234)
    stream.__enter__()

    # Key assertion: Lexical state, rather than loaded integration count, rejects a second enter.
    with pytest.raises(hs.errors.InvalidArgumentError) as raised:
        stream.__enter__()
    assert raised.value.token == "PYTHON_CUDA_STREAM_ALREADY_ENTERED"
    assert stream.__exit__(None, None, None) is False
