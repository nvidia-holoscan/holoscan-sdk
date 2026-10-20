# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal selection and metadata-aware publication for native Python Operators."""

import ctypes
import threading

import pytest

import holoscan as hs


def tensor_bytes(tensor):
    """Copy a retained host Tensor for test_selected_tensor_batch_retains_payloads."""
    storage = (ctypes.c_uint8 * tensor.nbytes).from_address(tensor.data)
    return bytes(storage)


class ScalarBatchSource(hs.Operator):
    """Publish four ordered scalars; used by test_selected_batch_is_ordered_and_callback_scoped."""

    out = hs.Output("out", hs.payload.uint32)
    cross_port_token = None

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(4)

    def compute(self, context):
        next_value = getattr(self, "next_value", 1)
        if next_value == 1:
            try:
                self.out.selected_batch()
            except hs.errors.InvalidArgumentError as error:
                type(self).cross_port_token = error.token
        self.out.emit(next_value)
        self.next_value = next_value + 1


class ScalarBorrowSource(hs.Operator):
    """Publish one metadata-bearing scalar for the selected-sample facade test."""

    out = hs.Output("out", hs.payload.uint32)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        self.out.emit(37, options=hs.EmitOptions(frame_id=91))


class ScalarBorrowSink(hs.Operator):
    """Inspect and retain one callback-scoped scalar selection."""

    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)
    value = None
    metadata = None
    is_stale = None
    retained = None
    cross_thread_token = None

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, ctx):
        selected = self.inp.selected_sample()
        type(self).value = selected.data
        type(self).metadata = selected.metadata
        type(self).is_stale = selected.is_stale
        type(self).retained = selected
        with pytest.raises(hs.errors.NotSupportedError):
            self.out.forward(selected)

        def inspect_from_foreign_thread():
            try:
                _ = selected.data
            except hs.errors.NotReadyError as error:
                type(self).cross_thread_token = error.token

        worker = threading.Thread(target=inspect_from_foreign_thread)
        worker.start()
        worker.join(timeout=1.0)
        assert not worker.is_alive()
        ctx.request_completion()


class ScalarBatchSink(hs.Operator):
    """Observe DrainAll views; used by test_selected_batch_is_ordered_and_callback_scoped."""

    inp = hs.Input("in", hs.payload.uint32)
    values = []
    batch_values = []
    repeated_values = []
    sliced_values = []
    states = []
    retained_batch = None
    retained_iterator = None
    retained_port = None
    retained_sample = None
    cross_thread_token = None

    def setup(self, spec):
        spec.input(self.inp, queue_depth=8)

    def contract(self):
        return (
            hs.Contract().trigger(hs.OnClock(period=hs.time.ms(10))).input(self.inp, hs.DrainAll())
        )

    def compute(self, ctx):
        batch = self.inp.selected_batch()
        type(self).retained_port = self.inp
        type(self).states.append(self.inp.selection())
        values = [sample.data for sample in batch]
        type(self).batch_values.append(values)
        type(self).values.extend(values)
        type(self).repeated_values.extend(batch[index].data for index in range(len(batch)))
        type(self).sliced_values.extend(sample.data for sample in batch[1::2])
        if type(self).cross_thread_token is None:

            def inspect_from_foreign_thread():
                try:
                    self.inp.selection()
                except hs.errors.NotReadyError as error:
                    type(self).cross_thread_token = error.token

            worker = threading.Thread(target=inspect_from_foreign_thread)
            worker.start()
            worker.join(timeout=1.0)
            assert not worker.is_alive()
        if batch:
            type(self).retained_batch = batch
            type(self).retained_iterator = iter(batch)
            type(self).retained_sample = batch[-1]
        if len(type(self).values) >= 4:
            ctx.request_completion()


class SilentBatchSource(hs.Operator):
    """Produce no payload; used by test_input_selection_reports_safe_default."""

    out = hs.Output("out", hs.payload.uint32)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        pass


class SafeDefaultSelectionSink(hs.Operator):
    """Observe a SafeDefault selection in test_input_selection_reports_safe_default."""

    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)
    selection = None
    forward_token = None

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.ms(1)))
            .input(self.inp, hs.LatestWithin(max_age=hs.time.ms(1)))
            .on_missing(self.inp, hs.SafeDefault())
        )

    def compute(self, ctx):
        type(self).selection = self.inp.selection()
        sample = self.inp.receive()
        try:
            self.out.forward(sample)
        except hs.errors.InvalidArgumentError as error:
            type(self).forward_token = error.token
        ctx.request_completion()


class OptionalSafeDefaultSelectionProbe(hs.Operator):
    """Observe an unwired optional input in test_optional_input_selection_reports_not_connected."""

    inp = hs.Input("in", hs.payload.uint32)
    selection = None
    sample = None

    def setup(self, spec):
        spec.input(self.inp, optional=True)

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.ms(1)))
            .input(self.inp, hs.Exact())
            .on_missing(self.inp, hs.SafeDefault())
        )

    def compute(self, ctx):
        type(self).selection = self.inp.selection()
        type(self).sample = self.inp.receive()
        ctx.request_completion()


class LazyTensorView:
    """Count requested Tensor views in test_selected_tensor_batch_retains_payloads."""

    materializations = 0

    @classmethod
    def from_dlpack(cls, tensor):
        cls.materializations += 1
        return tensor


class TensorBatchSource(hs.Operator):
    """Publish retained host tensors; used by test_selected_tensor_batch_retains_payloads."""

    out = hs.Output("out", hs.payload.Tensor)

    def setup(self, spec):
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
            ),
            hs.tensor.byte_span(4),
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(3)

    def compute(self, context):
        value = getattr(self, "value", 1)
        with self.out.allocate_tensor(shape=(4,), dtype=hs.tensor.DLDataType(1, 8, 1)) as loan:
            storage = (ctypes.c_uint8 * loan.tensor.nbytes).from_address(loan.tensor.data)
            storage[:] = [value] * loan.tensor.nbytes
        self.value = value + 1


class TensorBatchSink(hs.Operator):
    """Snapshot a Tensor DrainAll view; used by test_selected_tensor_batch_retains_payloads."""

    inp = hs.Input("in", hs.payload.Tensor, as_=LazyTensorView)
    values = []
    retained_batch = None
    retained_tensor = None
    materializations_after_len = None

    def setup(self, spec):
        spec.input(self.inp, queue_depth=8)

    def contract(self):
        return (
            hs.Contract().trigger(hs.OnClock(period=hs.time.ms(10))).input(self.inp, hs.DrainAll())
        )

    def compute(self, ctx):
        batch = self.inp.selected_batch()
        if batch:
            type(self).materializations_after_len = LazyTensorView.materializations
            type(self).values.extend(tensor_bytes(sample.data) for sample in batch)
            type(self).retained_batch = batch
            type(self).retained_tensor = batch[-1].data
        if len(type(self).values) >= 3:
            ctx.request_completion()


class SelectionDriverSource(hs.Operator):
    """Drive fusion ticks; used by test_input_selection_reports_reuse_and_staleness."""

    out = hs.Output("out", hs.payload.uint32)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(200))).complete_after(10)

    def compute(self, ctx):
        self.out.emit(1, options=hs.EmitOptions(capture_time=ctx.activation_time))


class SelectionStallSource(hs.Operator):
    """Stall after one side sample; used by test_input_selection_reports_reuse_and_staleness."""

    out = hs.Output("out", hs.payload.uint32)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

    def compute(self, ctx):
        tick = getattr(self, "tick", 0)
        self.tick = tick + 1
        if tick == 0:
            self.out.emit(tick, options=hs.EmitOptions(capture_time=ctx.activation_time))


class SelectionStateSink(hs.Operator):
    """Record side-input states; used by test_input_selection_reports_reuse_and_staleness."""

    driver = hs.Input("driver", hs.payload.uint32)
    side = hs.Input("side", hs.payload.uint32)
    states = []
    reasons = []
    stale_samples = []
    ready = threading.Event()

    def setup(self, spec):
        spec.input(self.driver, queue_depth=16)
        spec.input(self.side, queue_depth=16)

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnEach(self.driver))
            .input(self.side, hs.LatestWithin(max_age=hs.time.ms(500)))
            .on_missing(self.side, hs.ReuseLast(max_age=hs.time.ms(5_000)))
        )

    def compute(self, context):
        self.driver.receive_data()
        selection = self.side.selection()
        type(self).states.append(selection.state)
        type(self).reasons.append(selection.reason)
        if selection.state in {hs.InputState.SELECTED, hs.InputState.REUSED}:
            type(self).stale_samples.append(self.side.receive().is_stale)
        if len(type(self).states) >= 5:
            type(self).ready.set()


# REQ: PY-OP-012
def test_selected_sample_is_lazy_metadata_aware_and_callback_scoped():
    # Key setup: borrow one fixed scalar selection, materialize its value and metadata, and retain
    # the facade after the callback returns.
    #
    #   source[Python] --uint32/Exact--> sink[Python]
    ScalarBorrowSink.value = None
    ScalarBorrowSink.metadata = None
    ScalarBorrowSink.is_stale = None
    ScalarBorrowSink.retained = None
    ScalarBorrowSink.cross_thread_token = None
    graph = hs.Graph("python-scalar-selected-sample", default_clock=hs.RealtimeClock)
    source = graph.op(ScalarBorrowSource, "source")
    sink = graph.op(ScalarBorrowSink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=1)

    session = hs.run(hs.compile(graph))

    # Key assertion: The selection preserves envelope facts without an owning Core receive, its
    # materialized projections remain ordinary Python values, and the facade rejects escaped use.
    assert session.diagnostics == []
    assert ScalarBorrowSink.value == 37
    assert ScalarBorrowSink.metadata.frame_id == 91
    assert ScalarBorrowSink.metadata.sequence > 0
    assert ScalarBorrowSink.is_stale is False
    assert ScalarBorrowSink.cross_thread_token == "PYTHON_SELECTED_SAMPLE_EXPIRED"
    assert isinstance(ScalarBorrowSink.retained, hs.SelectedSample)
    for attribute in ("data", "metadata", "is_stale"):
        with pytest.raises(hs.errors.NotReadyError, match="PYTHON_SELECTED_SAMPLE_EXPIRED"):
            getattr(ScalarBorrowSink.retained, attribute)


# REQ: PY-API-021
def test_selected_batch_is_ordered_and_callback_scoped():
    # Key setup: drain four ordered scalars through callback-scoped Python selection views.
    #
    #   source[Python] --uint32/DrainAll--> sink[Python]
    ScalarBatchSink.values = []
    ScalarBatchSink.batch_values = []
    ScalarBatchSink.repeated_values = []
    ScalarBatchSink.sliced_values = []
    ScalarBatchSink.states = []
    ScalarBatchSink.retained_batch = None
    ScalarBatchSink.retained_iterator = None
    ScalarBatchSink.retained_port = None
    ScalarBatchSink.retained_sample = None
    ScalarBatchSink.cross_thread_token = None
    ScalarBatchSource.cross_port_token = None
    graph = hs.Graph("python-scalar-selected-batch", default_clock=hs.RealtimeClock)
    source = graph.op(ScalarBatchSource, "source")
    sink = graph.op(ScalarBatchSink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=8)

    session = hs.run(hs.compile(graph))

    # Key assertion: ordering and multipass access are stable, only the view expires after compute.
    assert session.diagnostics == []
    assert ScalarBatchSink.values == [1, 2, 3, 4]
    assert ScalarBatchSink.repeated_values == ScalarBatchSink.values
    # Scheduling can split the four values across several immutable DrainAll batches.
    assert ScalarBatchSink.sliced_values == [
        value for batch in ScalarBatchSink.batch_values for value in batch[1::2]
    ]
    assert any(selection.state is hs.InputState.SELECTED for selection in ScalarBatchSink.states)
    assert ScalarBatchSink.retained_sample.data == 4
    assert ScalarBatchSource.cross_port_token == "GRAPH_PORT_DIRECTION_MISMATCH"
    assert ScalarBatchSink.cross_thread_token == "PYTHON_INPUT_SELECTION_UNAVAILABLE"
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_SELECTED_BATCH_EXPIRED"):
        len(ScalarBatchSink.retained_batch)
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_SELECTED_BATCH_EXPIRED"):
        next(ScalarBatchSink.retained_iterator)
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_INPUT_SELECTION_UNAVAILABLE"):
        ScalarBatchSink.retained_port.selection()


# REQ: PY-API-021
def test_input_selection_reports_safe_default():
    # Key setup: connect a silent source to a clocked LatestWithin input with SafeDefault.
    #
    #   silent[Python] --uint32/LatestWithin+SafeDefault--> sink[Python]
    SafeDefaultSelectionSink.selection = None
    SafeDefaultSelectionSink.forward_token = None
    graph = hs.Graph("python-safe-default-selection", default_clock=hs.RealtimeClock)
    source = graph.op(SilentBatchSource, "silent")
    sink = graph.op(SafeDefaultSelectionSink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=2)

    session = hs.run(hs.compile(graph))

    # Key assertion: the immutable selection preserves the frozen missing-policy reason.
    assert session.diagnostics == []
    assert SafeDefaultSelectionSink.selection.state is hs.InputState.SAFE_DEFAULT
    assert SafeDefaultSelectionSink.selection.reason in {
        hs.SelectionReason.NEVER_SEEN,
        hs.SelectionReason.END_OF_STREAM,
    }
    assert SafeDefaultSelectionSink.selection == hs.InputSelection(
        state=hs.InputState.SAFE_DEFAULT,
        reason=SafeDefaultSelectionSink.selection.reason,
    )
    assert hash(SafeDefaultSelectionSink.selection) == hash(
        hs.InputSelection(
            state=hs.InputState.SAFE_DEFAULT,
            reason=SafeDefaultSelectionSink.selection.reason,
        )
    )
    assert repr(SafeDefaultSelectionSink.selection).startswith("InputSelection(state=")
    assert SafeDefaultSelectionSink.forward_token == "PYTHON_FORWARD_SAMPLE_MISSING"


# REQ: PY-API-021
def test_optional_input_selection_reports_not_connected():
    # Graph:
    #
    #   [unwired] --> probe.in
    #
    # Key setup: Run one explicitly optional Exact/SafeDefault input without a producer.
    OptionalSafeDefaultSelectionProbe.selection = None
    OptionalSafeDefaultSelectionProbe.sample = None
    graph = hs.Graph("python-optional-safe-default-selection", default_clock=hs.RealtimeClock)
    graph.op(OptionalSafeDefaultSelectionProbe, "probe")

    plan = hs.compile(graph)
    session = hs.run(plan)

    # Key assertion: the callback observes Core's precise safe-default/not-connected decision.
    assert plan.ok, plan.json()
    assert session.termination is hs.RunTermination.REQUESTED_COMPLETED
    assert OptionalSafeDefaultSelectionProbe.selection == hs.InputSelection(
        state=hs.InputState.SAFE_DEFAULT,
        reason=hs.SelectionReason.NOT_CONNECTED,
    )
    assert OptionalSafeDefaultSelectionProbe.sample.is_missing


# REQ: PY-API-021
def test_input_selection_reports_reuse_and_staleness():
    # Key setup: drive a LatestWithin side input across fresh and retained intervals.
    #
    #   driver[Python] --uint32-----------> sink[Python]
    #   side[Python, stalls] --uint32------/
    SelectionStateSink.states = []
    SelectionStateSink.reasons = []
    SelectionStateSink.stale_samples = []
    SelectionStateSink.ready = threading.Event()
    graph = hs.Graph("python-selection-states", default_clock=hs.RealtimeClock)
    side = graph.op(SelectionStallSource, "side")
    driver = graph.op(SelectionDriverSource, "driver")
    sink = graph.op(SelectionStateSink, "sink")
    graph.add_flow(
        driver.out,
        sink.driver,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=16,
    )
    graph.add_flow(
        side.out,
        sink.side,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=16,
    )
    graph.partition("side").add(side)
    graph.partition("driver").add(driver)
    graph.partition("sink").add(sink)

    session = hs.run_async(hs.compile(graph))
    observed = SelectionStateSink.ready.wait(timeout=3.0)
    session.request_stop()
    stopped = session.wait(timeout=5.0)

    # Key assertion: retained samples preserve the reused state and are marked stale.
    assert observed
    assert stopped
    assert session.diagnostics == []
    assert hs.InputState.SELECTED in SelectionStateSink.states
    assert hs.InputState.REUSED in SelectionStateSink.states
    assert any(
        state is hs.InputState.SELECTED and reason is hs.SelectionReason.NONE
        for state, reason in zip(
            SelectionStateSink.states, SelectionStateSink.reasons, strict=False
        )
    )
    assert any(state is hs.InputState.REUSED for state in SelectionStateSink.states)
    assert True in SelectionStateSink.stale_samples


# REQ: PY-API-021
# REQ: TENSOR-PY-016
def test_selected_tensor_batch_retains_payloads():
    # Key setup: drain three runtime-pool Tensors and retain one extracted payload after callback.
    #
    #   source[Python loan] --Tensor/DrainAll--> sink[Python]
    TensorBatchSink.values = []
    TensorBatchSink.retained_batch = None
    TensorBatchSink.retained_tensor = None
    TensorBatchSink.materializations_after_len = None
    LazyTensorView.materializations = 0
    graph = hs.Graph("python-tensor-selected-batch", default_clock=hs.RealtimeClock)
    source = graph.op(TensorBatchSource, "source")
    sink = graph.op(TensorBatchSink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=8)

    session = hs.run(hs.compile(graph))

    # Key assertion: extracted Tensors own backing while the batch facade remains callback-scoped.
    assert session.diagnostics == []
    assert TensorBatchSink.materializations_after_len == 0
    assert TensorBatchSink.values == [bytes([1] * 4), bytes([2] * 4), bytes([3] * 4)]
    assert tensor_bytes(TensorBatchSink.retained_tensor) == bytes([3] * 4)
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_SELECTED_BATCH_EXPIRED"):
        TensorBatchSink.retained_batch[0]


# REQ: PY-API-021
def test_callback_context_and_metadata_aware_publication():
    # Key setup: Stamp, forward, and explicitly override one field on a native Python sample.
    #
    #   source[Python] --uint32--> relay[Python] --uint32--> sink[Python]
    class MetadataSource(hs.Operator):
        """Stamp one event-time sample for this publication-parity test."""

        out = hs.Output("out", hs.payload.uint32)
        activation_time = None
        tag_time = None
        trace_id = None
        activation = None
        retained_context = None
        stop_requested = None

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            type(self).activation_time = ctx.activation_time
            type(self).tag_time = ctx.tag_time
            type(self).trace_id = ctx.trace_id
            type(self).activation = ctx.activation
            type(self).retained_context = ctx
            type(self).stop_requested = ctx.stop_requested
            self.out.emit(
                7,
                options=hs.EmitOptions(
                    capture_time=ctx.activation_time,
                    frame_id=41,
                    trace_id=73,
                    flags=hs.SampleFlags.DEGRADED,
                ),
            )

    class MetadataRelay(hs.Operator):
        """Forward one sample with an overridden frame ID for this test."""

        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)
        received = None

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            sample = self.inp.receive()
            type(self).received = sample
            self.out.forward(sample, options=hs.EmitOptions(frame_id=42))

    class CompletingSink(hs.Operator):
        """Request successful completion after retaining one forwarded sample for this test."""

        inp = hs.Input("in", hs.payload.uint32)
        received = None

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            type(self).received = self.inp.receive()
            ctx.request_completion()

    MetadataSource.activation_time = None
    MetadataSource.tag_time = None
    MetadataSource.trace_id = None
    MetadataSource.activation = None
    MetadataSource.retained_context = None
    MetadataSource.stop_requested = None
    MetadataRelay.received = None
    CompletingSink.received = None
    graph = hs.Graph("native-publication-metadata", default_clock=hs.RealtimeClock)
    source = graph.op(MetadataSource, "source")
    relay = graph.op(MetadataRelay, "relay")
    sink = graph.op(CompletingSink, "sink")
    graph.add_flow(source.out, relay["in"], queue_depth=2)
    graph.add_flow(relay.out, sink["in"], queue_depth=2)

    session = hs.run(hs.compile(graph))

    # Key assertion: Runtime fields are restamped while semantic metadata survives explicit merge.
    assert session.termination is hs.RunTermination.REQUESTED_COMPLETED
    assert session.diagnostics == []
    assert MetadataSource.stop_requested is False
    assert isinstance(MetadataSource.activation_time, hs.time.TimePoint)
    assert MetadataSource.activation_time.valid
    assert isinstance(MetadataSource.tag_time, hs.time.TagTime)
    assert MetadataSource.tag_time.valid
    assert MetadataSource.trace_id >= 0
    assert MetadataSource.activation == hs.ActivationStatus(
        reason=hs.ActivationReason.NOMINAL,
        policy=hs.OnViolation.EMIT_WITNESS,
    )
    assert hash(MetadataSource.activation) == hash(
        hs.ActivationStatus(
            reason=hs.ActivationReason.NOMINAL,
            policy=hs.OnViolation.EMIT_WITNESS,
        )
    )
    assert MetadataRelay.received.metadata.capture_time == MetadataSource.activation_time
    assert MetadataRelay.received.metadata.publish_time.valid
    assert CompletingSink.received.data == 7
    assert CompletingSink.received.metadata.capture_time == (
        MetadataRelay.received.metadata.capture_time
    )
    assert CompletingSink.received.metadata.frame_id == 42
    assert CompletingSink.received.metadata.trace_id == 73
    assert CompletingSink.received.metadata.flags == hs.SampleFlags.DEGRADED
    assert CompletingSink.received.metadata.publish_clock_id != 0
    assert CompletingSink.received.metadata.publish_time.valid
    assert hash(MetadataSource.activation_time) == hash(
        hs.time.TimePoint(
            MetadataSource.activation_time.timestamp_ns,
            MetadataSource.activation_time.clock_id,
        )
    )
    assert repr(MetadataSource.activation_time).startswith("TimePoint(timestamp_ns=")
    assert repr(MetadataSource.tag_time).startswith("TagTime(logical_ns=")
    with pytest.raises(AttributeError):
        MetadataSource.activation_time.timestamp_ns = 0
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_EXECUTION_CONTEXT_EXPIRED"):
        _ = MetadataSource.retained_context.activation_time
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_EXECUTION_CONTEXT_EXPIRED"):
        _ = MetadataSource.retained_context.stop_requested
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_EXECUTION_CONTEXT_EXPIRED"):
        MetadataSource.retained_context.request_completion()


# REQ: PY-API-021
def test_callback_context_rejects_live_cross_thread_access():
    # Key setup: Keep compute active while a worker tries both context observation and control.
    class ContextOwner(hs.Operator):
        """Exercise callback authority from its owner and from one worker thread."""

        results = []

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            def use_context_from_worker():
                operations = (
                    ("activation_time", lambda: ctx.activation_time),
                    ("request_stop", ctx.request_stop),
                )
                for name, operation in operations:
                    try:
                        operation()
                    except hs.errors.NotReadyError as error:
                        type(self).results.append((name, error.token))
                    else:
                        type(self).results.append((name, None))

            worker = threading.Thread(target=use_context_from_worker)
            worker.start()
            worker.join(timeout=1.0)
            assert not worker.is_alive()
            ctx.request_completion()

    ContextOwner.results = []
    graph = hs.Graph("python-live-cross-thread-context", default_clock=hs.RealtimeClock)
    graph.op(ContextOwner, "source")

    session = hs.run(hs.compile(graph))

    # Key assertion: A live pointer does not grant callback authority to another thread.
    assert ContextOwner.results == [
        ("activation_time", "PYTHON_EXECUTION_CONTEXT_EXPIRED"),
        ("request_stop", "PYTHON_EXECUTION_CONTEXT_EXPIRED"),
    ]
    assert session.termination is hs.RunTermination.REQUESTED_COMPLETED
    assert session.diagnostics == []


# REQ: PY-API-021
def test_callback_context_can_request_stop_and_options_accept_index_values():
    # Key setup: Stop a clocked Python source after constructing integer-like publication options.
    class IndexValue:
        """Provide __index__ conversion for this EmitOptions value-boundary test."""

        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    class StopSource(hs.Operator):
        """Request run termination from the callback in this representative test."""

        out = hs.Output("out", hs.payload.uint32)
        options_token = None

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            try:
                self.out.emit(1, options=object())
            except hs.errors.InvalidArgumentError as error:
                type(self).options_token = error.token
            ctx.request_stop()

    options = hs.EmitOptions(frame_id=IndexValue(41), trace_id=IndexValue(73))
    StopSource.options_token = None
    graph = hs.Graph("python-callback-request-stop", default_clock=hs.RealtimeClock)
    graph.op(StopSource, "source")

    session = hs.run(hs.compile(graph))

    # Key assertion: Value semantics and in-callback stop retain typed, inspectable behavior.
    assert options == hs.EmitOptions(frame_id=41, trace_id=73)
    assert hash(options) == hash(hs.EmitOptions(frame_id=41, trace_id=73))
    assert repr(options).startswith("EmitOptions(capture_time=None, frame_id=41")
    assert StopSource.options_token == "PYTHON_EMIT_OPTIONS_REQUIRED"
    assert session.termination is hs.RunTermination.STOPPED
    assert session.diagnostics == []


# REQ: PY-API-021
@pytest.mark.parametrize("forward", [False, True])
def test_pyobject_none_is_data_in_an_ordinary_graph(forward):
    observed = []

    class Source(hs.Operator):
        out = hs.Output("out", hs.pyobject)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            self.out.emit("replace" if forward else None)

    class Relay(hs.Operator):
        inp = hs.Input("in", hs.pyobject)
        out = hs.Output("out", hs.pyobject)

        def compute(self, context):
            sample = self.inp.receive()
            self.out.forward(sample, None)

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.pyobject)

        def compute(self, context):
            sample = self.inp.receive()
            observed.append(sample.data)
            context.request_completion()

    graph = hs.Graph("none-is-data")
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    if forward:
        relay = graph.op(Relay, "relay")
        graph.add_flow(source.out, relay["in"])
        graph.add_flow(relay.out, sink["in"])
    else:
        graph.add_flow(source.out, sink["in"])
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph)
    plan.require_no_errors()
    with hs.run_async(plan) as session:
        assert session.wait(timeout=5.0)
        assert session.diagnostics == []
    assert observed == [None]
