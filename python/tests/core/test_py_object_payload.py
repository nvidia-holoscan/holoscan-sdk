# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local Python-object payload ownership and policy behavior."""

import gc
import threading
import weakref

import pytest

import holoscan as hs


class PyObjectSource(hs.Operator):
    """Emit one Python dictionary; used by the identity and policy tests."""

    out = hs.Output("out", hs.pyobject)
    emitted = None

    def setup(self, spec):
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        value = {"labels": ["left", "right"]}
        type(self).emitted = value
        self.out.emit(value)


class PyObjectRelay(hs.Operator):
    """Forward one Python value; used by the process-local identity test."""

    inp = hs.Input("in", hs.pyobject)
    out = hs.Output("out", hs.pyobject)

    def setup(self, spec):
        spec.input(self.inp)
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        self.out.emit(self.inp.receive_data())


class PyObjectSink(hs.Operator):
    """Retain one Python value; used by the identity and fanout tests."""

    inp = hs.Input("in", hs.pyobject)
    received = None
    metadata = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        sample = self.inp.receive()
        type(self).received = sample.data
        type(self).metadata = sample.metadata


class PyObjectBatchSource(hs.Operator):
    """Publish two retained values for the selected-batch object dispatch regression."""

    out = hs.Output("out", hs.pyobject)
    emitted = []

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(2)

    def compute(self, context):
        value = {"sequence": len(type(self).emitted)}
        type(self).emitted.append(value)
        self.out.emit(value)


class PyObjectBatchSink(hs.Operator):
    """Drain Python objects through the selected-batch materialization path."""

    inp = hs.Input("in", hs.pyobject)
    received = []

    def setup(self, spec):
        spec.input(self.inp, queue_depth=4)

    def contract(self):
        return (
            hs.Contract().trigger(hs.OnClock(period=hs.time.ms(10))).input(self.inp, hs.DrainAll())
        )

    def compute(self, ctx):
        type(self).received.extend(sample.data for sample in self.inp.selected_batch())
        if len(type(self).received) >= 2:
            ctx.request_completion()


class EphemeralPyObject:
    """Expose final-release evidence for the session-teardown test."""


class EphemeralPyObjectSource(hs.Operator):
    """Emit an unretained object; used by the session-teardown test."""

    out = hs.Output("out", hs.pyobject)
    reference = None

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        value = EphemeralPyObject()
        type(self).reference = weakref.ref(value)
        self.out.emit(value)


class EphemeralPyObjectSink(hs.Operator):
    """Consume without retention; used by the session-teardown test."""

    inp = hs.Input("in", hs.pyobject)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        self.inp.receive_data()


# REQ: PY-OP-006
def test_pyobject_payload_preserves_identity_on_process_local_operator_edges():
    # Key setup: Forward one dictionary across two native Python Operator edges.
    #
    #   source[Python] --pyobject--> relay[Python] --pyobject--> sink[Python]
    PyObjectSource.emitted = None
    PyObjectSink.received = None
    PyObjectSink.metadata = None
    graph = hs.Graph("python-object-payload", default_clock=hs.RealtimeClock)
    source = graph.op(PyObjectSource, "source")
    relay = graph.op(PyObjectRelay, "relay")
    sink = graph.op(PyObjectSink, "sink")
    graph.add_flow(source.out, relay["in"])
    graph.add_flow(relay.out, sink["in"])

    plan = hs.compile(graph)
    session = hs.run(plan)

    # Key assertion: The fixed token is public and direct retention preserves Python identity.
    assert plan.ok
    assert session.diagnostics == []
    assert hs.pyobject is hs.payload.pyobject
    assert hs.pyobject.name == "holoscan.PyObjectPayload"
    assert PyObjectSink.received is PyObjectSource.emitted
    assert PyObjectSink.received == {"labels": ["left", "right"]}
    assert PyObjectSink.received is not None
    assert PyObjectSink.metadata.payload_kind is hs.PayloadKind.INLINE_TYPED


# REQ: PY-OP-006
def test_pyobject_selected_batch_preserves_identity():
    # Key setup: Drain two process-local Python objects through the selected-batch dispatch path.
    PyObjectBatchSource.emitted = []
    PyObjectBatchSink.received = []
    graph = hs.Graph("python-object-selected-batch", default_clock=hs.RealtimeClock)
    source = graph.op(PyObjectBatchSource, "source")
    sink = graph.op(PyObjectBatchSink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=4)

    session = hs.run(hs.compile(graph))

    # Key assertion: Schema dispatch never captures the fixed object identity on batch reads.
    assert session.diagnostics == []
    assert PyObjectBatchSink.received == [{"sequence": 0}, {"sequence": 1}]
    assert all(
        received is emitted
        for received, emitted in zip(
            PyObjectBatchSink.received, PyObjectBatchSource.emitted, strict=True
        )
    )


# REQ: PY-OP-006
def test_pyobject_payload_rejects_external_topics_at_compile_time():
    # Key setup: Route a Python object output to an external HoloMQ topic.
    #
    #   source[Python] --pyobject--> holomq://python-object
    graph = hs.Graph("python-object-external", default_clock=hs.RealtimeClock)
    source = graph.op(PyObjectSource, "source")
    exported = graph.export_topic("holomq://python-object", hs.pyobject)
    graph.add_flow(source.out, exported.input)

    plan = hs.compile(graph)

    # Key assertion: The identity has no wire codec and therefore cannot cross the boundary.
    assert not plan.ok
    assert any(item.token == "PROCESS_LOCAL_CHANNEL_PLAN_INVALID" for item in plan.errors)

    with pytest.raises(hs.errors.CompileError) as caught:
        hs.run(plan)
    assert caught.value.token == "EXECUTION_PLAN_REJECTED"


# REQ: PY-OP-006
def test_pyobject_payload_rejects_fanout_at_compile_time():
    # Key setup: Fan one process-local object output out to two consumers.
    #
    #   source[Python] --pyobject--+--> left[Python]
    #                              +--> right[Python]
    graph = hs.Graph("python-object-fanout", default_clock=hs.RealtimeClock)
    source = graph.op(PyObjectSource, "source")
    left = graph.op(PyObjectSink, "left")
    right = graph.op(PyObjectSink, "right")
    graph.add_flow(source.out, left["in"])
    graph.add_flow(source.out, right["in"])

    # Key assertion: An identity-preserving object edge admits exactly one consumer.
    with pytest.raises(hs.errors.InvalidArgumentError) as raised:
        hs.compile(graph)
    assert raised.value.token == "PROCESS_LOCAL_CHANNEL_PLAN_INVALID"


# REQ: PY-OP-006
def test_pyobject_payload_compile_policy_is_applied_from_python():
    # Key setup: Compile one process-local Python-object edge under every policy value.
    #
    #   source[Python] --pyobject--> sink[Python]
    graph = hs.Graph("python-object-policy", default_clock=hs.RealtimeClock)
    source = graph.op(PyObjectSource, "source")
    sink = graph.op(PyObjectSink, "sink")
    graph.add_flow(source.out, sink["in"])

    allowed = hs.compile(
        graph, hs.CompileOptions(opaque_object_payloads=hs.OpaqueObjectPayloadPolicy.ALLOW)
    )
    warned = hs.compile(
        graph, hs.CompileOptions(opaque_object_payloads=hs.OpaqueObjectPayloadPolicy.WARN)
    )
    rejected = hs.compile(
        graph, hs.CompileOptions(opaque_object_payloads=hs.OpaqueObjectPayloadPolicy.REJECT)
    )

    # Key assertion: Python selects allow, advisory, and rejection behavior end to end.
    assert set(hs.OpaqueObjectPayloadPolicy) == {
        hs.OpaqueObjectPayloadPolicy.ALLOW,
        hs.OpaqueObjectPayloadPolicy.WARN,
        hs.OpaqueObjectPayloadPolicy.REJECT,
    }
    assert hs.CompileOptions().opaque_object_payloads is hs.OpaqueObjectPayloadPolicy.ALLOW
    assert allowed.ok
    assert not any(item.token.startswith("PYOBJECT_PAYLOAD_") for item in allowed.diagnostics)
    assert warned.ok
    assert any(item.token == "PYOBJECT_PAYLOAD_ADVISORY" for item in warned.warnings)
    assert not rejected.ok
    assert any(item.token == "PYOBJECT_PAYLOAD_REJECTED" for item in rejected.errors)


# REQ: PY-OP-006
def test_pyobject_payload_releases_one_reference_from_a_runtime_thread():
    # Key setup: Give the binding probe sole ownership of a finalizer-bearing Python value.
    released_on = []

    class FinalizerProbe:
        def __del__(self):
            released_on.append(threading.get_ident())

    caller_thread = threading.get_ident()

    hs._core._test_release_pyobject_payload_off_thread(FinalizerProbe)

    # Key assertion: Final ownership release safely runs Python code on the worker thread.
    assert len(released_on) == 1
    assert released_on[0] != caller_thread


# REQ: PY-OP-006
def test_pyobject_payload_releases_on_session_teardown():
    # Key setup: Run an object edge whose producer and consumer retain no strong Python reference.
    #
    #   source[Python] --pyobject--> sink[Python]
    EphemeralPyObjectSource.reference = None
    graph = hs.Graph("python-object-teardown", default_clock=hs.RealtimeClock)
    source = graph.op(EphemeralPyObjectSource, "source")
    sink = graph.op(EphemeralPyObjectSink, "sink")
    graph.add_flow(source.out, sink["in"])

    session = hs.run(hs.compile(graph))
    diagnostics = session.diagnostics
    del session
    gc.collect()

    # Key assertion: Retirement and session teardown leave no hidden runtime ownership behind.
    assert diagnostics == []
    assert EphemeralPyObjectSource.reference is not None
    assert EphemeralPyObjectSource.reference() is None


# REQ: PY-OP-006
def test_pyobject_finalizer_reentry_observes_committed_sequence_state():
    # Key setup: Let drop-oldest retirement invoke a finalizer that emits one replacement sample.
    #
    #   source[Python] --pyobject(drop-oldest, depth=1)--> sink[Python]
    class ReentrantFinalizer:
        """Re-emit during retirement for this sequence-bookkeeping regression test."""

        def __init__(self, output):
            self.output = output

        def __del__(self):
            try:
                self.output.emit({"origin": "finalizer"})
                ReentrantSource.reentered = True
            except Exception as error:  # pragma: no cover - asserted regression evidence
                ReentrantSource.reentry_error = error

    class ReentrantSource(hs.Operator):
        """Trigger finalizer re-entry for this sequence-bookkeeping regression test."""

        out = hs.Output("out", hs.pyobject)
        reentered = False
        reentry_error = None

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            self.out.emit(ReentrantFinalizer(self.out))
            self.out.emit({"origin": "outer"})

    class ReentrantSink(hs.Operator):
        """Capture the final re-entrant value for this sequence-bookkeeping test."""

        inp = hs.Input("in", hs.pyobject)
        received = None

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            type(self).received = self.inp.receive_data()

    ReentrantSource.reentered = False
    ReentrantSource.reentry_error = None
    ReentrantSink.received = None
    graph = hs.Graph("python-object-finalizer-reentry", default_clock=hs.RealtimeClock)
    source = graph.op(ReentrantSource, "source")
    sink = graph.op(ReentrantSink, "sink")
    graph.add_flow(source.out, sink["in"], queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)
    graph.configure(source.out).max_emits_per_compute(3)

    session = hs.run(hs.compile(graph))

    # Key assertion: Re-entry sees the next sequence/budget and publishes without protocol failure.
    assert session.diagnostics == []
    assert ReentrantSource.reentered
    assert ReentrantSource.reentry_error is None
    assert ReentrantSink.received == {"origin": "finalizer"}
