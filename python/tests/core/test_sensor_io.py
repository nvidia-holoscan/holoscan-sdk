# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for sensor-operator sequence and provenance utilities."""

import pytest

import holoscan as hs


class MetadataSource(hs.Operator):
    """Emit one sample with complete application-owned provenance."""

    out = hs.Output("out", hs.payload.uint32)

    def setup(self, spec):
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        self.out.emit(
            7,
            options=hs.EmitOptions(
                capture_time=context.clock().at_ns(23),
                frame_id=29,
                trace_id=31,
                flags=hs.SampleFlags.DEGRADED,
            ),
        )


class ProvenanceSink(hs.Operator):
    """Accumulate one contributor and retain the resulting publication options."""

    inp = hs.Input("in", hs.payload.uint32)
    options = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        accumulator = hs.sensor_io.ProvenanceAccumulator()
        accumulator.observe(self.inp.receive().metadata)
        assert accumulator.observed == 1
        type(self).options = accumulator.emit_options(hs.sensor_io.CaptureTimePolicy.EARLIEST)


class ConflictingMetadataSource(hs.Operator):
    """Emit stamped and unstamped contributors with conflicting identities."""

    out = hs.Output("out", hs.payload.uint32)

    def setup(self, spec):
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(2)

    def start(self):
        self.index = 0

    def compute(self, context):
        if self.index == 0:
            options = hs.EmitOptions(
                capture_time=context.clock().at_ns(23),
                frame_id=29,
                trace_id=31,
            )
        else:
            options = hs.EmitOptions(frame_id=37, trace_id=41)
        self.out.emit(self.index, options=options)
        self.index += 1


class MultipleProvenanceSink(hs.Operator):
    """Combine two contributors and retain the resulting publication options."""

    inp = hs.Input("in", hs.payload.uint32)
    options = None
    state = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def start(self):
        self.accumulator = hs.sensor_io.ProvenanceAccumulator()

    def compute(self, context):
        self.accumulator.observe(self.inp.receive().metadata)
        if self.accumulator.observed == 2:
            type(self).options = self.accumulator.emit_options(
                hs.sensor_io.CaptureTimePolicy.LATEST
            )
            type(self).state = (
                self.accumulator.observed,
                self.accumulator.mixed_domains,
                self.accumulator.has_unstamped_contributor,
            )


class AccumulatorForwardProbe(hs.Operator):
    """Try to misuse authoritative many-to-one provenance for one-to-one forwarding."""

    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)
    error = None

    def setup(self, spec):
        spec.input(self.inp)
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def start(self):
        self.accumulator = hs.sensor_io.ProvenanceAccumulator()

    def compute(self, context):
        sample = self.inp.receive()
        self.accumulator.observe(sample.metadata)
        if self.accumulator.observed == 2:
            options = self.accumulator.emit_options(hs.sensor_io.CaptureTimePolicy.LATEST)
            try:
                self.out.forward(sample, options=options)
            except hs.errors.InvalidArgumentError as error:
                type(self).error = error


class ProvenanceCombiner(hs.Operator):
    """Publish one fused result with authoritative accumulated provenance."""

    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)

    def setup(self, spec):
        spec.input(self.inp)
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def start(self):
        self.accumulator = hs.sensor_io.ProvenanceAccumulator()

    def compute(self, context):
        self.accumulator.observe(self.inp.receive().metadata)
        if self.accumulator.observed == 2:
            options = self.accumulator.emit_options(hs.sensor_io.CaptureTimePolicy.LATEST)
            self.out.emit(42, options=options)
            self.accumulator.reset()


class PublishedMetadataSink(hs.Operator):
    """Retain metadata from one freshly emitted fused result."""

    inp = hs.Input("in", hs.payload.uint32)
    metadata = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).metadata = self.inp.receive().metadata


# REQ: SENSOR-IO-003
def test_sequence_tracker_reports_gaps_regressions_and_reset():
    # Key setup: Observe one epoch containing a gap and a repeated sequence number.
    tracker = hs.sensor_io.SequenceTracker()
    first = tracker.observe(10)
    advanced = tracker.observe(13)
    repeated = tracker.observe(13)

    # Key assertion: Python sees the native allocation-free tracking semantics exactly.
    assert (first.sequence, first.gap, first.first, first.regressed) == (10, 0, True, False)
    assert (advanced.sequence, advanced.gap, advanced.first, advanced.regressed) == (
        13,
        2,
        False,
        False,
    )
    assert repeated.regressed
    assert tracker.total_gap == 2
    assert tracker.previous == 13
    assert tracker.seen
    tracker.reset()
    assert not tracker.seen
    assert tracker.total_gap == 0


# REQ: SENSOR-IO-004
# REQ: SENSOR-IO-012
def test_provenance_accumulator_returns_public_emit_options():
    # Key setup: Accumulate metadata received by a Python operator in a running graph.
    graph = hs.Graph("python-provenance", default_clock=hs.RealtimeClock)
    ProvenanceSink.options = None
    source = graph.op(MetadataSource, "source")
    sink = graph.op(ProvenanceSink, "sink")
    graph.add_flow(source.out, sink["in"])

    session = hs.run(hs.compile(graph))

    # Key assertion: The native helper returns the ordinary public publication-options value.
    assert session.diagnostics == []
    options = ProvenanceSink.options
    assert isinstance(options, hs.EmitOptions)
    assert options.capture_time.timestamp_ns == 23
    assert options.frame_id == 29
    assert options.trace_id == 31
    assert options.flags == hs.SampleFlags.DEGRADED
    assert options.authoritative
    ordinary = hs.EmitOptions(
        capture_time=options.capture_time,
        frame_id=options.frame_id,
        trace_id=options.trace_id,
        flags=options.flags,
    )
    assert not ordinary.authoritative
    assert ordinary != options
    assert hash(ordinary) != hash(options)
    assert "authoritative=True" in repr(options)
    assert "authoritative=False" in repr(ordinary)

    accumulator = hs.sensor_io.ProvenanceAccumulator()
    with pytest.raises(hs.errors.InvalidArgumentError):
        accumulator.emit_options(hs.sensor_io.CaptureTimePolicy.LATEST)

    # Key setup: Combine contributors whose frame and trace identities conflict, where one lacks
    # capture time even though the other is stamped.
    graph = hs.Graph("python-multiple-provenance", default_clock=hs.RealtimeClock)
    MultipleProvenanceSink.options = None
    MultipleProvenanceSink.state = None
    source = graph.op(ConflictingMetadataSource, "source")
    sink = graph.op(MultipleProvenanceSink, "sink")
    graph.add_flow(source.out, sink["in"])
    session = hs.run(hs.compile(graph))

    # Key assertion: Python preserves the native absence and degradation semantics exactly.
    assert session.diagnostics == []
    assert MultipleProvenanceSink.state == (2, False, True)
    options = MultipleProvenanceSink.options
    assert options.capture_time is None
    assert options.frame_id is None
    assert options.trace_id is None
    assert options.flags == hs.SampleFlags.DEGRADED

    # Key assertion: Authoritative absences cannot be reinterpreted as inheritance by the
    # one-to-one forwarding convenience API.
    graph = hs.Graph("python-rejected-provenance-forward", default_clock=hs.RealtimeClock)
    AccumulatorForwardProbe.error = None
    source = graph.op(ConflictingMetadataSource, "source")
    forwarder = graph.op(AccumulatorForwardProbe, "forwarder")
    graph.add_flow(source.out, forwarder["in"])
    session = hs.run(hs.compile(graph))
    assert session.diagnostics == []
    assert AccumulatorForwardProbe.error is not None
    assert AccumulatorForwardProbe.error.token == "PYTHON_FORWARD_AUTHORITATIVE_OPTIONS_UNSUPPORTED"
    assert "use=Output.emit" in str(AccumulatorForwardProbe.error)

    # Key setup: Publish the fused result as a fresh sample with those authoritative options.
    graph = hs.Graph("python-published-provenance", default_clock=hs.RealtimeClock)
    PublishedMetadataSink.metadata = None
    source = graph.op(ConflictingMetadataSource, "source")
    combiner = graph.op(ProvenanceCombiner, "combiner")
    sink = graph.op(PublishedMetadataSink, "sink")
    graph.add_flow(source.out, combiner["in"])
    graph.add_flow(combiner.out, sink["in"])
    session = hs.run(hs.compile(graph))

    # Key assertion: Fresh emission does not restore provenance retired by the accumulator.
    assert session.diagnostics == []
    metadata = PublishedMetadataSink.metadata
    assert metadata is not None
    assert metadata.source_clock_id == 0
    assert metadata.frame_id == 0
    assert metadata.trace_id == 0
    assert metadata.flags == hs.SampleFlags.DEGRADED
