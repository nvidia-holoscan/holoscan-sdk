# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry-backed Python graph authoring tests."""

import gc

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

COUNTER = "holoscan.tests.CounterSource"
DOUBLER = "holoscan.tests.DoubleIt"
SINK = "holoscan.tests.Sink"
TENSOR_SINK = "holoscan.tests.TensorSink"


def assert_token(call, exception_type, token):
    """Check stable errors for tests such as test_authoring_rejections_preserve_core_tokens."""
    with pytest.raises(exception_type) as caught:
        call()
    assert caught.value.token == token


# REQ: PY-API-002
def test_first_pipeline_ports_flows_and_configurators():
    # Key setup: Compose and configure the canonical three-operator pipeline.
    #
    #   source --uint32--> doubler --uint32--> sink
    graph = hs.Graph("first-pipeline", default_clock=hs.RealtimeClock)
    source = graph.op(COUNTER, "source", limit=10)
    doubler = graph.op(DOUBLER, "doubler")
    sink = graph.op(SINK, "sink")

    first = graph.add_flow(
        source.out,
        doubler["in"],
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=4,
        topic_name="source-to-doubler",
        max_inflight_samples=8,
    )
    second = graph.add_flow(doubler.out, sink["in"], queue_policy=None, queue_depth=None)

    graph.configure(doubler["in"]).queue_depth(4).diagnostic_alias("values")
    graph.configure(source.out).max_emits_per_compute(1).late_joiner_history_depth(2)
    graph.configure(first).queue_policy(hs.QueuePolicy.BLOCKING).max_inflight_samples(4)

    # Key assertion: Handles expose configured topology without leaking mutable declarations.
    assert first.graph_valid
    assert second.graph_valid
    assert first.source is not source.out
    assert first.source.name == "out"
    assert first.target.name == "in"
    assert source.out.payload_type == hs.payload.uint32
    assert repr(source.out) == "<Output 'out' of source (CounterSource): uint32>"
    assert repr(first) == "<Flow source.out -> doubler.in>"
    assert first.tensor_projection is None
    assert graph.default_clock().name == "default_clock"

    foreign = hs.Graph("foreign")
    assert_token(
        lambda: foreign.configure(first),
        hs.errors.InvalidArgumentError,
        "GRAPH_FLOW_HANDLE_INVALID",
    )


# REQ: PY-API-002
def test_indexing_is_authoritative_for_keyword_port_names():
    # Key setup: Add one operator whose input name is a Python keyword.
    #
    #   doubler[in -> out]
    graph = hs.Graph("ports")
    doubler = graph.op(DOUBLER, "doubler")

    # Key assertion: Mapping syntax resolves every port while attribute syntax rejects `in`.
    assert doubler["in"].direction is hs.PortDirection.INPUT
    assert doubler.out.direction is hs.PortDirection.OUTPUT
    assert set(doubler.ports) == {"in", "out"}
    with pytest.raises(AttributeError, match=r"use operator\['in'\]"):
        getattr(doubler, "in")
    assert not hasattr(doubler, "missing")
    assert getattr(doubler, "missing", None) is None
    with pytest.raises(KeyError, match="missing"):
        doubler["missing"]


# REQ: PY-API-001
# REQ: PY-API-002
def test_authoring_rejections_preserve_core_tokens():
    # Key setup: Compose duplicate, mismatched, fan-in, and cyclic graph variants.
    #
    #   mismatch: source --uint32--> tensor_sink.input
    #   fan-in:   left  --+
    #                      +--> sink
    #             right --+
    #   cycle:    before <------> after
    duplicate_graph = hs.Graph("duplicate")
    duplicate_graph.op(COUNTER, "same")
    assert_token(
        lambda: duplicate_graph.op(DOUBLER, "same"),
        hs.errors.DuplicateNameError,
        "GRAPH_OPERATOR_NAME_DUPLICATE",
    )

    mismatch_graph = hs.Graph("mismatch")
    source = mismatch_graph.op(COUNTER, "source")
    tensor_sink = mismatch_graph.op(TENSOR_SINK, "tensor_sink")
    assert_token(
        lambda: mismatch_graph.add_flow(source.out, tensor_sink.input),
        hs.errors.IncompatibleError,
        "GRAPH_PORT_TYPE_MISMATCH",
    )

    fanin_graph = hs.Graph("fanin")
    left = fanin_graph.op(COUNTER, "left")
    right = fanin_graph.op(COUNTER, "right")
    sink = fanin_graph.op(SINK, "sink")
    fanin_graph.add_flow(left.out, sink["in"])
    assert_token(
        lambda: fanin_graph.add_flow(right.out, sink["in"]),
        hs.errors.IncompatibleError,
        "GRAPH_INPUT_FANIN_UNSUPPORTED",
    )

    cycle_graph = hs.Graph("cycle")
    before = cycle_graph.op(DOUBLER, "before")
    after = cycle_graph.op(DOUBLER, "after")
    cycle_graph.add_flow(before.out, after["in"])

    # Key assertion: Each invalid topology preserves its stable Core diagnostic token.
    assert_token(
        lambda: cycle_graph.add_flow(after.out, before["in"]),
        hs.errors.IncompatibleError,
        "GRAPH_DIRECT_CYCLE",
    )


# REQ: PY-API-002
def test_ports_and_flows_do_not_extend_graph_lifetime():
    # Key setup: Retain handles from a graph, then release every owning graph/operator object.
    #
    #   source --uint32--> sink
    graph = hs.Graph("retired")
    source = graph.op(COUNTER, "source")
    sink = graph.op(SINK, "sink")
    port = source.out
    flow = graph.add_flow(port, sink["in"])
    port_configurator = graph.configure(port)
    configurator = graph.configure(flow)

    del sink
    del source
    del graph
    gc.collect()

    # Key assertion: Ports, flows, and configurators all observe the retired graph.
    assert not port.graph_valid
    assert not flow.graph_valid
    replacement = hs.Graph("replacement")
    replacement_sink = replacement.op(SINK, "sink")
    assert_token(
        lambda: replacement.add_flow(port, replacement_sink["in"]),
        hs.errors.InvalidArgumentError,
        "GRAPH_HANDLE_INVALID",
    )
    assert_token(
        lambda: port_configurator.max_emits_per_compute(2),
        hs.errors.InvalidArgumentError,
        "GRAPH_HANDLE_INVALID",
    )
    assert_token(
        lambda: configurator.queue_depth(2),
        hs.errors.InvalidArgumentError,
        "GRAPH_HANDLE_INVALID",
    )


# REQ: PY-API-002
def test_clock_authoring_uses_registry_types_and_core_conversion_validation():
    # Key setup: Bind a default clock, an affine conversion, and an imported clock.
    #
    #   realtime(default) --affine--> manual    upstream(imported)
    graph = hs.Graph("clocks")
    realtime = graph.add_clock("RealtimeClock", "realtime")
    epoch = graph.add_clock(
        hs.RealtimeClock,
        "epoch",
        initial_time_offset=0.25,
        initial_time_scale=1.0,
        use_time_since_epoch=True,
    )
    manual = graph.add_clock(hs.ManualClock, "manual")
    imported = graph.import_clock("upstream")

    graph.set_default_clock(realtime)
    graph.add_clock_conversion(
        realtime,
        manual,
        hs.clocks.AffineClockConversion(scale=hs.clocks.ClockRateRatio(2, 4), offset_ns=17),
    )

    # Key assertion: Registry spellings resolve to stable clock handles and normalized values.
    assert graph.default_clock().name == "realtime"
    assert imported.name == "upstream"
    assert epoch.name == "epoch"
    assert not imported.is_default
    assert repr(hs.RealtimeClock) == "<ClockType holoscan.RealtimeClock>"

    assert_token(
        lambda: graph.add_clock(hs.RealtimeClock, "unknown-option", unknown=True),
        hs.errors.InvalidArgumentError,
        "GRAPH_CLOCK_OPTION_UNKNOWN",
    )
    assert_token(
        lambda: graph.add_clock(hs.ManualClock, "configured-manual", initial_time_offset=1.0),
        hs.errors.InvalidArgumentError,
        "GRAPH_CLOCK_OPTIONS_UNSUPPORTED",
    )
