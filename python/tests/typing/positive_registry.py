# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Positive inference contracts for registry-backed authoring."""

from collections.abc import Mapping
from datetime import timedelta
from typing import assert_type, cast

from holoscan_constructor_handle_provider import TestChannel, TestSensor
from holoscan_test_handle_operators import HandleReceiver, PassThroughHandleReceiver
from holoscan_test_operators import (
    AttributeCollisionOperator,
    CounterSource,
    ParameterProbe,
    Sink,
    _CounterSourceHandle,
    _ParameterProbeHandle,
    _SinkHandle,
)

import holoscan as hs

graph = hs.Graph("typing-positive")

assert_type(AttributeCollisionOperator.name, str)
assert_type(AttributeCollisionOperator.ports, Mapping[str, hs.registry.OperatorPortInfo])
assert_type(AttributeCollisionOperator.shared, hs.registry.OperatorPortInfo)
assert_type(AttributeCollisionOperator.class_, hs.registry.OperatorPortInfo)
collision = AttributeCollisionOperator(graph, "collision", shared=3)
assert_type(collision.name, str)
assert_type(collision.ports, dict[str, hs.Port])
assert_type(collision["name"], hs.Port)
assert_type(collision["ports"], hs.Port)
assert_type(collision.shared, hs.Port)
assert_type(collision.class_, hs.Port)

source = CounterSource(graph, "source", limit=4)
assert_type(source, _CounterSourceHandle)
assert_type(source.out, hs.Port)

looked_up_source = graph.op(CounterSource, "looked-up-source", limit=4)
assert_type(looked_up_source, _CounterSourceHandle)
assert_type(looked_up_source.out, hs.Port)

sink = Sink(graph, "sink")
assert_type(sink, _SinkHandle)
assert_type(sink.in_, hs.Port)

probe = ParameterProbe(
    graph,
    "probe",
    enabled=False,
    offset=-3,
    mode="manual",
    gain=1.5,
    label="camera",
    timeout=timedelta(milliseconds=5),
    even=4,
    levels=(-2, 4),
    labels=["rgb", "left"],
    optional_count=None,
    modes=("automatic", 7),
    sensor={"channels": 3},
    backup_sensor={"channels": 4},
    secret="redacted",
)
assert_type(probe, _ParameterProbeHandle)
assert_type(probe.output, hs.Port)

graph_type = cast(
    "hs.registry.GraphType[hs.GraphHandle]",
    hs.registry.resolve("holoscan.test.ScalePairGraph", kind="graph"),
)
child = graph.embed(graph_type, "child", factor=3)
assert_type(child, hs.GraphHandle)

# Constructor-handle keywords are typed with the provider's classes on both row kinds.
HandleReceiver(graph, "handle-receiver", data_channel=TestChannel(1), camera_sensor=TestSensor())
HandleReceiver(graph, "handle-receiver-no-sensor", data_channel=TestChannel(2), camera_sensor=None)
PassThroughHandleReceiver(graph, "pass-through-receiver", data_channel=TestChannel(3), frames=1)
