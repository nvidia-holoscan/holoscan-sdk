# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Negative inference contracts shared by pyright and ty."""

from holoscan_constructor_handle_provider import TestSensor
from holoscan_test_handle_operators import HandleReceiver, PassThroughHandleReceiver
from holoscan_test_operators import CounterSource, ParameterProbe

import holoscan as hs

graph = hs.Graph("typing-negative")

source = CounterSource(graph, "source")
_ = source.missing  # typing-error: operator-port
CounterSource(graph, "bad-limit", limit="four")  # typing-error: counter-limit
ParameterProbe(graph, "bad-mode", mode="unknown")  # typing-error: enum-name
ParameterProbe(graph, "bad-levels", levels=["high"])  # typing-error: sequence-element
ParameterProbe(graph, "bad-sensor", sensor={"channels": "three"})  # typing-error: nested-field
HandleReceiver(graph, "bad-channel", data_channel=TestSensor())  # typing-error: handle-type
PassThroughHandleReceiver(graph, "bad", data_channel=TestSensor())  # typing-error: handle-keyword
