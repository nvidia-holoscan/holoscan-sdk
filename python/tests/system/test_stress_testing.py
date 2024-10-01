"""
SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import datetime

import pytest

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.resources import RealtimeClock
from holoscan.schedulers import GreedyScheduler


class ValueData:
    def __init__(self, value):
        self.data = value

    def __repr__(self):
        return f"ValueData({self.data})"

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)


class PingTxOp(Operator):
    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out1")
        spec.output("out2")

    def compute(self, op_input, op_output, context):
        value1 = ValueData(self.index)
        self.index += 1
        op_output.emit(value1, "out1")

        value2 = ValueData(self.index)
        self.index += 1
        op_output.emit(value2, "out2")


class PingMiddleOp(Operator):
    def __init__(self, *args, **kwargs):
        # If `self.multiplier` is set here (e.g., `self.multiplier = 4`), then
        # the default value by `param()` in `setup()` will be ignored.
        # (you can just call `spec.param("multiplier")` in `setup()` to use the
        # default value)
        #
        # self.multiplier = 4
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.output("out1")
        spec.output("out2")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        value1 = op_input.receive("in1")
        value2 = op_input.receive("in2")
        self.count += 1

        # Multiply the values by the multiplier parameter
        value1.data *= self.multiplier
        value2.data *= self.multiplier

        op_output.emit(value1, "out1")
        op_output.emit(value2, "out2")


class PingRxOp(Operator):
    def __init__(self, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("receivers", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        assert values is not None
        self.count += 1


class MyPingApp(Application):
    def __init__(self, *args, count=10, period=None, **kwargs):
        self.count = count
        self.period = period
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_args = tuple()
        if self.period:
            tx_args += (PeriodicCondition(self, recess_period=self.period),)
        tx = PingTxOp(self, CountCondition(self, self.count), *tx_args, name="tx")
        mx = PingMiddleOp(self, self.from_config("mx"), name="mx")
        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, mx, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx, rx, {("out1", "receivers"), ("out2", "receivers")})


@pytest.mark.slow
def my_ping_app_stress_test(ping_config_file):
    for _ in range(1000):
        app = MyPingApp()
        app.config(ping_config_file)
        app.run()


@pytest.mark.slow
def my_ping_app_periodic_realtime_clock_stress_test(ping_config_file):
    for _ in range(1000):
        app = MyPingApp(count=10, period=datetime.timedelta(milliseconds=100))
        app.config(ping_config_file)
        app.scheduler(GreedyScheduler(app, clock=RealtimeClock(app)))
        app.run()


@pytest.mark.slow
def test_my_tracker_app(ping_config_file):
    for _ in range(1000):
        app = MyPingApp()
        app.config(ping_config_file)
        tracker = app.track()
        app.run()
        tracker.print()
