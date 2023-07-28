# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import time

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.resources import ManualClock, RealtimeClock
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
        spec.param("receivers", kind="receivers")

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
        if self.period:
            tx_args = (PeriodicCondition(self, recess_period=self.period),)
        else:
            tx_args = tuple()
        tx = PingTxOp(self, CountCondition(self, self.count), *tx_args, name="tx")
        mx = PingMiddleOp(self, self.from_config("mx"), name="mx")
        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, mx, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx, rx, {("out1", "receivers"), ("out2", "receivers")})


def file_contains_string(filename, string):
    try:
        with open(filename, "r") as f:
            return string in f.read()

    except FileNotFoundError:
        return False


def test_my_ping_app(ping_config_file):
    app = MyPingApp()
    app.config(ping_config_file)
    app.run()


def test_my_ping_app_periodic_manual_clock(ping_config_file):
    app = MyPingApp(count=10, period=datetime.timedelta(seconds=1))
    app.config(ping_config_file)
    tstart = time.time()
    app.scheduler(GreedyScheduler(app, clock=ManualClock(app)))
    app.run()
    duration = time.time() - tstart
    # If using manual clock duration should be much less than (count - 1) seconds since
    # period is not respected.
    assert duration < 9


def test_my_ping_app_periodic_realtime_clock(ping_config_file):
    app = MyPingApp(count=10, period=datetime.timedelta(milliseconds=100))
    app.config(ping_config_file)
    tstart = time.time()
    app.scheduler(GreedyScheduler(app, clock=RealtimeClock(app)))
    app.run()
    duration = time.time() - tstart
    # If realtime clock, duration will be > ((count - 1) * period)
    assert duration > 0.9


def test_my_ping_app_periodic_scaled_realtime_clock(ping_config_file):
    app = MyPingApp(count=10, period=datetime.timedelta(milliseconds=100))
    app.config(ping_config_file)
    tstart = time.time()
    app.scheduler(GreedyScheduler(app, clock=RealtimeClock(app, initial_time_scale=5.0)))
    app.run()
    duration = time.time() - tstart
    # If realtime clock initial_time_scale>1, duration will be less than ((count - 1) * period)
    assert duration < 0.9


def test_my_ping_app_periodic_scaled_realtime_clock2(ping_config_file):
    app = MyPingApp(count=10, period=datetime.timedelta(milliseconds=100))
    app.config(ping_config_file)
    tstart = time.time()
    app.scheduler(GreedyScheduler(app, clock=RealtimeClock(app, initial_time_scale=0.5)))
    app.run()
    duration = time.time() - tstart
    # If realtime clock initial_time_scale>1, duration will be less than ((count - 1) * period)
    assert duration > 1.8


def test_my_ping_app_graph_get_operators(ping_config_file):
    app = MyPingApp()
    app.config(ping_config_file)

    # prior to calling compose the list of operators will be empty
    assert len(app.graph.get_nodes()) == 0

    app.compose()
    graph = app.graph

    operator_list = graph.get_nodes()
    assert all(isinstance(op, Operator) for op in operator_list)
    assert set(op.name for op in graph.get_nodes()) == {"tx", "mx", "rx"}


def test_my_ping_app_graph_get_root_operators(ping_config_file):
    app = MyPingApp()
    app.config(ping_config_file)
    app.compose()
    graph = app.graph

    root_ops = graph.get_root_nodes()
    assert len(root_ops) == 1
    assert all(isinstance(op, Operator) for op in root_ops)
    assert root_ops[0].name == "tx"


def test_my_ping_app_graph_get_next_nodes(ping_config_file):
    app = MyPingApp()
    app.config(ping_config_file)
    app.compose()
    graph = app.graph

    operator_dict = {op.name: op for op in graph.get_nodes()}
    assert graph.get_next_nodes(operator_dict["tx"])[0].name == "mx"
    assert graph.get_next_nodes(operator_dict["mx"])[0].name == "rx"
    assert not graph.get_next_nodes(operator_dict["rx"])


def test_my_ping_app_graph_get_previous_nodes(ping_config_file):
    app = MyPingApp()
    app.config(ping_config_file)
    app.compose()
    graph = app.graph

    operator_dict = {op.name: op for op in graph.get_nodes()}
    assert graph.get_previous_nodes(operator_dict["mx"])[0].name == "tx"
    assert graph.get_previous_nodes(operator_dict["rx"])[0].name == "mx"
    assert not graph.get_previous_nodes(operator_dict["tx"])


def test_my_ping_app_graph_is_root_is_leaf(ping_config_file):
    app = MyPingApp()
    app.config(ping_config_file)
    app.compose()
    graph = app.graph

    operator_dict = {op.name: op for op in graph.get_nodes()}

    assert graph.is_root(operator_dict["tx"])
    assert not app.graph.is_root(operator_dict["mx"])
    assert not app.graph.is_root(operator_dict["rx"])

    assert not app.graph.is_leaf(operator_dict["tx"])
    assert not app.graph.is_leaf(operator_dict["mx"])
    assert app.graph.is_leaf(operator_dict["rx"])


def test_my_tracker_app(ping_config_file):
    app = MyPingApp()
    app.config(ping_config_file)
    tracker = app.track()
    app.run()
    tracker.print()


def test_my_tracker_logging_app(ping_config_file):
    filename = "logfile1.log"

    app = MyPingApp()
    app.config(ping_config_file)
    tracker = app.track()
    tracker.enable_logging(filename)
    app.run()
    tracker.end_logging()
    tracker.print()

    assert file_contains_string(filename, "10:")


# This test is intentionally not marked as slow to ensure every
# run of python-api test will loop an app a few times to try and catch
# intermittent seg faults
def test_my_tracker_context_manager_app(ping_config_file):
    for i in range(1000):
        app = MyPingApp()
        app.config(ping_config_file)

        with Tracker(app) as tracker:
            app.run()
            tracker.print()


def test_my_tracker_context_manager_logging_app(ping_config_file):
    filename = "logfile2.log"

    app = MyPingApp()
    app.config(ping_config_file)

    with Tracker(app, filename=filename) as tracker:
        app.run()
        tracker.print()

    assert file_contains_string(filename, "10:")
