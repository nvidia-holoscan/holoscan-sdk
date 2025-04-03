"""
SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import time

import cupy as cp
import numpy as np
import pytest

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec, Tracker
from holoscan.operators import PingTensorRxOp
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
    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

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
    def __init__(self, fragment, *args, **kwargs):
        # If `self.multiplier` is set here (e.g., `self.multiplier = 4`), then
        # the default value by `param()` in `setup()` will be ignored.
        # (you can just call `spec.param("multiplier")` in `setup()` to use the
        # default value)
        #
        # self.multiplier = 4
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

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
    def __init__(self, fragment, *args, use_new_receivers=True, **kwargs):
        self.count = 1
        self.use_new_receivers = use_new_receivers
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        if self.use_new_receivers:
            spec.input("receivers", size=IOSpec.ANY_SIZE)
        else:
            spec.param("receivers", kind="receivers")

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers")
        assert values is not None
        print(f"received message {self.count}")
        self.count += 1


class MyPingApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        period=None,
        explicitly_set_connectors=False,
        use_new_receivers=True,
        **kwargs,
    ):
        self.count = count
        self.period = period
        self.explicitly_set_connectors = explicitly_set_connectors
        self.use_new_receivers = use_new_receivers
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_args = ()
        if self.period:
            tx_args += (PeriodicCondition(self, recess_period=self.period),)
        tx = PingTxOp(
            self,
            CountCondition(self, self.count),
            *tx_args,
            explicitly_set_connectors=self.explicitly_set_connectors,
            name="tx",
        )
        mx = PingMiddleOp(self, self.from_config("mx"), name="mx")
        rx = PingRxOp(self, use_new_receivers=self.use_new_receivers, name="rx")
        self.add_flow(tx, mx, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx, rx, {("out1", "receivers"), ("out2", "receivers")})


def file_contains_string(filename, string):
    try:
        with open(filename) as f:
            return string in f.read()

    except FileNotFoundError:
        return False


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app(ping_config_file, use_new_receivers, capfd):
    count = 10
    app = MyPingApp(count=count, use_new_receivers=use_new_receivers)
    app.config(ping_config_file)
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_periodic_manual_clock(ping_config_file, use_new_receivers, capfd):
    count = 10
    app = MyPingApp(
        count=count, period=datetime.timedelta(seconds=1), use_new_receivers=use_new_receivers
    )
    app.config(ping_config_file)
    tstart = time.time()
    app.scheduler(GreedyScheduler(app, clock=ManualClock(app)))
    app.run()
    duration = time.time() - tstart
    # If using manual clock duration should be much less than (count - 1) seconds since
    # period is not respected.
    assert duration < 9

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_periodic_realtime_clock(ping_config_file, use_new_receivers, capfd):
    count = 10
    app = MyPingApp(
        count=count,
        period=datetime.timedelta(milliseconds=100),
        use_new_receivers=use_new_receivers,
    )
    app.config(ping_config_file)
    tstart = time.time()
    app.scheduler(GreedyScheduler(app, clock=RealtimeClock(app)))
    app.run()
    duration = time.time() - tstart
    # If realtime clock, duration will be > ((count - 1) * period)
    assert duration > 0.9

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_periodic_scaled_realtime_clock(ping_config_file, use_new_receivers, capfd):
    count = 10
    app = MyPingApp(
        count=count,
        period=datetime.timedelta(milliseconds=100),
        use_new_receivers=use_new_receivers,
    )
    app.config(ping_config_file)
    tstart = time.time()
    app.scheduler(GreedyScheduler(app, clock=RealtimeClock(app, initial_time_scale=5.0)))
    app.run()
    duration = time.time() - tstart
    # If realtime clock initial_time_scale>1, duration will be less than ((count - 1) * period)
    assert duration < 0.9

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_periodic_scaled_realtime_clock2(ping_config_file, use_new_receivers, capfd):
    count = 10
    app = MyPingApp(
        count=count,
        period=datetime.timedelta(milliseconds=100),
        use_new_receivers=use_new_receivers,
    )
    app.config(ping_config_file)
    tstart = time.time()
    app.scheduler(GreedyScheduler(app, clock=RealtimeClock(app, initial_time_scale=0.5)))
    app.run()
    duration = time.time() - tstart
    # If realtime clock initial_time_scale>1, duration will be less than ((count - 1) * period)
    assert duration > 1.8

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_graph_get_operators(ping_config_file, use_new_receivers):
    app = MyPingApp(use_new_receivers=use_new_receivers)
    app.config(ping_config_file)

    # prior to calling compose the list of operators will be empty
    assert len(app.graph.get_nodes()) == 0

    app.compose()
    graph = app.graph

    operator_list = graph.get_nodes()
    assert all(isinstance(op, Operator) for op in operator_list)
    assert set(op.name for op in graph.get_nodes()) == {"tx", "mx", "rx"}


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_graph_get_root_operators(ping_config_file, use_new_receivers):
    app = MyPingApp(use_new_receivers=use_new_receivers)
    app.config(ping_config_file)
    app.compose()
    graph = app.graph

    root_ops = graph.get_root_nodes()
    assert len(root_ops) == 1
    assert all(isinstance(op, Operator) for op in root_ops)
    assert root_ops[0].name == "tx"


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_graph_get_next_nodes(ping_config_file, use_new_receivers):
    app = MyPingApp(use_new_receivers=use_new_receivers)
    app.config(ping_config_file)
    app.compose()
    graph = app.graph

    operator_dict = {op.name: op for op in graph.get_nodes()}
    assert graph.get_next_nodes(operator_dict["tx"])[0].name == "mx"
    assert graph.get_next_nodes(operator_dict["mx"])[0].name == "rx"
    assert not graph.get_next_nodes(operator_dict["rx"])


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_graph_get_previous_nodes(ping_config_file, use_new_receivers):
    app = MyPingApp(use_new_receivers=use_new_receivers)
    app.config(ping_config_file)
    app.compose()
    graph = app.graph

    operator_dict = {op.name: op for op in graph.get_nodes()}
    assert graph.get_previous_nodes(operator_dict["mx"])[0].name == "tx"
    assert graph.get_previous_nodes(operator_dict["rx"])[0].name == "mx"
    assert not graph.get_previous_nodes(operator_dict["tx"])


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_ping_app_graph_is_root_is_leaf(ping_config_file, use_new_receivers):
    app = MyPingApp(use_new_receivers=use_new_receivers)
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


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_tracker_app(ping_config_file, use_new_receivers, capfd):
    count = 10
    app = MyPingApp(count=count, use_new_receivers=use_new_receivers)
    app.config(ping_config_file)
    tracker = app.track()
    app.run()
    tracker.print()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    assert f"received message {count}" in captured.out
    assert f"received message {count + 1}" not in captured.out


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_tracker_logging_app(ping_config_file, use_new_receivers, capfd):
    count = 10
    filename = "logfile1.log"

    app = MyPingApp(count=count, use_new_receivers=use_new_receivers)
    app.config(ping_config_file)
    tracker = app.track()
    tracker.enable_logging(filename)
    app.run()
    tracker.end_logging()
    tracker.print()

    assert file_contains_string(filename, "10:")

    # Assert that flow tracking output was printed and number of recorded
    # messages was as expected.
    captured = capfd.readouterr()

    assert "Data Flow Tracking Results" in captured.out
    assert f"Number of messages: {2 * count}" in captured.out
    assert f"tx->out1: {count}" in captured.out
    assert f"tx->out2: {count}" in captured.out


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_tracker_context_manager_app(ping_config_file, use_new_receivers, capfd):
    for _ in range(100):
        app = MyPingApp(use_new_receivers=use_new_receivers)
        app.config(ping_config_file)

        with Tracker(app) as tracker:
            app.run()
            tracker.print()

        # Capturing stdout/stderr to suppress extremely verbose output if the test suite is
        # run with the `-s` flag. Calling readouterr() here to flush the captured messages
        # after each run.
        captured = capfd.readouterr()
        assert "Path 1: tx,mx,rx" in captured.out


@pytest.mark.parametrize("is_limited_tracking", [True, False])
def test_tracker_with_limited_tracking(ping_config_file, is_limited_tracking, capfd):
    app = MyPingApp(use_new_receivers=True)
    app.config(ping_config_file)

    with Tracker(app, is_limited_tracking=is_limited_tracking) as tracker:
        app.run()
        tracker.print()

        captured = capfd.readouterr()
        if is_limited_tracking:
            assert "Path 1: tx,rx" in captured.out
        else:
            assert "Path 1: tx,mx,rx" in captured.out


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_my_tracker_context_manager_logging_app(ping_config_file, use_new_receivers, capfd):
    count = 10
    filename = "logfile2.log"

    app = MyPingApp(count=count, use_new_receivers=use_new_receivers)
    app.config(ping_config_file)

    with Tracker(app, filename=filename) as tracker:
        app.run()
        tracker.print()

    assert file_contains_string(filename, "10:")

    # Assert that flow tracking output was printed and number of recorded
    # messages was as expected.
    captured = capfd.readouterr()

    assert "Data Flow Tracking Results" in captured.out
    assert f"Number of messages: {2 * count}" in captured.out
    assert f"tx->out1: {count}" in captured.out
    assert f"tx->out2: {count}" in captured.out


class ThreeRxOp(Operator):
    """Receiver with three inputs."""

    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.input("in3")

    def compute(self, op_input, op_output, context):
        value1 = op_input.receive("in1")
        value2 = op_input.receive("in2")
        value3 = op_input.receive("in3")

        # Assertions below are specific to connections made from PingTxOp as in MyPingApp2
        assert value1 != value2
        assert value2 == value3

        self.index += 1
        print(f"received: {self.index}", file=sys.stdout)


class MyPingApp2(Application):
    """App created to replicate a reported bug in add_flow behavior.

    The `multiple_add_flow_calls=True` case corresponds to the case reported in NVBUG 4260969.
    """

    def __init__(self, *args, count=10, muliple_add_flow_calls=False, use_add_arg=False, **kwargs):
        self.count = count
        self.muliple_add_flow_calls = muliple_add_flow_calls
        self.use_add_arg = use_add_arg
        super().__init__(*args, **kwargs)

    def compose(self):
        if self.use_add_arg:
            tx = PingTxOp(self, name="tx")
            tx.add_arg(CountCondition(self, self.count))
        else:
            tx = PingTxOp(self, CountCondition(self, self.count), name="tx")
        rx = ThreeRxOp(self, name="rx")
        # the following ways of calling add_flow should be equivalent
        if self.muliple_add_flow_calls:
            self.add_flow(tx, rx, {("out1", "in1"), ("out2", "in2")})
            self.add_flow(tx, rx, {("out2", "in3")})
        else:
            self.add_flow(tx, rx, {("out1", "in1"), ("out2", "in2"), ("out2", "in3")})


@pytest.mark.parametrize("muliple_add_flow_calls", [False, True])
@pytest.mark.parametrize("use_add_arg", [False, True])
def test_my_ping_app2(muliple_add_flow_calls, use_add_arg, capfd):
    app = MyPingApp2(muliple_add_flow_calls=muliple_add_flow_calls, use_add_arg=use_add_arg)
    app.run()

    captured = capfd.readouterr()
    assert captured.out.count("received: 10") == 1
    assert captured.out.count("received: 11") == 0


class PyTensorSourceOp(Operator):
    """Simple transmitter operator that emits a CuPy array via the holoscan::Tensor emitter.

    It is expected that a wrapped C++ Operator like PingTensorRxOp can receive this tensor.
    """

    def __init__(self, fragment, *args, on_device=True, **kwargs):
        self.on_device = on_device
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        if self.on_device:
            data = cp.arange(10000, dtype=cp.int16)
        else:
            data = np.arange(10000, dtype=cp.int16)

        # Note: This emitter actually emits a TensorMap with a single tensor, not a plain
        # std::shared_ptr<holoscan::Tensor>. However, for a TensorMap with a single tensor
        # both of the following ways of receiving the tensor will work from a C++ operator's
        # compute method:
        #   - auto maybe_tensormap = receive<TensorMap>(port_name);
        #   - auto maybe_tensor = receive<std::shared_ptr<holoscan::Tensor>>(port_name);
        op_output.emit(data, "out", emitter_name="holoscan::Tensor")


class MyPingPythonTensorCppInteropApp(Application):
    """Ping app that sends Python tensors emitted with emitter_name="holoscan::Tensor" to a wrapped
    C++ operator.
    """

    count = 10

    def compose(self):
        tx = PyTensorSourceOp(self, CountCondition(self, self.count, name="tx_count"), name="tx")
        rx = PingTensorRxOp(self, name="rx")
        self.add_flow(tx, rx)


def test_my_ping_python_tensor_cpp_interop_app(capfd):
    """Verify that Python array object emitted as holoscan::Tensor can be received by a C++
    operator.
    """
    count = 10
    app = MyPingPythonTensorCppInteropApp()
    app.count = count
    app.run()

    captured = capfd.readouterr()
    assert captured.err.count(f"rx received message {count}") == 1
    assert captured.err.count(f"rx received message {count + 1}") == 0


class CustomRxOp(Operator):
    """Simple receiver that prints the type of the received data."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")
        print(f"type(data) received = {type(data)}")


class MyPingPythonCppEmitPythonReceive(Application):
    """Ping app that sends Python tensors emitted with emitter_name="holoscan::Tensor" to a native
    Python operator.
    """

    count = 10
    on_device = True

    def compose(self):
        tx = PyTensorSourceOp(
            self,
            CountCondition(self, self.count, name="tx_count"),
            on_device=self.on_device,
            name="tx",
        )
        rx = CustomRxOp(self, name="rx")
        self.add_flow(tx, rx)


@pytest.mark.parametrize("on_device", [True, False])
def test_my_ping_python_cpp_emit_python_receive(on_device, capfd):
    """Verify that a Python array object emitted with emitter_name="holoscan::Tensor" is received
    as a CuPy array if it was a device array or a NumPy array if it was a host array.
    """
    count = 10
    app = MyPingPythonCppEmitPythonReceive()
    app.count = count
    app.on_device = on_device
    app.run()

    captured = capfd.readouterr()
    if on_device:
        assert captured.out.count("type(data) received = <class 'cupy.ndarray'>") == count
    else:
        assert captured.out.count("type(data) received = <class 'numpy.ndarray'>") == count
