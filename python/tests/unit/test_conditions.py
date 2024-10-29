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

from holoscan.conditions import (
    AsynchronousCondition,
    AsynchronousEventState,
    BooleanCondition,
    CountCondition,
    CudaBufferAvailableCondition,
    CudaEventCondition,
    CudaStreamCondition,
    DownstreamMessageAffordableCondition,
    ExpiringMessageAvailableCondition,
    MessageAvailableCondition,
    PeriodicCondition,
)
from holoscan.core import Application, Condition, ConditionType, Operator
from holoscan.gxf import Entity, GXFCondition
from holoscan.resources import RealtimeClock


class TestBooleanCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "boolean"
        cond = BooleanCondition(fragment=app, name=name, enable_tick=True)
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::BooleanSchedulingTerm"

        assert f"""
name: {name}
fragment: ""
args:
  - name: enable_tick
    type: bool
    value: true
""" in repr(cond)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_enable_tick(self, app, capfd):
        cond = BooleanCondition(fragment=app, name="boolean", enable_tick=True)
        cond.disable_tick()
        assert not cond.check_tick_enabled()

        cond.enable_tick()
        assert cond.check_tick_enabled()

    def test_default_initialization(self, app):
        BooleanCondition(app)

    def test_positional_initialization(self, app):
        BooleanCondition(app, False, "bool")


@pytest.mark.parametrize("name", ["READY", "WAIT", "EVENT_WAITING", "EVENT_DONE", "EVENT_NEVER"])
def test_aynchronous_event_state_enum(name):
    assert hasattr(AsynchronousEventState, name)


class TestAsynchronousCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "async"
        cond = AsynchronousCondition(fragment=app, name=name)
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::AsynchronousSchedulingTerm"

        assert f"""
name: {name}
fragment: ""
args:
  []
""" in repr(cond)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_event_state(self, app, capfd):
        cond = AsynchronousCondition(fragment=app, name="async")
        assert cond.event_state == AsynchronousEventState.READY
        cond.event_state = AsynchronousEventState.EVENT_NEVER

    def test_default_initialization(self, app):
        AsynchronousCondition(app)


class TestCountCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "count"
        cond = CountCondition(fragment=app, name=name, count=100)
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::CountSchedulingTerm"

        assert f"""
name: {name}
fragment: ""
args:
  - name: count
    type: int64_t
    value: 100
""" in repr(cond)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_count(self, app, capfd):
        cond = CountCondition(fragment=app, name="count", count=100)
        cond.count = 10
        assert cond.count == 10

    def test_default_initialization(self, app):
        CountCondition(app)

    def test_positional_initialization(self, app):
        CountCondition(app, 100, "counter")


class TestDownstreamMessageAffordableCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "downstream_affordable"
        cond = DownstreamMessageAffordableCondition(fragment=app, name=name, min_size=10)
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::DownstreamReceptiveSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

        assert f"""
name: {name}
fragment: ""
args:
  - name: min_size
    type: uint64_t
    value: 10
""" in repr(cond)

    def test_default_initialization(self, app):
        DownstreamMessageAffordableCondition(app)

    def test_positional_initialization(self, app):
        DownstreamMessageAffordableCondition(app, 4, "affordable")


class TestMessageAvailableCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "message_available"
        cond = MessageAvailableCondition(
            fragment=app, name=name, min_size=1, front_stage_max_size=10
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::MessageAvailableSchedulingTerm"

        assert f"""
name: {name}
fragment: ""
args:
  - name: min_size
    type: uint64_t
    value: 1
  - name: front_stage_max_size
    type: uint64_t
    value: 10
""" in repr(cond)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        MessageAvailableCondition(app)

    def test_positional_initialization(self, app):
        MessageAvailableCondition(app, 1, 4, "available")


class TestExpiringMessageAvailableCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "expiring_message"
        cond = ExpiringMessageAvailableCondition(
            fragment=app, name=name, max_batch_size=1, max_delay_ns=10
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::ExpiringMessageAvailableSchedulingTerm"

        # verify that name is as expected and that clock argument was automatically added
        assert f"""
name: {name}
fragment: ""
args:
  - name: clock
    type: std::shared_ptr<Resource>
spec:
""" in repr(cond)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        ExpiringMessageAvailableCondition(app, 1, 4)

    def test_positional_initialization(self, app):
        ExpiringMessageAvailableCondition(app, 1, 4, RealtimeClock(app, name="clock"), "expiring")


class TestPeriodicCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "periodic"
        cond = PeriodicCondition(fragment=app, name=name, recess_period=100)
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::PeriodicSchedulingTerm"

        # args empty here because recess_period is passed to the constructor directly, not as Arg
        assert f"""
name: {name}
fragment: ""
args:
  []
""" in repr(cond)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    @pytest.mark.parametrize(
        "period",
        [
            1000,
            datetime.timedelta(minutes=1),
            datetime.timedelta(seconds=1),
            datetime.timedelta(milliseconds=1),
            datetime.timedelta(microseconds=1),
        ],
    )
    def test_periodic_constructors(self, app, capfd, period):
        cond = PeriodicCondition(fragment=app, name="periodic", recess_period=period)
        expected_ns = (
            period if isinstance(period, int) else int(period.total_seconds() * 1_000_000_000)
        )
        assert cond.recess_period_ns() == expected_ns

    @pytest.mark.parametrize(
        "period",
        [
            1000,
            datetime.timedelta(minutes=1),
            datetime.timedelta(seconds=1),
            datetime.timedelta(milliseconds=1),
            datetime.timedelta(microseconds=1),
        ],
    )
    def test_recess_period_method(self, app, capfd, period):
        cond = PeriodicCondition(fragment=app, name="periodic", recess_period=1)
        cond.recess_period(period)
        expected_ns = (
            period if isinstance(period, int) else int(period.total_seconds() * 1_000_000_000)
        )
        assert cond.recess_period_ns() == expected_ns

    def test_positional_initialization(self, app):
        PeriodicCondition(app, 100000, "periodic")

    def test_invalid_recess_period_type(self, app):
        with pytest.raises(TypeError):
            PeriodicCondition(app, recess_period="100s", name="periodic")


class TestCudaEventCondition:
    def test_kwarg_based_initialization(self, app):  # , capfd):
        name = "cuda_event_condition"
        event_name = "cuda_event"
        cond = CudaEventCondition(
            fragment=app,
            event_name=event_name,
            name=name,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::CudaEventSchedulingTerm"

        assert f"""
name: {name}
fragment: ""
args:
  - name: event_name
    type: std::string
    value: {event_name}
""" in repr(cond)

        # assert no warnings or errors logged
        # captured = capfd.readouterr()
        # assert "error" not in captured.err
        # assert "warning" not in captured.err
        pass

    def test_default_initialization(self, app):
        CudaEventCondition(app)


class TestCudaStreamCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "cuda_stream_condition"
        cond = CudaStreamCondition(fragment=app, name=name)
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::CudaStreamSchedulingTerm"

        assert f"""
name: {name}
fragment: ""
args:
  []
""" in repr(cond)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        CudaStreamCondition(app)


class TestCudaBufferAvailableCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "cuda_buffer_available_condition"
        cond = CudaBufferAvailableCondition(fragment=app, name=name)
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::CudaBufferAvailableSchedulingTerm"

        assert f"""
name: {name}
fragment: ""
args:
  []
""" in repr(cond)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        CudaBufferAvailableCondition(app)


####################################################################################################
# Test Ping app with no conditions on Rx operator
####################################################################################################


class PingTxOpNoCondition(Operator):
    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.output("out1")
        spec.output("out2")

    def compute(self, op_input, op_output, context):
        self.index += 1
        if self.index == 1:
            print(f"#TX{self.index}")  # no emit
        elif self.index == 2:
            print(f"#T1O{self.index}")  # emit only out1
            op_output.emit(self.index, "out1")
        elif self.index == 3:
            print(f"#T2O{self.index}")  # emit only out2 (Entity object)
            entity = Entity(context)
            op_output.emit(entity, "out2")
        elif self.index == 4:
            print(f"#TO{self.index}")  # emit both out1 and out2 (out2 is Entity object)
            op_output.emit(self.index, "out1")
            entity = Entity(context)
            op_output.emit(entity, "out2")
        else:
            print(f"#TX{self.index}")  # no emit


class PingRxOpNoInputCondition(Operator):
    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        # No input condition
        spec.input("in1").condition(ConditionType.NONE)
        spec.input("in2").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        self.index += 1
        value1 = op_input.receive("in1")
        value2 = op_input.receive("in2")

        # Since value can be an empty dict, we need to check for None explicitly
        if value1 is not None and value2 is None:
            print(f"#R1O{self.index}")
        elif value1 is None and value2 is not None:
            print(f"#R2O{self.index}")
        elif value1 is not None and value2 is not None:
            print(f"#RO{self.index}")
        else:
            print(f"#RX{self.index}")


class PingRxOpNoInputConditionApp(Application):
    def compose(self):
        tx = PingTxOpNoCondition(self, CountCondition(self, 5), name="tx")
        rx = PingRxOpNoInputCondition(self, CountCondition(self, 5), name="rx")
        self.add_flow(tx, rx, {("out1", "in1"), ("out2", "in2")})


def test_ping_no_input_condition(capfd):
    app = PingRxOpNoInputConditionApp()
    app.run()

    captured = capfd.readouterr()

    sequence = (line[1:] if line.startswith("#") else "" for line in captured.out.splitlines())
    # The following sequence is expected:
    #   TX1->RX1, T1O2-> R1O2, T2O3->R2O3, TO4->RO4, TX5->RX5
    assert "".join(sequence) == "TX1RX1T1O2R1O2T2O3R2O3TO4RO4TX5RX5"

    error_msg = captured.err.lower()
    assert "error" not in error_msg
    assert "warning" not in error_msg
