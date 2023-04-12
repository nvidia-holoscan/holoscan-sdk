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

from holoscan.conditions import (
    BooleanCondition,
    CountCondition,
    DownstreamMessageAffordableCondition,
    MessageAvailableCondition,
)
from holoscan.core import Application, Condition, ConditionType, Operator
from holoscan.gxf import Entity, GXFCondition


class TestBooleanCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        cond = BooleanCondition(
            fragment=app,
            name="boolean",
            enable_tick=True,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::BooleanSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_enable_tick(self, app, capfd):
        cond = BooleanCondition(
            fragment=app,
            name="boolean",
            enable_tick=True,
        )
        cond.disable_tick()
        assert not cond.check_tick_enabled()

        cond.enable_tick()
        assert cond.check_tick_enabled()

    def test_default_initialization(self, app):
        BooleanCondition(app)

    def test_positional_initialization(self, app):
        BooleanCondition(app, False, "bool")


class TestCountCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        cond = CountCondition(
            fragment=app,
            name="count",
            count=100,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::CountSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_count(self, app, capfd):
        cond = CountCondition(
            fragment=app,
            name="count",
            count=100,
        )
        cond.count = 10
        assert cond.count == 10

    def test_default_initialization(self, app):
        CountCondition(app)

    def test_positional_initialization(self, app):
        CountCondition(app, 100, "counter")


class TestDownstreamMessageAffordableCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        cond = DownstreamMessageAffordableCondition(
            fragment=app,
            name="downstream_affordable",
            min_size=10,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::DownstreamReceptiveSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        DownstreamMessageAffordableCondition(app)

    def test_positional_initialization(self, app):
        DownstreamMessageAffordableCondition(app, 4, "affordable")


class TestMessageAvailableCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        cond = MessageAvailableCondition(
            fragment=app,
            name="message_available",
            min_size=1,
            front_stage_max_size=10,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::MessageAvailableSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        MessageAvailableCondition(app)

    def test_positional_initialization(self, app):
        MessageAvailableCondition(app, 1, 4, "available")


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
            print(f"#TX{self.index}")
        elif self.index == 2:
            print(f"#T1O{self.index}")
            op_output.emit(self.index, "out1")
        elif self.index == 3:
            print(f"#T2O{self.index}")
            entity = Entity(context)
            op_output.emit(entity, "out2")
        elif self.index == 4:
            print(f"#TO{self.index}")
            op_output.emit(self.index, "out1")
            entity = Entity(context)
            op_output.emit(entity, "out2")
        else:
            print(f"#TX{self.index}")


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

        if value1 and not value2:
            print(f"#R1O{self.index}")
        elif not value1 and value2:
            print(f"#R2O{self.index}")
        elif value1 and value2:
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
    assert "".join(sequence) == "TX1RX1T1O2R1O2T2O3R2O3TO4RO4TX5RX5"

    error_msg = captured.err.lower()
    assert "error" not in error_msg
    assert "warning" not in error_msg
