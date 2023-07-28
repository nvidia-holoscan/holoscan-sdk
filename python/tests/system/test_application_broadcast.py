# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec


class TxOp(Operator):
    def __init__(self, fragment, *args, optional_output=False, **kwargs):
        # Need to call the base class constructor last
        self.optional_output = optional_output
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        if self.optional_output:
            # Receiver optional
            spec.output("series").condition(ConditionType.NONE)
        else:
            spec.output("series")

    def compute(self, op_input, op_output, context):
        op_output.emit([1, 2, 3], "series")
        print("TxOp compute complete")


class RxOp1(Operator):
    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("series")
        # optional output not used in the application
        spec.output("image_name").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        series = op_input.receive("series")
        assert series == [1, 2, 3]

        op_output.emit("image.dcm", "image_name")
        print("RxOp1 compute complete")


class RxOp2(Operator):
    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("series")
        # optional second input not used by the application
        spec.input("optional_in").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        series = op_input.receive("series")
        assert series == [1, 2, 3]

        # no connection to this optional input so will be None
        assert op_input.receive("optional_in") is None
        print("RxOp2 compute complete")


# Define a simple application broadcasting from one transmitter to multiple receivers.
class BasicBroadcastApp(Application):
    def __init__(
        self, *args, count=10, optional_output=False, additional_receivers=False, **kwargs
    ):
        self.optional_output = optional_output
        self.additional_receivers = additional_receivers
        super().__init__(*args, **kwargs)

    def compose(self):
        # transmit just once
        tx_op = TxOp(self, CountCondition(self, 1), optional_output=self.optional_output, name="tx")

        # same transmitter is connected to two receivers so a broadcast component will be used
        rx1_op = RxOp1(self, name="rx1")
        rx2_op = RxOp2(self, name="rx2")
        self.add_flow(tx_op, rx1_op, {("series", "series")})
        self.add_flow(tx_op, rx2_op, {("series", "series")})

        if self.additional_receivers:
            rx3_op = RxOp1(self, name="rx3")
            rx4_op = RxOp2(self, name="rx4")
            self.add_flow(tx_op, rx3_op, {("series", "series")})
            self.add_flow(tx_op, rx4_op, {("series", "series")})


class RxOp3(Operator):
    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("series")
        spec.input("optional_in").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        series = op_input.receive("series")
        assert series == [1, 2, 3]

        assert op_input.receive("optional_in") == "img.dcm"
        print("RxOp3 compute complete")


class MxOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("series")
        spec.output("image_name")

    def compute(self, op_input, op_output, context):
        series = op_input.receive("series")
        assert series == [1, 2, 3]

        # no connection to this optional input so will be None
        op_output.emit("img.dcm", "image_name")
        print("MxOp compute complete")


# Define a simple application broadcasting from one transmitter to multiple receivers.
class ForkJoinApp(Application):
    def __init__(self, *args, count=10, optional_output=False, **kwargs):
        self.optional_output = optional_output
        super().__init__(*args, **kwargs)

    def compose(self):
        # transmit just once
        tx_op = TxOp(self, CountCondition(self, 1), optional_output=self.optional_output, name="tx")
        mx_op = MxOp(self, name="mx")

        # output of transmitter takes three divergent paths, two of which converge to the same
        # downstream operator
        rx1_op = RxOp1(self, name="rx1")
        rx3_op = RxOp3(self, name="rx2")
        self.add_flow(tx_op, rx1_op, {("series", "series")})
        self.add_flow(tx_op, rx3_op, {("series", "series")})

        self.add_flow(tx_op, mx_op, {("series", "series")})
        self.add_flow(mx_op, rx3_op, {("image_name", "optional_in")})


@pytest.mark.parametrize("optional_output", [False, True])
@pytest.mark.parametrize("additional_receivers", [False, True])
def test_basic_broadcast_app(optional_output, additional_receivers, capfd):
    # with optional output, TxOp's output port will have ConditionType.NONE
    app = BasicBroadcastApp(
        optional_output=optional_output, additional_receivers=additional_receivers
    )
    app.run()

    # assert that all expected compute methods completed (no early termination)
    captured = capfd.readouterr()
    assert captured.out.count("TxOp compute complete") == 1

    count_expected = 2 if additional_receivers else 1
    assert captured.out.count("RxOp1 compute complete") == count_expected
    assert captured.out.count("RxOp2 compute complete") == count_expected


@pytest.mark.parametrize("optional_output", [False, True])
def test_fork_join_app(optional_output, capfd):
    """Fork/join topology test case

    Test case where a single transmitter forks into multiple paths, two of which join back
    at the same downstream operator.
    """
    app = ForkJoinApp(optional_output=optional_output)
    app.run()

    # assert that all expected compute methods completed (no early termination)
    captured = capfd.readouterr()
    assert captured.out.count("TxOp compute complete") == 1
    assert captured.out.count("MxOp compute complete") == 1
    assert captured.out.count("RxOp1 compute complete") == 1
    assert captured.out.count("RxOp3 compute complete") == 1
