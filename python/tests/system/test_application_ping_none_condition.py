"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec


class PingTxOp(Operator):
    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out1")

    def compute(self, op_input, op_output, context):
        self.index += 1
        op_output.emit(self.index, "out1")


class PingMiddleOp(Operator):
    """A Copy of PingMiddleOp but with ConditionType.NONE on the inputs."""

    def __init__(self, *args, **kwargs):
        # counter for the number of times a value was received
        self.value_count = 0
        # counter for the number of times no value was received
        self.none_count = 0

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Set ConditionType.NONE so compute will not wait for a message to be
        # available.
        spec.input("in1").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        value1 = op_input.receive("in1")
        if value1 is None:
            self.none_count += 1
            return

        # Multiply the value
        value1 *= 2

        self.value_count += 1


class MyPingApp(Application):
    def __init__(self, *args, count=10, **kwargs):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTxOp(self, CountCondition(self, self.count), name="tx")
        # receive 2x the number of transmitted messages, so 1/2 will be received as None
        mx = PingMiddleOp(
            self, CountCondition(self, 2 * self.count), self.from_config("mx"), name="mx"
        )
        self.add_flow(tx, mx, {("out1", "in1")})


def test_ping_app_none_condition():
    count = 15
    app = MyPingApp(count=count)
    app.run()
    mx_found = False
    for op in app.graph.get_nodes():
        if op.name == "mx":
            mx_found = True
            assert op.none_count == count
            assert op.value_count == count
    assert mx_found
