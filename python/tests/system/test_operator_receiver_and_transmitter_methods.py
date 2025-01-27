"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.resources import Receiver, Transmitter


class PingReceiveTestOp(Operator):
    """Simple receiver operator."""

    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in", size=IOSpec.IOSize(2))

    def compute(self, op_input, op_output, context):
        receiver = self.receiver("in")
        assert isinstance(receiver, Receiver)
        # initially will have 1 message in main stage
        assert receiver.capacity == 2
        assert receiver.size == 2
        assert receiver.back_size == 0

        value = op_input.receive("in")
        assert len(value) == 2
        print(f"Rx message value: {value}")

        # will now have 0 messages left in the main stage
        assert receiver.capacity == 2
        assert receiver.size == 0
        assert receiver.back_size == 0


class PingTransmitTestOp(Operator):
    """Simple transmitter operator."""

    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=3)

    def compute(self, op_input, op_output, context):
        transmitter = self.transmitter("out")
        assert isinstance(transmitter, Transmitter)
        assert transmitter.capacity == 3
        assert transmitter.size == 0
        # no messages will have been staged yet prior to the emit call
        assert transmitter.back_size == 0

        op_output.emit(self.index, "out")
        self.index += 1

        # calling emit puts message in the back stage
        # (later GXF call to sync_outbox will move it to the main stage)
        assert transmitter.capacity == 3
        assert transmitter.size == 0
        assert transmitter.back_size == 1


class MyRxTxTestApp(Application):
    def __init__(self, *args, count=10, **kwargs):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTransmitTestOp(self, CountCondition(self, self.count), name="tx")
        rx = PingReceiveTestOp(self, name="rx")
        self.add_flow(tx, rx)


def test_operator_receiver_and_transmitter_methods(capfd):
    count = 6
    app = MyRxTxTestApp(count=6)
    app.run()

    captured = capfd.readouterr()
    assert f"Rx message value: {(count - 1, count)}" in captured.out
