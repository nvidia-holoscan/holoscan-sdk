"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec


class TimestampTxOp(Operator):
    """Simple transmitter operator.

    On each tick, it transmits an integer to the "out" port.

    **==Named Outputs==**

        out : int
            An index value that increments by one on each call to `compute`. The starting value is
            1.
    """

    def __init__(self, fragment, *args, queue_size=1, initial_timestamp=555666777, **kwargs):
        self.index = 0
        self.queue_size = queue_size
        self.initial_timestamp = initial_timestamp
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=self.queue_size)

    def compute(self, op_input, op_output, context):
        print(f"Sending message: {self.index}")

        op_output.emit(self.index, "out", acq_timestamp=self.initial_timestamp + self.index)
        if self.queue_size > 1:
            for i in range(1, self.queue_size):
                op_output.emit(
                    self.index, "out", acq_timestamp=self.initial_timestamp + self.index + 1000 * i
                )
        self.index += 1


class TimestampRxOp(Operator):
    def __init__(self, fragment, *args, queue_size=1, initial_timestamp=555666777, **kwargs):
        self.initial_timestamp = initial_timestamp
        self.queue_size = queue_size
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in", size=IOSpec.IOSize(self.queue_size))

    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")
        if self.queue_size > 1:
            assert len(message) == self.queue_size
            for n in range(self.queue_size):
                assert message[n] == self.index

            print(f"Timestamp Testing ping: {message[0]}", flush=True)
        else:
            print(f"Timestamp Testing ping: {message}", flush=True)

        # can get the timestamp emitted by TimestampTxOp if it is needed
        timestamp = op_input.get_acquisition_timestamp("in")
        assert timestamp is not None
        assert isinstance(timestamp, int)
        assert timestamp == self.initial_timestamp + self.index

        # can receive as a vector of timestamps
        timestamps = op_input.get_acquisition_timestamps("in")
        assert len(timestamps) == self.queue_size
        for i in range(self.queue_size):
            assert isinstance(timestamps[i], int)
            assert timestamps[i] == self.initial_timestamp + self.index + 1000 * i

        self.index += 1


class TimestampTestApp(Application):
    def __init__(self, *args, count=5, period_ns=10_000_000, queue_size=1, **kwargs):
        self.count = count
        self.period_ns = period_ns
        self.queue_size = queue_size
        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        # PeriodicCondition is used so that each subsequent message is
        # sent only after a period of 10 milliseconds has elapsed.
        tx = TimestampTxOp(
            self,
            CountCondition(self, self.count),
            PeriodicCondition(self, self.period_ns),
            queue_size=self.queue_size,
            name="tx",
        )
        rx = TimestampRxOp(self, queue_size=self.queue_size, name="rx")

        # Connect the operators into the workflow:  tx -> rx
        self.add_flow(tx, rx)


@pytest.mark.parametrize("queue_size", [1, 2, 10])
def test_timestamp_emit_receive(queue_size, capfd):
    # Test single element and vector case where multiple messages are sent on the same port
    count = 5
    app = TimestampTestApp(count=count, queue_size=queue_size)
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    assert f"Timestamp Testing ping: {count - 1}" in captured.out
    assert f"Timestamp Testing ping: {count}" not in captured.out
    assert "error" not in captured.err


class TimestampRxMultiReceiverOp(Operator):
    def __init__(self, fragment, *args, initial_timestamps=(0, 0, 0), **kwargs):
        self.index = 0
        self.initial_timestamps = initial_timestamps
        self.expected_size = len(initial_timestamps)
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")
        assert len(message) == self.expected_size
        for n in range(self.expected_size):
            assert message[n] == self.index
        print(f"Timestamp Testing ping: {message[0]}", flush=True)

        # can get the timestamp emitted by TimestampTxOp if it is needed
        timestamp = op_input.get_acquisition_timestamp("in")
        assert timestamp is not None
        assert isinstance(timestamp, int)
        assert timestamp == self.initial_timestamps[0] + self.index

        # can receive as a vector of timestamps
        timestamps = op_input.get_acquisition_timestamps("in")
        assert len(timestamps) == self.expected_size
        for i in range(self.expected_size):
            assert isinstance(timestamps[i], int)
            assert timestamps[i] == self.initial_timestamps[i] + self.index

        self.index += 1


class TimestampTestMultiReceiverApp(Application):
    def __init__(self, *args, count=5, period_ns=10_000_000, **kwargs):
        self.count = count
        self.period_ns = period_ns
        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        # PeriodicCondition is used so that each subsequent message is
        # sent only after a period of 10 milliseconds has elapsed.
        initial_timestamps = [0, 5000, 10000]
        tx_ops = []
        for i, t in enumerate(initial_timestamps):
            tx_ops.append(
                TimestampTxOp(
                    self,
                    CountCondition(self, self.count),
                    PeriodicCondition(self, self.period_ns),
                    initial_timestamp=t,
                    queue_size=1,
                    name=f"tx{i}",
                )
            )

        rx = TimestampRxMultiReceiverOp(
            self,
            initial_timestamps=initial_timestamps,
            name="rx",
        )

        # Connect the operators into the workflow
        for i in range(len(tx_ops)):
            self.add_flow(tx_ops[i], rx, {("out", "in")})


def test_timestamp_emit_multi_receiver(capfd):
    # Test the case where multiple operators sennd a single message to the same port
    count = 5
    app = TimestampTestMultiReceiverApp(count=count)
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    assert f"Timestamp Testing ping: {count - 1}" in captured.out
    assert f"Timestamp Testing ping: {count}" not in captured.out
    assert "error" not in captured.err
