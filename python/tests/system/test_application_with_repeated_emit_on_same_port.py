"""
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.conditions import CountCondition
from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec


class PingMultiQueueTxOp(Operator):
    """Transmitter operator with a single port that has a user-defined queue size.

    This operator has a single output port:
        input: "out"

    If the specific values that are output are modified, then a corresponding change will
    have to be made to the assertions in `PingMultiQueueRxOp`.
    """

    def __init__(
        self, fragment, *args, queue_capacity=2, queue_policy=0, extra_emit=False, **kwargs
    ):
        self.index = 1
        self.queue_capacity = queue_capacity
        self.queue_policy = queue_policy
        self.extra_emit = extra_emit
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=self.queue_capacity,
            policy=self.queue_policy,
        ).condition(
            ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE,
            min_size=self.queue_capacity,
            front_stage_max_size=self.queue_capacity,
        )

    def compute(self, op_input, op_output, context):
        for i in range(self.queue_capacity):
            op_output.emit(self.index, "out")
            if self.extra_emit and i == self.queue_capacity - 1:
                # Emit the last value a second time. This will exceed the queue size, so should
                # result in a warning or error, depending on queue policy.
                op_output.emit(self.index, "out")
            self.index += 1


class PingMultiQueueRxOp(Operator):
    """Receiver operator with larger, user-defined queue size.

    This operator has a single input port:
        input: "in"

    The compute method in this test case hard-codes assertions about the expected values that
    would be received from the corresponding `PingMultiQueueTxOp`.
    """

    def __init__(
        self,
        fragment,
        *args,
        queue_capacity=2,
        queue_policy=0,
        extra_emit=False,
        **kwargs,
    ):
        # Need to call the base class constructor last
        self.queue_capacity = queue_capacity
        self.queue_policy = queue_policy
        self.extra_emit = extra_emit
        self.count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=self.queue_capacity,
            policy=self.queue_policy,
        ).condition(
            ConditionType.MESSAGE_AVAILABLE,
            min_size=self.queue_capacity,
            front_stage_max_size=self.queue_capacity,
        )

    def compute(self, op_input, op_output, context):
        offset = self.count * self.queue_capacity
        expected_values = tuple(offset + i for i in range(1, self.queue_capacity + 1))
        if self.extra_emit and self.queue_capacity > 1:  # noqa: SIM102
            if self.queue_policy == 0:
                # Policy 0 is 'pop', so the first value would have been popped from the queue.
                # For `PingMultiQueueTxOp`, the last value is repeated.
                expected_values = expected_values[1:] + (expected_values[-1],)
            # Policy 1 is 'reject' so any extra value would have been rejected instead
            # Policy 2 is 'fault' so the app will have terminated before this point

        for expected_value in expected_values:
            value = op_input.receive("in")
            assert value == expected_value
        self.count += 1
        print(f"PingMultiQueueRxOp has been called {self.count} times.")

        # any extra receive beyond the queue size will return None
        assert op_input.receive("in") is None


class MyCustomQueuePingApp(Application):
    """Ping application with custom queue size and policy.

    The purpose of this app is to test non-default condition/connector settings. Particularly,
    emit/receive will be called repeatedly on the same/input port.

    ``op_input.receive`` is always called `queue_capacity` times.

    The number of times ``op_output.emit`` is called will be equal to
    ``(queue_capacity + 1) if extra_emit else queue_capacity``

    Parameters
    ----------
    count : int
        The number of times `compute` will be called on the transmitter.
    queue_capacity : int
        A non-negative integer specifying the queue capacity. The same capacity is used for both
        the transmit and receive operators.
    queue_policy : {0, 1, 2}
        The policy to use when the queue is already full:
            - 0 == 'pop'
            - 1 == 'reject'
            - 2 == 'fault'

        The same policy is used for both the transmit and receive operators.
    extra_emit: bool
        Whether or not the transmitter calls emit an extra time after the queue is full.

    Raises
    ------
    RuntimeError
        If ``policy == 2`` (fault) and ``extra_emit == True``.
    """

    def __init__(
        self,
        *args,
        count=10,
        queue_capacity=2,
        queue_policy=0,
        extra_emit=False,
        **kwargs,
    ):
        self.count = count
        self.queue_capacity = queue_capacity
        self.queue_policy = queue_policy
        self.extra_emit = extra_emit
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingMultiQueueTxOp(
            self,
            CountCondition(self, count=self.count),
            queue_capacity=self.queue_capacity,
            queue_policy=self.queue_policy,
            extra_emit=self.extra_emit,
            name="tx",
        )
        rx = PingMultiQueueRxOp(
            self,
            queue_capacity=self.queue_capacity,
            queue_policy=self.queue_policy,
            extra_emit=self.extra_emit,
            name="rx",
        )
        self.add_flow(tx, rx)


@pytest.mark.parametrize("queue_capacity", [1, 2, 3, 100])
def test_ping_app_with_larger_queue(queue_capacity, capfd):
    """Test different queue sizes with the default policy"""
    count = 3
    app = MyCustomQueuePingApp(count=count, queue_capacity=queue_capacity, queue_policy=2)
    app.run()

    # assert that PingMultiQueueRxOp.compute was called the expected number of times
    # The compute method itself confirms the received values
    captured = capfd.readouterr()

    assert f"PingMultiQueueRxOp has been called {count} times." in captured.out


@pytest.mark.parametrize("queue_policy", [0, 1, 2])
@pytest.mark.parametrize("queue_capacity", [1, 2, 5])
def test_ping_app_with_larger_queue_and_extra_emit(queue_capacity, queue_policy, capfd):
    """Test different queue policies when there is an extra emit call on the Transmitter"""
    count = 3
    app = MyCustomQueuePingApp(
        count=count,
        queue_capacity=queue_capacity,
        queue_policy=queue_policy,
        extra_emit=True,
    )
    try:
        app.run()

        captured = capfd.readouterr()

        # Assert that `PingMultiQueueRxOp.compute` was called the expected number of times
        # (The compute method itself confirms the received values)
        assert f"PingMultiQueueRxOp has been called {count} times." in captured.out

        # queue size error was not logged
        assert "error" not in captured.err
        assert "GXF_EXCEEDING_PREALLOCATED_SIZE" not in captured.err
    except RuntimeError as err:
        if queue_policy == 2:
            # policy 2 = fault, so app will terminate on the first compute call to
            # PingMultiQueueTxOp when `extra_emit=True` as above.
            captured = capfd.readouterr()

            # verify that an error about exceeding the queue size was logged
            assert "error" in captured.err
            assert "GXF_EXCEEDING_PREALLOCATED_SIZE" in captured.err
        else:
            raise RuntimeError(f"unexpected Runtime error: {err}") from err
