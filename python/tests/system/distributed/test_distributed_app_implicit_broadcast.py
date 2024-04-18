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

import cupy as cp
import pytest
from env_wrapper import env_var_context
from utils import remove_ignored_errors

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment, Operator, OperatorSpec
from holoscan.operators import PingRxOp, PingTxOp

# Define versions for PingTxOp that send a tensor or dict of tensors.


class PingTensorTxOp(Operator):
    """Simple transmitter operator.

    This operator has a single output port:
        output: "out"

    On each tick, it transmits an integer to the "out" port.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        op_output.emit(cp.ones((16, 8)), "out")
        self.index += 1


class PingTensorMapTxOp(Operator):
    """Simple transmitter operator.

    This operator has a single output port:
        output: "out"

    On each tick, it transmits an integer to the "out" port.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        op_output.emit(dict(t1=cp.ones((16, 8))), "out")
        self.index += 1


# Define a dual fragment application designed to test the fix for issue 4290043:
#   - must broadcast an output (tx.out in this case)
#   - at least one target of the output being broadcast must be in another fragment
#   - must test cases emitting message types that are not a tensor map


class Fragment1(Fragment):
    def __init__(self, *args, message_type="tensor", **kwargs):
        self.message_type = message_type

        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        if self.message_type == "object":
            tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        elif self.message_type == "tensor":
            tx = PingTensorTxOp(self, CountCondition(self, 10), name="tx")
        elif self.message_type == "tensormap":
            tx = PingTensorMapTxOp(self, CountCondition(self, 10), name="tx")
        rx1 = PingRxOp(self, name="rx1")

        self.add_flow(tx, rx1)


class Fragment2(Fragment):
    def compose(self):
        rx2 = PingRxOp(self, name="rx2")

        # Add the operator (rx2) to the fragment
        self.add_operator(rx2)


class MyPingApp(Application):
    def __init__(self, *args, message_type="tensor", **kwargs):
        self.message_type = message_type

        super().__init__(*args, **kwargs)

    def compose(self):
        fragment1 = Fragment1(self, name="fragment1", message_type=self.message_type)
        fragment2 = Fragment2(self, name="fragment2")

        # Connect the two fragments (tx.out -> rx.in)
        # We can skip the "out" and "in" suffixes, as they are the default
        self.add_flow(fragment1, fragment2, {("tx", "rx2")})


# define the number of messages to send
NUM_MSGS = 10


def launch_app(message_type):
    env_var_settings = {
        # set the recession period to 10 ms to reduce debug messages
        ("HOLOSCAN_CHECK_RECESSION_PERIOD_MS", "10"),
        # set the max duration to 10s to have enough time to run the test
        # (connection time takes ~5 seconds)
        ("HOLOSCAN_MAX_DURATION_MS", "10000"),
        # set the stop on deadlock timeout to 5s to have enough time to run the test
        ("HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT", "5000"),
    }

    with env_var_context(env_var_settings):
        app = MyPingApp(message_type=message_type)
        app.run()


@pytest.mark.parametrize("message_type", ["object", "tensor", "tensormap"])
def test_distributed_implicit_broadcast_app(message_type, capfd):
    global NUM_MSGS
    # minimal app to test the fix for issue 4290043
    launch_app(message_type)

    # assert that no errors were logged
    captured = capfd.readouterr()
    # avoid catching the expected error message
    # : "error handling callback was invoked with status -25 (Connection reset by remote peer)"
    captured_error = captured.err.replace("error handling callback", "ucx handling callback")
    assert "error" not in remove_ignored_errors(captured_error)
    assert "Exception occurred" not in captured_error

    # assert that the expected number of messages were received
    # (`2 * num_messages` because there are two receivers)
    expected_num_messages = 2 * NUM_MSGS
    assert captured.out.count("Rx message value") == expected_num_messages


if __name__ == "__main__":
    launch_app()
