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

import math

import pytest

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler

"""
Note: This file contains a modified version of the app defined in
examples/multi_branch_pipeline. This variant is modified to test multiple different
ways of configuring a port's queue policy and to verify expected warning behavior.
"""


class PingTxOp(Operator):
    """Simple transmitter operator.

    This operator has:
        outputs: "out"

    On each tick, it transmits an integer on the "out" port. The value on the first call to
    compute is equal to `initial_value` and it increments by `increment` on each subsequent
    call.
    """

    def __init__(self, fragment, *args, initial_value=0, increment=0, **kwargs):
        self.count = 0
        self.initial_value = initial_value
        self.increment = increment

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Note: Setting ConditionType.NONE overrides the default of
        #   ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE. This means that the operator will be
        #   triggered regardless of whether any operators connected downstream have space in their
        #   queues.
        spec.output("out").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        value = self.initial_value + self.count * self.increment
        op_output.emit(value, "out")
        self.count += 1


class IncrementOp(Operator):
    """Add a fixed value to the input and transmit the result.

    `policy_set_mode` is used to configure how the setup method configures the receiver queue
    policy. It can be set to:

        - 'input' : pass policy as kwarg to OperatorSpec.input method
        - 'connector' : use the connector method to explicitly define a connector with policy arg
        - 'both' : test warning if both of the above are done simultaneously
        - None : do not set policy at all

    """

    def __init__(self, fragment, *args, increment=0, policy_set_mode="input", **kwargs):
        self.increment = increment

        # used to configure how setup method deals with policy
        # can be 'input', 'connector', 'both' or None
        self.policy_set_mode = policy_set_mode

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Setup the input and output ports with custom settings on the input port.

        Notes
        -----
        For policy:
          - IOSpec.QueuePolicy.POP = pop the oldest value in favor of the new one when the queue
            is full
          - IOSpec.QueuePolicy.REJECT = reject the new value when the queue is full
          - IOSpec.QueuePolicy.FAULT = fault if queue is full (default)

        For capacity:
          When capacity > 1, even once messages stop arriving, this entity will continue to
          call ``compute`` for each remaining item in the queue.

        The ``condition`` method call here is the same as the default setting, and is shown
        only for completeness. `min_size` = 1 means that this operator will not call compute
        unless there is at least one message in the queue.

        One could also set the receiver's capacity and policy via the connector method:

            .connector(
                IOSpec.ConnectorType.DOUBLE_BUFFER,
                capacity=1,
                policy=1,  # 1 = reject
            )

        but that is less flexible as `IOSpec::ConnectorType::kDoubleBuffer` is appropriate for
        within-fragment connections, but will not work if the operator was connected to a
        different fragment. By passing the capacity and policy as arguments to the
        `OperatorSpec::input` method, the SDK can still select the appropriate default receiver
        type depending on whether the connection is within-fragment or across fragments (or
        whether an annotated variants of the receiver class is needed for data flow tracking).

        Note that if policy was not specific in the `input` call as done here, it would also be
        possible for an application author to set the `policy` after the operator is constructed
        by using the `Operator.queue_policy` property (see commented code in
        `MultiRateApp.compose` below).
        """
        if self.policy_set_mode == "input":
            spec.input("in", policy=IOSpec.QueuePolicy.REJECT)
        elif self.policy_set_mode == "connector":
            spec.input("in").connector(
                IOSpec.ConnectorType.DOUBLE_BUFFER,
                capacity=1,
                policy=1,  # 1 = reject
            )
        elif self.policy_set_mode == "both":
            # this should produce a warning because value passed to input will be ignored
            # due to the explicitly provided connector
            spec.input("in", policy=IOSpec.QueuePolicy.FAULT).connector(
                IOSpec.ConnectorType.DOUBLE_BUFFER,
                capacity=1,
                policy=1,  # 1 = reject
            )
        elif self.policy_set_mode is None:
            spec.input("in")
        else:
            raise ValueError(f"unrecognized {self.policy_set_mode=}")

        spec.output("out")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        new_value = value + self.increment
        op_output.emit(new_value, "out")


class PingRxOp(Operator):
    """Simple (multi)-receiver operator.

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port.
    """

    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        # In this case, nothing will be printed until all messages have
        # been received.
        value = op_input.receive("in")
        print(f"receiver '{self.name}' received value: {value}")


# Now define a simple application using the operators defined above
class MultiRateApp(Application):
    """This application has a single transmitter connected to two parallel branches

    The geometry of the application is as shown below:

       increment1--rx1
      /
    tx
      \
       increment2--rx2

    The top branch is forced via a PeriodicCondition to run at a slower rate than
    the source. It is currently configured to discard any extra messages that arrive
    at increment1 before it is ready to execute again, but different behavior could be
    achieved via other settings to policy and/or queue sizes.

    The `policy_set_mode` parameter is used to configure various different possible
    ways of configuring the receive Queue policy for the input port of IncrementOp.
    It can be configured via:
        - 'input': configure in IncrementOp.setup via OperatorSpec.input
        - 'connector': configure in IncrementOp.setup via IOSpec.connector
        - 'both': combination of 'input' and 'connector'
        - 'operator-method' configure policy via Operator.queue_policy
        - None : leave queue policy as it's default

    """

    def __init__(self, *args, policy_set_mode="input", count=100, **kwargs):
        if policy_set_mode is not None and policy_set_mode not in [
            "input",
            "connector",
            "both",
            "operator-method",
        ]:
            raise ValueError(f"unrecognized {policy_set_mode=}")
        self.count = count
        self.policy_set_mode = policy_set_mode
        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent and a
        # PeriodicCondition to control how often messages are sent.
        source_rate_hz = 60  # messages sent per second
        period_source_ns = int(1e9 / source_rate_hz)  # period in nanoseconds

        tx = PingTxOp(
            self,
            CountCondition(self, self.count),
            PeriodicCondition(self, recess_period=period_source_ns),
            initial_value=0,
            increment=1,
            name="tx",
        )

        increment_op_queue_set_mode = (
            None if self.policy_set_mode == "operator-method" else self.policy_set_mode
        )

        # first branch will have a periodic condition so it can't run faster than 5 Hz
        branch1_hz = 5
        period_ns1 = int(1e9 / branch1_hz)
        increment1 = IncrementOp(
            self,
            PeriodicCondition(self, recess_period=period_ns1),
            name="increment1",
            policy_set_mode=increment_op_queue_set_mode,
        )
        if self.policy_set_mode == "operator-method":
            # set queue policy here instead of within PingTxOp.setup
            increment1.queue_policy(
                "in", port_type=IOSpec.IOType.INPUT, policy=IOSpec.QueuePolicy.REJECT
            )

        rx1 = PingRxOp(self, name="rx1")
        self.add_flow(tx, increment1)
        self.add_flow(increment1, rx1)

        # second branch does not have a periodic condition so will tick on every message sent by tx
        increment2 = IncrementOp(
            self,
            name="increment2",
            policy_set_mode=increment_op_queue_set_mode,
        )
        if self.policy_set_mode == "operator-method":
            # set queue policy here instead of within PingTxOp.setup
            increment2.queue_policy(
                "in", port_type=IOSpec.IOType.INPUT, policy=IOSpec.QueuePolicy.REJECT
            )

        rx2 = PingRxOp(self, name="rx2")
        self.add_flow(tx, increment2)
        self.add_flow(increment2, rx2)


@pytest.mark.parametrize("policy_set_mode", ["input", "connector", "operator-method", "both", None])
@pytest.mark.parametrize("threads", [0, 8])
def test_queue_policy_settings(policy_set_mode, threads, capfd):
    count = 30
    app = MultiRateApp(count=count, policy_set_mode=policy_set_mode)
    if threads == 0:
        scheduler = GreedyScheduler(app, name="greedy_scheduler")
    else:
        scheduler = EventBasedScheduler(
            app,
            worker_thread_number=threads,
            stop_on_deadlock=True,
            stop_on_deadlock_timeout=400,
            name="ebs",
        )
    app.scheduler(scheduler)
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()

    # tx is at rate of 60 hz, but IncrementOp1 has periodic condition at 5Hz,
    # so rx1 ticks ~12x less than rx2 (which does not have periodic condition)
    # (Add + 1 because first message is always received by both)
    max_rx1 = math.ceil(count / 12.0) + 1
    min_rx1 = math.floor(count / 12.0) + 1

    # determine number of messages expected to be received by rx1
    if policy_set_mode is None:
        min_warn = 12 * (count // 12)
        # many push failed warnings will have been logged
        assert captured.err.count("warning") >= min_warn
        assert captured.err.count("Push failed on 'in'") >= min_warn
    elif policy_set_mode == "both":
        # both copies of IncrementOp will have printed a warning about ignored policy
        if threads == 0:
            assert captured.err.count("warning") == 2
        else:
            # event-based scheduler may print an additional warning about deadlock if
            # stop_on_deadlock_timeout value is too small
            assert captured.err.count("warning") >= 2
        assert "The queue policy set for input port 'in' of operator 'increment1'" in captured.err
        assert "The queue policy set for input port 'in' of operator 'increment2'" in captured.err
        msg_common = "will be ignored because a connector (receiver) was explicitly set"
        assert captured.err.count(msg_common) == 2
    elif threads == 0:
        assert "warning" not in captured.err
    else:
        # event-based scheduler may print warning about deadlock if
        # stop_on_deadlock_timeout value is too small
        assert captured.err.count("warning") <= 1
    assert "exception occurred" not in captured.err.lower()

    # confirm expected number of messages were received on each branch
    num_received_rx1 = captured.out.count("receiver 'rx1' received value")
    num_received_rx2 = captured.out.count("receiver 'rx2' received value")
    assert num_received_rx2 == count
    assert num_received_rx1 >= min_rx1
    assert num_received_rx1 <= max_rx1


class PingTxPolicyOp(Operator):
    """Simple transmitter operator with configurable output queue policy

    This variant of PingTxOp, emits twice despite having a queue size of only 1.
    This is done to test the policy of the output queue.
    """

    def __init__(self, fragment, *args, initial_value=0, increment=0, policy=None, **kwargs):
        self.count = 0
        self.initial_value = initial_value
        self.increment = increment

        self.policy = policy

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out", policy=self.policy).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        value = self.initial_value + self.count * self.increment
        op_output.emit(value, "out")
        self.count += 1
        op_output.emit(value + 1, "out")
        self.count += 1


class OutputQueuePolicyTestApp(Application):
    """This application tests the output queue policy behavior for PingTxOp.

    Uses a transmit operator that emits twice with output queue size of 1.
    """

    def __init__(self, *args, tx_queue_policy=None, count=100, **kwargs):
        self.count = count
        self.tx_queue_policy = tx_queue_policy
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTxPolicyOp(
            self,
            CountCondition(self, self.count),
            initial_value=0,
            increment=1,
            policy=self.tx_queue_policy,
            name="tx",
        )

        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, rx)


@pytest.mark.parametrize(
    "tx_queue_policy",
    [IOSpec.QueuePolicy.REJECT, IOSpec.QueuePolicy.POP, IOSpec.QueuePolicy.FAULT, None],
)
def test_output_queue_policy(tx_queue_policy, capfd):
    count = 10
    app = OutputQueuePolicyTestApp(count=count, tx_queue_policy=tx_queue_policy)
    if tx_queue_policy == IOSpec.QueuePolicy.FAULT or tx_queue_policy is None:
        with pytest.raises(RuntimeError, match="Failed to publish output message with error"):
            app.run()
    else:
        app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()

    num_warn = captured.err.count("warning")
    num_push_failed = captured.err.count("Push failed on 'out'")
    if tx_queue_policy is None or tx_queue_policy == IOSpec.QueuePolicy.FAULT:
        assert num_push_failed == 1
        assert "exception occurred" in captured.err.lower()
    else:
        assert num_warn == 0
        assert num_push_failed == 0
        assert "exception occurred" not in captured.err.lower()
        # one message received for each tx call (output queue size is 1)
        assert captured.out.count("receiver 'rx' received") == count

    if tx_queue_policy == IOSpec.QueuePolicy.POP:
        # second of the two values (odd integers) would always be emitted
        for v in range(2 * count):
            if v % 2 == 1:
                assert f"receiver 'rx' received value: {v}\n" in captured.out
            else:
                assert f"receiver 'rx' received value: {v}\n" not in captured.out
    elif tx_queue_policy == IOSpec.QueuePolicy.REJECT:
        # first of the two values (even integers) would always be emitted
        for v in range(2 * count):
            if v % 2 == 0:
                assert f"receiver 'rx' received value: {v}\n" in captured.out
            else:
                assert f"receiver 'rx' received value: {v}\n" not in captured.out
