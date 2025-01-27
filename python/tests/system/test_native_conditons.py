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

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, ComponentSpec, Condition, SchedulingStatusType
from holoscan.operators import PingRxOp, PingTxOp

# Define native Python Conditions used in the test applications below


class ValidNativeCondition(Condition):
    def __init__(self, fragment, *args, custom_kwarg=0, **kwargs):
        self.custom_kwarg = custom_kwarg
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: ComponentSpec):
        print("** native condition setup method called **")

    def initialize(self):
        print("** native condition initialize method called **")

    def update_state(self, timestamp):
        print("** native condition update_state method called **")

    def check(self, timestamp):
        print("** native condition check method called **")
        # initially ready when the operator hasn't been called previously
        return (SchedulingStatusType.READY, None)

    def on_execute(self, timestamp):
        print("** native condition on_execute method called **")


class ConditionWithInvalidUpdateState(Condition):
    # invalid update_state, must have a second argument (timestamp)
    def update_state(self):
        print("** native condition update_state method called **")


class ConditionWithInvalidCheckReturnTupleSize(Condition):
    def __init__(self, fragment, *args, custom_kwarg=0, **kwargs):
        self.custom_kwarg = custom_kwarg
        super().__init__(fragment, *args, **kwargs)

    def check(self, timestamp):
        # invalid return type (tuple length != 2)
        return (None, None, None)


class ConditionWithInvalidCheckReturnNonTuple(Condition):
    def check(self, timestamp):
        # invalid return:  must be a tuple with (state, timestamp)
        return SchedulingStatusType.READY


class ConditionWithInvalidCheckReturnTupleItems(Condition):
    def check(self, timestamp):
        # invalid return type (first item is not a SchedulingStatusType)
        return (None, None)


class ConditionWithInvalidCheckReturnTupleItems2(Condition):
    def check(self, timestamp):
        # invalid return type (first item is not a SchedulingStatusType)
        return (SchedulingStatusType.READY, "invalid_timestamp")


class MyNativeConditionPingApp(Application):
    def __init__(self, *args, condition_class=ValidNativeCondition, **kwargs):
        self.native_condition_class = condition_class
        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        # PeriodicCondition is used so that each subsequent message is
        # sent only after a period of 200 milliseconds has elapsed.
        tx = PingTxOp(
            self,
            CountCondition(self, 5),
            self.native_condition_class(self, name="native"),
            name="tx",
        )
        rx = PingRxOp(self, name="rx")

        # Connect the operators into the workflow:  tx -> rx
        self.add_flow(tx, rx)


def test_valid_native_condition(capfd):
    app = MyNativeConditionPingApp(condition_class=ValidNativeCondition)
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    assert "native condition setup method called" in captured.out
    assert "native condition initialize method called" in captured.out
    assert "native condition update_state method called" in captured.out
    assert "native condition check method called" in captured.out
    assert "native condition on_execute method called" in captured.out
    assert "Rx message value: 5" in captured.out


def test_native_condition_invalid_update_state_method(capfd):
    app = MyNativeConditionPingApp(condition_class=ConditionWithInvalidUpdateState)
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    assert "Rx message value: 5" not in captured.out


@pytest.mark.parametrize(
    "condition_class, err_msg",
    [
        (ConditionWithInvalidCheckReturnNonTuple, "check method must return a tuple"),
        (ConditionWithInvalidCheckReturnTupleSize, "check method must return a tuple of size 2"),
        (
            ConditionWithInvalidCheckReturnTupleItems,
            "The first element of the tuple returned by check must be a "
            "`holoscan.core.SchedulingStatusType` enum value",
        ),
        (
            ConditionWithInvalidCheckReturnTupleItems2,
            "The second element of the tuple returned by check must be a Python int or None",
        ),
    ],
)
def test_native_condition_check_invalid_return_type(capfd, condition_class, err_msg):
    app = MyNativeConditionPingApp(condition_class=condition_class)
    with pytest.raises(RuntimeError, match=err_msg):
        app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    assert "Rx message value: 5" not in captured.out
