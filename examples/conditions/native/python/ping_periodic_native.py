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
from holoscan.core import Application, ComponentSpec, Condition, SchedulingStatusType
from holoscan.operators import PingRxOp, PingTxOp

# Now define a simple application using the operators defined above


class NativePeriodicCondition(Condition):
    """Example native Python periodic condition

    This behaves like holoscan.conditions.PeriodicCondition (which wraps an
    underlying C++ class). It is simplified in that it does not support a
    separate `policy` kwarg.

    Parameters
    ----------
    fragment: holoscan.core.Fragment
        The fragment (or Application) to which this condition will belong.
    recess_period : int, optional
        The time to wait before an operator can execute again (units are in
        nanoseconds).
    """

    def __init__(self, fragment, *args, recess_period=0, **kwargs):
        self.recess_period_ns = recess_period
        self.next_target = -1
        super().__init__(fragment, *args, **kwargs)

    # Could add a `recess_period` Parameter via `setup` like in the following
    #
    #   def setup(self, ComponentSpec: spec):
    #       spec.param("recess_period", 0)
    #
    # and then configure that parameter in `initialize`, but for Python it is
    # easier to just add parameters to `__init__` as shown above.

    def setup(self, spec: ComponentSpec):
        print("** native condition setup method called **")

    def initialize(self):
        print("** native condition initialize method called **")

    def update_state(self, timestamp):
        print("** native condition update_state method called **")

    def check(self, timestamp):
        print("** native condition check method called **")
        # initially ready when the operator hasn't been called previously
        if self.next_target < 0:
            return (SchedulingStatusType.READY, timestamp)

        # return WAIT_TIME and the timestamp if the specified `recess_period` hasn't been reached
        status_type = (
            SchedulingStatusType.READY
            if (timestamp > self.next_target)
            else SchedulingStatusType.WAIT_TIME
        )
        return (status_type, self.next_target)

    def on_execute(self, timestamp):
        print("** native condition on_execute method called **")
        if self.next_target > 0:
            self.next_target = self.next_target + self.recess_period_ns
        else:
            self.next_target = timestamp + self.recess_period_ns


class MyPingApp(Application):
    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        # PeriodicCondition is used so that each subsequent message is
        # sent only after a period of 200 milliseconds has elapsed.
        tx = PingTxOp(
            self,
            CountCondition(self, 10),
            NativePeriodicCondition(self, recess_period=200_000_000),
            name="tx",
        )
        rx = PingRxOp(self, name="rx")

        # Connect the operators into the workflow:  tx -> rx
        self.add_flow(tx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
