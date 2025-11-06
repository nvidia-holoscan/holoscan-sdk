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
from holoscan.core import (
    Application,
    ComponentSpec,
    Condition,
    SchedulingStatusType,
)
from holoscan.operators import PingRxOp, PingTxOp

# Now define a simple application using the operators defined above


class NativeMessageAvailableCondition(Condition):
    """Example native Python periodic condition

    This behaves like holoscan.conditions.MessageAvailableCondition (which
    wraps an underlying C++ class). It is simplified in that it does not
    support a separate `policy` kwarg.

    Parameters
    ----------
    fragment : holoscan.core.Fragment
        The fragment (or Application) to which this condition will belong.
    receiver : str
        The name of the input port on the operator whose Receiver queue this condition will apply
        to.
    min_size : int, optional
        The number of messages that must be present on the specified input port before the
        operator is allowed to execute.
    """

    def __init__(self, fragment, receiver_name: str, *args, min_size: int = 1, **kwargs):
        self.receiver_name = receiver_name

        if not isinstance(min_size, int) or min_size <= 0:
            raise ValueError("min_size must be a positive integer")
        self.min_size = min_size

        # The current state of the scheduling term
        self.current_state = SchedulingStatusType.WAIT
        # timestamp when the state changed the last time
        self.last_state_change_ = 0
        # Must pass 'receiver_name' or 'transmitter_name' as a kwarg to the parent constructor
        # to avoid creation of a default condition for the operator port with that name.
        super().__init__(fragment, *args, receiver_name=receiver_name, **kwargs)

    def setup(self, spec: ComponentSpec):
        print("** native condition setup method called **")

    def initialize(self):
        print("** native condition initialize method called **")
        self.receiver_obj = self.receiver(self.receiver_name)
        if self.receiver_obj is None:
            raise RuntimeError(f"Receiver for port '{self.receiver_name}' not found")

    def check_min_size(self):
        return self.receiver_obj.back_size + self.receiver_obj.size >= self.min_size

    def update_state(self, timestamp):
        print("** native condition update_state method called **")
        is_ready = self.check_min_size()
        if is_ready and self.current_state != SchedulingStatusType.READY:
            self.current_state = SchedulingStatusType.READY
            self.last_state_change = timestamp
        if not is_ready and self.current_state != SchedulingStatusType.WAIT:
            self.current_state = SchedulingStatusType.WAIT
            self.last_state_change = timestamp

    def check(self, timestamp):
        print("** native condition check method called **")
        return self.current_state, self.last_state_change

    def on_execute(self, timestamp):
        print("** native condition on_execute method called **")
        self.update_state(timestamp)


class MyPingApp(Application):
    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        # PeriodicCondition is used so that each subsequent message is
        # sent only after a period of 200 milliseconds has elapsed.
        tx = PingTxOp(
            self,
            CountCondition(self, 10),
            name="tx",
        )

        # receiver must be the name of the input port of the PingRxOp operator
        message_cond = NativeMessageAvailableCondition(
            self,
            name="in_native_message_available",
            receiver_name="in",
            min_size=1,
        )
        rx = PingRxOp(self, message_cond, name="rx")

        # Connect the operators into the workflow:  tx -> rx
        self.add_flow(tx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
